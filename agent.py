"""Single-file Harbor agent harness: --agent-import-path agent:AutoAgent."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from agents import Agent, Runner, function_tool
from agents.items import (
    ItemHelpers,
    MessageOutputItem,
    ReasoningItem,
    ToolCallItem,
    ToolCallOutputItem,
)
from agents.tool import FunctionTool
from agents.usage import Usage
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


# ============================================================================
# EDITABLE HARNESS — prompt, tools, agent construction
# ============================================================================

SYSTEM_PROMPT = """\
You are an expert software engineer solving a coding task inside a sandboxed Linux environment.

## Workflow
1. **Explore**: Read the task instruction. List /app to understand the repo structure. Read key source files and any paper/documentation.
2. **Understand**: Identify what needs to be implemented or fixed. Find placeholder/stub functions. Read the verifier test to understand how your output will be scored.
3. **Environment setup**: Ensure dependencies are installed. Run `pip install -e .` in repo dirs or set PYTHONPATH globally: `echo '/app/repo' > /usr/local/lib/python3.*/site-packages/app.pth` so imports work for ALL processes (including the verifier).
4. **Implement**: Make targeted code changes. Edit existing files rather than rewriting from scratch. Use `write_file` to write changes cleanly.
5. **Test**: Run the evaluation command from the instruction. Check the output file. If results are wrong, iterate.
6. **Verify**: Before finishing, confirm the output file exists and has the required format/keys. Run the same export/evaluation script one final time to make sure it works standalone.

## Key principles
- Read before writing. Understand the existing code before modifying it.
- Fix the actual source code in the repository, don't just write output files with hardcoded values.
- The verifier runs scripts independently — your changes must work without your shell session's env vars. Install packages and fix imports permanently.
- Check /tests/ if it exists to understand how your work will be verified.
- Budget your turns. Don't waste turns on exploration you don't need.
- For Python file edits, use the write_file tool for reliability.
"""
MODEL = "gpt-5"
MAX_TURNS = 60


def create_tools(environment: BaseEnvironment) -> list[FunctionTool]:
    """Create tools for the agent. Add new tools here."""

    @function_tool
    async def run_shell(command: str) -> str:
        """Run a shell command in the task environment. Returns stdout and stderr."""
        try:
            result = await environment.exec(command=command, timeout_sec=300)
            out = ""
            if result.stdout:
                out += result.stdout
            if result.stderr:
                out += f"\nSTDERR:\n{result.stderr}" if out else f"STDERR:\n{result.stderr}"
            return out or "(no output)"
        except Exception as exc:
            return f"ERROR: {exc}"

    @function_tool
    async def read_file(path: str, offset: int = 0, limit: int = 500) -> str:
        """Read lines from a file. Returns numbered lines starting from `offset` (0-based), up to `limit` lines."""
        try:
            result = await environment.exec(
                command=f"sed -n '{offset + 1},{offset + limit}p' '{path}' | cat -n",
                timeout_sec=30,
            )
            content = result.stdout or ""
            if not content.strip():
                return f"(empty or file not found: {path})"
            return content
        except Exception as exc:
            return f"ERROR: {exc}"

    @function_tool
    async def write_file(path: str, content: str) -> str:
        """Write content to a file, creating parent directories as needed. Overwrites if the file exists."""
        import base64
        encoded = base64.b64encode(content.encode()).decode()
        try:
            result = await environment.exec(
                command=f"mkdir -p \"$(dirname '{path}')\" && echo '{encoded}' | base64 -d > '{path}' && echo 'OK: wrote {path}'",
                timeout_sec=30,
            )
            out = (result.stdout or "") + (result.stderr or "")
            return out.strip() or f"OK: wrote {path}"
        except Exception as exc:
            return f"ERROR: {exc}"

    return [run_shell, read_file, write_file]


def create_agent(environment: BaseEnvironment) -> Agent:
    """Build the agent. Modify to add handoffs, sub-agents, or agent-as-tool."""
    tools = create_tools(environment)
    return Agent(
        name="autoagent",
        instructions=SYSTEM_PROMPT,
        tools=tools,
        model=MODEL,
    )


async def run_task(
    environment: BaseEnvironment,
    instruction: str,
) -> tuple[object, int]:
    """Run the agent on a task and return (result, duration_ms)."""
    agent = create_agent(environment)
    t0 = time.time()
    result = await Runner.run(agent, input=instruction, max_turns=MAX_TURNS)
    duration_ms = int((time.time() - t0) * 1000)
    return result, duration_ms


# ============================================================================
# FIXED ADAPTER BOUNDARY: do not modify unless the human explicitly asks.
# Harbor integration and trajectory serialization live here.
# ============================================================================

def to_atif(result: object, model: str, duration_ms: int = 0) -> dict:
    """Convert OpenAI Agents SDK RunResult to an ATIF trajectory dict."""
    steps: list[dict] = []
    step_id = 0
    now = datetime.now(timezone.utc).isoformat()

    def _step(source: str, message: str, **extra: object) -> dict:
        nonlocal step_id
        step_id += 1
        step = {
            "step_id": step_id,
            "timestamp": now,
            "source": source,
            "message": message,
        }
        step.update({key: value for key, value in extra.items() if value is not None})
        return step

    pending_tool_call = None
    for item in result.new_items:
        if isinstance(item, MessageOutputItem):
            text = ItemHelpers.text_message_output(item)
            if text:
                steps.append(_step("agent", text, model_name=model))
        elif isinstance(item, ReasoningItem):
            summaries = getattr(item.raw_item, "summary", None)
            reasoning = "\n".join(s.text for s in summaries if hasattr(s, "text")) if summaries else None
            if reasoning:
                steps.append(
                    _step(
                        "agent",
                        "(thinking)",
                        reasoning_content=reasoning,
                        model_name=model,
                    )
                )
        elif isinstance(item, ToolCallItem):
            raw = item.raw_item
            if hasattr(raw, "name"):
                pending_tool_call = raw
        elif isinstance(item, ToolCallOutputItem) and pending_tool_call:
            arguments = (
                json.loads(pending_tool_call.arguments)
                if isinstance(pending_tool_call.arguments, str)
                else pending_tool_call.arguments
            )
            output_str = str(item.output) if item.output else ""
            steps.append(
                _step(
                    "agent",
                    f"Tool: {pending_tool_call.name}",
                    tool_calls=[
                        {
                            "tool_call_id": pending_tool_call.call_id,
                            "function_name": pending_tool_call.name,
                            "arguments": arguments,
                        }
                    ],
                    observation={
                        "results": [
                            {
                                "source_call_id": pending_tool_call.call_id,
                                "content": output_str,
                            }
                        ]
                    },
                )
            )
            pending_tool_call = None

    if pending_tool_call:
        arguments = (
            json.loads(pending_tool_call.arguments)
            if isinstance(pending_tool_call.arguments, str)
            else pending_tool_call.arguments
        )
        steps.append(
            _step(
                "agent",
                f"Tool: {pending_tool_call.name}",
                tool_calls=[
                    {
                        "tool_call_id": pending_tool_call.call_id,
                        "function_name": pending_tool_call.name,
                        "arguments": arguments,
                    }
                ],
            )
        )

    if not steps:
        steps.append(_step("user", "(empty)"))

    usage = Usage()
    for response in result.raw_responses:
        usage.add(response.usage)

    return {
        "schema_version": "ATIF-v1.6",
        "session_id": getattr(result, "last_response_id", None) or "unknown",
        "agent": {"name": "autoagent", "version": "0.1.0", "model_name": model},
        "steps": steps,
        "final_metrics": {
            "total_prompt_tokens": usage.input_tokens,
            "total_completion_tokens": usage.output_tokens,
            "total_cached_tokens": getattr(usage.input_tokens_details, "cached_tokens", 0) or 0,
            "total_cost_usd": None,
            "total_steps": len(steps),
            "extra": {"duration_ms": duration_ms, "num_turns": len(result.raw_responses)},
        },
    }


class AutoAgent(BaseAgent):
    """Harbor agent adapter. Runs the OpenAI agent host-side and proxies shell into the container."""

    SUPPORTS_ATIF = True

    def __init__(self, *args, extra_env: dict[str, str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._extra_env = dict(extra_env) if extra_env else {}

    @staticmethod
    def name() -> str:
        return "autoagent"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        await environment.exec(command="mkdir -p /task")
        instr_file = self.logs_dir / "instruction.md"
        instr_file.write_text(instruction)
        await environment.upload_file(source_path=instr_file, target_path="/task/instruction.md")

        result, duration_ms = await run_task(environment, instruction)

        atif = to_atif(result, model=MODEL, duration_ms=duration_ms)
        traj_path = self.logs_dir / "trajectory.json"
        traj_path.write_text(json.dumps(atif, indent=2))

        try:
            final_metrics = atif.get("final_metrics", {})
            context.n_input_tokens = final_metrics.get("total_prompt_tokens", 0)
            context.n_output_tokens = final_metrics.get("total_completion_tokens", 0)
            context.n_cache_tokens = final_metrics.get("total_cached_tokens", 0)
        except Exception:
            pass

        usage = Usage()
        for response in result.raw_responses:
            usage.add(response.usage)
        print(
            f"turns={len(result.raw_responses)} duration_ms={duration_ms} "
            f"input={usage.input_tokens} output={usage.output_tokens}"
        )


__all__ = ["AutoAgent"]
