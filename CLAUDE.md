# AutoAgent

## Running Tasks

Tasks run in Daytona cloud sandboxes (there is no local Docker daemon).

Run a single task:
```bash
pixi run harbor run -p tasks/<task-name> -a oracle -e daytona --debug
```

Run all tasks in parallel:
```bash
rm -rf jobs; mkdir -p jobs && pixi run harbor run -p tasks/ -n 100 --agent-import-path agent:AutoAgent -e daytona -o jobs --job-name latest > run.log 2>&1
```

**Daytona sandbox limits**: max 4 vCPUs, 8GB RAM, and 10GB storage per sandbox; max 10 vCPUs, 16GB RAM, and 30GB total storage across all sandboxes. 

## Debugging Failures

Job artifacts are saved to `jobs/<timestamp>/<task-name>__<id>/`:

- `agent/oracle.txt` — stdout from solve.sh
- `verifier/test-stdout.txt` — stdout from test.sh
- `verifier/reward.json` — detailed score breakdown
