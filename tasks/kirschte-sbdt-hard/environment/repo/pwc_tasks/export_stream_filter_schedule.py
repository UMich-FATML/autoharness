from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CPP_ROOT = REPO_ROOT / "code" / "cpp_gbdt"
PROBE_SRC = Path(__file__).resolve().with_name("privacy_probe.cpp")
PROBE_BIN = Path("/tmp/kirschte_privacy_probe_hard")
OUTPUT_PATH = Path("/app/output/stream_filter_schedule.json")
COMPILED = False

PUBLIC_CASE = {
    "newton_boosting": True,
    "subsampling_ratio": 0.05,
    "leaf_noise_weight": 0.2,
    "active_threshold": 0.5,
    "hess_active_threshold": 0.25,
    "rdp_alpha": 9,
    "noise_scale": 1.8,
    "max_rho": 0.006,
    "initial_ids": [0, 1, 2],
    "novel_ids": [3, 4],
    "regular_traces": [
        {0: [0.18, 0.06], 1: [0.45, 0.22], 2: [0.09, 0.04]},
        {0: [0.2, 0.08], 1: [0.4, 0.2], 2: [0.15, 0.05]},
        {0: [0.25, 0.1], 1: [0.38, 0.2], 2: [0.17, 0.07]},
    ],
    "extra_traces": [
        {0: [0.22, 0.09], 1: [0.41, 0.21], 2: [0.18, 0.08], 3: [0.12, 0.05], 4: [0.3, 0.12]},
        {0: [0.24, 0.1], 1: [0.42, 0.22], 2: [0.2, 0.08], 3: [0.14, 0.06], 4: [0.32, 0.14]},
    ],
}


def compile_probe() -> None:
    global COMPILED
    if COMPILED and PROBE_BIN.exists():
        return
    cmd = [
        "g++",
        "-std=gnu++11",
        f"-I{CPP_ROOT / 'include'}",
        f"-I{CPP_ROOT / 'include' / 'gbdt'}",
        str(PROBE_SRC),
        str(CPP_ROOT / "src" / "rdp_accountant.cpp"),
        str(CPP_ROOT / "src" / "gbdt" / "utils.cpp"),
        "-o",
        str(PROBE_BIN),
    ]
    subprocess.run(cmd, check=True)
    COMPILED = True


def round_float(value: str | float) -> float:
    return round(float(value), 6)


def run_rho_query(case: dict[str, object], gradient: float, hessian: float) -> float:
    compile_probe()
    output = subprocess.check_output(
        [
            str(PROBE_BIN),
            "rho",
            "1" if case["newton_boosting"] else "0",
            str(case["leaf_noise_weight"]),
            str(case["subsampling_ratio"]),
            str(case["active_threshold"]),
            str(case["hess_active_threshold"]),
            str(case["rdp_alpha"]),
            str(case["noise_scale"]),
            str(gradient),
            str(hessian),
        ],
        text=True,
    ).strip()
    return float(output)


def simulate_case(case: dict[str, object]) -> dict[str, object]:
    all_ids = case["initial_ids"] + case["novel_ids"]
    accounted = {point_id: 0.0 for point_id in all_ids}
    active_ids_by_extra_round: list[list[int]] = []

    for traces, is_extra in (
        (case["regular_traces"], False),
        (case["extra_traces"], True),
    ):
        for round_trace in traces:
            active_ids: list[int] = []
            available_ids = case["novel_ids"] if is_extra else case["initial_ids"]
            for point_id in available_ids:
                if point_id not in round_trace:
                    continue

                gradient, hessian = round_trace[point_id]
                rho_value = run_rho_query(case, gradient, hessian)
                if accounted[point_id] + rho_value <= case["max_rho"] + 1e-7:
                    accounted[point_id] += rho_value
                    if is_extra:
                        active_ids.append(point_id)
                else:
                    if not is_extra:
                        accounted[point_id] = accounted[point_id]

            if is_extra:
                active_ids_by_extra_round.append(active_ids)

    final_accounted_rho = {
        str(point_id): round_float(value) for point_id, value in sorted(accounted.items())
    }
    last_active_ids = active_ids_by_extra_round[-1] if active_ids_by_extra_round else []
    retired_ids = sorted(point_id for point_id in all_ids if point_id not in last_active_ids)

    return {
        "rdp_alpha": int(case["rdp_alpha"]),
        "noise_scale": round_float(case["noise_scale"]),
        "max_rho": round_float(case["max_rho"]),
        "regular_rounds": len(case["regular_traces"]),
        "extra_rounds": len(case["extra_traces"]),
        "initial_ids": list(case["initial_ids"]),
        "novel_ids": list(case["novel_ids"]),
        "final_accounted_rho": final_accounted_rho,
        "active_ids_by_extra_round": active_ids_by_extra_round,
        "retired_ids": retired_ids,
    }


def build_report() -> dict[str, object]:
    return simulate_case(PUBLIC_CASE)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(build_report(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
