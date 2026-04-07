from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CPP_ROOT = REPO_ROOT / "code" / "cpp_gbdt"
PROBE_SRC = Path(__file__).resolve().with_name("privacy_probe.cpp")
PROBE_BIN = Path("/tmp/kirschte_privacy_probe_medium")
OUTPUT_PATH = Path("/app/output/renyi_approximation.json")
COMPILED = False

PUBLIC_CASE = {
    "newton_boosting": False,
    "subsampling_ratio": 0.1,
    "leaf_noise_weight": 0.2,
    "active_threshold": 0.1,
    "hess_active_threshold": 1.0,
    "rdp_alpha": 17,
    "noise_scale": 1.4,
    "queries": [
        [0.01234, 1.0],
        [0.04321, 1.0],
        [0.07777, 1.0],
        [0.09991, 1.0],
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


def run_approximation_case(case: dict[str, object]) -> dict[str, object]:
    compile_probe()
    args = [
        str(PROBE_BIN),
        "approx",
        "1" if case["newton_boosting"] else "0",
        str(case["leaf_noise_weight"]),
        str(case["subsampling_ratio"]),
        str(case["active_threshold"]),
        str(case["hess_active_threshold"]),
        str(case["rdp_alpha"]),
        str(case["noise_scale"]),
    ]
    for gradient, hessian in case["queries"]:
        args.extend([str(gradient), str(hessian)])

    output = subprocess.check_output(args, text=True).strip().splitlines()
    query_reports = []
    for line in output:
        gradient, hessian, exact, approx = line.split()
        query_reports.append(
            {
                "gradient_sensitivity": round_float(gradient),
                "hessian_sensitivity": round_float(hessian),
                "exact_rho": round_float(exact),
                "approx_rho": round_float(approx),
                "approx_ge_exact": round_float(approx) >= round_float(exact),
            }
        )

    return {
        "rdp_alpha": int(case["rdp_alpha"]),
        "noise_scale": round_float(case["noise_scale"]),
        "subsampling_ratio": round_float(case["subsampling_ratio"]),
        "leaf_noise_weight": round_float(case["leaf_noise_weight"]),
        "queries": query_reports,
    }


def build_report() -> dict[str, object]:
    return run_approximation_case(PUBLIC_CASE)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(build_report(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
