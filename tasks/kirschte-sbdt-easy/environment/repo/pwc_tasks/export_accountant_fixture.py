from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CPP_ROOT = REPO_ROOT / "code" / "cpp_gbdt"
PROBE_SRC = Path(__file__).resolve().with_name("privacy_probe.cpp")
PROBE_BIN = Path("/tmp/kirschte_privacy_probe_easy")
OUTPUT_PATH = Path("/app/output/accountant_fixture.json")
COMPILED = False

PUBLIC_CASE = {
    "case_name": "abalone_ablation_fixture",
    "newton_boosting": False,
    "subsampling_ratio": 0.1,
    "leaf_noise_weight": 0.2,
    "active_threshold": 0.1,
    "hess_active_threshold": 1.0,
    "leaf_eps": 0.0945,
    "compositions": 150,
    "rho_alpha": 31,
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


def run_accountant_case(case: dict[str, object]) -> dict[str, object]:
    compile_probe()
    output = subprocess.check_output(
        [
            str(PROBE_BIN),
            "accounting",
            "1" if case["newton_boosting"] else "0",
            str(case["leaf_noise_weight"]),
            str(case["subsampling_ratio"]),
            str(case["active_threshold"]),
            str(case["hess_active_threshold"]),
            str(case["leaf_eps"]),
            str(case["compositions"]),
            str(case["rho_alpha"]),
        ],
        text=True,
    ).strip()
    alpha, noise_scale, max_rho, eps, leaf_rho = output.split()
    return {
        "case_name": str(case["case_name"]),
        "newton_boosting": bool(case["newton_boosting"]),
        "subsampling_ratio": round_float(case["subsampling_ratio"]),
        "leaf_noise_weight": round_float(case["leaf_noise_weight"]),
        "compositions": int(case["compositions"]),
        "alpha": int(float(alpha)),
        "noise_scale": round_float(noise_scale),
        "max_rho": round_float(max_rho),
        "eps": round_float(eps),
        "leaf_rho_alpha_31": round_float(leaf_rho),
    }


def build_report() -> dict[str, object]:
    return run_accountant_case(PUBLIC_CASE)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(build_report(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
