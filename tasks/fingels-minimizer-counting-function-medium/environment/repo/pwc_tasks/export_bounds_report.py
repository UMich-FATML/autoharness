#!/usr/bin/env python3

import json
import math
from pathlib import Path

from src.lib import LexMinimizerCountingFunction


PANEL = ["AAAAAA", "ACACAA", "ACACAC", "CAAAAA", "GAAAAA", "TAAAAA"]
K_VALUES = [10, 16]
OUTPUT = Path("/app/output/bounds_report.json")


def round_float(value):
    return round(float(value), 12)


def build_report():
    report = {"panel": PANEL, "k_values": K_VALUES, "results": {}, "summary": {}}
    for k in K_VALUES:
        entries = []
        tight_count = 0
        equal_count = 0
        for minimizer in PANEL:
            obj = LexMinimizerCountingFunction(minimizer)
            lower = obj.kmer_lower_bound(k)
            exact = obj.kmer(k)
            upper = obj.kmer_upper_bound(k)
            log_exact = math.log(exact, 4)
            if minimizer.startswith("T"):
                eta_plus = 0.0
                eta_minus = 0.0
            else:
                eta_plus = (math.log(upper, 4) - log_exact) / log_exact
                eta_minus = (log_exact - math.log(lower, 4)) / log_exact
            tight = lower == exact or upper == exact
            equal_bounds = lower == upper == exact
            tight_count += int(tight)
            equal_count += int(equal_bounds)
            entries.append(
                {
                    "minimizer": minimizer,
                    "lower": lower,
                    "exact": exact,
                    "upper": upper,
                    "eta_plus": round_float(eta_plus),
                    "eta_minus": round_float(eta_minus),
                    "tight": tight,
                    "equal_bounds": equal_bounds,
                }
            )
        report["results"][str(k)] = entries
        report["summary"][str(k)] = {
            "panel_size": len(PANEL),
            "tight_count": tight_count,
            "equal_count": equal_count,
        }
    return report


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build_report(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
