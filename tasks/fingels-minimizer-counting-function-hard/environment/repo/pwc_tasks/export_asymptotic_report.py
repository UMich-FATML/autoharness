#!/usr/bin/env python3

import json
import math
from pathlib import Path

from src.lib import LexMinimizerCountingFunction
from src.utils import regress_and_predict


PANEL = ["AAAAAA", "ACACAA", "ACACAC", "CAAAAA", "GAAAAA", "TAAAAA"]
K_VALUES = list(range(6, 17))
PREDICTION_K = 20
OUTPUT = Path("/app/output/asymptotic_report.json")


def round_float(value):
    return round(float(value), 12)


def fit_line(xs, ys):
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    ss_x = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / ss_x
    intercept = mean_y - slope * mean_x
    predictions = [intercept + slope * x for x in xs]
    ss_res = sum((y - pred) ** 2 for y, pred in zip(ys, predictions))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot
    return slope, intercept, r2


def build_report():
    report = {"k_values": K_VALUES, "prediction_k": PREDICTION_K, "panel": []}
    for minimizer in PANEL:
        obj = LexMinimizerCountingFunction(minimizer)
        counts = [obj.kmer(k) for k in K_VALUES]
        log4_values = [math.log(value, 4) for value in counts]

        slope, intercept, r2 = fit_line(K_VALUES, counts)
        predicted = regress_and_predict(K_VALUES, counts, PREDICTION_K)

        report["panel"].append(
            {
                "minimizer": minimizer,
                "log4_values": [round_float(value) for value in log4_values],
                "slope": round_float(slope),
                "intercept": round_float(intercept),
                "r2": round_float(r2),
                "predicted_log4_pi_k20": round_float(predicted),
            }
        )
    return report


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build_report(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
