#!/usr/bin/env python3

import json
from pathlib import Path

from src.lib import LexMinimizerCountingFunction


K_VALUES = list(range(6, 17))
MINIMIZERS = ["ACACAA", "ACACAC"]
OUTPUT = Path("/app/output/example18_counts.json")


def build_report():
    counts = {}
    for minimizer in MINIMIZERS:
        obj = LexMinimizerCountingFunction(minimizer)
        counts[minimizer] = [obj.kmer(k) for k in K_VALUES]
    return {"k_values": K_VALUES, "counts": counts}


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build_report(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
