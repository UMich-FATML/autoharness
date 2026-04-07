#!/usr/bin/env python3

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "/app")

import modules
from data_loader import build_easy_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    with open(args.results, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert 0.0 <= payload["accuracy"] <= 1.0
    assert 0.0 <= payload["error_rate"] <= 1.0
    assert payload["feature_dim"] == 64

    samples = build_easy_split([{"digit": 8, "pos_idx": 0, "seed": 12345}])
    image = samples[0]["image"]
    left_loc = samples[0]["location"]
    right_loc = build_easy_split([{"digit": 8, "pos_idx": 8, "seed": 12345}])[0]["location"]

    left = np.asarray(modules.retina_glimpse(image, left_loc), dtype=float)
    right = np.asarray(modules.retina_glimpse(image, right_loc), dtype=float)
    assert left.shape == (64,)
    assert np.linalg.norm(left - right) > 1e-6, "retina_glimpse still ignores the fixation location"


if __name__ == "__main__":
    main()
