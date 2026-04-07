#!/usr/bin/env python3

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "/app")

from config import GLIMPSE_DIM, STATE_DIM
import modules
from data_loader import build_medium_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    with open(args.results, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert 0.0 <= payload["accuracy"] <= 1.0
    assert 0.0 <= payload["error_rate"] <= 1.0
    assert 0.0 <= payload["position_accuracy"] <= 1.0
    assert payload["mean_location_distance"] >= 0.0

    samples = build_medium_split([{"digit": 8, "pos_idx": 0, "seed": 12345}])
    image = samples[0]["image"]
    left_loc = samples[0]["location"]
    right_loc = build_medium_split([{"digit": 8, "pos_idx": 3, "seed": 12345}])[0]["location"]

    phi = np.asarray(modules.retina_glimpse(image, left_loc, scales=1), dtype=float)
    left = np.asarray(modules.glimpse_network(phi, left_loc, scales=1), dtype=float)
    right = np.asarray(modules.glimpse_network(phi, right_loc, scales=1), dtype=float)
    assert left.shape == (GLIMPSE_DIM,)
    assert np.linalg.norm(left) > 1e-6, "glimpse_network is still all zeros"
    assert np.linalg.norm(left - right) > 1e-6, "glimpse_network still ignores the location branch"

    zero_state = np.zeros(STATE_DIM, dtype=float)
    ones_state = np.ones(STATE_DIM, dtype=float)
    next_zero = np.asarray(modules.core_step(zero_state, left), dtype=float)
    next_one = np.asarray(modules.core_step(ones_state, left), dtype=float)
    assert next_zero.shape == (STATE_DIM,)
    assert np.linalg.norm(next_zero) > 1e-6, "core_step still returns all zeros"
    assert np.linalg.norm(next_zero - next_one) > 1e-6, "core_step still ignores the previous hidden state"

    weights = np.zeros((STATE_DIM, 2), dtype=float)
    weights[0, 0] = 0.5
    weights[1, 1] = -0.25
    bias = np.asarray([0.1, -0.2], dtype=float)
    loc_a = np.asarray(modules.location_network(next_zero, weights, bias), dtype=float)
    loc_b = np.asarray(modules.location_network(next_one, weights, bias), dtype=float)
    assert loc_a.shape == (2,)
    assert np.all(np.abs(loc_a) <= 1.0 + 1e-9)
    assert np.linalg.norm(loc_a - loc_b) > 1e-6, "location_network still ignores the learned head inputs"


if __name__ == "__main__":
    main()
