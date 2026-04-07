#!/usr/bin/env python3

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "/app")

import sdgnn_components as components
from edge_splits import TRAIN_DATA
from sdgnn_core import build_relations, num_nodes_for_edges, parse_edge_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    with open(args.results, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert 0.0 <= payload["macro_f1"] <= 1.0
    assert payload["direction_feature_dim"] >= 3

    train_edges = parse_edge_text(TRAIN_DATA)
    num_nodes = num_nodes_for_edges(train_edges)
    relations, _ = build_relations(num_nodes, train_edges)

    samples = [
        np.asarray(components.build_direction_features(source, target, relations), dtype=float)
        for source, target, _ in train_edges[:12]
    ]
    non_zero = any(np.linalg.norm(sample) > 1e-9 for sample in samples)
    assert non_zero, "direction features are still all zeros"


if __name__ == "__main__":
    main()

