#!/usr/bin/env python3

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "/app")

import sdgnn_components as components
from edge_splits import TRAIN_DATA
from sdgnn_core import (
    build_relations,
    compute_base_node_features,
    default_relation_weights,
    mean_relation_layer,
    num_nodes_for_edges,
    parse_edge_text,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    with open(args.results, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert 0.0 <= payload["macro_f1"] <= 1.0
    assert payload["embedding_dim"] == 10

    train_edges = parse_edge_text(TRAIN_DATA)
    num_nodes = num_nodes_for_edges(train_edges)
    relations, _ = build_relations(num_nodes, train_edges)
    base_features = compute_base_node_features(num_nodes, relations)
    one_layer = mean_relation_layer(base_features, relations, default_relation_weights()[0])
    actual = np.asarray(components.apply_relation_stack(base_features, relations), dtype=float)

    assert actual.shape == one_layer.shape
    assert not np.allclose(actual, one_layer), "encoder still matches the one-layer baseline"


if __name__ == "__main__":
    main()
