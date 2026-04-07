import argparse
import importlib.util
import os

import numpy as np

import sdgnn_components as components
from edge_splits import DEV_DATA, TRAIN_DATA
from sdgnn_core import (
    build_base_edge_features,
    build_relations,
    compute_base_node_features,
    fit_logistic_regression,
    macro_f1_from_probabilities,
    num_nodes_for_edges,
    parse_edge_text,
    sigmoid,
    write_results,
)


def load_hidden_edges(module_path: str):
    if not module_path:
        raise ValueError("--hidden-module is required for hidden_test runs")
    spec = importlib.util.spec_from_file_location("hidden_test_edges", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load hidden edges from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return parse_edge_text(module.TEST_DATA)


def as_feature_vector(value) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.asarray([float(array)], dtype=float)
    if array.ndim != 1:
        raise ValueError(f"Expected a one-dimensional feature vector, got shape {array.shape}")
    return array


def build_feature_matrix(train_edges, eval_edges, embeddings, relations, edge_lookup):
    def edge_vector(edge):
        source, target, label = edge
        base = build_base_edge_features(embeddings, source, target)
        direction = as_feature_vector(components.build_direction_features(source, target, relations))
        triangle = as_feature_vector(components.build_triangle_features(source, target, relations, edge_lookup))
        return np.concatenate([base, direction, triangle]), 1 if label > 0 else 0

    train_rows = [edge_vector(edge) for edge in train_edges]
    eval_rows = [edge_vector(edge) for edge in eval_edges]

    train_features = np.stack([row[0] for row in train_rows])
    train_labels = np.asarray([row[1] for row in train_rows], dtype=float)
    eval_features = np.stack([row[0] for row in eval_rows])
    eval_labels = np.asarray([row[1] for row in eval_rows], dtype=int)
    return train_features, train_labels, eval_features, eval_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "hidden_test"], default="dev")
    parser.add_argument("--hidden-module", default=None)
    parser.add_argument("--output", default="/app/results.json")
    args = parser.parse_args()

    train_edges = parse_edge_text(TRAIN_DATA)
    dev_edges = parse_edge_text(DEV_DATA)
    eval_edges = dev_edges if args.split == "dev" else load_hidden_edges(args.hidden_module)

    node_count = num_nodes_for_edges(train_edges, dev_edges, eval_edges)
    relations, edge_lookup = build_relations(node_count, train_edges)
    base_features = compute_base_node_features(node_count, relations)
    embeddings = np.asarray(components.apply_relation_stack(base_features, relations), dtype=float)

    train_features, train_labels, eval_features, eval_labels = build_feature_matrix(
        train_edges, eval_edges, embeddings, relations, edge_lookup
    )

    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0) + 1e-6
    train_features = (train_features - feature_mean) / feature_std
    eval_features = (eval_features - feature_mean) / feature_std

    weights, bias = fit_logistic_regression(train_features, train_labels)
    probabilities = sigmoid(eval_features @ weights + bias)
    macro_f1, accuracy = macro_f1_from_probabilities(eval_labels, probabilities)

    direction_dim = len(as_feature_vector(components.build_direction_features(*train_edges[0][:2], relations)))
    triangle_dim = len(as_feature_vector(components.build_triangle_features(*train_edges[0][:2], relations, edge_lookup)))
    payload = {
        "accuracy": accuracy,
        "embedding_dim": int(embeddings.shape[1]),
        "eval_edges": int(len(eval_edges)),
        "feature_dim": int(train_features.shape[1]),
        "macro_f1": macro_f1,
        "split": args.split,
        "train_edges": int(len(train_edges)),
        "triangle_feature_dim": int(triangle_dim),
        "direction_feature_dim": int(direction_dim),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    write_results(args.output, payload)


if __name__ == "__main__":
    main()

