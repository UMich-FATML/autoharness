import json
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

Edge = Tuple[int, int, int]
RelationMap = Dict[Tuple[str, str], List[List[int]]]


def parse_edge_text(text: str) -> List[Edge]:
    edges: List[Edge] = []
    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        source, target, label = line.split(",")
        edges.append((int(source), int(target), int(label)))
    return edges


def num_nodes_for_edges(*edge_groups: Sequence[Edge]) -> int:
    node_max = 0
    for edges in edge_groups:
        for source, target, _ in edges:
            node_max = max(node_max, source, target)
    return node_max + 1


def build_relations(num_nodes: int, train_edges: Sequence[Edge]) -> Tuple[RelationMap, Dict[Tuple[int, int], int]]:
    relations: RelationMap = {
        ("pos", "out"): [[] for _ in range(num_nodes)],
        ("pos", "in"): [[] for _ in range(num_nodes)],
        ("neg", "out"): [[] for _ in range(num_nodes)],
        ("neg", "in"): [[] for _ in range(num_nodes)],
    }
    edge_lookup: Dict[Tuple[int, int], int] = {}
    for source, target, label in train_edges:
        sign_key = "pos" if label > 0 else "neg"
        relations[(sign_key, "out")][source].append(target)
        relations[(sign_key, "in")][target].append(source)
        edge_lookup[(source, target)] = label
    return relations, edge_lookup


def compute_base_node_features(num_nodes: int, relations: RelationMap) -> np.ndarray:
    rows = []
    for node in range(num_nodes):
        rows.append(
            [
                len(relations[("pos", "out")][node]),
                len(relations[("pos", "in")][node]),
                len(relations[("neg", "out")][node]),
                len(relations[("neg", "in")][node]),
                np.sin(node * 0.17),
                np.cos(node * 0.11),
            ]
        )
    features = np.asarray(rows, dtype=float)
    return normalize_columns(features)


def normalize_columns(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0) + 1e-6
    return (matrix - mean) / std


def mean_relation_layer(embeddings: np.ndarray, relations: RelationMap, weight: np.ndarray) -> np.ndarray:
    blocks = [embeddings]
    for relation_key in [("pos", "out"), ("pos", "in"), ("neg", "out"), ("neg", "in")]:
        relation_block = np.zeros_like(embeddings)
        for node, neighbors in enumerate(relations[relation_key]):
            if neighbors:
                relation_block[node] = embeddings[neighbors].mean(axis=0)
        blocks.append(relation_block)
    stacked = np.concatenate(blocks, axis=1)
    return np.tanh(stacked @ weight)


def default_relation_weights() -> List[np.ndarray]:
    rng = np.random.default_rng(7)
    return [
        rng.normal(0.0, 0.24, (6 * 5, 10)),
        rng.normal(0.0, 0.24, (10 * 5, 10)),
    ]


def build_base_edge_features(embeddings: np.ndarray, source: int, target: int) -> np.ndarray:
    source_vec = embeddings[source]
    target_vec = embeddings[target]
    return np.concatenate(
        [
            source_vec,
            target_vec,
            source_vec * target_vec,
            np.abs(source_vec - target_vec),
            np.array([np.dot(source_vec, target_vec)], dtype=float),
        ]
    )


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def fit_logistic_regression(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    steps: int = 5000,
    learning_rate: float = 0.08,
    regularization: float = 5e-4,
) -> Tuple[np.ndarray, float]:
    weights = np.zeros(train_features.shape[1], dtype=float)
    bias = 0.0
    for _ in range(steps):
        logits = train_features @ weights + bias
        probabilities = sigmoid(logits)
        errors = probabilities - train_labels
        weights -= learning_rate * ((train_features.T @ errors) / len(train_features) + regularization * weights)
        bias -= learning_rate * errors.mean()
    return weights, bias


def macro_f1_from_probabilities(labels: np.ndarray, probabilities: np.ndarray) -> Tuple[float, float]:
    predictions = (probabilities >= 0.5).astype(int)
    accuracy = float((predictions == labels).mean())
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())

    def f1(one_tp: int, one_fp: int, one_fn: int) -> float:
        precision = one_tp / (one_tp + one_fp + 1e-9)
        recall = one_tp / (one_tp + one_fn + 1e-9)
        return 2.0 * precision * recall / (precision + recall + 1e-9)

    macro_f1 = 0.5 * (f1(tp, fp, fn) + f1(tn, fn, fp))
    return float(macro_f1), accuracy


def write_results(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

