#!/bin/bash
set -euo pipefail

# CANARY: harbor-pwc-sdgnn-hard

python - <<'PY'
from pathlib import Path

target = Path("/app/sdgnn_components.py")
target.write_text(
    '''import numpy as np

from sdgnn_core import default_relation_weights, mean_relation_layer


def apply_relation_stack(base_features, relations):
    weights = default_relation_weights()
    first = mean_relation_layer(base_features, relations, weights[0])
    return mean_relation_layer(first, relations, weights[1])


def _status_proxy(node, relations):
    signed_out = len(relations[("pos", "out")][node]) - len(relations[("neg", "out")][node])
    signed_in = len(relations[("pos", "in")][node]) - len(relations[("neg", "in")][node])
    return signed_in - signed_out


def build_direction_features(source, target, relations):
    delta = _status_proxy(target, relations) - _status_proxy(source, relations)
    return np.asarray([delta, delta * delta, abs(delta)], dtype=float)


def _all_neighbors(node, relations):
    neighbors = set()
    for relation_key in [("pos", "out"), ("pos", "in"), ("neg", "out"), ("neg", "in")]:
        neighbors.update(relations[relation_key][node])
    return neighbors


def _triangle_counts(source, target, candidate_label, relations, edge_lookup):
    common = _all_neighbors(source, relations) & _all_neighbors(target, relations)
    balanced = 0
    status = 0
    support = 0
    for pivot in common:
        left = edge_lookup.get((source, pivot), 0)
        right = edge_lookup.get((pivot, target), 0)
        if not left or not right:
            continue
        support += 1
        if left * right * candidate_label > 0:
            balanced += 1
        if (
            (left > 0 and right > 0 and candidate_label > 0)
            or (left < 0 and right < 0 and candidate_label > 0)
            or (left > 0 and right < 0 and candidate_label < 0)
        ):
            status += 1
    return balanced, status, support


def build_triangle_features(source, target, relations, edge_lookup):
    bal_pos, status_pos, support = _triangle_counts(source, target, 1, relations, edge_lookup)
    bal_neg, status_neg, _ = _triangle_counts(source, target, -1, relations, edge_lookup)
    return np.asarray(
        [
            bal_pos - bal_neg,
            status_pos - status_neg,
            bal_pos,
            bal_neg,
            status_pos,
            status_neg,
            support,
        ],
        dtype=float,
    )
''',
    encoding="utf-8",
)
PY

python /app/run_experiment.py --split dev --output /app/results.json
