#!/bin/bash
set -euo pipefail

# CANARY: harbor-pwc-sdgnn-medium

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


def build_triangle_features(source, target, relations, edge_lookup):
    return np.zeros(7, dtype=float)
''',
    encoding="utf-8",
)
PY

python /app/run_experiment.py --split dev --output /app/results.json
