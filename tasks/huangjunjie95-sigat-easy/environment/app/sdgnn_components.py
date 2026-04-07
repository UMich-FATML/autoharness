import numpy as np

from sdgnn_core import default_relation_weights, mean_relation_layer


def apply_relation_stack(base_features, relations):
    weights = default_relation_weights()
    return mean_relation_layer(base_features, relations, weights[0])


def build_direction_features(source, target, relations):
    return np.zeros(3, dtype=float)


def build_triangle_features(source, target, relations, edge_lookup):
    return np.zeros(7, dtype=float)

