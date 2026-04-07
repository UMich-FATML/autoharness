import importlib.util
import json
import os

import numpy as np


def load_hidden_specs(module_path: str):
    if not module_path:
        raise ValueError("--hidden-module is required for hidden_test runs")
    spec = importlib.util.spec_from_file_location("hidden_split", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load hidden split from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HIDDEN_SPECS


def fit_ridge_classifier(features: np.ndarray, labels: np.ndarray, num_classes: int, regularization: float):
    augmented = np.concatenate([features, np.ones((features.shape[0], 1), dtype=float)], axis=1)
    targets = np.eye(num_classes, dtype=float)[labels]
    gram = augmented.T @ augmented + regularization * np.eye(augmented.shape[1], dtype=float)
    weights = np.linalg.solve(gram, augmented.T @ targets)
    return weights


def predict_labels(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    augmented = np.concatenate([features, np.ones((features.shape[0], 1), dtype=float)], axis=1)
    return np.argmax(augmented @ weights, axis=1)


def accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean(labels == predictions))


def write_results(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
