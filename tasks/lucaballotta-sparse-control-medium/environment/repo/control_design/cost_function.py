import numpy as np


def normalize_schedule(schedule):
    return [tuple(sorted(set(slot))) for slot in schedule]


def reachability_matrix(A, B, schedule, horizon):
    blocks = []
    for step, active in enumerate(normalize_schedule(schedule)):
        if active:
            block = np.linalg.matrix_power(A, horizon - 1 - step) @ B[:, list(active)]
            blocks.append(block)
    if not blocks:
        return np.zeros((A.shape[0], 0), dtype=float)
    return np.concatenate(blocks, axis=1)


def controllability_gramian(A, B, schedule, horizon):
    phi = reachability_matrix(A, B, schedule, horizon)
    return phi @ phi.T


def schedule_rank(A, B, schedule, horizon, tol=1e-9):
    gramian = controllability_gramian(A, B, schedule, horizon)
    return int(np.linalg.matrix_rank(gramian, tol))


def trace_inverse_cost(A, B, schedule, horizon, delta=1e-9):
    gramian = controllability_gramian(A, B, schedule, horizon)
    if np.linalg.matrix_rank(gramian) == gramian.shape[0]:
        return float(np.trace(np.linalg.inv(gramian)))
    regularized = gramian + delta * np.eye(gramian.shape[0], dtype=float)
    return float(np.trace(np.linalg.inv(regularized)))


def fully_actuated_schedule(num_channels, horizon):
    return [tuple(range(num_channels)) for _ in range(horizon)]
