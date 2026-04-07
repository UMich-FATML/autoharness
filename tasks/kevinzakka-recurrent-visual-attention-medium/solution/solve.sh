#!/bin/bash
set -euo pipefail

# CANARY: harbor-pwc-rva-medium

python - <<'PY'
from pathlib import Path

target = Path("/app/modules.py")
target.write_text(
    '''import numpy as np

from config import GLIMPSE_DIM, IMAGE_SIZE, PARAMETER_SEED, PATCH_SIZE, STATE_DIM, WHAT_DIM, WHERE_DIM


def _fixed_parameters():
    rng = np.random.default_rng(PARAMETER_SEED)
    return {
        "w_phi1": rng.normal(0.0, 0.15, size=(PATCH_SIZE * PATCH_SIZE, WHAT_DIM)),
        "b_phi1": rng.normal(0.0, 0.05, size=(WHAT_DIM,)),
        "w_phi3": rng.normal(0.0, 0.12, size=(PATCH_SIZE * PATCH_SIZE * 3, WHAT_DIM)),
        "b_phi3": rng.normal(0.0, 0.05, size=(WHAT_DIM,)),
        "w_loc": rng.normal(0.0, 0.4, size=(2, WHERE_DIM)),
        "b_loc": rng.normal(0.0, 0.05, size=(WHERE_DIM,)),
        "w_what": rng.normal(0.0, 0.2, size=(WHAT_DIM, GLIMPSE_DIM)),
        "w_where": rng.normal(0.0, 0.2, size=(WHERE_DIM, GLIMPSE_DIM)),
        "b_g": rng.normal(0.0, 0.05, size=(GLIMPSE_DIM,)),
        "w_h": rng.normal(0.0, 0.2, size=(STATE_DIM, STATE_DIM)),
        "w_g": rng.normal(0.0, 0.2, size=(GLIMPSE_DIM, STATE_DIM)),
        "b_h": rng.normal(0.0, 0.05, size=(STATE_DIM,)),
    }


PARAMS = _fixed_parameters()


def _relu(value: np.ndarray) -> np.ndarray:
    return np.maximum(value, 0.0)


def _location_to_pixel(location: np.ndarray) -> tuple[int, int]:
    x, y = float(location[0]), float(location[1])
    col = int(round((x + 1.0) * 0.5 * (IMAGE_SIZE - 1)))
    row = int(round((y + 1.0) * 0.5 * (IMAGE_SIZE - 1)))
    return row, col


def _crop_and_resize(image: np.ndarray, location: np.ndarray, crop_size: int, output_size: int) -> np.ndarray:
    row, col = _location_to_pixel(location)
    half = crop_size // 2
    row0 = row - half
    col0 = col - half
    patch = np.zeros((crop_size, crop_size), dtype=float)

    src_row0 = max(row0, 0)
    src_col0 = max(col0, 0)
    src_row1 = min(row0 + crop_size, image.shape[0])
    src_col1 = min(col0 + crop_size, image.shape[1])

    dst_row0 = src_row0 - row0
    dst_col0 = src_col0 - col0
    patch[
        dst_row0 : dst_row0 + (src_row1 - src_row0),
        dst_col0 : dst_col0 + (src_col1 - src_col0),
    ] = image[src_row0:src_row1, src_col0:src_col1]

    if crop_size == output_size:
        return patch

    index = np.linspace(0, crop_size - 1, output_size)
    rows = np.clip(np.round(index).astype(int), 0, crop_size - 1)
    cols = np.clip(np.round(index).astype(int), 0, crop_size - 1)
    return patch[np.ix_(rows, cols)]


def retina_glimpse(image: np.ndarray, location: np.ndarray, patch_size: int = PATCH_SIZE, scales: int = 1) -> np.ndarray:
    glimpses = []
    for scale_index in range(scales):
        crop_size = patch_size * (2 ** scale_index)
        glimpse = _crop_and_resize(image, location, crop_size, patch_size)
        glimpses.append(glimpse.reshape(-1))
    return np.concatenate(glimpses, axis=0)


def glimpse_network(phi: np.ndarray, location: np.ndarray, scales: int = 1) -> np.ndarray:
    # Oracle implementation of the paper's what/where glimpse fusion.
    if scales == 1:
        what = _relu(phi @ PARAMS["w_phi1"] + PARAMS["b_phi1"])
    else:
        what = _relu(phi @ PARAMS["w_phi3"] + PARAMS["b_phi3"])
    where = _relu(location @ PARAMS["w_loc"] + PARAMS["b_loc"])
    return _relu(what @ PARAMS["w_what"] + where @ PARAMS["w_where"] + PARAMS["b_g"])


def core_step(previous_state: np.ndarray, glimpse_vector: np.ndarray) -> np.ndarray:
    # Oracle implementation of the rectified recurrent core from Section 4.
    return _relu(previous_state @ PARAMS["w_h"] + glimpse_vector @ PARAMS["w_g"] + PARAMS["b_h"])
''',
    encoding="utf-8",
)
PY

python /app/main.py --split dev --output /app/results.json
