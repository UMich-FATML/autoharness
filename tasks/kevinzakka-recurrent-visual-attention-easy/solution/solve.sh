#!/bin/bash
set -euo pipefail

# CANARY: harbor-pwc-rva-easy

python - <<'PY'
from pathlib import Path

target = Path("/app/modules.py")
target.write_text(
    '''import numpy as np

from config import IMAGE_SIZE, PATCH_SIZE


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
    # Oracle implementation: use the fixation location to extract the paper's retina crop.
    # This preserves the implement -> run -> observe loop while restoring the actual observation primitive.
    glimpses = []
    for scale_index in range(scales):
        crop_size = patch_size * (2 ** scale_index)
        glimpse = _crop_and_resize(image, location, crop_size, patch_size)
        glimpses.append(glimpse.reshape(-1))
    return np.concatenate(glimpses, axis=0)
''',
    encoding="utf-8",
)
PY

python /app/main.py --split dev --output /app/results.json
