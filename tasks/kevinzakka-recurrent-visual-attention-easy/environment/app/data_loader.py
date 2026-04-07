import numpy as np

from config import EASY_DEV_REPEATS, EASY_POSITIONS, EASY_TRAIN_REPEATS, IMAGE_SIZE

SEGMENTS = {
    0: "abcefd",
    1: "bc",
    2: "abged",
    3: "abgcd",
    4: "fgbc",
    5: "afgcd",
    6: "afgcde",
    7: "abc",
    8: "abcdefg",
    9: "abfgcd",
}

SEGMENT_COORDS = {
    "a": ((1, 1), (1, 6)),
    "b": ((1, 6), (3, 6)),
    "c": ((4, 6), (6, 6)),
    "d": ((6, 1), (6, 6)),
    "e": ((4, 1), (6, 1)),
    "f": ((1, 1), (3, 1)),
    "g": ((3, 1), (3, 6)),
}


def _draw_segment(image: np.ndarray, segment: str) -> None:
    (row0, col0), (row1, col1) = SEGMENT_COORDS[segment]
    if row0 == row1:
        image[row0 : row0 + 1, col0 : col1 + 1] = 1.0
        image[min(row0 + 1, 7) : min(row0 + 2, 8), col0 : col1 + 1] = 0.7
    else:
        image[row0 : row1 + 1, col0 : col0 + 1] = 1.0
        image[row0 : row1 + 1, min(col0 + 1, 7) : min(col0 + 2, 8)] = 0.7


def digit_patch(digit: int) -> np.ndarray:
    image = np.zeros((8, 8), dtype=float)
    for segment in SEGMENTS[digit]:
        _draw_segment(image, segment)
    return image


def pixel_to_location(row: int, col: int) -> np.ndarray:
    x = (col / (IMAGE_SIZE - 1)) * 2.0 - 1.0
    y = (row / (IMAGE_SIZE - 1)) * 2.0 - 1.0
    return np.asarray([x, y], dtype=float)


def easy_location(position_index: int) -> np.ndarray:
    row, col = EASY_POSITIONS[position_index]
    return pixel_to_location(row, col)


def render_easy_image(digit: int, position_index: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=float)
    row, col = EASY_POSITIONS[position_index]
    patch = digit_patch(digit)
    canvas[row - 4 : row + 4, col - 4 : col + 4] += patch
    canvas += rng.normal(0.0, 0.03, size=canvas.shape)
    return np.clip(canvas, 0.0, 1.0)


def easy_train_specs():
    return [
        {"digit": digit, "pos_idx": pos_idx, "seed": 1000 + digit * 100 + pos_idx * 10 + repeat}
        for digit in range(10)
        for pos_idx in range(len(EASY_POSITIONS))
        for repeat in range(EASY_TRAIN_REPEATS)
    ]


def easy_dev_specs():
    return [
        {
            "digit": digit,
            "pos_idx": pos_idx,
            "seed": 1000 + digit * 100 + pos_idx * 10 + EASY_TRAIN_REPEATS + repeat,
        }
        for digit in range(10)
        for pos_idx in range(len(EASY_POSITIONS))
        for repeat in range(EASY_DEV_REPEATS)
    ]


def build_easy_split(specs):
    split = []
    for spec in specs:
        split.append(
            {
                "digit": int(spec["digit"]),
                "image": render_easy_image(spec["digit"], spec["pos_idx"], spec["seed"]),
                "location": easy_location(spec["pos_idx"]),
                "position_index": int(spec["pos_idx"]),
                "seed": int(spec["seed"]),
            }
        )
    return split
