import numpy as np

import modules
from config import PATCH_SIZE


def easy_features(image: np.ndarray, location: np.ndarray) -> np.ndarray:
    return np.asarray(modules.retina_glimpse(image, location, patch_size=PATCH_SIZE, scales=1), dtype=float)
