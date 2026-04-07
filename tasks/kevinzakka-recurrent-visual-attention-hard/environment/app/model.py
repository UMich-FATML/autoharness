import numpy as np

import modules
from config import PATCH_SIZE, STATE_DIM


CENTER_LOCATION = np.asarray([0.0, 0.0], dtype=float)


def initial_state(image: np.ndarray) -> np.ndarray:
    first_glimpse = modules.retina_glimpse(image, CENTER_LOCATION, patch_size=PATCH_SIZE, scales=3)
    first_features = modules.glimpse_network(first_glimpse, CENTER_LOCATION, scales=3)
    return np.asarray(modules.core_step(np.zeros(STATE_DIM, dtype=float), first_features), dtype=float)


def attended_state(image: np.ndarray, previous_state: np.ndarray, location: np.ndarray) -> np.ndarray:
    second_glimpse = modules.retina_glimpse(image, location, patch_size=PATCH_SIZE, scales=1)
    second_features = modules.glimpse_network(second_glimpse, location, scales=1)
    return np.asarray(modules.core_step(previous_state, second_features), dtype=float)
