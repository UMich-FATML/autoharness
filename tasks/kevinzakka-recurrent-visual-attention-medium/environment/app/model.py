import numpy as np

import modules
from config import PATCH_SIZE, STATE_DIM


def medium_state(image: np.ndarray, location: np.ndarray) -> np.ndarray:
    phi = modules.retina_glimpse(image, location, patch_size=PATCH_SIZE, scales=1)
    glimpse = modules.glimpse_network(phi, location, scales=1)
    return np.asarray(modules.core_step(np.zeros(STATE_DIM, dtype=float), glimpse), dtype=float)
