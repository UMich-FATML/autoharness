import math

import numpy as np

ALPHA = 0.05
DEFAULT_SAMPLE_SIZE = 20
DEFAULT_PAIR_COUNT = 900
DEFAULT_BOOTSTRAP_REPS = 256
DEFAULT_PERMUTATION_REPS = 256
POOLED_STD = math.sqrt((1.0 ** 2 + 2.0 ** 2) / 2.0)

_STANDARD_LOGNORMAL_MEAN = math.exp(0.5)
_STANDARD_LOGNORMAL_STD = math.sqrt((math.e - 1.0) * math.e)
_STANDARD_LOGNORMAL_MEDIAN = (1.0 - _STANDARD_LOGNORMAL_MEAN) / _STANDARD_LOGNORMAL_STD

SHIFT_BY_KIND = {
    "mean": POOLED_STD,
    "median": POOLED_STD - 2.0 * _STANDARD_LOGNORMAL_MEDIAN,
}

PUBLIC_SCENARIOS = {
    "table13_medium_n20_welch": {
        "kind": "single",
        "metric_name": "welch_power",
        "paper_target": 0.973,
        "pair_count": DEFAULT_PAIR_COUNT,
        "sample_size": DEFAULT_SAMPLE_SIZE,
        "seed": 1904,
        "shift_kind": "mean",
        "test_name": "welch",
        "rng_seed_base": 5100,
        "bootstrap_reps": DEFAULT_BOOTSTRAP_REPS,
        "permutation_reps": DEFAULT_PERMUTATION_REPS,
    },
    "table13_medium_n20_bootstrap": {
        "kind": "single",
        "metric_name": "bootstrap_power",
        "paper_target": 0.966,
        "pair_count": DEFAULT_PAIR_COUNT,
        "sample_size": DEFAULT_SAMPLE_SIZE,
        "seed": 1904,
        "shift_kind": "mean",
        "test_name": "bootstrap",
        "rng_seed_base": 9100,
        "bootstrap_reps": DEFAULT_BOOTSTRAP_REPS,
        "permutation_reps": DEFAULT_PERMUTATION_REPS,
    },
    "table13_medium_n20_resampling_row": {
        "kind": "row",
        "pair_count": DEFAULT_PAIR_COUNT,
        "sample_size": DEFAULT_SAMPLE_SIZE,
        "mean_seed": 1904,
        "median_seed": 2013,
        "mean_tests": ["bootstrap", "permutation"],
        "median_tests": ["ranked_t"],
        "rng_seed_base": {
            "bootstrap": 12100,
            "permutation": 15100,
            "ranked_t": 17100,
        },
        "paper_targets": {
            "bootstrap": 0.966,
            "permutation": 0.989,
            "ranked_t": 1.0,
        },
        "bootstrap_reps": DEFAULT_BOOTSTRAP_REPS,
        "permutation_reps": DEFAULT_PERMUTATION_REPS,
    },
}


def standardized_lognormal(shape, rng):
    raw = rng.lognormal(mean=0.0, sigma=1.0, size=shape)
    return (raw - _STANDARD_LOGNORMAL_MEAN) / _STANDARD_LOGNORMAL_STD


def generate_pairs(pair_count, sample_size, seed, shift_kind):
    rng = np.random.default_rng(seed)
    samples_a = rng.normal(loc=0.0, scale=1.0, size=(pair_count, sample_size))
    standardized = standardized_lognormal((pair_count, sample_size), rng)
    shift = SHIFT_BY_KIND[shift_kind]
    samples_b = 2.0 * standardized + shift
    return samples_a, samples_b


def get_public_scenario(name):
    try:
        return dict(PUBLIC_SCENARIOS[name])
    except KeyError as exc:
        raise ValueError(f"unknown scenario: {name}") from exc
