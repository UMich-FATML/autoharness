import math

import mpmath as mp
import numpy as np

from rl_stats import ALPHA

mp.mp.dps = 50


def rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def student_t_two_sided_pvalue(t_stat, degrees_of_freedom):
    if degrees_of_freedom <= 0:
        return 1.0
    t_value = abs(float(t_stat))
    x_value = degrees_of_freedom / (degrees_of_freedom + t_value * t_value)
    p_value = mp.betainc(degrees_of_freedom / 2.0, 0.5, 0, x_value, regularized=True)
    return float(max(0.0, min(1.0, p_value)))


def equal_var_t_pvalue(sample_a, sample_b):
    sample_a = np.asarray(sample_a, dtype=float)
    sample_b = np.asarray(sample_b, dtype=float)
    n_a = len(sample_a)
    n_b = len(sample_b)
    mean_a = float(np.mean(sample_a))
    mean_b = float(np.mean(sample_b))
    var_a = float(np.var(sample_a, ddof=1))
    var_b = float(np.var(sample_b, ddof=1))
    pooled_variance = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    if pooled_variance <= 0.0:
        return 1.0
    t_stat = (mean_a - mean_b) / math.sqrt(pooled_variance * (1.0 / n_a + 1.0 / n_b))
    return student_t_two_sided_pvalue(t_stat, n_a + n_b - 2)


def heuristic_pvalue(sample_a, sample_b, scale=1.8):
    sample_a = np.asarray(sample_a, dtype=float)
    sample_b = np.asarray(sample_b, dtype=float)
    mean_gap = abs(float(np.mean(sample_a) - np.mean(sample_b)))
    variance_term = float(np.var(sample_a, ddof=1) / len(sample_a) + np.var(sample_b, ddof=1) / len(sample_b))
    normalized_gap = mean_gap / max(math.sqrt(variance_term) * scale, 1e-12)
    return max(0.0, min(1.0, math.erfc(normalized_gap / math.sqrt(2.0))))


def welch_t_pvalue(sample_a, sample_b):
    """Approximate placeholder. Replace this with the paper's Welch t-test."""
    return heuristic_pvalue(sample_a, sample_b, scale=2.25)


def mann_whitney_pvalue(sample_a, sample_b):
    sample_a = np.asarray(sample_a, dtype=float)
    sample_b = np.asarray(sample_b, dtype=float)
    n_a = len(sample_a)
    n_b = len(sample_b)
    ranks = rankdata(np.concatenate([sample_a, sample_b]))
    rank_sum_a = float(np.sum(ranks[:n_a]))
    u_stat = rank_sum_a - n_a * (n_a + 1) / 2.0
    mean_u = n_a * n_b / 2.0
    std_u = math.sqrt(n_a * n_b * (n_a + n_b + 1) / 12.0)
    if std_u == 0.0:
        return 1.0
    z_score = (u_stat - mean_u) / std_u
    return max(0.0, min(1.0, math.erfc(abs(z_score) / math.sqrt(2.0))))


def ranked_t_pvalue(sample_a, sample_b):
    joined = np.concatenate([np.asarray(sample_a, dtype=float), np.asarray(sample_b, dtype=float)])
    ranks = rankdata(joined)
    split = len(sample_a)
    return equal_var_t_pvalue(ranks[:split], ranks[split:])


def bootstrap_ci_rejects(sample_a, sample_b, alpha=ALPHA, bootstrap_reps=256, rng=None):
    sample_a = np.asarray(sample_a, dtype=float)
    sample_b = np.asarray(sample_b, dtype=float)
    observed_gap = abs(float(np.mean(sample_b) - np.mean(sample_a)))
    pooled_scale = math.sqrt(float(np.var(sample_a, ddof=1) + np.var(sample_b, ddof=1))) / max(len(sample_a), 1)
    return observed_gap > 2.2 * pooled_scale


def permutation_test_pvalue(sample_a, sample_b, permutation_reps=256, rng=None):
    return heuristic_pvalue(sample_a, sample_b, scale=2.0)


def rejection_rate_for_test(test_name, pair_batch, rng_seed_base, bootstrap_reps=256, permutation_reps=256):
    samples_a, samples_b = pair_batch
    decisions = []
    for index, (sample_a, sample_b) in enumerate(zip(samples_a, samples_b)):
        if test_name == "t_test":
            reject = equal_var_t_pvalue(sample_a, sample_b) < ALPHA
        elif test_name == "welch":
            reject = welch_t_pvalue(sample_a, sample_b) < ALPHA
        elif test_name == "mann_whitney":
            reject = mann_whitney_pvalue(sample_a, sample_b) < ALPHA
        elif test_name == "ranked_t":
            reject = ranked_t_pvalue(sample_a, sample_b) < ALPHA
        elif test_name == "bootstrap":
            rng = np.random.default_rng(rng_seed_base + index)
            reject = bootstrap_ci_rejects(
                sample_a,
                sample_b,
                alpha=ALPHA,
                bootstrap_reps=bootstrap_reps,
                rng=rng,
            )
        elif test_name == "permutation":
            rng = np.random.default_rng(rng_seed_base + index)
            reject = permutation_test_pvalue(
                sample_a,
                sample_b,
                permutation_reps=permutation_reps,
                rng=rng,
            ) < ALPHA
        else:
            raise ValueError(f"unknown test: {test_name}")
        decisions.append(bool(reject))
    return float(np.mean(decisions))
