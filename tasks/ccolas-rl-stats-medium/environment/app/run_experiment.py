#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from rl_stats.scenarios import get_public_scenario, generate_pairs
from rl_stats.stats_impl import rejection_rate_for_test


def run_single_metric(scenario_name: str) -> dict:
    scenario = get_public_scenario(scenario_name)
    pair_batch = generate_pairs(
        pair_count=scenario["pair_count"],
        sample_size=scenario["sample_size"],
        seed=scenario["seed"],
        shift_kind=scenario["shift_kind"],
    )
    achieved_power = rejection_rate_for_test(
        scenario["test_name"],
        pair_batch,
        rng_seed_base=scenario["rng_seed_base"],
        bootstrap_reps=scenario["bootstrap_reps"],
        permutation_reps=scenario["permutation_reps"],
    )
    return {
        "achieved_power": round(achieved_power, 6),
        "metric_name": scenario["metric_name"],
        "paper_target": scenario["paper_target"],
        "scenario": scenario_name,
    }


def run_resampling_row(scenario_name: str) -> dict:
    scenario = get_public_scenario(scenario_name)
    mean_pairs = generate_pairs(
        pair_count=scenario["pair_count"],
        sample_size=scenario["sample_size"],
        seed=scenario["mean_seed"],
        shift_kind="mean",
    )
    median_pairs = generate_pairs(
        pair_count=scenario["pair_count"],
        sample_size=scenario["sample_size"],
        seed=scenario["median_seed"],
        shift_kind="median",
    )
    achieved = {}
    for test_name in scenario["mean_tests"]:
        achieved[test_name] = round(
            rejection_rate_for_test(
                test_name,
                mean_pairs,
                rng_seed_base=scenario["rng_seed_base"][test_name],
                bootstrap_reps=scenario["bootstrap_reps"],
                permutation_reps=scenario["permutation_reps"],
            ),
            6,
        )
    for test_name in scenario["median_tests"]:
        achieved[test_name] = round(
            rejection_rate_for_test(
                test_name,
                median_pairs,
                rng_seed_base=scenario["rng_seed_base"][test_name],
                bootstrap_reps=scenario["bootstrap_reps"],
                permutation_reps=scenario["permutation_reps"],
            ),
            6,
        )
    return {
        "paper_targets": scenario["paper_targets"],
        "powers": achieved,
        "scenario": scenario_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", default="/app/results.json")
    args = parser.parse_args()

    scenario = get_public_scenario(args.scenario)
    if scenario["kind"] == "single":
        payload = run_single_metric(args.scenario)
    else:
        payload = run_resampling_row(args.scenario)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
