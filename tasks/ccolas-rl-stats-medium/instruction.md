Implement the bootstrap confidence-interval test from the paper and use it to reproduce the `Table 13` medium-effect (`epsilon = 1.0`) `N = 20` statistical power reported for the normal-vs-log-normal comparison. The target paper value for bootstrap power is `0.966`.

The codebase already generates a fixed deterministic study batch and writes `/app/results.json` when you run `python3 /app/run_experiment.py --scenario table13_medium_n20_bootstrap`. The current implementation in `/app/rl_stats/stats_impl.py` uses a crude shortcut instead of the paper's bootstrap resampling procedure.

You can modify the code, rerun the provided scenario, inspect `/app/results.json`, and iterate. Your final output must be `/app/results.json` with exactly the keys `"achieved_power"`, `"metric_name"`, `"paper_target"`, and `"scenario"`. Your score is based on how close the achieved power gets to the paper's reported value, and partial progress receives partial credit.
