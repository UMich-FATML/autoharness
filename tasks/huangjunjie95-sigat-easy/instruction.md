Implement the direction-aware component that corresponds to the paper's `Ldirection` objective and use it to improve the reduced Bitcoin-Alpha evaluation slice. In Table 5 of the SDGNN paper, adding direction reconstruction moves Macro-F1 from `0.6738` to `0.7414`. The current code already builds one signed directed relation mean-aggregation layer, but `build_direction_features` is a zero baseline.

The visible train/dev splits are already staged. You can edit the code, run `python /app/run_experiment.py --split dev --output /app/results.json`, inspect the dev Macro-F1, and iterate.

Write your final result to `/app/results.json`. Your score is based on how close the hidden Macro-F1 gets to the paper-aligned target, so partial progress receives partial credit.

