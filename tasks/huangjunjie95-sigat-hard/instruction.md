Implement the triangle-theory feature block that closes the remaining gap between the direction-aware model and the paper's full Table 5 objective. In the SDGNN paper, moving from `Lsign + Ldirection` to `Lsign + Ldirection + Ltriangle` improves Macro-F1 on Bitcoin-Alpha from `0.7414` to `0.7585`. The current code already has the direction-aware block and two signed relation mean-propagation layers, but `build_triangle_features` is still a zero baseline.

The visible train/dev splits are already staged. You can edit the code, run `python /app/run_experiment.py --split dev --output /app/results.json`, inspect the dev Macro-F1, and iterate.

Write your final result to `/app/results.json`. Your score is based on how close the hidden Macro-F1 gets to the paper-aligned target, so partial progress receives partial credit.
