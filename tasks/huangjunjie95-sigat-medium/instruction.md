Implement the second signed directed relation mean-aggregation layer so the encoder becomes truly layer-by-layer, as in Figure 2 and the Table 4 mean-aggregator ablation. In Table 4 of the SDGNN paper, the `2-Layer-MEAN-AGG` model slightly outperforms the `1-Layer-MEAN-AGG` model on Bitcoin-Alpha. The current code already includes the direction-aware feature block, but `apply_relation_stack` still stops after one propagation layer.

The visible train/dev splits are already staged. You can edit the code, run `python /app/run_experiment.py --split dev --output /app/results.json`, inspect the dev Macro-F1, and iterate.

Write your final result to `/app/results.json`. Your score is based on how close the hidden Macro-F1 gets to the paper-aligned target, so partial progress receives partial credit.
