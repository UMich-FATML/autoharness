Implement the asymptotic Wilcoxon-Mann-Whitney test in `/app/wmwssp/R/utility.R` so the provided power simulation reproduces the epilepsy result from Table 2 of the paper. The target is the balanced `24/24` design from the seizure example, which the paper reports at power `0.802`.

The codebase already contains the bootstrap simulation loop and the Table 1 seizure data. The only missing piece is the core p-value calculation, which is currently a placeholder. You can modify the R source, run `Rscript /app/wmwssp/eval/run_easy.R`, observe the metric in `/app/results.json`, and iterate.

Write your final result to `/app/results.json` as a JSON object with exactly one key: `epilepsy_balanced_power`. Your score is based on how close the measured power is to the paper's reported result, so partial implementations that move the metric toward `0.802` receive partial credit.
