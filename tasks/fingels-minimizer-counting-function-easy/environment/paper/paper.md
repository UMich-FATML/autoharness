# On the number of k-mers admitting a given lexicographical minimizer

## Abstract

The minimizer of a word of size `k` (a `k`-mer) is defined as its smallest substring of size `m` (with `m <= k`), according to some ordering on `m`-mers. Minimizers have been used in bioinformatics to partition sequencing datasets, binning together `k`-mers that share the same minimizer. It is folklore that using the lexicographical order leads to very unbalanced partitions, resulting in an abundant literature devoted to devising alternative orders for achieving better balanced partitions. This paper studies that imbalance theoretically and determines, for a given minimizer, how many `k`-mers admit the chosen minimizer. It gives an exact computation in `O(km)` space and `O(km^2)` time, introduces `O(k)`-space and `O(km)`-time approximations, and reports a close empirical correlation between theoretical and observed minimizer frequencies on genomic datasets.

## Links

- arXiv: `https://arxiv.org/abs/2412.17492`
- repository: `https://github.com/fingels/minimizer_counting_function`

## Recovered Grounding Notes

These are the paper claims that the reconstructed Harbor family is grounded on, recovered from the prior frozen pipeline run:

- Example 18 and Theorem 4 give exact `pi_k(w)` sequences for the `m=6` minimizers `ACACAA` and `ACACAC`.
- Section 5.2 and Figure 6 discuss lower and upper bounds for selected `m=6` minimizers and compare their tightness on the log scale.
- Section 5.2, Figure 8, and the setup around Conjecture 2 discuss fitting linear models to `log_4 pi_k(w)` as `k` grows.

## Fixed Panels Used By The Task Family

- Exact sequence panel: `ACACAA`, `ACACAC`
- Bounds and asymptotics panel: `AAAAAA`, `ACACAA`, `ACACAC`, `CAAAAA`, `GAAAAA`, `TAAAAA`
- Exact-sequence range: `k = 6..16`
- Bounds report checkpoints: `k = 10` and `k = 16`
- Asymptotic fit range: `k = 6..16`, prediction target `k = 20`
