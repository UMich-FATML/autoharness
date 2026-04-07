# MCF - Minimizer Counting Function

Provided an alphabet `Sigma`, two integers `m <= k`, and two words `w, gamma in Sigma^m`, we define the quantity

`pi_k^gamma(w) = |{x in Sigma^k : min_gamma(x) = w}|`

where `min_gamma(x)` is the smallest `m`-mer of `x` after XORing candidate `m`-mers against `gamma` and comparing them lexicographically.

The purpose of this module is to compute `pi_k^gamma(w)`.

## Lexicographical minimizer counting function

The special case `gamma = A...A` corresponds to the lexicographical minimizer counting function `pi_k(w)`.

The `LexMinimizerCountingFunction` class implements the equations and principles defined in:

> On the number of k-mers admitting a given lexicographical minimizer
> F. Ingels, C. Marchet, M. Salson
> https://arxiv.org/abs/2412.17492

Example:

```python
from src.lib import LexMinimizerCountingFunction

minimizer = "ACACAA"
k = 10

obj = LexMinimizerCountingFunction(minimizer)

print(obj.kmer_lower_bound(k))
print(obj.kmer(k))
print(obj.kmer_upper_bound(k))
```

Expected output:

```text
327
351
353
```

By default, the alphabet is `{'A', 'C', 'T', 'G'}`. A custom alphabet and custom letter order can also be supplied.

## Vigemin counting function

The original repository also exposes a `VigeminCountingFunction` for the XOR-keyed generalization. This reconstructed task artifact only vendors the lexicographical counting implementation needed for the Harbor tasks.

## Dependencies

- library runtime: Python standard library plus `numpy` for the regression helper
- plotting and paper-reproduction scripts in the original repository also use `matplotlib` and `scipy`
