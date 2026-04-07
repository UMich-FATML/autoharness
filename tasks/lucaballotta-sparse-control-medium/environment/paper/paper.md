# Pointwise-Sparse Actuator Scheduling for Linear Systems With Controllability Guarantee

This staged paper note is a task-grounding digest built from the accessible published PDF for the Ballotta, Joseph, and Thete paper. It keeps the definitions, Example 1 data, and algorithm details used by the task family.

## Metadata

- Venue: IEEE Control Systems Letters, 2024
- DOI: `10.1109/LCSYS.2024.3475886`
- arXiv: `2407.12125`
- Repository: `https://github.com/lucaballotta/sparse-control`

## Problem Setup

The paper studies sparse actuator schedules for the discrete-time linear system

`x(k + 1) = A x(k) + B u(k)`.

An actuator schedule is an ordered tuple `S = (S_0, ..., S_{h-1})` where each `S_k` is a subset of input-channel indices. The schedule is pointwise `s`-sparse if `|S_k| <= s` for every time step.

For a horizon `h`, the paper defines the schedule-dependent reachability matrix

`Phi_S^h = [A^(h-1) B_{S_0}  A^(h-2) B_{S_1}  ...  B_{S_{h-1}}]`

and the schedule-dependent controllability Gramian

`W_S^h = Phi_S^h (Phi_S^h)^T`.

Problem 1 asks for a pointwise `s`-sparse schedule that minimizes a control-energy proxy `rho(W_S^h)` while preserving controllability, i.e. `rank(W_S^h) = n`.

## Assumption 1

The design algorithms assume:

- `s >= max(n - rank(A), 1)`
- `h >= h*`, where `h*` is a controllability horizon allowed by Proposition 2

## Example 1 Benchmark

The paper's small worked example uses:

- `n = 5`
- `m = 7`
- `s = 1`
- `h = 5`
- cost metric `rho(W) = Tr(W^-1)`

with

```text
A =
[[0, 1, 0, 0, 0],
 [0, 0, 0, 1, 0],
 [0, 0, 1, 0, 0],
 [0, 0, 0, 0, 1],
 [0, 0, 0, 0, 0]]

B =
[[0, 0, 1, 0, 0, 0, 1],
 [0, 0, 1, 0, 0, 1, 0],
 [1, 0, 0, 0, 1, 0, 1],
 [1, 1, 0, 0, 0, 0, 1],
 [0, 0, 0, 1, 0, 0, 0]]
```

The paper reports in Table II:

- fully actuated baseline cost: approximately `1.7`
- `s`-sparse greedy cost: `5.0`

The fixed constrained schedule found by the greedy method activates the fourth input channel at time steps `k = 1, 2, 3, 4` in the paper's one-based indexing. In the zero-based indexing used by this staged code, that channel is index `3`.

## Algorithm 1: s-Sparse Greedy Selection

Algorithm 1 has three phases.

1. Start from a full schedule.
2. For each constrained time step `k = 1, ..., h - 1`, compute the left-kernel increment that must be spanned to preserve controllability, then greedily pick channels that span that increment.
3. If the schedule is still rank-deficient, greedily add channels that increase the controllability rank.
4. If slack remains in any time step, greedily add channels that further decrease the cost.

The key mandatory-direction object is the increment between consecutive left kernels of powers of `A`:

- compare `ker((A^T)^(h-k))` and `ker((A^T)^(h-1-k))`
- keep only the new direction(s) that appear at step `k`

The paper states that for Example 1 this preselection forces the key channel at every constrained time step.

## Algorithm 2: `greedy_k`

For one fixed time step `k`, Algorithm 2:

1. starts from the candidate channels that are not orthogonal to the required left-kernel increment,
2. adds the channel whose Gramian contribution gives the best cost,
3. removes candidates whose projected contributions are linearly dependent on already selected ones,
4. repeats until the increment is fully spanned.

The channel contribution used in the cost is

`phi_c^k = (A^(h-1-k) B_c) (A^(h-1-k) B_c)^T`.

## Algorithm 3: `greedy`

Algorithm 3 searches over all admissible `(time, channel)` additions and picks the one that minimizes the schedule cost. In the rank-enforcing mode, it also removes candidates that cannot increase the column-space rank any further.

## Table II Target Used by This Task Family

The task family uses the exact Example 1 quantitative claim:

- target metric: `Tr((W_S^h)^-1)`
- target value: `5.0`
- source: Table II, "Schedules and Costs for Example 1"

The fully actuated schedule is also reproduced in the staged code and evaluates to `1.6663785652549696`, which matches the paper's rounded `1.7`.
