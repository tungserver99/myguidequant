# GuideQuant with Curvature-Matched Exact Pair Coordinate Descent (`B = 2`)

## Implementation specification for Codex

This document specifies a **minimal solver-only modification** to the official GuideQuant/LNQ pipeline.

The mathematical block size is denoted by \(B\). This implementation uses:

\[
\boxed{B=2}
\]

Do **not** call the block size \(K\), because \(K\) is reserved for the number of codewords:

\[
\mathcal C_u=\{c_{u,1},\ldots,c_{u,K}\}.
\]

The new assignment solver is an exact two-coordinate block coordinate descent method, referred to below as:

\[
\boxed{\text{Curvature-Matched Exact Pair Descent}}
\]

The optional accelerated pair subsolver exploits the Monge/monotone structure and may be called:

\[
\boxed{\text{Monge-Accelerated Pair Descent}}.
\]

---

# 1. Critical scope

## 1.1 What must remain exactly the same as GuideQuant

The following parts must be reused without changing their mathematical definition, artifact format, calibration procedure, or outer optimization schedule:

1. Calibration data and preprocessing.
2. GuideQuant Hessian collection.
3. Hessian cache paths, filenames, tensor layouts, dtypes, and loading logic.
4. Initialization labels.
5. Initialization centroids/codebooks.
6. Initialization artifact paths and formats.
7. Positive-definiteness check and Hessian damping already performed by GuideQuant.
8. The objective used by GuideQuant/LNQ.
9. The exact codebook/centroid update.
10. The order of alternating assignment and codebook updates.
11. Objective logging, early stopping, and best-state restoration.
12. Quantized label and LUT serialization.
13. Layer/module loading, saving, and pipelined I/O.
14. Model evaluation and inference code.

In the official repository, this means that functions and logic corresponding to the following must remain unchanged unless a small dispatcher argument is required:

- `objective_function`
- `update_C`
- Hessian loading and `fix_hessian_shape`
- initialization loading
- Hessian PD check/damping in `train_least_squares`
- the outer alternating-minimization loop in `train_least_squares`
- `seed_layer`
- layer loader/saver functions
- output artifact names and dtypes

## 1.2 The only mathematical change

Replace the scalar cyclic coordinate-descent assignment update:

\[
\text{one input coordinate at a time}
\]

with an exact pair update:

\[
\text{two input coordinates at a time}.
\]

Concretely, replace the production call to the original scalar `update_P` with a new exact pair version, while retaining the original scalar implementation as a baseline:

```python
if assignment_solver == "cd":
    labels = update_P_cd(...)
elif assignment_solver == "pair":
    labels = update_P_pair(...)
else:
    raise ValueError(...)
```

The existing `cd_cycles` argument should remain the number of assignment sweeps. It may be reused for the pair solver to avoid changing existing scripts.

## 1.3 Explicit non-goals

Do **not**:

- recollect Hessians;
- introduce a different Hessian estimator;
- add end-loss gradients or a linear term;
- add \(r^\top e\);
- change initialization;
- change the exact codebook update;
- replace the exact codebook solve with gradient descent;
- introduce ADMM, MM penalties, learning rates, temperatures, or damping parameters;
- sparsify or approximate \(H\);
- discard non-pair Hessian interactions;
- update all pairs simultaneously using stale residuals;
- change saved labels/LUT formats;
- implement the general \(B\)-block production solver in this task;
- implement \(B=3\) in this task.

This task is only the \(B=2\) GuideQuant assignment solver.

---

# 2. Baseline tensors and notation

For one quantized linear module, use the same tensors as the current GuideQuant implementation:

\[
W\in\mathbb R^{O\times D},
\]

where:

- \(O\): output dimension;
- \(D\): input dimension.

The Hessian artifact has shape:

\[
H\in\mathbb R^{G\times D\times D},
\]

where \(G\) is the number of Hessian/output groups.

The output rows are divided into groups of size:

\[
R=\frac{O}{G}.
\]

After reshaping:

\[
W_{\mathrm{grp}}\in\mathbb R^{G\times R\times D}.
\]

The codebook is still output-channel specific:

\[
C\in\mathbb R^{O\times K},
\]

and after grouping:

\[
C_{\mathrm{grp}}\in\mathbb R^{G\times R\times K}.
\]

The labels have shape:

\[
P\in\{0,\ldots,K-1\}^{O\times D}.
\]

The dequantized weight is:

\[
\widehat W_{u,i}=C_{u,P_{u,i}}.
\]

Define the quantization error:

\[
E=\widehat W-W.
\]

After grouping:

\[
E_{\mathrm{grp}}\in\mathbb R^{G\times R\times D}.
\]

---

# 3. GuideQuant assignment objective

For output row \(u\) belonging to Hessian group \(g(u)\), the assignment objective is:

\[
J_u(e_u)
=
\frac12 e_u^\top H_{g(u)}e_u.
\]

The factor \(1/2\) does not affect the minimizer. It is included only to simplify derivatives.

The total assignment objective is:

\[
J(E)
=
\sum_{u=1}^{O}
\frac12 e_u^\top H_{g(u)}e_u.
\]

There is no separate gradient or linear end-loss term in this task.

All cross-coordinate interactions from the full Hessian remain present.

---

# 4. Why scalar CD can be improved

Scalar CD updates one coordinate \(i\) while all other coordinates are fixed.

At a scalar-CD fixed point, every single-coordinate move is non-improving. However, it can still be possible that:

\[
J(e_i',e_j)>J(e_i,e_j),
\]

and:

\[
J(e_i,e_j')>J(e_i,e_j),
\]

while the joint move satisfies:

\[
J(e_i',e_j')<J(e_i,e_j).
\]

Scalar CD cannot cross this two-coordinate barrier because either intermediate single-coordinate move is worse.

Exact pair descent evaluates the joint pair state directly.

---

# 5. Exact pair subproblem

Choose two distinct input coordinates:

\[
(i,j).
\]

All coordinates outside the pair are fixed.

For one output row, define candidate errors:

\[
e_i(k)=c_k-w_i,
\qquad
e_j(\ell)=c_\ell-w_j,
\]

for:

\[
k,\ell\in\{1,\ldots,K\}.
\]

Define the external curvature fields:

\[
s_i^{(-ij)}
=
\sum_{m\notin\{i,j\}}H_{im}e_m,
\]

and:

\[
s_j^{(-ij)}
=
\sum_{m\notin\{i,j\}}H_{jm}e_m.
\]

The exact conditional pair energy is:

\[
\begin{aligned}
\Phi_{ij}(k,\ell)
={}&
\frac12 H_{ii}e_i(k)^2
+
\frac12 H_{jj}e_j(\ell)^2\\
&+
H_{ij}e_i(k)e_j(\ell)\\
&+
s_i^{(-ij)}e_i(k)
+
s_j^{(-ij)}e_j(\ell).
\end{aligned}
\]

The pair update is:

\[
\boxed{
(k^\star,\ell^\star)
=
\arg\min_{k,\ell\in\{1,\ldots,K\}}
\Phi_{ij}(k,\ell)
}
\]

and then:

\[
P_i\leftarrow k^\star,
\qquad
P_j\leftarrow \ell^\star.
\]

The current pair assignment is included in the \(K^2\) feasible candidates. Therefore:

\[
J_{\mathrm{after}}\leq J_{\mathrm{before}}.
\]

This is an exact conditional minimization of the original GuideQuant objective.

---

# 6. Very important Hessian requirement

## 6.1 Use the original symmetric damped Hessian

The current scalar-CD implementation divides each Hessian row by its diagonal:

```python
H_normalized[row, :] = H[row, :] / H[row, row]
```

That row-normalized matrix is useful for the closed-form scalar update.

It must **not** be used as the pair energy matrix.

For exact pair descent, use the original Hessian after the existing GuideQuant PD/damping step:

\[
H_{\mathrm{pair}}=H_{\mathrm{damped}}.
\]

The pair terms must use:

\[
H_{ii},\quad H_{jj},\quad H_{ij}=H_{ji}
\]

from this original damped Hessian.

Do not independently normalize row \(i\) and row \(j\). Row normalization destroys the symmetric two-variable quadratic form and would no longer solve the original objective exactly.

## 6.2 Symmetry

The GuideQuant Hessian is expected to be symmetric. The pair solver should:

1. assert that dimensions are correct;
2. verify symmetry in debug tests;
3. use the same damped Hessian passed to `objective_function` and `update_C`.

Do not introduce a different matrix only for pair assignment.

## 6.3 Positive diagonal

After the existing GuideQuant damping/PD check:

\[
H_{ii}>0.
\]

Assert this condition before pair solving.

No additional pair-specific damping parameter is allowed.

---

# 7. Efficient interaction accumulator

Do not call the interaction accumulator a loss gradient. It is only a cached Hessian-error product.

For grouped errors:

\[
E_{\mathrm{grp}}\in\mathbb R^{G\times R\times D},
\]

define:

\[
Z=E_{\mathrm{grp}}H,
\]

implemented as:

```python
Z = torch.bmm(E_grp, H)
```

with shape:

\[
Z\in\mathbb R^{G\times R\times D}.
\]

Because \(H\) is symmetric:

\[
Z_i
=
\sum_m H_{im}e_m.
\]

Therefore:

\[
s_i^{(-ij)}
=
Z_i-H_{ii}e_i-H_{ij}e_j,
\]

and:

\[
s_j^{(-ij)}
=
Z_j-H_{ij}e_i-H_{jj}e_j.
\]

These formulas must be used to avoid summing over all outside coordinates for every pair.

---

# 8. Curvature-based pair construction

## 8.1 Normalized coupling

Pairs should be chosen using only the Hessian.

For Hessian group \(g\), define:

\[
\rho_{ij}^{(g)}
=
\frac{
|H_{ij}^{(g)}|
}{
\sqrt{H_{ii}^{(g)}H_{jj}^{(g)}}
}.
\]

Use dtype-derived numerical protection only:

```python
tiny = torch.finfo(H.dtype).tiny
denom = torch.sqrt(torch.clamp(diag_i * diag_j, min=tiny))
```

Do not expose an epsilon hyperparameter.

## 8.2 One common matching across Hessian groups

To preserve the same vectorization across all output/Hessian groups, build one common coupling matrix:

\[
\boxed{
\rho_{ij}
=
\frac1G
\sum_{g=1}^{G}
\rho_{ij}^{(g)}
}
\]

and construct one pair order shared by all groups.

When \(G=1\), this is exactly the ordinary normalized coupling.

A common pair order allows all Hessian groups and output rows to be processed in the same batched GPU kernels.

## 8.3 Deterministic greedy matching

Do not require Blossom or another generic graph library.

Use a deterministic greedy maximal matching:

1. Set the diagonal of \(\rho\) to \(-\infty\).
2. For every node \(i\), compute:

   \[
   p_i=\max_{j\neq i}\rho_{ij}.
   \]

3. Sort nodes by descending \(p_i\), with index order as the deterministic tie-break.
4. Iterate through this node order:
   - skip \(i\) if already used;
   - among currently unused nodes, choose:

     \[
     j^\star
     =
     \arg\max_{j\ \mathrm{unused},\,j\neq i}\rho_{ij};
     \]

   - pair \((i,j^\star)\);
   - mark both nodes used.
5. If \(D\) is odd, keep one singleton coordinate.

The matching is computed once per module/Hessian and reused for:

- all output rows;
- all assignment cycles;
- all outer alternating-minimization iterations, unless the Hessian changes.

The Hessian does not change in GuideQuant LNQ, so there is no need to rematch after codebook updates.

## 8.4 Permutation

Flatten the pair list into a permutation:

\[
\pi=(i_1,j_1,i_2,j_2,\ldots).
\]

If there is a singleton, append it at the end.

Internally permute the input-coordinate dimension:

\[
W^\pi=W_{:,\pi},
\]

\[
P^\pi=P_{:,\pi},
\]

\[
H^\pi=H_{\pi,\pi}.
\]

The codebook \(C\) is per output row, not per input coordinate, so it is not permuted.

Pairs are now adjacent:

\[
(0,1),(2,3),(4,5),\ldots.
\]

At the end of `update_P_pair`, inverse-permute the labels back to the original input-coordinate order before returning them to the unchanged GuideQuant pipeline.

---

# 9. Reference exact pair solver: vectorized \(K^2\)

Implement this solver first. It is the correctness oracle for all optimized backends.

For one adjacent pair \((i,j)\), construct:

\[
E_i^{\mathrm{cand}}
=
C_{\mathrm{grp}}-W_{\mathrm{grp},i},
\]

and:

\[
E_j^{\mathrm{cand}}
=
C_{\mathrm{grp}}-W_{\mathrm{grp},j}.
\]

Shapes:

```text
E_i_cand: [G, R, K]
E_j_cand: [G, R, K]
```

Broadcast to form the exact cost tensor:

```text
pair_cost: [G, R, K, K]
```

using:

\[
\begin{aligned}
\Phi(k,\ell)
={}&
\frac12H_{ii}E_i(k)^2
+
\frac12H_{jj}E_j(\ell)^2\\
&+
H_{ij}E_i(k)E_j(\ell)\\
&+
s_iE_i(k)
+
s_jE_j(\ell).
\end{aligned}
\]

Flatten the last two dimensions, take `argmin`, and recover:

```python
k_star = flat_index // K
l_star = flat_index % K
```

Update:

- labels at \(i,j\);
- dequantized weights at \(i,j\);
- errors at \(i,j\).

This solver must:

- contain no Python loop over output rows;
- contain no Python loop over Hessian groups;
- vectorize all \(K^2\) candidates;
- produce exact results for arbitrary unsorted codebooks.

Suggested function:

```python
solve_pair_bruteforce(
    W_i,
    W_j,
    C_grp,
    e_i_current,
    e_j_current,
    z_i,
    z_j,
    H_ii,
    H_jj,
    H_ij,
) -> PairSolveResult
```

---

# 10. Monge/one-variable elimination

After the brute-force solver is correct, add an exact accelerated backend.

## 10.1 Eliminate one coordinate

For fixed label \(\ell\) of coordinate \(j\), minimize over coordinate \(i\):

\[
\min_{e_i\in\mathcal E_i}
\left[
\frac12H_{ii}e_i^2
+
\left(
s_i^{(-ij)}+H_{ij}e_j(\ell)
\right)e_i
\right].
\]

The continuous minimizer is:

\[
\boxed{
\bar e_i(\ell)
=
-
\frac{
s_i^{(-ij)}+H_{ij}e_j(\ell)
}{
H_{ii}
}
}
\]

and the exact discrete minimizer is the candidate error nearest to \(\bar e_i(\ell)\):

\[
k^\star(\ell)
=
\arg\min_k
\left|
e_i(k)-\bar e_i(\ell)
\right|.
\]

Then evaluate only:

\[
\Phi\left(k^\star(\ell),\ell\right)
\]

for:

\[
\ell=1,\ldots,K.
\]

This reduces exact candidate evaluation from \(K^2\) to \(K\), assuming the nearest-codeword query is performed by the monotone scan.

## 10.2 Do not mutate the GuideQuant codebook

The exact `update_C` output is not required to be sorted.

Do not sort and overwrite the stored codebook, because the saved codebook and labels must preserve the existing GuideQuant representation.

Instead, create a temporary sorted view inside `update_P_pair`:

```python
C_sorted, sorted_to_original = torch.sort(C_grp, dim=-1)
```

For any coordinate \(i\):

\[
e_i^{\mathrm{sorted}}=C_{\mathrm{sorted}}-W_i.
\]

After selecting sorted indices, map them back to the original codebook labels:

```python
original_label = gather(sorted_to_original, sorted_label)
```

This temporary sorting does not alter:

- the exact codebook values;
- the quantized weights;
- the saved codebook;
- the `update_C` implementation.

## 10.3 Monotonicity

Because:

\[
e_j(1)<e_j(2)<\cdots<e_j(K),
\]

the continuous optimum:

\[
\bar e_i(\ell)
=
-\frac{s_i+H_{ij}e_j(\ell)}{H_{ii}}
\]

is monotonic in \(\ell\).

- If \(H_{ij}<0\), scan \(e_j\) in ascending order.
- If \(H_{ij}>0\), scan \(e_j\) in descending order.

Under this orientation, \(\bar e_i(\ell)\) is nondecreasing, so the nearest index in the sorted \(i\)-codebook is also nondecreasing.

A pointer therefore moves only forward across the \(K\) codewords.

The exact pair solve becomes:

\[
O(K)
\]

per output row/pair.

## 10.4 Required implementation order

Implement backends in this order:

1. `bruteforce`: exact \(K^2\), correctness oracle;
2. `searchsorted`: exact one-variable elimination with batched temporary sorting;
3. `monotone`: exact pointer scan, only after it matches brute force.

`searchsorted` is allowed as an intermediate production backend. It is exact and GPU-friendly, although its formal cost is \(O(K\log K)\).

The final `monotone` backend must be tested against brute force for random and adversarial cases.

Do not use SMAWK unless profiling proves it necessary.

For the practical GuideQuant codebook sizes \(K\in\{4,8,16\}\), benchmark all exact backends. A tensorized \(K^2\) implementation may be faster than an irregular pointer kernel despite worse asymptotic complexity.

The method must remain exact regardless of which backend is selected.

---

# 11. Lazy pair-panel updates

## 11.1 Parallelism and sequential dependence

The following dimensions are independent and should be processed in parallel:

- output rows within each Hessian group;
- Hessian groups;
- codeword candidates;
- pair-cost candidates for the brute-force backend;
- module/layer I/O exactly as in GuideQuant.

Pairs themselves are **not independent**, because different pairs still interact through the full dense Hessian.

Therefore, to preserve exact Gauss-Seidel block descent and monotonic decrease:

\[
\boxed{\text{pairs must be processed sequentially within one assignment sweep}.}
\]

Do not update all disjoint pairs simultaneously from stale external fields. That would be a Jacobi update and would lose the exact-descent guarantee.

## 11.2 Reuse GuideQuant's lazy batching idea

The current GuideQuant scalar CD uses a coordinate block size of 128 and defers global residual propagation until the end of each block.

Use the same idea for pairs.

Let:

```python
pair_panel_coord_size = existing_cd_block_size
```

Require it to be even. With the existing value:

```text
128 coordinates = 64 pairs per panel
```

Do not add a new tuning hyperparameter unless required for debugging. Reuse the existing constant/configuration.

## 11.3 Exact panel algorithm

After pair permutation, initialize at the start of every pair sweep:

\[
Z=E^\pi H^\pi.
\]

For a panel of coordinates:

\[
P=[s,e),
\]

where \(e-s\) is even, allocate:

\[
\Delta E_P=0.
\]

Process adjacent pairs sequentially:

\[
(s,s+1), (s+2,s+3),\ldots.
\]

For a pair \((i,j)\):

1. Read current:
   - \(e_i,e_j\);
   - \(Z_i,Z_j\);
   - \(H_{ii},H_{jj},H_{ij}\).

2. Compute:

   \[
   s_i=Z_i-H_{ii}e_i-H_{ij}e_j,
   \]

   \[
   s_j=Z_j-H_{ij}e_i-H_{jj}e_j.
   \]

3. Solve the exact pair problem.

4. Compute:

   \[
   \delta_i=e_i^{\mathrm{new}}-e_i,
   \qquad
   \delta_j=e_j^{\mathrm{new}}-e_j.
   \]

5. Update labels, dequantized weights, and errors.

6. Store \(\delta_i,\delta_j\) in \(\Delta E_P\).

7. Immediately update the interaction accumulator only for the remaining coordinates inside the same panel:

   \[
   Z_{j+1:e}
   \leftarrow
   Z_{j+1:e}
   +
   \delta_i H_{i,j+1:e}
   +
   \delta_j H_{j,j+1:e}.
   \]

After every pair in the panel has been processed, update all future coordinates with one batched matrix multiplication:

\[
Z_{e:D}
\leftarrow
Z_{e:D}
+
\Delta E_P H_{P,e:D}.
\]

In PyTorch, this should be a grouped `torch.bmm`.

This preserves the exact state needed by every future pair while avoiding a full rank-2 update over all \(D\) coordinates after every pair.

## 11.4 Do not update past coordinates

Past coordinates in a Gauss-Seidel sweep will not be revisited until the next sweep.

There is no need to propagate pair changes into the cached \(Z\) values of past coordinates.

At the beginning of the next sweep, recompute:

\[
Z=EH
\]

from the current full error matrix.

---

# 12. Odd input dimension

If \(D\) is odd, one coordinate remains after matching.

Process it with the original exact scalar coordinate update at the end of every pair sweep.

For singleton \(i\):

\[
s_i^{(-i)}
=
Z_i-H_{ii}e_i,
\]

and:

\[
e_i^\star
=
\operatorname{Nearest}_{e_i(k)}
\left(
-\frac{s_i^{(-i)}}{H_{ii}}
\right).
\]

This is not a new solver mode. It is only the exact handling of the unavoidable unpaired coordinate.

---

# 13. Integration into the existing GuideQuant code

## 13.1 Target file

The primary modification should be localized to:

```text
any_precision/quantization/layerwise_quantize.py
```

Avoid unrelated refactors.

## 13.2 Preserve the original assignment solver

Rename or wrap the current function as:

```python
update_P_cd(...)
```

Add:

```python
update_P_pair(...)
```

and a small dispatcher.

Suggested helper functions:

```python
build_normalized_curvature(H)
build_greedy_pair_matching(H)
build_pair_permutation(pairs, singleton, D)
permute_pair_problem(W, H, labels)
solve_pair_bruteforce(...)
solve_pair_searchsorted(...)
solve_pair_monotone(...)
update_P_pair(...)
```

Use small, testable functions. Do not place the entire implementation in one very large function.

## 13.3 Outer loop

Keep the current outer loop unchanged:

```text
initial labels and centroids
    ↓
exact GuideQuant codebook update
    ↓
assignment update for the configured number of cycles
    ↓
exact GuideQuant codebook update
    ↓
objective logging / best-state / early stop
```

Only replace:

```python
labels = update_P(...)
```

with the solver dispatcher.

Do not alter the behavior where the first outer iteration performs the existing codebook update before later assignment updates.

## 13.4 Exact codebook update

`update_C` must remain mathematically and programmatically unchanged.

In particular, retain:

- the existing Cholesky use;
- one-hot assignment construction;
- batching over output channels/input dimension;
- least-squares solve;
- existing regularization behavior;
- NaN checks;
- returned dtype and shape.

The pair method changes assignment only.

## 13.5 Artifact loading and saving

Do not change:

```text
initialization_path/weights/l{layer}.pt
initialization_path/lut_{bit}/l{layer}.pt
hessians_path/l{layer}.pt
output_folder/weights/l{layer}.pt
output_folder/lut_{bit}/l{layer}.pt
```

Do not add a required matching artifact.

Matching should be generated from the loaded Hessian in memory and reused.

Optional debug caching is allowed only if it is disabled by default and does not alter the standard pipeline.

## 13.6 Command-line interface

For clean ablations, add only one optional argument if needed:

```python
--assignment_solver {cd,pair}
```

Recommended experimental default:

```text
pair
```

Retain:

```python
--cd_cycles
```

as the number of assignment sweeps for both solvers.

An optional internal/debug argument may select:

```python
--pair_backend {bruteforce,searchsorted,monotone}
```

but it should not affect the mathematical result.

Do not add pairing thresholds, top-\(k\) values, penalties, or learning rates.

---

# 14. Pseudocode

## 14.1 Full alternating optimization

```text
INPUT:
    W
    damped GuideQuant Hessian H
    initial labels P
    initial codebook C
    num_iterations
    cd_cycles

BUILD ONCE:
    pairs, singleton = curvature_matching(H)
    permutation and inverse permutation

best_state = current GuideQuant state

for outer_iteration in range(num_iterations):

    if outer_iteration > 0:
        P = exact_pair_assignment(
            W=W,
            H=H,
            labels=P,
            C=C,
            pairs=pairs,
            singleton=singleton,
            cycles=cd_cycles,
        )

    log original GuideQuant objective

    C = unchanged_exact_GuideQuant_codebook_update(
        W=W,
        H=H,
        labels=P,
        C=C,
    )

    log original GuideQuant objective
    apply unchanged best-state and early-stopping logic

RETURN:
    labels and codebook in the unchanged GuideQuant format
```

## 14.2 Pair assignment sweep

```text
temporarily permute W, labels, and H by curvature pair order
construct W_hat from labels and C
E = W_hat - W

for cycle in range(cd_cycles):

    Z = E @ H

    for panel_start in range(0, D_even, panel_coord_size):

        panel_end = ...
        Delta_panel = zeros

        for pair_start in range(panel_start, panel_end, 2):

            i = pair_start
            j = pair_start + 1

            external_i = Z[i] - H[i,i] * E[i] - H[i,j] * E[j]
            external_j = Z[j] - H[i,j] * E[i] - H[j,j] * E[j]

            new_labels_i, new_labels_j = exact_pair_solve(...)

            delta_i = new_error_i - E[i]
            delta_j = new_error_j - E[j]

            update labels and E
            write deltas into Delta_panel

            update Z for later coordinates inside this panel

        update Z for all coordinates after the panel with one BMM

    if singleton exists:
        perform one exact scalar update

inverse-permute labels
return labels
```

---

# 15. Required correctness tests

Codex must add tests before relying on performance benchmarks.

## 15.1 Pair solver oracle

For random small tensors:

- \(D\in\{2,3,4,8\}\);
- \(K\in\{2,4,8\}\);
- random positive-definite \(H\);
- random unsorted codebooks;
- random labels;

verify that:

```text
solve_pair_searchsorted == solve_pair_bruteforce
solve_pair_monotone == solve_pair_bruteforce
```

Compare selected quantized values and pair energy, not only raw label IDs when duplicate codebook values exist.

## 15.2 Exact objective decrease

For every exact pair update:

\[
J_{\mathrm{after}}
\leq
J_{\mathrm{before}}+\tau,
\]

where \(\tau\) is a dtype-derived debug tolerance, for example:

```python
eps = torch.finfo(dtype).eps
tol = 32 * eps * max(1.0, abs(objective_before))
```

This tolerance is only for assertions. It is not an optimization hyperparameter.

## 15.3 Full sweep decrease

Verify that the original GuideQuant `objective_function` does not increase after every pair sweep.

## 15.4 Diagonal Hessian

When \(H\) is diagonal:

- pair descent must reduce to independent nearest-codeword assignments;
- pair and scalar CD must return equivalent quantized weights.

## 15.5 Zero pair coupling

When:

\[
H_{ij}=0,
\]

the exact pair solution must equal the independent exact updates of \(i\) and \(j\).

## 15.6 Two-coordinate barrier

Construct a small case where:

- changing \(i\) alone increases the objective;
- changing \(j\) alone increases the objective;
- changing both decreases the objective.

Verify that scalar CD stays fixed and pair descent accepts the joint move.

## 15.7 Lazy update equivalence

For a small random problem, verify that:

- naive full `Z` rank-2 updates after every pair;
- lazy pair-panel updates;

produce identical labels, quantized values, and objective.

## 15.8 Permutation correctness

Verify that pair permutation followed by inverse permutation preserves:

- original weight coordinate order;
- label semantics;
- dequantized weights;
- objective value.

## 15.9 Odd dimension

Test \(D\) odd and verify the singleton coordinate is updated correctly.

## 15.10 Multiple Hessian groups

Test \(G>1\) and verify:

- common matching construction;
- correct row-to-Hessian-group mapping;
- batched output equivalence to a per-group reference loop.

## 15.11 Baseline preservation

With:

```text
assignment_solver = cd
```

the code must reproduce the original GuideQuant assignment behavior.

No baseline numerical changes are allowed.

## 15.12 Exact codebook preservation

Verify that the `update_C` function and its outputs are unchanged when given the same \(W,H,P,C\).

---

# 16. Performance requirements

Profile separately:

1. matching construction;
2. pair permutation;
3. `Z = E @ H`;
4. pair candidate solving;
5. within-panel propagation;
6. deferred panel BMM;
7. inverse permutation;
8. unchanged exact codebook update.

The following must be parallelized:

- all output rows;
- all Hessian groups;
- all codeword candidates;
- all \(K^2\) candidates in the brute-force reference;
- matrix propagation with `torch.bmm`;
- existing layer/module I/O.

The following remains sequential to preserve exact Gauss-Seidel descent:

- the ordered pair updates within one assignment sweep.

The expected sequential depth changes from approximately:

\[
D
\]

scalar decisions per sweep to:

\[
\left\lfloor\frac D2\right\rfloor
\]

pair decisions per sweep.

Do not claim a speedup before benchmarking. For small \(K\), the vectorized \(K^2\) backend may outperform the formal \(O(K)\) backend.

Select the production backend by measured GPU runtime, subject to exact equality with the brute-force oracle.

---

# 17. Logging and diagnostics

Retain all existing GuideQuant objective logs.

Add the following optional logs:

```text
assignment solver: pair
pair backend: brute-force/searchsorted/monotone
number of pairs
singleton coordinate, if any
matching construction time
pair sweep time
percentage of coordinates whose labels changed
percentage of pairs with at least one changed label
objective before and after each pair sweep
```

Do not compute the full objective after every pair in production. That is only for small debug tests.

---

# 18. Acceptance criteria

The implementation is complete only when all conditions below hold:

1. No new calibration artifact is required.
2. GuideQuant Hessian collection is untouched.
3. Initialization artifacts are untouched.
4. Exact GuideQuant codebook update is untouched.
5. Saved labels and LUTs retain their existing format.
6. Pair assignment uses the full original damped Hessian.
7. Pair assignment does not use row-normalized Hessian energy.
8. Every pair update is an exact conditional minimization.
9. Every full pair sweep is non-increasing under the original objective.
10. Non-pair Hessian interactions are included through the external fields.
11. Output rows and Hessian groups are GPU-vectorized.
12. Lazy panel propagation is equivalent to naive exact propagation.
13. The optimized backend matches brute force.
14. The original scalar-CD path remains available for ablation.
15. No gradient term, \(r\), learning rate, penalty, or new optimization hyperparameter is introduced.

---

# 19. One-sentence instruction to Codex

> Keep the complete GuideQuant/LNQ pipeline, artifacts, Hessian collection, initialization, exact codebook update, outer alternating schedule, logging, and serialization unchanged; replace only the scalar `update_P` assignment solver with curvature-matched exact \(B=2\) pair coordinate descent using the full damped symmetric Hessian, a deterministic Hessian-based pair order, exact vectorized pair minimization, and GuideQuant-style lazy GPU panel propagation, while retaining the original scalar CD and a brute-force pair oracle for verification.
