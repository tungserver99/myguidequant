# K-way RBVT for GuideQuant/LNQ
## Detailed implementation specification for Codex

This document specifies a **minimal, controlled replacement of GuideQuant/LNQ's assignment update (`update_P`)** with a **K-way RBVT beam-search solver** while keeping the rest of the GuideQuant pipeline unchanged.

The goal of the first experiment is deliberately narrow:

> **Keep the exact GuideQuant quadratic objective, Hessian, damping, initialization, codebook update, outer alternating loop, evaluation, and packing pipeline. Replace only the coordinate-descent assignment solver with a multi-hypothesis K-way search over the same discrete assignment space.**

This is intended as an implementation-ready specification. Do not add NLL/KL gradients, low-rank Hessians, extra regularizers, neighbor-only moves, diagonal approximations, or any other new method in this first version.

---

# 1. Scope and non-goals

## 1.1 What changes

Only the assignment step changes:

```text
GuideQuant/LNQ current:
    update_P = cyclic coordinate descent (CD)

Proposed:
    update_P = K-way RBVT beam search
```

The proposed solver searches over the same codebook and the same GuideQuant quadratic objective, but instead of greedily committing to a single assignment trajectory at each coordinate, it keeps the best `B` partial trajectories.

## 1.2 What must remain unchanged

For the first controlled experiment, **do not change**:

- GuideQuant Hessian collection.
- Hessian grouping.
- Hessian damping / positive-definite check.
- Initial labels.
- Initial centroids.
- `objective_function`.
- `update_C`.
- The regularized least-squares codebook refit.
- The outer `P -> C` alternating loop.
- The "keep best and early stop" logic.
- Quantized model packing.
- PPL/downstream evaluation.
- Bitwidth / number of codewords.
- Calibration data.
- Number of outer iterations.
- Existing Hessian cache.

The new method is therefore a pure assignment-solver ablation.

---

# 2. Important GuideQuant code anchors

Official repository:

```text
https://github.com/snu-mllab/GuidedQuant
```

Main file for this change:

```text
any_precision/quantization/layerwise_quantize.py
```

Related entry points:

```text
any_precision/quantization/layerwise_main.py
layerwise_nuq.py
```

## 2.1 `objective_function` — KEEP UNCHANGED

Current GuideQuant constructs:

```python
W_hat = gather(C, labels)
delta_w = W_hat - W
```

and evaluates the quadratic objective using the grouped Hessian:

```python
objective_value = torch.einsum(
    'nij,njk,nik->i',
    delta_w,
    H,
    delta_w
)
total_error = objective_value.mean()
```

This function is the ground-truth surrogate objective for the experiment.

**Do not replace it with a new RBVT loss.**

The new assignment solver must minimize the same objective.

## 2.2 `update_P` — THIS IS THE ONLY ALGORITHMIC COMPONENT TO REPLACE

Current `update_P`:

- uses `labels` and `C`;
- reconstructs `W_hat`;
- normalizes Hessian rows by the diagonal;
- runs `cd_cycles`;
- visits input coordinates sequentially;
- computes the exact conditional minimizer of one coordinate;
- picks the nearest centroid;
- updates future coupling terms.

Keep the original `update_P` in the source tree for baseline comparison.

Add a new function, e.g.:

```python
update_P_rbvt(...)
```

Do not delete the original CD implementation.

## 2.3 `update_C` — KEEP UNCHANGED

Current `update_C`:

- Cholesky-factorizes each grouped Hessian;
- builds the least-squares design matrix from fixed labels;
- uses `torch.linalg.lstsq`;
- includes the existing small regularization (`lambda_reg = 1e-7`);
- processes channel chunks;
- returns updated per-row codebooks.

Do not modify its mathematics or implementation in the first experiment.

This is important: after RBVT returns new labels, GuideQuant's original codebook refit must run exactly as before.

## 2.4 `train_least_squares` — KEEP OUTER LOGIC, SWITCH ASSIGNMENT CALL

Current structure is approximately:

```text
damp H until PD

initialize best objective / labels / C

for iteration:
    if iteration > 0:
        labels = update_P(...)

    log P objective

    C = update_C(...)

    evaluate objective

    if improved:
        save best
    else:
        restore best
        break
```

Preserve this behavior.

Only replace the assignment call when the RBVT solver is selected.

---

# 3. Notation

Consider one output row.

Let:

```text
w ∈ R^d          original floating-point weight row
q ∈ R^d          quantized weight row
H ∈ R^(d×d)      GuideQuant Hessian assigned to this output-row group
C = {c1,...,cK}  codebook for this row
a_i              codeword index assigned to weight i
q_i = c_{a_i}
```

Define quantization error:

\[
e = q - w.
\]

For a `b`-bit quantizer:

\[
K = 2^b.
\]

**Important: `K` is not a new solver hyperparameter.**

It is simply the number of codewords already used by GuideQuant/LNQ:

```python
K = C.shape[-1]
```

Examples:

```text
2-bit -> K = 4
3-bit -> K = 8
4-bit -> K = 16
```

The new algorithmic setting is the **beam width `B`**, not `K`.

---

# 4. Start strictly from the GuideQuant objective

For one output row, GuideQuant minimizes:

\[
J(q)
=
(q-w)^\top H(q-w).
\]

Using:

\[
e=q-w,
\]

this is:

\[
J(e)=e^\top He.
\]

Let the assignment at the beginning of one RBVT sweep be:

\[
q^{(0)}.
\]

Define:

\[
e^{(0)}=q^{(0)}-w.
\]

A new assignment can be represented as a discrete transport:

\[
q=q^{(0)}+\delta.
\]

Therefore:

\[
e=e^{(0)}+\delta.
\]

Substitute into the exact GuideQuant objective:

\[
J(e^{(0)}+\delta)
=
(e^{(0)}+\delta)^\top H(e^{(0)}+\delta).
\]

Expand:

\[
J(e^{(0)}+\delta)
=
(e^{(0)})^\top He^{(0)}
+
2(e^{(0)})^\top H\delta
+
\delta^\top H\delta.
\]

Define the current residual:

\[
r^{(0)} = He^{(0)}.
\]

Then:

\[
\boxed{
J_{\text{new}}
=
J_0
+
2(r^{(0)})^\top\delta
+
\delta^\top H\delta
}
\]

where:

\[
J_0=(e^{(0)})^\top He^{(0)}.
\]

This derivation is exact. No approximation has been introduced.

---

# 5. K-way discrete transport

For weight coordinate `i`, the current quantized value is:

\[
q_i^{(0)}.
\]

Moving this weight to codeword `k` produces:

\[
\boxed{
d_{ik}=c_k-q_i^{(0)}
}
\]

for every:

\[
k\in\{1,\ldots,K\}.
\]

This includes the current codeword:

\[
d_{i,a_i^{(0)}}=0.
\]

Therefore each weight has exactly `K` possible choices:

```text
stay at current codeword
or jump to any other codeword
```

There is:

- no neighbor-only restriction;
- no left/right restriction;
- no sign constraint;
- no no-overshoot constraint;
- no diagonal approximation.

A weight may jump from, for example:

```text
codeword 1 -> codeword 7
```

if that branch is favorable under the full GuideQuant objective.

---

# 6. Exact fully coupled K-way objective

Let:

\[
\delta_i=d_{i,a_i}.
\]

Then:

\[
\Delta J
=
2(r^{(0)})^\top\delta
+
\delta^\top H\delta.
\]

Expanding all coordinates:

\[
\boxed{
\Delta J
=
\sum_i
\left[
2r_i^{(0)}d_{i,a_i}
+
H_{ii}d_{i,a_i}^2
\right]
+
2\sum_{i<j}
H_{ij}d_{i,a_i}d_{j,a_j}
}
\]

The first term is the single-move contribution.

The second term is the interaction among selected moves.

**Do not drop the second term.**

That interaction is exactly what makes the assignment problem combinatorial and is also exactly what CD treats greedily.

The proposed method keeps the full interaction implicitly through the residual state.

---

# 7. Exact single-coordinate transition

Suppose a beam state currently has:

\[
e=q-w,
\]

\[
r=He,
\]

\[
J=e^\top He.
\]

At coordinate `i`, consider changing:

\[
q_i \rightarrow c_k.
\]

Define:

\[
d=c_k-q_i.
\]

The new error is:

\[
e'=e+d\,u_i
\]

where `u_i` is the `i`-th basis vector.

Then:

\[
J'
=
(e+d\,u_i)^\top H(e+d\,u_i).
\]

Expanding:

\[
\boxed{
J'
=
J
+
2d\,r_i
+
d^2H_{ii}
}
\]

Thus the exact candidate score is:

\[
\boxed{
\Delta J_{ik}=2d\,r_i+d^2H_{ii}
}
\]

No Hessian approximation is needed.

The residual updates as:

\[
r'=H(e+d\,u_i)
\]

so:

\[
\boxed{
r'=r+dH_{:,i}
}
\]

or, using the row-vector convention that matches the PyTorch implementation:

\[
\boxed{
R'=R+dH_{i,:}.
}
\]

Because `H` is symmetric, both are mathematically equivalent, but the implementation should use **one convention consistently**.

This spec recommends the row-vector convention:

```python
E = W_hat - W            # [rows, d]
R = E @ H                # [rows, d]
J = (E * R).sum(-1)      # [rows]
```

Then a move at coordinate `i` uses:

```python
score_delta = 2 * delta * R[..., i] + delta.square() * H[i, i]
R_new = R + delta[..., None] * H[i, :][None, ...]
```

---

# 8. Why ordinary GuideQuant CD is the `B = 1` special case

At one coordinate, there are `K` possible codewords.

CD:

1. scores all `K`;
2. chooses the best codeword;
3. immediately discards the other `K-1` possibilities;
4. proceeds to the next coordinate.

This is equivalent to keeping exactly one trajectory.

Therefore:

\[
\boxed{
B=1
\Rightarrow
\text{greedy coordinate descent / Gauss-Seidel search}
}
\]

up to numerical/tie-breaking details.

This is an important implementation sanity check.

The new method generalizes this by keeping multiple trajectories.

---

# 9. K-way RBVT beam search

## 9.1 State

For each row, a beam state contains:

```text
current objective J
current future residual R
implicit assignment history represented by backpointers
```

Do **not** copy the full assignment vector for every child.

## 9.2 Beam expansion

At coordinate `i`, suppose the active beam width is `B_active`.

Each state expands to all `K` codewords:

\[
B_{\text{active}}
\rightarrow
B_{\text{active}}K
\]

children.

For parent beam `b` and codeword `k`:

\[
d_{bik}=c_k-q_i.
\]

Exact score:

\[
\boxed{
J_{bik}
=
J_b
+
2d_{bik}R_{b,i}
+
d_{bik}^2H_{ii}
}
\]

## 9.3 Global prune

Flatten all child scores:

```text
[B_active, K]
    ->
[B_active * K]
```

Then keep the globally best:

\[
\boxed{
B_{\text{next}}
=
\min(B, B_{\text{active}}K)
}
\]

children.

Important:

> **Prune globally across all `B_active × K` children. Do not keep one child per parent.**

Example:

```text
B = 8
K = 8

8 parents × 8 codewords = 64 candidate children

global topk(..., k=8)

keep the 8 lowest-objective children
```

All 8 survivors are allowed to come from the same parent if the scores dictate that.

## 9.4 Residual update only after pruning

This is critical for speed.

Do **not** materialize `B*K` residual tensors.

First:

```text
score B*K scalar candidates
```

Then:

```text
topk -> B survivors
```

Only then:

```text
gather B parent residuals
update B survivor residuals
```

Therefore the expensive residual propagation scales with `B`, not `B*K`.

---

# 10. Even faster residual handling: keep only the future suffix

A full residual update:

```python
R_new = R_parent + delta * H[i, :]
```

is exact but unnecessarily updates coordinates that will never be revisited within the current sweep.

Coordinates are processed sequentially:

```text
0, 1, 2, ..., d-1
```

After coordinate `i` has been finalized for a trajectory, residual entries `< i` are no longer needed in that sweep.

Therefore implement the residual as a shrinking future suffix.

Before processing coordinate `i`:

```text
R_future[..., 0] corresponds to residual coordinate i
R_future[..., 1] corresponds to i+1
...
```

Candidate scoring uses:

```python
r_i = R_future[..., 0]
```

After pruning:

```python
R_parent_tail = gathered_parent_R_future[..., 1:]
```

Then update only future coordinates:

```python
R_next = (
    R_parent_tail
    + selected_delta[..., None]
      * H[i, i+1:][None, None, :]
)
```

At the next iteration:

```text
R_next[..., 0] = exact residual for coordinate i+1
```

Benefits:

- does not update stale past coordinates;
- reduces memory traffic;
- residual tensor shrinks through the sweep;
- preserves the exact full-H interaction.

At the end of a sweep, reconstruct labels and recompute the residual from scratch before the next RBVT cycle.

---

# 11. Full-H interaction is preserved exactly

Suppose the beam selected a previous move:

\[
d_i.
\]

The residual at later coordinate `j` becomes:

\[
r'_j=r_j+d_iH_{ij}.
\]

When testing a later move `d_j`, the transition score contains:

\[
2d_jr'_j
=
2d_jr_j
+
2H_{ij}d_id_j.
\]

Therefore the cross term:

\[
\boxed{
2H_{ij}d_id_j
}
\]

is included exactly.

This is the key correctness property.

The algorithm is **not** a diagonal approximation.

The only approximation relative to exhaustive global search is beam pruning.

---

# 12. Beam width and settings

## 12.1 `K`

`K` is not manually selected.

Always derive it from the codebook:

```python
K = C.shape[-1]
```

or equivalently:

```python
K = 2 ** seed_precision
```

Use the existing GuideQuant bitwidth.

## 12.2 Main solver setting: `B = 8`

First implementation / pilot:

```text
beam_width B = 8
```

Reason:

- small enough for an initial GPU experiment;
- large enough to preserve several alternative assignment trajectories;
- for 3-bit (`K=8`) the beam reaches full width after the first coordinate;
- gives a clean comparison against `B=1`.

Recommended ablation after correctness is established:

```text
B ∈ {1, 4, 8, 16}
```

But the primary pilot setting is:

```text
B = 8
```

## 12.3 Number of sweeps / cycles

For the first controlled comparison, reuse the existing GuideQuant setting:

```text
cd_cycles = 4
```

as:

```text
rbvt_cycles = 4
```

Conceptually:

```text
one RBVT cycle = one complete coordinate sweep
```

This keeps the number of assignment sweeps identical to the baseline.

Do not silently reduce the number of sweeps in the first comparison.

Later, if `B=8` converges in fewer cycles, that can be evaluated separately.

## 12.4 Outer iterations

Keep:

```text
num_iterations = 3
```

or whatever value the original run uses.

Do not change it for the initial solver ablation.

---

# 13. GPU design: parallelize across rows, beams, and codewords

The only unavoidable sequential dimension is:

```text
input coordinate i = 0 ... d-1
```

Everything inside one coordinate should be vectorized.

There must be **no Python loop over output rows** and **no Python loop over beams or codewords**.

## 13.1 GuideQuant grouping

Current tensors:

```text
W:      [output_dim, input_dim]
H:      [num_groups, input_dim, input_dim]
labels: [output_dim, input_dim]
C:      [output_dim, K]
```

Rows are partitioned into Hessian groups:

```python
group_size = output_dim // num_groups
```

Each row in group `g` uses:

```python
H_g = H[g]
```

The safest first implementation is:

```text
loop over Hessian groups
    loop over row chunks
        process all rows in the chunk in parallel
```

The algorithm is row-independent once `H_g` and `C` are fixed, so row batching changes only runtime/memory, not the solution.

## 13.2 Row batching

Use a row batch/chunk.

Safe initial implementation setting:

```text
row_batch_size = 64
```

This is a **performance/memory knob**, not an algorithmic hyperparameter.

It may later be increased to 128/256 if GPU memory allows.

Important:

```text
do NOT iterate one row at a time
```

For each Hessian group, process a tensor:

```text
W_batch:      [R, d]
C_batch:      [R, K]
labels_batch: [R, d]
```

where:

```text
R <= row_batch_size
```

All `R` rows are searched simultaneously.

---

# 14. GPU tensor shapes during beam search

For one row batch:

```text
R_rows = number of rows in batch
A      = active beam width
K      = number of codewords
F      = number of future residual coordinates
```

Maintain:

```text
beam_obj:
    [R_rows, A]

residual_future:
    [R_rows, A, F]
```

At coordinate `i`, candidate deltas are:

```text
delta:
    [R_rows, K]
```

Broadcast to:

```text
[R_rows, A, K]
```

Current residual scalar:

```text
r_i = residual_future[..., 0]
shape [R_rows, A]
```

Candidate scores:

```python
candidate_scores = (
    beam_obj[:, :, None]
    + 2.0
      * r_i[:, :, None]
      * delta[:, None, :]
    + H_ii
      * delta[:, None, :].square()
)
```

Shape:

```text
[R_rows, A, K]
```

Flatten only the beam/codeword dimensions:

```python
flat_scores = candidate_scores.reshape(R_rows, A * K)
```

Then:

```python
next_A = min(beam_width, A * K)

new_obj, flat_idx = torch.topk(
    flat_scores,
    k=next_A,
    dim=-1,
    largest=False,
    sorted=True,
)
```

Recover:

```python
parent_idx = flat_idx // K
code_idx   = flat_idx % K
```

Shapes:

```text
parent_idx: [R_rows, next_A]
code_idx:   [R_rows, next_A]
```

Selected delta:

```python
selected_delta = torch.gather(
    delta,
    dim=1,
    index=code_idx
)
```

Shape:

```text
[R_rows, next_A]
```

---

# 15. Parent residual gather

Do not expand children before `topk`.

After pruning, gather only the selected parent future residuals.

If:

```text
residual_future:
    [R_rows, A, F]
```

and we need the future tail after current coordinate:

```python
parent_tail = residual_future[..., 1:]
```

shape:

```text
[R_rows, A, F-1]
```

Gather along beam dimension using `parent_idx`:

```python
gather_index = parent_idx[..., None].expand(
    -1, -1, parent_tail.shape[-1]
)

selected_parent_tail = torch.gather(
    parent_tail,
    dim=1,
    index=gather_index
)
```

Then:

```python
h_future = H_g[i, i+1:]
```

and:

```python
residual_future = (
    selected_parent_tail
    + selected_delta[..., None]
      * h_future[None, None, :]
)
```

This update is fully parallel over:

```text
row × beam × future-coordinate
```

No beam loop is needed.

---

# 16. Candidate delta optimization

Within one sweep, coordinate `i` has not yet been modified before it is visited.

Therefore its current value is the value at the start of the sweep:

```text
q_start[:, i]
```

for all beam states.

Thus candidate deltas do not depend on beam:

\[
d_{ik}=c_k-q^{\text{start}}_i.
\]

For a row batch:

```python
delta_i = C_batch - q_start[:, i:i+1]
```

shape:

```text
[R_rows, K]
```

No `[R_rows, A, K]` tensor needs to be permanently stored; broadcasting creates the logical beam dimension.

Optionally precompute all deltas for a row batch:

```python
all_delta = C_batch[:, None, :] - q_start[:, :, None]
```

shape:

```text
[R_rows, d, K]
```

but this is not required.

For memory efficiency, computing:

```python
delta_i
```

on demand is likely preferable.

---

# 17. Initial residual and objective for a row batch

Construct start quantized values:

```python
q_start = torch.gather(
    C_batch[:, None, :].expand(-1, d, -1),
    dim=2,
    index=labels_start[..., None].long(),
).squeeze(-1)
```

Then:

```python
E = q_start - W_batch
```

Use row-vector convention:

```python
R0 = E @ H_g
```

Shape:

```text
[R_rows, d]
```

Initial row objective:

```python
J0 = (E * R0).sum(dim=-1)
```

Shape:

```text
[R_rows]
```

Initialize beam:

```python
beam_obj = J0[:, None]
residual_future = R0[:, None, :]
active_beam = 1
```

Do not duplicate the initial residual `B` times.

Let the active beam grow naturally:

```python
next_active = min(B, active_beam * K)
```

This is necessary for bitwidths where `K < B`, e.g. 2-bit with:

```text
K=4, B=8
```

After the first coordinate the active beam is 4, after the second it can become 8.

---

# 18. Backpointers instead of copying assignments

Never construct `B*K` full assignment vectors at every coordinate.

Store only:

```text
parent beam index
chosen codeword index
```

for every surviving beam at each coordinate.

Suggested history tensors for one row batch:

```text
parent_history:
    [R_rows, d, B]

code_history:
    [R_rows, d, B]
```

The active beam may be smaller than `B` for early coordinates; unused entries can be left at zero.

Because:

```text
B = 8
K is normally <= 256
```

history can use a compact integer dtype.

Safe simple implementation:

```python
torch.int16
```

for stored history.

The live indices used by `torch.gather` / `torch.topk` can remain `torch.int64`.

## 18.1 Traceback

After the last coordinate, choose the best final beam for each row:

```python
best_beam = beam_obj.argmin(dim=-1)
```

Then traverse:

```text
i = d-1 ... 0
```

For each row:

```text
label_i = code_history[row, i, current_beam]
current_beam = parent_history[row, i, current_beam]
```

Vectorize traceback across all rows in the row batch.

A Python loop over the `d` coordinates during traceback is acceptable; do not loop over rows.

Write reconstructed labels into the output label tensor.

---

# 19. Multiple RBVT cycles

For:

```text
rbvt_cycles = 4
```

repeat:

1. reconstruct labels from the previous cycle;
2. build `q_start`;
3. recompute:
   ```python
   E = q_start - W_batch
   R0 = E @ H_g
   J0 = (E * R0).sum(-1)
   ```
4. start a fresh beam from width 1;
5. run a full coordinate sweep;
6. reconstruct best labels.

Do not try to carry the previous cycle's beam states into the next cycle.

Each new cycle searches around the new current assignment.

---

# 20. Monotonicity within a sweep

The current codeword is always one of the `K` candidates.

For that candidate:

\[
d=0.
\]

Therefore:

\[
J'=J.
\]

So every state has a no-change child.

Because pruning keeps the lowest objective children:

\[
J_{\text{best,new}}
\le
J_{\text{best,old}}.
\]

Thus the best beam objective cannot increase as coordinates are processed.

This is a useful assertion/debug property.

However, the **existing GuideQuant outer best-state / early-stop logic must still remain unchanged**.

---

# 21. Proposed function signature

Add to:

```text
any_precision/quantization/layerwise_quantize.py
```

something close to:

```python
@torch.no_grad()
def update_P_rbvt(
    W: torch.Tensor,          # [output_dim, input_dim]
    H: torch.Tensor,          # [num_groups, input_dim, input_dim]
    labels: torch.Tensor,     # [output_dim, input_dim]
    C: torch.Tensor,          # [output_dim, K]
    rbvt_cycles: int = 4,
    beam_width: int = 8,
    row_batch_size: int = 64,
    verbose: bool = True,
) -> torch.Tensor:
    ...
```

Do not hard-code `K=8`.

Always:

```python
K = C.shape[-1]
```

---

# 22. High-level implementation pseudocode

```python
@torch.no_grad()
def update_P_rbvt(
    W,
    H,
    labels,
    C,
    rbvt_cycles=4,
    beam_width=8,
    row_batch_size=64,
    verbose=True,
):
    device = W.device

    labels_in = labels.to(device=device, dtype=torch.long)
    C = C.to(device)
    output_dim, d = labels_in.shape

    num_groups = H.shape[0]
    assert output_dim % num_groups == 0
    group_size = output_dim // num_groups

    K = C.shape[-1]
    labels_out = labels_in.clone()

    for cycle in range(rbvt_cycles):

        labels_cycle_start = labels_out.clone()

        for g in range(num_groups):
            row0 = g * group_size
            row1 = (g + 1) * group_size

            H_g = H[g]
            H_diag = torch.diagonal(H_g)

            for st in range(row0, row1, row_batch_size):
                ed = min(st + row_batch_size, row1)

                Wb = W[st:ed]                    # [R, d]
                Cb = C[st:ed]                    # [R, K]
                Lb = labels_cycle_start[st:ed]   # [R, d]
                R_rows = Wb.shape[0]

                # start quantized row
                q_start = gather_codewords(Cb, Lb)  # [R, d]

                E = q_start - Wb
                residual = E @ H_g                 # [R, d]
                beam_obj = (E * residual).sum(-1, keepdim=True)
                                                   # [R, 1]

                residual_future = residual[:, None, :]
                                                   # [R, 1, d]
                active_beam = 1

                parent_history = empty_int_history(...)
                code_history = empty_int_history(...)

                for i in range(d):

                    # K candidate jumps for each row.
                    # Independent of beam inside this sweep.
                    delta = Cb - q_start[:, i:i+1]
                                                   # [R, K]

                    r_i = residual_future[..., 0]  # [R, A]

                    scores = (
                        beam_obj[:, :, None]
                        + 2.0 * r_i[:, :, None] * delta[:, None, :]
                        + H_diag[i] * delta[:, None, :].square()
                    )                              # [R, A, K]

                    flat = scores.reshape(R_rows, active_beam * K)

                    next_beam = min(
                        beam_width,
                        active_beam * K
                    )

                    new_obj, flat_idx = torch.topk(
                        flat,
                        k=next_beam,
                        dim=-1,
                        largest=False,
                        sorted=True,
                    )

                    parent_idx = flat_idx // K     # [R, next_beam]
                    code_idx = flat_idx % K        # [R, next_beam]

                    selected_delta = torch.gather(
                        delta,
                        dim=1,
                        index=code_idx,
                    )                              # [R, next_beam]

                    # save backpointers
                    parent_history[:, i, :next_beam] = parent_idx.to(torch.int16)
                    code_history[:, i, :next_beam] = code_idx.to(torch.int16)

                    # IMPORTANT: prune first.
                    # Only update residual for surviving B states.
                    if i + 1 < d:
                        tail = residual_future[..., 1:]  # [R, A, F-1]

                        gather_idx = parent_idx[..., None].expand(
                            -1, -1, tail.shape[-1]
                        )

                        selected_tail = torch.gather(
                            tail,
                            dim=1,
                            index=gather_idx,
                        )

                        h_future = H_g[i, i+1:]     # [F-1]

                        residual_future = (
                            selected_tail
                            + selected_delta[..., None]
                              * h_future[None, None, :]
                        )
                    else:
                        residual_future = None

                    beam_obj = new_obj
                    active_beam = next_beam

                # traceback best complete trajectory
                best_beam = beam_obj.argmin(dim=-1)
                best_labels = traceback(
                    parent_history,
                    code_history,
                    best_beam,
                    d,
                )

                labels_out[st:ed] = best_labels

        # Optional logging:
        # percentage labels changed in this cycle
        # objective before/after using exact objective_function

    if verbose:
        log change statistics

    return labels_out
```

The pseudocode is intentionally explicit. Codex may refactor helpers, but must preserve the mathematics and tensor semantics.

---

# 23. Important implementation note: `q_start[:, i]`

Within one sweep, when coordinate `i` is reached, no beam state has modified coordinate `i` yet.

All beam differences occur only in previously processed coordinates.

Therefore:

```python
delta = Cb - q_start[:, i:i+1]
```

is correct for every beam.

Do not attempt to maintain a separate `q_i` per beam for the current unprocessed coordinate.

This avoids substantial memory and indexing complexity.

---

# 24. Keep the GuideQuant outer P/C loop unchanged

Current `train_least_squares` should remain structurally the same.

Recommended minimal switch:

```python
if iteration > 0:
    if assignment_solver == "cd":
        labels = update_P(
            W, H, labels, C,
            cd_cycles=cd_cycles
        )
    elif assignment_solver == "rbvt":
        labels = update_P_rbvt(
            W, H, labels, C,
            rbvt_cycles=cd_cycles,
            beam_width=beam_width,
            row_batch_size=row_batch_size,
        )
    else:
        raise ValueError(...)
```

Then leave untouched:

```python
obj_value = objective_function(...)
C = update_C(...)
current_obj_value = objective_function(...)
best-state comparison
rollback
early stop
```

For a quick branch experiment, it is also acceptable to directly replace the `update_P` call, but keeping a solver switch is strongly preferred because it makes A/B testing trivial.

---

# 25. Keep `update_C` exactly as GuideQuant

Do not derive or implement a new codebook update in this experiment.

After RBVT produces labels:

```python
C = update_C(W, H, labels, C, iteration)
```

must execute unchanged.

Important existing behavior to retain:

```text
Cholesky(H)
sub_channel_size = 64
existing sub-input batching
one-hot labels
least-squares design matrix
lambda_reg = 1e-7
torch.linalg.lstsq
NaN check
```

The experiment is specifically:

```text
same C update
different P update
```

---

# 26. Preserve Hessian damping exactly

Current `train_least_squares` checks each grouped Hessian with:

```python
torch.linalg.cholesky(H[i])
```

and progressively adds diagonal damping until positive definite.

Keep this code unchanged.

The RBVT solver uses the already-damped `H` passed to `update_P_rbvt`.

Do not add a separate RBVT damping rule.

---

# 27. Propagate new configuration through the codebase

Recommended new arguments:

```text
assignment_solver = "rbvt"
beam_width = 8
row_batch_size = 64
```

Keep:

```text
cd_cycles = 4
```

for compatibility, or optionally expose an alias:

```text
rbvt_cycles
```

For the first version, the simplest fair behavior is:

```python
rbvt_cycles = cd_cycles
```

## 27.1 `layerwise_quantize.py`

Propagate:

```text
beam_width
row_batch_size
assignment_solver
```

through:

```text
train_least_squares
seed_layer
seed
```

## 27.2 `layerwise_main.py`

Add parameters to:

```python
layerwise_nuq(...)
```

e.g.:

```python
assignment_solver="rbvt",
beam_width=8,
row_batch_size=64,
```

Pass them into:

```python
seed(...)
```

## 27.3 root `layerwise_nuq.py`

Add CLI arguments:

```python
parser.add_argument(
    "--assignment_solver",
    type=str,
    default="rbvt",
    choices=["cd", "rbvt"],
)

parser.add_argument(
    "--beam_width",
    type=int,
    default=8,
)

parser.add_argument(
    "--row_batch_size",
    type=int,
    default=64,
)
```

Keep:

```python
--cd_cycles
```

for the number of sweeps.

Optionally rename only later after the experiment stabilizes.

---

# 28. Cache paths must distinguish CD and RBVT

Current GuideQuant cache path contains something like:

```text
..._iter{num_iterations}_cd{cd_cycles}
```

Do not let RBVT reuse the CD quantized cache.

For RBVT, use a separate tag, e.g.:

```text
..._iter3_rbvtB8_cyc4
```

and optionally include row batch only if desired for debugging:

```text
..._rbvtB8_cyc4
```

`row_batch_size` should not affect results, so it does not need to be part of the canonical experiment identifier.

Suggested logic:

```python
if assignment_solver == "cd":
    solver_tag = f"cd{cd_cycles}"
else:
    solver_tag = f"rbvtB{beam_width}_cyc{cd_cycles}"
```

Use this in:

```text
quantized_cache_path
model_output_path
log file name
```

This prevents accidental reuse of baseline outputs.

---

# 29. Correctness tests before running a full LLM

Codex must implement small deterministic unit tests first.

## 29.1 Exact transition identity

Generate:

```text
small d
random SPD H
random w
random q
```

For every coordinate/codeword move verify:

```python
J_direct = (e_new @ H * e_new).sum()

J_incremental = (
    J
    + 2 * delta * r_i
    + delta**2 * H_ii
)
```

Assert close.

This is the most important mathematical unit test.

## 29.2 Residual update identity

Verify:

```python
R_direct = e_new @ H
R_update = R + delta * H[i, :]
```

Assert close.

Also verify the suffix-only update gives the same values for future coordinates.

## 29.3 Exhaustive search on tiny problems

For:

```text
d = 4 or 5
K = 2 or 3
```

enumerate all:

\[
K^d
\]

assignments.

Use a beam width large enough to avoid pruning.

Verify RBVT returns the exact exhaustive optimum.

## 29.4 `B=1` sanity test

Run:

```text
beam_width = 1
```

and compare against GuideQuant CD on a small deterministic case.

Expected:

- same or numerically equivalent objective;
- usually same labels;
- tie cases may produce different but equal-objective labels.

This test verifies that the new search is a strict generalization of the greedy coordinate search.

## 29.5 Monotonic sweep test

For every row batch, log best beam objective after each coordinate or periodically.

Assert:

```text
best objective never increases
```

within numerical tolerance.

---

# 30. Runtime/memory tests

Before a full model:

1. one random synthetic matrix;
2. one real module from one layer;
3. one full transformer layer;
4. full model.

Measure:

```text
CD update_P time
RBVT B=1 time
RBVT B=4 time
RBVT B=8 time
RBVT B=16 time
peak CUDA memory
```

The initial expected computational pattern is approximately:

```text
CD             ~ O(rows * d^2)
RBVT           ~ O(rows * B * d^2)
```

with much of the work parallelized on GPU.

Candidate scoring adds only roughly:

```text
O(rows * B * K * d)
```

which is usually not the dominant term.

The expensive part is residual propagation.

Therefore:

- prune before residual update;
- update only surviving `B`;
- update only the future suffix;
- parallelize rows;
- do not copy full assignments.

---

# 31. Logging for the first experiment

Keep existing objective logs and add:

```text
assignment_solver
beam_width
rbvt_cycles
row_batch_size
```

For each P update, log:

```text
GuideQuant objective before P
GuideQuant objective after P
percentage of labels changed
RBVT runtime
peak/allocated CUDA memory if convenient
```

Useful extra diagnostics:

```text
mean absolute codeword-index jump
fraction of changed labels with |new_idx-old_idx| > 1
```

These diagnostics are not part of the objective. They only show whether the solver actually uses non-local codeword jumps.

Do not use them for pruning.

---

# 32. First experimental settings

Use the same model/configuration as the existing GuideQuant baseline.

Do not change calibration or Hessian settings.

Solver settings:

```text
assignment_solver = rbvt
beam_width         = 8
rbvt_cycles        = 4     # reuse existing cd_cycles
row_batch_size     = 64    # performance-only initial value
```

Codeword count:

```text
K = C.shape[-1]
```

Do not specify a separate `K`.

For comparison run:

```text
CD baseline:
    original GuideQuant update_P
    cd_cycles = 4

RBVT:
    B = 8
    cycles = 4
```

After correctness is established, optional beam-width ablation:

```text
B = 1, 4, 8, 16
```

---

# 33. Recommended implementation order

## Stage 1 — Preserve baseline

- Keep current `update_P`.
- Add solver switch.
- Confirm original CD output is unchanged.

## Stage 2 — Implement scalar/single-row reference RBVT

Implement a simple readable reference version first.

Use it only for unit tests.

Do not optimize yet.

## Stage 3 — Verify exactness

Pass:

- transition identity;
- residual identity;
- exhaustive tiny problem;
- B=1 test;
- monotonicity test.

## Stage 4 — Implement GPU row-batched version

Vectorize:

```text
rows × beam × codewords
```

Use:

```text
global topk over beam*codeword
```

and suffix residual update.

## Stage 5 — Add backpointers

Avoid copying full assignments.

Verify traceback against the reference implementation.

## Stage 6 — Integrate with `train_least_squares`

Only after solver tests pass.

Keep `update_C` and outer logic unchanged.

## Stage 7 — Run one module

Compare:

```text
objective after P
objective after C
runtime
memory
```

against original CD.

## Stage 8 — Full-model PPL

Only once local objective behavior is verified.

---

# 34. Common implementation mistakes to avoid

## Mistake 1: treating `K` as a beam-search setting

Wrong:

```python
K = 8  # hard-coded solver parameter
```

Correct:

```python
K = C.shape[-1]
```

The solver setting is:

```python
beam_width = 8
```

## Mistake 2: pruning per parent

Wrong:

```text
for each parent:
    keep its best child
```

Correct:

```text
flatten all parent × codeword children
global top-B
```

## Mistake 3: creating `B*K` residual tensors

Wrong and expensive.

Correct:

```text
compute B*K scalar scores
top-B
then update only B survivor residuals
```

## Mistake 4: dropping Hessian cross terms

Do not use:

```text
diagonal H only
```

The residual update with `H[i, i+1:]` is exactly how future cross terms are preserved.

## Mistake 5: neighbor-only codeword candidates

Every weight must try all:

```text
k = 0 ... K-1
```

## Mistake 6: copying full assignment histories

Use backpointers.

## Mistake 7: looping through output rows in Python

Rows must be a batch tensor.

## Mistake 8: modifying `update_C`

Do not change it in this experiment.

## Mistake 9: changing Hessian damping

Do not change it in this experiment.

## Mistake 10: reusing CD cache directories

RBVT results need a distinct cache/output tag.

## Mistake 11: updating `H[:, i]` vs `H[i, :]` inconsistently

Use the row-vector convention consistently:

```python
R = E @ H
R_new = R + delta * H[i, :]
```

Because `H` should be symmetric this may otherwise appear to work, hiding a convention bug.

## Mistake 12: carrying a beam across RBVT cycles

At the start of each new sweep:

```text
take the best labels from previous sweep
recompute q, E, R, J
start beam width at 1 again
```

---

# 35. Mathematical interpretation

The method does not introduce a new loss.

It changes the search strategy for:

\[
\boxed{
\min_{a_1,\ldots,a_d}
(q(a)-w)^\top H(q(a)-w)
}
\]

GuideQuant CD behaves like:

```text
one hypothesis
try K choices
commit immediately
repeat
```

K-way RBVT behaves like:

```text
B hypotheses
each tries K choices
keep global top-B
repeat
```

Thus:

\[
\boxed{
B=1
\rightarrow
\text{greedy CD-like search}
}
\]

and increasing `B` retains more jointly interacting assignment trajectories.

If no pruning were performed, the search would enumerate the full discrete assignment space and return the global optimum.

With finite `B`, the **objective and transition scores remain exact**; only the search-tree pruning is approximate.

---

# 36. Final target architecture

```text
GuideQuant Hessian + damping
            |
            v
initial labels + codebook
            |
            v
+---------------------------+
| K-way RBVT update_P       |
|                           |
| sequential coordinates    |
| GPU parallel:             |
|   rows × beams × K        |
| global top-B prune        |
| exact full-H residual     |
| suffix-only propagation   |
| backpointer traceback     |
+---------------------------+
            |
            v
new labels
            |
            v
+---------------------------+
| original GuideQuant       |
| update_C                  |
| UNCHANGED                 |
+---------------------------+
            |
            v
original GuideQuant
objective / best-state /
early-stop logic
            |
            v
packing + PPL evaluation
```

---

# 37. Minimal acceptance criteria

Do not call the implementation complete until all of the following hold:

1. Original CD mode still runs unchanged.
2. RBVT accepts any existing GuideQuant codebook size `K`.
3. Default pilot beam width is `B=8`.
4. All `K` codewords are considered for every coordinate.
5. Candidate scores use:
   \[
   J' = J + 2dr_i + d^2H_{ii}.
   \]
6. Future residuals are updated with full-H interactions.
7. Pruning is global top-B over `beam × codeword`.
8. Residuals are updated only after pruning.
9. Output rows are processed in parallel batches.
10. Backpointers are used instead of full trajectory copies.
11. `update_C` is unchanged.
12. Hessian damping is unchanged.
13. Outer best/early-stop logic is unchanged.
14. `B=1` passes the CD-like sanity test.
15. Tiny exhaustive search test passes.
16. RBVT cache/output path is distinct from CD.
17. Full-model PPL uses the already-correct evaluation pipeline.

---

# 38. Source files to inspect before editing

Codex should inspect the current repository version before changing anything:

```text
any_precision/quantization/layerwise_quantize.py
    objective_function
    update_P
    update_C
    train_least_squares
    seed_layer
    seed

any_precision/quantization/layerwise_main.py
    layerwise_nuq
    cache path construction
    seed(...) call

layerwise_nuq.py
    argparse definitions
```

Official repository:

```text
https://github.com/snu-mllab/GuidedQuant
```

Paper:

```text
https://openreview.net/pdf?id=ZawsPjlIGu
```

---

# 39. One-sentence implementation instruction for Codex

> **Implement `update_P_rbvt` as a row-batched, GPU-vectorized beam generalization of GuideQuant's exact coordinate assignment search: keep the original full-H quadratic objective, try every existing codeword at each coordinate, globally retain the best `B=8` trajectories using the exact incremental score, propagate only surviving future residuals with full Hessian interactions, reconstruct labels by backpointers, and leave GuideQuant's Hessian, damping, `update_C`, outer alternating loop, and evaluation unchanged.**
