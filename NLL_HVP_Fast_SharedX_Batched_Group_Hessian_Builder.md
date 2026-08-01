# Fast Hessian Builder for NLL-HVP GuideQuant/LNQ

## 0. Scope

This document specifies two implementation optimizations that must be used together:

1. **Batched GEMM across output-channel groups**.
2. **Fusion of projections that share the same input activation \(X\)**.

Target environment:

```text
PyTorch 2.5.1
CUDA 12.4
```

These changes affect only the construction of:

\[
H_g = X^\top \operatorname{Diag}(s_g) X.
\]

They must not change:

- ground-truth NLL;
- global HVP/Hutchinson estimator;
- probe count or seed;
- group definitions;
- probe averaging;
- positive projection;
- calibration normalization;
- damping;
- LNQ initialization;
- coordinate descent;
- exact codebook update.

Therefore, for identical \(X\) and \(s_g\):

\[
\boxed{H_{\text{optimized}} \approx H_{\text{legacy}}}
\]

up to floating-point accumulation order.

---

# 1. Current and optimized data flow

Current:

```text
NLL-HVP
  ↓
group statistics s[module]: [T, G_module]
  ↓
for each module:
    for each group:
        H_g = Xᵀ Diag(s_g) X
  ↓
damping
  ↓
LNQ
```

Optimized:

```text
NLL-HVP                                  unchanged
  ↓
group statistics s[module]              unchanged
  ↓
positive projection                     unchanged
  ↓
group modules by shared semantic X
  ↓
concatenate their group statistics
  ↓
process combined groups in chunks
  ↓
batched weighted GEMM
  ↓
split Hessians back to modules
  ↓
damping + LNQ                           unchanged
```

Both optimizations should live in one component:

```text
SharedXGroupedHessianBuilder
```

---

# 2. Mathematical form

For one linear module:

\[
X\in\mathbb R^{T\times D},
\qquad
S=[s_1,\ldots,s_G]\in\mathbb R^{T\times G},
\]

where \(S\) has already been normalized and positive-projected:

\[
s_g\ge 0.
\]

For each group:

\[
\boxed{
H_g
=
X^\top\operatorname{Diag}(s_g)X
}
\]

Define:

\[
\widetilde X_g
=
\operatorname{Diag}(\sqrt{s_g})X.
\]

Then:

\[
\boxed{
H_g
=
\widetilde X_g^\top\widetilde X_g.
}
\]

Stacking all \(\widetilde X_g\) gives a tensor of shape:

```text
[G, T, D]
```

and all Hessians can be computed by:

```text
[G, D, T] @ [G, T, D] → [G, D, D]
```

using `torch.bmm`.

---

# 3. Optimization 1: batched GEMM over groups

## 3.1 Legacy implementation

```python
def build_hessians_legacy(
    x: torch.Tensor,             # [T, D]
    stats: torch.Tensor,         # [T, G]
) -> torch.Tensor:               # [G, D, D]
    outputs = []

    for group_idx in range(stats.shape[1]):
        s = stats[:, group_idx]                  # [T]
        xw = x * torch.sqrt(s).unsqueeze(-1)     # [T, D]
        h = xw.transpose(0, 1) @ xw              # [D, D]
        outputs.append(h)

    return torch.stack(outputs, dim=0)
```

This creates:

- a Python loop over groups;
- many GEMM launches;
- repeated reads of \(X\).

## 3.2 Batched implementation

```python
sqrt_s = torch.sqrt(stats).transpose(0, 1)  # [G, T]

x_weighted = (
    sqrt_s.unsqueeze(-1)
    * x.unsqueeze(0)
)                                            # [G, T, D]

hessians = torch.bmm(
    x_weighted.transpose(1, 2),              # [G, D, T]
    x_weighted,                              # [G, T, D]
)                                            # [G, D, D]
```

This is mathematically identical to the legacy formula.

## 3.3 Mandatory group chunking

The temporary tensor:

\[
[G,T,D]
\]

may be too large. Process only \(C\) groups at once:

```python
def build_group_hessians_batched(
    x: torch.Tensor,
    stats: torch.Tensor,
    group_chunk_size: int,
) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"x must be [T,D], got {tuple(x.shape)}")
    if stats.ndim != 2:
        raise ValueError(f"stats must be [T,G], got {tuple(stats.shape)}")
    if x.shape[0] != stats.shape[0]:
        raise ValueError("Token dimension mismatch")
    if group_chunk_size <= 0:
        raise ValueError("group_chunk_size must be positive")
    if not torch.isfinite(x).all():
        raise ValueError("x contains non-finite values")
    if not torch.isfinite(stats).all():
        raise ValueError("stats contains non-finite values")
    if (stats < 0).any():
        raise ValueError(
            "Builder received negative stats; "
            "positive projection must happen upstream"
        )

    x = x.float()
    stats = stats.float()

    chunks = []

    for start in range(0, stats.shape[1], group_chunk_size):
        end = min(start + group_chunk_size, stats.shape[1])

        s_chunk = stats[:, start:end]                 # [T, C]
        sqrt_s = torch.sqrt(s_chunk).transpose(0, 1) # [C, T]

        xw = (
            sqrt_s.unsqueeze(-1)
            * x.unsqueeze(0)
        )                                             # [C, T, D]

        h_chunk = torch.bmm(
            xw.transpose(1, 2),
            xw,
        )                                             # [C, D, D]

        chunks.append(h_chunk)

    return torch.cat(chunks, dim=0)
```

`group_chunk_size` changes only execution and memory usage, not the objective.

Recommended initial benchmark values:

```text
1, 2, 4, 8
```

---

# 4. Optimization 2: fuse projections sharing the same \(X\)

## 4.1 Llama mappings

Within one transformer block:

```text
q_proj
k_proj
v_proj
```

share the same attention input \(X_{\mathrm{attn}}\).

Also:

```text
gate_proj
up_proj
```

share the same MLP input \(X_{\mathrm{mlp}}\).

Do not fuse:

```text
o_proj
```

with `q/k/v`, because its input is the post-attention representation.

Do not fuse:

```text
down_proj
```

with `gate/up`, because its input is after SiLU and element-wise multiplication.

## 4.2 Concatenate group statistics

For Q/K/V:

\[
S_q\in\mathbb R^{T\times G_q},
\quad
S_k\in\mathbb R^{T\times G_k},
\quad
S_v\in\mathbb R^{T\times G_v}.
\]

Concatenate:

\[
\boxed{
S_{\mathrm{qkv}}
=
[S_q\;S_k\;S_v]
}
\]

with shape:

\[
T\times(G_q+G_k+G_v).
\]

Run one chunked batched builder using \(X_{\mathrm{attn}}\), then split:

```python
h_q, h_k, h_v = torch.split(
    h_qkv,
    [G_q, G_k, G_v],
    dim=0,
)
```

The same applies to:

\[
S_{\mathrm{gateup}}
=
[S_{\mathrm{gate}}\;S_{\mathrm{up}}].
\]

## 4.3 Unequal group counts are valid

For GQA models:

\[
G_q,\;G_k,\;G_v
\]

may differ. Fusion is still valid because the modules share the same input dimension \(D\).

Never use:

```python
torch.chunk(h_qkv, 3)
```

Use exact split sizes:

```python
torch.split(h_qkv, [G_q, G_k, G_v], dim=0)
```

---

# 5. How to identify shared \(X\)

Do not fuse modules based only on equal tensor shapes.

Wrong:

```python
if x_a.shape == x_b.shape:
    fuse(a, b)
```

Correct: use an architecture-aware semantic identifier.

Example:

```python
shared_x_id["model.layers.0.self_attn.q_proj"] = (
    0, "attention_input"
)
shared_x_id["model.layers.0.self_attn.k_proj"] = (
    0, "attention_input"
)
shared_x_id["model.layers.0.self_attn.v_proj"] = (
    0, "attention_input"
)

shared_x_id["model.layers.0.mlp.gate_proj"] = (
    0, "mlp_input"
)
shared_x_id["model.layers.0.mlp.up_proj"] = (
    0, "mlp_input"
)
```

In debug mode, validate captured activations:

```python
torch.testing.assert_close(x_q, x_k, rtol=0, atol=0)
torch.testing.assert_close(x_q, x_v, rtol=0, atol=0)
torch.testing.assert_close(x_gate, x_up, rtol=0, atol=0)
```

If validation fails, do not fuse that group.

The implementation may store one copy of \(X\) per `shared_x_id` and let all associated modules reference it.

---

# 6. Unified builder interface

## 6.1 Input

```python
x: Tensor[T, D]

module_group_stats: OrderedDict[
    module_name,
    Tensor[T, G_module],
]
```

All statistics must already have undergone:

```text
probe averaging
group reduction
correct normalization
positive projection
```

The builder must not change these values.

## 6.2 Output

```python
dict[
    module_name,
    Tensor[G_module, D, D],
]
```

## 6.3 Reference pseudocode

```python
from collections import OrderedDict
from typing import Dict

def build_shared_x_group_hessians(
    x: torch.Tensor,
    module_group_stats: "OrderedDict[str, torch.Tensor]",
    group_chunk_size: int,
) -> Dict[str, torch.Tensor]:
    if x.ndim != 2:
        raise ValueError(f"x must be [T,D], got {tuple(x.shape)}")

    module_names = list(module_group_stats.keys())
    group_counts = []
    stats_list = []

    for module_name in module_names:
        stats = module_group_stats[module_name]

        if stats.ndim != 2:
            raise ValueError(
                f"{module_name}: stats must be [T,G], "
                f"got {tuple(stats.shape)}"
            )

        if stats.shape[0] != x.shape[0]:
            raise ValueError(
                f"{module_name}: token count mismatch: "
                f"x={x.shape[0]}, stats={stats.shape[0]}"
            )

        if not torch.isfinite(stats).all():
            raise ValueError(
                f"{module_name}: non-finite group statistics"
            )

        if (stats < 0).any():
            raise ValueError(
                f"{module_name}: negative statistics passed "
                "to the Hessian builder"
            )

        stats_list.append(stats.float())
        group_counts.append(stats.shape[1])

    all_stats = torch.cat(stats_list, dim=1)  # [T, G_total]

    all_hessians = build_group_hessians_batched(
        x=x.float(),
        stats=all_stats,
        group_chunk_size=group_chunk_size,
    )                                         # [G_total, D, D]

    split_hessians = torch.split(
        all_hessians,
        group_counts,
        dim=0,
    )

    return {
        module_name: hessian
        for module_name, hessian in zip(
            module_names,
            split_hessians,
        )
    }
```

Use deterministic module ordering, preferably `OrderedDict` or an explicit list.

---

# 7. Calibration micro-batches

For micro-batch \(b\):

\[
X_b,\qquad S_{m,b}.
\]

Build:

\[
H_{m,b}
=
X_b^\top
\operatorname{Diag}(S_{m,b})
X_b.
\]

Then accumulate:

\[
\boxed{
H_m
=
\sum_b H_{m,b}.
}
\]

Do not concatenate all calibration activations across batches.

Recommended flow:

```text
for each calibration micro-batch:
    compute HVP stats
    average probes
    group reduce
    positive-project

    for each shared-X group:
        concatenate module group stats
        batched weighted GEMM in group chunks
        split by module
        accumulate into module Hessian cache
```

Normalization must remain exactly as in the current implementation.

If the loss convention is `sum`, accumulate batch Hessians directly.

If the loss convention is `mean`, preserve weighting by valid-token count. Do not average micro-batches equally when they contain different numbers of valid tokens.

---

# 8. Numerical rules

Use FP32 for:

```text
group statistics passed to builder
sqrt
weighted activations
bmm
Hessian accumulation
```

Do not build Hessians in FP16.

After all calibration batches:

```python
h = 0.5 * (h + h.transpose(-1, -2))
```

Then apply the existing damping:

\[
\widetilde H = H+\varepsilon I.
\]

Do not:

- apply damping to \(S\);
- apply damping once per group chunk;
- alter the current damping value or formula;
- rescale Hessians independently by module/group.

---

# 9. Important implementation pitfalls

## 9.1 Token-order mismatch

\(X\) and \(S\) must have identical flattening order:

```text
batch index
sequence index
valid-token mask
```

Shape equality is insufficient.

## 9.2 Mask mismatch

If \(S\) excludes padding tokens, \(X\) must use the same token indices.

## 9.3 Silent clamping in the builder

Do not call:

```python
stats.clamp_min_(0)
```

inside the optimized builder.

Positive projection belongs upstream. Silent clamping can hide a bug and make old/new comparison invalid.

## 9.4 Wrong shared-X group

Do not include `o_proj` in Q/K/V fusion.

Do not include `down_proj` in gate/up fusion.

## 9.5 Wrong module split

Store exact group counts before concatenation and use them for `torch.split`.

## 9.6 Excessive temporary memory

Never assume the full `[G_total,T,D]` tensor fits.

Always support group chunking.

## 9.7 Non-contiguous tensors

`transpose` and broadcasting can create non-contiguous views. Benchmark whether:

```python
xw = xw.contiguous()
```

helps. Do not add it blindly, because the copy also costs memory bandwidth.

## 9.8 TF32 during equivalence tests

For strict testing:

- use FP64 toy tensors; or
- temporarily disable TF32.

Production may preserve the current GuideQuant TF32 settings.

## 9.9 Cache semantics

If the current cache stores one \(X\) per module, changing to one \(X\) per shared group must not change module-to-activation mapping.

Cache versioning may be needed to avoid loading an incompatible old cache.

---

# 10. Required tests

## 10.1 Batched vs legacy

```python
T, D, G = 32, 16, 5

x = torch.randn(
    T, D, device="cuda", dtype=torch.float64
)
s = torch.rand(
    T, G, device="cuda", dtype=torch.float64
)

h_old = build_hessians_legacy(x, s)
h_new = build_group_hessians_batched(
    x, s, group_chunk_size=3
)

torch.testing.assert_close(
    h_new,
    h_old,
    rtol=1e-10,
    atol=1e-10,
)
```

## 10.2 Chunk-size invariance

Compare chunk sizes:

```text
1
2
G
```

All outputs must match within tolerance.

## 10.3 Shared-X fused vs separate

Compute Q/K/V separately, then fused and split. Compare each module independently.

## 10.4 Unequal group counts

Required GQA test:

```text
G_q = 4
G_k = 2
G_v = 2
```

## 10.5 Real micro-batch regression

Use one real HVP result and cache the exact same \(X\) and \(S\).

Run only the two builders:

```text
legacy
optimized
```

Do not rerun HVP, because different probes would invalidate the comparison.

## 10.6 PSD and symmetry

For small matrices:

```python
eig_min = torch.linalg.eigvalsh(h.double()).min()
```

Only very small floating-point negatives are acceptable.

## 10.7 Cholesky after damping

```python
h_damped = h + damping * eye
torch.linalg.cholesky(h_damped)
```

must succeed exactly as in the legacy path.

## 10.8 End-to-end LNQ regression

With identical cached statistics:

- initial assignments unchanged;
- initial codebook unchanged;
- objective values nearly identical;
- no abnormal PPL change.

Tiny differences at assignment ties are possible due to floating-point order.

---

# 11. Benchmark plan

Measure separately:

```text
HVP time
group-reduction time
legacy Hessian-builder time
batched-only builder time
batched + shared-X builder time
total curvature time
peak allocated GPU memory
peak reserved GPU memory
```

Configurations:

```text
A. legacy
B. batched groups per module
C. batched groups + shared-X fusion
```

Keep fixed:

- calibration set;
- number of batches;
- probes;
- probe seed;
- dtype;
- group count;
- model;
- TF32 settings.

The total speedup may be smaller than the builder speedup if second-order HVP dominates runtime. The benchmark must report both.

---

# 12. Suggested flags

```bash
--nll_hessian_builder legacy
--nll_hessian_builder batched
--nll_hessian_builder batched_shared_x

--nll_hessian_group_chunk_size 4
--nll_hessian_validate_shared_x 1
```

`group_chunk_size` is a compute/memory setting, not a quantization hyperparameter.

Keep the legacy implementation until all regression tests pass.

---

# 13. Safe implementation order

1. Extract the current per-group code into:
   ```python
   build_hessians_legacy(...)
   ```

2. Implement:
   ```python
   build_group_hessians_batched(...)
   ```

3. Validate batched vs legacy on toy and real cached statistics.

4. Add architecture-aware shared-\(X\) mappings.

5. Implement:
   ```python
   build_shared_x_group_hessians(...)
   ```

6. Validate fused vs separate module outputs.

7. Enable both optimizations in one production path:
   ```text
   batched_shared_x
   ```

8. Keep `legacy` as a debug fallback.

---

# 14. Out of scope

Do not change in this task:

- reverse-over-reverse HVP;
- forward-over-reverse HVP;
- curvature propagation;
- calibration batch count;
- number of HVP probes;
- group definitions;
- positive projection;
- normalization;
- damping;
- LNQ;
- initialization;
- cache mathematics.

This task optimizes only:

\[
\boxed{
H_g
=
X^\top\operatorname{Diag}(s_g)X
}
\]

through GPU batching and reuse of shared \(X\).

---

# 15. Acceptance criteria

The implementation is complete when:

1. It runs on PyTorch 2.5.1 + CUDA 12.4.
2. `legacy`, `batched`, and `batched_shared_x` modes work.
3. Batched output matches legacy output within tolerance.
4. Fused output matches separate projection output.
5. Unequal Q/K/V group counts work.
6. `o_proj` and `down_proj` are not incorrectly fused.
7. Group chunking prevents OOM.
8. Hessians remain finite, symmetric and PSD within numerical tolerance.
9. Cholesky succeeds after the unchanged damping.
10. LNQ behavior remains consistent.
11. Timing and peak-memory logs are included.
12. The optimized builder is measurably faster than the legacy builder.

---

# 16. One-sentence instruction for Codex

> Keep the existing NLL-HVP statistics and LNQ solver unchanged; replace only the Hessian-construction stage with an architecture-aware shared-\(X\) builder that concatenates the already normalized and positive group statistics of projections using the same semantic input activation—`q_proj/k_proj/v_proj` and `gate_proj/up_proj` within each Llama block—processes the combined group axis in configurable chunks, computes every \(H_g=X^\top\operatorname{Diag}(s_g)X\) using FP32 weighted `torch.bmm`, splits results back to the exact original module/group ordering, preserves all existing masking, normalization, calibration accumulation, damping and caching semantics, and validates every optimized output against the legacy per-module/per-group implementation before enabling it by default.
