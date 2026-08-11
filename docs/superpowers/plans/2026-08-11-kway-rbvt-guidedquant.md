# K-way RBVT GuidedQuant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a K-way RBVT assignment solver to GuideQuant/LNQ without changing the existing CD solver or GuideQuant objective/codebook update pipeline.

**Architecture:** Add `update_P_rbvt` beside the existing `update_P`, then route `train_least_squares` through a solver switch. Propagate solver settings through CLI and layerwise entry points, use distinct cache tags, and add a 3-bit Llama-3.2-1B runner script.

**Tech Stack:** Python, PyTorch, NumPy, pytest, bash.

## Global Constraints

- Keep the existing `update_P` coordinate-descent implementation unchanged.
- Keep `objective_function`, `update_C`, Hessian damping, initialization, outer P/C loop behavior, best-state rollback, early stop, packing, and evaluation unchanged.
- RBVT must use the GuideQuant objective increment `J' = J + 2*d*r_i + d^2*H_ii`.
- RBVT must try all existing codewords, with `K = C.shape[-1]`.
- RBVT must prune globally over flattened `beam x K` children.
- RBVT must update residuals only after pruning, using future suffixes and full Hessian interactions.
- RBVT must use backpointers rather than copying full trajectories.
- Public defaults: `assignment_solver="cd"` to preserve current behavior, `beam_width=8`, `row_batch_size=64`.
- Runner defaults: `MODEL_NAME=meta-llama/Llama-3.2-1B`, `BITS=3`, `RANDOM_STATE=42`, `ASSIGNMENT_SOLVER=rbvt`.

---

## File Structure

- Modify `any_precision/quantization/layerwise_quantize.py`: add RBVT helpers and propagate solver parameters through `train_least_squares`, `seed_layer`, and `seed`.
- Modify `any_precision/quantization/layerwise_main.py`: add public parameters, create solver cache tags, pass settings into `seed`.
- Modify `layerwise_nuq.py`: add CLI args for solver, beam width, and row batch size.
- Create `tests/test_kway_rbvt.py`: deterministic small math and solver tests.
- Create `scripts/run_lnq_guidedquant_rbvt_c4_eval_ppl.sh`: 3-bit Llama-3.2-1B RBVT quantize/eval runner.

---

### Task 1: RBVT Math Tests

**Files:**
- Create: `tests/test_kway_rbvt.py`

**Interfaces:**
- Consumes: `any_precision.quantization.layerwise_quantize.update_P_rbvt`
- Produces: deterministic tests for transition, residual, exhaustive optimum, and valid label output.

- [ ] **Step 1: Write failing tests**

Create tests that import `update_P_rbvt`, construct tiny SPD Hessians, and verify:

```python
def _spd(dim):
    torch.manual_seed(0)
    A = torch.randn(dim, dim)
    return A.T @ A + torch.eye(dim) * 0.25

def test_rbvt_transition_identity():
    H = _spd(4)
    e = torch.randn(4)
    r = e @ H
    J = (e * r).sum()
    i = 2
    d = torch.tensor(0.37)
    e_new = e.clone()
    e_new[i] += d
    direct = (e_new * (e_new @ H)).sum()
    incremental = J + 2 * d * r[i] + d.square() * H[i, i]
    assert torch.allclose(direct, incremental, atol=1e-5, rtol=1e-5)
```

- [ ] **Step 2: Run tests and confirm import failure**

Run: `pytest tests/test_kway_rbvt.py -q`

Expected before implementation: FAIL because `update_P_rbvt` is not defined.

---

### Task 2: RBVT Solver Implementation

**Files:**
- Modify: `any_precision/quantization/layerwise_quantize.py`
- Test: `tests/test_kway_rbvt.py`

**Interfaces:**
- Produces: `update_P_rbvt(W, H, labels, C, rbvt_cycles, beam_width=8, row_batch_size=64, verbose=True) -> torch.Tensor`
- Produces helper: `_gather_codewords(C, labels) -> torch.Tensor`

- [ ] **Step 1: Add `_gather_codewords`**

Implement:

```python
def _gather_codewords(C: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.gather(
        C.unsqueeze(1).expand(-1, labels.shape[1], -1),
        dim=2,
        index=labels.unsqueeze(-1).long(),
    ).squeeze(-1)
```

- [ ] **Step 2: Add `update_P_rbvt`**

Implement row-batched RBVT with:

```python
def update_P_rbvt(W, H, labels, C, rbvt_cycles, beam_width=8, row_batch_size=64, verbose=True):
    ...
```

Use row-vector residual convention `residual = E @ H_g` and suffix update:

```python
residual_future = selected_tail + selected_delta[..., None] * H_g[i, i + 1:][None, None, :]
```

- [ ] **Step 3: Run unit tests**

Run: `pytest tests/test_kway_rbvt.py -q`

Expected: PASS.

---

### Task 3: Solver Switch And Argument Propagation

**Files:**
- Modify: `any_precision/quantization/layerwise_quantize.py`
- Modify: `any_precision/quantization/layerwise_main.py`
- Modify: `layerwise_nuq.py`

**Interfaces:**
- Consumes: `update_P_rbvt(...)`
- Produces CLI args: `--assignment_solver`, `--beam_width`, `--row_batch_size`

- [ ] **Step 1: Update `train_least_squares` signature**

Add:

```python
assignment_solver: str = "cd",
beam_width: int = 8,
row_batch_size: int = 64,
```

Route:

```python
if assignment_solver == "cd":
    labels = update_P(W, H, labels, C, cd_cycles=cd_cycles)
elif assignment_solver == "rbvt":
    labels = update_P_rbvt(W, H, labels, C, rbvt_cycles=cd_cycles, beam_width=beam_width, row_batch_size=row_batch_size)
else:
    raise ValueError(f"Unknown assignment_solver: {assignment_solver}")
```

- [ ] **Step 2: Propagate through `seed_layer` and `seed`**

Add the same parameters to signatures and pass them into `train_least_squares`.

- [ ] **Step 3: Propagate through `layerwise_main.layerwise_nuq`**

Add public parameters and pass them into `seed`.

- [ ] **Step 4: Add solver cache tags**

Use:

```python
solver_tag = f"cd{cd_cycles}" if assignment_solver == "cd" else f"rbvtB{beam_width}_cyc{cd_cycles}"
```

Use `solver_tag` in `quantized_cache_path` and `model_output_path`.

- [ ] **Step 5: Add CLI args**

In `layerwise_nuq.py`, add:

```python
parser.add_argument("--assignment_solver", type=str, default="cd", choices=["cd", "rbvt"])
parser.add_argument("--beam_width", type=int, default=8)
parser.add_argument("--row_batch_size", type=int, default=64)
```

- [ ] **Step 6: Run focused tests**

Run: `pytest tests/test_kway_rbvt.py -q`

Expected: PASS.

---

### Task 4: Runner Script

**Files:**
- Create: `scripts/run_lnq_guidedquant_rbvt_c4_eval_ppl.sh`

**Interfaces:**
- Consumes CLI args from Task 3.
- Produces a user runnable command for Llama-3.2-1B, 3-bit, RBVT, random state 42.

- [ ] **Step 1: Create script**

Base it on the existing runner shape, but do not include HNLL/HVP flags. Use:

```bash
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-1B}"
BITS="${BITS:-3}"
ASSIGNMENT_SOLVER="${ASSIGNMENT_SOLVER:-rbvt}"
BEAM_WIDTH="${BEAM_WIDTH:-8}"
ROW_BATCH_SIZE="${ROW_BATCH_SIZE:-64}"
RANDOM_STATE="${RANDOM_STATE:-42}"
```

- [ ] **Step 2: Use RBVT cache path**

Use packed model path:

```bash
PACKED_MODEL_DIR="${CACHE_DIR}/layerwise_packed/layerwise-${MODEL_BASENAME}-w${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}_g${NUM_GROUPS}_iter${NUM_ITERATIONS}_rbvtB${BEAM_WIDTH}_cyc${CD_CYCLES}"
```

- [ ] **Step 3: Syntax check**

Run: `bash -n scripts/run_lnq_guidedquant_rbvt_c4_eval_ppl.sh`

Expected: PASS.

---

### Task 5: Final Verification

**Files:**
- All modified files.

**Interfaces:**
- Produces final evidence that the implementation imports and focused tests pass.

- [ ] **Step 1: Run focused tests**

Run: `pytest tests/test_kway_rbvt.py -q`

Expected: PASS.

- [ ] **Step 2: Compile Python files**

Run: `python -m compileall -q any_precision layerwise_nuq.py tests/test_kway_rbvt.py`

Expected: PASS.

- [ ] **Step 3: Report runner command**

Provide:

```bash
RANDOM_STATE=42 BITS=3 bash scripts/run_lnq_guidedquant_rbvt_c4_eval_ppl.sh
```
