# K-way RBVT GuidedQuant Design

## Goal

Add a new K-way RBVT assignment solver for GuideQuant/LNQ while preserving the current GuideQuant baseline. The experiment changes only the assignment update used inside the existing least-squares outer loop.

## Scope

- Keep the existing `update_P` coordinate-descent implementation unchanged.
- Add `update_P_rbvt` as a separate solver.
- Keep `objective_function`, `update_C`, Hessian damping, initialization, outer P/C loop behavior, best-state rollback, early stop, packing, and evaluation unchanged.
- Add CLI/config propagation for `assignment_solver`, `beam_width`, and `row_batch_size`.
- Add cache/output/log tags that distinguish CD from RBVT.
- Add a 3-bit Llama-3.2-1B runner script with random state 42.

## Architecture

The main call path remains:

```text
layerwise_nuq.py
  -> any_precision.quantization.layerwise_main.layerwise_nuq
  -> seed
  -> seed_layer
  -> train_least_squares
```

`train_least_squares` will select the assignment update only when `iteration > 0`:

```text
assignment_solver == "cd"   -> update_P(...)
assignment_solver == "rbvt" -> update_P_rbvt(...)
```

All code after assignment update continues to call the original `objective_function` and `update_C`.

## RBVT Solver

`update_P_rbvt` processes Hessian groups and row chunks. For each RBVT cycle it starts from the current labels, reconstructs `q_start`, computes the exact row residual `R = (q_start - W) @ H_g`, and starts the beam at width 1.

For each coordinate it:

- tries every existing codeword, with `K = C.shape[-1]`;
- scores candidates using `J' = J + 2*d*r_i + d^2*H_ii`;
- globally prunes the flattened `beam x K` children to the best `beam_width`;
- updates future residual suffixes only after pruning with `H_g[i, i+1:]`;
- stores parent and codeword backpointers;
- reconstructs best labels by traceback at the end of the sweep.

`rbvt_cycles` is intentionally not a new public loop setting in the first version. RBVT reuses `cd_cycles` so assignment sweep count matches the CD baseline.

## Configuration And Tags

New public arguments:

```text
--assignment_solver {cd,rbvt}
--beam_width 8
--row_batch_size 64
```

Cache and packed-model path solver tags:

```text
cd:   cd{cd_cycles}
rbvt: rbvtB{beam_width}_cyc{cd_cycles}
```

This prevents RBVT outputs from reusing CD caches.

## Testing

Add deterministic CPU/GPU-safe unit tests for the math and tiny search behavior:

- incremental transition identity;
- residual and future-suffix residual identity;
- exhaustive optimum on a tiny problem when beam width is large enough;
- smoke test that `update_P_rbvt` accepts normal GuideQuant tensor shapes and returns valid labels.

The tests should avoid loading large models.

## Runner Script

Create a new shell script based on the existing GuidedQuant runner style. It should default to:

```text
MODEL_NAME=meta-llama/Llama-3.2-1B
BITS=3
DATASET=c4
NUM_ITERATIONS=3
CD_CYCLES=4
ASSIGNMENT_SOLVER=rbvt
BEAM_WIDTH=8
ROW_BATCH_SIZE=64
RANDOM_STATE=42
```

The script should avoid obsolete HNLL/HVP flags because the current CLI in this checkout does not expose them.
