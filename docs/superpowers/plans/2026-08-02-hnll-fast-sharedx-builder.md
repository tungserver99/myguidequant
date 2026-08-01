# HNLL Fast Shared-X Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new HNLL Hessian-builder variant that combines batched group GEMM and shared-X module fusion without changing the existing HNLL path.

**Architecture:** Keep the existing NLL-HVP statistic path intact. Add builder functions selected by new CLI arguments, with the production runner using `batched_shared_x`. Cache/output suffixes include builder mode and group chunk size.

**Tech Stack:** Python, PyTorch, existing GuideQuant/LNQ CLI and bash runners.

## Global Constraints

- Do not change NLL loss, HVP, probes, dtype handling, positive projection, damping, LNQ initialization, coordinate descent, or codebook updates.
- Default old HNLL behavior remains available.
- New runner uses `batched_shared_x` and CD.
- Builder uses FP32 for statistics, sqrt, weighted activations, bmm, and Hessian accumulation.

---

### Task 1: Builder Functions and Tests

**Files:**
- Modify: `any_precision/quantization/activations.py`
- Modify: `tests/test_hnll_hessian.py`

**Interfaces:**
- Produce: `build_group_hessians_legacy(x, stats)`, `build_group_hessians_batched(x, stats, group_chunk_size)`, `build_shared_x_group_hessians(x, module_group_stats, group_chunk_size)`.

- [ ] Add tests for batched vs legacy, chunk invariance, fused vs separate, and unequal group counts.
- [ ] Add builder implementations with validation and exact split sizes.
- [ ] Run `pytest -q tests/test_hnll_hessian.py`.

### Task 2: Integrate Builder Selection

**Files:**
- Modify: `any_precision/quantization/activations.py`
- Modify: `any_precision/quantization/layerwise_main.py`
- Modify: `layerwise_nuq.py`

**Interfaces:**
- Consume: builder functions from Task 1.
- Produce CLI args `--nll_hessian_builder`, `--nll_hessian_group_chunk_size`, `--nll_hessian_validate_shared_x`.

- [ ] Add builder args to accumulation, layerwise entrypoint, and CLI.
- [ ] Use old per-module build for legacy, per-module chunked build for batched, and semantic shared-X fusion for batched_shared_x.
- [ ] Include builder settings in cache suffix.
- [ ] Run HNLL and runner tests.

### Task 3: Dedicated Runner

**Files:**
- Create: `scripts/run_lnq_guidedquant_hnll_fast_sharedx_cd_c4_eval_ppl.sh`
- Modify: `tests/test_sqllm_runner_script.py`

**Interfaces:**
- Runner sets `--hessian_source nll_hvp`, `--assignment_solver cd`, `--nll_hessian_builder batched_shared_x`.

- [ ] Create runner with separate cache/result tag.
- [ ] Add script tests for builder args, suffix, CD, and no pair/triton.
- [ ] Run `pytest -q tests/test_sqllm_runner_script.py tests/test_hnll_hessian.py`.