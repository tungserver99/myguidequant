# HNLL GuideQuant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GuideQuant/LNQ variant that changes only the Hessian source to Group-Trace Positive NLL Hessian estimated by Hutchinson HVP.

**Architecture:** Keep the existing initialization, layerwise quantization, assignment solver, codebook update, damping, stopping, and packing flow unchanged. Add a new `hessian_source` switch in layerwise quantization that either runs the current saliency-weighted Hessian path or a new NLL-HVP Hessian accumulator that writes the same `{module_name: H}` cache format consumed by `seed`.

**Tech Stack:** Python, PyTorch autograd HVP, existing Hugging Face causal LM forward, existing bash runner scripts.

## Global Constraints

- Only replace the `H` construction; do not change initialization, LNQ optimizer flow, assignment solver, codebook update, damping, or packing.
- The new HNLL path must not use squared output gradients, teacher logits, KL, pseudo labels, or sampled Fisher.
- Logs and cache/result names must identify the HNLL model/result separately from existing GuidedQuant runs.
- End with a dedicated shell runner for the original CD GuideQuant flow.

---

### Task 1: Tests

**Files:**
- Create: `tests/test_hnll_hessian.py`
- Modify: `tests/test_sqllm_runner_script.py`

**Interfaces:**
- Produces tests for `_rademacher_like`, `reduce_channels_by_group_mean`, `build_group_hessians_from_curvature`, CLI text, and runner script text.

- [ ] Write failing tests for group reduction, weighted GEMM Hessian construction, CLI flags, and HNLL runner script.
- [ ] Run focused tests and confirm they fail because the HNLL API/script does not exist yet.

### Task 2: HNLL Hessian Backend

**Files:**
- Modify: `any_precision/quantization/activations.py`

**Interfaces:**
- Produce `accumulate_nll_hvp_hessians(analyzer, data, output_folder, num_groups, num_probes, random_state=None) -> bool`.
- Produce helper functions `_rademacher_like`, `reduce_channels_by_group_mean`, and `build_group_hessians_from_curvature`.

- [ ] Implement exact autograd HVP with respect to each target linear module output.
- [ ] Average probe samples first, then average by group, then clamp nonnegative.
- [ ] Build `H_k = X.T @ diag(s_k) @ X` using weighted GEMM and symmetrize.
- [ ] Save layer files in the existing `{module_name: tensor}` cache format.

### Task 3: CLI And Logging

**Files:**
- Modify: `any_precision/quantization/layerwise_main.py`
- Modify: `layerwise_nuq.py`

**Interfaces:**
- Add `--hessian_source {saliency,nll_hvp}` and `--nll_hvp_probes`; the HNLL runner must explicitly use `--assignment_solver cd`.
- Keep default behavior unchanged.

- [ ] Route `hessian_source=saliency` to the existing path.
- [ ] Route `hessian_source=nll_hvp` to the new accumulator and cache suffix.
- [ ] Log Hessian source, probe count, and HNLL cache/model output paths.

### Task 4: Runner Script

**Files:**
- Create: `scripts/run_lnq_guidedquant_hnll_cd_c4_eval_ppl.sh`

**Interfaces:**
- Mirror the original CD C4 runner flow, but use HNLL cache/result suffix and pass the new HNLL flags to `layerwise_nuq.py`.

- [ ] Create the dedicated script.
- [ ] Add script text assertions to tests.

### Task 5: Verification

**Files:**
- No new files.

- [ ] Run focused pytest tests.
- [ ] Run Python compile check for changed Python files.
- [ ] Check git diff to ensure no unrelated user files were modified.


