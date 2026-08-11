# Full-NLL Global HVP GuideQuant Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new full-NLL global HVP saliency producer that writes GuideQuant-compatible saliency artifacts, then reuses the original GuideQuant Hessian replay and LNQ path unchanged.

**Architecture:** The new method lives in a new producer module and saves `l{layer}.pt` saliency files in the existing `{module_name: Tensor[N,T,G]}` format. Existing GuideQuant internals remain black-box consumers; existing default saliency behavior is unchanged.

**Tech Stack:** Python, PyTorch autograd, HuggingFace causal LM outputs, existing GuideQuant analyzer and `accumulate_saliency_weighted_hessians`.

## Global Constraints

- Do not change original GuideQuant logic.
- Do not change `SaliencyEngine.add_batch()` or weighted `X^T Diag(s) X`.
- Do not change LNQ initialization, damping, Cholesky, CD, codebook update, packing, or default behavior.
- Add method-specific cache names, tags, and args for `nll_global_hvp`.
- Full-NLL HVP uses shifted causal NLL with `reduction="sum"`.
- HVP sample is `probe * hvp`, never square or absolute.
- Clamp only after averaging probes and output channels into groups.
- Save saliency cache as `{module_name: Tensor[N, seq_len, num_groups]}` per `l{layer}.pt`.

---

### Task 1: Producer Unit Tests

**Files:**
- Create: `tests/test_nll_global_hvp_saliency.py`
- Create later: `any_precision/quantization/nll_hvp_saliency.py`

**Interfaces:**
- Produces tests for `_prepare_causal_nll_inputs`, `reduce_output_channels_to_groups`, `complete_saliency_cache_exists`, and `collect_full_nll_hvp_saliencies`.

- [ ] **Step 1: Write failing tests**

```python
def test_reduce_output_channels_to_groups_averages_contiguous_output_channels():
    sample = torch.tensor([[[1.0, 3.0, 10.0, 14.0]]])
    grouped = reduce_output_channels_to_groups(sample, num_groups=2)
    torch.testing.assert_close(grouped, torch.tensor([[[2.0, 12.0]]]))
```

- [ ] **Step 2: Run focused tests to verify failure**

Run: `D:\anaconda3\envs\topmost\python.exe -m pytest -q tests/test_nll_global_hvp_saliency.py`
Expected: FAIL because `any_precision.quantization.nll_hvp_saliency` does not exist.

### Task 2: New Full-NLL HVP Saliency Producer

**Files:**
- Create: `any_precision/quantization/nll_hvp_saliency.py`

**Interfaces:**
- `collect_full_nll_hvp_saliencies(analyzer, input_tokens, output_folder, num_groups, num_probes=1, base_seed=0, layer_chunk_size=0, overwrite=False) -> None`
- `complete_saliency_cache_exists(analyzer, saliency_path, num_examples, seq_len, num_groups) -> bool`

- [ ] **Step 1: Implement helper functions**
- [ ] **Step 2: Implement hook-based target output capture**
- [ ] **Step 3: Implement summed shifted NLL and one global HVP per probe/chunk**
- [ ] **Step 4: Save cache files and metadata sidecar**
- [ ] **Step 5: Run focused producer tests**

### Task 3: Minimal CLI/Main Plumbing

**Files:**
- Modify: `layerwise_nuq.py`
- Modify: `any_precision/quantization/layerwise_main.py`

**Interfaces:**
- CLI accepts `--hessian_source nll_global_hvp`, `--curvature_mode nll_global_hvp`, `--nll_hvp_probes`, `--nll_hvp_seed`, `--nll_hvp_layer_chunk_size`, `--overwrite_saliency`, and `--overwrite_hessians`.
- `layerwise_nuq(...)` routes only the new mode to the new producer before calling `accumulate_saliency_weighted_hessians(...)`.

- [ ] **Step 1: Write CLI/static tests**
- [ ] **Step 2: Add parser args**
- [ ] **Step 3: Add main routing and method-specific cache suffixes**
- [ ] **Step 4: Run focused tests**

### Task 4: Llama-2-7B Script

**Files:**
- Create: `scripts/run_lnq_guidedquant_nll_global_hvp_cd_llama2_7b_c4_eval_ppl.sh`

**Interfaces:**
- Script calls `python layerwise_nuq.py` with `--hessian_source nll_global_hvp`.
- Script uses distinct `CACHE_DIR`, `RESULT_SUFFIX`, `HESSIAN_SUFFIX`, and PPL tag.

- [ ] **Step 1: Add script modeled on the FD GroupTrace Llama-2-7B script**
- [ ] **Step 2: Verify script text includes new args and tags**

### Task 5: Verification

**Files:**
- No new production files.

**Interfaces:**
- Focused pytest command.
- Compile command.

- [ ] **Step 1: Run focused tests**
- [ ] **Step 2: Run compileall**
- [ ] **Step 3: Report any unrelated pre-existing test failures separately**
