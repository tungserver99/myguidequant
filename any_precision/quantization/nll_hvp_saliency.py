import gc
import logging
import math
import os
import time
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm


TargetKey = Tuple[int, str]

def _sdpa_backend_context(backend: str):
    if backend == "auto" or not torch.cuda.is_available():
        return nullcontext()

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        backend_map = {
            "math": [SDPBackend.MATH],
            "flash": [SDPBackend.FLASH_ATTENTION],
            "efficient": [SDPBackend.EFFICIENT_ATTENTION],
            "cudnn": [SDPBackend.CUDNN_ATTENTION],
        }
        if backend not in backend_map:
            raise ValueError(f"Unsupported sdpa backend {backend!r}")
        return sdpa_kernel(backend_map[backend])
    except ImportError:
        pass

    if backend == "math":
        try:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
            )
        except Exception:
            return nullcontext()

    return nullcontext()


def _saved_tensors_context(saved_tensors_device: str):
    if saved_tensors_device == "gpu" or not torch.cuda.is_available():
        return nullcontext()
    if saved_tensors_device != "cpu":
        raise ValueError(
            f"Unsupported saved_tensors_device={saved_tensors_device!r}; expected 'cpu' or 'gpu'"
        )
    return torch.autograd.graph.save_on_cpu(pin_memory=True)

def _prepare_causal_nll_inputs(
    tokens: torch.Tensor,
    pad_token_id: Optional[int] = None,
    device: Optional[torch.device] = None,
):
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    if device is not None:
        tokens = tokens.to(device)

    labels = tokens.clone()
    attention_mask = None
    if pad_token_id is not None:
        pad_mask = labels.eq(pad_token_id)
        if pad_mask.any():
            labels = labels.masked_fill(pad_mask, -100)
            attention_mask = (~pad_mask).long()

    valid_tokens = int(labels[..., 1:].ne(-100).sum().item())
    return tokens, labels, attention_mask, valid_tokens


def reduce_output_channels_to_groups(diag_sample: torch.Tensor, num_groups: int) -> torch.Tensor:
    if diag_sample.ndim < 1:
        raise ValueError("diag_sample must have an output-channel dimension")
    if num_groups <= 0:
        raise ValueError(f"num_groups must be positive, got {num_groups}")
    out_features = diag_sample.shape[-1]
    if out_features % num_groups != 0:
        raise ValueError(
            f"output channels {out_features} must be divisible by num_groups {num_groups}"
        )
    group_size = out_features // num_groups
    return diag_sample.reshape(*diag_sample.shape[:-1], num_groups, group_size).mean(dim=-1)


def _normalise_token_sequence(input_tokens) -> List[torch.Tensor]:
    if isinstance(input_tokens, torch.Tensor):
        if input_tokens.dim() == 1:
            return [input_tokens]
        return [row for row in input_tokens]
    return list(input_tokens)


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _make_rademacher_like(tensor: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(int(seed))
    signs = torch.empty(tensor.shape, dtype=torch.float32, device=tensor.device)
    signs.bernoulli_(0.5, generator=generator)
    signs.mul_(2.0).sub_(1.0)
    return signs.to(dtype=tensor.dtype)


def _derive_probe_seed(base_seed: int, batch_idx: int, probe_idx: int, layer_idx: int, module_pos: int) -> int:
    return (
        int(base_seed) * 1_000_003
        + int(batch_idx) * 10_007
        + int(probe_idx) * 1_009
        + int(layer_idx) * 101
        + int(module_pos)
    ) & 0x7FFFFFFF


def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _metadata_matches(
    metadata: Dict,
    num_examples: int,
    seq_len: int,
    num_groups: int,
    num_probes: Optional[int] = None,
    base_seed: Optional[int] = None,
    layer_chunk_size: Optional[int] = None,
) -> bool:
    required = {
        "curvature_mode": "nll_global_hvp",
        "loss_reduction": "sum",
        "num_examples": int(num_examples),
        "seq_len": int(seq_len),
        "num_groups": int(num_groups),
    }
    if num_probes is not None:
        required["num_probes"] = int(num_probes)
    if base_seed is not None:
        required["seed"] = int(base_seed)
    if layer_chunk_size is not None:
        required["layer_chunk_size"] = int(layer_chunk_size)

    for key, expected in required.items():
        if metadata.get(key) != expected:
            return False
    return True


def complete_saliency_cache_exists(
    analyzer,
    saliency_path: str,
    num_examples: int,
    seq_len: int,
    num_groups: int,
    num_probes: Optional[int] = None,
    base_seed: Optional[int] = None,
    layer_chunk_size: Optional[int] = None,
) -> bool:
    if not saliency_path or not os.path.isdir(saliency_path):
        return False

    metadata_path = os.path.join(saliency_path, "metadata.pt")
    if not os.path.isfile(metadata_path):
        return False
    try:
        metadata = _safe_torch_load(metadata_path)
    except Exception:
        return False
    if not isinstance(metadata, dict) or not _metadata_matches(
        metadata,
        num_examples,
        seq_len,
        num_groups,
        num_probes=num_probes,
        base_seed=base_seed,
        layer_chunk_size=layer_chunk_size,
    ):
        return False

    layers = analyzer.get_layers()
    for layer_idx, layer in enumerate(layers):
        file_path = os.path.join(saliency_path, f"l{layer_idx}.pt")
        if not os.path.isfile(file_path):
            return False
        try:
            layer_saliencies = _safe_torch_load(file_path)
        except Exception:
            return False
        if not isinstance(layer_saliencies, dict):
            return False

        expected_modules = analyzer.get_modules(layer)
        for module_name in expected_modules:
            if module_name not in layer_saliencies:
                return False
            tensor = layer_saliencies[module_name]
            if not torch.is_tensor(tensor):
                return False
            if tuple(tensor.shape) != (num_examples, seq_len, num_groups):
                return False
            tensor_f = tensor.float()
            if not torch.isfinite(tensor_f).all():
                return False
            if (tensor_f < 0).any():
                return False
    return True


def _new_raw_stats() -> Dict[str, float]:
    return {
        "raw_sum": 0.0,
        "raw_sumsq": 0.0,
        "raw_min": math.inf,
        "raw_max": -math.inf,
        "raw_negative_count": 0,
        "raw_count": 0,
    }


def _update_raw_stats(stats: Dict[str, float], values: torch.Tensor) -> None:
    values = values.detach().float()
    stats["raw_sum"] += float(values.sum().item())
    stats["raw_sumsq"] += float(values.square().sum().item())
    stats["raw_min"] = min(stats["raw_min"], float(values.min().item()))
    stats["raw_max"] = max(stats["raw_max"], float(values.max().item()))
    stats["raw_negative_count"] += int(values.lt(0).sum().item())
    stats["raw_count"] += int(values.numel())


def _finalize_diagnostics(raw_stats: Dict[str, float], grouped_before_clamp: torch.Tensor, grouped_after_clamp: torch.Tensor) -> Dict[str, float]:
    raw_count = max(int(raw_stats["raw_count"]), 1)
    raw_mean = raw_stats["raw_sum"] / raw_count
    raw_var = max(raw_stats["raw_sumsq"] / raw_count - raw_mean * raw_mean, 0.0)
    grouped = grouped_before_clamp.detach().float()
    clamped = grouped_after_clamp.detach().float()
    positive = clamped[clamped > 0]
    return {
        "raw_mean": raw_mean,
        "raw_std": math.sqrt(raw_var),
        "raw_min": raw_stats["raw_min"],
        "raw_max": raw_stats["raw_max"],
        "fraction_raw_negative": raw_stats["raw_negative_count"] / raw_count,
        "grouped_mean": float(grouped.mean().item()),
        "grouped_std": float(grouped.std(unbiased=False).item()),
        "grouped_max": float(grouped.max().item()),
        "fraction_clamped_zero": float(clamped.eq(0).sum().item()) / max(clamped.numel(), 1),
        "mean_positive_curvature": float(positive.mean().item()) if positive.numel() else 0.0,
    }


def estimate_grouped_diag_from_hvp(
    loss_nll: torch.Tensor,
    z_list: List[torch.Tensor],
    metadata: List[TargetKey],
    num_groups: int,
    num_probes: Optional[int] = None,
    base_seed: int = 0,
    batch_idx: int = 0,
    probes_by_probe: Optional[List[List[torch.Tensor]]] = None,
    grad_fn=None,
):
    if grad_fn is None:
        grad_fn = torch.autograd.grad
    if probes_by_probe is None:
        if num_probes is None:
            raise ValueError("num_probes is required when probes_by_probe is not provided")
        probe_count = int(num_probes)
    else:
        probe_count = len(probes_by_probe)
    if probe_count < 1:
        raise ValueError(f"num_probes must be >= 1, got {probe_count}")
    if len(z_list) != len(metadata):
        raise ValueError("z_list and metadata must have the same length")

    first_reverse_start = time.perf_counter()
    grads_z = grad_fn(
        outputs=loss_nll,
        inputs=z_list,
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )
    first_reverse_time = time.perf_counter() - first_reverse_start

    accumulators = [
        torch.zeros(z.shape[0], z.shape[1], num_groups, dtype=torch.float32, device=z.device)
        for z in z_list
    ]
    raw_stats = [_new_raw_stats() for _ in z_list]
    second_order_calls = 0
    hvp_time = 0.0
    group_reduce_time = 0.0

    for probe_idx in range(probe_count):
        if probes_by_probe is None:
            probes = [
                _make_rademacher_like(
                    z,
                    _derive_probe_seed(base_seed, batch_idx, probe_idx, layer_idx, module_pos),
                )
                for module_pos, ((layer_idx, _module_name), z) in enumerate(zip(metadata, z_list))
            ]
        else:
            probes = [probe.to(device=z.device, dtype=z.dtype) for probe, z in zip(probes_by_probe[probe_idx], z_list)]

        hvp_start = time.perf_counter()
        hvps = grad_fn(
            outputs=grads_z,
            inputs=z_list,
            grad_outputs=probes,
            create_graph=False,
            retain_graph=(probe_idx + 1 < probe_count),
            allow_unused=False,
        )
        hvp_time += time.perf_counter() - hvp_start
        second_order_calls += 1

        reduce_start = time.perf_counter()
        for idx, (accumulator, probe, hvp) in enumerate(zip(accumulators, probes, hvps)):
            diag_sample = (probe * hvp).float()
            _update_raw_stats(raw_stats[idx], diag_sample)
            group_sample = reduce_output_channels_to_groups(diag_sample, num_groups)
            accumulator.add_(group_sample)
        group_reduce_time += time.perf_counter() - reduce_start

    grouped = []
    diagnostics = []
    for accumulator, stats in zip(accumulators, raw_stats):
        accumulator.div_(float(probe_count))
        before_clamp = accumulator.clone()
        accumulator.clamp_min_(0.0)
        grouped.append(accumulator)
        diag = _finalize_diagnostics(stats, before_clamp, accumulator)
        diag["first_reverse_time"] = first_reverse_time
        diag["hvp_time"] = hvp_time
        diag["group_reduce_time"] = group_reduce_time
        diagnostics.append(diag)

    return grouped, diagnostics, second_order_calls



def _call_model_for_logits(model, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]):
    if attention_mask is None:
        return model(input_ids=input_ids)
    return model(input_ids=input_ids, attention_mask=attention_mask)


def _capture_layer_input(model, layer: torch.nn.Module, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]):
    captured = {}

    def hook(_module, args, kwargs, _out):
        if not args or not torch.is_tensor(args[0]):
            raise TypeError("Expected transformer layer input hidden_states as first positional tensor")
        captured["hidden_states"] = args[0].detach()
        captured["kwargs"] = dict(kwargs)
        return None

    handle = layer.register_forward_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            _call_model_for_logits(model, input_ids, attention_mask)
    finally:
        handle.remove()

    if "hidden_states" not in captured:
        raise RuntimeError("Failed to capture transformer-layer input hidden states")
    return captured["hidden_states"], captured["kwargs"]


def _decoder_from_analyzer(analyzer):
    if hasattr(analyzer, "get_model"):
        return analyzer.get_model()
    return analyzer.model


def _run_suffix_from_layer(analyzer, start_layer_idx: int, hidden_states: torch.Tensor, layer_kwargs: Dict):
    layers = analyzer.get_layers()
    decoder = _decoder_from_analyzer(analyzer)
    kwargs = dict(layer_kwargs)
    kwargs.pop("past_key_value", None)

    for layer_idx in range(start_layer_idx, len(layers)):
        layer_out = layers[layer_idx](hidden_states, **kwargs)
        hidden_states = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

    if hasattr(decoder, "norm"):
        hidden_states = decoder.norm(hidden_states)
    if not hasattr(analyzer.model, "lm_head"):
        raise AttributeError("NLL global HVP suffix path expects analyzer.model.lm_head")
    return analyzer.model.lm_head(hidden_states)

def _save_layer_saliencies(output_folder: str, saliency_chunks: List[Dict[str, List[torch.Tensor]]]):
    os.makedirs(output_folder, exist_ok=True)
    for layer_idx, module_chunks in enumerate(saliency_chunks):
        layer_result = {}
        for module_name, chunks in module_chunks.items():
            if not chunks:
                raise RuntimeError(
                    f"No NLL global HVP saliency collected for layer={layer_idx}, module={module_name}"
                )
            layer_result[module_name] = torch.cat(chunks, dim=0)
        torch.save(layer_result, os.path.join(output_folder, f"l{layer_idx}.pt"))


def _save_metadata(
    output_folder: str,
    model_name: str,
    num_examples: int,
    seq_len: int,
    num_groups: int,
    num_probes: int,
    base_seed: int,
    layer_chunk_size: int,
    valid_tokens: int,
):
    metadata = {
        "curvature_mode": "nll_global_hvp",
        "loss_reduction": "sum",
        "model": model_name,
        "num_examples": int(num_examples),
        "seq_len": int(seq_len),
        "num_groups": int(num_groups),
        "num_probes": int(num_probes),
        "seed": int(base_seed),
        "layer_chunk_size": int(layer_chunk_size),
        "valid_tokens": int(valid_tokens),
    }
    torch.save(metadata, os.path.join(output_folder, "metadata.pt"))


def _log_diagnostics(layer_idx: int, module_name: str, diag: Dict[str, float], forward_time: float, save_time: float) -> None:
    logging.info(
        "NLL global HVP raw HVP saliency layer=%s module=%s mean=%.6e std=%.6e min=%.6e max=%.6e fraction(raw < 0)=%.6f",
        layer_idx,
        module_name,
        diag["raw_mean"],
        diag["raw_std"],
        diag["raw_min"],
        diag["raw_max"],
        diag["fraction_raw_negative"],
    )
    logging.info(
        "NLL global HVP grouped saliency layer=%s module=%s mean=%.6e std=%.6e max=%.6e fraction(clamped == 0)=%.6f mean_positive_curvature=%.6e",
        layer_idx,
        module_name,
        diag["grouped_mean"],
        diag["grouped_std"],
        diag["grouped_max"],
        diag["fraction_clamped_zero"],
        diag["mean_positive_curvature"],
    )
    logging.info(
        "NLL global HVP runtime layer=%s module=%s forward time=%.6fs first create-graph reverse time=%.6fs global HVP time=%.6fs group reduction time=%.6fs saliency save time=%.6fs",
        layer_idx,
        module_name,
        forward_time,
        diag["first_reverse_time"],
        diag["hvp_time"],
        diag["group_reduce_time"],
        save_time,
    )
    if torch.cuda.is_available():
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
    else:
        peak_allocated = 0
        peak_reserved = 0
    logging.info(
        "NLL global HVP memory layer=%s module=%s peak allocated memory=%d peak reserved memory=%d",
        layer_idx,
        module_name,
        peak_allocated,
        peak_reserved,
    )


def collect_full_nll_hvp_saliencies(
    analyzer,
    input_tokens,
    output_folder: str,
    num_groups: int,
    num_probes: int = 1,
    base_seed: int = 0,
    layer_chunk_size: int = 1,
    overwrite: bool = False,
    sdpa_backend: str = "math",
    saved_tensors_device: str = "cpu",
) -> None:
    """Collect full-NLL global HVP saliency in the original GuideQuant cache format."""
    if num_probes < 1:
        raise ValueError(f"num_probes must be >= 1, got {num_probes}")
    if layer_chunk_size < 0:
        raise ValueError(f"layer_chunk_size must be >= 0, got {layer_chunk_size}")

    token_list = _normalise_token_sequence(input_tokens)
    if not token_list:
        raise ValueError("input_tokens must contain at least one calibration sample")
    seq_len = token_list[0].shape[-1]
    if any(tokens.shape[-1] != seq_len for tokens in token_list):
        raise ValueError("all calibration samples must have the same sequence length")

    layers = analyzer.get_layers()
    num_layers = len(layers)
    if not overwrite and complete_saliency_cache_exists(
        analyzer,
        output_folder,
        len(token_list),
        seq_len,
        num_groups,
        num_probes=num_probes,
        base_seed=base_seed,
        layer_chunk_size=layer_chunk_size,
    ):
        logging.info(f"Cached NLL global HVP saliency found in {output_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)
    model = analyzer.model
    model.eval()

    original_requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if torch.cuda.is_available() and _model_device(model).type != "cuda":
        model.cuda()
    device = _model_device(model)

    old_use_cache = getattr(getattr(model, "config", None), "use_cache", None)
    if old_use_cache is not None:
        model.config.use_cache = False

    pad_token_id = getattr(getattr(analyzer, "tokenizer", None), "pad_token_id", None)
    total_valid_tokens = sum(
        _prepare_causal_nll_inputs(tokens, pad_token_id=pad_token_id, device=None)[3]
        for tokens in token_list
    )
    saliency_chunks: List[Dict[str, List[torch.Tensor]]] = [
        {module_name: [] for module_name in analyzer.get_modules(layer).keys()}
        for layer in layers
    ]

    try:
        with _sdpa_backend_context(sdpa_backend), _saved_tensors_context(saved_tensors_device):
            for layer_idx, layer in enumerate(layers):
                modules = analyzer.get_modules(layer)
                logging.info(
                    "Collecting NLL global HVP saliency for cut-prefix layer %s",
                    layer_idx,
                )

                for batch_idx, tokens in enumerate(tqdm(token_list, desc="NLL global HVP saliency")):
                    input_ids, labels, attention_mask, _valid_tokens = _prepare_causal_nll_inputs(
                        tokens,
                        pad_token_id=pad_token_id,
                        device=device,
                    )
                    prefix_hidden, layer_kwargs = _capture_layer_input(
                        model,
                        layer,
                        input_ids,
                        attention_mask,
                    )
                    hidden_states = prefix_hidden.detach().requires_grad_(True)
                    captured: Dict[TargetKey, torch.Tensor] = {}
                    hooks = []

                    def make_hook(captured_layer_idx: int, module_name: str):
                        key = (captured_layer_idx, module_name)

                        def hook(_module, _inp, out):
                            if not torch.is_tensor(out):
                                raise TypeError(
                                    f"NLL global HVP expects tensor output for layer {captured_layer_idx} {module_name}, got {type(out)}"
                                )
                            captured[key] = out
                            return None

                        return hook

                    for module_name, module in modules.items():
                        hooks.append(module.register_forward_hook(make_hook(layer_idx, module_name)))

                    try:
                        model.zero_grad(set_to_none=True)
                        forward_start = time.perf_counter()
                        logits = _run_suffix_from_layer(
                            analyzer,
                            layer_idx,
                            hidden_states,
                            layer_kwargs,
                        )
                        shift_logits = logits[:, :-1, :].float()
                        shift_labels = labels[:, 1:]
                        loss_nll = F.cross_entropy(
                            shift_logits.reshape(-1, shift_logits.shape[-1]),
                            shift_labels.reshape(-1),
                            ignore_index=-100,
                            reduction="sum",
                        )
                        forward_time = time.perf_counter() - forward_start

                        metadata: List[TargetKey] = []
                        z_list: List[torch.Tensor] = []
                        for module_name in modules:
                            key = (layer_idx, module_name)
                            if key not in captured:
                                raise RuntimeError(
                                    f"Did not capture target output for layer={layer_idx}, module={module_name}"
                                )
                            z = captured[key]
                            if z.shape[-1] % num_groups != 0:
                                raise ValueError(
                                    f"layer={layer_idx}, module={module_name} output channels {z.shape[-1]} "
                                    f"must be divisible by num_groups {num_groups}"
                                )
                            metadata.append(key)
                            z_list.append(z)

                        accumulators, diagnostics, _second_order_calls = estimate_grouped_diag_from_hvp(
                            loss_nll=loss_nll,
                            z_list=z_list,
                            metadata=metadata,
                            num_groups=num_groups,
                            num_probes=num_probes,
                            base_seed=base_seed,
                            batch_idx=batch_idx,
                        )

                        save_start = time.perf_counter()
                        for accumulator, _diag, (_layer_idx, module_name) in zip(accumulators, diagnostics, metadata):
                            saliency_chunks[layer_idx][module_name].append(
                                accumulator.detach().to(torch.bfloat16).cpu()
                            )
                        save_time = time.perf_counter() - save_start
                        for diag, (_layer_idx, module_name) in zip(diagnostics, metadata):
                            _log_diagnostics(layer_idx, module_name, diag, forward_time, save_time)

                    finally:
                        for hook in hooks:
                            hook.remove()
                        model.zero_grad(set_to_none=True)
                        del captured, hidden_states, prefix_hidden
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

        _save_layer_saliencies(output_folder, saliency_chunks)
        model_name = getattr(model, "name_or_path", model.__class__.__name__)
        _save_metadata(
            output_folder,
            model_name,
            len(token_list),
            seq_len,
            num_groups,
            num_probes,
            base_seed,
            layer_chunk_size,
            total_valid_tokens,
        )
    finally:
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache
        for parameter, requires_grad in zip(model.parameters(), original_requires_grad):
            parameter.requires_grad_(requires_grad)

def log_cached_hessian_diagnostics(analyzer, hessians_path: str) -> None:
    layers = analyzer.get_layers()
    for layer_idx, layer in enumerate(layers):
        file_path = os.path.join(hessians_path, f"l{layer_idx}.pt")
        if not os.path.isfile(file_path):
            logging.warning("NLL global HVP Hessian diagnostics skipped missing file: %s", file_path)
            continue
        layer_hessians = _safe_torch_load(file_path)
        if not isinstance(layer_hessians, dict):
            logging.warning("NLL global HVP Hessian diagnostics skipped non-dict file: %s", file_path)
            continue
        for module_name in analyzer.get_modules(layer):
            if module_name not in layer_hessians:
                logging.warning(
                    "NLL global HVP Hessian diagnostics missing module layer=%s module=%s",
                    layer_idx,
                    module_name,
                )
                continue
            hessian = layer_hessians[module_name]
            if not torch.is_tensor(hessian) or hessian.ndim != 3:
                logging.warning(
                    "NLL global HVP Hessian diagnostics expected [D,D,G] tensor for layer=%s module=%s",
                    layer_idx,
                    module_name,
                )
                continue
            for group_idx in range(hessian.shape[-1]):
                h_group = hessian[:, :, group_idx].double()
                diag = torch.diag(h_group)
                trace = float(torch.trace(h_group).item())
                fro_norm = float(torch.linalg.matrix_norm(h_group).item())
                min_diag = float(diag.min().item())
                max_diag = float(diag.max().item())
                symmetry_error = float((h_group - h_group.T).abs().max().item())
                avg_diag = float(diag.mean().item())
                damp = 0.0
                cholesky_damping_factor = 0.0
                while True:
                    try:
                        test_h = h_group.clone()
                        if damp > 0.0:
                            scale = avg_diag if math.isfinite(avg_diag) and avg_diag > 0 else 1.0
                            eye = torch.eye(test_h.shape[0], dtype=test_h.dtype, device=test_h.device)
                            test_h = test_h + eye * scale * damp
                        torch.linalg.cholesky(test_h)
                        cholesky_damping_factor = damp
                        break
                    except Exception:
                        damp = 1e-5 if damp == 0.0 else damp * 10.0
                        if damp > 1.0:
                            cholesky_damping_factor = float("inf")
                            break
                logging.info(
                    "NLL global HVP Hessian diagnostics layer=%s module=%s group=%s trace(H)=%.6e ||H||_F=%.6e min_diag=%.6e max_diag=%.6e symmetry error=%.6e Cholesky damping factor=%.6e",
                    layer_idx,
                    module_name,
                    group_idx,
                    trace,
                    fro_norm,
                    min_diag,
                    max_diag,
                    symmetry_error,
                    cholesky_damping_factor,
                )






