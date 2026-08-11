"""Smoke test forward-over-reverse HVP support on Llama-2.

This is intentionally a manual GPU smoke test, not a normal pytest test. It
loads the real model, builds one full-length sequence, and checks whether
``torch.func.jvp(torch.func.grad(nll_from_embeds))`` works through the whole
Llama path.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import time

import torch
import torch.nn.functional as F
from torch.func import grad, jvp
from transformers import AutoModelForCausalLM, AutoTokenizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single-sequence Llama-2 JVP/HVP smoke test.",
    )
    parser.add_argument(
        "--model_name",
        default="meta-llama/Llama-2-7b-hf",
        help="Hugging Face model name or local model path.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
        help="Sequence length to test. Keep 2048 for the production-shape smoke.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="Model dtype for loading and the embedding variable.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for the smoke test. The intended path is a single CUDA GPU.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the Rademacher probe.",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=("auto", "eager", "sdpa"),
        default="auto",
        help="Optional Hugging Face attention implementation override.",
    )
    parser.add_argument(
        "--sdpa_backend",
        choices=("auto", "math", "flash", "efficient", "cudnn"),
        default="auto",
        help="Optional PyTorch SDPA backend override during JVP.",
    )
    parser.add_argument(
        "--saved_tensors_device",
        choices=("none", "cpu"),
        default="none",
        help="Optionally offload tensors saved for reverse-mode grad to CPU.",
    )
    parser.add_argument(
        "--loss_chunk_tokens",
        type=int,
        default=32,
        help=(
            "Compute exact FP32 next-token NLL in token chunks. Use 0 to "
            "compute the loss over all shifted logits at once."
        ),
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load model/tokenizer from local Hugging Face cache only.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to Hugging Face loaders.",
    )
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _build_one_full_length_input(tokenizer, seq_len: int) -> torch.Tensor:
    text = (
        "This is a deterministic calibration-shaped smoke test for Llama "
        "forward-over-reverse Hessian vector products. "
    )
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    if not token_ids:
        raise RuntimeError("Tokenizer returned no tokens for the smoke prompt.")

    repeats = (seq_len + len(token_ids) - 1) // len(token_ids)
    token_ids = (token_ids * repeats)[:seq_len]
    return torch.tensor([token_ids], dtype=torch.long)


def _make_probe_like(tensor: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(seed)
    probe = torch.randint(
        low=0,
        high=2,
        size=tensor.shape,
        device=tensor.device,
        generator=generator,
        dtype=torch.int8,
    )
    return probe.to(dtype=tensor.dtype).mul_(2).sub_(1)


def chunked_ground_truth_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_tokens: int = 32,
) -> torch.Tensor:
    """Exact sum next-token NLL, processed in token chunks."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    if chunk_tokens <= 0:
        return F.cross_entropy(
            shift_logits.float().reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )

    total_loss = torch.zeros((), device=logits.device, dtype=torch.float32)
    seq_len = shift_logits.shape[1]
    for start in range(0, seq_len, chunk_tokens):
        end = min(start + chunk_tokens, seq_len)
        logits_chunk = shift_logits[:, start:end, :].float()
        labels_chunk = shift_labels[:, start:end]
        total_loss = total_loss + F.cross_entropy(
            logits_chunk.reshape(-1, logits_chunk.shape[-1]),
            labels_chunk.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )

    return total_loss


@contextlib.contextmanager
def _sdpa_backend_context(name: str):
    if name == "auto":
        yield
        return

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception as error:  # pragma: no cover - depends on torch version.
        raise RuntimeError(
            f"sdpa_backend={name!r} requires torch.nn.attention.sdpa_kernel"
        ) from error

    backend_map = {
        "math": SDPBackend.MATH,
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
    }
    if hasattr(SDPBackend, "CUDNN_ATTENTION"):
        backend_map["cudnn"] = SDPBackend.CUDNN_ATTENTION

    if name not in backend_map:
        raise RuntimeError(f"SDPA backend {name!r} is not available in this torch.")

    with sdpa_kernel(backend_map[name]):
        yield


@contextlib.contextmanager
def _saved_tensors_context(device: str):
    if device == "none":
        yield
        return
    if device != "cpu":
        raise ValueError(f"Unsupported saved_tensors_device: {device}")

    with torch.autograd.graph.save_on_cpu(pin_memory=True):
        yield


def _load_model(args: argparse.Namespace):
    dtype = _torch_dtype(args.dtype)
    load_kwargs = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    model.to(args.device)
    model.eval()
    model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if args.attn_implementation != "auto" and hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(args.attn_implementation)

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return tokenizer, model


def main() -> None:
    args = _parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this smoke test needs a GPU.")
    if args.saved_tensors_device != "none":
        raise RuntimeError(
            "torch.func.grad/jvp do not support saved tensor hooks, so "
            "--saved_tensors_device cpu cannot be used with this JVP smoke test."
        )

    tokenizer, model = _load_model(args)
    device = torch.device(args.device)

    input_ids = _build_one_full_length_input(tokenizer, args.seq_len).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    labels = input_ids.clone()
    assert input_ids.shape == (1, args.seq_len), tuple(input_ids.shape)

    with torch.no_grad():
        test_embeds = model.get_input_embeddings()(input_ids).detach()
    probe = _make_probe_like(test_embeds, args.seed)

    def ground_truth_nll_from_embeds(embeds: torch.Tensor) -> torch.Tensor:
        outputs = model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return chunked_ground_truth_nll(
            logits=outputs.logits,
            labels=labels,
            chunk_tokens=args.loss_chunk_tokens,
        )

    grad_fn = grad(ground_truth_nll_from_embeds)

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    try:
        with _sdpa_backend_context(args.sdpa_backend):
            with _saved_tensors_context(args.saved_tensors_device):
                gradient, hvp = jvp(grad_fn, (test_embeds,), (probe,))
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - start
        peak_gib = None
        if device.type == "cuda":
            peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3

        print("\n========== JVP SMOKE TEST PASSED ==========")
        print("torch:", torch.__version__)
        print("wheel CUDA:", torch.version.cuda)
        print("model:", args.model_name)
        print("model dtype:", next(model.parameters()).dtype)
        print("input shape:", tuple(input_ids.shape))
        print("attention implementation:", args.attn_implementation)
        print("sdpa backend:", args.sdpa_backend)
        print("saved tensors device:", args.saved_tensors_device)
        print("loss chunk tokens:", args.loss_chunk_tokens)
        print("gradient shape:", tuple(gradient.shape))
        print("HVP shape:", tuple(hvp.shape))
        print("gradient finite:", bool(torch.isfinite(gradient).all()))
        print("HVP finite:", bool(torch.isfinite(hvp).all()))
        print("gradient norm:", gradient.float().norm().item())
        print("HVP norm:", hvp.float().norm().item())
        print("elapsed seconds:", elapsed)
        if peak_gib is not None:
            print("peak allocated GiB:", peak_gib)
    except Exception as error:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        print("\n========== JVP SMOKE TEST FAILED ==========")
        print("torch:", torch.__version__)
        print("wheel CUDA:", torch.version.cuda)
        print("model:", args.model_name)
        print("input shape:", tuple(input_ids.shape))
        print("attention implementation:", args.attn_implementation)
        print("sdpa backend:", args.sdpa_backend)
        print("saved tensors device:", args.saved_tensors_device)
        print("loss chunk tokens:", args.loss_chunk_tokens)
        print("error type:", type(error).__name__)
        print("error:", error)
        raise


if __name__ == "__main__":
    main()


