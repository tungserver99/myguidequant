from pathlib import Path
import importlib.util
import sys
import types

import torch


def _load_nll_hvp_saliency():
    repo_root = Path(__file__).resolve().parents[1]
    any_precision_pkg = types.ModuleType("any_precision")
    any_precision_pkg.__path__ = [str(repo_root / "any_precision")]
    quantization_pkg = types.ModuleType("any_precision.quantization")
    quantization_pkg.__path__ = [str(repo_root / "any_precision" / "quantization")]
    sys.modules.setdefault("any_precision", any_precision_pkg)
    sys.modules.setdefault("any_precision.quantization", quantization_pkg)

    module_path = repo_root / "any_precision" / "quantization" / "nll_hvp_saliency.py"
    spec = importlib.util.spec_from_file_location(
        "any_precision.quantization.nll_hvp_saliency",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nll_hvp_saliency = _load_nll_hvp_saliency()
collect_full_nll_hvp_saliencies = nll_hvp_saliency.collect_full_nll_hvp_saliencies
complete_saliency_cache_exists = nll_hvp_saliency.complete_saliency_cache_exists
reduce_output_channels_to_groups = nll_hvp_saliency.reduce_output_channels_to_groups
_prepare_causal_nll_inputs = nll_hvp_saliency._prepare_causal_nll_inputs


def test_prepare_causal_nll_inputs_masks_pad_tokens():
    tokens = torch.tensor([[1, 2, 0, 0]])

    input_ids, labels, attention_mask, valid_tokens = _prepare_causal_nll_inputs(
        tokens,
        pad_token_id=0,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(input_ids, tokens)
    torch.testing.assert_close(labels, torch.tensor([[1, 2, -100, -100]]))
    torch.testing.assert_close(attention_mask, torch.tensor([[1, 1, 0, 0]]))
    assert valid_tokens == 1


def test_reduce_output_channels_to_groups_averages_contiguous_output_channels():
    sample = torch.tensor([[[1.0, 3.0, 10.0, 14.0]]])

    grouped = reduce_output_channels_to_groups(sample, num_groups=2)

    torch.testing.assert_close(grouped, torch.tensor([[[2.0, 12.0]]]))


def test_reduce_output_channels_to_groups_rejects_uneven_groups():
    sample = torch.zeros(1, 2, 5)

    try:
        reduce_output_channels_to_groups(sample, num_groups=2)
    except ValueError as exc:
        assert "divisible by num_groups" in str(exc)
    else:
        raise AssertionError("Expected uneven output groups to be rejected")


def test_collect_full_nll_hvp_saliencies_writes_guidequant_cache(tmp_path):
    class Output:
        def __init__(self, logits):
            self.logits = logits

    class ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 4, bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(
                    torch.tensor([[0.1], [0.2], [0.3], [0.4]])
                )

        def forward(self, x):
            return torch.tanh(self.linear(x))

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(use_cache=True)
            self.layers = torch.nn.ModuleList([ToyLayer()])
            self.lm_head = torch.nn.Linear(4, 6, bias=False)
            with torch.no_grad():
                self.lm_head.weight.copy_(
                    torch.arange(24, dtype=torch.float32).reshape(6, 4) / 20.0
                )

        def forward(self, input_ids, labels=None, attention_mask=None):
            x = input_ids.float().unsqueeze(-1)
            hidden = self.layers[0](x)
            logits = self.lm_head(hidden)
            return Output(logits)

    class ToyAnalyzer:
        def __init__(self):
            self.model = ToyModel()
            self.tokenizer = types.SimpleNamespace(pad_token_id=None)
            self.module_names = ["linear"]

        def get_layers(self):
            return list(self.model.layers)

        def get_modules(self, layer):
            return {"linear": layer.linear}

    analyzer = ToyAnalyzer()
    tokens = [torch.tensor([0, 1, 2, 3])]

    collect_full_nll_hvp_saliencies(
        analyzer,
        tokens,
        str(tmp_path),
        num_groups=2,
        num_probes=2,
        base_seed=123,
        layer_chunk_size=0,
    )

    saved = torch.load(Path(tmp_path) / "l0.pt", map_location="cpu", weights_only=True)
    metadata = torch.load(
        Path(tmp_path) / "metadata.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert set(saved) == {"linear"}
    assert saved["linear"].shape == (1, 4, 2)
    assert torch.isfinite(saved["linear"].float()).all()
    assert (saved["linear"].float() >= 0).all()
    assert metadata["curvature_mode"] == "nll_global_hvp"
    assert metadata["loss_reduction"] == "sum"
    assert complete_saliency_cache_exists(analyzer, str(tmp_path), 1, 4, 2)


def test_complete_saliency_cache_exists_rejects_wrong_group_count(tmp_path):
    class ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 4, bias=False)

    class ToyAnalyzer:
        def __init__(self):
            self.model = types.SimpleNamespace(name_or_path="toy")
            self.module_names = ["linear"]

        def get_layers(self):
            return [ToyLayer()]

        def get_modules(self, layer):
            return {"linear": layer.linear}

    analyzer = ToyAnalyzer()
    torch.save({"linear": torch.zeros(2, 5, 3)}, tmp_path / "l0.pt")

    assert not complete_saliency_cache_exists(analyzer, str(tmp_path), 2, 5, 2)


def test_layerwise_cli_exposes_nll_global_hvp_args():
    text = Path("layerwise_nuq.py").read_text()

    assert "--hessian_source" in text
    assert "nll_global_hvp" in text
    assert "--curvature_mode" in text
    assert "--nll_hvp_probes" in text
    assert "--nll_hvp_seed" in text
    assert "--nll_hvp_layer_chunk_size" in text
    assert "--overwrite_saliency" in text
    assert "--overwrite_hessians" in text


def test_layerwise_main_routes_nll_global_hvp_through_new_producer_only():
    text = Path("any_precision/quantization/layerwise_main.py").read_text()

    assert "collect_full_nll_hvp_saliencies" in text
    assert "complete_saliency_cache_exists" in text
    assert "saliency_nll_global_hvp" in text
    assert "hessians_nll_global_hvp" in text
    assert "nll_global_hvp_p" in text
    assert "accumulate_saliency_weighted_hessians" in text


def test_llama2_7b_nll_global_hvp_script_uses_distinct_tag_and_cache():
    script = Path(
        "scripts/run_lnq_guidedquant_nll_global_hvp_cd_llama2_7b_c4_eval_ppl.sh"
    )
    text = script.read_text()

    assert "cache_nll_global_hvp_cd_llama2_7b" in text
    assert "RESULT_SUFFIX=\"nll_global_hvp_cd_llama2_7b\"" in text
    assert "--hessian_source nll_global_hvp" in text
    assert "--nll_hvp_probes" in text
    assert "--nll_hvp_seed" in text
    assert "--nll_hvp_layer_chunk_size" in text
    assert "--overwrite_saliency" in text


def test_complete_saliency_cache_exists_rejects_stale_metadata(tmp_path):
    class ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 4, bias=False)

    class ToyAnalyzer:
        module_names = ["linear"]

        def get_layers(self):
            return [ToyLayer()]

        def get_modules(self, layer):
            return {"linear": layer.linear}

    analyzer = ToyAnalyzer()
    torch.save({"linear": torch.zeros(2, 5, 2)}, tmp_path / "l0.pt")
    torch.save(
        {
            "curvature_mode": "nll_global_hvp",
            "num_probes": 1,
            "seed": 0,
            "loss_reduction": "sum",
            "num_groups": 3,
            "num_examples": 2,
            "seq_len": 5,
            "model": "toy",
        },
        tmp_path / "metadata.pt",
    )

    assert not complete_saliency_cache_exists(analyzer, str(tmp_path), 2, 5, 2)


def test_estimate_grouped_diag_matches_exact_hessian_with_enumerated_probes():
    z = torch.tensor([[[0.2, -0.4]]], requires_grad=True)
    hessian = torch.tensor([[2.0, 0.3], [0.3, 4.0]])
    z_flat = z.reshape(-1)
    loss = 0.5 * z_flat @ hessian @ z_flat
    probes = [
        [torch.tensor([[[1.0, 1.0]]])],
        [torch.tensor([[[1.0, -1.0]]])],
        [torch.tensor([[[-1.0, 1.0]]])],
        [torch.tensor([[[-1.0, -1.0]]])],
    ]

    grouped, diagnostics, second_order_calls = nll_hvp_saliency.estimate_grouped_diag_from_hvp(
        loss,
        [z],
        [(0, "linear")],
        num_groups=2,
        probes_by_probe=probes,
    )

    torch.testing.assert_close(grouped[0], torch.tensor([[[2.0, 4.0]]]), atol=1e-6, rtol=1e-6)
    assert diagnostics[0]["raw_min"] < diagnostics[0]["raw_max"]
    assert second_order_calls == 4


def test_collect_full_nll_hvp_saliencies_counts_valid_tokens_once_across_chunks(tmp_path):
    class Output:
        def __init__(self, logits):
            self.logits = logits

    class ToyLayer(torch.nn.Module):
        def __init__(self, in_features):
            super().__init__()
            self.linear = torch.nn.Linear(in_features, 4, bias=False)

        def forward(self, x):
            return torch.tanh(self.linear(x))

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(use_cache=True)
            self.layers = torch.nn.ModuleList([ToyLayer(1), ToyLayer(4)])
            self.lm_head = torch.nn.Linear(4, 6, bias=False)

        def forward(self, input_ids, labels=None, attention_mask=None):
            x = input_ids.float().unsqueeze(-1)
            hidden = self.layers[0](x)
            hidden = self.layers[1](hidden)
            return Output(self.lm_head(hidden))

    class ToyAnalyzer:
        def __init__(self):
            self.model = ToyModel()
            self.tokenizer = types.SimpleNamespace(pad_token_id=None)
            self.module_names = ["linear"]

        def get_layers(self):
            return list(self.model.layers)

        def get_modules(self, layer):
            return {"linear": layer.linear}

    collect_full_nll_hvp_saliencies(
        ToyAnalyzer(),
        [torch.tensor([0, 1, 2, 3])],
        str(tmp_path),
        num_groups=2,
        num_probes=1,
        layer_chunk_size=1,
    )

    metadata = torch.load(tmp_path / "metadata.pt", map_location="cpu", weights_only=True)
    assert metadata["valid_tokens"] == 3


def test_collect_full_nll_hvp_saliencies_logs_diagnostics(tmp_path, caplog):
    class Output:
        def __init__(self, logits):
            self.logits = logits

    class ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 4, bias=False)

        def forward(self, x):
            return torch.sin(self.linear(x))

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(use_cache=True)
            self.layers = torch.nn.ModuleList([ToyLayer()])
            self.lm_head = torch.nn.Linear(4, 6, bias=False)

        def forward(self, input_ids, labels=None, attention_mask=None):
            x = input_ids.float().unsqueeze(-1)
            return Output(self.lm_head(self.layers[0](x)))

    class ToyAnalyzer:
        def __init__(self):
            self.model = ToyModel()
            self.tokenizer = types.SimpleNamespace(pad_token_id=None)
            self.module_names = ["linear"]

        def get_layers(self):
            return list(self.model.layers)

        def get_modules(self, layer):
            return {"linear": layer.linear}

    caplog.set_level("INFO")
    collect_full_nll_hvp_saliencies(
        ToyAnalyzer(),
        [torch.tensor([0, 1, 2, 3])],
        str(tmp_path),
        num_groups=2,
        num_probes=1,
    )

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "raw HVP saliency" in messages
    assert "fraction(raw < 0)" in messages
    assert "fraction(clamped == 0)" in messages
    assert "forward time" in messages
    assert "peak allocated memory" in messages


def test_structural_second_order_hvp_calls_equal_batches_chunks_probes(monkeypatch, tmp_path):
    class Output:
        def __init__(self, logits):
            self.logits = logits

    class ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_a = torch.nn.Linear(1, 4, bias=False)
            self.linear_b = torch.nn.Linear(4, 4, bias=False)

        def forward(self, x):
            return torch.tanh(self.linear_b(torch.tanh(self.linear_a(x))))

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(use_cache=True)
            self.layers = torch.nn.ModuleList([ToyLayer()])
            self.lm_head = torch.nn.Linear(4, 6, bias=False)

        def forward(self, input_ids, labels=None, attention_mask=None):
            x = input_ids.float().unsqueeze(-1)
            return Output(self.lm_head(self.layers[0](x)))

    class ToyAnalyzer:
        def __init__(self):
            self.model = ToyModel()
            self.tokenizer = types.SimpleNamespace(pad_token_id=None)
            self.module_names = ["linear_a", "linear_b"]

        def get_layers(self):
            return list(self.model.layers)

        def get_modules(self, layer):
            return {"linear_a": layer.linear_a, "linear_b": layer.linear_b}

    original_grad = nll_hvp_saliency.torch.autograd.grad
    second_order_calls = {"count": 0}

    def counting_grad(*args, **kwargs):
        outputs = kwargs.get("outputs", args[0] if args else None)
        if isinstance(outputs, tuple):
            second_order_calls["count"] += 1
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(nll_hvp_saliency.torch.autograd, "grad", counting_grad)
    collect_full_nll_hvp_saliencies(
        ToyAnalyzer(),
        [torch.tensor([0, 1, 2, 3])],
        str(tmp_path),
        num_groups=2,
        num_probes=3,
        layer_chunk_size=0,
    )

    assert second_order_calls["count"] == 3


def test_positive_projection_cache_yields_psd_group_hessian(tmp_path):
    class ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 4, bias=False)

    class ToyAnalyzer:
        module_names = ["linear"]

        def get_layers(self):
            return [ToyLayer()]

        def get_modules(self, layer):
            return {"linear": layer.linear}

    saliency = torch.tensor(
        [[[0.0, 1.0], [2.0, 0.5], [0.3, 0.0]]],
        dtype=torch.bfloat16,
    )
    torch.save({"linear": saliency}, tmp_path / "l0.pt")
    torch.save(
        {
            "curvature_mode": "nll_global_hvp",
            "num_probes": 1,
            "seed": 0,
            "loss_reduction": "sum",
            "num_groups": 2,
            "num_examples": 1,
            "seq_len": 3,
            "model": "toy",
        },
        tmp_path / "metadata.pt",
    )
    x = torch.tensor([[1.0, 2.0], [0.5, -1.0], [2.0, 0.0]])
    s = saliency.reshape(-1, 2).float()
    hessians = torch.stack([x.T @ (s[:, g : g + 1] * x) for g in range(2)])

    assert complete_saliency_cache_exists(ToyAnalyzer(), str(tmp_path), 1, 3, 2)
    assert torch.linalg.eigvalsh(hessians.double()).min().item() >= -1e-8
    for hessian in hessians:
        avg_diag = torch.mean(torch.diag(hessian))
        damped = hessian + torch.eye(hessian.shape[0]) * max(avg_diag.item(), 1.0) * 1e-5
        torch.linalg.cholesky(damped)


def test_layerwise_main_validates_full_nll_hvp_metadata_fields():
    text = Path("any_precision/quantization/layerwise_main.py").read_text()

    assert "num_probes=nll_hvp_probes" in text
    assert "base_seed=nll_hvp_seed" in text
    assert "layer_chunk_size=nll_hvp_layer_chunk_size" in text


def test_log_cached_hessian_diagnostics_reports_trace_and_symmetry(tmp_path, caplog):
    class ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 4, bias=False)

    class ToyAnalyzer:
        def get_layers(self):
            return [ToyLayer()]

        def get_modules(self, layer):
            return {"linear": layer.linear}

    hessian = torch.zeros(2, 2, 2)
    hessian[:, :, 0] = torch.tensor([[2.0, 0.1], [0.1, 1.0]])
    hessian[:, :, 1] = torch.tensor([[1.5, 0.0], [0.0, 0.5]])
    torch.save({"linear": hessian}, tmp_path / "l0.pt")

    caplog.set_level("INFO")
    nll_hvp_saliency.log_cached_hessian_diagnostics(ToyAnalyzer(), str(tmp_path))

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "trace(H)" in messages
    assert "||H||_F" in messages
    assert "symmetry error" in messages
    assert "Cholesky damping factor" in messages



def test_layerwise_cli_exposes_nll_hvp_sdpa_backend_math_default():
    text = Path("layerwise_nuq.py").read_text()

    assert "--nll_hvp_sdpa_backend" in text
    assert "default='math'" in text or 'default="math"' in text


def test_producer_uses_sdpa_backend_context():
    text = Path("any_precision/quantization/nll_hvp_saliency.py").read_text()

    assert "def _sdpa_backend_context" in text
    assert "with _sdpa_backend_context(sdpa_backend)" in text
    assert "sdpa_backend: str = \"math\"" in text


def test_script_passes_nll_hvp_sdpa_backend():
    text = Path("scripts/run_lnq_guidedquant_nll_global_hvp_cd_llama2_7b_c4_eval_ppl.sh").read_text()

    assert "NLL_HVP_SDPA_BACKEND=\"${NLL_HVP_SDPA_BACKEND:-math}\"" in text
    assert "--nll_hvp_sdpa_backend \"${NLL_HVP_SDPA_BACKEND}\"" in text


def test_producer_exposes_saved_tensors_cpu_offload_context():
    text = Path("any_precision/quantization/nll_hvp_saliency.py").read_text()

    assert "def _saved_tensors_context" in text
    assert "torch.autograd.graph.save_on_cpu" in text
    assert "saved_tensors_device: str = \"cpu\"" in text
    assert "_saved_tensors_context(saved_tensors_device)" in text


def test_layerwise_cli_exposes_saved_tensors_device_cpu_default():
    text = Path("layerwise_nuq.py").read_text()

    assert "--nll_hvp_saved_tensors_device" in text
    assert "default='cpu'" in text or 'default="cpu"' in text


def test_script_passes_saved_tensors_device_cpu_default():
    text = Path("scripts/run_lnq_guidedquant_nll_global_hvp_cd_llama2_7b_c4_eval_ppl.sh").read_text()

    assert "NLL_HVP_SAVED_TENSORS_DEVICE=\"${NLL_HVP_SAVED_TENSORS_DEVICE:-cpu}\"" in text
    assert "--nll_hvp_saved_tensors_device \"${NLL_HVP_SAVED_TENSORS_DEVICE}\"" in text





def test_layerwise_main_signature_accepts_saved_tensors_device():
    import ast

    tree = ast.parse(Path("any_precision/quantization/layerwise_main.py").read_text())
    func = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "layerwise_nuq"
    )
    params = {arg.arg for arg in func.args.args}

    assert "nll_hvp_saved_tensors_device" in params
