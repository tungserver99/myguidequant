from pathlib import Path
import ast
import importlib.util
import sys
import types
import tempfile

import torch
import transformers


def _load_activations():
    repo_root = Path(__file__).resolve().parents[1]

    any_precision_pkg = types.ModuleType("any_precision")
    any_precision_pkg.__path__ = [str(repo_root / "any_precision")]
    quantization_pkg = types.ModuleType("any_precision.quantization")
    quantization_pkg.__path__ = [str(repo_root / "any_precision" / "quantization")]
    analyzer_pkg = types.ModuleType("any_precision.analyzer")
    analyzer_pkg.get_analyzer = lambda *args, **kwargs: None
    analyzer_mod = types.ModuleType("any_precision.analyzer.analyzer")
    analyzer_mod.ModelAnalyzer = object

    sys.modules.setdefault("any_precision", any_precision_pkg)
    sys.modules.setdefault("any_precision.quantization", quantization_pkg)
    sys.modules.setdefault("any_precision.analyzer", analyzer_pkg)
    sys.modules.setdefault("any_precision.analyzer.analyzer", analyzer_mod)

    if not hasattr(transformers, "Gemma3Config"):
        transformers.Gemma3Config = type("Gemma3Config", (), {})

    module_path = repo_root / "any_precision" / "quantization" / "activations.py"
    spec = importlib.util.spec_from_file_location(
        "any_precision.quantization.activations",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


activations = _load_activations()
build_group_hessians_from_curvature = activations.build_group_hessians_from_curvature
build_group_hessians_legacy = activations.build_group_hessians_legacy
build_group_hessians_batched = activations.build_group_hessians_batched
build_shared_x_group_hessians = activations.build_shared_x_group_hessians
build_fd_shared_x_buffer_hessians = activations.build_fd_shared_x_buffer_hessians
reduce_channels_by_group_mean = activations.reduce_channels_by_group_mean
accumulate_nll_ggn_hessians = activations.accumulate_nll_ggn_hessians
accumulate_fast_hnll_base_residual_hvp_hessians = activations.accumulate_fast_hnll_base_residual_hvp_hessians
accumulate_fd_grouptrace_hnll_hessians = activations.accumulate_fd_grouptrace_hnll_hessians
_prepare_hnll_tokens_and_labels = activations._prepare_hnll_tokens_and_labels


def test_prepare_hnll_tokens_and_labels_counts_batched_next_token_labels():
    tokens = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]
    )

    batch, labels, attention_mask, valid_tokens = _prepare_hnll_tokens_and_labels(tokens)

    torch.testing.assert_close(batch, tokens)
    torch.testing.assert_close(labels, tokens)
    assert attention_mask is None
    assert valid_tokens == 6


def test_prepare_hnll_tokens_and_labels_masks_pad_tokens():
    tokens = torch.tensor([[1, 2, 0, 0]])

    batch, labels, attention_mask, valid_tokens = _prepare_hnll_tokens_and_labels(tokens, pad_token_id=0)

    torch.testing.assert_close(batch, tokens)
    torch.testing.assert_close(labels, torch.tensor([[1, 2, -100, -100]]))
    torch.testing.assert_close(attention_mask, torch.tensor([[1, 1, 0, 0]]))
    assert valid_tokens == 1

def test_reduce_channels_by_group_mean_matches_explicit_group_average():
    sample = torch.tensor(
        [
            [1.0, 3.0, 10.0, 14.0],
            [5.0, 7.0, 20.0, 24.0],
        ]
    )

    grouped = reduce_channels_by_group_mean(sample, num_groups=2)

    expected = torch.tensor(
        [
            [2.0, 12.0],
            [6.0, 22.0],
        ]
    )
    torch.testing.assert_close(grouped, expected)


def test_build_group_hessians_from_curvature_uses_weighted_gemm_and_symmetrizes():
    x = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    s_group = torch.tensor(
        [
            [1.0, 0.0],
            [4.0, 1.0],
            [0.0, 9.0],
        ]
    )

    hessians = build_group_hessians_from_curvature(x, s_group)

    expected_0 = x.T @ torch.diag(s_group[:, 0]) @ x
    expected_1 = x.T @ torch.diag(s_group[:, 1]) @ x
    expected = torch.stack([expected_0, expected_1], dim=0)
    torch.testing.assert_close(hessians, expected)
    torch.testing.assert_close(hessians, hessians.transpose(-1, -2))



def test_resolve_hnll_curvature_dtype_current_keeps_model_dtype():
    resolved = activations._resolve_hnll_curvature_dtype("current", torch.device("cpu"), torch.float16)

    assert resolved == torch.float16


def test_batched_group_hessians_match_legacy_for_chunk_sizes():
    x = torch.randn(8, 5, dtype=torch.float64)
    stats = torch.rand(8, 7, dtype=torch.float64)

    expected = build_group_hessians_legacy(x, stats)

    for chunk_size in [1, 2, 7]:
        actual = build_group_hessians_batched(x, stats, group_chunk_size=chunk_size)
        torch.testing.assert_close(actual.double(), expected.double(), rtol=1e-5, atol=1e-5)


def test_shared_x_group_hessians_match_separate_builds_with_unequal_groups():
    from collections import OrderedDict

    x = torch.randn(6, 4, dtype=torch.float64)
    module_stats = OrderedDict([
        ("self_attn.q_proj", torch.rand(6, 4, dtype=torch.float64)),
        ("self_attn.k_proj", torch.rand(6, 2, dtype=torch.float64)),
        ("self_attn.v_proj", torch.rand(6, 3, dtype=torch.float64)),
    ])

    fused = build_shared_x_group_hessians(x, module_stats, group_chunk_size=3)

    for module_name, stats in module_stats.items():
        expected = build_group_hessians_batched(x, stats, group_chunk_size=2)
        torch.testing.assert_close(fused[module_name].double(), expected.double(), rtol=1e-5, atol=1e-5)




def test_fd_shared_x_buffer_hessians_match_full_concat_without_duplicate_x():
    module_keys = [
        (0, "self_attn.q_proj"),
        (0, "self_attn.k_proj"),
        (0, "mlp.down_proj"),
    ]
    sample_xs = [
        {
            module_keys[0]: torch.randn(3, 4),
            module_keys[1]: None,
            module_keys[2]: torch.randn(3, 4),
        },
        {
            module_keys[0]: torch.randn(2, 4),
            module_keys[1]: None,
            module_keys[2]: torch.randn(2, 4),
        },
    ]
    sample_xs[0][module_keys[1]] = sample_xs[0][module_keys[0]]
    sample_xs[1][module_keys[1]] = sample_xs[1][module_keys[0]]
    sample_stats = [
        {key: torch.rand(3, 2) for key in module_keys},
        {key: torch.rand(2, 2) for key in module_keys},
    ]

    buffered = activations.create_fd_shared_x_buffer(module_keys)
    for xs_by_key, stats_by_key in zip(sample_xs, sample_stats):
        activations.append_fd_shared_x_buffer(buffered, module_keys, xs_by_key, stats_by_key)

    actual = build_fd_shared_x_buffer_hessians(
        buffered,
        builder="batched_shared_x",
        group_chunk_size=2,
        validate_shared_x=False,
        device=torch.device("cpu"),
    )

    for key in module_keys:
        x_all = torch.cat([sample[key] for sample in sample_xs], dim=0)
        stats_all = torch.cat([sample[key] for sample in sample_stats], dim=0)
        expected = build_group_hessians_batched(x_all, stats_all.clamp_min(0), group_chunk_size=2)
        torch.testing.assert_close(actual[key].double(), expected.double(), rtol=1e-5, atol=1e-5)

    q_shared_id = activations._shared_x_semantic_id(*module_keys[0])
    assert len(buffered["x_buffers"]) == 2
    assert len(buffered["x_buffers"][q_shared_id]) == 2

def test_batched_builder_rejects_negative_stats():
    x = torch.randn(3, 2)
    stats = torch.tensor([[1.0, -0.1], [0.5, 0.2], [0.1, 0.3]])

    try:
        build_group_hessians_batched(x, stats, group_chunk_size=2)
    except ValueError as exc:
        assert "negative stats" in str(exc)
    else:
        raise AssertionError("Expected negative stats to be rejected")

def test_accumulate_nll_ggn_hessians_runs_toy_lm():
    class Output:
        def __init__(self, logits):
            self.logits = logits
            self.loss = logits.sum() * 0.0

    class ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 3, bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(torch.tensor([[0.2], [0.4], [0.6]]))

        def forward(self, x):
            return self.linear(x)

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(use_cache=True)
            self.layers = torch.nn.ModuleList([ToyLayer()])

        def forward(self, input_ids, labels=None, attention_mask=None):
            x = input_ids.float().unsqueeze(-1)
            logits = self.layers[0](x)
            return Output(logits)

    class ToyAnalyzer:
        def __init__(self):
            self.model = ToyModel()
            self.tokenizer = types.SimpleNamespace(pad_token_id=None)

        def get_layers(self):
            return list(self.model.layers)

        def get_modules(self, layer):
            return {"linear": layer.linear}

    analyzer = ToyAnalyzer()
    data = [torch.tensor([[0, 1, 2]])]
    with tempfile.TemporaryDirectory() as tmpdir:
        accumulate_nll_ggn_hessians(
            analyzer,
            data,
            tmpdir,
            num_groups=1,
            num_probes=2,
            random_state=0,
            hessian_builder="batched",
            hessian_group_chunk_size=1,
            curvature_dtype="current",
        )
        saved = torch.load(Path(tmpdir) / "l0.pt", map_location="cpu", weights_only=True)

    assert "linear" in saved
    hessian = saved["linear"]
    assert hessian.shape == (1, 1, 1)
    assert torch.isfinite(hessian).all()
    assert hessian.item() >= 0

def test_accumulate_fast_hnll_base_residual_hvp_hessians_runs_toy_lm():
    class Output:
        def __init__(self, logits):
            self.logits = logits
            self.loss = logits.sum() * 0.0

    class ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 3, bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(torch.tensor([[0.2], [0.4], [0.6]]))

        def forward(self, x):
            return torch.tanh(self.linear(x))

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(use_cache=True)
            self.layers = torch.nn.ModuleList([ToyLayer()])
            self.lm_head = torch.nn.Linear(3, 5, bias=False)
            with torch.no_grad():
                self.lm_head.weight.copy_(torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10.0)

        def forward(self, input_ids, labels=None, attention_mask=None):
            x = input_ids.float().unsqueeze(-1)
            hidden = self.layers[0](x)
            logits = self.lm_head(hidden)
            return Output(logits)

    class ToyAnalyzer:
        def __init__(self):
            self.model = ToyModel()
            self.tokenizer = types.SimpleNamespace(pad_token_id=None)

        def get_layers(self):
            return list(self.model.layers)

        def get_modules(self, layer):
            return {"linear": layer.linear}

    analyzer = ToyAnalyzer()
    data = [torch.tensor([[0, 1, 2, 3]])]
    with tempfile.TemporaryDirectory() as tmpdir:
        accumulate_fast_hnll_base_residual_hvp_hessians(
            analyzer,
            data,
            tmpdir,
            num_groups=1,
            num_probes=1,
            random_state=0,
            hessian_builder="batched",
            hessian_group_chunk_size=1,
            curvature_dtype="current",
        )
        saved = torch.load(Path(tmpdir) / "l0.pt", map_location="cpu", weights_only=True)

    assert "linear" in saved
    hessian = saved["linear"]
    assert hessian.shape == (1, 1, 1)
    assert torch.isfinite(hessian).all()
    assert hessian.item() >= 0

def test_accumulate_fd_grouptrace_hnll_hessians_runs_toy_lm():
    class Output:
        def __init__(self, logits):
            self.logits = logits
            self.loss = logits.sum() * 0.0

    class ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 3, bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(torch.tensor([[0.2], [0.4], [0.6]]))

        def forward(self, x):
            return torch.tanh(self.linear(x))

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(use_cache=True)
            self.layers = torch.nn.ModuleList([ToyLayer()])
            self.lm_head = torch.nn.Linear(3, 5, bias=False)
            with torch.no_grad():
                self.lm_head.weight.copy_(torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10.0)

        def forward(self, input_ids, labels=None, attention_mask=None):
            x = input_ids.float().unsqueeze(-1)
            hidden = self.layers[0](x)
            logits = self.lm_head(hidden)
            return Output(logits)

    class ToyAnalyzer:
        def __init__(self):
            self.model = ToyModel()
            self.tokenizer = types.SimpleNamespace(pad_token_id=None)

        def get_layers(self):
            return list(self.model.layers)

        def get_modules(self, layer):
            return {"linear": layer.linear}

    analyzer = ToyAnalyzer()
    data = [torch.tensor([[0, 1, 2, 3]])]
    with tempfile.TemporaryDirectory() as tmpdir:
        accumulate_fd_grouptrace_hnll_hessians(
            analyzer,
            data,
            tmpdir,
            num_groups=1,
            num_probes=1,
            random_state=0,
            hessian_builder="batched",
            hessian_group_chunk_size=1,
            curvature_dtype="current",
            fd_execution_mode="sequential",
            fd_scale_mode="activation_rms",
            fd_scale_multiplier=1e-2,
        )
        saved = torch.load(Path(tmpdir) / "l0.pt", map_location="cpu", weights_only=True)

    assert "linear" in saved
    hessian = saved["linear"]
    assert hessian.shape == (1, 1, 1)
    assert torch.isfinite(hessian).all()
    assert hessian.item() >= 0
def test_accumulate_nll_hvp_hessians_finite_diff_runs_toy_model():
    class Output:
        def __init__(self, loss):
            self.loss = loss

    class ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 2, bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(torch.tensor([[1.0], [2.0]]))

        def forward(self, x):
            return self.linear(x)

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(use_cache=True)
            self.layers = torch.nn.ModuleList([ToyLayer()])

        def forward(self, input_ids, labels=None, attention_mask=None):
            x = input_ids.float().unsqueeze(-1)
            out = self.layers[0](x)
            return Output(out.pow(2).sum())

    class ToyAnalyzer:
        def __init__(self):
            self.model = ToyModel()
            self.tokenizer = types.SimpleNamespace(pad_token_id=None)

        def get_layers(self):
            return list(self.model.layers)

        def get_modules(self, layer):
            return {"linear": layer.linear}

    analyzer = ToyAnalyzer()
    data = [torch.tensor([[1, 2, 3]])]
    with tempfile.TemporaryDirectory() as tmpdir:
        activations.accumulate_nll_hvp_hessians(
            analyzer,
            data,
            tmpdir,
            num_groups=1,
            num_probes=1,
            random_state=0,
            hessian_builder="batched",
            hessian_group_chunk_size=1,
            hvp_engine="finite_diff",
            fd_epsilon=1e-3,
            sdpa_backend="math",
        )
        saved = torch.load(Path(tmpdir) / "l0.pt", map_location="cpu", weights_only=True)

    assert "linear" in saved
    hessian = saved["linear"]
    assert hessian.shape == (1, 1, 1)
    assert torch.isfinite(hessian).all()
    assert hessian.item() > 0

def test_layerwise_cli_exposes_hnll_hvp_flags():
    text = Path("layerwise_nuq.py").read_text()

    assert "--hessian_source" in text
    assert "nll_hvp" in text
    assert "nll_ggn" in text
    assert "nll_base_residual_hvp" in text
    assert "nll_fd_grouptrace" in text
    assert "--nll_hvp_probes" in text
    assert "--nll_hvp_layer_chunk_size" in text
    assert "--nll_hvp_profile" in text
    assert "--nll_hvp_dtype" in text
    assert "--nll_hessian_builder" in text
    assert "--nll_hessian_group_chunk_size" in text
    assert "--nll_hessian_validate_shared_x" in text
    assert "--nll_hvp_sdpa_backend" in text
    assert "--nll_hvp_engine" in text
    assert "--nll_hvp_fd_epsilon" in text
    assert "--nll_hvp_fd_batched_signs" in text
    assert "--fd_execution_mode" in text
    assert "--fd_scale_mode" in text
    assert "--fd_scale_multiplier" in text


def test_layerwise_main_accepts_hnll_finite_diff_cli_kwargs():
    tree = ast.parse(Path("any_precision/quantization/layerwise_main.py").read_text(encoding="utf-8-sig"))
    layerwise_func = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "layerwise_nuq"
    )
    params = {arg.arg for arg in layerwise_func.args.args}

    assert "nll_hvp_engine" in params
    assert "nll_hvp_fd_epsilon" in params
    assert "nll_hvp_fd_batched_signs" in params
    assert "fd_execution_mode" in params
    assert "fd_scale_mode" in params
    assert "fd_scale_multiplier" in params
    assert "fd_build_flush_interval" in params
    assert "overwrite_hessians" in params

def test_layerwise_main_forwards_hnll_finite_diff_kwargs_to_accumulator():
    tree = ast.parse(Path("any_precision/quantization/layerwise_main.py").read_text(encoding="utf-8-sig"))
    call = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "accumulate_nll_hvp_hessians"
    )
    keywords = {keyword.arg for keyword in call.keywords}

    assert "hvp_engine" in keywords
    assert "fd_epsilon" in keywords
    assert "fd_batched_signs" in keywords

def test_layerwise_cli_and_fd_grouptrace_script_expose_hessian_overwrite():
    cli_text = Path("layerwise_nuq.py").read_text()
    script_text = Path(
        "scripts/run_lnq_guidedquant_fd_grouptrace_cd_llama2_7b_c4_eval_ppl.sh"
    ).read_text()

    assert "--overwrite_hessians" in cli_text
    assert "hessian_overwrite_args" in script_text
    assert "--overwrite_hessians" in script_text

def test_layerwise_main_removes_hessian_cache_when_overwrite_hessians_is_set():
    text = Path("any_precision/quantization/layerwise_main.py").read_text(encoding="utf-8-sig")

    assert "overwrite_hessians and os.path.exists(hessians_cache_path)" in text
    assert "shutil.rmtree(hessians_cache_path)" in text
