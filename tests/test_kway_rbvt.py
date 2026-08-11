import itertools
import importlib.util
import sys
import types
from pathlib import Path

import torch


def _load_layerwise_quantize():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "any_precision" / "quantization" / "layerwise_quantize.py"

    any_precision = types.ModuleType("any_precision")
    any_precision.__path__ = [str(root / "any_precision")]
    analyzer_pkg = types.ModuleType("any_precision.analyzer")
    analyzer_pkg.__path__ = [str(root / "any_precision" / "analyzer")]
    analyzer_mod = types.ModuleType("any_precision.analyzer.analyzer")
    analyzer_mod.ModelAnalyzer = object
    quantization_pkg = types.ModuleType("any_precision.quantization")
    quantization_pkg.__path__ = [str(root / "any_precision" / "quantization")]

    sys.modules.setdefault("any_precision", any_precision)
    sys.modules.setdefault("any_precision.analyzer", analyzer_pkg)
    sys.modules.setdefault("any_precision.analyzer.analyzer", analyzer_mod)
    sys.modules.setdefault("any_precision.quantization", quantization_pkg)

    spec = importlib.util.spec_from_file_location(
        "any_precision.quantization.layerwise_quantize",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


layerwise_quantize = _load_layerwise_quantize()
update_P_rbvt = layerwise_quantize.update_P_rbvt


def _spd(dim):
    torch.manual_seed(0)
    A = torch.randn(dim, dim)
    return A.T @ A + torch.eye(dim) * 0.25


def _objective_row(W, H, labels, C):
    q = C[labels.long()]
    e = q - W
    return (e * (e @ H)).sum()


def test_rbvt_transition_identity():
    H = _spd(4)
    e = torch.randn(4)
    r = e @ H
    J = (e * r).sum()
    i = 2
    delta = torch.tensor(0.37)

    e_new = e.clone()
    e_new[i] += delta

    direct = (e_new * (e_new @ H)).sum()
    incremental = J + 2 * delta * r[i] + delta.square() * H[i, i]

    assert torch.allclose(direct, incremental, atol=1e-5, rtol=1e-5)


def test_rbvt_residual_and_suffix_identity():
    H = _spd(5)
    e = torch.randn(5)
    residual = e @ H
    i = 1
    delta = torch.tensor(-0.42)

    e_new = e.clone()
    e_new[i] += delta

    direct = e_new @ H
    updated = residual + delta * H[i, :]
    suffix_updated = residual[i + 1 :] + delta * H[i, i + 1 :]

    assert torch.allclose(direct, updated, atol=1e-5, rtol=1e-5)
    assert torch.allclose(direct[i + 1 :], suffix_updated, atol=1e-5, rtol=1e-5)


def test_rbvt_matches_exhaustive_tiny_problem():
    torch.manual_seed(1)
    d = 4
    K = 3
    W = torch.randn(1, d)
    H = _spd(d).unsqueeze(0)
    C = torch.tensor([[-1.0, 0.25, 1.5]])
    labels = torch.zeros(1, d, dtype=torch.long)

    rbvt_labels = update_P_rbvt(
        W,
        H,
        labels,
        C,
        rbvt_cycles=1,
        beam_width=K**d,
        row_batch_size=1,
        verbose=False,
    ).cpu()

    exhaustive = []
    for candidate in itertools.product(range(K), repeat=d):
        candidate_labels = torch.tensor(candidate, dtype=torch.long)
        exhaustive.append((_objective_row(W[0], H[0], candidate_labels, C[0]), candidate_labels))
    best_obj, best_labels = min(exhaustive, key=lambda item: item[0].item())
    rbvt_obj = _objective_row(W[0], H[0], rbvt_labels[0], C[0])

    assert torch.allclose(rbvt_obj, best_obj, atol=1e-5, rtol=1e-5)
    assert torch.equal(rbvt_labels[0], best_labels)


def test_rbvt_returns_valid_labels_for_batched_rows():
    torch.manual_seed(2)
    rows = 5
    d = 6
    K = 4
    W = torch.randn(rows, d)
    H = _spd(d).unsqueeze(0)
    C = torch.randn(rows, K).sort(dim=-1).values
    labels = torch.randint(0, K, (rows, d), dtype=torch.long)

    out = update_P_rbvt(
        W,
        H,
        labels,
        C,
        rbvt_cycles=2,
        beam_width=3,
        row_batch_size=2,
        verbose=False,
    )

    assert out.shape == labels.shape
    assert out.dtype == torch.long
    assert out.min().item() >= 0
    assert out.max().item() < K
