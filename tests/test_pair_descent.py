import importlib.util
import sys
import types
from pathlib import Path

import torch


def _load_layerwise_quantize():
    repo_root = Path(__file__).resolve().parents[1]

    any_precision_pkg = types.ModuleType("any_precision")
    any_precision_pkg.__path__ = [str(repo_root / "any_precision")]
    quantization_pkg = types.ModuleType("any_precision.quantization")
    quantization_pkg.__path__ = [str(repo_root / "any_precision" / "quantization")]
    analyzer_pkg = types.ModuleType("any_precision.analyzer")
    analyzer_mod = types.ModuleType("any_precision.analyzer.analyzer")
    analyzer_mod.ModelAnalyzer = object

    sys.modules.setdefault("any_precision", any_precision_pkg)
    sys.modules.setdefault("any_precision.quantization", quantization_pkg)
    sys.modules.setdefault("any_precision.analyzer", analyzer_pkg)
    sys.modules.setdefault("any_precision.analyzer.analyzer", analyzer_mod)

    module_path = repo_root / "any_precision" / "quantization" / "layerwise_quantize.py"
    spec = importlib.util.spec_from_file_location(
        "any_precision.quantization.layerwise_quantize",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


layerwise_quantize = _load_layerwise_quantize()
build_greedy_pair_matching = layerwise_quantize.build_greedy_pair_matching
build_pair_permutation = layerwise_quantize.build_pair_permutation
solve_pair_bruteforce = layerwise_quantize.solve_pair_bruteforce
update_P_cd = layerwise_quantize.update_P_cd
update_P_pair = layerwise_quantize.update_P_pair
update_P = layerwise_quantize.update_P


def _make_spd(d, dtype=torch.float64):
    a = torch.arange(1, d * d + 1, dtype=dtype).reshape(d, d) / (d * d)
    return a @ a.T + torch.eye(d, dtype=dtype) * 0.5


def _objective(W, H, labels, C):
    W_hat = torch.gather(
        C.unsqueeze(1).expand(-1, labels.shape[1], -1),
        dim=2,
        index=labels.long().unsqueeze(-1),
    ).squeeze(-1)
    E = W_hat - W
    G = H.shape[0]
    R = W.shape[0] // G
    E_grp = E.reshape(G, R, W.shape[1])
    return torch.einsum("grd,gde,gre->", E_grp, H, E_grp)


def test_matching_and_permutation_cover_each_coordinate_once():
    H0 = torch.tensor(
        [
            [4.0, 3.0, 0.1, 0.2, 0.0],
            [3.0, 5.0, 0.3, 0.1, 0.0],
            [0.1, 0.3, 6.0, 2.0, 0.4],
            [0.2, 0.1, 2.0, 7.0, 0.5],
            [0.0, 0.0, 0.4, 0.5, 3.0],
        ]
    )
    pairs, singleton = build_greedy_pair_matching(H0.unsqueeze(0))
    perm, inv_perm = build_pair_permutation(pairs, singleton, H0.shape[0])

    assert sorted(perm.tolist()) == list(range(H0.shape[0]))
    assert torch.equal(perm[inv_perm], torch.arange(H0.shape[0]))
    assert len(pairs) == 2
    assert singleton is not None


def test_bruteforce_pair_solver_selects_exact_minimum_for_unsorted_codebooks():
    C_grp = torch.tensor([[[0.5, -1.0, 2.0], [1.5, -0.5, 0.25]]], dtype=torch.float64)
    W_i = torch.tensor([[0.2, -0.8]], dtype=torch.float64)
    W_j = torch.tensor([[1.2, 0.4]], dtype=torch.float64)
    e_i = torch.tensor([[0.3, 0.2]], dtype=torch.float64)
    e_j = torch.tensor([[0.3, -0.15]], dtype=torch.float64)
    z_i = torch.tensor([[0.7, -0.2]], dtype=torch.float64)
    z_j = torch.tensor([[0.1, 0.5]], dtype=torch.float64)
    H_ii = torch.tensor([4.0], dtype=torch.float64)
    H_jj = torch.tensor([3.0], dtype=torch.float64)
    H_ij = torch.tensor([-1.2], dtype=torch.float64)

    result = solve_pair_bruteforce(W_i, W_j, C_grp, e_i, e_j, z_i, z_j, H_ii, H_jj, H_ij)

    si = z_i - H_ii.view(1, 1) * e_i - H_ij.view(1, 1) * e_j
    sj = z_j - H_ij.view(1, 1) * e_i - H_jj.view(1, 1) * e_j
    Ei = C_grp - W_i.unsqueeze(-1)
    Ej = C_grp - W_j.unsqueeze(-1)
    cost = (
        0.5 * H_ii.view(1, 1, 1, 1) * Ei.unsqueeze(-1).square()
        + 0.5 * H_jj.view(1, 1, 1, 1) * Ej.unsqueeze(-2).square()
        + H_ij.view(1, 1, 1, 1) * Ei.unsqueeze(-1) * Ej.unsqueeze(-2)
        + si.unsqueeze(-1).unsqueeze(-1) * Ei.unsqueeze(-1)
        + sj.unsqueeze(-1).unsqueeze(-2) * Ej.unsqueeze(-2)
    )
    expected = cost.flatten(-2).argmin(dim=-1)
    assert torch.equal(result.label_i, expected // C_grp.shape[-1])
    assert torch.equal(result.label_j, expected % C_grp.shape[-1])


def test_pair_sweep_does_not_increase_original_objective():
    dtype = torch.float64
    W = torch.tensor([[0.2, -0.7, 1.4, -1.2], [1.0, 0.3, -0.4, 0.8]], dtype=dtype)
    C = torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.5, 1.5]], dtype=dtype)
    labels = torch.tensor([[0, 2, 0, 2], [2, 0, 2, 0]])
    H = _make_spd(4, dtype=dtype).unsqueeze(0)

    before = _objective(W, H, labels, C)
    new_labels = update_P_pair(W, H, labels, C, cd_cycles=1, verbose=False)
    after = _objective(W, H, new_labels, C)

    assert after <= before + 32 * torch.finfo(dtype).eps * max(1.0, abs(before.item()))


def test_pair_descent_crosses_two_coordinate_barrier_that_scalar_cd_cannot():
    dtype = torch.float64
    W = torch.tensor([[-3.0, -3.0]], dtype=dtype)
    C = torch.tensor([[-2.0, -1.0]], dtype=dtype)
    labels = torch.tensor([[1, 1]])
    H = torch.tensor([[[1.0, -0.95], [-0.95, 1.0]]], dtype=dtype)

    cd_labels = update_P_cd(W, H, labels, C, cd_cycles=1, verbose=False)
    pair_labels = update_P_pair(W, H, labels, C, cd_cycles=1, verbose=False)

    assert torch.equal(cd_labels.cpu(), labels)
    assert torch.equal(pair_labels.cpu(), torch.tensor([[0, 0]]))
    assert _objective(W, H, pair_labels, C) < _objective(W, H, labels, C)



def test_update_p_dispatcher_defaults_to_pair_and_preserves_cd_baseline():
    dtype = torch.float64
    W = torch.tensor([[-3.0, -3.0]], dtype=dtype)
    C = torch.tensor([[-2.0, -1.0]], dtype=dtype)
    labels = torch.tensor([[1, 1]])
    H = torch.tensor([[[1.0, -0.95], [-0.95, 1.0]]], dtype=dtype)

    default_labels = update_P(W, H, labels, C, cd_cycles=1, verbose=False)
    cd_labels = update_P(W, H, labels, C, cd_cycles=1, verbose=False, assignment_solver="cd")

    assert torch.equal(default_labels.cpu(), torch.tensor([[0, 0]]))
    assert torch.equal(cd_labels.cpu(), labels)
