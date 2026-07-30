import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from any_precision.analyzer.analyzer import ModelAnalyzer
from typing import List, Tuple, Literal, Optional, NamedTuple
import time

from .utils import get_progress_bar

@torch.no_grad()
def objective_function(
    W: torch.Tensor, 
    H: torch.Tensor, 
    labels: torch.Tensor, 
    C: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the quantization error (objective value).
    
    Args:
    W: Weight matrix (output_dim, input_dim)
    H: Hessian matrix (num_groups, input_dim, input_dim)
    labels: Assignment matrix (output_dim, input_dim)
    C: Centroid matrix (output_dim, n_cluster)
    
    Returns:
    Objective value (scalar)
    """

    device = torch.device("cuda")
    labels, C = labels.to(device), C.to(device)
    W_hat = torch.gather(
        C.unsqueeze(1).expand(-1, labels.shape[1], -1), 
        dim=2, 
        index=labels.unsqueeze(-1).long()
    ).squeeze(-1)
    delta_w = W_hat - W

    num_groups = H.shape[0]
    group_size = W.shape[0] // num_groups

    delta_w = delta_w.reshape(num_groups, group_size, delta_w.shape[-1])
    objective_value = torch.einsum('nij,njk,nik->i', delta_w, H, delta_w) 
    total_error = objective_value.mean()

    return total_error


class PairSolveResult(NamedTuple):
    label_i: torch.Tensor
    label_j: torch.Tensor
    error_i: torch.Tensor
    error_j: torch.Tensor
    cost: torch.Tensor


def build_normalized_curvature(H: torch.Tensor) -> torch.Tensor:
    assert H.ndim == 3
    assert H.shape[1] == H.shape[2]
    diag = torch.diagonal(H, dim1=-2, dim2=-1)
    assert torch.all(diag > 0)
    tiny = torch.finfo(H.dtype).tiny
    denom = torch.sqrt(torch.clamp(diag.unsqueeze(-1) * diag.unsqueeze(-2), min=tiny))
    rho = torch.abs(H) / denom
    rho = rho.mean(dim=0)
    rho.fill_diagonal_(float("-inf"))
    return rho


def build_greedy_pair_matching(H: torch.Tensor):
    rho = build_normalized_curvature(H).detach().cpu()
    d = rho.shape[0]
    best_per_node = rho.max(dim=1).values
    node_order = sorted(range(d), key=lambda i: (-float(best_per_node[i]), i))

    used = [False] * d
    pairs = []
    singleton = None
    for i in node_order:
        if used[i]:
            continue
        best_j = None
        best_val = None
        for j in range(d):
            if i == j or used[j]:
                continue
            val = float(rho[i, j])
            if best_j is None or val > best_val or (val == best_val and j < best_j):
                best_j = j
                best_val = val
        if best_j is None:
            singleton = i
            used[i] = True
            continue
        pairs.append((i, best_j))
        used[i] = True
        used[best_j] = True

    if singleton is None:
        remaining = [i for i, flag in enumerate(used) if not flag]
        if remaining:
            singleton = remaining[0]
    return pairs, singleton


def build_pair_permutation(pairs, singleton, D: int):
    perm_list = [idx for pair in pairs for idx in pair]
    if singleton is not None:
        perm_list.append(singleton)
    assert sorted(perm_list) == list(range(D))
    perm = torch.tensor(perm_list, dtype=torch.long)
    inv_perm = torch.argsort(perm)
    return perm, inv_perm


def solve_pair_bruteforce(
    W_i: torch.Tensor,
    W_j: torch.Tensor,
    C_grp: torch.Tensor,
    e_i_current: torch.Tensor,
    e_j_current: torch.Tensor,
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    H_ii: torch.Tensor,
    H_jj: torch.Tensor,
    H_ij: torch.Tensor,
) -> PairSolveResult:
    K = C_grp.shape[-1]
    h_shape = (H_ii.shape[0],) + (1,) * (C_grp.ndim - 1)
    H_ii_v = H_ii.view(h_shape)
    H_jj_v = H_jj.view(h_shape)
    H_ij_v = H_ij.view(h_shape)

    s_i = z_i - H_ii.view(-1, 1) * e_i_current - H_ij.view(-1, 1) * e_j_current
    s_j = z_j - H_ij.view(-1, 1) * e_i_current - H_jj.view(-1, 1) * e_j_current

    E_i_cand = C_grp - W_i.unsqueeze(-1)
    E_j_cand = C_grp - W_j.unsqueeze(-1)
    pair_cost = (
        0.5 * H_ii_v.unsqueeze(-1) * E_i_cand.unsqueeze(-1).square()
        + 0.5 * H_jj_v.unsqueeze(-2) * E_j_cand.unsqueeze(-2).square()
        + H_ij_v.unsqueeze(-1) * E_i_cand.unsqueeze(-1) * E_j_cand.unsqueeze(-2)
        + s_i.unsqueeze(-1).unsqueeze(-1) * E_i_cand.unsqueeze(-1)
        + s_j.unsqueeze(-1).unsqueeze(-2) * E_j_cand.unsqueeze(-2)
    )
    min_cost, flat_index = pair_cost.flatten(-2).min(dim=-1)
    label_i = flat_index // K
    label_j = flat_index % K
    error_i = torch.gather(E_i_cand, dim=-1, index=label_i.unsqueeze(-1)).squeeze(-1)
    error_j = torch.gather(E_j_cand, dim=-1, index=label_j.unsqueeze(-1)).squeeze(-1)
    return PairSolveResult(label_i, label_j, error_i, error_j, min_cost)


def update_P_cd(
    W: torch.Tensor,  # Shape: (output_dim, input_dim)
    H: torch.Tensor,  # Shape: (num_groups, input_dim, input_dim)
    labels: torch.Tensor,  # Shape: (output_dim, input_dim)
    C: torch.Tensor,  # Shape: (output_dim, n_cluster)
    cd_cycles: int,
    verbose: bool = True,
):
    device = W.device
    C = C.to(device)
    H = H.to(device)
    assignments_prev = labels.to(device).long()  # Shape: (output_dim, input_dim)
    b, d = assignments_prev.shape
    n_cluster = C.size(1)
    num_groups = H.shape[0]
    group_size = W.shape[0] // num_groups


    assignments = assignments_prev.clone()

    update_size = cd_cycles * d

    W_hat = torch.gather(C.unsqueeze(1).expand(-1, d, -1), dim=2, index=assignments.unsqueeze(-1)).squeeze(-1) # Shape: (output_dim, input_dim)


    assert W.shape[0] % num_groups == 0

    pb = get_progress_bar(update_size, f"Updating P inside") if verbose else None

    W_grp = W.reshape(num_groups, group_size, W.shape[-1]) # Shape: (num_groups, group_size, input_dim)
    C_grp = C.reshape(num_groups, group_size, C.shape[-1]) # Shape: (num_groups, group_size, n_cluster)
    W_hat_grp = W_hat.reshape(num_groups, group_size, W_hat.shape[-1]) # Shape: (num_groups, group_size, input_dim)
    H_grp = H.clone().to(device)
    B_grp = torch.zeros_like(W_grp).to(device)

    for i in range(num_groups):
        H_grp_diag = H_grp[i, torch.arange(d), torch.arange(d)]
        H_grp_diag = H_grp_diag.reshape(1, 1, -1)
        H_grp[i, :, :] = H_grp[i, :, :] / H_grp_diag

    cd_block_size = 128

    for k in range(cd_cycles):

        B_grp = torch.bmm(W_hat_grp - W_grp, torch.tril(H_grp, diagonal=-1))

        for start_idx in range(0, d, cd_block_size):

            end_idx = min(start_idx + cd_block_size, d)

            for update_idx in range(start_idx, end_idx):

                index = torch.arange(update_idx, update_idx + 1, device=device)
                sol = W_grp[:, :, index] - B_grp[:, :, index]

                assert sol.shape == (num_groups, group_size, 1)

                sol_dist = torch.abs(sol - C_grp) # Shape: (num_groups, group_size, n_cluster)
                min_dist, argmin_dist = sol_dist.min(dim=-1) # Shape: (num_groups, group_size)

                assignments[:, index] = argmin_dist.reshape(-1, 1)
                W_hat_grp[:, :, index] = torch.gather(C_grp, dim=-1, index=argmin_dist.unsqueeze(-1))

                if update_idx < end_idx - 1:
                    B_grp[:, :, update_idx + 1:end_idx] += torch.bmm(W_hat_grp[:, :, index] - W_grp[:, :, index], H_grp[:, index, update_idx + 1:end_idx])
                if pb is not None:
                    pb.update(1)
            
            B_grp[:, :, end_idx:] += torch.bmm(W_hat_grp[:, :, start_idx:end_idx] - W_grp[:, :, start_idx:end_idx], H_grp[:, start_idx:end_idx, end_idx:])
    if pb is not None:
        pb.close()
    
    num_changed = (assignments_prev != assignments).sum().item()
    total_assignments = assignments_prev.numel()
    percentage_changed = num_changed / total_assignments * 100
    if verbose:
        logging.info(f"Percentage of assignments changed: {percentage_changed:.2f}%")

    return assignments


def solve_pair_monotone(
    W_i: torch.Tensor,
    W_j: torch.Tensor,
    C_grp: torch.Tensor,
    e_i_current: torch.Tensor,
    e_j_current: torch.Tensor,
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    H_ii: torch.Tensor,
    H_jj: torch.Tensor,
    H_ij: torch.Tensor,
) -> PairSolveResult:
    K = C_grp.shape[-1]
    G, R = W_i.shape
    C_sorted, sorted_to_original = torch.sort(C_grp, dim=-1)
    E_i_sorted = C_sorted - W_i.unsqueeze(-1)
    E_j_sorted = C_sorted - W_j.unsqueeze(-1)

    s_i = z_i - H_ii.view(-1, 1) * e_i_current - H_ij.view(-1, 1) * e_j_current
    s_j = z_j - H_ij.view(-1, 1) * e_i_current - H_jj.view(-1, 1) * e_j_current

    ptr_i = torch.zeros((G, R), dtype=torch.long, device=C_grp.device)
    best_cost = torch.full((G, R), float("inf"), dtype=C_grp.dtype, device=C_grp.device)
    best_label_i = torch.zeros((G, R), dtype=torch.long, device=C_grp.device)
    best_label_j = torch.zeros((G, R), dtype=torch.long, device=C_grp.device)
    best_error_i = torch.zeros((G, R), dtype=C_grp.dtype, device=C_grp.device)
    best_error_j = torch.zeros((G, R), dtype=C_grp.dtype, device=C_grp.device)

    scan_descending = H_ij > 0
    for scan_pos in range(K):
        j_sorted_idx_group = torch.where(
            scan_descending,
            torch.full((G,), K - 1 - scan_pos, dtype=torch.long, device=C_grp.device),
            torch.full((G,), scan_pos, dtype=torch.long, device=C_grp.device),
        )
        j_sorted_idx = j_sorted_idx_group.view(G, 1).expand(G, R)
        e_j = torch.gather(E_j_sorted, dim=-1, index=j_sorted_idx.unsqueeze(-1)).squeeze(-1)
        target_i = -(s_i + H_ij.view(-1, 1) * e_j) / H_ii.view(-1, 1)

        for _ in range(K - 1):
            next_ptr = torch.clamp(ptr_i + 1, max=K - 1)
            e_curr = torch.gather(E_i_sorted, dim=-1, index=ptr_i.unsqueeze(-1)).squeeze(-1)
            e_next = torch.gather(E_i_sorted, dim=-1, index=next_ptr.unsqueeze(-1)).squeeze(-1)
            advance = (ptr_i < K - 1) & (torch.abs(e_next - target_i) < torch.abs(e_curr - target_i))
            if not bool(advance.any()):
                break
            ptr_i = ptr_i + advance.long()

        e_i = torch.gather(E_i_sorted, dim=-1, index=ptr_i.unsqueeze(-1)).squeeze(-1)
        label_i = torch.gather(sorted_to_original, dim=-1, index=ptr_i.unsqueeze(-1)).squeeze(-1)
        label_j = torch.gather(sorted_to_original, dim=-1, index=j_sorted_idx.unsqueeze(-1)).squeeze(-1)
        cost = (
            0.5 * H_ii.view(-1, 1) * e_i.square()
            + 0.5 * H_jj.view(-1, 1) * e_j.square()
            + H_ij.view(-1, 1) * e_i * e_j
            + s_i * e_i
            + s_j * e_j
        )
        improve = cost < best_cost
        best_cost = torch.where(improve, cost, best_cost)
        best_label_i = torch.where(improve, label_i, best_label_i)
        best_label_j = torch.where(improve, label_j, best_label_j)
        best_error_i = torch.where(improve, e_i, best_error_i)
        best_error_j = torch.where(improve, e_j, best_error_j)

    return PairSolveResult(best_label_i, best_label_j, best_error_i, best_error_j, best_cost)



@torch.no_grad()
def update_P_pair(
    W: torch.Tensor,
    H: torch.Tensor,
    labels: torch.Tensor,
    C: torch.Tensor,
    cd_cycles: int,
    verbose: bool = True,
):
    device = W.device
    W = W.to(device)
    H = H.to(device)
    C = C.to(device)
    assignments_prev = labels.to(device).long()
    assignments = assignments_prev.clone()

    assert H.ndim == 3
    assert H.shape[1] == H.shape[2]
    assert W.shape[1] == H.shape[1]
    assert W.shape[0] % H.shape[0] == 0
    diag = torch.diagonal(H, dim1=-2, dim2=-1)
    assert torch.all(diag > 0)

    num_groups = H.shape[0]
    group_size = W.shape[0] // num_groups
    d = W.shape[1]
    pairs, singleton = build_greedy_pair_matching(H)
    perm_cpu, inv_perm_cpu = build_pair_permutation(pairs, singleton, d)
    perm = perm_cpu.to(device)
    inv_perm = inv_perm_cpu.to(device)

    W_perm = W[:, perm]
    H_perm = H.index_select(1, perm).index_select(2, perm)
    assignments_perm = assignments[:, perm].contiguous()

    W_grp = W_perm.reshape(num_groups, group_size, d)
    C_grp = C.reshape(num_groups, group_size, C.shape[-1])
    assignments_grp = assignments_perm.reshape(num_groups, group_size, d)
    E = torch.gather(C_grp, dim=-1, index=assignments_grp.long()).reshape(num_groups, group_size, d) - W_grp

    pair_coord_count = len(pairs) * 2
    panel_coord_size = 128
    if panel_coord_size % 2 != 0:
        raise ValueError("pair panel coordinate size must be even")

    update_size = cd_cycles * (len(pairs) + (1 if singleton is not None else 0))
    pb = get_progress_bar(update_size, "Updating P pair") if verbose else None
    changed_pairs = 0

    for _ in range(cd_cycles):
        Z = torch.bmm(E, H_perm)

        for panel_start in range(0, pair_coord_count, panel_coord_size):
            panel_end = min(panel_start + panel_coord_size, pair_coord_count)
            if (panel_end - panel_start) % 2 != 0:
                panel_end -= 1
            if panel_end <= panel_start:
                continue

            delta_panel = torch.zeros(
                num_groups,
                group_size,
                panel_end - panel_start,
                dtype=E.dtype,
                device=device,
            )

            for i in range(panel_start, panel_end, 2):
                j = i + 1
                old_label_i = assignments_grp[:, :, i].clone()
                old_label_j = assignments_grp[:, :, j].clone()
                result = solve_pair_monotone(
                    W_grp[:, :, i],
                    W_grp[:, :, j],
                    C_grp,
                    E[:, :, i],
                    E[:, :, j],
                    Z[:, :, i],
                    Z[:, :, j],
                    H_perm[:, i, i],
                    H_perm[:, j, j],
                    H_perm[:, i, j],
                )

                delta_i = result.error_i - E[:, :, i]
                delta_j = result.error_j - E[:, :, j]
                assignments_grp[:, :, i] = result.label_i
                assignments_grp[:, :, j] = result.label_j
                E[:, :, i] = result.error_i
                E[:, :, j] = result.error_j
                delta_panel[:, :, i - panel_start] = delta_i
                delta_panel[:, :, j - panel_start] = delta_j

                changed_pairs += torch.logical_or(
                    old_label_i != result.label_i,
                    old_label_j != result.label_j,
                ).sum().item()

                if j + 1 < panel_end:
                    future = slice(j + 1, panel_end)
                    Z[:, :, future] += (
                        delta_i.unsqueeze(-1) * H_perm[:, i, future].unsqueeze(1)
                        + delta_j.unsqueeze(-1) * H_perm[:, j, future].unsqueeze(1)
                    )
                if pb is not None:
                    pb.update(1)

            if panel_end < d:
                Z[:, :, panel_end:] += torch.bmm(
                    delta_panel,
                    H_perm[:, panel_start:panel_end, panel_end:],
                )

        if singleton is not None:
            i = pair_coord_count
            H_ii = H_perm[:, i, i]
            external = Z[:, :, i] - H_ii.view(-1, 1) * E[:, :, i]
            target = -external / H_ii.view(-1, 1)
            candidates = C_grp - W_grp[:, :, i].unsqueeze(-1)
            labels_i = torch.abs(candidates - target.unsqueeze(-1)).min(dim=-1).indices
            E[:, :, i] = torch.gather(candidates, dim=-1, index=labels_i.unsqueeze(-1)).squeeze(-1)
            assignments_grp[:, :, i] = labels_i
            if pb is not None:
                pb.update(1)

    if pb is not None:
        pb.close()

    assignments = assignments_grp.reshape(W.shape[0], d)[:, inv_perm].contiguous()
    num_changed = (assignments_prev != assignments).sum().item()
    total_assignments = assignments_prev.numel()
    percentage_changed = num_changed / total_assignments * 100
    if verbose:
        logging.info("assignment solver: pair")
        logging.info("pair backend: monotone-exact")
        logging.info(f"number of pairs: {len(pairs)}")
        if singleton is not None:
            logging.info(f"singleton coordinate: {singleton}")
        logging.info(f"Percentage of assignments changed: {percentage_changed:.2f}%")
        if len(pairs) > 0:
            total_pairs = len(pairs) * num_groups * group_size * cd_cycles
            logging.info(f"Percentage of pairs changed: {changed_pairs / total_pairs * 100:.2f}%")
    return assignments


def update_P(
    W: torch.Tensor,
    H: torch.Tensor,
    labels: torch.Tensor,
    C: torch.Tensor,
    cd_cycles: int,
    verbose: bool = True,
    assignment_solver: Literal["cd", "pair"] = "pair",
):
    if assignment_solver == "cd":
        return update_P_cd(W, H, labels, C, cd_cycles=cd_cycles, verbose=verbose)
    if assignment_solver == "pair":
        return update_P_pair(
            W,
            H,
            labels,
            C,
            cd_cycles=cd_cycles,
            verbose=verbose,
        )
    raise ValueError(f"Unsupported assignment solver: {assignment_solver}")

def update_C(
    W: torch.Tensor, # Shape: (output_dim, input_dim)
    H: torch.Tensor, # Shape: (num_groups, input_dim, input_dim)
    labels: torch.Tensor, # Shape: (output_dim, input_dim)
    C: torch.Tensor, # Shape: (output_dim, n_cluster)
    iteration: int
):
    device = torch.device("cuda")
    channel_size = W.shape[0]
    input_size = H.shape[1]
    sub_channel_size = 64
    sub_input_size = 2 ** 16

    num_groups = H.shape[0]
    group_size = W.shape[0] // num_groups
    L = torch.empty_like(H)
    for i in range(num_groups):
        L[i] = torch.linalg.cholesky(H[i])

    reduced_X = L.transpose(-2, -1)

    assert channel_size // sub_channel_size >= num_groups
    assert channel_size % (sub_channel_size * num_groups) == 0

    C_hat_list = []
    pb = get_progress_bar(channel_size // sub_channel_size, "Updating centroids")
    for st_idx in range(0, channel_size, sub_channel_size):

        group_idx = st_idx // group_size
        reduced_X_blk = reduced_X[group_idx] # Shape: (input_dim, input_dim)

        end_idx = min(st_idx + sub_channel_size, channel_size)
        
        A_batch_list, b_batch_list = [], []
        labels_batch = labels[st_idx:end_idx].to(device)
        for st_idx_inp in range(0, input_size, sub_input_size):
            end_idx_inp = min(st_idx_inp + sub_input_size, input_size)
            X_batch = reduced_X_blk[st_idx_inp:end_idx_inp].to(device)
            P_batch = torch.nn.functional.one_hot(labels_batch.long(), num_classes=C.shape[-1]).float()  # (i, j, c)
            A_batch_tmp = torch.einsum('bj,ijc->ibc', X_batch, P_batch) # Shape: (output_dim, num_samples, n_cluster)
            b_batch_tmp = torch.einsum('bj,ij->ib', X_batch, W[st_idx:end_idx]).unsqueeze(-1) # Shape: (output_dim, num_samples, 1)
            A_batch_list.append(A_batch_tmp)
            b_batch_list.append(b_batch_tmp)

        A_batch = torch.cat(A_batch_list, dim=1)
        b_batch = torch.cat(b_batch_list, dim=1)

        ######### REGULARIZATION #########
        lambda_reg = 1e-7
        # Get dimensions
        batch_size, num_samples, n_cluster = A_batch.shape
        dtype, device = A_batch.dtype, A_batch.device

        # Create sqrt(lambda) * I matrix for regularization
        sqrt_lambda = torch.sqrt(torch.tensor(lambda_reg, dtype=dtype, device=device))
        I = sqrt_lambda * torch.eye(n_cluster, dtype=dtype, device=device).unsqueeze(0).expand(batch_size, -1, -1)

        # Augment A_batch and b_batch for regularization
        A_batch = torch.cat([A_batch.transpose(1, 2), I], dim=2).transpose(1, 2)
        zeros = torch.zeros((batch_size, n_cluster, 1), dtype=dtype, device=device)
        b_batch = torch.cat([b_batch, zeros], dim=1)

        ##################################

        # Compute the least squares solution for this batch
        C_hat_batch = torch.linalg.lstsq(A_batch, b_batch).solution  # Shape: (output_dim, num_samples, n_cluster)
        # Check if C_hat_batch has nan values
        if torch.isnan(C_hat_batch).any():
            logging.error(f"NaN values detected in C_hat_batch for indices {st_idx} to {end_idx}")
            exit()

        C_hat_batch = C_hat_batch.squeeze(-1)  # Shape: (output_dim, num_samples, n_cluster)
        
        C_hat_list.append(C_hat_batch)
        pb.update(1)
    pb.close()

    C = torch.cat(C_hat_list, dim=0).cpu()

    return C

def train_least_squares(
    W: np.ndarray, # Shape: (output_dim, input_dim)
    init_labels: np.ndarray, # Shape: (output_dim, input_dim)
    init_centroids: np.ndarray, # Shape: (output_dim, n_cluster)
    H: np.ndarray, # Shape: (num_groups, input_dim, input_dim)
    num_iterations: int = 3,
    cd_cycles: int = 4,
    assignment_solver: Literal["cd", "pair"] = "pair",
) -> Tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda")

    labels = torch.tensor(init_labels, dtype=torch.int8, device="cpu")
    C = torch.tensor(init_centroids, dtype=torch.float32, device="cpu")
    W = torch.tensor(W, dtype=torch.float32).to(device)
    H = torch.tensor(H, dtype=torch.float32).to(device)

    diag = torch.arange(H.shape[1], device=device)
    for i in range(H.shape[0]):
        avg_diag = torch.mean(torch.diag(H[i]))
        damp, prev_damp = 1e-5, 0.
        while True:
            try:
                torch.linalg.cholesky(H[i])
                logging.info(f"{i+1}-th H is PD, dampening factor={prev_damp:.2e}")
                break
            except Exception as e:
                print(e)
                logging.info(f"{i+1}-th H is not PD, try dampening with factor={damp:.2e}")
                H[i, diag, diag] += (damp - prev_damp) * avg_diag
                prev_damp = damp
                damp *= 10
                if damp > 1e0:
                    exit()

    best_obj_value = objective_function(W, H, labels, C).item()
    best_labels, best_C = labels.detach().cpu().clone(), C.detach().cpu().clone()
    logging.info(f"Initial objective: {best_obj_value:.6f}")

    log_dict = {"objective": [], "iteration": []}
    log_dict["objective"].append(best_obj_value)
    log_dict["iteration"].append(0)

    for iteration in range(num_iterations):
        start_time = time.time()

        ######### Update P #########
        if iteration > 0:
            labels = update_P(
                W,
                H,
                labels,
                C,
                cd_cycles=cd_cycles,
                assignment_solver=assignment_solver,
            )

        # Compute objective value for logging
        obj_value = objective_function(W, H, labels, C).item()
        logging.info(f"Iteration {iteration + 1} (P update): Objective: {obj_value:.4f}")
        log_dict["objective"].append(obj_value)
        log_dict["iteration"].append(iteration + 1)


        ######### Update C #########
        C = update_C(W, H, labels, C, iteration)

        # Check if the objective value improved
        current_obj_value = objective_function(W, H, labels, C).item()
        log_dict["objective"].append(current_obj_value)
        log_dict["iteration"].append(iteration + 1)
        if current_obj_value < best_obj_value:
            best_obj_value = current_obj_value
            best_labels, best_C = labels.detach().cpu().clone(), C.detach().cpu().clone()
            logging.info(f"Iteration {iteration + 1} (C update): Objective: {current_obj_value:.4f} | Improved and using this one.")
        else:
            logging.info(f"Iteration {iteration + 1} (C update): Objective: {current_obj_value:.4f} | Not improved. Using previous best values.")
            labels, C = best_labels, best_C
            break  # Early stopping

        end_time = time.time()

        logging.info(f"Iteration {iteration + 1} / {num_iterations} completed. "
                     f"Update time: {end_time - start_time:.2f} sec")

    end_time = time.time()
    logging.info(f"Least squares training time: {end_time - start_time:.2f} seconds")

    labels = labels.detach().cpu().numpy()
    C = C.detach().cpu().numpy().astype(np.float32)

    return labels, C, log_dict

def free_and_log_memory():
    torch.cuda.empty_cache()
    import gc; gc.collect()
    logging.info(f"Left memory: {torch.cuda.mem_get_info()[0] / (10 ** 9)} GB")

def seed_layer(
    l: int,
    module_names: List[str],
    layer_modules: List[np.ndarray],
    layer_init_labels: List[np.ndarray],
    layer_init_centroids: List[np.ndarray],
    layer_hessian: List[np.ndarray],
    seed_bit: int,
    group_count: int,
    num_iterations: int = 3,
    cd_cycles: int = 4,
    assignment_solver: Literal["cd", "pair"] = "pair",
) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
    lut_by_bit_by_module = []
    parent_weights_by_modules = []
    log_dict_by_module = []

    n_cluster = 2**seed_bit

    for m_idx in range(len(layer_modules)):
        module_name = module_names[m_idx]
        logging.info(f"Quantizing Layer [{l}], Module [{module_name}] ({m_idx + 1}/{len(layer_modules)})")

        module_weight = layer_modules[m_idx]
        module_init_labels = layer_init_labels[m_idx]
        module_init_centroids = layer_init_centroids[m_idx]
        module_hessian = layer_hessian[m_idx]

        assert group_count == 1, "Group-wise quantization is not supported yet"

        output_dim = module_weight.shape[0]
        input_dim = module_weight.shape[1]

        lut_by_bit = []
        for bit in range(seed_bit, seed_bit + 1):
            lut_by_bit.append(
                np.empty((output_dim, 1, 2**bit), dtype=np.float32)
            )

        init_labels = module_init_labels.reshape(output_dim, input_dim)
        init_centroids = module_init_centroids.reshape(output_dim, n_cluster) # Shape: (output_dim, n_cluster)
        reshaped_module_weight = module_weight.reshape(output_dim, input_dim) # Shape: (output_dim, input_dim)

        labels, C, log_dict = train_least_squares(
            reshaped_module_weight,
            init_labels,
            init_centroids,
            module_hessian,
            num_iterations=num_iterations,
            cd_cycles=cd_cycles,
            assignment_solver=assignment_solver,
        )

        labels = labels.astype(np.uint8) # Shape: (output_dim, input_dim)
        labels = labels.reshape(output_dim, 1, input_dim) # Shape: (output_dim, 1, input_dim)
        C = C.reshape(output_dim, 1, n_cluster) # Shape: (output_dim, 1, n_cluster)

        for k, bit in enumerate(range(seed_bit, seed_bit + 1)):
            lut_by_bit[k] = C

        parent_weights_by_modules.append(labels)
        lut_by_bit_by_module.append(lut_by_bit)
        log_dict_by_module.append(log_dict)

    return lut_by_bit_by_module, parent_weights_by_modules, log_dict_by_module


# Minimal inline function to ensure shape is (num_groups, input_dim, input_dim)
def fix_hessian_shape(H: torch.Tensor) -> torch.Tensor:
    if H.shape[1] == H.shape[2]:
        # Already (num_groups, input_dim, input_dim)
        return H
    elif H.shape[0] == H.shape[1]:
        # Then it's (input_dim, input_dim, num_groups), so permute
        return H.permute(2, 0, 1)
    else:
        raise ValueError(f"Invalid Hessian shape: {H.shape}")


def get_layer_loader(analyzer, module_names, initialization_path, hessians_path, seed_precision):
    def layer_loader(l):
        # Load the initialization data (labels and centroids)
        init_labels_file_name = os.path.join(initialization_path, "weights", f"l{l}.pt")
        init_labels = torch.load(init_labels_file_name)
        init_centroids_file_name = os.path.join(initialization_path, f"lut_{seed_precision}", f"l{l}.pt")
        init_centroids = torch.load(init_centroids_file_name)
        hessian_file_name = os.path.join(hessians_path, f"l{l}.pt")
        hessian = torch.load(hessian_file_name)

        # Organize the data by module
        init_labels_layer = [
            init_labels[name] for name in module_names
        ]
        init_centroids_layer = [
            init_centroids[name].astype(np.float32) for name in module_names
        ]
        hessian_layer = [
            fix_hessian_shape(hessian[name]).float().numpy() for name in module_names
        ]
        model_layer = [
            analyzer.get_layer_weights(l)[name].float().numpy()
            for name in module_names
        ]
        return module_names, model_layer, init_labels_layer, init_centroids_layer, hessian_layer

    return layer_loader


def _save_results(
    parent_parameters_path,
    seed_precision,
    parent_precision,
    module_names,
    luts_by_bit_by_module,
    parent_weights,
    log_dict,
    l,
):
    # Note that it is important to cast the luts to fp16 before saving them.
    for i, bit in enumerate(range(seed_precision, parent_precision + 1)):
        output_lut_file_name = f"{parent_parameters_path}/lut_{bit}/l{l}.pt"
        output_log_dict_file_name = f"{parent_parameters_path}/lut_{bit}/log_dict{l}.pt"
        os.makedirs(os.path.dirname(output_lut_file_name), exist_ok=True)
        lut_dict = {}
        module_name_to_log_dict = {}
        for j in range(len(module_names)):
            lut_dict[module_names[j]] = luts_by_bit_by_module[j][i].astype(np.float16)
            module_name_to_log_dict[module_names[j]] = log_dict[j]
        torch.save(lut_dict, output_lut_file_name)
        torch.save(module_name_to_log_dict, output_log_dict_file_name)

    parent_weight_dict = {
        module_names[j]: parent_weights[j].astype(np.uint8)
        for j in range(len(module_names))
    }

    output_weights_layer_file_name = f"{parent_parameters_path}/weights/l{l}.pt"
    os.makedirs(os.path.dirname(output_weights_layer_file_name), exist_ok=True)
    torch.save(parent_weight_dict, output_weights_layer_file_name)


def get_saver(parent_parameters_path, seed_precision, parent_precision, module_names):
    """Returns a function that saves the results for a given layer"""

    def save_results(luts_by_bit_by_module, parent_weights, log_dict, l):
        return _save_results(
            parent_parameters_path,
            seed_precision,
            parent_precision,
            module_names,
            luts_by_bit_by_module,
            parent_weights,
            log_dict,
            l,
        )

    return save_results


def load_progress(
    parent_parameters_path, seed_precision, parent_precision, layer_count
):
    # Check if the layer has already been processed
    todo_ran = []
    processed_ran = []
    for l in range(layer_count):
        if all(
            [
                os.path.exists(f"{parent_parameters_path}/lut_{bit}/l{l}.pt")
                for bit in range(seed_precision, parent_precision + 1)
            ]
        ) and os.path.exists(f"{parent_parameters_path}/weights/l{l}.pt"):
            processed_ran.append(l)
        else:
            todo_ran.append(l)
    return todo_ran, processed_ran


def seed(
    analyzer: ModelAnalyzer,
    module_names: List[str],
    initialization_path: str,
    hessians_path: str,
    output_folder: str,
    seed_precision: int,
    cpu_count: int = None,
    num_iterations: int = 3,
    cd_cycles: int = 4,
    assignment_solver: Literal["cd", "pair"] = "pair",
    sub_qlayer: Tuple[int, int] = None,
):
    group_count = 1

    if cpu_count is None:
        cpu_count = int(os.popen("nproc").read().strip())
    # Determine IO and threading settings based on the number of cores
    if cpu_count >= 8:
        pipelined_io = True
        io_workers = 2 if cpu_count >= 64 else 1
    else:
        pipelined_io = False
        io_workers = 0  # No separate IO workers needed for non-pipelined IO

    logging.info(f"Using {cpu_count} cores for parallelization")

    logging.info(f"Seeding for {seed_precision}-bit")

    layers_to_process, completed_layers = load_progress(
        output_folder, seed_precision, seed_precision, analyzer.num_layers
    )

    if sub_qlayer:
        layers_to_process = [i for i in layers_to_process if i in range(sub_qlayer[0], sub_qlayer[1])]

    if completed_layers:
        logging.info(
            f"The following layers will be skipped as they have already been processed:\n{completed_layers}"
        )
        logging.info(
            f"To reprocess these layers, delete the corresponding files in {output_folder}"
        )

    if not layers_to_process:
        logging.info("All layers have already been processed. Exiting...")
        return

    logging.info(f"Quantizing layers {layers_to_process}")

    layer_loader = get_layer_loader(
        analyzer, module_names, initialization_path, hessians_path, seed_precision
    )
    layer_saver = get_saver(
        output_folder, seed_precision, seed_precision, module_names
    )

    if pipelined_io:
        with ThreadPoolExecutor(max_workers=io_workers) as io_executor:
            pb = get_progress_bar(len(layers_to_process), "Quantizing layers...")
            for l in layers_to_process:
                if l == layers_to_process[0]:
                    future_load = io_executor.submit(layer_loader, l)

                module_names, model_layer, init_labels_layer, init_centroids_layer, hessian_layer = future_load.result()

                if l != layers_to_process[-1]:
                    future_load = io_executor.submit(layer_loader, l + 1)

                luts_by_bit_by_module, parent_weights, log_dict = seed_layer(
                    l,
                    module_names,
                    model_layer,
                    init_labels_layer,
                    init_centroids_layer,
                    hessian_layer,
                    seed_precision,
                    group_count,
                    num_iterations=num_iterations,
                    cd_cycles=cd_cycles,
                    assignment_solver=assignment_solver,
                )

                io_executor.submit(
                    layer_saver, luts_by_bit_by_module, parent_weights, log_dict, l
                )
                pb.update(1)
            pb.close()
            logging.info("Waiting for IO to finish...")
    else:
        pb = get_progress_bar(len(layers_to_process), "Quantizing layers...")
        for l in layers_to_process:
            module_names, model_layer, init_labels_layer, init_centroids_layer, hessian_layer = layer_loader(l)

            luts_by_bit_by_module, parent_weights, log_dict = seed_layer(
                l,
                module_names,
                model_layer,
                init_labels_layer,
                init_centroids_layer,
                hessian_layer,
                seed_precision,
                group_count,
                num_iterations=num_iterations,
                cd_cycles=cd_cycles,
                assignment_solver=assignment_solver,
            )

            layer_saver(luts_by_bit_by_module, parent_weights, log_dict, l)
            pb.update(1)
        pb.close()
