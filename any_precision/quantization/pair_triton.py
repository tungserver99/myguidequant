import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised only when Triton is installed.
    triton = None
    tl = None


def triton_pair_available() -> bool:
    return triton is not None and tl is not None


if triton is not None:

    @triton.jit
    def _pair_monotone_kernel(
        W_I,
        W_J,
        C_SORTED,
        SORTED_TO_ORIGINAL,
        E_I_CURRENT,
        E_J_CURRENT,
        Z_I,
        Z_J,
        H_II,
        H_JJ,
        H_IJ,
        LABEL_I,
        LABEL_J,
        ERROR_I,
        ERROR_J,
        COST,
        N: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        group = offs // R
        code_base = offs * K

        w_i = tl.load(W_I + offs, mask=mask, other=0.0)
        w_j = tl.load(W_J + offs, mask=mask, other=0.0)
        e_i_current = tl.load(E_I_CURRENT + offs, mask=mask, other=0.0)
        e_j_current = tl.load(E_J_CURRENT + offs, mask=mask, other=0.0)
        z_i = tl.load(Z_I + offs, mask=mask, other=0.0)
        z_j = tl.load(Z_J + offs, mask=mask, other=0.0)
        h_ii = tl.load(H_II + group, mask=mask, other=1.0)
        h_jj = tl.load(H_JJ + group, mask=mask, other=1.0)
        h_ij = tl.load(H_IJ + group, mask=mask, other=0.0)

        s_i = z_i - h_ii * e_i_current - h_ij * e_j_current
        s_j = z_j - h_ij * e_i_current - h_jj * e_j_current

        ptr_i = tl.zeros((BLOCK_N,), dtype=tl.int64)
        best_cost = tl.full((BLOCK_N,), float("inf"), dtype=tl.float32)
        best_label_i = tl.zeros((BLOCK_N,), dtype=tl.int64)
        best_label_j = tl.zeros((BLOCK_N,), dtype=tl.int64)
        best_error_i = tl.zeros((BLOCK_N,), dtype=tl.float32)
        best_error_j = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for scan_pos in range(K):
            j_idx = tl.where(h_ij > 0.0, K - 1 - scan_pos, scan_pos)
            c_j = tl.load(C_SORTED + code_base + j_idx, mask=mask, other=0.0)
            e_j = c_j - w_j
            target_i = -(s_i + h_ij * e_j) / h_ii

            for _ in range(K - 1):
                next_ptr = tl.minimum(ptr_i + 1, K - 1)
                c_curr = tl.load(C_SORTED + code_base + ptr_i, mask=mask, other=0.0)
                c_next = tl.load(C_SORTED + code_base + next_ptr, mask=mask, other=0.0)
                e_curr = c_curr - w_i
                e_next = c_next - w_i
                advance = (ptr_i < K - 1) & (tl.abs(e_next - target_i) < tl.abs(e_curr - target_i))
                ptr_i += advance.to(tl.int64)

            c_i = tl.load(C_SORTED + code_base + ptr_i, mask=mask, other=0.0)
            e_i = c_i - w_i
            cost = (
                0.5 * h_ii * e_i * e_i
                + 0.5 * h_jj * e_j * e_j
                + h_ij * e_i * e_j
                + s_i * e_i
                + s_j * e_j
            )
            improve = cost < best_cost
            label_i = tl.load(SORTED_TO_ORIGINAL + code_base + ptr_i, mask=mask, other=0)
            label_j = tl.load(SORTED_TO_ORIGINAL + code_base + j_idx, mask=mask, other=0)

            best_cost = tl.where(improve, cost, best_cost)
            best_label_i = tl.where(improve, label_i, best_label_i)
            best_label_j = tl.where(improve, label_j, best_label_j)
            best_error_i = tl.where(improve, e_i, best_error_i)
            best_error_j = tl.where(improve, e_j, best_error_j)

        tl.store(LABEL_I + offs, best_label_i, mask=mask)
        tl.store(LABEL_J + offs, best_label_j, mask=mask)
        tl.store(ERROR_I + offs, best_error_i, mask=mask)
        tl.store(ERROR_J + offs, best_error_j, mask=mask)
        tl.store(COST + offs, best_cost, mask=mask)


    @triton.jit
    def _pair_bruteforce_k2_kernel(
        W_I,
        W_J,
        C_GRP,
        E_I_CURRENT,
        E_J_CURRENT,
        Z_I,
        Z_J,
        H_II,
        H_JJ,
        H_IJ,
        LABEL_I,
        LABEL_J,
        ERROR_I,
        ERROR_J,
        COST,
        N: tl.constexpr,
        R: tl.constexpr,
        K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        group = offs // R
        code_base = offs * K

        w_i = tl.load(W_I + offs, mask=mask, other=0.0)
        w_j = tl.load(W_J + offs, mask=mask, other=0.0)
        e_i_current = tl.load(E_I_CURRENT + offs, mask=mask, other=0.0)
        e_j_current = tl.load(E_J_CURRENT + offs, mask=mask, other=0.0)
        z_i = tl.load(Z_I + offs, mask=mask, other=0.0)
        z_j = tl.load(Z_J + offs, mask=mask, other=0.0)
        h_ii = tl.load(H_II + group, mask=mask, other=1.0)
        h_jj = tl.load(H_JJ + group, mask=mask, other=1.0)
        h_ij = tl.load(H_IJ + group, mask=mask, other=0.0)

        s_i = z_i - h_ii * e_i_current - h_ij * e_j_current
        s_j = z_j - h_ij * e_i_current - h_jj * e_j_current

        best_cost = tl.full((BLOCK_N,), float("inf"), dtype=tl.float32)
        best_label_i = tl.zeros((BLOCK_N,), dtype=tl.int64)
        best_label_j = tl.zeros((BLOCK_N,), dtype=tl.int64)
        best_error_i = tl.zeros((BLOCK_N,), dtype=tl.float32)
        best_error_j = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k in range(K):
            c_i = tl.load(C_GRP + code_base + k, mask=mask, other=0.0)
            e_i = c_i - w_i
            base_cost_i = 0.5 * h_ii * e_i * e_i + s_i * e_i
            for l in range(K):
                c_j = tl.load(C_GRP + code_base + l, mask=mask, other=0.0)
                e_j = c_j - w_j
                cost = (
                    base_cost_i
                    + 0.5 * h_jj * e_j * e_j
                    + h_ij * e_i * e_j
                    + s_j * e_j
                )
                improve = cost < best_cost
                best_cost = tl.where(improve, cost, best_cost)
                best_label_i = tl.where(improve, k, best_label_i)
                best_label_j = tl.where(improve, l, best_label_j)
                best_error_i = tl.where(improve, e_i, best_error_i)
                best_error_j = tl.where(improve, e_j, best_error_j)

        tl.store(LABEL_I + offs, best_label_i, mask=mask)
        tl.store(LABEL_J + offs, best_label_j, mask=mask)
        tl.store(ERROR_I + offs, best_error_i, mask=mask)
        tl.store(ERROR_J + offs, best_error_j, mask=mask)
        tl.store(COST + offs, best_cost, mask=mask)


def solve_pair_monotone_triton(
    W_i: torch.Tensor,
    W_j: torch.Tensor,
    C_sorted: torch.Tensor,
    sorted_to_original: torch.Tensor,
    e_i_current: torch.Tensor,
    e_j_current: torch.Tensor,
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    H_ii: torch.Tensor,
    H_jj: torch.Tensor,
    H_ij: torch.Tensor,
    block_n: int = 128,
):
    if not triton_pair_available():
        raise RuntimeError("Triton is not available")
    if not W_i.is_cuda:
        raise RuntimeError("Triton pair solver requires CUDA tensors")

    G, R = W_i.shape
    K = C_sorted.shape[-1]
    N = G * R

    W_i = W_i.contiguous()
    W_j = W_j.contiguous()
    C_sorted = C_sorted.contiguous()
    sorted_to_original = sorted_to_original.contiguous()
    e_i_current = e_i_current.contiguous()
    e_j_current = e_j_current.contiguous()
    z_i = z_i.contiguous()
    z_j = z_j.contiguous()
    H_ii = H_ii.contiguous()
    H_jj = H_jj.contiguous()
    H_ij = H_ij.contiguous()

    label_i = torch.empty((G, R), dtype=torch.long, device=W_i.device)
    label_j = torch.empty((G, R), dtype=torch.long, device=W_i.device)
    error_i = torch.empty((G, R), dtype=W_i.dtype, device=W_i.device)
    error_j = torch.empty((G, R), dtype=W_i.dtype, device=W_i.device)
    cost = torch.empty((G, R), dtype=W_i.dtype, device=W_i.device)

    grid = (triton.cdiv(N, block_n),)
    _pair_monotone_kernel[grid](
        W_i,
        W_j,
        C_sorted,
        sorted_to_original,
        e_i_current,
        e_j_current,
        z_i,
        z_j,
        H_ii,
        H_jj,
        H_ij,
        label_i,
        label_j,
        error_i,
        error_j,
        cost,
        N,
        R,
        K,
        BLOCK_N=block_n,
    )

    return label_i, label_j, error_i, error_j, cost



def solve_pair_bruteforce_k2_triton(
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
    block_n: int = 128,
):
    if not triton_pair_available():
        raise RuntimeError("Triton is not available")
    if not W_i.is_cuda:
        raise RuntimeError("Triton pair solver requires CUDA tensors")

    G, R = W_i.shape
    K = C_grp.shape[-1]
    N = G * R

    W_i = W_i.contiguous()
    W_j = W_j.contiguous()
    C_grp = C_grp.contiguous()
    e_i_current = e_i_current.contiguous()
    e_j_current = e_j_current.contiguous()
    z_i = z_i.contiguous()
    z_j = z_j.contiguous()
    H_ii = H_ii.contiguous()
    H_jj = H_jj.contiguous()
    H_ij = H_ij.contiguous()

    label_i = torch.empty((G, R), dtype=torch.long, device=W_i.device)
    label_j = torch.empty((G, R), dtype=torch.long, device=W_i.device)
    error_i = torch.empty((G, R), dtype=W_i.dtype, device=W_i.device)
    error_j = torch.empty((G, R), dtype=W_i.dtype, device=W_i.device)
    cost = torch.empty((G, R), dtype=W_i.dtype, device=W_i.device)

    grid = (triton.cdiv(N, block_n),)
    _pair_bruteforce_k2_kernel[grid](
        W_i,
        W_j,
        C_grp,
        e_i_current,
        e_j_current,
        z_i,
        z_j,
        H_ii,
        H_jj,
        H_ij,
        label_i,
        label_j,
        error_i,
        error_j,
        cost,
        N,
        R,
        K,
        BLOCK_N=block_n,
    )

    return label_i, label_j, error_i, error_j, cost
