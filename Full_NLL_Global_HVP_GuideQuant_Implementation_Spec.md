# Full-NLL Global HVP Curvature Using the Existing GuidedQuant Engine

## 0. Mục tiêu và quyết định thiết kế cuối cùng

Tài liệu này là **implementation specification** để thay curvature empirical-Fisher của GuidedQuant bằng curvature của **ground-truth next-token NLL**, trong khi giữ nguyên hạ tầng dựng Hessian và solver LNQ hiện có.

Quyết định cuối cùng là:

\[
\boxed{
\text{Ground-truth NLL}
\rightarrow
\text{full global HVP}
\rightarrow
V\odot HV
\rightarrow
\text{GuidedQuant output grouping}
\rightarrow
X^\top\operatorname{Diag}(s)X
\rightarrow
\text{unchanged LNQ}
}
\]

Không tách GGN và residual curvature thành hai đường tính riêng trong implementation mặc định.

Phân rã

\[
H_{\mathrm{NLL}}=G_{\mathrm{GGN}}+R
\]

chỉ dùng để giải thích ý nghĩa toán học của Hessian. Khi lấy HVP trực tiếp từ ground-truth NLL, autograd đã trả về đồng thời cả hai thành phần:

\[
H_{\mathrm{NLL}}V=(G_{\mathrm{GGN}}+R)V.
\]

Do đó implementation **không được**:

- tính GGN riêng;
- tính residual \(R\) riêng;
- cộng thêm GGN vào kết quả full-NLL HVP;
- thiết kế một Hessian builder mới thay cho `SaliencyEngine`;
- chạy HVP riêng cho từng layer hoặc từng output group.

Phần cần thay duy nhất về mặt kiến trúc là **saliency producer**:

```text
GuidedQuant gốc:
    output-gradient squared
        -> saliency cache
        -> SaliencyEngine
        -> grouped X^T Diag(s) X
        -> LNQ

Phương pháp mới:
    V * (H_NLL V)
        -> saliency cache cùng format
        -> SaliencyEngine giữ nguyên
        -> grouped X^T Diag(s) X giữ nguyên
        -> LNQ giữ nguyên
```

---

# 1. Ký hiệu

Giả sử mô hình có \(L\) target linear modules. Với module \(l\):

\[
Z^{(l)}=X^{(l)}{W^{(l)}}^\top,
\]

trong đó:

\[
X^{(l)}\in\mathbb R^{B\times T\times d_{\mathrm{in}}^{(l)}},
\]

\[
W^{(l)}\in\mathbb R^{d_{\mathrm{out}}^{(l)}\times d_{\mathrm{in}}^{(l)}},
\]

\[
Z^{(l)}\in\mathbb R^{B\times T\times d_{\mathrm{out}}^{(l)}}.
\]

PyTorch lưu weight của `nn.Linear` theo shape:

```text
[out_features, in_features]
```

nên:

- output channel tương ứng với **row của weight**;
- GuidedQuant grouping phải được thực hiện trên dimension `out_features`;
- Hessian truyền vào LNQ có dimension theo `in_features`.

Ký hiệu:

- \(B\): calibration micro-batch size;
- \(T\): sequence length;
- \(x_{b,t}^{(l)}\): actual input của module \(l\);
- \(z_{b,t,j}^{(l)}\): output tại channel \(j\);
- \(e_j^{(l)}=\hat w_j^{(l)}-w_j^{(l)}\): quantization error của một weight row;
- \(J_k^{(l)}\): tập output channels thuộc group \(k\);
- \(G_l\): số output groups của module \(l\).

Trong implementation GuidedQuant hiện tại, cùng một `num_groups` thường được dùng cho mọi module. Phải assert:

```python
out_features % num_groups == 0
```

cho từng target module.

---

# 2. Ground-truth next-token NLL

Với input token sequence và ground-truth next token labels:

\[
\boxed{
\mathcal L_{\mathrm{NLL}}
=
-\sum_{(b,t)\in\mathcal M}
\log p_W(y_{b,t}\mid x_{b,<t})
}
\]

trong đó \(\mathcal M\) là tập valid shifted-token positions.

## 2.1. Dùng `sum`, không dùng mean theo từng micro-batch

Implementation mặc định phải dùng:

```python
loss_nll = torch.nn.functional.cross_entropy(
    shift_logits.float().reshape(-1, vocab_size),
    shift_labels.reshape(-1),
    ignore_index=-100,
    reduction="sum",
)
```

Lý do:

- mỗi micro-batch có thể có số valid tokens khác nhau;
- `mean` theo từng micro-batch tạo scale khác nhau giữa các batch;
- `sum` cho phép cộng curvature chính xác qua calibration data;
- nếu cần mean curvature, chỉ chia một lần ở cuối cho tổng số valid tokens.

Nếu calibration data luôn là fixed-length sequences không padding, `labels=input_ids` vẫn phải được shift đúng như causal LM loss:

```python
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
```

## 2.2. Không kế thừa arbitrary gradient multiplier

Không dùng:

```python
grad * 1e3
```

trong full-NLL HVP path.

Multiplier này thuộc implementation saliency cũ của GuidedQuant và không cần thiết cho HVP thật. Nếu signal yếu trong BF16/FP16, tăng precision của curvature path thay vì nhân một constant tùy ý.

---

# 3. Khai triển Hessian của NLL

Gọi logits của mô hình là:

\[
f=f_W(x),
\]

và softmax probability:

\[
p=\operatorname{softmax}(f).
\]

Với một target token \(y\):

\[
\ell_{\mathrm{NLL}}=-\log p_y.
\]

Gradient theo logits:

\[
\nabla_f\ell_{\mathrm{NLL}}=p-e_y.
\]

Hessian theo logits:

\[
\nabla_f^2\ell_{\mathrm{NLL}}
=
\operatorname{Diag}(p)-pp^\top.
\]

Xét output \(Z^{(l)}\) của target linear module \(l\). Đặt:

\[
J_l=\frac{\partial f}{\partial Z^{(l)}}.
\]

Hessian NLL theo \(Z^{(l)}\) có phân rã:

\[
\boxed{
\mathcal H_{ll}
=
\nabla^2_{\operatorname{vec}(Z^{(l)})}
\mathcal L_{\mathrm{NLL}}
=
G_l+R_l
}
\]

với:

\[
G_l
=
J_l^\top
\left[\operatorname{Diag}(p)-pp^\top\right]
J_l,
\]

và:

\[
R_l
=
\sum_v
\left(p_v-\mathbf 1[v=y]\right)
\nabla^2_{Z^{(l)}}f_v.
\]

Trong đó:

- \(G_l\) là generalized Gauss–Newton curvature;
- \(R_l\) là residual/model curvature do suffix network phi tuyến phía sau module \(l\).

## 3.1. Phân rã này không phải yêu cầu implementation

Nếu tính trực tiếp:

\[
g_l
=
\nabla_{Z^{(l)}}\mathcal L_{\mathrm{NLL}},
\]

sau đó differentiate \(g_l\) lần nữa, autograd trả về full Hessian:

\[
\nabla_{Z^{(l)}}g_l
=
G_l+R_l.
\]

Vì vậy:

\[
\boxed{
\text{Full-NLL HVP tự động giữ cả GGN và residual curvature.}
}
\]

Không cần đồng bộ hai estimator vì chỉ có **một estimator duy nhất**.

---

# 4. Taylor expansion và mục tiêu LNQ

Perturb output của module \(l\):

\[
Z^{(l)}\rightarrow Z^{(l)}+\Delta Z^{(l)}.
\]

Taylor bậc hai:

\[
\begin{aligned}
\mathcal L_{\mathrm{NLL}}(Z^{(l)}+\Delta Z^{(l)})
\approx\;&
\mathcal L_{\mathrm{NLL}}(Z^{(l)})
+
\left\langle
\nabla_{Z^{(l)}}\mathcal L_{\mathrm{NLL}},
\Delta Z^{(l)}
\right\rangle
\\
&+
\frac12
\operatorname{vec}(\Delta Z^{(l)})^\top
\mathcal H_{ll}
\operatorname{vec}(\Delta Z^{(l)}).
\end{aligned}
\]

LNQ hiện tại giải quadratic objective không có linear term. Phương pháp này giữ phần quadratic:

\[
\Delta\mathcal L^{(l)}
\approx
\frac12
\operatorname{vec}(\Delta Z^{(l)})^\top
\mathcal H_{ll}
\operatorname{vec}(\Delta Z^{(l)}).
\]

Đây là approximation stack có chủ đích:

1. dùng end-to-end ground-truth NLL;
2. giữ quadratic term;
3. bỏ first-order term để tương thích LNQ hiện tại;
4. estimate diagonal activation curvature;
5. share diagonal curvature theo output groups;
6. project về nonnegative để tạo PSD weight-space Hessian;
7. dùng solver LNQ hiện tại.

---

# 5. Full global HVP cho tất cả target modules

## 5.1. Ghép các target outputs

Khái niệm hóa tất cả target outputs thành:

\[
\mathbf z
=
\begin{bmatrix}
\operatorname{vec}(Z^{(1)})\\
\vdots\\
\operatorname{vec}(Z^{(L)})
\end{bmatrix}.
\]

Đặt Jacobian của gradient field:

\[
\mathbb H
=
\frac{\partial}{\partial\mathbf z}
\begin{bmatrix}
\operatorname{vec}(g_1)\\
\vdots\\
\operatorname{vec}(g_L)
\end{bmatrix}
=
\begin{bmatrix}
\mathcal H_{11}&\cdots&\mathcal H_{1L}\\
\vdots&\ddots&\vdots\\
\mathcal H_{L1}&\cdots&\mathcal H_{LL}
\end{bmatrix}.
\]

Mục tiêu của module \(l\) là:

\[
\operatorname{diag}(\mathcal H_{ll}).
\]

Không materialize bất kỳ full Hessian block nào.

## 5.2. First reverse pass

Tính gradient NLL theo output của tất cả target modules cùng lúc:

\[
g_l
=
\nabla_{Z^{(l)}}\mathcal L_{\mathrm{NLL}}.
\]

PyTorch:

```python
grads_z = torch.autograd.grad(
    outputs=loss_nll,
    inputs=layer_outputs,
    create_graph=True,
    retain_graph=True,
    allow_unused=False,
)
```

Yêu cầu:

- `layer_outputs` chứa actual output tensor của từng target `nn.Linear` trong forward graph;
- không `.detach()`;
- không chuyển output sang CPU;
- không cần `retain_grad()` nếu dùng `torch.autograd.grad`;
- thứ tự `layer_outputs` phải deterministic và đi kèm metadata `(layer_idx, module_name)`.

## 5.3. Independent Rademacher probes

Với probe \(r\), sample một tensor riêng cho từng module:

\[
V_r^{(l)}
\in
\{-1,+1\}^{\operatorname{shape}(Z^{(l)})}.
\]

Yêu cầu:

\[
\mathbb E[V_{r,i}^{(l)}V_{r,q}^{(m)}]
=
0
\quad\text{khi }(l,i)\neq(m,q).
\]

Không reuse cùng sign pattern bằng broadcast giữa modules.

Seed nên được derive từ:

```text
base_seed, calibration_batch_id, probe_id, layer_id, module_id
```

## 5.4. Một global HVP cho mỗi probe

Về toán học, tạo scalar:

\[
A_r
=
\sum_{l=1}^L
\left\langle g_l,V_r^{(l)}\right\rangle.
\]

Sau đó:

\[
U_r^{(l)}
=
\nabla_{Z^{(l)}}A_r.
\]

Dưới dạng block:

\[
U_r^{(l)}
=
\sum_m\mathcal H_{lm}V_r^{(m)}.
\]

### Implementation khuyến nghị

Không cần tự cộng scalar giữa nhiều GPU. Dùng trực tiếp multi-output VJP của `torch.autograd.grad`:

```python
hvps = torch.autograd.grad(
    outputs=grads_z,
    inputs=layer_outputs,
    grad_outputs=probes,
    create_graph=False,
    retain_graph=(probe_idx + 1 < num_probes),
    allow_unused=False,
)
```

Lệnh này tương đương:

\[
\nabla_{\mathbf z}
\sum_l\langle g_l,V_l\rangle,
\]

nhưng an toàn hơn khi tensors nằm trên nhiều devices.

**Mỗi probe chỉ được có đúng một second-order `autograd.grad` call cho toàn bộ module list hoặc toàn bộ layer chunk.**

Không được viết:

```python
for module in target_modules:
    hvp_module = autograd.grad(...)
```

và tuyệt đối không được viết:

```python
for group in groups:
    hvp_group = autograd.grad(...)
```

---

# 6. Hutchinson diagonal estimator

Với module \(l\):

\[
U_r^{(l)}
=
\sum_m\mathcal H_{lm}V_r^{(m)}.
\]

Xét một element \(i\):

\[
V_{r,i}^{(l)}U_{r,i}^{(l)}
=
\sum_m\sum_q
[\mathcal H_{lm}]_{iq}
V_{r,i}^{(l)}V_{r,q}^{(m)}.
\]

Với probes độc lập và zero-mean:

\[
\mathbb E
\left[
V_{r,i}^{(l)}V_{r,q}^{(m)}
\right]
=
\mathbf 1[l=m]\mathbf 1[i=q].
\]

Do đó:

\[
\boxed{
\mathbb E
\left[
V_r^{(l)}\odot U_r^{(l)}
\right]
=
\operatorname{diag}(\mathcal H_{ll})
}
\]

và vì:

\[
\mathcal H_{ll}=G_l+R_l,
\]

nên:

\[
\boxed{
\mathbb E[V_r^{(l)}\odot U_r^{(l)}]
=
\operatorname{diag}(G_l+R_l).
}
\]

Cross-module terms không tạo bias; chúng chỉ có thể tăng variance khi số probes nhỏ.

---

# 7. Chuyển raw HVP thành GuidedQuant-compatible group saliency

Raw sample:

\[
D_r^{(l)}
=
V_r^{(l)}\odot U_r^{(l)}.
\]

Shape:

```text
[B, T, out_features]
```

Với contiguous output group \(J_k^{(l)}\):

\[
\widehat s_{b,t,k}^{(l)}
=
\frac{1}{P|J_k^{(l)}|}
\sum_{r=1}^P
\sum_{j\in J_k^{(l)}}
D_{r,b,t,j}^{(l)}.
\]

Implementation:

```python
diag_sample = (probe * hvp).float()

bsz, seq_len, out_features = diag_sample.shape
assert out_features % num_groups == 0
group_size = out_features // num_groups

group_sample = diag_sample.reshape(
    bsz,
    seq_len,
    num_groups,
    group_size,
).mean(dim=-1)

group_accumulator.add_(group_sample)
```

Sau tất cả probes:

```python
group_accumulator.div_(num_probes)
group_accumulator.clamp_min_(0.0)
```

## 7.1. Positive projection

Full NLL Hessian có thể indefinite do residual curvature. LNQ cần PSD group Hessian để Cholesky ổn định. Vì vậy:

\[
\boxed{
s_{b,t,k}^{(l)+}
=
\max(\widehat s_{b,t,k}^{(l)},0).
}
\]

Thứ tự bắt buộc:

```text
V * HV
-> average probes
-> average output channels trong group
-> clamp_min(0)
```

Average probes và group mean là linear nên có thể đổi thứ tự cho nhau. Clamp không được đưa lên trước hai phép average này.

Không được:

- clamp mỗi probe riêng;
- lấy absolute value;
- square `V * HV`;
- normalize mỗi module hoặc mỗi group về cùng norm;
- thêm GGN lần nữa sau full HVP.

---

# 8. Ánh xạ activation curvature về weight-space Hessian

Với weight row \(j\) thuộc group \(k\):

\[
e_j=\hat w_j-w_j.
\]

Output perturbation:

\[
\Delta z_{b,t,j}
=
{x_{b,t}}^\top e_j.
\]

Sau approximation diagonal và group sharing:

\[
\Delta\mathcal L_j
\approx
\frac12
\sum_{b,t}
 s_{b,t,k}^{+}
\left({x_{b,t}}^\top e_j\right)^2.
\]

Suy ra:

\[
\boxed{
H_k
=
X^\top\operatorname{Diag}(s_k^+)X
}
\]

và:

\[
\Delta\mathcal L_j
\approx
\frac12 e_j^\top H_k e_j.
\]

Do \(s_k^+\ge0\):

\[
H_k\succeq0.
\]

## 8.1. Grouping trước GEMM là chính xác

Với channel-specific blocks:

\[
H_j
=
X^\top\operatorname{Diag}(s_j)X,
\]

thì:

\[
\frac1{|J_k|}\sum_{j\in J_k}H_j
=
X^\top
\operatorname{Diag}
\left(
\frac1{|J_k|}\sum_{j\in J_k}s_j
\right)
X.
\]

Vì vậy GuidedQuant có thể average saliency theo output group trước rồi chỉ chạy một weighted GEMM cho mỗi group.

---

# 9. Chính xác phần nào của GuidedQuant được tái sử dụng

Repo hiện tại có hai khối tách biệt.

## 9.1. Producer cũ: `any_precision/quantization/gradients.py`

Function:

```python
get_gradients(...)
```

hiện dùng forward hook + output gradient hook để sinh saliency cũ và lưu:

```text
saliency_path/l{layer_idx}.pt
```

mỗi file có format:

```python
{
    module_name: Tensor[N, seq_len, num_groups]
}
```

**Không sửa saliency cũ thành HVP ngay bên trong gradient hook.** HVP cần `create_graph=True` và một second reverse pass; backward hook hiện tại không phải nơi phù hợp.

Nên tạo producer mới, ví dụ:

```text
any_precision/quantization/nll_hvp_saliency.py
```

với function:

```python
collect_full_nll_hvp_saliencies(...)
```

Producer mới phải lưu đúng cache format cũ.

## 9.2. Consumer giữ nguyên: `any_precision/quantization/activations.py`

Giữ nguyên class:

```python
SaliencyEngine
```

và đặc biệt giữ nguyên `add_batch()`:

```python
sal_weighted_X = torch.einsum("nj,ng->njg", X, S)
block = torch.einsum("ni,njg->ijg", X, sal_weighted_X)
self.XTX.add_(block)
```

Đây chính xác là:

\[
H_k=X^\top\operatorname{Diag}(s_k)X.
\]

Giữ nguyên:

```python
accumulate_saliency_weighted_hessians(...)
```

Function này:

1. load `saliency_path/l{l}.pt`;
2. replay từng transformer layer;
3. wrap actual linear submodules;
4. lấy actual input \(X\) riêng của từng module;
5. accumulate `XTX`;
6. save Hessian tại `hessians_path/l{l}.pt`.

Không build \(X^\top\operatorname{Diag}(s)X\) trong HVP producer. Nếu làm vậy sẽ duplicate toàn bộ GuidedQuant engine và dễ lấy sai input \(X\).

## 9.3. Solver giữ nguyên: `any_precision/quantization/layerwise_quantize.py`

Giữ nguyên:

- Hessian shape conversion;
- damping logic, ngoại trừ numerical zero-scale guard nếu cần;
- Cholesky;
- coordinate-descent assignment `update_P`;
- exact codebook update `update_C`;
- initialization cache;
- early stopping.

Hessian cache do `SaliencyEngine` tạo có shape:

```text
[in_features, in_features, num_groups]
```

trước khi LNQ permute thành:

```text
[num_groups, in_features, in_features].
```

---

# 10. Current pipeline gap cần Codex sửa

Trong repo hiện tại:

```text
layerwise_main.py
```

chỉ gọi:

```python
accumulate_saliency_weighted_hessians(
    analyzer,
    tokens,
    saliency_cache_path,
    hessians_cache_path,
    num_groups,
)
```

Nó giả định saliency cache đã tồn tại.

Saliency cũ thường được sinh ở pipeline khác thông qua:

```python
get_gradients(..., saliency_path=...)
```

Với full-NLL HVP, Codex phải thêm bước producer mới trước `accumulate_saliency_weighted_hessians`:

```text
Tokens
-> Full-NLL HVP saliency producer
-> Existing GuidedQuant SaliencyEngine replay
-> Existing LNQ
```

## 10.1. Cache path phải tách khỏi GuidedQuant cũ

Không dùng chung cache name với saliency empirical-Fisher cũ.

Ví dụ:

```python
saliency_cache_path = (
    f"{cache_dir}/saliency_nll_hvp/"
    f"{model_name}-{dataset}_s{num_examples}_blk{seq_len}"
    f"_g{num_groups}_p{nll_hvp_probes}_seed{nll_hvp_seed}_sum"
)
```

Hessian cache cũng phải chứa curvature mode để không load nhầm:

```python
hessians_cache_path = (
    f"{cache_dir}/hessians_nll_hvp/"
    f"{model_name}-{dataset}_s{num_examples}_blk{seq_len}"
    f"_g{num_groups}_p{nll_hvp_probes}_seed{nll_hvp_seed}"
)
```

---

# 11. Implementation architecture đề xuất

## 11.1. File mới

Tạo:

```text
any_precision/quantization/nll_hvp_saliency.py
```

Nội dung chính:

```python
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class TargetModule:
    layer_idx: int
    module_name: str
    module: torch.nn.Module


def collect_full_nll_hvp_saliencies(
    analyzer,
    input_tokens,
    output_folder: str,
    num_groups: int,
    num_probes: int = 1,
    base_seed: int = 0,
    layer_chunk_size: int = 0,
    overwrite: bool = False,
) -> None:
    ...
```

## 11.2. Target module order

Build deterministic list:

```python
targets = []
for layer_idx, layer in enumerate(analyzer.get_layers()):
    modules = analyzer.get_modules(layer)
    for module_name in analyzer.module_names:
        if module_name in modules:
            targets.append(
                TargetModule(
                    layer_idx=layer_idx,
                    module_name=module_name,
                    module=modules[module_name],
                )
            )
```

Không dựa vào unordered traversal nếu module order ảnh hưởng seed và cache.

## 11.3. Forward hooks

Hook chỉ lưu actual output tensor:

```python
def make_capture_hook(key, storage):
    def hook(module, args, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"Target module {key} returned non-Tensor output"
            )
        storage[key] = output
    return hook
```

Không detach output.

Không cần lưu actual input \(X\) trong HVP pass. `activations.py` sẽ replay và lấy \(X\) chính xác sau.

## 11.4. Rademacher helper

```python
def rademacher_like(
    tensor: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(seed)

    bits = torch.randint(
        low=0,
        high=2,
        size=tensor.shape,
        generator=generator,
        device=tensor.device,
        dtype=torch.int8,
    )

    return bits.to(tensor.dtype).mul_(2).sub_(1)
```

Seed combine function phải deterministic; không dùng Python `hash()` vì hash có thể thay đổi giữa processes.

Ví dụ dùng integer mixing hoặc `hashlib.sha256` từ tuple metadata.

---

# 12. Pseudocode đầy đủ của saliency producer

```python
def collect_full_nll_hvp_saliencies(
    analyzer,
    input_tokens,
    output_folder,
    num_groups,
    num_probes=1,
    base_seed=0,
    layer_chunk_size=0,
    overwrite=False,
):
    assert num_probes >= 1
    os.makedirs(output_folder, exist_ok=True)

    layers = analyzer.get_layers()

    # Per-layer cache buffers:
    # saliency_chunks[layer_idx][module_name]
    #     -> list[Tensor[B, T, G]]
    saliency_chunks = [
        {
            module_name: []
            for module_name in analyzer.get_modules(layer).keys()
        }
        for layer in layers
    ]

    model = analyzer.model
    model.eval()
    model.zero_grad(set_to_none=True)

    targets = build_deterministic_target_list(analyzer)
    chunks = split_targets_by_layer(
        targets,
        layer_chunk_size=layer_chunk_size,
    )

    for batch_idx, tokens in enumerate(input_tokens):
        tokens = normalize_token_batch(tokens)

        # If all targets fit, chunks contains one element.
        # If chunking is enabled, the model is forwarded once per chunk.
        for chunk_idx, target_chunk in enumerate(chunks):
            captured = {}
            hooks = []

            for target in target_chunk:
                key = (target.layer_idx, target.module_name)
                hooks.append(
                    target.module.register_forward_hook(
                        make_capture_hook(key, captured)
                    )
                )

            try:
                outputs = model(
                    input_ids=tokens,
                    attention_mask=attention_mask_if_any,
                    use_cache=False,
                    return_dict=True,
                )
                logits = outputs.logits

                loss_nll, valid_count = causal_nll_sum(
                    logits=logits,
                    labels=labels,
                    attention_mask=attention_mask_if_any,
                )

                metadata = [
                    (target.layer_idx, target.module_name)
                    for target in target_chunk
                ]
                z_list = [captured[key] for key in metadata]

                assert len(z_list) == len(target_chunk)
                validate_shapes(z_list, target_chunk, num_groups)

                grads_z = torch.autograd.grad(
                    outputs=loss_nll,
                    inputs=z_list,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=False,
                )

                accumulators = [
                    torch.zeros(
                        z.shape[0],
                        z.shape[1],
                        num_groups,
                        dtype=torch.float32,
                        device=z.device,
                    )
                    for z in z_list
                ]

                for probe_idx in range(num_probes):
                    probes = [
                        rademacher_like(
                            z,
                            seed=derive_seed(
                                base_seed,
                                batch_idx,
                                probe_idx,
                                layer_idx,
                                module_name,
                            ),
                        )
                        for z, (layer_idx, module_name)
                        in zip(z_list, metadata)
                    ]

                    hvps = torch.autograd.grad(
                        outputs=grads_z,
                        inputs=z_list,
                        grad_outputs=probes,
                        create_graph=False,
                        retain_graph=(probe_idx + 1 < num_probes),
                        allow_unused=False,
                    )

                    for i, (probe, hvp, z) in enumerate(
                        zip(probes, hvps, z_list)
                    ):
                        if hvp.shape != z.shape:
                            raise RuntimeError("HVP shape mismatch")

                        diag_sample = (probe * hvp).float()

                        bsz, seq_len, out_features = diag_sample.shape
                        if out_features % num_groups != 0:
                            raise ValueError(
                                "out_features must be divisible by num_groups"
                            )

                        group_size = out_features // num_groups
                        group_sample = diag_sample.reshape(
                            bsz,
                            seq_len,
                            num_groups,
                            group_size,
                        ).mean(dim=-1)

                        accumulators[i].add_(group_sample)

                for accumulator in accumulators:
                    accumulator.div_(num_probes)
                    accumulator.clamp_min_(0.0)

                for accumulator, (layer_idx, module_name) in zip(
                    accumulators,
                    metadata,
                ):
                    saliency_chunks[layer_idx][module_name].append(
                        accumulator.to(torch.bfloat16).cpu()
                    )

            finally:
                for hook in hooks:
                    hook.remove()

                del captured
                model.zero_grad(set_to_none=True)
                release_autograd_references()

    for layer_idx, module_dict in enumerate(saliency_chunks):
        layer_result = {}

        for module_name, chunks in module_dict.items():
            if not chunks:
                raise RuntimeError(
                    f"No HVP saliency collected for "
                    f"layer={layer_idx}, module={module_name}"
                )

            layer_result[module_name] = torch.cat(chunks, dim=0)

        output_file = os.path.join(
            output_folder,
            f"l{layer_idx}.pt",
        )
        atomic_torch_save(layer_result, output_file)
```

## 12.1. Không dùng `loss.backward()`

Producer mới nên dùng `torch.autograd.grad` hoàn toàn để:

- không accumulate weight gradients;
- không trigger `square_grad_hook` cũ;
- kiểm soát graph lifetime;
- tránh side effects lên initialization path.

## 12.2. Graph cleanup

Sau mỗi batch/chunk:

```python
del loss_nll, logits, outputs
del grads_z, probes, hvps, z_list
torch.cuda.empty_cache()  # chỉ khi thật sự cần; không bắt buộc mỗi batch
gc.collect()              # dùng có kiểm soát
```

Không lưu tensors còn dính autograd graph vào list dài hạn. Chỉ lưu saliency đã:

```python
accumulator.detach().to(...).cpu()
```

---

# 13. Layer chunking

Mặc định:

```text
layer_chunk_size = 0
```

nghĩa là tất cả target modules được xử lý trong một global HVP.

Nếu OOM, chia theo contiguous transformer-layer chunks.

Trong mỗi chunk:

- forward full model;
- hook target modules trong chunk;
- một first reverse pass;
- một global HVP cho toàn chunk mỗi probe.

Estimator diagonal của từng module vẫn unbiased. Chunking chỉ thay đổi:

- memory;
- runtime;
- realization của cross-module variance.

Không gọi đây là thay đổi objective.

Trade-off:

```text
chunk lớn:
    ít forward/backward hơn
    memory cao hơn

chunk nhỏ:
    memory thấp hơn
    nhiều full-model forward/reverse passes hơn
```

Không chunk theo output group.

---

# 14. Multi-GPU considerations

Model có thể được dispatch qua nhiều GPUs. Các nguyên tắc:

1. `z_list` có thể chứa tensors trên nhiều devices.
2. `grads_z` và corresponding probes phải cùng device/dtype.
3. Dùng multi-output form:

```python
torch.autograd.grad(
    outputs=grads_z,
    inputs=z_list,
    grad_outputs=probes,
)
```

thay vì tự cộng các scalar từ nhiều devices.
4. Nếu một fused kernel không hỗ trợ double backward, curvature pass phải chuyển sang mathematically equivalent unfused implementation.
5. `use_cache=False` để tránh KV-cache behavior không cần thiết trong calibration.

Calibration data parallelism là một optimization khác:

- mỗi GPU/process xử lý một calibration shard;
- mỗi process sinh local saliency hoặc local Hessian;
- nếu merge ở Hessian level, cộng local \(H_k\) vì loss dùng `sum`.

Không trộn model parallel global HVP và data parallel logic một cách ngầm định trong phiên bản đầu.

---

# 15. Precision policy

Khuyến nghị:

- model forward theo dtype hiện tại, thường BF16;
- logits đưa về FP32 khi tính cross entropy;
- `probe * hvp` cast FP32 trước group reduction;
- accumulators FP32;
- saliency cache có thể BF16 để giảm dung lượng;
- `SaliencyEngine.XTX` giữ FP32 như code hiện tại;
- Cholesky giữ logic hiện tại của LNQ.

Nếu HVP underflow hoặc double backward không ổn định:

1. tắt autocast trong curvature pass;
2. dùng FP32 cho module/path liên quan;
3. tắt fused kernels không hỗ trợ double backward;
4. không nhân arbitrary constant vào gradient.

---

# 16. GuidedQuant replay pass giữ nguyên

Sau khi producer mới tạo:

```text
saliency_path/l{layer}.pt
```

với:

```python
{
    module_name: Tensor[N, seq_len, num_groups]
}
```

chạy nguyên:

```python
accumulate_saliency_weighted_hessians(
    analyzer=analyzer,
    data=tokens,
    saliency_path=saliency_cache_path,
    output_folder=hessians_cache_path,
    num_groups=num_groups,
)
```

`SaliencyEngine.add_batch()` sẽ flatten:

```text
X: [B, T, D] -> [B*T, D]
S: [B, T, G] -> [B*T, G]
```

và tính:

```python
sal_weighted_X = torch.einsum("nj,ng->njg", X, S)
block = torch.einsum("ni,njg->ijg", X, sal_weighted_X)
```

Không thay code này trong implementation đầu tiên.

## 16.1. Sample order phải khớp tuyệt đối

Saliency producer và replay pass phải dùng:

- cùng token list;
- cùng thứ tự samples;
- cùng sequence length;
- không shuffle;
- cùng preprocessing.

`SaliencyEngine.index` giả định batch thứ \(i\) trong replay tương ứng đúng saliency slice thứ \(i\).

Nếu order lệch, Hessian vẫn có shape hợp lệ nhưng toán hoàn toàn sai.

---

# 17. Damping và LNQ

Sau replay, mỗi group Hessian:

\[
H_k=X^\top\operatorname{Diag}(s_k^+)X
\]

là PSD nhưng có thể singular.

Giữ damping logic hiện tại:

\[
\widetilde H_k=H_k+\epsilon_k I.
\]

Numerical guard cần có:

```python
avg_diag = torch.mean(torch.diag(H_k))
if not torch.isfinite(avg_diag):
    raise FloatingPointError(...)
if avg_diag <= 0:
    # Automatic numerical fallback only.
    avg_diag = torch.tensor(1.0, device=H_k.device, dtype=H_k.dtype)
```

Guard này chỉ xử lý trường hợp group saliency bị clamp toàn bộ về zero; không phải một balancing hyperparameter mới.

Không normalize từng Hessian về cùng trace/norm trước LNQ.

---

# 18. CLI và integration changes

Thêm flags:

```bash
--curvature_mode nll_global_hvp
--nll_hvp_probes 1
--nll_hvp_seed 0
--nll_hvp_layer_chunk_size 0
--overwrite_saliency
```

Trong `layerwise_nuq.py` parser và `layerwise_main.py`:

```python
if curvature_mode == "nll_global_hvp":
    if overwrite_saliency or not complete_saliency_cache_exists(...):
        collect_full_nll_hvp_saliencies(
            analyzer=analyzer,
            input_tokens=tokens,
            output_folder=saliency_cache_path,
            num_groups=num_groups,
            num_probes=nll_hvp_probes,
            base_seed=nll_hvp_seed,
            layer_chunk_size=nll_hvp_layer_chunk_size,
            overwrite=overwrite_saliency,
        )

    from_cache = accumulate_saliency_weighted_hessians(
        analyzer,
        tokens,
        saliency_cache_path,
        hessians_cache_path,
        num_groups,
    )
```

Default:

```text
nll_hvp_probes = 1
```

Chỉ dùng `2` nếu diagnostics cho thấy seed variance quá lớn.

---

# 19. Cache validation

Trước khi reuse saliency cache, verify:

- đủ `l0.pt ... l{L-1}.pt`;
- mỗi file là dictionary;
- đủ module names;
- tensor rank bằng 3;
- shape `[num_examples, seq_len, num_groups]`;
- finite;
- nonnegative sau projection;
- dtype hợp lệ;
- metadata config khớp.

Nên lưu sidecar metadata:

```json
{
  "curvature_mode": "nll_global_hvp",
  "num_probes": 1,
  "seed": 0,
  "loss_reduction": "sum",
  "num_groups": 8,
  "num_examples": 128,
  "seq_len": 2048,
  "model": "..."
}
```

Không suy đoán config chỉ từ filename.

---

# 20. Unit tests bắt buộc

## 20.1. Exact toy Hessian diagonal

Dùng một toy nonlinear network nhỏ:

1. chọn một intermediate linear output \(Z\);
2. materialize exact Hessian bằng `torch.autograd.functional.hessian` hoặc `torch.func.hessian`;
3. lấy exact diagonal;
4. average Hutchinson estimator qua nhiều probes;
5. kiểm tra convergence.

Expected:

\[
\frac{\|\widehat d-d_{\mathrm{exact}}\|}{\|d_{\mathrm{exact}}\|}
\rightarrow0
\]

khi số probes tăng.

## 20.2. Full Hessian bằng GGN cộng residual trên toy model

Trên toy nonlinear suffix:

1. tính exact full NLL Hessian;
2. tính exact GGN;
3. tính residual bằng hiệu;
4. verify:

\[
H_{\mathrm{full}}=G+R.
\]

5. verify full-NLL HVP khớp `H_full @ v`.

Test này xác nhận không cần tách hai đường trong production.

## 20.3. Global HVP so với per-module HVP

Trên toy multi-module model:

- global independent-probe estimator;
- per-module estimator;
- average nhiều seeds.

Hai cách phải hội tụ về cùng block diagonals.

## 20.4. Group algebra

Verify numerically:

\[
X^\top\operatorname{Diag}
\left(
\frac1{|J|}\sum_{j\in J}s_j
\right)X
=
\frac1{|J|}\sum_{j\in J}
X^\top\operatorname{Diag}(s_j)X.
\]

## 20.5. `SaliencyEngine` reference test

Với random nonnegative \(S\) và random \(X\):

```python
engine.add_batch(X)
```

phải khớp direct reference:

```python
for g in range(G):
    ref[:, :, g] = X_flat.T @ (S_flat[:, g:g+1] * X_flat)
```

## 20.6. Cache/replay order test

Dùng hai samples có activation distributions rất khác nhau. Cố tình đảo saliency order và chứng minh Hessian thay đổi. Production test phải xác nhận no-shuffle ordering.

## 20.7. One-global-HVP structural test

Instrument `torch.autograd.grad` hoặc wrapper function để verify:

```text
number of second-order calls
=
num_batches * num_chunks * num_probes
```

không nhân thêm số modules hoặc số groups.

## 20.8. PSD và Cholesky

Sau positive projection:

```python
eig_min = torch.linalg.eigvalsh(H.double()).min()
```

cho phép negative rất nhỏ do floating point, không cho phép negative đáng kể.

Sau damping, Cholesky phải thành công.

---

# 21. Diagnostics cần log

Cho mỗi module/group:

```text
raw HVP saliency mean/std/min/max
fraction(raw < 0)
mean/std/max after group averaging
fraction(clamped == 0)
mean positive curvature
```

Cho Hessian:

```text
trace(H)
||H||_F
min/max diagonal(H)
symmetry error
Cholesky damping factor
```

Cho runtime:

```text
forward time
first create-graph reverse time
each global HVP time
group reduction time
saliency save time
GuidedQuant replay/XTX time
peak allocated memory
peak reserved memory
```

Cho variance:

```text
relative difference seed 0 vs seed 1
relative difference probes=1 vs probes=2
```

Không tự động rescale groups dựa trên các diagnostics này trong phiên bản đầu.

---

# 22. Performance expectations

Với một probe và không chunk, mỗi calibration micro-batch cần khái niệm:

```text
1 full-model forward
1 reverse pass with create_graph=True
1 global second reverse pass
```

GuidedQuant gốc chỉ cần ordinary backward để sinh saliency, nên full-NLL HVP không thể có wall-clock bằng đúng GuidedQuant.

Tuy nhiên đây là đường nhanh nhất để giữ **full ground-truth NLL curvature** vì:

- không tách GGN và residual;
- không chạy HVP riêng từng layer;
- không chạy HVP riêng từng group;
- group reduction chỉ là tensor reshape/mean;
- weighted XTX chỉ chạy một lần qua engine hiện có.

Chi phí HVP là:

\[
\boxed{
P\text{ global HVPs per batch/chunk}
}
\]

không phải:

\[
L\times P
\]

và không phải:

\[
L\times G\times P.
\]

---

# 23. Những anti-pattern Codex tuyệt đối không được implement

## 23.1. Tách GGN và residual trong default path

Sai:

```text
GGN producer
+ residual HVP producer
+ merge
```

Đúng:

```text
full NLL HVP producer
```

## 23.2. Cộng GGN vào full HVP

Sai:

\[
H_{\mathrm{full-HVP}}+G
=(G+R)+G.
\]

## 23.3. HVP theo layer hoặc group

Sai:

```python
for layer:
    autograd.grad(... second order ...)
```

Đúng:

```python
hvps = autograd.grad(
    outputs=grads_z_for_all_targets,
    inputs=all_target_outputs,
    grad_outputs=all_probes,
)
```

## 23.4. Build Hessian trong HVP producer

Sai:

```text
capture all X and Z
build XTX inside global HVP graph
```

Đúng:

```text
HVP producer saves grouped saliency only
GuidedQuant replay captures actual X and builds XTX
```

## 23.5. Dùng chung transformer-block input cho mọi projection

Sai vì:

- `q_proj/k_proj/v_proj` có input riêng của attention projection;
- `o_proj` nhận attention output;
- `gate_proj/up_proj` nhận MLP input;
- `down_proj` nhận gated activation.

Phải để existing wrappers trong `activations.py` lấy actual module input.

## 23.6. Clamp sớm

Không clamp từng probe hoặc từng channel trước group/probe averaging.

## 23.7. Square hoặc absolute HVP sample

Không dùng:

```python
(probe * hvp).square()
```

hoặc:

```python
(probe * hvp).abs()
```

## 23.8. Mean loss theo từng sample rồi cộng ngang hàng

Dùng sum convention nhất quán.

## 23.9. Reuse cache cũ

Không load empirical-Fisher saliency/Hessian dưới tên cache giống full-NLL HVP.

---

# 24. Acceptance criteria cho pull request

Implementation chỉ được xem là hoàn thành nếu thỏa tất cả:

1. Có producer mới cho full-NLL HVP saliency.
2. Producer dùng ground-truth shifted NLL với `reduction="sum"`.
3. Producer dùng `create_graph=True` cho first derivative.
4. Mỗi probe/chunk dùng đúng một global second-order `autograd.grad` call.
5. Không có loop HVP theo module hoặc group.
6. Raw saliency là `probe * hvp`, không square, không abs.
7. Grouping nằm trên output-channel dimension.
8. Clamp chỉ sau probe/group averaging.
9. Cache format đúng `{module_name: [N,T,G]}` trong `l{layer}.pt`.
10. `SaliencyEngine` và weighted-XTX code được tái sử dụng, không duplicate.
11. Actual module inputs được capture bởi existing replay wrappers.
12. LNQ initialization, CD, exact codebook update và stopping logic giữ nguyên.
13. Cache name/metadata phân biệt rõ `nll_global_hvp` với GuidedQuant cũ.
14. Toy exact-Hessian tests pass.
15. Structural test xác nhận số global HVP calls không nhân số modules/groups.
16. End-to-end smoke test tạo đủ saliency files, Hessian files và Cholesky thành công.
17. Logs có runtime, memory, negative fraction và clamp fraction.

---

# 25. One-sentence instruction cho Codex

> Add a new full-NLL HVP saliency producer that computes summed ground-truth causal NLL, captures the actual outputs of all target linear modules, obtains all output gradients in one create-graph reverse pass, applies one multi-output global Hessian–vector product per independent Rademacher probe, converts each module result to signed `probe * hvp`, averages it over the existing contiguous output-channel groups and probes, clamps only after these averages, saves the resulting `[num_samples, seq_len, num_groups]` tensors in the existing `l{layer}.pt` saliency format, and then reuses `accumulate_saliency_weighted_hessians`, `SaliencyEngine`, the weighted `X^T Diag(s) X` computation, damping, Cholesky, coordinate descent, exact codebook update, initialization, and stopping logic unchanged.

---

# 26. Công thức cuối cùng

Ground-truth NLL:

\[
\boxed{
\mathcal L_{\mathrm{NLL}}
=
-\sum_{(b,t)\in\mathcal M}
\log p_W(y_{b,t}\mid x_{b,<t})
}
\]

Gradient theo all target outputs:

\[
\boxed{
g_l=\nabla_{Z^{(l)}}\mathcal L_{\mathrm{NLL}}}
\]

Global HVP:

\[
\boxed{
U_r^{(l)}
=
\nabla_{Z^{(l)}}
\sum_m\langle g_m,V_r^{(m)}\rangle
}
\]

Full-NLL diagonal estimator:

\[
\boxed{
\mathbb E
\left[
V_r^{(l)}\odot U_r^{(l)}
\right]
=
\operatorname{diag}
\left(
G_l+R_l
\right)
}
\]

Group saliency:

\[
\boxed{
\widehat s_{b,t,k}^{(l)}
=
\frac{1}{P|J_k^{(l)}|}
\sum_{r=1}^{P}
\sum_{j\in J_k^{(l)}}
V_{r,b,t,j}^{(l)}U_{r,b,t,j}^{(l)}
}
\]

Positive projection:

\[
\boxed{
s_{b,t,k}^{(l)+}
=
\max(\widehat s_{b,t,k}^{(l)},0)}
\]

GuidedQuant weight-space Hessian:

\[
\boxed{
H_k^{(l)}
=
{X^{(l)}}^\top
\operatorname{Diag}
\left(s_k^{(l)+}\right)
X^{(l)}
}
\]

LNQ objective:

\[
\boxed{
\mathcal Q
=
\sum_l\sum_k\sum_{j\in J_k^{(l)}}
{e_j^{(l)}}^\top
\widetilde H_k^{(l)}
e_j^{(l)}
}
\]

với:

\[
\widetilde H_k^{(l)}=H_k^{(l)}+\epsilon_k I.
\]

Đây là pipeline cuối cùng cần implement.
