# Finite-Difference GroupTrace Full-NLL Saliency for GuidedQuant/LNQ

## Implementation specification for Codex

---

## 0. Mục tiêu và quyết định thiết kế

Tài liệu này mô tả cách thay **saliency producer** của GuidedQuant bằng một estimator dựa trên **central finite difference của gradient**, nhằm xấp xỉ diagonal của **full ground-truth NLL Hessian** theo output activation của các linear modules.

Phương pháp được chốt:

\[
\boxed{
H_{\mathrm{NLL}}V
\approx
\frac{
\nabla L_{\mathrm{NLL}}(A+\alpha V)
-
\nabla L_{\mathrm{NLL}}(A-\alpha V)
}{
2\alpha
}
}
\]

sau đó dùng Hutchinson/GroupTrace:

\[
\boxed{
s
=
\operatorname{GroupMean}
\left(
V\odot H_{\mathrm{NLL}}V
\right).
}
\]

Tensor \(s\) được lưu đúng format saliency cache hiện tại của GuidedQuant, rồi đưa vào nguyên trạng:

\[
\boxed{
H_k
=
X^\top\operatorname{Diag}(s_k^+)X.
}
\]

### Các quyết định không được thay đổi

1. **Không dùng `create_graph=True`.**
2. **Không dùng double backward hoặc gradient-of-gradient trực tiếp.**
3. **Không giảm sequence length.**
4. **Không giảm số calibration samples.**
5. **Không thay loss ground-truth NLL bằng KL, Fisher hay squared gradient.**
6. **Không viết lại `SaliencyEngine`.**
7. **Không viết lại replay activation, weighted \(X^\top X\), LNQ, Cholesky, coordinate descent hay exact codebook update.**
8. **Không tính một finite-difference pair riêng cho từng group hoặc từng linear module.**
9. Tất cả target linear modules được perturb đồng thời trong mỗi plus/minus pass.
10. Mặc định dùng **central difference sequential**, một probe và automatic layerwise step scale.
11. Step scale là tham số số học được chọn tự động; không thực hiện PPL grid search cho \(\alpha_l\).

### Một câu tóm tắt

> Replace GuidedQuant's empirical-Fisher saliency producer with two ordinary first-order ground-truth NLL passes using additive \(+\alpha_lV^{(l)}\) and \(-\alpha_lV^{(l)}\) interventions at every target linear output, form grouped \(V^{(l)}\odot(G_+^{(l)}-G_-^{(l)})/(2\alpha_l)\), save the result in the existing `l{layer}.pt` saliency format, and leave `SaliencyEngine`, weighted \(X^\top\mathrm{Diag}(s)X\), damping and LNQ unchanged.

---

# 1. Tại sao cần finite difference?

## 1.1. GuidedQuant gốc

GuidedQuant gốc lấy gradient của loss theo output activation:

\[
g^{(l)}
=
\nabla_{A^{(l)}}L,
\]

rồi tạo empirical-Fisher-style saliency:

\[
s_{\mathrm{GQ}}^{(l)}
=
\operatorname{GroupMean}
\left(
g^{(l)}\odot g^{(l)}
\right).
\]

Pipeline tính toán:

```text
forward
→ ordinary backward
→ grad²
→ output-channel grouping
→ save saliency
→ replay từng transformer layer
→ Xᵀ Diag(s) X
→ LNQ
```

Nó không cần Hessian thật và không cần đạo hàm qua backward graph.

## 1.2. Mục tiêu mới

Ta cần full Hessian của **ground-truth next-token NLL** theo intermediate activation:

\[
H_{\mathrm{NLL}}^{(l)}
=
\nabla_{A^{(l)}}^2L_{\mathrm{NLL}}.
\]

Exact HVP bằng reverse-over-reverse autograd là:

\[
g^{(l)}
=
\nabla_{A^{(l)}}L_{\mathrm{NLL}},
\]

\[
H_{\mathrm{NLL}}V
=
\nabla_A
\left\langle
\nabla_A L_{\mathrm{NLL}},
V
\right\rangle.
\]

Nó cần:

```text
forward
→ backward với create_graph=True
→ second backward
```

Trên LLM lớn, `create_graph=True` phải giữ graph của backward/suffix network, gây VRAM rất cao.

## 1.3. Finite difference giải quyết đúng nút thắt

Thay vì đạo hàm gradient bằng autograd, ta đo thay đổi của ordinary gradient:

\[
H_{\mathrm{NLL}}V
\approx
\frac{
g(A+\alpha V)-g(A-\alpha V)
}{
2\alpha
}.
\]

Pipeline mới:

```text
plus ordinary forward/backward
→ compact grouped plus statistic
→ free graph

minus ordinary forward/backward
→ compact grouped minus statistic
→ free graph

central difference
→ positive projection
→ save saliency
→ existing GuidedQuant engine
```

Peak VRAM gần ordinary first-order backward hơn nhiều so với double backward.

---

# 2. Ground-truth next-token NLL

Với input sequence \(x_{1:T}\), labels \(y_{1:T}\), valid next-token positions \(\mathcal M\):

\[
\boxed{
L_{\mathrm{NLL}}
=
-\sum_{t\in\mathcal M}
\log p_W(y_t\mid x_{<t}).
}
\]

Khuyến nghị dùng `reduction="sum"`:

```python
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]

loss_nll = torch.nn.functional.cross_entropy(
    shift_logits.float().reshape(-1, shift_logits.shape[-1]),
    shift_labels.reshape(-1),
    ignore_index=-100,
    reduction="sum",
)
```

### Yêu cầu bắt buộc

Cả plus và minus passes phải dùng:

- cùng input IDs;
- cùng attention mask;
- cùng labels;
- cùng causal shift;
- cùng `ignore_index`;
- cùng `reduction="sum"`;
- cùng model weights;
- `model.eval()`;
- `use_cache=False`;
- dropout và mọi stochastic layer bị tắt.

Không dùng `outputs.loss` nếu không xác nhận rõ reduction convention.

---

# 3. Hessian NLL và phân rã \(GGN+R\)

Với target linear module \(l\):

\[
A^{(l)}
=
X^{(l)}{W^{(l)}}^\top.
\]

Gọi final logits là \(f\), và:

\[
J_l
=
\frac{\partial f}{\partial A^{(l)}}.
\]

Ground-truth NLL Hessian theo \(A^{(l)}\) có dạng:

\[
\boxed{
\mathcal H_{ll}^{\mathrm{NLL}}
=
J_l^\top
\left[
\nabla_f^2L_{\mathrm{NLL}}
\right]
J_l
+
\sum_v
\frac{\partial L_{\mathrm{NLL}}}{\partial f_v}
\nabla_{A^{(l)}}^2f_v.
}
\]

Đặt:

\[
G_l
=
J_l^\top
\left[
\nabla_f^2L_{\mathrm{NLL}}
\right]
J_l
\]

là GGN/Fisher-like term, và:

\[
R_l
=
\sum_v
\frac{\partial L_{\mathrm{NLL}}}{\partial f_v}
\nabla_{A^{(l)}}^2f_v
\]

là residual/model curvature.

Khi đó:

\[
\boxed{
\mathcal H_{ll}^{\mathrm{NLL}}
=
G_l+R_l.
}
\]

### Chú ý cực kỳ quan trọng cho Codex

Phân rã \(G_l+R_l\) chỉ dùng để giải thích lý thuyết.

**Không được tách implementation thành một GGN path và một residual path.**

Finite difference trực tiếp trên ground-truth NLL gradient tự động xấp xỉ full:

\[
(G_l+R_l)V.
\]

Không cộng thêm Fisher/GGN saliency vào kết quả finite difference, vì sẽ double-count GGN.

---

# 4. Full multi-module activation Hessian

Ghép output activation của tất cả target linear modules:

\[
\mathbf A
=
\begin{bmatrix}
\operatorname{vec}(A^{(1)})\\
\vdots\\
\operatorname{vec}(A^{(L)})
\end{bmatrix}.
\]

Full Hessian:

\[
\mathbb H_{\mathrm{NLL}}
=
\nabla_{\mathbf A}^2L_{\mathrm{NLL}}
=
\begin{bmatrix}
\mathcal H_{11}&\cdots&\mathcal H_{1L}\\
\vdots&\ddots&\vdots\\
\mathcal H_{L1}&\cdots&\mathcal H_{LL}
\end{bmatrix}.
\]

Mục tiêu không phải materialize \(\mathbb H_{\mathrm{NLL}}\). Ta chỉ cần diagonal của từng block:

\[
\operatorname{diag}(\mathcal H_{ll}).
\]

---

# 5. Rademacher probes và GroupTrace

Với mỗi target module \(l\), sample probe độc lập:

\[
\boxed{
V^{(l)}
\in
\{-1,+1\}^{\operatorname{shape}(A^{(l)})}.
}
\]

Các phần tử và các modules phải độc lập:

\[
\mathbb E[V_i^{(l)}V_j^{(m)}]
=
\mathbf 1[l=m,\ i=j].
\]

Nếu:

\[
U
=
\mathbb H_{\mathrm{NLL}}V,
\]

thì:

\[
\boxed{
\mathbb E
\left[
V^{(l)}\odot U^{(l)}
\right]
=
\operatorname{diag}(\mathcal H_{ll}).
}
\]

Cross-module Hessian blocks xuất hiện trong từng random sample nhưng triệt tiêu trong kỳ vọng nhờ probes độc lập.

### Probe generation

Probe phải reproducible theo:

```text
base_seed
calibration_sample_id
probe_id
transformer_layer_id
module_name
```

Không reuse cùng sign pattern cho nhiều modules.

---

# 6. Central finite-difference HVP

Định nghĩa:

\[
\mathbf g(\mathbf A)
=
\nabla_{\mathbf A}L_{\mathrm{NLL}}.
\]

Taylor:

\[
\mathbf g(\mathbf A+\Delta)
=
\mathbf g(\mathbf A)
+
\mathbb H_{\mathrm{NLL}}\Delta
+
O(\|\Delta\|^2).
\]

Với symmetric perturbation:

\[
\Delta
=
D\mathbf V,
\]

trong đó \(D\) chứa step \(\alpha_l\) của từng module:

\[
A_+^{(l)}
=
A^{(l)}+\alpha_lV^{(l)},
\]

\[
A_-^{(l)}
=
A^{(l)}-\alpha_lV^{(l)}.
\]

Central difference:

\[
\mathbf g(\mathbf A+D\mathbf V)
-
\mathbf g(\mathbf A-D\mathbf V)
=
2\mathbb H_{\mathrm{NLL}}D\mathbf V
+
O(\|D\mathbf V\|^3).
\]

Với module \(l\):

\[
G_+^{(l)}-G_-^{(l)}
\approx
2\sum_m
\alpha_m
\mathcal H_{lm}V^{(m)}.
\]

Đặt:

\[
\boxed{
U^{(l)}
=
\frac{
G_+^{(l)}-G_-^{(l)}
}{
2\alpha_l
}.
}
\]

Khi đó:

\[
U^{(l)}
\approx
\mathcal H_{ll}V^{(l)}
+
\sum_{m\neq l}
\frac{\alpha_m}{\alpha_l}
\mathcal H_{lm}V^{(m)}.
\]

Nhân với \(V^{(l)}\) và lấy kỳ vọng:

\[
\boxed{
\mathbb E
\left[
V^{(l)}\odot U^{(l)}
\right]
=
\operatorname{diag}(\mathcal H_{ll})
+
O(\alpha^2).
}
\]

Vì vậy các modules có thể dùng layer-specific \(\alpha_l\) và vẫn giữ estimator diagonal hợp lệ trong kỳ vọng.

---

# 7. Additive intervention tại output activation

Không perturb weights trực tiếp.

Với target linear module:

\[
A^{(l)}
=
\operatorname{Linear}(X^{(l)}),
\]

hook trả về:

\[
\widetilde A^{(l)}
=
A^{(l)}+\delta^{(l)}.
\]

Với plus pass:

\[
\delta_+^{(l)}
=
+\alpha_lV^{(l)}.
\]

Với minus pass:

\[
\delta_-^{(l)}
=
-\alpha_lV^{(l)}.
\]

`delta` phải là leaf tensor:

```python
delta = (
    sign * alpha * probe
).detach().requires_grad_(True)
```

Sau forward:

\[
\nabla_{\delta^{(l)}}L
=
\nabla_{\widetilde A^{(l)}}L,
\]

nên `autograd.grad(loss, delta_tensors)` cho đúng gradient theo perturbed module outputs.

### Không được làm

```python
output = output.detach().requires_grad_(True)
```

Việc detach actual module output sẽ cắt dependency giữa các modules và làm sai full end-to-end NLL Hessian.

Chỉ được **cộng** intervention vào output nguyên trạng:

```python
return output + delta
```

---

# 8. Automatic perturbation scale, không PPL tuning

Finite difference cần step size về mặt số học, nhưng không cần biến nó thành hyperparameter phải sweep theo PPL.

## 8.1. Chế độ ưu tiên: quantization activation error

Với initialization \(Q_0(W^{(l)})\):

\[
E_W^{(l)}
=
Q_0(W^{(l)})-W^{(l)}.
\]

Activation error:

\[
E_A^{(l)}
=
X^{(l)}{E_W^{(l)}}^\top.
\]

Chọn:

\[
\boxed{
\alpha_l
=
\operatorname{RMS}
\left(
E_A^{(l)}
\right).
}
\]

Ý nghĩa:

- step tự động theo layer;
- bám đúng scale perturbation mà quantization initialization thực sự gây ra;
- không cần absolute step chung cho mọi layer;
- không grid-search theo PPL.

### Scale calibration

Khuyến nghị tính \(\alpha_l\) một lần trước curvature collection:

1. dùng một full-length calibration sequence;
2. chạy `torch.no_grad()`;
3. hook actual input \(X^{(l)}\) của mỗi target module;
4. lấy initialization weight error \(E_W^{(l)}\);
5. tính RMS của \(X^{(l)}{E_W^{(l)}}^\top\);
6. lưu một scalar FP32 cho mỗi module;
7. dùng cố định scalars đó cho toàn bộ 128 calibration samples.

Sequence dùng để calibrate scale vẫn nằm trong calibration set; không thay đổi sequence length hoặc số samples dùng để estimate curvature.

## 8.2. Fallback tự động: activation RMS

Nếu initialization weights chưa truy cập được tại saliency stage:

\[
\boxed{
\alpha_l
=
\eta_{\mathrm{auto}}
\operatorname{RMS}(A^{(l)}).
}
\]

Dùng một fixed numerical default cho toàn mô hình, không sweep PPL:

\[
\eta_{\mathrm{auto}}
=
\epsilon_{\mathrm{FP32}}^{1/3}
\approx
4.9\times10^{-3}.
\]

Đây là scale numerical baseline cho central finite difference, cân bằng truncation và cancellation trong arithmetic FP32.

Nếu intervention phải cast về BF16, log tỷ lệ:

\[
\frac{\|\delta_l\|_{\mathrm{RMS}}}
{\|A_l\|_{\mathrm{RMS}}}.
\]

## 8.3. Stability check không phải tuning

Có thể chạy một smoke test:

\[
s(\alpha)
\quad\text{so với}\quad
s(\alpha/2)
\]

trên một calibration sequence để kiểm tra implementation và numerical resolution.

Đây là validation, không phải tìm \(\alpha\) cho PPL tốt nhất.

Production default:

```text
fd_scale_mode = quant_activation_error
fd_scale_multiplier = 1.0
```

Không expose hàng loạt per-layer scale knobs trong implementation đầu.

---

# 9. Sequential plus/minus execution

Mặc định dùng sequential mode vì peak memory thấp nhất.

## 9.1. Chuẩn bị model

```python
model.eval()
model.config.use_cache = False

for parameter in model.parameters():
    parameter.requires_grad_(False)
```

Freeze parameters giúp:

- không allocate parameter gradients;
- không cần `zero_grad`;
- ordinary backward chỉ truyền tới intervention leaves.

## 9.2. Plus pass

Cài hooks với:

\[
\delta_+^{(l)}
=
+\alpha_lV^{(l)}.
\]

Chạy forward NLL và:

```python
plus_grads = torch.autograd.grad(
    outputs=loss_plus,
    inputs=list(delta_plus.values()),
    create_graph=False,
    retain_graph=False,
    allow_unused=False,
)
```

## 9.3. Reduce ngay, không giữ full `grad_plus`

Không lưu toàn bộ `G_+^{(l)}` để chờ minus pass.

Vì grouping tuyến tính:

\[
\operatorname{GroupMean}
\left[
V\odot(G_+-G_-)
\right]
=
\operatorname{GroupMean}(V\odot G_+)
-
\operatorname{GroupMean}(V\odot G_-).
\]

Ngay sau plus backward, với từng module:

\[
P_{t,k}^{(l)}
=
\frac1{|J_k|}
\sum_{j\in J_k}
V_{t,j}^{(l)}
G_{+,t,j}^{(l)}.
\]

Chỉ giữ:

```text
plus_grouped[module_name]: [B, T, G]
```

trên CPU FP32 hoặc BF16 cache dtype.

Sau đó xóa:

- logits;
- loss;
- intervention tensors;
- full gradients;
- hooks;
- graph references.

## 9.4. Minus pass

Dùng chính xác cùng probes và \(\alpha_l\):

\[
\delta_-^{(l)}
=
-\alpha_lV^{(l)}.
\]

Tính:

\[
M_{t,k}^{(l)}
=
\frac1{|J_k|}
\sum_{j\in J_k}
V_{t,j}^{(l)}
G_{-,t,j}^{(l)}.
\]

## 9.5. Grouped FD-HVP saliency

Không cần materialize full \(H_{\mathrm{NLL}}V\).

Tính trực tiếp compact group statistic:

\[
\boxed{
s_{t,k}^{(l)}
=
\frac{
P_{t,k}^{(l)}
-
M_{t,k}^{(l)}
}{
2\alpha_l
}.
}
\]

Đây đúng bằng:

\[
\operatorname{GroupMean}
\left(
V^{(l)}
\odot
\frac{G_+^{(l)}-G_-^{(l)}}{2\alpha_l}
\right).
\]

### Lợi ích

Thay vì lưu gradient shape:

```text
[B, T, d_out]
```

giữa hai passes, chỉ lưu:

```text
[B, T, num_groups]
```

Điều này là tối ưu bắt buộc.

---

# 10. Grouping và positive projection

Với output channels được chia thành contiguous groups:

\[
J_1^{(l)},\ldots,J_G^{(l)}.
\]

Nếu dùng \(R\) probes:

\[
\bar s_{t,k}^{(l)}
=
\frac1R
\sum_{r=1}^R
s_{r,t,k}^{(l)}.
\]

Mặc định:

```text
num_probes = 1
```

Sau probe average:

\[
\boxed{
s_{t,k}^{(l)+}
=
\max
\left(
\bar s_{t,k}^{(l)},0
\right).
}
\]

### Thứ tự bắt buộc

```text
ordinary plus gradient
→ V ⊙ grad_plus
→ group mean

ordinary minus gradient
→ V ⊙ grad_minus
→ group mean

subtract and divide by 2 alpha
→ average probes
→ clamp_min(0)
→ save saliency
```

Không clamp:

- `grad_plus`;
- `grad_minus`;
- từng channel;
- từng plus/minus statistic;
- từng probe trước probe average.

Ground-truth NLL Hessian theo intermediate activation có thể indefinite; signed residual curvature phải được phép triệt tiêu positive curvature trước projection.

---

# 11. Ánh xạ saliency sang weight-space Hessian

Với quantization error của weight row:

\[
e_j
=
\hat w_j-w_j.
\]

Output perturbation:

\[
\Delta a_{t,j}
=
x_t^\top e_j.
\]

Sau diagonal activation Hessian approximation và group sharing:

\[
\Delta L
\approx
\frac12
\sum_{j\in J_k}
\sum_t
s_{t,k}^{+}
(x_t^\top e_j)^2.
\]

Suy ra:

\[
\boxed{
H_k
=
X^\top
\operatorname{Diag}
\left(
s_k^+
\right)
X.
}
\]

Đây chính xác là phần `SaliencyEngine` hiện tại của GuidedQuant thực hiện.

---

# 12. Ranh giới producer/consumer trong GuidedQuant repo

## 12.1. Producer cũ

File:

```text
any_precision/quantization/gradients.py
```

Function:

```python
get_gradients(...)
```

Producer cũ:

```python
grad_squared = (grad.float() * 1e3).pow(2)
mean_squared_grad = grad_squared.view(
    bsz,
    seq_len,
    num_groups,
    group_size,
).mean(dim=-1)
```

Finite-difference mode **không dùng đoạn này** để tạo NLL Hessian saliency.

Không sửa semantic của `get_gradients()` nếu nó vẫn cần cho pipeline khác. Thêm producer mới.

## 12.2. Producer mới

Thêm file:

```text
any_precision/quantization/nll_fd_saliency.py
```

Function public:

```python
collect_finite_difference_nll_saliencies(
    analyzer,
    input_tokens,
    saliency_path,
    num_groups,
    initialization_path,
    fd_scheme="central",
    fd_execution_mode="sequential",
    fd_num_probes=1,
    fd_probe_seed=0,
    fd_scale_mode="quant_activation_error",
    fd_scale_multiplier=1.0,
    overwrite=False,
)
```

Function này phải tạo cache:

```text
saliency_path/l0.pt
saliency_path/l1.pt
...
```

Mỗi file:

```python
{
    module_name: Tensor[
        num_samples,
        seq_len,
        producer_num_groups,
    ]
}
```

Recommended cache dtype:

```text
bfloat16 on CPU
```

nhưng mọi phép:

- gradient difference;
- division by \(2\alpha_l\);
- probe multiplication;
- group reduction;
- probe accumulation

phải dùng FP32 trước khi cast cache.

## 12.3. Consumer giữ nguyên

File:

```text
any_precision/quantization/activations.py
```

Giữ nguyên:

```python
accumulate_saliency_weighted_hessians(...)
SaliencyEngine
SaliencyEngine.add_batch(...)
init_saliency_engines_single_wrapper(...)
init_saliency_engines_parallel_wrapper(...)
```

Đặc biệt giữ nguyên:

```python
sal_weighted_X = torch.einsum("nj,ng->njg", X, S)
block = torch.einsum("ni,njg->ijg", X, sal_weighted_X)
self.XTX.add_(block)
```

Nó tính:

\[
X^\top\operatorname{Diag}(s)X.
\]

Không viết một Hessian builder mới trong `nll_fd_saliency.py`.

## 12.4. LNQ giữ nguyên

File:

```text
any_precision/quantization/layerwise_quantize.py
```

Không sửa:

- Hessian tensor layout;
- damping loop;
- Cholesky;
- `update_P`;
- `update_C`;
- objective;
- stopping logic;
- initialization.

Finite-difference chỉ thay nguồn saliency.

---

# 13. Tích hợp vào `layerwise_main.py`

Hiện tại:

```python
from .activations import accumulate_saliency_weighted_hessians
```

và:

```python
from_cache = accumulate_saliency_weighted_hessians(
    analyzer,
    tokens,
    saliency_cache_path,
    hessians_cache_path,
    num_groups,
)
```

Trước lời gọi này, thêm:

```python
from .nll_fd_saliency import (
    collect_finite_difference_nll_saliencies,
)
```

Logic:

```python
if curvature_mode == "finite_difference_nll":
    if not complete_saliency_cache_exists(
        saliency_cache_path,
        num_layers=len(analyzer.get_layers()),
    ):
        collect_finite_difference_nll_saliencies(
            analyzer=analyzer,
            input_tokens=tokens,
            saliency_path=saliency_cache_path,
            num_groups=num_groups,
            initialization_path=initialization_cache_path,
            fd_scheme=fd_scheme,
            fd_execution_mode=fd_execution_mode,
            fd_num_probes=fd_num_probes,
            fd_probe_seed=fd_probe_seed,
            fd_scale_mode=fd_scale_mode,
            fd_scale_multiplier=fd_scale_multiplier,
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

### Cache path phải phân biệt method

Không ghi đè saliency GuidedQuant gốc.

Ví dụ:

```text
cache/saliency/
MODEL-DATA_s128_blk2048_g8_fdnll-central-qerr-p1-seed0/
```

Hessian cache cũng phải chứa curvature mode:

```text
cache/hessians/
MODEL-DATA_s128_blk2048_g8_fdnll-central-qerr-p1-seed0/
```

Nếu không, code có thể vô tình load empirical-Fisher cache cũ.

---

# 14. Pseudocode producer hoàn chỉnh

```python
def collect_finite_difference_nll_saliencies(
    analyzer,
    input_tokens,
    saliency_path,
    num_groups,
    initialization_path,
    fd_num_probes=1,
    fd_probe_seed=0,
    fd_scale_mode="quant_activation_error",
    fd_scale_multiplier=1.0,
    overwrite=False,
):
    model = analyzer.model
    model.eval()
    model.config.use_cache = False

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    target_modules_by_layer = build_target_module_map(analyzer)

    layer_scales = calibrate_layer_scales(
        analyzer=analyzer,
        input_tokens=input_tokens,
        initialization_path=initialization_path,
        scale_mode=fd_scale_mode,
        scale_multiplier=fd_scale_multiplier,
    )

    saliency_chunks = [
        {
            module_name: []
            for module_name in target_modules_by_layer[layer_idx]
        }
        for layer_idx in range(len(target_modules_by_layer))
    ]

    for sample_id, tokens in enumerate(input_tokens):
        tokens = normalize_tokens(tokens).to(model.device)
        labels = tokens

        probe_accumulators = initialize_compact_accumulators(
            target_modules_by_layer,
            batch_size=tokens.shape[0],
            seq_len=tokens.shape[1],
            num_groups=num_groups,
            dtype=torch.float32,
            device="cpu",
        )

        for probe_id in range(fd_num_probes):
            probe_spec = ProbeSpec(
                base_seed=fd_probe_seed,
                sample_id=sample_id,
                probe_id=probe_id,
            )

            plus_grouped = run_fd_signed_pass(
                model=model,
                tokens=tokens,
                labels=labels,
                target_modules_by_layer=target_modules_by_layer,
                layer_scales=layer_scales,
                probe_spec=probe_spec,
                sign=+1,
                num_groups=num_groups,
            )

            release_cuda_graph_references()

            minus_grouped = run_fd_signed_pass(
                model=model,
                tokens=tokens,
                labels=labels,
                target_modules_by_layer=target_modules_by_layer,
                layer_scales=layer_scales,
                probe_spec=probe_spec,
                sign=-1,
                num_groups=num_groups,
            )

            for layer_idx, modules in target_modules_by_layer.items():
                for module_name in modules:
                    alpha = layer_scales[layer_idx][module_name]

                    group_sample = (
                        plus_grouped[layer_idx][module_name].float()
                        - minus_grouped[layer_idx][module_name].float()
                    ) / (2.0 * float(alpha))

                    probe_accumulators[layer_idx][module_name].add_(
                        group_sample.cpu()
                    )

        for layer_idx, modules in target_modules_by_layer.items():
            for module_name in modules:
                saliency = (
                    probe_accumulators[layer_idx][module_name]
                    / float(fd_num_probes)
                )

                saliency.clamp_min_(0.0)

                saliency_chunks[layer_idx][module_name].append(
                    saliency.to(torch.bfloat16)
                )

    os.makedirs(saliency_path, exist_ok=True)

    for layer_idx, module_chunks in enumerate(saliency_chunks):
        layer_dict = {
            module_name: torch.cat(chunks, dim=0)
            for module_name, chunks in module_chunks.items()
        }

        torch.save(
            layer_dict,
            os.path.join(saliency_path, f"l{layer_idx}.pt"),
        )
```

---

# 15. Pseudocode của một signed pass

```python
def run_fd_signed_pass(
    model,
    tokens,
    labels,
    target_modules_by_layer,
    layer_scales,
    probe_spec,
    sign,
    num_groups,
):
    interventions = {}
    probes = {}
    hooks = []

    def make_hook(layer_idx, module_name):
        def hook(module, inputs, output):
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"{layer_idx}:{module_name} output is not a Tensor"
                )

            probe = deterministic_rademacher_like(
                output,
                seed=probe_spec.seed_for(
                    layer_idx,
                    module_name,
                ),
            )

            alpha = float(
                layer_scales[layer_idx][module_name]
            )

            delta = (
                sign * alpha * probe
            ).detach().requires_grad_(True)

            probes[(layer_idx, module_name)] = probe
            interventions[(layer_idx, module_name)] = delta

            return output + delta

        return hook

    for layer_idx, modules in target_modules_by_layer.items():
        for module_name, module in modules.items():
            hooks.append(
                module.register_forward_hook(
                    make_hook(layer_idx, module_name)
                )
            )

    try:
        logits = model(
            input_ids=tokens,
            attention_mask=torch.ones_like(tokens),
            use_cache=False,
        ).logits

        loss_nll = ground_truth_nll_sum(
            logits=logits,
            labels=labels,
        )

        ordered_keys = list(interventions.keys())
        ordered_deltas = [
            interventions[key]
            for key in ordered_keys
        ]

        gradients = torch.autograd.grad(
            outputs=loss_nll,
            inputs=ordered_deltas,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )

        compact = nested_module_dict()

        for key, gradient in zip(ordered_keys, gradients):
            layer_idx, module_name = key
            probe = probes[key]

            # Reduce immediately; do not save full gradients.
            signed_probe_grad = (
                probe.float() * gradient.float()
            )

            compact[layer_idx][module_name] = (
                reduce_contiguous_output_groups(
                    signed_probe_grad,
                    num_groups=num_groups,
                )
                .detach()
                .cpu()
            )

            del signed_probe_grad

        return compact

    finally:
        for hook in hooks:
            hook.remove()

        interventions.clear()
        probes.clear()
```

### Group reduction

```python
def reduce_contiguous_output_groups(
    tensor,
    num_groups,
):
    bsz, seq_len, out_features = tensor.shape

    if out_features % num_groups != 0:
        raise ValueError(
            f"out_features={out_features} is not divisible "
            f"by num_groups={num_groups}"
        )

    group_size = out_features // num_groups

    return tensor.view(
        bsz,
        seq_len,
        num_groups,
        group_size,
    ).mean(dim=-1)
```

---

# 16. Không dùng `loss.backward()`

Ưu tiên:

```python
torch.autograd.grad(
    loss,
    intervention_tensors,
    create_graph=False,
    retain_graph=False,
)
```

thay vì:

```python
loss.backward()
```

Lý do:

- không tạo `.grad` cho model parameters;
- parameters đã frozen;
- không cần `zero_grad`;
- API trả đúng gradients cần thiết;
- giảm state và nguy cơ memory leak.

---

# 17. Sequential, paired-batch và dual-GPU modes

## 17.1. Sequential — default

```text
plus forward/backward
free graph
minus forward/backward
```

Ưu điểm:

- peak VRAM thấp nhất;
- dễ debug;
- chính xác.

Default:

```text
fd_execution_mode = sequential
```

## 17.2. Paired batch — optional

Concatenate plus/minus theo batch dimension:

```text
[input_plus, input_minus]
```

và intervention:

\[
[\alpha V,\ -\alpha V].
\]

Một forward và một backward nhưng activation memory gần gấp đôi.

Không dùng làm default trên model lớn.

## 17.3. Dual GPU plus/minus — optional

Nếu có hai model replicas:

- GPU 0 chạy plus;
- GPU 1 chạy minus;
- trừ compact grouped statistics trên CPU.

Wall-clock có thể gần một ordinary pass, nhưng cần hai model copies.

Không ảnh hưởng estimator.

---

# 18. Dtype và numerical handling

Khuyến nghị:

- model forward: BF16 như GuidedQuant hiện tại;
- intervention tensor: cùng dtype với module output;
- gradients: cast FP32 trước probe multiplication;
- group reduction: FP32;
- plus-minus subtraction: FP32;
- division by \(2\alpha_l\): FP32;
- probe accumulator: FP32;
- cache: BF16 CPU;
- weighted GEMM trong `SaliencyEngine`: FP32 hiện tại;
- Cholesky: logic hiện tại.

Không nhân gradient với `1e3`.

Nếu dùng một scale factor số học để tránh underflow, phải chia ngược chính xác trước khi lưu saliency. Phiên bản đầu không cần arbitrary gradient scaling.

---

# 19. Memory lifecycle bắt buộc

Sau plus pass:

```python
del logits_plus
del loss_plus
del gradients_plus
del interventions_plus
del probes_plus
remove_hooks()
gc.collect()
torch.cuda.empty_cache()
```

Sau minus pass tương tự.

`torch.cuda.empty_cache()` chỉ có tác dụng sau khi mọi live Python reference tới graph đã được xóa.

Không lưu:

- full module outputs cho toàn calibration set;
- full `grad_plus`;
- full `grad_minus`;
- full FD-HVP tensors;
- full per-channel saliency trên CPU.

Chỉ giữ compact:

```text
[B, T, G]
```

sau mỗi signed pass.

---

# 20. Cache contract với GuidedQuant

Mỗi:

```text
saliency_path/l{layer_idx}.pt
```

phải load thành:

```python
dict[str, torch.Tensor]
```

Ví dụ:

```python
{
    "self_attn.q_proj": Tensor[N, T, G],
    "self_attn.k_proj": Tensor[N, T, G],
    "self_attn.v_proj": Tensor[N, T, G],
    "self_attn.o_proj": Tensor[N, T, G],
    "mlp.gate_proj": Tensor[N, T, G],
    "mlp.up_proj": Tensor[N, T, G],
    "mlp.down_proj": Tensor[N, T, G],
}
```

`activations.py` hiện có thể down-group nếu producer groups chia hết cho requested `num_groups`:

```python
loaded = {
    k: v.view(
        v.shape[0],
        v.shape[1],
        num_groups,
        group_subchannels,
    ).mean(dim=-1)
    for k, v in loaded.items()
}
```

Đơn giản nhất là producer lưu đúng `num_groups` mà LNQ sử dụng.

---

# 21. CLI flags đề xuất

Trong `layerwise_nuq.py`:

```python
parser.add_argument(
    "--curvature_mode",
    type=str,
    default="finite_difference_nll",
    choices=[
        "guidedquant_empirical_fisher",
        "finite_difference_nll",
    ],
)

parser.add_argument(
    "--fd_scheme",
    type=str,
    default="central",
    choices=["central", "forward"],
)

parser.add_argument(
    "--fd_execution_mode",
    type=str,
    default="sequential",
    choices=["sequential", "paired_batch", "dual_gpu"],
)

parser.add_argument(
    "--fd_num_probes",
    type=int,
    default=1,
)

parser.add_argument(
    "--fd_probe_seed",
    type=int,
    default=0,
)

parser.add_argument(
    "--fd_scale_mode",
    type=str,
    default="quant_activation_error",
    choices=[
        "quant_activation_error",
        "activation_rms_auto",
    ],
)

parser.add_argument(
    "--fd_scale_multiplier",
    type=float,
    default=1.0,
)

parser.add_argument(
    "--overwrite_saliency",
    action="store_true",
)
```

### Default production configuration

```bash
--curvature_mode finite_difference_nll \
--fd_scheme central \
--fd_execution_mode sequential \
--fd_num_probes 1 \
--fd_probe_seed 0 \
--fd_scale_mode quant_activation_error \
--fd_scale_multiplier 1.0
```

---

# 22. Unit tests bắt buộc

## 22.1. Quadratic function

Với:

\[
L(a)
=
\frac12a^\top Ha+b^\top a,
\]

central finite difference phải cho:

\[
\frac{
\nabla L(a+\alpha v)
-
\nabla L(a-\alpha v)
}{
2\alpha
}
=
Hv
\]

gần machine precision.

## 22.2. Toy neural network

So sánh:

\[
U_{\mathrm{exact}}
=
H_{\mathrm{NLL}}V
\]

từ exact autograd HVP và:

\[
U_{\mathrm{FD}}
=
\frac{G_+-G_-}{2\alpha}.
\]

Metrics:

- cosine similarity;
- relative error;
- Pearson/Spearman của grouped \(V\odot U\).

## 22.3. Producer equivalence

Kiểm tra hai cách:

1. materialize full:
   \[
   V\odot(G_+-G_-)/(2\alpha)
   \]
   rồi group;
2. group \(V\odot G_+\), group \(V\odot G_-\), rồi trừ.

Hai kết quả phải giống trong FP tolerance.

## 22.4. Same-probe test

Plus và minus passes phải regenerate chính xác cùng \(V\).

## 22.5. Probe independence

Các modules phải có probe khác nhau.

## 22.6. Cache format test

Mỗi cache tensor phải có shape:

```text
[num_examples, seq_len, num_groups]
```

và key set phải khớp `analyzer.module_names`.

## 22.7. No-double-backward audit

Production file không được chứa:

```python
create_graph=True
```

hoặc second derivative call.

## 22.8. Sequential vs paired batch

Nếu paired mode được triển khai, compact group stats phải khớp sequential mode.

## 22.9. PSD test

Sau `clamp_min(0)` và `SaliencyEngine`:

\[
H_k
=
X^\top\operatorname{Diag}(s_k)X
\succeq0.
\]

Cho phép eigenvalue âm rất nhỏ do floating-point.

## 22.10. Cholesky test

Sau damping hiện tại, mọi group Hessian phải Cholesky thành công.

---

# 23. Diagnostics bắt buộc

Log cho từng module:

```text
alpha_l
activation RMS
quantization activation-error RMS nếu có
alpha_l / activation_RMS

plus NLL
minus NLL
relative NLL difference

mean/std/min/max grouped FD saliency trước clamp
fraction negative trước clamp
fraction zero sau clamp

finite count / NaN count / Inf count
```

Timing:

```text
plus forward
plus backward
plus compact reduction
minus forward
minus backward
minus compact reduction
central difference/postprocess
saliency save
GuidedQuant weighted-XTX replay
```

Memory:

```text
peak allocated
peak reserved
```

### Numerical stability metric

\[
r_g
=
\frac{
\|G_+-G_-\|
}{
\frac12(\|G_+\|+\|G_-\|)+\epsilon
}.
\]

Do implementation compact theo group, có thể log phiên bản compact:

\[
r_s
=
\frac{
\|P-M\|
}{
\frac12(\|P\|+\|M\|)+\epsilon
}.
\]

Nếu quá nhỏ, finite difference có thể bị cancellation.

---

# 24. Validation với quantization objective

Trên một small model/layer nơi exact HVP chạy được:

1. tạo cùng probes;
2. tính exact-HVP GroupTrace saliency;
3. tính finite-difference GroupTrace saliency;
4. đưa cả hai qua cùng `SaliencyEngine`;
5. so Hessians và quadratic predictions.

Metrics:

\[
\operatorname{corr}
\left(
s_{\mathrm{FD}},
s_{\mathrm{exact}}
\right),
\]

\[
\frac{
\|H_{\mathrm{FD}}-H_{\mathrm{exact}}\|_F
}{
\|H_{\mathrm{exact}}\|_F+\epsilon
},
\]

và với perturbation \(E\):

\[
C_{\mathrm{pred}}(E)
=
\sum_{l,k,j\in J_k}
e_j^\top H_{l,k}e_j.
\]

So với symmetric true NLL change:

\[
C_{\mathrm{true}}(E)
=
\frac{
L(W+\beta E)+L(W-\beta E)-2L(W)
}{
\beta^2
}.
\]

---

# 25. Những điều tuyệt đối không làm

Không:

1. dùng `grad²` của ground-truth NLL rồi gọi là full NLL Hessian;
2. dùng teacher KL hoặc pseudo-label Fisher thay full NLL;
3. dùng `create_graph=True`;
4. dùng double backward;
5. tính GGN và \(R\) riêng rồi ghép;
6. cộng Fisher saliency vào finite-difference saliency;
7. perturb weights thay vì additive activation interventions;
8. detach actual module outputs;
9. dùng probe khác giữa plus và minus;
10. reuse cùng probe cho mọi module;
11. clamp plus/minus gradients;
12. clamp từng probe trước probe average;
13. lưu full plus/minus gradients giữa hai passes;
14. build \(X^\top\operatorname{Diag}(s)X\) trong producer mới;
15. sửa `SaliencyEngine`;
16. sửa LNQ solver;
17. giảm sequence length;
18. giảm calibration samples;
19. dùng paired batch làm default nếu nó tăng peak VRAM;
20. sweep \(\alpha_l\) theo PPL như một hyperparameter model-specific.

---

# 26. Runtime kỳ vọng

Với one probe, central sequential FD cần:

```text
2 full-model forwards
2 ordinary first-order reverse passes
```

GuidedQuant gốc cần:

```text
1 forward
1 ordinary reverse pass
```

Vì vậy finite difference không thể rẻ bằng đúng GuidedQuant gốc.

Tuy nhiên nó tránh:

```text
create_graph=True
second-order reverse graph
```

và peak VRAM gần first-order backward.

Sau producer, chi phí:

```text
grouping
saliency cache
layer replay
Xᵀ Diag(s) X
LNQ
```

giống GuidedQuant.

---

# 27. Recommended implementation order

## Phase 1 — correctness trên một module

1. additive intervention;
2. central plus/minus sequential;
3. gradient chỉ theo delta;
4. exact-HVP comparison;
5. compact-before-subtract equivalence.

## Phase 2 — one transformer layer

1. tất cả linear modules trong layer;
2. independent module probes;
3. cache đúng format;
4. chạy qua `SaliencyEngine`;
5. LNQ smoke test.

## Phase 3 — global model

1. tất cả target modules perturb đồng thời;
2. one probe;
3. 128 samples, full sequence length;
4. compact group reduction sau từng pass;
5. save one file per transformer layer.

## Phase 4 — optional acceleration

1. dual-GPU plus/minus;
2. paired batch nếu VRAM đủ;
3. `torch.compile` ordinary first-order passes;
4. fused subtract/probe/group reduction;
5. CUDA graphs nếu shape cố định.

Không tối ưu trước khi correctness tests pass.

---

# 28. Acceptance criteria

Implementation được coi là hoàn thành khi:

1. Không có `create_graph=True` trong FD path.
2. Không OOM với cấu hình calibration hiện tại mà không giảm `seq_len` hoặc `num_examples`.
3. Producer tạo đủ `l{layer}.pt`.
4. Cache keys/shapes tương thích `accumulate_saliency_weighted_hessians`.
5. `activations.py` chạy không sửa semantic.
6. Hessian cache được tạo bằng existing `SaliencyEngine`.
7. LNQ chạy nguyên trạng.
8. Toy exact-HVP comparison đạt correlation cao.
9. Plus/minus dùng cùng probe.
10. Logs chứng minh peak VRAM gần ordinary backward.
11. Cache path không đụng empirical-Fisher cache.
12. Không có NaN/Inf trong saliency/Hessian.
13. Cholesky thành công sau existing damping.
14. Full run dùng đủ toàn bộ calibration samples và sequence length.

---

# 29. One-sentence instruction for Codex

> Implement a new finite-difference NLL saliency producer, not a new Hessian engine: for every calibration sequence and probe, add deterministic independent Rademacher interventions \(+\alpha_lV_l\) to all target linear outputs, compute ordinary ground-truth NLL gradients only with respect to the intervention leaves, immediately reduce \(V_l\odot G_{+,l}\) to the existing contiguous output-channel groups and free the graph, repeat with the identical probes and \(-\alpha_lV_l\), form \((\operatorname{GroupMean}(V_l\odot G_{+,l})-\operatorname{GroupMean}(V_l\odot G_{-,l}))/(2\alpha_l)\), average probes, clamp only then, save the existing `{module_name: [N,T,G]}` saliency cache, and pass it unchanged through `accumulate_saliency_weighted_hessians`, `SaliencyEngine`, damping and LNQ.

---

# 30. Pipeline cuối cùng

```text
GROUND-TRUTH NLL FINITE-DIFFERENCE SALIENCY PRODUCER
────────────────────────────────────────────────────
automatic alpha_l per module
independent deterministic probes V_l

PLUS:
    output_l → output_l + alpha_l V_l
    ordinary NLL forward
    autograd.grad(loss, delta_leaves)
    group_mean(V_l ⊙ grad_plus_l)
    offload compact [B,T,G]
    free graph

MINUS:
    output_l → output_l - alpha_l V_l
    ordinary NLL forward
    autograd.grad(loss, delta_leaves)
    group_mean(V_l ⊙ grad_minus_l)
    offload compact [B,T,G]
    free graph

FD GROUPTRACE:
    s_l = (plus_group_l - minus_group_l) / (2 alpha_l)
    average probes
    clamp_min(0)
    save saliency/l{layer}.pt


EXISTING GUIDEDQUANT CONSUMER — UNCHANGED
────────────────────────────────────────
load {module_name: [N,T,G]}
replay transformer layers
capture actual input X_l of each linear module
H_l,k = X_lᵀ Diag(s_l,k) X_l
save hessians/l{layer}.pt


EXISTING LNQ — UNCHANGED
────────────────────────
load grouped Hessians
existing damping
Cholesky
coordinate-descent assignment
exact codebook update
stopping logic
```
