# Fast Ground-Truth NLL Hessian Approximation by Finite-Difference Group-Trace HVP

## 0. Mục tiêu

Tài liệu này mô tả một phương pháp thay thế cho **autograd double-backward HVP** khi ước lượng Hessian của ground-truth next-token NLL.

Phương pháp hiện tại dùng:

```text
1 forward
1 backward với create_graph=True
1 second-order backward
```

để tính Hessian–vector product:

\[
H_{\mathrm{NLL}}V.
\]

Phương pháp mới dùng **finite difference của gradient**:

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
}.
}
\]

Nó chỉ cần ordinary first-order autograd:

```text
2 ordinary forward
2 ordinary backward
```

hoặc một paired batch:

```text
1 doubled-batch forward
1 doubled-batch backward
```

Mục tiêu:

- dùng đủ toàn bộ calibration set, ví dụ 128 sequences;
- xấp xỉ trực tiếp full ground-truth \(H_{\mathrm{NLL}}\);
- không dùng teacher-KL, empirical Fisher hoặc squared gradient thay cho Hessian;
- không cần `create_graph=True`;
- không cần double backward;
- không viết curvature rule riêng cho Llama, Mistral, Qwen hoặc từng kiến trúc;
- giữ nguyên GroupTrace reduction, PSD projection, weighted GEMM và solver hiện tại;
- kiểm tra xem ordinary first-order backward có nhanh hơn double backward trên LLM thực tế hay không.

Phương pháp này không bảo đảm nhanh hơn trước khi benchmark. Tuy nhiên, nó là một approximation thực sự của \(H_{\mathrm{NLL}}\), không phải heuristic Fisher/KL proxy.

---

# 1. Ký hiệu

Giả sử mô hình có \(L\) target linear layers.

Với layer \(l\):

\[
A^{(l)}
\in
\mathbb R^{B\times T\times d_{\mathrm{out}}^{(l)}}
\]

là output activation của layer.

Có thể flatten batch và token dimensions:

\[
A^{(l)}
\in
\mathbb R^{N_l\times d_{\mathrm{out}}^{(l)}},
\qquad
N_l=BT.
\]

Input activation của target linear layer:

\[
X^{(l)}
\in
\mathbb R^{N_l\times d_{\mathrm{in}}^{(l)}}.
\]

Ghép output activation của tất cả target layers:

\[
\mathbf A
=
\begin{bmatrix}
\operatorname{vec}(A^{(1)})\\
\vdots\\
\operatorname{vec}(A^{(L)})
\end{bmatrix}.
\]

Ground-truth next-token NLL:

\[
L_{\mathrm{NLL}}(\mathbf A)
=
-\sum_{t\in\mathcal M}
\log p(y_t^{\mathrm{gt}}),
\]

trong đó \(\mathcal M\) là tập valid next-token labels.

Gradient theo toàn bộ target-layer outputs:

\[
\mathbf g(\mathbf A)
=
\nabla_{\mathbf A}L_{\mathrm{NLL}}.
\]

Full activation Hessian:

\[
\mathbb H_{\mathrm{NLL}}
=
\nabla_{\mathbf A}^2L_{\mathrm{NLL}}.
\]

Mục tiêu là estimate diagonal của từng layer block:

\[
\operatorname{diag}(\mathcal H_{ll}),
\]

không materialize full Hessian.

---

# 2. \(V\) là gì?

\(V\) là một **probe ngẫu nhiên**, tức một hướng thử trong không gian activation.

Với layer \(l\):

\[
\boxed{
V^{(l)}
\in
\{-1,+1\}^{\operatorname{shape}(A^{(l)})}.
}
\]

Mỗi phần tử được sample độc lập:

\[
P(V_i^{(l)}=+1)=P(V_i^{(l)}=-1)=\frac12.
\]

Đây là Rademacher probe.

\(V^{(l)}\):

- không phải weight;
- không phải vocabulary vector;
- không phải gradient;
- có cùng shape với output activation \(A^{(l)}\);
- chỉ định hướng perturbation:
  \[
  A^{(l)}\rightarrow A^{(l)}\pm \alpha_lV^{(l)}.
  \]

Ý nghĩa:

```text
V                = hướng thử ngẫu nhiên
alpha * V        = perturbation thực tế
g_plus - g_minus = thay đổi gradient theo hướng V
V * FD-HVP       = estimator cho diagonal Hessian
```

Với zero-mean independent probes:

\[
\mathbb E[V_iV_j]
=
\begin{cases}
1,&i=j,\\
0,&i\ne j.
\end{cases}
\]

Do đó:

\[
\boxed{
\mathbb E[V\odot HV]
=
\operatorname{diag}(H).
}
\]

---

# 3. Derivation của finite-difference HVP

Xét gradient:

\[
\mathbf g(\mathbf A)
=
\nabla_{\mathbf A}L_{\mathrm{NLL}}(\mathbf A).
\]

Taylor quanh \(\mathbf A\):

\[
\mathbf g(\mathbf A+\Delta)
=
\mathbf g(\mathbf A)
+
\mathbb H_{\mathrm{NLL}}\Delta
+
O(\|\Delta\|^2).
\]

Với:

\[
\Delta=D\mathbf V,
\]

trong đó \(D\) chứa scale perturbation theo từng layer.

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

Nếu mọi layer dùng cùng scalar \(\alpha\):

\[
\boxed{
\mathbb H_{\mathrm{NLL}}\mathbf V
\approx
\frac{
\mathbf g(\mathbf A+\alpha\mathbf V)
-
\mathbf g(\mathbf A-\alpha\mathbf V)
}{2\alpha}.
}
\]

Sai số central difference:

\[
O(\alpha^2).
\]

Phương pháp target trực tiếp Hessian của ground-truth NLL vì cả hai loss đều là:

\[
L_{\mathrm{NLL}}^{+}
=
-\sum_t\log p_+(y_t^{\mathrm{gt}}),
\]

\[
L_{\mathrm{NLL}}^{-}
=
-\sum_t\log p_-(y_t^{\mathrm{gt}}).
\]

Nó giữ cả:

\[
H_{\mathrm{NLL}}
=
H_{\mathrm{GGN/KL}}
+
R_{\mathrm{GT}}.
\]

---

# 4. Layer-specific perturbation scale

Các layer có activation scale rất khác nhau. Không nên mặc định dùng cùng một absolute \(\alpha\).

Với layer \(l\):

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

Gradient tại plus/minus passes:

\[
G_+^{(l)}
=
\nabla_{A_+^{(l)}}L_{\mathrm{NLL}}^{+},
\]

\[
G_-^{(l)}
=
\nabla_{A_-^{(l)}}L_{\mathrm{NLL}}^{-}.
\]

Do tất cả layers được perturb đồng thời:

\[
G_+^{(l)}-G_-^{(l)}
\approx
2\sum_m
\alpha_m
\mathcal H_{lm}V^{(m)}.
\]

Nếu chia riêng cho \(\alpha_l\):

\[
U^{(l)}
=
\frac{G_+^{(l)}-G_-^{(l)}}{2\alpha_l}
\approx
\mathcal H_{ll}V^{(l)}
+
\sum_{m\ne l}
\frac{\alpha_m}{\alpha_l}
\mathcal H_{lm}V^{(m)}.
\]

Khi nhân với \(V^{(l)}\), cross-layer terms triệt tiêu trong kỳ vọng nếu probes độc lập:

\[
\boxed{
\mathbb E[V^{(l)}\odot U^{(l)}]
=
\operatorname{diag}(\mathcal H_{ll})
+
O(\alpha^2).
}
\]

Điều này giữ nguyên logic của global multi-layer GroupTrace HVP.

---

# 5. GroupTrace estimator

Với layer \(l\), output channels được chia thành groups:

\[
J_1^{(l)},\ldots,J_{K_l}^{(l)}.
\]

Raw FD-HVP:

\[
\boxed{
U^{(l)}
=
\frac{G_+^{(l)}-G_-^{(l)}}{2\alpha_l}.
}
\]

Raw diagonal sample:

\[
D^{(l)}
=
V^{(l)}\odot U^{(l)}.
\]

Group statistic:

\[
\boxed{
\widehat s_{t,k}^{(l)}
=
\frac1{|J_k^{(l)}|}
\sum_{j\in J_k^{(l)}}
V_{t,j}^{(l)}U_{t,j}^{(l)}.
}
\]

Với \(R\) probes:

\[
\boxed{
\widehat s_{t,k}^{(l)}
=
\frac1{R|J_k^{(l)}|}
\sum_{r=1}^{R}
\sum_{j\in J_k^{(l)}}
V_{r,t,j}^{(l)}
\frac{G_{r,+,t,j}^{(l)}-G_{r,-,t,j}^{(l)}}{2\alpha_l}.
}
\]

Mặc định benchmark đầu tiên:

```text
num_probes = 1
```

---

# 6. Additive intervention thay vì sửa weight

Không perturb trực tiếp weight vì:

- phức tạp khi nhiều layers cùng perturb;
- phải restore weights;
- dễ sai với tied weights;
- gây thêm overhead;
- khó lấy gradient đúng theo target-layer output.

Cách khuyến nghị là thêm một intervention tensor vào output:

\[
\widetilde A^{(l)}
=
A^{(l)}+\Delta^{(l)}.
\]

Trong forward hook:

```python
def hook(module, inputs, output):
    return output + delta[layer_name]
```

Với:

```python
delta.requires_grad_(True)
```

thì:

```python
delta.grad
```

chính là gradient của NLL theo perturbed layer output.

Điều này model-agnostic miễn target layer trả về tensor hoặc có adapter lấy đúng tensor output.

---

# 7. Sequential central finite difference

## 7.1. Plus pass

Với mỗi layer:

```python
delta_plus[layer] = (
    alpha[layer] * probe[layer]
).detach().requires_grad_(True)
```

Cài hooks:

```python
output -> output + delta_plus[layer]
```

Chạy:

```python
logits_plus = model(...).logits
loss_plus = ground_truth_nll_sum(
    logits_plus,
    labels,
)
```

Khuyến nghị lấy gradient chỉ theo intervention tensors:

```python
grad_plus = torch.autograd.grad(
    outputs=loss_plus,
    inputs=list(delta_plus.values()),
    create_graph=False,
    retain_graph=False,
    allow_unused=False,
)
```

Không cần materialize gradient cho toàn bộ model parameters.

## 7.2. Minus pass

```python
delta_minus[layer] = (
    -alpha[layer] * probe[layer]
).detach().requires_grad_(True)
```

Tương tự:

```python
grad_minus = torch.autograd.grad(
    outputs=loss_minus,
    inputs=list(delta_minus.values()),
    create_graph=False,
    retain_graph=False,
    allow_unused=False,
)
```

Cuối cùng:

```python
fd_hvp[layer] = (
    grad_plus[layer] - grad_minus[layer]
) / (2.0 * alpha[layer])
```

Không dùng:

```python
create_graph=True
retain_graph=True cho double backward
gradient-of-gradient
```

---

# 8. Pseudocode đầy đủ

```python
def collect_fd_grouptrace_nll_stats(
    model,
    batch,
    target_layers,
    group_indices,
    num_groups,
    layer_scales,
    probe_seed,
):
    # A. Sample probes with the same shapes as target outputs.
    probes = sample_layer_output_probes(
        model=model,
        batch=batch,
        target_layers=target_layers,
        seed=probe_seed,
    )

    # B. Plus pass.
    delta_plus = {
        name: (
            layer_scales[name] * probes[name]
        ).detach().requires_grad_(True)
        for name in target_layers
    }

    with install_additive_output_hooks(model, delta_plus):
        logits_plus = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits

        loss_plus = ground_truth_next_token_nll(
            logits=logits_plus,
            labels=batch["labels"],
            reduction="sum",
        )

        plus_tuple = torch.autograd.grad(
            outputs=loss_plus,
            inputs=list(delta_plus.values()),
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )

    grad_plus = {
        name: grad.detach()
        for name, grad in zip(target_layers, plus_tuple)
    }

    # C. Minus pass.
    delta_minus = {
        name: (
            -layer_scales[name] * probes[name]
        ).detach().requires_grad_(True)
        for name in target_layers
    }

    with install_additive_output_hooks(model, delta_minus):
        logits_minus = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits

        loss_minus = ground_truth_next_token_nll(
            logits=logits_minus,
            labels=batch["labels"],
            reduction="sum",
        )

        minus_tuple = torch.autograd.grad(
            outputs=loss_minus,
            inputs=list(delta_minus.values()),
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )

    grad_minus = {
        name: grad.detach()
        for name, grad in zip(target_layers, minus_tuple)
    }

    # D. FD-HVP and GroupTrace reduction.
    group_stats = {}

    for name in target_layers:
        fd_hvp = (
            grad_plus[name].float()
            - grad_minus[name].float()
        ) / (2.0 * float(layer_scales[name]))

        diag_sample = probes[name].float() * fd_hvp

        group_stats[name] = reduce_channels_by_group_mean(
            diag_sample=diag_sample,
            group_index=group_indices[name],
            num_groups=num_groups[name],
        )

    return group_stats
```

---

# 9. Paired-batch central difference

Hai passes plus/minus có cùng input IDs và labels. Có thể concatenate theo batch dimension:

\[
\text{batch}_{\mathrm{pair}}
=
\begin{bmatrix}
\text{batch}_{+}\\
\text{batch}_{-}
\end{bmatrix}.
\]

Input:

```python
input_ids_pair = torch.cat(
    [input_ids, input_ids],
    dim=0,
)

attention_mask_pair = torch.cat(
    [attention_mask, attention_mask],
    dim=0,
)

labels_pair = torch.cat(
    [labels, labels],
    dim=0,
)
```

Intervention:

```python
delta_pair[layer] = torch.cat(
    [
        alpha[layer] * probe[layer],
        -alpha[layer] * probe[layer],
    ],
    dim=0,
).detach().requires_grad_(True)
```

Chạy một lần:

```python
logits_pair = model(
    input_ids=input_ids_pair,
    attention_mask=attention_mask_pair,
    use_cache=False,
).logits

loss_pair = ground_truth_nll_sum(
    logits_pair,
    labels_pair,
)

pair_grads = torch.autograd.grad(
    outputs=loss_pair,
    inputs=list(delta_pair.values()),
    create_graph=False,
    retain_graph=False,
)
```

Tách gradient:

```python
grad_plus = pair_grad[:batch_size]
grad_minus = pair_grad[batch_size:]
```

Ưu điểm:

- một model invocation;
- một backward invocation;
- giảm Python overhead;
- giảm kernel-launch overhead;
- GPU utilization có thể tốt hơn.

Nhược điểm:

- activation memory gần gấp đôi;
- có thể OOM;
- không giảm FLOPs cơ bản;
- attention batch lớn hơn có thể thay đổi kernel selection.

Flags:

```bash
--fd_execution_mode sequential
--fd_execution_mode paired_batch
```

Mặc định:

```text
paired_batch nếu đủ VRAM
sequential nếu OOM
```

---

# 10. Forward finite difference

Nếu baseline gradient đã có:

\[
G_0^{(l)}
=
\nabla_{A^{(l)}}L_{\mathrm{NLL}}(A),
\]

có thể dùng:

\[
\boxed{
H_{\mathrm{NLL}}V
\approx
\frac{G_+-G_0}{\alpha}.
}
\]

Sai số:

\[
O(\alpha).
\]

Central difference có sai số:

\[
O(\alpha^2).
\]

Forward FD chỉ cần:

```text
baseline forward-backward
1 perturbed forward-backward
```

Nếu baseline gradient được reuse từ một bước đã có sẵn, chi phí bổ sung chỉ còn một perturbed pass.

Flags:

```bash
--fd_scheme central
--fd_scheme forward
```

Khuyến nghị:

1. triển khai `central` trước để kiểm tra độ đúng;
2. benchmark với full-NLL autograd HVP;
3. nếu central tốt nhưng runtime chưa đủ nhanh, thử `forward`;
4. chỉ giữ forward mode nếu PPL/correlation không giảm đáng kể.

---

# 11. Signed one-sided finite difference

Một biến thể tiết kiệm pass:

\[
\widehat U
=
\frac{G(A+\alpha V)-G(A)}{\alpha}.
\]

Nó không đối xứng nên bias lớn hơn central FD.

Có thể mỗi calibration micro-batch sample random sign:

\[
\sigma\in\{-1,+1\},
\]

và dùng:

\[
\widehat U
=
\frac{G(A+\sigma\alpha V)-G(A)}{\sigma\alpha}.
\]

Average qua nhiều calibration batches có thể giúp odd-order errors giảm, nhưng không thay thế central difference về độ chính xác.

Đây chỉ là optimization thứ cấp, không phải default.

---

# 12. Chọn perturbation scale \(\alpha_l\)

Đây là hyperparameter số quan trọng nhất.

Nếu \(\alpha_l\) quá nhỏ:

- \(G_+\) và \(G_-\) gần như giống nhau;
- subtraction cancellation;
- BF16/FP16 có thể làm perturbation biến mất;
- relative error lớn.

Nếu \(\alpha_l\) quá lớn:

- đo Hessian trung bình trên vùng rộng;
- Taylor higher-order bias;
- model output có thể thay đổi quá mạnh;
- NLL plus/minus có thể mất ổn định.

## 12.1. Scale theo activation RMS

Một baseline đơn giản:

\[
\boxed{
\alpha_l
=
\eta\cdot\operatorname{RMS}(A^{(l)}).
}
\]

Ưu điểm:

- model-agnostic;
- scale tự động theo layer.

Nhược điểm:

- cần chọn \(\eta\);
- activation RMS không phản ánh trực tiếp quantization error scale.

## 12.2. Scale theo quantization activation error

Khuyến nghị cho bài toán quantization:

\[
E_W^{(l)}
=
Q_0(W^{(l)})-W^{(l)}.
\]

Activation error do initialization:

\[
E_A^{(l)}
=
X^{(l)}{E_W^{(l)}}^\top.
\]

Đặt:

\[
\boxed{
\alpha_l
=
\operatorname{RMS}(E_A^{(l)}).
}
\]

Ưu điểm:

- không cần absolute scale chung;
- bám đúng vùng perturbation quantization;
- tự động theo layer/model;
- ít nguy cơ underflow hơn infinitesimal step.

Nhược điểm:

- estimator gần với curvature trung bình ở quantization scale;
- không còn là exact infinitesimal Hessian tại \(A\);
- nếu initialization rất xấu, step có thể quá lớn.

## 12.3. Automatic floor và ceiling

Để tránh \(\alpha_l\) quá nhỏ hoặc quá lớn:

```python
activation_rms = rms(layer_output.float())
quant_error_rms = rms(activation_quant_error.float())

dtype_floor = (
    torch.finfo(layer_output.dtype).eps
    * activation_rms
    * safety_factor
)

alpha = max(quant_error_rms, dtype_floor)
alpha = min(alpha, max_relative_step * activation_rms)
```

Các hệ số an toàn phải được log và ablate nếu đưa vào production.

---

# 13. Step-size stability test

Không chạy full 128 samples để sweep \(\alpha\).

Trên một smoke-test microbatch:

1. tính FD-HVP với \(\alpha_l\);
2. tính lại với \(\alpha_l/2\);
3. so sánh group statistics.

Metrics:

\[
\operatorname{cosine}(s(\alpha),s(\alpha/2)),
\]

\[
\operatorname{corr}(s(\alpha),s(\alpha/2)),
\]

\[
\frac{\|s(\alpha)-s(\alpha/2)\|}{\|s(\alpha/2)\|+\epsilon}.
\]

Nếu kết quả thay đổi quá mạnh:

- \(\alpha\) có thể quá lớn;
- hoặc quá nhỏ gây cancellation;
- hoặc finite difference không ổn định ở dtype hiện tại.

Cũng nên so với exact autograd HVP trên một model/layer nhỏ.

---

# 14. Loss convention và masking

Phải dùng đúng ground-truth next-token NLL.

Nếu labels chưa shift:

```python
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
```

Mask:

```python
valid_mask = shift_labels != ignore_index
```

Khuyến nghị:

```text
reduction = sum
```

để nhất quán với GuideQuant accumulation.

Cả plus và minus passes phải dùng:

- cùng input;
- cùng labels;
- cùng valid-token mask;
- cùng loss reduction;
- cùng model mode;
- dropout disabled.

Không được để plus và minus passes có stochastic differences khác probe, ví dụ:

- dropout;
- random routing noise;
- sampling;
- nondeterministic data augmentation.

Model phải ở:

```python
model.eval()
```

---

# 15. PSD projection

Ground-truth NLL Hessian theo intermediate activations có thể indefinite.

Do đó raw:

\[
s_{t,k}^{(l)}
\]

có thể âm.

Không clamp ở các bước:

```text
grad_plus
grad_minus
FD-HVP
V * FD-HVP
per-channel sample
```

Thứ tự đúng:

```text
finite difference
→ V ⊙ FD-HVP
→ channel-group average
→ probe average
→ calibration accumulation
→ positive projection
→ weighted GEMM
```

Positive projection:

\[
\boxed{
s_{t,k}^{(l),+}
=
\max(s_{t,k}^{(l)},0).
}
\]

Weight-space Hessian:

\[
\boxed{
H_k^{(l)}
=
{X^{(l)}}^\top
\operatorname{Diag}(s_k^{(l),+})
X^{(l)}.
}
\]

Do:

\[
s_k^{(l),+}\ge0,
\]

nên:

\[
H_k^{(l)}\succeq0.
\]

Damping:

\[
\widetilde H_k^{(l)}
=
H_k^{(l)}+\varepsilon I.
\]

Không apply damping trước weighted GEMM.

---

# 16. Weighted GEMM

Không tạo `torch.diag`.

```python
sqrt_s = torch.sqrt(
    group_stats.clamp_min(0.0)
).unsqueeze(-1)

x_weighted = layer_input.float() * sqrt_s

hessian = (
    x_weighted.transpose(-1, -2)
    @ x_weighted
)

hessian = 0.5 * (
    hessian + hessian.transpose(-1, -2)
)
```

Nếu có nhiều groups:

- loop groups bằng GPU GEMM;
- hoặc batched GEMM nếu memory đủ.

Không tạo tensor khổng lồ:

```text
[num_groups, tokens, d_in]
```

nếu gây OOM.

---

# 17. GPU parallelism

## 17.1. Global multi-layer perturbation

Tất cả target layers được perturb đồng thời trong cùng plus/minus pass.

Không chạy:

```text
2 passes × number_of_layers
```

Mà chạy:

```text
2 global passes cho tất cả layers
```

Probes giữa layers phải độc lập để cross-layer terms triệt tiêu trong kỳ vọng.

## 17.2. Calibration data parallelism

128 calibration sequences có thể shard qua nhiều GPUs.

Mỗi GPU:

1. xử lý calibration shard riêng;
2. sample probes riêng;
3. accumulate local group Hessians;
4. all-reduce/sum cuối cùng.

Với `sum` loss convention:

\[
H_{\mathrm{global}}
=
\sum_gH_g.
\]

Không cần gather full activation/group statistics.

## 17.3. Plus/minus parallelism trên hai GPUs

Nếu có hai GPUs và model copy vừa VRAM:

- GPU 0 chạy plus pass;
- GPU 1 chạy minus pass;
- gradients/group stats được trừ sau cùng.

Ưu điểm:

- wall-clock gần một ordinary forward-backward thay vì hai;
- không doubled-batch memory trên một GPU.

Nhược điểm:

- cần hai model replicas;
- model initialization và communication overhead;
- không phù hợp nếu model chỉ vừa khi model-parallel.

Đây là mode đáng thử khi có hai GPUs độc lập.

## 17.4. Probe parallelism

Nếu dùng nhiều probes:

- probe \(r\) độc lập;
- phân phối probes qua GPUs;
- sum raw group statistics;
- chia tổng số probes.

## 17.5. Layer chunking

Không cần layer chunking về logic nếu additive hooks cho tất cả layers vừa memory.

Nếu intervention tensors quá lớn:

- chia target layers thành chunks;
- chạy plus/minus cho từng chunk.

Tuy nhiên điều này tăng số passes tuyến tính theo số chunks, nên chỉ dùng khi OOM.

## 17.6. Pipeline/model parallelism

Nếu model đã shard qua nhiều GPUs:

- intervention tensor phải nằm cùng device với layer output;
- gradient extraction xảy ra local trên device;
- group reduction nên làm local;
- chỉ all-reduce final \(H_k\) hoặc compact group stats.

Không di chuyển full activation tensors giữa GPUs.

---

# 18. Memory optimization

Finite difference không giữ second-order graph, nhưng vẫn phải giữ activations cho một ordinary backward.

## Sequential mode

Peak memory gần ordinary first-order backward.

Sau plus pass:

- detach `grad_plus`;
- remove hooks;
- delete logits/loss/delta references;
- không giữ plus graph sang minus pass.

## Paired-batch mode

Memory gần gấp đôi activation memory.

Nên:

- giảm microbatch size;
- giữ total calibration samples không đổi;
- dùng gradient checkpointing chỉ nếu ordinary backward path hỗ trợ tốt;
- benchmark kernel choice vì doubled batch có thể nhanh hơn nhiều.

## Chỉ lấy gradient theo intervention tensors

Khuyến nghị mạnh:

```python
grads = torch.autograd.grad(
    outputs=loss,
    inputs=delta_tensors,
    create_graph=False,
    retain_graph=False,
    allow_unused=False,
)
```

thay vì:

```python
loss.backward()
```

Ưu điểm:

- không allocate parameter gradients;
- giảm memory;
- giảm `zero_grad` overhead;
- phù hợp curvature collection hơn.

Autograd vẫn phải reverse qua model để tới intervention tensors, nhưng không cần lưu `.grad` của toàn bộ parameters.

---

# 19. Tối ưu triển khai quan trọng

## 19.1. Reduce group stats ngay

Không giữ toàn bộ `fd_hvp` của mọi layer lâu hơn cần thiết.

```python
fd_hvp = (g_plus.float() - g_minus.float()) / (2 * alpha)
diag_sample = probe.float() * fd_hvp
group_sample = reduce_group(diag_sample)

del fd_hvp, diag_sample
```

## 19.2. Accumulate Hessian theo microbatch

Không cần giữ \(X\) và \(s\) của toàn calibration set.

Sau khi có group stats của microbatch:

\[
H_{k,\mathrm{batch}}
=
X_{\mathrm{batch}}^\top
\operatorname{Diag}(s_{\mathrm{batch}}^+)
X_{\mathrm{batch}}.
\]

Sau đó cộng vào accumulator.

## 19.3. CUDA Graphs

Plus/minus passes có shape cố định và cùng execution graph.

Có thể thử CUDA Graph capture nếu:

- interventions được preallocated;
- input shapes cố định;
- model không có dynamic control flow;
- paired/sequential path ổn định.

Đây là engineering optimization, không thay đổi estimator.

## 19.4. `torch.compile`

Có thể benchmark ordinary plus/minus forward-backward với `torch.compile`.

Finite difference có lợi thế hơn double backward vì compiler/kernel support cho first-order graphs thường tốt hơn.

Không mặc định rằng compile chắc chắn nhanh hơn; benchmark riêng.

## 19.5. Fused additive intervention

Python forward hooks có overhead nếu target layers rất nhiều.

Sau khi correctness ổn định, có thể:

- wrap target linear module;
- thêm intervention trực tiếp trong module forward;
- hoặc dùng lightweight module replacement.

Mục tiêu là tránh hàng trăm Python hook callbacks.

## 19.6. Dedicated Triton/CUDA kernel cho group reduction

Có thể fuse:

```text
subtract gradients
divide by 2 alpha
multiply probe
group reduce
accumulate FP32
```

trong một kernel.

Điều này không giảm cost model passes nhưng giảm memory traffic và kernel launches.

## 19.7. Paired-batch versus dual-GPU

Thứ tự ưu tiên benchmark:

1. sequential: correctness và memory baseline;
2. paired-batch: một GPU, nhiều VRAM;
3. dual-GPU: plus/minus chạy song song;
4. forward FD: giảm số perturbed passes nếu chất lượng chấp nhận được.

---

# 20. Dtype

Khuyến nghị:

- model forward: BF16/FP16 hiện tại;
- intervention tensors: cùng dtype với layer output;
- subtraction gradient: FP32;
- chia \(\alpha\): FP32;
- `probe * fd_hvp`: FP32;
- group accumulators: FP32;
- weighted GEMM: FP32;
- Cholesky: FP32 hoặc logic hiện tại.

Diagnostic:

\[
\text{relative gradient difference}
=
\frac{\|G_+-G_-\|}{\frac12(\|G_+\|+\|G_-\|)+\epsilon}.
\]

Nếu quá nhỏ, \(\alpha\) có thể dưới numerical resolution.

Không nhân arbitrary scale mà quên chia lại.

---

# 21. Probe generation

Probes phải:

- độc lập giữa layers;
- độc lập giữa probe indices;
- reproducible;
- cùng shape với actual output tensor;
- không reuse cùng sign pattern cho nhiều layers.

Seed đề xuất:

```text
base_seed
+ calibration_batch_id
+ probe_id
+ layer_id
```

Rademacher generation:

```python
probe = torch.empty_like(output).bernoulli_(0.5)
probe.mul_(2).sub_(1)
```

Nếu output là tuple:

- perturb đúng tensor downstream sử dụng;
- giữ nguyên các phần tuple khác.

---

# 22. Padding và valid positions

NLL chỉ dùng valid next-token positions, nhưng intermediate outputs tồn tại cho cả padding positions.

Hai lựa chọn:

## A. Probe toàn activation

Đơn giản và ít nguy cơ sai mask.

## B. Mask probes theo valid positions

```python
probe *= activation_valid_mask.unsqueeze(-1)
```

Ưu điểm:

- giảm noise từ padding positions;
- tránh perturbation không ảnh hưởng loss.

Nhược điểm:

- mask phải khớp chính xác positions ở từng architecture;
- causal shift cần kiểm tra.

Phiên bản đầu nên probe toàn activation; masked probes là optimization sau.

---

# 23. Validation với exact autograd HVP

Trước full-model run, dùng toy model hoặc một subset nhỏ.

Với cùng \(V\):

\[
U_{\mathrm{exact}}
=
H_{\mathrm{NLL}}V
\]

và:

\[
U_{\mathrm{FD}}
=
\frac{G_+-G_-}{2\alpha}.
\]

So sánh:

\[
\operatorname{cosine}(U_{\mathrm{FD}},U_{\mathrm{exact}}),
\]

\[
\frac{\|U_{\mathrm{FD}}-U_{\mathrm{exact}}\|}{\|U_{\mathrm{exact}}\|+\epsilon}.
\]

Quan trọng hơn, so group statistics:

\[
s_{\mathrm{FD}}
=
\operatorname{groupmean}(V\odot U_{\mathrm{FD}}),
\]

\[
s_{\mathrm{exact}}
=
\operatorname{groupmean}(V\odot U_{\mathrm{exact}}).
\]

Vì solver chỉ sử dụng group stats, đây là metric chính.

---

# 24. Unit tests bắt buộc

## 24.1. Quadratic function test

Với:

\[
L(a)=\frac12a^\top Ha+b^\top a,
\]

central finite difference phải cho:

\[
U_{\mathrm{FD}}=HV
\]

gần machine precision.

## 24.2. Toy neural network test

So exact autograd HVP và FD-HVP.

## 24.3. Plus/minus symmetry test

Cả hai loss phải finite và cùng convention.

## 24.4. Scale-halving test

So \(\alpha\) và \(\alpha/2\).

## 24.5. Cross-layer probe test

Independent probes phải tốt hơn reused identical patterns giữa layers.

## 24.6. Sequential versus paired-batch

Hai mode phải cho cùng group stats trong tolerance.

## 24.7. `autograd.grad` versus `.backward()`

Gradient intervention phải giống nhau.

## 24.8. PSD test

Sau clamp và weighted GEMM:

\[
\lambda_{\min}(H_k)\ge-\epsilon_{\mathrm{fp}}.
\]

## 24.9. Cholesky test

Sau damping, mọi group phải Cholesky thành công.

---

# 25. Timing logs

Log riêng:

```text
probe generation time
plus forward time
plus backward time
minus forward time
minus backward time
FD subtraction/group reduction time
weighted GEMM time
total curvature time
```

Memory:

```text
peak allocated
peak reserved
```

Compare với exact HVP:

```text
forward time
create-graph backward time
second-order backward time
total HVP time
```

Nếu paired batch:

```text
paired forward time
paired backward time
```

---

# 26. Experiment matrix

Giữ nguyên:

- 128 calibration sequences;
- sequence length;
- model;
- target layers;
- grouping;
- one probe;
- initialization;
- solver;
- damping;
- evaluation.

| Mode | Passes | Sai số |
|---|---:|---:|
| Exact autograd HVP | 1 forward + double backward | reference |
| FD central sequential | 2 forward + 2 backward | \(O(\alpha^2)\) |
| FD central paired | doubled batch | \(O(\alpha^2)\) |
| FD forward | baseline + 1 perturb | \(O(\alpha)\) |
| FD signed one-sided | 1 signed perturb + baseline | \(O(\alpha)\), average có thể giảm odd-order bias |

Metrics:

- runtime;
- peak VRAM;
- Pearson/Spearman với exact HVP group stats;
- seed stability;
- PPL;
- zero-shot accuracy.

---

# 27. Failure conditions

Dừng hướng finite difference nếu:

1. central FD chậm ngang hoặc hơn exact HVP;
2. paired batch OOM và sequential quá chậm;
3. group stats không ổn định theo \(\alpha\);
4. correlation với exact HVP thấp;
5. PPL giảm đáng kể so với exact HVP;
6. cần tuning \(\alpha\) riêng cho từng model/layer quá phức tạp;
7. cancellation quá mạnh trong BF16;
8. forward mode nhanh nhưng chất lượng không đạt;
9. runtime scale vẫn không hợp lý trên model lớn.

Không nên thêm nhiều heuristic correction nếu central FD cơ bản không khớp exact HVP.

---

# 28. Flags đề xuất

```bash
--nll_curvature_mode finite_difference_grouptrace

--fd_scheme central
# central | forward | signed_forward

--fd_execution_mode sequential
# sequential | paired_batch | dual_gpu

--fd_num_probes 1
--fd_probe_seed 0

--fd_scale_mode quant_activation_error
# quant_activation_error | activation_rms | fixed_relative

--fd_scale_multiplier 1.0
--fd_scale_smoke_test 1

--fd_mask_padding_probes 0

--fd_use_autograd_grad 1
--fd_layer_chunk_size 0

--nll_loss_reduction sum
```

Optimization flags:

```bash
--fd_compile_model 0
--fd_cuda_graph 0
--fd_fused_group_reduction 0
```

---

# 29. Recommended implementation order

## Phase 1: correctness

1. additive interventions cho một layer;
2. central FD sequential;
3. exact HVP comparison;
4. group stats comparison;
5. PSD/Cholesky tests.

## Phase 2: global multi-layer

1. independent probes cho mọi target layers;
2. plus/minus perturb đồng thời;
3. cross-layer expectation validation;
4. full GroupTrace Hessian construction.

## Phase 3: runtime

1. dùng `autograd.grad` chỉ theo intervention tensors;
2. paired-batch mode;
3. multi-GPU plus/minus;
4. immediate group reduction;
5. batched weighted GEMM.

## Phase 4: optional acceleration

1. forward FD reuse baseline;
2. signed one-sided FD;
3. fused Triton group-reduction kernel;
4. CUDA Graphs;
5. `torch.compile`.

---

# 30. One-sentence instruction cho Codex

> Replace the existing double-backward global HVP with a model-agnostic central finite-difference HVP over additive interventions at all target linear-layer outputs: sample independent Rademacher probes \(V^{(l)}\), run ground-truth NLL ordinary first-order gradients at \(A^{(l)}+\alpha_lV^{(l)}\) and \(A^{(l)}-\alpha_lV^{(l)}\), compute \(U^{(l)}=(G_+^{(l)}-G_-^{(l)})/(2\alpha_l)\), form \(V^{(l)}\odot U^{(l)}\), reduce over the existing output-channel groups, average probes, clamp only after group/probe accumulation, build \(H_k^{(l)}=X^\top\operatorname{Diag}(s_k^+)X\), preserve the existing damping and solver, and implement sequential, paired-batch and optional dual-GPU plus/minus execution modes with `autograd.grad` restricted to intervention tensors to minimize memory.
