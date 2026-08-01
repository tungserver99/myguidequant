# Thiết kế \(H_{\mathrm{NLL}}\) bằng Global Multi-Layer Group-Trace HVP cho GuideQuant/LNQ

## 0. Mục tiêu của tài liệu

Tài liệu này mô tả cách xây dựng curvature mới cho **ground-truth next-token NLL** nhằm thay empirical Fisher trong GuideQuant, trong khi vẫn giữ nguyên solver LNQ:

- giữ nguyên SqueezeLLM/GuideQuant initialization;
- giữ nguyên coordinate descent cho assignment;
- giữ nguyên exact closed-form codebook update;
- giữ nguyên Cholesky, output-channel grouping và stopping logic;
- chỉ thay cách tạo ma trận \(H\) truyền vào LNQ.

Phương pháp được chốt là:

\[
\boxed{
\text{Global Multi-Layer Group-Trace Positive NLL Hessian}
}
\]

được ước lượng bằng **một global Hessian–vector product cho tất cả target linear layers trong mỗi probe**, thay vì chạy HVP riêng cho từng layer.

Ý tưởng cốt lõi:

1. bắt đầu trực tiếp từ ground-truth NLL;
2. lấy Hessian thật của NLL theo output của các linear layers;
3. dùng một Hutchinson probe độc lập cho mỗi layer;
4. cộng tất cả gradient–probe inner products thành một scalar chung;
5. chạy một second-order backward chung để thu HVP cho mọi layer;
6. dùng tính zero-mean và độc lập của probes để các cross-layer terms triệt tiêu trong kỳ vọng;
7. average diagonal curvature theo output-channel group;
8. project curvature về nonnegative để tạo \(H\) PSD;
9. ánh xạ về weight space:
   \[
   H_k=X^\top\operatorname{Diag}(s_k^+)X;
   \]
10. thêm damping theo logic hiện tại của LNQ để Cholesky ổn định.

Phương pháp này:

- không dùng squared gradient;
- không dùng pseudo-label;
- không xấp xỉ \(H_{\mathrm{NLL}}\) bằng teacher-KL/Fisher;
- không chạy một HVP riêng cho từng layer trong configuration mặc định;
- có chi phí cùng bậc với GuideQuant khi dùng một global probe, dù wall-clock thực tế vẫn có thể lớn hơn do second-order autograd.

---

# 1. Ký hiệu

Giả sử mô hình có \(L\) target linear layers.

Với layer \(l\):

\[
Z^{(l)}=X^{(l)}{W^{(l)}}^\top,
\]

trong đó:

\[
X^{(l)}
\in
\mathbb R^{T_l\times d_{\mathrm{in}}^{(l)}},
\]

\[
W^{(l)}
\in
\mathbb R^{d_{\mathrm{out}}^{(l)}
\times
d_{\mathrm{in}}^{(l)}},
\]

\[
Z^{(l)}
\in
\mathbb R^{T_l\times d_{\mathrm{out}}^{(l)}}.
\]

Ký hiệu:

- \(T_l\): số valid activation positions của layer \(l\);
- \(x_t^{(l)}\): input activation tại position \(t\);
- \(z_{t,j}^{(l)}\): output của channel \(j\);
- \(w_j^{(l)}\): row thứ \(j\) của \(W^{(l)}\);
- \(\hat w_j^{(l)}\): quantized row;
- \(e_j^{(l)}=\hat w_j^{(l)}-w_j^{(l)}\): quantization error.

Các output channels của layer \(l\) được chia thành groups:

\[
J_1^{(l)},J_2^{(l)},\ldots,J_{K_l}^{(l)}.
\]

Mỗi group \(J_k^{(l)}\) dùng chung:

\[
H_{\mathrm{NLL},k}^{(l)}
\in
\mathbb R^{
d_{\mathrm{in}}^{(l)}
\times
d_{\mathrm{in}}^{(l)}
}.
\]

Để trình bày global HVP, vector hóa và ghép output của tất cả target layers:

\[
\mathbf z
=
\begin{bmatrix}
\operatorname{vec}(Z^{(1)})\\
\vdots\\
\operatorname{vec}(Z^{(L)})
\end{bmatrix}.
\]

---

# 2. Bắt đầu từ ground-truth NLL

Ground-truth next-token NLL:

\[
\boxed{
\mathcal L_{\mathrm{NLL}}
=
-\frac{1}{N_{\mathrm{valid}}}
\sum_{t\in\mathcal M}
\log p_W(y_t\mid x_{<t})
}
\]

trong đó:

- \(\mathcal M\) là tập valid next-token labels;
- padding và ignored labels không được đưa vào loss;
- \(N_{\mathrm{valid}}=|\mathcal M|\).

Có thể dùng convention `sum` thay vì `mean`, nhưng toàn bộ pipeline phải nhất quán.

Nếu code GuideQuant/LNQ hiện tại accumulate theo `sum`, thì:

- NLL loss;
- HVP statistics;
- weighted covariance;
- accumulation giữa calibration micro-batches

đều phải dùng cùng convention `sum`.

Loss duy nhất trong pass này là:

```text
ground-truth next-token NLL
```

Không dùng:

```text
teacher KL
teacher probabilities
pseudo labels
sampled Fisher
soft teacher targets
squared activation gradients
```

---

# 3. Tại sao không dùng empirical Fisher hoặc GGN?

Với logits:

\[
f_W(x)\in\mathbb R^V,
\]

probability:

\[
p=\operatorname{softmax}(f_W(x)),
\]

và target token \(y\):

\[
\ell_{\mathrm{NLL}}=-\log p_y.
\]

Hessian NLL theo tham số có phân rã:

\[
\boxed{
\nabla_W^2\ell_{\mathrm{NLL}}
=
J_W^\top
\left[
\operatorname{Diag}(p)-pp^\top
\right]
J_W
+
\sum_{v=1}^{V}
\left(
p_v-\mathbf 1[v=y]
\right)
\nabla_W^2 f_{W,v}.
}
\]

Đặt:

\[
G_{\mathrm{NLL}}
=
J_W^\top
\left[
\operatorname{Diag}(p)-pp^\top
\right]
J_W
\]

và:

\[
R_{\mathrm{NLL}}
=
\sum_v
\left(
p_v-\mathbf 1[v=y]
\right)
\nabla_W^2 f_{W,v}.
\]

Khi đó:

\[
\boxed{
H_{\mathrm{NLL}}
=
G_{\mathrm{NLL}}
+
R_{\mathrm{NLL}}.
}
\]

Teacher-KL Hessian quanh mô hình gốc tương ứng với phần GGN/Fisher-like:

\[
H_{\mathrm{KL}}
=
G_{\mathrm{NLL}}.
\]

Vì vậy, Fisher, sampled Fisher hoặc GGN không giữ được toàn bộ residual/model curvature đặc trưng của ground-truth NLL.

Phương pháp trong tài liệu này bắt đầu trực tiếp từ:

\[
\nabla^2_{Z^{(l)}}\mathcal L_{\mathrm{NLL}},
\]

nên có thể giữ cả:

\[
\text{GGN curvature}
+
\text{residual curvature của suffix phía sau layer }l.
\]

---

# 4. Taylor expansion theo output của một layer

Với layer \(l\), perturb:

\[
Z^{(l)}
\rightarrow
Z^{(l)}
+
\Delta Z^{(l)}.
\]

Taylor bậc hai:

\[
\begin{aligned}
\mathcal L_{\mathrm{NLL}}
\left(
Z^{(l)}+\Delta Z^{(l)}
\right)
\approx\;&
\mathcal L_{\mathrm{NLL}}
\left(
Z^{(l)}
\right)
\\
&+
\left\langle
G_l,
\Delta Z^{(l)}
\right\rangle
\\
&+
\frac12
\operatorname{vec}
\left(
\Delta Z^{(l)}
\right)^\top
\mathcal H_{ll}
\operatorname{vec}
\left(
\Delta Z^{(l)}
\right),
\end{aligned}
\]

trong đó:

\[
G_l
=
\nabla_{Z^{(l)}}
\mathcal L_{\mathrm{NLL}},
\]

và:

\[
\boxed{
\mathcal H_{ll}
=
\nabla^2_{\operatorname{vec}(Z^{(l)})}
\mathcal L_{\mathrm{NLL}}.
}
\]

Đây là activation Hessian block mà ta muốn xấp xỉ cho layer \(l\).

LNQ hiện chỉ nhận quadratic objective không có linear term. Vì vậy phiên bản này chỉ dùng phần quadratic:

\[
\Delta\mathcal L_{\mathrm{NLL}}^{(l)}
\approx
\frac12
\operatorname{vec}
\left(
\Delta Z^{(l)}
\right)^\top
\mathcal H_{ll}
\operatorname{vec}
\left(
\Delta Z^{(l)}
\right).
\]

Tài liệu này chưa thiết kế lại LNQ để xử lý first-order term.

---

# 5. Full multi-layer activation Hessian

Ghép output của tất cả target layers thành \(\mathbf z\).

Full Hessian theo các activation variables là block matrix:

\[
\boxed{
\mathbb H
=
\nabla_{\mathbf z}^2
\mathcal L_{\mathrm{NLL}}
=
\begin{bmatrix}
\mathcal H_{11} & \cdots & \mathcal H_{1L}\\
\vdots & \ddots & \vdots\\
\mathcal H_{L1} & \cdots & \mathcal H_{LL}
\end{bmatrix}.
}
\]

Trong đó:

\[
\mathcal H_{lm}
=
\frac{
\partial^2
\mathcal L_{\mathrm{NLL}}
}{
\partial\operatorname{vec}(Z^{(l)})
\partial\operatorname{vec}(Z^{(m)})
}.
\]

Mục tiêu của layer \(l\) chỉ là diagonal của block:

\[
\operatorname{diag}(\mathcal H_{ll}).
\]

Ta không cần materialize:

- full \(\mathbb H\);
- block \(\mathcal H_{ll}\);
- cross-layer blocks \(\mathcal H_{lm}\).

Cross-layer blocks chỉ xuất hiện tạm thời trong global HVP và sẽ triệt tiêu trong kỳ vọng nhờ probes độc lập.

---

# 6. Từ weight perturbation sang activation perturbation

Với layer \(l\):

\[
Z^{(l)}
=
X^{(l)}
{W^{(l)}}^\top.
\]

Weight perturbation:

\[
\Delta W^{(l)}
=
\hat W^{(l)}-W^{(l)}.
\]

Khi đó:

\[
\Delta Z^{(l)}
=
X^{(l)}
{\Delta W^{(l)}}^\top.
\]

Với output channel \(j\):

\[
\Delta z_j^{(l)}
=
X^{(l)}e_j^{(l)}.
\]

Nếu giữ full activation Hessian block, objective chứa tương tác:

- giữa activation positions;
- giữa output channels;
- giữa nhiều weight rows.

LNQ hiện cần objective dạng:

\[
\sum_{k}
\sum_{j\in J_k^{(l)}}
{e_j^{(l)}}^\top
H_k^{(l)}
e_j^{(l)}.
\]

Vì vậy cần đưa activation Hessian về cấu trúc tương thích LNQ.

---

# 7. Approximation tương thích LNQ

## 7.1. Chỉ giữ diagonal của block \(\mathcal H_{ll}\)

Đặt:

\[
h_{t,j}^{(l)}
=
\left[
\operatorname{diag}
\left(
\mathcal H_{ll}
\right)
\right]_{t,j}
=
\frac{
\partial^2
\mathcal L_{\mathrm{NLL}}
}{
\partial
\left(
z_{t,j}^{(l)}
\right)^2
}.
\]

Ta bỏ các cross terms bên trong layer:

\[
\frac{
\partial^2
\mathcal L
}{
\partial z_{t,j}^{(l)}
\partial z_{t',j'}^{(l)}
},
\qquad
(t,j)\ne(t',j').
\]

Quadratic term trở thành:

\[
\frac12
\sum_{t,j}
h_{t,j}^{(l)}
\left(
\Delta z_{t,j}^{(l)}
\right)^2.
\]

Mặc dù cuối cùng chỉ giữ diagonal, HVP vẫn đi qua full end-to-end graph.

## 7.2. Share curvature theo output-channel group

Với group \(J_k^{(l)}\):

\[
\boxed{
s_{t,k}^{(l)}
=
\frac{1}{|J_k^{(l)}|}
\sum_{j\in J_k^{(l)}}
h_{t,j}^{(l)}.
}
\]

Mọi channel trong cùng group dùng chung token-wise curvature.

Grouping được giữ để:

- tương thích LNQ hiện tại;
- giảm số matrix \(H\);
- giảm peak memory;
- dùng một group Hessian cho nhiều weight rows.

## 7.3. Positive projection

Raw NLL Hessian có thể có negative curvature:

\[
s_{t,k}^{(l)}<0.
\]

LNQ cần curvature PSD để codebook subproblem là convex. Vì vậy dùng:

\[
\boxed{
s_{t,k}^{(l)+}
=
\max
\left(
s_{t,k}^{(l)},0
\right).
}
\]

Thứ tự bắt buộc:

```text
average qua probes
→ average qua channels trong group
→ clamp_min(0)
```

Không clamp từng probe trước khi average.

## 7.4. Ánh xạ về weight space

Với \(j\in J_k^{(l)}\):

\[
\Delta z_{t,j}^{(l)}
=
{x_t^{(l)}}^\top e_j^{(l)}.
\]

Do đó:

\[
\frac12
\sum_t
s_{t,k}^{(l)+}
\left(
{x_t^{(l)}}^\top e_j^{(l)}
\right)^2
\]

bằng:

\[
\frac12
{e_j^{(l)}}^\top
\left[
\sum_t
s_{t,k}^{(l)+}
x_t^{(l)}
{x_t^{(l)}}^\top
\right]
e_j^{(l)}.
\]

Suy ra:

\[
\boxed{
H_{\mathrm{NLL},k}^{(l)}
=
{X^{(l)}}^\top
\operatorname{Diag}
\left(
s_k^{(l)+}
\right)
X^{(l)}.
}
\]

Vì:

\[
s_k^{(l)+}\ge0,
\]

nên:

\[
\boxed{
H_{\mathrm{NLL},k}^{(l)}
\succeq0.
}
\]

---

# 8. Global Hutchinson HVP cho tất cả layers

## 8.1. Independent block probes

Với probe \(r\), sample một Rademacher tensor riêng cho từng layer:

\[
V_r^{(l)}
\in
\{-1,+1\}^{\operatorname{shape}(Z^{(l)})}.
\]

Ghép lại:

\[
\mathbf v_r
=
\begin{bmatrix}
\operatorname{vec}(V_r^{(1)})\\
\vdots\\
\operatorname{vec}(V_r^{(L)})
\end{bmatrix}.
\]

Yêu cầu bắt buộc:

\[
V_r^{(1)},\ldots,V_r^{(L)}
\]

phải độc lập hoặc ít nhất uncorrelated và zero-mean giữa các layers.

Không reuse cùng một sign pattern cho nhiều layers.

## 8.2. Một scalar chung

Tính first derivatives:

\[
G_l
=
\nabla_{Z^{(l)}}
\mathcal L_{\mathrm{NLL}},
\qquad
l=1,\ldots,L.
\]

Với probe \(r\), tạo:

\[
\boxed{
S_r
=
\sum_{l=1}^{L}
\left\langle
G_l,
V_r^{(l)}
\right\rangle.
}
\]

Sau đó chạy một second-order backward chung:

\[
\boxed{
U_r^{(l)}
=
\nabla_{Z^{(l)}}S_r.
}
\]

Dưới dạng block matrix:

\[
\mathbf u_r
=
\mathbb H\mathbf v_r.
\]

Với layer \(l\):

\[
U_r^{(l)}
=
\sum_{m=1}^{L}
\mathcal H_{lm}V_r^{(m)}.
\]

## 8.3. Vì sao cross-layer terms không làm sai estimator?

Xét element \(i\) trong output vector của layer \(l\):

\[
v_{r,i}^{(l)}
u_{r,i}^{(l)}
=
v_{r,i}^{(l)}
\sum_m
\sum_q
[\mathcal H_{lm}]_{iq}
v_{r,q}^{(m)}.
\]

Lấy kỳ vọng:

\[
\begin{aligned}
\mathbb E
\left[
v_{r,i}^{(l)}
u_{r,i}^{(l)}
\right]
&=
\sum_q
[\mathcal H_{ll}]_{iq}
\mathbb E
\left[
v_{r,i}^{(l)}
v_{r,q}^{(l)}
\right]
\\
&\quad+
\sum_{m\ne l}
\sum_q
[\mathcal H_{lm}]_{iq}
\mathbb E
\left[
v_{r,i}^{(l)}
v_{r,q}^{(m)}
\right].
\end{aligned}
\]

Với Rademacher probes:

\[
\mathbb E
\left[
v_{r,i}^{(l)}
v_{r,q}^{(l)}
\right]
=
\mathbf 1[i=q].
\]

Do probes giữa layers độc lập và zero-mean:

\[
\mathbb E
\left[
v_{r,i}^{(l)}
v_{r,q}^{(m)}
\right]
=
0,
\qquad
m\ne l.
\]

Vì vậy:

\[
\boxed{
\mathbb E
\left[
V_r^{(l)}
\odot
U_r^{(l)}
\right]
=
\operatorname{diag}
\left(
\mathcal H_{ll}
\right).
}
\]

Như vậy:

- một global HVP có thể estimate diagonal Hessian block cho mọi layer;
- cross-layer terms không tạo bias;
- cross-layer terms có thể làm tăng variance khi \(R\) nhỏ;
- group averaging và calibration averaging giúp giảm variance.

---

# 9. Group-trace estimator cuối cùng

Với \(R\) global probes:

\[
\boxed{
\widehat s_{t,k}^{(l)}
=
\frac{
1
}{
R|J_k^{(l)}|
}
\sum_{r=1}^{R}
\sum_{j\in J_k^{(l)}}
V_{r,t,j}^{(l)}
U_{r,t,j}^{(l)}.
}
\]

Sau đó:

\[
\boxed{
s_{t,k}^{(l)+}
=
\max
\left(
\widehat s_{t,k}^{(l)},0
\right).
}
\]

Cuối cùng:

\[
\boxed{
H_{\mathrm{NLL},k}^{(l)}
=
{X^{(l)}}^\top
\operatorname{Diag}
\left(
s_k^{(l)+}
\right)
X^{(l)}.
}
\]

Tên ngắn:

```text
Global Group-Trace Positive NLL Hessian
```

Tên đầy đủ:

```text
Global Multi-Layer Group-Trace Positive NLL Hessian estimated by Hutchinson HVP
```

---

# 10. PyTorch implementation của global HVP

## 10.1. Capture target activations

Cần giữ actual output tensors của các target linear layers trong forward graph:

```python
layer_outputs = []
layer_inputs = []
```

Không detach `z` trước khi tính NLL.

Hook phải lưu:

```python
x = input[0]
z = output
```

Nếu output là tuple, lấy đúng tensor mà downstream graph sử dụng.

## 10.2. First derivatives cho tất cả layers

```python
grads_z = torch.autograd.grad(
    outputs=loss_nll,
    inputs=layer_outputs,
    create_graph=True,
    retain_graph=True,
    allow_unused=False,
)
```

Đây là một first-order reverse pass chung cho toàn bộ target layers.

## 10.3. Một global HVP cho mỗi probe

```python
for probe_idx in range(num_probes):
    probes = [
        rademacher_like(z, generator=layer_generators[layer_idx])
        for layer_idx, z in enumerate(layer_outputs)
    ]

    probe_scalar = sum(
        (g * v).sum()
        for g, v in zip(grads_z, probes)
    )

    hvps = torch.autograd.grad(
        outputs=probe_scalar,
        inputs=layer_outputs,
        retain_graph=(probe_idx + 1 < num_probes),
        create_graph=False,
        allow_unused=False,
    )

    for layer_idx, (v, hv) in enumerate(zip(probes, hvps)):
        diag_sample = (v * hv).float()

        group_sample = reduce_channels_by_group_mean(
            diag_sample,
            group_index[layer_idx],
            num_groups[layer_idx],
        )

        group_accumulators[layer_idx].add_(group_sample)
```

Sau probes:

```python
for layer_idx in range(num_layers):
    group_accumulators[layer_idx].div_(num_probes)
    group_accumulators[layer_idx].clamp_min_(0.0)
```

## 10.4. Pseudocode đầy đủ

```python
def collect_global_group_trace_nll_stats(
    loss_nll,
    layer_outputs,
    group_indices,
    num_groups_per_layer,
    num_probes,
    probe_generators,
):
    grads_z = torch.autograd.grad(
        outputs=loss_nll,
        inputs=layer_outputs,
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )

    accumulators = [
        torch.zeros(
            z.shape[0],
            num_groups,
            device=z.device,
            dtype=torch.float32,
        )
        for z, num_groups in zip(
            layer_outputs,
            num_groups_per_layer,
        )
    ]

    for probe_idx in range(num_probes):
        probes = [
            rademacher_like(
                z,
                generator=probe_generators[layer_idx],
            )
            for layer_idx, z in enumerate(layer_outputs)
        ]

        probe_scalar = sum(
            (g * v).sum()
            for g, v in zip(grads_z, probes)
        )

        hvps = torch.autograd.grad(
            outputs=probe_scalar,
            inputs=layer_outputs,
            retain_graph=(probe_idx + 1 < num_probes),
            create_graph=False,
            allow_unused=False,
        )

        for layer_idx, (v, hv) in enumerate(
            zip(probes, hvps)
        ):
            diag_sample = (v * hv).float()

            group_sample = reduce_channels_by_group_mean(
                diag_sample,
                group_indices[layer_idx],
                num_groups_per_layer[layer_idx],
            )

            accumulators[layer_idx].add_(group_sample)

    for accumulator in accumulators:
        accumulator.div_(num_probes)

        # Clamp only after averaging all probes.
        accumulator.clamp_min_(0.0)

    return accumulators
```

## 10.5. Probe independence

Không dùng một tensor `v` rồi slice hoặc broadcast cùng pattern sang mọi layer.

Safe options:

- một generator riêng cho mỗi layer;
- một global generator nhưng sample tuần tự tensor riêng cho từng layer;
- seed theo tuple:
  ```text
  base_seed, calibration_batch_id, probe_id, layer_id
  ```

Probe của cùng layer giữa các probe indices cũng phải độc lập.

---

# 11. Build \(H_{\mathrm{NLL},k}^{(l)}\) hiệu quả

Công thức:

\[
H_{\mathrm{NLL},k}^{(l)}
=
{X^{(l)}}^\top
\operatorname{Diag}
\left(
s_k^{(l)+}
\right)
X^{(l)}.
\]

Không tạo:

```python
torch.diag(s)
```

Đặt:

\[
\widetilde X_k^{(l)}
=
\operatorname{Diag}
\left(
\sqrt{s_k^{(l)+}}
\right)
X^{(l)}.
\]

Khi đó:

\[
\boxed{
H_{\mathrm{NLL},k}^{(l)}
=
{\widetilde X_k^{(l)}}^\top
\widetilde X_k^{(l)}.
}
\]

Implementation:

```python
sqrt_s = torch.sqrt(
    group_stats[layer_idx][:, group_idx]
).unsqueeze(-1)

x_weighted = layer_inputs[layer_idx].float() * sqrt_s

h_k = (
    x_weighted.transpose(0, 1)
    @ x_weighted
)
```

Symmetrize để loại floating-point asymmetry:

```python
h_k = 0.5 * (
    h_k + h_k.transpose(-1, -2)
)
```

Phép biến đổi này không thay đổi bản chất toán học.

---

# 12. Damping và Cholesky

\(H_{\mathrm{NLL},k}^{(l)}\) là PSD theo construction nhưng có thể singular.

Dùng damping hiện có của GuideQuant/LNQ:

\[
\boxed{
\widetilde H_{\mathrm{NLL},k}^{(l)}
=
H_{\mathrm{NLL},k}^{(l)}
+
\varepsilon I.
}
\]

Khi \(\varepsilon>0\):

\[
\widetilde H_{\mathrm{NLL},k}^{(l)}
\succ0,
\]

nên Cholesky ổn định.

Yêu cầu:

- reuse đúng damping logic hiện có;
- apply damping sau khi build \(H\);
- không cộng damping vào raw HVP samples;
- không thêm balancing coefficient giữa layers hoặc groups;
- không normalize từng \(H_k\) về cùng norm.

Objective sau damping:

\[
e^\top(H+\varepsilon I)e
=
e^\top He
+
\varepsilon\|e\|_2^2.
\]

---

# 13. Objective truyền vào LNQ

Với layer \(l\):

\[
\boxed{
\mathcal Q_{\mathrm{NLL}}^{(l)}
=
\sum_k
\sum_{j\in J_k^{(l)}}
{e_j^{(l)}}^\top
\widetilde H_{\mathrm{NLL},k}^{(l)}
e_j^{(l)}.
}
\]

Giữ nguyên:

```text
SqueezeLLM initialization
coordinate-descent assignment
exact codebook update
Cholesky solver
output-channel grouping
stopping condition
numerical fallback
```

Chỉ thay:

```text
GuideQuant empirical-Fisher curvature
```

bằng:

```text
Global Group-Trace Positive NLL Hessian
```

---

# 14. Pipeline triển khai hoàn chỉnh

```text
Input:
    pretrained model W
    calibration inputs and ground-truth labels
    target linear layers
    existing GuideQuant/LNQ grouping
    num_probes R
    existing LNQ settings

For each calibration micro-batch:

A. Forward once
    - capture X^(l) and Z^(l) for every target linear layer
    - compute ground-truth next-token NLL

B. First-order reverse pass once
    - compute G_l = d L_NLL / d Z^(l)
      for all target layers simultaneously
    - use create_graph=True

C. For each global probe r = 1,...,R
    - sample an independent Rademacher V_r^(l)
      for every layer
    - build:
          S_r = sum_l <G_l, V_r^(l)>
    - run one global second-order backward:
          U_r^(l) = d S_r / d Z^(l)
      for all layers simultaneously
    - compute:
          V_r^(l) ⊙ U_r^(l)
    - reduce output channels into existing groups
    - accumulate online

D. Average probes
    - divide every layer/group accumulator by R

E. Positive projection
    - clamp group curvature to nonnegative
    - clamp only after probe averaging

F. Build weight-space Hessians
    - for every layer/group:
          H_k^(l)
          =
          X^(l)^T Diag(s_k^(l)+) X^(l)
    - use weighted GEMM
    - accumulate across calibration micro-batches

G. Numerical handling
    - symmetrize H
    - apply existing damping
    - verify finite values
    - verify Cholesky success

H. Quantize
    - pass H matrices to unchanged LNQ
    - same initialization
    - same CD
    - same exact codebook update
    - same stopping

Output:
    quantized model
```

---

# 15. Runtime và memory

## 15.1. So với per-layer HVP

Không dùng:

```text
L layers × R HVPs
```

Mà dùng:

```text
R global HVPs cho toàn bộ L layers
```

Với:

```text
R = 1
```

mỗi calibration micro-batch cần về khái niệm:

```text
1 forward
1 first backward với create_graph=True
1 global second-order backward
weighted GEMMs giống GuideQuant
```

Do đó phương pháp có chi phí **cùng bậc** với GuideQuant, thay vì tăng tuyến tính theo số layer.

Không cam kết wall-clock bằng đúng GuideQuant vì:

- second-order backward đắt hơn first-order backward;
- `create_graph=True` tăng memory và overhead;
- một số fused kernels có double-backward chậm;
- global graph có peak memory cao hơn.

Mục tiêu thực tế:

```text
same runtime order as GuideQuant
```

không phải:

```text
exactly identical runtime
```

## 15.2. Số probes

Khuyến nghị mặc định:

```bash
--nll_hvp_probes 1
```

Chỉ dùng:

```bash
--nll_hvp_probes 2
```

nếu estimator giữa các seeds quá nhiễu hoặc curvature correlation tăng rõ rệt.

Mỗi probe bổ sung thêm một global second-order backward.

## 15.3. Memory

Global HVP yêu cầu giữ:

- outputs của target layers;
- first derivatives với computation graph;
- saved tensors cần cho second-order backward.

Memory cao hơn GuideQuant.

Đổi lại, nó tránh lặp second-order backward theo từng layer.

---

# 16. Tối ưu triển khai không thay đổi bản chất toán học

## 16.1. Một HVP cho mọi layers và mọi groups

Đây là tối ưu chính:

- một global HVP phục vụ mọi target layers;
- trong mỗi layer, cùng HVP phục vụ mọi output groups;
- không chạy HVP riêng cho layer;
- không chạy HVP riêng cho group.

## 16.2. Online group reduction

Không cần giữ:

```text
all probe tensors
all diagonal samples
full Hessian diagonal
```

Sau khi nhận `hvps`, xử lý lần lượt từng layer:

```python
diag_sample = (v * hv).float()
group_sample = reduce_channels_by_group_mean(...)
accumulator.add_(group_sample)
```

Sau khi reduce, giải phóng reference tới `v`, `hv` và `diag_sample` sớm nhất có thể.

## 16.3. Direct Hessian accumulation theo micro-batch

Không nhất thiết giữ toàn bộ:

```text
s_group for all calibration tokens
```

Sau khi average probes và positive projection trong micro-batch:

\[
H_{k,\mathrm{batch}}^{(l)}
=
{X_{\mathrm{batch}}^{(l)}}^\top
\operatorname{Diag}
\left(
s_{k,\mathrm{batch}}^{(l)+}
\right)
X_{\mathrm{batch}}^{(l)}.
\]

Accumulate:

\[
H_k^{(l)}
\leftarrow
H_k^{(l)}
+
H_{k,\mathrm{batch}}^{(l)}.
\]

Điều này giảm memory và không thay đổi phép tính khi các activation positions của micro-batches là disjoint.

Nếu loss dùng global `mean`, phải scale mỗi batch bằng đúng số valid tokens trước khi cộng.

## 16.4. Calibration data parallelism

Calibration examples độc lập nên có thể shard qua nhiều GPUs.

Mỗi GPU:

1. chạy global HVP trên calibration shard của mình;
2. build local \(H_k^{(l)}\);
3. all-reduce hoặc sum các local Hessians.

Với convention `sum`:

\[
H_{\mathrm{global}}
=
\sum_g H_g.
\]

Với convention `mean`, sum numerators và valid-token counts rồi chia một lần.

Đây là parallelism chính xác, không đổi estimator.

## 16.5. Weighted GEMM

Dùng:

```python
x_weighted = x * sqrt_s
h = x_weighted.T @ x_weighted
```

thay:

```python
x.T @ torch.diag(s) @ x
```

Đây là phép biến đổi đại số chính xác.

## 16.6. Batched group GEMM

Mọi groups trong một layer dùng chung \(X^{(l)}\).

Nếu memory đủ, có thể tạo:

```text
x_weighted:
[num_groups, T, d_in]
```

và dùng batched GEMM.

Nếu memory không đủ, loop theo group nhưng mỗi group vẫn dùng GEMM GPU.

## 16.7. Activation checkpointing

Có thể dùng checkpointing để đổi compute lấy memory, với điều kiện implementation hỗ trợ second-order autograd.

Trong PyTorch, cần kiểm tra cụ thể checkpoint mode và custom kernels. Không giả định mọi reentrant checkpoint path đều hỗ trợ double backward.

Checkpointing không thay đổi toán học nhưng có thể tăng runtime.

## 16.8. Layer chunking là memory fallback

Nếu toàn bộ target layers không vừa memory, chia target layers thành chunks:

```text
chunk 1: layers 1...C
chunk 2: layers C+1...2C
...
```

Trong mỗi chunk:

- first derivatives cho layers trong chunk;
- independent probes cho layers trong chunk;
- một global HVP cho toàn chunk.

Estimator của diagonal block trong chunk vẫn unbiased:

\[
\mathbb E
\left[
V^{(l)}\odot U^{(l)}
\right]
=
\operatorname{diag}(\mathcal H_{ll}).
\]

Không cần đưa layers ngoài chunk vào probe để estimate \(\mathcal H_{ll}\).

Đánh đổi:

- chunk càng lớn: nhanh hơn, memory cao hơn;
- chunk càng nhỏ: memory thấp hơn, nhiều reverse passes hơn.

Configuration mặc định nên là:

```text
all target layers in one global chunk
```

Layer chunking chỉ là fallback khi OOM.

## 16.9. Dtype

Khuyến nghị:

- model forward theo dtype hiện tại;
- cast `v * hv` sang FP32 trước group reduction;
- accumulate group statistics trong FP32;
- build \(H_k\) trong FP32;
- Cholesky theo dtype hiện tại của LNQ hoặc FP32/FP64 debug;
- không thêm arbitrary gradient multiplier.

Nếu second-order signal underflow trong BF16/FP16, ưu tiên tăng precision của HVP path thay vì nhân gradient bằng một constant không được hoàn nguyên.

## 16.10. Fused kernels

Một số FlashAttention, fused MLP hoặc custom CUDA kernels có thể:

- không hỗ trợ double backward;
- hỗ trợ nhưng chậm;
- trả gradient bậc hai không ổn định.

Cần có smoke test trên một batch trước khi chạy calibration đầy đủ.

Nếu kernel không hỗ trợ, ưu tiên đổi sang mathematically equivalent unfused implementation cho curvature collection pass.

---

# 17. Flags đề xuất

```bash
--nll_curvature_mode global_group_trace_hvp
--nll_hvp_probes 1
--nll_hvp_seed 0
--nll_hvp_layer_chunk_size 0
```

Ý nghĩa:

```text
nll_hvp_probes:
    số global HVP probes

nll_hvp_seed:
    seed cơ sở để sinh probe reproducible

nll_hvp_layer_chunk_size:
    0  = tất cả target layers trong một global HVP
    >0 = số target layers trong mỗi memory fallback chunk
```

`nll_hvp_layer_chunk_size` là computational setting, không thay đổi objective target. Nó chỉ thay đổi runtime/memory và random cross-term variance realization.

---

# 18. Fallback nếu double backward không được hỗ trợ

Finite-difference HVP theo từng layer vẫn có thể dùng để debug:

\[
\mathcal H_{ll}V^{(l)}
\approx
\frac{
G_l(Z^{(l)}+\delta V^{(l)})
-
G_l(Z^{(l)}-\delta V^{(l)})
}{
2\delta
}.
\]

Tuy nhiên fallback này:

- cần chọn \(\delta\);
- cần hai perturbed gradient passes;
- khó giữ runtime tương đương GuideQuant;
- phù hợp validation hoặc một số layer, không phải full-model mặc định.

Nếu mục tiêu là full-model runtime cùng bậc GuideQuant, exact global autograd HVP là implementation ưu tiên.

---

# 19. Unit tests và sanity checks

## 19.1. Toy-model exact Hessian test

Trên một mạng nhỏ:

1. materialize exact full multi-layer activation Hessian \(\mathbb H\);
2. lấy exact:
   \[
   \operatorname{diag}(\mathcal H_{ll});
   \]
3. average global HVP estimator qua nhiều probes;
4. kiểm tra convergence về exact diagonal.

## 19.2. Global HVP so với per-layer HVP

Trên toy model:

- estimator global;
- estimator chạy riêng từng layer.

Khi số probes đủ lớn, hai estimator phải hội tụ về cùng diagonal block.

## 19.3. Cross-layer probe independence

Test cố tình reuse cùng probe pattern giữa layers và so với independent probes.

Production code phải dùng independent probes.

## 19.4. Shape

Với mọi layer/group:

```python
H_nll[layer][group].shape == (d_in, d_in)
```

## 19.5. Symmetry

\[
\frac{
\|H-H^\top\|_F
}{
\|H\|_F+\epsilon
}
\]

phải rất nhỏ sau symmetrization.

## 19.6. PSD trước damping

Vì:

\[
H=X^\top\operatorname{Diag}(s^+)X,
\]

nên \(H\) PSD theo construction.

Trên layer nhỏ:

```python
eig_min = torch.linalg.eigvalsh(
    H.double()
).min()
```

Cho phép negative rất nhỏ do floating point, không cho phép negative đáng kể.

## 19.7. Cholesky sau damping

```python
H_damped = H + damping * I
L = torch.linalg.cholesky(H_damped)
```

Phải thành công trên mọi group.

## 19.8. Không dùng squared gradients

Code path mới không được dùng:

```python
grad_z.square()
```

để tạo curvature.

`grad_z` chỉ dùng trong:

```python
sum((grad_z * probe).sum())
```

## 19.9. Không dùng teacher distribution

Pass NLL-HVP không phụ thuộc vào:

```text
teacher logits
teacher probabilities
pseudo labels
KL loss
```

## 19.10. Reproducibility

Cùng:

```text
model
calibration shard
seed
probe count
layer order
```

phải tái lập được statistics trong tolerance.

## 19.11. Chunk invariance in expectation

So sánh:

```text
all layers in one chunk
```

và:

```text
multiple layer chunks
```

Khi average đủ probes/seeds, hai cách phải hội tụ về cùng block diagonals.

---

# 20. Đánh giá approximation trước full-model PPL

Với một weight perturbation \(E\), dùng symmetric NLL difference:

\[
\boxed{
C_{\mathrm{true}}(E)
=
\frac{
\mathcal L(W+\alpha E)
+
\mathcal L(W-\alpha E)
-
2\mathcal L(W)
}{
\alpha^2
}.
}
\]

First-order term tự triệt tiêu.

Prediction:

\[
\boxed{
C_{\mathrm{pred}}(E)
=
\sum_l
\sum_k
\sum_{j\in J_k^{(l)}}
{e_j^{(l)}}^\top
H_{\mathrm{NLL},k}^{(l)}
e_j^{(l)}.
}
\]

So sánh với empirical-Fisher GuideQuant trên cùng perturbations:

- SqueezeLLM initialization error;
- initial LNQ codebook error;
- final GuideQuant quantization error;
- random codeword reassignments.

Metrics:

- Pearson correlation;
- Spearman correlation;
- relative scale error;
- pairwise ordering accuracy.

Mục tiêu là curvature mới dự đoán thứ tự và mức độ NLL degradation tốt hơn empirical Fisher.

---

# 21. Debug statistics

Log cho mỗi layer/group:

```text
number of valid tokens
number of probes
layer chunk size

mean/std/min/max of raw group curvature
fraction(raw curvature < 0)

mean/std/max after positive projection
fraction(curvature == 0 after clamp)

trace(H)
||H||_F
min/max diagonal(H)

damping value
Cholesky success/failure
```

Log ở mức global HVP:

```text
forward time
first-backward time
each global-HVP time
weighted-GEMM time
peak allocated memory
peak reserved memory
```

Để đánh giá variance:

```text
relative difference between seeds
relative difference R=1 vs R=2
global-HVP vs per-layer-HVP on toy/small layer
```

Không tự normalize từng group hoặc layer về cùng norm trong phiên bản đầu tiên.

---

# 22. Những điều không làm trong phiên bản đầu

Không:

- thay \(H_{\mathrm{NLL}}\) bằng \(H_{\mathrm{KL}}\);
- dùng pseudo-label Fisher;
- dùng squared NLL gradients làm Hessian;
- materialize full Hessian;
- chạy HVP riêng cho từng layer trong default configuration;
- reuse correlated probes giữa layers;
- clamp từng probe trước khi average;
- normalize riêng từng group/layer;
- thêm balancing coefficient mới;
- thay initialization;
- thay coordinate descent;
- thay exact codebook update;
- thay stopping condition;
- scale gradients bằng arbitrary constant;
- dùng layer chunking mặc định nếu global all-layer HVP vẫn vừa memory.

---

# 23. Approximation stack cần ghi rõ

Phương pháp không phải exact full weight Hessian. Nó gồm:

1. dùng quadratic Taylor term của end NLL;
2. bỏ first-order term để tương thích LNQ;
3. target activation Hessian block của từng linear layer;
4. estimate mọi block diagonal bằng global Hutchinson HVP;
5. dùng independent layer probes để cross-layer terms triệt tiêu trong kỳ vọng;
6. chỉ giữ diagonal của mỗi activation Hessian block;
7. average diagonal curvature theo output-channel group;
8. project group curvature về nonnegative;
9. map về weight space bằng:
   \[
   X^\top\operatorname{Diag}(s)X;
   \]
10. thêm damping để Cholesky ổn định.

Điểm cốt lõi:

\[
\boxed{
\text{Estimator bắt đầu từ }
\nabla^2\mathcal L_{\mathrm{NLL}},
\text{ không bắt đầu từ outer product của gradient.}
}
\]

---

# 24. Công thức cuối cùng

Ground-truth NLL:

\[
\boxed{
\mathcal L_{\mathrm{NLL}}
=
-\sum_{t\in\mathcal M}
\log p_W(y_t\mid x_{<t})
}
\]

Full multi-layer activation Hessian:

\[
\boxed{
\mathbb H
=
\nabla_{\mathbf z}^2
\mathcal L_{\mathrm{NLL}}
}
\]

Independent block probes:

\[
\boxed{
\mathbf v_r
=
\left[
\operatorname{vec}(V_r^{(1)});
\ldots;
\operatorname{vec}(V_r^{(L)})
\right]
}
\]

Global HVP:

\[
\boxed{
\mathbf u_r
=
\mathbb H\mathbf v_r
}
\]

Layer block estimator:

\[
\boxed{
\mathbb E
\left[
V_r^{(l)}
\odot
U_r^{(l)}
\right]
=
\operatorname{diag}
\left(
\mathcal H_{ll}
\right)
}
\]

Group estimator:

\[
\boxed{
\widehat s_{t,k}^{(l)}
=
\frac{
1
}{
R|J_k^{(l)}|
}
\sum_{r=1}^{R}
\sum_{j\in J_k^{(l)}}
V_{r,t,j}^{(l)}
U_{r,t,j}^{(l)}
}
\]

Positive projection:

\[
\boxed{
s_{t,k}^{(l)+}
=
\max
\left(
\widehat s_{t,k}^{(l)},0
\right)
}
\]

Weight-space Hessian:

\[
\boxed{
H_{\mathrm{NLL},k}^{(l)}
=
{X^{(l)}}^\top
\operatorname{Diag}
\left(
s_k^{(l)+}
\right)
X^{(l)}
}
\]

Damped matrix:

\[
\boxed{
\widetilde H_{\mathrm{NLL},k}^{(l)}
=
H_{\mathrm{NLL},k}^{(l)}
+
\varepsilon I
}
\]

LNQ objective:

\[
\boxed{
\mathcal Q_{\mathrm{NLL}}
=
\sum_l
\sum_k
\sum_{j\in J_k^{(l)}}
{e_j^{(l)}}^\top
\widetilde H_{\mathrm{NLL},k}^{(l)}
e_j^{(l)}
}
\]

---

# 25. One-sentence instruction cho team

> Keep the existing GuideQuant/LNQ initialization and solver unchanged; in each calibration forward pass, capture all target linear-layer inputs and outputs, compute ground-truth NLL gradients with respect to all layer outputs in one create-graph reverse pass, sample independent Rademacher probes for every layer, form one global gradient–probe scalar and use one global second-order backward per probe to estimate every layer’s Hessian-block diagonal simultaneously, average the resulting \(V^{(l)}\odot U^{(l)}\) values over the existing output-channel groups, clamp only after probe averaging, build \(H_{\mathrm{NLL},k}^{(l)}={X^{(l)}}^\top\operatorname{Diag}(s_k^{(l)+})X^{(l)}\) with weighted GEMM, apply the existing damping, and pass the resulting group Hessians into the unchanged LNQ coordinate-descent and exact codebook-update pipeline.
