# Thiết kế \(H_{\mathrm{NLL}}\) bằng Group-Trace Hessian–Vector Product cho GuideQuant/LNQ

## 0. Mục tiêu

Tài liệu này mô tả cách xây dựng curvature mới cho **ground-truth next-token NLL** nhằm thay empirical Fisher trong GuideQuant, trong khi giữ nguyên solver LNQ:

- giữ nguyên SqueezeLLM/GuideQuant initialization;
- giữ nguyên coordinate descent cho assignment;
- giữ nguyên exact closed-form codebook update;
- giữ nguyên Cholesky, grouping và stopping logic;
- chỉ thay cách tạo ma trận \(H\) truyền vào LNQ.

Phương pháp được chốt là:

\[
\boxed{\text{Group-Trace Positive NLL Hessian estimated by Hessian–Vector Products}}
\]

Pipeline:

1. bắt đầu trực tiếp từ ground-truth NLL;
2. lấy Hessian thật của NLL theo output \(Z\) của linear layer;
3. không materialize full Hessian, mà ước lượng diagonal bằng Hutchinson HVP;
4. average diagonal curvature theo output-channel group;
5. loại negative curvature để tạo curvature PSD;
6. ánh xạ curvature về weight space:
   \[
   H_k=X^\top\operatorname{Diag}(s_k^+)X;
   \]
7. thêm damping theo logic hiện tại của LNQ để Cholesky ổn định.

Phương pháp này **không dùng squared gradient**, **không dùng pseudo-label**, và **không xấp xỉ \(H_{\mathrm{NLL}}\) bằng teacher-KL/Fisher**.


# 1. Ký hiệu

Xét một linear layer:

\[
Z=XW^\top,
\]

trong đó:

\[
X\in\mathbb R^{T\times d_{\mathrm{in}}},
\quad
W\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}},
\quad
Z\in\mathbb R^{T\times d_{\mathrm{out}}}.
\]

- \(T\): số valid calibration tokens;
- \(x_t\): input activation tại token \(t\);
- \(z_{t,j}\): output của channel \(j\) tại token \(t\);
- \(w_j\): row thứ \(j\) của \(W\);
- \(\hat w_j\): quantized row;
- \(e_j=\hat w_j-w_j\): quantization error.

Các output channels được chia thành groups:

\[
J_1,J_2,\ldots,J_K.
\]

Mỗi group \(J_k\) dùng chung:

\[
H_{\mathrm{NLL},k}\in\mathbb R^{d_{\mathrm{in}}\times d_{\mathrm{in}}}.
\]


# 2. Bắt đầu từ ground-truth NLL

\[
\boxed{
\mathcal L_{\mathrm{NLL}}
=
-\frac{1}{N_{\mathrm{valid}}}
\sum_{t\in\mathcal M}
\log p_W(y_t\mid x_{<t})
}
\]

với \(\mathcal M\) là tập valid tokens.

Có thể dùng `sum` thay vì `mean`, nhưng toàn bộ pipeline phải nhất quán. Nếu code hiện tại accumulate theo `sum`, thì NLL, HVP statistics và Hessian accumulation mới cũng phải theo `sum`.

Loss duy nhất được dùng là:

```text
ground-truth next-token NLL
```

Không dùng:

```text
teacher KL
pseudo-label Fisher
sampled Fisher
soft teacher target
squared activation gradient
```


# 3. Tại sao không dùng empirical Fisher hoặc GGN?

Với logits \(f_W(x)\), probability \(p=\operatorname{softmax}(f_W(x))\), và target \(y\):

\[
\ell_{\mathrm{NLL}}=-\log p_y.
\]

Hessian theo tham số có phân rã:

\[
\boxed{
\nabla_W^2\ell_{\mathrm{NLL}}
=
J_W^\top
\left[\operatorname{Diag}(p)-pp^\top\right]
J_W
+
\sum_{v=1}^{V}
\left(p_v-\mathbf 1[v=y]\right)
\nabla_W^2 f_{W,v}.
}
\]

Đặt:

\[
G_{\mathrm{NLL}}
=
J_W^\top
\left[\operatorname{Diag}(p)-pp^\top\right]
J_W
\]

và:

\[
R_{\mathrm{NLL}}
=
\sum_v
\left(p_v-\mathbf 1[v=y]\right)
\nabla_W^2 f_{W,v}.
\]

Khi đó:

\[
\boxed{
H_{\mathrm{NLL}}
=
G_{\mathrm{NLL}}+R_{\mathrm{NLL}}.
}
\]

Teacher-KL Hessian quanh mô hình gốc chỉ tương ứng với phần GGN/Fisher-like:

\[
H_{\mathrm{KL}}=G_{\mathrm{NLL}}.
\]

Vì vậy, nếu chỉ dùng Fisher, sampled Fisher hoặc GGN, ta không giữ được phần curvature đặc trưng của ground-truth NLL nằm trong \(R_{\mathrm{NLL}}\).

Phương pháp này lấy Hessian trực tiếp từ:

\[
\nabla_Z^2\mathcal L_{\mathrm{NLL}},
\]

nên có thể chứa cả GGN curvature và residual curvature của phần mạng phía sau layer đang xét.


# 4. Taylor expansion theo output \(Z\)

Perturb:

\[
Z\rightarrow Z+\Delta Z.
\]

Taylor bậc hai:

\[
\mathcal L_{\mathrm{NLL}}(Z+\Delta Z)
\approx
\mathcal L_{\mathrm{NLL}}(Z)
+
\left\langle G_Z,\Delta Z\right\rangle
+
\frac12
\operatorname{vec}(\Delta Z)^\top
\mathcal H_Z
\operatorname{vec}(\Delta Z),
\]

trong đó:

\[
G_Z=\nabla_Z\mathcal L_{\mathrm{NLL}},
\]

\[
\boxed{
\mathcal H_Z
=
\nabla_{\operatorname{vec}(Z)}^2
\mathcal L_{\mathrm{NLL}}.
}
\]

Đây là Hessian end-to-end NLL theo activation \(Z\), không phải empirical Fisher.

LNQ hiện chỉ nhận quadratic objective không có linear term. Vì vậy phiên bản này chỉ dùng phần quadratic:

\[
\Delta\mathcal L_{\mathrm{NLL}}
\approx
\frac12
\operatorname{vec}(\Delta Z)^\top
\mathcal H_Z
\operatorname{vec}(\Delta Z).
\]

Tài liệu này chưa thiết kế lại solver để xử lý first-order term.


# 5. Từ weight perturbation sang activation perturbation

Với:

\[
Z=XW^\top,
\qquad
\Delta W=\hat W-W,
\]

ta có:

\[
\Delta Z=X\Delta W^\top.
\]

Với channel \(j\):

\[
\Delta z_j=Xe_j,
\qquad
e_j=\hat w_j-w_j.
\]

Nếu giữ full \(\mathcal H_Z\), objective sẽ chứa tương tác giữa tokens, output channels và weight rows. Ma trận này quá lớn và không phù hợp interface LNQ.

Ta cần approximation có dạng:

\[
\sum_k\sum_{j\in J_k}
e_j^\top H_k e_j.
\]


# 6. Approximation được dùng

## 6.1. Chỉ giữ diagonal của activation Hessian

\[
h_{t,j}
=
\left[\operatorname{diag}(\mathcal H_Z)\right]_{t,j}
=
\frac{\partial^2\mathcal L_{\mathrm{NLL}}}
{\partial z_{t,j}^2}.
\]

Ta bỏ các cross terms:

\[
\frac{\partial^2\mathcal L}
{\partial z_{t,j}\partial z_{t',j'}},
\qquad
(t,j)\ne(t',j').
\]

Quadratic term trở thành:

\[
\frac12\sum_{t,j}h_{t,j}(\Delta z_{t,j})^2.
\]

Mặc dù chỉ giữ diagonal, HVP vẫn đi qua full end-to-end graph. Cross-token và cross-channel interactions của mô hình vẫn ảnh hưởng đến HVP; chỉ ở bước cuối ta lấy diagonal expectation.

## 6.2. Share curvature theo output-channel group

Với group \(J_k\):

\[
\boxed{
s_{t,k}
=
\frac1{|J_k|}
\sum_{j\in J_k}h_{t,j}.
}
\]

Mọi channel \(j\in J_k\) dùng chung token-wise curvature \(s_{t,k}\).

Grouping được giữ để:

- giảm memory;
- giảm số Hessian cần build;
- giữ nguyên LNQ interface.

## 6.3. Positive projection

Raw NLL Hessian có thể có negative curvature:

\[
s_{t,k}<0.
\]

Để có curvature PSD cho LNQ:

\[
\boxed{
s_{t,k}^{+}
=
\max(s_{t,k},0).
}
\]

Thứ tự bắt buộc:

```text
average qua probes
→ average qua channels trong group
→ clamp_min(0)
```

Không clamp từng HVP sample trước khi average.

## 6.4. Ánh xạ về weight space

Với \(j\in J_k\):

\[
\Delta z_{t,j}=x_t^\top e_j.
\]

Do đó:

\[
\frac12
\sum_t
s_{t,k}^+
(x_t^\top e_j)^2
=
\frac12
e_j^\top
\left[
\sum_t s_{t,k}^+x_tx_t^\top
\right]
e_j.
\]

Suy ra:

\[
\boxed{
H_{\mathrm{NLL},k}
=
X^\top
\operatorname{Diag}(s_k^+)
X.
}
\]

Vì \(s_k^+\ge0\):

\[
\boxed{
H_{\mathrm{NLL},k}\succeq0.
}
\]


# 7. Ước lượng diagonal bằng Hutchinson HVP

Full Hessian:

\[
\mathcal H_Z
\in
\mathbb R^{(Td_{\mathrm{out}})\times(Td_{\mathrm{out}})}
\]

không được materialize.

Sample Rademacher probe:

\[
V_r
\in
\{-1,+1\}^{T\times d_{\mathrm{out}}},
\qquad
r=1,\ldots,R.
\]

Tính:

\[
\boxed{
U_r=\mathcal H_ZV_r.
}
\]

Vì:

\[
\mathbb E[V_rV_r^\top]=I,
\]

nên:

\[
\boxed{
\mathbb E
\left[
V_r\odot U_r
\right]
=
\operatorname{diag}(\mathcal H_Z).
}
\]

Estimator:

\[
\widehat h
=
\frac1R
\sum_{r=1}^{R}
V_r\odot(\mathcal H_ZV_r).
\]

Không cần lưu toàn bộ \(\widehat h\); có thể reduce trực tiếp theo group.


# 8. Group-trace estimator cuối cùng

\[
\boxed{
\widehat s_{t,k}
=
\frac{1}{R|J_k|}
\sum_{r=1}^{R}
\sum_{j\in J_k}
V_{r,t,j}
\left[
\mathcal H_ZV_r
\right]_{t,j}.
}
\]

Positive projection:

\[
\boxed{
s_{t,k}^{+}
=
\max(\widehat s_{t,k},0).
}
\]

Weight-space curvature:

\[
\boxed{
H_{\mathrm{NLL},k}
=
X^\top
\operatorname{Diag}(s_k^+)
X.
}
\]

Tên phương pháp:

```text
Group-Trace Positive NLL Hessian
```

Tên đầy đủ:

```text
Group-Trace Positive NLL Hessian estimated by Hutchinson HVP
```


# 9. Tính HVP bằng PyTorch autograd

```python
grad_z = torch.autograd.grad(
    outputs=loss_nll,
    inputs=z,
    create_graph=True,
    retain_graph=True,
)[0]
```

Rademacher probe:

```python
v = torch.empty_like(z).bernoulli_(0.5).mul_(2.0).sub_(1.0)
```

HVP:

```python
hv = torch.autograd.grad(
    outputs=(grad_z * v).sum(),
    inputs=z,
    retain_graph=True,
    create_graph=False,
)[0]
```

Diagonal sample:

```python
diag_sample = v * hv
```

Group reduction:

```python
diag_sample = diag_sample.reshape(
    num_tokens,
    num_groups,
    channels_per_group,
)

group_sample = diag_sample.mean(dim=-1)
```

Accumulate:

```python
group_stats += group_sample
```

Cuối cùng:

```python
group_stats /= num_probes
group_stats = group_stats.clamp_min(0.0)
```

Pseudocode:

```python
def estimate_group_positive_nll_curvature(
    loss_nll,
    z,
    group_index,
    num_groups,
    num_probes,
):
    grad_z = torch.autograd.grad(
        loss_nll,
        z,
        create_graph=True,
        retain_graph=True,
    )[0]

    s_group = torch.zeros(
        z.shape[0],
        num_groups,
        device=z.device,
        dtype=torch.float32,
    )

    for _ in range(num_probes):
        v = rademacher_like(z)

        hv = torch.autograd.grad(
            (grad_z * v).sum(),
            z,
            retain_graph=True,
            create_graph=False,
        )[0]

        diag_sample = (v * hv).float()

        group_sample = reduce_channels_by_group_mean(
            diag_sample,
            group_index,
            num_groups,
        )

        s_group.add_(group_sample)

    s_group.div_(num_probes)
    s_group.clamp_min_(0.0)

    return s_group
```


# 10. Build \(H_{\mathrm{NLL},k}\) hiệu quả

Không tạo `torch.diag(s)`.

Đặt:

\[
\widetilde X_k
=
\operatorname{Diag}(\sqrt{s_k^+})X.
\]

Khi đó:

\[
\boxed{
H_{\mathrm{NLL},k}
=
\widetilde X_k^\top\widetilde X_k.
}
\]

Implementation:

```python
sqrt_s = torch.sqrt(s_group[:, k]).unsqueeze(-1)
x_weighted = x.float() * sqrt_s
h_k = x_weighted.transpose(0, 1) @ x_weighted
```

Đây là phép biến đổi đại số chính xác, dùng GEMM hiệu quả và không tạo matrix diagonal.

Loại asymmetry nhỏ do floating point:

```python
h_k = 0.5 * (h_k + h_k.transpose(-1, -2))
```


# 11. Damping và Cholesky

\(H_{\mathrm{NLL},k}\) là PSD nhưng có thể singular.

Dùng damping hiện có:

\[
\boxed{
\widetilde H_{\mathrm{NLL},k}
=
H_{\mathrm{NLL},k}
+
\varepsilon I.
}
\]

Khi \(\varepsilon>0\):

\[
\widetilde H_{\mathrm{NLL},k}\succ0,
\]

nên Cholesky ổn định.

Yêu cầu:

- tái sử dụng đúng damping logic hiện có của GuideQuant/LNQ;
- damping được apply sau khi build \(H\);
- không cộng damping vào raw HVP samples;
- không thêm balancing coefficient mới.

Objective sau damping:

\[
e^\top(H+\varepsilon I)e
=
e^\top He+\varepsilon\|e\|_2^2.
\]


# 12. Objective truyền vào LNQ

\[
\boxed{
\mathcal Q_{\mathrm{NLL}}
=
\sum_k
\sum_{j\in J_k}
(\hat w_j-w_j)^\top
\widetilde H_{\mathrm{NLL},k}
(\hat w_j-w_j).
}
\]

Giữ nguyên:

```text
SqueezeLLM initialization
coordinate-descent assignment update
exact codebook update
Cholesky solver
group structure
stopping condition
numerical fallback
```

Chỉ thay:

```text
GuideQuant empirical-Fisher H
```

bằng:

```text
Group-Trace Positive NLL Hessian H
```


# 13. Pipeline triển khai hoàn chỉnh

```text
Input:
    pretrained model W
    calibration inputs and labels
    existing GuideQuant/LNQ grouping
    number of HVP probes R
    all existing LNQ settings

For each target linear layer:

A. Capture layer input and output
    X = input activation
    Z = linear-layer output

B. Compute ground-truth NLL
    - use valid next-token labels
    - preserve sum/mean convention

C. Compute first derivative with graph
    G_Z = d L_NLL / d Z
    create_graph = True

D. Estimate grouped diagonal Hessian
    For r = 1,...,R:
        sample Rademacher V_r
        compute U_r = H_Z V_r
        compute V_r ⊙ U_r
        reduce channels by existing groups
        accumulate online

E. Average
    divide by number of probes
    group reduction already divides by group size

F. Positive projection
    s_group = clamp_min(s_group, 0)

G. Build group Hessians
    H_k = X^T Diag(s_k) X
    with weighted GEMM

H. Numerical handling
    symmetrize H_k
    apply existing damping
    verify finite values and Cholesky success

I. Quantize
    pass H_k to unchanged LNQ

Output:
    quantized layer/model
```


# 14. Tối ưu triển khai không đổi toán học

## 14.1. Một HVP cho tất cả groups của cùng layer

Một \(V_r\) và một \(\mathcal H_ZV_r\) đã chứa kết quả cho toàn bộ output channels.

Do đó:

- không chạy HVP riêng cho từng group;
- reduce `v * hv` theo group sau HVP;
- mọi groups trong cùng layer được xử lý song song trên GPU.

## 14.2. Online accumulation

Không lưu:

```text
all probes
all diag samples
full diagonal Hessian
```

Chỉ giữ:

```text
s_group_accumulator: [T, num_groups]
```

## 14.3. Micro-batch calibration

Có thể chia calibration thành micro-batches.

Với convention `sum`:

\[
H_{\mathrm{total}}=\sum_b H_b.
\]

Với convention `mean`, accumulate numerator và valid-token count rồi chia một lần cuối. Không average các micro-batch như nhau nếu số valid tokens khác nhau.

## 14.4. Xử lý từng layer

Double backward tốn memory. Implementation đầu tiên nên:

```text
mỗi target layer → một HVP computation riêng
```

Không naively gộp nhiều layers bằng:

```python
sum((grad_z_l * v_l).sum() for l in layers)
```

vì có thể tạo cross-layer Hessian terms.

Trong cùng một layer, tất cả groups vẫn song song.

## 14.5. Batched probes

Các probes độc lập có thể batch bằng `vmap`/batched HVP nếu memory cho phép.

Khuyến nghị:

1. code sequential probes trước;
2. kiểm tra đúng;
3. benchmark batched probes;
4. chỉ dùng nếu kết quả khớp trong tolerance.

## 14.6. Weighted GEMM

Dùng:

```python
x_weighted = x * sqrt_s
h = x_weighted.T @ x_weighted
```

thay:

```python
x.T @ torch.diag(s) @ x
```

## 14.7. Reuse activation \(X\)

Mọi groups trong layer dùng chung \(X\). Có thể dùng batched GEMM nếu memory đủ; nếu không, loop theo group nhưng mỗi group vẫn dùng GEMM.

## 14.8. Dtype

Khuyến nghị:

- HVP statistics accumulate FP32;
- build \(H_k\) FP32;
- tránh FP16/BF16 accumulation nếu underflow;
- không scale gradient bằng constant mới nếu không hoàn nguyên đúng scale.


# 15. Số HVP probes

\(R\) là estimator budget, không phải weighting hyperparameter.

Khuyến nghị:

```text
R = 1  : smoke test và runtime
R = 2  : variance cơ bản
R = 4  : chỉ khi R=1/2 quá nhiễu
```

Cần đo:

- variance của \(s_{t,k}\) giữa seeds;
- correlation với true symmetric NLL change;
- PPL cuối;
- runtime.

Fix seed để reproducible.


# 16. Fallback nếu double backward không hỗ trợ

Central finite-difference HVP:

\[
Z^+=Z+\delta V,
\qquad
Z^-=Z-\delta V.
\]

Tính:

\[
G^+
=
\nabla_Z\mathcal L_{\mathrm{NLL}}(Z+\delta V),
\]

\[
G^-
=
\nabla_Z\mathcal L_{\mathrm{NLL}}(Z-\delta V).
\]

Khi đó:

\[
\boxed{
\mathcal H_ZV
\approx
\frac{G^+-G^-}{2\delta}.
}
\]

Phần estimator phía sau không đổi.

Ưu điểm:

- ordinary forward/backward;
- tương thích fused kernels hơn.

Nhược điểm:

- cần chọn \(\delta\);
- có finite-difference error;
- hai perturbed gradient passes cho mỗi probe.

Đây chỉ là fallback; ưu tiên exact autograd HVP.


# 17. Unit tests và sanity checks

## Shape

```python
H_nll_k.shape == (d_in, d_in)
```

## Symmetry

\[
\frac{\|H-H^\top\|_F}{\|H\|_F+\epsilon}
\]

phải rất nhỏ.

## PSD trước damping

Trên layer nhỏ:

```python
eig_min = torch.linalg.eigvalsh(H.double()).min()
```

Cho phép negative rất nhỏ do floating point, không cho phép negative đáng kể.

## Cholesky sau damping

```python
H_damped = H + damping * I
L = torch.linalg.cholesky(H_damped)
```

## Không dùng squared gradients

Code path mới không được dùng:

```python
grad_z.square()
```

để tạo curvature.

## Không dùng teacher distribution

Không phụ thuộc vào:

```text
teacher logits
teacher probabilities
pseudo labels
KL loss
```

## Reproducibility

Cùng seed và calibration batch phải cho kết quả giống trong tolerance.

## Group reduction

So sánh group reduction implementation với explicit group mean trên tensor toy.


# 18. Đánh giá approximation trước full PPL

Với weight perturbation \(E\):

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
\sum_k
\sum_{j\in J_k}
e_j^\top H_{\mathrm{NLL},k}e_j.
}
\]

So sánh với empirical-Fisher GuideQuant trên cùng perturbations:

- SqueezeLLM initialization error;
- initial LNQ codebook error;
- final GuideQuant quantization error;
- random codeword reassignments.

Metrics:

- Pearson/Spearman correlation;
- relative scale error;
- ordering accuracy giữa candidate perturbations.


# 19. Debug statistics

Log cho mỗi layer/group:

```text
number of valid tokens
number of probes
mean/std/min/max of raw s_group
fraction(raw s_group < 0)
mean/std/max after clamp
fraction(s_group == 0)

trace(H)
||H||_F
min/max diagonal(H)

damping value
Cholesky success/failure
```

Không tự normalize từng group về cùng norm trong phiên bản đầu tiên.


# 20. Những điều không làm trong phiên bản đầu

Không:

- thay \(H_{\mathrm{NLL}}\) bằng \(H_{\mathrm{KL}}\);
- dùng pseudo-label Fisher;
- dùng squared NLL gradients làm Hessian;
- materialize full Hessian;
- clamp từng HVP sample trước khi average;
- normalize riêng từng group;
- thêm balancing coefficient mới;
- thay initialization;
- thay CD;
- thay exact codebook update;
- thay stopping condition;
- joint-HVP nhiều layers khi chưa kiểm soát cross-layer terms.


# 21. Approximation stack cần ghi rõ

Phương pháp gồm:

1. dùng quadratic Taylor term của end NLL;
2. bỏ first-order term để tương thích LNQ hiện tại;
3. tính Hessian theo output \(Z\);
4. chỉ giữ diagonal của activation Hessian;
5. share curvature theo output-channel group;
6. project group curvature về nonnegative;
7. map về weight space bằng \(X^\top\operatorname{Diag}(s)X\);
8. thêm damping cho Cholesky.

Điểm cốt lõi: approximation bắt đầu từ

\[
\nabla_Z^2\mathcal L_{\mathrm{NLL}},
\]

không bắt đầu từ outer product của gradient.


# 22. Công thức cuối cùng

\[
\boxed{
\mathcal L_{\mathrm{NLL}}
=
-\sum_{t\in\mathcal M}
\log p_W(y_t\mid x_{<t})
}
\]

\[
\boxed{
\mathcal H_Z
=
\nabla_{\operatorname{vec}(Z)}^2
\mathcal L_{\mathrm{NLL}}
}
\]

\[
\boxed{
\widehat s_{t,k}
=
\frac{1}{R|J_k|}
\sum_{r=1}^{R}
\sum_{j\in J_k}
V_{r,t,j}
\left[
\mathcal H_ZV_r
\right]_{t,j}
}
\]

\[
\boxed{
s_{t,k}^{+}
=
\max(\widehat s_{t,k},0)
}
\]

\[
\boxed{
H_{\mathrm{NLL},k}
=
X^\top
\operatorname{Diag}(s_k^+)
X
}
\]

\[
\boxed{
\widetilde H_{\mathrm{NLL},k}
=
H_{\mathrm{NLL},k}
+
\varepsilon I
}
\]

\[
\boxed{
\mathcal Q_{\mathrm{NLL}}
=
\sum_k
\sum_{j\in J_k}
(\hat w_j-w_j)^\top
\widetilde H_{\mathrm{NLL},k}
(\hat w_j-w_j)
}
\]


# 23. One-sentence instruction cho team

> Keep the existing GuideQuant/LNQ initialization and solver unchanged; for each linear layer, compute ground-truth NLL Hessian–vector products with respect to the layer output, use Hutchinson probes to estimate the diagonal curvature, average it over the existing output-channel groups, clamp the averaged curvature to be nonnegative, build \(H_{\mathrm{NLL},k}=X^\top\operatorname{Diag}(s_k^+)X\) with weighted GEMM, apply the existing damping, and pass the resulting group Hessians into the unchanged LNQ coordinate-descent and exact codebook-update pipeline.
