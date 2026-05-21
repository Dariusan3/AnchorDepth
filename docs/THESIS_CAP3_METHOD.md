# Thesis — Method Chapter
## Chapter 3: Method

This chapter describes the technical components of the proposed system, organized to mirror the order in which a training example is processed. Section 3.1 introduces notation and the high-level pipeline. Sections 3.2–3.5 describe the four building blocks that are largely standard: the Depth Pro backbone, the photometric loss, the pose estimator, and the differentiable warper. Sections 3.6–3.8 describe the three components that constitute the methodological contributions of this thesis: the LoRA adapter integration into Depth Pro, the consistency loss that anchors adaptation to the zero-shot baseline (the main contribution), and the numerical-stability recipe that makes mixed-precision LoRA training reliable at the foundation-model scale. Section 3.9 summarises the training pipeline end-to-end.

---

### 3.1 Notation and System Overview

A training sample consists of a target frame $I_t$ and two adjacent source frames $I_{t-1}, I_{t+1}$ from a monocular video sequence. The target frame is associated with a camera intrinsic matrix $K \in \mathbb{R}^{4 \times 4}$ (homogeneous) read from the KITTI calibration file. Following Monodepth2 convention, the network operates simultaneously at two resolutions: $(W_d, H_d) = (1536, 1536)$ for the Depth Pro encoder input, and $(W_p, H_p) = (416, 128)$ for the PoseNet input and the photometric loss computation. The 1536×1536 input is required by Depth Pro's multi-resolution patch encoder; the smaller pose resolution is dictated by the 12 GB VRAM budget.

The system's components are:

1. A **depth network** $f_\theta : \mathbb{R}^{3 \times H_d \times W_d} \to \mathbb{R}^{1 \times H_d \times W_d}$ producing canonical inverse depth for the target frame. $\theta$ comprises the frozen Depth Pro weights $\theta_{\text{frozen}}$, the LoRA adapters $\theta_{\text{LoRA}}$ (trainable, $\approx 2.36$ M parameters at rank 8), the decoder $\theta_{\text{dec}}$ and the depth head $\theta_{\text{head}}$ (also trainable).

2. A **pose network** $g_\phi : (\mathbb{R}^{3 \times H_p \times W_p})^2 \to \mathbb{R}^6$ that maps a (target, source) frame pair to a 6-degree-of-freedom relative pose. Two implementations are evaluated: a trainable ResNet-18 PoseNet (Section 3.4.1) and a precomputed VGGT cache (Section 3.4.2).

3. A **differentiable warper** $W : (\mathbb{R}^{3 \times H_p \times W_p}, \mathbb{R}^{1 \times H_p \times W_p}, \mathbb{R}^{4 \times 4}, \mathbb{R}^{4 \times 4}, \mathbb{R}^{4 \times 4}) \to \mathbb{R}^{3 \times H_p \times W_p}$ that, given a source image, the target depth, the target-to-source transformation, and the camera intrinsics, produces the source frame warped onto the target view.

4. A **loss function** combining the Monodepth2 photometric reconstruction loss with an edge-aware smoothness regulariser and, in our main contribution, a consistency loss against precomputed zero-shot predictions (Section 3.7).

Total trainable parameters in our v15 configuration: $\approx 34$ M out of $\approx 966$ M ($3.6\%$). Total trainable parameters in v16 (VGGT poses, PoseNet bypassed): $\approx 22.4$ M ($2.4\%$).

---

### 3.2 Depth Pro Backbone

We adopt Depth Pro (Bochkovskii et al., 2024) as the depth network. Depth Pro is a 952 million-parameter model with four functional sub-modules.

**Patch encoder.** A DINOv2-Large Vision Transformer (304 M parameters, 24 transformer blocks, hidden dimension 1024) that processes a multi-scale pyramid of image patches. At inference the 1536×1536 input is decomposed into patches at three pyramidal levels (1×, 2×, 4× downsampling) and each level is run through the same patch encoder. This shared-weight pyramid is what gives Depth Pro its characteristic sharpness on fine structures.

**Image encoder.** A second DINOv2-Large ViT (also 304 M parameters, 24 blocks, hidden 1024) that processes the full image at $384 \times 384$ resolution to inject global context into the prediction. The two encoders together contain 48 transformer blocks, each with 4 attention projection matrices (Q, K, V, output) — 192 matrices total. This is the surface on which LoRA adapters are placed (Section 3.6).

**Decoder.** A 19.67 M-parameter multi-scale fusion decoder that combines patch- and image-encoder features at four resolutions, producing a single feature map at half the input resolution (768 × 768). The decoder is structurally similar to the MiDaS/DPT decoder but with skip connections from both encoders.

**Depth head.** A small 0.40 M-parameter convolutional head that maps the decoder output to a single-channel canonical inverse depth map at the input resolution (1536 × 1536). The output is interpreted as $\hat{d}_{\text{canon}} = d^{-1}_{\text{metric}} \cdot \frac{f_{px}}{W_d}$, a focal-length-normalised inverse depth that becomes metric only after multiplication by $W_d / f_{px}$ where $f_{px}$ is the camera's horizontal focal length in pixels.

**FOV head (frozen throughout).** A 6.6 M-parameter head that predicts the camera's horizontal field of view in degrees. Depth Pro normally uses this prediction to compute $f_{px}$ at inference, but on KITTI we found the FOV head's predictions to be unreliable (Section 3.8.2), so the FOV head is kept in eval mode and its output is replaced by the ground-truth $f_{px}$ from the KITTI calibration file. The FOV head's weights are never updated.

The two encoders are by far the largest part of Depth Pro (608 M of 952 M parameters). Full fine-tuning of these encoders is infeasible on 12 GB; LoRA (Section 3.6) addresses this.

---

### 3.3 Self-Supervised Photometric Loss

The training signal comes from photometric reconstruction. For each (target, source) frame pair we warp the source to the target view using the predicted depth and pose; the difference between the warped source and the actual target measures both depth and pose quality. We adopt the Monodepth2 loss formulation (Godard et al., 2019) with three terms.

**Per-pixel reprojection error.** Following Monodepth2, we use a weighted combination of L1 and SSIM:
$$pe(I_a, I_b) = \frac{\alpha}{2}(1 - \text{SSIM}(I_a, I_b)) + (1 - \alpha) \, |I_a - I_b|_1$$
with $\alpha = 0.85$. SSIM is computed in $3 \times 3$ windows.

**Per-pixel minimum across source frames.** For each pixel $p$ in the target, the photometric error is computed against each warped source independently, and the per-pixel minimum is retained:
$$L_p(p) = \min_{s \in \{-1, +1\}} pe\bigl(I_t(p), \hat{I}_{s \to t}(p)\bigr)$$
This automatically handles pixels visible in one source but occluded in the other.

**Auto-masking.** Pixels for which the photometric error from the warped source frame exceeds the error from the unwarped source frame are masked out. This corresponds to pixels that are either static (camera not moving), moving with the camera (parallax-free), or in regions of homogeneous texture where the warping cannot reduce the error. Formally:
$$\mu(p) = \mathbb{1}\left[\min_s pe(I_t, \hat{I}_{s \to t})(p) < \min_s pe(I_t, I_s)(p)\right]$$
The mask is applied before averaging, so only pixels where the warping demonstrably helps contribute to the loss.

**Smoothness term.** To prevent depth from being too noisy in textureless regions, we add an edge-aware smoothness regulariser on the inverse depth:
$$L_s = \mathbb{E}_p \left[\, e^{-|\nabla_x I_t(p)|} |\nabla_x d^{-1}(p)| + e^{-|\nabla_y I_t(p)|} |\nabla_y d^{-1}(p)| \,\right]$$
The exponential weights reduce the regularisation strength at image edges (where depth discontinuities are expected).

The combined photometric loss is:
$$L_{\text{photo}} = \mathbb{E}_p \bigl[\mu(p) \cdot L_p(p)\bigr] + \lambda_s L_s$$
with smoothness weight $\lambda_s = 10^{-3}$ in our experiments. This is implemented in `src/depth_pro/selfsup/losses.py`.

---

### 3.4 Pose Estimation

Two pose estimators are evaluated.

#### 3.4.1 Trained PoseNet (ResNet-18 based)

The classical Monodepth2 PoseNet (Godard et al., 2019) is a ResNet-18 adapted to accept a 6-channel input formed by concatenating the target and source frames along the channel axis. The output is a 6-vector $(\boldsymbol{\omega}, \boldsymbol{t})$ with axis-angle rotation $\boldsymbol{\omega} \in \mathbb{R}^3$ and translation $\boldsymbol{t} \in \mathbb{R}^3$. The output is scaled by a small constant ($\times 0.01$) to keep the initial poses near identity.

The axis-angle vector is converted to a rotation matrix via the Rodrigues formula:
$$R = I + \sin\theta \, [\hat{\boldsymbol{\omega}}]_\times + (1-\cos\theta)\, [\hat{\boldsymbol{\omega}}]_\times^2, \quad \theta = \|\boldsymbol{\omega}\|, \quad \hat{\boldsymbol{\omega}} = \boldsymbol{\omega}/\theta$$
The full 4×4 target-to-source transformation is then:
$$T_{t \to s} = \begin{pmatrix} R & \boldsymbol{t} \\ \boldsymbol{0}^\top & 1 \end{pmatrix}$$
PoseNet has approximately 11.91 M parameters and is trained jointly with the depth network using the photometric loss. This is the configuration used in runs v15 and v17, v18.

#### 3.4.2 VGGT Precomputed Poses

In runs v16 and v19 we substitute PoseNet with poses precomputed offline by VGGT (Wang et al., CVPR 2025 Best Paper, 1.2 B parameters). Because VGGT cannot co-reside with Depth Pro on 12 GB during training, the precomputation is performed once over the entire KITTI training set before depth fine-tuning begins.

The precomputation procedure (implemented in `precompute_vggt_poses.py`):

1. For each KITTI training triplet $(I_{t-1}, I_t, I_{t+1})$, load the three frames as a 3-image batch.
2. Run VGGT-1B inference with `torch.amp.autocast(dtype=torch.bfloat16)` for memory efficiency.
3. Extract the 9-dimensional pose encoding per frame from the VGGT output and convert to a 3×4 extrinsic matrix via `pose_encoding_to_extri_intri()`. The result is a world-to-camera transformation in OpenCV convention.
4. Promote each 3×4 extrinsic to a 4×4 homogeneous matrix $E_i$ by appending the bottom row $(0,0,0,1)$.
5. Compute the two relative poses needed for training. For each source frame $s \in \{t-1, t+1\}$, the **target-to-source** transformation is:
$$T_{t \to s} = E_s \cdot E_t^{-1}$$
where $E_t^{-1}$ is the inverse of the world-to-target extrinsic (computed via $E^{-1} = \begin{pmatrix} R^\top & -R^\top \boldsymbol{t} \\ \boldsymbol{0}^\top & 1 \end{pmatrix}$ for an orthonormal $R$).
6. Save the dictionary $\{i \mapsto \{T_{\text{prev}}, T_{\text{next}}\}\}$ (sample index to relative pose pair) as a single `.pt` file (`vggt_poses_train_s6.pt`, about 5 MB).

At training time the cached relative poses are loaded by the `KITTIRawDataset` and exposed through the batch dictionary; the trainable PoseNet is skipped entirely. Total precomputation time on the same RTX 4070 Ti: approximately 90 minutes for 6,635 triplets at stride 6.

---

### 3.5 Differentiable Warping

Given the predicted target depth $D \in \mathbb{R}^{H_p \times W_p}$, the target-to-source transformation $T \in \mathbb{R}^{4 \times 4}$, the camera intrinsics $K \in \mathbb{R}^{4 \times 4}$, and the source image $I_s \in \mathbb{R}^{3 \times H_p \times W_p}$, the warper produces a reconstruction of the target image as observed from the source view.

The procedure is the standard backproject-transform-reproject pipeline (Garg et al., 2016; Zhou et al., 2017):

1. **Backproject.** Each target pixel $(u, v)$ with predicted depth $D(u, v)$ is unprojected to a 3D camera-frame coordinate:
$$\boldsymbol{P}_t = D(u, v) \cdot K^{-1} \cdot (u, v, 1)^\top$$

2. **Transform.** The 3D point is mapped into the source camera frame:
$$\boldsymbol{P}_s = T \cdot \boldsymbol{P}_t$$

3. **Reproject.** The source-frame point is projected back to 2D using the same intrinsics:
$$(u', v')^\top = K \cdot \boldsymbol{P}_s \quad \text{then divide by} \quad Z$$

4. **Sample.** The source image is sampled at the predicted projection coordinates using `torch.nn.functional.grid_sample` with `mode='bilinear'` and `padding_mode='border'`. This is the only non-trivially differentiable operation in the pipeline; the autograd convention for `grid_sample` is well-established and produces stable gradients.

Steps 1–3 are batched matrix operations on tensors of shape $(B, 3, HW)$ for efficiency. The entire warper is implemented in `src/depth_pro/selfsup/warping.py` as a single `nn.Module`.

A subtle but critical detail concerns the units of $D$ and $K$. Both must be in the same coordinate system. In our setup, the predicted canonical inverse depth $\hat{d}_{\text{canon}}$ is converted to metric depth via:
$$D = \frac{W_p}{f_{px,p}} \cdot \frac{1}{\hat{d}_{\text{canon}}}$$
where $f_{px,p}$ is the focal length expressed at the pose resolution (i.e., the original KITTI $f_{px}$ scaled by $W_p / W_{\text{orig}}$). This scaling is performed before the warper is invoked; the warper itself uses $K$ at pose resolution as well, so the two are consistent.

---

### 3.6 LoRA Adapters for Depth Pro

**Motivation.** Fully fine-tuning Depth Pro's 952 M parameters requires twice as much memory in optimiser states (AdamW maintains two FP32 moments per parameter) plus activations for backpropagation. On a 12 GB GPU this is impossible. LoRA addresses this by injecting tiny trainable matrices into each attention projection and freezing the original weights.

**LoRA module.** For each frozen weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ inside an attention block, we introduce two trainable matrices:
$$A \in \mathbb{R}^{r \times d_{\text{in}}}, \quad B \in \mathbb{R}^{d_{\text{out}} \times r}, \quad r \ll \min(d_{\text{out}}, d_{\text{in}})$$
The adapted forward pass is:
$$y = W x + \frac{\alpha}{r} \cdot B A x$$
where $\alpha/r$ is a fixed scaling factor. $A$ is initialised with Kaiming uniform; $B$ is initialised with zeros. This **identity-at-init** property ensures the LoRA-augmented model produces exactly the zero-shot Depth Pro output at the start of training — a critical property for our consistency-loss formulation (Section 3.7), since training begins exactly at the baseline we wish to anchor to.

**Placement.** We apply LoRA to all four attention projection matrices in each transformer block of both Depth Pro encoders:
- Q, K, V projections (combined into a single `qkv` matrix of shape $3d \times d$, $d = 1024$)
- Output projection (shape $d \times d$)

With 24 blocks per encoder × 2 encoders × 2 LoRA matrices per block (qkv and proj) × 2 trainable tensors per LoRA matrix (A and B) = **192 trainable LoRA tensors total**. With rank $r = 8$, $\alpha = 8$ (so $\alpha / r = 1$), the total LoRA parameter count is approximately **2.36 M** — a 257× reduction relative to fine-tuning the encoders directly.

**Combined trainable surface.** In our v15 configuration the trainable parameters are:
| Component | Parameters | Status |
|-----------|-----------|--------|
| LoRA in patch encoder | 1.18 M | trainable |
| LoRA in image encoder | 1.18 M | trainable |
| Decoder | 19.67 M | trainable |
| Depth head | 0.40 M | trainable |
| PoseNet (ResNet-18 + pose head) | 11.91 M | trainable |
| FOV head | 6.6 M | frozen |
| Patch encoder backbone | 304 M | frozen |
| Image encoder backbone | 304 M | frozen |
| **Total trainable** | **34.34 M** (**3.6%** of 966 M) | |

In v16, where VGGT replaces PoseNet, the trainable count drops by 11.91 M to **22.43 M** (2.4%).

**Optimiser configuration.** We use AdamW with three discriminative learning rate groups:
- LoRA parameters: $\eta_{\text{LoRA}} = 10^{-5}$, weight decay 0.01
- Decoder and head: $\eta_{\text{dec}} = 10^{-4}$, weight decay $10^{-4}$
- PoseNet: $\eta_{\text{pose}} = 10^{-4}$, weight decay 0 (or empty group in v16)

The lower LR for LoRA is intentional: the pretrained encoder features are nearly optimal at initialisation and only need gentle adjustment. The learning rate schedule is **CosineAnnealingWarmRestarts** with $T_0 = 10$ epochs, which cycles the LR rather than letting it decay to near-zero — this prevents the loss plateau we observed in an earlier (uncommitted) experiment with plain CosineAnnealing.

---

### 3.7 Consistency Loss — Main Contribution

The defining methodological contribution of this thesis is a **consistency loss** that anchors the fine-tuned model's predictions to the zero-shot Depth Pro baseline. The motivation, validated empirically in Chapter 4, is that pure self-supervised photometric optimisation systematically degrades a strong depth foundation model — the photometric objective and the depth-evaluation metric disagree once the starting point is near-optimal on the latter.

The consistency loss directly addresses this objective–metric gap.

**Formulation.** Let $d_{\text{pred}}(x)$ be the metric depth predicted by the (LoRA-augmented) Depth Pro at training time for pixel $x$, and let $d_{\text{zs}}(x)$ be the corresponding metric depth predicted by the zero-shot Depth Pro for the same pixel. Both are produced at pose resolution $(W_p, H_p) = (416, 128)$. The consistency loss is:
$$L_{\text{cons}} = \mathbb{E}_x \bigl[\, w(x) \cdot d_{\text{form}}(d_{\text{pred}}(x), d_{\text{zs}}(x))\, \bigr]$$
where $d_{\text{form}}$ is a per-pixel distance and $w(x)$ is an optional pixel weight.

**Distance forms.** Two distance forms are evaluated:
- **L1 metric (v15, v16).** $d_{\text{form}}(a, b) = |a - b|$. Simple, scale-aware, computationally cheap.
- **L1 log (v17, v18, v19).** $d_{\text{form}}(a, b) = |\log a - \log b|$. Directly minimises log-depth differences, which couples more directly to the RMSElog and δ-threshold evaluation metrics.

**Pixel weighting.** Two pixel weighting schemes are evaluated:
- **Uniform (default, v15, v18).** $w(x) = 1$.
- **Edge-aware (v16).** Compute the gradient magnitude of $d_{\text{zs}}$ as a proxy for depth discontinuities; reduce the consistency weight on high-gradient pixels (so photometric loss can refine boundaries) and increase it on smooth regions (so the zero-shot anchor dominates).
$$w_{\text{edge}}(x) = \exp(-2 \cdot \bar{e}(x))$$
where $\bar{e}(x)$ is the gradient magnitude at $x$ normalised by its median.
- **Depth-power (v17, ablated negatively).** $w(x) = (d_{\text{zs}}(x) / d_{\text{max}})^p$, $p = 2$. Emphasises distant pixels at the expense of near pixels. Empirically this caused near-pixel drift and is reported as a negative result.

**Total training loss.** The full training objective combines the Monodepth2 photometric loss with the consistency anchor:
$$L_{\text{total}} = L_{\text{photo}} + \lambda \cdot L_{\text{cons}}$$
$\lambda$ controls the strength of the anchor:
- $\lambda = 0$ recovers pure photometric training (configurations v6–v14, which all degrade zero-shot).
- $\lambda = 10$ (v15) makes the anchor approximately ten times stronger than the photometric signal; this is the configuration that produces our main positive result.
- $\lambda = 1$ (v16) balances anchor and photometric; this is the configuration combined with VGGT poses that produces the largest δ<1.25³ improvement.
- $\lambda \to \infty$ recovers the zero-shot predictions exactly (no adaptation).

**Implementation.** The zero-shot depths are precomputed offline (`precompute_zeroshot_depths.py`) once per pose resolution: run Depth Pro on every KITTI training frame at $1536 \times 1536$, downsample to pose resolution, save as fp16 to disk. Total precomputation cost: about 1 hour on the RTX 4070 Ti; total disk usage: about 700 MB at 416 × 128. At training time the cached depths are loaded by `KITTIRawDataset` and accessed through `batch["zeroshot_depth"]`. The consistency loss is computed inline in `train_one_epoch()` after the photometric loss and added to the total before backpropagation.

**Why consistency loss prevents the failure mode of pure photometric training.** The pure photometric gradient pushes the depth network towards any depth field that minimises reconstruction error — including degenerate fields where the depth is grossly miscalibrated but the warping is still consistent because the pose network has compensated. The consistency loss closes this degeneracy: the depth must remain near the zero-shot prediction, so the pose network has only one residual signal left to fit (the actual ego-motion), and the depth must remain in a regime where the evaluation metric is informative. Empirically this prevents the 5× AbsRel degradation we observe under pure photometric training.

---

### 3.8 Numerical Stability Recipe

Mixed-precision training of LoRA adapters at the scale of Depth Pro (96 attention blocks, 192 LoRA tensors) exposes two failure modes that, if uncorrected, silently corrupt training without producing observable errors until evaluation. We isolated these failure modes during development and provide a complete fix; both are described here.

#### 3.8.1 LoRA Weight Overflow under FP16

**Symptom.** Train Depth Pro + LoRA with `torch.amp.autocast(dtype=torch.float16)`. Training appears normal: the photometric loss decreases monotonically, the validation loss decreases, checkpoints are saved on improvement. On the held-out test set, however, evaluation produces NaN metrics on every checkpoint. Inspection of the saved state dicts reveals that all 192 LoRA matrices ($A$ and $B$ for each of the 96 attention projections) contain NaN values.

**Diagnosis.** The 16-bit floating-point format used in FP16 autocast has a 5-bit exponent, giving a maximum representable magnitude of $\sim 6.5 \times 10^4$. Gradients flowing backward through the dual ViT-Large encoders (48 transformer blocks) accumulate factors at each layer; even with the Kaiming initialisation of LoRA-A and the zero initialisation of LoRA-B, the gradient magnitudes occasionally exceed this range, producing infinities that turn into NaN under arithmetic. The `torch.nan_to_num` guard placed on the depth output (intended to suppress numerical errors in extreme regions) silently masks the symptom in the forward pass, so the model continues to "train" with corrupted parameters, photometric loss continues to decrease (now on garbage outputs), and the symptom does not surface until evaluation.

**Fix — three-part recipe.**

1. **Switch autocast to bfloat16.** The bf16 format has an 8-bit exponent (same range as FP32) but a 7-bit mantissa (lower precision). On Ampere and later NVIDIA GPUs (including RTX 4070 Ti), bf16 is hardware-accelerated and produces no overflow on the LoRA backward pass. This is the single change that eliminates the failure mode entirely.

2. **Pre-backward NaN guard.** Before `loss.backward()` we check `torch.isfinite(loss)` and skip the optimiser step if the loss is not finite. This catches the rare residual case where bf16 alone is insufficient.

3. **Post-unscale gradient sanitisation.** After `scaler.unscale_(optimizer)` we iterate over all parameters and zero out any non-finite gradient values. This prevents corrupted gradients from propagating into the AdamW moment buffers (which would persist NaN forever once introduced).

4. **Pre-save state dict validation.** Before writing a checkpoint to disk we check every floating-point tensor in `model.state_dict()` for finite values. If any tensor contains NaN or Inf, the save is refused with a warning. This prevents corrupt checkpoints from masquerading as the best model.

Together these four measures eliminated every LoRA-overflow failure we encountered.

#### 3.8.2 Focal-Length Scaling Mismatch Between Training and Evaluation

**Symptom.** A fine-tuned model that produces visibly sensible depth maps during training validation produces AbsRel of approximately $1.9$ during full test-set evaluation (versus zero-shot 0.087). The depth maps look correct qualitatively but the metric scale is wildly off.

**Diagnosis.** Depth Pro outputs canonical inverse depth $\hat{d}_{\text{canon}}$ that must be multiplied by $W / f_{px}$ to obtain metric depth. The catch: at training time we compute this with $W = W_p = 416$ and $f_{px}$ from the KITTI calibration scaled to pose resolution. At evaluation time the original `evaluate_kitti.py` used $W = W_{\text{orig}} = 1242$ and $f_{px}$ predicted by Depth Pro's FOV head. The two paths produce different metric scales, and median scaling at evaluation cannot correct for the structural inconsistency.

**Fix.** Replace the FOV head's predicted focal length with the KITTI ground-truth focal length from the calibration file at evaluation time. The relevant evaluation code path becomes:
```python
f_px = P2[0, 0]  # KITTI calibration K-matrix entry, NOT model FOV head
inv_depth = canonical_inv_depth * (orig_w / f_px)
```
This change brings training and evaluation back into geometric agreement and resolves the 1.88-AbsRel failure mode.

A natural follow-up question is why the FOV head's predictions are wrong on KITTI. The FOV head is frozen throughout our fine-tuning, so its predictions are exactly the zero-shot Depth Pro FOV predictions. We suspect that the model was trained predominantly on hand-held and indoor imagery and produces unreliable FOV predictions on KITTI's driving distribution (wide-angle automotive camera, dashcam mount). Re-training the FOV head with KITTI calibration as supervision is outside the scope of this work.

---

### 3.9 Training Pipeline — End-to-End Summary

Pulling all the components together, the per-step training procedure (within `train_one_epoch` in `train_kitti_selfsup_ms.py`) is:

```
for (target_depth_input, target, source_prev, source_next, K, inv_K, T_prev_cached, T_next_cached, depth_zs) in train_loader:

    # 1. Depth Pro forward pass at 1536×1536 (bf16 autocast)
    encodings = model.encoder(target_depth_input)
    features, _ = model.decoder(encodings)
    canonical_inv_depth = model.head(features)

    # 2. Interpolate to pose resolution and convert to metric depth
    inv_depth = F.interpolate(canonical_inv_depth, size=(H_p, W_p))
    inv_depth *= W_p / f_px_at_pose_resolution
    depth = 1 / clamp(inv_depth, 1e-4, 1e4)

    # 3. Pose estimation: trained PoseNet or cached VGGT
    if T_prev_cached is in batch:                # v16, v19
        T_prev, T_next = T_prev_cached, T_next_cached
    else:                                        # v15, v17, v18
        T_prev = pose_vec_to_matrix(pose_net(target, source_prev))
        T_next = pose_vec_to_matrix(pose_net(target, source_next))

    # 4. Differentiable warping
    warped_prev = warper(source_prev, depth, T_prev, K, inv_K)
    warped_next = warper(source_next, depth, T_next, K, inv_K)

    # 5. Photometric loss with auto-masking
    L_photo = monodepth2_loss(target, [source_prev, source_next], [warped_prev, warped_next], inv_depth)

    # 6. Consistency loss against zero-shot
    if depth_zs is in batch:
        L_cons = consistency_loss(depth, depth_zs, mode=args.consistency_mode, weight=args.depth_weight_power)
        L_total = L_photo + lambda * L_cons
    else:
        L_total = L_photo

    # 7. NaN-safe backward and step
    if not torch.isfinite(L_total): continue
    scaler.scale(L_total).backward()
    if step % grad_accum_steps == 0:
        scaler.unscale_(optimizer)
        sanitize_lora_gradients(model)
        clip_grad_norm_(all_trainable, max_norm=1.0)
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
```

Validation runs every two epochs and includes both the photometric reconstruction loss on the validation split and a 50-image GT depth evaluation on a deterministic test subset (the latter producing the `abs_rel`, `delta1`, etc. WandB curves). The best checkpoint is selected by the validation photometric loss, with the NaN check preventing corrupt checkpoints from being persisted.

Total training time for the v15 configuration (LoRA rank 8, 5 epochs, KITTI training split with stride 6, RTX 4070 Ti): approximately 12.5 hours including validation. The v16 configuration (with VGGT poses) is identical except for the offline VGGT precomputation step (one-time, 90 minutes). The next chapter reports the evaluation results.
