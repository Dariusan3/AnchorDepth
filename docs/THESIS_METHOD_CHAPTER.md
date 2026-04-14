# Thesis — Method Chapter
## Chapter 3: Method

---

### 3.1 Overview

This chapter presents our method for self-supervised monocular depth estimation using Depth Pro as a pretrained backbone. The core idea is to adapt a large pretrained metric depth model to learn from unlabeled monocular video sequences — without any ground truth depth annotations — using photometric consistency as the training signal.

We introduce four modules on top of the pretrained Depth Pro model:

1. **LoRA adapters** injected into the frozen ViT encoders — enabling parameter-efficient adaptation of the depth backbone
2. **PoseNet** — a new network that estimates camera ego-motion between consecutive frames
3. **Differentiable Warper** — a geometric module that reconstructs one frame from another using predicted depth and pose
4. **Self-supervised loss functions** — photometric reprojection loss and edge-aware smoothness regularization

Figure 3.1 shows how these modules interact during training.

```
                        ┌─────────────────────────────────────────┐
  Frame triplet         │          INTRODUCED MODULES              │
  (t-1, t, t+1)         │                                          │
        │               │  ┌──────────────────────────────────┐   │
        │               │  │        Depth Pro backbone        │   │
        ├──── I_t ──────┼─▶│  ViT Encoder                    │   │
        │    (1536²)    │  │  + LoRA adapters  ← (Module 1)  │   │
        │               │  │  Decoder + Head                  │   │
        │               │  └───────────┬──────────────────────┘   │
        │               │              │ canonical inv_depth        │
        │               │              ▼                            │
        │               │         depth scaling                     │
        │               │              │ depth D_t                  │
        │               │              ▼                            │
        ├── (I_t, I_s) ─┼─▶ PoseNet ──┐  ← (Module 2)             │
        │    (640×192)  │             │ T_{t→s}                    │
        │               │             ▼                            │
        │               │      Differentiable  ← (Module 3)        │
        │               │         Warper                           │
        │               │             │ Î_{s→t}                    │
        │               │             ▼                            │
        └───── I_t ─────┼─▶  Self-supervised  ← (Module 4)        │
               (640×192)│       Loss L                             │
                        └─────────────────────────────────────────┘
```

---

### 3.2 Background: Depth Pro

Before describing our contributions, we briefly review the Depth Pro architecture (Bochkovskii et al., 2025) that forms our backbone.

Depth Pro is a supervised metric depth estimation model trained on diverse internet imagery. It outputs absolute (metric) depth predictions for arbitrary cameras without requiring any camera intrinsic parameters as input.

#### 3.2.1 Multi-Resolution Encoder

The encoder uses two DINOv2-Large Vision Transformers (Oquab et al., 2024) — a *patch encoder* and an *image encoder* — each with 304M parameters. Input images must be at exactly 1536×1536 pixels. A three-level image pyramid is constructed:

| Pyramid level | Resolution | Patches |
|--------------|-----------|---------|
| High (1×) | 1536×1536 | 5×5 = 25 |
| Mid (0.5×) | 768×768 | 3×3 = 9 |
| Low (0.25×) | 384×384 | 1×1 = 1 |

All 35 patches (384×384 each) are processed in a single ViT forward pass, capturing both fine-grained texture (high-res) and global structure (low-res).

#### 3.2.2 Decoder and Canonical Inverse Depth

A `MultiresConvDecoder` (19.67M parameters) fuses the multi-resolution encoder features. A 4-layer CNN depth head (0.40M parameters) produces a **canonical inverse depth** map — a camera-agnostic representation independent of focal length:

```
d_canonical = f(I ; θ_encoder, θ_decoder, θ_head)
```

A separate FOV head predicts the camera's field of view `fov_deg`, which is used to convert canonical inverse depth to metric depth:

```
f_px = 0.5 · W / tan(0.5 · fov_deg · π/180)
d_metric = d_canonical · (W / f_px)
```

This canonical representation is what enables Depth Pro's zero-shot generalization, but it requires a careful adaptation for self-supervised warping (Section 3.4).

---

### 3.3 Module 1 — LoRA Adapters for the Depth Backbone

The ViT encoders of Depth Pro contain 608M parameters — far too many to fine-tune on a 12GB consumer GPU. We introduce **Low-Rank Adaptation (LoRA)** (Hu et al., 2022) adapters into the frozen encoder, allowing the depth backbone to be adapted with a fraction of the parameters and memory.

#### 3.3.1 LoRA Formulation

For each frozen attention weight matrix `W ∈ R^{d×k}`, LoRA introduces a parallel trainable branch:

```
h = Wx + ΔWx = Wx + BAx
```

where `B ∈ R^{d×r}` and `A ∈ R^{r×k}` are learnable matrices with rank `r ≪ min(d, k)`. During training, `W` is frozen and only `A`, `B` are updated. The rank `r` controls the capacity of the adaptation.

At inference, the LoRA branch can be merged into `W` with zero overhead:
```
W' = W + BA
```

#### 3.3.2 Application to Depth Pro

We inject LoRA into all attention Query, Key, Value, and output projection matrices in both ViT encoders:

| Component | Attention layers | LoRA matrices | Parameters added |
|-----------|-----------------|---------------|-----------------|
| Patch encoder | 48 layers × 4 projections | 192 | 1.18M |
| Image encoder | 48 layers × 4 projections | 192 | 1.18M |
| **Total** | **96 layers** | **384** | **2.36M** |

With rank `r = 8` and scaling factor `α = 8.0`, LoRA reduces encoder adaptation from 608M to 2.36M parameters — a **257× reduction**. The decoder (19.67M) and depth head (0.40M) remain fully trainable. The FOV head (304M) is fully frozen.

**Why LoRA and not full fine-tuning or adapter layers?**
- Full fine-tuning: requires >40GB VRAM for the encoder alone
- Adapter layers (bottleneck MLPs): modify the residual stream and risk disrupting pretrained representations
- LoRA: acts in parallel with frozen weights, preserving the pretrained feature space while adding low-rank updates

---

### 3.4 Module 2 — PoseNet for Ego-Motion Estimation

Self-supervised depth training requires knowing the camera's relative motion between frames to warp one view into another. We introduce **PoseNet** — a new lightweight network trained from scratch alongside the depth backbone.

#### 3.4.1 Architecture

PoseNet is based on ResNet-18 (He et al., 2016), modified for pose estimation:

```
Input:  [I_t ; I_s] ∈ R^{B × 6 × H × W}   (target + source concatenated)
        ↓
ResNet-18 (first conv: 3→6 channels, rest unchanged)
        ↓
Global average pooling
        ↓
Pose head: Conv(512→256) → ReLU → Conv(256→6)
        ↓
Output: [ω, t] ∈ R^{B × 6}   (axis-angle + translation)
```

The two RGB frames are concatenated along the channel dimension before entering the network, allowing it to reason about the correspondence between the two views jointly.

The output 6-vector is scaled by 0.01 to prevent large initial pose estimates that would produce poor initial frame reconstructions and destabilize early training.

| Property | Value |
|----------|-------|
| Backbone | ResNet-18 |
| Input channels | 6 (target + source RGB) |
| Input resolution | 640×192 |
| Output | 6-DoF: [ω₁, ω₂, ω₃, t₁, t₂, t₃] |
| Output scale | ×0.01 |
| Parameters | 11.91M (all trainable) |

PoseNet is run twice per training step to obtain poses for both source frames:
```
T_{t→t-1} = PoseNet(I_t, I_{t-1})
T_{t→t+1} = PoseNet(I_t, I_{t+1})
```

#### 3.4.2 Pose Representation

The 3D rotation is parameterized as an axis-angle vector `ω ∈ R^3`, where the direction gives the rotation axis and the magnitude gives the angle in radians. This is converted to a 3×3 rotation matrix `R` using Rodrigues' formula:

```
θ = ||ω||
R = I + sin(θ)/θ · K + (1 - cos(θ))/θ² · K²
```

where `K` is the skew-symmetric matrix of the unit axis `ω/θ`. The full 4×4 transformation matrix is:

```
T = | R  t |
    | 0  1 |
```

---

### 3.5 Module 3 — Differentiable Warper

The Differentiable Warper reconstructs the target frame by projecting source frame pixels through the predicted depth and pose. It is the geometric core of our self-supervised training loop — connecting the depth and pose predictions to the photometric loss.

#### 3.5.1 Camera Model

We use the standard pinhole camera model. For a pixel `p = (u, v)` in the target frame and its predicted depth `D_t(p)`, the 3D point in camera coordinates is:

```
P = D_t(p) · K^{-1} · [u, v, 1]ᵀ
```

where `K` is the 3×3 intrinsic matrix:

```
K = | f_x   0   c_x |
    |  0   f_y  c_y |
    |  0    0    1  |
```

#### 3.5.2 Canonical Depth Scaling (Critical Adaptation)

A key challenge in applying Depth Pro to self-supervised training is that its output is **canonical inverse depth** — not metric depth. The warper requires metric depth consistent with the camera's actual projection geometry.

We scale the canonical inverse depth using the FOV head's prediction:

```python
fov_clamped = clamp(fov_deg, 5°, 175°)          # numerical stability
f_px = 0.5 · W / tan(0.5 · fov_clamped · π/180) # focal length in pixels
scale = clamp(W / f_px, 0.01, 100.0)             # prevent extreme scaling
inv_depth = canonical_inv_depth · scale
```

Without this scaling, the warped image does not correspond to the camera's actual geometry, making the photometric loss an incorrect training signal.

#### 3.5.3 Warping Pipeline

Given metric depth `D_t`, pose `T_{t→s}`, and intrinsics `K`, the warped image is computed in four steps:

**Step 1 — Backproject target pixels to 3D:**
```
P = D_t · K^{-1} · p_t                (shape: B × 3 × H·W)
```

**Step 2 — Transform to source camera frame:**
```
P_s = R · P + t                        (rotation + translation)
```

**Step 3 — Project to source image coordinates:**
```
p_s = K · P_s / P_s[z]                (perspective division)
p_s_norm = 2 · p_s / [W-1, H-1] - 1  (normalize to [-1, 1] for grid_sample)
```

**Step 4 — Bilinear sampling:**
```
Î_{s→t} = bilinear_sample(I_s, p_s_norm)
```

All four steps are differentiable, so gradients flow back through `Î_{s→t}` into both the depth prediction `D_t` and the pose `T_{t→s}`.

---

### 3.6 Module 4 — Self-Supervised Loss Functions

We train the full system using two loss terms: a photometric reprojection loss and an edge-aware smoothness regularizer. No ground truth depth is used at any point.

#### 3.6.1 Photometric Reprojection Loss

The photometric loss measures the visual dissimilarity between the reconstructed target frame `Î_{s→t}` and the actual target frame `I_t`. Following Monodepth2 (Godard et al., 2019), we use a weighted combination of SSIM and L1:

```
pe(I_a, I_b) = α · (1 - SSIM(I_a, I_b)) / 2  +  (1 - α) · |I_a - I_b|₁
```

with `α = 0.85`. SSIM is computed with a 3×3 average pooling window.

The SSIM term captures structural similarity and is more robust to photometric distortions (lighting changes, reflections) than pure L1. The L1 term ensures pixel-level accuracy.

#### 3.6.2 Per-Pixel Minimum Reprojection

Since we have two source frames (`t-1` and `t+1`), some target pixels may be occluded in one source but visible in the other. Taking the minimum photometric error per pixel across both sources naturally handles occlusions — occluded pixels have high error from one source and low error from the other:

```
L_photo = mean( min( pe(I_t, Î_{t-1→t}),  pe(I_t, Î_{t+1→t}) ) )
```

#### 3.6.3 Edge-Aware Smoothness Loss

Depth predictions can be noisy in textureless regions where the photometric loss provides no gradient signal (e.g., sky, roads). We add a smoothness regularizer that penalizes depth gradients, weighted by image gradients to allow depth discontinuities at object boundaries:

```
L_smooth = mean( |∂d̂/∂x| · exp(-|∂I/∂x|)  +  |∂d̂/∂y| · exp(-|∂I/∂y|) )
```

where `d̂ = inv_depth / mean(inv_depth)` is the mean-normalized inverse depth. Mean normalization prevents the trivial solution of predicting near-zero disparity everywhere to minimize smoothness. The exponential edge weights suppress the smoothness penalty at object boundaries (large `|∂I|`), preserving sharp depth edges.

This is computed in FP32 to avoid NaN from FP16 overflow in the exponential.

#### 3.6.4 Total Loss

```
L = L_photo + λ_s · L_smooth,    λ_s = 1e-3
```

The smoothness weight `λ_s = 1e-3` follows the standard setting from Monodepth2.

---

### 3.7 Training Pipeline

#### 3.7.1 Data Loading

We train on the KITTI Raw dataset (Geiger et al., 2013) using the Eigen/Zhou split. For each target frame `I_t`, we load the temporally adjacent source frames `I_{t-1}` and `I_{t+1}` from the same driving sequence, along with the camera intrinsic matrix `K`.

Frames are loaded at two resolutions:
- **1536×1536** for the depth network (required by Depth Pro's encoder)
- **640×192** for PoseNet and the loss computation (standard resolution for self-supervised depth)

**Temporal stride:** We use every 3rd frame as the target (stride=3), reducing training samples from 39,810 to 13,270. Consecutive KITTI frames at 10Hz have only ~0.83m baseline at highway speeds, making them nearly identical. Stride=3 provides meaningful ~0.3s temporal baselines.

**Augmentation:** Color jitter (brightness, contrast, saturation, hue) and random horizontal flip are applied consistently across all frames in a triplet to preserve photometric consistency.

#### 3.7.2 Full Training Step

For each batch:

1. Load triplet `(I_{t-1}, I_t, I_{t+1})` at depth and pose resolutions
2. Forward pass through Depth Pro + LoRA → `d_canonical`, `fov_deg`
3. Scale canonical depth → `inv_depth`, compute `depth = 1 / inv_depth`
4. Forward pass through PoseNet twice → `T_{t→t-1}`, `T_{t→t+1}`
5. Warp `I_{t-1}` and `I_{t+1}` using Warper → `Î_{t-1→t}`, `Î_{t+1→t}`
6. Compute `L_photo` (min reprojection) and `L_smooth`
7. Backward pass through all modules jointly
8. Update LoRA, decoder, head, and PoseNet parameters

#### 3.7.3 Optimization

We use three parameter groups with discriminative learning rates:

| Group | Parameters | Learning rate |
|-------|-----------|---------------|
| LoRA adapters | 2.36M | 1e-5 |
| Decoder + Head | 20.07M | 1e-4 |
| PoseNet | 11.91M | 1e-4 |

Lower learning rate for LoRA prevents catastrophic forgetting of the pretrained DINOv2 representations. The decoder and PoseNet use a higher rate since they are either partially pretrained (decoder) or trained from scratch (PoseNet).

We use AdamW with weight decay 1e-4, gradient clipping (max norm=1.0), FP16 mixed precision with AMP GradScaler, and a CosineAnnealing scheduler with 3-epoch linear warmup. Gradient accumulation over 4 steps gives an effective batch size of 4.

Gradient checkpointing is applied to both ViT encoders to recompute intermediate activations during backpropagation, reducing peak VRAM from ~18GB to ~10.8GB.

---

### 3.8 Evaluation

We evaluate on the KITTI Eigen test split (697 frames) using velodyne LiDAR ground truth projected to the camera plane.

Following the standard self-supervised evaluation protocol:

- **Garg/Eigen crop:** Remove sky and car hood: rows `[0.40·H : 0.99·H]`, cols `[0.036·W : 0.964·W]`
- **Depth cap:** 1–80m
- **Median scaling:** `scale = median(gt) / median(pred)` per image — necessary because self-supervised methods predict relative depth (up to an unknown scale)

Reported metrics: AbsRel, SqRel, RMSE, RMSE_log, δ<1.25, δ<1.25², δ<1.25³.
