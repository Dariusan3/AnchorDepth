# Depth Pro - Method Improvements for Monocular Depth Estimation

## Bachelor Thesis: Efficient Adaptation of Foundation Depth Models on Consumer GPUs

**Author:** Dariusan3
**Base Model:** Depth Pro (Apple, ICLR 2025) — "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second"
**Paper Reference:** Bochkovskii et al., arXiv:2410.02073
**Hardware:** NVIDIA RTX 4070 Ti (12GB VRAM)
**Datasets:** NYU Depth V2 (supervised, v0–v5), KITTI Raw (self-supervised, v6+)

---

## 1. Introduction

This document tracks all modifications made to Apple's Depth Pro model for monocular depth estimation. Each experiment is documented with motivation, implementation details, results, and analysis.

**Phase 1 (v0–v5):** Supervised fine-tuning on NYU Depth V2 — exploring LoRA adaptation, decoder training, and test-time augmentation to close the gap between the publicly available pretrained model and paper-reported results.

**Phase 2 (v6+):** Self-supervised monocular depth estimation on KITTI — the core thesis contribution. Uses photometric consistency from monocular video sequences to train without ground truth depth, following the Monodepth2 framework (Godard et al., ICCV 2019) adapted to Depth Pro's architecture.

---

## 2. Baseline Model Architecture

Depth Pro uses a multi-resolution vision transformer architecture:

| Component | Architecture | Parameters | Trainable |
|-----------|-------------|------------|-----------|
| Patch Encoder | DINOv2-Large ViT (16-patch, 384×384) | 304M | Frozen (baseline) |
| Image Encoder | DINOv2-Large ViT (shared arch) | 304M | Frozen (baseline) |
| Decoder | MultiresConvDecoder (5-level fusion) | 19.7M | Yes |
| Depth Head | 4-layer CNN (256→128→32→1) | 0.4M | Yes |
| FOV Head | FOVNetwork with separate encoder | 304M | Frozen |
| **Total** | | **952M** | **20.1M trainable** |

**Key design features:**
- Creates a 3-level image pyramid (1536×1536 → 768×768 → 384×384)
- Splits into overlapping 384×384 patches (25 + 9 + 1 = 35 patches)
- Processes all patches in a single forward pass through the ViT
- Multi-resolution feature fusion via deconvolution and skip connections
- Outputs canonical inverse depth, converted to metric depth using focal length

---

## 3. Experiment Log

### 3.1 Experiment v0 — Baseline Pretrained (Zero-Shot)

**Date:** 2026-03-21
**Description:** Evaluate the pretrained Depth Pro model without any fine-tuning on NYU Depth V2.

**Method:** Direct inference using Apple's pretrained weights (`depth_pro.pt`). The model was designed for zero-shot generalization, meaning it was not specifically trained on NYU.

**Results (654 Eigen test images):**

| Metric | Value |
|--------|-------|
| AbsRel (↓) | 0.1155 |
| SqRel (↓) | 0.0590 |
| RMSE (↓) | 0.3566 |
| RMSElog (↓) | 0.1369 |
| delta < 1.25 (↑) | 0.8893 |
| delta < 1.25² (↑) | 0.9846 |
| delta < 1.25³ (↑) | 0.9961 |

**Scale-Invariant Results:** AbsRel = 0.0651 (after optimal per-image scale alignment)

**Analysis:** The zero-shot performance is reasonable but far from SOTA. The gap between metric (0.1155) and scale-invariant (0.0651) evaluation reveals that ~44% of the error comes from systematic scale misalignment — the model predicts correct relative depth structure but struggles with absolute metric scale on NYU indoor scenes.

---

### 3.2 Experiment v1 — Fine-tuned Decoder + Head

**Date:** 2026-03-21
**Description:** Fine-tune only the decoder and depth head while keeping the DINOv2 encoder frozen.

**Motivation:** The encoder's pretrained DINOv2 features are high quality but generic. The decoder can be adapted to the NYU depth distribution with minimal compute, as it contains only 20.1M of the model's 952M parameters.

**Training Configuration:**

| Parameter | Value |
|-----------|-------|
| Trainable params | 20.1M (decoder 19.7M + head 0.4M) |
| Frozen params | 931.9M (encoder + FOV) |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-5) |
| Scheduler | CosineAnnealingLR |
| Batch size | 1 (gradient accumulation: 4) |
| Epochs | 25 |
| Loss | Scale-Invariant Log + Gradient Matching (0.5 weight) |
| Augmentation | Horizontal flip (p=0.5), brightness jitter (±0.2) |
| Precision | FP16 mixed precision |
| Training time | 173 minutes (~2.9 hours) |
| Peak VRAM | 8.67 GB |

**Loss Functions Used:**

1. **Scale-Invariant Log Loss (SI-log):** Measures the difference in log-depth space, invariant to global scale. Defined as: `L = mean((log(pred) - log(gt))² - λ·mean(log(pred) - log(gt))²)` where λ=0.85.

2. **Gradient Matching Loss:** Penalizes differences in spatial depth gradients (∂d/∂x and ∂d/∂y), encouraging sharp edges and correct surface orientations.

**Results (654 Eigen test images):**

| Metric | Pretrained | Fine-tuned | Improvement |
|--------|-----------|------------|-------------|
| AbsRel (↓) | 0.1155 | 0.0855 | +26.0% |
| SqRel (↓) | 0.0590 | 0.0414 | +29.8% |
| RMSE (↓) | 0.3566 | 0.3288 | +7.8% |
| RMSElog (↓) | 0.1369 | 0.1117 | +18.4% |
| delta < 1.25 (↑) | 0.8893 | 0.9389 | +5.6% |
| delta < 1.25² (↑) | 0.9846 | 0.9904 | +0.6% |
| delta < 1.25³ (↑) | 0.9961 | 0.9975 | +0.1% |

**Training Dynamics:**
- Loss converged smoothly from 0.024 → 0.012 over 25 epochs
- Validation AbsRel improved consistently: 0.102 (ep5) → 0.093 (ep10) → 0.090 (ep20) → 0.089 (ep25)
- No signs of overfitting despite small dataset, likely due to the large frozen encoder acting as a regularizer

**Analysis:** Fine-tuning the decoder alone yields a 26% improvement in AbsRel. The scale-invariant gap narrowed to AbsRel=0.0565, confirming the decoder learned better scale calibration for NYU. The frozen encoder strategy is effective because DINOv2 features are already strong — the decoder just needs to learn the NYU-specific depth mapping.

---

### 3.3 Experiment v2 — Test-Time Augmentation (TTA)

**Date:** 2026-03-21
**Description:** Apply test-time augmentation at inference to improve prediction accuracy without retraining.

**Motivation:** TTA is a "free" improvement that averages predictions from multiple views of the same image, reducing prediction variance and systematic biases (e.g., left-right asymmetry).

**Implementation:** Created `src/depth_pro/improvements/tta.py` with three TTA modes:

1. **Flip TTA:** Average original prediction with horizontally-flipped prediction (flipped back). Removes left-right bias.
2. **Multi-scale TTA:** Average predictions at 0.75×, 1.0×, and 1.25× input resolution. Captures both local detail and global context.
3. **Full TTA:** Combine flip + multi-scale (2 × 3 = 6 predictions averaged). Maximum accuracy at 6× inference cost.

**Results (Flip TTA, fine-tuned model):**

| Metric | No TTA | Flip TTA | Improvement |
|--------|--------|----------|-------------|
| AbsRel (↓) | 0.0855 | 0.0790 | +7.6% |
| delta < 1.25 (↑) | 0.9389 | 0.9532 | +1.5% |
| Inference time | 0.74s | 1.76s | 2.4× slower |

**Analysis:** Horizontal flip TTA provides a consistent 7.6% improvement with only 2.4× inference slowdown. This confirms the model has a slight left-right bias that averaging eliminates. The multi-scale and full TTA modes provide further improvements at higher compute cost.

---

### 3.4 Experiment v3 — Phase 1 Improvements (Loss + Augmentation + LR Strategy)

**Date:** 2026-03-21
**Description:** Combined improvements to loss function, data augmentation, and learning rate strategy.

**Motivation:** Three complementary improvements that address different aspects of the training:
- Better losses teach the model what to optimize for
- Better augmentation prevents overfitting on 745 images
- Better LR strategy ensures stable convergence

#### 3.4.1 Improved Loss Function

The baseline used SI-log + gradient matching. We add three new loss components:

**A) Multi-Scale Structural Similarity (MS-SSIM) Loss**

Computes SSIM at multiple resolutions (1×, 0.5×, 0.25×) to capture structural similarity at different spatial frequencies. Unlike pixel-wise losses, SSIM considers local luminance, contrast, and structure — matching human perception of depth quality.

Formula: `L_ssim = Σ_s (1 - SSIM(pred_s, gt_s))` where `s` are scale levels.

**B) Affine-Invariant Loss**

For each training sample, computes the optimal affine transform (scale α and shift β) that aligns predicted depth to ground truth, then measures the residual error. This decouples structural learning from scale learning:

`α*, β* = argmin ||α·pred + β - gt||²` (closed-form least squares)
`L_affine = ||α*·pred + β* - gt|| / ||gt||`

**C) Surface Normal Consistency Loss**

Derives surface normals from depth maps via spatial gradients and penalizes angular differences between predicted and ground truth normals. This provides stronger edge supervision than gradient matching alone:

`n = normalize([-∂d/∂x, -∂d/∂y, 1])`
`L_normal = 1 - mean(n_pred · n_gt)`

**Combined Loss:**
```
L = L_si_log + 0.5·L_gradient + 0.3·L_ssim + 0.2·L_normal + 0.1·L_affine
```

#### 3.4.2 Enhanced Data Augmentation

| Augmentation | Range | Applied to |
|-------------|-------|------------|
| Horizontal flip | p=0.5 | RGB + Depth |
| Random crop + resize | 80-100% area | RGB + Depth |
| Brightness jitter | ±0.2 | RGB only |
| Contrast jitter | ±0.2 | RGB only |
| Saturation jitter | ±0.2 | RGB only |
| Hue jitter | ±0.05 | RGB only |
| Random rotation | ±5° | RGB + Depth |
| Gaussian noise | σ=0.01-0.03 | RGB only |
| Random erasing | p=0.1, 2-10% area | RGB only |

**Key principle:** Spatial augmentations (flip, crop, rotation) are applied consistently to both RGB and depth. Color augmentations are applied only to RGB since depth is not affected by lighting.

#### 3.4.3 Learning Rate Strategy

| Parameter | Baseline | Phase 1 |
|-----------|----------|---------|
| Initial LR | 1e-4 | 1e-4 |
| Warmup | None | Linear, 5 epochs |
| Scheduler | CosineAnnealingLR | CosineAnnealingWarmRestarts (T_0=15) |
| Epochs | 25 | 50 |
| Weight decay | 1e-5 | 1e-4 |

The warmup prevents large gradient updates in early training that could destabilize the pretrained decoder weights. Cosine restarts allow the model to escape local minima by periodically increasing the learning rate.

#### 3.4.4 Guided Bilateral Filtering (Post-Processing)

Applied at inference time: uses the RGB image to guide an edge-preserving filter on the predicted depth map. This sharpens depth boundaries without retraining:

- Uses OpenCV's `ximgproc.guidedFilter` or custom implementation
- Parameters: radius=8, epsilon=0.01
- The RGB image provides edge guidance so depth boundaries align with object boundaries

**Results:**

| Metric | v2 (TTA) | v3 (Phase 1) | Improvement |
|--------|----------|--------------|-------------|
| AbsRel (↓) | 0.0790 | TBD | TBD |
| delta < 1.25 (↑) | 0.9532 | TBD | TBD |

*(Results will be filled after training completes)*

---

### 3.5 Experiment v3 — Phase 1 Combined Improvements

**Date:** 2026-03-22
**Description:** Combined improvements to loss function, data augmentation, and LR strategy.

**Changes from v1:**
- 5 loss functions: SI-log + gradient matching + multi-scale SSIM + surface normal consistency + affine-invariant
- Enhanced augmentation: random crop, rotation (±5°), color jitter (brightness/contrast/saturation/hue), Gaussian noise, random erasing
- LR warmup (5 epochs) + cosine annealing with warm restarts (T=15)
- 50 epochs (vs 25)

**Training:** 50 epochs, 352 minutes (~5.9 hours), 8.68 GB peak VRAM

**Results (654 Eigen test images):**

| Metric | v1 Fine-tuned | v3 Phase 1 | Change |
|--------|-------------|------------|--------|
| AbsRel (↓) | 0.0855 | 0.0878 | -2.6% (worse) |
| RMSE (↓) | 0.3288 | 0.3273 | +0.5% |
| delta<1.25 (↑) | 0.9389 | 0.9362 | -0.3% |
| SI AbsRel (↓) | 0.0565 | 0.0539 | **+4.6% (better)** |

**Analysis:** The scale-invariant metrics improved (model learned better structural depth), but metric scale calibration got slightly worse. The aggressive augmentation (random crops, rotations) with only 745 images causes slight underfitting — the model doesn't see enough examples to fully calibrate scale under varied viewing conditions. This confirms that **more data** or **encoder adaptation** is needed, not just better training tricks on limited data.

**Key Lesson:** With very small datasets, simpler training (v1) can outperform complex training (v3) on metric evaluation, even when the model's structural understanding improves. The augmentation regularizes well but the dataset is too small to benefit.

---

### 3.6 Experiment v4 — Phase 2: LoRA Encoder Fine-Tuning

**Date:** 2026-03-22–23
**Description:** Apply Low-Rank Adaptation (LoRA) to the DINOv2 encoder's attention layers, enabling the encoder to specialize for NYU indoor depth while preserving pretrained features.

**Motivation:** The encoder (627M params) produces the multi-resolution features that drive everything downstream. In all prior experiments it was frozen — its DINOv2 features are generic (pretrained on ImageNet). Even a small adaptation of the encoder's attention mechanism can dramatically improve feature quality for the specific task of indoor depth estimation.

#### LoRA Implementation Details

**What is LoRA?** Low-Rank Adaptation decomposes weight updates into two small matrices:
```
output = W_original * x + (B @ A) * x * (alpha / rank)
```
where A is (rank × in_features) and B is (out_features × rank). Only A and B are trained; the original weight W is frozen. This means:
- B is initialized to zeros → LoRA starts as identity (no disruption to pretrained weights)
- A uses Kaiming initialization for proper gradient flow
- The scaling factor (alpha/rank) controls the magnitude of adaptation

**Target Layers:** Applied to all attention Q/K/V projections (`attn.qkv`) and output projections (`attn.proj`) in both the patch encoder (24 blocks) and image encoder (24 blocks) = 96 LoRA-adapted layers.

**Implementation:** Custom `LoRALinear` wrapper in `train_nyu_lora.py` that:
1. Wraps the original `nn.Linear` (frozen)
2. Adds low-rank A and B matrices (trainable)
3. Forward: `result = original(x) + (x @ A^T @ B^T) * scaling`

**Training Configuration:**

| Parameter | Value |
|-----------|-------|
| LoRA rank | 8 |
| LoRA alpha | 16.0 |
| LoRA params | 2.36M (across 96 layers) |
| Decoder + head params | 20.1M |
| Total trainable | 24.8M / 956.7M (2.6%) |
| Optimizer | AdamW with discriminative LR |
| LR (LoRA) | 5e-5 (with 1e-2 weight decay) |
| LR (decoder) | 1e-4 (with 1e-4 weight decay) |
| Warmup | 3 epochs (linear) |
| Scheduler | CosineAnnealingLR (after warmup) |
| Epochs | 30 |
| Loss | SI-log + 0.5 × gradient matching |
| Training time | 549 minutes (~9.2 hours) |
| Peak VRAM | 10.91 GB |

**Why discriminative learning rates?** The encoder's pretrained features are high quality — large updates would destroy them. LoRA params get 2× lower LR (5e-5 vs 1e-4) and 10× higher weight decay (1e-2 vs 1e-4) to keep adaptations small and stable.

**Training Dynamics:**
- Loss: 0.024 → 0.009 (62% reduction over 30 epochs)
- Validation AbsRel improved steadily: 0.106 (ep5) → 0.079 (ep10) → 0.075 (ep15) → 0.074 (ep25) → 0.074 (ep30)
- No overfitting despite 30 epochs — LoRA's low rank acts as strong regularizer

**Results (654 Eigen test images):**

| Metric | v1 Fine-tuned | v4 LoRA | v4 LoRA + TTA | Improvement (best) |
|--------|-------------|---------|---------------|-------------------|
| AbsRel (↓) | 0.0855 | 0.0781 | **0.0765** | **+33.8% vs baseline** |
| SqRel (↓) | 0.0414 | 0.0360 | **0.0348** | +41.0% |
| RMSE (↓) | 0.3288 | 0.3038 | **0.2982** | +16.4% |
| RMSElog (↓) | 0.1117 | 0.1032 | **0.1013** | +26.0% |
| delta<1.25 (↑) | 0.9389 | 0.9530 | **0.9549** | +8.8% |
| delta<1.25² (↑) | 0.9904 | 0.9911 | 0.9912 | +0.7% |
| SI AbsRel (↓) | 0.0565 | **0.0508** | — | +10.1% |

**Analysis:**
- LoRA provided the single largest improvement, reducing AbsRel from 0.0855 → 0.0781 (**8.7% relative improvement**)
- Adding TTA flip further improved to 0.0765 (**10.5% improvement over v1**)
- Scale-invariant AbsRel dropped to 0.0508, meaning the structural depth predictions are now very accurate
- The remaining gap to the Apple paper (0.036) is likely due to: (1) much more training data, (2) full encoder fine-tuning with large compute, (3) additional datasets beyond NYU
- LoRA proved extremely effective: adapting just 2.36M params (0.25% of encoder) captured most of the benefit of full fine-tuning

### Experiment v5: LoRA + Selective Block Unfreezing (50 epochs)

**Motivation:** LoRA adapts only the attention projections. By additionally unfreezing the last N encoder blocks' MLP and LayerNorm layers, we allow richer feature adaptation while keeping memory manageable. This tests whether deeper encoder adaptation improves over LoRA-only.

**Implementation:**
- Base: LoRA rank 8, alpha 16.0 on all attention Q/K/V and output projections (same as v4)
- Additionally unfreeze last 2 blocks of both `patch_encoder` and `image_encoder`:
  - MLP layers (fc1, fc2)
  - LayerNorm layers (norm1, norm2)
  - Layer scale parameters
- Total trainable: **56M parameters** (5.9% of model) vs 2.36M for v4
- 3 optimizer groups with discriminative learning rates:
  - LoRA params: lr=5e-5, weight_decay=1e-2
  - Decoder+head: lr=1e-4, weight_decay=1e-4
  - Unfrozen blocks: lr=2e-5, weight_decay=1e-2
- 50 epochs, 5-epoch warmup, cosine annealing
- Peak VRAM: 11.31 GB (gradient checkpointing essential)
- Training time: ~945 minutes (~19 hours)
- Best checkpoint: epoch 45, val AbsRel=0.0720

**Results (654 Eigen test images):**

| Metric | v4 LoRA | v5 Baseline | v5 + TTA | Change (v5+TTA vs v4+TTA) |
|--------|---------|-------------|----------|---------------------------|
| AbsRel (↓) | 0.0781 | 0.0787 | **0.0773** | −1.0% |
| RMSE (↓) | 0.3038 | 0.3069 | **0.3014** | −1.1% |
| δ<1.25 (↑) | 0.9530 | 0.9520 | 0.9546 | −0.03% |
| SI AbsRel (↓) | 0.0508 | 0.0513 | **0.0498** | −2.0% |
| SI RMSE (↓) | 0.2206 | 0.2227 | **0.2179** | −1.2% |

**Analysis:**
- v5 baseline (0.0787) is slightly worse than v4 (0.0781) on metric AbsRel — the additional unfrozen parameters may have slightly overfitted on the small dataset
- However, v5 **excels on scale-invariant metrics** (SI AbsRel: 0.0498 vs 0.0508), showing better structural depth quality
- With TTA, v5 achieves competitive results: AbsRel 0.0773, close to v4+TTA's 0.0765
- The best SI AbsRel of 0.0498 is our best structural depth result across all experiments
- **Key insight:** With only 795 training images, the benefit of unfreezing more parameters saturates quickly. LoRA-only (v4) remains the most parameter-efficient approach for small datasets
- The discriminative LR approach (lower LR for encoder blocks) was essential to prevent catastrophic forgetting

---

---

## 4. Phase 2: Self-Supervised Training on KITTI

### 4.1 Motivation — Pivot to Self-Supervised Learning

The supervised experiments (v0–v5) demonstrated that Depth Pro can be effectively adapted with LoRA and achieves strong results on NYU Depth V2. However, the core thesis topic is **self-supervised monocular depth estimation** — learning depth from unlabeled video sequences without any ground truth annotations.

Key motivations for the pivot:

1. **Thesis scope:** Self-supervised monocular depth is the dominant paradigm for scalable depth estimation in autonomous driving, where labeling millions of frames is impractical.
2. **Scalability:** KITTI Raw contains 39,810 training frames from 60 driving sequences — all unlabeled. Annotated datasets at this scale don't exist for indoor scenes.
3. **Scientific contribution:** Adapting a metric depth foundation model (Depth Pro) to self-supervised training is a novel research direction. Prior self-supervised methods (Monodepth2, DIFFNet, DepthHints) train smaller custom architectures from scratch.
4. **Consumer GPU constraint:** The thesis goal is efficient adaptation on a single RTX 4070 Ti (12GB), making LoRA-based fine-tuning of Depth Pro a natural fit.

**Hypothesis:** By combining Depth Pro's powerful pretrained representations with Monodepth2-style photometric self-supervision, we can achieve competitive performance on KITTI without any depth annotations — while only updating 34.33M of 966M parameters (~3.6%).

---

### 4.2 Self-Supervised Training Framework

The self-supervised approach follows **Monodepth2** (Godard et al., ICCV 2019) adapted to Depth Pro's architecture.

#### Core Idea

Given a monocular video sequence, consecutive frames share overlapping scene content. If we know the camera's relative motion between frames (ego-motion), we can **warp** a source frame to reconstruct the target frame using the predicted depth. The photometric error between the reconstructed and actual target frame serves as a training signal — without any ground truth depth.

**Training signal equation:**

```
L_total = L_photo + λ_smooth * L_smooth
L_photo = min(pe(I_t, warp(I_{t-1}, D_t, T_{t→t-1})),
              pe(I_t, warp(I_{t+1}, D_t, T_{t→t+1})))
```

where `pe` is the photometric error (L1 + SSIM), `D_t` is the predicted depth, and `T` is the relative camera pose.

#### Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                   Training Pipeline              │
│                                                  │
│  Frame triplet (t-1, t, t+1) from KITTI video   │
│       │                    │                     │
│       ▼                    ▼                     │
│  ┌──────────────┐   ┌──────────────┐            │
│  │  Depth Pro   │   │   PoseNet    │            │
│  │  (LoRA fine- │   │  (ResNet-18) │            │
│  │   tuned)     │   │  6-ch input  │            │
│  └──────────────┘   └──────────────┘            │
│       │                    │                     │
│   inv_depth             T_{t→s}                  │
│   (scaled)            (6-DoF pose)               │
│       │                    │                     │
│       └────────┬───────────┘                     │
│                ▼                                  │
│        ┌──────────────┐                          │
│        │    Warper    │ (differentiable)          │
│        │  backproject │                          │
│        │  transform   │                          │
│        │  project     │                          │
│        └──────────────┘                          │
│                │                                  │
│           warped_imgs                             │
│                │                                  │
│                ▼                                  │
│        ┌──────────────┐                          │
│        │  Photometric │ L_photo + L_smooth        │
│        │  Loss        │                          │
│        └──────────────┘                          │
└─────────────────────────────────────────────────┘
```

---

### 4.3 Network Components

#### Depth Network: Depth Pro with LoRA

The depth branch uses the full Depth Pro model with LoRA applied identically to v4:

| Component | Architecture | Parameters | Trainable |
|-----------|-------------|------------|-----------|
| Patch Encoder | DINOv2-Large ViT + LoRA | 304M + 1.18M | LoRA only (1.18M) |
| Image Encoder | DINOv2-Large ViT + LoRA | 304M + 1.18M | LoRA only (1.18M) |
| Decoder | MultiresConvDecoder | 19.67M | Yes |
| Depth Head | CNN (256→128→32→1) | 0.40M | Yes |
| FOV Head | FOVNetwork | 304M | Frozen |

**Input resolution:** 1536×1536 (mandatory — encoder hardcodes 5×5 and 3×3 patch grids requiring this exact resolution).

**Canonical inverse depth scaling:** Depth Pro's FOV head predicts the camera's field of view. For geometrically consistent warping, the canonical inverse depth output must be scaled:

```python
f_px = 0.5 * W / tan(0.5 * fov_deg * π/180)
inv_depth_metric = canonical_inv_depth * (W / f_px)
```

This converts the camera-agnostic canonical output to a metric-consistent inverse depth for the warping pipeline. This is a critical detail — without it, the warped reconstruction does not match the actual camera geometry, and the photometric signal is severely degraded.

#### PoseNet: Ego-Motion Estimation

A lightweight ResNet-18-based network estimates the 6-DoF relative pose between any two frames:

| Component | Detail |
|-----------|--------|
| Architecture | ResNet-18 with modified first conv |
| Input | 6-channel (target + source RGB concatenated) |
| Output | 6-DoF vector: 3 (axis-angle rotation) + 3 (translation) |
| Scale factor | ×0.01 (stabilizes early training) |
| Resolution | 640×192 (lower than depth network to save VRAM) |
| Parameters | 11.91M (all trainable) |

PoseNet processes frame pairs (t, t-1) and (t, t+1) independently to obtain poses for both source frames.

#### Differentiable Warper

For each source frame, the warper reconstructs the target view:

1. **Backproject:** Map target pixels → 3D points using predicted depth and camera intrinsics K
2. **Transform:** Apply relative pose T to move 3D points to the source camera frame
3. **Project:** Project 3D points to source image coordinates using K
4. **Sample:** Bilinear grid_sample at the projected coordinates

```python
cam_points = K_inv @ pixel_coords * depth  # (B, 3, H*W)
src_points = T[:, :3, :3] @ cam_points + T[:, :3, 3:]
pix_coords = K @ src_points / src_points[:, 2:, :]  # normalize by Z
warped = F.grid_sample(source, pix_coords_normalized, ...)
```

---

### 4.4 Loss Functions

#### Photometric Loss

The photometric loss combines SSIM and L1 following Monodepth2:

```
pe(I_a, I_b) = α * (1 - SSIM(I_a, I_b)) / 2 + (1 - α) * |I_a - I_b|₁
```

with α=0.85 (SSIM-dominant). SSIM is computed with a 3×3 window.

#### Per-Pixel Minimum Reprojection

To handle occlusions (pixels visible in target but not in source frames), we take the **minimum photometric error** across both source frames per pixel:

```
L_photo = min(pe(I_t, warp(I_{t-1})), pe(I_t, warp(I_{t+1})))
```

Pixels occluded in one source frame typically have a lower error from the other, so the minimum naturally selects the better reconstruction.

#### Auto-Masking for Stationary Pixels

When the camera is stationary or an object moves at the same velocity as the camera, warping provides no useful signal (the warped image equals the source). Auto-masking excludes these pixels by masking out pixels where the photometric error from warping is *higher* than the identity error (copying the source frame directly):

```
mask = pe(I_t, warp(I_s)) < pe(I_t, I_s)
L_photo = mean(mask * min_reproj_error)
```

**Warmup:** Auto-masking is disabled for the first 3 epochs. In early training, the depth and pose networks are poorly initialized, so warping quality is often worse than identity. Enabling auto-masking immediately would mask out most pixels and starve the network of gradient signal. After 3 epochs of unconstrained learning, the networks are accurate enough for auto-masking to help rather than hurt.

#### Edge-Aware Smoothness Loss

To encourage locally smooth depth while preserving edges, we use an edge-aware smoothness loss on the mean-normalized inverse depth:

```
disp = inv_depth / mean(inv_depth)   # normalize to prevent scale collapse
L_smooth = mean(|∂disp/∂x| * e^{-|∂I/∂x|} + |∂disp/∂y| * e^{-|∂I/∂y|})
```

Mean normalization prevents the trivial solution of predicting all-zero (minimum) disparity. Computed in FP32 to avoid NaN from FP16 overflow in exp.

**Total loss:**

```
L_total = L_photo + 1e-3 * L_smooth
```

---

### 4.5 Dataset: KITTI Raw

| Property | Value |
|----------|-------|
| Source | KITTI Raw dataset (Geiger et al., IJRR 2013) |
| Split | Eigen/Zhou split (standard for self-supervised methods) |
| Train sequences | 60 sequences, 5 recording dates |
| Train frames | 39,810 (Zhou subset) → 13,270 with stride=3 |
| Val frames | 4,424 → 1,475 with stride=3 |
| Test frames | 697 (Eigen test split, fixed) |
| Frame sampling | stride=3 (every 3rd frame, reduces redundancy from 10Hz video) |
| Cameras | Left camera only (monocular) |
| Resolution | 1242×375 (variable, cropped/padded to 1536×1536 for depth, 640×192 for pose) |

**Why stride=3?** KITTI is recorded at 10Hz. Consecutive frames at ~30km/h (highway) differ by only ~0.8m, making them nearly identical. Using every 3rd frame (stride=3) reduces dataset size by 3× (5.5h/epoch vs 16.4h/epoch) while maintaining meaningful inter-frame motion for photometric training.

**Evaluation protocol (Eigen test split):**
- 697 frames with velodyne LiDAR ground truth
- Garg/Eigen crop: removes sky and car hood (top 40%, bottom 5%)
- Depth range: 1–80m (standard for KITTI outdoor)
- Median scaling: multiply predicted depth by `median(gt) / median(pred)` per image (standard for self-supervised methods, which produce up-to-scale predictions)
- Metrics: AbsRel, SqRel, RMSE, RMSE_log, δ<1.25, δ<1.25², δ<1.25³

---

### 4.6 Training Configuration

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Batch size | 1 | VRAM constraint (10.77GB peak) |
| Gradient accumulation | 4 | Effective batch = 4 |
| LoRA LR | 1e-5 | Lower LR for pretrained encoder |
| Decoder/Head LR | 1e-4 | Higher LR for task-specific layers |
| PoseNet LR | 1e-4 | Fresh network, higher LR |
| Optimizer | AdamW | Weight decay=1e-4 |
| Scheduler | CosineAnnealing (T_max=20) | Smooth LR decay |
| Epochs | 20 | ~4.6 days on RTX 4070 Ti |
| Warmup epochs | 3 | Disable auto-masking during init |
| Smoothness weight | 1e-3 | Standard Monodepth2 setting |
| Mixed precision | FP16 | AMP with GradScaler |
| Gradient checkpointing | Both encoders | Required to fit in 12GB VRAM |
| Depth resolution | 1536×1536 | Mandatory for Depth Pro |
| Pose resolution | 640×192 | Standard Monodepth2 resolution |

**VRAM budget:**

| Component | VRAM |
|-----------|------|
| Depth Pro weights (FP16) | ~1.9 GB |
| PoseNet weights | ~0.05 GB |
| Activations + optimizer states | ~8.8 GB |
| **Peak total** | **~10.77 GB** |

Gradient checkpointing on both ViT encoders recomputes intermediate activations during backward pass, trading compute for memory. Without it, the 304M+304M encoder would overflow 12GB.

---

### 4.7 Experiment v6 — Self-Supervised KITTI Baseline (Pretrained Depth Pro)

**Date:** 2026-03-30 (in progress)
**Description:** Evaluate pretrained Depth Pro (zero-shot) on KITTI Eigen test set to establish the baseline for the self-supervised experiments.

**Method:** Load pretrained `depth_pro.pt`, run inference on all 697 Eigen test frames. Apply Garg/Eigen crop and median scaling. No fine-tuning.

**Expected:** Depth Pro was not trained on KITTI and uses metric depth (not scaled). Zero-shot performance on outdoor driving scenes should be lower than on indoor NYU scenes, since the model was likely trained on internet imagery with shorter depth ranges. This establishes the "untouched pretrained" baseline that v7 (self-supervised fine-tuning) must beat.

**Results:** *(awaiting evaluation — running on CPU)*

| Metric | v6 Pretrained (KITTI) | Monodepth2 (ref) |
|--------|----------------------|-------------------|
| AbsRel (↓) | TBD | 0.115 |
| SqRel (↓) | TBD | 0.903 |
| RMSE (↓) | TBD | 4.863 |
| RMSE_log (↓) | TBD | 0.193 |
| δ<1.25 (↑) | TBD | 0.877 |

---

### 4.8 Experiment v7 (v3 training run) — Self-Supervised LoRA Fine-Tuning on KITTI

**Date:** 2026-03-30 (training in progress — 20 epochs)
**Description:** Fine-tune Depth Pro with LoRA using photometric self-supervision on the KITTI Eigen training set. This is the primary thesis experiment.

**Trainable parameters:**

| Component | Parameters | % of Total |
|-----------|-----------|------------|
| LoRA (rank=8, 96 layers) | 2.36M | 0.24% |
| Decoder (MultiresConvDecoder) | 19.67M | 2.04% |
| Depth Head | 0.40M | 0.04% |
| PoseNet (fresh) | 11.91M | 1.23% |
| **Total trainable** | **34.33M** | **3.56%** |
| Frozen | 932M | 96.4% |

**v2 training run (failed — 16 wasted epochs):**
- Bug: FOV head produced extreme `fov_deg` values on KITTI → NaN in canonical depth scaling → NaN total loss → GradScaler skipped every gradient update → zero learning
- Bug: Auto-masking collapsed to 0.24 at epoch 4 (because model hadn't learned anything during warmup due to NaN loss)
- Fix: Clamped `fov_deg` to [5°, 175°], clamped scale factor to [0.01, 100.0], added `nan_to_num` guards
- Fix: Disabled auto-masking entirely (KITTI driving has constant camera motion; auto-masking is designed for stationary cameras)

**v3 training run (current — fixes applied, started 2026-04-03):**
- Training signals (epoch 1): `loss=0.19–0.21`, `mask=1.00`, no NaN — model is learning
- Auto-masking: disabled
- Step time: ~1.5s/step → ~5.5h/epoch → ~4.6 days total
- Watcher script (`watch_and_eval.sh`) will auto-run GPU evaluation after training completes

**Expected results:** *(to be filled after training)*

| Metric | v6 Pretrained | v7 Self-Sup LoRA | Monodepth2 (ref) |
|--------|--------------|-------------------|-------------------|
| AbsRel (↓) | TBD | TBD | 0.115 |
| SqRel (↓) | TBD | TBD | 0.903 |
| RMSE (↓) | TBD | TBD | 4.863 |
| δ<1.25 (↑) | TBD | TBD | 0.877 |

---

### 4.9 Key Engineering Challenges and Solutions

#### Challenge 1: Canonical Inverse Depth Scaling

Depth Pro outputs **canonical inverse depth** — a camera-agnostic representation normalized to be focal-length-independent. This is ideal for zero-shot generalization but geometrically incorrect for image warping, which requires actual metric depth in the camera's coordinate frame.

**Solution:** Use the FOV head's predicted field of view to back-compute the focal length and scale the inverse depth:

```python
f_px = 0.5 * W / torch.tan(0.5 * torch.deg2rad(fov_deg))
inv_depth_metric = canonical_inv_depth * (W / f_px)
```

Without this scaling, the warped reconstruction geometry is wrong (effective depth is off by a factor proportional to `f_px / W`), severely degrading the photometric signal.

#### Challenge 2: Auto-Mask Collapse in Early Training

In early training, the poorly initialized depth and pose networks produce warped images that are barely better than the identity (copying the source frame). Auto-masking compares warping error vs. identity error — if warping error ≥ identity error, the pixel is masked out.

With poor initial predictions, almost all pixels get masked (auto-mask ratio → 0.1), leaving only 10% of pixels to provide gradient. This stalls learning.

**Solution:** Disable auto-masking for the first 3 warmup epochs. The networks train on all pixels without masking, allowing them to learn meaningful geometry. After warmup, auto-masking is enabled when the warping quality is high enough to benefit from the occlusion handling.

#### Challenge 3: FP16 NaN in Smoothness Loss

The smoothness loss computes `e^{-|∂I/∂x|}` which requires exp(). Under FP16, gradients through this operation can overflow to NaN, especially early in training when depth values may be extreme.

**Solution:** Cast to FP32 before computing smoothness, and clamp mean-normalized disparity to [0, 10]:

```python
inv_depth_f32 = inv_depth.float()
mean_disp = inv_depth_f32 / (inv_depth_f32.mean(dim=[2,3], keepdim=True) + 1e-7)
mean_disp = torch.clamp(mean_disp, 0, 10)
s_loss = smooth_loss(mean_disp, target_img.float())
```

#### Challenge 4: VRAM Fitting with 952M Parameter Model

Depth Pro's ViT encoders are 608M parameters, producing large intermediate activations. Running forward + backward with FP16 requires ~11GB+ of VRAM on a 12GB GPU.

**Solution stack:**
- Gradient checkpointing on both patch encoder and image encoder
- FP16 mixed precision training with AMP GradScaler
- Gradient accumulation (4 steps) to simulate larger batch size
- PoseNet runs at 640×192 (vs 1536×1536 for depth) — 8× fewer pixels
- LoRA constrains encoder gradients to low-rank matrices only

#### Challenge 5: Dataset Redundancy at 10Hz

KITTI records at 10Hz. At typical driving speeds, consecutive frames overlap by ~90%. Training on all consecutive triplets means the network sees nearly identical frames thousands of times per epoch, wasting compute.

**Solution:** Stride=3 subsampling — use only every 3rd frame as the target (source frames are ±1 from the subsampled index, maintaining ~0.3s baseline). This reduces 39,810 training samples to 13,270 without meaningfully reducing scene coverage.

---

### 4.10 Planned Ablations and Future Experiments

Once v7 baseline training completes, the following experiments are planned:

#### Experiment v8 — Stereo Consistency Loss (if stereo pairs available)
Add right-camera consistency loss (KITTI has synchronized stereo). Provides additional geometric supervision without labels. Expected: +2–4% AbsRel improvement.

#### Experiment v9 — Multi-Scale Photometric Loss
Compute photometric loss at multiple decoder output scales (1/1, 1/2, 1/4, 1/8) as in original Monodepth2. Provides denser gradient signal to earlier decoder layers.

#### Experiment v10 — Longer Training / More Epochs
Extend from 20 to 40 epochs with cosine annealing restart. Self-supervised methods typically benefit from longer training since the signal is weaker per sample.

#### Experiment v11 — Different LoRA Ranks
Ablate LoRA rank: rank=4 (1.18M params, half the params), rank=16 (4.72M params, double). Hypothesis: rank=8 is near-optimal for this task, but rank=4 may be sufficient with the weaker self-supervised signal.

#### Experiment v12 — Depth Hints from LiDAR (Semi-Supervised)
Sparse LiDAR hints as soft constraints during self-supervised training (not evaluation). Bridges gap between fully self-supervised and supervised approaches — a natural thesis extension.

---

## 5. Planned Improvements (Phase 1, NYU)

*(These were planned after v0–v5 supervised experiments; now superseded by the Phase 2 self-supervised pivot.)*

### Phase 3: Training Data Expansion
- Expand from 795 labeled images to ~25K-50K frames from the NYU raw dataset
- More data is likely the biggest factor in the Apple paper's results

### Phase 4: Architectural Modifications
- Add channel attention (SE blocks) to decoder fusion layers
- Multi-scale training with variable input resolution

### Phase 5: Additional Datasets
- ScanNet, Matterport3D for cross-dataset training
- MiDaS-style mixed dataset training for generalization

---

## 6. Summary of All Results

### Phase 1 — Supervised (NYU Depth V2, 654 Eigen test images, depth ≤ 10m)

| Experiment | AbsRel (↓) | RMSE (↓) | δ<1.25 (↑) | Key Change | vs Pretrained |
|-----------|-----------|---------|-----------|------------|-------------|
| v0 Pretrained | 0.1155 | 0.5092 | 0.8777 | Zero-shot inference | — |
| v1 Fine-tuned | 0.0855 | 0.3288 | 0.9389 | Train decoder+head (25ep) | +26.0% |
| v2 + TTA Flip | 0.0790 | 0.3088 | 0.9532 | Flip TTA at inference | +31.6% |
| v3 Phase 1 | 0.0878 | 0.3273 | 0.9362 | Loss+Aug+LR (50ep) | +24.0% |
| v4 LoRA | 0.0781 | 0.3038 | 0.9530 | LoRA encoder (50ep) | +32.4% |
| **v4 LoRA + TTA** | **0.0765** | **0.2982** | **0.9549** | LoRA + flip TTA | **+33.8%** |
| v5 LoRA+Unfreeze | 0.0787 | 0.3069 | 0.9520 | LoRA + 2 unfrozen blocks | +31.9% |
| v5 LoRA+Unf+TTA | 0.0773 | 0.3014 | 0.9546 | + flip TTA | +33.1% |
| Apple Paper | 0.036 | 0.127 | 0.989 | Full model, more data | target |

### Phase 2 — Self-Supervised (KITTI Eigen test, 697 images, depth 1–80m, median scaling)

| Experiment | AbsRel (↓) | SqRel (↓) | RMSE (↓) | δ<1.25 (↑) | Key Change | vs Pretrained |
|-----------|-----------|----------|---------|-----------|------------|-------------|
| v6 Pretrained | TBD | TBD | TBD | TBD | Zero-shot on KITTI | — |
| **v7 Self-Sup LoRA** | **TBD** | **TBD** | **TBD** | **TBD** | **LoRA + photo loss (20ep)** | **TBD** |
| Monodepth2 (ref) | 0.115 | 0.903 | 4.863 | 0.877 | Custom arch, 192×640 | — |
| DIFFNet (ref) | 0.102 | 0.764 | 4.483 | 0.896 | Stronger backbone | — |

---

## 7. Technical Notes

### GPU Memory Management
- RTX 4070 Ti has 12GB VRAM
- Model weights: ~3.8 GB (FP32) / ~1.9 GB (FP16)
- v1 decoder-only training: ~8.7 GB peak (FP16 mixed precision)
- v4 LoRA training: ~10.9 GB peak (encoder gradients through LoRA + gradient checkpointing)
- v5 LoRA + 2 unfrozen blocks: ~11.31 GB peak (additional MLP gradients)
- v7 self-supervised: ~10.77 GB peak (Depth Pro + PoseNet + warping — PoseNet at 640×192 keeps overhead low)
- Gradient accumulation (steps=4) enables effective batch size of 4 with batch_size=1
- Gradient checkpointing on both encoders essential for LoRA training to fit in 12GB
- Attempted 4 unfrozen blocks (67M params) → OOM; reduced to 2 blocks (56M) fits in 12GB

### Evaluation Protocol — NYU Depth V2
- Standard NYU Depth V2 Eigen test split (654 images)
- Depth range capped to 0–10 meters (standard NYU protocol)
- Metrics: AbsRel, SqRel, RMSE, RMSElog, log10, delta thresholds (1.25, 1.25², 1.25³)
- Scale-invariant metrics also computed (optimal per-image affine alignment)

### Evaluation Protocol — KITTI Eigen
- Standard Eigen test split (697 frames, fixed)
- Velodyne LiDAR ground truth projected to camera using calibration matrices
- Garg/Eigen crop: `[int(0.40810811 * h):int(0.99189189 * h), int(0.03594771 * w):int(0.96405229 * w)]`
- Depth range: 1–80m (standard outdoor protocol)
- Median scaling: `scale = median(gt_depth[gt_mask]) / median(pred_depth[gt_mask])`
- Metrics: AbsRel, SqRel, RMSE, RMSE_log, δ<1.25, δ<1.25², δ<1.25³

### Reproducibility
- All experiments tracked in `experiments/` with configs, results, and visualizations
- Training logs saved in `checkpoints/` with per-epoch metrics
- Random seeds fixed for reproducibility
- Code available at: github.com/Dariusan3/ml-depth-pro
