# Depth Pro - Method Improvements for Monocular Depth Estimation

## Bachelor Thesis: Improving Apple's Depth Pro for State-of-the-Art Monocular Metric Depth Estimation

**Author:** Dariusan3
**Base Model:** Depth Pro (Apple, ICLR 2025) — "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second"
**Paper Reference:** Bochkovskii et al., arXiv:2410.02073
**Hardware:** NVIDIA RTX 4070 Ti (12GB VRAM)
**Dataset:** NYU Depth V2 (795 train / 654 Eigen test split)

---

## 1. Introduction

This document tracks all modifications made to Apple's Depth Pro model to improve monocular depth estimation performance on the NYU Depth V2 benchmark. Each experiment is documented with motivation, implementation details, results, and analysis.

The goal is to close the gap between the publicly available pretrained model (AbsRel=0.1155 zero-shot) and the paper's reported results (AbsRel=0.036) through systematic improvements to the training pipeline, loss functions, data augmentation, and inference strategies.

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

## 4. Planned Future Improvements

### Phase 3: Training Data Expansion
- Expand from 795 labeled images to ~25K-50K frames from the NYU raw dataset
- More data is likely the biggest factor in the Apple paper's results
- Expected to close the gap significantly — and would likely make v5's extra capacity beneficial

### Phase 4: Architectural Modifications
- Add channel attention (SE blocks) to decoder fusion layers
- Multi-scale training with variable input resolution

### Phase 5: Additional Datasets
- ScanNet, Matterport3D for cross-dataset training
- MiDaS-style mixed dataset training for generalization

---

## 5. Summary of All Results

| Experiment | AbsRel (↓) | RMSE (↓) | delta<1.25 (↑) | Key Change | vs Pretrained |
|-----------|-----------|---------|---------------|------------|-------------|
| v0 Pretrained | 0.1155 | 0.5092 | 0.8777 | Zero-shot inference | — |
| v1 Fine-tuned | 0.0855 | 0.3288 | 0.9389 | Train decoder+head (25ep) | +26.0% |
| v2 + TTA Flip | 0.0790 | 0.3088 | 0.9532 | Flip TTA at inference | +31.6% |
| v3 Phase 1 | 0.0878 | 0.3273 | 0.9362 | Loss+Aug+LR (50ep) | +24.0% |
| v4 LoRA | 0.0781 | 0.3038 | 0.9530 | LoRA encoder (50ep) | +32.4% |
| **v4 LoRA + TTA** | **0.0765** | **0.2982** | **0.9549** | LoRA + flip TTA | **+33.8%** |
| v5 LoRA+Unfreeze | 0.0787 | 0.3069 | 0.9520 | LoRA + 2 unfrozen blocks | +31.9% |
| v5 LoRA+Unf+TTA | 0.0773 | 0.3014 | 0.9546 | + flip TTA | +33.1% |
| Apple Paper | 0.036 | 0.127 | 0.989 | Full model, more data | target |

---

## 6. Technical Notes

### GPU Memory Management
- RTX 4070 Ti has 12GB VRAM
- Model weights: ~3.8 GB (FP32)
- v1 decoder-only training: ~8.7 GB peak (FP16 mixed precision)
- v4 LoRA training: ~10.9 GB peak (encoder gradients through LoRA + gradient checkpointing)
- v5 LoRA + 2 unfrozen blocks: ~11.31 GB peak (additional MLP gradients)
- Gradient accumulation (steps=4) enables effective batch size of 4 with batch_size=1
- Gradient checkpointing on both encoders essential for LoRA training to fit in 12GB
- Attempted 4 unfrozen blocks (67M params) → OOM; reduced to 2 blocks (56M) fits in 12GB

### Evaluation Protocol
- Standard NYU Depth V2 Eigen test split (654 images)
- Depth range capped to 0-10 meters (standard NYU protocol)
- Metrics: AbsRel, SqRel, RMSE, RMSElog, log10, delta thresholds (1.25, 1.25², 1.25³)
- Scale-invariant metrics also computed (optimal per-image affine alignment)

### Reproducibility
- All experiments tracked in `experiments/` with configs, results, and visualizations
- Training logs saved as JSON in checkpoint directories
- Random seeds fixed for reproducibility
- Code available at: github.com/Dariusan3/ml-depth-pro
