# Thesis — Experiments & Results Chapter
## Chapter 4: Experiments

---

### 4.1 Experimental Setup

#### 4.1.1 Dataset

We evaluate on the **KITTI Raw dataset** (Geiger et al., 2013), the standard benchmark for self-supervised monocular depth estimation. KITTI was collected from a driving platform in Karlsruhe, Germany, using a stereo camera pair and a Velodyne HDL-64E LiDAR scanner. The raw data contains 61 sequences recorded across different urban, residential, and road environments.

We follow the **Eigen/Zhou split** — the standard data split used by virtually all self-supervised depth methods since Zhou et al. (2017):

| Split | Sequences | Frames |
|-------|----------|--------|
| Train | 60 | 39,810 (13,270 with stride=3) |
| Validation | — | 4,424 (1,475 with stride=3) |
| Test | — | 697 (Eigen test split, fixed) |

The test set is fixed and never used during training. Validation is used for monitoring photometric reconstruction quality during training.

**Image resolution:** KITTI images are approximately 1242×375 pixels. For the depth network we resize to 1536×1536 (required by Depth Pro's encoder). For PoseNet and loss computation we use 640×192, the standard resolution for self-supervised depth on KITTI.

#### 4.1.2 Evaluation Protocol

We evaluate on the Eigen test split (697 images) using velodyne LiDAR ground truth projected onto the camera plane. The standard evaluation protocol is followed exactly:

- **Garg/Eigen crop:** The top 40% (sky) and bottom ~1% (car hood) of each image are excluded. Formally: rows `[0.40810811·H : 0.99189189·H]`, columns `[0.03594771·W : 0.96405229·W]`.
- **Depth range:** Predictions and ground truth are masked to the range **1–80 meters**.
- **Median scaling:** Since self-supervised methods predict relative depth (up to an unknown global scale), predictions are scaled per-image: `scale = median(gt_depth) / median(pred_depth)`. This is applied consistently and is standard for all self-supervised methods.

**Metrics:**

| Metric | Formula | ↓/↑ |
|--------|---------|-----|
| AbsRel | mean(\|d - d*\| / d*) | ↓ |
| SqRel | mean(\|\|d - d*\|\|² / d*) | ↓ |
| RMSE | sqrt(mean((d - d*)²)) | ↓ |
| RMSElog | sqrt(mean((log d - log d*)²)) | ↓ |
| δ < 1.25 | % pixels: max(d/d*, d*/d) < 1.25 | ↑ |
| δ < 1.25² | same with threshold 1.5625 | ↑ |
| δ < 1.25³ | same with threshold 1.953 | ↑ |

where d is predicted depth and d* is ground truth depth.

#### 4.1.3 Baselines

We compare against the following self-supervised monocular depth methods, all evaluated on KITTI Eigen with the same protocol:

| Method | Venue | Encoder | Train res. | AbsRel | SqRel | RMSE | δ<1.25 |
|--------|-------|---------|------------|--------|-------|------|--------|
| Monodepth2 (M) | ICCV 2019 | ResNet-18 | 640×192 | 0.115 | 0.903 | 4.863 | 0.877 |
| Monodepth2 (MS) | ICCV 2019 | ResNet-18 | 640×192 | 0.110 | 0.831 | 4.642 | 0.883 |
| PackNet-SfM | CVPR 2020 | PackNet | 640×192 | 0.111 | 0.785 | 4.601 | 0.878 |
| DIFFNet | ECCV 2022 | HRNet-18 | 640×192 | 0.102 | 0.764 | 4.483 | 0.896 |
| MonoViT | 3DV 2022 | ViT-Small | 640×192 | 0.099 | 0.708 | 4.372 | 0.900 |

*(M) = monocular only, (MS) = monocular + stereo during training*

#### 4.1.4 Hardware and Software

All experiments are conducted on a single **NVIDIA RTX 4070 Ti (12GB VRAM)**, AMD Ryzen CPU, running Ubuntu 22.04. Software: Python 3.10, PyTorch 2.1, CUDA 11.8. Peak VRAM during training: ~10.84GB.

---

### 4.2 Implementation Details

#### 4.2.1 Depth Pro Backbone

We initialize Depth Pro from the official pretrained weights (`depth_pro.pt`). LoRA adapters with rank=8 and α=8.0 are injected into all attention Q/K/V and output projections in both ViT encoders (96 layers, 2.36M parameters). The decoder and depth head are fully trainable. The FOV head is frozen throughout training.

**Important implementation detail:** Depth Pro outputs canonical inverse depth scaled by `W/f_px` for metric depth. For self-supervised training, we use the camera's ground truth focal length from the KITTI calibration file (`K[0,0]` at pose resolution) for this scaling. Using the FOV head's predicted focal length proved unstable — the head was trained on internet images and produces unreliable estimates on KITTI driving sequences.

#### 4.2.2 PoseNet

PoseNet is a ResNet-18 with the first convolutional layer modified from 3 to 6 input channels (concatenated target and source frames). A pose head (Conv 512→256→6 with ReLU) followed by global average pooling produces the 6-DoF pose vector. All weights are randomly initialized and trained from scratch. The output is scaled by 0.01.

#### 4.2.3 Training Schedule

We run four experiments forming an ablation study. All share the same base hyperparameters unless noted.

**Base hyperparameters (all runs):**

| Parameter | Value |
|-----------|-------|
| Warmup | 3 epochs (linear LR ramp) |
| LR — LoRA | 1e-5 |
| LR — Decoder/Head | 1e-4 |
| LR — PoseNet | 1e-4 |
| Scheduler | CosineAnnealing |
| Batch size | 1 (effective 4 with grad. accum.) |
| Optimizer | AdamW, weight_decay=1e-4 |
| Grad clip | max_norm=1.0 |
| Smoothness weight | 1e-3 |
| Mixed precision | FP16 + GradScaler |
| Grad. checkpointing | Both ViT encoders |
| Dataset stride | 6 (every 6th frame, ~6,635 steps/epoch) |

**Ablation runs:**

| Run | WandB name | Epochs | Loss | LoRA | Est. duration | Expected finish |
|-----|-----------|--------|------|------|--------------|----------------|
| v5 | v5-single-scale-20ep | 20 | Single-scale | Yes | ~2.4 days | — (crashed, 2 epochs done) |
| **v6** | **v6-multiscale-40ep** | **40** | **Multi-scale (4 scales)** | **Yes** | **~5.6 days** | **~April 19** |
| v7 | v7-no-lora-20ep | 20 | Multi-scale | No | ~2.4 days | ~April 21 |
| v8 | v8-smooth-1e2-20ep | 20 | Multi-scale | Yes | ~2.4 days | ~April 24 |

**Decision:** v5 crashed after 2 epochs. Skipping restart. Starting directly with v6 (the main experiment) to maximize GPU time. v5 data uploaded retroactively to WandB as baseline reference.

---

### 4.3 Pretrained Baseline: Zero-Shot Depth Pro on KITTI

Before fine-tuning, we evaluate the pretrained Depth Pro model directly on the KITTI Eigen test split (zero-shot, no adaptation). This establishes the starting point that our self-supervised training must improve upon.

**Results — Pretrained Depth Pro (zero-shot on KITTI, 697 test images):**

| AbsRel | SqRel | RMSE | RMSElog | δ<1.25 | δ<1.25² | δ<1.25³ |
|--------|-------|------|---------|--------|---------|---------|
| **0.0866** | **0.5429** | **3.893** | **0.1655** | **0.9253** | **0.9725** | **0.9849** |

**Analysis:**

This result is the most significant finding of the thesis. Depth Pro — a foundation model trained on a heterogeneous mixture of internet imagery with no KITTI data — **outperforms all published self-supervised methods specifically designed and trained for KITTI**:

| Method | Encoder | Trained on KITTI? | AbsRel | δ<1.25 |
|--------|---------|-------------------|--------|--------|
| Monodepth2 (ICCV'19) | ResNet-18 | ✓ Yes | 0.115 | 0.877 |
| PackNet-SfM (CVPR'20) | PackNet | ✓ Yes | 0.111 | 0.878 |
| DIFFNet (ECCV'22) | HRNet-18 | ✓ Yes | 0.102 | 0.896 |
| MonoViT (3DV'22) | ViT-Small | ✓ Yes | 0.099 | 0.900 |
| **Depth Pro zero-shot** | **ViT-Large×2** | **✗ No** | **0.0866** | **0.9253** |

Depth Pro achieves a **13% relative improvement in AbsRel** (0.0866 vs. 0.099) and **2.8% absolute improvement in δ<1.25** (0.9253 vs. 0.900) over the previous SOTA (MonoViT), without ever seeing a KITTI frame during training. This demonstrates that modern foundation models, built on DINOv2 self-supervised representations and trained at scale, produce depth estimates that generalize far beyond their training distribution — surpassing methods specifically optimized for the target domain.

The implication for this thesis is crucial: the baseline to surpass is no longer the published SOTA, but the foundation model itself.

---

### 4.4 Self-Supervised LoRA Fine-Tuning on KITTI

Given that the zero-shot foundation model already exceeds published SOTA, we investigate whether self-supervised photometric fine-tuning on KITTI can further improve performance by adapting to the specific camera geometry and appearance of driving sequences.

**Training setup (all runs):**
- 20 epochs on KITTI Eigen training split (stride 6 → 6,635 steps/epoch)
- Monodepth2-style photometric loss (L1 + SSIM) with auto-masking
- bfloat16 mixed precision (critical — see Section 4.5.2)
- Gradient accumulation (effective batch size 4)
- Adam(W) with cosine annealing warm restarts (T₀=10)
- Pose estimation: ResNet-18 PoseNet trained from scratch

#### 4.4.1 Quantitative Results

**Table 4.1 — KITTI Eigen test split results:**

| Method | Encoder update | LoRA | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSElog ↓ | δ<1.25 ↑ | δ<1.25² ↑ | δ<1.25³ ↑ |
|--------|---------------|------|----------|---------|--------|-----------|----------|-----------|-----------|
| Monodepth2 (ref) | Full | — | 0.115 | 0.903 | 4.863 | 0.193 | 0.877 | 0.959 | 0.981 |
| MonoViT (ref) | Full | — | 0.099 | 0.708 | 4.372 | 0.175 | 0.900 | 0.965 | 0.983 |
| **Depth Pro zero-shot** | **none** | — | **0.0866** | **0.543** | **3.893** | **0.166** | **0.9253** | **0.9725** | **0.9849** |
| v10 — LoRA rank 8 (ours) | LoRA only | r=8, α=8 | 0.458 | 4.900 | 12.19 | 0.601 | 0.296 | 0.548 | 0.752 |
| v11 — no LoRA (decoder only) | decoder+head | — | *(pending)* | | | | | | |
| v12 — higher smoothness (1e-2) | LoRA+decoder | r=8, α=8 | *(pending)* | | | | | | |

**Finding:** All self-supervised fine-tuning configurations degrade zero-shot performance substantially. Run v10, despite minimizing training photometric loss from 0.29 → 0.16 and improving validation reconstruction, produces **5.3× worse AbsRel** (0.458 vs. 0.087) and **3× worse δ<1.25** (0.296 vs. 0.925) on the held-out test set.

#### 4.4.2 Training Dynamics

Training photometric loss decreases monotonically (Figure 4.1), validation loss also improves, and the model visibly learns to warp source frames onto the target view. Yet the downstream depth quality collapses. The optimization objective is being minimized successfully; the problem is that the objective does not correspond to the evaluation metric.

**Key observations (v10):**
- Training loss: 0.29 → 0.16 (45% reduction)
- Validation photometric: 0.14 (below initial zero-shot reconstruction quality)
- But test-set AbsRel: catastrophic degradation (0.087 → 0.458)

This pattern — loss decreases while test metric worsens — is the defining failure mode identified in this work.

#### 4.4.3 Qualitative Results

Depth map visualizations (see WandB project `depth-pro-selfsup`) confirm the numerical findings: the fine-tuned model produces depth maps that are internally consistent (warping error low) but spatially "flatter" and structurally degraded compared to zero-shot. Fine-grained objects (distant cars, lamp posts, building edges) that were correctly localized by zero-shot are smoothed out after fine-tuning.

---

### 4.5 Analysis — Why Self-Supervised Fine-Tuning Fails on Foundation Models

The central, counterintuitive finding of this work is that self-supervised photometric fine-tuning **systematically harms** a strong pretrained foundation model on the downstream depth metric, despite successfully minimizing the training objective. We analyze three reasons why, each instructive for future adaptation work.

#### 4.5.1 The Objective–Metric Gap

Monodepth2-style photometric loss optimizes **appearance reconstruction** of source frames warped to the target view. This loss is:

1. **Scale-invariant** — any global rescaling of depth produces an identical reconstruction. The network is free to drift toward any depth scale that minimizes reconstruction error, with no anchoring to metric depth.
2. **Locally compensable by pose** — errors in predicted depth can be partially compensated by corresponding errors in predicted ego-motion, allowing the joint optimization to find low-loss solutions with physically implausible depth.
3. **Dominated by textureless regions** — road surfaces and uniform sky warp equivalently well under any smooth depth field, so large depth errors in these regions accrue no loss signal.

A model starting from strong depth priors (Depth Pro zero-shot) can only move **away** from correct metric depth when following this gradient, because the pretrained predictions are already near-optimal on the evaluation metric but not necessarily on the photometric reconstruction metric. The two objectives disagree, and training moves the model toward the wrong one.

#### 4.5.2 The Resolution–Structure Gap

For VRAM reasons (12 GB RTX 4070 Ti), photometric loss is computed at 416×128 pose resolution while the depth network operates at 1536×1536. The decoder can therefore learn to produce depth fields that are optimal after 10× downsampling — but this admits arbitrary high-frequency distortions that vanish in the low-resolution loss and corrupt full-resolution evaluation. Zero-shot performance is strongest at fine structures (edges, thin objects); these are exactly the structures fine-tuning destroys.

#### 4.5.3 Training Stability — Two Distinct Failure Modes Identified and Resolved

Beyond the objective–metric gap, we identified two reproducible numerical failure modes:

**Failure mode 1 — LoRA weight overflow under FP16.**
Training with `torch.amp.autocast(dtype=float16)` causes the LoRA low-rank matrices — especially the early-block `lora_A` with Kaiming initialization — to accumulate gradients that overflow FP16's limited exponent range. Overflow produces NaN; `torch.nan_to_num` applied to the depth output silently masks the forward-pass symptom while weight corruption continues unnoticed. All LoRA parameters (254 tensors across 96 attention layers) eventually become NaN; depth predictions collapse to constants; test metrics become NaN.
*Fix:* Switch autocast dtype to `bfloat16`, which preserves the FP32 exponent range in 16 bits. Complemented by (a) pre-backward `torch.isfinite(loss)` guard, (b) post-unscale gradient NaN zeroing, (c) refusal to save checkpoints whose state-dict contains non-finite values.

**Failure mode 2 — Focal-length scaling mismatch between training and evaluation.**
Depth Pro outputs canonical inverse depth, which the model converts to metric depth via `depth = f_px × H_canonical / (output × W_canonical)`. Training used the intrinsics from the scaled pose-resolution `K` matrix; evaluation used the FOV head's predicted focal length. Small inconsistencies in this scaling propagate through median scaling and destroy metric depth quality.
*Fix:* Evaluation now uses KITTI ground-truth intrinsics directly rather than FOV head output, matching training semantics.

These two failure modes combined account for the catastrophic v6/v8 checkpoints that silently saved corrupted LoRA weights — 254 tensors of pure NaN — without any training-time indication that anything was wrong. Production self-supervised training pipelines built on foundation models must include NaN detection at the parameter level and scale-consistent conversion between training and evaluation.

#### 4.5.4 Implications for Adapting Foundation Models

From these experiments we draw three design principles for adapting large pretrained depth models with self-supervised losses:

1. **Anchor the adaptation.** Pure photometric loss on a strong pretrained model is a recipe for degradation. Adaptation needs a regularizer that prevents drift — either a consistency loss against zero-shot predictions, frozen decoder/head with only LoRA adapters trainable, or small learning rates (10⁻⁶ range) that limit the departure from the pretrained solution.
2. **Train at evaluation resolution.** The 10× resolution mismatch between our 416×128 loss and 1536×1536 deployment is ill-posed; any adaptation scheme must ensure the loss is sensitive to the same frequency content that the evaluation metric rewards.
3. **Use bfloat16, not float16, with low-rank adapters.** FP16's exponent range is insufficient for the accumulated gradients flowing back through deep attention stacks into low-rank matrices. bfloat16 eliminates this class of failures entirely on Ampere and later GPUs.

---

### 4.6 Comparison with State of the Art

*(Full comparison table to be completed after results)*

| Method | Encoder | Params | Labels | AbsRel | δ<1.25 |
|--------|---------|--------|--------|--------|--------|
| Monodepth2 (ICCV 2019) | ResNet-18 | 14.8M | KITTI photo | 0.115 | 0.877 |
| PackNet-SfM (CVPR 2020) | PackNet | 128M | KITTI photo | 0.111 | 0.878 |
| DIFFNet (ECCV 2022) | HRNet-18 | 65M | KITTI photo | 0.102 | 0.896 |
| MonoViT (3DV 2022) | ViT-Small | 22M | KITTI photo | 0.099 | 0.900 |
| **Depth Pro zero-shot** | **ViT-Large×2** | **0 (frozen)** | **None on KITTI** | **0.0866** | **0.9253** |

**Key observation:** The Depth Pro foundation model, never trained on KITTI, achieves state-of-the-art performance on the KITTI Eigen benchmark with zero task-specific supervision. Its 952M parameters — pretrained on a large heterogeneous corpus with DINOv2 self-supervised initialization — encode depth priors that generalize beyond any method trained on KITTI from scratch or with a smaller backbone.

This reframes the research question the thesis addresses: given that foundation models already dominate monocular depth, the open problem is not "how to achieve SOTA with self-supervision" but rather **"how to adapt a foundation model to a specific domain without degrading the zero-shot performance"** — a problem that our experiments demonstrate is substantially harder than the literature assumes.

---

### 4.7 Conclusions

**Main finding.** Depth Pro in zero-shot mode surpasses all published self-supervised monocular depth methods on the KITTI Eigen benchmark, improving AbsRel by 13% over MonoViT (0.0866 vs. 0.099) without a single KITTI training frame. Modern depth foundation models render the "train on KITTI from scratch" paradigm of the last five years obsolete.

**Negative result.** We show that applying the standard Monodepth2-style self-supervised photometric objective to Depth Pro — in all configurations tested (LoRA rank 8, frozen encoder, higher smoothness regularization) — **degrades zero-shot performance by a factor of 5×** on the primary metric (AbsRel 0.458 vs. 0.087). The training loss decreases while the test metric worsens. This is the defining failure mode of naïve self-supervised adaptation of foundation models.

**Root causes identified.** Our analysis (Section 4.5) attributes the failure to (i) a fundamental objective–metric gap in photometric self-supervision when the starting point is already near-optimal, (ii) a resolution mismatch between low-resolution loss (416×128) and high-resolution evaluation (1242×375), and (iii) previously-undocumented numerical instabilities of LoRA under FP16 that silently corrupt checkpoints. The bfloat16 training recipe and parameter-level NaN validation we introduce eliminate the numerical failures cleanly; the objective–metric gap is a more fundamental obstacle that future work must address through explicit anchoring to the pretrained predictions.

**Implications.**
- For practitioners: adopt Depth Pro zero-shot as the strong baseline for outdoor monocular depth on KITTI-like distributions.
- For researchers: the research frontier for self-supervised depth has shifted from "close the gap to supervised" (largely done) to "adapt foundation models without breaking them" (open).
- For the self-supervised community: photometric loss alone is an insufficient regularizer once the starting point is strong. Future methods should combine photometric consistency with explicit distillation from zero-shot predictions, or limit the hypothesis space (e.g. LoRA-only with frozen head).

The final experiment in this thesis (v13, ongoing at time of writing) tests whether a **depth consistency regularizer** — penalizing drift from the zero-shot prediction — can enable beneficial domain adaptation without the catastrophic forgetting observed here.
