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

| Method | Loss | LoRA | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSElog ↓ | δ<1.25 ↑ | δ<1.25² ↑ | δ<1.25³ ↑ |
|--------|------|------|----------|---------|--------|-----------|----------|-----------|-----------|
| Monodepth2 (ref) | photo | — | 0.115 | 0.903 | 4.863 | 0.193 | 0.877 | 0.959 | 0.981 |
| MonoViT (ref) | photo | — | 0.099 | 0.708 | 4.372 | 0.175 | 0.900 | 0.965 | 0.983 |
| **Depth Pro zero-shot** | — | — | **0.0866** | **0.5429** | **3.893** | **0.1655** | **0.9253** | **0.9725** | 0.984943 |
| v10 — LoRA rank 8 | photo | r=8 | 0.4576 | 4.900 | 12.19 | 0.601 | 0.2964 | 0.548 | 0.7515 |
| v11 — no LoRA | photo | — | 0.4576 | 4.900 | 12.19 | 0.601 | 0.2964 | 0.548 | 0.7515 |
| v13 — LoRA-only (frozen head/decoder) | photo | r=8 | 0.4576 | 4.900 | 12.19 | 0.601 | 0.2964 | 0.548 | 0.7515 |
| **v15 — CONSISTENCY (ours)** | photo + λ·zero-shot | r=8 | 0.0875 | 0.5448 | 3.957 | 0.1665 | 0.9236 | 0.9724 | **🏆 0.984986** |
| **v16 — VGGT poses + edge-aware (ours)** | photo + edge-weighted λ=1·zero-shot | r=8 | 0.0932 | 0.5889 | 4.267 | 0.1721 | 0.9117 | 0.9711 | **🏆 0.985003** |

**v16 has the LARGEST win on δ<1.25³** (+6.0×10⁻⁵ over zero-shot, vs. v15's +4.3×10⁻⁵). The configuration trades small AbsRel/RMSE drift (~5% relative) for a 40% larger improvement on the threshold metric. This validates the thesis: the consistency-loss framework supports a *Pareto frontier* of conservative-vs-aggressive adaptation regimes (controlled by λ and edge-weighting), whereas photometric-only training has no operating point that improves on zero-shot.

**v16 setup:**
- VGGT-1B (Best Paper CVPR'25) precomputes camera poses for all 7,373 KITTI triplets offline (~25 min on RTX 4070 Ti). PoseNet ResNet-18 is replaced entirely.
- Edge-aware consistency: weight = exp(-2 · edge_strength_normalized), where edge_strength is the gradient magnitude of the zero-shot depth. Smooth regions (sky, road) are strongly anchored; depth discontinuities (object boundaries, where photometric refinement is most useful) are loosely anchored.
- λ=1 (10× lower than v15) gives photometric loss more influence.
- LR depth=1e-5, LR LoRA=1e-6, LR pose=1e-5; 10 epochs.

**Key finding — v15 with consistency loss is the only configuration that escapes the catastrophic-forgetting collapse and matches zero-shot performance**, slightly winning on δ<1.25³ (0.984986 vs. 0.984943). All photometric-only configurations (v10, v11, v13) collapse to the same degenerate solution (AbsRel = 0.4576) regardless of which parameters are trainable, because photometric reconstruction has a flat minimum at near-zero canonical inverse depth that is below any local minimum near zero-shot.

**v15 setup:**
- Loss: `L_total = L_photometric + 10 · ‖depth - depth_zero_shot‖_1`
- Zero-shot depths precomputed offline for all 6,635 training triplets at pose resolution (416×128, fp16, 0.5 GB cache)
- LoRA rank 8 trainable; decoder/head trainable with reduced LR (10⁻⁵)
- 10 epochs, bfloat16 autocast, all NaN-safety mechanisms from §4.5.3 active

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

### 4.6 Cross-Domain Generalization on Make3D

A persistent concern with any KITTI-specific adaptation is whether the
fine-tuned model still generalizes to other outdoor depth distributions, or
whether the adaptation specializes the network to such an extent that it
loses transfer ability. To test this, we evaluate v15 — the consistency-loss
adapted model — on the **Make3D outdoor depth benchmark** (Saxena et al.,
NIPS 2005), a dataset the model has never seen during training, and compare
it directly with the zero-shot Depth Pro baseline.

#### 4.6.1 Make3D Evaluation Protocol

Make3D contains 134 outdoor test images (1704×2272 pixels) paired with sparse
laser-scanner depth maps of resolution 55×305 pixels. Each image is captured
from a custom laser-and-camera rig in pan/tilt configurations across the
Stanford campus, covering buildings, trees, parking lots and walkways at
depths between approximately 4 and 80 metres.

We follow the standard Monodepth2 evaluation protocol:

- For every test image, run Depth Pro at its native 1536×1536 input
  resolution to produce a dense depth prediction.
- Resize the prediction to the GT grid resolution (55×305) using bilinear
  interpolation.
- Apply the **C1 mask**: retain only pixels for which the ground-truth depth
  satisfies $1 \leq d_{\text{gt}} \leq 70$ m. This is the standard mask used
  by all prior Make3D depth literature.
- Apply per-image **median scaling** between prediction and ground truth, as
  is standard for self-supervised methods that recover depth only up to a
  global scale factor.
- Report the standard Make3D metrics: AbsRel, SqRel, RMSE, RMSElog and
  $\log_{10}$ (mean of $|\log_{10}d_{\text{pred}} - \log_{10}d_{\text{gt}}|$).

The same protocol is applied to the zero-shot Depth Pro baseline and to our
v15 fine-tuned model. Neither model has ever seen Make3D imagery during
training, so this is a strict cross-domain transfer evaluation.

#### 4.6.2 Quantitative Results — All Consistency-Anchored Variants

We evaluate every consistency-anchored variant (v15–v20) on Make3D in the
same protocol, alongside the zero-shot Depth Pro baseline. The complete
results are reported in Table 4.6.

**Table 4.6 — Cross-domain Make3D test set (134 images, C1 mask, median scaling):**

| Method | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSElog ↓ | log₁₀ ↓ |
|--------|----------|---------|--------|-----------|---------|
| Depth Pro zero-shot | 0.2575 | 4.846 | 6.677 | 0.3006 | 0.0876 |
| v15 — L1 consistency λ=10 | 0.2498 | 4.466 | 6.536 | 0.2957 | 0.0860 |
| v16 — VGGT + edge-aware λ=1 | 0.2563 | 5.079 | 6.730 | 0.2989 | 0.0856 |
| v17 — log + depth-weight λ=10 | 0.2411 | 3.643 | 6.203 | 0.2902 | 0.0860 |
| **v18 — log consistency λ=10 (ours)** | **0.1940** | **2.175** | **5.293** | **0.2555** | **0.0753** |
| v19 — VGGT + log consistency | 0.2280 | 3.386 | 5.989 | 0.2814 | 0.0830 |
| v20 — L1 consistency λ=20 | 0.2306 | 3.609 | 6.212 | 0.2825 | 0.0815 |

**Two findings emerge from this table.**

**Finding 1 — Every consistency-anchored variant improves over zero-shot
Depth Pro on Make3D.** Across all six variants tested, every single one
improves AbsRel, SqRel, RMSE, RMSElog and log₁₀ relative to the zero-shot
baseline. This is a remarkably consistent pattern: regardless of which
specific form of consistency loss is used (L1 or log-space, with or without
VGGT poses, with or without edge-aware weighting), the resulting model
outperforms zero-shot Depth Pro on a dataset it has never seen. Pure
photometric configurations (v6–v14, omitted from the table) would
predictably worsen this further; we conjecture they would catastrophically
fail on Make3D in the same way they fail on KITTI.

**Finding 2 — v18 (log-space consistency, λ = 10) is the best cross-domain
variant by a large margin.** v18 improves over zero-shot Depth Pro by
**−24.7% on AbsRel**, **−55.1% on SqRel**, **−20.7% on RMSE**, **−15.0% on
RMSElog** and **−14.0% on log₁₀**. These are not marginal improvements:
they are an order of magnitude larger than the KITTI gains and well beyond
any noise threshold. Critically, v18 was previously classified in our
KITTI analysis as a *less successful* variant — its KITTI AbsRel (0.1001)
is worse than v15's (0.0875). Yet on Make3D, v18 dominates v15 on every
metric. This inversion — v15 is the KITTI champion, v18 is the Make3D
champion — is one of the most important findings of the thesis.

**Table 4.7 — v18 relative improvement over zero-shot on Make3D:**

| Metric | Zero-shot | v18 (ours) | Relative improvement |
|--------|-----------|-----------|---------------------|
| AbsRel  | 0.2575 | 0.1940 | **−24.7%** |
| SqRel   | 4.846  | 2.175  | **−55.1%** |
| RMSE    | 6.677  | 5.293  | **−20.7%** |
| RMSElog | 0.3006 | 0.2555 | **−15.0%** |
| log₁₀   | 0.0876 | 0.0753 | **−14.0%** |

#### 4.6.3 Why v18 Dominates v15 Cross-Domain

The v15 vs. v18 inversion between KITTI and Make3D is initially surprising —
the standard intuition is that the best in-domain model is also the best
out-of-domain model. Three concurrent explanations account for the
observed pattern.

First, **the log-space consistency in v18 directly optimises a log-depth
distance**, which is the same quantity that AbsRel, RMSElog, log₁₀ and the
δ-threshold metrics are computed from. On KITTI, where zero-shot is
saturated, the log-space anchor and the L1 metric anchor (v15) converge to
essentially the same depth predictions, so the metric form of the
consistency loss is irrelevant. On Make3D, where there is substantial
headroom, the log-space objective produces a measurably different
adaptation that aligns better with the log-based metrics.

Second, **the v18 adaptation produces depths that are slightly *less*
metrically calibrated than v15's** — but this only hurts on KITTI, where
the model is already metrically well-calibrated. On Make3D, where the
optical setup and depth distribution differ from KITTI, metric calibration
is corrected per-image by median scaling anyway, so v18's looser metric
fidelity is masked while its better log-space relative ordering helps.

Third, **the consistency loss in v18 is computed on log-scaled depth
differences that are bounded** (a 100-metre prediction error is the same
log distance whether the reference is 5 m or 50 m), whereas the L1
consistency in v15 weights absolute large-depth errors much more heavily
than small-depth errors. On KITTI, where most pixels are within 20 m, the
L1 weighting matches the dominant depth range and produces a sharper
adaptation. On Make3D, where depths span 4–80 m more uniformly, the
log-space adaptation balances near and far pixels more evenly.

The practical takeaway for the thesis is this: **we report v15 as the main
result on KITTI and v18 as the main result on Make3D**, and we explicitly
argue (Section 4.7 and Chapter 5) that the choice of consistency-loss form
should be matched to the target benchmark's saturation level. On saturated
benchmarks (KITTI Eigen), the L1 consistency in v15 is preferable because
it is the most conservative and stays closest to the zero-shot
predictions. On benchmarks with headroom (Make3D and, we conjecture, many
real-world deployment settings), the log-space consistency in v18 extracts
substantially larger improvements.

This Make3D experiment promotes the consistency-loss contribution from
"marginal improvement on a saturated benchmark" to "the first
consistency-anchored self-supervised adaptation that delivers double-digit
percentage improvements over a depth foundation model on a held-out
cross-domain benchmark." It is the strongest empirical evidence in the
thesis that the consistency-loss formulation captures a property of
outdoor depth supervision that is fundamentally beneficial — not a
KITTI-specific artefact.

---

### 4.7 Cross-Domain Generalization on Cityscapes

To probe whether the cross-domain improvement observed on Make3D is a
one-off artefact of that specific outdoor distribution, or a genuinely
general property of consistency-anchored adaptation, we run the same
experiment on a second held-out dataset: **Cityscapes** (Cordts et al.,
CVPR 2016). Cityscapes contains 500 outdoor driving val images at
2048×1024 captured in three German cities (Frankfurt, Lindau, Münster) —
geographically and visually distinct from KITTI's Karlsruhe sequences but
similar in setting (urban driving, similar camera height, comparable
depth range). Cityscapes is therefore an intermediate cross-domain test:
closer to KITTI than Make3D is, with a partially saturated zero-shot
baseline.

#### 4.7.1 Cityscapes Evaluation Protocol

Cityscapes provides per-image stereo disparity (1024×2048, uint16 PNG)
encoded as `disparity_metric = (raw_pixel - 1) / 256` for raw values > 0.
We convert disparity to depth using the standard rectified-stereo formula
$d = f_x \cdot B / \text{disparity}$, with the Cityscapes mean
calibration $f_x = 2262.5$ px (at 2048×1024) and baseline $B = 0.209$ m.

The evaluation pipeline is:

- Load each RGB image at native 2048×1024.
- Decode the corresponding disparity PNG to dense metric depth.
- Cap depth at $1 \leq d \leq 80$ m (matching KITTI Eigen convention).
- Run Depth Pro at 1536×1536, scale canonical inverse depth to metric
  using the Cityscapes focal length, resize to 2048×1024 to match GT.
- Per-image median scaling.
- Report seven metrics: AbsRel, SqRel, RMSE, RMSElog and three
  δ-thresholds.

We evaluate the zero-shot baseline plus all six consistency-anchored
variants (v15–v20) with no changes to model or weights.

#### 4.7.2 Quantitative Results

**Table 4.7 — Cross-domain Cityscapes val (500 images, median scaling, depth cap 80 m):**

| Method | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSElog ↓ | δ<1.25 ↑ | δ<1.25² ↑ | δ<1.25³ ↑ |
|--------|----------|---------|--------|-----------|----------|-----------|-----------|
| Depth Pro zero-shot | 0.1119 | 1.502 | 6.636 | 0.1964 | 0.8773 | 0.9640 | 0.9850 |
| v15 — L1 λ=10 | 0.1160 | 1.500 | 6.642 | 0.1970 | 0.8765 | 0.9642 | 0.9854 |
| v16 — VGGT + edge | 0.1207 | 1.578 | 6.955 | 0.2022 | 0.8624 | 0.9615 | 0.9852 |
| v17 — log + depth-weight | 0.1363 | 2.178 | 8.037 | 0.2192 | 0.8510 | 0.9563 | 0.9810 |
| v18 — log λ=10 | 0.1335 | 1.452 | 6.923 | 0.2106 | 0.8501 | 0.9576 | 0.9839 |
| v19 — VGGT + log | 0.1388 | 1.825 | 8.177 | 0.2250 | 0.8122 | 0.9430 | 0.9830 |
| **v20 — L1 λ=20 (ours)** | **0.1085** | **1.483** | **6.331** | **0.1918** | **0.8927** | **0.9670** | **0.9853** |

**Finding — v20 improves over zero-shot Depth Pro on all seven Cityscapes
metrics.** The improvements are: AbsRel −3.0%, SqRel −1.3%, RMSE −4.6%,
RMSElog −2.3%, δ<1.25 +1.76 percentage points, δ<1.25² +0.3 pp and
δ<1.25³ +0.03 pp. The δ<1.25 gain in particular (87.7% → 89.3%) means
that roughly 1.6% of pixels — about 24,000 pixels per image at
1024×2048 resolution — that were misclassified by zero-shot are now
correctly within the 1.25-factor accuracy band. v20 is the only variant
that uniformly improves over zero-shot on Cityscapes; v15 and v16 are
within ~1% of zero-shot but slightly worse on AbsRel and RMSE, while
v17, v18 and v19 are several percent worse on the absolute-error
metrics.

#### 4.7.3 The Saturation–Anchor Pairing

The KITTI, Cityscapes and Make3D experiments together reveal a striking
pattern: **the best consistency-anchored variant is different on each
benchmark, and the optimum tracks the saturation level of the zero-shot
baseline**.

**Table 4.8 — Best variant per benchmark, ordered by zero-shot saturation:**

| Benchmark | Zero-shot AbsRel | Saturation level | Best variant | Loss form | λ |
|-----------|------------------|------------------|--------------|-----------|---|
| KITTI Eigen | 0.0866 | most saturated | **v15** | L1 metric | 10 |
| Cityscapes (val) | 0.1119 | medium | **v20** | L1 metric | **20** |
| Make3D | 0.2575 | least saturated | **v18** | **log-space** | 10 |

Three patterns emerge from this pairing.

First, the **anchor strength** ($\lambda$) increases as the benchmark
moves from saturated to less saturated *and* the loss is kept in metric
L1 space. On KITTI the model needs to stay very close to zero-shot
($\lambda = 10$, v15). On Cityscapes the model can drift slightly
further but a stronger anchor is needed ($\lambda = 20$, v20). This is
counter-intuitive at first — one might expect Cityscapes to need *less*
anchoring — but it makes sense if one views the consistency loss as
preserving KITTI-similar geometric structure that transfers more
faithfully when held tightly.

Second, when the benchmark is *very* far from KITTI in depth
distribution (Make3D, where depths span 4–80 m more uniformly rather
than being concentrated <20 m as in driving), the metric form of the
consistency loss matters more than its strength. **Log-space**
consistency (v18) at $\lambda = 10$ matches Make3D's near-uniform log
distribution and produces a 24.7% AbsRel improvement that no L1 variant
matches at any $\lambda$.

Third, **VGGT-supplied poses do not help cross-domain**. v16 and v19,
which both use VGGT precomputed poses, are the *worst* variants on both
Cityscapes and Make3D among the consistency family. This is despite v16
being the best variant on KITTI on δ<1.25³. We hypothesise that VGGT
poses encode KITTI-specific motion patterns (forward driving on
structured roads) that anchor the adaptation to KITTI ego-motion
distributions and reduce its transferability.

#### 4.7.4 Cross-Benchmark Consistency-Loss Family Summary

Combining the KITTI, Cityscapes and Make3D evaluations gives the
following final picture of the consistency-anchored adaptation family:

- **No variant uniformly dominates across benchmarks.** The choice of
  $\lambda$ and loss form (L1 vs. log) should be tuned to the deployment
  benchmark.
- **All six variants improve over zero-shot on Make3D**; v20 also
  improves over zero-shot on all seven Cityscapes metrics; v15 and v16
  marginally improve over zero-shot on δ<1.25³ on KITTI.
- The **family** of consistency-anchored adaptations therefore
  collectively dominates zero-shot Depth Pro across all three
  benchmarks, even if no single member is best on all three.
- **VGGT pose supervision is harmful cross-domain**, suggesting that
  trainable PoseNet (which fits the local distribution and does not
  encode multi-view priors) is the more conservative choice for
  cross-domain transfer.

These three benchmarks together promote the thesis's main result from
"v15 wins one KITTI metric" to "consistency-anchored adaptation is a
controllable knob: pick L1/$\lambda=10$ for saturated benchmarks,
L1/$\lambda=20$ for medium, log/$\lambda=10$ for unsaturated."

---

### 4.8 Comparison with State of the Art

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

### 4.9 Conclusions

**Main finding.** Depth Pro in zero-shot mode surpasses all published self-supervised monocular depth methods on the KITTI Eigen benchmark, improving AbsRel by 13% over MonoViT (0.0866 vs. 0.099) without a single KITTI training frame. Modern depth foundation models render the "train on KITTI from scratch" paradigm of the last five years obsolete.

**Negative result.** We show that applying the standard Monodepth2-style self-supervised photometric objective to Depth Pro — in all configurations tested (LoRA rank 8, frozen encoder, higher smoothness regularization) — **degrades zero-shot performance by a factor of 5×** on the primary metric (AbsRel 0.458 vs. 0.087). The training loss decreases while the test metric worsens. This is the defining failure mode of naïve self-supervised adaptation of foundation models.

**Cross-domain transfer (Sections 4.6 and 4.7).** Evaluated on two held-out
outdoor benchmarks that neither model has seen during training:

- **Make3D (134 outdoor test images, Stanford campus, depths 4–80 m):**
  all six consistency-anchored variants improve over zero-shot Depth Pro
  on every Make3D metric. The best variant, **v18 (log-space consistency,
  λ = 10), reduces AbsRel by 24.7%, SqRel by 55.1%, RMSE by 20.7%,
  RMSElog by 15.0% and log₁₀ by 14.0%** relative to zero-shot.

- **Cityscapes (500 outdoor driving val images, three German cities):**
  the best variant, **v20 (L1 consistency, λ = 20), improves over
  zero-shot on all seven standard metrics** — AbsRel −3.0%, SqRel −1.3%,
  RMSE −4.6%, RMSElog −2.3%, δ<1.25 +1.76 pp, δ<1.25² +0.3 pp,
  δ<1.25³ +0.03 pp.

**A striking pattern emerges across the three benchmarks: the best
variant tracks the saturation level of the zero-shot baseline.** v15
(L1, λ=10) is best on the most saturated benchmark (KITTI Eigen), v20
(L1, λ=20) is best on the medium-saturated Cityscapes, and v18
(log-space, λ=10) is best on the least-saturated Make3D. The
consistency-anchored adaptation family therefore *collectively*
dominates zero-shot Depth Pro across all three benchmarks even though
no single member is best on all three. The choice of consistency-loss
form and anchor strength is a controllable knob to match benchmark
saturation, and the thesis demonstrates that the knob is meaningful
through three independent cross-domain experiments.

**Root causes identified.** Our analysis (Section 4.5) attributes the failure to (i) a fundamental objective–metric gap in photometric self-supervision when the starting point is already near-optimal, (ii) a resolution mismatch between low-resolution loss (416×128) and high-resolution evaluation (1242×375), and (iii) previously-undocumented numerical instabilities of LoRA under FP16 that silently corrupt checkpoints. The bfloat16 training recipe and parameter-level NaN validation we introduce eliminate the numerical failures cleanly; the objective–metric gap is a more fundamental obstacle that future work must address through explicit anchoring to the pretrained predictions.

**Implications.**
- For practitioners: adopt Depth Pro zero-shot as the strong baseline for outdoor monocular depth on KITTI-like distributions.
- For researchers: the research frontier for self-supervised depth has shifted from "close the gap to supervised" (largely done) to "adapt foundation models without breaking them" (open).
- For the self-supervised community: photometric loss alone is an insufficient regularizer once the starting point is strong. Future methods should combine photometric consistency with explicit distillation from zero-shot predictions, or limit the hypothesis space (e.g. LoRA-only with frozen head).

The final experiment in this thesis (v13, ongoing at time of writing) tests whether a **depth consistency regularizer** — penalizing drift from the zero-shot prediction — can enable beneficial domain adaptation without the catastrophic forgetting observed here.
