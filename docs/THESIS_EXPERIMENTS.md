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

**Results — Pretrained Depth Pro (zero-shot on KITTI):**

| AbsRel | SqRel | RMSE | RMSElog | δ<1.25 | δ<1.25² | δ<1.25³ |
|--------|-------|------|---------|--------|---------|---------|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD |

*(Evaluation running — results will be filled in)*

**Analysis:** *(to be written after results)*

Depth Pro was trained primarily on indoor and mixed internet imagery. KITTI outdoor driving scenes with depth ranges up to 80m are a challenging distribution shift. We expect the pretrained model to show weaker performance on AbsRel and RMSE compared to its indoor benchmarks, but with good δ<1.25 (structural correctness) due to the strong DINOv2 representations.

---

### 4.4 Main Result: Self-Supervised LoRA Fine-Tuning with Multi-Scale Loss

We train Depth Pro with LoRA using multi-scale photometric self-supervision for 40 epochs on the KITTI Eigen training split (v6 run). This is the main experiment of the thesis.

**WandB run:** `v6-multiscale-40ep` in project `depth-pro-selfsup` — live training curves available.

#### 4.4.1 Quantitative Results

**Results — Full ablation table:**

| Method | Epochs | Multi-scale | LoRA | AbsRel | SqRel | RMSE | RMSElog | δ<1.25 | δ<1.25² | δ<1.25³ |
|--------|--------|-------------|------|--------|-------|------|---------|--------|---------|---------|
| Monodepth2 (ref) | — | — | — | 0.115 | 0.903 | 4.863 | 0.193 | 0.877 | 0.959 | 0.981 |
| MonoViT (ref) | — | — | — | 0.099 | 0.708 | 4.372 | 0.175 | 0.900 | 0.965 | 0.983 |
| Depth Pro zero-shot | — | — | — | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| v5 — single-scale baseline | 20 | No | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| **v6 — full method (ours)** | **40** | **Yes** | **Yes** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| v7 — no LoRA | 20 | Yes | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| v8 — higher smoothness | 20 | Yes | Yes | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

*(Training started April 13, 2026. Results to be filled after each run completes.)*

#### 4.4.2 Training Dynamics

Figure 4.1 shows the photometric validation loss over 20 training epochs.

*(Plot to be generated after training — see `generate_results.py`)*

Key observations from training:

- **Epochs 1–3 (warmup):** Learning rate ramps linearly from 0 to target. Auto-masking disabled. Model learns basic depth structure.
- **Epochs 4–20:** Full learning rate, cosine annealing. Photometric loss expected to decrease steadily.
- **Best checkpoint:** Saved at the epoch with lowest validation photometric loss.

#### 4.4.3 Qualitative Results

*(Depth map visualizations to be generated after training — see `generate_results.py`)*

We show qualitative depth predictions on representative KITTI Eigen test images comparing:
1. Pretrained Depth Pro (zero-shot)
2. Our self-supervised fine-tuned model
3. Ground truth LiDAR depth (sparse)

---

### 4.5 Analysis

#### 4.5.1 Effect of LoRA vs. Frozen Encoder

*(To be added if ablation is run)*

#### 4.5.2 Training Stability

We encountered and resolved two training stability issues during development:

**Issue 1 — NaN loss from FOV head (v2 training run):**
The FOV head predicted unstable field-of-view values on KITTI distribution, causing NaN in the canonical depth scaling and silently disabling all gradient updates for 8 epochs.
*Fix:* Use ground truth focal length from KITTI calibration instead of FOV head prediction.

**Issue 2 — Loss plateau (v3 training run):**
Training converged to a degenerate solution at photometric loss ~0.297 — higher than the pretrained model's reconstruction quality — for all 8 epochs observed.
*Fix:* Same as above. Correct focal length scaling was the critical missing piece.

These issues highlight a non-obvious challenge in adapting metric depth foundation models to self-supervised training: the canonical depth representation, designed for zero-shot generalization, requires careful handling when the geometric consistency of photometric warping is critical.

---

### 4.6 Comparison with State of the Art

*(Full comparison table to be completed after results)*

| Method | Encoder | Params | Labels | AbsRel | δ<1.25 |
|--------|---------|--------|--------|--------|--------|
| Monodepth2 | ResNet-18 | 14.8M trainable | None | 0.115 | 0.877 |
| DIFFNet | HRNet-18 | 65M trainable | None | 0.102 | 0.896 |
| MonoViT | ViT-Small | 22M trainable | None | 0.099 | 0.900 |
| **Ours** | **ViT-Large×2 + LoRA** | **34.33M trainable** | **None** | **TBD** | **TBD** |

**Key observation:** Our method uses a 952M parameter foundation model but updates only 34.33M parameters — fewer than DIFFNet's 65M — while leveraging far stronger pretrained representations (DINOv2-Large vs. HRNet-18).
