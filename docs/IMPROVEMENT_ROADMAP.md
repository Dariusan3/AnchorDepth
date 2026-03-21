# Depth Pro — Improvement Roadmap

Research plan to push Depth Pro toward state-of-the-art on NYU Depth V2.

**Current best:** AbsRel 0.0855 (v1, fine-tuned decoder)
**Paper (Apple):** AbsRel 0.036
**Target:** Close the gap to paper and explore novel improvements

---

## Priority 1 — Quick Wins (no retraining needed)

### v2: Test-Time Augmentation (TTA)
**Expected impact:** 5-10% improvement
**Effort:** Low (inference-only modification)

Multi-scale and flip augmentation at inference time:
- Horizontal flip: average predictions from original + flipped image
- Multi-scale: run at 3 scales (0.75x, 1.0x, 1.25x), average
- Combines both for best results

**File:** `src/depth_pro/improvements/tta.py`

---

### v3: Sliding Window Ensemble
**Expected impact:** 3-5% improvement on boundary metrics
**Effort:** Low-Medium

Instead of resizing to 1536x1536, use overlapping sliding windows at native resolution:
- Process overlapping 1536x1536 crops
- Blend predictions in overlap regions
- Preserves fine details lost in downsampling

**File:** `src/depth_pro/improvements/sliding_window.py`

---

## Priority 2 — Training Improvements

### v4: Unfreeze Encoder Layers
**Expected impact:** 10-15% improvement
**Effort:** Medium (requires gradient checkpointing for VRAM)

Progressive unfreezing:
- Start: freeze all encoder, train decoder 10 epochs
- Phase 2: unfreeze last 4 transformer blocks, train 15 epochs at lower LR
- Phase 3: unfreeze last 8 blocks, train 10 epochs at even lower LR
- Uses gradient checkpointing to fit in 12GB VRAM

**File:** `scripts/train_nyu.py --unfreeze-encoder-layers 4`

---

### v5: Advanced Loss Functions
**Expected impact:** 5-10% improvement
**Effort:** Medium

Combine multiple losses:
1. **Scale-invariant log loss** (Eigen et al.) — already implemented
2. **Gradient matching loss** — already implemented
3. **SSIM loss** — structural similarity on depth
4. **Ranking loss** — enforce ordinal depth relationships
5. **Virtual normal loss** (VNL) — enforce geometric consistency via surface normals estimated from depth

**File:** `src/depth_pro/improvements/losses.py`

---

### v6: Data Augmentation
**Expected impact:** 5-8% improvement
**Effort:** Low-Medium

Stronger training augmentations:
- CutDepth: randomly zero out rectangular depth regions during training
- Color jitter (brightness, contrast, saturation, hue)
- Random erasing
- Mixup on depth maps
- Photometric distortion

**File:** `src/depth_pro/improvements/augmentations.py`

---

## Priority 3 — Architecture Modifications

### v7: Attention-Based Decoder
**Expected impact:** 10-20% improvement
**Effort:** High

Replace CNN decoder with transformer-based decoder:
- Cross-attention between encoder features at different scales
- Self-attention within decoder features
- Inspired by PixelFormer / NewCRFs architectures
- Learnable query tokens for depth prediction

**File:** `src/depth_pro/improvements/attention_decoder.py`

---

### v8: Multi-Scale Feature Fusion with ASPP
**Expected impact:** 5-10% improvement
**Effort:** Medium

Add Atrous Spatial Pyramid Pooling between encoder and decoder:
- Captures multi-scale context
- Parallel dilated convolutions at rates [6, 12, 18]
- Global average pooling branch
- Proven effective in dense prediction (DeepLab)

**File:** `src/depth_pro/improvements/aspp.py`

---

### v9: Edge-Aware Refinement Module
**Expected impact:** 5-8% on boundary F1
**Effort:** Medium

Post-decoder refinement focused on depth boundaries:
- Edge detection on RGB → guide depth refinement
- Bilateral filtering with learned parameters
- Boundary-aware upsampling instead of bilinear
- Guided by Canny/Sobel edges from input image

**File:** `src/depth_pro/improvements/edge_refine.py`

---

### v10: Depth Bins Classification (AdaBins-style)
**Expected impact:** 10-15% improvement
**Effort:** High

Instead of direct regression, predict depth as classification:
- Adaptive bin boundaries learned per-image
- Mini-ViT to predict bin edges from global features
- Linear combination of bin centers for final depth
- Reduces the impact of outliers in training

**File:** `src/depth_pro/improvements/adabins_head.py`

---

## Priority 4 — Training Data

### v11: Extended Training Data
**Expected impact:** 15-25% improvement
**Effort:** High (data collection/processing)

Fine-tune on multiple datasets simultaneously:
- NYU Depth V2 (indoor, Kinect)
- KITTI (outdoor, LiDAR)
- ScanNet (indoor, structured light)
- DIODE (indoor+outdoor, LiDAR)
- Hypersim (synthetic, perfect GT)

Multi-dataset training with dataset-specific scale heads.

---

### v12: Self-Training / Pseudo-Labels
**Expected impact:** 5-15% improvement
**Effort:** Medium-High

1. Run current best model on large unlabeled dataset (e.g., ImageNet, COCO)
2. Filter by confidence (low variance under TTA)
3. Fine-tune on pseudo-labeled + real data
4. Iterate

---

## Suggested Experiment Order

```
v2 (TTA)           → quick win, no training, ~1 hour
v4 (unfreeze)      → biggest bang, ~4 hours training
v5 (losses)        → combine with v4, ~4 hours
v6 (augmentation)  → combine with v4+v5, ~4 hours
v7 (attn decoder)  → major architecture change, ~6 hours
v3 (sliding window)→ inference improvement, ~2 hours
```

Each experiment builds on the previous best and is tracked in `docs/EXPERIMENTS.md`.
