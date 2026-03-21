# Experiment: v1_finetuned_decoder

**Date:** 2026-03-21
**Description:** Fine-tune decoder + head (20.1M / 952M params) on NYU train set, frozen DINOv2 encoder

## Configuration

```
Train args: --epochs 25 --lr 1e-4 --batch-size 1 --grad-accum 4
Trainable params: 20.1M (2.1% of model)
Frozen: encoder (627M) + FOV head (304M)
Loss: Scale-invariant log + gradient matching (weight 0.5)
Optimizer: AdamW (lr=1e-4, weight_decay=1e-4, cosine annealing)
Precision: FP16 mixed precision
VRAM: 8.67 GB / 12 GB
Training time: 2.9 hours (25 epochs × 7 min/epoch)
```

## Results — Metric Depth (NYU Depth V2, Eigen 654 test)

| Metric | Baseline (pretrained) | This Experiment | Change |
|--------|----------------------|-----------------|--------|
| AbsRel (↓) | 0.1155 | 0.0855 | +26.0% |
| SqRel (↓) | 0.0987 | 0.0414 | +58.0% |
| RMSE (↓) | 0.5092 | 0.3288 | +35.4% |
| RMSElog (↓) | 0.1905 | 0.1117 | +41.4% |
| delta<1.25 (↑) | 0.8777 | 0.9389 | +7.0% |
| delta<1.25² (↑) | 0.9596 | 0.9904 | +3.2% |
| delta<1.25³ (↑) | 0.9795 | 0.9975 | +1.8% |

## Results — Scale-Invariant (aligned)

| Metric | Value |
|--------|-------|
| AbsRel | 0.0565 |
| RMSE | 0.2348 |
| delta<1.25 | 0.9653 |

## Training Curve

Loss progression: 0.024 → 0.012 over 25 epochs (steady convergence, no overfitting).

Best validation AbsRel: 0.0894 (epoch 25)

## Key Findings

- Even fine-tuning only 2.1% of params gives massive improvements (+26% AbsRel)
- The frozen DINOv2 encoder features are already very good — the decoder was the bottleneck
- Scale-invariant results also improved significantly, showing better depth structure (not just scale)
- No overfitting observed — could potentially train longer

## Visual Results

See `results/comparison/` for side-by-side depth map comparisons.
