# Experiment: v0_baseline

**Date:** 2026-03-21
**Description:** Apple's pretrained Depth Pro model evaluated zero-shot (no fine-tuning)

## Configuration

```
Model: depth_pro.pt (Apple pretrained, 952M params)
Fine-tuning: None (zero-shot evaluation)
```

## Results — Metric Depth (NYU Depth V2, Eigen 654 test)

| Metric | Value |
|--------|-------|
| AbsRel (↓) | 0.1155 |
| SqRel (↓) | 0.0987 |
| RMSE (↓) | 0.5092 |
| RMSElog (↓) | 0.1905 |
| delta<1.25 (↑) | 0.8777 |
| delta<1.25² (↑) | 0.9596 |
| delta<1.25³ (↑) | 0.9795 |

## Results — Scale-Invariant (aligned)

| Metric | Value |
|--------|-------|
| AbsRel | 0.0834 |
| RMSE | 0.3631 |
| delta<1.25 | 0.9195 |

## Notes

- This is the reference baseline for all improvements
- The gap between metric and scale-invariant results shows scale estimation is a key bottleneck
- Inference time: 0.737s per image on RTX 4070 Ti
