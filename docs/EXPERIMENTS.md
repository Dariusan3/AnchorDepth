# Depth Pro — Experiment Log

Tracking all modifications and their impact on NYU Depth V2 (Eigen 654 test split).

**GPU:** NVIDIA RTX 4070 Ti 12GB | **Model:** 952M params | **Dataset:** NYU Depth V2 (795 train / 654 test)

| # | Experiment | Date | Description | AbsRel | RMSE | delta<1.25 | vs Baseline |
|---|-----------|------|-------------|--------|------|------------|-------------|
| 0 | [v0_baseline](../experiments/v0_baseline/README.md) | 2026-03-21 | Apple pretrained (zero-shot, no fine-tuning) | 0.1155 | 0.5092 | 0.8777 | baseline |
| 1 | [v1_finetuned_decoder](../experiments/v1_finetuned_decoder/README.md) | 2026-03-21 | Fine-tune decoder+head (20M params), frozen encoder, 25 epochs | 0.0855 | 0.3288 | 0.9389 | +26.0% |
