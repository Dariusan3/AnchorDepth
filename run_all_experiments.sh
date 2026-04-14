#!/bin/bash
# Run all 4 ablation experiments sequentially.
# Each run saves checkpoints to its own folder and logs to WandB.
#
# Usage:
#   bash run_all_experiments.sh
#
# Estimated total time: ~13 days (stride=6, RTX 4070 Ti)
# v6: ~5.6 days (40 epochs)
# v7: ~2.4 days (20 epochs)
# v8: ~2.4 days (20 epochs)

set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate depth-pro

DATA="datasets/kitti_raw"
LOG_DIR="checkpoints"

echo "============================================================"
echo "Starting ablation study — $(date)"
echo "============================================================"

# ============================================================
# v6: Full method — multi-scale loss, LoRA, 40 epochs
# ============================================================
echo ""
echo ">>> [v6] Multi-scale, LoRA, 40 epochs — $(date)"
python train_kitti_selfsup_ms.py \
    --data-path $DATA \
    --epochs 40 \
    --stride 6 \
    --save-dir checkpoints/selfsup_v6 \
    --wandb-name "v6-multiscale-40ep" \
    2>&1 | tee $LOG_DIR/selfsup_training_v6.log
echo ">>> [v6] DONE — $(date)"

# ============================================================
# v7: Ablation — no LoRA (decoder only), multi-scale, 20 epochs
# ============================================================
echo ""
echo ">>> [v7] No-LoRA ablation, 20 epochs — $(date)"
python train_kitti_selfsup_ms.py \
    --data-path $DATA \
    --epochs 20 \
    --stride 6 \
    --no-lora \
    --save-dir checkpoints/selfsup_v7 \
    --wandb-name "v7-no-lora-20ep" \
    2>&1 | tee $LOG_DIR/selfsup_training_v7.log
echo ">>> [v7] DONE — $(date)"

# ============================================================
# v8: Ablation — higher smoothness weight (1e-2), 20 epochs
# ============================================================
echo ""
echo ">>> [v8] Higher smoothness (1e-2), 20 epochs — $(date)"
python train_kitti_selfsup_ms.py \
    --data-path $DATA \
    --epochs 20 \
    --stride 6 \
    --smoothness-weight 1e-2 \
    --save-dir checkpoints/selfsup_v8 \
    --wandb-name "v8-smooth-1e2-20ep" \
    2>&1 | tee $LOG_DIR/selfsup_training_v8.log
echo ">>> [v8] DONE — $(date)"

echo ""
echo "============================================================"
echo "All experiments done — $(date)"
echo "Run: python evaluate_kitti.py to get final metrics"
echo "Run: python generate_results.py to generate thesis figures"
echo "============================================================"
