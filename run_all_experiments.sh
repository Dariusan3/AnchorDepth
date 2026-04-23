#!/bin/bash
# Run ablation experiments sequentially (with NaN-safe LoRA).
#
# Order: v10 (main LoRA) → v11 (no-LoRA) → v12 (high smoothness)
# All start from scratch (no resume) with fixed FP32 LoRA.
#
# Usage:
#   nohup bash run_all_experiments.sh > checkpoints/all_experiments.log 2>&1 &

set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate depth-pro

DATA="datasets/kitti_raw"
LOG_DIR="checkpoints"

echo "============================================================"
echo "Starting experiments (NaN-safe LoRA, FP32) — $(date)"
echo "============================================================"

# ============================================================
# v10: Main method — LoRA rank=8, 20 epochs, no-LoRA-rank-reduction
# No resume — start fresh with FP32 LoRA fix
# ============================================================
echo ""
echo ">>> [v10] LoRA r=8 (FP32 fix), 20 epochs — $(date)"
python train_kitti_selfsup_ms.py \
    --data-path $DATA \
    --epochs 20 \
    --stride 6 \
    --lora-rank 8 \
    --lora-alpha 8.0 \
    --save-dir checkpoints/selfsup_v10 \
    --wandb-name "v10-lora-r8-fp32fix" \
    2>&1 | tee $LOG_DIR/selfsup_training_v10.log
echo ">>> [v10] DONE — $(date)"
echo ">>> [v10] Evaluating..."
python evaluate_kitti.py \
    --checkpoint checkpoints/selfsup_v10/selfsup_best.pt \
    --lora-rank 8 --lora-alpha 8.0 \
    --output results/eval_v10_lora.json \
    --wandb-name "v10-lora-r8-eval" \
    2>&1 | tee $LOG_DIR/eval_v10.log

# ============================================================
# v11: Ablation — no LoRA (decoder only), 20 epochs
# ============================================================
echo ""
echo ">>> [v11] No-LoRA ablation, 20 epochs — $(date)"
python train_kitti_selfsup_ms.py \
    --data-path $DATA \
    --epochs 20 \
    --stride 6 \
    --no-lora \
    --save-dir checkpoints/selfsup_v11 \
    --wandb-name "v11-no-lora-20ep" \
    2>&1 | tee $LOG_DIR/selfsup_training_v11.log
echo ">>> [v11] DONE — $(date)"
echo ">>> [v11] Evaluating..."
python evaluate_kitti.py \
    --checkpoint checkpoints/selfsup_v11/selfsup_best.pt \
    --no-lora \
    --output results/eval_v11_no_lora.json \
    --wandb-name "v11-no-lora-eval" \
    2>&1 | tee $LOG_DIR/eval_v11.log

# ============================================================
# v12: Ablation — higher smoothness (1e-2), LoRA r=8
# ============================================================
echo ""
echo ">>> [v12] Higher smoothness (1e-2), LoRA r=8, 20 epochs — $(date)"
python train_kitti_selfsup_ms.py \
    --data-path $DATA \
    --epochs 20 \
    --stride 6 \
    --smoothness-weight 1e-2 \
    --lora-rank 8 \
    --lora-alpha 8.0 \
    --save-dir checkpoints/selfsup_v12 \
    --wandb-name "v12-smooth-1e2-lora" \
    2>&1 | tee $LOG_DIR/selfsup_training_v12.log
echo ">>> [v12] DONE — $(date)"
echo ">>> [v12] Evaluating..."
python evaluate_kitti.py \
    --checkpoint checkpoints/selfsup_v12/selfsup_best.pt \
    --lora-rank 8 --lora-alpha 8.0 \
    --output results/eval_v12_smooth.json \
    --wandb-name "v12-smooth-eval" \
    2>&1 | tee $LOG_DIR/eval_v12.log

echo ""
echo "============================================================"
echo "All experiments done — $(date)"
echo "============================================================"
