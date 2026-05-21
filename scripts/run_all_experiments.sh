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
# v13: LoRA-only, decoder + head FROZEN — tries to beat zero-shot
# Only 2.36M LoRA params train; decoder/head keep zero-shot weights
# This is the most promising configuration for beating zero-shot
# ============================================================
echo ""
echo ">>> [v13] LoRA-only, frozen head+decoder, 10 epochs — $(date)"
python train_kitti_selfsup_ms.py \
    --data-path $DATA \
    --epochs 10 \
    --stride 6 \
    --lora-rank 8 \
    --lora-alpha 8.0 \
    --lr-lora 1e-5 \
    --freeze-head \
    --freeze-decoder \
    --save-dir checkpoints/selfsup_v13 \
    --wandb-name "v13-lora-only-frozen-head" \
    2>&1 | tee $LOG_DIR/selfsup_training_v13.log
echo ">>> [v13] DONE — $(date)"
echo ">>> [v13] Evaluating..."
python evaluate_kitti.py \
    --checkpoint checkpoints/selfsup_v13/selfsup_best.pt \
    --lora-rank 8 --lora-alpha 8.0 \
    --output results/eval_v13_lora_only.json \
    --wandb-name "v13-lora-only-eval" \
    2>&1 | tee $LOG_DIR/eval_v13.log

# ============================================================
# v14: Gentle fine-tuning — LoRA + decoder/head with very low LR
# LR 100x smaller than v10, only 5 epochs, early stopping
# ============================================================
echo ""
echo ">>> [v14] Gentle fine-tuning (lr=1e-6), 5 epochs — $(date)"
python train_kitti_selfsup_ms.py \
    --data-path $DATA \
    --epochs 5 \
    --stride 6 \
    --lora-rank 8 \
    --lora-alpha 8.0 \
    --lr-depth 1e-6 \
    --lr-lora 1e-7 \
    --lr-pose 1e-5 \
    --save-dir checkpoints/selfsup_v14 \
    --wandb-name "v14-gentle-lowlr" \
    2>&1 | tee $LOG_DIR/selfsup_training_v14.log
echo ">>> [v14] DONE — $(date)"
echo ">>> [v14] Evaluating..."
python evaluate_kitti.py \
    --checkpoint checkpoints/selfsup_v14/selfsup_best.pt \
    --lora-rank 8 --lora-alpha 8.0 \
    --output results/eval_v14_gentle.json \
    --wandb-name "v14-gentle-eval" \
    2>&1 | tee $LOG_DIR/eval_v14.log

echo ""
echo "============================================================"
echo "All experiments done — $(date)"
echo "============================================================"
