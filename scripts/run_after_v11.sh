#!/bin/bash
# Wait for v11 (PID 184249) to finish, then run v11 eval + v13 + v14
# This script handles the queue after we kill the original shell script.

set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate depth-pro

DATA="datasets/kitti_raw"
LOG_DIR="checkpoints"
V11_PID=184249

echo "Waiting for v11 (PID $V11_PID) to finish..."
while ps -p $V11_PID > /dev/null 2>&1; do
    sleep 60
done
echo ">>> v11 finished at $(date)"

# v11 eval
echo ">>> [v11] Evaluating..."
python evaluate_kitti.py \
    --checkpoint checkpoints/selfsup_v11/selfsup_best.pt \
    --no-lora \
    --output results/eval_v11_no_lora.json \
    --wandb-name "v11-no-lora-eval" \
    2>&1 | tee $LOG_DIR/eval_v11.log

# v13: LoRA-only, frozen head+decoder (most likely to beat zero-shot)
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

# Precompute zero-shot Depth Pro depths for consistency loss (v15/v16)
ZSCACHE_TRAIN="checkpoints/zeroshot_depths_train_s6_416x128.pt"
ZSCACHE_VAL="checkpoints/zeroshot_depths_val_s6_416x128.pt"
if [ ! -f "$ZSCACHE_TRAIN" ]; then
    echo ""
    echo ">>> Precomputing zero-shot depths (~2h) — $(date)"
    python precompute_zeroshot_depths.py \
        --data-path $DATA --stride 6 --pose-size 416x128 \
        2>&1 | tee $LOG_DIR/precompute_zeroshot.log
    echo ">>> Zero-shot precompute DONE — $(date)"
fi

# ============================================================
# v15: Consistency loss with HEAVY anchor (lambda=10)
# Most likely to beat zero-shot on at least one metric
# ============================================================
echo ""
echo ">>> [v15] Consistency loss lambda=10, 5 epochs — $(date)"
python train_kitti_selfsup_ms.py \
    --data-path $DATA \
    --epochs 5 \
    --stride 6 \
    --lora-rank 8 \
    --lora-alpha 8.0 \
    --zeroshot-depths $ZSCACHE_TRAIN \
    --consistency-weight 10.0 \
    --save-dir checkpoints/selfsup_v15 \
    --wandb-name "v15-consistency-lambda10" \
    2>&1 | tee $LOG_DIR/selfsup_training_v15.log
echo ">>> [v15] DONE — $(date)"
echo ">>> [v15] Evaluating..."
python evaluate_kitti.py \
    --checkpoint checkpoints/selfsup_v15/selfsup_best.pt \
    --lora-rank 8 --lora-alpha 8.0 \
    --output results/eval_v15_consistency_lambda10.json \
    --wandb-name "v15-consistency-eval" \
    2>&1 | tee $LOG_DIR/eval_v15.log

# ============================================================
# v16: Consistency loss BALANCED (lambda=1)
# More room for photometric refinement
# ============================================================
echo ""
echo ">>> [v16] Consistency loss lambda=1, 5 epochs — $(date)"
python train_kitti_selfsup_ms.py \
    --data-path $DATA \
    --epochs 5 \
    --stride 6 \
    --lora-rank 8 \
    --lora-alpha 8.0 \
    --zeroshot-depths $ZSCACHE_TRAIN \
    --consistency-weight 1.0 \
    --save-dir checkpoints/selfsup_v16 \
    --wandb-name "v16-consistency-lambda1" \
    2>&1 | tee $LOG_DIR/selfsup_training_v16.log
echo ">>> [v16] DONE — $(date)"
echo ">>> [v16] Evaluating..."
python evaluate_kitti.py \
    --checkpoint checkpoints/selfsup_v16/selfsup_best.pt \
    --lora-rank 8 --lora-alpha 8.0 \
    --output results/eval_v16_consistency_lambda1.json \
    --wandb-name "v16-consistency-eval" \
    2>&1 | tee $LOG_DIR/eval_v16.log

echo ""
echo "============================================================"
echo "All experiments done — $(date)"
echo "Compare results: cat results/eval_v15*.json results/eval_v16*.json"
echo "============================================================"
