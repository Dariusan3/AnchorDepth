#!/bin/bash
# Wait for zero-shot precomputation, then run v15 (consistency loss).
# v15 uses LoRA r=8 + photometric + lambda*||depth - depth_zero_shot||_1
# This is our best shot at beating zero-shot on at least one metric.

set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate depth-pro

DATA="datasets/kitti_raw"
LOG_DIR="checkpoints"
ZS_TRAIN="checkpoints/zeroshot_depths_train_s6_416x128.pt"
PRECOMPUTE_PID=$(cat checkpoints/precompute.pid)

echo "Waiting for zero-shot precomputation (PID $PRECOMPUTE_PID)..."
while ps -p $PRECOMPUTE_PID > /dev/null 2>&1; do
    sleep 60
done
echo ">>> precompute finished at $(date)"

if [ ! -f "$ZS_TRAIN" ]; then
    echo "FATAL: zero-shot depths not found at $ZS_TRAIN"
    exit 1
fi

# v15: consistency loss with lambda=10 (strong anchor to zero-shot)
echo ""
echo ">>> [v15] Consistency loss (lambda=10), LoRA r=8, 10 epochs — $(date)"
python train_kitti_selfsup_ms.py \
    --data-path $DATA \
    --epochs 10 \
    --stride 6 \
    --lora-rank 8 \
    --lora-alpha 8.0 \
    --lr-depth 1e-5 \
    --lr-lora 1e-6 \
    --lr-pose 1e-5 \
    --zeroshot-depths $ZS_TRAIN \
    --consistency-weight 10.0 \
    --save-dir checkpoints/selfsup_v15 \
    --wandb-name "v15-consistency-lambda10" \
    2>&1 | tee $LOG_DIR/selfsup_training_v15.log
echo ">>> [v15] DONE — $(date)"

# v15 evaluation
echo ">>> [v15] Evaluating..."
python evaluate_kitti.py \
    --checkpoint checkpoints/selfsup_v15/selfsup_best.pt \
    --lora-rank 8 --lora-alpha 8.0 \
    --output results/eval_v15_consistency.json \
    --wandb-name "v15-consistency-eval" \
    2>&1 | tee $LOG_DIR/eval_v15.log

echo ""
echo "============================================================"
echo "v15 done — $(date). Check results/eval_v15_consistency.json"
echo "============================================================"
