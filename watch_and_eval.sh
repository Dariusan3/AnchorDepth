#!/bin/bash
# Monitors v3 self-supervised training and runs GPU evaluation automatically
# after training fully completes (GPU unavailable during training — 11.1/12GB used).
#
# Usage: nohup bash watch_and_eval.sh > checkpoints/watcher.log 2>&1 &

LOG=/home/ubuntu/ml-depth-pro/checkpoints/selfsup_training_v3.log
CHECKPOINT=/home/ubuntu/ml-depth-pro/checkpoints/selfsup/selfsup_best.pt
EVAL_SELFSUP=/home/ubuntu/ml-depth-pro/checkpoints/selfsup/eval_selfsup_v3.json
EVAL_BASELINE=/home/ubuntu/ml-depth-pro/eval_kitti_pretrained.json
PYTHON=/home/ubuntu/anaconda3/envs/depth-pro/bin/python
cd /home/ubuntu/ml-depth-pro

echo "============================================"
echo "  Training watcher started: $(date)"
echo "  Watching for training to complete..."
echo "  GPU eval will run after training finishes."
echo "============================================"

# Wait for training to finish
while true; do
    TRAINING_PID=$(pgrep -f "train_kitti_selfsup.py" | head -1)
    if [ -z "$TRAINING_PID" ]; then
        echo "[$(date)] Training process finished."
        break
    fi

    # Log progress every 30 min
    LATEST=$(grep -oP 'Epoch \d+/20.*?time' "$LOG" 2>/dev/null | tail -1)
    echo "[$(date)] Still training: $LATEST"
    sleep 1800
done

echo ""
echo "[$(date)] ========== Starting GPU evaluations =========="

# 1. Evaluate pretrained baseline (no checkpoint)
echo "[$(date)] Evaluating pretrained baseline on GPU..."
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PYTHON evaluate_kitti.py \
    --data-path datasets/kitti_raw \
    --split splits/eigen_test_files.txt \
    --output "$EVAL_BASELINE" \
    --device cuda \
    2>&1 | tee -a checkpoints/selfsup/eval_log.txt

echo "[$(date)] Baseline eval done."

# 2. Evaluate self-supervised v3 checkpoint
if [ -f "$CHECKPOINT" ]; then
    echo "[$(date)] Evaluating self-supervised v3 model on GPU..."
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PYTHON evaluate_kitti.py \
        --data-path datasets/kitti_raw \
        --split splits/eigen_test_files.txt \
        --checkpoint "$CHECKPOINT" \
        --output "$EVAL_SELFSUP" \
        --device cuda \
        2>&1 | tee -a checkpoints/selfsup/eval_log.txt

    echo "[$(date)] Self-supervised eval done."
else
    echo "[$(date)] WARNING: No checkpoint found at $CHECKPOINT"
fi

echo ""
echo "[$(date)] ========== All evaluations complete =========="
echo "Baseline:       $EVAL_BASELINE"
echo "Self-supervised: $EVAL_SELFSUP"

# Print results side by side
if [ -f "$EVAL_BASELINE" ] && [ -f "$EVAL_SELFSUP" ]; then
    echo ""
    echo "--- Pretrained baseline ---"
    cat "$EVAL_BASELINE"
    echo ""
    echo "--- Self-supervised v3 ---"
    cat "$EVAL_SELFSUP"
fi
