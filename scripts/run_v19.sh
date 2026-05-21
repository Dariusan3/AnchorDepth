#!/bin/bash
# v19: VGGT poses + log-space consistency
# Combines:
#   - v16's better poses (VGGT-1B precomputed)
#   - v18's direct RMSElog optimisation (log-space consistency)
#   - v15's heavy anchor (λ=10) to avoid drift
#
# Use case: if v18 (PoseNet + log) doesn't beat RMSElog, v19 may close the gap
# by reducing pose noise in the photometric refinement signal.
#
# Waits for v18 to finish before starting (PID detected from v18.pid).

set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate depth-pro

DATA="datasets/kitti_raw"
LOG_DIR="checkpoints"
ZSCACHE="checkpoints/zeroshot_depths_train_s6_416x128.pt"
VGGTCACHE="checkpoints/vggt_poses_train_s6.pt"

# Wait for GPU to be FREE (more robust than PID tracking).
# v19 needs 11+ GB; refuse to start until VRAM is essentially idle.
echo "Waiting for GPU to be free (current users including v18 must finish)..."
while true; do
    used_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ "$used_mb" -lt 2000 ]; then
        echo ">>> GPU free ($used_mb MiB used) — proceeding at $(date)"
        break
    fi
    sleep 60
done

echo ""
echo ">>> [v19] VGGT + log-consistency λ=10, 5 epochs — $(date)"
python train_kitti_selfsup_ms.py \
    --data-path $DATA \
    --epochs 5 \
    --stride 6 \
    --lora-rank 8 \
    --lora-alpha 8.0 \
    --zeroshot-depths $ZSCACHE \
    --consistency-weight 10.0 \
    --consistency-mode log \
    --vggt-poses $VGGTCACHE \
    --save-dir checkpoints/selfsup_v19 \
    --wandb-name "v19-vggt-log-lambda10" \
    2>&1 | tee $LOG_DIR/selfsup_training_v19.log

echo ">>> [v19] DONE — $(date)"
echo ">>> [v19] Evaluating..."
python evaluate_kitti.py \
    --checkpoint checkpoints/selfsup_v19/selfsup_best.pt \
    --lora-rank 8 --lora-alpha 8.0 \
    --output results/eval_v19_vggt_log.json \
    --wandb-name "v19-vggt-log-eval" \
    2>&1 | tee $LOG_DIR/eval_v19.log

echo ""
echo "============================================================"
echo "v19 done — $(date)"
echo "Compare RMSElog: v18 vs v19 vs zero-shot (0.16552)"
echo "============================================================"
