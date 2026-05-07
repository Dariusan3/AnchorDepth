#!/bin/bash
# v16: VGGT poses + edge-aware consistency loss
#
# Combines:
#   A. VGGT-1B precomputed poses (replaces noisy ResNet-18 PoseNet)
#   B. Edge-aware consistency: strong anchor in smooth regions, weak at edges
#
# Hypothesis: better poses + targeted regularization gives photometric loss
# real adaptation room without catastrophic drift.

set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate depth-pro

DATA="datasets/kitti_raw"
LOG_DIR="checkpoints"
VGGT_TRAIN="checkpoints/vggt_poses_train_s6.pt"
ZS_TRAIN="checkpoints/zeroshot_depths_train_s6_416x128.pt"

# ============================================================
# Step 1: Precompute VGGT poses if missing
# ============================================================
if [ ! -f "$VGGT_TRAIN" ]; then
    echo ">>> Precomputing VGGT poses — $(date)"
    python precompute_vggt_poses.py \
        --data-path $DATA \
        --stride 6 \
        --split both \
        2>&1 | tee $LOG_DIR/precompute_vggt.log
    echo ">>> VGGT precompute done — $(date)"
else
    echo ">>> VGGT poses cached at $VGGT_TRAIN"
fi

if [ ! -f "$VGGT_TRAIN" ]; then
    echo "FATAL: VGGT poses still missing after precompute"
    exit 1
fi
if [ ! -f "$ZS_TRAIN" ]; then
    echo "FATAL: zero-shot depths missing at $ZS_TRAIN"
    exit 1
fi

# ============================================================
# Step 2: Train v16
# ============================================================
echo ""
echo ">>> [v16] VGGT poses + edge-aware consistency (lambda=1) — $(date)"
python train_kitti_selfsup_ms.py \
    --data-path $DATA \
    --epochs 10 \
    --stride 6 \
    --lora-rank 8 \
    --lora-alpha 8.0 \
    --lr-depth 1e-5 \
    --lr-lora 1e-6 \
    --lr-pose 1e-5 \
    --vggt-poses $VGGT_TRAIN \
    --zeroshot-depths $ZS_TRAIN \
    --consistency-weight 1.0 \
    --edge-aware-consistency \
    --save-dir checkpoints/selfsup_v16 \
    --wandb-name "v16-vggt-edge-aware" \
    2>&1 | tee $LOG_DIR/selfsup_training_v16.log
echo ">>> [v16] DONE — $(date)"

# ============================================================
# Step 3: Evaluate v16
# ============================================================
echo ">>> [v16] Evaluating on KITTI Eigen test split..."
python evaluate_kitti.py \
    --checkpoint checkpoints/selfsup_v16/selfsup_best.pt \
    --lora-rank 8 --lora-alpha 8.0 \
    --output results/eval_v16_vggt_edge.json \
    --wandb-name "v16-vggt-edge-eval" \
    2>&1 | tee $LOG_DIR/eval_v16.log

echo ""
echo "============================================================"
echo "v16 done — $(date). Check results/eval_v16_vggt_edge.json"
echo "============================================================"
