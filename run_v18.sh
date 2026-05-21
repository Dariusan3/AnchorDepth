#!/bin/bash
# v18: same as v15 (consistency λ=10, 5 epochs) but consistency in LOG-SPACE.
#
# Why log-space:
#   - RMSElog is computed in log space — log consistency directly optimises it
#   - δ<1.25² counts pixels within log-ratio < log(1.5625) — log consistency
#     directly pushes the prediction inside that band
#   - v15 lost δ<1.25² by only 0.0002 — very close; a small bias change can flip it
#
# What we keep from v15:
#   - λ=10 (heavy anchor, prevents drift on AbsRel/RMSE)
#   - 5 epochs (any longer risks v17-style drift)
#   - LoRA rank 8, no edge-aware, no depth weighting
#
# What we change vs v17 (which failed):
#   - Drop --depth-weight-power 2 entirely (caused near-pixel drift)
#
# v15 and v16 are committed — this script does not touch them.

set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate depth-pro

DATA="datasets/kitti_raw"
LOG_DIR="checkpoints"
ZSCACHE="checkpoints/zeroshot_depths_train_s6_416x128.pt"

if [ ! -f "$ZSCACHE" ]; then
    echo "Zero-shot cache missing — regenerating..."
    python precompute_zeroshot_depths.py --data-path $DATA --stride 6 --pose-size 416x128
fi

echo ""
echo ">>> [v18] Log-space consistency λ=10, 5 epochs — $(date)"
python train_kitti_selfsup_ms.py \
    --data-path $DATA \
    --epochs 5 \
    --stride 6 \
    --lora-rank 8 \
    --lora-alpha 8.0 \
    --zeroshot-depths $ZSCACHE \
    --consistency-weight 10.0 \
    --consistency-mode log \
    --save-dir checkpoints/selfsup_v18 \
    --wandb-name "v18-log-consistency-lambda10" \
    2>&1 | tee $LOG_DIR/selfsup_training_v18.log

echo ">>> [v18] DONE — $(date)"
echo ">>> [v18] Evaluating..."
python evaluate_kitti.py \
    --checkpoint checkpoints/selfsup_v18/selfsup_best.pt \
    --lora-rank 8 --lora-alpha 8.0 \
    --output results/eval_v18_log_l10.json \
    --wandb-name "v18-log-l10-eval" \
    2>&1 | tee $LOG_DIR/eval_v18.log

echo ""
echo "============================================================"
echo "v18 done — $(date)"
echo "Target metrics to beat (zero-shot):"
echo "  RMSElog   < 0.1655   (v15: 0.1665, v16: 0.1721)"
echo "  δ<1.25²   > 0.9725   (v15: 0.9724, v16: 0.9711)"
echo "============================================================"
