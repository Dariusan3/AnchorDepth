#!/bin/bash
# v17: log-space consistency + distance weighting
# Goal: also beat zero-shot on δ<1.25² and RMSElog (v15/v16 already beat δ<1.25³).
#
# Strategy:
#   - log-space consistency directly optimises RMSElog
#   - depth-weight-power=2 emphasises far pixels where δ<1.25² typically loses
#   - λ=10 (heavy anchor, like v15) to avoid drift on AbsRel/RMSE
#
# v15 and v16 are SAFE — this script does not touch them.

set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate depth-pro

DATA="datasets/kitti_raw"
LOG_DIR="checkpoints"
ZSCACHE="checkpoints/zeroshot_depths_train_s6_416x128.pt"

if [ ! -f "$ZSCACHE" ]; then
    echo ">>> Precomputing zero-shot depths (~2h) — $(date)"
    python precompute_zeroshot_depths.py \
        --data-path $DATA --stride 6 --pose-size 416x128 \
        2>&1 | tee $LOG_DIR/precompute_zeroshot.log
fi

echo ""
echo ">>> [v17] Log-consistency λ=10 + depth-weight power=2, 5 epochs — $(date)"
python train_kitti_selfsup_ms.py \
    --data-path $DATA \
    --epochs 5 \
    --stride 6 \
    --lora-rank 8 \
    --lora-alpha 8.0 \
    --zeroshot-depths $ZSCACHE \
    --consistency-weight 10.0 \
    --consistency-mode log \
    --depth-weight-power 2.0 \
    --save-dir checkpoints/selfsup_v17 \
    --wandb-name "v17-log-consistency-far" \
    2>&1 | tee $LOG_DIR/selfsup_training_v17.log

echo ">>> [v17] DONE — $(date)"
echo ">>> [v17] Evaluating..."
python evaluate_kitti.py \
    --checkpoint checkpoints/selfsup_v17/selfsup_best.pt \
    --lora-rank 8 --lora-alpha 8.0 \
    --output results/eval_v17_log_far.json \
    --wandb-name "v17-log-far-eval" \
    2>&1 | tee $LOG_DIR/eval_v17.log

echo ""
echo "============================================================"
echo "v17 done — $(date)"
echo "Compare: results/eval_v17_log_far.json vs eval_v15_consistency.json"
echo "Goal beaten:"
echo "  zero-shot δ<1.25²:  0.9725  (need to exceed)"
echo "  zero-shot RMSElog:  0.1655  (need to be below)"
echo "============================================================"
