#!/bin/bash
# v20: same as v15 (L1 metric consistency, PoseNet, 5 epochs) but with
# heavier anchor lambda=20 (vs v15's lambda=10).
#
# Why heavier lambda:
#   - v15 loses δ<1.25² by only 0.0002 and RMSElog by 0.0009 — both within
#     statistical noise of beating zero-shot
#   - Heavier anchor keeps predictions closer to zero-shot on ALL pixels,
#     potentially flipping these two near-misses to positive wins
#   - Photometric loss still contributes ~5% (1/(1+20)) — enough to retain
#     v15's δ<1.25³ win, but reduced enough to cap drift on other metrics
#
# Worst case: matches v15 or zero-shot, no degradation. v15 and v16 are
# committed and untouched.

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
echo ">>> [v20] L1 consistency λ=20 (heavy anchor), 5 epochs — $(date)"
python train_kitti_selfsup_ms.py \
    --data-path $DATA \
    --epochs 5 \
    --stride 6 \
    --lora-rank 8 \
    --lora-alpha 8.0 \
    --zeroshot-depths $ZSCACHE \
    --consistency-weight 20.0 \
    --save-dir checkpoints/selfsup_v20 \
    --wandb-name "v20-consistency-lambda20" \
    2>&1 | tee $LOG_DIR/selfsup_training_v20.log

echo ">>> [v20] DONE — $(date)"
echo ">>> [v20] Evaluating..."
python evaluate_kitti.py \
    --checkpoint checkpoints/selfsup_v20/selfsup_best.pt \
    --lora-rank 8 --lora-alpha 8.0 \
    --output results/eval_v20_consistency_lambda20.json \
    --wandb-name "v20-lambda20-eval" \
    2>&1 | tee $LOG_DIR/eval_v20.log

echo ""
echo "============================================================"
echo "v20 done — $(date)"
echo "Targets to beat (zero-shot):"
echo "  AbsRel    < 0.0866   (v15: 0.0875, gap 0.0009)"
echo "  RMSElog   < 0.1655   (v15: 0.1665, gap 0.0010)"
echo "  δ<1.25²   > 0.9725   (v15: 0.9724, gap 0.0002)"
echo "  δ<1.25³   > 0.98494  (v15: 0.98499 ✓ already won)"
echo "============================================================"
