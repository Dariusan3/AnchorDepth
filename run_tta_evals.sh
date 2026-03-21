#!/bin/bash
PYTHON=/home/ubuntu/anaconda3/envs/depth-pro/bin/python
cd /home/ubuntu/ml-depth-pro

echo "=== Starting Flip TTA evaluation ==="
$PYTHON scripts/evaluate_nyu.py \
  --checkpoint checkpoints/depth_pro_finetuned_best.pt \
  --tta flip \
  --scale-invariant \
  --output experiments/v2_tta_flip/eval_results.json

echo "=== FLIP DONE ==="

echo "=== Starting Full TTA evaluation ==="
$PYTHON scripts/evaluate_nyu.py \
  --checkpoint checkpoints/depth_pro_finetuned_best.pt \
  --tta full \
  --scale-invariant \
  --output experiments/v2_tta_full/eval_results.json

echo "=== FULL DONE ==="
echo "=== ALL EVALUATIONS COMPLETE ==="
