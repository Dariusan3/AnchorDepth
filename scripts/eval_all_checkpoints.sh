#!/bin/bash
# Evaluate all available checkpoints on KITTI Eigen test split.
# Run this when GPU is free (between training runs or after all finish).
#
# Usage:
#   bash eval_all_checkpoints.sh
#
# Results saved to results/eval_*.json and printed as a summary table.

set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate depth-pro

mkdir -p results

echo "============================================================"
echo "KITTI Evaluation — $(date)"
echo "============================================================"

run_eval() {
    local name=$1
    local ckpt=$2
    local out="results/eval_${name}.json"

    if [ -f "$out" ]; then
        echo "  [SKIP] $name — already evaluated ($out exists)"
        return
    fi

    if [ -n "$ckpt" ] && [ ! -f "$ckpt" ]; then
        echo "  [SKIP] $name — checkpoint not found: $ckpt"
        return
    fi

    echo ""
    echo ">>> Evaluating: $name"
    if [ -z "$ckpt" ]; then
        python evaluate_kitti.py --output "$out" 2>&1 | tail -20
    else
        python evaluate_kitti.py --checkpoint "$ckpt" --output "$out" 2>&1 | tail -20
    fi
    echo "  -> Saved: $out"
}

# Zero-shot pretrained Depth Pro (no fine-tuning)
run_eval "pretrained_zeroshot" ""

# V5 — single-scale baseline (crashed at 2 epochs, best checkpoint)
run_eval "v5_selfsup_2ep" "checkpoints/selfsup_best.pt"

# V6 — multi-scale 40 epochs (best checkpoint so far)
run_eval "v6_multiscale_best" "checkpoints/selfsup_v6/selfsup_best.pt"

# V7 — no-LoRA ablation
run_eval "v7_no_lora_best" "checkpoints/selfsup_v7/selfsup_best.pt"

# V8 — higher smoothness
run_eval "v8_smooth_best" "checkpoints/selfsup_v8/selfsup_best.pt"

# Print summary table
echo ""
echo "============================================================"
echo "RESULTS SUMMARY"
echo "============================================================"
python - << 'EOF'
import json, os, glob

files = sorted(glob.glob("results/eval_*.json"))
if not files:
    print("No results yet.")
else:
    header = f"{'Run':<30} {'AbsRel':>8} {'SqRel':>8} {'RMSE':>8} {'d<1.25':>8}"
    print(header)
    print("-" * len(header))
    for f in files:
        with open(f) as fp:
            r = json.load(fp)
        name = os.path.basename(f).replace("eval_","").replace(".json","")
        absrel = r.get("abs_rel", r.get("AbsRel", "N/A"))
        sqrel  = r.get("sq_rel",  r.get("SqRel",  "N/A"))
        rmse   = r.get("rmse",    r.get("RMSE",    "N/A"))
        d125   = r.get("a1",      r.get("delta1",  "N/A"))
        def fmt(v): return f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"{name:<30} {fmt(absrel):>8} {fmt(sqrel):>8} {fmt(rmse):>8} {fmt(d125):>8}")
EOF

echo ""
echo "Done. Upload results to WandB with: python upload_v5_to_wandb.py"
