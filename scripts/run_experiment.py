#!/usr/bin/env python3
"""Run a complete experiment: train, evaluate, save results, and log to docs.

Usage:
  python scripts/run_experiment.py \
    --name v2_tta \
    --description "Test-time augmentation with multi-scale + flip" \
    --train-args "--epochs 25 --lr 1e-4" \
    --eval-samples 20

  # Skip training (eval-only with existing checkpoint):
  python scripts/run_experiment.py \
    --name v2_tta \
    --checkpoint checkpoints/depth_pro_finetuned_best.pt \
    --skip-training \
    --eval-samples 20
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_cmd(cmd, desc=""):
    """Run a shell command and stream output."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  $ {cmd}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"WARNING: Command exited with code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run a complete depth estimation experiment")
    parser.add_argument("--name", required=True, help="Experiment name (e.g., v2_tta)")
    parser.add_argument("--description", required=True, help="Short description of what changed")
    parser.add_argument("--train-args", type=str, default="--epochs 25 --lr 1e-4",
                        help="Arguments to pass to train_nyu.py")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Existing checkpoint to evaluate (skips training)")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, only evaluate")
    parser.add_argument("--eval-samples", type=int, default=20,
                        help="Number of visual samples to save")
    parser.add_argument("--python", type=str, default="/home/ubuntu/anaconda3/envs/depth-pro/bin/python")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    exp_dir = root / "experiments" / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    checkpoint_path = args.checkpoint

    # ── Step 1: Train ──
    if not args.skip_training:
        print("\n" + "=" * 60)
        print(f"  EXPERIMENT: {args.name}")
        print(f"  {args.description}")
        print("=" * 60)

        train_cmd = (
            f"{args.python} {root}/scripts/train_nyu.py "
            f"{args.train_args} "
            f"--save-dir {exp_dir}"
        )
        run_cmd(train_cmd, "Training")
        checkpoint_path = str(exp_dir / "depth_pro_finetuned_best.pt")

        # Copy training log
        if (exp_dir / "training_log.json").exists():
            print("Training log saved.")

    # ── Step 2: Evaluate on full test set ──
    eval_output = str(exp_dir / "eval_results.json")
    eval_cmd = (
        f"{args.python} {root}/scripts/evaluate_nyu.py "
        f"--scale-invariant "
        f"--output {eval_output}"
    )
    if checkpoint_path:
        eval_cmd += f" --checkpoint {checkpoint_path}"
    run_cmd(eval_cmd, "Full Evaluation (654 test images)")

    # ── Step 3: Save visual results ──
    results_dir = str(exp_dir / "results")
    save_cmd = (
        f"{args.python} {root}/scripts/save_results.py "
        f"--num-samples {args.eval_samples} "
        f"--output-dir {results_dir}"
    )
    if checkpoint_path:
        save_cmd += f" --checkpoint-finetuned {checkpoint_path}"
    run_cmd(save_cmd, f"Saving {args.eval_samples} visual samples")

    # ── Step 4: Load results and create experiment doc ──
    if Path(eval_output).exists():
        with open(eval_output) as f:
            results = json.load(f)
    else:
        results = {}

    # Load baseline for comparison
    baseline_path = root / "experiments" / "v0_baseline" / "eval_results_nyu_full.json"
    baseline = {}
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)

    # Create experiment markdown doc
    doc = f"""# Experiment: {args.name}

**Date:** {timestamp}
**Description:** {args.description}

## Configuration

```
Train args: {args.train_args}
Checkpoint: {checkpoint_path or 'pretrained (no fine-tuning)'}
```

## Results — Metric Depth (NYU Depth V2, Eigen 654 test)

| Metric | Baseline (pretrained) | This Experiment | Change |
|--------|----------------------|-----------------|--------|
"""
    metrics = ["abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3"]
    labels = ["AbsRel (↓)", "SqRel (↓)", "RMSE (↓)", "RMSElog (↓)",
              "delta<1.25 (↑)", "delta<1.25² (↑)", "delta<1.25³ (↑)"]

    for metric, label in zip(metrics, labels):
        val = results.get(metric, 0)
        base = baseline.get(metric, 0)
        if base > 0:
            if metric.startswith("delta"):
                change = ((val - base) / base) * 100
            else:
                change = ((base - val) / base) * 100
            arrow = "+" if change > 0 else ""
            doc += f"| {label} | {base:.4f} | {val:.4f} | {arrow}{change:.1f}% |\n"
        else:
            doc += f"| {label} | — | {val:.4f} | — |\n"

    if "scale_invariant" in results:
        si = results["scale_invariant"]
        doc += f"""
## Results — Scale-Invariant (aligned)

| Metric | Value |
|--------|-------|
| AbsRel | {si.get('abs_rel', 0):.4f} |
| RMSE | {si.get('rmse', 0):.4f} |
| delta<1.25 | {si.get('delta1', 0):.4f} |
"""

    doc += f"""
## Visual Results

See `results/comparison/` for side-by-side depth map comparisons.

## Files

- `eval_results.json` — Full evaluation metrics
- `training_log.json` — Per-epoch training loss and validation metrics
- `depth_pro_finetuned_best.pt` — Best checkpoint
- `results/` — Visual outputs (RGB, depth maps, error maps, comparisons)
"""

    doc_path = exp_dir / "README.md"
    with open(doc_path, "w") as f:
        f.write(doc)
    print(f"\nExperiment doc saved to {doc_path}")

    # ── Step 5: Update master experiments log ──
    update_master_log(root, args.name, args.description, results, baseline, timestamp)

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT {args.name} COMPLETE")
    print(f"  Results in: {exp_dir}")
    print(f"{'='*60}")


def update_master_log(root, name, description, results, baseline, timestamp):
    """Append to the master experiments log."""
    log_path = root / "docs" / "EXPERIMENTS.md"

    # Create header if file doesn't exist
    if not log_path.exists():
        header = """# Depth Pro — Experiment Log

Tracking all modifications and their impact on NYU Depth V2 (Eigen 654 test split).

| # | Experiment | Date | Description | AbsRel | RMSE | delta<1.25 | vs Baseline |
|---|-----------|------|-------------|--------|------|------------|-------------|
"""
        with open(log_path, "w") as f:
            f.write(header)

    # Count existing entries
    with open(log_path) as f:
        lines = f.readlines()
    num = sum(1 for l in lines if l.startswith("|") and not l.startswith("| #") and not l.startswith("|---"))

    abs_rel = results.get("abs_rel", 0)
    rmse = results.get("rmse", 0)
    delta1 = results.get("delta1", 0)

    base_absrel = baseline.get("abs_rel", 0)
    if base_absrel > 0:
        improvement = ((base_absrel - abs_rel) / base_absrel) * 100
        vs_baseline = f"{improvement:+.1f}%"
    else:
        vs_baseline = "baseline"

    entry = (
        f"| {num} | [{name}](../experiments/{name}/README.md) | {timestamp} | "
        f"{description} | {abs_rel:.4f} | {rmse:.4f} | {delta1:.4f} | {vs_baseline} |\n"
    )

    with open(log_path, "a") as f:
        f.write(entry)

    print(f"Master log updated: {log_path}")


if __name__ == "__main__":
    main()
