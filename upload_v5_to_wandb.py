#!/usr/bin/env python3
"""Upload v5 training results to WandB — works live while training runs.

Two modes:
  1. Live mode (default): watches the JSON log and uploads each new epoch as it arrives.
     Run this while v5 is still training:
       python upload_v5_to_wandb.py --live

  2. One-shot mode: upload all completed epochs and exit.
     Run after v5 finishes (or anytime):
       python upload_v5_to_wandb.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import wandb

LOG_PATH = Path("checkpoints/training_log_selfsup.json")


def load_log():
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def log_epoch(entry):
    epoch = entry["epoch"]
    log_dict = {
        "epoch": epoch,
        "train/loss": entry.get("train_loss"),
        "train/photometric": entry.get("train_photometric"),
        "train/smoothness": entry.get("train_smoothness"),
        "train/auto_mask_ratio": entry.get("auto_mask_ratio"),
        "lr/lora": entry.get("lr_lora"),
        "lr/depth": entry.get("lr_depth"),
        "lr/pose": entry.get("lr_pose"),
        "epoch_time_s": entry.get("epoch_time"),
    }
    if "val_photometric" in entry:
        log_dict["val/photometric"] = entry["val_photometric"]

    log_dict = {k: v for k, v in log_dict.items() if v is not None}
    wandb.log(log_dict, step=epoch)

    msg = f"  Epoch {epoch}: train_loss={entry.get('train_loss', 0):.4f}"
    if "val_photometric" in entry:
        msg += f", val_photo={entry['val_photometric']:.4f}"
    print(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="Watch log file and upload new epochs as they arrive")
    parser.add_argument("--poll-interval", type=int, default=120,
                        help="Seconds between checks in live mode (default: 120)")
    args = parser.parse_args()

    run = wandb.init(
        project="depth-pro-selfsup",
        name="v5-single-scale-20ep",
        config={
            "epochs": 20,
            "lr_depth": 1e-4,
            "lr_lora": 1e-5,
            "lr_pose": 1e-4,
            "lora_rank": 8,
            "lora_alpha": 8.0,
            "batch_size": 1,
            "grad_accum": 4,
            "smoothness_weight": 1e-3,
            "pose_size": "640x192",
            "stride": 3,
            "warmup_epochs": 3,
            "multi_scale": False,
            "notes": "First run with correct autograd fix in warping.py.",
        },
        resume="allow",
    )
    print(f"WandB run: {run.url}\n")

    uploaded_epochs = set()

    if args.live:
        print(f"Live mode: checking every {args.poll_interval}s for new epochs...")
        while True:
            entries = load_log()
            for entry in entries:
                if entry["epoch"] not in uploaded_epochs:
                    log_epoch(entry)
                    uploaded_epochs.add(entry["epoch"])

            # Exit when all 20 epochs are uploaded
            if len(uploaded_epochs) >= 20:
                print("\nAll 20 epochs uploaded.")
                break

            time.sleep(args.poll_interval)
    else:
        # One-shot: upload whatever is in the log right now
        entries = load_log()
        if not entries:
            print(f"No data in {LOG_PATH} yet.")
        for entry in entries:
            log_epoch(entry)
            uploaded_epochs.add(entry["epoch"])
        print(f"\nUploaded {len(uploaded_epochs)} epoch(s).")

    # Add eval results if available
    eval_results_path = Path("results/v5_eval.json")
    if eval_results_path.exists():
        with open(eval_results_path) as f:
            eval_results = json.load(f)
        wandb.summary.update(eval_results)
        print(f"Added eval results: {eval_results}")

    wandb.finish()
    print(f"Done! View at: {run.url}")


if __name__ == "__main__":
    main()
