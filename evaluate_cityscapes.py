#!/usr/bin/env python3
"""Cross-domain evaluation on Cityscapes (Cordts et al., CVPR 2016).

Cityscapes contains 500 outdoor driving val images at 2048×1024 from three
cities (Frankfurt, Lindau, Münster). Stereo-derived dense disparity is
provided per image; we convert to depth via depth = f * B / disparity.

Used by most self-supervised depth literature as a cross-domain test for
models trained on KITTI (similar driving distribution but different geography,
camera and weather conditions).

Protocol:
  - Load RGB image at 2048×1024
  - Decode disparity: D = (raw - 1) / 256 for raw > 0; depth = f·B / D
  - Standard Cityscapes calibration: f = 2262.5 px (at 2048×1024), B = 0.209 m
  - Cap depth at 80 m (consistent with KITTI Eigen)
  - Resize prediction to (1024, 2048), apply per-image median scaling
  - Report AbsRel, SqRel, RMSE, RMSElog, δ thresholds

Usage:
  python evaluate_cityscapes.py                                  # zero-shot
  python evaluate_cityscapes.py --checkpoint .../v18/...pt       # v18
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Normalize, ToTensor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))
import depth_pro
from evaluate_kitti import load_model

# Cityscapes stereo calibration (mean across all cities; differences are <1%)
CS_FOCAL_PX = 2262.5
CS_BASELINE = 0.209
MIN_DEPTH = 1.0
MAX_DEPTH = 80.0


def cityscapes_metrics(pred, gt, mask):
    pred_v = pred[mask]
    gt_v = gt[mask]
    if len(pred_v) < 10:
        return None
    scale = np.median(gt_v) / (np.median(pred_v) + 1e-8)
    pred_v = np.clip(pred_v * scale, MIN_DEPTH, MAX_DEPTH)

    thresh = np.maximum(pred_v / gt_v, gt_v / pred_v)
    return {
        "abs_rel": float(np.mean(np.abs(pred_v - gt_v) / gt_v)),
        "sq_rel": float(np.mean((pred_v - gt_v) ** 2 / gt_v)),
        "rmse": float(np.sqrt(np.mean((pred_v - gt_v) ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(pred_v) - np.log(gt_v)) ** 2))),
        "delta1": float(np.mean(thresh < 1.25)),
        "delta2": float(np.mean(thresh < 1.25 ** 2)),
        "delta3": float(np.mean(thresh < 1.25 ** 3)),
        "scale": float(scale),
    }


def decode_disparity_to_depth(disp_path: str):
    """Cityscapes disparity PNG → metric depth map."""
    disp_raw = np.array(Image.open(disp_path))
    valid_raw = disp_raw > 0
    disp = np.zeros_like(disp_raw, dtype=np.float32)
    disp[valid_raw] = (disp_raw[valid_raw].astype(np.float32) - 1.0) / 256.0
    depth = np.zeros_like(disp)
    depth[disp > 0] = CS_FOCAL_PX * CS_BASELINE / disp[disp > 0]
    return depth


def evaluate(model, data_path, device, label):
    data_path = Path(data_path)
    img_paths = sorted(glob.glob(str(data_path / "leftImg8bit/val/*/*.png")))
    print(f"\n{label}: {len(img_paths)} val images")

    normalize = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    to_tensor = ToTensor()
    accum = {k: 0.0 for k in ["abs_rel", "sq_rel", "rmse", "rmse_log",
                              "delta1", "delta2", "delta3"]}
    scales, count, skipped = [], 0, 0

    for img_path in tqdm(img_paths, desc=label):
        disp_path = (img_path.replace("leftImg8bit/", "disparity/")
                              .replace("_leftImg8bit.png", "_disparity.png"))
        if not Path(disp_path).exists():
            skipped += 1
            continue

        gt = decode_disparity_to_depth(disp_path)
        if (gt > 0).sum() < 1000:
            skipped += 1
            continue
        orig_h, orig_w = gt.shape

        img = Image.open(img_path).convert("RGB")
        inp = normalize(to_tensor(img.resize((1536, 1536), Image.LANCZOS))).unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            canon, fov_deg = model(inp)
            # Use known Cityscapes focal length (scaled to 1536-input)
            f_px_1536 = CS_FOCAL_PX * (1536.0 / orig_w)
            inv_depth = canon * (1536.0 / f_px_1536)
            depth_pred = 1.0 / torch.clamp(inv_depth, min=1e-4, max=1e4)

        pred = depth_pred.squeeze().cpu().float().numpy()
        pred = np.array(Image.fromarray(pred).resize((orig_w, orig_h), Image.BILINEAR))

        mask = (gt > MIN_DEPTH) & (gt < MAX_DEPTH)
        m = cityscapes_metrics(pred, gt, mask)
        if m is None:
            skipped += 1
            continue
        for k in accum:
            accum[k] += m[k]
        scales.append(m["scale"])
        count += 1

    results = {k: v / max(count, 1) for k, v in accum.items()}
    results["num_samples"] = count
    results["num_skipped"] = skipped
    results["mean_scale"] = float(np.mean(scales)) if scales else 0.0
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="datasets/cityscapes")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-project", default="depth-pro-selfsup")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model, _ = load_model(device, args.checkpoint, args.lora_rank, 8.0, no_lora=args.no_lora)
        label = args.wandb_name or "checkpoint"
    else:
        print("Zero-shot Depth Pro")
        model, _ = depth_pro.create_model_and_transforms(device=device)
        model.eval()
        label = args.wandb_name or "cityscapes_zeroshot"

    results = evaluate(model, args.data_path, device, label)

    print(f"\n{'='*60}")
    print(f"Cityscapes results — {label}")
    print(f"{'='*60}")
    print(f"  Samples:   {results['num_samples']}  (skipped {results['num_skipped']})")
    print(f"  AbsRel:    {results['abs_rel']:.4f}")
    print(f"  SqRel:     {results['sq_rel']:.4f}")
    print(f"  RMSE:      {results['rmse']:.4f}  m")
    print(f"  RMSElog:   {results['rmse_log']:.4f}")
    print(f"  δ<1.25:    {results['delta1']:.4f}")
    print(f"  δ<1.25²:   {results['delta2']:.4f}")
    print(f"  δ<1.25³:   {results['delta3']:.4f}")
    print(f"  Mean scale: {results['mean_scale']:.4f}")

    out_path = args.output or f"results/eval_cityscapes_{label.replace(' ', '_')}.json"
    Path(out_path).parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    if args.wandb_name:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=f"{args.wandb_name}-cityscapes",
            job_type="cross_domain_eval",
            config={"dataset": "Cityscapes", "checkpoint": args.checkpoint, "lora_rank": args.lora_rank},
        )
        wandb.log({
            "cityscapes/abs_rel": results["abs_rel"],
            "cityscapes/sq_rel": results["sq_rel"],
            "cityscapes/rmse": results["rmse"],
            "cityscapes/rmse_log": results["rmse_log"],
            "cityscapes/delta1": results["delta1"],
            "cityscapes/delta2": results["delta2"],
            "cityscapes/delta3": results["delta3"],
        })
        wandb.summary.update(results)
        wandb.finish()
        print(f"WandB: {run.url}")


if __name__ == "__main__":
    main()
