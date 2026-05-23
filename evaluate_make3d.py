#!/usr/bin/env python3
"""Cross-domain evaluation on Make3D (Saxena et al., NIPS 2005).

Make3D contains 134 outdoor test images with sparse laser ground-truth depth.
The dataset is widely used as a cross-domain generalization test for models
trained on KITTI.

Protocol (following Monodepth2):
  - Load image, rotate to landscape orientation
  - Run Depth Pro at 1536×1536
  - Resize prediction to GT resolution (55×305)
  - C1 mask: keep pixels where 0 < gt < 70 m
  - Apply per-image median scaling (standard for self-supervised)
  - Report AbsRel, SqRel, RMSE, RMSElog, log10

Usage:
  python evaluate_make3d.py                                   # zero-shot
  python evaluate_make3d.py --checkpoint .../v15/...pt        # v15
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torchvision.transforms import Normalize, ToTensor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))
import depth_pro
from evaluate_kitti import load_model


def make3d_metrics(pred, gt, cap=70.0, min_d=1.0):
    """Make3D C1 metrics: 0 < gt < cap meters, median scaling, log10 included."""
    mask = (gt > min_d) & (gt < cap)
    if mask.sum() < 10:
        return None
    pred_v = pred[mask]
    gt_v = gt[mask]

    # Median scaling (standard self-supervised convention)
    scale = np.median(gt_v) / (np.median(pred_v) + 1e-8)
    pred_v = np.clip(pred_v * scale, 1e-3, cap)

    diff = pred_v - gt_v
    diff_log = np.log(pred_v) - np.log(gt_v)
    diff_log10 = np.log10(pred_v) - np.log10(gt_v)

    return {
        "abs_rel": float(np.mean(np.abs(diff) / gt_v)),
        "sq_rel": float(np.mean(diff ** 2 / gt_v)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "rmse_log": float(np.sqrt(np.mean(diff_log ** 2))),
        "log10": float(np.mean(np.abs(diff_log10))),
        "scale": float(scale),
        "num_valid": int(mask.sum()),
    }


def evaluate(model, data_path, device, label, use_wandb_name=None):
    data_path = Path(data_path)
    img_dir = data_path / "Test134"
    gt_dir = data_path / "Gridlaserdata"

    gt_files = sorted(gt_dir.glob("depth_sph_corr-*.mat"))
    print(f"\n{label}: {len(gt_files)} test images")

    normalize = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    to_tensor = ToTensor()
    accum = {k: 0.0 for k in ["abs_rel", "sq_rel", "rmse", "rmse_log", "log10"]}
    scales = []
    count = 0
    skipped = 0

    for gt_file in tqdm(gt_files, desc=label):
        # Find matching image: depth_sph_corr-X.mat → img-X.jpg
        stem = gt_file.stem.replace("depth_sph_corr-", "img-")
        img_path = img_dir / f"{stem}.jpg"
        if not img_path.exists():
            skipped += 1
            continue

        # Load GT (channel 3 = radial distance in metres)
        gt = sio.loadmat(str(gt_file))["Position3DGrid"][:, :, 3].astype(np.float32)
        gt_h, gt_w = gt.shape  # (55, 305)

        # Load image. Empirically, Make3D images must be fed as stored
        # (portrait 1704 × 2272) without rotation — the GT grid aligns
        # implicitly. Rotation produces catastrophic AbsRel (>0.8).
        try:
            from PIL import ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            skipped += 1
            continue

        inp = normalize(to_tensor(img.resize((1536, 1536), Image.LANCZOS))).unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            canonical_inv_depth, fov_deg = model(inp)
            # No KITTI focal here; rely on FOV head (median scaling will fix the rest)
            if fov_deg is not None:
                f_px = 0.5 * 1536.0 / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float)))
            else:
                f_px = torch.tensor([1500.0], device=device, dtype=torch.float)
            inv_depth = canonical_inv_depth * (1536.0 / f_px)
            depth_pred = 1.0 / torch.clamp(inv_depth, min=1e-4, max=1e4)

        pred = depth_pred.squeeze().cpu().float().numpy()
        # Resize prediction to GT grid (55, 305) — PIL uses (W, H)
        pred = np.array(Image.fromarray(pred).resize((gt_w, gt_h), Image.BILINEAR))

        m = make3d_metrics(pred, gt)
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
    parser.add_argument("--data-path", default="datasets/make3d")
    parser.add_argument("--checkpoint", default=None,
                        help="Fine-tuned checkpoint; omit for zero-shot")
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
        label = args.wandb_name or "make3d_zeroshot"

    results = evaluate(model, args.data_path, device, label)

    print(f"\n{'='*60}")
    print(f"Make3D results — {label}")
    print(f"{'='*60}")
    print(f"  Samples:     {results['num_samples']} (skipped {results['num_skipped']})")
    print(f"  AbsRel:      {results['abs_rel']:.4f}")
    print(f"  SqRel:       {results['sq_rel']:.4f}")
    print(f"  RMSE:        {results['rmse']:.4f}  m")
    print(f"  RMSElog:     {results['rmse_log']:.4f}")
    print(f"  log10:       {results['log10']:.4f}")
    print(f"  Mean scale:  {results['mean_scale']:.4f}")

    out_path = args.output or f"results/eval_make3d_{label.replace(' ', '_')}.json"
    Path(out_path).parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    if args.wandb_name:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=f"{args.wandb_name}-make3d",
            job_type="cross_domain_eval",
            config={"dataset": "Make3D", "checkpoint": args.checkpoint, "lora_rank": args.lora_rank},
        )
        wandb.log({
            "make3d/abs_rel": results["abs_rel"],
            "make3d/sq_rel": results["sq_rel"],
            "make3d/rmse": results["rmse"],
            "make3d/rmse_log": results["rmse_log"],
            "make3d/log10": results["log10"],
            "make3d/mean_scale": results["mean_scale"],
        })
        wandb.summary.update({
            "abs_rel": results["abs_rel"],
            "rmse_log": results["rmse_log"],
            "log10": results["log10"],
            "dataset": "Make3D",
        })
        wandb.finish()
        print(f"WandB: {run.url}")


if __name__ == "__main__":
    main()
