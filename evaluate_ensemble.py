#!/usr/bin/env python3
"""Ensemble evaluation: combine zero-shot Depth Pro and v15 predictions.

No training involved. Runs both models once over the KITTI Eigen test split,
caches the raw per-image depth predictions, then evaluates several
combination rules:
  - arithmetic mean   : (d_zs + d_v15) / 2
  - geometric mean    : sqrt(d_zs * d_v15)
  - weighted blend    : w * d_zs + (1-w) * d_v15, w swept 0..1

Arithmetic and geometric means require no tuning and are reported as the
legitimate ensemble results. The weighted sweep is reported as analysis
(oracle upper bound) to show how much complementarity exists.

Usage:
  python evaluate_ensemble.py --v15 checkpoints/selfsup_v15/selfsup_best.pt
"""

import argparse
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
from evaluate_kitti import (
    load_velodyne_points, load_calib, project_velodyne_to_cam,
    garg_crop, compute_metrics, load_model,
)

ZS_FOCAL = True   # use KITTI GT focal length (matches evaluate_kitti)


def predict_all(model, data_path, filenames, device, use_gt_focal):
    """Run a model over all test images. Return {idx: pred_depth at orig res}."""
    data_path = Path(data_path)
    normalize = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    to_tensor = ToTensor()
    preds = {}

    for idx, (folder, frame_idx, side) in enumerate(tqdm(filenames, desc="predict")):
        cam = "image_02" if side == "l" else "image_03"
        img_path = data_path / folder / cam / "data" / f"{frame_idx:010d}.png"
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        date = folder.split("/")[0]
        P2, _, _ = load_calib(str(data_path / date))

        inp = normalize(to_tensor(img.resize((1536, 1536), Image.LANCZOS))).unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            canonical_inv_depth, fov_deg = model(inp)
            if use_gt_focal:
                f_px = torch.tensor([P2[0, 0]], device=device, dtype=torch.float)
            elif fov_deg is not None:
                f_px = 0.5 * orig_w / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float)))
            else:
                f_px = torch.tensor([P2[0, 0]], device=device, dtype=torch.float)
            inv_depth = canonical_inv_depth * (orig_w / f_px)
            depth = 1.0 / torch.clamp(inv_depth, min=1e-4, max=1e4)

        pred = depth.squeeze().cpu().float().numpy()
        if pred.shape != (orig_h, orig_w):
            pred = np.array(Image.fromarray(pred).resize((orig_w, orig_h), Image.BILINEAR))
        preds[idx] = pred.astype(np.float16)
    return preds


def load_gt(data_path, filenames):
    """Load Garg-cropped LiDAR GT depth for all test images."""
    data_path = Path(data_path)
    gts = {}
    for idx, (folder, frame_idx, side) in enumerate(tqdm(filenames, desc="gt")):
        date, drive = folder.split("/")[0], folder.split("/")[1]
        velo_path = data_path / date / drive / "velodyne_points" / "data" / f"{frame_idx:010d}.bin"
        cam = "image_02" if side == "l" else "image_03"
        img_path = data_path / folder / cam / "data" / f"{frame_idx:010d}.png"
        if not velo_path.exists() or not img_path.exists():
            continue
        orig_w, orig_h = Image.open(img_path).size
        velo = load_velodyne_points(str(velo_path))
        P2, R_rect, Tr = load_calib(str(data_path / date))
        gt = project_velodyne_to_cam(velo, P2, R_rect, Tr, orig_h, orig_w)
        gts[idx] = garg_crop(gt)
    return gts


def eval_combination(preds_zs, preds_v15, gts, combine_fn):
    """Apply combine_fn to each prediction pair, compute mean metrics."""
    accum = {k: 0.0 for k in ["abs_rel", "sq_rel", "rmse", "rmse_log",
                              "delta1", "delta2", "delta3"]}
    count = 0
    for idx in gts:
        if idx not in preds_zs or idx not in preds_v15:
            continue
        d_zs = preds_zs[idx].astype(np.float32)
        d_v15 = preds_v15[idx].astype(np.float32)
        d_comb = combine_fn(d_zs, d_v15)
        pred_crop = garg_crop(d_comb)
        m = compute_metrics(pred_crop, gts[idx])
        if m is None:
            continue
        for k in accum:
            accum[k] += m[k]
        count += 1
    return {k: v / max(count, 1) for k, v in accum.items()}, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="datasets/kitti_raw")
    parser.add_argument("--split", default="splits/eigen_test_files.txt")
    parser.add_argument("--v15", default="checkpoints/selfsup_v15/selfsup_best.pt")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--output", default="results/eval_ensemble.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    filenames = []
    with open(args.split) as f:
        for line in f:
            p = line.strip().split()
            filenames.append((p[0], int(p[1]), p[2]))

    # Pass 1: zero-shot predictions
    print("\n=== Pass 1: zero-shot Depth Pro ===")
    model_zs, _ = depth_pro.create_model_and_transforms(device=device)
    model_zs.eval()
    preds_zs = predict_all(model_zs, args.data_path, filenames, device, use_gt_focal=ZS_FOCAL)
    del model_zs
    torch.cuda.empty_cache()

    # Pass 2: v15 predictions
    print("\n=== Pass 2: v15 (consistency LoRA) ===")
    model_v15, _ = load_model(device, args.v15, args.lora_rank, 8.0, no_lora=False)
    preds_v15 = predict_all(model_v15, args.data_path, filenames, device, use_gt_focal=True)
    del model_v15
    torch.cuda.empty_cache()

    # Ground truth
    print("\n=== Loading GT ===")
    gts = load_gt(args.data_path, filenames)

    # Combination rules
    print("\n=== Evaluating combinations ===")
    results = {}

    arith, n = eval_combination(preds_zs, preds_v15, gts,
                                lambda a, b: 0.5 * (a + b))
    results["arithmetic_mean"] = arith
    print(f"\nArithmetic mean ({n} imgs): AbsRel={arith['abs_rel']:.5f} "
          f"RMSElog={arith['rmse_log']:.5f} d2={arith['delta2']:.5f} d3={arith['delta3']:.5f}")

    geo, n = eval_combination(preds_zs, preds_v15, gts,
                              lambda a, b: np.sqrt(np.clip(a, 1e-3, None) * np.clip(b, 1e-3, None)))
    results["geometric_mean"] = geo
    print(f"Geometric mean  ({n} imgs): AbsRel={geo['abs_rel']:.5f} "
          f"RMSElog={geo['rmse_log']:.5f} d2={geo['delta2']:.5f} d3={geo['delta3']:.5f}")

    # Weighted sweep (analysis / oracle upper bound)
    print("\nWeighted blend sweep  w*d_zs + (1-w)*d_v15:")
    best = {"rmse_log": (1e9, None), "delta2": (-1, None)}
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        m, n = eval_combination(preds_zs, preds_v15, gts,
                                lambda a, b, w=w: w * a + (1 - w) * b)
        results[f"blend_w{w:.1f}"] = m
        flag = ""
        if m["rmse_log"] < best["rmse_log"][0]:
            best["rmse_log"] = (m["rmse_log"], w)
        if m["delta2"] > best["delta2"][0]:
            best["delta2"] = (m["delta2"], w)
        print(f"  w={w:.1f}: AbsRel={m['abs_rel']:.5f} RMSElog={m['rmse_log']:.5f} "
              f"d2={m['delta2']:.5f} d3={m['delta3']:.5f}")

    # Reference numbers
    print("\n=== Targets (zero-shot baseline) ===")
    print("  AbsRel  < 0.08661")
    print("  RMSElog < 0.16552")
    print("  d<1.25^2 > 0.97254")
    print("  d<1.25^3 > 0.98494")
    print(f"\nBest RMSElog: {best['rmse_log'][0]:.5f} at w={best['rmse_log'][1]}")
    print(f"Best d<1.25^2: {best['delta2'][0]:.5f} at w={best['delta2'][1]}")

    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
