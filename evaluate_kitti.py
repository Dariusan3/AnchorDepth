#!/usr/bin/env python3
"""Evaluate self-supervised Depth Pro on KITTI Eigen test split.

Uses LiDAR ground truth depth projected to the image plane.
Applies median scaling (standard for self-supervised methods) and
Garg/Eigen crop for fair comparison with published results.

Metrics: AbsRel, SqRel, RMSE, RMSE_log, delta < 1.25/1.25^2/1.25^3

Usage:
  python evaluate_kitti.py [--checkpoint checkpoints/selfsup_best.pt]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))
import depth_pro
from train_nyu_lora import LoRALinear, apply_lora_to_encoder


def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI velodyne binary file."""
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]


def load_calib(calib_dir):
    """Load KITTI calibration files and return projection matrices."""
    data = {}
    for name in ["calib_cam_to_cam.txt", "calib_velo_to_cam.txt"]:
        filepath = os.path.join(calib_dir, name)
        with open(filepath) as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    data[key.strip()] = value.strip()

    # Camera intrinsics (P_rect_02)
    P2 = np.array(data["P_rect_02"].split(), dtype=np.float32).reshape(3, 4)

    # Rectification rotation
    R_rect = np.eye(4, dtype=np.float32)
    R_rect[:3, :3] = np.array(data["R_rect_00"].split(), dtype=np.float32).reshape(3, 3)

    # Velodyne to camera transformation
    Tr_velo = np.eye(4, dtype=np.float32)
    Tr_velo[:3, :] = np.array(data["T_02"].split() if "T_02" in data
                                else data["T_velo_to_cam"].split(),
                                dtype=np.float32).reshape(3, 4)

    return P2, R_rect, Tr_velo


def project_velodyne_to_cam(velo_pts, P2, R_rect, Tr_velo, img_h, img_w):
    """Project velodyne points to camera image plane.

    Returns:
        depth_map: (H, W) sparse depth map from LiDAR.
    """
    # Make homogeneous
    pts_hom = np.hstack([velo_pts, np.ones((velo_pts.shape[0], 1), dtype=np.float32)])

    # Project: P2 @ R_rect @ Tr_velo @ points
    pts_cam = (P2 @ R_rect @ Tr_velo @ pts_hom.T).T  # (N, 3)

    # Filter points behind camera
    mask = pts_cam[:, 2] > 0
    pts_cam = pts_cam[mask]

    # Normalize to pixel coordinates
    pts_2d = pts_cam[:, :2] / pts_cam[:, 2:3]
    depths = pts_cam[:, 2]

    # Filter points outside image
    mask = (
        (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < img_w) &
        (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < img_h)
    )
    pts_2d = pts_2d[mask]
    depths = depths[mask]

    # Create sparse depth map
    depth_map = np.zeros((img_h, img_w), dtype=np.float32)
    u = np.round(pts_2d[:, 0]).astype(int)
    v = np.round(pts_2d[:, 1]).astype(int)

    # Use closest point for each pixel (handle occlusions)
    for i in range(len(u)):
        if depth_map[v[i], u[i]] == 0 or depths[i] < depth_map[v[i], u[i]]:
            depth_map[v[i], u[i]] = depths[i]

    return depth_map


def garg_crop(depth):
    """Apply Garg/Eigen crop to depth map."""
    h, w = depth.shape
    crop = np.array([
        0.40810811 * h, 0.99189189 * h,
        0.03594771 * w, 0.96405229 * w
    ]).astype(np.int32)
    return depth[crop[0]:crop[1], crop[2]:crop[3]]


def compute_metrics(pred, gt):
    """Compute standard depth estimation metrics."""
    # Standard KITTI depth cap: 1m to 80m
    mask = (gt > 1e-3) & (gt < 80.0)

    pred = pred[mask]
    gt = gt[mask]

    if len(pred) < 10:
        return None

    # Median scaling (standard for self-supervised methods)
    scale = np.median(gt) / np.median(pred)
    pred = pred * scale

    # Cap prediction to valid range
    pred = np.clip(pred, 1e-3, 80.0)

    thresh = np.maximum(pred / gt, gt / pred)

    return {
        "abs_rel": float(np.mean(np.abs(pred - gt) / gt)),
        "sq_rel": float(np.mean((pred - gt) ** 2 / gt)),
        "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))),
        "delta1": float(np.mean(thresh < 1.25)),
        "delta2": float(np.mean(thresh < 1.25 ** 2)),
        "delta3": float(np.mean(thresh < 1.25 ** 3)),
        "scale": float(scale),
    }


def load_model(device, checkpoint_path=None, lora_rank=8, lora_alpha=8.0):
    """Load Depth Pro with LoRA structure."""
    model, transform = depth_pro.create_model_and_transforms(device=device)

    # Apply LoRA structure
    apply_lora_to_encoder(model, rank=lora_rank, alpha=lora_alpha)

    # Move LoRA params to device
    for enc_name in ["patch_encoder", "image_encoder"]:
        enc = getattr(model.encoder, enc_name)
        for block in enc.blocks:
            if isinstance(block.attn.qkv, LoRALinear):
                block.attn.qkv.lora_A = nn.Parameter(block.attn.qkv.lora_A.to(device))
                block.attn.qkv.lora_B = nn.Parameter(block.attn.qkv.lora_B.to(device))
            if isinstance(block.attn.proj, LoRALinear):
                block.attn.proj.lora_A = nn.Parameter(block.attn.proj.lora_A.to(device))
                block.attn.proj.lora_B = nn.Parameter(block.attn.proj.lora_B.to(device))

    if checkpoint_path is not None:
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        # Handle both full state_dict and wrapped dict
        state_dict = ckpt.get("depth_model", ckpt)
        model.load_state_dict(state_dict, strict=True)
        del ckpt
        torch.cuda.empty_cache()

    model.eval()
    return model, transform


def evaluate(
    model, data_path, split_file, device,
    use_median_scaling=True, verbose=True,
):
    """Evaluate model on KITTI Eigen test split."""
    data_path = Path(data_path)

    # Parse split file
    filenames = []
    with open(split_file) as f:
        for line in f:
            parts = line.strip().split()
            filenames.append((parts[0], int(parts[1]), parts[2]))

    # Depth Pro normalization
    from torchvision.transforms import Normalize
    normalize = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    metrics_accum = {k: 0.0 for k in [
        "abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3"
    ]}
    scales = []
    count = 0
    times = []

    for folder, frame_idx, side in tqdm(filenames, desc="Evaluating"):
        # Load image
        cam = "image_02" if side == "l" else "image_03"
        img_path = data_path / folder / cam / "data" / f"{frame_idx:010d}.png"
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Load LiDAR ground truth
        date = folder.split("/")[0]
        drive = folder.split("/")[1]
        velo_path = data_path / date / drive / "velodyne_points" / "data" / f"{frame_idx:010d}.bin"
        if not velo_path.exists():
            continue

        velo_pts = load_velodyne_points(str(velo_path))
        calib_dir = str(data_path / date)
        P2, R_rect, Tr_velo = load_calib(calib_dir)
        gt_depth = project_velodyne_to_cam(velo_pts, P2, R_rect, Tr_velo, orig_h, orig_w)

        # Apply Garg crop to GT
        gt_crop = garg_crop(gt_depth)
        if (gt_crop > 0).sum() < 100:
            continue

        # Run inference
        img_resized = img.resize((1536, 1536), Image.LANCZOS)
        from torchvision.transforms import ToTensor
        img_tensor = ToTensor()(img_resized).to(device)
        img_tensor = normalize(img_tensor).unsqueeze(0)

        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            canonical_inv_depth, fov_deg = model(img_tensor)

            # Convert to depth
            if fov_deg is not None:
                f_px = 0.5 * orig_w / torch.tan(
                    0.5 * torch.deg2rad(fov_deg.to(torch.float))
                )
            else:
                # Use KITTI focal length as fallback
                f_px = torch.tensor([P2[0, 0]], device=device)

            inv_depth = canonical_inv_depth * (orig_w / f_px)
            depth = 1.0 / torch.clamp(inv_depth, min=1e-4, max=1e4)

        pred_depth = depth.squeeze().cpu().numpy()
        times.append(time.time() - t0)

        # Resize prediction to original resolution
        if pred_depth.shape != (orig_h, orig_w):
            pred_pil = Image.fromarray(pred_depth)
            pred_pil = pred_pil.resize((orig_w, orig_h), Image.BILINEAR)
            pred_depth = np.array(pred_pil)

        # Apply Garg crop to prediction
        pred_crop = garg_crop(pred_depth)

        # Compute metrics
        m = compute_metrics(pred_crop, gt_crop)
        if m is None:
            continue

        for k in metrics_accum:
            metrics_accum[k] += m[k]
        scales.append(m["scale"])
        count += 1

    # Average
    for k in metrics_accum:
        metrics_accum[k] /= max(count, 1)

    metrics_accum["num_samples"] = count
    metrics_accum["avg_inference_time"] = float(np.mean(times)) if times else 0
    metrics_accum["mean_scale"] = float(np.mean(scales)) if scales else 0
    metrics_accum["use_median_scaling"] = use_median_scaling

    return metrics_accum


def main():
    parser = argparse.ArgumentParser(description="Evaluate on KITTI Eigen test split")
    parser.add_argument("--data-path", type=str, default="datasets/kitti_raw")
    parser.add_argument("--split", type=str, default="splits/eigen_test_files.txt")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (None = pretrained baseline)")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.checkpoint:
        print(f"Loading self-supervised model from {args.checkpoint}")
        model, transform = load_model(
            device, args.checkpoint, args.lora_rank, args.lora_alpha
        )
        label = "Self-supervised LoRA"
    else:
        print("Evaluating pretrained Depth Pro baseline")
        model, transform = depth_pro.create_model_and_transforms(device=device)
        model.eval()
        label = "Pretrained baseline"

    print(f"\nEvaluating: {label}")
    print(f"Data: {args.data_path}")
    print(f"Split: {args.split}")

    results = evaluate(model, args.data_path, args.split, device)

    print(f"\n{'='*50}")
    print(f"Results: {label}")
    print(f"{'='*50}")
    print(f"  Samples:      {results['num_samples']}")
    print(f"  AbsRel:       {results['abs_rel']:.4f}")
    print(f"  SqRel:        {results['sq_rel']:.4f}")
    print(f"  RMSE:         {results['rmse']:.4f}")
    print(f"  RMSE_log:     {results['rmse_log']:.4f}")
    print(f"  delta < 1.25: {results['delta1']:.4f}")
    print(f"  delta < 1.56: {results['delta2']:.4f}")
    print(f"  delta < 1.95: {results['delta3']:.4f}")
    print(f"  Mean scale:   {results['mean_scale']:.4f}")
    print(f"  Avg time:     {results['avg_inference_time']*1000:.1f}ms")

    # Save results
    output_path = args.output or (
        "eval_kitti_selfsup.json" if args.checkpoint else "eval_kitti_pretrained.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
