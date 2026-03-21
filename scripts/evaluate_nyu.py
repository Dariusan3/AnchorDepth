#!/usr/bin/env python3
"""Evaluate Depth Pro on NYU Depth V2 test set (Eigen split, 654 images).

Standard metrics:
  - AbsRel: mean(|pred - gt| / gt)
  - SqRel:  mean((pred - gt)^2 / gt)
  - RMSE:   sqrt(mean((pred - gt)^2))
  - RMSElog: sqrt(mean((log(pred) - log(gt))^2))
  - delta < 1.25^k for k=1,2,3
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import depth_pro

# Eigen test split indices (654 images from the 1449 labeled set).
# Standard split from: https://cs.nyu.edu/~deigen/dnl/
EIGEN_TEST_INDICES_URL = "eigen_test_indices.json"

# NYU Depth V2 camera intrinsics (from the Kinect sensor)
NYU_FOCAL_LENGTH_MM = 518.8579  # focal length in pixels for 640x480


def get_eigen_test_indices():
    """Return the 654 Eigen test split indices. If file doesn't exist, use standard set."""
    indices_file = Path(__file__).parent / EIGEN_TEST_INDICES_URL
    if indices_file.exists():
        with open(indices_file) as f:
            return json.load(f)
    # Standard Eigen test split - using all 1449 images if indices not available
    # We'll generate the standard split
    return None


def compute_metrics(pred, gt, min_depth=1e-3, max_depth=10.0):
    """Compute standard depth estimation metrics."""
    # Apply depth range mask
    mask = (gt > min_depth) & (gt < max_depth)
    pred = pred[mask]
    gt = gt[mask]

    if len(gt) == 0:
        return None

    # Threshold accuracies
    thresh = np.maximum(pred / gt, gt / pred)
    delta1 = (thresh < 1.25).mean()
    delta2 = (thresh < 1.25 ** 2).mean()
    delta3 = (thresh < 1.25 ** 3).mean()

    # Error metrics
    abs_rel = np.mean(np.abs(pred - gt) / gt)
    sq_rel = np.mean(((pred - gt) ** 2) / gt)
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))

    # Log10 error
    log10_err = np.mean(np.abs(np.log10(pred) - np.log10(gt)))

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "log10": log10_err,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
    }


def compute_scale_shift(pred, gt, mask):
    """Compute optimal scale and shift via least squares (for scale-invariant eval)."""
    pred_masked = pred[mask].flatten()
    gt_masked = gt[mask].flatten()
    A = np.stack([pred_masked, np.ones_like(pred_masked)], axis=1)
    result = np.linalg.lstsq(A, gt_masked, rcond=None)
    scale, shift = result[0]
    return scale, shift


def main():
    parser = argparse.ArgumentParser(description="Evaluate Depth Pro on NYU Depth V2")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/nyu_depth_v2_labeled.mat",
        help="Path to nyu_depth_v2_labeled.mat",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Limit number of samples for quick testing"
    )
    parser.add_argument(
        "--output", type=str, default="eval_results_nyu.json", help="Output JSON file for results"
    )
    parser.add_argument(
        "--scale-invariant", action="store_true", help="Also compute scale-invariant metrics"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a fine-tuned checkpoint (default: use original pretrained weights)",
    )
    parser.add_argument(
        "--tta", type=str, default=None, choices=["flip", "multiscale", "full"],
        help="Test-time augmentation mode (default: disabled)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Depth Pro - NYU Depth V2 Evaluation")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, transform = depth_pro.create_model_and_transforms(device=device)

    if args.checkpoint:
        print(f"Loading fine-tuned checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        torch.cuda.empty_cache()

    model.eval()
    print(f"Model loaded on {device}")

    # Load dataset
    print(f"\nLoading dataset from {args.dataset_path}...")
    f = h5py.File(args.dataset_path, "r")
    images = f["images"]  # (1449, 3, 640, 480) uint8
    depths = f["depths"]  # (1449, 640, 480) float32

    # Get test indices
    eigen_indices = get_eigen_test_indices()
    if eigen_indices is None:
        # Use all samples (or generate standard Eigen split)
        indices = list(range(images.shape[0]))
        print(f"Using all {len(indices)} images (Eigen split file not found)")
    else:
        indices = eigen_indices
        print(f"Using {len(indices)} Eigen test split images")

    if args.max_samples:
        indices = indices[: args.max_samples]
        print(f"Limited to {len(indices)} samples")

    # Evaluate
    all_metrics = []
    all_metrics_si = []
    inference_times = []

    print(f"\nRunning evaluation on {len(indices)} images...")
    for idx in tqdm(indices, desc="Evaluating"):
        # Load image: (3, 640, 480) -> transpose to (480, 640, 3) HWC
        img = images[idx]  # (3, 640, 480) uint8
        img = np.transpose(img, (2, 1, 0))  # (480, 640, 3) - note: NYU mat is transposed

        # Load ground truth depth
        gt_depth = depths[idx]  # (640, 480)
        gt_depth = gt_depth.T  # (480, 640) to match image orientation

        # Convert to PIL for the transform pipeline
        img_pil = Image.fromarray(img.astype(np.uint8))

        # Apply model transform
        img_tensor = transform(img_pil)

        # NYU focal length in pixels (for 640x480 resolution)
        f_px = torch.tensor(NYU_FOCAL_LENGTH_MM, dtype=torch.float32)

        # Run inference
        t0 = time.time()
        if args.tta:
            from depth_pro.improvements.tta import tta_infer
            prediction = tta_infer(model, img_tensor, f_px=f_px, mode=args.tta)
        else:
            with torch.no_grad():
                prediction = model.infer(img_tensor, f_px=f_px)
        inference_time = time.time() - t0
        inference_times.append(inference_time)

        # Get predicted depth
        pred_depth = prediction["depth"].cpu().numpy()

        # Resize prediction to match GT if needed
        if pred_depth.shape != gt_depth.shape:
            from PIL import Image as PILImage

            pred_pil = PILImage.fromarray(pred_depth)
            pred_pil = pred_pil.resize(
                (gt_depth.shape[1], gt_depth.shape[0]), PILImage.BILINEAR
            )
            pred_depth = np.array(pred_pil)

        # Compute metrics (metric depth)
        metrics = compute_metrics(pred_depth, gt_depth, min_depth=1e-3, max_depth=10.0)
        if metrics is not None:
            all_metrics.append(metrics)

        # Scale-invariant metrics
        if args.scale_invariant:
            mask = (gt_depth > 1e-3) & (gt_depth < 10.0)
            if mask.sum() > 0:
                scale, shift = compute_scale_shift(pred_depth, gt_depth, mask)
                pred_aligned = pred_depth * scale + shift
                pred_aligned = np.clip(pred_aligned, 1e-3, 10.0)
                si_metrics = compute_metrics(pred_aligned, gt_depth, min_depth=1e-3, max_depth=10.0)
                if si_metrics is not None:
                    all_metrics_si.append(si_metrics)

    f.close()

    # Aggregate results
    print("\n" + "=" * 60)
    print("RESULTS - Metric Depth Evaluation")
    print("=" * 60)

    results = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        results[key] = float(np.mean(values))

    print(f"\n{'Metric':<12} {'Value':>10}")
    print("-" * 24)
    print(f"{'AbsRel':<12} {results['abs_rel']:>10.4f}")
    print(f"{'SqRel':<12} {results['sq_rel']:>10.4f}")
    print(f"{'RMSE':<12} {results['rmse']:>10.4f}")
    print(f"{'RMSElog':<12} {results['rmse_log']:>10.4f}")
    print(f"{'log10':<12} {results['log10']:>10.4f}")
    print(f"{'delta<1.25':<12} {results['delta1']:>10.4f}")
    print(f"{'delta<1.25²':<12} {results['delta2']:>10.4f}")
    print(f"{'delta<1.25³':<12} {results['delta3']:>10.4f}")

    avg_time = np.mean(inference_times)
    print(f"\n{'Avg time':<12} {avg_time:>10.3f}s")
    print(f"{'Samples':<12} {len(all_metrics):>10d}")

    results["avg_inference_time"] = float(avg_time)
    results["num_samples"] = len(all_metrics)

    if args.scale_invariant and all_metrics_si:
        print("\n" + "=" * 60)
        print("RESULTS - Scale-Invariant (aligned) Evaluation")
        print("=" * 60)

        si_results = {}
        for key in all_metrics_si[0].keys():
            values = [m[key] for m in all_metrics_si]
            si_results[key] = float(np.mean(values))

        print(f"\n{'Metric':<12} {'Value':>10}")
        print("-" * 24)
        for key in ["abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3"]:
            print(f"{key:<12} {si_results[key]:>10.4f}")

        results["scale_invariant"] = si_results

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
