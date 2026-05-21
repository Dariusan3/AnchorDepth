"""Evaluate v3 Phase 1 model with all inference variants:
  1. v3 baseline (no TTA, no post-processing)
  2. v3 + TTA flip
  3. v3 + guided filter
  4. v3 + TTA flip + guided filter
  5. v3 + TTA full (flip + multiscale) + guided filter
"""

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))
import depth_pro
from depth_pro.improvements.tta import tta_infer
from depth_pro.improvements.postprocessing import guided_filter_depth


def compute_metrics(pred, gt, mask):
    p = pred[mask]
    g = gt[mask]
    thresh = np.maximum(p / g, g / p)
    abs_rel = np.mean(np.abs(p - g) / g)
    sq_rel = np.mean((p - g) ** 2 / g)
    rmse = np.sqrt(np.mean((p - g) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))
    log10 = np.mean(np.abs(np.log10(p) - np.log10(g)))
    d1 = np.mean(thresh < 1.25)
    d2 = np.mean(thresh < 1.25 ** 2)
    d3 = np.mean(thresh < 1.25 ** 3)
    return {
        "abs_rel": abs_rel, "sq_rel": sq_rel, "rmse": rmse,
        "rmse_log": rmse_log, "log10": log10,
        "delta1": d1, "delta2": d2, "delta3": d3,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print("Loading v3 Phase 1 model...")
    model, transform = depth_pro.create_model_and_transforms(device=device)
    ckpt = torch.load("checkpoints/depth_pro_v3_best.pt", map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    print(f"Model on {device}")

    # Load dataset
    print("Loading NYU dataset...")
    f = h5py.File("datasets/nyu_depth_v2_labeled.mat", "r")
    images = f["images"]
    depths = f["depths"]
    n_total = images.shape[0]

    # Load Eigen test indices
    eigen_file = Path("eigen_test_indices.json")
    if eigen_file.exists():
        with open(eigen_file) as ef:
            test_indices = json.load(ef)
        print(f"Using {len(test_indices)} Eigen test indices")
    else:
        test_indices = list(range(n_total))
        print(f"No Eigen indices, using all {n_total}")

    # Define evaluation variants
    variants = {
        "v3_baseline": {"tta": None, "guided": False},
        "v3_tta_flip": {"tta": "flip", "guided": False},
        "v3_guided": {"tta": None, "guided": True},
        "v3_tta_flip_guided": {"tta": "flip", "guided": True},
        "v3_tta_full_guided": {"tta": "full", "guided": True},
    }

    all_results = {}

    for vname, vcfg in variants.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {vname}")
        print(f"  TTA: {vcfg['tta'] or 'none'}, Guided filter: {vcfg['guided']}")
        print(f"{'='*60}")

        metrics_accum = {k: 0.0 for k in ["abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "delta1", "delta2", "delta3"]}
        count = 0
        times = []

        for idx in tqdm(test_indices, desc=f"  {vname}"):
            # Load image and depth
            rgb_raw = images[idx].transpose(2, 1, 0)  # (H, W, 3)
            gt_depth = depths[idx].T  # (H, W)

            img_pil = Image.fromarray(rgb_raw.astype(np.uint8))
            img_tensor = transform(img_pil)

            t0 = time.time()

            # Inference
            if vcfg["tta"]:
                prediction = tta_infer(model, img_tensor, f_px=None, mode=vcfg["tta"])
            else:
                prediction = model.infer(img_tensor, f_px=None)

            pred_depth = prediction["depth"].detach().cpu().numpy().squeeze()
            dt = time.time() - t0

            # Resize pred to GT size if needed
            if pred_depth.shape != gt_depth.shape:
                from PIL import Image as PILImage
                pred_pil = PILImage.fromarray(pred_depth)
                pred_pil = pred_pil.resize((gt_depth.shape[1], gt_depth.shape[0]), PILImage.BILINEAR)
                pred_depth = np.array(pred_pil)

            # Post-processing: guided filter
            if vcfg["guided"]:
                # Resize RGB to match depth if needed
                rgb_guide = rgb_raw
                if rgb_guide.shape[:2] != pred_depth.shape:
                    rgb_pil = Image.fromarray(rgb_guide.astype(np.uint8))
                    rgb_pil = rgb_pil.resize((pred_depth.shape[1], pred_depth.shape[0]), Image.BILINEAR)
                    rgb_guide = np.array(rgb_pil)
                pred_depth = guided_filter_depth(pred_depth, rgb_guide, radius=8, eps=0.01)

            times.append(dt)

            # Compute metrics
            mask = (gt_depth > 1e-3) & (pred_depth > 1e-3) & (gt_depth < 10.0)
            if mask.sum() < 100:
                continue

            pred_depth = np.clip(pred_depth, 1e-3, 10.0)
            m = compute_metrics(pred_depth, gt_depth, mask)
            for k in metrics_accum:
                metrics_accum[k] += m[k]
            count += 1

        # Average
        for k in metrics_accum:
            metrics_accum[k] /= count

        metrics_accum["avg_inference_time"] = np.mean(times)
        metrics_accum["num_samples"] = count
        all_results[vname] = metrics_accum

        print(f"\n  Results for {vname}:")
        print(f"  AbsRel:    {metrics_accum['abs_rel']:.4f}")
        print(f"  RMSE:      {metrics_accum['rmse']:.4f}")
        print(f"  delta<1.25: {metrics_accum['delta1']:.4f}")
        print(f"  Avg time:  {metrics_accum['avg_inference_time']:.3f}s")

    # Save all results
    with open("eval_results_v3_variants.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to eval_results_v3_variants.json")

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Variant':<30} {'AbsRel':>8} {'RMSE':>8} {'d<1.25':>8} {'Time':>8}")
    print(f"{'-'*80}")
    for vname, m in all_results.items():
        print(f"{vname:<30} {m['abs_rel']:>8.4f} {m['rmse']:>8.4f} {m['delta1']:>8.4f} {m['avg_inference_time']:>7.3f}s")


if __name__ == "__main__":
    main()
