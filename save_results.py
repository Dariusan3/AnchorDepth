#!/usr/bin/env python3
"""Run Depth Pro (pretrained & fine-tuned) on NYU test images and save visual results.

Saves for each image:
  - RGB input
  - Pretrained depth map (turbo colormap)
  - Fine-tuned depth map (turbo colormap)
  - Ground truth depth map (turbo colormap)
  - Side-by-side comparison panel
  - Raw depth as .npz

Usage:
  python save_results.py --num-samples 20
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import PIL.Image
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from tqdm import tqdm

import depth_pro

NYU_FOCAL_LENGTH_PX = 518.8579


def depth_to_colormap(depth, cmap_name="turbo", min_depth=0.1, max_depth=10.0):
    """Convert depth to turbo colormap image (same as Apple's CLI)."""
    inverse_depth = 1.0 / np.clip(depth, min_depth, max_depth)
    max_invdepth = min(inverse_depth.max(), 1.0 / min_depth)
    min_invdepth = max(1.0 / max_depth, inverse_depth.min())
    normalized = (inverse_depth - min_invdepth) / (max_invdepth - min_invdepth + 1e-8)
    cmap = plt.get_cmap(cmap_name)
    color_depth = (cmap(normalized)[..., :3] * 255).astype(np.uint8)
    return color_depth


def compute_error_map(pred, gt, min_depth=1e-3, max_depth=10.0):
    """Compute per-pixel absolute relative error map."""
    mask = (gt > min_depth) & (gt < max_depth)
    error = np.zeros_like(pred)
    error[mask] = np.abs(pred[mask] - gt[mask]) / gt[mask]
    return error, mask


def main():
    parser = argparse.ArgumentParser(description="Save Depth Pro results like Apple's CLI")
    parser.add_argument("--dataset-path", type=str, default="datasets/nyu_depth_v2_labeled.mat")
    parser.add_argument("--checkpoint-finetuned", type=str, default="checkpoints/depth_pro_finetuned_best.pt")
    parser.add_argument("--output-dir", type=str, default="output_results")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of test images to process")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)

    # Create output subdirectories
    (output_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (output_dir / "depth_pretrained").mkdir(parents=True, exist_ok=True)
    (output_dir / "depth_finetuned").mkdir(parents=True, exist_ok=True)
    (output_dir / "depth_gt").mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison").mkdir(parents=True, exist_ok=True)
    (output_dir / "error_maps").mkdir(parents=True, exist_ok=True)
    (output_dir / "npz").mkdir(parents=True, exist_ok=True)

    # Load test indices
    eigen_path = Path(__file__).parent / "eigen_test_indices.json"
    with open(eigen_path) as f:
        test_indices = json.load(f)

    # Select evenly spaced samples for diversity
    step = max(1, len(test_indices) // args.num_samples)
    selected = test_indices[::step][: args.num_samples]
    print(f"Processing {len(selected)} test images")

    # Load dataset
    f = h5py.File(args.dataset_path, "r")
    images = f["images"]
    depths = f["depths"]

    # ── Model 1: Pretrained ──
    print("\nLoading pretrained model...")
    model_pre, transform = depth_pro.create_model_and_transforms(device=device)
    model_pre.eval()

    print("Running pretrained inference...")
    pretrained_depths = {}
    for idx in tqdm(selected, desc="Pretrained"):
        img = np.transpose(images[idx], (2, 1, 0))  # HWC
        img_pil = PIL.Image.fromarray(img)
        img_tensor = transform(img_pil)
        f_px = torch.tensor(NYU_FOCAL_LENGTH_PX, dtype=torch.float32)

        with torch.no_grad():
            pred = model_pre.infer(img_tensor, f_px=f_px)
        pretrained_depths[idx] = pred["depth"].cpu().numpy()

    # Free pretrained model
    del model_pre
    torch.cuda.empty_cache()

    # ── Model 2: Fine-tuned ──
    print("\nLoading fine-tuned model...")
    model_ft, transform = depth_pro.create_model_and_transforms(device=device)
    state_dict = torch.load(args.checkpoint_finetuned, map_location="cpu")
    model_ft.load_state_dict(state_dict, strict=True)
    del state_dict
    torch.cuda.empty_cache()
    model_ft.eval()

    print("Running fine-tuned inference...")
    finetuned_depths = {}
    for idx in tqdm(selected, desc="Fine-tuned"):
        img = np.transpose(images[idx], (2, 1, 0))
        img_pil = PIL.Image.fromarray(img)
        img_tensor = transform(img_pil)
        f_px = torch.tensor(NYU_FOCAL_LENGTH_PX, dtype=torch.float32)

        with torch.no_grad():
            pred = model_ft.infer(img_tensor, f_px=f_px)
        finetuned_depths[idx] = pred["depth"].cpu().numpy()

    del model_ft
    torch.cuda.empty_cache()

    # ── Save results ──
    print("\nSaving results...")
    for i, idx in enumerate(tqdm(selected, desc="Saving")):
        img = np.transpose(images[idx], (2, 1, 0))
        gt_depth = depths[idx].T

        pred_pre = pretrained_depths[idx]
        pred_ft = finetuned_depths[idx]

        # Resize predictions to GT size if needed
        if pred_pre.shape != gt_depth.shape:
            pred_pre = np.array(
                PIL.Image.fromarray(pred_pre).resize(
                    (gt_depth.shape[1], gt_depth.shape[0]), PIL.Image.BILINEAR
                )
            )
        if pred_ft.shape != gt_depth.shape:
            pred_ft = np.array(
                PIL.Image.fromarray(pred_ft).resize(
                    (gt_depth.shape[1], gt_depth.shape[0]), PIL.Image.BILINEAR
                )
            )

        name = f"sample_{i:03d}_idx{idx}"

        # Save RGB
        PIL.Image.fromarray(img).save(output_dir / "rgb" / f"{name}.jpg", quality=95)

        # Save depth colormaps (turbo, like Apple)
        PIL.Image.fromarray(depth_to_colormap(pred_pre)).save(
            output_dir / "depth_pretrained" / f"{name}.jpg", quality=95
        )
        PIL.Image.fromarray(depth_to_colormap(pred_ft)).save(
            output_dir / "depth_finetuned" / f"{name}.jpg", quality=95
        )
        PIL.Image.fromarray(depth_to_colormap(gt_depth)).save(
            output_dir / "depth_gt" / f"{name}.jpg", quality=95
        )

        # Save raw depth as npz
        np.savez_compressed(
            output_dir / "npz" / f"{name}.npz",
            depth_pretrained=pred_pre,
            depth_finetuned=pred_ft,
            depth_gt=gt_depth,
        )

        # Error maps
        err_pre, mask = compute_error_map(pred_pre, gt_depth)
        err_ft, _ = compute_error_map(pred_ft, gt_depth)

        # ── Side-by-side comparison panel ──
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle(f"Depth Pro Comparison — Sample {i} (NYU idx {idx})", fontsize=16, fontweight="bold")

        # Row 1: RGB, GT, Error comparison
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("Input RGB", fontsize=13)
        axes[0, 0].axis("off")

        axes[0, 1].imshow(depth_to_colormap(gt_depth))
        axes[0, 1].set_title("Ground Truth Depth", fontsize=13)
        axes[0, 1].axis("off")

        # Error comparison bar
        abs_rel_pre = np.mean(np.abs(pred_pre[mask] - gt_depth[mask]) / gt_depth[mask])
        abs_rel_ft = np.mean(np.abs(pred_ft[mask] - gt_depth[mask]) / gt_depth[mask])
        bars = axes[0, 2].barh(
            ["Pretrained", "Fine-tuned"],
            [abs_rel_pre, abs_rel_ft],
            color=["#E74C3C", "#2ECC71"],
            alpha=0.85,
        )
        axes[0, 2].set_xlabel("AbsRel Error (lower is better)")
        axes[0, 2].set_title("Per-Image AbsRel", fontsize=13)
        for bar, val in zip(bars, [abs_rel_pre, abs_rel_ft]):
            axes[0, 2].text(
                bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontweight="bold",
            )
        axes[0, 2].set_xlim(0, max(abs_rel_pre, abs_rel_ft) * 1.3)

        # Row 2: Pretrained depth, Fine-tuned depth, Error difference
        axes[1, 0].imshow(depth_to_colormap(pred_pre))
        axes[1, 0].set_title(f"Pretrained (AbsRel: {abs_rel_pre:.4f})", fontsize=13, color="#E74C3C")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(depth_to_colormap(pred_ft))
        axes[1, 1].set_title(f"Fine-tuned (AbsRel: {abs_rel_ft:.4f})", fontsize=13, color="#2ECC71")
        axes[1, 1].axis("off")

        # Error difference map
        err_diff = err_pre - err_ft  # positive = fine-tuned is better
        vmax = max(np.abs(err_diff[mask]).max(), 0.1)
        im = axes[1, 2].imshow(err_diff, cmap="RdYlGn", vmin=-vmax, vmax=vmax)
        axes[1, 2].set_title("Error Improvement (green = fine-tuned better)", fontsize=11)
        axes[1, 2].axis("off")
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

        plt.tight_layout()
        fig.savefig(
            output_dir / "comparison" / f"{name}.png",
            dpi=120, bbox_inches="tight", facecolor="white",
        )
        plt.close(fig)

        # Individual error map
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig2.suptitle(f"Error Maps — Sample {i}", fontsize=14, fontweight="bold")
        im1 = ax1.imshow(err_pre, cmap="hot", vmin=0, vmax=0.3)
        ax1.set_title(f"Pretrained Error (AbsRel: {abs_rel_pre:.4f})", color="#E74C3C")
        ax1.axis("off")
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        im2 = ax2.imshow(err_ft, cmap="hot", vmin=0, vmax=0.3)
        ax2.set_title(f"Fine-tuned Error (AbsRel: {abs_rel_ft:.4f})", color="#2ECC71")
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        plt.tight_layout()
        fig2.savefig(
            output_dir / "error_maps" / f"{name}.png",
            dpi=120, bbox_inches="tight", facecolor="white",
        )
        plt.close(fig2)

    f.close()

    print(f"\nDone! Results saved to {output_dir}/")
    print(f"  rgb/              — {len(selected)} input images")
    print(f"  depth_pretrained/ — turbo colormapped depth (pretrained)")
    print(f"  depth_finetuned/  — turbo colormapped depth (fine-tuned)")
    print(f"  depth_gt/         — turbo colormapped ground truth")
    print(f"  comparison/       — side-by-side comparison panels")
    print(f"  error_maps/       — per-pixel error heatmaps")
    print(f"  npz/              — raw depth arrays")


if __name__ == "__main__":
    main()
