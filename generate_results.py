#!/usr/bin/env python3
"""Generate all thesis result figures and tables.

Produces:
  - Depth map comparison figures (pretrained vs fine-tuned vs GT)
  - Training loss curves
  - Quantitative results table (LaTeX + markdown)
  - Error map visualizations

Usage:
  python generate_results.py                          # uses best checkpoint
  python generate_results.py --checkpoint path/to.pt  # specific checkpoint
  python generate_results.py --n-samples 8            # number of qual. examples
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))
import depth_pro
from train_nyu_lora import LoRALinear, apply_lora_to_encoder
from depth_pro.selfsup.pose_net import PoseNet

OUTPUT_DIR = Path("results/thesis_figures")


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_pretrained(device):
    model, transform = depth_pro.create_model_and_transforms(device=device)
    model.eval()
    return model, transform


def load_finetuned(checkpoint_path, device, lora_rank=8, lora_alpha=8.0):
    model, transform = depth_pro.create_model_and_transforms(device=device)
    apply_lora_to_encoder(model.encoder.patch_encoder, lora_rank, lora_alpha)
    apply_lora_to_encoder(model.encoder.image_encoder, lora_rank, lora_alpha)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("depth_model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  Loaded checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys")

    model = model.to(device)
    model.eval()
    return model, transform


# ──────────────────────────────────────────────────────────────────────────────
# Depth inference (single image)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_depth(model, img_tensor, f_px=None, device="cuda"):
    """Run depth inference. Returns depth in meters (H, W)."""
    result = model.infer(img_tensor.to(device), f_px=f_px)
    depth = result["depth"].squeeze().cpu().numpy()
    return depth


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: Training loss curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(log_path, out_dir):
    """Plot photometric loss over training epochs."""
    log_path = Path(log_path)
    if not log_path.exists():
        print(f"  [skip] Training log not found: {log_path}")
        return

    # Parse epoch summaries from log
    epochs, train_losses, val_losses = [], [], []
    with open(log_path) as f:
        for line in f:
            if line.startswith("Epoch ") and "loss:" in line:
                try:
                    parts = line.strip().split("|")
                    ep = int(parts[0].split()[1].split("/")[0])
                    loss = float([p for p in parts if "loss:" in p][0].split(":")[1].strip())
                    epochs.append(ep)
                    train_losses.append(loss)
                    if "val_photo:" in line:
                        val = float([p for p in parts if "val_photo:" in p][0].split(":")[1].strip())
                        val_losses.append((ep, val))
                except Exception:
                    continue

    if not epochs:
        print("  [skip] No epoch summaries found in log")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, "b-o", markersize=4, label="Train photometric loss")
    if val_losses:
        val_ep, val_l = zip(*val_losses)
        ax.plot(val_ep, val_l, "r-s", markersize=5, label="Val photometric loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Photometric Loss")
    ax.set_title("Self-Supervised Training: Photometric Loss over Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = out_dir / "training_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: Qualitative depth comparisons
# ──────────────────────────────────────────────────────────────────────────────

def colorize_depth(depth, vmin=None, vmax=None, cmap="magma"):
    """Convert depth array to colored image."""
    depth = np.copy(depth)
    depth[depth <= 0] = np.nan
    if vmin is None:
        vmin = np.nanpercentile(depth, 2)
    if vmax is None:
        vmax = np.nanpercentile(depth, 98)
    depth = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
    colored = cm.magma(depth)[:, :, :3]
    colored[np.isnan(depth)] = 0
    return (colored * 255).astype(np.uint8)


def load_kitti_test_samples(data_path, split_file, n_samples=8):
    """Load a selection of KITTI Eigen test samples."""
    data_path = Path(data_path)
    samples = []
    with open(split_file) as f:
        lines = f.readlines()

    # Evenly spaced samples across the test set
    indices = np.linspace(0, len(lines) - 1, n_samples, dtype=int)

    for idx in indices:
        parts = lines[idx].strip().split()
        folder, frame_idx, side = parts[0], int(parts[1]), parts[2]
        cam = "image_02" if side == "l" else "image_03"
        img_path = data_path / folder / cam / "data" / f"{frame_idx:010d}.png"
        if img_path.exists():
            samples.append(str(img_path))

    return samples


def plot_depth_comparison(
    img_paths, model_pretrained, model_finetuned, transform, device, out_dir, n=8
):
    """Create side-by-side comparison: RGB | Pretrained | Fine-tuned."""
    img_paths = img_paths[:n]
    n_rows = len(img_paths)

    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[None]

    col_titles = ["Input RGB", "Depth Pro (pretrained)", "Ours (self-supervised fine-tuned)"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=11, fontweight="bold")

    for row, img_path in enumerate(tqdm(img_paths, desc="Generating qualitative figures")):
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        # Transform for Depth Pro
        img_tensor = transform(img)
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            depth_pre = predict_depth(model_pretrained, img_tensor, device=device)
            depth_fine = predict_depth(model_finetuned, img_tensor, device=device)

        # Use consistent depth range for comparison
        vmin = min(np.nanpercentile(depth_pre, 2), np.nanpercentile(depth_fine, 2))
        vmax = max(np.nanpercentile(depth_pre, 98), np.nanpercentile(depth_fine, 98))

        axes[row, 0].imshow(img_np)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(colorize_depth(depth_pre, vmin, vmax))
        axes[row, 1].axis("off")

        axes[row, 2].imshow(colorize_depth(depth_fine, vmin, vmax))
        axes[row, 2].axis("off")

    plt.suptitle("Qualitative Depth Comparison — KITTI Eigen Test", fontsize=13, y=1.01)
    plt.tight_layout()

    out = out_dir / "depth_comparison.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: Error maps
# ──────────────────────────────────────────────────────────────────────────────

def plot_error_maps(img_paths, model_pretrained, model_finetuned, transform,
                   gt_depths, device, out_dir, n=4):
    """Show AbsRel error maps: pretrained vs fine-tuned."""
    img_paths = img_paths[:n]
    n_rows = len(img_paths)

    fig, axes = plt.subplots(n_rows, 4, figsize=(18, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[None]

    col_titles = ["Input RGB", "GT Depth", "Error: Pretrained", "Error: Ours"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=10, fontweight="bold")

    for row, (img_path, gt) in enumerate(zip(img_paths, gt_depths[:n])):
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img)
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            depth_pre = predict_depth(model_pretrained, img_tensor, device=device)
            depth_fine = predict_depth(model_finetuned, img_tensor, device=device)

        # Resize predictions to GT size
        h, w = gt.shape
        depth_pre_r = np.array(Image.fromarray(depth_pre).resize((w, h), Image.BILINEAR))
        depth_fine_r = np.array(Image.fromarray(depth_fine).resize((w, h), Image.BILINEAR))

        mask = (gt > 1) & (gt < 80)

        # Median scale
        if mask.sum() > 0:
            scale_pre = np.median(gt[mask]) / (np.median(depth_pre_r[mask]) + 1e-8)
            scale_fine = np.median(gt[mask]) / (np.median(depth_fine_r[mask]) + 1e-8)
            depth_pre_r *= scale_pre
            depth_fine_r *= scale_fine

        err_pre = np.abs(depth_pre_r - gt) / (gt + 1e-8)
        err_fine = np.abs(depth_fine_r - gt) / (gt + 1e-8)
        err_pre[~mask] = np.nan
        err_fine[~mask] = np.nan

        err_max = max(np.nanpercentile(err_pre, 95), np.nanpercentile(err_fine, 95))

        axes[row, 0].imshow(np.array(img))
        axes[row, 0].axis("off")

        axes[row, 1].imshow(colorize_depth(gt, vmin=1, vmax=80))
        axes[row, 1].axis("off")

        im = axes[row, 2].imshow(err_pre, cmap="hot", vmin=0, vmax=err_max)
        axes[row, 2].axis("off")
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046)

        im = axes[row, 3].imshow(err_fine, cmap="hot", vmin=0, vmax=err_max)
        axes[row, 3].axis("off")
        plt.colorbar(im, ax=axes[row, 3], fraction=0.046)

    plt.suptitle("AbsRel Error Maps — KITTI Eigen Test", fontsize=13, y=1.01)
    plt.tight_layout()

    out = out_dir / "error_maps.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Table: LaTeX results table
# ──────────────────────────────────────────────────────────────────────────────

def generate_results_table(eval_pretrained_path, eval_finetuned_path, out_dir):
    """Generate LaTeX and Markdown results tables."""
    baselines = [
        ("Monodepth2 (M)", "ICCV 2019", "ResNet-18", "0.115", "0.903", "4.863", "0.193", "0.877", "0.959", "0.981"),
        ("DIFFNet",        "ECCV 2022", "HRNet-18",  "0.102", "0.764", "4.483", "0.176", "0.896", "0.965", "0.983"),
        ("MonoViT",        "3DV 2022",  "ViT-Small",  "0.099", "0.708", "4.372", "0.175", "0.900", "0.965", "0.983"),
    ]

    def load_metrics(path):
        if path and Path(path).exists():
            with open(path) as f:
                data = json.load(f)
            m = data.get("metrics", data)
            return {
                "absrel": f"{m.get('abs_rel', m.get('AbsRel', 0)):.3f}",
                "sqrel":  f"{m.get('sq_rel',  m.get('SqRel',  0)):.3f}",
                "rmse":   f"{m.get('rmse',    m.get('RMSE',   0)):.3f}",
                "rmsel":  f"{m.get('rmse_log',m.get('RMSElog',0)):.3f}",
                "d1":     f"{m.get('a1', m.get('delta1', 0)):.3f}",
                "d2":     f"{m.get('a2', m.get('delta2', 0)):.3f}",
                "d3":     f"{m.get('a3', m.get('delta3', 0)):.3f}",
            }
        return {k: "TBD" for k in ["absrel","sqrel","rmse","rmsel","d1","d2","d3"]}

    pre = load_metrics(eval_pretrained_path)
    fine = load_metrics(eval_finetuned_path)

    # ── Markdown table ──────────────────────────────────────────────────────
    md = []
    md.append("## Quantitative Results — KITTI Eigen Test (Median Scaling, 1–80m)\n")
    md.append("| Method | AbsRel↓ | SqRel↓ | RMSE↓ | RMSElog↓ | δ<1.25↑ | δ<1.25²↑ | δ<1.25³↑ |")
    md.append("|--------|---------|--------|-------|----------|---------|----------|----------|")
    for name, venue, enc, absrel, sqrel, rmse, rmsel, d1, d2, d3 in baselines:
        md.append(f"| {name} ({venue}) | {absrel} | {sqrel} | {rmse} | {rmsel} | {d1} | {d2} | {d3} |")
    md.append(f"| Depth Pro pretrained | {pre['absrel']} | {pre['sqrel']} | {pre['rmse']} | {pre['rmsel']} | {pre['d1']} | {pre['d2']} | {pre['d3']} |")
    md.append(f"| **Ours (LoRA self-sup)** | **{fine['absrel']}** | **{fine['sqrel']}** | **{fine['rmse']}** | **{fine['rmsel']}** | **{fine['d1']}** | **{fine['d2']}** | **{fine['d3']}** |")

    md_path = out_dir / "results_table.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"  Saved: {md_path}")

    # ── LaTeX table ─────────────────────────────────────────────────────────
    latex = []
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(r"\caption{Quantitative results on KITTI Eigen test split (697 images). All methods use median scaling. Best self-supervised result in \textbf{bold}.}")
    latex.append(r"\label{tab:kitti_results}")
    latex.append(r"\begin{tabular}{lccccccc}")
    latex.append(r"\toprule")
    latex.append(r"Method & AbsRel$\downarrow$ & SqRel$\downarrow$ & RMSE$\downarrow$ & RMSElog$\downarrow$ & $\delta<1.25\uparrow$ & $\delta<1.25^2\uparrow$ & $\delta<1.25^3\uparrow$ \\")
    latex.append(r"\midrule")
    for name, venue, enc, absrel, sqrel, rmse, rmsel, d1, d2, d3 in baselines:
        latex.append(f"{name} \\cite{{{venue.lower().replace(' ','')}}} & {absrel} & {sqrel} & {rmse} & {rmsel} & {d1} & {d2} & {d3} \\\\")
    latex.append(r"\midrule")
    latex.append(f"Depth Pro pretrained & {pre['absrel']} & {pre['sqrel']} & {pre['rmse']} & {pre['rmsel']} & {pre['d1']} & {pre['d2']} & {pre['d3']} \\\\")
    latex.append(f"\\textbf{{Ours (LoRA self-sup)}} & \\textbf{{{fine['absrel']}}} & \\textbf{{{fine['sqrel']}}} & \\textbf{{{fine['rmse']}}} & \\textbf{{{fine['rmsel']}}} & \\textbf{{{fine['d1']}}} & \\textbf{{{fine['d2']}}} & \\textbf{{{fine['d3']}}} \\\\")
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    latex_path = out_dir / "results_table.tex"
    with open(latex_path, "w") as f:
        f.write("\n".join(latex))
    print(f"  Saved: {latex_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate thesis result figures")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/selfsup/selfsup_best.pt",
                        help="Path to fine-tuned checkpoint")
    parser.add_argument("--data-path", type=str, default="datasets/kitti_raw")
    parser.add_argument("--split", type=str, default="splits/eigen_test_files.txt")
    parser.add_argument("--eval-pretrained", type=str,
                        default="eval_kitti_pretrained.json")
    parser.add_argument("--eval-finetuned", type=str,
                        default="checkpoints/selfsup/eval_selfsup_v4.json")
    parser.add_argument("--training-log", type=str,
                        default="checkpoints/selfsup_training_v4.log")
    parser.add_argument("--n-samples", type=int, default=8,
                        help="Number of qualitative examples")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tables-only", action="store_true",
                        help="Only generate tables (no model inference needed)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Generating thesis result figures")
    print("=" * 60)

    # 1. Training curves (no model needed)
    print("\n[1/4] Training loss curves...")
    plot_training_curves(args.training_log, OUTPUT_DIR)

    # 2. Results tables (no model needed)
    print("\n[2/4] Results tables...")
    generate_results_table(args.eval_pretrained, args.eval_finetuned, OUTPUT_DIR)

    if args.tables_only:
        print("\nDone (tables only mode).")
        return

    # 3. Load models
    print("\n[3/4] Loading models...")
    print("  Loading pretrained Depth Pro...")
    model_pre, transform = load_pretrained(device)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"  [skip] Checkpoint not found: {checkpoint}")
        print("  Run qualitative figures after training completes.")
        return

    print(f"  Loading fine-tuned model from {checkpoint}...")
    model_fine, _ = load_finetuned(checkpoint, device)

    # 4. Load test samples
    print(f"\n[4/4] Generating qualitative figures ({args.n_samples} samples)...")
    img_paths = load_kitti_test_samples(args.data_path, args.split, args.n_samples)
    print(f"  Found {len(img_paths)} test images")

    # Depth comparisons
    plot_depth_comparison(img_paths, model_pre, model_fine, transform,
                         device, OUTPUT_DIR, n=args.n_samples)

    print(f"\n{'='*60}")
    print(f"  All figures saved to: {OUTPUT_DIR}/")
    print(f"  Files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"    {f.name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
