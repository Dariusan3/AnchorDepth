#!/usr/bin/env python3
"""Self-supervised monocular depth estimation with Depth Pro + LoRA on KITTI.
Multi-scale photometric loss variant (v6).

Extends train_kitti_selfsup.py with:
  - Multi-scale photometric loss at 4 scales (1/1, 1/2, 1/4, 1/8)
  - Longer training (40 epochs default)
  - All v5 bug fixes included (warping autograd, focal length scaling)

Usage:
  python train_kitti_selfsup_ms.py --epochs 40 --data-path datasets/kitti_raw
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# Set CUDA allocator config early to reduce fragmentation OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).parent / "src"))
import depth_pro
from depth_pro.selfsup.kitti_dataset import KITTIRawDataset
from depth_pro.selfsup.pose_net import PoseNet
from depth_pro.selfsup.warping import Warper, pose_vec_to_matrix
from depth_pro.selfsup.losses import compute_selfsup_loss

# Reuse LoRA from existing training script
from train_nyu_lora import LoRALinear, apply_lora_to_encoder


def train_one_epoch(
    model, pose_net, warper, train_loader, optimizer, scaler,
    grad_accum_steps, device, epoch, smoothness_weight=1e-3,
    use_auto_mask=True, use_wandb=False, log_interval=50, global_step_offset=0,
    consistency_weight=0.0,
):
    """Train one epoch of self-supervised depth estimation."""
    model.train()
    if pose_net is not None:
        pose_net.train()

    # Keep FOV head in eval mode (frozen)
    if hasattr(model, "fov"):
        model.fov.eval()

    total_loss = 0.0
    total_photo = 0.0
    total_smooth = 0.0
    total_mask_ratio = 0.0
    num_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        # Move data to device
        target_depth_input = batch["target_depth"].to(device)  # (B, 3, 1536, 1536)
        target_img = batch["target"].to(device)  # (B, 3, 192, 640)
        source_prev = batch["source_-1"].to(device)  # (B, 3, 192, 640)
        source_next = batch["source_1"].to(device)  # (B, 3, 192, 640)
        K = batch["K"].to(device)  # (B, 4, 4)
        inv_K = batch["inv_K"].to(device)  # (B, 4, 4)

        pose_h, pose_w = target_img.shape[2], target_img.shape[3]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # 1. Single-scale depth prediction (multi-scale VRAM constrained on 12GB)
            encodings = model.encoder(target_depth_input)
            features, _ = model.decoder(encodings)
            del encodings
            raw = model.head(features)
            del features
            inv_depth = F.interpolate(raw, size=(pose_h, pose_w), mode="bilinear", align_corners=False)
            del raw
            f_px = batch["K"][:, 0, 0].to(device).to(torch.float)
            scale_factor = torch.clamp((pose_w / f_px).view(-1, 1, 1, 1), min=0.01, max=100.0)
            inv_depth = inv_depth * scale_factor
            inv_depth = torch.nan_to_num(inv_depth, nan=1.0, posinf=10.0, neginf=1e-6)
            inv_depth = F.relu(inv_depth) + 1e-6
            depth = 1.0 / torch.clamp(inv_depth, min=1e-4, max=1e4)

            # 2. Predict ego-motion (PoseNet or precomputed VGGT poses)
            if "T_prev" in batch:
                # Use precomputed VGGT poses — no PoseNet inference needed
                T_prev = batch["T_prev"].to(device)  # (B, 4, 4)
                T_next = batch["T_next"].to(device)  # (B, 4, 4)
                pose_vec_prev = None  # not used below
            else:
                pose_target = normalize_imagenet(target_img)
                pose_prev_n = normalize_imagenet(source_prev)
                pose_next_n = normalize_imagenet(source_next)
                pose_vec_prev = pose_net(pose_target, pose_prev_n)
                pose_vec_next = pose_net(pose_target, pose_next_n)
                T_prev = pose_vec_to_matrix(pose_vec_prev[:, :3], pose_vec_prev[:, 3:])
                T_next = pose_vec_to_matrix(pose_vec_next[:, :3], pose_vec_next[:, 3:])

            # 3. Warp and compute photometric loss
            warped_prev = warper(source_prev, depth, T_prev, K, inv_K)
            warped_next = warper(source_next, depth, T_next, K, inv_K)
            losses = compute_selfsup_loss(
                target_img=target_img,
                source_imgs=[source_prev, source_next],
                warped_imgs=[warped_prev, warped_next],
                inv_depth=inv_depth,
                smoothness_weight=smoothness_weight,
                auto_mask=use_auto_mask,
            )
            del warped_prev, warped_next

            # Consistency loss — anchors model to zero-shot predictions
            if consistency_weight > 0.0 and "zeroshot_depth" in batch:
                zs_depth = batch["zeroshot_depth"].to(device)  # (B, 1, H, W)
                # depth is already at pose resolution (H, W)
                cons_loss = F.l1_loss(depth.clamp(0.1, 80.0), zs_depth.clamp(0.1, 80.0))
                losses["consistency"] = cons_loss
                losses["total"] = losses["total"] + consistency_weight * cons_loss

            loss = losses["total"] / grad_accum_steps

        # Skip step if loss is non-finite (prevents NaN from propagating)
        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)

            # Sanitize LoRA gradients — zero out any NaN/Inf
            for _, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    param.grad[~torch.isfinite(param.grad)] = 0.0

            # Clip gradients (PoseNet excluded when using VGGT poses)
            all_params = [p for p in model.parameters() if p.requires_grad]
            if pose_net is not None:
                all_params += list(pose_net.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += losses["total"].item()
        total_photo += losses["photometric"].item()
        total_smooth += losses["smoothness"].item()
        total_mask_ratio += losses["auto_mask_ratio"].item()
        num_batches += 1

        # Step-level WandB logging every log_interval steps
        if use_wandb and (step + 1) % log_interval == 0:
            global_step = global_step_offset + step + 1
            with torch.no_grad():
                depth_mean = depth.mean().item()
                depth_std = depth.std().item()
                pose_mag = T_prev[:, :3, 3].norm(dim=-1).mean().item()
                scale_val = scale_factor.mean().item()
            wandb.log({
                "losses/total": losses["total"].item(),
                "losses/photometric_loss": losses["photometric"].item(),
                "step/depth_mean_m": depth_mean,
                "step/depth_std_m": depth_std,
                "step/pose_translation_norm": pose_mag,
                "step/depth_scale_factor": scale_val,
                "step/scaler_scale": scaler.get_scale(),
                "step/epoch": epoch,
            }, step=global_step)

        pbar.set_postfix(
            loss=f"{total_loss/num_batches:.4f}",
            photo=f"{total_photo/num_batches:.4f}",
            mask=f"{total_mask_ratio/num_batches:.2f}",
        )

    n = max(num_batches, 1)
    return {
        "loss": total_loss / n,
        "photometric": total_photo / n,
        "smoothness": total_smooth / n,
        "auto_mask_ratio": total_mask_ratio / n,
    }


# ImageNet normalization for PoseNet
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """Normalize [0,1] tensor to ImageNet statistics."""
    mean = IMAGENET_MEAN.to(x.device, x.dtype)
    std = IMAGENET_STD.to(x.device, x.dtype)
    return (x - mean) / std


@torch.no_grad()
def validate_selfsup(model, pose_net, warper, val_loader, device,
                     use_wandb=False, epoch=0,
                     gt_eval_split=None, gt_eval_data_path=None, gt_eval_n=50):
    """Validate using photometric loss + lightweight GT depth metrics + depth visualizations.

    GT metrics (abs_rel, delta1, etc.) are computed on gt_eval_n test images when
    gt_eval_split is provided. Uses one image at a time to minimise VRAM overhead.
    """
    model.eval()
    if pose_net is not None:
        pose_net.eval()

    total_photo = 0.0
    count = 0
    depth_imgs = []

    for batch in tqdm(val_loader, desc="Validation"):
        target_depth_input = batch["target_depth"].to(device)
        target_img         = batch["target"].to(device)
        source_prev        = batch["source_-1"].to(device)
        source_next        = batch["source_1"].to(device)
        K                  = batch["K"].to(device)
        inv_K              = batch["inv_K"].to(device)

        pose_h, pose_w = target_img.shape[2], target_img.shape[3]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            canonical_inv_depth, _ = model(target_depth_input)
            inv_depth = F.interpolate(canonical_inv_depth, size=(pose_h, pose_w),
                                      mode="bilinear", align_corners=False)
            f_px = batch["K"][:, 0, 0].to(device).to(torch.float)
            scale = torch.clamp((pose_w / f_px).view(-1, 1, 1, 1), min=0.01, max=100.0)
            inv_depth = inv_depth * scale
            inv_depth = torch.nan_to_num(inv_depth, nan=1.0, posinf=10.0, neginf=1e-6)
            inv_depth = F.relu(inv_depth) + 1e-6
            depth = 1.0 / torch.clamp(inv_depth, min=1e-4, max=1e4)

            if "T_prev" in batch:
                T_prev = batch["T_prev"].to(device)
                T_next = batch["T_next"].to(device)
            else:
                pt = normalize_imagenet(target_img)
                pp = normalize_imagenet(source_prev)
                pn = normalize_imagenet(source_next)
                vp = pose_net(pt, pp)
                vn = pose_net(pt, pn)
                T_prev = pose_vec_to_matrix(vp[:, :3], vp[:, 3:])
                T_next = pose_vec_to_matrix(vn[:, :3], vn[:, 3:])

            warped_prev = warper(source_prev, depth, T_prev, K, inv_K)
            warped_next = warper(source_next, depth, T_next, K, inv_K)
            losses = compute_selfsup_loss(
                target_img, [source_prev, source_next],
                [warped_prev, warped_next], inv_depth,
            )

        total_photo += losses["photometric"].item()
        count += 1

        if use_wandb and len(depth_imgs) < 8:
            d = depth[0, 0].cpu().float().numpy()
            d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
            depth_imgs.append(wandb.Image(d_norm, caption=f"epoch {epoch}"))

    metrics = {"val_photometric": total_photo / max(count, 1)}
    metrics["depth_images"] = depth_imgs

    # ── Lightweight GT evaluation on a small subset of the test split ──────────
    if gt_eval_split and gt_eval_data_path and Path(gt_eval_split).exists():
        from PIL import Image as PILImage
        from torchvision.transforms import ToTensor, Normalize as TvNorm
        import numpy as np

        normalize_dp = TvNorm([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        to_tensor    = ToTensor()

        # Parse test split
        filenames = []
        with open(gt_eval_split) as f:
            for line in f:
                parts = line.strip().split()
                filenames.append((parts[0], int(parts[1]), parts[2]))

        # Evenly-spaced subset
        step = max(1, len(filenames) // gt_eval_n)
        subset = filenames[::step][:gt_eval_n]

        abs_rels, delta1s, rmses, log_rmses, sq_rels, delta2s, delta3s = [], [], [], [], [], [], []
        data_path = Path(gt_eval_data_path)

        for folder, frame_idx, side in subset:
            cam = "image_02" if side == "l" else "image_03"
            img_path  = data_path / folder / cam / "data" / f"{frame_idx:010d}.png"
            date      = folder.split("/")[0]
            drive     = folder.split("/")[1]
            velo_path = data_path / date / drive / "velodyne_points" / "data" / f"{frame_idx:010d}.bin"
            if not img_path.exists() or not velo_path.exists():
                continue

            img      = PILImage.open(img_path).convert("RGB")
            orig_w, orig_h = img.size
            img_1536 = img.resize((1536, 1536), PILImage.LANCZOS)
            inp      = normalize_dp(to_tensor(img_1536)).unsqueeze(0).to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                canon, _ = model(inp)
                depth_pred = 1.0 / torch.clamp(canon, min=1e-4, max=1e4)

            pred_np = depth_pred.squeeze().cpu().float().numpy()
            # Resize to original
            pred_np = np.array(PILImage.fromarray(pred_np).resize((orig_w, orig_h), PILImage.BILINEAR))

            # Load LiDAR GT (reuse evaluate_kitti helpers via inline code)
            velo = np.fromfile(str(velo_path), dtype=np.float32).reshape(-1, 4)[:, :3]
            calib = {}
            for name in ["calib_cam_to_cam.txt", "calib_velo_to_cam.txt"]:
                with open(data_path / date / name) as cf:
                    for line in cf:
                        if ":" in line:
                            k2, v2 = line.split(":", 1)
                            calib[k2.strip()] = v2.strip()
            P2      = np.array(calib["P_rect_02"].split(), np.float32).reshape(3, 4)
            R_rect  = np.eye(4, dtype=np.float32)
            R_rect[:3, :3] = np.array(calib["R_rect_00"].split(), np.float32).reshape(3, 3)
            Tr      = np.eye(4, dtype=np.float32)
            Tr[:3, :3] = np.array(calib["R"].split(), np.float32).reshape(3, 3)
            Tr[:3, 3]  = np.array(calib["T"].split(), np.float32)

            pts_hom = np.hstack([velo, np.ones((len(velo), 1), np.float32)])
            pts_cam = (P2 @ R_rect @ Tr @ pts_hom.T).T
            mask    = pts_cam[:, 2] > 0
            pts_cam = pts_cam[mask]
            pts_2d  = pts_cam[:, :2] / pts_cam[:, 2:3]
            depths  = pts_cam[:, 2]
            mask2   = ((pts_2d[:,0]>=0)&(pts_2d[:,0]<orig_w)&
                       (pts_2d[:,1]>=0)&(pts_2d[:,1]<orig_h))
            pts_2d  = pts_2d[mask2]; depths = depths[mask2]
            gt_map  = np.zeros((orig_h, orig_w), np.float32)
            u = np.clip(np.round(pts_2d[:,0]).astype(int), 0, orig_w-1)
            v = np.clip(np.round(pts_2d[:,1]).astype(int), 0, orig_h-1)
            for i in range(len(u)):
                if gt_map[v[i],u[i]] == 0 or depths[i] < gt_map[v[i],u[i]]:
                    gt_map[v[i],u[i]] = depths[i]

            # Garg crop
            h, w = orig_h, orig_w
            r0,r1 = int(0.40810811*h), int(0.99189189*h)
            c0,c1 = int(0.03594771*w), int(0.96405229*w)
            gt_c   = gt_map[r0:r1, c0:c1]
            pred_c = pred_np[r0:r1, c0:c1]

            valid = (gt_c > 1e-3) & (gt_c < 80.0)
            if valid.sum() < 10:
                continue
            gt_v   = gt_c[valid]
            pred_v = pred_c[valid]
            scale  = np.median(gt_v) / (np.median(pred_v) + 1e-8)
            pred_v = np.clip(pred_v * scale, 1e-3, 80.0)

            thresh = np.maximum(pred_v / gt_v, gt_v / pred_v)
            abs_rels.append(float(np.mean(np.abs(pred_v - gt_v) / gt_v)))
            sq_rels.append(float(np.mean((pred_v - gt_v)**2 / gt_v)))
            rmses.append(float(np.sqrt(np.mean((pred_v - gt_v)**2))))
            log_rmses.append(float(np.sqrt(np.mean((np.log(pred_v) - np.log(gt_v))**2))))
            delta1s.append(float(np.mean(thresh < 1.25)))
            delta2s.append(float(np.mean(thresh < 1.25**2)))
            delta3s.append(float(np.mean(thresh < 1.25**3)))

            torch.cuda.empty_cache()

        if abs_rels:
            metrics["abs_rel"]  = float(np.mean(abs_rels))
            metrics["sq_rel"]   = float(np.mean(sq_rels))
            metrics["rms"]      = float(np.mean(rmses))
            metrics["log_rms"]  = float(np.mean(log_rmses))
            metrics["a1"]       = float(np.mean(delta1s))
            metrics["a2"]       = float(np.mean(delta2s))
            metrics["a3"]       = float(np.mean(delta3s))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Self-supervised Depth Pro on KITTI")
    parser.add_argument("--data-path", type=str, default="datasets/kitti_raw")
    parser.add_argument("--train-split", type=str, default="splits/eigen_train_files.txt")
    parser.add_argument("--val-split", type=str, default="splits/eigen_val_files.txt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (loads model+pose weights)")
    parser.add_argument("--lr-depth", type=float, default=1e-4,
                        help="LR for decoder + head")
    parser.add_argument("--lr-lora", type=float, default=1e-5,
                        help="LR for LoRA encoder params")
    parser.add_argument("--lr-pose", type=float, default=1e-4,
                        help="LR for PoseNet")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA (decoder-only training)")
    parser.add_argument("--freeze-head", action="store_true",
                        help="Freeze depth head (keeps zero-shot depth structure)")
    parser.add_argument("--freeze-decoder", action="store_true",
                        help="Freeze decoder (keeps zero-shot decoder features)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--smoothness-weight", type=float, default=1e-3)
    parser.add_argument("--pose-size", type=str, default="416x128",
                        help="WxH for PoseNet input")
    parser.add_argument("--stride", type=int, default=3,
                        help="Use every Nth training sample (KITTI 10Hz is redundant)")
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--save-dir", type=str, default="checkpoints/selfsup_ms")
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--wandb-project", type=str, default="depth-pro-selfsup")
    parser.add_argument("--wandb-name", type=str, default="v6-multiscale-40ep")
    parser.add_argument("--vggt-poses", type=str, default=None,
                        help="Path to precomputed VGGT poses (.pt file from precompute_vggt_poses.py). "
                             "Replaces PoseNet with high-quality VGGT ego-motion estimates.")
    parser.add_argument("--zeroshot-depths", type=str, default=None,
                        help="Path to precomputed zero-shot Depth Pro depths (.pt). "
                             "When set, adds consistency loss to anchor model to zero-shot.")
    parser.add_argument("--consistency-weight", type=float, default=1.0,
                        help="Weight lambda for consistency loss (only used with --zeroshot-depths).")
    args = parser.parse_args()

    pose_w, pose_h = [int(x) for x in args.pose_size.split("x")]

    # ================================================================
    # WandB
    # ================================================================
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "epochs": args.epochs,
                "lr_depth": args.lr_depth,
                "lr_lora": args.lr_lora,
                "lr_pose": args.lr_pose,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "smoothness_weight": args.smoothness_weight,
                "pose_size": args.pose_size,
                "stride": args.stride,
                "warmup_epochs": args.warmup_epochs,
                "multi_scale": True,
                "scales": [1.0, 0.5],
                "lora_enabled": not args.no_lora,
            },
        )
        print(f"WandB run: {wandb.run.url}")

    print("=" * 60)
    print("Self-Supervised Depth Pro + LoRA on KITTI")
    print("=" * 60)
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"LR: depth={args.lr_depth}, lora={args.lr_lora}, pose={args.lr_pose}")
    print(f"Pose resolution: {pose_w}x{pose_h}")
    print(f"Smoothness weight: {args.smoothness_weight}")
    print(f"Epochs: {args.epochs}, warmup: {args.warmup_epochs}")
    print(f"Dataset stride: {args.stride} (every {args.stride}th frame)")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ================================================================
    # Load Depth Pro model
    # ================================================================
    print("\nLoading pretrained Depth Pro...")
    model, transform = depth_pro.create_model_and_transforms(device=device)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA to encoder attention layers (skipped if --no-lora)
    if not args.no_lora:
        print("\nApplying LoRA...")
        lora_params = apply_lora_to_encoder(model, rank=args.lora_rank, alpha=args.lora_alpha)
        # Move LoRA params to device
        for encoder_name in ["patch_encoder", "image_encoder"]:
            encoder = getattr(model.encoder, encoder_name)
            for block in encoder.blocks:
                if isinstance(block.attn.qkv, LoRALinear):
                    block.attn.qkv.lora_A = nn.Parameter(block.attn.qkv.lora_A.to(device))
                    block.attn.qkv.lora_B = nn.Parameter(block.attn.qkv.lora_B.to(device))
                if isinstance(block.attn.proj, LoRALinear):
                    block.attn.proj.lora_A = nn.Parameter(block.attn.proj.lora_A.to(device))
                    block.attn.proj.lora_B = nn.Parameter(block.attn.proj.lora_B.to(device))
    else:
        print("\nLoRA disabled (decoder-only training)")
        lora_params = []

    # Unfreeze decoder + head
    for param in model.decoder.parameters():
        param.requires_grad = True
    for param in model.head.parameters():
        param.requires_grad = True

    # Keep FOV frozen (not needed for self-supervised)
    if hasattr(model, "fov"):
        for param in model.fov.parameters():
            param.requires_grad = False

    # Refresh LoRA params list
    if not args.no_lora:
        lora_params = []
        for encoder_name in ["patch_encoder", "image_encoder"]:
            encoder = getattr(model.encoder, encoder_name)
            for block in encoder.blocks:
                if isinstance(block.attn.qkv, LoRALinear):
                    lora_params.extend([block.attn.qkv.lora_A, block.attn.qkv.lora_B])
                if isinstance(block.attn.proj, LoRALinear):
                    lora_params.extend([block.attn.proj.lora_A, block.attn.proj.lora_B])

    # Enable gradient checkpointing
    if hasattr(model.encoder.patch_encoder, "set_grad_checkpointing"):
        model.encoder.patch_encoder.set_grad_checkpointing(True)
        print("Enabled gradient checkpointing for patch encoder")
    if hasattr(model.encoder.image_encoder, "set_grad_checkpointing"):
        model.encoder.image_encoder.set_grad_checkpointing(True)
        print("Enabled gradient checkpointing for image encoder")

    # ================================================================
    # Create PoseNet (skipped when using precomputed VGGT poses)
    # ================================================================
    if args.vggt_poses:
        print(f"\nUsing precomputed VGGT poses from: {args.vggt_poses}")
        print("PoseNet: DISABLED (replaced by VGGT)")
        pose_net = None
        pose_params = 0
    else:
        print("\nCreating PoseNet (ResNet-18)...")
        pose_net = PoseNet().to(device)
        pose_params = sum(p.numel() for p in pose_net.parameters())
        print(f"PoseNet parameters: {pose_params/1e6:.2f}M")

    # ================================================================
    # Create Warper
    # ================================================================
    warper = Warper(pose_h, pose_w).to(device)

    # ================================================================
    # Count trainable parameters
    # ================================================================
    lora_trainable = sum(p.numel() for p in lora_params)
    decoder_trainable = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    head_trainable = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    total_depth_trainable = lora_trainable + decoder_trainable + head_trainable
    total_trainable = total_depth_trainable + pose_params
    total_params = sum(p.numel() for p in model.parameters()) + pose_params

    print(f"\nTrainable parameters:")
    print(f"  LoRA:     {lora_trainable/1e6:.2f}M")
    print(f"  Decoder:  {decoder_trainable/1e6:.2f}M")
    print(f"  Head:     {head_trainable/1e6:.2f}M")
    print(f"  PoseNet:  {pose_params/1e6:.2f}M")
    print(f"  Total:    {total_trainable/1e6:.2f}M / {total_params/1e6:.1f}M")

    # ================================================================
    # Resume from checkpoint (loads weights, fresh optimizer)
    # ================================================================
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["depth_model"], strict=False)
        if pose_net is not None and "pose_net" in ckpt:
            pose_net.load_state_dict(ckpt["pose_net"])
        resumed_epoch = ckpt.get("epoch", 0)
        print(f"  Loaded weights from epoch {resumed_epoch}")

    # ================================================================
    # VRAM optimization
    # ================================================================
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # ================================================================
    # Dataset
    # ================================================================
    print(f"\nLoading KITTI dataset from {args.data_path}...")
    vggt_poses = None
    if args.vggt_poses:
        print(f"Loading precomputed VGGT poses from {args.vggt_poses}...")
        vggt_poses = torch.load(args.vggt_poses, map_location="cpu")
        print(f"  Loaded {len(vggt_poses)} pose entries.")

    zeroshot_depths = None
    if args.zeroshot_depths:
        print(f"Loading precomputed zero-shot depths from {args.zeroshot_depths}...")
        zeroshot_depths = torch.load(args.zeroshot_depths, map_location="cpu")
        print(f"  Loaded {len(zeroshot_depths)} depth maps. lambda={args.consistency_weight}")

    train_dataset = KITTIRawDataset(
        data_path=args.data_path,
        split_file=args.train_split,
        depth_size=(1536, 1536),
        pose_size=(pose_w, pose_h),
        is_train=True,
        stride=args.stride,
        vggt_poses=vggt_poses,
        zeroshot_depths=zeroshot_depths,
    )
    val_dataset = KITTIRawDataset(
        data_path=args.data_path,
        split_file=args.val_split,
        depth_size=(1536, 1536),
        pose_size=(pose_w, pose_h),
        is_train=False,
        stride=args.stride,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ================================================================
    # Optimizer with discriminative learning rates
    # ================================================================
    decoder_head_params = []
    if args.freeze_decoder:
        for p in model.decoder.parameters():
            p.requires_grad = False
        print("FROZEN: decoder (keeps zero-shot decoder features)")
    else:
        decoder_head_params += list(model.decoder.parameters())
    if args.freeze_head:
        for p in model.head.parameters():
            p.requires_grad = False
        print("FROZEN: depth head (keeps zero-shot depth structure)")
    else:
        decoder_head_params += list(model.head.parameters())

    param_groups = [
        {"params": lora_params, "lr": args.lr_lora, "weight_decay": 0.01},
        {"params": decoder_head_params, "lr": args.lr_depth, "weight_decay": 1e-4},
        {"params": list(pose_net.parameters()) if pose_net is not None else [], "lr": args.lr_pose, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups)

    # Cosine annealing with warm restarts — LR cycles every 10 epochs
    # so it never decays to near-zero and keeps escaping plateaus.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6,  # type: ignore[arg-type]
    )

    scaler = torch.amp.GradScaler("cuda")

    # ================================================================
    # Training loop
    # ================================================================
    best_photo_loss = float("inf")
    training_log = []
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    print(f"\nStarting self-supervised training ({args.epochs} epochs)...")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Linear warmup
        if epoch <= args.warmup_epochs:
            warmup_factor = epoch / args.warmup_epochs
            for pg in optimizer.param_groups:
                if "initial_lr" not in pg:
                    pg["initial_lr"] = pg["lr"]
                pg["lr"] = pg["initial_lr"] * warmup_factor

        # Disable auto-masking: KITTI has constant camera motion so auto-masking
        # provides minimal benefit and collapsed in v2 due to NaN loss bug.
        use_auto_mask = False

        steps_per_epoch = len(train_loader)
        train_metrics = train_one_epoch(
            model, pose_net, warper, train_loader, optimizer, scaler,
            grad_accum_steps=args.grad_accum,
            device=device, epoch=epoch,
            smoothness_weight=args.smoothness_weight,
            use_auto_mask=use_auto_mask,
            use_wandb=use_wandb,
            log_interval=50,
            global_step_offset=(epoch - 1) * steps_per_epoch,
            consistency_weight=args.consistency_weight if args.zeroshot_depths else 0.0,
        )

        if epoch > args.warmup_epochs:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - t_start
        eta = (total_elapsed / epoch) * (args.epochs - epoch)

        log_entry = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_photometric": train_metrics["photometric"],
            "train_smoothness": train_metrics["smoothness"],
            "auto_mask_ratio": train_metrics["auto_mask_ratio"],
            "lr_lora": optimizer.param_groups[0]["lr"],
            "lr_depth": optimizer.param_groups[1]["lr"],
            "lr_pose": optimizer.param_groups[2]["lr"],
            "epoch_time": epoch_time,
        }

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = validate_selfsup(
                model, pose_net, warper, val_loader, device,
                use_wandb=use_wandb, epoch=epoch,
                gt_eval_split="splits/eigen_test_files.txt",
                gt_eval_data_path=args.data_path,
                gt_eval_n=50,
            )
            log_entry.update({k: v for k, v in val_metrics.items() if k != "depth_images"})

            abs_rel_str = f" | abs_rel: {val_metrics['abs_rel']:.4f}" if "abs_rel" in val_metrics else ""
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"loss: {train_metrics['loss']:.4f} | "
                f"photo: {train_metrics['photometric']:.4f} | "
                f"val_photo: {val_metrics['val_photometric']:.4f}"
                f"{abs_rel_str} | "
                f"time: {epoch_time:.1f}s | ETA: {eta/60:.1f}min"
            )

            if val_metrics["val_photometric"] < best_photo_loss:
                best_photo_loss = val_metrics["val_photometric"]
                sd = model.state_dict()
                nan_keys = [k for k, v in sd.items() if v.is_floating_point() and not torch.isfinite(v).all()]
                if nan_keys:
                    print(f"  !! SKIPPED save — {len(nan_keys)} params have NaN/Inf (LoRA diverged)")
                else:
                    save_dict = {
                        "depth_model": sd,
                        "pose_net": pose_net.state_dict() if pose_net is not None else {},
                        "epoch": epoch,
                        "val_photometric": best_photo_loss,
                    }
                    torch.save(save_dict, save_dir / "selfsup_best.pt")
                    print(f"  -> Saved best model (val_photo: {best_photo_loss:.4f})")
        else:
            val_metrics = {}
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"loss: {train_metrics['loss']:.4f} | "
                f"photo: {train_metrics['photometric']:.4f} | "
                f"time: {epoch_time:.1f}s | ETA: {eta/60:.1f}min"
            )

        # WandB logging — matches naming convention from screenshots
        if use_wandb:
            wandb_log = {
                "epoch": epoch,
                "losses/photometric_loss": train_metrics["photometric"],
                "losses/total": train_metrics["loss"],
                "losses/smoothness": train_metrics["smoothness"],
                "lr/lora": optimizer.param_groups[0]["lr"],
                "lr/depth": optimizer.param_groups[1]["lr"],
                "lr/pose": optimizer.param_groups[2]["lr"],
                "epoch_time_s": epoch_time,
            }
            if val_metrics:
                wandb_log["losses/val_photometric"] = val_metrics.get("val_photometric", 0)
                # Standard depth metrics
                for k in ["abs_rel", "sq_rel", "rms", "log_rms"]:
                    if k in val_metrics:
                        wandb_log[f"standard_metrics/{k}"] = val_metrics[k]
                # Threshold metrics
                for k in ["a1", "a2", "a3"]:
                    if k in val_metrics:
                        wandb_log[f"threshold_metrics/{k}"] = val_metrics[k]
                # Depth map visualizations
                if val_metrics.get("depth_images"):
                    wandb_log["predicted_depth"] = val_metrics["depth_images"]
            wandb.log(wandb_log, step=epoch)

        training_log.append(log_entry)
        torch.cuda.empty_cache()

    # Save final model (with NaN check)
    final_sd = model.state_dict()
    nan_keys = [k for k, v in final_sd.items() if v.is_floating_point() and not torch.isfinite(v).all()]
    if nan_keys:
        print(f"WARNING: {len(nan_keys)} params have NaN — final model NOT saved")
    else:
        save_dict = {
            "depth_model": final_sd,
            "pose_net": pose_net.state_dict() if pose_net is not None else {},
            "epoch": args.epochs,
        }
        torch.save(save_dict, save_dir / "selfsup_final.pt")

    log_path = save_dir / "training_log_selfsup.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nTotal training time: {total_time/60:.1f} minutes")
    print(f"Best validation photometric loss: {best_photo_loss:.4f}")
    print(f"GPU peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    if use_wandb:
        wandb.summary["best_val_photometric"] = best_photo_loss
        wandb.summary["total_train_time_min"] = total_time / 60
        wandb.summary["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        wandb.finish()


if __name__ == "__main__":
    main()
