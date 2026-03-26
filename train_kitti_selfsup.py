#!/usr/bin/env python3
"""Self-supervised monocular depth estimation with Depth Pro + LoRA on KITTI.

Adapts the pretrained Depth Pro model using self-supervised photometric
reprojection loss (Monodepth2-style) on KITTI raw monocular sequences.
Uses LoRA for parameter-efficient encoder adaptation.

Architecture:
  - DepthNet: Depth Pro (frozen encoder + LoRA rank-8 + trainable decoder/head)
  - PoseNet: ResNet-18 based ego-motion estimator (trained from scratch)

Losses (Godard et al., ICCV 2019):
  - Photometric reprojection (L1 + SSIM)
  - Per-pixel minimum reprojection across source frames
  - Auto-masking for static pixels
  - Edge-aware smoothness regularization

Usage:
  python train_kitti_selfsup.py --epochs 20 --data-path datasets/kitti_raw
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

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
):
    """Train one epoch of self-supervised depth estimation."""
    model.train()
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

        with torch.amp.autocast("cuda"):
            # 1. Predict depth from target frame
            canonical_inv_depth, _ = model(target_depth_input)
            # Resize inverse depth to pose/loss resolution
            inv_depth = F.interpolate(
                canonical_inv_depth, size=(pose_h, pose_w),
                mode="bilinear", align_corners=False,
            )
            inv_depth = F.relu(inv_depth) + 1e-6  # ensure positive
            depth = 1.0 / inv_depth

            # 2. Predict ego-motion with PoseNet
            # Normalize for ImageNet-pretrained ResNet
            pose_target = normalize_imagenet(target_img)
            pose_prev = normalize_imagenet(source_prev)
            pose_next = normalize_imagenet(source_next)

            pose_vec_prev = pose_net(pose_target, pose_prev)  # (B, 6)
            pose_vec_next = pose_net(pose_target, pose_next)  # (B, 6)

            # Convert to 4x4 transformation matrices
            T_prev = pose_vec_to_matrix(
                pose_vec_prev[:, :3], pose_vec_prev[:, 3:]
            )
            T_next = pose_vec_to_matrix(
                pose_vec_next[:, :3], pose_vec_next[:, 3:]
            )

            # 3. Warp source images to target view
            warped_prev = warper(source_prev, depth, T_prev, K, inv_K)
            warped_next = warper(source_next, depth, T_next, K, inv_K)

            # 4. Compute self-supervised loss
            losses = compute_selfsup_loss(
                target_img=target_img,
                source_imgs=[source_prev, source_next],
                warped_imgs=[warped_prev, warped_next],
                inv_depth=inv_depth,
                smoothness_weight=smoothness_weight,
            )

            loss = losses["total"] / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            # Clip gradients for both networks
            all_params = (
                [p for p in model.parameters() if p.requires_grad]
                + list(pose_net.parameters())
            )
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += losses["total"].item()
        total_photo += losses["photometric"].item()
        total_smooth += losses["smoothness"].item()
        total_mask_ratio += losses["auto_mask_ratio"].item()
        num_batches += 1

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
def validate_selfsup(model, pose_net, warper, val_loader, device):
    """Validate using photometric reconstruction quality."""
    model.eval()
    pose_net.eval()

    total_photo = 0.0
    count = 0

    for batch in tqdm(val_loader, desc="Validation"):
        target_depth_input = batch["target_depth"].to(device)
        target_img = batch["target"].to(device)
        source_prev = batch["source_-1"].to(device)
        source_next = batch["source_1"].to(device)
        K = batch["K"].to(device)
        inv_K = batch["inv_K"].to(device)

        pose_h, pose_w = target_img.shape[2], target_img.shape[3]

        with torch.amp.autocast("cuda"):
            canonical_inv_depth, _ = model(target_depth_input)
            inv_depth = F.interpolate(
                canonical_inv_depth, size=(pose_h, pose_w),
                mode="bilinear", align_corners=False,
            )
            inv_depth = F.relu(inv_depth) + 1e-6
            depth = 1.0 / inv_depth

            pose_target = normalize_imagenet(target_img)
            pose_prev = normalize_imagenet(source_prev)
            pose_next = normalize_imagenet(source_next)

            pose_vec_prev = pose_net(pose_target, pose_prev)
            pose_vec_next = pose_net(pose_target, pose_next)

            T_prev = pose_vec_to_matrix(pose_vec_prev[:, :3], pose_vec_prev[:, 3:])
            T_next = pose_vec_to_matrix(pose_vec_next[:, :3], pose_vec_next[:, 3:])

            warped_prev = warper(source_prev, depth, T_prev, K, inv_K)
            warped_next = warper(source_next, depth, T_next, K, inv_K)

            losses = compute_selfsup_loss(
                target_img, [source_prev, source_next],
                [warped_prev, warped_next], inv_depth,
            )

        total_photo += losses["photometric"].item()
        count += 1

    return {"val_photometric": total_photo / max(count, 1)}


def main():
    parser = argparse.ArgumentParser(description="Self-supervised Depth Pro on KITTI")
    parser.add_argument("--data-path", type=str, default="datasets/kitti_raw")
    parser.add_argument("--train-split", type=str, default="splits/eigen_train_files.txt")
    parser.add_argument("--val-split", type=str, default="splits/eigen_val_files.txt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr-depth", type=float, default=1e-4,
                        help="LR for decoder + head")
    parser.add_argument("--lr-lora", type=float, default=1e-5,
                        help="LR for LoRA encoder params")
    parser.add_argument("--lr-pose", type=float, default=1e-4,
                        help="LR for PoseNet")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--smoothness-weight", type=float, default=1e-3)
    parser.add_argument("--pose-size", type=str, default="640x192",
                        help="WxH for PoseNet input")
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    pose_w, pose_h = [int(x) for x in args.pose_size.split("x")]

    print("=" * 60)
    print("Self-Supervised Depth Pro + LoRA on KITTI")
    print("=" * 60)
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"LR: depth={args.lr_depth}, lora={args.lr_lora}, pose={args.lr_pose}")
    print(f"Pose resolution: {pose_w}x{pose_h}")
    print(f"Smoothness weight: {args.smoothness_weight}")
    print(f"Epochs: {args.epochs}, warmup: {args.warmup_epochs}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ================================================================
    # Load Depth Pro model
    # ================================================================
    print("\nLoading pretrained Depth Pro...")
    model, transform = depth_pro.create_model_and_transforms(device=device)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA to encoder attention layers
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
    # Create PoseNet
    # ================================================================
    print("\nCreating PoseNet...")
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
    # VRAM optimization
    # ================================================================
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # ================================================================
    # Dataset
    # ================================================================
    print(f"\nLoading KITTI dataset from {args.data_path}...")
    train_dataset = KITTIRawDataset(
        data_path=args.data_path,
        split_file=args.train_split,
        depth_size=(1536, 1536),
        pose_size=(pose_w, pose_h),
        is_train=True,
    )
    val_dataset = KITTIRawDataset(
        data_path=args.data_path,
        split_file=args.val_split,
        depth_size=(1536, 1536),
        pose_size=(pose_w, pose_h),
        is_train=False,
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
    decoder_head_params = (
        list(model.decoder.parameters())
        + list(model.head.parameters())
    )

    param_groups = [
        {"params": lora_params, "lr": args.lr_lora, "weight_decay": 0.01},
        {"params": decoder_head_params, "lr": args.lr_depth, "weight_decay": 1e-4},
        {"params": list(pose_net.parameters()), "lr": args.lr_pose, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups)

    # Cosine annealing after warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6,
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

        train_metrics = train_one_epoch(
            model, pose_net, warper, train_loader, optimizer, scaler,
            grad_accum_steps=args.grad_accum,
            device=device, epoch=epoch,
            smoothness_weight=args.smoothness_weight,
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
            )
            log_entry.update(val_metrics)

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"loss: {train_metrics['loss']:.4f} | "
                f"photo: {train_metrics['photometric']:.4f} | "
                f"val_photo: {val_metrics['val_photometric']:.4f} | "
                f"mask: {train_metrics['auto_mask_ratio']:.2f} | "
                f"time: {epoch_time:.1f}s | ETA: {eta/60:.1f}min"
            )

            if val_metrics["val_photometric"] < best_photo_loss:
                best_photo_loss = val_metrics["val_photometric"]
                # Save both depth model and pose network
                save_dict = {
                    "depth_model": model.state_dict(),
                    "pose_net": pose_net.state_dict(),
                    "epoch": epoch,
                    "val_photometric": best_photo_loss,
                }
                torch.save(save_dict, save_dir / "selfsup_best.pt")
                print(f"  -> Saved best model (val_photo: {best_photo_loss:.4f})")
        else:
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"loss: {train_metrics['loss']:.4f} | "
                f"photo: {train_metrics['photometric']:.4f} | "
                f"mask: {train_metrics['auto_mask_ratio']:.2f} | "
                f"time: {epoch_time:.1f}s | ETA: {eta/60:.1f}min"
            )

        training_log.append(log_entry)
        torch.cuda.empty_cache()

    # Save final model and log
    save_dict = {
        "depth_model": model.state_dict(),
        "pose_net": pose_net.state_dict(),
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


if __name__ == "__main__":
    main()
