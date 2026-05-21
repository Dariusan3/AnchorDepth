#!/usr/bin/env python3
"""Fine-tune Depth Pro on NYU Depth V2 training set.

Strategy for RTX 4070 Ti (12GB VRAM):
  - Freeze encoder backbone (627M params) — only train decoder (20M) + head (0.5M)
  - Mixed precision (FP16) training
  - Gradient accumulation (effective batch size 4)
  - Combined loss: scale-invariant log + gradient matching

Usage:
  python train_nyu.py --epochs 25 --lr 1e-4
  python train_nyu.py --epochs 25 --lr 1e-4 --unfreeze-encoder-layers 2  # also fine-tune last 2 encoder blocks
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

import depth_pro
from depth_pro.depth_pro import DepthProConfig, DEFAULT_MONODEPTH_CONFIG_DICT


# NYU camera intrinsics
NYU_FOCAL_LENGTH_PX = 518.8579


class NYUDepthDataset(Dataset):
    """NYU Depth V2 dataset from the labeled .mat file."""

    def __init__(self, mat_path, indices, augment=True):
        self.mat_path = mat_path
        self.indices = indices
        self.augment = augment
        # Open file handle (h5py supports concurrent reads)
        self.f = h5py.File(mat_path, "r")
        self.images = self.f["images"]
        self.depths = self.f["depths"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        mat_idx = self.indices[idx]

        # Load image: (3, 640, 480) -> (480, 640, 3)
        img = self.images[mat_idx]
        img = np.transpose(img, (2, 1, 0))  # HWC

        # Load depth: (640, 480) -> (480, 640)
        depth = self.depths[mat_idx].T

        # Data augmentation
        if self.augment:
            # Random horizontal flip
            if np.random.random() > 0.5:
                img = np.ascontiguousarray(img[:, ::-1, :])
                depth = np.ascontiguousarray(depth[:, ::-1])

            # Random color jitter (brightness, contrast)
            if np.random.random() > 0.5:
                brightness = np.random.uniform(0.8, 1.2)
                img = np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

        # Convert to tensor and normalize (same as depth_pro transform but CPU-safe)
        img_tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor - 0.5) / 0.5  # Normalize to [-1, 1]

        # Depth to tensor
        depth_tensor = torch.from_numpy(depth.copy()).float()

        return img_tensor, depth_tensor

    def __del__(self):
        if hasattr(self, "f"):
            self.f.close()


class ScaleInvariantLogLoss(nn.Module):
    """Scale-invariant logarithmic loss from Eigen et al."""

    def __init__(self, si_lambda=0.5):
        super().__init__()
        self.si_lambda = si_lambda

    def forward(self, pred, target, mask):
        log_diff = torch.log(pred[mask]) - torch.log(target[mask])
        loss = torch.mean(log_diff ** 2) - self.si_lambda * (torch.mean(log_diff) ** 2)
        return loss


class GradientMatchingLoss(nn.Module):
    """Gradient matching loss for sharper edges."""

    def forward(self, pred, target, mask):
        # Compute gradients
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        # Masks for gradients
        mask_dx = mask[:, :, :, 1:] & mask[:, :, :, :-1]
        mask_dy = mask[:, :, 1:, :] & mask[:, :, :-1, :]

        loss_dx = torch.mean(torch.abs(pred_dx[mask_dx] - target_dx[mask_dx]))
        loss_dy = torch.mean(torch.abs(pred_dy[mask_dy] - target_dy[mask_dy]))

        return loss_dx + loss_dy


def freeze_encoder(model, unfreeze_last_n=0):
    """Freeze encoder, optionally unfreeze last N transformer blocks."""
    # Freeze all encoder parameters
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Optionally unfreeze last N blocks of patch encoder
    if unfreeze_last_n > 0:
        blocks = list(model.encoder.patch_encoder.blocks)
        for block in blocks[-unfreeze_last_n:]:
            for param in block.parameters():
                param.requires_grad = True
        print(f"Unfroze last {unfreeze_last_n} patch encoder blocks")

    # Freeze FOV head entirely (we use known focal length for NYU)
    if hasattr(model, "fov"):
        for param in model.fov.parameters():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.1f}%)")


def train_one_epoch(
    model, loader, optimizer, scaler, si_loss_fn, grad_loss_fn,
    grad_accum_steps, device, epoch, grad_loss_weight=0.5,
):
    model.train()
    # Keep encoder in eval mode (frozen batchnorm/dropout)
    model.encoder.eval()
    if hasattr(model, "fov"):
        model.fov.eval()

    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for step, (images, gt_depths) in enumerate(pbar):
        images = images.to(device)
        gt_depths = gt_depths.to(device)

        # Resize input to model's expected size (1536x1536)
        B, C, H, W = images.shape
        images_resized = F.interpolate(
            images, size=(model.img_size, model.img_size),
            mode="bilinear", align_corners=False,
        )

        with torch.cuda.amp.autocast():
            # Forward pass - get canonical inverse depth
            canonical_inverse_depth, _ = model(images_resized)

            # Convert to depth using known focal length
            # inverse_depth = canonical_inverse_depth * (W / f_px)
            inverse_depth = canonical_inverse_depth * (W / NYU_FOCAL_LENGTH_PX)

            # Resize prediction to GT resolution
            inverse_depth = F.interpolate(
                inverse_depth, size=(H, W), mode="bilinear", align_corners=False,
            )
            pred_depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)

            # Create valid mask
            gt_depths_4d = gt_depths.unsqueeze(1)  # (B, 1, H, W)
            mask = (gt_depths_4d > 1e-3) & (gt_depths_4d < 10.0)

            # Compute losses
            loss_si = si_loss_fn(pred_depth, gt_depths_4d, mask)
            loss_grad = grad_loss_fn(
                torch.log(torch.clamp(pred_depth, min=1e-4)),
                torch.log(torch.clamp(gt_depths_4d, min=1e-4)),
                mask,
            )
            loss = loss_si + grad_loss_weight * loss_grad
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        num_batches += 1
        pbar.set_postfix(loss=f"{total_loss/num_batches:.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, loader, device):
    """Quick validation computing AbsRel and delta<1.25."""
    model.eval()
    all_abs_rel = []
    all_delta1 = []

    for images, gt_depths in loader:
        images = images.to(device)
        gt_depths = gt_depths.to(device)

        B, C, H, W = images.shape
        images_resized = F.interpolate(
            images, size=(model.img_size, model.img_size),
            mode="bilinear", align_corners=False,
        )

        with torch.cuda.amp.autocast():
            canonical_inverse_depth, _ = model(images_resized)
            inverse_depth = canonical_inverse_depth * (W / NYU_FOCAL_LENGTH_PX)
            inverse_depth = F.interpolate(
                inverse_depth, size=(H, W), mode="bilinear", align_corners=False,
            )
            pred_depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)

        gt_4d = gt_depths.unsqueeze(1)
        mask = (gt_4d > 1e-3) & (gt_4d < 10.0)

        pred = pred_depth[mask].float()
        gt = gt_4d[mask].float()

        if len(gt) == 0:
            continue

        abs_rel = torch.mean(torch.abs(pred - gt) / gt).item()
        thresh = torch.max(pred / gt, gt / pred)
        delta1 = (thresh < 1.25).float().mean().item()

        all_abs_rel.append(abs_rel)
        all_delta1.append(delta1)

    return {
        "abs_rel": np.mean(all_abs_rel),
        "delta1": np.mean(all_delta1),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Depth Pro on NYU Depth V2")
    parser.add_argument("--dataset-path", type=str, default="datasets/nyu_depth_v2_labeled.mat")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--grad-loss-weight", type=float, default=0.5)
    parser.add_argument("--unfreeze-encoder-layers", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--eval-every", type=int, default=5, help="Validate every N epochs")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("=" * 60)
    print("Depth Pro - Fine-tuning on NYU Depth V2")
    print("=" * 60)
    print(f"Config: lr={args.lr}, epochs={args.epochs}, "
          f"batch={args.batch_size}, accum={args.grad_accum}, "
          f"effective_batch={args.batch_size * args.grad_accum}")
    print(f"Encoder unfrozen layers: {args.unfreeze_encoder_layers}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model with pretrained weights
    print("\nLoading pretrained model...")
    model, transform = depth_pro.create_model_and_transforms(device=device)

    # Freeze encoder
    freeze_encoder(model, unfreeze_last_n=args.unfreeze_encoder_layers)

    # Enable gradient checkpointing for memory savings
    if hasattr(model.encoder.patch_encoder, "set_grad_checkpointing"):
        model.encoder.patch_encoder.set_grad_checkpointing(True)

    # Load dataset indices
    eigen_indices_path = Path(__file__).parent / "eigen_test_indices.json"
    if eigen_indices_path.exists():
        with open(eigen_indices_path) as f:
            test_indices = set(json.load(f))
        train_indices = [i for i in range(1449) if i not in test_indices]
    else:
        # Fallback: use first 795 as train, rest as test
        train_indices = list(range(795))
        test_indices = set(range(795, 1449))

    # Use a small validation set from training for monitoring
    np.random.seed(42)
    np.random.shuffle(train_indices)
    val_indices = train_indices[:50]
    train_indices_final = train_indices[50:]

    print(f"\nTrain: {len(train_indices_final)}, Val: {len(val_indices)}, "
          f"Test: {len(test_indices)}")

    # Create datasets
    train_dataset = NYUDepthDataset(args.dataset_path, train_indices_final, augment=True)
    val_dataset = NYUDepthDataset(args.dataset_path, val_indices, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True,
    )

    # Loss functions
    si_loss_fn = ScaleInvariantLogLoss(si_lambda=0.5)
    grad_loss_fn = GradientMatchingLoss()

    # Optimizer — only trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    best_abs_rel = float("inf")
    training_log = []

    print(f"\nStarting training...")
    print(f"Steps per epoch: {len(train_loader)}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")

    # Time estimation
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler,
            si_loss_fn, grad_loss_fn,
            grad_accum_steps=args.grad_accum,
            device=device, epoch=epoch,
            grad_loss_weight=args.grad_loss_weight,
        )
        scheduler.step()

        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - t_start
        eta = (total_elapsed / epoch) * (args.epochs - epoch)

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": scheduler.get_last_lr()[0],
            "epoch_time": epoch_time,
        }

        # Validate periodically
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = validate(model, val_loader, device)
            log_entry.update(val_metrics)

            print(f"Epoch {epoch}/{args.epochs} | loss: {train_loss:.4f} | "
                  f"val_absrel: {val_metrics['abs_rel']:.4f} | "
                  f"val_d1: {val_metrics['delta1']:.4f} | "
                  f"time: {epoch_time:.1f}s | ETA: {eta/60:.1f}min")

            # Save best model
            if val_metrics["abs_rel"] < best_abs_rel:
                best_abs_rel = val_metrics["abs_rel"]
                save_path = Path(args.save_dir) / "depth_pro_finetuned_best.pt"
                torch.save(model.state_dict(), save_path)
                print(f"  -> Saved best model (AbsRel: {best_abs_rel:.4f})")
        else:
            print(f"Epoch {epoch}/{args.epochs} | loss: {train_loss:.4f} | "
                  f"time: {epoch_time:.1f}s | ETA: {eta/60:.1f}min")

        training_log.append(log_entry)
        torch.cuda.empty_cache()

    # Save final model
    final_path = Path(args.save_dir) / "depth_pro_finetuned_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved to {final_path}")

    # Save training log
    log_path = Path(args.save_dir) / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nTotal training time: {total_time/60:.1f} minutes")
    print(f"Best validation AbsRel: {best_abs_rel:.4f}")
    print(f"\nGPU peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
