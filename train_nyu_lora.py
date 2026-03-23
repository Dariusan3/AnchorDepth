#!/usr/bin/env python3
"""Phase 2: Fine-tune Depth Pro with LoRA on the DINOv2 encoder.

Key difference from v1: instead of only training decoder (20M params),
we also adapt the encoder's attention layers via Low-Rank Adaptation,
adding ~3.5M trainable params to the 627M frozen encoder.

Strategy for RTX 4070 Ti (12GB VRAM):
  - LoRA rank 16 on Q/K/V projections of both patch and image encoders
  - Decoder + head fully trainable
  - Discriminative learning rates: encoder LoRA 5e-5, decoder 1e-4
  - FP16 mixed precision + gradient checkpointing
  - Combined loss: SI-log + gradient matching (same as v1 that worked well)

Usage:
  python train_nyu_lora.py --epochs 30 --lora-rank 16
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


# ============================================================
# LoRA Implementation
# ============================================================

class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear.

    Adds a low-rank decomposition: output = W*x + (B @ A)*x * (alpha/rank)
    where A is (rank, in_features) and B is (out_features, rank).
    The original weight W is frozen; only A and B are trained.
    """

    def __init__(self, original_linear: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # LoRA matrices — A uses Kaiming init, B starts at zero
        # so the LoRA contribution starts at zero (no disruption to pretrained weights)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x):
        # Original output + low-rank adaptation
        result = self.original(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return result + lora_out


def apply_lora_to_encoder(model, rank=16, alpha=16.0):
    """Apply LoRA to all Q/K/V attention projections in both encoders.

    Targets: encoder.patch_encoder.blocks.*.attn.qkv
             encoder.image_encoder.blocks.*.attn.qkv
    """
    lora_params = []
    num_adapted = 0

    for encoder_name in ["patch_encoder", "image_encoder"]:
        encoder = getattr(model.encoder, encoder_name)
        for block_idx, block in enumerate(encoder.blocks):
            # Apply LoRA to the combined QKV projection
            if hasattr(block.attn, "qkv"):
                original_qkv = block.attn.qkv
                lora_qkv = LoRALinear(original_qkv, rank=rank, alpha=alpha)
                block.attn.qkv = lora_qkv
                lora_params.extend([lora_qkv.lora_A, lora_qkv.lora_B])
                num_adapted += 1

            # Also apply to output projection
            if hasattr(block.attn, "proj"):
                original_proj = block.attn.proj
                lora_proj = LoRALinear(original_proj, rank=rank, alpha=alpha)
                block.attn.proj = lora_proj
                lora_params.extend([lora_proj.lora_A, lora_proj.lora_B])
                num_adapted += 1

    lora_param_count = sum(p.numel() for p in lora_params)
    print(f"Applied LoRA (rank={rank}) to {num_adapted} layers")
    print(f"LoRA parameters: {lora_param_count/1e6:.2f}M")

    return lora_params


# ============================================================
# Dataset
# ============================================================

class NYUDepthDataset(Dataset):
    """NYU Depth V2 dataset from the labeled .mat file."""

    def __init__(self, mat_path, indices, augment=True):
        self.mat_path = mat_path
        self.indices = indices
        self.augment = augment
        self.f = h5py.File(mat_path, "r")
        self.images = self.f["images"]
        self.depths = self.f["depths"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        mat_idx = self.indices[idx]
        img = self.images[mat_idx]
        img = np.transpose(img, (2, 1, 0))  # HWC
        depth = self.depths[mat_idx].T

        if self.augment:
            if np.random.random() > 0.5:
                img = np.ascontiguousarray(img[:, ::-1, :])
                depth = np.ascontiguousarray(depth[:, ::-1])
            if np.random.random() > 0.5:
                brightness = np.random.uniform(0.8, 1.2)
                img = np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

        img_tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor - 0.5) / 0.5
        depth_tensor = torch.from_numpy(depth.copy()).float()
        return img_tensor, depth_tensor

    def __del__(self):
        if hasattr(self, "f"):
            self.f.close()


# ============================================================
# Loss functions (same as v1 — these worked best)
# ============================================================

class ScaleInvariantLogLoss(nn.Module):
    def __init__(self, si_lambda=0.5):
        super().__init__()
        self.si_lambda = si_lambda

    def forward(self, pred, target, mask):
        log_diff = torch.log(pred[mask]) - torch.log(target[mask])
        return torch.mean(log_diff ** 2) - self.si_lambda * (torch.mean(log_diff) ** 2)


class GradientMatchingLoss(nn.Module):
    def forward(self, pred, target, mask):
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        mask_dx = mask[:, :, :, 1:] & mask[:, :, :, :-1]
        mask_dy = mask[:, :, 1:, :] & mask[:, :, :-1, :]
        loss_dx = torch.mean(torch.abs(pred_dx[mask_dx] - target_dx[mask_dx]))
        loss_dy = torch.mean(torch.abs(pred_dy[mask_dy] - target_dy[mask_dy]))
        return loss_dx + loss_dy


# ============================================================
# Training loop
# ============================================================

def train_one_epoch(model, loader, optimizer, scaler, si_loss_fn, grad_loss_fn,
                    grad_accum_steps, device, epoch, grad_loss_weight=0.5):
    model.train()
    # Keep frozen parts of encoder in eval mode
    if hasattr(model, "fov"):
        model.fov.eval()

    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for step, (images, gt_depths) in enumerate(pbar):
        images = images.to(device)
        gt_depths = gt_depths.to(device)

        B, C, H, W = images.shape
        images_resized = F.interpolate(
            images, size=(model.img_size, model.img_size),
            mode="bilinear", align_corners=False,
        )

        with torch.amp.autocast('cuda'):
            canonical_inverse_depth, _ = model(images_resized)
            inverse_depth = canonical_inverse_depth * (W / NYU_FOCAL_LENGTH_PX)
            inverse_depth = F.interpolate(
                inverse_depth, size=(H, W), mode="bilinear", align_corners=False,
            )
            pred_depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)

            gt_depths_4d = gt_depths.unsqueeze(1)
            mask = (gt_depths_4d > 1e-3) & (gt_depths_4d < 10.0)

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

        with torch.amp.autocast('cuda'):
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

    return {"abs_rel": np.mean(all_abs_rel), "delta1": np.mean(all_delta1)}


def main():
    parser = argparse.ArgumentParser(description="Phase 2: LoRA fine-tuning of Depth Pro")
    parser.add_argument("--dataset-path", type=str, default="datasets/nyu_depth_v2_labeled.mat")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr-decoder", type=float, default=1e-4)
    parser.add_argument("--lr-lora", type=float, default=5e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--grad-loss-weight", type=float, default=0.5)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("=" * 60)
    print("Depth Pro - Phase 2: LoRA Encoder Fine-tuning")
    print("=" * 60)
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"LR: decoder={args.lr_decoder}, lora={args.lr_lora}")
    print(f"Epochs: {args.epochs}, warmup: {args.warmup_epochs}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model with pretrained weights
    print("\nLoading pretrained model...")
    model, transform = depth_pro.create_model_and_transforms(device=device)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA to encoder attention layers
    print("\nApplying LoRA...")
    lora_params = apply_lora_to_encoder(model, rank=args.lora_rank, alpha=args.lora_alpha)

    # Move LoRA params to device (they were created on CPU)
    for encoder_name in ["patch_encoder", "image_encoder"]:
        encoder = getattr(model.encoder, encoder_name)
        for block in encoder.blocks:
            if isinstance(block.attn.qkv, LoRALinear):
                block.attn.qkv.lora_A = nn.Parameter(block.attn.qkv.lora_A.to(device))
                block.attn.qkv.lora_B = nn.Parameter(block.attn.qkv.lora_B.to(device))
            if isinstance(block.attn.proj, LoRALinear):
                block.attn.proj.lora_A = nn.Parameter(block.attn.proj.lora_A.to(device))
                block.attn.proj.lora_B = nn.Parameter(block.attn.proj.lora_B.to(device))
    # Refresh lora_params list after moving to device
    lora_params = []
    for encoder_name in ["patch_encoder", "image_encoder"]:
        encoder = getattr(model.encoder, encoder_name)
        for block in encoder.blocks:
            if isinstance(block.attn.qkv, LoRALinear):
                lora_params.extend([block.attn.qkv.lora_A, block.attn.qkv.lora_B])
            if isinstance(block.attn.proj, LoRALinear):
                lora_params.extend([block.attn.proj.lora_A, block.attn.proj.lora_B])

    # Unfreeze decoder + head
    for param in model.decoder.parameters():
        param.requires_grad = True
    for param in model.head.parameters():
        param.requires_grad = True

    # Keep FOV frozen
    if hasattr(model, "fov"):
        for param in model.fov.parameters():
            param.requires_grad = False

    # Count parameters
    lora_trainable = sum(p.numel() for p in lora_params)
    decoder_trainable = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    head_trainable = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters:")
    print(f"  LoRA:    {lora_trainable/1e6:.2f}M")
    print(f"  Decoder: {decoder_trainable/1e6:.2f}M")
    print(f"  Head:    {head_trainable/1e6:.2f}M")
    print(f"  Total:   {total_trainable/1e6:.2f}M / {total_params/1e6:.1f}M ({100*total_trainable/total_params:.1f}%)")

    # Enable gradient checkpointing for memory savings
    if hasattr(model.encoder.patch_encoder, "set_grad_checkpointing"):
        model.encoder.patch_encoder.set_grad_checkpointing(True)
        print("Enabled gradient checkpointing for patch encoder")
    if hasattr(model.encoder.image_encoder, "set_grad_checkpointing"):
        model.encoder.image_encoder.set_grad_checkpointing(True)
        print("Enabled gradient checkpointing for image encoder")

    # Aggressive memory management
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Load dataset
    eigen_indices_path = Path(__file__).parent / "eigen_test_indices.json"
    if eigen_indices_path.exists():
        with open(eigen_indices_path) as f:
            test_indices = set(json.load(f))
        train_indices = [i for i in range(1449) if i not in test_indices]
    else:
        train_indices = list(range(795))

    np.random.seed(42)
    np.random.shuffle(train_indices)
    val_indices = train_indices[:50]
    train_indices_final = train_indices[50:]

    print(f"\nTrain: {len(train_indices_final)}, Val: {len(val_indices)}")

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

    # Discriminative learning rates: LoRA gets lower LR
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr_lora, "weight_decay": 1e-2},
        {"params": list(model.decoder.parameters()) + list(model.head.parameters()),
         "lr": args.lr_decoder, "weight_decay": 1e-4},
    ])

    # Cosine annealing (after warmup)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6,
    )

    scaler = torch.cuda.amp.GradScaler()

    best_abs_rel = float("inf")
    training_log = []

    print(f"\nStarting LoRA training ({args.epochs} epochs)...")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Linear warmup
        if epoch <= args.warmup_epochs:
            warmup_factor = epoch / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = pg["initial_lr"] * warmup_factor if "initial_lr" in pg else pg["lr"]
            # Set initial_lr for scheduler on first epoch
            if epoch == 1:
                for pg in optimizer.param_groups:
                    pg["initial_lr"] = pg["lr"]

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler,
            si_loss_fn, grad_loss_fn,
            grad_accum_steps=args.grad_accum,
            device=device, epoch=epoch,
            grad_loss_weight=args.grad_loss_weight,
        )

        if epoch > args.warmup_epochs:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - t_start
        eta = (total_elapsed / epoch) * (args.epochs - epoch)

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr_lora": optimizer.param_groups[0]["lr"],
            "lr_decoder": optimizer.param_groups[1]["lr"],
            "epoch_time": epoch_time,
        }

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = validate(model, val_loader, device)
            log_entry.update(val_metrics)

            print(f"Epoch {epoch}/{args.epochs} | loss: {train_loss:.4f} | "
                  f"val_absrel: {val_metrics['abs_rel']:.4f} | "
                  f"val_d1: {val_metrics['delta1']:.4f} | "
                  f"lr_lora: {optimizer.param_groups[0]['lr']:.6f} | "
                  f"time: {epoch_time:.1f}s | ETA: {eta/60:.1f}min")

            if val_metrics["abs_rel"] < best_abs_rel:
                best_abs_rel = val_metrics["abs_rel"]
                save_path = Path(args.save_dir) / "depth_pro_lora_best.pt"
                torch.save(model.state_dict(), save_path)
                print(f"  -> Saved best model (AbsRel: {best_abs_rel:.4f})")
        else:
            print(f"Epoch {epoch}/{args.epochs} | loss: {train_loss:.4f} | "
                  f"lr_lora: {optimizer.param_groups[0]['lr']:.6f} | "
                  f"time: {epoch_time:.1f}s | ETA: {eta/60:.1f}min")

        training_log.append(log_entry)
        torch.cuda.empty_cache()

    # Save final model and log
    final_path = Path(args.save_dir) / "depth_pro_lora_final.pt"
    torch.save(model.state_dict(), final_path)

    log_path = Path(args.save_dir) / "training_log_lora.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nTotal training time: {total_time/60:.1f} minutes")
    print(f"Best validation AbsRel: {best_abs_rel:.4f}")
    print(f"GPU peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
