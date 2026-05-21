#!/usr/bin/env python3
"""Phase 1 improved training: Better losses + augmentation + LR strategy.

Improvements over train_nyu.py:
  1. Multi-scale SSIM loss
  2. Affine-invariant loss
  3. Surface normal consistency loss
  4. Rich data augmentation (crop, color jitter, rotation, noise, erasing)
  5. LR warmup + cosine annealing with warm restarts
  6. Extended training (50 epochs)

Usage:
  python train_nyu_v3.py --epochs 50
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import depth_pro
from depth_pro.depth_pro import DepthProConfig, DEFAULT_MONODEPTH_CONFIG_DICT

NYU_FOCAL_LENGTH_PX = 518.8579


# ───────────────────────────────────────────────
# Dataset with rich augmentation
# ───────────────────────────────────────────────

class NYUDepthDatasetV3(Dataset):
    """NYU Depth V2 with rich augmentation for Phase 1."""

    def __init__(self, mat_path, indices, augment=True):
        self.mat_path = mat_path
        self.indices = indices
        self.augment = augment
        self.f = h5py.File(mat_path, "r")
        self.images = self.f["images"]
        self.depths = self.f["depths"]

    def __len__(self):
        return len(self.indices)

    def _augment(self, img, depth):
        """Apply rich augmentation. img: HWC uint8, depth: HW float."""
        h, w = depth.shape

        # 1. Random horizontal flip
        if np.random.random() > 0.5:
            img = np.ascontiguousarray(img[:, ::-1, :])
            depth = np.ascontiguousarray(depth[:, ::-1])

        # 2. Random crop + resize (80-100% area)
        if np.random.random() > 0.3:
            scale = np.random.uniform(0.8, 1.0)
            ch, cw = int(h * scale), int(w * scale)
            top = np.random.randint(0, h - ch + 1)
            left = np.random.randint(0, w - cw + 1)
            img = img[top:top+ch, left:left+cw]
            depth = depth[top:top+ch, left:left+cw]
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        # 3. Random rotation (±5°)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-5, 5)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
            depth = cv2.warpAffine(depth, M, (w, h), flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # 4. Color augmentation (RGB only)
        img = img.astype(np.float32)

        # Brightness
        if np.random.random() > 0.5:
            img *= np.random.uniform(0.8, 1.2)

        # Contrast
        if np.random.random() > 0.5:
            mean = img.mean()
            img = (img - mean) * np.random.uniform(0.8, 1.2) + mean

        # Saturation
        if np.random.random() > 0.5:
            gray = np.mean(img, axis=2, keepdims=True)
            sat = np.random.uniform(0.8, 1.2)
            img = gray + sat * (img - gray)

        # Hue shift (approximate via channel shuffle weight)
        if np.random.random() > 0.8:
            hue_shift = np.random.uniform(-0.05, 0.05)
            img[:, :, 0] *= (1 + hue_shift)
            img[:, :, 2] *= (1 - hue_shift)

        img = np.clip(img, 0, 255).astype(np.uint8)

        # 5. Gaussian noise
        if np.random.random() > 0.5:
            sigma = np.random.uniform(1, 8)
            noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 6. Random erasing (p=0.1)
        if np.random.random() > 0.9:
            eh = np.random.randint(int(h * 0.02), int(h * 0.1) + 1)
            ew = np.random.randint(int(w * 0.02), int(w * 0.1) + 1)
            et = np.random.randint(0, h - eh)
            el = np.random.randint(0, w - ew)
            img[et:et+eh, el:el+ew] = np.random.randint(0, 255, (eh, ew, 3), dtype=np.uint8)

        return img, depth

    def __getitem__(self, idx):
        mat_idx = self.indices[idx]

        img = self.images[mat_idx]
        img = np.transpose(img, (2, 1, 0))  # HWC
        depth = self.depths[mat_idx].T

        if self.augment:
            img, depth = self._augment(img, depth)

        img_tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor - 0.5) / 0.5

        depth_tensor = torch.from_numpy(depth.copy()).float()

        return img_tensor, depth_tensor

    def __del__(self):
        if hasattr(self, "f"):
            self.f.close()


# ───────────────────────────────────────────────
# Loss functions
# ───────────────────────────────────────────────

class ScaleInvariantLogLoss(nn.Module):
    """Scale-invariant log loss (Eigen et al.)."""

    def __init__(self, si_lambda=0.5):
        super().__init__()
        self.si_lambda = si_lambda

    def forward(self, pred, target, mask):
        log_diff = torch.log(pred[mask]) - torch.log(target[mask])
        return torch.mean(log_diff ** 2) - self.si_lambda * (torch.mean(log_diff) ** 2)


class GradientMatchingLoss(nn.Module):
    """Multi-scale gradient matching loss."""

    def forward(self, pred, target, mask):
        total_loss = 0.0
        for scale in [1, 2, 4]:
            if scale > 1:
                p = F.avg_pool2d(pred, scale)
                t = F.avg_pool2d(target, scale)
                m = F.max_pool2d(mask.float(), scale) > 0.5
            else:
                p, t, m = pred, target, mask

            dx_p = p[:, :, :, 1:] - p[:, :, :, :-1]
            dy_p = p[:, :, 1:, :] - p[:, :, :-1, :]
            dx_t = t[:, :, :, 1:] - t[:, :, :, :-1]
            dy_t = t[:, :, 1:, :] - t[:, :, :-1, :]

            mx = m[:, :, :, 1:] & m[:, :, :, :-1]
            my = m[:, :, 1:, :] & m[:, :, :-1, :]

            if mx.sum() > 0:
                total_loss += torch.mean(torch.abs(dx_p[mx] - dx_t[mx]))
            if my.sum() > 0:
                total_loss += torch.mean(torch.abs(dy_p[my] - dy_t[my]))

        return total_loss / 3.0


class MultiScaleSSIMLoss(nn.Module):
    """Multi-scale SSIM loss for structural similarity (simplified, robust)."""

    def __init__(self, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def _ssim_simple(self, pred, target):
        """Compute SSIM using simple averaging (no masking issues)."""
        mu_p = F.avg_pool2d(pred, 7, stride=1, padding=3)
        mu_t = F.avg_pool2d(target, 7, stride=1, padding=3)

        sigma_pp = F.avg_pool2d(pred * pred, 7, stride=1, padding=3) - mu_p ** 2
        sigma_tt = F.avg_pool2d(target * target, 7, stride=1, padding=3) - mu_t ** 2
        sigma_pt = F.avg_pool2d(pred * target, 7, stride=1, padding=3) - mu_p * mu_t

        ssim = ((2 * mu_p * mu_t + self.C1) * (2 * sigma_pt + self.C2)) / \
               ((mu_p ** 2 + mu_t ** 2 + self.C1) * (sigma_pp.clamp(0) + sigma_tt.clamp(0) + self.C2))

        return (1.0 - ssim).mean()

    def forward(self, pred, target, mask):
        # Apply mask by replacing invalid with mean
        p = pred.clone()
        t = target.clone()
        p[~mask] = p[mask].mean()
        t[~mask] = t[mask].mean()

        total = 0.0
        for scale in self.scales:
            if scale > 1:
                p_s = F.avg_pool2d(p, scale)
                t_s = F.avg_pool2d(t, scale)
            else:
                p_s, t_s = p, t
            total += self._ssim_simple(p_s, t_s)
        return total / len(self.scales)


class SurfaceNormalLoss(nn.Module):
    """Surface normal consistency loss from depth gradients."""

    def forward(self, pred, target, mask):
        # Compute depth gradients
        dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
        dy_t = target[:, :, 1:, :] - target[:, :, :-1, :]

        # Build normals: n = normalize([-dx, -dy, 1])
        # We need matching spatial dims, so crop to smallest
        min_h = min(dx_p.shape[2], dy_p.shape[2])
        min_w = min(dx_p.shape[3], dy_p.shape[3])

        dx_p = dx_p[:, :, :min_h, :min_w]
        dy_p = dy_p[:, :, :min_h, :min_w]
        dx_t = dx_t[:, :, :min_h, :min_w]
        dy_t = dy_t[:, :, :min_h, :min_w]

        ones = torch.ones_like(dx_p)

        n_pred = torch.cat([-dx_p, -dy_p, ones], dim=1)
        n_pred = F.normalize(n_pred, dim=1)

        n_target = torch.cat([-dx_t, -dy_t, ones], dim=1)
        n_target = F.normalize(n_target, dim=1)

        # Cosine similarity (dot product of normalized vectors)
        cos_sim = (n_pred * n_target).sum(dim=1, keepdim=True)

        # Mask for valid normals
        m = mask[:, :, :min_h, :min_w]
        mx = m[:, :, :, 1:] if m.shape[3] > min_w else m[:, :, :, :min_w]
        # Simplified: use center crop mask
        m_crop = mask[:, :, :min_h, :min_w]

        if m_crop.sum() > 0:
            return (1.0 - cos_sim[m_crop.expand_as(cos_sim)]).mean()
        return torch.tensor(0.0, device=pred.device)


class AffineInvariantLoss(nn.Module):
    """Affine-invariant loss: align pred to GT via optimal scale+shift, then measure residual."""

    def forward(self, pred, target, mask):
        p = pred[mask].float()
        t = target[mask].float()

        if len(p) < 10:
            return torch.tensor(0.0, device=pred.device)

        # Solve for optimal affine: t = alpha * p + beta
        # Using least squares: [p, 1] @ [alpha, beta]^T = t
        p_mean = p.mean()
        t_mean = t.mean()
        p_var = ((p - p_mean) ** 2).mean()

        if p_var < 1e-8:
            return torch.tensor(0.0, device=pred.device)

        alpha = ((p - p_mean) * (t - t_mean)).mean() / p_var
        beta = t_mean - alpha * p_mean

        aligned = alpha * p + beta
        residual = torch.abs(aligned - t) / (t + 1e-6)

        return residual.mean()


# ───────────────────────────────────────────────
# Training loop
# ───────────────────────────────────────────────

def freeze_encoder(model, unfreeze_last_n=0):
    """Freeze encoder, optionally unfreeze last N transformer blocks."""
    for param in model.encoder.parameters():
        param.requires_grad = False

    if unfreeze_last_n > 0:
        blocks = list(model.encoder.patch_encoder.blocks)
        for block in blocks[-unfreeze_last_n:]:
            for param in block.parameters():
                param.requires_grad = True
        print(f"Unfroze last {unfreeze_last_n} patch encoder blocks")

    if hasattr(model, "fov"):
        for param in model.fov.parameters():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.1f}%)")


def train_one_epoch(
    model, loader, optimizer, scaler, loss_fns, loss_weights,
    grad_accum_steps, device, epoch,
):
    model.train()
    model.encoder.eval()
    if hasattr(model, "fov"):
        model.fov.eval()

    total_loss = 0.0
    loss_components = {name: 0.0 for name in loss_fns}
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

        with torch.cuda.amp.autocast():
            canonical_inverse_depth, _ = model(images_resized)
            inverse_depth = canonical_inverse_depth * (W / NYU_FOCAL_LENGTH_PX)
            inverse_depth = F.interpolate(
                inverse_depth, size=(H, W), mode="bilinear", align_corners=False,
            )
            pred_depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)

            gt_depths_4d = gt_depths.unsqueeze(1)
            mask = (gt_depths_4d > 1e-3) & (gt_depths_4d < 10.0)

            log_pred = torch.log(torch.clamp(pred_depth, min=1e-4))
            log_gt = torch.log(torch.clamp(gt_depths_4d, min=1e-4))

            loss = torch.tensor(0.0, device=device)
            for name, loss_fn in loss_fns.items():
                if name in ("si_log", "affine"):
                    l = loss_fn(pred_depth, gt_depths_4d, mask)
                elif name in ("gradient", "normal"):
                    l = loss_fn(log_pred, log_gt, mask)
                elif name == "ssim":
                    # Use log-depth normalized to [0,1] for SSIM
                    lp = log_pred - log_pred[mask].min()
                    lt = log_gt - log_gt[mask].min()
                    range_max = max(lp[mask].max().item(), lt[mask].max().item(), 1e-6)
                    l = loss_fn(lp / range_max, lt / range_max, mask)
                else:
                    l = loss_fn(pred_depth, gt_depths_4d, mask)

                # Clamp individual losses to prevent blowup
                l = torch.clamp(l, max=10.0)
                loss = loss + loss_weights[name] * l
                loss_components[name] += l.item()

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

    avg_components = {k: v / max(num_batches, 1) for k, v in loss_components.items()}
    return total_loss / max(num_batches, 1), avg_components


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


class WarmupCosineScheduler:
    """Linear warmup + cosine annealing with warm restarts."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, restart_period=15, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.restart_period = restart_period
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch <= self.warmup_epochs:
            # Linear warmup
            factor = epoch / max(self.warmup_epochs, 1)
        else:
            # Cosine with warm restarts
            t = (epoch - self.warmup_epochs) % self.restart_period
            T = self.restart_period
            factor = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * t / T))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * factor

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Improved Depth Pro training")
    parser.add_argument("--dataset-path", type=str, default="datasets/nyu_depth_v2_labeled.mat")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--unfreeze-encoder-layers", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    # Loss weights
    parser.add_argument("--w-si", type=float, default=1.0, help="SI-log loss weight")
    parser.add_argument("--w-grad", type=float, default=0.5, help="Gradient loss weight")
    parser.add_argument("--w-ssim", type=float, default=0.3, help="SSIM loss weight")
    parser.add_argument("--w-normal", type=float, default=0.2, help="Normal loss weight")
    parser.add_argument("--w-affine", type=float, default=0.1, help="Affine-invariant loss weight")
    args = parser.parse_args()

    print("=" * 60)
    print("Depth Pro - Phase 1 Training (v3)")
    print("Improvements: Loss + Augmentation + LR Strategy")
    print("=" * 60)

    config_info = {
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "warmup_epochs": args.warmup_epochs,
        "loss_weights": {
            "si_log": args.w_si, "gradient": args.w_grad,
            "ssim": args.w_ssim, "normal": args.w_normal, "affine": args.w_affine,
        },
        "augmentation": "rich (crop, flip, rotate, color_jitter, noise, erasing)",
        "scheduler": f"warmup({args.warmup_epochs}ep) + cosine_restarts(T=15)",
    }
    print(json.dumps(config_info, indent=2))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    print("\nLoading pretrained model...")
    model, transform = depth_pro.create_model_and_transforms(device=device)
    freeze_encoder(model, unfreeze_last_n=args.unfreeze_encoder_layers)

    if hasattr(model.encoder.patch_encoder, "set_grad_checkpointing"):
        model.encoder.patch_encoder.set_grad_checkpointing(True)

    # Dataset
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

    print(f"Train: {len(train_indices_final)}, Val: {len(val_indices)}")

    train_dataset = NYUDepthDatasetV3(args.dataset_path, train_indices_final, augment=True)
    val_dataset = NYUDepthDatasetV3(args.dataset_path, val_indices, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True,
    )

    # Loss functions
    loss_fns = {
        "si_log": ScaleInvariantLogLoss(si_lambda=0.5),
        "gradient": GradientMatchingLoss(),
        "ssim": MultiScaleSSIMLoss(),
        "normal": SurfaceNormalLoss(),
        "affine": AffineInvariantLoss(),
    }
    loss_weights = {
        "si_log": args.w_si,
        "gradient": args.w_grad,
        "ssim": args.w_ssim,
        "normal": args.w_normal,
        "affine": args.w_affine,
    }

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    # LR scheduler with warmup
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs, restart_period=15, min_lr_ratio=0.01,
    )

    scaler = torch.cuda.amp.GradScaler()

    # Training
    best_abs_rel = float("inf")
    training_log = []
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\nStarting Phase 1 training ({args.epochs} epochs)...")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        scheduler.step(epoch)

        train_loss, loss_components = train_one_epoch(
            model, train_loader, optimizer, scaler,
            loss_fns, loss_weights,
            grad_accum_steps=args.grad_accum,
            device=device, epoch=epoch,
        )

        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - t_start
        eta = (total_elapsed / epoch) * (args.epochs - epoch)

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "loss_components": loss_components,
            "lr": scheduler.get_last_lr()[0],
            "epoch_time": epoch_time,
        }

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = validate(model, val_loader, device)
            log_entry.update(val_metrics)

            print(f"Epoch {epoch}/{args.epochs} | loss: {train_loss:.4f} | "
                  f"val_absrel: {val_metrics['abs_rel']:.4f} | "
                  f"val_d1: {val_metrics['delta1']:.4f} | "
                  f"lr: {scheduler.get_last_lr()[0]:.6f} | "
                  f"time: {epoch_time:.1f}s | ETA: {eta/60:.1f}min")

            if val_metrics["abs_rel"] < best_abs_rel:
                best_abs_rel = val_metrics["abs_rel"]
                save_path = Path(args.save_dir) / "depth_pro_v3_best.pt"
                torch.save(model.state_dict(), save_path)
                print(f"  -> Saved best model (AbsRel: {best_abs_rel:.4f})")
        else:
            print(f"Epoch {epoch}/{args.epochs} | loss: {train_loss:.4f} | "
                  f"lr: {scheduler.get_last_lr()[0]:.6f} | "
                  f"time: {epoch_time:.1f}s | ETA: {eta/60:.1f}min")

        training_log.append(log_entry)
        torch.cuda.empty_cache()

    # Save final
    final_path = Path(args.save_dir) / "depth_pro_v3_final.pt"
    torch.save(model.state_dict(), final_path)

    log_path = Path(args.save_dir) / "training_log_v3.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nTotal training time: {total_time/60:.1f} minutes")
    print(f"Best validation AbsRel: {best_abs_rel:.4f}")
    print(f"GPU peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
