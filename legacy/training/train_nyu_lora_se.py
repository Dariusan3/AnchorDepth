#!/usr/bin/env python3
"""v6: Fine-tune Depth Pro with LoRA + SE channel attention.

Builds on v4 (LoRA encoder adaptation) by adding a Squeeze-and-Excitation
channel attention block between the decoder output and depth head.
Starts from the best v4 LoRA checkpoint for faster convergence.

Key changes from v4:
  - SE block applied after decoder output (~16K new params)
  - Decoder frozen (already fine-tuned in v4), only LoRA + SE + head trained
  - Lower initial LR since starting from a good checkpoint
  - 20 epochs (less needed with warm start)

Usage:
  python train_nyu_lora_se.py --epochs 20
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
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))
import depth_pro
from depth_pro.improvements.enhanced_decoder import add_se_to_model

# Import LoRA from existing script
from train_nyu_lora import (
    LoRALinear, apply_lora_to_encoder,
    NYUDepthDataset, ScaleInvariantLogLoss, GradientMatchingLoss,
    train_one_epoch, validate,
)


def main():
    parser = argparse.ArgumentParser(description="v6: LoRA + SE channel attention")
    parser.add_argument("--dataset-path", type=str, default="datasets/nyu_depth_v2_labeled.mat")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/depth_pro_lora_best.pt",
                        help="v4 LoRA checkpoint to start from")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr-head", type=float, default=5e-5,
                        help="LR for depth head")
    parser.add_argument("--lr-lora", type=float, default=2e-5,
                        help="Lower LR for LoRA params (already adapted)")
    parser.add_argument("--lr-se", type=float, default=1e-4,
                        help="Higher LR for new SE block (randomly initialized)")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--se-reduction", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--grad-loss-weight", type=float, default=0.5)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("=" * 60)
    print("Depth Pro - v6: LoRA + SE Channel Attention")
    print("=" * 60)
    print(f"Starting from checkpoint: {args.checkpoint}")
    print(f"SE reduction: {args.se_reduction}")
    print(f"LR: head={args.lr_head}, lora={args.lr_lora}, se={args.lr_se}")
    print(f"Epochs: {args.epochs}, warmup: {args.warmup_epochs}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model with pretrained weights
    print("\nLoading pretrained model...")
    model, transform = depth_pro.create_model_and_transforms(device=device)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA to encoder (same structure as v4)
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

    # Load v4 checkpoint (LoRA + decoder + head weights)
    print(f"\nLoading v4 checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    del ckpt
    torch.cuda.empty_cache()
    print("Loaded v4 checkpoint successfully")

    # Add SE attention block after decoder (new randomly-initialized params)
    print("\nAdding SE channel attention...")
    add_se_to_model(model, reduction=args.se_reduction)
    model.se_attention = model.se_attention.to(device)

    # Unfreeze: LoRA + SE + head (decoder stays frozen from v4)
    for param in model.head.parameters():
        param.requires_grad = True

    # Refresh lora_params list after checkpoint load
    lora_params = []
    for encoder_name in ["patch_encoder", "image_encoder"]:
        encoder = getattr(model.encoder, encoder_name)
        for block in encoder.blocks:
            if isinstance(block.attn.qkv, LoRALinear):
                lora_params.extend([block.attn.qkv.lora_A, block.attn.qkv.lora_B])
            if isinstance(block.attn.proj, LoRALinear):
                lora_params.extend([block.attn.proj.lora_A, block.attn.proj.lora_B])

    # Keep FOV frozen
    if hasattr(model, "fov"):
        for param in model.fov.parameters():
            param.requires_grad = False

    # Count parameters
    se_params = list(model.se_attention.parameters())
    se_trainable = sum(p.numel() for p in se_params)
    lora_trainable = sum(p.numel() for p in lora_params)
    head_trainable = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters:")
    print(f"  LoRA:     {lora_trainable/1e6:.2f}M")
    print(f"  SE block: {se_trainable/1e3:.1f}K")
    print(f"  Head:     {head_trainable/1e6:.2f}M")
    print(f"  Total:    {total_trainable/1e6:.2f}M / {total_params/1e6:.1f}M ({100*total_trainable/total_params:.1f}%)")

    # Enable gradient checkpointing
    if hasattr(model.encoder.patch_encoder, "set_grad_checkpointing"):
        model.encoder.patch_encoder.set_grad_checkpointing(True)
        print("Enabled gradient checkpointing for patch encoder")
    if hasattr(model.encoder.image_encoder, "set_grad_checkpointing"):
        model.encoder.image_encoder.set_grad_checkpointing(True)
        print("Enabled gradient checkpointing for image encoder")

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

    # Discriminative learning rates
    param_groups = [
        {"params": lora_params, "lr": args.lr_lora, "weight_decay": 1e-2},
        {"params": list(model.head.parameters()),
         "lr": args.lr_head, "weight_decay": 1e-4},
        {"params": se_params, "lr": args.lr_se, "weight_decay": 1e-3},
    ]
    optimizer = torch.optim.AdamW(param_groups)

    # Cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6,
    )

    scaler = torch.amp.GradScaler('cuda')

    best_abs_rel = float("inf")
    training_log = []

    print(f"\nStarting v6 training ({args.epochs} epochs)...")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Linear warmup
        if epoch <= args.warmup_epochs:
            warmup_factor = epoch / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = pg.get("initial_lr", pg["lr"]) * warmup_factor
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
            "lr_head": optimizer.param_groups[1]["lr"],
            "lr_se": optimizer.param_groups[2]["lr"],
            "epoch_time": epoch_time,
        }

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = validate(model, val_loader, device)
            log_entry.update(val_metrics)

            print(f"Epoch {epoch}/{args.epochs} | loss: {train_loss:.4f} | "
                  f"val_absrel: {val_metrics['abs_rel']:.4f} | "
                  f"val_d1: {val_metrics['delta1']:.4f} | "
                  f"lr_se: {optimizer.param_groups[2]['lr']:.6f} | "
                  f"time: {epoch_time:.1f}s | ETA: {eta/60:.1f}min")

            if val_metrics["abs_rel"] < best_abs_rel:
                best_abs_rel = val_metrics["abs_rel"]
                save_path = Path(args.save_dir) / "depth_pro_lora_se_best.pt"
                torch.save(model.state_dict(), save_path)
                print(f"  -> Saved best model (AbsRel: {best_abs_rel:.4f})")
        else:
            print(f"Epoch {epoch}/{args.epochs} | loss: {train_loss:.4f} | "
                  f"lr_se: {optimizer.param_groups[2]['lr']:.6f} | "
                  f"time: {epoch_time:.1f}s | ETA: {eta/60:.1f}min")

        training_log.append(log_entry)
        torch.cuda.empty_cache()

    # Save final model and log
    final_path = Path(args.save_dir) / "depth_pro_lora_se_final.pt"
    torch.save(model.state_dict(), final_path)

    log_path = Path(args.save_dir) / "training_log_lora_se.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nTotal training time: {total_time/60:.1f} minutes")
    print(f"Best validation AbsRel: {best_abs_rel:.4f}")
    print(f"GPU peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
