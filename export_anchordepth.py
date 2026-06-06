#!/usr/bin/env python3
"""Export an AnchorDepth checkpoint as a standalone Depth Pro state_dict.

Merges LoRA adapters into the base linear weights so the output file can be
loaded by any Depth Pro instance without requiring LoRA code at inference time.

After running this script, the exported .pt file can be loaded directly:

    import depth_pro
    model, transform = depth_pro.create_model_and_transforms(device="cuda")
    sd = torch.load("anchordepth_v15.pt", map_location="cuda")
    model.load_state_dict(sd, strict=True)
    model.eval()
    # ready for inference — no LoRA imports needed

Usage:
    python export_anchordepth.py --variant v15            # default: v15 → KITTI
    python export_anchordepth.py --variant v18            # v18 → Make3D winner
    python export_anchordepth.py --variant v20            # v20 → Cityscapes winner
    python export_anchordepth.py --variant v15 --output anchordepth_kitti.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent / "src"))
import depth_pro
from train_nyu_lora import LoRALinear, apply_lora_to_encoder


VARIANT_CHECKPOINTS = {
    "v15": ("checkpoints/selfsup_v15/selfsup_best.pt", 8, 8.0),
    "v18": ("checkpoints/selfsup_v18/selfsup_best.pt", 8, 8.0),
    "v20": ("checkpoints/selfsup_v20/selfsup_best.pt", 8, 8.0),
    "v16": ("checkpoints/selfsup_v16/selfsup_best.pt", 8, 8.0),
}


def merge_lora_into_base(model: nn.Module):
    """Walk every LoRALinear in the model, fold (alpha/r)·B·A into original.weight, replace it with a plain nn.Linear."""
    merged_count = 0
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                W = child.original.weight.data
                A = child.lora_A.data
                B = child.lora_B.data
                scaling = child.alpha / child.rank
                delta = scaling * (B @ A)
                W_merged = (W + delta).to(W.dtype)

                in_features = child.original.in_features
                out_features = child.original.out_features
                new_linear = nn.Linear(in_features, out_features,
                                       bias=child.original.bias is not None)
                new_linear.weight = nn.Parameter(W_merged)
                if child.original.bias is not None:
                    new_linear.bias = nn.Parameter(child.original.bias.data.clone())
                new_linear = new_linear.to(W.device).to(W.dtype)
                setattr(parent, name, new_linear)
                merged_count += 1
    return merged_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANT_CHECKPOINTS.keys()), default="v15")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=float, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cpu", help="cpu is enough for export")
    args = parser.parse_args()

    default_ckpt, default_rank, default_alpha = VARIANT_CHECKPOINTS[args.variant]
    ckpt_path = args.checkpoint or default_ckpt
    rank = args.lora_rank or default_rank
    alpha = args.lora_alpha if args.lora_alpha is not None else default_alpha
    out_path = args.output or f"anchordepth_{args.variant}.pt"

    if not Path(ckpt_path).exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    device = torch.device(args.device)
    print(f"Loading Depth Pro skeleton on {device}...")
    model, _ = depth_pro.create_model_and_transforms(device=device)

    print(f"Applying LoRA (rank={rank}, alpha={alpha})...")
    apply_lora_to_encoder(model, rank=rank, alpha=alpha)
    for enc_name in ("patch_encoder", "image_encoder"):
        enc = getattr(model.encoder, enc_name)
        for block in enc.blocks:
            if isinstance(block.attn.qkv, LoRALinear):
                block.attn.qkv.lora_A = nn.Parameter(block.attn.qkv.lora_A.to(device))
                block.attn.qkv.lora_B = nn.Parameter(block.attn.qkv.lora_B.to(device))
            if isinstance(block.attn.proj, LoRALinear):
                block.attn.proj.lora_A = nn.Parameter(block.attn.proj.lora_A.to(device))
                block.attn.proj.lora_B = nn.Parameter(block.attn.proj.lora_B.to(device))

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("depth_model", ckpt)
    model_dict = model.state_dict()
    filtered = {k: v for k, v in sd.items()
                if k in model_dict and model_dict[k].shape == v.shape}
    skipped = set(sd.keys()) - set(filtered.keys())
    if skipped:
        print(f"  Skipped {len(skipped)} mismatched keys")
    model.load_state_dict(filtered, strict=False)

    print("Merging LoRA into base weights...")
    merged_count = merge_lora_into_base(model)
    print(f"  Merged {merged_count} LoRA-augmented linear layers")

    print("Sanity check: confirm no LoRALinear remains...")
    remaining = sum(1 for m in model.modules() if isinstance(m, LoRALinear))
    if remaining:
        print(f"  WARNING: {remaining} LoRALinear modules still present")
    else:
        print("  OK — clean Depth Pro architecture, no LoRA dependency.")

    sd_out = model.state_dict()
    nan_keys = [k for k, v in sd_out.items()
                if v.is_floating_point() and not torch.isfinite(v).all()]
    if nan_keys:
        print(f"ERROR: {len(nan_keys)} non-finite params, refusing to save")
        sys.exit(1)

    torch.save(sd_out, out_path)
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"\n✓ Exported AnchorDepth-{args.variant} → {out_path}  ({size_mb:.0f} MB)")
    print()
    print("Loading example for your interface:")
    print(f"    import torch, depth_pro")
    print(f"    model, transform = depth_pro.create_model_and_transforms(device='cuda')")
    print(f"    model.load_state_dict(torch.load('{out_path}', map_location='cuda'), strict=True)")
    print(f"    model.eval()")
    print(f"    # then use as a normal Depth Pro model — no LoRA imports needed")


if __name__ == "__main__":
    main()
