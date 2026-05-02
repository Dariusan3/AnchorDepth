#!/usr/bin/env python3
"""Precompute zero-shot Depth Pro depth predictions for all KITTI training/val triplets.

These cached depths serve as anchors for the consistency loss in v15:
    L = L_photometric + lambda * |depth - depth_zero_shot|_1

Usage (run when GPU is free):
    python precompute_zeroshot_depths.py --stride 6

Output: checkpoints/zeroshot_depths_train_s6.pt and val.
        Format: dict[sample_idx] -> tensor[H_pose, W_pose] (float16 to save disk).
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Normalize, ToTensor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))
import depth_pro
from depth_pro.selfsup.kitti_dataset import KITTIRawDataset


def precompute_split(model, data_path: str, split_file: str, stride: int,
                     pose_size: tuple[int, int], output_path: Path, device):
    """Run zero-shot Depth Pro on all triplets in a split, save depth at pose_size."""
    if not Path(split_file).exists():
        print(f"Split file missing: {split_file}")
        return

    dataset = KITTIRawDataset(
        data_path=data_path, split_file=split_file,
        stride=stride, is_train=False,
    )
    print(f"\n{Path(split_file).stem}: {len(dataset)} samples")

    pose_w, pose_h = pose_size
    normalize = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    to_tensor = ToTensor()
    depths = {}

    for idx in tqdm(range(len(dataset)), desc="zero-shot"):
        folder, frame_idx, side = dataset.filenames[idx]
        cam = "image_02" if side == "l" else "image_03"
        img_path = Path(data_path) / folder / cam / "data" / f"{frame_idx:010d}.png"
        if not img_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        # Same preprocessing as training (target_depth)
        img_1536 = img.resize((1536, 1536), Image.LANCZOS)
        inp = normalize(to_tensor(img_1536)).unsqueeze(0).to(device)

        # KITTI focal length at original resolution
        K = dataset._get_intrinsics(folder, side)  # at orig resolution
        f_px = float(K[0, 0])

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            canonical_inv_depth, _ = model(inp)
            # Convert to metric inv depth using KITTI GT focal (matches training)
            scale = pose_w / (f_px * pose_w / orig_w)  # = orig_w / f_px
            inv_depth = canonical_inv_depth * scale
            # Resize to pose resolution
            inv_depth = F.interpolate(inv_depth, size=(pose_h, pose_w),
                                      mode="bilinear", align_corners=False)
            inv_depth = torch.nan_to_num(inv_depth, nan=1.0, posinf=10.0, neginf=1e-6)
            inv_depth = F.relu(inv_depth) + 1e-6
            depth = 1.0 / torch.clamp(inv_depth, min=1e-4, max=1e4)

        # Save as fp16 to halve disk usage
        depths[idx] = depth.squeeze().cpu().half()  # [H_pose, W_pose]

        if (idx + 1) % 1000 == 0:
            torch.save(depths, output_path)

    torch.save(depths, output_path)
    print(f"  Saved {len(depths)} depths -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="datasets/kitti_raw")
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--pose-size", default="416x128")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    pose_w, pose_h = [int(x) for x in args.pose_size.split("x")]
    device = torch.device(args.device)

    print("Loading zero-shot Depth Pro...")
    model, _ = depth_pro.create_model_and_transforms(device=device)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    for split in ["train", "val"]:
        split_file = f"splits/eigen_{split}_files.txt"
        out = output_dir / f"zeroshot_depths_{split}_s{args.stride}_{pose_w}x{pose_h}.pt"
        if out.exists():
            print(f"[SKIP] {out} exists. Delete to recompute.")
            continue
        precompute_split(model, args.data_path, split_file,
                         args.stride, (pose_w, pose_h), out, device)

    print("\nDone. Use in v15 with --zeroshot-depths checkpoints/zeroshot_depths_train_s6_416x128.pt")


if __name__ == "__main__":
    main()
