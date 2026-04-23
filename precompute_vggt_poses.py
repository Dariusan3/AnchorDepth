#!/usr/bin/env python3
"""Precompute relative camera poses for all KITTI training/val triplets using VGGT.

VGGT (Visual Geometry Grounded Transformer, CVPR 2025 Best Paper) predicts
absolute camera poses from 2+ images. We pass each triplet (prev, target, next)
and extract the relative poses T_target→prev and T_target→next as 4×4 matrices.

These cached poses replace PoseNet during training — VGGT poses are far more
accurate than a randomly-initialized ResNet-18 PoseNet, especially early on.

Usage (run when GPU is free, i.e. after training finishes):
    python precompute_vggt_poses.py --data-path datasets/kitti_raw --stride 6
    python precompute_vggt_poses.py --data-path datasets/kitti_raw --stride 6 --split val

Output:
    checkpoints/vggt_poses_train_s6.pt  — {sample_idx: {"T_prev": [4,4], "T_next": [4,4]}}
    checkpoints/vggt_poses_val_s6.pt

Memory: VGGT needs ~6-8 GB VRAM. Run only when training is NOT active.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))
from depth_pro.selfsup.kitti_dataset import KITTIRawDataset


def extrinsics_to_4x4(ext_3x4: torch.Tensor) -> torch.Tensor:
    """Convert [3, 4] extrinsic to [4, 4] homogeneous matrix."""
    bottom = torch.tensor([[0., 0., 0., 1.]], device=ext_3x4.device, dtype=ext_3x4.dtype)
    return torch.cat([ext_3x4, bottom], dim=0)


def relative_pose(E_target: torch.Tensor, E_source: torch.Tensor) -> torch.Tensor:
    """Compute target-to-source transformation matrix.

    Args:
        E_target: [4, 4] world-to-target camera extrinsic
        E_source: [4, 4] world-to-source camera extrinsic

    Returns:
        [4, 4] T_target_to_source — transforms points in target frame to source frame.
        This matches the convention in warping.py: P_source = K @ T @ P_target
    """
    # T_target_to_source = E_source @ inv(E_target)
    # inv(E) for [R|t] = [R.T | -R.T @ t]
    R_t = E_target[:3, :3]
    t_t = E_target[:3, 3]
    E_target_inv = torch.zeros(4, 4, device=E_target.device, dtype=E_target.dtype)
    E_target_inv[:3, :3] = R_t.T
    E_target_inv[:3, 3] = -R_t.T @ t_t
    E_target_inv[3, 3] = 1.0
    return E_source @ E_target_inv


def precompute_split(data_path: str, split: str, stride: int, output_path: Path,
                     device: torch.device, model, pose_encoding_to_extri_intri,
                     load_and_preprocess_images):
    """Precompute VGGT poses for one split (train or val)."""
    split_file = f"splits/eigen_{split}_files.txt"
    if not Path(split_file).exists():
        print(f"Split file not found: {split_file}")
        return

    # Load dataset just to get file paths — no transforms needed
    dataset = KITTIRawDataset(
        data_path=data_path,
        split_file=split_file,
        stride=stride,
        is_train=False,
    )
    print(f"\n{split}: {len(dataset)} samples")

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    poses = {}
    errors = 0

    for idx in tqdm(range(len(dataset)), desc=f"VGGT {split}"):
        folder, frame_idx, side = dataset.filenames[idx]
        cam = "image_02" if side == "l" else "image_03"

        # Build paths for (prev, target, next)
        def img_path(fi):
            p = Path(data_path) / folder / cam / "data" / f"{fi:010d}.png"
            return p if p.exists() else Path(data_path) / folder / cam / "data" / f"{frame_idx:010d}.png"

        prev_path  = str(img_path(frame_idx - 1))
        tgt_path   = str(img_path(frame_idx))
        next_path  = str(img_path(frame_idx + 1))

        try:
            # VGGT takes a list of image paths: [prev, target, next]
            images = load_and_preprocess_images([prev_path, tgt_path, next_path]).to(device)
            # images: [S=3, 3, H, W] → need [1, S, 3, H, W]
            if images.dim() == 4:
                images = images.unsqueeze(0)

            h, w = images.shape[-2], images.shape[-1]

            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=dtype):
                    predictions = model(images)

            # Extract extrinsics: [B=1, S=3, 3, 4]
            pose_enc = predictions["pose_enc"]
            extrinsics, _ = pose_encoding_to_extri_intri(
                pose_enc, image_size_hw=(h, w), build_intrinsics=False
            )
            # extrinsics: [1, 3, 3, 4]
            E_prev   = extrinsics_to_4x4(extrinsics[0, 0].float())  # world→prev
            E_target = extrinsics_to_4x4(extrinsics[0, 1].float())  # world→target
            E_next   = extrinsics_to_4x4(extrinsics[0, 2].float())  # world→next

            # Relative poses (target-to-source convention for warping.py)
            T_prev = relative_pose(E_target, E_prev)   # target→prev
            T_next = relative_pose(E_target, E_next)   # target→next

            poses[idx] = {
                "T_prev": T_prev.cpu(),   # [4, 4]
                "T_next": T_next.cpu(),   # [4, 4]
            }

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\n  [WARN] idx={idx}: {e}")
            # Fall back to identity (will degrade gracefully)
            eye = torch.eye(4)
            poses[idx] = {"T_prev": eye, "T_next": eye}

        # Save checkpoint every 500 samples
        if (idx + 1) % 500 == 0:
            torch.save(poses, output_path)

    torch.save(poses, output_path)
    print(f"  Saved {len(poses)} poses → {output_path}  ({errors} errors)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="datasets/kitti_raw")
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Check VRAM (needs to be run with training NOT active)
    if device.type == "cuda":
        used_mb = torch.cuda.memory_allocated(device) / 1e6
        total_mb = torch.cuda.get_device_properties(device).total_memory / 1e6
        print(f"GPU: {used_mb:.0f} MB used / {total_mb:.0f} MB total")
        if used_mb > 2000:
            print("WARNING: >2GB VRAM already in use. Stop training first.")

    # Install and load VGGT
    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.image_utils import load_and_preprocess_images
    except ImportError:
        print("VGGT not installed. Run: pip install vggt")
        print("Or: pip install git+https://github.com/facebookresearch/vggt.git")
        return

    print("Loading VGGT-1B from HuggingFace (downloads ~2.4 GB on first run)...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    print("VGGT loaded.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    splits = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits:
        out = output_dir / f"vggt_poses_{split}_s{args.stride}.pt"
        if out.exists():
            print(f"[SKIP] {out} already exists. Delete to recompute.")
            continue
        precompute_split(
            args.data_path, split, args.stride, out,
            device, model, pose_encoding_to_extri_intri, load_and_preprocess_images,
        )

    print("\nDone. Use in training with:")
    print(f"  python train_kitti_selfsup_ms.py --vggt-poses checkpoints/vggt_poses_train_s{args.stride}.pt ...")


if __name__ == "__main__":
    main()
