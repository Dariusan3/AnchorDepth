"""KITTI raw dataset loader for self-supervised monocular depth estimation.

Loads sequential frame triplets (t-1, t, t+1) with camera intrinsics
from KITTI raw data. Follows the Eigen/Zhou split convention.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class KITTIRawDataset(Dataset):
    """KITTI raw dataset for self-supervised training.

    Loads frame triplets (t-1, t, t+1) and camera intrinsics.
    Returns images at two resolutions:
    - depth_size: for Depth Pro input (1536x1536 required by encoder pyramid)
    - pose_size: for PoseNet and loss computation (e.g., 640x192)
    """

    def __init__(
        self,
        data_path: str,
        split_file: str,
        depth_size: tuple[int, int] = (1536, 1536),
        pose_size: tuple[int, int] = (640, 192),
        is_train: bool = True,
        frame_ids: tuple[int, ...] = (-1, 0, 1),
        stride: int = 1,
        vggt_poses: Optional[dict] = None,
    ):
        """Initialize KITTI dataset.

        Args:
            data_path: Path to KITTI raw data root.
            split_file: Path to split file (e.g., eigen_train_files.txt).
            depth_size: (W, H) for Depth Pro input.
            pose_size: (W, H) for PoseNet / loss computation.
            is_train: Whether to apply color augmentation.
            frame_ids: Frame offsets to load (default: previous, current, next).
            stride: Use every Nth sample (reduces redundancy for 10Hz video).
            vggt_poses: Optional dict {sample_idx: {"T_prev": [4,4], "T_next": [4,4]}}
                        precomputed by precompute_vggt_poses.py. When provided,
                        horizontal flip is disabled to keep poses consistent.
        """
        self.data_path = Path(data_path)
        self.depth_size = depth_size  # (W, H)
        self.pose_size = pose_size  # (W, H)
        self.is_train = is_train
        self.frame_ids = frame_ids

        # Parse split file
        self.filenames = []
        with open(split_file) as f:
            for line in f:
                parts = line.strip().split()
                folder = parts[0]  # e.g., 2011_09_26/2011_09_26_drive_0001_sync
                frame_idx = int(parts[1])
                side = parts[2]  # 'l' or 'r'
                self.filenames.append((folder, frame_idx, side))

        # Subsample to reduce redundancy (KITTI is 10Hz video)
        if stride > 1:
            self.filenames = self.filenames[::stride]

        # Optional precomputed VGGT poses (replaces PoseNet)
        self.vggt_poses = vggt_poses  # {idx: {"T_prev": [4,4], "T_next": [4,4]}}

        # Depth Pro normalization
        self.depth_pro_normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )

        # Color augmentation for training
        self.color_aug = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ) if is_train else None

        # Cache calibration matrices per date
        self._calib_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.filenames)

    def _get_image_path(self, folder: str, frame_idx: int, side: str) -> Path:
        """Get path to a KITTI image."""
        cam = "image_02" if side == "l" else "image_03"
        return self.data_path / folder / cam / "data" / f"{frame_idx:010d}.png"

    def _load_image(self, folder: str, frame_idx: int, side: str) -> Image.Image:
        """Load a single KITTI image."""
        path = self._get_image_path(folder, frame_idx, side)
        return Image.open(path).convert("RGB")

    def _get_intrinsics(self, folder: str, side: str) -> np.ndarray:
        """Load camera intrinsics for a given date/side.

        Returns:
            (4, 4) intrinsic matrix K.
        """
        date = folder.split("/")[0]
        cache_key = f"{date}_{side}"

        if cache_key in self._calib_cache:
            return self._calib_cache[cache_key].copy()

        calib_path = self.data_path / date / "calib_cam_to_cam.txt"
        calib_data = {}
        with open(calib_path) as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    calib_data[key.strip()] = value.strip()

        # P_rect for the appropriate camera
        cam_idx = "02" if side == "l" else "03"
        P = np.array(calib_data[f"P_rect_{cam_idx}"].split(), dtype=np.float32).reshape(3, 4)

        # Extract 3x3 intrinsic matrix from projection matrix
        K = np.eye(4, dtype=np.float32)
        K[:3, :3] = P[:3, :3]

        self._calib_cache[cache_key] = K
        return K.copy()

    def _check_frame_exists(self, folder: str, frame_idx: int, side: str) -> bool:
        """Check if a frame exists on disk."""
        return self._get_image_path(folder, frame_idx, side).exists()

    def __getitem__(self, index: int) -> dict:
        """Load a training sample.

        Returns:
            Dict with keys:
                'target_depth': (3, depth_H, depth_W) normalized for Depth Pro
                'target': (3, pose_H, pose_W) target image for loss computation
                'source_-1': (3, pose_H, pose_W) previous frame
                'source_1': (3, pose_H, pose_W) next frame
                'K': (4, 4) scaled intrinsic matrix
                'inv_K': (4, 4) inverse intrinsic matrix
        """
        folder, frame_idx, side = self.filenames[index]

        # Load intrinsics
        K = self._get_intrinsics(folder, side)

        # Load images (target + source frames)
        images = {}
        for fid in self.frame_ids:
            idx = frame_idx + fid
            # Fall back to target frame if source doesn't exist
            if not self._check_frame_exists(folder, idx, side):
                idx = frame_idx
            images[fid] = self._load_image(folder, idx, side)

        orig_w, orig_h = images[0].size  # PIL gives (W, H)

        # Apply consistent color augmentation to all frames
        if self.is_train and self.color_aug is not None and random.random() > 0.5:
            # get_params returns (fn_order, brightness, contrast, saturation, hue)
            fn_idx, b_factor, c_factor, s_factor, h_factor = self.color_aug.get_params(
                self.color_aug.brightness,
                self.color_aug.contrast,
                self.color_aug.saturation,
                self.color_aug.hue,
            )
            # Apply in the randomized order, identically to all frames
            for fid in self.frame_ids:
                for fn_id in fn_idx:
                    if fn_id == 0:
                        images[fid] = transforms.functional.adjust_brightness(images[fid], b_factor)
                    elif fn_id == 1:
                        images[fid] = transforms.functional.adjust_contrast(images[fid], c_factor)
                    elif fn_id == 2:
                        images[fid] = transforms.functional.adjust_saturation(images[fid], s_factor)
                    elif fn_id == 3:
                        images[fid] = transforms.functional.adjust_hue(images[fid], h_factor)

        # Horizontal flip augmentation — disabled when using VGGT poses
        # (precomputed poses correspond to the un-flipped image geometry)
        do_flip = self.is_train and self.vggt_poses is None and random.random() > 0.5

        result = {}

        # Resize for PoseNet / loss computation
        pose_w, pose_h = self.pose_size
        for fid in self.frame_ids:
            img = images[fid].resize((pose_w, pose_h), Image.LANCZOS)
            if do_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            tensor = transforms.functional.to_tensor(img)  # [0, 1]
            if fid == 0:
                result["target"] = tensor
            else:
                result[f"source_{fid}"] = tensor

        # Resize target for Depth Pro input (square)
        depth_w, depth_h = self.depth_size
        target_depth = images[0].resize((depth_w, depth_h), Image.LANCZOS)
        if do_flip:
            target_depth = target_depth.transpose(Image.FLIP_LEFT_RIGHT)
        target_depth = transforms.functional.to_tensor(target_depth)
        target_depth = self.depth_pro_normalize(target_depth)
        result["target_depth"] = target_depth

        # Scale intrinsics for pose_size resolution
        K_scaled = K.copy()
        K_scaled[0, :] *= pose_w / orig_w
        K_scaled[1, :] *= pose_h / orig_h

        if do_flip:
            K_scaled[0, 2] = pose_w - K_scaled[0, 2]

        result["K"] = torch.from_numpy(K_scaled)
        result["inv_K"] = torch.from_numpy(np.linalg.inv(K_scaled))

        # Add precomputed VGGT poses if available
        if self.vggt_poses is not None and index in self.vggt_poses:
            entry = self.vggt_poses[index]
            result["T_prev"] = entry["T_prev"].float()  # [4, 4] target→prev
            result["T_next"] = entry["T_next"].float()  # [4, 4] target→next

        return result


class KITTIEigenTestDataset(Dataset):
    """KITTI Eigen test set with LiDAR ground truth for evaluation.

    Loads single images + projected LiDAR depth maps for metric evaluation.
    """

    def __init__(
        self,
        data_path: str,
        split_file: str,
        gt_path: Optional[str] = None,
        depth_size: tuple[int, int] = (1536, 1536),
    ):
        self.data_path = Path(data_path)
        self.depth_size = depth_size
        self.gt_path = Path(gt_path) if gt_path else None

        self.filenames = []
        with open(split_file) as f:
            for line in f:
                parts = line.strip().split()
                self.filenames.append((parts[0], int(parts[1]), parts[2]))

        self.depth_pro_normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )

        self._calib_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.filenames)

    def _get_intrinsics(self, folder: str, side: str) -> np.ndarray:
        date = folder.split("/")[0]
        cache_key = f"{date}_{side}"
        if cache_key in self._calib_cache:
            return self._calib_cache[cache_key].copy()

        calib_path = self.data_path / date / "calib_cam_to_cam.txt"
        calib_data = {}
        with open(calib_path) as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    calib_data[key.strip()] = value.strip()

        cam_idx = "02" if side == "l" else "03"
        P = np.array(calib_data[f"P_rect_{cam_idx}"].split(), dtype=np.float32).reshape(3, 4)
        K = np.eye(4, dtype=np.float32)
        K[:3, :3] = P[:3, :3]
        self._calib_cache[cache_key] = K
        return K.copy()

    def __getitem__(self, index: int) -> dict:
        folder, frame_idx, side = self.filenames[index]

        cam = "image_02" if side == "l" else "image_03"
        img_path = self.data_path / folder / cam / "data" / f"{frame_idx:010d}.png"
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # For Depth Pro
        depth_w, depth_h = self.depth_size
        img_depth = img.resize((depth_w, depth_h), Image.LANCZOS)
        img_tensor = transforms.functional.to_tensor(img_depth)
        img_tensor = self.depth_pro_normalize(img_tensor)

        result = {
            "image": img_tensor,
            "orig_size": torch.tensor([orig_h, orig_w]),
            "index": index,
        }

        # Load ground truth if available
        if self.gt_path is not None:
            gt_file = self.gt_path / f"{index:010d}.png"
            if gt_file.exists():
                gt_depth = np.array(Image.open(gt_file)).astype(np.float32) / 256.0
                result["gt_depth"] = torch.from_numpy(gt_depth)

        return result
