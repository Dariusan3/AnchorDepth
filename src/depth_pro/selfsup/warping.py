"""Differentiable image warping for self-supervised depth estimation.

Given a target depth map, camera intrinsics, and relative pose between
target and source frames, warps the source image to reconstruct the
target view. Uses differentiable bilinear sampling (grid_sample).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class BackprojectDepth(nn.Module):
    """Backproject pixel coordinates to 3D using depth and camera intrinsics."""

    def __init__(self, height: int, width: int):
        super().__init__()
        self.height = height
        self.width = width

        # Create meshgrid of pixel coordinates
        meshgrid = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        # (3, H*W) homogeneous pixel coordinates: [u, v, 1]
        pix_coords = torch.stack(
            [meshgrid[1].reshape(-1), meshgrid[0].reshape(-1), torch.ones(height * width)],
            dim=0,
        )
        self.register_buffer("pix_coords", pix_coords.unsqueeze(0))  # (1, 3, H*W)

    def forward(self, depth: torch.Tensor, inv_K: torch.Tensor) -> torch.Tensor:
        """Backproject pixels to 3D camera coordinates.

        Args:
            depth: (B, 1, H, W) depth map.
            inv_K: (B, 4, 4) inverse camera intrinsic matrix.

        Returns:
            (B, 4, H*W) 3D points in homogeneous coordinates.
        """
        B = depth.shape[0]
        # Unproject to 3D: P = D * K_inv @ [u, v, 1]^T
        cam_points = inv_K[:, :3, :3] @ self.pix_coords.expand(B, -1, -1)  # (B, 3, H*W)
        cam_points = depth.view(B, 1, -1) * cam_points  # (B, 3, H*W)
        # Add homogeneous coordinate
        cam_points = torch.cat(
            [cam_points, torch.ones(B, 1, self.height * self.width, device=depth.device)],
            dim=1,
        )  # (B, 4, H*W)
        return cam_points


class Project3D(nn.Module):
    """Project 3D points onto a camera image plane."""

    def __init__(self, height: int, width: int):
        super().__init__()
        self.height = height
        self.width = width

    def forward(
        self, points_3d: torch.Tensor, K: torch.Tensor, T: torch.Tensor
    ) -> torch.Tensor:
        """Project 3D points to 2D pixel coordinates in the source frame.

        Args:
            points_3d: (B, 4, H*W) 3D points in target camera frame.
            K: (B, 4, 4) camera intrinsic matrix.
            T: (B, 4, 4) relative pose (target-to-source transformation).

        Returns:
            (B, H, W, 2) normalized pixel coordinates in [-1, 1] for grid_sample.
        """
        B = points_3d.shape[0]
        # Transform to source frame
        P = K @ T @ points_3d  # (B, 4, H*W)

        # Project: divide by depth (z-coordinate)
        pix_coords = P[:, :2, :] / (P[:, 2:3, :] + 1e-7)  # (B, 2, H*W)

        # Normalize to [-1, 1] for grid_sample
        pix_coords = pix_coords.view(B, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)  # (B, H, W, 2)

        pix_coords[..., 0] = pix_coords[..., 0] / (self.width - 1) * 2 - 1
        pix_coords[..., 1] = pix_coords[..., 1] / (self.height - 1) * 2 - 1

        return pix_coords


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle rotation to rotation matrix (Rodrigues' formula).

    Args:
        axis_angle: (B, 3) axis-angle rotation vectors.

    Returns:
        (B, 3, 3) rotation matrices.
    """
    angle = axis_angle.norm(dim=1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
    axis = axis_angle / (angle.squeeze(-1) + 1e-8)  # (B, 3)

    # Skew-symmetric matrix [axis]_x
    B = axis_angle.shape[0]
    zero = torch.zeros(B, device=axis_angle.device)
    K = torch.stack(
        [
            zero, -axis[:, 2], axis[:, 1],
            axis[:, 2], zero, -axis[:, 0],
            -axis[:, 1], axis[:, 0], zero,
        ],
        dim=1,
    ).view(B, 3, 3)

    # Rodrigues: R = I + sin(θ) * K + (1 - cos(θ)) * K²
    eye = torch.eye(3, device=axis_angle.device).unsqueeze(0)
    R = eye + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
    return R


def pose_vec_to_matrix(axisangle: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Convert 6-DoF pose (axis-angle + translation) to 4x4 transformation matrix.

    Args:
        axisangle: (B, 3) rotation in axis-angle form.
        translation: (B, 3) translation vector.

    Returns:
        (B, 4, 4) transformation matrix.
    """
    B = axisangle.shape[0]
    R = axis_angle_to_matrix(axisangle)  # (B, 3, 3)

    T = torch.zeros(B, 4, 4, device=axisangle.device)
    T[:, :3, :3] = R
    T[:, :3, 3] = translation
    T[:, 3, 3] = 1.0
    return T


class Warper(nn.Module):
    """Warp source images to target view using depth and pose.

    Combines backprojection, transformation, and projection into a
    single differentiable module for self-supervised training.
    """

    def __init__(self, height: int, width: int):
        super().__init__()
        self.backproject = BackprojectDepth(height, width)
        self.project = Project3D(height, width)

    def forward(
        self,
        source_img: torch.Tensor,
        depth: torch.Tensor,
        T: torch.Tensor,
        K: torch.Tensor,
        inv_K: torch.Tensor,
    ) -> torch.Tensor:
        """Warp source image to target view.

        Args:
            source_img: (B, 3, H, W) source RGB image.
            depth: (B, 1, H, W) target frame depth map.
            T: (B, 4, 4) target-to-source transformation matrix.
            K: (B, 4, 4) camera intrinsic matrix.
            inv_K: (B, 4, 4) inverse camera intrinsic matrix.

        Returns:
            (B, 3, H, W) warped source image (reconstructed target view).
        """
        cam_points = self.backproject(depth, inv_K)
        pix_coords = self.project(cam_points, K, T)
        warped = F.grid_sample(
            source_img, pix_coords, mode="bilinear", padding_mode="border", align_corners=True
        )
        return warped
