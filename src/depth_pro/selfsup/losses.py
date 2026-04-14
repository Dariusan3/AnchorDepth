"""Self-supervised losses for monocular depth estimation.

Implements the loss functions from Monodepth2 (Godard et al., ICCV 2019):
- Photometric reprojection loss (L1 + SSIM)
- Per-pixel minimum reprojection across source frames
- Auto-masking for static pixels
- Edge-aware smoothness regularization
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute SSIM between two images using a 3x3 window.

    Args:
        x, y: (B, 3, H, W) images.

    Returns:
        (B, 1, H, W) per-pixel SSIM loss (1 - SSIM, so lower is better).
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)

    sigma_x = F.avg_pool2d(x ** 2, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    )

    return torch.clamp((1 - ssim_map) / 2, 0, 1).mean(dim=1, keepdim=True)


def photometric_loss(
    pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.85
) -> torch.Tensor:
    """Compute photometric reprojection loss (weighted L1 + SSIM).

    Args:
        pred: (B, 3, H, W) reconstructed image.
        target: (B, 3, H, W) original target image.
        alpha: SSIM weight (default 0.85 from Monodepth2).

    Returns:
        (B, 1, H, W) per-pixel photometric loss.
    """
    l1_loss = torch.abs(pred - target).mean(dim=1, keepdim=True)
    ssim_loss = ssim(pred, target)
    return alpha * ssim_loss + (1 - alpha) * l1_loss


def smooth_loss(disp: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    """Edge-aware smoothness loss on disparity (inverse depth).

    Penalizes depth gradients except at image edges, encouraging
    piece-wise smooth depth maps.

    Args:
        disp: (B, 1, H, W) mean-normalized inverse depth.
        img: (B, 3, H, W) corresponding RGB image.

    Returns:
        Scalar smoothness loss.
    """
    # Depth gradients
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    # Image gradients (for edge-awareness)
    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


def compute_selfsup_loss(
    target_img: torch.Tensor,
    source_imgs: list[torch.Tensor],
    warped_imgs: list[torch.Tensor],
    inv_depth: torch.Tensor,
    smoothness_weight: float = 1e-3,
    auto_mask: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute the full self-supervised loss with auto-masking.

    Follows Monodepth2: per-pixel minimum reprojection + auto-masking
    for static pixels + edge-aware smoothness.

    Args:
        target_img: (B, 3, H, W) target RGB image.
        source_imgs: List of (B, 3, H, W) source RGB images (t-1, t+1).
        warped_imgs: List of (B, 3, H, W) warped source images.
        inv_depth: (B, 1, H, W) predicted inverse depth (disparity).
        smoothness_weight: Weight for smoothness loss (default 1e-3).
        auto_mask: Whether to apply auto-masking (disable early in training).

    Returns:
        Dict with 'total', 'photometric', 'smoothness' losses.
    """
    # Photometric losses for each warped source
    reproj_losses = []
    for warped in warped_imgs:
        reproj_losses.append(photometric_loss(warped, target_img))

    # Per-pixel minimum reprojection (handles occlusions)
    reproj_loss = torch.cat(reproj_losses, dim=1)  # (B, N_sources, H, W)
    min_reproj, _ = reproj_loss.min(dim=1, keepdim=True)  # (B, 1, H, W)

    if auto_mask:
        # Auto-masking: compute identity photometric loss (unwarped source vs target)
        identity_losses = []
        for source in source_imgs:
            identity_losses.append(photometric_loss(source, target_img))
        identity_loss = torch.cat(identity_losses, dim=1)
        min_identity, _ = identity_loss.min(dim=1, keepdim=True)

        # Add random noise to break ties (Monodepth2 trick)
        min_identity = min_identity + torch.randn_like(min_identity) * 1e-5

        # Auto-mask: only supervise pixels where warping improves over identity
        mask = (min_reproj < min_identity).float()
        photo_loss = (mask * min_reproj).sum() / (mask.sum() + 1e-7)
        mask_ratio = mask.mean()
    else:
        photo_loss = min_reproj.mean()
        mask_ratio = torch.ones(1, device=inv_depth.device)

    # Edge-aware smoothness on mean-normalized disparity (compute in FP32 to avoid NaN)
    inv_depth_f32 = inv_depth.float()
    inv_depth_f32 = torch.nan_to_num(inv_depth_f32, nan=1.0, posinf=10.0, neginf=1e-6)
    target_f32 = target_img.float()
    mean_disp = inv_depth_f32 / (inv_depth_f32.mean(dim=[2, 3], keepdim=True) + 1e-7)
    mean_disp = torch.clamp(mean_disp, 0, 10)  # prevent extreme values
    s_loss = smooth_loss(mean_disp, target_f32)
    if not torch.isfinite(s_loss):
        s_loss = torch.zeros(1, device=inv_depth.device, dtype=torch.float32).squeeze()

    total = photo_loss + smoothness_weight * s_loss

    return {
        "total": total,
        "photometric": photo_loss,
        "smoothness": s_loss,
        "auto_mask_ratio": mask_ratio,
    }
