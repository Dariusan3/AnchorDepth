"""Test-Time Augmentation (TTA) for Depth Pro.

Improves accuracy at inference time by averaging predictions from
multiple augmented views of the same image. No retraining needed.

Augmentations:
  - Horizontal flip
  - Multi-scale (0.75x, 1.0x, 1.25x relative to model input)
  - Combined flip + multi-scale

Usage:
    from depth_pro.improvements.tta import tta_infer
    prediction = tta_infer(model, image_tensor, f_px=f_px, mode="full")
"""

from typing import Mapping, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _forward_depth(
    model: nn.Module,
    x: torch.Tensor,
    f_px: float,
    original_hw: tuple[int, int],
) -> torch.Tensor:
    """Run model forward and return depth at original resolution."""
    _, _, H, W = x.shape

    # Resize to model input size
    if H != model.img_size or W != model.img_size:
        x = F.interpolate(
            x, size=(model.img_size, model.img_size),
            mode="bilinear", align_corners=False,
        )

    canonical_inverse_depth, _ = model(x)
    inverse_depth = canonical_inverse_depth * (original_hw[1] / f_px)

    # Resize to original resolution
    inverse_depth = F.interpolate(
        inverse_depth, size=original_hw, mode="bilinear", align_corners=False,
    )

    depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)
    return depth


@torch.no_grad()
def tta_infer(
    model: nn.Module,
    x: torch.Tensor,
    f_px: Optional[Union[float, torch.Tensor]] = None,
    mode: str = "full",
    scales: tuple[float, ...] = (0.75, 1.0, 1.25),
) -> Mapping[str, torch.Tensor]:
    """Infer depth with test-time augmentation.

    Args:
        model: DepthPro model in eval mode.
        x: Input image tensor (C, H, W) or (1, C, H, W).
        f_px: Focal length in pixels. If None, uses model's FOV head.
        mode: TTA mode — "flip", "multiscale", "full" (flip + multiscale).
        scales: Scale factors for multi-scale TTA.

    Returns:
        Same dict as model.infer(): {"depth", "focallength_px"}.
    """
    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    _, _, H, W = x.shape

    # If no focal length, estimate it first (single pass, no TTA for FOV)
    if f_px is None:
        x_resized = F.interpolate(
            x, size=(model.img_size, model.img_size),
            mode="bilinear", align_corners=False,
        )
        _, fov_deg = model(x_resized)
        f_px = 0.5 * W / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float)))
        f_px = f_px.squeeze()

    if isinstance(f_px, torch.Tensor):
        f_px_val = f_px.item()
    else:
        f_px_val = float(f_px)

    predictions = []

    if mode in ("flip", "full"):
        # Original
        depth = _forward_depth(model, x, f_px_val, (H, W))
        predictions.append(depth)

        # Horizontal flip
        x_flip = torch.flip(x, dims=[3])
        depth_flip = _forward_depth(model, x_flip, f_px_val, (H, W))
        depth_flip = torch.flip(depth_flip, dims=[3])  # flip back
        predictions.append(depth_flip)

    if mode in ("multiscale", "full"):
        for scale in scales:
            if mode == "full" and scale == 1.0:
                continue  # already have 1.0x from flip pass

            sH, sW = int(H * scale), int(W * scale)
            x_scaled = F.interpolate(
                x, size=(sH, sW), mode="bilinear", align_corners=False,
            )
            # Adjust focal length for scale
            f_px_scaled = f_px_val * scale

            depth_scaled = _forward_depth(model, x_scaled, f_px_scaled, (H, W))
            predictions.append(depth_scaled)

            if mode == "full":
                # Also flip each scale
                x_scaled_flip = torch.flip(x_scaled, dims=[3])
                depth_scaled_flip = _forward_depth(
                    model, x_scaled_flip, f_px_scaled, (H, W),
                )
                depth_scaled_flip = torch.flip(depth_scaled_flip, dims=[3])
                predictions.append(depth_scaled_flip)

    elif mode not in ("flip", "multiscale", "full"):
        raise ValueError(f"Unknown TTA mode: {mode}. Use 'flip', 'multiscale', or 'full'.")

    # If only multiscale (no flip), we need original at 1.0x
    if mode == "multiscale" and 1.0 in scales:
        pass  # already included from scale loop
    elif mode == "multiscale":
        depth = _forward_depth(model, x, f_px_val, (H, W))
        predictions.append(depth)

    # Average all predictions
    depth_avg = torch.stack(predictions, dim=0).mean(dim=0)

    return {
        "depth": depth_avg.squeeze(),
        "focallength_px": f_px if isinstance(f_px, torch.Tensor) else torch.tensor(f_px),
    }
