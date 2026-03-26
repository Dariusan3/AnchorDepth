"""Enhanced decoder with Squeeze-and-Excitation (SE) channel attention.

Adds a lightweight SE block between the decoder output and the depth head
to allow selective emphasis of important feature channels. Applied as a
post-decoder attention module to avoid modifying the decoder's memory profile.
"""

from __future__ import annotations

import torch
from torch import nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention.

    Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )
        # Initialize last linear layer to zeros so SE starts as near-identity
        nn.init.zeros_(self.excitation[2].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1, 1)
        return x * w


def add_se_to_model(model: nn.Module, reduction: int = 8) -> None:
    """Add SE attention between decoder output and depth head.

    Modifies model.forward in-place. The SE block recalibrates decoder
    features before they enter the depth head, allowing the model to
    learn which feature channels are most important for depth prediction.

    Args:
        model: A DepthPro model instance.
        reduction: SE reduction ratio (default 8).
    """
    dim = model.decoder.dim_decoder  # 256
    se_block = SEBlock(dim, reduction=reduction)

    # Register as a submodule so it's included in state_dict
    model.se_attention = se_block

    # Store original forward
    original_forward = model.forward

    def enhanced_forward(x: torch.Tensor):
        _, _, H, W = x.shape
        assert H == model.img_size and W == model.img_size

        encodings = model.encoder(x)
        features, features_0 = model.decoder(encodings)

        # Apply SE channel attention before head
        features = model.se_attention(features)

        canonical_inverse_depth = model.head(features)

        fov_deg = None
        if hasattr(model, "fov"):
            fov_deg = model.fov.forward(x, features_0.detach())

        return canonical_inverse_depth, fov_deg

    model.forward = enhanced_forward

    n_params = sum(p.numel() for p in se_block.parameters())
    print(f"Added SE attention block after decoder: {n_params/1e3:.1f}K new parameters")
