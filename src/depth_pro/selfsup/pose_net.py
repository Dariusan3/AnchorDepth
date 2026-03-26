"""PoseNet for ego-motion estimation between frame pairs.

Lightweight ResNet-18-based network that takes two concatenated RGB frames
and outputs 6-DoF relative camera pose (3 axis-angle rotation + 3 translation).
Following Monodepth2 (Godard et al., ICCV 2019).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class PoseNet(nn.Module):
    """Estimate 6-DoF relative pose between two frames.

    Architecture: ResNet-18 encoder (modified for 6-channel input) followed
    by a small convolutional pose decoder. Outputs are scaled by 0.01 to
    prevent large initial poses.
    """

    def __init__(self, num_input_frames: int = 2):
        super().__init__()
        self.num_input_frames = num_input_frames

        # ResNet-18 encoder (pretrained)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Modify first conv to accept 6 channels (2 concatenated RGB frames)
        original_conv = resnet.conv1
        self.conv1 = nn.Conv2d(
            num_input_frames * 3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Initialize: copy pretrained weights, average across input frames
        with torch.no_grad():
            self.conv1.weight[:, :3] = original_conv.weight / num_input_frames
            self.conv1.weight[:, 3:6] = original_conv.weight / num_input_frames

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

        # Pose decoder: conv layers to predict 6-DoF pose
        self.pose_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 6 * (num_input_frames - 1), kernel_size=1),
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """Predict relative pose from target to source frame.

        Args:
            target: (B, 3, H, W) target RGB image.
            source: (B, 3, H, W) source RGB image.

        Returns:
            (B, 6) pose vector: [axis_angle(3), translation(3)].
            Translation is scaled by 0.01 to prevent large initial values.
        """
        x = torch.cat([target, source], dim=1)  # (B, 6, H, W)

        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B, 512, H/32, W/32)

        # Pose prediction
        x = self.pose_head(x)  # (B, 6, H/32, W/32)
        x = self.global_pool(x)  # (B, 6, 1, 1)
        x = x.view(x.shape[0], -1)  # (B, 6)

        # Scale translation to prevent large initial poses
        x = 0.01 * x

        return x
