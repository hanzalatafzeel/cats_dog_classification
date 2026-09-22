"""
CatDogCNN - a Convolutional Neural Network built entirely from scratch.

Architecture (improved notebook design - 4 double-conv blocks + global pooling):
    Input (128x128x3)
      block 1: Conv(3->32) + BN + ReLU -> Conv(32->32) + BN + ReLU -> MaxPool  (64x64x32)
      block 2: Conv(32->64) + BN + ReLU -> Conv(64->64) + BN + ReLU -> MaxPool  (32x32x64)
      block 3: Conv(64->128) + BN + ReLU -> Conv(128->128) + BN + ReLU -> MaxPool (16x16x128)
      block 4: Conv(128->256) + BN + ReLU -> Conv(256->256) + BN + ReLU -> MaxPool (8x8x256)
      -> AdaptiveAvgPool2d(1) -> Flatten (256)
      -> Dropout -> Linear(256->128) -> ReLU -> Dropout -> Linear(128->1)

This is a from-scratch CNN - no pretrained weights or transfer learning.
"""

import torch
import torch.nn as nn


class CatDogCNN(nn.Module):
    """From-scratch CNN for binary cat/dog classification.

    Architecture matches the checkpoint produced by
    notebooks/cat_dog_cnn_pytorch.ipynb so trained weights load 1:1.
    """

    def __init__(self, image_size: int = 128, arch: dict | None = None):
        super().__init__()
        arch = arch or {}
        conv_dims = arch.get("conv_dims", [32, 64, 128, 256])
        fc = arch.get("fc_units", 128)
        dropout = arch.get("dropout", 0.5)

        # -- Feature extractor: N double-conv blocks with batch norm + pool --
        blocks, in_ch = [], 3
        for out_ch in conv_dims:
            blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)

        # Global average pooling squeezes spatial dims -> fixed-size vector.
        self.pool = nn.AdaptiveAvgPool2d(1)

        # -- Classification head --
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, fc),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc, 1),   # single logit for binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities in [0, 1] (dog confidence)."""
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())