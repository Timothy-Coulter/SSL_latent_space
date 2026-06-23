# mypy: ignore-errors

"""Tiny neural network components for SSL algorithms.

This repo is a template and intentionally uses a small backbone by default so
smoke tests and CPU runs stay fast.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """A small CNN backbone that outputs a fixed-size feature vector."""

    def __init__(self, *, out_dim: int) -> None:
        """Initialize the encoder and output layer."""
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an image batch into feature vectors."""
        features = self.encoder(x)
        features = torch.flatten(features, 1)
        return self.fc(features)


class MLP(nn.Module):
    """A 2-layer MLP used as projector/predictor."""

    def __init__(self, *, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        """Initialize the MLP layers."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input vectors."""
        return self.net(x)


def make_backbone(name: str, *, feature_dim: int) -> nn.Module:
    """Create a backbone by name."""
    if name == "simple_cnn":
        return SimpleCNN(out_dim=feature_dim)
    raise ValueError(f"Unknown backbone '{name}'.")


def copy_model(model: nn.Module) -> nn.Module:
    """Deepcopy a module (weights + buffers)."""
    return copy.deepcopy(model)
