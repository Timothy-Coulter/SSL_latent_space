"""Smoke tests for model components."""

from __future__ import annotations

import torch

from src.training.models import MLP, SimpleCNN, make_backbone


def test_simple_cnn_forward() -> None:
    """Ensure the backbone produces expected feature shape."""
    model = SimpleCNN(out_dim=32)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 32)


def test_mlp_forward() -> None:
    """Ensure the MLP projects to desired dimension."""
    mlp = MLP(in_dim=16, hidden_dim=32, out_dim=8)
    x = torch.randn(4, 16)
    y = mlp(x)
    assert y.shape == (4, 8)


def test_make_backbone_simple_cnn() -> None:
    """Ensure the backbone factory works."""
    backbone = make_backbone("simple_cnn", feature_dim=16)
    x = torch.randn(2, 3, 32, 32)
    y = backbone(x)
    assert y.shape == (2, 16)
