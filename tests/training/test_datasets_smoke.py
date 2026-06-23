"""Smoke tests for SSL dataloader construction."""

from __future__ import annotations

import torch

from src.training.config import DataConfig
from src.training.datasets import build_ssl_dataloader


def test_build_ssl_dataloader_fake_yields_tensor_pairs() -> None:
    """Ensure 'fake' dataset yields two views as tensors."""
    data = build_ssl_dataloader(
        DataConfig(dataset="fake", batch_size=8, num_workers=0, image_size=32)
    )
    x1, x2 = next(iter(data.loader))
    assert isinstance(x1, torch.Tensor)
    assert isinstance(x2, torch.Tensor)
    assert x1.shape == x2.shape
    assert x1.ndim == 4
