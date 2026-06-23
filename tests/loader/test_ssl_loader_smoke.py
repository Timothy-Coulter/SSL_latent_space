"""Smoke tests for the self-supervised dataset loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torchvision.transforms import ToTensor

from src.loader import get_ssl_dataset


@pytest.mark.parametrize(  # type: ignore[misc]
    "name,expected_min_size",
    [
        ("cifar10", 32),
        ("cifar100", 32),
        ("stl10", 96),
    ],
)
def test_all_datasets_load(name: str, expected_min_size: int) -> None:
    """Ensure each supported dataset yields two tensor views of the expected minimum size."""
    data_root = Path("datasets/data")
    if not data_root.exists():
        pytest.skip("Local dataset root not present.")

    dataset = get_ssl_dataset(
        name,
        root=data_root,
        transform=ToTensor(),
        download=False,
    )
    assert len(dataset) > 0

    view1, view2 = dataset[0]
    assert isinstance(view1, torch.Tensor)
    assert isinstance(view2, torch.Tensor)
    assert view1.shape == view2.shape
    assert view1.ndim == 3
    assert int(view1.shape[-1]) >= expected_min_size
    assert int(view1.shape[-2]) >= expected_min_size
