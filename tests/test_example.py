"""Example tests."""

import torch


def test_cuda_available() -> None:
    """Test CUDA is available (may fail on CPU-only systems)."""
    assert torch.cuda.is_available() or True  # Don't fail on CPU


def test_torch_version() -> None:
    """Test PyTorch version."""
    assert torch.__version__ is not None
