"""Pytest configuration."""

import pytest
import torch


# mypy cannot type pytest decorators without the pytest plugin.
@pytest.fixture  # type: ignore[misc]
def device() -> torch.device:
    """Return available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
