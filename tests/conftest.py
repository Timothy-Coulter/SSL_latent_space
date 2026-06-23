"""Pytest configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Ensure the repo root is importable so tests can import local modules (e.g. `import src`).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# mypy cannot type pytest decorators without the pytest plugin.
@pytest.fixture  # type: ignore[misc]
def device() -> torch.device:
    """Return available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
