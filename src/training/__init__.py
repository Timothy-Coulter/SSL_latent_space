"""Self-supervised learning (SSL) training package."""

from __future__ import annotations

from .config import TrainingConfig
from .trainer import train_from_config

__all__ = ["TrainingConfig", "train_from_config"]
