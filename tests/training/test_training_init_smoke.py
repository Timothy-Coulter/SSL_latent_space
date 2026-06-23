"""Smoke tests for the training package exports."""

from __future__ import annotations

from src.training import TrainingConfig, train_from_config


def test_training_package_exports() -> None:
    """Ensure public API symbols exist and are callable."""
    assert TrainingConfig is not None
    assert train_from_config is not None
