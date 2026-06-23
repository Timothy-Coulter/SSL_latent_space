"""Smoke tests for the training loop."""

from __future__ import annotations

from pathlib import Path

from src.training.config import TrainingConfig
from src.training.trainer import train_from_config


def test_train_from_config_creates_run_dir(tmp_path: Path) -> None:
    """Run a tiny CPU training job and ensure it writes artifacts."""
    cfg = TrainingConfig.model_validate(
        {
            "algorithm": {"name": "simclr"},
            "data": {"dataset": "fake", "batch_size": 16, "num_workers": 0, "image_size": 32},
            "loop": {"epochs": 1, "steps_per_epoch": 1, "device": "cpu", "log_every_n_steps": 1},
            "logging": {
                "enabled": False,
                "results_dir": str(tmp_path),
                "experiment": "trainer_smoke",
            },
        }
    )
    run_dir = train_from_config(cfg)
    assert (run_dir / "config.json").exists()
