"""Smoke tests that exercise real datasets and (optionally) GPU."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.training.config import TrainingConfig
from src.training.trainer import train_from_config


def _has_real_datasets(root: Path) -> bool:
    return bool((root / "cifar-10-batches-py").exists() or (root / "cifar-100-python").exists())


def test_real_data_cpu_smoke(tmp_path: Path) -> None:
    """Run a tiny CPU training loop on a real dataset if available locally."""
    data_root = Path("datasets/data")
    if not data_root.exists() or not _has_real_datasets(data_root):
        pytest.skip("Real datasets not present under datasets/data.")

    cfg = TrainingConfig.model_validate(
        {
            "algorithm": {"name": "simclr"},
            "data": {
                "dataset": "cifar10",
                "root": str(data_root),
                "download": False,
                "batch_size": 32,
                "num_workers": 0,
            },
            "loop": {
                "epochs": 1,
                "steps_per_epoch": 1,
                "device": "cpu",
                "log_every_n_steps": 1,
            },
            "early_stopping": {"enabled": False},
            "checkpoint": {"save_best": False, "save_final": True},
            "logging": {
                "enabled": False,
                "results_dir": str(tmp_path),
                "experiment": "real_cpu_smoke",
            },
        }
    )

    run_dir = train_from_config(cfg)
    assert (run_dir / "config.json").exists()


def test_real_data_gpu_smoke_if_available(tmp_path: Path) -> None:
    """Run a tiny GPU training loop on a real dataset if CUDA + data exist."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available.")

    data_root = Path("datasets/data")
    if not data_root.exists() or not _has_real_datasets(data_root):
        pytest.skip("Real datasets not present under datasets/data.")

    cfg = TrainingConfig.model_validate(
        {
            "algorithm": {"name": "moco", "queue_size": 128},
            "data": {
                "dataset": "cifar10",
                "root": str(data_root),
                "download": False,
                "batch_size": 64,
                "num_workers": 2,
            },
            "loop": {
                "epochs": 1,
                "steps_per_epoch": 1,
                "device": "cuda",
                "log_every_n_steps": 1,
            },
            "early_stopping": {"enabled": False},
            "checkpoint": {"save_best": False, "save_final": True},
            "logging": {
                "enabled": False,
                "results_dir": str(tmp_path),
                "experiment": "real_gpu_smoke",
            },
        }
    )

    run_dir = train_from_config(cfg)
    assert (run_dir / "config.json").exists()
