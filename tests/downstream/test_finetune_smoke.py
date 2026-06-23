"""Smoke tests for downstream fine-tuning."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.downstream.config import FinetuneConfig
from src.downstream.finetune import finetune_from_config
from src.training.config import TrainingConfig
from src.training.trainer import train_from_config


def _has_cifar10(root: Path) -> bool:
    return bool((root / "cifar-10-batches-py").exists())


def test_finetune_smoke_real_data_gpu_if_available(tmp_path: Path) -> None:
    """Train SSL for 1 step, then fine-tune on CIFAR-10 if available."""
    data_root = Path("datasets/data")
    if not data_root.exists() or not _has_cifar10(data_root):
        pytest.skip("CIFAR-10 not available under datasets/data.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ssl_cfg = TrainingConfig.model_validate(
        {
            "algorithm": {"name": "simclr"},
            "data": {
                "dataset": "cifar10",
                "root": str(data_root),
                "download": False,
                "batch_size": 32,
                "num_workers": 2,
            },
            "loop": {
                "epochs": 1,
                "steps_per_epoch": 1,
                "device": device,
                "log_every_n_steps": 1,
            },
            "early_stopping": {"enabled": False},
            "checkpoint": {"save_best": False, "save_final": True},
            "logging": {
                "enabled": False,
                "results_dir": str(tmp_path),
                "experiment": "ssl_for_finetune_smoke",
            },
        }
    )
    ssl_run_dir = train_from_config(ssl_cfg)
    ckpt_path = ssl_run_dir / "checkpoints" / "final.pt"
    assert ckpt_path.exists()

    ft_cfg = FinetuneConfig.model_validate(
        {
            "pretrained_checkpoint": str(ckpt_path),
            "num_classes": 10,
            "data": {
                "dataset": "cifar10",
                "root": str(data_root),
                "download": False,
                "batch_size": 64,
                "num_workers": 2,
                "image_size": 32,
            },
            "loop": {
                "epochs": 1,
                "device": device,
                "log_every_n_steps": 1,
                "early_stopping_enabled": False,
            },
            "logging": {
                "enabled": False,
                "save_best_checkpoint": False,
                "results_dir": str(tmp_path),
                "experiment": "finetune_smoke",
            },
        }
    )
    run_dir = finetune_from_config(ft_cfg)
    assert (run_dir / "metrics.json").exists()
