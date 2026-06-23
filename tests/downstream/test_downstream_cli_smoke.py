"""Smoke tests for the downstream fine-tuning CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.training.config import TrainingConfig
from src.training.trainer import train_from_config


def _has_cifar10(root: Path) -> bool:
    return bool((root / "cifar-10-batches-py").exists())


def test_downstream_cli_runs_if_data_available(tmp_path: Path) -> None:
    """Run `python -m src.downstream.cli` end-to-end on CIFAR-10 if available."""
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
            "loop": {"epochs": 1, "steps_per_epoch": 1, "device": device, "log_every_n_steps": 1},
            "early_stopping": {"enabled": False},
            "checkpoint": {"save_best": False, "save_final": True},
            "logging": {
                "enabled": False,
                "results_dir": str(tmp_path),
                "experiment": "ssl_cli_seed",
            },
        }
    )
    ssl_run_dir = train_from_config(ssl_cfg)
    ckpt_path = ssl_run_dir / "checkpoints" / "final.pt"
    assert ckpt_path.exists()

    cfg_path = tmp_path / "finetune.toml"
    cfg_path.write_text(
        "\n".join(
            [
                f"pretrained_checkpoint = \"{ckpt_path.as_posix()}\"",
                "num_classes = 10",
                "",
                "[data]",
                "dataset = \"cifar10\"",
                f"root = \"{data_root.as_posix()}\"",
                "download = false",
                "batch_size = 64",
                "num_workers = 2",
                "image_size = 32",
                "",
                "[loop]",
                "epochs = 1",
                f"device = \"{device}\"",
                "log_every_n_steps = 1",
                "early_stopping_enabled = false",
                "",
                "[logging]",
                "enabled = false",
                "save_best_checkpoint = false",
                f"results_dir = \"{tmp_path.as_posix()}\"",
                "experiment = \"downstream_cli_smoke\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "-m", "src.downstream.cli", "--config", str(cfg_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
