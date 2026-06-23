"""Smoke tests for config parsing."""

from __future__ import annotations

from pathlib import Path

from src.training.config import TrainingConfig


def test_training_config_from_toml(tmp_path: Path) -> None:
    """Ensure TOML config is parsed and validated."""
    path = tmp_path / "cfg.toml"
    path.write_text("[algorithm]\nname = \"simclr\"\n", encoding="utf-8")
    cfg = TrainingConfig.from_toml(path)
    assert cfg.algorithm.name == "simclr"
