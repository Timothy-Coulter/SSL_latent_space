"""Smoke tests for the training CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_runs_smoke_config(tmp_path: Path) -> None:
    """Ensure the CLI can execute a tiny CPU run."""
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(
        "\n".join(
            [
                "[algorithm]",
                "name = \"simclr\"",
                "",
                "[data]",
                "dataset = \"fake\"",
                "batch_size = 16",
                "num_workers = 0",
                "image_size = 32",
                "",
                "[loop]",
                "epochs = 1",
                "steps_per_epoch = 1",
                "device = \"cpu\"",
                "log_every_n_steps = 1",
                "",
                "[early_stopping]",
                "enabled = false",
                "",
                "[checkpoint]",
                "save_best = false",
                "",
                "[logging]",
                "enabled = false",
                f"results_dir = \"{tmp_path.as_posix()}\"",
                "experiment = \"cli_smoke\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "-m", "src.training.cli", "--config", str(cfg_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
