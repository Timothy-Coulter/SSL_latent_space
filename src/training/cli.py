"""CLI entry point for SSL training."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import TrainingConfig
from .trainer import train_from_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSL training runner")
    parser.add_argument("--config", type=str, required=True, help="Path to TOML config file")
    return parser.parse_args()


def main() -> None:
    """Run training from a config file."""
    args = _parse_args()
    config_path = Path(args.config)
    cfg = TrainingConfig.from_toml(config_path)
    # Copy the config into the run directory. Keep the original config file in configs/.
    cfg_dict = cfg  # explicit local for debugger friendliness
    temp_copy = config_path.with_suffix(".tmp.toml")
    temp_copy.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    train_from_config(cfg_dict, config_path=temp_copy)


if __name__ == "__main__":
    main()
