"""CLI entry point for SSL training."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import TrainingConfig
from .trainer import train_from_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSL training runner")
    parser.add_argument("--config", type=str, required=True, help="Path to TOML config file")
    parser.add_argument(
        "--early-stopping",
        dest="early_stopping",
        action="store_true",
        help="Enable early stopping (epoch-average train loss).",
    )
    parser.add_argument(
        "--no-early-stopping",
        dest="early_stopping",
        action="store_false",
        help="Disable early stopping.",
    )
    parser.set_defaults(early_stopping=None)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Early stopping patience (epochs without improvement).",
    )
    parser.add_argument(
        "--save-best-checkpoint",
        dest="save_best_checkpoint",
        action="store_true",
        help="Save a best-model checkpoint when early stopping is enabled.",
    )
    parser.add_argument(
        "--no-save-best-checkpoint",
        dest="save_best_checkpoint",
        action="store_false",
        help="Do not save a best-model checkpoint.",
    )
    parser.set_defaults(save_best_checkpoint=None)
    return parser.parse_args()


def main() -> None:
    """Run training from a config file."""
    args = _parse_args()
    config_path = Path(args.config)
    cfg = TrainingConfig.from_toml(config_path)

    if args.early_stopping is not None:
        cfg.early_stopping.enabled = bool(args.early_stopping)
    if args.early_stopping_patience is not None:
        cfg.early_stopping.patience = int(args.early_stopping_patience)
    if args.save_best_checkpoint is not None:
        cfg.checkpoint.save_best = bool(args.save_best_checkpoint)

    # Copy the config into the run directory. Keep the original config file in configs/.
    cfg_dict = cfg  # explicit local for debugger friendliness
    temp_copy = config_path.with_suffix(".tmp.toml")
    temp_copy.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    train_from_config(cfg_dict, config_path=temp_copy)


if __name__ == "__main__":
    main()
