"""CLI entry point for downstream fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import FinetuneConfig
from .finetune import finetune_from_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downstream fine-tuning runner")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to TOML fine-tune config file"
    )
    return parser.parse_args()


def main() -> None:
    """Run downstream fine-tuning from a config file."""
    args = _parse_args()
    config_path = Path(args.config)
    cfg = FinetuneConfig.from_toml(config_path)
    temp_copy = config_path.with_suffix(".tmp.toml")
    temp_copy.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    finetune_from_config(cfg, config_path=temp_copy)


if __name__ == "__main__":
    main()
