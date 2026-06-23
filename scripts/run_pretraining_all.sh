#!/usr/bin/env bash
set -euo pipefail

configs=(
  "configs/byol_cifar10_train.toml"
  "configs/moco_cifar10_train.toml"
  "configs/simclr_cifar10_train.toml"
  "configs/swav_cifar10_train.toml"
  "configs/vicreg_cifar10_train.toml"
)

for cfg in "${configs[@]}"; do
  echo "[pretrain] $cfg"
  uv run python -m src.training.cli --config "$cfg"
done

