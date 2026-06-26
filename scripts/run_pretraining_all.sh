#!/usr/bin/env bash
set -euo pipefail

configs=(
  "configs/pretraining/byol_cifar10_train.toml"
  "configs/pretraining/moco_cifar10_train.toml"
  "configs/pretraining/simclr_cifar10_train.toml"
  "configs/pretraining/swav_cifar10_train.toml"
  "configs/pretraining/vicreg_cifar10_train.toml"
)

for cfg in "${configs[@]}"; do
  echo "[pretrain] $cfg"
  uv run python -m src.training.cli --config "$cfg"
done
