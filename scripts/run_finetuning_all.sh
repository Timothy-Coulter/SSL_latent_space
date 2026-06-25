#!/usr/bin/env bash
set -euo pipefail

configs=(
  "configs/finetune_cifar10_simclr_best.toml"
  "configs/finetune_cifar10_moco_best.toml"
  "configs/finetune_cifar10_byol_best.toml"
  "configs/finetune_cifar10_swav_best.toml"
  "configs/finetune_cifar10_vicreg_best.toml"
  "configs/finetune_cifar10_scratch.toml"

  "configs/finetune_cifar100_simclr_best.toml"
  "configs/finetune_cifar100_moco_best.toml"
  "configs/finetune_cifar100_byol_best.toml"
  "configs/finetune_cifar100_swav_best.toml"
  "configs/finetune_cifar100_vicreg_best.toml"
  "configs/finetune_cifar100_scratch.toml"

  "configs/finetune_stl10_simclr_best.toml"
  "configs/finetune_stl10_moco_best.toml"
  "configs/finetune_stl10_byol_best.toml"
  "configs/finetune_stl10_swav_best.toml"
  "configs/finetune_stl10_vicreg_best.toml"
  "configs/finetune_stl10_scratch.toml"
)

for cfg in "${configs[@]}"; do
  echo "[finetune] $cfg"
  uv run python -m src.downstream.cli --config "$cfg"
done

