#!/usr/bin/env bash
set -euo pipefail

datasets=("cifar10" "cifar100" "stl10")
algorithms=("simclr" "moco" "byol" "swav" "vicreg")

for dataset in "${datasets[@]}"; do
  for algo in "${algorithms[@]}"; do
    cfg="configs/pretraining/${algo}_${dataset}_train.toml"
    echo "[pretrain] $cfg"
    uv run python -m src.training.cli --config "$cfg"
  done
done
