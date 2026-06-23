"""Smoke tests for SSL algorithm implementations."""

from __future__ import annotations

from typing import cast

import torch

from src.training.algorithms import build_algorithm
from src.training.config import AlgorithmConfig, AlgorithmName, ModelConfig


def test_all_algorithms_forward_backward_cpu() -> None:
    """Run a single forward/backward step for each algorithm on CPU tensors."""
    device = torch.device("cpu")
    model = ModelConfig(backbone="simple_cnn", feature_dim=64, hidden_dim=128, projection_dim=64)

    x1 = torch.randn(4, 3, 32, 32, device=device)
    x2 = torch.randn(4, 3, 32, 32, device=device)

    for name in ["simclr", "moco", "byol", "swav", "vicreg"]:
        algo_cfg = AlgorithmConfig(
            name=cast(AlgorithmName, name),
            queue_size=32,
            swav_num_prototypes=10,
        )
        algo = build_algorithm(model, algo_cfg).to(device).train(True)
        loss = algo.step(x1, x2)
        assert torch.isfinite(loss).all()
        loss.backward()
