"""Smoke tests for resolving pretrained checkpoints in fine-tuning configs."""

from __future__ import annotations

from pathlib import Path

import torch

from src.downstream.config import FinetuneConfig
from src.downstream.finetune import _resolve_pretrained_checkpoint


def _write_dummy_ssl_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": 0,
            "cfg": {"algorithm": {"name": "simclr"}, "data": {"dataset": "cifar10"}},
            "algo_state": {},
        },
        path,
    )


def test_resolve_best_picks_latest_run(tmp_path: Path) -> None:
    """Resolve kind='best' by choosing the most recent run directory."""
    base = tmp_path / "results" / "simclr_cifar10"
    ckpt_old = base / "simclr-20240101-000000" / "checkpoints" / "best.pt"
    ckpt_new = base / "simclr-20250101-000000" / "checkpoints" / "best.pt"
    _write_dummy_ssl_checkpoint(ckpt_old)
    _write_dummy_ssl_checkpoint(ckpt_new)

    cfg = FinetuneConfig.model_validate(
        {
            "num_classes": 10,
            "pretrained": {
                "kind": "best",
                "algorithm": "simclr",
                "dataset": "cifar10",
                "results_dir": str(tmp_path / "results"),
            },
            "data": {"dataset": "cifar10"},
        }
    )
    resolved = _resolve_pretrained_checkpoint(cfg)
    assert resolved is not None
    assert resolved.resolve() == ckpt_new.resolve()


def test_resolve_none_returns_none(tmp_path: Path) -> None:
    """Resolve kind='none' to no checkpoint (random init)."""
    cfg = FinetuneConfig.model_validate(
        {
            "num_classes": 10,
            "pretrained": {"kind": "none"},
            "data": {"dataset": "cifar10"},
        }
    )
    assert _resolve_pretrained_checkpoint(cfg) is None
