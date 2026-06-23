"""Training loop for SSL algorithms."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from .algorithms import SSLStep, build_algorithm
from .config import TrainingConfig
from .datasets import build_ssl_dataloader


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_dir(cfg: TrainingConfig) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Path(cfg.logging.results_dir) / cfg.logging.experiment / f"{cfg.algorithm.name}-{stamp}"


def _tb_dir(cfg: TrainingConfig, *, stamp: str) -> Path:
    return (
        Path(cfg.logging.tensorboard_dir) / cfg.logging.experiment / f"{cfg.algorithm.name}-{stamp}"
    )


def _maybe_create_writer(cfg: TrainingConfig, tb_dir: Path) -> SummaryWriter | None:
    if not cfg.logging.enabled:
        return None
    tb_dir.mkdir(parents=True, exist_ok=False)
    return SummaryWriter(log_dir=str(tb_dir))


def _maybe_add_scalar(writer: SummaryWriter | None, tag: str, value: float, step: int) -> None:
    if writer is None:
        return
    writer.add_scalar(tag, value, step)


def train_from_config(cfg: TrainingConfig, *, config_path: str | Path | None = None) -> Path:
    """Run a training job and return the created run directory."""
    device = _resolve_device(cfg.loop.device)
    torch.manual_seed(cfg.loop.seed)

    data = build_ssl_dataloader(cfg.data)
    steps_per_epoch = cfg.loop.steps_per_epoch or data.steps_per_epoch

    algo: SSLStep = build_algorithm(cfg.model, cfg.algorithm)
    algo.to(device).train(True)

    opt = AdamW(algo.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = (
        Path(cfg.logging.results_dir) / cfg.logging.experiment / f"{cfg.algorithm.name}-{stamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    tb_dir = _tb_dir(cfg, stamp=stamp)

    if config_path is not None:
        Path(config_path).replace(run_dir / "config.toml")
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    writer = _maybe_create_writer(cfg, tb_dir)

    global_step = 0
    try:
        for epoch in range(cfg.loop.epochs):
            for step, (x1, x2) in enumerate(data.loader):
                if step >= steps_per_epoch:
                    break
                x1 = x1.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)

                loss = algo.step(x1, x2)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                should_log = global_step % cfg.loop.log_every_n_steps == 0
                if cfg.loop.verbose >= 1 and should_log:
                    print(
                        f"[train] epoch={epoch + 1}/{cfg.loop.epochs} "
                        f"step={step + 1}/{steps_per_epoch} "
                        f"global_step={global_step} "
                        f"loss={float(loss.detach().cpu().item()):.6f}"
                    )

                if should_log:
                    _maybe_add_scalar(
                        writer, "train/loss", float(loss.detach().cpu().item()), global_step
                    )
                global_step += 1

            _maybe_add_scalar(writer, "train/epoch", float(epoch), global_step)
    finally:
        if writer is not None:
            writer.flush()
            writer.close()

    return run_dir
