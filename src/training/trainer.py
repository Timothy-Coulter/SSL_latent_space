"""Training loop for SSL algorithms."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from .algorithms import SSLStep, algorithm_state_dict, build_algorithm
from .config import TrainingConfig
from .datasets import build_ssl_dataloader


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def _save_checkpoint(
    run_dir: Path, name: str, *, cfg: TrainingConfig, algo: SSLStep, epoch: int
) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "cfg": cfg.to_dict(),
        "algo_state": algorithm_state_dict(algo),
    }
    path = ckpt_dir / name
    torch.save(payload, path)
    return path


def _train_one_epoch(
    *,
    cfg: TrainingConfig,
    algo: SSLStep,
    opt: AdamW,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    writer: SummaryWriter | None,
    steps_per_epoch: int,
    global_step: int,
    epoch: int,
) -> tuple[float, int]:
    epoch_loss_sum = 0.0
    epoch_steps = 0

    for step, (x1, x2) in enumerate(loader):
        if step >= steps_per_epoch:
            break
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        loss = algo.step(x1, x2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        loss_value = float(loss.detach().cpu().item())
        epoch_loss_sum += loss_value
        epoch_steps += 1

        should_log = global_step % cfg.loop.log_every_n_steps == 0
        if cfg.loop.verbose >= 1 and should_log:
            print(
                f"[train] epoch={epoch + 1}/{cfg.loop.epochs} "
                f"step={step + 1}/{steps_per_epoch} "
                f"global_step={global_step} "
                f"loss={loss_value:.6f}"
            )
        if should_log:
            _maybe_add_scalar(writer, "train/loss", loss_value, global_step)
        global_step += 1

    epoch_loss = epoch_loss_sum / max(1, epoch_steps)
    _maybe_add_scalar(writer, "train/epoch_loss", float(epoch_loss), global_step)
    _maybe_add_scalar(writer, "train/epoch", float(epoch), global_step)
    return epoch_loss, global_step


def _update_early_stopping(
    *,
    cfg: TrainingConfig,
    epoch_loss: float,
    best_epoch_loss: float | None,
    epochs_since_improve: int,
) -> tuple[bool, float | None, int]:
    if best_epoch_loss is None:
        return True, epoch_loss, 0
    improved = (best_epoch_loss - epoch_loss) > cfg.early_stopping.min_delta
    if improved:
        return True, epoch_loss, 0
    return False, best_epoch_loss, epochs_since_improve + 1


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
    best_epoch_loss: float | None = None
    epochs_since_improve = 0
    last_epoch = -1

    try:
        for epoch in range(cfg.loop.epochs):
            last_epoch = epoch
            epoch_loss, global_step = _train_one_epoch(
                cfg=cfg,
                algo=algo,
                opt=opt,
                loader=data.loader,
                device=device,
                writer=writer,
                steps_per_epoch=steps_per_epoch,
                global_step=global_step,
                epoch=epoch,
            )

            improved, best_epoch_loss, epochs_since_improve = _update_early_stopping(
                cfg=cfg,
                epoch_loss=epoch_loss,
                best_epoch_loss=best_epoch_loss,
                epochs_since_improve=epochs_since_improve,
            )

            if improved and cfg.checkpoint.save_best:
                _save_checkpoint(run_dir, "best.pt", cfg=cfg, algo=algo, epoch=epoch)

            if cfg.early_stopping.enabled and epochs_since_improve > cfg.early_stopping.patience:
                if cfg.loop.verbose >= 1:
                    print(
                        f"[train] early stopping at epoch={epoch + 1} "
                        f"(best_epoch_loss={best_epoch_loss}, epoch_loss={epoch_loss})"
                    )
                break
    finally:
        if writer is not None:
            writer.flush()
            writer.close()

    if cfg.checkpoint.save_final:
        _save_checkpoint(run_dir, "final.pt", cfg=cfg, algo=algo, epoch=max(0, last_epoch))
    return run_dir
