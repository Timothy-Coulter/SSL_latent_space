"""Training loop for SSL algorithms."""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, cast

import torch
from torch.optim import AdamW, Optimizer
from torch.utils.tensorboard import SummaryWriter

from .algorithms import SSLStep, algorithm_state_dict, build_algorithm
from .config import TrainingConfig
from .datasets import build_ssl_dataloader
from .schedulers import (
    SchedulerT,
    build_scheduler,
    current_lr,
    is_reduce_on_plateau,
    step_scheduler_batch,
    step_scheduler_epoch,
)

_DATASET_NUM_CLASSES: dict[str, int] = {"cifar10": 10, "cifar100": 100, "stl10": 10}
_DATASET_IMAGE_SIZE: dict[str, int] = {"cifar10": 32, "cifar100": 32, "stl10": 96}


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
    opt: Optimizer,
    scheduler: SchedulerT | None,
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
        if cfg.scheduler.interval == "step":
            step_scheduler_batch(scheduler)

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
            _maybe_add_scalar(writer, "train/lr", current_lr(opt), global_step)
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


def train_from_config(  # noqa: C901
    cfg: TrainingConfig, *, config_path: str | Path | None = None
) -> Path:
    """Run a training job and return the created run directory."""
    device = _resolve_device(cfg.loop.device)
    torch.manual_seed(cfg.loop.seed)

    data = build_ssl_dataloader(cfg.data)
    steps_per_epoch = cfg.loop.steps_per_epoch or data.steps_per_epoch

    algo: SSLStep = build_algorithm(cfg.model, cfg.algorithm)
    algo.to(device).train(True)
    opt = AdamW(algo.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    scheduler = build_scheduler(
        opt,
        cfg=cfg.scheduler,
        total_epochs=cfg.loop.epochs,
        steps_per_epoch=steps_per_epoch,
    )

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
                scheduler=scheduler,
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

            if scheduler is not None and cfg.scheduler.interval == "epoch":
                if is_reduce_on_plateau(scheduler):
                    metric = epoch_loss
                    if cfg.scheduler.monitor == "train/loss":
                        metric = epoch_loss
                    step_scheduler_epoch(scheduler, metric=metric)
                else:
                    step_scheduler_epoch(scheduler)

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

    final_ckpt_path: Path | None = None
    if cfg.checkpoint.save_final:
        final_ckpt_path = _save_checkpoint(
            run_dir, "final.pt", cfg=cfg, algo=algo, epoch=max(0, last_epoch)
        )

    if cfg.post_pretrain_finetune.enabled:
        from src.downstream.config import FinetuneConfig
        from src.downstream.finetune import finetune_from_config

        checkpoints_dir = run_dir / "checkpoints"
        preferred_name = (
            "best.pt" if cfg.post_pretrain_finetune.checkpoint == "best" else "final.pt"
        )
        source_ckpt = checkpoints_dir / preferred_name
        if not source_ckpt.exists():
            fallback = checkpoints_dir / "final.pt"
            if fallback.exists():
                source_ckpt = fallback
            elif final_ckpt_path is not None and final_ckpt_path.exists():
                source_ckpt = final_ckpt_path
            else:
                raise FileNotFoundError(
                    "Post-pretrain finetune is enabled, but no pretrained checkpoint was found."
                )

        copied_ckpt = run_dir / "pretrained_for_finetune.pt"
        shutil.copy2(source_ckpt, copied_ckpt)

        finetune_cfg_path = Path(cfg.post_pretrain_finetune.finetune_config)
        finetune_cfg = FinetuneConfig.from_toml(finetune_cfg_path)
        if cfg.data.dataset in ("cifar10", "cifar100", "stl10"):
            dataset_name = cast(Literal["cifar10", "cifar100", "stl10"], cfg.data.dataset)
        else:
            raise ValueError(
                "Post-pretrain finetune requires a classification dataset "
                f"(got: {cfg.data.dataset})."
            )
        finetune_cfg.pretrained.kind = "path"
        finetune_cfg.pretrained.path = str(copied_ckpt)
        finetune_cfg.data.dataset = dataset_name
        finetune_cfg.data.image_size = _DATASET_IMAGE_SIZE[dataset_name]
        finetune_cfg.num_classes = _DATASET_NUM_CLASSES[dataset_name]
        finetune_cfg.logging.results_dir = str(run_dir)
        finetune_cfg.logging.experiment = f"finetune_{cfg.algorithm.name}_{dataset_name}"
        finetune_cfg.logging.tensorboard_dir = str(run_dir / "tensorboard")
        finetune_from_config(finetune_cfg, config_path=None)
    return run_dir
