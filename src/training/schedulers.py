"""Learning-rate scheduler utilities shared by pretraining and fine-tuning."""

from __future__ import annotations

import math
from typing import Literal, Protocol, cast

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LinearLR,
    LRScheduler,
    MultiStepLR,
    OneCycleLR,
    PolynomialLR,
    ReduceLROnPlateau,
    StepLR,
)

SchedulerName = Literal[
    "none",
    "step",
    "multistep",
    "exponential",
    "cosine",
    "cosine_warm_restarts",
    "linear",
    "polynomial",
    "onecycle",
    "reduce_on_plateau",
    "warmup_cosine",
    "warmup_linear",
    "constant",
]

SchedulerInterval = Literal["epoch", "step"]
SchedulerMode = Literal["min", "max"]


class SchedulerConfigLike(Protocol):
    """Protocol for scheduler config objects."""

    name: SchedulerName
    interval: SchedulerInterval
    mode: SchedulerMode
    monitor: str
    step_size: int
    gamma: float
    milestones: list[int]
    t_max: int | None
    t_0: int
    t_mult: int
    eta_min: float
    warmup_steps: int
    warmup_ratio: float
    power: float
    factor: float
    patience: int
    threshold: float
    cooldown: int
    min_lr: float
    max_lr: float | None
    pct_start: float
    div_factor: float
    final_div_factor: float
    total_steps: int | None


type SchedulerT = LRScheduler | ReduceLROnPlateau


def build_scheduler(  # noqa: C901
    optimizer: Optimizer,
    *,
    cfg: SchedulerConfigLike,
    total_epochs: int,
    steps_per_epoch: int,
) -> SchedulerT | None:
    """Build a PyTorch LR scheduler from config arguments."""
    if cfg.name == "none":
        return None

    total_steps = max(1, cfg.total_steps or (total_epochs * steps_per_epoch))
    default_t_max = max(1, total_steps if cfg.interval == "step" else total_epochs)
    t_max = max(1, cfg.t_max or default_t_max)

    if cfg.name == "step":
        return cast(SchedulerT, StepLR(optimizer, step_size=max(1, cfg.step_size), gamma=cfg.gamma))
    if cfg.name == "multistep":
        milestones = sorted({m for m in cfg.milestones if m > 0})
        return cast(SchedulerT, MultiStepLR(optimizer, milestones=milestones, gamma=cfg.gamma))
    if cfg.name == "exponential":
        return cast(SchedulerT, ExponentialLR(optimizer, gamma=cfg.gamma))
    if cfg.name == "cosine":
        return cast(SchedulerT, CosineAnnealingLR(optimizer, T_max=t_max, eta_min=cfg.eta_min))
    if cfg.name == "cosine_warm_restarts":
        return cast(
            SchedulerT,
            CosineAnnealingWarmRestarts(
                optimizer,
                T_0=max(1, cfg.t_0),
                T_mult=max(1, cfg.t_mult),
                eta_min=cfg.eta_min,
            ),
        )
    if cfg.name == "linear":
        return cast(
            SchedulerT, LinearLR(optimizer, start_factor=cfg.warmup_ratio, total_iters=t_max)
        )
    if cfg.name == "polynomial":
        return cast(SchedulerT, PolynomialLR(optimizer, total_iters=t_max, power=cfg.power))
    if cfg.name == "constant":
        return cast(SchedulerT, ConstantLR(optimizer, factor=1.0, total_iters=max(1, t_max)))
    if cfg.name == "onecycle":
        max_lr = cfg.max_lr if cfg.max_lr is not None else float(optimizer.param_groups[0]["lr"])
        return cast(
            SchedulerT,
            OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=cfg.pct_start,
                div_factor=cfg.div_factor,
                final_div_factor=cfg.final_div_factor,
            ),
        )
    if cfg.name == "reduce_on_plateau":
        return cast(
            SchedulerT,
            ReduceLROnPlateau(
                optimizer,
                mode=cfg.mode,
                factor=cfg.factor,
                patience=cfg.patience,
                threshold=cfg.threshold,
                cooldown=cfg.cooldown,
                min_lr=cfg.min_lr,
            ),
        )
    if cfg.name == "warmup_cosine":

        def lr_lambda(step: int) -> float:
            warmup = max(1, cfg.warmup_steps)
            if step < warmup:
                alpha = step / warmup
                return cfg.warmup_ratio + alpha * (1.0 - cfg.warmup_ratio)
            progress = (step - warmup) / max(1, total_steps - warmup)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return cosine if cosine > 0.0 else 0.0

        from torch.optim.lr_scheduler import LambdaLR

        return cast(SchedulerT, LambdaLR(optimizer, lr_lambda=lr_lambda))
    if cfg.name == "warmup_linear":

        def lr_lambda(step: int) -> float:
            warmup = max(1, cfg.warmup_steps)
            if step < warmup:
                alpha = step / warmup
                return cfg.warmup_ratio + alpha * (1.0 - cfg.warmup_ratio)
            progress = (step - warmup) / max(1, total_steps - warmup)
            linear = 1.0 - progress
            return linear if linear > 0.0 else 0.0

        from torch.optim.lr_scheduler import LambdaLR

        return cast(SchedulerT, LambdaLR(optimizer, lr_lambda=lr_lambda))

    raise ValueError(f"Unknown scheduler name: {cfg.name}")


def is_reduce_on_plateau(scheduler: object | None) -> bool:
    """Return True if scheduler is ReduceLROnPlateau."""
    return isinstance(scheduler, ReduceLROnPlateau)


def step_scheduler_batch(scheduler: SchedulerT | None) -> None:
    """Step scheduler for per-batch schedules."""
    if scheduler is None or is_reduce_on_plateau(scheduler):
        return
    scheduler.step()


def step_scheduler_epoch(scheduler: SchedulerT | None, *, metric: float | None = None) -> None:
    """Step scheduler for per-epoch schedules."""
    if scheduler is None:
        return
    if is_reduce_on_plateau(scheduler):
        scheduler.step(0.0 if metric is None else metric)
    else:
        scheduler.step()


def current_lr(optimizer: Optimizer) -> float:
    """Get current LR from the first param group."""
    return float(optimizer.param_groups[0]["lr"])
