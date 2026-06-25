# mypy: ignore-errors

"""Configuration for downstream fine-tuning."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.training.config import AlgorithmName

DownstreamDatasetName = Literal["cifar10", "cifar100", "stl10"]


class DownstreamDataConfig(BaseModel):
    """Dataset and dataloader settings for fine-tuning."""

    dataset: DownstreamDatasetName = "cifar10"
    root: str = "datasets/data"
    download: bool = False
    batch_size: int = Field(128, gt=0)
    num_workers: int = Field(4, ge=0)
    image_size: int = Field(32, gt=0)


class DownstreamLoopConfig(BaseModel):
    """Training loop settings for fine-tuning."""

    epochs: int = Field(5, gt=0)
    seed: int = 0
    device: str = "cuda"
    log_every_n_steps: int = Field(50, gt=0)
    verbose: int = Field(0, ge=0, le=2)

    early_stopping_enabled: bool = True
    early_stopping_patience: int = Field(5, gt=0)
    early_stopping_min_delta: float = Field(0.0, ge=0)


class DownstreamOptimConfig(BaseModel):
    """Optimizer settings for fine-tuning."""

    lr: float = Field(1e-3, gt=0)
    weight_decay: float = Field(1e-4, ge=0)


class DownstreamLoggingConfig(BaseModel):
    """Logging/checkpoint output settings for fine-tuning."""

    enabled: bool = True
    experiment: str = "finetune"
    results_dir: str = "results"
    tensorboard_dir: str = "runs"
    save_best_checkpoint: bool = True


class PretrainedConfig(BaseModel):
    """How to initialize the fine-tuning backbone."""

    kind: Literal["best", "path", "none"] = "best"
    path: str | None = None

    # Used when kind="best".
    algorithm: AlgorithmName | None = None
    dataset: DownstreamDatasetName | None = None
    results_dir: str = "results"
    experiment: str | None = None
    checkpoint: Literal["best", "final"] = "best"


class FinetuneConfig(BaseModel):
    """Top-level fine-tuning configuration."""

    # Backward compatible: old configs can still use this top-level field.
    pretrained_checkpoint: str | None = Field(
        None, description="Deprecated: use [pretrained] instead."
    )
    pretrained: PretrainedConfig = Field(default_factory=PretrainedConfig)

    num_classes: int = Field(10, gt=1)
    backbone: str = Field(
        "simple_cnn", description="Backbone name used when pretrained.kind='none'."
    )
    feature_dim: int = Field(
        128, gt=0, description="Backbone feature dim used when pretrained.kind='none'."
    )

    data: DownstreamDataConfig = Field(default_factory=DownstreamDataConfig)
    loop: DownstreamLoopConfig = Field(default_factory=DownstreamLoopConfig)
    optim: DownstreamOptimConfig = Field(default_factory=DownstreamOptimConfig)
    logging: DownstreamLoggingConfig = Field(default_factory=DownstreamLoggingConfig)

    @model_validator(mode="after")
    def _backfill_pretrained(self) -> FinetuneConfig:
        if (
            self.pretrained_checkpoint
            and self.pretrained.kind == "best"
            and self.pretrained.path is None
        ):
            self.pretrained.kind = "path"
            self.pretrained.path = self.pretrained_checkpoint
        if self.pretrained.kind == "best":
            if self.pretrained.algorithm is None:
                self.pretrained.algorithm = "simclr"
            if self.pretrained.dataset is None:
                self.pretrained.dataset = self.data.dataset
        return self

    @classmethod
    def from_toml(cls, path: str | Path) -> FinetuneConfig:
        """Load a config from a TOML file."""
        config_path = Path(path)
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)
