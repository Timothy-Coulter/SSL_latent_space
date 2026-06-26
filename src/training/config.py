# mypy: ignore-errors

"""Configuration models for SSL training."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

AlgorithmName = Literal["simclr", "moco", "byol", "swav", "vicreg"]
DatasetName = Literal["cifar10", "cifar100", "stl10", "fake"]


class OptimConfig(BaseModel):
    """Optimizer configuration."""

    lr: float = Field(1e-3, gt=0)
    weight_decay: float = Field(1e-4, ge=0)


class DataConfig(BaseModel):
    """Data pipeline configuration."""

    dataset: DatasetName = "fake"
    root: str = "datasets/data"
    split: str | None = None
    download: bool = False
    batch_size: int = Field(64, gt=0)
    num_workers: int = Field(2, ge=0)
    image_size: int = Field(32, gt=0, description="Used for the 'fake' dataset.")


class TrainLoopConfig(BaseModel):
    """Training loop configuration."""

    epochs: int = Field(1, gt=0)
    steps_per_epoch: int | None = Field(
        None, gt=0, description="Optional cap on steps per epoch (useful for smoke runs)."
    )
    seed: int = 0
    device: str = "cuda"
    log_every_n_steps: int = Field(10, gt=0)
    verbose: int = Field(0, ge=0, le=2, description="0=silent, 1=progress, 2=extra")


class EarlyStoppingConfig(BaseModel):
    """Early stopping settings (based on epoch-average train loss)."""

    enabled: bool = False
    patience: int = Field(10, ge=0, description="Stop after N non-improving epochs.")
    min_delta: float = Field(0.0, ge=0, description="Minimum improvement to reset patience.")


class CheckpointConfig(BaseModel):
    """Checkpointing settings."""

    save_best: bool = False
    save_final: bool = True


class ModelConfig(BaseModel):
    """Backbone + projection settings."""

    backbone: str = Field("simple_cnn", description="Backbone name (template).")
    feature_dim: int = Field(128, gt=0)
    projection_dim: int = Field(128, gt=0)
    hidden_dim: int = Field(256, gt=0)


class AlgorithmConfig(BaseModel):
    """Algorithm-specific configuration."""

    name: AlgorithmName = "simclr"

    # MoCo / queue settings.
    queue_size: int = Field(4096, gt=0)
    momentum: float = Field(0.99, gt=0, lt=1)
    temperature: float = Field(0.2, gt=0)

    # BYOL settings.
    byol_target_tau: float = Field(0.99, gt=0, lt=1)

    # SwAV settings (simplified).
    swav_num_prototypes: int = Field(100, gt=1)
    swav_temperature: float = Field(0.1, gt=0)

    # VICReg settings.
    vicreg_invariance_weight: float = Field(25.0, gt=0)
    vicreg_variance_weight: float = Field(25.0, gt=0)
    vicreg_covariance_weight: float = Field(1.0, gt=0)


class LoggingConfig(BaseModel):
    """Logging/output configuration."""

    enabled: bool = True
    experiment: str = "ssl"
    results_dir: str = "results"
    tensorboard_dir: str = "runs"


class TrainingConfig(BaseModel):
    """Top-level training configuration."""

    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    loop: TrainLoopConfig = Field(default_factory=TrainLoopConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> TrainingConfig:
        """Load a config from a TOML file."""
        config_path = Path(path)
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("TOML root must be a table.")

        model_config_path = raw.get("model_config")
        if model_config_path is not None:
            raw.pop("model_config", None)
            if not isinstance(model_config_path, str):
                raise TypeError("model_config must be a string path.")
            include_path = (config_path.parent / model_config_path).resolve()
            included = tomllib.loads(include_path.read_text(encoding="utf-8"))
            if not isinstance(included, dict):
                raise TypeError("Included model_config TOML root must be a table.")
            included_model = included.get("model")
            if included_model is None:
                raise ValueError(f"Included model_config file has no [model] table: {include_path}")
            if not isinstance(included_model, dict):
                raise TypeError("Included [model] table must be a table.")
            raw_model = raw.get("model") or {}
            if not isinstance(raw_model, dict):
                raise TypeError("[model] must be a table.")
            raw["model"] = {**included_model, **raw_model}
        return cls.model_validate(raw)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return self.model_dump(mode="json")
