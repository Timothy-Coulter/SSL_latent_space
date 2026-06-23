"""Smoke test that runs training over all dataset/algorithm combinations."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, assert_never

import pytest

from src.training.config import TrainingConfig
from src.training.trainer import train_from_config

type DatasetName = Literal["fake", "cifar10", "cifar100", "stl10"]
type AlgorithmName = Literal["simclr", "moco", "byol", "swav", "vicreg"]


def _dataset_available(dataset: DatasetName) -> bool:
    data_root = Path("datasets/data")
    if dataset == "fake":
        return True
    if not data_root.exists():
        return False

    match dataset:
        case "cifar10":
            return bool((data_root / "cifar-10-batches-py").exists())
        case "cifar100":
            return bool((data_root / "cifar-100-python").exists())
        case "stl10":
            return bool((data_root / "stl10_binary").exists())
        case _:
            assert_never(dataset)


@pytest.mark.parametrize(  # type: ignore[misc]
    "dataset",
    ["fake", "cifar10", "cifar100", "stl10"],
)
@pytest.mark.parametrize(  # type: ignore[misc]
    "algorithm",
    ["simclr", "moco", "byol", "swav", "vicreg"],
)
def test_train_all_dataset_algorithm_combinations_smoke(
    tmp_path: Path, dataset: DatasetName, algorithm: AlgorithmName
) -> None:
    """Run a single-step CPU training job for each dataset/algorithm pair."""
    if not _dataset_available(dataset):
        pytest.skip(f"Dataset '{dataset}' not available locally.")

    data_root = Path("datasets/data")
    data_cfg: dict[str, object] = {
        "dataset": dataset,
        "batch_size": 16,
        "num_workers": 0,
        "download": False,
    }
    if dataset != "fake":
        data_cfg["root"] = str(data_root)
    if dataset == "stl10":
        data_cfg["split"] = "unlabeled"

    algo_cfg: dict[str, object] = {"name": algorithm}
    if algorithm == "moco":
        algo_cfg["queue_size"] = 128
    if algorithm == "swav":
        algo_cfg["swav_num_prototypes"] = 20

    cfg = TrainingConfig.model_validate(
        {
            "algorithm": algo_cfg,
            "data": data_cfg,
            "loop": {
                "epochs": 1,
                "steps_per_epoch": 1,
                "device": "cpu",
                "log_every_n_steps": 1,
            },
            "early_stopping": {"enabled": False},
            "checkpoint": {"save_best": False, "save_final": True},
            "logging": {
                "enabled": False,
                "results_dir": str(tmp_path),
                "experiment": f"combo_{dataset}_{algorithm}",
            },
        }
    )

    run_dir = train_from_config(cfg)
    assert (run_dir / "config.json").exists()
