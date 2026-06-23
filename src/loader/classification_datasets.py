"""Supervised image classification dataset helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

from torchvision import datasets as tv_datasets

ClassificationDatasetName = Literal["cifar10", "cifar100", "stl10"]


def get_classification_dataset(
    name: str,
    *,
    root: str | Path = "datasets/data",
    split: str | None = None,
    transform: Callable[[Any], Any] | None = None,
    download: bool = False,
) -> Any:
    """Create a torchvision classification dataset.

    Args:
        name: Dataset identifier: "cifar10", "cifar100", or "stl10".
        root: Dataset root directory.
        split: Only used for STL10 ("train" or "test").
        transform: Transform applied to the image.
        download: Whether to download missing data.
    """
    normalized = name.strip().lower().replace("_", "-")
    aliases: dict[str, ClassificationDatasetName] = {
        "cifar10": "cifar10",
        "cifar-10": "cifar10",
        "cifar100": "cifar100",
        "cifar-100": "cifar100",
        "stl10": "stl10",
        "stl-10": "stl10",
    }
    if normalized not in aliases:
        raise ValueError(f"Unknown dataset '{name}'. Expected one of: {sorted(set(aliases))}.")

    ds = aliases[normalized]
    root_path = Path(root)
    if ds == "cifar10":
        return tv_datasets.CIFAR10(
            root=str(root_path), train=True, download=download, transform=transform
        )
    if ds == "cifar100":
        return tv_datasets.CIFAR100(
            root=str(root_path), train=True, download=download, transform=transform
        )

    stl_split = split or "train"
    return tv_datasets.STL10(
        root=str(root_path),
        split=cast(Any, stl_split),
        download=download,
        transform=transform,
    )
