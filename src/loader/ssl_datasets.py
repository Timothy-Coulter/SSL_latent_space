"""Self-supervised dataset loader helpers.

This module provides a small wrapper that turns common vision datasets into a
two-view dataset (two random/independent transforms of the same image), which
is a common input format for contrastive/self-supervised learning.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar, cast

from torchvision import datasets as tv_datasets

DatasetName = Literal["cifar10", "cifar100", "stl10"]

T_co = TypeVar("T_co", covariant=True)


class DatasetLike(Protocol[T_co]):
    """Structural type for torch-style indexable datasets."""

    def __len__(self) -> int:
        """Return dataset size."""
        ...

    def __getitem__(self, index: int) -> T_co:
        """Return item at `index`."""
        ...


def _normalize_dataset_name(name: str) -> DatasetName:
    normalized = name.strip().lower().replace("_", "-")
    aliases: dict[str, DatasetName] = {
        "cifar10": "cifar10",
        "cifar-10": "cifar10",
        "cifar100": "cifar100",
        "cifar-100": "cifar100",
        "stl10": "stl10",
        "stl-10": "stl10",
    }
    if normalized not in aliases:
        raise ValueError(f"Unknown dataset '{name}'. Expected one of: {sorted(set(aliases))}.")
    return aliases[normalized]


def _coerce_pair_transform(
    transform: Callable[[Any], Any] | Sequence[Callable[[Any], Any]] | None,
) -> tuple[Callable[[Any], Any], Callable[[Any], Any]]:
    if transform is None:
        return (lambda x: x), (lambda x: x)
    if isinstance(transform, Sequence):
        if len(transform) != 2:
            raise ValueError("If `transform` is a sequence, it must have length 2.")
        return transform[0], transform[1]
    return transform, transform


class SSLPairDataset:
    """Wrap a base dataset to return two transformed views of the same image."""

    base: DatasetLike[Any]
    transform1: Callable[[Any], Any]
    transform2: Callable[[Any], Any]
    return_target: bool
    return_index: bool

    def __init__(
        self,
        base: DatasetLike[Any],
        transform1: Callable[[Any], Any],
        transform2: Callable[[Any], Any],
        *,
        return_target: bool = False,
        return_index: bool = False,
    ) -> None:
        """Initialize the dataset wrapper."""
        self.base = base
        self.transform1 = transform1
        self.transform2 = transform2
        self.return_target = return_target
        self.return_index = return_index

    def __len__(self) -> int:
        """Return the number of items in the base dataset."""
        return len(self.base)

    def __getitem__(self, index: int) -> Any:
        """Return two transformed views (and optional metadata) for `index`."""
        sample = self.base[index]
        if isinstance(sample, tuple) and len(sample) >= 2:
            image, target = sample[0], sample[1]
        else:
            image, target = sample, None

        view1 = self.transform1(image)
        view2 = self.transform2(image)

        out: list[Any] = [view1, view2]
        if self.return_target:
            out.append(target)
        if self.return_index:
            out.append(index)
        return tuple(out)


def get_ssl_dataset(
    name: str,
    *,
    root: str | Path = "datasets/data",
    split: str | None = None,
    transform: Callable[[Any], Any] | Sequence[Callable[[Any], Any]] | None = None,
    download: bool = False,
    return_target: bool = False,
    return_index: bool = False,
) -> DatasetLike[Any]:
    """Create a two-view dataset for self-supervised learning.

    Args:
        name: Dataset identifier: "cifar10", "cifar100", or "stl10" (hyphens/underscores allowed).
        root: Dataset root directory. For this repo, extracted archives live in `datasets/data`.
        split: Only used for STL10 (e.g. "unlabeled", "train", "test"). If omitted, defaults to
            "unlabeled" for SSL.
        transform: Either a single transform applied twice, or a sequence of two transforms.
        download: Passed through to torchvision datasets. Defaults to False to avoid network.
        return_target: If True, include the original target/label in each item.
        return_index: If True, include the item index in each item.
    """
    dataset_name = _normalize_dataset_name(name)
    root_path = Path(root)
    t1, t2 = _coerce_pair_transform(transform)

    if dataset_name == "cifar10":
        base: DatasetLike[Any] = tv_datasets.CIFAR10(
            root=str(root_path),
            train=True,
            download=download,
            transform=None,
        )
    elif dataset_name == "cifar100":
        base = tv_datasets.CIFAR100(
            root=str(root_path),
            train=True,
            download=download,
            transform=None,
        )
    else:
        stl_split = split or "unlabeled"
        base = tv_datasets.STL10(
            root=str(root_path),
            split=cast(Any, stl_split),
            download=download,
            transform=None,
        )

    # Some torchvision datasets return NumPy arrays; transforms often expect PIL Images.
    # torchvision's datasets already return PIL for CIFAR and STL10 by default.
    return SSLPairDataset(
        base=base,
        transform1=t1,
        transform2=t2,
        return_target=return_target,
        return_index=return_index,
    )
