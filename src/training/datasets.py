# mypy: ignore-errors

"""Dataset + augmentation helpers for SSL training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as tv_datasets
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as transforms

from src.loader.ssl_datasets import SSLPairDataset, get_ssl_dataset

from .config import DataConfig

# mypy: disable-error-code=misc


@dataclass(frozen=True)
class SSLData:
    """Constructed dataloader + metadata."""

    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    steps_per_epoch: int


def _default_ssl_transforms(image_size: int) -> Callable[[Any], torch.Tensor]:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, interpolation=InterpolationMode.BICUBIC, antialias=True
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def _as_tensor_pair(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise TypeError("Expected dataset to yield (view1, view2, ...).")
    x1, x2 = batch[0], batch[1]
    if not isinstance(x1, torch.Tensor) or not isinstance(x2, torch.Tensor):
        raise TypeError("Expected views to be torch tensors.")
    return x1, x2


class _ViewOnlyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, base: Any) -> None:
        """Wrap a dataset that may return extra metadata."""
        self.base = base

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return just the two views for an index."""
        return _as_tensor_pair(self.base[idx])


def build_ssl_dataloader(cfg: DataConfig) -> SSLData:
    """Build a 2-view SSL dataloader from config."""
    dataset: Any
    if cfg.dataset == "fake":
        base = tv_datasets.FakeData(
            size=256,
            image_size=(3, cfg.image_size, cfg.image_size),
            num_classes=10,
            transform=None,
        )
        t = _default_ssl_transforms(cfg.image_size)
        dataset = SSLPairDataset(base=base, transform1=t, transform2=t)
    else:
        t = _default_ssl_transforms(96 if cfg.dataset == "stl10" else 32)
        dataset = get_ssl_dataset(
            cfg.dataset,
            root=Path(cfg.root),
            split=cfg.split,
            transform=t,
            download=cfg.download,
            return_target=False,
            return_index=False,
        )

    view_ds = _ViewOnlyDataset(dataset)
    loader = DataLoader(
        view_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    steps_per_epoch = max(1, len(loader))
    return SSLData(loader=loader, steps_per_epoch=steps_per_epoch)
