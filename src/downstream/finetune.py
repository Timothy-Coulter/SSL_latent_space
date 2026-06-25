"""Fine-tune a pretrained SSL backbone on image classification."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2 as transforms

from src.loader import get_classification_dataset
from src.training.algorithms import build_algorithm, get_backbone, load_algorithm_state_dict
from src.training.config import TrainingConfig
from src.training.models import make_backbone

from .config import FinetuneConfig


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_dir(cfg: FinetuneConfig) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Path(cfg.logging.results_dir) / cfg.logging.experiment / f"finetune-{stamp}"


def _tb_dir(cfg: FinetuneConfig, *, stamp: str) -> Path:
    return Path(cfg.logging.tensorboard_dir) / cfg.logging.experiment / f"finetune-{stamp}"


def _maybe_writer(cfg: FinetuneConfig, tb_dir: Path) -> SummaryWriter | None:
    if not cfg.logging.enabled:
        return None
    tb_dir.mkdir(parents=True, exist_ok=False)
    return SummaryWriter(log_dir=str(tb_dir))


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == targets).float().mean().detach().cpu().item())


def _resolve_best_pretrained_checkpoint(cfg: FinetuneConfig) -> Path:
    pretrained = cfg.pretrained
    if pretrained.algorithm is None or pretrained.dataset is None:
        raise ValueError(
            "pretrained.algorithm and pretrained.dataset are required when kind='best'."
        )

    experiment = pretrained.experiment or f"{pretrained.algorithm}_{pretrained.dataset}"
    base = Path(pretrained.results_dir) / experiment
    if not base.exists():
        raise FileNotFoundError(f"No pretraining results dir found at: {base}")

    # Run dirs are created as: <algorithm>-YYYYmmdd-HHMMSS
    prefix = f"{pretrained.algorithm}-"
    candidates = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)])
    if not candidates:
        raise FileNotFoundError(f"No pretraining runs found under: {base}")

    run_dir = candidates[-1]
    ckpt_dir = run_dir / "checkpoints"
    ckpt_name = "best.pt" if pretrained.checkpoint == "best" else "final.pt"
    ckpt_path = ckpt_dir / ckpt_name
    if ckpt_path.exists():
        return ckpt_path

    # Fallback: use final.pt if best.pt is missing.
    fallback = ckpt_dir / "final.pt"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No checkpoint found under: {ckpt_dir}")


def _resolve_pretrained_checkpoint(cfg: FinetuneConfig) -> Path | None:
    pretrained = cfg.pretrained
    if pretrained.kind == "none":
        return None
    if pretrained.kind == "path":
        if not pretrained.path:
            raise ValueError("pretrained.path is required when kind='path'.")
        return Path(pretrained.path)
    return _resolve_best_pretrained_checkpoint(cfg)


def _build_model(cfg: FinetuneConfig, *, device: torch.device) -> nn.Module:
    ckpt_path = _resolve_pretrained_checkpoint(cfg)
    if ckpt_path is None:
        feature_dim = int(cfg.feature_dim)
        backbone = make_backbone(cfg.backbone, feature_dim=feature_dim)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "backbone_state_dict" in ckpt:
            backbone_state = ckpt["backbone_state_dict"]
            feature_dim = int(ckpt["model"]["feature_dim"])
            backbone = make_backbone("simple_cnn", feature_dim=feature_dim)
            backbone.load_state_dict(backbone_state, strict=False)
        else:
            ssl_cfg = TrainingConfig.model_validate(ckpt["cfg"])
            algo = build_algorithm(ssl_cfg.model, ssl_cfg.algorithm)
            load_algorithm_state_dict(algo, ckpt["algo_state"])
            backbone = get_backbone(algo)
            feature_dim = int(ssl_cfg.model.feature_dim)

    head = nn.Linear(feature_dim, cfg.num_classes)
    return nn.Sequential(backbone, head).to(device)


def _build_loaders(cfg: FinetuneConfig) -> tuple[DataLoader, DataLoader]:
    t_train = transforms.Compose(
        [
            transforms.RandomCrop(cfg.data.image_size, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    t_eval = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_ds = get_classification_dataset(
        cfg.data.dataset, root=cfg.data.root, transform=t_train, download=cfg.data.download
    )
    test_ds = get_classification_dataset(
        cfg.data.dataset, root=cfg.data.root, transform=t_eval, download=cfg.data.download
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, test_loader


def _eval_accuracy(model: nn.Module, loader: DataLoader, *, device: torch.device) -> float:
    model.eval()
    accs: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            accs.append(_accuracy(logits, y))
    return float(sum(accs) / max(1, len(accs)))


def _write_run_artifacts(
    run_dir: Path, cfg: FinetuneConfig, config_path: str | Path | None
) -> None:
    if config_path is not None:
        Path(config_path).replace(run_dir / "config.toml")
    (run_dir / "config.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")


def finetune_from_config(cfg: FinetuneConfig, *, config_path: str | Path | None = None) -> Path:
    """Run fine-tuning and return the run directory."""
    device = _resolve_device(cfg.loop.device)
    torch.manual_seed(cfg.loop.seed)
    model = _build_model(cfg, device=device)
    train_loader, test_loader = _build_loaders(cfg)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.logging.results_dir) / cfg.logging.experiment / f"finetune-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_run_artifacts(run_dir, cfg, config_path)

    writer = _maybe_writer(cfg, _tb_dir(cfg, stamp=stamp))

    opt = AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_acc: float | None = None
    final_acc: float | None = None
    epochs_since_improve = 0
    global_step = 0
    best_path = run_dir / "best_finetuned.pt"
    final_path = run_dir / "final_finetuned.pt"

    try:
        for epoch in range(cfg.loop.epochs):
            model.train(True)
            for step, (x, y) in enumerate(train_loader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                loss = loss_fn(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                if cfg.loop.verbose >= 1 and global_step % cfg.loop.log_every_n_steps == 0:
                    print(
                        f"[finetune] epoch={epoch + 1}/{cfg.loop.epochs} "
                        f"step={step + 1} loss={float(loss.detach().cpu().item()):.6f}"
                    )
                if writer is not None and global_step % cfg.loop.log_every_n_steps == 0:
                    writer.add_scalar("train/loss", float(loss.detach().cpu().item()), global_step)
                global_step += 1

            test_acc = _eval_accuracy(model, test_loader, device=device)
            final_acc = test_acc
            if writer is not None:
                writer.add_scalar("eval/acc", test_acc, epoch)

            improved = best_acc is None or (test_acc - best_acc) > cfg.loop.early_stopping_min_delta
            if improved:
                best_acc = test_acc
                epochs_since_improve = 0
                if cfg.logging.save_best_checkpoint:
                    torch.save(
                        {"model_state_dict": model.state_dict(), "best_acc": best_acc}, best_path
                    )
            else:
                epochs_since_improve += 1

            if (
                cfg.loop.early_stopping_enabled
                and epochs_since_improve >= cfg.loop.early_stopping_patience
            ):
                break
    finally:
        if writer is not None:
            writer.flush()
            writer.close()

    torch.save({"model_state_dict": model.state_dict(), "final_acc": final_acc}, final_path)
    metrics = {
        "best_acc": best_acc,
        "best_checkpoint": str(best_path) if best_path.exists() else None,
        "final_acc": final_acc,
        "final_checkpoint": str(final_path),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return run_dir
