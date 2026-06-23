# mypy: ignore-errors

"""Implementations of popular SSL objectives.

These are compact, CPU-friendly implementations intended for training
experiments and as a foundation to extend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn
from torch.nn import functional as functional

from .config import AlgorithmConfig, ModelConfig
from .models import MLP, copy_model, make_backbone

# mypy: disable-error-code=misc


class SSLStep(Protocol):
    """Common interface for SSL algorithms."""

    def parameters(self) -> list[nn.Parameter]:
        """Return trainable parameters."""

    def train(self, mode: bool = True) -> SSLStep:
        """Set training mode."""

    def to(self, device: torch.device) -> SSLStep:
        """Move to device."""

    def step(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute the SSL loss for a batch."""


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return functional.normalize(x, dim=1, eps=1e-6)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, *, temperature: float) -> torch.Tensor:
    """Compute SimCLR-style NT-Xent loss for a batch."""
    if z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have the same shape.")
    batch = z1.shape[0]
    if batch < 2:
        raise ValueError("Batch size must be >= 2 for contrastive loss.")

    z = torch.cat([_l2_normalize(z1), _l2_normalize(z2)], dim=0)  # (2B, D)
    logits = (z @ z.t()) / temperature
    logits = logits - torch.diag_embed(torch.diagonal(logits))  # remove self-similarity

    targets = torch.arange(batch, device=z.device)
    targets = torch.cat([targets + batch, targets], dim=0)  # positives index

    loss = functional.cross_entropy(logits, targets)
    return loss


@dataclass
class SimCLR(SSLStep):
    """SimCLR: contrastive learning with NT-Xent."""

    backbone: nn.Module
    projector: nn.Module
    temperature: float

    @classmethod
    def build(cls, model: ModelConfig, algo: AlgorithmConfig) -> SimCLR:
        """Construct an algorithm instance from configs."""
        backbone = make_backbone(model.backbone, feature_dim=model.feature_dim)
        projector = MLP(
            in_dim=model.feature_dim, hidden_dim=model.hidden_dim, out_dim=model.projection_dim
        )
        return cls(backbone=backbone, projector=projector, temperature=algo.temperature)

    def parameters(self) -> list[nn.Parameter]:
        """Return trainable parameters."""
        return list(self.backbone.parameters()) + list(self.projector.parameters())

    def train(self, mode: bool = True) -> SimCLR:
        """Set training mode for modules."""
        self.backbone.train(mode)
        self.projector.train(mode)
        return self

    def to(self, device: torch.device) -> SimCLR:
        """Move modules to the requested device."""
        self.backbone.to(device)
        self.projector.to(device)
        return self

    def step(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute loss for a batch of two views."""
        h1 = self.backbone(x1)
        h2 = self.backbone(x2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        return nt_xent_loss(z1, z2, temperature=self.temperature)


@dataclass
class MoCo(SSLStep):
    """MoCo v1-style objective with a queue and momentum encoder."""

    online_backbone: nn.Module
    online_projector: nn.Module
    target_backbone: nn.Module
    target_projector: nn.Module
    queue: torch.Tensor
    queue_ptr: torch.Tensor
    temperature: float
    momentum: float

    @classmethod
    def build(cls, model: ModelConfig, algo: AlgorithmConfig) -> MoCo:
        """Construct an algorithm instance from configs."""
        online_backbone = make_backbone(model.backbone, feature_dim=model.feature_dim)
        online_projector = MLP(
            in_dim=model.feature_dim, hidden_dim=model.hidden_dim, out_dim=model.projection_dim
        )
        target_backbone = copy_model(online_backbone)
        target_projector = copy_model(online_projector)
        for p in target_backbone.parameters():
            p.requires_grad_(False)
        for p in target_projector.parameters():
            p.requires_grad_(False)
        queue = torch.zeros(algo.queue_size, model.projection_dim)
        queue_ptr = torch.zeros((), dtype=torch.long)
        return cls(
            online_backbone=online_backbone,
            online_projector=online_projector,
            target_backbone=target_backbone,
            target_projector=target_projector,
            queue=queue,
            queue_ptr=queue_ptr,
            temperature=algo.temperature,
            momentum=algo.momentum,
        )

    def parameters(self) -> list[nn.Parameter]:
        """Return trainable parameters (online encoder only)."""
        return list(self.online_backbone.parameters()) + list(self.online_projector.parameters())

    def train(self, mode: bool = True) -> MoCo:
        """Set training mode for modules."""
        self.online_backbone.train(mode)
        self.online_projector.train(mode)
        self.target_backbone.train(False)
        self.target_projector.train(False)
        return self

    def to(self, device: torch.device) -> MoCo:
        """Move modules and queue to the requested device."""
        self.online_backbone.to(device)
        self.online_projector.to(device)
        self.target_backbone.to(device)
        self.target_projector.to(device)
        self.queue = self.queue.to(device)
        self.queue_ptr = self.queue_ptr.to(device)
        return self

    @torch.no_grad()
    def _ema_update(self) -> None:
        for online, target in zip(
            self.online_backbone.parameters(), self.target_backbone.parameters(), strict=True
        ):
            target.data.mul_(self.momentum).add_(online.data, alpha=1.0 - self.momentum)
        for online, target in zip(
            self.online_projector.parameters(), self.target_projector.parameters(), strict=True
        ):
            target.data.mul_(self.momentum).add_(online.data, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        keys = _l2_normalize(keys)
        batch = keys.shape[0]
        qsize = self.queue.shape[0]
        ptr = int(self.queue_ptr.item())

        if batch >= qsize:
            self.queue.copy_(keys[-qsize:])
            self.queue_ptr.fill_(0)
            return

        end = ptr + batch
        if end <= qsize:
            self.queue[ptr:end] = keys
        else:
            first = qsize - ptr
            self.queue[ptr:] = keys[:first]
            self.queue[: end - qsize] = keys[first:]
        self.queue_ptr.fill_(end % qsize)

    def step(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute loss for a batch and update the queue."""
        q = _l2_normalize(self.online_projector(self.online_backbone(x1)))
        with torch.no_grad():
            self._ema_update()
            k = _l2_normalize(self.target_projector(self.target_backbone(x2)))

        # logits: (B, 1+K)
        l_pos = torch.sum(q * k, dim=1, keepdim=True)
        l_neg = q @ _l2_normalize(self.queue).t()
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
        loss = functional.cross_entropy(logits, labels)

        with torch.no_grad():
            self._dequeue_and_enqueue(k)
        return loss


@dataclass
class BYOL(SSLStep):
    """BYOL: online/target networks + predictor with EMA update."""

    online_backbone: nn.Module
    online_projector: nn.Module
    predictor: nn.Module
    target_backbone: nn.Module
    target_projector: nn.Module
    tau: float

    @classmethod
    def build(cls, model: ModelConfig, algo: AlgorithmConfig) -> BYOL:
        """Construct an algorithm instance from configs."""
        online_backbone = make_backbone(model.backbone, feature_dim=model.feature_dim)
        online_projector = MLP(
            in_dim=model.feature_dim, hidden_dim=model.hidden_dim, out_dim=model.projection_dim
        )
        predictor = MLP(
            in_dim=model.projection_dim, hidden_dim=model.hidden_dim, out_dim=model.projection_dim
        )
        target_backbone = copy_model(online_backbone)
        target_projector = copy_model(online_projector)
        for p in target_backbone.parameters():
            p.requires_grad_(False)
        for p in target_projector.parameters():
            p.requires_grad_(False)
        return cls(
            online_backbone=online_backbone,
            online_projector=online_projector,
            predictor=predictor,
            target_backbone=target_backbone,
            target_projector=target_projector,
            tau=algo.byol_target_tau,
        )

    def parameters(self) -> list[nn.Parameter]:
        """Return trainable parameters (online + predictor)."""
        return (
            list(self.online_backbone.parameters())
            + list(self.online_projector.parameters())
            + list(self.predictor.parameters())
        )

    def train(self, mode: bool = True) -> BYOL:
        """Set training mode for modules."""
        self.online_backbone.train(mode)
        self.online_projector.train(mode)
        self.predictor.train(mode)
        self.target_backbone.train(False)
        self.target_projector.train(False)
        return self

    def to(self, device: torch.device) -> BYOL:
        """Move modules to the requested device."""
        self.online_backbone.to(device)
        self.online_projector.to(device)
        self.predictor.to(device)
        self.target_backbone.to(device)
        self.target_projector.to(device)
        return self

    @torch.no_grad()
    def _ema_update(self) -> None:
        for online, target in zip(
            self.online_backbone.parameters(), self.target_backbone.parameters(), strict=True
        ):
            target.data.mul_(self.tau).add_(online.data, alpha=1.0 - self.tau)
        for online, target in zip(
            self.online_projector.parameters(), self.target_projector.parameters(), strict=True
        ):
            target.data.mul_(self.tau).add_(online.data, alpha=1.0 - self.tau)

    def step(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute loss for a batch and update target networks."""
        p1 = self.predictor(self.online_projector(self.online_backbone(x1)))
        p2 = self.predictor(self.online_projector(self.online_backbone(x2)))

        with torch.no_grad():
            self._ema_update()
            z1 = self.target_projector(self.target_backbone(x1))
            z2 = self.target_projector(self.target_backbone(x2))

        p1 = _l2_normalize(p1)
        p2 = _l2_normalize(p2)
        z1 = _l2_normalize(z1)
        z2 = _l2_normalize(z2)

        loss = 2 - 2 * (p1 * z2).sum(dim=1).mean()
        loss = loss + (2 - 2 * (p2 * z1).sum(dim=1).mean())
        return loss * 0.5


@dataclass
class SwAV(SSLStep):
    """Simplified SwAV-style prototype assignment objective."""

    backbone: nn.Module
    projector: nn.Module
    prototypes: nn.Linear
    temperature: float

    @classmethod
    def build(cls, model: ModelConfig, algo: AlgorithmConfig) -> SwAV:
        """Construct an algorithm instance from configs."""
        backbone = make_backbone(model.backbone, feature_dim=model.feature_dim)
        projector = MLP(
            in_dim=model.feature_dim, hidden_dim=model.hidden_dim, out_dim=model.projection_dim
        )
        prototypes = nn.Linear(model.projection_dim, algo.swav_num_prototypes, bias=False)
        return cls(
            backbone=backbone,
            projector=projector,
            prototypes=prototypes,
            temperature=algo.swav_temperature,
        )

    def parameters(self) -> list[nn.Parameter]:
        """Return trainable parameters."""
        return (
            list(self.backbone.parameters())
            + list(self.projector.parameters())
            + list(self.prototypes.parameters())
        )

    def train(self, mode: bool = True) -> SwAV:
        """Set training mode for modules."""
        self.backbone.train(mode)
        self.projector.train(mode)
        self.prototypes.train(mode)
        return self

    def to(self, device: torch.device) -> SwAV:
        """Move modules to the requested device."""
        self.backbone.to(device)
        self.projector.to(device)
        self.prototypes.to(device)
        return self

    def step(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute the prototype assignment consistency loss."""
        z1 = _l2_normalize(self.projector(self.backbone(x1)))
        z2 = _l2_normalize(self.projector(self.backbone(x2)))

        logits1 = self.prototypes(z1) / self.temperature
        logits2 = self.prototypes(z2) / self.temperature

        # Soft assignments (not full Sinkhorn-Knopp; kept fast for template use).
        q1 = functional.softmax(logits1.detach(), dim=1)
        q2 = functional.softmax(logits2.detach(), dim=1)

        loss = -torch.sum(q1 * functional.log_softmax(logits2, dim=1), dim=1).mean()
        loss = loss + (-torch.sum(q2 * functional.log_softmax(logits1, dim=1), dim=1).mean())
        return loss * 0.5


@dataclass
class VICReg(SSLStep):
    """VICReg: invariance + variance + covariance regularization."""

    backbone: nn.Module
    projector: nn.Module
    inv_w: float
    var_w: float
    cov_w: float

    @classmethod
    def build(cls, model: ModelConfig, algo: AlgorithmConfig) -> VICReg:
        """Construct an algorithm instance from configs."""
        backbone = make_backbone(model.backbone, feature_dim=model.feature_dim)
        projector = MLP(
            in_dim=model.feature_dim, hidden_dim=model.hidden_dim, out_dim=model.projection_dim
        )
        return cls(
            backbone=backbone,
            projector=projector,
            inv_w=algo.vicreg_invariance_weight,
            var_w=algo.vicreg_variance_weight,
            cov_w=algo.vicreg_covariance_weight,
        )

    def parameters(self) -> list[nn.Parameter]:
        """Return trainable parameters."""
        return list(self.backbone.parameters()) + list(self.projector.parameters())

    def train(self, mode: bool = True) -> VICReg:
        """Set training mode for modules."""
        self.backbone.train(mode)
        self.projector.train(mode)
        return self

    def to(self, device: torch.device) -> VICReg:
        """Move modules to the requested device."""
        self.backbone.to(device)
        self.projector.to(device)
        return self

    def step(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute the VICReg loss."""
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))

        repr_loss = functional.mse_loss(z1, z2)
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        std_loss = torch.mean(functional.relu(1 - std_z1)) + torch.mean(functional.relu(1 - std_z2))

        z1c = z1 - z1.mean(dim=0)
        z2c = z2 - z2.mean(dim=0)
        cov_z1 = (z1c.t() @ z1c) / (z1.shape[0] - 1)
        cov_z2 = (z2c.t() @ z2c) / (z2.shape[0] - 1)
        cov_loss = (
            _off_diagonal(cov_z1).pow(2).sum() / z1.shape[1]
            + _off_diagonal(cov_z2).pow(2).sum() / z2.shape[1]
        )

        return self.inv_w * repr_loss + self.var_w * std_loss + self.cov_w * cov_loss


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("Expected a square matrix.")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def build_algorithm(model: ModelConfig, algo: AlgorithmConfig) -> SSLStep:
    """Factory that creates an algorithm instance from config."""
    name = algo.name
    if name == "simclr":
        return SimCLR.build(model, algo)
    if name == "moco":
        return MoCo.build(model, algo)
    if name == "byol":
        return BYOL.build(model, algo)
    if name == "swav":
        return SwAV.build(model, algo)
    if name == "vicreg":
        return VICReg.build(model, algo)
    raise ValueError(f"Unknown algorithm '{name}'.")
