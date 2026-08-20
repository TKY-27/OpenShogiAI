"""PyTorch value_v0 MLP with a training-only policy-agreement head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from open_shogi_training.models.config import ModelConfig


@dataclass(frozen=True, slots=True)
class DeviceSelection:
    """Resolved torch device and an optional explicit fallback explanation."""

    device: torch.device
    requested: str
    fallback_reason: str | None


class ValueModel(nn.Module):
    """Compact uniform-width MLP with separate scalar training heads."""

    def __init__(self, input_dim: int, config: ModelConfig) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if config.activation != "relu":
            raise ValueError("OSAVAL architecture version 1 supports only relu activation")
        self.input_dim = input_dim
        self.config = config
        linears: list[nn.Linear] = []
        current_dim = input_dim
        for _ in range(config.hidden_layers):
            linears.append(nn.Linear(current_dim, config.hidden_dim))
            current_dim = config.hidden_dim
        self.trunk = nn.ModuleList(linears)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.value_head = nn.Linear(current_dim, 1)
        self.policy_agreement_head = nn.Linear(current_dim, 1)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        hidden = features
        for layer in self.trunk:
            hidden = self.dropout(self.activation(layer(hidden)))
        value = self.value_head(hidden).squeeze(-1)
        policy_agreement_logit = self.policy_agreement_head(hidden).squeeze(-1)
        return value, policy_agreement_logit

    def export_layers(self) -> tuple[nn.Linear, ...]:
        """Return only the inference trunk and value head in execution order."""

        return (*self.trunk, self.value_head)


def select_device(requested: str) -> DeviceSelection:
    """Select CPU/MPS deterministically and make every fallback observable."""

    if requested not in {"auto", "cpu", "mps"}:
        raise ValueError("requested device must be auto, cpu, or mps")
    if requested == "cpu":
        return DeviceSelection(torch.device("cpu"), requested, None)
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_built() and mps_backend.is_available():
        return DeviceSelection(torch.device("mps"), requested, None)
    reason = "MPS is not built or available; using CPU"
    return DeviceSelection(torch.device("cpu"), requested, reason)


def parameter_count(input_dim: int, config: ModelConfig) -> dict[str, int]:
    """Calculate parameter counts without constructing torch objects."""

    trunk = input_dim * config.hidden_dim + config.hidden_dim
    if config.hidden_layers > 1:
        trunk += (config.hidden_layers - 1) * (
            config.hidden_dim * config.hidden_dim + config.hidden_dim
        )
    head = config.hidden_dim + 1
    return {
        "trunk": trunk,
        "valueHead": head,
        "policyAgreementHead": head,
        "trainingTotal": trunk + 2 * head,
        "exportedTotal": trunk + head,
    }


def ensure_finite_model(model: nn.Module, *, include_gradients: bool) -> None:
    """Fail immediately when model parameters or gradients contain NaN/Inf."""

    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all().item():
            raise FloatingPointError(f"model parameter is non-finite: {name}")
        if (
            include_gradients
            and parameter.grad is not None
            and not torch.isfinite(parameter.grad).all().item()
        ):
            raise FloatingPointError(f"model gradient is non-finite: {name}")
