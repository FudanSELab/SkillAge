from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "training requires the 'train' extra: install skillage-repro[train]"
        ) from exc
    return torch


def create_controller(input_size: int, hidden_size: int):
    torch = require_torch()
    nn = torch.nn

    class FactorizedActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
            self.actor = nn.Linear(hidden_size, 5)
            self.critic = nn.Sequential(
                nn.Linear(hidden_size + 4, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, features, legal_mask, global_features=None):
            encoded = self.encoder(features)
            logits = self.actor(encoded)
            logits = logits.masked_fill(~legal_mask, float("-inf"))
            if encoded.shape[0]:
                pooled = encoded.mean(dim=0)
                global_features = features[0, -4:]
            else:
                pooled = torch.zeros(
                    self.actor.in_features,
                    dtype=features.dtype,
                    device=features.device,
                )
                if global_features is None:
                    raise ValueError(
                        "global_features are required for an empty repository"
                    )
            value = self.critic(
                torch.cat([pooled, global_features], dim=-1)
            ).squeeze(-1)
            return logits, value

    return FactorizedActorCritic()


def resolve_device(requested: str) -> str:
    torch = require_torch()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested
