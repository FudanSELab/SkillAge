from __future__ import annotations

import json
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np

from skillage.config import OfflineCQLConfig
from skillage.controller import require_torch, resolve_device
from skillage.environment import ACTIONS, FEATURE_NAMES, DecisionState
from skillage.errors import ConfigurationError
from skillage.offline_data import read_offline_dataset
from skillage.types import LifecycleAction, TrainingArtifacts


ALGORITHM = "skillage_base_offline_cql"
CHECKPOINT_SCHEMA_VERSION = 1


def _seed_everything(seed: int) -> None:
    torch = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _network(input_size: int, hidden_size: int):
    torch = require_torch()
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, len(ACTIONS)),
    )


def _arrays(rows: list[dict[str, object]], split: str, degradation_penalty: float):
    selected = [row for row in rows if row.get("split") == split]
    if not selected:
        raise ConfigurationError(f"offline dataset has no {split} rows")
    feature_size = len(FEATURE_NAMES)
    action_size = len(ACTIONS)
    states = np.asarray([row["state"] for row in selected], dtype=np.float32)
    next_states = np.asarray([row["next_state"] for row in selected], dtype=np.float32)
    masks = np.asarray([row["legal_mask"] for row in selected], dtype=bool)
    next_masks = np.asarray([row["next_legal_mask"] for row in selected], dtype=bool)
    actions = np.asarray([row["action_index"] for row in selected], dtype=np.int64)
    rewards = np.asarray(
        [
            float(row["environment_reward"])
            - degradation_penalty * float(row["degradation_cost"])
            for row in selected
        ],
        dtype=np.float32,
    )
    dones = np.asarray([row["done"] for row in selected], dtype=bool)
    decision_ids = [str(row["decision_id"]) for row in selected]
    decision_sizes = Counter(decision_ids)
    decision_weights = np.asarray(
        [1.0 / decision_sizes[decision_id] for decision_id in decision_ids],
        dtype=np.float32,
    )
    if states.shape[1:] != (feature_size,) or next_states.shape[1:] != (feature_size,):
        raise ConfigurationError("offline transition feature dimension does not match schema")
    if masks.shape[1:] != (action_size,) or next_masks.shape[1:] != (action_size,):
        raise ConfigurationError("offline transition action dimension does not match schema")
    if np.any(~masks[np.arange(len(actions)), actions]):
        raise ConfigurationError("offline dataset contains an illegal logged action")
    if np.any(~np.isfinite(states)) or np.any(~np.isfinite(next_states)):
        raise ConfigurationError("offline dataset contains non-finite features")
    return (
        states,
        next_states,
        masks,
        next_masks,
        actions,
        rewards,
        dones,
        decision_weights,
    )


def _normalization(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = states.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = states.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _objective(
    model,
    target_model,
    batch: tuple[np.ndarray, ...],
    indices: np.ndarray,
    config: OfflineCQLConfig,
    support_mask,
    mean: np.ndarray,
    std: np.ndarray,
):
    torch = require_torch()
    states, next_states, masks, next_masks, actions, rewards, dones, weights = batch
    device = next(model.parameters()).device
    x = torch.as_tensor(
        (states[indices] - mean) / std, dtype=torch.float32, device=device
    )
    nx = torch.as_tensor(
        (next_states[indices] - mean) / std, dtype=torch.float32, device=device
    )
    legal = torch.as_tensor(masks[indices], dtype=torch.bool, device=device)
    next_legal = torch.as_tensor(
        next_masks[indices], dtype=torch.bool, device=device
    ) & support_mask
    action_tensor = torch.as_tensor(actions[indices], dtype=torch.long, device=device)
    reward_tensor = torch.as_tensor(rewards[indices], dtype=torch.float32, device=device)
    done_tensor = torch.as_tensor(dones[indices], dtype=torch.bool, device=device)
    weight_tensor = torch.as_tensor(
        weights[indices], dtype=torch.float32, device=device
    )
    weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1e-12)
    q_values = model(x)
    selected = q_values.gather(1, action_tensor[:, None]).squeeze(1)
    with torch.no_grad():
        online_next = model(nx).masked_fill(~next_legal, -1e9)
        valid_next = next_legal.any(dim=1)
        next_actions = online_next.argmax(dim=1)
        target_next = target_model(nx).gather(1, next_actions[:, None]).squeeze(1)
        target_next = torch.where(valid_next & ~done_tensor, target_next, torch.zeros_like(target_next))
        targets = reward_tensor + config.gamma * target_next
    td_values = torch.nn.functional.huber_loss(
        selected, targets, delta=config.huber_delta, reduction="none"
    )
    td_loss = (td_values * weight_tensor).sum()
    masked_q = q_values.masked_fill(~legal, -1e9)
    cql_values = (
        config.cql_temperature
        * torch.logsumexp(masked_q / config.cql_temperature, dim=1)
        - selected
    )
    cql = (cql_values * weight_tensor).sum()
    return td_loss + config.cql_alpha * cql, td_loss, cql


def train_cql_policy(
    dataset_dir: Path, config: OfflineCQLConfig, output_dir: Path
) -> TrainingArtifacts:
    torch = require_torch()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ConfigurationError(f"training output already exists: {output_dir}")
    manifest, rows = read_offline_dataset(dataset_dir)
    if tuple(manifest.feature_names) != FEATURE_NAMES:
        raise ConfigurationError("offline dataset feature schema is incompatible")
    if tuple(manifest.action_names) != tuple(action.value for action in ACTIONS):
        raise ConfigurationError("offline dataset action schema is incompatible")
    train = _arrays(rows, "train", config.degradation_penalty)
    validation = _arrays(rows, "validation", config.degradation_penalty)
    train_states = train[0]
    mean, std = _normalization(train_states)
    support = np.asarray(
        [bool(np.any(train[4] == index)) for index in range(len(ACTIONS))],
        dtype=bool,
    )
    if int(support.sum()) < 2:
        raise ConfigurationError("offline training data must support at least two actions")

    _seed_everything(config.seed)
    device = resolve_device(config.device)
    model = _network(len(FEATURE_NAMES), config.hidden_size).to(device)
    target_model = _network(len(FEATURE_NAMES), config.hidden_size).to(device)
    target_model.load_state_dict(model.state_dict())
    support_tensor = torch.as_tensor(support, dtype=torch.bool, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(config.seed)
    train_indices = np.arange(len(train[0]), dtype=np.int64)
    validation_indices = np.arange(len(validation[0]), dtype=np.int64)
    best_state = deepcopy(model.state_dict())
    best_value = float("inf")
    patience_left = config.patience
    output_dir.mkdir(parents=True)
    metrics_path = output_dir / "metrics.jsonl"
    checkpoint_path = output_dir / "checkpoint.pt"

    with metrics_path.open("w", encoding="utf-8", newline="\n") as metrics_stream:
        completed = 0
        for update in range(1, config.updates + 1):
            completed = update
            indices = rng.choice(train_indices, size=config.batch_size, replace=True)
            batch_loss, td_loss, cql_loss = _objective(
                model,
                target_model,
                train,
                indices,
                config,
                support_tensor,
                mean,
                std,
            )
            optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            optimizer.step()
            if update % config.target_update_interval == 0:
                target_model.load_state_dict(model.state_dict())

            validation_value = None
            if update % config.validation_interval == 0 or update == config.updates:
                with torch.no_grad():
                    validation_loss, validation_td, validation_cql = _objective(
                        model,
                        target_model,
                        validation,
                        validation_indices,
                        config,
                        support_tensor,
                        mean,
                        std,
                    )
                validation_value = float(validation_loss.cpu())
                if validation_value < best_value - 1e-6:
                    best_value = validation_value
                    best_state = deepcopy(model.state_dict())
                    patience_left = config.patience
                else:
                    patience_left -= 1
                record = {
                    "update": update,
                    "loss": float(batch_loss.detach().cpu()),
                    "td_loss": float(td_loss.detach().cpu()),
                    "cql_loss": float(cql_loss.detach().cpu()),
                    "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
                    "validation_loss": validation_value,
                    "validation_td_loss": float(validation_td.cpu()),
                    "validation_cql_loss": float(validation_cql.cpu()),
                }
                metrics_stream.write(json.dumps(record, sort_keys=True) + "\n")
                metrics_stream.flush()
                if patience_left <= 0:
                    break

        model.load_state_dict(best_state)

    torch.save(
        {
            "algorithm": ALGORITHM,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_state": model.state_dict(),
            "input_size": len(FEATURE_NAMES),
            "hidden_size": config.hidden_size,
            "feature_names": FEATURE_NAMES,
            "action_names": tuple(action.value for action in ACTIONS),
            "support_mask": support,
            "normalization_mean": mean,
            "normalization_std": std,
            "train_config": config.model_dump(mode="json"),
            "updates_completed": completed,
            "torch_random_state": torch.get_rng_state(),
            "python_random_state": random.getstate(),
            "numpy_random_state": rng.bit_generator.state,
        },
        checkpoint_path,
    )
    return TrainingArtifacts(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        updates_completed=completed,
        device=device,
    )


class OfflineQPolicy:
    def __init__(self, checkpoint_path: Path, device: str) -> None:
        torch = require_torch()
        self.torch = torch
        self.device = resolve_device(device)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        if checkpoint.get("algorithm") != ALGORITHM:
            raise ConfigurationError("checkpoint is not a SkillAge-Base(Offline) policy")
        if tuple(checkpoint.get("feature_names", ())) != FEATURE_NAMES:
            raise ConfigurationError("offline checkpoint feature schema is incompatible")
        if tuple(checkpoint.get("action_names", ())) != tuple(action.value for action in ACTIONS):
            raise ConfigurationError("offline checkpoint action schema is incompatible")
        self.model = _network(checkpoint["input_size"], checkpoint["hidden_size"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
        self.std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
        self.support = np.asarray(checkpoint["support_mask"], dtype=bool)

    def actions(self, state: DecisionState) -> dict[str, LifecycleAction]:
        if not state.skill_ids:
            return {}
        with self.torch.no_grad():
            features = self.torch.as_tensor(
                (state.features - self.mean) / self.std,
                dtype=self.torch.float32,
                device=self.device,
            )
            values = self.model(features)
            legal = self.torch.as_tensor(
                state.legal_mask, dtype=self.torch.bool, device=self.device
            )
            supported = self.torch.as_tensor(
                self.support, dtype=self.torch.bool, device=self.device
            )
            deployment_mask = legal & supported[None, :]
            fallback = ~deployment_mask.any(dim=1)
            deployment_mask = self.torch.where(fallback[:, None], legal, deployment_mask)
            selected = values.masked_fill(~deployment_mask, -1e9).argmax(dim=1)
        return {
            skill_id: ACTIONS[int(selected[row].item())]
            for row, skill_id in enumerate(state.skill_ids)
        }
