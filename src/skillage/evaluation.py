from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from skillage.baselines import (
    EvictionPolicy,
    NoPrunePolicy,
    RandomPolicy,
    RetentionPolicy,
)
from skillage.config import EvalConfig
from skillage.controller import create_controller, require_torch, resolve_device
from skillage.environment import ACTIONS, DecisionState, LifecycleEnvironment
from skillage.errors import ConfigurationError
from skillage.offline_training import OfflineQPolicy
from skillage.runner import SkillFlowRunner
from skillage.types import EvaluationResult, LifecycleAction
from skillage.world import SkillFlowWorld


class LearnedPolicy:
    def __init__(self, checkpoint_path: Path, device: str) -> None:
        torch = require_torch()
        self.torch = torch
        self.device = resolve_device(device)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        if checkpoint.get("schema_version") != 1:
            raise ConfigurationError("unsupported checkpoint schema")
        self.controller = create_controller(
            checkpoint["input_size"], checkpoint["hidden_size"]
        ).to(self.device)
        self.controller.load_state_dict(checkpoint["controller_state"])
        self.controller.eval()

    def actions(self, state: DecisionState) -> dict[str, LifecycleAction]:
        if not state.skill_ids:
            return {}
        with self.torch.no_grad():
            features = self.torch.as_tensor(
                state.features, dtype=self.torch.float32, device=self.device
            )
            mask = self.torch.as_tensor(
                state.legal_mask, dtype=self.torch.bool, device=self.device
            )
            logits, _ = self.controller(features, mask)
            selected = logits.argmax(dim=-1)
        return {
            skill_id: ACTIONS[int(selected[row].item())]
            for row, skill_id in enumerate(state.skill_ids)
        }


def _make_policy(config: EvalConfig, checkpoint_path: Path | None):
    if config.policy in {"skillage", "online"}:
        if checkpoint_path is None:
            raise ConfigurationError("SkillAge-Online evaluation requires a checkpoint")
        return LearnedPolicy(checkpoint_path, config.device)
    if config.policy in {"base_offline", "offline_cql"}:
        if checkpoint_path is None:
            raise ConfigurationError("SkillAge-Base(Offline) evaluation requires a checkpoint")
        return OfflineQPolicy(checkpoint_path, config.device)
    if config.policy == "no_prune":
        return NoPrunePolicy()
    if config.policy == "random":
        return RandomPolicy(config.seed)
    if config.policy in {"lru", "lfu"}:
        return EvictionPolicy(config.policy)
    if config.policy == "retention":
        return RetentionPolicy(config.retention_threshold)
    raise ConfigurationError(f"unknown policy: {config.policy}")


def evaluate_policy(
    warmup_manifest: Path,
    checkpoint_path: Path | None,
    config: EvalConfig,
    output_dir: Path,
    *,
    runner: SkillFlowRunner,
) -> EvaluationResult:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ConfigurationError(f"evaluation output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    world = SkillFlowWorld(
        warmup_manifest,
        runner=runner,
        run_root=output_dir / "world-runs",
        decayed_retrieval_weight=config.decayed_retrieval_weight,
    )
    stream = world.task_stream(config.partition)
    if not stream:
        raise ConfigurationError(f"bundle has no {config.partition} tasks")
    environment = LifecycleEnvironment.from_config(world, config)
    policy = _make_policy(config, checkpoint_path)
    reference_repository = world.new_repository()
    action_counts: Counter[str] = Counter()
    decisions = []
    utilities: list[float] = []
    reference_utilities: list[float] = []
    token_ratios: list[float] = []

    for task in stream:
        state = environment.observe(task)
        actions = policy.actions(state)
        decision, _, _ = environment.apply(state, actions)
        decisions.append(decision)
        utilities.append(state.outcome.performance)
        reference_utilities.append(
            world.execute(
                task,
                reference_repository,
                retrieval_budget=config.retrieval_budget,
            ).performance
        )
        token_ratios.append(float(state.global_summary[1]))
        action_counts.update(action.value for action in actions.values())

    mean_utility = float(np.mean(utilities))
    mean_reference = float(np.mean(reference_utilities))
    metrics: dict[str, float | int] = {
        "mean_utility": mean_utility,
        "utility_retention": min(1.0, mean_utility / max(1e-12, mean_reference)),
        "repository_compression": environment.repository.compression(),
        "token_reduction": 1.0 - float(np.mean(token_ratios)),
        "cumulative_replay_loss": environment.cumulative_degradation,
        "constraint_violations": int(
            environment.cumulative_degradation
            > config.degradation_budget + 1e-12
        ),
    }
    metrics.update({f"action_{name}": count for name, count in action_counts.items()})
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    decisions_path: Path | None = None
    if config.record_decisions:
        decisions_path = output_dir / "decisions.jsonl"
        with decisions_path.open("w", encoding="utf-8", newline="\n") as stream_file:
            for decision in decisions:
                stream_file.write(decision.model_dump_json())
                stream_file.write("\n")
    return EvaluationResult(
        output_dir=output_dir,
        metrics_path=metrics_path,
        decisions_path=decisions_path,
        metrics=metrics,
    )
