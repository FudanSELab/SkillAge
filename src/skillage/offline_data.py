from __future__ import annotations

import json
from collections import Counter
from math import ceil
from pathlib import Path
from typing import Any

from skillage.baselines import (
    ActionPolicy,
    EvictionPolicy,
    NoPrunePolicy,
    RandomPolicy,
    RetentionPolicy,
)
from skillage.config import OfflineDataConfig
from skillage.artifacts import file_sha256
from skillage.environment import ACTIONS, FEATURE_NAMES, DecisionState, LifecycleEnvironment
from skillage.errors import BundleValidationError, ConfigurationError
from skillage.runner import SkillFlowRunner
from skillage.types import OfflineDatasetArtifacts, OfflineDatasetManifest, TaskRef
from skillage.warmup import read_warmup_manifest
from skillage.world import SkillFlowWorld


OFFLINE_SCHEMA_VERSION = 2
TRANSITIONS_FILE = "transitions.jsonl"
MANIFEST_FILE = "manifest.json"
SUPPORTED_BEHAVIOR_POLICIES = {
    "no_prune",
    "random",
    "lru",
    "lfu",
    "retention",
}


def _behavior_policy(name: str, seed: int) -> ActionPolicy:
    if name == "no_prune":
        return NoPrunePolicy()
    if name == "random":
        return RandomPolicy(seed)
    if name in {"lru", "lfu"}:
        return EvictionPolicy(name)
    if name == "retention":
        return RetentionPolicy(0.35)
    raise ConfigurationError(f"unknown offline behavior policy: {name}")


def _validate_config(config: OfflineDataConfig) -> None:
    if not config.behavior_policies:
        raise ConfigurationError("at least one offline behavior policy is required")
    unknown = set(config.behavior_policies) - SUPPORTED_BEHAVIOR_POLICIES
    if unknown:
        raise ConfigurationError(
            f"unknown offline behavior policies: {', '.join(sorted(unknown))}"
        )
    if len(set(config.behavior_policies)) != len(config.behavior_policies):
        raise ConfigurationError("offline behavior policies must be unique")


def _episode_rows(
    world: SkillFlowWorld,
    stream: list[TaskRef],
    train_boundary: int,
    config: OfflineDataConfig,
    behavior_name: str,
    behavior_seed: int,
    episode_id: str,
) -> list[dict[str, Any]]:
    environment = LifecycleEnvironment.from_config(world, config)
    policy = _behavior_policy(behavior_name, behavior_seed)
    state = environment.observe(stream[0])
    rows: list[dict[str, Any]] = []

    for position, _ in enumerate(stream):
        actions = policy.actions(state)
        decision, environment_reward, replay = environment.apply(state, actions)
        stream_terminal = position + 1 == len(stream)
        split_terminal = position + 1 == train_boundary
        next_state = None if stream_terminal else environment.observe(stream[position + 1])
        next_rows = (
            {skill_id: row for row, skill_id in enumerate(next_state.skill_ids)}
            if next_state is not None
            else {}
        )
        split = "train" if position < train_boundary else "validation"
        decision_id = f"{episode_id}:{position}:{state.task_id}"

        for state_row, skill_id in enumerate(state.skill_ids):
            action = actions[skill_id]
            removed = skill_id not in next_rows
            done = stream_terminal or split_terminal or removed
            if done:
                next_features = [0.0] * len(FEATURE_NAMES)
                next_legal_mask = [False] * len(ACTIONS)
            else:
                assert next_state is not None
                next_row = next_rows[skill_id]
                next_features = next_state.features[next_row].astype(float).tolist()
                next_legal_mask = next_state.legal_mask[next_row].astype(bool).tolist()
            rows.append(
                {
                    "schema_version": OFFLINE_SCHEMA_VERSION,
                    "split": split,
                    "episode_id": episode_id,
                    "decision_id": decision_id,
                    "step": position,
                    "task_id": state.task_id,
                    "skill_id": skill_id,
                    "behavior_policy": behavior_name,
                    "behavior_seed": behavior_seed,
                    "state": state.features[state_row].astype(float).tolist(),
                    "legal_mask": state.legal_mask[state_row].astype(bool).tolist(),
                    "action": action.value,
                    "action_index": ACTIONS.index(action),
                    "environment_reward": float(environment_reward),
                    "degradation_cost": float(replay.degradation),
                    "replay_accepted": skill_id in replay.accepted,
                    "replay_rejected": skill_id in replay.rejected,
                    "next_state": next_features,
                    "next_legal_mask": next_legal_mask,
                    "done": done,
                }
            )
        if next_state is not None:
            state = next_state
    return rows


def build_offline_transitions(
    warmup_manifest: Path,
    config: OfflineDataConfig,
    output_dir: Path,
    *,
    runner: SkillFlowRunner,
) -> OfflineDatasetArtifacts:
    _validate_config(config)
    warmup_manifest = Path(warmup_manifest)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ConfigurationError(f"offline dataset output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    warmup = read_warmup_manifest(warmup_manifest)
    world = SkillFlowWorld(
        warmup_manifest,
        runner=runner,
        run_root=output_dir / "world-runs",
        decayed_retrieval_weight=config.decayed_retrieval_weight,
    )
    stream = world.task_stream("main")
    if len(stream) < 2:
        raise ConfigurationError(
            "offline dataset construction requires at least two training tasks"
        )
    validation_count = max(1, ceil(len(stream) * config.validation_fraction))
    train_boundary = len(stream) - validation_count
    if train_boundary < 1:
        raise ConfigurationError("offline train/validation split leaves no training tasks")

    rows: list[dict[str, Any]] = []
    for policy_index, behavior_name in enumerate(config.behavior_policies):
        for episode_index in range(config.episodes_per_policy):
            behavior_seed = config.seed + policy_index * 10_000 + episode_index
            episode_id = f"{behavior_name}:{episode_index}:{behavior_seed}"
            rows.extend(
                _episode_rows(
                    world,
                    stream,
                    train_boundary,
                    config,
                    behavior_name,
                    behavior_seed,
                    episode_id,
                )
            )

    counts = Counter(str(row["split"]) for row in rows)
    action_counts = Counter(str(row["action"]) for row in rows)
    if not counts["train"] or not counts["validation"]:
        raise ConfigurationError("offline dataset produced an empty split")

    transitions_path = output_dir / TRANSITIONS_FILE
    with transitions_path.open("w", encoding="utf-8", newline="\n") as stream_file:
        for row in rows:
            stream_file.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            stream_file.write("\n")

    manifest = OfflineDatasetManifest(
        schema_version=OFFLINE_SCHEMA_VERSION,
        warmup_manifest_sha256=file_sha256(warmup_manifest),
        dataset_revision=warmup.dataset_revision,
        runner_version=runner.runner_version,
        replay_task_budget=config.replay_task_budget,
        feature_names=FEATURE_NAMES,
        action_names=tuple(action.value for action in ACTIONS),
        train_count=counts["train"],
        validation_count=counts["validation"],
        behavior_policies=config.behavior_policies,
        episodes_per_policy=config.episodes_per_policy,
        train_boundary=train_boundary,
        build_config={
            **config.model_dump(mode="json"),
            "warmup_dataset_revision": warmup.dataset_revision,
            "runner_version": runner.runner_version,
        },
        action_counts=dict(sorted(action_counts.items())),
        files={TRANSITIONS_FILE: file_sha256(transitions_path)},
    )
    manifest_path = output_dir / MANIFEST_FILE
    manifest_path.write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return OfflineDatasetArtifacts(
        output_dir=output_dir,
        transitions_path=transitions_path,
        manifest_path=manifest_path,
        train_count=counts["train"],
        validation_count=counts["validation"],
    )


def read_offline_dataset(dataset_dir: Path) -> tuple[OfflineDatasetManifest, list[dict[str, Any]]]:
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        raise BundleValidationError(f"missing offline dataset manifest: {manifest_path}")
    manifest = OfflineDatasetManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.schema_version != OFFLINE_SCHEMA_VERSION:
        raise BundleValidationError(
            f"unsupported offline dataset schema: {manifest.schema_version}"
        )
    transitions_path = dataset_dir / TRANSITIONS_FILE
    expected = manifest.files.get(TRANSITIONS_FILE)
    if not transitions_path.is_file() or expected != file_sha256(transitions_path):
        raise BundleValidationError("offline transition checksum mismatch")
    rows = [
        json.loads(line)
        for line in transitions_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if sum(row.get("split") == "train" for row in rows) != manifest.train_count:
        raise BundleValidationError("offline training row count mismatch")
    if sum(row.get("split") == "validation" for row in rows) != manifest.validation_count:
        raise BundleValidationError("offline validation row count mismatch")
    return manifest, rows
