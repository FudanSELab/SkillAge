from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RecordModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LifecycleMode(StrEnum):
    ACTIVE = "active"
    DECAYED = "decayed"
    QUARANTINED = "quarantined"
    PROTECTED = "protected"


class LifecycleAction(StrEnum):
    RETAIN = "retain"
    DECAY = "decay"
    REFRESH = "refresh"
    PROTECT = "protect"
    QUARANTINE = "quarantine"


class SkillRecord(RecordModel):
    skill_id: str
    source_model: str
    family_id: str
    name: str
    content: str
    token_length: int = Field(ge=1)
    support_files: dict[str, str] = Field(default_factory=dict)
    source_path: str


class TrainingArtifacts(RecordModel):
    output_dir: Path
    checkpoint_path: Path
    metrics_path: Path
    updates_completed: int
    device: str


class OfflineDatasetArtifacts(RecordModel):
    output_dir: Path
    transitions_path: Path
    manifest_path: Path
    train_count: int = Field(ge=1)
    validation_count: int = Field(ge=1)


class OfflineDatasetManifest(RecordModel):
    schema_version: int
    warmup_manifest_sha256: str
    dataset_revision: str
    runner_version: str
    replay_task_budget: int = Field(ge=1)
    feature_names: tuple[str, ...]
    action_names: tuple[str, ...]
    train_count: int
    validation_count: int
    behavior_policies: tuple[str, ...]
    episodes_per_policy: int
    train_boundary: int
    build_config: dict[str, Any]
    action_counts: dict[str, int]
    files: dict[str, str]


class EvaluationResult(RecordModel):
    output_dir: Path
    metrics_path: Path
    decisions_path: Path | None
    metrics: dict[str, float | int]


class LifecycleSignals(RecordModel):
    invocation_frequency: float
    invocation_performance: float
    contribution: float
    redundancy: float
    context_cost: float
    scarcity: float
    retention_evidence: float
    aging_evidence: float
    retention_score: float


class StepDecision(RecordModel):
    task_id: str
    actions: dict[str, LifecycleAction]
    removed_skill_ids: tuple[str, ...]
    degradation: float
    overhead: float
    performance: float
    remaining_budget: float


class TaskRef(RecordModel):
    task_id: str
    family_id: str
    order: int = Field(ge=0)
    relative_path: str
    instruction_sha256: str


class SkillPatch(RecordModel):
    skill_id: str
    name: str
    family_id: str
    content: str
    support_files: dict[str, str] = Field(default_factory=dict)


class TaskExecution(RecordModel):
    task_id: str
    performance: float = Field(ge=0.0, le=1.0)
    invoked_skill_ids: tuple[str, ...] = ()
    trajectory: list[dict[str, Any]] = Field(default_factory=list)
    verifier_output: dict[str, Any] = Field(default_factory=dict)
    skill_patches: tuple[SkillPatch, ...] = ()
    artifact_paths: tuple[str, ...] = ()


class WarmupManifest(RecordModel):
    schema_version: int = 1
    created_at_utc: str
    source_task_root: str
    source_tree_sha256: str
    dataset_revision: str
    model: str
    base_url: str
    prompt_version: str
    seed: int
    build_config: dict[str, Any]
    warmup_task_ids: tuple[str, ...]
    main_task_ids: tuple[str, ...]
    test_task_ids: tuple[str, ...]
    task_refs: tuple[TaskRef, ...]
    final_skill_ids: tuple[str, ...]
    files: dict[str, str]


class WarmupArtifacts(RecordModel):
    output_dir: Path
    manifest_path: Path
    final_skill_repository: Path
    warmup_task_count: int = Field(ge=1)
    main_task_count: int = Field(ge=1)
