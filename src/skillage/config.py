from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FrozenConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SplitConfig(FrozenConfig):
    train_fraction: float = Field(default=0.7, gt=0.0, lt=1.0)
    min_test_tasks: int = Field(default=2, ge=0)


class WarmupConfig(FrozenConfig):
    task_root: Path
    dataset_revision: str
    family_ids: tuple[str, ...] = ()
    model: str = "qwen3.7-plus"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    prompt_version: str = "skillflow-warmup-v1"
    seed: int = 0
    warmup_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    min_tasks_per_family: int = Field(default=1, ge=1)
    max_tasks_per_family: int | None = Field(default=None, ge=2)
    split: SplitConfig = SplitConfig()


class LifecycleRuntimeConfig(FrozenConfig):
    retrieval_budget: int = Field(default=10, ge=1)
    replay_task_budget: int = Field(default=10, ge=1)
    recent_window: int = Field(default=20, ge=1)
    degradation_budget: float = Field(default=0.05, gt=0.0, le=1.0)
    decayed_retrieval_weight: float = Field(default=0.25, ge=0.0, le=1.0)


class TrainConfig(LifecycleRuntimeConfig):
    seed: int = 0
    updates: int = Field(default=1500, ge=1)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    dual_learning_rate: float = Field(default=1e-3, gt=0.0)
    initial_lambda: float = Field(default=0.1, ge=0.0)
    entropy_coefficient: float = Field(default=0.01, ge=0.0)
    critic_coefficient: float = Field(default=0.5, ge=0.0)
    gradient_clip: float = Field(default=1.0, gt=0.0)
    hidden_size: int = Field(default=128, ge=8)
    checkpoint_interval: int = Field(default=100, ge=1)
    device: str = "auto"


class OfflineDataConfig(LifecycleRuntimeConfig):
    seed: int = 0
    episodes_per_policy: int = Field(default=4, ge=1)
    validation_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    behavior_policies: tuple[str, ...] = (
        "no_prune",
        "random",
        "lru",
        "lfu",
        "retention",
    )


class OfflineCQLConfig(FrozenConfig):
    seed: int = 0
    updates: int = Field(default=1500, ge=1)
    batch_size: int = Field(default=256, ge=1)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    cql_alpha: float = Field(default=1.0, ge=0.0)
    cql_temperature: float = Field(default=1.0, gt=0.0)
    huber_delta: float = Field(default=1.0, gt=0.0)
    degradation_penalty: float = Field(default=0.1, ge=0.0)
    gradient_clip: float = Field(default=5.0, gt=0.0)
    hidden_size: int = Field(default=128, ge=8)
    target_update_interval: int = Field(default=100, ge=1)
    validation_interval: int = Field(default=100, ge=1)
    patience: int = Field(default=20, ge=1)
    device: str = "auto"


class EvalConfig(LifecycleRuntimeConfig):
    seed: int = 0
    partition: Literal["main", "test"] = "test"
    policy: Literal[
        "skillage",
        "online",
        "base_offline",
        "offline_cql",
        "no_prune",
        "random",
        "lru",
        "lfu",
        "retention",
    ] = "online"
    retention_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    record_decisions: bool = True
    device: str = "auto"
