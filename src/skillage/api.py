from pathlib import Path

from skillage.config import (
    EvalConfig,
    OfflineCQLConfig,
    OfflineDataConfig,
    TrainConfig,
    WarmupConfig,
)
from skillage.offline_data import build_offline_transitions
from skillage.providers import TokenLengthProvider
from skillage.runner import SkillFlowRunner
from skillage.types import (
    EvaluationResult,
    OfflineDatasetArtifacts,
    TrainingArtifacts,
    WarmupArtifacts,
)
from skillage.warmup import run_warmup as run_skillflow_warmup


def train(
    warmup_manifest: Path,
    config: TrainConfig,
    output_dir: Path,
    *,
    runner: SkillFlowRunner,
) -> TrainingArtifacts:
    from skillage.training import train_policy

    return train_policy(warmup_manifest, config, output_dir, runner=runner)


def build_offline_dataset(
    warmup_manifest: Path,
    config: OfflineDataConfig,
    output_dir: Path,
    *,
    runner: SkillFlowRunner,
) -> OfflineDatasetArtifacts:
    return build_offline_transitions(
        warmup_manifest, config, output_dir, runner=runner
    )


def train_offline(
    dataset_dir: Path, config: OfflineCQLConfig, output_dir: Path
) -> TrainingArtifacts:
    from skillage.offline_training import train_cql_policy

    return train_cql_policy(dataset_dir, config, output_dir)


def evaluate(
    warmup_manifest: Path,
    checkpoint_path: Path | None,
    config: EvalConfig,
    output_dir: Path,
    *,
    runner: SkillFlowRunner,
) -> EvaluationResult:
    from skillage.evaluation import evaluate_policy

    return evaluate_policy(
        warmup_manifest, checkpoint_path, config, output_dir, runner=runner
    )


def run_warmup(
    config: WarmupConfig,
    output_dir: Path,
    *,
    runner: SkillFlowRunner,
    token_length_provider: TokenLengthProvider | None = None,
) -> WarmupArtifacts:
    return run_skillflow_warmup(
        config,
        output_dir,
        runner=runner,
        token_length_provider=token_length_provider,
    )
