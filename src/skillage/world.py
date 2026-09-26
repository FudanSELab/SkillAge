from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from skillage.core.repository import SkillRepository
from skillage.errors import BundleValidationError, ConfigurationError
from skillage.providers import EmbeddingProvider, TfidfEmbeddingProvider
from skillage.runner import SkillFlowRunner
from skillage.types import SkillRecord, TaskRef, WarmupManifest
from skillage.warmup import SKILLS_FILE, read_warmup_manifest


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    performance: float
    retrieved_ids: tuple[str, ...]
    reference_ids: tuple[str, ...]
    invoked_ids: tuple[str, ...]
    exposures: dict[str, float]
    verifier_output: dict[str, object]


class SkillFlowWorld:
    def __init__(
        self,
        warmup_manifest: Path,
        *,
        runner: SkillFlowRunner,
        run_root: Path,
        embedding_provider: EmbeddingProvider | None = None,
        decayed_retrieval_weight: float = 0.25,
        verify_source: bool = True,
    ) -> None:
        self.manifest_path = Path(warmup_manifest)
        self.manifest: WarmupManifest = read_warmup_manifest(
            self.manifest_path, verify_source=verify_source
        )
        self.task_root = Path(self.manifest.source_task_root)
        self.runner = runner
        self.run_root = Path(run_root)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.tasks = list(self.manifest.task_refs)
        self.task_by_id = {task.task_id: task for task in self.tasks}
        skills_path = self.manifest_path.parent / "final_skill_repository" / SKILLS_FILE
        self.skills = [
            SkillRecord.model_validate_json(line)
            for line in skills_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if not self.skills:
            raise BundleValidationError("warm-up final skill repository is empty")
        provider = embedding_provider or TfidfEmbeddingProvider(8192)
        instructions = [self._instruction(task) for task in self.tasks]
        matrix = np.asarray(
            provider.encode(instructions + [skill.content for skill in self.skills]),
            dtype=np.float32,
        )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
        self.task_vectors = matrix[: len(self.tasks)]
        self.skill_vectors = matrix[len(self.tasks) :]
        self.redundancy = np.clip(
            self.skill_vectors @ self.skill_vectors.T, 0.0, 1.0
        ).astype(np.float32)
        np.fill_diagonal(self.redundancy, 0.0)
        self.task_index = {task.task_id: index for index, task in enumerate(self.tasks)}
        self.decayed_retrieval_weight = decayed_retrieval_weight

    def _instruction(self, task: TaskRef) -> str:
        path = self.task_root / Path(task.relative_path) / "instruction.md"
        if not path.is_file():
            raise BundleValidationError(f"missing SkillFlow task instruction: {path}")
        return path.read_text(encoding="utf-8")

    def new_repository(self) -> SkillRepository:
        return SkillRepository(
            self.skills,
            self.skill_vectors,
            self.redundancy,
            decayed_retrieval_weight=self.decayed_retrieval_weight,
        )

    def task_stream(self, partition: str) -> list[TaskRef]:
        ids = {
            "warmup": self.manifest.warmup_task_ids,
            "main": self.manifest.main_task_ids,
            "test": self.manifest.test_task_ids,
        }.get(partition)
        if ids is None:
            raise ConfigurationError(f"unknown SkillFlow partition: {partition}")
        return [self.task_by_id[task_id] for task_id in ids]

    def execute(
        self,
        task: TaskRef,
        repository: SkillRepository,
        retrieval_budget: int,
        excluded: set[str] | None = None,
        *,
        allow_skill_updates: bool = False,
    ) -> TaskOutcome:
        if task.task_id not in self.task_index:
            raise ConfigurationError(f"unknown SkillFlow task: {task.task_id}")
        excluded = excluded or set()
        task_vector = self.task_vectors[self.task_index[task.task_id]]
        retrieved = repository.retrieve(task_vector, retrieval_budget, excluded=excluded)
        reference = repository.retrieve(task_vector, retrieval_budget, reference=True)
        selected = [repository.skills[skill_id].record for skill_id in retrieved]
        task_runs = self.run_root / task.task_id.replace("::", "--")
        task_runs.mkdir(parents=True, exist_ok=True)
        run_dir = task_runs / f"run-{len(tuple(task_runs.iterdir())):06d}"
        run_dir.mkdir()
        execution = self.runner.execute(
            task_root=self.task_root,
            task=task,
            skills=selected,
            excluded_skill_ids=frozenset(excluded),
            allow_skill_updates=allow_skill_updates,
            run_dir=run_dir,
            seed=self.manifest.seed,
        )
        if execution.task_id != task.task_id:
            raise ConfigurationError("runner returned an execution for the wrong task")
        if "performance" not in execution.verifier_output:
            raise ConfigurationError("runner result is missing verifier_output.performance")
        verifier_performance = float(execution.verifier_output["performance"])
        if abs(execution.performance - verifier_performance) > 1e-12:
            raise ConfigurationError(
                "runner performance must equal verifier_output.performance"
            )
        if execution.skill_patches and not allow_skill_updates:
            raise ConfigurationError("runner produced skill patches during a frozen execution")
        invoked = execution.invoked_skill_ids or tuple(retrieved)
        unknown = set(invoked) - set(retrieved)
        if unknown:
            raise ConfigurationError(
                f"runner invoked skills outside the supplied repository: {sorted(unknown)}"
            )
        exposure = 1.0 / len(invoked) if invoked else 0.0
        return TaskOutcome(
            task_id=task.task_id,
            performance=execution.performance,
            retrieved_ids=tuple(retrieved),
            reference_ids=tuple(reference),
            invoked_ids=tuple(invoked),
            exposures={skill_id: exposure for skill_id in invoked},
            verifier_output=dict(execution.verifier_output),
        )

    def repository_summary(
        self,
        task: TaskRef,
        repository: SkillRepository,
        retrieval_budget: int,
    ) -> np.ndarray:
        task_vector = self.task_vectors[self.task_index[task.task_id]]
        retrieved = repository.retrieve(task_vector, retrieval_budget)
        reference = repository.retrieve(task_vector, retrieval_budget, reference=True)
        return repository.global_summary(retrieved, reference)

    def average_performance(
        self,
        tasks: Sequence[str | TaskRef],
        repository: SkillRepository,
        retrieval_budget: int,
        excluded: set[str] | None = None,
    ) -> float:
        resolved = [self.task_by_id[item] if isinstance(item, str) else item for item in tasks]
        if not resolved:
            return 1.0
        scores = [
            self.execute(
                task,
                repository,
                retrieval_budget,
                excluded=excluded,
                allow_skill_updates=False,
            ).performance
            for task in resolved
        ]
        return float(np.mean(scores))
