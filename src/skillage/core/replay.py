from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from skillage.core.evidence import EvidenceTracker
from skillage.core.repository import SkillRepository
from skillage.errors import BudgetInvariantError
from skillage.types import LifecycleSignals

if TYPE_CHECKING:
    from skillage.world import SkillFlowWorld


@dataclass(frozen=True)
class ReplayResult:
    accepted: tuple[str, ...]
    rejected: tuple[str, ...]
    degradation: float
    remaining_budget: float
    benchmark_task_ids: tuple[str, ...]
    evaluated_task_count: int
    baseline_performance: float | None
    trial_performances: dict[str, float]


class QuarantineReplay:
    def __init__(
        self,
        world: SkillFlowWorld,
        retrieval_budget: int,
        replay_task_budget: int,
    ) -> None:
        self.world = world
        self.retrieval_budget = retrieval_budget
        self.replay_task_budget = replay_task_budget

    def verify(
        self,
        candidates: list[str],
        signals: dict[str, LifecycleSignals],
        repository: SkillRepository,
        evidence: EvidenceTracker,
        remaining_budget: float,
    ) -> ReplayResult:
        if not candidates:
            return ReplayResult((), (), 0.0, remaining_budget, (), 0, None, {})
        ordered = sorted(
            candidates,
            key=lambda skill_id: (signals[skill_id].retention_score, skill_id),
        )
        benchmark = tuple(
            dict.fromkeys(
                task_id
                for skill_id in ordered
                for task_id in evidence.invocation_tasks(skill_id)
            )
        )[: self.replay_task_budget]
        if not benchmark:
            for skill_id in ordered:
                repository.restore_protected(skill_id)
            return ReplayResult((), tuple(ordered), 0.0, remaining_budget, (), 0, None, {})
        baseline = self.world.average_performance(
            benchmark, repository, retrieval_budget=self.retrieval_budget
        )
        accepted: list[str] = []
        rejected: list[str] = []
        degradation = 0.0
        trial_performances: dict[str, float] = {}
        for skill_id in ordered:
            trial = set(accepted) | {skill_id}
            trial_performance = self.world.average_performance(
                benchmark,
                repository,
                excluded=trial,
                retrieval_budget=self.retrieval_budget,
            )
            trial_degradation = max(0.0, baseline - trial_performance)
            trial_performances[skill_id] = trial_performance
            if trial_degradation <= remaining_budget + 1e-12:
                accepted.append(skill_id)
                degradation = trial_degradation
            else:
                rejected.append(skill_id)
        if degradation > remaining_budget + 1e-12:
            raise BudgetInvariantError(
                f"replay degradation {degradation} exceeds budget {remaining_budget}"
            )
        for skill_id in accepted:
            repository.remove(skill_id)
        for skill_id in rejected:
            repository.restore_protected(skill_id)
        updated_budget = max(0.0, remaining_budget - degradation)
        return ReplayResult(
            accepted=tuple(accepted),
            rejected=tuple(rejected),
            degradation=degradation,
            remaining_budget=updated_budget,
            benchmark_task_ids=benchmark,
            evaluated_task_count=len(benchmark),
            baseline_performance=baseline,
            trial_performances=trial_performances,
        )
