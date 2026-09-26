from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import numpy as np

from skillage.types import LifecycleSignals


@dataclass
class SkillHistory:
    exposures: list[float] = field(default_factory=list)
    invocations: list[int] = field(default_factory=list)
    verifier_scores: list[float] = field(default_factory=list)
    contributions: list[float] = field(default_factory=list)
    invoked_task_ids: list[str] = field(default_factory=list)

    def clear(self) -> None:
        self.exposures.clear()
        self.invocations.clear()
        self.verifier_scores.clear()
        self.contributions.clear()
        self.invoked_task_ids.clear()


class EvidenceTracker:
    def __init__(self, skill_ids: list[str], recent_window: int) -> None:
        if recent_window < 1:
            raise ValueError("recent_window must be positive")
        self.recent_window = recent_window
        self.histories = {skill_id: SkillHistory() for skill_id in skill_ids}
        self.global_performance: list[float] = []

    @property
    def baseline(self) -> float:
        return (1.0 + sum(self.global_performance)) / (
            len(self.global_performance) + 2.0
        )

    def update(
        self,
        task_id: str,
        performance: float,
        exposures: dict[str, float],
    ) -> None:
        if not 0.0 <= performance <= 1.0:
            raise ValueError("performance must be in [0, 1]")
        baseline = self.baseline
        for skill_id, history in self.histories.items():
            exposure = float(exposures.get(skill_id, 0.0))
            if not 0.0 <= exposure <= 1.0:
                raise ValueError("exposure must be in [0, 1]")
            invoked = int(exposure > 0.0)
            history.exposures.append(exposure)
            history.invocations.append(invoked)
            history.verifier_scores.append(performance)
            history.contributions.append(exposure * (performance - baseline))
            if invoked:
                history.invoked_task_ids.append(task_id)
        self.global_performance.append(performance)

    def refresh(self, skill_id: str) -> None:
        self.histories[skill_id].clear()

    def invocation_tasks(self, skill_id: str) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.histories[skill_id].invoked_task_ids))

    def evidence_count(self, skill_id: str) -> float:
        return sum(self.histories[skill_id].exposures)

    def signals(
        self,
        skill_id: str,
        redundancy: float,
        context_cost: float,
    ) -> LifecycleSignals:
        history = self.histories[skill_id]
        total_steps = len(history.invocations)
        if total_steps:
            width = min(self.recent_window, total_steps)
            invocation_frequency = sum(history.invocations[-width:]) / width
        else:
            invocation_frequency = 0.0
        invoked_count = sum(history.invocations)
        weighted_performance = sum(
            invoked * score
            for invoked, score in zip(
                history.invocations, history.verifier_scores, strict=True
            )
        )
        invocation_performance = (1.0 + weighted_performance) / (
            2.0 + invoked_count
        )
        accumulated_exposure = sum(history.exposures)
        contribution = float(
            np.clip(
                sum(history.contributions) / (1.0 + accumulated_exposure),
                -1.0,
                1.0,
            )
        )
        scarcity = 1.0 / sqrt(1.0 + accumulated_exposure)
        prospective_utility = (max(contribution, 0.0) + scarcity) / 2.0
        retention_evidence = (
            prospective_utility
            + invocation_frequency
            + invocation_performance
        ) / 3.0
        aging_evidence = (
            redundancy + context_cost + max(-contribution, 0.0)
        ) / 3.0
        retention_score = (1.0 + retention_evidence - aging_evidence) / 2.0
        return LifecycleSignals(
            invocation_frequency=float(invocation_frequency),
            invocation_performance=float(invocation_performance),
            contribution=contribution,
            redundancy=float(redundancy),
            context_cost=float(context_cost),
            scarcity=float(scarcity),
            retention_evidence=float(retention_evidence),
            aging_evidence=float(aging_evidence),
            retention_score=float(retention_score),
        )

