from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from skillage.config import EvalConfig, OfflineDataConfig, TrainConfig
from skillage.core.evidence import EvidenceTracker
from skillage.core.replay import QuarantineReplay, ReplayResult
from skillage.core.repository import SkillRepository
from skillage.errors import BudgetInvariantError
from skillage.types import (
    LifecycleAction,
    LifecycleMode,
    LifecycleSignals,
    StepDecision,
    TaskRef,
)

if TYPE_CHECKING:
    from skillage.world import SkillFlowWorld, TaskOutcome


ACTIONS = tuple(LifecycleAction)
MODES = tuple(LifecycleMode)
FEATURE_NAMES = (
    "invocation_frequency",
    "invocation_performance",
    "contribution",
    "redundancy",
    "context_cost",
    "scarcity",
    "retention_score",
    "mode_active",
    "mode_decayed",
    "mode_quarantined",
    "mode_protected",
    "repository_ratio",
    "token_ratio",
    "mean_redundancy",
    "remaining_budget_ratio",
)


@dataclass(frozen=True)
class DecisionState:
    task: TaskRef
    task_id: str
    skill_ids: tuple[str, ...]
    features: np.ndarray
    legal_mask: np.ndarray
    global_summary: np.ndarray
    signals: dict[str, LifecycleSignals]
    outcome: TaskOutcome
    last_used_steps: dict[str, int]
    use_counts: dict[str, int]


class LifecycleEnvironment:
    def __init__(
        self,
        world: SkillFlowWorld,
        recent_window: int,
        degradation_budget: float,
        retrieval_budget: int,
        replay_task_budget: int,
    ) -> None:
        self.world = world
        self.recent_window = recent_window
        self.initial_budget = degradation_budget
        self.retrieval_budget = retrieval_budget
        self.replay = QuarantineReplay(
            world,
            retrieval_budget,
            replay_task_budget=replay_task_budget,
        )
        self.reset()

    @classmethod
    def from_config(
        cls, world: SkillFlowWorld, config: TrainConfig | EvalConfig | OfflineDataConfig
    ) -> "LifecycleEnvironment":
        return cls(
            world=world,
            recent_window=config.recent_window,
            degradation_budget=config.degradation_budget,
            retrieval_budget=config.retrieval_budget,
            replay_task_budget=config.replay_task_budget,
        )

    def reset(self) -> None:
        self.repository: SkillRepository = self.world.new_repository()
        self.evidence = EvidenceTracker(
            [skill.skill_id for skill in self.world.skills], self.recent_window
        )
        self.remaining_budget = self.initial_budget
        self.cumulative_degradation = 0.0
        self.step_index = 0

    def observe(self, task: TaskRef) -> DecisionState:
        outcome = self.world.execute(
            task, self.repository, retrieval_budget=self.retrieval_budget
        )
        self.repository.mark_used(list(outcome.invoked_ids), self.step_index)
        self.evidence.update(
            outcome.task_id, outcome.performance, outcome.exposures
        )
        signals = self.repository.signals(self.evidence)
        global_summary = self.repository.global_summary(
            list(outcome.retrieved_ids), list(outcome.reference_ids)
        )
        skill_ids = tuple(self.repository.active_ids)
        if skill_ids:
            features = np.stack(
                [
                    self._feature_vector(skill_id, signals[skill_id], global_summary)
                    for skill_id in skill_ids
                ],
                axis=0,
            ).astype(np.float32)
        else:
            features = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        legal_mask = np.zeros((len(skill_ids), len(ACTIONS)), dtype=np.bool_)
        for row, skill_id in enumerate(skill_ids):
            legal = self.repository.legal_actions(
                skill_id,
                signals[skill_id],
                self.evidence.evidence_count(skill_id),
            )
            for action in legal:
                legal_mask[row, ACTIONS.index(action)] = True
        return DecisionState(
            task=task,
            task_id=outcome.task_id,
            skill_ids=skill_ids,
            features=features,
            legal_mask=legal_mask,
            global_summary=global_summary,
            signals=signals,
            outcome=outcome,
            last_used_steps={
                skill_id: self.repository.skills[skill_id].last_used_step
                for skill_id in skill_ids
            },
            use_counts={
                skill_id: self.repository.skills[skill_id].use_count
                for skill_id in skill_ids
            },
        )

    def _feature_vector(
        self,
        skill_id: str,
        signals: LifecycleSignals,
        global_summary: np.ndarray,
    ) -> np.ndarray:
        mode = self.repository.skills[skill_id].mode
        mode_one_hot = np.array([float(mode == item) for item in MODES])
        beta = self.remaining_budget / self.initial_budget
        return np.concatenate(
            [
                np.array(
                    [
                        signals.invocation_frequency,
                        signals.invocation_performance,
                        signals.contribution,
                        signals.redundancy,
                        signals.context_cost,
                        signals.scarcity,
                        signals.retention_score,
                    ]
                ),
                mode_one_hot,
                global_summary,
                np.array([beta]),
            ]
        )

    def apply(
        self, state: DecisionState, actions: dict[str, LifecycleAction]
    ) -> tuple[StepDecision, float, ReplayResult]:
        if set(actions) != set(state.skill_ids):
            raise ValueError("actions must cover every active skill exactly once")
        for row, skill_id in enumerate(state.skill_ids):
            action_index = ACTIONS.index(actions[skill_id])
            if not state.legal_mask[row, action_index]:
                raise ValueError(
                    f"illegal action {actions[skill_id]} for skill {skill_id}"
                )
        candidates = self.repository.apply_reversible(actions, self.evidence)
        replay_result = self.replay.verify(
            candidates,
            state.signals,
            self.repository,
            self.evidence,
            self.remaining_budget,
        )
        self.remaining_budget = replay_result.remaining_budget
        self.cumulative_degradation += replay_result.degradation
        if self.cumulative_degradation > self.initial_budget + 1e-12:
            raise BudgetInvariantError("cumulative degradation exceeded initial budget")
        post_summary = self.world.repository_summary(
            state.task, self.repository, self.retrieval_budget
        )
        overhead = self.repository.overhead(post_summary)
        reward = -overhead
        decision = StepDecision(
            task_id=state.task_id,
            actions=actions,
            removed_skill_ids=replay_result.accepted,
            degradation=replay_result.degradation,
            overhead=overhead,
            performance=state.outcome.performance,
            remaining_budget=self.remaining_budget,
        )
        self.step_index += 1
        return decision, reward, replay_result
