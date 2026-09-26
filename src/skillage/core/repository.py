from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skillage.core.evidence import EvidenceTracker
from skillage.types import LifecycleAction, LifecycleMode, LifecycleSignals, SkillRecord


@dataclass
class RuntimeSkill:
    record: SkillRecord
    index: int
    mode: LifecycleMode = LifecycleMode.ACTIVE
    removed: bool = False
    last_used_step: int = -1
    use_count: int = 0


class SkillRepository:
    def __init__(
        self,
        skills: list[SkillRecord],
        skill_vectors: np.ndarray,
        redundancy_matrix: np.ndarray,
        decayed_retrieval_weight: float = 0.25,
    ) -> None:
        self.skills = {
            record.skill_id: RuntimeSkill(record=record, index=index)
            for index, record in enumerate(skills)
        }
        self.skill_vectors = skill_vectors
        self.redundancy_matrix = redundancy_matrix
        self.decayed_retrieval_weight = decayed_retrieval_weight
        self.initial_size = len(skills)
        self.reference_max_length = max(record.token_length for record in skills)

    def clone(self) -> "SkillRepository":
        clone = object.__new__(SkillRepository)
        clone.skill_vectors = self.skill_vectors
        clone.redundancy_matrix = self.redundancy_matrix
        clone.decayed_retrieval_weight = self.decayed_retrieval_weight
        clone.initial_size = self.initial_size
        clone.reference_max_length = self.reference_max_length
        clone.skills = {
            skill_id: RuntimeSkill(
                record=runtime.record,
                index=runtime.index,
                mode=runtime.mode,
                removed=runtime.removed,
                last_used_step=runtime.last_used_step,
                use_count=runtime.use_count,
            )
            for skill_id, runtime in self.skills.items()
        }
        return clone

    @property
    def active_ids(self) -> list[str]:
        return [
            skill_id
            for skill_id, runtime in self.skills.items()
            if not runtime.removed
        ]

    def retrieve(
        self,
        task_vector: np.ndarray,
        budget: int,
        excluded: set[str] | None = None,
        reference: bool = False,
    ) -> list[str]:
        excluded = excluded or set()
        ranked: list[tuple[float, str]] = []
        for skill_id, runtime in self.skills.items():
            if skill_id in excluded or (runtime.removed and not reference):
                continue
            score = float(task_vector @ self.skill_vectors[runtime.index])
            if not reference:
                if runtime.mode == LifecycleMode.QUARANTINED:
                    continue
                if runtime.mode == LifecycleMode.DECAYED:
                    score *= self.decayed_retrieval_weight
            ranked.append((score, skill_id))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [skill_id for _, skill_id in ranked[:budget]]

    def mark_used(self, skill_ids: list[str], step: int) -> None:
        for skill_id in skill_ids:
            runtime = self.skills[skill_id]
            runtime.last_used_step = step
            runtime.use_count += 1

    def context_cost(self, skill_id: str) -> float:
        return self.skills[skill_id].record.token_length / self.reference_max_length

    def redundancy(self, skill_id: str) -> float:
        runtime = self.skills[skill_id]
        others = [
            other.index
            for other in self.skills.values()
            if not other.removed and other.record.skill_id != skill_id
        ]
        if not others:
            return 0.0
        maximum_cosine = float(self.redundancy_matrix[runtime.index, others].max())
        return (1.0 + maximum_cosine) / 2.0

    def signals(self, evidence: EvidenceTracker) -> dict[str, LifecycleSignals]:
        return {
            skill_id: evidence.signals(
                skill_id,
                redundancy=self.redundancy(skill_id),
                context_cost=self.context_cost(skill_id),
            )
            for skill_id in self.active_ids
        }

    def legal_actions(
        self, skill_id: str, signals: LifecycleSignals, evidence_count: float
    ) -> tuple[LifecycleAction, ...]:
        legal = [
            LifecycleAction.RETAIN,
            LifecycleAction.DECAY,
            LifecycleAction.REFRESH,
            LifecycleAction.PROTECT,
        ]
        runtime = self.skills[skill_id]
        if (
            evidence_count > 0.0
            and signals.aging_evidence > signals.retention_evidence
            and runtime.mode != LifecycleMode.PROTECTED
        ):
            legal.append(LifecycleAction.QUARANTINE)
        return tuple(legal)

    def apply_reversible(
        self,
        actions: dict[str, LifecycleAction],
        evidence: EvidenceTracker,
    ) -> list[str]:
        candidates: list[str] = []
        for skill_id, action in actions.items():
            runtime = self.skills[skill_id]
            if runtime.removed:
                continue
            if action == LifecycleAction.RETAIN:
                runtime.mode = LifecycleMode.ACTIVE
            elif action == LifecycleAction.DECAY:
                runtime.mode = LifecycleMode.DECAYED
            elif action == LifecycleAction.REFRESH:
                evidence.refresh(skill_id)
                runtime.mode = LifecycleMode.ACTIVE
            elif action == LifecycleAction.PROTECT:
                runtime.mode = LifecycleMode.PROTECTED
            elif action == LifecycleAction.QUARANTINE:
                candidates.append(skill_id)
        return candidates

    def remove(self, skill_id: str) -> None:
        runtime = self.skills[skill_id]
        runtime.mode = LifecycleMode.QUARANTINED
        runtime.removed = True

    def restore_protected(self, skill_id: str) -> None:
        self.skills[skill_id].mode = LifecycleMode.PROTECTED

    def compression(self) -> float:
        return 1.0 - len(self.active_ids) / self.initial_size

    def global_summary(
        self,
        retrieved_ids: list[str],
        reference_ids: list[str],
    ) -> np.ndarray:
        repository_ratio = len(self.active_ids) / self.initial_size
        current_tokens = sum(self.skills[item].record.token_length for item in retrieved_ids)
        reference_tokens = sum(
            self.skills[item].record.token_length for item in reference_ids
        )
        token_ratio = min(1.0, current_tokens / max(1, reference_tokens))
        active = self.active_ids
        mean_redundancy = (
            sum(self.redundancy(skill_id) for skill_id in active) / len(active)
            if active
            else 0.0
        )
        return np.array(
            [repository_ratio, token_ratio, mean_redundancy], dtype=np.float32
        )

    def overhead(self, summary: np.ndarray) -> float:
        return float(np.mean(summary))

