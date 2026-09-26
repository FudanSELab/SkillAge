from __future__ import annotations

import random
from typing import Protocol

from skillage.environment import DecisionState
from skillage.types import LifecycleAction


class ActionPolicy(Protocol):
    def actions(self, state: DecisionState) -> dict[str, LifecycleAction]: ...


class NoPrunePolicy:
    def actions(self, state: DecisionState) -> dict[str, LifecycleAction]:
        return {skill_id: LifecycleAction.RETAIN for skill_id in state.skill_ids}


class RandomPolicy:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def actions(self, state: DecisionState) -> dict[str, LifecycleAction]:
        action_values = tuple(LifecycleAction)
        return {
            skill_id: self.rng.choice(
                [action_values[index] for index, legal in enumerate(state.legal_mask[row]) if legal]
            )
            for row, skill_id in enumerate(state.skill_ids)
        }


class EvictionPolicy:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def actions(self, state: DecisionState) -> dict[str, LifecycleAction]:
        actions = {skill_id: LifecycleAction.RETAIN for skill_id in state.skill_ids}
        eligible = [
            skill_id
            for row, skill_id in enumerate(state.skill_ids)
            if state.legal_mask[row, tuple(LifecycleAction).index(LifecycleAction.QUARANTINE)]
        ]
        if not eligible:
            return actions
        if self.mode == "lru":
            target = min(eligible, key=lambda item: state.last_used_steps[item])
        else:
            target = min(eligible, key=lambda item: state.use_counts[item])
        actions[target] = LifecycleAction.QUARANTINE
        return actions


class RetentionPolicy:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def actions(self, state: DecisionState) -> dict[str, LifecycleAction]:
        actions: dict[str, LifecycleAction] = {}
        quarantine_index = tuple(LifecycleAction).index(LifecycleAction.QUARANTINE)
        for row, skill_id in enumerate(state.skill_ids):
            if (
                state.signals[skill_id].retention_score < self.threshold
                and state.legal_mask[row, quarantine_index]
            ):
                actions[skill_id] = LifecycleAction.QUARANTINE
            else:
                actions[skill_id] = LifecycleAction.RETAIN
        return actions
