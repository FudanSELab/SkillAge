from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from skillage.types import SkillRecord, TaskExecution, TaskRef


class SkillFlowRunner(Protocol):
    @property
    def runner_version(self) -> str: ...

    def execute(
        self,
        *,
        task_root: Path,
        task: TaskRef,
        skills: Sequence[SkillRecord],
        excluded_skill_ids: frozenset[str],
        allow_skill_updates: bool,
        run_dir: Path,
        seed: int,
    ) -> TaskExecution: ...


# Imported here for the public runner module while keeping the official bridge
# implementation separate from the generic command runner.
from skillage.official_runner import SkillFlowOfficialRunner  # noqa: E402,F401
