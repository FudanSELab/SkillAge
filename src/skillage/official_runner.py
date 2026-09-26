from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from skillage.errors import ConfigurationError
from skillage.runner import SkillFlowRunner
from skillage.types import SkillRecord, TaskExecution, TaskRef


class SkillFlowOfficialRunner:
    """Run a task through the checked-out official SkillFlow implementation.

    The official repository remains an external, read-only source.  This class
    only writes a request envelope and invokes ``skillflow_official_bridge.py``
    in a subprocess, so Harbor, Docker, and the official Python dependencies do
    not become dependencies of the SkillAge test environment.
    """

    def __init__(
        self,
        official_root: Path,
        *,
        api_key: str,
        model: str = "qwen3.7-plus",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        version: str | None = None,
        expected_commit: str = "7b49ff5a7e26cd7706e959bfa0dba4746d18440d",
        python_executable: str | Path | None = None,
        bridge_script: str | Path | None = None,
        timeout_seconds: float = 3600.0,
    ) -> None:
        root = Path(official_root).expanduser().resolve()
        if not root.is_dir():
            raise ConfigurationError(f"official SkillFlow source does not exist: {root}")
        if not api_key.strip():
            raise ConfigurationError("Qwen API key must not be empty")
        if timeout_seconds <= 0:
            raise ConfigurationError("runner timeout must be positive")
        self.official_root = root
        actual_commit = _checkout_commit(root)
        if actual_commit is not None and expected_commit and actual_commit != expected_commit:
            raise ConfigurationError(
                "official SkillFlow checkout does not match the pinned source: "
                f"expected {expected_commit}, found {actual_commit}"
            )
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.version = version or f"skillflow-official-{expected_commit[:7]}"
        self.python_executable = str(python_executable or sys.executable)
        self.bridge_script = Path(bridge_script or Path(__file__).parents[2] / "scripts" / "skillflow_official_bridge.py").resolve()
        if not self.bridge_script.is_file():
            raise ConfigurationError(f"SkillFlow bridge script does not exist: {self.bridge_script}")
        self.timeout_seconds = timeout_seconds

    @property
    def runner_version(self) -> str:
        return self.version

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
    ) -> TaskExecution:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        request_path = run_dir / "request.json"
        result_path = run_dir / "result.json"
        request = {
            "schema_version": 1,
            "official_root": str(self.official_root),
            "task_root": str(Path(task_root).resolve()),
            "task": task.model_dump(mode="json"),
            "skills": [skill.model_dump(mode="json") for skill in skills],
            "excluded_skill_ids": sorted(excluded_skill_ids),
            "allow_skill_updates": allow_skill_updates,
            "model": self.model,
            "base_url": self.base_url,
            "seed": seed,
        }
        request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        environment = os.environ.copy()
        environment["DASHSCOPE_API_KEY"] = self.api_key
        environment["SKILLAGE_OFFICIAL_ROOT"] = str(self.official_root)
        try:
            completed = subprocess.run(
                [self.python_executable, str(self.bridge_script), "--request", str(request_path), "--result", str(result_path)],
                cwd=run_dir,
                env=environment,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        finally:
            _scrub_tree(run_dir, self.api_key, str(Path(task_root).resolve()))
        if completed.returncode != 0:
            raise ConfigurationError(
                f"official SkillFlow runner failed for {task.task_id} with exit code {completed.returncode}: "
                f"{_redact(completed.stderr, self.api_key)}"
            )
        if not result_path.is_file():
            raise ConfigurationError("official SkillFlow bridge did not produce result.json")
        try:
            execution = TaskExecution.model_validate_json(result_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ConfigurationError(f"invalid official SkillFlow result: {exc}") from exc
        if execution.task_id != task.task_id:
            raise ConfigurationError("official SkillFlow runner returned an execution for the wrong task")
        if "performance" not in execution.verifier_output:
            raise ConfigurationError("official SkillFlow result must include verifier_output.performance")
        verifier_performance = float(execution.verifier_output["performance"])
        if abs(execution.performance - verifier_performance) > 1e-12:
            raise ConfigurationError("TaskExecution.performance must equal verifier_output.performance")
        return execution


def _redact(value: str, api_key: str) -> str:
    return value.replace(api_key, "[REDACTED]")


def _checkout_commit(root: Path) -> str | None:
    """Read a cloned checkout's HEAD without invoking git."""
    git_dir = root / ".git"
    if not git_dir.is_dir():
        return None
    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head or None
    ref = head[6:]
    ref_path = git_dir / ref
    if ref_path.is_file():
        return ref_path.read_text(encoding="utf-8").strip() or None
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and not line.startswith("^"):
                commit, name = line.split(" ", 1)
                if name == ref:
                    return commit
    return None


def _scrub_tree(root: Path, api_key: str, task_root: str) -> None:
    """Remove credentials and machine-specific source paths from bridge output."""
    replacements = ((api_key, "[REDACTED]"), (task_root, "__external__/SkillFlow-Task"))
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
            text = data.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for old, new in replacements:
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")

