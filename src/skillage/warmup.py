from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from math import ceil
from pathlib import Path, PurePosixPath
from typing import Iterable

from skillage.artifacts import file_sha256
from skillage.config import WarmupConfig
from skillage.errors import BundleValidationError, ConfigurationError
from skillage.providers import TokenLengthProvider, WhitespaceTokenLengthProvider
from skillage.runner import SkillFlowRunner
from skillage.types import (
    SkillPatch,
    SkillRecord,
    TaskRef,
    WarmupArtifacts,
    WarmupManifest,
)


MANIFEST_FILE = "manifest.json"
SELECTION_FILE = "task_selection.json"
CHECKSUMS_FILE = "checksums.json"
SKILLS_FILE = "skills.jsonl"

FAMILY_ALIASES = {
    "econ-detrending-correlation": "industry-correlation-analysis",
    "harbor-gdpval-20": "financial-statement-rolling",
    "sec-financial-report": "sec-13f-financial-analysis",
    "harbor-gdpval-21": "supply-chain-replenishment",
    "harbor-gdpval-36": "production-capacity-planning",
    "merge-20-21": "inventory-finance-integration",
    "merge-35-37": "dmaic-quality-analysis",
    "merge-36-41": "operational-recovery-planning",
    "harbor-gdpval-42": "healthcare-cost-benefit-analysis",
    "lab-unit-harmonization": "medical-data-standardization",
    "harbor-gdpval-3": "distribution-center-auditing",
    "harbor-gdpval-33": "compensation-scenario-modeling",
    "invoice-fraud-detection": "document-fraud-detection",
    "exceltable-in-ppt": "embedded-data-repair",
    "jpg-ocr-stat": "ocr-data-extraction",
    "merge-court-offer": "hwpx-document-automation",
    "merge-pdf-xlsx": "cross-format-data-reconciliation",
    "merge-weight-reserves": "weighted-risk-assessment",
    "pptx-reference-formatting": "ppt-formatting-optimization",
    "sales-pivot-analysis": "sales-pivot-analysis",
}


def canonical_family(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return FAMILY_ALIASES.get(normalized, normalized)


def _ranking_entries(family_dir: Path) -> list[str]:
    ranking_path = family_dir / "ALL_TASK_DIFFICULTY_RANKING.json"
    if not ranking_path.is_file():
        raise ConfigurationError(f"missing SkillFlow ranking: {ranking_path}")
    payload = json.loads(ranking_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("tasks", "ranking", "order", "task_names"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ConfigurationError(f"unsupported ranking format: {ranking_path}")
    entries: list[str] = []
    for item in payload:
        if isinstance(item, str):
            entries.append(item)
        elif isinstance(item, dict):
            name = item.get("task_name") or item.get("name") or item.get("task")
            if name:
                entries.append(str(name))
    if not entries:
        raise ConfigurationError(f"empty SkillFlow ranking: {ranking_path}")
    return entries


def _task_ref(task_root: Path, family_dir: Path, task_name: str, order: int) -> TaskRef:
    task_dir = family_dir / task_name
    instruction = task_dir / "instruction.md"
    if not instruction.is_file():
        raise ConfigurationError(f"missing task instruction: {instruction}")
    return TaskRef(
        task_id=f"{canonical_family(family_dir.name)}::{task_name}",
        family_id=canonical_family(family_dir.name),
        order=order,
        relative_path=task_dir.relative_to(task_root).as_posix(),
        instruction_sha256=file_sha256(instruction),
    )


def select_tasks(config: WarmupConfig) -> tuple[list[TaskRef], list[TaskRef], list[TaskRef]]:
    root = config.task_root
    if not root.is_dir():
        raise ConfigurationError(f"SkillFlow task root does not exist: {root}")
    selected_families = {canonical_family(item) for item in config.family_ids}
    warmup: list[TaskRef] = []
    main: list[TaskRef] = []
    test: list[TaskRef] = []
    for family_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        family_id = canonical_family(family_dir.name)
        if selected_families and family_id not in selected_families:
            continue
        ranking = _ranking_entries(family_dir)
        if config.max_tasks_per_family is not None:
            ranking = ranking[: config.max_tasks_per_family]
        split_at = max(
            1,
            min(
                len(ranking) - config.split.min_test_tasks,
                ceil(len(ranking) * config.split.train_fraction),
            ),
        )
        train_names = ranking[:split_at]
        test_names = ranking[split_at:]
        if len(train_names) < 2:
            raise ConfigurationError(
                f"family {family_id} needs at least two training tasks for warm-up/main"
            )
        warmup_count = max(
            config.min_tasks_per_family,
            ceil(len(train_names) * config.warmup_fraction),
        )
        warmup_count = min(warmup_count, len(train_names) - 1)
        refs = [
            _task_ref(root, family_dir, name, order)
            for order, name in enumerate(ranking)
        ]
        warmup.extend(refs[:warmup_count])
        main.extend(refs[warmup_count:split_at])
        test.extend(refs[split_at:])
    if not warmup or not main:
        raise ConfigurationError("warm-up selection requires warm-up and main tasks")
    return warmup, main, test


def _source_digest(task_root: Path, task_refs: Iterable[TaskRef]) -> str:
    digest = hashlib.sha256()
    paths: set[Path] = set()
    family_dirs: set[Path] = set()
    for ref in task_refs:
        task_dir = task_root / Path(ref.relative_path)
        family_dirs.add(task_dir.parent)
        paths.update(path for path in task_dir.rglob("*") if path.is_file())
    for family_dir in family_dirs:
        ranking = family_dir / "ALL_TASK_DIFFICULTY_RANKING.json"
        if ranking.is_file():
            paths.add(ranking)
    for path in sorted(paths, key=lambda item: item.relative_to(task_root).as_posix()):
        relative = path.relative_to(task_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_skills(root: Path, skills: dict[str, SkillRecord]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    lines = [skills[skill_id].model_dump_json() for skill_id in sorted(skills)]
    (root / SKILLS_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    for skill_id in sorted(skills):
        record = skills[skill_id]
        directory = root / hashlib.sha256(skill_id.encode("utf-8")).hexdigest()[:16]
        directory.mkdir()
        (directory / "SKILL.md").write_text(record.content, encoding="utf-8")
        for relative_name, content in sorted(record.support_files.items()):
            relative = PurePosixPath(relative_name.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ConfigurationError(
                    f"unsafe skill support path for {skill_id}: {relative_name}"
                )
            support_path = directory.joinpath(*relative.parts)
            support_path.parent.mkdir(parents=True, exist_ok=True)
            support_path.write_text(content, encoding="utf-8")
        (directory / "metadata.json").write_text(
            json.dumps(
                {"skill_id": skill_id, "support_files": sorted(record.support_files)},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


def _apply_patch(
    patch: SkillPatch,
    skills: dict[str, SkillRecord],
    token_provider: TokenLengthProvider,
    patch_path: Path,
) -> None:
    patch_path.write_text(patch.model_dump_json(indent=2) + "\n", encoding="utf-8")
    skills[patch.skill_id] = SkillRecord(
        skill_id=patch.skill_id,
        source_model="warmup",
        family_id=patch.family_id,
        name=patch.name,
        content=patch.content,
        token_length=token_provider.count(patch.content),
        support_files=dict(sorted(patch.support_files.items())),
        source_path=f"skill_patches/{patch_path.name}",
    )


def _verified_performance(performance: float, verifier_output: dict[str, object]) -> float:
    if "performance" not in verifier_output:
        raise ConfigurationError("runner result is missing verifier_output.performance")
    verifier_performance = float(verifier_output["performance"])
    if not 0.0 <= verifier_performance <= 1.0:
        raise ConfigurationError("verifier performance must be in [0, 1]")
    if abs(performance - verifier_performance) > 1e-12:
        raise ConfigurationError(
            "runner performance must equal verifier_output.performance"
        )
    return verifier_performance


def _artifact_files(output_dir: Path) -> dict[str, str]:
    excluded = {MANIFEST_FILE, CHECKSUMS_FILE}
    return {
        path.relative_to(output_dir).as_posix(): file_sha256(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name not in excluded
    }


def run_warmup(
    config: WarmupConfig,
    output_dir: Path,
    *,
    runner: SkillFlowRunner,
    token_length_provider: TokenLengthProvider | None = None,
) -> WarmupArtifacts:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ConfigurationError(f"warm-up output already exists: {output_dir}")
    warmup_tasks, main_tasks, test_tasks = select_tasks(config)
    output_dir.mkdir(parents=True)
    for name in (
        "trajectories",
        "verifier_results",
        "skill_patches",
        "skill_snapshots",
        "final_skill_repository",
        "runs",
    ):
        (output_dir / name).mkdir()
    selection = {
        "warmup": [task.model_dump(mode="json") for task in warmup_tasks],
        "main": [task.model_dump(mode="json") for task in main_tasks],
        "test": [task.model_dump(mode="json") for task in test_tasks],
    }
    (output_dir / SELECTION_FILE).write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    skills: dict[str, SkillRecord] = {}
    token_provider = token_length_provider or WhitespaceTokenLengthProvider()
    for index, task in enumerate(warmup_tasks):
        run_dir = output_dir / "runs" / f"{index:04d}"
        run_dir.mkdir()
        execution = runner.execute(
            task_root=config.task_root,
            task=task,
            skills=tuple(skills.values()),
            excluded_skill_ids=frozenset(),
            allow_skill_updates=True,
            run_dir=run_dir,
            seed=config.seed,
        )
        if execution.task_id != task.task_id:
            raise ConfigurationError("runner returned an execution for the wrong task")
        _verified_performance(execution.performance, execution.verifier_output)
        (output_dir / "trajectories" / f"{index:04d}.json").write_text(
            json.dumps(execution.trajectory, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "verifier_results" / f"{index:04d}.json").write_text(
            json.dumps(execution.verifier_output, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for patch_index, patch in enumerate(execution.skill_patches):
            patch_path = output_dir / "skill_patches" / f"{index:04d}-{patch_index:03d}.json"
            _apply_patch(patch, skills, token_provider, patch_path)
        snapshot = output_dir / "skill_snapshots" / f"{index:04d}"
        _write_skills(snapshot, skills)
    if not skills:
        raise ConfigurationError("warm-up completed without producing any skills")
    final_repository = output_dir / "final_skill_repository"
    _write_skills(final_repository, skills)
    task_refs = tuple(warmup_tasks + main_tasks + test_tasks)
    files = _artifact_files(output_dir)
    manifest = WarmupManifest(
        created_at_utc=datetime.now(UTC).isoformat(),
        source_task_root=str(config.task_root.resolve()),
        source_tree_sha256=_source_digest(config.task_root, task_refs),
        dataset_revision=config.dataset_revision,
        model=config.model,
        base_url=config.base_url,
        prompt_version=config.prompt_version,
        seed=config.seed,
        build_config={
            **config.model_dump(mode="json"),
            "runner_version": runner.runner_version,
            "token_length_provider": token_provider.name,
        },
        warmup_task_ids=tuple(task.task_id for task in warmup_tasks),
        main_task_ids=tuple(task.task_id for task in main_tasks),
        test_task_ids=tuple(task.task_id for task in test_tasks),
        task_refs=task_refs,
        final_skill_ids=tuple(sorted(skills)),
        files=files,
    )
    manifest_path = output_dir / MANIFEST_FILE
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    checksums = {**files, MANIFEST_FILE: file_sha256(manifest_path)}
    (output_dir / CHECKSUMS_FILE).write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return WarmupArtifacts(
        output_dir=output_dir,
        manifest_path=manifest_path,
        final_skill_repository=final_repository,
        warmup_task_count=len(warmup_tasks),
        main_task_count=len(main_tasks),
    )


def read_warmup_manifest(manifest_path: Path, *, verify_source: bool = True) -> WarmupManifest:
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise BundleValidationError(f"missing warm-up manifest: {manifest_path}")
    root = manifest_path.parent
    manifest = WarmupManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.schema_version != 1:
        raise BundleValidationError(f"unsupported warm-up schema: {manifest.schema_version}")
    groups = [set(manifest.warmup_task_ids), set(manifest.main_task_ids), set(manifest.test_task_ids)]
    if not groups[0] or not groups[1] or any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise BundleValidationError("warm-up/main/test task partitions are invalid")
    known_ids = {ref.task_id for ref in manifest.task_refs}
    if set().union(*groups) != known_ids:
        raise BundleValidationError("warm-up task references do not match partitions")
    for name, expected in manifest.files.items():
        path = root / Path(name)
        if not path.is_file() or file_sha256(path) != expected:
            raise BundleValidationError(f"warm-up checksum mismatch: {name}")
    checksums_path = root / CHECKSUMS_FILE
    if not checksums_path.is_file():
        raise BundleValidationError("missing warm-up checksums.json")
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    expected_manifest = checksums.get(MANIFEST_FILE)
    if expected_manifest != file_sha256(manifest_path):
        raise BundleValidationError("warm-up manifest checksum mismatch")
    if {name: checksums.get(name) for name in manifest.files} != manifest.files:
        raise BundleValidationError("warm-up checksums.json does not match manifest")
    skills_path = root / "final_skill_repository" / SKILLS_FILE
    if not skills_path.is_file():
        raise BundleValidationError("warm-up final skill repository is missing")
    if verify_source:
        source_root = Path(manifest.source_task_root)
        if not source_root.is_dir():
            raise BundleValidationError(f"SkillFlow source root is unavailable: {source_root}")
        if _source_digest(source_root, manifest.task_refs) != manifest.source_tree_sha256:
            raise BundleValidationError("SkillFlow source checksum mismatch")
    return manifest
