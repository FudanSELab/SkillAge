from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skillage.runner import SkillFlowOfficialRunner
from skillage.config import SplitConfig, WarmupConfig
from skillage.warmup import run_warmup
from skillage.artifacts import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the frozen fixture from an official SkillFlow subset."
    )
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--family", default="sales-pivot-analysis")
    parser.add_argument("--task-count", type=int, default=2)
    parser.add_argument("--warmup-fraction", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--python-executable", default=None)
    parser.add_argument("--runner-version", default="skillflow-official-7b49ff5")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task_count < 2:
        raise SystemExit("--task-count must be at least 2")
    if not args.api_key.strip():
        raise SystemExit("--api-key must not be empty")
    output = args.output.resolve()
    task_root = args.task_root.resolve()
    if output in {Path(output.anchor), ROOT, task_root} or output == output.parent:
        raise SystemExit("refusing to replace a broad or source directory")
    if output.exists():
        if not args.replace:
            raise SystemExit(f"output exists; pass --replace: {output}")
    config = WarmupConfig(
        task_root=task_root,
        dataset_revision=args.dataset_revision,
        family_ids=(args.family,),
        warmup_fraction=args.warmup_fraction,
        max_tasks_per_family=args.task_count,
        split=SplitConfig(
            train_fraction=0.999999,
            min_test_tasks=0,
        ),
    )
    runner = SkillFlowOfficialRunner(
        args.official_root,
        api_key=args.api_key,
        model=config.model,
        base_url=config.base_url,
        version=args.runner_version,
        python_executable=args.python_executable,
        timeout_seconds=args.timeout,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="skillage-warmup-fixture-", dir=output.parent
    ) as temporary:
        staged = Path(temporary) / "skillflow_warmup"
        artifacts = run_warmup(config, staged / "warmup", runner=runner)
        manifest_path = artifacts.manifest_path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source_task_root"] = "__external__/SkillFlow-Task"
        manifest["build_config"]["task_root"] = "__external__/SkillFlow-Task"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checksums_path = manifest_path.parent / "checksums.json"
        checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
        checksums["manifest.json"] = file_sha256(manifest_path)
        checksums_path.write_text(
            json.dumps(checksums, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staged / "source_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "dataset_revision": args.dataset_revision,
                    "family": args.family,
                    "task_count": args.task_count,
                    "warmup_fraction": args.warmup_fraction,
                    "warmup_manifest": "warmup/manifest.json",
                    "source_tree_sha256": manifest["source_tree_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if output.exists():
            shutil.rmtree(output)
        shutil.copytree(staged, output)
    print(output / "warmup" / "manifest.json")


if __name__ == "__main__":
    main()
