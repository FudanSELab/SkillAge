# SkillAge

## Required SkillFlow Inputs

Prepare an official SkillFlow task root locally. The repository does not download or redistribute official task assets. Each workflow family must retain the upstream layout, including `ALL_TASK_DIFFICULTY_RANKING.json`, `instruction.md`, `task.toml`, `environment/`, and `tests/`.

Put the source and its task data at known local paths. The `SkillFlowOfficialRunner` invokes `scripts/skillflow_official_bridge.py`, which imports the official Harbor runner in a subprocess. 

## Warm-up

Warm-up is a separate data-production phase. Qwen starts with an empty skill library, executes the configured prefix of each workflow family, and persists trajectories, verifier results, skill patches, snapshots, and a final skill repository.

```python
from pathlib import Path

from skillage import WarmupConfig, run_warmup
from skillage.runner import SkillFlowOfficialRunner

runner = SkillFlowOfficialRunner(
    Path("external/skillflow-official"),
    api_key="YOUR_MODEL_STUDIO_API_KEY",
    model="qwen3.7-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    version="skillflow-official-7b49ff5",
)
warmup = run_warmup(
    WarmupConfig(
        task_root=Path("data/SkillFlow-Task"),
        dataset_revision="upstream-revision",
        warmup_fraction=0.2,
    ),
    Path("data/warmup/run-001"),
    runner=runner,
)
```

The warm-up manifest freezes the `warmup`, `main`, and `test` task IDs, source checksums, model configuration, and artifact checksums. Main experiments read this manifest and never write back to the warm-up directory.

## Online Training

```python
from pathlib import Path

from skillage import EvalConfig, TrainConfig, evaluate, train

training = train(
    Path("data/warmup/run-001/manifest.json"),
    TrainConfig(replay_task_budget=10),
    Path("outputs/online-training"),
    runner=runner,
)
result = evaluate(
    Path("data/warmup/run-001/manifest.json"),
    training.checkpoint_path,
    EvalConfig(policy="online", replay_task_budget=10),
    Path("outputs/online-evaluation"),
    runner=runner,
)
```

`retrieval_budget` controls skill retrieval top-k.
`replay_task_budget` independently caps the first-seen historical tasks used by each quarantine replay event.

## Offline Training

```python
from pathlib import Path

from skillage import (
    EvalConfig,
    OfflineCQLConfig,
    OfflineDataConfig,
    build_offline_dataset,
    evaluate,
    train_offline,
)

dataset = build_offline_dataset(
    Path("data/warmup/run-001/manifest.json"),
    OfflineDataConfig(replay_task_budget=10),
    Path("outputs/offline-dataset"),
    runner=runner,
)
training = train_offline(
    dataset.output_dir,
    OfflineCQLConfig(),
    Path("outputs/offline-training"),
)
result = evaluate(
    Path("data/warmup/run-001/manifest.json"),
    training.checkpoint_path,
    EvalConfig(policy="base_offline"),
    Path("outputs/offline-evaluation"),
    runner=runner,
)
```

The offline manifest records the warm-up manifest checksum, dataset revision, runner version, replay budget, behavior policies, and transition checksum.