from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from skillage.config import TrainConfig
from skillage.controller import create_controller, require_torch, resolve_device
from skillage.environment import ACTIONS, FEATURE_NAMES, LifecycleEnvironment
from skillage.errors import ConfigurationError
from skillage.runner import SkillFlowRunner
from skillage.types import TrainingArtifacts
from skillage.world import SkillFlowWorld


def _seed_everything(seed: int, device: str) -> None:
    torch = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _write_checkpoint(
    path: Path,
    controller,
    optimizer,
    dual_lambda: float,
    update: int,
    config: TrainConfig,
    input_size: int,
) -> None:
    torch = require_torch()
    torch.save(
        {
            "algorithm": "skillage_online_actor_critic",
            "schema_version": 1,
            "controller_state": controller.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "dual_lambda": dual_lambda,
            "update": update,
            "input_size": input_size,
            "hidden_size": config.hidden_size,
            "train_config": config.model_dump(mode="json"),
            "torch_random_state": torch.get_rng_state(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
        },
        path,
    )


def train_policy(
    warmup_manifest: Path,
    config: TrainConfig,
    output_dir: Path,
    *,
    runner: SkillFlowRunner,
) -> TrainingArtifacts:
    torch = require_torch()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ConfigurationError(f"training output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    device = resolve_device(config.device)
    _seed_everything(config.seed, device)
    world = SkillFlowWorld(
        warmup_manifest,
        runner=runner,
        run_root=output_dir / "world-runs",
        decayed_retrieval_weight=config.decayed_retrieval_weight,
    )
    stream = world.task_stream("main")
    if not stream:
        raise ConfigurationError("bundle has no training tasks")
    environment = LifecycleEnvironment.from_config(world, config)
    input_size = len(FEATURE_NAMES)
    controller = create_controller(input_size, config.hidden_size).to(device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=config.learning_rate)
    dual_lambda = config.initial_lambda
    metrics_path = output_dir / "metrics.jsonl"
    latest_checkpoint = output_dir / "checkpoint.pt"

    with metrics_path.open("w", encoding="utf-8", newline="\n") as metrics_stream:
        for update in range(1, config.updates + 1):
            environment.reset()
            log_probabilities = []
            entropies = []
            values = []
            rewards: list[float] = []
            total_overhead = 0.0
            for task in stream:
                state = environment.observe(task)
                features = torch.as_tensor(
                    state.features, dtype=torch.float32, device=device
                )
                legal_mask = torch.as_tensor(
                    state.legal_mask, dtype=torch.bool, device=device
                )
                global_features = torch.as_tensor(
                    np.concatenate(
                        [
                            state.global_summary,
                            np.array(
                                [
                                    environment.remaining_budget
                                    / environment.initial_budget
                                ],
                                dtype=np.float32,
                            ),
                        ]
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                logits, value = controller(
                    features, legal_mask, global_features=global_features
                )
                if state.skill_ids:
                    distribution = torch.distributions.Categorical(logits=logits)
                    sampled = distribution.sample()
                    actions = {
                        skill_id: ACTIONS[int(sampled[row].item())]
                        for row, skill_id in enumerate(state.skill_ids)
                    }
                    log_probability = distribution.log_prob(sampled).mean()
                    entropy = distribution.entropy().mean()
                else:
                    actions = {}
                    log_probability = value * 0.0
                    entropy = value * 0.0
                decision, reward, replay = environment.apply(state, actions)
                constrained_reward = reward - dual_lambda * replay.degradation
                log_probabilities.append(log_probability)
                entropies.append(entropy)
                values.append(value)
                rewards.append(constrained_reward)
                total_overhead += decision.overhead

            returns: list[float] = []
            running = 0.0
            for reward in reversed(rewards):
                running = reward + config.gamma * running
                returns.append(running)
            returns.reverse()
            return_tensor = torch.tensor(
                returns, dtype=torch.float32, device=device
            )
            value_tensor = torch.stack(values)
            advantage = return_tensor - value_tensor
            actor_loss = -torch.stack(log_probabilities).mul(
                advantage.detach()
            ).mean()
            critic_loss = advantage.square().mean()
            entropy = torch.stack(entropies).mean()
            loss = (
                actor_loss
                + config.critic_coefficient * critic_loss
                - config.entropy_coefficient * entropy
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                controller.parameters(), config.gradient_clip
            )
            optimizer.step()
            episode_degradation = environment.cumulative_degradation
            dual_lambda = max(
                0.0,
                dual_lambda
                + config.dual_learning_rate
                * (episode_degradation - config.degradation_budget),
            )
            metrics_stream.write(
                json.dumps(
                    {
                        "update": update,
                        "loss": float(loss.detach().cpu()),
                        "actor_loss": float(actor_loss.detach().cpu()),
                        "critic_loss": float(critic_loss.detach().cpu()),
                        "entropy": float(entropy.detach().cpu()),
                        "gradient_norm": float(
                            torch.as_tensor(gradient_norm).detach().cpu()
                        ),
                        "dual_lambda": dual_lambda,
                        "degradation": episode_degradation,
                        "mean_overhead": total_overhead / len(stream),
                    },
                    sort_keys=True,
                )
            )
            metrics_stream.write("\n")
            metrics_stream.flush()
            if update % config.checkpoint_interval == 0 or update == config.updates:
                _write_checkpoint(
                    latest_checkpoint,
                    controller,
                    optimizer,
                    dual_lambda,
                    update,
                    config,
                    input_size,
                )
    return TrainingArtifacts(
        output_dir=output_dir,
        checkpoint_path=latest_checkpoint,
        metrics_path=metrics_path,
        updates_completed=config.updates,
        device=device,
    )
