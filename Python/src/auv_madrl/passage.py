"""Fixed one-opening passage task and the discrete adapter for the shared PPO core."""

from __future__ import annotations

import copy
import io
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .actor_variants import actor_variant, unified_match_architecture_id
from .checkpoint import save_checkpoint
from .models import mlp_body
from .passage_metrics import PassageCohorts
from .training import PPO_DEFAULTS, _attach_gae, _ppo_losses, ppo_epoch_partitions

CONTRACT = "passage-11x11-onehot-absolute4-v1"
MOVES = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]], dtype=np.int64)


class PassageEnv:
    """One vectorized scene contract for CPU preview and CUDA training."""

    def __init__(
        self, count=1, seed=0, collision_termination=False, device="cpu", maximum_steps=64,
        left_probability=0.5,
    ):
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("Environment count must be a positive integer")
        self.rng = np.random.default_rng(seed)
        self.collision_termination = collision_termination
        self.maximum_steps = maximum_steps
        self.left_probability = _probability(left_probability)
        self.position = np.zeros((count, 2), dtype=np.int64)
        self.left = np.zeros(count, dtype=bool)
        self.steps = np.zeros(count, dtype=np.int64)
        self.done = np.zeros(count, dtype=bool)
        self.device = torch.device(device)
        self.tensor_backend = self.device.type != "cpu"
        self.moves = self._array(MOVES)
        self.start = self._array(np.array([5, 9], dtype=np.int64))
        self.goal = self._array(np.array([5, 1], dtype=np.int64))
        if self.tensor_backend:
            for key in ("position", "left", "steps", "done"):
                setattr(self, key, self._array(getattr(self, key)))
        self.reset(np.ones(count, dtype=bool))

    def _array(self, value):
        return (torch.as_tensor(value, device=self.device)
                if self.tensor_backend else np.asarray(value))

    @staticmethod
    def _cpu(value):
        return value.cpu().numpy().copy() if isinstance(value, torch.Tensor) else value.copy()

    def reset(self, mask=None, side=None):
        mask = np.ones(len(self.position), dtype=bool) if mask is None else mask
        mask = self._array(mask)
        if side not in (None, "left", "right"):
            raise ValueError("Invalid passage side")
        self.position[mask] = self.start
        self.steps[mask] = 0
        self.done[mask] = False
        self.left[mask] = (self._array(self.rng.random(int(mask.sum())) < self.left_probability)
                           if side is None else side == "left")
        return self.observe()

    def observe(self):
        if self.tensor_backend:
            return torch.column_stack(((self.position - 1) / 8, self.left, ~self.left)).float()
        return torch.from_numpy(
            np.column_stack(((self.position - 1) / 8, self.left, ~self.left)).astype(np.float32)
        )

    def step(self, action):
        action = self._array(action)
        integer = (action.dtype in (torch.int32, torch.int64) if self.tensor_backend
                   else np.issubdtype(action.dtype, np.integer))
        if (
            action.shape != self.steps.shape
            or not integer
            or ((action < 0) | (action > 3)).any()
            or self.done.any()
        ):
            raise ValueError("Invalid action or episode requires reset")
        target = self.position + self.moves[action]
        opening = (torch.where if self.tensor_backend else np.where)(self.left, 1, 9)
        collision = ((target < 1) | (target > 9)).any(axis=1) | (
            (target[:, 1] == 5) & (target[:, 0] != opening)
        )
        self.position[~collision] = target[~collision]
        self.steps += 1
        success = (self.position == self.goal).all(axis=1)
        terminated = success | (collision & self.collision_termination)
        truncated = (self.steps >= self.maximum_steps) & ~terminated
        self.done = terminated | truncated
        reward = success.float() if self.tensor_backend else success.astype(np.float32)
        return self.observe(), reward, terminated, truncated, collision

    def state(self):
        return dict(
            position=self._cpu(self.position),
            left=self._cpu(self.left),
            steps=self._cpu(self.steps),
            done=self._cpu(self.done),
            rng=copy.deepcopy(self.rng.bit_generator.state),
            collision_termination=self.collision_termination,
            maximum_steps=self.maximum_steps,
            left_probability=self.left_probability,
        )

    def restore(self, state):
        count = len(self.position)
        for key, shape in (
            ("position", (count, 2)),
            ("left", (count,)),
            ("steps", (count,)),
            ("done", (count,)),
        ):
            if np.asarray(state[key]).shape != shape:
                raise ValueError("Passage state shape mismatch")
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.collision_termination = state.get("collision_termination", False)
        self.maximum_steps = state.get('maximum_steps', 64)
        self.left_probability = _probability(state.get('left_probability', 0.5))
        for key in ("position", "left", "steps", "done"):
            setattr(self, key, self._array(state[key].copy()))


def _probability(value):
    if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('left_probability must be a finite probability in [0, 1]')
    return float(value)


@dataclass(frozen=True)
class PassageConfig:
    collision_termination: bool = False
    left_probability: float = 0.5
    maximum_episode_steps: int = 128
    # User-selected passage geometry; retain the shared PPO optimizer budget.
    environments: int = 4096
    rollout_steps: int = 8
    # Only legacy checkpoints specify a fixed minibatch.
    minibatch: int | None = None
    epochs: int = PPO_DEFAULTS['optimization_epochs']
    optimizer_steps: int = PPO_DEFAULTS['optimizer_steps_per_update']
    hidden: int = actor_variant(unified_match_architecture_id('separate', 0)).body_hidden_size
    architecture: str = 'shared_mlp_v2'
    seed: int = 0
    learning_rate: float = PPO_DEFAULTS['learning_rate']
    # Passage is not time-limited; retain the user-approved preference for earlier arrival.
    gamma: float = 0.99
    gae_lambda: float = PPO_DEFAULTS['gae_lambda']
    clip_epsilon: float = PPO_DEFAULTS['clip_epsilon']
    entropy_coefficient: float = PPO_DEFAULTS['entropy_coefficient']
    value_coefficient: float = PPO_DEFAULTS['value_coefficient']
    max_grad_norm: float = PPO_DEFAULTS['maximum_gradient_norm']
    # Optional SB3-style minibatch KL stopping; None preserves existing runs.
    target_kl: float | None = None

    @property
    def batches_per_epoch(self):
        return (self.environments * self.rollout_steps // self.minibatch
                if self.minibatch is not None else self.optimizer_steps // self.epochs)

    @property
    def effective_minibatch(self):
        return self.environments * self.rollout_steps / self.batches_per_epoch

    @property
    def metric_window_updates(self):
        return (self.maximum_episode_steps + self.rollout_steps - 1) // self.rollout_steps

    def __post_init__(self):
        _probability(self.left_probability)
        if type(self.collision_termination) is not bool:
            raise ValueError("collision_termination must be a boolean")
        for name in (
            "environments", "rollout_steps", "epochs", "optimizer_steps", "hidden", "seed",
            "maximum_episode_steps",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if name == "seed" else 1)
            ):
                raise ValueError(f"Invalid {name}")
        count = self.environments * self.rollout_steps
        if self.minibatch is not None:
            if type(self.minibatch) is not int or self.minibatch < 1 or count % self.minibatch:
                raise ValueError("Minibatch must divide environments × rollout steps")
        elif (self.optimizer_steps < self.epochs or self.optimizer_steps % self.epochs
              or count < self.batches_per_epoch):
            raise ValueError("Invalid fixed optimizer step budget")
        if self.architecture not in {'shared_mlp_v2', 'legacy_tanh_v1'}:
            raise ValueError("Unsupported passage architecture")
        if count > 1048576 or self.hidden > 1024 or self.epochs > 100:
            raise ValueError("Configuration exceeds supported allocation")
        for name in (
            "learning_rate",
            "gamma",
            "gae_lambda",
            "clip_epsilon",
            "entropy_coefficient",
            "value_coefficient",
            "max_grad_norm",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}")
        if (
            not 0 < self.learning_rate
            or not 0 < self.max_grad_norm
            or self.gamma > 1
            or self.gae_lambda > 1
        ):
            raise ValueError("Invalid optimizer configuration")
        if self.target_kl is not None and (
            isinstance(self.target_kl, bool)
            or not np.isfinite(self.target_kl)
            or self.target_kl <= 0
        ):
            raise ValueError("target_kl must be finite and positive, or None")


class PassageActorCritic(nn.Module):
    def __init__(self, hidden=128, architecture='shared_mlp_v2'):
        super().__init__()
        activation = nn.Tanh if architecture == 'legacy_tanh_v1' else nn.SiLU
        self.actor = nn.Sequential(*mlp_body(4, hidden, activation), nn.Linear(hidden, 4))
        self.critic = nn.Sequential(*mlp_body(4, hidden, activation), nn.Linear(hidden, 1))

    def forward(self, observation):
        return Categorical(logits=self.actor(observation)), self.critic(observation)


class PassageTrainer:
    """Discrete collection/update adapter; GAE, PPO loss, and atomic saves are shared."""

    def __init__(self, config=None, device="cpu"):
        config = config or PassageConfig()
        self.config = config
        self.device = torch.device(device)
        self.env = PassageEnv(
            config.environments, config.seed, config.collision_termination, self.device,
            maximum_steps=config.maximum_episode_steps,
            left_probability=config.left_probability,
        )
        # A private generator avoids disturbing continuous training/Viewer sampling.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            self.model = PassageActorCritic(config.hidden, config.architecture)
        self.model.to(self.device)
        self.generator = torch.Generator().manual_seed(config.seed + 1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.updates = self.frames = 0
        self.cohorts = PassageCohorts(config)

    def update(self):
        cfg = asdict(self.config)
        fields = {
            key: []
            for key in (
                "observation",
                "action",
                "old_log_probability",
                "value",
                "next_value",
                "reward",
                "done",
                "terminated",
            )
        }
        completed = successes = left_episodes = right_episodes = 0
        left_frames = left_successes = 0
        for _ in range(self.config.rollout_steps):
            left_frames += int(self.env.left.sum())
            obs = self.env.observe()
            with torch.no_grad():
                distribution, value = self.model(obs)
                # Preserve the saved CPU RNG stream across CPU/CUDA migrations.
                action = torch.multinomial(
                    distribution.probs.cpu(), 1, generator=self.generator
                ).squeeze(-1).to(self.device)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                _, next_value = self.model(next_obs)
            done = terminated | truncated
            values = dict(
                observation=obs,
                action=action,
                old_log_probability=distribution.log_prob(action).unsqueeze(-1),
                value=value,
                next_value=next_value,
                reward=torch.as_tensor(reward, device=self.device).unsqueeze(-1),
                done=torch.as_tensor(done, device=self.device).unsqueeze(-1),
                terminated=torch.as_tensor(terminated, device=self.device).unsqueeze(-1),
            )
            for key, value in values.items():
                fields[key].append(value)
            completed += int(done.sum())
            successes += int((reward > 0).sum())
            left_successes += int(((reward > 0) & self.env.left).sum())
            left_episodes += int((done & self.env.left).sum())
            right_episodes += int((done & ~self.env.left).sum())
            self.env.reset(done)
        rollout = {key: torch.stack(value) for key, value in fields.items()}
        # One compact transfer per rollout; reuse sampled outcomes and side flags.
        cohort_data = torch.stack((rollout['done'][..., 0], rollout['reward'][..., 0],
                                   rollout['observation'][..., 2]), dim=-1).cpu().numpy()
        training_cohorts = self.cohorts.observe(cohort_data)
        rollout["credit_member"] = torch.ones_like(rollout["done"])
        _attach_gae(rollout, cfg)
        flat = {key: value.flatten(0, 1) for key, value in rollout.items()}
        advantage = flat["advantage"]
        flat["advantage"] = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        count = len(advantage)
        metrics = []
        checked_kls = []
        early_stopped = False
        for _ in range(self.config.epochs):
            for index in ppo_epoch_partitions(count, self.config.batches_per_epoch,
                    device=self.device, generator=self.generator):
                distribution, value = self.model(flat["observation"][index])
                loss, result = _ppo_losses(
                    distribution,
                    value,
                    flat["action"][index],
                    flat["old_log_probability"][index],
                    flat["advantage"][index],
                    flat["value_target"][index],
                    cfg,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO loss")
                checked_kls.append(result[3])
                # Same estimate and 1.5 multiplier as SB3: stop before the next
                # gradient step, then collect fresh data on the next update.
                if (self.config.target_kl is not None
                        and result[3] > 1.5 * self.config.target_kl):
                    early_stopped = True
                    break
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm, error_if_nonfinite=True
                )
                self.optimizer.step()
                metrics.append(result)
            if early_stopped:
                break
        self.updates += 1
        self.frames += count
        averaged = torch.stack(metrics or [result]).mean(0).tolist()
        kl_checks = torch.stack(checked_kls)
        return dict(
            update=self.updates,
            frames=self.frames,
            training_cohorts=training_cohorts,
            optimizer_steps=len(metrics),
            kl_early_stopped=early_stopped,
            last_checked_kl=kl_checks[-1].item(),
            max_checked_kl=kl_checks.max().item(),
            effective_minibatch=self.config.effective_minibatch,
            episodes=completed,
            successes=successes,
            left_episodes=left_episodes,
            right_episodes=right_episodes,
            left_frames=left_frames,
            right_frames=count - left_frames,
            left_successes=left_successes,
            right_successes=successes - left_successes,
            left_success_rate=left_successes / left_episodes if left_episodes else None,
            right_success_rate=(successes-left_successes) / right_episodes
                if right_episodes else None,
            success_rate=successes / completed if completed else None,
            policy_loss=averaged[0],
            value_loss=averaged[1],
            entropy=averaged[2],
            approximate_kl=averaged[3],
            clip_fraction=averaged[4],
        )

    def save(self, path: Path):
        save_checkpoint(
            path,
            dict(
                contract=CONTRACT,
                config=asdict(self.config),
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                env=self.env.state(),
                rng=self.generator.get_state(),
                updates=self.updates,
                frames=self.frames,
                training_device=str(self.device),
                training_cohorts=self.cohorts.state(),
            ),
        )

    @classmethod
    def load(cls, path: Path, device="cpu"):
        # Release the replaceable file before deserialization or model construction.
        payload = torch.load(io.BytesIO(path.read_bytes()), map_location="cpu", weights_only=False)
        if payload.get("contract") != CONTRACT:
            raise ValueError("Checkpoint is not a compatible discrete passage model")
        config = dict(payload["config"])
        config.setdefault('architecture', 'legacy_tanh_v1')
        config.setdefault('maximum_episode_steps', 64)
        trainer = cls(PassageConfig(**config), device=device)
        trainer.model.load_state_dict(payload["model"], strict=True)
        trainer.optimizer.load_state_dict(payload["optimizer"])
        trainer.env.restore(payload["env"])
        trainer.generator.set_state(payload["rng"])
        trainer.updates, trainer.frames = payload["updates"], payload["frames"]
        if 'training_cohorts' in payload:
            trainer.cohorts.restore(payload['training_cohorts'])
        else:
            trainer.cohorts = PassageCohorts(trainer.config,
                tick=trainer.updates * trainer.config.rollout_steps,
                episode_steps=payload['env']['steps'])
        return trainer
