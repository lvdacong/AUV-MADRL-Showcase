from __future__ import annotations

import math
import json
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict
from torch import nn

from .actor_variants import entity_tensor_prefix
from .goal_reach_env import GoalReachEnv
from .match_metrics import (
    MATCH_END_REASON_CODES,
    MATCH_EVENT_INFO_KEYS,
    MatchEndReason,
    MatchOutcome,
)
from .models import ActorCritic, ClippedNormal


def concat_observation_batches(items, keys):
    """Pad only obstacle tokens; keep all other feature contracts strict."""
    if not items:
        raise ValueError('Expected at least one observation batch')
    reference = items[0]['obstacle_entities']
    for item in items:
        entities, mask = item['obstacle_entities'], item['obstacle_mask']
        if (entities.ndim != 3 or mask.shape != entities.shape[:2] or mask.dtype != torch.bool
                or entities.device != reference.device or mask.device != reference.device
                or entities.dtype != reference.dtype or entities.shape[-1] != reference.shape[-1]):
            raise ValueError('Inconsistent obstacle entity/mask batch')
    width = max(item['obstacle_mask'].shape[1] for item in items)
    result = {}
    for key in keys:
        values = []
        for item in items:
            value = item[key]
            if key == 'obstacle_entities':
                value = torch.nn.functional.pad(value, (0, 0, 0, width - value.shape[1])) if value.shape[1] != width else value
            elif key == 'obstacle_mask':
                value = torch.nn.functional.pad(value, (0, width - value.shape[1])) if value.shape[1] != width else value
            values.append(value)
        result[key] = torch.cat(values)
    return result


class TrainingMemoryProbe:
    """Opt-in bounded, in-process stage measurements; no tensor ownership retained."""
    def __init__(self, path, device, maximum_updates):
        self.path, self.device = path, torch.device(device)
        self.maximum_updates = maximum_updates
        self.update = 0

    def sample(self, stage, env, rollouts=None):
        if self.update > self.maximum_updates:
            return
        record = dict(pid=os.getpid(), wall_time=time.time(), update=self.update, stage=stage,
                      world_capacity=env.obstacle_capacity)
        record.update(getattr(env, '_training_memory_metadata', {}))
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            record.update(allocated=torch.cuda.memory_allocated(self.device),
                          reserved=torch.cuda.memory_reserved(self.device),
                          peak_allocated=torch.cuda.max_memory_allocated(self.device),
                          peak_reserved=torch.cuda.max_memory_reserved(self.device))
        if hasattr(env,'_collision_neighbor_counts'):
            record['collision_neighbor_counts'] = dict(zip(
                ('safe_lane_ticks','invalid_lane_ticks','distance_fallback_lane_ticks','cached_exact_pairs','dense_exact_pairs'),
                env._collision_neighbor_counts.tolist()))
        if getattr(env,'_neighbor_trace_events',None):
            timings = {}
            for label,start,end in env._neighbor_trace_events:
                timings[label] = timings.get(label,0.) + start.elapsed_time(end)/1000.
            record['collision_phase_cuda_seconds'] = timings
            env._neighbor_trace_events.clear()
        if rollouts is not None:
            storages = {}; obstacle_storages = {}; shapes = {}; samples = {}
            for role, rollout in rollouts.items():
                shapes[role] = {k:list(v.shape) for k,v in rollout.items()}
                samples[role] = {k:int(rollout[k].sum()) for k in ('credit_member','actor_alive') if k in rollout}
                for key,value in rollout.items():
                    storage=value.untyped_storage()
                    identity=(str(value.device),storage.data_ptr())
                    storages[identity]=storage.nbytes()
                    if key.startswith('obstacle_'):obstacle_storages[identity]=storage.nbytes()
            record.update(unique_rollout_bytes=sum(storages.values()),
                          unique_obstacle_bytes=sum(obstacle_storages.values()),shapes=shapes,samples=samples)
        with self.path.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
        if self.device.type == 'cuda':torch.cuda.reset_peak_memory_stats(self.device)


def _memory_sample(env, stage, rollouts=None):
    probe = getattr(env, '_training_memory_probe', None)
    if probe is not None:probe.sample(stage, env, rollouts)


@dataclass
class UpdateMetrics:
    policy_loss: float | None
    value_loss: float | None
    entropy: float | None
    approximate_kl: float | None
    clip_fraction: float | None
    gradient_norm: float
    parameter_delta_max: float
    log_probability_error_max: float
    finite: bool
    peak_reserved_gib: float
    actor_parameter_delta_max: float = 0.0
    critic_parameter_delta_max: float = 0.0
    explained_variance: float | None = 0.0
    actor_valid_frames: int = 0
    critic_valid_frames: int = 0
    optimizer_steps: int = 0
    microbatch_count: int = 0
    effective_minibatch_agent_frames: float = 0.0


@dataclass
class PolicyParameterGroup:
    key: str
    model: ActorCritic
    optimizer: torch.optim.Optimizer | None
    update_index: int = 0
    optimizer_steps: int = 0


@dataclass
class GoalReachRuntime:
    env: GoalReachEnv
    model: ActorCritic | None
    optimizer: torch.optim.Optimizer | None
    observation: dict[str, torch.Tensor]
    is_init: torch.Tensor
    training_target: str = "learner"
    update_index: int = 0
    agent_frames: int = 0
    weights_initialization: dict[str, object] | None = None
    training_restore: dict[str, object] | None = None
    parameter_group: PolicyParameterGroup | None = None
    rule: dict[str, object] | None = None
    episode_returns: torch.Tensor = field(init=False)
    match_event_counts: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        if self.rule is not None:
            if self.training_target != "learner" or self.model is not None or self.optimizer is not None or self.parameter_group is not None:
                raise ValueError("APF runtime must be a model-free fixed learner source")
            from .apf import APFConfig
            self.rule = APFConfig.from_dict(self.rule).to_dict()
        elif self.model is None:
            raise ValueError("Neural runtime requires its actual model")
        if self.parameter_group is not None:
            self.model = self.parameter_group.model
            self.optimizer = self.parameter_group.optimizer
        actor_shape = (
            tuple(self.observation["actor_active"].shape)
            if "actor_active" in self.observation
            else (self.env.batch_size,)
        )
        self.episode_returns = torch.zeros(
            actor_shape, dtype=torch.float32, device=self.env.device
        )
        self.match_event_counts = torch.zeros(
            (self.env.batch_size, len(MATCH_EVENT_INFO_KEYS)), dtype=torch.int32, device=self.env.device
        )


class _CollectionPhaseProfiler:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.cuda_events: dict[str, list[tuple[object, object]]] = {}
        self.cpu_seconds: dict[str, float] = {}

    def start(self) -> object:
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def stop(self, name: str, started: object) -> None:
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.cuda_events.setdefault(name, []).append((started, event))
            return
        elapsed = time.perf_counter() - float(started)
        self.cpu_seconds[name] = self.cpu_seconds.get(name, 0.0) + elapsed

    def synchronize_and_seconds(self) -> dict[str, float]:
        if self.device.type != "cuda":
            return dict(self.cpu_seconds)
        torch.cuda.synchronize(self.device)
        return {
            name: sum(start.elapsed_time(end) for start, end in events) / 1000.0
            for name, events in self.cuda_events.items()
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def _exclusive_episode_outcome_masks(
    done: torch.Tensor,
    success: torch.Tensor,
    leakage_failure: torch.Tensor,
    collision: torch.Tensor,
    timeout: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    leakage_failure = done & leakage_failure
    collision = done & collision & ~leakage_failure
    timeout = done & timeout & ~leakage_failure & ~collision
    success = done & success & ~leakage_failure & ~collision & ~timeout
    return success, leakage_failure, collision, timeout


def _observation_slice(
    values: dict[str, torch.Tensor],
    keys: tuple[str, ...],
    index: object,
) -> dict[str, torch.Tensor]:
    return {key: values[key][index] for key in keys}


def _flatten_actor_observation(
    observation: dict[str, torch.Tensor],
    keys: tuple[str, ...],
    actor_shape: torch.Size,
) -> dict[str, torch.Tensor]:
    leading_dimensions = len(actor_shape)
    actor_count = math.prod(actor_shape)
    return {
        key: observation[key].reshape(
            actor_count, *observation[key].shape[leading_dimensions:]
        )
        for key in keys
    }


def _stack_rollout_storage(
    storage: dict[str, list[torch.Tensor] | torch.Tensor], variant: object
) -> dict[str, torch.Tensor]:
    """Stack a rollout while padding only variable-capacity entity axes."""
    entity_axes: dict[str, tuple[str, int]] = {}
    observation_keys = set(variant.observation_keys)
    for buffer_name in variant.entity_sets:
        prefix = entity_tensor_prefix(buffer_name)
        for key in (f"{prefix}_entities", f"{prefix}_critic_entities"):
            if key in observation_keys:
                entity_axes[key] = (prefix, -2)
        for key in (f"{prefix}_mask", f"{prefix}_critic_mask"):
            if key in observation_keys:
                entity_axes[key] = (prefix, -1)

    entity_capacities: dict[str, int] = {}
    capacities_by_frame: dict[str, list[int]] = {}
    for key, (prefix, axis) in entity_axes.items():
        if key not in storage:
            raise ValueError(f"Rollout storage is missing entity tensor: {key}")
        values = storage[key]
        frames = tuple(values.unbind(0)) if isinstance(values, torch.Tensor) else tuple(values)
        capacities = [int(frame.shape[axis % frame.ndim]) for frame in frames]
        if prefix in capacities_by_frame and capacities_by_frame[prefix] != capacities:
            raise ValueError(f"Rollout entity tensors have unsynchronized capacity: {prefix}")
        capacities_by_frame[prefix] = capacities
        entity_capacities[prefix] = max(capacities, default=0)

    stacked: dict[str, torch.Tensor] = {}
    for key, values in storage.items():
        frames = tuple(values.unbind(0)) if isinstance(values, torch.Tensor) else tuple(values)
        if not frames:
            raise ValueError(f"Rollout storage is empty for {key}")
        entity_axis = entity_axes.get(key)
        if entity_axis is None:
            stacked[key] = torch.stack(frames, dim=0)
            continue
        prefix, axis = entity_axis
        rank = frames[0].ndim
        normalized_axis = axis % rank
        reference = list(frames[0].shape)
        capacity = entity_capacities[prefix]
        padded = []
        for frame in frames:
            if frame.ndim != rank or any(
                frame.shape[index] != size
                for index, size in enumerate(reference)
                if index != normalized_axis
            ):
                raise ValueError(f"Rollout entity tensor shape changed outside capacity: {key}")
            missing = capacity - int(frame.shape[normalized_axis])
            if missing:
                shape = list(frame.shape)
                shape[normalized_axis] = missing
                fill = torch.zeros(shape, dtype=frame.dtype, device=frame.device)
                frame = torch.cat((frame, fill), dim=normalized_axis)
            padded.append(frame)
        stacked[key] = torch.stack(padded, dim=0)
    return stacked


@torch.no_grad()
def collect_goal_reach(
    runtime: GoalReachRuntime,
    rollout_length: int,
    snapshot_callback: Callable[
        [
            GoalReachEnv,
            dict[str, torch.Tensor],
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        None,
    ]
    | None = None,
) -> tuple[TensorDict, dict[str, float | None]]:
    observation_keys = runtime.model.variant.observation_keys
    bootstrap_on_time_limit = bool(
        runtime.env.config.goal_reach_trainer["optimizer"].get(
            "bootstrap_on_time_limit", True
        )
    )
    storage: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            *observation_keys,
            "action",
            "old_log_probability",
            "value",
            "reward",
            "done",
            "terminated",
            "is_init",
            "actor_alive",
            "credit_member",
            "valid",
        )
    }
    terminal_bootstraps: list[torch.Tensor] = []
    completed_outcomes: list[torch.Tensor] = []
    completed_collision_causes: list[torch.Tensor] = []
    completed_lengths: list[torch.Tensor] = []
    completed_match_facts: list[torch.Tensor] = []
    completed_match_events: list[torch.Tensor] = []
    completed_utilities=[]
    target_casualties=[]
    explicit_end_reasons=[]
    sigma_sum=torch.zeros(2,device=runtime.env.device)
    sigma_count=0
    completed_episode_return_sum = torch.zeros((), device=runtime.env.device)
    rollout_reward_sum = torch.zeros((), device=runtime.env.device)
    final_next_observation: dict[str, torch.Tensor] | None = None
    collected_agent_frames = torch.zeros(
        (), dtype=torch.int64, device=runtime.env.device
    )
    started = time.perf_counter()

    for _ in range(rollout_length):
        observation = runtime.observation
        actor_alive = observation.get(
            "actor_alive",
            torch.ones(
                runtime.env.batch_size, dtype=torch.bool, device=runtime.env.device
            ),
        ).to(torch.bool)
        credit_member = observation.get("credit_member", actor_alive).to(torch.bool)
        actor_shape = actor_alive.shape
        flat_observation = _flatten_actor_observation(
            observation, observation_keys, actor_shape
        )
        flat_alive = actor_alive.reshape(-1)
        flat_credit = credit_member.reshape(-1)
        model_observation = {
            key: value[flat_alive] for key, value in flat_observation.items()
        }
        if runtime.model.uses_centralized_critic:
            distribution = runtime.model.actor_distribution_from_observation(
                model_observation
            )
            grouped_value = runtime.model.value_from_grouped_observation(
                _observation_slice(observation, observation_keys, slice(None)),
                credit_member,
            )
            credit_value = grouped_value.reshape(-1, 1)[flat_credit]
        else:
            distribution = runtime.model.actor_distribution_from_observation(
                model_observation
            )
            credit_value = runtime.model.value_from_observation(
                {key: value[flat_credit] for key, value in flat_observation.items()}
            )
        active_action = distribution.rsample()
        sigma_sum+=distribution.scale.sum(0)
        sigma_count+=distribution.scale.shape[0]
        flat_action = torch.zeros(
            (flat_alive.shape[0], 2), dtype=torch.float32, device=runtime.env.device
        )
        flat_value = torch.zeros(
            (flat_alive.shape[0], 1), dtype=torch.float32, device=runtime.env.device
        )
        flat_log_probability = torch.zeros_like(flat_value)
        flat_location = torch.zeros_like(flat_action)
        flat_scale = torch.zeros_like(flat_action)
        flat_action[flat_alive] = active_action
        flat_value[flat_credit] = credit_value
        flat_log_probability[flat_alive] = distribution.log_prob(
            active_action
        ).unsqueeze(-1)
        flat_location[flat_alive] = distribution.loc
        flat_scale[flat_alive] = distribution.scale
        action = flat_action.reshape(*actor_shape, 2)
        value = flat_value.reshape(*actor_shape, 1)
        old_log_probability = flat_log_probability.reshape(*actor_shape, 1)
        if snapshot_callback is not None:
            snapshot_callback(
                runtime.env,
                observation,
                action,
                flat_location.reshape(*actor_shape, 2),
                flat_scale.reshape(*actor_shape, 2),
                value,
            )
        transition = runtime.env.training_step(action)
        next_observation = transition.observation
        done = transition.terminated | transition.truncated
        truncated = transition.truncated & ~transition.terminated
        runtime.episode_returns.add_(transition.reward)
        rollout_reward_sum += transition.reward.sum()
        completed_episode_return_sum += torch.where(
            done, runtime.episode_returns, torch.zeros_like(runtime.episode_returns)
        ).sum()
        success_done, leakage_failure_done, collision_done, timeout_done = (
            _exclusive_episode_outcome_masks(
                done,
                transition.info["success"],
                transition.info.get(
                    "leakage_failure", torch.zeros_like(done)
                ),
                transition.info["collision"],
                transition.info.get("horizon", transition.truncated),
            )
        )
        step_match_events = []
        for key in MATCH_EVENT_INFO_KEYS:
            event_value = transition.info.get(key)
            step_match_events.append(
                event_value.reshape(runtime.env.batch_size, -1).sum(
                    dim=1, dtype=torch.int32
                )
                if isinstance(event_value, torch.Tensor)
                else torch.zeros(
                    runtime.env.batch_size,
                    dtype=torch.int32,
                    device=runtime.env.device,
                )
            )
        runtime.match_event_counts += torch.stack(step_match_events, dim=-1)
        match_done = transition.info.get("match_done")
        if isinstance(match_done, torch.Tensor):
            match_outcome = transition.info["match_outcome"]
            match_end_reason = transition.info["match_end_reason"]
            completed_utilities.append(transition.info['episode_defense_utility']*match_done)
            explicit_end_reasons.append(torch.where(match_done, match_end_reason, 0))
            target_casualties.append(match_done&(match_end_reason==MATCH_END_REASON_CODES[MatchEndReason.TARGET_CASUALTY]))
            completed_match_facts.append(
                torch.stack(
                    (
                        match_done,
                        match_done & (match_outcome == int(MatchOutcome.ATTACKER_WIN)),
                        match_done & (match_outcome == int(MatchOutcome.DEFENDER_WIN)),
                        match_done
                        & (
                            match_end_reason
                            == MATCH_END_REASON_CODES[MatchEndReason.BREACH]
                        ),
                        match_done
                        & (
                            match_end_reason
                            == MATCH_END_REASON_CODES[MatchEndReason.ELIMINATION]
                        ),
                        match_done
                        & (
                            match_end_reason
                            == MATCH_END_REASON_CODES[MatchEndReason.HORIZON]
                        ),
                        match_done
                        & (
                            match_end_reason
                            == MATCH_END_REASON_CODES[
                                MatchEndReason.ATTACKER_CASUALTY
                            ]
                        ),
                        match_done
                        & (
                            match_end_reason
                            == MATCH_END_REASON_CODES[
                                MatchEndReason.TARGET_CAPTURE
                            ]
                        ),
                        match_done & (match_end_reason == MATCH_END_REASON_CODES[MatchEndReason.DEFENDER_CASUALTY]),
                    ),
                    dim=-1,
                )
            )
            completed_match_events.append(
                runtime.match_event_counts.to(torch.float32)
                * match_done.to(torch.float32).unsqueeze(-1)
            )
            runtime.match_event_counts.masked_fill_(match_done.unsqueeze(-1), 0)
            target_win = (
                int(MatchOutcome.ATTACKER_WIN)
                if runtime.training_target == "learner"
                else int(MatchOutcome.DEFENDER_WIN)
            )
            completed_done = match_done
            completed_success = match_done & (match_outcome == target_win)
            completed_leakage_failure = (
                match_done
                & (match_outcome == int(MatchOutcome.ATTACKER_WIN))
                & (runtime.training_target == "defender")
            )
            completed_collision = torch.zeros_like(match_done)
            completed_timeout = match_done & (
                match_end_reason
                == MATCH_END_REASON_CODES[MatchEndReason.HORIZON]
            )
        else:
            completed_match_facts.append(
                torch.zeros(
                    (*runtime.env.done.shape, 9),
                    dtype=torch.bool,
                    device=runtime.env.device,
                )
            )
            completed_match_events.append(
                torch.zeros(
                    (*runtime.env.done.shape, len(MATCH_EVENT_INFO_KEYS)),
                    dtype=torch.float32,
                    device=runtime.env.device,
                )
            )
            completed_done = done
            completed_success = success_done
            completed_leakage_failure = leakage_failure_done
            completed_collision = collision_done
            completed_timeout = timeout_done
        boundary_collision_done = transition.info.get(
            "boundary_collision", torch.zeros_like(collision_done)
        ) & collision_done
        obstacle_collision_done = transition.info.get(
            "obstacle_collision", torch.zeros_like(collision_done)
        ) & collision_done
        pair_collision_done = transition.info.get(
            "pair_collision", torch.zeros_like(collision_done)
        ) & collision_done
        runtime.episode_returns.masked_fill_(done, 0.0)
        scene_done = runtime.env.done.clone()
        scene_truncated = truncated.reshape(runtime.env.batch_size, -1).any(dim=1)
        done_indices = torch.nonzero(scene_done, as_tuple=False).flatten()
        timeout_indices = (
            torch.nonzero(scene_truncated, as_tuple=False).flatten()
            if bootstrap_on_time_limit
            else torch.empty(0, dtype=torch.int64, device=runtime.env.device)
        )
        terminal_bootstrap = torch.zeros_like(value)
        if timeout_indices.numel() > 0:
            if runtime.model.uses_centralized_critic:
                next_credit_member = next_observation.get(
                    "credit_member", next_observation["actor_active"]
                ).to(torch.bool)
                next_grouped_value = runtime.model.value_from_grouped_observation(
                    _observation_slice(
                        next_observation, observation_keys, slice(None)
                    ),
                    next_credit_member,
                )
                terminal_bootstrap[truncated] = next_grouped_value[truncated]
            else:
                terminal_bootstrap[truncated] = runtime.model.value_from_observation(
                    _observation_slice(next_observation, observation_keys, truncated)
                )

        # Every active observation tensor changes with state, so rollout owns each snapshot.
        for key in observation_keys:
            storage[key].append(observation[key].clone())
        storage["action"].append(action)
        storage["old_log_probability"].append(old_log_probability)
        storage["value"].append(value)
        storage["reward"].append(transition.reward.unsqueeze(-1))
        storage["done"].append(done.unsqueeze(-1))
        storage["terminated"].append(transition.terminated.unsqueeze(-1))
        storage["is_init"].append(runtime.is_init.unsqueeze(-1))
        storage["actor_alive"].append(actor_alive.unsqueeze(-1))
        storage["credit_member"].append(credit_member.unsqueeze(-1))
        storage["valid"].append(actor_alive.unsqueeze(-1))
        terminal_bootstraps.append(terminal_bootstrap)

        completed_outcomes.append(
            torch.stack(
                (
                    completed_done,
                    completed_success,
                    completed_leakage_failure,
                    completed_collision,
                    completed_timeout,
                ),
                dim=-1,
            )
        )
        completed_collision_causes.append(
            torch.stack(
                (
                    boundary_collision_done,
                    obstacle_collision_done,
                    pair_collision_done,
                ),
                dim=-1,
            )
        )
        completed_lengths.append(
            torch.ceil(
                (
                    runtime.env.physics_ticks.to(torch.float32)
                    if isinstance(match_done, torch.Tensor)
                    else transition.info["physics_ticks"].to(torch.float32)
                )
                / runtime.env.config.goal_reach["clock"]["physics_ticks_per_decision"]
            )
        )
        if done_indices.numel() > 0:
            next_observation = runtime.env.reset(
                scene_done,
                base_observation=next_observation,
                env_indices=done_indices,
            )
        runtime.observation = next_observation
        if "credit_member" in next_observation:
            next_credit = next_observation["credit_member"].to(torch.bool)
            scene_reset = scene_done.reshape(
                runtime.env.batch_size,
                *([1] * (next_credit.ndim - 1)),
            )
            runtime.is_init = scene_reset & next_credit
        else:
            runtime.is_init = scene_done
        collected_agent_frames += actor_alive.sum()
        final_next_observation = transition.observation

    assert final_next_observation is not None
    final_credit_member = final_next_observation.get(
        "credit_member",
        torch.ones(
            runtime.env.batch_size, dtype=torch.bool, device=runtime.env.device
        ),
    )
    if runtime.model.uses_centralized_critic:
        final_next_value = runtime.model.value_from_grouped_observation(
            _observation_slice(
                final_next_observation, observation_keys, slice(None)
            ),
            final_credit_member,
        )
    else:
        flat_final_observation = _flatten_actor_observation(
            final_next_observation, observation_keys, final_credit_member.shape
        )
        flat_final_credit = final_credit_member.reshape(-1)
        flat_final_value = torch.zeros(
            (flat_final_credit.numel(), 1),
            dtype=torch.float32,
            device=runtime.env.device,
        )
        flat_final_value[flat_final_credit] = runtime.model.value_from_observation(
            {
                key: value[flat_final_credit]
                for key, value in flat_final_observation.items()
            }
        )
        final_next_value = flat_final_value.reshape(*final_credit_member.shape, 1)
    storage["next_value"] = _shifted_next_values(
        storage["value"],
        storage["done"],
        storage["terminated"],
        terminal_bootstraps,
        final_next_value,
    )
    if runtime.env.device.type == "cuda":
        torch.cuda.synchronize(runtime.env.device)
    elapsed = time.perf_counter() - started
    frames = int(collected_agent_frames.item())
    runtime.agent_frames += frames
    rollout_batch_size = [
        rollout_length,
        *storage["credit_member"][0].shape[:-1],
    ]
    rollout = TensorDict(
        _stack_rollout_storage(storage, runtime.model.variant),
        batch_size=rollout_batch_size,
        device=runtime.env.device,
    )
    _attach_gae(rollout, runtime.env.config.goal_reach_trainer["optimizer"])
    outcomes = torch.stack(completed_outcomes, dim=0)
    collision_causes = torch.stack(completed_collision_causes, dim=0)
    lengths = torch.stack(completed_lengths, dim=0)
    match_facts = torch.stack(completed_match_facts, dim=0)
    match_events = torch.stack(completed_match_events, dim=0)
    match_summary = torch.cat(
        (
            match_facts.sum(
                dim=tuple(range(match_facts.ndim - 1)), dtype=torch.float32
            ),
            (
                match_facts[..., 0].to(lengths.dtype) * lengths
            ).sum().reshape(1),
            match_events.sum(
                dim=tuple(range(match_events.ndim - 1)), dtype=torch.float32
            ),
        )
    ).cpu()
    summary = torch.cat(
        (
            outcomes.sum(
                dim=tuple(range(outcomes.ndim - 1)), dtype=torch.float32
            ),
            (outcomes.to(lengths.dtype) * lengths.unsqueeze(-1)).sum(
                dim=tuple(range(outcomes.ndim - 1))
            ),
            collision_causes.sum(
                dim=tuple(range(collision_causes.ndim - 1)), dtype=torch.float32
            ),
            completed_episode_return_sum.reshape(1),
            rollout_reward_sum.reshape(1),
        )
    ).cpu()
    (
        episodes,
        successes,
        leakage_failures,
        collisions,
        timeouts,
        episode_decisions,
        success_decisions,
        leakage_failure_decisions,
        collision_decisions,
        timeout_decisions,
        boundary_collisions,
        obstacle_collisions,
        pair_collisions,
        completed_return_sum,
        total_reward_sum,
    ) = summary.tolist()
    episode_return_mean = completed_return_sum / episodes if episodes else None
    (
        matches,
        attacker_wins,
        defender_wins,
        breach_ends,
        elimination_ends,
        horizon_ends,
        attacker_casualty_ends,
        target_capture_ends,
        defender_casualty_ends,
        match_decisions,
        attacker_reach_events,
        attacker_capture_events,
        attacker_static_collision_events,
        attacker_vessel_collision_events,
        defender_static_collision_events,
        defender_vessel_collision_events,
        target_capture_events, target_static_collision_events, target_vessel_collision_events,
        *boundary_events,
    ) = match_summary.tolist()
    return rollout, {
        "sigma": {runtime.training_target:(sigma_sum/max(1,sigma_count)).cpu().tolist()},
        "collect_seconds": elapsed,
        "agent_frames": frames,
        "agent_frames_per_second": frames / elapsed,
        "episodes": float(episodes),
        "matches": float(matches),
        "attacker_wins": float(attacker_wins),
        "defender_wins": float(defender_wins),
        "breach_ends": float(breach_ends),
        "elimination_ends": float(elimination_ends),
        "horizon_ends": float(horizon_ends),
        "truncation_ends": float((torch.stack(explicit_end_reasons) == MATCH_END_REASON_CODES[MatchEndReason.TRUNCATION]).sum()) if explicit_end_reasons else 0.,
        "defender_elimination_ends": float((torch.stack(explicit_end_reasons) == MATCH_END_REASON_CODES[MatchEndReason.DEFENDER_ELIMINATION]).sum()) if explicit_end_reasons else 0.,
        "attacker_casualty_ends": float(attacker_casualty_ends),
        "target_capture_ends": float(target_capture_ends),
        "target_casualty_ends": float(torch.stack(target_casualties).sum()) if target_casualties else None,
        "completed_defense_return": float(torch.stack(completed_utilities).sum()) if completed_utilities else None,
        "defender_casualty_ends": float(defender_casualty_ends),
        "match_decisions": float(match_decisions),
        "target_capture_events": float(target_capture_events),
        "target_static_collision_events": float(target_static_collision_events),
        "target_vessel_collision_events": float(target_vessel_collision_events),
        **{key.removeprefix('event_')+'_events':value for key,value in zip(MATCH_EVENT_INFO_KEYS[9:],boundary_events)},
        "attacker_reach_events": float(attacker_reach_events),
        "attacker_capture_events": float(attacker_capture_events),
        "attacker_static_collision_events": float(
            attacker_static_collision_events
        ),
        "attacker_vessel_collision_events": float(
            attacker_vessel_collision_events
        ),
        "defender_static_collision_events": float(
            defender_static_collision_events
        ),
        "defender_vessel_collision_events": float(
            defender_vessel_collision_events
        ),
        "successes": float(successes),
        "leakage_failures": float(leakage_failures),
        "collisions": float(collisions),
        "boundary_collisions": float(boundary_collisions),
        "obstacle_collisions": float(obstacle_collisions),
        "pair_collisions": float(pair_collisions),
        "timeouts": float(timeouts),
        "episode_return_mean": episode_return_mean,
        "reward_sum": total_reward_sum,
        "reward_rate": total_reward_sum / max(frames, 1),
        "episode_decisions_mean": episode_decisions / episodes if episodes else None,
        "success_episode_decisions_mean": (
            success_decisions / successes if successes else None
        ),
        "leakage_failure_episode_decisions_mean": (
            leakage_failure_decisions / leakage_failures
            if leakage_failures
            else None
        ),
        "collision_episode_decisions_mean": (
            collision_decisions / collisions if collisions else None
        ),
        "timeout_episode_decisions_mean": timeout_decisions / timeouts if timeouts else None,
    }


def _sample_policy_step(
    runtime: GoalReachRuntime,
    observation: dict[str, torch.Tensor],
    *,
    collect_training_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    sampled, _ = _sample_parameter_groups({runtime.training_target: runtime},
        {runtime.training_target} if collect_training_state else set())
    return sampled[runtime.training_target]


def _sample_population_parameter_groups(runtimes, collect_sides, population_context):
    """Batch frozen opponents by actual model identity and scatter back to their lanes."""
    if set(collect_sides) != set(population_context.active_roles):
        raise ValueError("Population collection must update exactly the active camp")
    env = next(iter(runtimes.values())).env
    requests: dict[int, dict[str, object]] = {}
    role_data = {}
    for role, runtime in runtimes.items():
        observation = runtime.observation
        shape = observation["actor_alive"].shape
        if shape[0] != population_context.lane_policy_index.numel():
            raise ValueError("Population lane assignment does not match the environment")
        flat_observation = _flatten_actor_observation(
            observation, runtime.model.variant.observation_keys, shape
        )
        alive = observation["actor_alive"].reshape(-1).to(torch.bool)
        lane = torch.arange(shape[0], device=env.device).reshape(
            shape[0], *([1] * (len(shape) - 1))
        ).expand(shape).reshape(-1)
        models = population_context.role_models(role)
        if models is None:
            models = (runtime.model,)
            assignments = torch.zeros(shape[0], dtype=torch.int64, device=env.device)
        else:
            assignments = population_context.lane_policy_index
        role_data[role] = (shape, flat_observation, alive)
        for model_index, model in enumerate(models):
            indices = (alive & (assignments[lane] == model_index)).nonzero(as_tuple=True)[0]
            request = requests.setdefault(id(model), {"model": model, "items": []})
            request["items"].append((role, indices))

    sampled = {}
    sigma = {}
    for role, runtime in runtimes.items():
        shape, _, _ = role_data[role]
        sampled[role] = (
            torch.zeros((*shape, 2), dtype=torch.float32, device=env.device),
            None,
            None,
        )
        sigma[role] = (torch.zeros(2, device=env.device), 0)

    for request in requests.values():
        model = request["model"]
        items = request["items"]
        keys = model.variant.actor_observation_keys
        total = sum(indices.numel() for _, indices in items)
        if not total:
            continue
        actor_observation = concat_observation_batches(
            [_observation_slice(role_data[role][1], keys, indices) for role, indices in items], keys)
        distribution = model.actor_distribution_from_observation(actor_observation)
        actions = distribution.rsample()
        logs = (
            distribution.log_prob(actions)
            if any(role in collect_sides for role, _ in items)
            else None
        )
        offset = 0
        for role, indices in items:
            count = indices.numel()
            if count:
                shape = role_data[role][0]
                flat_action = sampled[role][0].reshape(-1, 2)
                flat_action[indices] = actions[offset : offset + count]
                scale_sum, scale_count = sigma[role]
                sigma[role] = (
                    scale_sum + distribution.scale[offset : offset + count].sum(0),
                    scale_count + count,
                )
                if role in collect_sides:
                    assert logs is not None
                    current_log = sampled[role][2]
                    if current_log is None:
                        current_log = torch.zeros((*shape, 1), device=env.device)
                    current_log.reshape(-1, 1)[indices, 0] = logs[offset : offset + count]
                    sampled[role] = (sampled[role][0], sampled[role][1], current_log)
            offset += count

    values = _parameter_group_values(runtimes, collect_sides)
    for role in collect_sides:
        action, _, log = sampled[role]
        if log is None:
            log = torch.zeros_like(values[role])
        sampled[role] = (action, values[role], log)
    return sampled, sigma


def _sample_parameter_groups(runtimes, collect_sides, population_context=None):
    """Batch queries by actual parameter identity; only updating roles query values."""
    if population_context is not None:
        if any(getattr(runtime, "rule", None) is not None for runtime in runtimes.values()):
            raise ValueError("Population training does not support APF sources")
        return _sample_population_parameter_groups(runtimes, collect_sides, population_context)
    groups = {}
    sampled, sigma = {}, {}
    for role, runtime in runtimes.items():
        if getattr(runtime, "rule", None) is not None:
            if role in collect_sides:
                raise ValueError("Rule actions cannot enter PPO training samples")
            from .apf import apf_action_for_env
            sampled[role] = (apf_action_for_env(runtime.env, runtime.observation, config=runtime.rule), None, None)
        else:
            groups.setdefault(id(runtime.model), []).append(role)
    for roles in groups.values():
        model = runtimes[roles[0]].model
        keys = model.variant.observation_keys
        observations, living, credit, shapes = {}, {}, {}, {}
        for role in roles:
            obs = runtimes[role].observation
            shapes[role] = obs["actor_alive"].shape
            observations[role] = _flatten_actor_observation(obs, keys, shapes[role])
            living[role] = obs["actor_alive"].reshape(-1).nonzero(as_tuple=True)[0]
            credit[role] = obs["credit_member"].reshape(-1).nonzero(as_tuple=True)[0]
        actor_obs = concat_observation_batches([_observation_slice(observations[r], model.variant.actor_observation_keys, living[r]) for r in roles], model.variant.actor_observation_keys)
        total = sum(living[r].numel() for r in roles)
        distribution = model.actor_distribution_from_observation(actor_obs) if total else None
        actions = distribution.rsample() if total else None
        logs = distribution.log_prob(actions) if total and any(r in collect_sides for r in roles) else None
        value_roles = [r for r in roles if r in collect_sides]
        value_count = sum(credit[r].numel() for r in value_roles)
        values = model.value_from_observation(concat_observation_batches([_observation_slice(observations[r], model.variant.critic_observation_keys, credit[r]) for r in value_roles], model.variant.critic_observation_keys)) if value_count else None
        actor_offset = value_offset = 0
        for role in roles:
            runtime = runtimes[role]
            n, count = living[role].numel(), credit[role].numel()
            flat_action = torch.zeros((math.prod(shapes[role]), 2), device=runtime.env.device)
            if n:
                flat_action[living[role]] = actions[actor_offset:actor_offset+n]
                sigma[role] = (distribution.scale[actor_offset:actor_offset+n].sum(0), n)
            else:
                sigma[role] = (torch.zeros(2, device=runtime.env.device), 0)
            value = log = None
            if role in collect_sides:
                value = torch.zeros((flat_action.shape[0], 1), device=runtime.env.device)
                log = torch.zeros_like(value)
                if count:
                    value[credit[role]] = values[value_offset:value_offset+count]
                if n:
                    log[living[role], 0] = logs[actor_offset:actor_offset+n]
                value_offset += count
                value = value.reshape(*shapes[role], 1)
                log = log.reshape(*shapes[role], 1)
            sampled[role] = (flat_action.reshape(*shapes[role], 2), value, log)
            actor_offset += n
    return sampled, sigma


def _parameter_group_values(runtimes, collect_sides, observations=None):
    """Query selected credit rows by actual model identity; return owned role tensors."""
    groups, results = {}, {}
    for role, runtime in runtimes.items():
        if role not in collect_sides:
            continue
        if runtime.model.uses_centralized_critic:
            results[role] = _policy_value(
                runtime, runtime.observation if observations is None else observations[role]
            )
            continue
        groups.setdefault(id(runtime.model), []).append(role)
    for roles in groups.values():
        model = runtimes[roles[0]].model
        keys = model.variant.critic_observation_keys
        items, indices = [], {}
        for role in roles:
            runtime = runtimes[role]
            obs = runtime.observation if observations is None else observations[role]
            credit = obs['credit_member']
            index = credit.reshape(-1).nonzero(as_tuple=True)[0]
            indices[role] = index
            results[role] = torch.zeros((*credit.shape, 1), device=runtime.env.device)
            if index.numel():
                flat = _flatten_actor_observation(obs, keys, credit.shape)
                items.append({k: flat[k][index] for k in keys})
        if not items:
            continue
        values = model.value_from_observation(concat_observation_batches(items, keys))
        offset = 0
        for role in roles:
            index = indices[role]
            count = index.numel()
            results[role].reshape(-1, 1)[index] = values[offset:offset+count]
            offset += count
    return results


def _policy_value(
    runtime: GoalReachRuntime,
    observation: dict[str, torch.Tensor],
) -> torch.Tensor:
    observation_keys = runtime.model.variant.critic_observation_keys
    credit_member = observation.get(
        "credit_member", observation["actor_active"]
    ).to(torch.bool)
    if runtime.model.uses_centralized_critic:
        return runtime.model.value_from_grouped_observation(
            _observation_slice(observation, observation_keys, slice(None)),
            credit_member,
        )
    flat_observation = _flatten_actor_observation(
        observation, observation_keys, credit_member.shape
    )
    flat_credit = credit_member.reshape(-1)
    flat_value = torch.zeros(
        (flat_credit.numel(), 1), dtype=torch.float32, device=runtime.env.device
    )
    flat_value[flat_credit] = runtime.model.value_from_observation(
        {key: value[flat_credit] for key, value in flat_observation.items()}
    )
    return flat_value.reshape(*credit_member.shape, 1)


@torch.no_grad()
def collect_joint_goal_reach(
    runtimes: dict[str, GoalReachRuntime],
    rollout_length: int,
    collect_sides: set[str] | None = None,
    population_context=None,
) -> tuple[dict[str, TensorDict], dict[str, object]]:
    """Collect one shared match rollout for both policies without CPU-side arenas."""
    if set(runtimes) != set(runtimes["learner"].env.roles):
        raise ValueError("Joint training requires learner and defender runtimes")
    env = runtimes["learner"].env
    if any(runtime.env is not env for runtime in runtimes.values()) or not hasattr(env, "training_step_joint"):
        raise ValueError("Joint training requires one shared multi-agent environment")
    collect_sides = {role for role, runtime in runtimes.items() if getattr(runtime, "rule", None) is None} if collect_sides is None else set(collect_sides)
    if not collect_sides or not collect_sides.issubset(runtimes):
        raise ValueError("Joint rollout must update at least one configured side")
    if any(getattr(runtimes[role], "rule", None) is not None for role in collect_sides):
        raise ValueError("Rule actions cannot enter PPO training samples")
    if population_context is not None and set(collect_sides) != set(population_context.active_roles):
        raise ValueError("Population rollout active camp differs from collected sides")
    optimizer_config = env.config.goal_reach_trainer["optimizer"]
    storages: dict[str, dict[str, list[torch.Tensor]]] = {}
    terminal_bootstraps: dict[str, list[torch.Tensor]] = {}
    frames_by_side = {
        side: torch.zeros((), dtype=torch.int64, device=env.device)
        for side in collect_sides
    }
    reward_by_side = {
        side: torch.zeros((), dtype=torch.float32, device=env.device)
        for side in collect_sides
    }
    completed_return_by_side = {
        side: torch.zeros((), dtype=torch.float32, device=env.device)
        for side in collect_sides
    }
    for side in collect_sides:
        runtime = runtimes[side]
        storages[side] = {
            key: []
            for key in (
                *runtime.model.variant.observation_keys,
                "action",
                "old_log_probability",
                "value",
                "reward",
                "done",
                "terminated",
                "is_init",
                "actor_alive",
                "credit_member",
                "valid",
                *(("episode_decision_index",) if population_context is not None else ()),
            )
        }
        terminal_bootstraps[side] = []

    completed_match_facts: list[torch.Tensor] = []
    completed_match_events: list[torch.Tensor] = []
    completed_lengths: list[torch.Tensor] = []
    completed_durations: list[torch.Tensor] = []
    completed_utilities = []
    target_casualties = []
    explicit_end_reasons = []
    environment_steps = 0
    phase_profiler = _CollectionPhaseProfiler(env.device)
    reset_calls = 0
    reset_lanes = 0
    sampling_counts = dict(candidate_trials=0,candidate_accepts=0,layout_resamples=0)
    if env.device.type == "cuda":
        torch.cuda.synchronize(env.device)
    started = time.perf_counter()
    sigma_sums = {r: torch.zeros(2, device=env.device) for r in runtimes}
    sigma_counts = dict.fromkeys(runtimes, 0)
    for _ in range(rollout_length):
        pre_decision_index = None
        if population_context is not None:
            pre_decision_index = torch.div(
                env.physics_ticks,
                int(env.config.goal_reach["clock"]["physics_ticks_per_decision"]),
                rounding_mode="floor",
            )
        phase_started = phase_profiler.start()
        sampled, decision_sigma = _sample_parameter_groups(
            runtimes, collect_sides, population_context
        )
        for role, (scale_sum, count) in decision_sigma.items():
            sigma_sums[role] += scale_sum
            sigma_counts[role] += count
        phase_profiler.stop("policy", phase_started)
        phase_started = phase_profiler.start()
        joint = env.training_step_joint(
            sampled["learner"][0],
            sampled["defender"][0],
            target_action=sampled["target"][0] if "target" in sampled else None,
            build_observation=False,
        )
        phase_profiler.stop("environment_step", phase_started)
        phase_started = phase_profiler.start()
        transitions = {role: getattr(joint, role) for role in runtimes}
        truncated_values = {}
        truncated_scenes = None
        if env.bootstrap_on_time_limit:
            truncated_scenes = joint.learner.truncated.any(dim=1)
            if bool(truncated_scenes.any()):
                terminal_observations = env.training_observation_bundle(
                    collect_sides, truncated_scenes, consumer="critic")
                truncated_values = _parameter_group_values(
                    runtimes, collect_sides, terminal_observations
                )
        match_done = joint.learner.info["match_done"]
        match_outcome = joint.learner.info["match_outcome"]
        match_end_reason = joint.learner.info["match_end_reason"]
        completed_utilities.append(joint.learner.info["episode_defense_utility"]*match_done)
        explicit_end_reasons.append(torch.where(match_done, match_end_reason, 0))
        target_casualties.append(match_done & (match_end_reason == MATCH_END_REASON_CODES[MatchEndReason.TARGET_CASUALTY]))
        completed_match_facts.append(
            torch.stack(
                (
                    match_done,
                    match_done & (match_outcome == int(MatchOutcome.ATTACKER_WIN)),
                    match_done & (match_outcome == int(MatchOutcome.DEFENDER_WIN)),
                    match_done
                    & (match_end_reason == MATCH_END_REASON_CODES[MatchEndReason.BREACH]),
                    match_done
                    & (
                        match_end_reason
                        == MATCH_END_REASON_CODES[MatchEndReason.ELIMINATION]
                    ),
                    match_done
                    & (match_end_reason == MATCH_END_REASON_CODES[MatchEndReason.HORIZON]),
                    match_done
                    & (
                        match_end_reason
                        == MATCH_END_REASON_CODES[MatchEndReason.ATTACKER_CASUALTY]
                    ),
                    match_done
                    & (
                        match_end_reason
                        == MATCH_END_REASON_CODES[MatchEndReason.TARGET_CAPTURE]
                    ),
                    match_done & (match_end_reason == MATCH_END_REASON_CODES[MatchEndReason.DEFENDER_CASUALTY]),
                ),
                dim=-1,
            )
        )
        event_counts = torch.stack(
            [
                joint.learner.info.get(key, torch.zeros_like(match_done))
                .reshape(env.batch_size, -1)
                .sum(dim=1, dtype=torch.int32)
                for key in MATCH_EVENT_INFO_KEYS
            ],
            dim=-1,
        )
        primary_side = "learner"
        runtimes[primary_side].match_event_counts += event_counts
        completed_match_events.append(
            runtimes[primary_side].match_event_counts.to(torch.float32)
            * match_done.to(torch.float32).unsqueeze(-1)
        )
        runtimes[primary_side].match_event_counts.masked_fill_(match_done.unsqueeze(-1), 0)
        completed_lengths.append(
            torch.ceil(
                env.physics_ticks.to(torch.float32)
                / env.config.goal_reach["clock"]["physics_ticks_per_decision"]
            )
        )

        clock = env.config.goal_reach["clock"]
        dt = float(env.config.shared["clock"]["physics_step_s"])
        completed_durations.append(torch.stack((env.physics_ticks * dt, joint.learner.info["protected_time"],
                                                env.episode_pair_contacts), -1) * match_done[:, None])
        environment_steps += env.batch_size
        for side, runtime in runtimes.items():
            transition = transitions[side]
            done = transition.terminated | transition.truncated
            runtime.episode_returns.add_(transition.reward)
            if side not in collect_sides:
                runtime.episode_returns.masked_fill_(done,0.)
                continue
            observation = runtime.observation
            action, value, old_log_probability = sampled[side]
            assert value is not None and old_log_probability is not None
            transition = transitions[side]
            done = transition.terminated | transition.truncated
            actor_alive = observation.get(
                "actor_alive", torch.ones_like(done, dtype=torch.bool)
            ).to(torch.bool)
            credit_member = observation.get("credit_member", actor_alive).to(torch.bool)
            reward_by_side[side] += transition.reward.sum()
            completed_actor_returns = torch.where(
                done, runtime.episode_returns, torch.zeros_like(runtime.episode_returns)
            )
            if completed_actor_returns.ndim == 1:
                completed_team_returns = completed_actor_returns
            else:
                completed_team_returns = completed_actor_returns.flatten(1).sum(dim=1) / (
                    credit_member.flatten(1).sum(dim=1).clamp_min(1)
                )
            completed_return_by_side[side] += completed_team_returns.sum()
            runtime.episode_returns.masked_fill_(done, 0.0)
            storage = storages[side]
            for key in runtime.model.variant.observation_keys:
                storage[key].append(observation[key])
            storage["action"].append(action)
            storage["old_log_probability"].append(old_log_probability)
            storage["value"].append(value)
            storage["reward"].append(transition.reward.unsqueeze(-1))
            storage["done"].append(done.unsqueeze(-1))
            storage["terminated"].append(transition.terminated.unsqueeze(-1))
            storage["is_init"].append(runtime.is_init.unsqueeze(-1))
            storage["actor_alive"].append(actor_alive.unsqueeze(-1))
            storage["credit_member"].append(credit_member.unsqueeze(-1))
            storage["valid"].append(actor_alive.unsqueeze(-1))
            if population_context is not None:
                assert pre_decision_index is not None
                episode_index = pre_decision_index.reshape(
                    env.batch_size, *([1] * (actor_alive.ndim - 1))
                )
                storage["episode_decision_index"].append(
                    episode_index.expand_as(actor_alive).unsqueeze(-1).clone()
                )
            terminal_bootstrap = torch.zeros_like(value)
            if side in truncated_values:
                terminal_bootstrap[truncated_scenes] = truncated_values[side]
            terminal_bootstraps[side].append(terminal_bootstrap)
            frames_by_side[side] += actor_alive.sum()

        phase_profiler.stop("bookkeeping", phase_started)
        phase_started = phase_profiler.start()
        scene_done = env.done.clone()
        done_indices = torch.nonzero(scene_done, as_tuple=False).flatten()
        reset_lane_count = done_indices.numel()
        if reset_lane_count > 0:
            reset_calls += 1
            reset_lanes += reset_lane_count
            env.reset(
                scene_done,
                env_indices=done_indices,
                build_observation=False,
            )
            for key in sampling_counts:sampling_counts[key]+=env.reset_statistics.get(key,0)
            if population_context is not None:
                population_context.resample(done_indices)
        phase_profiler.stop("reset", phase_started)
        phase_started = phase_profiler.start()
        next_observations = env.training_observation_bundle(runtimes, record_neighbors=True)
        for side, runtime in runtimes.items():
            next_observation = next_observations[side]
            runtime.observation = next_observation
            next_credit = next_observation.get(
                "credit_member", next_observation["actor_active"]
            ).to(torch.bool)
            runtime.is_init = scene_done.reshape(
                env.batch_size, *([1] * (next_credit.ndim - 1))
            ) & next_credit
        phase_profiler.stop("observation_bundle", phase_started)

    phase_started = phase_profiler.start()
    rollouts: dict[str, TensorDict] = {}
    final_values = _parameter_group_values(runtimes, collect_sides)
    for side in collect_sides:
        runtime = runtimes[side]
        storage = storages[side]
        final_next_value = final_values[side]
        storage["next_value"] = _shifted_next_values(
            storage["value"],
            storage["done"],
            storage["terminated"],
            terminal_bootstraps[side],
            final_next_value,
        )
        rollout = TensorDict(
            _stack_rollout_storage(storage, runtime.model.variant),
            batch_size=[rollout_length, *storage["credit_member"][0].shape[:-1]],
            device=env.device,
        )
        storage.clear()
        terminal_bootstraps[side].clear()
        _attach_gae(rollout, optimizer_config)
        rollouts[side] = rollout

    _memory_sample(env, 'stack_gae', rollouts)

    match_facts = torch.stack(completed_match_facts, dim=0)
    match_events = torch.stack(completed_match_events, dim=0)
    lengths = torch.stack(completed_lengths, dim=0)
    totals = torch.cat(
        (
            match_facts.sum(
                dim=tuple(range(match_facts.ndim - 1)), dtype=torch.float32
            ),
            (match_facts[..., 0].to(lengths.dtype) * lengths).sum().reshape(1),
            match_events.sum(
                dim=tuple(range(match_events.ndim - 1)), dtype=torch.float32
            ),
        )
    )
    phase_profiler.stop("finalize", phase_started)
    phase_seconds = phase_profiler.synchronize_and_seconds()
    totals = totals.cpu().tolist()
    frames_by_side_values = {
        side: int(frames_by_side[side].item()) for side in collect_sides
    }
    for side, frames in frames_by_side_values.items():
        runtimes[side].agent_frames += frames
    elapsed = time.perf_counter() - started
    profiled_seconds = sum(phase_seconds.values())
    (
        matches,
        attacker_wins,
        defender_wins,
        breach_ends,
        elimination_ends,
        horizon_ends,
        attacker_casualty_ends,
        target_capture_ends,
        defender_casualty_ends,
        match_decisions,
        attacker_reach_events,
        attacker_capture_events,
        attacker_static_collision_events,
        attacker_vessel_collision_events,
        defender_static_collision_events,
        defender_vessel_collision_events,
        target_capture_events, target_static_collision_events, target_vessel_collision_events,
        *boundary_events,
    ) = totals
    return rollouts, {
        "scene_sampling": {k:int(v) for k,v in sampling_counts.items()},
        "sigma": {r:(sigma_sums[r]/max(1,sigma_counts[r])).cpu().tolist() for r in runtimes if getattr(runtimes[r], "rule", None) is None},
        "collect_seconds": elapsed,
        "phase_seconds": phase_seconds,
        "unattributed_seconds": max(0.0, elapsed - profiled_seconds),
        "reset_calls": reset_calls,
        "reset_lanes": reset_lanes,
        "reset_lane_fraction": reset_lanes / (env.batch_size * rollout_length),
        "matches": float(matches),
        "attacker_wins": float(attacker_wins),
        "defender_wins": float(defender_wins),
        "breach_ends": float(breach_ends),
        "elimination_ends": float(elimination_ends),
        "horizon_ends": float(horizon_ends),
        "truncation_ends": float((torch.stack(explicit_end_reasons) == MATCH_END_REASON_CODES[MatchEndReason.TRUNCATION]).sum()) if explicit_end_reasons else 0.,
        "defender_elimination_ends": float((torch.stack(explicit_end_reasons) == MATCH_END_REASON_CODES[MatchEndReason.DEFENDER_ELIMINATION]).sum()) if explicit_end_reasons else 0.,
        "attacker_casualty_ends": float(attacker_casualty_ends),
        "target_capture_ends": float(target_capture_ends),
        "target_casualty_ends": float(torch.stack(target_casualties).sum()),
        "completed_defense_return": float(torch.stack(completed_utilities).sum()),
        "defender_casualty_ends": float(defender_casualty_ends),
        "match_decisions": float(match_decisions),
        "completed_durations": torch.stack(completed_durations).sum((0, 1)).cpu().tolist(),
        "environment_steps": environment_steps,
        "environment_steps_per_second": environment_steps / elapsed,
        "target_capture_events": float(target_capture_events),
        "target_static_collision_events": float(target_static_collision_events),
        "target_vessel_collision_events": float(target_vessel_collision_events),
        **{key.removeprefix('event_')+'_events': value
           for key, value in zip(MATCH_EVENT_INFO_KEYS[9:], boundary_events)},
        "attacker_reach_events": float(attacker_reach_events),
        "attacker_capture_events": float(attacker_capture_events),
        "attacker_static_collision_events": float(attacker_static_collision_events),
        "attacker_vessel_collision_events": float(attacker_vessel_collision_events),
        "defender_static_collision_events": float(defender_static_collision_events),
        "defender_vessel_collision_events": float(defender_vessel_collision_events),
        "sides": {
            side: {
                "agent_frames": frames_by_side_values[side],
                "agent_frames_per_second": frames_by_side_values[side] / elapsed,
                "reward_sum": float(reward_by_side[side].item()),
                "completed_return_sum": float(completed_return_by_side[side].item()),
            }
            for side in collect_sides
        },
    }


def _shifted_next_values(
    values: list[torch.Tensor],
    dones: list[torch.Tensor],
    terminated: list[torch.Tensor],
    terminal_bootstraps: list[torch.Tensor],
    final_next_value: torch.Tensor,
) -> list[torch.Tensor]:
    if not values or not (len(values) == len(dones) == len(terminated) == len(terminal_bootstraps)):
        raise ValueError("Rollout bootstrap lists must be nonempty and aligned")
    result = []
    for index in range(len(values)):
        candidate = values[index + 1] if index + 1 < len(values) else final_next_value
        item = torch.where(dones[index], terminal_bootstraps[index], candidate)
        result.append(torch.where(terminated[index], torch.zeros_like(item), item))
    return result


def _attach_gae(rollout: TensorDict, optimizer_config: dict[str, Any]) -> None:
    gamma = float(optimizer_config["gamma"])
    lmbda = float(optimizer_config["gae_lambda"])
    reward = rollout["reward"]
    value = rollout["value"]
    next_value = rollout["next_value"]
    done = rollout["done"].to(torch.bool)
    terminated = rollout["terminated"].to(torch.bool)
    credit_member = rollout["credit_member"].to(torch.bool)
    advantage = torch.zeros_like(value)
    running = torch.zeros_like(value[0])
    for index in range(value.shape[0] - 1, -1, -1):
        delta = (
            reward[index]
            + gamma * (~terminated[index]).to(value.dtype) * next_value[index]
            - value[index]
        )
        running = delta + gamma * lmbda * (~done[index]).to(value.dtype) * running
        running = torch.where(
            credit_member[index], running, torch.zeros_like(running)
        )
        advantage[index] = running
    rollout["advantage"] = advantage
    rollout["value_target"] = advantage + value


def _ppo_objective(
    runtime: GoalReachRuntime,
    flat: dict[str, torch.Tensor],
    index: torch.Tensor,
    optimizer_config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    observation = _observation_slice(
        flat, runtime.model.variant.observation_keys, index
    )
    actor_alive = flat["actor_alive"][index].squeeze(-1).to(torch.bool)
    distribution = runtime.model.actor_distribution_from_observation(
        {key: value[actor_alive] for key, value in observation.items()}
    )
    value = runtime.model.value_from_observation(observation)
    return _ppo_losses(
        distribution,
        value,
        flat["action"][index][actor_alive],
        flat["old_log_probability"][index][actor_alive],
        flat["advantage"][index][actor_alive],
        flat["value_target"][index],
        optimizer_config,
    )


def _ppo_group_objective(
    runtime: GoalReachRuntime,
    grouped: dict[str, torch.Tensor],
    index: torch.Tensor,
    optimizer_config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = {key: value[index] for key, value in grouped.items()}
    actor_alive = selected["actor_alive"].squeeze(-1).to(torch.bool)
    credit_member = selected["credit_member"].squeeze(-1).to(torch.bool)
    actor_shape = actor_alive.shape
    actor_count = actor_alive.numel()
    flat_alive = actor_alive.reshape(-1)
    observation = _observation_slice(
        selected, runtime.model.variant.observation_keys, slice(None)
    )
    flat_observation = _flatten_actor_observation(
        observation, runtime.model.variant.observation_keys, actor_shape
    )
    distribution = runtime.model.actor_distribution_from_observation(
        {key: value[flat_alive] for key, value in flat_observation.items()}
    )
    value = runtime.model.value_from_grouped_observation(
        observation, credit_member
    )[credit_member]

    def active_items(name: str) -> torch.Tensor:
        value_tensor = selected[name]
        return value_tensor.reshape(
            actor_count, *value_tensor.shape[len(actor_shape) :]
        )[flat_alive]

    loss, metrics = _ppo_losses(
        distribution,
        value,
        active_items("action"),
        active_items("old_log_probability"),
        active_items("advantage"),
        selected["value_target"][credit_member],
        optimizer_config,
    )
    return loss, metrics, credit_member.sum().to(torch.float32)


def _policy_entropy(distribution, coefficient):
    # ClippedNormal entropy is analytic: monitoring with bonus Off needs no graph or RNG.
    with torch.set_grad_enabled(torch.is_grad_enabled() and coefficient != 0):
        return (distribution.entropy() if isinstance(distribution, (ClippedNormal, torch.distributions.Categorical))
                else -distribution.log_prob(distribution.rsample()))


def _ppo_losses(
    distribution: object,
    value: torch.Tensor,
    action: torch.Tensor,
    old_log_probability: torch.Tensor,
    advantage: torch.Tensor,
    value_target: torch.Tensor,
    optimizer_config: dict[str, Any],
    policy_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    new_log_probability = distribution.log_prob(action).unsqueeze(-1)
    ratio = (new_log_probability - old_log_probability).exp()
    epsilon = float(optimizer_config["clip_epsilon"])
    policy_terms = -torch.minimum(
        ratio * advantage,
        ratio.clamp(1.0 - epsilon, 1.0 + epsilon) * advantage,
    )
    actor_count = max(1, policy_terms.numel())
    if policy_weight is None:
        policy_loss = policy_terms.sum() / actor_count
    else:
        weights = policy_weight.to(dtype=policy_terms.dtype).expand_as(policy_terms)
        if not torch.isfinite(weights).all() or bool((weights <= 0).any()):
            raise ValueError("Policy weights must be finite and positive")
        policy_loss = (policy_terms * weights).sum() / weights.sum()
    value_loss = 0.5 * (value - value_target).square().mean()
    entropy = _policy_entropy(distribution, float(optimizer_config['entropy_coefficient'])).sum() / actor_count
    loss = (
        policy_loss
        + float(optimizer_config["value_coefficient"]) * value_loss
        - float(optimizer_config["entropy_coefficient"]) * entropy
    )
    with torch.no_grad():
        log_ratio = new_log_probability - old_log_probability
        approximate_kl = ((log_ratio.exp() - 1.0) - log_ratio).sum() / actor_count
        clip_fraction = (
            ((ratio - 1.0).abs() > epsilon).to(torch.float32).sum()
            / actor_count
        )
    return loss, torch.stack(
        (
            policy_loss.detach(),
            value_loss.detach(),
            entropy.detach(),
            approximate_kl.detach(),
            clip_fraction.detach(),
        )
    )


PPO_SAMPLE_CONTRACT = 'parameter_group_valid_frames_v1'

# Shared defaults for the continuous task and its categorical scene adapter.
PPO_DEFAULTS = dict(parallel_environments=1024, rollout_decisions=32,
    minibatch_agent_frames=4096, optimization_epochs=5,
    ppo_update_mode="fixed_optimizer_steps", optimizer_steps_per_update=40,
    learning_rate=5.0e-4, gamma=1.0, gae_lambda=0.99,
    bootstrap_on_time_limit=False, clip_epsilon=0.2, value_coefficient=0.5,
    entropy_coefficient=0.01, maximum_gradient_norm=0.5)


def ppo_epoch_partitions(total, batches, *, device, generator=None):
    """Partition every sample once; the optimizer budget determines batch size."""
    if batches < 1 or total < batches:
        raise ValueError("PPO needs at least one sample per optimizer batch")
    rng_device = generator.device if generator is not None else device
    order = torch.randperm(total, device=rng_device, generator=generator).to(device)
    return torch.tensor_split(order, batches)


def _prepare_parameter_group_samples(group, runtimes, rollouts):
    """Validate one complete PG before any optimizer mutation; keep observation views."""
    roles = tuple(r for r in runtimes if runtimes[r].model is group.model)
    if not roles or group.optimizer is None:
        raise ValueError('Parameter group has no trainable rollout')
    cfg = runtimes[roles[0]].env.config.goal_reach_trainer
    epochs = int(cfg['rollout']['optimization_epochs'])
    steps = int(cfg['rollout']['optimizer_steps_per_update'])
    if epochs <= 0 or steps < epochs or steps % epochs:
        raise ValueError('Optimizer steps must be a positive multiple of epochs')
    flats, credit, actors = {}, {}, {}
    checks = []
    for role in roles:
        if role not in rollouts:
            raise ValueError('Missing parameter group rollout: '+role)
        rollout = rollouts[role]
        ndim, count = len(rollout.batch_size), math.prod(rollout.batch_size)
        flat = {k:v.reshape(count,*v.shape[ndim:]) for k,v in rollout.items()}
        required = set(group.model.variant.observation_keys) | {'credit_member','actor_alive','advantage','action','old_log_probability','value_target'}
        if required-flat.keys():
            raise ValueError(f'{role}: Missing rollout fields: {sorted(required-flat.keys())}')
        for key in ('credit_member','actor_alive','advantage','old_log_probability','value_target'):
            if flat[key].shape != (count,1):
                raise ValueError('Invalid rollout shape: '+key)
        if flat['action'].shape != (count,2):
            raise ValueError('Invalid rollout action shape')
        if any(flat[k].dtype != torch.bool for k in ('credit_member','actor_alive')):
            raise ValueError('Invalid rollout membership mask dtype')
        pairs = set()
        for keys in (group.model.variant.actor_observation_keys, group.model.variant.critic_observation_keys):
            for buffer_name in group.model.variant.entity_sets:
                prefix = entity_tensor_prefix(buffer_name)
                entity_key = f'{prefix}_critic_entities' if f'{prefix}_critic_entities' in keys else f'{prefix}_entities'
                mask_key = f'{prefix}_critic_mask' if f'{prefix}_critic_mask' in keys else f'{prefix}_mask'
                if entity_key in keys or mask_key in keys:
                    pairs.add((entity_key,mask_key))
        variable_keys = {key for pair in pairs if pair[0] == 'obstacle_entities' for key in pair}
        for key in group.model.variant.observation_keys:
            expected = runtimes[role].observation[key].shape[2:]
            actual = flat[key].shape[1:]
            # Obstacle token count may vary across owned rollout snapshots.
            if key in variable_keys:
                valid = len(actual)==len(expected) and (len(actual)<2 or actual[-1]==expected[-1])
            else:
                valid = actual==expected
            if not valid:
                raise ValueError(f'{role}: Invalid rollout observation shape: {key}, actual {tuple(flat[key].shape)}, expected {(count,*expected)} (obstacle K may vary)')
        for entity_key,mask_key in sorted(pairs):
            if entity_key not in flat or mask_key not in flat:
                raise ValueError(f'{role}: Missing rollout entity/mask pair: {entity_key}, {mask_key}')
            entities,mask = flat[entity_key],flat[mask_key]
            if entities.ndim != 3 or mask.shape != entities.shape[:2] or mask.dtype != torch.bool:
                raise ValueError(f'{role}: Invalid rollout entity/mask pair: {entity_key} actual {tuple(entities.shape)}, expected [N,K,F]; {mask_key} actual {tuple(mask.shape)} {mask.dtype}, expected {tuple(entities.shape[:2])} torch.bool')
        member = flat['credit_member'].squeeze(-1).bool()
        active = member & flat['actor_alive'].squeeze(-1).bool()
        credit[role] = member.nonzero(as_tuple=True)[0]
        actors[role] = active.nonzero(as_tuple=True)[0]
        member_index, active_index = credit[role], actors[role]
        for key in set(group.model.variant.critic_observation_keys) | {'value_target'}:
            checks.append(torch.isfinite(flat[key][member_index]).all())
        for key in set(group.model.variant.actor_observation_keys) | {'action','old_log_probability','advantage'}:
            checks.append(torch.isfinite(flat[key][active_index]).all())
        flat['actor_alive'] = active[:,None]
        if 'episode_decision_index' in flat:
            index = flat['episode_decision_index']
            if index.shape != (count,1):raise ValueError('Invalid episode decision index shape')
            checks.extend((torch.isfinite(index[active_index]).all(),(index[active_index]>=0).all()))
            flat['policy_weight'] = torch.pow(torch.full_like(index,float(cfg['optimizer']['gamma']),dtype=torch.float32),index.float())
        else:
            flat['policy_weight'] = torch.ones_like(flat['advantage'])
        weights = flat['policy_weight'][active_index]
        checks.extend((torch.isfinite(weights).all(),(weights>=0).all()))
        flats[role] = flat
    if sum(i.numel() for i in credit.values()) < steps//epochs:
        raise ValueError('Insufficient credit samples for the optimizer step budget')
    if not bool(torch.stack(checks).all()):
        raise ValueError('Nonfinite or negative valid PPO input')
    # One set of weighted population moments across all roles of this PG.
    total = sum(f['policy_weight'][actors[r]].sum() for r,f in flats.items())
    denominator = torch.where(total>0,total,1.)
    center = sum((f['policy_weight'][actors[r]]*f['advantage'][actors[r]]).sum() for r,f in flats.items())/denominator
    variance = sum((f['policy_weight'][actors[r]]*(f['advantage'][actors[r]]-center).square()).sum() for r,f in flats.items())/denominator
    for r,f in flats.items():
        raw = f['advantage']
        normalized = torch.zeros_like(raw)
        index = actors[r]
        normalized[index] = torch.where(total>0,(raw[index]-center)/variance.sqrt().clamp_min(1e-8),0.)
        f['advantage'] = normalized
    return roles, flats, credit, epochs, steps


def validate_parameter_group_rollouts(groups, runtimes, rollouts):
    """Preflight every selected group before the first group can step."""
    return {group.key:_prepare_parameter_group_samples(group,runtimes,rollouts) for group in groups}


def _parameter_group_table(model, flats, indices):
    """Owned credit rows in original role/index order, alive rows filtered only at Actor input."""
    roles = tuple(flats)
    fields = ('action', 'old_log_probability', 'advantage', 'value_target', 'actor_alive')
    keys = set(model.variant.observation_keys) | set(fields) | {'policy_weight'}
    items = [{k: flats[r][k][indices[r]] for k in keys if k in flats[r]}
             for r in roles if indices[r].numel()]
    table = concat_observation_batches(items, model.variant.observation_keys)
    for key in fields:
        table[key] = torch.cat([item[key] for item in items])
    table['policy_weight'] = torch.cat([item['policy_weight'] if 'policy_weight' in item else torch.ones_like(item['advantage']) for item in items])
    table['role'] = torch.cat([indices[r].new_full((indices[r].numel(),), n) for n, r in enumerate(roles)])
    return table


def _backward_parameter_group_batch(model, table, pool, opt):
    """One complete logical minibatch, one Actor/Critic forward and one backward."""
    batch = {k: v[pool] for k, v in table.items()}
    alive = batch['actor_alive'].squeeze(-1).nonzero(as_tuple=True)[0]
    values = model.value_from_observation({k: batch[k] for k in model.variant.critic_observation_keys})
    value_terms = .5 * (values - batch['value_target']).square().squeeze(-1)
    # Per-credit-row detached numerators preserve role reporting without more network queries.
    terms_by_row = values.new_zeros((pool.numel(), 5))
    terms_by_row[:, 1] = value_terms.detach()
    loss = float(opt['value_coefficient']) * value_terms.mean()
    log_error = values.new_zeros(())
    if alive.numel():
        distribution = model.actor_distribution_from_observation({k: batch[k][alive] for k in model.variant.actor_observation_keys})
        difference = distribution.log_prob(batch['action'][alive]).unsqueeze(-1) - batch['old_log_probability'][alive]
        ratio = difference.exp()
        epsilon = float(opt['clip_epsilon'])
        policy_terms = -torch.minimum(ratio * batch['advantage'][alive],
                                     ratio.clamp(1-epsilon, 1+epsilon) * batch['advantage'][alive])
        weights = batch['policy_weight'][alive]
        policy_terms = (policy_terms * weights).squeeze(-1)
        entropy = _policy_entropy(distribution, float(opt['entropy_coefficient']))
        weight_sum = weights.sum()
        loss = loss + policy_terms.sum() / torch.where(weight_sum > 0, weight_sum, 1.) - float(opt['entropy_coefficient']) * entropy.mean()
        terms_by_row[alive, 0] = policy_terms.detach()
        terms_by_row[alive, 2] = entropy.detach()
        terms_by_row[alive, 3] = ((ratio - 1) - difference).detach().squeeze(-1)
        terms_by_row[alive, 4] = ((ratio - 1).abs() > epsilon).float().squeeze(-1)
        log_error = difference.detach().abs().max()
    loss.backward()
    return terms_by_row, log_error


def update_parameter_group(group, runtimes, rollouts, *, prepared=None):
    roles,flats,credit_indices,epochs,steps = prepared if prepared is not None else _prepare_parameter_group_samples(group,runtimes,rollouts)
    runtime=runtimes[roles[0]];opt=runtime.env.config.goal_reach_trainer['optimizer']
    _memory_sample(runtime.env,'before_optimize_'+roles[0])
    batches=steps//epochs;model=group.model
    before={name:value.detach().clone() for name,value in model.named_parameters()}
    metric_sums={r:torch.zeros(6,device=runtime.env.device) for r in roles}
    role_minibatches={r:0 for r in roles};log_error=torch.zeros((),device=runtime.env.device)
    total=sum(i.numel() for i in credit_indices.values())
    table = _parameter_group_table(model, flats, credit_indices)
    for epoch in range(epochs):
        partitions=ppo_epoch_partitions(total,batches,device=runtime.env.device)
        for j,pool in enumerate(partitions):
            group.optimizer.zero_grad(set_to_none=True)
            metrics,error=_backward_parameter_group_batch(model,table,pool,opt)
            if epoch==0 and j==0:log_error=error
            norm=nn.utils.clip_grad_norm_(model.parameters(),float(opt['maximum_gradient_norm']))
            group.optimizer.step()
            batch_roles = table['role'][pool]
            for role_index, r in enumerate(roles):
                member = batch_roles == role_index
                # Historical field: role participation, not independent backward calls.
                role_minibatches[r] += member.any()
                metric_sums[r] += torch.cat(((metrics * member[:, None]).sum(0), norm.detach().reshape(1)))
    group.update_index+=1;group.optimizer_steps+=steps
    stats=_parameter_statistics(model,before);results={}
    _memory_sample(runtime.env,'optimized_'+roles[0])
    with torch.no_grad():
        predictions = model.value_from_observation({k: table[k] for k in model.variant.critic_observation_keys})
        offset = 0
        for role in roles:
            flat=flats[role];index=credit_indices[role];n=index.numel()
            actor_count=int(flat['actor_alive'][index].sum())
            weights=flat['policy_weight'][index][flat['actor_alive'][index]].sum()
            denominators=torch.stack((weights*epochs,weights.new_tensor(n*epochs),weights.new_tensor(actor_count*epochs),weights.new_tensor(actor_count*epochs),weights.new_tensor(actor_count*epochs),weights.new_tensor(steps)))
            denominators=torch.where(denominators>0,denominators,1.)
            averages=metric_sums[role]/denominators
            explained=weights.new_zeros(())
            if n:
                target=flat['value_target'][index];variance=target.var(unbiased=False)
                explained=torch.where(variance>1e-12,1-(target-predictions[offset:offset+n]).var(unbiased=False)/variance.clamp_min(1e-12),variance.new_zeros(()))
            offset += n
            runtimes[role].update_index=group.update_index
            result=_summarize(runtimes[role],before,averages*steps,steps,int(role_minibatches[role]),total/batches,log_error,explained,stats)
            result.actor_valid_frames=actor_count;result.critic_valid_frames=n
            if not actor_count:
                result.policy_loss=result.entropy=result.approximate_kl=result.clip_fraction=None
            if not n:result.value_loss=result.explained_variance=None
            results[role]=result
    _memory_sample(runtime.env,'post_critic_'+roles[0])
    return results


def update_goal_reach(runtime: GoalReachRuntime, rollout: TensorDict) -> UpdateMetrics:
    if runtime.optimizer is None:
        raise RuntimeError("PPO update requires an optimizer for the active side")
    config = runtime.env.config.goal_reach_trainer
    optimizer_config = config["optimizer"]
    batch_dimensions = len(rollout.batch_size)
    all_items = math.prod(rollout.batch_size)
    flat_all = {
        key: value.reshape(all_items, *value.shape[batch_dimensions:])
        for key, value in rollout.items()
    }
    credit_member = flat_all["credit_member"].squeeze(-1).to(torch.bool)
    flat = {key: value[credit_member] for key, value in flat_all.items()}
    total = int(credit_member.sum().item())
    if total == 0:
        raise RuntimeError("PPO rollout contains no team credit frames")
    centralized = runtime.model.uses_centralized_critic
    grouped: dict[str, torch.Tensor] | None = None
    training_units = total
    actor_capacity = 1
    if centralized:
        if len(rollout.batch_size) != 3:
            raise RuntimeError(
                "Centralized Critic rollout must have time, scene, and patrol axes"
            )
        time_count, scene_count, actor_capacity = (
            int(value) for value in rollout.batch_size
        )
        group_count = time_count * scene_count
        grouped_all = {
            key: value.reshape(
                group_count, actor_capacity, *value.shape[3:]
            )
            for key, value in rollout.items()
        }
        group_valid = (
            grouped_all["credit_member"]
            .squeeze(-1)
            .to(torch.bool)
            .any(dim=1)
        )
        grouped = {key: value[group_valid] for key, value in grouped_all.items()}
        training_units = int(group_valid.sum().item())
        if training_units == 0:
            raise RuntimeError("MAPPO rollout contains no team credit groups")

    rollout_config = config["rollout"]
    update_mode = str(rollout_config.get("update_mode", "fixed_minibatch"))
    if update_mode not in {"fixed_minibatch", "fixed_optimizer_steps"}:
        raise ValueError(f"Unknown PPO update mode: {update_mode}")
    optimization_epochs = int(rollout_config["optimization_epochs"])
    if update_mode == "fixed_optimizer_steps":
        target_optimizer_steps = int(rollout_config["optimizer_steps_per_update"])
        if target_optimizer_steps < optimization_epochs:
            raise ValueError(
                "optimizer_steps_per_update must be at least optimization_epochs"
            )
        base_steps, extra_step_epochs = divmod(
            target_optimizer_steps, optimization_epochs
        )
        # The optimizer-step budget defines the PPO minibatches.  Process each
        # resulting minibatch in one pass; its size therefore scales
        # automatically with rollout length, scene count, and Actor count.
        microbatch_units = math.ceil(training_units / base_steps)
        microbatch = min(
            total,
            microbatch_units * actor_capacity if centralized else microbatch_units,
        )
    else:
        microbatch = min(
            int(rollout_config["minibatch_agent_frames_initial"]), total
        )
        microbatch_units = (
            max(1, microbatch // actor_capacity) if centralized else microbatch
        )
    if centralized:
        assert grouped is not None
        grouped_alive = grouped["actor_alive"].to(torch.bool)
        active_advantage = grouped["advantage"][grouped_alive]
        mean = active_advantage.mean()
        standard_deviation = active_advantage.std(unbiased=False).clamp_min(1.0e-8)
        grouped["advantage"] = torch.where(
            grouped_alive,
            (grouped["advantage"] - mean) / standard_deviation,
            grouped["advantage"],
        )
    else:
        flat_alive = flat["actor_alive"].to(torch.bool)
        advantage = flat["advantage"][flat_alive]
        normalized = (advantage - advantage.mean()) / advantage.std(
            unbiased=False
        ).clamp_min(1.0e-8)
        flat["advantage"][flat_alive] = normalized
    log_error = _log_probability_error(runtime.model, flat, microbatch)
    before = {name: value.detach().clone() for name, value in runtime.model.named_parameters()}
    metric_sum = torch.zeros(6, dtype=torch.float32, device=runtime.env.device)
    minibatch_count = 0
    microbatch_count = 0
    for epoch in range(optimization_epochs):
        permutation = torch.randperm(training_units, device=runtime.env.device)
        if update_mode == "fixed_minibatch":
            effective_batches = tuple(
                permutation[start : start + microbatch_units]
                for start in range(0, training_units, microbatch_units)
            )
        else:
            effective_minibatches_per_epoch = base_steps + (
                1 if epoch < extra_step_epochs else 0
            )
            if training_units < effective_minibatches_per_epoch:
                raise RuntimeError(
                    "PPO rollout has fewer training units than requested optimizer steps"
                )
            effective_batches = torch.tensor_split(
                permutation, effective_minibatches_per_epoch
            )

        for effective_index in effective_batches:
            effective_size = int(effective_index.numel())
            effective_frames = (
                grouped["credit_member"][effective_index].sum().to(torch.float32)
                if centralized and grouped is not None
                else effective_size
            )
            runtime.optimizer.zero_grad(set_to_none=True)
            objective_metrics = torch.zeros(
                5, dtype=torch.float32, device=runtime.env.device
            )
            for start in range(0, effective_size, microbatch_units):
                index = effective_index[start : start + microbatch_units]
                if centralized:
                    assert grouped is not None
                    loss, item_metrics, item_frames = _ppo_group_objective(
                        runtime, grouped, index, optimizer_config
                    )
                else:
                    loss, item_metrics = _ppo_objective(
                        runtime, flat, index, optimizer_config
                    )
                    item_frames = int(index.numel())
                weight = item_frames / effective_frames
                (loss * weight).backward()
                objective_metrics += item_metrics * weight
                microbatch_count += 1
            gradient_norm = nn.utils.clip_grad_norm_(
                runtime.model.parameters(),
                float(optimizer_config["maximum_gradient_norm"]),
            )
            runtime.optimizer.step()
            metric_sum += torch.cat(
                (objective_metrics, gradient_norm.detach().reshape(1))
            )
            minibatch_count += 1
    explained_variance = (
        _explained_variance_grouped(
            runtime.model, grouped, microbatch_units
        )
        if centralized and grouped is not None
        else _explained_variance(runtime.model, flat, microbatch)
    )
    runtime.update_index += 1
    return _summarize(
        runtime,
        before,
        metric_sum,
        minibatch_count,
        microbatch_count,
        total * optimization_epochs / minibatch_count,
        log_error,
        explained_variance,
    )


@torch.no_grad()
def _explained_variance(
    model: ActorCritic, flat: dict[str, torch.Tensor], chunk_size: int
) -> torch.Tensor:
    predicted = torch.cat(
        [
            model.value_from_observation(
                _observation_slice(
                    flat,
                    model.variant.observation_keys,
                    slice(start, start + chunk_size),
                )
            )
            for start in range(0, flat["value_target"].shape[0], chunk_size)
        ]
    )
    target = flat["value_target"]
    variance = target.var(unbiased=False)
    explained = 1.0 - (target - predicted).var(unbiased=False) / variance.clamp_min(1.0e-12)
    return torch.where(variance <= 1.0e-12, torch.zeros_like(explained), explained)


@torch.no_grad()
def _explained_variance_grouped(
    model: ActorCritic,
    grouped: dict[str, torch.Tensor],
    chunk_size: int,
) -> torch.Tensor:
    predicted_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    for start in range(0, grouped["value_target"].shape[0], chunk_size):
        item = slice(start, start + chunk_size)
        credit_member = (
            grouped["credit_member"][item].squeeze(-1).to(torch.bool)
        )
        observation = _observation_slice(
            grouped, model.variant.observation_keys, item
        )
        predicted_chunks.append(
            model.value_from_grouped_observation(observation, credit_member)[
                credit_member
            ]
        )
        target_chunks.append(grouped["value_target"][item][credit_member])
    predicted = torch.cat(predicted_chunks)
    target = torch.cat(target_chunks)
    variance = target.var(unbiased=False)
    explained = 1.0 - (target - predicted).var(unbiased=False) / variance.clamp_min(
        1.0e-12
    )
    return torch.where(
        variance <= 1.0e-12, torch.zeros_like(explained), explained
    )


@torch.no_grad()
def _log_probability_error(
    model: ActorCritic, flat: dict[str, torch.Tensor], chunk_size: int
) -> torch.Tensor:
    actor_alive = flat["actor_alive"].squeeze(-1).to(torch.bool)
    flat = {key: value[actor_alive] for key, value in flat.items()}
    maximum = torch.zeros((), dtype=torch.float32, device=flat["action"].device)
    if flat["action"].shape[0] == 0:
        return maximum
    for start in range(0, flat["action"].shape[0], chunk_size):
        distribution = model.actor_distribution_from_observation(
            _observation_slice(
                flat,
                model.variant.observation_keys,
                slice(start, start + chunk_size),
            )
        )
        actual = distribution.log_prob(flat["action"][start : start + chunk_size]).unsqueeze(-1)
        expected = flat["old_log_probability"][start : start + chunk_size]
        maximum = torch.maximum(maximum, (actual - expected).abs().max())
    return maximum


def _parameter_statistics(model, before):
    named_parameters = list(model.named_parameters())
    actor_delta = torch.stack(
        [
            (value.detach() - before[name]).abs().max()
            for name, value in named_parameters
            if name.startswith("actor")
        ]
    ).max()
    critic_delta = torch.stack(
        [
            (value.detach() - before[name]).abs().max()
            for name, value in named_parameters
            if not name.startswith("actor")
        ]
    ).max()
    parameter_delta = torch.maximum(actor_delta, critic_delta)
    parameters_finite = torch.stack([torch.isfinite(value).all() for _,value in named_parameters]).all()
    return actor_delta, critic_delta, parameter_delta, parameters_finite


def _summarize(
    runtime: GoalReachRuntime,
    before: dict[str, torch.Tensor],
    metric_sum: torch.Tensor,
    minibatch_count: int,
    microbatch_count: int,
    effective_minibatch_agent_frames: float,
    log_error: torch.Tensor,
    explained_variance: torch.Tensor,
    parameter_statistics=None,
) -> UpdateMetrics:
    if minibatch_count <= 0:
        raise RuntimeError("PPO update produced no minibatches")
    averages = metric_sum / minibatch_count
    actor_delta, critic_delta, parameter_delta, parameters_finite = (
        parameter_statistics if parameter_statistics is not None else _parameter_statistics(runtime.model, before))
    finite = torch.stack((torch.isfinite(averages).all(), torch.isfinite(log_error),
                          torch.isfinite(explained_variance), parameters_finite)).all()
    summary = torch.cat(
        (
            averages,
            torch.stack(
                (
                    parameter_delta,
                    log_error,
                    finite.to(torch.float32),
                    actor_delta,
                    critic_delta,
                    explained_variance,
                )
            ),
        )
    ).detach().cpu().tolist()
    peak = (
        torch.cuda.max_memory_reserved(runtime.env.device) / 1024**3
        if runtime.env.device.type == "cuda"
        else 0.0
    )
    return UpdateMetrics(
        policy_loss=summary[0],
        value_loss=summary[1],
        entropy=summary[2],
        approximate_kl=summary[3],
        clip_fraction=summary[4],
        gradient_norm=summary[5],
        parameter_delta_max=summary[6],
        log_probability_error_max=summary[7],
        finite=bool(summary[8]),
        peak_reserved_gib=float(peak),
        actor_parameter_delta_max=summary[9],
        critic_parameter_delta_max=summary[10],
        explained_variance=summary[11],
        optimizer_steps=minibatch_count,
        microbatch_count=microbatch_count,
        effective_minibatch_agent_frames=effective_minibatch_agent_frames,
    )
