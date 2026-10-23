from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence

import torch

from .actor_variants import CRAFT_SPEED_RATIO_BOUNDS, actor_variant, UNIFIED_MATCH_TARGET_ARCHITECTURE_IDS, ROLE_ORDER, ENTITY_SET_ORDER, policy_roles, ROLE_DESCRIPTORS, CAPTURE_RELATIONS
from .dynamics import AuvState
from .goal_reach_env import merge_reset_observation, GoalReachEnv, GoalReachStep, JointGoalReachStep
from .match_metrics import (
    MATCH_END_REASON_CODES,
    OUTCOME_TIME_COEFFICIENT,
    REWARD_DESCRIPTIONS,
    MatchEndReason,
    MatchEvent,
    MatchOutcome,
)
from .scene_sampling import ActorSpawnBatch, SceneBatch, deterministic_uniform

CONTACT_TOLERANCE = 1.0e-5

PATROL_SPEED_RATIO_BOUNDS = CRAFT_SPEED_RATIO_BOUNDS
PATROL_SPEED_RATIO_OBSERVATION_BOUNDS = (1.0, 5.0)


def _actor_view(state: AuvState, index: int) -> AuvState:
    return AuvState(**{name: value[:, index] for name, value in state.__dict__.items()})


def compact_visible_indices(visible):
    """Stable compaction with one batch-level dynamic-width synchronization."""
    if visible.ndim != 2 or visible.dtype != torch.bool:
        raise ValueError('Expected bool [batch, candidates]')
    b, n = visible.shape
    if b == 0 or n == 0:
        return (torch.empty((b, 0), dtype=torch.long, device=visible.device),
                torch.empty((b, 0), dtype=torch.bool, device=visible.device))
    ordinal = visible.to(torch.long).cumsum(-1) - 1
    width = int((ordinal[:, -1] + 1).max().item())
    destination = torch.where(visible, ordinal, width)
    source = torch.where(visible, torch.arange(1, n + 1, device=visible.device), 0)
    packed = torch.zeros((b, width + 1), dtype=torch.long, device=visible.device)
    packed.scatter_reduce_(1, destination, source, reduce='amax', include_self=True)
    one_based = packed[:, :width]
    return (one_based - 1).clamp_min(0), one_based > 0


class MultiGoalReachEnv(GoalReachEnv):
    """One batched two-team match trunk with shared policies and team credit."""

    state_schema = "fixed_world_local15_first10_multi_v1"

    def __init__(
        self,
        *args: object,
        learner_count_range: Sequence[int],
        defender_count_range: Sequence[int] | None = None,
        learner_speed_ratio_range: Sequence[float] = (1.0, 1.0),
        terminate_on_any_casualty: bool = True,
        terminate_on_defender_elimination: bool = False,
        apply_sampling_time_limit: bool = True,
        reward_objective: str = "outcome",
        **kwargs: object,
    ) -> None:
        learner_count_min, learner_count_max = (
            int(value) for value in learner_count_range
        )
        if not 1 <= learner_count_min <= learner_count_max <= 10:
            raise ValueError("Multi GoalReach learner count range must be inside [1,10]")
        learner_speed_ratio_min, learner_speed_ratio_max = (
            float(value) for value in learner_speed_ratio_range
        )
        if not (
            PATROL_SPEED_RATIO_BOUNDS[0]
            <= learner_speed_ratio_min
            <= learner_speed_ratio_max
            <= PATROL_SPEED_RATIO_BOUNDS[1]
        ):
            raise ValueError("Multi GoalReach learner speed ratio must stay inside [0.2,5]")
        if defender_count_range is None:
            fixed_defender_count = int(
                kwargs.get("defender_count", 1)
            )
            resolved_defender_count_range = (
                fixed_defender_count,
                fixed_defender_count,
            )
            kwargs["defender_count"] = fixed_defender_count
        else:
            defender_count_min, defender_count_max = (
                int(value) for value in defender_count_range
            )
            if not 1 <= defender_count_min <= defender_count_max <= 10:
                raise ValueError("Multi GoalReach patrol count range must be inside [1,10]")
            resolved_defender_count_range = (
                defender_count_min,
                defender_count_max,
            )
            kwargs["defender_count"] = defender_count_max
        self.learner_count_range = (learner_count_min, learner_count_max)
        self.defender_count_range = resolved_defender_count_range
        self.learner_speed_ratio_range = (
            learner_speed_ratio_min,
            learner_speed_ratio_max,
        )
        self.terminate_on_any_casualty = bool(terminate_on_any_casualty)
        self.terminate_on_defender_elimination = bool(terminate_on_defender_elimination)
        self.apply_sampling_time_limit = bool(apply_sampling_time_limit)
        if reward_objective not in REWARD_DESCRIPTIONS:
            raise ValueError("Unknown reward objective")
        self.reward_objective = reward_objective
        self.viewer_learner_capacity = learner_count_max
        super().__init__(*args, **kwargs)
        self.bootstrap_on_time_limit = bool(
            self.config.goal_reach_trainer['optimizer'].get('bootstrap_on_time_limit', False)
        )
        self.learners = self.learner_dynamics.empty_state(
            (self.batch_size, self.viewer_learner_capacity)
        )
        self.learner = _actor_view(self.learners, 0)
        self.learner_active = torch.zeros(
            (self.batch_size, self.viewer_learner_capacity),
            dtype=torch.bool,
            device=self.device,
        )
        self.primary_learner_active = self.learner_active[:, 0]
        self.learner_credit_member = torch.zeros_like(self.learner_active)
        self.learner_status = torch.zeros_like(self.learner_active, dtype=torch.int8)
        self.sampled_learner_count = torch.ones(
            self.batch_size, dtype=torch.int64, device=self.device
        )
        self.sampled_defender_count = torch.full(
            (self.batch_size,),
            self.defender_count_range[0],
            dtype=torch.int64,
            device=self.device,
        )
        self.defender_success = torch.zeros_like(
            self.defender_active, dtype=torch.bool
        )
        self.defender_collision = torch.zeros_like(
            self.defender_active, dtype=torch.bool
        )
        self.defender_pair_collision = torch.zeros_like(self.defender_collision)
        self.defender_credit_member = torch.zeros_like(self.defender_active)
        self.defender_speed_ratio = torch.ones(
            (self.batch_size, self.defender_count),
            dtype=torch.float32,
            device=self.device,
        )
        self.learner_speed_ratio = torch.ones(
            (self.batch_size, self.viewer_learner_capacity),
            dtype=torch.float32,
            device=self.device,
        )
        self._learner_action_provider: (
            Callable[[dict[str, torch.Tensor]], torch.Tensor] | None
        ) = None
        self._defender_action_provider: (
            Callable[[dict[str, torch.Tensor]], torch.Tensor] | None
        ) = None
        self.match_outcome = torch.full(
            (self.batch_size,),
            int(MatchOutcome.IN_PROGRESS),
            dtype=torch.int8,
            device=self.device,
        )
        self.match_end_reason = torch.full(
            (self.batch_size,),
            MATCH_END_REASON_CODES[MatchEndReason.NONE],
            dtype=torch.int8,
            device=self.device,
        )
        self._viewer_event_tick: torch.Tensor | None = None
        self._viewer_event_code: torch.Tensor | None = None
        self._viewer_event_actor_id: torch.Tensor | None = None
        self._viewer_event_pose: torch.Tensor | None = None

        self.target_mode = str(self.config.goal_reach.get("target_mode", "static"))
        self.physical_roles = ROLE_ORDER
        self.policy_roles = policy_roles(self.config.goal_reach)
        self.roles = self.policy_roles
        self.target_count_range = tuple(self.config.goal_reach.get('target_count_range',(1,1)))
        self.target_capacity = self.target_count_range[1]
        self.sampled_target_count = torch.ones(self.batch_size,dtype=torch.long,device=self.device)
        self.targets = self.learner_dynamics.empty_state((self.batch_size, self.target_capacity))
        self.target_active = torch.zeros((self.batch_size, self.target_capacity), dtype=torch.bool, device=self.device)
        self.target_credit_member = torch.zeros_like(self.target_active)
        self.target_collision = torch.zeros_like(self.target_active)
        self.target_captured = torch.zeros_like(self.target_active)
        self.target_speed_ratio = torch.ones_like(self.target_active, dtype=torch.float32)
        self.target_speed_ratio_range = tuple(self.config.goal_reach.get("target_speed_ratio_range", (0.5, 0.5)))
        self.target_variant = self.learner_variant
        self._target_action_provider = None
        slots = self.defender_count + self.viewer_learner_capacity + self.target_capacity
        circles = self.obstacle_centers.shape[1] + self.goal_centers.shape[1] + 2
        self.contact_static = torch.zeros((self.batch_size, slots, circles), dtype=torch.bool, device=self.device)
        self.contact_pairs = torch.zeros((self.batch_size, slots, slots), dtype=torch.bool, device=self.device)
        self.episode_contact_counts = torch.zeros((self.batch_size, slots, 2), dtype=torch.int64, device=self.device)
        self.episode_pair_contacts = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        self._contact_circle_capacities = (self.obstacle_centers.shape[1], self.goal_centers.shape[1])
        self.episode_defense_utility = torch.zeros(self.batch_size, device=self.device)
        self._prepare_contact_geometry()

    @property
    def region_reference_position(self):
        return self.targets.position[:, 0]

    @region_reference_position.setter
    def region_reference_position(self, value):
        self.targets.position[:, 0].copy_(value)

    def ensure_circle_storage(self, goal_count, obstacle_count):
        super().ensure_circle_storage(goal_count, obstacle_count)
        if not hasattr(self, "contact_static"):
            return
        no, ng = self.obstacle_centers.shape[1], self.goal_centers.shape[1]
        old_no, old_ng = self._contact_circle_capacities
        if (no, ng) != (old_no, old_ng):
            contacts = torch.zeros((*self.contact_static.shape[:2], no + ng + 2), dtype=torch.bool, device=self.device)
            contacts[..., :old_no] = self.contact_static[..., :old_no]
            contacts[..., no:no+old_ng] = self.contact_static[..., old_no:old_no+old_ng]
            contacts[..., -2:] = self.contact_static[..., -2:]
            self.contact_static = contacts
            self._contact_circle_capacities = (no, ng)

    def role_state(self, role):
        return self.learners if role == "learner" else self.defenders if role == "defender" else self.targets

    def role_variant(self, role):
        return self.learner_variant if role == "learner" else self.patrol_variant if role == "defender" else self.target_variant

    def set_target_action_provider(self, provider):
        if self.target_mode != "dynamic":
            raise ValueError("Static mode has no target policy")
        self._target_action_provider = provider

    def _bind_primary_learner(self) -> None:
        self.learner = _actor_view(self.learners, 0)
        self.primary_learner_active = self.learner_active[:, 0]


    def _apply_scene_batch_extension(
        self,
        env_indices: torch.Tensor,
        scene: SceneBatch,
        mask: torch.Tensor,
        learner_active: torch.Tensor,
        defender_active: torch.Tensor,
        learner_headings: torch.Tensor,
        defender_headings: torch.Tensor,
        learner_speed_ratios: torch.Tensor,
        defender_speed_ratios: torch.Tensor,
    ) -> None:
        del defender_headings
        learner_positions = self.learners.position.clone()
        all_learner_headings = self.learners.heading.clone()
        learner_positions[env_indices] = scene.learner_positions
        all_learner_headings[env_indices] = learner_headings
        self.learner_dynamics.reset(
            self.learners, learner_positions, all_learner_headings, mask[:, None]
        )
        self.learner_active[env_indices] = learner_active
        self.learner_credit_member[env_indices] = learner_active
        self.sampled_learner_count[env_indices] = scene.learner_counts
        self.sampled_defender_count[env_indices] = scene.defender_counts
        self.defender_credit_member[env_indices] = defender_active
        self.learner_speed_ratio[env_indices] = learner_speed_ratios
        self.defender_speed_ratio[env_indices] = defender_speed_ratios
        positions, headings = self.targets.position.clone(), self.targets.heading.clone()
        seeds = self._episode_seeds[env_indices]
        positions[env_indices] = scene.target_positions
        headings[env_indices] = 2.0 * torch.pi * deterministic_uniform(seeds ^ 0x54484541, (self.target_capacity,))
        self.learner_dynamics.reset(self.targets, positions, headings, mask[:, None])
        self.sampled_target_count[env_indices] = scene.target_counts
        target_active = torch.arange(self.target_capacity,device=self.device)[None] < scene.target_counts[:,None]
        self.target_active[env_indices] = target_active
        self.target_credit_member[env_indices] = target_active
        self.target_collision[env_indices] = False
        self.target_captured[env_indices] = False
        low, high = self.target_speed_ratio_range
        self.target_speed_ratio[env_indices] = low + (high - low) * deterministic_uniform(seeds ^ 0x54535044, (self.target_capacity,))
        self.episode_defense_utility[env_indices] = 0.
        self._bind_primary_learner()

    def _apply_fixed_spawn_batch(
        self,
        env_indices: torch.Tensor,
        spawns: ActorSpawnBatch,
        mask: torch.Tensor,
    ) -> None:
        learner_positions = self.learners.position.clone()
        learner_headings = self.learners.heading.clone()
        learner_positions[env_indices] = spawns.learner_positions.to(self.device)
        learner_headings[env_indices] = spawns.learner_headings.to(self.device)
        self.learner_dynamics.reset(
            self.learners, learner_positions, learner_headings, mask[:, None]
        )
        self.learner_active[env_indices] = spawns.learner_active.to(self.device)
        self.learner_credit_member[env_indices] = self.learner_active[env_indices]
        if self.defender_count:
            defender_positions = self.defenders.position.clone()
            defender_headings = self.defenders.heading.clone()
            defender_positions[env_indices] = spawns.defender_positions.to(self.device)
            defender_headings[env_indices] = spawns.defender_headings.to(self.device)
            self.defender_active[env_indices] = spawns.defender_active.to(self.device)
            self.defender_credit_member[env_indices] = self.defender_active[env_indices]
            self.defender_initial_position[env_indices] = defender_positions[env_indices]
            self.defender_initial_heading[env_indices] = defender_headings[env_indices]
            self.defender_dynamics.reset(
                self.defenders, defender_positions, defender_headings, mask[:, None]
            )
        self._bind_primary_learner()

    def viewer_learner_active(self, arena_index: int) -> torch.Tensor:
        return self.learner_active[int(arena_index)]

    def _viewer_learner_state(self, arena_index: int) -> tuple[torch.Tensor, ...]:
        arena_index = int(arena_index)
        return tuple(
            getattr(self.learners, name)[arena_index]
            for name in ("position", "heading", "body_velocity", "desired_body_velocity")
        )


    def patrol_limits(
        self, speed_ratio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        speed_ratio = speed_ratio.to(device=self.device, dtype=torch.float32)
        base_surge = self.learner_dynamics.maximum_surge
        maximum_surge = base_surge * speed_ratio
        maximum_yaw = self.learner_dynamics.maximum_yaw * speed_ratio
        linear_drag = self.learner_dynamics.linear_drag[0]
        quadratic_drag = self.learner_dynamics.quadratic_drag[0]
        base_resistance = linear_drag * base_surge + quadratic_drag * base_surge**2
        scaled_resistance = (
            linear_drag * maximum_surge
            + quadratic_drag * maximum_surge.square()
        )
        maximum_thruster = (
            self.learner_dynamics.horizontal_thruster_limit
            * scaled_resistance
            / base_resistance
        )
        return maximum_surge, maximum_yaw, maximum_thruster

    def learner_limits(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.patrol_limits(self.learner_speed_ratio)

    def _set_learner_commands(
        self, action: torch.Tensor,
        limits: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action = action.clamp(-1.0, 1.0)
        maximum_surge, maximum_yaw, maximum_thruster = self.learner_limits() if limits is None else limits
        self.learner_dynamics.set_command(
            self.learners,
            0.5 * (action[..., 0] + 1.0) * maximum_surge,
            maximum_yaw * action[..., 1],
            maximum_surge=maximum_surge,
            maximum_yaw=maximum_yaw,
        )
        return maximum_surge, maximum_yaw, maximum_thruster

    def _cpu_joint_limits(self):
        """One value-validated, privately owned set of CPU limits per environment."""
        dynamics = self.learner_dynamics
        dependencies = (dynamics.maximum_surge, dynamics.maximum_yaw,
                        dynamics.horizontal_thruster_limit, float(dynamics.linear_drag[0]),
                        float(dynamics.quadratic_drag[0]))
        ratios = (self.defender_speed_ratio, self.learner_speed_ratio, self.target_speed_ratio)
        previous = getattr(self, '_cpu_limit_cache', None)
        limits, snapshots, changed = [], [], False
        for index, ratio in enumerate(ratios):
            old = previous[1][index] if previous is not None else None
            if (previous is not None and previous[0] == dependencies and old.dtype == ratio.dtype
                    and old.device == ratio.device and old.shape == ratio.shape and torch.equal(old, ratio)):
                snapshots.append(old)
                limits.append(previous[2][index])
            else:
                # Cache ownership must survive a caller's inference_mode scope.
                with torch.inference_mode(False), torch.no_grad():
                    snapshots.append(ratio.clone())
                    limits.append(tuple(value.clone() for value in self.patrol_limits(ratio)))
                changed = True
        if changed:
            with torch.inference_mode(False), torch.no_grad():
                mixed = tuple(torch.cat(values, dim=1) for values in zip(*limits))
            self._cpu_limit_cache = (dependencies, snapshots, limits, mixed)
        return self._cpu_limit_cache[2:]

    def _cpu_role_bindings(self):
        return tuple(value for state in (self.defenders, self.learners, self.targets)
                     for value in state.__dict__.values())

    @staticmethod
    def _cpu_binding_layout(value):
        return (value.device, value.dtype, value.shape, value.stride(), value.data_ptr())

    def _cpu_joint_state(self):
        bindings = self._cpu_role_bindings()
        previous = getattr(self, '_cpu_state_cache', None)
        if previous is not None and len(bindings) == len(previous[1]) and all(
                value is old and self._cpu_binding_layout(value) == layout
                for value, old, layout in zip(bindings, previous[1], previous[2])):
            return previous[0]
        with torch.inference_mode(False), torch.no_grad():
            return self._joint_state()

    def _joint_state(self):
        return AuvState(**{
            name: torch.cat((getattr(self.defenders, name), getattr(self.learners, name),
                             getattr(self.targets, name)), dim=1)
            for name in self.defenders.__dict__
        })

    def set_learner_action_provider(
        self, provider: Callable[[dict[str, torch.Tensor]], torch.Tensor]
    ) -> None:
        self._learner_action_provider = provider

    def set_defender_action_provider(
        self, provider: Callable[[dict[str, torch.Tensor]], torch.Tensor]
    ) -> None:
        self._defender_action_provider = provider

    def enable_viewer_event_capture(self) -> None:
        """Enable a bounded, Viewer-only buffer for exact event-tick poses."""
        slot_count = 2 * self.viewer_learner_capacity + self.defender_count + 2*self.target_capacity
        shape = (self.batch_size, slot_count)
        self._viewer_event_tick = torch.full(
            shape, -1, dtype=torch.int64, device=self.device
        )
        self._viewer_event_code = torch.zeros(
            shape, dtype=torch.int8, device=self.device
        )
        self._viewer_event_actor_id = torch.full(
            shape, -1, dtype=torch.int64, device=self.device
        )
        self._viewer_event_pose = torch.zeros(
            (*shape, 3), dtype=torch.float32, device=self.device
        )

    def clear_viewer_event_capture(self, mask: torch.Tensor | None = None) -> None:
        if self._viewer_event_tick is None:
            return
        resolved_mask = (
            torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
            if mask is None
            else mask.to(device=self.device, dtype=torch.bool)
        )
        self._viewer_event_tick[resolved_mask] = -1

    def _capture_viewer_events(
        self,
        actor_state: AuvState,
        learner_boundary: torch.Tensor,
        learner_obstacle: torch.Tensor,
        learner_pair: torch.Tensor,
        captured: torch.Tensor,
        capture_assignment: torch.Tensor,
        defender_boundary: torch.Tensor,
        defender_obstacle: torch.Tensor,
        defender_pair: torch.Tensor,
        *, rows: torch.Tensor | None = None,
    ) -> None:
        if self._viewer_event_tick is None:
            return
        assert self._viewer_event_code is not None
        assert self._viewer_event_actor_id is not None
        assert self._viewer_event_pose is not None
        rows = torch.arange(self.batch_size, device=self.device) if rows is None else rows
        na, nd = self.viewer_learner_capacity, self.defender_count
        learner_event = learner_boundary | learner_obstacle | learner_pair | captured
        for mask, offset, state_offset, boundary, obstacle, pair, capture in (
            (learner_event, 0, nd, learner_boundary, learner_obstacle, learner_pair, captured),
            (defender_boundary | defender_obstacle | defender_pair, 2 * na, 0,
             defender_boundary, defender_obstacle, defender_pair, None),
        ):
            lane, actor = mask.nonzero(as_tuple=True)
            event_rows, slots = rows[lane], offset + actor
            code = torch.where(pair[lane, actor], 3, torch.where(obstacle[lane, actor], 2, 1)).to(torch.int8)
            if capture is not None:
                code = torch.where(capture[lane, actor], 4, code)
            self._viewer_event_tick[event_rows, slots] = self.physics_ticks[event_rows].long() + int(capture is None)
            self._viewer_event_code[event_rows, slots] = code
            self._viewer_event_pose[event_rows, slots] = torch.cat((
                actor_state.position[event_rows, state_offset + actor],
                actor_state.heading[event_rows, state_offset + actor, None]), -1)
        lane, attacker = captured.nonzero(as_tuple=True)
        event_rows, slots = rows[lane], na + attacker
        capturer = capture_assignment[lane, :, attacker].to(torch.int8).argmax(1)
        self._viewer_event_tick[event_rows, slots] = self.physics_ticks[event_rows].long()
        self._viewer_event_code[event_rows, slots] = 4
        self._viewer_event_actor_id[event_rows, slots] = capturer
        self._viewer_event_pose[event_rows, slots] = torch.cat((
            actor_state.position[event_rows, capturer], actor_state.heading[event_rows, capturer, None]), -1)

    def drain_viewer_event_freeze_frames(
        self, arena_index: int
    ) -> list[dict[str, object]]:
        if self._viewer_event_tick is None:
            return []
        assert self._viewer_event_code is not None
        assert self._viewer_event_actor_id is not None
        assert self._viewer_event_pose is not None
        arena_index = int(arena_index)
        slots = torch.nonzero(
            self._viewer_event_tick[arena_index] >= 0, as_tuple=False
        ).flatten()
        ticks = self._viewer_event_tick[arena_index, slots].tolist()
        codes = self._viewer_event_code[arena_index, slots].tolist()
        actor_ids = self._viewer_event_actor_id[arena_index, slots].tolist()
        poses = self._viewer_event_pose[arena_index, slots].tolist()
        learner_count = self.viewer_learner_capacity
        event_names = (
            "",
            "boundary_collision",
            "obstacle_collision",
            "pair_collision",
            "capture",
        )
        frames: list[dict[str, object]] = []
        for slot, tick, code, stored_actor_id, pose in zip(
            slots.tolist(), ticks, codes, actor_ids, poses, strict=True
        ):
            capturer = learner_count <= slot < 2 * learner_count
            learner = slot < learner_count
            target_slot = 2 * learner_count + self.defender_count
            dynamic_slot = slot >= target_slot
            target_id=(slot-target_slot)//2
            target_prey=dynamic_slot and (slot-target_slot)%2==0
            actor_kind = 'target' if target_prey else 'learner' if dynamic_slot or learner else 'defender'
            actor_id = (
                slot
                if learner
                else stored_actor_id
                if capturer
                else slot - 2 * learner_count
            )
            if dynamic_slot:
                actor_id = target_id if target_prey else stored_actor_id
                capturer = not target_prey
            event_type = event_names[code]
            source_learner_id = slot if learner else slot - learner_count
            event_id = (
                (f"capture:{tick}:target:{target_id}" if dynamic_slot else f"capture:{tick}:learner:{source_learner_id}")
                if code == 4
                else f"{event_type}:{tick}:{actor_kind}:{actor_id}"
            )
            frames.append(
                {
                    "event_id": event_id,
                    "event_type": event_type,
                    "physics_ticks": tick,
                    "actor_kind": actor_kind,
                    "actor_id": actor_id,
                    "role": (
                        "capturer"
                        if capturer
                        else "captured"
                        if code == 4
                        else "casualty"
                    ),
                    "x": float(pose[0]),
                    "z": float(pose[1]),
                    "h": float(pose[2]),
                    "collision_radius_m": self.vessel_radius_m,
                    "status": (
                        "capture"
                        if capturer
                        else "captured"
                        if code == 4
                        else event_type
                    ),
                }
            )
        self._viewer_event_tick[arena_index] = -1
        return frames

    def reset(
        self,
        mask: torch.Tensor | None = None,
        base_observation: dict[str, torch.Tensor] | None = None,
        env_indices_cpu: torch.Tensor | None = None,
        *,
        env_indices: torch.Tensor | None = None,
        episode_seeds_cpu: torch.Tensor | None = None,
        build_observation: bool = True,
    ) -> dict[str, torch.Tensor]:
        if mask is None:
            mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        mask = mask.to(device=self.device, dtype=torch.bool)
        self._invalidate_obstacle_neighbors(mask)
        self._bind_primary_learner()
        super().reset(
            mask,
            env_indices_cpu=env_indices_cpu,
            env_indices=env_indices,
            episode_seeds_cpu=episode_seeds_cpu,
            build_observation=False,
        )
        self.learner_status[mask] = 0
        self.defender_success[mask] = False
        self.defender_collision[mask] = False
        self.defender_pair_collision[mask] = False
        self.match_outcome[mask] = int(MatchOutcome.IN_PROGRESS)
        self.match_end_reason[mask] = MATCH_END_REASON_CODES[MatchEndReason.NONE]
        self.done[mask] = False
        for field in (self.contact_static, self.contact_pairs,
                      self.episode_contact_counts, self.episode_pair_contacts):
            field[mask] = 0
        self.clear_viewer_event_capture(mask)
        if not build_observation:
            return {}
        if base_observation is None:
            return self.training_observation()
        refreshed = self.training_observation(mask)
        return merge_reset_observation(base_observation, refreshed, mask)

    def reset_fixed_scene(
        self,
        *,
        goal_centers: Sequence[tuple[float, float]],
        goal_radii: Sequence[float],
        obstacle_centers: Sequence[tuple[float, float]],
        obstacle_radii: Sequence[float],
        mask: torch.Tensor | None = None,
        base_observation: dict[str, torch.Tensor] | None = None,
        env_indices_cpu: torch.Tensor | None = None,
        episode_seeds_cpu: torch.Tensor | None = None,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        return super().reset_fixed_scene(
            goal_centers=goal_centers,
            goal_radii=goal_radii,
            obstacle_centers=obstacle_centers,
            obstacle_radii=obstacle_radii,
            mask=mask,
            base_observation=base_observation,
            env_indices_cpu=env_indices_cpu,
            episode_seeds_cpu=episode_seeds_cpu,
            **kwargs,
        )

    def restart_edited_episode(self, mask: torch.Tensor) -> None:
        mask = mask.to(device=self.device, dtype=torch.bool)
        if self.defender_count:
            self.defender_dynamics.reset(
                self.defenders,
                self.defenders.position.clone(),
                self.defenders.heading.clone(),
                mask[:, None],
            )
            slots = torch.arange(self.defender_count, device=self.device)
            self.defender_active[mask] = (
                slots[None, :] < self.sampled_defender_count[mask, None]
            )
            self.defender_credit_member[mask] = self.defender_active[mask]
        self.learner_dynamics.reset(
            self.learners,
            self.learners.position.clone(),
            self.learners.heading.clone(),
            mask[:, None],
        )
        self.learner_dynamics.reset(self.targets, self.targets.position.clone(), self.targets.heading.clone(), mask[:, None])
        self.target_active[mask] = torch.arange(self.target_capacity,device=self.device)[None] < self.sampled_target_count[mask,None]
        self.target_credit_member[mask] = self.target_active[mask]
        self.target_captured[mask] = False
        self.target_collision[mask] = False
        self._bind_primary_learner()
        self.physics_ticks[mask] = 0
        self.episode_defense_utility[mask] = 0.
        self.done[mask] = False
        self.success[mask] = False
        self.collision[mask] = False
        self.defender_success[mask] = False
        self.defender_collision[mask] = False
        self.defender_pair_collision[mask] = False
        slots = torch.arange(self.viewer_learner_capacity, device=self.device)
        self.learner_active[mask] = (
            slots[None, :] < self.sampled_learner_count[mask, None]
        )
        self.learner_credit_member[mask] = self.learner_active[mask]
        self.learner_status[mask] = 0
        self.match_outcome[mask] = int(MatchOutcome.IN_PROGRESS)
        self.match_end_reason[mask] = MATCH_END_REASON_CODES[MatchEndReason.NONE]
        for field in (self.contact_static, self.contact_pairs,
                      self.episode_contact_counts, self.episode_pair_contacts):
            field[mask] = 0
        self.clear_viewer_event_capture(mask)

    def set_viewer_learner_pose(
        self,
        arena_index: int,
        learner_index: int,
        position: tuple[float, float],
        heading: float,
    ) -> None:
        arena_index = int(arena_index)
        learner_index = int(learner_index)
        if not 0 <= learner_index < self.viewer_learner_capacity:
            raise IndexError("GoalReach learner index is outside its capacity")
        positions = self.learners.position.clone()
        headings = self.learners.heading.clone()
        positions[arena_index, learner_index] = torch.tensor(
            position, dtype=torch.float32, device=self.device
        )
        headings[arena_index, learner_index] = float(heading)
        actor_mask = torch.zeros_like(self.learner_active)
        actor_mask[arena_index, learner_index] = True
        self.learner_dynamics.reset(self.learners, positions, headings, actor_mask)
        self._bind_primary_learner()
        self.learner_active[arena_index, learner_index] = True
        self.learner_status[arena_index, learner_index] = 0

    def observe_viewer_learners(self, arena_index: int) -> dict[str, torch.Tensor]:
        arena_index = int(arena_index)
        positions, headings, velocity, _ = self._viewer_learner_state(arena_index)
        actor_indices = torch.nonzero(
            self.learner_active[arena_index], as_tuple=False
        ).flatten()
        if self.learner_variant.unified_match_contract:
            arena_indices = torch.tensor(
                [arena_index], dtype=torch.long, device=self.device
            )
            observation = self._unified_learner_training_observation(arena_indices)
            return {
                key: value[0, actor_indices]
                for key, value in observation.items()
            }
        actor_count = actor_indices.shape[0]
        observation = self._goal_reach_actor_observation(
            positions[actor_indices],
            headings[actor_indices],
            velocity[actor_indices],
            self.obstacle_centers[arena_index].expand(actor_count, -1, -1),
            self.obstacle_radii[arena_index].expand(actor_count, -1),
            self.obstacle_active[arena_index].expand(actor_count, -1),
            self.goal_centers[arena_index].expand(actor_count, -1, -1),
            self.goal_radii[arena_index].expand(actor_count, -1),
            self.goal_active[arena_index].expand(actor_count, -1),
        )
        selected_positions = positions[actor_indices]
        selected_headings = headings[actor_indices]
        relative = self._rotate_geometry(
            positions[None, :, :] - selected_positions[:, None, :],
            selected_headings[:, None],
        ) / self.world.radius
        friend_forward = torch.stack((headings.sin(), headings.cos()), dim=-1)
        friend_right = torch.stack((headings.cos(), -headings.sin()), dim=-1)
        world_velocity = (
            friend_forward * velocity[:, :1] + friend_right * velocity[:, 1:2]
        )
        velocity_in_observer_frame = self._rotate_geometry(
            world_velocity[None, :, :].expand(actor_count, -1, -1),
            selected_headings[:, None],
        ) / 2.0
        relative_heading = headings[None, :] - selected_headings[:, None]
        relative_heading_features = torch.stack(
            (relative_heading.sin(), relative_heading.cos()), dim=-1
        )
        friend_yaw_rate = (
            velocity[None, :, 2:3].expand(actor_count, -1, -1) / 0.2
        )
        friendly_mask = self.learner_active[arena_index].expand(actor_count, -1).clone()
        friendly_mask[
            torch.arange(actor_count, device=self.device), actor_indices
        ] = False
        friendly_entities = torch.cat(
            (
                relative,
                velocity_in_observer_frame,
                relative_heading_features,
                friend_yaw_rate,
            ),
            dim=-1,
        ) * friendly_mask.unsqueeze(-1).to(torch.float32)
        observation["friendly_entities"] = friendly_entities
        observation["friendly_mask"] = friendly_mask
        return observation

    def _remaining_time(self, arena_indices: torch.Tensor) -> torch.Tensor:
        maximum_ticks = self.config.goal_reach["clock"]["maximum_physics_ticks"]
        if maximum_ticks is None or self.bootstrap_on_time_limit:
            return torch.ones(
                arena_indices.shape, dtype=torch.float32, device=self.device
            )
        return (
            1.0
            - self.physics_ticks[arena_indices].to(torch.float32) / float(maximum_ticks)
        ).clamp(0.0, 1.0)

    @staticmethod
    def _normalized_speed_ratio(speed_ratio: torch.Tensor) -> torch.Tensor:
        _, maximum = PATROL_SPEED_RATIO_BOUNDS
        return speed_ratio.log() / math.log(maximum)

    @staticmethod
    def _validate_observation_consumer(consumer):
        if consumer not in ("actor", "critic", "both"):
            raise ValueError(f"Unknown observation consumer: {consumer!r}")

    def _prepare_observation_context(self, arenas):
        """Local rows map to these global arena IDs; never retained across step/reset."""
        context = {}
        for role in ENTITY_SET_ORDER[:-1]:
            state = self.role_state(role)
            heading = state.heading[arenas]
            velocity = state.body_velocity[arenas]
            speed = getattr(self, role + "_speed_ratio")[arenas]
            sine, cosine = heading.sin(), heading.cos()
            forward = torch.stack((sine, cosine), dim=-1)
            right = torch.stack((cosine, -sine), dim=-1)
            context[role] = dict(
                position=state.position[arenas], heading=heading, velocity=velocity,
                world_velocity=forward * velocity[..., :1] + right * velocity[..., 1:2],
                normalized_speed=self._normalized_speed_ratio(speed),
                limits=self.patrol_limits(speed),
                active=getattr(self, role + "_active")[arenas],
                member=getattr(self, role + "_credit_member")[arenas])
        return context

    def _unified_craft_entities(self, position, heading, world_velocity, maximum,
                                entity, rows, member, *, consumer):
        relative_position = self._rotate_geometry(
            entity["position"][rows] - position[:, None], heading[:, None]
        ) / self.position_scale_m
        entity_maximum, entity_yaw_maximum, _ = entity["limits"]
        relative_velocity = self._rotate_geometry(
            entity["world_velocity"][rows] - world_velocity[:, None], heading[:, None]
        ) / ((maximum[:, None] + entity_maximum[rows]) * math.sqrt(1.25)).clamp_min(1.e-6).unsqueeze(-1)
        relative_heading = entity["heading"][rows] - heading[:, None]
        fields = (
            relative_position, relative_velocity,
            torch.stack((relative_heading.sin(), relative_heading.cos()), dim=-1),
            entity["velocity"][rows][..., 2:3] / entity_yaw_maximum[rows].clamp_min(1.e-6).unsqueeze(-1),
            entity["normalized_speed"][rows].unsqueeze(-1))
        if consumer != "actor":
            fields += (entity["active"][rows].to(torch.float32).unsqueeze(-1),)
        return torch.cat(fields, dim=-1) * member.unsqueeze(-1).to(torch.float32)

    def _neighbor_timing_start(self):
        probe = getattr(self,'_training_memory_probe',None)
        if probe is None or self.device.type != 'cuda':
            return None
        if not hasattr(self,'_neighbor_trace_update'):
            self._neighbor_trace_update = probe.update
        if probe.update != self._neighbor_trace_update:
            return None
        event = torch.cuda.Event(enable_timing=True);event.record()
        return event

    def _neighbor_timing_end(self, label, start):
        if start is not None:
            end = torch.cuda.Event(enable_timing=True);end.record()
            if not hasattr(self,'_neighbor_trace_events'):self._neighbor_trace_events=[]
            self._neighbor_trace_events.append((label,start,end))

    def _local_obstacle_observation(self, position, heading, alive, arena_indices, *, record=None):
        """Fixed first ten surface-visible instances; physical candidates stay complete."""
        from .terrain import compact_first_indices
        selection_started = self._neighbor_timing_start()
        centers = self.obstacle_centers[arena_indices]
        radii = self.obstacle_radii[arena_indices]
        active = self.obstacle_active[arena_indices]
        squared_distance = (centers - position[:, None]).square().sum(-1)
        visible = active & alive[:, None] & (squared_distance <= (15.0 + radii + CONTACT_TOLERANCE).square())
        index, mask = compact_first_indices(visible, 10)
        self._neighbor_timing_end('distance_mask_compact', selection_started)
        if record is not None and self._obstacle_neighbor_acceleration_enabled():
            physical = active & alive[:,None] & (squared_distance <=
                (self.world.radius+radii+CONTACT_TOLERANCE).square())
            physical_index, physical_mask = compact_first_indices(physical)
            self._remember_obstacle_neighbors(record, arena_indices, position, centers, radii,
                active, squared_distance.sqrt(), physical, physical_index, physical_mask)
        if centers.shape[1] == 0:
            return position.new_zeros((len(position),10,3)), mask
        selected_centers = centers.gather(1, index[..., None].expand(-1, -1, 2))
        selected_radii = radii.gather(1, index)
        features = self._goal_reach_entity_buffer(position, heading, selected_centers, selected_radii, mask)
        return torch.where(mask[..., None], features, 0.), mask

    def _remember_obstacle_neighbors(self, record, lanes, position, centers, radii, active, distance, visible, index, mask):
        role, actors = record
        offsets = {'defender':0,'learner':self.defender_count,
                   'target':self.defender_count+self.viewer_learner_capacity}
        slots = actors + offsets[role]
        width = index.shape[1]
        shape = (self.batch_size,self.defender_count+self.viewer_learner_capacity+self.target_capacity)
        cache = getattr(self,'_obstacle_neighbors',None)
        if cache is None or cache['valid'].shape != shape:
            # Viewer inference prepares these buffers; reset/Replay may invalidate
            # them outside inference mode, so the derived storage must be normal.
            with torch.inference_mode(False):
                cache = dict(valid=torch.zeros(shape,dtype=torch.bool,device=self.device),
                    index=torch.zeros((*shape,width),dtype=torch.long,device=self.device),
                    mask=torch.zeros((*shape,width),dtype=torch.bool,device=self.device),
                    p0=position.new_zeros((*shape,2)),gap=position.new_zeros(shape),scale=position.new_ones(shape))
            self._obstacle_neighbors = cache
        elif cache['index'].shape[-1] < width:
            with torch.inference_mode(False):
                for key in ('index', 'mask'):
                    cache[key] = torch.nn.functional.pad(cache[key], (0, width - cache[key].shape[-1]))
        finite = (torch.isfinite(centers).all(-1)&torch.isfinite(radii)&(radii>0)) | ~active
        torch._assert_async(finite.all() & torch.isfinite(position).all(), 'Nonfinite obstacle selection geometry')
        gap = (torch.where(active&~visible,distance-radii-self.vessel_radius_m,torch.inf).amin(-1)
               if active.shape[-1] else position.new_full(position.shape[:1], torch.inf))
        world_scale = (torch.where(active,torch.maximum(centers.abs().amax(-1),radii+self.vessel_radius_m),0.).amax(-1)
                       if active.shape[-1] else position.new_zeros(position.shape[:1]))
        scale = torch.maximum(world_scale,position.abs().amax(-1)).clamp_min(1.)
        padding = cache['index'].shape[-1]-width
        cache['index'][lanes,slots] = torch.nn.functional.pad(index,(0,padding))
        cache['mask'][lanes,slots] = torch.nn.functional.pad(mask,(0,padding))
        cache['p0'][lanes,slots] = position
        cache['gap'][lanes,slots] = gap
        cache['scale'][lanes,slots] = scale
        cache['valid'][lanes,slots] = True

    def _obstacle_neighbor_safe(self, position, endpoint, active):
        """One guard per physical substep; residual segments lie inside this segment."""
        cache = getattr(self,'_obstacle_neighbors',None)
        if cache is None:
            return torch.zeros(self.batch_size,dtype=torch.bool,device=self.device)
        guard_started = self._neighbor_timing_start()
        torch._assert_async(((torch.isfinite(position).all(-1)&torch.isfinite(endpoint).all(-1))|~active).all(),
                            'Nonfinite collision segment')
        displacement = torch.maximum((position-cache['p0']).norm(dim=-1),(endpoint-cache['p0']).norm(dim=-1))
        scale = torch.maximum(cache['scale'],torch.maximum(position.abs().amax(-1),endpoint.abs().amax(-1)))
        margin = CONTACT_TOLERANCE*(1+(endpoint-position).norm(dim=-1)) + 64*torch.finfo(position.dtype).eps*scale
        valid = (cache['valid']|~active).all(-1)
        distant = ((cache['gap']>displacement+margin)|~active).all(-1)
        safe = valid&distant
        self._neighbor_timing_end('guard',guard_started)
        probe = getattr(self,'_training_memory_probe',None)
        if probe is not None and probe.update <= probe.maximum_updates:
            if not hasattr(self,'_collision_neighbor_counts'):
                self._collision_neighbor_counts = torch.zeros(5,dtype=torch.int64,device=self.device)
            self._collision_neighbor_counts[:3] += torch.stack((safe.sum(),(~valid).sum(),(valid&~distant).sum()))
        return safe

    def _unified_role_observation(self, role, arena_indices, actor_indices, *,
                                  record_neighbors=False, consumer="both", context=None, context_rows=None):
        self._validate_observation_consumer(consumer)
        if context is None:
            arenas, context_rows = torch.unique(arena_indices, sorted=True, return_inverse=True)
            context = self._prepare_observation_context(arenas)
        if context_rows is None:
            raise ValueError("Observation context requires local arena rows")
        own = context[role]
        position = own["position"][context_rows, actor_indices]
        heading = own["heading"][context_rows, actor_indices]
        velocity = own["velocity"][context_rows, actor_indices]
        world_velocity = own["world_velocity"][context_rows, actor_indices]
        alive = own["active"][context_rows, actor_indices]
        speed_cap, yaw_cap = (v[context_rows, actor_indices] for v in own["limits"][:2])
        task_limit = {
            "learner": 1.0,
            "defender": float(self.config.goal_reach["safety_zones"]["engagement_radius_m"]) / self.world.radius,
            "target": 0.0,
        }[role]
        observation = self._assemble_actor_observation(self.role_variant(role), {}, {
            "surge_over_max": velocity[:, 0] / speed_cap.clamp_min(1.e-6),
            "sway_over_half_max": velocity[:, 1] / (0.5 * speed_cap).clamp_min(1.e-6),
            "yaw_rate_over_max": velocity[:, 2] / yaw_cap.clamp_min(1.e-6),
            "speed_ratio_normalized": own["normalized_speed"][context_rows, actor_indices],
            "alive": alive.float(),
            **({"remaining_time": self._remaining_time(arena_indices)} if "remaining_time" in self.role_variant(role).vector_inputs else {}),
            "task_limit_over_R": torch.full_like(alive, task_limit, dtype=torch.float32),
            **{f"role_{label}": torch.full_like(alive,float(role==r),dtype=torch.float32)
               for r,label in zip(ROLE_ORDER,("atk","def","tgt"))},
        })
        obstacle_entities, obstacle_mask = self._local_obstacle_observation(position, heading, alive, arena_indices,
            record=(role,actor_indices) if record_neighbors else None)
        observation["obstacle_entities"] = obstacle_entities
        observation["obstacle_mask"] = obstacle_mask
        for other in ENTITY_SET_ORDER[:-1]:
            entity = context[other]
            member = entity["member"][context_rows]
            slots = torch.arange(member.shape[1], device=self.device)
            is_self = (other == role) & (slots[None] == actor_indices[:, None])
            member = member & ~is_self
            features = self._unified_craft_entities(position, heading, world_velocity, speed_cap,
                entity, context_rows, member, consumer=consumer)
            if consumer != "critic":
                active = entity["active"][context_rows] & ~is_self
                within = (entity["position"][context_rows] - position[:, None]).norm(dim=-1) <= self.world.radius + CONTACT_TOLERANCE
                actor_mask = active & within
                actor_features = features if consumer == "actor" else features[..., :-1]
                observation[other+"_entities"] = actor_features * actor_mask.unsqueeze(-1)
                observation[other+"_mask"] = actor_mask
            if consumer != "actor":
                observation[other+"_critic_entities"] = features
                observation[other+"_critic_mask"] = member
        return observation

    def _unified_learner_training_observation(self, arena_indices):
        capacity = self.viewer_learner_capacity
        arenas = arena_indices[:, None].expand(-1, capacity).reshape(-1)
        actors = torch.arange(capacity, device=self.device).expand(arena_indices.numel(), -1).reshape(-1)
        observation = self._unified_role_observation("learner", arenas, actors)
        observation = {key: value.reshape(arena_indices.numel(), capacity, *value.shape[1:])
                       for key, value in observation.items()}
        observation.update(actor_alive=self.learner_active[arena_indices].clone(),
                           actor_active=self.learner_active[arena_indices].clone(),
                           credit_member=self.learner_credit_member[arena_indices].clone())
        return observation

    def _learner_training_observation(
        self, mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if mask is None:
            arena_indices = torch.arange(self.batch_size, device=self.device)
        else:
            mask = mask.to(device=self.device, dtype=torch.bool)
            arena_indices = torch.nonzero(mask, as_tuple=False).flatten()
        if self.learner_variant.unified_match_contract:
            return self._unified_learner_training_observation(arena_indices)

        positions = self.learners.position[arena_indices]
        headings = self.learners.heading[arena_indices]
        velocity = self.learners.body_velocity[arena_indices]
        active = self.learner_active[arena_indices]
        arena_count, capacity = active.shape
        actor_count = arena_count * capacity

        observation = self._goal_reach_actor_observation(
            positions.reshape(actor_count, 2),
            headings.reshape(actor_count),
            velocity.reshape(actor_count, 3),
            self.obstacle_centers[arena_indices, None]
            .expand(-1, capacity, -1, -1)
            .reshape(actor_count, self.obstacle_centers.shape[1], 2),
            self.obstacle_radii[arena_indices, None]
            .expand(-1, capacity, -1)
            .reshape(actor_count, self.obstacle_radii.shape[1]),
            self.obstacle_active[arena_indices, None]
            .expand(-1, capacity, -1)
            .reshape(actor_count, self.obstacle_active.shape[1]),
            self.goal_centers[arena_indices, None]
            .expand(-1, capacity, -1, -1)
            .reshape(actor_count, self.goal_centers.shape[1], 2),
            self.goal_radii[arena_indices, None]
            .expand(-1, capacity, -1)
            .reshape(actor_count, self.goal_radii.shape[1]),
            self.goal_active[arena_indices, None]
            .expand(-1, capacity, -1)
            .reshape(actor_count, self.goal_active.shape[1]),
        )

        relative_position = self._rotate_geometry(
            positions[:, None, :, :] - positions[:, :, None, :],
            headings[:, :, None],
        ) / self.world.radius
        friend_forward = torch.stack((headings.sin(), headings.cos()), dim=-1)
        friend_right = torch.stack((headings.cos(), -headings.sin()), dim=-1)
        world_velocity = (
            friend_forward * velocity[..., :1] + friend_right * velocity[..., 1:2]
        )
        velocity_in_observer_frame = self._rotate_geometry(
            world_velocity[:, None].expand(-1, capacity, -1, -1),
            headings[:, :, None],
        ) / 2.0
        relative_heading = headings[:, None, :] - headings[:, :, None]
        relative_heading_features = torch.stack(
            (relative_heading.sin(), relative_heading.cos()), dim=-1
        )
        friend_yaw_rate = velocity[:, None, :, 2:3].expand(
            -1, capacity, -1, -1
        ) / 0.2
        self_mask = torch.eye(capacity, dtype=torch.bool, device=self.device)[None]
        friendly_mask = active[:, None, :].expand(-1, capacity, -1) & ~self_mask
        friendly_entities = torch.cat(
            (
                relative_position,
                velocity_in_observer_frame,
                relative_heading_features,
                friend_yaw_rate,
            ),
            dim=-1,
        ) * friendly_mask.unsqueeze(-1).to(torch.float32)

        observation = {
            key: value.reshape(arena_count, capacity, *value.shape[1:])
            for key, value in observation.items()
        }
        observation["goal_mask"] = observation["goal_mask"].clone()
        observation["obstacle_mask"] = observation["obstacle_mask"].clone()
        observation["friendly_entities"] = friendly_entities
        observation["friendly_mask"] = friendly_mask
        observation["actor_alive"] = active.clone()
        observation["credit_member"] = self.learner_credit_member[
            arena_indices
        ].clone()
        observation["actor_active"] = active.clone()
        return observation

    def _unified_defender_observation(self, arena_indices, actor_indices):
        return self._unified_role_observation("defender", arena_indices, actor_indices)

    def _defender_pursuit_observation(
        self,
        arena_indices: torch.Tensor,
        actor_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        arena_indices = arena_indices.to(device=self.device, dtype=torch.int64)
        actor_indices = (
            torch.zeros_like(arena_indices)
            if actor_indices is None
            else actor_indices.to(device=self.device, dtype=torch.int64)
        )
        if actor_indices.shape != arena_indices.shape:
            raise ValueError("Patrol observation Actor and arena indices must align")
        if self.patrol_variant.unified_match_contract:
            return self._unified_defender_observation(arena_indices, actor_indices)
        positions = self.defenders.position[arena_indices, actor_indices]
        headings = self.defenders.heading[arena_indices, actor_indices]
        velocity = self.defenders.body_velocity[arena_indices, actor_indices]
        speed_ratio = self.defender_speed_ratio[arena_indices, actor_indices]
        speed_cap, yaw_cap, _ = self.patrol_limits(speed_ratio)
        map_position = self._rotate_geometry(positions, headings)
        observation = self._assemble_actor_observation(
            self.patrol_variant,
            {},
            {
                "map_frame_x_over_R": map_position[:, 0] / self.world.radius,
                "map_frame_z_over_R": map_position[:, 1] / self.world.radius,
                "surge_over_max": velocity[:, 0] / speed_cap,
                "sway_over_half_max": velocity[:, 1] / (0.5 * speed_cap),
                "yaw_rate_over_max": velocity[:, 2] / yaw_cap.clamp_min(1.0e-6),
                "defender_speed_ratio_normalized": (
                    2.0
                    * (speed_ratio - PATROL_SPEED_RATIO_OBSERVATION_BOUNDS[0])
                    / (
                        PATROL_SPEED_RATIO_OBSERVATION_BOUNDS[1]
                        - PATROL_SPEED_RATIO_OBSERVATION_BOUNDS[0]
                    )
                    - 1.0
                ),
            },
        )
        observation.update(
            {
                "goal_entities": self._goal_reach_entity_buffer(
                    positions,
                    headings,
                    self.goal_centers[arena_indices],
                    self.goal_radii[arena_indices],
                    self.goal_active[arena_indices],
                ),
                "goal_mask": self.goal_active[arena_indices].clone(),
                "obstacle_entities": self._goal_reach_entity_buffer(
                    positions,
                    headings,
                    self.obstacle_centers[arena_indices],
                    self.obstacle_radii[arena_indices],
                    self.obstacle_active[arena_indices],
                ),
                "obstacle_mask": self.obstacle_active[arena_indices].clone(),
            }
        )
        enemy_positions = self.learners.position[arena_indices]
        enemy_headings = self.learners.heading[arena_indices]
        enemy_velocity = self.learners.body_velocity[arena_indices]
        enemy_mask = self.learner_active[arena_indices]
        relative_position = self._rotate_geometry(
            enemy_positions - positions[:, None], headings[:, None]
        ) / self.world.radius
        enemy_forward = torch.stack(
            (enemy_headings.sin(), enemy_headings.cos()), dim=-1
        )
        enemy_right = torch.stack(
            (enemy_headings.cos(), -enemy_headings.sin()), dim=-1
        )
        enemy_world_velocity = (
            enemy_forward * enemy_velocity[..., :1]
            + enemy_right * enemy_velocity[..., 1:2]
        )
        enemy_velocity_in_body = self._rotate_geometry(
            enemy_world_velocity, headings[:, None]
        ) / self.learner_dynamics.maximum_surge
        relative_heading = enemy_headings - headings[:, None]
        enemy_features = [
            relative_position,
            enemy_velocity_in_body,
            torch.stack(
                (relative_heading.sin(), relative_heading.cos()), dim=-1
            ),
            enemy_velocity[..., 2:3] / 0.2,
        ]
        if (
            "learner_speed_ratio_normalized"
            in self.patrol_variant.entity_features_for("enemies")
        ):
            minimum, maximum = PATROL_SPEED_RATIO_BOUNDS
            enemy_features.append(
                (
                    2.0
                    * (self.learner_speed_ratio[arena_indices] - minimum)
                    / (maximum - minimum)
                    - 1.0
                ).unsqueeze(-1)
            )
        enemy_entities = torch.cat(
            enemy_features,
            dim=-1,
        ) * enemy_mask.unsqueeze(-1).to(torch.float32)
        observation["enemy_entities"] = enemy_entities
        observation["enemy_mask"] = enemy_mask.clone()

        friendly_positions = self.defenders.position[arena_indices]
        friendly_headings = self.defenders.heading[arena_indices]
        friendly_velocity = self.defenders.body_velocity[arena_indices]
        friendly_relative_position = self._rotate_geometry(
            friendly_positions - positions[:, None], headings[:, None]
        ) / self.world.radius
        friendly_forward = torch.stack(
            (friendly_headings.sin(), friendly_headings.cos()), dim=-1
        )
        friendly_right = torch.stack(
            (friendly_headings.cos(), -friendly_headings.sin()), dim=-1
        )
        friendly_world_velocity = (
            friendly_forward * friendly_velocity[..., :1]
            + friendly_right * friendly_velocity[..., 1:2]
        )
        friendly_velocity_in_body = self._rotate_geometry(
            friendly_world_velocity, headings[:, None]
        ) / speed_cap[:, None, None].clamp_min(1.0e-6)
        friendly_relative_heading = friendly_headings - headings[:, None]
        friendly_mask = self.defender_active[arena_indices].clone()
        if actor_indices.numel():
            friendly_mask[
                torch.arange(actor_indices.numel(), device=self.device), actor_indices
            ] = False
        friendly_entities = torch.cat(
            (
                friendly_relative_position,
                friendly_velocity_in_body,
                torch.stack(
                    (
                        friendly_relative_heading.sin(),
                        friendly_relative_heading.cos(),
                    ),
                    dim=-1,
                ),
                friendly_velocity[..., 2:3]
                / yaw_cap[:, None, None].clamp_min(1.0e-6),
            ),
            dim=-1,
        ) * friendly_mask.unsqueeze(-1).to(torch.float32)
        observation["friendly_entities"] = friendly_entities
        observation["friendly_mask"] = friendly_mask
        return observation

    def observe_defenders(self, arena_index: int) -> dict[str, torch.Tensor]:
        arena_index = int(arena_index)
        actor_indices = torch.nonzero(
            self.defender_active[arena_index], as_tuple=False
        ).flatten()
        return self._defender_pursuit_observation(
            torch.full_like(actor_indices, arena_index), actor_indices
        )

    def training_observation(
        self, mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        return self.training_observation_for(self.training_target, mask)

    def training_observation_for(
        self, actor_kind: str, mask: torch.Tensor | None = None, *, record_neighbors=False, consumer="both"
    ) -> dict[str, torch.Tensor]:
        return self.training_observation_bundle((actor_kind,), mask,
            record_neighbors=record_neighbors, consumer=consumer)[actor_kind]

    def training_observation_bundle(self, roles, mask=None, *, record_neighbors=False, consumer="both"):
        self._validate_observation_consumer(consumer)
        roles = tuple(roles)
        for role in roles:
            if role not in self.policy_roles:
                raise ValueError(f"Role has no policy in this mode: {role}")
        if not roles:
            return {}
        arenas = torch.arange(self.batch_size, device=self.device) if mask is None else mask.nonzero(as_tuple=True)[0]
        context = self._prepare_observation_context(arenas)
        observations = {}
        for role in roles:
            capacity = self.role_state(role).position.shape[1]
            slots = torch.arange(capacity, device=self.device).expand(arenas.numel(), -1)
            rows = torch.arange(arenas.numel(), device=self.device)[:, None].expand_as(slots).reshape(-1)
            flat = self._unified_role_observation(role, arenas[:, None].expand_as(slots).reshape(-1),
                slots.reshape(-1), record_neighbors=record_neighbors, consumer=consumer,
                context=context, context_rows=rows)
            obs = {k: v.reshape(arenas.numel(), capacity, *v.shape[1:]) for k, v in flat.items()}
            obs["actor_alive"] = context[role]["active"].clone()
            obs["actor_active"] = obs["actor_alive"]
            obs["credit_member"] = context[role]["member"].clone()
            observations[role] = obs
        return observations

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state.update(
            {
                "learner_count_range": self.learner_count_range,
                "defender_count_range": self.defender_count_range,
                "learner_speed_ratio_range": self.learner_speed_ratio_range,
                "match_rules_version": 7,
                "terminate_on_defender_elimination": self.terminate_on_defender_elimination,
                "include_remaining_time_observation": "remaining_time" in self.learner_variant.vector_inputs,
                "apply_sampling_time_limit": self.apply_sampling_time_limit,
                "terminate_on_any_casualty": self.terminate_on_any_casualty,
                "reward_objective": self.reward_objective,
                "bootstrap_on_time_limit": self.bootstrap_on_time_limit,
                "learners": {
                    name: value.detach().cpu()
                    for name, value in self.learners.__dict__.items()
                },
                "learner_active": self.learner_active.cpu(),
                "learner_credit_member": self.learner_credit_member.cpu(),
                "learner_status": self.learner_status.cpu(),
                "sampled_learner_count": self.sampled_learner_count.cpu(),
                "sampled_defender_count": self.sampled_defender_count.cpu(),
                "defender_success": self.defender_success.cpu(),
                "defender_collision": self.defender_collision.cpu(),
                "defender_pair_collision": self.defender_pair_collision.cpu(),
                "defender_credit_member": self.defender_credit_member.cpu(),
                "defender_speed_ratio": self.defender_speed_ratio.cpu(),
                "learner_speed_ratio": self.learner_speed_ratio.cpu(),
                "match_outcome": self.match_outcome.cpu(),
                "match_end_reason": self.match_end_reason.cpu(),
            }
        )
        for name in ("contact_static", "contact_pairs", "episode_contact_counts", "episode_pair_contacts"):
            state[name] = getattr(self, name).detach().cpu().clone()
        state["episode_defense_utility"] = self.episode_defense_utility.detach().cpu().clone()
        state["target_mode"] = self.target_mode
        state['target_count_range']=self.target_count_range
        state['sampled_target_count']=self.sampled_target_count.cpu().clone()
        state["targets"] = {name: value.detach().cpu().clone() for name, value in self.targets.__dict__.items()}
        for name in ("active", "credit_member", "collision", "captured", "speed_ratio"):
            state["target_" + name] = getattr(self, "target_" + name).cpu().clone()
        return state

    def load_state_dict(self, state: dict[str, object], *, source_launch_settings=None) -> None:
        self._validate_scene_restore(state)
        for key, expected in (("terminate_on_defender_elimination", self.terminate_on_defender_elimination),
                              ("include_remaining_time_observation", "remaining_time" in self.learner_variant.vector_inputs),
                              ("apply_sampling_time_limit", self.apply_sampling_time_limit)):
            if state.get(key, key != "terminate_on_defender_elimination") != expected:
                raise ValueError(f"Checkpoint {key} mismatch")
        if state.get("match_rules_version") == 6 and state.get("bootstrap_on_time_limit", False):
            raise ValueError("v6 truncation time observation is incompatible with task time semantics")
        if state.get('bootstrap_on_time_limit', False) != self.bootstrap_on_time_limit:
            raise ValueError('Checkpoint time limit boundary mismatch')
        if tuple(state.get('target_count_range',(1,1)))!=self.target_count_range:
            raise ValueError('Checkpoint target count mismatch')
        if state['targets']['position'].shape!=self.targets.position.shape:
            raise ValueError('Checkpoint target shape mismatch')
        self.sampled_target_count=state.get('sampled_target_count',state['target_credit_member'].sum(1)).to(self.device).clone()
        if state.get("match_rules_version") not in (6, 7) or state.get("terminate_on_any_casualty") != self.terminate_on_any_casualty or state.get("reward_objective") != self.reward_objective:
            raise ValueError("Checkpoint match rules mismatch")
        if tuple(state.get("learner_count_range", ())) != self.learner_count_range:
            raise ValueError("Multi GoalReach checkpoint learner count range mismatch")
        checkpoint_defender_count_range = tuple(
            state.get("defender_count_range", (self.defender_count, self.defender_count))
        )
        if checkpoint_defender_count_range != self.defender_count_range:
            raise ValueError("Multi GoalReach checkpoint patrol count range mismatch")
        if (
            tuple(state.get("learner_speed_ratio_range", (1.0, 1.0)))
            != self.learner_speed_ratio_range
        ):
            raise ValueError("Multi GoalReach checkpoint learner speed ratio mismatch")
        self.episode_defense_utility = state["episode_defense_utility"].to(self.device).clone()
        if state["target_mode"] != self.target_mode:
            raise ValueError("Checkpoint target mode mismatch")
        super().load_state_dict(state, source_launch_settings=source_launch_settings)
        # The restored tensors own their exact capacities, including shrinkage.
        self._contact_circle_capacities = (self.obstacle_capacity, self.goal_capacity)
        for name in ("contact_static", "contact_pairs", "episode_contact_counts", "episode_pair_contacts"):
            setattr(self, name, state[name].to(self.device).clone())
        for name, value in state["targets"].items():
            setattr(self.targets, name, value.to(self.device).clone())
        for name in ("active", "credit_member", "collision", "captured", "speed_ratio"):
            setattr(self, "target_" + name, state["target_" + name].to(self.device).clone())
        learners = state.get("learners")
        if not isinstance(learners, dict):
            raise ValueError("Multi GoalReach checkpoint has no learner state")
        for name, value in learners.items():
            setattr(self.learners, name, value.to(self.device))
        self.learner_active = state["learner_active"].to(self.device)
        self.learner_credit_member = (
            state["learner_credit_member"].to(self.device)
            if "learner_credit_member" in state
            else self.learner_active.clone()
        )
        self.learner_status = state["learner_status"].to(self.device)
        self.sampled_learner_count = state["sampled_learner_count"].to(self.device)
        self.sampled_defender_count = (
            state["sampled_defender_count"].to(self.device)
            if "sampled_defender_count" in state
            else torch.full(
                (self.batch_size,),
                self.defender_count,
                dtype=torch.int64,
                device=self.device,
            )
        )
        self.defender_success = (
            state["defender_success"].to(self.device)
            if "defender_success" in state
            else self.success[:, None].expand(-1, self.defender_count).clone()
        )
        self.defender_collision = (
            state["defender_collision"].to(self.device)
            if "defender_collision" in state
            else self.collision[:, None].expand(-1, self.defender_count).clone()
        )
        self.defender_pair_collision = (
            state["defender_pair_collision"].to(self.device)
            if "defender_pair_collision" in state
            else torch.zeros_like(self.defender_collision)
        )
        self.defender_credit_member = (
            state["defender_credit_member"].to(self.device)
            if "defender_credit_member" in state
            else self.defender_active.clone()
        )
        defender_speed_ratio = state["defender_speed_ratio"].to(self.device)
        self.defender_speed_ratio = (
            defender_speed_ratio[:, None]
            .expand(-1, self.defender_count)
            .clone()
            if defender_speed_ratio.ndim == 1
            else defender_speed_ratio
        )
        learner_speed_ratio = state.get("learner_speed_ratio")
        if learner_speed_ratio is None:
            self.learner_speed_ratio = torch.ones_like(self.learner_active, dtype=torch.float32)
        else:
            learner_speed_ratio = learner_speed_ratio.to(self.device)
            self.learner_speed_ratio = (
                learner_speed_ratio[:, None].expand_as(self.learner_active).clone()
                if learner_speed_ratio.ndim == 1
                else learner_speed_ratio
            )
        self.match_outcome = (
            state["match_outcome"].to(self.device)
            if "match_outcome" in state
            else torch.full(
                (self.batch_size,),
                int(MatchOutcome.IN_PROGRESS),
                dtype=torch.int8,
                device=self.device,
            )
        )
        self.match_end_reason = (
            state["match_end_reason"].to(self.device)
            if "match_end_reason" in state
            else torch.full(
                (self.batch_size,),
                MATCH_END_REASON_CODES[MatchEndReason.NONE],
                dtype=torch.int8,
                device=self.device,
            )
        )
        self._bind_primary_learner()

    def defender_defense_rate(self) -> torch.Tensor:
        return 1.0 - self.defender_leakage_rate()

    def defender_leakage_rate(self) -> torch.Tensor:
        resolved = self.learner_credit_member & (self.learner_status != 0)
        total = resolved.sum(dim=1, dtype=torch.float32)
        reached = (self.learner_status == 1).sum(dim=1, dtype=torch.float32)
        return torch.where(total > 0, reached / total, torch.zeros_like(total))

    @staticmethod
    def _swept_circle_time(q, d, radius, *, inside=False):
        """First valid circle contact along the segment, or infinity.

        Existing overlap is an immediate lethal contact. Interior constraints
        use the relative endpoint, including a moving engagement boundary.
        """
        return MultiGoalReachEnv._swept_circle_time_fields(
            q, d, radius ** 2,
            radius - CONTACT_TOLERANCE if inside else radius + CONTACT_TOLERANCE, inside)

    @staticmethod
    def _swept_circle_time_fields(q, d, radius_squared, contact_limit, inside):
        """One quadratic, with scalar or candidate-column interior constraints."""
        if (q.device.type == 'cpu' and isinstance(inside, bool)
                and isinstance(radius_squared, torch.Tensor) == isinstance(contact_limit, torch.Tensor)):
            from .contact_cpu import swept_scalar, swept_tensor
            sweep = swept_tensor if isinstance(radius_squared, torch.Tensor) else swept_scalar
            return sweep(q, d, radius_squared, contact_limit, inside, 1 + CONTACT_TOLERANCE)
        a = d.square().sum(-1)
        b = (q * d).sum(-1)
        c = q.square().sum(-1) - radius_squared
        disc = b.square() - a * c
        root = disc.clamp_min(0).sqrt()
        moving = a > 1.0e-16
        if isinstance(inside, torch.Tensor):
            alpha = (-b + torch.where(inside, root, -root)) / a.clamp_min(1.0e-16)
            distance = q.norm(dim=-1)
            touching = torch.where(inside, distance >= contact_limit, distance <= contact_limit)
            endpoint = q + d
            interior = moving & (endpoint.norm(dim=-1) >= contact_limit) & ((endpoint * d).sum(-1) > 0)
            exterior = moving & (disc >= 0) & (b < -1.0e-12) & (alpha >= 0)
            alpha = torch.where(touching, 0.0, torch.where(inside, alpha.clamp_min(0), alpha))
            valid = torch.where(inside, interior, exterior) | touching
        elif inside:
            alpha = (-b + root) / a.clamp_min(1.0e-16)
            endpoint = q + d
            valid = moving & (endpoint.norm(dim=-1) >= contact_limit) & ((endpoint * d).sum(-1) > 0)
            outside = q.norm(dim=-1) >= contact_limit
            alpha = torch.where(outside, 0.0, alpha.clamp_min(0))
            valid |= outside
        else:
            alpha = (-b - root) / a.clamp_min(1.0e-16)
            touching = q.norm(dim=-1) <= contact_limit
            entering = b < -1.0e-12
            valid = moving & (disc >= 0) & entering & (alpha >= 0)
            alpha = torch.where(touching, 0.0, alpha)
            valid |= touching
        return torch.where(valid & (alpha <= 1 + CONTACT_TOLERANCE), alpha.clamp(0, 1), torch.inf)

    def _multi_collection_contact_matrix(self, previous, desired, centers, radii, active):
        return torch.isfinite(self._static_contact_times(previous, desired-previous, centers,
                                                        radii+self.vessel_radius_m, active))

    def _obstacle_neighbor_acceleration_enabled(self) -> bool:
        return self.device.type == 'cuda'

    @staticmethod
    def _dense_contact_geometry(position, delta, centers, radii, active, live):
        valid = active[:, None] if live is None else active[:, None] & live[:, :, None]
        return position[:, :, None] - centers[:, None], delta[:, :, None], radii[:, None], valid

    def _prepare_static_contact_candidates(self, position, delta, centers, radii, active, live, *, rows=None, safe=None):
        """Prepare narrow and unsafe geometry without evaluating any circle times."""
        count, actors = position.shape[:2]
        capacity = centers.shape[1]
        cache = getattr(self, '_obstacle_neighbors', None)
        cached = cache is not None and safe is not None
        index = torch.empty((count, actors, 0), dtype=torch.int64, device=self.device)
        mask = torch.empty_like(index, dtype=torch.bool)
        q = position.new_empty((count, actors, 0, 2))
        r = position.new_empty((count, actors, 0))
        if cached:
            index = cache['index'] if rows is None else cache['index'][rows]
            valid = cache['valid'] if rows is None else cache['valid'][rows]
            mask = cache['mask'] if rows is None else cache['mask'][rows]
            allowed = safe[:, None] if live is None else safe[:, None] & live
            torch._assert_async((valid | ~allowed).all(), 'Invalid obstacle cache selected')
            mask = mask & valid[..., None] & allowed[..., None]
            index = torch.where(mask, index, 0)
            if capacity == 0:
                index, mask = index[..., :0], mask[..., :0]
            if index.shape[-1]:
                global_rows = torch.arange(count, device=self.device) if rows is None else rows
                q = position[:, :, None] - centers[global_rows[:, None, None], index]
                r = radii[global_rows[:, None, None], index]
            dense_rows = (~safe).nonzero(as_tuple=True)[0]
        else:
            dense_rows = torch.arange(count, device=self.device)
        dense_q = position.new_empty((dense_rows.numel(), actors, capacity, 2))
        dense_d = delta[:0, :, None]
        dense_r = radii[:0, None]
        dense_mask = active[:0, None]
        if dense_rows.numel():
            global_rows = rows if not cached else dense_rows if rows is None else rows[dense_rows]
            c = centers if global_rows is None else centers[global_rows]
            dense_r = radii if global_rows is None else radii[global_rows]
            enabled = active if global_rows is None else active[global_rows]
            torch._assert_async(((torch.isfinite(c).all(-1) & torch.isfinite(dense_r) & (dense_r > 0)) | ~enabled).all(), 'Nonfinite collision world geometry')
            p, d = (position, delta) if not cached else (position[dense_rows], delta[dense_rows])
            live_rows = live if not cached or live is None else live[dense_rows]
            dense_q, dense_d, dense_r, dense_mask = self._dense_contact_geometry(p, d, c, dense_r, enabled, live_rows)
        probe = getattr(self, '_training_memory_probe', None)
        if probe is not None and probe.update <= probe.maximum_updates and hasattr(self, '_collision_neighbor_counts'):
            self._collision_neighbor_counts[3] += count * actors * index.shape[-1]
            self._collision_neighbor_counts[4] += dense_rows.numel() * actors * capacity
        return cached, q, r, index, mask, dense_rows, dense_q, dense_d, dense_r, dense_mask

    def _static_contact_candidates(self, position, delta, centers, radii, active, live, *, rows=None, safe=None):
        """Single-family consumer of the same prepared geometry as the joint solver."""
        started = self._neighbor_timing_start()
        _, q, r, index, mask, dense_rows, dq, dd, dr, dm = self._prepare_static_contact_candidates(
            position, delta, centers, radii, active, live, rows=rows, safe=safe)
        minimum = position.new_full(position.shape[:2], torch.inf)
        narrow = position.new_empty(index.shape)
        if index.shape[-1]:
            narrow = torch.where(mask, self._swept_circle_time(q, delta[:, :, None], r), torch.inf)
            minimum = narrow.amin(-1)
        dense = position.new_empty(dq.shape[:-1])
        if dense_rows.numel():
            dense = torch.where(dm, self._swept_circle_time(dq, dd, dr), torch.inf)
            if centers.shape[1]:
                minimum[dense_rows] = dense.amin(-1)
        self._neighbor_timing_end('static_collision', started)
        return minimum, narrow, index, mask, dense_rows, dense

    def _joint_contact_times(self, position, delta, centers, radii, active, live, *, rows=None, safe=None):
        """Pack static, pair, and moving-boundary geometry on the candidate axis."""
        started = self._neighbor_timing_start()
        cached, sq, sr, index, mask, dense_rows, dq, dd, dr, dm = self._prepare_static_contact_candidates(
            position, delta, centers, radii, active, live, rows=rows, safe=safe)
        count, actors = position.shape[:2]
        end = self.defender_count + self.viewer_learner_capacity
        if not cached:
            sq, sr = dq, dr
        width = sq.shape[2]
        shape = (count, actors, width)
        pair_radius = 2 * self.vessel_radius_m
        # Derive Python scalar fields before casting, preserving the old rounding.
        signature = (position.device, position.dtype, actors, width, self._contact_geometry_signature)
        if signature != getattr(self, '_contact_fields_signature', None):
            self._contact_fields_signature = signature
            self._contact_inside = torch.arange(width + actors + 2, device=position.device) >= width + actors
            self._contact_pair_squared = position.new_tensor(pair_radius ** 2)
            self._contact_pair_limit = position.new_tensor(pair_radius + CONTACT_TOLERANCE)
        pair_q = position[:, :, None] - position[:, None]
        pair_d = delta[:, :, None] - delta[:, None]
        boundary_q = (position[:, :, None] - position[:, end:end+1, None]).expand(-1, -1, 2, -1)
        boundary_d = (delta[:, :, None] - delta[:, end:end+1, None]).expand(-1, -1, 2, -1)
        q = torch.cat((sq, pair_q, boundary_q), 2)
        d = torch.cat((delta[:, :, None].expand(*shape, 2), pair_d, boundary_d), 2)
        br = self._contact_boundary_radii
        r2 = torch.cat(((sr ** 2).expand(shape), self._contact_pair_squared.expand(count, actors, actors),
                        (br ** 2).expand(count, actors, 2)), 2)
        limit = torch.cat(((sr + CONTACT_TOLERANCE).expand(shape), self._contact_pair_limit.expand(count, actors, actors),
                           (br - CONTACT_TOLERANCE).expand(count, actors, 2)), 2)
        times = self._swept_circle_time_fields(q, d, r2, limit, self._contact_inside)
        static, pair, boundary = times.split((width, actors, 2), dim=2)
        minimum = position.new_full((count, actors), torch.inf)
        if cached:
            narrow = torch.where(mask, static, torch.inf)
            if width:
                minimum = narrow.amin(-1)
            dense = position.new_empty(dq.shape[:-1])
            if dense_rows.numel():
                dense = torch.where(dm, self._swept_circle_time(dq, dd, dr), torch.inf)
                if centers.shape[1]:
                    minimum[dense_rows] = dense.amin(-1)
        else:
            narrow = position.new_empty(index.shape)
            dense = torch.where(dm, static, torch.inf)
            if width:
                minimum = dense.amin(-1)
        self._neighbor_timing_end('joint_collision_queries', started)
        return minimum, narrow, index, mask, dense_rows, dense, boundary, pair

    def _static_contact_times(self, position, delta, centers, radii, active, live=None, *, rows=None, safe=None):
        """Materialize the full matrix only for callers that explicitly need it."""
        if not self._obstacle_neighbor_acceleration_enabled():
            q, d, r, valid = self._dense_contact_geometry(position, delta, centers, radii, active, live)
            return torch.where(valid, self._swept_circle_time(q, d, r), torch.inf)
        _, narrow, index, _, dense_rows, dense = self._static_contact_candidates(
            position, delta, centers, radii, active, live, rows=rows, safe=safe)
        if getattr(self, '_obstacle_neighbors', None) is None or safe is None:
            return dense
        result = position.new_full((*position.shape[:2], centers.shape[1]), torch.inf)
        if index.shape[-1]:
            result.scatter_reduce_(2, index, narrow, reduce='amin', include_self=True)
        result[dense_rows] = dense
        return result

    def _multi_collection_contact(self, previous, desired, centers, radii, active):
        return self._multi_collection_contact_matrix(previous, desired, centers, radii, active).any(-1)

    def _actor_pair_contact_matrix(self, previous, desired, active):
        d = desired - previous
        alpha = self._swept_circle_time(previous[:, :, None] - previous[:, None],
                                        d[:, :, None] - d[:, None], 2 * self.vessel_radius_m)
        contact = torch.isfinite(alpha) & active[:, :, None] & active[:, None]
        contact.diagonal(dim1=-2, dim2=-1).fill_(False)
        return contact

    def _learner_pair_contact(self, previous, desired, active):
        return self._actor_pair_contact_matrix(previous, desired, active).any(-1)

    def _prepare_contact_geometry(self):
        nd, na = self.defender_count, self.viewer_learner_capacity
        end = nd + na
        n = self.contact_pairs.shape[-1]
        safety = self.config.goal_reach["safety_zones"]
        signature = (nd, na, self.target_capacity, self.world.radius, self.vessel_radius_m,
                     bool(safety.get("enabled", False)), float(safety["engagement_radius_m"]))
        if signature == getattr(self, "_contact_geometry_signature", None):
            return
        self._contact_geometry_signature = signature
        self._contact_boundary_radii = torch.tensor(
            [self.world.radius, signature[-1] - self.vessel_radius_m], device=self.device)
        self._contact_boundary_enabled = torch.zeros((n, 2), dtype=torch.bool, device=self.device)
        self._contact_boundary_enabled[nd:end, 0] = True
        self._contact_boundary_enabled[:nd, 1] = signature[-2]
        pairs = torch.zeros((n, n), dtype=torch.bool, device=self.device)
        actor_roles = ("defender",)*nd + ("learner",)*na + ("target",)*self.target_capacity
        for capturer, prey in CAPTURE_RELATIONS:
            left = torch.tensor([role==capturer for role in actor_roles],device=self.device)
            right = torch.tensor([role==prey for role in actor_roles],device=self.device)
            pairs |= left[:,None] & right[None,:]
        self._contact_capture_pairs = pairs | pairs.T
        self._contact_upper = torch.ones((n, n), dtype=torch.bool, device=self.device).triu(1)

    def _resolve_contacts(self, state, previous, active):
        """Resolve ordered contacts on the device; compact only lanes with residual motion."""
        self._last_contacts_were_empty = False
        nd, na = self.defender_count, self.viewer_learner_capacity
        end = nd + na
        n = active.shape[1]
        centers = self.obstacle_centers
        radii = self.obstacle_radii + self.vessel_radius_m
        enabled = self.obstacle_active
        pos, delta = previous.clone(), state.position - previous
        neighbor_safe = (self._obstacle_neighbor_safe(previous,state.position,active)
                         if self._obstacle_neighbor_acceleration_enabled() else None)
        alive = active.clone()
        static_starts = torch.zeros_like(self.contact_static, dtype=torch.int32)
        pair_starts = torch.zeros_like(self.contact_pairs, dtype=torch.int32)
        captured = torch.zeros_like(self.learner_active)
        reached = torch.zeros_like(captured)
        assignments = torch.zeros((self.batch_size, nd, na), dtype=torch.bool, device=self.device)
        target_contacts = torch.zeros((self.batch_size, na, self.target_capacity), dtype=torch.bool, device=self.device)
        # Only these masks/radii depend on configuration; moving TGT geometry stays per tick.
        self._prepare_contact_geometry()
        boundary_radii = self._contact_boundary_radii
        boundary_enabled = self._contact_boundary_enabled
        capture_pairs = self._contact_capture_pairs
        upper = self._contact_upper
        dense_cpu = self.device.type == 'cpu' and neighbor_safe is None and centers.shape[1] > 0

        rows = torch.arange(self.batch_size, device=self.device)
        for iteration in range(2 * n + 1):
            if rows.numel() == 0:
                break
            p, d, live = (pos, delta, alive) if iteration == 0 else (pos[rows], delta[rows], alive[rows])
            if dense_cpu:
                from .contact_cpu import dense_first
                r = (radii if iteration == 0 else radii[rows])[:, None]
                pair_radius = 2 * self.vessel_radius_m
                st, pt, first = dense_first(p, d,
                    centers if iteration == 0 else centers[rows],
                    enabled if iteration == 0 else enabled[rows], live,
                    r ** 2, r + CONTACT_TOLERANCE,
                    boundary_radii ** 2, boundary_radii - CONTACT_TOLERANCE,
                    boundary_enabled, upper, end,
                    pair_radius ** 2, pair_radius + CONTACT_TOLERANCE, 1 + CONTACT_TOLERANCE)
            elif self.device.type == 'cuda':
                joint = self._joint_contact_times(p, d, centers, radii, enabled, live,
                    rows=None if iteration == 0 else rows,
                    safe=None if neighbor_safe is None else neighbor_safe if iteration == 0 else neighbor_safe[rows])
                candidates, bt, pt = joint[:6], joint[6], joint[7]
                ct = candidates[5]
            else:
                # CPU packing had no consistent end-to-end gain in the fixed NN pairs.
                if neighbor_safe is not None:
                    candidates = self._static_contact_candidates(p, d, centers, radii, enabled, live,
                        rows=None if iteration == 0 else rows,
                        safe=neighbor_safe if iteration == 0 else neighbor_safe[rows])
                else:
                    ct = self._static_contact_times(p, d,
                        centers if iteration == 0 else centers[rows],
                        radii if iteration == 0 else radii[rows],
                        enabled if iteration == 0 else enabled[rows], live)
                boundary_q = (p[:, :, None] - p[:, end:end+1, None]).expand(-1, -1, 2, -1)
                boundary_delta = (d[:, :, None] - d[:, end:end+1, None]).expand(-1, -1, 2, -1)
                bt = self._swept_circle_time(boundary_q, boundary_delta, boundary_radii, inside=True)
                pt = self._swept_circle_time(p[:, :, None] - p[:, None], d[:, :, None] - d[:, None], 2 * self.vessel_radius_m)
            if not dense_cpu:
                bt = torch.where(boundary_enabled & live[:, :, None], bt, torch.inf)
                if neighbor_safe is None:
                    st = torch.cat((ct, bt), -1)
                pt = torch.where(live[:, :, None] & live[:, None] & upper, pt, torch.inf)
                static_first = (torch.minimum(candidates[0].amin(1), bt.flatten(1).amin(1))
                                if neighbor_safe is not None else st.flatten(1).amin(1))
                first = torch.minimum(static_first, pt.flatten(1).amin(1))
            has = torch.isfinite(first)
            fraction = first.clamp(max=1)
            if self.device.type == 'cpu' and self.batch_size == 1 and iteration == 0 and not has[0]:
                # Same contact calculation and position expression; there is no
                # residual lane to scatter, compact, or test again this tick.
                state.position = p + fraction[:, None, None] * d
                self._last_contacts_were_empty = True
                return static_starts, pair_starts.sum(-1), captured, reached, assignments, target_contacts
            pos[rows] = p + fraction[:, None, None] * d
            delta[rows] = d * (1 - fraction[:, None, None])
            event_indices = has.nonzero(as_tuple=True)[0]
            if event_indices.numel() == 0:
                rows = rows[:0]
                if iteration == 0:
                    state.position = pos
                    return static_starts, pair_starts.sum(-1), captured, reached, assignments, target_contacts
                break
            rows = rows[event_indices]
            pt, first = pt[event_indices], first[event_indices]
            # Equal-time tolerance applies only after finding the earliest physical contact.
            if neighbor_safe is None:
                st = st[event_indices]
                sh = torch.isfinite(st) & ((st - first[:, None, None]).abs() <= CONTACT_TOLERANCE)
            else:
                _, narrow, index, mask, dense_rows, dense = candidates
                sh = torch.zeros((event_indices.numel(), n, centers.shape[1]+2), dtype=torch.int32, device=self.device)
                if index.shape[-1]:
                    alpha = narrow[event_indices]
                    hit = mask[event_indices] & torch.isfinite(alpha) & ((alpha-first[:, None, None]).abs() <= CONTACT_TOLERANCE)
                    sh.scatter_add_(2, index[event_indices], hit.to(torch.int32))
                if dense_rows.numel():
                    lookup = torch.full((p.shape[0],), -1, dtype=torch.int64, device=self.device)
                    lookup[dense_rows] = torch.arange(dense_rows.numel(), device=self.device)
                    selected = lookup[event_indices]
                    alpha = dense[selected.clamp_min(0)]
                    hit = (selected[:, None, None] >= 0) & torch.isfinite(alpha) & ((alpha-first[:, None, None]).abs() <= CONTACT_TOLERANCE)
                    sh[..., :-2] |= hit.to(torch.int32)
                alpha = bt[event_indices]
                sh[..., -2:] = (torch.isfinite(alpha) & ((alpha-first[:, None, None]).abs() <= CONTACT_TOLERANCE)).to(torch.int32)
                sh = sh > 0
            ph = torch.isfinite(pt) & ((pt - first[:, None, None]).abs() <= CONTACT_TOLERANCE)
            intercept = ph[:, :nd, nd:end]
            intercept &= intercept.to(torch.int32).cumsum(1) == 1
            caught = intercept.any(1)
            captured[rows] |= caught
            assignments[rows] |= intercept
            alive[rows, nd:end] &= ~caught
            target_hit = ph[:, nd:end, end:] & ~caught[:,:,None]
            target_hit &= target_hit.to(torch.int32).cumsum(1)==1
            target_contacts[rows] |= target_hit
            target_captured = target_hit.any(1)
            won = target_captured.any(1)
            alive[rows, end:] &= ~target_captured
            # Capture removes the caught actor from simultaneous ordinary contacts.
            ordinary_pair = ph & ~capture_pairs & alive[rows, :, None] & alive[rows, None, :]
            sh &= alive[rows, :, None]
            static_starts[rows] += sh.to(torch.int32)
            pair_starts[rows] += ordinary_pair.to(torch.int32)
            self.contact_static[rows] |= sh
            self.contact_pairs[rows] |= ordinary_pair
            dead = (sh.any(-1) | ordinary_pair.any(-1) | ordinary_pair.any(-2)) & alive[rows]
            alive[rows] &= ~dead
            ended = won | ~alive[rows, nd:end].any(1)
            ended |= dead[:, end:].any(1)
            ended |= self.terminate_on_defender_elimination & ~alive[rows, :nd].any(1)
            if self.terminate_on_any_casualty:
                ended |= caught.any(1) | dead.any(1)
            if self._viewer_event_tick is not None:
                # Freeze the contact pose now; capturers may move on when casualty is off.
                event_pair = ordinary_pair.any(-1) | ordinary_pair.any(-2)
                event_state = AuvState(**{**state.__dict__, "position": pos})
                self._capture_viewer_events(event_state,
                    sh[:, nd:end, -2:].any(-1), sh[:, nd:end, :-2].any(-1),
                    event_pair[:, nd:end], caught, intercept,
                    sh[:, :nd, -2:].any(-1), sh[:, :nd, :-2].any(-1), event_pair[:, :nd], rows=rows)
                slot = 2 * na + nd
                target_event = dead[:, end:] | target_captured
                lane, target_id = target_event.nonzero(as_tuple=True)
                event_rows = rows[lane]
                event_slots = slot + 2*target_id
                code = torch.where(target_captured,4,torch.where(event_pair[:,end:],3,
                    torch.where(sh[:,end:,:-2].any(-1),2,1))).to(torch.int8)
                self._viewer_event_tick[event_rows,event_slots] = self.physics_ticks[event_rows].long()+1
                self._viewer_event_code[event_rows,event_slots] = code[lane,target_id]
                self._viewer_event_pose[event_rows,event_slots] = torch.cat(
                    (pos[event_rows,end+target_id],state.heading[event_rows,end+target_id,None]),-1)
                lane,capturer,target_id = target_hit.nonzero(as_tuple=True)
                won_rows = rows[lane]
                event_slots = slot+2*target_id
                self._viewer_event_actor_id[won_rows,event_slots] = capturer
                self._viewer_event_actor_id[won_rows,event_slots+1] = capturer
                self._viewer_event_tick[won_rows,event_slots+1] = self.physics_ticks[won_rows].long()+1
                self._viewer_event_code[won_rows,event_slots+1] = 4
                self._viewer_event_pose[won_rows,event_slots+1] = torch.cat(
                    (pos[won_rows,nd+capturer],state.heading[won_rows,nd+capturer,None]),-1)
            delta[rows] = torch.where((~alive[rows] | ended[:, None])[:, :, None], 0.0, delta[rows])
            # Unchanged/free lanes are already complete; only collision-altered paths recur.
            rows = rows[~ended & (delta[rows].square().sum((1, 2)) > 1.0e-16)]
        if rows.numel():
            raise RuntimeError("Contact solver exceeded its 2N+1 progress bound")
        state.position = pos
        self.learner_dynamics.reconcile_planar_constraint(state, pos, active & ~alive)
        pair_actor = pair_starts.sum(-1) + pair_starts.sum(-2)
        self.episode_contact_counts[..., 0] += static_starts.sum(-1)
        self.episode_contact_counts[..., 1] += pair_actor
        self.episode_pair_contacts += pair_starts.sum((1, 2))
        return static_starts, pair_actor, captured, reached, assignments, target_contacts


    @property
    def applies_time_limit(self):
        return not self.bootstrap_on_time_limit or self.apply_sampling_time_limit

    def _advance_joint_ticks(self, actor_state, limits, tick_count, totals, *,
                             physics_deadline=None, finish_decision=True, viewer_cpu_chunk=False):
        """Single ordered physics/reward executor shared by complete and partial steps."""
        learner_end = self.defender_count + self.viewer_learner_capacity
        dynamic = self.target_mode == 'dynamic'
        maximum_ticks = int(self.config.goal_reach['clock']['maximum_physics_ticks'])
        mixed_speed_cap, mixed_yaw_cap, mixed_thruster_limit = limits
        static_counts = totals['static_counts']
        pair_counts = totals['pair_counts']
        learner_reach_event = totals['learner_reach_event']
        learner_capture_event = totals['learner_capture_event']
        target_capture_total = totals['target_capture_total']
        scene_done_now = totals['scene_done_now']
        attacker_win = totals['attacker_win']
        defender_win = totals['defender_win']
        scene_horizon_terminal = totals['scene_horizon_terminal']
        scene_fleet_success_terminal = totals['scene_fleet_success_terminal']
        defense_utility = totals['defense_utility']
        for tick_index in range(tick_count):
            if physics_deadline is not None and tick_index and time.perf_counter() >= physics_deadline:
                finish_decision = False
                break
            scene_active = ~self.done
            active = torch.cat([getattr(self, role + "_active") for role in ("defender", "learner", "target")], 1)
            active &= scene_active[:, None]
            previous = actor_state.position.clone()
            move_active = active.clone()
            if not dynamic:
                move_active[:, learner_end:] = False
            self.learner_dynamics.step(actor_state, move_active, maximum_surge=mixed_speed_cap,
                                      maximum_yaw=mixed_yaw_cap, horizontal_thruster_limit=mixed_thruster_limit)
            static, pairs, captured, reached, assignments, target_contacts = self._resolve_contacts(actor_state, previous, active)
            if (viewer_cpu_chunk and scene_active[0] and self.learner_active.any()
                    and (not self.applies_time_limit or self.physics_ticks[0] + 1 < maximum_ticks)
                    and (self._last_contacts_were_empty or not
                         (static.any() or pairs.any() or captured.any() or reached.any() or target_contacts.any()))):
                self.physics_ticks.add_(scene_active)
                if self.reward_objective == 'protection_time':
                    tick_utility = scene_active.to(torch.float32) / maximum_ticks
                    defense_utility += tick_utility
                    self.episode_defense_utility += tick_utility
                continue
            static_counts += static
            pair_counts += pairs
            learner_capture_event |= captured
            learner_reach_event |= reached
            collision = static.any(-1) | (pairs > 0)
            learner_dead, defender_dead = collision[:, self.defender_count:learner_end], collision[:, :self.defender_count]
            self.learner_active &= ~(captured | reached | learner_dead)
            self.learner_status[learner_dead] = 2
            self.learner_status[static[:, self.defender_count:learner_end, -2] > 0] = 6
            self.learner_status[pairs[:, self.defender_count:learner_end] > 0] = 5
            self.learner_status[captured] = 4
            self.learner_status[reached] = 1
            self.defender_active &= ~defender_dead
            self.defender_collision |= defender_dead
            self.defender_pair_collision |= pairs[:, :self.defender_count] > 0
            self.collision |= collision.any(1)
            defense_dead = defender_dead.any(1)
            target_capture = target_contacts.any(1)
            target_capture_total |= target_capture
            self.target_captured |= target_capture
            self.target_collision |= collision[:,learner_end:]
            self.target_active &= ~(target_capture | collision[:,learner_end:])
            breach = target_capture.any(1)
            target_casualty = collision[:,learner_end:].any(1) & ~breach
            target_lost = breach | target_casualty
            attacker_casualty = (captured | learner_dead).any(1) & self.terminate_on_any_casualty & ~target_lost
            defender_casualty = defense_dead & self.terminate_on_any_casualty & ~target_lost & ~attacker_casualty
            eliminated = ~self.learner_active.any(1) & scene_active & ~target_lost & ~attacker_casualty & ~defender_casualty
            defender_eliminated = self.terminate_on_defender_elimination & ~self.defender_active.any(1) & scene_active & ~target_lost & ~attacker_casualty & ~defender_casualty & ~eliminated
            self.physics_ticks.add_(scene_active)
            timeout = self.applies_time_limit & ~defender_eliminated & scene_active & (self.physics_ticks >= maximum_ticks) & ~target_lost & ~attacker_casualty & ~defender_casualty & ~eliminated
            won = target_lost | defender_casualty | defender_eliminated
            lost = attacker_casualty | eliminated
            if not self.bootstrap_on_time_limit:
                lost |= timeout
            ended = won | lost | timeout
            if self.reward_objective == "protection_time":
                tick_utility = scene_active.to(torch.float32) / maximum_ticks
                terminal_utility = lost * (maximum_ticks - self.physics_ticks).clamp_min(0) / maximum_ticks
                defense_utility += tick_utility
                defense_utility += terminal_utility
                self.episode_defense_utility += tick_utility
                self.episode_defense_utility += terminal_utility
            else:
                tick_utility = lost.to(torch.float32) - won.to(torch.float32)
                if self.reward_objective == 'outcome_time':
                    fraction = self.physics_ticks.to(torch.float32) / maximum_ticks
                    tick_utility = tick_utility * (1.0 - OUTCOME_TIME_COEFFICIENT * fraction)
                defense_utility += tick_utility
                self.episode_defense_utility += tick_utility
            # Episode totals follow physical ticks, independent of Viewer chunks.
            self.done |= ended
            scene_done_now |= ended
            attacker_win |= won
            defender_win |= lost
            scene_horizon_terminal |= timeout
            scene_fleet_success_terminal |= eliminated
            self.match_outcome[won] = int(MatchOutcome.ATTACKER_WIN)
            self.match_outcome[lost] = int(MatchOutcome.DEFENDER_WIN)
            self.match_end_reason[breach] = MATCH_END_REASON_CODES[MatchEndReason.TARGET_CAPTURE]
            self.match_end_reason[target_casualty] = MATCH_END_REASON_CODES[MatchEndReason.TARGET_CASUALTY]
            self.match_end_reason[attacker_casualty] = MATCH_END_REASON_CODES[MatchEndReason.ATTACKER_CASUALTY]
            self.match_end_reason[defender_casualty] = MATCH_END_REASON_CODES[MatchEndReason.DEFENDER_CASUALTY]
            self.match_end_reason[defender_eliminated] = MATCH_END_REASON_CODES[MatchEndReason.DEFENDER_ELIMINATION]
            self.match_end_reason[eliminated] = MATCH_END_REASON_CODES[MatchEndReason.ELIMINATION]
            self.match_end_reason[timeout] = MATCH_END_REASON_CODES[
                MatchEndReason.TRUNCATION if self.bootstrap_on_time_limit else MatchEndReason.HORIZON
            ]
            if viewer_cpu_chunk and self.done[0]:
                if tick_index + 1 < tick_count:
                    actor_state.position = actor_state.position + 0.0
                break
        return finish_decision

    def _step_joint_match(
        self,
        *,
        defender_action: torch.Tensor | None = None,
        target_action: torch.Tensor | None = None,
        learner_action: torch.Tensor | None = None,
        build_observation: bool = True,
        physics_ticks: int | None = None,
        finish_decision: bool = True,
        return_both: bool = False,
        physics_deadline: float | None = None,
    ) -> GoalReachStep | JointGoalReachStep:
        terrain_refresh = torch.nonzero(
            self.physics_ticks
            % int(self.config.goal_reach["clock"]["physics_ticks_per_decision"])
            == 0,
            as_tuple=False,
        ).flatten()
        self._refresh_terrain(terrain_refresh)
        learner_end = self.defender_count + self.viewer_learner_capacity
        if self.target_mode == "dynamic":
            torch._assert_async((~self.target_credit_member | self.target_active | self.target_captured | self.target_collision).all(), "TGT disappeared without a terminal event")
        dynamic = self.target_mode == "dynamic"
        cpu_limits = self._cpu_joint_limits() if self.device.type == "cpu" else None
        role_limits = cpu_limits[0] if cpu_limits is not None else tuple(
            self.patrol_limits(ratio) for ratio in
            (self.defender_speed_ratio, self.learner_speed_ratio, self.target_speed_ratio))
        target_speed, target_yaw, target_thruster = role_limits[2]
        if dynamic:
            if target_action is None:
                if self._target_action_provider is None:
                    raise RuntimeError("Dynamic match requires a target policy")
                target_action = self._target_action_provider(self.training_observation_for("target"))
            target_action = target_action.to(self.device).clamp(-1.0, 1.0)
            if target_action.shape != (self.batch_size, self.target_capacity, 2):
                raise ValueError("Target action must have shape [environments,target_capacity,2]")
            self.learner_dynamics.set_command(self.targets,
                0.5 * (target_action[..., 0] + 1.0) * target_speed,
                target_action[..., 1] * target_yaw, maximum_surge=target_speed, maximum_yaw=target_yaw)
        if defender_action is None:
            if self._defender_action_provider is None:
                raise RuntimeError("Unified match requires a configured defender policy")
            defender_observation = self.training_observation_for("defender")
            defender_action = self._defender_action_provider(defender_observation)
        action = defender_action.to(device=self.device, dtype=torch.float32)
        single_defender = self.defender_count == 1
        if single_defender and action.shape == (self.batch_size, 2):
            defender_action = action[:, None]
        elif action.shape == (self.batch_size, self.defender_count, 2):
            defender_action = action
        else:
            expected = (
                f"[{self.batch_size},2]"
                if single_defender
                else f"[{self.batch_size},{self.defender_count},2]"
            )
            raise ValueError(f"Expected action {expected}, got {tuple(action.shape)}")
        if learner_action is None:
            if self._learner_action_provider is None:
                raise RuntimeError("Patrol pursuit requires a configured infiltrator policy")
            learner_action = self._learner_action_provider(
                self._learner_training_observation()
            )
        learner_action = learner_action.to(device=self.device, dtype=torch.float32)
        expected_learner_shape = (
            self.batch_size,
            self.viewer_learner_capacity,
            2,
        )
        if learner_action.shape != expected_learner_shape:
            raise ValueError(
                f"Expected infiltrator action {list(expected_learner_shape)}, "
                f"got {tuple(learner_action.shape)}"
            )
        self._set_learner_commands(learner_action, role_limits[1])
        defender_speed_cap, defender_yaw_cap, _ = role_limits[0]
        self.defender_dynamics.set_command(
            self.defenders,
            0.5 * (defender_action[..., 0] + 1.0) * defender_speed_cap,
            defender_yaw_cap * defender_action[..., 1],
            maximum_surge=defender_speed_cap,
            maximum_yaw=defender_yaw_cap,
        )
        if self.device.type == "cpu":
            actor_state = self._cpu_joint_state()
            mixed_speed_cap, mixed_yaw_cap, mixed_thruster_limit = cpu_limits[1]
        else:
            actor_state = self._joint_state()
            mixed_speed_cap, mixed_yaw_cap, mixed_thruster_limit = (
                torch.cat(values, dim=1) for values in zip(*role_limits))
        n = actor_state.position.shape[1]
        static_counts = torch.zeros_like(self.contact_static, dtype=torch.int32)
        pair_counts = torch.zeros((self.batch_size, n), dtype=torch.int32, device=self.device)
        learner_reach_event = torch.zeros_like(self.learner_active)
        learner_capture_event = torch.zeros_like(self.learner_active)
        target_capture_total = torch.zeros_like(self.target_active)
        scene_done_now = torch.zeros_like(self.done)
        attacker_win = torch.zeros_like(self.done)
        defender_win = torch.zeros_like(self.done)
        scene_horizon_terminal = torch.zeros_like(self.done)
        scene_fleet_success_terminal = torch.zeros_like(self.done)
        clock = self.config.goal_reach["clock"]
        maximum_ticks = int(clock["maximum_physics_ticks"])
        defense_utility = torch.zeros(self.batch_size, device=self.device)
        viewer_cpu_chunk = (self.device.type == 'cpu' and self.batch_size == 1
                            and physics_ticks is not None and not build_observation and not return_both)
        tick_count = self._resolved_step_physics_ticks(physics_ticks)
        if viewer_cpu_chunk and self.done[0]:
            if tick_count:
                # The inactive contact path adds a zero displacement, including -0 -> +0.
                actor_state.position = actor_state.position + 0.0
            tick_count = 0
        totals = {'static_counts': static_counts, 'pair_counts': pair_counts, 'learner_reach_event': learner_reach_event, 'learner_capture_event': learner_capture_event, 'target_capture_total': target_capture_total, 'scene_done_now': scene_done_now, 'attacker_win': attacker_win, 'defender_win': defender_win, 'scene_horizon_terminal': scene_horizon_terminal, 'scene_fleet_success_terminal': scene_fleet_success_terminal, 'defense_utility': defense_utility}
        finish_decision = self._advance_joint_ticks(actor_state,
            (mixed_speed_cap, mixed_yaw_cap, mixed_thruster_limit), tick_count, totals,
            physics_deadline=physics_deadline, finish_decision=finish_decision, viewer_cpu_chunk=viewer_cpu_chunk)
        for name, value in actor_state.__dict__.items():
            setattr(self.defenders, name, value[:, :self.defender_count])
            setattr(self.learners, name, value[:, self.defender_count:learner_end])
            setattr(self.targets, name, value[:, learner_end:])
        if self.device.type == "cpu":
            bindings = self._cpu_role_bindings()
            self._cpu_state_cache = (actor_state, bindings,
                                     tuple(self._cpu_binding_layout(value) for value in bindings))
        self._bind_primary_learner()
        self.defender_success |= defender_win[:, None] & self.defender_credit_member
        self.success |= attacker_win if self.training_target == "learner" else defender_win
        boundary_collision = static_counts[:, :self.defender_count, -2:].sum(-1)
        obstacle_collision = static_counts[:, :self.defender_count, :-2].sum(-1)
        pair_collision = pair_counts[:, :self.defender_count]
        learner_distance_lost = static_counts[:, self.defender_count:learner_end, -2]
        learner_engagement_collision = static_counts[:, self.defender_count:learner_end, -1]
        learner_boundary_collision = learner_distance_lost + learner_engagement_collision
        learner_obstacle_collision = static_counts[:, self.defender_count:learner_end, :-2].sum(-1)
        learner_pair_collision_total = pair_counts[:, self.defender_count:learner_end]
        common_info = {
            "episode_defense_utility": self.episode_defense_utility.clone(),
            "defense_rate": self.defender_defense_rate(), "fleet_success": scene_fleet_success_terminal,
            "match_done": scene_done_now, "match_outcome": self.match_outcome.clone(),
            "match_end_reason": self.match_end_reason.clone(),
            "defense_utility": defense_utility, "attacker_utility": -defense_utility,
            "protected_time": torch.where(self.match_outcome == int(MatchOutcome.DEFENDER_WIN), 1.0,
                                            self.physics_ticks.to(torch.float32) / maximum_ticks),
            "episode_pair_contacts": self.episode_pair_contacts.clone(),
            "event_attacker_reach": learner_reach_event, "event_attacker_capture": learner_capture_event,
            "event_attacker_static_collision": learner_obstacle_collision,
            "event_attacker_distance_lost": learner_distance_lost,
            "event_attacker_vessel_collision": learner_pair_collision_total,
            "event_defender_static_collision": boundary_collision + obstacle_collision,
            "event_defender_vessel_collision": pair_collision,
        }
        target_boundary_total = static_counts[:, learner_end:, -2:].sum(-1)
        target_obstacle_total = static_counts[:, learner_end:, :-2].sum(-1)
        target_pair_total = pair_counts[:, learner_end:]
        common_info.update(event_target_capture=target_capture_total,
                           event_target_static_collision=target_boundary_total + target_obstacle_total,
                           event_target_vessel_collision=target_pair_total, target_alive=self.target_active.clone())
        for event_role, slots in (('defender', slice(0, self.defender_count)),
                                  ('attacker', slice(self.defender_count, learner_end)),
                                  ('target', slice(learner_end, None))):
            contacts = static_counts[:, slots]
            common_info.update({f'event_{event_role}_obstacle': contacts[..., :-2].sum(-1),
                                f'event_{event_role}_distance_lost': contacts[..., -2],
                                f'event_{event_role}_engagement_boundary': contacts[..., -1]})

        observations = (self.training_observation_bundle(
            self.policy_roles if return_both else (self.training_target,))
            if build_observation and finish_decision else {})

        def actor_result(actor_kind: str) -> GoalReachStep:
            if actor_kind == "learner":
                credit_member = self.learner_credit_member
                actor_collision = (self.learner_status == 2) | (
                    self.learner_status == 5
                )
                boundary_cause = learner_boundary_collision
                obstacle_cause = learner_obstacle_collision
                pair_cause = learner_pair_collision_total
                team_success = attacker_win
                team_failure = defender_win
            elif actor_kind == "target":
                credit_member = self.target_credit_member
                actor_collision = self.target_collision
                boundary_cause = target_boundary_total
                obstacle_cause = target_obstacle_total
                pair_cause = target_pair_total
                team_success, team_failure = defender_win, attacker_win
            else:
                credit_member = self.defender_credit_member
                actor_collision = self.defender_collision
                boundary_cause = boundary_collision
                obstacle_cause = obstacle_collision
                pair_cause = pair_collision
                team_success = defender_win
                team_failure = attacker_win
            team_sign = -defense_utility if actor_kind == "learner" else defense_utility
            truncated = (scene_horizon_terminal[:, None] & credit_member
                         if self.bootstrap_on_time_limit else torch.zeros_like(credit_member))
            terminated = (scene_done_now[:, None] & credit_member) & ~truncated
            reward = team_sign[:, None] * credit_member.to(torch.float32)
            actor_success = team_success[:, None] & credit_member
            actor_physics_ticks = self.physics_ticks[:, None].expand_as(credit_member)
            collapse = False

            def compatible(value: torch.Tensor) -> torch.Tensor:
                return value[:, 0] if collapse else value

            return GoalReachStep(
                observation=observations.get(actor_kind, {}),
                reward=compatible(
                    reward
                ),
                terminated=compatible(terminated),
                truncated=compatible(truncated),
                info={
                    **common_info,
                    "success": compatible(actor_success.clone()),
                    "collision": compatible(actor_collision.clone()),
                    "leakage_failure": compatible(
                        team_failure[:, None].expand_as(terminated)
                    ),
                    "boundary_collision": compatible(boundary_cause),
                    "obstacle_collision": compatible(obstacle_cause),
                    "pair_collision": compatible(pair_cause),
                    "horizon": compatible(
                        scene_horizon_terminal[:, None].expand_as(terminated)
                    ),
                    "physics_ticks": compatible(actor_physics_ticks),
                },
            )

        if return_both:
            return JointGoalReachStep(
                learner=actor_result("learner"),
                defender=actor_result("defender"),
                target=actor_result("target") if dynamic else None,
            )
        return actor_result(self.training_target)

    def training_step_joint(
        self,
        learner_action: torch.Tensor,
        defender_action: torch.Tensor,
        target_action: torch.Tensor | None = None,
        *,
        build_observation: bool = True,
    ) -> JointGoalReachStep:
        result = self._step_joint_match(
            learner_action=learner_action,
            defender_action=defender_action,
            target_action=target_action,
            build_observation=build_observation,
            return_both=True,
        )
        assert isinstance(result, JointGoalReachStep)
        return result

    def _step_training_defender(
        self,
        action: torch.Tensor,
        *,
        learner_action: torch.Tensor | None = None,
        target_action: torch.Tensor | None = None,
        build_observation: bool = True,
        physics_ticks: int | None = None,
        finish_decision: bool = True,
        physics_deadline: float | None = None,
    ) -> GoalReachStep:
        return self._step_joint_match(
            defender_action=action,
            learner_action=learner_action,
            target_action=target_action,
            build_observation=build_observation,
            physics_ticks=physics_ticks,
            finish_decision=finish_decision,
            physics_deadline=physics_deadline,
        )

    def _step_training_learner(
        self,
        action: torch.Tensor,
        *,
        defender_action: torch.Tensor | None = None,
        target_action: torch.Tensor | None = None,
        build_observation: bool = True,
        physics_ticks: int | None = None,
        finish_decision: bool = True,
        physics_deadline: float | None = None,
    ) -> GoalReachStep:
        return self._step_joint_match(
            defender_action=defender_action,
            learner_action=action,
            target_action=target_action,
            build_observation=build_observation,
            physics_ticks=physics_ticks,
            finish_decision=finish_decision,
            physics_deadline=physics_deadline,
        )
