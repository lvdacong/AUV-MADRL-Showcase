from __future__ import annotations

import copy
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .actor_variants import (
    CRAFT_SPEED_RATIO_BOUNDS,
    DEFAULT_ACTOR_VARIANT_IDS,
    ActorVariantSpec,
    actor_variant,
)
from .config import ProjectConfig
from .dynamics import AuvState, BatchedAuvDynamics
from .geometry import CircleWorld
from .scene_bank import TERRAIN_GENERATOR, scene_spec, scene_contract, scene_contract_diff, reject_scene_contract_differences
from .scene_sampling import (
    scene_validity,
    ActorSpawnBatch,
    SceneBatch,
    sample_scenes,
    compose_episode_seeds,
    sample_actor_spawns,
    episode_actor_randomization,
)
from .terrain import (
    load_periodic_maps, periodic_query, world_parameters, periodic_map_origins, compact_first_indices,
    TERRAIN_BLOCK_SIZE_M,
    pack_obstacles,
    required_blocks,
    terrain_phase,
)


@dataclass
class GoalReachStep:
    observation: dict[str, torch.Tensor]
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: dict[str, torch.Tensor]


@dataclass
class JointGoalReachStep:
    learner: GoalReachStep
    defender: GoalReachStep
    target: GoalReachStep | None = None


def smooth_tangent_navigation(
    target_direction: torch.Tensor,
    tangent_direction: torch.Tensor,
    outward_normal: torch.Tensor,
    surface_distance: torch.Tensor,
    outer_distance: float,
    inner_distance: float,
    outward_normal_coefficient: float,
) -> torch.Tensor:
    """Blend a target direction into one normalized tangent-avoidance direction."""
    q = ((outer_distance - surface_distance) / (outer_distance - inner_distance)).clamp(0.0, 1.0)
    weight = q.square() * (3.0 - 2.0 * q)
    avoid = tangent_direction + outward_normal_coefficient * outward_normal
    avoid = avoid / avoid.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    direction = (1.0 - weight.unsqueeze(-1)) * target_direction + weight.unsqueeze(-1) * avoid
    return direction / direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)


def merge_reset_observation(base, refreshed, mask):
    result = dict(base)
    for key, value in refreshed.items():
        old = base[key]
        if key in ('obstacle_entities', 'obstacle_mask'):
            axis = -2 if key.endswith('entities') else -1
            width = max(old.shape[axis], value.shape[axis])
            pad = (0, 0, 0, width - old.shape[axis]) if axis == -2 else (0, width - old.shape[axis])
            merged = torch.nn.functional.pad(old, pad) if old.shape[axis] != width else old.clone()
            pad = (0, 0, 0, width - value.shape[axis]) if axis == -2 else (0, width - value.shape[axis])
            value = torch.nn.functional.pad(value, pad) if value.shape[axis] != width else value
        else:
            if old.shape[1:] != value.shape[1:]:
                raise ValueError(f'Partial reset changed non-obstacle shape: {key}')
            merged = old.clone()
        merged[mask] = value
        result[key] = merged
    return result


class GoalReachEnv:
    """Batched learner or patrol PPO environment."""

    state_schema = "fixed_world_local15_first10_single_v1"
    _tensor_rng_modulus = 2_147_483_648
    _tensor_rng_multiplier = 1_103_515_245
    _tensor_rng_increment = 12_345

    def __init__(
        self,
        config: ProjectConfig,
        batch_size: int,
        device: torch.device | str = "cuda",
        base_seed: int = 42,
        env_id_offset: int = 0,
        map_device: torch.device | str | None = None,
        training_target: str = "learner",
        learner_architecture_id: str = DEFAULT_ACTOR_VARIANT_IDS["learner"],
        patrol_architecture_id: str = DEFAULT_ACTOR_VARIANT_IDS["defender"],
        defender_count: int | None = None,
    ):
        if training_target not in {"learner", "defender"}:
            raise ValueError("GoalReach training target must be learner or defender")
        self.config = config
        self.world_parameters = world_parameters(config.goal_reach)
        self.periodic_bank = load_periodic_maps(device, parameters=self.world_parameters)
        self.target_mode = str(config.goal_reach.get("target_mode", "static"))
        self.roles = ("learner", "defender")
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.vessel_radius_m = float(config.goal_reach["vessel_radius_m"])
        if self.vessel_radius_m <= 0.0:
            raise ValueError("Vessel radius must be positive")
        self.training_target = training_target
        self.learner_variant = actor_variant(learner_architecture_id)
        if self.learner_variant.actor_kind != "learner":
            raise ValueError("GoalReach learner architecture must belong to the learner task")
        self.set_patrol_architecture(patrol_architecture_id)
        self.map_device = self.device if map_device is None else torch.device(map_device)
        self.base_seed = int(base_seed)
        self.env_id_offset = int(env_id_offset)
        self.global_lane_capacity = self.batch_size
        self.world = CircleWorld(config.world, self.device)
        if self.world.radii.numel():
            raise ValueError("GoalReach does not contain fixed islands")

        self.learner_dynamics = BatchedAuvDynamics(config.shared, self.device)
        self.primary_learner_active = torch.ones(
            self.batch_size, dtype=torch.bool, device=self.device
        )
        self.learner_active = self.primary_learner_active[:, None]
        defender = config.goal_reach["defender"]
        self.defender_count = int(
            (0 if training_target == "learner" else 1)
            if defender_count is None
            else defender_count
        )
        if not 0 <= self.defender_count <= 20:
            raise ValueError("GoalReach defender count must be inside [0,20]")
        self.defender_dynamics = BatchedAuvDynamics(config.shared, self.device)
        self.learner: AuvState = self.learner_dynamics.empty_state((self.batch_size,))
        self.defenders: AuvState = self.defender_dynamics.empty_state(
            (self.batch_size, self.defender_count)
        )
        self.defender_initial_position = torch.zeros(
            (self.batch_size, self.defender_count, 2), dtype=torch.float32, device=self.device
        )
        self.defender_initial_heading = torch.zeros(
            (self.batch_size, self.defender_count), dtype=torch.float32, device=self.device
        )
        obstacle_specification = config.goal_reach["reset"]["random_obstacles"]
        self.position_scale_m = self.world.radius + self.periodic_bank.max_radius
        decision_s = (
            float(config.goal_reach["clock"]["physics_ticks_per_decision"])
            * float(config.shared["clock"]["physics_step_s"])
        )
        self.terrain_refresh_halo_m = (
            float(self.learner_dynamics.maximum_surge)
            * float(CRAFT_SPEED_RATIO_BOUNDS[1])
            * math.sqrt(1.25)
            * decision_s
            + self.vessel_radius_m
        )
        self.obstacle_capacity = int(obstacle_specification["capacity"])
        self.goal_capacity = 0
        self.targets = self.learner_dynamics.empty_state((self.batch_size, 1))
        self.region_reference_position = torch.zeros((self.batch_size, 2), device=self.device)
        self.sampled_goal_count = torch.zeros(
            self.batch_size, dtype=torch.int32, device=self.device
        )
        self.goal_active = torch.zeros(
            (self.batch_size, self.goal_capacity), dtype=torch.bool, device=self.device
        )
        self.goal_centers = torch.zeros(
            (self.batch_size, self.goal_capacity, 2),
            dtype=torch.float32,
            device=self.device,
        )
        self.goal_radii = torch.zeros(
            (self.batch_size, self.goal_capacity), dtype=torch.float32, device=self.device
        )
        self.sampled_obstacle_count = torch.zeros(
            self.batch_size, dtype=torch.int32, device=self.device
        )
        self.obstacle_active = torch.zeros(
            (self.batch_size, self.obstacle_capacity), dtype=torch.bool, device=self.device
        )
        self.obstacle_centers = torch.zeros(
            (self.batch_size, self.obstacle_capacity, 2),
            dtype=torch.float32,
            device=self.device,
        )
        self.obstacle_radii = torch.zeros(
            (self.batch_size, self.obstacle_capacity), dtype=torch.float32, device=self.device
        )
        self.defender_active = torch.zeros(
            (self.batch_size, self.defender_count), dtype=torch.bool, device=self.device
        )
        self.patrol_waypoint = torch.zeros(
            (self.batch_size, self.defender_count, 2), dtype=torch.float32, device=self.device
        )
        self.patrol_hold_ticks = torch.zeros(
            (self.batch_size, self.defender_count), dtype=torch.int32, device=self.device
        )
        self.defender_avoidance_hazard_id = torch.full(
            (self.batch_size, self.defender_count),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        self.obstacle_keys = torch.zeros(
            (self.batch_size, self.obstacle_capacity, 4),
            dtype=torch.int64,
            device=self.device,
        )
        self.terrain_world_seed = torch.zeros(
            self.batch_size, dtype=torch.int64, device=self.device
        )
        self.terrain_override_revision = torch.zeros_like(self.terrain_world_seed)
        self.terrain_layout_attempt = torch.zeros(
            self.batch_size, dtype=torch.int16, device=self.device
        )
        self.terrain_phase = torch.zeros(
            (self.batch_size, 2), dtype=torch.float32, device=self.device
        )
        self.terrain_manual_override = torch.zeros(
            self.batch_size, dtype=torch.bool, device=self.device
        )
        self.map_id = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        self.map_origin_xy = torch.zeros((self.batch_size, 2), device=self.device)
        self._terrain_coverage = None
        self.terrain_overrides: list[dict[tuple[int, int, int, int], tuple[tuple[float, float], float] | None]] = [
            {} for _ in range(self.batch_size)
        ]
        self.terrain_next_manual_id = torch.zeros(
            self.batch_size, dtype=torch.int64, device=self.device
        )
        self.terrain_override_active = torch.zeros(
            self.batch_size, dtype=torch.bool, device=self.device
        )
        self.defender_avoidance_side = torch.zeros(
            (self.batch_size, self.defender_count),
            dtype=torch.int8,
            device=self.device,
        )
        self.defender_avoidance_surface_distance = torch.full(
            (self.batch_size, self.defender_count),
            torch.inf,
            dtype=torch.float32,
            device=self.device,
        )
        self.defender_rng_state = torch.zeros(
            (self.batch_size, self.defender_count), dtype=torch.int64, device=self.device
        )
        direction_samples = int(defender["patrol_waypoint_direction_samples"])
        self._patrol_direction_offsets = torch.arange(
            direction_samples, dtype=torch.float32, device=self.device
        ) * (2.0 * torch.pi / direction_samples)
        self._patrol_distance_scales = torch.tensor(
            (1.0, 0.72, 0.72**2, 0.72**3),
            dtype=torch.float32,
            device=self.device,
        )

        self.physics_ticks = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
        self.episode_counters = torch.full(
            (self.batch_size,), -1, dtype=torch.int64, device=self.device
        )
        self.done = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        self.success = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self.collision = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self._episode_seeds = torch.zeros(
            self.batch_size, dtype=torch.int64, device=self.device
        )

        pixels = torch.arange(96, dtype=torch.float32, device=self.device)
        coordinates = -self.world.radius + (pixels + 0.5) * (2.0 * self.world.radius / 96.0)
        z, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        self._semantic_points = torch.stack((x, z), dim=-1).reshape(-1, 2)
        self._semantic_points_map = self._semantic_points.to(self.map_device)
        self._semantic_cell_size = 2.0 * self.world.radius / 96.0
        learner_count_range = tuple(
            int(value) for value in getattr(self, "learner_count_range", (1, 1))
        )
        defender_count_range = tuple(
            int(value)
            for value in getattr(
                self, "defender_count_range", (self.defender_count, self.defender_count)
            )
        )
        self.scene_spec = scene_spec(config, learner_count_range=learner_count_range,
                                     defender_count_range=defender_count_range)
        self.scene_contract = scene_contract(self.scene_spec,
            tuple(getattr(self, "learner_speed_ratio_range", (2.,2.))),
            tuple(defender["speed_ratio"]),tuple(config.goal_reach.get("target_speed_ratio_range",(2.,2.))))
        self.reset_statistics = {}
        self.performance_recorder = None

    def _invalidate_obstacle_neighbors(self, indices=None) -> None:
        cache = getattr(self, '_obstacle_neighbors', None)
        if cache is not None:
            if indices is None:
                cache['valid'].zero_()
            else:
                cache['valid'][indices] = False

    def ensure_circle_storage(
        self,
        required_goal_count: int,
        required_obstacle_count: int,
    ) -> None:
        """Grow Viewer circle tensors on demand without imposing an edit limit."""

        required_goal_count = int(required_goal_count)
        required_obstacle_count = int(required_obstacle_count)
        if required_goal_count < 0 or required_obstacle_count < 0:
            raise ValueError("Circle counts must be non-negative")
        goal_capacity = max(self.goal_capacity, required_goal_count)
        obstacle_capacity = max(self.obstacle_capacity, required_obstacle_count)
        if (
            goal_capacity == self.goal_capacity
            and obstacle_capacity == self.obstacle_capacity
        ):
            return
        self._invalidate_obstacle_neighbors()

        def expanded(tensor: torch.Tensor, capacity: int) -> torch.Tensor:
            shape = list(tensor.shape)
            shape[1] = capacity
            result = tensor.new_zeros(shape)
            result[:, : tensor.shape[1]] = tensor
            return result

        self.goal_active = expanded(self.goal_active, goal_capacity)
        self.goal_centers = expanded(self.goal_centers, goal_capacity)
        self.goal_radii = expanded(self.goal_radii, goal_capacity)
        self.obstacle_active = expanded(self.obstacle_active, obstacle_capacity)
        self.obstacle_centers = expanded(self.obstacle_centers, obstacle_capacity)
        self.obstacle_radii = expanded(self.obstacle_radii, obstacle_capacity)
        self.obstacle_keys = expanded(self.obstacle_keys, obstacle_capacity)

        self.goal_capacity = goal_capacity
        self.obstacle_capacity = obstacle_capacity
        self.config.goal_reach["reset"]["random_obstacles"][
            "capacity"
        ] = obstacle_capacity

    @staticmethod
    def _layout_point_clear(
        point: tuple[float, float],
        world_radius: float,
        obstacle_centers: Sequence[tuple[float, float]],
        obstacle_radii: Sequence[float],
        clearance: float,
    ) -> bool:
        del world_radius
        return all(
            math.dist(point, center) > radius + clearance
            for center, radius in zip(obstacle_centers, obstacle_radii, strict=True)
        )

    def set_fixed_circle_geometry(
        self,
        mask: torch.Tensor,
        *,
        goal_centers: Sequence[tuple[float, float]],
        goal_radii: Sequence[float],
        obstacle_centers: Sequence[tuple[float, float]],
        obstacle_radii: Sequence[float],
    ) -> None:
        """Apply one fixed goal/obstacle geometry to the selected environments."""

        if len(goal_centers) != len(goal_radii):
            raise ValueError("Fixed goal geometry is incomplete")
        if len(obstacle_centers) != len(obstacle_radii):
            raise ValueError("Fixed obstacle geometry is incomplete")
        if any(
            len(center) != 2
            or not all(math.isfinite(float(value)) for value in center)
            or not math.isfinite(float(radius))
            or float(radius) <= 0.0
            for centers, radii in (
                (goal_centers, goal_radii),
                (obstacle_centers, obstacle_radii),
            )
            for center, radius in zip(centers, radii, strict=True)
        ):
            raise ValueError("Fixed scene circles must have finite positions and positive radii")
        self.ensure_circle_storage(len(goal_centers), len(obstacle_centers))
        mask = mask.to(device=self.device, dtype=torch.bool)
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        if not indices.numel():
            return

        goal_count = len(goal_centers)
        self._invalidate_obstacle_neighbors(indices)
        self.sampled_goal_count[indices] = goal_count
        self.goal_active[indices] = False
        self.goal_centers[indices] = 0.0
        self.goal_radii[indices] = 0.0
        if goal_count:
            self.goal_active[indices, :goal_count] = True
            self.goal_centers[indices, :goal_count] = torch.tensor(
                goal_centers, dtype=torch.float32, device=self.device
            )
            self.goal_radii[indices, :goal_count] = torch.tensor(
                goal_radii, dtype=torch.float32, device=self.device
            )

        obstacle_count = len(obstacle_centers)
        self.sampled_obstacle_count[indices] = obstacle_count
        self.obstacle_active[indices] = False
        self.obstacle_centers[indices] = 0.0
        self.obstacle_radii[indices] = 0.0
        if obstacle_count:
            self.obstacle_active[indices, :obstacle_count] = True
            self.obstacle_centers[indices, :obstacle_count] = torch.tensor(
                obstacle_centers, dtype=torch.float32, device=self.device
            )
            self.obstacle_radii[indices, :obstacle_count] = torch.tensor(
                obstacle_radii, dtype=torch.float32, device=self.device
            )
        self.obstacle_keys[indices] = 0
        if obstacle_count:
            self.obstacle_keys[indices, :obstacle_count, 0] = -(1 << 63)
            self.obstacle_keys[indices, :obstacle_count, 3] = torch.arange(obstacle_count, device=self.device)
        self.terrain_next_manual_id[indices] = obstacle_count
        for lane in indices.tolist():
            self.terrain_overrides[lane].clear()
        self.terrain_override_active[indices] = False
        self.terrain_manual_override[indices] = True
        self.terrain_override_revision[indices] += 1

    def set_fixed_goal_geometry(
        self,
        mask: torch.Tensor,
        *,
        goal_centers: Sequence[tuple[float, float]],
        goal_radii: Sequence[float],
    ) -> None:
        if len(goal_centers) != len(goal_radii):
            raise ValueError("Fixed goal geometry is incomplete")
        self.ensure_circle_storage(len(goal_centers), 0)
        indices = torch.nonzero(mask.to(device=self.device, dtype=torch.bool), as_tuple=False).flatten()
        self.sampled_goal_count[indices] = len(goal_centers)
        self.goal_active[indices] = False
        self.goal_centers[indices] = 0
        self.goal_radii[indices] = 0
        if goal_centers:
            self.goal_active[indices, : len(goal_centers)] = True
            self.goal_centers[indices, : len(goal_centers)] = torch.tensor(
                goal_centers, dtype=torch.float32, device=self.device
            )
            self.goal_radii[indices, : len(goal_radii)] = torch.tensor(
                goal_radii, dtype=torch.float32, device=self.device
            )

    @staticmethod
    def terrain_key_text(key: Sequence[int]) -> str:
        values = tuple(int(value) for value in key)
        if len(values) != 4:
            raise ValueError("Terrain key must have four integer components")
        return ("m:" + str(values[3])) if values[0] == -(1 << 63) else "t:" + ":".join(map(str, values))

    @staticmethod
    def parse_terrain_key(value: object) -> tuple[int, int, int, int]:
        text = str(value)
        if text.startswith("m:"):
            return (-(1 << 63), 0, 0, int(text[2:]))
        parts = text.removeprefix("t:").split(":")
        if len(parts) != 4:
            raise ValueError("Unknown terrain obstacle key")
        return tuple(int(part) for part in parts)  # type: ignore[return-value]

    def edit_terrain_obstacle(
        self,
        lane: int,
        source_key: object,
        *,
        center: tuple[float, float] | None = None,
        radius: float | None = None,
        duplicate: bool = False,
        delete: bool = False,
    ) -> str | None:
        """Record one Viewer-only terrain override without replacing the world."""

        if not 0 <= int(lane) < self.batch_size:
            raise ValueError("Terrain lane is out of range")
        source = self.parse_terrain_key(source_key)
        overrides = self.terrain_overrides[int(lane)]
        if delete:
            overrides[source] = None
            result = None
        else:
            if center is None or radius is None or not all(math.isfinite(v) for v in center) or not math.isfinite(radius) or radius <= 0:
                raise ValueError("Terrain obstacle edit must be finite and positive")
            if duplicate:
                manual_id = int(self.terrain_next_manual_id[lane])
                self.terrain_next_manual_id[lane] += 1
                result_key = (-(1 << 63), 0, 0, manual_id)
            else:
                result_key = source
            overrides[result_key] = ((float(center[0]), float(center[1])), float(radius))
            if result_key != source:
                result = self.terrain_key_text(result_key)
            else:
                result = self.terrain_key_text(source)
        self.terrain_override_active[lane] = bool(overrides)
        self._repack_terrain_lanes(torch.tensor([lane], dtype=torch.int64, device=self.device))
        self.terrain_override_revision[lane] += 1
        return result

    def terrain_descriptor(self, lane: int, *, scene_epoch: int) -> dict[str, object]:
        """Detached static snapshot; no tensor/override references or physics mutation."""
        return {
            "generator": TERRAIN_GENERATOR,
            "bank_schema": "periodic_lognormal_regions_v1",
            "bank_hash": self.periodic_bank.asset_hash,
            "map_id": int(self.map_id[lane]),
            "world_parameters": dict(self.world_parameters),
            "map_origin_xy": tuple(float(v) for v in self.map_origin_xy[lane]),
            "scene_epoch": int(scene_epoch),
            "world_epoch": int(self.episode_counters[lane]),
            "original_episode_seed": int(self._episode_seeds[lane]),
            "world_seed": int(self.terrain_world_seed[lane]),
            "density_per_1000m2": self.world_parameters["obstacle_density_mean_per_1000m2"],
            "phase": tuple(float(-v) for v in self.map_origin_xy[lane]),
            "block_size_m": TERRAIN_BLOCK_SIZE_M,
            "sense_radius_m": float(self.world.radius),
            "obstacle_radius_range_m": (0., self.periodic_bank.max_radius),
            "manual_fixed": bool(self.terrain_manual_override[lane]),
            "override_revision": int(self.terrain_override_revision[lane]),
        }

    def _apply_terrain_overrides(
        self,
        lanes: torch.Tensor,
        centers: torch.Tensor,
        radii: torch.Tensor,
        keys: torch.Tensor,
        selected_lanes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Training has no manual overrides. Check their authoritative CPU
        # containers before asking the device which selected lanes have edits.
        if not any(self.terrain_overrides) or not bool(self.terrain_override_active[selected_lanes].any()):
            return lanes, centers, radii, keys
        keep = torch.ones(lanes.numel(), dtype=torch.bool, device=self.device)
        appended_lanes: list[int] = []
        appended_centers: list[tuple[float, float]] = []
        appended_radii: list[float] = []
        appended_keys: list[tuple[int, int, int, int]] = []
        override_lanes = torch.unique(torch.cat((lanes, selected_lanes))).tolist()
        for lane in override_lanes:
            lane_mask = lanes == lane
            for key, override in self.terrain_overrides[int(lane)].items():
                key_tensor = torch.tensor(key, dtype=torch.int64, device=self.device)
                matches = lane_mask & (keys == key_tensor).all(1)
                if bool(matches.any()):
                    keep &= ~matches
                if override is not None:
                    center, radius = override
                    appended_lanes.append(int(lane))
                    appended_centers.append(center)
                    appended_radii.append(radius)
                    appended_keys.append(key)
        lanes, centers, radii, keys = lanes[keep], centers[keep], radii[keep], keys[keep]
        if appended_lanes:
            lanes = torch.cat((lanes, torch.tensor(appended_lanes, dtype=torch.int64, device=self.device)))
            centers = torch.cat((centers, torch.tensor(appended_centers, dtype=torch.float32, device=self.device)))
            radii = torch.cat((radii, torch.tensor(appended_radii, dtype=torch.float32, device=self.device)))
            keys = torch.cat((keys, torch.tensor(appended_keys, dtype=torch.int64, device=self.device)))
        return lanes, centers, radii, keys

    def _repack_terrain_lanes(self, indices: torch.Tensor) -> None:
        # Edits are rare, explicit geometry revisions; rebuild the same local set.
        # Fixed scenes have no procedural base to refresh. Apply their overlay
        # to the installed active circles, retaining contact identities on install.
        for lane in indices[self.terrain_manual_override[indices]].tolist():
            selected = torch.tensor([lane], dtype=torch.int64, device=self.device)
            active = self.obstacle_active[lane]
            keys = self.obstacle_keys[lane, active]
            lanes = torch.full((len(keys),), lane, dtype=torch.int64, device=self.device)
            _, centers, radii, keys = self._apply_terrain_overrides(
                lanes, self.obstacle_centers[lane, active],
                self.obstacle_radii[lane, active], keys, selected,
            )
            self._install_local_terrain(
                selected, centers[None], radii[None], keys[None],
                torch.ones((1, len(keys)), dtype=torch.bool, device=self.device),
            )
        self._refresh_terrain(indices, force=True)

    def _install_local_terrain(self, indices, centers, radii, keys, active):
        old_keys = self.obstacle_keys[indices].clone()
        old_centers = self.obstacle_centers[indices].clone()
        old_radii = self.obstacle_radii[indices].clone()
        old_contacts = (self.contact_static[indices, :, :self.obstacle_capacity].clone()
                        if hasattr(self, 'contact_static') else None)
        packing = active
        # Preserve historical contact identities until reset, even outside coverage.
        if old_contacts is not None:
            seen = (old_keys[:, :, None] == keys[:, None]).all(-1) & active[:, None]
            retain = old_contacts.any(1) & ~seen.any(-1)
            centers = torch.cat((centers, old_centers), 1)
            radii = torch.cat((radii, old_radii), 1)
            keys = torch.cat((keys, old_keys), 1)
            packing = torch.cat((active, retain), 1)
            active = torch.cat((active, torch.zeros_like(retain)), 1)
        index, storage_mask = compact_first_indices(packing)
        mask = active.gather(1,index) & storage_mask
        keys = keys.gather(1, index[..., None].expand(-1,-1,4))
        centers = centers.gather(1, index[..., None].expand(-1,-1,2))
        radii = radii.gather(1,index)
        width = index.shape[1]
        self.ensure_circle_storage(0, width)
        self._invalidate_obstacle_neighbors(indices)
        self.obstacle_centers[indices] = 0
        self.obstacle_radii[indices] = 0
        self.obstacle_keys[indices] = 0
        self.obstacle_active[indices] = False
        self.obstacle_centers[indices,:width] = centers
        self.obstacle_radii[indices,:width] = radii
        self.obstacle_keys[indices,:width] = keys
        self.obstacle_active[indices,:width] = mask
        if old_contacts is not None:
            matches = (keys[:,:,None] == old_keys[:,None]).all(-1) & storage_mask[:,:,None]
            contacts = (matches[:,None] & old_contacts[:,:,None]).any(-1)
            self.contact_static[indices,:,:self.obstacle_capacity] = False
            self.contact_static[indices,:,:width] = contacts
        self.sampled_obstacle_count[indices] = mask.sum(1).to(self.sampled_obstacle_count.dtype)

    def _terrain_actor_geometry(
        self, indices: torch.Tensor, *, initial: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        names = ("learner", "defender", "target") if hasattr(self, "target_active") else ("learner", "defender")
        positions = torch.cat(tuple(self.role_state(name).position[indices] for name in names), dim=1)
        active = torch.cat(tuple(getattr(self, name + "_active")[indices] for name in names), dim=1)
        if initial and hasattr(self, "target_active"):
            angles = torch.arange(8, device=self.device, dtype=torch.float32) * (torch.pi / 4.0)
            ring = self.region_reference_position[indices, None] + self.world.radius * torch.stack(
                (angles.sin(), angles.cos()), dim=-1
            )[None]
            positions = torch.cat((positions, ring), dim=1)
            active = torch.cat((active, torch.ones(ring.shape[:2], dtype=torch.bool, device=self.device)), dim=1)
        return positions, active

    def _refresh_terrain(self, indices: torch.Tensor, *, initial=False, force=False) -> None:
        indices = indices.to(device=self.device, dtype=torch.int64)
        indices = indices[~self.terrain_manual_override[indices]]
        if not indices.numel():
            return
        if self._terrain_coverage is None:
            self._terrain_coverage = (torch.full((self.batch_size,2), torch.inf, device=self.device),
                                      torch.full((self.batch_size,2), -torch.inf, device=self.device))
        pending = indices
        while pending.numel():
            positions, active = self._terrain_actor_geometry(pending, initial=initial)
            lower = positions.masked_fill(~active[...,None], torch.inf).amin(1)
            upper = positions.masked_fill(~active[...,None], -torch.inf).amax(1)
            any_active = active.any(1)
            lower = torch.where(any_active[:,None],lower,self.region_reference_position[pending])
            upper = torch.where(any_active[:,None],upper,lower)
            radius = self.position_scale_m + self.terrain_refresh_halo_m
            lower, upper = lower-radius, upper+radius
            previous_lo, previous_hi = self._terrain_coverage
            rebuild = ((lower < previous_lo[pending]) | (upper > previous_hi[pending])).any(1)
            if initial or force:
                rebuild.fill_(True)
            pending, lower, upper = pending[rebuild], lower[rebuild], upper[rebuild]
            if not pending.numel():
                return
            # One extra cell of coverage amortizes query/repacking, without world generation.
            lower, upper = lower-25, upper+25
            centers, radii, keys, active = periodic_query(self.periodic_bank, self.map_id[pending],
                lower+self.map_origin_xy[pending], upper+self.map_origin_xy[pending])
            centers -= self.map_origin_xy[pending,None]
            if any(self.terrain_overrides):
                rows, slots = active.nonzero(as_tuple=True)
                lanes, c, r, k = self._apply_terrain_overrides(pending[rows], centers[rows,slots],
                    radii[rows,slots], keys[rows,slots], pending)
                # Stable row packing of the finite local overlay, no distance sorting.
                rows = torch.searchsorted(pending, lanes)
                counts = torch.bincount(rows, minlength=len(pending))
                width = int(counts.max().item())
                centers = centers.new_zeros((len(pending),width,2)); radii = radii.new_zeros((len(pending),width))
                keys = keys.new_zeros((len(pending),width,4)); active = torch.zeros_like(radii,dtype=torch.bool)
                for row in range(len(pending)):
                    selected = rows == row
                    count = int(counts[row])
                    centers[row,:count], radii[row,:count], keys[row,:count] = c[selected],r[selected],k[selected]
                    active[row,:count] = True
            self._install_local_terrain(pending, centers, radii, keys, active)
            previous_lo[pending], previous_hi[pending] = lower, upper
            if not initial or not hasattr(self, 'target_active'):
                return
            distance = (self.obstacle_centers[pending]-self.region_reference_position[pending,None]).norm(dim=-1)
            invalid = (self.obstacle_active[pending] &
                       (distance <= self.obstacle_radii[pending]+self.vessel_radius_m+1.e-5)).any(1)
            pending = pending[invalid]
            self.terrain_layout_attempt[pending] += 1
            if bool((self.terrain_layout_attempt[pending] >= 8).any()):
                raise ValueError('Periodic map exhausted eight reference TGT origin candidates')
            self.map_origin_xy[pending] = periodic_map_origins(self._episode_seeds[pending],self.terrain_layout_attempt[pending])

    def validate_fixed_actor_poses(
        self,
        *,
        learner_positions: torch.Tensor,
        learner_active: torch.Tensor,
        defender_positions: torch.Tensor,
        defender_active: torch.Tensor,
        target_positions: torch.Tensor | None = None,
        goal_centers: Sequence[tuple[float, float]],
        goal_radii: Sequence[float],
        obstacle_centers: Sequence[tuple[float, float]],
        obstacle_radii: Sequence[float],
    ) -> None:
        """Reject an edited fixed scene without changing any current state."""

        if goal_centers or goal_radii:
            raise ValueError("Circular Goals are not part of the unified match")
        target=self.targets.position[0] if target_positions is None else target_positions
        learners=learner_positions[learner_active.bool()]
        defenders=defender_positions[defender_active.bool()]
        scene=SceneBatch(torch.tensor([len(obstacle_centers)],device=self.device),
            torch.tensor(obstacle_centers,dtype=torch.float32,device=self.device).reshape(1,-1,2),
            torch.tensor(obstacle_radii,dtype=torch.float32,device=self.device).reshape(1,-1),
            torch.tensor([len(learners)],device=self.device),learners[None],
            torch.tensor([len(defenders)],device=self.device),defenders[None],target.reshape(1,-1,2),
            self.sampled_target_count[:1])
        if not bool(scene_validity(scene,self.scene_spec).all()):
            raise ValueError("Edited scene conflicts with world/spawn boundaries, craft separation or obstacles; original geometry preserved")

    def _construct_fixed_scene_spawns(
        self,
        env_indices: torch.Tensor,
        *,
        goal_centers: Sequence[tuple[float, float]],
        goal_radii: Sequence[float],
        obstacle_centers: Sequence[tuple[float, float]],
        obstacle_radii: Sequence[float],
    ) -> ActorSpawnBatch:
        count = int(env_indices.numel())
        def repeated_centers(values: Sequence[tuple[float, float]]) -> torch.Tensor:
            return torch.tensor(
                values, dtype=torch.float32, device=self.device
            ).reshape(1, -1, 2).expand(count, -1, -1)

        def repeated_radii(values: Sequence[float]) -> torch.Tensor:
            return torch.tensor(
                values, dtype=torch.float32, device=self.device
            ).reshape(1, -1).expand(count, -1)

        fixed_goal_centers = repeated_centers(goal_centers)
        fixed_goal_radii = repeated_radii(goal_radii)
        fixed_obstacle_centers = repeated_centers(obstacle_centers)
        fixed_obstacle_radii = repeated_radii(obstacle_radii)
        goal_active = torch.ones_like(fixed_goal_radii, dtype=torch.bool)
        obstacle_active = torch.ones_like(fixed_obstacle_radii, dtype=torch.bool)
        learner_counts = (
            self.sampled_learner_count.index_select(0, env_indices)
            if hasattr(self, "sampled_learner_count")
            else torch.ones(count, dtype=torch.int64, device=self.device)
        )
        defender_counts = (
            self.sampled_defender_count.index_select(0, env_indices)
            if hasattr(self, "sampled_defender_count")
            else self.defender_active.index_select(0, env_indices).sum(dim=1)
        )
        return sample_actor_spawns(
            self._episode_seeds.index_select(0, env_indices),
            world_radius=self.world.radius,
            safety_zone_specification=self.config.goal_reach["safety_zones"],
            vessel_radius=self.vessel_radius_m,
            reference_positions=self.region_reference_position[env_indices],
            target_positions=self.targets.position[env_indices],
            target_counts=self.sampled_target_count[env_indices],
            learner_capacity=int(getattr(self, "viewer_learner_capacity", 1)),
            learner_counts=learner_counts,
            defender_capacity=self.defender_count,
            defender_counts=defender_counts,
            goal_centers=fixed_goal_centers,
            goal_radii=fixed_goal_radii,
            goal_active=goal_active,
            obstacle_centers=fixed_obstacle_centers,
            obstacle_radii=fixed_obstacle_radii,
            obstacle_active=obstacle_active,
            validate_fixed_hazards=True,
        )

    def _apply_fixed_spawn_batch(
        self,
        env_indices: torch.Tensor,
        spawns: ActorSpawnBatch,
        mask: torch.Tensor,
    ) -> None:
        learner_positions = self.learner.position.clone()
        learner_headings = self.learner.heading.clone()
        learner_positions[env_indices] = spawns.learner_positions[:, 0].to(self.device)
        learner_headings[env_indices] = spawns.learner_headings[:, 0].to(self.device)
        self.learner_dynamics.reset(self.learner, learner_positions, learner_headings, mask)
        if self.defender_count:
            defender_positions = self.defenders.position.clone()
            defender_headings = self.defenders.heading.clone()
            defender_positions[env_indices] = spawns.defender_positions.to(self.device)
            defender_headings[env_indices] = spawns.defender_headings.to(self.device)
            self.defender_active[env_indices] = spawns.defender_active.to(self.device)
            self.defender_initial_position[env_indices] = defender_positions[env_indices]
            self.defender_initial_heading[env_indices] = defender_headings[env_indices]
            self.defender_dynamics.reset(
                self.defenders, defender_positions, defender_headings, mask[:, None]
            )

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
    ) -> dict[str, torch.Tensor]:
        """Reset selected lanes with fixed circles and a deterministic random Actor pose."""

        if mask is None:
            mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
            if env_indices_cpu is None:
                env_indices_cpu = torch.arange(self.batch_size, dtype=torch.int64)
        mask = mask.to(device=self.device, dtype=torch.bool)
        if env_indices_cpu is None:
            env_indices_cpu = torch.nonzero(mask, as_tuple=False).flatten().cpu()
        else:
            env_indices_cpu = env_indices_cpu.to(device="cpu", dtype=torch.int64)
        env_indices = env_indices_cpu.to(self.device)

        if not env_indices.numel():return base_observation or {}
        spawns = self._construct_fixed_scene_spawns(env_indices,
            goal_centers=goal_centers,goal_radii=goal_radii,
            obstacle_centers=obstacle_centers,obstacle_radii=obstacle_radii)
        self.set_fixed_circle_geometry(mask,goal_centers=goal_centers,goal_radii=goal_radii,
            obstacle_centers=obstacle_centers,obstacle_radii=obstacle_radii)
        self._apply_fixed_spawn_batch(env_indices, spawns, mask)
        self.restart_edited_episode(mask)
        if base_observation is None:
            return self.training_observation()
        refreshed = self.training_observation(mask)
        return merge_reset_observation(base_observation, refreshed, mask)

    @staticmethod
    def _layout_segment_clear(
        start: tuple[float, float],
        end: tuple[float, float],
        obstacle_centers: Sequence[tuple[float, float]],
        obstacle_radii: Sequence[float],
        clearance: float = 0.0,
    ) -> bool:
        dx, dz = end[0] - start[0], end[1] - start[1]
        denominator = dx * dx + dz * dz
        for center, radius in zip(obstacle_centers, obstacle_radii, strict=True):
            projection = (
                0.0
                if denominator <= 1.0e-12
                else ((center[0] - start[0]) * dx + (center[1] - start[1]) * dz) / denominator
            )
            projection = max(0.0, min(1.0, projection))
            closest = (start[0] + projection * dx, start[1] + projection * dz)
            if math.dist(closest, center) <= radius + clearance:
                return False
        return True

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
            if env_indices is None and env_indices_cpu is None:
                env_indices = torch.arange(
                    self.batch_size, dtype=torch.int64, device=self.device
                )
        mask = mask.to(device=self.device, dtype=torch.bool)
        if env_indices is not None and env_indices_cpu is not None:
            raise ValueError("Reset environment indices must use one device")
        if env_indices is not None:
            env_indices_tensor = env_indices.to(
                device=self.device, dtype=torch.int64
            )
        elif env_indices_cpu is not None:
            env_indices_tensor = env_indices_cpu.to(
                device=self.device, dtype=torch.int64
            )
        else:
            env_indices_tensor = torch.nonzero(mask, as_tuple=False).flatten()

        if env_indices_tensor.numel() == 0:
            return base_observation if base_observation is not None else self.observation() if build_observation else {}

        learner_positions = self.learner.position.clone()
        learner_headings = self.learner.heading.clone()
        defender_positions = self.defenders.position.clone()
        defender_headings = self.defenders.heading.clone()
        if env_indices_tensor.numel():
            next_episodes = self.episode_counters[env_indices_tensor] + 1
            if episode_seeds_cpu is None:
                episode_seeds = compose_episode_seeds(
                    self.base_seed,
                    self.env_id_offset + env_indices_tensor,
                    next_episodes,
                )
            else:
                episode_seeds = episode_seeds_cpu.to(
                    device=self.device, dtype=torch.int64
                )
                if episode_seeds.shape != env_indices_tensor.shape:
                    raise ValueError("Reset episode seed count must match selected lanes")
            self._episode_seeds[env_indices_tensor] = episode_seeds
            self.terrain_override_revision[env_indices_tensor] = 0
            self.terrain_layout_attempt[env_indices_tensor] = 0
            if hasattr(self, "contact_static"):
                self.contact_static[env_indices_tensor] = False
            self.terrain_world_seed[env_indices_tensor] = episode_seeds
            self.map_id[env_indices_tensor] = 0
            self.map_origin_xy[env_indices_tensor] = periodic_map_origins(episode_seeds, self.terrain_layout_attempt[env_indices_tensor])
            self.terrain_manual_override[env_indices_tensor] = False
            if self._terrain_coverage is not None:
                self._terrain_coverage[0][env_indices_tensor] = torch.inf
                self._terrain_coverage[1][env_indices_tensor] = -torch.inf
            if any(self.terrain_overrides):
                for lane in env_indices_tensor.tolist():
                    self.terrain_overrides[lane].clear()
            self.terrain_next_manual_id[env_indices_tensor] = 0
            self.terrain_override_active[env_indices_tensor] = False
            scene = sample_scenes(episode_seeds, self.scene_spec, statistics=self.reset_statistics)
            self.targets.position[env_indices_tensor] = scene.target_positions

            learner_active = (
                torch.arange(
                    scene.learner_positions.shape[1], device=self.device
                )[None]
                < scene.learner_counts[:, None]
            )
            defender_active = (
                torch.arange(
                    scene.defender_positions.shape[1], device=self.device
                )[None]
                < scene.defender_counts[:, None]
            )
            (
                randomized_learner_headings,
                randomized_defender_headings,
                learner_speed_ratios,
                defender_speed_ratios,
            ) = episode_actor_randomization(
                episode_seeds,
                learner_counts=scene.learner_counts,
                learner_capacity=scene.learner_positions.shape[1],
                learner_speed_ratio_range=tuple(
                    float(value)
                    for value in getattr(
                        self, "learner_speed_ratio_range", (1.0, 1.0)
                    )
                ),
                defender_counts=scene.defender_counts,
                defender_capacity=scene.defender_positions.shape[1],
                defender_speed_ratio_range=tuple(
                    float(value)
                    for value in self.config.goal_reach["defender"]["speed_ratio"]
                ),
            )

            self.episode_counters[env_indices_tensor] = next_episodes
            learner_positions[env_indices_tensor] = scene.learner_positions[:, 0]
            learner_headings[env_indices_tensor] = randomized_learner_headings[:, 0]
            self.sampled_obstacle_count[env_indices_tensor] = scene.obstacle_counts.to(
                self.sampled_obstacle_count.dtype
            )
            self.obstacle_active[env_indices_tensor] = False
            self.obstacle_centers[env_indices_tensor] = 0.0
            self.obstacle_radii[env_indices_tensor] = 0.0
            sampled_obstacle_capacity = scene.obstacle_centers.shape[1]
            self.obstacle_active[env_indices_tensor, :sampled_obstacle_capacity] = (
                torch.arange(sampled_obstacle_capacity, device=self.device)[None]
                < scene.obstacle_counts[:, None]
            )
            self.obstacle_centers[env_indices_tensor, :sampled_obstacle_capacity] = (
                scene.obstacle_centers
            )
            self.obstacle_radii[env_indices_tensor, :sampled_obstacle_capacity] = (
                scene.obstacle_radii
            )
            self.defender_active[env_indices_tensor] = defender_active
            defender_positions[env_indices_tensor] = scene.defender_positions
            defender_headings[env_indices_tensor] = randomized_defender_headings
            self.defender_initial_position[env_indices_tensor] = scene.defender_positions
            self.defender_initial_heading[env_indices_tensor] = (
                randomized_defender_headings
            )

        self.learner_dynamics.reset(self.learner, learner_positions, learner_headings, mask)
        if self.defender_count:
            self.defender_dynamics.reset(
                self.defenders,
                defender_positions,
                defender_headings,
                mask[:, None],
            )
        self.physics_ticks[mask] = 0
        self.done[mask] = False
        self.success[mask] = False
        self.collision[mask] = False
        primary_learner_enabled = self.training_target == "learner"
        self.primary_learner_active[mask] = primary_learner_enabled
        if env_indices_tensor.numel():
            self._apply_scene_batch_extension(
                env_indices_tensor,
                scene,
                mask,
                learner_active,
                defender_active,
                randomized_learner_headings,
                randomized_defender_headings,
                learner_speed_ratios,
                defender_speed_ratios,
            )
            self._refresh_terrain(env_indices_tensor, initial=True)
            if hasattr(self, "sampled_learner_count"):
                spawns = sample_actor_spawns(
                    self._episode_seeds[env_indices_tensor],
                    world_radius=self.world.radius,
                    safety_zone_specification=self.config.goal_reach["safety_zones"],
                    vessel_radius=self.vessel_radius_m,
                    reference_positions=self.region_reference_position[env_indices_tensor],
                    target_positions=self.targets.position[env_indices_tensor],
                    target_counts=self.sampled_target_count[env_indices_tensor],
                    learner_capacity=int(getattr(self, "viewer_learner_capacity", 1)),
                    learner_counts=self.sampled_learner_count[env_indices_tensor],
                    defender_capacity=self.defender_count,
                    defender_counts=self.sampled_defender_count[env_indices_tensor],
                    goal_centers=self.goal_centers[env_indices_tensor],
                    goal_radii=self.goal_radii[env_indices_tensor],
                    goal_active=self.goal_active[env_indices_tensor],
                    obstacle_centers=self.obstacle_centers[env_indices_tensor],
                    obstacle_radii=self.obstacle_radii[env_indices_tensor],
                    obstacle_active=self.obstacle_active[env_indices_tensor],
                    validate_fixed_hazards=True,
                )
                self._apply_fixed_spawn_batch(env_indices_tensor, spawns, mask)
        if not build_observation:
            return {}
        if base_observation is None:
            return self.training_observation()
        refreshed = self.training_observation(mask)
        return merge_reset_observation(base_observation, refreshed, mask)

    def _scene_defender_counts(self, episode_seeds: torch.Tensor) -> torch.Tensor:
        return torch.full_like(episode_seeds, self.defender_count, dtype=torch.int64)

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
        del (
            env_indices,
            scene,
            mask,
            learner_active,
            defender_active,
            learner_headings,
            defender_headings,
            learner_speed_ratios,
            defender_speed_ratios,
        )

    def role_variant(self, role):
        return self.learner_variant if role == "learner" else self.patrol_variant

    def role_state(self, role):
        if role == "learner":
            return AuvState(**{name: value.unsqueeze(1) for name, value in self.learner.__dict__.items()})
        return self.defenders

    def viewer_learner_active(self, arena_index: int) -> torch.Tensor:
        arena_index = int(arena_index)
        return self.primary_learner_active[arena_index : arena_index + 1]

    def _viewer_learner_state(self, arena_index: int) -> tuple[torch.Tensor, ...]:
        arena_index = int(arena_index)
        return tuple(
            getattr(self.learner, name)[arena_index : arena_index + 1]
            for name in ("position", "heading", "body_velocity", "desired_body_velocity")
        )

    def set_viewer_learner_pose(
        self,
        arena_index: int,
        learner_index: int,
        position: tuple[float, float],
        heading: float,
    ) -> None:
        arena_index = int(arena_index)
        learner_index = int(learner_index)
        if learner_index != 0:
            raise IndexError("GoalReach contains exactly one infiltrator Actor")
        value = torch.tensor(position, dtype=torch.float32, device=self.device)
        angle = torch.tensor(float(heading), dtype=torch.float32, device=self.device)
        positions = self.learner.position.clone()
        headings = self.learner.heading.clone()
        positions[arena_index] = value
        headings[arena_index] = angle
        mask = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        mask[arena_index] = True
        self.learner_dynamics.reset(self.learner, positions, headings, mask)

    def _circle_semantic_map(
        self,
        map_headings: torch.Tensor,
        centers: torch.Tensor,
        radii: torch.Tensor,
        active: torch.Tensor,
        *,
        extra_centers: torch.Tensor | None = None,
        extra_radii: torch.Tensor | None = None,
        extra_active: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rasterize any active circle collection on the shared map device."""

        if extra_centers is not None and extra_radii is not None and extra_active is not None:
            centers = torch.cat((centers, extra_centers), dim=1)
            radii = torch.cat((radii, extra_radii), dim=1)
            active = torch.cat((active, extra_active), dim=1)
        centers, radii, active, _ = self._active_circle_columns(
            centers, radii, active
        )
        map_centers = centers.to(self.map_device)
        map_radii = radii.to(self.map_device)
        map_active = active.to(self.map_device)
        rotated = self._rotate_geometry(
            map_centers, map_headings[:, None]
        )
        offset = self._semantic_points_map[None, :, None, :] - rotated[:, None, :, :]
        occupied = (
            (offset.square().sum(dim=-1) <= map_radii[:, None, :].square())
            & map_active[:, None, :]
        ).any(dim=-1)
        return occupied

    def _obstacle_semantic_channel(
        self,
        arena_index: int,
        headings: torch.Tensor,
        *,
        include_goal: bool,
    ) -> torch.Tensor:
        count = int(headings.shape[0])
        goal_centers = (
            self.goal_centers[int(arena_index)].expand(count, -1, -1)
            if include_goal
            else None
        )
        goal_radii = (
            self.goal_radii[int(arena_index)].expand(count, -1) if include_goal else None
        )
        goal_active = (
            self.goal_active[int(arena_index)].expand(count, -1) if include_goal else None
        )
        blocked = self._circle_semantic_map(
            headings.to(self.map_device),
            self.obstacle_centers[int(arena_index)].expand(count, -1, -1),
            self.obstacle_radii[int(arena_index)].expand(count, -1),
            self.obstacle_active[int(arena_index)].expand(count, -1),
            extra_centers=goal_centers,
            extra_radii=goal_radii,
            extra_active=goal_active,
        )
        return blocked.to(self.device).reshape(count, 1, 96, 96).to(torch.uint8)

    @staticmethod
    def _active_circle_columns(
        centers: torch.Tensor,
        radii: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Drop unused CPU storage slots without introducing CUDA synchronization."""

        if centers.device.type != "cpu":
            return centers, radii, active, None
        indices = torch.nonzero(active.any(dim=0), as_tuple=False).flatten()
        return (
            centers.index_select(1, indices),
            radii.index_select(1, indices),
            active.index_select(1, indices),
            indices,
        )

    def _defender_hazard_circles(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the single authoritative circle set used by patrol geometry."""

        return (
            torch.cat((self.obstacle_centers, self.goal_centers), dim=1),
            torch.cat((self.obstacle_radii, self.goal_radii), dim=1),
            torch.cat((self.obstacle_active, self.goal_active), dim=1),
        )

    def set_patrol_architecture(self, architecture_id: str) -> None:
        variant = actor_variant(architecture_id)
        self.patrol_architecture_id = architecture_id
        self.patrol_variant = variant
        self.patrol_semantic_channels = variant.semantic_input_shape[0]
        self.patrol_vector_size = variant.vector_input_size

    def set_learner_architecture(self, architecture_id: str) -> None:
        variant = actor_variant(architecture_id)
        if variant.actor_kind != "learner":
            raise ValueError("GoalReach learner architecture must belong to the learner task")
        self.learner_variant = variant

    def _ownship_position_channel(self, map_position: torch.Tensor) -> torch.Tensor:
        marker_radius = 1.5 * self._semantic_cell_size
        return (
            (self._semantic_points[None] - map_position[:, None, :]).square().sum(dim=-1)
            <= marker_radius**2
        ).reshape(map_position.shape[0], 96, 96).to(torch.uint8)

    def _assemble_actor_observation(
        self,
        variant: ActorVariantSpec,
        semantic_components: dict[str, torch.Tensor],
        vector_components: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        actor_count = next(iter(vector_components.values())).shape[0]
        semantic_map = (
            torch.stack(
                tuple(semantic_components[name] for name in variant.semantic_channels),
                dim=1,
            ).to(torch.uint8)
            if variant.semantic_channels
            else torch.zeros(
                (actor_count, 0, 96, 96),
                dtype=torch.uint8,
                device=self.device,
            )
        )
        vector = torch.stack(
            tuple(vector_components[name] for name in variant.vector_inputs),
            dim=-1,
        )
        if variant.unified_match_contract:
            torch._assert_async(torch.isfinite(vector).all(), "Unified Actor vector must be finite")
            torch._assert_async((vector.abs() <= 1.0 + 1.0e-5).all(), "Unified Actor vector exceeds its registered normalization")
        else:
            vector = vector.clamp(-1.0, 1.0)
        critic_vector = (
            torch.stack(
                tuple(
                    vector_components[name]
                    for name in variant.critic_vector_input_names
                ),
                dim=-1,
            )
            if variant.critic_vector_input_names != variant.vector_inputs
            else None
        )
        extent = torch.full(
            (actor_count,), 2.0 * self.world.radius, dtype=torch.float32, device=self.device
        )
        observation = {
            "semantic_map": semantic_map,
            "vector": vector,
            "map_center": torch.zeros(
                (actor_count, 2), dtype=torch.float32, device=self.device
            ),
            "map_extent": extent,
            "meters_per_pixel": extent / 96.0,
        }
        if critic_vector is not None:
            observation["critic_vector"] = critic_vector
        return observation

    def _goal_reach_entity_buffer(
        self,
        positions: torch.Tensor,
        headings: torch.Tensor,
        centers: torch.Tensor,
        radii: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        relative = self._rotate_geometry(
            centers - positions[:, None, :],
            headings[:, None],
        ) / self.position_scale_m
        entities = torch.cat(
            (relative, (radii / self.position_scale_m).unsqueeze(-1)),
            dim=-1,
        )
        return entities * active.unsqueeze(-1).to(entities.dtype)

    @staticmethod
    def _order_goal_reach_entities(
        entities: torch.Tensor,
        active: torch.Tensor,
        ordering: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active = active.to(torch.bool)
        if ordering in {None, "storage"}:
            return entities, active
        inactive_key = torch.full_like(entities[..., 0], torch.inf)

        def stable_sort(key: torch.Tensor) -> None:
            nonlocal entities, active
            order = torch.argsort(
                torch.where(active, key, inactive_key),
                dim=1,
                stable=True,
            )
            entities = entities.gather(
                1, order.unsqueeze(-1).expand(-1, -1, entities.shape[-1])
            )
            active = active.gather(1, order)

        stable_sort(entities[..., 2])
        stable_sort(entities[..., 1])
        stable_sort(entities[..., 0])
        distance = entities[..., :2].square().sum(dim=-1)
        if ordering == "distance":
            stable_sort(distance)
        elif ordering == "absolute_bearing":
            stable_sort(distance)
            stable_sort(torch.atan2(entities[..., 0], entities[..., 1]).abs())
        else:
            raise ValueError(f"Unsupported entity ordering: {ordering}")
        return entities, active

    def _goal_reach_actor_observation(
        self,
        positions: torch.Tensor,
        headings: torch.Tensor,
        velocity: torch.Tensor,
        obstacle_centers: torch.Tensor,
        obstacle_radii: torch.Tensor,
        obstacle_active: torch.Tensor,
        goal_centers: torch.Tensor,
        goal_radii: torch.Tensor,
        goal_active: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        actor_count = positions.shape[0]
        if actor_count == 0 and self.learner_variant.input_kind == "semantic_map":
            empty_semantic = torch.zeros(
                (0, 96, 96), dtype=torch.uint8, device=self.device
            )
            empty_vector = torch.zeros(0, dtype=torch.float32, device=self.device)
            return self._assemble_actor_observation(
                self.learner_variant,
                {name: empty_semantic for name in self.learner_variant.semantic_channels},
                {name: empty_vector for name in self.learner_variant.vector_inputs},
            )

        map_position = self._rotate_geometry(positions, headings)
        vector_components = {
            "map_frame_x_over_R": map_position[:, 0] / self.world.radius,
            "map_frame_z_over_R": map_position[:, 1] / self.world.radius,
            "sin_heading": headings.sin(),
            "cos_heading": headings.cos(),
            "surge_over_2": velocity[:, 0] / 2.0,
            "sway": velocity[:, 1],
            "yaw_rate_over_0_2": velocity[:, 2] / 0.2,
        }
        if self.learner_variant.uses_entity_buffers:
            observation = self._assemble_actor_observation(
                self.learner_variant,
                {},
                vector_components,
            )
            goal_entities, goal_mask = self._order_goal_reach_entities(
                self._goal_reach_entity_buffer(
                    positions,
                    headings,
                    goal_centers,
                    goal_radii,
                    goal_active,
                ),
                goal_active,
                self.learner_variant.entity_ordering,
            )
            obstacle_entities, obstacle_mask = self._order_goal_reach_entities(
                self._goal_reach_entity_buffer(
                    positions,
                    headings,
                    obstacle_centers,
                    obstacle_radii,
                    obstacle_active,
                ),
                obstacle_active,
                self.learner_variant.entity_ordering,
            )
            observation.update(
                {
                    "goal_entities": goal_entities,
                    "goal_mask": goal_mask,
                    "obstacle_entities": obstacle_entities,
                    "obstacle_mask": obstacle_mask,
                }
            )
            return observation

        map_headings = headings.to(self.map_device)
        obstacle = self._circle_semantic_map(
            map_headings,
            obstacle_centers,
            obstacle_radii,
            obstacle_active,
        )
        goal_region = self._circle_semantic_map(
            map_headings,
            goal_centers,
            goal_radii,
            goal_active,
        )
        circle_maps = torch.stack((obstacle, goal_region), dim=1).to(self.device)
        semantic_components = {
            "true_fatal_geometry": circle_maps[:, 0].reshape(actor_count, 96, 96),
            "goal_region": circle_maps[:, 1].reshape(actor_count, 96, 96),
        }
        if "ownship_position" in self.learner_variant.semantic_channels:
            semantic_components["ownship_position"] = self._ownship_position_channel(
                map_position
            )
        return self._assemble_actor_observation(
            self.learner_variant, semantic_components, vector_components
        )

    def observe_viewer_learners(self, arena_index: int) -> dict[str, torch.Tensor]:
        positions, headings, velocity, _ = self._viewer_learner_state(arena_index)
        actor_indices = torch.nonzero(
            self.viewer_learner_active(arena_index), as_tuple=False
        ).flatten()
        actor_count = actor_indices.shape[0]
        return self._goal_reach_actor_observation(
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

    def viewer_learner_rule_action(self, arena_index: int) -> torch.Tensor:
        positions, headings, _, _ = self._viewer_learner_state(arena_index)
        actor_indices = torch.nonzero(
            self.viewer_learner_active(arena_index), as_tuple=False
        ).flatten()
        positions = positions[actor_indices]
        headings = headings[actor_indices]
        goal_offset = self.goal_centers[int(arena_index)][None, :, :] - positions[:, None, :]
        goal_distance = goal_offset.norm(dim=-1)
        goal_distance = torch.where(
            self.goal_active[int(arena_index)][None, :],
            goal_distance,
            torch.full_like(goal_distance, torch.inf),
        )
        nearest_goal = goal_distance.argmin(dim=-1)
        attraction = goal_offset[
            torch.arange(positions.shape[0], device=self.device), nearest_goal
        ]
        has_goal = self.goal_active[int(arena_index)].any()
        attraction = torch.where(has_goal, attraction, torch.zeros_like(attraction))
        attraction = attraction / attraction.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        centers = self.obstacle_centers[int(arena_index)]
        radii = self.obstacle_radii[int(arena_index)]
        active = self.obstacle_active[int(arena_index)]
        offset = positions[:, None, :] - centers[None, :, :]
        distance = offset.norm(dim=-1)
        surface = distance - radii[None, :]
        influence = 15.0
        weight = ((influence - surface) / influence).clamp(0.0, 1.0).square()
        weight *= active[None, :]
        repulsion = (offset / distance.unsqueeze(-1).clamp_min(1.0e-8) * weight.unsqueeze(-1)).sum(
            dim=1
        )
        boundary_distance = self.world.radius - positions.norm(dim=-1)
        boundary_weight = ((influence - boundary_distance) / influence).clamp(0.0, 1.0).square()
        repulsion -= (
            positions / positions.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        ) * boundary_weight.unsqueeze(-1)
        direction = attraction + 2.0 * repulsion
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        desired_heading = torch.atan2(direction[:, 0], direction[:, 1])
        heading_error = (
            torch.remainder(desired_heading - headings + torch.pi, 2.0 * torch.pi) - torch.pi
        )
        alignment = torch.cos(heading_error).clamp(0.0, 1.0)
        return torch.stack(
            (2.0 * alignment - 1.0, (2.0 * heading_error).clamp(-1.0, 1.0)),
            dim=-1,
        )

    def _restart_patrol(self, mask: torch.Tensor) -> None:
        if self.defender_count == 0:
            return
        mask = mask.to(device=self.device, dtype=torch.bool)
        self.patrol_waypoint[mask] = 0.0
        self.patrol_hold_ticks[mask] = 0
        self.defender_avoidance_hazard_id[mask] = -1
        self.defender_avoidance_side[mask] = 0
        self.defender_avoidance_surface_distance[mask] = torch.inf
        episode_seeds = self._episode_seeds.to(device=self.device)[:, None]
        defender_indices = torch.arange(self.defender_count, dtype=torch.int64, device=self.device)[
            None, :
        ]
        seeded = torch.remainder(
            episode_seeds + (defender_indices + 2) * 2_654_435_761,
            self._tensor_rng_modulus,
        )
        seeded = torch.remainder(
            seeded * self._tensor_rng_multiplier + self._tensor_rng_increment,
            self._tensor_rng_modulus,
        )
        reset_state = torch.where(self.defender_active, seeded, 0)
        self.defender_rng_state = torch.where(mask[:, None], reset_state, self.defender_rng_state)
        self._plan_patrol_for(mask[:, None] & self.defender_active)

    def restart_edited_episode(self, mask: torch.Tensor) -> None:
        mask = mask.to(device=self.device, dtype=torch.bool)
        if self.defender_count:
            self.defender_dynamics.reset(
                self.defenders,
                self.defenders.position.clone(),
                self.defenders.heading.clone(),
                mask[:, None],
            )
        self.physics_ticks[mask] = 0
        self.done[mask] = False
        self.success[mask] = False
        self.collision[mask] = False
        self.primary_learner_active[mask] = self.training_target == "learner"

    def state_dict(self) -> dict[str, object]:
        return {
            "state_schema": self.state_schema,
            "training_target": self.training_target,
            "defender_count": self.defender_count,
            "base_seed": self.base_seed,
            "batch_size": self.batch_size,
            "env_id_offset": self.env_id_offset,
            "scene_contract": dict(self.scene_contract),
            "learner": {
                name: value.detach().cpu() for name, value in self.learner.__dict__.items()
            },
            "defenders": {
                name: value.detach().cpu() for name, value in self.defenders.__dict__.items()
            },
            "defender_initial_position": self.defender_initial_position.cpu(),
            "defender_initial_heading": self.defender_initial_heading.cpu(),
            "sampled_goal_count": self.sampled_goal_count.cpu(),
            "goal_active": self.goal_active.cpu(),
            "goal_centers": self.goal_centers.cpu(),
            "goal_radii": self.goal_radii.cpu(),
            "primary_learner_active": self.primary_learner_active.cpu(),
            "sampled_obstacle_count": self.sampled_obstacle_count.cpu(),
            "obstacle_active": self.obstacle_active.cpu(),
            "obstacle_centers": self.obstacle_centers.cpu(),
            "obstacle_radii": self.obstacle_radii.cpu(),
            "obstacle_keys": self.obstacle_keys.cpu(),
            "terrain_world_seed": self.terrain_world_seed.cpu(),
            "map_id": self.map_id.cpu(),
            "map_origin_xy": self.map_origin_xy.cpu(),
            "terrain_override_revision": self.terrain_override_revision.cpu(),
            "terrain_layout_attempt": self.terrain_layout_attempt.cpu(),
            "terrain_phase": self.terrain_phase.cpu(),
            "terrain_manual_override": self.terrain_manual_override.cpu(),
            "terrain_overrides": copy.deepcopy(self.terrain_overrides),
            "terrain_next_manual_id": self.terrain_next_manual_id.cpu(),
            "terrain_override_active": self.terrain_override_active.cpu(),
            "defender_active": self.defender_active.cpu(),
            "patrol_waypoint": self.patrol_waypoint.cpu(),
            "patrol_hold_ticks": self.patrol_hold_ticks.cpu(),
            "defender_avoidance_hazard_id": self.defender_avoidance_hazard_id.cpu(),
            "defender_avoidance_side": self.defender_avoidance_side.cpu(),
            "defender_rng_state": self.defender_rng_state.cpu(),
            "physics_ticks": self.physics_ticks.cpu(),
            "episode_counters": self.episode_counters.cpu().clone(),
            "done": self.done.cpu(),
            "success": self.success.cpu(),
            "collision": self.collision.cpu(),
            "episode_seeds": self._episode_seeds.cpu().clone(),
        }

    def _validate_scene_restore(self, state: dict[str, object]) -> None:
        if "scene_contract" not in state or "episode_counters" not in state:
            raise ValueError("Checkpoint has no online episode RNG state")
        reject_scene_contract_differences(
            scene_contract_diff(state['scene_contract'], self.scene_contract)
        )
        if state.get("state_schema") != self.state_schema:
            raise ValueError("GoalReach checkpoint environment state schema mismatch")
        maps, origin = state.get('map_id'), state.get('map_origin_xy')
        if (not isinstance(maps, torch.Tensor) or maps.dtype != torch.int64
                or maps.shape != (self.batch_size,) or not isinstance(origin, torch.Tensor)
                or origin.dtype != torch.float32 or origin.shape != (self.batch_size,2)
                or not bool(torch.isfinite(origin).all())
                or bool((maps != 0).any())
                or not torch.equal(origin, periodic_map_origins(state['episode_seeds'].to(origin.device),
                                                                state['terrain_layout_attempt'].to(origin.device)))):
            raise ValueError('Checkpoint periodic map identity/origin is invalid')
        if state['obstacle_keys'].shape != (*state['obstacle_active'].shape,4):
            raise ValueError('Checkpoint periodic obstacle identity shape is invalid')

    def load_state_dict(self, state: dict[str, object], *, source_launch_settings=None) -> None:
        self._validate_scene_restore(state)
        self._terrain_coverage = None
        if state.get("training_target") != self.training_target:
            raise ValueError("GoalReach checkpoint training target mismatch")
        checkpoint_defender_count = int(
            state.get(
                "defender_count",
                0 if self.training_target == "learner" else 1,
            )
        )
        if checkpoint_defender_count != self.defender_count:
            raise ValueError("GoalReach checkpoint defender count mismatch")
        identity = (state["base_seed"], state["batch_size"], state.get("env_id_offset", 0))
        expected = (self.base_seed, self.batch_size, self.env_id_offset)
        if identity != expected:
            raise ValueError("GoalReach checkpoint environment identity mismatch")
        self._invalidate_obstacle_neighbors()
        for container_name in ("learner", "defenders"):
            container_state = state[container_name]
            assert isinstance(container_state, dict)
            container = getattr(self, container_name)
            for name, value in container_state.items():
                setattr(container, name, value.to(self.device))
        tensor_names = (
            "sampled_goal_count",
            "goal_active",
            "goal_centers",
            "goal_radii",
            "primary_learner_active",
            "defender_initial_position",
            "defender_initial_heading",
            "sampled_obstacle_count",
            "obstacle_active",
            "obstacle_centers",
            "obstacle_radii",
            "obstacle_keys",
            "terrain_world_seed",
            "map_id",
            "map_origin_xy",
            "terrain_override_revision",
            "terrain_layout_attempt",
            "terrain_phase",
            "terrain_manual_override",
            "terrain_next_manual_id",
            "terrain_override_active",
            "defender_active",
            "patrol_waypoint",
            "patrol_hold_ticks",
            "defender_avoidance_hazard_id",
            "defender_avoidance_side",
            "defender_rng_state",
            "physics_ticks",
            "done",
            "success",
            "collision",
        )
        for name in tensor_names:
            setattr(self, name, state[name].to(self.device))
        # Restoring a smaller snapshot also shrinks capacity metadata; otherwise
        # later contact remapping mistakes boundary slots for obstacle slots.
        self.goal_capacity = self.goal_centers.shape[1]
        self.obstacle_capacity = self.obstacle_centers.shape[1]
        self.config.goal_reach["reset"]["random_obstacles"]["capacity"] = self.obstacle_capacity
        self.episode_counters = state["episode_counters"].to(self.device).clone()
        self._episode_seeds = state["episode_seeds"].to(self.device).clone()
        self.terrain_overrides = copy.deepcopy(state["terrain_overrides"])

    @torch.no_grad()
    def training_step(
        self, action: torch.Tensor, *, build_observation: bool = True
    ) -> GoalReachStep:
        if self.training_target == "learner":
            return self._step_training_learner(action, build_observation=build_observation)
        if self.training_target == "defender":
            defender_step = getattr(self, "_step_training_defender", None)
            if defender_step is None:
                raise RuntimeError(
                    "Standalone patrol exploration was removed; use MultiGoalReachEnv"
                )
            return defender_step(action, build_observation=build_observation)
        raise RuntimeError("training_step requires a single training target")

    def _resolved_step_physics_ticks(self, physics_ticks: int | None) -> int:
        configured = int(self.config.goal_reach["clock"]["physics_ticks_per_decision"])
        resolved = configured if physics_ticks is None else int(physics_ticks)
        if not 1 <= resolved <= configured:
            raise ValueError(f"Physics ticks per step must be inside [1,{configured}]")
        return resolved

    def _step_training_learner(
        self,
        action: torch.Tensor,
        *,
        build_observation: bool = True,
        physics_ticks: int | None = None,
        finish_decision: bool = True,
    ) -> GoalReachStep:
        action = action.to(device=self.device, dtype=torch.float32)
        if action.shape != (self.batch_size, 2):
            raise ValueError(f"Expected action [{self.batch_size},2], got {tuple(action.shape)}")
        self.learner_dynamics.set_normalized_action(self.learner, action)
        terminated = torch.zeros_like(self.done)
        truncated = torch.zeros_like(self.done)
        boundary_collision = torch.zeros_like(self.done)
        obstacle_collision = torch.zeros_like(self.done)
        horizon = torch.zeros_like(self.done)
        active = ~self.done
        clock = self.config.goal_reach["clock"]
        for _ in range(self._resolved_step_physics_ticks(physics_ticks)):
            (
                collision,
                tick_boundary_collision,
                tick_obstacle_collision,
                previous,
                desired,
            ) = self._step_learner(active & self.primary_learner_active)
            boundary_collision |= tick_boundary_collision
            obstacle_collision |= tick_obstacle_collision
            reached = (
                self._goal_contact(previous, desired)
                & active
                & ~collision
            )
            self.physics_ticks[active] += 1
            maximum_physics_ticks = clock["maximum_physics_ticks"]
            timeout = (
                torch.zeros_like(active)
                if maximum_physics_ticks is None
                else (
                    (self.physics_ticks >= int(maximum_physics_ticks))
                    & active
                    & ~collision
                    & ~reached
                )
            )
            terminated |= collision | reached
            self._apply_time_limit_boundary(timeout, terminated, truncated)
            horizon |= timeout
            self.collision |= collision
            self.success |= reached
            self.primary_learner_active &= ~(collision | reached)
            self.done |= collision | reached | timeout
            active = ~self.done
        reward = (
            torch.where(
                terminated,
                torch.where(
                    self.success,
                    torch.ones_like(terminated, dtype=torch.float32),
                    -torch.ones_like(terminated, dtype=torch.float32),
                ),
                torch.zeros_like(terminated, dtype=torch.float32),
            )
            if finish_decision
            else torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        )
        return GoalReachStep(
            observation=(
                self.training_observation()
                if build_observation and finish_decision
                else {}
            ),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={
                "success": self.success.clone(),
                "collision": self.collision.clone(),
                "boundary_collision": boundary_collision,
                "obstacle_collision": obstacle_collision,
                "pair_collision": torch.zeros_like(self.done),
                "horizon": horizon,
                "physics_ticks": self.physics_ticks.clone(),
            },
        )

    @staticmethod
    def _rotate_geometry(points: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
        cosine = heading.cos()
        sine = heading.sin()
        x = cosine * points[..., 0] - sine * points[..., 1]
        z = sine * points[..., 0] + cosine * points[..., 1]
        return torch.stack((x, z), dim=-1)

    def observe(self, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        selection: slice | torch.Tensor
        if mask is None:
            selection = slice(None)
        else:
            mask = mask.to(device=self.device, dtype=torch.bool)
            selection = torch.nonzero(mask, as_tuple=False).flatten()
        return self._goal_reach_actor_observation(
            self.learner.position[selection],
            self.learner.heading[selection],
            self.learner.body_velocity[selection],
            self.obstacle_centers[selection],
            self.obstacle_radii[selection],
            self.obstacle_active[selection],
            self.goal_centers[selection],
            self.goal_radii[selection],
            self.goal_active[selection],
        )

    def training_observation(
        self, mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if self.training_target == "defender":
            raise RuntimeError(
                "Standalone patrol exploration was removed; use MultiGoalReachEnv"
            )
        return self.observe(mask)

    def _step_learner(
        self, active: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        previous = self.learner.position
        self.learner_dynamics.step(self.learner, active)
        desired = self.learner.position
        boundary_collision = torch.zeros_like(active)
        obstacle_collision = self._obstacle_contact(previous, desired) & active
        blocked = boundary_collision | obstacle_collision
        self.learner_dynamics.reconcile_planar_constraint(self.learner, previous, blocked)
        return blocked, boundary_collision, obstacle_collision, previous, desired

    def _tensor_random_uniform(self, mask: torch.Tensor) -> torch.Tensor:
        next_state = torch.remainder(
            self.defender_rng_state * self._tensor_rng_multiplier + self._tensor_rng_increment,
            self._tensor_rng_modulus,
        )
        self.defender_rng_state = torch.where(mask, next_state, self.defender_rng_state)
        return next_state.to(torch.float32) / float(self._tensor_rng_modulus)

    def _plan_patrol_for(self, mask: torch.Tensor) -> None:
        defender = self.config.goal_reach["defender"]
        mask = mask.to(device=self.device, dtype=torch.bool)
        start = self.defenders.position
        heading = self.defenders.heading
        phase = self._tensor_random_uniform(mask) * (2.0 * torch.pi)
        angles = phase.unsqueeze(-1) + self._patrol_direction_offsets
        directions = torch.stack((angles.sin(), angles.cos()), dim=-1)

        circle_centers, circle_radii, circle_active = self._defender_hazard_circles()
        full_clearance = (
            self.vessel_radius_m
            + float(defender["avoidance_inner_distance_m"])
            + float(defender["patrol_waypoint_reached_distance_m"])
        )
        collision_clearance = self.vessel_radius_m
        start_circle_distance = (start[:, :, None, :] - circle_centers[:, None, :, :]).norm(dim=-1)
        start_circle_clear = (
            (start_circle_distance > circle_radii[:, None, :] + full_clearance)
            | ~circle_active[:, None, :]
        ).all(dim=-1)
        safety_zones = self.config.goal_reach["safety_zones"]
        safety_zones_enabled = bool(safety_zones.get("enabled", False))
        target = self.region_reference_position[:, None, :]
        target_active = torch.ones(start.shape[:2], dtype=torch.bool, device=self.device)
        start_boundary_clear = torch.ones_like(target_active)
        if safety_zones_enabled:
            start_boundary_clear &= (
                (start - target).norm(dim=-1)
                <= float(safety_zones["engagement_radius_m"]) - full_clearance
            ) | ~target_active
        path_clearance = torch.where(
            start_circle_clear & start_boundary_clear,
            torch.full_like(heading, full_clearance),
            torch.full_like(heading, collision_clearance),
        )

        start_expanded = start[:, :, None, :]
        boundary_exit = torch.full_like(angles, torch.inf)
        if safety_zones_enabled:
            relative_target_start = start_expanded - target[:, :, None, :]
            target_projection = (relative_target_start * directions).sum(dim=-1)
            effective_engagement_radius = (
                float(safety_zones["engagement_radius_m"])
                - path_clearance
            ).clamp_min(0.0)
            target_discriminant = (
                target_projection.square()
                + effective_engagement_radius.unsqueeze(-1).square()
                - relative_target_start.square().sum(dim=-1)
            ).clamp_min(0.0)
            target_exit = -target_projection + target_discriminant.sqrt()
            boundary_exit = torch.where(
                target_active[:, :, None],
                torch.minimum(boundary_exit, target_exit),
                boundary_exit,
            )

        relative_centers = circle_centers[:, None, None, :, :] - start[:, :, None, None, :]
        projection = (relative_centers * directions[:, :, :, None, :]).sum(dim=-1)
        perpendicular_squared = (
            relative_centers.square().sum(dim=-1) - projection.square()
        ).clamp_min(0.0)
        path_radii = circle_radii[:, None, None, :] + path_clearance[:, :, None, None]
        circle_discriminant = path_radii.square() - perpendicular_squared
        half_chord = circle_discriminant.clamp_min(0.0).sqrt()
        near = projection - half_chord
        far = projection + half_chord
        intersects = circle_active[:, None, None, :] & (circle_discriminant >= 0.0) & (far >= 0.0)
        circle_entry = torch.where(
            intersects,
            near.clamp_min(0.0),
            torch.full_like(near, torch.inf),
        ).amin(dim=-1)
        maximum = float(defender["patrol_waypoint_maximum_distance_m"])
        ray_limit = torch.minimum(boundary_exit, circle_entry).clamp_max(maximum)
        available = (ray_limit - 2.0).clamp_min(0.0)

        candidate_distances = available.unsqueeze(-1) * self._patrol_distance_scales
        endpoints = start[:, :, None, None, :] + directions[
            :, :, :, None, :
        ] * candidate_distances.unsqueeze(-1)
        endpoint_boundary_clear = torch.ones(endpoints.shape[:-1], dtype=torch.bool, device=self.device)
        if safety_zones_enabled:
            endpoint_boundary_clear &= (
                endpoints - target[:, :, None, None]
            ).norm(dim=-1) <= float(safety_zones["engagement_radius_m"]) - full_clearance
        endpoint_circle_distance = (
            endpoints[..., None, :] - circle_centers[:, None, None, None, :, :]
        ).norm(dim=-1)
        endpoint_circle_clear = (
            (endpoint_circle_distance > circle_radii[:, None, None, None, :] + full_clearance)
            | ~circle_active[:, None, None, None, :]
        ).all(dim=-1)
        legal_scale = (
            (candidate_distances >= float(defender["patrol_waypoint_minimum_distance_m"]))
            & endpoint_boundary_clear
            & endpoint_circle_clear
        )
        legal = legal_scale.any(dim=-1)
        scale_index = legal_scale.to(torch.int64).argmax(dim=-1)
        selected_distance = candidate_distances.gather(-1, scale_index.unsqueeze(-1)).squeeze(-1)
        candidate_points = start_expanded + directions * selected_distance.unsqueeze(-1)

        forward = torch.stack((heading.sin(), heading.cos()), dim=-1)
        alignment = (directions * forward.unsqueeze(-2)).sum(dim=-1)
        negative_infinity = torch.full_like(selected_distance, -torch.inf)
        forward_legal = legal & (alignment >= 0.0)
        forward_score = torch.where(
            forward_legal,
            selected_distance + 0.01 * alignment,
            negative_infinity,
        )
        forward_index = forward_score.argmax(dim=-1)
        has_forward = forward_legal.any(dim=-1)
        fallback_score = torch.where(
            legal,
            alignment + 0.001 * selected_distance / maximum,
            negative_infinity,
        )
        fallback_index = fallback_score.argmax(dim=-1)
        has_fallback = legal.any(dim=-1)
        selected_index = torch.where(has_forward, forward_index, fallback_index)
        waypoint = candidate_points.gather(
            -2, selected_index[..., None, None].expand(-1, -1, 1, 2)
        ).squeeze(-2)
        has_solution = has_forward | has_fallback
        waypoint = torch.where(has_solution.unsqueeze(-1), waypoint, start)

        waypoint_offset = waypoint - start
        desired_heading = torch.atan2(waypoint_offset[..., 0], waypoint_offset[..., 1])
        heading_error = (
            torch.remainder(desired_heading - heading + torch.pi, 2.0 * torch.pi) - torch.pi
        )
        alignment_turn = (heading_error.abs() - torch.pi / 2.0).clamp_min(0.0)
        hold_ticks = torch.ceil(
            alignment_turn
            / (
                float(defender["patrol_yaw_rate_max_radps"])
                * float(self.config.shared["clock"]["physics_step_s"])
            )
        ).to(self.patrol_hold_ticks.dtype)
        hold_ticks = torch.where(
            has_solution,
            hold_ticks,
            torch.full_like(hold_ticks, 50),
        )
        self.patrol_waypoint = torch.where(mask.unsqueeze(-1), waypoint, self.patrol_waypoint)
        self.patrol_hold_ticks = torch.where(mask, hold_ticks, self.patrol_hold_ticks)

    def _environment_dangers(
        self,
        points: torch.Tensor,
        target: torch.Tensor,
        target_direction: torch.Tensor,
        motion_direction: torch.Tensor,
        clearance: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        circle_centers, circle_radii, circle_active = self._defender_hazard_circles()
        circle_radii = circle_radii + clearance

        offset = points[:, :, None, :] - circle_centers[:, None, :, :]
        center_distance = offset.norm(dim=-1)
        circle_normal = offset / center_distance.unsqueeze(-1).clamp_min(1.0e-8)
        circle_distance = center_distance - circle_radii[:, None, :]

        to_center = -offset
        turn_radius = float(self.config.goal_reach["defender"]["avoidance_inner_distance_m"])
        influence_radii = circle_radii[:, None, :] + turn_radius

        def circle_entry(direction: torch.Tensor, maximum: torch.Tensor | None) -> torch.Tensor:
            projection = (to_center * direction[:, :, None, :]).sum(dim=-1)
            perpendicular_squared = (
                to_center.square().sum(dim=-1) - projection.square()
            ).clamp_min(0.0)
            discriminant = influence_radii.square() - perpendicular_squared
            hit = (projection > 0.0) & (discriminant >= 0.0)
            entry = (projection - discriminant.clamp_min(0.0).sqrt()).clamp_min(0.0)
            if maximum is not None:
                hit &= entry <= maximum.unsqueeze(-1)
            inside = center_distance <= influence_radii
            return torch.where(inside, 0.0, torch.where(hit, entry, torch.inf))

        target_entry = circle_entry(target_direction, (target - points).norm(dim=-1))
        motion_entry = circle_entry(motion_direction, None)
        circle_imminence = torch.minimum(target_entry, motion_entry)

        reference_offset = points - self.region_reference_position[:, None]
        radial_norm = reference_offset.norm(dim=-1)
        engagement = float(self.config.goal_reach["safety_zones"]["engagement_radius_m"])
        boundary_distance = engagement - clearance - radial_norm
        boundary_normal = -reference_offset / radial_norm.unsqueeze(-1).clamp_min(1.0e-8)
        influence_boundary_radius = engagement - clearance - turn_radius

        def boundary_entry(direction: torch.Tensor) -> torch.Tensor:
            radial_projection = (reference_offset * direction).sum(dim=-1)
            discriminant = (
                radial_projection.square() + influence_boundary_radius**2 - radial_norm.square()
            ).clamp_min(0.0)
            entry = -radial_projection + discriminant.sqrt()
            outside_influence = radial_norm >= influence_boundary_radius
            return torch.where(outside_influence, 0.0, entry.clamp_min(0.0))

        boundary_imminence = torch.minimum(
            boundary_entry(target_direction), boundary_entry(motion_direction)
        )
        boundary_active = torch.ones_like(boundary_distance, dtype=torch.bool)

        return (
            torch.cat((circle_distance, boundary_distance.unsqueeze(-1)), dim=-1),
            torch.cat((circle_normal, boundary_normal.unsqueeze(-2)), dim=-2),
            torch.cat(
                (
                    circle_active[:, None, :].expand(-1, self.defender_count, -1),
                    boundary_active.unsqueeze(-1),
                ),
                dim=-1,
            ),
            torch.cat((circle_imminence, boundary_imminence.unsqueeze(-1)), dim=-1),
        )

    def _defender_navigation_direction(
        self, target: torch.Tensor, active_defenders: torch.Tensor
    ) -> torch.Tensor:
        defender = self.config.goal_reach["defender"]
        position = self.defenders.position
        forward = torch.stack((self.defenders.heading.sin(), self.defenders.heading.cos()), dim=-1)
        target_offset = target - position
        target_distance = target_offset.norm(dim=-1, keepdim=True)
        target_direction = target_offset / target_distance.clamp_min(1.0e-8)
        target_direction = torch.where(target_distance > 1.0e-6, target_direction, forward)

        pair_delta = position[:, :, None, :] - position[:, None, :, :]
        pair_distance = pair_delta.norm(dim=-1)
        pair_active = self.defender_active[:, :, None] & self.defender_active[:, None, :]
        peer_distance = float(defender["peer_repulsion_distance_m"])
        peer_valid = (pair_distance > 0.01) & (pair_distance < peer_distance) & pair_active
        peer_direction = pair_delta / pair_distance.unsqueeze(-1).clamp_min(1.0e-8)
        peer_q = ((peer_distance - pair_distance) / peer_distance).clamp(0.0, 1.0)
        peer_weight = peer_q.square() * (3.0 - 2.0 * peer_q)
        peer = (
            peer_direction
            * (peer_valid * peer_weight * float(defender["peer_repulsion_weight"])).unsqueeze(-1)
        ).sum(dim=2)
        separated_target = target_direction + peer
        separated_target = separated_target / separated_target.norm(dim=-1, keepdim=True).clamp_min(
            1.0e-8
        )

        clearance = self.vessel_radius_m
        distances, normals, hazard_active, imminence = self._environment_dangers(
            position, target, target_direction, forward, clearance
        )
        outer = float(defender["avoidance_outer_distance_m"])
        release = float(defender["avoidance_release_distance_m"])
        candidate = (
            hazard_active & imminence.isfinite() & (imminence <= outer) & (distances <= outer)
        )
        candidate &= active_defenders.unsqueeze(-1)
        candidate_distance = torch.where(candidate, distances, torch.inf)
        nearest_distance, nearest_id = candidate_distance.min(dim=-1)
        has_nearest = nearest_distance.isfinite()

        locked_id = self.defender_avoidance_hazard_id
        locked_index = locked_id.clamp_min(0).unsqueeze(-1)
        locked_distance = distances.gather(-1, locked_index).squeeze(-1)
        locked_active = hazard_active.gather(-1, locked_index).squeeze(-1)
        preempt_lock = (
            has_nearest
            & (nearest_id != locked_id)
            & (nearest_distance <= float(defender["avoidance_inner_distance_m"]))
            & (nearest_distance + (release - outer) < locked_distance)
        )
        retain_lock = (
            (locked_id >= 0)
            & locked_active
            & (locked_distance <= release)
            & active_defenders
            & ~preempt_lock
        )
        selected_id = torch.where(retain_lock, locked_id, nearest_id)
        selected = retain_lock | has_nearest
        selected_id = torch.where(selected, selected_id, -1)
        selected_index = selected_id.clamp_min(0).unsqueeze(-1)
        selected_distance = distances.gather(-1, selected_index).squeeze(-1)
        selected_normal = normals.gather(
            -2, selected_index.unsqueeze(-1).expand(-1, -1, 1, 2)
        ).squeeze(-2)

        positive_tangent = torch.stack((selected_normal[..., 1], -selected_normal[..., 0]), dim=-1)
        positive_is_shorter = (positive_tangent * forward).sum(dim=-1) >= (
            -positive_tangent * forward
        ).sum(dim=-1)
        new_side = torch.where(
            positive_is_shorter,
            torch.ones_like(self.defender_avoidance_side),
            -torch.ones_like(self.defender_avoidance_side),
        )
        same_locked_hazard = selected & (selected_id == locked_id) & (locked_id >= 0)
        selected_side = torch.where(
            same_locked_hazard,
            self.defender_avoidance_side,
            torch.where(selected, new_side, 0),
        )
        tangent = positive_tangent * selected_side.unsqueeze(-1).to(torch.float32)

        environment_target = torch.where(selected.unsqueeze(-1), target_direction, separated_target)
        direction = smooth_tangent_navigation(
            environment_target,
            tangent,
            selected_normal,
            selected_distance,
            outer,
            float(defender["avoidance_inner_distance_m"]),
            float(defender["avoidance_outward_normal_coefficient"]),
        )
        direction = torch.where(selected.unsqueeze(-1), direction, separated_target)
        self.defender_avoidance_hazard_id = torch.where(active_defenders, selected_id, -1)
        self.defender_avoidance_side = torch.where(active_defenders, selected_side, 0).to(
            torch.int8
        )
        self.defender_avoidance_surface_distance = torch.where(
            selected & active_defenders,
            selected_distance,
            torch.full_like(selected_distance, torch.inf),
        )
        return direction

    def _step_defenders(self, active: torch.Tensor) -> torch.Tensor:
        defender = self.config.goal_reach["defender"]
        active_defenders = active[:, None] & self.defender_active
        reached = (self.defenders.position - self.patrol_waypoint).norm(dim=-1) < float(
            defender["patrol_waypoint_reached_distance_m"]
        )
        needs_plan = reached & (self.patrol_hold_ticks <= 0) & active_defenders
        self._plan_patrol_for(needs_plan)
        holding = self.patrol_hold_ticks > 0
        self.patrol_hold_ticks = torch.where(
            holding & active_defenders, self.patrol_hold_ticks - 1, self.patrol_hold_ticks
        )
        direction = self._defender_navigation_direction(
            self.patrol_waypoint, active_defenders
        )
        desired_heading = torch.atan2(direction[..., 0], direction[..., 1])
        heading_error = (
            torch.remainder(
                desired_heading - self.defenders.heading + torch.pi, 2.0 * torch.pi
            )
            - torch.pi
        )
        maximum_surge = torch.full_like(
            self.defenders.heading, float(defender["patrol_surge_mps"])
        )
        maximum_yaw = torch.full_like(
            maximum_surge, float(defender["patrol_yaw_rate_max_radps"])
        )
        forward = torch.stack(
            (self.defenders.heading.sin(), self.defenders.heading.cos()), dim=-1
        )
        heading_alignment = (direction * forward).sum(dim=-1).clamp(0.0, 1.0)
        surface_alignment = (
            self.defender_avoidance_surface_distance
            / float(defender["avoidance_inner_distance_m"])
        ).clamp(0.0, 1.0)
        surge_command = maximum_surge * heading_alignment * surface_alignment.square()
        surge_command = torch.where(holding, 0.0, surge_command)
        surge_command = torch.where(active_defenders, surge_command, 0.0)
        yaw_command = (heading_error * float(defender["patrol_heading_gain"])).clamp(
            -maximum_yaw, maximum_yaw
        )
        yaw_command = torch.where(active_defenders, yaw_command, 0.0)
        self.defender_dynamics.set_command(
            self.defenders, surge_command, yaw_command, maximum_surge, maximum_yaw
        )

        previous = self.defenders.position
        self.defender_dynamics.step(
            self.defenders,
            active_defenders,
            maximum_surge=maximum_surge,
            maximum_yaw=maximum_yaw,
            horizontal_thruster_limit=torch.full_like(
                maximum_surge, self.defender_dynamics.horizontal_thruster_limit
            ),
        )
        desired = self.defenders.position
        dead = self._static_contact(
            previous,
            desired,
            include_goals=True,
        )
        dead |= self._defender_engagement_boundary_contact(desired)
        dead &= active_defenders
        self.defender_active &= ~dead
        self.defender_avoidance_hazard_id = torch.where(dead, -1, self.defender_avoidance_hazard_id)
        self.defender_avoidance_side = torch.where(dead, 0, self.defender_avoidance_side).to(
            torch.int8
        )
        self.defender_avoidance_surface_distance = torch.where(
            dead, torch.inf, self.defender_avoidance_surface_distance
        )
        return dead

    def _defender_engagement_boundary_contact(
        self, desired: torch.Tensor
    ) -> torch.Tensor:
        safety_zones = self.config.goal_reach["safety_zones"]
        if not bool(safety_zones.get("enabled", False)):
            return torch.zeros(
                desired.shape[:-1], dtype=torch.bool, device=self.device
            )
        dynamic = self.config.goal_reach.get("target_mode", "static") == "dynamic"
        target_active = torch.ones_like(self.done) if dynamic else self.goal_active[:, 0]
        target = self.region_reference_position[:, None] if dynamic else self.goal_centers[:, 0, None, :]
        center_limit = max(
            0.0,
            float(safety_zones["engagement_radius_m"])
            - self.vessel_radius_m,
        )
        return (
            (desired - target).norm(dim=-1) >= center_limit
        ) & target_active[:, None]

    def _circle_collection_contact(
        self,
        previous: torch.Tensor,
        desired: torch.Tensor,
        centers: torch.Tensor,
        radii: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        centers, radii, active, _ = self._active_circle_columns(
            centers, radii, active
        )
        if desired.ndim == 2:
            pass
        elif desired.ndim == 3:
            centers = centers[:, None]
            radii = radii[:, None]
            active = active[:, None]
        else:
            raise ValueError("Actor positions must have shape [B,2] or [B,N,2]")
        segment = desired - previous
        length_squared = segment.square().sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        relative = centers - previous.unsqueeze(-2)
        projection = (relative * segment.unsqueeze(-2)).sum(dim=-1) / length_squared
        projection = projection.clamp(0.0, 1.0)
        closest = previous.unsqueeze(-2) + projection.unsqueeze(-1) * segment.unsqueeze(-2)
        segment_distance_squared = (closest - centers).square().sum(dim=-1)
        segment_contact = segment_distance_squared <= (
            radii + self.vessel_radius_m
        ).square()
        return (segment_contact & active).any(dim=-1)

    def _obstacle_contact(
        self,
        previous: torch.Tensor,
        desired: torch.Tensor,
        *,
        include_goals: bool = False,
    ) -> torch.Tensor:
        centers, radii, active = (
            self._defender_hazard_circles()
            if include_goals
            else (self.obstacle_centers, self.obstacle_radii, self.obstacle_active)
        )
        return self._circle_collection_contact(previous, desired, centers, radii, active)

    def _goal_contact(
        self, previous: torch.Tensor, desired: torch.Tensor
    ) -> torch.Tensor:
        return self._circle_collection_contact(
            previous,
            desired,
            self.goal_centers,
            self.goal_radii,
            self.goal_active,
        )

    def _static_contact(
        self,
        previous: torch.Tensor,
        desired: torch.Tensor,
        *,
        include_goals: bool = False,
    ) -> torch.Tensor:
        return self._obstacle_contact(
            previous, desired, include_goals=include_goals
        )

    def _active_hazard_geometry(
        self, env_index: int
    ) -> tuple[list[tuple[float, float]], list[float]]:
        centers, radii, active = self._defender_hazard_circles()
        indices = torch.nonzero(active[env_index], as_tuple=False).flatten().tolist()
        return (
            [tuple(centers[env_index, index].tolist()) for index in indices],
            [float(radii[env_index, index]) for index in indices],
        )

    def patrol_position_invalid_reason(
        self,
        env_index: int,
        position: tuple[float, float],
        *,
        obstacle_centers: Sequence[tuple[float, float]] | None = None,
        obstacle_radii: Sequence[float] | None = None,
    ) -> str | None:
        specification = self.config.goal_reach["defender"]
        if obstacle_centers is None or obstacle_radii is None:
            resolved_centers, resolved_radii = self._active_hazard_geometry(env_index)
        else:
            resolved_centers = list(obstacle_centers)
            resolved_radii = list(obstacle_radii)
        clearance = (
            self.vessel_radius_m
            + float(specification["avoidance_inner_distance_m"])
            + float(specification["patrol_waypoint_reached_distance_m"])
        )
        reference = tuple(float(value) for value in self.region_reference_position[int(env_index)])
        engagement = float(self.config.goal_reach["safety_zones"]["engagement_radius_m"])
        if math.dist(position, reference) > engagement - clearance + 1.0e-5:
            return "DEF position exceeds the moving engagement boundary"
        if any(
            math.dist(position, center) <= radius + clearance - 1.0e-5
            for center, radius in zip(resolved_centers, resolved_radii, strict=True)
        ):
            return "DEF position violates obstacle clearance"
        return None

    def _apply_time_limit_boundary(
        self,
        timeout: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        bootstrap = bool(
            self.config.goal_reach_trainer["optimizer"].get(
                "bootstrap_on_time_limit", False
            )
        )
        if bootstrap:
            truncated |= timeout
        else:
            terminated |= timeout
