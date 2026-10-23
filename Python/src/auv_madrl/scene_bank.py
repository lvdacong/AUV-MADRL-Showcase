from __future__ import annotations

from dataclasses import asdict, dataclass
from .terrain import TERRAIN_BLOCK_SIZE_M, world_parameters, load_periodic_maps, PERIODIC_MAP_SCHEMA
import json

TERRAIN_GENERATOR = "periodic_lognormal_regions_v1"
SCENE_SCHEMA = "fixed_world_local15_first10_v1"
DENSITY_SAMPLING = "fixed_100m_regions_lognormal_rounded_counts"


@dataclass(frozen=True)
class SceneSpec:
    sense_radius_m: float
    vessel_radius_m: float
    learner_count_range: tuple[int, int]
    defender_count_range: tuple[int, int]
    defender_spawn_radius_m: float
    attacker_spawn_inner_radius_m: float
    attacker_spawn_outer_radius_m: float
    engagement_radius_m: float
    target_count_range: tuple[int, int] = (1, 1)
    obstacle_radius_mean_m: float = 1.8
    obstacle_radius_std_m: float = .7
    obstacle_density_mean_per_1000m2: float = 2.5
    obstacle_density_std_per_1000m2: float = 1.5
    world_map_seed: int = 41001

    def to_payload(self):
        return {k: list(v) if isinstance(v, tuple) else v for k,v in asdict(self).items()}


def scene_spec(config, *, learner_count_range, defender_count_range):
    cfg = config.goal_reach
    safety = cfg["safety_zones"]
    radius = float(config.shared["sensing"]["radius_m"])
    return SceneSpec(radius, float(cfg["vessel_radius_m"]), tuple(learner_count_range), tuple(defender_count_range),
        float(safety["defender_spawn_radius_m"]) if safety["enabled"] else .4*radius,
        float(safety["attacker_spawn_inner_radius_m"]) if safety["enabled"] else 2*radius/3,
        radius-float(cfg["vessel_radius_m"])-1e-5,
        float(safety["engagement_radius_m"]) if safety["enabled"] else 2*radius,
        tuple(cfg.get('target_count_range',(1,1))), **world_parameters(cfg))


def scene_contract(spec, learner_speed_ratio_range, defender_speed_ratio_range,
                   target_speed_ratio_range=(2., 2.)):
    return {"schema": SCENE_SCHEMA,
            "observations": {"obstacles": {
                "schema": "periodic_local15_first10_v1", "capacity": 10, "surface_radius_m": 15.,
                "selection": "stable_cell_id_order", "truncation": "first10",
                "visibility": "own_alive_surface_distance_15m",
                "ordering": "stable_world_key",
                "critic_selection": "same_observer_tensor_and_mask"}},
            "layout": {"generator": TERRAIN_GENERATOR, "spec": spec.to_payload(),
                        "candidate_count": 32, "candidate_batches": 8, "layout_attempts": 8},
            "episode_randomization": {"seed_source": "seed_lane_episode_field_slot_candidate_retry",
                "roles": ["learner", "defender", "target"],
                "learner_speed_ratio_range": list(learner_speed_ratio_range),
                "defender_speed_ratio_range": list(defender_speed_ratio_range),
                "target_speed_ratio_range": list(target_speed_ratio_range)},
            "terrain": {"block_size_m": TERRAIN_BLOCK_SIZE_M, "density_unit": "centers_per_1000m2",
                        "density_sampling": DENSITY_SAMPLING,
                        "density_seed_source": "world_map_seed", "density_region_size_m": 100.,
                        "bank_schema": PERIODIC_MAP_SCHEMA, "bank_hash": load_periodic_maps(parameters=spec.to_payload()).asset_hash,
                        "length_m": 1000., "rotation": False,
                        "identity": ["map_id", "base_obstacle_id", "tile_x", "tile_z"],
                        "cache": "live_sensing_union_plus_halo"},
            "task_boundaries": {"attacker": "center_within_reference_target_sense_radius",
                                "defender": "full_body_within_reference_target_engagement_radius",
                                "target": "none"},
            "circle_overlap": "allowed", "actor_placement": "area_uniform_task_regions"}


def scene_contract_diff(recorded, expected, path='scene_contract'):
    """Pure, complete field diagnostics; missing is distinct from explicit null."""
    if isinstance(recorded, dict) and isinstance(expected, dict):
        differences = []
        for key in sorted(recorded.keys() | expected.keys()):
            field = path + '.' + key
            if key not in recorded or key not in expected:
                differences.append(dict(path=field,
                    recorded=recorded[key] if key in recorded else {'missing': True},
                    expected=expected[key] if key in expected else {'missing': True}))
            else:
                differences.extend(scene_contract_diff(recorded[key], expected[key], field))
        return differences
    return [] if recorded == expected else [dict(path=path, recorded=recorded, expected=expected)]


def reject_scene_contract_differences(differences):
    if differences:
        raise ValueError('GoalReach checkpoint is incompatible with a different scene_contract; '
                         'start a new randomly initialized experiment without old weights: '
                         + json.dumps(differences, ensure_ascii=False, sort_keys=True))
