from __future__ import annotations

from .terrain import WORLD_DEFAULTS, WORLD_BOUNDS, world_parameters

import argparse
import bisect
import copy
import gc
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from functools import lru_cache
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import torch
from tensorboard.backend.event_processing.event_accumulator import SCALARS, EventAccumulator
from torch.utils.tensorboard import SummaryWriter
from .viewer_replay import ViewerPolicyNoise, ViewerEpisodeRecord, physical_digest

from .actor_training import (
    TRAINING_ARCHIVE_NAME,
    TRAINING_DELETE_REQUEST_NAME,
    TRAINING_REDO_DIRECTORY,
    TRAINING_STOP_REQUEST_NAME,
    TRAINING_UNDO_DIRECTORY,
    ActorTrainingLauncher,
    MAX_PARALLEL_TRAINING_RUNS,
    TrainingCapacityError,
    record_training_attempt_failure,
    training_metrics_source,
    role_metric_record,
    _last_jsonl_object,
    actor_training_lock_path,
    archive_actor_training_run,
    checkpoint_training_parameters,
    checkpoint_training_series_signature,
    claim_actor_training_lock,
    require_actor_training_owner,
    require_active_training_checkpoint,
    receive_actor_training_lock,
    is_training_model_timestamp_id,
    model_name_fields,
    read_actor_training_lock,
    release_actor_training_lock,
    training_model_timestamp_id,
    training_run_sort_key,
    training_run_started_at_ns,
    update_actor_training_runtime,
)
from .actor_variants import (
    ROLE_DESCRIPTORS, ROLE_ORDER, policy_roles, train_roles, parameter_group_bindings, UNIFIED_MATCH_ARCHITECTURE_IDS, unified_match_architecture_id,
    ADVERSARIAL_GAME_BASELINE_ARCHITECTURE_ID,
    DEFAULT_ACTOR_VARIANT_IDS,
    MULTI_GOAL_REACH_ARCHITECTURE_ID,
    MULTI_PATROL_VIEWER_ARCHITECTURE_IDS,
    PATROL_PURSUIT_ARCHITECTURE_ID,
    PATROL_VIEWER_ARCHITECTURE_IDS,
    UNIFIED_MATCH_ATTACKER_ARCHITECTURE_IDS,
    UNIFIED_MATCH_DEFENDER_ARCHITECTURE_IDS,
    UNIFIED_MATCH_TARGET_ARCHITECTURE_IDS,
    active_roles,
    actor_variant,
    actor_variant_ids,
    actor_variants_for_kind,
    patrol_defender_count_bounds,
    viewer_actor_variant_ids,
    viewer_actor_variants,
)
from .checkpoint import rng_state, save_checkpoint, load_rng_state
from .config import ProjectConfig, load_project_config
from .controllers import ActorControlMux
from .apf import APFConfig, apf_action
from .goal_reach_env import GoalReachEnv, GoalReachStep
from .live import (
    _encode_actor_input,
    _handler_factory,
    _MemorySnapshot,
)
from .match_metrics import (
    MATCH_END_REASON_CODES,
    MATCH_EVENT_INFO_KEYS,
    METRIC_REGISTRY,
    MatchEndReason,
    MatchOutcome,
    match_registry_payload,
    metric_registry_payload, metric_applicable,
)
from .models import (
    ActorCritic,
    BoundedActionDistribution,
    actor_deployment_state_dict,
)
from .multi_goal_reach_env import (
    PATROL_SPEED_RATIO_BOUNDS,
    MultiGoalReachEnv,
)
from .rng import compose_episode_seed
from .run_evidence import (
    current_timestamp,
    repository_state,
    runtime_information,
    update_json_atomic,
    write_json_atomic,
    create_json_atomic,
)
from .viewer_terrain import ViewerTerrain
from .scene_bank import TERRAIN_BLOCK_SIZE_M, TERRAIN_GENERATOR, scene_spec, scene_contract, scene_contract_diff, reject_scene_contract_differences
from .training import (
    PPO_DEFAULTS,
    GoalReachRuntime, PolicyParameterGroup, update_parameter_group,
    PPO_SAMPLE_CONTRACT, validate_parameter_group_rollouts,
    collect_joint_goal_reach,
    _sample_parameter_groups,
    seed_everything,
    update_goal_reach,
)
from .population import (
    POPULATION_PROTOCOL_VERSION,
    ImmutablePolicyRef,
    MatrixCell,
    PopulationOpponentContext,
    PopulationState,
    StageTracker,
    actor_state_hash,
    validate_evaluation_summary,
    validate_population_evaluation_matches,
    validate_population_protocol_version,
    validate_population_settings,
)

# Current GoalReach-only values. Shared action mapping, 3-DOF dynamics and
# 0.02 s x 25 physics cadence remain in their existing shared implementation.
VESSEL_RADIUS_M = 1.0
SCENE_SURFACE_GAP_M = 2.0
CRAFT_PLACEMENT_CLEARANCE_M = VESSEL_RADIUS_M
CRAFT_CENTER_SEPARATION_M = 2.0 * VESSEL_RADIUS_M + SCENE_SURFACE_GAP_M
SENSE_RADIUS_M = 25.0
LEGACY_SAFETY_ZONE_RADIUS_FRACTIONS = (
    0.39936102470097695,
    0.7311580462862142,
    0.8633935078340117,
)
DEFENDER_SPAWN_RADIUS_FRACTION = 0.32
ENGAGEMENT_RADIUS_FRACTION = 17.0 / SENSE_RADIUS_M
ATTACKER_SPAWN_INNER_RADIUS_FRACTION = ENGAGEMENT_RADIUS_FRACTION
DEFENDER_SPAWN_RADIUS_FRACTION_BOUNDS = (0.26, 0.50)
ENGAGEMENT_RADIUS_FRACTION_BOUNDS = (0.36, 0.90)
ATTACKER_SPAWN_INNER_RADIUS_FRACTION_BOUNDS = (
    0.45,
    1.0 - (VESSEL_RADIUS_M + 0.5 * SCENE_SURFACE_GAP_M) / SENSE_RADIUS_M,
)
SAFETY_ZONE_EDIT_SETTINGS = {
    "defender_spawn": "defender_spawn_radius_fraction",
    "engagement": "engagement_radius_fraction",
    "attacker_spawn_inner": "attacker_spawn_inner_radius_fraction",
}
GOAL_REACH_FIXED_UPDATES = 1600
SCENE_ENTITY_RADIUS_BOUNDS_M = (
    0.02 * SENSE_RADIUS_M,
    0.2 * SENSE_RADIUS_M,
)
VIEWER_CONTROL_MODES = frozenset({"manual", "nn", "rule"})
PPO_UPDATE_MODES = frozenset({"fixed_minibatch", "fixed_optimizer_steps"})
TIME_LIMIT_BOUNDARIES = frozenset({"termination", "truncation"})
MULTI_AGENT_ALGORITHMS = frozenset({"shared_ppo"})
ENTITY_SELF_ATTENTION_LAYER_BOUNDS = (0, 4)
TENSORBOARD_SCALAR_TAGS = frozenset(METRIC_REGISTRY)
TRAINING_DASHBOARD_METRICS = tuple(METRIC_REGISTRY)
LEGACY_TRAINING_DASHBOARD_METRICS = {
    "optimizer/policy_loss": "train/policy_loss",
    "optimizer/value_loss": "train/value_loss",
    "optimizer/entropy": "train/entropy",
    "optimizer/approx_kl": "train/approx_kl",
    "optimizer/explained_variance": "train/explained_variance",
}
POLICY_LOSS_TENSORBOARD_TAGS = ("optimizer/policy_loss", "train/policy_loss")
TRAINING_DASHBOARD_TASK_IDS = frozenset({"goal_reach", "patrol_pursuit"})
TRAINING_UNDO_LIMIT = 10
GOAL_REACH = {
    "profile_id": "shared-actor-scenarios",
    "seed": 42,
    "updates": GOAL_REACH_FIXED_UPDATES,
    "sense_radius_m": SENSE_RADIUS_M,
    "vessel_radius_m": VESSEL_RADIUS_M,
    "obstacle_capacity": 1,
    "obstacle_center_radius_max_m": SENSE_RADIUS_M,
    "maximum_episode_decisions": 150,
    "defender_speed_ratio": (2.0, 2.0),
    "patrol_surge_mps": 2.0,
    "patrol_yaw_rate_max_radps": 0.2,
    "patrol_heading_gain": 1.8,
    "defender_avoidance_outer_distance_m": 26.0,
    "defender_avoidance_inner_distance_m": 10.0,
    "defender_avoidance_release_distance_m": 28.0,
    "defender_avoidance_outward_normal_coefficient": 0.35,
    "defender_peer_repulsion_distance_m": 8.0,
    "defender_peer_repulsion_weight": 1.4,
    "patrol_waypoint_minimum_distance_m": 6.0,
    "patrol_waypoint_preferred_distance_m": 27.0,
    "patrol_waypoint_maximum_distance_m": 55.0,
    "patrol_waypoint_reached_distance_m": 6.0,
    "patrol_waypoint_direction_samples": 16,
    "patrol_waypoint_planner": "gpu_analytic_ray16",
    "defender_rng": "per_defender_gpu_lcg31_masked",
    **PPO_DEFAULTS,
}

FULL_SPEED = 1_000_000.0
INITIAL_CHECKPOINT_NAME = "checkpoint-initial.pt"
TRAINING_CHECKPOINT_INTERVAL_UPDATES = 50
VIEWER_TRAJECTORY_MAX_POINTS = 81
VIEWER_MAX_PACED_PHYSICS_TICKS_PER_LOOP = 8
VIEWER_PHYSICS_BATCH_BUDGET_S = 1.0 / 120.0
VIEWER_PERFORMANCE_WINDOW_SAMPLES = 1024
VIEWER_PERFORMANCE_REFRESH_S = 0.5
VIEWER_RATE_OPTIONS = {
    "time_scale": (1, 5, 10, 20, int(FULL_SPEED)),
    "snapshot_rate_hz": (10, 20, 60, int(FULL_SPEED)),
}


def _viewer_terminal_snapshot_hold_s(snapshot_rate_hz: int) -> float:
    maximum_bounded_rate = max(
        rate for rate in VIEWER_RATE_OPTIONS["snapshot_rate_hz"] if rate != FULL_SPEED
    )
    return 1.0 / min(float(snapshot_rate_hz), float(maximum_bounded_rate))


VIEWER_SNAPSHOT_STATIC_KEYS = frozenset(
    {
        "type",
        "viewer_mode",
        "task_id",
        "arena_count",
        "visible_arena_count",
        "gpu_parallel_arena_count",
        "sense_radius_m",
        "learner_center_spawn_range_m",
        "control_capabilities",
        "metric_registry",
        "match_registry",
        "goal_reach_scale",
        "actor_scenario_identity",
    }
)
VIEWER_SNAPSHOT_STATUS_KEYS = frozenset(
    {
        "actor_training",
    }
)
VIEWER_SNAPSHOT_CACHE_KEYS = VIEWER_SNAPSHOT_STATIC_KEYS
VIEWER_SNAPSHOT_ACTOR_INPUT_KEYS = frozenset({"actor_inputs"})
VIEWER_ACTOR_INPUTS_RATE_HZ = 20.0
VIEWER_MAIN_VIEW_FRAMES = ("union", "union_follow", "local", "follow")
VIEWER_MAIN_VIEW_ZOOM_STEP_BOUNDS = (-16, 12)
VIEWER_LEARNER_COUNT_BOUNDS = (1, 10)
VIEWER_DEFENDER_COUNT_BOUNDS = (1, 10)
VIEWER_OBSTACLE_DENSITY_BOUNDS_PER_1000M2 = (0.0, 10.0)
VIEWER_OBSTACLE_RADIUS_BOUNDS_M = SCENE_ENTITY_RADIUS_BOUNDS_M
_TRAINING_INVENTORY_WORKER_LAUNCHERS: dict[str, ActorTrainingLauncher] = {}


class _RollingPerformanceWindow:
    """Exact rolling percentiles without periodic full-window sorting."""

    def __init__(self, maxlen: int):
        self.maxlen = int(maxlen)
        self._values: deque[float] = deque()
        self._ordered: list[float] = []
        self._total_count = 0
        self._total_sum = 0.0
        self._window_sum = 0.0
        self._lock = threading.Lock()

    def append(self, value: float) -> None:
        with self._lock:
            if len(self._values) >= self.maxlen:
                expired = self._values.popleft()
                self._window_sum -= expired
                index = bisect.bisect_left(self._ordered, expired)
                if index < len(self._ordered):
                    self._ordered.pop(index)
            self._values.append(value)
            bisect.insort(self._ordered, value)
            self._window_sum += value
            self._total_sum += value
            self._total_count += 1

    def clear(self) -> None:
        with self._lock:
            self._values.clear()
            self._ordered.clear()
            self._total_count = 0
            self._total_sum = 0.0
            self._window_sum = 0.0

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)

    def summary(self) -> dict[str, float | None]:
        with self._lock:
            size = len(self._ordered)
            if not size:
                return {"p95": None, "p99": None, "max": None}
            return {
                "count": self._total_count,
                "total_ms": self._total_sum,
                "window_count": size,
                "window_total_ms": self._window_sum,
                "p95": self._ordered[math.ceil(0.95 * size) - 1],
                "p99": self._ordered[math.ceil(0.99 * size) - 1],
                "max": self._ordered[-1],
            }

    def integer_counts(self, maximum: int) -> dict[str, int]:
        with self._lock:
            return {
                str(value): sum(sample == value for sample in self._values)
                for value in range(1, maximum + 1)
            }
VIEWER_ENVIRONMENT_SETTING_KEYS = frozenset(
    {
        "target_mode", "target_count_range", "target_speed_ratio_range",
        "terminate_on_any_casualty", "reward_objective",
        "include_remaining_time_observation", "terminate_on_defender_elimination", "viewer_apply_sampling_limit",
        "safety_zones_enabled",
        "defender_spawn_radius_fraction",
        "engagement_radius_fraction",
        "attacker_spawn_inner_radius_fraction",
        "learner_count_range",
        "defender_count_range",
        "learner_speed_ratio_range",
        "entity_streams",
        "entity_self_attention_layers",
        *WORLD_DEFAULTS,
        "defender_speed_ratio_range",
    }
)


TARGET_SETTING_DEFAULTS = {"target_mode": "dynamic", "target_count_range": [1, 1],
                           "target_speed_ratio_range": [2.0, 2.0]}


def _viewer_settings_defaults() -> dict[str, object]:
    return {
        "target_mode": "dynamic", "target_count_range": [1,1], "target_speed_ratio_range": [2.,2.],
        "target_control_mode": "nn", "match_rules_version": 7, "sense_radius_m": SENSE_RADIUS_M,
        "terminate_on_any_casualty": False, "reward_objective": "outcome",
        "include_remaining_time_observation": True, "terminate_on_defender_elimination": False, "viewer_apply_sampling_limit": True,
        "safety_zones_enabled": True, "defender_spawn_radius_fraction": DEFENDER_SPAWN_RADIUS_FRACTION,
        "engagement_radius_fraction": ENGAGEMENT_RADIUS_FRACTION,
        "attacker_spawn_inner_radius_fraction": ATTACKER_SPAWN_INNER_RADIUS_FRACTION,
        "active_actor_variant_id": UNIFIED_MATCH_ARCHITECTURE_IDS[("separate",0)],
        "selected_actor_kind": "learner",
        "training_settings": {"maximum_episode_decisions":150,"parallel_environments":1024,
            "rollout_decisions":32,"minibatch_agent_frames":4096,"updates":0,
            "optimizer_steps_per_update":40,
            "gamma":1.,"gae_lambda":.99,"entropy_coefficient":.01,"time_limit_boundary":"termination"},
        "training_side_modes": {r:"train" for r in ROLE_ORDER},
        "joint_training_mode":"joint", "alternating_updates":1,"multi_agent_algorithm":"shared_ppo",
        "policy_sharing":"team",
        "learner_count_range":[2,2],"defender_count_range":[2,2],
        "learner_speed_ratio_range":[2.5,2.5],"defender_speed_ratio_range":[2.5,2.5],
        "include_learner_speed_ratio_observation":True,"entity_streams":"separate","entity_self_attention_layers":0,
        **WORLD_DEFAULTS,
        "learner_control_mode":"nn","defender_control_mode":"nn",
        "learner_rule": APFConfig().to_dict(),
        "deployment_model_ids":{r:"" for r in ROLE_ORDER},"time_scale":5.,"snapshot_rate_hz":60,
        "main_view_frame":"union","main_view_zoom_step":0,
        "policy_sampling":"stochastic","evaluation_scene_preset":None,
        "training_group_labels":{},
        "auto_deploy_training_run_group_id": None,
    }


def _validated_viewer_settings(value: dict[str, object], *, historical=False) -> dict[str, object]:
    if not historical and "role_loss_weighting" in value:
        raise ValueError("Obsolete setting role_loss_weighting; remove it and use random cold initialization")
    if 'obstacle_count_range' in value:
        raise ValueError('Legacy obstacle_count_range is unsupported; use obstacle_density_range_per_1000m2')
    if not historical and {'world_map_mode','obstacle_density_range_per_1000m2','obstacle_radius_range_m'} & set(value):
        raise ValueError('Legacy map controls are unsupported; use the four world parameters')
    defaults = _viewer_settings_defaults()
    unknown_legacy = {"arena_radius_m", "world_fixed", "selected_actor_centered_heading_up"} & set(value)
    if unknown_legacy:
        raise ValueError("Legacy fixed-world Viewer settings are not supported")
    result = {k:copy.deepcopy(value.get(k,v)) for k,v in defaults.items()}
    result.update(world_parameters(result))
    follow_group = result['auto_deploy_training_run_group_id']
    if follow_group is not None and (not isinstance(follow_group, str) or not follow_group.strip()):
        raise ValueError('Auto deploy requires a training group ID or null')
    labels = result['training_group_labels']
    if not isinstance(labels, dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in labels.items()):
        raise ValueError('training_group_labels must map run IDs to text labels')
    for key, allowed in (("target_mode",{"static","dynamic"}),("entity_streams",{"separate","unified"}),
        ("policy_sharing",{"role","team","all"}),("reward_objective",{"outcome","outcome_time","protection_time"}),
        ("joint_training_mode",{"joint","alternating"}),("selected_actor_kind",set(ROLE_ORDER)),
        ("policy_sampling",{"deterministic","stochastic"}),("main_view_frame",VIEWER_MAIN_VIEW_FRAMES)):
        if result[key] not in allowed:raise ValueError(f"Unsupported {key}: {result[key]}")
    zoom_step = result["main_view_zoom_step"]
    if (
        isinstance(zoom_step, bool)
        or not isinstance(zoom_step, int)
        or not VIEWER_MAIN_VIEW_ZOOM_STEP_BOUNDS[0]
        <= zoom_step
        <= VIEWER_MAIN_VIEW_ZOOM_STEP_BOUNDS[1]
    ):
        raise ValueError("main_view_zoom_step must be an integer inside [-16, 12]")
    for key in ("terminate_on_any_casualty","safety_zones_enabled","include_remaining_time_observation","terminate_on_defender_elimination","viewer_apply_sampling_limit"):
        if not isinstance(result[key],bool):raise ValueError(f"{key} must be boolean")
    ranges = {**{r+"_count_range":((1,10),int) for r in ROLE_ORDER},
        **{r+"_speed_ratio_range":(PATROL_SPEED_RATIO_BOUNDS,float) for r in ROLE_ORDER}}
    for key,(bounds,cast) in ranges.items():
        values=result[key]
        if not isinstance(values,(list,tuple)) or len(values)!=2 or any(
                isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x)
                or (cast is int and x!=int(x)) for x in values):
            raise ValueError(f'{key} requires two finite {"integers" if cast is int else "numbers"}')
        low,high=map(cast,result[key])
        if not all(math.isfinite(x) for x in (low,high)) or not bounds[0]<=low<=high<=bounds[1]:
            raise ValueError(f"{key} must stay inside {bounds}")
        result[key]=[low,high]
    for key,bounds in (("defender_spawn_radius_fraction",DEFENDER_SPAWN_RADIUS_FRACTION_BOUNDS),
        ("engagement_radius_fraction",ENGAGEMENT_RADIUS_FRACTION_BOUNDS),
        ("attacker_spawn_inner_radius_fraction",ATTACKER_SPAWN_INNER_RADIUS_FRACTION_BOUNDS)):
        result[key]=float(result[key])
        if not bounds[0]<=result[key]<=bounds[1]:raise ValueError(f"{key} out of range")
    if not result["defender_spawn_radius_fraction"]+VESSEL_RADIUS_M/SENSE_RADIUS_M<result["engagement_radius_fraction"]<=result["attacker_spawn_inner_radius_fraction"]:
        raise ValueError("Require DEF spawn plus hull clearance < engagement <= ATK spawn")
    attacker_spawn_inner_radius_m = (
        result["attacker_spawn_inner_radius_fraction"] * SENSE_RADIUS_M
    )
    attacker_spawn_outer_radius_m = SENSE_RADIUS_M - VESSEL_RADIUS_M - 1e-5
    if not attacker_spawn_inner_radius_m < attacker_spawn_outer_radius_m:
        raise ValueError("ATK spawn inner radius must be below its physical outer radius")
    schedule=result['training_settings']
    if not isinstance(schedule,dict):raise ValueError("training_settings must be an object")
    if 'goal_reach' in schedule:
        raise ValueError('Legacy nested GoalReach schedules are not supported')
    result['training_settings']={k:copy.deepcopy(schedule.get(k,v)) for k,v in defaults['training_settings'].items()}
    # Missing in historical settings means the formerly fixed bonus, never Off.
    result['training_settings']['entropy_coefficient'] = _entropy_coefficient(schedule)
    for k,v in defaults['training_settings'].items():
        if isinstance(v,int):
            schedule_value=result['training_settings'][k]
            if k == 'maximum_episode_decisions' and (type(schedule_value) is not int or schedule_value < 1):
                raise ValueError('maximum_episode_decisions must be a positive integer')
            if k == 'updates' and (type(schedule_value) is not int or schedule_value < 0):
                raise ValueError('updates must be a nonnegative integer (0 means unlimited)')
            result['training_settings'][k]=int(schedule_value)
            if k != 'updates' and result['training_settings'][k]<1:raise ValueError(f"{k} must be positive")
    steps=result['training_settings']['optimizer_steps_per_update']
    raw_steps=schedule.get('optimizer_steps_per_update',40)
    if type(raw_steps) is not int or steps < 5 or steps % 5:
        raise ValueError('optimizer_steps_per_update must be a positive multiple of 5 epochs')
    for key in ('gamma','gae_lambda'):
        if type(result['training_settings'][key]) not in (int,float):raise ValueError(f'{key} must be numeric')
        result['training_settings'][key]=float(result['training_settings'][key])
        if not 0<=result['training_settings'][key]<=1:raise ValueError(f"{key} must be inside [0,1]")
    boundary = result['training_settings']['time_limit_boundary']
    if boundary not in ('termination', 'truncation'):
        raise ValueError('Time limit must be termination or truncation')
    if boundary == 'truncation' and result['reward_objective'] != 'outcome':
        raise ValueError('Truncation requires Win/Loss reward; time-based rewards require a task deadline')
    if result['reward_objective']=='protection_time' and result['training_settings']['gamma']!=1:raise ValueError('Protection time requires gamma=1')
    if result['reward_objective']=='outcome_time' and result['training_settings']['gamma']!=1:raise ValueError('Outcome time requires gamma=1; discounted gamma duplicates time preference')
    result['entity_self_attention_layers']=int(result['entity_self_attention_layers'])
    if not 0<=result['entity_self_attention_layers']<=4:raise ValueError('Self-attention layers must be inside [0,4]')
    result['alternating_updates']=int(result['alternating_updates'])
    if result['alternating_updates']<1:raise ValueError('Alternating K must be positive')
    result['training_side_modes']={r:str(result['training_side_modes'].get(r,'train')) for r in ROLE_ORDER}
    if any(v not in {'train','fixed'} for v in result['training_side_modes'].values()):raise ValueError('Role mode must be Train or Fixed')
    if result['joint_training_mode']=='alternating' and len(set(parameter_group_bindings(result).values()))<2:
        raise ValueError('Alternating requires at least two trainable parameter groups')
    for role in ROLE_ORDER:
        allowed = VIEWER_CONTROL_MODES if role == 'learner' else {'manual', 'nn'}
        if result[role+'_control_mode'] not in allowed:
            raise ValueError(f'{role} control mode must be {sorted(allowed)}')
    result['learner_rule'] = APFConfig.from_dict(result['learner_rule']).to_dict()
    if result['learner_control_mode'] == 'rule' and result['training_side_modes']['learner'] != 'fixed':
        raise ValueError('Rule (APF) ATK must be Fixed; rule actions cannot train PPO')
    result['deployment_model_ids']={r:str(result['deployment_model_ids'].get(r) or '') for r in ROLE_ORDER}
    for key in ('time_scale','snapshot_rate_hz'):
        if result[key] not in VIEWER_RATE_OPTIONS[key]:raise ValueError(f'Unsupported {key}')
    result['active_actor_variant_id']=unified_match_architecture_id(result['entity_streams'],result['entity_self_attention_layers'],result['include_remaining_time_observation'])
    result['match_rules_version']=7
    result['sense_radius_m']=SENSE_RADIUS_M
    population = value.get('population')
    if population is not None:
        if boundary != 'termination' and isinstance(population, dict) and population.get('condition') is not None:
            raise ValueError('Population experiments require their fixed terminal horizon')
        if not isinstance(population, dict):
            raise ValueError('population must be an object')
        result['population'] = validate_population_settings(population)
        if result['policy_sharing'] != 'team':
            raise ValueError('Population response training requires Team policy sharing')
        if result['joint_training_mode'] != 'alternating':
            raise ValueError('Population response training controls its own alternating phases')
        if result['training_settings']['gae_lambda'] != .99:
            raise ValueError('Population response training requires GAE lambda 0.99')
        condition = result['population']['condition']
        if condition is not None and (not result['include_remaining_time_observation'] or result['terminate_on_defender_elimination']):
            raise ValueError('New observation/terminal rules require condition:null and an independent batch_identity')
        expected = {
            'A': ('outcome', 1., 0.),
            'B': ('outcome', 1., .01),
            'C': ('outcome', math.exp(math.log(.75) / 149), 0.),
            'D': ('outcome_time', 1., 0.),
        }.get(condition)
        actual = (result['reward_objective'], result['training_settings']['gamma'],
                  result['training_settings']['entropy_coefficient'])
        if condition is not None and actual != expected:
            raise ValueError(f'Population condition {condition} does not match its fixed objective/gamma/entropy')
    return result


def _migrated_live_viewer_settings_candidate(
    value: dict[str, object],
) -> dict[str, object]:
    """Migrate only the exact legacy LIVE radius triplet without mutating input."""
    result = copy.deepcopy(value)
    zone_keys = (
        "defender_spawn_radius_fraction",
        "engagement_radius_fraction",
        "attacker_spawn_inner_radius_fraction",
    )
    supplied = tuple(value.get(key) for key in zone_keys)
    if (
        all(
            type(item) in (int, float) and math.isfinite(item)
            for item in supplied
        )
        and supplied == LEGACY_SAFETY_ZONE_RADIUS_FRACTIONS
    ):
        for key, migrated in zip(
            zone_keys,
            (
                DEFENDER_SPAWN_RADIUS_FRACTION,
                ENGAGEMENT_RADIUS_FRACTION,
                ATTACKER_SPAWN_INNER_RADIUS_FRACTION,
            ),
        ):
            result[key] = migrated
    return result


def _store_viewer_settings(path: Path, settings: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _load_viewer_settings(config):
    path=config.repository_root/'Artifacts'/'GoalReach'/'viewer-settings.json'
    value=json.loads(path.read_text(encoding='utf-8')) if path.is_file() else _viewer_settings_defaults()
    if 'role_loss_weighting' in value:
        value.pop('role_loss_weighting')
        _store_viewer_settings(path,value)
    settings=_validated_viewer_settings(value)
    return path,settings


def _apply_persisted_randomization(
    config: ProjectConfig,
    settings: dict[str, object],
    architecture_id: str,
) -> None:
    config.goal_reach.update({key: copy.deepcopy(settings.get(key, default))
                              for key, default in TARGET_SETTING_DEFAULTS.items()})
    config.goal_reach.update(world_parameters(settings))
    config.goal_reach.pop("world_map_mode", None)
    obstacle = config.goal_reach["reset"]["random_obstacles"]
    obstacle.pop("density_per_1000m2", None)
    obstacle.pop("radius_m", None)
    obstacle["capacity"] = 1
    defender = config.goal_reach["defender"]
    defender["speed_ratio"] = list(settings["defender_speed_ratio_range"])
    safety_zones = config.goal_reach["safety_zones"]
    sense_radius_m = float(config.shared["sensing"]["radius_m"])
    safety_zones.update(
        {
            "enabled": bool(settings["safety_zones_enabled"]),
            "defender_spawn_radius_fraction": float(
                settings["defender_spawn_radius_fraction"]
            ),
            "engagement_radius_fraction": float(
                settings["engagement_radius_fraction"]
            ),
            "attacker_spawn_inner_radius_fraction": float(
                settings["attacker_spawn_inner_radius_fraction"]
            ),
            "defender_spawn_radius_m": sense_radius_m
            * float(settings["defender_spawn_radius_fraction"]),
            "engagement_radius_m": sense_radius_m
            * float(settings["engagement_radius_fraction"]),
            "attacker_spawn_inner_radius_m": sense_radius_m
            * float(settings["attacker_spawn_inner_radius_fraction"]),
        }
    )


def _training_budget_exhausted(budget: int, update: int) -> bool:
    """Zero denotes unlimited updates; Stop still saves at a consistent boundary."""
    return budget > 0 and update >= budget


def _training_schedule(
    settings: dict[str, object], architecture_id: str
) -> dict[str, object]:
    schedules = settings["training_settings"]
    schedule = {
        key: int(schedules[key])
        for key in (
            "maximum_episode_decisions",
            "parallel_environments",
            "rollout_decisions",
            "minibatch_agent_frames",
            "optimizer_steps_per_update",
            "updates",
        )
    }
    schedule.update(
        {
            "gamma": float(schedules["gamma"]),
            "gae_lambda": float(schedules["gae_lambda"]),
            "entropy_coefficient": _entropy_coefficient(schedules),
            "time_limit_boundary": schedules["time_limit_boundary"],
            "bootstrap_on_time_limit": schedules["time_limit_boundary"] == "truncation",
        }
    )
    return schedule


def _apply_training_schedule(
    settings: dict[str, object], architecture_id: str
) -> dict[str, object]:
    schedule = _training_schedule(settings, architecture_id)
    GOAL_REACH.update(schedule)
    return schedule


def _apply_variant_trainer_overrides(architecture_id: str) -> None:
    overrides = actor_variant(architecture_id).trainer_override_map
    unknown = sorted(set(overrides) - set(GOAL_REACH))
    if unknown:
        raise ValueError("Unknown GoalReach trainer overrides: " + ", ".join(unknown))
    GOAL_REACH.update(overrides)


def _training_parallel_environments(
    settings: dict[str, object], architecture_id: str
) -> int:
    return int(_training_schedule(settings, architecture_id)["parallel_environments"])


def _multi_agent_algorithm(
    settings: dict[str, object], architecture_id: str
) -> str:
    return "shared_ppo"


def _critic_mode(settings: dict[str, object], architecture_id: str) -> str:
    return (
        "centralized_team"
        if _multi_agent_algorithm(settings, architecture_id) == "mappo"
        else "local"
    )


def _entropy_coefficient(values):
    value = values.get('entropy_coefficient', .01)
    if type(value) not in (int, float) or value not in (0., .01):
        raise ValueError('entropy_coefficient must be numeric 0 or 0.01')
    return float(value)


def _training_rule(settings, role="learner"):
    """Resolve only an explicitly selected rule; historical omissions remain NN."""
    if settings.get(role + "_control_mode", "nn") != "rule":
        return None
    if role != "learner":
        raise ValueError("Only ATK supports Rule (APF)")
    from .apf import APFConfig
    return APFConfig.from_dict(settings.get("learner_rule")).to_dict()


def _validate_training_control_sources(settings):
    for role in policy_roles(settings):
        mode = settings.get(role + "_control_mode", "nn")
        if mode == "manual":
            raise ValueError("Manual control is Viewer-only and cannot produce PPO training data")
        rule = _training_rule(settings, role)
        if rule is not None and role in train_roles(settings):
            raise ValueError("Rule (APF) ATK must be Fixed; it has no trainable model")
        if rule is not None and "population" in settings:
            raise ValueError("Population training does not support APF sources")


def _initial_training_role_bindings(settings, model_ids=None, source_metadata=None):
    bindings = {}
    for role in ROLE_ORDER:
        rule = _training_rule(settings, role)
        if rule is not None:
            bindings[role] = dict(mode="rule", parameter_group=None, model_id=None,
                                  source=None, source_update=None, rule=rule)
        else:
            bindings[role] = dict(mode="train" if role in train_roles(settings) else "fixed" if role in policy_roles(settings) else "static",
                parameter_group=parameter_group_bindings(settings).get(role),
                model_id=(model_ids or {}).get(role),
                source_update=((source_metadata or {}).get(role) or {}).get("source_update"))
    return bindings


def _normalized_training_identity(identity):
    result = copy.deepcopy(identity)
    result['entropy_coefficient'] = _entropy_coefficient(identity)
    if result.get('match_rules_version') == 6:
        if result.get('time_limit_boundary', 'termination') == 'truncation':
            raise ValueError('v6 sampling countdown is incompatible with v7 task time')
        result['match_rules_version'] = 7
        result.setdefault('include_remaining_time_observation', True)
        result.setdefault('terminate_on_defender_elimination', False)
    return result


def _training_settings_identity(settings, architecture_id):
    normalized=_validated_viewer_settings(settings, historical=True)
    fields=('target_mode','target_count_range','target_speed_ratio_range','match_rules_version','sense_radius_m',
        'terminate_on_any_casualty','terminate_on_defender_elimination','include_remaining_time_observation','reward_objective','safety_zones_enabled','defender_spawn_radius_fraction',
        'engagement_radius_fraction','attacker_spawn_inner_radius_fraction','training_side_modes','policy_sharing',
        'learner_count_range','defender_count_range','learner_speed_ratio_range','defender_speed_ratio_range',
        *WORLD_DEFAULTS,'entity_streams','entity_self_attention_layers','joint_training_mode','alternating_updates')
    identity = {**{k:copy.deepcopy(normalized[k]) for k in fields},
        **{k:v for k,v in _training_schedule(normalized,architecture_id).items() if k not in {'updates','optimizer_steps_per_update'}},
        'architecture_id':architecture_id,'task_id':'goal_reach','scene_generator':TERRAIN_GENERATOR,
        'action_distribution':actor_variant(architecture_id).action_distribution,
        'ppo_update':{'sample_contract':PPO_SAMPLE_CONTRACT,'mode':'fixed_optimizer_steps','optimizer_steps_per_update':normalized['training_settings']['optimizer_steps_per_update'],'optimization_epochs':5},
        'credit':'actor_alive_policy_mask_and_initial_member_value_mask'}
    rule = _training_rule(normalized)
    if rule is not None:
        identity['policy_sources'] = {'learner': {'mode': 'rule', 'rule': rule}}
    if 'population' in normalized:
        identity['population'] = copy.deepcopy(normalized['population'])
    return identity


def _checkpoint_viewer_settings(payload, architecture_id):
    _validate_group_checkpoint(payload)
    return _validated_viewer_settings(payload['launch_settings'], historical=True)


def _validate_collection_change(before, after, architecture):
    """An explicit collector change keeps learning/game contracts and work per U."""
    old = _training_settings_identity(before, architecture)
    new = _training_settings_identity(after, architecture)
    if 'population' in old or 'population' in new:
        raise ValueError('Collection reconfiguration does not support population training')
    fields = ('parallel_environments', 'rollout_decisions')
    old_shape = tuple(old.pop(k) for k in fields)
    new_shape = tuple(new.pop(k) for k in fields)
    if (old != new or old_shape == new_shape
            or math.prod(old_shape) != math.prod(new_shape)
            or before['training_settings']['updates'] != after['training_settings']['updates']):
        raise ValueError('Collection reconfiguration may only change parallel environments/rollout with equal work per update')


def _collection_history(metadata):
    history = metadata.get('collection_history', [])
    if not isinstance(history, list):
        raise ValueError('Invalid collection history')
    if not history:return history
    previous_update = -1
    previous_settings = None
    architecture = metadata['architecture_id']
    for index, item in enumerate(history, 1):
        if not isinstance(item, dict):raise ValueError('Invalid collection history entry')
        update = _nonnegative_checkpoint_integer(item.get('after_update'), 'collection.after_update')
        reset_episode = _nonnegative_checkpoint_integer(item.get('reset_episode_index'), 'collection.reset_episode_index')
        if reset_episode < 1:raise ValueError('Collection reset must use a new episode epoch')
        if update <= previous_update:
            raise ValueError('Collection history must have strictly increasing update boundaries')
        expected_name = f'checkpoint-collection-{update:06d}-{index:03d}.pt'
        if item.get('boundary_checkpoint') != expected_name:
            raise ValueError('Invalid collection boundary checkpoint name')
        source = item.get('source_checkpoint', '')
        if source != (INITIAL_CHECKPOINT_NAME if update == 0 else f'checkpoint-{update:06d}.pt'):
            raise ValueError('Collection source must be the completed numbered update')
        if not isinstance(item.get('source_sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', item['source_sha256']):
            raise ValueError('Invalid collection source digest')
        before, after = item.get('from_settings'), item.get('to_settings')
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ValueError('Collection history is missing its settings')
        _validate_collection_change(before, after, architecture)
        if previous_settings is not None and _training_settings_identity(before, architecture) != previous_settings:
            raise ValueError('Collection history settings chain is broken')
        previous_settings = _training_settings_identity(after, architecture)
        previous_update = update
    if history and (previous_settings != _training_settings_identity(metadata['launch_settings'], architecture)
                    or metadata.get('global_update', -1) < previous_update):
        raise ValueError('Collection history does not match the active phase')
    return history


def _release_phase(manifest, filename, update):
    history = _collection_history(manifest)
    settings = history[0]['from_settings'] if history else manifest['launch_settings']
    phase = 0
    for index, item in enumerate(history, 1):
        boundary = filename == item['boundary_checkpoint']
        if boundary and update != item['after_update']:
            raise ValueError('Collection boundary update mismatch')
        if update > item['after_update'] or boundary:
            settings, phase = item['to_settings'], index
        else:break
    if filename.startswith('checkpoint-collection-') and not any(
            item['boundary_checkpoint'] == filename and item['after_update'] == update for item in history):
        raise ValueError('Unknown collection boundary checkpoint')
    return settings, history[:phase]


def _manifest_checkpoint_path(output, manifest):
    update = _nonnegative_checkpoint_integer(manifest.get('checkpoint_update'), 'manifest.checkpoint_update')
    numbered = INITIAL_CHECKPOINT_NAME if update == 0 else f'checkpoint-{update:06d}.pt'
    name = manifest.get('checkpoint_file', numbered)
    allowed = {numbered, *(item['boundary_checkpoint'] for item in _collection_history(manifest)
                          if item['after_update'] == update)}
    if not isinstance(name, str) or name not in allowed:
        raise ValueError('Invalid manifest checkpoint_file')
    return Path(output) / name


def _resolved_defender_count_range(
    settings: dict[str, object], architecture_id: str
) -> tuple[int, int]:
    variant = actor_variant(architecture_id)
    configured = settings.get("defender_count_range")
    if configured is None:
        fixed = int(settings.get("defender_count", variant.defender_count))
        defender_count_range = (fixed, fixed)
    else:
        defender_count_range = tuple(int(value) for value in configured)
        if len(defender_count_range) != 2:
            raise ValueError("DEF count range requires a minimum and maximum")
    if not variant.unified_match_contract and variant.actor_kind != "defender":
        return defender_count_range
    minimum, maximum = patrol_defender_count_bounds(architecture_id)
    defender_count_min, defender_count_max = defender_count_range
    if not minimum <= defender_count_min <= defender_count_max <= maximum:
        raise ValueError(
            f"The selected DEF architecture requires a count in [{minimum},{maximum}]"
        )
    if (
        architecture_id in MULTI_PATROL_VIEWER_ARCHITECTURE_IDS.values()
        and defender_count_max == 1
    ):
        raise ValueError("A multi-DEF architecture requires a maximum count above 1")
    return defender_count_min, defender_count_max


def _resolved_defender_count(
    settings: dict[str, object], architecture_id: str
) -> int:
    return _resolved_defender_count_range(settings, architecture_id)[1]


def _build_config() -> ProjectConfig:
    base = load_project_config()
    shared = {
        key: dict(value) if isinstance(value, dict) else value for key, value in base.shared.items()
    }
    sense_radius_m = float(GOAL_REACH["sense_radius_m"])
    maximum_decisions = int(GOAL_REACH["maximum_episode_decisions"])
    ticks_per_decision = int(shared["clock"]["physics_ticks_per_decision"])
    maximum_physics_ticks = maximum_decisions * ticks_per_decision
    shared.pop("arena", None)
    shared["sensing"] = {"radius_m": sense_radius_m}
    shared["clock"].update(
        {
            "maximum_physics_ticks": maximum_physics_ticks,
            "maximum_decisions": maximum_decisions,
            "episode_duration_s": maximum_physics_ticks * float(shared["clock"]["physics_step_s"]),
        }
    )

    goal_reach = {
        "vessel_radius_m": GOAL_REACH["vessel_radius_m"],
        "safety_zones": {
            "enabled": True,
            "defender_spawn_radius_fraction": DEFENDER_SPAWN_RADIUS_FRACTION,
            "engagement_radius_fraction": ENGAGEMENT_RADIUS_FRACTION,
            "attacker_spawn_inner_radius_fraction": (
                ATTACKER_SPAWN_INNER_RADIUS_FRACTION
            ),
            "defender_spawn_radius_m": (
                sense_radius_m * DEFENDER_SPAWN_RADIUS_FRACTION
            ),
            "engagement_radius_m": (
                sense_radius_m * ENGAGEMENT_RADIUS_FRACTION
            ),
            "attacker_spawn_inner_radius_m": (
                sense_radius_m * ATTACKER_SPAWN_INNER_RADIUS_FRACTION
            ),
        },
        "clock": {
            "physics_ticks_per_decision": ticks_per_decision,
            "maximum_physics_ticks": maximum_physics_ticks,
        },
        "defender": {
            "speed_ratio": list(GOAL_REACH["defender_speed_ratio"]),
            "patrol_surge_mps": GOAL_REACH["patrol_surge_mps"],
            "patrol_yaw_rate_max_radps": GOAL_REACH["patrol_yaw_rate_max_radps"],
            "patrol_heading_gain": GOAL_REACH["patrol_heading_gain"],
            "avoidance_outer_distance_m": GOAL_REACH["defender_avoidance_outer_distance_m"],
            "avoidance_inner_distance_m": GOAL_REACH["defender_avoidance_inner_distance_m"],
            "avoidance_release_distance_m": GOAL_REACH["defender_avoidance_release_distance_m"],
            "avoidance_outward_normal_coefficient": GOAL_REACH[
                "defender_avoidance_outward_normal_coefficient"
            ],
            "peer_repulsion_distance_m": GOAL_REACH["defender_peer_repulsion_distance_m"],
            "peer_repulsion_weight": GOAL_REACH["defender_peer_repulsion_weight"],
            "patrol_waypoint_minimum_distance_m": GOAL_REACH["patrol_waypoint_minimum_distance_m"],
            "patrol_waypoint_preferred_distance_m": GOAL_REACH[
                "patrol_waypoint_preferred_distance_m"
            ],
            "patrol_waypoint_maximum_distance_m": GOAL_REACH["patrol_waypoint_maximum_distance_m"],
            "patrol_waypoint_reached_distance_m": GOAL_REACH["patrol_waypoint_reached_distance_m"],
            "patrol_waypoint_direction_samples": GOAL_REACH["patrol_waypoint_direction_samples"],
        },
        "reset": {
            "random_obstacles": {
                "capacity": GOAL_REACH["obstacle_capacity"],
                "center_radius_max_m": GOAL_REACH["obstacle_center_radius_max_m"],
            },
        },
    }
    world = {
        "world_id": "PythonGoalReach",
        "arena": {
            "radius_m": sense_radius_m,
            "grid_cell_size_m": 2.0 * sense_radius_m / 200.0,
            "shore_band_m": 0.0,
        },
        "obstacles": {"islands": []},
    }

    trainer = {
        "rollout": {
            "minibatch_agent_frames_initial": GOAL_REACH["minibatch_agent_frames"],
            "optimization_epochs": GOAL_REACH["optimization_epochs"],
            "update_mode": GOAL_REACH["ppo_update_mode"],
            "optimizer_steps_per_update": GOAL_REACH["optimizer_steps_per_update"],
        },
        "optimizer": {
            "learning_rate": GOAL_REACH["learning_rate"],
            "gamma": GOAL_REACH["gamma"],
            "gae_lambda": GOAL_REACH["gae_lambda"],
            "bootstrap_on_time_limit": GOAL_REACH["bootstrap_on_time_limit"],
            "clip_epsilon": GOAL_REACH["clip_epsilon"],
            "value_coefficient": GOAL_REACH["value_coefficient"],
            "entropy_coefficient": GOAL_REACH["entropy_coefficient"],
            "maximum_gradient_norm": GOAL_REACH["maximum_gradient_norm"],
        },
    }
    return ProjectConfig(
        repository_root=base.repository_root,
        shared=shared,
        goal_reach=goal_reach,
        world=world,
        goal_reach_trainer=trainer,
    )


def _uses_multi_learner_environment(architecture_id: str) -> bool:
    return (
        architecture_id == MULTI_GOAL_REACH_ARCHITECTURE_ID
        or actor_variant(architecture_id).actor_kind == "defender"
        or actor_variant(architecture_id).unified_match_contract
    )


def _selected_scenario_architecture_id(architecture_id: str) -> str:
    if architecture_id == MULTI_GOAL_REACH_ARCHITECTURE_ID:
        return ADVERSARIAL_GAME_BASELINE_ARCHITECTURE_ID
    if (
        architecture_id in PATROL_VIEWER_ARCHITECTURE_IDS.values()
        or architecture_id in MULTI_PATROL_VIEWER_ARCHITECTURE_IDS.values()
    ):
        return PATROL_PURSUIT_ARCHITECTURE_ID
    return architecture_id


def _architecture_uses_learner_speed_observation(architecture_id: str) -> bool:
    variant = actor_variant(architecture_id)
    return (
        variant.unified_match_contract
        or (
            variant.actor_kind == "defender"
            and "learner_speed_ratio_normalized"
            in variant.entity_features_for("enemies")
        )
    )


def _architecture_entity_streams(architecture_id: str) -> str:
    variant = actor_variant(architecture_id)
    return variant.entity_streams


def _architecture_entity_self_attention_layers(architecture_id: str) -> int:
    return int(actor_variant(architecture_id).entity_self_attention_layers)


def _deployment_architecture_ids(actor_kind: str) -> tuple[str, ...]:
    if actor_kind not in {"learner", "defender", "target"}:
        raise ValueError("Viewer model target must be learner or defender")
    return tuple(architecture
        for variant in viewer_actor_variants()
        if variant.unified_match_contract or variant.actor_kind == actor_kind
        for architecture in ((variant.architecture_id,
            variant.architecture_id.replace('_actor_critic', '_no_time_actor_critic'))
            if variant.unified_match_contract else (variant.architecture_id,)))


def _unified_match_architecture_id(
    settings: dict[str, object], actor_kind: str
) -> str:
    key = (
        str(settings.get("entity_streams", "separate")),
        int(settings.get("entity_self_attention_layers", 0)),
    )
    architecture_ids = (
        UNIFIED_MATCH_ATTACKER_ARCHITECTURE_IDS
        if actor_kind == "learner"
        else UNIFIED_MATCH_DEFENDER_ARCHITECTURE_IDS
        if actor_kind == "defender"
        else UNIFIED_MATCH_TARGET_ARCHITECTURE_IDS
        if actor_kind == "target"
        else None
    )
    if architecture_ids is None or key not in architecture_ids:
        raise ValueError("Unified Match training side or entity architecture is unsupported")
    return unified_match_architecture_id(*key, settings.get("include_remaining_time_observation", True))


def _configured_architecture_id(
    settings: dict[str, object], selected_architecture_id: str
) -> str:
    selected_variant = actor_variant(selected_architecture_id)
    if selected_variant.unified_match_contract:
        return _unified_match_architecture_id(settings, selected_variant.actor_kind)
    if (
        selected_architecture_id == ADVERSARIAL_GAME_BASELINE_ARCHITECTURE_ID
        and int(settings["learner_count_range"][1]) > 1
    ):
        return MULTI_GOAL_REACH_ARCHITECTURE_ID
    if selected_architecture_id == PATROL_PURSUIT_ARCHITECTURE_ID:
        defender_count_max = int(settings["defender_count_range"][1])
        key = (
            bool(settings["include_learner_speed_ratio_observation"]),
            str(settings["entity_streams"]),
            int(settings["entity_self_attention_layers"]),
        )
        if defender_count_max > 1:
            return MULTI_PATROL_VIEWER_ARCHITECTURE_IDS[key]
        return PATROL_VIEWER_ARCHITECTURE_IDS[key]
    return selected_architecture_id


def _checkpoint_training_lineage(
    payload: dict[str, object],
    visited: set[Path] | None = None,
) -> tuple[int, int]:
    update = int(payload.get("update_index", 0))
    agent_frames = int(payload.get("agent_frames", 0))
    initialization = payload.get("weights_initialization")
    if not (
        isinstance(initialization, dict)
        and initialization.get("kind") == "transfer_learning"
    ):
        return update, agent_frames
    prior_update = int(initialization.get("source_update", 0))
    prior_agent_frames = int(initialization.get("source_agent_frames", 0))
    source_value = initialization.get("source_checkpoint")
    if source_value:
        source = Path(str(source_value)).resolve()
        visited = set() if visited is None else set(visited)
        if source.is_file() and source not in visited:
            visited.add(source)
            source_payload = torch.load(source, map_location="cpu", weights_only=False)
            if isinstance(source_payload, dict):
                recursive_update, recursive_agent_frames = (
                    _checkpoint_training_lineage(source_payload, visited)
                )
                prior_update = max(prior_update, recursive_update)
                prior_agent_frames = max(
                    prior_agent_frames, recursive_agent_frames
                )
    return prior_update + update, prior_agent_frames + agent_frames


def _numbered_tensorboard_runs(root: Path) -> list[tuple[int, Path]]:
    runs = []
    if not root.is_dir():
        return runs
    for candidate in root.iterdir():
        prefix, separator, _name = candidate.name.partition("-")
        if candidate.is_dir() and separator and prefix.isdigit():
            runs.append((int(prefix), candidate))
    return runs


def _existing_tensorboard_path(output: Path) -> Path:
    suffix = f"-{output.name}"
    matching = [
        (order, candidate / "tensorboard")
        for order, candidate in _numbered_tensorboard_runs(output.parent)
        if candidate.name.endswith(suffix) and (candidate / "tensorboard").is_dir()
    ]
    if matching:
        return max(matching, key=lambda item: item[0])[1]
    return output / "tensorboard"


def _checkpoint_output_path(source_checkpoint: Path) -> Path:
    output = source_checkpoint.parent
    if output.parent.name == TRAINING_ARCHIVE_NAME:
        output = output.parent.parent
    return output


def _checkpoint_local_metrics_path(source_checkpoint: Path) -> Path:
    return source_checkpoint.parent / "metrics.jsonl"


def _resolve_recorded_goal_reach_checkpoint(
    reference_path: Path, source_value: object
) -> Path:
    source_text = str(source_value)
    source_checkpoint = Path(source_text)
    if source_checkpoint.is_file():
        return source_checkpoint.resolve()
    normalized_source = source_text.replace("\\", "/")
    marker = "Artifacts/GoalReach/"
    marker_index = normalized_source.find(marker)
    goal_reach_root = next(
        (
            parent
            for parent in (reference_path.parent, *reference_path.parents)
            if parent.name == "GoalReach"
        ),
        None,
    )
    if marker_index >= 0 and goal_reach_root is not None:
        return (
            goal_reach_root / normalized_source[marker_index + len(marker) :]
        ).resolve()
    return source_checkpoint.resolve()


def _checkpoint_lineage_elapsed_seconds(
    source_checkpoint: Path,
    visited: set[Path] | None = None,
    *,
    record_cache: dict[Path, dict[str, object]] | None = None,
    maximum_update: int | None = None,
) -> float:
    source_checkpoint = source_checkpoint.resolve()
    visited = set() if visited is None else set(visited)
    if source_checkpoint in visited or not source_checkpoint.is_file():
        return 0.0
    visited.add(source_checkpoint)
    try:
        stat = source_checkpoint.stat()
        payload = _training_dashboard_checkpoint_metadata(
            str(source_checkpoint), stat.st_mtime_ns, stat.st_size,
        )
    except (EOFError, OSError, RuntimeError, ValueError):
        return 0.0
    if not isinstance(payload, dict):
        return 0.0
    checkpoint_update = int(payload.get("update_index", 0)) if maximum_update is None else maximum_update
    local_elapsed = 0.0
    lineage_elapsed: float | None = None
    metrics_path = _checkpoint_local_metrics_path(source_checkpoint)
    if training_metrics_source(metrics_path).is_file():
        records, _ = _read_training_dashboard_records(metrics_path, record_cache)
        for record in records:
            if record.get("update") is None or int(record["update"]) > checkpoint_update:
                continue
            local_elapsed = float(record["elapsed_seconds"])
            if record.get("lineage_elapsed_seconds") is not None:
                lineage_elapsed = float(record["lineage_elapsed_seconds"])
    if lineage_elapsed is not None:
        return lineage_elapsed
    training_restore = payload.get("training_restore")
    if isinstance(training_restore, dict) and training_restore.get(
        "source_checkpoint"
    ):
        local_elapsed += _checkpoint_lineage_elapsed_seconds(
            _resolve_recorded_goal_reach_checkpoint(
                source_checkpoint, training_restore["source_checkpoint"]
            ),
            visited,
            record_cache=record_cache,
        )
    else:
        initialization = payload.get("weights_initialization")
        if (
            isinstance(initialization, dict)
            and initialization.get("kind") == "transfer_learning"
            and initialization.get("source_checkpoint")
        ):
            local_elapsed += _checkpoint_lineage_elapsed_seconds(
                _resolve_recorded_goal_reach_checkpoint(
                    source_checkpoint, initialization["source_checkpoint"]
                ),
                visited,
                record_cache=record_cache,
                maximum_update=int(initialization.get("source_update", 0)),
            )
    return local_elapsed


def _read_training_dashboard_records(
    metrics_path: Path,
    cache: dict[Path, dict[str, object]] | None = None,
    *,
    record_kind: str = "continuous",
) -> tuple[list[dict[str, object]], object]:
    resolved = metrics_path.resolve()
    state = cache.get(resolved) if cache is not None else None
    source = resolved if record_kind == "passage" else training_metrics_source(resolved)
    manifest_path = resolved.parent / "run.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if record_kind != "passage" and manifest_path.is_file() else {})
    if not source.is_file() or manifest.get("kind") == "binding-only":
        return [], manifest.get("architecture_id")
    stat = source.stat()
    identity = (int(stat.st_dev), int(stat.st_ino))
    if (
        state is None
        or state.get("record_kind", "continuous") != record_kind
        or state.get("identity") != identity
        or int(state.get("offset", 0)) > stat.st_size
        or (state.get('mtime_ns') != stat.st_mtime_ns and state.get('size') == stat.st_size)
    ):
        state = {
            "identity": identity,
            "record_kind": record_kind,
            "offset": 0,
            "remainder": b"",
            "records": [],
            "architecture_id": None,
            "elapsed_seconds": 0.0,
        }
        if cache is not None:
            cache[resolved] = state

    with source.open("rb") as stream:
        stream.seek(int(state["offset"]))
        chunk = stream.read()
        state["offset"] = stream.tell()
    data = bytes(state["remainder"]) + chunk
    lines = data.split(b"\n")
    state["remainder"] = b"" if data.endswith(b"\n") else lines.pop()
    if data.endswith(b"\n"):
        lines.pop()

    records = state["records"]
    assert isinstance(records, list)
    architecture_id = state["architecture_id"]
    elapsed_seconds = float(state["elapsed_seconds"])
    for raw_line in lines:
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            state.setdefault("errors", []).append(str(error))
            continue
        if not isinstance(value, dict):
            state.setdefault("errors", []).append("Metrics row is not an object")
            continue
        if record_kind == "passage":
            if type(value.get("update")) is not int or value["update"] < 0:
                state.setdefault("errors", []).append("Passage metrics update is not a nonnegative integer")
                continue
            records.append(value)
            continue
        if "roles" in value and manifest.get("checkpoint_schema") != "mobile_local_sensing_group":
            role = manifest.get("train_side")
            if role not in value["roles"]:
                continue
            value = role_metric_record(value, role)
        architecture_id = value.get("train/architecture_id", architecture_id)
        update_seconds = value.get("time/update_seconds")
        if (
            isinstance(update_seconds, int | float)
            and math.isfinite(float(update_seconds))
            and float(update_seconds) >= 0.0
        ):
            elapsed_seconds += float(update_seconds)
        lineage_elapsed_seconds = value.get(
            "train/progress/lineage_elapsed_seconds"
        )
        if not (
            isinstance(lineage_elapsed_seconds, int | float)
            and math.isfinite(float(lineage_elapsed_seconds))
            and float(lineage_elapsed_seconds) >= 0.0
        ):
            lineage_elapsed_seconds = None
        lineage_agent_frames = value.get("train/progress/lineage_agent_frames")
        record = {
            "update": value.get("train/joint_global_update", value.get("train/progress/update_index")),
            "role_update": value.get("train/progress/update_index"),
            "environment_steps": value.get("train/environment_steps", value.get("train/progress/environment_steps")),
            "lineage_update": value.get("train/progress/lineage_update_index"),
            "agent_frames": (
                lineage_agent_frames
                if lineage_agent_frames is not None
                else value.get("train/progress/agent_frames")
            ),
            "elapsed_seconds": (
                float(lineage_elapsed_seconds)
                if lineage_elapsed_seconds is not None
                else elapsed_seconds
            ),
            "lineage_elapsed_seconds": lineage_elapsed_seconds,
        }
        record.update({key: value.get(key) for key in TRAINING_DASHBOARD_METRICS})
        record["roles"] = value.get("roles", {})
        record['parameter_groups'] = value.get('parameter_groups', {})
        record["match/attacker_return"] = value.get("match/attacker_return")
        if (record.get('match/defender_win_rate') is None and value.get('match/count',0)>0
                and value.get('match/attacker_win_rate') is not None and manifest.get('match_rules_version') in (6,7)):
            record['match/defender_win_rate'] = 1.0-value['match/attacker_win_rate']-value.get('match/truncation_rate',0.)
        for key, legacy_key in LEGACY_TRAINING_DASHBOARD_METRICS.items():
            if record[key] is None:
                record[key] = value.get(legacy_key)
        record["time/average_updates_per_minute"] = (
            60.0 * (len(records) + 1) / elapsed_seconds
            if elapsed_seconds > 0.0
            else None
        )
        records.append(record)
    state["architecture_id"] = architecture_id
    state["elapsed_seconds"] = elapsed_seconds
    state['mtime_ns'] = stat.st_mtime_ns
    state['size'] = stat.st_size
    return records, architecture_id


def _training_dashboard_architecture_name(architecture_id: object) -> str:
    architecture = str(architecture_id or "unknown")
    if architecture.startswith("patrol_pursuit"):
        actor_label = "DEF"
    elif "multi_learner" in architecture:
        actor_label = "ATK · Multi"
    else:
        actor_label = "ATK · Single"
    try:
        architecture_label = actor_variant(architecture).display_name
    except ValueError:
        architecture_label = (
            architecture.removeprefix("goal_reach_")
            .removeprefix("patrol_pursuit_")
            .removesuffix("_actor_critic")
            .replace("_", " ")
        )
    return f"{actor_label} · {architecture_label}"


def _training_dashboard_parameters(payload: dict[str, object]) -> dict[str, object]:
    parameters = checkpoint_training_parameters(payload)
    training_settings = payload.get("training_settings")
    joint_training = (
        training_settings.get("joint_training")
        if isinstance(training_settings, dict)
        else None
    )
    train_sides = (
        joint_training.get("train_sides")
        if isinstance(joint_training, dict)
        else None
    )
    if isinstance(train_sides, (list, tuple)):
        configured_sides = {str(side) for side in train_sides}
        side_order = active_roles(training_settings)
        parameters["training_sides"] = [
            side for side in side_order if side in configured_sides
        ]
        parameters["fixed_sides"] = [
            side for side in side_order if side not in configured_sides
        ]
    if isinstance(joint_training, dict):
        for key, field in (
            ("mode", "joint_training_mode"),
            ("alternating_updates", "alternating_updates"),
        ):
            if key in joint_training:
                parameters[field] = joint_training[key]
    return parameters


@lru_cache(maxsize=128)
def _training_dashboard_checkpoint_metadata(
    checkpoint: str, mtime_ns: int, size: int
) -> dict[str, object]:
    del mtime_ns, size
    try:
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    except (EOFError, OSError, RuntimeError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    metadata = {
        name: payload[name]
        for name in (
            "update_index", "architecture_id",
            "goal_reach",
            "initial_rollout_metrics",
            "model_id",
            "scene_contract",
            "training_settings",
            "training_restore",
            "weights_initialization",
        )
        if name in payload
    }
    environment = payload.get("environment")
    if (
        isinstance(environment, dict)
        and environment.get("learner_count_range") is not None
    ):
        metadata["environment"] = {
            "learner_count_range": environment["learner_count_range"]
        }
    return metadata


def _training_dashboard_metadata_payload(metrics_path: Path) -> dict[str, object]:
    output = metrics_path.parent
    manifest_path = output / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    metadata = {}
    for checkpoint in (output / "checkpoint-latest.pt", output / INITIAL_CHECKPOINT_NAME):
        if checkpoint.is_file():
            stat = checkpoint.stat()
            metadata = dict(_training_dashboard_checkpoint_metadata(str(checkpoint.resolve()), stat.st_mtime_ns, stat.st_size))
            break
    if manifest:
        metadata["_run_manifest"] = manifest
        for key in ("architecture_id", "model_id", "scene_contract", "weights_initialization"):
            metadata.setdefault(key, manifest.get(key))
        launch = manifest.get("launch_settings")
        if "training_settings" not in metadata and isinstance(launch, dict):
            identity = _training_settings_identity(launch, str(manifest["architecture_id"]))
            identity["joint_training"] = {"run_group_id": manifest.get("run_group_id"),
                "mode": manifest.get("training_mode"), "train_sides": manifest.get("train_sides", [])}
            metadata["training_settings"] = identity
        metadata.setdefault("training_restore", {"mode": manifest.get("initialization_mode")})
    return metadata


def _training_dashboard_run_group_id(payload: dict[str, object]) -> str | None:
    training_settings = payload.get("training_settings")
    joint_training = (
        training_settings.get("joint_training")
        if isinstance(training_settings, dict)
        else None
    )
    if not isinstance(joint_training, dict) or not joint_training.get("run_group_id"):
        return None
    return str(joint_training["run_group_id"])


def _training_dashboard_update_offset(
    metrics_path: Path,
    payload: dict[str, object],
    visited: set[Path] | None = None,
) -> int:
    """Map checkpoint-local counters to ancestry, using the saved branch point."""
    visited = set() if visited is None else set(visited)
    if metrics_path.resolve() in visited:
        return 0
    visited.add(metrics_path.resolve())
    restore = payload.get("training_restore") or {}
    initialization = payload.get("weights_initialization") or {}
    restore = restore if isinstance(restore, dict) else {}
    initialization = initialization if isinstance(initialization, dict) else {}
    source_value = restore.get("source_checkpoint") or initialization.get("source_checkpoint")
    source_offset = 0
    if source_value:
        source = _resolve_recorded_goal_reach_checkpoint(metrics_path, source_value)
        source_metrics = _checkpoint_local_metrics_path(source)
        source_offset = _training_dashboard_update_offset(
            source_metrics, _training_dashboard_metadata_payload(source_metrics), visited,
        )
    if restore.get("mode") in {"continue", "joint", "alternating"}:
        return source_offset
    if initialization.get("kind") == "transfer_learning":
        # Never use the parent's latest update: it may have progressed since branching.
        return source_offset + int(initialization.get("source_update", 0))
    return source_offset


def _training_dashboard_lineage(
    metrics_path: Path,
    *,
    maximum_update: int | None = None,
    visited: set[Path] | None = None,
    record_cache: dict[Path, dict[str, object]] | None = None,
) -> tuple[list[dict[str, object]], object, str | None, dict[str, object]]:
    metrics_path = metrics_path.resolve()
    visited = set() if visited is None else set(visited)
    if metrics_path in visited:
        return [], None, None, {}
    visited.add(metrics_path)
    records, architecture_id = _read_training_dashboard_records(
        metrics_path, record_cache
    )
    elapsed_offset = 0.0
    source_update_offset = 0
    run_metadata: dict[str, object] = {}
    initial_payload = _training_dashboard_metadata_payload(metrics_path)
    training_restore: object = None
    if initial_payload:
        architecture_id = architecture_id or initial_payload.get("architecture_id")
        source_update_offset = _training_dashboard_update_offset(metrics_path, initial_payload)
        if initial_payload.get("model_id"):
            run_metadata["model_id"] = str(initial_payload["model_id"])
        initialization = (
            initial_payload.get("weights_initialization")
        )
        training_restore = (
            initial_payload.get("training_restore")
        )
        if isinstance(initialization, dict):
            run_metadata["weights_initialization"] = initialization
        if isinstance(training_restore, dict):
            run_metadata["training_restore"] = training_restore
        run_group_id = _training_dashboard_run_group_id(initial_payload)
        if run_group_id is not None:
            run_metadata["run_group_id"] = run_group_id
        series_signature = checkpoint_training_series_signature(initial_payload)
        if series_signature is not None:
            run_metadata["training_series_signature"] = series_signature
        if isinstance(training_restore, dict) and training_restore.get("mode"):
            run_metadata["training_mode"] = str(training_restore["mode"])
        training_settings = initial_payload.get("training_settings")
        if isinstance(training_settings, dict):
            run_metadata["training_settings"] = training_settings
        scene_contract = initial_payload.get("scene_contract")
        if isinstance(scene_contract, dict):
            run_metadata["scene_contract"] = scene_contract
        run_manifest = initial_payload.get("_run_manifest")
        if isinstance(run_manifest, dict):
            run_metadata["run_manifest"] = run_manifest
            if run_manifest.get("training_mode"):
                run_metadata["training_mode"] = str(
                    run_manifest["training_mode"]
                )
        if (
            isinstance(initialization, dict)
            and initialization.get("kind") == "transfer_learning"
        ):
            run_metadata["source_update"] = source_update_offset
            if initialization.get("source_checkpoint"):
                elapsed_offset = _checkpoint_lineage_elapsed_seconds(
                    _resolve_recorded_goal_reach_checkpoint(metrics_path, initialization["source_checkpoint"]),
                    record_cache=record_cache,
                    maximum_update=int(initialization.get("source_update", 0)),
                )
        if isinstance(training_restore, dict) and training_restore.get(
            "source_checkpoint"
        ):
            source_checkpoint = _resolve_recorded_goal_reach_checkpoint(
                metrics_path, training_restore["source_checkpoint"]
            )
            elapsed_offset = _checkpoint_lineage_elapsed_seconds(
                source_checkpoint, record_cache=record_cache,
                maximum_update=(initial_payload.get("_run_manifest") or {}).get("source_update"),
            )
            if records:
                first_update = records[0].get("update")
                try:
                    manifest = initial_payload.get("_run_manifest") or {}
                    local_start = (
                        int(manifest.get("source_update", max(0, int(first_update) - 1)))
                        if training_restore.get("mode") != "transfer_full_actor_critic" else 0
                    )
                    run_metadata["source_update"] = source_update_offset + local_start
                except (TypeError, ValueError):
                    pass
        parameters = _training_dashboard_parameters(initial_payload)
        if parameters:
            run_metadata["training_parameters"] = parameters
    normalized_records: list[dict[str, object]] = []
    for record in records:
        try:
            local_update = int(record.get("update"))
        except (TypeError, ValueError):
            continue
        update = source_update_offset + local_update
        if maximum_update is not None and update > maximum_update:
            continue
        normalized = dict(record)
        normalized["update"] = update
        normalized["lineage_update"] = update
        normalized["checkpoint_update"] = local_update
        if record.get("lineage_elapsed_seconds") is None:
            normalized["elapsed_seconds"] = elapsed_offset + float(
                record.get("elapsed_seconds", 0.0)
            )
        normalized_records.append(normalized)
    initial_metrics = initial_payload.get("initial_rollout_metrics")
    if (
        source_update_offset > 0
        and not (isinstance(training_restore, dict)
            and training_restore.get("mode") in {"continue", "joint", "alternating"})
        and isinstance(initial_metrics, dict)
        and not any(
            int(record["update"]) == source_update_offset
            for record in normalized_records
        )
    ):
        normalized_records.insert(
            0,
            {
                **{
                    key: value
                    for key, value in initial_metrics.items()
                    if key in TRAINING_DASHBOARD_METRICS
                },
                "update": source_update_offset,
                "agent_frames": int(
                    initialization.get("source_agent_frames", 0)
                    if isinstance(initialization, dict)
                    else 0
                ),
                "elapsed_seconds": elapsed_offset,
            },
        )
    display_name = _training_dashboard_architecture_name(architecture_id)
    root_metrics = metrics_path
    root_architecture_id = architecture_id
    root_display_name = display_name
    root_metadata = {
        key: value for key, value in run_metadata.items()
        if not key.startswith("lineage_")
    }
    if (
        isinstance(training_restore, dict)
        and training_restore.get("source_checkpoint")
        and normalized_records
    ):
        source_checkpoint = _resolve_recorded_goal_reach_checkpoint(
            metrics_path, training_restore["source_checkpoint"]
        )
        source_metrics = _checkpoint_local_metrics_path(source_checkpoint)
        source_payload = (
            _training_dashboard_metadata_payload(source_metrics)
            if source_metrics.is_file()
            else {}
        )
        source_signature = checkpoint_training_series_signature(source_payload)
        current_signature = run_metadata.get("training_series_signature")
        same_training_series = (
            training_restore.get("mode") == "continue"
            and initial_payload.get("model_id") == source_payload.get("model_id")
            and current_signature is not None and str(current_signature) == source_signature
        )
        if source_metrics.is_file() and source_metrics.resolve() not in visited:
            (
                source_records,
                source_architecture_id,
                source_display_name,
                source_metadata,
            ) = _training_dashboard_lineage(
                source_metrics,
                maximum_update=int(normalized_records[0]["update"]) - 1,
                visited=visited,
                record_cache=record_cache,
            )
            if same_training_series and source_records:
                normalized_records = [*source_records, *normalized_records]
                root_metrics = Path(
                    str(source_metadata.get("lineage_root_metrics", source_metrics))
                ).resolve()
                root_architecture_id = source_metadata.get(
                    "lineage_root_architecture_id", source_architecture_id
                )
                root_display_name = str(
                    source_metadata.get(
                        "lineage_root_display_name", source_display_name
                    )
                )
                root_metadata_value = source_metadata.get("lineage_root_metadata")
                root_metadata = (
                    dict(root_metadata_value)
                    if isinstance(root_metadata_value, dict)
                    else {
                        key: value for key, value in source_metadata.items()
                        if not key.startswith("lineage_")
                        and key != "source_segment_metrics"
                    }
                )
                source_lineage = source_metadata.get("source_segment_metrics", [])
                source_lineage_values = (
                    source_lineage if isinstance(source_lineage, list) else []
                )
                run_metadata["source_segment_metrics"] = [
                    str(source_metrics.resolve()),
                    *(str(path) for path in source_lineage_values),
                ]
            elif not same_training_series:
                parent_training_entry_id = source_metadata.get(
                    "training_entry_id"
                )
                if parent_training_entry_id is not None:
                    run_metadata["parent_training_entry_id"] = str(
                        parent_training_entry_id
                    )
                    root_metadata["parent_training_entry_id"] = str(
                        parent_training_entry_id
                    )
                stored_model_id = run_metadata.get("model_id")
                source_model_id = source_payload.get("model_id")
                training_entry_id = (
                    str(stored_model_id)
                    if is_training_model_timestamp_id(stored_model_id)
                    and str(stored_model_id) != str(source_model_id or "")
                    else training_model_timestamp_id(
                        training_run_started_at_ns(metrics_path.parent)
                    )
                )
                run_metadata["training_entry_id"] = training_entry_id
                root_metadata["training_entry_id"] = training_entry_id
    run_metadata["update_offset"] = source_update_offset
    training_entry_id = root_metadata.get("training_entry_id")
    if training_entry_id is None:
        stored_model_id = root_metadata.get("model_id")
        training_entry_id = (
            str(stored_model_id)
            if is_training_model_timestamp_id(stored_model_id)
            else training_model_timestamp_id(
                training_run_started_at_ns(root_metrics.parent)
            )
        )
        root_metadata["training_entry_id"] = training_entry_id
    run_metadata["training_entry_id"] = str(training_entry_id)
    run_metadata["lineage_root_metrics"] = str(root_metrics)
    run_metadata["lineage_root_architecture_id"] = root_architecture_id
    run_metadata["lineage_root_display_name"] = root_display_name
    run_metadata["lineage_root_metadata"] = root_metadata
    run_metadata["training_series_id"] = str(
        root_metadata.get("run_group_id") or root_metrics
    )
    return (
        normalized_records,
        architecture_id,
        display_name,
        run_metadata,
    )


def _training_dashboard_started_at(metrics_path: Path) -> str:
    started_at_ns = training_run_started_at_ns(metrics_path.parent)
    return (
        datetime.fromtimestamp(started_at_ns / 1_000_000_000)
        .astimezone()
        .isoformat(timespec="seconds")
    )


def _tensorboard_history_paths(output: Path) -> list[Path]:
    suffix = f"-{output.name}"
    matching = sorted(
        (
            (order, candidate / "tensorboard")
            for order, candidate in _numbered_tensorboard_runs(output.parent)
            if candidate.name.endswith(suffix) and (candidate / "tensorboard").is_dir()
        ),
        reverse=True,
    )
    paths = [path for _order, path in matching]
    direct = output / "tensorboard"
    if direct.is_dir():
        paths.append(direct)
    return paths


@contextmanager
def _tensorboard_order_lock(root: Path):
    lock_path = root / ".tensorboard-order.lock"
    with lock_path.open("a+b") as lock_file:
        lock_file.seek(0, 2)
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.01)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _allocate_tensorboard_path(
    output: Path, *, branch_source_update: int | None = None
) -> Path:
    root = output.parent
    with _tensorboard_order_lock(root):
        numbered = _numbered_tensorboard_runs(root)
        numbered_roots = {candidate.resolve() for _order, candidate in numbered}
        legacy_runs = {
            event.parent.resolve()
            for event in root.rglob("events.out.tfevents.*")
            if not any(numbered_root in event.parents for numbered_root in numbered_roots)
        }
        order = max(
            len(legacy_runs),
            max((item[0] for item in numbered), default=0),
        ) + 1
        run_name = (
            output.name
            if branch_source_update is None
            else f"from-u{int(branch_source_update)}-{output.name}"
        )
        while True:
            run_root = root / f"{order:04d}-{run_name}"
            try:
                run_root.mkdir()
            except FileExistsError:
                order += 1
                continue
            return run_root / "tensorboard"


def _copy_tensorboard_scalar_prefix(
    source_checkpoint: Path,
    writer: SummaryWriter,
    source_agent_frames: int,
    visited: set[Path] | None = None,
) -> None:
    source_checkpoint = source_checkpoint.resolve()
    visited = set() if visited is None else set(visited)
    if source_checkpoint in visited:
        raise ValueError("Transfer TensorBoard source chain contains a cycle")
    visited.add(source_checkpoint)
    source_output = _checkpoint_output_path(source_checkpoint)
    source_histories = _tensorboard_history_paths(source_output)
    if not source_histories:
        raise ValueError("Transfer source has no TensorBoard history to branch from")

    events = None
    scalar_tags: list[str] = []
    policy_prefix = []
    has_policy_series = False
    for source_tensorboard in source_histories:
        candidate_events = EventAccumulator(
            str(source_tensorboard), size_guidance={SCALARS: 0}
        )
        candidate_events.Reload()
        candidate_scalar_tags = candidate_events.Tags().get("scalars", [])
        policy_tag = next(
            (
                tag
                for tag in POLICY_LOSS_TENSORBOARD_TAGS
                if tag in candidate_scalar_tags
            ),
            None,
        )
        if policy_tag is None:
            continue
        has_policy_series = True
        candidate_policy_prefix = [
            event
            for event in candidate_events.Scalars(policy_tag)
            if int(event.step) <= source_agent_frames
        ]
        if (
            candidate_policy_prefix
            and int(candidate_policy_prefix[-1].step) == source_agent_frames
        ):
            events = candidate_events
            scalar_tags = candidate_scalar_tags
            policy_prefix = candidate_policy_prefix
            break
    if not has_policy_series:
        raise ValueError("Transfer source TensorBoard history has no policy-loss series")
    if events is not None and policy_prefix:
        for tag in scalar_tags:
            if tag not in TENSORBOARD_SCALAR_TAGS:
                continue
            for event in events.Scalars(tag):
                if int(event.step) <= source_agent_frames:
                    writer.add_scalar(
                        tag,
                        event.value,
                        int(event.step),
                        walltime=event.wall_time,
                    )
        writer.flush()
        return

    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    initialization = payload.get("weights_initialization") if isinstance(payload, dict) else None
    if not (
        isinstance(payload, dict)
        and isinstance(initialization, dict)
        and initialization.get("kind") == "transfer_learning"
        and initialization.get("source_checkpoint")
    ):
        raise ValueError("Transfer source TensorBoard history does not reach its checkpoint")
    ancestor_checkpoint = Path(str(initialization["source_checkpoint"])).resolve()
    if not ancestor_checkpoint.is_file():
        raise ValueError("Transfer source TensorBoard ancestor checkpoint is missing")
    ancestor_payload = torch.load(
        ancestor_checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(ancestor_payload, dict):
        raise ValueError("Transfer source TensorBoard ancestor checkpoint is invalid")
    _, ancestor_agent_frames = _checkpoint_training_lineage(ancestor_payload)
    stored_ancestor_agent_frames = int(
        initialization.get("source_agent_frames", 0)
    )
    local_agent_frames = int(payload.get("agent_frames", 0))
    stored_lineage_end = stored_ancestor_agent_frames + local_agent_frames

    local_events = None
    local_scalar_tags: list[str] = []
    for source_tensorboard in source_histories:
        candidate_events = EventAccumulator(
            str(source_tensorboard), size_guidance={SCALARS: 0}
        )
        candidate_events.Reload()
        candidate_scalar_tags = candidate_events.Tags().get("scalars", [])
        policy_tag = next(
            (
                tag
                for tag in POLICY_LOSS_TENSORBOARD_TAGS
                if tag in candidate_scalar_tags
            ),
            None,
        )
        if policy_tag is None:
            continue
        local_policy = [
            event
            for event in candidate_events.Scalars(policy_tag)
            if stored_ancestor_agent_frames < int(event.step) <= stored_lineage_end
        ]
        if local_policy and int(local_policy[-1].step) == stored_lineage_end:
            local_events = candidate_events
            local_scalar_tags = candidate_scalar_tags
            break
    if local_events is None:
        raise ValueError("Transfer source TensorBoard local branch is incomplete")

    _copy_tensorboard_scalar_prefix(
        ancestor_checkpoint,
        writer,
        ancestor_agent_frames,
        visited,
    )
    step_offset = ancestor_agent_frames - stored_ancestor_agent_frames
    for tag in local_scalar_tags:
        if tag not in TENSORBOARD_SCALAR_TAGS:
            continue
        for event in local_events.Scalars(tag):
            step = int(event.step)
            if stored_ancestor_agent_frames < step <= stored_lineage_end:
                writer.add_scalar(
                    tag,
                    event.value,
                    step + step_offset,
                    walltime=event.wall_time,
                )
    writer.flush()


class _ViewerPolicy:
    def __init__(self, model: ActorCritic):
        self.model = model

    @torch.no_grad()
    def refresh(self, observation: dict[str, torch.Tensor]) -> None:
        del observation

    def invalidate(self) -> None:
        return

    def actor_distribution(self, observation: dict[str, torch.Tensor]) -> object:
        return self.model.actor_distribution_from_observation(observation)


@dataclass
class _ViewerActorFrame:
    indices: torch.Tensor
    observation: dict[str, torch.Tensor]
    distribution: BoundedActionDistribution | None
    rule_action: torch.Tensor
    action: torch.Tensor
    display_context: dict[str, object] | None = None
    input_limits: tuple | None = None
    source_names: tuple = ()
    physics_ticks: int = 0
    physics_step_s: float = 0.0


@dataclass(frozen=True)
class _PreparedViewerDeployment:
    generation: int
    actor_kind: str
    model_id: str
    architecture_id: str
    source_kind: str
    selection_mode: str
    model: ActorCritic
    payload: dict[str, object] | None
    checkpoint: Path | None
    checkpoint_mtime_ns: int | None
    candidate: dict[str, object] | None
    persist: bool
    preserve_training_follow: bool


@dataclass(frozen=True)
class _PreparedJointTrainingLaunch:
    launch_settings: dict[str, object]
    architecture_ids: dict[str, str]
    sources: dict[str, str | None]
    source_metadata: dict[str, dict[str, object] | None]
    model_ids: dict[str, str]
    train_sides: frozenset[str]
    new_series_sides: frozenset[str]
    joint_mode: str
    alternating_updates: int
    updates: int
    seed: int


class GoalReachViewerSession:
    """B=1 interaction over the current GoalReach environment."""

    scene_mode = 'continuous'
    passage = None

    def __init__(self, checkpoint: Path | None = None):
        self._closed = False
        self.scene_mode = 'continuous'
        self.passage = None
        self._continuous_running = False
        self._discrete_running = True
        self._close_requested = False
        self._close_lock = threading.Lock()
        try:
            self._initialize(checkpoint)
        except BaseException:
            self.close()
            raise

    def _initialize(self, checkpoint: Path | None = None):
        self.device = torch.device("cpu")
        # B=1 viewer work is small; keep it off the GPU used by training.
        self.map_device = self.device
        self._snapshot_static_revision = 0
        self._snapshot_status_revision = 0
        self._cached_snapshot_static_revision = -1
        self._cached_snapshot_static: dict[str, object] = {}
        self._snapshot_static_dirty_deferred = False
        torch.manual_seed(int(GOAL_REACH["seed"]))
        base_config = _build_config()
        self.viewer_settings_path, self.viewer_settings = _load_viewer_settings(base_config)
        stored_architecture_id = str(
            self.viewer_settings["active_actor_variant_id"]
        )
        selected_architecture_id = _selected_scenario_architecture_id(
            stored_architecture_id
        )
        if stored_architecture_id != selected_architecture_id:
            self.viewer_settings["active_actor_variant_id"] = selected_architecture_id
            if stored_architecture_id == MULTI_GOAL_REACH_ARCHITECTURE_ID:
                learner_count_min, learner_count_max = self.viewer_settings[
                    "learner_count_range"
                ]
                self.viewer_settings["learner_count_range"] = [
                    int(learner_count_min), max(2, int(learner_count_max))
                ]
            elif actor_variant(stored_architecture_id).actor_kind == "defender":
                self.viewer_settings[
                    "include_learner_speed_ratio_observation"
                ] = _architecture_uses_learner_speed_observation(
                    stored_architecture_id
                )
                self.viewer_settings["entity_streams"] = (
                    _architecture_entity_streams(stored_architecture_id)
                )
                self.viewer_settings["entity_self_attention_layers"] = (
                    _architecture_entity_self_attention_layers(
                        stored_architecture_id
                    )
                )
            _store_viewer_settings(self.viewer_settings_path, self.viewer_settings)
        self.active_actor_variant_id = _configured_architecture_id(
            self.viewer_settings, selected_architecture_id
        )
        self.active_actor_kind = "learner"
        self.selected_actor_kind = self.viewer_settings.get("selected_actor_kind", "learner")
        self.selected_actor_index = 0
        payload: dict[str, object] | None = None
        if checkpoint is not None:
            requested_checkpoint = require_active_training_checkpoint(checkpoint)
            if not requested_checkpoint.is_file():
                raise FileNotFoundError(
                    f"Viewer checkpoint does not exist: {requested_checkpoint}"
                )
            cli_source_cache = {}
            requested_checkpoint, _, raw, _ = _read_group_release(requested_checkpoint, cli_source_cache)
            checkpoint = requested_checkpoint
            self.viewer_settings = _checkpoint_viewer_settings(raw, raw['architecture_id'])
            self.active_actor_variant_id = raw['architecture_id']
            payload = (_group_actor_payload(raw, 'learner', requested_checkpoint)
                       if self.viewer_settings['learner_control_mode'] != 'rule' else None)
            cli_release = raw
        self.reset_progress = {"phase": "ready"}
        self._edited_reset_generation = None
        self.config, self.env = self._new_environment(self.active_actor_variant_id)
        self.deployment_checkpoints = {
            variant.architecture_id: (
                self.config.repository_root
                / variant.output_directory
                / "checkpoint-latest.pt"
            ).resolve()
            for variant in viewer_actor_variants()
        }
        self.checkpoint = self.deployment_checkpoints.setdefault(self.active_actor_variant_id, (self.config.repository_root / actor_variant(self.active_actor_variant_id).output_directory / "checkpoint-latest.pt").resolve())
        if checkpoint is not None:
            self.checkpoint = checkpoint.resolve()
            self.deployment_checkpoints[self.active_actor_variant_id] = self.checkpoint
        self.metrics_path = self.checkpoint.parent / "metrics.jsonl"
        self._training_metrics_request_lock = threading.Lock()
        self._training_metrics_executor: ProcessPoolExecutor | None = None
        self._training_metrics_future: Future[tuple[str, bytes]] | None = None
        self._training_metrics_payload: bytes | None = None
        self._training_metrics_version = ""
        self._training_metrics_rebuild_dirty = True
        self._training_metrics_invalidation_generation = 0
        self._training_metrics_next_rebuild_at = 0.0
        self._training_lineage_future: Future[tuple[str, bytes]] | None = None
        self._training_lineage_payload: bytes | None = None
        self._training_lineage_version = ""
        self._training_lineage_rebuild_dirty = True
        self._training_lineage_next_rebuild_at = 0.0
        self.initial_checkpoint = self.checkpoint.parent / INITIAL_CHECKPOINT_NAME
        self._checkpoint_mtime_ns = (
            self.checkpoint.stat().st_mtime_ns if self.checkpoint.is_file() else None
        )
        self.model = self._new_actor_model(self.active_actor_variant_id)
        self._active_checkpoint_loaded = payload is not None
        self.weights_initialization = (
            dict(payload["weights_initialization"])
            if payload is not None
            and isinstance(payload.get("weights_initialization"), dict)
            else None
        )
        if payload is not None:
            self.model.load_state_dict(
                actor_deployment_state_dict(self.model, payload["model"]),
                strict=True,
            )
        self.model.eval()
        self.policy = _ViewerPolicy(self.model)
        self._checkpoint_candidates: list[dict[str, object]] = []
        self._initial_checkpoint_mtime_ns: int | None = None
        self.checkpoint_selection_mode = "auto"
        self._manual_checkpoint_update: int | None = None
        self._archived_checkpoint_model_id: str | None = None
        self._archived_checkpoint_path: Path | None = None
        self._checkpoint_source_model_id: str | None = None
        self._checkpoint_source_user_selected = checkpoint is not None
        self.latest_checkpoint_update = -1
        self.checkpoint_update = -1
        self.checkpoint_agent_frames = 0
        self.checkpoint_success_rate: float | None = None
        self.checkpoint_collision_rate: float | None = None
        self.checkpoint_refresh_error: str | None = None
        self._auto_deploy_training_run_group_id = (
            self.viewer_settings.get('auto_deploy_training_run_group_id') if checkpoint is None else None)
        self._auto_deploy_training_sides = set(self.env.roles) if self._auto_deploy_training_run_group_id else set()
        self._reset_checkpoint_candidates_on_next_update = False
        self._last_checkpoint_poll = time.perf_counter()
        self._checkpoint_load_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="goal-reach-checkpoint",
        )
        self._checkpoint_load_future: Future[
            tuple[Path, int, dict[str, object], ActorCritic]
        ] | None = None
        self._initial_checkpoint_load_future: Future[
            tuple[Path, int, dict[str, object], ActorCritic]
        ] | None = None
        self._pending_initial_checkpoint_load: (
            tuple[Path, int, dict[str, object]] | None
        ) = None
        self._opponent_checkpoint_load_future: Future[
            tuple[Path, int, dict[str, object], ActorCritic]
        ] | None = None
        self._pending_opponent_checkpoint_load: (
            tuple[Path, int, dict[str, object], ActorCritic] | None
        ) = None
        self._opponent_checkpoint_mtime_ns: int | None = None
        self._deployment_generations = {role: 0 for role in ("learner", "defender", "target")}
        self._deployment_futures: dict[
            str, tuple[int, Future[_PreparedViewerDeployment]]
        ] = {}
        self._pending_deployments: dict[str, _PreparedViewerDeployment] = {}
        self._deployment_errors = {}
        self._deployed_sources = {}
        self._deployment_client_request_ids = {}
        self._training_inventory_executor: ProcessPoolExecutor | None = (
            ProcessPoolExecutor(max_workers=1)
        )
        self._training_inventory_future: Future[
            tuple[str, dict[str, object]]
        ] | None = None
        self._training_status_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="goal-reach-training-status",
        )
        self._training_status_future: Future[
            tuple[
                int,
                str,
                dict[str, object],
                tuple[Path, ...],
                dict[str, tuple[float, float, float]],
            ]
        ] | None = None
        self._training_status_generation = 0
        self._joint_training_launch_executor: ProcessPoolExecutor | None = None
        self._joint_training_launch_future: Future[
            _PreparedJointTrainingLaunch
        ] | None = None
        self._joint_training_launch_pending: dict[str, object] | None = None
        self._joint_training_launch_queue: list[dict[str, object]] = []
        self._joint_training_starting: dict[str, dict[str, object]] = {}
        self._training_command_instance = None
        self._training_follow_request = None
        self._training_scene_load_future: Future[dict[str, object]] | None = None
        self._pending_checkpoint_load: (
            tuple[Path, int, dict[str, object], ActorCritic] | None
        ) = None
        self._pending_checkpoint_source_reset = False
        self._environment_update_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="goal-reach-environment",
        )
        self._environment_update_generation = 0
        self._environment_update_future: Future[
            tuple[int, str, ProjectConfig, GoalReachEnv | None, bool]
        ] | None = None
        self._environment_update_future_generation: int | None = None
        self._pending_environment_update: (
            tuple[
                int,
                str,
                int,
                ProjectConfig,
                type[GoalReachEnv],
                dict[str, object],
                bool,
            ]
            | None
        ) = None
        if payload is not None:
            self._remember_checkpoint(payload, latest=True)
        else:
            self._checkpoint_candidates = [
                {
                    "update": 0,
                    "agent_frames": 0,
                    "model": self.model.state_dict(),
                    "architecture_id": self.active_actor_variant_id,
                    "training_settings": None,
                    "success_rate": None,
                    "collision_rate": None,
                    "historical": False,
                }
            ]
            self.latest_checkpoint_update = 0
            self.checkpoint_update = 0
        self._initialize_actor_runtime_state()
        self._opponent_checkpoint_mtime_ns = (
            self.opponent_checkpoint.stat().st_mtime_ns
            if self.opponent_checkpoint is not None
            and self.opponent_checkpoint.is_file()
            else None
        )
        self.commands: deque[dict[str, object]] = deque()
        self.commands_lock = threading.Lock()
        self.snapshot = _MemorySnapshot()
        self.running = False
        self.time_scale = float(self.viewer_settings["time_scale"])
        self.snapshot_rate_hz = int(self.viewer_settings["snapshot_rate_hz"])
        self.rate_revision = 0
        self.sequence = 0
        self._next_snapshot_publish_at = 0.0
        self._snapshot_publish_times: deque[float] = deque(
            maxlen=2
            * max(
                rate
                for rate in VIEWER_RATE_OPTIONS["snapshot_rate_hz"]
                if rate != FULL_SPEED
            )
            + 1
        )
        self._performance_samples = {
            name: _RollingPerformanceWindow(VIEWER_PERFORMANCE_WINDOW_SAMPLES)
            for name in (
                "loop_gap_ms",
                "loop_work_ms",
                "snapshot_publish_gap_ms",
                "terrain_required_blocks_ms",
                "terrain_generate_ms",
                "terrain_cache_merge_ms",
                "terrain_repack_ms",
                *(
                    f"{stage}_{clock}_ms"
                    for stage in (
                        "checkpoint_refresh",
                        "training_status_refresh",
                        "training_status_registry_full",
                        "training_status_registry_incremental",
                        "training_status_process_poll",
                        "training_status_lock_scan",
                        "training_status_metrics_tail",
                        "training_status_delete_scan",
                        "training_status_apply",
                        "training_status_inventory_apply",
                        "training_status_delete_apply",
                        "training_status_follow_checkpoints",
                        "training_status_checkpoint_scan",
                        "training_status_inventory_launch",
                        "training_status_source_inventory",
                        "physics_step",
                        "physics_environment_step",
                        "physics_bookkeeping",
                        "actor_frame_refresh",
                        "applied_frame_encode",
                        "actor_learner_observation",
                        "actor_learner_inference",
                        "actor_defender_observation",
                        "actor_defender_inference",
                        "actor_target_observation",
                        "actor_target_inference",
                        "actor_action_merge",
                        "python_gc",
                        "python_gc_generation2",
                        "snapshot_publish",
                        "snapshot_performance_summary",
                        "snapshot_actor_inputs_encode",
                        "snapshot_static_status_build",
                        "snapshot_encode_store",
                        "environment_update_poll",
                    )
                    for clock in (
                        "wall",
                        "thread_cpu",
                        "process_cpu",
                        "off_cpu",
                    )
                ),
            )
        }
        self._performance_statistics: dict[str, dict[str, float | None]] = {}
        self._performance_statistics_updated_at = 0.0
        self.env.performance_recorder = self._record_performance_sample
        self._paced_due_ticks = _RollingPerformanceWindow(
            VIEWER_PERFORMANCE_WINDOW_SAMPLES
        )
        self._paced_batch_ticks = _RollingPerformanceWindow(
            VIEWER_PERFORMANCE_WINDOW_SAMPLES
        )
        self._paced_scheduler_statistics: dict[str, object] = {}
        self._last_loop_started_at: float | None = None
        self._last_snapshot_published_at: float | None = None
        self._next_physics_step_at = 0.0
        self._physics_rate_reset_requested = True
        self._physics_rate_measured = False
        self._physics_ticks_into_decision = 0
        self._physics_ticks_total = 0
        self._terminal_snapshot_until = 0.0
        self._terminal_environment_update_pending = False
        self._resume_after_environment_update = False
        self.command_error: str | None = None
        self.actor_training = ActorTrainingLauncher(self.config.repository_root)
        self._actor_training_catalog_state_signature: tuple[object, ...] = ()
        self._actor_training_lifecycle_state_signature: tuple[object, ...] = ()
        self._latest_effective_minibatch_agent_frames: dict[str, float | None] = {
            "learner": None,
            "defender": None,
        }
        self._known_running_deployment_checkpoints: dict[Path, tuple[int, int] | None] = {}
        self._actor_training_status_cache: dict[str, object] = {}
        self.edit_preview: dict[str, object] | None = None
        self._edited_scene = False
        self._pending_edited_scene_restore: (
            tuple[int, dict[str, object]] | None
        ) = None
        self._current_episode_initial_state: dict[str, object] | None = None
        self.scene_locked = False
        self.scene_lock_message: str | None = None
        self._locked_scene_state: dict[str, object] | None = None
        self._viewer_episode_revision = 0
        self._viewer_policy_session_seed = secrets.randbits(64)
        self._viewer_live_attempt = 0
        self._policy_noise = ViewerPolicyNoise(self._viewer_policy_session_seed, int(self.config.shared['clock']['maximum_decisions']))
        self._live_record = None
        self._last_complete_record = None
        self._replay_record = None
        self._replay_frame = None
        self._replay_index = 0
        self._replay_error = None
        self._live_scene_cursor = None
        self._run_request_id: str | None = None
        self._training_command_results: dict[str, dict[str, object]] = {}
        self._refresh_actor_training_status()
        self._training_status_reader = self.actor_training.inventory_clone()
        if checkpoint is None:
            self._restore_deployment_models()
        self._reset()
        # Startup has no published frame or HTTP clients yet.
        self._environment_update_future.result()
        self._poll_environment_update()
        if checkpoint is not None:
            cache = cli_source_cache
            # A CLI checkpoint is an explicit choice; late inventory must not restore saved defaults.
            for role in self.env.policy_roles:
                self._deployment_generations[role] += 1
            bundle = [self._prepare_viewer_deployment(self._deployment_generations[role], role,
                'saved_target:' + checkpoint.resolve().relative_to(self.config.repository_root).as_posix(),
                self.active_actor_variant_id, 'checkpoint', None, checkpoint, 'manual', None, False, False,
                prepared_sources=cache) for role in self.env.policy_roles]
            self._install_viewer_deployments(bundle)
            del cache, cli_release
        self._publish_snapshot(0.0)
        self._gc_started_at: tuple[float, float, float] | None = None
        self._gc_callback = self._handle_gc_event
        gc.callbacks.append(self._gc_callback)
        self._reset_performance_diagnostics()

    @staticmethod
    def _validate_checkpoint(payload: dict[str, object]) -> str:
        model = payload["model"]
        training_settings = payload.get("training_settings")
        payload_architecture_id = payload.get("architecture_id")
        training_architecture_id = (
            training_settings.get("architecture_id")
            if isinstance(training_settings, dict)
            else None
        )
        if (
            payload_architecture_id is not None
            and training_architecture_id is not None
            and payload_architecture_id != training_architecture_id
        ):
            raise ValueError("Viewer checkpoint training identity does not match its architecture")
        architecture_id = payload_architecture_id or training_architecture_id
        if not actor_variant(architecture_id).viewer_enabled:
            raise ValueError("Viewer checkpoint Actor scenario is unsupported")
        if not isinstance(model, dict):
            raise ValueError("Viewer checkpoint model state is incomplete")
        variant = actor_variant(architecture_id)
        if variant.checkpoint_shape_identity_from(model) != variant.checkpoint_shape_identity:
            raise ValueError("Viewer checkpoint architecture does not match its identity")
        return str(architecture_id)

    def _new_actor_model(self, architecture_id: str, actor_kind: str = "learner") -> ActorCritic:
        model = _initialized_model(architecture_id, int(GOAL_REACH["seed"]), self.device, build_critic=False)
        model.eval()
        return model

    def _initialize_actor_runtime_state(
        self,
        *,
        opponent_state: dict[str, object] | None = None,
    ) -> None:
        stochastic = self.viewer_settings["policy_sampling"] == "stochastic"
        active_kind = self.active_actor_kind
        opponent_kind = "defender" if active_kind == "learner" else "learner"
        opponent_architecture_id = (
            self.env.patrol_variant.architecture_id
            if opponent_kind == "defender"
            else self.env.learner_variant.architecture_id
        )
        preserved_opponent = bool(
            opponent_state is not None
            and opponent_state.get("actor_kind") == opponent_kind
            and opponent_state.get("architecture_id") == opponent_architecture_id
            and isinstance(opponent_state.get("model"), ActorCritic)
        )
        self.opponent_actor_kind = opponent_kind
        if preserved_opponent:
            self.opponent_model_id = str(opponent_state["model_id"])
            self.opponent_architecture_id = opponent_architecture_id
            checkpoint = opponent_state.get("checkpoint")
            self.opponent_checkpoint = (
                checkpoint if isinstance(checkpoint, Path) else None
            )
            self.opponent_update = int(opponent_state.get("update", 0))
            self.opponent_agent_frames = int(opponent_state.get("agent_frames", 0))
            self.opponent_selection = str(opponent_state.get("selection", "random"))
            option_cache = opponent_state.get("option_cache")
            self._opponent_option_cache = (
                option_cache if isinstance(option_cache, list) else None
            )
            self.opponent_model = opponent_state["model"]
        else:
            self.opponent_model_id = "random_current_architecture"
            self.opponent_architecture_id = opponent_architecture_id
            self.opponent_checkpoint = None
            self.opponent_update = 0
            self.opponent_agent_frames = 0
            self.opponent_selection = "random"
            self._opponent_option_cache = None
            self.opponent_model = self._new_actor_model(opponent_architecture_id, opponent_kind)
        self.opponent_model.requires_grad_(False)
        self.opponent_model.eval()
        self.opponent_policy = _ViewerPolicy(self.opponent_model)
        self.control = ActorControlMux(
            int(getattr(self.env, "viewer_learner_capacity", 1)),
            self.device,
            neural_policy=self.policy if active_kind == "learner" else self.opponent_policy,
            default_mode="nn",
            stochastic_neural=stochastic,
        )
        self.patrol_control = ActorControlMux(
            self.env.defender_count,
            self.device,
            neural_policy=self.policy if active_kind == "defender" else self.opponent_policy,
            default_mode="nn",
            stochastic_neural=stochastic,
        )
        for index in range(self.control.capacity):
            self.control.set_mode(index, str(self.viewer_settings["learner_control_mode"]))
        for index in range(self.patrol_control.capacity):
            self.patrol_control.set_mode(
                index, str(self.viewer_settings["defender_control_mode"])
            )
        previous_target = getattr(self, "target_runtime", None)
        self.role_controls = {"learner": self.control, "defender": self.patrol_control}
        if self.env.target_mode == "dynamic":
            architecture = self.env.target_variant.architecture_id
            if previous_target is None or previous_target["architecture_id"] != architecture:
                previous_target = dict(architecture_id=architecture, model=self._new_actor_model(architecture, "target"),
                                       model_id="random_current_architecture", checkpoint=None, update=0, agent_frames=0, loaded=True)
            self.target_runtime = previous_target
            self.target_control = ActorControlMux(self.env.target_capacity, self.device, neural_policy=_ViewerPolicy(previous_target["model"]),
                                                  default_mode=str(self.viewer_settings.get("target_control_mode", "nn")), stochastic_neural=stochastic)
            self.role_controls["target"] = self.target_control
            self.last_target_action = torch.tensor([[-1., 0.]], device=self.device).repeat(self.env.target_capacity,1)
        else:
            for name in ("target_runtime", "target_control", "last_target_action"):
                if hasattr(self, name):
                    delattr(self, name)
        self.last_learner_action = torch.zeros(
            (self.control.capacity, 2), dtype=torch.float32, device=self.device
        )
        self.last_learner_action[:, 0] = -1.0
        self.last_defender_action = torch.zeros(
            (self.env.defender_count, 2), dtype=torch.float32, device=self.device
        )
        if self.env.defender_count:
            self.last_defender_action[:, 0] = -1.0
        self.current_learner_outcomes = ["running"] * self.control.capacity
        self.episode_outcome_history: deque[dict[str, object]] = deque(maxlen=5)
        self.defender_episode_outcome_history: deque[dict[str, object]] = deque(maxlen=5)
        self.match_summary_history: deque[dict[str, object]] = deque(maxlen=5)
        self.current_match_event_counts = [0] * len(MATCH_EVENT_INFO_KEYS)
        self._actor_inputs_revision = 0
        self._applied_frame_revision = getattr(self, '_applied_frame_revision', 0)
        self._applied_frame_cache = None
        self._applied_frame_signature = None
        self._applied_frame_actions = {}
        self._applied_frame_modes = {}
        self._applied_frame_tick = 0
        self._encoded_actor_inputs_revision = -1
        self._next_actor_inputs_encode_at = 0.0
        self.encoded_actor_inputs: dict[str, list[dict[str, object]]] = {
            "learners": [],
            "defenders": [],
        }
        self.trajectory_sample_physics_ticks: list[int] = []
        self.trajectory_participants: dict[str, list[bool]] = {}
        self._trajectory_actor_was_active: dict[str, list[bool]] = {}
        self.trajectories: dict[
            str, list[list[tuple[float, float] | None]]
        ] = {
            "learner": [[] for _ in range(self.control.capacity)],
            "defender": [[] for _ in range(self.env.defender_count)],
        }
        self.trajectories["target"] = [[] for _ in range(self.env.target_capacity)]
        self.encoded_actor_inputs["targets"] = []
        self.event_freeze_frames: list[dict[str, object]] = []
        self._event_freeze_frame_keys: set[tuple[object, ...]] = set()

    def _checkpoint_metric(
        self, update: int, *, metrics_path: Path | None = None, actor_kind: str | None = None
    ) -> tuple[float | None, float | None, float | None] | None:
        source = training_metrics_source(metrics_path or self.metrics_path)
        if not source.is_file():
            return None
        actor_kind = actor_kind or self.active_actor_kind
        target_side = "attacker" if actor_kind == "learner" else "defender"

        def metric(record: dict[str, object]) -> tuple[float | None, float | None, float | None]:
            success_rate = record.get(f"match/{target_side}_win_rate")
            static_collision_rate = record.get(
                f"event/{target_side}/static_collision_per_episode"
            )
            vessel_collision_rate = record.get(
                f"event/{target_side}/vessel_collision_per_episode"
            )
            collision_rate = (
                float(static_collision_rate) + float(vessel_collision_rate)
                if static_collision_rate is not None
                and vessel_collision_rate is not None
                else record.get("rollout/collision_rate")
            )
            if success_rate is None and target_side == "defender" and record.get("match/attacker_win_rate") is not None:
                success_rate = 1.0 - float(record["match/attacker_win_rate"])
            if success_rate is None:
                success_rate = record.get("rollout/success_rate")
            return (
                None if success_rate is None else float(success_rate),
                None if collision_rate is None else float(collision_rate),
                record.get("match/attacker_return" if actor_kind == "learner" else "match/defense_return"),
            )

        latest = _last_jsonl_object(source)
        if latest is not None:
            latest = role_metric_record(latest, actor_kind)
        if latest is not None and latest.get("train/progress/update_index") == update:
            return metric(latest)
        return None

    def _clear_archived_checkpoint_selection(self) -> None:
        self._archived_checkpoint_model_id = None
        self._archived_checkpoint_path = None

    def _discard_training_follow_side(self, actor_kind: str) -> None:
        # A manual mix ends group following; never keep refreshing half a group.
        self._auto_deploy_training_sides.clear()
        self._auto_deploy_training_run_group_id = None
        future = getattr(self, "_follow_bundle_future", None)
        if future is not None:
            future.cancel()
            self._follow_bundle_future = None
        self._follow_bundle_signature = None
        if self.viewer_settings.get('auto_deploy_training_run_group_id'):
            self.viewer_settings['auto_deploy_training_run_group_id'] = None
            _store_viewer_settings(self.viewer_settings_path, self.viewer_settings)

    def _set_training_follow(self, group_id: str | None) -> None:
        if group_id:
            # Explicit selection must see runs launched/archived since startup;
            # the asynchronous status reader owns a separate registry cache.
            self.actor_training.refresh_training_run_registry()
            run = next((run for run in self.actor_training.training_run_registry()
                        if (run.get('manifest') or {}).get('run_group_id') == group_id), None)
            if run is None or run.get('archived'):
                raise ValueError('Auto deploy requires an available active training group')
            manifest = run['manifest']
            source_settings = _validated_viewer_settings(manifest['launch_settings'], historical=True)
            if ((source_settings['learner_control_mode'] == 'rule') != (self.viewer_settings['learner_control_mode'] == 'rule')
                    or (source_settings['learner_control_mode'] == 'rule'
                        and source_settings['learner_rule'] != self.viewer_settings['learner_rule'])):
                raise ValueError('Follow control source differs; load the training scene before following this group')
            if set(policy_roles(manifest['launch_settings'])) != set(self.env.roles):
                raise ValueError('Auto deploy group roles differ from the Viewer scene')
        self._discard_training_follow_side(self.active_actor_kind)
        self._auto_deploy_training_run_group_id = group_id or None
        self._auto_deploy_training_sides = set(self.env.roles) if group_id else set()
        self.viewer_settings['auto_deploy_training_run_group_id'] = group_id or None
        _store_viewer_settings(self.viewer_settings_path, self.viewer_settings)
        self.checkpoint_refresh_error = None
        self._follow_training_checkpoints(self._actor_training_status_cache)
        self._mark_snapshot_status_dirty()

    def _invalidate_rule_neural_candidates(self):
        """A rule selection owns ATK controls; late NN requests cannot replace it."""
        actor = 'learner'
        self._discard_training_follow_side(actor)
        self._deployment_generations[actor] += 1
        future = self._deployment_futures.pop(actor, None)
        if future is not None:
            future[1].cancel()
        self._pending_deployments.pop(actor, None)
        getattr(self, '_deployment_requested_models', {}).pop(actor, None)
        self._deployment_errors.pop(actor, None)
        for name in ('_checkpoint_load_future', '_initial_checkpoint_load_future'):
            future = getattr(self, name, None)
            if future is not None:
                future.cancel()
            setattr(self, name, None)
        self._pending_checkpoint_load = self._pending_initial_checkpoint_load = None

    def _set_checkpoint_source(
        self,
        checkpoint: Path,
        *,
        payload: dict[str, object] | None = None,
        prepared_model: ActorCritic | None = None,
        source_model_id: str | None,
        user_selected: bool,
        refresh_actor_frame: bool = True,
        preserve_training_follow: bool = False,
        mark_snapshot_static_dirty: bool = True,
    ) -> None:
        checkpoint = checkpoint.resolve()
        if payload is None:
            raise ValueError('Checkpoint installation requires its prepared role Actor')
        if self._validate_checkpoint(payload) != self.active_actor_variant_id:
            raise ValueError(
                "Deployment checkpoint does not match the active Actor scenario"
            )

        self.checkpoint = checkpoint
        self.metrics_path = checkpoint.parent / "metrics.jsonl"
        self.initial_checkpoint = checkpoint.parent / INITIAL_CHECKPOINT_NAME
        self._checkpoint_candidates = []
        self._initial_checkpoint_mtime_ns = None
        self.latest_checkpoint_update = -1
        self.checkpoint_update = -1
        self.checkpoint_agent_frames = 0
        self.checkpoint_success_rate = None
        self.checkpoint_collision_rate = None
        self.checkpoint_selection_mode = "auto"
        self._manual_checkpoint_update = None
        self._clear_archived_checkpoint_selection()
        self._checkpoint_source_model_id = source_model_id
        self._checkpoint_source_user_selected = user_selected
        if not preserve_training_follow:
            self._discard_training_follow_side(
                self.active_actor_kind
            )
        self._reset_checkpoint_candidates_on_next_update = False
        self._pending_checkpoint_load = None
        self._pending_checkpoint_source_reset = False
        if self._initial_checkpoint_load_future is not None:
            self._initial_checkpoint_load_future.cancel()
            self._initial_checkpoint_load_future = None
        self._pending_initial_checkpoint_load = None
        self.weights_initialization = (
            dict(payload["weights_initialization"])
            if isinstance(payload.get("weights_initialization"), dict)
            else None
        )
        if prepared_model is not None:
            self.model = prepared_model
            self.model.eval()
            self.policy.model = prepared_model
            self.policy.invalidate()
        self._remember_checkpoint(
            payload,
            latest=True,
            refresh_actor_frame=refresh_actor_frame,
            load_model=prepared_model is None,
            mark_snapshot_static_dirty=mark_snapshot_static_dirty,
        )
        self._checkpoint_mtime_ns = checkpoint.stat().st_mtime_ns
        self._active_checkpoint_loaded = True
        self.checkpoint_refresh_error = None

    def _deploy_checkpoint(
        self,
        candidate: dict[str, object],
        *,
        force: bool = False,
        refresh_actor_frame: bool = True,
        load_model: bool = True,
        mark_snapshot_static_dirty: bool = True,
    ) -> None:
        selected_update = int(candidate["update"])
        if force or selected_update != self.checkpoint_update:
            if candidate["architecture_id"] != self.active_actor_variant_id:
                raise ValueError("Viewer checkpoint directory mixes Actor scenarios")
            if load_model:
                self.model.load_state_dict(
                    actor_deployment_state_dict(self.model, candidate["model"]),
                    strict=True,
                )
                self.model.eval()
                self.policy.invalidate()
                if refresh_actor_frame and hasattr(self, "actor_frames"):
                    self._refresh_actor_frame()
        self.checkpoint_update = selected_update
        self.checkpoint_agent_frames = int(candidate["agent_frames"])
        self.checkpoint_success_rate = candidate["success_rate"]
        self.checkpoint_collision_rate = candidate["collision_rate"]
        initialization = candidate.get("weights_initialization")
        self.weights_initialization = (
            dict(initialization) if isinstance(initialization, dict) else None
        )
        if (
            self.checkpoint_agent_frames > 0
            or candidate.get("checkpoint_role")
            in {"transfer_initialization", "training_initialization"}
        ):
            self._active_checkpoint_loaded = True
        if mark_snapshot_static_dirty:
            self._mark_snapshot_static_dirty()

    def _select_latest_checkpoint(
        self,
        *,
        refresh_actor_frame: bool = True,
        load_model: bool = True,
        mark_snapshot_static_dirty: bool = True,
    ) -> None:
        selected = next(
            (
                candidate
                for candidate in self._checkpoint_candidates
                if int(candidate["update"]) == self.latest_checkpoint_update
            ),
            None,
        )
        if selected is None:
            raise ValueError("Viewer has no latest checkpoint")
        self._deploy_checkpoint(
            selected,
            refresh_actor_frame=refresh_actor_frame,
            load_model=load_model,
            mark_snapshot_static_dirty=mark_snapshot_static_dirty,
        )

    def _remember_checkpoint(
        self,
        payload: dict[str, object],
        *,
        latest: bool = False,
        historical: bool = False,
        refresh_actor_frame: bool = True,
        load_model: bool = True,
        mark_snapshot_static_dirty: bool = True,
    ) -> None:
        update = int(payload["update_index"])
        if latest:
            self.latest_checkpoint_update = update
            self._checkpoint_candidates = [
                candidate
                for candidate in self._checkpoint_candidates
                if bool(candidate["historical"])
                or int(candidate["update"]) == self._manual_checkpoint_update
            ]
        existing = next(
            (
                candidate
                for candidate in self._checkpoint_candidates
                if int(candidate["update"]) == update
            ),
            None,
        )
        if existing is None:
            metric = payload["_viewer_checkpoint_metric"] if "_viewer_checkpoint_metric" in payload else self._checkpoint_metric(update)
            initial_metrics = payload.get("initial_rollout_metrics")
            if update == 0 and isinstance(initial_metrics, dict):
                actor_kind = self.active_actor_kind
                target_side = "attacker" if actor_kind == "learner" else "defender"
                success_value = initial_metrics.get(
                    f"match/{target_side}_win_rate",
                    initial_metrics.get("success_rate"),
                )
                static_collision_value = initial_metrics.get(
                    f"event/{target_side}/static_collision_per_episode"
                )
                vessel_collision_value = initial_metrics.get(
                    f"event/{target_side}/vessel_collision_per_episode"
                )
                collision_value = (
                    float(static_collision_value) + float(vessel_collision_value)
                    if static_collision_value is not None
                    and vessel_collision_value is not None
                    else initial_metrics.get("collision_rate")
                )
                metric = (
                    None if success_value is None else float(success_value),
                    None if collision_value is None else float(collision_value),
                    (
                        2.0 * float(success_value) - 1.0
                        if success_value is not None
                        else None
                    ),
                )
            architecture_id = self._validate_checkpoint(payload)
            if architecture_id != self.active_actor_variant_id:
                raise ValueError("Viewer checkpoint source changed Actor scenario")
            weights_initialization = payload.get("weights_initialization")
            self._checkpoint_candidates.append(
                {
                    "update": update,
                    "agent_frames": int(payload["agent_frames"]),
                    "model": payload["model"],
                    "architecture_id": architecture_id,
                    "training_settings": payload.get("training_settings"),
                    "success_rate": metric[0] if metric is not None else None,
                    "collision_rate": metric[1] if metric is not None else None,
                    "episode_return_mean": metric[2] if metric is not None else None,
                    "historical": historical,
                    "weights_initialization": weights_initialization,
                    "checkpoint_role": payload.get("checkpoint_role"),
                    "scene_contract": payload.get("scene_contract"),
                    "initial_rollout_metrics": initial_metrics,
                    "name_fields": model_name_fields(
                        architecture_id,
                        self.checkpoint.parent.name,
                        update=update,
                        reward=metric[2] if metric is not None else None,
                        payload=payload,
                    ),
                }
            )
        elif historical:
            existing["historical"] = True
        if latest and self.checkpoint_selection_mode == "auto":
            self._select_latest_checkpoint(
                refresh_actor_frame=refresh_actor_frame,
                load_model=load_model,
                mark_snapshot_static_dirty=mark_snapshot_static_dirty,
            )

    def _opponent_model_options(self) -> list[dict[str, object]]:
        if self._opponent_option_cache is None:
            actor_kind = self.opponent_actor_kind
            random_architecture_id = _unified_match_architecture_id(
                self.viewer_settings, actor_kind
            )
            options: list[dict[str, object]] = [
                {
                    "id": "random_current_architecture",
                    "kind": "random_initial",
                    "label": (
                        "ATK" if actor_kind == "learner" else "DEF"
                    ) + " Unified Match · Random Init · Untrained",
                    "name_fields": model_name_fields(
                        random_architecture_id,
                        "random",
                        update=0,
                        reward=None,
                        training_mode="fresh",
                        training_parameters={"seed": int(GOAL_REACH["seed"])},
                    ),
                }
            ]
            options.extend(self._training_opponent_options(actor_kind))
            if (
                self.opponent_model_id is not None
                and not any(
                    str(option["id"]) == self.opponent_model_id
                    for option in options
                )
            ):
                options.append(
                    {
                        "id": self.opponent_model_id,
                        "kind": "archived_trained_target",
                        "label": (
                            ("Current ATK" if actor_kind == "learner" else "Current DEF")
                            + " · u"
                            f"{self.opponent_update} · Direct"
                        ),
                    }
                )
            self._opponent_option_cache = options
        resolved_options: list[dict[str, object]] = []
        for option in self._opponent_option_cache:
            selected = str(option["id"]) == self.opponent_model_id
            name_fields = option.get("name_fields")
            if selected and isinstance(name_fields, dict):
                name_fields = {**name_fields, "update": self.opponent_update}
            resolved_options.append(
                {
                    **option,
                    "name_fields": name_fields,
                    "selected": selected,
                }
            )
        return resolved_options

    def _opponent_model_descriptor(self) -> dict[str, object]:
        options = self._opponent_model_options()
        selected = next(
            (
                option
                for option in options
                if str(option["id"]) == self.opponent_model_id
            ),
            None,
        )
        return {
            "actor_kind": self.opponent_actor_kind,
            "architecture_id": self.opponent_architecture_id,
            "checkpoint": (
                str(self.opponent_checkpoint)
                if self.opponent_checkpoint is not None
                else None
            ),
            "update": self.opponent_update,
            "agent_frames": self.opponent_agent_frames,
            "label": (
                str(selected["label"])
                if selected is not None
                else ("Multi-ATK NN" if self.opponent_actor_kind == "learner" else "Multi-DEF NN")
            ),
            "selected_model_id": self.opponent_model_id,
            "selection": self.opponent_selection,
            "selection_mode": self.opponent_selection,
            "options": options,
        }

    def _select_opponent_checkpoint(
        self,
        model_id: str,
        *,
        checkpoint_override: Path | None = None,
        persist: bool = False,
    ) -> None:
        architecture_id = self.opponent_architecture_id
        source_kind = "random"
        source_id: str | None = None
        checkpoint: Path | None = None
        if checkpoint_override is None and model_id.startswith("embedded_opponent:"):
            checkpoint_override = (
                self.config.repository_root
                / model_id.removeprefix("embedded_opponent:")
            ).resolve()
            artifact_root = (
                self.config.repository_root / "Artifacts" / "GoalReach"
            ).resolve()
            if (
                not checkpoint_override.is_file()
                or not checkpoint_override.is_relative_to(artifact_root)
            ):
                raise ValueError("Embedded opponent checkpoint is unavailable")
        if checkpoint_override is not None:
            checkpoint = checkpoint_override.resolve()
            source_kind = "embedded"
        elif model_id == "random_current_architecture":
            architecture_id = _unified_match_architecture_id(
                self.viewer_settings, self.opponent_actor_kind
            )
        else:
            option = next(
                (
                    item
                    for item in self._opponent_model_options()
                    if str(item["id"]) == model_id
                ),
                None,
            )
            if option is None:
                raise ValueError("Opponent selection requires a valid model ID")
            architecture_id = str(option["source_architecture_id"])
            if model_id.startswith("saved_target:"):
                source_id = model_id.removeprefix("saved_target:")
                source_kind = "saved"
            elif model_id.startswith("archived_target:"):
                source_id = model_id.removeprefix("archived_target:")
                source_kind = "archived"
            else:
                raise ValueError("Opponent selection requires a valid model ID")
        self._queue_viewer_deployment(
            actor_kind=self.opponent_actor_kind,
            model_id=model_id,
            architecture_id=architecture_id,
            source_kind=source_kind,
            source_id=source_id,
            checkpoint=checkpoint,
            selection_mode=source_kind,
            persist=persist,
        )

    def _apply_opponent_model(
        self,
        *,
        model_id: str,
        architecture_id: str,
        checkpoint: Path | None,
        model: ActorCritic,
        update: int,
        agent_frames: int,
        selection: str,
        refresh_actor_frame: bool = True,
    ) -> None:
        model.requires_grad_(False)
        model.eval()
        architecture_changed = architecture_id != self.opponent_architecture_id
        if self.opponent_actor_kind == "learner":
            if architecture_changed:
                self.env.set_learner_architecture(architecture_id)
            self.control.neural_policy = _ViewerPolicy(model)
        else:
            if architecture_changed:
                self.env.set_patrol_architecture(architecture_id)
            self.patrol_control.neural_policy = _ViewerPolicy(model)
        self.opponent_model = model
        self.opponent_policy = (
            self.control.neural_policy
            if self.opponent_actor_kind == "learner"
            else self.patrol_control.neural_policy
        )
        self.opponent_model_id = model_id
        self.opponent_architecture_id = architecture_id
        self.opponent_checkpoint = checkpoint
        self.opponent_update = update
        self.opponent_agent_frames = agent_frames
        self.opponent_selection = selection
        if hasattr(self, "_opponent_checkpoint_mtime_ns"):
            self._opponent_checkpoint_mtime_ns = (
                checkpoint.stat().st_mtime_ns
                if checkpoint is not None and checkpoint.is_file()
                else None
            )
            self._pending_opponent_checkpoint_load = None
        # Selection and update labels are resolved from live fields when the
        # descriptor is built. Switching among known models does not change the
        # catalog and must not trigger a filesystem scan during publication.
        if architecture_changed or (
            self._opponent_option_cache is not None
            and not any(str(option["id"]) == model_id for option in self._opponent_option_cache)
        ):
            self._opponent_option_cache = None
        if refresh_actor_frame and hasattr(self, "actor_frames"):
            self._refresh_actor_frame()

    def _select_random_current_checkpoint(
        self,
        *,
        forget_saved_checkpoints: bool = False,
        prepared_model: ActorCritic | None = None,
        refresh_actor_frame: bool = True,
    ) -> None:
        random_model = (
            self._new_actor_model(self.active_actor_variant_id)
            if prepared_model is None
            else prepared_model
        )
        self.model = random_model
        self.policy.model = random_model
        self.model.eval()
        self.policy.invalidate()
        if refresh_actor_frame and hasattr(self, "actor_frames"):
            self._refresh_actor_frame()
        self.checkpoint_selection_mode = "random"
        self._manual_checkpoint_update = None
        self._clear_archived_checkpoint_selection()
        self.checkpoint_update = 0
        self.checkpoint_agent_frames = 0
        self.checkpoint_success_rate = None
        self.checkpoint_collision_rate = None
        self.weights_initialization = None
        self._active_checkpoint_loaded = False
        if forget_saved_checkpoints:
            self._checkpoint_candidates = [
                {
                    "update": 0,
                    "agent_frames": 0,
                    "model": self.model.state_dict(),
                    "architecture_id": self.active_actor_variant_id,
                    "training_settings": None,
                    "success_rate": None,
                    "collision_rate": None,
                    "historical": False,
                    "weights_initialization": None,
                }
            ]
            self._initial_checkpoint_mtime_ns = None
            self._checkpoint_mtime_ns = None
            self.latest_checkpoint_update = 0

    def _persist_deployment_model(self, actor: str, model_id: str) -> None:
        deployment_model_ids = dict(self.viewer_settings["deployment_model_ids"])
        if deployment_model_ids.get(actor) == model_id:
            return
        deployment_model_ids[actor] = model_id
        self.viewer_settings["deployment_model_ids"] = deployment_model_ids
        _store_viewer_settings(self.viewer_settings_path, self.viewer_settings)

    def _restore_deployment_models(self, *, only_unrequested: bool = False) -> None:
        deployment_model_ids = self.viewer_settings["deployment_model_ids"]
        # The single loader worker can share immutable releases and their models
        # across this restore batch, just as Follow and initial bundles do.
        prepared_sources = {}
        active_kind = self.active_actor_kind
        actor_order = [
            active_kind,
            *(kind for kind in active_roles(self.viewer_settings) if kind != active_kind),
        ]
        for actor in actor_order:
            if only_unrequested and self._deployment_generations[actor] != 0:
                continue
            if actor in self._auto_deploy_training_sides and not (
                    actor == 'learner' and self.viewer_settings['learner_control_mode'] == 'rule'):
                continue
            model_id = str(deployment_model_ids.get(actor) or "")
            if not model_id:
                continue
            try:
                self._select_checkpoint(
                    {"actor": actor, "model_id": model_id}, persist=False,
                    prepared_sources=prepared_sources,
                )
            except (KeyError, OSError, RuntimeError, ValueError):
                continue

    def _select_checkpoint(self, command: dict[str, object], *, persist: bool = True,
                           prepared_sources: dict | None = None) -> None:
        role = str(command.get("actor", self.active_actor_kind))
        if role not in self.env.roles:
            raise ValueError("Cannot deploy an inactive role")
        if persist and role == 'learner' and self.viewer_settings['learner_control_mode'] == 'rule':
            raise ValueError('Switch ATK to NN before deploying weights')
        self._deployment_client_request_ids[role] = command.get("client_request_id")
        try:
            model_id = str(command.get("model_id", command.get("value", "")))
            if not hasattr(self,"_deployment_requested_models"):self._deployment_requested_models={}
            self._deployment_requested_models[role]=model_id
            if model_id in {"random_initial", "random_current_architecture"}:
                source, architecture = None, _unified_match_architecture_id(self.viewer_settings, role)
                model_id = "random_current_architecture"
            else:
                source, architecture = self._resolve_model_reference(role, model_id)
                # A release alias is not an identity. Reuse only the actual loaded
                # immutable file, matching its role, publication and file timestamp.
                loaded = self._deployed_sources.get(role)
                manifest = json.loads((source.parent/'run.json').read_text(encoding='utf-8'))
                current = Path(loaded['checkpoint_path']) if loaded else None
                expected = _manifest_checkpoint_path(source.parent, manifest) if source.name == 'checkpoint-latest.pt' and not manifest.get('launch_settings', {}).get('population') else source
                mtime = (self.target_runtime.get('mtime_ns') if role == 'target' else
                    self._checkpoint_mtime_ns if role == self.active_actor_kind else self._opponent_checkpoint_mtime_ns)
                if (loaded and current == expected and current.is_file() and current.stat().st_mtime_ns == mtime
                        and loaded['role'] == role and loaded['run_group_id'] == manifest.get('run_group_id')
                        and loaded['architecture_id'] == architecture
                        and loaded.get('publication_sequence') == manifest.get('publication_sequence')):
                    self._validate_deployment_task(SimpleNamespace(actor_kind=role, architecture_id=architecture,
                        payload={'launch_settings':manifest['launch_settings']}))
                    self._deployment_generations[role] += 1
                    previous = self._deployment_futures.pop(role, None)
                    if previous is not None: previous[1].cancel()
                    self._pending_deployments.pop(role, None)
                    self._deployment_errors.pop(role, None)
                    self._mark_snapshot_static_dirty()
                    return
            self._queue_viewer_deployment(actor_kind=role, model_id=model_id, architecture_id=architecture,
                source_kind="random" if source is None else "checkpoint", checkpoint=source,
                selection_mode="random" if source is None else "manual", persist=persist,
                prepared_sources=prepared_sources,
                preserve_training_follow=(not persist and role == 'learner'
                    and self.viewer_settings['learner_control_mode'] == 'rule'))
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            self._deployment_generations[role] += 1
            previous = self._deployment_futures.pop(role, None)
            if previous is not None:
                previous[1].cancel()
            self._pending_deployments.pop(role, None)
            self._deployment_errors[role] = str(error)
            self._mark_snapshot_static_dirty()
            raise

    def _checkpoint_options(self) -> list[dict[str, object]]:
        deployment_options = self._actor_training_status_cache.get(
            "deployment_checkpoint_options", []
        )
        try:
            current_source_id = self.checkpoint.resolve().relative_to(
                self.config.repository_root
            ).as_posix()
        except ValueError:
            current_source_id = None
        current_source_option = next(
            (
                option
                for option in deployment_options
                if (
                    str(option.get("model_id")) == self._checkpoint_source_model_id
                    or str(option.get("id")) == current_source_id
                )
            ),
            None,
        )
        current_run_root = self.checkpoint.parent
        current_run_name = str(
            current_source_option.get("run_name", current_run_root.name)
            if current_source_option is not None
            else current_run_root.name
        )
        def copied_name_fields(value: object) -> dict[str, object] | None:
            return dict(value) if isinstance(value, dict) else None

        random_architecture_id = _unified_match_architecture_id(
            self.viewer_settings,
            self.active_actor_kind,
        )
        random_name_fields = model_name_fields(
            random_architecture_id,
            "random",
            update=0,
            reward=None,
            training_mode="fresh",
            training_parameters={"seed": int(GOAL_REACH["seed"])},
        )

        def saved_run_options() -> list[dict[str, object]]:
            resolved_options: list[dict[str, object]] = []
            for option in deployment_options:
                selected = (
                    self.checkpoint_selection_mode not in {"random", "archived"}
                    and (
                        str(option["model_id"]) == self._checkpoint_source_model_id
                        or str(option.get("run_name", "")) == current_run_name
                    )
                )
                name_fields = copied_name_fields(option.get("name_fields"))
                if selected and name_fields is not None:
                    name_fields["update"] = self.checkpoint_update
                resolved_options.append(
                    {
                        "id": str(option["model_id"]),
                        "kind": str(option["kind"]),
                        "label": str(option["label"]),
                        "name_fields": name_fields,
                        "source_architecture_id": str(
                            option.get("source_architecture_id", "")
                        ),
                        "selected": selected,
                        "run_name": str(option.get("run_name", "")),
                        "run_started_at_ns": int(option.get("run_started_at_ns", 0)),
                        "model_timestamp_id": str(
                            option.get("model_timestamp_id", "")
                        ),
                        "run_id": str(option.get("run_id", "")),
                    }
                )
            return resolved_options

        def archived_options() -> list[dict[str, object]]:
            return [
                {
                    "id": str(option["model_id"]),
                    "kind": str(option["kind"]),
                    "label": str(option["label"]),
                    "name_fields": copied_name_fields(option.get("name_fields")),
                    "source_architecture_id": str(
                        option.get("source_architecture_id", "")
                    ),
                    "selected": (
                        self.checkpoint_selection_mode == "archived"
                        and str(option["model_id"])
                        == self._archived_checkpoint_model_id
                    ),
                    "run_name": str(option.get("run_name", "")),
                    "run_started_at_ns": int(option.get("run_started_at_ns", 0)),
                    "model_timestamp_id": str(
                        option.get("model_timestamp_id", "")
                    ),
                    "run_id": str(option.get("run_id", "")),
                }
                for option in self._actor_training_status_cache.get(
                    "archived_checkpoint_options", []
                )
            ]

        options = [
            {
                "id": "random_current_architecture",
                "kind": "random_initial",
                "label": (
                    "Unified Match · Random Init"
                    f" · seed{int(GOAL_REACH['seed'])} · r=— · Untrained"
                ),
                "name_fields": random_name_fields,
                "selected": self.checkpoint_selection_mode == "random",
            },
        ]
        options.extend(saved_run_options())
        options.extend(archived_options())
        return options

    def _selected_checkpoint_model_id(self) -> str:
        if self.checkpoint_selection_mode == "random":
            return "random_current_architecture"
        selected = next(
            (
                str(option["id"])
                for option in self._checkpoint_options()
                if bool(option.get("selected"))
            ),
            None,
        )
        if selected is not None:
            return selected
        return ""

    def _deployment_policy_label(self, prefix: str = "NN") -> str:
        if self.checkpoint_selection_mode == "random":
            return (
                f"{prefix} · Random Init · "
                f"seed{int(GOAL_REACH['seed'])} · Untrained"
            )
        if self.checkpoint_selection_mode == "archived":
            return (
                f"{prefix} · Archived · "
                f"update {self.checkpoint_update} · Manual"
            )
        if self._checkpoint_source_model_id is not None:
            return (
                f"{prefix} · {self.checkpoint.parent.name} · "
                f"update {self.checkpoint_update} · "
                + ("Latest" if self.checkpoint_selection_mode == "auto" else "Manual")
            )
        initialization = self.weights_initialization
        if (
            initialization is not None
            and initialization.get("transfer_kind")
            == "shape_exact_legacy_checkpoint_import"
        ):
            return (
                f"{prefix} · Legacy model u{self.checkpoint_update} · "
                "Compatible speed-input import"
            )
        if initialization is not None and self.checkpoint_agent_frames == 0:
            if initialization.get("kind") == "transfer_learning":
                return (
                    f"{prefix} · Same-architecture transfer from {initialization['source_label']} "
                    f"u{initialization['source_update']} · Not fine-tuned"
                )
            return (
                f"{prefix} · Initialized from {initialization['source_label']} "
                f"u{initialization['source_update']} · Untrained"
            )
        if initialization is not None and initialization.get("kind") == "transfer_learning":
            return (
                f"{prefix} · update {self.checkpoint_update} · Fine-tuned from "
                f"{initialization['source_label']} u{initialization['source_update']}"
            )
        if initialization is not None:
            return (
                f"{prefix} · update {self.checkpoint_update} · Initialized from "
                f"{initialization['source_label']} u{initialization['source_update']}"
            )
        return (
            f"{prefix} · update {self.checkpoint_update} · "
            + ("Latest" if self.checkpoint_selection_mode == "auto" else "Manual")
        )

    def _deployment_weights_state(self) -> str:
        if self._archived_checkpoint_path is not None:
            return str(self._archived_checkpoint_path)
        initialization = self.weights_initialization
        if initialization is not None and self.checkpoint_agent_frames == 0:
            if initialization.get("kind") == "transfer_learning":
                return (
                    "transfer_learning:"
                    f"{initialization['source_architecture_id']}"
                    f"@u{initialization['source_update']}"
                )
        return str(self.checkpoint)

    def _actor_scenario_options(self) -> list[dict[str, object]]:
        variant = actor_variant(self._selected_scenario_architecture_id())
        return [
            {
                "architecture_id": variant.architecture_id,
                "actor_kind": variant.actor_kind,
                "display_name": variant.display_name,
                "selected": True,
            }
        ]

    def _selected_scenario_architecture_id(self) -> str:
        return _selected_scenario_architecture_id(
            self.active_actor_variant_id
        )

    def _select_actor_scenario(self, command, *, internal=False):
        if not internal:
            raise ValueError('Viewer only supports the unified ATK/DEF/TGT match')
        architecture = actor_variant(str(command['architecture_id']))
        self._set_viewer_settings({'entity_streams': _architecture_entity_streams(architecture.architecture_id),
            'entity_self_attention_layers': _architecture_entity_self_attention_layers(architecture.architecture_id)})

    def _training_metrics_run(self, metrics_path: Path) -> dict[str, object]:
        records, run_architecture_id, display_name, run_metadata = (
            _training_dashboard_lineage(
                metrics_path,
                record_cache=self._training_metrics_file_cache,
            )
        )
        root_metrics = Path(
            str(run_metadata.get("lineage_root_metrics", metrics_path))
        ).resolve()
        root_metadata_value = run_metadata.get("lineage_root_metadata")
        root_metadata = (
            dict(root_metadata_value)
            if isinstance(root_metadata_value, dict)
            else {}
        )
        run_architecture_id = run_metadata.get(
            "lineage_root_architecture_id", run_architecture_id
        )
        display_name = str(
            run_metadata.get("lineage_root_display_name", display_name)
        )
        source_segment_values = run_metadata.get("source_segment_metrics", [])
        if not isinstance(source_segment_values, list):
            source_segment_values = []
        source_segment_paths = tuple(
            Path(str(path)).resolve() for path in source_segment_values
        )
        try:
            source = metrics_path.relative_to(
                self.config.repository_root
            ).as_posix()
            lineage_id = root_metrics.relative_to(
                self.config.repository_root
            ).as_posix()
        except ValueError:
            source = str(metrics_path)
            lineage_id = str(root_metrics)
        source_segment_ids: list[str] = []
        for source_path in source_segment_paths:
            try:
                source_segment_ids.append(
                    source_path.relative_to(
                        self.config.repository_root
                    ).as_posix()
                )
            except ValueError:
                source_segment_ids.append(str(source_path))
        started_at = _training_dashboard_started_at(root_metrics)
        run_started_at_ns = training_run_started_at_ns(root_metrics.parent)
        model_id = str(
            root_metadata.get("training_entry_id")
            or training_model_timestamp_id(run_started_at_ns)
        )
        training_mode = root_metadata.get("training_mode")
        mode_label = {
            "joint": "Joint",
            "alternating": "Alternating",
        }.get(str(training_mode))
        source_update = root_metadata.get("source_update")
        training_parameters_value = root_metadata.get("training_parameters", {})
        training_parameters = (
            dict(training_parameters_value)
            if isinstance(training_parameters_value, dict)
            else {}
        )
        parent_training_entry_id = root_metadata.get(
            "parent_training_entry_id"
        )
        training_settings_value = root_metadata.get("training_settings", {})
        training_settings = (
            dict(training_settings_value)
            if isinstance(training_settings_value, dict)
            else {}
        )
        scene_contract_value = root_metadata.get("scene_contract", {})
        scene_contract = (
            dict(scene_contract_value)
            if isinstance(scene_contract_value, dict)
            else {}
        )
        run_manifest_value = run_metadata.get("run_manifest", {})
        run_manifest = (
            dict(run_manifest_value)
            if isinstance(run_manifest_value, dict)
            else {}
        )
        root_run_manifest_value = root_metadata.get("run_manifest", {})
        root_run_manifest = (
            dict(root_run_manifest_value)
            if isinstance(root_run_manifest_value, dict)
            else {}
        )
        directory_name = root_metrics.parent.name.removeprefix("run-")
        run_label = model_id
        label_parts = [
            started_at[:19].replace("T", " "),
            run_label,
            mode_label,
            display_name,
        ]
        if source_update is not None:
            label_parts.append(f"from u{int(source_update)}")
        reward_key = "match/attacker_return" if actor_variant(str(run_architecture_id)).actor_kind == "learner" else "match/defense_return"
        latest_reward = records[-1].get(reward_key) if records else None
        name_fields = model_name_fields(
            str(run_architecture_id),
            directory_name,
            update=(int(records[-1]["update"]) if records else None),
            reward=(None if latest_reward is None else float(latest_reward)),
            training_mode=training_mode,
            source_update=source_update,
            training_parameters=training_parameters,
            architecture_name=display_name,
            payload=root_metadata,
        )
        if parent_training_entry_id is not None:
            name_fields["parent_training_entry_id"] = str(
                parent_training_entry_id
            )
        return {
            "id": lineage_id,
            "segment_id": source,
            "source": source,
            "model_run_id": metrics_path.parent.relative_to(
                self.config.repository_root
            ).as_posix(),
            "source_segment_ids": source_segment_ids,
            "run_group_id": str(run_metadata.get("run_group_id") or source),
            "training_series_id": str(
                run_metadata.get("training_series_id") or lineage_id
            ),
            "training_series_signature": run_metadata.get(
                "training_series_signature"
            ),
            "architecture_id": run_architecture_id,
            "actor_kind": actor_variant(str(run_architecture_id)).actor_kind,
            "records": records,
            "environment_steps_supported": bool(records) and all(isinstance(record.get("environment_steps"), (int, float)) and math.isfinite(record["environment_steps"]) for record in records),
            "record_errors": self._training_metrics_file_cache.get(metrics_path.resolve(), {}).get("errors", []),
            "display_name": " · ".join(part for part in label_parts if part),
            "name_fields": name_fields,
            "started_at": started_at,
            "model_id": model_id,
            "training_entry_id": model_id,
            "run_started_at_ns": run_started_at_ns,
            "training_mode": training_mode,
            "source_update": source_update,
            "update_offset": int(run_metadata.get("update_offset", 0)),
            "parent_training_entry_id": parent_training_entry_id,
            "training_parameters": training_parameters,
            "match_settings": {**{key: training_settings.get(key) for key in ("target_mode", "terminate_on_any_casualty", "reward_objective",
        "include_remaining_time_observation", "terminate_on_defender_elimination", "viewer_apply_sampling_limit", "maximum_episode_decisions")},
                "horizon_seconds": run_manifest.get("resolved_environment_settings", {}).get("shared", {}).get("clock", {}).get("episode_duration_s")},
            "_training_settings": training_settings,
            "_training_restore": root_metadata.get("training_restore"),
            "_weights_initialization": root_metadata.get(
                "weights_initialization"
            ),
            "_scene_contract": scene_contract,
            "_run_manifest": run_manifest,
            "_root_run_manifest": root_run_manifest,
            "_source_segment_paths": source_segment_paths,
        }

    @staticmethod
    def _public_training_metrics_run(run: dict[str, object]) -> dict[str, object]:
        public = {
            key: value
            for key, value in run.items()
            if not key.startswith("_")
        }
        if "records" in public:
            public["records"] = [
                {key: value for key, value in record.items() if value is not None}
                for record in public["records"]
            ]
        return public

    def _refresh_training_history_sync(self) -> None:
        """Update the one incremental history state without serializing curves."""
        task_id = "all"
        now = time.monotonic()
        if (
            task_id != self._training_metrics_discovery_task_id
            or now - self._training_metrics_discovery_at >= 60.0
        ):
            self.actor_training.refresh_training_run_registry()
            self._training_metrics_discovery_at = now
        selected = tuple(
            (self.config.repository_root / str(run["metrics_id"])).resolve()
            for run in self.actor_training.training_run_registry()
            if run.get("task_id") in TRAINING_DASHBOARD_TASK_IDS
        )
        if (
            task_id != self._training_metrics_discovery_task_id
            or selected != self._training_metrics_run_paths
        ):
            self._training_metrics_discovery_task_id = task_id
            self._training_metrics_run_paths = selected

        signatures: list[tuple[Path, int, int, int]] = []
        for path in self._training_metrics_run_paths:
            try:
                stat = training_metrics_source(path).stat()
            except (OSError, ValueError):
                stat = None
            try:
                manifest_mtime = (path.parent / "run.json").stat().st_mtime_ns
            except FileNotFoundError:
                manifest_mtime = 0
            signatures.append((path, stat.st_mtime_ns if stat else 0, stat.st_size if stat else 0, manifest_mtime))
        signature = tuple(signatures)
        undo_entries = self._training_undo_entries()
        redo_entries = self._training_redo_entries()
        version = task_id + ":" + ";".join(
            f"{path.as_posix()}:{mtime_ns}:{size}:{manifest_mtime}"
            for path, mtime_ns, size, manifest_mtime in signature
        ) + ":undo:" + ";".join(entry.name for entry in undo_entries)
        version += ":redo:" + ";".join(entry.name for entry in redo_entries)
        cache_signature = (
            task_id,
            signature,
            tuple(entry.name for entry in undo_entries),
            tuple(entry.name for entry in redo_entries),
        )
        with self._training_metrics_cache_lock:
            if cache_signature == self._training_metrics_cache_signature:
                return

            previous_version = self._training_metrics_cache_version
            previous_visible_run_ids = self._training_metrics_visible_run_ids
            current_signatures = {
                path.resolve(): (mtime_ns, size, manifest_mtime)
                for path, mtime_ns, size, manifest_mtime in signature
            }
            changed_paths = {
                path
                for path, value in current_signatures.items()
                if self._training_metrics_signatures.get(path) != value
                or path not in self._training_metrics_runs_by_path
            }
            removed_paths = set(self._training_metrics_signatures) - set(
                current_signatures
            )
            for path in removed_paths:
                self._training_metrics_runs_by_path.pop(path, None)
                self._training_metrics_file_cache.pop(path, None)
            history_errors = getattr(self, "_training_history_errors", {})
            for path in removed_paths:
                history_errors.pop(str(path), None)
            for metrics_path in changed_paths:
                history_errors.pop(str(metrics_path), None)
                try:
                    self._training_metrics_runs_by_path[metrics_path] = (
                        self._training_metrics_run(metrics_path)
                    )
                except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                    history_errors[str(metrics_path)] = str(error)
                    # Preserve any readable last version and identify it as stale.
                    previous_run = self._training_metrics_runs_by_path.get(metrics_path)
                    if previous_run is not None:
                        previous_run["read_error"] = str(error)
                    continue
            self._training_history_errors = history_errors
            source_segments = {
                source
                for run in self._training_metrics_runs_by_path.values()
                for source in run.get("_source_segment_paths", ())
                if isinstance(source, Path)
            }
            runs = [
                run
                for path, run in self._training_metrics_runs_by_path.items()
                if path not in source_segments
            ]
            runs.sort(
                key=lambda run: training_run_sort_key(
                    int(run["run_started_at_ns"]),
                    int(run["records"][-1]["update"]) if run["records"] else 0,
                    str(run["id"]),
                ),
                reverse=True,
            )
            visible_run_ids = tuple(str(run["segment_id"]) for run in runs)
            self._training_metrics_delta_base_version = previous_version
            self._training_metrics_changed_run_ids = {
                str(self._training_metrics_runs_by_path[path]["segment_id"])
                for path in changed_paths
                if path in self._training_metrics_runs_by_path
            } | (set(visible_run_ids) - set(previous_visible_run_ids))
            self._training_metrics_removed_run_ids = (
                set(previous_visible_run_ids) - set(visible_run_ids)
            )
            self._training_metrics_cache_signature = cache_signature
            self._training_metrics_cache_version = version
            self._training_metrics_signatures = current_signatures
            self._training_metrics_visible_run_ids = visible_run_ids
            self._training_metrics_cache_payload = b""
            self._training_metrics_full_payload_version = ""

    def _read_training_metrics_sync(
        self,
        after_version: str | None = None,
        requested_task_id: str | None = None,
        *,
        refresh_history: bool = True,
    ) -> bytes | None:
        if refresh_history:
            self._refresh_training_history_sync()
        with self._training_metrics_cache_lock:
            version = self._training_metrics_cache_version
            if after_version == version:
                return None
            delta_available = bool(after_version) and (
                after_version == self._training_metrics_delta_base_version
            )
            if not delta_available and self._training_metrics_full_payload_version == version:
                return self._training_metrics_cache_payload
            runs_by_id = {
                str(run["segment_id"]): run
                for run in self._training_metrics_runs_by_path.values()
            }
            response = {
                "architecture_id": self.active_actor_variant_id,
                "task_id": "all",
                "errors": [{"source": source, "error": error} for source, error in getattr(self, "_training_history_errors", {}).items()]
                    + getattr(self.actor_training, "registry_errors", []),
                "version": version,
                "runs": [
                    self._public_training_metrics_run(runs_by_id[run_id])
                    for run_id in self._training_metrics_visible_run_ids
                    if run_id in runs_by_id and (
                        not delta_available or run_id in self._training_metrics_changed_run_ids
                    )
                ],
                "undo_available": bool(self._training_metrics_cache_signature[2]),
                "redo_available": bool(self._training_metrics_cache_signature[3]),
            }
            if delta_available:
                response.update({
                    "delta": True,
                    "base_version": after_version,
                    "run_order": list(self._training_metrics_visible_run_ids),
                    "removed_run_ids": sorted(self._training_metrics_removed_run_ids),
                })
            payload = json.dumps(
                response, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            if not delta_available:
                self._training_metrics_cache_payload = payload
                self._training_metrics_full_payload_version = version
            return payload

    def _submit_training_history_rebuild(self, kind: str) -> None:
        if getattr(self, '_closed', False):
            return
        if self._training_metrics_executor is None:
            self._training_metrics_executor = ProcessPoolExecutor(max_workers=1)
        prefix = f"_training_{kind}"
        context = (
            self.active_actor_variant_id,
            self._training_metrics_invalidation_generation,
        )
        builder = (
            _build_training_metrics_payload if kind == "metrics"
            else _build_training_lineage_payload
        )
        future = self._training_metrics_executor.submit(
            builder, str(self.config.repository_root), *context,
            *((True,) if kind == "metrics" else ()),
            passage_root=str(self.viewer_settings_path.parent / "passage"),
        )
        setattr(self, prefix + "_future", future)
        setattr(self, prefix + "_future_context", context)
        setattr(self, prefix + "_rebuild_dirty", False)
        setattr(self, prefix + "_error", None)
        setattr(self, prefix + "_next_rebuild_at", time.monotonic() + 5.0)

    def _submit_training_metrics_rebuild(self) -> None:
        self._submit_training_history_rebuild("metrics")

    def _submit_training_lineage_rebuild(self) -> None:
        self._submit_training_history_rebuild("lineage")

    def _read_training_history(
        self,
        kind: str,
        after_version: str | None,
        *,
        ready_only: bool,
        response_state: dict[str, object] | None,
    ) -> bytes | None:
        # Only actual invalidations dirty a pending job. Collection never starts
        # another routine job, even when the completed job took longer than 5 s.
        prefix = f"_training_{kind}"
        with self._training_metrics_request_lock:
            if getattr(self, '_closed', False):
                return None
            now = time.monotonic()
            context = (
                self.active_actor_variant_id,
                self._training_metrics_invalidation_generation,
            )
            future = getattr(self, prefix + "_future")
            collected = future is not None and future.done()
            if collected:
                setattr(self, prefix + "_future", None)
                try:
                    version, payload = future.result()
                except (OSError, RuntimeError, TypeError, ValueError) as error:
                    setattr(self, prefix + "_error", str(error))
                    setattr(self, prefix + "_rebuild_dirty", True)
                    if isinstance(error, BrokenProcessPool):
                        self._training_metrics_executor.shutdown(
                            wait=False, cancel_futures=True,
                        )
                        self._training_metrics_executor = None
                else:
                    if getattr(self, prefix + "_future_context") == context:
                        if isinstance(payload, tuple):
                            payload, base_version, delta_payload = payload
                            setattr(self, prefix + '_delta_response', (base_version, delta_payload))
                        else:
                            setattr(self, prefix + '_delta_response', None)
                        setattr(self, prefix + "_version", version)
                        setattr(self, prefix + "_payload", payload)
                        setattr(self, prefix + "_payload_context", context)
                        setattr(self, prefix + "_rebuild_dirty", False)
                        setattr(self, prefix + "_error", None)
                    else:
                        setattr(self, prefix + "_rebuild_dirty", True)
                setattr(self, prefix + "_next_rebuild_at", now + 5.0)

            payload = getattr(self, prefix + "_payload")
            if payload is not None and getattr(self, prefix + "_payload_context", None) != context:
                setattr(self, prefix + "_rebuild_dirty", True)
            dirty = getattr(self, prefix + "_rebuild_dirty")
            error = getattr(self, prefix + "_error", None)
            due = now >= getattr(self, prefix + "_next_rebuild_at")
            if (
                not ready_only
                and getattr(self, prefix + "_future") is None
                and (not error or due)
                and (dirty or (not collected and due))
            ):
                try:
                    self._submit_training_history_rebuild(kind)
                except (BrokenProcessPool, OSError, RuntimeError) as error:
                    if self._training_metrics_executor is not None:
                        self._training_metrics_executor.shutdown(
                            wait=False, cancel_futures=True,
                        )
                    self._training_metrics_executor = None
                    setattr(self, prefix + "_error", str(error))
                    setattr(self, prefix + "_rebuild_dirty", True)
                    setattr(self, prefix + "_next_rebuild_at", now + 5.0)

            pending = (
                "building" if getattr(self, prefix + "_future") is not None
                else "invalidated" if getattr(self, prefix + "_rebuild_dirty") else ""
            )
            if response_state is not None:
                response_state.update({
                    "pending": pending,
                    "generation": context[1],
                    "error": getattr(self, prefix + "_error", None),
                })
            # Keep the client's last valid content while a new generation builds.
            # Never deliver a pre-invalidation payload as the new generation.
            if getattr(self, prefix + "_payload_context", None) != context:
                return None
            if after_version == getattr(self, prefix + "_version"):
                return None
            delta = getattr(self, prefix + '_delta_response', None)
            if delta is not None and after_version == delta[0]:
                return delta[1]
            return getattr(self, prefix + "_payload")

    def _read_training_metrics(
        self,
        after_version: str | None = None,
        requested_task_id: str | None = None,
        *,
        ready_only: bool = False,
        response_state: dict[str, object] | None = None,
    ) -> bytes | None:
        if not hasattr(self, "_training_metrics_request_lock"):
            return self._read_training_metrics_sync(after_version, requested_task_id)
        return self._read_training_history(
            "metrics", after_version, ready_only=ready_only, response_state=response_state,
        )

    def _read_training_lineage(
        self,
        after_version: str | None = None,
        *,
        ready_only: bool = False,
        response_state: dict[str, object] | None = None,
    ) -> bytes | None:
        if not hasattr(self, "_training_metrics_request_lock"):
            version, payload = _build_training_lineage_payload(
                str(self.config.repository_root), self.active_actor_variant_id,
            )
            return None if after_version == version else payload
        return self._read_training_history(
            "lineage", after_version, ready_only=ready_only, response_state=response_state,
        )

    def _invalidate_training_metrics(self) -> None:
        with self._training_metrics_request_lock:
            self._training_metrics_invalidation_generation += 1
            for kind in ("metrics", "lineage"):
                setattr(self, f"_training_{kind}_rebuild_dirty", True)
                setattr(self, f"_training_{kind}_error", None)
        self._mark_snapshot_status_dirty()

    def _training_undo_entries(self) -> list[Path]:
        undo_root = (
            self.config.repository_root
            / "Artifacts"
            / "GoalReach"
            / TRAINING_UNDO_DIRECTORY
        )
        if not undo_root.is_dir():
            return []
        return sorted(
            entry for entry in undo_root.iterdir()
            if entry.is_dir() and (entry / "manifest.json").is_file()
        )

    def _training_redo_entries(self) -> list[Path]:
        redo_root = (
            self.config.repository_root
            / "Artifacts"
            / "GoalReach"
            / TRAINING_REDO_DIRECTORY
        )
        if not redo_root.is_dir():
            return []
        return sorted(
            entry for entry in redo_root.iterdir()
            if entry.is_dir() and (entry / "manifest.json").is_file()
        )

    def _clear_training_redo(self) -> None:
        redo_root = (
            self.config.repository_root
            / "Artifacts"
            / "GoalReach"
            / TRAINING_REDO_DIRECTORY
        )
        if redo_root.is_dir():
            shutil.rmtree(redo_root)

    def _delete_training_segment(
        self,
        command: dict[str, object],
        *,
        refresh_status: bool = True,
        clear_redo: bool = True,
        invalidate_history: bool = True,
    ) -> None:
        source = str(command.get("source", "")).strip()
        if not source:
            raise ValueError("Training segment source is required")
        if source.startswith('passage:'):
            if self.passage is None:
                raise ValueError('Select the discrete scene first')
            undo_root = (self.config.repository_root / 'Artifacts' / 'GoalReach'
                         / TRAINING_UNDO_DIRECTORY)
            entry = undo_root / f'{time.time_ns()}-passage'
            entry.mkdir(parents=True)
            try:
                self.passage.move_training_group(entry, source.removeprefix('passage:'))
            except Exception:
                if (entry / 'segment').is_dir() and not any((entry / 'segment').iterdir()):
                    (entry / 'segment').rmdir()
                if not any(entry.iterdir()):
                    entry.rmdir()
                raise
            if clear_redo:
                self._clear_training_redo()
            for expired in self._training_undo_entries()[:-TRAINING_UNDO_LIMIT]:
                shutil.rmtree(expired)
            self._invalidate_training_metrics()
            return
        artifact_root = (
            self.config.repository_root / "Artifacts" / "GoalReach"
        ).resolve()
        target = (self.config.repository_root / source).resolve()
        if target.name != "metrics.jsonl" or not target.is_relative_to(artifact_root):
            raise ValueError("Training segment source is outside GoalReach artifacts")
        undo_root = (artifact_root / TRAINING_UNDO_DIRECTORY).resolve()
        if target.is_relative_to(undo_root):
            raise ValueError("Training segment is already deleted")
        manifest_path = target.parent / "run.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        if manifest.get("checkpoint_schema") != "mobile_local_sensing_group":
            if not target.is_file():
                return
            with target.open("r", encoding="utf-8") as stream:
                identity = next((json.loads(line) for line in stream if line.strip()), {})
            if identity.get("train/task_id") not in TRAINING_DASHBOARD_TASK_IDS:
                raise ValueError("Training segment task is unsupported")

        segment = target.parent.resolve()
        original_segment = segment

        primary_checkpoint = (
            self._archived_checkpoint_path
            if self.checkpoint_selection_mode == "archived"
            else self.checkpoint
        )
        deletes_primary_checkpoint = bool(
            self._active_checkpoint_loaded
            and primary_checkpoint is not None
            and primary_checkpoint.resolve().is_relative_to(segment)
        )
        deletes_opponent_checkpoint = bool(
            self.opponent_checkpoint is not None
            and self.opponent_checkpoint.resolve().is_relative_to(segment)
        )
        target_checkpoint = getattr(self, "target_runtime", {}).get("checkpoint")
        deletes_target_checkpoint = bool(
            target_checkpoint is not None and target_checkpoint.resolve().is_relative_to(segment)
        )
        status = self.actor_training.refresh()
        for run in status.get("all_runs", []):
            output_value = run.get("output") if isinstance(run, dict) else None
            if isinstance(output_value, str) and Path(output_value).resolve() == segment:
                raise ValueError("Stop training before deleting this segment")
        for training_session in status.get("grouped_sessions", []):
            for source in training_session.get("sources", {}).values():
                if source and Path(str(source)).resolve().is_relative_to(segment):
                    raise ValueError("Pause the training session using this model before deleting it")

        current_outputs = {
            (
                self.config.repository_root / actor_variant(architecture_id).output_directory
            ).resolve()
            for architecture_id in actor_variant_ids()
        }
        if segment in current_outputs or (segment / TRAINING_ARCHIVE_NAME).is_dir():
            relocated = archive_actor_training_run(segment, target)
            if relocated is None:
                raise ValueError("Training segment has no saved data")
            target = relocated.resolve()
            segment = target.parent

        if clear_redo:
            self._clear_training_redo()
        undo_root.mkdir(parents=True, exist_ok=True)
        entry = undo_root / f"{time.time_ns()}-{segment.name}"
        entry.mkdir()
        restore_path = segment.relative_to(self.config.repository_root.resolve())
        segment.replace(entry / "segment")
        (entry / "manifest.json").write_text(
            json.dumps(
                {"restore_path": restore_path.as_posix()},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        entries = self._training_undo_entries()
        for expired in entries[:-TRAINING_UNDO_LIMIT]:
            shutil.rmtree(expired)
        if deletes_primary_checkpoint:
            self._select_random_current_checkpoint(
                forget_saved_checkpoints=(
                    self.checkpoint.resolve().is_relative_to(segment)
                )
            )
        if deletes_opponent_checkpoint:
            self._select_opponent_checkpoint("random_current_architecture")
        if deletes_target_checkpoint:
            self._select_checkpoint({"actor": "target", "model_id": "random_current_architecture"})
        self.actor_training.forget_training_run(original_segment)
        self.actor_training.forget_training_run(segment)
        self.actor_training.invalidate_checkpoint_option_caches()
        self._opponent_option_cache = None
        if refresh_status:
            self._refresh_actor_training_status(refresh_source_options=True)
        if invalidate_history:
            self._invalidate_training_metrics()

    def _undo_training_segment_delete(self) -> None:
        entries = self._training_undo_entries()
        if not entries:
            raise ValueError("No deleted training segment to restore")
        entry = entries[-1]
        manifest = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get('scene_mode') == 'discrete':
            if self.passage is None:
                from .passage_viewer import PassageViewer
                self.passage = PassageViewer(self.viewer_settings_path.parent,
                    checkpoint_executor=self._checkpoint_load_executor)
            self.passage.move_training_group(entry, manifest['name'], restore=True)
            redo_entry = (self.config.repository_root / 'Artifacts' / 'GoalReach'
                / TRAINING_REDO_DIRECTORY / entry.name)
            redo_entry.mkdir(parents=True)
            (entry / 'manifest.json').replace(redo_entry / 'manifest.json')
            (entry / 'segment').rmdir()
            entry.rmdir()
            for expired in self._training_redo_entries()[:-TRAINING_UNDO_LIMIT]:
                shutil.rmtree(expired)
            self._invalidate_training_metrics()
            return
        restore_path = (
            self.config.repository_root / str(manifest["restore_path"])
        ).resolve()
        artifact_root = (
            self.config.repository_root / "Artifacts" / "GoalReach"
        ).resolve()
        if not restore_path.is_relative_to(artifact_root):
            raise ValueError("Undo target is outside GoalReach artifacts")
        if restore_path.exists():
            raise ValueError("The original segment path is occupied")
        restore_path.parent.mkdir(parents=True, exist_ok=True)
        (entry / "segment").replace(restore_path)
        redo_root = (artifact_root / TRAINING_REDO_DIRECTORY).resolve()
        redo_root.mkdir(parents=True, exist_ok=True)
        redo_entry = redo_root / f"{time.time_ns()}-{restore_path.name}"
        redo_entry.mkdir()
        (redo_entry / "manifest.json").write_text(
            json.dumps(
                {"restore_path": restore_path.relative_to(
                    self.config.repository_root.resolve()
                ).as_posix()},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        for expired in self._training_redo_entries()[:-TRAINING_UNDO_LIMIT]:
            shutil.rmtree(expired)
        (entry / "manifest.json").unlink()
        entry.rmdir()
        self.actor_training.register_training_run(restore_path)
        self._invalidate_training_metrics()
        self.actor_training.invalidate_checkpoint_option_caches()
        self._opponent_option_cache = None
        self._refresh_actor_training_status(refresh_source_options=True)

    def _redo_training_segment_delete(self) -> None:
        entries = self._training_redo_entries()
        if not entries:
            raise ValueError("No restored training segment to delete again")
        entry = entries[-1]
        manifest = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get('scene_mode') == 'discrete':
            self._delete_training_segment({'source':'passage:'+manifest['name']}, clear_redo=False)
            (entry / 'manifest.json').unlink()
            entry.rmdir()
            return
        restore_path = (
            self.config.repository_root / str(manifest["restore_path"])
        ).resolve()
        artifact_root = (
            self.config.repository_root / "Artifacts" / "GoalReach"
        ).resolve()
        if not restore_path.is_relative_to(artifact_root):
            raise ValueError("Redo target is outside GoalReach artifacts")
        metrics = restore_path / "metrics.jsonl"
        if not metrics.is_file() and not (restore_path / "run.json").is_file():
            raise ValueError("The restored training segment is unavailable")
        self._delete_training_segment(
            {
                "source": metrics.relative_to(
                    self.config.repository_root
                ).as_posix()
            },
            clear_redo=False,
            invalidate_history=False,
        )
        (entry / "manifest.json").unlink()
        entry.rmdir()
        self._invalidate_training_metrics()

    def _prepare_follow_bundle(self, checkpoint, manifest, generations):
        require_active_training_checkpoint(checkpoint)
        sources = {}
        locator = "saved_target:" + checkpoint.relative_to(self.config.repository_root).as_posix()
        # Resolve the mutable alias once. A publication between role reads must
        # not put different releases into one supposedly atomic deployment.
        release, _, payload, _ = _read_group_release(checkpoint, sources)
        result = [self._prepare_viewer_deployment(
            generation, role, locator, manifest["architecture_id"], "checkpoint", None,
            release, "manual", None, True, True, prepared_sources=sources)
            for role, generation in generations.items()]
        if any(item.payload["run_group_id"] != manifest["run_group_id"]
               or item.payload['global_update'] != payload['global_update'] for item in result):
            raise ValueError("Follow group/release identity mismatch")
        return result

    def _follow_training_checkpoints(
        self, status: dict[str, object], *, discover: bool = True,
    ) -> None:
        if self.scene_mode != 'continuous':
            return
        if getattr(self, '_replay_record', None) is not None or getattr(self, '_requested_viewer_settings', None) is not None:
            return
        if getattr(self, '_deployment_futures', None) or getattr(self, '_pending_deployments', None):
            return
        group_id = self._auto_deploy_training_run_group_id
        future = getattr(self, "_follow_bundle_future", None)
        if future is not None:
            if not future.done() or (self.running and self._physics_ticks_into_decision != 0):
                return
            self._follow_bundle_future = None
            try:
                bundle = future.result()
                # One shared checkpoint is installed between physics ticks and snapshots.
                if group_id == self._follow_bundle_group:
                    if set(self._auto_deploy_training_sides) != set(self.env.roles) or any(
                        item.generation != self._deployment_generations[item.actor_kind] for item in bundle
                    ):
                        self._follow_bundle_signature = None
                        return
                    if {item.actor_kind for item in bundle} != set(self.env.roles):
                        raise ValueError("Follow checkpoint is missing an active role")
                    self._install_viewer_deployments(bundle)
                    self.checkpoint_refresh_error = None
            except Exception as error:
                self.checkpoint_refresh_error = str(error)
                for role in self._auto_deploy_training_sides:
                    self._deployment_errors[role] = str(error)
            return
        if not discover or not group_id or not self._auto_deploy_training_sides:
            return
        for run in self.actor_training.training_run_registry():
            manifest = run.get("manifest") or {}
            if manifest.get("run_group_id") != group_id or not manifest.get("role_bindings"):
                continue
            if run.get('archived'):
                self._discard_training_follow_side('learner')
                self.checkpoint_refresh_error = 'Archived training group cannot Follow'
                return
            if set(policy_roles(manifest["launch_settings"])) != set(self.env.roles):
                self.checkpoint_refresh_error = "Follow session roles differ from the active Viewer scene"
                return
            checkpoint = Path(run["output"]) / "checkpoint-latest.pt"
            if not checkpoint.is_file():
                return
            # The asynchronous status reader refreshes its own registry. Its
            # published update also catches the manifest write after latest.pt.
            published = next((g for g in status.get('groups', []) if g.get('run_group_id') == group_id), {})
            signature = (group_id, published.get('checkpoint_update', manifest.get('checkpoint_update')),
                         checkpoint.stat().st_mtime_ns, tuple(sorted(self._auto_deploy_training_sides)))
            if signature == getattr(self, "_follow_bundle_signature", None):
                return
            generations = {}
            for role in self._auto_deploy_training_sides:
                self._deployment_generations[role] += 1
                generations[role] = self._deployment_generations[role]
            for role in generations:
                self._deployment_errors.pop(role, None)
                self._deployment_client_request_ids.pop(role, None)
            self._follow_bundle_group = group_id
            self._follow_bundle_signature = signature
            self._follow_bundle_future = self._checkpoint_load_executor.submit(self._prepare_follow_bundle, checkpoint, manifest, generations)
            return

    def _training_opponent_options(
        self, actor_kind: str
    ) -> list[dict[str, object]]:
        candidates = self._actor_training_status_cache.get("role_model_options", {}).get(actor_kind, [])
        options: list[dict[str, object]] = []
        seen: set[str] = set()
        for option in candidates:
            model_id = str(option.get("model_id", ""))
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            options.append({**option, "id": model_id})
        return options

    def _resolve_model_reference(
        self,
        actor_kind: str,
        model_id: str,
        *,
        options: object = None,
    ) -> tuple[Path | str, str]:
        candidates = (
            options
            if isinstance(options, list)
            else self._training_opponent_options(actor_kind)
        )
        if model_id == "random_current_architecture":
            return model_id, _unified_match_architecture_id(
                self.viewer_settings, actor_kind
            )
        if model_id.startswith('saved_target:'):
            path = (self.config.repository_root / model_id.removeprefix('saved_target:')).resolve()
            path.relative_to((self.config.repository_root / 'Artifacts' / 'GoalReach').resolve())
            require_active_training_checkpoint(path)
            if not path.is_file():
                raise ValueError('Selected U unavailable: checkpoint is missing')
            manifest_path=path.parent/'run.json'
            manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
            architecture=manifest.get('architecture_id')
            if not actor_variant(architecture).viewer_enabled:
                raise ValueError('Selected group has no valid architecture metadata')
            return path, architecture
        option = next(
            (
                candidate
                for candidate in candidates
                if str(candidate["id"]) == model_id
                or str(candidate.get("model_timestamp_id", "")) == model_id
            ),
            None,
        )
        if option is None:
            raise ValueError("Model must be selected from Model Library")
        architecture_id = str(option.get("source_architecture_id", ""))
        if not actor_variant(architecture_id).unified_match_contract and actor_variant(architecture_id).actor_kind != actor_kind:
            raise ValueError("Model role does not match the selected side")
        locator = str(option.get("model_id") or option.get("id") or "")
        if locator == "random_current_architecture":
            source = locator
        elif locator.startswith("saved_target:"):
            source = self.actor_training._resolve_model_checkpoint(locator.removeprefix("saved_target:"))
        elif locator.startswith("archived_target:"):
            source = self.actor_training.resolve_archived_checkpoint(
                architecture_id,
                locator.removeprefix("archived_target:"),
            )
        else:
            raise ValueError("Model must be a saved checkpoint")
        if isinstance(source, Path):
            require_active_training_checkpoint(source)
        return source, architecture_id

    def _model_entry_checkpoint(self, actor_kind, model_id):
        source, architecture = self._resolve_model_reference(actor_kind, model_id)
        if not isinstance(source, Path):
            raise ValueError('Random has no saved checkpoint')
        source, payload = self.actor_training.load_deployment_checkpoint(architecture,
            source.relative_to(self.config.repository_root).as_posix(), role=actor_kind, map_location=self.device)
        return source, payload, architecture

    def _role_deployment_descriptor(self, role, *, borrow_source=False, neural_state=False):
        if role not in self.env.policy_roles:
            return {"actor_kind":role,"label":ROLE_DESCRIPTORS[role]["label"],"mode":"static","loaded":False,"pending":False,"model_id":None,"update":None}
        if not neural_state and role == 'learner' and self.viewer_settings['learner_control_mode'] == 'rule':
            return dict(actor_kind=role, label='ATK Rule (APF)', mode='rule', control_mode='rule',
                loaded=True, pending=False, following=False, error=None, model_id=None, selected_model_id=None,
                update=None, checkpoint=None, actor_hash=None, source_metadata=None,
                source='Rule (APF)', sampling='not_applicable', rule=copy.deepcopy(self.viewer_settings['learner_rule']),
                architecture_id=self.env.role_variant(role).architecture_id,
                preserved_nn_model_id=self._runtime_deployment_model_id(role),
                request_id=self._deployment_generations[role])
        active = self.active_actor_kind
        if role == "target":
            runtime = self.target_runtime
            descriptor = {key: value for key, value in runtime.items() if key not in {"model", "mtime_ns"}}
        elif role == active:
            descriptor = {"architecture_id": self.active_actor_variant_id, "update": self.checkpoint_update,
                          "agent_frames": self.checkpoint_agent_frames, "checkpoint": self.checkpoint if self._active_checkpoint_loaded else None,
                          "loaded": self.model is not None}
        else:
            descriptor = {"architecture_id": self.opponent_architecture_id, "update": self.opponent_update,
                          "agent_frames": self.opponent_agent_frames, "checkpoint": self.opponent_checkpoint,
                          "loaded": self.opponent_model is not None}
        control = self.role_controls[role]
        model_id = self._runtime_deployment_model_id(role)
        descriptor.update(actor_kind=role, label=ROLE_DESCRIPTORS[role]["label"],
                          model_id=model_id,
                          selected_model_id=getattr(self,"_deployment_requested_models",{}).get(role,model_id),
                          sampling="stochastic" if control.stochastic_neural else "deterministic",
                          control_mode=self.viewer_settings[role + "_control_mode"],
                          pending=role in self._deployment_futures or role in self._pending_deployments or (role in self._auto_deploy_training_sides and getattr(self, "_follow_bundle_future", None) is not None),
                          request_id=self._deployment_generations[role],
                          client_request_id=self._deployment_client_request_ids.get(role),
                          error=self._deployment_errors.get(role),
                          following=role in self._auto_deploy_training_sides)
        descriptor["checkpoint"] = str(descriptor["checkpoint"]) if descriptor.get("checkpoint") else None
        source = self._deployed_sources.get(role)
        descriptor['source_metadata'] = source if borrow_source else copy.deepcopy(source)
        rules = (descriptor['source_metadata'] or {}).get('training_rules', {})
        descriptor['scene_differences'] = {key:{'scene':self.viewer_settings.get(key),'training':value}
                                           for key,value in rules.items() if key in VIEWER_ENVIRONMENT_SETTING_KEYS
                                           and self.viewer_settings.get(key) != value}
        sources = [self._deployed_sources.get(r) for r in self.env.policy_roles]
        descriptor['mixed_sources'] = len({(source.get('run_group_id'),source.get('source_weight_update'))
                                            if source else ('Random',0) for source in sources}) > 1
        descriptor['checkpoint_global_update'] = (descriptor['source_metadata'] or {}).get('checkpoint_global_update')
        descriptor['actor_hash'] = getattr(control.neural_policy.model, '_viewer_actor_hash', None)
        descriptor["source"] = descriptor["checkpoint"] or "Random"
        if not descriptor.get('source_metadata'):
            model = control.neural_policy.model
            if not hasattr(model, '_viewer_source_identity'):
                self._viewer_model_sequence = getattr(self, '_viewer_model_sequence', 0) + 1
                model._viewer_source_identity = f'{self._viewer_policy_session_seed}:{self._viewer_model_sequence}'
            descriptor['random_instance'] = model._viewer_source_identity
        return descriptor

    def _runtime_deployment_model_id(self, actor_kind: str) -> str:
        if actor_kind == "target" and hasattr(self, "target_runtime"):
            return self.target_runtime["model_id"]
        active_kind = self.active_actor_kind
        if actor_kind == active_kind:
            if self.checkpoint_selection_mode == "random" or not self._active_checkpoint_loaded:
                return "random_current_architecture"
            if self._checkpoint_source_model_id:
                return str(self._checkpoint_source_model_id)
            try:
                relative = self.checkpoint.resolve().relative_to(
                    self.config.repository_root.resolve()
                )
            except ValueError:
                return str(
                    self.viewer_settings["deployment_model_ids"].get(actor_kind) or ""
                )
            return "saved_target:" + relative.as_posix()
        if actor_kind == self.opponent_actor_kind:
            return str(self.opponent_model_id or "")
        return str(self.viewer_settings["deployment_model_ids"].get(actor_kind) or "")

    def _runtime_deployment_model_reference(self, actor_kind: str) -> str:
        source = self._deployed_sources.get(actor_kind)
        if source is None:
            return 'random_current_architecture'
        path = Path(source['checkpoint_path'])
        if not path.is_file():
            raise ValueError(f"Selected U{source['checkpoint_global_update']} unavailable")
        return 'saved_target:' + path.relative_to(self.config.repository_root).as_posix()

    def _resolve_evaluation_model_reference(self, actor_kind: str, model_reference: str) -> tuple[str, bool]:
        if model_reference == 'random_current_architecture':
            return model_reference, False
        self._resolve_model_reference(actor_kind, model_reference)
        return model_reference, False

    def _matching_model_option(
        self,
        options: list[dict[str, object]],
        model_reference: str,
        *,
        id_prefix: str = "",
    ) -> dict[str, object] | None:
        if not model_reference:
            return None
        candidates = [
            option
            for option in options
            if not id_prefix or str(option.get("id", "")).startswith(id_prefix)
        ]
        exact = next(
            (
                option
                for option in candidates
                if str(option.get("id", "")) == model_reference
            ),
            None,
        )
        if exact is not None:
            return exact
        canonical = self.actor_training.canonical_model_reference(model_reference)
        return next(
            (
                option
                for option in candidates
                if str(option.get("model_timestamp_id", "")) == canonical
                or self.actor_training.canonical_model_reference(
                    option.get("id", "")
                )
                == canonical
            ),
            None,
        )

    def _matching_training_model_option(
        self,
        actor_kind: str,
        model_reference: str,
    ) -> dict[str, object] | None:
        return self._matching_model_option(
            self._training_opponent_options(actor_kind), model_reference
        )

    def _prepare_training_scene(self, actor_kind, model_id, current_settings,
                                current_architecture_id, episode_index,
                                current_opponent_model_id, *, use_current_environment=False, expected_sources=None):
        del current_architecture_id, current_opponent_model_id
        settings = copy.deepcopy(current_settings)
        sources = {}
        selected_source = None
        if not use_current_environment and model_id != "random_current_architecture":
            selected_source, _ = self._resolve_model_reference(actor_kind, model_id)
            selected_source = selected_source.resolve()
            selected_source, mtime, payload, _ = _read_group_release(selected_source, sources)
            settings = _checkpoint_viewer_settings(payload, payload["architecture_id"])
            settings["evaluation_scene_preset"] = current_settings.get("evaluation_scene_preset")
        elif not use_current_environment:
            settings["deployment_model_ids"][actor_kind] = model_id

        deployments = []
        for role in policy_roles(settings):
            role_model_id = settings["deployment_model_ids"].get(role) or "random_current_architecture"
            architecture = _unified_match_architecture_id(settings, role)
            if role == 'learner' and settings['learner_control_mode'] == 'rule' and selected_source is None:
                deployments.append(self._prepare_rule_deployment(self._deployment_generations[role], architecture))
                continue
            if selected_source is not None:
                path = selected_source
                role_model_id = "saved_target:" + path.relative_to(self.config.repository_root).as_posix()
                architecture = payload["architecture_id"]
            elif role_model_id == "random_current_architecture":
                path = None
            else:
                path, architecture = self._resolve_model_reference(role, role_model_id)
            deployments.append(self._prepare_viewer_deployment(
                self._deployment_generations[role], role, role_model_id, architecture,
                "checkpoint" if path is not None else "random", None, path,
                "manual" if path is not None else "random", None, False, False,
                prepared_sources=sources))
            expected = (expected_sources or {}).get(role)
            if expected and deployments[-1].payload['source_metadata'] != expected:
                raise ValueError(f"Selected U{expected['checkpoint_global_update']} unavailable")
            if settings[role + "_control_mode"] != "rule":
                settings["deployment_model_ids"][role] = role_model_id
            if settings[role + '_control_mode'] != 'rule':
                settings[role + '_control_mode'] = 'nn'

        by_role = {item.actor_kind: item for item in deployments}
        architecture = by_role["learner"].architecture_id
        config, environment = self._new_environment(architecture, settings=settings)
        environment.set_patrol_architecture(by_role["defender"].architecture_id)
        if "target" in by_role:
            environment.target_variant = actor_variant(by_role["target"].architecture_id)
        self._reset_viewer_environment(environment, episode_index=episode_index)
        return {"actor_kind": actor_kind, "architecture_id": architecture,
                "settings": settings, "config": config, "environment": environment,
                "deployments": deployments, "warning": None}

    def _queue_training_scene_load(self, prepare, *args, preserve_follow=False) -> None:
        receipt = getattr(self, "_viewer_settings_result", None)
        if receipt and receipt["phase"] == "preparing":
            receipt["phase"] = "superseded"
        # A scene request supersedes earlier Apply/Deploy requests. Later requests
        # invalidate this snapshot before the prepared scene can touch live state.
        self._invalidate_environment_updates()
        if not preserve_follow:
            self._discard_training_follow_side("learner")
        for role in self._deployment_generations:
            self._deployment_generations[role] += 1
        for _, future in self._deployment_futures.values():
            future.cancel()
        self._deployment_futures.clear()
        self._pending_deployments.clear()
        for name in ("_checkpoint_load_future", "_initial_checkpoint_load_future", "_opponent_checkpoint_load_future"):
            future = getattr(self, name, None)
            if future is not None:
                future.cancel()
            setattr(self, name, None)
        self._pending_checkpoint_load = None
        self._pending_initial_checkpoint_load = None
        self._pending_opponent_checkpoint_load = None
        self._pending_checkpoint_source_reset = False
        self._training_scene_request = (
            self._environment_update_generation, dict(self._deployment_generations),
            copy.deepcopy(self.viewer_settings))
        self._training_scene_load_future = self._checkpoint_load_executor.submit(prepare, *args) if prepare is not None else None

    def _load_training_scene(self, command: dict[str, object]) -> None:
        actor_kind = str(command.get("actor_kind", ""))
        model_id = str(command.get("model_id", ""))
        if actor_kind not in {"learner", "defender", "target"}:
            raise ValueError("Training scene Actor side is invalid")
        if model_id != "random_current_architecture" and not model_id.startswith(
            "saved_target:"
        ):
            raise ValueError("Training scene requires a saved full checkpoint")
        if self._training_scene_load_future is not None:
            raise ValueError("Training environment is loading")
        opponent_kind = "defender" if actor_kind == "learner" else "learner"
        self._queue_training_scene_load(
            self._prepare_training_scene,
            actor_kind,
            model_id,
            copy.deepcopy(self.viewer_settings),
            self.active_actor_variant_id,
            int(self.env.episode_counters[0]) + 1,
            self._runtime_deployment_model_id(opponent_kind),
        )

    def _save_evaluation_scene(self) -> None:
        settings = copy.deepcopy(self.viewer_settings)
        settings.pop("evaluation_scene_preset", None)
        settings["deployment_model_ids"] = {
            actor_kind: self._runtime_deployment_model_id(actor_kind)
            for actor_kind in active_roles(settings)
        }
        preset = {
            "viewer_settings": settings,
            "source_metadata": copy.deepcopy(self._deployed_sources),
            "deployment_model_references": {
                actor_kind: self._runtime_deployment_model_reference(actor_kind)
                for actor_kind in active_roles(settings)
            },
        }
        preset["edited_scene"] = self._capture_edited_scene() if self._edited_scene else None
        self.viewer_settings["evaluation_scene_preset"] = preset
        _store_viewer_settings(self.viewer_settings_path, self.viewer_settings)
        self._mark_snapshot_static_dirty()

    def _prepare_evaluation_scene(
        self,
        actor_kind: str,
        model_id: str,
        settings: dict[str, object],
        current_architecture_id: str,
        episode_index: int,
        opponent_model_id: str,
        warning: str | None,
    ) -> dict[str, object]:
        scene = self._prepare_training_scene(
            actor_kind,
            model_id,
            settings,
            current_architecture_id,
            episode_index,
            opponent_model_id,
            use_current_environment=True,
            expected_sources=(settings.get('evaluation_scene_preset') or {}).get('source_metadata'),
        )
        scene["evaluation_scene_preset"] = True
        scene["edited_scene"] = copy.deepcopy((settings.get("evaluation_scene_preset") or {}).get("edited_scene"))
        if scene["edited_scene"]:
            self._apply_preserved_edited_scene(scene["edited_scene"], env=scene["environment"])
        if warning is not None:
            scene["warning"] = warning
        return scene

    def _load_evaluation_scene(self) -> None:
        if self._training_scene_load_future is not None:
            raise ValueError("Viewer environment is loading")
        preset = self.viewer_settings.get("evaluation_scene_preset")
        if not isinstance(preset, dict):
            raise ValueError("No evaluation scene has been saved")
        stored_settings = preset.get("viewer_settings")
        stored_references = preset.get("deployment_model_references")
        if not isinstance(stored_settings, dict) or not isinstance(
            stored_references, dict
        ):
            raise ValueError("The saved evaluation scene is invalid")

        settings_input = copy.deepcopy(stored_settings)
        settings_input["evaluation_scene_preset"] = copy.deepcopy(preset)
        settings = _validated_viewer_settings(settings_input)
        resolved_models: dict[str, str] = {}
        missing: list[str] = []
        for actor_kind in active_roles(settings_input):
            model_reference = str(
                stored_references.get(actor_kind) or "random_current_architecture"
            )
            model_id, unavailable = self._resolve_evaluation_model_reference(
                actor_kind, model_reference
            )
            resolved_models[actor_kind] = model_id
            if unavailable:
                missing.append(ROLE_DESCRIPTORS[actor_kind]["label"])
        settings["deployment_model_ids"] = resolved_models

        selected_architecture_id = str(settings["active_actor_variant_id"])
        active_architecture_id = _configured_architecture_id(
            settings, selected_architecture_id
        )
        actor_kind = self.active_actor_kind
        opponent_kind = "defender" if actor_kind == "learner" else "learner"
        warning = (
            "Saved " + "/".join(missing) + " model is unavailable; Random is used"
            if missing
            else None
        )
        self._queue_training_scene_load(
            self._prepare_evaluation_scene,
            actor_kind,
            resolved_models[actor_kind],
            settings,
            self.active_actor_variant_id,
            int(self.env.episode_counters[0]) + 1,
            resolved_models[opponent_kind],
            warning,
        )

    def _apply_prepared_training_scene(self, scene: dict[str, object]) -> None:
        deployments = scene["deployments"]
        if any(item.generation != self._deployment_generations[item.actor_kind] for item in deployments):
            return
        for item in deployments:
            if item.checkpoint is not None and (
                not item.checkpoint.is_file()
                or item.checkpoint.stat().st_mtime_ns != item.checkpoint_mtime_ns
            ):
                raise ValueError("Selected checkpoint changed while preparing scene")
        for item in deployments:
            self._validate_deployment_task(item, scene['settings'])
        by_role = {item.actor_kind: item for item in deployments}
        # Keep the old environment and controls untouched until all preparation
        # succeeds; a failed installation restores their original references.
        previous = dict(self.__dict__)
        reset_state = self._capture_environment_runtime()
        self._auto_deploy_training_sides = set(self._auto_deploy_training_sides)
        self._deployment_errors = dict(self._deployment_errors)
        try:
            self.viewer_settings = scene["settings"]
            if scene.get('settings_request'):
                self._auto_deploy_training_sides.clear()
                self._auto_deploy_training_run_group_id = None
                self.viewer_settings['auto_deploy_training_run_group_id'] = None
                self._follow_bundle_signature = None
            if scene.get("configuration_changed"):
                self._clear_scene_lock()
            self.config, self.env = scene["config"], scene["environment"]
            self.env.performance_recorder = self._record_performance_sample
            self.active_actor_variant_id = str(scene["architecture_id"])
            self.selected_actor_kind = str(scene["actor_kind"])
            self.selected_actor_index = 0
            self.viewer_settings["active_actor_variant_id"] = _selected_scenario_architecture_id(self.active_actor_variant_id)
            self.model = by_role["learner"].model
            self.policy = _ViewerPolicy(self.model)
            self.checkpoint = (previous['checkpoint']
                if by_role['learner'].source_kind == 'rule' and self.active_actor_variant_id == previous['active_actor_variant_id']
                else self.deployment_checkpoints.setdefault(self.active_actor_variant_id, (self.config.repository_root / actor_variant(self.active_actor_variant_id).output_directory / "checkpoint-latest.pt").resolve()))
            self.metrics_path = self.checkpoint.parent / "metrics.jsonl"
            self.initial_checkpoint = self.checkpoint.parent / INITIAL_CHECKPOINT_NAME
            opponent = by_role["defender"]
            if "target" in by_role:
                target = by_role["target"]
                self.target_runtime = dict(architecture_id=target.architecture_id, model=target.model,
                    model_id=target.model_id, checkpoint=target.checkpoint,
                    update=int((target.payload or {}).get("update_index", 0)), agent_frames=0, loaded=True)
            self.actor_frames = {}
            self._initialize_actor_runtime_state(opponent_state={
                "actor_kind": "defender", "architecture_id": opponent.architecture_id,
                "model_id": opponent.model_id, "model": opponent.model,
                "checkpoint": opponent.checkpoint, "selection": opponent.selection_mode,
                "update": int((opponent.payload or {}).get("update_index", 0)), "agent_frames": 0})
            # Snapshot clients keep this revision across scene/model replacements.
            self._actor_inputs_revision = previous['_actor_inputs_revision']
            for item in deployments:
                self._apply_prepared_viewer_deployment(item, finalize=False)
                self._deployment_errors.pop(item.actor_kind, None)
            self._physics_ticks_into_decision = 0
            self._terminal_snapshot_until = 0.0
            self._terminal_environment_update_pending = False
            self._pending_target_mode = None
            self._edited_scene = bool(scene.get("edited_scene"))
            self._reset_current_learner_outcomes()
            self._reset_current_match_facts()
            self._reset_trajectories()
            self._record_spawn()
            self._remember_episode_initial_state()
            self._refresh_actor_frame()
            next_time_scale = float(self.viewer_settings["time_scale"])
            if next_time_scale != self.time_scale:
                self.rate_revision += 1
            self.time_scale = next_time_scale
            self.snapshot_rate_hz = int(self.viewer_settings["snapshot_rate_hz"])
            self._next_physics_step_at = time.perf_counter()
            self._physics_rate_reset_requested = True
            self.command_error = scene.get("warning")
            self._refresh_actor_training_status()
            self._mark_snapshot_status_dirty()
            self._mark_snapshot_static_dirty()
            _store_viewer_settings(self.viewer_settings_path, self.viewer_settings)
            self._viewer_fault = None
            self.reset_progress = dict(phase='ready', generation=self._environment_update_generation)
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(previous)
            self._restore_environment_runtime(reset_state)
            raise

    def _poll_training_scene_load(self) -> bool:
        if self.scene_mode != 'continuous':
            return False
        if getattr(self, '_closed', False):
            return False
        future = self._training_scene_load_future
        if future is None or not future.done():
            return False
        if (self._horizon_pending() is not None and future.exception() is None
                and not self._terminal_environment_update_pending):
            return False
        if self.running and self._physics_ticks_into_decision != 0:
            return False
        self._training_scene_load_future = None
        self._requested_viewer_settings = None
        request = getattr(self, "_training_scene_request", None)
        if request is not None and request != (
            self._environment_update_generation, self._deployment_generations, self.viewer_settings
        ):
            receipt = getattr(self, "_viewer_settings_result", None)
            if receipt and receipt["phase"] == "preparing":
                receipt["phase"] = "superseded"
            return True
        self._requested_viewer_settings = None
        try:
            scene = future.result()
            resume_after = self._resume_after_environment_update
            self._apply_prepared_training_scene(scene)
            self._terminal_environment_update_pending = False
            self._resume_after_environment_update = False
            if resume_after:
                self.running = True
        except Exception as error:
            self._viewer_failure('settings_commit',error)
            receipt = getattr(self, "_viewer_settings_result", None)
            if receipt and receipt["phase"] == "preparing":
                receipt.update(phase="failed", error=str(error))
        else:
            receipt = getattr(self, "_viewer_settings_result", None)
            if receipt and receipt["phase"] == "preparing":
                receipt["phase"] = "applied"
        self._environment_reset_request = None
        return True

    @staticmethod
    def _training_resume_identity(manifest):
        # Runtime diagnostics are allowed to change; research/release history is not.
        return json.dumps({key: manifest.get(key) for key in (
            'run_group_id', 'launch_settings', 'role_bindings', 'seed',
            'global_update', 'checkpoint_update', 'update_budget', 'checkpoint_file', 'collection_history',
        )}, sort_keys=True)

    def _training_request_result(self, item, **changes):
        result = self._training_command_results.get(item['command_group_id'])
        if result and result.get('client_request_id') == item['client_request_id']:
            result.update(changes)
        self._mark_snapshot_status_dirty()

    def _enqueue_training_launch(self, item, request_id):
        key = item['command_group_id']
        previous = self._training_command_results.get(key, {})
        if (request_id and previous.get('client_request_id') == request_id
                or previous.get('phase') in {'queued', 'preparing', 'starting'}):
            return previous
        item.update(client_request_id=request_id or training_model_timestamp_id(time.time_ns()),
                    deployment_generations=dict(self._deployment_generations),
                    follow_intent=(self._auto_deploy_training_run_group_id,
                                   frozenset(self._auto_deploy_training_sides)))
        result = dict(client_request_id=item['client_request_id'], type=item['type'],
                      run_group_id=item.get('run_group_id'), phase='queued', error=None,
                      observed_attempt_id=item.get('observed_attempt_id'),
                      viewer_instance=self._training_command_instance, queue_position=0)
        self._training_command_results[key] = result
        self._training_follow_request = item['client_request_id']
        self._joint_training_launch_queue.append(item)
        self._pump_training_launch_queue()
        return result

    def _start_joint_training(self, request_id=None):
        previous = self._training_command_results.get('new', {})
        if previous.get('phase') in {'queued', 'preparing', 'starting'}:
            return previous
        settings = copy.deepcopy(self.viewer_settings)
        sides = sorted(train_roles(settings))
        if not sides:
            raise ValueError("Set at least one NN role to Train")
        if self._training_scene_load_future is not None or self._deployment_futures or getattr(self, '_follow_bundle_future', None):
            raise ValueError('Wait for the pending Viewer deployment/settings request')
        return self._enqueue_training_launch(dict(
            command_group_id='new', type='start_joint_training', settings=settings,
            sources={role:None if role in sides or settings.get(role + '_control_mode') == 'rule'
                     else copy.deepcopy(self._deployed_sources.get(role)) for role in policy_roles(settings)},
            model_ids={role:None if settings.get(role + '_control_mode') == 'rule'
                       else 'random_current_architecture' if role in sides
                       else self._runtime_deployment_model_id(role) for role in policy_roles(settings)},
            train_sides=sides, seed=42, resume=False), request_id)

    def _resume_joint_training(self, group_id, updates=None, request_id=None):
        previous = self._training_command_results.get(group_id, {})
        if (request_id and previous.get('client_request_id') == request_id
                or previous.get('phase') in {'queued', 'preparing', 'starting'}):
            return previous
        self.actor_training.refresh_training_run_registry()
        entry = next((r for r in self.actor_training.training_run_registry()
                      if r['manifest'].get('run_group_id') == group_id), None)
        if entry is None:
            raise ValueError('Training group is unavailable')
        if entry.get('archived'):
            raise ValueError('Archived training group cannot Continue')
        manifest = entry['manifest']
        if manifest.get('launch_settings', {}).get('population') and updates is not None:
            raise ValueError('Population training budget follows its stage protocol')
        if updates is not None and (isinstance(updates, bool) or not isinstance(updates, int)
                                    or updates < 0 or (updates != 0 and updates <= int(manifest['update_budget']))):
            raise ValueError('New update budget must be an integer greater than the current budget, or 0 for unlimited')
        budget = updates if updates is not None else int(manifest['update_budget'])
        if _training_budget_exhausted(budget, int(manifest['global_update'])):
            raise ValueError('Completed: explicitly raise this group budget to continue')
        settings = copy.deepcopy(manifest['launch_settings'])
        settings['training_settings']['updates'] = budget
        roles = policy_roles(settings)
        source = str(Path(entry['output']) / 'checkpoint-latest.pt')
        return self._enqueue_training_launch(dict(
            command_group_id=group_id, run_group_id=group_id, type='resume_joint_training',
            observed_attempt_id=(manifest.get('launch_attempt') or {}).get('id'),
            output=entry['output'], resume_identity=self._training_resume_identity(manifest),
            settings=settings, sources={r:source for r in roles},
            model_ids={r:manifest['role_bindings'][r]['model_id'] for r in roles},
            train_sides=list(train_roles(settings)), seed=manifest['seed'], resume=True), request_id)

    def _validate_queued_resume(self, item):
        if not item['resume']:
            return
        self.actor_training.refresh_training_run_registry()
        entry = next((r for r in self.actor_training.training_run_registry()
                      if r['manifest'].get('run_group_id') == item['run_group_id']), None)
        if entry is None or self._training_resume_identity(entry['manifest']) != item['resume_identity']:
            raise ValueError('Queued request expired: group configuration or release history changed')

    def _fail_training_launch(self, item, error):
        self._training_request_result(item, phase='failed', error=str(error), queue_position=None)
        if isinstance(error, BrokenProcessPool) and self._joint_training_launch_executor is not None:
            self._joint_training_launch_executor.shutdown(wait=False, cancel_futures=True)
            self._joint_training_launch_executor = None

    def _pump_training_launch_queue(self):
        if getattr(self, '_closed', False):
            return
        # Called on commands, preparation completion, or the existing status poll, not each RAF.
        queue = self._joint_training_launch_queue
        if self._joint_training_launch_pending is None and queue:
            active = self.actor_training.active_training_group_ids()
            while queue and self._joint_training_launch_pending is None:
                item = queue[0]
                if item.get('run_group_id') in active:
                    queue.pop(0)
                    self._training_request_result(item, phase='failed',
                        error='Group was started externally; inspect its current attempt before retrying', queue_position=None)
                    continue
                if len(active) >= MAX_PARALLEL_TRAINING_RUNS:
                    self._training_request_result(item, waiting_for_slot=True)
                    break
                queue.pop(0)
                try:
                    self._validate_queued_resume(item)
                    item['launch_attempt'] = dict(id=training_model_timestamp_id(time.time_ns()),
                                                 started_at=current_timestamp(), stderr_offset=0)
                    if item.get('run_group_id'):
                        self.actor_training.clear_group_process_error(item['run_group_id'])
                    self._joint_training_launch_pending = item
                    self._training_request_result(item, phase='preparing', queue_position=None,
                        waiting_for_slot=False, launch_attempt=item['launch_attempt'])
                    if self._joint_training_launch_executor is None:
                        self._joint_training_launch_executor = ProcessPoolExecutor(max_workers=1)
                    self._joint_training_launch_future = self._joint_training_launch_executor.submit(
                        _prepare_joint_training_launch, str(self.config.repository_root),
                        item['settings'], item['sources'], item['model_ids'],
                        item['train_sides'], item['seed'], resume=item['resume'])
                except Exception as error:
                    self._joint_training_launch_pending = None
                    self._joint_training_launch_future = None
                    self._fail_training_launch(item, error)
        for position, item in enumerate(queue, 1):
            self._training_request_result(item, queue_position=position)

    def _cancel_training_launch(self, group_id, request_id):
        result = self._training_command_results.get(group_id, {})
        if not request_id or result.get('client_request_id') != request_id:
            raise ValueError('Cancellation does not match the accepted request')
        if result.get('phase') == 'cancelled':
            return
        if result.get('phase') not in {'queued', 'preparing'}:
            raise ValueError('Already started; use Pause')
        for item in self._joint_training_launch_queue:
            if item['command_group_id'] == group_id and item['client_request_id'] == request_id:
                self._joint_training_launch_queue.remove(item)
                self._training_request_result(item, phase='cancelled', queue_position=None)
                self._pump_training_launch_queue()
                return
        item = self._joint_training_launch_pending
        if item and item['client_request_id'] == request_id and item['command_group_id'] == group_id:
            item['cancelled'] = True
            self._training_request_result(item, phase='cancelled', queue_position=None)
            # A running prepare is collected and discarded; it never reaches Popen.
            self._joint_training_launch_future.cancel()

    def _collect_joint_training_launch(self) -> bool:
        if getattr(self, '_closed', False):
            return False
        future = self._joint_training_launch_future
        if future is None or not future.done():
            return False
        item = self._joint_training_launch_pending
        self._joint_training_launch_future = None
        self._joint_training_launch_pending = None
        try:
            if not item.get('cancelled'):
                prepared = future.result()
                self._validate_queued_resume(item)
                launch = self.actor_training.start_joint(
                    prepared=prepared, architecture_ids=prepared.architecture_ids,
                    sources={side:Path(source) if source is not None else None for side, source in prepared.sources.items()},
                    source_metadata=prepared.source_metadata, model_ids=prepared.model_ids,
                    train_sides=set(prepared.train_sides), new_series_sides=set(prepared.new_series_sides),
                    joint_mode=prepared.joint_mode, alternating_updates=prepared.alternating_updates,
                    updates=prepared.updates, training_settings=prepared.launch_settings,
                    seed=prepared.seed, launch_attempt=item['launch_attempt'])
                item.update(launch)
                self._joint_training_starting[item['client_request_id']] = item
                self._training_request_result(item, phase='starting', run_group_id=launch['run_group_id'],
                                              launch_attempt=launch['launch_attempt'], pid=launch['pid'])
                for side in prepared.train_sides:
                    self._latest_effective_minibatch_agent_frames[side] = None
                if (item['client_request_id'] == self._training_follow_request
                        and item['deployment_generations'] == self._deployment_generations
                        and item['follow_intent'] == (self._auto_deploy_training_run_group_id,
                                                    frozenset(self._auto_deploy_training_sides))):
                    self._checkpoint_source_user_selected = False
                    self._auto_deploy_training_sides = set(prepared.architecture_ids)
                    self._auto_deploy_training_run_group_id = str(launch['run_group_id'])
        except TrainingCapacityError:
            # A process can claim the final slot while preparation is running.
            # Drop prepared tensors and wait without retaining a loaded model.
            item.pop('launch_attempt', None)
            self._joint_training_launch_queue.insert(0, item)
            self._training_request_result(item, phase='queued', launch_attempt=None, waiting_for_slot=True)
            return True
        except Exception as error:
            self._fail_training_launch(item, error)
        self._pump_training_launch_queue()
        self._mark_snapshot_status_dirty()
        return True

    def _collect_training_starting(self, status):
        finished = False
        for request_id, item in list(self._joint_training_starting.items()):
            group_id = item['run_group_id']
            group = next((g for g in status.get('groups', [])
                          if g.get('run_group_id') == group_id
                          and (g.get('launch_attempt') or {}).get('id') == item['launch_attempt']['id']), None)
            process_running, exit_code = self.actor_training.launch_process_status(group_id)
            if group and group.get('running') and not group.get('starting') and process_running:
                self._training_request_result(item, phase='running', error=None)
            elif not process_running and exit_code is not None:
                self._training_request_result(item, phase='failed',
                    error=self.actor_training.group_process_error(group_id)
                    or f'Training exited before startup confirmation (exit code {exit_code})')
            else:
                continue
            del self._joint_training_starting[request_id]
            finished = True
        return finished

    def _stop_joint_training(self, run_group_id: str) -> None:
        self.actor_training.stop_joint(run_group_id, status=self._actor_training_status_cache)
        self._training_status_generation += 1
        for run in self._actor_training_status_cache.get("all_runs", []):
            if str(run.get("run_group_id")) == run_group_id:
                run["stop_requested"] = True
        self._start_actor_training_status_refresh()

    def _target_outcomes(self) -> list[str]:
        start=self.env.defender_count+self.env.viewer_learner_capacity
        return ['captured' if captured else 'pair_collision' if collision and pair else 'static_collision' if collision else 'running'
                for captured,collision,pair in zip(self.env.target_captured[0,:self.env.sampled_target_count[0]].tolist(),
                    self.env.target_collision[0,:self.env.sampled_target_count[0]].tolist(),
                    self.env.episode_contact_counts[0,start:,1].tolist())]

    def _refresh_actor_training_status(
        self,
        *,
        refresh_source_options: bool = True,
        mark_snapshot_static_dirty: bool = True,
    ) -> None:
        self._training_status_generation += 1
        # Explicit management refresh: discover cold changes now, then replace
        # the reader rather than mutating a launcher used by an in-flight job.
        self.actor_training.refresh_training_run_registry()
        with self._measure_performance_stage("training_status_refresh"):
            self._refresh_actor_training_status_unmeasured(
                refresh_source_options=refresh_source_options,
                mark_snapshot_static_dirty=mark_snapshot_static_dirty,
            )
        self._training_status_reader = self.actor_training.inventory_clone()

    def _refresh_actor_training_status_unmeasured(
        self,
        *,
        refresh_source_options: bool = True,
        mark_snapshot_static_dirty: bool = True,
    ) -> None:
        timings: dict[str, list[float]] = {}
        status = self.actor_training.refresh(
            self.active_actor_variant_id,
            timings=timings,
        )
        self._record_performance_timings(timings)
        artifact_root = (
            self.config.repository_root / "Artifacts" / "GoalReach"
        ).resolve()
        with self._measure_performance_stage("training_status_delete_scan"):
            delete_requests = tuple(
                artifact_root.glob(f"*/{TRAINING_DELETE_REQUEST_NAME}")
            )
        self._apply_actor_training_status(
            status,
            delete_requests,
            refresh_source_options=refresh_source_options,
            mark_snapshot_static_dirty=mark_snapshot_static_dirty,
        )

    @staticmethod
    def _read_actor_training_status(
        actor_training: ActorTrainingLauncher,
        generation: int,
        architecture_id: str,
        artifact_root: Path,
    ) -> tuple[
        int,
        str,
        dict[str, object],
        tuple[Path, ...],
        dict[str, tuple[float, float, float]],
    ]:
        timings: dict[str, list[float]] = {}
        total_started_at = (
            time.perf_counter(),
            time.thread_time(),
            time.process_time(),
        )
        status = actor_training.refresh(
            architecture_id,
            prune_stale_locks=False,
            poll_processes=False,
            timings=timings,
        )
        delete_scan_started_at = (
            time.perf_counter(),
            time.thread_time(),
            time.process_time(),
        )
        delete_requests = tuple(
            artifact_root.glob(f"*/{TRAINING_DELETE_REQUEST_NAME}")
        )

        def add_elapsed(
            name: str,
            started_at: tuple[float, float, float],
        ) -> None:
            wall_started_at, thread_cpu_started_at, process_cpu_started_at = (
                started_at
            )
            elapsed = (
                time.perf_counter() - wall_started_at,
                time.thread_time() - thread_cpu_started_at,
                time.process_time() - process_cpu_started_at,
            )
            totals = timings.setdefault(name, [0.0, 0.0, 0.0])
            for index, value in enumerate(elapsed):
                totals[index] += max(0.0, value)

        add_elapsed("training_status_delete_scan", delete_scan_started_at)
        add_elapsed("training_status_refresh", total_started_at)
        return (
            generation,
            architecture_id,
            status,
            delete_requests,
            {
                name: (values[0], values[1], values[2])
                for name, values in timings.items()
            },
        )

    def _start_actor_training_status_refresh(self) -> None:
        if self._training_status_future is not None:
            return
        with self._measure_performance_stage("training_status_process_poll"):
            self.actor_training.poll_started_processes()
        artifact_root = (
            self.config.repository_root / "Artifacts" / "GoalReach"
        ).resolve()
        self._training_status_future = self._training_status_executor.submit(
            self._read_actor_training_status,
            self._training_status_reader,
            self._training_status_generation,
            self.active_actor_variant_id,
            artifact_root,
        )

    def _collect_actor_training_status(
        self,
        static_revision_before_refresh: int,
    ) -> bool:
        future = self._training_status_future
        if future is None or not future.done():
            return False
        self._training_status_future = None
        try:
            (
                generation,
                architecture_id,
                status,
                delete_requests,
                timings,
            ) = future.result()
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            self.checkpoint_refresh_error = str(error)
            return False
        self._record_performance_timings(timings)
        if (
            generation != self._training_status_generation
            or architecture_id != self.active_actor_variant_id
        ):
            return False
        coalesce_snapshot_dirty = bool(
            self._snapshot_static_revision != static_revision_before_refresh
        )
        with self._measure_performance_stage("training_status_apply"):
            return self._apply_actor_training_status(
                status,
                delete_requests,
                refresh_source_options=False,
                mark_snapshot_static_dirty=not coalesce_snapshot_dirty,
            )

    @staticmethod
    def _public_actor_training_status(
        status: dict[str, object],
    ) -> dict[str, object]:
        return {
            key: value
            for key, value in status.items()
            if key not in {
                "archived_checkpoint_options",
                "deployment_checkpoint_options",
            }
        }

    def _published_actor_training_status(self) -> dict[str, object]:
        public = self._public_actor_training_status(
            self._actor_training_status_cache
        )
        public["history_generation"] = self._training_metrics_invalidation_generation
        return public

    @staticmethod
    def _actor_training_catalog_signature(status: dict[str, object]) -> tuple[object, ...]:
        return tuple((role, tuple((o.get("model_id"), o.get("model_timestamp_id"),
            o.get("source_architecture_id"), o.get("disabled"), o.get("error")) for o in options))
            for role, options in sorted(status.get("role_model_options", {}).items()))

    @classmethod
    def _actor_training_lifecycle_signature(
        cls,
        status: dict[str, object],
    ) -> tuple[object, ...]:
        public = cls._public_actor_training_status(status)
        runs = public.get("all_runs", [])
        run_signatures = tuple(
            (
                run.get("run_id"),
                run.get("run_group_id"),
                run.get("actor_kind"),
                run.get("model_id"),
                run.get("running"),
                run.get("starting"),
                run.get("process_identity_status"),
                run.get("capacity_reserved"),
                run.get("update"),
                run.get("source_update"),
                run.get("target_update"),
                run.get("runtime_phase"),
                run.get("runtime_phase_updated_at_ns"),
                run.get("stop_requested"),
                run.get("delete_requested"),
            )
            for run in runs
            if isinstance(run, dict)
        )
        return (
            public.get("running"),
            public.get("any_running"),
            public.get("running_count"),
            public.get("new_session_allowed"),
            public.get("new_session_disabled_reason"),
            public.get("process_error"),
            public.get("last_exit_code"),
            tuple(public.get("running_architecture_ids", [])),
            tuple(sorted(public.get("effective_minibatch_agent_frames", {}).items())),
            run_signatures,
            tuple((group.get('run_group_id'),group.get('status'),group.get('process_identity_status'),group.get('capacity_reserved'),
                   json.dumps(group.get('launch_attempt'),sort_keys=True),
                   group.get('error'),group.get('error_stage'),group.get('exit_code'),group.get('checkpoint_update'))
                  for group in public.get('groups',[]) if isinstance(group,dict)),
        )

    def _apply_actor_training_status(
        self,
        status: dict[str, object],
        delete_requests: tuple[Path, ...],
        *,
        refresh_source_options: bool,
        mark_snapshot_static_dirty: bool,
    ) -> bool:
        process_error, last_exit_code = self.actor_training.process_status(
            self.active_actor_variant_id
        )
        status["process_error"] = process_error
        status["last_exit_code"] = last_exit_code
        refreshed_inventory: dict[str, object] | None = None
        with self._measure_performance_stage("training_status_inventory_apply"):
            inventory_future = self._training_inventory_future
            if inventory_future is not None and inventory_future.done():
                self._training_inventory_future = None
                try:
                    inventory_architecture_id, inventory = inventory_future.result()
                    if inventory_architecture_id == self.active_actor_variant_id:
                        refreshed_inventory = inventory
                except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                    self.checkpoint_refresh_error = str(error)
                    self._known_running_deployment_checkpoints.clear()
                    if isinstance(error, BrokenProcessPool):
                        assert self._training_inventory_executor is not None
                        self._training_inventory_executor.shutdown(
                            wait=False,
                            cancel_futures=True,
                        )
                        self._training_inventory_executor = None

        active_outputs = {
            Path(str(run["output"])).resolve()
            for run in status.get("all_runs", [])
            if isinstance(run, dict)
            and bool(run.get("capacity_reserved", run.get("running")))
            and isinstance(run.get("output"), str)
        }
        completed_delete = False
        with self._measure_performance_stage("training_status_delete_apply"):
            for delete_request in delete_requests:
                output = delete_request.parent.resolve()
                if output in active_outputs or not delete_request.is_file():
                    continue
                metrics = output / "metrics.jsonl"
                if not metrics.is_file():
                    continue
                self._delete_training_segment(
                    {
                        "source": metrics.relative_to(
                            self.config.repository_root
                        ).as_posix()
                    },
                    refresh_status=False,
                )
                refresh_source_options = True
                completed_delete = True
        if completed_delete:
            timings: dict[str, list[float]] = {}
            status = self.actor_training.refresh(
                self.active_actor_variant_id,
                timings=timings,
            )
            self._record_performance_timings(timings)

        selected_group = self._auto_deploy_training_run_group_id
        selected = next((g for g in status.get('groups', []) if g['run_group_id'] == selected_group), None)
        status['selected_group_id'] = selected_group
        status['effective_minibatch_agent_frames'] = {
            role:(selected or {}).get('effective_minibatch_agent_frames', {}).get(role) for role in ROLE_ORDER}
        with self._measure_performance_stage(
            "training_status_follow_checkpoints"
        ):
            self._follow_training_checkpoints(status)

        running_architecture_ids = sorted(
            str(value)
            for value in status.get("running_architecture_ids", [])
        )
        if not running_architecture_ids:
            running_architecture_ids = sorted(
                {
                    str(run["architecture_id"])
                    for run in status.get("all_runs", [])
                    if isinstance(run, dict)
                    and bool(run.get("running"))
                    and isinstance(run.get("architecture_id"), str)
                }
            )
        previous_status = self._actor_training_status_cache
        with self._measure_performance_stage(
            "training_status_checkpoint_scan"
        ):
            running_deployment_checkpoints = {}
            for run in status.get("all_runs", []):
                if not isinstance(run, dict) or run.get("actor_kind") not in self.env.roles or not isinstance(run.get("output"), str):
                    continue
                checkpoint = Path(run["output"]) / "checkpoint-latest.pt"
                try:
                    stat = checkpoint.stat()
                    running_deployment_checkpoints[checkpoint] = (stat.st_mtime_ns, stat.st_size)
                except OSError:
                    running_deployment_checkpoints[checkpoint] = None
        new_running_deployment_checkpoints = running_deployment_checkpoints != self._known_running_deployment_checkpoints
        if (
            (new_running_deployment_checkpoints or refresh_source_options
             or set(previous_status.get("role_model_options", {})) != set(self.env.roles))
            and self._training_inventory_future is None
        ):
            with self._measure_performance_stage(
                "training_status_inventory_launch"
            ):
                if self._training_inventory_executor is None:
                    self._training_inventory_executor = ProcessPoolExecutor(
                        max_workers=1
                    )
                try:
                    self._training_inventory_future = (
                        self._training_inventory_executor.submit(
                            self._training_inventory,
                            str(self.config.repository_root),
                            self.active_actor_variant_id,
                            tuple(self.env.roles),
                        )
                    )
                except (OSError, RuntimeError) as error:
                    self.checkpoint_refresh_error = str(error)
                    self._training_inventory_executor.shutdown(
                        wait=False,
                        cancel_futures=True,
                    )
                    self._training_inventory_executor = None
                else:
                    self._known_running_deployment_checkpoints = running_deployment_checkpoints

        with self._measure_performance_stage(
            "training_status_source_inventory"
        ):
            if refreshed_inventory is not None:
                archived_checkpoints = refreshed_inventory[
                    "archived_checkpoint_options"
                ]
                deployment_checkpoints = refreshed_inventory[
                    "deployment_checkpoint_options"
                ]
                self._opponent_option_cache = None
            else:
                archived_checkpoints = previous_status.get(
                    "archived_checkpoint_options", []
                )
                deployment_checkpoints = previous_status.get(
                    "deployment_checkpoint_options", []
                )

        configured_train_sides = {
            side
            for side, mode in self.viewer_settings["training_side_modes"].items()
            if mode == "train"
        }
        if int(status["running_count"]) >= int(status["max_parallel_runs"]):
            new_session_disabled_reason = (
                f"{status['running_count']}/{status['max_parallel_runs']} "
                "training slots are reserved"
            )
        elif not configured_train_sides:
            new_session_disabled_reason = "Set ATK or DEF to Train"
        else:
            new_session_disabled_reason = None
        status.update(
            {
                "selected_architecture_id": self.active_actor_variant_id,
                "any_running": bool(running_architecture_ids),
                "running_architecture_ids": running_architecture_ids,
                "new_session_allowed": new_session_disabled_reason is None,
                "new_session_disabled_reason": new_session_disabled_reason,
                "archived_checkpoint_options": archived_checkpoints,
                "deployment_checkpoint_options": deployment_checkpoints,
                "role_model_options": (refreshed_inventory or previous_status).get("role_model_options", {}),
            }
        )
        pending_launch_finished = self._collect_training_starting(status)
        self._pump_training_launch_queue()
        self._invalidate_training_history_for_sessions(
            previous_status, status, launch_finished=pending_launch_finished,
        )
        catalog_may_have_changed = bool(
            refresh_source_options or refreshed_inventory is not None
        )
        if (
            not catalog_may_have_changed
            and self._actor_training_catalog_state_signature
        ):
            catalog_signature = self._actor_training_catalog_state_signature
        else:
            catalog_signature = self._actor_training_catalog_signature(status)
        lifecycle_signature = self._actor_training_lifecycle_signature(status)
        self._actor_training_status_cache = status
        if refreshed_inventory is not None and not previous_status.get("role_model_options"):
            if self.scene_mode == 'continuous':
                self._restore_deployment_models(only_unrequested=True)
            else:
                self._restore_continuous_deployments_pending = True
        catalog_changed = (
            catalog_signature != self._actor_training_catalog_state_signature
        )
        lifecycle_changed = (
            lifecycle_signature != self._actor_training_lifecycle_state_signature
        )
        self._actor_training_catalog_state_signature = catalog_signature
        self._actor_training_lifecycle_state_signature = lifecycle_signature
        lifecycle_changed = lifecycle_changed or pending_launch_finished
        if lifecycle_changed:
            self._mark_snapshot_status_dirty()
        if catalog_changed:
            if mark_snapshot_static_dirty:
                self._mark_snapshot_static_dirty()
            else:
                self._snapshot_static_dirty_deferred = True
        return lifecycle_changed or catalog_changed

    def _invalidate_training_history_for_sessions(
        self,
        previous: dict[str, object],
        current: dict[str, object],
        *,
        launch_finished: bool = False,
    ) -> None:
        def active_groups(status: dict[str, object]) -> set[str]:
            return {
                str(run["run_group_id"])
                for run in status.get("all_runs", [])
                if isinstance(run, dict) and run.get("run_group_id")
                and (run.get("running") or run.get("starting"))
            }

        # Locks disappear after the final metrics files close. Compare each
        # session, not any_running, update counters, or rollout phases.
        if launch_finished or active_groups(previous) != active_groups(current):
            self._invalidate_training_metrics()

    @staticmethod
    def _training_inventory(
        repository_root: str,
        architecture_id: str,
        roles: tuple[str, ...] = ("learner", "defender"),
    ) -> tuple[str, dict[str, object]]:
        root = str(Path(repository_root).resolve())
        actor_training = _TRAINING_INVENTORY_WORKER_LAUNCHERS.get(root)
        if actor_training is None:
            actor_training = ActorTrainingLauncher(Path(root))
            _TRAINING_INVENTORY_WORKER_LAUNCHERS[root] = actor_training
        actor_training.refresh_training_run_registry()
        architectures=tuple(dict.fromkeys(architecture for role in roles for architecture in _deployment_architecture_ids(role)))
        options=actor_training.deployment_checkpoint_inventory(architectures)
        return architecture_id, {
            "archived_checkpoint_options": [],
            "deployment_checkpoint_options": [o for o in options if o['actor_kind']=='learner'],
            "role_model_options": {role:[o for o in options if o['actor_kind']==role] for role in roles},
        }

    def _delete_actor_training_run(self, command: dict[str, object]) -> None:
        run_id = str(command.get("run_id", ""))
        if not run_id:
            raise ValueError("Select a training run to delete")
        status = self.actor_training.refresh()
        run = next(
            (
                item
                for item in status.get("all_runs", [])
                if isinstance(item, dict) and str(item.get("run_id", "")) == run_id
            ),
            None,
        )
        if run is None or not isinstance(run.get("output"), str):
            raise ValueError("Training run is unavailable")
        if bool(run.get("running")):
            raise ValueError("Pause joint training before deleting its history")
        metrics = Path(str(run["output"])) / "metrics.jsonl"
        if not metrics.is_file():
            raise ValueError("Training run has no saved history")
        self._delete_training_segment(
            {
                "source": metrics.relative_to(
                    self.config.repository_root
                ).as_posix()
            }
        )

    @staticmethod
    def _candidate_checkpoint_payload(
        candidate: dict[str, object],
    ) -> dict[str, object]:
        payload = {
            "architecture_id": candidate["architecture_id"],
            "update_index": candidate["update"],
            "agent_frames": candidate["agent_frames"],
            "model": candidate["model"],
        }
        for key in (
            "training_settings",
            "initial_rollout_metrics",
            "weights_initialization",
            "checkpoint_role",
            "scene_contract",
        ):
            if key in candidate:
                payload[key] = candidate[key]
        return payload

    def _prepare_checkpoint_model(
        self, payload: dict[str, object]
    ) -> tuple[dict[str, object], ActorCritic]:
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint payload must be a mapping")
        model_state = payload.get("model")
        if not isinstance(model_state, dict):
            raise ValueError("Checkpoint model state must be a mapping")
        compact_payload = {
            key: payload[key]
            for key in (
                "source_metadata", "role", "parameter_group_update", "launch_settings", "role_bindings", "global_update", "run_group_id",
                "architecture_id",
                "update_index",
                "agent_frames",
                "training_settings",
                "initial_rollout_metrics",
                "weights_initialization",
                "checkpoint_role",
                "goal_reach",
                "training_restore",
                "scene_contract",
            )
            if key in payload
        }
        compact_payload["model"] = {
            name: tensor.detach().clone()
            for name, tensor in model_state.items()
            if isinstance(tensor, torch.Tensor) and name.startswith("actor")
        }
        architecture_id = self._validate_checkpoint(compact_payload)
        prepared_model = self._new_actor_model(architecture_id)
        prepared_model.load_state_dict(
            actor_deployment_state_dict(prepared_model, compact_payload["model"]),
            strict=True,
        )
        prepared_model.requires_grad_(False)
        prepared_model.eval()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return compact_payload, prepared_model

    def _prepare_rule_deployment(self, generation, architecture_id, payload=None, *, preserve_training_follow=False):
        model = self.control.neural_policy.model
        model_id = self._runtime_deployment_model_id('learner')
        if model.variant.architecture_id != architecture_id:
            model = self._new_actor_model(architecture_id, 'learner')
            model.requires_grad_(False)
            model_id = 'random_current_architecture'
        return _PreparedViewerDeployment(generation, 'learner', model_id,
            architecture_id, 'rule', 'rule', model, payload, None, None, None, False, preserve_training_follow)

    def _prepare_viewer_deployment(
        self,
        generation: int,
        actor_kind: str,
        model_id: str,
        architecture_id: str,
        source_kind: str,
        source_id: str | None,
        checkpoint: Path | None,
        selection_mode: str,
        candidate: dict[str, object] | None,
        persist: bool,
        preserve_training_follow: bool,
        *,
        prepared_sources: dict | None = None,
    ) -> _PreparedViewerDeployment:
        payload: dict[str, object] | None = None
        checkpoint_mtime_ns: int | None = None
        prepared_models = {}
        if source_kind == "random":
            model = self._new_actor_model(architecture_id, actor_kind)
            model.requires_grad_(False)
        else:
            if source_kind == "saved":
                if source_id is None:
                    raise ValueError("Saved deployment source is missing")
                checkpoint = self.actor_training.resolve_deployment_checkpoint(
                    architecture_id, source_id
                )
            elif source_kind == "archived":
                if source_id is None:
                    raise ValueError("Archived deployment source is missing")
                checkpoint = self.actor_training.resolve_archived_checkpoint(
                    architecture_id, source_id
                )
            elif source_kind not in {"checkpoint", "embedded", "candidate"}:
                raise ValueError("Deployment source is unsupported")
            if source_kind == "candidate":
                if candidate is None:
                    raise ValueError("Deployment candidate is missing")
                payload = self._candidate_checkpoint_payload(candidate)
            else:
                if checkpoint is None:
                    raise ValueError("Deployment checkpoint is missing")
                require_active_training_checkpoint(checkpoint)
                checkpoint, checkpoint_mtime_ns, payload, prepared_models = _read_group_release(checkpoint, prepared_sources)
            if source_kind != 'candidate':
                payload = _group_actor_payload(payload, actor_kind, checkpoint)
                if payload.get('binding', {}).get('mode') == 'rule':
                    if actor_kind != 'learner':
                        raise ValueError('Only ATK supports Rule (APF)')
                    return self._prepare_rule_deployment(generation, architecture_id, payload,
                        preserve_training_follow=preserve_training_follow)
                if payload.get('binding', {}).get('mode') == 'static':
                    raise ValueError('Static TGT has no Actor to deploy')
                metadata = payload['source_metadata']
                key = tuple(metadata[name] for name in ('run_group_id','parameter_group','source_weight_update','architecture_id'))
                checkpoint = Path(metadata['checkpoint_path'])
                checkpoint_mtime_ns = checkpoint.stat().st_mtime_ns
            else:
                key = None
            if key is not None and key in prepared_models:
                model = prepared_models[key]
                payload['model'] = model.state_dict()
            else:
                payload, model = self._prepare_checkpoint_model(payload)
                if key is not None:
                    prepared_models[key] = model
            loaded_architecture_id = self._validate_checkpoint(payload)
            if source_kind == "embedded":
                architecture_id = loaded_architecture_id
            elif loaded_architecture_id != architecture_id:
                raise ValueError("Selected checkpoint architecture changed while loading")
            if checkpoint is not None:
                checkpoint = checkpoint.resolve()
        if not actor_variant(architecture_id).unified_match_contract and actor_variant(architecture_id).actor_kind != actor_kind:
            raise ValueError("The selected checkpoint has the wrong Actor side")
        if payload is not None and checkpoint is not None:
            payload["_viewer_checkpoint_metric"] = self._checkpoint_metric(
                int(payload.get("update_index", 0)), metrics_path=checkpoint.parent / "metrics.jsonl", actor_kind=actor_kind)
        return _PreparedViewerDeployment(
            generation=generation,
            actor_kind=actor_kind,
            model_id=model_id,
            architecture_id=architecture_id,
            source_kind=source_kind,
            selection_mode=selection_mode,
            model=model,
            payload=payload,
            checkpoint=checkpoint,
            checkpoint_mtime_ns=checkpoint_mtime_ns,
            candidate=candidate,
            persist=persist,
            preserve_training_follow=preserve_training_follow,
        )

    def _queue_viewer_deployment(
        self,
        *,
        actor_kind: str,
        model_id: str,
        architecture_id: str,
        source_kind: str,
        selection_mode: str,
        source_id: str | None = None,
        checkpoint: Path | None = None,
        candidate: dict[str, object] | None = None,
        persist: bool,
        preserve_training_follow: bool = False,
        prepared_sources: dict | None = None,
    ) -> None:
        self._deployment_generations[actor_kind] += 1
        generation = self._deployment_generations[actor_kind]
        self._deployment_errors.pop(actor_kind, None)
        previous = self._deployment_futures.pop(actor_kind, None)
        if previous is not None:
            previous[1].cancel()
        self._pending_deployments.pop(actor_kind, None)
        active_kind = self.active_actor_kind
        if actor_kind == active_kind:
            if self._checkpoint_load_future is not None:
                self._checkpoint_load_future.cancel()
                self._checkpoint_load_future = None
            if self._initial_checkpoint_load_future is not None:
                self._initial_checkpoint_load_future.cancel()
                self._initial_checkpoint_load_future = None
            self._pending_checkpoint_load = None
            self._pending_initial_checkpoint_load = None
            self._pending_checkpoint_source_reset = False
        elif actor_kind == self.opponent_actor_kind:
            if self._opponent_checkpoint_load_future is not None:
                self._opponent_checkpoint_load_future.cancel()
                self._opponent_checkpoint_load_future = None
            self._pending_opponent_checkpoint_load = None
        future = self._checkpoint_load_executor.submit(
            self._prepare_viewer_deployment,
            generation,
            actor_kind,
            model_id,
            architecture_id,
            source_kind,
            source_id,
            checkpoint,
            selection_mode,
            candidate,
            persist,
            preserve_training_follow,
            prepared_sources=prepared_sources,
        )
        self._deployment_futures[actor_kind] = (generation, future)

    def _install_active_actor_model(
        self, architecture_id: str, model: ActorCritic, *, persist: bool = True
    ) -> None:
        active_kind = self.active_actor_kind
        if not actor_variant(architecture_id).unified_match_contract and actor_variant(architecture_id).actor_kind != active_kind:
            raise ValueError("A hot deployment cannot change the active Actor side")
        architecture_changed = architecture_id != self.active_actor_variant_id
        if architecture_changed:
            self._invalidate_environment_updates()
            if active_kind == "learner":
                self.env.set_learner_architecture(architecture_id)
            else:
                self.env.set_patrol_architecture(architecture_id)
            self.active_actor_variant_id = architecture_id
            self.viewer_settings["active_actor_variant_id"] = (
                _selected_scenario_architecture_id(architecture_id)
            )
            if active_kind == "defender":
                self.viewer_settings["include_learner_speed_ratio_observation"] = (
                    _architecture_uses_learner_speed_observation(architecture_id)
                )
                self.viewer_settings["entity_streams"] = _architecture_entity_streams(
                    architecture_id
                )
                self.viewer_settings["entity_self_attention_layers"] = (
                    _architecture_entity_self_attention_layers(architecture_id)
                )
            if persist:
                _store_viewer_settings(self.viewer_settings_path, self.viewer_settings)
        self.model = model
        self.policy = _ViewerPolicy(model)
        if active_kind == "learner":
            self.control.neural_policy = self.policy
        else:
            self.patrol_control.neural_policy = self.policy

    def _validate_deployment_task(self, deployment, settings=None):
        expected = copy.deepcopy(self.viewer_settings if settings is None else settings)
        if settings is None:
            expected.update(include_remaining_time_observation='remaining_time' in self.env.learner_variant.vector_inputs,
                            terminate_on_defender_elimination=self.env.terminate_on_defender_elimination,
                            terminate_on_any_casualty=self.env.terminate_on_any_casualty,
                            reward_objective=self.env.reward_objective,
                            active_actor_variant_id=self.env.learner_variant.architecture_id)
            expected['training_settings'].update(maximum_episode_decisions=int(self.config.shared['clock']['maximum_decisions']),
                time_limit_boundary='truncation' if self.env.bootstrap_on_time_limit else 'termination')
        if deployment.architecture_id != _unified_match_architecture_id(expected, deployment.actor_kind):
            raise ValueError('Selected deployment observation architecture differs from the Viewer task')
        if deployment.payload and deployment.payload.get('launch_settings'):
            from .goal_reach_evaluation import evaluation_game_contract
            source_contract = evaluation_game_contract(deployment.payload['launch_settings'])
            viewer_contract = evaluation_game_contract(expected)
            # A Viewer control switch does not alter saved weights or their task.
            # Preserve strict physical/observation compatibility while allowing
            # an old NN to remain dormant behind Rule. Training and evaluation
            # continue to compare their complete policy-source identities.
            source_contract.pop('policy_sources', None)
            viewer_contract.pop('policy_sources', None)
            if source_contract != viewer_contract:
                raise ValueError('Selected deployment task/observation contract differs from the applied Viewer')

    def _apply_prepared_viewer_deployment(
        self, deployment: _PreparedViewerDeployment, *, finalize: bool = True
    ) -> None:
        if (
            deployment.generation
            != self._deployment_generations[deployment.actor_kind]
        ):
            return
        if (deployment.actor_kind == 'learner' and deployment.persist
                and deployment.source_kind != 'rule'
                and self.viewer_settings['learner_control_mode'] == 'rule'):
            # User requests issued before the Rule selection cannot replace the
            # preserved NN; internal persisted-source restores use persist=False.
            return
        checkpoint = deployment.checkpoint
        if deployment.source_kind == 'rule':
            if deployment.model_id == 'random_current_architecture':
                self._apply_prepared_viewer_deployment(replace(deployment,
                    source_kind='random', selection_mode='random', payload=None), finalize=False)
            rule = (deployment.payload or {}).get('binding', {}).get('rule', self.viewer_settings['learner_rule'])
            self.viewer_settings['learner_rule'] = APFConfig.from_dict(rule).to_dict()
            self.viewer_settings['learner_control_mode'] = 'rule'
            self.viewer_settings['training_side_modes']['learner'] = 'fixed'
            for index in range(self.control.capacity):
                self.control.set_mode(index, 'rule')
            return
        if finalize:
            self._validate_deployment_task(deployment)
        if checkpoint is not None and not deployment.preserve_training_follow and (
            not checkpoint.is_file()
            or checkpoint.stat().st_mtime_ns != deployment.checkpoint_mtime_ns
        ):
            raise ValueError("Selected checkpoint changed while preparing deployment")
        # Viewer models are immutable between deployments; hash the actual candidate
        # once, not the selected path or a tensor copy on every snapshot.
        deployment.model._viewer_actor_hash = actor_state_hash(deployment.model.state_dict())
        active_kind = self.active_actor_kind
        if deployment.actor_kind == active_kind:
            previous_architecture_id = self.active_actor_variant_id
            self._install_active_actor_model(
                deployment.architecture_id, deployment.model, persist=finalize
            )
            if deployment.source_kind == "random":
                if deployment.architecture_id != previous_architecture_id:
                    self.checkpoint = self.deployment_checkpoints[
                        deployment.architecture_id
                    ]
                    self.metrics_path = self.checkpoint.parent / "metrics.jsonl"
                    self.initial_checkpoint = (
                        self.checkpoint.parent / INITIAL_CHECKPOINT_NAME
                    )
                    self._checkpoint_mtime_ns = (
                        self.checkpoint.stat().st_mtime_ns
                        if self.checkpoint.is_file()
                        else None
                    )
                self._select_random_current_checkpoint(
                    forget_saved_checkpoints=(
                        deployment.architecture_id != previous_architecture_id
                    ),
                    prepared_model=deployment.model,
                    refresh_actor_frame=False,
                )
                self._checkpoint_source_model_id = deployment.model_id
                self._checkpoint_source_user_selected = True
            elif deployment.source_kind == "candidate":
                if deployment.candidate is None:
                    raise ValueError("Prepared deployment candidate is missing")
                self.checkpoint_selection_mode = deployment.selection_mode
                self._manual_checkpoint_update = (
                    int(deployment.candidate["update"])
                    if deployment.selection_mode in {"manual", "initial"}
                    else None
                )
                self._clear_archived_checkpoint_selection()
                self._checkpoint_source_user_selected = True
                self._deploy_checkpoint(
                    deployment.candidate,
                    force=True,
                    refresh_actor_frame=False,
                    load_model=False,
                )
            else:
                if checkpoint is None or deployment.payload is None:
                    raise ValueError("Prepared deployment checkpoint is incomplete")
                self._set_checkpoint_source(
                    checkpoint,
                    payload=deployment.payload,
                    prepared_model=deployment.model,
                    source_model_id=deployment.model_id,
                    user_selected=True,
                    refresh_actor_frame=False,
                    preserve_training_follow=not finalize or deployment.preserve_training_follow,
                    mark_snapshot_static_dirty=False,
                )
                self.checkpoint_selection_mode = deployment.selection_mode
                self._manual_checkpoint_update = self.checkpoint_update if deployment.selection_mode == "manual" else None
                if deployment.selection_mode == "archived":
                    self.checkpoint_selection_mode = "archived"
                    self._manual_checkpoint_update = None
                    self._archived_checkpoint_model_id = deployment.model_id
                    self._archived_checkpoint_path = checkpoint
                    if deployment.candidate is not None:
                        self.checkpoint_success_rate = deployment.candidate.get(
                            "success_rate"
                        )
                        self.checkpoint_collision_rate = deployment.candidate.get(
                            "collision_rate"
                        )
            if deployment.persist and finalize:
                self._persist_deployment_model(
                    deployment.actor_kind, deployment.model_id
                )
            if finalize and not self._actor_frame_updates_deferred():
                self._refresh_actor_frame()
        elif deployment.actor_kind == self.opponent_actor_kind:
            payload = deployment.payload or {}
            self._apply_opponent_model(
                model_id=deployment.model_id,
                architecture_id=deployment.architecture_id,
                checkpoint=checkpoint,
                model=deployment.model,
                update=int(payload.get("update_index", 0)),
                agent_frames=int(payload.get("agent_frames", 0)),
                selection=deployment.selection_mode,
                refresh_actor_frame=finalize and not self._actor_frame_updates_deferred(),
            )
            if deployment.persist and finalize:
                self._persist_deployment_model(
                    deployment.actor_kind, deployment.model_id
                )
        elif deployment.actor_kind == "target" and self.env.target_mode == "dynamic":
            self.env.target_variant = actor_variant(deployment.architecture_id)
            self.target_runtime = dict(architecture_id=deployment.architecture_id, model=deployment.model,
                                       model_id=deployment.model_id, checkpoint=checkpoint,
                                       update=int((deployment.payload or {}).get("update_index", 0)),
                                       agent_frames=int((deployment.payload or {}).get("agent_frames", 0)),
                                       loaded=True,
                                       mtime_ns=deployment.checkpoint_mtime_ns, selection_mode=deployment.selection_mode)
            self.target_control.neural_policy = _ViewerPolicy(deployment.model)
            if deployment.persist and finalize:
                self._persist_deployment_model("target", deployment.model_id)
            if finalize and not self._actor_frame_updates_deferred():
                self._refresh_actor_frame()
        else:
            return
        self._deployed_sources = {**self._deployed_sources, deployment.actor_kind:
            copy.deepcopy((deployment.payload or {}).get('source_metadata'))}
        if finalize and not deployment.preserve_training_follow:
            self._discard_training_follow_side(deployment.actor_kind)
        self.command_error = None
        if finalize:
            self._mark_snapshot_static_dirty()

    def _install_viewer_deployments(self, deployments):
        if not deployments or any(item.generation != self._deployment_generations[item.actor_kind] for item in deployments):
            return
        if any(item.actor_kind == 'learner' and item.persist and item.source_kind != 'rule'
                and self.viewer_settings['learner_control_mode'] == 'rule' for item in deployments):
            return
        for item in deployments:
            self._validate_deployment_task(item)
        previous = self._capture_joint_checkpoint_runtime()
        try:
            for item in deployments:
                self._apply_prepared_viewer_deployment(item, finalize=False)
            if not self._actor_frame_updates_deferred():
                self._refresh_actor_frame()
            model_ids = dict(self.viewer_settings["deployment_model_ids"])
            model_ids.update({item.actor_kind: item.model_id if item.preserve_training_follow else
                self._runtime_deployment_model_reference(item.actor_kind) for item in deployments if item.persist})
            self.viewer_settings["deployment_model_ids"] = model_ids
            ends_follow = any(not item.preserve_training_follow for item in deployments)
            self.viewer_settings['auto_deploy_training_run_group_id'] = None if ends_follow else self._auto_deploy_training_run_group_id
            if any(item.persist for item in deployments):
                _store_viewer_settings(self.viewer_settings_path, self.viewer_settings)
            if ends_follow:
                self._discard_training_follow_side(deployments[0].actor_kind)
            for item in deployments:
                self._deployment_errors.pop(item.actor_kind, None)
            self._mark_snapshot_static_dirty()
        except Exception:
            self._restore_joint_checkpoint_runtime(previous)
            raise

    def _poll_viewer_deployments(self) -> bool:
        if self.scene_mode != 'continuous':
            return False
        if getattr(self, '_closed', False):
            return False
        if getattr(self, '_replay_record', None) is not None:
            return False
        changed = False
        for actor_kind, (generation, future) in tuple(
            self._deployment_futures.items()
        ):
            if not future.done():
                continue
            self._deployment_futures.pop(actor_kind, None)
            try:
                deployment = future.result()
            except Exception as error:
                if generation == self._deployment_generations[actor_kind]:
                    self.command_error = str(error)
                    self._deployment_errors[actor_kind] = str(error)
                    self._mark_snapshot_static_dirty()
                    changed = True
                continue
            if generation == self._deployment_generations[actor_kind]:
                self._pending_deployments[actor_kind] = deployment
        if self.running and self._physics_ticks_into_decision != 0:
            return changed
        for actor_kind, deployment in tuple(self._pending_deployments.items()):
            self._pending_deployments.pop(actor_kind, None)
            if deployment.generation != self._deployment_generations[actor_kind]:
                continue
            try:
                self._install_viewer_deployments([deployment])
            except Exception as error:
                self.command_error = str(error)
                self._deployment_errors[actor_kind] = str(error)
                self._mark_snapshot_static_dirty()
            changed = True
        return changed

    def _load_checkpoint_in_background(
        self,
        checkpoint: Path,
        _requested_mtime_ns: int,
        role: str = 'learner',
    ) -> tuple[Path, int, dict[str, object], ActorCritic]:
        checkpoint, mtime_ns, payload, _ = _read_group_release(checkpoint)
        compact_payload, prepared_model = self._prepare_checkpoint_model(_group_actor_payload(payload, role, checkpoint))
        return checkpoint, mtime_ns, compact_payload, prepared_model

    def _capture_joint_checkpoint_runtime(self) -> dict[str, object]:
        return {
            "attributes": {
                name: getattr(self, name)
                for name in (
                    "active_actor_variant_id", "checkpoint", "metrics_path", "initial_checkpoint",
                    "checkpoint_selection_mode", "_manual_checkpoint_update", "_checkpoint_source_model_id",
                    "_checkpoint_source_user_selected", "_archived_checkpoint_model_id", "_archived_checkpoint_path",
                    "opponent_architecture_id", "opponent_checkpoint", "opponent_model_id", "opponent_selection",
                    "_opponent_option_cache",
                    "model",
                    "policy",
                    "_checkpoint_candidates",
                    "_initial_checkpoint_mtime_ns",
                    "latest_checkpoint_update",
                    "checkpoint_update",
                    "checkpoint_agent_frames",
                    "checkpoint_success_rate",
                    "checkpoint_collision_rate",
                    "weights_initialization",
                    "_active_checkpoint_loaded",
                    "_pending_checkpoint_source_reset",
                    "_checkpoint_mtime_ns",
                    "_pending_checkpoint_load",
                    "opponent_model",
                    "opponent_policy",
                    "opponent_update",
                    "opponent_agent_frames",
                    "_opponent_checkpoint_mtime_ns",
                    "_pending_opponent_checkpoint_load",
                    "_snapshot_static_revision",
                    "_snapshot_static_dirty_deferred",
                    "_actor_inputs_revision",
                    "_applied_frame_revision", "_applied_frame_cache", "_applied_frame_signature",
                    "_applied_frame_actions", "_applied_frame_modes", "_applied_frame_tick",
                    "_encoded_actor_inputs_revision",
                    "actor_frames",
                    "observation",
                    "actor_frame_physics_ticks",
                    "last_action",
                )
            },
            "viewer_settings": copy.deepcopy(self.viewer_settings),
            "target_runtime": dict(self.target_runtime) if self.env.target_mode == "dynamic" else None,
            "deployed_sources": copy.deepcopy(self._deployed_sources),
            "target_variant": self.env.target_variant if self.env.target_mode == "dynamic" else None,
            "target_policy": self.target_control.neural_policy if self.env.target_mode == "dynamic" else None,
            "last_target_action": self.last_target_action.clone() if self.env.target_mode == "dynamic" else None,
            "env_architectures": (self.env.learner_variant.architecture_id, self.env.patrol_variant.architecture_id),
            "learner_policy": self.control.neural_policy,
            "defender_policy": self.patrol_control.neural_policy,
            "last_learner_action": self.last_learner_action.clone(),
            "last_defender_action": self.last_defender_action.clone(),
            "torch_rng_state": torch.get_rng_state(),
        }

    def _restore_joint_checkpoint_runtime(self, state: dict[str, object]) -> None:
        self.viewer_settings = state["viewer_settings"]
        learner_architecture, defender_architecture = state["env_architectures"]
        self.env.set_learner_architecture(learner_architecture)
        self.env.set_patrol_architecture(defender_architecture)
        self._deployed_sources = state['deployed_sources']
        if state["target_runtime"] is not None:
            self.target_runtime = state["target_runtime"]
            self.env.target_variant = state["target_variant"]
            self.target_control.neural_policy = state["target_policy"]
            self.last_target_action.copy_(state["last_target_action"])
        attributes = state["attributes"]
        assert isinstance(attributes, dict)
        for name, value in attributes.items():
            setattr(self, name, value)
        self.control.neural_policy = state["learner_policy"]
        self.patrol_control.neural_policy = state["defender_policy"]
        last_learner_action = state["last_learner_action"]
        last_defender_action = state["last_defender_action"]
        torch_rng_state = state["torch_rng_state"]
        assert isinstance(last_learner_action, torch.Tensor)
        assert isinstance(last_defender_action, torch.Tensor)
        assert isinstance(torch_rng_state, torch.Tensor)
        self.last_learner_action.copy_(last_learner_action)
        self.last_defender_action.copy_(last_defender_action)
        torch.set_rng_state(torch_rng_state)

    def _refresh_checkpoint(self) -> bool:
        with self._measure_performance_stage("checkpoint_refresh"):
            return self._refresh_checkpoint_unmeasured()

    def _refresh_checkpoint_unmeasured(self) -> bool:
        if getattr(self, '_closed', False):
            return False
        revision = self._snapshot_static_revision
        changed = self._collect_joint_training_launch() | self._poll_viewer_deployments()
        changed = self._collect_actor_training_status(revision) or changed
        now = time.perf_counter()
        poll_due = now-self._last_checkpoint_poll >= 1.0
        # Poll completed loads every loop so installation still happens at the
        # next decision boundary. Registry/path discovery only needs the status
        # cadence; doing Windows path resolution every physics batch is costly.
        # Keep this poll even if the asynchronous status reader has failed.
        self._follow_training_checkpoints(
            self._actor_training_status_cache, discover=poll_due,
        )
        if poll_due:
            self._last_checkpoint_poll=now
            self._start_actor_training_status_refresh()
        return changed

    def _environment_specification(
        self,
        architecture_id: str,
        *,
        settings: dict[str, object] | None = None,
    ) -> tuple[ProjectConfig, type[GoalReachEnv], dict[str, object]]:
        resolved_settings = self.viewer_settings if settings is None else settings
        variant = actor_variant(architecture_id)
        _apply_training_schedule(resolved_settings, architecture_id)
        config = _build_config()
        _apply_persisted_randomization(
            config,
            resolved_settings,
            architecture_id,
        )
        environment_type = (
            MultiGoalReachEnv
            if _uses_multi_learner_environment(architecture_id)
            else GoalReachEnv
        )
        extra_arguments: dict[str, object] = {
            "learner_architecture_id": (
                architecture_id
                if variant.actor_kind == "learner"
                else _unified_match_architecture_id(resolved_settings, "learner")
                if variant.unified_match_contract
                else MULTI_GOAL_REACH_ARCHITECTURE_ID
            ),
            "patrol_architecture_id": (
                architecture_id
                if variant.actor_kind == "defender"
                else _unified_match_architecture_id(resolved_settings, "defender")
                if variant.unified_match_contract
                else DEFAULT_ACTOR_VARIANT_IDS["defender"]
            ),
        }
        if environment_type is MultiGoalReachEnv:
            extra_arguments["terminate_on_any_casualty"] = resolved_settings["terminate_on_any_casualty"]
            extra_arguments["terminate_on_defender_elimination"] = resolved_settings["terminate_on_defender_elimination"]
            extra_arguments["apply_sampling_time_limit"] = resolved_settings["viewer_apply_sampling_limit"]
            extra_arguments["reward_objective"] = resolved_settings["reward_objective"]
            extra_arguments["learner_count_range"] = resolved_settings[
                "learner_count_range"
            ]
            extra_arguments["learner_speed_ratio_range"] = resolved_settings[
                "learner_speed_ratio_range"
            ]
            extra_arguments["defender_count_range"] = _resolved_defender_count_range(
                resolved_settings, architecture_id
            )
        return config, environment_type, extra_arguments

    def _instantiate_environment(
        self,
        architecture_id: str,
        config: ProjectConfig,
        environment_type: type[GoalReachEnv],
        extra_arguments: dict[str, object],
    ) -> GoalReachEnv:
        variant = actor_variant(architecture_id)
        env = environment_type(
            config,
            1,
            self.device,
            base_seed=int(GOAL_REACH["seed"]),
            map_device=self.map_device,
            training_target=variant.actor_kind,
            **extra_arguments,
        )
        if isinstance(env, MultiGoalReachEnv):
            env.enable_viewer_event_capture()
        return env

    def _new_environment(
        self,
        architecture_id: str,
        *,
        settings: dict[str, object] | None = None,
    ) -> tuple[ProjectConfig, GoalReachEnv]:
        config, environment_type, extra_arguments = self._environment_specification(
            architecture_id,
            settings=settings,
        )
        env = self._instantiate_environment(
            architecture_id,
            config,
            environment_type,
            extra_arguments,
        )
        return config, env

    def _build_environment_update(
        self,
        generation: int,
        architecture_id: str,
        episode_index: int,
        config: ProjectConfig,
        environment_type: type[GoalReachEnv],
        extra_arguments: dict[str, object],
        static_changed: bool,
        initial_state=None,
    ) -> tuple[int, str, ProjectConfig, GoalReachEnv | None, bool]:
        if generation != self._environment_update_generation:
            return generation, architecture_id, config, None, static_changed
        env = self._instantiate_environment(
            architecture_id,
            config,
            environment_type,
            extra_arguments,
        )
        if generation != self._environment_update_generation:
            return generation, architecture_id, config, None, static_changed
        if initial_state is None:
            self._reset_viewer_environment(env, episode_index=episode_index)
        else:
            env.load_state_dict(initial_state)
        return generation, architecture_id, config, env, static_changed

    @staticmethod
    def _reset_viewer_environment(
        env: GoalReachEnv,
        *,
        episode_index: int | None = None,
    ) -> None:
        resolved_episode_index = (
            int(env.episode_counters[0]) + 1
            if episode_index is None
            else int(episode_index)
        )
        env.episode_counters[0] = resolved_episode_index - 1
        episode_seed = compose_episode_seed(
            env.base_seed,
            env.env_id_offset,
            resolved_episode_index,
        )
        env.reset(
            episode_seeds_cpu=torch.tensor(
                [episode_seed], dtype=torch.int64, device=env.device
            ),
            build_observation=False,
        )

    def _start_pending_environment_update(self) -> None:
        if (
            self._environment_update_future is not None
            or self._pending_environment_update is None
        ):
            return
        pending = self._pending_environment_update
        self._pending_environment_update = None
        (
            generation,
            architecture_id,
            episode_index,
            config,
            environment_type,
            extra_arguments,
            static_changed,
            initial_state,
        ) = pending
        self._environment_update_future_generation = generation
        try:
            self._environment_update_future = self._environment_update_executor.submit(
                self._build_environment_update, generation, architecture_id, episode_index,
                config, environment_type, extra_arguments, static_changed, initial_state)
        except Exception as error:
            self._environment_update_future_generation = None
            self._fail_environment_update('reset_submit', error, generation)

    def _queue_environment_update(
        self,
        architecture_id: str,
        settings: dict[str, object],
        *,
        static_changed: bool = False,
        episode_index: int | None = None,
        initial_state=None,
    ) -> None:
        config, environment_type, extra_arguments = self._environment_specification(
            architecture_id,
            settings=settings,
        )
        self._environment_update_generation += 1
        generation = self._environment_update_generation
        episode_index = int(self.env.episode_counters[0]) + 1 if episode_index is None else episode_index
        self._pending_environment_update = (
            generation,
            architecture_id,
            episode_index,
            config,
            environment_type,
            extra_arguments,
            static_changed,
            initial_state,
        )
        self._start_pending_environment_update()

    def _invalidate_environment_updates(self) -> None:
        self._environment_reset_request = None
        self._environment_update_generation += 1
        self._pending_environment_update = None
        self._pending_edited_scene_restore = None
        if self.reset_progress.get('phase') == 'preparing':
            self.reset_progress = dict(phase='superseded', generation=self._environment_update_generation)
            self._terminal_environment_update_pending = False
        future = self._environment_update_future
        if future is not None and future.cancel():
            self._environment_update_future = None
            self._environment_update_future_generation = None

    def _opponent_runtime_state(self) -> dict[str, object]:
        return {
            "actor_kind": self.opponent_actor_kind,
            "model_id": self.opponent_model_id,
            "architecture_id": self.opponent_architecture_id,
            "checkpoint": self.opponent_checkpoint,
            "update": self.opponent_update,
            "agent_frames": self.opponent_agent_frames,
            "selection": self.opponent_selection,
            "option_cache": self._opponent_option_cache,
            "model": self.opponent_model,
        }

    def _poll_environment_update(self) -> bool:
        with self._measure_performance_stage("environment_update_poll"):
            try:
                return self._poll_environment_update_unmeasured()
            except Exception as error:
                self._fail_environment_update('reset_commit', error, self._environment_update_generation)
                return True

    def _poll_environment_update_unmeasured(self) -> bool:
        if self.scene_mode != 'continuous':
            return False
        if getattr(self, '_closed', False):
            return False
        future = self._environment_update_future
        if (
            future is None
            or not future.done()
            or (
                self._terminal_environment_update_pending
                and self._terminal_snapshot_until > time.perf_counter()
            )
        ):
            return False
        submitted_generation = self._environment_update_future_generation
        self._environment_update_future = None
        self._environment_update_future_generation = None
        try:
            generation, architecture_id, config, env, static_changed = future.result()
        except Exception as error:
            if submitted_generation == self._environment_update_generation:
                self._pending_edited_scene_restore = None
                self._fail_environment_update('reset_prepare', error, submitted_generation)
                self._start_pending_environment_update()
                return True
            self._start_pending_environment_update()
            return False

        applied = False
        if (
            generation == self._environment_update_generation
            and architecture_id == self.active_actor_variant_id
            and env is not None
        ):
            learner_capacity = int(getattr(env, "viewer_learner_capacity", 1))
            defender_capacity = int(env.defender_count)
            opponent_state = self._opponent_runtime_state()
            if opponent_state["actor_kind"] == "learner":
                env.set_learner_architecture(str(opponent_state["architecture_id"]))
            else:
                env.set_patrol_architecture(str(opponent_state["architecture_id"]))
            capacities_changed = self._environment_capacity_signature(env) != self._environment_capacity_signature(self.env)
            self._environment_reset_backup = (generation, self._capture_environment_runtime())
            request = getattr(self, '_environment_reset_request', None)
            if request is not None and request['generation'] == generation:
                if request['exit_replay']:
                    self._replay_record = None
                    self._replay_frame = None
                    self._replay_error = None
                    for role, modes in self._live_control_modes.items():
                        self.role_controls[role].modes = list(modes)
                if request['target_mode'] is not None:
                    self._clear_scene_lock()
                    self._pending_target_mode = None
            episode_outcome_history = self.episode_outcome_history
            defender_episode_outcome_history = self.defender_episode_outcome_history
            match_summary_history = self.match_summary_history
            actor_inputs_revision = self._actor_inputs_revision
            self.config, self.env = config, env
            self.env.performance_recorder = self._record_performance_sample
            selected_capacity = env.role_state(self.selected_actor_kind).position.shape[1]
            self.selected_actor_index = min(self.selected_actor_index, selected_capacity-1)
            if capacities_changed:
                previous_controls = dict(self.role_controls)
                self._initialize_actor_runtime_state(
                    opponent_state=opponent_state
                )
                for role,previous_control in previous_controls.items():
                    if role in self.role_controls:
                        control = self.role_controls[role]
                        count = min(control.capacity, previous_control.capacity)
                        control.modes[:count] = previous_control.modes[:count]
                        control.manual_action[:count] = previous_control.manual_action[:count]
                        control.stochastic_neural = previous_control.stochastic_neural
                self.episode_outcome_history = episode_outcome_history
                self.defender_episode_outcome_history = defender_episode_outcome_history
                self.match_summary_history = match_summary_history
                self._actor_inputs_revision = actor_inputs_revision
                self._encoded_actor_inputs_revision = -1
            preserved_edit = self._pending_edited_scene_restore
            if (
                preserved_edit is not None
                and preserved_edit[0] == generation
            ):
                self._apply_preserved_edited_scene(preserved_edit[1])
                self._pending_edited_scene_restore = None
                self._edited_scene = True
            else:
                self._edited_scene = generation == self._edited_reset_generation
            self.edit_preview = None
            self._physics_ticks_into_decision = 0
            self._terminal_snapshot_until = 0.0
            self._terminal_environment_update_pending = False
            self._reset_current_learner_outcomes()
            self._reset_current_match_facts()
            self._remember_episode_initial_state()
            self._refresh_actor_frame()
            self._reset_trajectories()
            self._record_spawn()
            if getattr(self, '_viewer_fault', None):
                self.command_error = None
            self._viewer_fault = None
            if static_changed:
                self._mark_snapshot_static_dirty()
            self._next_physics_step_at = time.perf_counter()
            if self._resume_after_environment_update:
                self.running = True
                self._resume_after_environment_update = False
            self.reset_progress = dict(phase='ready', generation=generation)
            self._environment_reset_backup = None
            self._environment_reset_request = None
            applied = True
        self._start_pending_environment_update()
        return applied

    @staticmethod
    def _environment_capacity_signature(env):
        return (int(getattr(env, 'viewer_learner_capacity', 1)), int(env.defender_count),
                int(env.target_capacity), env.target_mode, tuple(env.policy_roles))

    def _capture_environment_runtime(self):
        # Explicit reset/scene-commit write set. Models, environments, records and
        # executors retain their references; only mutable presentation state is copied.
        names = (
            'env config role_controls control patrol_control target_control target_runtime '
            'model policy checkpoint metrics_path initial_checkpoint active_actor_variant_id '
            'opponent_actor_kind opponent_model_id opponent_architecture_id opponent_checkpoint '
            'opponent_update opponent_agent_frames opponent_selection _opponent_option_cache '
            'opponent_model opponent_policy _deployed_sources _deployment_errors '
            '_auto_deploy_training_sides _auto_deploy_training_run_group_id '
            'last_learner_action last_defender_action last_target_action last_action '
            'actor_frames observation actor_frame_physics_ticks encoded_actor_inputs '
            '_actor_inputs_revision _encoded_actor_inputs_revision _next_actor_inputs_encode_at '
            '_applied_frame_revision _applied_frame_cache _applied_frame_signature '
            '_applied_frame_actions _applied_frame_modes _applied_frame_tick '
            '_snapshot_static_revision _snapshot_static_dirty_deferred '
            '_viewer_episode_revision _viewer_live_attempt _policy_noise '
            '_current_episode_initial_state _locked_scene_state scene_locked scene_lock_message '
            '_replay_record _replay_frame _replay_error _replay_index _live_record _last_complete_record '
            '_live_scene_cursor _live_control_modes current_learner_outcomes current_match_event_counts '
            'episode_outcome_history defender_episode_outcome_history match_summary_history '
            'trajectories trajectory_sample_physics_ticks trajectory_participants '
            '_trajectory_actor_was_active event_freeze_frames _event_freeze_frame_keys spawn '
            'selected_actor_kind selected_actor_index edit_preview _edited_scene _edited_reset_generation '
            '_pending_target_mode _pending_edited_scene_restore _physics_ticks_into_decision '
            '_terminal_snapshot_until _terminal_environment_update_pending '
            '_next_physics_step_at _physics_rate_reset_requested viewer_settings '
            'time_scale snapshot_rate_hz rate_revision command_error _viewer_fault'
        ).split()
        copied = set(('last_learner_action last_defender_action last_target_action '
            'encoded_actor_inputs _deployed_sources _deployment_errors _auto_deploy_training_sides '
            '_live_control_modes current_learner_outcomes current_match_event_counts '
            'episode_outcome_history defender_episode_outcome_history match_summary_history '
            'trajectories trajectory_sample_physics_ticks trajectory_participants '
            '_trajectory_actor_was_active event_freeze_frames _event_freeze_frame_keys '
            'viewer_settings').split())
        values = {name:copy.deepcopy(getattr(self,name)) if name in copied else getattr(self,name)
                  for name in names if hasattr(self,name)}
        if 'target_runtime' in values:
            values['target_runtime'] = dict(values['target_runtime'])
        return dict(values=values, absent=set(names)-values.keys(),
                    modes={role:list(control.modes) for role,control in self.role_controls.items()},
                    rng=torch.get_rng_state())

    def _restore_environment_runtime(self, state):
        for name in state['absent']:
            self.__dict__.pop(name, None)
        for name,value in state['values'].items():
            setattr(self,name,value)
        for role,modes in state['modes'].items():
            self.role_controls[role].modes = modes
        self.last_action = self.last_learner_action
        torch.set_rng_state(state['rng'])

    def _restore_environment_update_backup(self):
        backup = getattr(self, '_environment_reset_backup', None)
        self._environment_reset_backup = None
        if backup is not None:
            self._restore_environment_runtime(backup[1])

    def _fail_environment_update(self, stage, error, generation):
        restore_failure = None
        try:
            self._restore_environment_update_backup()
        except Exception as restore_error:
            # Keep the original failure as the actionable first error.
            restore_failure = str(restore_error)
        self._pending_environment_update = None
        self._pending_edited_scene_restore = None
        self._environment_reset_request = None
        self._viewer_failure(stage, error)
        self.reset_progress['generation'] = generation
        if restore_failure is not None:
            self.reset_progress['restore_error'] = restore_failure

    def _project_edit(self, command: dict[str, object]) -> dict[str, object]:
        kind = str(command.get("edit_kind", ""))
        mode = str(command.get("mode", "replace"))
        raw = (
            float(command.get("x", math.nan)),
            float(command.get("z", math.nan)),
        )
        if not all(math.isfinite(value) for value in raw):
            raise ValueError("GoalReach edit position must be finite")

        if kind == "target":
            if "target" not in self.env.physical_roles:
                raise ValueError("Static mode has no TGT to edit")
            index=int(command.get('index',command.get('source_id',0)))
            if not 0<=index<int(self.env.sampled_target_count[0]):raise ValueError('Unknown target slot')
            heading = float(command.get("heading", float(self.env.targets.heading[0, index])))
            if not math.isfinite(heading):
                raise ValueError("TGT heading must be finite")
            return dict(kind=kind, mode=mode, id=index, source_id=index, raw_x=raw[0], raw_z=raw[1],
                        x=raw[0], z=raw[1], h=heading, valid=True, reason=None, preview=True)
        obstacle_centers, obstacle_radii = self._obstacle_geometry()
        goal_centers, goal_radii = self._goal_geometry()
        learner_positions = self.env._viewer_learner_state(0)[0]
        learner_active = self.env.viewer_learner_active(0)
        defender_positions, defender_headings = self._defender_geometry()

        if kind == "zone_radius":
            zone = str(command.get("zone", command.get("index", "")))
            setting = SAFETY_ZONE_EDIT_SETTINGS.get(zone)
            if setting is None:
                raise ValueError("Safety-zone radius is unsupported")
            target = self.env.region_reference_position[0].tolist()
            requested_radius = math.dist(target, raw)
            requested_fraction = requested_radius / float(self.env.world.radius)
            reason: str | None = None
            try:
                _validated_viewer_settings(
                    {
                        **self.viewer_settings,
                        setting: requested_fraction,
                    }
                )
            except ValueError as error:
                reason = str(error)
            return {
                "kind": kind,
                "mode": "radius",
                "zone": zone,
                "source_id": zone,
                "raw_x": raw[0],
                "raw_z": raw[1],
                "x": target[0],
                "z": target[1],
                "radius": requested_radius,
                "radius_fraction": requested_fraction,
                "valid": reason is None,
                "reason": reason,
            }

        if kind == "goal":
            source_index = int(command.get("index", 0))
            duplicate = bool(command.get("duplicate", False))
            if not 0 <= source_index < len(goal_centers):
                raise ValueError("Goal index is inactive")
            current_position = goal_centers[source_index]
            current_radius = goal_radii[source_index]
            if mode == "radius":
                requested_radius = math.dist(current_position, raw)
            elif mode == "replace":
                requested_radius = float(command.get("radius", current_radius))
            else:
                requested_radius = current_radius
            if not math.isfinite(requested_radius) or requested_radius <= 0.0:
                raise ValueError("Goal radius must be finite and positive")
            requested_position = (
                raw if mode in {"position", "duplicate", "replace"} else current_position
            )
            return {
                "kind": kind,
                "mode": mode,
                "id": len(goal_centers) if duplicate else source_index,
                "source_id": source_index,
                "duplicate": duplicate,
                "raw_x": raw[0],
                "raw_z": raw[1],
                "x": requested_position[0],
                "z": requested_position[1],
                "radius": requested_radius,
                "valid": True,
                "reason": None,
            }

        if kind == "learner":
            index = int(command.get("index", 0))
            if not 0 <= index < learner_active.shape[0] or not bool(learner_active[index]):
                raise ValueError("Selected infiltrator is not active")
            current_position = tuple(learner_positions[index].tolist())
            current_heading = float(self.env._viewer_learner_state(0)[1][index])
            requested_position = raw if mode in {"position", "replace"} else current_position
            requested_heading = (
                math.atan2(raw[0] - current_position[0], raw[1] - current_position[1])
                if mode == "heading"
                else math.remainder(float(command.get("heading", current_heading)), 2.0 * math.pi)
                if mode == "replace"
                else current_heading
            )
            if not math.isfinite(requested_heading):
                raise ValueError("Actor heading must be finite")
            return {
                "kind": kind,
                "mode": mode,
                "source_id": index,
                "raw_x": raw[0],
                "raw_z": raw[1],
                "x": requested_position[0],
                "z": requested_position[1],
                "h": requested_heading,
                "valid": True,
                "reason": None,
            }

        return self._project_non_goal_edit(
            command,
            kind=kind,
            mode=mode,
            raw=raw,
            obstacle_centers=obstacle_centers,
            obstacle_radii=obstacle_radii,
            defender_positions=defender_positions,
            defender_headings=defender_headings,
        )

    def _project_non_goal_edit(
        self,
        command: dict[str, object],
        *,
        kind: str,
        mode: str,
        raw: tuple[float, float],
        obstacle_centers: list[tuple[float, float]],
        obstacle_radii: list[float],
        defender_positions: list[tuple[float, float]],
        defender_headings: list[float],
    ) -> dict[str, object]:
        if kind == "obstacle":
            source_index, source_key = self._resolve_obstacle_source(command.get("index", ""))
            duplicate = bool(command.get("duplicate", False))
            current_position = obstacle_centers[source_index]
            current_radius = obstacle_radii[source_index]
            if mode == "radius":
                requested_radius = math.dist(current_position, raw)
            elif mode == "replace":
                requested_radius = float(command.get("radius", current_radius))
            else:
                requested_radius = current_radius
            if not math.isfinite(requested_radius) or requested_radius <= 0.0:
                raise ValueError("Obstacle radius must be finite and positive")
            requested_position = (
                raw if mode in {"position", "duplicate", "replace"} else current_position
            )
            if not self._viewer_point_is_sensed(current_position, radius=current_radius):
                raise ValueError("Obstacle is outside the current sensed view")
            if mode in {"position", "duplicate", "replace"} and not self._viewer_point_is_sensed(raw):
                raise ValueError("Obstacle edit point is outside the current sensed view")
            return {
                "kind": kind,
                "mode": mode,
                "id": source_key,
                "source_id": source_key,
                "duplicate": duplicate,
                "raw_x": raw[0],
                "raw_z": raw[1],
                "x": requested_position[0],
                "z": requested_position[1],
                "radius": requested_radius,
                "valid": True,
                "reason": None,
            }

        if kind == "defender":
            source_index = int(command.get("index", 0))
            if not 0 <= source_index < len(defender_positions):
                raise ValueError("GoalReach patrol craft index is not active")
            current_position = defender_positions[source_index]
            current_heading = defender_headings[source_index]
            requested_position = (
                raw if mode in {"position", "replace"} else current_position
            )
            direction = math.atan2(raw[0] - current_position[0], raw[1] - current_position[1])
            requested_heading = (
                direction
                if mode == "heading"
                else math.remainder(float(command.get("heading", current_heading)), 2.0 * math.pi)
                if mode == "replace"
                else current_heading
            )
            if not math.isfinite(requested_heading):
                raise ValueError("DEF heading must be finite")
            return {
                "kind": kind,
                "mode": mode,
                "id": source_index,
                "source_id": source_index,
                "raw_x": raw[0],
                "raw_z": raw[1],
                "x": requested_position[0],
                "z": requested_position[1],
                "h": requested_heading,
                "state": "patrol",
                "guide_target_m": None,
                "collision_radius_m": self.env.vessel_radius_m,
                "valid": True,
                "reason": None,
                "preview": True,
            }

        if kind == "waypoint":
            source_index = int(command.get("index", 0))
            if not 0 <= source_index < self.env.defender_count or not bool(
                self.env.defender_active[0, source_index]
            ):
                raise ValueError("GoalReach patrol craft index is not active")
            current_position = tuple(self.env.patrol_waypoint[0, source_index].tolist())
            return {
                "kind": kind,
                "mode": mode,
                "id": source_index,
                "source_id": source_index,
                "raw_x": raw[0],
                "raw_z": raw[1],
                "x": raw[0],
                "z": raw[1],
                "valid": True,
                "reason": None,
            }

        raise ValueError(f"Unsupported GoalReach edit kind: {kind}")

    def _commit_edit(self, command: dict[str, object]) -> None:
        preview = self._project_edit(command)
        if not bool(preview["valid"]):
            raise ValueError(str(preview["reason"] or "GoalReach edit has no feasible candidate"))
        kind = str(preview["kind"])
        if kind in {"goal", "obstacle"}:
            centers, radii = self._circle_geometry(kind)
            if bool(preview["duplicate"]):
                centers.append((float(preview["x"]), float(preview["z"])))
                radii.append(float(preview["radius"]))
            else:
                index = (
                    self._resolve_obstacle_source(preview["source_id"])[0]
                    if kind == "obstacle"
                    else int(preview["source_id"])
                )
                centers[index] = (float(preview["x"]), float(preview["z"]))
                radii[index] = float(preview["radius"])
            if kind == "obstacle":
                learner_active, defender_active = self._edited_scene_actor_active_masks()
                self.env.validate_fixed_actor_poses(
                    learner_positions=self.env.learners.position[0],
                    learner_active=learner_active,
                    defender_positions=self.env.defenders.position[0],
                    defender_active=defender_active,
                    target_positions=self.env.targets.position[0],
                    goal_centers=self._goal_geometry()[0],
                    goal_radii=self._goal_geometry()[1],
                    obstacle_centers=centers,
                    obstacle_radii=radii,
                )
                source_key = str(preview["source_id"])
                self.env.edit_terrain_obstacle(
                    0,
                    source_key,
                    center=(float(preview["x"]), float(preview["z"])),
                    radius=float(preview["radius"]),
                    duplicate=bool(preview["duplicate"]),
                )
                self.env.restart_edited_episode(torch.ones(1, dtype=torch.bool, device=self.device))
                self._finish_scene_edit()
            else:
                self._reset_edited_scene(goal_centers=centers, goal_radii=radii)
        elif kind == "zone_radius":
            zone = str(preview["zone"])
            setting = SAFETY_ZONE_EDIT_SETTINGS[zone]
            preserved_scene = self._capture_edited_scene()
            previous_generation = self._environment_update_generation
            self._set_viewer_settings(
                {
                    setting: float(preview["radius_fraction"]),
                }
            )
            if self._environment_update_generation != previous_generation:
                self._pending_edited_scene_restore = (
                    self._environment_update_generation,
                    preserved_scene,
                )
        elif kind == "learner":
            index = int(preview["source_id"])
            learner_positions = self.env._viewer_learner_state(0)[0].clone()
            learner_positions[index] = torch.tensor(
                (float(preview["x"]), float(preview["z"])),
                dtype=torch.float32,
                device=self.device,
            )
            defender_positions, _ = self._defender_geometry()
            learner_active, defender_active = self._edited_scene_actor_active_masks()
            self.env.validate_fixed_actor_poses(
                learner_positions=learner_positions,
                learner_active=learner_active,
                defender_positions=torch.tensor(
                    defender_positions, dtype=torch.float32, device=self.device
                ).reshape(-1, 2),
                defender_active=defender_active,
                goal_centers=self._goal_geometry()[0],
                goal_radii=self._goal_geometry()[1],
                obstacle_centers=self._obstacle_geometry()[0],
                obstacle_radii=self._obstacle_geometry()[1],
            )
            self.env.set_viewer_learner_pose(
                0,
                index,
                (float(preview["x"]), float(preview["z"])),
                float(preview["h"]),
            )
            mask = torch.ones(1, dtype=torch.bool, device=self.device)
            self.env.restart_edited_episode(mask)
            self._finish_scene_edit()
        elif kind == "defender":
            positions, headings = self._defender_geometry()
            candidate_position = (float(preview["x"]), float(preview["z"]))
            index = int(preview["source_id"])
            positions[index] = candidate_position
            headings[index] = float(preview["h"])
            self._reset_edited_scene(
                defender_positions=positions,
                defender_headings=headings,
            )
        elif kind == "target":
            index=int(preview['source_id'])
            candidate = self.env.targets.position[0].clone()
            candidate[index]=torch.tensor([preview['x'],preview['z']],device=self.device)
            learner_active, defender_active = self._edited_scene_actor_active_masks()
            self.env.validate_fixed_actor_poses(learner_positions=self.env.learners.position[0], learner_active=learner_active,
                defender_positions=self.env.defenders.position[0], defender_active=defender_active, target_positions=candidate,
                goal_centers=self._goal_geometry()[0], goal_radii=self._goal_geometry()[1],
                obstacle_centers=self._obstacle_geometry()[0], obstacle_radii=self._obstacle_geometry()[1])
            self.env.targets.position[0] = candidate
            self.env.targets.heading[0, index] = preview["h"]
            self.env.restart_edited_episode(torch.ones(1, dtype=torch.bool, device=self.device))
            self._finish_scene_edit()
        elif kind == "waypoint":
            index = int(preview["source_id"])
            self.env.patrol_waypoint[0, index] = torch.tensor(
                (float(preview["x"]), float(preview["z"])),
                dtype=torch.float32,
                device=self.device,
            )
            self.env.patrol_hold_ticks[0, index] = 0
            self.edit_preview = None
        else:
            raise ValueError(f"Unsupported GoalReach edit kind: {kind}")

    def _set_edit_preview(self, command: dict[str, object]) -> None:
        preview = self._project_edit(command)
        self.edit_preview = preview

    def _clear_edit_preview(self) -> None:
        self.edit_preview = None

    def _finish_scene_edit(self) -> None:
        self.edit_preview = None
        self._edited_scene = True
        self._reset_current_learner_outcomes()
        self._reset_current_match_facts()
        self._remember_episode_initial_state()
        self._refresh_actor_frame()
        self._reset_trajectories()
        self._record_spawn()

    def _edited_scene_actor_active_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        learner_active = self.env.viewer_learner_active(0).clone()
        if hasattr(self.env, "sampled_learner_count"):
            learner_active = torch.arange(
                learner_active.shape[0], device=self.device
            ) < int(self.env.sampled_learner_count[0])
        defender_active = self.env.defender_active[0].clone()
        if hasattr(self.env, "sampled_defender_count"):
            defender_active = torch.arange(
                defender_active.shape[0], device=self.device
            ) < int(self.env.sampled_defender_count[0])
        return learner_active, defender_active

    def _capture_edited_scene(self) -> dict[str, object]:
        learner_positions, learner_headings, *_ = self.env._viewer_learner_state(0)
        defender_positions, defender_headings = self._defender_geometry()
        goal_centers, goal_radii = self._goal_geometry()
        obstacle_centers, obstacle_radii = self._obstacle_geometry()
        learner_active, defender_active = self._edited_scene_actor_active_masks()
        return {
            "reference_position": self.env.region_reference_position[0].tolist(),
            "target_positions": self.env.targets.position[0].tolist() if "target" in self.env.physical_roles else None,
            "target_headings": self.env.targets.heading[0].tolist() if "target" in self.env.physical_roles else None,
            'target_count':int(self.env.sampled_target_count[0]),
            "goal_centers": goal_centers,
            "goal_radii": goal_radii,
            "obstacle_centers": obstacle_centers,
            "obstacle_radii": obstacle_radii,
            "learner_positions": learner_positions.detach().cpu().tolist(),
            "learner_headings": learner_headings.detach().cpu().tolist(),
            "learner_count": int(learner_active.sum()),
            "defender_positions": defender_positions,
            "defender_headings": defender_headings,
            "defender_count": int(defender_active.sum()),
        }

    def _apply_preserved_edited_scene(self, state: dict[str, object], *, env=None) -> None:
        env = self.env if env is None else env
        env.region_reference_position[0] = torch.tensor(state["reference_position"], device=self.device)
        if "target" in env.physical_roles:
            env.sampled_target_count[0]=state.get('target_count',1)
            env.targets.position[0] = torch.tensor(state["target_positions"], device=self.device)
            env.targets.heading[0] = torch.tensor(state["target_headings"], device=self.device)
        goal_centers = state["goal_centers"]
        goal_radii = state["goal_radii"]
        obstacle_centers = state["obstacle_centers"]
        obstacle_radii = state["obstacle_radii"]
        learner_positions = torch.tensor(
            state["learner_positions"], dtype=torch.float32, device=self.device
        ).reshape(-1, 2)
        learner_headings = torch.tensor(
            state["learner_headings"], dtype=torch.float32, device=self.device
        ).reshape(-1)
        defender_positions = torch.tensor(
            state["defender_positions"], dtype=torch.float32, device=self.device
        ).reshape(-1, 2)
        defender_headings = torch.tensor(
            state["defender_headings"], dtype=torch.float32, device=self.device
        ).reshape(-1)
        learner_count = int(state["learner_count"])
        defender_count = int(state["defender_count"])
        learner_active = torch.arange(
            learner_positions.shape[0], device=self.device
        ) < learner_count
        defender_active = torch.arange(
            defender_positions.shape[0], device=self.device
        ) < defender_count
        env.validate_fixed_actor_poses(
            learner_positions=learner_positions,
            learner_active=learner_active,
            defender_positions=defender_positions,
            defender_active=defender_active,
            goal_centers=goal_centers,
            goal_radii=goal_radii,
            obstacle_centers=obstacle_centers,
            obstacle_radii=obstacle_radii,
        )
        mask = torch.ones(1, dtype=torch.bool, device=self.device)
        env.set_fixed_circle_geometry(
            mask,
            goal_centers=goal_centers,
            goal_radii=goal_radii,
            obstacle_centers=obstacle_centers,
            obstacle_radii=obstacle_radii,
        )
        if hasattr(env, "sampled_learner_count"):
            env.sampled_learner_count[0] = learner_count
        for index in range(learner_count):
            env.set_viewer_learner_pose(
                0,
                index,
                tuple(learner_positions[index].tolist()),
                float(learner_headings[index]),
            )
        if hasattr(env, "sampled_defender_count"):
            env.sampled_defender_count[0] = defender_count
        if env.defender_count:
            env.defenders.position[0] = defender_positions
            env.defenders.heading[0] = defender_headings
            env.defender_initial_position[0] = defender_positions
            env.defender_initial_heading[0] = defender_headings
        env.restart_edited_episode(mask)

    def _prepare_settings_scene(self, settings, deployed, initial_state, target_only, episode_index, clock_only=False):
        architecture = _unified_match_architecture_id(settings, 'learner')
        deployments = []
        sources = {}
        for role in policy_roles(settings):
            previous = deployed.get(role)
            if role == 'learner' and settings['learner_control_mode'] == 'rule':
                deployment = self._prepare_rule_deployment(self._deployment_generations[role], architecture)
            elif (previous is not None and previous.architecture_id == architecture
                    and (previous.payload is None or all(
                        settings.get(key) == self.viewer_settings.get(key)
                        for key in ("terminate_on_any_casualty", "terminate_on_defender_elimination",
                                    "include_remaining_time_observation", "reward_objective"))
                         and settings["training_settings"] == self.viewer_settings["training_settings"])):
                deployment = previous
            else:
                source = self._deployed_sources.get(role) if previous is None else None
                if source and source['architecture_id'] != architecture:
                    source = None
                path = Path(source['checkpoint_path']) if source else None
                model_id = ('saved_target:' + path.relative_to(self.config.repository_root).as_posix()) if path else 'random_current_architecture'
                deployment = self._prepare_viewer_deployment(self._deployment_generations[role], role, model_id,
                    architecture, 'checkpoint' if path else 'random', None, path,
                    'manual' if path else 'random', None, False, False, prepared_sources=sources)
                if source and deployment.payload['source_metadata'] != source:
                    raise ValueError(f"Selected U{source['checkpoint_global_update']} unavailable")
            deployments.append(deployment)
            settings['deployment_model_ids'][role] = ('saved_target:' + Path(deployment.payload['source_metadata']['checkpoint_path']).relative_to(self.config.repository_root).as_posix()) if deployment.payload and deployment.payload.get('source_metadata') else deployment.model_id
        config, environment = self._new_environment(architecture, settings=settings)
        if target_only and initial_state is not None:
            self._apply_scene_task_settings(initial_state, settings)
            environment.load_state_dict(initial_state)
        else:
            self._reset_viewer_environment(environment, episode_index=episode_index)
        return {'actor_kind': settings['selected_actor_kind'], 'architecture_id': architecture,
            'settings': settings, 'config': config, 'environment': environment,
            'deployments': deployments, 'warning': None, 'settings_request': True, 'configuration_changed': not clock_only}

    def _set_viewer_settings(self, command):
        previous_rule_runtime = (self._capture_environment_runtime()
            if {'learner_control_mode', 'learner_rule'} & command.keys() else None)
        previous_future = getattr(self, "_training_scene_load_future", None)
        receipt = dict(client_request_id=command.get("client_request_id"),
                       viewer_instance=command.get("_viewer_instance"), phase="preparing", error=None)
        try:
            self._apply_viewer_settings_command(command)
        except Exception as error:
            if previous_rule_runtime is not None:
                self._restore_environment_runtime(previous_rule_runtime)
                _store_viewer_settings(self.viewer_settings_path, self.viewer_settings)
            receipt.update(phase="failed", error=str(error))
            raise
        else:
            if (getattr(self, "_requested_viewer_settings", None) is None
                    or getattr(self, "_training_scene_load_future", None) is previous_future):
                receipt["phase"] = "applied"
        finally:
            self._viewer_settings_result = receipt
            self._mark_snapshot_status_dirty()

    def _discard_settings_candidate(self):
        if getattr(self, '_requested_viewer_settings', None) is None:
            return
        self._training_scene_load_future.cancel()
        self._training_scene_load_future = None
        self._requested_viewer_settings = None
        self._training_scene_request = None
        self._invalidate_environment_updates()

    def _apply_viewer_settings_command(self, command):
        previous = self.viewer_settings
        base = getattr(self, '_requested_viewer_settings', None) or previous
        requested = {key:value for key,value in command.items()
                     if key not in {'type', 'client_request_id', '_viewer_instance'}}
        if 'training_settings' in requested:
            requested['training_settings'] = {**base['training_settings'], **requested['training_settings']}
        candidate = {**base, **requested}
        if requested.get('joint_training_mode') == 'joint' and not requested.get('population'):
            candidate.pop('population', None)
        settings = _validated_viewer_settings(candidate)
        changed = {key for key,value in settings.items() if value != previous.get(key)}
        reset_request = (copy.deepcopy(getattr(self, '_environment_reset_request', None))
                         if getattr(self, '_requested_viewer_settings', None) is not None
                         and self._terminal_environment_update_pending else None)
        if not changed:
            self._discard_settings_candidate()
            if reset_request is not None:
                self._reset(episode_index=reset_request['episode_index'],
                            initial_state=reset_request['initial_state'])
            return
        architecture = _unified_match_architecture_id(settings, 'learner')
        old_config, _, old_arguments = self._environment_specification(self.active_actor_variant_id, settings=copy.deepcopy(previous))
        config, _, arguments = self._environment_specification(architecture, settings=settings)
        environment_changed = (old_config.shared != config.shared or old_config.goal_reach != config.goal_reach or old_arguments != arguments
                                   or settings['training_settings']['time_limit_boundary'] != previous['training_settings']['time_limit_boundary'])
        clock_only = changed <= {'training_settings'}
        if (environment_changed and architecture == self.active_actor_variant_id
                and 'training_settings' in changed):
            without_clock = copy.deepcopy(settings)
            without_clock['training_settings'] = copy.deepcopy(previous['training_settings'])
            non_clock_config, _, non_clock_arguments = self._environment_specification(
                architecture, settings=without_clock
            )
            clock_only = (old_config.shared == non_clock_config.shared
                          and old_config.goal_reach == non_clock_config.goal_reach
                          and old_arguments == non_clock_arguments)
        contract_only = changed <= {'training_settings', 'terminate_on_any_casualty',
            'terminate_on_defender_elimination', 'include_remaining_time_observation',
            'viewer_apply_sampling_limit', 'reward_objective', 'active_actor_variant_id'}
        if (getattr(self, "_replay_record", None) is None
                and not contract_only
                and not clock_only
                and (architecture != self.active_actor_variant_id or environment_changed)):
            settings = _validated_viewer_settings(
                _migrated_live_viewer_settings_candidate(settings)
            )
            changed = {key for key,value in settings.items() if value != previous.get(key)}
            architecture = _unified_match_architecture_id(settings, 'learner')
            config, _, arguments = self._environment_specification(architecture, settings=settings)
            environment_changed = (old_config.shared != config.shared
                                   or old_config.goal_reach != config.goal_reach
                                   or old_arguments != arguments
                                   or settings['training_settings']['time_limit_boundary'] != previous['training_settings']['time_limit_boundary'])
            clock_only = False
        if architecture != self.active_actor_variant_id or environment_changed:
            # Requested values never become applied/persisted until this entire bundle succeeds.
            self._queue_training_scene_load(None, preserve_follow=True)
            deployed = {}
            for role in self.env.policy_roles:
                descriptor = self._role_deployment_descriptor(role, neural_state=True)
                model = self.role_controls[role].neural_policy.model
                source = self._deployed_sources.get(role)
                path = Path(source['checkpoint_path']) if source else None
                payload = None if source is None else {'source_metadata':copy.deepcopy(source),
                    'architecture_id':descriptor['architecture_id'], 'update_index':descriptor['update'],
                    'agent_frames':descriptor['agent_frames'], 'model':model.state_dict()}
                deployed[role] = _PreparedViewerDeployment(self._deployment_generations[role], role,
                    descriptor['model_id'], descriptor['architecture_id'], 'checkpoint' if source else 'random',
                    'manual' if source else 'random', model, payload, path,
                    path.stat().st_mtime_ns if path and path.is_file() else None, None, False, False)
            initial = copy.deepcopy((self._current_episode_initial_state or {}).get('environment'))
            if contract_only and self.scene_locked:
                initial = copy.deepcopy(self._locked_scene_state)
            self._requested_viewer_settings = settings
            self._training_scene_load_future = self._checkpoint_load_executor.submit(self._prepare_settings_scene,
                settings, deployed, initial, changed <= {'target_mode'} or (contract_only and self.scene_locked),
                int(self.env.episode_counters[0])+1, contract_only)
            if reset_request is not None:
                self._reset(episode_index=reset_request['episode_index'],
                            initial_state=reset_request['initial_state'])
            self._mark_snapshot_status_dirty()
            return
        # Training-only values affect the next explicit launch; display/control values update live controls.
        _store_viewer_settings(self.viewer_settings_path, settings)
        self._discard_settings_candidate()
        self.viewer_settings = settings
        if settings['learner_control_mode'] == 'rule' and ('learner_control_mode' in changed or 'learner_rule' in changed):
            self._invalidate_rule_neural_candidates()
        elif previous['learner_control_mode'] == 'rule' and settings['learner_control_mode'] != 'rule':
            self._discard_training_follow_side('learner')
        self.selected_actor_kind = str(settings['selected_actor_kind'])
        self.time_scale = float(settings['time_scale'])
        self.snapshot_rate_hz = int(settings['snapshot_rate_hz'])
        for role,control in self.role_controls.items():
            control.stochastic_neural = settings['policy_sampling'] == 'stochastic'
            for index in range(control.capacity):
                control.set_mode(index, settings[role+'_control_mode'])
        if ({'learner_control_mode', 'defender_control_mode', 'target_control_mode', 'learner_rule'} & changed
                and not self._actor_frame_updates_deferred()):
            self._refresh_actor_frame()
        self.rate_revision += 1
        self._mark_snapshot_status_dirty()
        self._mark_snapshot_static_dirty()

        if reset_request is not None:
            self._reset(episode_index=reset_request['episode_index'],
                        initial_state=reset_request['initial_state'])

    def _set_learner(self, command: dict[str, object]) -> None:
        self._commit_edit({**command, "edit_kind": "learner", "mode": "replace"})

    def _obstacle_geometry(self) -> tuple[list[tuple[float, float]], list[float]]:
        active_indices = (
            torch.nonzero(self.env.obstacle_active[0], as_tuple=False).flatten().tolist()
        )
        return (
            [tuple(self.env.obstacle_centers[0, index].tolist()) for index in active_indices],
            [float(self.env.obstacle_radii[0, index]) for index in active_indices],
        )

    def _obstacle_geometry_with_keys(
        self,
    ) -> tuple[list[tuple[float, float]], list[float], list[str]]:
        active_indices = torch.nonzero(
            self.env.obstacle_active[0], as_tuple=False
        ).flatten().tolist()
        return (
            [tuple(self.env.obstacle_centers[0, index].tolist()) for index in active_indices],
            [float(self.env.obstacle_radii[0, index]) for index in active_indices],
            [self.env.terrain_key_text(self.env.obstacle_keys[0, index]) for index in active_indices],
        )

    def _resolve_obstacle_source(self, value: object) -> tuple[int, str]:
        centers, _, keys = self._obstacle_geometry_with_keys()
        if isinstance(value, str):
            try:
                return keys.index(value), value
            except ValueError as error:
                raise ValueError("GoalReach obstacle key is not active") from error
        index = int(value)
        if not 0 <= index < len(centers):
            raise ValueError("GoalReach obstacle index is not active")
        return index, keys[index]

    def _viewer_sensing_centers(self) -> list[tuple[float, float]]:
        if self.viewer_settings.get("main_view_frame") in {"union", "union_follow"}:
            centers: list[tuple[float, float]] = []
            for role in ("learner", "defender", "target"):
                state = self.env.role_state(role)
                active = getattr(self.env, role + "_active")[0]
                centers.extend(
                    tuple(state.position[0, index].tolist())
                    for index in torch.nonzero(active, as_tuple=False).flatten().tolist()
                )
            return centers
        role = self.selected_actor_kind
        index = self.selected_actor_index
        active = getattr(self.env, role + "_active")[0]
        if not 0 <= index < active.shape[0] or not bool(active[index]):
            return []
        return [tuple(self.env.role_state(role).position[0, index].tolist())]

    def _viewer_point_is_sensed(self, point: tuple[float, float], *, radius: float = 0.0) -> bool:
        return any(
            math.dist(point, center) <= self.env.world.radius + float(radius) + 1.0e-7
            for center in self._viewer_sensing_centers()
        )

    def _goal_geometry(self) -> tuple[list[tuple[float, float]], list[float]]:
        active_indices = (
            torch.nonzero(self.env.goal_active[0], as_tuple=False).flatten().tolist()
        )
        return (
            [tuple(self.env.goal_centers[0, index].tolist()) for index in active_indices],
            [float(self.env.goal_radii[0, index]) for index in active_indices],
        )

    def _circle_geometry(
        self, kind: str
    ) -> tuple[list[tuple[float, float]], list[float]]:
        if kind == "goal":
            return self._goal_geometry()
        if kind == "obstacle":
            return self._obstacle_geometry()
        raise ValueError("Circle type must be goal or obstacle")

    def _delete_circle(self, command: dict[str, object]) -> None:
        kind = str(command.get("edit_kind", ""))
        if kind == "obstacle":
            _, key = self._resolve_obstacle_source(command.get("index", ""))
            self.env.edit_terrain_obstacle(0, key, delete=True)
            self.env.restart_edited_episode(torch.ones(1, dtype=torch.bool, device=self.device))
            self._finish_scene_edit()
            return
        index = int(command.get("index", -1))
        centers, radii = self._circle_geometry(kind)
        if not 0 <= index < len(centers):
            raise ValueError("Circle index is inactive")
        del centers[index]
        del radii[index]
        geometry = (
            {"goal_centers": centers, "goal_radii": radii}
            if kind == "goal"
            else {"obstacle_centers": centers, "obstacle_radii": radii}
        )
        self._reset_edited_scene(**geometry)

    def _defender_geometry(
        self,
    ) -> tuple[list[tuple[float, float]], list[float]]:
        scene_indices = range(self.env.defender_count)
        return (
            [
                tuple(
                    (
                        self.env.defenders.position[0, index]
                        if bool(self.env.defender_active[0, index])
                        else self.env.defender_initial_position[0, index]
                    ).tolist()
                )
                for index in scene_indices
            ],
            [
                float(
                    self.env.defenders.heading[0, index]
                    if bool(self.env.defender_active[0, index])
                    else self.env.defender_initial_heading[0, index]
                )
                for index in scene_indices
            ],
        )

    def _set_defender(self, command: dict[str, object]) -> None:
        self._commit_edit({**command, "edit_kind": "defender", "mode": "replace"})

    def _set_patrol_waypoint(self, command: dict[str, object]) -> None:
        self._commit_edit({**command, "edit_kind": "waypoint", "mode": "replace"})

    def _reset_edited_scene(
        self,
        *,
        goal_centers: list[tuple[float, float]] | None = None,
        goal_radii: list[float] | None = None,
        learner: tuple[float, float] | None = None,
        heading: float | None = None,
        obstacle_centers: list[tuple[float, float]] | None = None,
        obstacle_radii: list[float] | None = None,
        defender_positions: list[tuple[float, float]] | None = None,
        defender_headings: list[float] | None = None,
    ) -> None:
        current_goal_centers, current_goal_radii = self._goal_geometry()
        resolved_goal_centers = (
            current_goal_centers if goal_centers is None else goal_centers
        )
        resolved_goal_radii = current_goal_radii if goal_radii is None else goal_radii
        resolved_learner = (
            learner if learner is not None else tuple(self.env.learner.position[0].tolist())
        )
        resolved_heading = (
            float(heading) if heading is not None else float(self.env.learner.heading[0])
        )
        current_obstacle_centers, current_obstacle_radii = self._obstacle_geometry()
        resolved_obstacle_centers = (
            current_obstacle_centers if obstacle_centers is None else obstacle_centers
        )
        resolved_obstacle_radii = (
            current_obstacle_radii if obstacle_radii is None else obstacle_radii
        )
        current_defender_positions, current_defender_headings = (
            self._defender_geometry()
        )
        resolved_defender_positions = (
            current_defender_positions if defender_positions is None else defender_positions
        )
        resolved_defender_headings = (
            current_defender_headings if defender_headings is None else defender_headings
        )
        for label, centers, radii in (
            ("Goal", resolved_goal_centers, resolved_goal_radii),
            ("Obstacle", resolved_obstacle_centers, resolved_obstacle_radii),
        ):
            if len(centers) != len(radii):
                raise ValueError(f"{label} edit data is incomplete")
            if any(
                not all(math.isfinite(value) for value in center)
                or not math.isfinite(radius)
                or radius <= 0.0
                for center, radius in zip(centers, radii, strict=True)
            ):
                raise ValueError(f"{label} position must be finite and radius positive")
        if not all(math.isfinite(value) for value in (*resolved_learner, resolved_heading)):
            raise ValueError("Actor poses must be finite")
        if not (
            len(resolved_defender_positions)
            == len(resolved_defender_headings)
            == self.env.defender_count
        ):
            raise ValueError("DEF edit data is incomplete")

        learner_positions = self.env._viewer_learner_state(0)[0].clone()
        if learner is not None:
            learner_positions[0] = torch.tensor(
                resolved_learner, dtype=torch.float32, device=self.device
            )
        prospective_defender_positions = torch.tensor(
            resolved_defender_positions,
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1, 2)
        learner_active, defender_active = self._edited_scene_actor_active_masks()
        self.env.validate_fixed_actor_poses(
            learner_positions=learner_positions,
            learner_active=learner_active,
            defender_positions=prospective_defender_positions,
            defender_active=defender_active,
            goal_centers=resolved_goal_centers,
            goal_radii=resolved_goal_radii,
            obstacle_centers=resolved_obstacle_centers,
            obstacle_radii=resolved_obstacle_radii,
        )

        mask = torch.ones(1, dtype=torch.bool, device=self.device)
        self.env.set_fixed_goal_geometry(
            mask,
            goal_centers=resolved_goal_centers,
            goal_radii=resolved_goal_radii,
        )
        self.env.set_viewer_learner_pose(
            0,
            0,
            resolved_learner,
            resolved_heading,
        )
        if len(resolved_defender_positions) != self.env.defender_count:
            raise ValueError("Viewer scene must contain its single active Actor")
        defender_count = self.env.defender_count
        self.env.defenders.position[0] = 0.0
        self.env.defenders.heading[0] = 0.0
        if defender_count:
            self.env.defenders.position[0, :defender_count] = torch.tensor(
                resolved_defender_positions, dtype=torch.float32, device=self.device
            )
            self.env.defenders.heading[0, :defender_count] = torch.tensor(
                resolved_defender_headings, dtype=torch.float32, device=self.device
            )
        self.env.defender_initial_position[0] = self.env.defenders.position[0]
        self.env.defender_initial_heading[0] = self.env.defenders.heading[0]
        self.env.restart_edited_episode(mask)
        self._finish_scene_edit()

    def _respawn_actor_in_edited_scene(self) -> None:
        self._reset(initial_state=self._current_episode_initial_state['environment'])

    def _reset_current_learner_outcomes(self) -> None:
        sampled_count = int(getattr(self.env, "sampled_learner_count", (1,))[0])
        self.current_learner_outcomes = [
            "running" if index < sampled_count else "unused"
            for index in range(self.control.capacity)
        ]

    def _update_current_learner_outcomes(self, step: GoalReachStep) -> None:
        statuses = getattr(self.env, "learner_status", None)
        if isinstance(statuses, torch.Tensor):
            labels = {
                0: "running",
                1: "success",
                2: "static_collision",
                3: "timeout",
                4: "captured",
                5: "pair_collision",
                6: "distance_lost",
            }
            sampled_count = int(getattr(self.env, "sampled_learner_count", (1,))[0])
            for index in range(min(sampled_count, self.control.capacity)):
                self.current_learner_outcomes[index] = labels.get(
                    int(statuses[0, index]), "running"
                )
            return

        def flags(name: str) -> torch.Tensor:
            value = step.info.get(name)
            if not isinstance(value, torch.Tensor):
                return torch.zeros(self.control.capacity, dtype=torch.bool)
            return value[0].reshape(-1).to(torch.bool).cpu()

        success = flags("success")
        collision = flags("collision")
        boundary = flags("boundary_collision")
        obstacle = flags("obstacle_collision")
        pair = flags("pair_collision")
        timeout = flags("timeout")
        sampled_count = int(getattr(self.env, "sampled_learner_count", (1,))[0])
        for index in range(min(sampled_count, self.control.capacity)):
            if self.current_learner_outcomes[index] != "running":
                continue
            if index < pair.numel() and bool(pair[index]):
                self.current_learner_outcomes[index] = "pair_collision"
            elif (
                index < collision.numel()
                and bool(collision[index])
                and (
                    index >= boundary.numel()
                    or index >= obstacle.numel()
                    or bool(boundary[index] | obstacle[index])
                )
            ):
                self.current_learner_outcomes[index] = "static_collision"
            elif index < success.numel() and bool(success[index]):
                self.current_learner_outcomes[index] = "success"
            elif index < timeout.numel() and bool(timeout[index]):
                self.current_learner_outcomes[index] = "timeout"
            elif index < collision.numel() and bool(collision[index]):
                self.current_learner_outcomes[index] = "static_collision"

    def _current_defender_outcomes(self) -> list[str]:
        if not self.env.defender_count:
            return []
        pair_collisions = getattr(self.env, "defender_pair_collision", None)
        outcomes: list[str] = []
        for index in range(self.env.defender_count):
            if (
                isinstance(pair_collisions, torch.Tensor)
                and bool(pair_collisions[0, index])
            ):
                outcomes.append("pair_collision")
            elif bool(self.env.defender_collision[0, index]):
                outcomes.append("static_collision")
            else:
                outcomes.append("running")
        return outcomes

    def _current_team_outcomes(self) -> dict[str, str | None]:
        if not bool(self.env.done[0]):
            return {role: None for role in ("attacker", "defender", "target")}
        if int(self.env.match_end_reason[0]) == MATCH_END_REASON_CODES[MatchEndReason.TRUNCATION]:
            return {role: 'truncated' for role in ('attacker', 'defender', 'target')}
        attacker_win = int(self.env.match_outcome[0]) == int(
            MatchOutcome.ATTACKER_WIN
        )
        return {
            "attacker": "win" if attacker_win else "loss",
            "defender": "loss" if attacker_win else "win",
            "target": "loss" if attacker_win else "win",
        }

    def _reset_current_match_facts(self) -> None:
        self._viewer_episode_revision = getattr(self, "_viewer_episode_revision", 0) + 1
        self.current_match_event_counts = [0] * len(MATCH_EVENT_INFO_KEYS)

    def _update_current_match_facts(self, step: GoalReachStep) -> None:
        actor_state_changed = False
        for index, key in enumerate(MATCH_EVENT_INFO_KEYS):
            value = step.info.get(key)
            if isinstance(value, torch.Tensor):
                event_count = int(
                    value[0].reshape(-1).sum(dtype=torch.int32)
                )
                self.current_match_event_counts[index] += event_count
                actor_state_changed |= event_count > 0
        if actor_state_changed:
            self._invalidate_actor_inputs()

    def _current_match_summary(self) -> dict[str, object]:
        outcome_code = int(self.env.match_outcome[0])
        outcome = (
            "attacker_win"
            if outcome_code == int(MatchOutcome.ATTACKER_WIN)
            else "defender_win"
            if outcome_code == int(MatchOutcome.DEFENDER_WIN)
            else None
        )
        elapsed_s = int(self.env.physics_ticks[0]) * float(self.config.shared["clock"]["physics_step_s"])
        counts = self.current_match_event_counts
        return {
            "episode": self._viewer_episode_revision,
            "outcome": outcome,
            "end_reason": self._current_match_end_reason(),
            "events": {
                **({"target": dict(capture=counts[6], static_collision=counts[7], vessel_collision=counts[8])} if "target" in self.env.physical_roles else {}),
                "attacker": {
                    "reach": counts[0],
                    "capture": counts[1],
                    "static_collision": counts[2],
                    "vessel_collision": counts[3],
                },
                "defender": {
                    "static_collision": counts[4],
                    "vessel_collision": counts[5],
                },
            },
            "match_length": elapsed_s,
            "match_horizon": None if self.env.bootstrap_on_time_limit else float(self.config.shared["clock"]["episode_duration_s"]),
            "target_outcomes": self._target_outcomes(),
            "actor_contacts": self.env.episode_contact_counts[0].tolist(),
        }

    def _current_match_end_reason(self) -> str | None:
        if not bool(self.env.done[0]):
            return None
        code = int(self.env.match_end_reason[0])
        for reason, reason_code in MATCH_END_REASON_CODES.items():
            if code == reason_code:
                return None if reason == MatchEndReason.NONE else reason.value
        raise RuntimeError("Match end reason is outside the authoritative registry")

    def _archive_current_learner_outcomes(self) -> None:
        sampled_count = int(getattr(self.env, "sampled_learner_count", (1,))[0])
        team_outcomes = self._current_team_outcomes()
        self.match_summary_history.appendleft(self._current_match_summary())
        self.episode_outcome_history.appendleft(
            {
                "episode": self._viewer_episode_revision,
                "outcomes": list(self.current_learner_outcomes[:sampled_count]),
                "team_outcomes": team_outcomes,
                "match_end_reason": self._current_match_end_reason(),
            }
        )
        if self.env.defender_count:
            self.defender_episode_outcome_history.appendleft(
                {
                    "episode": self._viewer_episode_revision,
                    "outcomes": self._current_defender_outcomes(),
                    "team_outcomes": team_outcomes,
                    "match_end_reason": self._current_match_end_reason(),
                }
            )

    def _horizon_pending(self):
        candidate = getattr(self, '_requested_viewer_settings', None) or self.viewer_settings
        schedule = candidate['training_settings']
        active = {
            'maximum_episode_decisions': int(self.config.shared['clock']['maximum_decisions']),
            'time_limit_boundary': 'truncation' if self.env.bootstrap_on_time_limit else 'termination',
        }
        requested = {**schedule, **candidate}
        active.update(include_remaining_time_observation='remaining_time' in self.env.learner_variant.vector_inputs,
                      terminate_on_defender_elimination=self.env.terminate_on_defender_elimination,
                      viewer_apply_sampling_limit=self.env.apply_sampling_time_limit,
                      terminate_on_any_casualty=self.env.terminate_on_any_casualty,
                      reward_objective=self.env.reward_objective,
                      active_actor_variant_id=self.env.learner_variant.architecture_id)
        return active if any(requested[key] != value for key, value in active.items()) else None

    @staticmethod
    def _apply_scene_task_settings(initial, settings):
        # This is a Viewer geometry reuse, never a checkpoint compatibility conversion.
        initial['match_rules_version'] = 7
        initial['bootstrap_on_time_limit'] = settings['training_settings']['time_limit_boundary'] == 'truncation'
        initial['apply_sampling_time_limit'] = settings['viewer_apply_sampling_limit']
        for key in ('terminate_on_any_casualty', 'terminate_on_defender_elimination',
                    'include_remaining_time_observation', 'reward_objective', 'target_mode'):
            initial[key] = settings[key]

    def _reset_prepared_settings_scene(self, future, initial, episode_index):
        scene = future.result()
        if initial is not None:
            self._apply_scene_task_settings(initial, scene['settings'])
            scene['environment'].load_state_dict(initial)
        elif episode_index is not None:
            self._reset_viewer_environment(scene['environment'], episode_index=episode_index)
        return scene

    def _reset(self, *, episode_index: int | None = None, initial_state=None) -> None:
        if getattr(self, '_training_scene_load_future', None) is not None and getattr(self, '_requested_viewer_settings', None) is not None:
            if (not self._terminal_environment_update_pending
                    or initial_state is not None or episode_index is not None):
                self._environment_reset_request = dict(generation=self._environment_update_generation,
                    exit_replay=False, target_mode=None, episode_index=episode_index,
                    initial_state=copy.deepcopy(initial_state))
                initial = copy.deepcopy(initial_state)
                if initial is None and self.scene_locked and episode_index is None:
                    initial = copy.deepcopy(self._locked_scene_state)
                if initial is not None or episode_index is not None:
                    self._training_scene_load_future = self._checkpoint_load_executor.submit(
                        self._reset_prepared_settings_scene, self._training_scene_load_future,
                        initial, episode_index)
            self._resume_after_environment_update = self.running or self._resume_after_environment_update
            self.running = False
            self._terminal_environment_update_pending = True
            self.reset_progress = dict(phase='preparing', generation=self._environment_update_generation)
            self._poll_training_scene_load()
            return
        # Repeated R coalesces; explicit scene selection may supersede one queued candidate.
        if self.reset_progress.get('phase') == 'preparing' and episode_index is None:
            return
        self._environment_reset_backup = None
        initial = copy.deepcopy(initial_state)
        exit_replay = self._replay_record is not None
        if exit_replay:
            if episode_index is None:episode_index = self._live_scene_cursor
        pending = getattr(self,'_pending_target_mode',None)
        if pending is None and self.scene_locked and episode_index is None:
            initial = copy.deepcopy(self._locked_scene_state)
        if initial is not None:
            self._apply_scene_task_settings(initial, self.viewer_settings)
        self._environment_reset_request = dict(generation=self._environment_update_generation+1,
            exit_replay=exit_replay, target_mode=pending)
        self._resume_after_environment_update = self.running or self._resume_after_environment_update
        self.running = False
        self._terminal_environment_update_pending = True
        self._terminal_snapshot_until = 0.
        try:
            self._queue_environment_update(self.active_actor_variant_id,copy.deepcopy(self.viewer_settings),
                                           episode_index=episode_index,initial_state=initial,
                                           static_changed=pending is not None or self._horizon_pending() is not None)
        except Exception as error:
            self._environment_update_generation += 1
            self._fail_environment_update('reset_prepare',error,self._environment_update_generation)
            return
        if self._environment_reset_request is None:
            return
        self._edited_reset_generation = self._environment_update_generation if initial_state is not None else None
        self.reset_progress = dict(phase='preparing',generation=self._environment_update_generation)
        self._mark_snapshot_status_dirty()

    def _remember_episode_initial_state(self) -> None:
        # Environment state contains no policy/global torch RNG; never copy models here.
        self._current_episode_initial_state = {
            "environment": copy.deepcopy(self.env.state_dict()),
        }
        if self.scene_locked:
            self._locked_scene_state = copy.deepcopy(self._current_episode_initial_state["environment"])
        self._viewer_live_attempt += 1
        policy_seed = (self._viewer_policy_session_seed + self._viewer_live_attempt) % (2**64)
        self._policy_noise = ViewerPolicyNoise(policy_seed, int(self.config.shared['clock']['maximum_decisions']))
        self._live_record = ViewerEpisodeRecord(self._current_episode_initial_state['environment'],
            dict(episode=self._viewer_episode_revision, scene_index=int(self.env.episode_counters[0]),
                 scene_seed=int(self.env._episode_seeds[0]), policy_seed=str(policy_seed)), self._replay_contract(),
            initial_terrain_coverage=self.env._terrain_coverage)
        previous = self._last_complete_record
        if previous is not None and not previous.error:
            self._live_record.BYTE_LIMIT = max(0, ViewerEpisodeRecord.BYTE_LIMIT - previous.bytes)
            self._live_record._limit()

    def _replay_contract(self):
        settings = {key:value for key,value in self.viewer_settings.items() if key not in {
            'deployment_model_ids', 'time_scale', 'snapshot_rate_hz', 'main_view_frame', 'main_view_zoom_step',
            'policy_sampling', 'evaluation_scene_preset', 'training_settings', 'training_group_labels',
            'auto_deploy_training_run_group_id',
            'learner_control_mode', 'defender_control_mode', 'target_control_mode', 'learner_rule',
            'policy_sharing', 'training_side_modes', 'joint_training_mode', 'alternating_updates'}}
        settings.update(include_remaining_time_observation='remaining_time' in self.env.learner_variant.vector_inputs,
                        terminate_on_defender_elimination=self.env.terminate_on_defender_elimination,
                        viewer_apply_sampling_limit=self.env.apply_sampling_time_limit,
                        terminate_on_any_casualty=self.env.terminate_on_any_casualty,
                        reward_objective=self.env.reward_objective,
                        active_actor_variant_id=self.env.learner_variant.architecture_id)
        return json.dumps([settings, self.config.shared['clock'], self.env.bootstrap_on_time_limit,
                           self.env.scene_contract.get('observations'),
                           'applied_decision_context_v1'], sort_keys=True)

    def _replay_status(self):
        record = self._last_complete_record
        reason = ('Complete a LIVE episode first' if record is None else record.error
                  or ('Replay incompatible with current environment' if record.contract != self._replay_contract() else None))
        return dict(active=self._replay_record is not None, available=record is not None and record.complete and not reason,
                    reason=reason, error=self._replay_error,
                    source=(self._replay_record or record).identity if (self._replay_record or record) else None,
                    policy_seed=str(self._policy_noise.seed))

    def _start_replay(self):
        status = self._replay_status()
        if not status['available']:
            raise ValueError(status['reason'] or 'Last LIVE episode is not replayable')
        if (self._training_scene_load_future is not None or self._environment_update_future is not None
                or self._pending_environment_update is not None or getattr(self, '_pending_target_mode', None) is not None):
            raise ValueError('Wait for the pending scene change before replay')
        record = self._last_complete_record
        previous = copy.deepcopy(self.env.state_dict())
        previous_coverage = copy.deepcopy(self.env._terrain_coverage)
        try:
            self.env.load_state_dict(copy.deepcopy(record.initial))
            self.env._terrain_coverage = copy.deepcopy(record.initial_terrain_coverage)
        except Exception:
            self.env.load_state_dict(previous)
            self.env._terrain_coverage = previous_coverage
            raise
        if self._replay_record is None:
            self._live_scene_cursor = int(previous['episode_counters'][0])
            self._live_control_modes = {role:list(control.modes) for role,control in self.role_controls.items()}
        self._live_record = None
        # Follow requests are reread from the latest release on return to LIVE.
        follow = getattr(self, '_follow_bundle_future', None)
        if follow is not None:
            follow.cancel()
            self._follow_bundle_future = None
            self._follow_bundle_signature = None
        self._replay_record = record
        self._applied_frame_cache = None
        self._replay_frame = None
        self._replay_index = 0
        self._replay_error = None
        self.running = False
        self._physics_ticks_into_decision = 0
        self._terminal_snapshot_until = 0.0
        self._next_physics_step_at = 0.0
        self._run_request_id = None
        self.edit_preview = None
        self._reset_current_learner_outcomes()
        self._reset_current_match_facts()
        self._reset_trajectories()
        self._record_spawn()
        self._prepare_replay_actions()
        self._mark_snapshot_static_dirty()

    def _prepare_replay_actions(self):
        record = self._replay_record
        tick = int(self.env.physics_ticks[0])
        while self._replay_index < len(record.intervals) and record.intervals[self._replay_index]['end'] <= tick:
            self._replay_index += 1
        if self._replay_index >= len(record.intervals):
            raise ValueError(f'Replay has no actions at tick {tick}')
        interval = record.intervals[self._replay_index]
        if not interval['start'] <= tick < interval['end']:
            raise ValueError(f'Replay action gap at tick {tick}')
        for role, action in interval['actions'].items():
            setattr(self, 'last_'+role+'_action', action.clone())
        self.last_action = self.last_learner_action
        if self._replay_frame is not interval['frame']:
            self._replay_frame = interval['frame']
            self.actor_frame_physics_ticks = self._replay_frame['tick']
            for role,modes in self._replay_frame['modes'].items():
                self.role_controls[role].modes = list(modes)
            self._invalidate_actor_inputs()
        return interval['end'] - tick

    def _record_applied_frame(self):
        # Cache and Replay share this owned frame; invalidate by replacement only.
        if self._applied_frame_cache is None:
            with self._measure_performance_stage('applied_frame_encode'):
                self._applied_frame_cache = dict(inputs=self._encode_decision_inputs(),
                    tick=self._applied_frame_tick, modes=copy.deepcopy(self._applied_frame_modes),
                    sources={role:copy.deepcopy(frame.display_context['deployment'])
                             for role,frame in self.actor_frames.items()})
        return self._applied_frame_cache

    def _advance_viewer_physics(self, ticks, deadline=None):
        replay = self._replay_record
        if replay is not None:
            ticks = min(ticks, self._prepare_replay_actions())
        before = int(self.env.physics_ticks[0])
        cadence = int(self.config.shared['clock']['physics_ticks_per_decision'])
        ticks = min(ticks, cadence - before % cadence)
        actions = {role:getattr(self,'last_'+role+'_action') for role in self.role_controls}
        active_before = {role:getattr(self.env,role+'_active').clone() for role in self.role_controls}
        frame_key = self._applied_frame_revision
        record = self._live_record
        frame = (self._record_applied_frame() if record is not None and not record.error
                 and frame_key != record._last_frame_key else None)
        arguments = dict(build_observation=False, physics_ticks=ticks,
                         finish_decision=(before+ticks)%cadence == 0, physics_deadline=deadline,
                         target_action=actions['target'].unsqueeze(0) if 'target' in actions else None)
        if self.active_actor_kind == 'learner':
            step = self.env._step_training_learner(actions['learner'].unsqueeze(0),
                defender_action=actions['defender'].unsqueeze(0), **arguments)
        else:
            step = self.env._step_training_defender(actions['defender'].unsqueeze(0),
                learner_action=actions['learner'].unsqueeze(0), **arguments)
        after = int(self.env.physics_ticks[0])
        if after > before:
            if replay is not None:
                expected = replay.checks.get(after)
                if expected is not None and physical_digest(self.env) != expected:
                    self._replay_error = f'Replay mismatch at physics tick {after}'
                    self.running = False
                    self.command_error = self._replay_error
                if bool(self.env.done[0]) and after != replay.intervals[-1]['end']:
                    self._replay_error = f'Replay ended at unexpected physics tick {after}'
                    self.running = False
                    self.command_error = self._replay_error
            elif record is not None:
                check_boundary = (after % cadence == 0 or bool(self.env.done[0]) or any(
                    not torch.equal(active, getattr(self.env,role+'_active')) for role,active in active_before.items()))
                record.append(before, after, actions, frame_key, frame,
                              physical_digest(self.env) if check_boundary else None)
                if bool(self.env.done[0]):
                    record.finish(after)
                    self._last_complete_record = record
                    self._live_record = None
        return step

    def _clear_scene_lock(self) -> None:
        if self.scene_locked:
            self.scene_locked = False
            self._locked_scene_state = None
            self.scene_lock_message = "Scene lock cleared: configuration changed"

    @torch.inference_mode()
    def _refresh_actor_frame(self) -> None:
        with self._measure_performance_stage("actor_frame_refresh"):
            self._refresh_actor_frame_unmeasured()

    @torch.inference_mode()
    def _refresh_actor_frame_unmeasured(self) -> None:
        if getattr(self,'_replay_record',None) is not None:
            return
        self.actor_frames = {}
        by_model = {}
        requests = {}
        for role, control in self.role_controls.items():
            active = getattr(self.env, role + '_active')[0]
            indices = active.nonzero(as_tuple=True)[0]
            neural = any(control.source_names()[index] == 'nn' for index in indices.tolist())
            rule = role == 'learner' and any(control.source_names()[index] == 'rule' for index in indices.tolist())
            requests[role] = (indices, neural, rule)
        context = (self.env._prepare_observation_context(torch.zeros(1, dtype=torch.long, device=self.device))
                   if any(neural or rule for _, neural, rule in requests.values()) else None)
        for role, control in self.role_controls.items():
            indices, neural, rule = requests[role]
            with self._measure_performance_stage('actor_' + role + '_observation'):
                if not indices.numel() or not (neural or rule):
                    observation = {}
                else:
                    observation = self.env._unified_role_observation(role, torch.zeros_like(indices), indices,
                        record_neighbors=True, consumer="actor", context=context, context_rows=torch.zeros_like(indices))
            action = torch.zeros((indices.numel(), 2), device=self.device)
            action[:, 0] = -1.0
            if rule and observation:
                action = apf_action(observation, position_scale_m=self.env.position_scale_m,
                    vessel_radius_m=self.env.vessel_radius_m, config=self.viewer_settings['learner_rule'])
            self.actor_frames[role] = _ViewerActorFrame(indices, observation, None, action, action.clone())
            frame = self.actor_frames[role]
            # Capture the runtime which produced this observation/distribution, not a pending request.
            frame.display_context = dict(deployment=self._role_deployment_descriptor(role),
                scene_rule=self.viewer_settings.get('terminate_on_any_casualty'),
                scene_episode=self._viewer_episode_revision,
                policy_label='ATK Rule (APF)' if rule else 'TGT NN' if role == 'target' else self._deployment_policy_label() if role == self.active_actor_kind else str(self._opponent_model_descriptor()['label']))
            frame.input_limits = (context[role]['limits'] if context is not None else
                self.env.patrol_limits(getattr(self.env, role + '_speed_ratio'))) if isinstance(self.env, MultiGoalReachEnv) else (
                torch.full((1, control.capacity), float(self.env.learner_dynamics.maximum_surge), device=self.device),
                torch.full((1, control.capacity), float(self.env.learner_dynamics.maximum_yaw), device=self.device),
                torch.full((1, control.capacity), float(self.env.learner_dynamics.horizontal_thruster_limit), device=self.device))
            if observation and neural:
                by_model.setdefault(control.neural_policy.model, []).append(role)
        for model, roles in by_model.items():
            observations = [self.actor_frames[r].observation for r in roles]
            keys = model.variant.actor_observation_keys if hasattr(model.variant, 'actor_observation_keys') else model.variant.observation_keys
            keys = [key for key in keys if key in observations[0]]
            from .training import concat_observation_batches
            observation = concat_observation_batches(observations, keys)
            with self._measure_performance_stage('actor_' + roles[0] + '_inference'):
                distribution = model.actor_distribution_from_observation(observation)
            offset = 0
            for role in roles:
                frame = self.actor_frames[role];count=frame.indices.numel()
                frame.distribution = type(distribution)(distribution.loc[offset:offset+count], distribution.scale[offset:offset+count])
                offset += count
        self.observation = self.actor_frames["learner"].observation
        self._resolve_actor_frame_actions(new_decision=True)
        self.actor_frame_physics_ticks = int(self.env.physics_ticks[0])

    def _resolve_actor_frame_actions(self, *, new_decision=False) -> None:
        for role, control in self.role_controls.items():
            frame = self.actor_frames[role]
            previous = None if new_decision else (
                frame.action.clone(), frame.display_context, frame.source_names, frame.physics_ticks)
            noise = None
            if control.stochastic_neural:
                slots = [int(slot) for slot in frame.indices.tolist() if control.source_names()[int(slot)] == 'nn']
                if slots:
                    decision = int(self.env.physics_ticks[0]) // int(self.config.shared['clock']['physics_ticks_per_decision'])
                    noise = torch.zeros((frame.indices.numel(),2),device=self.device)
                    values = self._policy_noise.at(role, slots, decision)
                    lookup = {slot:row for row,slot in enumerate(frame.indices.tolist())}
                    noise[[lookup[slot] for slot in slots]] = values.to(self.device)
            frame.action, _ = control.resolve_active(frame.indices, frame.rule_action, frame.distribution, standard_noise=noise)
            self._capture_actor_decision_context(role)
            if (previous is not None and torch.equal(previous[0], frame.action)
                    and previous[1]['source_key'] == frame.display_context['source_key']
                    and previous[2] == frame.source_names):
                frame.display_context, frame.source_names, frame.physics_ticks = previous[1:]
            action = getattr(self, "last_" + role + "_action")
            action.zero_(); action[:, 0] = -1.0
            action[frame.indices] = frame.action
        self.last_action = self.last_learner_action
        self._commit_applied_frame(force=new_decision)
        self._invalidate_actor_inputs()

    def _commit_applied_frame(self, *, force=False):
        modes={role:list(control.modes) for role,control in self.role_controls.items()}
        signature=(self._viewer_episode_revision, tuple((role,frame.display_context['source_key'],tuple(modes[role]))
            for role,frame in self.actor_frames.items()))
        actions={role:getattr(self,'last_'+role+'_action') for role in self.role_controls}
        if (not force and signature == self._applied_frame_signature
                and all(torch.equal(action,self._applied_frame_actions[role]) for role,action in actions.items())):
            return False
        self._applied_frame_revision += 1
        self._applied_frame_cache = None
        self._applied_frame_signature = signature
        self._applied_frame_actions = {role:action.detach().clone() for role,action in actions.items()}
        self._applied_frame_modes = modes
        self._applied_frame_tick = int(self.env.physics_ticks[0])
        return True

    def _capture_actor_decision_context(self, role):
        frame, control = self.actor_frames[role], self.role_controls[role]
        context = copy.deepcopy(frame.display_context)
        deployment = context['deployment']
        # Re-resolving an existing distribution changes controls, never its loaded weights.
        deployment['sampling'] = 'not_applicable' if deployment.get('mode') == 'rule' else 'stochastic' if control.stochastic_neural else 'deterministic'
        deployment['control_mode'] = self.viewer_settings[role + '_control_mode']
        for key in ('selected_model_id', 'pending', 'request_id', 'client_request_id', 'error', 'following'):
            deployment.pop(key, None)
        frame.source_names = tuple(control.source_names())
        context['source_key'] = json.dumps([role, deployment, context['scene_rule'], frame.source_names], sort_keys=True)
        frame.display_context = context
        frame.physics_ticks = int(self.env.physics_ticks[0])
        frame.physics_step_s = float(self.config.shared['clock']['physics_step_s'])

    def _reset_trajectories(self) -> None:
        self.trajectory_sample_physics_ticks.clear()
        for actor_trajectories in self.trajectories.values():
            for trajectory in actor_trajectories:
                trajectory.clear()
        self.event_freeze_frames.clear()
        self._event_freeze_frame_keys.clear()
        if isinstance(self.env, MultiGoalReachEnv):
            self.env.clear_viewer_event_capture()
        self.trajectory_participants = {role: getattr(self.env, role + "_active")[0].tolist()
                                        for role in self.env.physical_roles}
        self._trajectory_actor_was_active = {
            kind: [False] * len(participants)
            for kind, participants in self.trajectory_participants.items()
        }
        self._record_trajectory_positions()

    def _record_event_freeze_frames(self) -> None:
        if not isinstance(self.env, MultiGoalReachEnv):
            return
        episode = self._viewer_episode_revision
        maximum_frames = self.env._viewer_event_tick.shape[1] if self.env._viewer_event_tick is not None else 0
        for frame in self.env.drain_viewer_event_freeze_frames(0):
            key = (
                episode,
                int(frame["physics_ticks"]),
                str(frame["event_type"]),
                str(frame["actor_kind"]),
                int(frame["actor_id"]),
            )
            if key in self._event_freeze_frame_keys:
                continue
            if len(self.event_freeze_frames) >= maximum_frames:
                break
            self._event_freeze_frame_keys.add(key)
            self.event_freeze_frames.append({"episode": episode, **frame})

    def _record_trajectory_positions(self) -> None:
        physics_ticks = int(self.env.physics_ticks[0])
        duplicate_time = bool(
            self.trajectory_sample_physics_ticks
            and self.trajectory_sample_physics_ticks[-1] == physics_ticks
        )
        if not duplicate_time:
            self.trajectory_sample_physics_ticks.append(physics_ticks)
        learner_positions = self.env._viewer_learner_state(0)[0]
        for kind, positions, active in ((role, self.env.role_state(role).position[0], getattr(self.env, role + "_active")[0]) for role in self.env.physical_roles):
            current_active = [bool(value) for value in active.tolist()]
            for index, participant in enumerate(
                self.trajectory_participants[kind]
            ):
                position: tuple[float, float] | None = None
                if participant and (
                    current_active[index]
                    or self._trajectory_actor_was_active[kind][index]
                ):
                    coordinates = positions[index].tolist()
                    position = (float(coordinates[0]), float(coordinates[1]))
                trajectory = self.trajectories[kind][index]
                if duplicate_time:
                    if position is not None:
                        trajectory[-1] = position
                else:
                    trajectory.append(position)
                self._trajectory_actor_was_active[kind][index] = current_active[index]

    def _trajectory_activity_changed(self) -> bool:
        return any(getattr(self.env, role + "_active")[0].tolist() != self._trajectory_actor_was_active[role]
                   for role in self.env.physical_roles)

    @staticmethod
    def _uniform_trajectory_indices(
        sample_count: int, required_indices: set[int]
    ) -> list[int]:
        if sample_count <= VIEWER_TRAJECTORY_MAX_POINTS:
            return list(range(sample_count))
        selected = set(required_indices)
        while len(selected) < VIEWER_TRAJECTORY_MAX_POINTS:
            ordered = sorted(selected)
            older, newer = max(
                zip(ordered, ordered[1:]),
                key=lambda pair: pair[1] - pair[0],
            )
            if newer - older <= 1:
                break
            selected.add((older + newer) // 2)
        return sorted(selected)

    def _trajectory_snapshot(self) -> dict[str, object]:
        sample_ticks = list(self.trajectory_sample_physics_ticks)
        trajectories = {
            kind: [list(trajectory) for trajectory in actor_trajectories]
            for kind, actor_trajectories in self.trajectories.items()
        }
        current_ticks = int(self.env.physics_ticks[0])
        if not sample_ticks or sample_ticks[-1] != current_ticks:
            sample_ticks.append(current_ticks)
            learner_positions = self.env._viewer_learner_state(0)[0]
            for kind, positions, active in ((role, self.env.role_state(role).position[0], getattr(self.env, role + "_active")[0]) for role in self.env.physical_roles):
                current_active = [bool(value) for value in active.tolist()]
                for index, participant in enumerate(
                    self.trajectory_participants[kind]
                ):
                    position = None
                    if participant and current_active[index]:
                        coordinates = positions[index].tolist()
                        position = (float(coordinates[0]), float(coordinates[1]))
                    trajectories[kind][index].append(position)

        physics_ticks_per_decision = int(
            self.config.shared["clock"]["physics_ticks_per_decision"]
        )
        episode_horizon = max(int(self.config.shared["clock"]["maximum_decisions"]), math.ceil(int(self.env.physics_ticks[0]) / physics_ticks_per_decision))
        decision_dt = physics_ticks_per_decision * float(self.config.shared["clock"]["physics_step_s"])
        target_interval_s = max(2.0, episode_horizon * decision_dt / 40.0)
        marker_interval_decisions = max(1, math.ceil(target_interval_s / decision_dt))
        marker_interval_ticks = marker_interval_decisions * physics_ticks_per_decision
        marker_ticks = {
            tick for tick in sample_ticks if tick % marker_interval_ticks == 0
        }
        if bool(self.env.done[0]) and sample_ticks:
            marker_ticks.add(sample_ticks[-1])
        required_indices = {
            index for index, tick in enumerate(sample_ticks) if tick in marker_ticks
        }
        required_indices.update(
            boundary
            for actor_trajectories in trajectories.values()
            for trajectory in actor_trajectories
            for index in range(len(trajectory) - 1)
            if (trajectory[index] is None) != (trajectory[index + 1] is None)
            for boundary in (index, index + 1)
        )
        if sample_ticks:
            required_indices.update((0, len(sample_ticks) - 1))
        selected_indices = self._uniform_trajectory_indices(
            len(sample_ticks), required_indices
        )
        actor_count = sum(len(items) for items in trajectories.values())
        point_limit=VIEWER_TRAJECTORY_MAX_POINTS + math.ceil(episode_horizon / marker_interval_decisions) + 2 * actor_count + 2
        assert len(selected_indices) <= point_limit
        selected_ticks = [sample_ticks[index] for index in selected_indices]
        return {
            "sample_physics_ticks": selected_ticks,
            "point_limit":point_limit,
            "marker_physics_ticks": [
                tick for tick in selected_ticks if tick in marker_ticks
            ],
            "marker_interval_decisions": marker_interval_decisions,
            "horizon_physics_ticks": episode_horizon * physics_ticks_per_decision,
            "positions": {
                kind: [
                    [trajectory[index] for index in selected_indices]
                    for trajectory in actor_trajectories
                ]
                for kind, actor_trajectories in trajectories.items()
            },
        }

    def _record_spawn(self) -> None:
        position = self.env.learner.position[0].tolist()
        heading = float(self.env.learner.heading[0])
        self.spawn = {
            "learner_start_radius_m": math.hypot(*position),
            "learner_spawn_phase_deg": math.degrees(math.atan2(position[0], position[1])) % 360.0,
            "learner_heading_deg": math.degrees(heading) % 360.0,
            "learner_initial_surge_mps": 0.0,
            "learner_initial_sway_mps": 0.0,
            "learner_initial_yaw_rate_radps": 0.0,
            "obstacle_count": int(self.env.sampled_obstacle_count[0]),
            "goal_count": int(self.env.sampled_goal_count[0]),
            "learner_count": int(
                getattr(self.env, "sampled_learner_count", (1,))[0]
            ),
            "defender_count": int(
                getattr(self.env, "sampled_defender_count", (self.env.defender_count,))[0]
            ),
        }
        goal_centers, goal_radii = self._goal_geometry()
        if goal_centers:
            goal = goal_centers[0]
            bearing = math.atan2(goal[0] - position[0], goal[1] - position[1])
            heading_error = (bearing - heading + math.pi) % (2.0 * math.pi) - math.pi
            self.spawn.update(
                {
                    "learner_heading_error_deg": math.degrees(heading_error),
                    "goal_position_m": list(goal),
                    "goal_radius_m": goal_radii[0],
                    "start_goal_distance_m": math.dist(position, goal),
                }
            )

    def _obstacle_snapshot(self) -> list[dict[str, object]]:
        # Viewer CPU only: bulk content fingerprint, no per-circle tensor dispatch.
        active = self.env.obstacle_active[0].detach().cpu().numpy()
        keys = self.env.obstacle_keys[0].detach().cpu().numpy()[active]
        centers = self.env.obstacle_centers[0].detach().cpu().numpy()[active]
        radii = self.env.obstacle_radii[0].detach().cpu().numpy()[active]
        fingerprint = (keys.tobytes(), centers.tobytes(), radii.tobytes())
        if fingerprint != getattr(self, '_local_obstacle_fingerprint', None):
            self._local_obstacle_fingerprint = fingerprint
            self._local_obstacle_revision = getattr(self, '_local_obstacle_revision', 0) + 1
            self._local_obstacle_snapshot = [dict(id=self.env.terrain_key_text(key),x=x,z=z,r=r)
                for key,(x,z),r in zip(keys.tolist(),centers.tolist(),radii.tolist())]
        return self._local_obstacle_snapshot

    def _world_descriptor_snapshot(self, obstacles):
        if not hasattr(self, '_viewer_terrain'):
            self._viewer_terrain = ViewerTerrain()
        key = (self._viewer_episode_revision, int(self.env.terrain_override_revision[0]))
        if key != getattr(self, '_published_world_key', None):
            descriptor = self.env.terrain_descriptor(0, scene_epoch=self._viewer_episode_revision)
            descriptor['episode_index'] = descriptor['world_epoch']
            descriptor['world_epoch'] = self._viewer_episode_revision
            descriptor['world_geometry_revision'] = descriptor['override_revision']
            self._viewer_terrain.publish(descriptor, self.env.terrain_overrides[0],
                obstacles if descriptor['manual_fixed'] else [])
            self._published_world_descriptor = descriptor
            self._published_world_key = key
            self._world_serial_revision = getattr(self, '_world_serial_revision', 0) + 1
        return self._published_world_descriptor

    def _goal_snapshot(self) -> list[dict[str, float | int]]:
        return [
            {
                "id": index,
                "x": float(self.env.goal_centers[0, index, 0]),
                "z": float(self.env.goal_centers[0, index, 1]),
                "r": float(self.env.goal_radii[0, index]),
            }
            for index in torch.nonzero(self.env.goal_active[0], as_tuple=False)
            .flatten()
            .tolist()
        ]

    def _role_snapshot(self, role, trajectory_positions):
        state, control = self.env.role_state(role), self.role_controls.get(role)
        snapshots = []
        # Snapshot needs only the two speed caps, not the thrust/drag calculation.
        ratio = getattr(self.env, role + '_speed_ratio').to(device=self.env.device, dtype=torch.float32)
        speed_limits = self.env.learner_dynamics.maximum_surge * ratio
        yaw_limits = self.env.learner_dynamics.maximum_yaw * ratio
        # Snapshot encoding owns these Python values; avoid dozens of tiny Tensor
        # slice/scalar dispatches per craft, without caching across physics ticks.
        positions, bodies = state.position[0].tolist(), state.body_velocity[0].tolist()
        commands, headings = state.desired_body_velocity[0].tolist(), state.heading[0].tolist()
        active = getattr(self.env, role + '_active')[0].tolist()
        ratios = getattr(self.env, role + '_speed_ratio')[0].tolist()
        speeds, yaws = speed_limits[0].tolist(), yaw_limits[0].tolist()
        contacts = self.env.episode_contact_counts[0].tolist()
        status = self.env.learner_status[0].tolist() if role == 'learner' else None
        target_captured = self.env.target_captured[0].tolist() if role == 'target' else None
        source_names = control.source_names() if control else None
        actions = getattr(self, 'last_' + role + '_action').tolist() if control else None
        state_time = int(self.env.physics_ticks[0]) * float(self.config.shared['clock']['physics_step_s'])
        offset = 0 if role == 'defender' else self.env.defender_count if role == 'learner' else self.env.defender_count + self.env.viewer_learner_capacity
        for index, participant in enumerate(self.trajectory_participants[role]):
            if not participant:
                continue
            position, body, desired = positions[index], bodies[index], commands[index]
            captured = target_captured[index] if role == 'target' else status[index] == 4 if role == 'learner' else False
            distance_lost = role == 'learner' and status[index] == 6
            snapshots.append(dict(life_status="LIVE" if active[index] else "CAPTURED" if captured else "EXITED" if distance_lost else "DEAD",
                                  contact_counts=contacts[offset+index],
                                  id=index, x=position[0], z=position[1], h=headings[index],
                                  u=body[0], v=body[1], r=body[2], ud=desired[0], rd=desired[2],
                                  active=active[index], speed_ratio=ratios[index],
                                  velocity_limits=dict(surge_mps=speeds[index], yaw_radps=yaws[index]),
                                  state_time_s=state_time,
                                  control_mode=source_names[index] if control else "Static",
                                  control_action=actions[index] if control else None,
                                  state="patrol" if role == "defender" else role, navigation_target_kind=None, guide_target_m=None,
                                  collision_radius_m=self.env.vessel_radius_m,
                                  trajectory_position_m=trajectory_positions[index]))
        return snapshots

    def submit(self, command: dict[str, object]) -> None:
        with self.commands_lock:
            if getattr(self, '_closed', False) or getattr(self, '_close_requested', False):
                raise ValueError('Viewer is closing')
            kind = str(command.get("type", ""))
            if (
                kind in {"preview_edit", "set_viewer_settings"}
                and self.commands
                and str(self.commands[-1].get("type", "")) == kind
            ):
                self.commands[-1] = command
            else:
                self.commands.append(command)

    def _set_rate(self, name: str, raw_value: object) -> None:
        if name not in VIEWER_RATE_OPTIONS:
            raise ValueError("Viewer rate name is unsupported")
        try:
            value = float(raw_value) if name == "time_scale" else int(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Viewer {name} value must be numeric") from error
        if value not in VIEWER_RATE_OPTIONS[name]:
            raise ValueError(f"Viewer {name} value is unsupported")

        previous_value = getattr(self, name)
        setattr(self, name, value)
        self.viewer_settings[name] = value
        if name == "time_scale":
            if value != previous_value:
                self.rate_revision += 1
            self._next_physics_step_at = time.perf_counter()
            self._physics_rate_reset_requested = True
        self._reset_performance_diagnostics()
        self._snapshot_publish_times.clear()
        self._next_snapshot_publish_at = 0.0
        _store_viewer_settings(self.viewer_settings_path, self.viewer_settings)

    def _actor_frame_updates_deferred(self):
        return (self.scene_mode != 'continuous'
            or self._environment_update_future is not None
            or self._pending_environment_update is not None
            or getattr(self, '_viewer_fault', None) is not None)

    def _drain_commands(self) -> bool:
        force_snapshot = False
        while True:
            with self.commands_lock:
                if not self.commands:
                    return force_snapshot
                command = self.commands.popleft()
            kind = str(command.get("type", ""))
            self.command_error = None
            if kind == 'set_scene_mode':
                mode = command.get('value')
                if mode not in {'continuous', 'discrete'}:
                    raise ValueError('Unsupported scene mode')
                if mode != self.scene_mode:
                    if mode == 'discrete':
                        self._frozen_continuous_snapshot = json.loads(self.snapshot.read())
                        if self.passage is None:
                            from .passage_viewer import PassageViewer
                            self.passage = PassageViewer(self.viewer_settings_path.parent,
                                checkpoint_executor=self._checkpoint_load_executor)
                        self.passage.running = (self._discrete_running
                                                and self.passage.sampling != 'manual')
                        self._continuous_running, self.running = (
                            self.running or self._resume_after_environment_update, False)
                        self._continuous_terminal_hold = max(
                            0., self._terminal_snapshot_until - time.perf_counter())
                    else:
                        self._discrete_running = self.passage.running
                        self.passage.running = False
                        preparing = (self._environment_update_future is not None
                            or self._pending_environment_update is not None
                            or self._training_scene_load_future is not None)
                        self.running = self._continuous_running and not preparing
                        self._resume_after_environment_update = (
                            self._continuous_running and preparing)
                        self._next_physics_step_at = time.perf_counter()
                        self._last_checkpoint_poll = 0.0
                        self._mark_snapshot_static_dirty()
                        if self._terminal_snapshot_until:
                            self._terminal_snapshot_until = (
                                time.perf_counter() + self._continuous_terminal_hold)
                    self.scene_mode = mode
                    if mode == 'continuous' and getattr(self, '_restore_continuous_deployments_pending', False):
                        self._restore_continuous_deployments_pending = False
                        self._restore_deployment_models(only_unrequested=True)
                    self._physics_rate_reset_requested = True
                    self._snapshot_publish_times.clear()
                    self._next_snapshot_publish_at = 0.0
                    self._mark_snapshot_status_dirty()
                force_snapshot = True
                continue
            if kind == 'passage':
                if self.scene_mode != 'discrete':
                    raise ValueError('Select the discrete scene first')
                self.passage.command(command)
                force_snapshot = True
                continue
            if (self.scene_mode == 'discrete' and kind not in {'shutdown_viewer', 'set_viewer_visibility',
                    'delete_training_segment', 'undo_training_segment_delete',
                    'redo_training_segment_delete'} and not (
                        kind == 'set_rate' and command.get('name') == 'snapshot_rate_hz')):
                self.passage.control(command)
                force_snapshot = True
                continue
            if kind == 'shutdown_viewer':
                self._close_requested = True
                return False
            preserve_frame = self._actor_frame_updates_deferred()
            if self._replay_record is not None and kind not in {
                'run','set_rate','set_main_view_frame','set_main_view_zoom_step','set_viewer_visibility','select_actor',
                'reset_one','reset_all','restore_previous_episode','replay_last',
                'start_joint_training','resume_joint_training','stop_joint_training','cancel_joint_training'}:
                raise ValueError('Press R to return to LIVE first')
            if kind == 'replay_last':
                self._start_replay()
                force_snapshot = True
                continue
            if kind == "run":
                if bool(command.get("value", True)) and getattr(self, '_viewer_fault', None):
                    raise ValueError('Viewer state failed; press R to prepare a complete replacement')
                if bool(command.get("value", True)) and self._replay_record is not None and (self._replay_error or bool(self.env.done[0])):
                    raise ValueError('Replay is finished or failed; replay again or press R for LIVE')
                requested_running = bool(command.get("value", True))
                preparing = (self._environment_update_future is not None
                    or self._pending_environment_update is not None
                    or self._training_scene_load_future is not None)
                self.running = requested_running and not preparing
                self._resume_after_environment_update = requested_running and preparing
                self._run_request_id = command.get("client_request_id")
                self._mark_snapshot_status_dirty()
                self._next_physics_step_at = time.perf_counter() if self.running else 0.0
                self._physics_rate_reset_requested = True
                self._snapshot_publish_times.clear()
                self._next_snapshot_publish_at = 0.0
            elif kind == "set_rate":
                self._set_rate(str(command.get("name", "")), command.get("value"))
            elif kind == "set_main_view_frame":
                value = str(command.get("value", "union"))
                if value not in VIEWER_MAIN_VIEW_FRAMES:
                    raise ValueError("Viewer main view frame is unsupported")
                self.viewer_settings["main_view_frame"] = value
                _store_viewer_settings(self.viewer_settings_path, self.viewer_settings)
            elif kind == "set_main_view_zoom_step":
                value = command.get("value")
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not VIEWER_MAIN_VIEW_ZOOM_STEP_BOUNDS[0]
                    <= value
                    <= VIEWER_MAIN_VIEW_ZOOM_STEP_BOUNDS[1]
                ):
                    raise ValueError("Viewer main view zoom step is unsupported")
                self.viewer_settings["main_view_zoom_step"] = value
                _store_viewer_settings(self.viewer_settings_path, self.viewer_settings)
            elif kind == "set_mode":
                actor = str(command.get("actor", "learner"))
                mode = str(command.get("mode", "manual"))
                allowed = VIEWER_CONTROL_MODES if actor == 'learner' else {'manual', 'nn'}
                if mode not in allowed:
                    raise ValueError(f'{actor} mode must be {sorted(allowed)}')
                if mode == 'rule' and self.viewer_settings['training_side_modes']['learner'] != 'fixed':
                    raise ValueError('Rule (APF) ATK must be Fixed before selecting Rule')
                if actor not in self.role_controls:
                    raise ValueError("Actor role is not active")
                control = self.role_controls[actor]
                if actor == 'learner' and self.viewer_settings['learner_control_mode'] == 'rule' and mode != 'rule':
                    self._discard_training_follow_side(actor)
                if mode == 'rule':
                    self._set_viewer_settings({'learner_control_mode': 'rule'})
                    force_snapshot = True
                    continue
                changed=any(source!=mode for source in control.source_names()) or (
                    mode=='manual' and not torch.equal(control.manual_action,
                        control.manual_action.new_tensor((-1.,0.)).expand_as(control.manual_action)))
                for index in range(control.capacity):
                    control.set_mode(index, mode)
                    if mode == "manual":
                        control.set_manual_action(index, (-1., 0.))
                self.viewer_settings[actor + "_control_mode"] = mode
                _store_viewer_settings(self.viewer_settings_path, self.viewer_settings)
                if not preserve_frame and changed:
                    self._refresh_actor_frame()
            elif kind == "set_viewer_visibility":
                self._actor_inputs_visible = bool(command.get("actor_inputs", True))
                if self._actor_inputs_visible:
                    self._invalidate_actor_inputs()
            elif kind == "select_actor":
                actor = str(command.get("actor", "learner"))
                index = int(command.get("index", 0))
                if actor not in self.env.physical_roles:
                    raise ValueError("Viewer actor role is not present")
                participants = self.trajectory_participants[actor]
                if not 0 <= index < len(participants) or not participants[index]:
                    raise ValueError("Selected craft is not a member of this episode")
                previous_kind = self.selected_actor_kind
                previous_index = self.selected_actor_index
                previous_control = self.role_controls.get(previous_kind)
                if self._replay_record is None and previous_control is not None and 0 <= previous_index < previous_control.capacity:
                    if (previous_control.source_names()[previous_index] == "manual"
                            and (previous_kind,previous_index)!=(actor,index)):
                        previous_action=getattr(self,"last_"+previous_kind+"_action")[previous_index]
                        action_changed=not torch.equal(previous_action,previous_action.new_tensor((-1.,0.)))
                        previous_control.set_manual_action(previous_index, (-1., 0.))
                        if not preserve_frame and action_changed:
                            frame = self.actor_frames[previous_kind]
                            frame.action[frame.indices == previous_index] = frame.action.new_tensor((-1., 0.))
                            getattr(self, "last_" + previous_kind + "_action")[previous_index] = frame.action.new_tensor((-1., 0.))
                            self._capture_actor_decision_context(previous_kind)
                            self._commit_applied_frame()
                            self._invalidate_actor_inputs()
                self.selected_actor_kind = actor
                self.selected_actor_index = index
            elif kind == "select_checkpoint":
                self.command_error = None
                self._select_checkpoint(command)
            elif kind == "load_training_scene":
                self.command_error = None
                self._load_training_scene(command)
            elif kind == "save_evaluation_scene":
                self.command_error = None
                self._save_evaluation_scene()
            elif kind == "load_evaluation_scene":
                self.command_error = None
                self._load_evaluation_scene()
            elif kind in {"start_joint_training", "resume_joint_training", "stop_joint_training", "cancel_joint_training"}:
                group_id = str(command.get("run_group_id") or "new")
                request_id = command.get("client_request_id")
                self._training_command_instance = command.get('_viewer_instance')
                previous_result = self._training_command_results.get(group_id, {})
                if kind != 'cancel_joint_training' and request_id and previous_result.get("client_request_id") == request_id:
                    continue
                result = {"client_request_id": request_id, "type": kind, "error": None}
                try:
                    if kind == "start_joint_training":
                        self._start_joint_training(request_id)
                    elif kind == "resume_joint_training":
                        self._resume_joint_training(group_id, command.get("updates"), request_id)
                    elif kind == 'cancel_joint_training':
                        self._cancel_training_launch(group_id, request_id)
                    else:
                        self._stop_joint_training(group_id)
                        self._training_command_results[group_id] = result
                except (KeyError, OSError, RuntimeError, ValueError) as error:
                    if kind == 'cancel_joint_training':
                        # A stale Cancel cannot replace the accepted retry identity.
                        self.command_error = str(error)
                    else:
                        result.update(error=str(error), phase='failed', viewer_instance=self._training_command_instance)
                        self._training_command_results[group_id] = result
                self._mark_snapshot_status_dirty()
            elif kind == "follow_training_session":
                try:
                    self._set_training_follow(str(command.get('run_group_id') or '') or None)
                    self.command_error = None
                except (KeyError, OSError, ValueError) as error:
                    self.command_error = str(error)
                    self._mark_snapshot_status_dirty()
            elif kind == "delete_training_run":
                self.command_error = None
                self._delete_actor_training_run(command)
            elif kind == "delete_training_segment":
                self.command_error = None
                self._delete_training_segment(command)
            elif kind == "undo_training_segment_delete":
                self.command_error = None
                self._undo_training_segment_delete()
            elif kind == "redo_training_segment_delete":
                self.command_error = None
                self._redo_training_segment_delete()
            elif kind == "set_policy_sampling":
                mode = str(command.get("mode", "stochastic"))
                if mode not in {"deterministic", "stochastic"}:
                    raise ValueError(
                        "GoalReach viewer policy sampling must be deterministic or stochastic"
                    )
                self.control.set_neural_sampling(mode == "stochastic")
                self.patrol_control.set_neural_sampling(mode == "stochastic")
                if "target" in self.role_controls:
                    self.target_control.set_neural_sampling(mode == "stochastic")
                self.viewer_settings["policy_sampling"] = mode
                _store_viewer_settings(self.viewer_settings_path, self.viewer_settings)
                if not preserve_frame:
                    self._resolve_actor_frame_actions()
            elif kind == "manual_action":
                actor = str(command.get("actor", "learner"))
                action = (
                    float(command.get("surge", -1.0)),
                    float(command.get("yaw", 0.0)),
                )
                index = int(command.get("index", 0))
                control = self.role_controls.get(actor)
                if control is None or not 0 <= index < control.capacity or not bool(getattr(self.env, actor + "_active")[0, index]):
                    raise ValueError("Selected craft is not active")
                if control.source_names()[index] != "manual":
                    raise ValueError("Select manual mode before supplying an action")
                control.set_manual_action(index, action)
                if not preserve_frame:
                    self._resolve_actor_frame_actions()
            elif kind == "set_viewer_settings":
                self.command_error = None
                self._set_viewer_settings(command)
            elif kind in {"reset_one", "reset_all"}:
                self._reset()
            elif kind == "set_scene_lock":
                self.scene_locked = bool(command.get("value", not self.scene_locked))
                self.scene_lock_message = None
                self._locked_scene_state = (copy.deepcopy(self._current_episode_initial_state["environment"])
                                            if self.scene_locked else None)
            elif kind == "restore_previous_episode":
                cursor = self._live_scene_cursor if self._replay_record is not None else int(self.env.episode_counters[0])
                if cursor <= 0:
                    raise ValueError("No previous scene is available")
                self._reset(
                    episode_index=cursor - 1
                )
            elif kind == "set_learner":
                self.command_error = None
                self._set_learner(command)
            elif kind == "delete_circle":
                self.command_error = None
                self._delete_circle(command)
            elif kind == "set_defender":
                self.command_error = None
                self._set_defender(command)
            elif kind == "set_patrol_waypoint":
                self.command_error = None
                self._set_patrol_waypoint(command)
            elif kind == "preview_edit":
                self.command_error = None
                self._set_edit_preview(command)
            elif kind == "commit_edit":
                self.command_error = None
                self._commit_edit(command)
            elif kind == "clear_edit_preview":
                self.command_error = None
                self._clear_edit_preview()
            elif kind != "select_arena":
                raise ValueError(f"Unsupported GoalReach viewer command: {kind}")
            if kind in {"start_joint_training", "stop_joint_training"}:
                self._mark_snapshot_status_dirty()
            elif kind not in {
                "manual_action",
                "select_actor",
                "run",
                "reset_one",
                "reset_all",
                "restore_previous_episode",
                "replay_last",
                "preview_edit",
                "clear_edit_preview",
                "select_arena",
                "set_viewer_settings",
            }:
                self._mark_snapshot_static_dirty()
            if kind not in {"manual_action", "select_arena"}:
                force_snapshot = True

    @torch.no_grad()
    def _actor_inputs(self):
        if self._replay_record is not None and self._replay_frame is not None:
            inputs=self._replay_frame['inputs']
        else:
            inputs=self._record_applied_frame()['inputs']
        return self._filter_actor_inputs(inputs)

    def _filter_actor_inputs(self, inputs):
        # A display projection never mutates the immutable applied decision rows.
        active={role:set(getattr(self.env,role+'_active')[0].nonzero(as_tuple=True)[0].tolist())
                for role in self.role_controls}
        return {role+'s':[row for row in inputs.get(role+'s',[])
                         if row['actor_index'] in indices] for role,indices in active.items()}

    def _encode_decision_inputs(self):
        result = {}
        for role, control in self.role_controls.items():
            frame = self.actor_frames[role]
            inputs = []
            if not frame.indices.numel() or not frame.observation:
                result[role + "s"] = []
                continue
            speed, yaw, thrust = frame.input_limits
            for row, index in enumerate(frame.indices.tolist()):
                distribution = frame.distribution
                descriptor = frame.display_context['deployment']
                label = frame.display_context['policy_label']
                checkpoint = descriptor.get('checkpoint')
                value = _encode_actor_input(frame.observation, row,
                    None if distribution is None else distribution.loc,
                    None if distribution is None else distribution.scale,
                    frame.source_names[index], frame.action[row],
                    architecture_id=self.env.role_variant(role).architecture_id,
                    maximum_surge_mps=float(speed[0,index]), maximum_yaw_radps=float(yaw[0,index]),
                    horizontal_thruster_limit_n=float(thrust[0,index]), stochastic_action=descriptor['sampling'] == 'stochastic',
                    policy_label=label, weights_state="not_applicable_rule" if descriptor.get("mode") == "rule" else str(checkpoint or "random_initial"))
                architecture = value['policy']['architecture']
                architecture['role_descriptors'] = copy.deepcopy(architecture['role_descriptors'])
                value.update(actor_kind=role, arena_index=0, actor_index=index, physics_ticks=frame.physics_ticks,
                    physics_step_s=frame.physics_step_s, display_context=copy.deepcopy(frame.display_context),
                    speed_ratio=float(getattr(self.env, role + "_speed_ratio")[0,index]) if isinstance(self.env, MultiGoalReachEnv) else 1.0)
                inputs.append(value)
            result[role + "s"] = inputs
        return result

    def _invalidate_actor_inputs(self) -> None:
        self._actor_inputs_revision += 1

    def _mark_snapshot_static_dirty(self) -> None:
        self._snapshot_static_revision += 1
        self._snapshot_static_dirty_deferred = False

    def _mark_snapshot_status_dirty(self) -> None:
        self._snapshot_status_revision += 1

    def _cache_actor_inputs(self, published_at: float, *, force: bool = False) -> None:
        if self._encoded_actor_inputs_revision == self._actor_inputs_revision:
            return
        if (
            not force
            and self._encoded_actor_inputs_revision >= 0
            and published_at < self._next_actor_inputs_encode_at
        ):
            filtered=self._filter_actor_inputs(self.encoded_actor_inputs)
            if all(len(filtered[k])==len(self.encoded_actor_inputs.get(k,[])) for k in filtered):
                return
        self.encoded_actor_inputs = self._actor_inputs()
        self._encoded_actor_inputs_revision = self._actor_inputs_revision
        self._next_actor_inputs_encode_at = (
            published_at + 1.0 / VIEWER_ACTOR_INPUTS_RATE_HZ
        )

    @staticmethod
    def _scaled_interval_s(base_interval_s: float, scale: float) -> float:
        return 0.0 if scale == FULL_SPEED else base_interval_s / scale

    @staticmethod
    def _performance_summary(
        samples: _RollingPerformanceWindow,
    ) -> dict[str, float | None]:
        return samples.summary()

    def _reset_performance_diagnostics(self) -> None:
        for samples in self._performance_samples.values():
            samples.clear()
        self._paced_due_ticks.clear()
        self._paced_batch_ticks.clear()
        self._paced_scheduler_statistics = {}
        self._performance_statistics = {}
        self._performance_statistics_updated_at = 0.0
        self._last_loop_started_at = None
        self._last_snapshot_published_at = None

    def _record_performance_sample(self, name: str, seconds: float) -> None:
        self._performance_samples[name].append(max(0.0, seconds) * 1000.0)

    @staticmethod
    def _performance_clock_snapshot() -> tuple[float, float, float]:
        return time.perf_counter(), time.thread_time(), time.process_time()

    def _record_performance_measurement(
        self,
        name: str,
        wall_seconds: float,
        thread_cpu_seconds: float,
        process_cpu_seconds: float,
    ) -> None:
        self._record_performance_sample(f"{name}_wall_ms", wall_seconds)
        self._record_performance_sample(
            f"{name}_thread_cpu_ms", thread_cpu_seconds
        )
        self._record_performance_sample(
            f"{name}_process_cpu_ms", process_cpu_seconds
        )
        self._record_performance_sample(
            f"{name}_off_cpu_ms", max(0.0, wall_seconds - thread_cpu_seconds)
        )

    def _record_performance_measurement_since(
        self,
        name: str,
        started_at: tuple[float, float, float],
    ) -> None:
        wall_started_at, thread_cpu_started_at, process_cpu_started_at = started_at
        self._record_performance_measurement(
            name,
            time.perf_counter() - wall_started_at,
            time.thread_time() - thread_cpu_started_at,
            time.process_time() - process_cpu_started_at,
        )

    def _record_performance_timings(
        self,
        timings: dict[str, list[float] | tuple[float, float, float]],
    ) -> None:
        for name, values in timings.items():
            if len(values) != 3:
                continue
            self._record_performance_measurement(
                name,
                float(values[0]),
                float(values[1]),
                float(values[2]),
            )

    def _handle_gc_event(self, phase: str, info: dict[str, int]) -> None:
        if phase == "start":
            self._gc_started_at = self._performance_clock_snapshot()
        elif phase == "stop" and self._gc_started_at is not None:
            started_at = self._gc_started_at
            self._gc_started_at = None
            self._record_performance_measurement_since("python_gc", started_at)
            if info["generation"] == 2:
                self._record_performance_measurement_since(
                    "python_gc_generation2", started_at
                )

    @contextmanager
    def _measure_performance_stage(self, name: str):
        started_at = self._performance_clock_snapshot()
        try:
            yield
        finally:
            self._record_performance_measurement_since(name, started_at)

    def _performance_snapshot(self, now: float) -> dict[str, object]:
        if (
            not self._performance_statistics
            or now - self._performance_statistics_updated_at
            >= VIEWER_PERFORMANCE_REFRESH_S
        ):
            self._performance_statistics = {
                name: self._performance_summary(samples)
                for name, samples in self._performance_samples.items()
            }
            batch_tick_counts = self._paced_batch_ticks.integer_counts(
                VIEWER_MAX_PACED_PHYSICS_TICKS_PER_LOOP
            )
            self._paced_scheduler_statistics = {
                "hard_cap_ticks": VIEWER_MAX_PACED_PHYSICS_TICKS_PER_LOOP,
                "batch_budget_ms": VIEWER_PHYSICS_BATCH_BUDGET_S * 1000.0,
                "samples": len(self._paced_batch_ticks),
                "due_ticks": self._performance_summary(self._paced_due_ticks),
                "batch_ticks": self._performance_summary(self._paced_batch_ticks),
                "batch_tick_counts": batch_tick_counts,
                "hard_cap_hits": batch_tick_counts[
                    str(VIEWER_MAX_PACED_PHYSICS_TICKS_PER_LOOP)
                ],
            }
            self._performance_statistics_updated_at = now
        return {
            "window_max_samples": VIEWER_PERFORMANCE_WINDOW_SAMPLES,
            "physics_ticks_total": getattr(self, '_physics_ticks_total', 0),
            "monotonic_seconds": now,
            "statistics": self._performance_statistics,
            "paced_scheduler": self._paced_scheduler_statistics,
        }

    @torch.no_grad()
    def _update_snapshot_controls(self, snapshot, *, borrow=False):
        # Control facts remain live even when the last complete arena is frozen.
        # Borrow only until synchronous store() serializes bytes on this thread.
        # Actor decision frames and Replay retain their default owned copies.
        owned = (lambda value: value) if borrow else copy.deepcopy
        snapshot['viewer_settings'].update(owned(self.viewer_settings))
        snapshot['settings_pending'] = self._training_scene_load_future is not None
        snapshot['requested_viewer_settings'] = getattr(self, '_requested_viewer_settings', None)
        snapshot['horizon_pending'] = self._horizon_pending()
        snapshot['role_deployments'] = {role: self._role_deployment_descriptor(role, borrow_source=borrow)
            for role in self.env.physical_roles}
        snapshot['actor_training'] = {**self._published_actor_training_status(),
            'trainable_parameter_group_count':len(set(parameter_group_bindings(self.viewer_settings).values())),
            'train_roles':list(train_roles(self.viewer_settings)),
            'can_start':bool(train_roles(self.viewer_settings)) and self._training_scene_load_future is None
                and self._training_command_results.get('new', {}).get('phase') not in {'queued','preparing','starting'}}
        selected = dict(kind=self.selected_actor_kind, index=self.selected_actor_index,
            control_mode=self.role_controls[self.selected_actor_kind].source_names()[self.selected_actor_index]
                if self.selected_actor_kind in self.role_controls else 'Static')
        snapshot['selected_actor'] = selected
        snapshot['goal_reach_viewer'].update(running=self.running,
            time_scale=self.time_scale, rate_revision=self.rate_revision,
            snapshot_rate_hz=self.snapshot_rate_hz, command_error=self.command_error,
            policy_sampling=('stochastic' if self.control.stochastic_neural else 'deterministic')
                if self._replay_frame is None else self._replay_frame['sources'].get('learner', {}).get('sampling', 'deterministic'),
            reset=dict(self.reset_progress), run_request_id=self._run_request_id,
            selected_actor=selected, training_command_results=owned(self._training_command_results),
            checkpoint_refresh_error=self.checkpoint_refresh_error,
            environment_update_pending=(self._training_scene_load_future is not None
                or self._environment_update_future is not None or self._pending_environment_update is not None),
            settings_command_result=owned(getattr(self, '_viewer_settings_result', None)))

    def _publish_snapshot(self, physics_hz: float, *, rate_sample: bool = False) -> None:
        with self._measure_performance_stage("snapshot_publish"):
            if not getattr(self, '_viewer_fault', None):
                try:
                    self._publish_snapshot_unmeasured(physics_hz, rate_sample=rate_sample)
                    return
                except Exception as error:
                    self._viewer_failure('snapshot', error)
                    if 'goal_reach_viewer' not in json.loads(self.snapshot.read() or b'{}'):
                        raise
            # Serve the last complete pixels, with a truthful fault/reset status.
            previous = self.snapshot.read()
            if previous:
                value = json.loads(previous)
                self.sequence += 1
                value['sequence'] = self.sequence
                value['published_unix_s'] = time.time()
                self._update_snapshot_controls(value, borrow=True)
                value['goal_reach_viewer'].update(running=False, actual_time_scale=0.,
                    time_scale_limited=False, snapshot_actual_hz=0.)
                self.snapshot.store(value)

    def _viewer_failure(self, stage, error):
        if not getattr(self, '_viewer_fault', None):
            self._viewer_fault = dict(stage=stage,error=str(error),traceback=traceback.format_exc())
        self.command_error = self._viewer_fault['error']
        self.reset_progress = dict(phase='failed',generation=self._environment_update_generation,**self._viewer_fault)
        self.running = False
        self._resume_after_environment_update = False
        self._terminal_snapshot_until = 0.
        self._terminal_environment_update_pending = False
        self._mark_snapshot_status_dirty()

    @torch.no_grad()
    def _publish_snapshot_unmeasured(
        self, physics_hz: float, *, rate_sample: bool = False
    ) -> None:
        published_at = time.perf_counter()
        with self._measure_performance_stage("snapshot_actor_inputs_encode"):
            if self.scene_mode == 'continuous' and getattr(self, "_actor_inputs_visible", True):
                self._cache_actor_inputs(published_at, force=not rate_sample)
        cached_static = (
            self._cached_snapshot_static
            if self._cached_snapshot_static_revision
            == self._snapshot_static_revision
            else None
        )
        snapshot_interval_s = self._scaled_interval_s(1.0, self.snapshot_rate_hz)
        if rate_sample:
            self._snapshot_publish_times.append(published_at)
            cutoff = published_at - 2.0
            while (
                len(self._snapshot_publish_times) > 1
                and self._snapshot_publish_times[0] < cutoff
            ):
                self._snapshot_publish_times.popleft()
            if snapshot_interval_s == 0.0:
                self._next_snapshot_publish_at = published_at
            elif self._next_snapshot_publish_at <= 0.0:
                self._next_snapshot_publish_at = published_at + snapshot_interval_s
            else:
                while self._next_snapshot_publish_at <= published_at:
                    self._next_snapshot_publish_at += snapshot_interval_s
        else:
            if not (self.passage.running if self.scene_mode == 'discrete' else self.running):
                self._snapshot_publish_times.clear()
            self._next_snapshot_publish_at = published_at + snapshot_interval_s
        snapshot_actual_hz = (
            (len(self._snapshot_publish_times) - 1)
            / (self._snapshot_publish_times[-1] - self._snapshot_publish_times[0])
            if len(self._snapshot_publish_times) >= 2
            and self._snapshot_publish_times[-1] > self._snapshot_publish_times[0]
            else 0.0
        )
        if self.scene_mode == 'discrete':
            # Keep the inactive arena frozen; training/control facts stay live.
            snapshot = {**self._frozen_continuous_snapshot,
                'viewer_settings': dict(self._frozen_continuous_snapshot['viewer_settings']),
                'goal_reach_viewer': dict(self._frozen_continuous_snapshot['goal_reach_viewer'])}
            self.sequence += 1
            snapshot.update(sequence=self.sequence, published_unix_s=time.time(),
                scene_mode='discrete', passage=self.passage.snapshot())
            self._update_snapshot_controls(snapshot, borrow=True)
            snapshot['goal_reach_viewer'].update(actual_time_scale=0.0,
                snapshot_actual_hz=snapshot_actual_hz, time_scale_limited=False,
                performance=self._performance_snapshot(published_at))
            self._store_scene_snapshot(snapshot)
            return
        safety_zone_specification = self.config.goal_reach["safety_zones"]
        physics_step_s = float(self.config.shared["clock"]["physics_step_s"])
        actual_time_scale = (
            physics_hz * physics_step_s
            if self.running and self._physics_rate_measured
            else 0.0
            if not self.running
            else None
        )
        time_scale_limited = (
            self.running
            and self.time_scale != FULL_SPEED
            and actual_time_scale is not None
            and actual_time_scale > 0.0
            and actual_time_scale < 0.9 * self.time_scale
        )
        position = self.env.learner.position[0].cpu()
        body = self.env.learner.body_velocity[0].cpu()
        desired = self.env.learner.desired_body_velocity[0].cpu()
        heading = float(self.env.learner.heading[0].cpu())
        ticks = int(self.env.physics_ticks[0].cpu())
        active_kind = self.active_actor_kind
        status = (
            "interception_success"
            if active_kind == "defender"
            and bool(self.env.done[0])
            and bool(self.env.success[0])
            else "interception_failed"
            if active_kind == "defender" and bool(self.env.done[0])
            else "infiltration_success"
            if active_kind == "learner" and bool(self.env.success[0])
            else "interception_success"
            if active_kind == "learner" and bool(self.env.done[0])
            else "running"
        )
        thruster_limit = (
            float(self.env.learner_limits()[2][0, 0])
            if isinstance(self.env, MultiGoalReachEnv)
            else float(self.config.shared["dynamics"]["horizontal_thruster_limit_n"])
        )
        sense_radius_m = float(self.env.world.radius)
        sense_diameter_m = 2.0 * sense_radius_m
        learner = {
            "position_m": position.tolist(),
            "heading_rad": heading,
            "collision_radius_m": self.env.vessel_radius_m,
            "x": float(position[0]),
            "z": float(position[1]),
            "h": heading,
            "u": float(body[0]),
            "v": float(body[1]),
            "r": float(body[2]),
            "ud": float(desired[0]),
            "rd": float(desired[2]),
            "util": float(self.env.learner.thruster_forces[0].abs().max().cpu()) / thruster_limit,
        }
        obstacles = self._obstacle_snapshot()
        world_descriptor = self._world_descriptor_snapshot(obstacles)
        safety_zones = {
            "enabled": bool(safety_zone_specification.get("enabled", False)),
            "center_m": self.env.targets.position[0,0].tolist(),
            "defender_spawn_radius_fraction": float(
                safety_zone_specification["defender_spawn_radius_fraction"]
            ),
            "engagement_radius_fraction": float(
                safety_zone_specification["engagement_radius_fraction"]
            ),
            "attacker_spawn_inner_radius_fraction": float(
                safety_zone_specification["attacker_spawn_inner_radius_fraction"]
            ),
            "defender_spawn_radius_m": float(
                safety_zone_specification["defender_spawn_radius_m"]
            ),
            "engagement_radius_m": float(
                safety_zone_specification["engagement_radius_m"]
            ),
            "attacker_spawn_inner_radius_m": float(
                safety_zone_specification["attacker_spawn_inner_radius_m"]
            ),
            "attacker_spawn_outer_radius_m": float(
                self.env.scene_contract["layout"]["spec"][
                    "attacker_spawn_outer_radius_m"
                ]
            ),
        }
        trajectory_snapshot = self._trajectory_snapshot()
        trajectory_positions = trajectory_snapshot["positions"]
        assert isinstance(trajectory_positions, dict)
        defenders = self._role_snapshot("defender", trajectory_positions["defender"])
        learners = self._role_snapshot("learner", trajectory_positions["learner"])
        trajectory = next(
            (learner["trajectory_position_m"] for learner in learners if int(learner["id"]) == 0),
            [],
        )
        actor_inputs = self.encoded_actor_inputs
        selected_kind = self.selected_actor_kind if self.selected_actor_kind in self.env.physical_roles else "learner"
        participants = self.trajectory_participants[selected_kind]
        if not 0 <= self.selected_actor_index < len(participants) or not participants[self.selected_actor_index]:
            self.selected_actor_index = next(i for i,present in enumerate(participants) if present)
        selected_inputs = actor_inputs.get(f"{selected_kind}s",[])
        actor_input = next(
            (
                value
                for value in selected_inputs
                if int(value["actor_index"]) == self.selected_actor_index
            ),
            None,
        )
        selected_actor = {
            "kind": selected_kind,
            "index": self.selected_actor_index,
            "control_mode": self.role_controls[selected_kind].source_names()[self.selected_actor_index] if selected_kind in self.role_controls else "Static",
        }
        mode = selected_actor["control_mode"]
        with self._measure_performance_stage("snapshot_performance_summary"):
            performance = self._performance_snapshot(published_at)
        self.sequence += 1
        snapshot = {
            "type": "snapshot",
            "sequence": self.sequence,
            "published_unix_s": time.time(),
            "viewer_mode": "goal_reach_shared_python_backend",
            "task_id": actor_variant(self.active_actor_variant_id).task_id,
            "arena_count": 1,
            "visible_arena_count": 1,
            "gpu_parallel_arena_count": 1,
            "sense_radius_m": sense_radius_m,
            "learner_center_spawn_range_m": [
                0.0,
                sense_radius_m,
            ],
            "maximum_episode_time_s": float(self.config.shared["clock"]["episode_duration_s"]),
            "obstacles": obstacles,
            "safety_zones": safety_zones,
            "physics_ticks": [ticks],
            "physicsHz": float(physics_hz),
            "counts": (
                {mode: len(actor_inputs[f"{selected_kind}s"])}
                if mode is not None
                else {}
            ),
            "actor_inputs": actor_inputs,
            "selected_actor": selected_actor,
            "viewer_settings": {
                **self.viewer_settings,
                "learner_count_bounds": list(VIEWER_LEARNER_COUNT_BOUNDS),
                "defender_count_bounds": list(VIEWER_DEFENDER_COUNT_BOUNDS),
                'target_count_bounds':[1,10],
                "world_parameter_bounds": dict(WORLD_BOUNDS),
                "defender_spawn_radius_fraction_bounds": list(
                    DEFENDER_SPAWN_RADIUS_FRACTION_BOUNDS
                ),
                "engagement_radius_fraction_bounds": list(
                    ENGAGEMENT_RADIUS_FRACTION_BOUNDS
                ),
                "attacker_spawn_inner_radius_fraction_bounds": list(
                    ATTACKER_SPAWN_INNER_RADIUS_FRACTION_BOUNDS
                ),
                "defender_spawn_radius_m": float(
                    safety_zone_specification["defender_spawn_radius_m"]
                ),
                "engagement_radius_m": float(
                    safety_zone_specification["engagement_radius_m"]
                ),
                "attacker_spawn_inner_radius_m": float(
                    safety_zone_specification["attacker_spawn_inner_radius_m"]
                ),
                "ppo_update_mode": str(GOAL_REACH["ppo_update_mode"]),
                "optimization_epochs": int(GOAL_REACH["optimization_epochs"]),
                "optimizer_steps_per_update": int(
                    GOAL_REACH["optimizer_steps_per_update"]
                ),
                "defender_speed_ratio_bounds": list(PATROL_SPEED_RATIO_BOUNDS),
                "learner_speed_ratio_bounds": list(PATROL_SPEED_RATIO_BOUNDS),
                "entity_self_attention_layer_bounds": list(
                    ENTITY_SELF_ATTENTION_LAYER_BOUNDS
                ),
                "persisted": True,
            },
            "control_capabilities": cached_static["control_capabilities"] if cached_static else [
                "replay_last",
                "run",
                "set_rate",
                "manual_action",
                "set_mode",
                "select_actor",
                "set_viewer_settings",
                "select_checkpoint",
                "load_training_scene",
                "save_evaluation_scene",
                "load_evaluation_scene",
                "start_joint_training",
                "stop_joint_training", "resume_joint_training", "follow_training_session",
                "delete_training_run", "shutdown_viewer",
                "delete_training_segment",
                "undo_training_segment_delete",
                "redo_training_segment_delete",
                "set_main_view_frame",
                "set_main_view_zoom_step",
                "set_policy_sampling",
                "reset_one",
                "reset_all",
                "restore_previous_episode",
                "preview_edit",
                "commit_edit",
                "clear_edit_preview",
                "delete_circle",
            ],
            "actor_scenario_identity": (
                cached_static["actor_scenario_identity"] if cached_static else {
                "profile_id": GOAL_REACH["profile_id"],
                "network": "ActorCritic",
                "active_architecture_id": self.active_actor_variant_id,
                "selected_architecture_id": self._selected_scenario_architecture_id(),
                "scenarios": self._actor_scenario_options(),
                "weights_state": (
                    self._deployment_weights_state()
                    if self._active_checkpoint_loaded
                    else "independent_random_initialization_not_checkpoint"
                ),
                "checkpoint_loaded": self._active_checkpoint_loaded,
                "trained_policy": (
                    self._active_checkpoint_loaded and self.checkpoint_agent_frames > 0
                ),
                "architecture": actor_variant(
                    self.active_actor_variant_id
                ).architecture_contract(),
                "model": {
                    "checkpoint": str(self._archived_checkpoint_path or self.checkpoint),
                    "update": self.checkpoint_update,
                    "agent_frames": self.checkpoint_agent_frames,
                    "label": self._deployment_policy_label(),
                    "selected_model_id": self._selected_checkpoint_model_id(),
                    "selection": (
                        "latest"
                        if self.checkpoint_selection_mode == "auto"
                        else "random_initial"
                        if self.checkpoint_selection_mode == "random"
                        else "archived"
                        if self.checkpoint_selection_mode == "archived"
                        else "manual"
                    ),
                    "selection_mode": self.checkpoint_selection_mode,
                    "latest_update": self.latest_checkpoint_update,
                    "success_rate": self.checkpoint_success_rate,
                    "collision_rate": self.checkpoint_collision_rate,
                    "initialization": self.weights_initialization,
                    "options": self._checkpoint_options(),
                },
                }
            ),
            "metric_registry": (
                cached_static["metric_registry"]
                if cached_static
                else metric_registry_payload()
            ),
            "match_registry": (
                cached_static["match_registry"]
                if cached_static
                else match_registry_payload(self.viewer_settings["reward_objective"], first_casualty=self.viewer_settings['terminate_on_any_casualty'])
            ),
            "pending_target_mode": getattr(self,"_pending_target_mode",None),
            "settings_pending": self._training_scene_load_future is not None,
            "requested_viewer_settings": getattr(self, '_requested_viewer_settings', None),
            "horizon_pending": self._horizon_pending(),
            "goal_reach_parity": {
                "neural_policy_available": True,
                "browser_physics_or_policy": False,
                "environment": type(self.env).__name__,
                "network": "ActorCritic",
                "runtime_device": str(self.device),
                "map_compute_device": str(self.env.map_device),
                "render_batch_size": 1,
                "training_parallel_environments": int(GOAL_REACH["parallel_environments"]),
                "resolved_training_parallel_environments": (
                    _training_parallel_environments(
                        self.viewer_settings, self.active_actor_variant_id
                    )
                ),
            },
            "goal_reach_viewer": {
                "backend": "shared_python",
                "fail_closed_without_backend": True,
                "running": self.running,
                "time_scale": self.time_scale,
                "rate_revision": self.rate_revision,
                "actual_time_scale": actual_time_scale,
                "time_scale_limited": time_scale_limited,
                "snapshot_rate_hz": self.snapshot_rate_hz,
                "snapshot_actual_hz": snapshot_actual_hz,
                "performance": performance,
                "actor_inputs_revision": self._encoded_actor_inputs_revision,
                "rate_options": {
                    name: list(options) for name, options in VIEWER_RATE_OPTIONS.items()
                },
                "policy_sampling": (
                    self._replay_frame['sources'].get('learner', {}).get('sampling', 'deterministic')
                    if self._replay_frame is not None else
                    "stochastic" if self.control.stochastic_neural else "deterministic"
                ),
                "scene_mode": (
                    "fixed_geometry_random_actor" if self._edited_scene else "random_all"
                ),
                "environment_update_pending": (
                    self._training_scene_load_future is not None
                    or self._environment_update_future is not None
                    or self._pending_environment_update is not None
                ),
                "reset": dict(self.reset_progress),
                "command_error": self.command_error,
                "settings_command_result": getattr(self, "_viewer_settings_result", None),
                "run_request_id": self._run_request_id,
                "scene_locked": self.scene_locked,
                "replay": {**self._replay_status(), 'sources':self._replay_frame['sources'] if self._replay_frame else None},
                "scene_index": int(self.env.episode_counters[0]),
                "scene_lock_message": self.scene_lock_message,
                "training_command_results": self._training_command_results,
                "checkpoint_refresh_error": self.checkpoint_refresh_error,
                "selected_actor": selected_actor,
                "training_ready": actor_variant(
                    self.active_actor_variant_id
                ).training_enabled,
            },
            "goal_reach_scale": cached_static["goal_reach_scale"] if cached_static else {
                "sensing_diameter_m": sense_diameter_m,
                "sensing_radius_m": sense_radius_m,
                "semantic_resolution_m_per_pixel": sense_diameter_m / 96.0,
                "world_grid_cell_size_m": float(self.env.world.cell_size),
                "ood": False,
            },
            "goal_reach_contract": {
                "attacker_reference_distance_limit_m": sense_radius_m,
                "rollout_horizon_decisions": int(GOAL_REACH["rollout_decisions"]),
                "maximum_episode_decisions": int(
                    self.config.shared["clock"]["maximum_decisions"]
                ),
                "defender_reward": (
                    self.viewer_settings["reward_objective"]
                ),
                "defender_episode_success": (
                    "attacker_elimination_or_time_limit_with_target_protected"
                    if not self.env.bootstrap_on_time_limit else "attacker_elimination"
                ),
                "time_limit_boundary": 'truncation' if self.env.bootstrap_on_time_limit else 'termination',
                "viewer_apply_sampling_limit": self.env.apply_sampling_time_limit,
                "terminate_on_defender_elimination": self.env.terminate_on_defender_elimination,
                "include_remaining_time_observation": 'remaining_time' in self.env.learner_variant.vector_inputs,
                "defender_episode_end": (
                    "target_loss_or_attacker_elimination"
                    + ("_or_first_casualty" if self.env.terminate_on_any_casualty else "")
                    + ("_or_defender_elimination" if self.env.terminate_on_defender_elimination else "")
                    + ("_or_horizon_termination" if not self.env.bootstrap_on_time_limit else
                       "_or_sampling_truncation" if self.env.apply_sampling_time_limit else "_continuous")
                ),
                "viewer_infiltrator_reach": "terminate_and_reset_like_training",
                "goal_guard_learner_minimum_goal_distance_m": (
                    float(safety_zone_specification["attacker_spawn_inner_radius_m"])
                    if bool(safety_zone_specification.get("enabled", False))
                    else 2.0 * sense_radius_m / 3.0
                ),
                "safety_zones": {key: value for key, value in safety_zones.items() if key != "center_m"},
                "goal_guard_patrol_goal_surface_gap_m": [1.0, 2.0],
                "target_role": "protected_craft",
                **world_parameters(self.config.goal_reach),
                "obstacle_capacity": self.env.obstacle_capacity,
                "obstacle_center_radius_max_m": float(GOAL_REACH["obstacle_center_radius_max_m"]),
                "obstacle_density_unit": "centers_per_1000m2",
                "terrain_generator": TERRAIN_GENERATOR,
                "terrain_block_size_m": TERRAIN_BLOCK_SIZE_M,
                "terrain_descriptor": self.env.terrain_descriptor(0, scene_epoch=self._viewer_episode_revision),
                "reference_target_index": 0,
                "defender_position_clearance_m": float(
                    CRAFT_PLACEMENT_CLEARANCE_M
                ),
                "vessel_radius_m": self.env.vessel_radius_m,
                "scene_surface_gap_m": SCENE_SURFACE_GAP_M,
                "craft_center_separation_m": CRAFT_CENTER_SEPARATION_M,
            },
            "arenas": [
                {
                    "id": 0,
                    "episode": self._viewer_episode_revision,
                    "seed": int(self.env._episode_seeds[0]),
                    "task": "goal_reach",
                    "mode": mode,
                    "policy": mode,
                    "status": status,
                    "time": ticks * float(self.config.shared["clock"]["physics_step_s"]),
                    "physics_ticks": ticks,
                    "remaining_time_s": None if self.env.bootstrap_on_time_limit else max(0.0, (int(self.config.goal_reach["clock"]["maximum_physics_ticks"]) - ticks) * float(self.config.shared["clock"]["physics_step_s"])),
                    "decisions": ticks
                    // int(self.config.shared["clock"]["physics_ticks_per_decision"]),
                    "action": self.last_action[0].cpu().tolist(),
                    "trajectory_position_m": trajectory,
                    "trajectory_sample_physics_ticks": trajectory_snapshot[
                        "sample_physics_ticks"
                    ],
                    "trajectory_marker_physics_ticks": trajectory_snapshot[
                        "marker_physics_ticks"
                    ],
                    "trajectory_marker_interval_decisions": trajectory_snapshot[
                        "marker_interval_decisions"
                    ],
                    "trajectory_horizon_physics_ticks": trajectory_snapshot[
                        "horizon_physics_ticks"
                    ],
                    "trajectory_max_points": VIEWER_TRAJECTORY_MAX_POINTS,
                    "trajectory_point_limit":trajectory_snapshot['point_limit'],
                    "physics_step_s": float(self.config.shared["clock"]["physics_step_s"]),
                    "physics_ticks_per_decision": int(self.config.shared["clock"]["physics_ticks_per_decision"]),
                    "event_freeze_frames": list(self.event_freeze_frames),
                    "scene_sampled": dict(self.spawn),
                    "obstacles": obstacles,
                    "target_outcomes": self._target_outcomes(),
                    "targets": self._role_snapshot("target", trajectory_positions["target"]),
                    "defenders": defenders,
                    "learners": learners,
                    "edit_preview": self.edit_preview,
                    "learner": learner if bool(self.env.primary_learner_active[0]) else None,
                }
            ],
            "multi_learner": {
                "enabled": int(getattr(self.env, "viewer_learner_capacity", 1)) > 1,
                "learner_capacity": int(
                    getattr(self.env, "viewer_learner_capacity", 1)
                ),
                "sampled_learner_count": int(
                    getattr(self.env, "sampled_learner_count", (1,))[0]
                ),
                "learner_speed_ratios": (
                    self.env.learner_speed_ratio[
                        0,
                        : int(getattr(self.env, "sampled_learner_count", (1,))[0]),
                    ].tolist()
                    if isinstance(self.env, MultiGoalReachEnv)
                    else [1.0]
                ),
                "current_outcomes": list(
                    self.current_learner_outcomes[
                        : int(getattr(self.env, "sampled_learner_count", (1,))[0])
                    ]
                ),
                "episode_outcome_history": list(self.episode_outcome_history),
                "defender_capacity": int(self.env.defender_count),
                "sampled_defender_count": int(
                    getattr(
                        self.env,
                        "sampled_defender_count",
                        (self.env.defender_count,),
                    )[0]
                ),
                "defender_current_outcomes": self._current_defender_outcomes(),
                "current_team_outcomes": self._current_team_outcomes(),
                "current_match_end_reason": self._current_match_end_reason(),
                "current_match_summary": self._current_match_summary(),
                "match_summary_history": list(self.match_summary_history),
                "defender_episode_outcome_history": list(
                    self.defender_episode_outcome_history
                ),
                "training_ready": actor_variant(
                    self.active_actor_variant_id
                ).training_enabled,
                "friendly_velocity_semantics": (
                    "friend_minus_observer_ground_velocity_expressed_in_observer_body_frame"
                ),
                "friendly_heading_semantics": (
                    "sin_cos_friend_heading_minus_observer_heading"
                ),
                "friendly_yaw_rate_semantics": (
                    "friend_world_yaw_rate_normalized_by_friend_limit"
                ),
            },
        }
        with self._measure_performance_stage("snapshot_static_status_build"):
            if cached_static is None:
                primary_model = snapshot["actor_scenario_identity"]["model"]
                deployment_models = {
                    active_kind: {
                        **primary_model,
                        "actor_kind": active_kind,
                        "architecture_id": self.active_actor_variant_id,
                    }
                }
                deployment_models[self.opponent_actor_kind] = (
                    self._opponent_model_descriptor()
                )
                if self.env.target_mode == "dynamic":
                    target = self.target_runtime
                    deployment_models["target"] = dict(actor_kind="target", architecture_id=target["architecture_id"],
                        checkpoint=str(target["checkpoint"]) if target["checkpoint"] else None,
                        update=target["update"], agent_frames=target.get("agent_frames", 0), label="TGT NN", selected_model_id=target["model_id"],
                        loaded=target.get("loaded", True), sampling="stochastic" if self.target_control.stochastic_neural else "deterministic",
                        selection="saved" if target["checkpoint"] else "random",
                        selection_mode="saved" if target["checkpoint"] else "random",
                        options=[dict(id="random_current_architecture", label="Random", model_id="random_current_architecture",
                                      actor_kind="target", source_architecture_id=target["architecture_id"], source_update=0,
                                      source_agent_frames=0, kind="random"),
                                 *self._training_opponent_options("target")])
                snapshot["actor_scenario_identity"]["deployment_models"] = (
                    deployment_models
                )
                self._cached_snapshot_static = {
                    key: snapshot[key] for key in VIEWER_SNAPSHOT_CACHE_KEYS
                }
                self._cached_snapshot_static_revision = (
                    self._snapshot_static_revision
                )
        with self._measure_performance_stage("snapshot_encode_store"):
            self._update_snapshot_controls(snapshot, borrow=True)
            wire_snapshot = {**snapshot, "world_descriptor": world_descriptor,
                "arenas": [{k:v for k,v in arena.items() if k != "obstacles"} for arena in snapshot["arenas"]]}
            wire_snapshot['scene_mode'] = self.scene_mode
            wire_snapshot['passage'] = self.passage.snapshot(active=False) if self.passage is not None else None
            self._store_scene_snapshot(wire_snapshot)

    def _store_scene_snapshot(self, wire_snapshot):
        self._published_passage_revision = (wire_snapshot.get('passage') or {}).get('catalog_revision')
        self.snapshot.store(
                wire_snapshot,
                blocks={
                    "passage": (frozenset({"passage"}),
                                (wire_snapshot.get('passage') or {}).get('snapshot_revision', 0)),
                    "terrain": (frozenset({"obstacles"}), self._local_obstacle_revision),
                    "world": (frozenset({"world_descriptor"}), self._world_serial_revision),
                    "static": (
                        VIEWER_SNAPSHOT_STATIC_KEYS,
                        self._snapshot_static_revision,
                    ),
                    "status": (
                        VIEWER_SNAPSHOT_STATUS_KEYS,
                        self._snapshot_status_revision,
                    ),
                    "actor_inputs": (
                        VIEWER_SNAPSHOT_ACTOR_INPUT_KEYS,
                        self._encoded_actor_inputs_revision,
                    ),
                },
            )
        completed_at = time.perf_counter()
        if self._last_snapshot_published_at is not None:
            self._record_performance_sample(
                "snapshot_publish_gap_ms",
                completed_at - self._last_snapshot_published_at,
            )
        self._last_snapshot_published_at = completed_at

    def close(self) -> None:
        """Revoke pending work, then join this session's helpers, never training workers."""
        if getattr(self, 'passage', None) is not None:
            self.passage.close()
        with self._close_lock:
            if self._closed:
                return
            # HTTP history requests use this lock; release it before joining workers.
            with getattr(self, '_training_metrics_request_lock', threading.Lock()):
                self._closed = True
                self._close_requested = True
            with getattr(self, 'commands_lock', threading.Lock()):
                if hasattr(self, 'commands'):
                    self.commands.clear()
            for name in ('_environment_update_generation', '_training_status_generation',
                         '_training_metrics_invalidation_generation'):
                if hasattr(self, name):
                    setattr(self, name, getattr(self, name) + 1)
            for role in getattr(self, '_deployment_generations', {}):
                self._deployment_generations[role] += 1
            for item in getattr(self, '_joint_training_launch_queue', []):
                item['cancelled'] = True
                self._training_request_result(item, phase='cancelled', queue_position=None)
            if hasattr(self, '_joint_training_launch_queue'):
                self._joint_training_launch_queue.clear()
            item = getattr(self, '_joint_training_launch_pending', None)
            if item is not None:
                item['cancelled'] = True
                self._training_request_result(item, phase='cancelled')
            for _, future in getattr(self, '_deployment_futures', {}).values():
                future.cancel()
            for name in ('_deployment_futures', '_pending_deployments'):
                if hasattr(self, name):
                    getattr(self, name).clear()
            for name, value in list(vars(self).items()):
                if name.endswith('_future') and value is not None:
                    value.cancel()
                    setattr(self, name, None)
            for name in ('_joint_training_launch_pending', '_pending_environment_update',
                         '_pending_checkpoint_load', '_pending_initial_checkpoint_load',
                         '_pending_opponent_checkpoint_load', '_training_follow_request'):
                if hasattr(self, name):
                    setattr(self, name, None)
            callback = getattr(self, '_gc_callback', None)
            if callback in gc.callbacks:
                gc.callbacks.remove(callback)
            for name in ('_checkpoint_load_executor', '_training_inventory_executor',
                         '_training_status_executor', '_environment_update_executor',
                         '_training_metrics_executor', '_joint_training_launch_executor'):
                executor = getattr(self, name, None)
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)

    def run(self, viewer_html: Path, port: int, *, reserved_socket: socket.socket | None = None) -> None:
        try:
            if self._closed:
                raise ValueError('Viewer is closed')
            self._run(viewer_html, port, reserved_socket=reserved_socket)
        finally:
            self.close()

    @torch.inference_mode()
    def _run(
        self,
        viewer_html: Path,
        port: int,
        *,
        reserved_socket: socket.socket | None = None,
    ) -> None:
        handler = _handler_factory(
            viewer_html,
            self.snapshot.read,
            self.submit,
            self._read_training_metrics,
            self._read_training_lineage,
            terrain_reader=self._viewer_terrain.read,
        )
        server = ThreadingHTTPServer(
            ("127.0.0.1", int(port)),
            handler,
            bind_and_activate=reserved_socket is None,
        )
        server_thread = None
        try:
            if reserved_socket is not None:
                server.socket.close()
                server.socket = reserved_socket
                server.server_address = reserved_socket.getsockname()
                server.server_name = socket.getfqdn(str(server.server_address[0]))
                server.server_port = int(server.server_address[1])
                server.server_activate()
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            physics_step_s = float(self.config.shared["clock"]["physics_step_s"])
            physics_ticks_per_decision = int(
                self.config.shared["clock"]["physics_ticks_per_decision"]
            )
            measured_started = time.perf_counter()
            measured_ticks = 0
            physics_hz = 0.0
            while not self._close_requested:
                loop_started_at = time.perf_counter()
                if self._last_loop_started_at is not None:
                    self._record_performance_sample(
                        "loop_gap_ms",
                        loop_started_at - self._last_loop_started_at,
                    )
                self._last_loop_started_at = loop_started_at
                command_snapshot_requested = False
                try:
                    command_snapshot_requested = self._refresh_checkpoint()
                except (KeyError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
                    self.checkpoint_refresh_error = str(error)
                try:
                    command_snapshot_requested = (
                        self._drain_commands() or command_snapshot_requested
                    )
                except (KeyError, OSError, RuntimeError, ValueError) as error:
                    self.command_error = str(error)
                    command_snapshot_requested = True
                if self._close_requested:
                    break
                command_snapshot_requested = (
                    self._poll_training_scene_load() or command_snapshot_requested
                )
                command_snapshot_requested = (
                    self._poll_environment_update() or command_snapshot_requested
                )
                if self._physics_rate_reset_requested:
                    measured_started = time.perf_counter()
                    measured_ticks = 0
                    physics_hz = 0.0
                    self._physics_rate_measured = False
                    self._physics_rate_reset_requested = False
                if command_snapshot_requested:
                    self._publish_snapshot(physics_hz)
                force_snapshot = False
                if self.scene_mode == 'discrete':
                    passage_was_done = bool(self.passage.env.done[0])
                    self.passage.tick(_viewer_terminal_snapshot_hold_s(self.snapshot_rate_hz))
                    force_snapshot = not passage_was_done and bool(self.passage.env.done[0])
                state_advanced = False
                now = time.perf_counter()
                terminal_snapshot_active = (
                    self._terminal_environment_update_pending
                    or self._terminal_snapshot_until > now
                )
                if (self.scene_mode == 'continuous' and self._terminal_snapshot_until
                        and now >= self._terminal_snapshot_until):
                    if self._edited_scene:
                        self._respawn_actor_in_edited_scene()
                        self._terminal_snapshot_until = 0.0
                    elif not self._terminal_environment_update_pending:
                        self._reset()
                    force_snapshot = True
                if self.scene_mode == 'continuous' and self.running and not terminal_snapshot_active and not force_snapshot:
                    # Publish the completed decision's real physics state before
                    # preparing the next action. A checkpoint/control refresh may
                    # already have prepared this boundary's actor frame.
                    if (
                        self._physics_ticks_into_decision == 0
                        and self.actor_frame_physics_ticks != int(self.env.physics_ticks[0])
                    ):
                        try:
                            self._refresh_actor_frame()
                        except Exception as error:
                            self._viewer_failure('actor_frame', error)
                            continue
                    full_speed = self.time_scale == FULL_SPEED
                    remaining_decision_ticks = (
                        physics_ticks_per_decision - self._physics_ticks_into_decision
                    )
                    if full_speed:
                        step_physics_ticks = min(
                            remaining_decision_ticks,
                            VIEWER_MAX_PACED_PHYSICS_TICKS_PER_LOOP,
                        )
                    else:
                        physics_interval_s = self._scaled_interval_s(
                            physics_step_s, self.time_scale
                        )
                        now = time.perf_counter()
                        if self._next_physics_step_at <= 0.0:
                            self._next_physics_step_at = now
                        due_ticks = (
                            1
                            + int((now - self._next_physics_step_at) / physics_interval_s)
                            if now >= self._next_physics_step_at
                            else 0
                        )
                        step_physics_ticks = min(
                            due_ticks,
                            remaining_decision_ticks,
                            VIEWER_MAX_PACED_PHYSICS_TICKS_PER_LOOP,
                        )
                        if step_physics_ticks:
                            self._paced_due_ticks.append(due_ticks)
                    if step_physics_ticks:
                        physics_stage_started_at = (
                            self._performance_clock_snapshot()
                        )
                        physics_environment_step_started_at = (
                            self._performance_clock_snapshot()
                        )
                        finish_decision = step_physics_ticks == remaining_decision_ticks
                        physics_ticks_before = int(self.env.physics_ticks[0])
                        try:
                            step = self._advance_viewer_physics(step_physics_ticks,
                                physics_environment_step_started_at[0]
                                + VIEWER_PHYSICS_BATCH_BUDGET_S)
                        except Exception as error:
                            if self._replay_record is not None:self._replay_error = str(error)
                            self._viewer_failure('physics', error)
                            self._publish_snapshot(physics_hz)
                            continue
                        self._record_performance_measurement_since(
                            "physics_environment_step",
                            physics_environment_step_started_at,
                        )
                        physics_bookkeeping_started_at = (
                            self._performance_clock_snapshot()
                        )
                        physics_ticks_after = int(self.env.physics_ticks[0])
                        step_physics_ticks = physics_ticks_after - physics_ticks_before
                        if step_physics_ticks == 0:
                            continue
                        finish_decision = step_physics_ticks == remaining_decision_ticks
                        self._paced_batch_ticks.append(step_physics_ticks)
                        if not full_speed:
                            self._next_physics_step_at += (
                                step_physics_ticks * physics_interval_s
                            )
                        state_advanced = True
                        self._update_current_learner_outcomes(step)
                        self._update_current_match_facts(step)
                        self._record_event_freeze_frames()
                        measured_ticks += step_physics_ticks
                        self._physics_ticks_total += step_physics_ticks
                        self._physics_ticks_into_decision += step_physics_ticks
                        if (
                            finish_decision
                            or bool(self.env.done[0])
                            or self._trajectory_activity_changed()
                        ):
                            self._record_trajectory_positions()
                        self._record_performance_measurement_since(
                            "physics_bookkeeping",
                            physics_bookkeeping_started_at,
                        )
                        self._record_performance_measurement_since(
                            "physics_step",
                            physics_stage_started_at,
                        )
                        if bool(self.env.done[0]):
                            if self._replay_record is None:
                                self._archive_current_learner_outcomes()
                            self._physics_ticks_into_decision = 0
                            self._terminal_snapshot_until = 0.0 if self._replay_record is not None else (
                                time.perf_counter()
                                + _viewer_terminal_snapshot_hold_s(
                                    self.snapshot_rate_hz
                                )
                            )
                            if self._replay_record is not None:
                                self.running = False
                            force_snapshot = True
                        elif finish_decision:
                            self._physics_ticks_into_decision = 0
                now = time.perf_counter()
                if now - measured_started >= 1.0:
                    physics_hz = measured_ticks / (now - measured_started)
                    self._physics_rate_measured = True
                    measured_started = now
                    measured_ticks = 0
                if force_snapshot or (
                    not command_snapshot_requested
                    and (state_advanced or self.scene_mode == 'discrete'
                        or (self.passage is not None and self.passage._catalog_revision
                            != getattr(self, '_published_passage_revision', None)))
                    and now >= self._next_snapshot_publish_at
                ):
                    self._publish_snapshot(physics_hz, rate_sample=state_advanced
                        or (self.scene_mode == 'discrete' and self.passage.running))
                remaining = (
                    max(0.0, min(0.1, self.passage.next_step - time.monotonic()))
                    if self.scene_mode == 'discrete' and self.passage.running
                    else 0.1
                    if self.scene_mode == 'discrete'
                    else
                    0.01
                    if self._terminal_environment_update_pending
                    else max(0.0, self._terminal_snapshot_until - time.perf_counter())
                    if self._terminal_snapshot_until
                    else 0.0
                    if self.running and self.time_scale == FULL_SPEED
                    else self._next_physics_step_at - time.perf_counter()
                    if self.running
                    else 0.03
                    if self.edit_preview is not None
                    else 0.1
                )
                self._record_performance_sample(
                    "loop_work_ms",
                    time.perf_counter() - loop_started_at,
                )
                if remaining > 0.0:
                    time.sleep(remaining)
        finally:
            if server_thread is not None and server_thread.is_alive():
                server.shutdown()
                server_thread.join()
            server.server_close()



def _resume_scene_contract_diff(payload, settings):
    # Rebuild only the descriptor from this group's persisted settings, never a Viewer scene/reset.
    architecture = _unified_match_architecture_id(settings, 'learner')
    config = _build_config()
    _apply_persisted_randomization(config, settings, architecture)
    spec = scene_spec(config, learner_count_range=settings['learner_count_range'],
                      defender_count_range=settings['defender_count_range'])
    expected = scene_contract(spec, settings['learner_speed_ratio_range'],
                              settings['defender_speed_ratio_range'], settings['target_speed_ratio_range'])
    state = payload.get('environment', {})
    top, inner = payload.get('scene_contract'), state.get('scene_contract')
    disagreement = scene_contract_diff(top, inner, 'checkpoint_vs_environment.scene_contract')
    if disagreement:
        return disagreement
    return scene_contract_diff(top, expected)


def _validate_training_sample_contract(payload):
    """Check the raw saved recipes before normalization, state loads or writes."""
    recipes = [payload.get(key) for key in ('training_settings','training_identity')]
    if any(not isinstance(recipe,dict) or not isinstance(recipe.get('ppo_update'),dict)
           or recipe['ppo_update'].get('sample_contract') != PPO_SAMPLE_CONTRACT for recipe in recipes):
        raise ValueError('PPO sample contract missing or obsolete: random cold initialization required')
    if recipes[0] != recipes[1]:
        raise ValueError('Conflicting saved training recipes: random cold initialization required')


def _validate_source_game_contract(payload, settings):
    from .goal_reach_evaluation import evaluation_game_contract
    if evaluation_game_contract(payload['launch_settings']) != evaluation_game_contract(settings):
        raise ValueError('Source task/observation contract differs from this training configuration')


def _prepare_joint_training_launch(repository_root, launch_settings, sources, model_ids, train_sides_value, seed=42, resume=False):
    settings = _validated_viewer_settings(launch_settings)
    _validate_training_control_sources(settings)
    architecture = _unified_match_architecture_id(settings,'learner')
    identity = _training_settings_identity(settings,architecture)
    metadata = {}; payloads = {}; paths = {}; origins = set(); releases = {}
    for role in policy_roles(settings):
        source = sources.get(role)
        if _training_rule(settings, role) is not None and not resume:
            if source:
                raise ValueError("Rule ATK has no model initialization source")
            metadata[role] = None; paths[role] = None
            continue
        if not source:
            metadata[role] = None; paths[role] = None
            if resume:
                raise ValueError('Resume requires a complete group checkpoint')
            continue
        path = Path(source['checkpoint_path'] if isinstance(source,dict) else source).resolve()
        require_active_training_checkpoint(path)
        if not path.is_file():
            raise ValueError(f'Missing complete checkpoint: {path}')
        if path not in releases:
            releases[path] = _read_group_release(path)
        path, _, payload, _ = releases[path]
        if resume or role in train_sides_value:
            _validate_training_sample_contract(payload)
        payloads[path] = payload
        _check_source_metadata(payload, role, source if isinstance(source,dict) else None)
        if payload['architecture_id'] != architecture:
            raise ValueError('Source architecture differs from the working configuration')
        binding = payload['role_bindings'][role]
        key = binding['parameter_group']
        update = None if binding['mode'] == 'rule' else _group_role_weight_update(payload,role)
        metadata[role] = {'checkpoint_path':str(path), 'role':role, 'run_group_id':payload['run_group_id'],
            'parameter_group':key,'source_weight_update':update, 'checkpoint_global_update':int(payload['global_update']),
            'architecture_id':architecture, 'model_id':binding['model_id'],
            'source_update':update, 'checkpoint_update':int(payload['global_update'])}
        paths[role] = str(path)
        origins.add((payload['run_group_id'],int(payload['global_update'])))
        if resume and (_normalized_training_identity(payload['training_identity']) != identity or payload['seed'] != seed):
            raise ValueError('Changed research configuration requires a new group')
        if not resume:
            _validate_source_game_contract(payload, settings)
    if resume:
        if len(origins) != 1 or len(payloads) != 1:
            raise ValueError('Resume requires one complete group release')
        payload = next(iter(payloads.values()))
        reject_scene_contract_differences(_resume_scene_contract_diff(payload, settings))
        _validate_resume_history(next(iter(payloads)).parent, payload['global_update'], payload['run_group_id'])
        if _training_budget_exhausted(settings['training_settings']['updates'], int(payload['global_update'])):
            raise ValueError('Completed: explicitly raise this group budget to continue')
    mapping = parameter_group_bindings(settings)
    for group in set(mapping.values()):
        refs = {('Random',) if metadata.get(r) is None else tuple(metadata[r][k] for k in
            ('run_group_id','parameter_group','source_weight_update','architecture_id')) for r,g in mapping.items() if g==group}
        if len(refs)>1:
            raise ValueError('Shared groups require one initialization source')
    return _PreparedJointTrainingLaunch(launch_settings=settings,
        architecture_ids={r:architecture for r in policy_roles(settings)}, sources=paths, source_metadata=metadata,
        model_ids=dict(model_ids),train_sides=frozenset(train_sides_value),
        new_series_sides=frozenset() if resume else frozenset(train_sides_value),
        joint_mode=settings['joint_training_mode'],alternating_updates=settings['alternating_updates'],
        updates=settings['training_settings']['updates'],seed=seed)

def _training_initialization_mode(
    source: Path | None,
    source_payload: dict[str, object] | None,
    model_id: str,
    initial_payload: dict[str, object],
) -> str:
    restore = initial_payload.get("training_restore")
    if isinstance(restore, dict) and restore.get("mode") in {
        "fresh", "continue", "transfer_full_actor_critic",
    }:
        return str(restore["mode"])
    if source is None:
        return "fresh"
    if source_payload is None:
        return "continue"
    source_model_id = str(source_payload.get("model_id") or "")
    source_signature = checkpoint_training_series_signature(source_payload)
    initial_signature = checkpoint_training_series_signature(initial_payload)
    new_series = source_payload.get("scene_contract") is None or bool(
        source_model_id and model_id != source_model_id
    ) or bool(
        source_signature is not None
        and initial_signature is not None
        and initial_signature != source_signature
    )
    return "transfer_full_actor_critic" if new_series else "continue"


def _training_lineage_reward_points(
    records: list[dict[str, object]], maximum_points: int = 64, *, actor_kind: str = "defender"
) -> tuple[list[list[float | int]], str | None]:
    def finite_metric(record: dict[str, object], candidate: str) -> bool:
        try:
            return math.isfinite(float(record[candidate]))
        except (KeyError, TypeError, ValueError):
            return False

    metric = next(
        (
            candidate
            for candidate in (
                "match/attacker_return" if actor_kind == "learner" else "match/defense_return",
                "rollout/ep_rew_mean",
                "match/trained_side_mean_reward",
            )
            if any(
                finite_metric(record, candidate)
                for record in records
            )
        ),
        None,
    )
    if metric is None:
        return [], None
    values: dict[int, float] = {}
    for record in records:
        try:
            update = int(record["update"])
            reward = float(record[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(reward):
            values[update] = reward
    points = sorted(values.items())
    if len(points) <= maximum_points:
        return [[update, reward] for update, reward in points], metric

    interior = points[1:-1]
    bucket_count = min((maximum_points - 2) // 2, len(interior))
    selected: list[tuple[int, float]] = [points[0]]
    for bucket in range(bucket_count):
        start = bucket * len(interior) // bucket_count
        end = (bucket + 1) * len(interior) // bucket_count
        candidates = interior[start:end]
        extrema = {
            min(range(len(candidates)), key=lambda index: candidates[index][1]),
            max(range(len(candidates)), key=lambda index: candidates[index][1]),
        }
        selected.extend(candidates[index] for index in sorted(extrema))
    selected.append(points[-1])
    return [[update, reward] for update, reward in selected[:maximum_points]], metric


def _training_lineage_value_label(value: object) -> str:
    if isinstance(value, bool):
        return "On" if value else "Off"
    if isinstance(value, list | tuple):
        return "–".join(_training_lineage_value_label(item) for item in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _training_lineage_change_summary(
    parent: dict[str, object], child: dict[str, object],
) -> list[str]:
    # Compare the current configurable fields, not implementation metadata.
    # A missing recorded value is not evidence that a parameter changed.
    titles = {
        "training_sides": "Training",
        "joint_training_mode": "Update mode",
        "alternating_updates": "Alternate every",
        "terminate_on_any_casualty": "First casualty ends match",
        "reward_objective": "Reward objective",
        "learner_count_range": "ATK count",
        "defender_count_range": "DEF count",
        "learner_speed_ratio_range": "ATK speed",
        "defender_speed_ratio_range": "DEF speed",
        "safety_zones_enabled": "Safety zones",
        "defender_spawn_radius_fraction": "DEF spawn R",
        "engagement_radius_fraction": "Engagement R",
        "attacker_spawn_inner_radius_fraction": "ATK spawn R",
        "goal_count_range": "Goals",
        "goal_radius_range_m": "Goal radius",
        "obstacle_density_mean_per_1000m2": "Density mean",
        "obstacle_density_std_per_1000m2": "Density std",
        "obstacle_radius_mean_m": "Radius mean",
        "obstacle_radius_std_m": "Radius std",
        "world_map_seed": "World",
        "entity_streams": "Entity streams",
        "entity_self_attention_layers": "Attention layers",
        "include_learner_speed_ratio_observation": "Speed observation",
        "maximum_episode_decisions": "Episode length",
        "time_limit_boundary": "Time limit",
        "rollout_decisions": "Rollout",
        "minibatch_agent_frames": "Batch",
        "parallel_environments": "Envs",
        "learning_rate": "LR",
        "entropy_coefficient": "Entropy",
        "actor_initial_log_scale": "Initial log σ",
        "gamma": "γ",
        "gae_lambda": "GAE λ",
        "optimization_epochs": "Epochs",
        "ppo_update_mode": "PPO mode",
        "optimizer_steps_per_update": "Optimizer steps",
        "seed": "Seed",
    }
    before = parent.get("training_parameters") or {}
    after = child.get("training_parameters") or {}

    def value_label(key: str, value: object) -> str:
        if key == "training_sides":
            return {
                frozenset(("learner", "defender")): "Joint",
                frozenset(("learner",)): "ATK only",
                frozenset(("defender",)): "DEF only",
            }.get(frozenset(value), "/".join(str(role).upper() for role in value))
        if key.endswith("_radius_fraction"):
            return f"{float(value):.3g}R"
        if key in {"minibatch_agent_frames", "parallel_environments"}:
            number = int(value)
            return f"{number // 1024}k" if number >= 1024 and number % 1024 == 0 else str(number)
        if isinstance(value, str):
            return {
                "fixed_optimizer_steps": "Fixed steps",
                "termination": "Terminal",
                "truncation": "Bootstrap",
            }.get(value, value)
        if isinstance(value, (list, tuple)) and len(value) == 2 and value[0] == value[1]:
            return _training_lineage_value_label(value[0])
        return _training_lineage_value_label(value)

    return [
        f"{title} {value_label(key, before[key])} → {value_label(key, after[key])}"
        for key in dict.fromkeys([*titles, *sorted(set(before) | set(after))])
        for title in (titles.get(key, key.replace("_", " ")),)
        if key not in {"updates", "training_segment_updates"} and before.get(key) is not None and after.get(key) is not None
        and before[key] != after[key]
    ]



def _training_lineage_train_sides(run: dict[str, object]) -> set[str]:
    settings = run.get("_training_settings")
    joint_training = settings.get("joint_training") if isinstance(settings, dict) else None
    train_sides = (
        joint_training.get("train_sides")
        if isinstance(joint_training, dict)
        else None
    )
    if not isinstance(train_sides, list | tuple):
        manifest = run.get("_run_manifest")
        train_sides = manifest.get("train_sides") if isinstance(manifest, dict) else None
    return (
        {str(side) for side in train_sides if side in {"learner", "defender", "target"}}
        if isinstance(train_sides, list | tuple)
        else set()
    )


def _training_lineage_has_random_initialization(run: dict[str, object]) -> bool:
    manifest = run.get("_root_run_manifest")
    if not isinstance(manifest, dict):
        manifest = run.get("_run_manifest")
    manifest = manifest if isinstance(manifest, dict) else {}
    if str(manifest.get("initialization_mode") or "") in {
        "fresh",
        "random",
        "adaptive_random",
    }:
        return True
    initialization = run.get("_weights_initialization")
    if not isinstance(initialization, dict):
        initialization = manifest.get("weights_initialization")
    if isinstance(initialization, dict) and (
        str(initialization.get("mode") or "") in {"fresh", "random", "adaptive_random"}
        or str(initialization.get("kind") or "") in {"fresh", "random"}
    ):
        return True
    restore = run.get("_training_restore")
    if not isinstance(restore, dict) or restore.get("source_checkpoint"):
        return False
    if str(restore.get("mode") or "") == "fresh":
        return True
    restored_components = restore.get("restored_components")
    return bool(
        isinstance(restored_components, dict)
        and restored_components
        and not any(bool(value) for value in restored_components.values())
    )


_GROUP_HISTORY_RECORDS = {}


_GROUP_HISTORY_STATES = {}


def _passage_history_entries(passage_root, *, include_records):
    """Adapt metadata only; reuse the shared incremental metric reader and transport."""
    from .passage import CONTRACT, PassageConfig

    root = Path(passage_root).resolve()
    catalog_path = root / "catalog.json"
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        if not isinstance(catalog, dict):
            raise ValueError("Passage catalog is not an object")
    except FileNotFoundError:
        return [], []
    except (OSError, UnicodeError, ValueError) as error:
        return [], [dict(source=str(catalog_path), error=str(error))]
    entries, errors = [], []
    for name, row in catalog.items():
        metrics_path = root / (str(name).removesuffix(".pt") + ".metrics.jsonl")
        try:
            if not isinstance(name, str) or Path(name).name != name or not name.endswith(".pt"):
                raise ValueError("Invalid passage catalog model name")
            checkpoint = root / name
            if checkpoint.is_symlink() or checkpoint.resolve().parent != root:
                raise ValueError("Passage checkpoint leaves its catalog directory")
            cfg = row["config"]
            group_id = "passage:" + name
            records, record_state = [], None
            if include_records:
                if row['updates'] and not metrics_path.is_file():
                    raise ValueError("Full passage metrics file is unavailable")
                records, _ = _read_training_dashboard_records(
                    metrics_path, _GROUP_HISTORY_RECORDS, record_kind="passage")
                record_state = _GROUP_HISTORY_RECORDS.get(metrics_path)
            record_errors = (record_state or {}).get("errors", [])
            owner = read_actor_training_lock(checkpoint.with_suffix(".lock"), prune_stale=False)
            attempt = row.get("launch_attempt") or {}
            matches = bool(owner and owner.get("run_group_id") == group_id
                           and owner.get("launch_attempt_id") == attempt.get("id"))
            identity = (owner.get("process_identity_status", "unknown")
                        if matches else "unknown" if owner else "dead")
            reserved = bool(owner and owner.get("capacity_reserved"))
            status = row["status"]
            if status == "running" and identity != "alive":
                status = "unknown" if reserved else "stopped"
            bindings = {role: dict(key=group_id + ":" + role,
                mode="train" if role == "learner" else "unavailable",
                parameter_group="agent" if role == "learner" else None,
                source_update=0, architecture_id=CONTRACT,
                deploy_reference=group_id if role == "learner" else None)
                for role in ("learner", "defender", "target")}
            meta = dict(scene_mode="discrete", name=name, run_group_id=group_id,
                started_at=row["started_at"], global_update=row["updates"],
                checkpoint_update=row["updates"], update_budget=row.get("update_budget", row["updates"]),
                archived=False, status=status, error=row.get("error"), config=cfg,
                launch_attempt=row.get("launch_attempt"), update_timing=row.get("update_timing"),
                evaluation=row.get('evaluation'),
                running=status == "running", process_identity_status=identity, capacity_reserved=reserved,
                settings=dict(policy_sharing="role", terminate_on_any_casualty=False, training_settings=cfg),
                role_bindings=bindings, parameter_groups=dict(agent=dict(members=["learner"], update_index=row["updates"])),
                credit_minibatches=dict(agent=dict(members=["learner"], value=PassageConfig(**cfg).effective_minibatch, update=row["updates"])),
                record_errors=record_errors[-3:], stale=bool(record_errors or (record_state or {}).get("remainder")),
                source=str(metrics_path), catalog_source=str(catalog_path))
            entries.append((meta, records, record_state))
        except (OSError, UnicodeError, ValueError, TypeError, KeyError) as error:
            errors.append(dict(source=str(metrics_path), error=str(error)))
    return entries, errors


def _group_history_payload(repository_root, *, include_records, include_delta=False, passage_root=None):
    root=Path(repository_root).resolve()
    passage_root=Path(passage_root or root/'Artifacts'/'GoalReach'/'passage').resolve()
    state=_GROUP_HISTORY_STATES.setdefault((str(root),include_records,str(passage_root)), {'revision':0,'groups':{},'record_states':{},'version':''})
    config = _build_config()
    launcher=ActorTrainingLauncher(root)
    live={g['run_group_id']:g for g in launcher.refresh()['groups']}
    groups=[];changed=[];next_groups={};record_states={};entries=[]
    previous=state['groups']
    for entry in reversed(launcher.training_run_registry()):
        manifest=entry['manifest'];output=Path(entry['output'])
        if manifest.get('checkpoint_schema')!='mobile_local_sensing_group':continue
        group_id=manifest['run_group_id'];runtime=live[group_id]
        records=[];record_state=None;errors=[]
        metrics_path=(output/'metrics.jsonl').resolve()
        if include_records:
            records,_=_read_training_dashboard_records(metrics_path,_GROUP_HISTORY_RECORDS)
            record_state=_GROUP_HISTORY_RECORDS.get(metrics_path)
            errors=(record_state or {}).get('errors', [])
        bindings=copy.deepcopy(manifest['role_bindings'])
        checkpoint=output/'checkpoint-latest.pt'
        for role,binding in bindings.items():
            binding.update(key=f'{group_id}:{role}',architecture_id=manifest['architecture_id'],
                deploy_reference='saved_target:'+checkpoint.relative_to(root).as_posix() if checkpoint.is_file() and binding['mode'] not in {'static', 'rule'} else None,
                checkpoint_global_update=manifest.get('checkpoint_update',0))
            source=binding.get('source')
            binding['parent_key']=f"{source['run_group_id']}:{source['role']}" if isinstance(source,dict) else None
        credit_minibatches = {}
        latest = runtime.get('latest_record', {})
        for key, parameter_group in manifest.get('parameter_groups', {}).items():
            members = parameter_group.get('members', [])
            values = [runtime['effective_minibatch_agent_frames'].get(role)
                      if bindings.get(role, {}).get('mode') == 'train'
                      and bindings[role].get('parameter_group') == key else None for role in members]
            valid = bool(values) and all(isinstance(value, (int, float)) and math.isfinite(value)
                                        and value > 0 and value == values[0] for value in values)
            credit_minibatches[key] = {'members': members, 'value': values[0] if valid else None,
                                      'update': latest.get('train/joint_global_update') if valid else None}
        expected_inputs=actor_variant(manifest['architecture_id']).vector_input_size
        stored_inputs=manifest.get('network',{}).get('actor_inputs')
        resume_error=(f'Incompatible actor inputs: stored {stored_inputs}, current {expected_inputs}; new initialization required'
                      if stored_inputs is not None and stored_inputs!=expected_inputs else 'Checkpoint metadata not yet verified' if stored_inputs is None else None)
        for binding in bindings.values():
            if binding['mode']!='static':binding['deploy_error']=resume_error
        meta={'run_group_id':group_id,'started_at':manifest['started_at'],'status':runtime['status'],
            'archived':bool(entry.get('archived')),
            'launch_attempt':manifest.get('launch_attempt'),'publication_sequence':manifest.get('publication_sequence'),
            'population':manifest.get('population'),
            'running':runtime['running'],'runtime_phase':runtime['runtime_phase'],
            'process_identity_status':runtime['process_identity_status'],'capacity_reserved':runtime['capacity_reserved'],
            'global_update':manifest.get('global_update',0),'checkpoint_update':manifest.get('checkpoint_update',0),
            'update_budget':manifest['update_budget'],'role_bindings':bindings,'parameter_groups':manifest.get('parameter_groups',{}),
            'settings':manifest['launch_settings'],'error':runtime.get('error'),'resume_error':resume_error, 'record_errors':errors[-3:], 'stale':bool(errors or (record_state or {}).get('remainder')),
            'effective_minibatch_agent_frames':runtime['effective_minibatch_agent_frames'],
            'credit_minibatches':credit_minibatches,
            'update_timing':runtime.get('update_timing'),
            'horizon_seconds':manifest['launch_settings']['training_settings']['maximum_episode_decisions']*float(config.shared['clock']['physics_ticks_per_decision'])*float(config.shared['clock']['physics_step_s']),
            'metric_applicability':{spec.key:metric_applicable(spec, manifest['launch_settings']) for spec in METRIC_REGISTRY.values()},
            'source':entry['metrics_id']}
        entries.append((meta,records,record_state))
    passage_entries,passage_errors=_passage_history_entries(passage_root,include_records=include_records)
    history_errors=[*launcher.registry_errors,*passage_errors]
    for meta,records,record_state in [*entries,*passage_entries]:
        group_id=meta['run_group_id']
        old=previous.get(group_id)
        reset=old is None or (include_records and state['record_states'].get(group_id) is not record_state)
        count=0 if reset else old['count']
        appended=records[count:]
        if reset or old['meta']!=meta or appended:
            changed.append({**meta,'records':appended,'reset_records':reset})
        # Encode new records only; full fallback reuses the encoded prefix.
        encoded=[] if reset else old['encoded']
        if appended:encoded=encoded+[json.dumps(row,ensure_ascii=False) for row in appended]
        next_groups[group_id]={'meta':meta,'count':len(records),'encoded':encoded}
        record_states[group_id]=record_state
        groups.append(json.dumps(meta,ensure_ascii=False)[:-1]+',"records":['+','.join(encoded)+']}')
    unreadable={str(Path(item['source']).resolve()):item['error'] for item in history_errors}
    for group_id,old in previous.items():
        error=(unreadable.get(str((root/old['meta']['source']).resolve()))
               or unreadable.get(old['meta'].get('catalog_source')))
        if group_id not in next_groups and error:
            meta={**old['meta'],'stale':True,'record_errors':[error],'resume_error':error}
            next_groups[group_id]={**old,'meta':meta}
            record_states[group_id]=state['record_states'].get(group_id)
            groups.append(json.dumps(meta,ensure_ascii=False)[:-1]+',"records":['+','.join(old['encoded'])+']}')
            changed.append({**meta,'records':[],'reset_records':False})
    removed=list(set(previous)-set(next_groups))
    undo={'can_undo':any((root/'Artifacts'/'GoalReach'/TRAINING_UNDO_DIRECTORY).glob('*/manifest.json')),
          'can_redo':any((root/'Artifacts'/'GoalReach'/TRAINING_REDO_DIRECTORY).glob('*/manifest.json'))}
    base=state['version']
    if changed or removed or state.get('undo')!=undo:
        state['revision']+=1
    version=('metrics:' if include_records else 'groups:')+str(state['revision'])
    common={'version':version,'task_id':'all','metric_registry':metric_registry_payload(),'errors':history_errors,
            'role_descriptors':ROLE_DESCRIPTORS,'undo_redo':undo}
    full=(json.dumps({**common,'delta':False},ensure_ascii=False)[:-1]+',"groups":['+','.join(groups)+']}').encode('utf-8')
    delta=json.dumps({**common,'delta':True,'base_version':base,'groups':changed,'removed_group_ids':removed},ensure_ascii=False).encode('utf-8')
    state.update(groups=next_groups,record_states=record_states,version=version,undo=undo)
    valid_paths={(root/g['meta']['source']).resolve() for g in next_groups.values()}
    for path in tuple(_GROUP_HISTORY_RECORDS):
        if path.is_relative_to(root) and path not in valid_paths:del _GROUP_HISTORY_RECORDS[path]
    return version,(full,base,delta) if include_delta else full


def _training_lineage_payload(session, *, live_runs=None):
    return _group_history_payload(session.config.repository_root,include_records=False)


def _build_training_metrics_payload(repository_root,architecture_id,invalidation_generation=0,include_delta=False,passage_root=None):
    return _group_history_payload(repository_root,include_records=True,include_delta=include_delta,passage_root=passage_root)


def _build_training_lineage_payload(repository_root,architecture_id,invalidation_generation=0,passage_root=None):
    return _group_history_payload(repository_root,include_records=False,include_delta=True,passage_root=passage_root)


def _reserve_viewer_socket(port: int) -> socket.socket:
    reserved = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if os.name == "nt":
        reserved.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    try:
        reserved.bind(("127.0.0.1", int(port)))
    except OSError:
        reserved.close()
        raise
    return reserved


@contextmanager
def _cpu_thread_affinity(*, training=False):
    """Keep submission local; give training and Viewer separate available cores.

    Bind only the calling thread, inside the user's existing process affinity.
    Torch thread counts and child-process affinity are independent of this scope.
    """
    if sys.platform != "win32":
        yield
        return
    import ctypes
    import struct
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    # GetProcessAffinityMask addresses one processor group. Leave larger hosts
    # to their scheduler rather than guessing a group for this thread.
    if kernel.GetActiveProcessorGroupCount() != 1:
        yield
        return
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetCurrentThread.restype = wintypes.HANDLE
    kernel.GetProcessAffinityMask.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel.SetThreadAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    kernel.SetThreadAffinityMask.restype = ctypes.c_size_t
    allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(
        kernel.GetCurrentProcess(), ctypes.byref(allowed), ctypes.byref(system)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    size = wintypes.DWORD()
    kernel.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))
    topology = ctypes.create_string_buffer(size.value)
    if not kernel.GetLogicalProcessorInformationEx(0, topology, ctypes.byref(size)):
        raise ctypes.WinError(ctypes.get_last_error())
    pointer_format = "Q" if ctypes.sizeof(ctypes.c_size_t) == 8 else "I"
    cores = []
    offset = 0
    while offset < size.value:
        _, record_size = struct.unpack_from("II", topology, offset)
        # PROCESSOR_RELATIONSHIP: efficiency at byte 9, GROUP_AFFINITY at 32.
        mask = struct.unpack_from(pointer_format, topology, offset + 32)[0]
        mask &= allowed.value
        if mask:
            efficiency = topology.raw[offset + 9]
            cores.append((efficiency, mask))
        offset += record_size
    cores.sort(reverse=True)  # Prefer performance cores, retaining SMT siblings.
    # Leave a restricted single-core process alone when it cannot be separated.
    if not cores or (training and len(cores) < 2):
        yield
        return
    _, mask = cores[1 if training else 0]
    thread = kernel.GetCurrentThread()
    class GroupAffinity(ctypes.Structure):
        _fields_ = [("mask", ctypes.c_size_t), ("group", wintypes.WORD),
                    ("reserved", wintypes.WORD * 3)]
    kernel.GetThreadGroupAffinity.argtypes = [wintypes.HANDLE, ctypes.POINTER(GroupAffinity)]
    original = GroupAffinity()
    if not kernel.GetThreadGroupAffinity(thread, ctypes.byref(original)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if training and torch.get_num_threads() > 1:
            # The first ATen OpenMP parallel region can restore process-wide
            # affinity on Windows. Warm it before binding, retaining the mask
            # from BEFORE initialization for scoped restoration (also on error).
            # This CPU-only operation consumes no RNG or training data.
            torch.ones((512, 512), device="cpu").sum()
        if not kernel.SetThreadAffinityMask(thread, mask):
            raise ctypes.WinError(ctypes.get_last_error())
        print(json.dumps({("training" if training else "viewer") + "_thread_cpu_mask": mask}), flush=True)
        yield
    finally:
        kernel.SetThreadAffinityMask(thread, original.mask)


def viewer_main() -> None:
    parser = argparse.ArgumentParser(description="Run the current GoalReach viewer")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    try:
        reserved_socket = _reserve_viewer_socket(args.port)
    except OSError as error:
        parser.error(
            f"GoalReach Viewer port {args.port} is already owned by another instance: "
            f"{error}"
        )
    # Keep imported, process-lifetime library objects out of repeated full GC
    # scans. Freeze before creating the session so replaced models, scenes and
    # snapshots still participate in normal cyclic collection.
    gc.collect()
    gc.freeze()
    session = None
    try:
        try:
            session = GoalReachViewerSession(args.checkpoint)
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        viewer_html = session.config.repository_root / "Python" / "viewer" / "goal_reach.html"
        if not viewer_html.is_file():
            parser.error(f"Missing GoalReach viewer: {viewer_html}")
        print(
            json.dumps(
                {
                    "url": f"http://127.0.0.1:{args.port}/",
                    "pid": os.getpid(),
                    "cwd": str(Path.cwd()),
                    "python": sys.executable,
                    "python_version": sys.version.split()[0],
                    "source_file": str(Path(__file__).resolve()),
                    "source_revision": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=session.config.repository_root, text=True
                    ).strip(),
                    "torch_threads": torch.get_num_threads(),
                    "torch_interop_threads": torch.get_num_interop_threads(),
                    "control": "nn",
                    "neural_policy": (
                        str(session.checkpoint)
                        if session._active_checkpoint_loaded
                        else "independent_random_initialization_not_checkpoint"
                    ),
                    "device": "cpu",
                    "batch_size": 1,
                    "default_speed": (
                        "full" if session.time_scale == FULL_SPEED else f"{session.time_scale:g}x"
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        try:
            with _cpu_thread_affinity():
                session.run(
                    viewer_html,
                    args.port,
                    reserved_socket=reserved_socket,
                )
        except KeyboardInterrupt:
            pass
    finally:
        if session is not None:
            session.close()
        gc.unfreeze()
        reserved_socket.close()



def _read_group_release(checkpoint, cache=None):
    """Read one actual immutable release per preparation task; raw payload never escapes that task."""
    requested = Path(checkpoint).resolve()
    path = requested
    expected_group = None
    expected_update = None
    expected_publication_sequence = None
    manifest_path = requested.parent/'run.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if not isinstance(manifest, dict):
            raise ValueError('Training manifest must be a JSON object')
        if manifest.get('checkpoint_schema') != 'mobile_local_sensing_group':
            raise ValueError('Training manifest checkpoint schema is invalid')
        expected_group = manifest.get('run_group_id')
        if not isinstance(expected_group, str) or not expected_group:
            raise ValueError('Training manifest run_group_id is missing')
        if requested.name == 'checkpoint-latest.pt':
            if type(manifest.get('checkpoint_update')) is not int:
                raise ValueError('Training manifest checkpoint_update is missing')
            expected_update = manifest['checkpoint_update']
            if isinstance(manifest.get('launch_settings', {}).get('population'), dict):
                if type(manifest.get('publication_sequence')) is not int:
                    raise ValueError('Population manifest publication_sequence is missing')
                expected_publication_sequence = manifest['publication_sequence']
            else:
                path = _manifest_checkpoint_path(requested.parent, manifest)
    numbered = re.fullmatch(r'checkpoint-(\d+)\.pt', path.name)
    if numbered:
        expected_update = int(numbered.group(1))
    elif path.name == INITIAL_CHECKPOINT_NAME:
        expected_update = 0
    elif re.fullmatch(r'checkpoint-collection-\d+-\d+\.pt', path.name):
        expected_update = int(path.name.split('-')[2])
    if not path.is_file():
        raise FileNotFoundError(f'Release U{expected_update} unavailable: {path.name}')
    if cache is not None and path in cache:
        signature, payload, models = cache[path]
        current = path.stat()
        if (current.st_mtime_ns, current.st_size) != signature:
            raise ValueError('Selected checkpoint changed after preparation')
    else:
        payload = None
        for attempt in range(2):
            before = path.stat()
            signature = (before.st_mtime_ns, before.st_size)
            candidate = _validate_group_checkpoint(torch.load(path,map_location='cpu',weights_only=False,mmap=True))
            after = path.stat()
            if (after.st_mtime_ns, after.st_size) == signature:
                payload = candidate
                break
            if attempt:
                raise ValueError('Selected checkpoint changed while loading')
        assert payload is not None
        models = {}
        if cache is not None:
            cache[path] = (signature,payload,models)
    mtime = signature[0]
    if expected_group is not None and payload['run_group_id'] != expected_group:
        raise ValueError(f'Selected U{expected_update} unavailable: release metadata mismatch')
    if expected_group is not None:
        manifest_architecture = manifest.get('architecture_id')
        manifest_settings = manifest.get('launch_settings')
        if (not isinstance(manifest_architecture, str)
                or manifest_architecture != payload.get('architecture_id')
                # Resuming a supported v6 release writes a v7 manifest before
                # the next publication. Semantic compatibility is checked by
                # the normalized training identity below, also for old releases.
                or manifest.get('match_rules_version') not in (6, 7)
                or not isinstance(manifest_settings, dict)):
            raise ValueError(f'Selected U{expected_update} unavailable: release metadata mismatch')
        manifest_population = isinstance(manifest_settings.get('population'), dict)
        payload_population = isinstance(payload.get('launch_settings', {}).get('population'), dict)
        if manifest_population != payload_population:
            raise ValueError(f'Selected U{expected_update} unavailable: release metadata mismatch')
        phase_settings, phase_history = _release_phase(manifest, path.name, payload['global_update'])
        if payload.get('collection_history', []) != phase_history:
            raise ValueError('Checkpoint collection history differs from its release phase')
        for entry in _collection_history(manifest):
            if path.name == entry['source_checkpoint'] and hashlib.sha256(path.read_bytes()).hexdigest() != entry['source_sha256']:
                raise ValueError('Collection source checkpoint digest mismatch')
        expected_identity = _training_settings_identity(phase_settings, manifest_architecture)
        if 'sample_contract' not in payload.get('training_identity', {}).get('ppo_update', {}):
            expected_identity['ppo_update'].pop('sample_contract')
            if 'role_loss_weighting' in manifest_settings:
                expected_identity['role_loss_weighting'] = manifest_settings['role_loss_weighting']
        if _normalized_training_identity(payload.get('training_identity', {})) != expected_identity:
            raise ValueError(f'Selected U{expected_update} unavailable: release metadata mismatch')
        manifest_scene_diff = scene_contract_diff(
            manifest.get('scene_contract'), payload.get('scene_contract')
        )
        if manifest_scene_diff:
            raise ValueError(
                f'Selected U{expected_update} unavailable: release scene metadata mismatch: '
                + json.dumps(manifest_scene_diff, ensure_ascii=False, sort_keys=True)
            )
        payload_scene_diff = scene_contract_diff(
            payload.get('scene_contract'), payload.get('environment', {}).get('scene_contract')
        )
        if payload_scene_diff:
            raise ValueError(
                f'Selected U{expected_update} unavailable: checkpoint scene metadata mismatch: '
                + json.dumps(payload_scene_diff, ensure_ascii=False, sort_keys=True)
            )
    actual_update = int(payload['global_update'])
    if expected_publication_sequence is None:
        if expected_update is not None and actual_update != expected_update:
            raise ValueError(f'Selected U{expected_update} unavailable: release metadata mismatch')
    else:
        actual_sequence = int(payload.get('publication_sequence', -2))
        if (actual_sequence < expected_publication_sequence
                or (actual_sequence == expected_publication_sequence and actual_update != expected_update)
                or (actual_sequence > expected_publication_sequence and actual_update < expected_update)):
            raise ValueError('Population latest checkpoint conflicts with its manifest')
    return path,mtime,payload,models


def _group_actor_payload(payload, role, checkpoint):
    _validate_group_checkpoint(payload)
    binding = payload['role_bindings'][role]
    if binding['mode'] in {'static', 'rule'}:
        return {'role': role, 'binding': copy.deepcopy(binding), 'source_metadata': None,
                'launch_settings': copy.deepcopy(payload['launch_settings']), 'run_group_id': payload['run_group_id'],
                'global_update': int(payload['global_update']), 'architecture_id': payload['architecture_id']}
    key = binding['parameter_group']
    weight_update = _group_role_weight_update(payload, role)
    global_update = int(payload['global_update'])
    path = Path(checkpoint).resolve()
    weights = payload['parameter_groups'][key]['model'] if binding['mode'] == 'train' else payload['frozen_actors'][key]
    source = {'run_group_id': payload['run_group_id'], 'parameter_group': key, 'role': role,
              'checkpoint_global_update': global_update, 'source_weight_update': weight_update,
              'checkpoint_path': str(path), 'architecture_id': payload['architecture_id'],
              'model_id': binding['model_id'],
              'training_rules': {k:payload['launch_settings'][k] for k in (*sorted(VIEWER_ENVIRONMENT_SETTING_KEYS),'policy_sharing') if k in payload.get('launch_settings',{})}}
    if payload.get('publication_sequence') is not None:
        source['publication_sequence'] = payload['publication_sequence']
    result = {k: copy.deepcopy(payload[k]) for k in ('architecture_id', 'launch_settings', 'training_settings',
        'global_update', 'run_group_id', 'role_bindings', 'scene_contract') if k in payload}
    result['launch_settings'] = _validated_viewer_settings(payload['launch_settings'], historical=True)
    result['training_settings'] = _normalized_training_identity(payload['training_settings'])
    result.update(role=role, source_metadata=source, update_index=weight_update,
        parameter_group_update=int(payload['parameter_groups'][key]['update_index']) if binding['mode']=='train' else None,
        agent_frames=int(payload.get('role_runtime', {}).get(role, {}).get('agent_frames', 0)),
        model={name: tensor.detach().clone() for name,tensor in weights.items() if name.startswith('actor')})
    return result


def _check_source_metadata(payload, role, expected):
    if not expected:
        return
    binding = payload['role_bindings'][role]
    actual = {'run_group_id':payload['run_group_id'], 'parameter_group':binding['parameter_group'],
              'checkpoint_global_update':int(payload['global_update']),
              'source_weight_update':None if binding['mode'] == 'rule' else _group_role_weight_update(payload,role), 'architecture_id':payload['architecture_id']}
    for key,value in actual.items():
        if key in expected and expected[key] != value:
            raise ValueError(f"Selected U{expected.get('checkpoint_global_update', '?')} unavailable: {key} changed")


def _nonnegative_checkpoint_integer(value, field):
    if type(value) is not int or value < 0:
        raise ValueError(f'Invalid checkpoint {field}: expected a nonnegative integer, got {value!r}')
    return value


def _validate_group_checkpoint(payload):
    _nonnegative_checkpoint_integer(payload.get('global_update'), 'global_update')
    if payload.get('collection_history'):
        _collection_history(payload)
    if payload.get('checkpoint_schema')!='mobile_local_sensing_group' or payload.get('match_rules_version') not in (6,7):
        raise ValueError('Old checkpoint input/rules contract is incompatible with this training line')
    if payload.get('match_rules_version') == 6 and payload.get('launch_settings', {}).get('training_settings', {}).get('time_limit_boundary', 'termination') == 'truncation':
        raise ValueError('v6 truncation observation is not compatible with v7 task time')
    if not isinstance(payload.get('launch_settings'), dict):
        raise ValueError('Checkpoint is missing its observation contract settings')
    reject_scene_contract_differences(_resume_scene_contract_diff(payload, payload['launch_settings']))
    state = payload.get('environment', {})
    if state.get('state_schema') != MultiGoalReachEnv.state_schema:
        raise ValueError('Checkpoint terrain environment state schema mismatch')
    if not isinstance(payload.get('parameter_groups'),dict) or not isinstance(payload.get('role_bindings'),dict):
        raise ValueError('Incomplete group checkpoint')
    for key, group in payload['parameter_groups'].items():
        for field in ('update_index', 'optimizer_steps'):
            _nonnegative_checkpoint_integer(group.get(field), f'parameter_groups.{key}.{field}')
    variant = actor_variant(payload['architecture_id'])
    launch = payload['launch_settings']
    rule = _training_rule(launch)
    binding = payload['role_bindings'].get('learner', {})
    if rule is not None:
        if launch.get('learner_rule') != rule:
            raise ValueError('Rule checkpoint requires the complete actual APF configuration')
        _validate_training_control_sources(launch)
        expected_binding = dict(mode='rule', parameter_group=None, model_id=None,
                                source=None, source_update=None, rule=rule)
        if binding != expected_binding:
            raise ValueError('Checkpoint rule binding differs from its declared APF source')
        if payload.get('training_identity', {}).get('policy_sources') != {'learner': {'mode': 'rule', 'rule': rule}}:
            raise ValueError('Checkpoint rule training identity is missing or inconsistent')
        if set(payload['parameter_groups']) != set(parameter_group_bindings(launch).values()):
            raise ValueError('Rule checkpoint contains an unexpected trainable parameter group')
        mapping = parameter_group_bindings(launch)
        fixed_keys = set()
        for role in ROLE_ORDER:
            if role == 'learner':
                continue
            binding = payload['role_bindings'][role]
            expected_mode = 'train' if role in mapping else 'fixed' if role in policy_roles(launch) else 'static'
            if binding.get('mode') != expected_mode:
                raise ValueError('Rule checkpoint has an invalid neural role binding')
            if expected_mode == 'train' and binding.get('parameter_group') != mapping[role]:
                raise ValueError('Rule checkpoint neural parameter group differs from settings')
            if expected_mode == 'fixed':
                fixed_keys.add(binding.get('parameter_group'))
        if set(payload.get('frozen_actors', {})) != fixed_keys:
            raise ValueError('Rule checkpoint has missing or unreferenced frozen neural actors')
    elif any(b.get('mode') == 'rule' for b in payload['role_bindings'].values()):
        raise ValueError('Checkpoint rule binding has no selected rule source')
    if _unified_match_architecture_id(launch, 'learner') != variant.architecture_id:
        raise ValueError('Checkpoint architecture and requested observation contract differ')
    if payload['match_rules_version'] == 7:
        metadata = payload.get('observation_metadata')
        expected = dict(semantics_version=7, architecture_id=variant.architecture_id,
                        actor_features=list(variant.vector_inputs), critic_features=list(variant.critic_vector_input_names))
        if metadata != expected:
            raise ValueError('Checkpoint observation metadata differs from its architecture')
    for state in [*(group['model'] for group in payload['parameter_groups'].values()), *payload.get('frozen_actors',{}).values()]:
        if variant.checkpoint_shape_identity_from(state) != variant.checkpoint_shape_identity:
            raise ValueError('Checkpoint is incompatible with the current 32/1/128 network')
    population = payload.get('launch_settings', {}).get('population')
    if population is not None:
        sequence = _nonnegative_checkpoint_integer(payload.get('publication_sequence'), 'publication_sequence')
        if 'population_state' not in payload or sequence < 1:
            raise ValueError('Incomplete population checkpoint')
        ticks_per_decision = int(_build_config().shared['clock']['physics_ticks_per_decision'])
        state = PopulationState.from_dict(payload['population_state'], ticks_per_decision=ticks_per_decision,
            horizon_ticks=int(launch['training_settings']['maximum_episode_decisions']) * ticks_per_decision)
        if (state.objective != launch['reward_objective'] or state.gamma != launch['training_settings']['gamma']
                or state.condition != population['condition']):
            raise ValueError('Population checkpoint scientific contract differs from launch settings')
        _nonnegative_checkpoint_integer(state.publication_sequence, 'population_state.publication_sequence')
        if state.publication_sequence != sequence:
            raise ValueError('Population checkpoint sequence mismatch')
    return payload


def _group_role_weight_update(payload, role):
    binding = payload['role_bindings'][role]
    if binding['mode'] in {'static', 'rule'}:
        raise ValueError('Static and Rule roles have no policy weights')
    return int(payload['parameter_groups'][binding['parameter_group']]['update_index'] if binding['mode'] == 'train' else binding['source_update'])


def _validate_resume_history(output, checkpoint_update, run_group_id=None):
    metrics = Path(output) / 'metrics.jsonl'
    context = f'Resume refused: {metrics}, expected U{checkpoint_update!r}'
    _nonnegative_checkpoint_integer(checkpoint_update, f'{metrics}: global_update')
    last_update = 0
    line_number = 0
    try:
        with metrics.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if run_group_id is not None and record.get('train/run_group_id') != run_group_id:
                    raise ValueError('history run group identity differs')
                update = record['train/joint_global_update']
                if type(update) is not int or update < 0:
                    raise ValueError(f'invalid JSON integer U {update!r}')
                if update <= last_update:
                    raise ValueError(f'U{update} is not strictly above U{last_update}')
                last_update = update
    except FileNotFoundError as error:
        if checkpoint_update == 0:
            return
        raise ValueError(f'{context}: missing history; evidence preserved') from error
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f'{context}, line {line_number}: {error}; evidence preserved') from error
    if last_update != checkpoint_update:
        raise ValueError(f'{context}, line {line_number}: actual U{last_update} differs from checkpoint; evidence preserved')


def _population_validation_due(stage, global_update):
    scores = list(stage.validation_scores)
    maximum = int(stage.start_update) + int(stage.maximum_updates)
    if scores:
        updates = [int(item[0]) for item in scores]
        if (updates != sorted(set(updates))
                or any(not math.isfinite(float(item[1])) for item in scores)):
            raise ValueError('Population validation scores are not strictly increasing')
        if updates[0] != int(stage.start_update) or any(
            right - left != int(stage.validation_interval)
            for left, right in zip(updates, updates[1:])
        ) or updates[-1] > maximum:
            raise ValueError('Population validation history is off cadence')
        if updates[-1] >= maximum:
            return None
        due = updates[-1] + int(stage.validation_interval)
    else:
        due = int(stage.start_update)
    due = min(due, maximum)
    if int(global_update) > due:
        raise ValueError(f'Population resume passed missing validation U{due}')
    return due


def _population_evaluation_budget_allows(state, sample_count=256, maximum=16640):
    used = validate_population_evaluation_matches(state.evaluation_matches, maximum)
    requested = int(sample_count)
    if used < 0 or requested < 1 or used + requested > int(maximum):
        raise RuntimeError('Population evaluation budget would exceed 16640 matches')


def _restore_population_evaluation_keys(state, evidence_path, *, horizon_ticks, ticks_per_decision=None):
    matches = validate_population_evaluation_matches(state.evaluation_matches)
    if len(set(state.evaluation_keys)) != len(state.evaluation_keys) or len(state.evaluation_keys) * 256 not in (0, matches):
        raise ValueError('Population evaluation identities disagree with the counter')
    keys = []
    records = {}
    path = Path(evidence_path)
    if path.is_file():
        with path.open(encoding='utf-8') as evidence:
            for line_number, line in enumerate(evidence, 1):
                try:
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise ValueError('Evaluation journal record must be an object')
                    if item.get('kind') != 'evaluation':
                        continue
                    version = validate_population_protocol_version(item.get('protocol_version', 1))
                    key = item.get('evaluation_key')
                    if key is None:
                        continue
                    if not isinstance(key, str) or re.fullmatch(r'[0-9a-f]{64}', key) is None:
                        raise ValueError('Population evaluation journal has an invalid evaluation_key')
                    summary = {name:item[name] for name in ('sample_id','sample_count','utility_mean',
                        'utility_standard_error','attacker_wins','terminal_ticks_sum','end_reasons','defender_wins','unresolved','evaluation_decisions','tail_error_bound') if name in item}
                    validate_evaluation_summary(summary, expected_samples=256, horizon_ticks=horizon_ticks,
                        ticks_per_decision=ticks_per_decision, objective=state.objective, gamma=state.gamma,
                        require_extended=version == POPULATION_PROTOCOL_VERSION)
                    signature = {name:item.get(name) for name in ('domain','attack','defense')}
                    signature['summary'] = summary
                    if key in records and records[key] != signature:
                        raise ValueError(f'Population evaluation journal conflicts for {key}')
                    if key not in records:
                        keys.append(key)
                        records[key] = signature
                except (ValueError, TypeError, KeyError) as error:
                    raise ValueError(f'Population evaluation journal {path} line {line_number}: {error}') from error
    counted = matches // 256
    if counted > len(keys):
        raise ValueError('Population evaluation counter exceeds its journal')
    if state.evaluation_keys:
        missing = [key for key in state.evaluation_keys if key not in records]
        if missing:
            raise ValueError(f'Population evaluation journal is missing counted key {missing[0]}')
    else:
        state.evaluation_keys.extend(keys[:counted])


class _PopulationStopRequested(Exception):
    pass


def _initialized_model(architecture, init_seed, device, *, build_critic=True):
    # Isolate CPU initialization without reseeding the CUDA rollout stream.
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        torch.default_generator.manual_seed(int(init_seed))
        model = ActorCritic(architecture, build_critic=build_critic)
    return model.to(device)


def _build_training_group(settings, *, seed=42, device='cuda', sources=None, resume=None,
                          reconfigure_collection=False):
    if reconfigure_collection and resume is None:
        raise ValueError('Collection reconfiguration requires a complete resume checkpoint')
    if resume is not None:
        _validate_training_sample_contract(resume)
    for role,ref in (sources or {}).items():
        if ref and role in train_roles(settings):
            path=require_active_training_checkpoint(Path(ref['path'] if isinstance(ref,dict) else ref))
            _validate_training_sample_contract(torch.load(path,map_location='cpu',weights_only=False))
    for ref in (sources or {}).values():
        if ref:
            require_active_training_checkpoint(Path(ref['path'] if isinstance(ref, dict) else ref))
    settings=_validated_viewer_settings(settings)
    _validate_training_control_sources(settings)
    architecture=_unified_match_architecture_id(settings,'learner')
    _apply_training_schedule(settings,architecture)
    config=_build_config()
    _apply_persisted_randomization(config,settings,architecture)
    device=torch.device(device)
    env=MultiGoalReachEnv(config,int(settings['training_settings']['parallel_environments']),device,
        base_seed=seed,training_target='learner',learner_architecture_id=architecture,patrol_architecture_id=architecture,
        learner_count_range=settings['learner_count_range'],defender_count_range=settings['defender_count_range'],
        learner_speed_ratio_range=settings['learner_speed_ratio_range'],terminate_on_any_casualty=settings['terminate_on_any_casualty'],
        reward_objective=settings['reward_objective'],terminate_on_defender_elimination=settings['terminate_on_defender_elimination'])
    if resume is None:env.reset(build_observation=False)
    else:
        _validate_group_checkpoint(resume)
        if resume['seed']!=seed:
            raise ValueError('Changed training configuration requires a new run')
        if reconfigure_collection:
            _validate_collection_change(resume['launch_settings'], settings, architecture)
        elif _normalized_training_identity(resume['training_identity'])!=_training_settings_identity(settings,architecture):
            raise ValueError('Changed training configuration requires a new run')
        reject_scene_contract_differences(_resume_scene_contract_diff(resume, settings))
        if reconfigure_collection:
            # Abandon unfinished collector episodes without fabricating terminal
            # samples. Start a new deterministic episode epoch beyond all old lanes.
            env.episode_counters.fill_(int(resume['environment']['episode_counters'].max()))
            env.reset(build_observation=False)
        else:
            env.load_state_dict(resume['environment'], source_launch_settings=resume['launch_settings'])
    mapping=parameter_group_bindings(settings)
    if not mapping:raise ValueError('At least one moving role must be Train')
    loaded={}
    source_records={}
    for role,ref in (sources or {}).items():
        if not ref:continue
        if _training_rule(settings, role) is not None:
            raise ValueError('Rule ATK has no model initialization source')
        path=Path(ref['path'] if isinstance(ref,dict) else ref).resolve()
        if path not in loaded:loaded[path]=_validate_group_checkpoint(torch.load(path,map_location='cpu',weights_only=False))
        payload=loaded[path]
        _validate_source_game_contract(payload, settings)
        if payload['architecture_id'] != architecture:
            raise ValueError('Source architecture differs from this training configuration')
        source_role=ref.get('role',role) if isinstance(ref,dict) else role
        _check_source_metadata(payload,source_role,ref if isinstance(ref,dict) else None)
        binding=payload['role_bindings'][source_role]
        if binding['mode'] == 'rule':
            raise ValueError('Rule checkpoints cannot initialize an NN model')
        key=binding['parameter_group']
        expected=ref.get('source_update') if isinstance(ref,dict) else None
        weight_update = _group_role_weight_update(payload,source_role)
        expected_checkpoint=ref.get('checkpoint_update') if isinstance(ref,dict) else None
        if expected is not None and int(expected)!=weight_update:raise ValueError('Source weight update changed before launch')
        if expected_checkpoint is not None and int(expected_checkpoint)!=int(payload['global_update']):raise ValueError('Source checkpoint update changed before launch')
        source_records[role]={'path':str(path),'role':source_role,'run_group_id':payload['run_group_id'],
            'parameter_group':key,'source_update':weight_update,'checkpoint_update':int(payload['global_update']),
            'model_id':binding['model_id'],'checkpoint_path':str(path),'checkpoint_global_update':int(payload['global_update']),
            'source_weight_update':weight_update,'architecture_id':architecture}
    groups={}; bindings={}; frozen={}
    def source_weights(role):
        ref=source_records[role];payload=loaded[Path(ref['path'])]
        key=ref['parameter_group']
        return payload['parameter_groups'][key]['model'] if key in payload['parameter_groups'] else payload['frozen_actors'][key]
    for key in dict.fromkeys(mapping.values()):
        members=[r for r,g in mapping.items() if g==key]
        identities={tuple(source_records[r][k] for k in ('run_group_id','parameter_group','source_weight_update','architecture_id')) if r in source_records else ('Random',) for r in members}
        if len(identities)!=1:raise ValueError('Shared groups require one initialization source')
        model=_initialized_model(architecture, seed, device)
        if resume is not None:model.load_state_dict(resume['parameter_groups'][key]['model'],strict=True)
        elif members[0] in source_records:
            weights = source_weights(members[0])
            if any(name.startswith('critic') for name in weights):
                model.load_state_dict(weights,strict=True)
            else:
                actor_state = {name: tensor for name,tensor in model.state_dict().items() if name.startswith('actor')}
                if weights.keys() != actor_state.keys():
                    raise ValueError('Source frozen Actor keys do not match the selected architecture')
                model.load_state_dict({**model.state_dict(), **weights},strict=True)
        optimizer=torch.optim.Adam(model.parameters(),lr=config.goal_reach_trainer['optimizer']['learning_rate'])
        group=PolicyParameterGroup(key,model,optimizer)
        if resume is not None:
            stored=resume['parameter_groups'][key]
            optimizer.load_state_dict(stored['optimizer']);group.update_index=stored['update_index'];group.optimizer_steps=stored['optimizer_steps']
        groups[key]=group
    for role in ROLE_ORDER:
        rule = _training_rule(settings, role)
        if rule is not None:
            bindings[role] = dict(mode='rule', parameter_group=None, model_id=None, source=None, source_update=None, rule=rule)
            continue
        if role not in env.policy_roles:
            bindings[role]={'mode':'static','parameter_group':None,'model_id':None,'source':None,'source_update':None}
            continue
        mode='train' if role in mapping else 'fixed'
        key=mapping.get(role)
        if mode=='fixed':
            ref=source_records.get(role)
            origin=(resume['role_bindings'][role]['parameter_group'],) if resume else tuple(ref[k] for k in ('run_group_id','parameter_group','source_weight_update','architecture_id')) if ref else ('Random',role)
            key='frozen_'+str(next((i for i,identity in enumerate(frozen) if identity==origin),len(frozen)))
            if origin not in frozen:
                model=_initialized_model(architecture, seed, device, build_critic=False)
                if resume is not None:
                    key=resume['role_bindings'][role]['parameter_group']
                    model.load_state_dict(resume['frozen_actors'][key],strict=True)
                elif ref:model.load_state_dict(actor_deployment_state_dict(model,source_weights(role)),strict=True)
                model.requires_grad_(False).eval()
                frozen[origin]=(key,model)
            key,model=frozen[origin]
        model_id = (source_records[role]['model_id'] if role in source_records else f'Random:{seed}:{role}:{architecture}') if mode=='fixed' else key
        bindings[role]={'mode':mode,'parameter_group':key,'model_id':model_id,'source':source_records.get(role),
                        'source_update':source_records.get(role,{}).get('source_update',0)}
    runtimes={}
    frozen_models={key:model for key,model in frozen.values()}
    observations=env.training_observation_bundle(env.policy_roles,record_neighbors=True)
    for role in env.policy_roles:
        key=bindings[role]['parameter_group'];group=groups.get(key)
        model=group.model if group else frozen_models.get(key)
        obs=observations[role]
        runtime=GoalReachRuntime(env,model,group.optimizer if group else None,obs,obs['actor_alive'].clone(),
            training_target=role,parameter_group=group,rule=bindings[role].get('rule'))
        if resume is not None:
            stored=resume['role_runtime'][role]
            if not reconfigure_collection:
                for name in ('is_init','episode_returns','match_event_counts'):
                    setattr(runtime,name,stored[name].to(device))
            runtime.agent_frames=stored['agent_frames']
        if group:runtime.update_index=group.update_index
        runtimes[role]=runtime
    if resume is not None:
        bindings=copy.deepcopy(resume['role_bindings'])
        load_rng_state(resume['rng'])
    return env,groups,runtimes,bindings,frozen_models


def _training_group_checkpoint(settings,seed,run_group_id,global_update,env,groups,runtimes,bindings,frozen,
                               population_state=None):
    payload = {'checkpoint_schema':'mobile_local_sensing_group','match_rules_version':7,
        'run_group_id':run_group_id,'model_id':run_group_id,'global_update':global_update,'update_index':global_update,
        'architecture_id':env.learner_variant.architecture_id,'seed':seed,'launch_settings':copy.deepcopy(settings),
        'training_identity':_training_settings_identity(settings,env.learner_variant.architecture_id),
        'training_settings':_training_settings_identity(settings,env.learner_variant.architecture_id),
        'observation_metadata': {'semantics_version':7, 'architecture_id':env.learner_variant.architecture_id,
            'actor_features':list(env.learner_variant.vector_inputs),
            'critic_features':list(env.learner_variant.critic_vector_input_names)},
        'role_bindings':copy.deepcopy(bindings),'scene_contract':env.scene_contract,
        'parameter_groups':{key:{'model':g.model.state_dict(),'optimizer':g.optimizer.state_dict(),
            'update_index':g.update_index,'optimizer_steps':g.optimizer_steps,'parameter_count':sum(p.numel() for p in g.model.parameters())} for key,g in groups.items()},
        'frozen_actors':{key:model.state_dict() for key,model in frozen.items()},
        'environment':env.state_dict(),'rng':rng_state(),
        'role_runtime':{r:{'is_init':v.is_init,'episode_returns':v.episode_returns,'match_event_counts':v.match_event_counts,
            'agent_frames':v.agent_frames} for r,v in runtimes.items()}}
    if population_state is not None:
        sequence = population_state.publication_sequence + 1
        state_payload = population_state.to_dict()
        state_payload['publication_sequence'] = sequence
        payload['publication_sequence'] = sequence
        payload['population_state'] = state_payload
    return payload


def _preflight_training_output(output, group_id, *, sources=None, resume=None, release=None,
                               lock=None, owner=None, attempt=None, reconfigure_collection=False):
    """Read-only output identity gate; a grant permits only its preparation files."""
    output = Path(output).resolve()
    if resume is not None:
        if output != Path(release).resolve().parent or group_id != resume['run_group_id']:
            raise ValueError(f'resume_output_identity: {output}: release group/output differs')
        manifest = json.loads((output/'run.json').read_text(encoding='utf-8'))
        if manifest.get('collection_history') or reconfigure_collection:
            if _manifest_checkpoint_path(output, manifest).resolve() != Path(release).resolve():
                raise ValueError('Training must resume the current published collection phase; historical releases are read-only')
        if reconfigure_collection:
            history = _collection_history(resume)
            boundary = output/f"checkpoint-collection-{resume['global_update']:06d}-{len(history)+1:03d}.pt"
            if boundary.exists():
                raise ValueError('Collection boundary already exists; preserve it and reconcile the interrupted publication before retrying')
        return output
    for ref in (sources or {}).values():
        if ref and output == Path(ref['path'] if isinstance(ref, dict) else ref).resolve().parent:
            raise ValueError(f'fresh_output_not_empty: {output}: source group is not a fresh output')
    if 'archive' in (part.casefold() for part in output.parts):
        raise ValueError(f'fresh_output_not_empty: {output}: archive output is read-only')
    if not output.exists():
        if owner is not None:raise ValueError(f'fresh_output_not_empty: {output}: granted directory disappeared')
        return output
    if not output.is_dir():raise ValueError(f'fresh_output_not_empty: {output}: not a directory')
    entries = list(output.iterdir())
    if owner is None:
        if entries:raise ValueError(f'fresh_output_not_empty: {output}: {", ".join(sorted(p.name for p in entries))}')
        return output
    require_actor_training_owner(lock, owner)
    if (owner['output'] != str(output) or owner['run_group_id'] != group_id or owner['launch_attempt_id'] != attempt['id']
            or attempt.get('output_file_id') != output.stat().st_ino):
        raise ValueError(f'fresh_output_not_empty: {output}: grant identity differs')
    allowed = {'run.json', 'training.stdout.log', 'training.stderr.log', TRAINING_STOP_REQUEST_NAME}
    bad = [p.name for p in entries if p.name not in allowed or not p.is_file() or p.is_symlink() or p.resolve().parent != output]
    if bad:raise ValueError(f'fresh_output_not_empty: {output}: {", ".join(sorted(bad))}')
    manifest = json.loads((output/'run.json').read_text(encoding='utf-8'))
    if (manifest.get('run_group_id') != group_id or manifest.get('status') != 'preparing'
        or manifest.get('global_update') != 0 or manifest.get('checkpoint_update', 0) != 0
        or manifest.get('launch_attempt') != attempt):
        raise ValueError(f'fresh_output_not_empty: {output}: preparing manifest identity differs')
    stderr = (output/'training.stderr.log').stat()
    if (type(attempt.get('stderr_file_id')) is not int or stderr.st_ino != attempt['stderr_file_id']
        or type(attempt.get('stderr_offset')) is not int or not 0 <= attempt['stderr_offset'] <= stderr.st_size):
        raise ValueError(f'fresh_output_not_empty: {output}: preparation stderr identity/offset differs')
    return output


def train_main() -> None:
    # Only the training child entrypoint owns this process-wide setting.
    # Preserve an explicitly lower intra-op limit; never change Viewer threads.
    if torch.get_num_threads() > 2:torch.set_num_threads(2)
    if torch.get_num_interop_threads() > 1:torch.set_num_interop_threads(1)
    parser=argparse.ArgumentParser(description='Train one unified ATK/DEF/TGT group')
    parser.add_argument('--settings-file',type=Path)
    parser.add_argument('--training-settings-json')
    parser.add_argument('--first-casualty',choices=('on','off'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--reconfigure-collection',action='store_true',
        help='Explicitly change parallel environments/rollout at fixed work per update; preserve learning state and reset collector episodes')
    parser.add_argument('--sources-json')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--run-group-id')
    parser.add_argument('--launch-attempt-json')
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    if args.reconfigure_collection and not args.resume:
        parser.error('--reconfigure-collection requires --resume')
    launch_attempt=json.loads(args.launch_attempt_json) if args.launch_attempt_json else {
        'id':training_model_timestamp_id(time.time_ns()),'started_at':current_timestamp(),
        'stderr_offset':(args.resume.parent/'training.stderr.log').stat().st_size if args.resume and (args.resume.parent/'training.stderr.log').is_file() else 0}
    # Read-only settings/source checks precede ownership and all GPU construction.
    settings=json.loads(args.settings_file.read_text(encoding='utf-8')) if args.settings_file else _viewer_settings_defaults()
    if args.training_settings_json:settings=json.loads(args.training_settings_json)
    if args.first_casualty:settings['terminate_on_any_casualty']=args.first_casualty=='on'
    settings=_validated_viewer_settings(settings)
    _validate_training_control_sources(settings)
    resume = None
    if args.resume:
        require_active_training_checkpoint(args.resume)
        args.resume, _, resume, _ = _read_group_release(args.resume)
        _validate_training_sample_contract(resume)
        _validate_resume_history(args.resume.resolve().parent,resume['global_update'],resume['run_group_id'])
        if _training_budget_exhausted(settings['training_settings']['updates'], int(resume['global_update'])):
            raise ValueError('Completed: explicitly raise this group budget to continue')
    sources=json.loads(args.sources_json) if args.sources_json else None
    root = _build_config().repository_root
    group_id=resume['run_group_id'] if resume else args.run_group_id or training_model_timestamp_id(time.time_ns())
    if resume is not None and args.run_group_id is not None and args.run_group_id != group_id:
        raise ValueError('Resume run group identity changed')
    output=(args.resume.parent if args.resume else args.output or root/'Artifacts'/'GoalReach'/f'{group_id}-unified-match-seed{args.seed}').resolve()
    if resume is not None:
        if args.output is not None and args.output.resolve() != output:
            raise ValueError(f'resume_output_identity: {args.output.resolve()}: release output differs')
        if resume['seed'] != args.seed:
            raise ValueError('Changed training configuration requires a new run')
        if args.reconfigure_collection:
            _validate_collection_change(resume['launch_settings'], settings, resume['architecture_id'])
            history = _collection_history(resume)
            if history and history[-1]['after_update'] >= resume['global_update']:
                raise ValueError('Complete an update before another collection reconfiguration')
            expected_name = INITIAL_CHECKPOINT_NAME if not resume['global_update'] else f"checkpoint-{resume['global_update']:06d}.pt"
            if args.resume.name != expected_name:
                raise ValueError('Collection reconfiguration requires the completed numbered release')
        elif _normalized_training_identity(resume['training_identity']) != _training_settings_identity(settings, resume['architecture_id']):
            raise ValueError('Changed training configuration requires a new run')
    if not args.launch_attempt_json:
        _preflight_training_output(output, group_id, sources=sources, resume=resume, release=args.resume,
            reconfigure_collection=args.reconfigure_collection)
    for role, ref in (sources or {}).items():
        if ref:
            if _training_rule(settings, role) is not None:
                raise ValueError('Rule ATK cannot use a neural initialization source')
            source = require_active_training_checkpoint(Path(ref['path'] if isinstance(ref, dict) else ref))
            _, _, payload, _ = _read_group_release(source)
            source_role = ref.get('role', role) if isinstance(ref, dict) else role
            if payload['role_bindings'][source_role]['mode'] == 'rule':
                raise ValueError('Rule source has no neural model to initialize this role')
            if role in train_roles(settings):
                _validate_training_sample_contract(payload)
            if payload['architecture_id'] != _unified_match_architecture_id(settings, 'learner'):
                raise ValueError('Source architecture differs from this training configuration')
            _check_source_metadata(payload, ref.get('role',role) if isinstance(ref,dict) else role, ref if isinstance(ref,dict) else {})
    if args.validate_only:
        if args.launch_attempt_json:
            raise ValueError('validate-only cannot accept a launch grant')
        print(json.dumps({'validated':True,'seed':args.seed,'settings':settings},ensure_ascii=False));return
    architecture = _unified_match_architecture_id(settings, 'learner')
    lock = actor_training_lock_path(root, architecture, group_id)
    if args.launch_attempt_json:
        owner = receive_actor_training_lock(lock, output=output, run_group_id=group_id, attempt=launch_attempt)
    else:
        if not output.exists():output.mkdir(parents=True, exist_ok=False)
        owner = claim_actor_training_lock(lock, architecture, output, mode=settings['joint_training_mode'],
            source=args.resume, seed=args.seed, model_id=group_id, target_update=settings['training_settings']['updates'],
            run_group_id=group_id, launch_attempt_id=launch_attempt['id'])
    primary_error = None
    output_accepted = False
    try:
        require_actor_training_owner(lock, owner)
        _preflight_training_output(output, group_id, sources=sources, resume=resume, release=args.resume,
            reconfigure_collection=args.reconfigure_collection,
            **({'lock':lock, 'owner':owner, 'attempt':launch_attempt} if args.launch_attempt_json else {}))
        path = output/'run.json'
        fresh = resume is None and not args.launch_attempt_json
        manifest = json.loads(path.read_text(encoding='utf-8')) if not fresh else {
            'schema_version':3,'kind':'training','checkpoint_schema':'mobile_local_sensing_group',
            'run_group_id':group_id,'architecture_id':architecture,'launch_settings':settings,
            'global_update':int(resume['global_update']) if resume else 0,
            'update_budget':int(settings['training_settings']['updates']),'seed':args.seed,
            'started_at':launch_attempt['started_at'],'training_mode':settings['joint_training_mode'],
            'train_sides':list(train_roles(settings)),
            'role_bindings':_initial_training_role_bindings(settings)}
        if args.launch_attempt_json and ((manifest.get('launch_attempt') or {}).get('id') != launch_attempt['id']
                or manifest.get('run_group_id') != group_id):
            raise RuntimeError('Training launch attempt was superseded')
        manifest.update(status='preparing', launch_attempt=launch_attempt, pid=owner['pid'],
            process_start_token=owner['process_start_token'], owner_kind='worker')
        for field in ('error','error_stage','exit_code','finished_at'):manifest.pop(field,None)
        (create_json_atomic if fresh else write_json_atomic)(path, manifest)
        output_accepted = True
        if resume is not None and not args.launch_attempt_json:
            (output/TRAINING_STOP_REQUEST_NAME).unlink(missing_ok=True)
        args.memory_limit = int(os.environ.get('AUV_MEMORY_TRACE_UPDATES', '0'))
        if args.memory_limit < 0:raise ValueError('AUV_MEMORY_TRACE_UPDATES must be nonnegative')
        if (output/TRAINING_STOP_REQUEST_NAME).is_file():
            manifest.update(status='stopped', exit_code=0, finished_at=current_timestamp())
            write_json_atomic(path, manifest)
            return
        with _cpu_thread_affinity(training=True):
            _train_owned_group(args, settings, resume, sources, root, output, group_id, launch_attempt, lock, owner)
    except BaseException as error:
        primary_error = error
        try:
            if output_accepted:record_training_attempt_failure(output, group_id, launch_attempt, error, 'preparation',
                lock_path=lock, expected_owner=owner)
        except BaseException:pass
        raise
    finally:
        def clear_owned_stop():
            if not output_accepted:return
            manifest=json.loads((output/'run.json').read_text(encoding='utf-8'))
            if (manifest.get('run_group_id')==group_id
                    and (manifest.get('launch_attempt') or {}).get('id')==launch_attempt['id']):
                (output/TRAINING_STOP_REQUEST_NAME).unlink(missing_ok=True)
        cleanup_error = None
        try:
            require_actor_training_owner(lock, owner)
        except BaseException as error:
            cleanup_error = error
        else:
            for stage, action in (
                ('stopflag_cleanup', clear_owned_stop),
                ('lock_release', lambda: release_actor_training_lock(lock, expected_output=output,
                    expected_run_group_id=group_id, expected_owner=owner)),
            ):
                try:
                    require_actor_training_owner(lock, owner)
                    action()
                except BaseException as error:
                    cleanup_error = cleanup_error or error
                    try:
                        record_training_attempt_failure(output, group_id, launch_attempt, error, stage,
                            lock_path=lock, expected_owner=owner, cleanup=True)
                    except BaseException:pass
        if cleanup_error is not None:
            if primary_error is None:
                raise cleanup_error
            try:print(f'Training cleanup: {type(cleanup_error).__name__}: {cleanup_error}', file=sys.stderr)
            except BaseException:pass


def _train_owned_group(args, settings, resume, sources, root, output, group_id, launch_attempt, lock, owner):
    require_actor_training_owner(lock, owner)
    seed_everything(args.seed)
    reconfigure_collection = bool(getattr(args, 'reconfigure_collection', False))
    env,groups,runtimes,bindings,frozen=_build_training_group(settings,seed=args.seed,sources=sources,resume=resume,
        reconfigure_collection=reconfigure_collection)
    collection_history = copy.deepcopy(_collection_history(resume)) if resume else []
    if reconfigure_collection:
        update = int(resume['global_update'])
        collection_history.append(dict(
            after_update=update, from_settings=copy.deepcopy(resume['launch_settings']),
            to_settings=copy.deepcopy(settings), source_checkpoint=args.resume.name,
            source_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest(),
            boundary_checkpoint=f'checkpoint-collection-{update:06d}-{len(collection_history)+1:03d}.pt',
            reset_episode_index=int(env.episode_counters.min()),
            environment_reset='new_episodes_after_max_source_episode; no synthetic terminal samples',
            attempt_id=launch_attempt['id'], created_at=current_timestamp()))
    for role,binding in bindings.items():
        if binding['mode']=='train':binding['model_id']=f"{group_id}:{binding['parameter_group']}"
    global_update=int(resume['global_update']) if resume else 0
    from .training import TrainingMemoryProbe, _memory_sample
    resumed_update = global_update
    memory_limit = args.memory_limit
    if memory_limit:
        env._training_memory_probe = TrainingMemoryProbe(output/'memory.jsonl', env.device, global_update+memory_limit)
        env._training_memory_metadata = dict(
            model_bytes=sum(p.numel()*p.element_size() for group in groups.values() for p in group.model.parameters()),
            optimizer_steps_per_group_update=int(env.config.goal_reach_trainer['rollout']['optimizer_steps_per_update']),
            torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads())
        env._training_memory_probe.update = global_update
        _memory_sample(env, 'initialized')
    budget=int(settings['training_settings']['updates'])
    population_state = None
    population_context = None
    population_actor_cache = {}
    if 'population' in settings:
        population_state = (
            PopulationState.from_dict(resume['population_state'],
                horizon_ticks=int(env.config.goal_reach['clock']['maximum_physics_ticks']),
                ticks_per_decision=int(env.config.shared['clock']['physics_ticks_per_decision']))
            if resume is not None
            else PopulationState(
                condition=settings['population']['condition'],
                objective=settings['reward_objective'],
                gamma=float(settings['training_settings']['gamma']),
            )
        )
        if population_state.condition != settings['population']['condition']:
            raise ValueError('Population condition changed during resume')
        _restore_population_evaluation_keys(
            population_state,
            output/'population.jsonl',
            horizon_ticks=int(env.config.goal_reach['clock']['maximum_physics_ticks']),
            ticks_per_decision=int(env.config.shared['clock']['physics_ticks_per_decision']),
        )
    manifest_path=output/'run.json'
    def require_current_attempt():
        require_actor_training_owner(lock, owner)
        if not manifest_path.is_file():raise RuntimeError('Training manifest disappeared')
        current=json.loads(manifest_path.read_text(encoding='utf-8'))
        attempt=current.get('launch_attempt') or {}
        if attempt.get('id')!=launch_attempt['id']:
            raise RuntimeError('Training launch attempt was superseded')
    try:
        require_current_attempt()
        lock=actor_training_lock_path(root,env.learner_variant.architecture_id,group_id)
        manifest_path=output/'run.json'
        previous=json.loads(manifest_path.read_text(encoding='utf-8')) if resume and manifest_path.is_file() else {}
        manifest={'schema_version':3,'kind':'training','checkpoint_schema':'mobile_local_sensing_group','match_rules_version':7,
            'run_group_id':group_id,'run_id':group_id,'model_id':group_id,'status':'running','pid':os.getpid(),
            'launch_attempt':launch_attempt,'process_start_token':owner['process_start_token'],'owner_kind':'worker',
            'runtime':{**runtime_information(env.device), 'torch_threads':torch.get_num_threads(),
                'torch_interop_threads':torch.get_num_interop_threads(),
                'memory_trace_updates':memory_limit,
                'neighbor_trace':bool(memory_limit)},
            'architecture_id':env.learner_variant.architecture_id,'task_id':'goal_reach','started_at':previous.get('started_at',current_timestamp()),
            'seed':args.seed,'parameter_group_init_seed':args.seed,
            'network':{'embedding':env.learner_variant.attention_embedding_size,'heads':env.learner_variant.attention_heads,'body_hidden':env.learner_variant.body_hidden_size,
                       'actor_inputs':env.learner_variant.vector_input_size,'distribution_outputs':4},
            'launch_settings':settings,'role_bindings':bindings,'train_sides':list(train_roles(settings)),
            'training_mode':settings['joint_training_mode'],'initialization_mode':'continue' if resume else 'fresh' if not sources else 'transfer',
            'update_budget':budget,'global_update':global_update,'metrics_source':str(output/'metrics.jsonl'),
            'scene_contract':env.scene_contract,'resolved_training_settings':{'trainer':env.config.goal_reach_trainer,'ppo_update':{'sample_contract':PPO_SAMPLE_CONTRACT}},
            'resolved_environment_settings':{'shared':env.config.shared,'goal_reach':env.config.goal_reach},'command':sys.argv,
            **repository_state(root,excluded_paths=(output,))}
        if resume is not None:
            manifest['checkpoint_update'] = int(resume['global_update'])
            if previous.get('checkpoint_file'):
                manifest['checkpoint_file'] = previous['checkpoint_file']
        if collection_history:
            manifest['collection_history'] = collection_history
            _collection_history(manifest)
        if reconfigure_collection:
            manifest['initialization_mode'] = 'continue_collection_reconfigured'
        active_started=time.perf_counter()
        active_prior=float(previous['active_elapsed_seconds']) if resume else 0.
        if not math.isfinite(active_prior) or active_prior < 0:
            raise ValueError('Invalid active_elapsed_seconds')
        consistent_boundary=False
        failure_stage='initialization'
    except BaseException as error:
        try:record_training_attempt_failure(output,group_id,launch_attempt,error,'initialization',lock_path=lock,expected_owner=owner)
        except BaseException:pass
        raise
    def publish(initial=False, *, stage=None):
        nonlocal consistent_boundary, failure_stage
        previous_boundary=consistent_boundary
        if stage is not None:
            failure_stage=stage
        consistent_boundary=False
        require_current_attempt()
        update_actor_training_runtime(lock,expected_owner=owner,phase='checkpoint',global_update=global_update)
        if population_state is not None and population_context is not None:
            population_state.opponent = population_context.state_dict()
        payload=_training_group_checkpoint(settings,args.seed,group_id,global_update,env,groups,runtimes,bindings,frozen,
                                           population_state=population_state)
        if collection_history:
            payload['collection_history'] = copy.deepcopy(collection_history)
        if initial:
            save_checkpoint(output/INITIAL_CHECKPOINT_NAME,payload)
        boundary = next((item['boundary_checkpoint'] for item in collection_history
                         if item['after_update'] == global_update), None)
        release_name = boundary or (f'checkpoint-{global_update:06d}.pt' if global_update else INITIAL_CHECKPOINT_NAME)
        if boundary or global_update:
            numbered = output/release_name
            if boundary or collection_history:
                # A collector boundary is a new immutable release, not a fake U
                # and never an overwrite of the original numbered checkpoint.
                if not numbered.exists():save_checkpoint(numbered,payload)
            elif population_state is None or not numbered.exists():save_checkpoint(numbered,payload)
        save_checkpoint(output/'checkpoint-latest.pt',payload)
        _memory_sample(env, 'saved')
        if population_state is not None:
            population_state.publication_sequence = int(payload['publication_sequence'])
        manifest['checkpoint_update']=global_update
        if collection_history:manifest['checkpoint_file']=release_name
        update_manifest()
        consistent_boundary=previous_boundary
    def update_manifest():
        require_current_attempt()
        manifest['global_update']=global_update
        manifest['active_elapsed_seconds']=active_prior+time.perf_counter()-active_started
        manifest['parameter_groups']={key:{'members':[r for r,b in bindings.items() if b['parameter_group']==key],
            'update_index':g.update_index,'optimizer_steps':g.optimizer_steps,'parameter_count':sum(p.numel() for p in g.model.parameters())} for key,g in groups.items()}
        if population_state is not None:
            manifest['publication_sequence'] = population_state.publication_sequence
            manifest['population'] = {
                'condition': population_state.condition,
                'round': population_state.round_index,
                'active_camp': population_state.active_camp,
                'phase_status': population_state.phase_status,
                'phase_update': population_state.phase_update,
                'attack_pool_size': len(population_state.attack_pool),
                'defense_pool_size': len(population_state.defense_pool),
                'matrix_generation': population_state.matrix_generation,
                'restricted_game_value': population_state.restricted_game_value,
                'restricted_game_duality_gap': population_state.restricted_game_duality_gap,
                'train_updates': population_state.train_updates,
                'evaluation_matches': population_state.evaluation_matches,
                'drain_transitions': population_state.drain_transitions,
            }
        write_json_atomic(manifest_path,manifest)
    def population_checkpoint_path(update):
        return output/INITIAL_CHECKPOINT_NAME if update == 0 else output/f'checkpoint-{update:06d}.pt'
    def population_policy_ref(camp, update, *, policy_id=None):
        role = 'learner' if camp == 'attack' else 'defender'
        key = bindings[role]['parameter_group']
        path = population_checkpoint_path(update).resolve()
        if int(update) == global_update:
            state = groups[key].model.state_dict()
        else:
            archived = _validate_group_checkpoint(
                torch.load(path,map_location='cpu',weights_only=False)
            )
            state = archived['parameter_groups'][key]['model']
        digest = actor_state_hash(state)
        return ImmutablePolicyRef(
            policy_id or f'{camp[0]}-u{update}-{digest[:12]}', camp,
            str(path), key, digest, int(update))
    def population_event(kind, **values):
        record = {'event_id':f'{population_state.publication_sequence + 1}:{kind}',
                  'kind':kind,'timestamp':current_timestamp(),'global_update':global_update,
                  'round':population_state.round_index,'active_camp':population_state.active_camp,
                  **values}
        with (output/'population.jsonl').open('a',encoding='utf-8') as evidence:
            evidence.write(json.dumps(record,ensure_ascii=False,default=lambda item:item.tolist() if isinstance(item,torch.Tensor) else str(item))+'\n')
            evidence.flush()
    def population_seed(domain, *parts):
        encoded='|'.join(map(str,(group_id, args.seed, domain, *parts))).encode('utf-8')
        return int.from_bytes(__import__('hashlib').sha256(encoded).digest()[:8],'big') % 2_000_000_000
    def stop_requested():
        return (output/TRAINING_STOP_REQUEST_NAME).is_file()
    def population_evaluate(attack_refs,defense_refs,p,q,domain,*parts):
        from .goal_reach_evaluation import evaluate_population_match, evaluation_sample_id
        sample_seed=population_seed(domain,*parts)
        evaluation_key=__import__('hashlib').sha256(json.dumps({
            'domain':domain,'parts':parts,'attack':[r.actor_hash for r in attack_refs],
            'defense':[r.actor_hash for r in defense_refs],'p':list(p),'q':list(q),
            'objective':settings['reward_objective'],'gamma':settings['training_settings']['gamma'],
            'sample_seed':sample_seed,'sample_count':256,'run_group_id':group_id,
            'architecture_id':env.learner_variant.architecture_id,'scene_contract':env.scene_contract},
            sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()
        expected_sample_id=evaluation_sample_id(settings,attack_refs,defense_refs,p,q,sample_seed,256,
                                               settings['training_settings']['maximum_episode_decisions'])
        evaluation_key=__import__('hashlib').sha256((evaluation_key+expected_sample_id).encode()).hexdigest()
        results=[]
        evidence_path=output/'population.jsonl'
        if evidence_path.is_file():
            with evidence_path.open(encoding='utf-8') as evidence:
                for line in evidence:
                    item=json.loads(line)
                    if item.get('kind')=='evaluation' and item.get('evaluation_key')==evaluation_key:
                        result={key:item[key] for key in ('sample_id','sample_count','utility_mean',
                            'utility_standard_error','attacker_wins','terminal_ticks_sum','end_reasons','defender_wins','unresolved','evaluation_decisions','tail_error_bound')}
                        if (item.get('domain') != domain
                                or item.get('attack') != [ref.policy_id for ref in attack_refs]
                                or item.get('defense') != [ref.policy_id for ref in defense_refs]):
                            raise ValueError('Population evaluation journal identity conflicts with its key')
                        validate_evaluation_summary(
                            result,
                            expected_samples=256,
                            horizon_ticks=int(env.config.goal_reach['clock']['maximum_physics_ticks']),
                            expected_sample_id=expected_sample_id,
                            objective=settings['reward_objective'], gamma=settings['training_settings']['gamma'],
                            evaluation_decisions=settings['training_settings']['maximum_episode_decisions'],
                            ticks_per_decision=int(env.config.shared['clock']['physics_ticks_per_decision']), require_extended=True,
                        )
                        results.append(result)
        if len(results) > 1 and any(item != results[0] for item in results[1:]):
            raise ValueError('Population evaluation journal has conflicting results')
        result=results[0] if results else None
        already_counted=evaluation_key in population_state.evaluation_keys
        if result is None:
            if stop_requested():
                raise _PopulationStopRequested()
            _population_evaluation_budget_allows(population_state)
            result=evaluate_population_match(settings,list(attack_refs),list(defense_refs),list(p),list(q),
                sample_count=256,sample_seed=sample_seed,device=env.device)
            validate_evaluation_summary(
                result,
                expected_samples=256,
                horizon_ticks=int(env.config.goal_reach['clock']['maximum_physics_ticks']),
                expected_sample_id=expected_sample_id,
                objective=settings['reward_objective'], gamma=settings['training_settings']['gamma'],
                evaluation_decisions=settings['training_settings']['maximum_episode_decisions'],
                ticks_per_decision=int(env.config.shared['clock']['physics_ticks_per_decision']), require_extended=True,
            )
        if not results:
            population_event('evaluation',protocol_version=2,domain=domain,evaluation_key=evaluation_key,
                             sample_id=result['sample_id'],sample_count=256,
                             attack=[r.policy_id for r in attack_refs],defense=[r.policy_id for r in defense_refs],
                             utility_mean=result['utility_mean'],utility_standard_error=result['utility_standard_error'],
                             attacker_wins=result['attacker_wins'],terminal_ticks_sum=result['terminal_ticks_sum'],
                             end_reasons=result['end_reasons'],
                             **{key:result[key] for key in ('defender_wins','unresolved','evaluation_decisions','tail_error_bound')},
                             **{key:result[key] for key in ('terminal_ticks','resolved','terminal_outcomes') if key in result})
        if not already_counted:
            _population_evaluation_budget_allows(population_state)
            population_state.evaluation_matches += 256
            population_state.evaluation_keys.append(evaluation_key)
        return result
    def population_load_models(refs):
        from .goal_reach_evaluation import _population_actor
        return tuple(_population_actor(ref,env.learner_variant.architecture_id,env.device,population_actor_cache,settings) for ref in refs)
    def population_phase_context():
        refs = (population_state.defense_pool if population_state.active_camp=='attack'
                else population_state.attack_pool)
        probabilities = (population_state.defense_probabilities if population_state.active_camp=='attack'
                         else population_state.attack_probabilities)
        context=PopulationOpponentContext.create(active_camp=population_state.active_camp,seed=args.seed,
            probabilities=probabilities,policy_ids=[r.policy_id for r in refs],models=population_load_models(refs),
            lane_count=env.batch_size,device=env.device,
            sample_domain=population_seed('opponent',population_state.round_index,population_state.active_camp))
        saved=population_state.opponent
        if isinstance(saved,dict) and saved.get('active_camp')==population_state.active_camp and saved.get('policy_ids')==list(context.policy_ids):
            context.lane_policy_index.copy_(saved['lane_policy_index'].to(env.device))
            context.lane_episode_count.copy_(saved['lane_episode_count'].to(env.device))
        return context
    def population_validate_stage():
        candidate=population_policy_ref(population_state.active_camp,global_update,
            policy_id=f'candidate-{population_state.active_camp}-u{global_update}')
        if population_state.active_camp=='attack':
            result=population_evaluate([candidate],population_state.defense_pool,[1.],
                population_state.defense_probabilities,'validation',population_state.round_index,'attack',global_update)
            score=float(result['utility_mean'])
        else:
            result=population_evaluate(population_state.attack_pool,[candidate],
                population_state.attack_probabilities,[1.],'validation',population_state.round_index,'defense',global_update)
            score=-float(result['utility_mean'])
        stop=population_state.stage.observe(global_update,score)
        population_event('validation',score=score,stop=stop,patience=population_state.stage.patience,
                         raw_best=population_state.stage.raw_best,significant_best=population_state.stage.significant_best)
        publish()
        return stop
    def population_complete_missing_cells():
        for attack in population_state.attack_pool:
            for defense in population_state.defense_pool:
                key=population_state.cell_key(attack.policy_id,defense.policy_id)
                if key in population_state.cells:
                    from .goal_reach_evaluation import evaluation_sample_id
                    cell = population_state.cells[key]
                    # Legacy terminal-only evidence can be restored, but cannot
                    # masquerade as a result for the current scientific request.
                    expected_id = evaluation_sample_id(settings, [attack], [defense], [1.], [1.],
                        population_seed('matrix', attack.policy_id, defense.policy_id), 256,
                        settings['training_settings']['maximum_episode_decisions'])
                    cell.validate(256, int(env.config.goal_reach['clock']['maximum_physics_ticks']),
                        expected_sample_id=expected_id, objective=population_state.objective, gamma=population_state.gamma,
                        evaluation_decisions=settings['training_settings']['maximum_episode_decisions'],
                        ticks_per_decision=int(env.config.shared['clock']['physics_ticks_per_decision']))
                    continue
                result=population_evaluate([attack],[defense],[1.],[1.],'matrix',attack.policy_id,defense.policy_id)
                population_state.cells[key]=MatrixCell(attack.policy_id,defense.policy_id,
                    result['sample_id'],result['sample_count'],result['utility_mean'],
                    result['utility_standard_error'],result['attacker_wins'],
                    result['terminal_ticks_sum'],result['end_reasons'],
                    **{key:result[key] for key in ('defender_wins','unresolved','evaluation_decisions','tail_error_bound')})
                population_state.cells[key].validate(
                    256, int(env.config.goal_reach['clock']['maximum_physics_ticks'])
                )
                publish()
        solved_shape = (
            len(population_state.attack_probabilities) == len(population_state.attack_pool)
            and len(population_state.defense_probabilities) == len(population_state.defense_pool)
        )
        if not solved_shape:
            if stop_requested():
                raise _PopulationStopRequested()
            population_state.solve(
                256, int(env.config.goal_reach['clock']['maximum_physics_ticks'])
            )
            population_event('matrix_solved',generation=population_state.matrix_generation,
                             p=population_state.attack_probabilities,q=population_state.defense_probabilities,
                             value=population_state.restricted_game_value,
                             restricted_game_duality_gap=population_state.restricted_game_duality_gap)
            publish()
    def population_drain():
        nonlocal population_context
        pending=torch.ones(env.batch_size,dtype=torch.bool,device=env.device)
        start=(population_state.drain or {}).get('decisions',0)
        if type(start) is not int or not 0 <= start <= 150:
            raise ValueError('Invalid population drain decisions: expected integer 0..150')
        if population_state.drain and 'pending' in population_state.drain:
            pending=population_state.drain['pending']
            if not isinstance(pending,torch.Tensor) or pending.dtype != torch.bool or pending.shape != (env.batch_size,):
                raise ValueError('Invalid population drain pending: expected batch-shaped bool tensor')
            pending=pending.to(env.device)
        if stop_requested():raise _PopulationStopRequested()
        has_pending=bool(pending.any())
        if has_pending and start == 150:raise RuntimeError('Population phase drain exceeded 150 decisions')
        with torch.no_grad():
            for decision in range(start,150 if has_pending else start):
                population_state.drain={'decisions':decision,'pending':pending.detach().cpu()}
                if stop_requested():
                    raise _PopulationStopRequested()
                active_at_start=int(pending.sum().item())
                sampled,_=_sample_parameter_groups(runtimes,set(population_context.active_roles),population_context)
                joint=env.training_step_joint(sampled['learner'][0],sampled['defender'][0],
                    target_action=sampled['target'][0] if 'target' in sampled else None,build_observation=False)
                pending &= ~joint.learner.info['match_done'].to(torch.bool)
                population_state.drain_transitions += active_at_start
                population_state.drain={'decisions':decision+1,'pending':pending.detach().cpu()}
                observations=env.training_observation_bundle(runtimes,record_neighbors=True)
                for role,runtime in runtimes.items():
                    runtime.observation=observations[role]
                publish()
                if not bool(pending.any()):break
        if stop_requested():raise _PopulationStopRequested()
        if bool(pending.any()):raise RuntimeError('Population phase drain exceeded 150 decisions')
        all_lanes=torch.ones(env.batch_size,dtype=torch.bool,device=env.device)
        env.reset(all_lanes,env_indices=torch.arange(env.batch_size,device=env.device),build_observation=False)
        observations=env.training_observation_bundle(runtimes,record_neighbors=True)
        for role,runtime in runtimes.items():
            runtime.observation=observations[role]
            credit=runtime.observation.get('credit_member',runtime.observation['actor_active']).to(torch.bool)
            runtime.is_init=credit.clone();runtime.episode_returns.zero_();runtime.match_event_counts.zero_()
        population_state.drain=None
        population_state.phase_status='integrating'
        population_event('drain_completed')
        publish()
    def population_finish_phase():
        nonlocal population_context
        best_update=population_state.stage.raw_best_update
        if best_update is None:raise RuntimeError('Population stage has no validated candidate')
        if population_state.phase_status=='training':
            population_state.phase_status='draining'
            population_state.drain={'decisions':0}
            publish()
        if population_state.phase_status=='draining':
            population_drain()
        candidate=population_policy_ref(population_state.active_camp,best_update)
        added=population_state.add_policy(candidate)
        population_event('policy_selected',policy_id=added.policy_id,actor_hash=added.actor_hash,
                         best_update=best_update,best_score=population_state.stage.raw_best)
        population_complete_missing_cells()
        if population_state.active_camp=='attack':
            population_state.active_camp='defense'
        else:
            population_state.active_camp='attack';population_state.round_index+=1
        population_state.phase_update=0;population_state.stage=None;population_state.opponent=None
        population_state.phase_status='training'
        population_context=None
        publish()
    status='completed'
    def finalize_stop():
        nonlocal status, failure_stage, consistent_boundary
        status='stopped'
        failure_stage='stop_publish'
        consistent_boundary=False
        try:
            publish(stage='stop_publish')
        except BaseException as error:
            status='failed'
            manifest['error']=f'{type(error).__name__}: {error}'
            manifest['error_stage']=failure_stage
            raise
        consistent_boundary=True
    try:
        if not resume:
            if population_state is not None:
                population_state.add_policy(population_policy_ref('attack',0,policy_id='a0'))
                population_state.add_policy(population_policy_ref('defense',0,policy_id='d0'))
            publish(initial=True,stage='initial_publish')
        if reconfigure_collection:
            publish(stage='collection_reconfiguration_publish')
        update_manifest()
        with (output/'metrics.jsonl').open('a' if resume else 'x',encoding='utf-8') as stream, SummaryWriter(str(output/'tensorboard'),purge_step=global_update+1 if resume else None) as writer:
            if population_state is not None:
                population_load_models(population_state.attack_pool + population_state.defense_pool)
                if not population_state.attack_probabilities or not population_state.defense_probabilities:
                    failure_stage='population_initial_matrix'
                    population_complete_missing_cells()
                if population_state.round_index < 4 and population_state.stage is None:
                    population_state.stage=StageTracker(start_update=global_update)
                    population_state.phase_update=0
                    publish(stage='population_stage_publish')
                if population_state.round_index < 4:
                    due=_population_validation_due(population_state.stage,global_update)
                    if global_update == due:
                        failure_stage='population_due_validation'
                        population_validate_stage()
                    population_context=population_phase_context()
                    publish(stage='population_context_publish')
                    elapsed=global_update-population_state.stage.start_update
                    stage_stopped=(elapsed>=population_state.stage.maximum_updates or
                        (elapsed>=population_state.stage.minimum_updates and population_state.stage.patience>=2))
                    if population_state.phase_status!='training' or stage_stopped:
                        failure_stage='population_phase_recovery'
                        population_finish_phase()
                        if population_state.round_index < 4:
                            population_state.stage=StageTracker(start_update=global_update)
                            publish(stage='population_recovery_publish')
                            failure_stage='population_initial_validation'
                            population_validate_stage()
                            population_context=population_phase_context();publish(stage='population_context_publish')
            consistent_boundary=True
            while not _training_budget_exhausted(budget, global_update):
                if population_state is not None and population_state.round_index >= 4:
                    break
                if (output/TRAINING_STOP_REQUEST_NAME).is_file():
                    update_actor_training_runtime(lock,expected_owner=owner,phase='stopping',global_update=global_update)
                    raise _PopulationStopRequested()
                started=time.perf_counter()
                if global_update == resumed_update and env.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(env.device)
                if getattr(env, '_training_memory_probe', None) is not None:
                    env._training_memory_probe.update = global_update+1
                _memory_sample(env, 'before_collect')
                keys=list(groups)
                if population_state is not None:
                    active_role='learner' if population_state.active_camp=='attack' else 'defender'
                    selected=[bindings[active_role]['parameter_group']]
                    updating=set(population_context.active_roles)
                else:
                    selected=keys if settings['joint_training_mode']=='joint' else [keys[(global_update//settings['alternating_updates'])%len(keys)]]
                    updating={r for r,b in bindings.items() if b['mode']=='train' and b['parameter_group'] in selected}
                failure_stage='collect'
                consistent_boundary=False
                update_actor_training_runtime(lock,expected_owner=owner,phase='collect',global_update=global_update+1)
                rollouts,collected=collect_joint_goal_reach(runtimes,int(settings['training_settings']['rollout_decisions']),updating,
                    population_context=population_context)
                consistent_boundary=True
                optimization={};role_metrics={};steps_before={key:g.optimizer_steps for key,g in groups.items()}
                consistent_boundary=False
                prepared_samples=validate_parameter_group_rollouts([groups[key] for key in selected],runtimes,rollouts)
                for key in selected:
                    failure_stage='optimize_'+key
                    update_actor_training_runtime(lock,expected_owner=owner,phase='optimize_'+key,global_update=global_update+1)
                    tick=time.perf_counter();results=update_parameter_group(groups[key],runtimes,rollouts,prepared=prepared_samples.pop(key))
                    optimization[key]=time.perf_counter()-tick
                    for role,result in results.items():
                        if not result.finite:raise RuntimeError(f'Nonfinite PPO update: {role}')
                        role_metrics[role]={'optimizer/policy_loss':result.policy_loss,'optimizer/value_loss':result.value_loss,
                            'optimizer/entropy':result.entropy,'optimizer/explained_variance':result.explained_variance,'optimizer/approx_kl':result.approximate_kl,'optimizer/clip_fraction':result.clip_fraction,
                            'train/actor_valid_frames':result.actor_valid_frames,'train/critic_valid_frames':result.critic_valid_frames,
                            'train/sample_contract':PPO_SAMPLE_CONTRACT,
                            'train/learning_rate':groups[key].optimizer.param_groups[0]['lr'],
                            'train/effective_minibatch_agent_frames':result.effective_minibatch_agent_frames,
                            'train/agent_frames_this_update':collected['sides'][role]['agent_frames'],
                            'train/credit_frames_this_update':int(rollouts[role]['credit_member'].sum()),
                            'train/mean_actor_minibatch_frames':collected['sides'][role]['agent_frames']/8,
                            'train/progress/update_index':groups[key].update_index,
                            'train/parameter_group':key,'train/updated_this_update':True,
                            'train/optimizer_steps_this_update':groups[key].optimizer_steps-steps_before[key]}
                for role in train_roles(settings):
                    role_metrics.setdefault(role,{}).setdefault('train/updated_this_update',False)
                    role_metrics[role]['train/parameter_group']=bindings[role]['parameter_group']
                for role, binding in bindings.items():
                    if binding['mode'] == 'rule':
                        role_metrics[role] = {'control/source':'rule', 'control/rule':binding['rule'],
                            'optimizer/policy_loss':None, 'optimizer/value_loss':None, 'optimizer/entropy':None,
                            'train/optimizer_steps_this_update':None, 'train/parameter_group':None,
                            'train/updated_this_update':False}
                for role,scale in collected['sigma'].items():
                    if role in train_roles(settings):role_metrics.setdefault(role,{}).update({'policy/sigma_surge':scale[0],'policy/sigma_yaw':scale[1]})
                global_update+=1;matches=collected['matches']
                if population_state is not None:
                    population_state.phase_update += 1
                    population_state.train_updates += 1
                def mean(value):return value/matches if matches else None
                record={'train/architecture_id':env.learner_variant.architecture_id,'train/task_id':'goal_reach',
                    'train/run_group_id':group_id,'train/joint_global_update':global_update,
                    'train/environment_steps':global_update*env.batch_size*int(settings['training_settings']['rollout_decisions']),
                    'train/optimizer_steps':sum(groups[key].optimizer_steps-steps_before[key] for key in selected),
                    'roles':role_metrics,'parameter_groups':{key:{'optimizer_steps':groups[key].optimizer_steps,'update_index':groups[key].update_index,
                        'optimizer_steps_this_update':groups[key].optimizer_steps-steps_before[key],
                        'effective_minibatch_agent_frames':next((m['train/effective_minibatch_agent_frames'] for r,m in role_metrics.items() if bindings[r]['parameter_group']==key and 'train/effective_minibatch_agent_frames' in m),None),
                        'optimization_seconds':optimization.get(key,0.),'parameter_count':sum(p.numel() for p in groups[key].model.parameters())} for key in groups},
                    'match/count':matches,'match/attacker_win_rate':mean(collected['attacker_wins']),
                    'match/defender_win_rate':mean(collected['defender_wins']),
                    'match/episode_time_s':mean(collected['completed_durations'][0]),
                    'match/protected_time':None if env.bootstrap_on_time_limit else mean(collected['completed_durations'][1]),
                    'match/defense_return':mean(collected['completed_defense_return']),'match/attacker_return':mean(-collected['completed_defense_return']),
                    'event/pair_contacts_per_episode':mean(collected['completed_durations'][2]),
                    'time/collect_seconds':collected['collect_seconds'],'time/collect_reset_seconds':collected['phase_seconds'].get('reset',0.),
                    'time/collect_environment_step_seconds':collected['phase_seconds'].get('environment_step',0.),
                    'time/collect_policy_seconds':collected['phase_seconds'].get('policy',0.),
                    'time/optimization_seconds':sum(optimization.values()),
                    'time/collect_observation_bundle_seconds':collected['phase_seconds'].get('observation_bundle',0.),
                    'time/environment_steps_per_second':collected['environment_steps_per_second'],
                    'collect/reset_calls':collected['reset_calls'],'collect/reset_lanes':collected['reset_lanes'],
                    **{'collect/'+key:value for key,value in collected['scene_sampling'].items()}}
                for label in ('target_capture','target_casualty','elimination','defender_elimination','horizon','truncation','attacker_casualty','defender_casualty'):
                    record[f'match/{label}_rate']=mean(collected[label+'_ends'])
                for role,desc in ROLE_DESCRIPTORS.items():
                    for event in ('capture','static_collision','vessel_collision','obstacle','distance_lost','engagement_boundary'):
                        key=desc['event_role']+'_'+event+'_events'
                        if key in collected:record[f"event/{desc['event_role']}/{event}_per_episode"]=mean(collected[key])
                del rollouts
                if env.device.type == 'cuda':
                    # WDDM can spill idle allocator cache across concurrent training processes.
                    torch.cuda.empty_cache()
                record['time/update_seconds'] = time.perf_counter() - started
                _memory_sample(env, 'rollout_released')
                record['runtime']={'pid':os.getpid(), 'launch_attempt_id':launch_attempt['id'],
                    'update_started_monotonic':started, 'update_finished_monotonic':time.perf_counter()}
                if env.device.type == 'cuda':
                    record['runtime'].update(allocated=torch.cuda.memory_allocated(env.device),
                        reserved=torch.cuda.memory_reserved(env.device))
                    if global_update == resumed_update+1:
                        record['runtime'].update(peak_allocated=torch.cuda.max_memory_allocated(env.device),
                            peak_reserved=torch.cuda.max_memory_reserved(env.device))
                stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
                for key,value in record.items():
                    if isinstance(value,(int,float)):writer.add_scalar(key,value,global_update)
                for role,values in role_metrics.items():
                    for key,value in values.items():
                        if isinstance(value,(int,float)):writer.add_scalar(f'role/{role}/{key}',value,global_update)
                writer.flush();print(json.dumps(record,ensure_ascii=False),flush=True)
                update_manifest()
                consistent_boundary=True
                if population_state is not None:
                    due=_population_validation_due(population_state.stage,global_update)
                    if global_update == due:
                        consistent_boundary=False
                        publish(stage='population_pre_validation_publish')
                        failure_stage='population_validation'
                        stage_complete=population_validate_stage()
                        consistent_boundary=True
                    else:
                        stage_complete=False
                    if stage_complete:
                        failure_stage='population_phase_transition'
                        consistent_boundary=False
                        population_finish_phase()
                        if population_state.round_index < 4:
                            population_state.stage=StageTracker(start_update=global_update)
                            publish(stage='population_transition_publish')
                            population_validate_stage()
                            population_context=population_phase_context()
                            publish(stage='population_context_publish')
                        consistent_boundary=True
                elif global_update == 1 or global_update%50==0:
                    # Publish the first cold update for health/Follow without stopping
                    # the worker; subsequent publications retain the normal cadence.
                    publish(stage='periodic_publish')
            if status != 'stopped' and population_state is not None and population_state.round_index < 4:
                raise RuntimeError('Population phase budget ended before the fixed four rounds')
            failure_stage='final_publish'
            consistent_boundary=False
            publish(stage='final_publish')
            consistent_boundary=True
    except _PopulationStopRequested:
        finalize_stop()
    except KeyboardInterrupt as error:
        if consistent_boundary:
            finalize_stop()
        else:
            status='failed'
            manifest['error']=f'{type(error).__name__}: interrupted_unsafe_update'
            manifest['error_stage']='interrupted_unsafe_update'
            raise
    except BaseException as error:
        status='failed';manifest['error']=f'{type(error).__name__}: {error}';manifest['error_stage']=failure_stage
        raise
    finally:
        primary_exc_info=sys.exc_info()
        primary_error=primary_exc_info[1]
        first_error=primary_error
        first_error_traceback=primary_exc_info[2]
        cleanup_failures=[]
        def emit_cleanup_diagnostic(message):
            try:
                print(message,file=sys.stderr,flush=True)
            except BaseException:
                pass
        def remember_cleanup_failure(stage, error):
            nonlocal status, first_error, first_error_traceback
            cleanup_failures.append((stage,error))
            if first_error is None:
                first_error=error
                first_error_traceback=error.__traceback__
                status='failed'
                manifest.update(status='failed',exit_code=1,
                    error=f'{type(error).__name__}: {error}',error_stage=stage)
            else:
                manifest['cleanup_errors']=[
                    f'{item_stage}: {type(item_error).__name__}: {item_error}'
                    for item_stage,item_error in cleanup_failures
                    if item_error is not first_error
                ]
            emit_cleanup_diagnostic(
                f'Training cleanup {stage} failed: {type(error).__name__}: {error}'
            )
        def owns_current_attempt():
            require_current_attempt()
            require_actor_training_owner(lock, owner)
            return True
        def prove_write_right(stage):
            try:
                if owns_current_attempt():
                    return True
                remember_cleanup_failure(stage,RuntimeError('write ownership is no longer proven'))
            except BaseException as error:
                remember_cleanup_failure(stage,error)
            return False
        def persist_failure_diagnostics():
            if not prove_write_right('failure_manifest_ownership'):
                emit_cleanup_diagnostic(
                    'Training final failed status was not persisted: write ownership is no longer proven'
                )
                return
            try:
                update_manifest()
            except BaseException as error:
                remember_cleanup_failure('failure_manifest',error)
                emit_cleanup_diagnostic(
                    'Training final failed status was not persisted: '
                    f'{type(error).__name__}: {error}'
                )
        try:
            manifest.update(status=status,exit_code=0 if status!='failed' else 1)
            try:manifest['finished_at']=current_timestamp()
            except BaseException as error:remember_cleanup_failure('finished_at',error)
            if prove_write_right('final_manifest_ownership'):
                try:
                    update_manifest()
                except BaseException as error:
                    remember_cleanup_failure('final_manifest',error)
            if cleanup_failures:
                persist_failure_diagnostics()
        finally:
            pass  # The entrypoint's owner scope performs the single lock release.
        if primary_error is None and first_error is not None:
            raise first_error.with_traceback(first_error_traceback)
