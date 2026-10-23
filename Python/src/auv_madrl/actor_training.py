from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import torch

from .actor_variants import (
    actor_variant,
    actor_variant_ids,
)
from .run_evidence import (
    current_timestamp,
    write_json_atomic,
    create_json_atomic,
)

ACTOR_TRAINING_LOCK = Path("Artifacts/GoalReach/patrol-training.lock")
ACTOR_TRAINING_LOCK_PREFIX = "actor-training-"
TRAINING_ARCHIVE_NAME = "archive"
TRAINING_UNDO_DIRECTORY = ".training-undo"
TRAINING_REDO_DIRECTORY = ".training-redo"
TRAINING_STOP_REQUEST_NAME = "training-stop.request"
TRAINING_DELETE_REQUEST_NAME = "training-delete.request"
MAX_PARALLEL_TRAINING_RUNS = 4


def spawn_training_process(command, *, cwd, env, stdout, stderr):
    """Shared hidden worker launch, retaining the venv and the real process identity."""
    command, env = list(command), dict(env)
    if os.name == 'nt':
        command[0] = sys._base_executable
        env['__PYVENV_LAUNCHER__'] = sys.executable
    return subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)


def require_active_training_checkpoint(source: Path) -> Path:
    """Reject archived policy consumption; historical checkpoint readers stay usable."""
    source = Path(source).resolve()
    if TRAINING_ARCHIVE_NAME in source.parts:
        raise ValueError('Archived training group is read-only; cannot Train, Continue or Deploy')
    return source


class TrainingCapacityError(RuntimeError):
    """No process slot is available; the Viewer may leave the intent queued."""
REGISTERED_TRAINING_TASK_IDS = frozenset({"goal_reach", "patrol_pursuit"})
MODEL_TIMESTAMP_ID_LENGTH = 25


def _performance_clock_snapshot() -> tuple[float, float, float]:
    return time.perf_counter(), time.thread_time(), time.process_time()


def _record_elapsed_timing(
    timings: dict[str, list[float]] | None,
    name: str,
    started_at: tuple[float, float, float],
) -> None:
    if timings is None:
        return
    wall_started_at, thread_cpu_started_at, process_cpu_started_at = started_at
    elapsed = (
        time.perf_counter() - wall_started_at,
        time.thread_time() - thread_cpu_started_at,
        time.process_time() - process_cpu_started_at,
    )
    totals = timings.setdefault(name, [0.0, 0.0, 0.0])
    for index, value in enumerate(elapsed):
        totals[index] += max(0.0, value)


def training_run_started_at_ns(output: Path) -> int:
    """Return one shared creation timestamp for every training-run surface."""
    # The allocated ID survives copies/restores; filesystem creation times can
    # collide for concurrently created directories on Windows.
    timestamp_id = output.name[:MODEL_TIMESTAMP_ID_LENGTH]
    if is_training_model_timestamp_id(timestamp_id):
        try:
            seconds = int(datetime.strptime(timestamp_id[:15], "%Y%m%d-%H%M%S").timestamp())
            return seconds * 1_000_000_000 + int(timestamp_id[16:])
        except ValueError:
            pass
    try:
        return int(output.stat().st_ctime_ns)
    except FileNotFoundError:
        pass
    for candidate in (
        output / "checkpoint-initial.pt",
        output / "metrics.jsonl",
        output / "checkpoint-latest.pt",
    ):
        if candidate.is_file():
            return int(candidate.stat().st_mtime_ns)
    return 0


def training_model_timestamp_id(started_at_ns: int) -> str:
    """Format one Run identity as a local timestamp with nanosecond precision."""
    seconds, nanoseconds = divmod(int(started_at_ns), 1_000_000_000)
    return (
        datetime.fromtimestamp(seconds).astimezone().strftime("%Y%m%d-%H%M%S-")
        + f"{nanoseconds:09d}"
    )


def is_training_model_timestamp_id(value: object) -> bool:
    text = str(value or "")
    return (
        len(text) == MODEL_TIMESTAMP_ID_LENGTH
        and text[:8].isdigit()
        and text[8] == "-"
        and text[9:15].isdigit()
        and text[15] == "-"
        and text[16:].isdigit()
    )


def _replace_lock_file(temporary: Path, target: Path) -> None:
    """Replace one lock file despite brief Windows reader/antivirus holds."""

    for attempt in range(8):
        try:
            temporary.replace(target)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.01 * (attempt + 1))


def training_run_sort_key(
    started_at_ns: int,
    checkpoint_update: int = 0,
    identity: str = "",
) -> tuple[int, int, str]:
    """Newest Run first; checkpoints within the same Run use newest update first."""
    return int(started_at_ns), int(checkpoint_update), str(identity)


def _last_jsonl_object(path: Path) -> dict[str, object] | None:
    """Read the newest complete JSON object without rescanning a growing log."""

    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        start = max(0, end - 65_536)
        stream.seek(start)
        tail = stream.read()
    if start:
        _, _, tail = tail.partition(b"\n")
    for line in reversed(tail.splitlines()):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _update_timing(record, group_id, attempt_id):
    """Project one complete record, retaining its actual group/attempt identity."""
    runtime = record.get('runtime')
    update, seconds = record.get('train/joint_global_update'), record.get('time/update_seconds')
    if (not attempt_id or record.get('train/run_group_id') != group_id
        or not isinstance(runtime, dict) or runtime.get('launch_attempt_id') != attempt_id
        or type(update) is not int or not 0 < update <= 2**53 - 1
        or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0
        or not math.isfinite(60 / seconds)):
        return None
    timing = {'run_group_id': record['train/run_group_id'],
              'launch_attempt_id': runtime['launch_attempt_id'], 'update': update, 'seconds': seconds}
    for key in ('collect_seconds', 'optimization_seconds'):
        value = record.get('time/' + key)
        if type(value) in (int, float) and math.isfinite(value) and value >= 0:
            timing[key] = value
    return timing


def checkpoint_training_parameters(payload: Mapping[str, object]) -> dict[str, object]:
    checkpoint_parameters = payload.get("goal_reach")
    goal_reach = (
        checkpoint_parameters if isinstance(checkpoint_parameters, dict) else {}
    )
    checkpoint_settings = payload.get("training_settings")
    settings = checkpoint_settings if isinstance(checkpoint_settings, dict) else {}
    checkpoint_environment = payload.get("environment")
    environment = (
        checkpoint_environment if isinstance(checkpoint_environment, dict) else {}
    )

    def available(name: str, fallback_name: str | None = None) -> object | None:
        for source, key in (
            (settings, name),
            (goal_reach, name),
            (goal_reach, fallback_name or name),
            (environment, name),
        ):
            value = source.get(key)
            if value is not None:
                return value
        return None

    fallbacks = {
        "goal_count_range": "goal_count",
        "goal_radius_range_m": "goal_radius_m",
        "obstacle_density_range_per_1000m2": "obstacle_density_per_1000m2",
        "obstacle_radius_range_m": "obstacle_radius_m",
        "defender_speed_ratio_range": "defender_speed_ratio",
    }
    parameters: dict[str, object] = {}
    for name in (
        "seed",
        "updates",
        "maximum_episode_decisions",
        "parallel_environments",
        "rollout_decisions",
        "minibatch_agent_frames",
        "optimization_epochs",
        "ppo_update_mode",
        "optimizer_steps_per_update",
        "gamma",
        "gae_lambda",
        "bootstrap_on_time_limit",
        "algorithm",
        "reward_method",
        "learner_count_range",
        "learner_speed_ratio_range",
        "include_learner_speed_ratio_observation",
        "entity_streams",
        "entity_self_attention_layers",
        "defender_count_range",
        "defender_count",
        "goal_count_range",
        "goal_radius_range_m",
        "obstacle_density_mean_per_1000m2",
        "obstacle_density_std_per_1000m2",
        "obstacle_radius_mean_m",
        "obstacle_radius_std_m",
        "world_map_seed",
        "defender_speed_ratio_range",
    ):
        value = available(name, fallbacks.get(name))
        if value is not None:
            parameters[name] = list(value) if isinstance(value, tuple) else value

    if "entity_streams" not in parameters:
        legacy_type_encoding = available("include_entity_type_encoding")
        if isinstance(legacy_type_encoding, bool):
            parameters["entity_streams"] = (
                "unified" if legacy_type_encoding else "separate"
            )

    # Preserve every scalar/range training control carried by the checkpoint
    # identity. The Viewer discovers these keys dynamically, so a newly added
    # control appears in new Runs without requiring a second naming schema;
    # older Runs simply have no value for that column.
    non_control_identity_keys = {
        "architecture_id",
        "actor_kind",
        "task_id",
        "sense_radius_m",
        "reward",
        "episode_end",
        "policy_sharing",
        "capture_contact",
        "goal_role",
        "include_entity_type_encoding",
        "train_side",
        "frozen_opponent_checkpoint",
        "frozen_opponent_architecture_id",
        "frozen_opponent_seed",
        "target_update", "target_updates", "target_model_id", "target_checkpoint",
        "training_segment_updates",
    }

    def add_dynamic_parameter(name: object, value: object) -> None:
        key = str(name)
        if (
            not key
            or key.startswith("_")
            or key in parameters
            or key in non_control_identity_keys
            or value is None
        ):
            return
        if isinstance(value, tuple):
            parameters[key] = list(value)
        elif isinstance(value, (bool, int, float, str, list)):
            parameters[key] = value

    for name, value in settings.items():
        add_dynamic_parameter(name, value)
    for container_name in ("trainer_overrides", "ppo_update"):
        nested = settings.get(container_name)
        if not isinstance(nested, Mapping):
            continue
        for name, value in nested.items():
            normalized_name = (
                "ppo_update_mode"
                if container_name == "ppo_update" and name == "mode"
                else name
            )
            add_dynamic_parameter(normalized_name, value)

    if "defender_count_range" not in parameters and "defender_count" in parameters:
        fixed_defender_count = int(parameters.pop("defender_count"))
        parameters["defender_count_range"] = [
            fixed_defender_count,
            fixed_defender_count,
        ]

    bootstrap = parameters.get("bootstrap_on_time_limit")
    if isinstance(bootstrap, bool):
        parameters["time_limit_boundary"] = (
            "truncation" if bootstrap else "termination"
        )
    parallel_collection = settings.get("parallel_collection")
    if isinstance(parallel_collection, dict):
        mode = parallel_collection.get("mode")
        if mode is not None:
            parameters["parallel_collection_mode"] = mode
        resolved = parallel_collection.get("resolved_parallel_environments")
        if resolved is not None:
            parameters["parallel_environments"] = resolved
    trainer_overrides = settings.get("trainer_overrides")
    if (
        "parallel_environments" not in parameters
        and isinstance(trainer_overrides, dict)
        and trainer_overrides.get("parallel_environments") is not None
    ):
        parameters["parallel_environments"] = trainer_overrides[
            "parallel_environments"
        ]
    if "parallel_environments" not in parameters:
        parallel_environments = goal_reach.get("parallel_environments")
        if parallel_environments is not None:
            parameters["parallel_environments"] = parallel_environments
    return parameters


def checkpoint_training_series_signature(
    payload: Mapping[str, object],
) -> str | None:
    training_settings = payload.get("training_settings")
    if not isinstance(training_settings, Mapping):
        return None
    recipe = json.loads(json.dumps(training_settings))
    joint_training = recipe.get("joint_training")
    train_sides: list[str] = []
    if isinstance(joint_training, dict):
        joint_training.pop("run_group_id", None)
        train_sides = sorted(str(side) for side in joint_training.get("train_sides", ()))
        joint_training["train_sides"] = train_sides
    deployment_model_ids = recipe.get("deployment_model_ids")
    if isinstance(deployment_model_ids, dict):
        for side in train_sides:
            deployment_model_ids.pop(side, None)
    parameters = checkpoint_training_parameters(payload)
    parameters.pop("updates", None)
    return json.dumps(
        {
            "training_settings": recipe,
            "training_parameters": parameters,
            "scene_contract": payload.get("scene_contract"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalized_seed_training_identity(
    architecture_id: str,
    training_identity: Mapping[str, object],
) -> dict[str, object]:
    identity = dict(training_identity)
    for redundant_key in (
        "train_side",
        "frozen_opponent_checkpoint",
        "frozen_opponent_architecture_id",
        "frozen_opponent_seed",
    ):
        identity.pop(redundant_key, None)
    if (
        architecture_id in actor_variant_ids()
        and actor_variant(architecture_id).actor_kind == "defender"
    ):
        identity.setdefault("algorithm", "shared_ppo")
    return identity


def model_name_fields(
    architecture_id: str,
    run_name: str,
    *,
    update: int | None,
    reward: float | None,
    payload: Mapping[str, object] | None = None,
    training_mode: object | None = None,
    source_update: object | None = None,
    training_parameters: Mapping[str, object] | None = None,
    architecture_name: str | None = None,
) -> dict[str, object]:
    payload = payload or {}
    restore = payload.get("training_restore")
    if training_mode is None and isinstance(restore, dict):
        training_mode = restore.get("mode")
    initialization = payload.get("weights_initialization")
    if source_update is None and isinstance(initialization, dict):
        source_update = initialization.get("source_update")
    mode_label = {
        "joint": "Joint",
        "alternating": "Alternating",
    }.get(str(training_mode))
    sequence, separator, _suffix = (
        Path(run_name).name.removeprefix("run-").partition("-")
    )
    variant = None
    try:
        variant = actor_variant(architecture_id)
    except ValueError:
        canonical_name = None
        architecture_label = architecture_name or architecture_id
    else:
        canonical_name = variant.output_directory.name
        architecture_label = variant.display_name
    stored_model_id = payload.get("model_id")
    if is_training_model_timestamp_id(stored_model_id):
        run_label = str(stored_model_id)
    elif is_training_model_timestamp_id(Path(run_name).name[:MODEL_TIMESTAMP_ID_LENGTH]):
        run_label = Path(run_name).name[:MODEL_TIMESTAMP_ID_LENGTH]
    elif separator and sequence.isdigit():
        run_label = Path(run_name).name
    elif canonical_name is not None and Path(run_name).name == canonical_name:
        run_label = "Base"
    else:
        run_label = Path(run_name).name.removeprefix("run-")
    parameters = (
        dict(training_parameters)
        if training_parameters is not None
        else checkpoint_training_parameters(payload)
    )
    if variant is not None and variant.input_kind == "entity_attention":
        parameters.setdefault("entity_streams", variant.entity_streams)
        parameters.setdefault(
            "entity_self_attention_layers",
            variant.entity_self_attention_layers,
        )
    return {
        "run": run_label,
        "mode": mode_label,
        "architecture": architecture_label,
        "source_update": (
            None if source_update is None else int(source_update)
        ),
        "update": None if update is None else int(update),
        "reward": None if reward is None else float(reward),
        "parameters": parameters,
    }


def archive_actor_training_run(output: Path, source: Path | None = None) -> Path | None:
    entries = [
        entry
        for entry in output.iterdir()
        if entry.name != TRAINING_ARCHIVE_NAME
    ]
    unknown = [
        entry.name
        for entry in entries
        if not (
            entry.name.startswith("checkpoint-")
            or entry.name
            in {
                "metrics.jsonl",
                "run.json",
                "tensorboard",
                TRAINING_STOP_REQUEST_NAME,
                TRAINING_DELETE_REQUEST_NAME,
            }
            or entry.suffix.lower() == ".log"
        )
    ]
    if unknown:
        raise ValueError(
            "Training output contains unknown files and was not rotated: "
            + ", ".join(sorted(unknown))
        )
    if not entries:
        return source
    archive = output / TRAINING_ARCHIVE_NAME / datetime.now().strftime(
        "run-%Y%m%d-%H%M%S-%f"
    )
    archive.mkdir(parents=True)
    resolved_source = source.resolve() if source is not None else None
    relocated_source = source
    for entry in entries:
        target = archive / entry.name
        resolved_entry = entry.resolve()
        entry.replace(target)
        if resolved_source is not None:
            try:
                source_relative = resolved_source.relative_to(resolved_entry)
            except ValueError:
                continue
            relocated_source = target / source_relative
    return relocated_source


def _query_process_identity(pid: int) -> tuple[str, str | None]:
    """Read OS creation identity without treating denied/partial reads as death."""
    if type(pid) is not int or pid <= 0:
        return 'unknown', None
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ('dead' if ctypes.get_last_error() == 87 else 'unknown'), None
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                return 'unknown', None
            if code.value != 259:
                return 'dead', None
            created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
            if not kernel.GetProcessTimes(handle, *(ctypes.byref(v) for v in (created, exited, system, user))):
                return 'unknown', None
            return 'alive', f'win-filetime:{(created.dwHighDateTime << 32) | created.dwLowDateTime}'
        finally:
            kernel.CloseHandle(handle)
    try:
        # comm may itself contain spaces and parentheses. Fields after its last
        # ')' start at stat field3; field22 is the process start clock tick.
        fields = Path(f'/proc/{pid}/stat').read_text().rpartition(')')[2].split()
        if fields[0] in {'Z', 'X'}:
            return 'dead', None
        start = fields[19]
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if not start.isdigit() or not boot:
            return 'unknown', None
        return 'alive', f'linux:{boot}:{start}'
    except FileNotFoundError:
        return ('dead', None) if not Path(f'/proc/{pid}').exists() else ('unknown', None)
    except (OSError, IndexError, ValueError):
        return 'unknown', None


def process_start_token(pid: int) -> str:
    status, token = _query_process_identity(pid)
    if status != 'alive' or token is None:
        raise RuntimeError(f'Process {pid} creation identity is {status}')
    return token


_LOCK_IDENTITY = ('run_group_id', 'output', 'launch_attempt_id', 'pid', 'process_start_token', 'owner_kind')
_LOCK_OBSERVATIONS = ('process_identity_status', 'capacity_reserved', 'identity_diagnostic', '_lock_path')


def _lock_identity(state):
    if not isinstance(state, Mapping):
        return None
    if (any(not isinstance(state.get(k), str) or not state[k] for k in
            ('run_group_id', 'output', 'launch_attempt_id', 'process_start_token', 'owner_kind'))
            or type(state.get('pid')) is not int or state['pid'] <= 0
            or state['owner_kind'] not in {'launcher', 'worker'}):
        return None
    try:
        return tuple(str(Path(state[k]).resolve()) if k == 'output' else state[k] for k in _LOCK_IDENTITY)
    except (OSError, ValueError):
        return None


def read_actor_training_lock(path: Path, *, prune_stale: bool = True) -> dict[str, object] | None:
    """Read-only, including when legacy callers pass prune_stale=True."""
    try:
        raw = path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as error:
        state = {'identity_diagnostic': str(error)}
    else:
        try:
            state = json.loads(raw)
            if not isinstance(state, dict):
                raise ValueError('Lock is not an object')
        except (ValueError, TypeError) as error:
            state = {'identity_diagnostic': str(error)}
    status = 'unknown'
    if _lock_identity(state) is not None:
        try:
            observed, token = _query_process_identity(state['pid'])
        except OSError as error:
            observed, token = 'unknown', None
            state['identity_diagnostic'] = str(error)
        status = 'dead' if observed == 'dead' or observed == 'alive' and token != state['process_start_token'] else observed
    if status == 'unknown':
        state.setdefault('identity_diagnostic', 'Process/attempt identity cannot be verified; capacity remains reserved')
    state.update(process_identity_status=status, capacity_reserved=status != 'dead', _lock_path=str(path.resolve()))
    return state


def require_actor_training_owner(path: Path, expected: Mapping[str, object]) -> dict:
    state = read_actor_training_lock(path)
    if (_lock_identity(expected) is None or _lock_identity(state) != _lock_identity(expected)
            or state['process_identity_status'] != 'alive' or state['pid'] != os.getpid()):
        raise RuntimeError('Actor training lock ownership changed or is unknown')
    return state


def _write_owned_lock(path, expected, state):
    require_actor_training_owner(path, expected)
    write_json_atomic(path, {k: v for k, v in state.items() if k not in _LOCK_OBSERVATIONS})


def _remove_dead_owner(path, expected):
    current = read_actor_training_lock(path)
    if (_lock_identity(expected) is None or _lock_identity(current) != _lock_identity(expected)
            or current['process_identity_status'] != 'dead'):
        raise RuntimeError('Actor training lock owner is not the same verified dead process')
    # The one launcher serializes claims. The dead process cannot write again.
    path.unlink()


def actor_training_lock_path(
    repository_root: Path,
    architecture_id: str,
    concurrent_run_name: str | None = None,
) -> Path:
    actor_variant(architecture_id)
    run_suffix = ""
    if concurrent_run_name:
        run_number = concurrent_run_name.partition("-")[0]
        timestamp_id = concurrent_run_name[:MODEL_TIMESTAMP_ID_LENGTH]
        if is_training_model_timestamp_id(timestamp_id):
            run_identity = timestamp_id
        elif run_number.isdigit():
            run_identity = run_number
        else:
            raise ValueError(
                "Concurrent Actor training output has no timestamp or numeric run prefix"
            )
        run_suffix = f"-run-{run_identity}"
    return (
        repository_root
        / ACTOR_TRAINING_LOCK.parent
        / f"{ACTOR_TRAINING_LOCK_PREFIX}{architecture_id}{run_suffix}.lock"
    ).resolve()


def _active_actor_training_locks(
    goal_reach_directory: Path,
    *,
    prune_stale: bool = True,
) -> list[dict[str, object]]:
    paths = [goal_reach_directory / ACTOR_TRAINING_LOCK.name]
    paths.extend(goal_reach_directory.glob(f"{ACTOR_TRAINING_LOCK_PREFIX}*.lock"))
    active: list[dict[str, object]] = []
    for path in paths:
        state = read_actor_training_lock(path, prune_stale=prune_stale)
        if state is None:
            continue
        if state['capacity_reserved']:
            active.append(state)
    return active


def _actor_training_run_group_id(state: Mapping[str, object]) -> str:
    run_group_id = str(state.get("run_group_id") or "").strip()
    if run_group_id:
        return run_group_id
    return 'unknown-lock:' + str(state.get('_lock_path') or state.get('pid'))


def claim_actor_training_lock(path: Path, architecture_id: str, output: Path, *, mode: str,
        source: Path | None, seed: int = 42, started_at: str | None = None,
        model_id: str | None = None, source_update: int = 0, target_update: int | None = None,
        run_group_id: str | None = None, launch_attempt_id: str | None = None,
        owner_kind: str = 'worker') -> dict:
    actor_variant(architecture_id)
    state = dict(pid=os.getpid(), process_start_token=process_start_token(os.getpid()),
        owner_kind=owner_kind, launch_attempt_id=launch_attempt_id or training_model_timestamp_id(time.time_ns()),
        run_group_id=run_group_id or model_id or training_model_timestamp_id(time.time_ns()),
        architecture_id=architecture_id, output=str(output.resolve()), mode=mode,
        source=str(source.resolve()) if source else None, seed=int(seed),
        started_at=started_at or current_timestamp(), model_id=str(model_id or ''),
        source_update=int(source_update), target_update=target_update, runtime_phase='preparing')
    return claim_training_lock(path, state)


def claim_training_lock(path, state):
    """Claim one output using the common process/attempt ownership contract."""
    if _lock_identity(state) is None:
        raise ValueError('Incomplete Actor training lock identity')
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = read_actor_training_lock(path)
    if previous is not None:
        if previous['process_identity_status'] != 'dead':
            raise RuntimeError('Another Actor training owner is alive or unknown')
        _remove_dead_owner(path, previous)
    create_json_atomic(path,state)
    return state


def handoff_actor_training_lock(path, bootstrap, child_pid):
    state = require_actor_training_owner(path, bootstrap)
    if state['owner_kind'] != 'launcher':
        raise RuntimeError('Only bootstrap launcher can hand off training')
    manifest = json.loads((Path(state['output']) / 'run.json').read_text(encoding='utf-8'))
    if (manifest.get('run_group_id') != state['run_group_id']
            or (manifest.get('launch_attempt') or {}).get('id') != state['launch_attempt_id']
            or manifest.get('status') == 'failed'):
        raise RuntimeError('Training attempt changed before handoff')
    return handoff_training_lock(path, bootstrap, child_pid)


def handoff_training_lock(path, bootstrap, child_pid):
    state = require_actor_training_owner(path, bootstrap)
    if state['owner_kind'] != 'launcher':
        raise RuntimeError('Only bootstrap launcher can hand off training')
    worker = {**state, 'pid': child_pid, 'process_start_token': process_start_token(child_pid), 'owner_kind': 'worker'}
    try:
        _write_owned_lock(path, bootstrap, worker)
    except BaseException:
        # Atomic replace can succeed before its response is lost. Read exact owner.
        readback = read_actor_training_lock(path)
        if _lock_identity(readback) != _lock_identity(worker):
            raise
    return {k: v for k, v in worker.items() if k not in _LOCK_OBSERVATIONS}


def receive_actor_training_lock(path, *, output, run_group_id, attempt, timeout=30.):
    parent = attempt.get('bootstrap_owner')
    if (_lock_identity(parent) is None or parent['owner_kind'] != 'launcher'
            or parent['run_group_id'] != run_group_id or parent['output'] != str(output.resolve())
            or parent['launch_attempt_id'] != attempt.get('id')):
        raise ValueError('Invalid parent training launch identity')
    expected = {**parent, 'owner_kind': 'worker', 'pid': os.getpid(),
                'process_start_token': process_start_token(os.getpid())}
    deadline = time.monotonic() + min(30., timeout)
    while True:
        current = read_actor_training_lock(path)
        if _lock_identity(current) == _lock_identity(expected):
            return require_actor_training_owner(path, expected)
        if _lock_identity(current) != _lock_identity(parent) or current['process_identity_status'] == 'dead':
            raise RuntimeError('Parent bootstrap died or owner changed before handoff')
        if time.monotonic() >= deadline:
            raise TimeoutError('Training worker lock handoff timed out')
        time.sleep(.05)


def update_actor_training_runtime(path: Path, *, phase: str, global_update: int,
        last_update_reset: Mapping[str, int | float] | None = None,
        preparation: Mapping[str, object] | None = None, expected_owner=None) -> None:
    state = require_actor_training_owner(path, expected_owner)
    state.update(runtime_phase=str(phase), runtime_phase_updated_at_ns=time.time_ns(),
                 runtime_global_update=int(global_update))
    if last_update_reset is not None:
        state['runtime_last_update_reset'] = dict(last_update_reset)
    if preparation is None:
        state.pop('preparation', None)
    else:
        state['preparation'] = dict(preparation)
    _write_owned_lock(path, expected_owner, state)


def release_actor_training_lock(path: Path, *, expected_output: Path | None = None,
        expected_run_group_id: str | None = None, expected_owner=None) -> None:
    state = read_actor_training_lock(path)
    # Evaluation retains its no-argument release interface, still creation-bound.
    expected = expected_owner if expected_owner is not None else state
    state = require_actor_training_owner(path, expected)
    if expected_output is not None and Path(state['output']).resolve() != expected_output.resolve():
        raise RuntimeError('Actor training lock output changed before release')
    if expected_run_group_id is not None and state['run_group_id'] != expected_run_group_id:
        raise RuntimeError('Actor training lock group changed before release')
    path.unlink()


def training_metrics_source(path: Path) -> Path:
    """A role keeps its own identity while referencing the group's sole metric stream."""
    manifest_path = path.parent / "run.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("metrics_source"):
            return Path(manifest["metrics_source"]).resolve()
    return path.resolve()


def role_metric_record(record: dict, role: str) -> dict:
    roles = record.get("roles")
    if not isinstance(roles, dict):
        return record
    return {**{key: value for key, value in record.items() if not key.startswith(("optimizer/", "train/")) and key != "roles"},
            **roles.get(role, {}),
            "train/joint_global_update": record.get("train/joint_global_update"),
            "train/environment_steps": record.get("train/environment_steps")}


def record_training_attempt_failure(output, group_id, attempt, error, stage, *, lock_path=None,
                                    expected_owner=None, exit_code=1, cleanup=False):
    """Only the exact attempt's proven owner may persist a failure; first wins."""
    if output is None or lock_path is None or expected_owner is None:
        return False
    try:
        state = require_actor_training_owner(lock_path, expected_owner)
        if (state['run_group_id'] != group_id or state['output'] != str(Path(output).resolve())
                or state['launch_attempt_id'] != attempt.get('id')):
            return False
        path = Path(output) / 'run.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        previous = manifest.get('launch_attempt') or {}
        if manifest.get('run_group_id') != group_id or previous.get('id') != attempt.get('id'):
            return False
        if manifest.get('status') == 'failed' and manifest.get('error'):
            if cleanup and manifest['error'] != f'{type(error).__name__}: {error}':
                item = f'{stage}: {type(error).__name__}: {error}'
                errors = manifest.setdefault('cleanup_errors', [])
                if item not in errors:errors.append(item)
                require_actor_training_owner(lock_path, expected_owner)
                write_json_atomic(path, manifest)
            return True
        require_actor_training_owner(lock_path, expected_owner)
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}',
            error_stage=stage, exit_code=exit_code, finished_at=current_timestamp())
        write_json_atomic(path, manifest)
        return True
    except Exception as diagnostic_error:
        try:print(f'Training failure diagnostic: {type(diagnostic_error).__name__}: {diagnostic_error}',file=sys.stderr)
        except Exception:pass
        return False  # A diagnostic failure must not mask the primary exception.


class ActorTrainingLauncher:
    def __init__(
        self,
        repository_root: Path,
    ):
        self.repository_root = repository_root.resolve()
        self.lock_path = (self.repository_root / ACTOR_TRAINING_LOCK).resolve()
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._started_sessions: dict[str, dict[str, object]] = {}
        self._launch_lock = threading.Lock()
        self._started_name_fields: dict[str, dict[str, object]] = {}
        self._last_exit_codes: dict[str, int] = {}
        self._last_group_exit_codes: dict[str, int] = {}
        self._group_process_errors: dict[str, str] = {}
        self._diagnostic_cache: dict[Path, tuple[object, str | None]] = {}
        self._last_process_error: str | None = None
        self._last_started_run_group_id: str | None = None
        self._training_run_registry_cache: dict[Path, dict[str, object]] | None = None
        self._registry_last_discovery = float('-inf')
        self._registry_hot_outputs: set[Path] = set()
        self.registry_errors = []
        self._archived_options_cache_key: tuple[object, ...] | None = None
        self._archived_options_cache: list[dict[str, object]] = []
        self._deployment_run_options_cache: dict[
            tuple[str, Path], tuple[tuple[object, ...], list[dict[str, object]]]
        ] = {}
        self._checkpoint_architecture_cache: dict[
            Path, tuple[tuple[int, int], str | None]
        ] = {}
        self._checkpoint_model_id_cache: dict[
            Path, tuple[tuple[int, int], str]
        ] = {}
        self._latest_metrics_cache: dict[
            Path, tuple[tuple[int, int], dict[str, object] | None]
        ] = {}
        self._last_allocated_model_id_ns = 0

    @staticmethod
    def _traceback_summary(stderr_path: Path, start_offset: int = 0) -> str | None:
        try:
            with stderr_path.open("rb") as stderr:
                stderr.seek(0, os.SEEK_END)
                if start_offset > stderr.tell():
                    return 'Current attempt stderr boundary unavailable: log truncated or rotated'
                stderr.seek(max(start_offset, stderr.tell() - 65_536))
                raw = stderr.read()
                try:
                    decoded = raw.decode('utf-8')
                except UnicodeDecodeError:
                    decoded = raw.decode('mbcs' if os.name == 'nt' else 'utf-8', errors='replace')
                lines = decoded.splitlines()
        except OSError:
            return None
        blocks = []
        location = None
        chained = False
        for line in lines:
            if line.startswith(('During handling of the above exception', 'The above exception was the direct cause')):
                chained = True
            if line.startswith('Traceback (most recent call last):'):
                if not chained:
                    blocks.clear()
                chained = False
                location = None
            if line.lstrip().startswith('File "'):
                location = line.strip()
            if re.match(r'^(?:[\w]+\.)*[\w]*(?:Error|Exception|Interrupt|Exit|Warning)(?::|$)', line):
                blocks.append(line.strip() + (f' ({location})' if location else ''))
        if blocks:
            return ('; caused by: '.join(reversed(blocks[-2:])))[:1000]
        native = [line.strip() for line in lines if line.strip() and not line.startswith(
            ('Search for ', 'For more detailed error information', 'Traceback ('))]
        return '\n'.join(native[-4:])[:1000] or None

    def _attempt_diagnostic(self, output, manifest):
        path = Path(output)/'training.stderr.log'
        attempt = manifest.get('launch_attempt') or {}
        if not attempt:
            return manifest.get('error') or 'No current-attempt diagnostic; historical stderr is not attributed'
        stat = path.stat() if path.is_file() else None
        key = (attempt.get('id'),attempt.get('stderr_offset'),attempt.get('stderr_file_id'),
               (stat.st_ino,stat.st_size,stat.st_mtime_ns) if stat else None)
        cached = self._diagnostic_cache.get(path)
        if cached is None or cached[0] != key:
            if stat and attempt.get('stderr_file_id') not in (None,stat.st_ino):
                summary = 'Current attempt stderr boundary unavailable: log rotated'
            else:
                summary = self._traceback_summary(path,int(attempt.get('stderr_offset',0)))
            cached=(key,summary);self._diagnostic_cache[path]=cached
        return cached[1] or manifest.get('error') or f"Training process exited with code {manifest.get('exit_code','unknown')} without a current traceback"

    def _variant_output(self, architecture_id: str) -> Path:
        return (self.repository_root / actor_variant(architecture_id).output_directory).resolve()

    def _latest_metrics(self, path: Path) -> dict[str, object] | None:
        path = path.resolve()
        source = training_metrics_source(path)
        stat = source.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = self._latest_metrics_cache.get(path)
        if cached is not None and cached[0] == signature:
            return cached[1]
        latest = _last_jsonl_object(source)
        if latest and "roles" in latest:
            manifest = json.loads((path.parent / "run.json").read_text(encoding="utf-8"))
            latest = role_metric_record(latest, manifest["train_side"])
        self._latest_metrics_cache[path] = (signature, latest)
        return latest

    def allocate_model_id(self) -> str:
        started_at_ns = max(time.time_ns(), self._last_allocated_model_id_ns + 1)
        self._last_allocated_model_id_ns = started_at_ns
        return training_model_timestamp_id(started_at_ns)

    def _recorded_checkpoint_path(self, value: object) -> Path | None:
        text = str(value or "").strip()
        if not text:
            return None
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = self.repository_root / candidate
        if candidate.is_file():
            return candidate.resolve()
        normalized = text.replace("\\", "/")
        marker = "Artifacts/GoalReach/"
        marker_index = normalized.find(marker)
        if marker_index < 0:
            return None
        relocated = self.repository_root / normalized[marker_index:]
        return relocated.resolve() if relocated.is_file() else None

    def checkpoint_model_timestamp_id(
        self,
        checkpoint: Path,
        payload: Mapping[str, object] | None = None,
        *,
        visited: set[Path] | None = None,
    ) -> str:
        checkpoint = checkpoint.resolve()
        stat = checkpoint.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = self._checkpoint_model_id_cache.get(checkpoint)
        if cached is not None and cached[0] == signature:
            return cached[1]
        loaded = payload
        if loaded is None:
            value = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
            loaded = value if isinstance(value, dict) else {}
        stored = loaded.get("model_id")
        restore = loaded.get("training_restore")
        source_checkpoint = (
            self._recorded_checkpoint_path(restore.get("source_checkpoint"))
            if isinstance(restore, Mapping) and restore.get("source_checkpoint")
            else None
        )
        visited = set() if visited is None else set(visited)
        visited.add(checkpoint)
        source_payload: Mapping[str, object] | None = None
        if source_checkpoint is not None and source_checkpoint not in visited:
            source_value = torch.load(
                source_checkpoint,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            if isinstance(source_value, Mapping):
                source_payload = source_value
        current_signature = checkpoint_training_series_signature(loaded)
        source_signature = (
            checkpoint_training_series_signature(source_payload)
            if source_payload is not None
            else None
        )
        same_training_series = (
            source_payload is not None
            and current_signature is not None
            and source_signature is not None
            and current_signature == source_signature
        )
        if is_training_model_timestamp_id(stored):
            model_id = str(stored)
        elif same_training_series:
            assert source_checkpoint is not None
            model_id = self.checkpoint_model_timestamp_id(
                source_checkpoint,
                source_payload,
                visited=visited,
            )
        elif (
            is_training_model_timestamp_id(stored)
            and (
                source_payload is None
                or str(stored) != str(source_payload.get("model_id") or "")
            )
        ):
            model_id = str(stored)
        else:
            model_id = training_model_timestamp_id(
                training_run_started_at_ns(checkpoint.parent)
            )
        self._checkpoint_model_id_cache[checkpoint] = (signature, model_id)
        return model_id

    def canonical_model_reference(self, value: object) -> str:
        text = str(value or "").strip()
        if text in {"random_initial", "random_current_architecture"}:
            return "random_current_architecture"
        if is_training_model_timestamp_id(text):
            return text
        for prefix in ("saved_target:", "archived_target:"):
            if text.startswith(prefix):
                text = text.removeprefix(prefix)
                break
        checkpoint = self._recorded_checkpoint_path(text)
        return (
            self.checkpoint_model_timestamp_id(checkpoint)
            if checkpoint is not None
            else text
        )

    def _resolve_source(self, architecture_id: str, source_id: str) -> Path:
        source = (self.repository_root / source_id).resolve()
        output = self._variant_output(architecture_id)
        try:
            source.relative_to(output)
        except ValueError as error:
            raise ValueError("Training source is outside the selected Actor scenario") from error
        if source.suffix.lower() != ".pt" or not source.is_file():
            raise ValueError("Selected training source checkpoint does not exist")
        return source

    def _resolve_model_checkpoint(self, source_id: str) -> Path:
        source = (self.repository_root / source_id).resolve()
        artifact_root = (self.repository_root / "Artifacts" / "GoalReach").resolve()
        try:
            source.relative_to(artifact_root)
        except ValueError as error:
            raise ValueError("Model checkpoint is outside GoalReach artifacts") from error
        if not source.name.startswith('checkpoint-') or source.suffix != '.pt' or not source.is_file():
            raise ValueError("Selected model checkpoint does not exist")
        return source

    @staticmethod
    def _candidate_checkpoint_paths(output: Path) -> list[Path]:
        latest = output / "checkpoint-latest.pt"
        return [latest] if latest.is_file() else []

    @staticmethod
    def _archived_checkpoint_paths(output: Path) -> list[tuple[Path, Path]]:
        archive = output / TRAINING_ARCHIVE_NAME
        if not archive.is_dir():
            return []
        return [
            (run, path)
            for run in sorted(archive.glob("run-*"), reverse=True)
            if run.is_dir()
            for path in ActorTrainingLauncher._candidate_checkpoint_paths(run)
        ]

    @staticmethod
    def _metric_values(
        output: Path, metric_name: str
    ) -> dict[int, float | None]:
        metrics = training_metrics_source(output / "metrics.jsonl")
        if not metrics.is_file():
            return {}
        values: dict[int, float | None] = {}
        for line in metrics.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                update = int(record["train/progress/update_index"])
                value = record.get(metric_name)
                values[update] = None if value is None else float(value)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return values

    @staticmethod
    def _rewards(output: Path) -> dict[int, float | None]:
        return ActorTrainingLauncher._metric_values(output, "rollout/ep_rew_mean")

    @staticmethod
    def _success_rates(output: Path) -> dict[int, float | None]:
        return ActorTrainingLauncher._metric_values(output, "rollout/success_rate")

    @staticmethod
    def _collision_rates(output: Path) -> dict[int, float | None]:
        return ActorTrainingLauncher._metric_values(output, "rollout/collision_rate")

    @staticmethod
    def _range_label(value: object) -> str:
        if not isinstance(value, list | tuple) or len(value) != 2:
            return "?"
        lower, upper = (int(item) for item in value)
        return str(lower) if lower == upper else f"{lower}–{upper}"

    def archived_checkpoint_options(self, architecture_id):
        return [{'id':'archived_target:'+str(Path(run['output']).relative_to(self.repository_root)/'checkpoint-latest.pt'),
                 'kind':'archived_trained_target','label':'Archived · unavailable for current deployment',
                 'disabled':True,'source_architecture_id':architecture_id,
                 'source_update':run['manifest'].get('checkpoint_update',0),
                 'model_timestamp_id':run['manifest'].get('run_group_id'),
                 'source_id':str(Path(run['output']).relative_to(self.repository_root)/'checkpoint-latest.pt')}
                for run in self.training_run_registry(architecture_id=architecture_id) if run['archived']]

    def resolve_archived_checkpoint(
        self,
        architecture_id: str,
        source_id: str,
    ) -> Path:
        source = self._resolve_source(architecture_id, source_id)
        archive = self._variant_output(architecture_id) / TRAINING_ARCHIVE_NAME
        try:
            relative = source.relative_to(archive.resolve())
        except ValueError as error:
            raise ValueError(
                "Archived checkpoint is outside this Actor scenario archive"
            ) from error
        top_level = len(relative.parts) == 2 and relative.name.startswith("checkpoint-")
        if not relative.parts[0].startswith("run-") or not top_level:
            raise ValueError("Selected file is not an archived training checkpoint")
        if self._checkpoint_architecture_id(source) != architecture_id:
            raise ValueError("Archived checkpoint belongs to another Actor scenario")
        return source

    def load_archived_checkpoint(
        self,
        architecture_id: str,
        source_id: str,
        *,
        role: str,
        map_location: str | torch.device = "cpu",
    ) -> tuple[Path, dict[str, object]]:
        source = self.resolve_archived_checkpoint(architecture_id, source_id)
        return self._load_model_checkpoint(
            architecture_id,
            source.relative_to(self.repository_root).as_posix(),
            role=role,
            map_location=map_location,
        )

    @staticmethod
    def _reward_label(value: float | None) -> str:
        return "—" if value is None else f"{value:.3f}"

    def training_metrics_paths(
        self,
        *,
        architecture_id: str | None = None,
        top_level_only: bool = False,
    ) -> list[Path]:
        return [
            (self.repository_root / str(run["metrics_id"])).resolve()
            for run in self.training_run_registry(
                architecture_id=architecture_id,
                top_level_only=top_level_only,
            )
        ]

    def training_run_registry(
        self,
        *,
        architecture_id: str | None = None,
        top_level_only: bool = False,
    ) -> list[dict[str, object]]:
        """Authoritative visible Run index shared by every model/data consumer."""
        artifact_root = (self.repository_root / "Artifacts" / "GoalReach").resolve()
        if self._training_run_registry_cache is None:
            self.refresh_training_run_registry()
        runs = [
            dict(run)
            for run in self._training_run_registry_cache.values()
            if (
                architecture_id is None
                or str(run["architecture_id"]) == architecture_id
            )
            and (
                not top_level_only
                or Path(str(run["output"])).resolve().parent == artifact_root
            )
        ]
        return sorted(
            runs,
            key=lambda run: training_run_sort_key(
                int(run["run_started_at_ns"]),
                identity=str(run["run_id"]),
            ),
            reverse=True,
        )

    def refresh_training_run_registry(self, outputs=None) -> None:
        artifact_root = (self.repository_root / "Artifacts" / "GoalReach").resolve()
        previous = self._training_run_registry_cache or {}
        discovered = {} if outputs is None else dict(previous)
        self.registry_errors = []
        try:
            if outputs is None:
                paths = {path for path in artifact_root.rglob("metrics.jsonl")}
                paths.update(path.parent / "metrics.jsonl" for path in artifact_root.rglob("run.json"))
            else:
                paths = {Path(output).resolve() / 'metrics.jsonl' for output in outputs}
        except OSError as error:
            self.registry_errors.append({'source': str(artifact_root), 'error': str(error)})
            return
        for path in paths:
            output = path.parent.resolve()
            try:
                run = self._training_run_from_metrics(path)
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.registry_errors.append({"source": str(path), "error": str(error)})
                if output in previous:discovered[output] = previous[output]
                continue
            if run is not None:
                discovered[output] = run
            else:
                discovered.pop(output, None)
        self._training_run_registry_cache = discovered
        if outputs is None:self._registry_last_discovery = time.monotonic()

    def _training_run_from_metrics(
        self, metrics_path: Path
    ) -> dict[str, object] | None:
        artifact_root = (self.repository_root / "Artifacts" / "GoalReach").resolve()
        if not artifact_root.is_dir():
            return None
        undo_root = (artifact_root / TRAINING_UNDO_DIRECTORY).resolve()
        resolved = metrics_path.resolve()
        if resolved.is_relative_to(undo_root):
            return None
        output = resolved.parent
        try:
            relative_output = output.relative_to(artifact_root)
            manifest_path = output / "run.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
            if manifest.get("checkpoint_schema") != "mobile_local_sensing_group":
                return None
            actor_variant(manifest['architecture_id'])
            for field in ('run_group_id','started_at','status','training_mode'):
                if not isinstance(manifest.get(field),str) or not manifest[field]:
                    raise ValueError('Invalid manifest '+field)
            for field in ('role_bindings','launch_settings'):
                if not isinstance(manifest.get(field),dict):raise ValueError('Invalid manifest '+field)
            for field in ('global_update','update_budget'):
                if type(manifest.get(field)) is not int or manifest[field]<0:raise ValueError('Invalid manifest '+field)
            for role,binding in manifest['role_bindings'].items():
                if not isinstance(binding,dict) or binding.get('mode') not in {'train','fixed','static','rule'}:
                    raise ValueError('Invalid role binding '+role)
                if 'parameter_group' not in binding:raise ValueError('Missing parameter_group '+role)
                if binding['mode'] == 'rule':
                    from .goal_reach import _training_rule
                    rule = _training_rule(manifest['launch_settings'], role)
                    if rule is None or binding.get('rule') != rule or binding['parameter_group'] is not None or binding.get('model_id') is not None:
                        raise ValueError('Invalid Rule source binding '+role)
            if 'network' in manifest and not isinstance(manifest['network'],dict):raise ValueError('Invalid manifest network')
            if manifest.get("architecture_id"):
                identity = {"train/task_id": manifest.get("task_id") or actor_variant(manifest["architecture_id"]).task_id, "train/architecture_id": manifest.get("architecture_id")}
            else:
                with training_metrics_source(resolved).open("r", encoding="utf-8") as stream:
                    identity = next((json.loads(line) for line in stream if line.strip()), {})
        except FileNotFoundError:
            return None
        task_id = identity.get("train/task_id")
        architecture_id = identity.get("train/architecture_id")
        if (
            task_id not in REGISTERED_TRAINING_TASK_IDS
            or not isinstance(architecture_id, str)
        ):
            return None
        return {
            "run_id": output.relative_to(self.repository_root).as_posix(),
            "metrics_id": resolved.relative_to(self.repository_root).as_posix(),
            "architecture_id": architecture_id,
            "task_id": task_id,
            "output": str(output),
            "run_name": output.name,
            "run_started_at_ns": training_run_started_at_ns(output),
            "archived": TRAINING_ARCHIVE_NAME in relative_output.parts,
            "manifest": manifest,
        }

    def register_training_run(self, output: Path) -> None:
        cache = self._training_run_registry_cache
        resolved_output = output.resolve()
        if cache is None:
            return
        self.refresh_training_run_registry([resolved_output])
        self._registry_hot_outputs.add(resolved_output)

    def forget_training_run(self, output: Path) -> None:
        cache = self._training_run_registry_cache
        if cache is None:
            return
        resolved_output = output.resolve()
        self._training_run_registry_cache = {
            path: run
            for path, run in cache.items()
            if path != resolved_output and not path.is_relative_to(resolved_output)
        }

    def inventory_clone(self) -> ActorTrainingLauncher:
        clone = ActorTrainingLauncher(self.repository_root)
        clone._registry_last_discovery = self._registry_last_discovery
        clone._registry_hot_outputs = set(self._registry_hot_outputs)
        clone.registry_errors = list(self.registry_errors)
        clone._group_tail_cache = dict(getattr(self, '_group_tail_cache', {}))
        if self._training_run_registry_cache is not None:
            clone._training_run_registry_cache = {
                path: dict(run)
                for path, run in self._training_run_registry_cache.items()
            }
        clone._archived_options_cache_key = self._archived_options_cache_key
        clone._archived_options_cache = [
            dict(option) for option in self._archived_options_cache
        ]
        clone._deployment_run_options_cache = {
            key: (signature, [dict(option) for option in options])
            for key, (signature, options) in self._deployment_run_options_cache.items()
        }
        clone._checkpoint_architecture_cache = dict(
            self._checkpoint_architecture_cache
        )
        clone._started_name_fields = {
            run_id: dict(fields)
            for run_id, fields in self._started_name_fields.items()
        }
        clone._latest_metrics_cache = {
            path: (
                signature,
                None if latest is None else dict(latest),
            )
            for path, (signature, latest) in self._latest_metrics_cache.items()
        }
        return clone

    def invalidate_checkpoint_option_caches(self) -> None:
        self._archived_options_cache_key = None
        self._archived_options_cache = []
        self._deployment_run_options_cache.clear()

    @staticmethod
    def _deployment_checkpoint_path(output: Path) -> Path | None:
        latest = output / "checkpoint-latest.pt"
        return latest if latest.is_file() else None

    @staticmethod
    def _deployment_checkpoint_paths(output: Path) -> list[Path]:
        primary = ActorTrainingLauncher._deployment_checkpoint_path(output)
        return [primary] if primary is not None else []

    @staticmethod
    def _run_started_at_ns(output: Path) -> int:
        return training_run_started_at_ns(output)

    @staticmethod
    def _run_label(output: Path) -> str:
        timestamp_id = output.name[:MODEL_TIMESTAMP_ID_LENGTH]
        if is_training_model_timestamp_id(timestamp_id):
            return timestamp_id
        return training_model_timestamp_id(training_run_started_at_ns(output))

    def _checkpoint_architecture_id(self, checkpoint: Path) -> str | None:
        checkpoint = checkpoint.resolve()
        stat = checkpoint.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = self._checkpoint_architecture_cache.get(checkpoint)
        if cached is not None and cached[0] == signature:
            return cached[1]
        try:
            payload = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            value = payload.get("architecture_id") if isinstance(payload, dict) else None
            architecture_id = str(value) if isinstance(value, str) else None
        except (OSError, RuntimeError, TypeError, ValueError):
            architecture_id = None
        self._checkpoint_architecture_cache[checkpoint] = (signature, architecture_id)
        return architecture_id

    def deployment_checkpoint_options(self, architecture_id):
        options=[]
        for entry in self.training_run_registry(architecture_id=architecture_id,top_level_only=True):
            manifest=entry['manifest'];output=Path(entry['output'])
            checkpoint=output/'checkpoint-latest.pt'
            if not checkpoint.is_file():continue
            reference='saved_target:'+checkpoint.relative_to(self.repository_root).as_posix()
            for role,binding in manifest['role_bindings'].items():
                if binding['mode'] in {'static','rule'}:continue
                update=manifest.get('checkpoint_update',0) if binding['mode']=='train' else binding['source_update']
                options.append(dict(id=reference,model_id=reference,kind='trained_target',actor_kind=role,
                    label=f"{role.upper()} · {manifest['run_group_id']} · U{update}",
                    source_architecture_id=architecture_id,source_update=update,source_agent_frames=0,
                    run_started_at_ns=entry['run_started_at_ns'],run_name=entry['run_name'],
                    run_group_id=manifest['run_group_id'],parameter_group=binding['parameter_group'],
                    model_identity=binding['model_id'],checkpoint_update=manifest.get('checkpoint_update',0),
                    source_checkpoint=str(checkpoint)))
        return options

    def resolve_deployment_checkpoint(
        self,
        architecture_id: str,
        source_id: str,
    ) -> Path:
        source = self._resolve_model_checkpoint(source_id)
        artifact_root = (self.repository_root / "Artifacts" / "GoalReach").resolve()
        try:
            relative = source.relative_to(artifact_root)
        except ValueError as error:
            raise ValueError(
                "Deployment checkpoint is outside GoalReach artifacts"
            ) from error
        top_level = len(relative.parts) == 2 and relative.name.startswith("checkpoint-")
        if not top_level:
            raise ValueError(
                "Deployment checkpoint is not a saved training Run checkpoint"
            )
        manifest_path=source.parent/'run.json'
        manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
        if manifest.get('architecture_id') != architecture_id:
            raise ValueError("Deployment checkpoint belongs to another Actor scenario")
        return source

    def load_deployment_checkpoint(
        self,
        architecture_id: str,
        source_id: str,
        *,
        role: str,
        map_location: str | torch.device = "cpu",
    ) -> tuple[Path, dict[str, object]]:
        source = self.resolve_deployment_checkpoint(architecture_id, source_id)
        return self._load_model_checkpoint(
            architecture_id,
            source.relative_to(self.repository_root).as_posix(),
            role=role,
            map_location=map_location,
        )

    def deployment_checkpoint_inventory(self, architecture_ids):
        return [option for architecture in dict.fromkeys(architecture_ids)
                for option in self.deployment_checkpoint_options(architecture)]

    @staticmethod
    def _metrics_signature(output: Path) -> tuple[object, ...] | None:
        metrics = training_metrics_source(output / "metrics.jsonl")
        if not metrics.is_file():
            return None
        stat = metrics.stat()
        return (str(metrics.resolve()), stat.st_mtime_ns, stat.st_size)

    def _load_model_checkpoint(
        self,
        architecture_id: str,
        source_id: str,
        *,
        role: str,
        map_location: str | torch.device = "cpu",
    ) -> tuple[Path, dict[str, object]]:
        source = self._resolve_model_checkpoint(source_id)
        from .goal_reach import _read_group_release, _group_actor_payload
        source, _, raw, _ = _read_group_release(source)
        if raw['architecture_id'] != architecture_id:
            raise ValueError('Model checkpoint belongs to another architecture')
        payload = _group_actor_payload(raw, role, source)
        if 'model' in payload:
            payload['model'] = {key:value.to(map_location) for key,value in payload['model'].items()}
        return source, payload

    def _reap_dead_worker(self, path, expected, *, exit_code=None):
        """Explicit exit collection, never called as a side effect of lock reads."""
        with self._launch_lock:
            current = read_actor_training_lock(path)
            if (_lock_identity(current) != _lock_identity(expected) or _lock_identity(current) is None
                    or current['owner_kind'] != 'worker' or current['process_identity_status'] != 'dead'):
                return False
            output = Path(current['output'])
            try:
                manifest = json.loads((output/'run.json').read_text(encoding='utf-8'))
            except (OSError, ValueError):
                return False
            attempt = manifest.get('launch_attempt') or {}
            if manifest.get('run_group_id') != current['run_group_id'] or attempt.get('id') != current['launch_attempt_id']:
                return False
            owner = {**current, 'pid': os.getpid(), 'process_start_token': process_start_token(os.getpid()), 'owner_kind': 'launcher'}
            again = read_actor_training_lock(path)
            if _lock_identity(again) != _lock_identity(current) or again['process_identity_status'] != 'dead':
                return False
            write_json_atomic(path, {k:v for k,v in owner.items() if k not in _LOCK_OBSERVATIONS})
            can_release = False
            try:
                if manifest.get('status') not in {'completed', 'stopped', 'failed'} or exit_code not in (None, 0):
                    diagnostic = self._attempt_diagnostic(output, {**manifest, 'exit_code': exit_code})
                    error = RuntimeError(diagnostic or f'Worker exited without completed cleanup (exit={exit_code})')
                    can_release = record_training_attempt_failure(output, current['run_group_id'], attempt, error,
                        'process_exit', lock_path=path, expected_owner=owner, exit_code=exit_code if exit_code else 1)
                else:
                    can_release = True
                if not can_release:
                    raise RuntimeError('Worker exit diagnostic not persisted; recovery owner retained')
                return True
            finally:
                if can_release:release_actor_training_lock(path, expected_owner=owner)

    def _poll_process(self, run_group_id: str) -> subprocess.Popen[bytes] | None:
        process = self._processes.get(run_group_id)
        if process is None or process.poll() is None:
            return process
        exit_code = int(process.returncode)
        session = self._started_sessions.get(run_group_id, {})
        owner = session.get('worker_owner')
        if owner is not None:
            path = Path(session['lock_path'])
            current = read_actor_training_lock(path)
            if current is not None and _lock_identity(current) != _lock_identity(owner):
                self._processes.pop(run_group_id, None)
                self._started_sessions.pop(run_group_id, None)
                return None
            output = Path(owner['output'])
            try:
                manifest = json.loads((output/'run.json').read_text(encoding='utf-8'))
            except (OSError, ValueError):
                manifest = {}
            if (manifest.get('launch_attempt') or {}).get('id') != owner['launch_attempt_id']:
                self._processes.pop(run_group_id, None)
                self._started_sessions.pop(run_group_id, None)
                return None
            if current is not None:
                self._reap_dead_worker(path, owner, exit_code=exit_code)
            self._last_group_exit_codes[run_group_id] = exit_code
            for architecture in session.get('architecture_ids', {}).values():
                self._last_exit_codes[architecture] = exit_code
            if exit_code:
                error = self._attempt_diagnostic(output, manifest)
                self._last_process_error = f'Training session {run_group_id}; exit code {exit_code}; {error}'
                self._group_process_errors[run_group_id] = self._last_process_error
        self._processes.pop(run_group_id, None)
        self._started_sessions.pop(run_group_id, None)
        return None

    def running_architecture_ids(self) -> list[str]:
        active = {
            str(lock.get("architecture_id", "unknown"))
            for lock in _active_actor_training_locks(
                (self.repository_root / ACTOR_TRAINING_LOCK.parent).resolve()
            )
        }
        for run_group_id in tuple(self._processes):
            if self._poll_process(run_group_id) is None:
                continue
            session = self._started_sessions.get(run_group_id, {})
            architecture_ids = session.get("architecture_ids", {})
            train_sides = session.get("train_sides", set())
            if isinstance(architecture_ids, Mapping) and isinstance(
                train_sides, set | list | tuple
            ):
                active.update(
                    str(architecture_ids[side])
                    for side in train_sides
                    if side in architecture_ids
                )
        return sorted(active)

    def _run_name_fields(
        self,
        run_id: str,
        *,
        architecture_id: str,
        output: str | None,
        mode: str | None,
        update: int,
        seed: int,
        source_update: int,
    ) -> dict[str, object]:
        cached = self._started_name_fields.get(run_id)
        if cached is None and output is not None:
            manifest_path = Path(output) / "run.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                manifest = None
            if isinstance(manifest, Mapping):
                stored_fields = manifest.get("name_fields")
                if isinstance(stored_fields, Mapping):
                    cached = dict(stored_fields)
                else:
                    resolved_settings = manifest.get("resolved_training_settings")
                    training_parameters = (
                        resolved_settings
                        if isinstance(resolved_settings, Mapping)
                        else {"seed": seed}
                    )
                    joint_training = training_parameters.get("joint_training")
                    manifest_mode = (
                        joint_training.get("mode")
                        if isinstance(joint_training, Mapping)
                        else mode
                    )
                    cached = model_name_fields(
                        architecture_id,
                        run_id,
                        update=update,
                        reward=None,
                        training_mode=manifest_mode,
                        source_update=manifest.get("source_update", source_update),
                        training_parameters=training_parameters,
                    )
                self._started_name_fields[run_id] = cached
        if cached is None:
            cached = model_name_fields(
                architecture_id,
                run_id,
                update=update,
                reward=None,
                training_mode=mode,
                source_update=source_update if source_update > 0 else None,
                training_parameters={"seed": seed},
            )
        fields = dict(cached)
        fields["update"] = int(update)
        return fields

    def poll_started_processes(self) -> None:
        for run_group_id in tuple(self._processes):
            self._poll_process(run_group_id)
        directory = self.repository_root / ACTOR_TRAINING_LOCK.parent
        for path in directory.glob(f'{ACTOR_TRAINING_LOCK_PREFIX}*.lock'):
            state = read_actor_training_lock(path)
            if state and state['process_identity_status'] == 'dead' and state.get('owner_kind') == 'worker':
                self._reap_dead_worker(path, state)

    def process_status(
        self, architecture_id: str | None
    ) -> tuple[str | None, int | None]:
        return (
            self._last_process_error,
            self._last_exit_codes.get(architecture_id),
        )

    def launch_process_status(self, run_group_id: str) -> tuple[bool, int | None]:
        """Return whether a Viewer-started process is live and its final exit code."""
        process = self._processes.get(run_group_id)
        if process is not None:
            process = self._poll_process(run_group_id)
        return process is not None, self._last_group_exit_codes.get(run_group_id)

    def active_training_group_ids(self) -> set[str]:
        active = _active_actor_training_locks((self.repository_root / ACTOR_TRAINING_LOCK.parent).resolve())
        return {_actor_training_run_group_id(lock) for lock in active}

    def group_process_error(self, run_group_id: str) -> str | None:
        return self._group_process_errors.get(run_group_id)

    def clear_group_process_error(self, run_group_id: str) -> None:
        old_error = self._group_process_errors.pop(run_group_id, None)
        if self._last_process_error == old_error:
            self._last_process_error = None
        self._last_group_exit_codes.pop(run_group_id, None)

    def refresh(self, architecture_id=None, *, prune_stale_locks=True, poll_processes=True, timings=None):
        if poll_processes:self.poll_started_processes()
        reservations = _active_actor_training_locks(self.repository_root / ACTOR_TRAINING_LOCK.parent,
                                                    prune_stale=prune_stale_locks)
        current_hot = {Path(lock['output']).resolve() for lock in reservations if lock.get('output')}
        hot_outputs = current_hot | self._registry_hot_outputs
        full = self._training_run_registry_cache is None or time.monotonic()-self._registry_last_discovery >= 60.
        started = (time.perf_counter(), time.thread_time(), time.process_time())
        self.refresh_training_run_registry(None if full else hot_outputs)
        _record_elapsed_timing(timings, 'training_status_registry_full' if full else 'training_status_registry_incremental', started)
        self._registry_hot_outputs = current_hot
        all_runs=[];sessions=[];groups=[]
        locks_by_group = {_actor_training_run_group_id(lock): lock for lock in reservations}
        cache = getattr(self, '_group_tail_cache', {})
        self._group_tail_cache = cache
        visible = set()
        for entry in self.training_run_registry():
            manifest=entry['manifest']
            if manifest.get('checkpoint_schema')!='mobile_local_sensing_group':
                continue
            output=Path(entry['output']);group_id=manifest['run_group_id'];visible.add(output)
            lock=locks_by_group.get(group_id, {})
            identity_status = lock.get('process_identity_status', 'dead')
            same_group = (lock.get('run_group_id') == group_id
                and Path(str(lock.get('output') or output)).resolve() == output.resolve())
            same_attempt = lock.get('launch_attempt_id') == (manifest.get('launch_attempt') or {}).get('id')
            if lock and (not same_group or not same_attempt):
                identity_status = 'unknown'
            reserved = bool(lock) and identity_status != 'dead'
            verified = reserved and identity_status == 'alive'
            preparing = verified and (lock.get('owner_kind') == 'launcher' or lock.get('runtime_phase') == 'preparing')
            running = verified and not preparing
            stop = (output/TRAINING_STOP_REQUEST_NAME).is_file()
            status = (manifest['status'] if identity_status == 'unknown' else
                'stopping' if verified and stop else 'preparing' if preparing else
                manifest['status'] if running or manifest['status'] not in {'running','preparing'} else 'failed')
            stderr_path = output/'training.stderr.log'
            failure = None
            if identity_status == 'unknown':
                failure = lock.get('identity_diagnostic') or 'Process/attempt identity unknown; capacity reserved'
            elif status == 'failed' and not reserved:
                failure = self._attempt_diagnostic(output,manifest)
            metrics=output/'metrics.jsonl'
            stored=cache.get(output)
            if full or output in hot_outputs or stored is None:
                stat=metrics.stat() if metrics.is_file() else None
                signature=(stat.st_ino,stat.st_mtime_ns,stat.st_size) if stat else None
                if stored is None or stored[0]!=signature:
                    latest=_last_jsonl_object(metrics) if stat else None
                    stored=(signature,latest);cache[output]=stored
            record=stored[1] or {}
            effective={r:(record.get('roles',{}).get(r,{}) or {}).get('train/effective_minibatch_agent_frames')
                if b['mode']=='train' else None for r,b in manifest['role_bindings'].items()}
            common={'run_group_id':group_id,'output':str(output),'running':running,'starting':preparing,
                'process_identity_status':identity_status,'capacity_reserved':reserved,
                'status':status,'update':manifest.get('global_update',0),'checkpoint_update':manifest.get('checkpoint_update',0),
                'target_update':manifest['update_budget'],'seed':manifest['seed'],'started_at':manifest['started_at'],
                'mode':manifest['training_mode'],'training_settings':manifest['launch_settings'], 'architecture_id':manifest['architecture_id'],
                'stop_requested':stop,'preparation':{},'runtime_global_update':lock.get('runtime_global_update',manifest.get('global_update',0)),
                'runtime_phase':'stopping' if running and stop else lock.get('runtime_phase',status) if running else status,
                'runtime_phase_updated_at_ns':lock.get('runtime_phase_updated_at_ns'),
                'runtime_last_update_reset':lock.get('runtime_last_update_reset'),
                'error':failure,
                'launch_attempt':manifest.get('launch_attempt'),
                'error_stage':manifest.get('error_stage'), 'exit_code':self._last_group_exit_codes.get(group_id, manifest.get('exit_code')),
                'stderr_path':str(output/'training.stderr.log')}
            runs=[]
            for role,binding in manifest['role_bindings'].items():
                run={**common,'actor_kind':role,'run_id':f'{group_id}:{role}','model_id':binding.get('model_id'),
                    'update':manifest.get('parameter_groups',{}).get(binding['parameter_group'],{}).get('update_index',manifest.get('global_update',0)) if binding['mode']=='train' else binding.get('source_update'),
                    'source_update':binding.get('source_update'),'source':binding.get('source'),'role_binding':binding,
                    'effective_minibatch_agent_frames':effective[role], 'metrics':record.get('roles',{}).get(role,{}) if binding['mode'] in {'train','rule'} else {}}
                runs.append(run)
                if reserved:all_runs.append(run)
            group={**common,'runs':runs,'effective_minibatch_agent_frames':effective,
                'update_timing':_update_timing(record, group_id, (manifest.get('launch_attempt') or {}).get('id')),
                'parameter_group_metrics':record.get('parameter_groups',{}), 'latest_record':record,
                'train_sides':manifest['train_sides'], 'architecture_ids':{r:manifest['architecture_id'] for r in manifest['train_sides']},
                'model_ids':{r:b['model_id'] for r,b in manifest['role_bindings'].items()},
                'outputs':{r:str(output) for r in manifest['role_bindings']},'sources':{}}
            groups.append(group)
            if reserved:sessions.append(group)
        for output in set(cache)-visible:del cache[output]
        runs=[r for r in all_runs if architecture_id is None or r['architecture_id']==architecture_id]
        return {'architecture_id':architecture_id,'stop_requested':False,'running':any(r['running'] for r in runs),'all_runs':all_runs,'runs':runs,
            'groups':groups,'grouped_sessions':sessions,'running_count':len({_actor_training_run_group_id(lock) for lock in reservations}),
            'capacity_reserved':len({_actor_training_run_group_id(lock) for lock in reservations}),
            'running_architecture_ids':sorted({r['architecture_id'] for r in all_runs if r['running']}),
            'max_parallel_runs':MAX_PARALLEL_TRAINING_RUNS,'process_error':self._last_process_error,
            'last_exit_code':None,'viewer_run_running':any(r['running'] for r in runs)}

    def _allocate_joint_output(
        self,
        architecture_id: str,
        *,
        source_update: int,
        joint_mode: str,
        alternating_updates: int,
        seed: int,
    ) -> Path:
        artifact_root = (self.repository_root / "Artifacts" / "GoalReach").resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)
        base_name = self._variant_output(architecture_id).name.removesuffix("-seed42")
        source_prefix = f"from-u{source_update}-" if source_update else ""
        mode_slug = (
            "joint"
            if joint_mode == "joint"
            else f"alternating-k{int(alternating_updates)}"
        )
        while True:
            timestamp_id = training_model_timestamp_id(time.time_ns())
            output = artifact_root / (
                f"{timestamp_id}-{source_prefix}{base_name}-{mode_slug}-seed{seed}"
            )
            try:
                output.mkdir(exist_ok=False)
            except FileExistsError:
                continue
            return output.resolve()

    def _start_joint(self, *, architecture_ids, sources, model_ids, train_sides, new_series_sides,
                     joint_mode, alternating_updates, updates, training_settings, source_metadata=None, seed=42, prepared=None, launch_attempt=None):
        from .goal_reach import (_prepare_joint_training_launch, _validated_viewer_settings, _preflight_training_output,
                                 _initial_training_role_bindings, _training_rule)
        from .actor_variants import policy_roles, parameter_group_bindings, ROLE_ORDER
        settings=_validated_viewer_settings({**training_settings,'joint_training_mode':joint_mode,'alternating_updates':alternating_updates,
            'training_settings':{**training_settings['training_settings'],'updates':int(updates)},
            'training_side_modes':{role:'train' if role in train_sides else 'fixed' for role in ('learner','defender','target')}})
        if prepared is None:
            prepared=_prepare_joint_training_launch(str(self.repository_root),settings,
                {role:str(path) if path else None for role,path in sources.items()},model_ids,list(train_sides),seed,
                resume=not new_series_sides and all(sources.values()))
        from .goal_reach import _read_group_release, _validate_training_sample_contract
        for role, source in prepared.sources.items():
            if source and role in train_sides:
                _, _, payload, _ = _read_group_release(Path(source))
                _validate_training_sample_contract(payload)
        if settings != prepared.launch_settings:
            raise ValueError('Prepared launch settings changed')
        sources={role:Path(path) if path else None for role,path in prepared.sources.items()}
        for source in sources.values():
            if source is not None:
                require_active_training_checkpoint(source)
        if architecture_ids!=prepared.architecture_ids:raise ValueError('Architecture does not match normalized settings')
        active=_active_actor_training_locks((self.repository_root/ACTOR_TRAINING_LOCK.parent).resolve())
        active_ids={_actor_training_run_group_id(lock) for lock in active}
        if len(active_ids)>=MAX_PARALLEL_TRAINING_RUNS:raise TrainingCapacityError('4/4 training slots are reserved')
        continuing=not new_series_sides and not prepared.new_series_sides
        primary=next(role for role in policy_roles(settings) if role in train_sides)
        prior=prepared.source_metadata.get(primary)
        if continuing:
            group_id=prior['run_group_id'];output=Path(sources[primary]).resolve().parent
            if group_id in active_ids:raise ValueError('This training group is already running')
        else:
            group_id=training_model_timestamp_id(time.time_ns())
            output=self.repository_root/'Artifacts'/'GoalReach'/f'{group_id}-unified-match-seed{seed}'
            output.mkdir(parents=True,exist_ok=False)
        if not continuing:_preflight_training_output(output, group_id, sources=sources)
        command=[sys.executable,'-m','auv_madrl','goal-reach-train','--output',str(output),'--seed',str(seed),
            '--run-group-id',group_id,'--training-settings-json',json.dumps(settings,ensure_ascii=False)]
        if continuing:command.extend(('--resume',str(Path(sources[primary]).resolve())))
        else:
            refs={role:{**prepared.source_metadata[role], 'path':str(Path(source).resolve())}
                  for role,source in sources.items() if source is not None}
            if refs:command.extend(('--sources-json',json.dumps(refs,ensure_ascii=False)))
        env=os.environ.copy();env['PYTHONPATH']=str(self.repository_root/'Python'/'src')
        env['PYTHONIOENCODING']='utf-8'
        outputs={role:str(output) for role in policy_roles(settings)}
        models={role:None if _training_rule(settings, role) is not None else f'{group_id}:{role}' for role in policy_roles(settings)}
        launch_attempt = dict(launch_attempt or {'id':training_model_timestamp_id(time.time_ns()),'started_at':current_timestamp()})
        lock_path = actor_training_lock_path(self.repository_root, architecture_ids[primary], group_id)
        bootstrap = claim_actor_training_lock(lock_path, architecture_ids[primary], output,
            mode=joint_mode, source=sources[primary] if continuing else None, seed=seed,
            model_id=group_id, target_update=updates, run_group_id=group_id,
            launch_attempt_id=launch_attempt['id'], owner_kind='launcher')
        process = None
        worker_owner = None
        attempt_published = False
        try:
            if not continuing:_preflight_training_output(output, group_id, sources=sources)
            manifest_path=output/'run.json'
            previous=json.loads(manifest_path.read_text(encoding='utf-8')) if continuing else {
                'schema_version':3,'kind':'training','checkpoint_schema':'mobile_local_sensing_group',
                'run_group_id':group_id,'architecture_id':architecture_ids[primary],
                'launch_settings':settings,'update_budget':updates,'global_update':0,
                'seed':seed,'started_at':launch_attempt['started_at'],'training_mode':joint_mode,
                'train_sides':list(train_sides),
                'role_bindings':_initial_training_role_bindings(settings, models, prepared.source_metadata)}
            require_actor_training_owner(lock_path, bootstrap)
            # Publish the exact attempt before opening logs; offsets describe only it.
            stderr_path = output/'training.stderr.log'
            stderr_offset = stderr_path.stat().st_size if stderr_path.exists() else 0
            launch_attempt.update(stderr_offset=stderr_offset, bootstrap_owner=bootstrap, output_file_id=output.stat().st_ino)
            if stderr_path.exists():launch_attempt['stderr_file_id']=stderr_path.stat().st_ino
            previous.update(launch_attempt=launch_attempt,status='preparing',pid=bootstrap['pid'],
                process_start_token=bootstrap['process_start_token'],owner_kind='launcher')
            for field in ('error','error_stage','exit_code','finished_at'):previous.pop(field,None)
            (write_json_atomic if continuing else create_json_atomic)(manifest_path,previous)
            attempt_published = True
            if continuing:output.joinpath(TRAINING_STOP_REQUEST_NAME).unlink(missing_ok=True)
            with (output/'training.stdout.log').open('a',encoding='utf-8') as stdout, stderr_path.open('a',encoding='utf-8') as stderr:
                launch_attempt['stderr_file_id'] = stderr_path.stat().st_ino
                require_actor_training_owner(lock_path, bootstrap)
                previous['launch_attempt'] = launch_attempt
                write_json_atomic(manifest_path, previous)
                command.extend(('--launch-attempt-json',json.dumps(launch_attempt)))
                process=spawn_training_process(command,cwd=self.repository_root,env=env,
                    stdout=stdout,stderr=stderr)
                if process.poll() is not None:
                    raise RuntimeError(f'Training child exited before handoff: {process.returncode}')
                worker_owner = handoff_actor_training_lock(lock_path, bootstrap, process.pid)
        except BaseException as error:
            try:
                current = read_actor_training_lock(lock_path)
                if _lock_identity(current) == _lock_identity(bootstrap):
                    try:
                        recorded = record_training_attempt_failure(output, group_id, launch_attempt, error, 'process_start',
                            lock_path=lock_path, expected_owner=bootstrap)
                        if attempt_published and not recorded:
                            raise RuntimeError('Failure diagnostic was not persisted')
                    finally:
                        # Exact release checks again; never delete an accepted worker/unknown owner.
                        release_actor_training_lock(lock_path, expected_owner=bootstrap)
                    # Unaccepted children see missing/changed bootstrap and exit before build.
                elif (current and current.get('launch_attempt_id') == launch_attempt['id']
                        and current.get('owner_kind') == 'worker' and current.get('output') == str(output.resolve())):
                    output.joinpath(TRAINING_STOP_REQUEST_NAME).touch()
            except BaseException as cleanup_error:
                try:print(f'Training launch cleanup: {type(cleanup_error).__name__}: {cleanup_error}', file=sys.stderr)
                except BaseException:pass
            raise
        session={'run_group_id':group_id,'pid':process.pid,'worker_owner':worker_owner,'lock_path':str(lock_path),'stderr_offset':stderr_offset,'launch_attempt':launch_attempt,'architecture_ids':architecture_ids,'outputs':outputs,
            'sources':{r:str(v) if v else None for r,v in sources.items()},'model_ids':models,
            'source_updates':{r:(prepared.source_metadata.get(r) or {}).get('source_update',0) for r in policy_roles(settings)},
            'target_updates':{r:updates for r in train_sides},'train_sides':list(train_sides),
            'new_series_sides':list(prepared.new_series_sides),'joint_mode':joint_mode,'alternating_updates':alternating_updates,
            'seed':seed,'started_at':datetime.now().astimezone().isoformat(timespec='seconds'),'training_settings':settings}
        self._processes[group_id]=process;self._started_sessions[group_id]=session;self._last_started_run_group_id=group_id
        self.clear_group_process_error(group_id)
        return session

    def start_joint(
        self,
        *,
        architecture_ids: dict[str, str],
        sources: dict[str, Path | None],
        model_ids: dict[str, str],
        train_sides: set[str],
        new_series_sides: set[str],
        joint_mode: str,
        alternating_updates: int,
        updates: int,
        training_settings: dict[str, object],
        source_metadata: dict[str, Mapping[str, object] | None] | None = None,
        seed: int = 42,
        prepared=None,
        launch_attempt=None,
    ) -> dict[str, object]:
        with self._launch_lock:
            return self._start_joint(
                architecture_ids=architecture_ids,
                sources=sources,
                model_ids=model_ids,
                train_sides=train_sides,
                new_series_sides=new_series_sides,
                joint_mode=joint_mode,
                alternating_updates=alternating_updates,
                updates=updates,
                training_settings=training_settings,
                source_metadata=source_metadata,
                seed=seed,
                prepared=prepared,
                launch_attempt=launch_attempt,
            )

    def stop_joint(self, run_group_id: str, *, status: dict[str, object] | None = None) -> dict[str, object]:
        run_group_id = str(run_group_id).strip()
        if not run_group_id:
            raise ValueError("Training session run_group_id is required")
        with self._launch_lock:
            locks = [state for state in _active_actor_training_locks(self.repository_root / ACTOR_TRAINING_LOCK.parent)
                     if state.get('run_group_id') == run_group_id]
            if len(locks) != 1 or locks[0]['process_identity_status'] != 'alive':
                raise ValueError('Training owner is absent or unknown; Stop identity cannot be verified')
            owner = locks[0]
            if status is not None:
                known = next((g for g in status.get('groups', []) if g.get('run_group_id') == run_group_id), None)
                if known and (known.get('launch_attempt') or {}).get('id') != owner['launch_attempt_id']:
                    raise ValueError('Stop request belongs to a previous training attempt')
            path = Path(owner['_lock_path'])
            current = read_actor_training_lock(path)
            if _lock_identity(current) != _lock_identity(owner) or current['process_identity_status'] != 'alive':
                raise ValueError('Training owner changed before Stop')
            Path(owner['output']).joinpath(TRAINING_STOP_REQUEST_NAME).touch()
        return status if status is not None else self.refresh()
