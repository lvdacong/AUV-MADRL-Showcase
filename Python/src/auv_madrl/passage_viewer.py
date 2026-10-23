"""Passage mode owned by the existing Viewer, with an isolated training lifecycle."""

from __future__ import annotations

import copy
import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .actor_training import (
    claim_training_lock,
    handoff_training_lock,
    process_start_token,
    read_actor_training_lock,
    release_actor_training_lock,
    spawn_training_process,
)
from .checkpoint import save_checkpoint
from .passage import CONTRACT, PassageConfig, PassageEnv, PassageTrainer
from .run_evidence import write_json_atomic


class PassageViewer:
    def __init__(self, root: Path, *, checkpoint_executor=None):
        self.root = root / "passage"
        self.trainer = PassageTrainer()
        self.config = self.trainer.config
        self.env = PassageEnv(maximum_steps=self.config.maximum_episode_steps)
        self.update_budget = 0
        self.deployed = None
        self.policy_sampling = "stochastic"
        self.speed = 1.0
        self.followed = None
        self._follow_mtime = None
        self._checkpoint_executor = checkpoint_executor
        self._follow_future = None
        self._follow_checked_at = float('-inf')
        self.deployment_error = None
        self.catalog = {}
        if (self.root / "catalog.json").is_file():
            self.catalog = json.loads((self.root / "catalog.json").read_text(encoding="utf-8"))
            for name, row in self.catalog.items():
                self._path(name)
                if row["status"] == "running":
                    row.update(
                        status="stopped", error="Viewer closed; resume the last complete save"
                    )
        self.generator = torch.Generator().manual_seed(10000)
        self.running = False
        self.next_step = 0.0
        self.sampling = self.policy_sampling
        self.episode = 0
        self.history = []
        self.outcome = "running"
        self.worker = None
        self.jobs = {}
        self.lock = threading.Lock()
        self.trail = [[5, 9]]
        self._preview_key = self._preview_cache = None
        self._catalog_revision = 0
        self._metadata_key = self._metadata_cache = None
        self._files_checked_at = float("-inf")
        self._models = []
        self._scene_saved = False
        self._steps_total = 0
        self._rate_samples = deque()
        self._snapshot_key = None
        self._snapshot_revision = 0

    def close(self):
        self._request_stop()
        if self._follow_future is not None:
            self._follow_future[2].cancel()
            self._follow_future = None
        for job in list(self.jobs.values()):
            job['worker'].join()

    def _request_stop(self, name=None):
        with self.lock:
            if name is not None and name not in self.jobs:
                raise ValueError('Unknown passage training job')
            for key, job in self.jobs.items():
                if (name is None or key == name) and job['worker'].is_alive():
                    job['stop_path'].touch()

    @property
    def training(self):
        states = [job['state'] for job in self.jobs.values()]
        active = [state for state in states if state['status'] == 'running']
        state = (active or states or [dict(status='idle', error=None,
                                          metrics=[], checkpoint=None)])[-1]
        return dict(state, active_count=len(active),
                    active_names=[state['checkpoint'] for state in active])

    def _path(self, name):
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".pt"):
            raise ValueError("Select a passage model from the library")
        path = self.root / name
        if path.is_symlink() or path.resolve().parent != self.root.resolve():
            raise ValueError("Invalid model path")
        return path

    def _busy(self, name=None):
        return any(job['worker'].is_alive() for key, job in self.jobs.items()
                   if name is None or key == name)

    def move_training_group(self, entry, name, *, restore=False):
        """Adapt the flat model files to the Viewer's existing Undo/Redo archive."""
        checkpoint = self._path(name)
        owner = dict(pid=os.getpid(), process_start_token=process_start_token(os.getpid()),
            owner_kind='launcher', launch_attempt_id=uuid.uuid4().hex,
            run_group_id='passage:'+name, output=str(self.root))
        lock_path = checkpoint.with_suffix('.lock')
        allowed = {name, checkpoint.stem+'.metrics.jsonl', checkpoint.stem+'.training.stderr.log',
            checkpoint.stem+'.evaluation.jsonl'}
        with self.lock:
            if (name in self.catalog) == restore:
                raise ValueError('Training group is missing or its restore path is occupied')
            claim_training_lock(lock_path, owner)
            try:
                manifest_path = entry / 'manifest.json'
                if restore:
                    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                    if (manifest['name'] != name or name not in manifest['files']
                            or len(set(manifest['files'])) != len(manifest['files'])
                            or not set(manifest['files']) <= allowed):
                        raise ValueError('Invalid archived training group')
                    source, target = entry / 'segment', self.root
                else:
                    source, target = self.root, entry / 'segment'
                    manifest = dict(scene_mode='discrete', name=name, row=self.catalog[name],
                        files=sorted(n for n in allowed if (source / n).is_file()))
                    if name not in manifest['files']:
                        raise ValueError('Training group has no complete checkpoint')
                if source.is_symlink() or target.is_symlink():
                    raise ValueError('Training group path is unsafe or occupied')
                paths = [(source / n, target / n) for n in manifest['files']]
                for src, dst in paths:
                    if src.is_symlink() or src.resolve().parent != source.resolve() or dst.exists():
                        raise ValueError('Training group path is unsafe or occupied')
                target.mkdir(parents=True, exist_ok=True)
                moved = []
                try:
                    if not restore:
                        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
                    for src, dst in paths:
                        src.replace(dst)
                        moved.append((src, dst))
                    if restore:
                        self.catalog[name] = manifest['row']
                    else:
                        del self.catalog[name]
                    self._store_catalog()
                except Exception:
                    if restore:
                        self.catalog.pop(name, None)
                    else:
                        self.catalog[name] = manifest['row']
                    for src, dst in reversed(moved):
                        dst.replace(src)
                    if not restore:
                        manifest_path.unlink(missing_ok=True)
                    raise
                if not restore and self.followed == name:
                    self.followed = None
                if not restore and self.deployed == name:
                    self.command(dict(action='deploy_random'))
                if not restore:
                    self.jobs.pop(name, None)
                self._catalog_revision += 1
                self._files_checked_at = float('-inf')
            finally:
                release_actor_training_lock(lock_path, expected_owner=owner)

    def _reset(self):
        self.env.reset()
        self.episode += 1
        self.outcome = "running"
        self.trail = [[5, 9]]

    def _collision_rule(self, enabled):
        self.env.collision_termination = enabled
        self.env.maximum_steps = self.config.maximum_episode_steps
        self.env.left_probability = self.config.left_probability
        self.config = replace(self.config, collision_termination=enabled)
        self.trainer.config = replace(self.trainer.config, collision_termination=enabled,
            maximum_episode_steps=self.config.maximum_episode_steps,
            left_probability=self.config.left_probability)
        self.trainer.env.collision_termination = enabled
        self.trainer.env.maximum_steps = self.config.maximum_episode_steps
        self.trainer.env.left_probability = self.config.left_probability

    def _step(self, action=None):
        if self.env.done[0]:
            return
        if action is None:
            if self.sampling == "manual":
                return
            with torch.no_grad():
                distribution, _ = self.trainer.model(self.env.observe())
                action = (
                    int(distribution.probs.argmax(-1)[0])
                    if self.sampling == "deterministic"
                    else int(
                        torch.multinomial(distribution.probs, 1, generator=self.generator)[0, 0]
                    )
                )
        _, reward, terminated, truncated, collision = self.env.step([action])
        self._steps_total += 1
        self.trail.append(self.env.position[0].tolist())
        if self.env.done[0]:
            self.outcome = (
                "success"
                if reward[0]
                else "static_collision"
                if collision[0] and terminated[0]
                else "truncated"
            )
            self.history = [
                dict(
                    episode=self.episode,
                    outcome=self.outcome,
                    steps=int(self.env.steps[0]),
                    maximum_episode_steps=self.env.maximum_steps,
                    left=bool(self.env.left[0]),
                    terminated=bool(terminated[0]),
                    truncated=bool(truncated[0]),
                ),
                *self.history,
            ][:5]

    def _install_followed(self, name, modified, load):
        # Retry only after another publication; never discard the last good model.
        self._follow_mtime = modified
        try:
            loaded = load()
        except Exception as error:
            self.deployment_error = f"Auto deploy {name}: {type(error).__name__}: {error}"
            logging.getLogger(__name__).warning(self.deployment_error)
            return
        self.trainer, self.deployed, self.config = loaded, name, loaded.config
        self._collision_rule(loaded.config.collision_termination)
        self.deployment_error = None

    def tick(self, terminal_hold_s=0.1):
        if self._follow_future is not None and self._follow_future[2].done():
            name, modified, future = self._follow_future
            self._follow_future = None
            if name == self.followed:
                self._install_followed(name, modified, future.result)
        now = time.monotonic()
        if self.followed and self._follow_future is None and (
                self._checkpoint_executor is None or now - self._follow_checked_at >= 1.0):
            self._follow_checked_at = now
            path = self._path(self.followed)
            modified = path.stat().st_mtime_ns if path.exists() else None
            if modified is not None and modified != self._follow_mtime:
                if self._checkpoint_executor is not None:
                    self._follow_future = (self.followed, modified,
                        self._checkpoint_executor.submit(PassageTrainer.load, path))
                else:
                    self._install_followed(
                        self.followed, modified, lambda: PassageTrainer.load(path))
        if self.running and time.monotonic() >= self.next_step:
            if self.env.done[0]:
                self._reset()
            else:
                self._step()
            self.next_step = time.monotonic() + (
                terminal_hold_s if self.env.done[0] else 0.2 / self.speed
            )

    def _store_catalog(self):
        write_json_atomic(self.root / 'catalog.json', self.catalog, compact=True)

    def _record(
        self, path, trainer, status, metric=None, error=None, update_budget=None, attempt=None
    ):
        with self.lock:
            row = self.catalog.setdefault(
                path.name,
                dict(name=path.name, started_at=time.strftime("%Y-%m-%dT%H:%M:%S"), records=[]),
            )
            row.update(
                config=asdict(trainer.config),
                updates=trainer.updates,
                frames=trainer.frames,
                status=status,
                error=error,
            )
            if metric is not None:
                row["records"] = [*row["records"], metric][-200:]
                row["update_timing"] = dict(run_group_id="passage:" + path.name,
                    launch_attempt_id=metric["launch_attempt_id"], update=metric["update"],
                    seconds=metric["update_seconds"])
            if attempt is not None:
                if (row.get('launch_attempt') or {}).get('id') != attempt:
                    row.pop('runtime', None)
                row["launch_attempt"] = dict(id=attempt)
            if update_budget is not None:
                row["update_budget"] = update_budget
            self._store_catalog()
            self._catalog_revision += 1
            self._files_checked_at = float("-inf")

    def control(self, command):
        """Route the existing Viewer controls without modifying continuous settings."""
        kind = command["type"]
        if kind == "run":
            if self.sampling == "manual" and command.get("value", True):
                raise ValueError("Use direction keys in Manual mode, or select NN")
            self.running = bool(command.get("value", True))
        elif kind in {"reset_one", "reset_all"}:
            self.command(dict(action="reset"))
        elif kind == "select_actor" and command.get("actor") == "learner":
            return
        elif kind == "set_mode":
            if command.get("actor", "learner") != "learner" or command["mode"] not in {
                "manual",
                "nn",
            }:
                raise ValueError("Passage has one agent with Manual / NN control")
            self.sampling = "manual" if command["mode"] == "manual" else self.policy_sampling
            if self.sampling == "manual":
                self.running = False
        elif kind == "set_policy_sampling":
            if command["mode"] not in {"stochastic", "deterministic"}:
                raise ValueError("Unsupported policy sampling")
            self.policy_sampling = command["mode"]
            if self.sampling != "manual":
                self.sampling = self.policy_sampling
        elif kind == "set_rate" and command["name"] == "time_scale":
            speed = float(command["value"])
            if speed <= 0 or not torch.isfinite(torch.tensor(speed)):
                raise ValueError("Invalid speed")
            self.speed = speed
        elif kind in {"save_evaluation_scene", "load_evaluation_scene"}:
            self.command(dict(action="save_scene" if kind.startswith("save") else "load_scene"))
        else:
            raise ValueError("This control is not applicable to the discrete scene")

    def command(self, command):
        action = command.get("action")
        if action == "reset":
            self._reset()
        elif action == "step":
            self.running = False
            value = command.get("value")
            if value is not None and (type(value) is not int or value not in range(4)):
                raise ValueError("Invalid direction")
            self._step(value)
        elif action == "run":
            if self.sampling == "manual":
                raise ValueError("Select a neural policy to play; use arrows for manual movement")
            self.running = not self.running
        elif action == "sampling":
            value = command.get("value")
            if value not in ("manual", "stochastic", "deterministic"):
                raise ValueError("Invalid policy mode")
            self.sampling, self.running = value, False
            if value != "manual":
                self.policy_sampling = value
        elif action == "configure":
            if self._busy():
                raise ValueError("Stop passage training before changing its configuration")
            # Explicitly starts a fresh model; never silently changes a resumed contract.
            self.trainer = PassageTrainer(PassageConfig(**command["config"]))
            self.config = self.trainer.config
            self._collision_rule(self.config.collision_termination)
            self.deployed = None
            self.running = False
        elif action == "settings":
            config = PassageConfig(**command["config"])
            updates = command["updates"]
            if type(updates) is not int or not 0 <= updates <= 100000:
                raise ValueError("Updates must be an integer from 0 (Unlimited) to 100000")
            self.config, self.update_budget = config, updates
            self._collision_rule(config.collision_termination)
        elif action == "deploy_random":
            self.trainer = PassageTrainer(self.config)
            self._collision_rule(self.config.collision_termination)
            self.deployed = None
            self.followed = None
            self.deployment_error = None
        elif action == "follow":
            name = command.get("name")
            if name is not None:
                self._path(name)
                if name not in self.catalog:
                    raise ValueError("Unknown passage training group")
            self.followed, self._follow_mtime = name, None
            self._follow_checked_at = float('-inf')
            self.deployment_error = None
        elif action == "save_model":
            path = self._path("model-" + uuid.uuid4().hex[:12] + ".pt")
            self.trainer.save(path)
            self._record(path, self.trainer, "stopped")
        elif action == "load_model":
            loaded = PassageTrainer.load(self._path(command.get("name")))
            self.trainer, self.running = loaded, False
            self.config = loaded.config
            self._collision_rule(loaded.config.collision_termination)
            self.deployed = command["name"]
            self.followed = None
            self.deployment_error = None
        elif action in ("save_scene", "load_scene"):
            path = self.root / "scene.pt"
            if action == "save_scene":
                save_checkpoint(
                    path,
                    dict(
                        contract=CONTRACT,
                        env=self.env.state(),
                        episode=self.episode,
                        history=self.history,
                        outcome=self.outcome,
                        trail=self.trail,
                        sampling=self.sampling,
                        policy_sampling=self.policy_sampling,
                        rng=self.generator.get_state(),
                    ),
                )
            else:
                state = torch.load(path, map_location="cpu", weights_only=False)
                if state.get("contract") != CONTRACT:
                    raise ValueError("Scene contract mismatch")
                restored = PassageEnv()
                restored.restore(state["env"])
                self.env, self.trail = restored, state["trail"]
                self.config = replace(self.config, maximum_episode_steps=restored.maximum_steps,
                    left_probability=restored.left_probability)
                self.episode, self.history = state.get("episode", 0), state.get("history", [])
                self.outcome = state.get(
                    "outcome",
                    "success"
                    if restored.position[0].tolist() == [5, 1]
                    else "truncated"
                    if restored.done[0]
                    else "running",
                )
                self._collision_rule(restored.collision_termination)
                self.sampling = state["sampling"]
                self.policy_sampling = state.get(
                    "policy_sampling", self.sampling if self.sampling != "manual" else "stochastic"
                )
                self.generator.set_state(state["rng"])
                self.running = False
                self._preview_key = None
            self._files_checked_at = float("-inf")
        elif action == "train":
            updates = command.get("updates")
            if type(updates) is not int or not 0 <= updates <= 100000:
                raise ValueError("Updates must be an integer from 0 (Unlimited) to 100000")
            evaluation_updates = command.get('evaluation_updates', [])
            if (not isinstance(evaluation_updates, list) or len(evaluation_updates) > 100
                    or any(type(u) is not int or not 0 <= u <= 100000 for u in evaluation_updates)
                    or evaluation_updates != sorted(set(evaluation_updates))):
                raise ValueError('Evaluation updates must be distinct sorted update indices')
            name = command.get("name")
            if name:
                if self._busy(name):
                    raise ValueError('This passage model is already training')
                owner = read_actor_training_lock(self._path(name).with_suffix(".lock"))
                if owner and owner["capacity_reserved"]:
                    raise ValueError("Checkpoint has a live or unverified training owner")
            worker_trainer = (
                PassageTrainer.load(self._path(name))
                if name
                else PassageTrainer(self.config)
                if command.get("fresh")
                else copy.deepcopy(self.trainer)
            )
            path = self._path(name or "run-" + uuid.uuid4().hex[:12] + ".pt")
            attempt = uuid.uuid4().hex
            job = dict(stop_path=self.root / (path.stem + "." + attempt + ".stop"),
                       state=dict(status='running', error=None, metrics=[], checkpoint=path.name))
            worker = threading.Thread(
                target=self._train,
                args=(worker_trainer, updates, path, attempt, job, evaluation_updates),
                name='passage-training-' + path.stem,
                daemon=False,
            )
            job['worker'] = worker
            self._record(
                path, worker_trainer, "running",
                update_budget=worker_trainer.updates + updates if updates else 0,
                attempt=attempt,
            )
            with self.lock:
                self.jobs[path.name] = job
                self.worker = worker  # Last launched thread; close/stop own every job.
            worker.start()
        elif action == "stop":
            self._request_stop(command.get('name'))
        else:
            raise ValueError("Unsupported passage control")

    def _train(self, trainer, updates, path, attempt, job, evaluation_updates=()):
        process = None
        owner = None
        lock_path = path.with_suffix(".lock")
        try:
            owner = claim_training_lock(lock_path, dict(
                pid=os.getpid(), process_start_token=process_start_token(os.getpid()),
                owner_kind="launcher", launch_attempt_id=attempt,
                run_group_id="passage:" + path.name, output=str(path.parent.resolve()),
            ))
            # Transfer a full checkpoint; the Viewer and its preview remain CPU-only.
            trainer.save(path)
            command = [sys.executable, "-m", "auv_madrl.passage_worker",
                       "--checkpoint", str(path.resolve()), "--updates", str(updates),
                       "--attempt", attempt, "--stop-file", str(job['stop_path'].resolve()),
                       "--lock", str(lock_path.resolve()),
                       "--bootstrap", json.dumps(owner),
                       "--parent-pid", str(os.getpid()),
                       "--parent-token", process_start_token(os.getpid())]
            if evaluation_updates:
                command += ['--evaluation-updates', *map(str, evaluation_updates)]
            env = os.environ.copy()
            env.update(PYTHONPATH=str(Path(__file__).resolve().parents[1]),
                       PYTHONIOENCODING="utf-8")
            stderr_path = path.with_suffix(".training.stderr.log")
            with stderr_path.open("w", encoding="utf-8") as stderr:
                process = spawn_training_process(command, cwd=Path(__file__).resolve().parents[1],
                    env=env, stdout=subprocess.PIPE, stderr=stderr)
                handoff_training_lock(lock_path, owner, process.pid)
                owner = None  # The worker now owns and releases the lock.
                final_status = None
                for line in process.stdout:
                    event = json.loads(line)
                    if not event.get("passage_worker"):
                        continue
                    if "pid" in event:
                        with self.lock:
                            self.catalog[path.name]["runtime"] = event
                        self._record(path, trainer, "running")
                    metric = event.get("metric")
                    if metric is not None:
                        trainer.updates, trainer.frames = metric["update"], metric["frames"]
                        self._record(path, trainer, "running", metric)
                        with self.lock:
                            job['state']["metrics"] = [*job['state']["metrics"], metric][-200:]
                    if 'evaluation' in event:
                        evaluation = event['evaluation']
                        with self.lock:
                            self.catalog[path.name]['evaluation'] = {
                                **{key:evaluation[key] for key in
                                   ('update', 'left_probability', 'model_sha256', 'sampling')},
                                'results': {side:{key:value for key,value in result.items()
                                                  if key != 'episode_steps'}
                                            for side,result in evaluation['results'].items()},
                            }
                            self._store_catalog()
                            self._catalog_revision += 1
                    if "status" in event:
                        final_status = event["status"]
                        trainer.updates, trainer.frames = event["update"], event["frames"]
                code = process.wait()
            if code or final_status is None:
                raise RuntimeError(f"GPU training worker exited ({code}): "
                                   + stderr_path.read_text(encoding="utf-8")[-2000:])
            with self.lock:
                job['state']["status"] = final_status
            self._record(path, trainer, final_status)
        except Exception as error:
            self._request_stop(path.name)
            if process is not None:
                process.wait()
            with self.lock:
                job['state'].update(status="failed", error=f"{type(error).__name__}: {error}")
            self._record(path, trainer, "failed", error=job['state']["error"])
        finally:
            if owner is not None:
                release_actor_training_lock(lock_path, expected_output=path.parent,
                    expected_run_group_id="passage:" + path.name, expected_owner=owner)
            if process is not None and process.stdout is not None:
                process.stdout.close()
            with self.lock:
                job['stop_path'].unlink(missing_ok=True)

    def snapshot(self, *, active=True):
        with self.lock:
            metrics = self.training["metrics"]
            catalog_revision = self._catalog_revision
            metadata_key = (
                catalog_revision, self.training["status"], self.training["error"],
                self.training["checkpoint"], metrics[-1]["update"] if metrics else None,
            )
            if metadata_key != self._metadata_key:
                self._metadata_cache = dict(
                    training=copy.deepcopy(self.training),
                    catalog=copy.deepcopy(list(self.catalog.values())),
                )
                self._metadata_key = metadata_key
        # Only the selected preview performs inference. Cached lists are owned,
        # so later steps/training records cannot mutate an earlier snapshot.
        position = self.env.position[0].tolist()
        key = (id(self.trainer.model), self.trainer.updates, *position,
               bool(self.env.left[0]), int(self.env.steps[0]), bool(self.env.done[0]), self.episode)
        if active and key != self._preview_key:
            observation = self.env.observe()
            with torch.no_grad():
                distribution, value = self.trainer.model(observation)
            self._preview_cache = dict(
                position=position, left=bool(self.env.left[0]), steps=int(self.env.steps[0]),
                done=bool(self.env.done[0]), success=position == [5, 1],
                observation=observation[0].tolist(), probabilities=distribution.probs[0].tolist(),
                value=float(value[0, 0]), history=copy.deepcopy(self.history),
                trail=copy.deepcopy(self.trail),
            )
            self._preview_key = key
        now = time.monotonic()
        if now - self._files_checked_at >= 1.0:
            self._models = sorted(p.name for p in self.root.glob("*.pt") if p.name != "scene.pt")
            self._scene_saved = (self.root / "scene.pt").exists()
            self._files_checked_at = now
        if active and self.running:
            self._rate_samples.append((now, self._steps_total))
            while len(self._rate_samples) > 2 and self._rate_samples[1][0] < now - 2.0:
                self._rate_samples.popleft()
        else:
            self._rate_samples.clear()
        elapsed = now - self._rate_samples[0][0] if self._rate_samples else 0.0
        actual = ((self._steps_total - self._rate_samples[0][1]) * 0.2 / elapsed
                  if elapsed > 0 else 0.0)
        revision_key = (self._preview_key, self._metadata_key, self.running, self.episode,
            self.outcome, self.sampling, self.config, self.update_budget, self.deployed,
            self.followed, self.policy_sampling, self.speed, actual, self.trainer.updates,
            self.trainer.frames, tuple(self._models), self._scene_saved, self.deployment_error)
        if revision_key != self._snapshot_key:
            self._snapshot_key = revision_key
            self._snapshot_revision += 1
        return dict(
            **(self._preview_cache or {}), **self._metadata_cache,
            snapshot_revision=self._snapshot_revision,
            contract=CONTRACT,
            catalog_revision=catalog_revision,
            running=self.running,
            episode=self.episode,
            outcome=self.outcome,
            sampling=self.sampling,
            config=asdict(self.config),
            update_budget=self.update_budget,
            deployed=self.deployed,
            followed=self.followed,
            deployment_error=self.deployment_error,
            policy_sampling=self.policy_sampling,
            speed=self.speed,
            actual_time_scale=actual,
            updates=self.trainer.updates,
            frames=self.trainer.frames,
            models=self._models,
            scene_saved=self._scene_saved,
        )
