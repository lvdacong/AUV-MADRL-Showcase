"""CUDA passage adapter using the shared training launch, affinity and PPO core."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from .actor_training import (
    _query_process_identity,
    process_start_token,
    receive_actor_training_lock,
    release_actor_training_lock,
    require_actor_training_owner,
)
from .goal_reach import _cpu_thread_affinity
from .passage import PassageTrainer
from .passage_evaluation import model_sha256


def success_window(records, config):
    """Publish only a complete, globally aligned interval of equal training work."""
    count = config.metric_window_updates
    if not records or records[-1]['update'] % count:
        return None
    end = records[-1]['update']
    if [r['update'] for r in records] != list(range(end-count+1, end+1)):
        raise ValueError('Incomplete aligned passage metric window')
    keys = ('left_successes', 'right_successes', 'left_episodes',
            'right_episodes', 'left_frames', 'right_frames')
    if any(key not in record for record in records for key in keys):
        return None  # Legacy logs cannot supply a complete per-side window.
    diagnostics = {}
    if all('kl_early_stopped' in r and 'optimizer_steps' in r for r in records):
        diagnostics = dict(
            kl_early_stop_rate=sum(r['kl_early_stopped'] for r in records) / count,
            optimizer_steps=sum(r['optimizer_steps'] for r in records))
    return dict(start_update=end-count+1, end_update=end, updates=count,
                steps_per_environment=count*config.rollout_steps,
                **diagnostics,
                **{key:sum(r[key] for r in records) for key in keys})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--bootstrap", required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--parent-token", required=True)
    parser.add_argument('--evaluation-updates', type=int, nargs='*', default=[])
    args = parser.parse_args()
    owner = receive_actor_training_lock(args.lock, output=args.checkpoint.parent,
        run_group_id="passage:" + args.checkpoint.name,
        attempt=dict(id=args.attempt, bootstrap_owner=json.loads(args.bootstrap)))
    try:
        run(args, owner)
    finally:
        release_actor_training_lock(args.lock, expected_output=args.checkpoint.parent,
            expected_run_group_id="passage:" + args.checkpoint.name, expected_owner=owner)


def run(args, owner):
    if not torch.cuda.is_available():
        raise RuntimeError("Passage training requires CUDA; CPU fallback is disabled")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    def emit(**values):
        print(json.dumps(dict(passage_worker=True, **values), allow_nan=False), flush=True)

    with _cpu_thread_affinity(training=True):
        trainer = PassageTrainer.load(args.checkpoint, device="cuda")
        metrics_path = args.checkpoint.with_suffix(".metrics.jsonl")
        pending_count = trainer.updates % trainer.config.metric_window_updates
        pending = []
        if pending_count:
            pending = [json.loads(line) for line in metrics_path.read_text(encoding='utf-8')
                       .splitlines()[-pending_count:]]
            if [r['update'] for r in pending] != list(
                    range(trainer.updates-pending_count+1, trainer.updates+1)):
                raise ValueError('Cannot restore partial passage metric window')
        def evaluate():
            if trainer.updates not in args.evaluation_updates:
                return
            from .passage_evaluation import evaluate_policy
            result = evaluate_policy(trainer)
            result.update(launch_attempt_id=args.attempt)
            with args.checkpoint.with_suffix('.evaluation.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(result, allow_nan=False) + '\n')
            emit(evaluation=result)

        emit(pid=os.getpid(), process_start_token=process_start_token(os.getpid()),
             device=str(trainer.device), environment_device=str(trainer.env.device),
             update=trainer.updates, frames=trainer.frames,
             model_sha256=model_sha256(trainer.model))
        evaluate()
        completed = 0
        while args.updates == 0 or completed < args.updates:
            require_actor_training_owner(args.lock, owner)
            state, token = _query_process_identity(args.parent_pid)
            if args.stop_file.exists() or state == "dead" or (
                state == "alive" and token != args.parent_token
            ):
                break
            torch.cuda.synchronize(trainer.device)
            started = time.monotonic()
            metrics = trainer.update()
            torch.cuda.synchronize(trainer.device)
            metrics.update(update_seconds=time.monotonic() - started,
                           launch_attempt_id=args.attempt)
            pending.append(metrics)
            window = success_window(pending, trainer.config)
            if window is not None:
                metrics['success_window'] = window
            if metrics['update'] % trainer.config.metric_window_updates == 0:
                pending = []
            require_actor_training_owner(args.lock, owner)
            trainer.save(args.checkpoint)
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            emit(metric=metrics)
            evaluate()
            completed += 1
        require_actor_training_owner(args.lock, owner)
        trainer.save(args.checkpoint)
        emit(status="complete" if args.updates and completed == args.updates else "stopped",
             update=trainer.updates, frames=trainer.frames)


if __name__ == "__main__":
    main()
