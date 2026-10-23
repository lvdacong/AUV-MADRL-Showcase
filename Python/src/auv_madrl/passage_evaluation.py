"""CPU evaluation of supported sides with private environments and policy RNG."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict

import numpy as np
import torch

from .passage import CONTRACT, PassageEnv


def model_sha256(model):
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def evaluate_policy(trainer, *, episodes_per_side=256, evaluation_seed=20260923):
    if type(episodes_per_side) is not int or episodes_per_side < 1:
        raise ValueError('Evaluation requires a positive number of episodes per side')
    model = copy.deepcopy(trainer.model).cpu().eval()
    results = {}
    for side_index, side in enumerate(('left', 'right')):
        probability = (trainer.config.left_probability if side == 'left'
                       else 1 - trainer.config.left_probability)
        if probability == 0:
            continue  # A fixed-map study evaluates its training map only.
        env = PassageEnv(episodes_per_side, seed=evaluation_seed,
            collision_termination=trainer.config.collision_termination,
            maximum_steps=trainer.config.maximum_episode_steps)
        env.reset(side=side)
        generator = torch.Generator().manual_seed(evaluation_seed + side_index)
        pending = np.ones(episodes_per_side, dtype=bool)
        success = np.zeros(episodes_per_side, dtype=bool)
        truncated = np.zeros(episodes_per_side, dtype=bool)
        steps = np.zeros(episodes_per_side, dtype=np.int64)
        collisions = np.zeros(episodes_per_side, dtype=np.int64)
        for _ in range(env.maximum_steps):
            distribution, _ = model(env.observe())
            action = torch.multinomial(distribution.probs, 1, generator=generator).squeeze(-1)
            _, reward, _, truncation, collision = env.step(action.numpy())
            collisions[pending] += collision[pending]
            finished = pending & env.done
            success[finished] = reward[finished] > 0
            truncated[finished] = truncation[finished]
            steps[finished] = env.steps[finished]
            pending[finished] = False
            if not pending.any():
                break
            # Only the first episode of each lane counts. Reset completed lanes
            # so they can coexist with unfinished lanes in the shared step API.
            env.reset(env.done, side=side)
        if pending.any():
            raise AssertionError('Evaluation left unfinished first episodes')
        results[side] = dict(episodes=episodes_per_side, successes=int(success.sum()),
            success_rate=float(success.mean()), truncations=int(truncated.sum()),
            collision_terminations=int((~success & ~truncated).sum()),
            mean_steps=float(steps.mean()),
            mean_success_steps=float(steps[success].mean()) if success.any() else None,
            collision_steps=int(collisions.sum()), episode_steps=steps.tolist())
    return dict(contract=CONTRACT, update=trainer.updates, frames=trainer.frames,
        training_seed=trainer.config.seed, left_probability=trainer.config.left_probability,
        config=asdict(trainer.config), evaluation_seed=evaluation_seed,
        sampling='stochastic', device='cpu', model_sha256=model_sha256(model), results=results)
