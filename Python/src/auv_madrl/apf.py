"""Observation-only artificial potential field for the unified match ATK.

There is no model, random draw, trajectory planner, or retained lane state here.
Training, evaluation and Viewer must persist and pass the same ``APFConfig``.
The returned normalized commands go through the existing physical actuator path;
in particular ``[-1, 0]`` requests a stop, rather than clearing physical velocity.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import math
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .multi_goal_reach_env import MultiGoalReachEnv


@dataclass(frozen=True)
class APFConfig:
    """One attraction and one shared surface-clearance repulsion coefficient.

    v2 deliberately differs from the earlier experimental, per-role v1 rule.
    DEF, other ATK and obstacles share the same gain and ten-metre influence.
    """
    type: str = "apf"
    version: int = 2
    attraction_gain: float = 1.0
    repulsion_gain_m3: float = 8.0

    def __post_init__(self):
        if self.type != 'apf' or type(self.version) is not int or self.version != 2:
            raise ValueError('Unsupported APF rule version')
        for name in ('attraction_gain', 'repulsion_gain_m3'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'APF {name} must be finite')
            if value < 0 or (name == 'attraction_gain' and value == 0):
                raise ValueError(f'Invalid APF {name}')
            object.__setattr__(self, name, float(value))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping) or value.keys() - {f.name for f in fields(cls)}:
            raise ValueError('Invalid APF configuration')
        return cls(**dict(value))


def _positive_scale(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"APF {name} must be finite and positive")
    return value


def _entity_geometry(observation, name, vector, active, position_scale_m, *, obstacle=False):
    entities = observation[name + "_entities"]
    mask = observation[name + "_mask"]
    expected_dimensions = vector.ndim + 1
    feature_count = 3 if obstacle else 2
    if (entities.ndim != expected_dimensions or entities.shape[:-2] != vector.shape[:-1]
            or entities.shape[-1] < feature_count or mask.shape != entities.shape[:-1]):
        raise ValueError(f"APF {name} entities/mask have incompatible shapes")
    if entities.device != vector.device or mask.device != vector.device:
        raise ValueError("APF observation tensors must share a device")
    visible = mask.to(torch.bool) & active.unsqueeze(-1)
    # Mask before arithmetic: NaN/uninitialized padding is never observed.
    geometry = torch.where(visible.unsqueeze(-1), entities[..., :feature_count], 0.)
    torch._assert_async(torch.isfinite(geometry).all(), f"Nonfinite observable APF {name} geometry")
    return geometry * position_scale_m, visible


def _repulsion(geometry, visible, vessel_radius_m, gain, influence, config, *, obstacle=False):
    position = geometry[..., :2]
    distance = position.norm(dim=-1)
    other_radius = geometry[..., 2] if obstacle else vessel_radius_m
    if obstacle:
        torch._assert_async((other_radius >= 0).all(), "Negative observable APF obstacle radius")
    gap = (distance - vessel_radius_m - other_radius).clamp_min(0.1)
    magnitude = (gain * (gap.reciprocal() - 1. / influence).clamp_min(0.) / gap.square())
    magnitude = torch.where(visible, magnitude.clamp_max(20.0), 0.)
    outward = -position / distance.clamp_min(1.0e-6).unsqueeze(-1)
    force = (outward * magnitude.unsqueeze(-1)).sum(dim=-2)
    # A coincident center has no observable outward direction; stop safely.
    coincident = (visible & (distance <= 1.0e-6)).any(dim=-1)
    return force, coincident


@torch.no_grad()
def apf_action(
    observation: Mapping[str, torch.Tensor],
    *,
    position_scale_m: float,
    vessel_radius_m: float,
    active: torch.Tensor | None = None,
    config: APFConfig | Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """Return ``[..., 2]`` actions from existing unified *actor* observations.

    Accepts flat actor rows, ``[lane, actor, ...]``, and empty batches on CPU or
    CUDA. Only body-frame position/radius fields, actor-visible masks, own alive
    flag and own normalized yaw are read. Critic streams and all other features
    are ignored. Nearest visible live target wins, with stable slot-order ties.
    No target, dead/padding, coincident centers, or a zero resultant field yield
    the exact stop request ``[-1, 0]``. A symmetric obstacle can produce a local
    minimum; this basic APF deliberately does not introduce an escape planner.
    """
    config = APFConfig() if config is None else config
    if not isinstance(config, APFConfig):
        config = APFConfig.from_dict(config)
    position_scale_m = _positive_scale(position_scale_m, "position_scale_m")
    vessel_radius_m = _positive_scale(vessel_radius_m, "vessel_radius_m")
    required = ("vector", "target_entities", "target_mask", "defender_entities", "defender_mask",
                "obstacle_entities", "obstacle_mask")
    missing = [key for key in required if key not in observation]
    if missing:
        raise ValueError(f"APF requires unified actor observation fields: {missing}")
    vector = observation["vector"]
    if vector.ndim < 1 or vector.shape[-1] < 5 or not vector.is_floating_point():
        raise ValueError("APF requires the unified actor vector with own yaw and alive fields")
    alive = vector[..., 4] > 0.5
    if active is None:
        active = observation.get("actor_alive", alive)
    if active.shape != vector.shape[:-1] or active.device != vector.device:
        raise ValueError("APF active mask must match the actor leading dimensions and device")
    active = active.to(torch.bool) & alive
    own_yaw = torch.where(active, vector[..., 2], 0.)
    torch._assert_async(torch.isfinite(own_yaw).all(), "Nonfinite observable APF own yaw")
    stop = vector.new_tensor([-1., 0.]).expand(*vector.shape[:-1], 2)
    target, target_mask = _entity_geometry(observation, "target", vector, active, position_scale_m)
    if target.shape[-2] == 0:
        return stop.clone()
    target_distance = target.norm(dim=-1)
    nearest = torch.where(target_mask, target_distance, torch.inf).argmin(dim=-1)
    chosen = target.gather(-2, nearest[..., None, None].expand(*nearest.shape, 1, 2)).squeeze(-2)
    chosen_distance = chosen.norm(dim=-1)
    force = config.attraction_gain * chosen / chosen_distance.clamp_min(1.0e-6).unsqueeze(-1)
    valid = active & target_mask.any(dim=-1) & (chosen_distance > 1.0e-6)
    sources = (
        ("defender", config.repulsion_gain_m3, 10.0, False),
        ("obstacle", config.repulsion_gain_m3, 10.0, True),
        ("learner", config.repulsion_gain_m3, 10.0, False),
    )
    for name, gain, influence, obstacle in sources:
        if gain == 0:
            continue
        if name + "_entities" not in observation or name + "_mask" not in observation:
            raise ValueError(f"APF {name} repulsion requires actor-visible entities and mask")
        geometry, visible = _entity_geometry(
            observation, name, vector, active, position_scale_m, obstacle=obstacle)
        repulsion, coincident = _repulsion(
            geometry, visible, vessel_radius_m, gain, influence, config, obstacle=obstacle)
        force = force + repulsion
        valid = valid & ~coincident
    norm = force.norm(dim=-1)
    valid = valid & (norm > 1.0e-6)
    bearing = torch.atan2(force[..., 0], force[..., 1])
    surge_fraction = (force[..., 1] / norm.clamp_min(1.0e-6)).clamp(0., 1.).square()
    surge_fraction = surge_fraction * (norm / config.attraction_gain).clamp(max=1.)
    yaw = 1.5 * bearing - 0.35 * own_yaw
    action = torch.stack((2. * surge_fraction - 1., yaw.clamp(-1., 1.)), dim=-1)
    torch._assert_async(torch.isfinite(action).all(), "Nonfinite APF action")
    return torch.where(valid.unsqueeze(-1), action, stop)


def apf_action_for_env(
    env: MultiGoalReachEnv,
    observation: Mapping[str, torch.Tensor] | None = None,
    *,
    config: APFConfig | Mapping[str, Any] | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Thin adapter using only the public ATK observation plus fixed scales.

    ``mask`` selects lanes when requesting a fresh observation. Supplying an
    observation is useful for a decision's existing shared observation bundle.
    No provider, environment, RNG, or physical state is modified by this call.
    """
    if observation is not None and mask is not None:
        raise ValueError("Pass an APF observation or a lane selection, not both")
    if observation is None:
        observation = env.training_observation_for("learner", mask, consumer="actor")
    return apf_action(observation, position_scale_m=env.position_scale_m,
                      vessel_radius_m=env.vessel_radius_m, config=config)
