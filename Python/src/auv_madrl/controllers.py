from __future__ import annotations

from enum import IntEnum
from typing import Protocol

import torch


class _ActionDistribution(Protocol):
    loc: torch.Tensor
    scale: torch.Tensor

    def rsample(self) -> torch.Tensor: ...

    def from_standard_noise(self, noise: torch.Tensor) -> torch.Tensor: ...

    @property
    def deterministic_sample(self) -> torch.Tensor: ...


class _Policy(Protocol):
    def actor_distribution(
        self, observation: dict[str, torch.Tensor]
    ) -> _ActionDistribution: ...


class ActorControlMode(IntEnum):
    RULE = 0
    MANUAL = 1
    NEURAL = 2

    @classmethod
    def parse(cls, value: ActorControlMode | str) -> ActorControlMode:
        if isinstance(value, cls):
            return value
        names = {
            "rule": cls.RULE,
            "manual": cls.MANUAL,
            "neural": cls.NEURAL,
            "nn": cls.NEURAL,
        }
        try:
            return names[value.lower()]
        except KeyError as error:
            raise ValueError(f"Unsupported actor control mode: {value}") from error


class ActorControlMux:
    """Shared rule/manual/NN control for either infiltrator or patrol instances."""

    def __init__(
        self,
        capacity: int,
        device: torch.device | str,
        neural_policy: _Policy,
        default_mode: ActorControlMode | str = ActorControlMode.RULE,
        stochastic_neural: bool = False,
    ):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.neural_policy = neural_policy
        self.modes = [int(ActorControlMode.parse(default_mode))] * self.capacity
        self.manual_action = (
            torch.tensor([-1.0, 0.0], dtype=torch.float32, device=self.device)
            .expand(self.capacity, -1)
            .clone()
        )
        self.stochastic_neural = bool(stochastic_neural)

    def set_neural_sampling(self, stochastic: bool) -> None:
        self.stochastic_neural = bool(stochastic)

    def set_mode(self, index: int, mode: ActorControlMode | str) -> None:
        if not 0 <= int(index) < self.capacity:
            raise IndexError(f"Actor index is outside [0,{self.capacity})")
        self.modes[int(index)] = int(ActorControlMode.parse(mode))

    def _mode_tensor(self) -> torch.Tensor:
        return torch.tensor(self.modes, dtype=torch.int8, device=self.device)

    def set_manual_action(self, index: int, action: torch.Tensor | tuple[float, float]) -> None:
        if not 0 <= int(index) < self.capacity:
            raise IndexError(f"Actor index is outside [0,{self.capacity})")
        value = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if value.shape != (2,):
            raise ValueError("A manual action must contain normalized [surge,yaw]")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("A manual action must contain finite values")
        self.manual_action[int(index)] = value.clamp(-1.0, 1.0)

    @torch.no_grad()
    def resolve(
        self,
        observation: dict[str, torch.Tensor],
        active: torch.Tensor,
        rule_action: torch.Tensor,
        standard_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if observation["vector"].shape[0] != self.capacity:
            raise ValueError("Actor control mux and observation sizes differ")
        if rule_action.shape != (self.capacity, 2):
            raise ValueError("Rule action must be [actor_capacity,2]")
        active = active.to(device=self.device, dtype=torch.bool)
        modes = self._mode_tensor()
        action = rule_action.to(device=self.device, dtype=torch.float32).clone()
        manual = (modes == int(ActorControlMode.MANUAL)) & active
        action[manual] = self.manual_action[manual]
        neural = (modes == int(ActorControlMode.NEURAL)) & active
        if bool(neural.any()):
            distribution = self.neural_policy.actor_distribution(
                {key: value[neural] for key, value in observation.items()}
            )
            action[neural] = (
                distribution.from_standard_noise(standard_noise[neural])
                if self.stochastic_neural and standard_noise is not None else distribution.rsample()
                if self.stochastic_neural
                else distribution.deterministic_sample
            )
        override = (modes != int(ActorControlMode.RULE)) & active
        return action.clamp(-1.0, 1.0), override

    @torch.no_grad()
    def resolve_active(
        self,
        actor_indices: torch.Tensor,
        rule_action: torch.Tensor,
        distribution: _ActionDistribution | None,
        standard_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actor_indices = actor_indices.to(device=self.device, dtype=torch.int64)
        expected = (actor_indices.shape[0], 2)
        if rule_action.shape != expected:
            raise ValueError(f"Active rule action must be {list(expected)}")
        modes = self._mode_tensor()[actor_indices]
        action = rule_action.to(device=self.device, dtype=torch.float32).clone()
        manual = modes == int(ActorControlMode.MANUAL)
        action[manual] = self.manual_action[actor_indices[manual]]
        neural = modes == int(ActorControlMode.NEURAL)
        if bool(neural.any()):
            if distribution is None or distribution.loc.shape[0] != actor_indices.shape[0]:
                raise ValueError("Active Actor distribution does not match active indices")
            action[neural] = (
                distribution.from_standard_noise(standard_noise)[neural]
                if self.stochastic_neural and standard_noise is not None else torch.distributions.Normal(
                    distribution.loc[neural], distribution.scale[neural]
                ).rsample().tanh()
                if self.stochastic_neural
                else distribution.deterministic_sample[neural]
            )
        return action.clamp(-1.0, 1.0), modes != int(ActorControlMode.RULE)

    def source_names(self) -> list[str]:
        names = {
            int(ActorControlMode.RULE): "rule",
            int(ActorControlMode.MANUAL): "manual",
            int(ActorControlMode.NEURAL): "nn",
        }
        return [names[value] for value in self.modes]
