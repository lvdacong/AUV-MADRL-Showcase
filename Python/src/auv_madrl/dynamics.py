from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import copy
import threading
from typing import Any

import torch


# Episodes rebuild their environment. Reuse only immutable programs whose exact
# coefficient bytes and input layout match, never physical state/output buffers.
_CPU_STEP_GRAPHS = OrderedDict()
_CPU_STEP_GRAPH_LOCK = threading.Lock()


@dataclass
class AuvState:
    position: torch.Tensor
    heading: torch.Tensor
    body_velocity: torch.Tensor
    desired_body_velocity: torch.Tensor
    error_integral: torch.Tensor
    adaptive_bounds: torch.Tensor
    generalized_force: torch.Tensor
    sliding_surface: torch.Tensor
    thruster_forces: torch.Tensor

    def clone(self) -> AuvState:
        return AuvState(**{name: value.clone() for name, value in self.__dict__.items()})


class BatchedAuvDynamics:
    """Torch float32 port of the Unity planar 3-DOF AFNITSMC plant."""

    def __init__(
        self,
        shared_config: dict[str, Any],
        device: torch.device | str,
        maximum_surge: float = 2.0,
        maximum_yaw: float = 0.2,
        horizontal_thruster_limit: float = 35.0,
    ):
        self.device = torch.device(device)
        cfg = shared_config["dynamics"]
        self.dt = float(shared_config["clock"]["physics_step_s"])
        self.maximum_surge = float(maximum_surge)
        self.maximum_sway = 0.5 * self.maximum_surge
        self.maximum_yaw = float(maximum_yaw)
        self.horizontal_thruster_limit = float(horizontal_thruster_limit)
        self.mass = self._tensor(cfg["mass"])
        self.linear_drag = self._tensor(cfg["linear_drag"])
        self.quadratic_drag = self._tensor(cfg["quadratic_drag"])
        self.k1 = self._tensor(cfg["k1"])
        self.k2 = self._tensor(cfg["k2"])
        self.k3 = self._tensor(cfg["k3"])
        self.k4 = self._tensor(cfg["k4"])
        self.adaptation = self._tensor(cfg["adaptation"])
        self.integral_limit = self._tensor(cfg["integral_limit"])
        self.adaptive_minimum = self._tensor(cfg["adaptive_minimum"])
        self.adaptive_maximum = self._tensor(cfg["adaptive_maximum"])
        self.switching_width = self._tensor(cfg["switching_width"])
        self.q_ratio = float(cfg["q_ratio"])
        self.gamma_ratio = float(cfg["gamma_ratio"])
        self.integral_gain = float(cfg["integral_acceleration_gain"])
        self.adaptive_relaxation = float(cfg["adaptive_relaxation"])
        axes = self._tensor(cfg["thruster_axes"])
        positions = self._tensor(cfg["thruster_positions_m"])
        yaw_moment = positions[:, 0] * axes[:, 1] - positions[:, 1] * axes[:, 0]
        self.allocation = torch.stack((axes[:, 0], axes[:, 1], yaw_moment), dim=0)
        gram = self.allocation @ self.allocation.transpose(0, 1)
        self.pseudo_inverse = self.allocation.transpose(0, 1) @ torch.linalg.inv(gram)

    def _tensor(self, value: Any) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.float32, device=self.device)

    def empty_state(self, batch_shape: tuple[int, ...]) -> AuvState:
        vector_shape = (*batch_shape, 3)
        return AuvState(
            position=torch.zeros((*batch_shape, 2), dtype=torch.float32, device=self.device),
            heading=torch.zeros(batch_shape, dtype=torch.float32, device=self.device),
            body_velocity=torch.zeros(vector_shape, dtype=torch.float32, device=self.device),
            desired_body_velocity=torch.zeros(
                vector_shape, dtype=torch.float32, device=self.device
            ),
            error_integral=torch.zeros(vector_shape, dtype=torch.float32, device=self.device),
            adaptive_bounds=self.adaptive_minimum.expand(vector_shape).clone(),
            generalized_force=torch.zeros(vector_shape, dtype=torch.float32, device=self.device),
            sliding_surface=torch.zeros(vector_shape, dtype=torch.float32, device=self.device),
            thruster_forces=torch.zeros((*batch_shape, 4), dtype=torch.float32, device=self.device),
        )

    def reset(
        self,
        state: AuvState,
        position: torch.Tensor,
        heading: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        if mask is None:
            mask = torch.ones_like(state.heading, dtype=torch.bool)
        vector_mask = mask.unsqueeze(-1)
        state.position = torch.where(vector_mask, position, state.position)
        state.heading = torch.where(mask, heading, state.heading)
        zero3 = torch.zeros_like(state.body_velocity)
        for name in (
            "body_velocity",
            "desired_body_velocity",
            "error_integral",
            "generalized_force",
            "sliding_surface",
        ):
            value = getattr(state, name)
            setattr(state, name, torch.where(vector_mask, zero3, value))
        minimum = self.adaptive_minimum.expand_as(state.adaptive_bounds)
        state.adaptive_bounds = torch.where(vector_mask, minimum, state.adaptive_bounds)
        state.thruster_forces = torch.where(
            mask.unsqueeze(-1), torch.zeros_like(state.thruster_forces), state.thruster_forces
        )

    @staticmethod
    def map_action(action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action = action.clamp(-1.0, 1.0)
        return action[..., 0] + 1.0, 0.2 * action[..., 1]

    def set_normalized_action(self, state: AuvState, action: torch.Tensor) -> None:
        action = action.clamp(-1.0, 1.0)
        state.desired_body_velocity[..., 0] = (action[..., 0] + 1.0).clamp(0.0, self.maximum_surge)
        state.desired_body_velocity[..., 1] = 0.0
        state.desired_body_velocity[..., 2] = (0.2 * action[..., 1]).clamp(
            -self.maximum_yaw, self.maximum_yaw
        )

    def set_command(
        self,
        state: AuvState,
        desired_surge: torch.Tensor,
        desired_yaw: torch.Tensor,
        maximum_surge: torch.Tensor | float | None = None,
        maximum_yaw: torch.Tensor | float | None = None,
    ) -> None:
        surge_cap = self.maximum_surge if maximum_surge is None else maximum_surge
        yaw_cap = self.maximum_yaw if maximum_yaw is None else maximum_yaw
        if not torch.is_tensor(surge_cap):
            surge_cap = torch.tensor(surge_cap, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(yaw_cap):
            yaw_cap = torch.tensor(yaw_cap, dtype=torch.float32, device=self.device)
        state.desired_body_velocity[..., 0] = torch.maximum(
            torch.zeros_like(desired_surge), torch.minimum(desired_surge, surge_cap)
        )
        state.desired_body_velocity[..., 1] = 0.0
        state.desired_body_velocity[..., 2] = torch.maximum(
            -yaw_cap, torch.minimum(desired_yaw, yaw_cap)
        )

    @staticmethod
    def _signed_power(value: torch.Tensor, power: float) -> torch.Tensor:
        return value.sign() * value.abs().pow(power)

    def _coriolis(self, velocity: torch.Tensor) -> torch.Tensor:
        u, v, r = velocity.unbind(-1)
        return torch.stack(
            (
                -self.mass[1] * v * r,
                self.mass[0] * u * r,
                (self.mass[1] - self.mass[0]) * u * v,
            ),
            dim=-1,
        )

    def _damping(self, velocity: torch.Tensor) -> torch.Tensor:
        return self.linear_drag * velocity + self.quadratic_drag * velocity.abs() * velocity

    def step(
        self,
        state: AuvState,
        active: torch.Tensor | None = None,
        maximum_surge: torch.Tensor | float | None = None,
        maximum_yaw: torch.Tensor | float | None = None,
        horizontal_thruster_limit: torch.Tensor | float | None = None,
    ) -> None:
        if self.device.type != 'cuda':
            if self.device.type == 'cpu' and state.heading.ndim == 2 and state.heading.shape[0] == 1:
                return self._step_cpu_single(state, active, maximum_surge, maximum_yaw, horizontal_thruster_limit)
            return self._step(state, active, maximum_surge, maximum_yaw, horizontal_thruster_limit)
        active = torch.ones_like(state.heading, dtype=torch.bool) if active is None else active
        limits = tuple(value if isinstance(value, torch.Tensor) else self._tensor(default if value is None else value)
                       for value, default in ((maximum_surge, self.maximum_surge),
                           (maximum_yaw, self.maximum_yaw),
                           (horizontal_thruster_limit, self.horizontal_thruster_limit)))
        inputs = (*vars(state).values(), active, *limits)
        if any(value.requires_grad for value in inputs):
            return self._step(state, active, *limits)
        # One shape per plant, bounded even when actor capacity changes. Constants
        # remain live graph inputs; replacing a tensor or scalar rebuilds the graph.
        signature = (torch.is_inference_mode_enabled(),
            tuple((value.shape, value.dtype, value.device) for value in inputs),
            tuple((name, value.data_ptr(), value.shape, value.stride())
                  for name, value in vars(self).items() if isinstance(value, torch.Tensor)),
            self.dt, self.q_ratio, self.gamma_ratio, self.integral_gain, self.adaptive_relaxation)
        if signature != getattr(self, '_step_graph_signature', None):
            self._step_graph_signature = None
            self._step_graph = None
            self._step_graph_inputs = None
            self._step_graph_output = None
            buffers = tuple(value.clone() for value in inputs)
            names = tuple(vars(state))

            def evaluate():
                working = AuvState(**dict(zip(names, buffers[:len(names)], strict=True)))
                self._step(working, *buffers[len(names):])
                return working

            # These tiny 3x4/4x3 products need no BLAS workspace. A workspace
            # captured on a pooled stream can be freed by empty_cache after a
            # later evaluation reuses that stream, invalidating a live graph.
            workspace = torch.backends.cuda.cublas_workspace_size()
            torch.backends.cuda.cublas_workspace_size(0)
            try:
                stream = torch.cuda.Stream(device=self.device)
                stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        evaluate()
                torch.cuda.current_stream(self.device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = evaluate()
            finally:
                torch.backends.cuda.cublas_workspace_size(workspace)
            self._step_graph_inputs, self._step_graph_output = buffers, output
            self._step_graph, self._step_graph_signature = graph, signature
        for target, source in zip(self._step_graph_inputs, inputs, strict=True):
            target.copy_(source)
        self._step_graph.replay()
        # Consumers retain old states and reconcile/reset fields in place. Never
        # expose reusable graph storage, or replay would overwrite those owners.
        for name, value in vars(self._step_graph_output).items():
            if name != 'desired_body_velocity':
                setattr(state, name, value.clone())

    def _step_cpu_single(self, state, active, maximum_surge, maximum_yaw, horizontal_thruster_limit):
        """Execute the same ATen operations with one CPU graph dispatch per tick.

        Specialize immutable coefficients and Python scalars. Tensor replacement
        or in-place mutation rebuilds one graph before the next physical tick.
        No CUDA graphs, approximation, reusable output buffers, or RNG are used.
        """
        active = torch.ones_like(state.heading, dtype=torch.bool) if active is None else active
        limits = tuple(value if isinstance(value, torch.Tensor) else self._tensor(default if value is None else value)
                       for value, default in ((maximum_surge, self.maximum_surge),
                           (maximum_yaw, self.maximum_yaw),
                           (horizontal_thruster_limit, self.horizontal_thruster_limit)))
        fields = tuple(vars(state))
        constants = tuple(name for name, value in vars(self).items() if isinstance(value, torch.Tensor))
        state_values = tuple(vars(state).values())
        coefficients = tuple(getattr(self, name) for name in constants)
        if (any(value.requires_grad for value in (*state_values, active, *limits, *coefficients))
                or any(value.is_inference() for value in coefficients)):
            return self._step(state, active, *limits)
        arguments = (*state_values, active, *limits)
        signature = (fields, constants,
            tuple((v.shape, v.dtype, v.stride()) for v in arguments),
            tuple((id(v), v.data_ptr(), v._version, v.shape, v.dtype, v.device, v.stride()) for v in coefficients),
            self.dt, self.q_ratio, self.gamma_ratio, self.integral_gain, self.adaptive_relaxation)
        if signature != getattr(self, '_cpu_step_signature', None):
            # Content is read only when this plant's coefficient identity/version
            # changes, not per tick. Object identity alone cannot span episodes.
            content = tuple((v.shape, v.dtype, v.stride(),
                             v.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
                            for v in coefficients)
            key = (signature[:3], content, signature[4:])
            with _CPU_STEP_GRAPH_LOCK:
                graph = _CPU_STEP_GRAPHS.get(key)
                if graph is None:
                    template = copy.copy(self)
                    # trace retains this closure. Do not let its template keep
                    # prior graphs alive after their bounded cache eviction.
                    for name in tuple(vars(template)):
                        if name.startswith('_cpu_step_'):
                            delattr(template, name)
                    for name, tensor in zip(constants, coefficients):setattr(template, name, tensor.clone())
                    def evaluate(*values):
                        working = AuvState(**dict(zip(fields, values[:len(fields)])))
                        template._step(working, *values[len(fields):])
                        return tuple(vars(working).values())
                    graph = torch.jit.trace(evaluate, arguments, check_trace=True)
                    _CPU_STEP_GRAPHS[key] = graph
                    if len(_CPU_STEP_GRAPHS) > 8:_CPU_STEP_GRAPHS.popitem(last=False)
                _CPU_STEP_GRAPHS.move_to_end(key)
            self._cpu_step_graph = graph
            # Retain the old objects until replacement, preventing allocator/id
            # reuse from disguising a new coefficient tensor as the cached one.
            self._cpu_step_coefficients = coefficients
            self._cpu_step_signature = signature
        output = self._cpu_step_graph(*arguments)
        for name, value in zip(fields, output):
            if name != 'desired_body_velocity':setattr(state, name, value)

    def _step(
        self,
        state: AuvState,
        active: torch.Tensor | None = None,
        maximum_surge: torch.Tensor | float | None = None,
        maximum_yaw: torch.Tensor | float | None = None,
        horizontal_thruster_limit: torch.Tensor | float | None = None,
    ) -> None:
        if active is None:
            active = torch.ones_like(state.heading, dtype=torch.bool)
        active3 = active.unsqueeze(-1)
        surge_cap = self.maximum_surge if maximum_surge is None else maximum_surge
        yaw_cap = self.maximum_yaw if maximum_yaw is None else maximum_yaw
        thruster_cap = (
            self.horizontal_thruster_limit
            if horizontal_thruster_limit is None
            else horizontal_thruster_limit
        )
        if not torch.is_tensor(surge_cap):
            surge_cap = torch.tensor(surge_cap, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(yaw_cap):
            yaw_cap = torch.tensor(yaw_cap, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(thruster_cap):
            thruster_cap = torch.tensor(thruster_cap, dtype=torch.float32, device=self.device)
        dt = self.dt
        error = state.desired_body_velocity - state.body_velocity
        integral = (state.error_integral + error * dt).clamp(
            -self.integral_limit, self.integral_limit
        )
        surface = (
            error
            + self.k3 * self._signed_power(error, self.q_ratio)
            + self.k4 * self._signed_power(integral, self.gamma_ratio)
        )
        adaptive = state.adaptive_bounds + self.adaptation * surface.abs() * dt
        adaptive -= self.adaptive_relaxation * (state.adaptive_bounds - self.adaptive_minimum) * dt
        adaptive = torch.maximum(
            self.adaptive_minimum, torch.minimum(self.adaptive_maximum, adaptive)
        )
        desired_acceleration = (
            self.k1 * error
            + self.k2 * self._signed_power(error, self.q_ratio)
            + self.integral_gain * integral
        )
        compensation = self._coriolis(state.body_velocity) + self._damping(state.body_velocity)
        switching = adaptive * torch.tanh(surface / self.switching_width)
        generalized = compensation + self.mass * desired_acceleration + switching
        thruster = torch.matmul(generalized, self.pseudo_inverse.transpose(0, 1))
        peak = thruster.abs().amax(dim=-1, keepdim=True)
        expanded_thruster_cap = (
            thruster_cap.unsqueeze(-1) if thruster_cap.ndim > 0 else thruster_cap
        )
        scale = torch.where(
            peak > expanded_thruster_cap,
            expanded_thruster_cap / peak.clamp_min(1.0e-12),
            torch.ones_like(peak),
        )
        thruster = thruster * scale
        realized = torch.matmul(thruster, self.allocation.transpose(0, 1))
        acceleration = (realized - compensation) / self.mass
        velocity = state.body_velocity + acceleration * dt
        sway_cap = 0.5 * surge_cap
        velocity[..., 0] = torch.maximum(
            torch.zeros_like(velocity[..., 0]), torch.minimum(velocity[..., 0], surge_cap)
        )
        velocity[..., 1] = torch.maximum(-sway_cap, torch.minimum(velocity[..., 1], sway_cap))
        velocity[..., 2] = torch.maximum(-yaw_cap, torch.minimum(velocity[..., 2], yaw_cap))
        forward = torch.stack((state.heading.sin(), state.heading.cos()), dim=-1)
        right = torch.stack((state.heading.cos(), -state.heading.sin()), dim=-1)
        world_velocity = forward * velocity[..., :1] + right * velocity[..., 1:2]
        position = state.position + world_velocity * dt
        heading = torch.remainder(state.heading + velocity[..., 2] * dt, 2.0 * torch.pi)
        state.error_integral = torch.where(active3, integral, state.error_integral)
        state.sliding_surface = torch.where(active3, surface, state.sliding_surface)
        state.adaptive_bounds = torch.where(active3, adaptive, state.adaptive_bounds)
        state.generalized_force = torch.where(active3, generalized, state.generalized_force)
        state.thruster_forces = torch.where(active.unsqueeze(-1), thruster, state.thruster_forces)
        state.body_velocity = torch.where(active3, velocity, state.body_velocity)
        state.position = torch.where(active.unsqueeze(-1), position, state.position)
        state.heading = torch.where(active, heading, state.heading)

    def reconcile_planar_constraint(
        self,
        state: AuvState,
        previous: torch.Tensor,
        blocked: torch.Tensor,
    ) -> None:
        """Constrain translation to the supplied contact position; preserve yaw/commands."""
        if blocked.device.type == "cpu" and not bool(blocked.any()):
            return
        planar_mask = blocked.unsqueeze(-1)
        state.position = torch.where(planar_mask, previous, state.position)
        for field in (
            state.body_velocity,
            state.error_integral,
            state.sliding_surface,
        ):
            field[..., :2] = torch.where(planar_mask, 0.0, field[..., :2])
        state.adaptive_bounds[..., :2] = torch.where(
            planar_mask, self.adaptive_minimum[:2], state.adaptive_bounds[..., :2]
        )
