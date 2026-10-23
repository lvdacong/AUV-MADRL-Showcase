from __future__ import annotations

from typing import Any

import torch


class CircleWorld:
    def __init__(self, world_config: dict[str, Any], device: torch.device | str):
        self.device = torch.device(device)
        self.radius = float(world_config["arena"]["radius_m"])
        self.cell_size = float(world_config["arena"]["grid_cell_size_m"])
        self.radii = torch.empty(0, dtype=torch.float32, device=self.device)

    def point_clear(self, points: torch.Tensor, clearance: float) -> torch.Tensor:
        points = points.to(device=self.device, dtype=torch.float32)
        return points.square().sum(dim=-1) < (self.radius - clearance) ** 2

    def project_motion(
        self,
        previous: torch.Tensor,
        desired: torch.Tensor,
        clearance: float,
        axis_slide: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del axis_slide
        allowed = self.point_clear(desired, clearance)
        accepted = torch.where(allowed.unsqueeze(-1), desired, previous)
        return accepted, ~allowed
