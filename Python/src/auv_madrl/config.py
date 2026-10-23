from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    repository_root: Path
    shared: dict[str, Any]
    goal_reach: dict[str, Any]
    world: dict[str, Any]
    goal_reach_trainer: dict[str, Any]


def load_project_config(config_root: str | Path | None = None) -> ProjectConfig:
    repository_root = next(
        (
            candidate
            for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
            if (candidate / "Python" / "viewer" / "goal_reach.html").is_file()
        ),
        None,
    )
    if repository_root is None:
        raise FileNotFoundError("Run GoalReach from the AUV-MADRL checkout")
    root = (
        Path(config_root).resolve()
        if config_root is not None
        else repository_root / "Python" / "configs"
    )
    shared_path = root / "contracts" / "shared.yaml"
    with shared_path.open(encoding="utf-8") as stream:
        shared = yaml.safe_load(stream)
    if not isinstance(shared, dict):
        raise ValueError(f"Invalid shared configuration: {shared_path}")
    return ProjectConfig(repository_root, shared, {}, {}, {})
