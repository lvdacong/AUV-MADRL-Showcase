from __future__ import annotations

import hashlib
import json
import math
import numbers
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linprog

from .match_metrics import MatchEndReason

POPULATION_PROTOCOL_VERSION = 2
CAMPS = ("attack", "defense")
EVALUATION_END_REASONS = frozenset(reason.value for reason in MatchEndReason)


def validate_population_protocol_version(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value not in (1, POPULATION_PROTOCOL_VERSION):
        raise ValueError(f"Population protocol_version must be integer 1 or {POPULATION_PROTOCOL_VERSION}; got {value!r}")
    return int(value)


def normalized_evaluation_summary(value, *, require_extended=False):
    result = dict(value)
    extended = {"unresolved", "defender_wins", "evaluation_decisions", "tail_error_bound"}
    if (require_extended or extended.intersection(result)) and not extended.issubset(result):
        raise ValueError("Censored evaluation summary is incomplete; missing: " + ", ".join(sorted(extended - result.keys())))
    result.setdefault("unresolved", 0)
    result.setdefault("defender_wins", result["sample_count"] - result["attacker_wins"])
    result.setdefault("evaluation_decisions", None)
    result.setdefault("tail_error_bound", 0.0)
    result["end_reasons"] = dict(result["end_reasons"])
    for reason in ("defender_elimination", "truncation"):
        result["end_reasons"].setdefault(reason, 0)
    return result


def validate_evaluation_summary(
    value: Mapping[str, Any],
    *,
    expected_samples: int,
    horizon_ticks: int,
    expected_sample_id: str | None = None,
    objective: str | None = None,
    gamma: float | None = None,
    evaluation_decisions: int | None = None,
    ticks_per_decision: int | None = None,
    require_extended: bool = False,
) -> None:
    """Reject incomplete or numerically impossible fixed-sample evaluation evidence."""
    if not isinstance(value, Mapping):
        raise ValueError("Evaluation summary must be a mapping")
    required = {
        "sample_id", "sample_count", "utility_mean", "utility_standard_error",
        "attacker_wins", "terminal_ticks_sum", "end_reasons",
    }
    missing = required - set(value)
    if missing:
        raise ValueError("Evaluation summary is missing: " + ", ".join(sorted(missing)))

    value = normalized_evaluation_summary(value, require_extended=require_extended)

    def strict_int(name: str) -> int:
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, numbers.Integral):
            raise ValueError(f"Evaluation {name} must be an integer")
        return int(item)

    if isinstance(expected_samples, bool) or not isinstance(expected_samples, numbers.Integral) or int(expected_samples) < 1:
        raise ValueError("Expected sample count must be a positive integer")
    if isinstance(horizon_ticks, bool) or not isinstance(horizon_ticks, numbers.Integral) or int(horizon_ticks) < 1:
        raise ValueError("Evaluation horizon must be a positive integer")
    count = strict_int("sample_count")
    if count != int(expected_samples):
        raise ValueError("Evaluation sample count is incomplete")
    sample_id = value["sample_id"]
    if not isinstance(sample_id, str) or re.fullmatch(r"[0-9a-f]{64}", sample_id) is None:
        raise ValueError("Evaluation sample_id must be a SHA-256 identity")
    if expected_sample_id is not None and sample_id != expected_sample_id:
        raise ValueError("Evaluation sample_id does not match the request")
    wins = strict_int("attacker_wins")
    ticks = strict_int("terminal_ticks_sum")
    unresolved = strict_int("unresolved")
    defender_wins = strict_int("defender_wins")
    if unresolved < 0 or defender_wins < 0 or wins + defender_wins + unresolved != count:
        raise ValueError("Evaluation outcomes must partition all samples")
    budget = value["evaluation_decisions"]
    if budget is not None and (type(budget) is not int or budget < 1):
        raise ValueError("Evaluation decisions must be a positive integer")
    if unresolved and budget is None:
        raise ValueError("Unresolved evaluation requires an explicit observation budget")
    if evaluation_decisions is not None and budget != evaluation_decisions:
        raise ValueError("Evaluation decisions do not match the request")
    tail = value["tail_error_bound"]
    if type(tail) not in (int,float) or not math.isfinite(tail) or not 0 <= tail <= unresolved / count:
        raise ValueError("Evaluation tail error bound is invalid")
    if objective is not None and objective != 'outcome' and unresolved:
        raise ValueError("Time rewards require a complete terminal evaluation")
    if objective == 'outcome' and gamma is not None and budget is not None:
        if not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("Evaluation gamma is invalid")
        if not math.isclose(tail, unresolved / count * gamma ** budget, rel_tol=1e-12, abs_tol=0):
            raise ValueError("Evaluation tail error bound differs from the scientific contract")
    if not 0 <= wins <= count:
        raise ValueError("Evaluation attacker_wins is out of range")
    observed_horizon = budget * ticks_per_decision if budget is not None and ticks_per_decision is not None else horizon_ticks
    if not 0 <= ticks <= (count - unresolved) * int(observed_horizon):
        raise ValueError("Evaluation terminal_ticks_sum is out of range")
    statistics = []
    for name in ("utility_mean", "utility_standard_error"):
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, numbers.Real):
            raise ValueError(f"Evaluation {name} must be numeric")
        statistics.append(float(item))
    mean, standard_error = statistics
    if not math.isfinite(mean) or not -1.0 <= mean <= 1.0:
        raise ValueError("Evaluation utility_mean is invalid")
    if not math.isfinite(standard_error) or standard_error < 0.0:
        raise ValueError("Evaluation utility_standard_error is invalid")
    if unresolved == count and (mean != 0 or standard_error != 0):
        raise ValueError("All-unresolved evaluation must have zero observed mean and standard error")
    reasons = value["end_reasons"]
    if not isinstance(reasons, Mapping) or set(reasons) != EVALUATION_END_REASONS:
        raise ValueError("Evaluation end_reasons has an invalid key set")
    reason_total = 0
    for name, item in reasons.items():
        if isinstance(item, bool) or not isinstance(item, numbers.Integral) or int(item) < 0:
            raise ValueError(f"Evaluation end reason {name} must be a nonnegative integer")
        reason_total += int(item)
    if int(reasons["truncation"]) != unresolved:
        raise ValueError("Evaluation truncation count must equal unresolved")
    attack_reasons = ("breach", "target_capture", "target_casualty", "defender_casualty", "defender_elimination")
    if sum(int(reasons[name]) for name in attack_reasons) != wins:
        raise ValueError("Evaluation end reasons disagree with the winner counts")
    if reason_total != count or int(reasons[MatchEndReason.NONE.value]) != 0:
        raise ValueError("Evaluation end_reasons is not a complete terminal partition")


def actor_state_hash(state: Mapping[str, torch.Tensor]) -> str:
    """Hash Actor tensors without depending on pickle or mapping insertion order."""
    digest = hashlib.sha256()
    actor_items = [(name, value) for name, value in state.items() if name.startswith("actor")]
    if not actor_items:
        raise ValueError("Actor state is empty")
    for name, value in sorted(actor_items):
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ImmutablePolicyRef:
    policy_id: str
    camp: str
    checkpoint: str
    parameter_group: str
    actor_hash: str
    global_update: int

    def __post_init__(self) -> None:
        if self.camp not in CAMPS:
            raise ValueError(f"Unsupported camp: {self.camp}")
        if not self.policy_id or not self.parameter_group:
            raise ValueError("Policy identity is incomplete")
        path = Path(self.checkpoint)
        if path.name == "checkpoint-latest.pt" or not (
            path.name == "checkpoint-initial.pt" or path.name.startswith("checkpoint-")
        ):
            raise ValueError("Population policies require immutable numeric checkpoints")
        if len(self.actor_hash) != 64:
            raise ValueError("Actor hash must be SHA-256")
        if self.global_update < 0:
            raise ValueError("Policy update must be nonnegative")


@dataclass
class MatrixCell:
    attack_policy_id: str
    defense_policy_id: str
    sample_id: str
    sample_count: int
    utility_mean: float
    utility_standard_error: float
    attacker_wins: int
    terminal_ticks_sum: int
    end_reasons: dict[str, int]
    defender_wins: int | None = None
    unresolved: int = 0
    evaluation_decisions: int | None = None
    tail_error_bound: float = 0.0

    def __post_init__(self):
        if self.defender_wins is None:
            self.defender_wins = self.sample_count - self.attacker_wins - self.unresolved
        self.end_reasons = normalized_evaluation_summary(asdict(self))["end_reasons"]

    def validate(self, expected_samples: int, horizon_ticks: int, **context) -> None:
        validate_evaluation_summary(
            asdict(self), expected_samples=expected_samples, horizon_ticks=horizon_ticks, **context
        )


def solve_zero_sum(
    matrix: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    payoff = np.asarray(matrix, dtype=np.float64)
    if payoff.ndim != 2 or min(payoff.shape, default=0) == 0 or not np.isfinite(payoff).all():
        raise ValueError("Expected a nonempty, finite payoff matrix")
    rows, columns = payoff.shape
    options = {
        "time_limit": 5.0,
        "primal_feasibility_tolerance": 1e-9,
        "dual_feasibility_tolerance": 1e-9,
    }
    row = linprog(
        np.r_[np.zeros(rows), -1.0],
        A_ub=np.c_[-payoff.T, np.ones(columns)],
        b_ub=np.zeros(columns),
        A_eq=np.r_[np.ones(rows), 0.0][None, :],
        b_eq=[1.0],
        bounds=[(0.0, 1.0)] * rows + [(None, None)],
        method="highs",
        options=options,
    )
    column = linprog(
        np.r_[np.zeros(columns), 1.0],
        A_ub=np.c_[payoff, -np.ones(rows)],
        b_ub=np.zeros(rows),
        A_eq=np.r_[np.ones(columns), 0.0][None, :],
        b_eq=[1.0],
        bounds=[(0.0, 1.0)] * columns + [(None, None)],
        method="highs",
        options=options,
    )
    if not row.success or not column.success:
        raise RuntimeError(f"LP failed: row={row.message}; column={column.message}")

    def probability(value: np.ndarray) -> np.ndarray:
        if not np.isfinite(value).all() or value.min() < -1e-8 or abs(value.sum() - 1.0) > 1e-8:
            raise RuntimeError("Invalid LP probabilities")
        result = np.maximum(value, 0.0)
        return result / result.sum()

    p, q = probability(row.x[:-1]), probability(column.x[:-1])
    lower, upper = float(np.min(p @ payoff)), float(np.max(payoff @ q))
    value = float(p @ payoff @ q)
    gap = max(0.0, upper - lower)
    if gap > 1e-7 * max(1.0, float(np.abs(payoff).max())):
        raise RuntimeError("LP residual exceeds tolerance")
    return p, q, value, gap


def payoff_utility(
    attacker_won: torch.Tensor,
    terminal_ticks: torch.Tensor,
    *,
    objective: str,
    gamma: float,
    horizon_ticks: int,
    ticks_per_decision: int,
    protected_time: torch.Tensor | None = None,
    physics_step_s: float | None = None,
    resolved: torch.Tensor | None = None,
) -> torch.Tensor:
    sign = attacker_won.to(torch.float64).mul(2.0).sub(1.0)
    if resolved is not None:
        if resolved.shape != attacker_won.shape or resolved.dtype != torch.bool:
            raise ValueError("Resolved mask must be bool with the outcome shape")
        if objective != "outcome" and not bool(resolved.all()):
            raise ValueError("Time objectives require complete terminal evaluation")
        sign = torch.where(resolved, sign, 0.0)
    ticks = terminal_ticks.to(torch.float64)
    if objective == "outcome":
        if gamma == 1.0:
            return sign
        decision = torch.ceil(ticks / ticks_per_decision).clamp_min(1.0)
        return sign * gamma ** (decision - 1.0)
    if objective == "outcome_time":
        if gamma != 1.0:
            raise ValueError("Outcome time cannot also use gamma discounting")
        return sign * (1.0 - 0.25 * ticks / horizon_ticks)
    if objective == "protection_time":
        if gamma != 1.0:
            raise ValueError("Protection time requires gamma=1")
        if protected_time is None or physics_step_s is None:
            raise ValueError("Protection time utility requires protected-time observations")
        horizon_seconds = horizon_ticks * physics_step_s
        return 2.0 * protected_time.to(torch.float64) / horizon_seconds - 1.0
    raise ValueError(f"Unsupported payoff objective: {objective}")


@dataclass
class StageTracker:
    start_update: int
    maximum_updates: int = 400
    minimum_updates: int = 200
    validation_interval: int = 100
    improvement_threshold: float = 0.02
    significant_best: float | None = None
    raw_best: float | None = None
    raw_best_update: int | None = None
    patience: int = 0
    validation_scores: list[tuple[int, float]] = field(default_factory=list)

    def observe(self, global_update: int, score: float) -> bool:
        if not math.isfinite(score):
            raise ValueError("Validation score must be finite")
        elapsed = int(global_update) - self.start_update
        if elapsed < 0 or elapsed > self.maximum_updates or elapsed % self.validation_interval:
            raise ValueError("Validation is outside the fixed stage cadence")
        if self.validation_scores and global_update <= self.validation_scores[-1][0]:
            raise ValueError("A validation point may be recorded only once")
        self.validation_scores.append((int(global_update), float(score)))
        if self.raw_best is None or score > self.raw_best:
            self.raw_best, self.raw_best_update = float(score), int(global_update)
        if (
            self.significant_best is None
            or score > self.significant_best + self.improvement_threshold
        ):
            self.significant_best, self.patience = float(score), 0
        else:
            self.patience += 1
        return elapsed >= self.maximum_updates or (
            elapsed >= self.minimum_updates and self.patience >= 2
        )


def _unit_uniform(
    seed: int, lane: torch.Tensor, episode: torch.Tensor, domain: int
) -> torch.Tensor:
    # Integer arithmetic is deterministic on CPU/CUDA and independent of global RNG state.
    value = (
        lane.to(torch.int64) * 1_103_515_245
        + episode.to(torch.int64) * 2_654_435_761
        + int(seed) * 97_531
        + int(domain) * 433_494_437
    )
    value = torch.remainder(value, 2_147_483_647)
    return (value.to(torch.float64) + 0.5) / 2_147_483_647.0


@dataclass
class PopulationOpponentContext:
    """Frozen phase-local opponent assignment, sampled only at real episode boundaries."""

    active_camp: str
    seed: int
    probabilities: torch.Tensor
    policy_ids: tuple[str, ...]
    models: tuple[Any, ...]
    lane_policy_index: torch.Tensor
    lane_episode_count: torch.Tensor
    sample_domain: int

    @classmethod
    def create(
        cls,
        *,
        active_camp: str,
        seed: int,
        probabilities: Sequence[float],
        policy_ids: Sequence[str],
        models: Sequence[Any],
        lane_count: int,
        device: torch.device,
        sample_domain: int,
        lane_episode_count: torch.Tensor | None = None,
    ) -> PopulationOpponentContext:
        if active_camp not in CAMPS:
            raise ValueError("Unsupported active camp")
        probs = torch.as_tensor(probabilities, dtype=torch.float64, device=device)
        if (
            probs.ndim != 1
            or probs.numel() == 0
            or len(policy_ids) != probs.numel()
            or len(models) != probs.numel()
            or not torch.isfinite(probs).all()
            or bool((probs < 0).any())
            or not math.isclose(float(probs.sum()), 1.0, abs_tol=1e-8)
            or len(set(policy_ids)) != len(policy_ids)
        ):
            raise ValueError("Invalid frozen opponent mixture")
        counts = (
            torch.zeros(lane_count, dtype=torch.int64, device=device)
            if lane_episode_count is None
            else lane_episode_count.to(device=device, dtype=torch.int64).clone()
        )
        if counts.shape != (lane_count,) or bool((counts < 0).any()):
            raise ValueError("Invalid lane episode counters")
        context = cls(
            active_camp=active_camp,
            seed=int(seed),
            probabilities=probs / probs.sum(),
            policy_ids=tuple(policy_ids),
            models=tuple(models),
            lane_policy_index=torch.zeros(lane_count, dtype=torch.int64, device=device),
            lane_episode_count=counts,
            sample_domain=int(sample_domain),
        )
        context.resample(torch.arange(lane_count, device=device), increment=False)
        return context

    @property
    def frozen_roles(self) -> frozenset[str]:
        return frozenset(("defender", "target") if self.active_camp == "attack" else ("learner",))

    @property
    def active_roles(self) -> frozenset[str]:
        return frozenset(("learner",) if self.active_camp == "attack" else ("defender", "target"))

    def resample(self, lanes: torch.Tensor, *, increment: bool = True) -> None:
        lanes = lanes.to(device=self.lane_policy_index.device, dtype=torch.int64).reshape(-1)
        if lanes.numel() == 0:
            return
        if bool(((lanes < 0) | (lanes >= self.lane_policy_index.numel())).any()):
            raise ValueError("Opponent resample lane is out of range")
        if increment:
            self.lane_episode_count[lanes] += 1
        uniform = _unit_uniform(
            self.seed,
            lanes,
            self.lane_episode_count[lanes],
            self.sample_domain,
        )
        index = torch.searchsorted(self.probabilities.cumsum(0), uniform, right=False)
        self.lane_policy_index[lanes] = index.clamp_max(len(self.models) - 1)

    def role_models(self, role: str) -> tuple[Any, ...] | None:
        return self.models if role in self.frozen_roles else None

    def state_dict(self) -> dict[str, Any]:
        return {
            "active_camp": self.active_camp,
            "seed": self.seed,
            "probabilities": self.probabilities.detach().cpu(),
            "policy_ids": list(self.policy_ids),
            "lane_policy_index": self.lane_policy_index.detach().cpu(),
            "lane_episode_count": self.lane_episode_count.detach().cpu(),
            "sample_domain": self.sample_domain,
        }


@dataclass
class PopulationState:
    condition: str
    objective: str
    gamma: float
    round_index: int = 0
    active_camp: str = "attack"
    phase_status: str = "training"
    phase_update: int = 0
    publication_sequence: int = 0
    train_updates: int = 0
    evaluation_matches: int = 0
    evaluation_keys: list[str] = field(default_factory=list)
    drain_transitions: int = 0
    matrix_generation: int = 0
    attack_pool: list[ImmutablePolicyRef] = field(default_factory=list)
    defense_pool: list[ImmutablePolicyRef] = field(default_factory=list)
    cells: dict[str, MatrixCell] = field(default_factory=dict)
    attack_probabilities: list[float] = field(default_factory=list)
    defense_probabilities: list[float] = field(default_factory=list)
    restricted_game_value: float | None = None
    restricted_game_duality_gap: float | None = None
    stage: StageTracker | None = None
    opponent: dict[str, Any] | None = None
    drain: dict[str, Any] | None = None

    @staticmethod
    def cell_key(attack_policy_id: str, defense_policy_id: str) -> str:
        return f"{attack_policy_id}\0{defense_policy_id}"

    def payoff_matrix(self, expected_samples: int, horizon_ticks: int) -> np.ndarray:
        if not self.attack_pool or not self.defense_pool:
            raise ValueError("Both population axes must be nonempty")
        rows = []
        for attack in self.attack_pool:
            row = []
            for defense in self.defense_pool:
                cell = self.cells.get(self.cell_key(attack.policy_id, defense.policy_id))
                if cell is None:
                    raise ValueError("Payoff matrix has a missing cell")
                cell.validate(expected_samples, horizon_ticks, objective=self.objective, gamma=self.gamma)
                row.append(cell.utility_mean)
            rows.append(row)
        return np.asarray(rows, dtype=np.float64)

    def solve(self, expected_samples: int, horizon_ticks: int) -> None:
        matrix = self.payoff_matrix(expected_samples, horizon_ticks)
        p, q, value, gap = solve_zero_sum(matrix)
        self.attack_probabilities = p.tolist()
        self.defense_probabilities = q.tolist()
        self.restricted_game_value = value
        self.restricted_game_duality_gap = gap
        self.matrix_generation += 1

    def add_policy(self, policy: ImmutablePolicyRef) -> ImmutablePolicyRef:
        pool = self.attack_pool if policy.camp == "attack" else self.defense_pool
        duplicate = next((item for item in pool if item.actor_hash == policy.actor_hash), None)
        if duplicate is not None:
            return duplicate
        if len(pool) >= 5:
            raise ValueError("Population pool cap exceeded")
        if any(item.policy_id == policy.policy_id for item in pool):
            raise ValueError("Policy id is not unique")
        pool.append(policy)
        return policy

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["protocol_version"] = POPULATION_PROTOCOL_VERSION
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, horizon_ticks=None, ticks_per_decision=None) -> PopulationState:
        version = validate_population_protocol_version(value.get("protocol_version"))
        fields = dict(value)
        fields.pop("protocol_version", None)
        fields["evaluation_matches"] = validate_population_evaluation_matches(
            fields.get("evaluation_matches", 0)
        )
        fields["attack_pool"] = [
            ImmutablePolicyRef(**item) for item in fields.get("attack_pool", [])
        ]
        fields["defense_pool"] = [
            ImmutablePolicyRef(**item) for item in fields.get("defense_pool", [])
        ]
        cells = {}
        for key, item in fields.get("cells", {}).items():
            try:
                raw = normalized_evaluation_summary(item, require_extended=version == POPULATION_PROTOCOL_VERSION)
                validate_evaluation_summary(raw, expected_samples=raw['sample_count'],
                    horizon_ticks=horizon_ticks if horizon_ticks is not None else max(1, raw['terminal_ticks_sum']),
                    ticks_per_decision=ticks_per_decision, objective=fields['objective'], gamma=fields['gamma'])
                cells[key] = MatrixCell(**raw)
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f'Population cell {key!r}: {error}') from error
        fields['cells'] = cells
        if fields.get("stage") is not None:
            stage = dict(fields["stage"])
            stage["validation_scores"] = [
                tuple(item) for item in stage.get("validation_scores", [])
            ]
            fields["stage"] = StageTracker(**stage)
        state = cls(**fields)
        if (
            state.active_camp not in CAMPS
            or state.phase_status not in {"training", "draining", "integrating"}
            or state.round_index not in range(5)
        ):
            raise ValueError("Invalid population phase state")
        return state


def validate_population_evaluation_matches(value: Any, maximum: int = 16640) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or int(value) < 0
        or int(value) > maximum
        or int(value) % 256
    ):
        raise ValueError("Population evaluation counter is invalid")
    return int(value)


def validate_population_settings(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "protocol": "team_response_v1",
        "condition": None,
        "batch_identity": "issue-27-team-ablation-seed42",
        "rounds": 4,
        "pool_cap": 5,
        "matrix_matches_per_cell": 256,
        "validation_matches": 256,
        "validation_interval_updates": 100,
        "minimum_phase_updates": 200,
        "maximum_phase_updates": 400,
        "significant_improvement": 0.02,
    }
    unknown = set(value) - set(expected)
    if unknown:
        raise ValueError("Unknown population settings: " + ", ".join(sorted(unknown)))
    result = {key: value.get(key, default) for key, default in expected.items()}
    if result["protocol"] != "team_response_v1" or result["condition"] not in {"A", "B", "C", "D", None}:
        raise ValueError("Invalid population protocol or condition")
    if "condition" not in value:
        raise ValueError("Population condition must be explicit")
    custom = result["condition"] is None
    if custom and (not isinstance(value.get("batch_identity"), str) or not value["batch_identity"].strip() or value["batch_identity"] == expected["batch_identity"]):
        raise ValueError("Custom population requires its own explicit batch_identity")
    fixed = {key: expected[key] for key in expected if key not in ({"condition", "batch_identity"} if custom else {"condition"})}
    if any(result[key] != fixed[key] for key in fixed):
        raise ValueError("Population budgets and cadence are fixed by protocol")
    return result
