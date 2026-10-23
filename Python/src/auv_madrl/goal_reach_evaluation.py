from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .actor_variants import (
    actor_variant,
)
from .goal_reach import (
    GOAL_REACH,
    _apply_persisted_randomization,
    _build_config,
    _apply_training_schedule,
    _validate_group_checkpoint,
    _training_settings_identity,
)
from .match_metrics import MATCH_END_REASON_CODES, MatchOutcome
from .models import ActorCritic, actor_deployment_state_dict
from .terrain import load_periodic_maps, world_parameters
from .multi_goal_reach_env import MultiGoalReachEnv
from .population import EVALUATION_END_REASONS, ImmutablePolicyRef, actor_state_hash, payoff_utility, validate_evaluation_summary


@contextmanager
def _preserve_evaluation_process_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_initialized = torch.cuda.is_initialized()
    cuda_state = torch.cuda.get_rng_state_all() if cuda_initialized else None
    goal_reach_state = copy.deepcopy(GOAL_REACH)
    deterministic = torch.are_deterministic_algorithms_enabled()
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    precision = torch.get_float32_matmul_precision()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_initialized and cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        GOAL_REACH.clear()
        GOAL_REACH.update(goal_reach_state)
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.set_float32_matmul_precision(precision)


def _resolved_population_device(device: torch.device | str) -> torch.device:
    requested = torch.device(device)
    if requested.type == "cpu":
        return torch.device("cpu")
    if requested.type == "cuda":
        index = requested.index
        if index is None:
            index = torch.cuda.current_device()
        return torch.device("cuda", index)
    return requested


def _population_actor(
    ref: ImmutablePolicyRef,
    architecture_id: str,
    device: torch.device,
    cache: dict[tuple[str, str, str, str, str], tuple[tuple[int, int], ActorCritic]],
    settings=None,
) -> ActorCritic:
    checkpoint = Path(ref.checkpoint).resolve()
    if checkpoint.name == "checkpoint-latest.pt" or not checkpoint.is_file():
        raise ValueError(f"Population source is not an immutable checkpoint: {checkpoint}")
    resolved_device = _resolved_population_device(device)
    device_identity = str(resolved_device)
    key = (
        str(checkpoint), ref.parameter_group, ref.actor_hash,
        architecture_id, json.dumps(evaluation_game_contract(settings), sort_keys=True) if settings is not None else "", device_identity,
    )
    stat = checkpoint.stat()
    signature = (stat.st_mtime_ns, stat.st_size)
    if key in cache:
        cached_signature, model = cache[key]
        if cached_signature != signature:
            raise ValueError("Population Actor checkpoint changed after caching")
        return model
    payload = _validate_group_checkpoint(
        torch.load(checkpoint, map_location="cpu", weights_only=False)
    )
    if settings is not None and evaluation_game_contract(payload["launch_settings"]) != evaluation_game_contract(settings):
        raise ValueError("Population Actor task/observation contract changed")
    if payload["architecture_id"] != architecture_id:
        raise ValueError("Population Actor architecture changed")
    group = payload["parameter_groups"].get(ref.parameter_group)
    state = group["model"] if group is not None else payload["frozen_actors"].get(ref.parameter_group)
    if state is None or actor_state_hash(state) != ref.actor_hash:
        raise ValueError("Population Actor hash changed")
    with _preserve_evaluation_process_state():
        model = ActorCritic(architecture_id, build_critic=False).to(resolved_device)
        model.load_state_dict(actor_deployment_state_dict(model, state), strict=True)
        model.requires_grad_(False).eval()
    after = checkpoint.stat()
    if (after.st_mtime_ns, after.st_size) != signature:
        raise ValueError("Population Actor checkpoint changed while loading")
    cache[key] = (signature, model)
    return model


def _indexed_population_actions(
    observations: dict[str, dict[str, torch.Tensor]],
    models_by_role: dict[str, tuple[ActorCritic, ...]],
    assignments_by_role: dict[str, torch.Tensor],
    unfinished: torch.Tensor,
    *,
    sample_seed: int,
    decision_index: int,
) -> dict[str, torch.Tensor]:
    requests: dict[int, dict[str, object]] = {}
    outputs = {}
    role_codes = {"learner": 11, "defender": 23, "target": 37}
    for role, observation in observations.items():
        shape = observation["actor_alive"].shape
        alive = observation["actor_alive"].reshape(-1).to(torch.bool)
        lane = torch.arange(shape[0], device=alive.device).reshape(
            shape[0], *([1] * (len(shape) - 1))
        ).expand(shape).reshape(-1)
        alive &= unfinished[lane]
        outputs[role] = torch.zeros((*shape, 2), device=alive.device)
        generator = torch.Generator(device=alive.device)
        generator.manual_seed(
            int(sample_seed) + decision_index * 1_000_003 + role_codes[role] * 10_007
        )
        full_noise = torch.randn((*shape, 2), generator=generator, device=alive.device).reshape(-1, 2)
        keys = models_by_role[role][0].variant.observation_keys
        flat_observation = {
            key: value.reshape(math.prod(shape), *value.shape[len(shape) :])
            for key, value in observation.items()
            if key in keys
        }
        for model_index, model in enumerate(models_by_role[role]):
            indices = (
                alive & (assignments_by_role[role][lane] == model_index)
            ).nonzero(as_tuple=True)[0]
            request = requests.setdefault(id(model), {"model": model, "items": []})
            request["items"].append((role, indices, flat_observation, full_noise))
    for request in requests.values():
        model = request["model"]
        items = request["items"]
        count = sum(indices.numel() for _, indices, _, _ in items)
        if not count:
            continue
        keys = model.variant.observation_keys
        actor_observation = {
            key: torch.cat([flat[key][indices] for _, indices, flat, _ in items])
            for key in keys
        }
        noise = torch.cat([full_noise[indices] for _, indices, _, full_noise in items])
        distribution = model.actor_distribution_from_observation(actor_observation)
        actions = distribution.from_standard_noise(noise)
        offset = 0
        for role, indices, _, _ in items:
            outputs[role].reshape(-1, 2)[indices] = actions[offset : offset + indices.numel()]
            offset += indices.numel()
    return outputs


def evaluation_game_contract(settings):
    contract = _training_settings_identity(settings, settings["active_actor_variant_id"])
    # Optimizer and observer settings do not define the fixed-policy game.
    for key in ("ppo_update", "parallel_environments", "rollout_decisions", "minibatch_agent_frames",
                "joint_training_mode", "alternating_updates", "training_side_modes", "entropy_coefficient", "gae_lambda"):
        contract.pop(key, None)
    contract.pop("population", None)
    contract.pop("policy_sharing", None)
    contract["map_asset_hash"] = load_periodic_maps(parameters=world_parameters(settings)).asset_hash
    variant = actor_variant(settings["active_actor_variant_id"])
    contract["actor_features"] = list(variant.vector_inputs)
    contract["critic_features"] = list(variant.critic_vector_input_names)
    return contract


def evaluation_sample_id(settings, attack_refs, defense_refs, p, q, sample_seed, sample_count, evaluation_decisions):
    contract = evaluation_game_contract(settings)
    return hashlib.sha256(json.dumps(dict(seed=sample_seed, count=sample_count,
        attack=[ref.actor_hash for ref in attack_refs], defense=[ref.actor_hash for ref in defense_refs],
        p=list(p), q=list(q), contract=contract, evaluation_decisions=evaluation_decisions,
        statistics_version=2), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@torch.no_grad()
def _evaluate_population_match_impl(
    settings: dict[str, object],
    attack_refs: list[ImmutablePolicyRef],
    defense_refs: list[ImmutablePolicyRef],
    attack_probabilities: list[float],
    defense_probabilities: list[float],
    *,
    sample_count: int,
    sample_seed: int,
    device: str | torch.device = "cuda",
    evaluation_decisions: int | None = None,
) -> dict[str, object]:
    """Evaluate exactly one fixed sample set without touching the training environment or RNG."""
    settings = copy.deepcopy(settings)
    if sample_count < 1 or not attack_refs or not defense_refs:
        raise ValueError("Population evaluation requires policies and samples")
    if len(attack_refs) != len(attack_probabilities) or len(defense_refs) != len(
        defense_probabilities
    ):
        raise ValueError("Population evaluation mixture axes differ")
    device = torch.device(device)
    architecture_id = actor_variant(
        settings["active_actor_variant_id"]
    ).architecture_id
    _apply_training_schedule(settings, architecture_id)
    config = _build_config()
    _apply_persisted_randomization(config, settings, architecture_id)
    env = MultiGoalReachEnv(
        config,
        sample_count,
        device,
        base_seed=int(sample_seed),
        training_target="learner",
        learner_architecture_id=architecture_id,
        patrol_architecture_id=architecture_id,
        learner_count_range=settings["learner_count_range"],
        defender_count_range=settings["defender_count_range"],
        learner_speed_ratio_range=settings["learner_speed_ratio_range"],
        terminate_on_any_casualty=settings["terminate_on_any_casualty"],
        reward_objective=settings["reward_objective"],
        terminate_on_defender_elimination=settings.get("terminate_on_defender_elimination", False),
        apply_sampling_time_limit=False,
    )
    env.reset(build_observation=False)
    cache: dict[
        tuple[str, str, str, str, str], tuple[tuple[int, int], ActorCritic]
    ] = {}
    attack_models = tuple(_population_actor(ref, architecture_id, device, cache, settings) for ref in attack_refs)
    defense_models = tuple(_population_actor(ref, architecture_id, device, cache, settings) for ref in defense_refs)
    assignment_generator = torch.Generator(device=device)
    assignment_generator.manual_seed(int(sample_seed) + 91_019)
    attack_assignment = torch.multinomial(
        torch.as_tensor(attack_probabilities, dtype=torch.float64, device=device),
        sample_count,
        replacement=True,
        generator=assignment_generator,
    )
    defense_assignment = torch.multinomial(
        torch.as_tensor(defense_probabilities, dtype=torch.float64, device=device),
        sample_count,
        replacement=True,
        generator=assignment_generator,
    )
    models = {"learner": attack_models, "defender": defense_models, "target": defense_models}
    assignments = {
        "learner": attack_assignment,
        "defender": defense_assignment,
        "target": defense_assignment,
    }
    unfinished = torch.ones(sample_count, dtype=torch.bool, device=device)
    attacker_won = torch.zeros(sample_count, dtype=torch.bool, device=device)
    terminal_ticks = torch.zeros(sample_count, dtype=torch.int64, device=device)
    protected_time = torch.zeros(sample_count, dtype=torch.float32, device=device)
    end_reason = torch.zeros(sample_count, dtype=torch.int64, device=device)
    maximum_decisions = int(settings["training_settings"]["maximum_episode_decisions"])
    maximum_decisions = maximum_decisions if evaluation_decisions is None else evaluation_decisions
    if type(maximum_decisions) is not int or maximum_decisions < 1:
        raise ValueError("evaluation_decisions must be a positive integer")
    observations = {role: env.training_observation_for(role) for role in env.policy_roles}
    for decision in range(maximum_decisions):
        actions = _indexed_population_actions(
            observations,
            models,
            assignments,
            unfinished,
            sample_seed=sample_seed,
            decision_index=decision,
        )
        joint = env.training_step_joint(
            actions["learner"], actions["defender"], target_action=actions.get("target"),
            build_observation=False,
        )
        done = joint.learner.info["match_done"].to(torch.bool)
        newly_done = unfinished & done
        outcome = joint.learner.info["match_outcome"]
        attacker_won[newly_done] = outcome[newly_done] == int(MatchOutcome.ATTACKER_WIN)
        terminal_ticks[newly_done] = env.physics_ticks[newly_done].to(torch.int64)
        protected_time[newly_done] = joint.learner.info["protected_time"][newly_done]
        end_reason[newly_done] = joint.learner.info["match_end_reason"][newly_done].to(
            torch.int64
        )
        unfinished &= ~newly_done
        if not bool(unfinished.any()):
            break
        observations = {role: env.training_observation_for(role) for role in env.policy_roles}
    if bool(unfinished.any()) and settings["reward_objective"] != "outcome":
        raise ValueError("Time objectives require complete finite-task evaluation")
    end_reason[unfinished] = MATCH_END_REASON_CODES["truncation"]
    clock = config.goal_reach["clock"]
    utility = payoff_utility(
        attacker_won,
        terminal_ticks,
        objective=str(settings["reward_objective"]),
        gamma=float(settings["training_settings"]["gamma"]),
        horizon_ticks=int(clock["maximum_physics_ticks"]),
        ticks_per_decision=int(clock["physics_ticks_per_decision"]),
        protected_time=protected_time,
        resolved=~unfinished,
        physics_step_s=float(config.shared["clock"]["physics_step_s"]),
    )
    reason_counts = {
        reason.value: int((end_reason == code).sum().item())
        for reason, code in MATCH_END_REASON_CODES.items()
        if reason.value in EVALUATION_END_REASONS
    }
    identity = evaluation_sample_id(settings, attack_refs, defense_refs, attack_probabilities, defense_probabilities,
                                    sample_seed, sample_count, maximum_decisions)
    result = {
        "sample_id": identity,
        "sample_count": sample_count,
        "utility_mean": float(utility.mean().item()),
        "utility_standard_error": float(
            utility.std(unbiased=True).item() / math.sqrt(sample_count)
        ) if sample_count > 1 else 0.0,
        "attacker_wins": int(attacker_won.sum().item()),
        "defender_wins": int((~attacker_won & ~unfinished).sum().item()),
        "unresolved": int(unfinished.sum().item()),
        "evaluation_decisions": maximum_decisions,
        "tail_error_bound": float(unfinished.sum().item()) / sample_count * float(settings["training_settings"]["gamma"]) ** maximum_decisions,
        "observed_ticks_sum": int(env.physics_ticks.sum().item()),
        "terminal_ticks": terminal_ticks.cpu(),
        "resolved": (~unfinished).cpu(),
        "terminal_outcomes": env.match_outcome.cpu(),
        "terminal_ticks_sum": int(terminal_ticks.sum().item()),
        "end_reasons": reason_counts,
        "attack_assignment": attack_assignment.cpu(),
        "defense_assignment": defense_assignment.cpu(),
    }
    validate_evaluation_summary(result, expected_samples=sample_count,
        horizon_ticks=int(clock['maximum_physics_ticks']), ticks_per_decision=int(clock['physics_ticks_per_decision']),
        objective=settings['reward_objective'], gamma=float(settings['training_settings']['gamma']),
        evaluation_decisions=maximum_decisions, expected_sample_id=identity, require_extended=True)
    return result


def evaluate_population_match(*args, **kwargs):
    with _preserve_evaluation_process_state():
        return _evaluate_population_match_impl(*args, **kwargs)
