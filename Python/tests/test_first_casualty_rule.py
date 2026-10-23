from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from auv_madrl.actor_training import checkpoint_training_series_signature
from auv_madrl.goal_reach import (
    GOAL_REACH,
    VIEWER_ENVIRONMENT_SETTING_KEYS,
    GoalReachViewerSession,
    _build_config,
    _checkpoint_viewer_settings,
    _load_viewer_settings,
    _store_viewer_settings,
    _training_settings_identity,
    _validated_viewer_settings,
    _viewer_settings_defaults,
    _apply_persisted_randomization,
    _unified_match_architecture_id,
    _build_training_group,
    _training_group_checkpoint,
)
from auv_madrl.match_metrics import (
    MATCH_END_REASON_CODES,
    METRIC_REGISTRY,
    MatchEndReason,
    MatchOutcome,
    match_registry_payload,
)
from auv_madrl.multi_goal_reach_env import MultiGoalReachEnv


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _environment(
    *,
    terminate_on_any_casualty: bool,
    learner_count_range: tuple[int, int] = (2, 2),
    defender_count_range: tuple[int, int] = (2, 2),
    target_mode: str = "static",
    reward_objective: str = "outcome",
) -> MultiGoalReachEnv:
    settings = _viewer_settings_defaults()
    settings.update(target_mode=target_mode, target_count_range=[1, 1],
                    learner_count_range=list(learner_count_range),
                    defender_count_range=list(defender_count_range),
                    reward_objective=reward_objective)
    config = _build_config()
    _apply_persisted_randomization(config, settings, _unified_match_architecture_id(settings, "learner"))
    config.goal_reach["safety_zones"]["enabled"] = False
    return MultiGoalReachEnv(
        config,
        1,
        "cpu",
        base_seed=int(GOAL_REACH["seed"]),
        training_target="defender",
        learner_architecture_id=_unified_match_architecture_id(settings, "learner"),
        patrol_architecture_id=_unified_match_architecture_id(settings, "defender"),
        learner_count_range=learner_count_range,
        defender_count_range=defender_count_range,
        terminate_on_any_casualty=terminate_on_any_casualty,
        reward_objective=reward_objective,
    )


def _set_open_scene(
    env: MultiGoalReachEnv,
    *,
    learner_positions: torch.Tensor,
    defender_positions: torch.Tensor,
    learner_members: torch.Tensor | None = None,
    defender_members: torch.Tensor | None = None,
) -> None:
    env.reset()
    env.obstacle_active.zero_()
    env.terrain_manual_override.fill_(True)
    target_positions = torch.tensor([[[0., -10.]]], device=env.device)
    env.target_active.fill_(True)
    env.target_credit_member.fill_(True)
    env.sampled_target_count.fill_(1)
    env.learner_dynamics.reset(env.targets, target_positions, torch.zeros_like(env.targets.heading), env.target_active)
    learner_members = (
        torch.ones_like(env.learner_active)
        if learner_members is None
        else learner_members.to(dtype=torch.bool)
    )
    defender_members = (
        torch.ones_like(env.defender_active)
        if defender_members is None
        else defender_members.to(dtype=torch.bool)
    )
    env.learner_dynamics.reset(
        env.learners,
        learner_positions,
        torch.zeros(learner_positions.shape[:-1]),
        learner_members,
    )
    env.defender_dynamics.reset(
        env.defenders,
        defender_positions,
        torch.zeros(defender_positions.shape[:-1]),
        defender_members,
    )
    env._bind_primary_learner()
    env.learner_active.copy_(learner_members)
    env.learner_credit_member.copy_(learner_members)
    env.defender_active.copy_(defender_members)
    env.defender_credit_member.copy_(defender_members)
    env.sampled_learner_count.copy_(learner_members.sum(dim=1))
    env.sampled_defender_count.copy_(defender_members.sum(dim=1))
    env.learner_status.zero_()
    env.physics_ticks.zero_()
    env.done.zero_()
    env.success.zero_()
    env.collision.zero_()
    env.defender_success.zero_()
    env.defender_collision.zero_()
    env.defender_pair_collision.zero_()
    env.match_outcome.fill_(int(MatchOutcome.IN_PROGRESS))
    env.match_end_reason.fill_(MATCH_END_REASON_CODES[MatchEndReason.NONE])


def _zero_actions(env: MultiGoalReachEnv) -> tuple[torch.Tensor, torch.Tensor]:
    learner_action = torch.zeros(
        (env.batch_size, env.viewer_learner_capacity, 2), dtype=torch.float32
    )
    defender_action = torch.zeros((env.batch_size, env.defender_count, 2), dtype=torch.float32)
    learner_action[..., 0] = -1.0
    defender_action[..., 0] = -1.0
    return learner_action, defender_action


@pytest.mark.parametrize("casualty_side", ("attacker", "defender"))
def test_disabled_rule_preserves_single_casualty_behavior(casualty_side: str) -> None:
    env = _environment(terminate_on_any_casualty=False)
    _set_open_scene(
        env,
        learner_positions=torch.tensor([[[0.0, 0.0], [10.0, 0.0]]]),
        defender_positions=torch.tensor([[[0.0, 0.0], [-10.0, 0.0]]]),
    )
    if casualty_side == "defender":
        env.learners.position[0, 0] = torch.tensor([10.0, 10.0])
        env.obstacle_active[0, 0] = True
        env.obstacle_centers[0, 0] = env.defenders.position[0, 0]
        env.obstacle_radii[0, 0] = 1.0
    learner_action, defender_action = _zero_actions(env)

    step = env.training_step_joint(learner_action, defender_action)

    assert not bool(env.done[0])
    assert env.match_outcome.tolist() == [int(MatchOutcome.IN_PROGRESS)]
    assert not bool(step.learner.terminated.any())
    assert not bool(step.defender.terminated.any())
    if casualty_side == "attacker":
        assert env.learner_active.tolist() == [[False, True]]
    else:
        assert env.defender_active.tolist() == [[False, True]]


@pytest.mark.parametrize("attacker_cause", ("capture", "static_collision", "vessel_collision"))
def test_enabled_rule_ends_on_attacker_casualty(attacker_cause: str) -> None:
    env = _environment(terminate_on_any_casualty=True)
    _set_open_scene(
        env,
        learner_positions=torch.tensor([[[0.0, 0.0], [10.0, 0.0]]]),
        defender_positions=torch.tensor([[[0.0, 0.0], [-10.0, 0.0]]]),
    )
    if attacker_cause == "static_collision":
        env.defenders.position[0, 0] = torch.tensor([-10.0, 10.0])
        env.obstacle_active[0, 0] = True
        env.obstacle_centers[0, 0] = env.learners.position[0, 0]
        env.obstacle_radii[0, 0] = 1.0
    elif attacker_cause == "vessel_collision":
        env.defenders.position[0, 0] = torch.tensor([-10.0, 10.0])
        env.learners.position[0, 1] = env.learners.position[0, 0]
    learner_action, defender_action = _zero_actions(env)
    episode_counters = env.episode_counters.clone()

    step = env.training_step_joint(learner_action, defender_action)

    assert env.match_outcome.tolist() == [int(MatchOutcome.DEFENDER_WIN)]
    assert env.match_end_reason.tolist() == [
        MATCH_END_REASON_CODES[MatchEndReason.ATTACKER_CASUALTY]
    ]
    assert env.physics_ticks.tolist() == [1]
    assert step.learner.reward.tolist() == [[-1.0, -1.0]]
    assert step.defender.reward.tolist() == [[1.0, 1.0]]
    assert not bool(step.learner.info["success"].any())
    assert bool(step.defender.info["success"].all())
    assert bool(step.learner.terminated.all())
    assert bool(step.defender.terminated.all())
    assert torch.equal(env.episode_counters, episode_counters)


def test_enabled_rule_ends_on_defender_collision() -> None:
    env = _environment(terminate_on_any_casualty=True)
    _set_open_scene(
        env,
        learner_positions=torch.tensor([[[10.0, 0.0], [-10.0, 0.0]]]),
        defender_positions=torch.tensor([[[0.0, 0.0], [0.0, 10.0]]]),
    )
    env.obstacle_active[0, 0] = True
    env.obstacle_centers[0, 0] = env.defenders.position[0, 0]
    env.obstacle_radii[0, 0] = 1.0
    learner_action, defender_action = _zero_actions(env)

    step = env.training_step_joint(learner_action, defender_action)

    assert env.match_outcome.tolist() == [int(MatchOutcome.ATTACKER_WIN)]
    assert env.match_end_reason.tolist() == [
        MATCH_END_REASON_CODES[MatchEndReason.DEFENDER_CASUALTY]
    ]
    assert env.physics_ticks.tolist() == [1]
    assert step.learner.reward.tolist() == [[1.0, 1.0]]
    assert step.defender.reward.tolist() == [[-1.0, -1.0]]
    assert bool(step.learner.info["success"].all())
    assert not bool(step.defender.info["success"].any())
    assert bool(step.learner.terminated.all())
    assert bool(step.defender.terminated.all())


def test_first_tick_casualty_freezes_remaining_physics_substeps() -> None:
    long_step_env = _environment(terminate_on_any_casualty=True)
    one_tick_env = _environment(terminate_on_any_casualty=True)
    learner_positions = torch.tensor([[[0.0, 0.0], [10.0, 0.0]]])
    defender_positions = torch.tensor([[[0.0, 0.0], [-10.0, 0.0]]])
    for env in (long_step_env, one_tick_env):
        _set_open_scene(
            env,
            learner_positions=learner_positions,
            defender_positions=defender_positions,
        )
    learner_action, defender_action = _zero_actions(long_step_env)
    learner_action[0, 1, 0] = 1.0
    one_tick_learner_action = learner_action.clone()
    one_tick_defender_action = defender_action.clone()

    long_step_env._step_joint_match(
        learner_action=learner_action,
        defender_action=defender_action,
        physics_ticks=25,
        return_both=True,
    )
    one_tick_env._step_joint_match(
        learner_action=one_tick_learner_action,
        defender_action=one_tick_defender_action,
        physics_ticks=1,
        return_both=True,
    )

    assert long_step_env.physics_ticks.tolist() == [1]
    for long_state, one_tick_state in (
        (long_step_env.learners, one_tick_env.learners),
        (long_step_env.defenders, one_tick_env.defenders),
    ):
        for name, value in long_state.__dict__.items():
            assert torch.equal(value, getattr(one_tick_state, name))


def test_target_capture_precedes_same_tick_other_attacker_casualty() -> None:
    env = _environment(terminate_on_any_casualty=True)
    _set_open_scene(
        env,
        learner_positions=torch.tensor([[[0.0, 0.0], [10.0, 0.0]]]),
        defender_positions=torch.tensor([[[10.0, 0.0], [-10.0, 0.0]]]),
    )
    env.targets.position[0, 0] = env.learners.position[0, 0]
    learner_action, defender_action = _zero_actions(env)

    step = env.training_step_joint(learner_action, defender_action)

    assert env.match_outcome.tolist() == [int(MatchOutcome.ATTACKER_WIN)]
    assert env.match_end_reason.tolist() == [MATCH_END_REASON_CODES[MatchEndReason.TARGET_CAPTURE]]
    assert step.learner.reward.tolist() == [[1.0, 1.0]]
    assert step.defender.reward.tolist() == [[-1.0, -1.0]]


def test_simultaneous_team_casualties_are_defender_win() -> None:
    env = _environment(terminate_on_any_casualty=True)
    _set_open_scene(
        env,
        learner_positions=torch.tensor([[[10.0, 0.0], [-10.0, 0.0]]]),
        defender_positions=torch.tensor([[[0.0, 0.0], [10.0, 0.0]]]),
    )
    env.obstacle_active[0, 0] = True
    env.obstacle_centers[0, 0] = env.defenders.position[0, 0]
    env.obstacle_radii[0, 0] = 1.0
    learner_action, defender_action = _zero_actions(env)

    step = env.training_step_joint(learner_action, defender_action)

    assert env.match_outcome.tolist() == [int(MatchOutcome.DEFENDER_WIN)]
    assert env.match_end_reason.tolist() == [
        MATCH_END_REASON_CODES[MatchEndReason.ATTACKER_CASUALTY]
    ]
    assert step.learner.reward.tolist() == [[-1.0, -1.0]]
    assert step.defender.reward.tolist() == [[1.0, 1.0]]


def test_inactive_capacity_slots_do_not_trigger_casualty_terminal() -> None:
    env = _environment(
        terminate_on_any_casualty=True,
        learner_count_range=(1, 2),
        defender_count_range=(1, 2),
    )
    _set_open_scene(
        env,
        learner_positions=torch.tensor([[[10.0, 0.0], [0.0, 0.0]]]),
        defender_positions=torch.tensor([[[-10.0, 0.0], [0.0, 0.0]]]),
        learner_members=torch.tensor([[True, False]]),
        defender_members=torch.tensor([[True, False]]),
    )
    learner_action, defender_action = _zero_actions(env)

    step = env.training_step_joint(learner_action, defender_action)

    assert not bool(env.done[0])
    assert env.match_outcome.tolist() == [int(MatchOutcome.IN_PROGRESS)]
    assert not bool(step.learner.terminated.any())
    assert not bool(step.defender.terminated.any())


def obsolete_setting_migrates_persists_and_restores_from_checkpoint(tmp_path) -> None:
    config = replace(_build_config(), repository_root=tmp_path)
    settings_path = tmp_path / "Artifacts" / "GoalReach" / "viewer-settings.json"
    settings_path.parent.mkdir(parents=True)
    legacy = _viewer_settings_defaults()
    legacy["match_rules_version"] = 2
    legacy.pop("terminate_on_any_casualty")
    settings_path.write_text(json.dumps(legacy), encoding="utf-8")

    path, migrated = _load_viewer_settings(config)

    assert path == settings_path
    assert migrated["match_rules_version"] == 6
    assert migrated["terminate_on_any_casualty"] is False
    assert (
        json.loads(settings_path.read_text(encoding="utf-8"))["terminate_on_any_casualty"] is False
    )

    migrated["terminate_on_any_casualty"] = True
    persisted = _validated_viewer_settings(migrated)
    _store_viewer_settings(settings_path, persisted)
    _, reloaded = _load_viewer_settings(config)
    assert reloaded["terminate_on_any_casualty"] is True

    architecture = _unified_match_architecture_id(reloaded, "learner")
    reloaded['training_settings']['parallel_environments'] = 1
    reloaded['obstacle_density_range_per_1000m2'] = [0, 0]
    env, groups, runtimes, bindings, frozen = _build_training_group(reloaded, device='cpu')
    payload = _training_group_checkpoint(reloaded, 42, 'fixture', 0, env, groups, runtimes, bindings, frozen)
    restored = _checkpoint_viewer_settings(payload, architecture)
    assert restored["terminate_on_any_casualty"] is True

    # Current checkpoint contract explicitly rejects pre-unified rules.
    with pytest.raises(ValueError, match='Old checkpoint'):
        _checkpoint_viewer_settings({**payload, 'match_rules_version': 2}, architecture)


def test_setting_changes_training_identity_and_rebuilds_environment() -> None:
    disabled = _viewer_settings_defaults()
    disabled["terminate_on_any_casualty"] = False
    enabled = _viewer_settings_defaults()
    enabled["terminate_on_any_casualty"] = True

    architecture = _unified_match_architecture_id(enabled, "learner")
    disabled_identity = _training_settings_identity(disabled, architecture)
    enabled_identity = _training_settings_identity(enabled, architecture)

    assert disabled_identity["terminate_on_any_casualty"] is False
    assert enabled_identity["terminate_on_any_casualty"] is True
    assert disabled_identity != enabled_identity
    legacy_identity = dict(disabled_identity)
    legacy_identity.pop("terminate_on_any_casualty")
    assert checkpoint_training_series_signature(
        {"training_settings": legacy_identity}
    ) != checkpoint_training_series_signature({"training_settings": disabled_identity})
    assert checkpoint_training_series_signature(
        {"training_settings": disabled_identity}
    ) != checkpoint_training_series_signature({"training_settings": enabled_identity})
    assert "terminate_on_any_casualty" in VIEWER_ENVIRONMENT_SETTING_KEYS

    session = GoalReachViewerSession.__new__(GoalReachViewerSession)
    session.viewer_settings = enabled
    session.active_actor_kind = "learner"
    _, environment_type, arguments = session._environment_specification(
        architecture,
        settings=enabled,
    )
    assert environment_type is MultiGoalReachEnv
    assert arguments["terminate_on_any_casualty"] is True


def test_setting_validation_requires_boolean() -> None:
    settings = _viewer_settings_defaults()
    settings["terminate_on_any_casualty"] = "true"

    with pytest.raises(ValueError, match="must be boolean"):
        _validated_viewer_settings(settings)


def test_casualty_end_reasons_are_distinct_and_registered() -> None:
    reasons = {item["id"]: item for item in match_registry_payload()["end_reasons"]}

    assert reasons[MatchEndReason.ATTACKER_CASUALTY.value]["winner"] == "defender"
    assert reasons[MatchEndReason.DEFENDER_CASUALTY.value]["winner"] == "attacker"
    assert (
        MATCH_END_REASON_CODES[MatchEndReason.DEFENDER_CASUALTY]
        != (MATCH_END_REASON_CODES[MatchEndReason.BREACH])
    )
    assert "match/attacker_casualty_rate" in METRIC_REGISTRY
    assert "match/defender_casualty_rate" in METRIC_REGISTRY


@pytest.mark.parametrize("cause", ["capture_target", "target_obstacle", "friendly_collision", "all_attackers", "capture_attacker", "defender_obstacle", "timeout", "same_tick"])
@pytest.mark.parametrize("first_casualty", [False, True])
@pytest.mark.parametrize("reward_objective", ["outcome", "protection_time"])
def test_dynamic_target_terminal_events(cause, first_casualty, reward_objective, tmp_path):
    env = _environment(terminate_on_any_casualty=first_casualty, target_mode="dynamic", reward_objective=reward_objective)
    config = env.config
    _set_open_scene(env, learner_positions=torch.tensor([[[10.,0.],[10.,8.]]]),
                    defender_positions=torch.tensor([[[-10.,0.],[-10.,8.]]]))
    for role, positions in (("learner",[[10.,0.],[10.,8.]]),("defender",[[-10.,0.],[-10.,8.]]),("target",[[0.,0.]])):
        state=env.role_state(role)
        env.learner_dynamics.reset(state,torch.tensor([positions],device=env.device),torch.zeros_like(state.heading),getattr(env,role+"_active"))
    env._bind_primary_learner()
    if cause in {"capture_target", "same_tick"}:
        env.learners.position[0,0]=env.targets.position[0,0]
    if cause in {"target_obstacle", "defender_obstacle", "same_tick"}:
        env.obstacle_active[0,0]=True
        env.obstacle_centers[0,0]=env.targets.position[0,0] if cause=="target_obstacle" else env.defenders.position[0,0]
        env.obstacle_radii[0,0]=1.
    if cause=="capture_attacker":
        env.defenders.position[0,0]=env.learners.position[0,0]
    if cause=="target_boundary":
        env.targets.position[0,0,0]=env.world.radius
    if cause=="friendly_collision":
        env.defenders.position[0,0]=env.targets.position[0,0]
    if cause=="all_attackers":
        env.defenders.position.copy_(env.learners.position)
    if cause=="timeout":
        env.physics_ticks.fill_(config.goal_reach["clock"]["maximum_physics_ticks"]-1)
    actions={role:torch.zeros((*getattr(env,role+"_active").shape,2),device=env.device) for role in env.roles}
    for action in actions.values(): action[...,0]=-1.
    step=env.training_step_joint(actions["learner"],actions["defender"],actions["target"])
    attacker_win=cause in {"capture_target","same_tick","target_obstacle","target_boundary","friendly_collision"} or (cause=="defender_obstacle" and first_casualty)
    defender_win=cause in {"timeout", "all_attackers"} or (cause=="capture_attacker" and first_casualty)
    expected=int(MatchOutcome.ATTACKER_WIN if attacker_win else MatchOutcome.DEFENDER_WIN if defender_win else MatchOutcome.IN_PROGRESS)
    assert env.match_outcome.item()==expected
    assert bool(env.done.item())==(attacker_win or defender_win)
    for role in env.roles:
        result=getattr(step,role)
        sign=1 if (role=="learner")==attacker_win else -1
        if reward_objective == "outcome":
            assert torch.equal(result.reward,torch.full_like(result.reward,sign if attacker_win or defender_win else 0))
        else:
            defense_reward = step.defender.reward[:, :1]
            assert torch.isfinite(result.reward).all() and (defense_reward >= 0).all()
            assert torch.equal(result.reward, (defense_reward * (-1 if role == "learner" else 1)).expand_as(result.reward))
        assert bool(result.terminated.all())==(attacker_win or defender_win)
    if attacker_win or defender_win:
        before = env.physics_ticks.clone()
        repeated = env.training_step_joint(actions["learner"],actions["defender"],actions["target"])
        assert torch.equal(env.physics_ticks, before)
        assert all(not getattr(repeated, role).reward.any() for role in env.roles)
    if cause in {"target_obstacle", "target_boundary", "friendly_collision"}:
        assert env.match_end_reason.item() == MATCH_END_REASON_CODES[MatchEndReason.TARGET_CASUALTY]
        assert env.target_collision.all() and not env.target_captured.any()
    if cause=="capture_target":
        assert env.learner_active.all()  # Normal capture does not kill the capturer.
    if cause=="defender_obstacle" and not first_casualty:
        env.defender_active.zero_()
        env.training_step_joint(actions["learner"],actions["defender"],actions["target"])
        assert not env.done.any()


def test_dynamic_motion_relative_sweep_and_complete_tick_slicing(tmp_path):
    import copy
    env = _environment(terminate_on_any_casualty=True, target_mode="dynamic")
    _set_open_scene(env, learner_positions=torch.tensor([[[10.,0.],[10.,8.]]]),
                    defender_positions=torch.tensor([[[-10.,0.],[-10.,8.]]]))
    initial=copy.deepcopy(env.state_dict())
    actions={role:torch.tensor([1.,.3],device=env.device).expand(*getattr(env,role+"_active").shape,2) for role in env.roles}
    env.training_step_joint(actions["learner"],actions["defender"],actions["target"])
    full=env.state_dict()
    assert not torch.equal(full["targets"]["position"],initial["targets"]["position"])
    env.load_state_dict(initial)
    for tick in range(25):
        env._step_joint_match(learner_action=actions["learner"],defender_action=actions["defender"],
            target_action=actions["target"],physics_ticks=25-tick,
            physics_deadline=0.,finish_decision=tick==24)
    def equal(left,right):
        if isinstance(left,torch.Tensor): assert torch.equal(left,right)
        elif isinstance(left,dict):
            assert left.keys() == right.keys()
            for key in left: equal(left[key],right[key])
        elif isinstance(left,(list,tuple)):
            assert len(left) == len(right)
            for a,b in zip(left,right): equal(a,b)
        else: assert left == right
    equal(full,env.state_dict())
    previous=torch.tensor([[[-2.,0.],[2.,0.]]],device=env.device)
    desired=previous.flip(1)
    contact=env._actor_pair_contact_matrix(previous,desired,torch.ones((1,2),dtype=torch.bool,device=env.device))
    assert contact[0,0,1] and contact[0,1,0] and not contact[0,0,0]
    assert not env._actor_pair_contact_matrix(previous,desired,torch.tensor([[True,False]],device=env.device)).any()

    # Controlled relative-boundary cases; no resampling or dynamics uncertainty.
    from auv_madrl.dynamics import AuvState
    from auv_madrl.multi_goal_reach_env import CONTACT_TOLERANCE
    env.obstacle_active.zero_()
    env.config.goal_reach["safety_zones"]["enabled"] = True
    radius = env.config.goal_reach["safety_zones"]["engagement_radius_m"]
    gap_radius = radius - env.vessel_radius_m
    for mode, defender_dx, target_dx, contact in (
        ("dynamic", 0., 0., False), ("dynamic", 1., 0., True),
        ("dynamic", 0., -1., True), ("dynamic", .5, .5, False),
        ("dynamic", 0., 1., False), ("static", 0., -1., False),
    ):
        env.target_mode = mode
        env.region_reference_position.zero_()
        env.contact_static.zero_(); env.contact_pairs.zero_()
        state = AuvState(**{name: torch.cat([getattr(env.role_state(role), name) for role in ("defender", "learner", "target")],1).clone()
                            for name in env.defenders.__dict__})
        previous = torch.tensor([[[gap_radius-.5,0.], [0.,-12.], [0.,15.], [0.,-16.], [0.,0.]]],device=env.device)
        state.position = previous.clone()
        state.position[0,0,0] += defender_dx
        if mode == "dynamic": state.position[0,4,0] += target_dx
        desired = state.position.clone()
        active = torch.tensor([[True,False,True,False,True]],device=env.device)
        static, pairs, captured, reached, *_ = env._resolve_contacts(state,previous,active)
        assert bool(static[0,0,-1]) == contact, (mode, defender_dx, target_dx, radius, previous, state.position)
        assert not pairs.any() and not captured.any() and not reached.any()
        if contact:
            actual = (state.position[0,0]-state.position[0,4]).norm()+env.vessel_radius_m
            assert abs(actual.item()-radius) <= CONTACT_TOLERANCE
            assert torch.allclose(state.position,previous+.5*(desired-previous),atol=CONTACT_TOLERANCE)
        else:
            assert torch.equal(state.position, desired)
