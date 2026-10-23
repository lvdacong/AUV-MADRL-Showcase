from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum, IntEnum

from .actor_variants import ROLE_DESCRIPTORS

try:
    from enum import StrEnum
except ImportError:  # Python 3.10 Viewer runtime
    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return str(self.value)


class MatchOutcome(IntEnum):
    IN_PROGRESS = 0
    ATTACKER_WIN = 1
    DEFENDER_WIN = 2


class MatchEndReason(StrEnum):
    NONE = "none"
    BREACH = "breach"
    ELIMINATION = "elimination"
    HORIZON = "horizon"
    ATTACKER_CASUALTY = "attacker_casualty"
    TARGET_CAPTURE = "target_capture"
    DEFENDER_CASUALTY = "defender_casualty"
    TARGET_CASUALTY = "target_casualty"
    TRUNCATION = "truncation"
    DEFENDER_ELIMINATION = "defender_elimination"


class MatchEvent(StrEnum):
    REACH = "reach"
    CAPTURE = "capture"
    STATIC_COLLISION = "static_collision"
    VESSEL_COLLISION = "vessel_collision"


MATCH_EVENT_INFO_KEYS = (
    f"event_attacker_{MatchEvent.REACH.value}",
    f"event_attacker_{MatchEvent.CAPTURE.value}",
    f"event_attacker_{MatchEvent.STATIC_COLLISION.value}",
    f"event_attacker_{MatchEvent.VESSEL_COLLISION.value}",
    f"event_defender_{MatchEvent.STATIC_COLLISION.value}",
    f"event_defender_{MatchEvent.VESSEL_COLLISION.value}",
    "event_target_capture", "event_target_static_collision", "event_target_vessel_collision",
    *(f"event_{desc['event_role']}_{event}" for desc in ROLE_DESCRIPTORS.values()
      for event in ("obstacle", "distance_lost", "engagement_boundary")),
)


MATCH_END_REASON_CODES = {
    MatchEndReason.NONE: 0,
    MatchEndReason.BREACH: 1,
    MatchEndReason.ELIMINATION: 2,
    MatchEndReason.HORIZON: 3,
    MatchEndReason.ATTACKER_CASUALTY: 4,
    MatchEndReason.TARGET_CAPTURE: 5,
    MatchEndReason.DEFENDER_CASUALTY: 6,
    MatchEndReason.TARGET_CASUALTY: 7,
    MatchEndReason.TRUNCATION: 8,
    MatchEndReason.DEFENDER_ELIMINATION: 9,
}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    category: str
    category_label: str
    label: str
    unit: str
    aggregation: str
    tooltip: str
    denominator: str | None = None
    scopes: tuple[str, ...] = ("training", "tensorboard")
    training_series: str = "shared"
    scope: str = "session"
    role: str | None = None
    requires_first_casualty: bool = False
    target_modes: tuple[str, ...] = ("static", "dynamic")
    axis_min: float | None = None
    axis_max: float | str | None = None


METRIC_SPECS = (
    MetricSpec("train/effective_minibatch_agent_frames", "ppo", "Optimization", "Parameter group credit minibatch", "frames", "update_mean",
               "Credit minibatch of the owning parameter group; shared members report the same group batch.", scopes=("tensorboard",), scope="role", axis_min=0),
    *(MetricSpec(key, "outcome", "Match outcome", label, unit, "episode_mean", tip, "match/count",
                 target_modes=modes, axis_min=minimum, axis_max=maximum, requires_first_casualty=key in {"match/attacker_casualty_rate","match/defender_casualty_rate"})
      for key, label, unit, tip, modes, minimum, maximum in (
        ("match/attacker_win_rate", "Observed ATK wins", "%", "ATK wins / ended sampling episodes, including truncations; online policies are not fixed-policy evaluation.", ("static", "dynamic"), 0, 1),
        ("match/defender_win_rate", "Observed DEF wins", "%", "DEF wins / ended sampling episodes, including truncations; unresolved episodes are not wins.", ("static", "dynamic"), 0, 1),
        ("match/attacker_return", "Attack team return", "reward", "Negative defense utility; counted once per team.", ("static", "dynamic"), -1, 1),
        ("match/target_capture_rate", "TGT captured", "%", "Unique target captured by an active ATK.", ("static", "dynamic"), 0, 1),
        ("match/target_casualty_rate", "TGT collision loss", "%", "Target collision always ends the match.", ("static", "dynamic"), 0, 1),
        ("match/defender_elimination_rate", "ATK by DEF elimination", "%", "All DEF eliminated with the task rule enabled.", ("static", "dynamic"), 0, 1),
        ("match/elimination_rate", "Defense by elimination", "%", "All ATK eliminated by capture or collision.", ("static", "dynamic"), 0, 1),
        ("match/horizon_rate", "Defense by time limit", "%", "Defense wins at a terminal task horizon; excludes truncations.", ("static", "dynamic"), 0, 1),
        ("match/truncation_rate", "Truncated episodes", "%", "Reset at the sampling time limit without a winner; excluded from both win counts.", ("static", "dynamic"), 0, 1),
        ("match/attacker_casualty_rate", "Defense by first casualty", "%", "ATK death or capture ends the match when first casualty is enabled.", ("static", "dynamic"), 0, 1),
        ("match/defender_casualty_rate", "Defense camp casualty", "%", "DEF casualty ends the match only with First casualty enabled.", ("static", "dynamic"), 0, 1),
        ("match/episode_time_s", "Mean episode time", "s", "Executed integer physics ticks times physics_step_s.", ("static", "dynamic"), 0, "horizon"),
        ("match/protected_time", "Protected-time utility", "%", "Capture time / horizon; early defense wins receive the remaining time.", ("static", "dynamic"), 0, 1),
        ("match/defense_return", "Defense camp return", "reward", "One camp utility shared by DEF and TGT. ATK receives its negative.", ("static", "dynamic"), -1, 1),
      )),
    MetricSpec("match/count", "outcome", "Match outcome", "Ended sampling episodes", "episodes", "count", "True terminals plus technical truncations in this update.", axis_min=0),
    *(MetricSpec(f"event/{desc['event_role']}/{event}_per_episode", "behavior", "Role behavior",
                 f"{desc['label']} {label}", "events/episode", "episode_mean", tip, "match/count",
                 role=role, target_modes=("static", "dynamic"), axis_min=0)
      for role, desc in ROLE_DESCRIPTORS.items()
      for event, label, tip in (
        ("static_collision", "static contacts", "Lethal craft-object contact at the first swept contact position."),
        ("obstacle", "obstacle contacts", "Lethal obstacle contact participation, not a death probability."),
        ("distance_lost", "TGT distance loss", "ATK permanently exited after first outward crossing of the reference TGT sensing radius."),
        ("engagement_boundary", "engagement boundary", "Lethal moving TGT defense boundary contact participation; only DEF when enabled."),
        ("vessel_collision", "vessel contact participation", "Each participating craft counts once per new unordered pair contact; global pairs count once."),
        *(((("capture", "captured", "Capture by the opposing camp; not an ordinary collision."),)) if role != "defender" else ()),
      )),
    MetricSpec("event/pair_contacts_per_episode", "behavior", "Role behavior", "Physical vessel contacts", "events/episode", "episode_mean",
               "One per unordered pair contact; excludes captures.", "match/count", axis_min=0),
    *(MetricSpec(key, "ppo", "Optimization", label, unit, "update_mean", tip, scope="role", training_series="sides",
                 axis_min=minimum, axis_max=maximum)
      for key, label, unit, tip, minimum, maximum in (
        ("policy/sigma_surge", "Collect surge Gaussian scale", "", "Behavior policy Gaussian scale used to collect this rollout.", 0, None),
        ("policy/sigma_yaw", "Collect yaw Gaussian scale", "", "Behavior policy Gaussian scale used to collect this rollout.", 0, None),
        ("optimizer/policy_loss", "Policy loss", "", "Policy loss for this trained role.", None, None),
        ("optimizer/value_loss", "Value loss", "", "Value loss including captured members' team credit.", 0, None),
        ("optimizer/explained_variance", "Explained variance", "", "Post-update Critic explained variance on credit-member value targets.", None, 1),
        ("optimizer/entropy", "Entropy", "", "Policy entropy.", None, None),
        ("optimizer/approx_kl", "KL", "", "Approximate policy KL.", 0, None),
        ("optimizer/clip_fraction", "Clip fraction", "%", "Fraction clipped by PPO.", 0, 1),
        ("train/learning_rate", "Learning rate", "", "Actual optimizer learning rate.", 0, None),
        ("train/mean_actor_minibatch_frames", "Mean actor minibatch", "frames", "Mean actual active policy samples per minibatch for this role.", 0, None),
        ("train/agent_frames_this_update", "Active role frames", "frames", "Policy-valid active craft frames.", 0, None),
      )),
    *(MetricSpec(key, "profiling", "Performance", label, unit, "update_mean", label, axis_min=0)
      for key, label, unit in (
        ("time/collect_seconds", "Shared collection", "s"),
        ("time/collect_environment_step_seconds", "Shared physics", "s"),
        ("time/collect_reset_seconds", "Shared reset", "s"),
        ("time/update_seconds", "Collect + optimize time", "s"),
        ("time/environment_steps_per_second", "Environment throughput", "steps/s"),
        ("train/optimizer_steps", "Optimizer steps per update", "steps"),
        ("time/collect_policy_seconds", "Shared policy inference", "s"),
        ("time/collect_observation_bundle_seconds", "Shared observation preparation", "s"),
        ("time/optimization_seconds", "Parameter group optimization", "s"),
        *((f"time/collect_observation_{role}_seconds", f"{desc['label']} observation", "s")
          for role, desc in ROLE_DESCRIPTORS.items()),
      )),
)

METRIC_REGISTRY = {spec.key: spec for spec in METRIC_SPECS}


def metric_registry_payload() -> list[dict[str, object]]:
    result = []
    for spec in METRIC_SPECS:
        item = asdict(spec)
        group, label = spec.key, spec.label
        series = [{'key':spec.key, 'role':spec.role, 'label':spec.label, 'color':'#8caab3'}]
        if spec.scope == 'role':
            series = [{'key':spec.key, 'role':role, 'label':desc['label'], 'color':desc['color'], 'role_metric':True}
                      for role,desc in ROLE_DESCRIPTORS.items()]
        elif spec.role:
            event = spec.key.split('/')[2]
            group = 'event/'+event
            label = spec.label.removeprefix(ROLE_DESCRIPTORS[spec.role]['label']+' ')
            series = [{'key':other.key, 'role':other.role, 'label':ROLE_DESCRIPTORS[other.role]['label'],
                       'color':ROLE_DESCRIPTORS[other.role]['color']}
                      for other in METRIC_SPECS if other.role and other.key.split('/')[2]==event]
        elif spec.key in ('match/attacker_win_rate','match/defender_win_rate','match/attacker_return','match/defense_return'):
            rate = spec.key.endswith('win_rate')
            group, label = ('team/win_rate','Team win rate') if rate else ('team/return','Team return')
            series = [{'key':'match/'+key, 'role':None, 'color_role':'learner' if team=='Attack' else 'defender', 'label':team, 'color':color} for key,team,color in
                      ((('attacker_win_rate' if rate else 'attacker_return'),'Attack','#e45e69'),
                       (('defender_win_rate' if rate else 'defense_return'),'Defense','#55bda0'))]
        item.update(plot_group=group, plot_label=label, series=series)
        result.append(item)
    return result


def metric_applicable(spec, settings):
    return (settings['target_mode'] in spec.target_modes
            and (spec.key != 'match/protected_time' or settings.get('training_settings', {}).get('time_limit_boundary', 'termination') == 'termination')
            and (not spec.requires_first_casualty or settings['terminate_on_any_casualty'])
            and ('/engagement_boundary_' not in spec.key or
                 (spec.role == 'defender' and settings.get('safety_zones_enabled', False))))


OUTCOME_TIME_COEFFICIENT = .25
REWARD_DESCRIPTIONS = {
    'protection_time': 'Defense utility: executed ticks / horizon; early defense wins receive remaining time',
    'outcome': 'Terminal camp reward +1 / -1; reported returns are undiscounted even when PPO gamma < 1',
    'outcome_time': f'Terminal camp reward sign * (1 - {OUTCOME_TIME_COEFFICIENT} * physics_ticks / horizon_ticks)',
}


def match_registry_payload(reward_objective="protection_time", *, first_casualty=True) -> dict[str, object]:
    if reward_objective not in REWARD_DESCRIPTIONS:
        raise ValueError('Unsupported reward objective')
    return {
        "roles": ROLE_DESCRIPTORS,
        "applicable_events": [event.value for event in MatchEvent if event != MatchEvent.REACH],
        "applicable_end_reasons": [reason.value for reason in MatchEndReason
                                   if reason not in (MatchEndReason.NONE, MatchEndReason.BREACH)
                                   and (first_casualty or reason not in (MatchEndReason.ATTACKER_CASUALTY, MatchEndReason.DEFENDER_CASUALTY))],
        "reward_objective": reward_objective,
        "reward_description": REWARD_DESCRIPTIONS[reward_objective],
        "outcomes": [
            {
                "code": int(MatchOutcome.ATTACKER_WIN),
                "id": "attacker_win",
                "label": "Attacker Win",
                "reward": {"attacker": 1, "defender": -1, "target": -1} if reward_objective=="outcome" else None,
            },
            {
                "code": int(MatchOutcome.DEFENDER_WIN),
                "id": "defender_win",
                "label": "Defender Win",
                "reward": {"attacker": -1, "defender": 1, "target": 1} if reward_objective=="outcome" else None,
            },
        ],
        "end_reasons": [
            {"id":"defender_elimination", "short_label":"DE", "label":"DEF ELIMINATED", "icon":"∅", "winner":"attacker"},
            {"id":"truncation", "short_label":"TR", "label":"TRUNCATED", "icon":"◷", "winner":None},
            {"id":"target_casualty","short_label":"TX","label":"TGT COLLISION","icon":"✹","winner":"attacker"},
            {
                "id": MatchEndReason.BREACH.value,
                "short_label": "BR",
                "label": "TARGET REACHED",
                "icon": "⚑",
                "winner": "attacker",
            },
            {
                "id": MatchEndReason.ELIMINATION.value,
                "short_label": "EL",
                "label": "ELIMINATION",
                "icon": "∅",
                "winner": "defender",
            },
            {
                "id": MatchEndReason.HORIZON.value,
                "short_label": "TL",
                "label": "HOLD",
                "icon": "◷",
                "winner": "defender",
            },
            {
                "id": MatchEndReason.ATTACKER_CASUALTY.value,
                "short_label": "AC",
                "label": "ATK CASUALTY",
                "icon": "☠",
                "winner": "defender",
            },
            {
                "id": MatchEndReason.DEFENDER_CASUALTY.value,
                "short_label": "DC",
                "label": "DEFENSE CASUALTY",
                "icon": "☠",
                "winner": "attacker",
            },
            {
                "id": MatchEndReason.TARGET_CAPTURE.value,
                "short_label": "TC",
                "label": "TGT CAPTURED",
                "icon": "☠",
                "winner": "attacker",
            },
        ],
        "events": [
            {
                "id": event.value,
                "label": label,
                "short_label": short_label,
                "icon": icon,
                "actors": actors,
            }
            for event, label, short_label, icon, actors in (
                (MatchEvent.REACH, "Reached Target", "REACH", "⚑", ("attacker",)),
                (MatchEvent.CAPTURE, "Captured", "CAPT", "☠", ("attacker", "target")),
                (
                    MatchEvent.STATIC_COLLISION,
                    "Static Collision",
                    "S-COLL",
                    "✹",
                    ("attacker", "defender", "target"),
                ),
                (
                    MatchEvent.VESSEL_COLLISION,
                    "Vessel Collision",
                    "V-COLL",
                    "↯",
                    ("attacker", "defender", "target"),
                ),
            )
        ],
        "aggregation": {
            "viewer": "current match and five completed matches",
            "training": "complete episodes finished in one update",
            "tensorboard": "same per-update values as training",
        },
    }
