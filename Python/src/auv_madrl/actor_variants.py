from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

ROLE_DESCRIPTORS = {
    "learner": {"label": "ATK", "camp": "attack", "event_role": "attacker", "color": "#e45e69", "line_style": "solid"},
    "defender": {"label": "DEF", "camp": "defense", "event_role": "defender", "color": "#55bda0", "line_style": "solid"},
    "target": {"label": "TGT", "camp": "defense", "event_role": "target", "color": "#e2ba63", "line_style": "solid"},
}

ActorKind = Literal["learner", "defender", "target"]
ActorInputKind = Literal["semantic_map", "entity_attention", "entity_mlp"]
EntityAttentionKind = Literal[
    "independent_ownship_cross_attention",
    "joint_self_then_ownship_cross_attention",
]
EntityMlpKind = Literal["flattened", "tied_linear_sum", "deep_sets"]
EntityOrdering = Literal["storage", "distance", "absolute_bearing"]
EntityPooling = Literal["learned_attention", "masked_mean"]
EntityStreams = Literal["separate", "unified"]
EntityAttentionMode = Literal[
    "independent_cross",
    "mean_attention_residual",
    "direct_qkv_cross",
    "same_type_self_then_cross",
    "token_mlp_then_cross",
    "cross_type_softmax_self_then_cross",
    "typed_unified_cross",
]
ActionDistributionKind = Literal["tanh_normal", "truncated_normal", "clipped_normal"]
TrainerOverrideValue = int | float

CRAFT_SPEED_RATIO_BOUNDS = (0.2, 5.0)
ROLE_ORDER = tuple(ROLE_DESCRIPTORS)
CAPTURE_RELATIONS = (("defender", "learner"), ("learner", "target"))
ENTITY_SET_ORDER = ("target", "learner", "defender", "obstacles")


def physical_roles(settings=None):
    return ROLE_ORDER


def policy_roles(settings):
    return tuple(role for role in ROLE_ORDER if role != "target" or settings.get("target_mode", "dynamic") == "dynamic")


def train_roles(settings):
    modes = settings.get("training_side_modes", {})
    return tuple(role for role in policy_roles(settings) if modes.get(role, "train") == "train")


def parameter_group_bindings(settings):
    sharing = settings.get("policy_sharing", "role")
    if sharing not in {"role", "team", "all"}:
        raise ValueError("policy_sharing must be role, team or all")
    return {role: role if sharing == "role" else ROLE_DESCRIPTORS[role]["camp"] if sharing == "team" else "all"
            for role in train_roles(settings)}


def entity_tensor_prefix(buffer_name: str) -> str:
    return {
        "goals": "goal",
        "obstacles": "obstacle",
        "friendlies": "friendly",
        "enemies": "enemy",
        "target": "target",
        "learner": "learner",
        "defender": "defender",
    }[buffer_name]


@dataclass(frozen=True)
class ActorSemanticFeature:
    name: str
    viewer_label: str
    viewer_color_rgb: tuple[int, int, int]
    values: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class ActorVectorFeature:
    name: str
    viewer_label: str
    value_range: tuple[float, float]


@dataclass(frozen=True)
class ActorVariantSpec:
    architecture_id: str
    actor_kind: ActorKind
    task_id: str
    display_name: str
    input_kind: ActorInputKind
    semantic_channels: tuple[str, ...]
    vector_inputs: tuple[str, ...]
    entity_features: tuple[str, ...]
    coordinate_frame: str
    output_directory: Path
    entity_sets: tuple[str, ...] = ("goals", "obstacles")
    entity_set_feature_overrides: tuple[tuple[str, tuple[str, ...]], ...] = ()
    critic_vector_inputs: tuple[str, ...] | None = None
    critic_entity_set_feature_overrides: tuple[
        tuple[str, tuple[str, ...]], ...
    ] = ()
    entity_attention_kind: EntityAttentionKind | None = None
    entity_mlp_kind: EntityMlpKind | None = None
    entity_ordering: EntityOrdering | None = None
    goal_entity_capacity: int | None = None
    obstacle_entity_capacity: int | None = None
    viewer_value_bar_track_width_px: int | None = None
    entity_pooling: EntityPooling = "learned_attention"
    entity_streams: EntityStreams = "separate"
    entity_attention_mode: EntityAttentionMode = "independent_cross"
    attention_embedding_size: int = 64
    attention_heads: int = 4
    entity_self_attention_layers: int = 0
    body_hidden_size: int = 256
    actor_initial_log_scale: float = -1.5
    action_distribution: ActionDistributionKind = "tanh_normal"
    trainer_overrides: tuple[tuple[str, TrainerOverrideValue], ...] = ()
    defender_count: int = 1
    training_enabled: bool = True
    viewer_enabled: bool = True
    unified_match_contract: bool = False

    def entity_features_for(self, buffer_name: str) -> tuple[str, ...]:
        return dict(self.entity_set_feature_overrides).get(
            buffer_name, self.entity_features
        )

    def entity_feature_size_for(self, buffer_name: str) -> int:
        return len(self.entity_features_for(buffer_name))

    @property
    def critic_vector_input_names(self) -> tuple[str, ...]:
        return (
            self.vector_inputs
            if self.critic_vector_inputs is None
            else self.critic_vector_inputs
        )

    def critic_entity_features_for(self, buffer_name: str) -> tuple[str, ...]:
        return dict(self.critic_entity_set_feature_overrides).get(
            buffer_name, self.entity_features_for(buffer_name)
        )

    def critic_entity_feature_size_for(self, buffer_name: str) -> int:
        return len(self.critic_entity_features_for(buffer_name))

    @property
    def trainer_override_map(self) -> dict[str, TrainerOverrideValue]:
        return dict(self.trainer_overrides)

    @property
    def semantic_input_shape(self) -> tuple[int, int, int]:
        return (len(self.semantic_channels), 96, 96)

    @property
    def vector_input_size(self) -> int:
        return len(self.vector_inputs)

    @property
    def critic_vector_input_size(self) -> int:
        return len(self.critic_vector_input_names)

    @property
    def entity_feature_size(self) -> int:
        return len(self.entity_features)

    @property
    def uses_entity_buffers(self) -> bool:
        return self.input_kind in {"entity_attention", "entity_mlp"}

    @property
    def actor_observation_keys(self) -> tuple[str, ...]:
        if not self.uses_entity_buffers:
            return ("semantic_map", "vector")
        keys = ["vector"]
        for buffer_name in self.entity_sets:
            prefix = entity_tensor_prefix(buffer_name)
            keys.extend((f"{prefix}_entities", f"{prefix}_mask"))
        return tuple(keys)

    @property
    def critic_observation_keys(self) -> tuple[str, ...]:
        if not self.uses_entity_buffers:
            return ("semantic_map", "vector")
        keys = ["critic_vector" if self.critic_vector_input_names != self.vector_inputs else "vector"]
        for buffer_name in self.entity_sets:
            prefix = entity_tensor_prefix(buffer_name)
            separate = (
                (self.unified_match_contract and buffer_name != "obstacles")
                or self.critic_entity_features_for(buffer_name)
                != self.entity_features_for(buffer_name)
            )
            keys.append(f"{prefix}_critic_entities" if separate else f"{prefix}_entities")
            keys.append(f"{prefix}_critic_mask" if self.unified_match_contract and buffer_name != "obstacles" else f"{prefix}_mask")
        return tuple(keys)

    @property
    def observation_keys(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.actor_observation_keys, *self.critic_observation_keys)))

    @property
    def checkpoint_shape_parameter_names(self) -> tuple[str, ...]:
        if self.input_kind == "semantic_map":
            return (
                "actor_semantic.network.0.weight",
                "actor_vector.0.weight",
            )
        if self.entity_attention_kind == "independent_ownship_cross_attention":
            if self.entity_streams == "unified":
                return (
                    "actor_vector.0.weight",
                    *(
                        "actor_typed_entities.encoders."
                        f"{buffer_name}.0.weight"
                        for buffer_name in self.entity_sets
                    ),
                    "actor_typed_entities.type_embedding.weight",
                    *(
                        "actor_typed_entities.self_attention_blocks."
                        f"{layer_index}.attention.in_proj_weight"
                        for layer_index in range(
                            self.entity_self_attention_layers
                        )
                    ),
                    "actor_typed_entities.cross_attention.in_proj_weight",
                    "actor_body.0.weight",
                )
            if self.entity_attention_mode == "cross_type_softmax_self_then_cross":
                return (
                    "actor_vector.0.weight",
                    *(
                        "actor_cross_type_entities.encoders."
                        f"{buffer_name}.0.weight"
                        for buffer_name in self.entity_sets
                    ),
                    "actor_cross_type_entities.type_embedding.weight",
                    "actor_cross_type_entities.self_attention.in_proj_weight",
                    "actor_cross_type_entities.pool_attentions.goals.in_proj_weight",
                    "actor_body.0.weight",
                )
            if self.entity_attention_mode == "direct_qkv_cross":
                return (
                    "actor_vector.0.weight",
                    *(
                        f"actor_{entity_tensor_prefix(buffer_name)}_entities."
                        "attention.k_proj_weight"
                        for buffer_name in self.entity_sets
                    ),
                    f"actor_{entity_tensor_prefix(self.entity_sets[0])}_entities."
                    "attention.q_proj_weight",
                    "actor_body.0.weight",
                )
            names = ("actor_vector.0.weight",) + tuple(
                f"actor_{entity_tensor_prefix(buffer_name)}_entities.encoder.0.weight"
                for buffer_name in self.entity_sets
            )
            if self.entity_pooling == "learned_attention":
                names += (
                    f"actor_{entity_tensor_prefix(self.entity_sets[0])}_entities."
                    "attention.in_proj_weight",
                )
            if self.entity_self_attention_layers:
                names += tuple(
                    f"actor_{entity_tensor_prefix(self.entity_sets[0])}_entities."
                    f"self_attention_blocks.{layer_index}.attention.in_proj_weight"
                    for layer_index in range(self.entity_self_attention_layers)
                )
            if self.entity_attention_mode == "mean_attention_residual":
                names += (
                    f"actor_{entity_tensor_prefix(self.entity_sets[0])}_entities.alpha",
                )
            elif self.entity_attention_mode == "same_type_self_then_cross":
                names += (
                    f"actor_{entity_tensor_prefix(self.entity_sets[0])}_entities."
                    "pre_attention.in_proj_weight",
                )
            elif self.entity_attention_mode == "token_mlp_then_cross":
                names += (
                    f"actor_{entity_tensor_prefix(self.entity_sets[0])}_entities."
                    "pre_mlp.0.weight",
                )
            return (*names, "actor_body.0.weight")
        if self.entity_attention_kind == "joint_self_then_ownship_cross_attention":
            return (
                "actor_vector.0.weight",
                "actor_relational_entities.encoder.0.weight",
                "actor_relational_entities.type_embedding.weight",
                "actor_relational_entities.self_attention.in_proj_weight",
                "actor_relational_entities.goal_attention.in_proj_weight",
                "actor_relational_entities.obstacle_attention.in_proj_weight",
                "actor_body.0.weight",
            )
        if self.entity_mlp_kind == "flattened":
            return (
                "actor_vector.0.weight",
                "actor_flat_entities.0.weight",
                "actor_body.0.weight",
            )
        if self.entity_mlp_kind == "tied_linear_sum":
            return (
                "actor_vector.0.weight",
                "actor_tied_entities.0.weight",
                "actor_body.0.weight",
            )
        if self.entity_mlp_kind == "deep_sets":
            return (
                "actor_vector.0.weight",
                "actor_goal_entities.encoder.0.weight",
                "actor_obstacle_entities.encoder.0.weight",
                "actor_body.0.weight",
            )
        raise ValueError("Entity-buffer architecture kind is missing")

    @property
    def checkpoint_shape_identity(self) -> tuple[tuple[int, ...], ...]:
        if self.input_kind == "semantic_map":
            return (
                (32, self.semantic_input_shape[0], 5, 5),
                (64, self.vector_input_size),
            )
        if self.entity_attention_kind == "independent_ownship_cross_attention":
            embedding_size = self.attention_embedding_size
            fusion_size = (1 + len(self.entity_sets)) * embedding_size
            if self.entity_streams == "unified":
                return (
                    (embedding_size, self.vector_input_size),
                    *(
                        (embedding_size, self.entity_feature_size_for(buffer_name))
                        for buffer_name in self.entity_sets
                    ),
                    (len(self.entity_sets), embedding_size),
                    *(
                        (3 * embedding_size, embedding_size)
                        for _ in range(self.entity_self_attention_layers)
                    ),
                    (3 * embedding_size, embedding_size),
                    (self.body_hidden_size, 2 * embedding_size),
                )
            if self.entity_attention_mode == "cross_type_softmax_self_then_cross":
                return (
                    (embedding_size, self.vector_input_size),
                    *(
                        (embedding_size, self.entity_feature_size_for(buffer_name))
                        for buffer_name in self.entity_sets
                    ),
                    (len(self.entity_sets), embedding_size),
                    (3 * embedding_size, embedding_size),
                    (3 * embedding_size, embedding_size),
                    (self.body_hidden_size, fusion_size),
                )
            if self.entity_attention_mode == "direct_qkv_cross":
                return (
                    (embedding_size, self.vector_input_size),
                    *(
                        (embedding_size, self.entity_feature_size_for(buffer_name))
                        for buffer_name in self.entity_sets
                    ),
                    (embedding_size, embedding_size),
                    (self.body_hidden_size, fusion_size),
                )
            identity = (
                (embedding_size, self.vector_input_size),
                *(
                    (embedding_size, self.entity_feature_size_for(buffer_name))
                    for buffer_name in self.entity_sets
                ),
            )
            if self.entity_pooling == "learned_attention":
                identity += ((3 * embedding_size, embedding_size),)
            identity += tuple(
                (3 * embedding_size, embedding_size)
                for _ in range(self.entity_self_attention_layers)
            )
            if self.entity_attention_mode == "mean_attention_residual":
                identity += ((),)
            elif self.entity_attention_mode == "same_type_self_then_cross":
                identity += ((3 * embedding_size, embedding_size),)
            elif self.entity_attention_mode == "token_mlp_then_cross":
                identity += ((2 * embedding_size, embedding_size),)
            return (
                *identity,
                (self.body_hidden_size, fusion_size),
            )
        if self.entity_attention_kind == "joint_self_then_ownship_cross_attention":
            embedding_size = self.attention_embedding_size
            return (
                (embedding_size, self.vector_input_size),
                (embedding_size, self.entity_feature_size),
                (2, embedding_size),
                (3 * embedding_size, embedding_size),
                (3 * embedding_size, embedding_size),
                (3 * embedding_size, embedding_size),
                (self.body_hidden_size, 3 * embedding_size),
            )
        if self.entity_mlp_kind == "flattened":
            slot_count = int(self.goal_entity_capacity or 0) + int(
                self.obstacle_entity_capacity or 0
            )
            return (
                (64, self.vector_input_size),
                (64, slot_count * (self.entity_feature_size + 1)),
                (256, 2 * 64),
            )
        if self.entity_mlp_kind == "tied_linear_sum":
            return (
                (64, self.vector_input_size),
                (64, 2 * (self.entity_feature_size + 1)),
                (256, 2 * 64),
            )
        if self.entity_mlp_kind == "deep_sets":
            return (
                (64, self.vector_input_size),
                (32, self.entity_feature_size),
                (32, self.entity_feature_size),
                (256, 64 + 2 * 32),
            )
        raise ValueError("Entity-buffer architecture kind is missing")

    def checkpoint_shape_identity_from(
        self, state_dict: Mapping[str, object]
    ) -> tuple[tuple[int, ...], ...]:
        try:
            return tuple(
                tuple(state_dict[name].shape)  # type: ignore[attr-defined]
                for name in self.checkpoint_shape_parameter_names
            )
        except (AttributeError, KeyError) as error:
            raise ValueError("Checkpoint model state is incomplete") from error

    def architecture_contract(self) -> dict[str, object]:
        semantic = tuple(ACTOR_SEMANTIC_FEATURES[name] for name in self.semantic_channels)
        vector = tuple(ACTOR_VECTOR_FEATURES[name] for name in self.vector_inputs)
        critic_vector = tuple(
            ACTOR_VECTOR_FEATURES[name] for name in self.critic_vector_input_names
        )
        entity = tuple(ACTOR_ENTITY_FEATURES[name] for name in self.entity_features)
        attention: dict[str, object] | None = None
        critic = "independent_visual_encoder_vector_branch_and_value_head"
        if self.entity_attention_kind == "independent_ownship_cross_attention":
            interactions = {
                "independent_cross": "none_before_pooling",
                "mean_attention_residual": "none_before_pooling",
                "direct_qkv_cross": "none_before_pooling",
                "same_type_self_then_cross": "same_type_self_attention",
                "token_mlp_then_cross": "none_tokenwise_depth_control",
                "cross_type_softmax_self_then_cross": "cross_type_self_attention",
                "typed_unified_cross": (
                    "cross_type_self_attention"
                    if self.entity_self_attention_layers
                    else "none_before_unified_pooling"
                ),
            }
            entity_interaction = interactions[self.entity_attention_mode]
            if (
                self.entity_streams == "separate"
                and self.entity_self_attention_layers > 0
            ):
                entity_interaction = "same_type_self_attention"
            attention = {
                "kind": self.entity_attention_kind,
                "mode": self.entity_attention_mode,
                "entity_streams": self.entity_streams,
                "pooling": self.entity_pooling,
                "embedding_size": self.attention_embedding_size,
                "heads": (
                    self.attention_heads
                    if self.entity_pooling == "learned_attention"
                    else None
                ),
                "query": "ownship_vector_embedding",
                "sets": list(self.entity_sets),
                "entity_interaction": entity_interaction,
                "type_encoding": (
                    "learned_per_entity_type_embedding"
                    if self.entity_streams == "unified"
                    else None
                ),
                "self_attention_layers": self.entity_self_attention_layers,
                "cross_attention_outputs": (
                    ["all_entity_types"]
                    if self.entity_streams == "unified"
                    else list(self.entity_sets)
                ),
                "padding_mask": "active_entity_mask",
                "permutation_invariant": True,
            }
            critic = "independent_goal_obstacle_attention_vector_branch_and_value_head"
        elif self.entity_attention_kind == "joint_self_then_ownship_cross_attention":
            attention = {
                "kind": self.entity_attention_kind,
                "embedding_size": self.attention_embedding_size,
                "heads": self.attention_heads,
                "self_attention_layers": 1,
                "feed_forward_size": 128,
                "type_encoding": "learned_goal_obstacle_embedding",
                "self_attention_set": "goals_plus_obstacles",
                "query": "ownship_vector_embedding",
                "cross_attention_outputs": ["goals", "obstacles"],
                "padding_mask": "active_entity_mask",
                "permutation_invariant": True,
            }
            critic = "independent_joint_entity_relation_attention_and_value_head"
        entity_aggregation: dict[str, object] | None = None
        if self.entity_mlp_kind == "flattened":
            entity_aggregation = {
                "kind": "flattened_mlp",
                "ordering": self.entity_ordering,
                "goal_slots": self.goal_entity_capacity,
                "obstacle_slots": self.obstacle_entity_capacity,
                "slot_features": [*self.entity_features, "active_mask"],
                "permutation_invariant": self.entity_ordering != "storage",
            }
            critic = "independent_flat_entity_mlp_and_value_head"
        elif self.entity_mlp_kind == "tied_linear_sum":
            entity_aggregation = {
                "kind": "tied_linear_sum",
                "reduction": "masked_sum_before_linear_encoding",
                "sets": ["goals", "obstacles"],
                "goal_slots": self.goal_entity_capacity,
                "obstacle_slots": self.obstacle_entity_capacity,
                "slot_features": [*self.entity_features, "active_mask"],
                "per_entity_nonlinearity": False,
                "permutation_invariant": True,
            }
            critic = "independent_tied_linear_entity_set_and_value_head"
        elif self.entity_mlp_kind == "deep_sets":
            entity_aggregation = {
                "kind": "deep_sets",
                "embedding_size": 32,
                "reduction": "masked_sum",
                "sets": ["goals", "obstacles"],
                "goal_slots": self.goal_entity_capacity,
                "obstacle_slots": self.obstacle_entity_capacity,
                "permutation_invariant": True,
            }
            critic = "independent_deep_sets_and_value_head"
        return {
            "architecture_id": self.architecture_id,
            "actor_kind": None if self.unified_match_contract else self.actor_kind,
            "role_descriptors": ROLE_DESCRIPTORS,
            "role_onehot_order": list(ROLE_ORDER),
            "task_id": self.task_id,
            "defender_count": self.defender_count,
            "unified_match_contract": self.unified_match_contract,
            "speed_ratio_normalization": (
                {
                    "kind": "symmetric_log",
                    "physical_range": list(CRAFT_SPEED_RATIO_BOUNDS),
                    "network_range": [-1.0, 1.0],
                    "neutral_ratio": 1.0,
                }
                if self.unified_match_contract
                else None
            ),
            "display_name": self.display_name,
            "input_kind": self.input_kind,
            "coordinate_frame": self.coordinate_frame,
            "semantic_input_shape": list(self.semantic_input_shape),
            "semantic_channels": list(self.semantic_channels),
            "semantic_values": {
                feature.name: dict(feature.values) for feature in semantic
            },
            "viewer_channel_labels": [feature.viewer_label for feature in semantic],
            "viewer_channel_colors": [
                list(feature.viewer_color_rgb) for feature in semantic
            ],
            "vector_input_size": self.vector_input_size,
            "vector_inputs": list(self.vector_inputs),
            "viewer_vector_labels": [feature.viewer_label for feature in vector],
            "viewer_vector_ranges": [list(feature.value_range) for feature in vector],
            "critic_vector_input_size": self.critic_vector_input_size,
            "critic_vector_inputs": list(self.critic_vector_input_names),
            "critic_vector_labels": [
                feature.viewer_label for feature in critic_vector
            ],
            "critic_vector_ranges": [
                list(feature.value_range) for feature in critic_vector
            ],
            "viewer_value_bar_track_width_px": self.viewer_value_bar_track_width_px,
            "entity_feature_size": self.entity_feature_size,
            "entity_features": list(self.entity_features),
            "viewer_entity_labels": [feature.viewer_label for feature in entity],
            "viewer_entity_ranges": [list(feature.value_range) for feature in entity],
            "entity_attention_kind": self.entity_attention_kind,
            "entity_streams": self.entity_streams,
            "entity_attention_mode": self.entity_attention_mode,
            "entity_mlp_kind": self.entity_mlp_kind,
            "entity_ordering": self.entity_ordering,
            "goal_entity_capacity": self.goal_entity_capacity,
            "obstacle_entity_capacity": self.obstacle_entity_capacity,
            "trainer_overrides": self.trainer_override_map,
            "entity_buffers": list(self.entity_sets) if self.uses_entity_buffers else [],
            "entity_buffer_schemas": {
                buffer_name: {
                    "feature_size": self.entity_feature_size_for(buffer_name),
                    "features": list(self.entity_features_for(buffer_name)),
                    "viewer_labels": [
                        ACTOR_ENTITY_FEATURES[name].viewer_label
                        for name in self.entity_features_for(buffer_name)
                    ],
                    "viewer_ranges": [
                        list(ACTOR_ENTITY_FEATURES[name].value_range)
                        for name in self.entity_features_for(buffer_name)
                    ],
                }
                for buffer_name in self.entity_sets
            },
            "critic_entity_buffer_schemas": {
                buffer_name: {
                    "feature_size": self.critic_entity_feature_size_for(
                        buffer_name
                    ),
                    "features": list(
                        self.critic_entity_features_for(buffer_name)
                    ),
                }
                for buffer_name in self.entity_sets
            },
            "attention": attention,
            "entity_aggregation": entity_aggregation,
            "fusion": (
                f"vector_{self.attention_embedding_size}_concat_"
                f"unified_entity_attention_{self.attention_embedding_size}"
                if self.entity_streams == "unified"
                else "_concat_".join(
                    (
                        f"vector_{self.attention_embedding_size}",
                        *(
                            f"{entity_tensor_prefix(buffer_name)}_"
                            f"{self.entity_pooling}_{self.attention_embedding_size}"
                            for buffer_name in self.entity_sets
                        ),
                    )
                )
                if self.input_kind == "entity_attention"
                else "vector_64_concat_flat_entities_64"
                if self.entity_mlp_kind == "flattened"
                else "vector_64_concat_tied_linear_entity_sum_64"
                if self.entity_mlp_kind == "tied_linear_sum"
                else "vector_64_concat_goal_deepset_32_concat_obstacle_deepset_32"
                if self.entity_mlp_kind == "deep_sets"
                else "semantic_64_concat_vector_64"
            ),
            "hidden_layers": [self.body_hidden_size, self.body_hidden_size],
            "activation": "SiLU",
            "action_dimensions": ["surge", "yaw"],
            "distribution": (
                "TanhNormal"
                if self.action_distribution == "tanh_normal"
                else "TruncatedNormal"
                if self.action_distribution == "truncated_normal"
                else "ClippedNormal"
            ),
            "actor_initial_log_scale": self.actor_initial_log_scale,
            "critic": critic,
            "actor_critic_weights_shared": False,
            "recurrent": False,
            "training_ready": self.training_enabled,
            "viewer_ready": self.viewer_enabled,
        }


ACTOR_SEMANTIC_FEATURES = {
    feature.name: feature
    for feature in (
        ActorSemanticFeature(
            "true_fatal_geometry",
            "Obstacle Regions",
            (255, 102, 121),
            (("free", 0), ("blocked", 1)),
        ),
        ActorSemanticFeature(
            "goal_region",
            "Goal Regions",
            (244, 200, 92),
            (("outside", 0), ("inside", 1)),
        ),
        ActorSemanticFeature(
            "ownship_position",
            "Own Position",
            (231, 244, 248),
            (("absent", 0), ("present", 1)),
        ),
    )
}

ACTOR_VECTOR_FEATURES = {
    feature.name: feature
    for feature in (
        ActorVectorFeature("map_frame_x_over_R", "map_x/R", (-1.0, 1.0)),
        ActorVectorFeature("map_frame_z_over_R", "map_z/R", (-1.0, 1.0)),
        ActorVectorFeature("sin_heading", "sin(psi)", (-1.0, 1.0)),
        ActorVectorFeature("cos_heading", "cos(psi)", (-1.0, 1.0)),
        ActorVectorFeature("surge_over_2", "u/2", (0.0, 1.0)),
        ActorVectorFeature("sway", "v/1", (-1.0, 1.0)),
        ActorVectorFeature("surge_over_max", "u/u_max", (0.0, 1.0)),
        ActorVectorFeature("sway_over_half_max", "v/(u_max/2)", (-1.0, 1.0)),
        ActorVectorFeature("yaw_rate_over_max", "r/r_max", (-1.0, 1.0)),
        ActorVectorFeature(
            "defender_speed_ratio_normalized", "lin₁₋₅(k)", (-1.0, 1.0)
        ),
        ActorVectorFeature("yaw_rate_over_0_2", "r/0.2", (-1.0, 1.0)),
        ActorVectorFeature("speed_ratio_normalized", "log₅(k)", (-1.0, 1.0)),
        ActorVectorFeature("alive", "alive", (0.0, 1.0)),
        ActorVectorFeature("remaining_time", "1-t/H", (0.0, 1.0)),
        ActorVectorFeature("role_atk", "ATK", (0.0, 1.0)),
        ActorVectorFeature("role_def", "DEF", (0.0, 1.0)),
        ActorVectorFeature("role_tgt", "TGT", (0.0, 1.0)),
        ActorVectorFeature("task_limit_over_R", "task limit/R", (0.0, 1.0)),
    )
}

ACTOR_ENTITY_FEATURES = {
    feature.name: feature
    for feature in (
        ActorVectorFeature("relative_x_over_R", "x/R", (-2.0, 2.0)),
        ActorVectorFeature("relative_z_over_R", "z/R", (-2.0, 2.0)),
        ActorVectorFeature("radius_over_R", "r/R", (0.0, 0.2)),
        ActorVectorFeature("velocity_x_body_over_2", "v_x/2", (-1.0, 1.0)),
        ActorVectorFeature("velocity_z_body_over_2", "v_z/2", (-1.0, 1.0)),
        ActorVectorFeature("relative_heading_sin", "sin(Δψ)", (-1.0, 1.0)),
        ActorVectorFeature("relative_heading_cos", "cos(Δψ)", (-1.0, 1.0)),
        ActorVectorFeature("friend_yaw_rate_over_0_2", "r_f/0.2", (-1.0, 1.0)),
        ActorVectorFeature("enemy_yaw_rate_over_0_2", "r_e/0.2", (-1.0, 1.0)),
        ActorVectorFeature("craft_yaw_rate_over_0_2", "r/0.2", (-1.0, 1.0)),
        ActorVectorFeature(
            "learner_speed_ratio_normalized", "lin₀.₂₋₅(k)", (-1.0, 1.0)
        ),
        ActorVectorFeature("speed_ratio_normalized", "log₅(k)", (-1.0, 1.0)),
        ActorVectorFeature("alive", "alive", (0.0, 1.0)),
        ActorVectorFeature("relative_x_over_L", "x/Lpos", (-1.0, 1.0)),
        ActorVectorFeature("relative_z_over_L", "z/Lpos", (-1.0, 1.0)),
        ActorVectorFeature("radius_over_L", "r/Lpos", (0.0, 0.2)),
        ActorVectorFeature(
            "relative_velocity_x_body_over_sum", "dv_right/(vmax_i+vmax_j)", (-1.0, 1.0)
        ),
        ActorVectorFeature(
            "relative_velocity_z_body_over_sum", "dv_forward/(vmax_i+vmax_j)", (-1.0, 1.0)
        ),
        ActorVectorFeature("craft_yaw_rate_over_max", "r/rmax", (-1.0, 1.0)),
    )
}

ADVERSARIAL_GAME_BASELINE_ARCHITECTURE_ID = (
    "goal_reach_entity_attention_clipped_rollout32_p1024_mb4096_e5_actor_critic"
)
MULTI_GOAL_REACH_ARCHITECTURE_ID = (
    "goal_reach_multi_learner_entity_attention_p1024_r32_mb4096_e5_"
    "friendly_motion_actor_critic"
)
PATROL_PURSUIT_ARCHITECTURE_ID = (
    "patrol_pursuit_unified_speed_ratio_entity_attention_actor_critic"
)
PATROL_PURSUIT_LEARNER_SPEED_INPUT_ARCHITECTURE_ID = (
    "patrol_pursuit_learner_speed_input_entity_attention_actor_critic"
)
PATROL_TYPED_UNIFIED_CROSS_ATTENTION_ARCHITECTURE_ID = (
    "patrol_pursuit_typed_unified_cross_attention_actor_critic"
)
PATROL_TYPED_UNIFIED_CROSS_ATTENTION_LEARNER_SPEED_INPUT_ARCHITECTURE_ID = (
    "patrol_pursuit_typed_unified_cross_attention_learner_speed_input_actor_critic"
)
PATROL_TYPED_UNIFIED_ATTENTION_ARCHITECTURE_IDS: dict[tuple[bool, int], str] = {
    (False, 0): PATROL_TYPED_UNIFIED_CROSS_ATTENTION_ARCHITECTURE_ID,
    (True, 0): (
        PATROL_TYPED_UNIFIED_CROSS_ATTENTION_LEARNER_SPEED_INPUT_ARCHITECTURE_ID
    ),
    **{
        (speed_input, layers): (
            "patrol_pursuit_typed_unified_self_attention_"
            f"l{layers}{'_learner_speed_input' if speed_input else ''}_actor_critic"
        )
        for layers in range(1, 5)
        for speed_input in (False, True)
    },
}
MULTI_PATROL_PURSUIT_ARCHITECTURE_ID = (
    "patrol_pursuit_multi_shared_entity_attention_actor_critic"
)
MULTI_PATROL_PURSUIT_LEARNER_SPEED_INPUT_ARCHITECTURE_ID = (
    "patrol_pursuit_multi_shared_learner_speed_input_entity_attention_actor_critic"
)
PATROL_SEPARATE_ATTENTION_ARCHITECTURE_IDS: dict[tuple[bool, int], str] = {
    (False, 0): PATROL_PURSUIT_ARCHITECTURE_ID,
    (True, 0): PATROL_PURSUIT_LEARNER_SPEED_INPUT_ARCHITECTURE_ID,
    **{
        (speed_input, layers): (
            "patrol_pursuit_separate_self_attention_"
            f"l{layers}{'_learner_speed_input' if speed_input else ''}_actor_critic"
        )
        for layers in range(1, 5)
        for speed_input in (False, True)
    },
}
MULTI_PATROL_ATTENTION_ARCHITECTURE_IDS: dict[
    tuple[bool, EntityStreams, int], str
] = {
    (False, "separate", 0): MULTI_PATROL_PURSUIT_ARCHITECTURE_ID,
    (True, "separate", 0): (
        MULTI_PATROL_PURSUIT_LEARNER_SPEED_INPUT_ARCHITECTURE_ID
    ),
    **{
        (speed_input, streams, layers): (
            "patrol_pursuit_multi_shared_"
            f"{streams}_{'cross_attention' if layers == 0 else f'self_attention_l{layers}'}"
            f"{'_learner_speed_input' if speed_input else ''}_actor_critic"
        )
        for streams in ("separate", "unified")
        for layers in range(5)
        for speed_input in (False, True)
        if not (streams == "separate" and layers == 0)
    },
}
PATROL_MEAN_ATTENTION_RESIDUAL_ARCHITECTURE_ID = (
    "patrol_pursuit_mean_attention_residual_actor_critic"
)
PATROL_DIRECT_QKV_CROSS_ATTENTION_ARCHITECTURE_ID = (
    "patrol_pursuit_direct_qkv_cross_attention_actor_critic"
)
PATROL_SAME_TYPE_SELF_THEN_CROSS_ATTENTION_ARCHITECTURE_ID = (
    "patrol_pursuit_same_type_self_then_cross_attention_actor_critic"
)
PATROL_TOKEN_MLP_THEN_CROSS_ATTENTION_ARCHITECTURE_ID = (
    "patrol_pursuit_token_mlp_then_cross_attention_actor_critic"
)
PATROL_CROSS_TYPE_SOFTMAX_SELF_THEN_CROSS_ATTENTION_ARCHITECTURE_ID = (
    "patrol_pursuit_cross_type_softmax_self_then_cross_attention_actor_critic"
)
UNIFIED_MATCH_ARCHITECTURE_IDS = {
    (streams, layers): f"unified_match_{streams}_self_attention_l{layers}_actor_critic"
    for streams in ("separate", "unified") for layers in range(5)
}
# Role bindings select the same architecture contract.
UNIFIED_MATCH_ATTACKER_ARCHITECTURE_IDS = UNIFIED_MATCH_ARCHITECTURE_IDS
UNIFIED_MATCH_DEFENDER_ARCHITECTURE_IDS = UNIFIED_MATCH_ARCHITECTURE_IDS
UNIFIED_MATCH_TARGET_ARCHITECTURE_IDS = UNIFIED_MATCH_ARCHITECTURE_IDS


_GOAL_REACH_ENTITY_ATTENTION_BASE = ActorVariantSpec(
    architecture_id=ADVERSARIAL_GAME_BASELINE_ARCHITECTURE_ID,
    actor_kind="learner",
    task_id="goal_reach",
    display_name="Goal Reach · Entity Attention",
    input_kind="entity_attention",
    semantic_channels=(),
    vector_inputs=(
        "map_frame_x_over_R",
        "map_frame_z_over_R",
        "surge_over_2",
        "sway",
        "yaw_rate_over_0_2",
    ),
    entity_features=(
        "relative_x_over_R",
        "relative_z_over_R",
        "radius_over_R",
    ),
    coordinate_frame="ownship_centered_heading_up_entity_buffers",
    output_directory=Path(
        "Artifacts/GoalReach/goal-reach-entity-attention-seed42"
    ),
    entity_attention_kind="independent_ownship_cross_attention",
    viewer_value_bar_track_width_px=32,
)


_ADVERSARIAL_GAME_BASELINE_VARIANT = replace(
    _GOAL_REACH_ENTITY_ATTENTION_BASE,
    architecture_id=ADVERSARIAL_GAME_BASELINE_ARCHITECTURE_ID,
    display_name="Unified Match",
    output_directory=Path(
        "Artifacts/GoalReach/"
        "goal-reach-entity-attention-clipped-rollout32-p1024-mb4096-e5-seed42"
    ),
    attention_embedding_size=32,
    attention_heads=1,
    body_hidden_size=128,
    actor_initial_log_scale=-1.0,
    action_distribution="clipped_normal",
    trainer_overrides=tuple(
        {
            "parallel_environments": 1024,
            "rollout_decisions": 32,
            "updates": 75,
            "minibatch_agent_frames": 4096,
            "optimization_epochs": 5,
            "learning_rate": 5.0e-4,
            "gamma": 0.99,
            "gae_lambda": 0.99,
            "entropy_coefficient": 0.01,
        }.items()
    ),
)


_MULTI_GOAL_REACH_VARIANT = replace(
    _ADVERSARIAL_GAME_BASELINE_VARIANT,
    architecture_id=MULTI_GOAL_REACH_ARCHITECTURE_ID,
    output_directory=Path("Artifacts/GoalReach/adversarial-game-multi-learner-seed42"),
    entity_sets=("goals", "obstacles", "friendlies"),
    entity_set_feature_overrides=(
        (
            "friendlies",
            (
                "relative_x_over_R",
                "relative_z_over_R",
                "velocity_x_body_over_2",
                "velocity_z_body_over_2",
                "relative_heading_sin",
                "relative_heading_cos",
                "friend_yaw_rate_over_0_2",
            ),
        ),
    ),
    training_enabled=True,
)


_PATROL_PURSUIT_VARIANT = ActorVariantSpec(
    architecture_id=PATROL_PURSUIT_ARCHITECTURE_ID,
    actor_kind="defender",
    task_id="patrol_pursuit",
    display_name="Patrol Pursuit · Unified Speed Input",
    input_kind="entity_attention",
    semantic_channels=(),
    vector_inputs=(
        "map_frame_x_over_R",
        "map_frame_z_over_R",
        "surge_over_max",
        "sway_over_half_max",
        "yaw_rate_over_max",
        "defender_speed_ratio_normalized",
    ),
    entity_features=(
        "relative_x_over_R",
        "relative_z_over_R",
        "radius_over_R",
    ),
    coordinate_frame="ownship_centered_heading_up_entity_buffers",
    output_directory=Path(
        "Artifacts/GoalReach/patrol-pursuit-unified-speed-ratio-entity-attention-seed42"
    ),
    entity_sets=("goals", "obstacles", "enemies"),
    entity_set_feature_overrides=(
        (
            "enemies",
            (
                "relative_x_over_R",
                "relative_z_over_R",
                "velocity_x_body_over_2",
                "velocity_z_body_over_2",
                "relative_heading_sin",
                "relative_heading_cos",
                "enemy_yaw_rate_over_0_2",
            ),
        ),
    ),
    entity_attention_kind="independent_ownship_cross_attention",
    viewer_value_bar_track_width_px=32,
    attention_embedding_size=32,
    attention_heads=1,
    body_hidden_size=128,
    actor_initial_log_scale=-1.0,
    action_distribution="clipped_normal",
    trainer_overrides=tuple(
        {
            "parallel_environments": 1024,
            "rollout_decisions": 32,
            "updates": 150,
            "minibatch_agent_frames": 4096,
            "optimization_epochs": 5,
            "learning_rate": 5.0e-4,
            "gamma": 0.99,
            "gae_lambda": 0.99,
            "entropy_coefficient": 0.01,
        }.items()
    ),
)


_PATROL_PURSUIT_LEARNER_SPEED_INPUT_VARIANT = replace(
    _PATROL_PURSUIT_VARIANT,
    architecture_id=PATROL_PURSUIT_LEARNER_SPEED_INPUT_ARCHITECTURE_ID,
    display_name="Patrol Pursuit · ATK Speed Input",
    output_directory=Path(
        "Artifacts/GoalReach/patrol-pursuit-learner-speed-input-seed42"
    ),
    entity_set_feature_overrides=(
        (
            "enemies",
            (
                "relative_x_over_R",
                "relative_z_over_R",
                "velocity_x_body_over_2",
                "velocity_z_body_over_2",
                "relative_heading_sin",
                "relative_heading_cos",
                "enemy_yaw_rate_over_0_2",
                "learner_speed_ratio_normalized",
            ),
        ),
    ),
)


_PATROL_TYPED_UNIFIED_CROSS_ATTENTION_VARIANT = replace(
    _PATROL_PURSUIT_VARIANT,
    architecture_id=PATROL_TYPED_UNIFIED_CROSS_ATTENTION_ARCHITECTURE_ID,
    display_name="Patrol Pursuit · Unified Cross-Attention",
    output_directory=Path(
        "Artifacts/GoalReach/patrol-pursuit-typed-unified-cross-attention-seed42"
    ),
    entity_streams="unified",
    entity_attention_mode="typed_unified_cross",
)


_PATROL_TYPED_UNIFIED_CROSS_ATTENTION_LEARNER_SPEED_INPUT_VARIANT = replace(
    _PATROL_PURSUIT_LEARNER_SPEED_INPUT_VARIANT,
    architecture_id=(
        PATROL_TYPED_UNIFIED_CROSS_ATTENTION_LEARNER_SPEED_INPUT_ARCHITECTURE_ID
    ),
    display_name="Patrol Pursuit · Unified Cross-Attention · ATK Speed Input",
    output_directory=Path(
        "Artifacts/GoalReach/"
        "patrol-pursuit-typed-unified-cross-attention-speed-input-seed42"
    ),
    entity_streams="unified",
    entity_attention_mode="typed_unified_cross",
)


_PATROL_TYPED_UNIFIED_SELF_ATTENTION_VARIANTS = tuple(
    replace(
        (
            _PATROL_PURSUIT_LEARNER_SPEED_INPUT_VARIANT
            if speed_input
            else _PATROL_PURSUIT_VARIANT
        ),
        architecture_id=architecture_id,
        display_name=(
            f"Patrol Pursuit · Unified · Self-Attention L{layers}"
            + (" · ATK Speed Input" if speed_input else "")
        ),
        output_directory=Path(
            "Artifacts/GoalReach/patrol-pursuit-typed-unified-"
            f"self-attention-l{layers}"
            f"{'-speed-input' if speed_input else ''}-seed42"
        ),
        entity_streams="unified",
        entity_attention_mode="typed_unified_cross",
        entity_self_attention_layers=layers,
    )
    for (speed_input, layers), architecture_id in (
        PATROL_TYPED_UNIFIED_ATTENTION_ARCHITECTURE_IDS.items()
    )
    if layers > 0
)


_PATROL_SEPARATE_SELF_ATTENTION_VARIANTS = tuple(
    replace(
        (
            _PATROL_PURSUIT_LEARNER_SPEED_INPUT_VARIANT
            if speed_input
            else _PATROL_PURSUIT_VARIANT
        ),
        architecture_id=architecture_id,
        display_name=(
            f"Patrol Pursuit · Separate · Self-Attention L{layers}"
            + (" · ATK Speed Input" if speed_input else "")
        ),
        output_directory=Path(
            "Artifacts/GoalReach/patrol-pursuit-separate-"
            f"self-attention-l{layers}"
            f"{'-speed-input' if speed_input else ''}-seed42"
        ),
        entity_self_attention_layers=layers,
    )
    for (speed_input, layers), architecture_id in (
        PATROL_SEPARATE_ATTENTION_ARCHITECTURE_IDS.items()
    )
    if layers > 0
)


_MULTI_PATROL_PURSUIT_VARIANT = replace(
    _PATROL_PURSUIT_VARIANT,
    architecture_id=MULTI_PATROL_PURSUIT_ARCHITECTURE_ID,
    display_name="Patrol Pursuit · Shared Cross-Attention",
    output_directory=Path(
        "Artifacts/GoalReach/patrol-pursuit-multi-shared-cross-attention-seed42"
    ),
    entity_sets=("goals", "obstacles", "enemies", "friendlies"),
    entity_set_feature_overrides=(
        *_PATROL_PURSUIT_VARIANT.entity_set_feature_overrides,
        (
            "friendlies",
            (
                "relative_x_over_R",
                "relative_z_over_R",
                "velocity_x_body_over_2",
                "velocity_z_body_over_2",
                "relative_heading_sin",
                "relative_heading_cos",
                "friend_yaw_rate_over_0_2",
            ),
        ),
    ),
    defender_count=2,
)


_MULTI_PATROL_PURSUIT_LEARNER_SPEED_INPUT_VARIANT = replace(
    _MULTI_PATROL_PURSUIT_VARIANT,
    architecture_id=MULTI_PATROL_PURSUIT_LEARNER_SPEED_INPUT_ARCHITECTURE_ID,
    display_name="Patrol Pursuit · Shared Cross-Attention · ATK Speed Input",
    output_directory=Path(
        "Artifacts/GoalReach/"
        "patrol-pursuit-multi-shared-cross-attention-speed-input-seed42"
    ),
    entity_set_feature_overrides=(
        *_PATROL_PURSUIT_LEARNER_SPEED_INPUT_VARIANT.entity_set_feature_overrides,
        (
            "friendlies",
            _MULTI_PATROL_PURSUIT_VARIANT.entity_features_for("friendlies"),
        ),
    ),
)


_MULTI_PATROL_ATTENTION_VARIANTS = tuple(
    replace(
        (
            _MULTI_PATROL_PURSUIT_LEARNER_SPEED_INPUT_VARIANT
            if speed_input
            else _MULTI_PATROL_PURSUIT_VARIANT
        ),
        architecture_id=architecture_id,
        display_name=(
            f"Patrol Pursuit · Shared · {streams.title()} · Self-Attention L{layers}"
            + (" · ATK Speed Input" if speed_input else "")
        ),
        output_directory=Path(
            "Artifacts/GoalReach/patrol-pursuit-multi-shared-"
            f"{streams}-self-attention-l{layers}"
            f"{'-speed-input' if speed_input else ''}-seed42"
        ),
        entity_streams=streams,
        entity_attention_mode=(
            "typed_unified_cross" if streams == "unified" else "independent_cross"
        ),
        entity_self_attention_layers=layers,
    )
    for (speed_input, streams, layers), architecture_id in (
        MULTI_PATROL_ATTENTION_ARCHITECTURE_IDS.items()
    )
    if not (streams == "separate" and layers == 0)
)


_UNIFIED_MATCH_ACTOR_VECTOR_INPUTS = (
    "surge_over_max", "sway_over_half_max", "yaw_rate_over_max",
    "speed_ratio_normalized", "alive", "remaining_time", "role_atk",
    "role_def", "role_tgt", "task_limit_over_R",
)
_UNIFIED_MATCH_CRITIC_VECTOR_INPUTS = _UNIFIED_MATCH_ACTOR_VECTOR_INPUTS
_UNIFIED_MATCH_ACTOR_CRAFT_FEATURES = (
    "relative_x_over_L", "relative_z_over_L", "relative_velocity_x_body_over_sum",
    "relative_velocity_z_body_over_sum", "relative_heading_sin", "relative_heading_cos",
    "craft_yaw_rate_over_max", "speed_ratio_normalized",
)
_UNIFIED_MATCH_CRITIC_CRAFT_FEATURES = (*_UNIFIED_MATCH_ACTOR_CRAFT_FEATURES, "alive")
_UNIFIED_MATCH_OBSTACLE_FEATURES = (
    "relative_x_over_L", "relative_z_over_L", "radius_over_L",
)
_UNIFIED_MATCH_ENTITY_OVERRIDES = tuple(
    (role, _UNIFIED_MATCH_ACTOR_CRAFT_FEATURES) for role in ROLE_ORDER
) + (("obstacles", _UNIFIED_MATCH_OBSTACLE_FEATURES),)
_UNIFIED_MATCH_CRITIC_ENTITY_OVERRIDES = tuple(
    (role, _UNIFIED_MATCH_CRITIC_CRAFT_FEATURES) for role in ROLE_ORDER
) + (("obstacles", _UNIFIED_MATCH_OBSTACLE_FEATURES),)


def unified_match_architecture_id(streams, layers, include_remaining_time=True):
    base = UNIFIED_MATCH_ARCHITECTURE_IDS[(streams, layers)]
    return base if include_remaining_time else base.replace("_actor_critic", "_no_time_actor_critic")


def _unified_match_variant(*, streams: EntityStreams, layers: int, include_remaining_time: bool = True) -> ActorVariantSpec:
    return replace(
        _ADVERSARIAL_GAME_BASELINE_VARIANT,
        architecture_id=unified_match_architecture_id(streams, layers, include_remaining_time),
        task_id="goal_reach",
        display_name=f"Unified Match · {streams.title()} · Self-Attn L{layers}",
        output_directory=Path("Artifacts/GoalReach/unified-match-seed42"),
        vector_inputs=tuple(name for name in _UNIFIED_MATCH_ACTOR_VECTOR_INPUTS if include_remaining_time or name != "remaining_time"),
        critic_vector_inputs=tuple(name for name in _UNIFIED_MATCH_CRITIC_VECTOR_INPUTS if include_remaining_time or name != "remaining_time"),
        entity_sets=ENTITY_SET_ORDER,
        entity_set_feature_overrides=_UNIFIED_MATCH_ENTITY_OVERRIDES,
        critic_entity_set_feature_overrides=_UNIFIED_MATCH_CRITIC_ENTITY_OVERRIDES,
        entity_streams=streams,
        entity_attention_mode="typed_unified_cross" if streams == "unified" else "independent_cross",
        entity_self_attention_layers=layers,
        attention_embedding_size=32, attention_heads=1, body_hidden_size=128,
        defender_count=2, viewer_enabled=True, unified_match_contract=True,
        trainer_overrides=(),
    )


# Legacy callers use this only for policy participation; physical presence is separate.
active_roles = policy_roles

_UNIFIED_MATCH_VARIANTS = tuple(
    _unified_match_variant(streams=streams, layers=layers)
    for streams in ("separate", "unified") for layers in range(5)
)


_PATROL_EXPERIMENT_VARIANTS = (
    replace(
        _PATROL_PURSUIT_VARIANT,
        architecture_id=PATROL_MEAN_ATTENTION_RESIDUAL_ARCHITECTURE_ID,
        display_name="Patrol Pursuit · Mean Baseline + Attention",
        output_directory=Path(
            "Artifacts/GoalReach/patrol-pursuit-mean-attention-residual-seed42"
        ),
        entity_attention_mode="mean_attention_residual",
        viewer_enabled=False,
    ),
    replace(
        _PATROL_PURSUIT_VARIANT,
        architecture_id=PATROL_DIRECT_QKV_CROSS_ATTENTION_ARCHITECTURE_ID,
        display_name="Patrol Pursuit · Direct QKV Cross-Attention",
        output_directory=Path(
            "Artifacts/GoalReach/patrol-pursuit-direct-qkv-cross-attention-seed42"
        ),
        entity_attention_mode="direct_qkv_cross",
        viewer_enabled=False,
    ),
    replace(
        _PATROL_PURSUIT_VARIANT,
        architecture_id=PATROL_SAME_TYPE_SELF_THEN_CROSS_ATTENTION_ARCHITECTURE_ID,
        display_name="Patrol Pursuit · Self-Attention + Cross-Query",
        output_directory=Path(
            "Artifacts/GoalReach/"
            "patrol-pursuit-same-type-self-then-cross-attention-seed42"
        ),
        entity_attention_mode="same_type_self_then_cross",
        viewer_enabled=False,
    ),
    replace(
        _PATROL_PURSUIT_VARIANT,
        architecture_id=PATROL_TOKEN_MLP_THEN_CROSS_ATTENTION_ARCHITECTURE_ID,
        display_name="Patrol Pursuit · Entity MLP + Cross-Query",
        output_directory=Path(
            "Artifacts/GoalReach/patrol-pursuit-token-mlp-then-cross-attention-seed42"
        ),
        entity_attention_mode="token_mlp_then_cross",
        viewer_enabled=False,
    ),
    replace(
        _PATROL_PURSUIT_VARIANT,
        architecture_id=(
            PATROL_CROSS_TYPE_SOFTMAX_SELF_THEN_CROSS_ATTENTION_ARCHITECTURE_ID
        ),
        display_name="Patrol Pursuit · Cross-Type Softmax Attention",
        output_directory=Path(
            "Artifacts/GoalReach/"
            "patrol-pursuit-cross-type-softmax-self-then-cross-attention-seed42"
        ),
        entity_attention_mode="cross_type_softmax_self_then_cross",
        viewer_enabled=False,
    ),
)


_ACTOR_VARIANTS = _UNIFIED_MATCH_VARIANTS

_ACTOR_VARIANTS_BY_ID = {
    variant.architecture_id: variant for variant in _ACTOR_VARIANTS
}

DEFAULT_ACTOR_VARIANT_IDS: dict[ActorKind, str] = {
    "learner": UNIFIED_MATCH_ATTACKER_ARCHITECTURE_IDS[("separate", 0)],
    "defender": UNIFIED_MATCH_DEFENDER_ARCHITECTURE_IDS[("separate", 0)],
    "target": UNIFIED_MATCH_TARGET_ARCHITECTURE_IDS[("separate", 0)],
}

PATROL_VIEWER_ARCHITECTURE_IDS: dict[
    tuple[bool, EntityStreams, int], str
] = {
    **{
        (speed_input, "separate", layers): architecture_id
        for (speed_input, layers), architecture_id in (
            PATROL_SEPARATE_ATTENTION_ARCHITECTURE_IDS.items()
        )
    },
    **{
        (speed_input, "unified", layers): architecture_id
        for (speed_input, layers), architecture_id in (
            PATROL_TYPED_UNIFIED_ATTENTION_ARCHITECTURE_IDS.items()
        )
    },
}

MULTI_PATROL_VIEWER_ARCHITECTURE_IDS = MULTI_PATROL_ATTENTION_ARCHITECTURE_IDS


def actor_variants() -> tuple[ActorVariantSpec, ...]:
    return _ACTOR_VARIANTS


def actor_variant(architecture_id: str) -> ActorVariantSpec:
    if isinstance(architecture_id, str) and architecture_id.endswith("_no_time_actor_critic"):
        base = actor_variant(architecture_id.replace("_no_time_actor_critic", "_actor_critic"))
        if base.unified_match_contract:
            return _unified_match_variant(streams=base.entity_streams, layers=base.entity_self_attention_layers, include_remaining_time=False)
    try:
        return _ACTOR_VARIANTS_BY_ID[architecture_id]
    except KeyError as error:
        raise ValueError(f"Unsupported Actor architecture: {architecture_id}") from error


def patrol_defender_count_bounds(architecture_id: str) -> tuple[int, int]:
    variant = actor_variant(architecture_id)
    if not variant.unified_match_contract and variant.actor_kind != "defender":
        raise ValueError("Patrol count is only defined for defender architectures")
    return (
        (1, 10)
        if architecture_id in MULTI_PATROL_VIEWER_ARCHITECTURE_IDS.values()
        or variant.unified_match_contract
        else (1, 1)
    )


def actor_variants_for_kind(actor_kind: ActorKind) -> tuple[ActorVariantSpec, ...]:
    return tuple(
        variant for variant in _ACTOR_VARIANTS if variant.unified_match_contract or variant.actor_kind == actor_kind
    )


def actor_variant_ids(actor_kind: ActorKind | None = None) -> tuple[str, ...]:
    variants = (
        _ACTOR_VARIANTS
        if actor_kind is None
        else actor_variants_for_kind(actor_kind)
    )
    return tuple(variant.architecture_id for variant in variants)


def viewer_actor_variants() -> tuple[ActorVariantSpec, ...]:
    return tuple(variant for variant in _ACTOR_VARIANTS if variant.viewer_enabled)


def viewer_actor_variant_ids() -> tuple[str, ...]:
    return tuple(variant.architecture_id for variant in viewer_actor_variants())
