"""Current mobile-world Actor registry contract."""

import pytest

from auv_madrl.actor_variants import (
    DEFAULT_ACTOR_VARIANT_IDS,
    UNIFIED_MATCH_ARCHITECTURE_IDS,
    actor_variant,
    actor_variant_ids,
    actor_variants,
    actor_variants_for_kind,
    viewer_actor_variant_ids,
)


def test_registry_contains_only_the_ten_current_unified_match_variants() -> None:
    expected = tuple(
        UNIFIED_MATCH_ARCHITECTURE_IDS[(streams, layers)]
        for streams in ("separate", "unified")
        for layers in range(5)
    )
    assert actor_variant_ids() == expected
    assert viewer_actor_variant_ids() == expected
    assert tuple(variant.architecture_id for variant in actor_variants()) == expected
    for role in ("learner", "defender", "target"):
        assert actor_variant_ids(role) == expected
        assert actor_variants_for_kind(role) == actor_variants()
        assert DEFAULT_ACTOR_VARIANT_IDS[role] == expected[0]


def test_current_variants_share_role_agnostic_mobile_contract() -> None:
    for variant in actor_variants():
        assert variant.task_id == "goal_reach"
        assert variant.unified_match_contract
        assert variant.training_enabled and variant.viewer_enabled
        assert variant.entity_sets == ("target", "learner", "defender", "obstacles")
        assert variant.output_directory.as_posix() == "Artifacts/GoalReach/unified-match-seed42"
        assert len(variant.vector_inputs) == 10
        assert "is_reference_target" not in variant.vector_inputs
        assert "relative_x_over_L" in variant.entity_features_for("learner")
        assert len(variant.entity_features_for("learner")) == 8
        assert len(variant.critic_entity_features_for("learner")) == 9
        assert "can_move" not in variant.entity_features_for("learner")
        assert "is_reference_target" not in variant.entity_features_for("target")
        assert "alive" not in variant.entity_features_for("learner")
        assert "alive" in variant.critic_entity_features_for("learner")
        for entity_set in variant.entity_sets:
            if entity_set == "obstacles":
                assert "obstacle_entities" in variant.critic_observation_keys
                continue
            assert f"{entity_set[:-1] if entity_set.endswith('s') else entity_set}_critic_entities" in variant.observation_keys


@pytest.mark.parametrize("legacy", [
    "goal_reach_entity_attention_clipped_rollout32_p1024_mb4096_e5_actor_critic",
    "goal_reach_multi_learner_entity_attention_p1024_r32_mb4096_e5_friendly_motion_actor_critic",
    "patrol_pursuit_unified_speed_ratio_entity_attention_actor_critic",
    "patrol_pursuit_multi_shared_entity_attention_actor_critic",
])
def test_removed_architecture_id_is_rejected(legacy: str) -> None:
    with pytest.raises(ValueError, match="Unsupported Actor architecture"):
        actor_variant(legacy)
