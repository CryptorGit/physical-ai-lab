import json

import numpy as np
import pytest
from types import SimpleNamespace
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from safe_gait_experts.h4_training_alignment import (
    reverse_iteration_v6_absolute_full_leg_targets,
)
from safe_gait_experts.h4_post_training import sha256_file
from safe_gait_experts.h5_target_contract import (
    H5_ACTION_WIDTH,
    H5_HEAD_INDICES,
    H5TransitionState,
    h5_blend_targets,
    h5_contract_manifest,
    h5_decode_absolute_targets,
    h5_decode_and_blend,
    h5_final_guard_step,
    h5_reverse_entry_state,
)
from safe_gait_experts.h5_command_contract import (
    H5_UNIFIED_COMMAND_CONTRACT_V2_ID,
    H5_UNIFIED_COMMAND_CONTRACT_V3_ID,
    canonical_h5_unified_command_mapper,
    h5_unified_direct_policy_command,
    h5_unified_direct_policy_command_xp,
    h5_unified_command_contract_id,
    h5_unified_command_contract_manifest,
    h5_unified_policy_command,
)
from safe_gait_experts.h5_training_alignment import make_h5_unified_command_mapper
from safe_gait_experts.h5_routed_policy import (
    H5DomainCandidate,
    H5RoutedPolicyBank,
)
from scripts.evaluate_h5_routed_transitions import (
    H5RoutedSimulator,
    _load_training_command_provenance,
)


def test_h5_decoder_is_exactly_the_canonical_safe_decoder_for_both_domains():
    rng = np.random.default_rng(20260811)
    for domain in ("planar", "reverse"):
        for _ in range(128):
            action = rng.uniform(-1.25, 1.25, size=H5_ACTION_WIDTH)
            expected = reverse_iteration_v6_absolute_full_leg_targets(action)
            actual = h5_decode_absolute_targets(action, domain=domain)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(actual[list(H5_HEAD_INDICES)], 0.0)


def test_h5_blends_decoded_targets_not_raw_actions():
    source_action = np.linspace(-1.0, 1.0, H5_ACTION_WIDTH)
    target_action = np.linspace(1.0, -1.0, H5_ACTION_WIDTH)
    actual = h5_decode_and_blend(
        source_action,
        target_action,
        from_domain="planar",
        to_domain="reverse",
        alpha=0.35,
    )
    expected = h5_blend_targets(
        h5_decode_absolute_targets(source_action, domain="planar"),
        h5_decode_absolute_targets(target_action, domain="reverse"),
        0.35,
    )
    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(
        actual,
        (1.0 - 0.35) * source_action + 0.35 * target_action,
    )


def test_h5_guard_is_single_safe_stateful_step_and_keeps_heads_zero():
    previous = np.asarray(
        reverse_iteration_v6_absolute_full_leg_targets(
            np.zeros(H5_ACTION_WIDTH)
        )
    )
    desired = h5_decode_absolute_targets(np.ones(H5_ACTION_WIDTH), domain="planar")
    applied = h5_final_guard_step(desired, previous)
    assert applied.shape == (H5_ACTION_WIDTH,)
    np.testing.assert_array_equal(applied[list(H5_HEAD_INDICES)], 0.0)
    assert np.max(np.abs(applied - previous)) <= 0.040000000001


def test_h5_reverse_entry_preserves_target_and_contact_history_but_sets_phase_7():
    previous = H5TransitionState(
        phase_index=23.0,
        active_domain="planar",
        previous_applied_targets=tuple(
            reverse_iteration_v6_absolute_full_leg_targets(
                np.zeros(H5_ACTION_WIDTH)
            )
        ),
        contact_continuity={"left": True, "right": False},
    )
    entered = h5_reverse_entry_state(previous)
    assert entered.phase_index == 7.0
    assert entered.active_domain == "reverse"
    assert entered.previous_applied_targets == previous.previous_applied_targets
    assert entered.contact_continuity == previous.contact_continuity


def test_h5_contract_metadata_is_explicit_and_hardware_prohibited():
    manifest = h5_contract_manifest()
    assert manifest["actor_observation_width"] == 116
    assert manifest["action_width"] == 14
    assert manifest["composition_order"][2] == "blend_target_space"
    assert manifest["reverse_first_observation_phase_index"] == 7.0
    assert manifest["hardware_deployment"] == "PROHIBITED"


def test_direct_unified_command_mapper_preserves_pure_axis_commands():
    direct_forward = h5_unified_direct_policy_command((0.05, 0.0, 0.0))
    legacy_forward = h5_unified_policy_command((0.05, 0.0, 0.0))
    np.testing.assert_array_equal(direct_forward, np.asarray((0.10, 0.0, 0.0)))
    np.testing.assert_array_equal(
        legacy_forward, np.asarray((0.10, -0.018, -0.170))
    )
    np.testing.assert_array_equal(
        h5_unified_direct_policy_command((0.0, 0.06, 0.0)),
        np.asarray((0.0, 0.10, 0.0)),
    )


def test_direct_unified_command_mapper_xp_matches_numpy_contract():
    command = np.asarray((0.04, -0.03, -0.15, 0.0, 0.0, 0.0, 0.0))
    expected = np.asarray((0.08, -0.05, -0.30, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_array_equal(
        h5_unified_direct_policy_command_xp(command, xp=np), expected
    )


def test_direct_unified_mapper_has_a_separate_immutable_v3_contract():
    assert canonical_h5_unified_command_mapper("direct_normalized") == (
        "direct_normalized_v3"
    )
    assert h5_unified_command_contract_id("legacy_h4_compensated") == (
        H5_UNIFIED_COMMAND_CONTRACT_V2_ID
    )
    assert h5_unified_command_contract_id("direct_normalized_v3") == (
        H5_UNIFIED_COMMAND_CONTRACT_V3_ID
    )
    direct = h5_unified_command_contract_manifest("direct_normalized_v3")
    assert direct["contract_id"] == H5_UNIFIED_COMMAND_CONTRACT_V3_ID
    assert direct["axis_separable"] is True
    assert direct["positive_vx_cross_axis_compensation"] == [0.0, 0.0]


def test_training_command_mapper_requires_explicit_direct_mode():
    command = np.asarray((0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    direct = make_h5_unified_command_mapper(
        None, np, mapper_mode="direct_normalized_v3"
    )
    legacy = make_h5_unified_command_mapper(None, np)
    np.testing.assert_array_equal(
        direct(command), np.asarray((0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    )
    np.testing.assert_array_equal(
        legacy(command), np.asarray((0.10, -0.018, -0.170, 0.0, 0.0, 0.0, 0.0))
    )
    with pytest.raises(ValueError, match="unsupported H5 unified command mapper"):
        make_h5_unified_command_mapper(None, np, mapper_mode="unknown")


@pytest.mark.parametrize(
    ("mapper", "contract_id", "expected_mapper", "inferred"),
    (
        (
            "direct_normalized_v3",
            H5_UNIFIED_COMMAND_CONTRACT_V3_ID,
            "direct_normalized_v3",
            False,
        ),
        (None, H5_UNIFIED_COMMAND_CONTRACT_V2_ID, "legacy_h4_compensated", True),
    ),
)
def test_h5_candidate_training_provenance_binds_contract_and_legacy_migration(
    tmp_path, mapper, contract_id, expected_mapper, inferred
):
    params_path = tmp_path / "final_params.pkl"
    params_path.write_bytes(b"test-actor-parameters")
    params_sha = sha256_file(params_path)
    config = {
        "h5_command_contract_id": contract_id,
        "h5_command_contract": {"contract_id": contract_id},
    }
    if mapper is not None:
        config["h5_unified_command_mapper"] = mapper
    config_path = tmp_path / "resolved_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    training_manifest = {
        "status": "COMPLETED",
        "expert": "unified",
        "hardware_deployment": "PROHIBITED",
        "qualification_use": "DIAGNOSTIC_REWARD_EXPLORATION_NOT_QUALIFICATION",
        "outputs": {
            "final_params": {"path": str(params_path), "sha256": params_sha}
        },
        "resolved_config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
    }
    training_path = tmp_path / "run_manifest.json"
    training_path.write_text(json.dumps(training_manifest), encoding="utf-8")
    provenance = _load_training_command_provenance(
        {"source_manifest": {"path": str(training_path), "sha256": sha256_file(training_path)}},
        params_path=params_path.resolve(),
        params_sha256=params_sha,
        domain="planar",
    )
    assert provenance["training_command_contract_id"] == contract_id
    assert provenance["training_command_mapper"] == expected_mapper
    assert provenance["training_command_mapper_inferred_from_v2_contract"] is inferred


def test_h5_reverse_policy_target_is_actor_authoritative():
    simulator = object.__new__(H5RoutedSimulator)
    desired = np.linspace(-0.2, 0.2, H5_ACTION_WIDTH)
    simulator.bank = SimpleNamespace(
        last_step=SimpleNamespace(blended_targets=tuple(desired))
    )
    actual = simulator._policy_target(
        np.zeros(H5_ACTION_WIDTH),
        np.asarray((-0.05, 0.0, 0.0)),
        7.0,
        np.zeros(H5_ACTION_WIDTH),
    )
    np.testing.assert_array_equal(actual, desired)


def test_h5_bank_is_fail_closed_without_explicit_legacy_fallback():
    class LegacyBank:
        def __init__(self):
            self.calls = 0

        def infer(self, role, observation):
            del role, observation
            self.calls += 1
            return np.zeros(H5_ACTION_WIDTH)

    legacy = LegacyBank()
    bank = H5RoutedPolicyBank(
        legacy,
        {
            "planar": H5DomainCandidate("planar", object()),
            "reverse": H5DomainCandidate("reverse", object()),
        },
    )
    bank.infer = lambda role, observation: (_ for _ in ()).throw(
        ValueError("synthetic actor failure")
    )
    with pytest.raises(ValueError, match="synthetic actor failure"):
        bank.infer_or_legacy("forward", np.zeros(116))
    assert legacy.calls == 0
    assert bank.legacy_fallback_count == 0
    assert bank.manifest()["legacy_fallback"]["strict_fail_closed"] is True


def test_h5_legacy_fallback_is_counted_only_when_explicitly_enabled():
    class LegacyBank:
        def infer(self, role, observation):
            del role, observation
            return np.zeros(H5_ACTION_WIDTH)

    bank = H5RoutedPolicyBank(
        LegacyBank(),
        {
            "planar": H5DomainCandidate("planar", object()),
            "reverse": H5DomainCandidate("reverse", object()),
        },
        allow_legacy_fallback=True,
    )
    bank.infer = lambda role, observation: (_ for _ in ()).throw(
        ValueError("synthetic actor failure")
    )
    np.testing.assert_array_equal(
        bank.infer_or_legacy("forward", np.zeros(116)),
        np.zeros(H5_ACTION_WIDTH),
    )
    assert bank.legacy_fallback_count == 1
    assert bank.manifest()["legacy_fallback"]["roles"] == {"forward": 1}


@pytest.mark.parametrize("alpha", (-0.01, 1.01, np.nan))
def test_h5_blend_rejects_invalid_alpha(alpha):
    values = np.zeros(H5_ACTION_WIDTH)
    with pytest.raises(ValueError):
        h5_blend_targets(values, values, alpha)
