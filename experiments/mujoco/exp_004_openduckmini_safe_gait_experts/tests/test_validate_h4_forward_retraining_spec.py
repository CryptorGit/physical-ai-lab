from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from scripts import validate_h4_forward_retraining_spec as validator


@pytest.fixture()
def canonical_spec() -> dict:
    return validator.load_spec()


def test_canonical_forward_retraining_spec_passes_and_is_pure(canonical_spec: dict) -> None:
    before = copy.deepcopy(canonical_spec)

    checks = validator.validate_spec(canonical_spec)

    assert canonical_spec == before
    assert checks == (
        "identity_and_scope",
        "evidence_provenance",
        "physical_curriculum_60_percent_anchor",
        "physical_policy_command_separation",
        "runtime_guard_and_contact_alignment",
        "phase17_coupled_causality_not_single_joint_scale",
        "exact_quality_reward_terms",
        "v22_read_only_low_lr_250k_pilot",
        "fail_closed_qualification",
    )


def test_canonical_digest_is_deterministic_and_does_not_accept_nan(canonical_spec: dict) -> None:
    shuffled = json.loads(json.dumps(canonical_spec, sort_keys=True))
    assert validator.canonical_json_sha256(canonical_spec) == validator.canonical_json_sha256(shuffled)

    broken = copy.deepcopy(canonical_spec)
    broken["curriculum"]["physical_command_distribution"]["probability_sum"] = float("nan")
    with pytest.raises(validator.ForwardRetrainingSpecError, match="non-finite"):
        validator.validate_spec(broken)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("primary_probability", 0.59, "exact_primary_anchor probability drifted"),
        ("transition_reset", True, "transition continuity forbids phase_reset"),
        ("physical_mapping_mutation", True, "mapping may not mutate physical command"),
        ("right_knee_pass", True, "failed 0.70 candidate may not be promoted"),
        ("v22_write", True, "in-place v22 writes are forbidden"),
        ("pilot_learning_rate", 0.0003, "pilot learning rate must be 5e-5"),
    ],
)
def test_fail_closed_mutations(
    canonical_spec: dict, field: str, value: object, message: str
) -> None:
    broken = copy.deepcopy(canonical_spec)
    if field == "primary_probability":
        broken["curriculum"]["physical_command_distribution"]["modes"][0][
            "probability"
        ] = value
    elif field == "transition_reset":
        broken["curriculum"]["physical_command_distribution"][
            "transition_continuity"
        ]["phase_reset_on_command_change"] = value
    elif field == "physical_mapping_mutation":
        broken["curriculum"]["command_separation_contract"][
            "physical_command_may_be_mutated_by_mapping"
        ] = value
    elif field == "right_knee_pass":
        broken["causal_basis"]["rejected_single_joint_transform"][
            "central_gait_quality_passed"
        ] = value
    elif field == "v22_write":
        broken["v22_preserving_fine_tune"]["in_place_checkpoint_write_allowed"] = value
    elif field == "pilot_learning_rate":
        broken["v22_preserving_fine_tune"]["recommended_250k_pilot"][
            "learning_rate"
        ] = value
    else:  # pragma: no cover - guarded by parameterization
        raise AssertionError(field)

    with pytest.raises(validator.ForwardRetrainingSpecError, match=message):
        validator.validate_spec(broken)


def test_physical_and_policy_anchors_cannot_be_aliased(canonical_spec: dict) -> None:
    broken = copy.deepcopy(canonical_spec)
    separation = broken["curriculum"]["command_separation_contract"]
    separation["policy_observation_anchor"] = separation["physical_anchor_mps_radps"]

    with pytest.raises(validator.ForwardRetrainingSpecError, match="policy anchor drifted"):
        validator.validate_spec(broken)


def test_phase17_requires_support_clamp_and_right_leg_lag(canonical_spec: dict) -> None:
    broken = copy.deepcopy(canonical_spec)
    del broken["reward_specification"]["phase_17_causal_terms"][
        "opposite_leg_lag_term"
    ]["joints_and_weights"]["right_knee"]

    with pytest.raises(validator.ForwardRetrainingSpecError, match="lag weights drifted"):
        validator.validate_spec(broken)


def test_left_and_right_force_slip_terms_are_both_mandatory(canonical_spec: dict) -> None:
    broken = copy.deepcopy(canonical_spec)
    del broken["reward_specification"]["per_foot_force_slip"]["right_term"]

    with pytest.raises(validator.ForwardRetrainingSpecError, match="missing required field"):
        validator.validate_spec(broken)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("h4_actor_observation_width", 114, "H4 observation width drifted"),
        ("new_h4_observation_rows", 13, "H4 added observation rows drifted"),
    ],
)
def test_h4_observation_contract_requires_all_fifteen_new_rows(
    canonical_spec: dict, field: str, value: int, message: str
) -> None:
    broken = copy.deepcopy(canonical_spec)
    broken["v22_preserving_fine_tune"]["initialization"][field] = value

    with pytest.raises(validator.ForwardRetrainingSpecError, match=message):
        validator.validate_spec(broken)


def test_cli_reports_valid_contract(capsys: pytest.CaptureFixture[str]) -> None:
    assert validator.main([]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["check_count"] == 9
    assert len(payload["canonical_spec_sha256"]) == 64
