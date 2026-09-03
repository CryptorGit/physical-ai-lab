from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

import pytest

from scripts import diagnose_forward_straight_quality as forward


ARTIFACTS = forward.EXP_ROOT / "artifacts"
CONCLUSION = ARTIFACTS / "forward_straight_retraining_required_conclusion_v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_segment() -> dict[str, object]:
    return {
        "completed": True,
        "fell": False,
        "completed_physics_substeps": 3000,
        "expected_physics_substeps": 3000,
        "metrics": {
            "projected_primary_velocity": 0.05,
            "signed_orthogonal_velocity": 0.010,
            "absolute_orthogonal_velocity": 0.010,
            "mean_local_yaw_rate": 0.05,
            "single_support_rate": 0.25,
            "flight_rate": 0.01,
        },
        "safety_audit": {
            "preclip_target_limit_violations": 0,
            "applied_target_limit_violations": 0,
            "unauthorized_applied_target_margin_violations": 0,
            "target_slew_violations": 0,
            "qpos_limit_violations": 0,
            "nonfinite_sample_count": 0,
        },
        "physics_substep_audit": {
            "sample_count": 3000,
            "qpos_limit_violations": 0,
            "nonfinite_full_qpos_samples": 0,
            "nonfinite_full_qvel_samples": 0,
            "nonfinite_leg_qpos_samples": 0,
            "nonfinite_pose_samples": 0,
            "nonfinite_state_samples": 0,
            "fall_or_nonfinite_detected": False,
        },
    }


def _strict_heading() -> dict[str, float]:
    return {
        "total_heading_change_rad": 0.15,
        "maximum_rolling_six_second_heading_change_rad": 0.15,
    }


def _strict_slip() -> dict[str, object]:
    return {
        "combined": {
            "tangential_speed_rms_mps": 0.015,
            "tangential_speed_p95_mps": 0.030,
            "maximum_integrated_slip_proxy_per_stance_m": 0.020,
        }
    }


def test_formal_seed_derivation_and_physical_command_are_pinned() -> None:
    assert forward.PHYSICAL_COMMAND == (0.05, 0.0, 0.0)
    runs = forward.stage_runs("formal")
    assert len(runs) == 20
    assert tuple(run["seed"] for run in runs) == tuple(
        20_260_809 + index * 1000 for index in range(20)
    )
    assert all(run["joint_noise_scale"] == 1.0 for run in runs)
    assert all(run["initial_base_speed"] == 0.10 for run in runs)


def test_candidate_validation_covers_route_local_knee_transform() -> None:
    forward.Candidate("valid", (0.10, -0.02, -0.17), 26.0, 0.6, -0.05).validate(27)
    with pytest.raises(ValueError, match="below phase_steps"):
        forward.Candidate("bad-phase", (0.10, 0.0, 0.0), 27.0).validate(27)
    with pytest.raises(ValueError, match="left-knee scale"):
        forward.Candidate(
            "bad-scale",
            (0.10, 0.0, 0.0),
            forward_left_knee_target_scale=0.0,
        ).validate(27)
    with pytest.raises(ValueError, match="left-knee bias"):
        forward.Candidate(
            "bad-bias",
            (0.10, 0.0, 0.0),
            forward_left_knee_target_bias_rad=-0.21,
        ).validate(27)


def test_candidate_loader_defaults_to_no_phase_or_knee_mutation(tmp_path) -> None:
    path = tmp_path / "candidate.json"
    path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "mapping-only",
                        "policy_observation_command": [0.10, -0.02, -0.17],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    candidate = forward.load_candidates(path, stage="screen")[0]
    assert candidate.forward_entry_preincrement_phase is None
    assert candidate.forward_left_knee_target_scale == 1.0
    assert candidate.forward_left_knee_target_bias_rad == 0.0


def test_strict_gate_boundaries_pass_with_all_substeps_audited() -> None:
    quality = forward.strict_quality(
        _strict_segment(),
        _strict_heading(),
        _strict_slip(),
        seconds=6.0,
        require_slip=True,
    )
    assert quality["passed"] is True
    assert all(quality["checks"].values())
    assert all(quality["safety_zero_checks"].values())
    assert all(quality["provisional_slip_checks"].values())


@pytest.mark.parametrize(
    ("field", "value", "failed_check"),
    (
        ("projected_primary_velocity", 0.03749, "vx_tracking"),
        ("projected_primary_velocity", 0.06251, "vx_tracking"),
        ("absolute_orthogonal_velocity", 0.01001, "cross_velocity"),
        ("mean_local_yaw_rate", 0.05001, "uncommanded_yaw_rate"),
        ("single_support_rate", 0.2499, "single_support_lower"),
        ("single_support_rate", 0.6001, "single_support_upper"),
        ("flight_rate", 0.0101, "flight_rate"),
    ),
)
def test_strict_motion_gates_reject_just_outside_values(
    field: str, value: float, failed_check: str
) -> None:
    segment = _strict_segment()
    segment["metrics"][field] = value  # type: ignore[index]
    quality = forward.strict_quality(
        segment,
        _strict_heading(),
        _strict_slip(),
        seconds=6.0,
        require_slip=False,
    )
    assert quality["passed"] is False
    assert quality["checks"][failed_check] is False


def test_substep_qpos_sample_is_a_hard_failure() -> None:
    segment = _strict_segment()
    segment["physics_substep_audit"]["qpos_limit_violations"] = 1  # type: ignore[index]
    quality = forward.strict_quality(
        segment,
        _strict_heading(),
        _strict_slip(),
        seconds=6.0,
        require_slip=False,
    )
    assert quality["passed"] is False
    assert quality["safety_zero_checks"]["substep_qpos_violations_zero"] is False


def test_wrapped_heading_delta_crosses_pi_continuously() -> None:
    assert forward.wrapped_delta(-math.pi + 0.01, math.pi - 0.01) == pytest.approx(0.02)


def test_retraining_conclusion_is_non_adopted_and_hardware_prohibited() -> None:
    conclusion = _load(CONCLUSION)
    assert conclusion["status"] == "RETRAINING_REQUIRED"
    assert conclusion["adoption_status"] == "NOT_ADOPTED"
    assert conclusion["hardware_deployment"] == "PROHIBITED"
    assert conclusion["central_sources_modified_by_this_pdca"] is False
    assert conclusion["package_modified"] is False
    assert conclusion["docs_modified"] is False
    assert conclusion["local_screen"]["non_reference_candidate_count"] == 23
    assert conclusion["local_screen"]["central_gait_quality_pass_count"] == 0
    assert conclusion["local_screen"]["all_hard_safety_checks_passed"] is True
    assert all(conclusion["qualification_not_run"].values())


def test_retraining_conclusion_pins_every_evidence_hash() -> None:
    conclusion = _load(CONCLUSION)
    for binding in conclusion["evidence"].values():
        path = (ARTIFACTS / binding["path"]).resolve()
        assert path.is_file()
        assert _sha256(path) == binding["sha256"]


def test_twenty_three_local_candidates_recompute_zero_quality_passes() -> None:
    conclusion = _load(CONCLUSION)
    sources = (
        ("forward_straight_slip_phase_screen_worst15s_v1.json", {"selected_no_phase_reset"}),
        (
            "forward_straight_slip_slew_screen_worst15s_v1.json",
            {"selected_reference", "phase_plus1_reference"},
        ),
        ("forward_straight_slip_clearance_screen_worst15s_v1.json", {"selected_reference"}),
    )
    records = []
    fixed_sha = conclusion["fixed_dependency_sha256"]["central_evaluator"]
    for filename, references in sources:
        payload = _load(ARTIFACTS / filename)
        assert payload["configuration"]["pre_run_central_sha256"] == fixed_sha
        assert payload["configuration"]["post_run_central_sha256"] == fixed_sha
        records.extend(
            record
            for record in payload["records"]
            if record["candidate"]["candidate_id"] not in references
        )
    assert len(records) == 23
    assert sum(record["strict_quality"]["passed"] for record in records) == 14
    assert not any(
        record["segment"]["gait_quality_acceptance"]["passed"] for record in records
    )
    assert all(
        all(record["strict_quality"]["safety_zero_checks"].values())
        for record in records
    )

    best = conclusion["local_screen"]["best_metrics"]
    assert min(
        record["segment"]["gait_quality_metrics"]["left_stance_slip_rms_mps"]
        for record in records
    ) == pytest.approx(best["minimum_left_stance_slip_rms_mps"]["value"])
    assert min(
        record["segment"]["gait_quality_metrics"]["stance_slip_rms_mps"]
        for record in records
    ) == pytest.approx(best["minimum_combined_stance_slip_rms_mps"]["value"])
    assert min(
        record["segment"]["gait_quality_metrics"]["stance_slip_p95_mps"]
        for record in records
    ) == pytest.approx(best["minimum_combined_stance_slip_p95_mps"]["value"])


def test_causal_trace_is_finite_and_phase_local() -> None:
    conclusion = _load(CONCLUSION)
    causal = _load(ARTIFACTS / conclusion["evidence"]["causal_trace"]["path"])
    assert causal["serialization_nonfinite_replacements"] == []
    comparisons = causal["slip_causal_outcome_comparisons"]
    assert all(
        row["selected_minus_baseline"]["stance_slip_rms_mps"] > 0.0
        and row["selected_minus_baseline"]["stance_slip_p95_mps"] > 0.0
        for row in comparisons.values()
    )
    selected = [
        analysis
        for key, analysis in causal["slip_causal_analyses"].items()
        if key.endswith("::selected_vy_m018_yaw_m170")
    ]
    assert len(selected) == 2
    assert all(
        analysis["feet"]["left"]["slip_energy_top_three_phase_bins"][0][
            "phase_bin"
        ]
        == 17
        for analysis in selected
    )
    assert all(
        analysis["feet"]["left"]["slip_energy_top_three_phase_bins"][0][
            "energy_fraction"
        ]
        >= 0.50
        for analysis in selected
    )
