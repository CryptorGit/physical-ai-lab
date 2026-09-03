"""Pure, fail-closed validation for the H4 forward retraining specification.

The validator deliberately imports only the Python standard library.  It does
not import MuJoCo, JAX, the trainer, or any runtime/evaluator module, and it
does not mutate the supplied mapping.  This keeps the curriculum contract
auditable before any expensive PPO process is authorized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC_PATH = (
    EXPERIMENT_ROOT
    / "artifacts"
    / "h4_forward_retraining_minimum_spec_from_slip_causality_v1.json"
)


class ForwardRetrainingSpecError(ValueError):
    """Raised when a forward retraining contract fails closed."""


def canonical_json_sha256(value: Any) -> str:
    """Return a deterministic digest without changing ``value``."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_spec(path: Path = DEFAULT_SPEC_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ForwardRetrainingSpecError("top-level specification must be an object")
    return value


def _at(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise ForwardRetrainingSpecError(f"missing required field: {path}")
        current = current[component]
    return current


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ForwardRetrainingSpecError(message)


def _isclose(actual: Any, expected: float, *, atol: float = 1.0e-12) -> bool:
    return isinstance(actual, (int, float)) and math.isfinite(float(actual)) and math.isclose(
        float(actual), expected, rel_tol=0.0, abs_tol=atol
    )


def _numeric_tree_is_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_numeric_tree_is_finite(item) for item in value.values())
    if isinstance(value, Sequence):
        return all(_numeric_tree_is_finite(item) for item in value)
    return False


def _require_vector(actual: Any, expected: Sequence[float], label: str) -> None:
    _require(isinstance(actual, list) and len(actual) == len(expected), f"{label} has wrong shape")
    _require(
        all(_isclose(item, float(want)) for item, want in zip(actual, expected)),
        f"{label} drifted from {list(expected)}",
    )


def validate_spec(spec: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate the immutable semantics and return named checks.

    The function is pure with respect to ``spec``: it performs no writes and
    never aliases or modifies nested containers.
    """

    _require(isinstance(spec, Mapping), "specification must be a mapping")
    _require(_numeric_tree_is_finite(spec), "specification contains non-finite or unsupported values")
    checks: list[str] = []

    _require(_at(spec, "schema_version") == 1, "schema_version must be 1")
    _require(
        _at(spec, "artifact_kind") == "openduckmini_h4_forward_retraining_minimum_spec",
        "artifact_kind drifted",
    )
    _require(_at(spec, "hardware_deployment") == "PROHIBITED", "hardware must remain prohibited")
    _require(_at(spec, "scope.new_ppo_run_performed") is False, "this artifact may not claim a PPO run")
    _require(_at(spec, "scope.central_runtime_or_evaluator_modified") is False, "central sources must remain untouched")
    checks.append("identity_and_scope")

    _require(
        _at(spec, "provenance.retraining_conclusion.sha256")
        == "2012f9266b0e4043a6f76497b649577e678e354a7c41b7caf20de08a70a32e0c",
        "retraining conclusion digest drifted",
    )
    _require(
        _at(spec, "provenance.primary_causal_evidence.sha256")
        == "e21c45be464f5ab45e53ef2f0d05de0cb20ab701516a35d66f847fcc47a13fce",
        "causal evidence digest drifted",
    )
    _require(
        _at(spec, "provenance.right_knee_scale_rejection.sha256")
        == "0952b3739179d9fe7f3325d92adc1eee5451ec4147912f3adb4f9ccdf8d88066",
        "right-knee rejection digest drifted",
    )
    checks.append("evidence_provenance")

    distribution = _at(spec, "curriculum.physical_command_distribution")
    modes = _at(distribution, "modes")
    _require(isinstance(modes, list) and len(modes) == 5, "curriculum must contain exactly five modes")
    by_name = {
        item.get("name"): item
        for item in modes
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    _require(len(by_name) == 5, "curriculum mode names must be unique strings")
    expected_probabilities = {
        "exact_primary_anchor": 0.60,
        "local_anchor_0p04": 0.10,
        "local_anchor_0p06": 0.10,
        "exact_stand": 0.10,
        "stand_to_forward_entry_band": 0.10,
    }
    for name, probability in expected_probabilities.items():
        _require(name in by_name, f"missing curriculum mode {name}")
        _require(_isclose(by_name[name].get("probability"), probability), f"{name} probability drifted")
    _require(
        _isclose(sum(float(item["probability"]) for item in modes), 1.0),
        "curriculum probabilities must sum to one",
    )
    _require(_isclose(by_name["exact_primary_anchor"].get("vx_mps"), 0.05), "0.05 anchor missing")
    _require(_isclose(by_name["local_anchor_0p04"].get("vx_mps"), 0.04), "0.04 anchor missing")
    _require(_isclose(by_name["local_anchor_0p06"].get("vx_mps"), 0.06), "0.06 anchor missing")
    _require(_isclose(by_name["exact_stand"].get("vx_mps"), 0.0), "stand must be exact zero")
    _require_vector(
        by_name["stand_to_forward_entry_band"].get("vx_uniform_mps"),
        (0.025, 0.04),
        "transition band",
    )
    _require(_isclose(distribution.get("vy_mps"), 0.0), "forward vy must be physically zero")
    _require(_isclose(distribution.get("yaw_radps"), 0.0), "forward yaw must be physically zero")
    continuity = _at(distribution, "transition_continuity")
    for field in (
        "phase_reset_on_command_change",
        "target_guard_state_reset_on_command_change",
        "contact_accumulator_reset_on_command_change",
    ):
        _require(continuity.get(field) is False, f"transition continuity forbids {field}")
    checks.append("physical_curriculum_60_percent_anchor")

    separation = _at(spec, "curriculum.command_separation_contract")
    _require_vector(separation.get("physical_anchor_mps_radps"), (0.05, 0.0, 0.0), "physical anchor")
    _require_vector(separation.get("policy_observation_anchor"), (0.10, -0.018, -0.170), "policy anchor")
    _require(
        separation.get("physical_anchor_mps_radps") != separation.get("policy_observation_anchor"),
        "physical and policy commands must remain separate",
    )
    _require(separation.get("stop_maps_to_exact_zero") is True, "stand must map to exact policy zero")
    _require(separation.get("physical_command_may_be_mutated_by_mapping") is False, "mapping may not mutate physical command")
    _require(separation.get("policy_compensation_is_physical_motion") is False, "policy compensation is not physical motion")
    expected_examples = {
        0.04: (0.08, -0.0144, -0.136),
        0.05: (0.10, -0.018, -0.170),
        0.06: (0.12, -0.0216, -0.204),
    }
    examples = separation.get("anchor_examples")
    _require(isinstance(examples, list) and len(examples) == 3, "three mapping examples are required")
    for example in examples:
        physical = example.get("physical_command")
        _require(isinstance(physical, list) and len(physical) == 3, "mapping example physical shape drifted")
        vx = float(physical[0])
        _require(vx in expected_examples, "unexpected physical anchor example")
        _require_vector(example.get("policy_observation_command"), expected_examples[vx], f"mapping at {vx}")
    checks.append("physical_policy_command_separation")

    runtime = _at(spec, "training_runtime_contract")
    _require(_isclose(runtime.get("control_period_s"), 0.02), "control period drifted")
    _require(_isclose(runtime.get("target_slew_limit_rad_per_s"), 2.0), "slew limit drifted")
    _require(_isclose(runtime.get("leg_target_envelope_margin_rad"), 0.05), "target margin drifted")
    _require(
        _isclose(
            float(runtime["control_period_s"]) * float(runtime["target_slew_limit_rad_per_s"]),
            float(runtime["maximum_applied_target_delta_per_control_rad"]),
        ),
        "per-tick target delta is inconsistent with dt and slew limit",
    )
    _require(runtime.get("head_action_indices_zero_locked") == [5, 6, 7, 8], "head lock drifted")
    _require(_at(runtime, "contact_contract.measurement") == "per-foot normal-force-weighted relative contact-point tangential velocity", "force slip measurement drifted")
    _require(_at(runtime, "contact_contract.world_site_finite_difference_allowed") is False, "site finite difference must remain forbidden")
    checks.append("runtime_guard_and_contact_alignment")

    causal = _at(spec, "causal_basis.phase_17_coupled_hotspot")
    _require(causal.get("phase_period_bins") == 27, "causal phase period drifted")
    _require(causal.get("primary_phase_index") == 17, "phase-17 hotspot missing")
    _require(causal.get("support_foot") == "left", "phase-17 support foot must be left")
    _require(causal.get("support_knee") == "left_knee", "phase-17 support knee must be left knee")
    _require(_isclose(causal.get("support_knee_target_envelope_upper_rad"), 0.425534), "left-knee envelope drifted")
    priorities = causal.get("opposite_leg_slew_lag_priority")
    _require(isinstance(priorities, list) and priorities[0] == "right_knee", "right knee must lead opposite-leg lag priority")
    _require(all(str(item).startswith("right_") for item in priorities), "opposite-leg lag list must be right-leg only")
    rejected = _at(spec, "causal_basis.rejected_single_joint_transform")
    _require(rejected.get("candidate_id") == "right_knee_scale070", "0.70 rejection evidence missing")
    _require(_isclose(rejected.get("right_knee_scale"), 0.70), "rejected right-knee scale drifted")
    _require(rejected.get("central_gait_quality_passed") is False, "failed 0.70 candidate may not be promoted")
    _require(float(rejected["stance_slip_rms_mps"]) > float(rejected["stance_slip_rms_gate_mps"]), "0.70 RMS evidence must fail")
    _require(float(rejected["stance_slip_p95_mps"]) > float(rejected["stance_slip_p95_gate_mps"]), "0.70 p95 evidence must fail")
    checks.append("phase17_coupled_causality_not_single_joint_scale")

    reward = _at(spec, "reward_specification")
    slip = _at(reward, "per_foot_force_slip")
    _require(_isclose(slip.get("strict_rms_mps"), 0.015), "slip RMS normalization drifted")
    _require(_isclose(slip.get("strict_p95_mps"), 0.030), "slip p95 normalization drifted")
    _require(_at(slip, "left_term.name") == "h4_left_force_slip", "left slip term missing")
    _require(_at(slip, "right_term.name") == "h4_right_force_slip", "right slip term missing")
    _require(float(_at(slip, "left_term.scale")) < 0.0 and float(_at(slip, "right_term.scale")) < 0.0, "per-foot slip scales must be costs")
    phase_terms = _at(reward, "phase_17_causal_terms")
    _require(_at(phase_terms, "support_knee_pre_guard_excess_term.joint") == "left_knee", "phase-17 clamp reward missing")
    weights = _at(phase_terms, "opposite_leg_lag_term.joints_and_weights")
    _require(set(weights) == {"right_knee", "right_hip_roll", "right_ankle", "right_hip_pitch"}, "opposite-leg lag weights drifted")
    _require(_isclose(sum(float(item) for item in weights.values()), 1.0), "opposite-leg lag weights must sum to one")
    _require(float(weights["right_knee"]) == max(float(item) for item in weights.values()), "right knee must have maximum lag weight")
    heading = _at(reward, "heading_and_cross")
    _require(_isclose(_at(heading, "cross_term.strict_boundary_mps"), 0.012), "cross boundary drifted")
    _require(_isclose(_at(heading, "yaw_rate_term.strict_boundary_radps"), 0.05), "yaw boundary drifted")
    _require(_isclose(_at(heading, "heading_term.strict_boundary_rad"), 0.15), "heading boundary drifted")
    support = _at(reward, "support_and_alternation")
    _require(_isclose(_at(support, "support_band_term.minimum_single_support_fraction"), 0.25), "support lower bound drifted")
    _require(_isclose(_at(support, "support_band_term.maximum_single_support_fraction"), 0.60), "support upper bound drifted")
    _require(float(_at(support, "alternation_term.scale")) > 0.0, "alternation must be a reward")
    _require(float(_at(support, "flight_term.scale")) < 0.0, "flight must be a cost")
    checks.append("exact_quality_reward_terms")

    fine_tune = _at(spec, "v22_preserving_fine_tune")
    _require(fine_tune.get("source_checkpoint_mode") == "READ_ONLY", "v22 source must be read-only")
    _require(fine_tune.get("in_place_checkpoint_write_allowed") is False, "in-place v22 writes are forbidden")
    initialization = _at(fine_tune, "initialization")
    _require(initialization.get("mode") == "explicit_v22_checkpoint_transplant", "explicit v22 transplant required")
    _require(initialization.get("copy_existing_actor_rows_exactly") is True, "old actor rows must be preserved")
    _require(initialization.get("new_first_layer_rows_initialization") == "EXACT_ZERO", "new rows must start at zero")
    _require(initialization.get("legacy_actor_observation_width") == 101, "legacy observation width drifted")
    _require(initialization.get("h4_actor_observation_width") == 116, "H4 observation width drifted")
    _require(initialization.get("new_h4_observation_rows") == 15, "H4 added observation rows drifted")
    pilot = _at(fine_tune, "recommended_250k_pilot")
    exact_pilot = {
        "num_timesteps": 250000,
        "num_envs": 1250,
        "unroll_length": 20,
        "batch_size": 125,
        "num_minibatches": 20,
        "num_updates_per_batch": 4,
        "expected_training_steps": 5,
        "expected_optimizer_updates": 400,
    }
    for field, expected in exact_pilot.items():
        _require(pilot.get(field) == expected, f"250k pilot field {field} drifted")
    _require(_isclose(pilot.get("learning_rate"), 0.00005), "pilot learning rate must be 5e-5")
    _require(float(pilot["learning_rate"]) <= float(pilot["maximum_learning_rate"]), "pilot learning rate exceeds cap")
    _require(float(pilot["maximum_learning_rate"]) <= 0.0001, "fine-tune learning-rate cap is too high")
    _require(pilot.get("overwrite_source_checkpoint") is False, "pilot may not overwrite v22")
    checks.append("v22_read_only_low_lr_250k_pilot")

    qualification = _at(spec, "qualification_plan")
    _require(qualification.get("causal_two_seed_15s") == [20261809, 20262809], "causal seeds drifted")
    _require(
        qualification.get("five_seed_15s")
        == [20260809, 20261809, 20262809, 20263809, 20264809],
        "five-seed qualification set drifted",
    )
    _require(qualification.get("mandatory_contact_sensitivity_windows_ms") == [10, 20, 30, 40], "contact sensitivity windows drifted")
    _require(_at(spec, "decision.ppo_training") == "NOT_RUN_BY_THIS_SPEC_TASK", "spec may not claim training")
    _require(_at(spec, "decision.hardware") == "PROHIBITED", "decision must prohibit hardware")
    checks.append("fail_closed_qualification")

    return tuple(checks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        spec = load_spec(args.spec)
        checks = validate_spec(spec)
    except (OSError, json.JSONDecodeError, ForwardRetrainingSpecError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "valid": True,
                "check_count": len(checks),
                "checks": checks,
                "canonical_spec_sha256": canonical_json_sha256(spec),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
