"""Validate the simulation-only H4 reverse training-composition authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


EXP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    EXP_ROOT / "artifacts" / "h4_reverse_training_composition_authorization_v1.json"
)


class ReverseTrainingCompositionError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_strict(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ReverseTrainingCompositionError(f"non-finite JSON constant: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReverseTrainingCompositionError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReverseTrainingCompositionError(message)


def validate_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    resolved = path.resolve()
    contract = load_json_strict(resolved)
    _require(contract.get("schema_version") == 1, "schema drifted")
    _require(
        contract.get("artifact_kind")
        == "openduckmini_h4_reverse_training_composition_authorization",
        "artifact kind drifted",
    )
    _require(
        contract.get("status")
        == "SIMULATION_TRAINING_COMPOSITION_AUTHORIZED_NOT_ADOPTED",
        "status drifted",
    )
    _require(contract.get("hardware_deployment") == "PROHIBITED", "hardware must be prohibited")
    _require(contract.get("simulation_adoption_allowed") is False, "adoption must remain false")
    _require(contract.get("release_allowed") is False, "release must remain false")

    scope = contract["scope"]
    _require(scope.get("physical_command_mps_radps") == [-0.05, 0.0, 0.0], "physical command drifted")
    _require(scope.get("maximum_pilot_interactions") == 250000, "pilot cap drifted")
    _require(scope.get("actor_observation_width") == 116, "actor width drifted")

    components = contract["pinned_components"]
    component_audit: dict[str, dict[str, Any]] = {}
    for name, item in components.items():
        component_path = (EXP_ROOT / item["path"]).resolve()
        _require(component_path.is_file(), f"missing component: {name}")
        actual = sha256_file(component_path)
        _require(actual == item["sha256"], f"component SHA drifted: {name}")
        component_audit[name] = {"path": str(component_path), "sha256": actual}

    selected = load_json_strict(Path(component_audit["selected_teacher"]["path"]))
    _require(selected.get("decision", {}).get("adoption") is False, "selected teacher adoption drifted")
    _require(selected.get("decision", {}).get("direct_runtime_use") == "PROHIBITED", "standalone runtime prohibition drifted")
    _require(selected.get("teacher", {}).get("candidate_id") == "cbe8decf6a7c4e5e", "candidate id drifted")
    _require(selected.get("teacher", {}).get("validation", {}).get("passed") is True, "teacher pure validation failed")

    composition = contract["composition_contract"]
    exact = {
        "training_use": "PERSISTENT_DETERMINISTIC_BASELINE_PLUS_TRAINABLE_RESIDUAL",
        "standalone_teacher_direct_runtime_use": "PROHIBITED",
        "candidate_evaluation_use": "ALLOWED_ONLY_WITH_IDENTICAL_PINNED_COMPOSITION",
        "candidate_runtime_adoption": "PROHIBITED_UNTIL_STRICT_QUALIFICATION",
        "teacher_phase_steps": 54,
        "teacher_cadence_hz": 1.5,
        "teacher_phase_advance_bins_per_control": 1.62,
        "teacher_entry_phase_preincrement_bins": 14.0,
        "first_reference_phase_after_increment_bins": 15.620000000000001,
        "control_period_s": 0.02,
        "maximum_residual_scale": 0.12,
        "maximum_training_visible_target_delta_rad_per_control": 0.04,
        "head_targets_exactly_zero": True,
    }
    for key, expected in exact.items():
        _require(composition.get(key) == expected, f"composition field drifted: {key}")
    _require(len(composition.get("guard_order", [])) == 5, "guard order drifted")

    authorization = contract["authorization"]
    required_auth = {
        "simulation_wiring_training": True,
        "simulation_250k_pilot_training": True,
        "simulation_1m_training": False,
        "candidate_strict_evaluation": True,
        "candidate_adoption": False,
        "package_release": False,
        "hardware": False,
    }
    for key, expected in required_auth.items():
        _require(authorization.get(key) is expected, f"authorization drifted: {key}")
    boundary = contract["supersession_boundary"]
    _require(
        boundary.get("supersedes_selected_teacher_training_use_text")
        == "ALLOWED_AS_INITIALIZATION_PRIOR_ONLY",
        "supersession source drifted",
    )
    _require(len(boundary.get("does_not_supersede", [])) == 5, "prohibitions drifted")
    return {
        "valid": True,
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "component_audit": component_audit,
        "hardware_deployment": "PROHIBITED",
        "simulation_250k_pilot_training": True,
        "simulation_1m_training": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args(argv)
    print(json.dumps(validate_contract(args.contract), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
