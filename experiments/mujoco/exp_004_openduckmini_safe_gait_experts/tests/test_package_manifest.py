from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import package_manifest as package_manifest_module
from package_manifest import (
    BASE_V22_MODEL_ID,
    CONTROL_FIRST_STARTUP_DT_S,
    FORMAL_EVALUATOR_ID,
    FORMAL_RELEASE_EVIDENCE_SHA256,
    FORMAL_RELEASE_EVIDENCE_SHA256_ALLOWLIST,
    FORMAL_RELEASE_EVIDENCE_SIZE_BYTES,
    FORMAL_RELEASE_MASTER_SEED,
    FORMAL_RELEASE_SCALE,
    FORMAL_RELEASE_STATUS,
    PERTURBED_RESET_QPOS_MARGIN_RAD,
    REJECTED_LINEAGES,
    REVERSE_MODEL_ID,
    REVERSE_PROFILE_RELEASE_ID,
    REVERSE_RESIDUAL_SCALE,
    RUNTIME_MAX_TARGET_DELTA_PER_TICK_RAD,
    RUNTIME_TARGET_SAFETY_MARGIN_RAD,
    RUNTIME_TARGET_SLEW_RATE_RAD_S,
    YAW_RIGHT_POLICY_OFFSET,
    build_router_package,
    load_and_validate_formal_release_evidence,
    load_and_validate_package,
    sha256_file,
)
from router import (
    ALLOWED_EXPERTS,
    DEFAULT_COMMAND_MAX,
    DEFAULT_COMMAND_MIN,
    REVERSE,
    REVERSE_TURN_ENDPOINTS,
    REVERSE_TURN_LEFT,
    REVERSE_TURN_RIGHT,
    YAW_RIGHT,
)
from safe_gait_experts.contract import (
    CONTRACT,
    validate_contract as validate_frozen_contract,
)
from scripts.build_router_package import build_parser


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
POLICY_ROLES = {
    "stand",
    "forward",
    "reverse",
    "lateral_left",
    "lateral_right",
    "yaw_left",
    "yaw_right",
    "compound",
}


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _sources(tmp_path: Path) -> dict[str, Path | str]:
    source = tmp_path / "sources"
    base = _write(source / "base_v22.onnx", b"test base v22\n")
    artifacts = EXPERIMENT_ROOT / "artifacts"
    generated_data = (
        artifacts
        / "generated_playground"
        / "playground"
        / "open_duck_mini_v2"
        / "data"
    )
    generated_xmls = generated_data.parent / "xmls"
    return {
        "package_id": "test-safe-router-v1",
        "base_v22_onnx": base,
        "scene": (
            generated_xmls
            / "scene_flat_terrain_backlash_hardware_safe_calibrated.xml"
        ),
        "reference": _write(source / "reference.pkl", b"reference\n"),
        "reverse_profile": (
            artifacts
            / "optimized_reverse_margin050_slew200_h1_phase7_rate105_candidate_v1.json"
        ),
        "reverse_turn_left_profile": (
            artifacts
            / "reverse_turn_candidates_v1"
            / "optimized_reverse_turn_left_margin050_slew200_candidate_v1.json"
        ),
        "reverse_turn_right_profile": (
            artifacts
            / "reverse_turn_candidates_v1"
            / "optimized_reverse_turn_right_margin050_slew200_candidate_v1.json"
        ),
        "router_source": EXPERIMENT_ROOT / "router.py",
        "target_safety_source": EXPERIMENT_ROOT / "target_safety.py",
        "contract_source": EXPERIMENT_ROOT / "contract.json",
        "expected_base_v22_sha256": sha256_file(base),
    }


def _production_sources() -> dict[str, Path | str]:
    artifacts = EXPERIMENT_ROOT / "artifacts"
    generated_robot = (
        artifacts
        / "generated_playground"
        / "playground"
        / "open_duck_mini_v2"
    )
    base = (
        EXPERIMENT_ROOT.parents[2]
        / ".openduck_runtime_source_review"
        / "calibrated_hybrid_policy_v22.onnx"
    )
    return {
        "package_id": "test-h3-release-router-v1",
        "base_v22_onnx": base,
        "scene": (
            generated_robot
            / "xmls"
            / "scene_flat_terrain_backlash_hardware_safe_calibrated.xml"
        ),
        "reference": (
            generated_robot
            / "data"
            / "polynomial_coefficients_calibrated.pkl"
        ),
        "reverse_profile": (
            artifacts
            / "optimized_reverse_margin050_slew200_h1_phase7_rate105_candidate_v1.json"
        ),
        "reverse_turn_left_profile": (
            artifacts
            / "reverse_turn_candidates_v1"
            / "optimized_reverse_turn_left_margin050_slew200_candidate_v1.json"
        ),
        "reverse_turn_right_profile": (
            artifacts
            / "reverse_turn_candidates_v1"
            / "optimized_reverse_turn_right_margin050_slew200_candidate_v1.json"
        ),
        "router_source": EXPERIMENT_ROOT / "router.py",
        "target_safety_source": EXPERIMENT_ROOT / "target_safety.py",
        "contract_source": EXPERIMENT_ROOT / "contract.json",
        "expected_base_v22_sha256": sha256_file(base),
    }


def _bindings(arguments: dict[str, Path | str]) -> dict[str, object]:
    return {
        "base": str(arguments["expected_base_v22_sha256"]),
        "profiles": {
            "straight": sha256_file(Path(arguments["reverse_profile"])),
            "left": sha256_file(Path(arguments["reverse_turn_left_profile"])),
            "right": sha256_file(Path(arguments["reverse_turn_right_profile"])),
        },
        "scene": sha256_file(Path(arguments["scene"])),
        "reference": sha256_file(Path(arguments["reference"])),
    }


def _formal_payload(bindings: dict[str, object]) -> dict[str, object]:
    base_hash = str(bindings["base"])
    profile_hashes = dict(bindings["profiles"])
    return {
        "schema_version": 1,
        "evaluator_id": FORMAL_EVALUATOR_ID,
        "evaluation_mode": "RELEASE_QUALIFICATION",
        "release_qualification": {
            "status": "RELEASE_QUALIFICATION",
            "release_qualification_eligible": True,
            "scale_matches_frozen_contract": True,
            "diagnostic_mode_disabled": True,
            "master_seed_matches_recommendation": True,
            "expected": {"recommended_master_seed": FORMAL_RELEASE_MASTER_SEED},
            "actual": dict(FORMAL_RELEASE_SCALE),
        },
        "configuration": {"seed": FORMAL_RELEASE_MASTER_SEED},
        "simulation_suite_acceptance_passed": True,
        "simulation_acceptance_passed": True,
        "adoption_contract": {"passed": True},
        "command_mapping_contract": {
            "validation_status_gate": {"passed": True}
        },
        "suites": {
            name: {"acceptance": {"passed": True}}
            for name in ("primitives", "compounds", "transitions")
        },
        "runtime_dependency_provenance": {
            "verified": True,
            "pre_post_source_and_data_hashes_unchanged": True,
            "all_onnx_sessions_cpu_only_verified": True,
        },
        "policy_provenance": {
            "mode": "FORMAL_BASE_V22_ONLY",
            "adoption_eligible": True,
            "all_roles_allowlisted": True,
            "diagnostic_unadopted": False,
            "roles": {
                role: {
                    "sha256": base_hash,
                    "formal_base_v22_allowlisted": True,
                    "adopted": True,
                }
                for role in POLICY_ROLES
            },
        },
        "reverse_profile_adoption": {
            "status": FORMAL_RELEASE_STATUS,
            "passed": True,
            "roles": {
                role: {
                    "profile_sha256": profile_hashes[role],
                    "profile_hash_allowlisted": True,
                    "evidence_hash_allowlisted": True,
                    "status_nonblocked": True,
                    "passed": True,
                }
                for role in ("straight", "left", "right")
            },
        },
        "exact_hardware_safe_assets": {
            "real_hardware_deployment_allowed": False,
            "verified_files": {
                "scene": {"sha256": bindings["scene"]},
                "reference": {"sha256": bindings["reference"]},
            },
        },
        "formal_reverse_phase_entry_contract": {
            "status": FORMAL_RELEASE_STATUS,
            "enabled_by_default": True,
            "diagnostic_only": False,
            "adoption_eligible": True,
            "current_endpoint_requalified": True,
            "hardware_deployment": "PROHIBITED",
        },
        "formal_backward_exit_recovery_contract": {
            "status": FORMAL_RELEASE_STATUS,
            "enabled_by_default": True,
            "diagnostic_unadopted_only": False,
            "adoption_eligible": True,
            "simulation_acceptance_eligible": True,
            "hardware_deployment": "PROHIBITED",
            "runtime_contract": {
                "extra_upper_margin_rad": 0.0225,
                "upper_target_rad": 0.403034,
                "hold_control_ticks": 13,
                "hold_seconds": 0.26,
                "hardware_deployment": "PROHIBITED",
            },
            "execution_bundle_binding": {
                "passed": True,
                "profile_sha256s": profile_hashes,
                "policy_sha256": base_hash,
                "policy_roles": sorted(POLICY_ROLES),
            },
        },
        "hardware_gate": {
            "status": "PROHIBITED",
            "hardware_deployment_allowed": False,
            "simulation_pass_does_not_promote_hardware": True,
        },
    }


def _write_allowlisted_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
) -> Path:
    path = tmp_path / "formal_release_evidence.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        package_manifest_module,
        "FORMAL_RELEASE_EVIDENCE_SHA256_ALLOWLIST",
        frozenset({sha256_file(path)}),
    )
    return path


def _validate(
    path: Path, bindings: dict[str, object]
) -> dict[str, object]:
    return load_and_validate_formal_release_evidence(
        path,
        expected_base_v22_sha256=str(bindings["base"]),
        expected_profile_sha256=dict(bindings["profiles"]),
        expected_scene_sha256=str(bindings["scene"]),
        expected_reference_sha256=str(bindings["reference"]),
    )


def _promoted_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path | str]:
    """Create an explicit test-only future-adoption fixture.

    Production constants and the checked-in contract remain fail-closed.  This
    fixture changes all independent promotion gates together so validator
    coverage does not require weakening the public builder.
    """

    arguments = _sources(tmp_path)
    bindings = _bindings(arguments)
    evidence = _write_allowlisted_evidence(
        tmp_path, monkeypatch, _formal_payload(bindings)
    )
    promoted_contract = copy.deepcopy(CONTRACT)
    promoted_recovery = promoted_contract["target_safety"][
        "backward_exit_recovery"
    ]
    promoted_recovery["status"] = FORMAL_RELEASE_STATUS
    promoted_recovery["enabled_by_default"] = True
    promoted_recovery["formal_candidate_only"] = False
    promoted_recovery["diagnostic_unadopted_only"] = False
    promoted_recovery["adoption_eligible"] = True
    promoted_recovery["simulation_acceptance_eligible"] = True
    promoted_recovery["safety_component_only"] = False
    promoted_recovery["combined_5x15_required"] = False
    promoted_recovery["requires_formal_20x30_requalification"] = False
    promoted_contract_path = tmp_path / "sources" / "promoted_contract.json"
    promoted_contract_path.write_text(
        json.dumps(promoted_contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def validate_test_promoted_contract(candidate: dict[str, object]) -> None:
        recovery = candidate["target_safety"]["backward_exit_recovery"]
        if (
            recovery.get("status") != FORMAL_RELEASE_STATUS
            or recovery.get("enabled_by_default") is not True
            or recovery.get("diagnostic_unadopted_only") is not False
        ):
            raise ValueError("test promoted recovery contract mismatch")
        frozen_equivalent = copy.deepcopy(candidate)
        frozen_equivalent["target_safety"]["backward_exit_recovery"] = (
            copy.deepcopy(
                CONTRACT["target_safety"]["backward_exit_recovery"]
            )
        )
        validate_frozen_contract(frozen_equivalent)

    monkeypatch.setattr(
        package_manifest_module,
        "REVERSE_PHASE_ENTRY_RELEASE_STATUS",
        FORMAL_RELEASE_STATUS,
    )
    monkeypatch.setattr(
        package_manifest_module,
        "BACKWARD_EXIT_RECOVERY_STATUS",
        FORMAL_RELEASE_STATUS,
    )
    monkeypatch.setattr(
        package_manifest_module,
        "BACKWARD_EXIT_RECOVERY_ENABLED_BY_DEFAULT",
        True,
    )
    monkeypatch.setattr(
        package_manifest_module,
        "validate_contract",
        validate_test_promoted_contract,
    )
    arguments["formal_release_evidence"] = evidence
    arguments["contract_source"] = promoted_contract_path
    return arguments


def _build_promoted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    **overrides: Path | str | None,
) -> tuple[Path, dict[str, object]]:
    arguments = _promoted_sources(tmp_path, monkeypatch)
    arguments.update(overrides)
    output = tmp_path / "package"
    manifest_path = build_router_package(output, **arguments)
    manifest = load_and_validate_package(
        output,
        expected_base_v22_sha256=str(arguments["expected_base_v22_sha256"]),
    )
    assert manifest_path == output / "package_manifest.json"
    return output, manifest


def test_release_allowlist_is_empty_until_h4_quality_requalification() -> None:
    assert FORMAL_RELEASE_EVIDENCE_SHA256_ALLOWLIST == frozenset()
    evidence = (
        EXPERIMENT_ROOT
        / "artifacts"
        / "h3_formal_release_20x30_seed20260808_v1.json"
    )
    assert evidence.stat().st_size == FORMAL_RELEASE_EVIDENCE_SIZE_BYTES
    assert sha256_file(evidence) == FORMAL_RELEASE_EVIDENCE_SHA256


def test_exact_h3_release_evidence_is_historical_but_semantically_auditable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _production_sources()
    evidence = (
        EXPERIMENT_ROOT
        / "artifacts"
        / "h3_formal_release_20x30_seed20260808_v1.json"
    )

    monkeypatch.setattr(
        package_manifest_module,
        "FORMAL_RELEASE_EVIDENCE_SHA256_ALLOWLIST",
        frozenset({FORMAL_RELEASE_EVIDENCE_SHA256}),
    )
    record = _validate(evidence, _bindings(arguments))

    assert record["sha256"] == FORMAL_RELEASE_EVIDENCE_SHA256
    assert record["payload"]["evaluation_mode"] == "RELEASE_QUALIFICATION"
    assert record["payload"]["hardware_gate"]["status"] == "PROHIBITED"


def test_package_build_cli_requires_explicit_formal_evidence() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["build", "--output", "package"])

    args = parser.parse_args(
        [
            "build",
            "--output",
            "package",
            "--formal-evidence",
            "formal-20x30.json",
        ]
    )
    assert args.formal_evidence == Path("formal-20x30.json")


@pytest.mark.parametrize(
    "candidate_name",
    (
        "h2_integrated_phase744_rate105_recovery0175_hold13_transition20x9_v1.json",
        "h2_combined_candidate_5x15_seed20260808_v1.json",
        "h2_formal_candidate_pending_20x30_seed20260808_v1.json",
        "h2_aggressive_short_transition_recovery0225_hold13_20seed_v1.json",
        "h3_combined_candidate_5x15_seed20260808_v1.json",
        "h3_formal_candidate_pending_20x30_seed20260808_v1.json",
    ),
)
def test_screening_or_safety_only_evidence_cannot_satisfy_release_allowlist_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, candidate_name: str
) -> None:
    arguments = _sources(tmp_path)
    bindings = _bindings(arguments)
    candidate = EXPERIMENT_ROOT / "artifacts" / candidate_name
    monkeypatch.setattr(
        package_manifest_module,
        "FORMAL_RELEASE_EVIDENCE_SHA256_ALLOWLIST",
        frozenset({sha256_file(candidate)}),
    )

    with pytest.raises(
        ValueError, match="evaluator_id mismatch|release qualification|20x30"
    ):
        _validate(candidate, bindings)


def test_builder_requires_formal_evidence_before_reading_sources(
    tmp_path: Path,
) -> None:
    arguments = _sources(tmp_path)
    output = tmp_path / "package"

    with pytest.raises(ValueError, match="fail-closed.*formal 20x30"):
        build_router_package(output, **arguments)

    assert not output.exists()


def test_builder_refuses_to_overwrite_before_release_checks(tmp_path: Path) -> None:
    arguments = _sources(tmp_path)
    output = tmp_path / "package"
    output.mkdir()
    sentinel = _write(output / "keep.txt", b"keep\n")

    with pytest.raises(FileExistsError, match="overwrite"):
        build_router_package(output, **arguments)

    assert sentinel.read_bytes() == b"keep\n"


def test_unallowlisted_evidence_is_rejected_without_output(tmp_path: Path) -> None:
    arguments = _sources(tmp_path)
    evidence = _write(tmp_path / "unallowlisted.json", b"{}\n")
    output = tmp_path / "package"

    with pytest.raises(ValueError, match="SHA-256.*allowlist"):
        build_router_package(
            output,
            formal_release_evidence=evidence,
            **arguments,
        )

    assert not output.exists()


def test_formal_evidence_validator_binds_scale_seed_and_all_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _sources(tmp_path)
    bindings = _bindings(arguments)
    evidence = _write_allowlisted_evidence(
        tmp_path, monkeypatch, _formal_payload(bindings)
    )

    record = _validate(evidence, bindings)

    assert record["sha256"] == sha256_file(evidence)


def test_formal_evidence_rejects_non_master_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings = _bindings(_sources(tmp_path))
    payload = _formal_payload(bindings)
    payload["release_qualification"]["actual"]["master_seed"] = 7
    payload["configuration"]["seed"] = 7
    evidence = _write_allowlisted_evidence(tmp_path, monkeypatch, payload)

    with pytest.raises(ValueError, match="master_seed=20260808"):
        _validate(evidence, bindings)


def test_h3_shaped_tampered_nested_audit_is_rejected_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (
        EXPERIMENT_ROOT
        / "artifacts"
        / "h3_formal_release_20x30_seed20260808_v1.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["suites"]["primitives"]["episodes"][0]["segments"][0][
        "safety_audit"
    ]["qpos_limit_violations"] = 1
    tampered = tmp_path / "tampered-h3-release.json"
    tampered.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        package_manifest_module,
        "FORMAL_RELEASE_EVIDENCE_SHA256_ALLOWLIST",
        frozenset({sha256_file(tampered)}),
    )
    output = tmp_path / "package"

    with pytest.raises(ValueError, match="safety audit failed"):
        build_router_package(
            output,
            formal_release_evidence=tampered,
            **_production_sources(),
        )

    assert not output.exists()


def test_diagnostic_phase_contract_cannot_qualify_a_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings = _bindings(_sources(tmp_path))
    payload = _formal_payload(bindings)
    payload.pop("formal_reverse_phase_entry_contract")
    payload["diagnostic_reverse_phase_entry_contract"] = {
        "diagnostic_only": True,
        "adoption_eligible": False,
        "hardware_deployment": "PROHIBITED",
    }
    evidence = _write_allowlisted_evidence(tmp_path, monkeypatch, payload)

    with pytest.raises(ValueError, match="formal_reverse_phase_entry_contract"):
        _validate(evidence, bindings)


def test_diagnostic_recovery_contract_cannot_qualify_a_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings = _bindings(_sources(tmp_path))
    payload = _formal_payload(bindings)
    payload.pop("formal_backward_exit_recovery_contract")
    payload["diagnostic_backward_exit_recovery_contract"] = {
        "status": "DIAGNOSTIC_UNADOPTED",
        "enabled_by_default": False,
        "adoption_eligible": False,
        "hardware_deployment": "PROHIBITED",
    }
    evidence = _write_allowlisted_evidence(tmp_path, monkeypatch, payload)

    with pytest.raises(
        ValueError, match="formal_backward_exit_recovery_contract"
    ):
        _validate(evidence, bindings)


def test_evidence_that_allows_hardware_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings = _bindings(_sources(tmp_path))
    payload = _formal_payload(bindings)
    payload["hardware_gate"]["status"] = "ALLOWED"
    payload["hardware_gate"]["hardware_deployment_allowed"] = True
    evidence = _write_allowlisted_evidence(tmp_path, monkeypatch, payload)

    with pytest.raises(ValueError, match="hardware PROHIBITED"):
        _validate(evidence, bindings)


@pytest.mark.parametrize(
    "evidence_name",
    (
        "h3_formal_candidate_pending_20x30_seed20260808_v1.json",
        "h3_combined_candidate_5x15_seed20260808_v1.json",
        "h2_aggressive_short_transition_recovery0225_hold13_20seed_v1.json",
        "h2_formal_candidate_pending_20x30_seed20260808_v1.json",
    ),
)
def test_builder_rejects_old_nonrelease_evidence_without_output(
    tmp_path: Path, evidence_name: str
) -> None:
    arguments = _production_sources()
    evidence = EXPERIMENT_ROOT / "artifacts" / evidence_name
    output = tmp_path / "package"

    assert FORMAL_RELEASE_EVIDENCE_SHA256_ALLOWLIST == frozenset()
    with pytest.raises(ValueError, match="not in the frozen adoption allowlist"):
        build_router_package(
            output,
            formal_release_evidence=evidence,
            **arguments,
        )

    assert not output.exists()


def test_superseded_package_without_formal_binding_is_not_loadable() -> None:
    superseded = (
        EXPERIMENT_ROOT
        / "artifacts"
        / "router_packages"
        / "exp004-safe-gait-router-v1"
    )

    with pytest.raises(ValueError, match="-0.050|recovery|formal"):
        load_and_validate_package(superseded)


def test_mutating_a_bound_profile_hash_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings = _bindings(_sources(tmp_path))
    payload = _formal_payload(bindings)
    payload["reverse_profile_adoption"]["roles"]["straight"][
        "profile_sha256"
    ] = "0" * 64
    evidence = _write_allowlisted_evidence(tmp_path, monkeypatch, payload)

    with pytest.raises(ValueError, match="profile straight binding"):
        _validate(evidence, bindings)


def test_promoted_fixture_package_is_closed_portable_and_hardware_prohibited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, manifest = _build_promoted(tmp_path, monkeypatch)

    assert manifest["safety"]["hardware_deployment"] == "PROHIBITED"
    assert manifest["safety"]["simulation_use_only"] is True
    assert manifest["controller"]["dynamic_model_lookup"] is False
    assert manifest["controller"]["unknown_expert_behavior"] == "REJECT"
    assert set(manifest["routes"]) == set(ALLOWED_EXPERTS)
    assert set(manifest["safety"]["reachable_model_ids"]) == {
        BASE_V22_MODEL_ID
    }
    assert manifest["routes"][REVERSE]["model_id"] == BASE_V22_MODEL_ID
    assert manifest["safety"]["rejected_lineages"] == list(
        REJECTED_LINEAGES
    )
    assert manifest["safety"]["rejected_lineages_reachable"] is False

    envelope = manifest["controller"]["command_envelope"]
    assert envelope["minimum"] == list(DEFAULT_COMMAND_MIN)
    assert envelope["maximum"] == list(DEFAULT_COMMAND_MAX)
    assert envelope["enforcement"] == "clip_before_slew_and_routing"
    assert manifest["safety"]["formal_release_gate"]["status"] == (
        FORMAL_RELEASE_STATUS
    )
    assert (
        manifest["safety"]["formal_release_gate"]["hardware_deployment"]
        == "PROHIBITED"
    )

    recovery = manifest["controller"]["runtime_backward_exit_recovery"]
    assert recovery["enabled_by_default"] is True
    assert recovery["extra_upper_margin_rad"] == 0.0225
    assert recovery["hold_control_ticks"] == 13
    assert recovery["hold_seconds"] == 0.26
    assert recovery["upper_target_rad"] == 0.403034
    assert recovery["hardware_deployment"] == "PROHIBITED"

    for record in manifest["integrity"]["files"].values():
        assert not Path(record["path"]).is_absolute()
        assert "\\" not in record["path"]
        assert ".." not in Path(record["path"]).parts
        assert (output / record["path"]).is_file()


def test_manifest_records_scene_reference_contract_and_evidence_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, manifest = _build_promoted(tmp_path, monkeypatch)
    files = manifest["integrity"]["files"]

    for asset_name, file_id in (
        ("scene", "scene"),
        ("reference", "reference"),
        ("safety_contract", "contract"),
        ("formal_release_evidence", "formal_release_evidence"),
    ):
        assert manifest["assets"][asset_name]["sha256"] == files[file_id][
            "sha256"
        ]


def test_yaw_right_policy_offset_is_exact_and_route_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, manifest = _build_promoted(tmp_path, monkeypatch)
    correction = manifest["corrections"][
        "yaw_right_policy_command_offset"
    ]

    assert correction["stage"] == "before_policy_inference"
    assert correction["axis"] == "yaw_rate"
    assert correction["operation"] == "add"
    assert correction["value"] == YAW_RIGHT_POLICY_OFFSET == -0.30
    assert correction["requested_command_unchanged"] is True
    assert manifest["routes"][YAW_RIGHT]["correction_ids"] == [
        "yaw_right_policy_command_offset"
    ]
    assert all(
        "yaw_right_policy_command_offset" not in route["correction_ids"]
        for expert, route in manifest["routes"].items()
        if expert != YAW_RIGHT
    )


def test_optional_reverse_onnx_is_packaged_but_causally_inert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reverse = _write(
        tmp_path / "sources" / "reverse_exp004.onnx", b"reverse model\n"
    )
    output, manifest = _build_promoted(
        tmp_path, monkeypatch, reverse_onnx=reverse
    )

    assert set(manifest["safety"]["reachable_model_ids"]) == {
        BASE_V22_MODEL_ID
    }
    assert manifest["routes"][REVERSE]["model_id"] == BASE_V22_MODEL_ID
    assert manifest["routes"][REVERSE]["correction_ids"] == [
        "reverse_profile"
    ]
    assert manifest["routes"][REVERSE]["residual_model_id"] == REVERSE_MODEL_ID
    assert manifest["routes"][REVERSE]["residual_scale"] == REVERSE_RESIDUAL_SCALE
    assert manifest["models"][REVERSE_MODEL_ID]["execution_status"] == "DISABLED"
    assert manifest["models"][REVERSE_MODEL_ID]["evaluation_status"] == (
        "REJECTED_NOT_ADOPTED"
    )
    assert (output / "models" / "reverse_exp004.onnx").read_bytes() == (
        b"reverse model\n"
    )


def test_valid_reverse_export_report_is_bundled_and_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reverse = _write(
        tmp_path / "sources" / "reverse_exp004.onnx", b"reverse model\n"
    )
    report = {
        "hardware_deployment": "PROHIBITED",
        "onnx": {"sha256": sha256_file(reverse)},
        "interface": {
            "input_name": "obs",
            "input_shape": [1, 101],
            "output_shape": [1, 14],
        },
    }
    report_path = reverse.with_suffix(".onnx.json")
    report_path.write_text(json.dumps(report), encoding="utf-8")

    output, manifest = _build_promoted(
        tmp_path, monkeypatch, reverse_onnx=reverse
    )

    assert manifest["models"][REVERSE_MODEL_ID]["interface_verification"] == (
        "export_report_verified"
    )
    assert (output / "models" / "reverse_exp004.onnx.json").is_file()


def test_reverse_release_is_derived_from_formal_evidence_not_old_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, manifest = _build_promoted(tmp_path, monkeypatch)
    reverse = manifest["corrections"]["reverse_profile"]
    decision = manifest["safety"]["reverse_release_decision"]

    assert reverse["release_id"] == REVERSE_PROFILE_RELEASE_ID
    assert reverse["base_model_id"] == BASE_V22_MODEL_ID
    assert reverse["residual_scale"] == 0.0
    assert decision["validated_command"] == [-0.050, 0.0, 0.0]
    assert decision["status"] == FORMAL_RELEASE_STATUS
    assert decision["profile_evidence"]["passed"] is True
    assert decision["formal_evidence"]["master_seed"] == 20260808
    assert "acceptance" not in decision


def test_reverse_turns_are_atomic_validated_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, manifest = _build_promoted(tmp_path, monkeypatch)

    for expert in (REVERSE_TURN_LEFT, REVERSE_TURN_RIGHT):
        maneuver = manifest["controller"]["atomic_maneuvers"][expert]
        assert maneuver["command"] == list(REVERSE_TURN_ENDPOINTS[expert])
        assert maneuver["enter_via"] == maneuver["exit_via"] == "stand"
        assert maneuver["profile_interpolation"] == "PROHIBITED"
        assert maneuver["action_blending"] == "PROHIBITED"
        release = manifest["safety"]["reverse_turn_release_decisions"][expert]
        assert release["profile_evidence"]["passed"] is True
        assert "falls" not in release


def _validate_mutation(
    output: Path, manifest: dict[str, object], expected_hash: str
) -> None:
    package_manifest_module.validate_package_manifest(
        manifest,
        output,
        expected_base_v22_sha256=expected_hash,
    )


def test_validator_rejects_symmetric_or_looser_reverse_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, manifest = _build_promoted(tmp_path, monkeypatch)
    mutated = copy.deepcopy(manifest)
    mutated["controller"]["command_envelope"]["minimum"][0] = -0.08

    with pytest.raises(ValueError, match="-0.050"):
        _validate_mutation(
            output,
            mutated,
            manifest["integrity"]["files"]["base_v22_onnx"]["sha256"],
        )


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        (
            "runtime_target_safety",
            "runtime_target_safety_margin_rad",
            0.0,
            "2.0 rad/s slew and 0.050 rad margin",
        ),
        (
            "runtime_target_safety",
            "max_target_slew_rate_rad_s",
            3.0,
            "2.0 rad/s slew",
        ),
        (
            "runtime_reset_safety",
            "perturbed_reset_qpos_margin_rad",
            0.0,
            "0.005 rad margin",
        ),
        (
            "runtime_startup_safety",
            "physics_steps_before_guarded_control",
            1,
            "control-first startup",
        ),
    ],
)
def test_validator_rejects_runtime_safety_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    output, manifest = _build_promoted(tmp_path, monkeypatch)
    mutated = copy.deepcopy(manifest)
    mutated["controller"][section][field] = value

    with pytest.raises(ValueError, match=message):
        _validate_mutation(
            output,
            mutated,
            manifest["integrity"]["files"]["base_v22_onnx"]["sha256"],
        )


def test_validator_rejects_home_precharge_or_double_slew(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, manifest = _build_promoted(tmp_path, monkeypatch)
    expected_hash = manifest["integrity"]["files"]["base_v22_onnx"]["sha256"]

    home_precharge = copy.deepcopy(manifest)
    home_precharge["controller"]["runtime_startup_safety"][
        "home_only_precharge"
    ] = "ALLOWED"
    with pytest.raises(ValueError, match="control-first startup"):
        _validate_mutation(output, home_precharge, expected_hash)

    double_slew = copy.deepcopy(manifest)
    double_slew["controller"]["runtime_target_safety"][
        "slew_applications_per_tick"
    ] = 2
    with pytest.raises(ValueError, match="runtime target guard"):
        _validate_mutation(output, double_slew, expected_hash)


def test_package_validator_cross_checks_packaged_runtime_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, manifest = _build_promoted(tmp_path, monkeypatch)
    mutated = copy.deepcopy(manifest)
    contract_record = mutated["integrity"]["files"]["contract"]
    contract_path = output / contract_record["path"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["target_safety"]["control_first_startup"][
        "control_dt_seconds"
    ] = 0.01
    contract_path.write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    contract_record["sha256"] = sha256_file(contract_path)
    contract_record["size_bytes"] = contract_path.stat().st_size
    mutated["assets"]["safety_contract"]["sha256"] = contract_record["sha256"]

    with pytest.raises(ValueError, match="startup dt.*0.02"):
        _validate_mutation(
            output,
            mutated,
            manifest["integrity"]["files"]["base_v22_onnx"]["sha256"],
        )


def test_validator_rejects_enabling_optional_reverse_residual(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reverse = _write(
        tmp_path / "sources" / "reverse_exp004.onnx", b"reverse model\n"
    )
    output, manifest = _build_promoted(
        tmp_path, monkeypatch, reverse_onnx=reverse
    )
    mutated = copy.deepcopy(manifest)
    mutated["routes"][REVERSE]["residual_scale"] = 0.12

    with pytest.raises(ValueError, match="causally inert"):
        _validate_mutation(
            output,
            mutated,
            manifest["integrity"]["files"]["base_v22_onnx"]["sha256"],
        )


@pytest.mark.parametrize(
    "name", ["v59.onnx", "omnidirectional_v60.onnx", "safe-v59-copy.onnx"]
)
def test_rejected_lineage_cannot_be_supplied_as_reverse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    reverse = _write(tmp_path / "sources" / name, b"rejected\n")
    arguments = _promoted_sources(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="v59/v60"):
        build_router_package(
            tmp_path / "package", reverse_onnx=reverse, **arguments
        )


def test_validator_rejects_a_route_to_v59(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, manifest = _build_promoted(tmp_path, monkeypatch)
    mutated = copy.deepcopy(manifest)
    mutated["models"]["v59"] = {
        "release_id": "v59",
        "source_role": "rejected",
        "file_id": "base_v22_onnx",
    }
    mutated["routes"][YAW_RIGHT]["model_id"] = "v59"

    with pytest.raises(ValueError, match="model set|v59/v60|rejected"):
        _validate_mutation(
            output,
            mutated,
            manifest["integrity"]["files"]["base_v22_onnx"]["sha256"],
        )


def test_hash_verification_detects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, manifest = _build_promoted(tmp_path, monkeypatch)
    (output / manifest["integrity"]["files"]["scene"]["path"]).write_text(
        "tampered", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="size mismatch|SHA-256 mismatch"):
        load_and_validate_package(
            output,
            expected_base_v22_sha256=manifest["integrity"]["files"][
                "base_v22_onnx"
            ]["sha256"],
        )


def test_wrong_base_v22_hash_is_rejected_before_output_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _promoted_sources(tmp_path, monkeypatch)
    arguments["expected_base_v22_sha256"] = "0" * 64
    output = tmp_path / "package"

    with pytest.raises(ValueError, match="frozen release hash"):
        build_router_package(output, **arguments)

    assert not output.exists()
