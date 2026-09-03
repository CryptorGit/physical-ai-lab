from __future__ import annotations

import ast
from dataclasses import fields
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from safe_gait_experts.gait_quality import (
    GaitQualityMetrics,
    rederive_gait_quality_acceptance,
)
from safe_gait_experts.h4_post_training import load_json_strict


EXP_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = EXP_ROOT / "scripts" / "evaluate_h4_training_candidate.py"
EXPORTER_PATH = EXP_ROOT / "scripts" / "export_h4_expert_onnx.py"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _calls_named_train(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (
            (isinstance(node.func, ast.Name) and node.func.id == "train")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "train")
        ):
            lines.append(node.lineno)
    return lines


def test_post_training_entrypoints_never_start_ppo() -> None:
    assert _calls_named_train(EVALUATOR_PATH) == []
    assert _calls_named_train(EXPORTER_PATH) == []


def test_evaluator_parser_is_cpu_only_and_hash_bound() -> None:
    module = _load(EVALUATOR_PATH, "test_h4_candidate_evaluator")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--params",
            "params.pkl",
            "--params-sha256",
            "1" * 64,
            "--manifest",
            "manifest.json",
            "--manifest-sha256",
            "2" * 64,
            "--output",
            "strict.json",
        ]
    )
    assert args.platform == "cpu"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--params",
                "params.pkl",
                "--params-sha256",
                "1" * 64,
                "--manifest",
                "manifest.json",
                "--manifest-sha256",
                "2" * 64,
                "--output",
                "strict.json",
                "--platform",
                "gpu",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--params",
                "params.pkl",
                "--params-sha256",
                "1" * 64,
                "--manifest",
                "manifest.json",
                "--manifest-sha256",
                "2" * 64,
                "--output",
                "strict.json",
                "--baseline-artifact",
                "self_reported.json",
            ]
        )


def test_evaluator_freezes_relative_paths_before_runtime_cwd_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load(EVALUATOR_PATH, "test_h4_candidate_path_freeze")
    process_start = tmp_path / "process_start"
    later_cwd = tmp_path / "legacy_runtime_cwd"
    process_start.mkdir()
    later_cwd.mkdir()
    monkeypatch.chdir(process_start)
    args = module.build_parser().parse_args(
        [
            "--params",
            "run/final_params.pkl",
            "--params-sha256",
            "1" * 64,
            "--manifest",
            "run/training_manifest.json",
            "--manifest-sha256",
            "2" * 64,
            "--output",
            "run/strict.json",
            "--promotion-evidence-output",
            "run/promotion.json",
            "--trusted-run-root",
            "trusted",
            "--source-root",
            "source",
            "--generated-root",
            "generated",
            "--v22-parent-checkpoint",
            "v22",
        ]
    )
    args = module._resolve_process_start_paths(args)
    expected_output = (process_start / "run" / "strict.json").resolve()
    expected_params = (process_start / "run" / "final_params.pkl").resolve()
    assert args.output == expected_output
    assert args.params == expected_params
    for name in (
        "params",
        "manifest",
        "output",
        "promotion_evidence_output",
        "trusted_run_root",
        "source_root",
        "generated_root",
        "v22_parent_checkpoint",
    ):
        assert getattr(args, name).is_absolute()

    monkeypatch.chdir(later_cwd)
    assert args.output == expected_output
    assert args.params == expected_params


def test_reverse_evaluator_requires_pinned_teacher_and_authorization() -> None:
    module = _load(EVALUATOR_PATH, "test_h4_reverse_composition_loader")
    selected = (
        EXP_ROOT / "artifacts" / "h4_reverse_slew_feasible_teacher_selected_v1.json"
    )
    authorization = (
        EXP_ROOT
        / "artifacts"
        / "h4_reverse_training_composition_authorization_v1.json"
    )
    bundle = SimpleNamespace(
        config={
            "selected_reverse_teacher": {
                "path": str(selected),
                "sha256": (
                    "7a24a7c9096a1c4a9dc72ac85ec01c5e0a41acf8214d80cc7e2cf4ccc50ae237"
                ),
            },
            "reverse_composition_authorization": {
                "path": str(authorization),
                "sha256": (
                    "082405e34b8a46e7d4a9ccf7b8c0729871fee1eb202b4a1ed8c758b2c7a52900"
                ),
            },
        }
    )
    composition = module._load_reverse_teacher(bundle)
    assert composition["table"].shape == (54, 14)
    assert composition["cadence_hz"] == 1.5
    assert composition["phase_advance_bins"] == 1.62
    assert composition["entry_phase_bins"] == 14.0
    assert all(composition["checks"].values())
    tampered = SimpleNamespace(config={**bundle.config})
    tampered.config = {
        **bundle.config,
        "reverse_composition_authorization": {
            **bundle.config["reverse_composition_authorization"],
            "sha256": "0" * 64,
        },
    }
    with pytest.raises(ValueError, match="binding is incomplete"):
        module._load_reverse_teacher(tampered)
    tampered_teacher = SimpleNamespace(
        config={
            **bundle.config,
            "selected_reverse_teacher": {
                **bundle.config["selected_reverse_teacher"],
                "sha256": "0" * 64,
            },
        }
    )
    with pytest.raises(ValueError, match="binding is incomplete"):
        module._load_reverse_teacher(tampered_teacher)


def test_evaluator_source_exposes_full_substep_and_gait_contract() -> None:
    source = EVALUATOR_PATH.read_text(encoding="utf-8")
    for required in (
        "H4_STRICT_PHYSICS_SUBSTEPS",
        "H4_STRICT_GAIT_SAMPLES",
        "GaitQualityAccumulator",
        "GaitQualitySubstep",
        "PhysicsSubstepAudit",
        "SafetyAudit",
        "FULL_CURRENT_P0_RECOMPUTED_FROM_N_PLUS_ONE_SUBSTEP_TRACE",
        "validate_h4_strict_artifact",
        "build_integrated_promotion_evidence",
        "audit_v22_to_h4_transplant",
        "optimizer_updates",
        "official_v22_baseline",
        'gait_payload = {**gait_metrics.as_dict(), "measurement_complete": True}',
        "PINNED_PERSISTENT_TEACHER_PLUS_TRAINABLE_RESIDUAL",
        "reverse_teacher_source_phase_before",
        "reverse_teacher_table_phase",
        "reverse_action_delay_index",
        "reverse_delayed_applied_action",
        "reverse_upstream_margin_targets",
        "H4_REVERSE_COMPOSITION_TRACE_SEMANTICS",
    ):
        assert required in source

    assert source.index("validate_h4_training_source_closure(bundle, expected_paths)") < (
        source.index("params, restore_audit = load_trusted_h4_params(bundle)")
    )


def test_current_central_baseline_has_no_omitted_gait_metric_field() -> None:
    baseline_path = (
        EXP_ROOT / "artifacts" / "h4_gait_quality_baseline_1x6_seed20260808_v5.json"
    )
    payload = load_json_strict(baseline_path)
    segment = payload["suites"]["primitives"]["episodes"][0]["segments"][0]
    metrics = segment["gait_quality_metrics"]
    expected = {definition.name for definition in fields(GaitQualityMetrics)}
    assert set(metrics) == expected | {"measurement_complete"}
    assert metrics["measurement_complete"] is True
    assert (
        rederive_gait_quality_acceptance(metrics).as_dict()
        == segment["gait_quality_acceptance"]
    )


def test_onnx_export_contract_is_actor116_cpu_and_three_way_parity() -> None:
    module = _load(EXPORTER_PATH, "test_h4_onnx_exporter")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--params",
            "params.pkl",
            "--params-sha256",
            "1" * 64,
            "--manifest",
            "manifest.json",
            "--manifest-sha256",
            "2" * 64,
            "--output",
            "actor.onnx",
        ]
    )
    assert args.output.suffix == ".onnx"
    source = EXPORTER_PATH.read_text(encoding="utf-8")
    for required in (
        "H4_ACTOR_OBSERVATION_WIDTH",
        'providers=["CPUExecutionProvider"]',
        "numpy_brax_parity",
        "numpy_onnx_parity",
        "brax_onnx_parity",
        "post_inference_mask_slice",
        "standalone_direct_runtime_allowed",
        "PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256",
        "PINNED_SELECTED_REVERSE_TEACHER_SHA256",
    ):
        assert required in source

    assert source.index("bundle = validate_h4_training_source_closure(") < source.index(
        "params, restore_audit = load_trusted_h4_params(bundle)"
    )


def test_onnx_export_refuses_to_overwrite_before_loading_bundle(tmp_path: Path) -> None:
    module = _load(EXPORTER_PATH, "test_h4_onnx_no_overwrite")
    output = tmp_path / "actor.onnx"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.run_export(SimpleNamespace(output=output))
