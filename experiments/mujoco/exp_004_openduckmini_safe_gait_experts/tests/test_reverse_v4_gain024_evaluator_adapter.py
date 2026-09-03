from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np
import pytest


EXP_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = (
    EXP_ROOT
    / "scripts"
    / "evaluate_h4_training_candidate_reverse_v4_gain024_v1.py"
)
FIXTURE_SOURCE_PATH = EXP_ROOT / "tests" / "test_h4_post_training.py"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


adapter = _load_module("exp004_reverse_v4_gain024_adapter_test", ADAPTER_PATH)


def _exact_args() -> argparse.Namespace:
    return adapter._resolve_process_start_paths(
        argparse.Namespace(
            params=adapter.EXPECTED_PARAMS_PATH,
            params_sha256=adapter.PINNED_CANDIDATE_FILES["candidate_params"][1],
            manifest=adapter.EXPECTED_MANIFEST_PATH,
            manifest_sha256=adapter.PINNED_CANDIDATE_FILES[
                "candidate_manifest"
            ][1],
            output=adapter.EXPECTED_OUTPUT_PATH,
            trusted_run_root=adapter.TRUSTED_RUN_ROOT,
            adapter_authorization=adapter.ADAPTER_AUTHORIZATION_PATH,
            source_root=adapter.DEFAULT_SOURCE_ROOT,
            generated_root=adapter.DEFAULT_GENERATED_ROOT,
            v22_parent_checkpoint=adapter.DEFAULT_V22_PARENT_CHECKPOINT,
            platform="cpu",
        )
    )


def test_authorization_and_candidate_bindings_are_exact_but_v4_sources_are_stale(
) -> None:
    payload = adapter.load_and_validate_adapter_authorization(
        adapter.ADAPTER_AUTHORIZATION_PATH
    )
    assert payload["scope"]["contract_id"] == adapter.ADAPTER_CONTRACT_ID
    assert adapter.sha256_file(adapter.ADAPTER_AUTHORIZATION_PATH) == (
        adapter.PINNED_ADAPTER_AUTHORIZATION_SHA256
    )
    assert adapter._validate_exact_candidate_files() == {
        name: binding[1]
        for name, binding in adapter.PINNED_CANDIDATE_FILES.items()
    }
    with pytest.raises(ValueError, match="pinned file SHA256 drifted"):
        adapter._verify_file_bindings(adapter.PINNED_FROZEN_SOURCES)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("scope", "contract_id"), "WRONG"),
        (("scope", "evaluation_artifact_kind"), "WRONG"),
        (("scope", "promotion_eligible"), True),
        (("causal_basis", "base_composition_maximum_residual_scale"), 0.24),
        (("candidate_binding", "backward_residual_scale"), 0.12),
        (
            ("authorization_bindings", "reverse_iteration_v4", "sha256"),
            "0" * 64,
        ),
        (
            ("frozen_source_bindings", "h4_candidate_evaluator", "sha256"),
            "0" * 64,
        ),
        (("authorization", "promotion_evidence"), True),
    ),
)
def test_authorization_semantic_drift_fails_closed(
    path: tuple[str, ...], value: Any
) -> None:
    payload = adapter.load_json_strict(adapter.ADAPTER_AUTHORIZATION_PATH)
    current = payload
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value
    with pytest.raises(ValueError, match="authorization"):
        adapter.validate_adapter_authorization_payload(payload)


@pytest.mark.parametrize(
    ("record", "key", "value"),
    (
        ("config", "backward_residual_scale", 0.12),
        ("config", "backward_residual_scale", 0.24000001),
        ("config", "reverse_iteration_v4_residual_transfer_gain_024", False),
        ("config", "training_contract_id", "WRONG"),
        ("manifest", "status", "WIRING_PASS"),
        ("manifest", "reverse_iteration_v4_residual_transfer_gain_024", False),
        ("result", "activity", "PPO_WIRING_TRAINING"),
        (
            "result",
            "reverse_iteration_v4_residual_transfer_gain_024_authorization_sha256",
            "0" * 64,
        ),
    ),
)
def test_candidate_metadata_drift_fails_closed(
    record: str, key: str, value: Any
) -> None:
    records = {
        "config": adapter.load_json_strict(adapter.EXPECTED_CONFIG_PATH),
        "manifest": adapter.load_json_strict(adapter.EXPECTED_MANIFEST_PATH),
        "result": adapter.load_json_strict(adapter.EXPECTED_RESULT_PATH),
    }
    records[record][key] = value
    with pytest.raises(ValueError, match="candidate metadata"):
        adapter._validate_candidate_metadata(
            records["config"], records["manifest"], records["result"]
        )


def test_base_composition_remains_0p12_and_0p24_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter._validate_base_authorization_semantics()
    base_path = adapter.PINNED_FROZEN_SOURCES[
        "base_reverse_composition_authorization"
    ][0].resolve()
    base = adapter.load_json_strict(base_path)
    v4 = adapter.load_json_strict(
        adapter.PINNED_FROZEN_SOURCES["reverse_v4_authorization"][0]
    )
    base["composition_contract"]["maximum_residual_scale"] = 0.24

    def fake_load(path: Path) -> dict[str, Any]:
        return copy.deepcopy(base if Path(path).resolve() == base_path else v4)

    monkeypatch.setattr(adapter, "load_json_strict", fake_load)
    with pytest.raises(ValueError, match="authorization semantics"):
        adapter._validate_base_authorization_semantics()


def test_functiontype_call_graph_injects_0p24_without_module_mutation() -> None:
    base, post = adapter._load_frozen_modules()
    before = adapter._module_contract_snapshot(base, post)
    graph = adapter.build_gain024_call_graph(base, post)
    assert base.H4_REVERSE_RESIDUAL_SCALE == 0.12
    assert post.H4_REVERSE_RESIDUAL_SCALE == 0.12
    assert graph["make_environment"].__globals__["H4_REVERSE_RESIDUAL_SCALE"] == 0.24
    assert graph["control"].__globals__["H4_REVERSE_RESIDUAL_SCALE"] == 0.24
    assert graph["safety"].__globals__["rederive_h4_control_contract"] is graph[
        "control"
    ]
    assert graph["episode_validator"].__globals__[
        "rederive_h4_safety_acceptance"
    ] is graph["safety"]
    assert graph["artifact_validator"].__globals__[
        "validate_h4_strict_episode"
    ] is graph["episode_validator"]
    assert graph["run_evaluation"].__globals__["_make_environment_and_policy"] is (
        graph["make_environment"]
    )
    assert graph["run_evaluation"].__globals__["_run_episode"] is graph[
        "run_episode"
    ]
    assert graph["run_evaluation"].__globals__["STRICT_ARTIFACT_KIND"] == (
        adapter.DIAGNOSTIC_ARTIFACT_KIND
    )
    adapter._assert_module_contract_unchanged(base, post, before)


def test_cloned_function_restores_nothing_even_when_it_raises() -> None:
    namespace: dict[str, Any] = {"VALUE": 0.12}
    exec(
        "def probe(raise_now=False):\n"
        "    if raise_now:\n"
        "        raise RuntimeError('probe')\n"
        "    return VALUE\n",
        namespace,
    )
    original = namespace["probe"]
    clone = adapter._clone_function(original, global_overrides={"VALUE": 0.24})
    assert clone() == 0.24
    assert original() == 0.12
    with pytest.raises(RuntimeError, match="probe"):
        clone(True)
    assert original() == 0.12
    assert original.__globals__["VALUE"] == 0.12


def test_exact_cli_and_promotion_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _exact_args()
    adapter._validate_exact_cli(args, require_output_absent=False)
    wrong = copy.copy(args)
    wrong.params_sha256 = "0" * 64
    with pytest.raises(ValueError, match="CLI drifted"):
        adapter._validate_exact_cli(wrong, require_output_absent=False)

    parser = adapter.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--params",
                str(adapter.EXPECTED_PARAMS_PATH),
                "--params-sha256",
                adapter.PINNED_CANDIDATE_FILES["candidate_params"][1],
                "--manifest",
                str(adapter.EXPECTED_MANIFEST_PATH),
                "--manifest-sha256",
                adapter.PINNED_CANDIDATE_FILES["candidate_manifest"][1],
                "--output",
                str(adapter.EXPECTED_OUTPUT_PATH),
                "--promotion-evidence-output",
                str(tmp_path / "promotion.json"),
            ]
        )

    existing = tmp_path / "existing.json"
    existing.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(adapter, "EXPECTED_OUTPUT_PATH", existing)
    existing_args = copy.copy(args)
    existing_args.output = existing.resolve()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        adapter._validate_exact_cli(existing_args)


def test_evaluation_source_extension_preserves_base_and_binds_adapter() -> None:
    base_hashes = {
        "scripts/evaluate_h4_training_candidate.py": adapter.PINNED_FROZEN_SOURCES[
            "h4_candidate_evaluator"
        ][1],
        "safe_gait_experts/h4_post_training.py": adapter.PINNED_FROZEN_SOURCES[
            "h4_post_training"
        ][1],
        "extra.py": "9" * 64,
    }
    adapter_hashes = {
        adapter.ADAPTER_SOURCE_KEY: "8" * 64,
        adapter.ADAPTER_AUTHORIZATION_SOURCE_KEY: (
            adapter.PINNED_ADAPTER_AUTHORIZATION_SHA256
        ),
    }
    augmented = adapter._augment_evaluation_hashes(base_hashes, adapter_hashes)
    assert augmented == {**base_hashes, **adapter_hashes}
    assert base_hashes == {
        "scripts/evaluate_h4_training_candidate.py": adapter.PINNED_FROZEN_SOURCES[
            "h4_candidate_evaluator"
        ][1],
        "safe_gait_experts/h4_post_training.py": adapter.PINNED_FROZEN_SOURCES[
            "h4_post_training"
        ][1],
        "extra.py": "9" * 64,
    }
    drifted = dict(base_hashes)
    drifted["scripts/evaluate_h4_training_candidate.py"] = "0" * 64
    with pytest.raises(ValueError, match="source provenance"):
        adapter._augment_evaluation_hashes(drifted, adapter_hashes)


def _minimal_six_episode_artifact(scale: float = 0.24) -> dict[str, Any]:
    episode = {
        "reverse_composition_contract": {"residual_scale": scale},
        "h4_control_contract": {
            "checks": {
                "reverse_composition_contract_exact": True,
                "reverse_teacher_table_phase_exact_after_preincrement": True,
            }
        },
    }
    episodes = [copy.deepcopy(episode) for _ in range(3)]
    return {
        "episodes": episodes,
        "official_v22_baseline": {"episodes": copy.deepcopy(episodes)},
    }


def test_all_six_episode_scale_is_float_exact() -> None:
    adapter._validate_six_episode_gain024(_minimal_six_episode_artifact())
    for drift in (0.12, 0.24000001, float(np.nextafter(0.24, np.inf))):
        artifact = _minimal_six_episode_artifact()
        artifact["official_v22_baseline"]["episodes"][2][
            "reverse_composition_contract"
        ]["residual_scale"] = drift
        with pytest.raises(ValueError, match="episode 5"):
            adapter._validate_six_episode_gain024(artifact)


def _gain024_fixture() -> tuple[
    dict[str, Any], Any, dict[str, Any], dict[str, str], dict[str, str]
]:
    fixture_source = _load_module(
        "exp004_gain024_adapter_fixture_source", FIXTURE_SOURCE_PATH
    )
    base, post = adapter._load_frozen_modules()
    graph = adapter.build_gain024_call_graph(base, post)

    proxy = types.ModuleType("exp004_gain024_fixture_post_proxy")
    proxy.__dict__.update(vars(fixture_source.h4pt))
    proxy.H4_REVERSE_RESIDUAL_SCALE = adapter.GAIN024_RESIDUAL_SCALE
    reverse_episode_builder = adapter._clone_function(
        fixture_source._reverse_safety_episode,
        global_overrides={
            "h4pt": proxy,
            "rederive_h4_control_contract": graph["control"],
        },
    )

    def episode(seed: int) -> dict[str, Any]:
        record = reverse_episode_builder()
        record.update(
            {
                "seed": seed,
                "segment_id": f"h4_reverse_seed{seed}_6s",
                "expert": "reverse",
                "physical_command_mps_radps": list(
                    post.H4_STRICT_COMMANDS["reverse"]
                ),
                "source_segment_kind": "H4_STRICT_6S",
                "gait_quality_metrics": fixture_source._gait_metrics(),
                "gait_quality_acceptance": fixture_source._fake_rederive(None),
            }
        )
        record["metrics"] = post.legacy_metrics_from_gait_quality(
            record["gait_quality_metrics"]
        )
        record["h4_safety_acceptance"] = graph["safety"](record)
        record["strict_passed"] = True
        return record

    seeds = post.H4_STRICT_SEEDS["reverse"]
    episodes = [episode(seed) for seed in seeds]
    sources = {
        "selected_reverse_teacher": {
            "path": str(
                (EXP_ROOT / "artifacts" / "h4_reverse_slew_feasible_teacher_selected_v1.json").resolve()
            ),
            "sha256": post.PINNED_SELECTED_REVERSE_TEACHER_SHA256,
        },
        "reverse_composition_authorization": {
            "path": str(
                (EXP_ROOT / "artifacts" / "h4_reverse_training_composition_authorization_v1.json").resolve()
            ),
            "sha256": post.PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256,
        },
    }
    training = {
        "schema_version": 1,
        "training_execution_provider": "JAX_GPU",
        "platform": "gpu",
        "cross_bound_config_manifest_result": True,
        "passed": True,
    }
    candidate = {
        "run_name": "gain024-fixture",
        "expert": "reverse",
        "status": "COMPLETED",
        "activity": "PPO_PILOT_TRAINING",
        "final_params_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
        "resolved_config_sha256": "3" * 64,
        "source_and_teacher_hashes_sha256": post.canonical_json_sha256(sources),
        "training_provenance_sha256": post.canonical_json_sha256(training),
    }
    central = {"safe_gait_experts/gait_quality.py": "4" * 64}
    base_evaluation_hashes = {
        "scripts/evaluate_h4_training_candidate.py": adapter.PINNED_FROZEN_SOURCES[
            "h4_candidate_evaluator"
        ][1],
        "safe_gait_experts/h4_post_training.py": adapter.PINNED_FROZEN_SOURCES[
            "h4_post_training"
        ][1],
    }
    artifact = {
        "schema_version": 1,
        "artifact_kind": adapter.DIAGNOSTIC_ARTIFACT_KIND,
        "hardware_deployment": "PROHIBITED",
        "execution_provider": "CPU",
        "candidate": candidate,
        "evaluation_contract": {
            "fixed_seeds": list(seeds),
            "physical_command_mps_radps": list(
                post.H4_STRICT_COMMANDS["reverse"]
            ),
            "duration_s": post.H4_STRICT_DURATION_S,
            "control_timestep_s": post.H4_CONTROL_DT_S,
            "physics_timestep_s": post.H4_PHYSICS_DT_S,
            "control_tick_count": post.H4_STRICT_CONTROL_TICKS,
            "physics_substep_count": post.H4_STRICT_PHYSICS_SUBSTEPS,
            "gait_sample_count": post.H4_STRICT_GAIT_SAMPLES,
            "gait_quality_semantics": (
                "FULL_CURRENT_P0_RECOMPUTED_FROM_N_PLUS_ONE_SUBSTEP_TRACE"
            ),
            "reverse_composition": (
                "PINNED_PERSISTENT_TEACHER_PLUS_TRAINABLE_RESIDUAL"
            ),
        },
        "central_hashes": central,
        "episodes": copy.deepcopy(episodes),
        "summary": {
            "passing_seed_count": 3,
            "passing_seeds": list(seeds),
            "all_three_strict_pass": True,
        },
        "official_v22_baseline": {
            "source_checkpoint": {
                "kind": "OFFICIAL_FROZEN_V22_BRAX_CHECKPOINT",
                "path": str(EXP_ROOT.resolve()),
                "sha256_tree_pre": "a" * 64,
                "sha256_tree_post": "a" * 64,
                "unchanged": True,
            },
            "transplant_audit": {
                "method": "OFFICIAL_V22_101_TO_116_ZERO_ROW_TRANSPLANT",
                "source_actor_width": 101,
                "target_actor_width": 116,
                "source_critic_width": 212,
                "target_critic_width": 227,
                "insert_offset": 101,
                "inserted_feature_count": 15,
                "optimizer_updates": 0,
                "passed": True,
                "checks": {"fixture": True},
            },
            "transplanted_params_numeric_sha256": "b" * 64,
            "evaluation_process": (
                "SAME_PROCESS_ENVIRONMENT_CONTRACT_AND_FIXED_SEEDS_AS_CANDIDATE"
            ),
            "optimizer_updates": 0,
            "policy_inference": "BRAX_DETERMINISTIC_NORMAL_TANH_ACTOR116",
            "episodes": copy.deepcopy(episodes),
            "summary": {
                "passing_seed_count": 3,
                "passing_seeds": list(seeds),
                "all_three_strict_pass": True,
            },
        },
        "runtime_provenance": {
            "execution_provider": "CPU",
            "jax_default_backend": "cpu",
            "jax_devices": [{"description": "cpu", "platform": "cpu"}],
            "candidate_manifest_sha256": candidate["manifest_sha256"],
            "candidate_final_params_sha256": candidate["final_params_sha256"],
            "candidate_resolved_config_sha256": candidate[
                "resolved_config_sha256"
            ],
            "source_and_teacher_hashes": sources,
            "training_provenance": training,
            "central_hashes": central,
            "evaluation_source_hashes_pre": base_evaluation_hashes,
            "evaluation_source_hashes_post": base_evaluation_hashes,
            "pre_post_source_hashes_unchanged": True,
            "reverse_composition_checks": {"fixture": True},
        },
    }
    validator = graph["artifact_validator"]
    validator.__kwdefaults__["gait_quality_rederive"] = (
        fixture_source._fake_rederive
    )
    validator.__globals__["PINNED_V22_PARENT_TREE_SHA256"] = "a" * 64
    validator.__globals__["sha256_tree"] = lambda _path: "a" * 64
    return artifact, validator, central, base_evaluation_hashes, graph


def test_actual_cloned_validator_accepts_diagnostic_kind_and_augmented_sources() -> None:
    artifact, validator, central, base_hashes, graph = _gain024_fixture()
    source_hashes = {
        adapter.ADAPTER_SOURCE_KEY: "c" * 64,
        adapter.ADAPTER_AUTHORIZATION_SOURCE_KEY: (
            adapter.PINNED_ADAPTER_AUTHORIZATION_SHA256
        ),
    }
    frozen = {"frozen": "d" * 64}
    candidate = {"candidate": "e" * 64}
    augmented_artifact, current = adapter._augment_and_revalidate(
        artifact=artifact,
        bundle=None,
        central_hashes=central,
        base_evaluation_hashes=base_hashes,
        adapter_hashes_pre=source_hashes,
        adapter_hashes_post=source_hashes,
        frozen_hashes_pre=frozen,
        frozen_hashes_post=frozen,
        candidate_hashes_pre=candidate,
        candidate_hashes_post=candidate,
        validator=validator,
    )
    provenance = augmented_artifact["runtime_provenance"]
    assert augmented_artifact["artifact_kind"] == adapter.DIAGNOSTIC_ARTIFACT_KIND
    assert augmented_artifact["promotion_allowed"] is False
    assert provenance["evaluation_source_hashes_pre"] == current
    assert provenance["evaluation_source_hashes_post"] == current
    assert provenance["evaluation_source_hashes_current"] == current
    assert provenance["reverse_v4_gain024_evaluator_adapter"][
        "base_composition_maximum_residual_scale"
    ] == 0.12
    assert provenance["reverse_v4_gain024_evaluator_adapter"][
        "effective_runtime_residual_scale"
    ] == 0.24
    assert augmented_artifact["summary"]["recomputed_validation_passed"] is True
    assert graph["control"].__globals__["H4_REVERSE_RESIDUAL_SCALE"] == 0.24


def test_actual_cloned_validator_rejects_one_episode_legacy_scale() -> None:
    artifact, validator, central, base_hashes, _graph = _gain024_fixture()
    artifact["official_v22_baseline"]["episodes"][2][
        "reverse_composition_contract"
    ]["residual_scale"] = 0.12
    source_hashes = {
        adapter.ADAPTER_SOURCE_KEY: "c" * 64,
        adapter.ADAPTER_AUTHORIZATION_SOURCE_KEY: (
            adapter.PINNED_ADAPTER_AUTHORIZATION_SHA256
        ),
    }
    with pytest.raises(ValueError, match="episode 5"):
        adapter._augment_and_revalidate(
            artifact=artifact,
            bundle=None,
            central_hashes=central,
            base_evaluation_hashes=base_hashes,
            adapter_hashes_pre=source_hashes,
            adapter_hashes_post=source_hashes,
            frozen_hashes_pre={"frozen": "d" * 64},
            frozen_hashes_post={"frozen": "d" * 64},
            candidate_hashes_pre={"candidate": "e" * 64},
            candidate_hashes_post={"candidate": "e" * 64},
            validator=validator,
        )
