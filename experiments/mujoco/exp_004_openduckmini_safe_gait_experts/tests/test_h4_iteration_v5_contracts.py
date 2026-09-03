from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import pytest


EXP_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = EXP_ROOT / "scripts" / "train_h4_aligned_expert.py"
POST_PATH = EXP_ROOT / "safe_gait_experts" / "h4_post_training.py"
SMOKE_PATH = EXP_ROOT / "scripts" / "smoke_h4_training_alignment.py"
FORWARD_AUTH_PATH = (
    EXP_ROOT
    / "artifacts"
    / "h4_forward_iteration_v5_contact_pulse_abort_scale_only_authorization.json"
)
REVERSE_AUTH_PATH = (
    EXP_ROOT
    / "artifacts"
    / "h4_reverse_iteration_v5_no_contact_imitation_authorization.json"
)

FORWARD_AUTH_SHA256 = (
    "c8a197e2b2eeb1b24cce1cace560841bd2620ee6bce5f97506c8c9f7518b210b"
)
REVERSE_AUTH_SHA256 = (
    "1a0da8b77110c92fdaa0a81cdc41879a1a45660456567dfe68f88b2b5deb5976"
)
HISTORICAL_V4_SOURCE_SHA256 = {
    "h4_training_alignment": (
        "872a11a817bb068e3a0819c0afca12ae9e7f2dbfcc103c6569b9081b8d5fbebb"
    ),
    "h4_runner": (
        "b15b9692a72deadd34790d442f4ab4263c3f987255173566a62438e0d380da13"
    ),
    "h4_post_training": (
        "afdfcf9da43a7a7e5824ce7562c489b5e5e20a32e83af817be9e80d740a27b3f"
    ),
    "h4_candidate_evaluator": (
        "c214d086e6d66f6f9f98c7268481899e4133961dcc5355d738d4cd134a82e6ae"
    ),
    "h4_no_ppo_smoke": (
        "410924542bac85f70de3a4055f617a85e93eb841cd403f5280699778ac96710d"
    ),
}
MODE_ATTRIBUTES = (
    "forward_iteration_v2",
    "forward_iteration_v3_touchdown_balance",
    "forward_iteration_v4_contact_event_validity_persistence",
    "forward_v5_contact_pulse_abort_scale_only",
    "reverse_iteration_v2",
    "reverse_iteration_v3_no_target_imitation",
    "reverse_iteration_v4_residual_transfer_gain_024",
    "reverse_iteration_v5_no_contact_imitation",
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_module("exp004_h4_iteration_v5_contract_runner", RUNNER_PATH)
post = _load_module("exp004_h4_iteration_v5_contract_post", POST_PATH)
smoke = _load_module("exp004_h4_iteration_v5_contract_smoke", SMOKE_PATH)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _v5_args(expert: str, *, wiring_only: bool = False) -> Any:
    flag = (
        "--forward-v5-contact-pulse-abort-scale-only"
        if expert == "forward"
        else "--reverse-iteration-v5-no-contact-imitation"
    )
    argv = [
        "--expert",
        expert,
        "--observation-mode",
        "h4_116_transplant",
        "--allow-verified-v22-transplant",
        "--seed",
        "20260809" if expert == "forward" else "20260810",
        "--reset-noise-multiplier",
        "1.0",
        flag,
    ]
    if expert == "reverse":
        argv.extend(("--backward-residual-scale", "0.12"))
    if wiring_only:
        argv.extend(("--wiring-only", "--num-timesteps", "40"))
    else:
        argv.append("--authorize-simulation-training")
    args = runner.build_parser().parse_args(argv)
    args.learning_rate = 5.0e-5 if expert == "forward" else 3.0e-5
    return args


def _synthetic_post_v5_contract(
    expert: str, *, wiring_only: bool
) -> tuple[dict[str, Any], dict[str, dict[str, str]], dict[str, Any]]:
    """Build post inputs from the loader without synthesizing a PPO bundle."""

    authorization = runner.load_iteration_v5_authorization(expert=expert)
    spec = post._iteration_v5_spec(expert)
    payload = authorization["payload"]
    legacy_environment = {
        "target_imitation": 0.0,
        "contact_imitation": 0.0,
        "tracking_sigma": 0.01,
        "backward_residual_scale": 0.12,
    }
    auth_config: dict[str, Any] = {
        "path": str(authorization["path"]),
        "sha256": authorization["sha256"],
        "contract_id": authorization["contract_id"],
        "status": payload["status"],
        "semantic_audit": copy.deepcopy(authorization["semantic_audit"]),
        "bound_causal_inputs": copy.deepcopy(
            authorization["bound_causal_inputs"]
        ),
        "bound_historical_v4_sources": copy.deepcopy(
            authorization["bound_historical_v4_sources"]
        ),
        "scope": "ONE_NEW_SIMULATION_250K_FROM_FROZEN_V22_ONLY",
        "adoption_release_hardware": "PROHIBITED",
    }
    if expert == "reverse":
        auth_config.update(
            {
                "legacy_reward_config_audit": {
                    "expected": legacy_environment,
                    "per_environment": {
                        "train": legacy_environment,
                        "eval": legacy_environment,
                    },
                    "passed": True,
                },
                "rejected_v4_diagnostic_promotion_allowed": False,
            }
        )

    config: dict[str, Any] = {
        mode: mode == spec["flag"] for mode in MODE_ATTRIBUTES
    }
    config.update(
        {
            "expert": expert,
            "wiring_only": wiring_only,
            "activity": (
                "PPO_WIRING_TRAINING" if wiring_only else "PPO_PILOT_TRAINING"
            ),
            "training_contract_id": (
                spec["wiring_contract"] if wiring_only else spec["contract"]
            ),
            "authorized_iteration_v2_250k_contract_id": None,
            "authorized_iteration_v3_250k_contract_id": None,
            "authorized_iteration_v4_250k_contract_id": None,
            "authorized_iteration_v5_250k_contract_id": spec["contract"],
            "initialization_source": "V22_BRAX_CHECKPOINT",
            "trusted_h4_parent": None,
            "pinned_v22_parent_tree_sha256": post.PINNED_V22_PARENT_TREE_SHA256,
            "reset_noise_multiplier": 1.0,
            "reward_scales": copy.deepcopy(
                payload["reward_contract"]["exact_scales"]
            ),
            "forward_v4_substep_contact": expert == "forward",
            "backward_residual_scale": 0.12,
            spec["auth_key"]: auth_config,
        }
    )

    source_hashes: dict[str, dict[str, str]] = {
        spec["auth_label"]: {
            "path": str(authorization["path"]),
            "sha256": authorization["sha256"],
        }
    }
    for source_label in spec["causal_labels"].values():
        bound_key = source_label.removeprefix(spec["prefix"])
        source_hashes[source_label] = copy.deepcopy(
            authorization["bound_causal_inputs"][bound_key]
        )
    for label, relative in post.H4_ITERATION_V4_CAUSAL_SOURCE_PATHS.items():
        path = (EXP_ROOT / relative).resolve()
        source_hashes[f"{spec['prefix']}current_source_{label}"] = {
            "path": str(path),
            "sha256": post.sha256_file(path),
        }
    return config, source_hashes, spec


@pytest.mark.parametrize(
    ("path", "expected_sha", "expert", "kind", "contract_id"),
    (
        (
            FORWARD_AUTH_PATH,
            FORWARD_AUTH_SHA256,
            "forward",
            "openduckmini_h4_forward_iteration_v5_contact_pulse_abort_scale_only_authorization",
            "H4_FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_250K_FROM_V22",
        ),
        (
            REVERSE_AUTH_PATH,
            REVERSE_AUTH_SHA256,
            "reverse",
            "openduckmini_h4_reverse_iteration_v5_no_contact_imitation_authorization",
            "H4_REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_250K_FROM_V22",
        ),
    ),
)
def test_v5_authorizations_are_exact_hash_bound_and_non_promotional(
    path: Path,
    expected_sha: str,
    expert: str,
    kind: str,
    contract_id: str,
) -> None:
    payload = _payload(path)
    assert _sha256(path) == expected_sha
    assert payload["schema_version"] == 1
    assert payload["artifact_kind"] == kind
    assert payload["status"] == "AUTHORIZED_SIMULATION_250K_ONLY"
    assert payload["hardware_deployment"] == "PROHIBITED"
    assert payload["scope"]["expert"] == expert
    assert payload["scope"]["contract_id"] == contract_id
    assert payload["scope"]["training_launch_performed_by_this_artifact"] is False
    assert payload["authorization"] == {
        "simulation_250k_training": True,
        "simulation_1m_training": False,
        "candidate_adoption": False,
        "release": False,
        "hardware": False,
    }
    assert payload["decision"]["training_launch"] == "NOT_PERFORMED"
    assert payload["decision"]["candidate_adoption"] == "BLOCKED"
    assert payload["decision"]["release"] == "BLOCKED"
    assert payload["decision"]["hardware"] == "PROHIBITED"


@pytest.mark.parametrize("path", (FORWARD_AUTH_PATH, REVERSE_AUTH_PATH))
def test_v5_authorizations_bind_historical_v4_snapshot_not_mutable_current_files(
    path: Path,
) -> None:
    closure = _payload(path)["historical_v4_source_closure"]
    assert closure.pop("verification_source") == (
        "BOUND_V4_MANIFEST_PRE_POST_SNAPSHOT_NOT_CURRENT_FILES"
    )
    assert set(closure) == set(HISTORICAL_V4_SOURCE_SHA256)
    assert {
        label: record["sha256"] for label, record in closure.items()
    } == HISTORICAL_V4_SOURCE_SHA256
    assert {
        label: record["path"] for label, record in closure.items()
    } == {
        "h4_training_alignment": "safe_gait_experts/h4_training_alignment.py",
        "h4_runner": "scripts/train_h4_aligned_expert.py",
        "h4_post_training": "safe_gait_experts/h4_post_training.py",
        "h4_candidate_evaluator": "scripts/evaluate_h4_training_candidate.py",
        "h4_no_ppo_smoke": "scripts/smoke_h4_training_alignment.py",
    }
    # The v5 runner is intentionally a new current source.  Historical v4
    # authority must be checked against the bound v4 manifest snapshot, never
    # silently reinterpreted as a hash of this changed file.
    assert _sha256(RUNNER_PATH) != HISTORICAL_V4_SOURCE_SHA256["h4_runner"]


@pytest.mark.parametrize(
    ("expert", "expected_sha"),
    (
        ("forward", FORWARD_AUTH_SHA256),
        ("reverse", REVERSE_AUTH_SHA256),
    ),
)
def test_v5_loader_closes_auth_inputs_and_historical_manifest_snapshots(
    expert: str, expected_sha: str
) -> None:
    authorization = runner.load_iteration_v5_authorization(expert=expert)
    assert set(authorization) == {
        "path",
        "sha256",
        "payload",
        "semantic_audit",
        "contract_id",
        "bound_causal_inputs",
        "bound_historical_v4_sources",
    }
    assert authorization["sha256"] == expected_sha
    assert all(authorization["semantic_audit"].values())
    assert authorization["bound_causal_inputs"]
    historical = authorization["bound_historical_v4_sources"]
    assert set(historical) == set(HISTORICAL_V4_SOURCE_SHA256)
    for label, expected_sha256 in HISTORICAL_V4_SOURCE_SHA256.items():
        record = historical[label]
        assert set(record) == {"path", "sha256", "manifest_pre", "manifest_post"}
        assert record["sha256"] == expected_sha256
        assert record["manifest_pre"] == record["manifest_post"]
        assert record["manifest_pre"]["sha256"] == expected_sha256
        assert record["path"] == authorization["payload"][
            "historical_v4_source_closure"
        ][label]["path"]
        assert record["manifest_pre"]["path"].replace("\\", "/").endswith(
            record["path"]
        )


@pytest.mark.parametrize("expert", ("forward", "reverse"))
@pytest.mark.parametrize("wiring_only", (False, True))
def test_post_v5_synthetic_config_closes_current_and_historical_snapshots(
    expert: str, wiring_only: bool
) -> None:
    config, source_hashes, spec = _synthetic_post_v5_contract(
        expert, wiring_only=wiring_only
    )
    paths = post._validated_iteration_v5_source_paths(
        expert=expert,
        config=config,
        source_hashes=source_hashes,
    )
    assert set(paths) == set(source_hashes)
    assert paths[spec["auth_label"]] == Path(
        source_hashes[spec["auth_label"]]["path"]
    ).resolve()
    for label, relative in post.H4_ITERATION_V4_CAUSAL_SOURCE_PATHS.items():
        current_label = f"{spec['prefix']}current_source_{label}"
        assert paths[current_label] == (EXP_ROOT / relative).resolve()

    auth_config = config[spec["auth_key"]]
    historical = auth_config["bound_historical_v4_sources"]
    v4_manifest = _payload(paths[spec["v4_manifest_label"]])
    for label, record in historical.items():
        manifest_label = f"{spec['v4_source_prefix']}{label}"
        assert record["manifest_pre"] == v4_manifest[
            "source_and_teacher_hashes_pre"
        ][manifest_label]
        assert record["manifest_post"] == v4_manifest[
            "source_and_teacher_hashes_post"
        ][manifest_label]
    assert historical["h4_runner"]["sha256"] != post.sha256_file(RUNNER_PATH)
    if expert == "forward":
        assert config["forward_v4_substep_contact"] is True
        assert config["reward_scales"]["h4_contact_pulse_40ms"] == -2.0
    else:
        assert config["forward_v4_substep_contact"] is False
        assert config["backward_residual_scale"] == 0.12
        assert auth_config["rejected_v4_diagnostic_promotion_allowed"] is False


@pytest.mark.parametrize("expert", ("forward", "reverse"))
@pytest.mark.parametrize("drift", ("missing", "false", "extra"))
def test_post_v5_semantic_audit_schema_fails_closed(
    expert: str, drift: str
) -> None:
    config, source_hashes, spec = _synthetic_post_v5_contract(
        expert, wiring_only=False
    )
    semantic = config[spec["auth_key"]]["semantic_audit"]
    if drift == "missing":
        semantic.pop(next(iter(semantic)))
    elif drift == "false":
        semantic[next(iter(semantic))] = False
    elif drift == "extra":
        semantic["unexpected"] = True
    else:  # pragma: no cover
        raise AssertionError(drift)
    with pytest.raises(ValueError, match="authorization binding drifted"):
        post._validated_iteration_v5_source_paths(
            expert=expert,
            config=config,
            source_hashes=source_hashes,
        )


@pytest.mark.parametrize("expert", ("forward", "reverse"))
@pytest.mark.parametrize("drift", ("current_source", "historical_manifest"))
def test_post_v5_source_snapshot_near_misses_fail_closed(
    expert: str, drift: str
) -> None:
    config, source_hashes, spec = _synthetic_post_v5_contract(
        expert, wiring_only=True
    )
    if drift == "current_source":
        source_hashes[
            f"{spec['prefix']}current_source_h4_post_training"
        ]["sha256"] = "0" * 64
        message = "current source drifted"
    else:
        config[spec["auth_key"]]["bound_historical_v4_sources"]["h4_runner"][
            "manifest_post"
        ]["sha256"] = "0" * 64
        message = "historical source drifted"
    with pytest.raises(ValueError, match=message):
        post._validated_iteration_v5_source_paths(
            expert=expert,
            config=config,
            source_hashes=source_hashes,
        )


def test_forward_v5_has_exactly_one_reward_scale_delta() -> None:
    v4 = runner.forward_iteration_v2_reward_scales().as_reward_scale_dict()
    v5 = (
        runner.forward_iteration_v5_contact_pulse_abort_scale_only_reward_scales()
        .as_reward_scale_dict()
    )
    assert {name for name in v4 if v4[name] != v5[name]} == {
        "h4_contact_pulse_40ms"
    }
    assert v4["h4_contact_pulse_40ms"] == -1.0
    assert v5["h4_contact_pulse_40ms"] == -2.0
    reward = _payload(FORWARD_AUTH_PATH)["reward_contract"]
    assert reward["exact_scales"] == v5
    assert reward["only_scale_delta"] == {
        "name": "h4_contact_pulse_40ms",
        "iteration_v4_scale": -1.0,
        "iteration_v5_scale": -2.0,
    }
    assert reward["all_other_scales_match_iteration_v4"] is True


def test_reverse_v5_has_exactly_one_legacy_scale_delta_and_residual_point12() -> None:
    v3 = dict(runner.REVERSE_ITERATION_V3_NO_TARGET_IMITATION_LEGACY_REWARD_CONFIG)
    v5 = dict(runner.REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_LEGACY_REWARD_CONFIG)
    assert {name for name in v3 if v3[name] != v5[name]} == {
        "contact_imitation"
    }
    assert v3 == {
        "target_imitation": 0.0,
        "contact_imitation": 15.0,
        "tracking_sigma": 0.01,
    }
    assert v5 == {
        "target_imitation": 0.0,
        "contact_imitation": 0.0,
        "tracking_sigma": 0.01,
    }
    payload = _payload(REVERSE_AUTH_PATH)
    assert payload["legacy_reward_config"] == {
        "iteration_v3_baseline": v3,
        "iteration_v5_exact": v5,
        "only_scale_delta": {
            "name": "contact_imitation",
            "iteration_v3_scale": 15.0,
            "iteration_v5_scale": 0.0,
        },
    }
    teacher = payload["teacher_and_guard_contract"]
    assert teacher["backward_residual_scale"] == 0.12
    assert teacher["rejected_v4_backward_residual_scale"] == 0.24
    assert teacher["v4_gain_inherited"] is False
    assert payload["reward_contract"]["exact_scales"] == (
        runner.reverse_iteration_v2_reward_scales().as_reward_scale_dict()
    )


def test_reverse_v4_diagnostic_is_bound_as_non_promotion_evidence() -> None:
    record = _payload(REVERSE_AUTH_PATH)["causal_inputs"]["rejected_v4_diagnostic"]
    assert record["artifact_kind"] == (
        "openduckmini_h4_reverse_iteration_v4_gain024_strict_evaluation_diagnostic"
    )
    assert record["strict_pass_count"] == 0
    assert record["promotion_allowed"] is False
    assert _payload(REVERSE_AUTH_PATH)["manifest_binding"][
        "rejected_v4_diagnostic_never_promotion_evidence"
    ] is True


@pytest.mark.parametrize(
    "first_mode,second_mode", tuple(itertools.combinations(MODE_ATTRIBUTES, 2))
)
def test_all_eight_runner_modes_are_pairwise_mutually_exclusive(
    first_mode: str, second_mode: str
) -> None:
    args = runner.build_parser().parse_args(["--expert", "forward"])
    args.learning_rate = 5.0e-5
    setattr(args, first_mode, True)
    setattr(args, second_mode, True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        runner._validate_scalar_configuration(args)


@pytest.mark.parametrize(
    "first_mode,second_mode", tuple(itertools.combinations(MODE_ATTRIBUTES, 2))
)
def test_all_eight_no_ppo_smoke_modes_are_pairwise_mutually_exclusive(
    first_mode: str, second_mode: str
) -> None:
    args = smoke.build_parser().parse_args([])
    setattr(args, first_mode, True)
    setattr(args, second_mode, True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        smoke._validate_smoke_cli(args)


@pytest.mark.parametrize("expert", ("forward", "reverse"))
@pytest.mark.parametrize("wiring_only", (False, True))
def test_v5_runner_cli_separates_wiring_and_full_contracts(
    expert: str, wiring_only: bool
) -> None:
    args = _v5_args(expert, wiring_only=wiring_only)
    runner._validate_scalar_configuration(args)
    authorization = runner.load_iteration_v5_authorization(expert=expert)
    actual = runner.resolve_execution_contract_id(
        args,
        forward_iteration_v2_authorization=None,
        forward_iteration_v3_touchdown_balance_authorization=None,
        forward_iteration_v4_contact_event_validity_persistence_authorization=None,
        forward_v5_contact_pulse_abort_scale_only_authorization=(
            authorization if expert == "forward" else None
        ),
        reverse_iteration_v2_authorization=None,
        reverse_iteration_v3_no_target_imitation_authorization=None,
        reverse_iteration_v4_residual_transfer_gain_024_authorization=None,
        reverse_iteration_v5_no_contact_imitation_authorization=(
            authorization if expert == "reverse" else None
        ),
    )
    assert actual == (
        authorization["payload"]["scope"]["wiring_contract_id"]
        if wiring_only
        else authorization["contract_id"]
    )
    shape = runner.resolve_training_shape(args, runner._load_legacy_trainer())
    assert (shape.num_timesteps, shape.num_envs) == (
        (40, 2) if wiring_only else (250_000, 1250)
    )
    if expert == "reverse":
        assert args.backward_residual_scale == 0.12


@pytest.mark.parametrize("expert", ("forward", "reverse"))
@pytest.mark.parametrize("wiring_only", (False, True))
def test_v5_runner_authorized_contract_closes_shape_rewards_and_anchors(
    expert: str, wiring_only: bool
) -> None:
    args = _v5_args(expert, wiring_only=wiring_only)
    runner._validate_scalar_configuration(args)
    shape = runner.resolve_training_shape(args, runner._load_legacy_trainer())
    reward_scales = runner.resolve_reward_scales(args).as_reward_scale_dict()
    anchors = runner.resolve_anchor_config(
        expert,
        forward_v5_contact_pulse_abort_scale_only=(expert == "forward"),
        reverse_iteration_v5_no_contact_imitation=(expert == "reverse"),
    )
    v5_authorization = runner.load_iteration_v5_authorization(expert=expert)
    runner._validate_authorized_training_contract(
        args,
        shape=shape,
        reward_scale_dict=reward_scales,
        anchors=anchors,
        forward_spec=(runner.load_forward_minimum_spec() if expert == "forward" else None),
        forward_iteration_v2_authorization=None,
        forward_iteration_v3_touchdown_balance_authorization=None,
        forward_iteration_v4_contact_event_validity_persistence_authorization=None,
        forward_v5_contact_pulse_abort_scale_only_authorization=(
            v5_authorization if expert == "forward" else None
        ),
        reverse_spec=(runner.load_reverse_minimum_spec() if expert == "reverse" else None),
        reverse_authorization=(
            runner.load_reverse_composition_authorization()
            if expert == "reverse"
            else None
        ),
        reverse_iteration_v2_authorization=None,
        reverse_iteration_v3_no_target_imitation_authorization=None,
        reverse_iteration_v4_residual_transfer_gain_024_authorization=None,
        reverse_iteration_v5_no_contact_imitation_authorization=(
            v5_authorization if expert == "reverse" else None
        ),
    )


@pytest.mark.parametrize(
    ("expert", "flag", "seed", "mode", "preflight_id"),
    (
        (
            "forward",
            "--forward-v5-contact-pulse-abort-scale-only",
            "20260809",
            "forward_v5_contact_pulse_abort_scale_only",
            "H4_FORWARD_ITERATION_V5_CONTACT_PULSE_ABORT_SCALE_ONLY_NO_PPO_PREFLIGHT_FROM_V22",
        ),
        (
            "reverse",
            "--reverse-iteration-v5-no-contact-imitation",
            "20260810",
            "reverse_iteration_v5_no_contact_imitation",
            "H4_REVERSE_ITERATION_V5_NO_CONTACT_IMITATION_NO_PPO_PREFLIGHT_FROM_V22",
        ),
    ),
)
def test_v5_no_ppo_smoke_resolves_exact_diagnostic_contract(
    expert: str,
    flag: str,
    seed: str,
    mode: str,
    preflight_id: str,
) -> None:
    args = smoke.build_parser().parse_args(
        [
            "--expert",
            expert,
            "--seed",
            seed,
            flag,
            "--reset-noise-multiplier",
            "1.0",
        ]
    )
    backend = smoke._load_h4_runner_backend_contract()
    contract = smoke.resolve_smoke_contract(args, backend)
    assert contract["mode"] == mode
    assert contract["preflight_contract_id"] == preflight_id
    assert contract["authorized_250k_contract_id"].endswith("_250K_FROM_V22")
    assert all(contract["authorization"]["semantic_audit"].values())
    if expert == "forward":
        assert contract["forward_v4_substep_contact"] is True
        assert contract["reward_scales"].as_reward_scale_dict()[
            "h4_contact_pulse_40ms"
        ] == -2.0
        assert contract["legacy_reward_config_overrides"] is None
    else:
        assert contract["forward_v4_substep_contact"] is False
        assert contract["backward_residual_scale"] == 0.12
        assert contract["legacy_reward_config_overrides"] == {
            "target_imitation": 0.0,
            "contact_imitation": 0.0,
            "tracking_sigma": 0.01,
        }


@dataclass
class _TrainingShape:
    num_timesteps: int
    num_envs: int
    unroll_length: int
    batch_size: int
    num_minibatches: int
    num_updates_per_batch: int
    num_evals: int

    @property
    def interactions_per_training_step(self) -> int:
        return self.num_timesteps

    @property
    def expected_training_steps(self) -> int:
        return 1

    @property
    def expected_optimizer_updates(self) -> int:
        return self.num_minibatches * self.num_updates_per_batch


def test_v5_authorization_failure_occurs_before_any_pickle_or_training_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _v5_args("forward")
    pickle_calls: list[object] = []

    def forbidden_pickle(*_args: object, **_kwargs: object) -> object:
        pickle_calls.append(object())
        raise AssertionError("pickle must not be opened before v5 authorization")

    def rejected_authorization(*_args: object, **_kwargs: object) -> object:
        raise ValueError("injected v5 authorization drift")

    monkeypatch.setattr(runner, "_load_legacy_trainer", lambda: type(
        "FakeTrainer", (), {"TrainingShape": _TrainingShape}
    )())
    monkeypatch.setattr(runner.pickle, "load", forbidden_pickle)
    monkeypatch.setattr(
        runner,
        "load_forward_iteration_v5_contact_pulse_abort_scale_only_authorization",
        rejected_authorization,
    )
    with pytest.raises(ValueError, match="injected v5 authorization drift"):
        runner.run(args)
    assert pickle_calls == []


@pytest.mark.parametrize("expert", ("forward", "reverse"))
def test_v5_authorization_semantic_mutation_fails_closed(expert: str) -> None:
    path = FORWARD_AUTH_PATH if expert == "forward" else REVERSE_AUTH_PATH
    broken = copy.deepcopy(_payload(path))
    broken["authorization"]["candidate_adoption"] = True
    with pytest.raises(ValueError, match="iteration-v5 authorization drifted"):
        runner.validate_iteration_v5_authorization_payload(broken, expert=expert)


@pytest.mark.parametrize("expert", ("forward", "reverse"))
@pytest.mark.parametrize(
    "mutation",
    (
        "scope_purpose",
        "manifest_binding_false",
        "causal_extra_key",
        "causal_record_path",
    ),
)
def test_v5_authorization_nested_schema_mutations_fail_closed(
    expert: str, mutation: str
) -> None:
    path = FORWARD_AUTH_PATH if expert == "forward" else REVERSE_AUTH_PATH
    broken = copy.deepcopy(_payload(path))
    if mutation == "scope_purpose":
        broken["scope"]["purpose"] = "mutated"
    elif mutation == "manifest_binding_false":
        key = next(iter(broken["manifest_binding"]))
        broken["manifest_binding"][key] = False
    elif mutation == "causal_extra_key":
        broken["causal_inputs"]["unexpected"] = {
            "path": "unexpected",
            "sha256": "0" * 64,
        }
    elif mutation == "causal_record_path":
        key = "final_params" if expert == "forward" else "v3_final_params"
        broken["causal_inputs"][key]["path"] = "wrong.pkl"
    else:  # pragma: no cover
        raise AssertionError(mutation)
    with pytest.raises(ValueError, match="iteration-v5 authorization drifted"):
        runner.validate_iteration_v5_authorization_payload(broken, expert=expert)


@pytest.mark.parametrize(
    ("expert", "source"),
    (("forward", FORWARD_AUTH_PATH), ("reverse", REVERSE_AUTH_PATH)),
)
def test_v5_loader_rejects_authorization_byte_drift_before_semantics(
    tmp_path: Path, expert: str, source: Path
) -> None:
    drifted = tmp_path / source.name
    drifted.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="authorization SHA256 drifted"):
        runner.load_iteration_v5_authorization(expert=expert, path=drifted)


@pytest.mark.parametrize("expert", ("forward", "reverse"))
def test_active_v4_loader_rejects_stale_current_source_after_v5_change(
    expert: str,
) -> None:
    loader = (
        runner.load_forward_iteration_v4_contact_event_validity_persistence_authorization
        if expert == "forward"
        else runner.load_reverse_iteration_v4_residual_transfer_gain_024_authorization
    )
    with pytest.raises(ValueError, match="iteration-v4 causal source drifted"):
        loader()
