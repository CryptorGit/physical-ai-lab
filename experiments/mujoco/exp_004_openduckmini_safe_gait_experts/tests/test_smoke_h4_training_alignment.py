from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_PATH = EXPERIMENT_ROOT / "scripts" / "smoke_h4_training_alignment.py"
SPEC = importlib.util.spec_from_file_location(
    "exp004_smoke_h4_training_alignment", SMOKE_PATH
)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = smoke
SPEC.loader.exec_module(smoke)


def _v6_authorization(contract_id: str) -> dict[str, object]:
    return {
        "contract_id": contract_id,
        "semantic_audit": {
            "top_level_fields_exact": True,
            "nested_schema_exact": True,
            "numeric_types_and_values_exact": True,
            "causal_identity_exact": True,
            "training_contract_exact": True,
            "runtime_contract_exact": True,
            "passed": True,
        },
    }


@pytest.mark.parametrize(
    ("expert", "flag", "seed", "contract_id"),
    (
        (
            "forward",
            "--forward-iteration-v2",
            "20260809",
            "H4_FORWARD_ITERATION_V2_250K_FROM_V22",
        ),
        (
            "reverse",
            "--reverse-iteration-v2",
            "20260810",
            "H4_REVERSE_ITERATION_V2_250K_FROM_V22",
        ),
    ),
)
def test_no_ppo_smoke_resolves_exact_v2_contract_without_optimizer(
    expert: str, flag: str, seed: str, contract_id: str
) -> None:
    args = smoke.build_parser().parse_args(
        ["--expert", expert, "--seed", seed, flag]
    )
    runner = smoke._load_h4_runner_backend_contract()
    contract = smoke.resolve_smoke_contract(args, runner)

    assert contract["mode"] == f"{expert}_iteration_v2"
    assert contract["preflight_contract_id"].endswith(
        "_ITERATION_V2_NO_PPO_PREFLIGHT_FROM_V22"
    )
    assert contract["authorized_250k_contract_id"] == contract_id
    assert all(contract["authorization"]["semantic_audit"].values())
    assert len(contract["authorization"]["bound_causal_inputs"]) == 3
    assert contract["authorization"]["payload"]["manifest_binding"]
    assert contract["minimum_spec"]["sha256"]
    probability_sum = sum(
        contract["anchors"][key]
        for key in (
            "stand_probability",
            "exact_primary_probability",
            "local_probability",
            "transition_probability",
        )
    )
    assert probability_sum == pytest.approx(1.0)
    scales = contract["reward_scales"].as_reward_scale_dict()
    if expert == "forward":
        assert scales["h4_total_normal_force_band"] == -1.0
        assert scales["h4_contact_pulse_40ms"] == -1.0
        assert contract["legacy_reward_config_overrides"] is None
    else:
        assert scales["h4_reverse_speed_boundary"] == -8.0
        assert scales["h4_total_normal_force_band"] == 0.0
        assert contract["legacy_reward_config_overrides"] == {
            "target_imitation": -20.0,
            "contact_imitation": 15.0,
            "tracking_sigma": 0.01,
        }
        assert contract["reverse_composition"]["sha256"]


@pytest.mark.parametrize(
    ("argv", "message"),
    (
        (
            (
                "--expert",
                "forward",
                "--seed",
                "20260809",
                "--forward-iteration-v2",
                "--reverse-iteration-v2",
            ),
            "mutually exclusive",
        ),
        (
            ("--expert", "reverse", "--seed", "20260809", "--forward-iteration-v2"),
            "only for forward",
        ),
        (
            ("--expert", "reverse", "--seed", "20260809", "--reverse-iteration-v2"),
            "seed 20260810",
        ),
        (
            (
                "--expert",
                "forward",
                "--seed",
                "20260809",
                "--forward-iteration-v2",
                "--physical-command",
                "0.05,0,0",
            ),
            "forbids command overrides",
        ),
        (
            (
                "--expert",
                "forward",
                "--seed",
                "20260809",
                "--forward-iteration-v2",
                "--steps",
                "6",
            ),
            "steps must be",
        ),
    ),
)
def test_no_ppo_smoke_v2_cross_mode_shape_and_flag_near_misses_fail_closed(
    argv: tuple[str, ...], message: str
) -> None:
    args = smoke.build_parser().parse_args(argv)
    with pytest.raises(ValueError, match=message):
        smoke._validate_smoke_cli(args)


def test_no_ppo_smoke_rejects_training_authorization_flag() -> None:
    with pytest.raises(SystemExit):
        smoke.build_parser().parse_args(
            [
                "--expert",
                "forward",
                "--seed",
                "20260809",
                "--forward-iteration-v2",
                "--authorize-simulation-training",
            ]
        )


@pytest.mark.parametrize(
    ("expert", "flag", "seed", "mode", "preflight_id", "authorized_id"),
    (
        (
            "forward",
            "--forward-iteration-v3-touchdown-balance",
            "20260809",
            "forward_iteration_v3_touchdown_balance",
            "H4_FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_NO_PPO_PREFLIGHT_FROM_V22",
            "H4_FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_250K_FROM_V22",
        ),
        (
            "reverse",
            "--reverse-iteration-v3-no-target-imitation",
            "20260810",
            "reverse_iteration_v3_no_target_imitation",
            "H4_REVERSE_ITERATION_V3_NO_TARGET_IMITATION_NO_PPO_PREFLIGHT_FROM_V22",
            "H4_REVERSE_ITERATION_V3_NO_TARGET_IMITATION_250K_FROM_V22",
        ),
    ),
)
def test_no_ppo_smoke_resolves_exact_v3_single_factor_contract(
    expert: str,
    flag: str,
    seed: str,
    mode: str,
    preflight_id: str,
    authorized_id: str,
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
    runner = smoke._load_h4_runner_backend_contract()
    contract = smoke.resolve_smoke_contract(args, runner)
    assert contract["mode"] == mode
    assert contract["preflight_contract_id"] == preflight_id
    assert contract["authorized_250k_contract_id"] == authorized_id
    assert all(contract["authorization"]["semantic_audit"].values())
    assert len(contract["authorization"]["bound_causal_inputs"]) == 3
    assert sum(
        contract["anchors"][key]
        for key in (
            "stand_probability",
            "exact_primary_probability",
            "local_probability",
            "transition_probability",
        )
    ) == pytest.approx(1.0)
    scales = contract["reward_scales"].as_reward_scale_dict()
    if expert == "forward":
        v2 = runner.forward_iteration_v2_reward_scales().as_reward_scale_dict()
        assert {name for name in v2 if v2[name] != scales[name]} == {
            "h4_touchdown_count_balance"
        }
        assert scales["h4_touchdown_count_balance"] == -4.0
        assert contract["legacy_reward_config_overrides"] is None
    else:
        assert scales == runner.reverse_iteration_v2_reward_scales().as_reward_scale_dict()
        assert contract["legacy_reward_config_overrides"] == {
            "target_imitation": 0.0,
            "contact_imitation": 15.0,
            "tracking_sigma": 0.01,
        }
        assert contract["reverse_composition"]["sha256"]


@pytest.mark.parametrize(
    ("argv", "message"),
    (
        (
            (
                "--expert",
                "forward",
                "--seed",
                "20260809",
                "--forward-iteration-v2",
                "--forward-iteration-v3-touchdown-balance",
            ),
            "mutually exclusive",
        ),
        (
            (
                "--expert",
                "reverse",
                "--seed",
                "20260810",
                "--reverse-iteration-v3-no-target-imitation",
                "--reset-noise-multiplier",
                "0",
            ),
            "reset-noise multiplier 1.0",
        ),
        (
            (
                "--expert",
                "forward",
                "--seed",
                "20260810",
                "--forward-iteration-v3-touchdown-balance",
                "--reset-noise-multiplier",
                "1",
            ),
            "seed 20260809",
        ),
        (
            (
                "--expert",
                "forward",
                "--seed",
                "20260810",
                "--reverse-iteration-v3-no-target-imitation",
                "--reset-noise-multiplier",
                "1",
            ),
            "only for reverse",
        ),
        (
            (
                "--expert",
                "forward",
                "--seed",
                "20260809",
                "--forward-iteration-v3-touchdown-balance",
                "--reset-noise-multiplier",
                "1",
                "--policy-command",
                "0.1,-0.018,-0.17",
            ),
            "forbids command overrides",
        ),
    ),
)
def test_no_ppo_smoke_v3_near_misses_fail_closed(
    argv: tuple[str, ...], message: str
) -> None:
    args = smoke.build_parser().parse_args(argv)
    with pytest.raises(ValueError, match=message):
        smoke._validate_smoke_cli(args)


@pytest.mark.parametrize(
    ("expert", "flag", "seed"),
    (
        (
            "forward",
            "--forward-iteration-v4-contact-event-validity-persistence",
            "20260809",
        ),
        (
            "reverse",
            "--reverse-iteration-v4-residual-transfer-gain-024",
            "20260810",
        ),
    ),
)
def test_no_ppo_smoke_v4_active_loader_rejects_stale_current_source_closure(
    expert: str,
    flag: str,
    seed: str,
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
    runner = smoke._load_h4_runner_backend_contract()
    with pytest.raises(ValueError, match="iteration-v4 causal source drifted"):
        smoke.resolve_smoke_contract(args, runner)


@pytest.mark.parametrize(
    ("argv", "message"),
    (
        (
            (
                "--expert",
                "forward",
                "--seed",
                "20260809",
                "--forward-iteration-v3-touchdown-balance",
                "--forward-iteration-v4-contact-event-validity-persistence",
                "--reset-noise-multiplier",
                "1",
            ),
            "mutually exclusive",
        ),
        (
            (
                "--expert",
                "reverse",
                "--seed",
                "20260810",
                "--reverse-iteration-v3-no-target-imitation",
                "--reverse-iteration-v4-residual-transfer-gain-024",
                "--reset-noise-multiplier",
                "1",
            ),
            "mutually exclusive",
        ),
        (
            (
                "--expert",
                "reverse",
                "--seed",
                "20260809",
                "--forward-iteration-v4-contact-event-validity-persistence",
                "--reset-noise-multiplier",
                "1",
            ),
            "only for forward",
        ),
        (
            (
                "--expert",
                "reverse",
                "--seed",
                "20260809",
                "--reverse-iteration-v4-residual-transfer-gain-024",
                "--reset-noise-multiplier",
                "1",
            ),
            "seed 20260810",
        ),
        (
            (
                "--expert",
                "forward",
                "--seed",
                "20260809",
                "--forward-iteration-v4-contact-event-validity-persistence",
                "--reset-noise-multiplier",
                "0",
            ),
            "reset-noise multiplier 1.0",
        ),
    ),
)
def test_no_ppo_smoke_v4_near_misses_fail_closed(
    argv: tuple[str, ...], message: str
) -> None:
    args = smoke.build_parser().parse_args(argv)
    with pytest.raises(ValueError, match=message):
        smoke._validate_smoke_cli(args)


@pytest.mark.parametrize(
    ("expert", "flag", "seed", "mode", "full_id", "no_ppo_id"),
    (
        (
            "forward",
            "--forward-iteration-v6-contact-abort-island-only",
            "20260809",
            "forward_iteration_v6_contact_abort_island_only",
            "H4_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_250K_FROM_V22",
            "H4_FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_NO_PPO_PREFLIGHT_FROM_V22",
        ),
        (
            "reverse",
            "--reverse-iteration-v6-absolute-full-leg-targets",
            "20260810",
            "reverse_iteration_v6_absolute_full_leg_targets",
            "H4_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_250K_FROM_V22",
            "H4_REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_NO_PPO_PREFLIGHT_FROM_V22",
        ),
    ),
)
def test_no_ppo_smoke_resolves_v6_without_ppo_or_pickle(
    expert: str,
    flag: str,
    seed: str,
    mode: str,
    full_id: str,
    no_ppo_id: str,
    monkeypatch: pytest.MonkeyPatch,
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
    runner = smoke._load_h4_runner_backend_contract()
    monkeypatch.setattr(
        runner,
        (
            "load_forward_iteration_v6_contact_abort_island_only_authorization"
            if expert == "forward"
            else "load_reverse_iteration_v6_absolute_full_leg_targets_authorization"
        ),
        lambda: _v6_authorization(full_id),
    )
    monkeypatch.setattr(
        runner,
        "load_forward_minimum_spec" if expert == "forward" else "load_reverse_minimum_spec",
        lambda: {"sha256": "f" * 64},
    )
    if expert == "reverse":
        monkeypatch.setattr(
            runner,
            "load_reverse_composition_authorization",
            lambda: {"sha256": "e" * 64},
        )
    contract = smoke.resolve_smoke_contract(args, runner)
    assert contract["mode"] == mode
    assert contract["preflight_contract_id"] == no_ppo_id
    assert contract["authorized_250k_contract_id"] == full_id
    assert contract["forward_iteration_v6_contact_abort_island_only"] is (
        expert == "forward"
    )
    assert contract["reverse_iteration_v6_absolute_full_leg_targets"] is (
        expert == "reverse"
    )
    assert contract["forward_v4_substep_contact"] is (expert == "forward")
    assert contract["backward_residual_scale"] == (
        0.12 if expert == "forward" else 0.0
    )
    assert contract["iteration_v6_core_source"] == {
        "path": str(runner.ALIGNMENT_MODULE_PATH.resolve()),
        "sha256": runner.PINNED_ITERATION_V6_H4_TRAINING_ALIGNMENT_SHA256,
    }
    assert contract["reward_scales"].as_reward_scale_dict() == (
        runner.forward_iteration_v2_reward_scales().as_reward_scale_dict()
        if expert == "forward"
        else runner.reverse_iteration_v2_reward_scales().as_reward_scale_dict()
    )
    assert contract["legacy_reward_config_overrides"] == (
        None
        if expert == "forward"
        else {
            "target_imitation": 0.0,
            "contact_imitation": 0.0,
            "tracking_sigma": 0.01,
        }
    )


def test_no_ppo_v6_core_drift_fails_before_authorization_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = smoke.build_parser().parse_args(
        [
            "--expert",
            "forward",
            "--seed",
            "20260809",
            "--forward-iteration-v6-contact-abort-island-only",
            "--reset-noise-multiplier",
            "1.0",
        ]
    )
    runner = smoke._load_h4_runner_backend_contract()
    calls = {"authorization": 0}

    def reject_core() -> object:
        raise ValueError("iteration-v6 core source SHA256 drifted")

    def record_authorization() -> object:
        calls["authorization"] += 1
        raise AssertionError("authorization load reached")

    monkeypatch.setattr(runner, "require_iteration_v6_core_source", reject_core)
    monkeypatch.setattr(
        runner,
        "load_forward_iteration_v6_contact_abort_island_only_authorization",
        record_authorization,
    )
    with pytest.raises(ValueError, match="core source SHA256 drifted"):
        smoke.resolve_smoke_contract(args, runner)
    assert calls == {"authorization": 0}


@pytest.mark.parametrize(
    ("argv", "message"),
    (
        (
            (
                "--expert",
                "forward",
                "--seed",
                "20260809",
                "--forward-iteration-v6-contact-abort-island-only",
                "--forward-v5-contact-pulse-abort-scale-only",
                "--reset-noise-multiplier",
                "1.0",
            ),
            "mutually exclusive",
        ),
        (
            (
                "--expert",
                "reverse",
                "--seed",
                "20260810",
                "--forward-iteration-v6-contact-abort-island-only",
                "--reset-noise-multiplier",
                "1.0",
            ),
            "only for forward",
        ),
        (
            (
                "--expert",
                "reverse",
                "--seed",
                "20260809",
                "--reverse-iteration-v6-absolute-full-leg-targets",
                "--reset-noise-multiplier",
                "1.0",
            ),
            "seed 20260810",
        ),
        (
            (
                "--expert",
                "forward",
                "--seed",
                "20260809",
                "--forward-iteration-v6-contact-abort-island-only",
                "--reset-noise-multiplier",
                "0.0",
            ),
            "reset-noise multiplier 1.0",
        ),
    ),
)
def test_no_ppo_smoke_v6_near_misses_fail_before_backend_load(
    argv: tuple[str, ...], message: str
) -> None:
    args = smoke.build_parser().parse_args(argv)
    with pytest.raises(ValueError, match=message):
        smoke._validate_smoke_cli(args)


def _forward_v4_single_authority_trace() -> dict[str, object]:
    return {
        "forward_v4_single_authority": {
            "dynamic6_exact": True,
            "dynamic6_max_abs_error": 0.0,
            "dynamic6_field_count": 6,
            "dynamic6_field_count_exact": True,
            "saved_dynamic6_substep_count": 10,
            "saved_dynamic6_field_count": 6,
            "saved_dynamic6_field_count_exact": True,
            "saved_dynamic6_all_finite": True,
            "telemetry_force_shape_valid": True,
            "telemetry_force_all_finite": True,
            "authority_violation": False,
            "assertion_token": 0,
        }
    }


def test_no_ppo_forward_v4_result_binds_exact_single_authority_runtime() -> None:
    backend = smoke._load_h4_runner_backend_contract()
    traces = [_forward_v4_single_authority_trace() for _ in range(4)]
    assert smoke.require_forward_v4_no_ppo_single_authority(
        traces, backend
    ) == {
        "dynamic6_exact": True,
        "dynamic6_max_abs_error": 0.0,
        "dynamic6_field_count": 6,
        "dynamic6_field_count_exact": True,
        "saved_dynamic6_substep_count": 10,
        "saved_dynamic6_field_count": 6,
        "saved_dynamic6_field_count_exact": True,
        "saved_dynamic6_all_finite": True,
        "telemetry_force_shape": [2],
        "telemetry_force_shape_valid": True,
        "telemetry_force_all_finite": True,
        "authority_violation_count": 0,
        "assertion_token_sum": 0.0,
        "observed_step_count": 4,
        "passed": True,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("dynamic6_exact", False),
        ("dynamic6_max_abs_error", 1.0e-12),
        ("dynamic6_field_count", 0),
        ("dynamic6_field_count_exact", False),
        ("saved_dynamic6_substep_count", 9),
        ("saved_dynamic6_all_finite", False),
        ("saved_dynamic6_field_count_exact", False),
        ("telemetry_force_shape_valid", False),
        ("telemetry_force_all_finite", False),
        ("authority_violation", True),
        ("assertion_token", 1),
    ),
)
def test_no_ppo_forward_v4_injected_single_authority_mismatch_aborts(
    field: str, value: object
) -> None:
    backend = smoke._load_h4_runner_backend_contract()
    trace = _forward_v4_single_authority_trace()
    trace["forward_v4_single_authority"][field] = value
    with pytest.raises(RuntimeError, match="single-authority audit failed"):
        smoke.require_forward_v4_no_ppo_single_authority([trace], backend)
