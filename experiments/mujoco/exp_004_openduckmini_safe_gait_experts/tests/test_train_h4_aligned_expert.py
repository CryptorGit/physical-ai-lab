from __future__ import annotations

import ast
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = EXPERIMENT_ROOT / "scripts" / "train_h4_aligned_expert.py"
SPEC = importlib.util.spec_from_file_location("exp004_train_h4_aligned", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_runner_ast_has_no_duplicate_boolean_terms_or_literal_dict_keys() -> None:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.BoolOp):
            terms = [ast.dump(value, include_attributes=False) for value in node.values]
            assert len(terms) == len(set(terms)), f"duplicate bool term at {node.lineno}"
        if isinstance(node, ast.Dict):
            literal_keys = [
                ast.literal_eval(key)
                for key in node.keys
                if isinstance(key, ast.Constant)
            ]
            assert len(literal_keys) == len(set(literal_keys)), (
                f"duplicate literal dict key at {node.lineno}"
            )


def test_run_passes_both_iteration_v6_authorizations_to_execution_contract_resolver() -> None:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    run_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    calls = [
        node
        for node in ast.walk(run_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "resolve_execution_contract_id"
    ]
    assert len(calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    expected = {
        "forward_iteration_v6_contact_abort_island_only_authorization",
        "reverse_iteration_v6_absolute_full_leg_targets_authorization",
    }
    assert expected <= set(keywords)
    for name in expected:
        assert isinstance(keywords[name], ast.Name)
        assert keywords[name].id == name


class _FakeDevice:
    def __init__(self, platform: str, label: str = "device:0") -> None:
        self.platform = platform
        self.label = label

    def __str__(self) -> str:
        return self.label


class _FakeJax:
    def __init__(
        self,
        backend: str,
        device_platforms: tuple[str, ...],
        cpu_callback_platforms: tuple[str, ...] = ("cpu",),
    ) -> None:
        self.backend = backend
        self._devices = tuple(
            _FakeDevice(platform, f"{platform}:{index}")
            for index, platform in enumerate(device_platforms)
        )
        self._cpu_devices = tuple(
            _FakeDevice(platform, f"{platform}-callback:{index}")
            for index, platform in enumerate(cpu_callback_platforms)
        )

    def default_backend(self) -> str:
        return self.backend

    def devices(self, backend: str | None = None) -> tuple[_FakeDevice, ...]:
        if backend == "cpu":
            return self._cpu_devices
        if backend is None or backend == self.backend:
            return self._devices
        return ()


def test_jax_backend_selector_maps_gpu_to_cuda_with_cpu_callbacks() -> None:
    assert runner.resolve_jax_backend_selector("cpu") == "cpu"
    assert runner.resolve_jax_backend_selector("gpu") == "cuda,cpu"
    audit = runner.validate_resolved_jax_backend(
        _FakeJax("gpu", ("gpu",)),
        requested_platform="gpu",
        selector="cuda,cpu",
    )
    assert audit == {
        "requested_cli_platform": "gpu",
        "jax_platform_selector": "cuda,cpu",
        "expected_resolved_backend": "gpu",
        "resolved_default_backend": "gpu",
        "resolved_device_platforms": ["gpu"],
        "resolved_devices": ["gpu:0"],
        "local_cpu_callback_devices": ["cpu-callback:0"],
        "local_cpu_callback_available": True,
        "passed": True,
    }


@pytest.mark.parametrize("near_miss", ("cuda", "GPU", "rocm", ""))
def test_jax_backend_selector_rejects_cli_near_miss(near_miss: str) -> None:
    with pytest.raises(ValueError, match="unsupported requested"):
        runner.resolve_jax_backend_selector(near_miss)


@pytest.mark.parametrize("initial", (None, ""))
def test_gpu_xla_policy_pins_correctness_checked_level4_before_import(
    initial: str | None,
) -> None:
    environ = {} if initial is None else {"XLA_FLAGS": initial}
    audit = runner.configure_xla_autotune_policy("gpu", environ=environ)
    assert environ["XLA_FLAGS"] == "--xla_gpu_autotune_level=4"
    assert audit["xla_flags_before"] == initial
    assert audit["xla_flags_effective"] == "--xla_gpu_autotune_level=4"
    assert audit["policy"] == "CORRECTNESS_CHECKED_LEVEL4_DISQUALIFY_MISMATCH"
    assert audit["configured_before_training_stack_import"] is True
    assert audit["correctness_check_enabled"] is True
    assert audit["mismatching_autotune_candidates_disqualified"] is True


@pytest.mark.parametrize("initial", (None, ""))
def test_cpu_xla_policy_does_not_set_gpu_flag(initial: str | None) -> None:
    environ = {} if initial is None else {"XLA_FLAGS": initial}
    audit = runner.configure_xla_autotune_policy("cpu", environ=environ)
    assert environ.get("XLA_FLAGS") == initial
    assert audit["xla_flags_effective"] == ""
    assert audit["policy"] == "CPU_NO_GPU_AUTOTUNE_FLAG"
    assert audit["correctness_check_enabled"] is False
    assert audit["cpu_mode_did_not_set_xla_flags"] is True


@pytest.mark.parametrize(
    "override",
    (
        "--xla_gpu_autotune_level=4",
        "--xla_gpu_autotune_level=0",
        " --xla_gpu_autotune_level=4",
        "--xla_gpu_autotune_level=4 ",
        "--xla_force_host_platform_device_count=2",
        "--xla_gpu_autotune_level=4 --xla_dump_to=/tmp/xla",
    ),
)
@pytest.mark.parametrize("platform", ("cpu", "gpu"))
def test_xla_policy_rejects_every_preexisting_override(
    platform: str, override: str
) -> None:
    with pytest.raises(ValueError, match="preexisting XLA_FLAGS override"):
        runner.configure_xla_autotune_policy(
            platform, environ={"XLA_FLAGS": override}
        )


@pytest.mark.parametrize(
    (
        "requested",
        "selector",
        "backend",
        "device_platforms",
        "cpu_callback_platforms",
        "error",
    ),
    (
        ("gpu", "gpu", "gpu", ("gpu",), ("cpu",), ValueError),
        ("gpu", "cuda", "gpu", ("gpu",), ("cpu",), ValueError),
        ("gpu", "cuda,cpu", "cpu", ("cpu",), ("cpu",), RuntimeError),
        ("gpu", "cuda,cpu", "gpu", ("cpu",), ("cpu",), RuntimeError),
        ("gpu", "cuda,cpu", "gpu", (), ("cpu",), RuntimeError),
        ("gpu", "cuda,cpu", "gpu", ("gpu",), (), RuntimeError),
        ("cpu", "cpu", "cpu", ("cpu",), ("cpu", "gpu"), RuntimeError),
    ),
)
def test_jax_backend_resolution_fails_closed_on_any_mismatch(
    requested: str,
    selector: str,
    backend: str,
    device_platforms: tuple[str, ...],
    cpu_callback_platforms: tuple[str, ...],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        runner.validate_resolved_jax_backend(
            _FakeJax(backend, device_platforms, cpu_callback_platforms),
            requested_platform=requested,
            selector=selector,
        )


def test_wiring_shape_is_exactly_forty_interactions() -> None:
    args = runner.build_parser().parse_args(
        ["--expert", "forward", "--wiring-only"]
    )
    legacy = runner._load_legacy_trainer()
    shape = runner.resolve_training_shape(args, legacy)

    assert shape.num_timesteps == 40
    assert shape.num_envs == 2
    assert shape.interactions_per_training_step == 40
    assert shape.expected_training_steps == 1
    assert shape.expected_optimizer_updates == 2
    assert args.learning_rate is None
    assert args.entropy_cost == pytest.approx(1.0e-3)
    assert args.observation_mode == "legacy101"


def test_wiring_shape_rejects_any_non_forty_override() -> None:
    args = runner.build_parser().parse_args(
        [
            "--expert",
            "reverse",
            "--wiring-only",
            "--num-timesteps",
            "41",
        ]
    )
    with pytest.raises(ValueError, match="exactly 40"):
        runner.resolve_training_shape(args, runner._load_legacy_trainer())


def test_non_wiring_training_is_fail_closed_without_authorization() -> None:
    args = runner.build_parser().parse_args(["--expert", "forward"])
    with pytest.raises(ValueError, match="blocked"):
        runner.resolve_training_shape(args, runner._load_legacy_trainer())


def test_candidate_and_promoted_training_shape_contract() -> None:
    legacy = runner._load_legacy_trainer()
    candidate = runner.build_parser().parse_args(
        ["--expert", "forward", "--authorize-simulation-training"]
    )
    shape = runner.resolve_training_shape(candidate, legacy)
    assert shape.num_timesteps == 250_000
    assert shape.interactions_per_training_step == 50_000
    assert shape.expected_training_steps == 5

    promoted = runner.build_parser().parse_args(
        [
            "--expert",
            "forward",
            "--authorize-simulation-training",
            "--num-timesteps",
            "1000000",
        ]
    )
    with pytest.raises(RuntimeError, match="H4_STRICT_PROMOTION_PRODUCER_NOT_READY"):
        runner.resolve_training_shape(promoted, legacy)


def test_anchor_configs_separate_physical_and_policy_commands() -> None:
    forward = runner.resolve_anchor_config("forward")
    reverse = runner.resolve_anchor_config("reverse")

    assert forward["physical_primary"] == (0.05, 0.0, 0.0)
    assert forward["policy_observation_anchor"] == (0.10, -0.018, -0.170)
    assert reverse["physical_primary"] == (-0.05, 0.0, 0.0)
    assert reverse["policy_observation_anchor"] == (-0.05, 0.0, 0.0)
    assert forward["exact_primary_probability"] >= 0.50
    assert reverse["exact_primary_probability"] >= 0.50
    assert -0.10 not in reverse["local_vx_m_s"]


@pytest.mark.parametrize(
    ("expert", "override"),
    [("forward", (0.06, 0.0, 0.0)), ("reverse", (-0.10, 0.0, 0.0))],
)
def test_physical_anchor_override_cannot_weaken_h4_endpoint(
    expert: str, override: tuple[float, float, float]
) -> None:
    with pytest.raises(ValueError, match="physical vx anchor"):
        runner.resolve_anchor_config(
            expert, physical_anchor_override=override
        )


def test_selected_reverse_teacher_is_pinned_and_reduces_startup_jump() -> None:
    selected = runner.load_selected_reverse_teacher(
        runner.DEFAULT_SELECTED_REVERSE_TEACHER
    )
    audit = runner.reverse_teacher_startup_audit(selected)

    assert selected["sha256"] == runner.PINNED_SELECTED_REVERSE_TEACHER_SHA256
    assert selected["table"].shape == (54, 14)
    assert selected["cadence_hz"] == pytest.approx(1.5)
    assert selected["phase_advance_bins"] == pytest.approx(1.62)
    assert selected["entry_phase_bins"] == pytest.approx(14.0)
    assert selected["first_phase_bins"] == pytest.approx(15.62)
    assert audit["selected_teacher_upstream_table_jump_rad"] == pytest.approx(
        0.29785791493541736
    )
    assert audit["training_visible_precomposed_first_jump_rad"] == pytest.approx(0.04)
    assert audit["final_guard_maximum_applied_delta_rad"] == pytest.approx(0.04)


def test_periodic_teacher_interpolation_wraps_exactly() -> None:
    table = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    np.testing.assert_allclose(
        runner.interpolate_periodic_table(table, 2.5), [2.0, 3.0]
    )
    np.testing.assert_allclose(
        runner.interpolate_periodic_table(table, -0.5), [2.0, 3.0]
    )


def test_116_observations_require_explicit_transplant_flag() -> None:
    args = runner.build_parser().parse_args(
        ["--expert", "forward", "--wiring-only", "--observation-mode", "h4_116_transplant"]
    )
    args.learning_rate = 5.0e-5
    with pytest.raises(ValueError, match="require"):
        runner._validate_scalar_configuration(args)


def test_reward_scales_are_cli_resolved_and_sign_checked() -> None:
    args = runner.build_parser().parse_args(
        [
            "--expert",
            "forward",
            "--wiring-only",
            "--reward-left-force-slip",
            "-0.7",
            "--reward-alternation",
            "5.0",
        ]
    )
    scales = runner.resolve_reward_scales(args).as_reward_scale_dict()
    assert scales["h4_left_force_slip"] == pytest.approx(-0.7)
    assert scales["h4_alternation"] == pytest.approx(5.0)

    args.reward_right_force_slip = 0.1
    with pytest.raises(ValueError):
        runner.resolve_reward_scales(args).as_reward_scale_dict()


def test_authorized_reverse_requires_exact_composition_residual_scale() -> None:
    args = runner.build_parser().parse_args(
        [
            "--expert",
            "reverse",
            "--authorize-simulation-training",
            "--observation-mode",
            "h4_116_transplant",
            "--allow-verified-v22-transplant",
            "--backward-residual-scale",
            "0.11",
        ]
    )
    args.learning_rate = 3.0e-5
    with pytest.raises(ValueError, match="exact residual scale 0.12"):
        runner._validate_scalar_configuration(args)


@pytest.mark.parametrize("expert", ("forward", "reverse"))
@pytest.mark.parametrize(
    ("knob", "offset"),
    (
        ("learning_rate", 1.0e-9),
        ("entropy_cost", 1.0e-9),
        ("clipping_epsilon", 1.0e-9),
        ("discounting", 1.0e-9),
        ("max_grad_norm", 1.0e-9),
    ),
)
def test_authorized_optimizer_contract_rejects_near_miss(
    expert: str, knob: str, offset: float
) -> None:
    args = runner.build_parser().parse_args(
        [
            "--expert",
            expert,
            "--authorize-simulation-training",
            "--observation-mode",
            "h4_116_transplant",
            "--allow-verified-v22-transplant",
        ]
    )
    args.learning_rate = 5.0e-5 if expert == "forward" else 3.0e-5
    setattr(args, knob, float(getattr(args, knob)) + offset)
    with pytest.raises(ValueError, match="optimizer contract drifted"):
        runner._validate_scalar_configuration(args)


def test_unique_output_claim_refuses_overwrite(tmp_path: Path) -> None:
    claimed = runner.claim_unique_run_directory(
        tmp_path, "forward", "one_immutable_run"
    )
    assert claimed.is_dir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.claim_unique_run_directory(
            tmp_path, "forward", "one_immutable_run"
        )


def test_promotion_evidence_requires_three_strict_six_second_seeds(
    tmp_path: Path,
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(
        '{"hardware_deployment":"PROHIBITED","promotion_gate":'
        '{"strict_seed_count":2,"evaluation_duration_s":6,'
        '"strict_improvement_demonstrated":true}}',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="H4_STRICT_PROMOTION_PRODUCER_NOT_READY"):
        runner.validate_promotion_evidence(bad)


def test_minimum_specs_and_reverse_composition_are_hash_pinned() -> None:
    forward = runner.load_forward_minimum_spec()
    reverse = runner.load_reverse_minimum_spec()
    authorization = runner.load_reverse_composition_authorization()
    assert forward["declared_actor_width"] == 116
    assert forward["declared_extra_rows"] == 15
    assert not forward["stale_width_declaration_detected"]
    assert reverse["sha256"] == runner.PINNED_REVERSE_MINIMUM_SPEC_SHA256
    assert authorization["sha256"] == (
        runner.PINNED_REVERSE_COMPOSITION_AUTHORIZATION_SHA256
    )
    assert authorization["semantic_audit"]["valid"] is True


@pytest.mark.parametrize(
    ("expert", "flag", "seed", "learning_rate", "contract_id"),
    (
        (
            "forward",
            "--forward-iteration-v2",
            20260809,
            5.0e-5,
            "H4_FORWARD_ITERATION_V2_250K_FROM_V22",
        ),
        (
            "reverse",
            "--reverse-iteration-v2",
            20260810,
            3.0e-5,
            "H4_REVERSE_ITERATION_V2_250K_FROM_V22",
        ),
    ),
)
def test_iteration_v2_contracts_are_explicit_hash_bound_and_authorized(
    expert: str,
    flag: str,
    seed: int,
    learning_rate: float,
    contract_id: str,
) -> None:
    args = runner.build_parser().parse_args(
        [
            "--expert",
            expert,
            "--authorize-simulation-training",
            "--observation-mode",
            "h4_116_transplant",
            "--allow-verified-v22-transplant",
            "--seed",
            str(seed),
            flag,
        ]
    )
    args.learning_rate = learning_rate
    runner._validate_scalar_configuration(args)
    scales = runner.resolve_reward_scales(args).as_reward_scale_dict()
    anchors = runner.resolve_anchor_config(
        expert,
        forward_iteration_v2=args.forward_iteration_v2,
        reverse_iteration_v2=args.reverse_iteration_v2,
    )
    forward_spec = runner.load_forward_minimum_spec() if expert == "forward" else None
    reverse_spec = runner.load_reverse_minimum_spec() if expert == "reverse" else None
    reverse_authorization = (
        runner.load_reverse_composition_authorization()
        if expert == "reverse"
        else None
    )
    forward_v2 = (
        runner.load_forward_iteration_v2_authorization()
        if expert == "forward"
        else None
    )
    reverse_v2 = (
        runner.load_reverse_iteration_v2_authorization()
        if expert == "reverse"
        else None
    )
    selected = forward_v2 or reverse_v2
    assert selected is not None
    assert selected["contract_id"] == contract_id
    assert all(selected["semantic_audit"].values())
    assert set(selected["bound_causal_inputs"]) == {
        "failed_candidate_params",
        "failed_candidate_manifest",
        "integrated_strict_evaluation",
    }
    runner._validate_authorized_training_contract(
        args,
        shape=SimpleNamespace(num_envs=1250, num_timesteps=250_000),
        reward_scale_dict=scales,
        anchors=anchors,
        forward_spec=forward_spec,
        forward_iteration_v2_authorization=forward_v2,
        reverse_spec=reverse_spec,
        reverse_authorization=reverse_authorization,
        reverse_iteration_v2_authorization=reverse_v2,
    )


@pytest.mark.parametrize(
    ("expert", "flag", "seed", "learning_rate", "wiring_contract_id"),
    (
        (
            "forward",
            "--forward-iteration-v2",
            20260809,
            5.0e-5,
            "H4_FORWARD_ITERATION_V2_WIRING_PREFLIGHT_40_FROM_V22",
        ),
        (
            "reverse",
            "--reverse-iteration-v2",
            20260810,
            3.0e-5,
            "H4_REVERSE_ITERATION_V2_WIRING_PREFLIGHT_40_FROM_V22",
        ),
    ),
)
def test_iteration_v2_wiring_preflight_is_exact_and_separate_from_250k(
    expert: str,
    flag: str,
    seed: int,
    learning_rate: float,
    wiring_contract_id: str,
) -> None:
    args = runner.build_parser().parse_args(
        [
            "--expert",
            expert,
            "--wiring-only",
            "--num-timesteps",
            "40",
            "--observation-mode",
            "h4_116_transplant",
            "--allow-verified-v22-transplant",
            "--seed",
            str(seed),
            flag,
        ]
    )
    args.learning_rate = learning_rate
    runner._validate_scalar_configuration(args)
    shape = runner.resolve_training_shape(args, runner._load_legacy_trainer())
    assert (
        shape.num_timesteps,
        shape.num_envs,
        shape.expected_training_steps,
        shape.expected_optimizer_updates,
    ) == (40, 2, 1, 2)

    scales = runner.resolve_reward_scales(args).as_reward_scale_dict()
    anchors = runner.resolve_anchor_config(
        expert,
        forward_iteration_v2=args.forward_iteration_v2,
        reverse_iteration_v2=args.reverse_iteration_v2,
    )
    forward_spec = runner.load_forward_minimum_spec() if expert == "forward" else None
    reverse_spec = runner.load_reverse_minimum_spec() if expert == "reverse" else None
    reverse_authorization = (
        runner.load_reverse_composition_authorization()
        if expert == "reverse"
        else None
    )
    forward_v2 = (
        runner.load_forward_iteration_v2_authorization()
        if expert == "forward"
        else None
    )
    reverse_v2 = (
        runner.load_reverse_iteration_v2_authorization()
        if expert == "reverse"
        else None
    )
    runner._validate_authorized_training_contract(
        args,
        shape=shape,
        reward_scale_dict=scales,
        anchors=anchors,
        forward_spec=forward_spec,
        forward_iteration_v2_authorization=forward_v2,
        reverse_spec=reverse_spec,
        reverse_authorization=reverse_authorization,
        reverse_iteration_v2_authorization=reverse_v2,
    )
    assert runner.resolve_execution_contract_id(
        args,
        forward_iteration_v2_authorization=forward_v2,
        reverse_iteration_v2_authorization=reverse_v2,
    ) == wiring_contract_id
    selected = forward_v2 or reverse_v2
    assert selected is not None
    assert selected["contract_id"].endswith("_250K_FROM_V22")


def test_iteration_v2_wiring_shape_and_authorization_bindings_fail_closed() -> None:
    args = runner.build_parser().parse_args(
        [
            "--expert",
            "forward",
            "--wiring-only",
            "--observation-mode",
            "h4_116_transplant",
            "--allow-verified-v22-transplant",
            "--seed",
            "20260809",
            "--forward-iteration-v2",
        ]
    )
    args.learning_rate = 5.0e-5
    scales = runner.resolve_reward_scales(args).as_reward_scale_dict()
    anchors = runner.resolve_anchor_config("forward", forward_iteration_v2=True)
    spec = runner.load_forward_minimum_spec()
    authorization = runner.load_forward_iteration_v2_authorization()
    bad_shape = SimpleNamespace(
        num_timesteps=40,
        num_envs=2,
        interactions_per_training_step=40,
        expected_training_steps=1,
        expected_optimizer_updates=3,
    )
    with pytest.raises(ValueError, match="wiring shape drifted"):
        runner._validate_authorized_training_contract(
            args,
            shape=bad_shape,
            reward_scale_dict=scales,
            anchors=anchors,
            forward_spec=spec,
            forward_iteration_v2_authorization=authorization,
            reverse_spec=None,
            reverse_authorization=None,
            reverse_iteration_v2_authorization=None,
        )
    good_shape = SimpleNamespace(
        num_timesteps=40,
        num_envs=2,
        interactions_per_training_step=40,
        expected_training_steps=1,
        expected_optimizer_updates=2,
    )
    with pytest.raises(ValueError, match="requires its pinned authorization"):
        runner._validate_authorized_training_contract(
            args,
            shape=good_shape,
            reward_scale_dict=scales,
            anchors=anchors,
            forward_spec=spec,
            forward_iteration_v2_authorization=None,
            reverse_spec=None,
            reverse_authorization=None,
            reverse_iteration_v2_authorization=None,
        )


def test_forward_iteration_v2_exact_curriculum_and_bounded_scales() -> None:
    anchors = runner.resolve_anchor_config("forward", forward_iteration_v2=True)
    scales = runner.forward_iteration_v2_reward_scales().as_reward_scale_dict()

    assert (
        anchors["stand_probability"],
        anchors["exact_primary_probability"],
        anchors["local_probability"],
        anchors["transition_probability"],
    ) == (0.05, 0.70, 0.20, 0.05)
    assert scales["h4_force_slip"] == -3.0
    assert scales["h4_total_normal_force_band"] == -1.0
    assert scales["h4_total_normal_force_tail"] == -1.0
    assert scales["h4_contact_pulse_40ms"] == -1.0
    assert scales["h4_forward_heading_drift"] == -2.0


def test_reverse_iteration_v2_exact_curriculum_rewards_and_legacy_config() -> None:
    anchors = runner.resolve_anchor_config("reverse", reverse_iteration_v2=True)
    scales = runner.reverse_iteration_v2_reward_scales().as_reward_scale_dict()

    assert (
        anchors["stand_probability"],
        anchors["exact_primary_probability"],
        anchors["local_probability"],
        anchors["transition_probability"],
    ) == (0.05, 0.75, 0.15, 0.05)
    assert scales["h4_reverse_speed_boundary"] == -8.0
    assert scales["h4_single_support"] == 4.0
    assert scales["h4_single_support_band"] == -4.0
    assert scales["h4_alternation"] == 6.0
    assert scales["h4_per_foot_stance_slip_budget"] == -2.0
    assert scales["h4_total_normal_force_band"] == 0.0
    assert scales["h4_total_normal_force_tail"] == 0.0
    assert scales["h4_contact_pulse_40ms"] == 0.0
    assert dict(runner.REVERSE_ITERATION_V2_LEGACY_REWARD_CONFIG) == {
        "target_imitation": -20.0,
        "contact_imitation": 15.0,
        "tracking_sigma": 0.01,
    }
    authorization = runner.load_reverse_iteration_v2_authorization()
    integrated = authorization["payload"]["causal_input"][
        "integrated_strict_evaluation"
    ]
    assert integrated["legacy_schema3_composition_trace_complete"] is False
    assert integrated["safety_trace_used_for_qualification"] is False
    assert integrated["causal_basis"] == (
        "GAIT_QUALITY_0_OF_3_AND_STEADY_REVERSE_SPEED_ONLY"
    )


@pytest.mark.parametrize(
    ("argv", "message"),
    (
        (
            (
                "--expert",
                "forward",
                "--wiring-only",
                "--seed",
                "20260809",
                "--forward-iteration-v2",
            ),
            "actor116",
        ),
        (
            (
                "--expert",
                "forward",
                "--wiring-only",
                "--num-timesteps",
                "41",
                "--observation-mode",
                "h4_116_transplant",
                "--allow-verified-v22-transplant",
                "--seed",
                "20260809",
                "--forward-iteration-v2",
            ),
            "exactly 40",
        ),
        (
            (
                "--expert",
                "reverse",
                "--wiring-only",
                "--observation-mode",
                "h4_116_transplant",
                "--seed",
                "20260810",
                "--reverse-iteration-v2",
            ),
            "require",
        ),
    ),
)
def test_iteration_v2_wiring_flag_and_shape_near_misses_fail_closed(
    argv: tuple[str, ...], message: str
) -> None:
    args = runner.build_parser().parse_args(argv)
    args.learning_rate = 5.0e-5 if args.expert == "forward" else 3.0e-5
    with pytest.raises(ValueError, match=message):
        runner._validate_scalar_configuration(args)


@pytest.mark.parametrize(
    ("validator", "loader", "mutation"),
    (
        (
            runner.validate_forward_iteration_v2_authorization_payload,
            runner.load_forward_iteration_v2_authorization,
            ("curriculum", "exact_primary_probability", 0.69),
        ),
        (
            runner.validate_reverse_iteration_v2_authorization_payload,
            runner.load_reverse_iteration_v2_authorization,
            ("legacy_reward_config", "iteration_v2_exact", {"tracking_sigma": 0.02}),
        ),
    ),
)
def test_iteration_v2_authorization_semantic_mutations_fail_closed(
    validator: object,
    loader: object,
    mutation: tuple[str, str, object],
) -> None:
    loaded = loader()
    broken = copy.deepcopy(loaded["payload"])
    first, second, value = mutation
    broken[first][second] = value
    with pytest.raises(ValueError, match="authorization drifted"):
        validator(broken)


@pytest.mark.parametrize(
    ("expert", "flag", "seed", "learning_rate", "extra", "message"),
    (
        ("forward", "--forward-iteration-v2", 20260810, 5.0e-5, (), "seed 20260809"),
        ("reverse", "--reverse-iteration-v2", 20260809, 3.0e-5, (), "seed 20260810"),
        ("reverse", "--forward-iteration-v2", 20260809, 3.0e-5, (), "only for forward"),
        ("forward", "--reverse-iteration-v2", 20260810, 5.0e-5, (), "only for reverse"),
        (
            "forward",
            "--forward-iteration-v2",
            20260809,
            5.0e-5,
            ("--wiring-only",),
            "must not use",
        ),
        (
            "reverse",
            "--reverse-iteration-v2",
            20260810,
            3.0e-5,
            ("--num-timesteps", "1000000"),
            "separate 250k",
        ),
    ),
)
def test_iteration_v2_cli_near_misses_fail_closed(
    expert: str,
    flag: str,
    seed: int,
    learning_rate: float,
    extra: tuple[str, ...],
    message: str,
) -> None:
    argv = [
        "--expert",
        expert,
        "--authorize-simulation-training",
        "--observation-mode",
        "h4_116_transplant",
        "--allow-verified-v22-transplant",
        "--seed",
        str(seed),
        flag,
        *extra,
    ]
    args = runner.build_parser().parse_args(argv)
    args.learning_rate = learning_rate
    with pytest.raises(ValueError, match=message):
        runner._validate_scalar_configuration(args)


@pytest.mark.parametrize(
    ("expert", "flag", "seed", "learning_rate", "override"),
    (
        (
            "forward",
            "--forward-iteration-v2",
            20260809,
            5.0e-5,
            ("--reward-force-slip", "-2.9"),
        ),
        (
            "reverse",
            "--reverse-iteration-v2",
            20260810,
            3.0e-5,
            ("--reward-single-support", "4"),
        ),
    ),
)
def test_iteration_v2_reward_cli_overrides_are_forbidden(
    expert: str,
    flag: str,
    seed: int,
    learning_rate: float,
    override: tuple[str, str],
) -> None:
    args = runner.build_parser().parse_args(
        [
            "--expert",
            expert,
            "--authorize-simulation-training",
            "--observation-mode",
            "h4_116_transplant",
            "--allow-verified-v22-transplant",
            "--seed",
            str(seed),
            flag,
            *override,
        ]
    )
    args.learning_rate = learning_rate
    with pytest.raises(ValueError, match="authorization-controlled"):
        runner.resolve_reward_scales(args)


def _iteration_v3_args(
    expert: str,
    *,
    wiring_only: bool = False,
) -> object:
    flag = (
        "--forward-iteration-v3-touchdown-balance"
        if expert == "forward"
        else "--reverse-iteration-v3-no-target-imitation"
    )
    argv = [
        "--expert",
        expert,
        "--observation-mode",
        "h4_116_transplant",
        "--allow-verified-v22-transplant",
        "--seed",
        "20260809" if expert == "forward" else "20260810",
        flag,
    ]
    if wiring_only:
        argv.extend(("--wiring-only", "--num-timesteps", "40"))
    else:
        argv.append("--authorize-simulation-training")
    args = runner.build_parser().parse_args(argv)
    args.learning_rate = 5.0e-5 if expert == "forward" else 3.0e-5
    return args


def test_forward_iteration_v3_authorization_is_hash_bound_and_one_scale_only() -> None:
    authorization = runner.load_forward_iteration_v3_touchdown_balance_authorization()
    assert authorization["sha256"] == (
        runner.PINNED_FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_AUTHORIZATION_SHA256
    )
    assert all(authorization["semantic_audit"].values())
    assert set(authorization["bound_causal_inputs"]) == {
        "failed_candidate_params",
        "failed_candidate_manifest",
        "integrated_strict_evaluation",
    }
    v2 = runner.forward_iteration_v2_reward_scales().as_reward_scale_dict()
    v3 = (
        runner.forward_iteration_v3_touchdown_balance_reward_scales()
        .as_reward_scale_dict()
    )
    assert {name for name in v2 if v2[name] != v3[name]} == {
        "h4_touchdown_count_balance"
    }
    assert v2["h4_touchdown_count_balance"] == -2.0
    assert v3["h4_touchdown_count_balance"] == -4.0


def test_reverse_iteration_v3_authorization_is_hash_bound_and_one_scale_only() -> None:
    authorization = runner.load_reverse_iteration_v3_no_target_imitation_authorization()
    assert authorization["sha256"] == (
        runner.PINNED_REVERSE_ITERATION_V3_NO_TARGET_IMITATION_AUTHORIZATION_SHA256
    )
    assert all(authorization["semantic_audit"].values())
    assert set(authorization["bound_causal_inputs"]) == {
        "failed_candidate_params",
        "failed_candidate_manifest",
        "integrated_strict_evaluation",
    }
    v2 = dict(runner.REVERSE_ITERATION_V2_LEGACY_REWARD_CONFIG)
    v3 = dict(runner.REVERSE_ITERATION_V3_NO_TARGET_IMITATION_LEGACY_REWARD_CONFIG)
    assert {name for name in v2 if v2[name] != v3[name]} == {
        "target_imitation"
    }
    assert v2["target_imitation"] == -20.0
    assert v3 == {
        "target_imitation": 0.0,
        "contact_imitation": 15.0,
        "tracking_sigma": 0.01,
    }
    assert (
        runner.reverse_iteration_v2_reward_scales().as_reward_scale_dict()
        == authorization["payload"]["reward_contract"]["exact_scales"]
    )


@pytest.mark.parametrize(
    ("expert", "full_contract", "wiring_contract"),
    (
        (
            "forward",
            runner.FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_CONTRACT_ID,
            runner.FORWARD_ITERATION_V3_TOUCHDOWN_BALANCE_WIRING_CONTRACT_ID,
        ),
        (
            "reverse",
            runner.REVERSE_ITERATION_V3_NO_TARGET_IMITATION_CONTRACT_ID,
            runner.REVERSE_ITERATION_V3_NO_TARGET_IMITATION_WIRING_CONTRACT_ID,
        ),
    ),
)
@pytest.mark.parametrize("wiring_only", (False, True))
def test_iteration_v3_contracts_validate_exact_full_and_wiring_shapes(
    expert: str,
    full_contract: str,
    wiring_contract: str,
    wiring_only: bool,
) -> None:
    args = _iteration_v3_args(expert, wiring_only=wiring_only)
    runner._validate_scalar_configuration(args)
    shape = (
        runner.resolve_training_shape(args, runner._load_legacy_trainer())
        if wiring_only
        else SimpleNamespace(num_envs=1250, num_timesteps=250_000)
    )
    if wiring_only:
        assert (
            shape.num_timesteps,
            shape.num_envs,
            shape.interactions_per_training_step,
            shape.expected_training_steps,
            shape.expected_optimizer_updates,
        ) == (40, 2, 40, 1, 2)
    forward_authorization = (
        runner.load_forward_iteration_v3_touchdown_balance_authorization()
        if expert == "forward"
        else None
    )
    reverse_authorization_v3 = (
        runner.load_reverse_iteration_v3_no_target_imitation_authorization()
        if expert == "reverse"
        else None
    )
    runner._validate_authorized_training_contract(
        args,
        shape=shape,
        reward_scale_dict=runner.resolve_reward_scales(args).as_reward_scale_dict(),
        anchors=runner.resolve_anchor_config(
            expert,
            forward_iteration_v3_touchdown_balance=(expert == "forward"),
            reverse_iteration_v3_no_target_imitation=(expert == "reverse"),
        ),
        forward_spec=(
            runner.load_forward_minimum_spec() if expert == "forward" else None
        ),
        forward_iteration_v2_authorization=None,
        forward_iteration_v3_touchdown_balance_authorization=(
            forward_authorization
        ),
        reverse_spec=(
            runner.load_reverse_minimum_spec() if expert == "reverse" else None
        ),
        reverse_authorization=(
            runner.load_reverse_composition_authorization()
            if expert == "reverse"
            else None
        ),
        reverse_iteration_v2_authorization=None,
        reverse_iteration_v3_no_target_imitation_authorization=(
            reverse_authorization_v3
        ),
    )
    assert runner.resolve_execution_contract_id(
        args,
        forward_iteration_v2_authorization=None,
        forward_iteration_v3_touchdown_balance_authorization=(
            forward_authorization
        ),
        reverse_iteration_v2_authorization=None,
        reverse_iteration_v3_no_target_imitation_authorization=(
            reverse_authorization_v3
        ),
    ) == (wiring_contract if wiring_only else full_contract)


@pytest.mark.parametrize("expert", ("forward", "reverse"))
def test_iteration_v3_reset_noise_drift_fails_closed(expert: str) -> None:
    args = _iteration_v3_args(expert)
    args.reset_noise_multiplier = 0.0
    with pytest.raises(ValueError, match="reset-noise multiplier 1.0"):
        runner._validate_scalar_configuration(args)


def test_all_iteration_modes_are_mutually_exclusive() -> None:
    args = _iteration_v3_args("forward")
    args.reverse_iteration_v2 = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        runner._validate_scalar_configuration(args)


@pytest.mark.parametrize(
    ("loader", "validator", "section", "key", "value"),
    (
        (
            runner.load_forward_iteration_v3_touchdown_balance_authorization,
            runner.validate_forward_iteration_v3_touchdown_balance_authorization_payload,
            "training_contract",
            "reset_noise_multiplier",
            0.0,
        ),
        (
            runner.load_forward_iteration_v3_touchdown_balance_authorization,
            runner.validate_forward_iteration_v3_touchdown_balance_authorization_payload,
            "reward_contract",
            "exact_scales",
            {},
        ),
        (
            runner.load_reverse_iteration_v3_no_target_imitation_authorization,
            runner.validate_reverse_iteration_v3_no_target_imitation_authorization_payload,
            "training_contract",
            "reset_noise_multiplier",
            2.0,
        ),
        (
            runner.load_reverse_iteration_v3_no_target_imitation_authorization,
            runner.validate_reverse_iteration_v3_no_target_imitation_authorization_payload,
            "legacy_reward_config",
            "iteration_v3_exact",
            {"target_imitation": 1.0},
        ),
    ),
)
def test_iteration_v3_authorization_mutations_fail_closed(
    loader: object,
    validator: object,
    section: str,
    key: str,
    value: object,
) -> None:
    broken = copy.deepcopy(loader()["payload"])
    broken[section][key] = value
    with pytest.raises(ValueError, match="authorization drifted"):
        validator(broken)


def _iteration_v4_args(
    expert: str,
    *,
    wiring_only: bool = False,
) -> object:
    flag = (
        "--forward-iteration-v4-contact-event-validity-persistence"
        if expert == "forward"
        else "--reverse-iteration-v4-residual-transfer-gain-024"
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
        argv.extend(("--backward-residual-scale", "0.24"))
    if wiring_only:
        argv.extend(("--wiring-only", "--num-timesteps", "40"))
    else:
        argv.append("--authorize-simulation-training")
    args = runner.build_parser().parse_args(argv)
    args.learning_rate = 5.0e-5 if expert == "forward" else 3.0e-5
    return args


@pytest.mark.parametrize("expert", ("forward", "reverse"))
@pytest.mark.parametrize("wiring_only", (False, True))
def test_iteration_v4_cli_reward_anchor_and_shape_contracts_are_exact(
    expert: str, wiring_only: bool
) -> None:
    args = _iteration_v4_args(expert, wiring_only=wiring_only)
    runner._validate_scalar_configuration(args)
    scales = runner.resolve_reward_scales(args).as_reward_scale_dict()
    expected_scales = (
        runner.forward_iteration_v2_reward_scales().as_reward_scale_dict()
        if expert == "forward"
        else runner.reverse_iteration_v2_reward_scales().as_reward_scale_dict()
    )
    assert scales == expected_scales
    anchors = runner.resolve_anchor_config(
        expert,
        forward_iteration_v4_contact_event_validity_persistence=(
            expert == "forward"
        ),
        reverse_iteration_v4_residual_transfer_gain_024=(expert == "reverse"),
    )
    expected_anchors = (
        runner.FORWARD_ITERATION_V2_ANCHOR_CONFIG
        if expert == "forward"
        else runner.REVERSE_ITERATION_V2_ANCHOR_CONFIG
    )
    assert anchors == expected_anchors
    if wiring_only:
        shape = runner.resolve_training_shape(args, runner._load_legacy_trainer())
        assert (
            shape.num_timesteps,
            shape.num_envs,
            shape.interactions_per_training_step,
            shape.expected_training_steps,
            shape.expected_optimizer_updates,
        ) == (40, 2, 40, 1, 2)


@pytest.mark.parametrize(
    ("expert", "full_id", "wiring_id"),
    (
        (
            "forward",
            runner.FORWARD_ITERATION_V4_CONTACT_EVENT_VALIDITY_PERSISTENCE_CONTRACT_ID,
            runner.FORWARD_ITERATION_V4_CONTACT_EVENT_VALIDITY_PERSISTENCE_WIRING_CONTRACT_ID,
        ),
        (
            "reverse",
            runner.REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN_024_CONTRACT_ID,
            runner.REVERSE_ITERATION_V4_RESIDUAL_TRANSFER_GAIN_024_WIRING_CONTRACT_ID,
        ),
    ),
)
@pytest.mark.parametrize("wiring_only", (False, True))
def test_iteration_v4_execution_contract_ids_separate_wiring_from_full(
    expert: str, full_id: str, wiring_id: str, wiring_only: bool
) -> None:
    args = _iteration_v4_args(expert, wiring_only=wiring_only)
    forward_auth = {"contract_id": full_id} if expert == "forward" else None
    reverse_auth = {"contract_id": full_id} if expert == "reverse" else None
    actual = runner.resolve_execution_contract_id(
        args,
        forward_iteration_v2_authorization=None,
        forward_iteration_v3_touchdown_balance_authorization=None,
        forward_iteration_v4_contact_event_validity_persistence_authorization=(
            forward_auth
        ),
        reverse_iteration_v2_authorization=None,
        reverse_iteration_v3_no_target_imitation_authorization=None,
        reverse_iteration_v4_residual_transfer_gain_024_authorization=(
            reverse_auth
        ),
    )
    assert actual == (wiring_id if wiring_only else full_id)


@pytest.mark.parametrize("expert", ("forward", "reverse"))
def test_iteration_v4_full_contract_requires_exact_1250_envs(expert: str) -> None:
    args = _iteration_v4_args(expert)
    forward_auth = {} if expert == "forward" else None
    reverse_auth = {} if expert == "reverse" else None
    with pytest.raises(ValueError, match="exactly 1250"):
        runner._validate_authorized_training_contract(
            args,
            shape=SimpleNamespace(num_envs=1249, num_timesteps=250_000),
            reward_scale_dict=runner.resolve_reward_scales(
                args
            ).as_reward_scale_dict(),
            anchors=runner.resolve_anchor_config(
                expert,
                forward_iteration_v4_contact_event_validity_persistence=(
                    expert == "forward"
                ),
                reverse_iteration_v4_residual_transfer_gain_024=(
                    expert == "reverse"
                ),
            ),
            forward_spec=(
                runner.load_forward_minimum_spec()
                if expert == "forward"
                else None
            ),
            forward_iteration_v2_authorization=None,
            forward_iteration_v3_touchdown_balance_authorization=None,
            forward_iteration_v4_contact_event_validity_persistence_authorization=(
                forward_auth
            ),
            reverse_spec=(
                runner.load_reverse_minimum_spec()
                if expert == "reverse"
                else None
            ),
            reverse_authorization=(
                runner.load_reverse_composition_authorization()
                if expert == "reverse"
                else None
            ),
            reverse_iteration_v2_authorization=None,
            reverse_iteration_v3_no_target_imitation_authorization=None,
            reverse_iteration_v4_residual_transfer_gain_024_authorization=(
                reverse_auth
            ),
        )


@pytest.mark.parametrize(
    ("expert", "mutation", "message"),
    (
        ("forward", {"reverse_iteration_v4_residual_transfer_gain_024": True}, "mutually exclusive"),
        ("forward", {"seed": 20260810}, "seed 20260809"),
        ("forward", {"reset_noise_multiplier": 0.0}, "reset-noise multiplier 1.0"),
        ("reverse", {"backward_residual_scale": 0.12}, "residual scale 0.24"),
        ("reverse", {"backward_residual_scale": 0.25}, "residual scale 0.24"),
        ("reverse", {"seed": 20260809}, "seed 20260810"),
    ),
)
def test_iteration_v4_cli_near_misses_fail_closed(
    expert: str, mutation: dict[str, object], message: str
) -> None:
    args = _iteration_v4_args(expert)
    for key, value in mutation.items():
        setattr(args, key, value)
    with pytest.raises(ValueError, match=message):
        runner._validate_scalar_configuration(args)


def _v4_payload_with_current_source_closure(expert: str) -> dict[str, object]:
    spec = runner._iteration_v4_spec(expert)
    payload = runner._load_json_strict(spec["auth_path"])
    relative_paths = {
        "h4_training_alignment": "safe_gait_experts/h4_training_alignment.py",
        "h4_runner": "scripts/train_h4_aligned_expert.py",
        "h4_post_training": "safe_gait_experts/h4_post_training.py",
        "h4_candidate_evaluator": "scripts/evaluate_h4_training_candidate.py",
        "h4_no_ppo_smoke": "scripts/smoke_h4_training_alignment.py",
    }
    payload["causal_source_closure"] = {
        label: {
            "path": relative,
            "sha256": runner.sha256_file(runner.EXP_ROOT / relative),
        }
        for label, relative in relative_paths.items()
    }
    return payload


@pytest.mark.parametrize("expert", ("forward", "reverse"))
def test_iteration_v4_authorization_semantics_and_source_schema_validate(
    expert: str,
) -> None:
    checks = runner.validate_iteration_v4_authorization_payload(
        _v4_payload_with_current_source_closure(expert), expert=expert
    )
    assert all(checks.values())


@pytest.mark.parametrize(
    ("expert", "authorization_sha256", "causal_input_sha256"),
    (
        (
            "forward",
            "a808e329af37387466f9229dd587abf5fd90bcea08f1133295bd8551c3115a1e",
            (
                "93daa0c35f08929c17c6eef799565d327ce362c1c1ebdeaf9aa22ca6cc5d153f",
                "8946249b3531957166dc13005df7b2f25e50feefe03d78e9657e4724973e5dfa",
                "4dfef12700363ae9274e1e8d9371a3780bf4871a1fe2a03d1e806749cc7deb92",
                "3375ad29f0443ac95637c1970b73f355a8ae2ee856903a0a43f79b8c7d74fd0f",
            ),
        ),
        (
            "reverse",
            "93e3a53d5b601987df7a4efb84de5fb0ae499dc0ea0dc93acdbb074d96510312",
            (
                "b27d3e12f5619bf008b5034f33e561a8ab8d06c3880a914f1a28781c0a3bb5c7",
                "59871b9c35ea34ed3f62b8157d5afe8e2c8277cdc97e763c4a70dfafd8720414",
                "a80801d81118ed557b8b32426307543cd0d298dbc9d57837a6517d8e4b66c67c",
                "a52054327ec6c65326f4a869260cc4dd55b3935fe7375cededd3551f8b56ece2",
            ),
        ),
    ),
)
def test_iteration_v4_raw_authorizations_close_historical_inputs_and_sources(
    expert: str,
    authorization_sha256: str,
    causal_input_sha256: tuple[str, ...],
) -> None:
    spec = runner._iteration_v4_spec(expert)
    payload = runner._load_json_strict(spec["auth_path"])
    assert runner.sha256_file(spec["auth_path"]) == authorization_sha256
    assert all(
        runner.validate_iteration_v4_authorization_payload(
            payload, expert=expert
        ).values()
    )
    causal_input = payload["causal_input"]
    assert (
        causal_input["previous_iteration_authorization"]["sha256"],
        causal_input["failed_candidate_final_params_sha256"],
        causal_input["failed_candidate_manifest_sha256"],
        causal_input["integrated_strict_evaluation"]["sha256"],
    ) == causal_input_sha256
    assert {
        label: item["sha256"]
        for label, item in payload["causal_source_closure"].items()
    } == {
        "h4_training_alignment": "872a11a817bb068e3a0819c0afca12ae9e7f2dbfcc103c6569b9081b8d5fbebb",
        "h4_runner": "b15b9692a72deadd34790d442f4ab4263c3f987255173566a62438e0d380da13",
        "h4_post_training": "afdfcf9da43a7a7e5824ce7562c489b5e5e20a32e83af817be9e80d740a27b3f",
        "h4_candidate_evaluator": "c214d086e6d66f6f9f98c7268481899e4133961dcc5355d738d4cd134a82e6ae",
        "h4_no_ppo_smoke": "410924542bac85f70de3a4055f617a85e93eb841cd403f5280699778ac96710d",
    }


@pytest.mark.parametrize(
    ("expert", "section", "key", "value"),
    (
        ("forward", "core_contract", "exact_value", False),
        ("forward", "reward_contract", "exact_scales", {}),
        ("reverse", "teacher_and_guard_contract", "backward_residual_scale_iteration_v4", 0.12),
        ("reverse", "causal_input", "hypothesis", {}),
    ),
)
def test_iteration_v4_authorization_near_misses_fail_closed(
    expert: str, section: str, key: str, value: object
) -> None:
    payload = _v4_payload_with_current_source_closure(expert)
    payload[section][key] = value
    with pytest.raises(ValueError, match="iteration-v4 authorization drifted"):
        runner.validate_iteration_v4_authorization_payload(payload, expert=expert)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        (
            "official_source_provenance",
            "joystick_sha256",
            "0" * 64,
        ),
        (
            "official_source_provenance",
            "mjx_env_relative_path",
            "mujoco_playground/_src/mjx_env.py",
        ),
        ("official_source_provenance", "step_source_sha256", "0" * 64),
        ("official_source_provenance", "step_source_semantics", "WRONG"),
        ("preflight_probe_contract", "seed", 20260810),
        ("preflight_probe_contract", "action_shape", [13]),
        ("preflight_probe_contract", "action_dtype", "float64"),
        ("preflight_probe_contract", "action_all_zero", False),
    ),
)
def test_forward_v4_authorization_source_theorem_near_misses_fail_closed(
    section: str, key: str, value: object
) -> None:
    payload = _v4_payload_with_current_source_closure("forward")
    payload["core_contract"]["source_semantic_theorem"][section][key] = value
    with pytest.raises(ValueError, match="core_opt_in_exact"):
        runner.validate_iteration_v4_authorization_payload(
            payload, expert="forward"
        )


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("dynamic_field_count_exact_metric_required", False),
        ("saved_dynamic_field_count_exact_metric_required", False),
        ("episode_field_count_exact_totals_equal_length", False),
        ("diagnostic_count_totals_qualification_role", "QUALIFYING"),
        ("host_count_multiplication_for_qualification", True),
        ("numeric_tolerance_used", True),
    ),
)
def test_forward_v4_authorization_aggregate_exactness_near_misses_fail_closed(
    key: str, value: object
) -> None:
    payload = _v4_payload_with_current_source_closure("forward")
    payload["core_contract"]["runtime_authority_assertion"][key] = value
    with pytest.raises(ValueError, match="core_opt_in_exact"):
        runner.validate_iteration_v4_authorization_payload(
            payload, expert="forward"
        )


def _forward_v4_single_authority_sample() -> dict[str, object]:
    return {
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


def test_forward_v4_single_authority_samples_require_exact_runtime_contract() -> None:
    samples = [
        _forward_v4_single_authority_sample(),
        {
            key: np.asarray(value)
            for key, value in _forward_v4_single_authority_sample().items()
        },
    ]
    assert runner.require_forward_v4_single_authority_samples(
        samples, label="test"
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
        "observed_step_count": 2,
        "passed": True,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("dynamic6_exact", False),
        ("dynamic6_exact", 1),
        ("dynamic6_max_abs_error", np.nextafter(0.0, 1.0)),
        ("dynamic6_max_abs_error", float("nan")),
        ("dynamic6_field_count", 5),
        ("dynamic6_field_count_exact", False),
        ("saved_dynamic6_substep_count", 9),
        ("saved_dynamic6_field_count", 5),
        ("saved_dynamic6_field_count_exact", False),
        ("saved_dynamic6_all_finite", False),
        ("telemetry_force_shape_valid", False),
        ("telemetry_force_all_finite", False),
        ("authority_violation", True),
        ("assertion_token", 1),
    ),
)
def test_forward_v4_single_authority_samples_reject_every_near_miss(
    field: str, value: object
) -> None:
    sample = _forward_v4_single_authority_sample()
    sample[field] = value
    with pytest.raises(
        RuntimeError, match="single-authority|must be finite|boolean scalar"
    ):
        runner.require_forward_v4_single_authority_samples(
            [sample], label="test"
        )


def _forward_v4_source_provenance() -> dict[str, object]:
    source_root = "/home/user/openduck_training_20260729"
    return {
        "source_root": source_root,
        "joystick": {
            "resolved_path": (
                f"{source_root}/playground/open_duck_mini_v2/joystick.py"
            ),
            "relative_path": "playground/open_duck_mini_v2/joystick.py",
            "sha256": runner.PINNED_FORWARD_V4_OFFICIAL_JOYSTICK_SHA256,
        },
        "mjx_env": {
            "resolved_path": (
                f"{source_root}/.venv/lib/python3.12/site-packages/"
                "mujoco_playground/_src/mjx_env.py"
            ),
            "relative_path": (
                ".venv/lib/python3.12/site-packages/"
                "mujoco_playground/_src/mjx_env.py"
            ),
            "sha256": runner.PINNED_FORWARD_V4_OFFICIAL_MJX_ENV_SHA256,
        },
        "step_source_sha256": (
            runner.PINNED_FORWARD_V4_OFFICIAL_STEP_SOURCE_SHA256
        ),
        "step_source_semantics": runner.FORWARD_V4_OFFICIAL_STEP_SOURCE_SEMANTICS,
        "all_files_under_requested_source_root": True,
        "passed": True,
    }


def _forward_v4_probe_input() -> dict[str, object]:
    return {
        "seed": 20260809,
        "reset_noise_multiplier": 1.0,
        "initial_state_source": "ENV_RESET_JAX_PRNGKEY_SEED",
        "action_shape": [14],
        "action_dtype": "float32",
        "action_all_zero": True,
    }


def test_forward_v4_source_semantic_preflight_qualifies_only_dynamic6() -> None:
    audit = runner.V4SourceSemanticPreflight(
        True,
        0.0,
        6,
        False,
        0.140460968,
        False,
        0.14032745,
    )
    result = runner.require_forward_v4_source_semantic_preflight(
        audit,
        source_provenance=_forward_v4_source_provenance(),
        probe_input=_forward_v4_probe_input(),
    )
    assert result["dynamic6_exact"] is True
    assert result["dynamic6_max_abs_error"] == 0.0
    assert result["dynamic6_field_count"] == 6
    assert result["derived_diagnostics"] == {
        "qualification_role": "NON_QUALIFYING_OBSERVED_DIAGNOSTICS_ONLY",
        "fields": {
            "cfrc_int": {"exact": False, "max_abs_error": 0.140460968},
            "cfrc_ext": {"exact": False, "max_abs_error": 0.14032745},
        },
        "all_finite": True,
        "exclusion_is_semantic_not_tolerance": True,
        "numeric_tolerance_used": False,
    }
    assert result["observed_reference_count"] == 1
    assert result["passed"] is True


@pytest.mark.parametrize(
    ("target", "field", "value"),
    (
        ("joystick", "relative_path", "playground/wrong.py"),
        ("joystick", "sha256", "0" * 64),
        ("mjx_env", "resolved_path", "/tmp/mjx_env.py"),
        ("provenance", "step_source_sha256", "0" * 64),
        ("provenance", "step_source_semantics", "WRONG"),
        ("probe", "seed", 20260810),
        ("probe", "action_shape", [13]),
        ("probe", "action_dtype", "float64"),
        ("probe", "action_all_zero", False),
    ),
)
def test_forward_v4_source_semantic_preflight_rejects_provenance_and_probe_drift(
    target: str, field: str, value: object
) -> None:
    provenance = _forward_v4_source_provenance()
    probe = _forward_v4_probe_input()
    if target == "probe":
        probe[field] = value
    elif target == "provenance":
        provenance[field] = value
    else:
        provenance[target][field] = value
    audit = runner.V4SourceSemanticPreflight(
        True, 0.0, 6, False, 0.14, False, 0.13
    )
    with pytest.raises(RuntimeError, match="provenance/probe"):
        runner.require_forward_v4_source_semantic_preflight(
            audit,
            source_provenance=provenance,
            probe_input=probe,
        )


@pytest.mark.parametrize(
    "audit",
    (
        runner.V4SourceSemanticPreflight(
            False, 0.0, 6, True, 0.0, True, 0.0
        ),
        runner.V4SourceSemanticPreflight(
            True, np.nextafter(0.0, 1.0), 6, True, 0.0, True, 0.0
        ),
        runner.V4SourceSemanticPreflight(
            True, 0.0, 5, True, 0.0, True, 0.0
        ),
        runner.V4SourceSemanticPreflight(
            True, 0.0, 6, False, float("nan"), True, 0.0
        ),
        runner.V4SourceSemanticPreflight(
            True, 0.0, 6, True, 1.0e-12, True, 0.0
        ),
    ),
)
def test_forward_v4_source_semantic_preflight_rejects_qualifying_or_diagnostic_drift(
    audit: object,
) -> None:
    with pytest.raises(
        RuntimeError, match="source-semantic|must be finite|inconsistent"
    ):
        runner.require_forward_v4_source_semantic_preflight(
            audit,
            source_provenance=_forward_v4_source_provenance(),
            probe_input=_forward_v4_probe_input(),
        )


def _forward_v4_single_authority_episode_row(length: float = 20.0) -> dict[str, float]:
    return {
        "episode/length": length,
        "episode/h4/v4_single_authority_dynamic6_exact": length,
        "episode/h4/v4_single_authority_dynamic6_max_abs_error": 0.0,
        "episode/h4/v4_single_authority_dynamic6_field_count": 6.0 * length,
        "episode/h4/v4_single_authority_dynamic6_field_count_exact": length,
        "episode/h4/v4_saved_dynamic6_substep_count": 10.0 * length,
        "episode/h4/v4_saved_dynamic6_field_count": 6.0 * length,
        "episode/h4/v4_saved_dynamic6_field_count_exact": length,
        "episode/h4/v4_saved_dynamic6_all_finite": length,
        "episode/h4/v4_telemetry_force_shape_valid": length,
        "episode/h4/v4_telemetry_force_all_finite": length,
        "episode/h4/v4_single_authority_violation": 0.0,
        "episode/h4/v4_single_authority_assertion_token": 0.0,
    }


def test_forward_v4_full_runtime_authority_requires_exact_episode_rows() -> None:
    assert runner.require_forward_v4_single_authority_runtime_progress(
        [_forward_v4_single_authority_episode_row()]
    ) == {
        "audit_mode": runner.FORWARD_V4_FULL_RUNTIME_AUDIT_MODE,
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
        "observed_episode_metric_rows": 1,
        "authority_violation_count": 0.0,
        "assertion_token_sum": 0.0,
        "passed": True,
    }


def test_forward_v4_captured_fractional_episode_row_uses_device_exact_totals() -> None:
    length = 37.43478260869565
    row = _forward_v4_single_authority_episode_row(length)
    # Captured from failed v1: PPO's aggregate is one ULP above an independent
    # Python ``6 * length``.  Counts are finite diagnostics; the exact
    # per-step predicates aggregate identically to episode/length.
    row["episode/h4/v4_single_authority_dynamic6_field_count"] = (
        224.6086956521739
    )
    row["episode/h4/v4_saved_dynamic6_field_count"] = 224.6086956521739
    result = runner.require_forward_v4_single_authority_runtime_progress([row])
    assert result["dynamic6_field_count_exact"] is True
    assert result["saved_dynamic6_field_count_exact"] is True
    assert result["authority_violation_count"] == 0.0


def test_forward_v4_count_totals_are_finite_nonqualifying_diagnostics() -> None:
    row = _forward_v4_single_authority_episode_row()
    row["episode/h4/v4_single_authority_dynamic6_field_count"] = 119.0
    row["episode/h4/v4_saved_dynamic6_substep_count"] = 199.0
    row["episode/h4/v4_saved_dynamic6_field_count"] = 119.0
    assert runner.require_forward_v4_single_authority_runtime_progress([row])[
        "passed"
    ] is True
    row["episode/h4/v4_saved_dynamic6_field_count"] = float("nan")
    with pytest.raises(RuntimeError, match="must be finite"):
        runner.require_forward_v4_single_authority_runtime_progress([row])


def test_forward_v4_wiring_no_episode_rows_uses_compiled_assertion_contract() -> None:
    completion = dict(runner.FORWARD_V4_WIRING_COMPLETION_REQUIREMENT)
    assert runner.require_forward_v4_single_authority_runtime_progress(
        [], wiring_only=True, wiring_completion=completion
    ) == {
        "audit_mode": runner.FORWARD_V4_WIRING_RUNTIME_AUDIT_MODE,
        "observed_episode_metric_rows": 0,
        "episode_metric_rows_exact_if_observed": True,
        **runner.FORWARD_V4_WIRING_COMPLETION_REQUIREMENT,
        "authority_violation_count": 0.0,
        "assertion_token_sum": 0.0,
        "passed": True,
    }


def test_forward_v4_full_runtime_no_episode_rows_still_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="no forward-v4 single-authority episode"):
        runner.require_forward_v4_single_authority_runtime_progress([])


def test_forward_v4_wiring_injected_compiled_violation_aborts() -> None:
    completion = dict(runner.FORWARD_V4_WIRING_COMPLETION_REQUIREMENT)
    completion["per_step_compiled_fail_closed_assertion_bound"] = False
    with pytest.raises(RuntimeError, match="completion evidence is not exact"):
        runner.require_forward_v4_single_authority_runtime_progress(
            [], wiring_only=True, wiring_completion=completion
        )

    row = _forward_v4_single_authority_episode_row()
    row["episode/h4/v4_single_authority_violation"] = 1.0
    with pytest.raises(RuntimeError, match="PPO single-authority audit failed"):
        runner.require_forward_v4_single_authority_runtime_progress(
            [row],
            wiring_only=True,
            wiring_completion=runner.FORWARD_V4_WIRING_COMPLETION_REQUIREMENT,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("episode/h4/v4_single_authority_dynamic6_exact", 19.0),
        (
            "episode/h4/v4_single_authority_dynamic6_max_abs_error",
            np.nextafter(0.0, 1.0),
        ),
        ("episode/h4/v4_single_authority_dynamic6_field_count_exact", 19.0),
        ("episode/h4/v4_saved_dynamic6_field_count_exact", 19.0),
        ("episode/h4/v4_saved_dynamic6_all_finite", 19.0),
        ("episode/h4/v4_telemetry_force_shape_valid", 19.0),
        ("episode/h4/v4_telemetry_force_all_finite", 19.0),
        ("episode/h4/v4_single_authority_violation", 1.0),
        ("episode/h4/v4_single_authority_assertion_token", 1.0),
    ),
)
def test_forward_v4_wiring_runtime_authority_rejects_injected_mismatch(
    field: str, value: float
) -> None:
    row = _forward_v4_single_authority_episode_row()
    row[field] = value
    with pytest.raises(RuntimeError, match="PPO single-authority audit failed"):
        runner.require_forward_v4_single_authority_runtime_progress(
            [row]
        )


def _iteration_v6_args(expert: str, *, wiring_only: bool = False) -> object:
    flag = (
        "--forward-iteration-v6-contact-abort-island-only"
        if expert == "forward"
        else "--reverse-iteration-v6-absolute-full-leg-targets"
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
        argv.extend(("--backward-residual-scale", "0.0"))
    argv.extend(
        ("--wiring-only", "--num-timesteps", "40")
        if wiring_only
        else ("--authorize-simulation-training",)
    )
    args = runner.build_parser().parse_args(argv)
    args.learning_rate = 5.0e-5 if expert == "forward" else 3.0e-5
    return args


@pytest.mark.parametrize("expert", ("forward", "reverse"))
@pytest.mark.parametrize("wiring_only", (False, True))
def test_iteration_v6_cli_reward_anchor_shape_and_ids_are_exact(
    expert: str, wiring_only: bool
) -> None:
    args = _iteration_v6_args(expert, wiring_only=wiring_only)
    runner._validate_scalar_configuration(args)
    scales = runner.resolve_reward_scales(args).as_reward_scale_dict()
    anchors = runner.resolve_anchor_config(
        expert,
        forward_iteration_v6_contact_abort_island_only=(expert == "forward"),
        reverse_iteration_v6_absolute_full_leg_targets=(expert == "reverse"),
    )
    assert anchors == (
        runner.FORWARD_ITERATION_V2_ANCHOR_CONFIG
        if expert == "forward"
        else runner.REVERSE_ITERATION_V2_ANCHOR_CONFIG
    )
    assert runner.resolve_physical_sampler_family(args) == (
        "forward_iteration_v2" if expert == "forward" else "reverse_iteration_v2"
    )
    assert scales == (
        runner.forward_iteration_v2_reward_scales().as_reward_scale_dict()
        if expert == "forward"
        else runner.reverse_iteration_v2_reward_scales().as_reward_scale_dict()
    )
    if expert == "forward":
        assert scales["h4_contact_pulse_40ms"] == -1.0
        assert args.forward_v5_contact_pulse_abort_scale_only is False
        full_id = runner.FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_CONTRACT_ID
        wiring_id = (
            runner.FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_WIRING_CONTRACT_ID
        )
    else:
        assert args.backward_residual_scale == 0.0
        assert args.reverse_iteration_v5_no_contact_imitation is False
        full_id = runner.REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_CONTRACT_ID
        wiring_id = (
            runner.REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_WIRING_CONTRACT_ID
        )
    authorization = {"contract_id": full_id}
    assert runner.resolve_execution_contract_id(
        args,
        forward_iteration_v2_authorization=None,
        forward_iteration_v6_contact_abort_island_only_authorization=(
            authorization if expert == "forward" else None
        ),
        reverse_iteration_v2_authorization=None,
        reverse_iteration_v6_absolute_full_leg_targets_authorization=(
            authorization if expert == "reverse" else None
        ),
    ) == (wiring_id if wiring_only else full_id)
    if wiring_only:
        shape = runner.resolve_training_shape(args, runner._load_legacy_trainer())
        assert (
            shape.num_timesteps,
            shape.num_envs,
            shape.interactions_per_training_step,
            shape.expected_training_steps,
            shape.expected_optimizer_updates,
        ) == (40, 2, 40, 1, 2)


@pytest.mark.parametrize("expert", ("forward", "reverse"))
def test_iteration_v6_sampler_family_cannot_leak_to_legacy(expert: str) -> None:
    v6_args = _iteration_v6_args(expert)
    assert runner.resolve_physical_sampler_family(v6_args) != expert

    legacy_args = runner.build_parser().parse_args(("--expert", expert))
    assert runner.resolve_physical_sampler_family(legacy_args) == expert


@pytest.mark.parametrize("expert", ("forward", "reverse"))
@pytest.mark.parametrize("wiring_only", (False, True))
def test_iteration_v6_authorized_contract_accepts_only_own_authority(
    expert: str, wiring_only: bool
) -> None:
    args = _iteration_v6_args(expert, wiring_only=wiring_only)
    runner._validate_scalar_configuration(args)
    shape = (
        runner.resolve_training_shape(args, runner._load_legacy_trainer())
        if wiring_only
        else SimpleNamespace(num_timesteps=250_000, num_envs=1250)
    )
    kwargs = {
        "shape": shape,
        "reward_scale_dict": runner.resolve_reward_scales(args).as_reward_scale_dict(),
        "anchors": runner.resolve_anchor_config(
            expert,
            forward_iteration_v6_contact_abort_island_only=(expert == "forward"),
            reverse_iteration_v6_absolute_full_leg_targets=(expert == "reverse"),
        ),
        "forward_spec": (
            {"stale_width_declaration_detected": False}
            if expert == "forward"
            else None
        ),
        "forward_iteration_v2_authorization": None,
        "forward_iteration_v3_touchdown_balance_authorization": None,
        "forward_iteration_v4_contact_event_validity_persistence_authorization": None,
        "forward_v5_contact_pulse_abort_scale_only_authorization": None,
        "forward_iteration_v6_contact_abort_island_only_authorization": (
            {} if expert == "forward" else None
        ),
        "reverse_spec": {} if expert == "reverse" else None,
        "reverse_authorization": {} if expert == "reverse" else None,
        "reverse_iteration_v2_authorization": None,
        "reverse_iteration_v3_no_target_imitation_authorization": None,
        "reverse_iteration_v4_residual_transfer_gain_024_authorization": None,
        "reverse_iteration_v5_no_contact_imitation_authorization": None,
        "reverse_iteration_v6_absolute_full_leg_targets_authorization": (
            {} if expert == "reverse" else None
        ),
    }
    runner._validate_authorized_training_contract(args, **kwargs)
    prior = (
        "forward_v5_contact_pulse_abort_scale_only_authorization"
        if expert == "forward"
        else "reverse_iteration_v5_no_contact_imitation_authorization"
    )
    kwargs[prior] = {}
    with pytest.raises(ValueError, match="prior .* authorization cannot bind a v6"):
        runner._validate_authorized_training_contract(args, **kwargs)


@pytest.mark.parametrize(
    ("expert", "mutation", "message"),
    (
        ("forward", {"seed": 20260810}, "seed 20260809"),
        ("forward", {"forward_v5_contact_pulse_abort_scale_only": True}, "mutually exclusive"),
        ("forward", {"reset_noise_multiplier": 0.0}, "reset-noise multiplier 1.0"),
        ("reverse", {"seed": 20260809}, "seed 20260810"),
        (
            "reverse",
            {"backward_residual_scale": np.nextafter(0.0, 1.0)},
            "residual authority scale 0",
        ),
        ("reverse", {"reverse_iteration_v5_no_contact_imitation": True}, "mutually exclusive"),
    ),
)
def test_iteration_v6_cli_near_misses_fail_closed(
    expert: str, mutation: dict[str, object], message: str
) -> None:
    args = _iteration_v6_args(expert)
    for name, value in mutation.items():
        setattr(args, name, value)
    with pytest.raises(ValueError, match=message):
        runner._validate_scalar_configuration(args)


@pytest.mark.parametrize("expert", ("forward", "reverse"))
def test_iteration_v6_authorization_schema_is_exact_and_ulp_closed(
    expert: str,
) -> None:
    payload = runner._iteration_v6_expected_payload(expert)
    assert runner.validate_iteration_v6_authorization_payload(
        copy.deepcopy(payload), expert=expert
    )["passed"] is True

    missing = copy.deepcopy(payload)
    missing.pop("manifest_binding")
    with pytest.raises(ValueError, match="schema drift"):
        runner.validate_iteration_v6_authorization_payload(missing, expert=expert)

    extra = copy.deepcopy(payload)
    extra["training_contract"]["extra"] = 0
    with pytest.raises(ValueError, match="schema drift"):
        runner.validate_iteration_v6_authorization_payload(extra, expert=expert)

    ulp = copy.deepcopy(payload)
    ulp["training_contract"]["learning_rate"] = float(
        np.nextafter(ulp["training_contract"]["learning_rate"], np.inf)
    )
    with pytest.raises(ValueError, match="float drift"):
        runner.validate_iteration_v6_authorization_payload(ulp, expert=expert)


def _h5_v3_substep_preflight_args(*extra: str) -> object:
    args = runner.build_parser().parse_args(
        (
            "--expert",
            "unified",
            "--diagnostic-reward-exploration",
            "--observation-mode",
            "h4_116_transplant",
            "--allow-verified-v22-transplant",
            "--seed",
            "20260823",
            "--h5-unified-command-mapper",
            "direct_normalized_v3",
            "--h5-v3-command-conditioned-se2-alignment",
            "--h5-v3-substep-contact-alignment",
            "--h5-v3-substep-contact-preflight-only",
            *extra,
        )
    )
    args.learning_rate = 5.0e-5
    return args


def test_h5_v3_substep_preflight_is_clean_no_ppo_and_has_only_three_scales() -> None:
    args = _h5_v3_substep_preflight_args()
    runner._validate_scalar_configuration(args)
    assert args.authorize_simulation_training is False
    assert args.wiring_only is False
    scales = runner.resolve_reward_scales(args).as_reward_scale_dict(
        include_h5_substep_contact_alignment=True
    )
    assert {
        name: scales[name]
        for name in scales
        if name.startswith("h5_all_substep_")
    } == {
        "h5_all_substep_strict20ms_slip_rms": -1.0,
        "h5_all_substep_slip_tail": -1.0,
        "h5_all_substep_force_tail": -1.0,
    }
    assert "h5_all_substep_force_tail" not in (
        runner.H4QualityRewardScales().as_reward_scale_dict()
    )


def test_h5_v3_substep_preflight_raw_comparator_preserves_nan_and_signed_zero() -> None:
    equal_nan = np.asarray([np.nan], dtype=np.float32)
    assert runner.h5_preflight_raw_array_equal(equal_nan, equal_nan.copy())
    assert not runner.h5_preflight_raw_array_equal(
        np.asarray([0.0], dtype=np.float32),
        np.asarray([-0.0], dtype=np.float32),
    )
    assert runner.h5_preflight_raw_array_digest(equal_nan) == (
        runner.h5_preflight_raw_array_digest(equal_nan.copy())
    )


def test_h5_v3_substep_preflight_difference_records_exact_float_evidence() -> None:
    difference = runner.h5_preflight_raw_array_difference(
        np.asarray([0.0, 1.0], dtype=np.float32),
        np.asarray([-0.0, 1.0000001], dtype=np.float32),
    )

    assert difference["exact_raw_equal"] is False
    assert difference["first_differing_flat_index"] == 0
    assert difference["first_differing_index"] == [0]
    assert difference["differing_element_count"] == 2
    assert difference["max_ulp_difference"] >= 1


def test_h5_v3_substep_preflight_uses_exact_vector_witness_without_training_auth() -> None:
    args = _h5_v3_substep_preflight_args()
    shape = runner.resolve_training_shape(args, runner._load_legacy_trainer())

    assert (
        shape.num_timesteps,
        shape.num_envs,
        shape.unroll_length,
        shape.batch_size,
        shape.num_minibatches,
        shape.num_updates_per_batch,
    ) == (40, 2, 20, 1, 2, 1)
    args.num_timesteps = 40
    with pytest.raises(ValueError, match="permits no custom timestep count"):
        runner.resolve_training_shape(args, runner._load_legacy_trainer())


def test_h5_v3_substep_t1_eager_probe_remains_no_ppo() -> None:
    args = _h5_v3_substep_preflight_args(
        "--h5-v3-substep-contact-t1-diagnostic-arm",
        "treatment",
        "--h5-v3-substep-contact-t1-diagnostic-output",
        "artifacts/t1_eager_probe_test_only.json",
        "--h5-v3-substep-contact-t1-diagnostic-mode",
        "eager_s1",
    )

    runner._validate_scalar_configuration(args)

    assert args.h5_v3_substep_contact_t1_diagnostic_mode == "eager_s1"
    assert args.authorize_simulation_training is False
    assert args.wiring_only is False


def test_h5_v3_substep_t1_vmap_b1_lane_one_probe_remains_no_ppo() -> None:
    args = _h5_v3_substep_preflight_args(
        "--h5-v3-substep-contact-t1-diagnostic-arm",
        "treatment",
        "--h5-v3-substep-contact-t1-diagnostic-output",
        "artifacts/t1_vmap_b1_lane_one_probe_test_only.json",
        "--h5-v3-substep-contact-t1-diagnostic-mode",
        "vmap_b1_s1",
    )

    runner._validate_scalar_configuration(args)

    assert args.h5_v3_substep_contact_t1_diagnostic_mode == "vmap_b1_s1"
    assert args.authorize_simulation_training is False
    assert args.wiring_only is False


def test_h5_v3_substep_fixed_replay_ablation_is_t1_only() -> None:
    without_t1 = _h5_v3_substep_preflight_args(
        "--h5-v3-substep-contact-t1-fixed-quality-replay-ablation"
    )
    with pytest.raises(ValueError, match="requires a T=1 arm"):
        runner._validate_scalar_configuration(without_t1)

    eager_t1 = _h5_v3_substep_preflight_args(
        "--h5-v3-substep-contact-t1-diagnostic-arm",
        "treatment",
        "--h5-v3-substep-contact-t1-diagnostic-output",
        "artifacts/t1_fixed_replay_probe_test_only.json",
        "--h5-v3-substep-contact-t1-diagnostic-mode",
        "eager_s1",
        "--h5-v3-substep-contact-t1-fixed-quality-replay-ablation",
    )
    runner._validate_scalar_configuration(eager_t1)


def test_v4_collector_trace_preflight_is_h5_free_no_ppo_b2_t20() -> None:
    args = runner.build_parser().parse_args(
        [
            "--expert",
            "unified",
            "--diagnostic-reward-exploration",
            "--h5-unified-command-mapper",
            "direct_normalized_v3",
            "--v4-substep-collector-trace-preflight-only",
        ]
    )
    args.learning_rate = 5.0e-5
    runner._validate_scalar_configuration(args)
    shape = runner.resolve_training_shape(args, runner._load_legacy_trainer())
    assert (
        shape.num_timesteps,
        shape.num_envs,
        shape.unroll_length,
        shape.batch_size,
        shape.num_minibatches,
        shape.num_updates_per_batch,
    ) == (40, 2, 20, 1, 2, 1)


def test_v4_collector_trace_preflight_rejects_h5_v3_or_training() -> None:
    for extra in (
        ["--h5-v3-command-conditioned-se2-alignment"],
        ["--authorize-simulation-training"],
        ["--wiring-only"],
    ):
        args = runner.build_parser().parse_args(
            [
                "--expert",
                "unified",
                "--diagnostic-reward-exploration",
                "--h5-unified-command-mapper",
                "direct_normalized_v3",
                "--v4-substep-collector-trace-preflight-only",
                *extra,
            ]
        )
        args.learning_rate = 5.0e-5
        with pytest.raises(ValueError, match="V4 collector trace preflight"):
            runner._validate_scalar_configuration(args)


def _v4_authoritative_primitive_parity_args(*extra: str) -> object:
    args = runner.build_parser().parse_args(
        (
            "--expert",
            "unified",
            "--platform",
            "gpu",
            "--diagnostic-reward-exploration",
            "--h5-unified-command-mapper",
            "direct_normalized_v3",
            "--v4-authoritative-primitive-batch-parity-preflight-only",
            "--v4-authoritative-primitive-batch-parity-preflight-output",
            "artifacts/v4_primitive_batch_parity_test_only.json",
            *extra,
        )
    )
    args.learning_rate = 5.0e-5
    return args


def test_v4_authoritative_primitive_parity_is_gpu_only_no_ppo_b2_t20() -> None:
    args = _v4_authoritative_primitive_parity_args()

    # Validation must be pure argument checking: this used to attempt to access
    # runtime-only env/stack/paths values before run() had constructed them.
    runner._validate_scalar_configuration(args)
    shape = runner.resolve_training_shape(args, runner._load_legacy_trainer())

    assert args.authorize_simulation_training is False
    assert args.wiring_only is False
    assert (
        shape.num_timesteps,
        shape.num_envs,
        shape.unroll_length,
        shape.batch_size,
        shape.num_minibatches,
        shape.num_updates_per_batch,
    ) == (40, 2, 20, 1, 2, 1)


@pytest.mark.parametrize(
    ("extra", "message"),
    (
        (("--platform", "cpu"), "GPU-only"),
        (("--authorize-simulation-training",), "no-PPO"),
        (("--v4-substep-collector-trace-preflight-only",), "cannot combine"),
        (("--run-name", "must_not_write"), "cannot combine"),
        (("--h5-unified-reverse-route-probability", "0.5"), "cannot combine"),
    ),
)
def test_v4_authoritative_primitive_parity_rejects_invalid_modes(
    extra: tuple[str, ...], message: str
) -> None:
    args = _v4_authoritative_primitive_parity_args(*extra)
    with pytest.raises(ValueError, match=message):
        runner._validate_scalar_configuration(args)


def test_v4_authoritative_primitive_parity_dispatch_is_before_eval_and_ppo_boundary() -> None:
    module = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    validator = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_validate_scalar_configuration"
    )
    run_function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )

    def preflight_calls(function: ast.FunctionDef) -> list[ast.Call]:
        return [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_v4_authoritative_primitive_batch_parity_preflight"
        ]

    assert preflight_calls(validator) == []
    dispatches = preflight_calls(run_function)
    assert len(dispatches) == 1
    eval_environment_assignment = next(
        node
        for node in ast.walk(run_function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "eval_env"
            for target in node.targets
        )
    )
    checkpoint_import = next(
        node
        for node in ast.walk(run_function)
        if isinstance(node, ast.ImportFrom)
        and node.module == "brax.training.agents.ppo"
    )
    assert dispatches[0].lineno < eval_environment_assignment.lineno
    assert dispatches[0].lineno < checkpoint_import.lineno


def test_v4_stablehlo_dump_requires_collector_preflight() -> None:
    args = runner.build_parser().parse_args(
        [
            "--expert",
            "unified",
            "--diagnostic-reward-exploration",
            "--h5-unified-command-mapper",
            "direct_normalized_v3",
            "--v4-substep-collector-trace-stablehlo-dump-output",
            "artifacts/collector.mlir",
        ]
    )
    with pytest.raises(ValueError, match="StableHLO dumping"):
        runner._validate_scalar_configuration(args)


def test_stablehlo_location_stripped_hash_preserves_operations() -> None:
    first = """\
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32> loc(#loc2)
    return %0 : tensor<f32> loc(#loc3)
    #loc2 = loc(\"first.py\":10:2)
    #loc3 = loc(\"first.py\":11:2)
    """
    changed_locations = """\
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32> loc(#loc8)
    return %0 : tensor<f32> loc(#loc9)
    #loc8 = loc(\"second.py\":900:4)
    #loc9 = loc(\"second.py\":901:4)
    """
    changed_operation = changed_locations.replace("stablehlo.add", "stablehlo.subtract")

    assert (
        runner.stablehlo_location_stripped_sha256(first)
        == runner.stablehlo_location_stripped_sha256(changed_locations)
    )
    assert (
        runner.stablehlo_location_stripped_sha256(first)
        != runner.stablehlo_location_stripped_sha256(changed_operation)
    )


def test_stablehlo_semantic_hash_normalizes_only_callback_registry_handle() -> None:
    first = """\
    %c_0 = stablehlo.constant dense<1001> : tensor<i64> loc(#loc2)
    %0 = stablehlo.custom_call @xla_python_cpu_callback(%c_0, %arg0) {api_version = 2 : i32, backend_config = \"1001\", has_side_effect = true} : (tensor<i64>, tensor<f32>) -> tuple<> loc(#loc3)
    #loc2 = loc(\"first.py\":10:2)
    #loc3 = loc(\"first.py\":11:2)
    """
    changed_handle = first.replace("1001", "9002").replace("first.py", "second.py")
    changed_effect = changed_handle.replace("has_side_effect = true", "has_side_effect = false")
    unrelated_i64 = changed_handle.replace(
        "%c_0 = stablehlo.constant dense<9002>",
        "%c_0 = stablehlo.constant dense<7>",
    ).replace("@xla_python_cpu_callback", "@not_a_python_callback")

    assert (
        runner.stablehlo_semantic_sha256(first)
        == runner.stablehlo_semantic_sha256(changed_handle)
    )
    assert (
        runner.stablehlo_semantic_sha256(first)
        != runner.stablehlo_semantic_sha256(changed_effect)
    )
    assert (
        runner.stablehlo_semantic_sha256(first)
        != runner.stablehlo_semantic_sha256(unrelated_i64)
    )


def test_sealed_trace_sidecar_preflight_never_constructs_an_environment(
    tmp_path: Path,
) -> None:
    sidecar_runner_path = (
        EXPERIMENT_ROOT / "scripts/run_h5_sidecar_sealed_trace_preflight.py"
    )
    sidecar_spec = importlib.util.spec_from_file_location(
        "exp004_sealed_sidecar_preflight", sidecar_runner_path
    )
    assert sidecar_spec is not None and sidecar_spec.loader is not None
    sidecar_runner = importlib.util.module_from_spec(sidecar_spec)
    sys.modules[sidecar_spec.name] = sidecar_runner
    sidecar_spec.loader.exec_module(sidecar_runner)

    time = (
        np.arange(20, dtype=np.float32)[:, None, None] * np.float32(0.02)
        + np.arange(10, dtype=np.float32)[None, None, :] * np.float32(0.002)
    )
    fields = {
        "time_s": np.broadcast_to(time, (20, 2, 10)).copy(),
        "normalized_normal_force": np.full((20, 2, 10, 2), 0.5, dtype=np.float32),
        "tangential_speed_m_s": np.zeros((20, 2, 10, 2), dtype=np.float32),
        "terminal_after_tick": np.zeros((20, 2), dtype=bool),
        "reset_normalized_force": np.full((2, 2), 0.5, dtype=np.float32),
        "base_reward": np.zeros((20, 2), dtype=np.float32),
    }
    sealed_path = tmp_path / "trace.npz"
    np.savez(sealed_path, **fields)
    source_paths = {
        "h4_training_alignment": EXPERIMENT_ROOT
        / "safe_gait_experts/h4_training_alignment.py",
        "h5_substep_contact_alignment": EXPERIMENT_ROOT
        / "safe_gait_experts/h5_substep_contact_alignment.py",
        "h5_sidecar_quality": EXPERIMENT_ROOT / "safe_gait_experts/h5_sidecar_quality.py",
    }
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(
        json.dumps(
            {
                "status": "V4_COLLECTOR_TRACE_RAW_PARITY_PASS_NOT_A_TRAINING_CANDIDATE",
                "hardware_deployment": "PROHIBITED",
                "checks": {
                    name: True for name in sidecar_runner.PARENT_REQUIRED_CHECKS
                },
                "trace_repeat": {
                    "raw_equal": True,
                    "first_raw_tree_sha256": "synthetic-test-only",
                },
                "no_ppo_tripwire": {
                    "ppo_train_called": False,
                    "checkpoint_written": False,
                    "training_run_directory_created": False,
                    "preflight_returns_before_ppo_path": True,
                },
                "bound_inputs_pre_and_post": {
                    name: {
                        "path": str(path),
                        "sha256": sidecar_runner.sha256_file(path),
                    }
                    for name, path in source_paths.items()
                },
                "sealed_trace": {
                    "path": str(sealed_path),
                    "sha256": sidecar_runner.sha256_file(sealed_path),
                    "serialization": "numpy.savez allow_pickle=false v1",
                    "field_order": list(sidecar_runner.SEALED_TRACE_FIELD_ORDER),
                    "field_raw_bytes_sha256": {
                        name: sidecar_runner.raw_array_digest(value)
                        for name, value in fields.items()
                    },
                    "ordered_field_bundle_sha256": (
                        sidecar_runner.ordered_array_bundle_digest(fields)
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    result = sidecar_runner.run(parent_path, tmp_path / "sidecar.json")

    failed_checks = [
        name for name, passed in result["checks"].items() if not passed
    ]
    assert result["status"] == "CPU_PURE_H5_SIDECAR_SEALED_TRACE_NO_PPO_PASS", failed_checks
    assert result["execution"]["environment_instances"] == 0
    assert result["execution"]["simulator_calls"] == 0
    assert all(result["checks"].values())


@pytest.mark.parametrize(
    ("extra", "message"),
    (
        (("--authorize-simulation-training",), "no-PPO"),
        (("--wiring-only",), "no-PPO"),
        (("--reward-force-slip", "-2.1"), "authorization-controlled"),
    ),
)
def test_h5_v3_substep_preflight_rejects_training_or_reward_drift(
    extra: tuple[str, ...], message: str
) -> None:
    args = _h5_v3_substep_preflight_args(*extra)
    if extra[0].startswith("--reward-"):
        runner._validate_scalar_configuration(args)
        with pytest.raises(ValueError, match=message):
            runner.resolve_reward_scales(args)
    else:
        with pytest.raises(ValueError, match=message):
            runner._validate_scalar_configuration(args)


@pytest.mark.parametrize("expert", ("forward", "reverse"))
def test_iteration_v6_real_authorization_loader_closes_frozen_bytes_and_inputs(
    expert: str,
) -> None:
    loaded = runner.load_iteration_v6_authorization(expert=expert)
    spec = runner._iteration_v6_spec(expert)
    assert loaded["sha256"] == spec["auth_sha"]
    assert loaded["contract_id"] == spec["contract_id"]
    assert loaded["payload"] == runner._iteration_v6_expected_payload(expert)
    assert loaded["semantic_audit"] == {
        "top_level_fields_exact": True,
        "nested_schema_exact": True,
        "numeric_types_and_values_exact": True,
        "causal_identity_exact": True,
        "training_contract_exact": True,
        "runtime_contract_exact": True,
        "passed": True,
    }
    assert all(
        Path(record["path"]).is_file()
        and runner.sha256_file(Path(record["path"])) == record["sha256"]
        for record in loaded["bound_causal_inputs"].values()
    )
    assert len(loaded["bound_historical_v5_sources"]) == 5


def test_iteration_v6_core_source_is_byte_pinned_before_execution(
    tmp_path: Path,
) -> None:
    expected = {
        "path": str(runner.ALIGNMENT_MODULE_PATH.resolve()),
        "sha256": runner.PINNED_ITERATION_V6_H4_TRAINING_ALIGNMENT_SHA256,
    }
    assert runner.require_iteration_v6_core_source() == expected

    drifted = tmp_path / "h4_training_alignment.py"
    frozen_bytes = runner.ALIGNMENT_MODULE_PATH.read_bytes()
    drifted.write_bytes(frozen_bytes + b"\n")
    with pytest.raises(ValueError, match="core source SHA256 drifted"):
        runner.require_iteration_v6_core_source(drifted)


@pytest.mark.parametrize("expert", ("forward", "reverse"))
def test_iteration_v6_authorization_byte_drift_fails_before_json_parse(
    expert: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = runner._iteration_v6_spec(expert)
    path = tmp_path / spec["auth_filename"]
    frozen_bytes = Path(spec["auth_path"]).read_bytes()
    mutated_bytes = bytearray(frozen_bytes)
    mutated_bytes[-1] = 0x20 if mutated_bytes[-1] != 0x20 else 0x0A
    path.write_bytes(mutated_bytes)
    calls = {"json": 0}

    def reject_json_parse(_path: Path) -> object:
        calls["json"] += 1
        raise AssertionError("JSON parse reached")

    monkeypatch.setattr(runner, "_load_json_strict", reject_json_parse)
    with pytest.raises(ValueError, match="SHA256 drifted"):
        runner.load_iteration_v6_authorization(expert=expert, path=path)
    assert calls == {"json": 0}


def _forward_v6_sample() -> dict[str, object]:
    return {
        "routing_exact": True,
        "island_loss": 1.25,
        "off_gap_diagnostic_loss": 0.75,
        "off_gap_reward_contribution": 0.0,
        "pulse_reward_scale": -1.0,
        "routing_violation": False,
        "assertion_token": 0.0,
    }


def _reverse_v6_sample() -> dict[str, object]:
    return {
        "decoder_exact": True,
        "max_abs_error": 0.0,
        "leg_count": 10.0,
        "leg_count_exact": True,
        "head_zero_exact": True,
        "teacher_target_contribution_zero_exact": True,
        "residual_authority_scale": 0.0,
        "decoder_all_finite": True,
        "margin_saturation_count": 2.0,
        "action_clip_count": 1.0,
        "guard_lag_max_rad": 0.2,
        "precomposer_call_count": 1.0,
        "precomposer_call_count_exact": True,
        "final_guard_call_count": 1.0,
        "final_guard_call_count_exact": True,
        "decoder_violation": False,
        "assertion_token": 0.0,
    }


@pytest.mark.parametrize(
    ("expert", "sample"),
    (("forward", _forward_v6_sample()), ("reverse", _reverse_v6_sample())),
)
def test_iteration_v6_runtime_samples_are_exact(
    expert: str, sample: dict[str, object]
) -> None:
    assert runner.require_iteration_v6_runtime_samples(
        [sample], expert=expert, label="test"
    ) == {
        "expert": expert,
        "observed_step_count": 1,
        "compiled_invariant_assertion_passed": True,
        "passed": True,
    }


@pytest.mark.parametrize(
    ("expert", "field", "value"),
    (
        ("forward", "routing_exact", False),
        ("forward", "island_loss", float("nan")),
        ("forward", "off_gap_diagnostic_loss", -1.0),
        ("forward", "off_gap_reward_contribution", np.nextafter(0.0, 1.0)),
        ("forward", "pulse_reward_scale", np.nextafter(-1.0, 0.0)),
        ("forward", "routing_violation", True),
        ("forward", "assertion_token", 1.0),
        ("reverse", "decoder_exact", False),
        ("reverse", "max_abs_error", np.nextafter(0.0, 1.0)),
        ("reverse", "leg_count", float("nan")),
        ("reverse", "leg_count_exact", False),
        ("reverse", "head_zero_exact", False),
        ("reverse", "teacher_target_contribution_zero_exact", False),
        ("reverse", "residual_authority_scale", np.nextafter(0.0, 1.0)),
        ("reverse", "decoder_all_finite", False),
        ("reverse", "margin_saturation_count", float("nan")),
        ("reverse", "action_clip_count", -1.0),
        ("reverse", "guard_lag_max_rad", float("inf")),
        ("reverse", "precomposer_call_count", float("nan")),
        ("reverse", "precomposer_call_count_exact", False),
        ("reverse", "final_guard_call_count", -1.0),
        ("reverse", "final_guard_call_count_exact", False),
        ("reverse", "decoder_violation", True),
        ("reverse", "assertion_token", 1.0),
    ),
)
def test_iteration_v6_runtime_samples_reject_each_near_miss(
    expert: str, field: str, value: object
) -> None:
    sample = _forward_v6_sample() if expert == "forward" else _reverse_v6_sample()
    sample[field] = value
    with pytest.raises(RuntimeError):
        runner.require_iteration_v6_runtime_samples(
            [sample], expert=expert, label="test"
        )


def test_reverse_iteration_v6_decoder_vectors_are_complete_and_finite() -> None:
    sample = {
        "action": np.linspace(-1.0, 1.0, 14),
        "raw_targets": np.r_[np.ones(5), np.zeros(4), np.ones(5)],
        "margin_targets": np.r_[np.full(5, 0.5), np.zeros(4), np.full(5, 0.5)],
    }
    assert runner.require_reverse_iteration_v6_decoder_vector_samples(
        [sample], label="test"
    ) == {
        "action_shape": [14],
        "raw_targets_shape": [14],
        "margin_targets_shape": [14],
        "target_head_channels_exact_zero": True,
        "all_finite": True,
        "observed_step_count": 1,
        "passed": True,
    }
    for field, broken_value in (
        ("action", np.zeros(13)),
        ("raw_targets", np.r_[np.ones(5), np.ones(4), np.ones(5)]),
        ("margin_targets", np.full(14, np.nan)),
    ):
        broken = {key: value.copy() for key, value in sample.items()}
        broken[field] = broken_value
        with pytest.raises(RuntimeError):
            runner.require_reverse_iteration_v6_decoder_vector_samples(
                [broken], label="test"
            )


def _iteration_v6_episode_row(expert: str, length: float = 20.0) -> dict[str, float]:
    sample = _forward_v6_sample() if expert == "forward" else _reverse_v6_sample()
    keys = (
        runner.FORWARD_ITERATION_V6_REWARD_ROUTING_EPISODE_KEYS
        if expert == "forward"
        else runner.REVERSE_ITERATION_V6_DECODER_EPISODE_KEYS
    )
    row = {"episode/length": length}
    for name, key in keys.items():
        value = sample[name]
        row[key] = float(value) * length
    return row


@pytest.mark.parametrize("expert", ("forward", "reverse"))
def test_iteration_v6_full_and_wiring_runtime_stage_contracts(expert: str) -> None:
    full_completion = dict(runner.ITERATION_V6_FULL_COMPLETION_REQUIREMENT)
    assert runner.require_iteration_v6_runtime_progress(
        [_iteration_v6_episode_row(expert)],
        expert=expert,
        full_completion=full_completion,
    )["passed"] is True
    with pytest.raises(RuntimeError, match="no .* iteration-v6 episode"):
        runner.require_iteration_v6_runtime_progress(
            [], expert=expert, full_completion=full_completion
        )
    full_completion["completed_environment_interactions"] = 249999
    with pytest.raises(RuntimeError, match="full completion evidence is not exact"):
        runner.require_iteration_v6_runtime_progress(
            [_iteration_v6_episode_row(expert)],
            expert=expert,
            full_completion=full_completion,
        )
    completion = dict(runner.ITERATION_V6_WIRING_COMPLETION_REQUIREMENT)
    assert runner.require_iteration_v6_runtime_progress(
        [], expert=expert, wiring_only=True, wiring_completion=completion
    )["passed"] is True
    completion["per_step_compiled_fail_closed_assertion_bound"] = False
    with pytest.raises(RuntimeError, match="completion evidence is not exact"):
        runner.require_iteration_v6_runtime_progress(
            [], expert=expert, wiring_only=True, wiring_completion=completion
        )


def test_reverse_iteration_v6_fractional_episode_uses_device_exact_count_bools() -> None:
    length = 37.43478260869565
    row = _iteration_v6_episode_row("reverse", length)
    row[
        runner.REVERSE_ITERATION_V6_DECODER_EPISODE_KEYS["leg_count"]
    ] = 374.3478260869565
    row[
        runner.REVERSE_ITERATION_V6_DECODER_EPISODE_KEYS["precomposer_call_count"]
    ] = 37.43478260869564
    row[
        runner.REVERSE_ITERATION_V6_DECODER_EPISODE_KEYS["final_guard_call_count"]
    ] = 37.43478260869566
    assert runner.require_iteration_v6_runtime_progress(
        [row],
        expert="reverse",
        full_completion=dict(runner.ITERATION_V6_FULL_COMPLETION_REQUIREMENT),
    )["passed"] is True


def _iteration_v6_cross_bound_artifacts(
    expert: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    payload = runner._iteration_v6_expected_payload(expert)
    contract_id = payload["scope"]["contract_id"]
    sha = "a" * 64
    selected = (
        "forward_iteration_v6_contact_abort_island_only"
        if expert == "forward"
        else "reverse_iteration_v6_absolute_full_leg_targets"
    )
    requirement_key = (
        "forward_iteration_v6_reward_routing_runtime_requirement"
        if expert == "forward"
        else "reverse_iteration_v6_decoder_runtime_requirement"
    )
    requirement = (
        runner.FORWARD_ITERATION_V6_REWARD_ROUTING_RUNTIME_REQUIREMENT
        if expert == "forward"
        else runner.REVERSE_ITERATION_V6_DECODER_RUNTIME_REQUIREMENT
    )
    runtime_key = (
        "forward_iteration_v6_reward_routing_runtime"
        if expert == "forward"
        else "reverse_iteration_v6_decoder_runtime"
    )
    authorization_key = (
        "forward_iteration_v6_contact_abort_island_only_authorization"
        if expert == "forward"
        else "reverse_iteration_v6_absolute_full_leg_targets_authorization"
    )
    result_sha_key = f"{authorization_key}_sha256"
    common = {
        **{key: key == selected for key in runner.ITERATION_MODE_BOOLEAN_FIELDS},
        "training_contract_id": "V6_TEST_EXECUTION",
        "authorized_iteration_v6_250k_contract_id": contract_id,
        "iteration_v6_core_source": {
            "path": str(runner.ALIGNMENT_MODULE_PATH.resolve()),
            "sha256": runner.PINNED_ITERATION_V6_H4_TRAINING_ALIGNMENT_SHA256,
        },
        requirement_key: dict(requirement),
        **{
            key: copy.deepcopy(payload[key])
            for key in (
                ("reward_routing_contract",)
                if expert == "forward"
                else ("action_parameterization_contract", "teacher_timing_contract")
            )
        },
    }
    config = {
        **copy.deepcopy(common),
        authorization_key: {"sha256": sha, "contract_id": contract_id},
    }
    manifest = {
        **copy.deepcopy(common),
        authorization_key: {"sha256": sha, "contract_id": contract_id},
        runtime_key: {"expert": expert, "passed": True},
    }
    result = {
        **copy.deepcopy(common),
        result_sha_key: sha,
        runtime_key: {"expert": expert, "passed": True},
    }
    authorization = {
        "payload": payload,
        "contract_id": contract_id,
        "sha256": sha,
    }
    return config, manifest, result, authorization


@pytest.mark.parametrize("expert", ("forward", "reverse"))
@pytest.mark.parametrize("wiring_only", (True, False))
def test_iteration_v6_real_authorization_synthetic_run_artifacts_bind_exact_execution_contract(
    expert: str, wiring_only: bool
) -> None:
    args = _iteration_v6_args(expert, wiring_only=wiring_only)
    authorization = (
        runner.load_forward_iteration_v6_contact_abort_island_only_authorization()
        if expert == "forward"
        else runner.load_reverse_iteration_v6_absolute_full_leg_targets_authorization()
    )
    full_id = (
        runner.FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_CONTRACT_ID
        if expert == "forward"
        else runner.REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_CONTRACT_ID
    )
    wiring_id = (
        runner.FORWARD_ITERATION_V6_CONTACT_ABORT_ISLAND_ONLY_WIRING_CONTRACT_ID
        if expert == "forward"
        else runner.REVERSE_ITERATION_V6_ABSOLUTE_FULL_LEG_TARGETS_WIRING_CONTRACT_ID
    )
    execution_id = runner.resolve_execution_contract_id(
        args,
        forward_iteration_v2_authorization=None,
        forward_iteration_v6_contact_abort_island_only_authorization=(
            authorization if expert == "forward" else None
        ),
        reverse_iteration_v2_authorization=None,
        reverse_iteration_v6_absolute_full_leg_targets_authorization=(
            authorization if expert == "reverse" else None
        ),
    )
    assert authorization["contract_id"] == full_id
    assert execution_id == (wiring_id if wiring_only else full_id)
    assert execution_id not in {"H4_FORWARD_ITERATION_V1", "H4_REVERSE_ITERATION_V1"}

    config, manifest, result, _ = _iteration_v6_cross_bound_artifacts(expert)
    authorization_key = (
        "forward_iteration_v6_contact_abort_island_only_authorization"
        if expert == "forward"
        else "reverse_iteration_v6_absolute_full_leg_targets_authorization"
    )
    result_sha_key = f"{authorization_key}_sha256"
    for artifact in (config, manifest, result):
        artifact["training_contract_id"] = execution_id
    for artifact in (config, manifest):
        artifact[authorization_key] = {
            "sha256": authorization["sha256"],
            "contract_id": authorization["contract_id"],
        }
    result[result_sha_key] = authorization["sha256"]

    assert {
        config["training_contract_id"],
        manifest["training_contract_id"],
        result["training_contract_id"],
    } == {wiring_id if wiring_only else full_id}
    assert runner.require_iteration_v6_artifact_cross_binding(
        config,
        manifest,
        result,
        authorization,
        expert=expert,
    )["passed"] is True


@pytest.mark.parametrize("expert", ("forward", "reverse"))
def test_iteration_v6_config_manifest_result_cross_binding_is_exact(
    expert: str,
) -> None:
    artifacts = _iteration_v6_cross_bound_artifacts(expert)
    assert runner.require_iteration_v6_artifact_cross_binding(
        *artifacts[:3], artifacts[3], expert=expert
    ) == {
        "all_ten_iteration_mode_booleans_exact": True,
        "authorization_sha_and_contract_id_exact": True,
        "execution_contract_id_cross_bound": True,
        "runtime_requirement_cross_bound": True,
        "core_source_cross_bound": True,
        "authorization_contracts_cross_bound": True,
        "expert_runtime_evidence_cross_bound": True,
        "passed": True,
    }


@pytest.mark.parametrize(
    ("expert", "artifact_index", "mutation"),
    (
        ("forward", 0, "contract"),
        ("forward", 1, "generic_runtime"),
        ("forward", 2, "mode"),
        ("reverse", 0, "contract"),
        ("reverse", 1, "generic_runtime"),
        ("reverse", 2, "mode"),
    ),
)
def test_iteration_v6_artifact_cross_binding_near_misses_fail_closed(
    expert: str, artifact_index: int, mutation: str
) -> None:
    artifacts = list(_iteration_v6_cross_bound_artifacts(expert))
    artifact = artifacts[artifact_index]
    if mutation == "contract":
        key = (
            "reward_routing_contract"
            if expert == "forward"
            else "action_parameterization_contract"
        )
        artifact[key] = {**artifact[key], "extra": True}
    elif mutation == "generic_runtime":
        artifact["iteration_v6_runtime"] = {"passed": True}
    else:
        selected = (
            "forward_iteration_v6_contact_abort_island_only"
            if expert == "forward"
            else "reverse_iteration_v6_absolute_full_leg_targets"
        )
        artifact[selected] = False
    with pytest.raises(RuntimeError, match="iteration-v6"):
        runner.require_iteration_v6_artifact_cross_binding(
            *artifacts[:3], artifacts[3], expert=expert
        )


@pytest.mark.parametrize("expert", ("forward", "reverse"))
@pytest.mark.parametrize("artifact_index", (0, 1, 2))
@pytest.mark.parametrize("mutation", ("path", "sha", "missing", "extra"))
def test_iteration_v6_core_source_cross_binding_near_misses_fail_closed(
    expert: str, artifact_index: int, mutation: str
) -> None:
    artifacts = list(_iteration_v6_cross_bound_artifacts(expert))
    artifact = artifacts[artifact_index]
    core_source = copy.deepcopy(artifact["iteration_v6_core_source"])
    if mutation == "path":
        core_source["path"] += ".drift"
    elif mutation == "sha":
        core_source["sha256"] = "0" * 64
    elif mutation == "missing":
        core_source.pop("path")
    else:
        core_source["extra"] = True
    artifact["iteration_v6_core_source"] = core_source
    with pytest.raises(RuntimeError, match="core source binding drifted"):
        runner.require_iteration_v6_artifact_cross_binding(
            *artifacts[:3], artifacts[3], expert=expert
        )
