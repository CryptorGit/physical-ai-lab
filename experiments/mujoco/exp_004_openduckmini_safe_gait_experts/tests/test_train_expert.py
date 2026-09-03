from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
TRAINER_PATH = EXPERIMENT_DIR / "scripts" / "train_expert.py"
SPEC = importlib.util.spec_from_file_location("exp004_train_expert", TRAINER_PATH)
assert SPEC is not None and SPEC.loader is not None
trainer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trainer
SPEC.loader.exec_module(trainer)


def test_cpu_import_and_production_parser_defaults() -> None:
    args = trainer.build_parser().parse_args(["--expert", "forward"])

    assert args.expert == "forward"
    assert args.num_timesteps == 1_000_000
    assert args.num_envs == 1_250
    assert args.backward_residual_scale == pytest.approx(0.12)
    assert args.parent_checkpoint.as_posix().endswith(
        "calibrated_hybrid_yaw_cost_v22_300m/2026_07_29_154427_10485760"
    )
    shape = trainer.resolve_training_shape(args)
    assert shape.interactions_per_training_step == 50_000
    assert shape.expected_training_steps == 20
    assert shape.expected_optimizer_updates == 1_600


def test_wiring_only_is_exactly_two_envs_and_40_interactions() -> None:
    args = trainer.build_parser().parse_args(
        ["--expert", "lateral_right", "--wiring-only"]
    )
    shape = trainer.resolve_training_shape(args)

    assert shape.num_timesteps == 40
    assert shape.num_envs == 2
    assert shape.interactions_per_training_step == 40
    assert shape.expected_training_steps == 1
    assert shape.expected_optimizer_updates == 2


def test_optional_teacher_override_arguments_are_path_resolved(tmp_path: Path) -> None:
    generated = trainer.generated_paths(tmp_path / "generated")
    generated["backward"].parent.mkdir(parents=True)
    payload = {
        "parameters": {
            "joint_amplitude_scales": [1.0] * 10,
            "phase_rate": 2.0,
        }
    }
    for key in ("backward", "backward_left", "backward_right"):
        generated[key].write_text(__import__("json").dumps(payload), encoding="utf-8")
    override = tmp_path / "optimized_reverse_exact_safe_v1.json"
    override.write_text(__import__("json").dumps(payload), encoding="utf-8")
    args = trainer.build_parser().parse_args(
        ["--expert", "reverse", "--backward-gait", str(override)]
    )

    resolved = trainer.resolve_teacher_gaits(args, generated)

    assert resolved["backward"] == override.resolve()
    assert resolved["backward_left"] == generated["backward_left"].resolve()
    assert resolved["backward_right"] == generated["backward_right"].resolve()


def test_head_action_mask_is_exact_and_non_mutating() -> None:
    action = np.linspace(-1.0, 1.0, 14)
    original = action.copy()
    masked = trainer.mask_head_action(action)

    assert np.array_equal(action, original)
    assert np.array_equal(masked[5:9], np.zeros(4))
    assert np.array_equal(masked[:5], action[:5])
    assert np.array_equal(masked[9:], action[9:])


@pytest.mark.parametrize("bad_shape", [np.zeros(13), np.zeros(15), np.zeros(())])
def test_head_action_mask_rejects_wrong_action_width(bad_shape: np.ndarray) -> None:
    with pytest.raises(ValueError):
        trainer.mask_head_action(bad_shape)


@pytest.mark.parametrize("expert", trainer.EXPERT_CHOICES)
def test_sampler_is_finite_bounded_and_head_locked(expert: str) -> None:
    commands = trainer.sample_reference_commands(expert, seed=17, count=4_000)

    assert commands.shape == (4_000, 7)
    assert np.isfinite(commands).all()
    assert np.array_equal(commands[:, 3:7], np.zeros((4_000, 4)))
    assert np.all(
        np.abs(commands[:, :3])
        <= trainer.COMMAND_MAX_ABS[np.newaxis, :] + 1e-12
    )
    moving = np.linalg.norm(commands[:, :3], axis=1) > 0.0
    assert moving.any()
    assert (~moving).any(), "stand samples must remain in every curriculum"


@pytest.mark.parametrize(
    ("expert", "axis", "sign"),
    [
        ("forward", 0, 1.0),
        ("reverse", 0, -1.0),
        ("lateral_left", 1, 1.0),
        ("lateral_right", 1, -1.0),
        ("yaw_left", 2, 1.0),
        ("yaw_right", 2, -1.0),
    ],
)
def test_axis_expert_sampler_never_crosses_its_sign_constraint(
    expert: str, axis: int, sign: float
) -> None:
    commands = trainer.sample_reference_commands(expert, seed=91, count=8_000)
    motion = commands[np.abs(commands[:, axis]) > 0.0, :3]

    assert len(motion) > 0
    assert np.all(np.sign(motion[:, axis]) == sign)
    inactive_axes = [index for index in range(3) if index != axis]
    assert np.array_equal(
        motion[:, inactive_axes], np.zeros((len(motion), len(inactive_axes)))
    )


def test_compound_sampler_covers_forward_backward_and_both_turn_signs() -> None:
    commands = trainer.sample_reference_commands("compound", seed=3, count=12_000)
    motion = commands[np.linalg.norm(commands[:, :3], axis=1) > 0.0, :3]

    assert np.all(np.count_nonzero(motion, axis=1) >= 2)
    assert np.any(motion[:, 0] > 0.0)
    assert np.any(motion[:, 0] < 0.0)
    assert np.any(motion[:, 2] > 0.0)
    assert np.any(motion[:, 2] < 0.0)
    # Sign-constrained jitter must preserve one of the declared anchor octants.
    allowed_signs = {
        tuple(np.sign(anchor)) for anchor in trainer.EXPERT_ANCHORS["compound"]
    }
    assert {tuple(np.sign(row)) for row in motion} <= allowed_signs


def test_generated_paths_can_only_name_exp004_exact_safe_artifacts() -> None:
    paths = trainer.generated_paths(Path("/tmp/generated"))

    assert paths["scene"].name == (
        "scene_flat_terrain_backlash_hardware_safe_calibrated.xml"
    )
    assert paths["reference"].name == "polynomial_coefficients_calibrated.pkl"
    assert "generated" in paths["manifest"].parent.name


def test_invalid_expert_and_invalid_training_divisibility_fail_closed() -> None:
    with pytest.raises(ValueError):
        trainer.sample_reference_commands("omnidirectional_v59", seed=0, count=1)

    args = trainer.build_parser().parse_args(
        ["--expert", "forward", "--num-timesteps", "1000001"]
    )
    with pytest.raises(ValueError):
        trainer.resolve_training_shape(args)
