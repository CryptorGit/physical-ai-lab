"""Train one hardware-safe OpenDuckMini gait expert with Brax PPO.

This entrypoint is deliberately isolated from the historical exp_003 source
tree.  It imports a frozen OpenDuck training checkout at runtime, redirects
its calibrated task constant to exp_004's generated exact-safe scene, and
reads the generated exact-safe reference from the same artifact tree.

The module keeps JAX, Brax, MuJoCo, and the external source tree behind a lazy
runtime boundary.  Pure command-sampling, argument-parsing, and head-mask
tests can therefore run on a CPU-only host without the training environment.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import importlib.metadata
import json
import os
import pickle
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from safe_gait_experts.contract import CONTRACT, CONTRACT_PATH  # noqa: E402
from safe_gait_experts.reward import bounded_axis_tracking  # noqa: E402
from safe_gait_experts.safe_randomization import (  # noqa: E402
    actuator_name_to_index,
    build_qpos_noise_scale,
    make_domain_randomizer,
)


TASK = "flat_terrain_backlash_calibrated"
EXPERT_CHOICES = (
    "forward",
    "reverse",
    "lateral_left",
    "lateral_right",
    "yaw_left",
    "yaw_right",
    "compound",
)
HEAD_ACTION_SLICE = slice(5, 9)

DEFAULT_SOURCE_ROOT = Path(
    os.environ.get(
        "OPENDUCK_EXPERT_SOURCE_ROOT",
        "/home/user/openduck_training_20260729",
    )
)
DEFAULT_GENERATED_ROOT = Path(
    os.environ.get(
        "OPENDUCK_EXPERT_GENERATED_ROOT",
        str(EXP_ROOT / "artifacts" / "generated_playground"),
    )
)
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "OPENDUCK_EXPERT_OUTPUT_ROOT",
        str(EXP_ROOT / "artifacts" / "training_runs"),
    )
)
DEFAULT_PARENT_CHECKPOINT = Path(
    os.environ.get(
        "OPENDUCK_EXPERT_PARENT_CHECKPOINT",
        "/home/user/openduck_training_runs/"
        "calibrated_hybrid_yaw_cost_v22_300m/"
        "2026_07_29_154427_10485760",
    )
)

PRODUCTION_NUM_TIMESTEPS = 1_000_000
PRODUCTION_NUM_ENVS = 1_250
PRODUCTION_UNROLL_LENGTH = 20
PRODUCTION_BATCH_SIZE = 125
PRODUCTION_NUM_MINIBATCHES = 20
PRODUCTION_NUM_UPDATES_PER_BATCH = 4
NUM_EVALS = 2
DEFAULT_SEED = 20260807
DEFAULT_BACKWARD_RESIDUAL_SCALE = 0.12

WIRING_NUM_TIMESTEPS = 40
WIRING_NUM_ENVS = 2
WIRING_UNROLL_LENGTH = 20
WIRING_BATCH_SIZE = 1
WIRING_NUM_MINIBATCHES = 2
WIRING_NUM_UPDATES_PER_BATCH = 1

SAMPLER_WEIGHTS = {
    "stand": 0.10,
    "recovery": 0.15,
    "anchor": 0.50,
    "jitter": 0.25,
}
RECOVERY_SCALE_RANGE = (0.35, 0.65)
COMMAND_MAX_ABS = np.asarray((0.15, 0.20, 1.00), dtype=np.float64)
COMMAND_MIN_ACTIVE_ABS = np.asarray((0.04, 0.04, 0.12), dtype=np.float64)
COMMAND_JITTER_STD = np.asarray((0.015, 0.015, 0.06), dtype=np.float64)

# Each compound anchor keeps longitudinal and yaw motion active.  It covers
# forward/backward left/right turns; four variants add a small lateral term.
EXPERT_ANCHORS: Mapping[str, tuple[tuple[float, float, float], ...]] = {
    "forward": ((0.10, 0.0, 0.0),),
    "reverse": ((-0.10, 0.0, 0.0),),
    "lateral_left": ((0.0, 0.10, 0.0),),
    "lateral_right": ((0.0, -0.10, 0.0),),
    "yaw_left": ((0.0, 0.0, 0.55),),
    "yaw_right": ((0.0, 0.0, -0.55),),
    "compound": (
        (0.10, 0.00, 0.35),
        (0.10, 0.00, -0.35),
        (-0.09, 0.00, 0.20),
        (-0.09, 0.00, -0.20),
        (0.08, 0.06, 0.28),
        (0.08, -0.06, -0.28),
        (-0.08, 0.05, 0.18),
        (-0.08, -0.05, -0.18),
    ),
}


@dataclass(frozen=True)
class TrainingShape:
    """Resolved interaction/optimizer shape handed to Brax PPO."""

    num_timesteps: int
    num_envs: int
    unroll_length: int
    batch_size: int
    num_minibatches: int
    num_updates_per_batch: int
    num_evals: int = NUM_EVALS

    @property
    def interactions_per_training_step(self) -> int:
        return self.batch_size * self.unroll_length * self.num_minibatches

    @property
    def expected_training_steps(self) -> int:
        return self.num_timesteps // self.interactions_per_training_step

    @property
    def expected_optimizer_updates(self) -> int:
        return (
            self.expected_training_steps
            * self.num_updates_per_batch
            * self.num_minibatches
        )


def validate_expert(expert: str) -> str:
    if expert not in EXPERT_CHOICES:
        raise ValueError(
            f"unknown expert {expert!r}; expected one of {EXPERT_CHOICES}"
        )
    return expert


def generated_paths(generated_root: Path) -> dict[str, Path]:
    package = generated_root / "playground" / "open_duck_mini_v2"
    return {
        "manifest": generated_root / "hardware_safe_manifest.json",
        "package": package,
        "scene": package
        / "xmls"
        / "scene_flat_terrain_backlash_hardware_safe_calibrated.xml",
        "reference": package
        / "data"
        / "polynomial_coefficients_calibrated.pkl",
        "backward": package / "data" / "optimized_backward_gait.json",
        "backward_left": package
        / "data"
        / "optimized_backward_left_turn_gait.json",
        "backward_right": package
        / "data"
        / "optimized_backward_right_turn_gait.json",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Hash a file or a directory tree, including relative filenames."""

    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"cannot hash empty directory: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def canonical_json_sha(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def mask_head_action(action: Any, *, xp: Any = np) -> Any:
    """Return a copy/JAX update with policy channels 5:9 fixed to zero."""

    array = xp.asarray(action)
    if array.ndim < 1 or array.shape[-1] != 14:
        raise ValueError("policy action must end with exactly 14 actuators")
    if hasattr(array, "at"):
        return array.at[..., HEAD_ACTION_SLICE].set(0.0)
    result = np.array(array, copy=True)
    result[..., HEAD_ACTION_SLICE] = 0.0
    return result


def _project_signed_numpy(candidate: np.ndarray, base: np.ndarray) -> np.ndarray:
    active = np.abs(base) > 0.0
    magnitude = np.clip(
        np.abs(candidate), COMMAND_MIN_ACTIVE_ABS, COMMAND_MAX_ABS
    )
    return np.where(active, np.sign(base) * magnitude, 0.0)


def sample_reference_commands(
    expert: str, *, seed: int, count: int
) -> np.ndarray:
    """CPU reference implementation of the fixed expert curriculum.

    This is used for contract tests and manifest previews.  Training uses an
    equivalent JAX implementation so the sampler remains jit/vmap compatible.
    """

    validate_expert(expert)
    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    anchors = np.asarray(EXPERT_ANCHORS[expert], dtype=np.float64)
    result = np.zeros((count, 7), dtype=np.float64)
    stand_edge = SAMPLER_WEIGHTS["stand"]
    recovery_edge = stand_edge + SAMPLER_WEIGHTS["recovery"]
    anchor_edge = recovery_edge + SAMPLER_WEIGHTS["anchor"]
    for index in range(count):
        base = anchors[int(rng.integers(0, len(anchors)))]
        mode = float(rng.uniform())
        if mode < stand_edge:
            command = np.zeros(3, dtype=np.float64)
        elif mode < recovery_edge:
            command = base * rng.uniform(*RECOVERY_SCALE_RANGE)
        elif mode < anchor_edge:
            command = base.copy()
        else:
            candidate = base + rng.normal(size=3) * COMMAND_JITTER_STD
            command = _project_signed_numpy(candidate, base)
        result[index, :3] = command
        # Head command channels 3:7 are intentionally left at exact zero.
    return result


def resolve_training_shape(args: argparse.Namespace) -> TrainingShape:
    if args.wiring_only:
        shape = TrainingShape(
            num_timesteps=WIRING_NUM_TIMESTEPS,
            num_envs=WIRING_NUM_ENVS,
            unroll_length=WIRING_UNROLL_LENGTH,
            batch_size=WIRING_BATCH_SIZE,
            num_minibatches=WIRING_NUM_MINIBATCHES,
            num_updates_per_batch=WIRING_NUM_UPDATES_PER_BATCH,
        )
    else:
        shape = TrainingShape(
            num_timesteps=args.num_timesteps,
            num_envs=args.num_envs,
            unroll_length=PRODUCTION_UNROLL_LENGTH,
            batch_size=PRODUCTION_BATCH_SIZE,
            num_minibatches=PRODUCTION_NUM_MINIBATCHES,
            num_updates_per_batch=PRODUCTION_NUM_UPDATES_PER_BATCH,
        )
    if shape.num_timesteps <= 0 or shape.num_envs <= 0:
        raise ValueError("num_timesteps and num_envs must be positive")
    if shape.num_timesteps % shape.interactions_per_training_step:
        raise ValueError(
            "num_timesteps must be divisible by interactions_per_training_step "
            f"({shape.interactions_per_training_step})"
        )
    return shape


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one exact-safe OpenDuckMini gait expert."
    )
    parser.add_argument("--expert", required=True, choices=EXPERT_CHOICES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--num-timesteps", type=int, default=PRODUCTION_NUM_TIMESTEPS
    )
    parser.add_argument("--num-envs", type=int, default=PRODUCTION_NUM_ENVS)
    parser.add_argument(
        "--wiring-only",
        action="store_true",
        help="Use 2 envs and exactly 40 environment interactions.",
    )
    parser.add_argument(
        "--source-root", type=Path, default=DEFAULT_SOURCE_ROOT
    )
    parser.add_argument(
        "--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument(
        "--parent-checkpoint", type=Path, default=DEFAULT_PARENT_CHECKPOINT
    )
    parser.add_argument(
        "--backward-residual-scale",
        type=float,
        default=DEFAULT_BACKWARD_RESIDUAL_SCALE,
        help=(
            "Policy residual in radians around the periodic reverse teacher. "
            "The frozen shallow source defaults to zero, so this is explicit."
        ),
    )
    parser.add_argument(
        "--backward-gait",
        type=Path,
        help=(
            "Optional straight-reverse teacher override. The generated legacy "
            "profile is still hash-validated even when this is supplied."
        ),
    )
    parser.add_argument(
        "--backward-left-gait",
        type=Path,
        help="Optional reverse-left-turn teacher override.",
    )
    parser.add_argument(
        "--backward-right-gait",
        type=Path,
        help="Optional reverse-right-turn teacher override.",
    )
    parser.add_argument(
        "--run-name",
        help=(
            "Immutable output directory name. Defaults to a deterministic "
            "expert/seed/interaction label; an existing directory is refused."
        ),
    )
    return parser


def _default_run_name(
    expert: str, seed: int, shape: TrainingShape, wiring_only: bool
) -> str:
    prefix = "wiring" if wiring_only else "pilot"
    return f"{prefix}_{expert}_seed{seed}_{shape.num_timesteps}"


def _validate_generated_manifest(paths: Mapping[str, Path]) -> dict[str, Any]:
    for label, path in paths.items():
        if label != "package" and not path.is_file():
            raise FileNotFoundError(f"missing generated {label}: {path}")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    expected = {
        "generated_scene": paths["scene"],
        "generated_reference": paths["reference"],
        "legacy_v22_optimized_backward_gait": paths["backward"],
        "legacy_v22_optimized_backward_left_turn_gait": paths["backward_left"],
        "legacy_v22_optimized_backward_right_turn_gait": paths["backward_right"],
    }
    for manifest_key, path in expected.items():
        recorded = manifest["files"][manifest_key]["sha256"]
        actual = sha256_file(path)
        if actual != recorded:
            raise ValueError(
                f"generated artifact hash mismatch for {manifest_key}: "
                f"recorded={recorded}, actual={actual}"
            )
    if manifest.get("real_hardware_deployment_allowed") is not False:
        raise ValueError("generated manifest must prohibit hardware deployment")
    return manifest


def _load_generated_gait(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))["parameters"]
    scales = payload["joint_amplitude_scales"]
    biases = payload.get("joint_bias_offsets", [0.0] * 10)
    if len(scales) != 10 or len(biases) != 10:
        raise ValueError(f"gait profile must contain ten leg values: {path}")
    return {
        "scales": tuple(float(value) for value in scales),
        "biases": tuple(float(value) for value in biases),
        "phase_rate": float(payload["phase_rate"]),
    }


def resolve_teacher_gaits(
    args: argparse.Namespace, generated: Mapping[str, Path]
) -> dict[str, Path]:
    """Resolve optional teacher overrides without weakening generated checks."""

    requested = {
        "backward": args.backward_gait,
        "backward_left": args.backward_left_gait,
        "backward_right": args.backward_right_gait,
    }
    result: dict[str, Path] = {}
    for name, override in requested.items():
        path = (override if override is not None else generated[name]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing {name} teacher gait: {path}")
        # Parse and validate the schema before allocating an MJX environment.
        _load_generated_gait(path)
        result[name] = path
    return result


def _runtime_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("jax", "jaxlib", "brax", "mujoco", "mujoco-mjx"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _load_training_stack(source_root: Path) -> dict[str, Any]:
    """Import the WSL/GPU training stack only when a run is requested."""

    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"OpenDuck source root does not exist: {source_root}")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    import jax
    import jax.numpy as jp
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper
    from mujoco_playground.config import locomotion_params
    from playground.open_duck_mini_v2 import constants, joystick

    resolved_joystick = Path(joystick.__file__).resolve()
    if source_root not in resolved_joystick.parents:
        raise RuntimeError(
            "playground.open_duck_mini_v2 resolved outside --source-root: "
            f"{resolved_joystick}"
        )
    return {
        "jax": jax,
        "jp": jp,
        "ppo_networks": ppo_networks,
        "ppo": ppo,
        "wrapper": wrapper,
        "locomotion_params": locomotion_params,
        "constants": constants,
        "joystick": joystick,
    }


def _make_jax_sampler(jax: Any, jp: Any, expert: str):
    validate_expert(expert)
    anchors = jp.asarray(EXPERT_ANCHORS[expert])
    max_abs = jp.asarray(COMMAND_MAX_ABS)
    min_abs = jp.asarray(COMMAND_MIN_ACTIVE_ABS)
    jitter_std = jp.asarray(COMMAND_JITTER_STD)
    stand_edge = SAMPLER_WEIGHTS["stand"]
    recovery_edge = stand_edge + SAMPLER_WEIGHTS["recovery"]
    anchor_edge = recovery_edge + SAMPLER_WEIGHTS["anchor"]

    def sample(rng: Any) -> Any:
        mode_key, anchor_key, recovery_key, jitter_key = jax.random.split(rng, 4)
        anchor_index = jax.random.randint(
            anchor_key, shape=(), minval=0, maxval=anchors.shape[0]
        )
        base = anchors[anchor_index]
        mode = jax.random.uniform(mode_key)
        recovery_scale = jax.random.uniform(
            recovery_key,
            minval=RECOVERY_SCALE_RANGE[0],
            maxval=RECOVERY_SCALE_RANGE[1],
        )
        candidate = base + jax.random.normal(jitter_key, shape=(3,)) * jitter_std
        active = jp.abs(base) > 0.0
        jittered = jp.where(
            active,
            jp.sign(base) * jp.clip(jp.abs(candidate), min_abs, max_abs),
            0.0,
        )
        locomotion = jp.where(
            mode < stand_edge,
            jp.zeros(3),
            jp.where(
                mode < recovery_edge,
                base * recovery_scale,
                jp.where(mode < anchor_edge, base, jittered),
            ),
        )
        return jp.concatenate((locomotion, jp.zeros(4)))

    return sample


def _make_environment_class(
    *,
    stack: Mapping[str, Any],
    expert: str,
    paths: Mapping[str, Path],
    teacher_gaits: Mapping[str, Path],
    backward_residual_scale: float,
):
    """Build the source-compatible Joystick subclass for one fixed expert."""

    jax = stack["jax"]
    jp = stack["jp"]
    joystick = stack["joystick"]
    sampler = _make_jax_sampler(jax, jp, expert)
    gait = _load_generated_gait(teacher_gaits["backward"])
    left_gait = _load_generated_gait(teacher_gaits["backward_left"])
    right_gait = _load_generated_gait(teacher_gaits["backward_right"])
    tracking = CONTRACT["tracking_reward"]
    sigma = tuple(float(value) for value in tracking["sigma"])

    class SafeGaitExpertJoystick(joystick.Joystick):
        """Exact-safe calibrated task with one sign-constrained curriculum."""

        expert_name = expert

        def __init__(self):
            # PolyReferenceMotion in the frozen source resolves a historical
            # relative path.  Point that path at exp_004's immutable generated
            # playground instead of either mutable source checkout.
            os.chdir(paths["manifest"].parent)
            super().__init__(task=TASK)
            scales = self._config.reward_config.scales
            # Every progress/error movement objective is disabled.  The only
            # command-tracking objective left is the bounded Gaussian pair
            # installed in _get_reward below.
            for name in (
                "command_progress",
                "command_velocity_error",
                "command_yaw_error",
                "backward_progress",
                "backward_velocity_error",
                "backward_yaw_error",
                "backward_lateral_error",
            ):
                if name in scales:
                    scales[name] = 0.0

        def _post_init(self):
            super()._post_init()
            # Replace any mutable source-root reverse profiles with the exact
            # profiles copied and hashed beside the generated scene/reference.
            self._backward_gait_scales = jp.asarray(gait["scales"])
            self._backward_gait_biases = jp.asarray(gait["biases"])
            self._backward_phase_rate = gait["phase_rate"]
            # The frozen shallow source initializes this to 0.0.  Leaving it
            # unchanged makes every reverse policy action causally inert.
            self._backward_residual_scale = float(backward_residual_scale)
            self._backward_left_gait_scales = jp.asarray(left_gait["scales"])
            self._backward_left_gait_biases = jp.asarray(left_gait["biases"])
            self._backward_left_phase_rate = left_gait["phase_rate"]
            self._backward_right_gait_scales = jp.asarray(right_gait["scales"])
            self._backward_right_gait_biases = jp.asarray(right_gait["biases"])
            self._backward_right_phase_rate = right_gait["phase_rate"]

            actuator_indices = actuator_name_to_index(self._mj_model)
            noise = CONTRACT["qpos_noise_scale_rad"]
            qpos_noise = build_qpos_noise_scale(
                actuator_indices,
                hip_scale=float(noise["hip"]),
                knee_scale=float(noise["knee"]),
                ankle_scale=float(noise["ankle"]),
            )
            self._qpos_noise_scale = jp.asarray(qpos_noise)
            default_actuator = np.asarray(self._default_actuator)
            if not np.array_equal(default_actuator[HEAD_ACTION_SLICE], np.zeros(4)):
                raise ValueError("generated scene home must lock all head targets at zero")
            if not np.array_equal(qpos_noise[HEAD_ACTION_SLICE], np.zeros(4)):
                raise ValueError("head reset noise must remain exactly zero")

        def sample_command(self, rng):
            return sampler(rng)

        def step(self, state, action):
            return super().step(state, mask_head_action(action, xp=jp))

        def _get_reward(self, *args, **kwargs):
            rewards = super()._get_reward(*args, **kwargs)
            data = args[0]
            info = args[2]
            actual = jp.asarray(
                (
                    self.get_local_linvel(data)[0],
                    self.get_local_linvel(data)[1],
                    self.get_gyro(data)[2],
                )
            )
            axis_tracking = bounded_axis_tracking(
                info["command"][:3], actual, sigma=sigma, xp=jp
            )
            rewards["tracking_lin_vel"] = jp.mean(axis_tracking[:2])
            rewards["tracking_ang_vel"] = axis_tracking[2]
            for name in (
                "command_progress",
                "command_velocity_error",
                "command_yaw_error",
                "backward_progress",
                "backward_velocity_error",
                "backward_yaw_error",
                "backward_lateral_error",
            ):
                if name in rewards:
                    rewards[name] = jp.zeros(())
            return rewards

    SafeGaitExpertJoystick.__name__ = (
        "Safe" + "".join(part.title() for part in expert.split("_")) + "Joystick"
    )
    return SafeGaitExpertJoystick


def _save_params(path: Path, jax: Any, params: Any) -> None:
    # Save only at the end of the uninterrupted pilot.  Per-update device_get
    # and GPU telemetry are intentionally absent due the known WSL/MJX fault.
    host_params = jax.tree_util.tree_map(np.asarray, params)
    with path.open("wb") as stream:
        pickle.dump(host_params, stream, protocol=pickle.HIGHEST_PROTOCOL)


def _scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in metrics.items():
        array = np.asarray(value)
        if array.size == 1:
            result[key] = float(array.reshape(()))
    return result


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    expert = validate_expert(args.expert)
    shape = resolve_training_shape(args)
    source_root = args.source_root.resolve()
    generated_root = args.generated_root.resolve()
    output_root = args.output_root.resolve()
    parent_checkpoint = args.parent_checkpoint.resolve()
    if not 0.0 < args.backward_residual_scale <= 0.25:
        raise ValueError("--backward-residual-scale must be in (0, 0.25]")
    paths = generated_paths(generated_root)
    _validate_generated_manifest(paths)
    teacher_gaits = resolve_teacher_gaits(args, paths)
    if not parent_checkpoint.exists():
        raise FileNotFoundError(
            f"v22 parent checkpoint does not exist: {parent_checkpoint}"
        )

    run_name = args.run_name or _default_run_name(
        expert, args.seed, shape, args.wiring_only
    )
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("--run-name must be one non-empty directory name")
    run_dir = output_root / expert / run_name
    if run_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing immutable run: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=False)

    stack = _load_training_stack(source_root)
    constants = stack["constants"]
    scene_type = type(constants.FLAT_TERRAIN_BACKLASH_CALIBRATED_XML)
    # The source base loader collects include files/meshes from ROOT_PATH.
    # Redirect that asset root together with the task constant so the scene's
    # sibling model include cannot silently resolve back into the source tree.
    constants.ROOT_PATH = scene_type(paths["package"].as_posix())
    constants.FLAT_TERRAIN_BACKLASH_CALIBRATED_XML = scene_type(
        paths["scene"].as_posix()
    )
    resolved_scene = Path(constants.task_to_xml(TASK).as_posix()).resolve()
    if resolved_scene != paths["scene"].resolve():
        raise RuntimeError(
            "failed to redirect calibrated scene constant: "
            f"resolved={resolved_scene}, expected={paths['scene']}"
        )

    Environment = _make_environment_class(
        stack=stack,
        expert=expert,
        paths=paths,
        teacher_gaits=teacher_gaits,
        backward_residual_scale=args.backward_residual_scale,
    )
    env = Environment()
    eval_env = Environment()
    for label, instance in (("train", env), ("eval", eval_env)):
        actual_residual = float(instance._backward_residual_scale)
        if not np.isclose(actual_residual, args.backward_residual_scale):
            raise RuntimeError(
                f"{label} reverse residual wiring mismatch: "
                f"{actual_residual} != {args.backward_residual_scale}"
            )
    randomizer = make_domain_randomizer(env.mj_model)

    locomotion_params = stack["locomotion_params"]
    ppo_networks = stack["ppo_networks"]
    ppo_config = locomotion_params.brax_ppo_config(
        "BerkeleyHumanoidJoystickFlatTerrain"
    )
    network_config = ppo_config.network_factory.to_dict()
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **network_config
    )
    training = ppo_config.to_dict()
    training.pop("network_factory")
    training.update(
        {
            **asdict(shape),
            "seed": args.seed,
            "run_evals": False,
            "log_training_metrics": True,
            "restore_checkpoint_path": str(parent_checkpoint),
        }
    )

    resolved_config = {
        "schema_version": 1,
        "expert": expert,
        "task": TASK,
        "seed": args.seed,
        "wiring_only": bool(args.wiring_only),
        "shape": asdict(shape),
        "interactions_per_training_step": shape.interactions_per_training_step,
        "expected_training_steps": shape.expected_training_steps,
        "expected_optimizer_updates": shape.expected_optimizer_updates,
        "run_evals": False,
        "gpu_telemetry_during_updates": False,
        "source_root": str(source_root),
        "generated_root": str(generated_root),
        "scene": str(paths["scene"]),
        "reference": str(paths["reference"]),
        "teacher_gaits": {
            name: str(path) for name, path in teacher_gaits.items()
        },
        "teacher_gait_overrides": {
            "backward": args.backward_gait is not None,
            "backward_left": args.backward_left_gait is not None,
            "backward_right": args.backward_right_gait is not None,
        },
        "parent_checkpoint": str(parent_checkpoint),
        "output_dir": str(run_dir),
        "network_factory": network_config,
        "ppo": training,
        "sampler": {
            "weights": SAMPLER_WEIGHTS,
            "recovery_scale_range": RECOVERY_SCALE_RANGE,
            "anchors": EXPERT_ANCHORS[expert],
            "head_command": [0.0, 0.0, 0.0, 0.0],
        },
        "movement_objective": {
            "bounded_axis_tracking": CONTRACT["tracking_reward"],
            "command_progress_scale": 0.0,
            "unbounded_dot_product_enabled": False,
        },
        "backward_residual_scale": args.backward_residual_scale,
        "head_action_mask_indices": [5, 6, 7, 8],
        "domain_randomizer": "safe_gait_experts.make_domain_randomizer(name-resolved)",
    }
    config_path = run_dir / "resolved_config.json"
    config_path.write_text(
        json.dumps(resolved_config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    source_joystick = Path(stack["joystick"].__file__).resolve()
    source_constants = Path(stack["constants"].__file__).resolve()
    manifest = {
        "schema_version": 1,
        "status": "STARTED",
        "hardware_deployment": "PROHIBITED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "expert": expert,
        "run_name": run_name,
        "seed": args.seed,
        "inputs": {
            "contract": {
                "path": str(CONTRACT_PATH),
                "sha256": sha256_file(CONTRACT_PATH),
            },
            "generated_manifest": {
                "path": str(paths["manifest"]),
                "sha256": sha256_file(paths["manifest"]),
            },
            "scene": {
                "path": str(paths["scene"]),
                "sha256": sha256_file(paths["scene"]),
            },
            "reference": {
                "path": str(paths["reference"]),
                "sha256": sha256_file(paths["reference"]),
            },
            "generated_legacy_backward_gaits": {
                key: {
                    "path": str(paths[key]),
                    "sha256": sha256_file(paths[key]),
                }
                for key in ("backward", "backward_left", "backward_right")
            },
            "teacher_gaits": {
                key: {
                    "path": str(teacher_gaits[key]),
                    "sha256": sha256_file(teacher_gaits[key]),
                    "override": teacher_gaits[key] != paths[key].resolve(),
                }
                for key in ("backward", "backward_left", "backward_right")
            },
            "parent_checkpoint": {
                "path": str(parent_checkpoint),
                "sha256_tree": sha256_tree(parent_checkpoint),
            },
            "source_joystick": {
                "path": str(source_joystick),
                "sha256": sha256_file(source_joystick),
            },
            "source_constants": {
                "path": str(source_constants),
                "sha256": sha256_file(source_constants),
            },
            "trainer": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "resolved_config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "canonical_sha256": canonical_json_sha(resolved_config),
        },
        "versions": _runtime_versions(),
        "jax_devices": [str(device) for device in stack["jax"].devices()],
        "notes": [
            "Simulation research artifact only; not cleared for hardware.",
            "Training is uninterrupted and performs no per-update GPU telemetry.",
            "All four head command/action channels are fixed to exact zero.",
        ],
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    curve_path = run_dir / "training_curve.csv"
    curve_rows: list[dict[str, Any]] = []

    def progress(step: int, metrics: Mapping[str, Any]) -> None:
        # Brax calls this only at the configured host boundary (not per PPO
        # update).  It records scalar training metrics without GPU polling.
        row: dict[str, Any] = {"environment_interactions": int(step)}
        row.update(_scalar_metrics(metrics))
        curve_rows.append(row)
        fields = sorted({key for item in curve_rows for key in item})
        with curve_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(curve_rows)

    make_policy, params, metrics = stack["ppo"].train(
        environment=env,
        eval_env=eval_env,
        network_factory=network_factory,
        randomization_fn=randomizer,
        wrap_env_fn=stack["wrapper"].wrap_for_brax_training,
        progress_fn=progress,
        **training,
    )
    del make_policy
    params_path = run_dir / "final_params.pkl"
    _save_params(params_path, stack["jax"], params)
    result = {
        "status": "COMPLETED",
        "hardware_deployment": "PROHIBITED",
        "expert": expert,
        "requested_environment_interactions": shape.num_timesteps,
        "expected_optimizer_updates": shape.expected_optimizer_updates,
        "final_metrics": _scalar_metrics(metrics),
        "params_leaf_count": len(stack["jax"].tree_util.tree_leaves(params)),
        "final_params": {
            "path": str(params_path),
            "sha256": sha256_file(params_path),
        },
        "resolved_config_sha256": sha256_file(config_path),
    }
    result_path = run_dir / "run_result.json"
    result_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    manifest["status"] = "COMPLETED"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["outputs"] = {
        "params": result["final_params"],
        "result": {
            "path": str(result_path),
            "sha256": sha256_file(result_path),
        },
        "training_curve": (
            {"path": str(curve_path), "sha256": sha256_file(curve_path)}
            if curve_path.exists()
            else None
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    args = build_parser().parse_args(argv)
    result = run_training(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
