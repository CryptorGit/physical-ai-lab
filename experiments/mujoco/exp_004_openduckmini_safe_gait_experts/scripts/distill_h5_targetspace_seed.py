"""Build a simulation-only H5 reverse actor seed in absolute target space.

The H5 actor emits bounded actions which are decoded directly into absolute
joint targets.  A legacy 101-wide actor transplant therefore has the wrong
output semantics even when its network topology is compatible.  This script
records observations from the exact H5 simulator and fits only the 14 policy
mean channels of ``hidden_3`` to a causal target-space seed:

    clip_margin(T_teacher(2*phase) + gain*(D(a_v7)-SAFE_INIT))

The evaluator and the PPO trainer remain unchanged by this diagnostic
artifact.  Hardware use and adoption are explicitly prohibited.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from safe_gait_experts.contract import (  # noqa: E402
    ACTUATOR_JOINT_ORDER,
    SAFE_INIT_POS,
    SAFE_JOINT_LIMITS,
)
from safe_gait_experts.h4_post_training import (  # noqa: E402
    H4_ACTOR_OBSERVATION_WIDTH,
    H4_ACTION_WIDTH,
    infer_h4_action_numpy,
    mask_h4_head_action,
    sha256_file,
    validate_h4_params,
)
from safe_gait_experts.h4_training_alignment import (  # noqa: E402
    contract_target_vectors,
    margin_clip_targets,
    reverse_iteration_v6_absolute_full_leg_targets,
)
from safe_gait_experts.h5_target_contract import (  # noqa: E402
    h5_decode_absolute_targets,
    h5_domain_for_route,
)
from safe_gait_experts.h5_command_contract import (  # noqa: E402
    H5_UNIFIED_PHYSICAL_COMMANDS,
    H5_UNIFIED_ROUTE_NAMES,
)
from scripts import evaluate_h5_routed_transitions as h5_evaluator  # noqa: E402
from scripts.train_h4_aligned_expert import (  # noqa: E402
    interpolate_periodic_table,
    load_selected_reverse_teacher,
)


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _sha(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("SHA256 must be lowercase hex")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", action="append", required=True, metavar="ROLE=PATH")
    parser.add_argument("--generated-root", type=_path, required=True)
    parser.add_argument("--reverse-params", type=_path, required=True)
    parser.add_argument("--reverse-params-sha256", type=_sha, required=True)
    parser.add_argument("--reverse-manifest", type=_path, required=True)
    parser.add_argument("--reverse-manifest-sha256", type=_sha, required=True)
    parser.add_argument("--rollout-params", type=_path)
    parser.add_argument("--rollout-params-sha256", type=_sha)
    parser.add_argument("--rollout-manifest", type=_path)
    parser.add_argument("--rollout-manifest-sha256", type=_sha)
    parser.add_argument("--planar-params", type=_path, required=True)
    parser.add_argument("--planar-params-sha256", type=_sha, required=True)
    parser.add_argument("--planar-manifest", type=_path, required=True)
    parser.add_argument("--planar-manifest-sha256", type=_sha, required=True)
    parser.add_argument("--selected-teacher", type=_path)
    parser.add_argument(
        "--teacher-mode",
        choices=(
            "selected",
            "calibrated_profile",
            "h5_profile",
            "h3_profile_residual",
        ),
        default="selected",
        help=(
            "Use the bounded H4 training prior or the calibrated exp003 "
            "reference plus optimized reverse gait profile as the temporary "
            "data-generation teacher."
        ),
    )
    parser.add_argument("--calibrated-reference", type=_path)
    parser.add_argument("--calibrated-gait-profile", type=_path)
    parser.add_argument(
        "--teacher-phase-rate",
        type=float,
        default=2.0,
        help=(
            "Map H5 table index q to the calibrated 27-frame phase as "
            "q/2 * rate + offset; 2.0 reproduces the historical T(2*phase)."
        ),
    )
    parser.add_argument(
        "--teacher-phase-offset",
        type=float,
        default=0.0,
        help="Add a periodic calibrated-profile phase offset in frame units.",
    )
    parser.add_argument("--output-params", type=_path, required=True)
    parser.add_argument("--output-manifest", type=_path, required=True)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--rollouts", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument(
        "--rollout-joint-noise-scale",
        type=float,
        default=0.0,
        help=(
            "Initial joint perturbation used while collecting teacher states. "
            "This is diagnostic data generation, not a deployment setting."
        ),
    )
    parser.add_argument(
        "--rollout-initial-base-speed",
        type=float,
        default=0.0,
        help=(
            "Initial planar base-speed perturbation used while collecting "
            "teacher states."
        ),
    )
    parser.add_argument(
        "--rollout-warmup-seconds",
        type=float,
        default=0.0,
        help="Warmup duration for each diagnostic teacher rollout.",
    )
    parser.add_argument("--residual-gain", type=float, default=0.90)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument(
        "--semantic-reset",
        action="store_true",
        help=(
            "Fit hidden_3 mean channels from a zero semantic prior and lock "
            "the four head mean channels to exact zero. This is the required "
            "initialization for absolute H5 target semantics."
        ),
    )
    parser.add_argument(
        "--unified-collection",
        action="store_true",
        help=(
            "Collect all thirteen unified physical command routes in each "
            "rollout. Non-reverse targets are preserved from the current H5 "
            "actor while reverse targets use the selected diagnostic teacher."
        ),
    )
    parser.add_argument(
        "--dagger-collection",
        action="store_true",
        help=(
            "Run the current candidate actor closed-loop while recording the "
            "same teacher labels. The teacher target is not applied during "
            "collection; use with --unified-collection for DAgger." 
        ),
    )
    parser.add_argument(
        "--fit-all-actor-layers",
        action="store_true",
        help=(
            "Run the bounded full actor-location fine-tune even when the "
            "output-head fit already meets the seed fidelity gate."
        ),
    )
    return parser


def _silu(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    positive = values >= 0.0
    result = np.empty_like(values)
    result[positive] = values[positive] / (
        np.float32(1.0) + np.exp(-values[positive]).astype(np.float32)
    )
    exp_values = np.exp(values[~positive]).astype(np.float32)
    result[~positive] = values[~positive] * exp_values / (
        np.float32(1.0) + exp_values
    )
    return result


def _hidden_3_features(params: Any, observations: np.ndarray) -> np.ndarray:
    """Return the input to the actor's final 128->28 layer."""

    normalizer, actor, _critic = params
    values = np.asarray(observations, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != H4_ACTOR_OBSERVATION_WIDTH:
        raise ValueError("observations must be a batch of 116-wide vectors")
    hidden = (values - np.asarray(normalizer.mean["state"], dtype=np.float32)) / np.asarray(
        normalizer.std["state"], dtype=np.float32
    )
    layers = actor["params"]
    for index in range(3):
        layer = layers[f"hidden_{index}"]
        hidden = (
            hidden @ np.asarray(layer["kernel"], dtype=np.float32)
            + np.asarray(layer["bias"], dtype=np.float32)
        ).astype(np.float32)
        hidden = _silu(hidden)
    if hidden.shape[1] != 128 or not np.all(np.isfinite(hidden)):
        raise RuntimeError("hidden_3 feature extraction failed")
    return hidden


def _decoder_reachable_target(target: np.ndarray) -> np.ndarray:
    """Project a target into the exact range of the H5 bounded decoder."""

    lower, upper, initial = contract_target_vectors(xp=np)
    values = np.asarray(target, dtype=np.float64).copy()
    positive_span = 0.90 * (np.asarray(upper) - np.asarray(initial))
    negative_span = 0.90 * (np.asarray(initial) - np.asarray(lower))
    values = np.minimum(values, np.asarray(initial) + positive_span)
    values = np.maximum(values, np.asarray(initial) - negative_span)
    values[5:9] = 0.0
    return values


def _inverse_decoder(target: np.ndarray) -> np.ndarray:
    """Invert the official linear-plus-quintic H5 decoder."""

    lower, upper, initial = contract_target_vectors(xp=np)
    values = np.asarray(target, dtype=np.float64)
    if values.shape != (H4_ACTION_WIDTH,) or not np.all(np.isfinite(values)):
        raise ValueError("target must be finite and 14-wide")
    action = np.zeros(H4_ACTION_WIDTH, dtype=np.float32)
    for index in range(H4_ACTION_WIDTH):
        if 5 <= index <= 8:
            continue
        delta = float(values[index] - initial[index])
        span = float(
            0.90 * (upper[index] - initial[index])
            if delta >= 0.0
            else 0.90 * (initial[index] - lower[index])
        )
        magnitude = abs(delta)
        base = min(0.25, span)
        if magnitude <= 0.0:
            normalized = 0.0
        elif magnitude >= span:
            normalized = 1.0
        else:
            lo, hi = 0.0, 1.0
            for _ in range(70):
                mid = 0.5 * (lo + hi)
                candidate = base * mid + (span - base) * mid**5
                if candidate < magnitude:
                    lo = mid
                else:
                    hi = mid
            normalized = 0.5 * (lo + hi)
        action[index] = np.float32(np.sign(delta) * normalized)
    return action


def _phase_from_observation(observation: np.ndarray, phase_steps: float) -> float:
    angle = float(np.arctan2(float(observation[100]), float(observation[99])))
    if angle < 0.0:
        angle += 2.0 * np.pi
    return angle / (2.0 * np.pi) * float(phase_steps)


def _calibrated_profile_teacher_table(
    reference_path: Path,
    gait_path: Path,
    *,
    teacher_phase_rate: float = 2.0,
    teacher_phase_offset: float = 0.0,
) -> np.ndarray:
    """Return a 54-row H5 table from the exp003 calibrated 27-frame gait.

    The calibrated reference is a polynomial table with a 27 control-frame
    period. H5 exposes a 54-row lookup domain and evaluates it at ``2 * phase``.
    ``teacher_phase_rate`` and ``teacher_phase_offset`` let the table preserve
    the measured H3 profile timing under that H5 lookup, instead of silently
    reverting to the unwarped one-times gait.
    """

    with reference_path.open("rb") as stream:
        reference = pickle.load(stream)
    key = min(
        reference,
        key=lambda name: np.linalg.norm(
            np.asarray([float(value) for value in name.split("_")], dtype=np.float64)
            - np.asarray([-0.1, 0.0, 0.0], dtype=np.float64)
        ),
    )
    motion = reference[key]
    period = int(round(float(motion["period"]) * float(motion["fps"])))
    coefficients = list(motion["coefficients"].values())
    frames = np.asarray(
        [
            [
                np.polyval(np.flip(np.asarray(coefficient)), index / period)
                for coefficient in coefficients
            ]
            for index in range(period)
        ],
        dtype=np.float64,
    )
    if frames.shape[1] < 16 or period != 27:
        raise ValueError(
            f"calibrated reverse reference drifted: period={period}, shape={frames.shape}"
        )
    reference_leg_indices = np.asarray([0, 1, 2, 3, 4, 11, 12, 13, 14, 15])
    leg_frames = frames[:, reference_leg_indices]
    leg_means = leg_frames.mean(axis=0)
    leg_deviations = leg_frames - leg_means
    gait = json.loads(gait_path.read_text(encoding="utf-8"))
    parameters = gait["parameters"]
    scales = np.asarray(parameters["joint_amplitude_scales"], dtype=np.float64)
    biases = np.asarray(
        parameters.get("joint_bias_offsets", [0.0] * 10), dtype=np.float64
    )
    if scales.shape != (10,) or biases.shape != (10,):
        raise ValueError("calibrated reverse gait profile must contain ten leg values")
    leg_targets = leg_means + biases + scales * leg_deviations
    table27 = np.zeros((period, H4_ACTION_WIDTH), dtype=np.float64)
    table27[:, np.asarray([0, 1, 2, 3, 4, 9, 10, 11, 12, 13])] = leg_targets
    lower, upper, _initial = contract_target_vectors(xp=np)
    table27 = np.clip(table27, np.asarray(lower), np.asarray(upper))
    # The runtime H5 guard owns the inward 0.050-rad margin.  Bake the same
    # margin into the teacher table so the BC target and the applied target
    # have one authority and the distilled seed is not trained toward values
    # that the final guard must later truncate.
    table27 = np.clip(
        table27,
        np.asarray(lower, dtype=np.float64) + 0.050,
        np.asarray(upper, dtype=np.float64) - 0.050,
    )
    table27[:, 5:9] = 0.0
    if (
        not np.isfinite(float(teacher_phase_rate))
        or float(teacher_phase_rate) <= 0.0
        or not np.isfinite(float(teacher_phase_offset))
    ):
        raise ValueError("teacher phase rate must be positive and phase offset finite")
    table54 = np.empty((period * 2, H4_ACTION_WIDTH), dtype=np.float64)
    for table_index in range(table54.shape[0]):
        source_phase = (
            (float(table_index) / 2.0) * float(teacher_phase_rate)
            + float(teacher_phase_offset)
        ) % float(period)
        source_index = int(np.floor(source_phase))
        fraction = source_phase - float(source_index)
        table54[table_index] = (
            (1.0 - fraction) * table27[source_index]
            + fraction * table27[(source_index + 1) % period]
        )
    table54[:, 5:9] = 0.0
    if not np.all(np.isfinite(table54)):
        raise ValueError("calibrated profile teacher table is non-finite")
    return table54


def _h5_profile_teacher_table(
    simulator: Any,
    gait_profile_path: Path,
    *,
    teacher_phase_rate: float,
    teacher_phase_offset: float,
) -> np.ndarray:
    """Build a table from the exact H5 evaluator backward target generator.

    The calibrated polynomial helper is useful as a portable seed source, but
    it is not identical to the runtime's loaded backward profile statistics.
    This path samples the same ``OfficialPolicyEvaluator`` method used by the
    target-program probe, then applies the standard inward margin before the
    table becomes a BC label.
    """

    if (
        not np.isfinite(float(teacher_phase_rate))
        or float(teacher_phase_rate) <= 0.0
        or not np.isfinite(float(teacher_phase_offset))
    ):
        raise ValueError("teacher phase rate must be positive and phase offset finite")
    simulator.evaluator._evaluator.load_backward_profile(gait_profile_path.resolve())
    scales, biases, _ = simulator.evaluator.backward_parameters(0.0)
    initial = np.asarray(
        [float(SAFE_INIT_POS[name]) for name in ACTUATOR_JOINT_ORDER],
        dtype=np.float64,
    )
    joint_ranges = np.asarray(
        [
            [0.0, 0.0]
            if name in {"neck_pitch", "head_pitch", "head_yaw", "head_roll"}
            else SAFE_JOINT_LIMITS[name]
            for name in ACTUATOR_JOINT_ORDER
        ],
        dtype=np.float64,
    )
    table = np.empty((54, H4_ACTION_WIDTH), dtype=np.float64)
    for table_index in range(table.shape[0]):
        phase = (
            (float(table_index) / 2.0) * float(teacher_phase_rate)
            + float(teacher_phase_offset)
        ) % float(simulator.evaluator.phase_steps)
        target = simulator.evaluator._backward_feedforward(
            phase,
            initial,
            joint_ranges,
            np.zeros(H4_ACTION_WIDTH, dtype=np.float64),
            gait_scales=scales,
            gait_biases=biases,
            leg_residual_factor=0.0,
            head_residual_factor=0.0,
        )
        target[5:9] = 0.0
        table[table_index] = np.asarray(
            margin_clip_targets(target, xp=np), dtype=np.float64
        )
    if not np.all(np.isfinite(table)) or not np.all(table[:, 5:9] == 0.0):
        raise ValueError("H5 profile teacher table is not finite or head-locked")
    return table


def _h5_profile_teacher_target(
    simulator: Any,
    phase: float,
    yaw_command: float,
) -> np.ndarray:
    """Evaluate the loaded calibrated profile for a straight/turn reverse command."""

    scales, biases, _ = simulator.evaluator.backward_parameters(float(yaw_command))
    initial = np.asarray(
        [float(SAFE_INIT_POS[name]) for name in ACTUATOR_JOINT_ORDER],
        dtype=np.float64,
    )
    joint_ranges = np.asarray(
        [
            [0.0, 0.0]
            if name in {"neck_pitch", "head_pitch", "head_yaw", "head_roll"}
            else SAFE_JOINT_LIMITS[name]
            for name in ACTUATOR_JOINT_ORDER
        ],
        dtype=np.float64,
    )
    target = simulator.evaluator._evaluator._backward_feedforward(
        float(phase),
        initial,
        joint_ranges,
        np.zeros(H4_ACTION_WIDTH, dtype=np.float64),
        gait_scales=scales,
        gait_biases=biases,
        leg_residual_factor=0.0,
        head_residual_factor=0.0,
    )
    target[5:9] = 0.0
    return np.asarray(margin_clip_targets(target, xp=np), dtype=np.float64)


def _make_h5_args(args: argparse.Namespace) -> SimpleNamespace:
    rollout_params = args.rollout_params or args.reverse_params
    rollout_params_sha256 = args.rollout_params_sha256 or args.reverse_params_sha256
    rollout_manifest = args.rollout_manifest or args.reverse_manifest
    rollout_manifest_sha256 = args.rollout_manifest_sha256 or args.reverse_manifest_sha256
    return SimpleNamespace(
        policy=list(args.policy),
        generated_root=args.generated_root,
        # The distillation rollout is intentionally a reverse-domain
        # diagnostic, not a unified same-weight evaluation.  Keep the
        # evaluator's explicit mode flag present so this helper remains
        # compatible with the strict H5 evaluator contract.
        unified_single_weight=bool(args.unified_collection),
        strict_actor_only=False,
        h5_planar_params=args.planar_params,
        h5_planar_params_sha256=args.planar_params_sha256,
        h5_planar_manifest=args.planar_manifest,
        h5_planar_manifest_sha256=args.planar_manifest_sha256,
        h5_reverse_params=rollout_params,
        h5_reverse_params_sha256=rollout_params_sha256,
        h5_reverse_manifest=rollout_manifest,
        h5_reverse_manifest_sha256=rollout_manifest_sha256,
    )


def _fit_output_head(
    template: Any,
    observations: np.ndarray,
    target_actions: np.ndarray,
    *,
    ridge: float,
    semantic_reset: bool = False,
) -> tuple[Any, dict[str, float]]:
    features = _hidden_3_features(template, observations).astype(np.float64)
    labels = np.arctanh(np.clip(np.asarray(target_actions, dtype=np.float64), -0.999, 0.999))
    design = np.concatenate((features, np.ones((len(features), 1))), axis=1)
    old_kernel = np.asarray(template[1]["params"]["hidden_3"]["kernel"], dtype=np.float64)
    old_bias = np.asarray(template[1]["params"]["hidden_3"]["bias"], dtype=np.float64)
    if semantic_reset:
        # V22/H4's first fourteen channels are residual-action means. H5
        # interprets the same topology as absolute-target actions. A zero
        # prior prevents old residual semantics from surviving in an
        # underdetermined ridge fit; only the target labels establish the new
        # meaning. The scale channels [14:28] remain untouched below.
        prior = np.zeros((design.shape[1], H4_ACTION_WIDTH), dtype=np.float64)
    else:
        prior = np.concatenate((old_kernel[:, :14], old_bias[None, :14]), axis=0)
    normal = design.T @ design + float(ridge) * np.eye(design.shape[1])
    rhs = design.T @ labels + float(ridge) * prior
    solution = np.linalg.solve(normal, rhs).astype(np.float32)
    fitted = copy.deepcopy(template)
    fitted_kernel = np.asarray(fitted[1]["params"]["hidden_3"]["kernel"], dtype=np.float32).copy()
    fitted_bias = np.asarray(fitted[1]["params"]["hidden_3"]["bias"], dtype=np.float32).copy()
    fitted_kernel[:, :14] = solution[:-1]
    fitted_bias[:14] = solution[-1]
    if semantic_reset:
        fitted_kernel[:, 5:9] = 0.0
        fitted_bias[5:9] = 0.0
    fitted[1]["params"]["hidden_3"]["kernel"] = fitted_kernel
    fitted[1]["params"]["hidden_3"]["bias"] = fitted_bias
    validate_h4_params(fitted)
    return fitted, {
        "sample_count": float(len(features)),
        "feature_rank": float(np.linalg.matrix_rank(design)),
        "ridge": float(ridge),
        "semantic_reset": bool(semantic_reset),
        "head_mean_channels_exact_zero": bool(
            semantic_reset
            and np.array_equal(fitted_kernel[:, 5:9], 0.0)
            and np.array_equal(fitted_bias[5:9], 0.0)
        ),
    }


def _lock_semantic_head_mean_channels(params: Any) -> Any:
    """Keep H5 head mean channels exact-zero after any optional fine-tune."""

    fitted = copy.deepcopy(params)
    kernel = np.asarray(
        fitted[1]["params"]["hidden_3"]["kernel"], dtype=np.float32
    ).copy()
    bias = np.asarray(
        fitted[1]["params"]["hidden_3"]["bias"], dtype=np.float32
    ).copy()
    kernel[:, 5:9] = 0.0
    bias[5:9] = 0.0
    fitted[1]["params"]["hidden_3"]["kernel"] = kernel
    fitted[1]["params"]["hidden_3"]["bias"] = bias
    validate_h4_params(fitted)
    return fitted


def _fine_tune_hidden_2_and_output_head(
    initial: Any,
    observations: np.ndarray,
    target_actions: np.ndarray,
    target_targets: np.ndarray,
    *,
    steps: int = 2500,
    learning_rate: float = 1.0e-3,
) -> tuple[Any, dict[str, float]]:
    """Escalate the seed fit by unfreezing hidden_2 only.

    The first pass deliberately fits only the output head.  If the existing
    representation cannot express the inverse decoder accurately, this
    bounded second pass adapts hidden_2 and the first fourteen output channels
    while leaving the normalizer, hidden_0/1, output scale channels, and
    critic untouched.
    """

    import jax
    import jax.numpy as jnp
    import optax

    normalizer, actor, _critic = initial
    layers = actor["params"]
    observations_j = jnp.asarray(np.asarray(observations, dtype=np.float32))
    labels_j = jnp.asarray(
        np.arctanh(np.clip(np.asarray(target_actions, dtype=np.float32), -0.999, 0.999))
    )
    targets_j = jnp.asarray(np.asarray(target_targets, dtype=np.float32))
    _lower, _upper, _initial = contract_target_vectors(xp=np)
    lower_j = jnp.asarray(np.asarray(_lower, dtype=np.float32))
    upper_j = jnp.asarray(np.asarray(_upper, dtype=np.float32))
    initial_j = jnp.asarray(np.asarray(_initial, dtype=np.float32))
    mean_j = jnp.asarray(np.asarray(normalizer.mean["state"], dtype=np.float32))
    std_j = jnp.asarray(np.asarray(normalizer.std["state"], dtype=np.float32))
    fixed_layers = []
    for index in (0, 1):
        fixed_layers.append(
            (
                jnp.asarray(np.asarray(layers[f"hidden_{index}"]["kernel"], dtype=np.float32)),
                jnp.asarray(np.asarray(layers[f"hidden_{index}"]["bias"], dtype=np.float32)),
            )
        )
    initial_w2 = jnp.asarray(np.asarray(layers["hidden_2"]["kernel"], dtype=np.float32))
    initial_b2 = jnp.asarray(np.asarray(layers["hidden_2"]["bias"], dtype=np.float32))
    initial_w3 = jnp.asarray(
        np.asarray(layers["hidden_3"]["kernel"], dtype=np.float32)[:, :14]
    )
    initial_b3 = jnp.asarray(
        np.asarray(layers["hidden_3"]["bias"], dtype=np.float32)[:14]
    )
    trainable = {
        "w2": initial_w2,
        "b2": initial_b2,
        "w3": initial_w3,
        "b3": initial_b3,
    }
    optimizer = optax.adam(float(learning_rate))
    opt_state = optimizer.init(trainable)

    @jax.jit
    def update(parameters: Any, state: Any) -> tuple[Any, Any, Any]:
        def loss_fn(current: Any) -> Any:
            hidden = (observations_j - mean_j) / std_j
            for kernel, bias in fixed_layers:
                hidden = jax.nn.silu(hidden @ kernel + bias)
            hidden = jax.nn.silu(hidden @ current["w2"] + current["b2"])
            logits = hidden @ current["w3"] + current["b3"]
            action = jnp.tanh(logits)
            positive_span = 0.90 * (upper_j - initial_j)
            negative_span = 0.90 * (initial_j - lower_j)
            directional_span = jnp.where(action >= 0.0, positive_span, negative_span)
            base_span = jnp.minimum(0.25, directional_span)
            magnitude = jnp.abs(action)
            target_magnitude = base_span * magnitude + (
                directional_span - base_span
            ) * magnitude**5
            decoded = initial_j + jnp.sign(action) * target_magnitude
            decoded = decoded.at[:, 5:9].set(0.0)
            target_error = decoded - targets_j
            logit_error = logits - labels_j
            regularization = 1.0e-7 * (
                jnp.mean((current["w2"] - initial_w2) ** 2)
                + jnp.mean((current["b2"] - initial_b2) ** 2)
                + jnp.mean((current["w3"] - initial_w3) ** 2)
                + jnp.mean((current["b3"] - initial_b3) ** 2)
            )
            return jnp.mean(target_error**2) + 0.05 * jnp.mean(logit_error**2) + regularization

        value, gradients = jax.value_and_grad(loss_fn)(parameters)
        updates, next_state = optimizer.update(gradients, state, parameters)
        return optax.apply_updates(parameters, updates), next_state, value

    loss_value = float("nan")
    for _ in range(int(steps)):
        trainable, opt_state, loss_value = update(trainable, opt_state)
    trained = copy.deepcopy(initial)
    trained_w2 = np.asarray(trainable["w2"], dtype=np.float32)
    trained_b2 = np.asarray(trainable["b2"], dtype=np.float32)
    trained_w3 = np.asarray(trainable["w3"], dtype=np.float32)
    trained_b3 = np.asarray(trainable["b3"], dtype=np.float32)
    trained[1]["params"]["hidden_2"]["kernel"] = trained_w2
    trained[1]["params"]["hidden_2"]["bias"] = trained_b2
    full_w3 = np.asarray(trained[1]["params"]["hidden_3"]["kernel"], dtype=np.float32).copy()
    full_b3 = np.asarray(trained[1]["params"]["hidden_3"]["bias"], dtype=np.float32).copy()
    full_w3[:, :14] = trained_w3
    full_b3[:14] = trained_b3
    trained[1]["params"]["hidden_3"]["kernel"] = full_w3
    trained[1]["params"]["hidden_3"]["bias"] = full_b3
    validate_h4_params(trained)
    return trained, {
        "hidden_2_unfrozen": 1.0,
        "optimizer_steps": float(steps),
        "learning_rate": float(learning_rate),
        "final_pre_tanh_mse": loss_value,
    }


def _fine_tune_all_actor_location_layers(
    initial: Any,
    observations: np.ndarray,
    target_actions: np.ndarray,
    target_targets: np.ndarray,
    *,
    steps: int = 6000,
    learning_rate: float = 3.0e-4,
) -> tuple[Any, dict[str, float]]:
    """Last-resort simulation seed fit for the full actor location path."""

    import jax
    import jax.numpy as jnp
    import optax

    normalizer, actor, _critic = initial
    layers = actor["params"]
    observations_j = jnp.asarray(np.asarray(observations, dtype=np.float32))
    labels_j = jnp.asarray(
        np.arctanh(np.clip(np.asarray(target_actions, dtype=np.float32), -0.999, 0.999))
    )
    targets_j = jnp.asarray(np.asarray(target_targets, dtype=np.float32))
    lower, upper, safe_init = contract_target_vectors(xp=np)
    lower_j = jnp.asarray(np.asarray(lower, dtype=np.float32))
    upper_j = jnp.asarray(np.asarray(upper, dtype=np.float32))
    safe_init_j = jnp.asarray(np.asarray(safe_init, dtype=np.float32))
    initial_parameters: dict[str, Any] = {}
    for index in range(3):
        initial_parameters[f"w{index}"] = jnp.asarray(
            np.asarray(layers[f"hidden_{index}"]["kernel"], dtype=np.float32)
        )
        initial_parameters[f"b{index}"] = jnp.asarray(
            np.asarray(layers[f"hidden_{index}"]["bias"], dtype=np.float32)
        )
    initial_parameters["w3"] = jnp.asarray(
        np.asarray(layers["hidden_3"]["kernel"], dtype=np.float32)[:, :14]
    )
    initial_parameters["b3"] = jnp.asarray(
        np.asarray(layers["hidden_3"]["bias"], dtype=np.float32)[:14]
    )
    trainable = dict(initial_parameters)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(float(learning_rate))
    )
    opt_state = optimizer.init(trainable)

    @jax.jit
    def update(parameters: Any, state: Any) -> tuple[Any, Any, Any]:
        def loss_fn(current: Any) -> Any:
            hidden = (observations_j - jnp.asarray(normalizer.mean["state"])) / jnp.asarray(
                normalizer.std["state"]
            )
            for index in range(3):
                hidden = jax.nn.silu(hidden @ current[f"w{index}"] + current[f"b{index}"])
            logits = hidden @ current["w3"] + current["b3"]
            action = jnp.tanh(logits)
            positive_span = 0.90 * (upper_j - safe_init_j)
            negative_span = 0.90 * (safe_init_j - lower_j)
            directional_span = jnp.where(action >= 0.0, positive_span, negative_span)
            base_span = jnp.minimum(0.25, directional_span)
            magnitude = jnp.abs(action)
            target_magnitude = base_span * magnitude + (
                directional_span - base_span
            ) * magnitude**5
            decoded = safe_init_j + jnp.sign(action) * target_magnitude
            decoded = decoded.at[:, 5:9].set(0.0)
            target_error = decoded - targets_j
            logit_error = logits - labels_j
            regularization = 1.0e-8 * sum(
                jnp.mean((current[name] - initial_parameters[name]) ** 2)
                for name in initial_parameters
            )
            return jnp.mean(target_error**2) + 0.05 * jnp.mean(logit_error**2) + regularization

        value, gradients = jax.value_and_grad(loss_fn)(parameters)
        updates, next_state = optimizer.update(gradients, state, parameters)
        return optax.apply_updates(parameters, updates), next_state, value

    loss_value = float("nan")
    for _ in range(int(steps)):
        trainable, opt_state, loss_value = update(trainable, opt_state)
    trained = copy.deepcopy(initial)
    for index in range(3):
        trained[1]["params"][f"hidden_{index}"]["kernel"] = np.asarray(
            trainable[f"w{index}"], dtype=np.float32
        )
        trained[1]["params"][f"hidden_{index}"]["bias"] = np.asarray(
            trainable[f"b{index}"], dtype=np.float32
        )
    full_w3 = np.asarray(trained[1]["params"]["hidden_3"]["kernel"], dtype=np.float32).copy()
    full_b3 = np.asarray(trained[1]["params"]["hidden_3"]["bias"], dtype=np.float32).copy()
    full_w3[:, :14] = np.asarray(trainable["w3"], dtype=np.float32)
    full_b3[:14] = np.asarray(trainable["b3"], dtype=np.float32)
    trained[1]["params"]["hidden_3"]["kernel"] = full_w3
    trained[1]["params"]["hidden_3"]["bias"] = full_b3
    validate_h4_params(trained)
    return trained, {
        "all_actor_location_layers_unfrozen": 1.0,
        "optimizer_steps": float(steps),
        "learning_rate": float(learning_rate),
        "final_target_loss": loss_value,
    }


def _target_metrics(params: Any, observations: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    actions = mask_h4_head_action(infer_h4_action_numpy(params, observations))
    decoded = np.asarray(
        [h5_decode_absolute_targets(row, domain="reverse") for row in actions],
        dtype=np.float64,
    )
    errors = decoded - np.asarray(targets, dtype=np.float64)
    absolute = np.abs(errors)
    per_joint = np.sqrt(np.mean(errors**2, axis=0))
    return {
        "target_rmse_rad": float(np.sqrt(np.mean(errors**2))),
        "target_p95_abs_error_rad": float(np.quantile(absolute, 0.95)),
        "target_p99_abs_error_rad": float(np.quantile(absolute, 0.99)),
        "target_max_abs_error_rad": float(np.max(absolute)),
        "action_max_abs_p99": float(np.quantile(np.abs(actions), 0.99)),
        "decoded_head_exact_zero": bool(np.array_equal(decoded[:, 5:9], np.zeros((len(decoded), 4)))),
        "per_joint_rmse_rad": [float(value) for value in per_joint],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.rollouts <= 0 or args.seconds <= 0.0:
        raise ValueError("rollouts and seconds must be positive")
    for label, value in (
        ("rollout_joint_noise_scale", args.rollout_joint_noise_scale),
        ("rollout_initial_base_speed", args.rollout_initial_base_speed),
        ("rollout_warmup_seconds", args.rollout_warmup_seconds),
    ):
        if not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{label} must be finite and non-negative")
    if float(args.rollout_warmup_seconds) >= float(args.seconds):
        raise ValueError("rollout warmup must be shorter than rollout duration")
    if not 0.0 <= args.residual_gain <= 1.0:
        raise ValueError("residual gain must be in [0, 1]")
    if args.ridge < 0.0 or not np.isfinite(args.ridge):
        raise ValueError("ridge must be finite and non-negative")
    rollout_fields = (
        args.rollout_params,
        args.rollout_params_sha256,
        args.rollout_manifest,
        args.rollout_manifest_sha256,
    )
    if any(value is not None for value in rollout_fields) and not all(
        value is not None for value in rollout_fields
    ):
        raise ValueError("rollout params/manifest and all four hashes are required together")
    reverse_params_path = args.reverse_params.resolve()
    if sha256_file(reverse_params_path) != args.reverse_params_sha256:
        raise ValueError("reverse template params SHA mismatch")
    with reverse_params_path.open("rb") as stream:
        template = pickle.load(stream)
    validate_h4_params(template)
    rollout_params_path = (
        args.rollout_params.resolve()
        if args.rollout_params is not None
        else reverse_params_path
    )
    rollout_params_sha256 = (
        args.rollout_params_sha256
        if args.rollout_params_sha256 is not None
        else args.reverse_params_sha256
    )
    if sha256_file(rollout_params_path) != rollout_params_sha256:
        raise ValueError("rollout policy params SHA mismatch")
    with rollout_params_path.open("rb") as stream:
        fit_template = pickle.load(stream)
    validate_h4_params(fit_template)
    if args.teacher_mode == "calibrated_profile":
        if args.calibrated_reference is None or args.calibrated_gait_profile is None:
            raise ValueError(
                "calibrated_profile mode requires --calibrated-reference and "
                "--calibrated-gait-profile"
            )
        teacher_table = _calibrated_profile_teacher_table(
            args.calibrated_reference.resolve(),
            args.calibrated_gait_profile.resolve(),
            teacher_phase_rate=float(args.teacher_phase_rate),
            teacher_phase_offset=float(args.teacher_phase_offset),
        )
        teacher_source = {
            "mode": "calibrated_profile",
            "reference_path": str(args.calibrated_reference.resolve()),
            "gait_profile_path": str(args.calibrated_gait_profile.resolve()),
            "teacher_phase_rate": float(args.teacher_phase_rate),
            "teacher_phase_offset": float(args.teacher_phase_offset),
            "table_index_to_profile_phase": "q/2*teacher_phase_rate+teacher_phase_offset",
        }
    elif args.teacher_mode in {"h5_profile", "h3_profile_residual"}:
        if args.calibrated_gait_profile is None:
            raise ValueError(
                f"{args.teacher_mode} mode requires --calibrated-gait-profile"
            )
        teacher_table = None
        teacher_source = {
            "mode": str(args.teacher_mode),
            "gait_profile_path": str(args.calibrated_gait_profile.resolve()),
            "teacher_phase_rate": float(args.teacher_phase_rate),
            "teacher_phase_offset": float(args.teacher_phase_offset),
            "table_index_to_profile_phase": "q/2*teacher_phase_rate+teacher_phase_offset",
            "target_generator": (
                "OfficialPolicyEvaluator._backward_feedforward"
                if args.teacher_mode == "h5_profile"
                else "OfficialPolicyEvaluator._backward_feedforward_with_v22_residual"
            ),
            "v22_residual_scale": 0.12 if args.teacher_mode == "h3_profile_residual" else 0.0,
            "leg_residual_factor": 0.50 if args.teacher_mode == "h3_profile_residual" else 0.0,
        }
    else:
        selected = load_selected_reverse_teacher(
            args.selected_teacher.resolve()
            if args.selected_teacher is not None
            else EXP_ROOT / "artifacts" / "h4_reverse_slew_feasible_teacher_selected_v1.json"
        )
        teacher_table = np.asarray(selected["table"], dtype=np.float64)
        teacher_source = {
            "mode": "selected",
            "selected_teacher_path": str(
                args.selected_teacher.resolve()
                if args.selected_teacher is not None
                else EXP_ROOT / "artifacts" / "h4_reverse_slew_feasible_teacher_selected_v1.json"
            ),
        }
    teacher_source["target_table_contract"] = "H5_54_ROW_ABSOLUTE_TARGET_TABLE_V1"
    teacher_source["target_table_shape"] = [54, H4_ACTION_WIDTH]
    safe_init = np.asarray([float(SAFE_INIT_POS[name]) for name in ACTUATOR_JOINT_ORDER])

    simulator, bank, simulator_metadata = h5_evaluator._build_simulator(_make_h5_args(args))
    if args.teacher_mode in {"h5_profile", "h3_profile_residual"}:
        teacher_table = _h5_profile_teacher_table(
            simulator,
            args.calibrated_gait_profile.resolve(),
            teacher_phase_rate=float(args.teacher_phase_rate),
            teacher_phase_offset=float(args.teacher_phase_offset),
        )
    if teacher_table is None and args.teacher_mode not in {"h3_profile_residual"}:
        raise RuntimeError("teacher table was not constructed")
    if teacher_table is not None:
        teacher_source["target_table_rad"] = np.asarray(
            teacher_table, dtype=np.float64
        ).tolist()
    h3_scales = None
    h3_biases = None
    h3_joint_ranges = None
    if args.teacher_mode == "h3_profile_residual":
        # The formal H5 evaluator deliberately disables the legacy residual
        # path.  For this diagnostic data-generation pass only, reconstruct
        # the audited H3 composition explicitly: calibrated backward profile
        # plus 0.12 * 0.50 of the frozen V22 leg action.  The resulting label
        # is still inverse-decoded into the H5 absolute-target action space;
        # no legacy residual remains reachable at runtime.
        simulator.evaluator._evaluator.backward_residual_scale = 0.12
        h3_scales, h3_biases, _ = simulator.evaluator.backward_parameters(0.0)
        h3_joint_ranges = np.asarray(
            [
                [0.0, 0.0]
                if name in {"neck_pitch", "head_pitch", "head_yaw", "head_roll"}
                else SAFE_JOINT_LIMITS[name]
                for name in ACTUATOR_JOINT_ORDER
            ],
            dtype=np.float64,
        )
    records: list[dict[str, Any]] = []
    current_rollout = -1
    original_infer_route = bank.infer_route

    def capture_infer_route(
        decision: Any,
        observation: np.ndarray,
        previous_applied_targets: np.ndarray | None = None,
        *call_args: Any,
        **call_kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        result = original_infer_route(
            decision,
            observation,
            previous_applied_targets,
            *call_args,
            **call_kwargs,
        )
        route = str(decision.blend_to_expert)
        domain = h5_domain_for_route(route)
        if domain != "reverse" and not args.unified_collection:
            return result
        values = np.asarray(observation, dtype=np.float32).copy()
        actor_action = mask_h4_head_action(infer_h4_action_numpy(template, values))
        phase = _phase_from_observation(values, simulator.evaluator.phase_steps)
        if domain == "reverse" and args.teacher_mode == "h3_profile_residual":
            if h3_scales is None or h3_biases is None or h3_joint_ranges is None:
                raise RuntimeError("H3 residual teacher was not initialized")
            profile_phase = (
                phase * float(args.teacher_phase_rate)
                + float(args.teacher_phase_offset)
            ) % float(simulator.evaluator.phase_steps)
            v22_action = bank.base_bank.infer(
                route, np.asarray(values[:101], dtype=np.float32)
            )
            teacher_target = simulator.evaluator._backward_feedforward(
                profile_phase,
                safe_init,
                h3_joint_ranges,
                np.clip(np.asarray(v22_action, dtype=np.float64), -1.0, 1.0),
                gait_scales=h3_scales,
                gait_biases=h3_biases,
                leg_residual_factor=0.50,
                head_residual_factor=0.0,
            )
            composite = np.asarray(
                margin_clip_targets(teacher_target, xp=np), dtype=np.float64
            )
        elif domain == "reverse":
            yaw_command = float(
                np.asarray(getattr(decision, "effective_command", (0.0, 0.0, 0.0)))[2]
            )
            if args.unified_collection and abs(yaw_command) > 1.0e-6:
                profile_phase = (
                    phase * float(args.teacher_phase_rate)
                    + float(args.teacher_phase_offset)
                ) % float(simulator.evaluator.phase_steps)
                teacher_target = _h5_profile_teacher_target(
                    simulator, profile_phase, yaw_command
                )
            else:
                teacher_target = interpolate_periodic_table(teacher_table, 2.0 * phase)
            actor_target = np.asarray(
                h5_decode_absolute_targets(actor_action, domain="reverse"), dtype=np.float64
            )
            composite = np.asarray(
                margin_clip_targets(
                    teacher_target + float(args.residual_gain) * (actor_target - safe_init),
                    xp=np,
                ),
                dtype=np.float64,
            )
        else:
            # For non-reverse routes the current H5 actor is the preservation
            # teacher.  This keeps stand/forward/lateral/yaw command
            # conditioning intact while only the reverse target semantics are
            # replaced by the audited diagnostic teacher.
            if bank.last_step is None:
                raise RuntimeError("H5 bank did not publish a route step")
            composite = np.asarray(
                margin_clip_targets(np.asarray(bank.last_step.blended_targets), xp=np),
                dtype=np.float64,
            )
        # Data-generation-only teacher drive.  The final artifact receives
        # only the inverse-decoded actor label; the evaluator never composes
        # this teacher target at runtime.
        if bank.last_step is None:
            raise RuntimeError("H5 bank did not publish a route step")
        if not args.dagger_collection:
            bank.last_step = replace(
                bank.last_step,
                blended_targets=tuple(float(value) for value in composite),
            )
        reachable = _decoder_reachable_target(composite)
        target_action = _inverse_decoder(reachable)
        records.append(
            {
                "observation": values,
                "target_action": target_action,
                "target": reachable,
                "composite_target": composite,
                "phase": phase,
                "route": route,
                "rollout": int(current_rollout),
            }
        )
        return result

    bank.infer_route = capture_infer_route  # type: ignore[method-assign]
    if args.unified_collection:
        schedule = [
            (name, tuple(H5_UNIFIED_PHYSICAL_COMMANDS[name]), float(args.seconds))
            for name in H5_UNIFIED_ROUTE_NAMES
        ]
    else:
        schedule = [("reverse", (-0.05, 0.0, 0.0), float(args.seconds))]
    for rollout in range(args.rollouts):
        current_rollout = int(rollout)
        simulator.run_schedule(
            schedule,
            seed=int(args.seed) + rollout,
            joint_noise_scale=float(args.rollout_joint_noise_scale),
            initial_base_speed=float(args.rollout_initial_base_speed),
            warmup_seconds=float(args.rollout_warmup_seconds),
        )
    if len(records) < 300:
        raise RuntimeError(f"insufficient reverse seed observations: {len(records)}")

    observations = np.stack([row["observation"] for row in records]).astype(np.float32)
    target_actions = np.stack([row["target_action"] for row in records]).astype(np.float32)
    targets = np.stack([row["target"] for row in records]).astype(np.float64)
    # Hold out complete causal rollouts.  A random row split can report a
    # deceptively good fit because adjacent phase/state rows from one rollout
    # appear in both partitions.  The grouped split is a real teacher-state
    # generalization check and is reproducible from the rollout seed order.
    rollout_ids = np.asarray([int(row["rollout"]) for row in records], dtype=np.int64)
    unique_rollouts = np.unique(rollout_ids)
    if len(unique_rollouts) < 2:
        raise RuntimeError("grouped held-out split requires at least two rollouts")
    heldout_rollout_count = max(1, int(np.ceil(len(unique_rollouts) * 0.25)))
    heldout_rollouts = unique_rollouts[-heldout_rollout_count:]
    train_rollouts = unique_rollouts[:-heldout_rollout_count]
    train_indices = np.flatnonzero(np.isin(rollout_ids, train_rollouts))
    heldout_indices = np.flatnonzero(np.isin(rollout_ids, heldout_rollouts))
    if len(train_indices) == 0 or len(heldout_indices) == 0:
        raise RuntimeError("grouped held-out split is empty")
    fitted, fit_audit = _fit_output_head(
        fit_template,
        observations[train_indices],
        target_actions[train_indices],
        ridge=float(args.ridge),
        semantic_reset=bool(args.semantic_reset),
    )
    train_metrics = _target_metrics(
        fitted, observations[train_indices], targets[train_indices]
    )
    heldout_metrics = _target_metrics(
        fitted, observations[heldout_indices], targets[heldout_indices]
    )
    if (
        args.fit_all_actor_layers
        or train_metrics["target_rmse_rad"] > 0.01
        or train_metrics["target_p99_abs_error_rad"] > 0.02
        or heldout_metrics["target_rmse_rad"] > 0.01
        or heldout_metrics["target_p99_abs_error_rad"] > 0.02
    ):
        fitted, fine_tune_audit = _fine_tune_hidden_2_and_output_head(
            fitted,
            observations[train_indices],
            target_actions[train_indices],
            targets[train_indices],
        )
        if args.semantic_reset:
            fitted = _lock_semantic_head_mean_channels(fitted)
        fit_audit = {**fit_audit, **fine_tune_audit}
        train_metrics = _target_metrics(
            fitted, observations[train_indices], targets[train_indices]
        )
        heldout_metrics = _target_metrics(
            fitted, observations[heldout_indices], targets[heldout_indices]
        )
    if (
        args.fit_all_actor_layers
        or train_metrics["target_rmse_rad"] > 0.01
        or train_metrics["target_p99_abs_error_rad"] > 0.02
        or heldout_metrics["target_rmse_rad"] > 0.01
        or heldout_metrics["target_p99_abs_error_rad"] > 0.02
    ):
        fitted, full_tune_audit = _fine_tune_all_actor_location_layers(
            fitted,
            observations[train_indices],
            target_actions[train_indices],
            targets[train_indices],
        )
        if args.semantic_reset:
            fitted = _lock_semantic_head_mean_channels(fitted)
        fit_audit = {**fit_audit, **full_tune_audit}
        train_metrics = _target_metrics(
            fitted, observations[train_indices], targets[train_indices]
        )
        heldout_metrics = _target_metrics(
            fitted, observations[heldout_indices], targets[heldout_indices]
        )
    if heldout_metrics["target_rmse_rad"] > 0.01 or heldout_metrics["target_p99_abs_error_rad"] > 0.02:
        raise RuntimeError(
            "target-space held-out fidelity failed: "
            f"train={train_metrics}, heldout={heldout_metrics}, fit={fit_audit}"
        )

    output_params = args.output_params.resolve()
    output_manifest = args.output_manifest.resolve()
    output_params.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_params.open("wb") as stream:
        pickle.dump(fitted, stream, protocol=pickle.HIGHEST_PROTOCOL)
    params_sha = sha256_file(output_params)
    phase_values = np.asarray([row["phase"] for row in records], dtype=np.float64)
    manifest = {
        "schema_version": 1,
        "status": "COMPLETED",
        "hardware_deployment": "PROHIBITED",
        "run_name": output_params.parent.name,
        "expert": "reverse",
        "qualification_use": "DIAGNOSTIC_REWARD_EXPLORATION_NOT_QUALIFICATION",
        "candidate_kind": "H5_TARGET_SPACE_DISTILLED_SEED",
        "actor_observation_width": H4_ACTOR_OBSERVATION_WIDTH,
        "source_template": {
            "params_path": str(reverse_params_path),
            "params_sha256": args.reverse_params_sha256,
        },
        "teacher_source": teacher_source,
        "rollout_policy": {
            "params_path": str(rollout_params_path),
            "params_sha256": rollout_params_sha256,
            "data_generation_target_authority": "teacher_target_plus_v7_residual",
            "runtime_target_authority": "actor_only",
        },
        "selected_reverse_teacher": (
            {
                "path": str(selected["path"]),
                "sha256": selected["sha256"],
                "phase_advance_bins_per_control": selected["phase_advance_bins"],
                "teacher_phase_multiplier": 2.0,
            }
            if args.teacher_mode == "selected"
            else {
                "mode": str(args.teacher_mode),
                "teacher_phase_multiplier": 2.0,
                "teacher_phase_rate": float(args.teacher_phase_rate),
                "teacher_phase_offset": float(args.teacher_phase_offset),
                "h5_phase_delta_bins_per_control": 0.81,
                **(
                    {
                        "target_generator": (
                            "OfficialPolicyEvaluator._backward_feedforward"
                        )
                    }
                    if args.teacher_mode in {"h5_profile", "h3_profile_residual"}
                    else {}
                ),
            }
        ),
        "target_space_distillation": {
            "formula": (
                (
                    "margin_clip(H3_profile(phase*rate+offset) + "
                    "0.12*0.50*V22_action_residual)"
                    if args.teacher_mode == "h3_profile_residual"
                    else "margin_clip(T_54(2*phase) + gain*(D(a_v7)-SAFE_INIT)); "
                    "T_54[q]=profile[q/2*teacher_phase_rate+teacher_phase_offset]"
                )
            ),
            "decoder_contract": "OPEN_DUCK_MINI_H5_TARGET_SPACE_ROUTING_V1",
            "residual_gain": float(args.residual_gain),
            "fitted_parameters": ["actor.params.hidden_3.kernel[:,0:14]", "actor.params.hidden_3.bias[0:14]"],
            "preserved_parameters": ["normalizer", "hidden_0", "hidden_1", "hidden_2", "hidden_3 scale channels", "critic"],
            "semantic_initialization": {
                "enabled": bool(args.semantic_reset),
                "mean_head_semantics": (
                    "absolute_h5_targets" if args.semantic_reset else "inherited_template_prior"
                ),
                "head_mean_channels_locked_zero": bool(args.semantic_reset),
                "scale_channels_preserved": True,
            },
            "rollouts": int(args.rollouts),
            "unified_collection": bool(args.unified_collection),
            "dagger_collection": bool(args.dagger_collection),
            "collection_routes": list(H5_UNIFIED_ROUTE_NAMES)
            if args.unified_collection
            else ["reverse"],
            "seconds_per_rollout": float(args.seconds),
            "rollout_joint_noise_scale": float(args.rollout_joint_noise_scale),
            "rollout_initial_base_speed": float(args.rollout_initial_base_speed),
            "rollout_warmup_seconds": float(args.rollout_warmup_seconds),
            "split": {
                "kind": "grouped_by_causal_rollout",
                "train_rollouts": [int(value) for value in train_rollouts],
                "heldout_rollouts": [int(value) for value in heldout_rollouts],
                "train_count": int(len(train_indices)),
                "heldout_count": int(len(heldout_indices)),
            },
            "record_count": len(records),
            "phase_min": float(np.min(phase_values)),
            "phase_max": float(np.max(phase_values)),
            "first_reverse_phase_expected": 7.0,
            "first_teacher_phase_expected": 14.0,
            "fit": fit_audit,
            "train_metrics": train_metrics,
            "heldout_metrics": heldout_metrics,
            "passed": True,
        },
        "simulator_metadata": simulator_metadata,
        "outputs": {
            "final_params": {
                "path": str(output_params),
                "sha256": params_sha,
            }
        },
        "notes": [
            "Simulation-only seed; not a qualification or release candidate.",
            "No legacy runtime teacher target is used by the H5 evaluator.",
            "PPO may use this artifact only through the explicit H5 seed flag.",
        ],
    }
    output_manifest.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    manifest_sha = sha256_file(output_manifest)
    manifest["outputs"]["manifest"] = {"path": str(output_manifest), "sha256": manifest_sha}
    output_manifest.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    manifest_sha = sha256_file(output_manifest)
    return {
        "output_params": str(output_params),
        "params_sha256": params_sha,
        "output_manifest": str(output_manifest),
        "manifest_sha256": manifest_sha,
        "record_count": len(records),
        "train_metrics": train_metrics,
        "heldout_metrics": heldout_metrics,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
