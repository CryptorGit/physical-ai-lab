"""Build a simulation-only receding-horizon teacher for H5 reverse motion.

This is an experiment tool, not an adoption or deployment producer.  It uses
the exact exp_004 MuJoCo scene, command mapper, target decoder, and final
target safety contract.  The existing V22 waveform is used only as a one-step
warm-start prior; all labels are the first absolute target selected by CEM.

The default CEM configuration is intentionally frozen for the first teacher
screen suggested in the exp_004 PDCA record:

* 0.02 s control step, 40 control ticks (0.8 s) horizon;
* 256 candidates, 32 elites, 5 iterations;
* 10 leg target velocities represented as 8 blocks of 5 ticks;
* replan at every control tick with a one-tick shifted warm start.

The generated files are hardware-prohibited diagnostic artifacts.  A teacher
is not eligible for distillation until the separate strict three-seed screen
has passed every frozen gate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
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
from safe_gait_experts.h4_training_alignment import (  # noqa: E402
    OBSERVATION_POLICY_COMMAND_SLICE,
)
from safe_gait_experts.h5_command_contract import (  # noqa: E402
    h5_unified_policy_command,
)
from safe_gait_experts.h5_target_contract import (  # noqa: E402
    h5_decode_absolute_targets,
)
from scripts import evaluate_h5_routed_transitions as h5  # noqa: E402
from scripts import evaluate_h4_routed_transitions as h4  # noqa: E402
from scripts import evaluate_routed_transitions as routed  # noqa: E402


LEG_INDICES = np.asarray(
    [
        index
        for index, name in enumerate(ACTUATOR_JOINT_ORDER)
        if name not in {"neck_pitch", "head_pitch", "head_yaw", "head_roll"}
    ],
    dtype=np.int32,
)
HEAD_INDICES = np.asarray([5, 6, 7, 8], dtype=np.int32)
PHYSICAL_COMMAND = np.asarray((-0.05, 0.0, 0.0), dtype=np.float64)
POLICY_COMMAND = np.asarray(
    h5_unified_policy_command(PHYSICAL_COMMAND), dtype=np.float64
)
COMMAND7 = np.asarray((*POLICY_COMMAND, 0.0, 0.0, 0.0, 0.0), dtype=np.float32)

CONTROL_DT_S = 0.02
TARGET_DELTA_RAD = 0.04
TARGET_MARGIN_RAD = 0.050
DEFAULT_HORIZON_TICKS = 40
DEFAULT_BLOCK_TICKS = 5
DEFAULT_POPULATION = 256
DEFAULT_ELITES = 32
DEFAULT_ITERATIONS = 5
DEFAULT_WARM_STD = 0.65
DEFAULT_SEED = (20260810,)

DEFAULT_PARAMS = (
    EXP_ROOT
    / "artifacts"
    / "h5_training_runs_diagnostic_20260811"
    / "semantic"
    / "h5_profile_semantic_reset_unified_v2"
    / "final_params.pkl"
)
DEFAULT_PLANAR_MANIFEST = (
    EXP_ROOT
    / "artifacts"
    / "h5_diagnostic_wrappers"
    / "h5_profile_semantic_reset_unified_v2"
    / "planar"
    / "manifest.json"
)
DEFAULT_REVERSE_MANIFEST = (
    EXP_ROOT
    / "artifacts"
    / "h5_diagnostic_wrappers"
    / "h5_profile_semantic_reset_unified_v2"
    / "reverse"
    / "manifest.json"
)
DEFAULT_GENERATED_ROOT = EXP_ROOT / "artifacts" / "generated_playground"
DEFAULT_OUTPUT_PREFIX = EXP_ROOT / "artifacts" / "h5_mpc_reverse_teacher_20260811"
BASE_V22 = (
    EXP_ROOT
    / "artifacts"
    / "router_packages"
    / "exp004-safe-gait-router-h3-release-20260808-v1"
    / "models"
    / "base_v22.onnx"
)


def _finite_vector(value: Sequence[float], shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _inverse_decoder(target: Sequence[float]) -> np.ndarray:
    """Invert the official H5 linear-plus-quintic absolute decoder."""

    values = _finite_vector(target, (14,), "target")
    initial = np.asarray(
        [float(SAFE_INIT_POS[name]) for name in ACTUATOR_JOINT_ORDER],
        dtype=np.float64,
    )
    lower = np.asarray(
        [
            0.0
            if name in {"neck_pitch", "head_pitch", "head_yaw", "head_roll"}
            else float(SAFE_JOINT_LIMITS[name][0])
            for name in ACTUATOR_JOINT_ORDER
        ],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            0.0
            if name in {"neck_pitch", "head_pitch", "head_yaw", "head_roll"}
            else float(SAFE_JOINT_LIMITS[name][1])
            for name in ACTUATOR_JOINT_ORDER
        ],
        dtype=np.float64,
    )
    action = np.zeros(14, dtype=np.float32)
    for index in LEG_INDICES:
        delta = float(values[index] - initial[index])
        span = 0.90 * (
            upper[index] - initial[index]
            if delta >= 0.0
            else initial[index] - lower[index]
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


def _exact_guard_step(
    simulator: Any,
    previous: Sequence[float],
    desired: Sequence[float],
) -> np.ndarray:
    """Apply the same margin, slew, and physical-safe stages as the runtime."""

    previous_values = _finite_vector(previous, (14,), "previous targets")
    desired_values = _finite_vector(desired, (14,), "desired targets").copy()
    desired_values[HEAD_INDICES] = 0.0
    margin_clipped = routed.apply_final_target_safety(
        desired_values,
        SAFE_JOINT_LIMITS,
        margin_rad=TARGET_MARGIN_RAD,
    )
    applied = previous_values.copy()
    applied[LEG_INDICES] = previous_values[LEG_INDICES] + np.clip(
        margin_clipped[LEG_INDICES] - previous_values[LEG_INDICES],
        -TARGET_DELTA_RAD,
        TARGET_DELTA_RAD,
    )
    applied[LEG_INDICES] = np.clip(
        applied[LEG_INDICES],
        simulator.safe_lower[LEG_INDICES],
        simulator.safe_upper[LEG_INDICES],
    )
    applied[HEAD_INDICES] = 0.0
    return applied


def _target_from_z(
    simulator: Any,
    previous: np.ndarray,
    z_block: Sequence[float],
) -> np.ndarray:
    values = _finite_vector(z_block, (10,), "CEM block")
    desired = previous.copy()
    desired[LEG_INDICES] = previous[LEG_INDICES] + TARGET_DELTA_RAD * np.tanh(values)
    desired[HEAD_INDICES] = 0.0
    return _exact_guard_step(simulator, previous, desired)


def _copy_data(simulator: Any, source: Any) -> Any:
    copy = simulator.mujoco.MjData(simulator.model)
    simulator.mujoco.mj_copyData(copy, simulator.model, source)
    simulator.mujoco.mj_forward(simulator.model, copy)
    return copy


def _trunk_rotation(simulator: Any, data: Any) -> np.ndarray:
    return np.asarray(
        data.xmat[simulator.evaluator.trunk_body_id], dtype=np.float64
    ).reshape(3, 3)


def _rollout_cost(
    simulator: Any,
    base_data: Any,
    previous_targets: np.ndarray,
    z: np.ndarray,
    *,
    horizon_ticks: int,
    block_ticks: int,
    collect_trace: bool = False,
    proposal_basis: dict[str, np.ndarray] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Roll out one CEM proposal with hard safety rejection."""

    proposal_width = 10 if proposal_basis is None else 3
    if z.shape != (int(np.ceil(horizon_ticks / block_ticks)), proposal_width):
        raise ValueError("CEM proposal shape does not match horizon and blocks")
    if proposal_basis is not None:
        for name in ("trace_delta", "sagittal_delta"):
            values = np.asarray(proposal_basis[name], dtype=np.float64)
            if values.shape != (horizon_ticks, 10):
                raise ValueError(f"structured proposal {name} must be horizon x 10")
        roll_transfer = np.asarray(proposal_basis["roll_transfer"], dtype=np.float64)
        if roll_transfer.shape != (10,):
            raise ValueError("structured proposal roll_transfer must be ten-wide")
    data = _copy_data(simulator, base_data)
    previous = previous_targets.copy()
    start_position = np.asarray(
        data.xpos[simulator.evaluator.trunk_body_id], dtype=np.float64
    ).copy()
    local_velocities: list[np.ndarray] = []
    normal_forces: list[np.ndarray] = []
    tangential_speeds: list[np.ndarray] = []
    contacts: list[np.ndarray] = []
    upright_values: list[float] = []
    heights: list[float] = []
    target_deltas: list[np.ndarray] = []
    joint_limit_margins: list[float] = []
    joint_limit_margins_rad: list[float] = []
    joint_positions: list[np.ndarray] = []
    qpos_violation = False
    nonfinite = False
    fell = False
    for tick in range(horizon_ticks):
        block = min(tick // block_ticks, z.shape[0] - 1)
        if proposal_basis is None:
            applied = _target_from_z(simulator, previous, z[block])
        else:
            delta = (
                float(z[block, 0]) * proposal_basis["trace_delta"][tick]
                + float(z[block, 1]) * proposal_basis["roll_transfer"]
                + float(z[block, 2]) * proposal_basis["sagittal_delta"][tick]
            )
            desired = previous.copy()
            desired[LEG_INDICES] = previous[LEG_INDICES] + delta
            desired[HEAD_INDICES] = 0.0
            applied = _exact_guard_step(simulator, previous, desired)
        target_deltas.append(applied[LEG_INDICES] - previous[LEG_INDICES])
        data.ctrl[:] = applied
        simulator.mujoco.mj_step(
            simulator.model,
            data,
            nstep=int(simulator.runtime.DECIMATION),
        )
        previous = applied
        rotation = _trunk_rotation(simulator, data)
        position = np.asarray(
            data.xpos[simulator.evaluator.trunk_body_id], dtype=np.float64
        )
        local_velocity = np.asarray(
            simulator.evaluator._sensor(data, "local_linvel"), dtype=np.float64
        )
        normal_force, tangential_speed = simulator._quality_contact_kinematics(data)
        contact = np.asarray(simulator.evaluator._feet_contacts(data), dtype=bool)
        local_velocities.append(local_velocity)
        normal_forces.append(normal_force)
        tangential_speeds.append(tangential_speed)
        contacts.append(contact)
        upright_values.append(float(rotation[2, 2]))
        heights.append(float(position[2]))
        qpos = np.asarray(
            data.qpos[simulator.evaluator.actuator_qpos_addr], dtype=np.float64
        )
        normalized_joint_margin = np.minimum(
            (qpos[LEG_INDICES] - simulator.safe_lower[LEG_INDICES])
            / (simulator.safe_upper[LEG_INDICES] - simulator.safe_lower[LEG_INDICES]),
            (simulator.safe_upper[LEG_INDICES] - qpos[LEG_INDICES])
            / (simulator.safe_upper[LEG_INDICES] - simulator.safe_lower[LEG_INDICES]),
        )
        joint_limit_margins.append(float(np.min(normalized_joint_margin)))
        joint_limit_margins_rad.append(
            float(
                np.min(
                    np.minimum(
                        qpos[LEG_INDICES] - simulator.safe_lower[LEG_INDICES],
                        simulator.safe_upper[LEG_INDICES] - qpos[LEG_INDICES],
                    )
                )
            )
        )
        joint_positions.append(qpos.copy())
        if (
            not np.all(np.isfinite(data.qpos))
            or not np.all(np.isfinite(data.qvel))
            or not np.all(np.isfinite(qpos))
        ):
            nonfinite = True
            break
        if np.any(qpos[LEG_INDICES] < simulator.safe_lower[LEG_INDICES] - 1.0e-8) or np.any(
            qpos[LEG_INDICES] > simulator.safe_upper[LEG_INDICES] + 1.0e-8
        ):
            qpos_violation = True
            break
        if float(position[2]) < 0.12 or float(rotation[2, 2]) < 0.65:
            fell = True
            break

    velocities = np.asarray(local_velocities, dtype=np.float64)
    forces = np.asarray(normal_forces, dtype=np.float64)
    slips = np.asarray(tangential_speeds, dtype=np.float64)
    contact_array = np.asarray(contacts, dtype=bool)
    if not len(velocities):
        return float("inf"), {
            "valid": False,
            "fell": fell,
            "qpos_violation": qpos_violation,
            "nonfinite": nonfinite,
            "completed_ticks": 0,
        }

    # These are deliberately the fixed teacher objective components.  They
    # favor the required signed speed and support while keeping the target
    # motion smooth and the two feet usable.  The final frozen gait evaluator
    # remains the authority; this is only CEM's search objective.
    projected_speed = -velocities[:, 0]
    speed_loss = np.mean(((projected_speed - 0.05) / 0.05) ** 2)
    cross_loss = np.mean((velocities[:, 1] / 0.05) ** 2)
    yaw_loss = np.mean((velocities[:, 2] / 0.20) ** 2)
    force_total = forces.sum(axis=1)
    force_band_loss = np.mean((force_total - 1.0) ** 2)
    force_tail_loss = max(0.0, float(np.percentile(force_total, 99)) - 3.0) ** 2
    # Normalize slip before weighting.  The frozen gates are in m/s, so an
    # unnormalized squared speed would be numerically dominated by the
    # velocity term and would reward sliding.
    slip_loss = np.mean((slips / 0.03) ** 2)
    slip_tail_loss = max(0.0, float(np.percentile(slips, 95)) / 0.03 - 1.0) ** 2
    support_loss = float(np.mean(force_total < 0.05))
    single_support_rate = float(
        np.mean(np.logical_xor(contact_array[:, 0], contact_array[:, 1]))
    )
    single_support_band_loss = (
        max(0.0, 0.25 - single_support_rate) / 0.25
    ) ** 2 + (max(0.0, single_support_rate - 0.60) / 0.40) ** 2
    touchdown_feet: list[int] = []
    previous_contact: np.ndarray | None = None
    for contact in contact_array:
        if previous_contact is not None:
            for foot in (0, 1):
                if not bool(previous_contact[foot]) and bool(contact[foot]):
                    touchdown_feet.append(foot)
        previous_contact = contact
    if len(touchdown_feet) < 2:
        alternation_fraction = 0.0
    else:
        alternation_fraction = float(
            np.mean(
                np.asarray(touchdown_feet[1:], dtype=np.int32)
                != np.asarray(touchdown_feet[:-1], dtype=np.int32)
            )
        )
    alternation_loss = (1.0 - alternation_fraction) ** 2
    left_right_imbalance = abs(
        float(np.mean(contact_array[:, 0])) - float(np.mean(contact_array[:, 1]))
    )
    upright_loss = np.mean((1.0 - np.asarray(upright_values)) ** 2)
    height_loss = np.mean(np.maximum(0.0, 0.16 - np.asarray(heights)) ** 2)
    effort_loss = np.mean(
        (np.asarray(target_deltas, dtype=np.float64) / TARGET_DELTA_RAD) ** 2
    )
    terminal_loss = (1.0 - upright_values[-1]) ** 2 + max(
        0.0, 0.16 - heights[-1]
    ) ** 2
    joint_limit_barrier = np.mean(
        np.maximum(0.0, 0.10 - np.asarray(joint_limit_margins)) ** 2
    )
    safe_init = np.asarray(
        [float(SAFE_INIT_POS[name]) for name in ACTUATOR_JOINT_ORDER],
        dtype=np.float64,
    )
    joint_span = simulator.safe_upper[LEG_INDICES] - simulator.safe_lower[LEG_INDICES]
    joint_posture_loss = np.mean(
        (
            (
                np.asarray(joint_positions, dtype=np.float64)[:, LEG_INDICES]
                - safe_init[LEG_INDICES]
            )
            / joint_span
        )
        ** 2
    )
    cost = (
        10.0 * speed_loss
        + 2.0 * cross_loss
        + 0.5 * yaw_loss
        + 2.0 * slip_loss
        + 1.0 * slip_tail_loss
        + 1.0 * force_band_loss
        + 1.0 * force_tail_loss
        + 10.0 * support_loss
        + 1.0 * single_support_band_loss
        + 4.0 * alternation_loss
        + 1.0 * left_right_imbalance
        + 4.0 * upright_loss
        + 2.0 * height_loss
        + 20.0 * joint_limit_barrier
        + 1.0 * joint_posture_loss
        + 0.25 * effort_loss
        + 1.0 * terminal_loss
    )
    invalid = fell or qpos_violation or nonfinite or len(velocities) < horizon_ticks
    if invalid:
        cost += 1.0e5
    details = {
        "valid": not invalid,
        "fell": fell,
        "qpos_violation": qpos_violation,
        "nonfinite": nonfinite,
        "completed_ticks": len(velocities),
        "projected_speed_mean_mps": float(np.mean(projected_speed)),
        "speed_loss": float(speed_loss),
        "support_loss": support_loss,
        "single_support_rate": single_support_rate,
        "single_support_band_loss": float(single_support_band_loss),
        "touchdown_count": len(touchdown_feet),
        "alternation_fraction": float(alternation_fraction),
        "alternation_loss": float(alternation_loss),
        "slip_rms_mps": float(np.sqrt(np.mean(slips**2))),
        "slip_p95_mps": float(np.percentile(slips, 95)),
        "force_p99_body_weight": float(np.percentile(force_total, 99)),
        "upright_min": float(np.min(upright_values)),
        "height_min_m": float(np.min(heights)),
        "minimum_normalized_joint_margin": float(np.min(joint_limit_margins)),
        "minimum_joint_margin_rad": float(np.min(joint_limit_margins_rad)),
        "joint_limit_barrier": float(joint_limit_barrier),
        "joint_posture_loss": float(joint_posture_loss),
        "cost": float(cost),
    }
    if collect_trace:
        details["local_velocities"] = velocities
        details["normal_forces"] = forces
        details["tangential_speeds"] = slips
        details["contacts"] = contact_array
    return float(cost), details


def _warm_seed_target(simulator: Any, observation: np.ndarray, phase_index: float) -> np.ndarray:
    """Return one diagnostic warm-start target; never used as a teacher label."""

    policy_observation = np.asarray(observation[:101], dtype=np.float32).copy()
    warped_angle = (
        (0.75 * float(phase_index) + 13.5)
        / float(simulator.evaluator.phase_steps)
        * 2.0
        * np.pi
    )
    policy_observation[99:101] = np.asarray(
        (np.cos(warped_angle), np.sin(warped_angle)), dtype=np.float32
    )
    forward_policy_command = np.asarray(
        h5_unified_policy_command((0.05, 0.0, 0.0)), dtype=np.float32
    )
    policy_observation[OBSERVATION_POLICY_COMMAND_SLICE] = np.asarray(
        (*forward_policy_command, 0.0, 0.0, 0.0, 0.0), dtype=np.float32
    )
    action = simulator.bank.base_bank.infer("forward", policy_observation)
    target = np.asarray(
        h5_decode_absolute_targets(action, domain="planar"), dtype=np.float64
    )
    initial = np.asarray(
        [float(SAFE_INIT_POS[name]) for name in ACTUATOR_JOINT_ORDER], dtype=np.float64
    )
    delta = target - initial
    # The selected historical prior is only the sagittal mirror hypothesis.
    target[[2, 4, 11, 13]] = initial[[2, 4, 11, 13]] - delta[[2, 4, 11, 13]]
    target[HEAD_INDICES] = 0.0
    return target


def _initial_mean(
    simulator: Any,
    data: Any,
    previous_targets: np.ndarray,
    action_history: list[np.ndarray],
    phase_index: float,
) -> tuple[np.ndarray, np.ndarray]:
    h4._H4_RUN_CONTEXT["physical_command"] = PHYSICAL_COMMAND.copy()
    observation = simulator.evaluator._observation(
        data,
        COMMAND7,
        np.asarray(simulator.model.keyframe("home").ctrl, dtype=np.float64),
        previous_targets,
        action_history,
        phase_index / float(simulator.evaluator.phase_steps) * 2.0 * np.pi,
    )
    warm_target = _warm_seed_target(simulator, observation, phase_index)
    first_delta = np.clip(
        warm_target[LEG_INDICES] - previous_targets[LEG_INDICES],
        -0.039,
        0.039,
    )
    first_z = np.arctanh(first_delta / TARGET_DELTA_RAD)
    return np.vstack(
        [first_z, np.zeros((7, 10), dtype=np.float64)]
    ), warm_target


def _shift_prior(plan_ticks: np.ndarray, horizon_ticks: int) -> np.ndarray:
    if plan_ticks.shape != (horizon_ticks, 10):
        raise ValueError("prior tick plan has the wrong shape")
    shifted = np.empty_like(plan_ticks)
    shifted[:-1] = plan_ticks[1:]
    shifted[-1] = plan_ticks[-1]
    return shifted


def _compress_ticks(plan_ticks: np.ndarray, block_ticks: int, blocks: int) -> np.ndarray:
    components = int(plan_ticks.shape[1])
    padded = np.repeat(
        plan_ticks[-1][None, :], blocks * block_ticks, axis=0
    ).reshape(blocks * block_ticks, components)
    padded[: min(len(plan_ticks), len(padded))] = plan_ticks[: len(padded)]
    return padded.reshape(blocks, block_ticks, components).mean(axis=1)


def _cem_plan(
    simulator: Any,
    data: Any,
    previous_targets: np.ndarray,
    prior_ticks: np.ndarray,
    rng: np.random.Generator,
    *,
    horizon_ticks: int,
    block_ticks: int,
    population: int,
    elites: int,
    iterations: int,
    warm_std: float,
    proposal_basis: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    blocks = int(np.ceil(horizon_ticks / block_ticks))
    proposal_width = 10 if proposal_basis is None else 3
    if prior_ticks.shape != (horizon_ticks, proposal_width):
        raise ValueError("prior tick plan width does not match proposal basis")
    mean = _compress_ticks(prior_ticks, block_ticks, blocks)
    std = np.full((blocks, proposal_width), float(warm_std), dtype=np.float64)
    best_z = mean.copy()
    best_cost = float("inf")
    best_details: dict[str, Any] = {}
    iteration_records: list[dict[str, Any]] = []
    for iteration in range(iterations):
        samples = rng.normal(
            mean, std, size=(population, blocks, proposal_width)
        )
        samples = np.clip(samples, -2.5, 2.5)
        # Always test conservative feasibility anchors.  This is a safety
        # invariant, not a performance shortcut: CEM must not discard a
        # current/zero-motion recovery option merely because random samples
        # all push the same joint toward its physical limit.
        samples[0] = mean
        if population > 1:
            samples[1] = 0.0
        if population > 2:
            samples[2] = np.clip(-mean, -2.5, 2.5)
        costs = np.empty(population, dtype=np.float64)
        details: list[dict[str, Any]] = []
        for index, sample in enumerate(samples):
            cost, sample_details = _rollout_cost(
                simulator,
                data,
                previous_targets,
                sample,
                horizon_ticks=horizon_ticks,
                block_ticks=block_ticks,
                proposal_basis=proposal_basis,
            )
            costs[index] = cost
            details.append(sample_details)
        elite_count = min(max(1, int(elites)), population)
        elite_indices = np.argsort(costs)[:elite_count]
        elite_values = samples[elite_indices]
        mean = np.mean(elite_values, axis=0)
        std = np.maximum(0.10, np.std(elite_values, axis=0))
        candidate_index = int(elite_indices[0])
        if float(costs[candidate_index]) < best_cost:
            best_cost = float(costs[candidate_index])
            best_z = samples[candidate_index].copy()
            best_details = dict(details[candidate_index])
        iteration_records.append(
            {
                "iteration": iteration,
                "best_cost": float(costs[candidate_index]),
                "elite_cost_mean": float(np.mean(costs[elite_indices])),
                "valid_elite_count": int(
                    sum(bool(details[index].get("valid", False)) for index in elite_indices)
                ),
                "population_valid_count": int(
                    sum(bool(item.get("valid", False)) for item in details)
                ),
            }
        )
    return best_z, {
        "best_cost": best_cost,
        "best_details": best_details,
        "iterations": iteration_records,
        "mean": mean,
        "std": std,
    }


def _build_args(
    params: Path,
    planar_manifest: Path,
    reverse_manifest: Path,
    generated_root: Path,
) -> argparse.Namespace:
    policy = [f"{role}={BASE_V22}" for role in routed.REQUIRED_POLICY_ROLES]
    params_sha = h5.sha256_file(params)
    planar_manifest_sha = h5.sha256_file(planar_manifest)
    reverse_manifest_sha = h5.sha256_file(reverse_manifest)
    return argparse.Namespace(
        policy=policy,
        generated_root=generated_root.resolve(),
        unified_single_weight=True,
        h5_planar_params=params.resolve(),
        h5_planar_params_sha256=params_sha,
        h5_planar_manifest=planar_manifest.resolve(),
        h5_planar_manifest_sha256=planar_manifest_sha,
        h5_reverse_params=params.resolve(),
        h5_reverse_params_sha256=params_sha,
        h5_reverse_manifest=reverse_manifest.resolve(),
        h5_reverse_manifest_sha256=reverse_manifest_sha,
        strict_actor_only=True,
    )


def _run_seed(
    simulator: Any,
    *,
    seed: int,
    seconds: float,
    horizon_ticks: int,
    block_ticks: int,
    population: int,
    elites: int,
    iterations: int,
    warm_std: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if horizon_ticks <= 0 or block_ticks <= 0 or horizon_ticks % block_ticks != 0:
        raise ValueError("horizon_ticks must be a positive multiple of block_ticks")
    data, reset_audit = simulator._initial_data(seed, 0.0, 0.0)
    default = np.asarray(simulator.model.keyframe("home").ctrl, dtype=np.float64).copy()
    previous_targets = np.asarray(
        data.qpos[simulator.evaluator.actuator_qpos_addr], dtype=np.float64
    ).copy()
    target_guard = routed.FinalTargetSafetyGuard(
        SAFE_JOINT_LIMITS,
        previous_targets,
        margin_rad=TARGET_MARGIN_RAD,
        max_slew_rate_rad_s=2.0,
    )
    previous_targets = target_guard.previous_targets
    action_history = [np.zeros(14, dtype=np.float32) for _ in range(3)]
    phase_index = 7.0
    control_ticks = int(round(float(seconds) / CONTROL_DT_S))
    rng = np.random.default_rng(seed + 810_000)
    warm_mean, warm_target = _initial_mean(
        simulator, data, previous_targets, action_history, phase_index
    )
    prior_ticks = np.repeat(warm_mean, block_ticks, axis=0)
    observations: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    applied_targets: list[np.ndarray] = []
    phases: list[float] = []
    local_velocities: list[np.ndarray] = []
    normal_forces: list[np.ndarray] = []
    tangential_speeds: list[np.ndarray] = []
    contacts: list[np.ndarray] = []
    heights: list[float] = []
    uprights: list[float] = []
    plan_records: list[dict[str, Any]] = []
    fell = False
    qpos_violation = False
    nonfinite = False
    no_feasible_plan = False
    max_guard_delta = 0.0
    start_position = np.asarray(
        data.xpos[simulator.evaluator.trunk_body_id], dtype=np.float64
    ).copy()

    for tick in range(control_ticks):
        h4._H4_RUN_CONTEXT["physical_command"] = PHYSICAL_COMMAND.copy()
        phase_angle = phase_index / float(simulator.evaluator.phase_steps) * 2.0 * np.pi
        observation = np.asarray(
            simulator.evaluator._observation(
                data,
                COMMAND7,
                default,
                previous_targets,
                action_history,
                phase_angle,
            ),
            dtype=np.float32,
        )
        if observation.shape != (116,):
            raise RuntimeError(f"teacher observation width drifted: {observation.shape}")
        best_z, plan_info = _cem_plan(
            simulator,
            data,
            previous_targets,
            prior_ticks,
            rng,
            horizon_ticks=horizon_ticks,
            block_ticks=block_ticks,
            population=population,
            elites=elites,
            iterations=iterations,
            warm_std=warm_std,
        )
        if not bool(plan_info["best_details"].get("valid", False)):
            no_feasible_plan = True
            plan_records.append(
                {
                    "tick": tick,
                    "phase_index": float(phase_index),
                    "best_cost": float(plan_info["best_cost"]),
                    "best_details": plan_info["best_details"],
                    "warm_start_only": tick == 0,
                    "executed": False,
                    "failure": "no_feasible_mpc_plan",
                }
            )
            break
        desired = previous_targets.copy()
        desired[LEG_INDICES] = previous_targets[LEG_INDICES] + TARGET_DELTA_RAD * np.tanh(
            best_z[0]
        )
        desired[HEAD_INDICES] = 0.0
        label_target = _exact_guard_step(simulator, previous_targets, desired)
        applied_before = target_guard.previous_targets
        applied = target_guard.step(desired, CONTROL_DT_S)
        if not np.allclose(applied, label_target, atol=1.0e-12, rtol=0.0):
            raise RuntimeError("CEM target and exact runtime guard diverged")
        data.ctrl[:] = applied
        simulator.mujoco.mj_step(
            simulator.model,
            data,
            nstep=int(simulator.runtime.DECIMATION),
        )
        action_label = _inverse_decoder(label_target)
        observations.append(observation)
        labels.append(action_label)
        applied_targets.append(applied.copy())
        phases.append(float(phase_index))
        action_history = [
            action_label.astype(np.float32),
            action_history[0].copy(),
            action_history[1].copy(),
        ]
        previous_targets = applied.copy()
        prior_ticks = _shift_prior(np.repeat(best_z, block_ticks, axis=0), horizon_ticks)
        plan_records.append(
            {
                "tick": tick,
                "phase_index": float(phase_index),
                "label_target": label_target.tolist(),
                "label_action": action_label.astype(float).tolist(),
                "best_cost": float(plan_info["best_cost"]),
                "best_details": plan_info["best_details"],
                "warm_start_only": tick == 0,
            }
        )
        max_guard_delta = max(
            max_guard_delta,
            float(np.max(np.abs(applied[LEG_INDICES] - applied_before[LEG_INDICES]))),
        )
        rotation = _trunk_rotation(simulator, data)
        position = np.asarray(
            data.xpos[simulator.evaluator.trunk_body_id], dtype=np.float64
        )
        local_velocities.append(
            np.asarray(simulator.evaluator._sensor(data, "local_linvel"), dtype=np.float64)
        )
        force, slip = simulator._quality_contact_kinematics(data)
        normal_forces.append(force)
        tangential_speeds.append(slip)
        contacts.append(np.asarray(simulator.evaluator._feet_contacts(data), dtype=bool))
        heights.append(float(position[2]))
        uprights.append(float(rotation[2, 2]))
        qpos = np.asarray(
            data.qpos[simulator.evaluator.actuator_qpos_addr], dtype=np.float64
        )
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            nonfinite = True
        qpos_violation = qpos_violation or bool(
            np.any(qpos[LEG_INDICES] < simulator.safe_lower[LEG_INDICES] - 1.0e-8)
            or np.any(qpos[LEG_INDICES] > simulator.safe_upper[LEG_INDICES] + 1.0e-8)
        )
        if heights[-1] < 0.12 or uprights[-1] < 0.65:
            fell = True
        phase_index = (phase_index + 0.81) % float(simulator.evaluator.phase_steps)
        if fell or qpos_violation or nonfinite:
            break

    velocities = np.asarray(local_velocities, dtype=np.float64)
    forces = np.asarray(normal_forces, dtype=np.float64)
    slips = np.asarray(tangential_speeds, dtype=np.float64)
    contact_array = np.asarray(contacts, dtype=bool)
    final_position = np.asarray(
        data.xpos[simulator.evaluator.trunk_body_id], dtype=np.float64
    )
    projected = -velocities[:, 0] if len(velocities) else np.zeros(1)
    steady_start_index = int(1.5 / CONTROL_DT_S)
    steady_projected = (
        projected[steady_start_index:]
        if len(projected) > steady_start_index
        else projected
    )
    force_total = forces.sum(axis=1) if len(forces) else np.zeros(1)
    metrics = {
        "seed": int(seed),
        "requested_seconds": float(seconds),
        "completed_ticks": len(labels),
        "expected_ticks": control_ticks,
        "completed": len(labels) == control_ticks and not fell and not qpos_violation and not nonfinite,
        "fell": fell,
        "qpos_violation": qpos_violation,
        "nonfinite": nonfinite,
        "no_feasible_mpc_plan": no_feasible_plan,
        "reset_qpos_audit": reset_audit,
        "projected_speed_mean_mps": float(np.mean(projected)),
        "projected_speed_steady_mps": float(
            np.mean(steady_projected)
        ),
        "net_displacement_world_m": (final_position - start_position).tolist(),
        "height_min_m": float(min(heights)) if heights else float(final_position[2]),
        "upright_min": float(min(uprights)) if uprights else 0.0,
        "flight_rate_control_samples": float(np.mean(contact_array.sum(axis=1) == 0)) if len(contact_array) else 1.0,
        "single_support_rate_control_samples": float(np.mean(np.logical_xor(contact_array[:, 0], contact_array[:, 1]))) if len(contact_array) else 0.0,
        "left_contact_rate": float(np.mean(contact_array[:, 0])) if len(contact_array) else 0.0,
        "right_contact_rate": float(np.mean(contact_array[:, 1])) if len(contact_array) else 0.0,
        "slip_rms_mps": float(np.sqrt(np.mean(slips**2))) if len(slips) else None,
        "slip_p95_mps": float(np.percentile(slips, 95)) if len(slips) else None,
        "force_p99_body_weight": float(np.percentile(force_total, 99)) if len(force_total) else None,
        "max_guard_delta_rad": max_guard_delta,
        "warm_start_target_not_label": warm_target.tolist(),
        "strict_three_seed_screen": "NOT_RUN",
        "adoption_allowed": False,
        "release_allowed": False,
        "hardware_deployment": "PROHIBITED",
    }
    arrays = {
        "observations": np.asarray(observations, dtype=np.float32),
        "target_actions": np.asarray(labels, dtype=np.float32),
        "applied_targets": np.asarray(applied_targets, dtype=np.float32),
        "phase_indices": np.asarray(phases, dtype=np.float32),
        "local_velocities": velocities.astype(np.float32),
        "normal_forces": forces.astype(np.float32),
        "tangential_speeds": slips.astype(np.float32),
        "contacts": contact_array.astype(np.uint8),
        "seed": np.full(len(labels), int(seed), dtype=np.int64),
    }
    return {"metrics": metrics, "plans": plan_records}, arrays


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument(
        "--planar-manifest", type=Path, default=DEFAULT_PLANAR_MANIFEST
    )
    parser.add_argument(
        "--reverse-manifest", type=Path, default=DEFAULT_REVERSE_MANIFEST
    )
    parser.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--horizon-ticks", type=int, default=DEFAULT_HORIZON_TICKS)
    parser.add_argument("--block-ticks", type=int, default=DEFAULT_BLOCK_TICKS)
    parser.add_argument("--population", type=int, default=DEFAULT_POPULATION)
    parser.add_argument("--elites", type=int, default=DEFAULT_ELITES)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--warm-std", type=float, default=DEFAULT_WARM_STD)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a short non-evidence smoke configuration for script validation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = tuple(args.seeds) if args.seeds else DEFAULT_SEED
    if args.smoke:
        args.seconds = min(args.seconds, 0.20)
        args.horizon_ticks = min(args.horizon_ticks, 10)
        args.block_ticks = min(args.block_ticks, 5)
        args.population = min(args.population, 16)
        args.elites = min(args.elites, 4)
        args.iterations = min(args.iterations, 1)
    if args.seconds <= 0.0 or args.population <= 0 or args.elites <= 0 or args.iterations <= 0:
        raise ValueError("seconds, population, elites, and iterations must be positive")
    if args.horizon_ticks <= 0 or args.block_ticks <= 0 or args.horizon_ticks % args.block_ticks:
        raise ValueError("horizon_ticks must be a positive multiple of block_ticks")
    if args.elites > args.population:
        raise ValueError("elites cannot exceed population")
    params = args.params.expanduser().resolve()
    planar_manifest = args.planar_manifest.expanduser().resolve()
    reverse_manifest = args.reverse_manifest.expanduser().resolve()
    generated_root = args.generated_root.expanduser().resolve()
    output_prefix = args.output_prefix.expanduser().resolve()
    npz_path = output_prefix.with_suffix(".npz")
    json_path = output_prefix.with_suffix(".json")
    if npz_path.exists() or json_path.exists():
        raise FileExistsError(f"refusing to overwrite teacher artifacts: {npz_path} / {json_path}")
    if (
        not params.is_file()
        or not planar_manifest.is_file()
        or not reverse_manifest.is_file()
        or not BASE_V22.is_file()
    ):
        raise FileNotFoundError("teacher inputs are missing")

    simulator, bank, build_metadata = h5._build_simulator(
        _build_args(params, planar_manifest, reverse_manifest, generated_root)
    )
    seed_records: list[dict[str, Any]] = []
    aggregate: dict[str, list[np.ndarray]] = {}
    for seed in seeds:
        record, arrays = _run_seed(
            simulator,
            seed=int(seed),
            seconds=args.seconds,
            horizon_ticks=args.horizon_ticks,
            block_ticks=args.block_ticks,
            population=args.population,
            elites=args.elites,
            iterations=args.iterations,
            warm_std=args.warm_std,
        )
        seed_records.append(record)
        for key, value in arrays.items():
            aggregate.setdefault(key, []).append(value)
    if bank.legacy_fallback_count != 0:
        raise RuntimeError("MPC teacher unexpectedly used an H5 legacy fallback")
    merged = {
        key: np.concatenate(values, axis=0) if values and values[0].ndim else np.asarray(values)
        for key, values in aggregate.items()
    }
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **merged)
    payload = {
        "schema_version": 1,
        "evaluator_id": "openduckmini-exp004-h5-mpc-reverse-teacher-v1",
        "evaluation_mode": "DIAGNOSTIC_MPC_TEACHER_NOT_QUALIFIED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_deployment": "PROHIBITED",
        "configuration": {
            "seeds": list(seeds),
            "seconds": float(args.seconds),
            "control_dt_s": CONTROL_DT_S,
            "horizon_ticks": int(args.horizon_ticks),
            "horizon_seconds": float(args.horizon_ticks * CONTROL_DT_S),
            "block_ticks": int(args.block_ticks),
            "population": int(args.population),
            "elites": int(args.elites),
            "iterations": int(args.iterations),
            "warm_std": float(args.warm_std),
            "physical_command": PHYSICAL_COMMAND.tolist(),
            "policy_command": POLICY_COMMAND.tolist(),
            "phase_contract": {
                "source_phase_start": 7.0,
                "source_phase_delta_per_control": 0.81,
                "source_period_bins": 27,
                "decoder_table_coordinate": "q=(2*p)%54",
            },
            "target_parameterization": {
                "description": "absolute target = exact_guard(previous + 0.04*tanh(z))",
                "leg_indices": LEG_INDICES.tolist(),
                "head_indices": HEAD_INDICES.tolist(),
                "margin_rad": TARGET_MARGIN_RAD,
                "max_delta_per_tick_rad": TARGET_DELTA_RAD,
                "final_guard_owner": "target_safety.FinalTargetSafetyGuard",
            },
            "cem_objective": {
                "physical_vx_tracking_weight": 10.0,
                "combined_and_per_foot_slip_weight": 2.0,
                "slip_tail_weight": 1.0,
                "support_weight": 10.0,
                "alternation_proxy_weight": 1.0,
                "force_band_and_tail_weight": 1.0,
                "terminal_upright_weight": 1.0,
            },
            "warm_start": {
                "source": "legacy_v22_forward_sagittal_mirror_diagnostic_only",
                "phase_rate": 0.75,
                "phase_offset": 13.5,
                "used_for_labels": False,
            },
        },
        "provenance": {
            "h5_build_metadata": build_metadata,
            "teacher_script": str(Path(__file__).resolve()),
            "teacher_script_sha256": h5.sha256_file(Path(__file__).resolve()),
            "params": {"path": str(params), "sha256": h5.sha256_file(params)},
            "manifests": {
                "planar": {
                    "path": str(planar_manifest),
                    "sha256": h5.sha256_file(planar_manifest),
                },
                "reverse": {
                    "path": str(reverse_manifest),
                    "sha256": h5.sha256_file(reverse_manifest),
                },
            },
            "base_v22": {"path": str(BASE_V22), "sha256": h5.sha256_file(BASE_V22)},
            "policy_bank": bank.manifest(),
        },
        "seeds": _json_safe(seed_records),
        "outputs": {
            "npz": {"path": str(npz_path), "sha256": h5.sha256_file(npz_path)},
            "json": {"path": str(json_path)},
            "observation_width": 116,
            "label_action_width": 14,
        },
        "strict_three_seed_screen": {
            "status": "NOT_RUN",
            "passed": False,
            "adoption_allowed": False,
            "release_allowed": False,
        },
    }
    json_path.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "json": str(json_path),
                "npz": str(npz_path),
                "seeds": list(seeds),
                "completed_ticks": [record["metrics"]["completed_ticks"] for record in seed_records],
                "fell": [record["metrics"]["fell"] for record in seed_records],
                "hardware_deployment": "PROHIBITED",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
