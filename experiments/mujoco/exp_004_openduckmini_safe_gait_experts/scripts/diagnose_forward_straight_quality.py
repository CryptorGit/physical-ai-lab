"""Isolated PDCA harness for strict straight-forward gait quality.

This diagnostic never edits the routed evaluator, router, package, or hardware
assets.  It changes only the policy-visible command (and optionally one
forward-only phase-entry index) in process, while the physical command, final
target guard, and every central physics-substep safety audit stay unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = EXP_ROOT.parents[2]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from scripts import evaluate_routed_transitions as central  # noqa: E402
from safe_gait_experts.routed_evaluation import segment_acceptance  # noqa: E402


BASE_POLICY = (
    WORKSPACE / ".openduck_runtime_source_review" / "calibrated_hybrid_policy_v22.onnx"
).resolve()
V2_MANIFEST = (
    WORKSPACE / "media" / "openduck_exp004_h3_release" / "video_manifest_v2_final.json"
).resolve()
DEFAULT_OUTPUT = EXP_ROOT / "artifacts" / "forward_straight_quality_diagnostic.json"
PHYSICAL_COMMAND = (0.05, 0.0, 0.0)
EXPECTED_EXPERT = "forward"
EXPECTED_POLICY_ROLE = "forward"
V2_SEED = 20_260_808
FORMAL_FORWARD_SEED_BASE = 20_260_809
SLIP_CAUSAL_RUNS = (
    {
        "run_id": "median_force_weighted_slip_seed01",
        "seed": 20_261_809,
        "joint_noise_scale": 1.0,
        "initial_base_speed": 0.10,
        "exact_home": False,
        "selection": "median combined stance-slip RMS among selected 5x15 v3",
    },
    {
        "run_id": "worst_force_weighted_slip_seed02",
        "seed": 20_262_809,
        "joint_noise_scale": 1.0,
        "initial_base_speed": 0.10,
        "exact_home": False,
        "selection": "maximum combined stance-slip RMS and p95 among selected 5x15 v3",
    },
)


STRICT_GATES: Mapping[str, float] = {
    "minimum_vx_tracking_ratio": 0.75,
    "maximum_vx_tracking_ratio": 1.25,
    "maximum_cross_velocity_mps": min(0.012, 0.20 * PHYSICAL_COMMAND[0]),
    "maximum_uncommanded_yaw_rate_rad_s": 0.05,
    "maximum_six_second_heading_change_rad": 0.15,
    "minimum_single_support_rate": 0.25,
    "maximum_single_support_rate": 0.60,
    "maximum_flight_rate": 0.01,
}

PROVISIONAL_SLIP_GATES: Mapping[str, float] = {
    "maximum_contact_point_tangential_rms_mps": 0.015,
    "maximum_contact_point_tangential_p95_mps": 0.030,
    "maximum_integrated_slip_proxy_per_stance_m": 0.020,
}

OBSERVATION_BLOCKS: Mapping[str, tuple[int, int]] = {
    "gyro": (0, 3),
    "accelerometer": (3, 6),
    "command": (6, 13),
    "joint_position": (13, 27),
    "joint_velocity": (27, 41),
    "action_history_0": (41, 55),
    "action_history_1": (55, 69),
    "action_history_2": (69, 83),
    "motor_targets": (83, 97),
    "contacts": (97, 99),
    "phase": (99, 101),
}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    policy_observation_command: tuple[float, float, float]
    forward_entry_preincrement_phase: float | None = None
    forward_left_knee_target_scale: float = 1.0
    forward_left_knee_target_bias_rad: float = 0.0
    forward_target_scale_overrides: Mapping[str, float] = field(default_factory=dict)
    forward_target_bias_overrides_rad: Mapping[str, float] = field(default_factory=dict)
    note: str = ""

    def validate(self, phase_steps: float | None = None) -> None:
        command = np.asarray(self.policy_observation_command, dtype=np.float64)
        if not self.candidate_id or command.shape != (3,) or not np.all(np.isfinite(command)):
            raise ValueError("candidate id and finite three-axis policy command are required")
        if self.forward_entry_preincrement_phase is not None:
            phase = float(self.forward_entry_preincrement_phase)
            if not np.isfinite(phase) or phase < 0.0:
                raise ValueError("forward entry phase must be finite and non-negative")
            if phase_steps is not None and phase >= float(phase_steps):
                raise ValueError("forward entry phase must be below phase_steps")
        scale = float(self.forward_left_knee_target_scale)
        bias = float(self.forward_left_knee_target_bias_rad)
        if not np.isfinite(scale) or not 0.0 < scale <= 1.25:
            raise ValueError("forward left-knee scale must be finite and in (0, 1.25]")
        if not np.isfinite(bias) or not -0.20 <= bias <= 0.05:
            raise ValueError("forward left-knee bias must be finite and in [-0.20, 0.05]")
        known = set(central.ACTUATOR_JOINT_ORDER)
        scale_overrides = dict(self.forward_target_scale_overrides)
        bias_overrides = dict(self.forward_target_bias_overrides_rad)
        if not set(scale_overrides) <= known or not set(bias_overrides) <= known:
            raise ValueError("forward target overrides contain an unknown actuator")
        if "left_knee" in scale_overrides or "left_knee" in bias_overrides:
            if scale != 1.0 or bias != 0.0:
                raise ValueError("left-knee legacy and general target transforms may not overlap")
        if any(
            not np.isfinite(float(value)) or not 0.0 < float(value) <= 1.25
            for value in scale_overrides.values()
        ):
            raise ValueError("forward target scale overrides must be finite and in (0, 1.25]")
        if any(
            not np.isfinite(float(value)) or not -0.15 <= float(value) <= 0.15
            for value in bias_overrides.values()
        ):
            raise ValueError("forward target bias overrides must be finite and in [-0.15, 0.15]")


TRACE_CANDIDATES = (
    Candidate("v2_baseline", (0.10, 0.0, 0.0), note="exact V2 mapping"),
    Candidate("probe_vy_minus", (0.10, -0.03, 0.0), note="negative vy finite difference"),
    Candidate("probe_vy_plus", (0.10, 0.03, 0.0), note="positive vy finite difference"),
    Candidate("probe_yaw_minus", (0.10, 0.0, -0.15), note="negative yaw finite difference"),
    Candidate("probe_yaw_plus", (0.10, 0.0, 0.15), note="positive yaw finite difference"),
    Candidate(
        "probe_combined_minus",
        (0.10, -0.03, -0.15),
        note="combined negative vy/yaw probe",
    ),
    Candidate(
        "probe_knee_bias_m050",
        (0.10, 0.0, 0.0),
        forward_left_knee_target_bias_rad=-0.05,
        note="forward-only pre-guard left-knee bias; no command change",
    ),
    Candidate(
        "probe_knee_compress060_bias_m050",
        (0.10, 0.0, 0.0),
        forward_left_knee_target_scale=0.60,
        forward_left_knee_target_bias_rad=-0.05,
        note="forward-only compression about home plus bias; no command change",
    ),
    Candidate(
        "probe_combined_command_knee",
        (0.10, -0.03, -0.15),
        forward_left_knee_target_scale=0.60,
        forward_left_knee_target_bias_rad=-0.05,
        note="combined command and forward-only left-knee causal probe",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_finite_copy(value: Any, replacements: list[str], path: str = "$") -> Any:
    """Make undefined diagnostic statistics explicit without hiding their paths."""
    if isinstance(value, Mapping):
        return {
            str(key): json_finite_copy(item, replacements, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            json_finite_copy(item, replacements, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        replacements.append(path)
        return None
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "trace",
            "grid",
            "screen",
            "formal",
            "causal",
            "slip-screen",
            "median-screen",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--candidates-json",
        type=Path,
        help="JSON object containing a candidates array; optional only for trace stage.",
    )
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--warmup-seconds", type=float, default=1.5)
    parser.add_argument(
        "--require-provisional-slip-gates",
        action="store_true",
        help="Make the provisional kinematic foot-slip checks promotion-blocking.",
    )
    args = parser.parse_args(argv)
    defaults = {
        "trace": 6.0,
        "grid": 6.0,
        "screen": 15.0,
        "formal": 30.0,
        "causal": 15.0,
        "slip-screen": 15.0,
        "median-screen": 15.0,
    }
    args.seconds = defaults[args.stage] if args.seconds is None else args.seconds
    if args.seconds <= 0.0 or not 0.0 <= args.warmup_seconds < args.seconds:
        parser.error("seconds must be positive and warmup must be in [0, seconds)")
    if args.stage != "trace" and args.candidates_json is None:
        parser.error("non-trace stages require --candidates-json")
    return args


def load_candidates(path: Path | None, *, stage: str) -> tuple[Candidate, ...]:
    if path is None:
        if stage != "trace":
            raise ValueError("only trace stage has built-in candidates")
        return TRACE_CANDIDATES
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    rows = payload.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidate spec must contain a non-empty candidates array")
    candidates = tuple(
        Candidate(
            candidate_id=str(row["candidate_id"]),
            policy_observation_command=tuple(
                float(value) for value in row["policy_observation_command"]
            ),
            forward_entry_preincrement_phase=(
                None
                if row.get("forward_entry_preincrement_phase") is None
                else float(row["forward_entry_preincrement_phase"])
            ),
            forward_left_knee_target_scale=float(
                row.get("forward_left_knee_target_scale", 1.0)
            ),
            forward_left_knee_target_bias_rad=float(
                row.get("forward_left_knee_target_bias_rad", 0.0)
            ),
            forward_target_scale_overrides={
                str(name): float(value)
                for name, value in row.get("forward_target_scale_overrides", {}).items()
            },
            forward_target_bias_overrides_rad={
                str(name): float(value)
                for name, value in row.get(
                    "forward_target_bias_overrides_rad", {}
                ).items()
            },
            note=str(row.get("note", "")),
        )
        for row in rows
    )
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("candidate ids must be unique")
    return candidates


def stage_runs(stage: str) -> tuple[dict[str, Any], ...]:
    if stage == "slip-screen":
        return (dict(SLIP_CAUSAL_RUNS[1]),)
    if stage == "median-screen":
        return (dict(SLIP_CAUSAL_RUNS[0]),)
    if stage == "causal":
        return tuple(dict(run) for run in SLIP_CAUSAL_RUNS)
    if stage in {"trace", "grid"}:
        return (
            {
                "run_id": "exact_home_v2_seed20260808",
                "seed": V2_SEED,
                "joint_noise_scale": 0.0,
                "initial_base_speed": 0.0,
                "exact_home": True,
            },
        )
    count = 5 if stage == "screen" else 20
    return tuple(
        {
            "run_id": f"formal_perturbed_{index:02d}",
            "seed": FORMAL_FORWARD_SEED_BASE + index * 1000,
            "joint_noise_scale": 1.0,
            "initial_base_speed": 0.10,
            "exact_home": False,
        }
        for index in range(count)
    )


def yaw_from_data(data: Any, trunk_body_id: int) -> float:
    rotation = np.asarray(data.xmat[trunk_body_id], dtype=np.float64).reshape(3, 3)
    return float(math.atan2(rotation[1, 0], rotation[0, 0]))


def wrapped_delta(current: float, previous: float) -> float:
    return float(math.atan2(math.sin(current - previous), math.cos(current - previous)))


class RunTrace:
    def __init__(
        self,
        evaluator: Any,
        mujoco: Any,
        candidate: Candidate,
        *,
        sim_dt: float,
        capture_trace: bool,
    ) -> None:
        self.evaluator = evaluator
        self.mujoco = mujoco
        self.candidate = candidate
        self.sim_dt = float(sim_dt)
        self.capture_trace = bool(capture_trace)
        self.current_decision: Any | None = None
        self.current_requested_command: list[float] | None = None
        self.control_rows: list[dict[str, Any]] = []
        self.substep_rows: list[dict[str, Any]] = []
        self.substep_count = 0
        self.previous_forward_active = False
        self.phase_entry_events: list[dict[str, Any]] = []
        self._last_phase_reset = False
        self.initial_yaw: float | None = None
        self.previous_raw_yaw: float | None = None
        self.unwrapped_heading = 0.0
        self.heading_samples: list[float] = []
        self.contact_point_speeds: dict[str, list[float]] = {"left": [], "right": []}
        self.contact_point_forces: dict[str, list[float]] = {"left": [], "right": []}
        self.force_weighted_speed_sum: dict[str, float] = {"left": 0.0, "right": 0.0}
        self.force_sum: dict[str, float] = {"left": 0.0, "right": 0.0}
        self.current_stance_distance: dict[str, float] = {"left": 0.0, "right": 0.0}
        self.stance_distances: dict[str, list[float]] = {"left": [], "right": []}
        self.previous_contact: dict[str, bool] = {"left": False, "right": False}

    def initialize_heading(self, data: Any) -> None:
        if self.initial_yaw is not None:
            return
        raw = yaw_from_data(data, self.evaluator.trunk_body_id)
        self.initial_yaw = raw
        self.previous_raw_yaw = raw

    def _contact_kinematics(self, data: Any) -> dict[str, dict[str, Any]]:
        by_foot: dict[str, list[tuple[float, float]]] = {"left": [], "right": []}
        body_to_label = {
            int(self.evaluator.left_foot_body_id): "left",
            int(self.evaluator.right_foot_body_id): "right",
        }
        spatial_cache: dict[int, np.ndarray] = {}
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            body_1 = int(self.evaluator.model.geom_bodyid[contact.geom1])
            body_2 = int(self.evaluator.model.geom_bodyid[contact.geom2])
            if body_1 == int(self.evaluator.floor_body_id) and body_2 in body_to_label:
                foot_body = body_2
            elif body_2 == int(self.evaluator.floor_body_id) and body_1 in body_to_label:
                foot_body = body_1
            else:
                continue
            if foot_body not in spatial_cache:
                spatial = np.zeros(6, dtype=np.float64)
                self.mujoco.mj_objectVelocity(
                    self.evaluator.model,
                    data,
                    self.mujoco.mjtObj.mjOBJ_BODY,
                    foot_body,
                    spatial,
                    0,
                )
                spatial_cache[foot_body] = spatial
            spatial = spatial_cache[foot_body]
            offset = np.asarray(contact.pos, dtype=np.float64) - np.asarray(
                data.xpos[foot_body], dtype=np.float64
            )
            point_velocity = spatial[3:] + np.cross(spatial[:3], offset)
            normal = np.asarray(contact.frame[:3], dtype=np.float64)
            tangential = point_velocity - float(np.dot(point_velocity, normal)) * normal
            tangential_speed = float(np.linalg.norm(tangential))
            force = np.zeros(6, dtype=np.float64)
            self.mujoco.mj_contactForce(self.evaluator.model, data, index, force)
            normal_force = abs(float(force[0]))
            by_foot[body_to_label[foot_body]].append((tangential_speed, normal_force))
        result: dict[str, dict[str, Any]] = {}
        for label, samples in by_foot.items():
            speeds = np.asarray([sample[0] for sample in samples], dtype=np.float64)
            forces = np.asarray([sample[1] for sample in samples], dtype=np.float64)
            result[label] = {
                "contact": bool(samples),
                "contact_point_count": len(samples),
                "tangential_speed_rms_mps": (
                    None if not samples else float(np.sqrt(np.mean(np.square(speeds))))
                ),
                "tangential_speed_max_mps": None if not samples else float(np.max(speeds)),
                "normal_force_sum_n": float(np.sum(forces)),
                "force_weighted_tangential_speed_mps": (
                    None
                    if not samples or float(np.sum(forces)) <= 0.0
                    else float(np.sum(speeds * forces) / np.sum(forces))
                ),
                "point_speeds": speeds.tolist(),
                "point_forces": forces.tolist(),
            }
        return result

    def sample_substep(self, data: Any, control_tick: int) -> None:
        self.initialize_heading(data)
        raw_yaw = yaw_from_data(data, self.evaluator.trunk_body_id)
        assert self.previous_raw_yaw is not None
        self.unwrapped_heading += wrapped_delta(raw_yaw, self.previous_raw_yaw)
        self.previous_raw_yaw = raw_yaw
        self.heading_samples.append(self.unwrapped_heading)
        feet = self._contact_kinematics(data)
        for label in ("left", "right"):
            record = feet[label]
            contact = bool(record["contact"])
            if contact:
                speeds = record["point_speeds"]
                forces = record["point_forces"]
                self.contact_point_speeds[label].extend(float(value) for value in speeds)
                self.contact_point_forces[label].extend(float(value) for value in forces)
                self.force_weighted_speed_sum[label] += sum(
                    float(speed) * float(force) for speed, force in zip(speeds, forces)
                )
                self.force_sum[label] += sum(float(force) for force in forces)
                rms = float(record["tangential_speed_rms_mps"])
                self.current_stance_distance[label] += rms * self.sim_dt
            elif self.previous_contact[label]:
                self.stance_distances[label].append(self.current_stance_distance[label])
                self.current_stance_distance[label] = 0.0
            self.previous_contact[label] = contact
        if self.capture_trace:
            control_row = self.control_rows[control_tick]
            trunk_velocity = np.zeros(6, dtype=np.float64)
            self.mujoco.mj_objectVelocity(
                self.evaluator.model,
                data,
                self.mujoco.mjtObj.mjOBJ_BODY,
                int(self.evaluator.trunk_body_id),
                trunk_velocity,
                1,
            )
            self.substep_rows.append(
                {
                    "physics_substep": self.substep_count,
                    "control_tick": int(control_tick),
                    "time_seconds": float(data.time),
                    "phase_index": float(control_row["phase_index"]),
                    "heading_change_rad": self.unwrapped_heading,
                    "local_trunk_linear_velocity_mps": trunk_velocity[3:].tolist(),
                    "joint_qpos_rad": np.asarray(
                        data.qpos[self.evaluator.actuator_qpos_addr], dtype=float
                    ).tolist(),
                    "joint_qvel_rad_s": np.asarray(
                        data.qvel[self.evaluator.actuator_qvel_addr], dtype=float
                    ).tolist(),
                    "left": dict(feet["left"]),
                    "right": dict(feet["right"]),
                }
            )
        self.substep_count += 1

    def finalize_stances(self) -> None:
        for label in ("left", "right"):
            if self.previous_contact[label]:
                self.stance_distances[label].append(self.current_stance_distance[label])
                self.current_stance_distance[label] = 0.0
                self.previous_contact[label] = False

    def heading_summary(self) -> dict[str, Any]:
        values = np.asarray(self.heading_samples, dtype=np.float64)
        if not len(values):
            return {
                "sample_count": 0,
                "total_heading_change_rad": None,
                "maximum_absolute_heading_change_rad": None,
                "maximum_rolling_six_second_heading_change_rad": None,
            }
        window = int(round(6.0 / self.sim_dt))
        rolling = (
            np.abs(values[window:] - values[:-window])
            if len(values) > window
            else np.asarray([abs(float(values[-1]))], dtype=np.float64)
        )
        return {
            "sample_count": len(values),
            "total_heading_change_rad": float(values[-1]),
            "maximum_absolute_heading_change_rad": float(np.max(np.abs(values))),
            "maximum_rolling_six_second_heading_change_rad": float(np.max(rolling)),
        }

    def slip_summary(self) -> dict[str, Any]:
        self.finalize_stances()
        feet: dict[str, Any] = {}
        all_speeds: list[float] = []
        all_stances: list[float] = []
        for label in ("left", "right"):
            speeds = np.asarray(self.contact_point_speeds[label], dtype=np.float64)
            stances = self.stance_distances[label]
            all_speeds.extend(float(value) for value in speeds)
            all_stances.extend(float(value) for value in stances)
            feet[label] = {
                "contact_point_sample_count": len(speeds),
                "tangential_speed_rms_mps": (
                    0.0 if not len(speeds) else float(np.sqrt(np.mean(np.square(speeds))))
                ),
                "tangential_speed_p95_mps": (
                    0.0 if not len(speeds) else float(np.percentile(speeds, 95.0))
                ),
                "tangential_speed_max_mps": 0.0 if not len(speeds) else float(np.max(speeds)),
                "force_weighted_mean_tangential_speed_mps": (
                    0.0
                    if self.force_sum[label] == 0.0
                    else self.force_weighted_speed_sum[label] / self.force_sum[label]
                ),
                "stance_count": len(stances),
                "maximum_integrated_slip_proxy_per_stance_m": max(stances, default=0.0),
                "total_integrated_slip_proxy_m": float(sum(stances)),
            }
        speeds = np.asarray(all_speeds, dtype=np.float64)
        combined = {
            "contact_point_sample_count": len(speeds),
            "tangential_speed_rms_mps": (
                0.0 if not len(speeds) else float(np.sqrt(np.mean(np.square(speeds))))
            ),
            "tangential_speed_p95_mps": (
                0.0 if not len(speeds) else float(np.percentile(speeds, 95.0))
            ),
            "tangential_speed_max_mps": 0.0 if not len(speeds) else float(np.max(speeds)),
            "maximum_integrated_slip_proxy_per_stance_m": max(all_stances, default=0.0),
        }
        return {"measurement": "world-contact-point tangential kinematics", "feet": feet, "combined": combined}

    def target_chain_summary(self) -> dict[str, Any]:
        """Audit the route-local knee transform before the unchanged final guard."""
        if not self.control_rows:
            return {"control_tick_count": 0}
        left_index = int(self.evaluator.model.actuator("left_knee").id)
        right_index = int(self.evaluator.model.actuator("right_knee").id)
        upper = float(self.evaluator.model.jnt_range[
            self.evaluator.actuator_joint_ids[left_index], 1
        ])
        # The central simulator exposes the exact final target envelope through
        # each trace row, avoiding a duplicated safety constant here.
        margin_upper = float(
            self.control_rows[0]["forward_left_knee_transform"]["target_upper_rad"]
        )
        raw_left = np.asarray(
            [row["policy_targets_before_forward_knee_transform_rad"][left_index] for row in self.control_rows],
            dtype=np.float64,
        )
        transformed_left = np.asarray(
            [row["candidate_targets_rad"][left_index] for row in self.control_rows],
            dtype=np.float64,
        )
        applied_left = np.asarray(
            [row["applied_targets_rad"][left_index] for row in self.control_rows],
            dtype=np.float64,
        )
        raw_right = np.asarray(
            [row["policy_targets_before_forward_knee_transform_rad"][right_index] for row in self.control_rows],
            dtype=np.float64,
        )
        applied_right = np.asarray(
            [row["applied_targets_rad"][right_index] for row in self.control_rows],
            dtype=np.float64,
        )
        return {
            "control_tick_count": len(self.control_rows),
            "left_knee_safe_upper_rad": upper,
            "left_knee_target_margin_upper_rad": margin_upper,
            "left_knee": {
                "raw_policy_target_minimum_rad": float(np.min(raw_left)),
                "raw_policy_target_maximum_rad": float(np.max(raw_left)),
                "raw_policy_target_above_margin_upper_ticks": int(np.sum(raw_left > margin_upper)),
                "raw_policy_target_above_margin_upper_rate": float(np.mean(raw_left > margin_upper)),
                "transformed_target_minimum_rad": float(np.min(transformed_left)),
                "transformed_target_maximum_rad": float(np.max(transformed_left)),
                "transformed_target_at_margin_upper_ticks": int(
                    np.sum(np.isclose(transformed_left, margin_upper, atol=1e-12, rtol=0.0))
                ),
                "applied_target_minimum_rad": float(np.min(applied_left)),
                "applied_target_maximum_rad": float(np.max(applied_left)),
                "applied_target_at_margin_upper_ticks": int(
                    np.sum(np.isclose(applied_left, margin_upper, atol=1e-12, rtol=0.0))
                ),
            },
            "right_knee": {
                "raw_policy_target_minimum_rad": float(np.min(raw_right)),
                "raw_policy_target_maximum_rad": float(np.max(raw_right)),
                "applied_target_minimum_rad": float(np.min(applied_right)),
                "applied_target_maximum_rad": float(np.max(applied_right)),
            },
        }


def safety_zero_checks(segment: Mapping[str, Any]) -> dict[str, bool]:
    safety = segment["safety_audit"]
    physics = segment["physics_substep_audit"]
    return {
        "completed": bool(segment["completed"]),
        "no_fall": not bool(segment["fell"]),
        "preclip_target_limit_violations_zero": int(safety["preclip_target_limit_violations"]) == 0,
        "applied_target_limit_violations_zero": int(safety["applied_target_limit_violations"]) == 0,
        "unauthorized_target_margin_violations_zero": int(
            safety["unauthorized_applied_target_margin_violations"]
        )
        == 0,
        "target_slew_violations_zero": int(safety["target_slew_violations"]) == 0,
        "control_qpos_violations_zero": int(safety["qpos_limit_violations"]) == 0,
        "control_nonfinite_samples_zero": int(safety["nonfinite_sample_count"]) == 0,
        "substep_qpos_violations_zero": int(physics["qpos_limit_violations"]) == 0,
        "substep_nonfinite_samples_zero": all(
            int(physics[key]) == 0
            for key in (
                "nonfinite_full_qpos_samples",
                "nonfinite_full_qvel_samples",
                "nonfinite_leg_qpos_samples",
                "nonfinite_pose_samples",
                "nonfinite_state_samples",
            )
        ),
        "substep_no_fall": not bool(physics["fall_or_nonfinite_detected"]),
        "all_completed_substeps_audited": int(physics["sample_count"])
        == int(segment["completed_physics_substeps"]),
        "all_expected_substeps_completed": int(physics["sample_count"])
        == int(segment["expected_physics_substeps"]),
    }


def strict_quality(
    segment: Mapping[str, Any],
    heading: Mapping[str, Any],
    slip: Mapping[str, Any],
    *,
    seconds: float,
    require_slip: bool,
) -> dict[str, Any]:
    metrics = segment["metrics"]
    vx_ratio = float(metrics["projected_primary_velocity"]) / PHYSICAL_COMMAND[0]
    total_heading = abs(float(heading["total_heading_change_rad"]))
    scaled_total_limit = STRICT_GATES["maximum_six_second_heading_change_rad"] * max(
        1.0, seconds / 6.0
    )
    checks = {
        "vx_tracking": STRICT_GATES["minimum_vx_tracking_ratio"]
        <= vx_ratio
        <= STRICT_GATES["maximum_vx_tracking_ratio"],
        "cross_velocity": float(metrics["absolute_orthogonal_velocity"])
        <= STRICT_GATES["maximum_cross_velocity_mps"],
        "uncommanded_yaw_rate": abs(float(metrics["mean_local_yaw_rate"]))
        <= STRICT_GATES["maximum_uncommanded_yaw_rate_rad_s"],
        "rolling_six_second_heading": float(
            heading["maximum_rolling_six_second_heading_change_rad"]
        )
        <= STRICT_GATES["maximum_six_second_heading_change_rad"],
        "duration_scaled_total_heading": total_heading <= scaled_total_limit,
        "single_support_lower": float(metrics["single_support_rate"])
        >= STRICT_GATES["minimum_single_support_rate"],
        "single_support_upper": float(metrics["single_support_rate"])
        <= STRICT_GATES["maximum_single_support_rate"],
        "flight_rate": float(metrics["flight_rate"]) <= STRICT_GATES["maximum_flight_rate"],
    }
    safety_checks = safety_zero_checks(segment)
    combined_slip = slip["combined"]
    slip_checks = {
        "tangential_rms": float(combined_slip["tangential_speed_rms_mps"])
        <= PROVISIONAL_SLIP_GATES["maximum_contact_point_tangential_rms_mps"],
        "tangential_p95": float(combined_slip["tangential_speed_p95_mps"])
        <= PROVISIONAL_SLIP_GATES["maximum_contact_point_tangential_p95_mps"],
        "integrated_per_stance": float(
            combined_slip["maximum_integrated_slip_proxy_per_stance_m"]
        )
        <= PROVISIONAL_SLIP_GATES["maximum_integrated_slip_proxy_per_stance_m"],
    }
    passed = all(checks.values()) and all(safety_checks.values())
    if require_slip:
        passed = passed and all(slip_checks.values())
    return {
        "passed": bool(passed),
        "checks": checks,
        "safety_zero_checks": safety_checks,
        "provisional_slip_checks": slip_checks,
        "provisional_slip_checks_required": bool(require_slip),
        "values": {
            "vx_tracking_ratio": vx_ratio,
            "signed_cross_velocity_mps": float(metrics["signed_orthogonal_velocity"]),
            "cross_velocity_mps": float(metrics["absolute_orthogonal_velocity"]),
            "signed_uncommanded_yaw_rate_rad_s": float(metrics["mean_local_yaw_rate"]),
            "uncommanded_yaw_rate_rad_s": abs(float(metrics["mean_local_yaw_rate"])),
            "total_heading_change_rad": float(heading["total_heading_change_rad"]),
            "maximum_rolling_six_second_heading_change_rad": float(
                heading["maximum_rolling_six_second_heading_change_rad"]
            ),
            "single_support_rate": float(metrics["single_support_rate"]),
            "flight_rate": float(metrics["flight_rate"]),
            "scaled_total_heading_limit_rad": scaled_total_limit,
        },
    }


def run_candidate(
    simulator: Any,
    bank: Any,
    evaluator: Any,
    mujoco: Any,
    runtime: Any,
    candidate: Candidate,
    run: Mapping[str, Any],
    *,
    seconds: float,
    warmup_seconds: float,
    capture_trace: bool,
    require_slip: bool,
) -> dict[str, Any]:
    trace = RunTrace(
        evaluator,
        mujoco,
        candidate,
        sim_dt=runtime.SIM_DT,
        capture_trace=capture_trace,
    )
    original_router = central.SafeGaitRouter
    original_advance = central.advance_routed_phase
    original_step = central.apply_guarded_control_then_step_physics
    original_observation = evaluator._observation
    original_infer = bank.infer_route
    original_policy_target = simulator._policy_target

    class TracedRouter(original_router):
        def route(self, command: Sequence[float], dt: float) -> Any:
            decision = super().route(command, dt)
            trace.current_decision = decision
            trace.current_requested_command = np.asarray(command, dtype=np.float64).tolist()
            return decision

    def traced_advance(phase_index: float, **kwargs: Any) -> tuple[float, bool, Any]:
        effective = np.asarray(kwargs["effective_command"], dtype=np.float64)
        active = bool(
            effective[0] > 0.02 and kwargs["current_expert"] == EXPECTED_EXPERT
        )
        reset = bool(
            candidate.forward_entry_preincrement_phase is not None
            and active
            and not trace.previous_forward_active
            and not trace.phase_entry_events
        )
        event = None
        if reset:
            before = float(phase_index)
            phase_index = float(candidate.forward_entry_preincrement_phase)
            event = {
                "global_control_tick": int(kwargs["global_control_tick"]),
                "control_step": int(kwargs["control_step"]),
                "activation": "effective_vx_gt_0p02_false_to_true",
                "effective_command": effective.tolist(),
                "global_phase_index_before_reset": before,
                "reset_preincrement_phase_index": phase_index,
                "phase_delta": float(kwargs["phase_delta"]),
                "first_used_phase_index": (
                    phase_index + float(kwargs["phase_delta"])
                )
                % float(kwargs["phase_steps"]),
                "status": "DIAGNOSTIC_ONLY_NOT_ADOPTED",
            }
        result = original_advance(phase_index, **kwargs)
        if event is not None:
            event["advanced_phase_index"] = float(result[0])
            trace.phase_entry_events.append(event)
        trace.previous_forward_active = active
        trace._last_phase_reset = reset
        return result

    def traced_observation(
        data: Any,
        command: np.ndarray,
        default_actuator: np.ndarray,
        motor_targets: np.ndarray,
        action_history: list[np.ndarray],
        phase: float,
    ) -> np.ndarray:
        trace.initialize_heading(data)
        observation = original_observation(
            data, command, default_actuator, motor_targets, action_history, phase
        )
        decision = trace.current_decision
        if decision is None:
            raise RuntimeError("route decision missing")
        trace.control_rows.append(
            {
                "control_tick": len(trace.control_rows),
                "time_before_step_seconds": float(data.time),
                "requested_physical_command": list(trace.current_requested_command or ()),
                "effective_physical_command": list(decision.effective_command),
                "policy_observation_command": np.asarray(command[:3], dtype=float).tolist(),
                "expert": decision.expert,
                "blend_from_expert": decision.blend_from_expert,
                "blend_to_expert": decision.blend_to_expert,
                "blend_alpha": float(decision.blend_alpha),
                "phase_index": float(phase / (2.0 * np.pi) * evaluator.phase_steps),
                "phase_reset": bool(trace._last_phase_reset),
                "observation": observation.astype(float).tolist(),
                "contacts_before_step": evaluator._feet_contacts(data).astype(bool).tolist(),
            }
        )
        return observation

    def traced_infer(decision: Any, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw, applied = original_infer(decision, observation)
        row = trace.control_rows[-1]
        row["raw_action"] = np.asarray(raw, dtype=float).tolist()
        row["applied_action"] = np.asarray(applied, dtype=float).tolist()
        return raw, applied

    def traced_policy_target(
        action: np.ndarray,
        effective_command: np.ndarray,
        phase_index: float,
        default: np.ndarray,
    ) -> np.ndarray:
        targets = np.asarray(
            original_policy_target(action, effective_command, phase_index, default),
            dtype=np.float64,
        ).copy()
        raw_targets = targets.copy()
        generic_transforms: dict[str, Any] = {}
        generic_names = sorted(
            set(candidate.forward_target_scale_overrides)
            | set(candidate.forward_target_bias_overrides_rad)
        )
        for joint_name in generic_names:
            joint_index = int(simulator.name_to_index[joint_name])
            scale = float(candidate.forward_target_scale_overrides.get(joint_name, 1.0))
            bias = float(
                candidate.forward_target_bias_overrides_rad.get(joint_name, 0.0)
            )
            anchor = float(default[joint_index])
            affine = (
                anchor
                + scale * (float(raw_targets[joint_index]) - anchor)
                + bias
            )
            output = float(
                np.clip(
                    affine,
                    simulator.target_lower[joint_index],
                    simulator.target_upper[joint_index],
                )
            )
            targets[joint_index] = output
            generic_transforms[joint_name] = {
                "joint_index": joint_index,
                "anchor_default_rad": anchor,
                "scale": scale,
                "bias_rad": bias,
                "raw_target_rad": float(raw_targets[joint_index]),
                "affine_target_rad": affine,
                "output_target_rad": output,
                "clipped_to_target_envelope": abs(output - affine) > 1e-12,
                "target_lower_rad": float(simulator.target_lower[joint_index]),
                "target_upper_rad": float(simulator.target_upper[joint_index]),
            }
        left_index = int(simulator.left_knee_index)
        enabled = bool(
            abs(candidate.forward_left_knee_target_scale - 1.0) > 1e-12
            or abs(candidate.forward_left_knee_target_bias_rad) > 1e-12
        )
        affine_target = float(raw_targets[left_index])
        clipped = False
        if enabled:
            anchor = float(default[left_index])
            affine_target = (
                anchor
                + candidate.forward_left_knee_target_scale
                * (float(raw_targets[left_index]) - anchor)
                + candidate.forward_left_knee_target_bias_rad
            )
            transformed = float(
                np.clip(
                    affine_target,
                    simulator.target_lower[left_index],
                    simulator.target_upper[left_index],
                )
            )
            clipped = abs(transformed - affine_target) > 1e-12
            targets[left_index] = transformed
        row = trace.control_rows[-1]
        row["forward_joint_target_transforms"] = generic_transforms
        row["policy_targets_before_forward_knee_transform_rad"] = raw_targets.tolist()
        row["forward_left_knee_transform"] = {
            "enabled": enabled,
            "route_scope": "forward",
            "anchor_default_rad": float(default[left_index]),
            "scale": float(candidate.forward_left_knee_target_scale),
            "bias_rad": float(candidate.forward_left_knee_target_bias_rad),
            "raw_target_rad": float(raw_targets[left_index]),
            "affine_target_rad": affine_target,
            "output_target_rad": float(targets[left_index]),
            "clipped_to_target_envelope": clipped,
            "target_lower_rad": float(simulator.target_lower[left_index]),
            "target_upper_rad": float(simulator.target_upper[left_index]),
        }
        row["candidate_targets_rad"] = targets.tolist()
        return targets

    def traced_step(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        callback = kwargs.get("physics_substep_callback")
        control_tick = len(trace.control_rows) - 1

        def traced_callback() -> bool:
            terminated = bool(callback()) if callback is not None else False
            trace.sample_substep(kwargs["data"], control_tick)
            return terminated

        kwargs["physics_substep_callback"] = traced_callback
        result = original_step(*args, **kwargs)
        previous, applied, _, completed_substeps, terminated = result
        data = kwargs["data"]
        rotation = np.asarray(
            data.xmat[evaluator.trunk_body_id], dtype=np.float64
        ).reshape(3, 3)
        row = trace.control_rows[-1]
        preclip = np.asarray(row["candidate_targets_rad"], dtype=np.float64)
        desired = np.clip(preclip, simulator.target_lower, simulator.target_upper)
        applied_array = np.asarray(applied, dtype=np.float64)
        previous_array = np.asarray(previous, dtype=np.float64)
        slew_limit = float(simulator.target_slew_rate_rad_s * runtime.CONTROL_DT)
        clamp_mask = np.abs(desired - preclip) > 1e-12
        slew_mask = np.abs(applied_array - previous_array) >= slew_limit - 1e-12
        envelope_mask = np.logical_or(
            np.isclose(applied_array, simulator.target_lower, atol=1e-12, rtol=0.0),
            np.isclose(applied_array, simulator.target_upper, atol=1e-12, rtol=0.0),
        )
        row.update(
            {
                "completed_physics_substeps": int(completed_substeps),
                "substep_terminated": bool(terminated),
                "desired_targets_after_static_guard_rad": desired.tolist(),
                "static_clamp_mask": clamp_mask.astype(bool).tolist(),
                "static_clamp_count": int(np.count_nonzero(clamp_mask)),
                "static_clamp_excess_rad": np.abs(desired - preclip).tolist(),
                "applied_targets_rad": applied_array.tolist(),
                "applied_target_delta_rad": (
                    applied_array - previous_array
                ).tolist(),
                "slew_limit_delta_rad": slew_limit,
                "slew_limit_active_mask": slew_mask.astype(bool).tolist(),
                "slew_limit_active_count": int(np.count_nonzero(slew_mask)),
                "applied_at_target_envelope_mask": envelope_mask.astype(bool).tolist(),
                "applied_at_target_envelope_count": int(np.count_nonzero(envelope_mask)),
                "contacts_after_step": evaluator._feet_contacts(data).astype(bool).tolist(),
                "heading_change_after_step_rad": trace.unwrapped_heading,
                "upright_after_step": float(rotation[2, 2]),
            }
        )
        return result

    central.SafeGaitRouter = TracedRouter
    central.advance_routed_phase = traced_advance
    central.apply_guarded_control_then_step_physics = traced_step
    evaluator._observation = traced_observation
    bank.infer_route = traced_infer
    simulator._policy_target = traced_policy_target
    schedule = (
        (
            "forward",
            PHYSICAL_COMMAND,
            seconds,
            candidate.policy_observation_command,
            EXPECTED_EXPERT,
            EXPECTED_POLICY_ROLE,
        ),
    )
    try:
        result = simulator.run_schedule(
            schedule,
            seed=int(run["seed"]),
            joint_noise_scale=float(run["joint_noise_scale"]),
            initial_base_speed=float(run["initial_base_speed"]),
            warmup_seconds=warmup_seconds,
        )
    finally:
        central.SafeGaitRouter = original_router
        central.advance_routed_phase = original_advance
        central.apply_guarded_control_then_step_physics = original_step
        evaluator._observation = original_observation
        bank.infer_route = original_infer
        simulator._policy_target = original_policy_target
    segment = result["segments"][0]
    heading = trace.heading_summary()
    slip = trace.slip_summary()
    target_chain = trace.target_chain_summary()
    quality = strict_quality(
        segment,
        heading,
        slip,
        seconds=seconds,
        require_slip=require_slip,
    )
    startup = trace.control_rows[: min(75, len(trace.control_rows))]
    contacts = [tuple(row["contacts_after_step"]) for row in startup]
    first_single = next(
        (index for index, values in enumerate(contacts) if values[0] != values[1]),
        None,
    )
    payload = {
        **dict(run),
        "candidate": asdict(candidate),
        "physical_command": list(PHYSICAL_COMMAND),
        "physical_command_changed": False,
        "seconds": seconds,
        "warmup_seconds": warmup_seconds,
        "central_acceptance": segment_acceptance(segment),
        "strict_quality": quality,
        "heading": heading,
        "foot_slip": slip,
        "target_chain": target_chain,
        "startup": {
            "window_control_ticks": len(startup),
            "first_single_support_tick": first_single,
            "first_single_support_seconds": (
                None if first_single is None else first_single * float(runtime.CONTROL_DT)
            ),
            "contact_transition_count": sum(a != b for a, b in zip(contacts, contacts[1:])),
            "minimum_upright": min(float(row["upright_after_step"]) for row in startup),
        },
        "phase_entry_events": trace.phase_entry_events,
        "reset_qpos_audit": result["reset_qpos_audit"],
        "control_first_startup_audit": result["control_first_startup_audit"],
        "backward_exit_recovery_audit": result["backward_exit_recovery_audit"],
        "segment": segment,
    }
    if capture_trace:
        payload["control_trace"] = trace.control_rows
        payload["physics_substep_trace"] = trace.substep_rows
    return payload


def compare_control_traces(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    left = baseline["control_trace"]
    right = candidate["control_trace"]
    count = min(len(left), len(right))
    first_contact_divergence = None
    action_l2: list[float] = []
    observation_blocks: dict[str, list[float]] = {
        name: [] for name in OBSERVATION_BLOCKS
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


def slip_causal_analysis(
    record: Mapping[str, Any], joint_names: Sequence[str], *, sim_dt: float
) -> dict[str, Any]:
    """Reduce a full synchronized trace into stance/phase/joint causal evidence."""
    substeps = record["physics_substep_trace"]
    controls = record["control_trace"]
    phase_steps = 27
    result: dict[str, Any] = {
        "synchronization": {
            "physics_to_control_key": "physics_substep_trace[*].control_tick",
            "control_to_phase_key": "control_trace[*].phase_index",
            "control_target_chain": [
                "policy_targets_before_forward_knee_transform_rad",
                "candidate_targets_rad",
                "desired_targets_after_static_guard_rad",
                "applied_targets_rad",
            ],
            "physics_state": ["joint_qpos_rad", "joint_qvel_rad_s"],
            "physics_contact": [
                "point_speeds",
                "point_forces",
                "force_weighted_tangential_speed_mps",
                "normal_force_sum_n",
                "contact_point_count",
            ],
        },
        "feet": {},
    }
    for foot in ("left", "right"):
        sample_rows = [
            row
            for row in substeps
            if row[foot]["contact"]
            and row[foot]["force_weighted_tangential_speed_mps"] is not None
        ]
        speeds = np.asarray(
            [row[foot]["force_weighted_tangential_speed_mps"] for row in sample_rows],
            dtype=np.float64,
        )
        forces = np.asarray(
            [row[foot]["normal_force_sum_n"] for row in sample_rows], dtype=np.float64
        )
        qpos = np.asarray([row["joint_qpos_rad"] for row in sample_rows], dtype=np.float64)
        qvel = np.asarray([row["joint_qvel_rad_s"] for row in sample_rows], dtype=np.float64)
        phase = np.asarray([row["phase_index"] for row in sample_rows], dtype=np.float64)
        control_indices = np.asarray(
            [row["control_tick"] for row in sample_rows], dtype=np.int64
        )
        phase_bins: dict[str, Any] = {}
        energy_by_phase: list[tuple[int, float]] = []
        for phase_bin in range(phase_steps):
            mask = np.floor(phase).astype(np.int64) % phase_steps == phase_bin
            if not np.any(mask):
                continue
            phase_speeds = speeds[mask]
            energy = float(np.sum(np.square(phase_speeds)))
            energy_by_phase.append((phase_bin, energy))
            phase_bins[str(phase_bin)] = {
                "sample_count": int(np.count_nonzero(mask)),
                "force_weighted_speed_rms_mps": float(
                    np.sqrt(np.mean(np.square(phase_speeds)))
                ),
                "force_weighted_speed_p95_mps": float(
                    np.percentile(phase_speeds, 95.0)
                ),
                "force_weighted_speed_maximum_mps": float(np.max(phase_speeds)),
                "mean_normal_force_n": float(np.mean(forces[mask])),
                "mean_contact_point_count": float(
                    np.mean(
                        [
                            row[foot]["contact_point_count"]
                            for row, include in zip(sample_rows, mask)
                            if include
                        ]
                    )
                ),
            }
        total_energy = float(np.sum(np.square(speeds)))
        ranked_phase_energy = sorted(energy_by_phase, key=lambda item: item[1], reverse=True)

        stances: list[dict[str, Any]] = []
        active: list[Mapping[str, Any]] = []

        def finalize_stance(rows: list[Mapping[str, Any]]) -> None:
            if not rows:
                return
            stance_speeds = np.asarray(
                [row[foot]["force_weighted_tangential_speed_mps"] for row in rows],
                dtype=np.float64,
            )
            peak_index = int(np.argmax(stance_speeds))
            peak = rows[peak_index]
            stances.append(
                {
                    "stance_index": len(stances),
                    "start_physics_substep": int(rows[0]["physics_substep"]),
                    "stop_physics_substep": int(rows[-1]["physics_substep"]),
                    "start_time_seconds": float(rows[0]["time_seconds"]),
                    "stop_time_seconds": float(rows[-1]["time_seconds"]),
                    "duration_seconds": len(rows) * sim_dt,
                    "start_phase_index": float(rows[0]["phase_index"]),
                    "stop_phase_index": float(rows[-1]["phase_index"]),
                    "peak_phase_index": float(peak["phase_index"]),
                    "sample_count": len(rows),
                    "force_weighted_speed_rms_mps": float(
                        np.sqrt(np.mean(np.square(stance_speeds)))
                    ),
                    "force_weighted_speed_p95_mps": float(
                        np.percentile(stance_speeds, 95.0)
                    ),
                    "force_weighted_speed_maximum_mps": float(np.max(stance_speeds)),
                    "integrated_slip_proxy_m": float(np.sum(stance_speeds) * sim_dt),
                    "mean_normal_force_n": float(
                        np.mean([row[foot]["normal_force_sum_n"] for row in rows])
                    ),
                    "maximum_normal_force_n": float(
                        np.max([row[foot]["normal_force_sum_n"] for row in rows])
                    ),
                    "mean_contact_point_count": float(
                        np.mean([row[foot]["contact_point_count"] for row in rows])
                    ),
                    "maximum_contact_point_count": int(
                        np.max([row[foot]["contact_point_count"] for row in rows])
                    ),
                }
            )

        for row in substeps:
            if row[foot]["contact"]:
                if row[foot]["force_weighted_tangential_speed_mps"] is not None:
                    active.append(row)
            elif active:
                finalize_stance(active)
                active = []
        finalize_stance(active)

        top_indices = np.argsort(speeds)[-20:][::-1]
        peak_rows: list[dict[str, Any]] = []
        for index in top_indices:
            physics_row = sample_rows[int(index)]
            control = controls[int(physics_row["control_tick"])]
            peak_rows.append(
                {
                    "rank": len(peak_rows) + 1,
                    "physics_substep": int(physics_row["physics_substep"]),
                    "control_tick": int(physics_row["control_tick"]),
                    "time_seconds": float(physics_row["time_seconds"]),
                    "phase_index": float(physics_row["phase_index"]),
                    "force_weighted_tangential_speed_mps": float(speeds[index]),
                    "normal_force_n": float(forces[index]),
                    "contact_point_count": int(physics_row[foot]["contact_point_count"]),
                    "point_speeds_mps": physics_row[foot]["point_speeds"],
                    "point_forces_n": physics_row[foot]["point_forces"],
                    "joint_qpos_rad": physics_row["joint_qpos_rad"],
                    "joint_qvel_rad_s": physics_row["joint_qvel_rad_s"],
                    "raw_action": control["raw_action"],
                    "applied_action": control["applied_action"],
                    "raw_policy_targets_rad": control[
                        "policy_targets_before_forward_knee_transform_rad"
                    ],
                    "preclip_targets_rad": control["candidate_targets_rad"],
                    "desired_targets_after_static_guard_rad": control[
                        "desired_targets_after_static_guard_rad"
                    ],
                    "applied_targets_rad": control["applied_targets_rad"],
                    "static_clamp_mask": control["static_clamp_mask"],
                    "slew_limit_active_mask": control["slew_limit_active_mask"],
                    "applied_at_target_envelope_mask": control[
                        "applied_at_target_envelope_mask"
                    ],
                }
            )

        joint_associations: list[dict[str, Any]] = []
        for joint_index, joint_name in enumerate(joint_names):
            clamp = np.asarray(
                [controls[index]["static_clamp_mask"][joint_index] for index in control_indices],
                dtype=bool,
            )
            slew = np.asarray(
                [
                    controls[index]["slew_limit_active_mask"][joint_index]
                    for index in control_indices
                ],
                dtype=bool,
            )
            envelope = np.asarray(
                [
                    controls[index]["applied_at_target_envelope_mask"][joint_index]
                    for index in control_indices
                ],
                dtype=bool,
            )

            def conditional_mean(mask: np.ndarray) -> float | None:
                return None if not np.any(mask) else float(np.mean(speeds[mask]))

            joint_associations.append(
                {
                    "joint_index": joint_index,
                    "joint_name": joint_name,
                    "speed_vs_signed_qvel_correlation": _correlation(
                        speeds, qvel[:, joint_index]
                    ),
                    "speed_vs_absolute_qvel_correlation": _correlation(
                        speeds, np.abs(qvel[:, joint_index])
                    ),
                    "speed_vs_qpos_correlation": _correlation(
                        speeds, qpos[:, joint_index]
                    ),
                    "static_clamp_sample_rate": float(np.mean(clamp)),
                    "mean_speed_when_static_clamped_mps": conditional_mean(clamp),
                    "mean_speed_when_not_static_clamped_mps": conditional_mean(~clamp),
                    "slew_limit_sample_rate": float(np.mean(slew)),
                    "mean_speed_when_slew_limited_mps": conditional_mean(slew),
                    "mean_speed_when_not_slew_limited_mps": conditional_mean(~slew),
                    "envelope_sample_rate": float(np.mean(envelope)),
                    "mean_speed_when_at_envelope_mps": conditional_mean(envelope),
                    "mean_speed_when_not_at_envelope_mps": conditional_mean(~envelope),
                }
            )
        ranked_abs_qvel = sorted(
            joint_associations,
            key=lambda row: abs(row["speed_vs_absolute_qvel_correlation"] or 0.0),
            reverse=True,
        )
        result["feet"][foot] = {
            "contact_sample_count": len(sample_rows),
            "force_weighted_speed_rms_mps": float(
                np.sqrt(np.mean(np.square(speeds)))
            ),
            "force_weighted_speed_p95_mps": float(np.percentile(speeds, 95.0)),
            "force_weighted_speed_maximum_mps": float(np.max(speeds)),
            "mean_normal_force_n": float(np.mean(forces)),
            "speed_vs_normal_force_correlation": _correlation(speeds, forces),
            "phase_bins": phase_bins,
            "slip_energy_top_three_phase_bins": [
                {
                    "phase_bin": phase_bin,
                    "energy_fraction": 0.0 if total_energy == 0.0 else energy / total_energy,
                }
                for phase_bin, energy in ranked_phase_energy[:3]
            ],
            "top_joint_absolute_qvel_correlations": ranked_abs_qvel[:5],
            "joint_target_saturation_associations": joint_associations,
            "stances": stances,
            "top_twenty_slip_peaks": peak_rows,
        }
    return result
    joint_delta_sum = np.zeros(14, dtype=np.float64)
    for index in range(count):
        left_obs = np.asarray(left[index]["observation"], dtype=np.float64)
        right_obs = np.asarray(right[index]["observation"], dtype=np.float64)
        for name, (start, stop) in OBSERVATION_BLOCKS.items():
            observation_blocks[name].append(
                float(np.linalg.norm(right_obs[start:stop] - left_obs[start:stop]))
            )
        action_delta = np.asarray(right[index]["applied_action"], dtype=np.float64) - np.asarray(
            left[index]["applied_action"], dtype=np.float64
        )
        action_l2.append(float(np.linalg.norm(action_delta)))
        joint_delta_sum += action_delta
        if (
            first_contact_divergence is None
            and left[index]["contacts_after_step"] != right[index]["contacts_after_step"]
        ):
            first_contact_divergence = index
    return {
        "compared_control_ticks": count,
        "first_contact_divergence_tick": first_contact_divergence,
        "first_tick_applied_action_l2": action_l2[0],
        "startup_50_tick_applied_action_l2_mean": float(np.mean(action_l2[:50])),
        "applied_action_l2_mean": float(np.mean(action_l2)),
        "applied_action_l2_maximum": float(np.max(action_l2)),
        "mean_applied_action_delta_by_joint": (joint_delta_sum / count).tolist(),
        "observation_block_l2": {
            name: {
                "first_tick": values[0],
                "startup_50_tick_mean": float(np.mean(values[:50])),
                "mean": float(np.mean(values)),
                "maximum": float(np.max(values)),
            }
            for name, values in observation_blocks.items()
        },
        "outcome_delta": {
            "vx_tracking_ratio": candidate["strict_quality"]["values"]["vx_tracking_ratio"]
            - baseline["strict_quality"]["values"]["vx_tracking_ratio"],
            "cross_velocity_mps": candidate["strict_quality"]["values"]["cross_velocity_mps"]
            - baseline["strict_quality"]["values"]["cross_velocity_mps"],
            "signed_cross_velocity_mps": candidate["strict_quality"]["values"][
                "signed_cross_velocity_mps"
            ]
            - baseline["strict_quality"]["values"]["signed_cross_velocity_mps"],
            "uncommanded_yaw_rate_rad_s": candidate["strict_quality"]["values"][
                "uncommanded_yaw_rate_rad_s"
            ]
            - baseline["strict_quality"]["values"]["uncommanded_yaw_rate_rad_s"],
            "signed_uncommanded_yaw_rate_rad_s": candidate["strict_quality"]["values"][
                "signed_uncommanded_yaw_rate_rad_s"
            ]
            - baseline["strict_quality"]["values"]["signed_uncommanded_yaw_rate_rad_s"],
            "total_heading_change_rad": candidate["strict_quality"]["values"][
                "total_heading_change_rad"
            ]
            - baseline["strict_quality"]["values"]["total_heading_change_rad"],
        },
    }


def candidate_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    qualities = [record["strict_quality"] for record in records]
    values = [quality["values"] for quality in qualities]
    central_gait_metrics = [
        record["segment"].get("gait_quality_metrics") for record in records
    ]
    central_gait_acceptances = [
        record["segment"].get("gait_quality_acceptance") for record in records
    ]
    if any(item is None for item in central_gait_metrics + central_gait_acceptances):
        raise RuntimeError("central gait-quality instrumentation is required")
    gait_metrics = [dict(item) for item in central_gait_metrics]
    gait_acceptances = [dict(item) for item in central_gait_acceptances]
    failed_checks: dict[str, int] = {}
    for quality in qualities:
        for name, passed in {
            **quality["checks"],
            **{f"safety:{key}": value for key, value in quality["safety_zero_checks"].items()},
            **(
                {
                    f"slip:{key}": value
                    for key, value in quality["provisional_slip_checks"].items()
                }
                if quality["provisional_slip_checks_required"]
                else {}
            ),
        }.items():
            if not passed:
                failed_checks[name] = failed_checks.get(name, 0) + 1
    central_gait_failed_checks: dict[str, int] = {}
    for acceptance in gait_acceptances:
        for name in acceptance["failures"]:
            central_gait_failed_checks[name] = central_gait_failed_checks.get(name, 0) + 1

    def maximum_optional(name: str) -> float | None:
        optional = [metric[name] for metric in gait_metrics]
        return None if any(value is None for value in optional) else max(
            float(value) for value in optional
        )

    debounce_sensitivity: dict[str, Any] = {}
    for window in ("10ms", "20ms", "30ms", "40ms"):
        rows = [metric["contact_debounce_sensitivity"][window] for metric in gait_metrics]
        alternating = [
            row["alternating_touchdown_fraction"]
            for row in rows
            if row["alternating_touchdown_fraction"] is not None
        ]
        debounce_sensitivity[window] = {
            "minimum_left_touchdowns": min(int(row["left_touchdowns"]) for row in rows),
            "maximum_left_touchdowns": max(int(row["left_touchdowns"]) for row in rows),
            "minimum_right_touchdowns": min(int(row["right_touchdowns"]) for row in rows),
            "maximum_right_touchdowns": max(int(row["right_touchdowns"]) for row in rows),
            "minimum_single_support_rate": min(
                float(row["single_support_rate"]) for row in rows
            ),
            "maximum_single_support_rate": max(
                float(row["single_support_rate"]) for row in rows
            ),
            "maximum_flight_rate": max(float(row["flight_rate"]) for row in rows),
            "minimum_alternating_touchdown_fraction": (
                None if not alternating else min(float(value) for value in alternating)
            ),
        }

    return {
        "run_count": len(records),
        "passed_count": sum(bool(quality["passed"]) for quality in qualities),
        "all_runs_passed": all(bool(quality["passed"]) for quality in qualities),
        "failed_check_counts": dict(sorted(failed_checks.items())),
        "central_gait_quality": {
            "passed_count": sum(bool(item["passed"]) for item in gait_acceptances),
            "all_runs_passed": all(bool(item["passed"]) for item in gait_acceptances),
            "failed_check_counts": dict(sorted(central_gait_failed_checks.items())),
            "measurement_sources": {
                "contact_state_source": sorted(
                    {str(metric["contact_state_source"]) for metric in gait_metrics}
                ),
                "stance_slip_measurement_source": sorted(
                    {
                        str(metric["stance_slip_measurement_source"])
                        for metric in gait_metrics
                    }
                ),
                "minimum_contact_force_sample_count": min(
                    int(metric["contact_force_sample_count"]) for metric in gait_metrics
                ),
                "minimum_contact_velocity_sample_count": min(
                    int(metric["contact_velocity_sample_count"]) for metric in gait_metrics
                ),
            },
            "contact_debounce_sensitivity": debounce_sensitivity,
            "worst_values": {
                "maximum_t30_s": maximum_optional("t30_s"),
                "maximum_t75_s": maximum_optional("t75_s"),
                "maximum_first_single_support_s": maximum_optional(
                    "first_single_support_s"
                ),
                "minimum_single_support_rate": min(
                    float(metric["single_support_rate"]) for metric in gait_metrics
                ),
                "maximum_single_support_rate": max(
                    float(metric["single_support_rate"]) for metric in gait_metrics
                ),
                "maximum_flight_rate": max(
                    float(metric["flight_rate"]) for metric in gait_metrics
                ),
                "maximum_contact_duty_imbalance": max(
                    float(metric["contact_duty_imbalance"]) for metric in gait_metrics
                ),
                "minimum_alternating_touchdown_fraction": min(
                    float(metric["alternating_touchdown_fraction"])
                    for metric in gait_metrics
                    if metric["alternating_touchdown_fraction"] is not None
                ),
                "maximum_stance_slip_rms_mps": maximum_optional(
                    "stance_slip_rms_mps"
                ),
                "maximum_stance_slip_p95_mps": maximum_optional(
                    "stance_slip_p95_mps"
                ),
                "maximum_per_stance_cumulative_slip_m": maximum_optional(
                    "maximum_per_stance_cumulative_slip_m"
                ),
                "minimum_steady_linear_tracking_ratio": min(
                    float(metric["steady_linear_tracking_ratio"])
                    for metric in gait_metrics
                    if metric["steady_linear_tracking_ratio"] is not None
                ),
                "maximum_steady_linear_tracking_ratio": max(
                    float(metric["steady_linear_tracking_ratio"])
                    for metric in gait_metrics
                    if metric["steady_linear_tracking_ratio"] is not None
                ),
                "maximum_steady_cross_drift_mps": maximum_optional(
                    "steady_cross_drift_mps"
                ),
                "maximum_uncommanded_yaw_rate_radps": maximum_optional(
                    "uncommanded_yaw_rate_radps"
                ),
                "maximum_absolute_uncommanded_heading_drift_rad": max(
                    abs(float(metric["uncommanded_heading_drift_rad"]))
                    for metric in gait_metrics
                    if metric["uncommanded_heading_drift_rad"] is not None
                ),
            },
        },
        "worst_values": {
            "minimum_vx_tracking_ratio": min(value["vx_tracking_ratio"] for value in values),
            "maximum_vx_tracking_ratio": max(value["vx_tracking_ratio"] for value in values),
            "maximum_cross_velocity_mps": max(value["cross_velocity_mps"] for value in values),
            "maximum_uncommanded_yaw_rate_rad_s": max(
                value["uncommanded_yaw_rate_rad_s"] for value in values
            ),
            "maximum_absolute_total_heading_change_rad": max(
                abs(value["total_heading_change_rad"]) for value in values
            ),
            "maximum_rolling_six_second_heading_change_rad": max(
                value["maximum_rolling_six_second_heading_change_rad"]
                for value in values
            ),
            "minimum_single_support_rate": min(value["single_support_rate"] for value in values),
            "maximum_single_support_rate": max(value["single_support_rate"] for value in values),
            "maximum_flight_rate": max(value["flight_rate"] for value in values),
            "maximum_contact_point_tangential_rms_mps": max(
                record["foot_slip"]["combined"]["tangential_speed_rms_mps"]
                for record in records
            ),
            "maximum_contact_point_tangential_p95_mps": max(
                record["foot_slip"]["combined"]["tangential_speed_p95_mps"]
                for record in records
            ),
            "maximum_integrated_slip_proxy_per_stance_m": max(
                record["foot_slip"]["combined"][
                    "maximum_integrated_slip_proxy_per_stance_m"
                ]
                for record in records
            ),
            "maximum_left_knee_raw_policy_target_above_margin_upper_rate": max(
                record["target_chain"]["left_knee"][
                    "raw_policy_target_above_margin_upper_rate"
                ]
                for record in records
            ),
            "maximum_left_knee_transformed_target_at_margin_upper_ticks": max(
                record["target_chain"]["left_knee"][
                    "transformed_target_at_margin_upper_ticks"
                ]
                for record in records
            ),
        },
    }


def v2_reproduction(record: Mapping[str, Any]) -> dict[str, Any]:
    manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    expected = next(segment for segment in manifest["segments"] if segment["name"] == "forward")
    expected_metrics = expected["metrics"]
    actual_metrics = record["segment"]["metrics"]
    keys = (
        "projected_primary_velocity",
        "absolute_orthogonal_velocity",
        "mean_local_yaw_rate",
        "single_support_rate",
        "flight_rate",
        "minimum_upright",
    )
    deltas = {
        key: float(actual_metrics[key]) - float(expected_metrics[key]) for key in keys
    }
    return {
        "expected_seed": V2_SEED,
        "expected_policy_observation_command": expected["policy_observation_command"],
        "metric_deltas": deltas,
        "exact_match_within_1e_minus_12": all(abs(value) <= 1e-12 for value in deltas.values()),
        "manifest_sha256": sha256(V2_MANIFEST),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic artifact: {output}")
    candidates = load_candidates(args.candidates_json, stage=args.stage)
    central_path = Path(central.__file__).resolve()
    pre_run_central_sha256 = sha256(central_path)
    mujoco, onnxruntime, runtime, provenance = central._load_runtime(include_provenance=True)
    assets = central.generated_asset_paths(central.DEFAULT_GENERATED_ROOT.resolve())
    policy_paths = {role: BASE_POLICY for role in central.REQUIRED_POLICY_ROLES}
    bank = central.RoutedPolicyBank(policy_paths, onnxruntime)
    evaluator = runtime.OfficialPolicyEvaluator(assets["scene"], BASE_POLICY, assets["reference"])
    evaluator.backward_residual_scale = 0.0
    evaluator.load_backward_profile(central.DEFAULT_BACKWARD_PROFILE)
    evaluator.load_backward_turn_profile(1, central.DEFAULT_BACKWARD_LEFT_PROFILE)
    evaluator.load_backward_turn_profile(-1, central.DEFAULT_BACKWARD_RIGHT_PROFILE)
    evaluator.backward_turn_minimum_yaw = 0.0
    evaluator.backward_turn_minimum_blend = 0.0
    evaluator.backward_turn_maximum_blend = 1.0
    central.validate_model_contract(evaluator)
    for candidate in candidates:
        candidate.validate(evaluator.phase_steps)
    simulator = central.RoutedSimulator(
        evaluator,
        bank,
        mujoco,
        runtime,
        leg_target_margin_rad=central.RUNTIME_TARGET_SAFETY_MARGIN_RAD,
        target_slew_rate_rad_s=central.RUNTIME_TARGET_SLEW_RATE_RAD_S,
        diagnostic_noncontract_safety=False,
        left_knee_extra_upper_margin_rad=0.0125,
        formal_candidate_default=True,
    )
    runs = stage_runs(args.stage)
    records: list[dict[str, Any]] = []
    by_candidate: dict[str, list[dict[str, Any]]] = {
        candidate.candidate_id: [] for candidate in candidates
    }
    for candidate in candidates:
        for run in runs:
            record = run_candidate(
                simulator,
                bank,
                evaluator,
                mujoco,
                runtime,
                candidate,
                run,
                seconds=args.seconds,
                warmup_seconds=args.warmup_seconds,
                capture_trace=args.stage in {"trace", "causal"},
                require_slip=args.require_provisional_slip_gates,
            )
            records.append(record)
            by_candidate[candidate.candidate_id].append(record)
    summaries = {
        candidate_id: candidate_summary(candidate_records)
        for candidate_id, candidate_records in by_candidate.items()
    }
    trace_comparisons: dict[str, Any] = {}
    causal_analyses: dict[str, Any] = {}
    causal_outcome_comparisons: dict[str, Any] = {}
    reproduction = None
    if args.stage == "trace":
        baseline = by_candidate["v2_baseline"][0]
        reproduction = v2_reproduction(baseline)
        trace_comparisons = {
            candidate.candidate_id: compare_control_traces(
                baseline, by_candidate[candidate.candidate_id][0]
            )
            for candidate in candidates
            if candidate.candidate_id != "v2_baseline"
        }
    if args.stage == "causal":
        causal_analyses = {
            f"{record['run_id']}::{record['candidate']['candidate_id']}": (
                slip_causal_analysis(record, simulator.joint_names, sim_dt=runtime.SIM_DT)
            )
            for record in records
        }
        for run in runs:
            run_records = [record for record in records if record["run_id"] == run["run_id"]]
            if len(run_records) != 2:
                continue
            baseline = next(
                record
                for record in run_records
                if tuple(record["candidate"]["policy_observation_command"])
                == (0.10, 0.0, 0.0)
            )
            selected = next(record for record in run_records if record is not baseline)
            baseline_gait = baseline["segment"]["gait_quality_metrics"]
            selected_gait = selected["segment"]["gait_quality_metrics"]
            causal_outcome_comparisons[run["run_id"]] = {
                "baseline_candidate_id": baseline["candidate"]["candidate_id"],
                "selected_candidate_id": selected["candidate"]["candidate_id"],
                "selected_minus_baseline": {
                    name: float(selected_gait[name]) - float(baseline_gait[name])
                    for name in (
                        "stance_slip_rms_mps",
                        "stance_slip_p95_mps",
                        "maximum_per_stance_cumulative_slip_m",
                        "steady_linear_tracking_ratio",
                        "steady_cross_drift_mps",
                        "uncommanded_yaw_rate_radps",
                        "uncommanded_heading_drift_rad",
                    )
                },
                "baseline_gait_quality_failures": baseline["segment"][
                    "gait_quality_acceptance"
                ]["failures"],
                "selected_gait_quality_failures": selected["segment"][
                    "gait_quality_acceptance"
                ]["failures"],
            }
    post_run_central_sha256 = sha256(central_path)
    if post_run_central_sha256 != pre_run_central_sha256:
        raise RuntimeError(
            "central evaluator changed during diagnostic run; refusing mixed-snapshot artifact"
        )
    payload = {
        "schema_version": 1,
        "artifact_kind": "openduckmini_forward_straight_quality_pdca",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DIAGNOSTIC_ONLY_NOT_ADOPTED",
        "hardware_deployment": "PROHIBITED",
        "central_sources_modified": False,
        "package_modified": False,
        "physical_command_changed": False,
        "route_scope": ["forward"],
        "stage": args.stage,
        "configuration": {
            "physical_command": list(PHYSICAL_COMMAND),
            "seconds": args.seconds,
            "warmup_seconds": args.warmup_seconds,
            "candidate_count": len(candidates),
            "run_count_per_candidate": len(runs),
            "joint_noise_and_initial_speed": [dict(run) for run in runs],
            "require_provisional_slip_gates": args.require_provisional_slip_gates,
            "pre_run_central_sha256": pre_run_central_sha256,
            "post_run_central_sha256": post_run_central_sha256,
            "central_sha256_unchanged_during_run": True,
        },
        "strict_gates": dict(STRICT_GATES),
        "provisional_slip_gates": dict(PROVISIONAL_SLIP_GATES),
        "v2_exact_reproduction": reproduction,
        "candidates": [asdict(candidate) for candidate in candidates],
        "candidate_summaries": summaries,
        "trace_comparisons_against_v2_baseline": trace_comparisons,
        "slip_causal_analyses": causal_analyses,
        "slip_causal_outcome_comparisons": causal_outcome_comparisons,
        "records": records,
        "dependencies": {
            "diagnostic_script_sha256": sha256(Path(__file__).resolve()),
            "central_evaluator_sha256": post_run_central_sha256,
            "policy_sha256": sha256(BASE_POLICY),
            "v2_manifest_sha256": sha256(V2_MANIFEST),
            "runtime": provenance,
            "onnx_providers": bank.session_providers,
        },
    }
    serialization_replacements: list[str] = []
    finite_payload = json_finite_copy(payload, serialization_replacements)
    finite_payload["serialization_nonfinite_replacements"] = serialization_replacements
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(finite_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    concise = {
        candidate_id: summary for candidate_id, summary in summaries.items()
    }
    print(
        json.dumps(
            {
                "output": str(output),
                "stage": args.stage,
                "v2_exact_reproduction": reproduction,
                "candidate_summaries": concise,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
