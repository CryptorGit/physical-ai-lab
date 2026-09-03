"""Reproduce H1 transition overshoots and screen a tighter left-knee cap.

This is an isolated, simulation-only diagnostic.  It imports the central
evaluator but never edits its contract, package, or runtime sources.  Candidate
constants are substituted only inside this Python process and restored after
each run.  Hardware deployment is always prohibited.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = EXP_ROOT.parents[2]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from scripts import evaluate_routed_transitions as central  # noqa: E402
from scripts.diagnose_h1_reverse_robustness import (  # noqa: E402
    advance_routed_phase_candidate as advance_integrated_phase,
)
import safe_gait_experts.routed_evaluation as routed_contract  # noqa: E402
import target_safety as target_safety_runtime  # noqa: E402


SOURCE_H1_ARTIFACT = (
    EXP_ROOT
    / "artifacts"
    / "formal_candidate_pending_20x30_seed20260808_v1.json"
)
STRAIGHT_SELECTION_EVIDENCE_PATHS = (
    EXP_ROOT
    / "artifacts"
    / "h1_phase57_rate105_formal20x30s_v1.json",
    EXP_ROOT
    / "artifacts"
    / "h1_phase7_rate105_formal_transition_reverse_prefix_20seed_v1.json",
    EXP_ROOT
    / "artifacts"
    / "h1_phase7_rate105_cap01625_formal20x30s_v1.json",
)
DEFAULT_OUTPUT = (
    EXP_ROOT
    / "artifacts"
    / "h1_left_knee_exit_recovery_cap_decoupled_focus5_transition20_v1.json"
)
BASE_POLICY = (
    WORKSPACE / ".openduck_runtime_source_review" / "calibrated_hybrid_policy_v22.onnx"
)
PROFILE_PATHS = {
    key: Path(value).resolve()
    for key, value in routed_contract.FORMAL_CANDIDATE_PROFILE_PATHS.items()
}
PROFILE_PATHS["straight"] = (
    EXP_ROOT
    / "artifacts"
    / "optimized_reverse_margin050_slew200_h1_phase7_rate105_candidate_v1.json"
).resolve()

FORMAL_MASTER_SEED = 20_260_808
FORMAL_TRANSITION_SEEDS = tuple(22_260_808 + index for index in range(20))
FOCUS_TRANSITION_SEEDS = (
    22_260_808,
    22_260_813,
    22_260_818,
    22_260_819,
    22_260_824,
)
KNOWN_FAILING_TRANSITION_SEEDS = (
    22_260_808,
    22_260_813,
    22_260_819,
    22_260_824,
)
NEAR_BOUND_PASSING_SEED = 22_260_818
SCREEN_CAPS_RAD = (0.01625, 0.0175, 0.0200, 0.0225)
SCREEN_HOLD_TICKS = (13, 16)
BASELINE_CAP_RAD = 0.0125
BASELINE_HOLD_TICKS = 13
MOVING_SECONDS = 30.0
STAND_SECONDS = 5.0
WARMUP_SECONDS = 1.5
INITIAL_JOINT_NOISE_SCALE = 1.0
INITIAL_BASE_SPEED_MPS = 0.10
H1_BASELINE_PHASE_ENTRY_INDICES = {
    "reverse": 6.0,
    "reverse_turn_left": 4.0,
    "reverse_turn_right": 4.0,
}
PHASE_ENTRY_INDICES = {
    "reverse": 7.0,
    "reverse_turn_left": 4.0,
    "reverse_turn_right": 4.0,
}
PREFIX_SEGMENT_NAMES = (
    "transition_stand_0",
    "transition_forward",
    "transition_stand_after_forward",
    "transition_reverse",
    "transition_stand_after_reverse",
    "transition_reverse_turn_left",
    "transition_stand_after_reverse_turn_left",
    "transition_reverse_turn_right",
    "transition_stand_after_reverse_turn_right",
)
BACKWARD_SEGMENT_NAMES = frozenset(
    {
        "transition_reverse",
        "transition_reverse_turn_left",
        "transition_reverse_turn_right",
    }
)
DIAGNOSTIC_STATUS = "DIAGNOSTIC_H1_LEFT_KNEE_CAP_SWEEP_NOT_ADOPTED"


@dataclass(frozen=True)
class CapCandidate:
    """One process-local exit-recovery cap with frozen profile composition."""

    recovery_extra_upper_margin_rad: float
    recovery_hold_ticks: int
    profile_extra_upper_margin_rad: float = BASELINE_CAP_RAD

    def validate(self) -> None:
        for label, margin in (
            ("profile", float(self.profile_extra_upper_margin_rad)),
            ("recovery", float(self.recovery_extra_upper_margin_rad)),
        ):
            if (
                not np.isfinite(margin)
                or margin < 0.0
                or margin > routed_contract.MAX_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
            ):
                raise ValueError(
                    f"left-knee {label} extra margin is outside the diagnostic bound"
                )
        if (
            isinstance(self.recovery_hold_ticks, bool)
            or int(self.recovery_hold_ticks) != self.recovery_hold_ticks
            or int(self.recovery_hold_ticks) <= 0
        ):
            raise ValueError("recovery hold must be a positive whole control tick count")

    @property
    def recovery_upper_target_rad(self) -> float:
        return (
            float(central.SAFE_JOINT_LIMITS["left_knee"][1])
            - central.RUNTIME_TARGET_SAFETY_MARGIN_RAD
            - float(self.recovery_extra_upper_margin_rad)
        )

    @property
    def profile_upper_target_rad(self) -> float:
        return (
            float(central.SAFE_JOINT_LIMITS["left_knee"][1])
            - central.RUNTIME_TARGET_SAFETY_MARGIN_RAD
            - float(self.profile_extra_upper_margin_rad)
        )

    @property
    def hold_seconds(self) -> float:
        return int(self.recovery_hold_ticks) * target_safety_runtime.CONTROL_FIRST_STARTUP_DT_S

    @property
    def candidate_id(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]

    def contract(self) -> dict[str, Any]:
        self.validate()
        return {
            **asdict(self),
            "candidate_id": self.candidate_id,
            "left_knee_safe_upper_rad": float(
                central.SAFE_JOINT_LIMITS["left_knee"][1]
            ),
            "base_target_margin_rad": central.RUNTIME_TARGET_SAFETY_MARGIN_RAD,
            "profile_upper_target_rad": self.profile_upper_target_rad,
            "recovery_upper_target_rad": self.recovery_upper_target_rad,
            "profile_cap_held_at_h1_value": (
                self.profile_extra_upper_margin_rad == BASELINE_CAP_RAD
            ),
            "recovery_hold_seconds": self.hold_seconds,
            "recovery_release": "instant_after_hold",
        }


def exact_formal_transition_prefix() -> tuple[tuple[Any, ...], ...]:
    """Return the exact 30/5 s formal prefix through all backward families."""

    schedule = tuple(
        routed_contract.transition_schedule(MOVING_SECONDS, STAND_SECONDS)[:9]
    )
    if tuple(case[0] for case in schedule) != PREFIX_SEGMENT_NAMES:
        raise RuntimeError("central transition prefix changed")
    return schedule


def screening_candidates() -> tuple[CapCandidate, ...]:
    return tuple(
        CapCandidate(cap, hold)
        for cap in SCREEN_CAPS_RAD
        for hold in SCREEN_HOLD_TICKS
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_paths() -> tuple[Path, ...]:
    return tuple(
        dict.fromkeys(
            [
                Path(__file__).resolve(),
                Path(central.__file__).resolve(),
                Path(sys.modules[advance_integrated_phase.__module__].__file__).resolve(),
                Path(routed_contract.__file__).resolve(),
                Path(target_safety_runtime.__file__).resolve(),
                BASE_POLICY.resolve(),
                SOURCE_H1_ARTIFACT.resolve(),
                *PROFILE_PATHS.values(),
                *(path.resolve() for path in STRAIGHT_SELECTION_EVIDENCE_PATHS),
            ]
        )
    )


def _closure(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    return {
        str(path.resolve()): {
            "sha256": _sha256(path.resolve()),
            "size_bytes": path.resolve().stat().st_size,
        }
        for path in paths
    }


@contextmanager
def _candidate_runtime_constants(candidate: CapCandidate) -> Iterator[None]:
    """Temporarily bind one cap into all imported runtime constant aliases."""

    candidate.validate()
    values = {
        "BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD": float(
            candidate.recovery_extra_upper_margin_rad
        ),
        "BACKWARD_EXIT_RECOVERY_HOLD_TICKS": int(candidate.recovery_hold_ticks),
        "BACKWARD_EXIT_RECOVERY_HOLD_SECONDS": candidate.hold_seconds,
        "BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD": (
            candidate.recovery_upper_target_rad
        ),
    }
    modules = (target_safety_runtime, routed_contract, central)
    saved: list[tuple[Any, str, Any]] = []
    saved_advance = central.advance_routed_phase
    try:
        for module in modules:
            for name, value in values.items():
                if hasattr(module, name):
                    saved.append((module, name, getattr(module, name)))
                    setattr(module, name, value)
        central.advance_routed_phase = advance_integrated_phase
        yield
    finally:
        central.advance_routed_phase = saved_advance
        for module, name, value in reversed(saved):
            setattr(module, name, value)


def _source_segment(
    payload: Mapping[str, Any], seed: int, segment_name: str
) -> Mapping[str, Any]:
    episode = next(
        episode
        for episode in payload["suites"]["transitions"]["episodes"]
        if int(episode["seed"]) == int(seed)
    )
    return next(
        segment for segment in episode["segments"] if segment["name"] == segment_name
    )


def _source_acceptance(
    payload: Mapping[str, Any], seed: int, segment_name: str
) -> Mapping[str, Any]:
    episode = next(
        episode
        for episode in payload["suites"]["transitions"]["acceptance"][
            "episode_checks"
        ]
        if int(episode["seed"]) == int(seed)
    )
    return next(
        segment for segment in episode["segments"] if segment["name"] == segment_name
    )


def _load_h1_source(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    configuration = payload["configuration"]
    expected_configuration = {
        "episodes": 20,
        "seconds": MOVING_SECONDS,
        "transition_seconds": MOVING_SECONDS,
        "transition_stand_seconds": STAND_SECONDS,
        "warmup_seconds": WARMUP_SECONDS,
        "initial_joint_noise_scale": INITIAL_JOINT_NOISE_SCALE,
        "initial_base_speed": INITIAL_BASE_SPEED_MPS,
        "seed": FORMAL_MASTER_SEED,
        "leg_target_margin_rad": central.RUNTIME_TARGET_SAFETY_MARGIN_RAD,
        "target_slew_rate_rad_per_s": central.RUNTIME_TARGET_SLEW_RATE_RAD_S,
        "left_knee_extra_upper_margin_rad": BASELINE_CAP_RAD,
    }
    configuration_checks = {
        name: configuration.get(name) == expected
        for name, expected in expected_configuration.items()
    }
    configuration_checks.update(
        {
            "historical_phase_6_4_4": payload[
                "formal_reverse_phase_entry_contract"
            ][
                "preincrement_phase_indices"
            ]
            == H1_BASELINE_PHASE_ENTRY_INDICES,
            "recovery_hold_13": payload["target_safety_contract"][
                "backward_exit_recovery"
            ]["hold_control_ticks"]
            == BASELINE_HOLD_TICKS,
            "recovery_cap_matches_profile": payload["target_safety_contract"][
                "backward_exit_recovery"
            ]["left_knee_upper_target_rad"]
            == CapCandidate(
                BASELINE_CAP_RAD, BASELINE_HOLD_TICKS
            ).recovery_upper_target_rad,
            "reverse_endpoint_minus_0p05": payload[
                "formal_reverse_phase_entry_contract"
            ]["current_formal_reverse_endpoint_mps"]
            == -0.05,
            "release_scale_exact": payload["release_qualification"][
                "scale_matches_frozen_contract"
            ]
            is True,
            "hardware_prohibited": payload["hardware_gate"]["status"]
            == "PROHIBITED"
            and payload["hardware_gate"]["hardware_deployment_allowed"] is False,
        }
    )

    failures: list[dict[str, Any]] = []
    for episode in payload["suites"]["transitions"]["acceptance"][
        "episode_checks"
    ]:
        seed = int(episode["seed"])
        for acceptance in episode["segments"]:
            false_checks = sorted(
                name
                for name, passed in acceptance["checks"].items()
                if passed is False
            )
            if not false_checks:
                continue
            raw = _source_segment(payload, seed, acceptance["name"])
            substep = raw["physics_substep_audit"]
            if (
                int(substep["qpos_limit_violations"]) > 0
                and float(substep["joint_qpos_max_rad"]["left_knee"])
                > float(central.SAFE_JOINT_LIMITS["left_knee"][1])
            ):
                failures.append(
                    {
                        "seed": seed,
                        "segment": acceptance["name"],
                        "false_acceptance_checks": false_checks,
                        "qpos_limit_violations": int(
                            substep["qpos_limit_violations"]
                        ),
                        "maximum_qpos_excess_rad": float(
                            substep["maximum_qpos_excess_rad"]
                        ),
                        "left_knee_max_qpos_rad": float(
                            substep["joint_qpos_max_rad"]["left_knee"]
                        ),
                    }
                )

    near_bound = _source_segment(
        payload,
        NEAR_BOUND_PASSING_SEED,
        "transition_stand_after_reverse_turn_left",
    )
    near_acceptance = _source_acceptance(
        payload,
        NEAR_BOUND_PASSING_SEED,
        "transition_stand_after_reverse_turn_left",
    )
    evidence = {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "configuration_checks": configuration_checks,
        "configuration_matches_exact_replay": all(configuration_checks.values()),
        "left_knee_violation_failures": failures,
        "left_knee_violation_failure_count": len(failures),
        "known_failing_seed_set_matches": tuple(sorted({row["seed"] for row in failures}))
        == KNOWN_FAILING_TRANSITION_SEEDS,
        "near_bound_passing_case": {
            "seed": NEAR_BOUND_PASSING_SEED,
            "segment": near_bound["name"],
            "standard_acceptance_passed": near_acceptance["passed"] is True,
            "qpos_limit_violations": near_bound["physics_substep_audit"][
                "qpos_limit_violations"
            ],
            "left_knee_max_qpos_rad": near_bound["physics_substep_audit"][
                "joint_qpos_max_rad"
            ]["left_knee"],
            "remaining_safe_upper_margin_rad": float(
                central.SAFE_JOINT_LIMITS["left_knee"][1]
                - near_bound["physics_substep_audit"]["joint_qpos_max_rad"][
                    "left_knee"
                ]
            ),
        },
    }
    evidence["passed"] = bool(
        evidence["configuration_matches_exact_replay"]
        and evidence["left_knee_violation_failure_count"] == 5
        and evidence["known_failing_seed_set_matches"]
        and evidence["near_bound_passing_case"]["standard_acceptance_passed"]
        and evidence["near_bound_passing_case"]["qpos_limit_violations"] == 0
    )
    return payload, evidence


def _compact_segment(segment: Mapping[str, Any]) -> dict[str, Any]:
    acceptance = routed_contract.segment_acceptance(segment)
    safety = segment["safety_audit"]
    substep = segment["physics_substep_audit"]
    routing = segment["routing"]
    recovery = segment["backward_exit_recovery_audit"]
    checks = acceptance["checks"]
    hard_checks = {
        "central_segment_acceptance": acceptance["passed"] is True,
        "completed": segment["completed"] is True,
        "fall_free": segment["fell"] is False,
        "all_physics_substeps_audited": (
            int(substep["sample_count"])
            == int(segment["completed_physics_substeps"])
            == int(segment["expected_physics_substeps"])
        ),
        "contact_sample_count_exact": (
            int(substep["contact_sample_count"]) == int(substep["sample_count"])
            and substep["contact_sample_count_matches_sample_count"] is True
        ),
        "qpos_zero": (
            int(safety["qpos_limit_violations"]) == 0
            and int(substep["qpos_limit_violations"]) == 0
        ),
        "finite": (
            int(safety["nonfinite_sample_count"]) == 0
            and int(substep["nonfinite_state_samples"]) == 0
        ),
        "target_limit_margin_zero": (
            int(safety["applied_target_limit_violations"]) == 0
            and int(safety["desired_target_margin_violations"]) == 0
            and int(safety["unauthorized_applied_target_margin_violations"]) == 0
        ),
        "target_slew_zero": int(safety["target_slew_violations"]) == 0,
        "head_locked": (
            float(safety["applied_head_action_peak"]) == 0.0
            and float(safety["head_target_peak_rad"]) == 0.0
        ),
        "route_exact": all(
            bool(checks[name])
            for name in (
                "steady_route_expected_expert",
                "steady_route_expected_policy_role",
                "steady_route_sample_count",
                "prohibited_experts_absent",
                "atomic_endpoint_exact",
                "command_not_clipped",
                "reverse_entry_phase_audit",
            )
        ),
        "motion_contact_gates": all(
            bool(checks[name])
            for name in (
                "minimum_upright",
                "minimum_height",
                "primary_velocity",
                "signed_linear_progress",
                "orthogonal_velocity",
                "yaw_rate",
                "signed_yaw_progress",
                "stop_drift",
                "moving_single_support",
                "flight_rate",
            )
        ),
        "recovery_audit": recovery["passed"] is True,
    }
    return {
        "name": segment["name"],
        "passed": all(hard_checks.values()),
        "hard_checks": hard_checks,
        "standard_acceptance": acceptance,
        "completed": segment["completed"],
        "fell": segment["fell"],
        "expected_physics_substeps": segment["expected_physics_substeps"],
        "completed_physics_substeps": segment["completed_physics_substeps"],
        "motion_contact": {
            key: segment["metrics"][key]
            for key in (
                "projected_primary_velocity",
                "commanded_linear_speed",
                "primary_velocity_error",
                "absolute_orthogonal_velocity",
                "mean_local_yaw_rate",
                "yaw_rate_error",
                "planar_displacement",
                "minimum_height_m",
                "minimum_upright",
                "single_support_rate",
                "flight_rate",
                "contact_sample_count",
            )
        },
        "physics_substep_audit": {
            key: substep[key]
            for key in (
                "sample_count",
                "contact_sample_count",
                "contact_sample_count_matches_sample_count",
                "qpos_limit_violations",
                "maximum_qpos_excess_rad",
                "joint_qpos_max_rad",
                "joint_qpos_min_rad",
                "nonfinite_state_samples",
                "height_fall_samples",
                "upright_fall_samples",
                "minimum_height_m",
                "minimum_upright",
                "single_support_rate",
                "flight_rate",
            )
        },
        "target_head_slew_audit": {
            key: safety[key]
            for key in (
                "applied_target_limit_violations",
                "desired_target_margin_violations",
                "unauthorized_applied_target_margin_violations",
                "target_slew_violations",
                "qpos_limit_violations",
                "nonfinite_sample_count",
                "applied_head_action_peak",
                "head_target_peak_rad",
                "maximum_target_slew_rate_rad_per_s",
            )
        },
        "routing_audit": {
            "command_clip_events": routing["command_clip_events"],
            "prohibited_expert_steps": routing["prohibited_expert_steps"],
            "atomic_endpoint_mismatch_steps": routing[
                "atomic_endpoint_mismatch_steps"
            ],
            "steady_state_steps": routing["steady_state_steps"],
            "steady_state_routed_expert_steps": routing[
                "steady_state_routed_expert_steps"
            ],
            "steady_state_policy_role_steps": routing[
                "steady_state_policy_role_steps"
            ],
            "reverse_entry_phase": routing["reverse_entry_phase"],
        },
        "backward_exit_recovery_audit": recovery,
    }


def _summarize_runs(
    candidate: CapCandidate,
    runs: Sequence[Mapping[str, Any]],
    *,
    seed_scope: str,
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    all_segments: list[dict[str, Any]] = []
    state_acceptances: list[dict[str, Any]] = []
    for run in runs:
        segments = [_compact_segment(segment) for segment in run["segments"]]
        state_acceptance = routed_contract.backward_exit_recovery_state_acceptance(
            run["backward_exit_recovery_audit"]
        )
        state_acceptances.append(state_acceptance)
        episode_passed = bool(
            int(run["completed_segment_count"]) == len(PREFIX_SEGMENT_NAMES)
            and int(run["requested_segment_count"]) == len(PREFIX_SEGMENT_NAMES)
            and run["fell"] is False
            and run["reset_qpos_audit"]["passed"] is True
            and run["control_first_startup_audit"]["passed"] is True
            and state_acceptance["passed"] is True
            and all(segment["passed"] for segment in segments)
        )
        episodes.append(
            {
                "seed": int(run["seed"]),
                "passed": episode_passed,
                "reset_qpos_audit": run["reset_qpos_audit"],
                "control_first_startup_audit": run[
                    "control_first_startup_audit"
                ],
                "backward_exit_recovery_state_audit": run[
                    "backward_exit_recovery_audit"
                ],
                "backward_exit_recovery_state_acceptance": state_acceptance,
                "segments": segments,
            }
        )
        all_segments.extend(segments)

    moving = [
        segment
        for segment in all_segments
        if float(segment["motion_contact"]["commanded_linear_speed"]) > 0.0
        or abs(
            float(
                next(
                    source["command"][2]
                    for run in runs
                    for source in run["segments"]
                    if source["name"] == segment["name"]
                )
            )
        )
        > 0.0
    ]
    total_expected = sum(
        int(segment["expected_physics_substeps"]) for segment in all_segments
    )
    total_completed = sum(
        int(segment["completed_physics_substeps"]) for segment in all_segments
    )
    total_audited = sum(
        int(segment["physics_substep_audit"]["sample_count"])
        for segment in all_segments
    )
    total_contacts = sum(
        int(segment["physics_substep_audit"]["contact_sample_count"])
        for segment in all_segments
    )
    left_knee_max = max(
        float(
            segment["physics_substep_audit"]["joint_qpos_max_rad"]["left_knee"]
        )
        for segment in all_segments
    )
    summary = {
        "candidate": candidate.contract(),
        "seed_scope": seed_scope,
        "seeds": [int(run["seed"]) for run in runs],
        "episode_count": len(episodes),
        "passed_episode_count": sum(episode["passed"] for episode in episodes),
        "segment_count": len(all_segments),
        "passed_segment_count": sum(segment["passed"] for segment in all_segments),
        "fall_count": sum(segment["fell"] for segment in all_segments),
        "qpos_limit_violation_samples": sum(
            int(segment["physics_substep_audit"]["qpos_limit_violations"])
            for segment in all_segments
        ),
        "maximum_qpos_excess_rad": max(
            float(segment["physics_substep_audit"]["maximum_qpos_excess_rad"])
            for segment in all_segments
        ),
        "maximum_left_knee_qpos_rad": left_knee_max,
        "minimum_left_knee_safe_upper_margin_rad": (
            float(central.SAFE_JOINT_LIMITS["left_knee"][1]) - left_knee_max
        ),
        "expected_physics_substeps": total_expected,
        "completed_physics_substeps": total_completed,
        "audited_physics_substeps": total_audited,
        "contact_samples": total_contacts,
        "all_physics_and_contact_counts_exact": (
            total_expected == total_completed == total_audited == total_contacts
        ),
        "target_limit_margin_slew_violation_count": sum(
            int(segment["target_head_slew_audit"][key])
            for segment in all_segments
            for key in (
                "applied_target_limit_violations",
                "desired_target_margin_violations",
                "unauthorized_applied_target_margin_violations",
                "target_slew_violations",
            )
        ),
        "maximum_head_action_or_target_peak": max(
            max(
                float(segment["target_head_slew_audit"]["applied_head_action_peak"]),
                float(segment["target_head_slew_audit"]["head_target_peak_rad"]),
            )
            for segment in all_segments
        ),
        "route_violation_segment_count": sum(
            not segment["hard_checks"]["route_exact"] for segment in all_segments
        ),
        "motion_contact_violation_segment_count": sum(
            not segment["hard_checks"]["motion_contact_gates"]
            for segment in all_segments
        ),
        "minimum_height_m": min(
            float(segment["physics_substep_audit"]["minimum_height_m"])
            for segment in all_segments
        ),
        "minimum_upright": min(
            float(segment["physics_substep_audit"]["minimum_upright"])
            for segment in all_segments
        ),
        "minimum_moving_single_support_rate": min(
            float(segment["physics_substep_audit"]["single_support_rate"])
            for segment in moving
        ),
        "maximum_flight_rate": max(
            float(segment["physics_substep_audit"]["flight_rate"])
            for segment in all_segments
        ),
        "recovery_exit_event_count": sum(
            int(episode["backward_exit_recovery_state_audit"]["exit_event_count"])
            for episode in episodes
        ),
        "recovery_active_tick_count": sum(
            int(episode["backward_exit_recovery_state_audit"]["active_tick_count"])
            for episode in episodes
        ),
        "phase_entry_event_count": sum(
            int(segment["routing_audit"]["reverse_entry_phase"]["event_count"])
            for segment in all_segments
        ),
        "episodes": episodes,
    }
    summary["passed"] = bool(
        summary["passed_episode_count"] == summary["episode_count"]
        and summary["passed_segment_count"] == summary["segment_count"]
        and summary["fall_count"] == 0
        and summary["qpos_limit_violation_samples"] == 0
        and summary["maximum_qpos_excess_rad"] == 0.0
        and summary["all_physics_and_contact_counts_exact"]
        and summary["target_limit_margin_slew_violation_count"] == 0
        and summary["maximum_head_action_or_target_peak"] == 0.0
        and summary["route_violation_segment_count"] == 0
        and summary["motion_contact_violation_segment_count"] == 0
        and all(item["passed"] for item in state_acceptances)
    )
    return summary


def _run_candidate(
    candidate: CapCandidate,
    seeds: Sequence[int],
    *,
    evaluator: Any,
    bank: Any,
    mujoco: Any,
    runtime: Any,
    seed_scope: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    with _candidate_runtime_constants(candidate):
        simulator = central.RoutedSimulator(
            evaluator,
            bank,
            mujoco,
            runtime,
            leg_target_margin_rad=central.RUNTIME_TARGET_SAFETY_MARGIN_RAD,
            target_slew_rate_rad_s=central.RUNTIME_TARGET_SLEW_RATE_RAD_S,
            diagnostic_noncontract_safety=False,
            left_knee_extra_upper_margin_rad=(
                candidate.profile_extra_upper_margin_rad
            ),
            diagnostic_reverse_entry_phase_indices=(
                H1_BASELINE_PHASE_ENTRY_INDICES
            ),
            formal_candidate_default=False,
        )
        # Keep the exact Stage-A phase and recovery mechanics while decoupling
        # the recovery cap from the profile cap only inside this diagnostic.
        simulator.backward_exit_recovery_enabled = True
        simulator.reverse_entry_phase_indices = dict(PHASE_ENTRY_INDICES)
        simulator.phase_entry_status = DIAGNOSTIC_STATUS
        simulator.phase_entry_diagnostic_only = True
        runs = [
            simulator.run_schedule(
                exact_formal_transition_prefix(),
                seed=int(seed),
                joint_noise_scale=INITIAL_JOINT_NOISE_SCALE,
                initial_base_speed=INITIAL_BASE_SPEED_MPS,
                warmup_seconds=WARMUP_SECONDS,
            )
            for seed in seeds
        ]
        result = _summarize_runs(candidate, runs, seed_scope=seed_scope)
    result["elapsed_seconds"] = time.perf_counter() - start
    return result


def _baseline_reproduction(
    source_payload: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    replay_by_seed = {
        int(episode["seed"]): episode for episode in baseline["episodes"]
    }
    comparisons: list[dict[str, Any]] = []
    for seed in FOCUS_TRANSITION_SEEDS:
        source_episode = next(
            episode
            for episode in source_payload["suites"]["transitions"]["episodes"]
            if int(episode["seed"]) == seed
        )
        replay_episode = replay_by_seed[seed]
        replay_segments = {
            segment["name"]: segment for segment in replay_episode["segments"]
        }
        for source_segment in source_episode["segments"][:9]:
            replay_segment = replay_segments[source_segment["name"]]
            source_substep = source_segment["physics_substep_audit"]
            replay_substep = replay_segment["physics_substep_audit"]
            checks = {
                "qpos_limit_violations": (
                    int(replay_substep["qpos_limit_violations"])
                    == int(source_substep["qpos_limit_violations"])
                ),
                "maximum_qpos_excess_rad": (
                    float(replay_substep["maximum_qpos_excess_rad"])
                    == float(source_substep["maximum_qpos_excess_rad"])
                ),
                "left_knee_max_qpos_rad": (
                    float(replay_substep["joint_qpos_max_rad"]["left_knee"])
                    == float(source_substep["joint_qpos_max_rad"]["left_knee"])
                ),
                "completed_physics_substeps": (
                    int(replay_segment["completed_physics_substeps"])
                    == int(source_segment["completed_physics_substeps"])
                ),
                "fell": bool(replay_segment["fell"])
                == bool(source_segment["fell"]),
            }
            comparisons.append(
                {
                    "seed": seed,
                    "segment": source_segment["name"],
                    "passed": all(checks.values()),
                    "checks": checks,
                    "source": {
                        "qpos_limit_violations": source_substep[
                            "qpos_limit_violations"
                        ],
                        "maximum_qpos_excess_rad": source_substep[
                            "maximum_qpos_excess_rad"
                        ],
                        "left_knee_max_qpos_rad": source_substep[
                            "joint_qpos_max_rad"
                        ]["left_knee"],
                    },
                    "replay": {
                        "qpos_limit_violations": replay_substep[
                            "qpos_limit_violations"
                        ],
                        "maximum_qpos_excess_rad": replay_substep[
                            "maximum_qpos_excess_rad"
                        ],
                        "left_knee_max_qpos_rad": replay_substep[
                            "joint_qpos_max_rad"
                        ]["left_knee"],
                    },
                }
            )
    return {
        "passed": all(row["passed"] for row in comparisons),
        "comparison_count": len(comparisons),
        "exact_comparison_count": sum(row["passed"] for row in comparisons),
        "comparisons": comparisons,
    }


def _rank_key(result: Mapping[str, Any]) -> tuple[Any, ...]:
    candidate = result["candidate"]
    return (
        not bool(result["passed"]),
        float(candidate["recovery_extra_upper_margin_rad"]),
        int(candidate["recovery_hold_ticks"]),
        -float(result["minimum_left_knee_safe_upper_margin_rad"]),
        str(candidate["candidate_id"]),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite existing artifact: {args.output}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    script_path = Path(__file__).resolve()
    source_paths = _source_paths()
    source_closure_pre = _closure(source_paths)
    h1_payload, h1_evidence = _load_h1_source(SOURCE_H1_ARTIFACT)
    if not h1_evidence["passed"]:
        raise RuntimeError(f"H1 source contract mismatch: {h1_evidence}")

    schedule = exact_formal_transition_prefix()
    schedule_contract = [
        {
            "name": case[0],
            "physical_command": list(case[1]),
            "seconds": float(case[2]),
            "policy_observation_command": (
                None if case[3] is None else list(case[3])
            ),
            "expected_expert": case[4],
            "expected_policy_role": case[5],
        }
        for case in schedule
    ]
    asset_paths = central.generated_asset_paths(central.DEFAULT_GENERATED_ROOT.resolve())
    mujoco, onnxruntime, runtime, runtime_provenance = central._load_runtime(
        include_provenance=True
    )
    policy_paths = {
        role: BASE_POLICY.resolve() for role in routed_contract.REQUIRED_POLICY_ROLES
    }
    bank = central.RoutedPolicyBank(policy_paths, onnxruntime)
    evaluator = runtime.OfficialPolicyEvaluator(
        asset_paths["scene"], BASE_POLICY.resolve(), asset_paths["reference"]
    )
    evaluator.backward_residual_scale = 0.0
    evaluator.load_backward_profile(PROFILE_PATHS["straight"])
    evaluator.load_backward_turn_profile(1, PROFILE_PATHS["left"])
    evaluator.load_backward_turn_profile(-1, PROFILE_PATHS["right"])
    evaluator.backward_turn_minimum_yaw = 0.0
    evaluator.backward_turn_minimum_blend = 0.0
    evaluator.backward_turn_maximum_blend = 1.0

    print(
        "screening integrated phase7/rate1.05 baseline recovery cap=0.0125 "
        "hold=13 on focus seeds",
        flush=True,
    )
    baseline = _run_candidate(
        CapCandidate(BASELINE_CAP_RAD, BASELINE_HOLD_TICKS),
        FOCUS_TRANSITION_SEEDS,
        evaluator=evaluator,
        bank=bank,
        mujoco=mujoco,
        runtime=runtime,
        seed_scope="focus5_integrated_phase7_rate105_baseline_recovery",
    )
    print(
        json.dumps(
            {
                "baseline_integrated_passed": baseline["passed"],
                "qpos_violation_samples": baseline[
                    "qpos_limit_violation_samples"
                ],
                "maximum_qpos_excess_rad": baseline["maximum_qpos_excess_rad"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    focus_results: list[dict[str, Any]] = []
    for candidate in screening_candidates():
        print(
            "screening "
            f"recovery_cap={candidate.recovery_extra_upper_margin_rad:.5f} "
            f"hold={candidate.recovery_hold_ticks}",
            flush=True,
        )
        result = _run_candidate(
            candidate,
            FOCUS_TRANSITION_SEEDS,
            evaluator=evaluator,
            bank=bank,
            mujoco=mujoco,
            runtime=runtime,
            seed_scope="focus5_h1_failures_plus_near_bound_pass",
        )
        focus_results.append(result)
        print(
            json.dumps(
                {
                    "candidate_id": candidate.candidate_id,
                    "profile_cap": candidate.profile_extra_upper_margin_rad,
                    "recovery_cap": candidate.recovery_extra_upper_margin_rad,
                    "hold": candidate.recovery_hold_ticks,
                    "passed": result["passed"],
                    "qpos_violation_samples": result[
                        "qpos_limit_violation_samples"
                    ],
                    "maximum_qpos_excess_rad": result[
                        "maximum_qpos_excess_rad"
                    ],
                    "minimum_left_knee_safe_upper_margin_rad": result[
                        "minimum_left_knee_safe_upper_margin_rad"
                    ],
                    "falls": result["fall_count"],
                    "motion_contact_failures": result[
                        "motion_contact_violation_segment_count"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    focus_ranking = sorted(focus_results, key=_rank_key)
    focus_passing = [result for result in focus_ranking if result["passed"]]
    confirmation_attempts: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for focus_result in focus_passing:
        contract = focus_result["candidate"]
        candidate = CapCandidate(
            float(contract["recovery_extra_upper_margin_rad"]),
            int(contract["recovery_hold_ticks"]),
            float(contract["profile_extra_upper_margin_rad"]),
        )
        print(
            "confirming 20 seeds "
            f"recovery_cap={candidate.recovery_extra_upper_margin_rad:.5f} "
            f"hold={candidate.recovery_hold_ticks}",
            flush=True,
        )
        confirmation = _run_candidate(
            candidate,
            FORMAL_TRANSITION_SEEDS,
            evaluator=evaluator,
            bank=bank,
            mujoco=mujoco,
            runtime=runtime,
            seed_scope="all20_formal_transition_prefix",
        )
        confirmation_attempts.append(confirmation)
        print(
            json.dumps(
                {
                    "candidate_id": candidate.candidate_id,
                    "passed": confirmation["passed"],
                    "passed_episodes": confirmation["passed_episode_count"],
                    "qpos_violation_samples": confirmation[
                        "qpos_limit_violation_samples"
                    ],
                    "falls": confirmation["fall_count"],
                    "audited_physics_substeps": confirmation[
                        "audited_physics_substeps"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if confirmation["passed"]:
            selected = confirmation
            break

    source_closure_post = _closure(source_paths)
    source_unchanged = source_closure_pre == source_closure_post
    passed = bool(
        h1_evidence["passed"]
        and source_unchanged
        and selected is not None
    )
    payload = {
        "schema_version": 1,
        "artifact_kind": "openduckmini_h1_left_knee_cap_sweep",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DIAGNOSTIC_PASS" if passed else "DIAGNOSTIC_FAIL",
        "simulation_only": True,
        "diagnostic_not_adopted": True,
        "hardware_deployment": "PROHIBITED",
        "central_evaluator_or_contract_modified_by_this_script": False,
        "configuration": {
            "formal_master_seed": FORMAL_MASTER_SEED,
            "focus_transition_seeds": list(FOCUS_TRANSITION_SEEDS),
            "known_failing_transition_seeds": list(
                KNOWN_FAILING_TRANSITION_SEEDS
            ),
            "near_bound_passing_seed": NEAR_BOUND_PASSING_SEED,
            "formal_transition_seeds": list(FORMAL_TRANSITION_SEEDS),
            "moving_seconds": MOVING_SECONDS,
            "stand_seconds": STAND_SECONDS,
            "warmup_seconds": WARMUP_SECONDS,
            "initial_joint_noise_scale": INITIAL_JOINT_NOISE_SCALE,
            "initial_base_speed_mps": INITIAL_BASE_SPEED_MPS,
            "target_margin_rad": central.RUNTIME_TARGET_SAFETY_MARGIN_RAD,
            "target_slew_rate_rad_per_s": central.RUNTIME_TARGET_SLEW_RATE_RAD_S,
            "reverse_endpoint_mps": -0.05,
            "phase_preincrement_indices": PHASE_ENTRY_INDICES,
            "profile_extra_upper_margin_rad_fixed": BASELINE_CAP_RAD,
            "profile_caps_remain_unchanged_for_all_backward_families": True,
            "candidate_exit_recovery_caps_rad": list(SCREEN_CAPS_RAD),
            "candidate_recovery_hold_ticks": list(SCREEN_HOLD_TICKS),
            "schedule": schedule_contract,
            "all_backward_families_included": True,
        },
        "source_h1_evidence": h1_evidence,
        "integrated_baseline_focus5": baseline,
        "historical_h1_exact_reproduction": {
            "replayed_separately_before_integrated_run": True,
            "historical_source_is_immutable": True,
            "source_configuration_and_five_left_knee_failures_validated": (
                h1_evidence["passed"]
            ),
        },
        "focus5_screen": {
            "candidate_count": len(focus_results),
            "passing_candidate_count": len(focus_passing),
            "ranking_candidate_ids": [
                result["candidate"]["candidate_id"] for result in focus_ranking
            ],
            "results": focus_results,
        },
        "all20_confirmation": {
            "attempt_count": len(confirmation_attempts),
            "attempts": confirmation_attempts,
            "selected_candidate": (
                None if selected is None else selected["candidate"]
            ),
            "selected_summary": (
                None
                if selected is None
                else {
                    key: selected[key]
                    for key in (
                        "passed",
                        "episode_count",
                        "passed_episode_count",
                        "segment_count",
                        "passed_segment_count",
                        "fall_count",
                        "qpos_limit_violation_samples",
                        "maximum_qpos_excess_rad",
                        "maximum_left_knee_qpos_rad",
                        "minimum_left_knee_safe_upper_margin_rad",
                        "expected_physics_substeps",
                        "completed_physics_substeps",
                        "audited_physics_substeps",
                        "contact_samples",
                        "all_physics_and_contact_counts_exact",
                        "target_limit_margin_slew_violation_count",
                        "maximum_head_action_or_target_peak",
                        "route_violation_segment_count",
                        "motion_contact_violation_segment_count",
                        "minimum_height_m",
                        "minimum_upright",
                        "minimum_moving_single_support_rate",
                        "maximum_flight_rate",
                        "recovery_exit_event_count",
                        "recovery_active_tick_count",
                        "phase_entry_event_count",
                    )
                },
            ),
        },
        "selection": {
            "passed": selected is not None,
            "rule": (
                "focus5 hard-gate pass with the phase7/rate1.05 straight profile, "
                "then minimize exit-recovery extra upper margin, "
                "then minimize hold ticks; require independent all20 exact-prefix pass"
            ),
            "recommended_profile_extra_upper_margin_rad": (
                None
                if selected is None
                else selected["candidate"]["profile_extra_upper_margin_rad"]
            ),
            "recommended_exit_recovery_extra_upper_margin_rad": (
                None
                if selected is None
                else selected["candidate"]["recovery_extra_upper_margin_rad"]
            ),
            "recommended_profile_upper_target_rad": (
                None
                if selected is None
                else selected["candidate"]["profile_upper_target_rad"]
            ),
            "recommended_exit_recovery_upper_target_rad": (
                None
                if selected is None
                else selected["candidate"]["recovery_upper_target_rad"]
            ),
            "recommended_recovery_hold_ticks": (
                None
                if selected is None
                else selected["candidate"]["recovery_hold_ticks"]
            ),
            "requires_central_implementation_and_full_20x30_requalification": True,
            "adoption_status": "NOT_ADOPTED",
            "hardware_deployment": "PROHIBITED",
        },
        "provenance": {
            "script": {"path": str(script_path), "sha256": _sha256(script_path)},
            "policy": {
                "path": str(BASE_POLICY.resolve()),
                "sha256": _sha256(BASE_POLICY.resolve()),
            },
            "profiles": {
                key: {"path": str(path), "sha256": _sha256(path)}
                for key, path in PROFILE_PATHS.items()
            },
            "straight_selection_evidence": [
                {"path": str(path.resolve()), "sha256": _sha256(path.resolve())}
                for path in STRAIGHT_SELECTION_EVIDENCE_PATHS
            ],
            "runtime_dependency_provenance": runtime_provenance,
            "onnx_providers": bank.session_providers,
            "source_closure_pre": source_closure_pre,
            "source_closure_post": source_closure_post,
            "source_closure_unchanged": source_unchanged,
        },
        "recommendation": (
            "No candidate passed; retain the package block and investigate further."
            if selected is None
            else (
                "Use the selected constants only as a central implementation input, "
                "then rerun the complete frozen 20x30 release suite.  This diagnostic "
                "does not authorize hardware deployment."
            )
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "output_sha256": _sha256(output),
                "status": payload["status"],
                "historical_h1_source_validated": h1_evidence["passed"],
                "selected_candidate": payload["all20_confirmation"][
                    "selected_candidate"
                ],
                "hardware_deployment": "PROHIBITED",
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
