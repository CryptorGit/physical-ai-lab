"""Screen route-activation phase resets on the short H3 showcase schedule.

This is an isolated PDCA diagnostic.  It monkeypatches only process-local
call sites, never changes the central contract/package, never injects base
velocity or joint noise, and preserves the final guard plus every physics
substep audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
)
DEFAULT_OUTPUT = EXP_ROOT / "artifacts" / "h3_continuous_motion_entry_screen_v1.json"

MOTION_SCHEDULE: tuple[
    tuple[
        str,
        tuple[float, float, float],
        float,
        tuple[float, float, float] | None,
        str,
        str,
    ],
    ...,
] = (
    ("intro_stand", (0.0, 0.0, 0.0), 1.5, None, "stand", "stand"),
    ("forward", (0.05, 0.0, 0.0), 4.0, (0.10, 0.0, 0.0), "forward", "forward"),
    ("stand_after_forward", (0.0, 0.0, 0.0), 5.0, None, "stand", "stand"),
    ("reverse", (-0.05, 0.0, 0.0), 4.5, None, "reverse", "reverse"),
    ("stand_after_reverse", (0.0, 0.0, 0.0), 5.0, None, "stand", "stand"),
    ("lateral_left", (0.0, 0.06, 0.0), 3.5, (0.0, 0.10, 0.0), "lateral_left", "lateral_left"),
    ("stand_after_lateral_left", (0.0, 0.0, 0.0), 5.0, None, "stand", "stand"),
    ("lateral_right", (0.0, -0.06, 0.0), 3.5, (0.0, -0.10, 0.0), "lateral_right", "lateral_right"),
    ("stand_after_lateral_right", (0.0, 0.0, 0.0), 5.0, None, "stand", "stand"),
    ("yaw_left", (0.0, 0.0, 0.30), 3.5, (0.0, -0.06, 0.60), "yaw_left", "yaw_left"),
    ("stand_after_yaw_left", (0.0, 0.0, 0.0), 5.0, None, "stand", "stand"),
    ("yaw_right", (0.0, 0.0, -0.30), 3.5, (0.0, 0.0, -0.80), "yaw_right", "yaw_right"),
    ("stand_after_yaw_right", (0.0, 0.0, 0.0), 5.0, None, "stand", "stand"),
    (
        "reverse_turn_left",
        (-0.03, 0.0, 0.20),
        4.0,
        None,
        "reverse_turn_left",
        "compound",
    ),
    (
        "stand_after_reverse_turn_left",
        (0.0, 0.0, 0.0),
        5.0,
        None,
        "stand",
        "stand",
    ),
    (
        "reverse_turn_right",
        (-0.04, 0.0, -0.20),
        4.0,
        None,
        "reverse_turn_right",
        "compound",
    ),
    (
        "stand_after_reverse_turn_right",
        (0.0, 0.0, 0.0),
        5.0,
        None,
        "stand",
        "stand",
    ),
    (
        "forward_turn_left",
        (0.04, 0.0, 0.30),
        4.0,
        (0.08, 0.0, 0.30),
        "compound",
        "compound",
    ),
    (
        "stand_after_forward_turn_left",
        (0.0, 0.0, 0.0),
        5.0,
        None,
        "stand",
        "stand",
    ),
    (
        "forward_turn_right",
        (0.04, 0.0, -0.22),
        4.0,
        (0.08, 0.0, -0.45),
        "compound",
        "compound",
    ),
    (
        "forward_lateral_left_turn",
        (0.04, 0.05, 0.17),
        4.0,
        (0.06, 0.05, 0.20),
        "compound",
        "compound",
    ),
    (
        "forward_lateral_right_turn",
        (0.04, -0.03, -0.15),
        4.0,
        (0.06, -0.05, -0.35),
        "compound",
        "compound",
    ),
    ("final_stand", (0.0, 0.0, 0.0), 5.0, None, "stand", "stand"),
)

DEFAULT_PHASES: Mapping[str, float] = {
    "forward": 13.0,
    "yaw_left": 6.0,
    "compound_left": 10.0,
    "compound_right": 11.0,
    "reverse_turn_left": 15.0,
    "reverse_turn_right": 6.0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _family(expert: str, command: np.ndarray) -> str | None:
    if expert in {"stand", "reverse", "reverse_turn_left", "reverse_turn_right"}:
        return None
    if expert == "compound":
        return "compound_left" if float(command[2]) >= 0.0 else "compound_right"
    return expert


def _motion_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in result["segments"]:
        if str(segment["name"]).startswith("stand") or segment["name"] == "intro_stand" or segment["name"] == "final_stand":
            continue
        metrics = segment["metrics"]
        command = np.asarray(segment["command"], dtype=np.float64)
        primary = float(metrics["projected_primary_velocity"])
        linear_command = float(np.hypot(command[0], command[1]))
        linear_ratio = None if linear_command == 0.0 else primary / linear_command
        yaw_command = abs(float(command[2]))
        yaw_rate = abs(float(metrics["mean_local_yaw_rate"]))
        yaw_ratio = None if yaw_command == 0.0 else yaw_rate / yaw_command
        required = [value for value in (linear_ratio, yaw_ratio) if value is not None]
        motion_pass = all(value >= 0.30 for value in required)
        rows.append(
            {
                "name": segment["name"],
                "completed": bool(segment["completed"]),
                "fell": bool(segment["fell"]),
                "central_acceptance": bool(segment_acceptance(segment)["passed"]),
                "linear_progress_ratio": linear_ratio,
                "yaw_progress_ratio": yaw_ratio,
                "short_motion_pass": bool(motion_pass),
                "physics_substep_qpos_violations": int(
                    segment["physics_substep_audit"]["qpos_limit_violations"]
                ),
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument(
        "--independent",
        action="store_true",
        help="Reset exact-home state, router, phase, and action history for each motion.",
    )
    parser.add_argument(
        "--uniform-motion-seconds",
        type=float,
        default=None,
        help="Override every non-stand motion duration for horizon screening.",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=0.0,
        help="Motion-metric warmup passed unchanged to the central evaluator.",
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if args.uniform_motion_seconds is not None and args.uniform_motion_seconds <= 0.0:
        parser.error("--uniform-motion-seconds must be positive")
    if args.warmup_seconds < 0.0:
        parser.error("--warmup-seconds must be non-negative")

    mujoco, onnxruntime, runtime = central._load_runtime()
    assets = central.generated_asset_paths(central.DEFAULT_GENERATED_ROOT.resolve())
    policies = {role: BASE_POLICY.resolve() for role in central.REQUIRED_POLICY_ROLES}
    bank = central.RoutedPolicyBank(policies, onnxruntime)
    evaluator = runtime.OfficialPolicyEvaluator(
        assets["scene"], BASE_POLICY.resolve(), assets["reference"]
    )
    evaluator.backward_residual_scale = 0.0
    evaluator.load_backward_profile(central.DEFAULT_BACKWARD_PROFILE)
    evaluator.load_backward_turn_profile(1, central.DEFAULT_BACKWARD_LEFT_PROFILE)
    evaluator.load_backward_turn_profile(-1, central.DEFAULT_BACKWARD_RIGHT_PROFILE)
    evaluator.backward_turn_minimum_yaw = 0.0
    evaluator.backward_turn_minimum_blend = 0.0
    evaluator.backward_turn_maximum_blend = 1.0
    central.validate_model_contract(evaluator)
    simulator = central.RoutedSimulator(
        evaluator,
        bank,
        mujoco,
        runtime,
        leg_target_margin_rad=0.05,
        target_slew_rate_rad_s=2.0,
        diagnostic_noncontract_safety=False,
        left_knee_extra_upper_margin_rad=0.0125,
        formal_candidate_default=True,
    )

    original = central.advance_routed_phase
    state: dict[str, Any] = {"active_family": None, "events": []}

    def diagnostic_advance(phase_index: float, **kwargs: Any) -> tuple[float, bool, Any]:
        command = np.asarray(kwargs["effective_command"], dtype=np.float64)
        expert = str(kwargs["current_expert"])
        family = _family(expert, command)
        active = family is not None and bool(np.any(command != 0.0))
        event = bool(active and state["active_family"] != family)
        if event and not args.baseline and family in DEFAULT_PHASES:
            phase_index = float(DEFAULT_PHASES[family])
            state["events"].append(
                {
                    "global_control_tick": kwargs.get("global_control_tick"),
                    "expert": expert,
                    "family": family,
                    "effective_command": command.tolist(),
                    "preincrement_phase_index": phase_index,
                }
            )
        result_phase, backward_active, reverse_event = original(phase_index, **kwargs)
        # The central adopted reverse event remains authoritative.  This
        # diagnostic then substitutes only the screened turn entry index.
        if (
            reverse_event is not None
            and not args.baseline
            and expert in {"reverse_turn_left", "reverse_turn_right"}
        ):
            desired = float(DEFAULT_PHASES[expert])
            delta = float(kwargs["phase_delta"])
            result_phase = (desired + delta) % float(kwargs["phase_steps"])
            reverse_event = dict(reverse_event)
            reverse_event["diagnostic_postcentral_preincrement_phase_index"] = desired
            reverse_event["first_feedforward_phase_index"] = result_phase
        state["active_family"] = family if active else None
        return result_phase, backward_active, reverse_event

    central.advance_routed_phase = diagnostic_advance
    try:
        if args.independent:
            segments: list[dict[str, Any]] = []
            for scheduled_case in MOTION_SCHEDULE:
                if scheduled_case[4] == "stand":
                    continue
                if args.uniform_motion_seconds is not None:
                    scheduled_case = (
                        scheduled_case[0],
                        scheduled_case[1],
                        float(args.uniform_motion_seconds),
                        scheduled_case[3],
                        scheduled_case[4],
                        scheduled_case[5],
                    )
                state["active_family"] = None
                isolated = simulator.run_schedule(
                    (scheduled_case,),
                    seed=20260808,
                    joint_noise_scale=0.0,
                    initial_base_speed=0.0,
                    warmup_seconds=float(args.warmup_seconds),
                )
                segments.extend(isolated["segments"])
            result = {"segments": segments}
        else:
            result = simulator.run_schedule(
                MOTION_SCHEDULE,
                seed=20260808,
                joint_noise_scale=0.0,
                initial_base_speed=0.0,
                warmup_seconds=float(args.warmup_seconds),
            )
    finally:
        central.advance_routed_phase = original

    rows = _motion_rows(result)
    payload = {
        "artifact_kind": "openduckmini_h3_continuous_motion_entry_diagnostic",
        "status": "DIAGNOSTIC_PASS" if all(row["short_motion_pass"] for row in rows) else "DIAGNOSTIC_FAIL",
        "central_sources_modified": False,
        "physical_base_velocity_injected": False,
        "joint_noise_scale": 0.0,
        "initial_base_speed": 0.0,
        "independent_exact_home_reset_per_motion": bool(args.independent),
        "uniform_motion_seconds": args.uniform_motion_seconds,
        "warmup_seconds": float(args.warmup_seconds),
        "candidate": "baseline" if args.baseline else dict(DEFAULT_PHASES),
        "activation_predicate": "nonback_effective_command_nonzero_family_false_to_true",
        "phase_reset_events": state["events"],
        "motions": rows,
        "all_short_motions_passed": all(row["short_motion_pass"] for row in rows),
        "all_segments_completed": all(bool(segment["completed"]) for segment in result["segments"]),
        "total_physics_substeps": sum(int(segment["completed_physics_substeps"]) for segment in result["segments"]),
        "total_qpos_violations": sum(int(segment["physics_substep_audit"]["qpos_limit_violations"]) for segment in result["segments"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": _sha256(output), "status": payload["status"], "motions": rows}, indent=2))


if __name__ == "__main__":
    main()
