"""Diagnose a phase reset when entering the margin-aware reverse profile.

This simulation-only screen preserves the central one-guard control-first loop
and its per-physics-substep audit.  The only diagnostic change is local to this
subclass: on the first tick for which the router's effective ``vx`` activates
the backward feedforward profile, the profile phase is reset to the same phase
used by an independently reset reverse rollout (``4 + phase_rate`` steps).

Hardware deployment is PROHIBITED regardless of the result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = EXP_ROOT.parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for import_root in (EXP_ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from safe_gait_experts.routed_evaluation import (  # noqa: E402
    segment_acceptance,
    sha256_file,
)
from optimize_margin_aware_reverse import (  # noqa: E402
    DEFAULT_GENERATED_ROOT,
    DEFAULT_POLICY,
    MarginAwareReverseEvaluator,
    PerJointCapRoutedSimulator,
    ProfileParameters,
    SharedV22PolicyBank,
    TARGET_MARGIN_RAD,
    TARGET_SLEW_RAD_S,
    load_profile,
)


DEFAULT_PROFILE = (
    EXP_ROOT / "artifacts" / "optimized_reverse_margin050_slew200_candidate_v3.json"
)
DEFAULT_BASELINE = (
    EXP_ROOT
    / "artifacts"
    / "routed_candidate_v3_control_first_5x15_fulltrans_v1.json"
)
DEFAULT_OUTPUT = (
    EXP_ROOT
    / "artifacts"
    / "reverse_transition_candidate_v3_phase_reset_5x15_v1.json"
)
DEFAULT_SEEDS = tuple(22_260_808 + index for index in range(5))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay stand->forward->stand->reverse with a diagnostic reverse-entry "
            "phase reset and full central substep safety auditing."
        )
    )
    parser.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--stand-seconds", type=float, default=5.0)
    parser.add_argument("--forward-seconds", type=float, default=15.0)
    parser.add_argument("--reverse-seconds", type=float, default=15.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.5)
    parser.add_argument("--initial-joint-noise-scale", type=float, default=1.0)
    parser.add_argument("--initial-base-speed", type=float, default=0.10)
    parser.add_argument("--reset-qpos-inward-margin-rad", type=float, default=0.005)
    parser.add_argument("--minimum-reverse-vx", type=float, default=-0.02)
    parser.add_argument(
        "--router-preactivation-phase-steps",
        type=float,
        default=4.0,
        help=(
            "Non-backward phase increments before vx<-0.02 activates the profile; "
            "the frozen router ramp produces four."
        ),
    )
    parser.add_argument(
        "--activation-blend-seconds",
        type=float,
        default=0.0,
        help=(
            "Diagnostic blend from the policy-derived target to the backward "
            "feedforward target after vx<-0.02 activates it."
        ),
    )
    args = parser.parse_args(argv)
    if args.output.exists() and not args.overwrite:
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if len(args.seeds) < 5 or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds requires at least five distinct fixed seeds")
    for name in ("stand_seconds", "forward_seconds", "reverse_seconds"):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 <= args.warmup_seconds < args.reverse_seconds:
        parser.error("--warmup-seconds must be inside the reverse segment")
    if args.initial_joint_noise_scale != 1.0:
        parser.error("this full-perturbation diagnostic freezes joint noise scale at 1")
    if args.initial_base_speed != 0.10:
        parser.error("this full-perturbation diagnostic freezes base speed at 0.10")
    if args.reset_qpos_inward_margin_rad != 0.005:
        parser.error("positive-noise reset inward margin is frozen at 0.005 rad")
    if args.minimum_reverse_vx >= 0.0:
        parser.error("--minimum-reverse-vx must be negative")
    if not 0.0 <= args.activation_blend_seconds <= args.warmup_seconds:
        parser.error(
            "--activation-blend-seconds must be in [0, warmup-seconds]"
        )
    return args


class ReverseEntryPhaseResetSimulator(PerJointCapRoutedSimulator):
    """Reset only the backward profile phase at a non-backward entry."""

    def __init__(
        self,
        *args: Any,
        router_preactivation_phase_steps: float = 4.0,
        activation_blend_seconds: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.router_preactivation_phase_steps = float(
            router_preactivation_phase_steps
        )
        self.activation_blend_seconds = float(activation_blend_seconds)
        self.activation_blend_ticks = int(
            round(self.activation_blend_seconds / float(self.runtime.CONTROL_DT))
        )
        self._backward_active = False
        self._backward_phase_offset = 0.0
        self._backward_activation_tick = 0
        self._current_seed: int | None = None
        self.entry_events: list[dict[str, Any]] = []

    def begin_diagnostic_run(self, seed: int) -> None:
        self._backward_active = False
        self._backward_phase_offset = 0.0
        self._backward_activation_tick = 0
        self._current_seed = int(seed)

    def _policy_target(
        self,
        applied_action: np.ndarray,
        effective_command: np.ndarray,
        phase_index: float,
        default: np.ndarray,
    ) -> np.ndarray:
        backward_active = bool(effective_command[0] < -0.02)
        adjusted_phase = float(phase_index)
        if backward_active and not self._backward_active:
            _, _, phase_rate = self.evaluator.backward_parameters(
                float(effective_command[2])
            )
            independent_first_phase = (
                self.router_preactivation_phase_steps + float(phase_rate)
            ) % float(self.evaluator.phase_steps)
            self._backward_phase_offset = (
                independent_first_phase - float(phase_index)
            ) % float(self.evaluator.phase_steps)
            self.entry_events.append(
                {
                    "seed": self._current_seed,
                    "global_phase_index_at_activation": float(phase_index),
                    "independent_equivalent_phase_index": independent_first_phase,
                    "applied_phase_offset_steps": self._backward_phase_offset,
                    "profile_phase_rate": float(phase_rate),
                    "phase_steps": int(self.evaluator.phase_steps),
                    "effective_command": effective_command.tolist(),
                    "activation_blend_seconds": self.activation_blend_seconds,
                    "activation_blend_ticks": self.activation_blend_ticks,
                }
            )
        if backward_active:
            adjusted_phase = (
                float(phase_index) + self._backward_phase_offset
            ) % float(self.evaluator.phase_steps)
        targets = super()._policy_target(
            applied_action,
            effective_command,
            adjusted_phase,
            default,
        )
        if backward_active:
            self._backward_activation_tick += 1
            if self.activation_blend_ticks > 0:
                blend_alpha = min(
                    1.0,
                    self._backward_activation_tick / self.activation_blend_ticks,
                )
                policy_only_command = np.asarray(
                    effective_command, dtype=np.float64
                ).copy()
                policy_only_command[0] = 0.0
                policy_targets = super()._policy_target(
                    applied_action,
                    policy_only_command,
                    adjusted_phase,
                    default,
                )
                targets = (
                    (1.0 - blend_alpha) * policy_targets
                    + blend_alpha * targets
                )
                targets[self.left_knee_index] = min(
                    targets[self.left_knee_index],
                    self.left_knee_profile_upper_target_rad,
                )
        self._backward_active = backward_active
        if not backward_active:
            self._backward_phase_offset = 0.0
            self._backward_activation_tick = 0
        return targets


def _load_candidate(path: Path) -> tuple[ProfileParameters, float, dict[str, Any]]:
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    composition = payload.get("composition")
    if not isinstance(composition, Mapping):
        raise ValueError("candidate profile is missing composition")
    cap = float(composition.get("left_knee_extra_upper_margin_rad", 0.0))
    if not np.isfinite(cap) or not 0.0 <= cap <= 0.05:
        raise ValueError("candidate left-knee extra upper margin is invalid")
    if composition.get("backward_residual_scale") != 0.0:
        raise ValueError("candidate backward residual must remain zero")
    if composition.get("leg_target_margin_rad") != TARGET_MARGIN_RAD:
        raise ValueError("candidate leg target margin must remain 0.050 rad")
    if composition.get("target_slew_rate_rad_per_s") != TARGET_SLEW_RAD_S:
        raise ValueError("candidate target slew must remain 2.0 rad/s")
    if payload.get("hardware_deployment") != "PROHIBITED":
        raise ValueError("candidate must explicitly prohibit hardware deployment")
    return load_profile(resolved), cap, payload


def _compact_segment(segment: Mapping[str, Any]) -> dict[str, Any]:
    metrics = segment["metrics"]
    safety = segment["safety_audit"]
    substep = segment["physics_substep_audit"]
    routing = segment["routing"]
    return {
        "name": segment["name"],
        "completed": bool(segment["completed"]),
        "completed_seconds": float(segment["completed_seconds"]),
        "fell": bool(segment["fell"]),
        "mean_local_velocity_xyz": metrics["mean_local_velocity_xyz"],
        "mean_local_yaw_rate": float(metrics["mean_local_yaw_rate"]),
        "projected_primary_velocity": float(metrics["projected_primary_velocity"]),
        "commanded_linear_speed": float(metrics["commanded_linear_speed"]),
        "signed_linear_progress_fraction": (
            float(metrics["projected_primary_velocity"])
            / max(float(metrics["commanded_linear_speed"]), 1e-12)
        ),
        "minimum_height_m": float(substep["minimum_height_m"]),
        "minimum_upright": float(substep["minimum_upright"]),
        "physics_substep_audit": {
            key: substep[key]
            for key in (
                "sample_count",
                "qpos_limit_violations",
                "maximum_qpos_excess_rad",
                "nonfinite_state_samples",
                "height_fall_samples",
                "upright_fall_samples",
                "fall_or_nonfinite_detected",
            )
        },
        "target_and_head_audit": {
            key: safety[key]
            for key in (
                "applied_target_limit_violations",
                "desired_target_margin_violations",
                "unauthorized_applied_target_margin_violations",
                "target_slew_violations",
                "applied_head_action_peak",
                "head_target_peak_rad",
                "qpos_limit_violations",
            )
        },
        "routing": {
            key: routing[key]
            for key in (
                "steady_state_steps",
                "steady_state_routed_expert_steps",
                "steady_state_policy_role_steps",
                "prohibited_expert_steps",
                "command_clip_events",
            )
        },
    }


def _baseline_evidence(path: Path) -> dict[str, Any] | None:
    resolved = path.resolve()
    if not resolved.is_file():
        return None
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    rows = []
    for episode in payload.get("suites", {}).get("transitions", {}).get(
        "episodes", []
    ):
        matches = [
            segment
            for segment in episode.get("segments", [])
            if segment.get("name") == "transition_reverse"
        ]
        if len(matches) != 1:
            continue
        segment = matches[0]
        metrics = segment["metrics"]
        rows.append(
            {
                "seed": int(episode["seed"]),
                "fell": bool(segment["fell"]),
                "completed_seconds": float(segment["completed_seconds"]),
                "mean_vx": float(metrics["mean_local_velocity_xyz"][0]),
                "signed_linear_progress_fraction": (
                    float(metrics["projected_primary_velocity"])
                    / max(float(metrics["commanded_linear_speed"]), 1e-12)
                ),
                "minimum_upright": float(
                    segment["physics_substep_audit"]["minimum_upright"]
                ),
            }
        )
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "transition_reverse": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    profile_path = args.profile.resolve()
    parameters, knee_cap, profile_payload = _load_candidate(profile_path)
    evaluator = MarginAwareReverseEvaluator(args)
    evaluator.evaluator.backward_gait_scales = parameters.amplitude_scales.copy()
    evaluator.evaluator.backward_gait_biases = parameters.bias_offsets.copy()
    evaluator.evaluator.backward_phase_rate = float(parameters.phase_rate)
    evaluator.evaluator.backward_residual_scale = 0.0
    bank = SharedV22PolicyBank(evaluator.evaluator)
    simulator = ReverseEntryPhaseResetSimulator(
        evaluator.evaluator,
        bank,
        evaluator.mujoco,
        evaluator.runtime,
        leg_target_margin_rad=TARGET_MARGIN_RAD,
        target_slew_rate_rad_s=TARGET_SLEW_RAD_S,
        diagnostic_noncontract_safety=False,
        left_knee_extra_upper_margin_rad=knee_cap,
        router_preactivation_phase_steps=args.router_preactivation_phase_steps,
        activation_blend_seconds=args.activation_blend_seconds,
    )
    schedule = (
        (
            "transition_stand_0",
            (0.0, 0.0, 0.0),
            args.stand_seconds,
            None,
            "stand",
            "stand",
        ),
        (
            "transition_forward",
            (0.05, 0.0, 0.0),
            args.forward_seconds,
            (0.10, 0.0, 0.0),
            "forward",
            "forward",
        ),
        (
            "transition_stand_after_forward",
            (0.0, 0.0, 0.0),
            args.stand_seconds,
            None,
            "stand",
            "stand",
        ),
        (
            "transition_reverse",
            (-0.050, 0.0, 0.0),
            args.reverse_seconds,
            None,
            "reverse",
            "reverse",
        ),
    )

    episodes = []
    for seed in args.seeds:
        simulator.begin_diagnostic_run(seed)
        run = simulator.run_schedule(
            schedule,
            seed=int(seed),
            joint_noise_scale=args.initial_joint_noise_scale,
            initial_base_speed=args.initial_base_speed,
            warmup_seconds=args.warmup_seconds,
        )
        reverse = [
            segment
            for segment in run["segments"]
            if segment["name"] == "transition_reverse"
        ][0]
        acceptance = segment_acceptance(reverse)
        event = [
            item for item in simulator.entry_events if item["seed"] == int(seed)
        ][-1]
        episodes.append(
            {
                "seed": int(seed),
                "phase_reset_event": event,
                "reverse_segment": _compact_segment(reverse),
                "reverse_acceptance": acceptance,
                "all_prefix_segments_completed_without_fall": all(
                    segment["completed"] and not segment["fell"]
                    for segment in run["segments"][:-1]
                ),
                "reset_qpos_audit": run["reset_qpos_audit"],
                "control_first_startup_audit": run[
                    "control_first_startup_audit"
                ],
            }
        )

    velocities = np.asarray(
        [episode["reverse_segment"]["mean_local_velocity_xyz"][0] for episode in episodes]
    )
    progress = np.asarray(
        [
            episode["reverse_segment"]["signed_linear_progress_fraction"]
            for episode in episodes
        ]
    )
    passed = bool(
        all(episode["reverse_acceptance"]["passed"] for episode in episodes)
        and all(
            episode["reverse_segment"]["physics_substep_audit"][
                "qpos_limit_violations"
            ]
            == 0
            and not episode["reverse_segment"]["fell"]
            for episode in episodes
        )
        and float(np.max(velocities)) <= args.minimum_reverse_vx
        and float(np.min(progress)) >= 0.30
    )
    output = args.output.resolve()
    payload = {
        "schema_version": 1,
        "artifact_kind": "openduckmini_reverse_transition_phase_reset_diagnostic",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "TRANSITION_SCREEN_PASS" if passed else "TRANSITION_SCREEN_FAIL",
        "hardware_deployment": "PROHIBITED",
        "simulation_only": True,
        "diagnostic_unadopted": True,
        "configuration": {
            "schedule": [
                {
                    "name": item[0],
                    "command": list(item[1]),
                    "seconds": item[2],
                    "policy_observation_command": (
                        None if item[3] is None else list(item[3])
                    ),
                    "expected_expert": item[4],
                    "expected_policy_role": item[5],
                }
                for item in schedule
            ],
            "fixed_perturb_seeds": [int(seed) for seed in args.seeds],
            "initial_joint_noise_scale": args.initial_joint_noise_scale,
            "initial_base_speed_max_mps": args.initial_base_speed,
            "warmup_seconds": args.warmup_seconds,
            "leg_target_margin_rad": TARGET_MARGIN_RAD,
            "target_slew_rate_rad_per_s": TARGET_SLEW_RAD_S,
            "left_knee_extra_upper_margin_rad": knee_cap,
            "backward_residual_scale": 0.0,
            "head_target_indices": [5, 6, 7, 8],
            "head_target_value": 0.0,
            "phase_reset": {
                "enabled": True,
                "activation": "first effective vx < -0.02 after non-backward",
                "independent_equivalent_phase_formula": (
                    "router_preactivation_phase_steps + profile_phase_rate"
                ),
                "router_preactivation_phase_steps": (
                    args.router_preactivation_phase_steps
                ),
                "activation_blend": {
                    "seconds": args.activation_blend_seconds,
                    "ticks": int(
                        round(
                            args.activation_blend_seconds
                            / float(evaluator.runtime.CONTROL_DT)
                        )
                    ),
                    "source": "policy-derived target with backward vx disabled",
                    "destination": "phase-reset backward feedforward target",
                    "left_knee_extra_cap_reapplied_after_blend": True,
                },
            },
        },
        "candidate_profile": {
            "path": str(profile_path),
            "sha256": sha256_file(profile_path),
            "release_id": profile_payload.get("release_id"),
            "parameters": parameters.as_json(),
        },
        "provenance": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "optimizer_script": str(
                (SCRIPT_DIR / "optimize_margin_aware_reverse.py").resolve()
            ),
            "optimizer_script_sha256": sha256_file(
                (SCRIPT_DIR / "optimize_margin_aware_reverse.py").resolve()
            ),
            "central_evaluator_script": str(
                (SCRIPT_DIR / "evaluate_routed_transitions.py").resolve()
            ),
            "central_evaluator_script_sha256": sha256_file(
                (SCRIPT_DIR / "evaluate_routed_transitions.py").resolve()
            ),
            "central_package": str(
                (EXP_ROOT / "safe_gait_experts" / "routed_evaluation.py").resolve()
            ),
            "central_package_sha256": sha256_file(
                EXP_ROOT / "safe_gait_experts" / "routed_evaluation.py"
            ),
            "base_policy": {
                "path": str(args.policy.resolve()),
                "sha256": sha256_file(args.policy.resolve()),
            },
            "exact_generated_assets": evaluator.asset_evidence,
            "model_contract": evaluator.model_evidence,
            "baseline_no_phase_reset": _baseline_evidence(args.baseline),
        },
        "acceptance": {
            "passed": passed,
            "required_episode_passes": len(episodes),
            "actual_episode_passes": sum(
                episode["reverse_acceptance"]["passed"] for episode in episodes
            ),
            "requires_zero_falls_qpos_target_head_slew_and_route_violations": True,
            "minimum_signed_linear_progress_fraction": 0.30,
            "maximum_worst_vx": args.minimum_reverse_vx,
        },
        "metrics": {
            "episode_count": len(episodes),
            "mean_vx": float(np.mean(velocities)),
            "worst_vx": float(np.max(velocities)),
            "best_vx": float(np.min(velocities)),
            "minimum_signed_linear_progress_fraction": float(np.min(progress)),
            "falls": sum(episode["reverse_segment"]["fell"] for episode in episodes),
            "physics_substeps_audited": sum(
                episode["reverse_segment"]["physics_substep_audit"]["sample_count"]
                for episode in episodes
            ),
        },
        "episodes": episodes,
        "adoption": {
            "status": "NOT_ADOPTED_DIAGNOSTIC_ONLY",
            "phase_reset_requires_central_contract_review": True,
            "simulation_pass_does_not_authorize_hardware": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "status": payload["status"],
                "episode_passes": payload["acceptance"]["actual_episode_passes"],
                **payload["metrics"],
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
