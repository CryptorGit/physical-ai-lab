"""Synchronize H4 reverse force-contact slip peaks with control and joint state."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from scripts import diagnose_h4_reverse_pdca as pdca  # noqa: E402


central = pdca.central
EXPECTED_CENTRAL_SHA256 = "c4a0fff4d8726dc46ec462a6a5f32c599decc211a269779806f30559baf978c5"
EXPECTED_GAIT_QUALITY_SHA256 = "20a5010037f2157a089501e012881cabceed794bc77b1cca8ca5eaf6f7e88b61"
EXPECTED_ROUTED_EVALUATION_SHA256 = "fff136407b64090be41498f97eba5274e1cae4d373d3519fdc805226851ae047"
FORCE_CONTACT_THRESHOLD_BODY_WEIGHT = 0.010
TOP_PEAK_COUNT = 20
SEED_ROLES = {"worst": 20_265_810, "median": 20_260_810}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"sample_count": 0, "mean": 0.0, "rms": 0.0, "p95": 0.0, "maximum": 0.0}
    return {
        "sample_count": int(array.size),
        "mean": float(array.mean()),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "p95": float(np.percentile(array, 95.0)),
        "maximum": float(array.max()),
    }


def _correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.size < 3 or right.size != left.size:
        return None
    if float(np.ptp(left)) <= 1.0e-12 or float(np.ptp(right)) <= 1.0e-12:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


class SynchronizedProbe:
    """Process-local wrappers; central source remains byte-identical."""

    def __init__(self, simulator: Any, evaluator: Any, runtime: Any):
        self.simulator = simulator
        self.evaluator = evaluator
        self.runtime = runtime
        self.joint_names = tuple(simulator.joint_names)
        self.leg_indices = np.asarray(evaluator.backward_actuator_indices, dtype=int)
        self.leg_joint_names = tuple(self.joint_names[index] for index in self.leg_indices)
        self.phase_steps = float(evaluator.phase_steps)
        if self.phase_steps <= 0.0 or self.phase_steps != int(self.phase_steps):
            raise ValueError("profile phase_steps must be a positive integer")
        self.dof_indices = np.asarray(
            simulator.model.jnt_dofadr[evaluator.actuator_joint_ids], dtype=int
        )
        self.original_policy_target = simulator._policy_target
        self.original_quality_contact = simulator._quality_contact_kinematics
        self.original_guard_step = central.FinalTargetSafetyGuard.step
        self.phase_context: dict[str, Any] | None = None
        self.control_context: dict[str, Any] | None = None
        self.control_tick = -1
        self.origin_time: float | None = None
        self.samples: list[dict[str, Any]] = []

    def install(self) -> None:
        probe = self

        def policy_target(
            applied_action: np.ndarray,
            effective_command: np.ndarray,
            phase_index: float,
            default: np.ndarray,
        ) -> np.ndarray:
            targets = probe.original_policy_target(
                applied_action, effective_command, phase_index, default
            )
            probe.phase_context = {
                "phase_index": float(phase_index),
                "phase_fraction": float(
                    (phase_index % probe.phase_steps) / probe.phase_steps
                ),
                "effective_command": np.asarray(effective_command, dtype=np.float64).tolist(),
                "applied_policy_action": np.asarray(applied_action, dtype=np.float64)[
                    probe.leg_indices
                ].tolist(),
            }
            return targets

        def guard_step(guard: Any, targets: Sequence[float], dt: float) -> np.ndarray:
            raw = np.asarray(targets, dtype=np.float64)
            previous = guard.previous_targets
            desired = central.apply_final_target_safety(
                raw,
                central.SAFE_JOINT_LIMITS,
                margin_rad=central.RUNTIME_TARGET_SAFETY_MARGIN_RAD,
            )
            applied = probe.original_guard_step(guard, raw, dt)
            probe.control_tick += 1
            probe.control_context = {
                **({} if probe.phase_context is None else probe.phase_context),
                "control_tick": probe.control_tick,
                "raw_target": raw.copy(),
                "desired_margin_clamped_target": desired.copy(),
                "previous_applied_target": previous.copy(),
                "applied_target": np.asarray(applied, dtype=np.float64).copy(),
                "margin_clamp_delta": desired - raw,
                "slew_gap": desired - np.asarray(applied, dtype=np.float64),
            }
            return applied

        def quality_contact(data: Any) -> tuple[np.ndarray, np.ndarray]:
            normal_force, slip_speed = probe.original_quality_contact(data)
            now = float(data.time)
            if probe.origin_time is None:
                probe.origin_time = now
            context = probe.control_context
            if context is not None:
                qpos = np.asarray(
                    data.qpos[probe.evaluator.actuator_qpos_addr], dtype=np.float64
                )
                qvel = np.asarray(data.qvel[probe.dof_indices], dtype=np.float64)
                applied = context["applied_target"]
                sample_index = len(probe.samples)
                probe.samples.append(
                    {
                        "sample_index": sample_index,
                        "time_s": now - probe.origin_time,
                        "control_tick": int(context["control_tick"]),
                        "substep_in_control_tick": sample_index % int(probe.runtime.DECIMATION),
                        "phase_index": float(context["phase_index"]),
                        "phase_fraction": float(context["phase_fraction"]),
                        "normal_force_fraction_body_weight": np.asarray(
                            normal_force, dtype=np.float64
                        ).copy(),
                        "force_contact_slip_mps": np.asarray(
                            slip_speed, dtype=np.float64
                        ).copy(),
                        "leg_qpos_rad": qpos[probe.leg_indices].copy(),
                        "leg_qvel_radps": qvel[probe.leg_indices].copy(),
                        "leg_raw_target_rad": context["raw_target"][
                            probe.leg_indices
                        ].copy(),
                        "leg_desired_margin_clamped_target_rad": context[
                            "desired_margin_clamped_target"
                        ][probe.leg_indices].copy(),
                        "leg_applied_target_rad": applied[probe.leg_indices].copy(),
                        "leg_margin_clamp_delta_rad": context["margin_clamp_delta"][
                            probe.leg_indices
                        ].copy(),
                        "leg_slew_gap_rad": context["slew_gap"][
                            probe.leg_indices
                        ].copy(),
                        "leg_tracking_error_rad": (
                            applied[probe.leg_indices] - qpos[probe.leg_indices]
                        ).copy(),
                    }
                )
            return normal_force, slip_speed

        self.simulator._policy_target = policy_target
        self.simulator._quality_contact_kinematics = quality_contact
        central.FinalTargetSafetyGuard.step = guard_step

    def restore(self) -> None:
        self.simulator._policy_target = self.original_policy_target
        self.simulator._quality_contact_kinematics = self.original_quality_contact
        central.FinalTargetSafetyGuard.step = self.original_guard_step

    def _event(self, sample: Mapping[str, Any], foot: int) -> dict[str, Any]:
        arrays = {
            name: np.asarray(sample[name], dtype=np.float64)
            for name in (
                "leg_qpos_rad",
                "leg_qvel_radps",
                "leg_raw_target_rad",
                "leg_desired_margin_clamped_target_rad",
                "leg_applied_target_rad",
                "leg_margin_clamp_delta_rad",
                "leg_slew_gap_rad",
                "leg_tracking_error_rad",
            )
        }
        clamp_indices = np.flatnonzero(
            np.abs(arrays["leg_margin_clamp_delta_rad"]) > 1.0e-12
        )
        slew_indices = np.flatnonzero(np.abs(arrays["leg_slew_gap_rad"]) > 1.0e-12)
        return {
            "sample_index": int(sample["sample_index"]),
            "time_s": float(sample["time_s"]),
            "control_tick": int(sample["control_tick"]),
            "substep_in_control_tick": int(sample["substep_in_control_tick"]),
            "phase_index": float(sample["phase_index"]),
            "phase_fraction": float(sample["phase_fraction"]),
            "foot": ("left", "right")[foot],
            "normal_force_fraction_body_weight": float(
                sample["normal_force_fraction_body_weight"][foot]
            ),
            "force_contact_slip_mps": float(sample["force_contact_slip_mps"][foot]),
            "leg_joint_order": list(self.leg_joint_names),
            **{name: value.tolist() for name, value in arrays.items()},
            "margin_clamped_joints": [
                self.leg_joint_names[index] for index in clamp_indices
            ],
            "slew_limited_joints": [
                self.leg_joint_names[index] for index in slew_indices
            ],
        }

    def _phase_bins(self, force_samples: Sequence[Mapping[str, Any]], foot: int) -> list[dict[str, Any]]:
        result = []
        for phase_bin in range(int(self.phase_steps)):
            values = [
                sample
                for sample in force_samples
                if int(np.floor(float(sample["phase_index"]))) % int(self.phase_steps)
                == phase_bin
            ]
            slips = [float(sample["force_contact_slip_mps"][foot]) for sample in values]
            if not values:
                result.append({"phase_bin": phase_bin, **_summary([])})
                continue
            clamp_counts = {
                name: sum(
                    abs(float(sample["leg_margin_clamp_delta_rad"][index])) > 1.0e-12
                    for sample in values
                )
                for index, name in enumerate(self.leg_joint_names)
            }
            slew_counts = {
                name: sum(
                    abs(float(sample["leg_slew_gap_rad"][index])) > 1.0e-12
                    for sample in values
                )
                for index, name in enumerate(self.leg_joint_names)
            }
            result.append(
                {
                    "phase_bin": phase_bin,
                    **_summary(slips),
                    "mean_normal_force_fraction_body_weight": float(
                        np.mean(
                            [
                                sample["normal_force_fraction_body_weight"][foot]
                                for sample in values
                            ]
                        )
                    ),
                    "margin_clamp_sample_counts_by_joint": clamp_counts,
                    "slew_limited_sample_counts_by_joint": slew_counts,
                }
            )
        return result

    def _correlations(self, force_samples: Sequence[Mapping[str, Any]], foot: int) -> dict[str, Any]:
        slips = [float(sample["force_contact_slip_mps"][foot]) for sample in force_samples]
        drivers = {
            "absolute_qvel": "leg_qvel_radps",
            "absolute_tracking_error": "leg_tracking_error_rad",
            "absolute_margin_clamp_delta": "leg_margin_clamp_delta_rad",
            "absolute_slew_gap": "leg_slew_gap_rad",
        }
        result: dict[str, Any] = {}
        for label, field in drivers.items():
            values = {
                name: _correlation(
                    slips,
                    [abs(float(sample[field][index])) for sample in force_samples],
                )
                for index, name in enumerate(self.leg_joint_names)
            }
            result[label] = values
            result[f"{label}_ranked"] = [
                {"joint": name, "correlation": value}
                for name, value in sorted(
                    ((name, value) for name, value in values.items() if value is not None),
                    key=lambda item: abs(item[1]),
                    reverse=True,
                )
            ]
        return result

    def finalize(self) -> dict[str, Any]:
        if len(self.samples) != int(round(6.0 / self.runtime.SIM_DT)):
            raise RuntimeError("causal probe must cover all 6-second physics substeps")
        result: dict[str, Any] = {
            "sample_count": len(self.samples),
            "leg_joint_order": list(self.leg_joint_names),
            "profile_phase_steps": int(self.phase_steps),
            "force_contact_threshold_fraction_body_weight": (
                FORCE_CONTACT_THRESHOLD_BODY_WEIGHT
            ),
            "peak_count_per_foot": TOP_PEAK_COUNT,
            "feet": {},
        }
        for foot, name in enumerate(("left", "right")):
            force_samples = [
                sample
                for sample in self.samples
                if float(sample["normal_force_fraction_body_weight"][foot])
                >= FORCE_CONTACT_THRESHOLD_BODY_WEIGHT
            ]
            peaks = sorted(
                force_samples,
                key=lambda sample: float(sample["force_contact_slip_mps"][foot]),
                reverse=True,
            )[:TOP_PEAK_COUNT]
            result["feet"][name] = {
                "force_contact_sample_count": len(force_samples),
                "slip_summary_mps": _summary(
                    [sample["force_contact_slip_mps"][foot] for sample in force_samples]
                ),
                "top_synchronized_peaks": [self._event(sample, foot) for sample in peaks],
                "phase_bins": self._phase_bins(force_samples, foot),
                "synchronized_correlations": self._correlations(force_samples, foot),
            }
        return result


def _dependency_paths() -> dict[str, Path]:
    return {
        "central_evaluator": Path(central.__file__).resolve(),
        "gait_quality": (EXP_ROOT / "safe_gait_experts" / "gait_quality.py").resolve(),
        "routed_evaluation": (
            EXP_ROOT / "safe_gait_experts" / "routed_evaluation.py"
        ).resolve(),
    }


def _verify_frozen_dependencies() -> dict[str, str]:
    paths = _dependency_paths()
    hashes = {name: sha256(path) for name, path in paths.items()}
    expected = {
        "central_evaluator": EXPECTED_CENTRAL_SHA256,
        "gait_quality": EXPECTED_GAIT_QUALITY_SHA256,
        "routed_evaluation": EXPECTED_ROUTED_EVALUATION_SHA256,
    }
    if hashes != expected:
        raise RuntimeError(f"frozen provenance changed: actual={hashes}, expected={expected}")
    return hashes


def _setup() -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
    asset_paths = central.generated_asset_paths(central.DEFAULT_GENERATED_ROOT.resolve())
    mujoco, onnxruntime, runtime, runtime_provenance = central._load_runtime(
        include_provenance=True
    )
    bank = central.RoutedPolicyBank(
        {role: pdca.BASE_POLICY.resolve() for role in central.REQUIRED_POLICY_ROLES},
        onnxruntime,
    )
    evaluator = runtime.OfficialPolicyEvaluator(
        asset_paths["scene"], pdca.BASE_POLICY.resolve(), asset_paths["reference"]
    )
    evaluator.load_backward_profile(pdca.CURRENT_PROFILE)
    evaluator.load_backward_turn_profile(1, pdca.LEFT_PROFILE)
    evaluator.load_backward_turn_profile(-1, pdca.RIGHT_PROFILE)
    evaluator.backward_turn_minimum_yaw = 0.0
    evaluator.backward_turn_minimum_blend = 0.0
    evaluator.backward_turn_maximum_blend = 1.0
    return evaluator, bank, mujoco, runtime, runtime_provenance


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite causal artifact: {output}")
    frozen_hashes = _verify_frozen_dependencies()
    evaluator, bank, mujoco, runtime, runtime_provenance = _setup()
    base = json.loads(pdca.CURRENT_PROFILE.read_text(encoding="utf-8"))
    candidates = (
        pdca.Candidate("h3_baseline"),
        pdca.Candidate.from_mapping(
            {
                "name": "stage6_nearest_sag114_abs060",
                "amplitude_factors": [1, 1, 1.14, 1.14, 1.14, 1, 1, 1.14, 1.14, 1.14],
                "backward_residual_scale": 0.06,
                "policy_observation_command": [0.05, 0.0, 0.0],
            }
        ),
    )
    trace = pdca.SubstepTrace(evaluator, runtime.SIM_DT)
    original_advance = central.advance_routed_phase
    central.advance_routed_phase = pdca.advance_routed_phase_candidate
    trace.install()
    records = []
    try:
        for candidate in candidates:
            pdca._apply_candidate(evaluator, base, candidate)
            runs = []
            for seed_role, seed in SEED_ROLES.items():
                simulator = pdca._make_simulator(
                    evaluator, bank, mujoco, runtime, candidate
                )
                probe = SynchronizedProbe(simulator, evaluator, runtime)
                probe.install()
                try:
                    run = pdca._run_record(
                        simulator,
                        trace,
                        candidate,
                        runtime,
                        seed=seed,
                        seconds=6.0,
                        warmup_seconds=1.5,
                        joint_noise_scale=1.0,
                        initial_base_speed=0.10,
                    )
                    runs.append(
                        {
                            "seed_role": seed_role,
                            "seed": seed,
                            "strict_result": {
                                "passed": run["strict_passed"],
                                "checks": run["strict_checks"],
                                "metrics": run["strict_metrics"],
                                "separate_hard_gates": run["separate_hard_gates"],
                            },
                            "central_gait_quality_metrics": run["central_run"][
                                "segments"
                            ][0]["gait_quality_metrics"],
                            "central_gait_quality_acceptance": run["central_run"][
                                "segments"
                            ][0]["gait_quality_acceptance"],
                            "synchronized_causal_probe": probe.finalize(),
                        }
                    )
                finally:
                    probe.restore()
            records.append(
                {
                    "name": candidate.name,
                    "candidate_id": candidate.candidate_id,
                    "parameters": asdict(candidate),
                    "runs": runs,
                }
            )
    finally:
        trace.restore()
        central.advance_routed_phase = original_advance
    payload = {
        "schema_version": 1,
        "artifact_kind": "openduckmini_h4_reverse_slip_causal_decomposition",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DIAGNOSTIC_ONLY_NO_PROMOTION",
        "hardware_deployment": "PROHIBITED",
        "scope": "four requested causal replays; not a candidate grid",
        "seed_selection": {
            "worst": {
                "seed": SEED_ROLES["worst"],
                "rule": "largest normalized stage6 strict-boundary excess",
            },
            "median": {
                "seed": SEED_ROLES["median"],
                "rule": "median normalized stage6 strict-boundary excess",
            },
        },
        "configuration": {
            "physical_command": list(pdca.PHYSICAL_COMMAND),
            "seconds": 6.0,
            "warmup_seconds": 1.5,
            "initial_joint_noise_scale": 1.0,
            "initial_base_speed": 0.10,
            "candidate_count": len(candidates),
            "seed_count_per_candidate": len(SEED_ROLES),
        },
        "dependencies": {
            **frozen_hashes,
            "isolated_script_sha256_before_output": sha256(Path(__file__).resolve()),
            "profile_sha256": sha256(pdca.CURRENT_PROFILE),
            "base_policy_sha256": sha256(pdca.BASE_POLICY),
            "runtime": runtime_provenance,
            "onnx_providers": bank.session_providers,
        },
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "record_count": len(records)}))


if __name__ == "__main__":
    main()
