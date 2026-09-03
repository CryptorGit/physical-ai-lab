"""Run the pre-registered simulation-only H5 V3 measurement-alignment gate.

The script deliberately evaluates the rejected frozen 250k V3 actor twice:
once normally and once with append-only measurement capture.  It proves that
capture does not change the action/target/physics trace, then rederives the
strict evaluator's contact, slip, and normal-force measurements from the raw
2 ms capture.  It is not a training, promotion, package, or hardware command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from safe_gait_experts.h4_post_training import infer_h4_action_numpy, mask_h4_head_action
from safe_gait_experts.h5_substep_contact_alignment import (
    FROZEN_DEBOUNCE_WINDOWS_S,
    h5_multiwindow_debounce_summaries,
    h5_rederive_strict_20ms_slip_segment,
)


FROZEN_PARAMS_SHA256 = "d9ff9552f7ba62cc86ecf0bd92b33dfec153aadd6a4c0101af2e946dfc553f41"
FROZEN_PLANAR_MANIFEST_SHA256 = "e7dd906ee97f9b86b81d457482812a9a8eb8dd9da25dddcd4a17bed27a3d778e"
FROZEN_REVERSE_MANIFEST_SHA256 = "d13cf2575f5708a9bd2d68775b01c8bc142e4956bb9620b8eaf5398b8c59b0db"
FROZEN_TRAINING_MANIFEST_SHA256 = "673eec20bb7f782e3aa47b8ee79d0f82c61db65e41cbc495243df8ccabfd8252"
FROZEN_COMMAND_CONTRACT = "OPEN_DUCK_MINI_H5_UNIFIED_COMMAND_ROUTING_V3_DIRECT_NORMALIZED"
PHASE_COS_INDEX = 99
PHASE_SIN_INDEX = 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=_path, required=True)
    parser.add_argument(
        "--resume-analysis",
        action="store_true",
        help=(
            "Analyze an already completed baseline/capture pair after an "
            "analysis-only script repair.  Never reruns simulation or overwrites inputs."
        ),
    )
    return parser.parse_args(argv)


def _all_segments(
    payload: Mapping[str, Any],
) -> Iterable[tuple[int, int, Mapping[str, Any]]]:
    suites = payload["suites"]
    for key in ("primitive_cases", "compound_cases", "transition_cases"):
        for case in suites[key]:
            for index, segment in enumerate(case["segments"]):
                # Independent primitive/compound suites execute each segment
                # as its own one-segment schedule.  Their capture key is that
                # segment's simulation seed plus index zero; transition is the
                # only multi-segment schedule and therefore uses case seed plus
                # its schedule index.
                simulation_seed = segment.get("simulation_seed")
                if simulation_seed is None:
                    yield int(case["seed"]), index, segment
                else:
                    yield int(simulation_seed), 0, segment


def _all_cases(payload: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    suites = payload["suites"]
    for key in ("primitive_cases", "compound_cases", "transition_cases"):
        yield from suites[key]


def _float_equal(actual: object, expected: object, *, label: str) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise RuntimeError(f"{label} null mismatch: {actual!r} != {expected!r}")
        return
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if not np.isclose(float(actual), float(expected), rtol=0.0, atol=1.0e-12):
            raise RuntimeError(f"{label} mismatch: {actual!r} != {expected!r}")
        return
    if actual != expected:
        raise RuntimeError(f"{label} mismatch: {actual!r} != {expected!r}")


def _assert_mapping_equal(actual: Any, expected: Any, *, label: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise RuntimeError(f"{label} keys differ")
        for key in sorted(expected):
            _assert_mapping_equal(actual[key], expected[key], label=f"{label}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise RuntimeError(f"{label} list length differs")
        for index, (actual_value, expected_value) in enumerate(zip(actual, expected)):
            _assert_mapping_equal(
                actual_value, expected_value, label=f"{label}[{index}]"
            )
    else:
        _float_equal(actual, expected, label=label)


def _evaluator_command(
    *,
    output_json: Path,
    capture_npz: Path | None,
    observation_npz: Path | None,
) -> list[str]:
    run = (
        EXP_ROOT
        / "artifacts/h5_training_runs_diagnostic_20260811/v3_mapper_only_clean_v22"
        / "unified/h5_unified_250k_v3_direct_cleanv22_notarget_v1"
    )
    wrapper_root = (
        EXP_ROOT
        / "artifacts/h5_diagnostic_wrappers"
        / "h5_unified_250k_v3_direct_cleanv22_notarget_v1"
    )
    params = run / "final_params.pkl"
    planar_manifest = wrapper_root / "planar/manifest.json"
    reverse_manifest = wrapper_root / "reverse/manifest.json"
    policy = EXP_ROOT / "artifacts/single_policy_deployment_v1/models/base_v22.onnx"
    required = (params, planar_manifest, reverse_manifest, policy)
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("a frozen V3 preflight input is missing")
    hashes = (_sha256(params), _sha256(planar_manifest), _sha256(reverse_manifest))
    if hashes != (
        FROZEN_PARAMS_SHA256,
        FROZEN_PLANAR_MANIFEST_SHA256,
        FROZEN_REVERSE_MANIFEST_SHA256,
    ):
        raise RuntimeError("frozen V3 preflight input hash drifted")
    command = [
        sys.executable,
        str(EXP_ROOT / "scripts/evaluate_h5_routed_transitions.py"),
    ]
    for role in (
        "stand",
        "forward",
        "reverse",
        "lateral_left",
        "lateral_right",
        "yaw_left",
        "yaw_right",
        "compound",
    ):
        command.extend(("--policy", f"{role}={policy}"))
    command.extend(
        (
            "--generated-root",
            str(EXP_ROOT / "artifacts/generated_playground"),
            "--output",
            str(output_json),
            "--seed",
            "20260833",
            "--episodes",
            "1",
            "--seconds",
            "6",
            "--transition-seconds",
            "6",
            "--transition-stand-seconds",
            "2",
            "--warmup-seconds",
            "1.5",
            "--initial-joint-noise-scale",
            "1",
            "--initial-base-speed",
            "0.1",
            "--unified-single-weight",
            "--unified-command-mapper",
            "direct_normalized_v3",
            "--h5-planar-params",
            str(params),
            "--h5-planar-params-sha256",
            FROZEN_PARAMS_SHA256,
            "--h5-planar-manifest",
            str(planar_manifest),
            "--h5-planar-manifest-sha256",
            FROZEN_PLANAR_MANIFEST_SHA256,
            "--h5-reverse-params",
            str(params),
            "--h5-reverse-params-sha256",
            FROZEN_PARAMS_SHA256,
            "--h5-reverse-manifest",
            str(reverse_manifest),
            "--h5-reverse-manifest-sha256",
            FROZEN_REVERSE_MANIFEST_SHA256,
        )
    )
    if capture_npz is not None:
        command.extend(("--h5-substep-capture-npz", str(capture_npz)))
    if observation_npz is not None:
        command.extend(("--h5-policy-observation-capture-npz", str(observation_npz)))
    return command


def _assert_trace_parity(baseline: Mapping[str, Any], captured: Mapping[str, Any]) -> dict[str, int]:
    baseline_segments = list(_all_segments(baseline))
    captured_segments = list(_all_segments(captured))
    if len(baseline_segments) != 38 or len(captured_segments) != 38:
        raise RuntimeError("preflight must contain exactly 38 strict segments per arm")
    if baseline["acceptance"] != captured["acceptance"]:
        raise RuntimeError("capture altered strict suite acceptance")
    total_ticks = 0
    for (base_seed, base_index, base_segment), (cap_seed, cap_index, cap_segment) in zip(
        baseline_segments, captured_segments
    ):
        if (base_seed, base_index, base_segment["name"]) != (
            cap_seed,
            cap_index,
            cap_segment["name"],
        ):
            raise RuntimeError("capture changed strict segment ordering")
        base_trace = base_segment["h5_control_trace"]
        cap_trace = cap_segment["h5_control_trace"]
        if base_trace != cap_trace:
            raise RuntimeError(f"capture changed H5 trace for {base_segment['name']}")
        total_ticks += int(base_trace["control_tick_count"])
    for payload in (baseline, captured):
        for case in _all_cases(payload):
            traces = [segment["h5_control_trace"] for segment in case["segments"]]
            protocol = case.get("h5_trace_protocol")
            case_ticks = sum(int(trace["control_tick_count"]) for trace in traces)
            if case_ticks <= 0:
                raise RuntimeError("H5 case has no control trace ticks")
            if protocol is not None:
                if not isinstance(protocol, Mapping) or (
                    case_ticks != int(protocol["total_control_ticks"])
                    or int(protocol["final_guard_call_count"]) != case_ticks
                    or not bool(protocol["exactly_one_guard_call_per_control_tick"])
                ):
                    raise RuntimeError("H5 schedule guard/count trace contract failed")
            if any(not bool(trace["trace_sha256"]) for trace in traces):
                raise RuntimeError("missing H5 control trace hash")
    return {"segments": len(baseline_segments), "control_ticks": total_ticks}


def _capture_groups(
    capture: Mapping[str, np.ndarray],
) -> Mapping[tuple[int, int], Mapping[str, np.ndarray]]:
    keys = ("run_seed", "segment_index", "segment_name", "time_s", "normal_force_fraction", "tangential_speed_mps")
    if set(capture) != set(keys):
        raise RuntimeError("unexpected H5 substep capture fields")
    count = int(capture["time_s"].shape[0])
    if count <= 0 or any(int(capture[key].shape[0]) != count for key in keys):
        raise RuntimeError("H5 substep capture has inconsistent sample counts")
    groups: dict[tuple[int, int], Mapping[str, np.ndarray]] = {}
    seeds = np.asarray(capture["run_seed"], dtype=np.int64)
    indices = np.asarray(capture["segment_index"], dtype=np.int32)
    for key in sorted({(int(seed), int(index)) for seed, index in zip(seeds, indices)}):
        mask = (seeds == key[0]) & (indices == key[1])
        groups[key] = {name: np.asarray(values)[mask] for name, values in capture.items()}
    return groups


def _assert_measurement_alignment(
    captured_payload: Mapping[str, Any], capture: Mapping[str, np.ndarray]
) -> dict[str, int]:
    expected_segments = {
        (seed, index): segment
        for seed, index, segment in _all_segments(captured_payload)
    }
    groups = _capture_groups(capture)
    if set(groups) != set(expected_segments):
        raise RuntimeError("captured 2 ms segment keys do not match strict evidence")
    continuity_by_seed: dict[int, Any] = {}
    for key in sorted(expected_segments):
        segment = expected_segments[key]
        values = groups[key]
        times = np.asarray(values["time_s"], dtype=np.float64)
        force = np.asarray(values["normal_force_fraction"], dtype=np.float64)
        speed = np.asarray(values["tangential_speed_mps"], dtype=np.float64)
        if (
            times.ndim != 1
            or force.shape != (len(times), 2)
            or speed.shape != force.shape
            or len(times) < 2
            or not np.isclose(times[0], 0.0, atol=0.0, rtol=0.0)
            or not np.allclose(np.diff(times), 0.002, atol=1.0e-12, rtol=0.0)
        ):
            raise RuntimeError(f"invalid 2 ms capture timing for {key}")
        metrics = segment["gait_quality_metrics"]
        if not bool(metrics["measurement_complete"]):
            raise RuntimeError("preflight cannot accept incomplete strict measurement")
        actual_debounce = h5_multiwindow_debounce_summaries(times, force)
        _assert_mapping_equal(
            actual_debounce,
            metrics["contact_debounce_sensitivity"],
            label=f"{key}.contact_debounce_sensitivity",
        )
        slip, next_continuity = h5_rederive_strict_20ms_slip_segment(
            times,
            force,
            speed,
            initial_state=continuity_by_seed.get(key[0]),
        )
        continuity_by_seed[key[0]] = next_continuity
        _float_equal(
            slip["stance_slip_rms_mps"],
            metrics["stance_slip_rms_mps"],
            label=f"{key}.stance_slip_rms_mps",
        )
        _float_equal(
            slip["stance_slip_p95_mps"],
            metrics["stance_slip_p95_mps"],
            label=f"{key}.stance_slip_p95_mps",
        )
        _float_equal(
            slip["maximum_completed_stance_cumulative_slip_m"],
            metrics["maximum_per_stance_cumulative_slip_m"],
            label=f"{key}.maximum_per_stance_cumulative_slip_m",
        )
        _float_equal(
            float(np.percentile(np.sum(force, axis=1), 99)),
            metrics["total_normal_force_p99_fraction_body_weight"],
            label=f"{key}.total_normal_force_p99_fraction_body_weight",
        )
        steady = times >= 1.5
        _float_equal(
            float(np.mean(np.sum(force[steady], axis=1))),
            metrics["steady_mean_total_normal_force_fraction_body_weight"],
            label=f"{key}.steady_mean_total_normal_force_fraction_body_weight",
        )
        if set(actual_debounce) != {
            f"{int(window * 1000)}ms" for window in FROZEN_DEBOUNCE_WINDOWS_S
        }:
            raise RuntimeError("not all frozen debounce windows were rederived")
    return {"segments": len(groups), "substep_samples": int(capture["time_s"].shape[0])}


def _phase_probe(observation_capture: Mapping[str, np.ndarray], params_path: Path) -> Mapping[str, object]:
    required = {"run_seed", "segment_index", "segment_name", "control_step", "observation"}
    if set(observation_capture) != required:
        raise RuntimeError("unexpected H5 observation capture fields")
    observations = np.asarray(observation_capture["observation"], dtype=np.float32)
    if observations.ndim != 2 or observations.shape[1] != 116 or not np.all(np.isfinite(observations)):
        raise RuntimeError("captured H5 actor observations are not finite 116-wide")
    norms = np.sqrt(
        np.square(observations[:, PHASE_COS_INDEX])
        + np.square(observations[:, PHASE_SIN_INDEX])
    )
    if not np.allclose(norms, 1.0, atol=2.0e-6, rtol=0.0):
        raise RuntimeError("captured phase cos/sin is not unit-normalized")
    with params_path.open("rb") as stream:
        params = pickle.load(stream)
    original = mask_h4_head_action(infer_h4_action_numpy(params, observations))
    phase_zero = observations.copy()
    phase_zero[:, PHASE_COS_INDEX] = 1.0
    phase_zero[:, PHASE_SIN_INDEX] = 0.0
    phase_half_cycle = observations.copy()
    phase_half_cycle[:, PHASE_COS_INDEX : PHASE_SIN_INDEX + 1] *= -1.0
    zero_action = mask_h4_head_action(infer_h4_action_numpy(params, phase_zero))
    half_action = mask_h4_head_action(infer_h4_action_numpy(params, phase_half_cycle))
    def delta(candidate: np.ndarray) -> Mapping[str, float]:
        absolute = np.abs(np.asarray(candidate, dtype=np.float64) - original)
        return {
            "maximum_abs_action_delta": float(np.max(absolute)),
            "mean_abs_action_delta": float(np.mean(absolute)),
            "p95_abs_action_delta": float(np.percentile(absolute, 95)),
        }
    return {
        "sample_count": int(observations.shape[0]),
        "observation_width": int(observations.shape[1]),
        "phase_cos_index": PHASE_COS_INDEX,
        "phase_sin_index": PHASE_SIN_INDEX,
        "minimum_phase_norm": float(np.min(norms)),
        "maximum_phase_norm": float(np.max(norms)),
        "phase_zero_action_delta": delta(zero_action),
        "phase_half_cycle_action_delta": delta(half_action),
        "interpretation": (
            "offline actor-input sensitivity only; no simulation action, target, "
            "guard, training, or hardware state was changed"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir
    baseline_json = output_dir / "baseline_no_capture.json"
    captured_json = output_dir / "capture_enabled.json"
    substep_npz = output_dir / "substep_measurements.npz"
    observation_npz = output_dir / "actor_observations.npz"
    result_path = output_dir / "preflight_result.json"
    if args.resume_analysis:
        required = (baseline_json, captured_json, substep_npz, observation_npz)
        if (
            not output_dir.is_dir()
            or any(not path.is_file() for path in required)
            or result_path.exists()
        ):
            raise FileNotFoundError(
                "resume-analysis requires completed immutable inputs and no result file"
            )
    else:
        if output_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite preflight directory: {output_dir}"
            )
        output_dir.mkdir(parents=True)
    try:
        if not args.resume_analysis:
            subprocess.run(
                _evaluator_command(
                    output_json=baseline_json, capture_npz=None, observation_npz=None
                ),
                check=True,
                cwd=EXP_ROOT,
            )
            subprocess.run(
                _evaluator_command(
                    output_json=captured_json,
                    capture_npz=substep_npz,
                    observation_npz=observation_npz,
                ),
                check=True,
                cwd=EXP_ROOT,
            )
        baseline = json.loads(baseline_json.read_text(encoding="utf-8"))
        captured = json.loads(captured_json.read_text(encoding="utf-8"))
        if (
            baseline["provenance"]["single_policy_mode"]["params_sha256"]
            != FROZEN_PARAMS_SHA256
            or captured["provenance"]["single_policy_mode"]["params_sha256"]
            != FROZEN_PARAMS_SHA256
            or baseline["provenance"]["h5_command_contract"] != FROZEN_COMMAND_CONTRACT
            or captured["provenance"]["h5_command_contract"] != FROZEN_COMMAND_CONTRACT
        ):
            raise RuntimeError("preflight evaluator did not bind the frozen direct-V3 candidate")
        trace_parity = _assert_trace_parity(baseline, captured)
        with np.load(substep_npz, allow_pickle=False) as archive:
            substep_capture = {name: archive[name] for name in archive.files}
        with np.load(observation_npz, allow_pickle=False) as archive:
            observation_capture = {name: archive[name] for name in archive.files}
        alignment = _assert_measurement_alignment(captured, substep_capture)
        params_path = (
            EXP_ROOT
            / "artifacts/h5_training_runs_diagnostic_20260811/v3_mapper_only_clean_v22"
            / "unified/h5_unified_250k_v3_direct_cleanv22_notarget_v1/final_params.pkl"
        )
        phase = _phase_probe(observation_capture, params_path)
        result = {
            "schema_version": 1,
            "status": "PREFLIGHT_PASS_NOT_A_TRAINING_CANDIDATE",
            "hardware_deployment": "PROHIBITED",
            "frozen_candidate": {
                "params_sha256": FROZEN_PARAMS_SHA256,
                "training_manifest_sha256": FROZEN_TRAINING_MANIFEST_SHA256,
                "command_contract": FROZEN_COMMAND_CONTRACT,
                "command_mapper": "direct_normalized_v3",
            },
            "inputs": {
                "baseline_json": {"path": str(baseline_json), "sha256": _sha256(baseline_json)},
                "capture_json": {"path": str(captured_json), "sha256": _sha256(captured_json)},
                "substep_npz": {"path": str(substep_npz), "sha256": _sha256(substep_npz)},
                "observation_npz": {"path": str(observation_npz), "sha256": _sha256(observation_npz)},
            },
            "control_trace_parity": {**trace_parity, "passed": True},
            "strict_measurement_alignment": {**alignment, "passed": True},
            "phase_observation_probe": phase,
            "strict_candidate_result": {
                "baseline_strict_suite_passed": bool(baseline["suites"]["acceptance"]["passed"]),
                "capture_strict_suite_passed": bool(captured["suites"]["acceptance"]["passed"]),
                "candidate_remains_rejected": True,
            },
            "next_authority": (
                "none: this preflight only proves measurement and capture parity; "
                "a separately reviewed training contract is required before any PPO run"
            ),
        }
        result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(result_path), "status": result["status"]}, indent=2))
        return 0
    except BaseException:
        # Preserve produced diagnostics for root-cause analysis, but do not
        # create a misleading PASS result after a partial preflight.
        raise


if __name__ == "__main__":
    raise SystemExit(main())
