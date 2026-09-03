"""Exact-home failure3 screen for pure-validated H4 reverse teachers.

The script injects an explicit target table into a fresh evaluator instance;
it never edits the central evaluator or an adopted profile.  It is pinned to
the synchronized force-contact H4 runtime hashes and runs only four bounded
teacher designs across the three causal failure seeds.  No 5x15 or 20x30
expansion occurs here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = EXP_ROOT.parents[2]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from scripts import build_h4_slew_feasible_reverse_teacher_bank as teacher  # noqa: E402
from scripts import diagnose_h4_reverse_pdca as pdca  # noqa: E402
from scripts import evaluate_routed_transitions as central  # noqa: E402


DEFAULT_BANK = (
    EXP_ROOT / "artifacts" / "h4_reverse_slew_feasible_teacher_bank_v1.json"
)
DEFAULT_OUTPUT = (
    EXP_ROOT
    / "artifacts"
    / "h4_reverse_slew_feasible_teacher_exact_home_failure3_v1.json"
)
FAILURE3_SEEDS = (20_260_810, 20_265_810, 20_271_810)
SELECTED_NAMES = (
    "h4_reverse_c1p50_h1_e1p00",
    "h4_reverse_c1p50_h1_e0p75",
    "h4_reverse_c1p50_h2_e1p00",
    "h4_reverse_c1p75_h1_e1p00",
)
HISTORICAL_FORCE_CONTACT_V1_SNAPSHOT_ID = "h4_force_contact_v1_historical"
CURRENT_STRICT_QUALITY_SNAPSHOT_ID = "h4_strict_quality_gate_v2"
CURRENT_CENTRAL_SNAPSHOT_ID = CURRENT_STRICT_QUALITY_SNAPSHOT_ID
CENTRAL_SNAPSHOT_SHA256 = MappingProxyType(
    {
        HISTORICAL_FORCE_CONTACT_V1_SNAPSHOT_ID: MappingProxyType(
            {
                "scripts/evaluate_routed_transitions.py": (
                    "c4a0fff4d8726dc46ec462a6a5f32c599decc211a269779806f30559baf978c5"
                ),
                "safe_gait_experts/gait_quality.py": (
                    "20a5010037f2157a089501e012881cabceed794bc77b1cca8ca5eaf6f7e88b61"
                ),
                "safe_gait_experts/routed_evaluation.py": (
                    "fff136407b64090be41498f97eba5274e1cae4d373d3519fdc805226851ae047"
                ),
            }
        ),
        CURRENT_STRICT_QUALITY_SNAPSHOT_ID: MappingProxyType(
            {
                "scripts/evaluate_routed_transitions.py": (
                    "31fb8846fc6267f28d032bca164dee2c872bfb484ebe272850100834bf1b1a9b"
                ),
                "safe_gait_experts/gait_quality.py": (
                    "b28e1ceb4cb6406411150bbad772a78203b8163bf10adcd79a5f31f83da5f2de"
                ),
                "safe_gait_experts/routed_evaluation.py": (
                    "f25ed858fbb5753fdcfd9e76f08396d0f09f95ac4696eb2e50cb5c128b80db09"
                ),
            }
        ),
    }
)
EXPECTED_CENTRAL_SHA256 = CENTRAL_SNAPSHOT_SHA256[CURRENT_CENTRAL_SNAPSHOT_ID]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_central_snapshot(
    snapshot_id: str = CURRENT_CENTRAL_SNAPSHOT_ID,
) -> dict[str, str]:
    if snapshot_id not in CENTRAL_SNAPSHOT_SHA256:
        raise ValueError(f"unsupported central H4 snapshot id: {snapshot_id}")
    expected = CENTRAL_SNAPSHOT_SHA256[snapshot_id]
    actual = {
        relative: _sha256(EXP_ROOT / relative)
        for relative in expected
    }
    mismatches = {
        relative: {"expected": expected[relative], "actual": value}
        for relative, value in actual.items()
        if value != expected[relative]
    }
    if mismatches:
        raise ValueError(
            f"central H4 snapshot changed for {snapshot_id}: {mismatches}"
        )
    return actual


def selected_candidates(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("teacher bank candidates must be a list")
    by_name = {candidate.get("name"): candidate for candidate in candidates}
    missing = set(SELECTED_NAMES) - set(by_name)
    if missing:
        raise ValueError(f"teacher bank is missing selected candidates: {sorted(missing)}")
    result = tuple(by_name[name] for name in SELECTED_NAMES)
    for candidate in result:
        validation = teacher.validate_candidate(candidate)
        if not validation["passed"]:
            raise ValueError(
                f"selected candidate {candidate['name']} failed pure validation: "
                f"{validation['failures']}"
            )
    return result


def inject_teacher_table(evaluator: Any, candidate: Mapping[str, Any]) -> None:
    """Install one explicit target table in an isolated evaluator object."""

    validation = teacher.validate_candidate(candidate)
    if not validation["passed"]:
        raise ValueError(f"cannot inject invalid teacher: {validation['failures']}")
    targets = np.asarray(candidate["target_table_rad"], dtype=np.float64)
    leg_targets = targets[:, teacher.LEG_ACTUATOR_INDICES]
    phase_steps = int(candidate["construction"]["phase_steps"])
    if leg_targets.shape != (phase_steps, 10):
        raise ValueError("explicit reverse teacher must contain phase_steps x 10 legs")

    # OfficialPolicyEvaluator interpolates mean + scale * deviation.  A zero
    # mean, unit scale, and the explicit table as deviation reproduce it
    # exactly without changing the evaluator implementation.
    evaluator.phase_steps = phase_steps
    reference_width = int(np.asarray(evaluator.backward_reference_frames).shape[1])
    evaluator.backward_reference_frames = np.zeros(
        (phase_steps, reference_width), dtype=np.float64
    )
    evaluator.backward_leg_means = np.zeros(10, dtype=np.float64)
    evaluator.backward_leg_deviations = leg_targets.copy()
    evaluator.backward_gait_scales = np.ones(10, dtype=np.float64)
    evaluator.backward_gait_biases = np.zeros(10, dtype=np.float64)
    evaluator.backward_phase_rate = float(
        candidate["construction"]["phase_advance_bins_per_control"]
    )
    evaluator.backward_residual_scale = 0.0


def _pdca_candidate(candidate: Mapping[str, Any]) -> pdca.Candidate:
    return pdca.Candidate(
        name=str(candidate["name"]),
        phase_entry_preincrement=float(
            candidate["construction"]["phase_entry_preincrement_bins"]
        ),
        upper_cap_extras_rad=(0.0,) * 10,
        backward_residual_scale=0.0,
        policy_observation_command=teacher.PHYSICAL_COMMAND,
    )


def _full_h4_gate(run: Mapping[str, Any]) -> dict[str, Any]:
    segment = run["central_run"]["segments"][0]
    quality = segment["gait_quality_acceptance"]
    checks = {
        "kinematic_safety_contact_audit": bool(run["strict_passed"]),
        "central_force_contact_gait_quality": bool(quality["passed"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "gait_quality_failures": list(quality["failures"]),
    }


def _ranking_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    summary = record["summary"]
    quality = summary["central_gait_quality"]
    slip = quality["stance_slip"]
    minimum_ratio = float(summary["minimum_speed_ratio"])
    maximum_ratio = float(summary["maximum_speed_ratio"])
    speed_violation = max(0.0, 0.75 - minimum_ratio, maximum_ratio - 1.25)
    return (
        -int(record["full_h4_pass_count"]),
        -int(quality["pass_count"]),
        int(summary["fall_count"]),
        float(slip["rms_mps"]["maximum"]),
        float(slip["p95_mps"]["maximum"]),
        float(slip["maximum_per_stance_cumulative_slip_m"]["maximum"]),
        speed_violation,
        float(summary["maximum_cross_velocity_mps"]),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.5)
    parser.add_argument(
        "--seeds",
        type=pdca._csv_ints,
        default=FAILURE3_SEEDS,
    )
    args = parser.parse_args(argv)
    if args.seconds != 6.0:
        parser.error("this bounded screen is frozen at exactly 6.0 seconds")
    if args.warmup_seconds != 1.5:
        parser.error("this bounded screen is frozen at exactly 1.5 seconds warmup")
    if tuple(args.seeds) != FAILURE3_SEEDS:
        parser.error(f"this bounded screen requires failure3 seeds {FAILURE3_SEEDS}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite reverse screen: {output}")
    central_hashes = verify_central_snapshot()
    bank_path = args.bank.resolve()
    bank_payload = json.loads(bank_path.read_text(encoding="utf-8"))
    bank_validation = teacher.validate_bank(bank_payload)
    if not bank_validation["passed"]:
        raise ValueError(f"teacher bank failed validation: {bank_validation}")
    candidates = selected_candidates(bank_payload)

    asset_paths = central.generated_asset_paths(central.DEFAULT_GENERATED_ROOT.resolve())
    mujoco, onnxruntime, runtime, runtime_provenance = central._load_runtime(
        include_provenance=True
    )
    policy_bank = central.RoutedPolicyBank(
        {role: pdca.BASE_POLICY.resolve() for role in central.REQUIRED_POLICY_ROLES},
        onnxruntime,
    )
    original_advance = central.advance_routed_phase
    central.advance_routed_phase = pdca.advance_routed_phase_candidate
    records = []
    try:
        for teacher_candidate in candidates:
            evaluator = runtime.OfficialPolicyEvaluator(
                asset_paths["scene"],
                pdca.BASE_POLICY.resolve(),
                asset_paths["reference"],
            )
            inject_teacher_table(evaluator, teacher_candidate)
            diagnostic_candidate = _pdca_candidate(teacher_candidate)
            simulator = pdca._make_simulator(
                evaluator, policy_bank, mujoco, runtime, diagnostic_candidate
            )
            trace = pdca.SubstepTrace(evaluator, runtime.SIM_DT)
            trace.install()
            try:
                runs = [
                    pdca._run_record(
                        simulator,
                        trace,
                        diagnostic_candidate,
                        runtime,
                        seed=seed,
                        seconds=args.seconds,
                        warmup_seconds=args.warmup_seconds,
                        joint_noise_scale=0.0,
                        initial_base_speed=0.0,
                    )
                    for seed in FAILURE3_SEEDS
                ]
            finally:
                trace.restore()
            gates = [_full_h4_gate(run) for run in runs]
            for run, gate in zip(runs, gates):
                run["full_h4_gate"] = gate
            summary = pdca._candidate_summary(diagnostic_candidate, runs)
            records.append(
                {
                    "candidate_id": teacher_candidate["candidate_id"],
                    "name": teacher_candidate["name"],
                    "construction": teacher_candidate["construction"],
                    "pure_validation": teacher.validate_candidate(
                        teacher_candidate
                    ),
                    "full_h4_pass_count": sum(gate["passed"] for gate in gates),
                    "full_h4_failure3_passed": all(gate["passed"] for gate in gates),
                    "summary": summary,
                    "runs": runs,
                }
            )
    finally:
        central.advance_routed_phase = original_advance

    ranking = sorted(records, key=_ranking_key)
    passed = [record for record in ranking if record["full_h4_failure3_passed"]]
    payload = {
        "schema_version": 1,
        "artifact_kind": "openduckmini_h4_reverse_teacher_exact_home_failure3_screen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "EXACT_HOME_FAILURE3_CANDIDATE_FOUND_REQUIRES_PERTURBED_REQUALIFICATION"
            if passed
            else "NO_EXACT_HOME_FAILURE3_PASS_TEACHERS_REMAIN_TRAINING_PRIORS_ONLY"
        ),
        "hardware_deployment": "PROHIBITED",
        "simulation_adoption_allowed": False,
        "release_allowed": False,
        "configuration": {
            "physical_command_mps_radps": list(teacher.PHYSICAL_COMMAND),
            "policy_observation_command_mps_radps": list(teacher.PHYSICAL_COMMAND),
            "seeds": list(FAILURE3_SEEDS),
            "seconds": args.seconds,
            "warmup_seconds": args.warmup_seconds,
            "initial_joint_noise_scale": 0.0,
            "initial_base_speed_mps": 0.0,
            "initial_condition": "EXACT_SAFE_HOME_DIAGNOSTIC_ONLY",
            "candidate_count": len(candidates),
            "episode_count": len(candidates) * len(FAILURE3_SEEDS),
            "expansion": "STOP_AFTER_FAILURE3",
        },
        "provenance": {
            "teacher_bank_path": str(bank_path.relative_to(EXP_ROOT)),
            "teacher_bank_sha256": _sha256(bank_path),
            "central_snapshot_sha256": central_hashes,
            "central_snapshot_id": CURRENT_CENTRAL_SNAPSHOT_ID,
            "screen_script_sha256_before_output": _sha256(Path(__file__).resolve()),
            "base_policy_sha256": _sha256(pdca.BASE_POLICY),
            "runtime": runtime_provenance,
            "onnx_providers": policy_bank.session_providers,
        },
        "gate": {
            "central_kinematic_safety_and_contact_audit": True,
            "central_force_contact_gait_quality_all_checks": True,
            "full_pass_requires_both": True,
            "qualification_warning": (
                "exact-home zero-perturbation failure3 is an early rejection "
                "screen, not the minimum-spec perturbed qualification"
            ),
        },
        "ranking_candidate_ids": [record["candidate_id"] for record in ranking],
        "full_h4_failure3_pass_candidate_ids": [
            record["candidate_id"] for record in passed
        ],
        "candidates": records,
        "decision": {
            "best_screened_candidate_id": ranking[0]["candidate_id"],
            "all_strict_candidate_count": len(passed),
            "expand_to_5x15": False,
            "train": False,
            "teacher_use": (
                "REQUIRES_PERTURBED_REQUALIFICATION_BEFORE_TRAINING"
                if passed
                else "TRAINING_PRIOR_ONLY_RANK_BY_FORCE_CONTACT_AND_CADENCE"
            ),
            "hardware": "PROHIBITED",
        },
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
                "status": payload["status"],
                "best_screened_candidate_id": payload["decision"][
                    "best_screened_candidate_id"
                ],
                "all_strict_candidate_count": len(passed),
                "ranking_candidate_ids": payload["ranking_candidate_ids"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
