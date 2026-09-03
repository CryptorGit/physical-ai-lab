"""Validate immutable raw evidence against the revised forward/no-fall scope.

The revised scope requires forward progress in the robot-local forward axis and
no fall.  It expressly does not require a straight world-frame path, so a
curving trajectory cannot by itself fail this gate.  This script does not
replace or reinterpret the raw strict-gait verdict: it records that verdict
verbatim as a known limitation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


EXPECTED_POLICY_SHA256 = (
    "f7a2731330cd3be52858989b021423a5f363cc4a8f9850512281da745a7617c0"
)
EXPECTED_GATE_ID = "V22_FORWARD_ONLY_20X30_HARDWARE_SAFE_ROUTED_RUNTIME_V1"
REQUIRED_SEGMENT_CHECKS = (
    "all_physics_substeps_audited",
    "applied_targets_safe",
    "completed",
    "contact_rates_from_all_physics_substeps",
    "desired_targets_inside_margin",
    "finite",
    "flight_rate",
    "head_applied_action_locked",
    "head_target_locked",
    "joint_qpos_safe",
    "minimum_height",
    "minimum_upright",
    "moving_single_support",
    "no_fall",
    "primary_velocity",
    "signed_linear_progress",
    "steady_route_expected_expert",
    "steady_route_expected_policy_role",
    "substep_finite",
    "substep_joint_qpos_safe",
    "substep_no_fall",
    "target_slew_safe",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.evidence.is_file():
        parser.error(f"missing raw evidence: {args.evidence}")
    if args.output.exists():
        parser.error(f"refusing to overwrite immutable verdict: {args.output}")
    return args


def _require(condition: bool, label: str, failures: list[str]) -> None:
    if not condition:
        failures.append(label)


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def validate(payload: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    _require(payload.get("gate_id") == EXPECTED_GATE_ID, "gate_id", failures)
    scope = payload.get("scope", {})
    _require(scope.get("hardware_actuation") == "NOT_ATTEMPTED", "hardware_free", failures)
    config = payload.get("configuration", {})
    _require(config.get("episodes") == 20, "episodes_20", failures)
    _require(config.get("seconds") == 30.0, "seconds_30", failures)
    _require(config.get("physical_command") == [0.05, 0.0, 0.0], "physical_command", failures)
    _require(
        config.get("policy_observation_command") == [0.1, 0.0, 0.0],
        "policy_observation_command",
        failures,
    )
    _require(config.get("initial_joint_noise_scale") == 1.0, "joint_noise", failures)
    _require(config.get("initial_base_speed") == 0.10, "initial_base_speed", failures)
    _require(config.get("warmup_seconds") == 1.5, "warmup", failures)
    _require(config.get("leg_target_margin_rad") == 0.05, "target_margin", failures)
    _require(config.get("target_slew_rate_rad_s") == 2.0, "target_slew", failures)

    policies = payload.get("policies", {})
    _require(bool(policies), "policies_present", failures)
    for role, record in sorted(policies.items()):
        _require(
            record.get("sha256") == EXPECTED_POLICY_SHA256,
            f"policy_sha256_{role}",
            failures,
        )
        _require(
            record.get("cpu_only_provider_verified") is True,
            f"cpu_only_{role}",
            failures,
        )
    runtime = payload.get("runtime_dependency_provenance", {})
    _require(
        runtime.get("pre_post_source_and_data_hashes_unchanged") is True,
        "runtime_immutable",
        failures,
    )
    _require(
        runtime.get("all_onnx_sessions_cpu_only_verified") is True,
        "runtime_cpu_only",
        failures,
    )

    episodes = payload.get("episodes", [])
    _require(len(episodes) == 20, "raw_episode_count", failures)
    acceptance_episodes = payload.get("acceptance", {}).get("episode_checks", [])
    _require(
        len(acceptance_episodes) == len(episodes),
        "acceptance_episode_count",
        failures,
    )
    episode_rows: list[dict[str, Any]] = []
    strict_gait_failures = 0
    for episode_index, episode in enumerate(episodes):
        segments = episode.get("segments", [])
        _require(len(segments) == 1, f"episode_{episode_index}_one_segment", failures)
        if len(segments) != 1:
            continue
        segment = segments[0]
        metrics = segment.get("metrics", {})
        safety = segment.get("safety_audit", {})
        accepted_segments = (
            acceptance_episodes[episode_index].get("segments", [])
            if episode_index < len(acceptance_episodes)
            else []
        )
        _require(
            len(accepted_segments) == 1,
            f"episode_{episode_index}_acceptance_segment",
            failures,
        )
        accepted_segment = accepted_segments[0] if len(accepted_segments) == 1 else {}
        checks = accepted_segment.get("checks", {})
        for check in REQUIRED_SEGMENT_CHECKS:
            _require(checks.get(check) is True, f"episode_{episode_index}_{check}", failures)
        _require(segment.get("name") == "forward", f"episode_{episode_index}_forward", failures)
        _require(segment.get("completed") is True, f"episode_{episode_index}_completed", failures)
        _require(segment.get("fell") is False, f"episode_{episode_index}_fell", failures)
        velocity = _number(metrics.get("projected_primary_velocity"))
        _require(
            velocity is not None and velocity > 0.0,
            f"episode_{episode_index}_local_forward_progress",
            failures,
        )
        for field in (
            "applied_target_limit_violations",
            "qpos_limit_violations",
            "target_slew_violations",
            "nonfinite_sample_count",
        ):
            _require(
                safety.get(field) == 0,
                f"episode_{episode_index}_{field}",
                failures,
            )
        _require(
            safety.get("head_target_peak_rad") == 0.0,
            f"episode_{episode_index}_head_target_lock",
            failures,
        )
        strict_gait_passed = checks.get("strict_gait_quality") is True
        strict_gait_failures += int(not strict_gait_passed)
        displacement = metrics.get("displacement_xyz", [None, None, None])
        episode_rows.append(
            {
                "seed": segment.get("simulation_seed"),
                "completed": segment.get("completed"),
                "fell": segment.get("fell"),
                "local_forward_velocity_mps": velocity,
                "world_displacement_xyz_m": displacement,
                "minimum_height_m": metrics.get("minimum_height_m"),
                "minimum_upright": metrics.get("minimum_upright"),
                "strict_gait_quality_passed": strict_gait_passed,
            }
        )
    return {
        "revised_scope": {
            "requirement": "forward_local_progress_without_fall",
            "straight_world_frame_path_required": False,
            "full_strict_gait_quality_required": False,
            "hardware_claim": "NONE",
        },
        "required_segment_checks": list(REQUIRED_SEGMENT_CHECKS),
        "episodes": episode_rows,
        "strict_gait_quality": {
            "passed_episodes": len(episode_rows) - strict_gait_failures,
            "failed_episodes": strict_gait_failures,
            "not_used_for_revised_forward_no_fall_gate": True,
        },
        "failures": failures,
        "passed": not failures,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    evidence = args.evidence.resolve()
    raw = json.loads(evidence.read_text(encoding="utf-8"))
    verdict = validate(raw)
    result = {
        "schema_version": 1,
        "gate_id": "V22_FORWARD_NO_FALL_REVISED_SCOPE_VERDICT_V1",
        "raw_evidence": {"path": str(evidence), "sha256": _sha256(evidence)},
        **verdict,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
