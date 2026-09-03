"""Freeze GO2_ENDPOINT_EVALUATION_V1 and execute its pure-math unit tests."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage6_corrected_endpoint_formal"
CONFIG = EXP / "configs/stage6_go2_endpoint_evaluation_v1.yaml"
sys.path.insert(0, str(EXP / "src"))

from go2_bidirectional.stage6_endpoint_protocol import (  # noqa: E402
    heading_error,
    physical_slip_intervals,
    quat_xyzw_to_gravity_tilt,
    quat_xyzw_to_roll_pitch_yaw,
    quat_xyzw_to_rotation_matrix,
)


def dump(name: str, value) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> list[float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def matrix_checks(matrix: list[list[float]]) -> tuple[float, float]:
    orthogonality = max(
        abs(sum(matrix[k][i] * matrix[k][j] for k in range(3)) - (1.0 if i == j else 0.0))
        for i in range(3) for j in range(3)
    )
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    return orthogonality, determinant


def old_wxyz_decode(xyzw: list[float]) -> tuple[float, float, float]:
    w, x, y, z = xyzw
    return (
        math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
        math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
        math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
    )


def main() -> None:
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CONFIG, OUT / "stage6_go2_endpoint_evaluation_v1.yaml")
    dump("protocol.json", protocol)
    dump("protocol_hash.json", {
        "algorithm": "sha256",
        "canonicalization": "UTF-8 JSON, sorted keys, compact separators, resolved YAML values",
        "sha256": digest,
        "frozen_before_formal_rollout": True,
    })

    cases = {
        "identity": (0.0, 0.0, 0.0),
        "roll_+30": (math.radians(30), 0.0, 0.0),
        "roll_-30": (math.radians(-30), 0.0, 0.0),
        "pitch_+20": (0.0, math.radians(20), 0.0),
        "pitch_-20": (0.0, math.radians(-20), 0.0),
        "yaw_+90": (0.0, 0.0, math.radians(90)),
        "yaw_-90": (0.0, 0.0, math.radians(-90)),
        "combined": (math.radians(18), math.radians(-13), math.radians(47)),
    }
    rows = []
    for label, expected in cases.items():
        quaternion = quaternion_from_rpy(*expected)
        actual = quat_xyzw_to_roll_pitch_yaw(quaternion)
        matrix = quat_xyzw_to_rotation_matrix(quaternion)
        orthogonality, determinant = matrix_checks(matrix)
        old = old_wxyz_decode(quaternion)
        rows.append({
            "case": label,
            "quaternion_xyzw": quaternion,
            "expected_rpy": expected,
            "actual_rpy": actual,
            "max_rpy_error": max(abs(heading_error(a, b)) for a, b in zip(actual, expected)),
            "round_trip_max_error": max(abs(heading_error(a, b)) for a, b in zip(actual, expected)),
            "matrix_orthogonality_max_error": orthogonality,
            "matrix_determinant": determinant,
            "gravity_tilt_rad": quat_xyzw_to_gravity_tilt(quaternion),
            "legacy_wxyz_rpy": old,
            "legacy_max_error": max(abs(heading_error(a, b)) for a, b in zip(old, expected)),
            "finite": all(math.isfinite(value) for value in actual),
        })
    quaternion_pass = all(
        row["max_rpy_error"] <= 1e-12
        and row["matrix_orthogonality_max_error"] <= 1e-12
        and abs(row["matrix_determinant"] - 1.0) <= 1e-12
        and row["finite"]
        for row in rows
    )
    legacy_fails = any(row["legacy_max_error"] > 1e-3 for row in rows)
    dump("quaternion_contract.json", {
        "source": "robot.data.root_quat_w.torch",
        "order": "xyzw",
        "common_helper": "go2_bidirectional.stage6_endpoint_protocol",
        "implicit_wxyz_conversion_prohibited": True,
    })
    dump("quaternion_unit_test_results.json", {
        "pass": quaternion_pass and legacy_fails,
        "xyzw_tests_pass": quaternion_pass,
        "legacy_wxyz_decode_fails_same_suite": legacy_fails,
        "cases": rows,
    })

    def trace(points, forces=None):
        return physical_slip_intervals(
            forces or [10.0] * len(points), points, dt=0.02,
            exclude_onset_steps=2, exclude_release_steps=2,
        )

    fixed = [[0.0, 0.0]] * 20
    touchdown = [[0.05, 0.0], [0.03, 0.0]] + [[0.0, 0.0]] * 18
    drift_001 = [[0.01 * i / 19, 0.0] for i in range(20)]
    drift_004 = [[0.0, 0.0]] * 5 + [[0.04 * min(i, 5) / 5, 0.0] for i in range(1, 16)]
    fast_004 = [[0.0, 0.0]] * 4 + [[0.008 * i, 0.0] for i in range(3)] + [[0.016, 0.0]] * 13
    fast_012 = [[0.0, 0.0]] * 4 + [[0.008 * i, 0.0] for i in range(7)] + [[0.048, 0.0]] * 9
    body_rotation_fixed_contact = [[0.0, 0.0]] * 20
    slip_cases = [
        ("fixed_contact_point", fixed, False),
        ("touchdown_only_transient", touchdown, False),
        ("0.01m_drift", drift_001, False),
        ("0.04m_sustained_drift", drift_004, True),
        ("0.40mps_for_0.04s", fast_004, False),
        ("0.40mps_for_0.12s", fast_012, True),
        ("foot_body_rotation_fixed_contact_point", body_rotation_fixed_contact, False),
    ]
    slip_rows = []
    for label, points, expected in slip_cases:
        result = trace(points)
        slip_rows.append({
            "case": label,
            "expected_dangerous": expected,
            "actual_dangerous": result["dangerous"],
            "pass": result["dangerous"] == expected,
            "result": result,
        })
    dump("slip_metric_unit_test_results.json", {
        "pass": all(row["pass"] for row in slip_rows),
        "cases": slip_rows,
    })
    if not quaternion_pass or not legacy_fails or not all(row["pass"] for row in slip_rows):
        raise SystemExit("Stage 6 unit-test gate failed")
    print(digest)


if __name__ == "__main__":
    main()
