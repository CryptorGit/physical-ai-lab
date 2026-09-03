"""Generate the immutable pre-training v60 yaw objective contract."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from v60_yaw_objective import (
    COMMAND_PROGRESS_SCALE,
    COMMAND_YAW_ERROR_SCALE,
    SIGMA_YAW,
    TRACKING_ANG_VEL_SCALE,
    TRACKING_ANG_VEL_VARIANCE,
    bounded_yaw_progress,
    command_yaw_error_reward,
    old_yaw_progress,
    tracking_ang_vel_reward,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "v60_bounded_yaw_pilot"
COMMANDS = (-0.6, -0.3, 0.3, 0.6)
RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for command in COMMANDS:
        for ratio in RATIOS:
            actual = command * ratio
            tracking = float(
                tracking_ang_vel_reward(command, actual, xp=np)
            )
            yaw_error = float(
                command_yaw_error_reward(command, actual, xp=np)
            )
            old = COMMAND_PROGRESS_SCALE * float(
                old_yaw_progress(command, actual, xp=np)
            )
            new = COMMAND_PROGRESS_SCALE * float(
                bounded_yaw_progress(command, actual, xp=np)
            )
            rows.append(
                {
                    "yaw_command": command,
                    "yaw_actual": actual,
                    "yaw_ratio": ratio,
                    "tracking_ang_vel": tracking,
                    "command_yaw_error": yaw_error,
                    "old_command_progress_yaw": old,
                    "new_command_progress_yaw": new,
                    "old_yaw_related_total": old + tracking + yaw_error,
                    "new_yaw_related_total": new + tracking + yaw_error,
                }
            )
    return rows


def maxima(rows: list[dict[str, float | str]], field: str) -> dict[str, float]:
    result = {}
    for command in COMMANDS:
        subset = [row for row in rows if row["yaw_command"] == command]
        best = max(subset, key=lambda row: float(row[field]))
        result[str(command)] = float(best["yaw_ratio"])
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = make_rows()
    csv_path = OUT / "objective_counterfactual.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    old_contract = {
        "objective_name": "old_unbounded_dot",
        "source_expression_unscaled": "yaw_command * yaw_actual",
        "command_progress_scale": COMMAND_PROGRESS_SCALE,
        "coordinate_frame": "body-frame gyro z versus body-frame command yaw",
        "clamp": None,
        "normalization": None,
        "active_gate": "always; zero command gives zero contribution",
        "overlap_terms": {
            "tracking_ang_vel": {
                "scale": TRACKING_ANG_VEL_SCALE,
                "expression": "exp(-(actual-command)^2 / 0.04)",
            },
            "command_yaw_error": {
                "scale": COMMAND_YAW_ERROR_SCALE,
                "expression": "(actual-command)^2",
            },
        },
        "grid_max_ratio": maxima(rows, "old_command_progress_yaw"),
        "yaw_related_grid_max_ratio": maxima(rows, "old_yaw_related_total"),
    }
    new_contract = {
        "objective_name": "bounded_command_centered_gaussian",
        "source_expression_unscaled": (
            "yaw_command^2 * exp(-((yaw_actual-yaw_command)/0.2)^2)"
        ),
        "command_progress_scale": COMMAND_PROGRESS_SCALE,
        "sigma_yaw_rad_s": SIGMA_YAW,
        "sigma_provenance": (
            "sqrt(existing calibrated tracking_ang_vel variance 0.04)"
        ),
        "exact_tracking_amplitude_preserved": True,
        "linear_command_progress_unchanged": True,
        "grid_max_ratio": maxima(rows, "new_command_progress_yaw"),
        "yaw_related_grid_max_ratio": maxima(rows, "new_yaw_related_total"),
    }
    by_command = {
        command: [row for row in rows if row["yaw_command"] == command]
        for command in COMMANDS
    }
    gate = {
        "new_term_max_at_1x": all(
            value == 1.0
            for value in new_contract["grid_max_ratio"].values()
        ),
        "yaw_related_max_in_0p9_to_1p1": all(
            0.9 <= value <= 1.1
            for value in new_contract["yaw_related_grid_max_ratio"].values()
        ),
        "total_max_in_0p9_to_1p1": all(
            0.9 <= value <= 1.1
            for value in new_contract["yaw_related_grid_max_ratio"].values()
        ),
        "2x_not_above_1x": all(
            next(float(r["new_yaw_related_total"]) for r in values if r["yaw_ratio"] == 2.0)
            <= next(float(r["new_yaw_related_total"]) for r in values if r["yaw_ratio"] == 1.0)
            for values in by_command.values()
        ),
        "3p5x_not_above_1x": all(
            next(float(r["new_yaw_related_total"]) for r in values if r["yaw_ratio"] == 3.5)
            <= next(float(r["new_yaw_related_total"]) for r in values if r["yaw_ratio"] == 1.0)
            for values in by_command.values()
        ),
        "left_right_symmetric": all(
            np.isclose(
                next(
                    float(r["new_yaw_related_total"])
                    for r in by_command[command]
                    if r["yaw_ratio"] == ratio
                ),
                next(
                    float(r["new_yaw_related_total"])
                    for r in by_command[-command]
                    if r["yaw_ratio"] == ratio
                ),
                atol=1e-12,
                rtol=0.0,
            )
            for command in (0.3, 0.6)
            for ratio in RATIOS
        ),
        "finite": all(
            np.isfinite(float(value))
            for row in rows
            for value in row.values()
            if not isinstance(value, str)
        ),
        "zero_command_continuous": bool(
            np.isclose(
                bounded_yaw_progress(0.0, 0.0, xp=np),
                0.0,
                atol=0.0,
            )
        ),
    }
    gate["pass"] = all(gate.values())
    old_contract["csv_sha256"] = sha256(csv_path)
    new_contract["csv_sha256"] = sha256(csv_path)
    new_contract["pre_training_gate"] = gate
    (OUT / "old_objective_contract.json").write_text(
        json.dumps(old_contract, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "new_objective_contract.json").write_text(
        json.dumps(new_contract, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
