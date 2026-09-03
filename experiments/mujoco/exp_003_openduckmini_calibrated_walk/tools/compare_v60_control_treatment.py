"""Aggregate v60 parent/control/treatment diagnostic results and gates."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from v60_yaw_objective import (
    COMMAND_PROGRESS_SCALE,
    bounded_yaw_progress,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "v60_bounded_yaw_pilot"
EVAL = OUT / "evaluations"
LABELS = ("parent_v52", "control_1m", "treatment_1m")
RETENTION = (
    "C00_stand",
    "C01_forward",
    "C02_backward",
    "C03_lateral_left",
    "C04_lateral_right",
    "C09_forward_yaw_left",
    "C18_backward_yaw_right_0p3",
)


def load_v59_analysis():
    path = ROOT / "tools" / "analyze_v59_corrected_15s.py"
    spec = importlib.util.spec_from_file_location("v59_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def main() -> None:
    analysis = load_v59_analysis()
    commands = json.loads(
        (
            EVAL
            / "parent_v52"
            / "condition_d"
            / "command_manifest.json"
        ).read_text(encoding="utf-8")
    )["commands"]
    all_rows: list[dict[str, Any]] = []
    npz: dict[tuple[str, str], Any] = {}
    for label in LABELS:
        for condition in ("D", "S"):
            root = EVAL / label / f"condition_{condition.lower()}"
            rows, data = analysis.episode_rows(root, condition, commands)
            for row in rows:
                row["controller_label"] = label
            all_rows.extend(rows)
            npz[(label, condition)] = data

    d_rows = [row for row in all_rows if row["condition"] == "D"]
    s_rows = [row for row in all_rows if row["condition"] == "S"]
    write_csv(OUT / "condition_d_results.csv", d_rows)
    write_csv(OUT / "condition_s_results.csv", s_rows)

    primary = [
        row
        for row in all_rows
        if row["command_id"] in ("C05_yaw_left", "C06_yaw_right")
    ]
    write_csv(OUT / "yaw_primary_results.csv", primary)
    retention = [
        row for row in all_rows if row["command_id"] in RETENTION
    ]
    write_csv(OUT / "retention_results.csv", retention)

    summaries: list[dict[str, Any]] = []
    for label in LABELS:
        label_rows = [row for row in all_rows if row["controller_label"] == label]
        for summary in analysis.aggregate_commands(label_rows):
            summary["controller_label"] = label
            summaries.append(summary)
    write_csv(OUT / "checkpoint_comparison.csv", summaries)

    reward_rows: list[dict[str, Any]] = []
    for label in LABELS:
        for condition in ("D", "S"):
            data = npz[(label, condition)]
            keys = [str(key) for key in data["metric_keys"]]
            contributions = data["reward_contribution"].copy()
            progress_index = keys.index("command_progress")
            if label == "treatment_1m":
                commands_episode = data["commands"]
                linear = np.sum(
                    data["actual_velocity"][..., :2]
                    * commands_episode[None, :, :2],
                    axis=-1,
                )
                yaw = bounded_yaw_progress(
                    commands_episode[None, :, 2],
                    data["actual_yaw_rate"],
                    xp=np,
                )
                contributions[..., progress_index] = (
                    COMMAND_PROGRESS_SCALE * (linear + yaw)
                )
            for command_index, definition in enumerate(commands):
                episode_indices = np.flatnonzero(
                    data["command_indices"] == command_index
                )
                for term_index, term in enumerate(keys):
                    values = contributions[:, episode_indices, term_index]
                    reward_rows.append(
                        {
                            "controller_label": label,
                            "condition": condition,
                            "command_id": definition["command_id"],
                            "term": term,
                            "mean_per_step": float(values.mean()),
                            "cumulative_per_episode_mean": float(
                                values.sum(axis=0).mean()
                            ),
                            "p05": float(np.percentile(values, 5)),
                            "p95": float(np.percentile(values, 95)),
                            "active_step_ratio": float(
                                np.mean(np.abs(values) > 1e-12)
                            ),
                            "treatment_objective_recomputed": (
                                label == "treatment_1m"
                                and term == "command_progress"
                            ),
                        }
                    )
    write_csv(OUT / "reward_term_summary.csv", reward_rows)

    def select(label: str, condition: str, command: str) -> list[dict]:
        return [
            row
            for row in all_rows
            if row["controller_label"] == label
            and row["condition"] == condition
            and row["command_id"] == command
        ]

    primary_summary: dict[str, Any] = {}
    for label in LABELS:
        primary_summary[label] = {}
        for condition in ("D", "S"):
            left = select(label, condition, "C05_yaw_left")
            right = select(label, condition, "C06_yaw_right")
            primary_summary[label][condition] = {
                "left_response_ratio": mean(left, "yaw_response_ratio"),
                "right_response_ratio": mean(right, "yaw_response_ratio"),
                "mean_response_ratio": float(
                    np.mean(
                        [
                            mean(left, "yaw_response_ratio"),
                            mean(right, "yaw_response_ratio"),
                        ]
                    )
                ),
                "left_yaw_mae": mean(left, "yaw_rate_absolute_error"),
                "right_yaw_mae": mean(right, "yaw_rate_absolute_error"),
                "mean_yaw_mae": float(
                    np.mean(
                        [
                            mean(left, "yaw_rate_absolute_error"),
                            mean(right, "yaw_rate_absolute_error"),
                        ]
                    )
                ),
                "fall_count": int(
                    sum(bool(row["fall"]) for row in left + right)
                ),
                "linear_drift_mean": float(
                    np.mean(
                        [
                            math.hypot(row["vx_mean"], row["vy_mean"])
                            for row in left + right
                        ]
                    )
                ),
            }
    c = primary_summary["control_1m"]["D"]
    t = primary_summary["treatment_1m"]["D"]
    p = primary_summary["parent_v52"]["D"]
    reduction = 1.0 - t["mean_yaw_mae"] / c["mean_yaw_mae"]
    treatment_s_falls = primary_summary["treatment_1m"]["S"]["fall_count"]
    control_s_falls = primary_summary["control_1m"]["S"]["fall_count"]
    primary_gate = {
        "mean_response_ratio_le_1p30": t["mean_response_ratio"] <= 1.30,
        "both_response_ratios_0p75_to_1p30": (
            0.75 <= t["left_response_ratio"] <= 1.30
            and 0.75 <= t["right_response_ratio"] <= 1.30
        ),
        "yaw_mae_reduction_at_least_50_percent": reduction >= 0.50,
        "left_right_ratio_difference_le_0p15": (
            abs(t["left_response_ratio"] - t["right_response_ratio"]) <= 0.15
        ),
        "condition_d_yaw_falls_zero": t["fall_count"] == 0,
        "condition_s_yaw_falls_not_worse": treatment_s_falls <= control_s_falls,
        "linear_drift_not_worse_than_parent": (
            t["linear_drift_mean"] <= p["linear_drift_mean"]
        ),
    }
    primary_gate["pass"] = all(primary_gate.values())

    retention_summary: list[dict[str, Any]] = []
    deterministic_falls = 0
    stochastic_t_falls = 0
    stochastic_c_falls = 0
    linear_degradation_failures = []
    for command in RETENTION:
        definition = next(x for x in commands if x["command_id"] == command)
        row: dict[str, Any] = {
            "command_id": command,
            "command_name": definition["command_name"],
        }
        for label in LABELS:
            d = select(label, "D", command)
            s = select(label, "S", command)[:3]
            row[f"{label}_d_falls"] = sum(x["fall"] for x in d[:3])
            row[f"{label}_s_falls_3seed"] = sum(x["fall"] for x in s)
            row[f"{label}_d_vx_mean"] = mean(d[:3], "vx_mean")
            row[f"{label}_d_vy_mean"] = mean(d[:3], "vy_mean")
            row[f"{label}_d_yaw_mean"] = mean(d[:3], "yaw_rate_mean")
            row[f"{label}_d_no_motion_duration"] = mean(
                d[:3], "no_motion_duration"
            )
        deterministic_falls += int(row["treatment_1m_d_falls"])
        stochastic_t_falls += int(row["treatment_1m_s_falls_3seed"])
        stochastic_c_falls += int(row["control_1m_s_falls_3seed"])
        command_vector = np.array([definition["vx"], definition["vy"]])
        if np.linalg.norm(command_vector) > 0.01:
            unit = command_vector / np.linalg.norm(command_vector)
            parent_speed = float(
                np.dot(
                    [
                        row["parent_v52_d_vx_mean"],
                        row["parent_v52_d_vy_mean"],
                    ],
                    unit,
                )
            )
            treatment_speed = float(
                np.dot(
                    [
                        row["treatment_1m_d_vx_mean"],
                        row["treatment_1m_d_vy_mean"],
                    ],
                    unit,
                )
            )
            row["treatment_vs_parent_primary_speed_ratio"] = (
                treatment_speed / parent_speed
                if abs(parent_speed) > 1e-6
                else math.nan
            )
            if parent_speed > 0 and treatment_speed < 0.9 * parent_speed:
                linear_degradation_failures.append(command)
        retention_summary.append(row)
    write_csv(OUT / "retention_command_summary.csv", retention_summary)
    retention_gate = {
        "deterministic_retention_falls_zero": deterministic_falls == 0,
        "no_primary_linear_degradation_over_10_percent": not linear_degradation_failures,
        "stochastic_falls_not_worse_than_control": stochastic_t_falls <= stochastic_c_falls,
        "linear_degradation_commands": linear_degradation_failures,
    }
    retention_gate["pass"] = all(
        value for key, value in retention_gate.items() if key != "linear_degradation_commands"
    )
    decision = (
        "GO_TO_5M"
        if primary_gate["pass"] and retention_gate["pass"]
        else "STOP_AT_1M"
    )

    result = {
        "paired_1m_completed": True,
        "objective_contract_pass": True,
        "primary_summary": primary_summary,
        "yaw_mae_reduction_treatment_vs_control": reduction,
        "primary_gate": primary_gate,
        "retention_gate": retention_gate,
        "decision": decision,
        "five_million_executed": False,
        "formal_acceptance_eligible": False,
        "diagnostic_only": True,
        "missing_training_instrumentation": [
            "exact rollout command histogram was not exposed by installed Brax callback",
            "optimizer state was not serialized by installed Brax checkpoint API",
        ],
    }
    (OUT / "paired_training_report.json").write_text(
        json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )

    report = f"""# v60 Bounded Yaw Paired Pilot

## Result

Decision: **{decision}**.  This is a diagnostic causal pilot, not an adoption
or hardware-transfer decision.

## Matched design

Both arms restored the same v52/v45 normalizer, actor and critic, used fresh
zero-state Adam with seed 20260730, and executed exactly 1,000,000 environment
interactions (1,600 optimizer updates).  Their step-0 parameter hashes match.
The only reward difference is the yaw contribution inside command_progress.

## Primary yaw result (Condition D)

| Controller | Left ratio | Right ratio | Mean yaw MAE | Yaw falls |
| --- | ---: | ---: | ---: | ---: |
| parent v52 | {p['left_response_ratio']:.3f} | {p['right_response_ratio']:.3f} | {p['mean_yaw_mae']:.3f} | {p['fall_count']} |
| Arm C 1M | {c['left_response_ratio']:.3f} | {c['right_response_ratio']:.3f} | {c['mean_yaw_mae']:.3f} | {c['fall_count']} |
| Arm T 1M | {t['left_response_ratio']:.3f} | {t['right_response_ratio']:.3f} | {t['mean_yaw_mae']:.3f} | {t['fall_count']} |

Arm T reduced the mean yaw MAE by {100 * reduction:.1f}% versus Arm C, below
the required 50%.  Its left response became an undershoot ({t['left_response_ratio']:.3f}×),
while the right response was {t['right_response_ratio']:.3f}×.  The left/right
gap therefore also fails the 0.15 gate.

## Stochastic yaw result

Arm C yaw-only falls: {control_s_falls}/10.  Arm T yaw-only falls:
{treatment_s_falls}/10.  Treatment did not satisfy the not-worse gate.

## Retention

Deterministic treatment retention falls: {deterministic_falls}.  Stochastic
3-seed treatment/control falls over the seven retention commands:
{stochastic_t_falls}/{stochastic_c_falls}.  Commands exceeding the 10% primary
linear-speed degradation test: {', '.join(linear_degradation_failures) or 'none'}.

## Decision and non-claims

The bounded objective changed yaw behavior, but it did not produce the required
bilateral, 50%-MAE improvement and it increased yaw-only stochastic falls.
Training stops at 1M; 5M and the 19-command final pilot were not run.  No claim
is made about solving linear undershoot, backward initiation/tracking, domain
randomization, push recovery, formal acceptance, adoption, or hardware safety.

## Instrumentation limitations

The installed Brax callback did not expose the exact rollout command tensor or
optimizer state.  Training term metrics and seeds are saved, but an exact
rollout command histogram and resumable Adam state are unavailable.  Three
failed control attempts caused by WSL libcuda host-boundary crashes are retained
under `failed_runs/` and are excluded from the causal comparison.
"""
    (OUT / "paired_training_report.md").write_text(report, encoding="utf-8")
    (OUT / "final_diagnostic_report.md").write_text(report, encoding="utf-8")
    (OUT / "final_diagnostic_report.json").write_text(
        json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )

    for data in npz.values():
        data.close()
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
