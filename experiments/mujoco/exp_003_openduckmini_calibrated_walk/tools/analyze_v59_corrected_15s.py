#!/usr/bin/env python3
"""Aggregate the v59 corrected 15-second diagnostic without rerunning physics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np


DT = 0.02
NO_MOTION_LINEAR = 0.02
NO_MOTION_YAW = 0.05


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def longest_true(mask: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(mask, bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def first_true(mask: np.ndarray) -> float | None:
    indices = np.flatnonzero(mask)
    return float((indices[0] + 1) * DT) if len(indices) else None


def settling_time(
    primary: np.ndarray,
    target: float,
    orthogonal: np.ndarray,
    yaw: np.ndarray,
    yaw_target: float,
) -> float | None:
    tolerance = max(abs(target) * 0.2, 0.01)
    good = np.abs(primary - target) <= tolerance
    good &= np.abs(orthogonal) <= 0.04
    good &= np.abs(yaw - yaw_target) <= 0.2
    window = 50
    if len(good) < window:
        return None
    valid = np.convolve(good.astype(np.int32), np.ones(window, int), "valid")
    indices = np.flatnonzero(valid == window)
    return float((indices[0] + 1) * DT) if len(indices) else None


def p(value: np.ndarray, percentile: float) -> float:
    return float(np.percentile(value, percentile))


def classify_episode(row: dict[str, Any]) -> tuple[str, list[str]]:
    secondary = []
    if row["fall"]:
        return "FALL", ["UNSTABLE_GAIT"]
    if row["termination"]:
        return "TERMINATED_OTHER", []
    moving_command = (
        math.hypot(row["commanded_vx"], row["commanded_vy"]) > 0.01
        or abs(row["commanded_yaw_rate"]) > 0.01
    )
    if moving_command and row["no_motion_fraction"] > 0.8:
        return "NO_MOTION", []
    if math.hypot(row["commanded_vx"], row["commanded_vy"]) > 0.01:
        if row["direction_cosine_similarity"] < 0:
            secondary.append("WRONG_DIRECTION")
        if row["speed_ratio"] < 0.6:
            secondary.append("LINEAR_UNDERSHOOT")
        elif row["speed_ratio"] > 1.4:
            secondary.append("LINEAR_OVERSHOOT")
        if abs(row["orthogonal_velocity_mean"]) > 0.04:
            secondary.append("LATERAL_DRIFT")
    if abs(row["commanded_yaw_rate"]) > 0.01:
        if abs(row["yaw_response_ratio"]) > 1.25:
            secondary.append("YAW_OVERSHOOT")
        elif np.sign(row["yaw_rate_mean"]) != np.sign(row["commanded_yaw_rate"]):
            secondary.append("WRONG_DIRECTION")
        elif abs(row["yaw_response_ratio"]) < 0.75:
            secondary.append("YAW_UNDERSHOOT")
    elif abs(row["yaw_rate_mean"]) > 0.1:
        secondary.append("YAW_DRIFT")
    if row["roll_pitch_excursion"] > 0.5 or row["base_height_min"] < 0.12:
        secondary.append("UNSTABLE_GAIT")
    secondary = list(dict.fromkeys(secondary))
    if not secondary:
        return "TRACKING_OK", []
    severe = [
        value
        for value in secondary
        if value in ("WRONG_DIRECTION", "LATERAL_DRIFT", "YAW_OVERSHOOT", "UNSTABLE_GAIT")
    ]
    if len(secondary) > 1 and severe:
        return "MIXED_FAILURE", secondary
    return secondary[0], secondary[1:]


def episode_rows(root: Path, condition: str, commands: list[dict]) -> tuple[list[dict], np.lib.npyio.NpzFile]:
    data = np.load(
        root / "raw_logs" / f"condition_{condition.lower()}_raw.npz",
        allow_pickle=False,
    )
    rows = []
    for episode in range(95):
        command_index = int(data["command_indices"][episode])
        seed = int(data["seed_indices"][episode])
        definition = commands[command_index]
        termination_series = data["termination"][:, episode].astype(bool)
        termination_indices = np.flatnonzero(termination_series)
        active_steps = (
            int(termination_indices[0]) + 1 if len(termination_indices) else 750
        )
        sl = slice(0, active_steps)
        velocity = data["actual_velocity"][sl, episode]
        vx, vy = velocity[:, 0], velocity[:, 1]
        yaw = data["actual_yaw_rate"][sl, episode]
        cmd = np.array(
            [definition["vx"], definition["vy"], definition["yaw_rate"]],
            np.float64,
        )
        cmd_linear = cmd[:2]
        cmd_speed = float(np.linalg.norm(cmd_linear))
        mean_linear = np.array([vx.mean(), vy.mean()])
        actual_speed = np.linalg.norm(velocity[:, :2], axis=1)
        if cmd_speed:
            unit = cmd_linear / cmd_speed
            primary = velocity[:, :2] @ unit
            orthogonal = velocity[:, :2] @ np.array([-unit[1], unit[0]])
            speed_ratio = float(primary.mean() / cmd_speed)
            denominator = np.linalg.norm(mean_linear) * cmd_speed
            cosine = (
                float(np.dot(mean_linear, cmd_linear) / denominator)
                if denominator
                else 0.0
            )
            onset = first_true(primary >= 0.2 * cmd_speed)
            t50 = first_true(primary >= 0.5 * cmd_speed)
            t80 = first_true(primary >= 0.8 * cmd_speed)
            settle = settling_time(
                primary, cmd_speed, orthogonal, yaw, cmd[2]
            )
        else:
            primary = np.sign(cmd[2]) * yaw if cmd[2] else actual_speed
            orthogonal = np.zeros_like(primary)
            speed_ratio = math.nan
            cosine = math.nan
            threshold = 0.2 * abs(cmd[2]) if cmd[2] else 0.02
            onset = first_true(primary >= threshold)
            t50 = first_true(primary >= 0.5 * abs(cmd[2])) if cmd[2] else None
            t80 = first_true(primary >= 0.8 * abs(cmd[2])) if cmd[2] else None
            settle = (
                settling_time(primary, abs(cmd[2]), orthogonal, yaw, cmd[2])
                if cmd[2]
                else None
            )
        contacts = data["foot_contacts"][sl, episode].astype(bool)
        double = contacts.all(axis=1)
        single = np.logical_xor(contacts[:, 0], contacts[:, 1])
        flight = ~contacts.any(axis=1)
        transitions = int(np.count_nonzero(contacts[1:] != contacts[:-1]))
        no_motion = (actual_speed < NO_MOTION_LINEAR) & (
            np.abs(yaw) < NO_MOTION_YAW
        )
        actor = data["actor_residual"][sl, episode]
        target = data["motor_target"][sl, episode]
        frozen = np.zeros(active_steps, dtype=bool)
        if active_steps > 1:
            frozen[1:] = (
                np.max(np.abs(actor[1:] - actor[:-1]), axis=1) < 1e-4
            ) & (
                np.max(np.abs(target[1:] - target[:-1]), axis=1) < 1e-4
            )
        heading = np.unwrap(data["heading"][sl, episode])
        expected_heading = heading[0] + cmd[2] * np.arange(1, active_steps + 1) * DT
        heading_error = heading - expected_heading
        fall_series = data["fall"][sl, episode].astype(bool)
        reason_code = data["termination_reason_code"][sl, episode].astype(int)
        termination = bool(len(termination_indices))
        fall = bool(fall_series.any())
        reason = ""
        if termination:
            code = int(reason_code[active_steps - 1])
            names = []
            if code & 1:
                names.append("fall_upright_or_height")
            if code & 2:
                names.append("head_frame_violation")
            if code & 4:
                names.append("nan_state")
            reason = "+".join(names) or "environment_done_unspecified"
        head = data["initial_head_command"][episode]
        delay = data["delay_index"][sl, episode].astype(int)
        push = data["push_velocity_increment"][sl, episode]
        joint_limit = data["joint_limit_state"][sl, episode]
        target_limit = data["target_limit_state"][sl, episode]
        action_clip = data["action_clip_state"][sl, episode]
        slip = data["foot_slip"][sl, episode]
        actuator_force = data["actuator_force"][sl, episode]
        roll = data["roll"][sl, episode]
        pitch = data["pitch"][sl, episode]
        yaw_ratio = float(yaw.mean() / cmd[2]) if cmd[2] else math.nan
        signed_yaw = np.sign(cmd[2]) * yaw if cmd[2] else np.abs(yaw)
        overshoot = (
            float(p(signed_yaw, 95) / abs(cmd[2])) if cmd[2] else math.nan
        )
        row = {
            "condition": condition,
            "command_id": definition["command_id"],
            "command_name": definition["command_name"],
            "seed": seed,
            "diagnostic_only": True,
            "formal_acceptance_eligible": False,
            "enough_episodes": False,
            "requested_seconds": 15.0,
            "completed_seconds_before_first_termination": active_steps * DT,
            "completed_steps_before_first_termination": active_steps,
            "commanded_vx": cmd[0],
            "commanded_vy": cmd[1],
            "commanded_yaw_rate": cmd[2],
            "vx_mean": float(vx.mean()),
            "vx_median": float(np.median(vx)),
            "vx_p05": p(vx, 5),
            "vx_p95": p(vx, 95),
            "vy_mean": float(vy.mean()),
            "vy_median": float(np.median(vy)),
            "vy_p05": p(vy, 5),
            "vy_p95": p(vy, 95),
            "vx_absolute_error": float(np.mean(np.abs(vx - cmd[0]))),
            "vy_absolute_error": float(np.mean(np.abs(vy - cmd[1]))),
            "linear_vector_rmse": float(
                np.sqrt(np.mean(np.sum((velocity[:, :2] - cmd_linear) ** 2, axis=1)))
            ),
            "direction_cosine_similarity": cosine,
            "speed_ratio": speed_ratio,
            "orthogonal_velocity_mean": float(orthogonal.mean()),
            "yaw_rate_mean": float(yaw.mean()),
            "yaw_rate_median": float(np.median(yaw)),
            "yaw_rate_p05": p(yaw, 5),
            "yaw_rate_p95": p(yaw, 95),
            "yaw_rate_absolute_error": float(np.mean(np.abs(yaw - cmd[2]))),
            "yaw_response_ratio": yaw_ratio,
            "yaw_overshoot_ratio": overshoot,
            "integrated_yaw_error": float(np.sum(yaw - cmd[2]) * DT),
            "integrated_absolute_yaw_error": float(np.sum(np.abs(yaw - cmd[2])) * DT),
            "final_heading_error": float(heading_error[-1]),
            "movement_onset_time": onset,
            "time_to_50_percent": t50,
            "time_to_80_percent": t80,
            "settling_time": settle,
            "no_motion_duration": float(no_motion.sum() * DT),
            "no_motion_fraction": float(no_motion.mean()),
            "double_support_duration": float(double.sum() * DT),
            "double_support_ratio": float(double.mean()),
            "single_support_ratio": float(single.mean()),
            "flight_ratio": float(flight.mean()),
            "frozen_controller_duration": float(longest_true(frozen) * DT),
            "left_contact_ratio": float(contacts[:, 0].mean()),
            "right_contact_ratio": float(contacts[:, 1].mean()),
            "support_transition_count": transitions,
            "left_right_contact_asymmetry": float(
                abs(contacts[:, 0].mean() - contacts[:, 1].mean())
            ),
            "foot_slip_mean": float(np.mean(slip * contacts)),
            "foot_slip_p95": p((slip * contacts).reshape(-1), 95),
            "roll_excursion": float(np.max(np.abs(roll))),
            "pitch_excursion": float(np.max(np.abs(pitch))),
            "roll_pitch_excursion": float(
                max(np.max(np.abs(roll)), np.max(np.abs(pitch)))
            ),
            "base_height_min": float(
                data["base_height"][sl, episode].min()
            ),
            "vertical_velocity_p95_abs": p(
                np.abs(data["vertical_velocity"][sl, episode]), 95
            ),
            "teacher_active": bool(data["teacher_active"][0, episode]),
            "teacher_mode": definition["teacher_mode"],
            "teacher_action_rms": float(
                np.sqrt(np.mean(data["teacher_action"][sl, episode] ** 2))
            ),
            "actor_residual_rms": float(np.sqrt(np.mean(actor**2))),
            "motor_target_peak": float(np.max(np.abs(target))),
            "action_clipping_count": int(action_clip.sum()),
            "joint_limit_activation_count": int(joint_limit.sum()),
            "target_limit_activation_count": int(target_limit.sum()),
            "actuator_force_p95_abs": p(np.abs(actuator_force).reshape(-1), 95),
            "actuator_force_max_abs": float(np.max(np.abs(actuator_force))),
            "solver_niter_max": int(
                np.max(data["solver_niter"][sl, episode])
            ),
            "delay_index_min": int(delay.min()),
            "delay_index_max": int(delay.max()),
            "delay_index_mean": float(delay.mean()),
            "push_event_count": int(
                np.count_nonzero(np.linalg.norm(push, axis=1) > 0)
            ),
            "push_max_velocity_increment": float(
                np.max(np.linalg.norm(push, axis=1))
            ),
            "head_command": json.dumps(head.tolist()),
            "head_locked": False,
            "head_command_source": "training-compatible reset sample; fixed",
            "head_command_changes": 0,
            "termination": termination,
            "termination_time": active_steps * DT if termination else None,
            "termination_reason": reason,
            "fall": fall,
            "fall_time": first_true(fall_series),
            "contact_force": "UNAVAILABLE",
            "joint_torque": "UNAVAILABLE_SEPARATE_FROM_ACTUATOR_FORCE",
        }
        primary, secondary = classify_episode(row)
        row["primary_failure"] = primary
        row["secondary_failures"] = json.dumps(secondary)
        rows.append(row)
    return rows, data


def aggregate_commands(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    keys = sorted({(row["condition"], row["command_id"]) for row in rows})
    for condition, command_id in keys:
        group = [
            row
            for row in rows
            if row["condition"] == condition and row["command_id"] == command_id
        ]
        first = group[0]
        primary_counts = {}
        for row in group:
            primary_counts[row["primary_failure"]] = (
                primary_counts.get(row["primary_failure"], 0) + 1
            )
        primary = (
            "FALL"
            if any(row["fall"] for row in group)
            else sorted(
                primary_counts,
                key=lambda value: (-primary_counts[value], value),
            )[0]
        )
        numeric_means = {}
        for key in (
            "vx_mean",
            "vy_mean",
            "linear_vector_rmse",
            "speed_ratio",
            "direction_cosine_similarity",
            "orthogonal_velocity_mean",
            "yaw_rate_mean",
            "yaw_response_ratio",
            "yaw_overshoot_ratio",
            "integrated_absolute_yaw_error",
            "no_motion_duration",
            "double_support_ratio",
            "single_support_ratio",
            "flight_ratio",
            "left_contact_ratio",
            "right_contact_ratio",
            "foot_slip_mean",
            "roll_pitch_excursion",
            "base_height_min",
            "actor_residual_rms",
            "teacher_action_rms",
            "push_event_count",
        ):
            values = np.asarray([row[key] for row in group], float)
            numeric_means[key] = (
                float(np.nanmean(values))
                if not np.isnan(values).all()
                else math.nan
            )
        output.append(
            {
                "condition": condition,
                "command_id": command_id,
                "command_name": first["command_name"],
                "vx": first["commanded_vx"],
                "vy": first["commanded_vy"],
                "yaw_rate": first["commanded_yaw_rate"],
                "seed_count": 5,
                "diagnostic_only": True,
                "formal_acceptance_eligible": False,
                "enough_episodes": False,
                **numeric_means,
                "fall_count": sum(row["fall"] for row in group),
                "termination_count": sum(row["termination"] for row in group),
                "earliest_fall_time": min(
                    (row["fall_time"] for row in group if row["fall_time"] is not None),
                    default=None,
                ),
                "primary_failure": primary,
                "failure_counts": json.dumps(primary_counts, sort_keys=True),
            }
        )
    return output


def reward_rows(
    condition: str,
    data: np.lib.npyio.NpzFile,
    commands: list[dict],
) -> list[dict[str, Any]]:
    output = []
    metric_keys = [str(value) for value in data["metric_keys"]]
    scales = data["metric_scales"]
    for command_index, definition in enumerate(commands):
        episode_indices = np.arange(command_index * 5, command_index * 5 + 5)
        masks = []
        max_steps = 750
        for episode in episode_indices:
            term = np.flatnonzero(data["termination"][:, episode])
            active = int(term[0]) + 1 if len(term) else max_steps
            mask = np.zeros(max_steps, bool)
            mask[:active] = True
            masks.append(mask)
        mask = np.stack(masks, axis=1)
        for term_index, term in enumerate(metric_keys):
            logged_double_scaled = data["reward_contribution"][
                :, episode_indices, term_index
            ][mask]
            # joystick.step stores already-weighted values in state.metrics:
            # reward/key = weighted positive contribution;
            # cost/key = negative(weighted negative contribution).
            # The evaluator's diagnostic array multiplied abs(weight) once
            # more. Undo that logged scaling and restore the cost sign.
            scale = float(scales[term_index])
            values = (
                logged_double_scaled
                / abs(scale)
                * (1.0 if scale > 0 else -1.0)
            )
            output.append(
                {
                    "condition": condition,
                    "command_id": definition["command_id"],
                    "term": term,
                    "weight": float(scales[term_index]),
                    "mean_per_step_pre_dt": float(values.mean()),
                    "cumulative_weighted_return": float(values.sum() * DT),
                    "p05": p(values, 5),
                    "p95": p(values, 95),
                    "active_step_ratio": float(np.mean(np.abs(values) > 1e-12)),
                }
            )
    return output


def legacy_comparison(
    root: Path, summary: list[dict], legacy_matrix: Path
) -> list[dict[str, Any]]:
    legacy = list(csv.DictReader(legacy_matrix.open(encoding="utf-8")))
    mapping = {tuple(round(float(row[key]), 6) for key in ("vx_cmd", "vy_cmd", "yaw_cmd")): row for row in legacy}
    rows = []
    for corrected in summary:
        key = tuple(round(float(corrected[name]), 6) for name in ("vx", "vy", "yaw_rate"))
        old = mapping[key]
        legacy_falls = int(old["fall_count"])
        legacy_no_motion = int(old["no_motion_seed_count_proxy"])
        corrected_failure = corrected["primary_failure"]
        legacy_yaw_overshoot = float(old["yaw_overshoot_ratio_abs"] or 0)
        corrected_yaw_overshoot = abs(
            float(corrected["yaw_response_ratio"])
        ) if not math.isnan(float(corrected["yaw_response_ratio"])) else 0.0
        if legacy_yaw_overshoot > 1.25 and corrected_yaw_overshoot > 1.25:
            category = "policy_induced"
        elif (
            legacy_falls
            and not corrected["fall_count"]
            and corrected_failure == "TRACKING_OK"
        ):
            category = "resolved_by_corrected_path"
        elif legacy_falls and not corrected["fall_count"]:
            category = "mixed"
        elif legacy_no_motion == 5 and corrected_failure != "NO_MOTION":
            category = (
                "evaluation_induced"
                if corrected_failure == "TRACKING_OK"
                else "mixed"
            )
        elif legacy_falls and corrected["fall_count"]:
            category = "policy_induced"
        elif corrected_failure == "TRACKING_OK":
            category = "resolved_by_corrected_path"
        else:
            category = "insufficient_data"
        rows.append(
            {
                "condition": corrected["condition"],
                "command_id": corrected["command_id"],
                "vx": corrected["vx"],
                "vy": corrected["vy"],
                "yaw_rate": corrected["yaw_rate"],
                "legacy_vx_mean": old["vx_mean"],
                "legacy_vy_mean": old["vy_mean"],
                "legacy_yaw_mean": old["yaw_mean"],
                "legacy_no_motion_seed_count": legacy_no_motion,
                "legacy_fall_count": legacy_falls,
                "corrected_vx_mean": corrected["vx_mean"],
                "corrected_vy_mean": corrected["vy_mean"],
                "corrected_yaw_mean": corrected["yaw_rate_mean"],
                "corrected_fall_count": corrected["fall_count"],
                "corrected_primary_failure": corrected_failure,
                "classification": category,
                "legacy_raw_timeseries_available": False,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--legacy-matrix", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    command_manifest = json.loads(
        (root / "command_manifest.json").read_text(encoding="utf-8")
    )
    commands = command_manifest["commands"]
    all_episodes = []
    reward = []
    data_by_condition = {}
    for condition in ("D", "S"):
        rows, data = episode_rows(root, condition, commands)
        write_csv(root / f"condition_{condition.lower()}_episode_results.csv", rows)
        all_episodes.extend(rows)
        reward.extend(reward_rows(condition, data, commands))
        data_by_condition[condition] = data
    summary = aggregate_commands(all_episodes)
    write_csv(root / "command_summary.csv", summary)
    write_csv(root / "failure_matrix.csv", summary)
    write_csv(root / "reward_term_summary.csv", reward)
    legacy = legacy_comparison(root, summary, Path(args.legacy_matrix))
    write_csv(root / "legacy_vs_corrected.csv", legacy)

    seed_rows = []
    for condition in ("D", "S"):
        data = data_by_condition[condition]
        for episode in range(95):
            command = commands[int(data["command_indices"][episode])]
            seed = int(data["seed_indices"][episode])
            seed_rows.append(
                {
                    "condition": condition,
                    "command_id": command["command_id"],
                    "seed": seed,
                    "master_seed": 0,
                    "environment_seed": data["environment_keys"][episode].tolist(),
                    "reset_seed": data["reset_keys"][episode].tolist(),
                    "domain_randomization_seed": (
                        data["environment_keys"][episode].tolist()
                        if condition == "S"
                        else None
                    ),
                    "noise_seed": "state.info.rng in episode snapshot",
                    "delay_sample": "per-step delay_index in raw NPZ",
                    "command_seed": "reset seed; body command then frozen by manifest",
                    "head_command_seed": (
                        data["environment_keys"][episode].tolist()
                    ),
                    "push_seed": (
                        "state.info.rng per-step split[2]" if condition == "S" else None
                    ),
                    "head_command": data["initial_head_command"][episode].tolist(),
                }
            )
    seed_manifest = {
        "schema_version": 1,
        "master_seed": 0,
        "seed_derivation": "training_environment_keys; process_id folded with 0",
        "same_backend_replay": "episode snapshot and raw keys saved",
        "episodes": seed_rows,
    }
    (root / "seed_manifest.json").write_text(
        json.dumps(seed_manifest, indent=2), encoding="utf-8"
    )

    summary_json = {
        "diagnostic_only": True,
        "formal_acceptance_eligible": False,
        "enough_episodes": False,
        "scheduled_episode_steps_executed": {
            condition: 19 * 5 * 750 for condition in ("D", "S")
        },
        "episode_termination_counts": {
            condition: sum(
                row["termination"]
                for row in all_episodes
                if row["condition"] == condition
            )
            for condition in ("D", "S")
        },
        "fall_counts": {
            condition: sum(
                row["fall"]
                for row in all_episodes
                if row["condition"] == condition
            )
            for condition in ("D", "S")
        },
        "summary": summary,
    }
    (root / "analysis_summary.json").write_text(
        json.dumps(summary_json, indent=2, allow_nan=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
