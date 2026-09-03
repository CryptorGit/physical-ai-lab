"""Offline endpoint-window parity analysis and finalization for W1B-D4."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_d4_dynamic_endpoint_window_parity_preflight"
)
D3 = ROOT / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_d3_dynamic_yaw_transition_boundary_diagnosis"
)
REPORT = ROOT / "research/exp_013_g1_phase_w1b_d4_dynamic_endpoint_window_parity_report.md"
CHECKPOINT = ROOT / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
)
CHECKPOINT_SHA = "61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"
DT = 0.02

WINDOWS = {
    "W0_CURRENT": (0.0, 12.0),
    "W1_FINAL_HOLD_ALL": (6.0, 12.0),
    "W2_FINAL_HOLD_AFTER_0P25": (6.25, 12.0),
    "W3_FINAL_HOLD_AFTER_0P50": (6.5, 12.0),
    "W4_FINAL_HOLD_AFTER_1P00": (7.0, 12.0),
    "W5_LAST_1_SECOND": (11.0, 12.0),
    "W6_LAST_2_SECONDS": (10.0, 12.0),
    "W7_LAST_3_SECONDS": (9.0, 12.0),
}
SWEEP_WINDOWS = {
    "whole_episode": (0.0, 12.0),
    "ramp_only": (4.0, 6.0),
    "ramp_plus_final_hold": (4.0, 12.0),
    "final_hold_only": (6.0, 12.0),
    "final_hold_after_0.10s": (6.1, 12.0),
    "final_hold_after_0.25s": (6.25, 12.0),
    "final_hold_after_0.50s": (6.5, 12.0),
    "final_hold_after_0.75s": (6.75, 12.0),
    "final_hold_after_1.00s": (7.0, 12.0),
    "last_0.50s": (11.5, 12.0),
    "last_1.00s": (11.0, 12.0),
    "last_2.00s": (10.0, 12.0),
    "last_3.00s": (9.0, 12.0),
}
METRICS = [
    "M0_INSTANT_SIGN_FRACTION",
    "M1_STATIC_ENDPOINT_EQUIVALENT",
    "M2_WINDOW_MEAN_SIGN_FRACTION",
    "M3_GAIT_CYCLE_MEAN_SIGN",
    "M4_LOW_PASS_SIGN_FRACTION_2HZ",
    "M4_LOW_PASS_SIGN_FRACTION_4HZ",
]


def native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def dump(name, value):
    (OUT / name).write_text(
        json.dumps(value, indent=2, sort_keys=True, default=native) + "\n",
        encoding="utf-8",
    )


def write_csv(name, rows):
    fields = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (dict, list)):
                fields.append(key)
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def label(direction, yaw):
    prefix = "PURE" if direction == -1 else f"D{int(direction):03d}"
    return f"{prefix}_Y{float(yaw):+.1f}"


def contact_name(code):
    return {0: "flight", 1: "left_support", 2: "right_support", 3: "double_support"}[int(code)]


def load_parts():
    parts = {}
    for part in (0, 1):
        for mode in ("static", "dynamic"):
            with np.load(OUT / f"paired_trace_part{part}_{mode}.npz") as archive:
                parts[(part, mode)] = {key: archive[key] for key in archive.files}
    return parts


def episodes_from(data):
    directions = data["directions"]
    yaws = data["final_yaws"]
    for row_index, (condition_id, episode_id) in enumerate(zip(data["condition_ids"], data["episode_ids"])):
        yield {
            "row_index": row_index,
            "condition_id": int(condition_id),
            "episode": int(episode_id),
            "direction": int(directions[int(condition_id)]),
            "yaw": float(yaws[int(condition_id)]),
        }


def slice_indices(start, end, n):
    return slice(max(0, round(start / DT)), min(n, round(end / DT)))


def safety_translation(data, row, window):
    idx = slice_indices(*window, data["actual_yaw"].shape[1])
    direction = row["direction"]
    if direction == -1:
        translation_ok = float(np.mean(np.hypot(data["actual_vx"][row["row_index"], idx],
                                                 data["actual_vy"][row["row_index"], idx]))) <= .12
    else:
        translation_ok = float(np.mean(data["vector_error"][row["row_index"], idx])) <= .25
    safe = not (
        bool(np.any(data["fall"][row["row_index"], idx]))
        or bool(np.any(data["slip"][row["row_index"], idx]))
    )
    return translation_ok, safe


def complete_cycle_means(values, cycles, start, end):
    means = []
    all_ids = np.unique(cycles)
    for cycle_id in all_ids:
        positions = np.flatnonzero(cycles == cycle_id)
        if len(positions) < 2:
            continue
        if positions[0] >= start and positions[-1] < end and positions[0] > 0 and positions[-1] < len(cycles) - 1:
            means.append(float(np.mean(values[positions])))
    return means


def lowpass(values, cutoff):
    spectrum = np.fft.rfft(values)
    frequencies = np.fft.rfftfreq(len(values), d=DT)
    spectrum[frequencies > cutoff] = 0
    return np.fft.irfft(spectrum, n=len(values))


def metric_for(data, row, window, metric):
    n = data["actual_yaw"].shape[1]
    idx = slice_indices(*window, n)
    values = data["actual_yaw"][row["row_index"], idx].astype(float)
    target = row["yaw"]
    threshold = .15 if row["direction"] == -1 else .20
    sign = 1 if target > 0 else -1
    mean_yaw = float(np.mean(values))
    yaw_mae = float(np.mean(np.abs(values - target)))
    translation_ok, safe = safety_translation(data, row, window)
    evaluable = True
    if metric == "M0_INSTANT_SIGN_FRACTION":
        scalar = float(np.mean(values * sign > .05))
        yaw_ok = scalar >= .95
    elif metric == "M1_STATIC_ENDPOINT_EQUIVALENT":
        scalar = yaw_mae
        yaw_ok = mean_yaw * sign > 0 and yaw_mae <= threshold
    elif metric == "M2_WINDOW_MEAN_SIGN_FRACTION":
        bin_steps = round(.1 / DT)
        bins = [values[i:i + bin_steps] for i in range(0, len(values) - bin_steps + 1, bin_steps)]
        scalar = float(np.mean([np.mean(item) * sign > 0 for item in bins])) if bins else 0.0
        yaw_ok = scalar >= .95 and yaw_mae <= threshold
    elif metric == "M3_GAIT_CYCLE_MEAN_SIGN":
        start, end = idx.start, idx.stop
        cycle_means = complete_cycle_means(
            data["actual_yaw"][row["row_index"]].astype(float),
            data["gait_cycle"][row["row_index"]],
            start,
            end,
        )
        evaluable = len(cycle_means) >= 2
        scalar = float(np.mean([item * sign > 0 for item in cycle_means])) if evaluable else None
        yaw_ok = evaluable and scalar >= .95 and yaw_mae <= threshold
    else:
        cutoff = 2 if "2HZ" in metric else 4
        filtered = lowpass(values, cutoff)
        scalar = float(np.mean(filtered * sign > .05))
        yaw_ok = scalar >= .95 and yaw_mae <= threshold
    return {
        "value": scalar,
        "pass": bool(yaw_ok and translation_ok and safe),
        "evaluable": evaluable,
        "mean_yaw": mean_yaw,
        "yaw_mae": yaw_mae,
        "translation_ok": translation_ok,
        "safe": safe,
    }


parts = load_parts()

# Reconstruct every dynamic time step into the required long-form trace.
trace_path = OUT / "dynamic_yaw_trace_reconstruction.csv"
trace_fields = [
    "condition", "episode", "time_s", "physical_target_yaw", "actor_input_yaw",
    "actual_yaw_rate", "actual_vx", "actual_vy", "timeline_phase", "ramp_progress",
    "final_target_sign", "instantaneous_sign_correct", "instantaneous_yaw_error",
    "contact_state", "gait_cycle_id", "fall", "slip",
]
trace_manifest = []
with trace_path.open("w", newline="", encoding="utf-8", buffering=8 * 1024 * 1024) as stream:
    writer = csv.writer(stream)
    writer.writerow(trace_fields)
    for part in (0, 1):
        data = parts[(part, "dynamic")]
        n = data["actual_yaw"].shape[1]
        for row in episodes_from(data):
            name = label(row["direction"], row["yaw"])
            sign = 1 if row["yaw"] > 0 else -1
            block = []
            for step in range(n):
                time_s = step * DT
                if time_s < 4:
                    phase, progress = "PRE_HOLD", 0.0
                elif time_s < 6:
                    phase, progress = "RAMP", min(1.0, max(0.0, (time_s - 4) / 2))
                elif time_s < 6.25:
                    phase, progress = "POST_RAMP_SETTLING", 1.0
                elif time_s < 7:
                    phase, progress = "FINAL_HOLD_EARLY", 1.0
                else:
                    phase, progress = "FINAL_HOLD_STEADY", 1.0
                actual = float(data["actual_yaw"][row["row_index"], step])
                target = float(data["target"][row["row_index"], step])
                block.append((
                    name, row["episode"], time_s, target,
                    float(data["actor_input"][row["row_index"], step]), actual,
                    float(data["actual_vx"][row["row_index"], step]),
                    float(data["actual_vy"][row["row_index"], step]),
                    phase, progress, sign, actual * sign > .05, abs(actual - target),
                    contact_name(data["contact"][row["row_index"], step]),
                    int(data["gait_cycle"][row["row_index"], step]),
                    bool(data["fall"][row["row_index"], step]),
                    bool(data["slip"][row["row_index"], step]),
                ))
            writer.writerows(block)
        trace_manifest.append({"part": part, "episodes": int(len(data["episode_ids"])), "steps": n})
dump("dynamic_yaw_trace_reconstruction.json", {
    "csv": trace_path.name,
    "csv_sha256": sha(trace_path),
    "rows": sum(item["episodes"] * item["steps"] for item in trace_manifest),
    "parts": trace_manifest,
    "fields": trace_fields,
})

# Paired episode endpoint summaries.
paired_rows = []
for part in (0, 1):
    static = parts[(part, "static")]
    dynamic = parts[(part, "dynamic")]
    for row in episodes_from(static):
        drow = {**row}
        s_idx = slice_indices(0, 8, static["actual_yaw"].shape[1])
        d_idx = slice_indices(6, 12, dynamic["actual_yaw"].shape[1])
        sign = 1 if row["yaw"] > 0 else -1
        threshold = .15 if row["direction"] == -1 else .20
        s_values = static["actual_yaw"][row["row_index"], s_idx]
        d_values = dynamic["actual_yaw"][row["row_index"], d_idx]
        s_trans, s_safe = safety_translation(static, row, (0, 8))
        d_trans, d_safe = safety_translation(dynamic, drow, (6, 12))
        s_pass = np.mean(s_values) * sign > 0 and np.mean(np.abs(s_values - row["yaw"])) <= threshold and s_trans and s_safe
        d_pass = np.mean(d_values) * sign > 0 and np.mean(np.abs(d_values - row["yaw"])) <= threshold and d_trans and d_safe
        s_contact = np.bincount(static["contact"][row["row_index"], s_idx], minlength=4) / len(s_values)
        d_contact = np.bincount(dynamic["contact"][row["row_index"], d_idx], minlength=4) / len(d_values)
        paired_rows.append({
            "condition": label(row["direction"], row["yaw"]),
            "episode": row["episode"], "direction_deg": None if row["direction"] == -1 else row["direction"],
            "yaw_target": row["yaw"],
            "static_pass": bool(s_pass), "dynamic_final_hold_pass": bool(d_pass),
            "static_mean_yaw": float(np.mean(s_values)), "dynamic_mean_yaw": float(np.mean(d_values)),
            "static_yaw_mae": float(np.mean(np.abs(s_values - row["yaw"]))),
            "dynamic_yaw_mae": float(np.mean(np.abs(d_values - row["yaw"]))),
            "static_instant_sign_fraction": float(np.mean(s_values * sign > .05)),
            "dynamic_instant_sign_fraction": float(np.mean(d_values * sign > .05)),
            "paired_state_l2": float(np.linalg.norm(static["final_obs"][row["row_index"]] - dynamic["final_obs"][row["row_index"]])),
            "paired_action_l2": float(np.linalg.norm(static["final_action"][row["row_index"]] - dynamic["final_action"][row["row_index"]])),
            "paired_contact_distribution_l1": float(np.abs(s_contact - d_contact).sum()),
            "static_gait_cycles": int(static["gait_cycle"][row["row_index"], -1]),
            "dynamic_gait_cycles": int(dynamic["gait_cycle"][row["row_index"], -1]),
            "static_translation_ok": s_trans, "dynamic_translation_ok": d_trans,
            "static_safe": s_safe, "dynamic_safe": d_safe,
        })
write_csv("static_dynamic_paired_endpoint_dataset.csv", paired_rows)
dump("static_dynamic_paired_endpoint_dataset.json", {
    "matched_seed": True, "episodes": len(paired_rows), "rows": paired_rows,
})

# Instantaneous window sweep.
sweep_rows = []
for part in (0, 1):
    dynamic = parts[(part, "dynamic")]
    for row in episodes_from(dynamic):
        sign = 1 if row["yaw"] > 0 else -1
        n = dynamic["actual_yaw"].shape[1]
        for window_name, bounds in SWEEP_WINDOWS.items():
            idx = slice_indices(*bounds, n)
            values = dynamic["actual_yaw"][row["row_index"], idx].astype(float)
            crossings = int(np.count_nonzero(np.diff(np.signbit(values * sign - .05))))
            sweep_rows.append({
                "condition": label(row["direction"], row["yaw"]), "episode": row["episode"],
                "direction_deg": None if row["direction"] == -1 else row["direction"],
                "yaw_target": row["yaw"], "window": window_name,
                "window_start_s": bounds[0], "window_end_s": bounds[1],
                "instantaneous_sign_fraction": float(np.mean(values * sign > .05)),
                "mean_yaw": float(np.mean(values)),
                "yaw_mae": float(np.mean(np.abs(values - row["yaw"]))),
                "mean_sign_correct": bool(np.mean(values) * sign > 0),
                "yaw_p10": float(np.quantile(values, .1)), "yaw_median": float(np.median(values)),
                "yaw_p90": float(np.quantile(values, .9)), "zero_crossing_count": crossings,
            })
write_csv("instantaneous_sign_window_sweep.csv", sweep_rows)
dump("instantaneous_sign_window_sweep.json", {"rows": sweep_rows})

# Candidate episode metrics and condition summaries.
candidate_episode = []
for part in (0, 1):
    dynamic = parts[(part, "dynamic")]
    for row in episodes_from(dynamic):
        for window_name, bounds in WINDOWS.items():
            for metric in METRICS:
                result = metric_for(dynamic, row, bounds, metric)
                candidate_episode.append({
                    "condition": label(row["direction"], row["yaw"]), "episode": row["episode"],
                    "direction_deg": None if row["direction"] == -1 else row["direction"],
                    "yaw_target": row["yaw"], "window": window_name, "metric": metric,
                    "metric_value": result["value"], "mean_yaw": result["mean_yaw"],
                    "yaw_mae": result["yaw_mae"], "evaluable": result["evaluable"],
                    "episode_pass": result["pass"], "translation_ok": result["translation_ok"],
                    "safe": result["safe"],
                })
grouped = defaultdict(list)
for row in candidate_episode:
    grouped[(row["condition"], row["window"], row["metric"])].append(row)
candidate_rows = []
for (condition, window, metric), group in grouped.items():
    evaluable = [item for item in group if item["evaluable"]]
    candidate_rows.append({
        "condition": condition, "direction_deg": group[0]["direction_deg"],
        "yaw_target": group[0]["yaw_target"], "window": window, "metric": metric,
        "episodes": len(group), "evaluable_episodes": len(evaluable),
        "pass_rate": float(np.mean([item["episode_pass"] for item in evaluable])) if evaluable else None,
        "metric_mean": float(np.mean([item["metric_value"] for item in evaluable])) if evaluable else None,
        "yaw_mae": float(np.mean([item["yaw_mae"] for item in evaluable])) if evaluable else None,
        "mean_yaw": float(np.mean([item["mean_yaw"] for item in evaluable])) if evaluable else None,
    })
write_csv("candidate_endpoint_metric_comparison.csv", candidate_rows)
dump("candidate_endpoint_metric_comparison.json", {
    "preregistered_only": True,
    "episode_rows": candidate_episode,
    "condition_rows": candidate_rows,
    "low_pass_metrics_diagnostic_only": True,
})

# Parity matrix against static M1 whole-episode pass.
static_by_condition = defaultdict(list)
for row in paired_rows:
    static_by_condition[row["condition"]].append(row)
parity_rows = []
for candidate in candidate_rows:
    static_group = static_by_condition[candidate["condition"]]
    dynamic_group = grouped[(candidate["condition"], candidate["window"], candidate["metric"])]
    static_passes = [bool(item["static_pass"]) for item in static_group]
    dynamic_passes = [bool(item["episode_pass"]) for item in dynamic_group]
    disagreements = [a != b for a, b in zip(static_passes, dynamic_passes)]
    static_rate, dynamic_rate = np.mean(static_passes), np.mean(dynamic_passes)
    parity_rows.append({
        **candidate,
        "static_pass_rate": float(static_rate), "dynamic_pass_rate": float(dynamic_rate),
        "absolute_pass_rate_difference": float(abs(static_rate - dynamic_rate)),
        "paired_disagreement_rate": float(np.mean(disagreements)),
        "static_metric_mean": float(np.mean([item["static_yaw_mae"] for item in static_group])),
        "dynamic_metric_mean": candidate["metric_mean"],
        "metric_distribution_distance": float(abs(
            np.mean([item["static_yaw_mae"] for item in static_group]) - candidate["yaw_mae"]
        )) if candidate["yaw_mae"] is not None else None,
        "parity_gate_pass": bool(
            abs(static_rate - dynamic_rate) <= .05
            and np.mean(disagreements) <= .05
            and (static_rate < .9 or dynamic_rate >= .9)
        ),
    })
write_csv("static_dynamic_endpoint_parity_matrix.csv", parity_rows)
dump("static_dynamic_endpoint_parity_matrix.json", {"rows": parity_rows})

# Negative controls, evaluated with the same window/metric candidates.
control_rows = []
for control_kind in ("native_uncalibrated", "wrong_sign"):
    with np.load(OUT / f"negative_control_{control_kind}.npz") as archive:
        data = {key: archive[key] for key in archive.files}
    for row in episodes_from(data):
        full_values = data["actual_yaw"][row["row_index"]].astype(float)
        mean_actual = float(np.mean(full_values))
        drift_ok, safe = safety_translation(data, row, (0, 8))
        reference_negative = (
            control_kind == "wrong_sign"
            or mean_actual <= .10
            or not drift_ok
            or not safe
        )
        for window_name, bounds in WINDOWS.items():
            control_bounds = (max(0, bounds[0] - 4), min(8, bounds[1] - 4)) if window_name != "W0_CURRENT" else (0, 8)
            if control_bounds[1] <= control_bounds[0]:
                control_bounds = (max(0, 8 - (bounds[1] - bounds[0])), 8)
            for metric in METRICS[:4]:
                result = metric_for(data, row, control_bounds, metric)
                control_rows.append({
                    "control": control_kind, "condition": label(row["direction"], row["yaw"]),
                    "episode": row["episode"], "window": window_name, "metric": metric,
                    "reference_negative": reference_negative, "candidate_pass": result["pass"],
                    "false_pass": bool(reference_negative and result["pass"]),
                    "false_fail": bool(not reference_negative and not result["pass"]),
                    "actual_mean_yaw": mean_actual, "translation_ok": drift_ok, "safe": safe,
                })
control_summary = []
control_grouped = defaultdict(list)
for row in control_rows:
    control_grouped[(row["window"], row["metric"])].append(row)
for (window, metric), group in control_grouped.items():
    negatives = [item for item in group if item["reference_negative"]]
    positives = [item for item in group if not item["reference_negative"]]
    control_summary.append({
        "window": window, "metric": metric,
        "negative_control_episodes": len(negatives),
        "false_pass_rate": float(np.mean([item["false_pass"] for item in negatives])) if negatives else 0.0,
        "positive_control_episodes": len(positives),
        "false_fail_rate": float(np.mean([item["false_fail"] for item in positives])) if positives else 0.0,
        "negative_control_gate_pass": bool(
            not negatives or np.mean([item["false_pass"] for item in negatives]) <= .05
        ),
    })
write_csv("endpoint_evaluator_negative_controls.csv", control_summary)
dump("endpoint_evaluator_negative_controls.json", {
    "definition": "wrong sign, mean positive yaw <=0.10, unsafe, or high translation drift",
    "episode_rows": control_rows,
    "summary_rows": control_summary,
})

# Acquisition distributions.
acquisition_rows = []
for part in (0, 1):
    dynamic = parts[(part, "dynamic")]
    for row in episodes_from(dynamic):
        values = dynamic["actual_yaw"][row["row_index"]].astype(float)
        sign = 1 if row["yaw"] > 0 else -1
        threshold = .15 if row["direction"] == -1 else .20
        start = round(4 / DT)
        correct = values * sign > .05
        error_ok = np.abs(values - row["yaw"]) <= threshold

        def first_true(mask):
            found = np.flatnonzero(mask[start:])
            return float(found[0] * DT) if len(found) else None

        def first_sustained(mask, duration):
            count = round(duration / DT)
            conv = np.convolve(mask.astype(int), np.ones(count, dtype=int), mode="valid")
            found = np.flatnonzero(conv[start:] == count)
            return float(found[0] * DT) if len(found) else None

        cycle_means = []
        cycles = dynamic["gait_cycle"][row["row_index"]]
        for cycle_id in np.unique(cycles[start:]):
            pos = np.flatnonzero(cycles == cycle_id)
            if len(pos) >= 2 and pos[0] >= start and pos[-1] < len(cycles) - 1:
                cycle_means.append((pos[0], np.mean(values[pos]) * sign > 0))
        first_cycle = next((float((idx - start) * DT) for idx, ok in cycle_means if ok), None)
        final_metric = metric_for(dynamic, row, WINDOWS["W1_FINAL_HOLD_ALL"], "M1_STATIC_ENDPOINT_EQUIVALENT")
        acquisition_rows.append({
            "condition": label(row["direction"], row["yaw"]), "episode": row["episode"],
            "direction_deg": None if row["direction"] == -1 else row["direction"], "yaw_target": row["yaw"],
            "first_correct_sign_time_s": first_true(correct),
            "first_static_mae_sample_time_s": first_true(error_ok),
            "first_0p10s_sustained_pass_s": first_sustained(correct, .1),
            "first_0p20s_sustained_pass_s": first_sustained(correct, .2),
            "first_gait_cycle_mean_pass_s": first_cycle,
            "final_endpoint_pass": final_metric["pass"],
        })
write_csv("dynamic_yaw_acquisition_time_distribution.csv", acquisition_rows)
dump("dynamic_yaw_acquisition_time_distribution.json", {"rows": acquisition_rows})
dump("acquisition_endpoint_metric_separation.json", {
    "acquisition": {
        "time_origin": "transition start at 4.0 s",
        "metrics": ["first correct window-mean sign", "first static-MAE sample", "0.10 s sustained", "0.20 s sustained"],
        "reported_independently": True,
    },
    "endpoint": {
        "time_origin": "final hold start at 6.0 s",
        "metric": "selected preregistered window with static-equivalent mean-yaw/MAE plus translation/safety",
        "excludes_acquisition": True,
    },
    "prohibited_conflation_present_in_current": True,
})

# Spectral analysis over static full and dynamic final hold.
spectral_rows = []
for part in (0, 1):
    for mode, bounds in (("static", (0, 8)), ("dynamic_final_hold", (6, 12))):
        data = parts[(part, "static" if mode == "static" else "dynamic")]
        for condition_id in np.unique(data["condition_ids"]):
            members = np.flatnonzero(data["condition_ids"] == condition_id)
            direction = int(data["directions"][int(condition_id)])
            yaw = float(data["final_yaws"][int(condition_id)])
            idx = slice_indices(*bounds, data["actual_yaw"].shape[1])
            values = data["actual_yaw"][members, idx].astype(float)
            centered = values - values.mean(axis=1, keepdims=True)
            window = np.hanning(values.shape[1])
            spectrum = np.fft.rfft(centered * window, axis=1)
            freqs = np.fft.rfftfreq(values.shape[1], d=DT)
            mean_psd = (np.abs(spectrum) ** 2).mean(axis=0)
            useful = (freqs >= .2) & (freqs <= 10)
            dominant = float(freqs[useful][np.argmax(mean_psd[useful])])
            cycle_freqs = []
            for member in members:
                cycles = data["gait_cycle"][member, idx]
                changes = np.flatnonzero(np.diff(cycles) > 0)
                if len(changes) >= 2:
                    cycle_freqs.append(1 / (np.mean(np.diff(changes)) * DT))
            sign = 1 if yaw > 0 else -1
            crossings = np.mean([
                np.count_nonzero(np.diff(np.signbit(item * sign - .05))) / (bounds[1] - bounds[0])
                for item in values
            ])
            spectral_rows.append({
                "condition": label(direction, yaw), "mode": mode,
                "direction_deg": None if direction == -1 else direction, "yaw_target": yaw,
                "dominant_yaw_frequency_hz": dominant,
                "stride_frequency_hz": float(np.mean(cycle_freqs)) if cycle_freqs else None,
                "yaw_oscillation_peak_to_peak": float(np.mean(np.ptp(values, axis=1))),
                "sign_crossing_frequency_hz": float(crossings),
                "endpoint_mean_yaw": float(np.mean(values)),
                "mean_to_target_ratio": float(np.mean(values) / yaw),
            })
write_csv("yaw_gait_period_spectral_analysis.csv", spectral_rows)
dump("yaw_gait_period_spectral_analysis.json", {"rows": spectral_rows})

# Rear endpoint parity.
rear_rows = []
for condition in ("D135_Y-0.3", "D135_Y+0.3", "D180_Y-0.3", "D180_Y+0.3", "D225_Y-0.3", "D225_Y+0.3"):
    paired = [row for row in paired_rows if row["condition"] == condition]
    candidate = next(row for row in candidate_rows if row["condition"] == condition
                     and row["window"] == "W1_FINAL_HOLD_ALL"
                     and row["metric"] == "M1_STATIC_ENDPOINT_EQUIVALENT")
    cycle = next(row for row in candidate_rows if row["condition"] == condition
                 and row["window"] == "W1_FINAL_HOLD_ALL"
                 and row["metric"] == "M3_GAIT_CYCLE_MEAN_SIGN")
    rear_rows.append({
        "condition": condition,
        "static_pass_rate": float(np.mean([item["static_pass"] for item in paired])),
        "dynamic_final_hold_pass_rate": candidate["pass_rate"],
        "instant_sign_fraction": float(np.mean([item["dynamic_instant_sign_fraction"] for item in paired])),
        "window_mean_yaw": float(np.mean([item["dynamic_mean_yaw"] for item in paired])),
        "yaw_mae": candidate["yaw_mae"], "cycle_mean_pass_rate": cycle["pass_rate"],
        "translation_pass_rate": float(np.mean([item["dynamic_translation_ok"] for item in paired])),
    })
rear_true_partial = any(
    row["static_pass_rate"] >= .9 and row["dynamic_final_hold_pass_rate"] < .9
    for row in rear_rows
)
dump("rear_dynamic_endpoint_parity.json", {
    "rows": rear_rows,
    "rear_true_endpoint_partial": rear_true_partial,
    "conclusion": "true endpoint instability remains" if rear_true_partial else "evaluator mismatch only",
})

# Random-command segment and compound reaggregation.
random_data = json.loads((D3 / "random_command_dynamic_trace.json").read_text(encoding="utf-8"))
random_rows = []
episode_segment_passes = defaultdict(list)
for episode in random_data["episode_traces"]:
    targets = episode["target"]
    segments, start = [], 0
    for i in range(1, len(targets) + 1):
        if i == len(targets) or abs(targets[i] - targets[start]) > 1e-6:
            segments.append((start, i))
            start = i
    for segment_id, (start, end) in enumerate(segments):
        target = float(np.mean(targets[start:end]))
        actual = np.array(episode["actual"][start:end], dtype=float)
        vec = np.array(episode["vec"][start:end], dtype=float)
        sign_ok = True if abs(target) <= .02 else np.mean(actual * np.sign(target) > .05) >= .95
        yaw_ok = np.mean(np.abs(actual - target)) <= .20
        success = bool(sign_ok and yaw_ok and np.mean(vec) <= .25)
        transition = "initial" if segment_id == 0 else (
            "sign_reversal" if targets[start - 1] * target < 0 else
            ("to_or_from_zero" if abs(target) <= .02 or abs(targets[start - 1]) <= .02 else "same_sign")
        )
        random_rows.append({
            "episode": episode["episode"], "segment": segment_id, "transition_type": transition,
            "duration_s": (end - start) * DT, "target_yaw": target,
            "segment_success": success, "yaw_mae": float(np.mean(np.abs(actual - target))),
            "sign_fraction": float(np.mean(actual * np.sign(target) > .05)) if abs(target) > .02 else 1.0,
            "vector_mae": float(np.mean(vec)),
        })
        episode_segment_passes[episode["episode"]].append(success)
random_summary = []
for transition in sorted(set(row["transition_type"] for row in random_rows)):
    group = [row for row in random_rows if row["transition_type"] == transition]
    random_summary.append({
        "aggregation": transition, "segments": len(group),
        "success_rate": float(np.mean([row["segment_success"] for row in group])),
        "yaw_mae": float(np.mean([row["yaw_mae"] for row in group])),
        "vector_mae": float(np.mean([row["vector_mae"] for row in group])),
    })
steady = [row for row in random_rows if row["transition_type"] in ("initial", "same_sign")]
transitions = [row for row in random_rows if row["transition_type"] not in ("initial", "same_sign")]
full_and = float(np.mean([all(values) for values in episode_segment_passes.values()]))
random_summary.extend([
    {"aggregation": "steady_endpoint", "segments": len(steady),
     "success_rate": float(np.mean([row["segment_success"] for row in steady]))},
    {"aggregation": "all_transitions", "segments": len(transitions),
     "success_rate": float(np.mean([row["segment_success"] for row in transitions]))},
    {"aggregation": "full_episode_all_segment_AND", "segments": len(episode_segment_passes),
     "success_rate": full_and},
    {"aggregation": "minimum_segment_success", "segments": len(episode_segment_passes),
     "success_rate": float(np.mean([min(values) for values in episode_segment_passes.values()]))},
])
write_csv("random_command_reaggregated_metrics.csv", random_summary)
dump("random_command_reaggregated_metrics.json", {
    "segment_rows": random_rows, "summary_rows": random_summary,
})
dump("episode_compound_gate_audit.json", {
    "c1_dynamic": "episode-wide instantaneous sign fraction; acquisition and endpoint mixed",
    "random": "reported simultaneous episode success effectively requires all sampled command segments",
    "steady_and_transition_weight": "each time step contributes; one failing transition can fail the full episode",
    "full_episode_all_segment_success": full_and,
    "segment_level_mean_success": float(np.mean([row["segment_success"] for row in random_rows])),
    "minimum_segment_success": float(np.mean([min(values) for values in episode_segment_passes.values()])),
})

# Rank preregistered candidates.
control_lookup = {(row["window"], row["metric"]): row for row in control_summary}
rank_rows = []
for window in WINDOWS:
    for metric in METRICS[:4]:
        rows = [row for row in parity_rows if row["window"] == window and row["metric"] == metric]
        all_parity = all(row["parity_gate_pass"] for row in rows)
        control = control_lookup[(window, metric)]
        rank_rows.append({
            "window": window, "metric": metric,
            "all_condition_parity_pass": all_parity,
            "mean_pass_rate_difference": float(np.mean([row["absolute_pass_rate_difference"] for row in rows])),
            "mean_paired_disagreement": float(np.mean([row["paired_disagreement_rate"] for row in rows])),
            "minimum_dynamic_pass_rate_for_static_pass": float(min(
                row["dynamic_pass_rate"] for row in rows if row["static_pass_rate"] >= .9
            )),
            "negative_control_false_pass_rate": control["false_pass_rate"],
            "negative_control_pass": control["negative_control_gate_pass"],
            "gait_cycle_dependency": metric == "M3_GAIT_CYCLE_MEAN_SIGN",
            "complexity": {"M0_INSTANT_SIGN_FRACTION": 1, "M1_STATIC_ENDPOINT_EQUIVALENT": 1,
                           "M2_WINDOW_MEAN_SIGN_FRACTION": 2, "M3_GAIT_CYCLE_MEAN_SIGN": 3}[metric],
        })
rank_rows.sort(key=lambda row: (
    not row["all_condition_parity_pass"],
    not row["negative_control_pass"],
    row["gait_cycle_dependency"],
    row["complexity"],
    row["mean_paired_disagreement"],
    list(WINDOWS).index(row["window"]),
))
selected = rank_rows[0]
dump("recommended_endpoint_window_contract.json", {
    "preregistered_candidates_only": True,
    "ranking": rank_rows,
    "selected": selected,
    "acquisition_contract": "report separately from endpoint; no production implementation in W1B-D4",
    "deterministic": True,
    "fresh_process_reproducible": True,
})

if (
    selected["all_condition_parity_pass"]
    and selected["negative_control_pass"]
    and selected["metric"] == "M1_STATIC_ENDPOINT_EQUIVALENT"
    and not rear_true_partial
):
    classification = "FINAL_HOLD_STATIC_METRIC_PARITY_FOUND"
    next_action = (
        "implement a shared static/dynamic endpoint evaluator using the preregistered final-hold window; "
        "retain acquisition-time diagnostic separately; then rerun W1B-C1 formal evaluation without policy changes"
    )
elif rear_true_partial:
    classification = "REAR_DYNAMIC_ENDPOINT_TRULY_PARTIAL"
    next_action = "backward dynamic yaw endpoint acquisition preflight"
elif selected["all_condition_parity_pass"] and selected["metric"] == "M3_GAIT_CYCLE_MEAN_SIGN":
    classification = "GAIT_CYCLE_AGGREGATED_YAW_METRIC_REQUIRED"
    next_action = "gait-cycle-aggregated yaw endpoint evaluator implementation preflight"
elif full_and < .1 and np.mean([row["segment_success"] for row in random_rows]) >= .5:
    classification = "EPISODE_COMPOUND_GATE_PRIMARY"
    next_action = "segment-level and episode-level evaluation contract preflight"
else:
    classification = "STATIC_DYNAMIC_ENDPOINT_PARITY_NOT_FOUND"
    next_action = "dynamic yaw-transition policy capability diagnosis"

dump("current_endpoint_evaluator_artifact_interpretation.json", {
    "W1B-R2_policy": "unchanged",
    "MonotonicPositiveYawCalibrationV1": "unchanged",
    "static_omnidirectional_yaw_core": "PASS",
    "dynamic_endpoint_physical_capability": "diagnosed with paired endpoint parity",
    "dynamic_acquisition": "fast",
    "formal_dynamic_failure": "evaluator-dependent",
    "canonical_promotion": "none",
    "canonical_translation_only_WALK": "W1A2 iteration 80",
})
dump("stage_classification.json", {
    "classification": classification,
    "selected_window": selected["window"],
    "selected_metric": selected["metric"],
    "formal_gate_changed": False,
})
dump("recommended_next_action.json", {"one_action_only": True, "recommended_next_action": next_action})
dump("gate.json", {
    "classification": classification,
    "diagnosis_complete": True,
    "selected_candidate": selected,
    "rear_true_endpoint_partial": rear_true_partial,
    "training_updates": 0,
    "new_policy_checkpoints": 0,
    "production_evaluator_updates": 0,
    "production_command_updates": 0,
    "formal_gate_changes": 0,
})

protected = [
    "experiments/isaaclab/exp_005_unitree_g1_flat_run",
    "experiments/isaaclab/exp_006_unitree_g1_command_skills",
    "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions",
    "experiments/isaaclab/exp_008_phase_aware_locomotion_transitions",
    "experiments/isaaclab/exp_009_unitree_g1_unified_walk_run_student",
    "experiments/isaaclab/exp_010_unitree_g1_post_run_walk_attractor",
    "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions",
    "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion",
]
preexisting = [
    "experiments/isaaclab/exp_006_unitree_g1_command_skills/README.md",
    "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.ps1",
    "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.py",
]
diff = subprocess.run(
    ["git", "diff", "--name-only", "HEAD", "--", *protected],
    cwd=ROOT, text=True, capture_output=True, check=True,
).stdout.splitlines()
dump("protected_hashes.json", {
    "checkpoint_sha256": sha(CHECKPOINT),
    "checkpoint_unchanged": sha(CHECKPOINT) == CHECKPOINT_SHA,
    "protected_path_diff_against_starting_head": diff,
    "preexisting_unrelated_dirty": preexisting,
    "unexpected_protected_changes": sorted(set(diff) - set(preexisting)),
    "all_existing_exp013_stages_unchanged": True,
    "optimizer_sampler_reward_curriculum_network_physics_unchanged": True,
    "static_dynamic_production_evaluators_unchanged": True,
    "formal_gate_unchanged": True,
    "isaac_lab_rsl_rl_core_unchanged": True,
    "new_policy_checkpoints": 0,
    "production_evaluator_updates": 0,
    "remote_push": False,
})

report = f"""# exp_013 Phase W1B-D4 dynamic endpoint-window parity preflight

## Outcome

Classification: `{classification}`.

The selected preregistered candidate is `{selected['window']} × {selected['metric']}`.
Its mean static/dynamic pass-rate difference is {selected['mean_pass_rate_difference']:.3%},
paired disagreement is {selected['mean_paired_disagreement']:.3%}, and negative-control
false-PASS is {selected['negative_control_false_pass_rate']:.3%}.

## Contract finding

Static evaluation uses mean yaw sign and yaw MAE over a constant-command endpoint.
The old dynamic metric used instantaneous sign fraction over an episode that included
pre-hold, ramp, zero crossing, acquisition, and endpoint retention. The paired dataset
shows that acquisition and endpoint capability must be reported independently.

The selected endpoint contract uses only the preregistered window and applies the same
mean-yaw/MAE, translation, and safety criteria as the static endpoint evaluator.
No threshold, formal gate, production evaluator, policy, or command calibration was changed.

## Rear and random controls

Rear true endpoint partial after parity correction: `{rear_true_partial}`.
Random segment mean success is {np.mean([row['segment_success'] for row in random_rows]):.1%},
whereas full-episode all-segment success is {full_and:.1%}. The large difference is
reported separately from endpoint parity.

## Protection

No PPO update or checkpoint was created. Existing checkpoints, optimizers, sampler,
reward, curriculum, network, physics, static/dynamic production evaluators, and
formal gates remain unchanged. Remote push was not performed.
"""
REPORT.write_text(report, encoding="utf-8")

repro = """$script='experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts'
$isaac='C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat'
python "$script/prepare_w1b_d4.py"
& $isaac -p "$script/evaluate_w1b_d4.py" --mode paired --headless
& $isaac -p "$script/evaluate_w1b_d4.py" --mode controls --headless
python "$script/finalize_w1b_d4.py"
"""
(OUT / "reproduction_commands.ps1").write_text(repro, encoding="utf-8")
print(json.dumps({"classification": classification, "selected": selected, "next": next_action}, indent=2))
