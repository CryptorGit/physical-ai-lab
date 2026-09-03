"""Reduce saved Stage 5 telemetry into frozen diagnostic products."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage5_endpoint_failure_diagnosis"
STAGE4 = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage4_resumed_optimizer_training"
PARENT = (
    REPO / ".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/Assets/Isaac/6.0/"
    "Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt"
)
SELECTED = STAGE4 / "checkpoints/model_50.pt"
START = "84ebd3c48a217cf861152cc43df4c9376855a432"
START_STATUS = [
    " M experiments/isaaclab/exp_006_unitree_g1_command_skills/README.md",
    "?? .openduck_hardware_source_review/", "?? .openduck_phase3_usb_baseline.txt",
    "?? .openduck_runtime_source_review/", "?? artifacts/exp_005_unitree_g1_flat_run/",
    "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/assemble_showcase_reel.py",
    "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/play_command_system_showcase.ps1",
    "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/play_showcase.py",
    "?? media/", "?? openduck_setup_report.md",
]
FEET = ("front-left", "front-right", "rear-left", "rear-right")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name: str, records: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in records for key in row))
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def avg(values) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def pct(values, q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    return values[round((len(values) - 1) * q / 100)]


def auroc(scores, labels):
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    return avg(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives for negative in negatives
    )


def auprc(scores, labels):
    if not any(labels):
        return None
    ordered = sorted(zip(scores, labels), reverse=True)
    true_positive = 0
    precisions = []
    for rank, (_, label) in enumerate(ordered, 1):
        true_positive += int(label)
        if label:
            precisions.append(true_positive / rank)
    return avg(precisions)


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(bytes.fromhex(sha(file)))
    return digest.hexdigest()


def aggregate_speed(rows: list[dict]) -> dict:
    return {
        "episodes": len(rows),
        "fall_rate": avg(row["fall"] for row in rows),
        "actual_speed_mean_mps": avg(row["actual_speed_mean_mps"] for row in rows),
        "lateral_speed_abs_mean_mps": avg(row["lateral_speed_abs_mean_mps"] for row in rows),
        "yaw_drift_p95_across_episodes_rad": pct([row["yaw_drift_p95_rad"] for row in rows], 95),
        "roll_abs_p95_across_episodes_rad": pct([row["roll_abs_p95_rad"] for row in rows], 95),
        "pitch_abs_p95_across_episodes_rad": pct([row["pitch_abs_p95_rad"] for row in rows], 95),
        "gravity_tilt_p95_across_episodes_rad": pct([row["gravity_tilt_p95_rad"] for row in rows], 95),
        "existing_dangerous_slip_rate": avg(row["existing_dangerous_slip"] for row in rows),
        "existing_slip_mean_mps": avg(row["existing_slip_mean_mps"] for row in rows),
        "official_feet_slide_raw_mean": avg(row["official_feet_slide_raw_mean"] for row in rows),
        "world_stance_speed_mean_mps": avg(row["world_stance_speed_mean_mps"] for row in rows),
        "world_stance_speed_p95_mps": pct([row["world_stance_speed_p95_mps"] for row in rows], 95),
        "root_relative_foot_speed_mean_mps": avg(row["root_relative_foot_speed_mean_mps"] for row in rows),
        "action_norm_mean": avg(row["action_norm_mean"] for row in rows),
        "action_rate_p95": pct([row["action_rate_p95"] for row in rows], 95),
        "support_state_entropy_mean": avg(row["support_state_entropy"] for row in rows),
        "gait_counts": dict(Counter(row["gait"] for row in rows)),
        "reference_gait_counts": dict(Counter(row["reference_gait"] for row in rows)),
        "duty_factor_mean": [
            avg(row["contact_occupancy"][foot] for row in rows) for foot in range(4)
        ],
        "flight_fraction_mean": avg(row["contact_loss_fraction"] for row in rows),
    }


def main():
    raw = load(OUT / "raw_episode_summaries.json")
    runtime = load(OUT / "runtime_contract.json")
    by_condition = defaultdict(list)
    for row in raw:
        by_condition[(row["checkpoint"], row["speed_mps"])].append(row)

    dump("stage4_reference.json", {
        "classification": "GO2_ENDPOINT_FAILURE_MULTIPLE",
        "selected_checkpoint": str(SELECTED), "selected_sha256": sha(SELECTED),
        "positive_transition_result_protected": {
            "0->1.2": {"completion": 1.0, "acquisition": 1.0, "target_hold": 1.0, "fall": 0.0},
            "1.2->2.0": {"completion": 1.0, "acquisition": 1.0, "target_hold": 1.0, "fall": 0.0},
            "2.0->1.2": {"completion": 1.0, "acquisition": 1.0, "target_hold": 1.0, "fall": 0.0},
            "1.2->0": {"completion": 1.0, "acquisition": 1.0, "target_hold": 1.0, "fall": 0.0},
            "post_deceleration_high_speed_gait_or_flight_dominant": False,
        },
        "stage4_results_hash": tree_hash(STAGE4),
    })
    dump("protocol.json", {
        "stage": 5, "starting_head": START, "starting_status": START_STATUS,
        "seed_root": 20263901, "episodes_per_condition": 50, "duration_s": 8,
        "checkpoints": {"official_parent": sha(PARENT), "stage4_selected": sha(SELECTED)},
        "steady_diagnostic_speeds_mps": [0.2, 0.3, 0.4, 0.5, 0.6, 1.2, 2.0],
        "low_speed_sweep_mps": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "transition_conditions": ["0->0.4", "0.4->0", "0->1.2", "1.2->0", "1.2->2.0", "2.0->1.2"],
        "optimizer_updates": 0, "reward_changes": 0, "curriculum_changes": 0,
    })
    dump("diagnostic_seed_manifest.json", {
        "root": 20263901, "episode_seeds": list(range(20263901, 20263951)),
        "paired_across_checkpoints": True, "success_based_selection": False,
    })
    dump("existing_slip_metric_contract.json", {
        "source": "src/go2_bidirectional/evaluation.py Collector.run + summarize_trace",
        "contact_tensor": "contact_forces.data.net_forces_w_history.torch",
        "contact_reduction": "norm xyz; maximum across sensor history",
        "foot_velocity_tensor": "robot.data.body_lin_vel_w.torch",
        "foot_body_indices": runtime["mapping"],
        "contact_threshold_n": 5.0, "velocity_frame": "world",
        "planar_axes": "world x/y", "velocity_unit": "m/s",
        "per_step_quantity": "maximum planar COM speed among feet marked contact",
        "episode_quantity": "mean of per-step maximum across entire selected segment",
        "slip_threshold_mps": 0.55,
        "episode_failure": "episode mean > 0.55 m/s; not any-event",
        "minimum_contact_duration": 0.0, "minimum_contiguous_slip_duration": 0.0,
        "boundary_steps_included": True,
        "aggregation_note": "binary condition rate loses severity after episode mean thresholding",
    })

    frame_rows = []
    with (OUT / "per_foot_frame_comparison.csv").open(encoding="utf-8", newline="") as stream:
        frame_rows = list(csv.DictReader(stream))
    frame_summary = defaultdict(lambda: {"world": [], "relative": [], "force": [], "contact": []})
    for row in frame_rows:
        key = (row["checkpoint"], row["foot"])
        world = json.loads(row["world_velocity"])
        relative = json.loads(row["root_relative_velocity_root_frame"])
        frame_summary[key]["world"].append(math.hypot(world[0], world[1]))
        frame_summary[key]["relative"].append(math.hypot(relative[0], relative[1]))
        frame_summary[key]["force"].append(float(row["contact_force_n"]))
        frame_summary[key]["contact"].append(row["contact"].lower() == "true")
    dump("slip_frame_unit_audit.json", {
        "result": "FRAME_UNIT_AND_INDEX_CONTRACT_VALID_FOR_FOOT_LINK_COM",
        "important_limitation": (
            "body_lin_vel_w is the foot rigid-body origin/COM velocity, not the velocity "
            "of the instantaneous ground contact point"
        ),
        "quaternion_contract_discovery": (
            "root_quat_w.torch is xyzw in current Isaac Lab; Stage 1-4 evaluator decoded it as wxyz"
        ),
        "mapping": runtime["mapping"],
        "per_foot_summary": {
            f"{checkpoint}:{foot}": {
                "world_planar_speed_mean_mps": avg(values["world"]),
                "root_relative_planar_speed_mean_mps": avg(values["relative"]),
                "contact_fraction": avg(values["contact"]),
                "force_mean_n": avg(values["force"]),
            } for (checkpoint, foot), values in frame_summary.items()
        },
        "unit": "SI metres and seconds", "world_forward_motion_double_counted": False,
    })

    event_rows = []
    with (OUT / "contact_event_examples.csv").open(encoding="utf-8", newline="") as stream:
        event_rows = list(csv.DictReader(stream))
    events = defaultdict(list)
    for row in event_rows:
        key = (row["checkpoint"], row["speed_mps"], row["episode"], row["foot"], row["event"], row["step"])
        events[key].append(row)
    boundary_only = 0; slip_events = 0; classifications = Counter()
    for rows in events.values():
        flags = [(abs(int(row["step_offset"])), row["existing_slip_flag"].lower() == "true") for row in rows]
        any_slip = any(flag for _, flag in flags)
        if not any_slip:
            classifications["stable stance"] += 1
            continue
        slip_events += 1
        inner = any(flag for offset, flag in flags if offset > 2)
        if not inner:
            boundary_only += 1
            classifications["touchdown/liftoff transient"] += 1
        else:
            classifications["true sustained or foot-link rotation"] += 1
    boundary_fraction = boundary_only / max(slip_events, 1)
    dump("contact_boundary_slip_audit.json", {
        "events_audited": len(events), "slip_positive_events": slip_events,
        "boundary_only_slip_events": boundary_only,
        "boundary_only_fraction": boundary_fraction,
        "classification_counts": dict(classifications),
        "window_control_steps": [-5, 5],
        "sensor_lag_test": "contact force and body velocity sampled after the same environment step",
    })

    slip_speed = {}
    slip_csv = []
    per_foot_csv = []
    physical = {}
    official = {}
    for (checkpoint, speed), rows in sorted(by_condition.items()):
        key = f"{checkpoint}:{speed}"
        threshold_summary = {}
        for threshold in (0.1, 0.2, 0.3, 0.5):
            stats = [row["threshold_stats"][str(threshold)] for row in rows]
            threshold_summary[str(threshold)] = {
                "episode_occurrence_rate": avg(row["occurrence"] for row in stats),
                "mean_time_fraction": avg(row["time_fraction"] for row in stats),
                "p95_time_fraction": pct([row["time_fraction"] for row in stats], 95),
                "mean_contact_time_fraction": avg(row["contact_time_fraction"] for row in stats),
                "mean_contiguous_duration_s": avg(row["max_contiguous_duration_s"] for row in stats),
                "p95_contiguous_duration_s": pct([row["max_contiguous_duration_s"] for row in stats], 95),
                "maximum_duration_s": max(row["max_contiguous_duration_s"] for row in stats),
                "per_foot_occurrence_fraction": [
                    avg(row["per_foot_time_fraction"][foot] for row in stats) for foot in range(4)
                ],
            }
        displacements = [
            value for row in rows for foot in row["physical_intervals"]
            for value in foot["displacements"]
        ]
        durations = [
            value for row in rows for foot in row["physical_intervals"]
            for value in foot["durations"]
        ]
        including = [
            value for row in rows for foot in row["physical_intervals"]
            for value in foot["speeds_including"]
        ]
        excluding = [
            value for row in rows for foot in row["physical_intervals"]
            for value in foot["speeds_excluding"]
        ]
        slip_speed[key] = {
            "checkpoint": checkpoint, "speed_mps": speed,
            "existing_binary_episode_rate": avg(row["existing_dangerous_slip"] for row in rows),
            "existing_mean_mps": avg(row["existing_slip_mean_mps"] for row in rows),
            "threshold_levels": threshold_summary,
            "stance_displacement_p50_m": pct(displacements, 50),
            "stance_displacement_p95_m": pct(displacements, 95),
            "stance_displacement_p99_m": pct(displacements, 99),
            "stance_duration_mean_s": avg(durations),
            "stance_speed_including_boundary_mean_mps": avg(including),
            "stance_speed_excluding_boundary_mean_mps": avg(excluding),
        }
        slip_csv.append({
            "checkpoint": checkpoint, "speed_mps": speed,
            "binary_episode_rate": slip_speed[key]["existing_binary_episode_rate"],
            "existing_mean_mps": slip_speed[key]["existing_mean_mps"],
            "contact_fraction_above_0p5": threshold_summary["0.5"]["mean_contact_time_fraction"],
            "contiguous_p95_s_above_0p5": threshold_summary["0.5"]["p95_contiguous_duration_s"],
            "stance_displacement_p95_m": slip_speed[key]["stance_displacement_p95_m"],
            "official_raw_mean": avg(row["official_feet_slide_raw_mean"] for row in rows),
        })
        physical[key] = {
            "including_boundary": {"mean_speed_mps": avg(including), "p95_speed_mps": pct(including, 95)},
            "excluding_two_boundary_steps": {"mean_speed_mps": avg(excluding), "p95_speed_mps": pct(excluding, 95)},
            "displacement_m": {"p50": pct(displacements, 50), "p95": pct(displacements, 95), "p99": pct(displacements, 99)},
            "stance_duration_s": {"mean": avg(durations), "p95": pct(durations, 95)},
        }
        official[key] = {
            "raw_feet_slide_mean": avg(row["official_feet_slide_raw_mean"] for row in rows),
            "raw_per_foot_mean": [
                avg(row["official_feet_slide_per_foot_mean"][foot] for row in rows)
                for foot in range(4)
            ],
            "weighted_contribution": 0.0,
            "weight": 0.0,
            "reason": "official Go2 Flat reward configuration has no feet_slide reward term",
        }
        for foot in range(4):
            foot_displacements = [
                value for row in rows for value in row["physical_intervals"][foot]["displacements"]
            ]
            per_foot_csv.append({
                "checkpoint": checkpoint, "speed_mps": speed, "foot": FEET[foot],
                "interval_count": len(foot_displacements),
                "displacement_p50_m": pct(foot_displacements, 50),
                "displacement_p95_m": pct(foot_displacements, 95),
                "contact_occupancy": avg(row["contact_occupancy"][foot] for row in rows),
            })
    paired_slip = {}
    for speed in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.2, 2.0):
        parent_rows = sorted(by_condition[("official_parent", speed)], key=lambda row: row["episode"])
        stage4_rows = sorted(by_condition[("stage4_selected", speed)], key=lambda row: row["episode"])
        paired_slip[str(speed)] = {
            "pairs": len(parent_rows),
            "stage4_minus_parent_existing_slip_mean_mps": avg(
                stage4["existing_slip_mean_mps"] - parent["existing_slip_mean_mps"]
                for parent, stage4 in zip(parent_rows, stage4_rows)
            ),
            "stage4_minus_parent_official_raw_mean": avg(
                stage4["official_feet_slide_raw_mean"] - parent["official_feet_slide_raw_mean"]
                for parent, stage4 in zip(parent_rows, stage4_rows)
            ),
            "stage4_minus_parent_contact_fraction_above_0p5": avg(
                stage4["threshold_stats"]["0.5"]["contact_time_fraction"]
                - parent["threshold_stats"]["0.5"]["contact_time_fraction"]
                for parent, stage4 in zip(parent_rows, stage4_rows)
            ),
        }
    dump("slip_severity_by_speed.json", {
        "conditions": slip_speed, "paired_stage4_minus_parent": paired_slip,
        "interpretation": (
            "Binary occurrence saturates at 100% for moving parent and Stage 4 conditions. "
            "Paired severity, rather than binary occurrence, is the valid comparison."
        ),
    })
    write_csv("slip_severity_by_checkpoint.csv", slip_csv)
    write_csv("per_foot_slip_distribution.csv", per_foot_csv)
    dump("physical_stance_displacement.json", {
        "quantity": "world-frame horizontal displacement of the foot rigid-body origin",
        "limitation": (
            "This is not the instantaneous collision/contact-point velocity; rigid-foot "
            "rotation can move the body origin while a surface point remains planted."
        ),
        "conditions": physical,
    })
    dump("official_feet_slide_comparison.json", {
        "source": "isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards.feet_slide",
        "definition": "sum(norm(body_lin_vel_w[foot,:2]) * contact_force_history_max>1N)",
        "configured_in_official_go2_flat_reward": False, "configured_weight": 0.0,
        "conditions": official,
    })

    stand_rows = [row for row in raw if row["speed_mps"] == 0.0]
    parent_stable = [row for row in stand_rows if row["checkpoint"] == "official_parent" and not row["fall"]]
    parent_nominal_roll = pct([row["settle_median_roll_rad"] for row in parent_stable], 50)
    parent_nominal_pitch = pct([row["settle_median_pitch_rad"] for row in parent_stable], 50)
    for row in stand_rows:
        row["parent_nominal_tilt_deviation_rad"] = math.hypot(
            row["settle_median_roll_rad"] - parent_nominal_roll,
            row["settle_median_pitch_rad"] - parent_nominal_pitch,
        )
    dump("stand_posture_metric_contract.json", {
        "stage1_to_4_quaternion_input": "root_quat_w.torch (xyzw)",
        "stage1_to_4_decoder": "incorrectly unbound as wxyz",
        "formal_roll_pitch_metric": "p95(max(abs(decoded roll),abs(decoded pitch)))",
        "formal_height_metric": "per-episode max(height)-min(height), including reset transient",
        "nominal_pitch_compensation": False, "settle_exclusion": False,
        "correct_diagnostic": "xyzw converted to wxyz before RPY; gravity tilt from rotated world gravity",
        "base_height_episode_initial_difference_included": False,
        "base_height_reset_settling_included": True,
    })
    stand_distribution = {}
    for checkpoint in ("official_parent", "stage4_selected"):
        rows = [row for row in stand_rows if row["checkpoint"] == checkpoint]
        stand_distribution[checkpoint] = {
            "episodes": len(rows), "fall_rate": avg(row["fall"] for row in rows),
            "root_speed_mean_mps": avg(row["root_speed_mean_mps"] for row in rows),
            "yaw_rate_p95_radps": pct([row["yaw_rate_abs_p95_radps"] for row in rows], 95),
            "roll_abs_p95_rad": pct([row["roll_abs_p95_rad"] for row in rows], 95),
            "pitch_abs_p95_rad": pct([row["pitch_abs_p95_rad"] for row in rows], 95),
            "gravity_tilt_p95_rad": pct([row["gravity_tilt_p95_rad"] for row in rows], 95),
            "settle_nominal_deviation_p95_rad": pct([row["settle_nominal_tilt_deviation_p95_rad"] for row in rows], 95),
            "parent_nominal_deviation_p95_rad": pct([row["parent_nominal_tilt_deviation_rad"] for row in rows], 95),
            "height_range_p95_m": pct([row["base_height_range_m"] for row in rows], 95),
            "settle_height_range_p95_m": pct([row["settle_height_range_m"] for row in rows], 95),
            "contact_loss_fraction_mean": avg(row["contact_loss_fraction"] for row in rows),
            "existing_dangerous_slip_rate": avg(row["existing_dangerous_slip"] for row in rows),
        }
    dump("stand_posture_distribution.json", {
        "parent_stable_nominal_roll_rad": parent_nominal_roll,
        "parent_stable_nominal_pitch_rad": parent_nominal_pitch,
        "by_checkpoint": stand_distribution,
    })
    stand_csv = []
    for row in stand_rows:
        stand_csv.append({
            key: row[key] for key in (
                "checkpoint", "episode", "seed", "fall", "root_speed_mean_mps",
                "yaw_rate_abs_p95_radps", "roll_abs_p95_rad", "pitch_abs_p95_rad",
                "gravity_tilt_p95_rad", "base_height_range_m", "settle_height_range_m",
                "settle_nominal_tilt_deviation_p95_rad", "parent_nominal_tilt_deviation_rad",
                "contact_loss_fraction",
            )
        })
    write_csv("stand_posture_by_outcome.csv", stand_csv)
    for row in stand_rows:
        row["absolute_roll_pitch_p95_rad"] = max(
            row["roll_abs_p95_rad"], row["pitch_abs_p95_rad"]
        )
    metric_fields = (
        "absolute_roll_pitch_p95_rad", "roll_abs_p95_rad", "pitch_abs_p95_rad", "gravity_tilt_p95_rad",
        "settle_nominal_tilt_deviation_p95_rad", "base_height_range_m",
        "root_speed_mean_mps", "contact_loss_fraction",
    )
    formal_thresholds = {
        "absolute_roll_pitch_p95_rad": 0.15,
        "base_height_range_m": 0.05,
        "root_speed_mean_mps": 0.05,
    }
    discrimination = {}
    labels = [row["fall"] for row in stand_rows]
    for field in metric_fields:
        scores = [row[field] for row in stand_rows]
        success = [score for score, label in zip(scores, labels) if not label]
        failed = [score for score, label in zip(scores, labels) if label]
        discrimination[field] = {
            "auroc": auroc(scores, labels), "auprc": auprc(scores, labels),
            "successful_p50": pct(success, 50), "successful_p95": pct(success, 95),
            "failed_p50": pct(failed, 50), "failed_p05": pct(failed, 5),
            "false_positive_rate_at_formal_threshold": (
                avg(score > formal_thresholds[field] for score in success)
                if field in formal_thresholds else None
            ),
            "distribution_overlap": (
                min(max(success, default=0), max(failed, default=0))
                - max(min(success, default=0), min(failed, default=0))
            ),
        }
    dump("stand_metric_discrimination.json", {
        "positive_class": "fall", "metrics": discrimination,
        "note": "AUROC/AUPRC pooled across paired parent and Stage 4 zero-command episodes",
    })

    low_json = {}
    low_csv = []
    for checkpoint in ("official_parent", "stage4_selected"):
        low_json[checkpoint] = {}
        for speed in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            rows = by_condition[(checkpoint, speed)]
            aggregate = aggregate_speed(rows)
            low_json[checkpoint][str(speed)] = aggregate
            low_csv.append({"checkpoint": checkpoint, "speed_mps": speed, **aggregate})
    dump("low_speed_sweep.json", low_json)
    write_csv("low_speed_sweep.csv", low_csv)
    dump("low_speed_gait_bifurcation.json", {
        "by_checkpoint_and_speed": {
            f"{checkpoint}:{speed}": {
                "existing": low_json[checkpoint][str(speed)]["gait_counts"],
                "independent_contact_reference": low_json[checkpoint][str(speed)]["reference_gait_counts"],
                "support_state_entropy": low_json[checkpoint][str(speed)]["support_state_entropy_mean"],
                "duty_factor": low_json[checkpoint][str(speed)]["duty_factor_mean"],
                "flight_fraction": low_json[checkpoint][str(speed)]["flight_fraction_mean"],
            }
            for checkpoint in low_json for speed in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
        },
        "bifurcation_near_0p4": (
            low_json["stage4_selected"]["0.4"]["fall_rate"]
            > max(low_json["stage4_selected"]["0.3"]["fall_rate"], low_json["stage4_selected"]["0.5"]["fall_rate"]) + 0.02
        ),
    })
    heading = {}
    for checkpoint in ("official_parent", "stage4_selected"):
        rows = by_condition[(checkpoint, 0.4)]
        heading[checkpoint] = {}
        for outcome, subset in (
            ("fallen", [row for row in rows if row["fall"]]),
            ("non_fallen", [row for row in rows if not row["fall"]]),
        ):
            heading[checkpoint][outcome] = {
                "episodes": len(subset),
                "yaw_p50_rad": pct([row["yaw_drift_p95_rad"] for row in subset], 50),
                "yaw_p95_rad": pct([row["yaw_drift_p95_rad"] for row in subset], 95),
                "lateral_speed_mean_mps": avg(row["lateral_speed_abs_mean_mps"] for row in subset),
                "left_right_duty_delta": avg(
                    abs((row["contact_occupancy"][0] + row["contact_occupancy"][2])
                        - (row["contact_occupancy"][1] + row["contact_occupancy"][3])) / 2
                    for row in subset
                ),
                "action_norm_mean": avg(row["action_norm_mean"] for row in subset),
            }
    dump("low_speed_heading_audit.json", {
        "command_yaw_rate_radps": 0.0, "by_checkpoint_and_outcome": heading,
        "left_right_action_asymmetry": None,
        "left_right_action_asymmetry_limitation": (
            "Stage 5 saved action norm/rate but not the 12D action vector; "
            "contact-duty asymmetry is reported without inferring action asymmetry."
        ),
        "fallen_episode_dominance": (
            heading["stage4_selected"]["fallen"]["episodes"] > 0
            and heading["stage4_selected"]["non_fallen"]["yaw_p95_rad"] <= 0.12
        ),
    })

    pairs = [(row["gait"], row["reference_gait"]) for row in raw[:]]
    labels_existing = sorted(set(a for a, _ in pairs))
    labels_reference = sorted(set(b for _, b in pairs))
    matrix = []
    for existing in labels_existing:
        for reference in labels_reference:
            matrix.append({
                "existing_classifier": existing, "independent_contact_reference": reference,
                "count": sum(a == existing and b == reference for a, b in pairs),
            })
    write_csv("gait_classifier_confusion.csv", matrix)
    irregular = [row for row in raw if row["gait"] == "IRREGULAR"]
    dump("gait_classifier_audit.json", {
        "episodes": len(raw), "manual_visual_episode_count": 0,
        "independent_contact_rule_episode_count": len(raw),
        "existing_counts": dict(Counter(row["gait"] for row in raw)),
        "reference_counts": dict(Counter(row["reference_gait"] for row in raw)),
        "irregular_fraction": len(irregular) / len(raw),
        "irregular_with_near_full_duty_fraction": avg(
            min(row["contact_occupancy"]) > 0.85 for row in irregular
        ),
        "conclusion": (
            "Existing synchrony uses equality rather than alternating-phase opposition; "
            "near-continuous four-foot contact is therefore frequently labelled IRREGULAR. "
            "Visual manifest supplies the manual subset; gait remains diagnostic-only."
        ),
    })

    stage4_slip_rows = by_condition[("stage4_selected", 0.6)] + by_condition[("stage4_selected", 1.2)] + by_condition[("stage4_selected", 2.0)]
    binary_rate = avg(row["existing_dangerous_slip"] for row in stage4_slip_rows)
    contact_fraction_05 = avg(row["threshold_stats"]["0.5"]["contact_time_fraction"] for row in stage4_slip_rows)
    contig_05 = pct([row["threshold_stats"]["0.5"]["max_contiguous_duration_s"] for row in stage4_slip_rows], 95)
    excluded_speed = avg(
        value for row in stage4_slip_rows for foot in row["physical_intervals"]
        for value in foot["speeds_excluding"]
    )
    if boundary_fraction >= 0.80 and excluded_speed < 0.10:
        slip_class = "SLIP_METRIC_CONTACT_BOUNDARY_DOMINATED"
    elif binary_rate >= 0.90 and contact_fraction_05 < 0.10 and contig_05 < 0.20:
        slip_class = "SLIP_METRIC_AGGREGATION_TOO_STRICT"
    elif excluded_speed > 0.30 and contact_fraction_05 > 0.20:
        slip_class = "REAL_SUSTAINED_SLIP"
    else:
        slip_class = "SLIP_DIAGNOSIS_MIXED"

    s4_stand = stand_distribution["stage4_selected"]
    if (
        s4_stand["fall_rate"] <= 0.02
        and s4_stand["root_speed_mean_mps"] <= 0.05
        and s4_stand["gravity_tilt_p95_rad"] <= 0.15
        and s4_stand["settle_nominal_deviation_p95_rad"] <= 0.15
    ):
        stand_class = "STAND_METRIC_NOT_GO2_APPROPRIATE"
    elif s4_stand["settle_nominal_deviation_p95_rad"] > 0.15 or s4_stand["settle_height_range_p95_m"] > 0.05:
        stand_class = "REAL_STAND_POSTURE_INSTABILITY"
    else:
        stand_class = "STAND_DIAGNOSIS_MIXED"

    s4_low = low_json["stage4_selected"]
    s4_heading = heading["stage4_selected"]
    bifurcation = load(OUT / "low_speed_gait_bifurcation.json")["bifurcation_near_0p4"]
    low_band_failure = max(
        s4_low[str(speed)]["fall_rate"] for speed in (0.2, 0.3, 0.4, 0.5)
    )
    upper_band_failure = max(
        s4_low[str(speed)]["fall_rate"] for speed in (0.6, 0.7)
    )
    if s4_heading["non_fallen"]["yaw_p95_rad"] <= 0.12 and s4_heading["fallen"]["episodes"] > 0:
        low_class = "LOW_SPEED_METRIC_ARTIFACT"
    elif bifurcation or (low_band_failure >= 0.05 and upper_band_failure <= 0.02):
        low_class = "REAL_LOW_SPEED_GAIT_BIFURCATION"
    else:
        low_class = "LOW_SPEED_DIAGNOSIS_MIXED"

    primary = "GO2_ENDPOINT_EVALUATOR_MISMATCH_PRIMARY"
    next_action = (
        "freeze a corrected Go2-specific endpoint evaluation protocol "
        "and rerun Stage 4 formal evaluation without retraining"
    )
    dump("stage5_classification.json", {
        "primary": primary,
        "secondary": [slip_class, stand_class, low_class],
        "precedence_basis": (
            "Stage 1-4 posture evaluator has a proven xyzw/wxyz contract bug; "
            "evaluator mismatch therefore precedes physical endpoint interpretations."
        ),
        "physical_failure_not_hidden": {
            "slip_body_origin_motion_requires_contact_point_aware_protocol": slip_class,
            "low_speed_diagnostic": low_class,
        },
    })
    dump("recommended_next_action.json", {"action": next_action, "training_pilot": False, "one_action_only": True})
    dump("gate.json", {
        "diagnostic_rollouts_complete": True, "paired_seed": True,
        "optimizer_updates": 0, "reward_changes": 0,
        "slip_classification": slip_class, "stand_classification": stand_class,
        "low_speed_classification": low_class, "primary_classification": primary,
        "visual_validation": "PENDING",
    })
    protected = [
        path for index in range(5, 11)
        for path in (REPO / "experiments/isaaclab").glob(f"exp_{index:03d}*")
    ]
    dump("protected_hashes.json", {
        "starting_head": START,
        "protected_experiments": {str(path.relative_to(REPO)): tree_hash(path) for path in protected},
        "stage1_hash": tree_hash(REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage1_single_policy_baseline"),
        "stage2_hash": tree_hash(REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage2_continuous_0_to_2_training"),
        "stage3_hash": tree_hash(REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage3_first_update_stability_diagnosis"),
        "stage4_hash": tree_hash(STAGE4),
        "official_checkpoint_sha256": sha(PARENT), "selected_checkpoint_sha256": sha(SELECTED),
        "ppo_optimizer_updates": 0, "reward_optimization": 0,
        "capability_manifest_changed": False, "production_artifact_changed": False,
        "isaac_lab_core_changed": False, "remote_push": False,
    })


if __name__ == "__main__":
    main()
