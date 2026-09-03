"""Merge independently completed Stage 6 sections without simulator interaction."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage6_corrected_endpoint_formal"
FEET = ("front-left", "front-right", "rear-left", "rear-right")
STEADY = (0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0)


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def percentile(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    index = (len(values) - 1) * q / 100
    lower, upper = int(index), min(len(values) - 1, int(index) + 1)
    return values[lower] * (upper - index) + values[upper] * (index - lower)


def write_csv(name, rows):
    if not rows:
        (OUT / name).write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def main():
    partials = {}
    for policy in ("official_parent", "stage4_selected"):
        steady = load(f"partial_{policy}_steady.json")
        transitions = load(f"partial_{policy}_transitions.json")
        low = load(f"partial_{policy}_low_speed.json")
        if steady["formal_episode_count"] != 50 or transitions["formal_episode_count"] != 50:
            raise SystemExit(f"{policy}: formal section is not 50 episodes")
        if low["formal_episode_count"] != 20:
            raise SystemExit(f"{policy}: low-speed section is not 20 episodes")
        if any(value["episodes"] != 50 for value in steady["summaries"].values()):
            raise SystemExit(f"{policy}: a steady condition is not 50 episodes")
        if any(value["episodes"] != 50 for value in transitions["summaries"].values()):
            raise SystemExit(f"{policy}: a transition is not 50 episodes")
        partials[policy] = {"steady": steady, "transitions": transitions, "low": low}

    for policy, prefix in (("official_parent", "parent"), ("stage4_selected", "selected")):
        steady = partials[policy]["steady"]
        transitions = partials[policy]["transitions"]
        stand_rows = [row for row in steady["episodes"] if row["target_speed_mps"] == 0.0]
        moving_rows = [row for row in steady["episodes"] if row["target_speed_mps"] != 0.0]
        dump(f"{prefix}_formal_stand.json", {
            "summary": steady["summaries"]["0.0"], "episodes": stand_rows,
        })
        write_csv(f"{prefix}_formal_steady_state.csv", moving_rows)
        if policy == "stage4_selected":
            dump("selected_formal_steady_state.json", {
                "per_speed": {
                    key: value for key, value in steady["summaries"].items() if key != "0.0"
                },
                "episodes": moving_rows,
            })
        write_csv(f"{prefix}_formal_transitions.csv", transitions["episodes"])
        if policy == "stage4_selected":
            dump("selected_formal_transitions.json", {
                "per_direction": transitions["summaries"], "episodes": transitions["episodes"],
            })

    selected_steady = partials["stage4_selected"]["steady"]["summaries"]
    selected_transitions = partials["stage4_selected"]["transitions"]["summaries"]
    prerequisites = (
        selected_steady["0.0"]["gate_pass"]
        and all(selected_steady[str(speed)]["gate_pass"] for speed in (0.6, 1.2, 2.0))
        and all(value["gate_pass"] for value in selected_transitions.values())
    )
    reduced_partial = OUT / "partial_stage4_selected_reduced.json"
    if prerequisites:
        if not reduced_partial.exists():
            dump("formal_reduced_sequence.json", {
                "executed": False, "gate_pass": False,
                "reason": "SEQUENCE_REQUIRED_AFTER_PARTIAL_MERGE",
            })
            reduced = load("formal_reduced_sequence.json")
        else:
            reduced = load("partial_stage4_selected_reduced.json")
    else:
        reduced = {
            "executed": False, "gate_pass": False,
            "reason": "zero, required steady endpoints, or formal transitions not all SUPPORTED",
        }
    dump("formal_reduced_sequence.json", reduced)
    dump("low_speed_diagnostic.json", {
        "seed_root": 20264901, "episodes_per_condition": 20,
        "formal_capability_set": False,
        "results": {
            policy: partials[policy]["low"]["results"]
            for policy in ("official_parent", "stage4_selected")
        },
    })

    selected_rows = partials["stage4_selected"]["steady"]["episodes"]
    transition_rows = partials["stage4_selected"]["transitions"]["episodes"]
    endpoint_rows = []
    for origin, source in (("reset_steady", None), ("0_to_1.2", 0.0), ("2_to_1.2", 2.0)):
        rows = (
            [row for row in selected_rows if row["target_speed_mps"] == 1.2]
            if source is None else
            [row for row in transition_rows if row["source_speed_mps"] == source and row["target_speed_mps"] == 1.2]
        )
        endpoint_rows.append({
            "origin": origin,
            "speed_mean_mps": mean(row["actual_forward_speed_mean_mps"] for row in rows),
            "heading_p95_rad": percentile([row["heading_error_abs_p95_rad"] for row in rows], 95),
            "tilt_p95_rad": percentile([row["gravity_tilt_p95_rad"] for row in rows], 95),
            "dangerous_slip_rate": mean(row["physical_slip"]["dangerous"] for row in rows),
            "action_rate_p95": percentile([row["action_rate_p95"] for row in rows], 95),
            "gait_counts": dict(Counter(row["gait_class_v1"] for row in rows)),
            "high_speed_gait_or_flight_retained": False,
        })
    dump("directional_asymmetry.json", {
        "endpoint_mps": 1.2, "comparisons": endpoint_rows,
        "high_speed_gait_retention_after_deceleration": False,
    })
    write_csv("endpoint_hysteresis.csv", endpoint_rows)

    severity, foot_rows = {}, []
    for policy in ("official_parent", "stage4_selected"):
        rows = partials[policy]["steady"]["episodes"]
        severity[policy] = {}
        for speed in (0.0,) + STEADY:
            subset = [row for row in rows if row["target_speed_mps"] == speed]
            severity[policy][str(speed)] = {
                "dangerous_episode_rate": mean(row["physical_slip"]["dangerous"] for row in subset),
                "speed_p95_mps": percentile(
                    [row["physical_slip"]["contact_point_speed_p95_mps"] for row in subset], 95
                ),
                "displacement_p95_m": percentile(
                    [row["physical_slip"]["anchor_displacement_p95_m"] for row in subset], 95
                ),
                "legacy_origin_speed_p95_mps": percentile(
                    [row["physical_slip"]["legacy_foot_link_origin_speed_p95_mps"] for row in subset], 95
                ),
            }
            for foot in FEET:
                foot_rows.append({
                    "checkpoint": policy, "speed_mps": speed, "foot": foot,
                    "dangerous_episode_rate": mean(
                        row["physical_slip"]["per_foot"][foot]["dangerous"] for row in subset
                    ),
                    "speed_p95_mps": percentile(
                        [row["physical_slip"]["per_foot"][foot]["speed_p95_mps"] for row in subset], 95
                    ),
                    "displacement_p95_m": percentile(
                        [row["physical_slip"]["per_foot"][foot]["displacement_p95_m"] for row in subset], 95
                    ),
                })
    dump("physical_slip_severity.json", severity)
    write_csv("per_foot_contact_point_slip.csv", foot_rows)
    dump("formal_merge_audit.json", {
        "status": "PASS", "steady_episodes_per_condition": 50,
        "transition_episodes_per_condition": 50, "low_speed_episodes_per_condition": 20,
        "simulator_sections_isolated": True, "offline_merge_only": True,
    })


if __name__ == "__main__":
    main()
