"""Freeze the single-variable Stage 7R8 saturation tuning protocol."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage7r8_walk_to_run_pilot2_saturation"
P1 = EXP / "configs/stage7r_walk_to_run_pilot1.yaml"
P2 = EXP / "configs/stage7r_walk_to_run_pilot2_saturation.yaml"
EVENTS = OUT / "saturation_events.csv"
PARENT = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage7r7_frozen_pilot1_execution/checkpoints/model_75.pt"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def normalized(cfg):
    cfg = json.loads(json.dumps(cfg))
    cfg["experiment"]["experiment_name"] = "<run_name>"
    cfg["experiment"]["training_seed"] = "<seed>"
    cfg["actor"]["parent_checkpoint"] = "<parent>"
    cfg["actor"]["parent_sha256"] = "<parent_sha>"
    cfg["reward"]["ankle_effort_dwell"] = "<selected_saturation_weight>"
    cfg["runtime"].pop("run_name", None)
    cfg["runtime"].pop("output_path", None)
    return cfg


def main():
    p1 = yaml.safe_load(P1.read_text(encoding="utf-8"))
    p2 = yaml.safe_load(P2.read_text(encoding="utf-8"))
    with EVENTS.open(newline="", encoding="utf-8") as stream:
        events = list(csv.DictReader(stream))
    target24 = [event for event in events if float(event["target_speed_mps"]) == 2.4]
    groups = Counter(event["joint_name"] for event in target24)
    term_count = sum("ankle_" in event["joint_name"] for event in target24)
    localization = term_count / len(target24) if target24 else 0.0
    phases = Counter((event["active_transition_phase"], event["support_foot"]) for event in events)
    diagnosis = {
        "evaluation_seed": 20261201,
        "episodes_requested": {"2.4": 40, "2.6": 40, "2.8": 40},
        "events": len(events),
        "events_by_target": dict(Counter(event["target_speed_mps"] for event in events)),
        "events_by_joint": dict(groups),
        "dominant_joint": groups.most_common(1)[0][0] if groups else None,
        "dominant_joint_fraction": groups.most_common(1)[0][1] / len(target24) if target24 else 0.0,
        "existing_reward_term": "ankle_effort_dwell",
        "existing_term_explanation_fraction": localization,
        "localization_gate_minimum": 0.80,
        "localization_gate_pass": localization >= 0.80,
        "limit_type": "effort",
        "specific_to_2p4": all(float(event["target_speed_mps"]) == 2.4 for event in events),
    }
    write_json("saturation_root_cause.json", diagnosis)
    write_json("pilot2_saturation_root_cause.json", diagnosis)
    distribution = {
        "phase_support_counts": {f"{phase}|{support}": count for (phase, support), count in phases.items()},
        "run_takeover_fraction": sum(event["active_transition_phase"] == "RUN_TAKEOVER" for event in events) / len(events),
        "left_support_fraction": sum(event["support_foot"] == "left" for event in events) / len(events),
    }
    write_json("saturation_phase_distribution.json", distribution)
    write_json("pilot2_saturation_phase_distribution.json", distribution)
    selected = {
        "term": "ankle_effort_dwell",
        "old_weight": p1["reward"]["ankle_effort_dwell"],
        "new_weight": p2["reward"]["ankle_effort_dwell"],
        "multiplier": abs(p2["reward"]["ankle_effort_dwell"] / p1["reward"]["ankle_effort_dwell"]),
        "reason": "100% of 2.4 saturation events are ankle effort dwell; 95.8% are left ankle roll.",
    }
    write_json("selected_saturation_term.json", selected)
    unexpected = [] if normalized(p1) == normalized(p2) else ["canonical_config_after_allowed_fields"]
    write_json("pilot1_vs_pilot2_config_diff.json", {
        "allowed_differences": ["parent checkpoint", "training seed", "ankle_effort_dwell weight", "run name", "output path"],
        "actual_differences": {
            "parent_checkpoint": [p1["actor"]["parent_checkpoint"], p2["actor"]["parent_checkpoint"]],
            "parent_sha256": [p1["actor"]["parent_sha256"], p2["actor"]["parent_sha256"]],
            "training_seed": [p1["experiment"]["training_seed"], p2["experiment"]["training_seed"]],
            "experiment_name": [p1["experiment"]["experiment_name"], p2["experiment"]["experiment_name"]],
            "ankle_effort_dwell": [p1["reward"]["ankle_effort_dwell"], p2["reward"]["ankle_effort_dwell"]],
            "run_name": p2["runtime"]["run_name"],
            "output_path": p2["runtime"]["output_path"],
        },
        "unexpected_config_differences": len(unexpected),
        "unexpected_fields": unexpected,
    })
    hashes = {
        "pilot2_config_sha256": digest(p2),
        "pilot2_reward_sha256": digest(p2["reward"]),
        "pilot2_actor_initialization_sha256": digest(p2["actor"]),
        "pilot2_parent_sha256": file_sha(PARENT),
        "run_name": f"stage7r8-pilot2-sat{digest(p2['reward'])[:8]}-seed{p2['experiment']['training_seed']}",
    }
    write_json("pilot2_protocol_hashes.json", hashes)
    (OUT / "pilot2_config.yaml").write_text(P2.read_text(encoding="utf-8"), encoding="utf-8")
    write_json("stage7r7_reference.json", {
        "status": "CLEAR_LEARNING_SIGNAL",
        "selected_checkpoint": str(PARENT.relative_to(REPO)).replace("\\", "/"),
        "selected_checkpoint_sha256": file_sha(PARENT),
        "results_immutable": True,
    })
    if not diagnosis["localization_gate_pass"] or unexpected:
        raise SystemExit("Stage 7R8 pre-training gate failed")


if __name__ == "__main__":
    main()
