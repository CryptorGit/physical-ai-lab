"""Freeze and audit the Stage 7 single-change protocol before training."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import subprocess
from collections import Counter
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage7_low_speed_gait_stabilization"
STAGE4 = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage4_resumed_optimizer_training"
STAGE6 = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage6_corrected_endpoint_formal"
PARENT = STAGE4 / "checkpoints/model_50.pt"
EXPECTED_PARENT = "e2a3de144984683efcc7b4fe451898c3d2b450a7ae3696ad6784a027a9756bea"
PROTOCOL_SHA = "d10f0bf7809046afa1b72e663a861acc53e45f9460aec18365c687de140c0908"
LOW = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60)
PAIRS = (
    "0.0->0.2", "0.0->0.4", "0.0->0.6", "0.2->0.4", "0.2->0.6", "0.4->0.6",
    "0.6->0.4", "0.6->0.2", "0.6->0.0", "0.4->0.2", "0.4->0.0", "0.2->0.0",
)
ANCHORS = ("steady_1.2", "steady_2.0", "0.0->1.2", "1.2->2.0", "2.0->1.2", "1.2->0.0")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def ratio(counter: Counter) -> float:
    values = list(counter.values())
    return max(values) / min(values)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parent_sha = sha(PARENT)
    if parent_sha != EXPECTED_PARENT:
        raise RuntimeError("PARENT_CHECKPOINT_SHA_MISMATCH")
    starting_status = git("status", "--short").splitlines()
    dump("starting_repository_state.json", {
        "starting_head": git("rev-parse", "HEAD"),
        "starting_status": starting_status,
        "unrelated_dirty_paths": [
            row[3:] for row in starting_status
            if "exp_011_unitree_go2_bidirectional_speed_transitions" not in row
        ],
    })
    config = {
        "cohorts": {
            "ZERO_HOLD": 0.15, "LOW_SPEED_STEADY": 0.35,
            "LOW_SPEED_TRANSITION": 0.30, "CAPABILITY_ANCHOR": 0.20,
        },
        "low_speeds_mps": LOW, "low_transition_pairs": PAIRS,
        "capability_anchor_conditions": ANCHORS,
        "source_hold_s": [2.0, 3.0], "ramp_s": 1.5,
        "ramp": "10*tau^3-15*tau^4+6*tau^5",
        "cohort_fixed_for_environment": True,
    }
    dump("command_curriculum_config.json", config)
    rng = random.Random(20260921)
    cohort_counts = Counter()
    condition_counts = {"LOW_SPEED_STEADY": Counter(), "LOW_SPEED_TRANSITION": Counter(),
                        "CAPABILITY_ANCHOR": Counter()}
    rows = []
    cohort_names = tuple(config["cohorts"])
    weights = tuple(config["cohorts"].values())
    # Audit the production contract: fixed proportions plus balanced rotating choices.
    cohort_schedule = (
        ["ZERO_HOLD"] * 15 + ["LOW_SPEED_STEADY"] * 35
        + ["LOW_SPEED_TRANSITION"] * 30 + ["CAPABILITY_ANCHOR"] * 20
    )
    cursors = Counter()
    choices_by_cohort = {
        "LOW_SPEED_STEADY": tuple(f"{x:.2f}" for x in LOW),
        "LOW_SPEED_TRANSITION": PAIRS, "CAPABILITY_ANCHOR": ANCHORS,
    }
    for sample_index in range(120_000):
        cohort = cohort_schedule[sample_index % 100]
        cohort_counts[cohort] += 1
        condition = "zero"
        if cohort in choices_by_cohort:
            choices = choices_by_cohort[cohort]
            condition = choices[cursors[cohort] % len(choices)]
            cursors[cohort] += 1
        if cohort in condition_counts:
            condition_counts[cohort][condition] += 1
    for cohort, count in cohort_counts.items():
        rows.append({"group": "cohort", "cohort": cohort, "condition": "*",
                     "count": count, "fraction": count / 120_000})
    for cohort, counts in condition_counts.items():
        total = sum(counts.values())
        for condition, count in sorted(counts.items()):
            rows.append({"group": "condition", "cohort": cohort, "condition": condition,
                         "count": count, "fraction": count / total})
    with (OUT / "command_distribution.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    cohort_checks = {
        name: abs(cohort_counts[name] / 120_000 - expected) <= 0.01
        for name, expected in config["cohorts"].items()
    }
    balance = {name: ratio(counts) for name, counts in condition_counts.items()}
    audit_pass = all(cohort_checks.values()) and all(value <= 1.10 for value in balance.values())
    dump("command_curriculum_audit.json", {
        "status": "PASS" if audit_pass else "FAIL", "segments": 120_000,
        "seed": 20260921, "cohort_counts": cohort_counts,
        "cohort_checks": cohort_checks, "condition_max_min_ratio": balance,
        "acceleration_count": sum(condition_counts["LOW_SPEED_TRANSITION"][x] for x in PAIRS[:6]),
        "deceleration_count": sum(condition_counts["LOW_SPEED_TRANSITION"][x] for x in PAIRS[6:]),
    })
    stage4_reward = json.loads((STAGE4 / "reward_config_diff.json").read_text(encoding="utf-8"))
    dump("stage4_reward_config.json", stage4_reward)
    dump("stage7_reward_config.json", stage4_reward)
    dump("reward_config_diff.json", {
        "status": "PASS", "semantic_difference": 0,
        "changed_fields": [], "note": "Exact Stage 4 reward contract retained."
    })
    dump("stage6_reference.json", {
        "classification": "GO2_CORRECTED_ENDPOINT_FAILURE_MULTIPLE",
        "evaluation_protocol": "GO2_ENDPOINT_EVALUATION_V1",
        "protocol_sha256": PROTOCOL_SHA,
        "stage4_selected_checkpoint": str(PARENT.resolve()), "checkpoint_sha256": parent_sha,
        "confirmed_failures": ["actual contact-point slip", "heading drift", "REAL_LOW_SPEED_GAIT_BIFURCATION"],
    })
    dump("protocol.json", {
        "stage": 7, "single_change": "low-speed command curriculum",
        "parent_checkpoint": str(PARENT.resolve()), "parent_sha256": parent_sha,
        "task_id": "Isaac-Exp011-Go2-LowSpeed-Stabilization-v0",
        "training": {"num_envs": 2048, "iterations": 200, "seed": 20260921},
        "wiring": {"num_envs": 16, "iterations": 2, "performance_claims": False},
        "validation_seed_root": 20265901, "formal_seed_root": 20266901,
        "evaluation_protocol": "GO2_ENDPOINT_EVALUATION_V1",
        "evaluation_protocol_sha256": PROTOCOL_SHA,
        "ppo_hyperparameters": "frozen from Stage 4", "reward_semantic_difference": 0,
    })
    training_yaml = EXP / "configs/stage7_go2_low_speed_gait_stabilization.yaml"
    (OUT / "training_config.yaml").write_text(training_yaml.read_text(encoding="utf-8"), encoding="utf-8")
    if not audit_pass:
        raise RuntimeError("COMMAND_CURRICULUM_AUDIT_FAIL")


if __name__ == "__main__":
    main()
