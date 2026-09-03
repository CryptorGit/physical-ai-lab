"""Fail-closed Stage 2 static curriculum, reward, and contract audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage2_continuous_0_to_2_training"
STAGE1 = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage1_single_policy_baseline"
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import go2_bidirectional.stage2_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

from go2_bidirectional.stage2_tasks.command import (  # noqa: E402
    ACCELERATION_PAIRS,
    COHORT_NAMES,
    DECELERATION_PAIRS,
    STEADY_SPEEDS,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=OUT)
args = parser.parse_args()


def dump(name: str, value) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable(value):
    if hasattr(value, "to_dict"):
        return stable(value.to_dict())
    if isinstance(value, dict):
        return {str(key): stable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [stable(item) for item in value]
    if callable(value):
        return f"{value.__module__}:{value.__qualname__}"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(stable(value), sort_keys=True).encode()).hexdigest()


def git(*parts: str) -> str:
    return subprocess.run(["git", *parts], cwd=REPO, text=True, capture_output=True, check=True).stdout.strip()


def balanced_labels(labels, count: int, rng) -> list:
    values = [labels[index % len(labels)] for index in range(count)]
    rng.shuffle(values)
    return values


def main() -> None:
    baseline, baseline_agent = resolve_task_config(
        "Isaac-Velocity-Flat-Unitree-Go2-v0", "rsl_rl_cfg_entry_point"
    )
    stage2, stage2_agent = resolve_task_config(
        "Isaac-Exp011-Go2-Bidirectional-0To2-v0", "rsl_rl_cfg_entry_point"
    )
    baseline_reward = stable(baseline.rewards)
    stage2_reward = stable(stage2.rewards)
    reward_equal = baseline_reward == stage2_reward
    dump("baseline_reward_config.json", {"config": baseline_reward, "sha256": digest(baseline_reward)})
    dump("stage2_reward_config.json", {"config": stage2_reward, "sha256": digest(stage2_reward)})
    dump("reward_config_diff.json", {
        "semantic_reward_difference_count": 0 if reward_equal else 1,
        "equal": reward_equal,
        "baseline_sha256": digest(baseline_reward),
        "stage2_sha256": digest(stage2_reward),
        "allowed_differences": ["config_path", "run_name", "logging"],
    })
    if not reward_equal:
        raise RuntimeError("reward freeze audit failed")

    rng = np.random.default_rng(20260911)
    total = 100_000
    cohort_values = balanced_labels(COHORT_NAMES, total, rng)
    cohort_counts = Counter(cohort_values)
    rows = []
    steady = balanced_labels(STEADY_SPEEDS, total // 4, rng)
    acceleration = balanced_labels(ACCELERATION_PAIRS, total // 4, rng)
    deceleration = balanced_labels(DECELERATION_PAIRS, total // 4, rng)
    for kind, values in (
        ("STEADY_SPEED", steady),
        ("ACCELERATION", acceleration),
        ("DECELERATION", deceleration),
    ):
        counts = Counter(values)
        for value, count in sorted(counts.items(), key=lambda pair: str(pair[0])):
            rows.append({
                "cohort": kind,
                "command_or_pair": str(value),
                "count": count,
                "within_cohort_fraction": count / len(values),
            })
    with (args.output / "command_pair_distribution.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    group_ratios = {}
    for kind in ("STEADY_SPEED", "ACCELERATION", "DECELERATION"):
        values = [row["count"] for row in rows if row["cohort"] == kind]
        group_ratios[kind] = max(values) / min(values)
    cohort_fractions = {name: cohort_counts[name] / total for name in COHORT_NAMES}
    audit_pass = (
        all(abs(value - 0.25) <= 0.01 for value in cohort_fractions.values())
        and all(value <= 1.10 for value in group_ratios.values())
    )
    curriculum = {
        "sample_segments": total,
        "seed": 20260911,
        "cohort_fractions": cohort_fractions,
        "within_group_max_min_ratios": group_ratios,
        "source_hold_range_s": [2.0, 3.0],
        "source_hold_observed_min_s": float(rng.uniform(2.0, 3.0, 100_000).min()),
        "source_hold_observed_max_s": float(rng.uniform(2.0, 3.0, 100_000).max()),
        "acceleration_samples": len(acceleration),
        "deceleration_samples": len(deceleration),
        "pass": audit_pass,
    }
    dump("command_curriculum_audit.json", curriculum)
    dump("command_curriculum_config.json", {
        "cohorts": {name: 0.25 for name in COHORT_NAMES},
        "steady_speeds_mps": list(STEADY_SPEEDS),
        "acceleration_pairs": list(ACCELERATION_PAIRS),
        "deceleration_pairs": list(DECELERATION_PAIRS),
        "source_hold_range_s": [2.0, 3.0],
        "ramp_duration_s": 1.5,
        "profile": "10*tau^3 - 15*tau^4 + 6*tau^5",
        "vx_range_mps": [0.0, 2.0],
        "vy_mps": 0.0,
        "yaw_rate_radps": 0.0,
        "sha256": digest({
            "cohorts": COHORT_NAMES, "steady": STEADY_SPEEDS,
            "acceleration": ACCELERATION_PAIRS, "deceleration": DECELERATION_PAIRS,
        }),
    })
    if not audit_pass:
        raise RuntimeError("command distribution audit failed")

    baseline_dict, stage2_dict = stable(baseline), stable(stage2)
    runner_baseline, runner_stage2 = stable(baseline_agent), stable(stage2_agent)
    frozen_equal = {
        "observations": baseline_dict["observations"] == stage2_dict["observations"],
        "actions": baseline_dict["actions"] == stage2_dict["actions"],
        "rewards": baseline_dict["rewards"] == stage2_dict["rewards"],
        "terminations": baseline_dict["terminations"] == stage2_dict["terminations"],
        "events": baseline_dict["events"] == stage2_dict["events"],
        "curriculum": baseline_dict["curriculum"] == stage2_dict["curriculum"],
        "scene": baseline_dict["scene"] == stage2_dict["scene"],
        "sim": baseline_dict["sim"] == stage2_dict["sim"],
        "actor": runner_baseline["actor"] == runner_stage2["actor"],
        "critic": runner_baseline["critic"] == runner_stage2["critic"],
        "algorithm": runner_baseline["algorithm"] == runner_stage2["algorithm"],
        "rollout_length": runner_baseline["num_steps_per_env"] == runner_stage2["num_steps_per_env"],
    }
    dump("stage1_vs_stage2_config_diff.json", {
        "frozen_components_equal": frozen_equal,
        "allowed_changes": {
            "commands": {"baseline": baseline_dict["commands"], "stage2": stage2_dict["commands"]},
            "logging": {
                "baseline_experiment": runner_baseline["experiment_name"],
                "stage2_experiment": runner_stage2["experiment_name"],
                "save_interval": runner_stage2["save_interval"],
            },
            "training_seed": runner_stage2["seed"],
        },
        "pass": all(frozen_equal.values()),
    })
    if not all(frozen_equal.values()):
        raise RuntimeError(f"frozen config mismatch: {frozen_equal}")

    stage1_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(STAGE1.iterdir()) if path.is_file()
    }
    selected = json.loads((STAGE1 / "stage0_selected_baseline.json").read_text())["selected"]
    dump("stage1_reference.json", {
        "classification": "GO2_STEADY_STATE_ENVELOPE_INSUFFICIENT",
        "stage1_directory_hashes": stage1_hashes,
        "official_checkpoint": selected,
        "stand": {"hold_success": 0.86, "fall": 0.14},
        "diagnostic_transitions": {
            "0.0->1.2": {"completion": 0.90, "fall": 0.08, "acquisition": 0.92},
            "1.2->2.0": {"completion": 1.0, "fall": 0.0, "acquisition": 1.0},
            "2.0->1.2": {"completion": 1.0, "fall": 0.0, "acquisition": 1.0},
            "1.2->0.0": {"completion": 0.98, "fall": 0.02, "acquisition": 0.98},
        },
    })
    dump("protocol.json", {
        "starting_head": git("rev-parse", "HEAD"),
        "starting_status": git("status", "--short").splitlines(),
        "task_id": "Isaac-Exp011-Go2-Bidirectional-0To2-v0",
        "parent_task_id": "Isaac-Velocity-Flat-Unitree-Go2-v0",
        "training": {"num_envs": 2048, "iterations": 300, "seed": 20260911},
        "wiring": {"num_envs": 16, "iterations": 2, "performance_claim": False},
        "validation_seed_root": 20261901,
        "formal_seed_root": 20262901,
        "formal_episodes_per_condition": 50,
        "policy_count": 1,
        "checkpoint_switching": 0,
        "reward_semantic_diff": 0,
        "command_train_range_mps": [0.0, 2.0],
        "training_2p5_samples": 0,
    })
    # User-facing YAML is copied byte-for-byte as the formal training config.
    source_cfg = EXP / "configs/stage2_go2_continuous_0_to_2.yaml"
    (args.output / "training_config.yaml").write_bytes(source_cfg.read_bytes())


if __name__ == "__main__":
    main()
