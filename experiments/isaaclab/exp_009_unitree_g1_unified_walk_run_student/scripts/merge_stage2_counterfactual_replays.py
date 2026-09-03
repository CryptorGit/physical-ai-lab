"""Match fresh-app plus/minus replays by physical env ID and reject mismatched states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage2_dynamics_sensitive_distillation"
CFG = yaml.safe_load((EXP / "configs/stage2_dynamics_sensitive_distillation.yaml").read_text(encoding="utf-8"))

CRITICAL_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "left_hip_roll_joint", "right_hip_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_roll_joint",
]
CRITICAL_INDICES = [0, 1, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20]


def key(data, index):
    return str(data["regime"][index]), int(data["cycle"][index]), int(data["physical_env_id"][index])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="")
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--minimum-states", type=int, default=0)
    args = parser.parse_args()
    suffix = f"_{args.suffix}" if args.suffix else ""
    with np.load(OUT / f"sensitivity_replay_plus{suffix}.npz", allow_pickle=True) as archive:
        plus = {name: archive[name] for name in archive.files}
    with np.load(OUT / f"sensitivity_replay_minus{suffix}.npz", allow_pickle=True) as archive:
        minus = {name: archive[name] for name in archive.files}
    plus_index = {key(plus, index): index for index in range(len(plus["regime"]))}
    minus_index = {key(minus, index): index for index in range(len(minus["regime"]))}
    common = sorted(set(plus_index) & set(minus_index))
    tolerances = CFG["sensitivity"]
    retained, rejected, comparisons = [], 0, {}
    for item in common:
        pi, mi = plus_index[item], minus_index[item]
        root_error = float(np.max(np.abs(plus["branch_root"][pi] - minus["branch_root"][mi])))
        joint_error = float(np.max(np.abs(plus["branch_joint"][pi] - minus["branch_joint"][mi])))
        velocity_error = float(np.max(np.abs(plus["branch_velocity"][pi] - minus["branch_velocity"][mi])))
        valid = (
            root_error <= tolerances["state_match_root_tolerance_m"]
            and joint_error <= tolerances["state_match_joint_tolerance_rad"]
            and velocity_error <= tolerances["state_match_velocity_tolerance"]
            and int(plus["action_dimension"][pi]) == int(minus["action_dimension"][mi])
        )
        group = comparisons.setdefault((item[0], item[1]), {
            "regime": item[0], "cycle": item[1], "common_envs": 0, "matched_states": 0,
            "root_position_max_error_m": 0.0, "joint_position_max_error_rad": 0.0, "velocity_max_error": 0.0,
        })
        group["common_envs"] += 1
        group["root_position_max_error_m"] = max(group["root_position_max_error_m"], root_error)
        group["joint_position_max_error_rad"] = max(group["joint_position_max_error_rad"], joint_error)
        group["velocity_max_error"] = max(group["velocity_max_error"], velocity_error)
        if valid:
            retained.append((pi, mi))
            group["matched_states"] += 1
        else:
            rejected += 1
    if len(retained) < args.minimum_states:
        raise RuntimeError(f"matched states {len(retained)} below required {args.minimum_states}")
    pi = np.asarray([item[0] for item in retained])
    mi = np.asarray([item[1] for item in retained])
    packed = {
        "regime": plus["regime"][pi],
        "cycle": plus["cycle"][pi],
        "physical_env_id": plus["physical_env_id"][pi],
        "target_speed_mps": plus["target_speed_mps"][pi],
        "support_phase": plus["support_phase"][pi],
        "action_dimension": plus["action_dimension"][pi],
        "observation": plus["observation"][pi],
        "teacher_action": plus["teacher_action"][pi],
    }
    for horizon in CFG["sensitivity"]["horizons_steps"]:
        packed[f"plus_continuous_{horizon}"] = plus[f"continuous_{horizon}"][pi]
        packed[f"minus_continuous_{horizon}"] = minus[f"continuous_{horizon}"][mi]
        packed[f"plus_discrete_{horizon}"] = plus[f"discrete_{horizon}"][pi]
        packed[f"minus_discrete_{horizon}"] = minus[f"discrete_{horizon}"][mi]
    np.savez_compressed(OUT / f"dynamic_sensitivity_samples{suffix}.npz", **packed)
    counts = {regime: int(np.sum(packed["regime"] == regime)) for regime in ("walk_steady", "run_steady", "walk_to_run")}
    speed_counts = {
        f"{speed:.1f}": int(np.sum(np.isclose(packed["target_speed_mps"], speed)))
        for speed in (.6, .8, 1., 1.2, 2.4, 2.6, 2.8)
        if np.any(np.isclose(packed["target_speed_mps"], speed))
    }
    (OUT / f"prebranch_state_matching{suffix}.json").write_text(json.dumps({
        "method": "fresh Isaac app per sign; same task/reset seed, source route, physical env ID and prebranch actions",
        "comparisons": list(comparisons.values()),
        "plus_states": len(plus["regime"]), "minus_states": len(minus["regime"]),
        "common_states": len(common), "retained_states": len(retained), "rejected_mismatched_states": rejected,
        "all_within_tolerance": rejected == 0,
        "all_retained_within_tolerance": True,
        "tolerances": {
            "root_m": tolerances["state_match_root_tolerance_m"],
            "joint_rad": tolerances["state_match_joint_tolerance_rad"],
            "velocity": tolerances["state_match_velocity_tolerance"],
        },
        "state_copy": False,
    }, indent=2) + "\n")
    (OUT / f"counterfactual_branch_manifest{suffix}.json").write_text(json.dumps({
        "total_branch_states": len(retained), "regime_counts": counts, "target_speed_counts": speed_counts,
        "critical_joint_names": CRITICAL_NAMES, "critical_action_indices": CRITICAL_INDICES,
        "waist_included": False, "perturbation_delta_normalized": args.delta,
        "physical_target_delta_rad": args.delta * .5,
        "horizons_steps": CFG["sensitivity"]["horizons_steps"],
        "fresh_isaac_app_per_sign": True, "teacher_gradients": 0, "state_copy": False,
    }, indent=2) + "\n")
    print(json.dumps({"matched": len(retained), "rejected": rejected, "speed_counts": speed_counts}))


if __name__ == "__main__":
    main()
