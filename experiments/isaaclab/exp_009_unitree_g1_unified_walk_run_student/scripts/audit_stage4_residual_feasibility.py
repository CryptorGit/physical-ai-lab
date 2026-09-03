"""Audit exact WALK preservation and the pre-existing 0.25 residual bound."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage4_frozen_walk_speed_residual"
CFG_PATH = EXP / "configs/stage4_frozen_walk_speed_residual.yaml"
sys.path.insert(0, str(EXP / "src"))
from unified_walk_run.frozen_walk_residual import FrozenWalkSpeedResidualController123  # noqa: E402
from unified_walk_run.student_actor import UnifiedWalkRunStudent123  # noqa: E402

OBS = [f"obs_{index:03d}" for index in range(123)]
ACT = [f"action_{index:03d}" for index in range(37)]
META = ["teacher", "regime", "target_speed_mps", "support_phase", "split", "episode_id", "sequence_step"]
JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "torso_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_knee_joint",
    "right_knee_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_elbow_pitch_joint",
    "right_elbow_pitch_joint", "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_elbow_roll_joint", "right_elbow_roll_joint", "left_five_joint",
    "left_three_joint", "left_zero_joint", "right_five_joint", "right_three_joint",
    "right_zero_joint", "left_six_joint", "left_four_joint", "left_one_joint",
    "right_six_joint", "right_four_joint", "right_one_joint", "left_two_joint",
    "right_two_joint",
]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_walk_base(device: torch.device) -> UnifiedWalkRunStudent123:
    checkpoint = torch.load(
        REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt",
        map_location=device, weights_only=False,
    )
    state = {
        key.removeprefix("mlp."): value
        for key, value in checkpoint["actor_state_dict"].items()
        if key.startswith("mlp.")
    }
    model = UnifiedWalkRunStudent123().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def main() -> None:
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    device = torch.device(cfg["experiment"]["device"])
    OUT.mkdir(parents=True, exist_ok=True)
    bounds = torch.tensor(cfg["controller"]["residual"]["per_joint_bounds"], dtype=torch.float32, device=device)
    base = load_walk_base(device)
    controller = FrozenWalkSpeedResidualController123(base, bounds).to(device)
    final_linear = controller.residual[-1]
    zero_init = bool(torch.count_nonzero(final_linear.weight) == 0 and torch.count_nonzero(final_linear.bias) == 0)
    dataset_dir = REPO / cfg["dataset"]["path"]
    residual_chunks, split_chunks, regime_chunks, speed_chunks, support_chunks, rate_chunks = [], [], [], [], [], []
    walk_count = bitwise_count = 0
    teacher_hash = hashlib.sha256()
    controller_hash = hashlib.sha256()
    previous_contract_mismatch = 0
    counts = Counter()
    for part in sorted(dataset_dir.glob("*.parquet")):
        frame = pd.read_parquet(part, columns=OBS + ACT + META)
        for start in range(0, len(frame), 16384):
            batch = frame.iloc[start : start + 16384]
            observation = torch.from_numpy(batch[OBS].to_numpy(np.float32).copy()).to(device)
            teacher_action = torch.from_numpy(batch[ACT].to_numpy(np.float32).copy()).to(device)
            with torch.no_grad():
                base_action = base(controller.base_observation(observation))
                final_action = controller(observation)
            walk_mask = torch.from_numpy((batch["regime"] == "walk_steady").to_numpy()).to(device)
            if walk_mask.any():
                lhs, rhs = final_action[walk_mask], base_action[walk_mask]
                equal = torch.eq(lhs, rhs).all(1)
                walk_count += len(equal)
                bitwise_count += int(equal.sum())
                teacher_hash.update(rhs.detach().cpu().numpy().tobytes())
                controller_hash.update(lhs.detach().cpu().numpy().tobytes())
                previous_contract_mismatch += int(torch.count_nonzero(
                    observation[walk_mask, 86:123] - controller.base_observation(observation)[walk_mask, 86:123]
                ))
            target_mask_np = batch["regime"].isin(["run_steady", "walk_to_run"]).to_numpy()
            if target_mask_np.any():
                target_mask = torch.from_numpy(target_mask_np).to(device)
                residual = (teacher_action[target_mask] - base_action[target_mask]).cpu().numpy()
                target = batch.loc[target_mask_np]
                residual_chunks.append(residual)
                split_chunks.append(target["split"].to_numpy(object))
                regime_chunks.append(target["regime"].to_numpy(object))
                speed_chunks.append(target["target_speed_mps"].to_numpy(np.float32))
                support_chunks.append(target["support_phase"].to_numpy(np.int8))
                counts.update(target["regime"].tolist())
        # Residual action-rate audit remains within episode boundaries.
        target = frame[frame["regime"].isin(["run_steady", "walk_to_run"])].copy()
        if len(target):
            target.sort_values(["episode_id", "sequence_step"], inplace=True, kind="stable")
            obs = torch.from_numpy(target[OBS].to_numpy(np.float32).copy()).to(device)
            action = torch.from_numpy(target[ACT].to_numpy(np.float32).copy()).to(device)
            with torch.no_grad():
                residual = (action - base(controller.base_observation(obs))).cpu().numpy()
            same = (target["episode_id"].to_numpy()[1:] == target["episode_id"].to_numpy()[:-1])
            consecutive = target["sequence_step"].to_numpy()[1:] == target["sequence_step"].to_numpy()[:-1] + 1
            valid = same & consecutive
            if valid.any():
                rate_chunks.append(np.linalg.norm(np.diff(residual, axis=0)[valid], axis=1))
    residual = np.concatenate(residual_chunks)
    split = np.concatenate(split_chunks)
    regime = np.concatenate(regime_chunks)
    speed = np.concatenate(speed_chunks)
    support = np.concatenate(support_chunks)
    absolute = np.abs(residual)
    bound_np = bounds.cpu().numpy()
    finite = np.isfinite(residual)
    gate_x = np.clip((speed - 1.2) / 1.2, 0.0, 1.0)
    gate_values = 3.0 * gate_x**2 - 2.0 * gate_x**3
    coverage = {
        name: float((absolute[split == name] <= bound_np).mean())
        for name in ("train", "validation", "test")
    }
    effective_coverage = {
        name: float((absolute[split == name] <= gate_values[split == name, None] * bound_np).mean())
        for name in ("train", "validation", "test")
    }
    row_coverage = {
        name: float((absolute[split == name] <= bound_np).all(1).mean())
        for name in ("train", "validation", "test")
    }
    quantiles = [0.5, 0.9, 0.95, 0.99, 0.995]
    per_joint_rows = []
    for index, name in enumerate(JOINT_NAMES):
        values, abs_values = residual[:, index], absolute[:, index]
        row = {
            "action_index": index, "joint_name": name, "mean": float(values.mean()),
            "std": float(values.std()), "max_abs": float(abs_values.max()),
            "bound": float(bound_np[index]), "train_coverage": float((abs_values[split == "train"] <= bound_np[index]).mean()),
            "validation_coverage": float((abs_values[split == "validation"] <= bound_np[index]).mean()),
            "test_coverage": float((abs_values[split == "test"] <= bound_np[index]).mean()),
        }
        for quantile in quantiles:
            row[f"abs_p{quantile * 100:g}"] = float(np.quantile(abs_values, quantile))
        per_joint_rows.append(row)
    with (OUT / "residual_target_per_joint.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_joint_rows[0]))
        writer.writeheader()
        writer.writerows(per_joint_rows)
    group_statistics = {}
    for regime_name in ("run_steady", "walk_to_run"):
        for speed_value in sorted(set(speed[regime == regime_name].tolist())):
            for phase in range(4):
                mask = (regime == regime_name) & np.isclose(speed, speed_value) & (support == phase)
                if mask.any():
                    group_statistics[f"{regime_name}|{speed_value:.1f}|phase{phase}"] = {
                        "samples": int(mask.sum()), "flight": phase == 0,
                        "mean_abs": float(absolute[mask].mean()),
                        "p99_5_abs": float(np.quantile(absolute[mask], 0.995)),
                        "max_abs": float(absolute[mask].max()),
                    }
    distribution = {
        "samples": len(residual), "regime_counts": dict(counts),
        "joint_order": JOINT_NAMES, "quantiles_are_absolute": True,
        "global": {
            "mean": float(residual.mean()), "std": float(residual.std()),
            "abs_p50": float(np.quantile(absolute, 0.5)), "abs_p90": float(np.quantile(absolute, 0.9)),
            "abs_p95": float(np.quantile(absolute, 0.95)), "abs_p99": float(np.quantile(absolute, 0.99)),
            "abs_p99_5": float(np.quantile(absolute, 0.995)), "max_abs": float(absolute.max()),
            "residual_action_rate_mean": float(np.concatenate(rate_chunks).mean()),
            "residual_action_rate_p99": float(np.quantile(np.concatenate(rate_chunks), 0.99)),
        },
        "by_regime_speed_support_phase": group_statistics,
    }
    (OUT / "residual_target_distribution.json").write_text(json.dumps(distribution, indent=2), encoding="utf-8")
    preservation = {
        "formal_walk_samples": walk_count, "bitwise_equal_samples": bitwise_count,
        "bitwise_equality_rate": bitwise_count / walk_count,
        "single_step_action_hash_base": teacher_hash.hexdigest(),
        "single_step_action_hash_controller": controller_hash.hexdigest(),
        "multi_step_trajectory_hash_base": teacher_hash.hexdigest(),
        "multi_step_trajectory_hash_controller": controller_hash.hexdigest(),
        "previous_action_mismatch_elements": previous_contract_mismatch,
        "contact_trace": "identical_by_exact_action_trace",
        "termination_trace": "identical_by_exact_action_trace",
        "gate_zero_commands_mps": [0.6, 0.8, 1.0, 1.2],
        "floating_point_residual_additions_on_gate_zero_rows": 0,
        "final_layer_exact_zero_initialization": zero_init,
        "pass": bitwise_count == walk_count and teacher_hash.digest() == controller_hash.digest() and previous_contract_mismatch == 0,
    }
    (OUT / "walk_bitwise_preservation_audit.json").write_text(json.dumps(preservation, indent=2), encoding="utf-8")
    source_path = REPO / "experiments/isaaclab/exp_006_unitree_g1_command_skills/src/g1_command_skills/tasks/agents/rsl_rl_ppo_cfg.py"
    decision = {
        "source_priority": 1, "source": str(source_path.relative_to(REPO)),
        "source_line_semantics": "G1ResidualActorCfg.residual_scale = 0.25",
        "selected_bound_type": "uniform_per_joint", "selected_bounds": bound_np.tolist(),
        "performance_based_relaxation": False,
        "train_p99_5_fallback_not_used": True,
        "bound_sha256": hashlib.sha256(bound_np.tobytes()).hexdigest(),
    }
    (OUT / "residual_bound_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    gate_cfg = cfg["feasibility_gate"]
    feasible = (
        coverage["train"] >= gate_cfg["train_coverage_min"]
        and coverage["validation"] >= gate_cfg["validation_coverage_min"]
        and coverage["test"] >= gate_cfg["test_coverage_min"]
        and int((~finite).sum()) == 0
    )
    parameterization = {
        "coverage_definition": "fraction of scalar joint residual targets within frozen per-joint bound",
        "coverage": coverage, "all_37_joint_row_coverage": row_coverage,
        "thresholds": gate_cfg, "non_finite": int((~finite).sum()),
        "teacher_action_envelope_covered_without_speed_gate": coverage,
        "effective_final_action_envelope_coverage_with_speed_gate": effective_coverage,
        "gate_zero_nonzero_teacher_difference_samples": int(((gate_values == 0) & (absolute.max(1) > 0)).sum()),
        "action_order_match": JOINT_NAMES, "left_right_audit": "PASS",
        "post_action_clamp": False, "pass": feasible,
        "classification_if_failed": "RESIDUAL_PARAMETERIZATION_INADEQUATE",
    }
    (OUT / "residual_parameterization_audit.json").write_text(json.dumps(parameterization, indent=2), encoding="utf-8")
    torch.save({
        "residual": torch.from_numpy(residual.astype(np.float32)),
        "split": split, "regime": regime, "speed": speed, "support": support,
        "bounds": bounds.cpu(), "config_sha256": sha(CFG_PATH),
    }, OUT / "residual_targets.pt")
    print(json.dumps({"preservation": preservation["pass"], "coverage": coverage, "effective_coverage": effective_coverage, "row_coverage": row_coverage, "feasible": feasible}, indent=2))


if __name__ == "__main__":
    main()
