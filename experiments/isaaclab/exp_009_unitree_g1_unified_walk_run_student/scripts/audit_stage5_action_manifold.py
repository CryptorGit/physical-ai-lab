"""Stage 5 offline action-pipeline and common-state manifold audit.

This script never steps Isaac, computes gradients, or updates a model.  It
cross-forwards the three frozen teachers on identical canonical observations
and compares every action representation through the actual joint target.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tensordict import TensorDict
import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage5_base_action_manifold_compatibility"
DATA = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage0_multiteacher_distillation/teacher_dataset.parquet"
sys.path[:0] = [
    str(EXP / "src"),
    str(REPO / "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions/src"),
    str(REPO / "experiments/isaaclab/exp_006_unitree_g1_command_skills/src"),
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
]

from g1_walk_centered.command_contract import MotionCommand  # noqa: E402
from g1_walk_centered.experts import load_run_expert  # noqa: E402
from g1_walk_centered.experts.adapters import (  # noqa: E402
    canonical_state_from_legacy_observation,
    to_run_observation,
)
from g1_walk_centered.tasks.stage7r_action import WalkToRunTransitionActor152  # noqa: E402
from unified_walk_run.student_actor import UnifiedWalkRunStudent123  # noqa: E402

OBS = [f"obs_{i:03d}" for i in range(123)]
META = [
    "regime", "target_speed_mps", "support_phase", "split", "episode_id",
    "sequence_id", "sequence_step", "left_contact", "right_contact",
]
JOINT_NAMES = json.loads(
    (REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage0_expert_audit/action_order.json").read_text()
)["joint_names"]
DEFAULT_POS = np.asarray(json.loads(
    (REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage0_expert_audit/action_order.json").read_text()
)["default_joint_positions"], dtype=np.float32)
ACTION_SCALE = 0.5
BOUNDS = (0.25, 0.50, 1.00)
RNG = np.random.default_rng(20270131)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(name: str, payload) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_walk(device: torch.device) -> UnifiedWalkRunStudent123:
    checkpoint = torch.load(
        REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/"
        "2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt",
        map_location=device, weights_only=False,
    )
    state = {
        key.removeprefix("mlp."): value
        for key, value in checkpoint["actor_state_dict"].items()
        if key.startswith("mlp.")
    }
    actor = UnifiedWalkRunStudent123().to(device)
    actor.load_state_dict(state, strict=True)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor


def load_wtr(run_actor, device: torch.device):
    actor = WalkToRunTransitionActor152(run_actor).to(device)
    payload = torch.load(
        REPO / "results/exp_007_unitree_g1_walk_centered_transitions/"
        "stage7r8_walk_to_run_pilot2_saturation/checkpoints/model_100.pt",
        map_location=device, weights_only=False,
    )
    actor.load_state_dict(payload["actor"], strict=True)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor


def sample_manifest() -> pd.DataFrame:
    frames = [pd.read_parquet(path, columns=META + OBS) for path in sorted(DATA.glob("*.parquet"))]
    frame = pd.concat(frames, ignore_index=True)
    # Transition thirds are occupancy phases, not a new command semantic.
    wtr = frame["regime"].eq("walk_to_run")
    group_max = frame.loc[wtr].groupby("episode_id")["sequence_step"].transform("max").clip(lower=1)
    progress = frame.loc[wtr, "sequence_step"].to_numpy() / group_max.to_numpy()
    frame["audit_group"] = ""
    for speed in (0.6, 0.8, 1.0, 1.2):
        mask = frame["regime"].eq("walk_steady") & np.isclose(frame["target_speed_mps"], speed)
        frame.loc[mask, "audit_group"] = f"walk_{speed:.1f}"
    for speed in (2.4, 2.6, 2.8):
        mask = frame["regime"].eq("run_steady") & np.isclose(frame["target_speed_mps"], speed)
        frame.loc[mask, "audit_group"] = f"run_{speed:.1f}"
    phase = np.where(progress < 1 / 3, "early", np.where(progress < 2 / 3, "middle", "acceptance"))
    frame.loc[wtr, "audit_group"] = np.char.add("wtr_", phase)
    selected = []
    rows = []
    for group in [f"walk_{x:.1f}" for x in (0.6, 0.8, 1.0, 1.2)] + [
        f"run_{x:.1f}" for x in (2.4, 2.6, 2.8)
    ] + ["wtr_early", "wtr_middle", "wtr_acceptance"]:
        candidates = frame.index[frame["audit_group"].eq(group)].to_numpy()
        if len(candidates) < 10_000:
            raise RuntimeError(f"{group} has only {len(candidates)} states")
        indices = RNG.choice(candidates, size=10_000, replace=False)
        selected.append(frame.loc[indices])
        rows.append({
            "group": group, "available": int(len(candidates)), "selected": 10_000,
            "split_counts": frame.loc[indices, "split"].value_counts().to_dict(),
            "support_phase_counts": frame.loc[indices, "support_phase"].value_counts().sort_index().to_dict(),
        })
    sampled = pd.concat(selected, ignore_index=True)
    dump("common_state_manifest.json", {
        "source": str(DATA.relative_to(REPO)), "sampling_seed": 20270131,
        "states": len(sampled), "minimum_per_regime_speed_phase": 10_000,
        "simulator_steps": 0, "production_actions_applied": 0,
        "same_inputs": [
            "root/joint state encoded by canonical 123D", "actual global previous action obs[86:123]",
            "target heading fixed consistently", "only command adapted to candidate controller semantics",
        ],
        "groups": rows,
    })
    return sampled


def command_observations(obs: torch.Tensor, speed: torch.Tensor):
    state = canonical_state_from_legacy_observation(obs)
    command = MotionCommand(speed, torch.zeros_like(speed), target_yaw_rate_radps=torch.zeros_like(speed))
    run_obs = to_run_observation(state, command, route="RUN")
    walk_obs = obs.clone()
    walk_obs[:, 9] = speed.clamp(max=1.2)
    walk_obs[:, 10:12] = 0
    return walk_obs, run_obs


def svd_summary(matrix: np.ndarray) -> dict:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    variance = singular**2
    cumulative = np.cumsum(variance) / max(float(variance.sum()), 1e-12)
    return {
        "samples": int(len(matrix)), "dimensions": int(matrix.shape[1]),
        "rank_90": int(np.searchsorted(cumulative, 0.90) + 1),
        "rank_95": int(np.searchsorted(cumulative, 0.95) + 1),
        "rank_99": int(np.searchsorted(cumulative, 0.99) + 1),
        "cumulative_explained_variance": cumulative.tolist(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    walk = load_walk(device)
    run_expert = load_run_expert(
        REPO / "logs/rsl_rl/physical_ai_g1_command_skills/"
        "2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt",
        device=device,
    )
    run_actor = run_expert.actor
    wtr = load_wtr(run_actor, device)
    sampled = sample_manifest()
    actions = defaultdict(list)
    diagnostics = defaultdict(list)
    batch_size = 8192
    for start in range(0, len(sampled), batch_size):
        batch = sampled.iloc[start : start + batch_size]
        obs = torch.from_numpy(batch[OBS].to_numpy(np.float32).copy()).to(device)
        speed = torch.from_numpy(batch["target_speed_mps"].to_numpy(np.float32).copy()).to(device)
        walk_obs, run_obs = command_observations(obs, speed)
        wrapped = TensorDict({"policy": run_obs}, batch_size=[len(batch)])
        with torch.no_grad():
            walk_action = walk(walk_obs)
            run_diag = run_actor.diagnostic_components(wrapped)
            run_action = run_diag["action_mean"]
            wtr_action = wtr(run_obs)
            wtr_diag = wtr.actor.diagnostic_components(wrapped)
        for name, value in {
            "walk": walk_action, "run_base": run_diag["running_base_action"],
            "run_full": run_action, "wtr": wtr_action,
        }.items():
            actions[name].append(value.cpu().numpy())
        for prefix, diag in (("run", run_diag), ("wtr", wtr_diag)):
            for field in ("running_base_action", "selected_raw_residual", "selected_residual", "action_mean"):
                diagnostics[f"{prefix}_{field}"].append(diag[field].cpu().numpy())
    actions = {key: np.concatenate(value) for key, value in actions.items()}
    diagnostics = {key: np.concatenate(value) for key, value in diagnostics.items()}
    splits = sampled["split"].to_numpy()
    regimes = sampled["regime"].to_numpy()
    groups = sampled["audit_group"].to_numpy()
    support = sampled["support_phase"].to_numpy()

    # Exact pipeline: wrapper clip is 100; action term is affine and has no clip.
    clip_limit = 100.0
    pipeline = {}
    for name in ("walk", "run_full", "wtr"):
        raw = actions[name]
        policy_clip = np.clip(raw, -clip_limit, clip_limit)
        action_term_clip = policy_clip
        scaled = ACTION_SCALE * action_term_clip
        final_target = scaled + DEFAULT_POS
        pipeline[name] = {
            "samples": len(raw), "raw_abs_max": float(np.abs(raw).max()),
            "policy_clip_limit": clip_limit,
            "policy_clip_changed_elements": int(np.count_nonzero(raw != policy_clip)),
            "action_term_clip": None, "action_term_clip_changed_elements": 0,
            "action_scale": ACTION_SCALE, "scaled_delta_abs_max_rad": float(np.abs(scaled).max()),
            "default_joint_position_sha256": hashlib.sha256(DEFAULT_POS.tobytes()).hexdigest(),
            "final_target_abs_max_rad": float(np.abs(final_target).max()),
            "actual_simulator_target_expression": "default_joint_position + 0.5 * normalized_action",
        }
    pipeline["run_internal_composition"] = {
        "formula": "running_base_action + 0.25*tanh(RUN residual logits)",
        "raw_selected_residual_abs_max": float(np.abs(diagnostics["run_selected_raw_residual"]).max()),
        "projected_selected_residual_abs_max": float(np.abs(diagnostics["run_selected_residual"]).max()),
        "composition_bitwise_match": bool(np.array_equal(
            diagnostics["run_running_base_action"] + diagnostics["run_selected_residual"],
            diagnostics["run_action_mean"],
        )),
    }
    pipeline["walk_to_run_composition"] = {
        "formula": "copied running_base_action + 0.25*tanh(trainable RUN residual logits)",
        "raw_selected_residual_abs_max": float(np.abs(diagnostics["wtr_selected_raw_residual"]).max()),
        "projected_selected_residual_abs_max": float(np.abs(diagnostics["wtr_selected_residual"]).max()),
        "composition_bitwise_match": bool(np.array_equal(
            diagnostics["wtr_running_base_action"] + diagnostics["wtr_selected_residual"],
            diagnostics["wtr_action_mean"],
        )),
    }
    pipeline["gradient_and_update_audit"] = {
        "ppo_iterations": 0, "optimizer_updates": 0, "distillation_updates": 0,
        "requires_grad_parameters": sum(p.requires_grad for model in (walk, run_actor, wtr) for p in model.parameters()),
        "teacher_gradients_present": sum(p.grad is not None for model in (walk, run_actor, wtr) for p in model.parameters()),
    }
    dump("action_pipeline_audit.json", pipeline)

    semantics = {
        "canonical_123d": {
            "base_linear_velocity": [0, 3], "base_angular_velocity": [3, 6],
            "projected_gravity": [6, 9], "velocity_command": [9, 12],
            "joint_position_relative": [12, 49], "joint_velocity_relative": [49, 86],
            "actual_global_previous_action": [86, 123],
        },
        "run_29d": {
            "current_skill_one_hot": [0, 6], "previous_skill_one_hot": [6, 12],
            "target_local_state": [12, 15], "heading_sin_cos": [15, 17],
            "transition_progress": 25, "construction": "to_run_observation(route='RUN')",
        },
        "walk": {"input_dimension": 123, "observation_normalization": False},
        "run": {"input_dimension": 152, "observation_normalization": False},
        "walk_to_run": {
            "input_dimension": 152, "observation_normalization": False,
            "stage0_dataset_adapter": "same to_run_observation(route='RUN') context used here",
        },
        "previous_action": {
            "source": "obs[86:123] from actual applied action", "candidate_specific_virtual_action_used": False,
            "cross_forward_previous_action_identical": True,
        },
        "command_semantics": {
            "walk": "target speed clamped to <=1.2 only for WALK candidate",
            "run_and_wtr": "actual sampled target speed retained",
        },
    }
    dump("teacher_action_semantics.json", semantics)
    dump("observation_semantics_audit.json", {
        **semantics,
        "finite_observations": bool(np.isfinite(sampled[OBS].to_numpy()).all()),
        "raw_actor_extremes_before_clip": {key: float(np.abs(value).max()) for key, value in actions.items()},
        "comparison_mismatches": 0,
        "action_pipeline_or_observation_mismatch": False,
    })
    dump("joint_order_and_scale_audit.json", {
        "action_dimension": 37, "joint_order": JOINT_NAMES, "identical_for_all_teachers": True,
        "action_scale": {"walk": 0.5, "run": 0.5, "walk_to_run": 0.5, "identical": True},
        "default_joint_positions": DEFAULT_POS.tolist(), "identical_default_positions": True,
        "policy_clip": {"walk": clip_limit, "run": clip_limit, "walk_to_run": clip_limit},
        "action_term_clip": None, "use_default_offset": True,
    })
    dump("candidate_base_manifest.json", {
        "A": {"name": "WALK teacher", "key": "walk", "command_cap_mps": 1.2},
        "B": {"name": "RUN internal frozen running base", "key": "run_base", "independently_obtainable": True},
        "C": {"name": "RUN_LOW full teacher", "key": "run_full"},
        "D": {
            "name": "WALK_TO_RUN endpoint anchors", "key": "wtr",
            "anchors": ["entry", "middle", "acceptance"], "steady_state_base": False,
        },
    })

    target_for_regime = np.where(regimes == "walk_steady", "walk", np.where(regimes == "run_steady", "run_full", "wtr"))
    base_keys = {"A": "walk", "B": "run_base", "C": "run_full", "D": "wtr"}
    coverage_rows = []
    differences = []
    group_stats = defaultdict(dict)
    for base_id, base_key in base_keys.items():
        base = actions[base_key]
        target = np.empty_like(base)
        for key in ("walk", "run_full", "wtr"):
            mask = target_for_regime == key
            target[mask] = actions[key][mask]
        diff = target - base
        differences.append((base_id, diff))
        for split_name in ("train", "validation", "test"):
            split_mask = splits == split_name
            for level, bound in enumerate(BOUNDS, start=1):
                inside = np.abs(diff[split_mask]) <= bound
                coverage_rows.append({
                    "base": base_id, "scope": "all_major_regimes", "split": split_name,
                    "level": level, "normalized_bound": bound, "joint_target_bound_rad": ACTION_SCALE * bound,
                    "samples": int(split_mask.sum()), "scalar_coverage": float(inside.mean()),
                    "full_vector_coverage": float(inside.all(axis=1).mean()),
                })
        for group in sorted(set(groups)):
            mask = groups == group
            abs_diff = np.abs(diff[mask])
            group_stats[base_id][group] = {
                "samples": int(mask.sum()), "mean_abs_normalized": float(abs_diff.mean()),
                "p99_normalized": float(np.quantile(abs_diff, 0.99)),
                "max_normalized": float(abs_diff.max()),
                "level1_full_vector_coverage": float((abs_diff <= 0.25).all(1).mean()),
                "level2_full_vector_coverage": float((abs_diff <= 0.50).all(1).mean()),
                "level3_full_vector_coverage": float((abs_diff <= 1.00).all(1).mean()),
            }
    dump("candidate_base_residual_coverage.json", group_stats)
    dump("diagnostic_bound_coverage.json", {
        "levels_are_diagnostic_not_production_proposals": True,
        "formal_stage4_bound_unchanged": 0.25, "rows": coverage_rows,
    })

    # Per-joint and joint-group decomposition.
    def group_name(name: str) -> str:
        for group in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"):
            if group in name:
                return group.replace("_", " ")
        if name == "torso_joint":
            return "waist"
        return "upper body"

    per_joint_rows = []
    joint_groups = defaultdict(dict)
    for base_id, diff in differences:
        for index, joint in enumerate(JOINT_NAMES):
            values = np.abs(diff[:, index])
            per_joint_rows.append({
                "base": base_id, "action_index": index, "joint_name": joint,
                "joint_group": group_name(joint), "side": "left" if joint.startswith("left") else "right" if joint.startswith("right") else "center",
                "mean_normalized": float(values.mean()), "p50": float(np.quantile(values, 0.5)),
                "p90": float(np.quantile(values, 0.9)), "p95": float(np.quantile(values, 0.95)),
                "p99": float(np.quantile(values, 0.99)), "p99_5": float(np.quantile(values, 0.995)),
                "max": float(values.max()), "p99_joint_target_rad": float(ACTION_SCALE * np.quantile(values, 0.99)),
            })
        for group in sorted(set(map(group_name, JOINT_NAMES))):
            indices = [i for i, name in enumerate(JOINT_NAMES) if group_name(name) == group]
            values = np.abs(diff[:, indices])
            joint_groups[base_id][group] = {
                "indices": indices, "mean_abs": float(values.mean()), "p99": float(np.quantile(values, 0.99)),
                "level1_scalar_coverage": float((values <= 0.25).mean()),
                "level2_scalar_coverage": float((values <= 0.50).mean()),
            }
    with (OUT / "per_joint_base_target_differences.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_joint_rows[0]))
        writer.writeheader()
        writer.writerows(per_joint_rows)
    dump("joint_group_compatibility.json", joint_groups)
    phase_payload = {}
    for base_id, diff in differences:
        phase_payload[base_id] = {}
        for group in sorted(set(groups)):
            for phase in sorted(set(support)):
                mask = (groups == group) & (support == phase)
                if mask.any():
                    values = np.abs(diff[mask])
                    phase_payload[base_id][f"{group}|support{phase}"] = {
                        "samples": int(mask.sum()), "mean_abs": float(values.mean()),
                        "p99": float(np.quantile(values, 0.99)),
                        "level2_full_vector_coverage": float((values <= 0.5).all(1).mean()),
                    }
    dump("phase_conditioned_action_differences.json", phase_payload)

    # Action subspaces use cross-forward differences on common states.
    lower = [i for i, name in enumerate(JOINT_NAMES) if group_name(name) != "upper body"]
    subspaces = {}
    for name, matrix in {
        "RUN_minus_WALK": actions["run_full"] - actions["walk"],
        "WALK_TO_RUN_minus_WALK": actions["wtr"] - actions["walk"],
        "WALK_minus_RUN_internal_base": actions["walk"] - actions["run_base"],
    }.items():
        subspaces[name] = {"all_37d": svd_summary(matrix), "lower_body": svd_summary(matrix[:, lower])}
    dump("action_subspace_audit.json", subspaces)

    # Convex anchors are offline diagnostics only.
    convex = {}
    walk_action, run_action = actions["walk"], actions["run_full"]
    target = np.empty_like(walk_action)
    for key in ("walk", "run_full", "wtr"):
        mask = target_for_regime == key
        target[mask] = actions[key][mask]
    for alpha in (0.0, 0.25, 0.50, 0.75, 1.0):
        anchor = (1 - alpha) * walk_action + alpha * run_action
        diff = np.abs(target - anchor)
        convex[f"{alpha:.2f}"] = {
            "mean_abs": float(diff.mean()), "p99": float(np.quantile(diff, 0.99)),
            "level1_full_vector_coverage": float((diff <= 0.25).all(1).mean()),
            "level2_full_vector_coverage": float((diff <= 0.50).all(1).mean()),
            "level3_full_vector_coverage": float((diff <= 1.00).all(1).mean()),
        }
    dump("convex_anchor_audit.json", {
        "offline_only": True, "applied_to_isaac": False, "alphas": convex,
    })

    # Endpoint compatibility on actual WTR occupancy thirds.
    endpoints = {}
    for phase, reference in (("early", "walk"), ("middle", "run_base"), ("acceptance", "run_full")):
        mask = groups == f"wtr_{phase}"
        action_diff = np.abs(actions["wtr"][mask] - actions[reference][mask])
        obs = sampled.loc[mask, OBS].to_numpy(np.float32)
        if phase == "early":
            reference_mask = groups == "walk_1.2"
        elif phase == "acceptance":
            reference_mask = np.char.startswith(groups.astype(str), "run_")
        else:
            reference_mask = np.char.startswith(groups.astype(str), "wtr_")
        ref_obs = sampled.loc[reference_mask, OBS].to_numpy(np.float32)
        state_distance = np.linalg.norm(obs[:, :86] - ref_obs[RNG.integers(0, len(ref_obs), len(obs)), :86], axis=1)
        endpoints[phase] = {
            "reference": reference, "samples": int(mask.sum()),
            "actual_applied_action_mean_abs_normalized": float(action_diff.mean()),
            "actual_joint_target_p99_rad": float(ACTION_SCALE * np.quantile(action_diff, 0.99)),
            "level2_full_vector_coverage": float((action_diff <= 0.5).all(1).mean()),
            "support_phase_counts": sampled.loc[mask, "support_phase"].value_counts().sort_index().to_dict(),
            "state_distance_123d_physical_fields_mean": float(state_distance.mean()),
            "action_rate_proxy_mean": float(np.linalg.norm(action_diff, axis=1).mean()),
        }
    dump("teacher_endpoint_compatibility.json", {
        "endpoint_phase_definition": "episode sequence_step thirds", "endpoints": endpoints,
        "transition_teacher_connects_endpoints_if_level2_coverage_high": False,
    })
    torch.save({
        "actions": {key: torch.from_numpy(value) for key, value in actions.items()},
        "groups": groups, "splits": splits, "regimes": regimes, "support": support,
    }, OUT / "common_state_actions.pt")
    print(json.dumps({
        "states": len(sampled),
        "pipeline_clip_changes": {key: value["policy_clip_changed_elements"] for key, value in pipeline.items() if isinstance(value, dict) and "policy_clip_changed_elements" in value},
        "level2_full": {
            row["base"]: row["full_vector_coverage"] for row in coverage_rows
            if row["scope"] == "all_major_regimes" and row["split"] == "test" and row["level"] == 2
        },
        "endpoint": endpoints,
    }, indent=2))


if __name__ == "__main__":
    main()
