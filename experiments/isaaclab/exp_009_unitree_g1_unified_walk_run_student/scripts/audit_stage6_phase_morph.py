"""Stage 6 oracle base-morph and diagnostic alpha identifiability audit.

No action generated here is applied to Isaac.  The only trainable objects are
small diagnostic regressors for oracle alpha; all locomotion teachers remain
frozen and are cross-forwarded under inference mode.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from tensordict import TensorDict
import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage6_phase_conditioned_base_morph_feasibility"
DATA = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage0_multiteacher_distillation/teacher_dataset.parquet"
sys.path[:0] = [
    str(EXP / "src"),
    str(REPO / "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions/src"),
    str(REPO / "experiments/isaaclab/exp_006_unitree_g1_command_skills/src"),
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
]

from g1_walk_centered.command_contract import MotionCommand  # noqa: E402
from g1_walk_centered.experts import load_run_expert  # noqa: E402
from g1_walk_centered.experts.adapters import canonical_state_from_legacy_observation, to_run_observation  # noqa: E402
from g1_walk_centered.tasks.stage7r_action import WalkToRunTransitionActor152  # noqa: E402
from unified_walk_run.student_actor import UnifiedWalkRunStudent123  # noqa: E402

OBS = [f"obs_{i:03d}" for i in range(123)]
META = [
    "regime", "target_speed_mps", "support_phase", "split", "episode_id", "sequence_id",
    "sequence_step", "left_contact", "right_contact", "left_foot_air_time", "right_foot_air_time",
]
BOUND = 0.25
ACTION_SCALE = 0.5
RNG = np.random.default_rng(20270211)
JOINT_NAMES = json.loads((
    REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage0_expert_audit/action_order.json"
).read_text())["joint_names"]


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_walk(device: torch.device):
    checkpoint = torch.load(
        REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/"
        "2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt",
        map_location=device, weights_only=False,
    )
    state = {k.removeprefix("mlp."): v for k, v in checkpoint["actor_state_dict"].items() if k.startswith("mlp.")}
    actor = UnifiedWalkRunStudent123().to(device)
    actor.load_state_dict(state, strict=True)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor


def load_wtr(run_actor, device):
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


def build_common_states() -> pd.DataFrame:
    frames = [pd.read_parquet(path, columns=META + OBS) for path in sorted(DATA.glob("*.parquet"))]
    frame = pd.concat(frames, ignore_index=True)
    frame["audit_group"] = ""
    selected = []
    manifest = []
    for speed in (0.6, 0.8, 1.0, 1.2):
        mask = frame["regime"].eq("walk_steady") & np.isclose(frame["target_speed_mps"], speed)
        indices = RNG.choice(frame.index[mask].to_numpy(), 10_000, replace=False)
        part = frame.loc[indices].copy()
        part["audit_group"] = f"walk_{speed:.1f}"
        selected.append(part)
        manifest.append({"group": f"walk_{speed:.1f}", "selected": 10_000, "available": int(mask.sum())})
    for speed in (2.4, 2.6, 2.8):
        mask = frame["regime"].eq("run_steady") & np.isclose(frame["target_speed_mps"], speed)
        indices = RNG.choice(frame.index[mask].to_numpy(), 10_000, replace=False)
        part = frame.loc[indices].copy()
        part["audit_group"] = f"run_{speed:.1f}"
        selected.append(part)
        manifest.append({"group": f"run_{speed:.1f}", "selected": 10_000, "available": int(mask.sum())})
    wtr = frame[frame["regime"].eq("walk_to_run")].copy()
    episodes = wtr.groupby("episode_id", sort=True).agg(split=("split", "first"), target=("target_speed_mps", "max"))
    picked = []
    for split_name, count in (("train", 210), ("validation", 45), ("test", 45)):
        candidates = episodes.index[episodes["split"].eq(split_name)].to_numpy()
        # Balanced target selection without changing physical phase occupancy.
        by_target = []
        for target in (2.6, 2.8):
            values = episodes.index[(episodes["split"].eq(split_name)) & np.isclose(episodes["target"], target)].to_numpy()
            by_target.extend(RNG.choice(values, count // 2, replace=False).tolist())
        if len(by_target) < count:
            remaining = np.setdiff1d(candidates, np.asarray(by_target))
            by_target.extend(RNG.choice(remaining, count - len(by_target), replace=False).tolist())
        picked.extend(by_target)
    transition = wtr[wtr["episode_id"].isin(picked)].copy()
    if len(transition) != 30_000:
        raise RuntimeError(f"expected 30,000 complete transition states, got {len(transition)}")
    progress = transition["sequence_step"].to_numpy(np.float32) / 99.0
    names = np.asarray(["entry", "early", "middle", "late", "acceptance"], dtype=object)
    phase_index = np.minimum((progress * 5).astype(int), 4)
    transition["audit_group"] = np.char.add("wtr_", names[phase_index].astype(str))
    selected.append(transition)
    for phase in names:
        manifest.append({"group": f"wtr_{phase}", "selected": int((transition["audit_group"] == f"wtr_{phase}").sum())})
    sampled = pd.concat(selected, ignore_index=True)
    # Runtime-computable last landing proxy: contact plus smaller current air time.
    left, right = sampled["left_contact"].to_numpy(bool), sampled["right_contact"].to_numpy(bool)
    la, ra = sampled["left_foot_air_time"].to_numpy(), sampled["right_foot_air_time"].to_numpy()
    sampled["last_landing_foot"] = np.where(left & ~right, 0, np.where(right & ~left, 1, np.where(la <= ra, 0, 1)))
    dump("common_state_manifest.json", {
        "source": str(DATA.relative_to(REPO)), "states": len(sampled), "sampling_seed": 20270211,
        "walk_states": 40_000, "run_states": 30_000, "walk_to_run_states": 30_000,
        "complete_wtr_episodes": 300, "groups": manifest,
        "new_replay_required": False, "isaac_steps": 0, "state_setter": 0, "teleport": 0,
    })
    return sampled


def cross_forward(frame: pd.DataFrame, device):
    walk = load_walk(device)
    run = load_run_expert(
        REPO / "logs/rsl_rl/physical_ai_g1_command_skills/"
        "2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt", device=device,
    )
    wtr = load_wtr(run.actor, device)
    output = defaultdict(list)
    for start in range(0, len(frame), 8192):
        batch = frame.iloc[start : start + 8192]
        obs = torch.from_numpy(batch[OBS].to_numpy(np.float32).copy()).to(device)
        speed = torch.from_numpy(batch["target_speed_mps"].to_numpy(np.float32).copy()).to(device)
        state = canonical_state_from_legacy_observation(obs)
        command = MotionCommand(speed, torch.zeros_like(speed), target_yaw_rate_radps=torch.zeros_like(speed))
        walk_obs = obs.clone()
        walk_obs[:, 9] = speed.clamp(max=1.2)
        walk_obs[:, 10:12] = 0
        obs152 = to_run_observation(state, command, route="RUN")
        wrapped = TensorDict({"policy": obs152}, batch_size=[len(batch)])
        with torch.inference_mode():
            walk_action = walk(walk_obs)
            run_diag = run.actor.diagnostic_components(wrapped)
            run_action = run_diag["action_mean"]
            wtr_action = wtr(obs152)
        for key, value in {
            "walk": walk_action, "run_base": run_diag["running_base_action"],
            "run_full": run_action, "wtr": wtr_action,
        }.items():
            output[key].append(value.cpu().numpy())
    actions = {key: np.concatenate(value) for key, value in output.items()}
    dump("teacher_action_manifest.json", {
        "action_semantics": "actual normalized action before environment scale",
        "joint_target_semantics": "default_joint_position + 0.5 * actual_normalized_action",
        "action_dimension": 37, "joint_order": JOINT_NAMES,
        "teachers": {
            "walk": {"sha256": "9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa"},
            "run_full": {"sha256": "60675a79187ef00935069f3bf85293e72477fa1d408b80d3325232d805b0d266"},
            "walk_to_run": {"sha256": "d94a94409ed9651734ae8ebc85313b827a809de139135115e833c76ca3bb9fd0"},
            "run_internal_base": {"source": "RUN_LOW actor diagnostic_components.running_base_action"},
        },
        "parameters_frozen": True, "teacher_gradients": 0,
    })
    return actions, (walk, run.actor, wtr)


def oracle_alpha(target, walk, run_base, indices=None):
    if indices is not None:
        target, walk, run_base = target[:, indices], walk[:, indices], run_base[:, indices]
    direction = run_base - walk
    numerator = ((target - walk) * direction).sum(1)
    denominator = (direction * direction).sum(1) + 1e-8
    return np.clip(numerator / denominator, 0.0, 1.0)


def coverage_payload(residual, mask):
    values = residual[mask]
    absolute = np.abs(values)
    critical = [0, 1, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20]
    return {
        "samples": int(mask.sum()), "scalar_coverage": float((absolute <= BOUND).mean()),
        "full_vector_coverage": float((absolute <= BOUND).all(1).mean()),
        "contact_critical_scalar_coverage": float((absolute[:, critical] <= BOUND).mean()),
        "joint_target_residual_p99_rad": float(ACTION_SCALE * np.quantile(absolute, 0.99)),
        "per_joint": {
            JOINT_NAMES[j]: {
                "p95": float(np.quantile(absolute[:, j], 0.95)), "p99": float(np.quantile(absolute[:, j], 0.99)),
                "p99_5": float(np.quantile(absolute[:, j], 0.995)), "max": float(absolute[:, j].max()),
            } for j in range(37)
        },
    }


class AlphaMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x).squeeze(-1)


def feature_conditions(frame):
    obs = frame[OBS].to_numpy(np.float32)
    requested = frame["target_speed_mps"].to_numpy(np.float32)
    actual = obs[:, 0]
    is_wtr = frame["regime"].eq("walk_to_run").to_numpy()
    progress = np.where(is_wtr, frame["sequence_step"].to_numpy(np.float32) / 99.0, np.where(frame["regime"].eq("run_steady"), 1.0, 0.0))
    elapsed = progress * 1.98
    remaining = np.maximum(1.98 - elapsed, 0)
    base = np.column_stack((requested, actual)).astype(np.float32)
    condition_b = obs
    condition_c = np.column_stack((obs, requested, progress, elapsed, remaining)).astype(np.float32)
    support = np.eye(4, dtype=np.float32)[frame["support_phase"].to_numpy(np.int64)]
    landing = np.eye(2, dtype=np.float32)[frame["last_landing_foot"].to_numpy(np.int64)]
    phase = np.column_stack((
        frame["left_contact"].to_numpy(np.float32), frame["right_contact"].to_numpy(np.float32),
        frame["left_foot_air_time"].to_numpy(np.float32), frame["right_foot_air_time"].to_numpy(np.float32),
        landing, support, (frame["support_phase"].to_numpy() == 0).astype(np.float32),
    ))
    condition_d = np.column_stack((condition_c, phase)).astype(np.float32)
    return {"A_speed_only": base, "B_123D": condition_b, "C_123D_transition_scalars": condition_c, "D_explicit_phase": condition_d}


def monotonic_metrics(frame, alpha):
    rows = []
    episode_metrics = []
    wtr_indices = np.flatnonzero(frame["regime"].eq("walk_to_run").to_numpy())
    subset = frame.iloc[wtr_indices].copy()
    subset["alpha"] = alpha[wtr_indices]
    for episode, group in subset.groupby("episode_id", sort=True):
        order = group["sequence_step"].to_numpy().argsort()
        values = group["alpha"].to_numpy()[order]
        steps = group["sequence_step"].to_numpy()[order]
        target_endpoint = float(group["target_speed_mps"].max())
        derivative = np.diff(values, prepend=values[0])
        second = np.diff(derivative, prepend=derivative[0])
        for i, row_index in enumerate(group.index.to_numpy()[order]):
            rows.append({
                "episode_id": episode, "sequence_step": int(steps[i]), "transition_progress": float(steps[i] / 99),
                "requested_speed_mps": float(frame.loc[row_index, "target_speed_mps"]),
                "target_endpoint_mps": target_endpoint,
                "support_phase": int(frame.loc[row_index, "support_phase"]),
                "last_landing_foot": int(frame.loc[row_index, "last_landing_foot"]),
                "alpha_oracle": float(values[i]), "derivative": float(derivative[i]), "second_derivative": float(second[i]),
            })
        episode_metrics.append({
            "episode_id": episode, "total_variation": float(np.abs(np.diff(values)).sum()),
            "max_one_step_jump": float(np.abs(np.diff(values)).max(initial=0)),
            "large_backward_jump_rate": float((np.diff(values) < -0.10).mean()),
            "monotonic_violation_count": int((np.diff(values) < 0).sum()),
            "largest_backward_step": float(np.diff(values).min(initial=0)),
            "start_alpha": float(values[0]), "end_alpha": float(values[-1]),
        })
    return rows, episode_metrics


def train_probes(features, labels, splits, actions, frame, oracle_residual_feasible):
    config = {
        "purpose": "diagnostic oracle-alpha identifiability only", "production": False,
        "models": {
            "linear": {"type": "ridge regression", "ridge": 1e-4},
            "small_mlp": {"layers": ["input", 128, 64, 1], "activation": "ELU", "output": "sigmoid"},
        },
        "training": {"seed": 20270212, "epochs": 20, "batch_size": 2048, "optimizer": "Adam", "learning_rate": 0.001},
        "split": "existing episode/seed/trajectory Stage 0 split 70/15/15",
        "teacher_identity_input": False,
    }
    dump("alpha_probe_config.json", config)
    if not oracle_residual_feasible:
        reason = {"status": "not_executed", "reason": "oracle_morph_inadequate"}
        dump("alpha_probe_results.json", reason)
        dump("per_feature_condition_results.json", reason)
        dump("predicted_anchor_residual_coverage.json", reason)
        return reason
    train, validation, test = splits == "train", splits == "validation", splits == "test"
    target_action = np.where(
        frame["regime"].eq("walk_steady").to_numpy()[:, None], actions["walk"],
        np.where(frame["regime"].eq("run_steady").to_numpy()[:, None], actions["run_full"], actions["wtr"]),
    )
    results = {}
    predicted_coverage = {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20270212)
    for condition, x in features.items():
        mean, std = x[train].mean(0), x[train].std(0)
        std[std < 1e-6] = 1
        z = ((x - mean) / std).astype(np.float32)
        models = {}
        # Ridge linear diagnostic.
        xtx = z[train].T @ z[train] + 1e-4 * np.eye(z.shape[1], dtype=np.float32)
        weights = np.linalg.solve(xtx, z[train].T @ labels[train])
        bias = float(labels[train].mean() - (z[train].mean(0) @ weights))
        models["linear"] = np.clip(z @ weights + bias, 0, 1)
        # Fixed small MLP diagnostic.
        model = AlphaMLP(z.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_ids = np.flatnonzero(train)
        best_state, best_val = None, float("inf")
        for _ in range(20):
            RNG.shuffle(train_ids)
            model.train()
            for start in range(0, len(train_ids), 2048):
                ids = train_ids[start : start + 2048]
                xb = torch.from_numpy(z[ids]).to(device)
                yb = torch.from_numpy(labels[ids].astype(np.float32)).to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.mean((model(xb) - yb) ** 2)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                value = torch.mean((
                    model(torch.from_numpy(z[validation]).to(device))
                    - torch.from_numpy(labels[validation].astype(np.float32)).to(device)
                ) ** 2).item()
            if value < best_val:
                best_val = value
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(z), 8192):
                predictions.append(model(torch.from_numpy(z[start : start + 8192]).to(device)).cpu().numpy())
        models["small_mlp"] = np.concatenate(predictions)
        results[condition] = {}
        predicted_coverage[condition] = {}
        for model_name, prediction in models.items():
            error = np.abs(prediction[test] - labels[test])
            y = labels[test]
            r2 = 1 - float(np.sum((prediction[test] - y) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12))
            endpoint = (
                ((frame["regime"].eq("walk_steady").to_numpy()[test]) & (prediction[test] <= 0.05))
                | ((frame["regime"].eq("run_steady").to_numpy()[test]) & (prediction[test] >= 0.95))
            )
            endpoint_mask = ~frame["regime"].eq("walk_to_run").to_numpy()[test]
            anchor = actions["walk"] + prediction[:, None] * (actions["run_base"] - actions["walk"])
            residual = target_action - anchor
            wtr_test = test & frame["regime"].eq("walk_to_run").to_numpy()
            full_coverage = float((np.abs(residual[wtr_test]) <= BOUND).all(1).mean())
            results[condition][model_name] = {
                "test_alpha_mae": float(error.mean()), "test_alpha_p95_error": float(np.quantile(error, 0.95)),
                "test_r2": r2, "endpoint_accuracy": float(endpoint[endpoint_mask].mean()),
                "wtr_bounded_residual_full_vector_coverage": full_coverage,
            }
            predicted_coverage[condition][model_name] = {
                regime: coverage_payload(residual, test & frame["regime"].eq(regime).to_numpy())
                for regime in ("walk_steady", "run_steady", "walk_to_run")
            }
    dump("alpha_probe_results.json", results)
    dump("per_feature_condition_results.json", results)
    dump("predicted_anchor_residual_coverage.json", predicted_coverage)
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    frame = build_common_states()
    actions, teacher_models = cross_forward(frame, device)
    walk, run_base = actions["walk"], actions["run_base"]
    regime = frame["regime"].to_numpy()
    target = np.where(
        (regime == "walk_steady")[:, None], walk,
        np.where((regime == "run_steady")[:, None], actions["run_full"], actions["wtr"]),
    )
    alpha = oracle_alpha(target, walk, run_base)
    anchor = walk + alpha[:, None] * (run_base - walk)
    residual = target - anchor
    distribution = pd.DataFrame({
        "state_index": np.arange(len(frame)), "episode_id": frame["episode_id"], "sequence_step": frame["sequence_step"],
        "split": frame["split"], "source_regime": regime, "audit_group": frame["audit_group"],
        "target_speed_mps": frame["target_speed_mps"], "support_phase": frame["support_phase"],
        "last_landing_foot": frame["last_landing_foot"], "left_foot_air_time": frame["left_foot_air_time"],
        "right_foot_air_time": frame["right_foot_air_time"], "flight": frame["support_phase"].eq(0),
        "alpha_oracle": alpha, "residual_l2": np.linalg.norm(residual, axis=1),
        "full_vector_within_0_25": (np.abs(residual) <= BOUND).all(1),
    })
    distribution.to_csv(OUT / "oracle_alpha_distribution.csv", index=False)
    coverage = {
        name: coverage_payload(residual, regime == name)
        for name in ("walk_steady", "run_steady", "walk_to_run")
    }
    thresholds = {"walk_steady": 0.995, "run_steady": 0.99, "walk_to_run": 0.99}
    oracle_feasible = all(coverage[name]["full_vector_coverage"] >= thresholds[name] for name in thresholds)
    dump("oracle_scalar_morph.json", {
        "formula": "clip(dot(target-walk,runbase-walk)/(norm(runbase-walk)^2+epsilon),0,1)",
        "epsilon": 1e-8, "residual_bound": 0.25, "states": len(frame),
        "alpha_mean_by_regime": {name: float(alpha[regime == name].mean()) for name in set(regime)},
        "finite": bool(np.isfinite(alpha).all() and np.isfinite(residual).all()),
    })
    dump("oracle_residual_coverage.json", {
        "bound_normalized": BOUND, "joint_target_bound_rad": ACTION_SCALE * BOUND,
        "coverage": coverage, "thresholds": thresholds, "oracle_morph_feasible": oracle_feasible,
    })
    walk_alpha, run_alpha = alpha[regime == "walk_steady"], alpha[regime == "run_steady"]
    endpoints = {
        "walk_alpha_p95": float(np.quantile(walk_alpha, 0.95)), "walk_threshold_max": 0.05,
        "run_alpha_p05": float(np.quantile(run_alpha, 0.05)), "run_threshold_min": 0.95,
    }
    endpoints["pass"] = endpoints["walk_alpha_p95"] <= 0.05 and endpoints["run_alpha_p05"] >= 0.95
    dump("endpoint_consistency.json", endpoints)

    trajectory_rows, episode_metrics = monotonic_metrics(frame, alpha)
    with (OUT / "walk_to_run_alpha_trajectories.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectory_rows[0]))
        writer.writeheader()
        writer.writerows(trajectory_rows)
    tv = np.asarray([row["total_variation"] for row in episode_metrics])
    jump = np.concatenate([
        np.abs(np.diff(group["alpha_oracle"].to_numpy()))
        for _, group in pd.DataFrame(trajectory_rows).groupby("episode_id")
    ])
    backward_rates = np.asarray([row["large_backward_jump_rate"] for row in episode_metrics])
    monotonic = {
        "episodes": len(episode_metrics), "alpha_total_variation_p95": float(np.quantile(tv, 0.95)),
        "one_step_alpha_jump_p99": float(np.quantile(jump, 0.99)),
        "large_backward_jump_rate_overall": float(np.mean([
            row["large_backward_jump_rate"] for row in episode_metrics
        ])),
        "episodes_meeting_total_variation": float((tv <= 1.5).mean()),
        "episodes_meeting_backward_rate": float((backward_rates <= 0.05).mean()),
        "non_finite": 0,
    }
    monotonic["pass"] = (
        monotonic["alpha_total_variation_p95"] <= 1.5
        and monotonic["one_step_alpha_jump_p99"] <= 0.20
        and monotonic["large_backward_jump_rate_overall"] <= 0.05
    )
    monotonic["progresses_zero_to_one"] = float(np.mean([row["start_alpha"] <= 0.1 and row["end_alpha"] >= 0.9 for row in episode_metrics]))
    dump("alpha_monotonicity.json", monotonic)
    phase_stats = {}
    for group in sorted(frame["audit_group"].unique()):
        mask = frame["audit_group"].eq(group).to_numpy()
        phase_stats[group] = {
            "samples": int(mask.sum()), "alpha_mean": float(alpha[mask].mean()),
            "alpha_p10": float(np.quantile(alpha[mask], 0.10)), "alpha_p50": float(np.quantile(alpha[mask], 0.50)),
            "alpha_p90": float(np.quantile(alpha[mask], 0.90)), "alpha_variance": float(alpha[mask].var()),
        }
    for support in sorted(frame["support_phase"].unique()):
        mask = frame["support_phase"].eq(support).to_numpy() & (regime == "walk_to_run")
        phase_stats[f"wtr_support_{support}"] = {
            "samples": int(mask.sum()), "alpha_mean": float(alpha[mask].mean()), "alpha_variance": float(alpha[mask].var()),
        }
    dump("phase_conditioned_alpha_statistics.json", phase_stats)

    requested = frame["target_speed_mps"].to_numpy(np.float32)
    x = np.clip((requested - 1.2) / (2.4 - 1.2), 0, 1)
    speed_alpha = 3 * x**2 - 2 * x**3
    speed_anchor = walk + speed_alpha[:, None] * (run_base - walk)
    speed_residual = target - speed_anchor
    speed_error = np.abs(speed_alpha - alpha)
    speed_reference = {
        "alpha_mae": float(speed_error.mean()), "alpha_p95_error": float(np.quantile(speed_error, 0.95)),
        "coverage": {name: coverage_payload(speed_residual, regime == name) for name in set(regime)},
        "by_transition_phase": {
            group: {
                "alpha_mae": float(speed_error[frame["audit_group"].eq(group)].mean()),
                "full_vector_coverage": float((np.abs(speed_residual[frame["audit_group"].eq(group)]) <= BOUND).all(1).mean()),
            } for group in frame["audit_group"].unique() if group.startswith("wtr_")
        },
    }
    speed_reference["sufficient"] = (
        speed_reference["alpha_mae"] <= 0.05 and speed_reference["alpha_p95_error"] <= 0.10
        and speed_reference["coverage"]["walk_to_run"]["full_vector_coverage"] >= 0.99
    )
    dump("speed_only_reference.json", speed_reference)

    # Fixed joint-group oracle diagnostic.
    groups = {
        "lower_body_proximal": [i for i, name in enumerate(JOINT_NAMES) if any(key in name for key in ("hip_yaw", "hip_roll", "hip_pitch"))],
        "knee": [i for i, name in enumerate(JOINT_NAMES) if "knee" in name],
        "ankle": [i for i, name in enumerate(JOINT_NAMES) if "ankle" in name],
        "waist": [i for i, name in enumerate(JOINT_NAMES) if name == "torso_joint"],
        "upper_body": [i for i, name in enumerate(JOINT_NAMES) if not any(key in name for key in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")) and name != "torso_joint"],
    }
    group_alpha = {}
    group_anchor = walk.copy()
    for name, indices in groups.items():
        values = oracle_alpha(target, walk, run_base, indices)
        group_alpha[name] = values
        group_anchor[:, indices] = walk[:, indices] + values[:, None] * (run_base[:, indices] - walk[:, indices])
    group_residual = target - group_anchor
    dump("groupwise_oracle_morph.json", {
        "groups": groups,
        "alpha_distribution": {
            name: {
                "mean": float(values.mean()), "p05": float(np.quantile(values, 0.05)),
                "p50": float(np.quantile(values, 0.50)), "p95": float(np.quantile(values, 0.95)),
            } for name, values in group_alpha.items()
        },
        "mean_per_state_group_alpha_range": float(np.mean(
            np.max(np.column_stack(list(group_alpha.values())), axis=1)
            - np.min(np.column_stack(list(group_alpha.values())), axis=1)
        )),
    })
    group_coverage = {
        name: coverage_payload(group_residual, regime == name)
        for name in ("walk_steady", "run_steady", "walk_to_run")
    }
    group_feasible = all(group_coverage[name]["full_vector_coverage"] >= thresholds[name] for name in thresholds)
    dump("groupwise_residual_coverage.json", {
        "coverage": group_coverage, "thresholds": thresholds, "groupwise_morph_feasible": group_feasible,
    })

    features = feature_conditions(frame)
    probe_results = train_probes(features, alpha, frame["split"].to_numpy(), actions, frame, oracle_feasible)
    torch.save({
        "frame_metadata": frame[META + ["audit_group", "last_landing_foot"]].to_dict("list"),
        "actions": {key: torch.from_numpy(value) for key, value in actions.items()},
        "alpha": torch.from_numpy(alpha), "residual": torch.from_numpy(residual),
    }, OUT / "stage6_intermediate.pt")
    print(json.dumps({
        "oracle_feasible": oracle_feasible, "endpoint_pass": endpoints["pass"],
        "monotonic_pass": monotonic["pass"], "speed_only_sufficient": speed_reference["sufficient"],
        "group_feasible": group_feasible,
        "probe_summary": probe_results,
    }, indent=2))


if __name__ == "__main__":
    main()
