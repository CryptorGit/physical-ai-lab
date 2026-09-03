"""EXP014 Phase 2-D1 Specialist-S reset-boundary positive control.

This is deliberately fail-closed: reset-boundary labels are written only when
the frozen Specialist S passes the preregistered physical gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import torch

HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
BASE = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion"
OUT = BASE / "phase_2_d1_reset_boundary_causal_dagger_v2"
V1 = BASE / "phase1_dataset/phase1_batch_00.pt"
STOP = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"

sys.path[:0] = [
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),
    str(REPO / "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src"),
    str(EXP / "src"),
]
import isaaclab_tasks  # noqa: F401,E402
import g1_omnidirectional.tasks  # noqa: F401,E402
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402
from g1_explicit_motion_mode.contract import ExplicitMotionModeCommand, build_observation_141  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

DT = 0.02
EXPECTED_TEACHER_SHA = "66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def q_to_roll_pitch(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Isaac Lab root_quat_w is scalar-first (w, x, y, z).
    w, x, y, z = q.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = (2 * (w * y - z * x)).clamp(-1, 1)
    return roll, torch.asin(sinp)


def vec(value: torch.Tensor) -> str:
    return json.dumps([round(float(x), 9) for x in value.detach().cpu().flatten()])


def main() -> None:
    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    args, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra_args]

    teacher_sha = sha256(STOP)
    if teacher_sha != EXPECTED_TEACHER_SHA:
        raise RuntimeError(f"Specialist S hash mismatch: {teacher_sha}")
    v1_before = sha256(V1)
    v1 = torch.load(V1, map_location="cpu", weights_only=False)
    recipes = torch.unique(v1["recipe_id"].flatten()).sort().values
    split_by_recipe = {}
    for recipe in recipes.tolist():
        splits = torch.unique(v1["split_id"][v1["recipe_id"].flatten() == recipe].flatten())
        if len(splits) != 1:
            raise RuntimeError(f"recipe {recipe} has non-unique split")
        split_by_recipe[int(recipe)] = int(splits.item())
    n = len(recipes)
    if n != 680:
        raise RuntimeError(f"registered recipe count changed: {n}")

    cfg, agent = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = n
    cfg.episode_length_s = 16.0
    cfg.seed = 20260803
    cfg.observations.policy.enable_corruption = False
    if args.device:
        cfg.sim.device = agent.device = args.device
    OUT.mkdir(parents=True, exist_ok=True)

    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(
            gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg),
            clip_actions=agent.clip_actions,
        )
        env = wrapped.unwrapped
        dev = env.device
        robot = env.scene["robot"]
        sensor = env.scene["contact_forces"]
        sensor_feet = sensor.find_bodies(".*_ankle_roll_link")[0]
        robot_feet = robot.find_bodies(".*_ankle_roll_link")[0]
        term = env.command_manager.get_term("base_velocity")
        term.external_override_enabled = True
        ids = torch.arange(n, device=dev)
        env.reset(env_ids=ids)
        term.external_override.zero_()
        term._update_command()
        obs = wrapped.get_observations().to(dev)
        state = ExplicitMotionModeCommand.zeros(n, device=dev)
        teacher = FrozenGaitActor(STOP).to(dev).eval()
        gait = torch.zeros(n, device=dev)

        reset_episode_length = env.episode_length_buf.clone()
        reset_corruption = reset_episode_length.ne(0)
        reset_corruption |= ~torch.isfinite(robot.data.root_state_w).all(1)
        reset_corruption |= ~torch.isfinite(robot.data.joint_pos).all(1)
        reset_corruption |= ~torch.isfinite(robot.data.joint_vel).all(1)

        limits = robot.data.joint_vel_limits
        limits = limits[..., 1].abs() if limits.ndim == 3 else limits
        fall = torch.zeros(n, dtype=torch.bool, device=dev)
        slip = fall.clone()
        impact = fall.clone()
        saturation = fall.clone()
        slip_streak = torch.zeros(n, dtype=torch.long, device=dev)
        saturation_streak = slip_streak.clone()
        mean_speed = []
        mean_yaw = []
        boundary: list[dict] = []
        boundary_tensors: dict[str, list[torch.Tensor]] = {
            key: [] for key in ("observation_141", "teacher_action", "recipe_id", "episode_id", "control_step", "split_id")
        }
        nan_inf_count = 0
        bounds_violations = 0
        configured_action_bound = None if agent.clip_actions is None else float(agent.clip_actions)

        for step in range(100):
            physical = torch.zeros(n, 3, device=dev)
            state.advance(physical, torch.ones(n, device=dev), DT)
            term.external_override.zero_()
            term._update_command()
            obs = wrapped.get_observations().to(dev)
            x141 = build_observation_141(obs["policy"], state)
            with torch.inference_mode():
                action = teacher(obs["policy"], gait)
            nan_inf_count += int((~torch.isfinite(x141)).sum()) + int((~torch.isfinite(action)).sum())
            if configured_action_bound is not None:
                bounds_violations += int((action.abs() > configured_action_bound + 1e-6).any(1).sum())

            if step < 4:
                roll, pitch = q_to_roll_pitch(robot.data.root_quat_w)
                contact_force = sensor.data.net_forces_w_history[:, -1, sensor_feet, :].norm(dim=-1)
                previous_action = env.action_manager.prev_action.clone()
                for i, recipe in enumerate(recipes.tolist()):
                    boundary.append({
                        "recipe_id": int(recipe),
                        "episode_id": int(recipe),
                        "split": ("train", "validation", "held-out")[split_by_recipe[int(recipe)]],
                        "control_step": step,
                        "teacher_action": vec(action[i]),
                        "root_state": vec(robot.data.root_state_w[i]),
                        "joint_position": vec(robot.data.joint_pos[i]),
                        "joint_velocity": vec(robot.data.joint_vel[i]),
                        "previous_action": vec(previous_action[i]),
                        "contact_force": vec(contact_force[i]),
                        "base_linear_velocity": vec(robot.data.root_lin_vel_b[i]),
                        "base_angular_velocity": vec(robot.data.root_ang_vel_b[i]),
                        "roll": float(roll[i]),
                        "pitch": float(pitch[i]),
                    })
                boundary_tensors["observation_141"].append(x141.detach().cpu())
                boundary_tensors["teacher_action"].append(action.detach().cpu())
                boundary_tensors["recipe_id"].append(recipes[:, None].clone())
                boundary_tensors["episode_id"].append(recipes[:, None].clone())
                boundary_tensors["control_step"].append(torch.full((n, 1), step, dtype=torch.long))
                boundary_tensors["split_id"].append(torch.tensor([split_by_recipe[int(r)] for r in recipes.tolist()])[:, None])

            obs, _, done, extras = wrapped.step(action)
            obs = obs.to(dev)
            timeout = extras.get("time_outs", torch.zeros_like(done)).bool()
            fall |= done.bool() & ~timeout
            force = sensor.data.net_forces_w_history[:, -1, sensor_feet, :].norm(dim=-1)
            contact = force > 5
            feet_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1)
            bad_slip = ((feet_speed > 0.55) & contact).any(1)
            slip_streak = torch.where(bad_slip, slip_streak + 1, torch.zeros_like(slip_streak))
            slip |= slip_streak >= 5
            impact |= force.amax(1) > 3500
            ratio = robot.data.joint_vel.abs().div(limits.clamp_min(1e-6)).amax(1)
            saturation_streak = torch.where(ratio > 0.95, saturation_streak + 1, torch.zeros_like(saturation_streak))
            saturation |= saturation_streak >= 5
            mean_speed.append(torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=1))
            mean_yaw.append(robot.data.root_ang_vel_b[:, 2].abs())

        speed = torch.stack(mean_speed).mean(0)
        yaw = torch.stack(mean_yaw).mean(0)
        practical = (speed <= 0.08) & (yaw <= 0.08) & ~fall & ~slip & ~impact
        rates = {
            "practical_stand": float(practical.float().mean()),
            "fall": float(fall.float().mean()),
            "dangerous_slip": float(slip.float().mean()),
            "impact": float(impact.float().mean()),
            "long_dwell_saturation": float(saturation.float().mean()),
        }
        gate = (
            rates["practical_stand"] >= 0.95
            and rates["fall"] <= 0.02
            and rates["dangerous_slip"] <= 0.05
            and rates["impact"] <= 0.05
            and rates["long_dwell_saturation"] <= 0.05
            and nan_inf_count == 0
            and bounds_violations == 0
            and int(reset_corruption.sum()) == 0
        )
        outcome = {
            int(recipe): {
                "fall": bool(fall[i]),
                "dangerous_slip": bool(slip[i]),
                "impact": bool(impact[i]),
                "long_dwell_saturation": bool(saturation[i]),
                "practical_stand": bool(practical[i]),
            }
            for i, recipe in enumerate(recipes.tolist())
        }
        for row in boundary:
            row.update(outcome[row["recipe_id"]])
        csv_path = OUT / "specialist_s_reset_positive_control.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(boundary[0]))
            writer.writeheader()
            writer.writerows(boundary)
        summary = {
            "status": "PASS" if gate else "FAIL",
            "classification_on_fail": None if gate else "EXP014_RESET_BOUNDARY_SPECIALIST_SCOPE_FAIL",
            "reset_distribution": {
                "task": "Isaac-Exp013-G1-DirectionalBaseline-v0",
                "seed": 20260803,
                "recipes": n,
                "recipe_ids_preserved": True,
                "split_recipe_counts": {
                    name: sum(v == i for v in split_by_recipe.values())
                    for i, name in enumerate(("train", "validation", "held-out"))
                },
                "control_steps": 100,
                "control_dt": DT,
            },
            "teacher": {"path": STOP.relative_to(REPO).as_posix(), "sha256": teacher_sha, "deterministic_mean": True},
            "metrics": rates,
            "boundary_checks": {
                "steps": [0, 1, 2, 3],
                "candidate_labels": n * 4,
                "nan_inf": nan_inf_count,
                "action_bounds_violation": bounds_violations,
                "configured_action_bound": configured_action_bound,
                "reset_buffer_corruption": int(reset_corruption.sum()),
            },
            "gate": {
                "practical_stand_min": 0.95,
                "fall_max": 0.02,
                "dangerous_slip_max": 0.05,
                "impact_max": 0.05,
                "long_dwell_saturation_max": 0.05,
            },
            "v1_sha256_before": v1_before,
            "v1_sha256_after": sha256(V1),
            "boundary_labels_published": gate,
        }
        (OUT / "specialist_s_reset_positive_control.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if gate:
            overlay = {key: torch.cat(value) for key, value in boundary_tensors.items()}
            overlay.update({
                "dataset_name": "Exp014StandOmniWalkTrajectoryDatasetV2",
                "overlay_kind": "RESET_BOUNDARY_POSITIVE_CONTROL",
                "target_mode": torch.zeros(n * 4, 1, dtype=torch.long),
                "physical_command": torch.zeros(n * 4, 3),
                "teacher_source_metadata": "Specialist S (not actor input)",
                "teacher_sha256": teacher_sha,
            })
            torch.save(overlay, OUT / "reset_boundary_positive_control_overlay.pt")
        print(json.dumps(summary, indent=2))
        wrapped.close()


if __name__ == "__main__":
    main()
