"""Branch parent STOP snapshots for checkpoint and local-action response diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import torch
from tensordict import TensorDict


SCRIPT = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT.parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
import g1_command_skills.tasks  # noqa: E402,F401
from g1_command_skills.models import G1CommandResidualActor  # noqa: E402
from g1_command_skills.models.residual_actor import _LATEST_STOP_CORRECTION  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402
from rsl_rl.models import MLPModel  # noqa: E402


JOINTS = (
    ("torso", 2),
    ("left_hip_yaw", 7), ("right_hip_yaw", 8),
    ("left_hip_roll", 3), ("right_hip_roll", 4),
    ("left_ankle_roll", 19), ("right_ankle_roll", 20),
    ("left_hip_pitch", 0), ("right_hip_pitch", 1),
)
PERTURBATIONS = (-0.01, -0.005, 0.005, 0.01)
REPRESENTATIVES = ("braking_early", "braking_late", "hold_start", "large_heading_drift", "pre_fall")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--max-episodes", type=int, default=10)
parser.add_argument("--snapshot-cache", type=Path, required=True)
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--capture-only", action="store_true")
mode.add_argument("--branch-only", action="store_true")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def make_models(observations: TensorDict, checkpoint_path: Path, device: str):
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = G1CommandResidualActor(
        observations,
        groups,
        "actor",
        37,
        hidden_dims=[256, 128, 128],
        activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
        trainable_skill_ids=[],
        train_stop_correction=False,
    ).to(device)
    critic = MLPModel(
        observations,
        groups,
        "critic",
        1,
        hidden_dims=[256, 128, 128],
        activation="elu",
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)
    actor.eval()
    critic.eval()
    return actor, critic


def observation_slice(observation, start: int, end: int) -> TensorDict:
    if isinstance(observation, TensorDict):
        return observation[start:end].clone()
    return TensorDict(
        {name: value[start:end].clone() for name, value in observation.items()},
        batch_size=[end - start],
    )


def as_tensordict(observation) -> TensorDict:
    if isinstance(observation, TensorDict):
        return observation
    first = next(iter(observation.values()))
    return TensorDict(observation, batch_size=[first.shape[0]])


def tensor_state(obj, num_envs: int, env_id: int) -> dict[str, torch.Tensor]:
    state = {}
    for name, value in vars(obj).items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == num_envs:
            state[name] = value[env_id].detach().clone()
    return state


def snapshot(unwrapped, term, observation: TensorDict, env_id: int = 0) -> dict:
    robot = unwrapped.scene["robot"]
    origin = unwrapped.scene.env_origins[env_id]
    return {
        "root_position_local": (robot.data.root_pos_w.torch[env_id] - origin).detach().clone(),
        "root_quaternion": robot.data.root_quat_w.torch[env_id].detach().clone(),
        "root_velocity": torch.cat((
            robot.data.root_lin_vel_w.torch[env_id], robot.data.root_ang_vel_w.torch[env_id]
        )).detach().clone(),
        "joint_position": robot.data.joint_pos.torch[env_id].detach().clone(),
        "joint_velocity": robot.data.joint_vel.torch[env_id].detach().clone(),
        "command": tensor_state(term, unwrapped.num_envs, env_id),
        "action": unwrapped.action_manager.action[env_id].detach().clone(),
        "previous_action": unwrapped.action_manager.prev_action[env_id].detach().clone(),
        "episode_length": unwrapped.episode_length_buf[env_id].detach().clone(),
        "observation": observation_slice(observation, env_id, env_id + 1),
    }


def restore(unwrapped, term, state: dict) -> None:
    robot = unwrapped.scene["robot"]
    ids = torch.arange(unwrapped.num_envs, device=unwrapped.device, dtype=torch.long)
    origins = unwrapped.scene.env_origins[ids]
    root_pose = torch.cat((
        state["root_position_local"].unsqueeze(0) + origins,
        state["root_quaternion"].unsqueeze(0).repeat(unwrapped.num_envs, 1),
    ), dim=1)
    robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=ids)
    robot.write_root_velocity_to_sim_index(
        root_velocity=state["root_velocity"].unsqueeze(0).repeat(unwrapped.num_envs, 1), env_ids=ids
    )
    robot.write_joint_state_to_sim(
        state["joint_position"].unsqueeze(0).repeat(unwrapped.num_envs, 1),
        state["joint_velocity"].unsqueeze(0).repeat(unwrapped.num_envs, 1),
        env_ids=ids,
    )
    for name, saved in state["command"].items():
        target = getattr(term, name)
        repeated = saved.unsqueeze(0).repeat((unwrapped.num_envs,) + (1,) * saved.ndim)
        if name in {"target_position_w", "path_origin_w"}:
            repeated = repeated + origins[:, :2]
        target[ids] = repeated
    unwrapped.action_manager._action[ids] = state["action"]
    unwrapped.action_manager._prev_action[ids] = state["previous_action"]
    unwrapped.episode_length_buf[ids] = state["episode_length"]
    unwrapped.sim.forward()


def attitude(robot, env_ids) -> tuple[torch.Tensor, torch.Tensor]:
    gravity = robot.data.projected_gravity_b.torch[env_ids]
    roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
    pitch = torch.atan2(gravity[:, 0], torch.sqrt(gravity[:, 1].square() + gravity[:, 2].square()))
    return roll, pitch


def physical_metrics(unwrapped, env_ids: torch.Tensor, foot_ids, ankle_ids, all_joint_ids) -> dict[str, torch.Tensor]:
    robot = unwrapped.scene["robot"]
    contact = unwrapped.scene.sensors["contact_forces"]
    roll, pitch = attitude(robot, env_ids)
    foot_forces = contact.data.net_forces_w_history.torch[env_ids][:, :, foot_ids, :].norm(dim=-1).amax(dim=1)
    joint_ratio = (
        robot.data.joint_vel.torch[env_ids][:, all_joint_ids].abs()
        / robot.data.joint_vel_limits.torch[env_ids][:, all_joint_ids].abs().clamp_min(1.0e-6)
    )
    ankle_ratio = (
        robot.data.applied_torque.torch[env_ids][:, ankle_ids].abs()
        / robot.data.joint_effort_limits.torch[env_ids][:, ankle_ids].abs().clamp_min(1.0e-6)
    )
    tilt = torch.sqrt(roll.square() + pitch.square())
    yaw_rate = robot.data.root_ang_vel_b.torch[env_ids, 2]
    return {
        "yaw_rate": yaw_rate,
        "heading": robot.data.heading_w.torch[env_ids],
        "roll": roll,
        "pitch": pitch,
        "roll_rate": robot.data.root_ang_vel_b.torch[env_ids, 0],
        "forward_speed": robot.data.root_lin_vel_b.torch[env_ids, 0],
        "left_contact": foot_forces[:, 0] > 1.0,
        "right_contact": foot_forces[:, 1] > 1.0,
        "joint_velocity_saturation": (joint_ratio >= 0.95).float().mean(dim=1),
        "ankle_torque_saturation": (ankle_ratio >= 0.95).float().mean(dim=1),
        "fall_risk": tilt + 0.20 * yaw_rate.abs() + 0.20 * joint_ratio.amax(dim=1),
    }


def scalar(value: torch.Tensor, index: int) -> float:
    return float(value[index].item())


def pearson(rows: list[dict], left: str, right: str) -> float:
    pairs = [(float(row[left]), float(row[right])) for row in rows if left in row and right in row]
    if len(pairs) < 2:
        return 0.0
    xs, ys = zip(*pairs)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - mx) * (y - my) for x, y in pairs)
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator > 1.0e-12 else 0.0


def main() -> None:
    cfg, _ = resolve_task_config("Isaac-Motion-Flat-G1-Command-Stop-Eval-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 1 if args_cli.capture_only else 37
    cfg.seed = 42
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device
    with launch_simulation(cfg, args_cli):
        env = gym.make("Isaac-Motion-Flat-G1-Command-Stop-Eval-v0", cfg=cfg)
        unwrapped = env.unwrapped
        device = unwrapped.device
        placeholder = TensorDict(
            {"policy": torch.zeros(unwrapped.num_envs, 152, device=device)},
            batch_size=[unwrapped.num_envs],
        )
        checkpoint_paths = sorted(args_cli.run_dir.resolve().glob("model_*.pt"), key=lambda p: int(p.stem[6:]))
        if args_cli.capture_only:
            checkpoint_paths = [args_cli.run_dir.resolve() / "model_0.pt"]
        models = {path.stem: make_models(placeholder, path, device) for path in checkpoint_paths}
        parent_actor = models["model_0"][0]
        observations, _ = env.reset()
        observations = as_tensordict(observations)
        term = unwrapped.command_manager.get_term("base_velocity")
        robot = unwrapped.scene["robot"]
        contact = unwrapped.scene.sensors["contact_forces"]
        _, foot_names = robot.find_bodies(".*_ankle_roll_link")
        foot_ids = [contact.body_names.index(name) for name in foot_names]
        ankle_ids, _ = robot.find_joints(".*_ankle_.*_joint")
        all_joint_ids, _ = robot.find_joints(".*")

        if args_cli.capture_only:
            representatives: dict[str, dict] = {}
            largest_heading = -1.0
            largest_risk = -1.0
            risk_snapshot = None
            episodes = 0
            previous = None
            while episodes < args_cli.max_episodes and len(representatives) < len(REPRESENTATIVES):
                current = snapshot(unwrapped, term, observations)
                if int(term.skill_id[0].item()) == 1:
                    progress = float(term.stop_progress[0].item())
                    hold = float(term.stop_hold_progress[0].item())
                    heading_error = abs(float(term.heading_error[0].item()))
                    if "braking_early" not in representatives and 0.15 <= progress <= 0.35 and hold == 0.0:
                        representatives["braking_early"] = current
                    if "braking_late" not in representatives and 0.70 <= progress <= 0.95 and hold == 0.0:
                        representatives["braking_late"] = current
                    if "hold_start" not in representatives and 0.0 < hold <= 0.10:
                        representatives["hold_start"] = current
                    if heading_error > largest_heading:
                        largest_heading = heading_error
                        representatives["large_heading_drift"] = current
                    policy = current["observation"]["policy"][0]
                    risk = float(torch.linalg.vector_norm(policy[6:8]).item() + 0.20 * abs(policy[5].item()))
                    if risk > largest_risk:
                        largest_risk = risk
                        risk_snapshot = current
                with torch.inference_mode():
                    actions = parent_actor(observations)
                observations, _, terminated, truncated, _ = env.step(actions)
                observations = as_tensordict(observations)
                if bool(terminated[0].item()):
                    representatives["pre_fall"] = previous or current
                if bool((terminated | truncated)[0].item()):
                    episodes += 1
                previous = current
            pre_fall_surrogate = False
            if "pre_fall" not in representatives and risk_snapshot is not None:
                representatives["pre_fall"] = risk_snapshot
                pre_fall_surrogate = True
            missing = [name for name in REPRESENTATIVES if name not in representatives]
            if missing:
                env.close()
                raise RuntimeError(f"Could not capture representative states: {missing}")
            args_cli.snapshot_cache.resolve().parent.mkdir(parents=True, exist_ok=True)
            torch.save(representatives, args_cli.snapshot_cache.resolve())
            report = {
                "captured": list(representatives),
                "episodes_consumed": episodes,
                "num_envs": 1,
                "snapshot_cache": str(args_cli.snapshot_cache.resolve()),
                "pre_fall_is_max_risk_surrogate": pre_fall_surrogate,
                "max_surrogate_risk": largest_risk,
            }
            args_cli.output.resolve().mkdir(parents=True, exist_ok=True)
            (args_cli.output.resolve() / "capture_summary.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(report, indent=2))
            env.close()
            return

        representatives = torch.load(
            args_cli.snapshot_cache.resolve(strict=True), map_location=device, weights_only=False
        )

        local_rows = []
        checkpoint_rows = []
        checkpoint_names = [path.stem for path in checkpoint_paths]
        reward_names = (
            "stop_heading", "stop_yaw_rate_tracking", "stop_attitude_stability",
            "stop_joint_velocity_saturation", "stop_ankle_torque_saturation",
            "stop_parent_action_deviation",
        )
        reward_indices = {name: unwrapped.reward_manager.active_terms.index(name) for name in reward_names}

        for representative_name, state in representatives.items():
            # 9 joints x 4 signed perturbations plus an unperturbed control.
            restore(unwrapped, term, state)
            ids = torch.arange(37, device=device)
            initial = physical_metrics(unwrapped, ids, foot_ids, ankle_ids, all_joint_ids)
            branch_specs = [("control", -1, 0.0)] + [
                (name, index, perturbation) for name, index in JOINTS for perturbation in PERTURBATIONS
            ]
            step_metrics = {}
            for step in range(1, 4):
                obs = as_tensordict(unwrapped.observation_manager.compute())
                with torch.inference_mode():
                    actions = parent_actor(obs)
                actions = actions.clone()
                for branch, (_, joint_index, perturbation) in enumerate(branch_specs):
                    if joint_index >= 0:
                        actions[branch, joint_index] += perturbation
                obs, _, terminated, truncated, _ = env.step(actions)
                obs = as_tensordict(obs)
                step_metrics[step] = physical_metrics(unwrapped, ids, foot_ids, ankle_ids, all_joint_ids)
            control = 0
            for branch, (joint_name, joint_index, perturbation) in enumerate(branch_specs[1:], start=1):
                row = {
                    "representative": representative_name,
                    "joint": joint_name,
                    "joint_index": joint_index,
                    "perturbation": perturbation,
                    "yaw_rate_delta_1step": scalar(step_metrics[1]["yaw_rate"], branch) - scalar(step_metrics[1]["yaw_rate"], control),
                    "yaw_delta_3step": (
                        scalar(step_metrics[3]["heading"], branch) - scalar(initial["heading"], branch)
                        - scalar(step_metrics[3]["heading"], control) + scalar(initial["heading"], control)
                    ),
                    "roll_delta_3step": scalar(step_metrics[3]["roll"], branch) - scalar(step_metrics[3]["roll"], control),
                    "pitch_delta_3step": scalar(step_metrics[3]["pitch"], branch) - scalar(step_metrics[3]["pitch"], control),
                    "forward_speed_delta_3step": scalar(step_metrics[3]["forward_speed"], branch) - scalar(step_metrics[3]["forward_speed"], control),
                    "left_contact_3step": bool(step_metrics[3]["left_contact"][branch].item()),
                    "right_contact_3step": bool(step_metrics[3]["right_contact"][branch].item()),
                    "joint_velocity_saturation_3step": scalar(step_metrics[3]["joint_velocity_saturation"], branch),
                    "ankle_torque_saturation_3step": scalar(step_metrics[3]["ankle_torque_saturation"], branch),
                    "fall_risk_delta_3step": scalar(step_metrics[3]["fall_risk"], branch) - scalar(step_metrics[3]["fall_risk"], control),
                    "terminated_within_3step": bool(terminated[branch].item() and not truncated[branch].item()),
                }
                local_rows.append(row)

            # Compare all corrective checkpoints from the exact same snapshot.
            restore(unwrapped, term, state)
            model_ids = torch.arange(len(checkpoint_names), device=device)
            initial_physical = physical_metrics(unwrapped, model_ids, foot_ids, ankle_ids, all_joint_ids)
            initial_obs = as_tensordict(unwrapped.observation_manager.compute())
            initial_components = {}
            initial_values = {}
            for branch, name in enumerate(checkpoint_names):
                actor, critic = models[name]
                obs_slice = observation_slice(initial_obs, branch, branch + 1)
                with torch.inference_mode():
                    initial_components[name] = actor.diagnostic_components(obs_slice)
                    initial_values[name] = float(critic(obs_slice)[0].item())
            step_one_rewards = {}
            step_one_terms = {}
            step_one_advantages = {}
            physical_steps = {}
            for step in range(1, 6):
                obs = as_tensordict(unwrapped.observation_manager.compute())
                with torch.inference_mode():
                    actions = parent_actor(obs)
                actions = actions.clone()
                corrections = torch.zeros_like(actions)
                for branch, name in enumerate(checkpoint_names):
                    actor, _ = models[name]
                    with torch.inference_mode():
                        components = actor.diagnostic_components(observation_slice(obs, branch, branch + 1))
                    actions[branch] = components["action_mean"][0]
                    corrections[branch] = components["parent_action_deviation"][0]
                _LATEST_STOP_CORRECTION.clear()
                _LATEST_STOP_CORRECTION.update({
                    "corrective_residual": corrections,
                    "parent_action_deviation": corrections,
                })
                next_obs, rewards, terminated, truncated, _ = env.step(actions)
                next_obs = as_tensordict(next_obs)
                physical_steps[step] = physical_metrics(unwrapped, model_ids, foot_ids, ankle_ids, all_joint_ids)
                if step == 1:
                    for branch, name in enumerate(checkpoint_names):
                        _, critic = models[name]
                        with torch.inference_mode():
                            next_value = float(critic(observation_slice(next_obs, branch, branch + 1))[0].item())
                        reward = float(rewards[branch].item())
                        step_one_rewards[name] = reward
                        step_one_advantages[name] = reward + 0.99 * next_value - initial_values[name]
                        step_one_terms[name] = {
                            reward_name: float(unwrapped.reward_manager._step_reward[branch, reward_indices[reward_name]].item())
                            for reward_name in reward_names
                        }
                obs = next_obs

            parent_heading_after = abs(float(term.heading_error[0].item()))
            parent_saturation_after = scalar(physical_steps[5]["joint_velocity_saturation"], 0)
            for branch, name in enumerate(checkpoint_names):
                components = initial_components[name]
                correction = components["parent_action_deviation"][0]
                parent_action = components["parent_action_mean"][0]
                final_action = components["action_mean"][0]
                joint_values = {joint_name: float(correction[joint_index].item()) for joint_name, joint_index in JOINTS}
                heading_after = abs(float(term.heading_error[branch].item()))
                saturation_after = scalar(physical_steps[5]["joint_velocity_saturation"], branch)
                row = {
                    "representative": representative_name,
                    "model": name,
                    "heading_error_initial": abs(float(state["command"]["heading_error"].item())),
                    "actual_yaw_rate_initial": scalar(initial_physical["yaw_rate"], branch),
                    "legacy_yaw_rate_command": float(state["command"]["vel_command_b"][2].item()) if "vel_command_b" in state["command"] else float(state["observation"]["policy"][0, 11].item()),
                    "roll_initial": scalar(initial_physical["roll"], branch),
                    "pitch_initial": scalar(initial_physical["pitch"], branch),
                    "corrective_norm": float(torch.linalg.vector_norm(correction).item()),
                    "parent_action_json": json.dumps(parent_action.detach().cpu().tolist(), separators=(",", ":")),
                    "final_action_json": json.dumps(final_action.detach().cpu().tolist(), separators=(",", ":")),
                    "yaw_rate_delta_1step_vs_parent": scalar(physical_steps[1]["yaw_rate"], branch) - scalar(physical_steps[1]["yaw_rate"], 0),
                    "yaw_rate_delta_5step_vs_parent": scalar(physical_steps[5]["yaw_rate"], branch) - scalar(physical_steps[5]["yaw_rate"], 0),
                    "heading_improvement_5step_vs_parent": parent_heading_after - heading_after,
                    "saturation_improvement_5step_vs_parent": parent_saturation_after - saturation_after,
                    "td_advantage_proxy": step_one_advantages[name],
                    **joint_values,
                    **{f"reward_{key}": value for key, value in step_one_terms[name].items()},
                }
                row["hip_yaw_lr_difference"] = row["left_hip_yaw"] - row["right_hip_yaw"]
                row["hip_roll_lr_difference"] = row["left_hip_roll"] - row["right_hip_roll"]
                row["ankle_roll_lr_difference"] = row["left_ankle_roll"] - row["right_ankle_roll"]
                checkpoint_rows.append(row)

        correlations = {
            "torso_corrective_vs_heading_error": pearson(checkpoint_rows, "torso", "heading_error_initial"),
            "hip_yaw_lr_difference_vs_heading_error": pearson(checkpoint_rows, "hip_yaw_lr_difference", "heading_error_initial"),
            "hip_roll_lr_difference_vs_heading_error": pearson(checkpoint_rows, "hip_roll_lr_difference", "heading_error_initial"),
            "ankle_roll_lr_difference_vs_heading_error": pearson(checkpoint_rows, "ankle_roll_lr_difference", "heading_error_initial"),
            "corrective_norm_vs_next_yaw_rate_delta": pearson(checkpoint_rows, "corrective_norm", "yaw_rate_delta_1step_vs_parent"),
            "corrective_norm_vs_heading_improvement": pearson(checkpoint_rows, "corrective_norm", "heading_improvement_5step_vs_parent"),
            "corrective_norm_vs_saturation_improvement": pearson(checkpoint_rows, "corrective_norm", "saturation_improvement_5step_vs_parent"),
        }

        output = args_cli.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        for filename, rows in (("local_action_response.csv", local_rows), ("checkpoint_state_comparison.csv", checkpoint_rows)):
            with (output / filename).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        report = {
            "checkpoint_names": checkpoint_names,
            "representatives": list(representatives),
            "local_response_rows": len(local_rows),
            "checkpoint_comparison_rows": len(checkpoint_rows),
            "advantage_definition": "one-step TD residual r + 0.99 V(s_next) - V(s); original PPO rollout advantages are not checkpointed",
            "correlations": correlations,
        }
        (output / "diagnostic_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        env.close()


if __name__ == "__main__":
    main()
