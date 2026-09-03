"""Capture and branch deterministic STOP-entry snapshots for safety redesign."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from importlib import metadata
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict


SCRIPT = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT.parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
import g1_command_skills.tasks  # noqa: E402,F401
from g1_command_skills.fixed_feedback import StopFeedbackConfig, StopFixedFeedbackController  # noqa: E402
from g1_command_skills.models import G1CommandResidualActor  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402


FAILURE_IDS = (9, 11, 18, 19, 28, 37, 38, 47, 48)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("capture", "replay"), required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--snapshots", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--failure-ids", type=int, nargs="*", default=list(FAILURE_IDS))
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def as_tensordict(observation) -> TensorDict:
    if isinstance(observation, TensorDict):
        return observation
    first = next(iter(observation.values()))
    return TensorDict(observation, batch_size=[first.shape[0]])


def current_config(**overrides) -> StopFeedbackConfig:
    values = dict(
        k_heading=0.095, k_yaw_rate=0.09, alpha=0.5, max_delta_per_step=0.0025,
        braking_scale=1.0, hold_scale=1.0, single_support_scale=0.5, flight_scale=0.0,
        yaw_soft_threshold=1.2, yaw_hard_threshold=2.5, hard_guard_mode="damping_only",
    )
    values.update(overrides)
    return StopFeedbackConfig(**values)


def candidates() -> list[tuple[str, StopFeedbackConfig]]:
    recovery = dict(flight_hard_zero=True, contact_recovery_zero_steps=3, contact_recovery_ramp_steps=5)
    load_gate = dict(
        ankle_utilization_soft=0.90, ankle_utilization_hard=0.98,
        joint_velocity_soft=0.90, joint_velocity_hard=1.00,
        tilt_soft_rad=0.20, tilt_hard_rad=0.35,
        angular_velocity_soft_rps=1.5, angular_velocity_hard_rps=3.0,
    )
    return [
        ("current", current_config()),
        ("flight_zero", current_config(flight_hard_zero=True)),
        ("flight_recovery", current_config(**recovery)),
        ("recovery_hard015", current_config(**recovery, hard_guard_action_limit=0.015)),
        ("recovery_hard010", current_config(**recovery, hard_guard_action_limit=0.010)),
        ("recovery_hard_zero", current_config(**recovery, hard_guard_action_limit=0.0)),
        ("recovery_ankle_gate", current_config(**recovery, **load_gate)),
        ("combined_hard010_gate", current_config(
            **recovery, **load_gate, hard_guard_action_limit=0.010,
            hard_guard_disable_torso=True, worsening_yaw_scale=0.25,
        )),
    ]


def tensor_state(obj, num_envs: int, env_id: int) -> dict[str, torch.Tensor]:
    return {
        name: value[env_id].detach().clone()
        for name, value in vars(obj).items()
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == num_envs
    }


def robot_state(unwrapped, env_id: int = 0) -> dict:
    robot = unwrapped.scene["robot"]
    origin = unwrapped.scene.env_origins[env_id]
    return {
        "root_position_local": (robot.data.root_pos_w.torch[env_id] - origin).detach().clone(),
        "root_quaternion": robot.data.root_quat_w.torch[env_id].detach().clone(),
        "root_velocity": torch.cat((robot.data.root_lin_vel_w.torch[env_id], robot.data.root_ang_vel_w.torch[env_id])).detach().clone(),
        "joint_position": robot.data.joint_pos.torch[env_id].detach().clone(),
        "joint_velocity": robot.data.joint_vel.torch[env_id].detach().clone(),
    }


def snapshot(unwrapped, term, controller, episode_initial: dict, episode: int, run_commands: list[float]) -> dict:
    state = robot_state(unwrapped)
    state.update({
        "episode": episode,
        "master_seed": 42,
        "episode_initial": episode_initial,
        "command": tensor_state(term, unwrapped.num_envs, 0),
        "action": unwrapped.action_manager.action[0].detach().clone(),
        "previous_action": unwrapped.action_manager.prev_action[0].detach().clone(),
        "episode_length": unwrapped.episode_length_buf[0].detach().clone(),
        "controller": {
            "filtered_signal": controller.filtered_signal[0].detach().clone(),
            "applied_action": controller.applied_action[0].detach().clone(),
            "was_stop": controller.was_stop[0].detach().clone(),
            "previous_contacts": controller.previous_contacts[0].detach().clone(),
            "support_stable_steps": controller.support_stable_steps[0].detach().clone(),
            "previous_yaw_abs": controller.previous_yaw_abs[0].detach().clone(),
        },
        "rng": {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "preceding_run_commanded_speed_mean_mps": sum(run_commands) / max(len(run_commands), 1),
        "stop_entry_speed_mps": float(term.stop_entry_speed[0].item()),
        "stopping_distance_m": float(term.stop_initial_distance[0].item()),
    })
    return state


def physical_inputs(unwrapped, foot_ids, ankle_side_ids, all_joint_ids):
    robot = unwrapped.scene["robot"]
    contact = unwrapped.scene.sensors["contact_forces"]
    forces = contact.data.net_forces_w_history.torch[:, :, foot_ids, :]
    contacts = forces.norm(dim=-1).amax(dim=1) > 1.0
    ankle = torch.stack([
        (
            robot.data.applied_torque.torch[:, ids].abs()
            / robot.data.joint_effort_limits.torch[:, ids].abs().clamp_min(1e-6)
        ).amax(dim=1)
        for ids in ankle_side_ids
    ], dim=1)
    joint = robot.data.joint_vel.torch[:, all_joint_ids].abs() / robot.data.joint_vel_limits.torch[:, all_joint_ids].abs().clamp_min(1e-6)
    gravity = robot.data.projected_gravity_b.torch
    roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
    pitch = torch.atan2(gravity[:, 0], torch.sqrt(gravity[:, 1].square() + gravity[:, 2].square()))
    return contacts, ankle, joint, torch.stack((roll, pitch), dim=1), robot.data.root_ang_vel_b.torch


def controller_step(controller, observation, term, inputs):
    contacts, ankle, joint, roll_pitch, angular = inputs
    return controller.step(
        observation["policy"], term.skill_id == 1, term.stop_hold_progress, contacts.sum(dim=1),
        contacts=contacts, ankle_utilization=ankle,
        joint_velocity_utilization=joint.amax(dim=1), roll_pitch=roll_pitch, angular_velocity=angular,
    )


def make_direct_actor(num_envs: int, device: str) -> G1CommandResidualActor:
    placeholder = TensorDict({"policy": torch.zeros(num_envs, 152, device=device)}, batch_size=[num_envs])
    actor = G1CommandResidualActor(
        placeholder, {"actor": ["policy"]}, "actor", 37, hidden_dims=[256, 128, 128], activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
        trainable_skill_ids=[], train_stop_correction=False,
    ).to(device)
    checkpoint = torch.load(args_cli.checkpoint.resolve(strict=True), map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    actor.eval()
    actor.configure_stop_fixed_feedback(0.0, 0.0)
    return actor


def capture() -> None:
    cfg, agent_cfg = resolve_task_config("Isaac-Motion-Flat-G1-Command-Stop-Eval-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 1
    cfg.seed = 42
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device
    with launch_simulation(cfg, args_cli):
        raw = gym.make("Isaac-Motion-Flat-G1-Command-Stop-Eval-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        unwrapped = raw.unwrapped
        agent_cfg.device = unwrapped.device
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(args_cli.checkpoint.resolve(strict=True)), load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False})
        actor = runner.alg.actor
        actor.configure_stop_fixed_feedback(0.0, 0.0)
        policy = runner.get_inference_policy(device=unwrapped.device)
        controller = StopFixedFeedbackController(1, 37, unwrapped.device, current_config())
        term = unwrapped.command_manager.get_term("base_velocity")
        robot = unwrapped.scene["robot"]
        contact = unwrapped.scene.sensors["contact_forces"]
        _, foot_names = robot.find_bodies(".*_ankle_roll_link")
        foot_ids = [contact.body_names.index(name) for name in foot_names]
        left_ankle_ids, _ = robot.find_joints("left_ankle_.*_joint")
        right_ankle_ids, _ = robot.find_joints("right_ankle_.*_joint")
        ankle_side_ids = (left_ankle_ids, right_ankle_ids)
        all_joint_ids, _ = robot.find_joints(".*")
        wrapped.reset()
        episode = 0
        initial = robot_state(unwrapped)
        run_commands: list[float] = []
        snapshots: dict[int, dict] = {}
        outcomes: dict[int, dict] = {}
        trajectory: list[dict] = []
        previous_skill = int(term.skill_id[0].item())
        heading_values: list[float] = []
        joint_sat_sum = ankle_sat_sum = 0.0
        stop_steps = 0
        while episode <= max(args_cli.failure_ids):
            observations = wrapped.get_observations()
            skill = int(term.skill_id[0].item())
            if skill == 0:
                run_commands.append(float(observations["policy"][0, 9].item()))
            if skill == 1 and previous_skill != 1 and episode in args_cli.failure_ids:
                snapshots[episode] = snapshot(unwrapped, term, controller, initial, episode, run_commands)
                args_cli.snapshots.resolve().parent.mkdir(parents=True, exist_ok=True)
                torch.save(snapshots, args_cli.snapshots.resolve())
            inputs = physical_inputs(unwrapped, foot_ids, ankle_side_ids, all_joint_ids)
            correction, diag = controller_step(controller, observations, term, inputs)
            with torch.inference_mode():
                actions = policy(observations) + correction
            _, _, dones, infos = wrapped.step(actions)
            done = bool(dones[0].item())
            if skill == 1 and not done:
                contacts, ankle, joint, roll_pitch, angular = inputs
                heading_values.append(abs(float(term.heading_error[0].item())))
                joint_sat_sum += float((joint[0] >= 0.95).float().mean().item())
                ankle_sat_sum += float((ankle[0] >= 0.95).float().mean().item())
                stop_steps += 1
                if episode in args_cli.failure_ids:
                    trajectory.append({
                        "episode": episode, "step": stop_steps,
                        "heading_error_rad": float(term.heading_error[0].item()),
                        "yaw_rate_rps": float(robot.data.root_ang_vel_b.torch[0, 2].item()),
                        "left_contact": bool(contacts[0, 0].item()), "right_contact": bool(contacts[0, 1].item()),
                        "left_ankle_utilization": float(ankle[0, 0].item()), "right_ankle_utilization": float(ankle[0, 1].item()),
                        "feedback_raw": float(diag["raw_signal"][0].item()),
                        "feedback_filtered": float(diag["filtered_signal"][0].item()),
                        "feedback_applied_json": json.dumps(diag["feedback_action"][0].tolist()),
                        "spike_guard": bool(diag["spike_guard_active"][0].item()),
                        "hard_guard": bool(diag["hard_guard_active"][0].item()),
                    })
            hold_complete = skill == 1 and bool(term.stop_hold_complete[0].item())
            if done or hold_complete:
                if episode in args_cli.failure_ids:
                    count = max(stop_steps, 1)
                    timed_out = False
                    if done:
                        timeout = infos.get("time_outs") if isinstance(infos, dict) else None
                        timed_out = bool(timeout[0].item()) if timeout is not None else False
                    fall = done and not timed_out and skill == 1
                    heading = sum(heading_values) / count
                    saturation = joint_sat_sum / count > 0.05 or ankle_sat_sum / count > 0.20
                    failure = "fall" if fall else "saturation_failure" if saturation else "heading_failure" if heading > 0.12 else "success"
                    outcomes[episode] = {"failure": failure, "fall": fall, "heading_mean_rad": heading, "saturation": saturation}
                episode += 1
                if episode > max(args_cli.failure_ids):
                    break
                if hold_complete and not done:
                    wrapped.reset()
                controller.reset()
                initial = robot_state(unwrapped)
                run_commands = []
                heading_values = []
                joint_sat_sum = ankle_sat_sum = 0.0
                stop_steps = 0
                previous_skill = int(term.skill_id[0].item())
                continue
            previous_skill = skill
        torch.save(snapshots, args_cli.snapshots.resolve())
        report = {"master_seed": 42, "requested_ids": args_cli.failure_ids, "captured_ids": sorted(snapshots), "outcomes": outcomes}
        args_cli.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        with args_cli.output.resolve().with_suffix(".trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trajectory[0]))
            writer.writeheader(); writer.writerows(trajectory)
        print(json.dumps(report, indent=2))
        wrapped.close()


def restore_batch(unwrapped, term, states: list[dict]) -> None:
    robot = unwrapped.scene["robot"]
    device = unwrapped.device
    def on_device(value):
        return value.to(device=device) if isinstance(value, torch.Tensor) else value
    ids = torch.arange(len(states), device=unwrapped.device, dtype=torch.long)
    origins = unwrapped.scene.env_origins[ids]
    root_pose = torch.stack([torch.cat((on_device(state["root_position_local"]) + origins[i], on_device(state["root_quaternion"]))) for i, state in enumerate(states)])
    root_velocity = torch.stack([on_device(state["root_velocity"]) for state in states])
    joints = torch.stack([on_device(state["joint_position"]) for state in states])
    joint_vel = torch.stack([on_device(state["joint_velocity"]) for state in states])
    robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=ids)
    robot.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=ids)
    robot.write_joint_state_to_sim(joints, joint_vel, env_ids=ids)
    names = states[0]["command"]
    for name in names:
        target = getattr(term, name)
        values = torch.stack([on_device(state["command"][name]) for state in states])
        if name in {"target_position_w", "path_origin_w"}:
            values = values + origins[:, :2]
        target[ids] = values
    unwrapped.action_manager._action[ids] = torch.stack([on_device(state["action"]) for state in states])
    unwrapped.action_manager._prev_action[ids] = torch.stack([on_device(state["previous_action"]) for state in states])
    unwrapped.episode_length_buf[ids] = torch.stack([on_device(state["episode_length"]) for state in states])
    unwrapped.sim.forward()


def replay() -> None:
    saved = torch.load(args_cli.snapshots.resolve(strict=True), map_location="cpu", weights_only=False)
    selected = [saved[episode] for episode in args_cli.failure_ids if episode in saved]
    configs = candidates()
    num_envs = len(configs) * len(selected)
    cfg, _ = resolve_task_config("Isaac-Motion-Flat-G1-Command-Stop-Eval-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = num_envs
    cfg.seed = 42
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device
    with launch_simulation(cfg, args_cli):
        env = gym.make("Isaac-Motion-Flat-G1-Command-Stop-Eval-v0", cfg=cfg)
        unwrapped = env.unwrapped
        env.reset()
        actor = make_direct_actor(num_envs, unwrapped.device)
        term = unwrapped.command_manager.get_term("base_velocity")
        states = [state for _name, _cfg in configs for state in selected]
        restore_batch(unwrapped, term, states)
        observations = as_tensordict(unwrapped.observation_manager.compute())
        controllers = [StopFixedFeedbackController(len(selected), 37, unwrapped.device, item) for _name, item in configs]
        robot = unwrapped.scene["robot"]
        contact = unwrapped.scene.sensors["contact_forces"]
        _, foot_names = robot.find_bodies(".*_ankle_roll_link")
        foot_ids = [contact.body_names.index(name) for name in foot_names]
        left_ankle_ids, _ = robot.find_joints("left_ankle_.*_joint")
        right_ankle_ids, _ = robot.find_joints("right_ankle_.*_joint")
        ankle_side_ids = (left_ankle_ids, right_ankle_ids)
        all_joint_ids, _ = robot.find_joints(".*")
        accum = [{"heading": [], "yaw": [], "ankle": [], "joint": [], "trajectory": []} for _ in range(num_envs)]
        finished = torch.zeros(num_envs, dtype=torch.bool, device=unwrapped.device)
        while not bool(finished.all().item()):
            inputs = physical_inputs(unwrapped, foot_ids, ankle_side_ids, all_joint_ids)
            with torch.inference_mode():
                inference_actions = actor(observations)
            with torch.inference_mode(False):
                actions = inference_actions.clone()
            all_diag = {}
            for candidate_id, controller in enumerate(controllers):
                sl = slice(candidate_id * len(selected), (candidate_id + 1) * len(selected))
                sliced_obs = TensorDict({"policy": observations["policy"][sl]}, batch_size=[len(selected)])
                sliced_inputs = tuple(value[sl] for value in inputs)
                correction, diag = controller_step(controller, sliced_obs, term_slice(term, sl), sliced_inputs)
                actions[sl] += correction
                all_diag[candidate_id] = diag
            observations, _, terminated, truncated, _ = env.step(actions)
            observations = as_tensordict(observations)
            done = terminated | truncated
            contacts, ankle, joint, roll_pitch, angular = inputs
            for env_id in range(num_envs):
                if finished[env_id]:
                    continue
                candidate_id, failure_index = divmod(env_id, len(selected))
                diag = all_diag[candidate_id]
                local = failure_index
                item = accum[env_id]
                item["heading"].append(abs(float(term.heading_error[env_id].item())))
                item["yaw"].append(abs(float(robot.data.root_ang_vel_b.torch[env_id, 2].item())))
                item["ankle"].append(float(ankle[env_id].amax().item()))
                item["joint"].append(float(joint[env_id].amax().item()))
                item["trajectory"].append({
                    "candidate": configs[candidate_id][0], "episode": selected[failure_index]["episode"],
                    "step": len(item["heading"]), "heading_error_rad": item["heading"][-1], "yaw_rate_abs_rps": item["yaw"][-1],
                    "left_contact": bool(contacts[env_id, 0].item()), "right_contact": bool(contacts[env_id, 1].item()),
                    "left_ankle_utilization": float(ankle[env_id, 0].item()), "right_ankle_utilization": float(ankle[env_id, 1].item()),
                    "feedback_norm": float(diag["feedback_norm"][local].item()),
                    "spike_guard": bool(diag["spike_guard_active"][local].item()),
                    "hard_guard": bool(diag["hard_guard_active"][local].item()),
                    "recovery_scale": float(diag["contact_recovery_scale"][local].item()),
                    "safety_scale": float(diag["combined_safety_scale"][local].item()),
                })
                hold = bool(term.stop_hold_complete[env_id].item())
                if bool(done[env_id].item()) or hold:
                    item["fall"] = bool(terminated[env_id].item())
                    finished[env_id] = True
        outcomes = []
        trajectories = []
        for env_id, item in enumerate(accum):
            candidate_id, failure_index = divmod(env_id, len(selected))
            fall = bool(item.get("fall", False))
            heading = sum(item["heading"]) / max(len(item["heading"]), 1)
            ankle_sat = sum(value >= 0.95 for value in item["ankle"]) / max(len(item["ankle"]), 1) > 0.20
            failure = "fall" if fall else "saturation_failure" if ankle_sat else "heading_failure" if heading > 0.12 else "success"
            outcomes.append({
                "candidate": configs[candidate_id][0], "episode": selected[failure_index]["episode"],
                "failure": failure, "fall": fall, "heading_mean_rad": heading,
                "yaw_p99_rps": percentile(item["yaw"], 99), "yaw_max_rps": max(item["yaw"], default=0.0),
                "ankle_saturation": ankle_sat, "ankle_max_utilization": max(item["ankle"], default=0.0),
            })
            trajectories.extend(item["trajectory"])
        summaries = []
        for name, config_item in configs:
            rows = [row for row in outcomes if row["candidate"] == name]
            summaries.append({
                "name": name, "config": config_item.as_dict(), "episodes": len(rows),
                "fall_rate": sum(row["fall"] for row in rows) / len(rows),
                "success_rate": sum(row["failure"] == "success" for row in rows) / len(rows),
                "heading_failure_rate": sum(row["failure"] == "heading_failure" for row in rows) / len(rows),
                "ankle_saturation_rate": sum(row["ankle_saturation"] for row in rows) / len(rows),
                "heading_mean_rad": sum(row["heading_mean_rad"] for row in rows) / len(rows),
                "yaw_p99_rps": sum(row["yaw_p99_rps"] for row in rows) / len(rows),
                "yaw_max_rps": max(row["yaw_max_rps"] for row in rows),
            })
        report = {"failure_ids": [state["episode"] for state in selected], "candidates": summaries, "outcomes": outcomes}
        args_cli.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        with args_cli.output.resolve().with_suffix(".trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trajectories[0]))
            writer.writeheader(); writer.writerows(trajectories)
        print(json.dumps({"failure_ids": report["failure_ids"], "candidates": summaries}, indent=2))
        env.close()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(round((len(ordered) - 1) * q / 100.0)), len(ordered) - 1)] if ordered else 0.0


class term_slice:
    """Minimal slice exposing the command tensors consumed by the controller."""
    def __init__(self, term, selection):
        self.skill_id = term.skill_id[selection]
        self.stop_hold_progress = term.stop_hold_progress[selection]


def main() -> None:
    capture() if args_cli.mode == "capture" else replay()


if __name__ == "__main__":
    main()
