"""Collect no-state-copy symmetric finite-difference sensitivity in Isaac Sim."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import gymnasium as gym
import numpy as np
import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage2_dynamics_sensitive_distillation"
CFG_PATH = EXP / "configs/stage2_dynamics_sensitive_distillation.yaml"
sys.path[:0] = [
    str(EXP / "src"),
    str(REPO / "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions/src"),
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    str(REPO / "experiments/isaaclab/exp_006_unitree_g1_command_skills/src"),
]

import g1_command_skills.tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
import g1_walk_centered.tasks  # noqa: E402,F401
import isaaclab_tasks  # noqa: E402,F401
from g1_walk_centered.command_contract import MotionCommand  # noqa: E402
from g1_walk_centered.experts import load_run_expert, load_walk_expert  # noqa: E402
from g1_walk_centered.experts.adapters import canonical_state_from_legacy_observation, to_run_observation, to_walk_observation  # noqa: E402
from g1_walk_centered.tasks.stage7r_action import WalkToRunTransitionActor152  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mj(value):
    value = value.clamp(0, 1)
    return 10 * value**3 - 15 * value**4 + 6 * value**5


def roll_pitch(quat):
    w, x, y, z = quat.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x.square() + y.square()))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1, 1))
    return roll, pitch


parser = argparse.ArgumentParser()
parser.add_argument("--diagnostic-delta", type=float, default=None)
parser.add_argument("--diagnostic-cycles", type=int, default=None)
parser.add_argument("--diagnostic-envs", type=int, default=None)
parser.add_argument("--output-suffix", type=str, default="")
parser.add_argument("--sign", choices=("plus", "minus"), required=True)
parser.add_argument("--seed-offset", type=int, default=0)
parser.add_argument("--regime", choices=("walk_steady", "run_steady", "walk_to_run"), default=None)
parser.add_argument("--cycle-index", type=int, default=None)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def main():
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    for teacher in cfg["teachers"].values():
        if sha(REPO / teacher["path"]) != teacher["sha256"]:
            raise RuntimeError(f"teacher hash mismatch: {teacher['path']}")
    task_cfg, agent_cfg = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    task_cfg.scene.num_envs = int(args.diagnostic_envs or cfg["sensitivity"]["physical_envs"])
    task_cfg.seed = int(cfg["experiment"]["seed"])
    task_cfg.episode_length_s = 35.0
    task_cfg.sim.device = cfg["experiment"]["device"]
    args.device = cfg["experiment"]["device"]
    OUT.mkdir(parents=True, exist_ok=True)

    with launch_simulation(task_cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=task_cfg), clip_actions=agent_cfg.clip_actions)
        env, device, n = wrapped.unwrapped, wrapped.unwrapped.device, wrapped.num_envs
        dt = float(env.step_dt)
        walk = load_walk_expert(REPO / cfg["teachers"]["walk"]["path"], device=device)
        run = load_run_expert(REPO / cfg["teachers"]["run"]["path"], device=device)
        stand = load_walk_expert(REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt", device=device)
        stw = load_walk_expert(REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100/model_0.pt", device=device)
        wtr = WalkToRunTransitionActor152(run.actor).to(device)
        wtr.load_state_dict(torch.load(REPO / cfg["teachers"]["walk_to_run"]["path"], map_location=device, weights_only=False)["actor"], strict=True)
        for module in (walk.actor, run.actor, stand.actor, stw.actor, wtr):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        robot, command_term = env.scene["robot"], env.command_manager.get_term("base_velocity")
        sensor = env.scene.sensors["contact_forces"]
        feet, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        joints, joint_names = robot.find_joints(".*")
        critical_tokens = cfg["sensitivity"]["critical_joint_tokens"]
        critical = [index for index, name in enumerate(joint_names) if any(token in name for token in critical_tokens)]
        expected = [
            "left_hip_pitch_joint", "right_hip_pitch_joint", "left_hip_roll_joint", "right_hip_roll_joint",
            "left_hip_yaw_joint", "right_hip_yaw_joint", "left_knee_joint", "right_knee_joint",
            "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_roll_joint",
        ]
        if [joint_names[index] for index in critical] != expected:
            raise RuntimeError(f"critical action order mismatch: {[joint_names[index] for index in critical]}")
        ankle = [index for index, name in enumerate(joint_names) if "ankle" in name]
        knee = [index for index, name in enumerate(joint_names) if "knee" in name]

        def contacts():
            forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :]
            contact = forces.norm(dim=-1).amax(1) > 5
            return contact, forces

        def canonical():
            legacy = wrapped.get_observations()["policy"]
            return canonical_state_from_legacy_observation(legacy, heading_w_rad=robot.data.heading_w.torch)

        def set_command(speed, heading, gain=0.8):
            error = torch.atan2(torch.sin(heading - robot.data.heading_w.torch), torch.cos(heading - robot.data.heading_w.torch))
            yaw = (gain * error - 0.1 * robot.data.root_ang_vel_b.torch[:, 2]).clamp(-1.5, 1.5)
            command_term.vel_command_b.zero_()
            command_term.vel_command_b[:, 0], command_term.vel_command_b[:, 2] = speed, yaw
            return MotionCommand(speed, heading, target_yaw_rate_radps=yaw)

        def prepare_walk(targets):
            wrapped.reset()
            heading = robot.data.heading_w.torch.clone()
            phase = torch.zeros(n, dtype=torch.long, device=device)
            elapsed, good = torch.zeros(n, device=device), torch.zeros(n, device=device)
            switches = torch.zeros(n, dtype=torch.long, device=device)
            previous_support = torch.zeros(n, dtype=torch.long, device=device)
            previous = torch.zeros(n, 37, device=device)
            for step in range(700):
                state = canonical()
                speed = torch.where(phase < 2, torch.zeros_like(targets), torch.where(phase == 2, targets * mj(elapsed / 1.5), targets))
                command = set_command(speed, heading)
                with torch.no_grad():
                    actions = torch.stack((stand(state, command), stw(state, command), walk(state, command)))
                    selector = torch.where(phase < 2, torch.zeros_like(phase), torch.where(phase == 2, torch.ones_like(phase), torch.full_like(phase, 2)))
                    action = actions[selector, torch.arange(n, device=device)]
                    _, _, dones, _ = wrapped.step(action)
                previous.copy_(action)
                contact, _ = contacts()
                support = contact[:, 0].long() + 2 * contact[:, 1].long()
                actual = robot.data.root_lin_vel_b.torch[:, 0]
                settled = (actual.abs() < .1) & contact.all(1) & ~dones.bool()
                good = torch.where((phase == 0) & settled, good + dt, torch.where(phase == 0, 0, good))
                advance = (phase == 0) & (good >= .4); phase[advance], elapsed[advance], good[advance] = 1, 0, 0
                advance = (phase == 1) & (elapsed >= .8); phase[advance], elapsed[advance] = 2, 0
                changed = (support != previous_support) & ((support == 1) | (support == 2)) & (phase == 2)
                switches[changed] += 1
                acquire = (phase == 2) & ((actual - targets).abs() <= .2) & (switches >= 2) & ~dones.bool()
                good = torch.where(acquire, good + dt, torch.where(phase == 2, 0, good))
                advance = (phase == 2) & (good >= .4); phase[advance], elapsed[advance], good[advance] = 3, 0, 0
                valid = (phase == 3) & ((actual - targets).abs() <= .2) & ~dones.bool()
                good = torch.where(valid, good + dt, torch.where(phase == 3, 0, good))
                previous_support = support
                elapsed += dt
                if int((good >= .3).sum()) >= math.ceil(.8 * n):
                    return heading, previous, good >= .3
            raise RuntimeError("WALK preparation failed")

        def prepare_run(targets):
            heading, previous, ready = prepare_walk(torch.full((n,), 1.2, device=device))
            good = torch.zeros(n, device=device)
            for step in range(500):
                state = canonical()
                speed = 1.2 + (targets - 1.2) * mj(torch.full((n,), step * dt / 1.4, device=device))
                command = set_command(speed, heading, 1.5)
                with torch.no_grad():
                    action = run(state, command)
                    _, _, dones, _ = wrapped.step(action)
                previous.copy_(action)
                valid = ((robot.data.root_lin_vel_b.torch[:, 0] - targets).abs() <= .2) & ~dones.bool()
                good = torch.where(valid, good + dt, torch.zeros_like(good))
            ready &= good >= .3
            if int(ready.sum()) < math.ceil(.5 * n):
                raise RuntimeError("RUN preparation failed")
            return heading, previous, ready

        def snapshot(air_time, last_contact, last_landing):
            contact, forces = contacts()
            support = contact[:, 0].long() + 2 * contact[:, 1].long()
            roll, pitch = roll_pitch(robot.data.root_quat_w.torch)
            effort = robot.data.applied_torque.torch[:, joints].abs()
            continuous = torch.cat([
                robot.data.root_lin_vel_b.torch,
                roll[:, None], pitch[:, None],
                robot.data.root_ang_vel_b.torch,
                forces[:, :, :, 2].abs().mean(1),
                robot.data.body_pos_w.torch[:, feet, 2] - env.scene.env_origins[:, 2, None],
                air_time, last_contact,
                robot.data.joint_pos.torch[:, critical],
                robot.data.joint_vel.torch[:, critical],
                effort[:, ankle].mean(1, keepdim=True).repeat(1, 2),
                robot.data.joint_vel.torch[:, knee],
            ], dim=1)
            flight = ~contact.any(1)
            gait = torch.where(flight, torch.full_like(support, 2), torch.where(contact.all(1), torch.zeros_like(support), torch.ones_like(support)))
            discrete = torch.cat([
                contact.long(), torch.nn.functional.one_hot(support, 4),
                flight[:, None].long(), torch.nn.functional.one_hot(last_landing.clamp(0, 2), 3),
                torch.nn.functional.one_hot(gait, 3),
            ], dim=1)
            return continuous, discrete, support

        def replay(regime, cycle, sign):
            seed = (
                int(cfg["experiment"]["seed"]) + int(args.seed_offset) + cycle
                + {"walk_steady": 1000, "run_steady": 2000, "walk_to_run": 3000}[regime]
            )
            wrapped.seed(seed); torch.manual_seed(seed)
            if regime == "walk_steady":
                targets = torch.tensor([.6, .8, 1., 1.2], device=device).repeat((n + 3) // 4)[:n]
                heading, previous, ready = prepare_walk(targets)
                branch_at = torch.zeros(n, dtype=torch.long, device=device)
            elif regime == "run_steady":
                targets = torch.tensor([2.4, 2.6, 2.8], device=device).repeat((n + 2) // 3)[:n]
                heading, previous, ready = prepare_run(targets)
                branch_at = torch.zeros(n, dtype=torch.long, device=device)
            else:
                targets = torch.where(torch.arange(n, device=device) % 2 == 0, 2.6, 2.8)
                heading, previous, ready = prepare_walk(torch.full((n,), 1.2, device=device))
                branch_at = 10 + 10 * (torch.arange(n, device=device) % 7)
            dimension_local = torch.arange(n, device=device) % len(critical)
            dimension = torch.tensor(critical, device=device)[dimension_local]
            branched = torch.zeros(n, dtype=torch.bool, device=device)
            relative = torch.full((n,), -1, dtype=torch.long, device=device)
            air_time = torch.zeros(n, 2, device=device)
            last_contact = torch.zeros(n, 2, device=device)
            last_landing = torch.zeros(n, dtype=torch.long, device=device)
            previous_contact, _ = contacts()
            branch_payload = {}
            outcomes = {h: {} for h in cfg["sensitivity"]["horizons_steps"]}
            for step in range(int(branch_at.max()) + 9):
                state = canonical()
                if regime == "walk_steady":
                    speed = targets; command = set_command(speed, heading); obs = to_walk_observation(state, command)
                    with torch.no_grad(): teacher_action = walk(state, command)
                elif regime == "run_steady":
                    speed = targets; command = set_command(speed, heading, 1.5); obs = to_walk_observation(state, command)
                    with torch.no_grad(): teacher_action = run(state, command)
                else:
                    speed = 1.2 + (targets - 1.2) * mj(torch.full((n,), step * dt / 1.4, device=device))
                    command = set_command(speed, heading, 1.5); obs = to_walk_observation(state, command)
                    with torch.no_grad(): teacher_action = wtr(to_run_observation(state, command, route="RUN"))
                just = ready & ~branched & (step >= branch_at)
                if just.any():
                    continuous, discrete, support = snapshot(air_time, last_contact, last_landing)
                    ids_cpu = torch.nonzero(just).flatten().detach().cpu().numpy()
                    if not branch_payload:
                        branch_payload = {
                            "obs": np.full((n, obs.shape[1]), np.nan, dtype=np.float32),
                            "action": np.full((n, teacher_action.shape[1]), np.nan, dtype=np.float32),
                            "root": np.full((n, 3), np.nan, dtype=np.float32),
                            "joint": np.full((n, robot.data.joint_pos.torch.shape[1]), np.nan, dtype=np.float32),
                            "velocity": np.full((n, 3 + robot.data.joint_vel.torch.shape[1]), np.nan, dtype=np.float32),
                            "phase": np.full(n, -1, dtype=np.int64),
                            "target": targets.detach().cpu().numpy(),
                            "ready": ready.detach().cpu().numpy(),
                            "dimension": dimension.detach().cpu().numpy(),
                            "captured": np.zeros(n, dtype=bool),
                        }
                    root = robot.data.root_pos_w.torch - env.scene.env_origins
                    velocity = torch.cat((robot.data.root_lin_vel_b.torch, robot.data.joint_vel.torch), 1)
                    branch_payload["obs"][ids_cpu] = obs[just].detach().cpu().numpy()
                    branch_payload["action"][ids_cpu] = teacher_action[just].detach().cpu().numpy()
                    branch_payload["root"][ids_cpu] = root[just].detach().cpu().numpy()
                    branch_payload["joint"][ids_cpu] = robot.data.joint_pos.torch[just].detach().cpu().numpy()
                    branch_payload["velocity"][ids_cpu] = velocity[just].detach().cpu().numpy()
                    branch_payload["phase"][ids_cpu] = support[just].detach().cpu().numpy()
                    branch_payload["captured"][ids_cpu] = True
                    branched |= just; relative[just] = 0
                action = teacher_action.clone()
                ids = torch.nonzero(just).flatten()
                if len(ids):
                    action[ids, dimension[ids]] += float(sign) * float(
                        args.diagnostic_delta or cfg["sensitivity"]["perturbation_delta"]
                    )
                action.clamp_(-1, 1)
                with torch.no_grad():
                    _, _, _, _ = wrapped.step(action)
                previous.copy_(action)
                relative[branched] += 1
                contact, _ = contacts()
                landing = contact & ~previous_contact
                last_landing = torch.where(landing[:, 0], torch.ones_like(last_landing), torch.where(landing[:, 1], torch.full_like(last_landing, 2), last_landing))
                air_time = torch.where(contact, torch.zeros_like(air_time), air_time + dt)
                last_contact = torch.where(contact, last_contact + dt, torch.zeros_like(last_contact))
                previous_contact = contact
                for horizon in cfg["sensitivity"]["horizons_steps"]:
                    take = relative == horizon
                    if take.any():
                        continuous, discrete, _ = snapshot(air_time, last_contact, last_landing)
                        ids_cpu = torch.nonzero(take).flatten().detach().cpu().numpy()
                        if not outcomes[horizon]:
                            outcomes[horizon] = {
                                "continuous": np.full((n, continuous.shape[1]), np.nan, dtype=np.float32),
                                "discrete": np.full((n, discrete.shape[1]), -1, dtype=np.int8),
                                "captured": np.zeros(n, dtype=bool),
                            }
                        outcomes[horizon]["continuous"][ids_cpu] = continuous[take].detach().cpu().numpy()
                        outcomes[horizon]["discrete"][ids_cpu] = discrete[take].detach().cpu().numpy()
                        outcomes[horizon]["captured"][ids_cpu] = True
            if not branch_payload:
                raise RuntimeError("no branch payload")
            branch_payload["ready"] &= branch_payload["captured"]
            for horizon in cfg["sensitivity"]["horizons_steps"]:
                if not outcomes[horizon]:
                    raise RuntimeError(f"no outcomes captured for horizon {horizon}")
                branch_payload["ready"] &= outcomes[horizon]["captured"]
            return branch_payload, outcomes

        # Each sign is collected in a fresh Isaac application. Re-resetting the
        # same application is not a valid deterministic counterfactual replay.
        # A separate merge step retains only exactly matched physical env IDs.
        sign_value = +1 if args.sign == "plus" else -1
        records = []
        regimes_to_run = (args.regime,) if args.regime else ("walk_steady", "run_steady", "walk_to_run")
        cycles_to_run = (
            (args.cycle_index,) if args.cycle_index is not None
            else range(int(args.diagnostic_cycles or cfg["sensitivity"]["cycles_per_regime"]))
        )
        for regime in regimes_to_run:
            for cycle in cycles_to_run:
                branch, sign_out = replay(regime, cycle, sign_value)
                indices = np.where(branch["ready"])[0]
                for env_id in indices:
                    item = {
                        "regime": regime, "cycle": cycle, "physical_env_id": int(env_id),
                        "target_speed_mps": float(branch["target"][env_id]),
                        "support_phase": int(branch["phase"][env_id]),
                        "action_dimension": int(branch["dimension"][env_id]),
                        "observation": branch["obs"][env_id].astype(np.float32),
                        "teacher_action": branch["action"][env_id].astype(np.float32),
                        "branch_root": branch["root"][env_id].astype(np.float32),
                        "branch_joint": branch["joint"][env_id].astype(np.float32),
                        "branch_velocity": branch["velocity"][env_id].astype(np.float32),
                    }
                    for horizon in cfg["sensitivity"]["horizons_steps"]:
                        item[f"continuous_{horizon}"] = sign_out[horizon]["continuous"][env_id].astype(np.float32)
                        item[f"discrete_{horizon}"] = sign_out[horizon]["discrete"][env_id].astype(np.int8)
                    records.append(item)
                print(
                    f"[stage2 sensitivity] sign={args.sign} regime={regime} cycle={cycle} ready={len(indices)}",
                    flush=True,
                )
        packed = {}
        for key in records[0]:
            dtype = object if key == "regime" else None
            packed[key] = np.asarray([row[key] for row in records], dtype=dtype)
        suffix = f"_{args.output_suffix}" if args.output_suffix else ""
        np.savez_compressed(OUT / f"sensitivity_replay_{args.sign}{suffix}.npz", **packed)
        (OUT / f"counterfactual_replay_{args.sign}{suffix}.json").write_text(json.dumps({
            "fresh_isaac_application": True,
            "sign": args.sign,
            "branch_states": len(records),
            "delta": float(args.diagnostic_delta or cfg["sensitivity"]["perturbation_delta"]),
            "state_copy": False,
            "teacher_gradients": 0,
        }, indent=2) + "\n")
        return

        records, matches = [], []
        for regime in ("walk_steady", "run_steady", "walk_to_run"):
            for cycle in range(int(args.diagnostic_cycles or cfg["sensitivity"]["cycles_per_regime"])):
                plus_branch, plus_out = replay(regime, cycle, +1)
                minus_branch, minus_out = replay(regime, cycle, -1)
                ready = plus_branch["ready"] & minus_branch["ready"]
                root_error = np.max(np.abs(plus_branch["root"][ready] - minus_branch["root"][ready]))
                joint_error = np.max(np.abs(plus_branch["joint"][ready] - minus_branch["joint"][ready]))
                velocity_error = np.max(np.abs(plus_branch["velocity"][ready] - minus_branch["velocity"][ready]))
                matches.append({"regime": regime, "cycle": cycle, "matched_states": int(ready.sum()), "root_position_max_error_m": float(root_error), "joint_position_max_error_rad": float(joint_error), "velocity_max_error": float(velocity_error)})
                indices = np.where(ready)[0]
                for env_id in indices:
                    item = {
                        "regime": regime, "cycle": cycle, "physical_env_id": int(env_id),
                        "target_speed_mps": float(plus_branch["target"][env_id]),
                        "support_phase": int(plus_branch["phase"][env_id]),
                        "action_dimension": int(plus_branch["dimension"][env_id]),
                        "observation": plus_branch["obs"][env_id].astype(np.float32),
                        "teacher_action": plus_branch["action"][env_id].astype(np.float32),
                    }
                    for horizon in cfg["sensitivity"]["horizons_steps"]:
                        item[f"plus_continuous_{horizon}"] = plus_out[horizon]["continuous"][env_id].astype(np.float32)
                        item[f"minus_continuous_{horizon}"] = minus_out[horizon]["continuous"][env_id].astype(np.float32)
                        item[f"plus_discrete_{horizon}"] = plus_out[horizon]["discrete"][env_id].astype(np.int8)
                        item[f"minus_discrete_{horizon}"] = minus_out[horizon]["discrete"][env_id].astype(np.int8)
                    records.append(item)
                print(f"[stage2 sensitivity] regime={regime} cycle={cycle} matched={int(ready.sum())}", flush=True)

        # Packed NPZ is diagnostic data, not a production state snapshot.
        packed = {}
        for key in records[0]:
            if key in ("regime",):
                packed[key] = np.asarray([row[key] for row in records], dtype=object)
            else:
                packed[key] = np.asarray([row[key] for row in records])
        suffix = f"_{args.output_suffix}" if args.output_suffix else ""
        np.savez_compressed(OUT / f"dynamic_sensitivity_samples{suffix}.npz", **packed)
        tolerances = cfg["sensitivity"]
        all_match = all(
            row["root_position_max_error_m"] <= tolerances["state_match_root_tolerance_m"]
            and row["joint_position_max_error_rad"] <= tolerances["state_match_joint_tolerance_rad"]
            and row["velocity_max_error"] <= tolerances["state_match_velocity_tolerance"]
            for row in matches
        )
        (OUT / f"prebranch_state_matching{suffix}.json").write_text(json.dumps({
            "method": "same reset seed, physical env ID, source route and prebranch teacher actions; no state copy",
            "comparisons": matches, "all_within_tolerance": all_match,
            "tolerances": {
                "root_m": tolerances["state_match_root_tolerance_m"],
                "joint_rad": tolerances["state_match_joint_tolerance_rad"],
                "velocity": tolerances["state_match_velocity_tolerance"],
            },
        }, indent=2) + "\n")
        counts = {}
        for regime in ("walk_steady", "run_steady", "walk_to_run"):
            mask = packed["regime"] == regime
            counts[regime] = int(mask.sum())
        (OUT / f"counterfactual_branch_manifest{suffix}.json").write_text(json.dumps({
            "total_branch_states": len(records), "regime_counts": counts,
            "critical_joint_names": [joint_names[index] for index in critical],
            "critical_action_indices": critical, "waist_included": False,
            "perturbation_delta_normalized": float(args.diagnostic_delta or cfg["sensitivity"]["perturbation_delta"]),
            "physical_target_delta_rad": float(args.diagnostic_delta or cfg["sensitivity"]["perturbation_delta"]) * .5,
            "horizons_steps": cfg["sensitivity"]["horizons_steps"],
            "teacher_gradients": 0, "state_copy": False,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
