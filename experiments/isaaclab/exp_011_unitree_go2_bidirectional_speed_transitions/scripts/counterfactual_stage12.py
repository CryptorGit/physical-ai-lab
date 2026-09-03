"""Same-seed action-prefix replay for local tangential-slip controllability."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage12_tangential_slip_reward_directionality"
RAW = OUT / "raw"
CHECKPOINT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage11_tangential_slip_reduction/checkpoints/model_initial.pt"
SEED = 20272901
DT = 0.02
SPEEDS = (0.2, 0.4, 0.6, 1.2, 2.0)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--speed-index", type=int, required=True)
parser.add_argument("--mode", choices=("standard", "linearity"), default="standard")
parser.add_argument("--dimensions", type=int, default=12, choices=range(1, 13))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app
sys.argv = [sys.argv[0]]

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: E402,F401
import go2_bidirectional.stage11_tasks  # noqa: E402,F401
from go2_bidirectional.stage11_tasks.command import wrap_angle, yaw_xyzw  # noqa: E402
from go2_bidirectional.stage6_endpoint_protocol import quat_xyzw_to_gravity_tilt_torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402

speed = SPEEDS[args.speed_index]
num_envs = 100 if args.mode == "standard" else 20
cfg, agent_cfg = resolve_task_config(
    "Isaac-Exp011-Go2-Tangential-Slip-v0", "rsl_rl_cfg_entry_point"
)
cfg.scene.num_envs = num_envs
cfg.seed = SEED
cfg.episode_length_s = 60.0
cfg.observations.policy.enable_corruption = False
cfg.events.base_external_force_torque = None
cfg.events.push_robot = None
if args.device:
    cfg.sim.device = args.device
    agent_cfg.device = args.device
raw = gym.make("Isaac-Exp011-Go2-Tangential-Slip-v0", cfg=cfg)
wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
agent_cfg = handle_deprecated_rsl_rl_cfg(
    agent_cfg, __import__("importlib.metadata").metadata.version("rsl-rl-lib")
)
runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(
    str(CHECKPOINT),
    load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": True, "rnd": False},
    strict=True, map_location=runner.device,
)
runner.alg.actor.eval()
env = wrapped.unwrapped
robot = env.scene["robot"]
command = env.command_manager.get_term("base_velocity")
slip_term = env.reward_manager.get_term_cfg("go2_contact_tangential_slip").func
all_ids = torch.arange(num_envs, device=env.device)
branch_search_start = round(2.0 / DT)
# The replay only needs the first verified stable-contact state.  A fixed
# 0.5-s search window preserves the pre-registered selection rule while
# avoiding redundant simulation after every branch has been identified.
branch_search_end = round(2.5 / DT)
max_horizon = 8


def configure(time_s):
    command.source_speed.fill_(speed)
    command.target_speed.fill_(speed)
    command.source_hold_s.zero_()
    command.elapsed_s.fill_(time_s - DT)
    command._update_command()


def capture_state(previous_action):
    return {
        "root": torch.cat((
            robot.data.root_pos_w.torch, robot.data.root_quat_w.torch,
            robot.data.root_lin_vel_w.torch, robot.data.root_ang_vel_w.torch,
        ), dim=1).detach().cpu(),
        "joint": torch.cat((robot.data.joint_pos.torch, robot.data.joint_vel.torch), dim=1).detach().cpu(),
        "previous_action": previous_action.detach().cpu(),
        "contact_age": slip_term.contact_age.detach().cpu(),
        "heading_state": torch.stack((
            command.heading_reference, command.heading_gate, command.heading_raw,
            command.heading_command, command.heading_active.float(),
            command.acquisition_age.float(),
        ), dim=1).detach().cpu(),
    }


def replay(dimension=None, delta=0.0, baseline_branch_steps=None):
    # Same-seed replay contract: reset every RNG that can feed Isaac Lab
    # reset/event sampling before replaying the identical action prefix.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # Reinitialize the PhysX runtime between branches so contact warm-start
    # caches from the preceding trajectory cannot contaminate a same-seed
    # replay.  This does not write robot state; the ordinary environment reset
    # below remains the sole state initialization mechanism.
    env.sim.reset()
    # Pass the seed through Gymnasium's reset contract so it is applied
    # immediately before ManagerBasedEnv._reset_idx, rather than in a
    # separate call that other manager work could advance.
    raw.reset(seed=SEED)
    command._resample_command(all_ids)
    slip_term.reset(all_ids)
    previous_action = torch.zeros(num_envs, 12, device=env.device)
    branch_steps = (
        torch.full((num_envs,), -1, dtype=torch.long, device=env.device)
        if baseline_branch_steps is None else baseline_branch_steps.to(env.device)
    )
    prebranch = None
    branch_contact = torch.zeros(num_envs, 4, dtype=torch.bool)
    cumulative = torch.zeros(num_envs, max_horizon)
    tangent_samples = torch.zeros(num_envs, max_horizon, 4)
    speed_error = torch.zeros(num_envs, max_horizon)
    heading_error = torch.zeros(num_envs, max_horizon)
    tilt = torch.zeros(num_envs, max_horizon)
    contact_loss = torch.zeros(num_envs, max_horizon, dtype=torch.bool)
    flight = torch.zeros(num_envs, max_horizon, dtype=torch.bool)
    fall = torch.zeros(num_envs, max_horizon, dtype=torch.bool)
    saturation = torch.zeros(num_envs, max_horizon, dtype=torch.bool)
    dangerous = torch.zeros(num_envs, max_horizon, 4, dtype=torch.bool)
    max_step = branch_search_end + max_horizon + 1
    for step in range(max_step):
        configure(step * DT)
        observation = wrapped.get_observations()
        with torch.no_grad():
            action = runner.alg.actor(observation, stochastic_output=False)
        action = action.clone()
        if baseline_branch_steps is None and branch_search_start <= step <= branch_search_end:
            eligible = (
                (branch_steps < 0)
                & (slip_term.contact_age >= 8).any(1)
                & ((robot.data.root_lin_vel_b.torch[:, 0] - speed).abs() <= 0.15)
            )
            branch_steps[eligible] = step
        branch_now = branch_steps == step
        if branch_now.any():
            branch_now_cpu = branch_now.cpu()
            if prebranch is None:
                prebranch = capture_state(previous_action)
            else:
                current = capture_state(previous_action)
                for key in prebranch:
                    prebranch[key][branch_now_cpu] = current[key][branch_now_cpu]
            branch_contact[branch_now_cpu] = (
                slip_term.last_normal_force[branch_now] > 5.0
            ).cpu()
            if dimension is not None:
                action[branch_now, dimension] += delta
        # Step outside inference_mode: reward/contact state is mutable and must
        # remain a normal tensor so the next same-seed replay can reset it.
        _, _, dones, _ = wrapped.step(action)
        vx = robot.data.root_lin_vel_b.torch[:, 0]
        yaw = yaw_xyzw(robot.data.root_quat_w.torch)
        heading = wrap_angle(command.heading_reference - yaw)
        gravity_tilt = quat_xyzw_to_gravity_tilt_torch(robot.data.root_quat_w.torch)
        contact = slip_term.last_normal_force > 5.0
        vel_ratio = (
            robot.data.joint_vel.torch.abs()
            / robot.data.joint_vel_limits.torch.abs().clamp_min(1.0e-6)
        ).amax(1)
        torque_ratio = (
            robot.data.applied_torque.torch.abs()
            / robot.data.joint_effort_limits.torch.abs().clamp_min(1.0e-6)
        ).amax(1)
        for horizon in range(max_horizon):
            active = (step - branch_steps) == horizon
            if not active.any():
                continue
            ids = torch.where(active)[0]
            ids_cpu = ids.cpu()
            score = slip_term.last_raw_score[ids].cpu()
            cumulative[ids_cpu, horizon:] += score[:, None]
            tangent_samples[ids_cpu, horizon] = slip_term.last_foot_speed[ids].cpu()
            speed_error[ids_cpu, horizon] = (vx[ids] - speed).abs().cpu()
            heading_error[ids_cpu, horizon] = heading[ids].abs().cpu()
            tilt[ids_cpu, horizon] = gravity_tilt[ids].cpu()
            lost = branch_contact[ids_cpu] & ~contact[ids].cpu()
            contact_loss[ids_cpu, horizon] = lost.any(1)
            flight[ids_cpu, horizon] = (contact[ids].sum(1) == 0).cpu()
            fall[ids_cpu, horizon] = dones[ids].bool().cpu()
            saturation[ids_cpu, horizon] = ((vel_ratio[ids] >= 0.95) | (torque_ratio[ids] >= 0.95)).cpu()
            dangerous[ids_cpu, horizon] = (
                (slip_term.last_foot_speed[ids] > 0.30) & (slip_term.contact_age[ids] >= 3)
            ).cpu()
        previous_action = action.detach()
    if (branch_steps < 0).any():
        raise RuntimeError(f"stable branch state unavailable for {int((branch_steps < 0).sum())} envs")
    return {
        "branch_steps": branch_steps.cpu(), "prebranch": prebranch,
        "branch_contact": branch_contact.cpu(), "cumulative_score": cumulative,
        "tangent_speed_p95": tangent_samples.quantile(0.95, dim=1),
        "dangerous_steps": dangerous.sum(1), "speed_error": speed_error,
        "heading_error": heading_error, "tilt": tilt, "contact_loss": contact_loss,
        "flight": flight, "fall": fall, "saturation": saturation,
    }


baseline = replay()
rows = []
matching = []
deltas = (0.02, -0.02) if args.mode == "standard" else (0.01, -0.01, 0.04, -0.04)
for dimension in range(args.dimensions):
    for delta in deltas:
        result = replay(dimension, delta, baseline["branch_steps"])
        errors = {}
        for key in baseline["prebranch"]:
            difference = (result["prebranch"][key] - baseline["prebranch"][key]).abs()
            errors[key] = float(difference.max())
        match = max(errors.values()) <= 1.0e-5
        matching.append({
            "dimension": dimension, "delta": delta, "matched": match, **errors,
        })
        for episode in range(num_envs):
            rows.append({
                "speed": speed, "episode": episode, "seed": SEED + episode,
                "dimension": dimension, "delta": delta, "prebranch_matched": match,
                "branch_step": int(baseline["branch_steps"][episode]),
                "branch_contact_pattern": int(
                    (baseline["branch_contact"][episode].long() * torch.tensor([1, 2, 4, 8])).sum()
                ),
                "baseline_cumulative_score_8": float(baseline["cumulative_score"][episode, 7]),
                "perturbed_cumulative_score_8": float(result["cumulative_score"][episode, 7]),
                "baseline_tangent_p95": float(baseline["tangent_speed_p95"][episode].max()),
                "perturbed_tangent_p95": float(result["tangent_speed_p95"][episode].max()),
                "baseline_dangerous_steps": int(baseline["dangerous_steps"][episode].sum()),
                "perturbed_dangerous_steps": int(result["dangerous_steps"][episode].sum()),
                "baseline_speed_error_8": float(baseline["speed_error"][episode].mean()),
                "perturbed_speed_error_8": float(result["speed_error"][episode].mean()),
                "baseline_heading_8": float(baseline["heading_error"][episode].max()),
                "perturbed_heading_8": float(result["heading_error"][episode].max()),
                "baseline_tilt_8": float(baseline["tilt"][episode].max()),
                "perturbed_tilt_8": float(result["tilt"][episode].max()),
                "new_contact_loss": bool(
                    result["contact_loss"][episode].any() and not baseline["contact_loss"][episode].any()
                ),
                "flight": bool(result["flight"][episode].any()),
                "fall": bool(result["fall"][episode].any()),
                "saturation_increase": bool(
                    result["saturation"][episode].any() and not baseline["saturation"][episode].any()
                ),
                "horizon_scores": result["cumulative_score"][episode, [0, 1, 3, 7]].tolist(),
            })
        print(f"STAGE12 counterfactual speed={speed:g} dim={dimension} delta={delta:+.2f}", flush=True)

payload = {
    "speed": speed, "mode": args.mode, "episodes": num_envs,
    "branch_steps": baseline["branch_steps"], "baseline": baseline,
    "rows": rows, "matching": matching, "checkpoint": str(CHECKPOINT),
}
path = RAW / f"counterfactual_{str(speed).replace('.', 'p')}_{args.mode}.pt"
torch.save(payload, path)
manifest = {
    "speed": speed, "mode": args.mode, "episodes": num_envs,
    "branch_states": num_envs, "variants": len(deltas) * args.dimensions,
    "path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "bytes": path.stat().st_size,
    "all_prebranch_matched": all(item["matched"] for item in matching),
    "maximum_prebranch_error": max(
        max(value for key, value in item.items() if key not in ("dimension", "delta", "matched"))
        for item in matching
    ),
    "state_setter": False, "teleport": False, "state_injection": False,
    "production_ppo_update": 0,
}
(OUT / f"counterfactual_manifest_{str(speed).replace('.', 'p')}_{args.mode}.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
wrapped.close()
simulation_app.close()
