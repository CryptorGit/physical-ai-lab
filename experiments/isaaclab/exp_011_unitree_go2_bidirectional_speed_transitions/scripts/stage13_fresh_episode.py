"""Run exactly one Stage 13 episode in one fresh Isaac application process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import struct
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage13_fresh_process_counterfactual_replay"
RAW = OUT / "raw"
CHECKPOINT = REPO / (
    "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
    "stage11_tangential_slip_reduction/checkpoints/model_initial.pt"
)
DT = 0.02

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-id", required=True)
parser.add_argument("--speed", type=float, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--branch-step", type=int, default=-1)
parser.add_argument("--action-dimension", type=int, default=-1)
parser.add_argument("--delta", type=float, default=0.0)
parser.add_argument("--duration", type=float, default=8.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app
sys.argv = [sys.argv[0]]

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: E402,F401
import go2_bidirectional.stage11_tasks  # noqa: E402,F401
from go2_bidirectional.stage11_tasks.command import wrap_angle, yaw_xyzw  # noqa: E402
from go2_bidirectional.stage6_endpoint_protocol import quat_xyzw_to_gravity_tilt_torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_bytes(value, dtype):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype).reshape(-1).tobytes(order="C")


def digest_parts(parts):
    digest = hashlib.sha256()
    for value, dtype in parts:
        payload = raw_bytes(value, dtype)
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def tensor_or_zeros(obj, name, shape):
    value = getattr(obj, name, None)
    if value is None:
        return torch.zeros(shape, device=env.device)
    return value.torch if hasattr(value, "torch") else value


started = time.time()
RAW.mkdir(parents=True, exist_ok=True)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

cfg, agent_cfg = resolve_task_config(
    "Isaac-Exp011-Go2-Tangential-Slip-v0", "rsl_rl_cfg_entry_point"
)
cfg.scene.num_envs = 1
cfg.seed = args.seed
cfg.episode_length_s = max(60.0, args.duration + 1.0)
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
    load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": True, "rnd": False},
    strict=True,
    map_location=runner.device,
)
runner.alg.actor.eval()
env = wrapped.unwrapped
robot = env.scene["robot"]
command = env.command_manager.get_term("base_velocity")
slip = env.reward_manager.get_term_cfg("go2_contact_tangential_slip").func
sensor = env.scene.sensors["stage11_contact"]
all_ids = torch.arange(1, device=env.device)

# Seed through Gymnasium immediately before the only reset in this process.
raw.reset(seed=args.seed)
command._resample_command(all_ids)
slip.reset(all_ids)

previous_action = torch.zeros(1, 12, device=env.device)
hash_rows = []
trace = {
    key: [] for key in (
        "root", "joint", "observation", "previous_action", "mean_action", "applied_action",
        "requested_speed", "actual_speed", "yaw_command", "heading_reference",
        "heading_error", "heading_gate", "heading_active", "acquisition_age",
        "foot_contact", "contact_age", "air_time", "normal_force", "contact_count",
        "raw_slip_score", "tangential_speed", "friction_utilization", "gravity_tilt",
        "speed_error", "fall", "termination", "timeout", "saturation",
    )
}
termination_step = None
max_steps = round(args.duration / DT)
if args.branch_step >= 0:
    max_steps = min(max_steps, args.branch_step + 9)

for step in range(max_steps):
    command.source_speed.fill_(args.speed)
    command.target_speed.fill_(args.speed)
    command.source_hold_s.zero_()
    command.elapsed_s.fill_(step * DT - DT)
    command._update_command()
    observations = wrapped.get_observations()
    policy_obs = observations["policy"]
    with torch.no_grad():
        mean_action = runner.alg.actor(observations, stochastic_output=False)
    applied_action = mean_action.clone()
    if step == args.branch_step and args.action_dimension >= 0:
        applied_action[:, args.action_dimension] += args.delta

    root = torch.cat((
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        robot.data.root_lin_vel_w.torch,
        robot.data.root_ang_vel_w.torch,
    ), dim=1)
    joint = torch.cat((robot.data.joint_pos.torch, robot.data.joint_vel.torch), dim=1)
    yaw = yaw_xyzw(robot.data.root_quat_w.torch)
    heading_error = wrap_angle(command.heading_reference - yaw)
    foot_contact = slip.last_normal_force > 5.0
    air_time_data = getattr(sensor.data, "current_air_time", None)
    if air_time_data is None:
        air_time = torch.zeros(1, 4, device=env.device)
    else:
        air_time = air_time_data.torch if hasattr(air_time_data, "torch") else air_time_data
    _, _, _, _, count_raw, _ = sensor.contact_view.get_contact_data(
        dt=sensor._sim_physics_dt
    )
    contact_count = wp.to_torch(count_raw).reshape(1, 4, -1)[:, :, 0].long()
    controller_float = torch.stack((
        command.heading_reference,
        heading_error,
        command.heading_gate,
        command.heading_command,
    ), dim=1)
    controller_int = torch.stack((
        command.heading_active.long(),
        command.acquisition_age.long(),
    ), dim=1)
    state_hash = digest_parts([
        (root, "<f4"), (joint, "<f4"), (policy_obs, "<f4"),
        (previous_action, "<f4"), (mean_action, "<f4"),
        (torch.tensor([[args.speed]], device=env.device), "<f4"),
        (robot.data.root_lin_vel_b.torch[:, :1], "<f4"),
        (controller_float, "<f4"), (controller_int, "<i8"),
        (foot_contact, "|u1"), (slip.contact_age, "<i8"), (air_time, "<f4"),
        (slip.last_normal_force, "<f4"), (contact_count, "<i8"),
        (torch.tensor([[step]], device=env.device), "<i8"),
    ])
    observation_hash = digest_parts([(policy_obs, "<f4")])
    action_hash = digest_parts([(mean_action, "<f4")])
    applied_action_hash = digest_parts([(applied_action, "<f4")])
    contact_hash = digest_parts([
        (foot_contact, "|u1"), (slip.contact_age, "<i8"), (air_time, "<f4"),
        (slip.last_normal_force, "<f4"), (contact_count, "<i8"),
    ])
    controller_hash = digest_parts([(controller_float, "<f4"), (controller_int, "<i8")])
    hash_rows.append({
        "step": step,
        "state_hash": state_hash,
        "action_hash": action_hash,
        "applied_action_hash": applied_action_hash,
        "observation_hash": observation_hash,
        "contact_hash": contact_hash,
        "controller_state_hash": controller_hash,
    })

    trace["root"].append(root.detach().cpu())
    trace["joint"].append(joint.detach().cpu())
    trace["observation"].append(policy_obs.detach().cpu())
    trace["previous_action"].append(previous_action.detach().cpu())
    trace["mean_action"].append(mean_action.detach().cpu())
    trace["applied_action"].append(applied_action.detach().cpu())
    trace["requested_speed"].append(torch.tensor([args.speed]))
    trace["actual_speed"].append(robot.data.root_lin_vel_b.torch[:, 0].detach().cpu())
    trace["yaw_command"].append(command.heading_command.detach().cpu())
    trace["heading_reference"].append(command.heading_reference.detach().cpu())
    trace["heading_error"].append(heading_error.detach().cpu())
    trace["heading_gate"].append(command.heading_gate.detach().cpu())
    trace["heading_active"].append(command.heading_active.detach().cpu())
    trace["acquisition_age"].append(command.acquisition_age.detach().cpu())
    trace["foot_contact"].append(foot_contact.detach().cpu())
    trace["contact_age"].append(slip.contact_age.detach().cpu())
    trace["air_time"].append(air_time.detach().cpu())
    trace["normal_force"].append(slip.last_normal_force.detach().cpu())
    trace["contact_count"].append(contact_count.detach().cpu())

    _, _, dones, extras = wrapped.step(applied_action)
    friction = slip.diagnostic_friction_utilization()
    timeout = extras.get("time_outs", torch.zeros_like(dones)).bool()
    fall = dones.bool() & ~timeout
    gravity_tilt = quat_xyzw_to_gravity_tilt_torch(robot.data.root_quat_w.torch)
    velocity_ratio = (
        robot.data.joint_vel.torch.abs()
        / robot.data.joint_vel_limits.torch.abs().clamp_min(1.0e-6)
    ).amax(1)
    torque_ratio = (
        robot.data.applied_torque.torch.abs()
        / robot.data.joint_effort_limits.torch.abs().clamp_min(1.0e-6)
    ).amax(1)
    trace["raw_slip_score"].append(slip.last_raw_score.detach().cpu())
    trace["tangential_speed"].append(slip.last_foot_speed.detach().cpu())
    trace["friction_utilization"].append(friction.detach().cpu())
    trace["gravity_tilt"].append(gravity_tilt.detach().cpu())
    trace["speed_error"].append(
        (robot.data.root_lin_vel_b.torch[:, 0] - args.speed).abs().detach().cpu()
    )
    trace["fall"].append(fall.detach().cpu())
    trace["termination"].append(dones.bool().detach().cpu())
    trace["timeout"].append(timeout.detach().cpu())
    trace["saturation"].append(((velocity_ratio >= 0.95) | (torque_ratio >= 0.95)).detach().cpu())
    previous_action = applied_action.detach()
    if bool(dones.item()):
        termination_step = step
        break

trace = {key: torch.stack(values) for key, values in trace.items()}
payload = {
    "run_id": args.run_id,
    "speed": args.speed,
    "seed": args.seed,
    "branch_step": args.branch_step,
    "action_dimension": args.action_dimension,
    "delta": args.delta,
    "dt": DT,
    "trace": trace,
    "hash_rows": hash_rows,
    "termination_step": termination_step,
}
raw_path = RAW / f"{args.run_id}.pt"
torch.save(payload, raw_path)
summary = {
    "run_id": args.run_id,
    "status": "COMPLETE",
    "speed": args.speed,
    "seed": args.seed,
    "branch_step": args.branch_step,
    "action_dimension": args.action_dimension,
    "delta": args.delta,
    "steps": len(hash_rows),
    "termination_step": termination_step,
    "raw_trace": str(raw_path.resolve()),
    "raw_trace_sha256": sha_file(raw_path),
    "trace_hashes": hash_rows,
    "checkpoint": str(CHECKPOINT.resolve()),
    "checkpoint_sha256": sha_file(CHECKPOINT),
    "python_executable": sys.executable,
    "working_directory": str(Path.cwd()),
    "cuda_device": str(env.device),
    "runtime": {
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    },
    "seeds": {
        "python": args.seed, "numpy": args.seed, "torch_cpu": args.seed,
        "torch_cuda": args.seed, "environment": args.seed,
    },
    "process_started_unix": started,
    "process_finished_unix": time.time(),
    "state_injection": 0,
    "resets": 1,
    "episodes": 1,
    "action_variants": 1 if args.action_dimension >= 0 else 0,
}
(RAW / f"{args.run_id}.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
wrapped.close()
simulation_app.close()
print(json.dumps(summary, sort_keys=True))
