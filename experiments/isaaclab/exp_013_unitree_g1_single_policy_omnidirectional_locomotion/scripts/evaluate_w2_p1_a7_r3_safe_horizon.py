"""Measure the shortest A7-teacher takeover horizon from an A4-V2 B0 state."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
EXP = HERE.parent.parent
BASE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
M0 = BASE / "phase_w2_p1_a7_m0_accepted_env_masked_ppo_preflight"
A4 = BASE / "phase_w2_p1_a4_versioned_b0_label_contract_preflight"
R2 = BASE / "phase_w2_p1_r2_long_horizon_group_balanced_stop_integration"
STOP = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
sys.path[:0] = [
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),
    str(EXP / "src"), str(HERE.parent),
]
import isaaclab_tasks  # noqa:F401,E402
import g1_omnidirectional.tasks  # noqa:F401,E402
from g1_omnidirectional.policy import FrozenGaitActor  # noqa:E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa:E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa:E402
from w2_p1_a5_common import reproduce_a4  # noqa:E402

parser = argparse.ArgumentParser()
parser.add_argument("--teacher-policy", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--batch", type=int, default=5)
parser.add_argument("--episodes", type=int, default=200)
parser.add_argument("--horizon", type=int, required=True, choices=HORIZONS if "HORIZONS" in globals() else (2,4,6,8,12,16,24))
parser.add_argument("--direction", type=int, required=True)
parser.add_argument("--yaw", type=float, required=True)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]

N = 1024
HORIZONS = (2, 4, 6, 8, 12, 16, 24)


def minjerk(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0, 1)
    return x**3 * (10 - 15*x + 6*x*x)


def reconstruct_candidate(device: str) -> torch.nn.Module:
    candidate, fingerprint, _, _, _ = reproduce_a4(torch.device(device))
    expected = "db65a3069d665b8012fd9d264b7fd54e629a22d25b05a9ff793e23bfc549ac5f"
    if fingerprint["tensor_hash"] != expected:
        raise RuntimeError(f"A4 V2 candidate reproduction failed: {fingerprint['tensor_hash']}")
    return candidate


cfg, agent = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
cfg.scene.num_envs = N
cfg.episode_length_s = 12.0
cfg.seed = 20278501
cfg.observations.policy.enable_corruption = False
if args.device:
    cfg.sim.device = agent.device = args.device
masks = json.loads((M0 / "a7_environment_masks.json").read_text(encoding="utf-8"))["batches"][str(args.batch)]
ids_cpu = torch.nonzero(torch.tensor(masks["heldout_mask"], dtype=torch.bool)).flatten()[:args.episodes]
active_cpu = torch.zeros(N, dtype=torch.bool)
active_cpu[ids_cpu] = True
rows: list[dict[str, object]] = []

with launch_simulation(cfg, args):
    wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg), clip_actions=agent.clip_actions)
    env = wrapped.unwrapped
    robot = env.scene["robot"]
    sensor = env.scene["contact_forces"]
    sensor_feet = sensor.find_bodies(".*_ankle_roll_link")[0]
    robot_feet = robot.find_bodies(".*_ankle_roll_link")[0]
    command = env.command_manager.get_term("base_velocity")
    command.external_override_enabled = True
    stop = FrozenGaitActor(STOP).to(env.device).eval()
    gait = torch.zeros(N, device=env.device)
    all_ids = torch.arange(N, device=env.device)
    active = active_cpu.to(env.device)
    for horizon in (args.horizon,):
        for direction in (args.direction,):
            for yaw in (args.yaw,):
                # ReplayV2 lifecycle: stop teacher is the only policy object before reset/roll-in.
                for _batch in range(args.batch + 1):
                    env.reset(env_ids=all_ids)
                    command.external_override.zero_(); command._update_command()
                    observations = wrapped.get_observations().to(env.device)
                    for _ in range(150):
                        with torch.inference_mode(): action = stop(observations["policy"], gait)
                        observations, _, _, _ = wrapped.step(action); observations = observations.to(env.device)
                teacher = FrozenGaitActor(Path(args.teacher_policy)).to(env.device).eval()
                candidate = reconstruct_candidate(env.device)
                target = torch.zeros(N, 3, device=env.device)
                angle = math.radians(direction)
                target[active, 0] = .3 * math.cos(angle); target[active, 1] = .3 * math.sin(angle); target[active, 2] = yaw
                # B0 remains the A4 V2 stop-consistent candidate action.
                command.external_override.zero_(); command._update_command()
                observations = wrapped.get_observations().to(env.device)
                with torch.inference_mode(): ca = candidate(observations["policy"], gait); ha = stop(observations["policy"], gait)
                observations, _, _, _ = wrapped.step(torch.where(active[:, None], ca, ha)); observations = observations.to(env.device)
                fall = torch.zeros(N, dtype=torch.bool, device=env.device); slip = fall.clone(); impact = fall.clone()
                streak = torch.zeros(N, dtype=torch.long, device=env.device); sustained = torch.zeros_like(streak); acquired = torch.zeros_like(fall)
                endpoint_vector = torch.zeros(N, device=env.device); endpoint_yaw = torch.zeros_like(endpoint_vector); endpoint_count = torch.zeros_like(streak)
                for step in range(275):
                    physical = target * minjerk(torch.tensor(step / 75, device=env.device))
                    actor_command = physical.clone(); actor_command[:, 2] = torch.where(actor_command[:, 2] > 0, 1.5*actor_command[:, 2], actor_command[:, 2])
                    command.external_override.zero_(); command.external_override[active] = actor_command[active]; command._update_command()
                    observations = wrapped.get_observations().to(env.device)
                    with torch.inference_mode():
                        ta = teacher(observations["policy"], gait); ca = candidate(observations["policy"], gait); ha = stop(observations["policy"], gait)
                        active_action = ta if step < horizon else ca
                        action = torch.where(active[:, None], active_action, ha)
                    observations, _, done, extras = wrapped.step(action); observations = observations.to(env.device)
                    timeout = extras.get("time_outs", torch.zeros_like(done)).bool(); fall |= done.bool() & ~timeout
                    force = sensor.data.net_forces_w_history[:, -1, sensor_feet, :].norm(dim=-1); contact = force > 5
                    foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1)
                    streak = torch.where(((foot_speed > .55) & contact).any(1), streak+1, torch.zeros_like(streak)); slip |= streak >= 5; impact |= force.amax(1) > 3500
                    actual = robot.data.root_lin_vel_b[:, :2]; actual_yaw = robot.data.root_ang_vel_b[:, 2]
                    vector = torch.linalg.vector_norm(actual-target[:, :2], dim=1); target_angle = torch.atan2(target[:,1],target[:,0]); actual_angle = torch.atan2(actual[:,1],actual[:,0]); direction_error = torch.atan2(torch.sin(actual_angle-target_angle),torch.cos(actual_angle-target_angle)).abs()*180/math.pi
                    yaw_pass = torch.where(target[:,2].abs()<1e-8, actual_yaw.abs()<=.2, (actual_yaw*target[:,2]>0)&((actual_yaw-target[:,2]).abs()<=.2))
                    passed = (vector<=.25)&(direction_error<=25)&yaw_pass&contact.any(1)&~fall&~slip&~impact
                    if step >= 75:
                        sustained = torch.where(passed, sustained+1, torch.zeros_like(sustained)); acquired |= (sustained>=10)&(step<=224)
                    if step >= 175:
                        endpoint_vector += vector; endpoint_yaw += (actual_yaw-target[:,2]).abs(); endpoint_count += 1
                endpoint = (endpoint_vector/endpoint_count.clamp_min(1)<=.25)&(endpoint_yaw/endpoint_count.clamp_min(1)<=.2)&~fall&~slip&~impact
                sel = ids_cpu.to(env.device)
                rows.append({"horizon":horizon,"direction":direction,"yaw":yaw,"episodes":args.episodes,"endpoint_success":float(endpoint[sel].float().mean()),"acquisition_0p20":float(acquired[sel].float().mean()),"fall_rate":float(fall[sel].float().mean()),"dangerous_slip_rate":float(slip[sel].float().mean()),"impact_rate":float(impact[sel].float().mean())})
                del candidate, teacher
                print(json.dumps({"horizon":horizon,"direction":direction,"yaw":yaw}), flush=True)
    result = {"teacher":args.teacher_policy,"candidate":"A4 V2 in-memory exact reconstruction","B0":"A4 V2 stop-maintenance action","episodes":args.episodes,"row":rows[0],"diagnostic_only":True,"runtime_teacher_authorized":False}
    Path(args.output).write_text(json.dumps(result, indent=2)+"\n",encoding="utf-8")
    wrapped.close()
