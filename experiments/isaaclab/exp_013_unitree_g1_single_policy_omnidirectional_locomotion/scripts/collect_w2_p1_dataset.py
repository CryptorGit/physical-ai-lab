"""Collect bounded W2-P1 supervised datasets with frozen actors."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import gymnasium as gym
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
EXP = HERE.parent.parent
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w2_p1_practical_stop_endpoint_acquisition"
)
RAW = OUT / "raw"
PARENT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
)
TEACHER = REPO / (
    "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/"
    "stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
)
sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))
sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"))
sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: F401
import g1_omnidirectional.tasks  # noqa: F401
from g1_omnidirectional.policy import FrozenGaitActor
from g1_omnidirectional.yaw_calibration import calibrate_yaw
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("stop_recovery", "steady_stop", "moving_retention", "start_retention"), required=True)
parser.add_argument("--max-envs", type=int, default=1800)
parser.add_argument("--record-stride", type=int, default=5)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def minjerk(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0, 1)
    return x**3 * (10 - 15*x + 6*x*x)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jobs_for_mode(mode: str) -> list[dict]:
    jobs = []
    if mode == "stop_recovery":
        for direction in range(0, 360, 45):
            for yaw in (-.3, 0., .3):
                for episode in range(300):
                    jobs.append({"subgroup": "STOP_RECOVERY", "direction": direction, "speed": .3,
                                 "yaw": yaw, "episode": episode, "dynamic": False})
    elif mode == "steady_stop":
        for episode in range(1000):
            jobs.append({"subgroup": "STEADY_STOP", "direction": 0, "speed": 0.,
                         "yaw": 0., "episode": episode, "dynamic": False})
    elif mode == "start_retention":
        for direction in range(0, 360, 45):
            for yaw in (-.3, 0., .3):
                for episode in range(100):
                    jobs.append({"subgroup": "START_RETENTION", "direction": direction,
                                 "speed": .3, "yaw": yaw, "episode": episode, "dynamic": False})
    else:
        for direction in [i * 22.5 for i in range(16)]:
            for episode in range(200):
                jobs.append({"subgroup": "ZERO_YAW_TRANSLATION", "direction": direction,
                             "speed": .3, "yaw": 0., "episode": episode, "dynamic": False})
        for speed in (.6, 1.2):
            for episode in range(300):
                jobs.append({"subgroup": "FORWARD_ANCHOR", "direction": 0., "speed": speed,
                             "yaw": 0., "episode": episode, "dynamic": False})
        for yaw in (-.3, .3):
            for episode in range(300):
                jobs.append({"subgroup": "PURE_YAW", "direction": 0., "speed": 0.,
                             "yaw": yaw, "episode": episode, "dynamic": False})
        for direction in range(0, 360, 45):
            for yaw in (-.3, 0., .3):
                for episode in range(100):
                    jobs.append({"subgroup": "MOVING_TURN", "direction": direction,
                                 "speed": .3, "yaw": yaw, "episode": episode, "dynamic": False})
        independence = [(270,-.3),(270,.3),(90,-.3),(90,.3),(135,.3),(45,-.3),
                        (225,.3),(315,-.3),(180,-.3),(180,.3)]
        for direction, yaw in independence:
            for episode in range(150):
                jobs.append({"subgroup": "INDEPENDENCE", "direction": direction,
                             "speed": .3, "yaw": yaw, "episode": episode, "dynamic": False})
        transitions = [(-.3,.3),(.3,-.3),(0.,.3),(0.,-.3)]
        for direction in [None, 0, 45, 90, 135, 180, 225, 270, 315]:
            for initial_yaw, yaw in transitions:
                for episode in range(50):
                    jobs.append({"subgroup": "DYNAMIC_YAW_ENDPOINT",
                                 "direction": 0 if direction is None else direction,
                                 "speed": 0. if direction is None else .3, "yaw": yaw,
                                 "initial_yaw": initial_yaw, "episode": episode, "dynamic": True})
    condition_ids: dict[str, int] = {}
    for job in jobs:
        job["condition"] = (
            f"{job['subgroup']}:{job['direction']}:{job['speed']}:{job['yaw']}:"
            f"{job.get('initial_yaw', 'static')}"
        )
        job["condition_id"] = condition_ids.setdefault(job["condition"], len(condition_ids))
    return jobs


def command_for(
    mode: str,
    target: torch.Tensor,
    initial: torch.Tensor,
    dynamic: torch.Tensor,
    t: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one full command batch without per-environment Python work."""
    label_teacher = torch.zeros(target.shape[0], dtype=torch.bool, device=target.device)
    if mode == "stop_recovery":
        scale = 1.0 if t < 3.0 else 1.0 - float(minjerk(torch.tensor((t - 3.0) / 1.5)))
        return target * scale, label_teacher | (t >= 4.5)
    if mode == "steady_stop":
        return torch.zeros_like(target), ~label_teacher
    if mode == "start_retention":
        scale = 0.0 if t < 3.0 else float(minjerk(torch.tensor((t - 3.0) / 1.5)))
        return target * scale, label_teacher | (t < 3.0)
    if t < 4.0:
        dynamic_command = initial
    elif t < 6.0:
        alpha = float(minjerk(torch.tensor((t - 4.0) / 2.0)))
        dynamic_command = initial + alpha * (target - initial)
    else:
        dynamic_command = target
    return torch.where(dynamic[:, None], dynamic_command, target), label_teacher


def main() -> None:
    jobs = jobs_for_mode(args.mode)
    duration = {"stop_recovery": 10.5, "steady_stop": 8., "moving_retention": 8., "start_retention": 8.5}[args.mode]
    cfg, agent_cfg = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = min(args.max_envs, len(jobs)); cfg.episode_length_s = max(15., duration+1)
    cfg.seed = {"stop_recovery": 20276031, "steady_stop": 20276032,
                "moving_retention": 20276033, "start_retention": 20276034}[args.mode]
    if args.device: cfg.sim.device = agent_cfg.device = args.device
    RAW.mkdir(parents=True, exist_ok=True)
    chunks = []
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg),
                                     clip_actions=agent_cfg.clip_actions)
        env = wrapped.unwrapped; device = env.device
        parent = FrozenGaitActor(PARENT).to(device).eval(); teacher = FrozenGaitActor(TEACHER).to(device).eval()
        robot = env.scene["robot"]; sensor = env.scene["contact_forces"]
        feet = sensor.find_bodies(".*_ankle_roll_link")[0]
        robot_feet = robot.find_bodies(".*_ankle_roll_link")[0]
        command = env.command_manager.get_term("base_velocity"); command.external_override_enabled = True
        cursor = 0; chunk_index = 0
        while cursor < len(jobs):
            actual = jobs[cursor:cursor+env.num_envs]; count = len(actual)
            padded = actual + [actual[i % count] for i in range(env.num_envs-count)] if count < env.num_envs else actual
            direction = torch.tensor([math.radians(j["direction"]) for j in padded], device=device)
            speed = torch.tensor([j["speed"] for j in padded], device=device)
            yaw_target = torch.tensor([j["yaw"] for j in padded], device=device)
            target = torch.stack((speed * direction.cos(), speed * direction.sin(), yaw_target), dim=-1)
            initial = target.clone()
            initial[:, 2] = torch.tensor([j.get("initial_yaw", j["yaw"]) for j in padded], device=device)
            dynamic = torch.tensor([j.get("dynamic", False) for j in padded], device=device)
            env.reset(env_ids=torch.arange(env.num_envs, device=device)); obs = wrapped.get_observations().to(device)
            fields = {key: [] for key in ("observation","gait_cmd","physical_command","actor_command",
                                           "source_action","teacher_action","target_action","label_source",
                                           "phase","translation_speed","absolute_yaw_rate","contact","flight")}
            fall = torch.zeros(env.num_envs,dtype=torch.bool,device=device); slip=torch.zeros_like(fall)
            slip_streak=torch.zeros(env.num_envs,dtype=torch.long,device=device)
            final_speed=torch.zeros(env.num_envs,device=device); final_yaw=torch.zeros_like(final_speed); final_n=0
            for step in range(int(round(duration/env.step_dt))):
                t=step*env.step_dt; physical,label_teacher=command_for(args.mode,target,initial,dynamic,t)
                command.external_override[:,:2]=physical[:,:2]; command.external_override[:,2]=calibrate_yaw(physical[:,2])
                if step==0: command._update_command(); obs=wrapped.get_observations().to(device)
                gait=torch.zeros(env.num_envs,device=device)
                with torch.inference_mode():
                    source_action=parent(obs["policy"],gait); teacher_action=teacher(obs["policy"],gait)
                target_action=torch.where(label_teacher[:,None],teacher_action,source_action)
                runtime_action=torch.where(label_teacher[:,None],teacher_action,source_action)
                # The first three seconds of the start protocol establish a teacher-produced
                # stop state.  They are initialization, not W1B start-retention labels.
                record = step % args.record_stride == 0 and not (
                    args.mode == "start_retention" and t < 3.0
                )
                if record:
                    contact=(sensor.data.net_forces_w_history[:,-1,feet,:].norm(dim=-1)>5)
                    fields["observation"].append(obs["policy"].cpu()); fields["gait_cmd"].append(gait.cpu())
                    fields["physical_command"].append(physical.cpu())
                    actor_cmd=physical.clone(); actor_cmd[:,2]=calibrate_yaw(actor_cmd[:,2]); fields["actor_command"].append(actor_cmd.cpu())
                    fields["source_action"].append(source_action.cpu()); fields["teacher_action"].append(teacher_action.cpu())
                    fields["target_action"].append(target_action.cpu()); fields["label_source"].append(label_teacher.cpu())
                    phase=torch.where(label_teacher,torch.ones_like(gait),torch.zeros_like(gait)); fields["phase"].append(phase.cpu())
                    fields["translation_speed"].append(torch.linalg.vector_norm(robot.data.root_lin_vel_b[:,:2],dim=-1).cpu())
                    fields["absolute_yaw_rate"].append(robot.data.root_ang_vel_b[:,2].abs().cpu())
                    fields["contact"].append(contact.cpu()); fields["flight"].append((~contact.any(-1)).cpu())
                obs,_,done,extras=wrapped.step(runtime_action); obs=obs.to(device)
                timeout=extras.get("time_outs",torch.zeros_like(done)).bool(); fall |= done.bool() & ~timeout
                force=sensor.data.net_forces_w_history[:,-1,feet,:].norm(dim=-1)
                foot_speed=torch.linalg.vector_norm(robot.data.body_lin_vel_w[:,robot_feet,:2],dim=-1)
                slipping=((foot_speed>.55)&(force>5)).any(-1); slip_streak=torch.where(slipping,slip_streak+1,torch.zeros_like(slip_streak)); slip|=slip_streak>=5
                if t>=duration-2:
                    final_speed+=torch.linalg.vector_norm(robot.data.root_lin_vel_b[:,:2],dim=-1); final_yaw+=robot.data.root_ang_vel_b[:,2].abs(); final_n+=1
            data={key:torch.stack(value)[:, :count] for key,value in fields.items()}
            speed=(final_speed/final_n)[:count].cpu(); yaw=(final_yaw/final_n)[:count].cpu()
            success=(speed<=.08)&(yaw<=.08)&~fall[:count].cpu()&~slip[:count].cpu()
            if args.mode == "start_retention":
                # Start labels are retained only when the moving endpoint is reached safely.
                success=(speed>.08)&~fall[:count].cpu()&~slip[:count].cpu()
                keep=torch.nonzero(success,as_tuple=False).flatten()
                data={key:value[:,keep] for key,value in data.items()}
            elif args.mode == "steady_stop":
                # Only formal-stop teacher episodes are valid endpoint labels.  Failures are
                # retained separately in the manifest rather than entering supervised data.
                keep=torch.nonzero(success,as_tuple=False).flatten()
                data={key:value[:,keep] for key,value in data.items()}
            else: keep=torch.arange(count)
            data.update({
                "episode_id":torch.arange(cursor,cursor+count)[keep],
                "condition_id":torch.tensor([j["condition_id"] for j in actual])[keep],
                "condition":[actual[int(i)]["condition"] for i in keep],
                "subgroup":[actual[int(i)]["subgroup"] for i in keep],
                "final_speed":speed[keep],"final_abs_yaw":yaw[keep],
                "fall":fall[:count].cpu()[keep],"slip":slip[:count].cpu()[keep],
                "record_stride":args.record_stride,
            })
            path=RAW/f"{args.mode}_chunk_{chunk_index:03d}.pt"; torch.save(data,path); chunks.append(path)
            cursor+=count; chunk_index+=1
            print(json.dumps({"mode":args.mode,"processed":cursor,"total":len(jobs),"saved":str(path)}),flush=True)
        wrapped.close()
        requested = len(jobs)
        saved = sum(torch.load(path,map_location="cpu",weights_only=False)["episode_id"].numel() for path in chunks)
        manifest={
            "mode":args.mode,"requested_episodes":requested,"saved_episodes":saved,
            "excluded_failed_episodes": requested - saved if args.mode in {"steady_stop", "start_retention"} else 0,
            "record_stride":args.record_stride,"source_parent_sha256":sha(PARENT),"teacher_sha256":sha(TEACHER),
            "chunks":[{"path":str(p.relative_to(REPO)).replace("\\","/"),"sha256":sha(p)} for p in chunks],
            "subgroup_counts":dict(Counter(j["subgroup"] for j in jobs)),
            "runtime_teacher_usage": "diagnostic dataset generation only",
        }
        (OUT/f"{args.mode}_dataset_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")


if __name__ == "__main__": main()
