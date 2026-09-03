"""Paired teacher/student endpoint evaluation for one frozen std multiplier."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2m_stochastic_gait_endpoint_robustness"
K = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2k_gait_latent_preflight"
L = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2l_gait_conditioned_gaussian_std_preflight"
WALK = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
RUN = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-18_00-44-32_stage4_1024_500/model_5244.pt"
CORE = SCRIPT.with_name("evaluate_stage2k_gait_latent.py")

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--alpha", type=float, required=True)
known, remaining = pre.parse_known_args()
if not 0 <= known.alpha <= 1.2:
    raise ValueError("alpha outside diagnostic contract")
sys.argv = [sys.argv[0], "--mode", "endpoints", *remaining]
spec = importlib.util.spec_from_file_location("stage2m_eval_core", CORE)
core = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(core)


def actor_from_state(state):
    actor = nn.Sequential(
        nn.Linear(123, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 37),
    )
    actor.load_state_dict(OrderedDict(
        (key.removeprefix("mlp."), value) for key, value in state.items() if key.startswith("mlp.")
    ), strict=True)
    return actor


class MultiplierPolicy(nn.Module):
    last_mean = None

    def __init__(self):
        super().__init__()
        student = torch.load(K / "student/selected_gait_latent_student.pt", map_location="cpu", weights_only=False)
        state = student["model_state_dict"]
        self.first_base_weight = nn.Parameter(state["first_base_weight"].clone(), requires_grad=False)
        self.first_gait_column = nn.Parameter(state["first_gait_column"].clone(), requires_grad=False)
        self.first_bias = nn.Parameter(state["first_bias"].clone(), requires_grad=False)
        self.hidden = nn.Sequential(
            nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 37),
        )
        self.hidden.load_state_dict(OrderedDict(
            (key.removeprefix("hidden."), value) for key, value in state.items() if key.startswith("hidden.")
        ), strict=True)
        walk_state = torch.load(WALK, map_location="cpu", weights_only=False)["actor_state_dict"]
        run_state = torch.load(RUN, map_location="cpu", weights_only=False)["actor_state_dict"]
        self.walk_actor = actor_from_state(walk_state)
        self.run_actor = actor_from_state(run_state)
        self.register_buffer("walk_std", walk_state["distribution.std_param"].clone())
        self.register_buffer("run_std", run_state["distribution.std_param"].clone())

    def load_state_dict(self, state_dict, strict=True):
        return nn.modules.module._IncompatibleKeys([], [])

    def student_mean(self, observation, gait):
        first = nn.functional.linear(observation, self.first_base_weight, self.first_bias)
        return self.hidden(first + gait.reshape(-1, 1) * self.first_gait_column.T)

    def forward(self, observation, encoded):
        # Encodings: teacher WALK=-1, teacher RUN=2, student WALK=0, student RUN=1.
        student = (encoded >= 0) & (encoded <= 1)
        run = encoded > .5
        gait = run.float()
        student_mean = self.student_mean(observation, gait)
        teacher_mean = torch.where(run.reshape(-1, 1), self.run_actor(observation), self.walk_actor(observation))
        mean = torch.where(student.reshape(-1, 1), student_mean, teacher_mean)
        base_std = torch.where(run.reshape(-1, 1), self.run_std, self.walk_std)
        # The first and second 400 environments have identical endpoint/episode order.
        # Duplicate the same epsilon block to obtain paired teacher/student noise.
        epsilon = torch.randn((4, 100, 37), device=mean.device).reshape(400, 37)
        epsilon = torch.cat((epsilon, epsilon), dim=0)
        MultiplierPolicy.last_mean = mean.detach()
        return mean if known.alpha == 0 else mean + known.alpha * base_std * epsilon


def conditions(_mode):
    specs = []
    endpoints = (("walk_1p2", 1.2, -1.0), ("run_1p2", 1.2, 2.0),
                 ("run_2p4", 2.4, 2.0), ("run_2p6", 2.6, 2.0))
    for name, speed, gait in endpoints:
        specs.append({"name": f"teacher_{name}", "speed": speed, "gait": gait, "episodes": 100, "duration": 10.0})
    for name, speed, gait in endpoints:
        student_gait = 0.0 if name == "walk_1p2" else 1.0
        specs.append({"name": f"student_{name}", "speed": speed, "gait": student_gait, "episodes": 100, "duration": 10.0})
    return specs


class Telemetry:
    step = 0
    initialized = False
    fall_time = None
    run_basin_time = None
    divergence_rows = []


original_step = core.RslRlVecEnvWrapper.step
original_close = core.RslRlVecEnvWrapper.close


def instrumented_step(wrapper, action):
    result = original_step(wrapper, action)
    observation, reward, dones, extras = result
    env = wrapper.unwrapped
    count = 800
    if not Telemetry.initialized:
        Telemetry.initialized = True
        Telemetry.fall_time = torch.full((count,), float("nan"), device=action.device)
        Telemetry.run_basin_time = torch.full((count,), float("nan"), device=action.device)
        Telemetry.flight_events = torch.zeros(count, dtype=torch.long, device=action.device)
        Telemetry.safe = torch.zeros_like(Telemetry.flight_events)
        Telemetry.alt = torch.zeros_like(Telemetry.flight_events)
        Telemetry.streak = torch.zeros_like(Telemetry.flight_events)
        Telemetry.last_landing = torch.full_like(Telemetry.flight_events, -1)
        Telemetry.reward_sum = torch.zeros(count, device=action.device)
        Telemetry.yaw_rate_sum = torch.zeros(count, device=action.device)
        sensor = env.scene.sensors["contact_forces"]
        Telemetry.sensor = sensor
        Telemetry.feet = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
    time = Telemetry.step * float(env.step_dt)
    Telemetry.reward_sum += reward
    Telemetry.yaw_rate_sum += env.scene["robot"].data.root_ang_vel_b[:, 2]
    timeout = extras.get("time_outs", torch.zeros_like(dones)).bool()
    fell = dones.bool() & ~timeout
    new_fall = fell & torch.isnan(Telemetry.fall_time)
    Telemetry.fall_time[new_fall] = time
    forces = Telemetry.sensor.data.net_forces_w_history[:, -1, Telemetry.feet, :].norm(dim=-1)
    contacts = forces > 5
    flight = contacts.sum(-1) == 0
    previous = Telemetry.streak.clone()
    Telemetry.flight_events += (flight & (previous == 0)).long()
    Telemetry.streak = torch.where(flight, previous + 1, torch.zeros_like(previous))
    landing = ~flight & (previous > 0)
    single = landing & (contacts.sum(-1) == 1)
    foot = contacts.long().argmax(-1)
    safe = single & (previous >= 2) & (previous <= 8)
    alt = safe & (Telemetry.last_landing >= 0) & (foot != Telemetry.last_landing)
    Telemetry.safe += safe.long()
    Telemetry.alt += alt.long()
    Telemetry.last_landing[single] = foot[single]
    periodic = (Telemetry.flight_events >= 4) & (Telemetry.safe >= 3) & (Telemetry.alt >= 3)
    new_periodic = periodic & torch.isnan(Telemetry.run_basin_time)
    Telemetry.run_basin_time[new_periodic] = time
    if Telemetry.step % 10 == 0:
        robot = env.scene["robot"]
        teacher_ids = torch.arange(0, 100, device=action.device)
        student_ids = torch.arange(400, 500, device=action.device)
        root_t = torch.cat((robot.data.root_pos_w[teacher_ids], robot.data.root_quat_w[teacher_ids],
                            robot.data.root_lin_vel_w[teacher_ids], robot.data.root_ang_vel_w[teacher_ids]), dim=-1)
        root_s = torch.cat((robot.data.root_pos_w[student_ids], robot.data.root_quat_w[student_ids],
                            robot.data.root_lin_vel_w[student_ids], robot.data.root_ang_vel_w[student_ids]), dim=-1)
        joint_t = torch.cat((robot.data.joint_pos[teacher_ids], robot.data.joint_vel[teacher_ids]), dim=-1)
        joint_s = torch.cat((robot.data.joint_pos[student_ids], robot.data.joint_vel[student_ids]), dim=-1)
        contact_difference = (contacts[teacher_ids] != contacts[student_ids]).float().mean(-1)
        mean_difference = (MultiplierPolicy.last_mean[teacher_ids] - MultiplierPolicy.last_mean[student_ids]).norm(dim=-1)
        Telemetry.divergence_rows.append({
            "alpha": known.alpha, "time_s": time,
            "root_state_l2_mean": float((root_t - root_s).norm(dim=-1).mean()),
            "joint_state_l2_mean": float((joint_t - joint_s).norm(dim=-1).mean()),
            "contact_pattern_difference_mean": float(contact_difference.mean()),
            "action_mean_l2_mean": float(mean_difference.mean()),
        })
    Telemetry.step += 1
    return result


def instrumented_close(wrapper):
    raw = OUT / "raw" / f"alpha_{known.alpha:.4f}"
    rows_path = raw / "endpoints_evaluation.csv"
    rows = list(csv.DictReader(rows_path.open(encoding="utf-8")))
    for index, row in enumerate(rows):
        row["alpha"] = known.alpha
        row["episode_return"] = float(Telemetry.reward_sum[index])
        row["actual_yaw_rate_mean"] = float(Telemetry.yaw_rate_sum[index] / max(Telemetry.step, 1))
        row["time_to_fall_s"] = (
            float(Telemetry.fall_time[index]) if torch.isfinite(Telemetry.fall_time[index]) else ""
        )
        is_walk = "walk_1p2" in row["condition"]
        if is_walk:
            value = Telemetry.run_basin_time[index]
            row["time_to_gait_basin_failure_s"] = float(value) if torch.isfinite(value) else ""
        elif row["gait_classification"] != "PERIODIC_RUNNING":
            value = Telemetry.fall_time[index]
            row["time_to_gait_basin_failure_s"] = float(value) if torch.isfinite(value) else 10.0
        else:
            row["time_to_gait_basin_failure_s"] = ""
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (raw / "walk_divergence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Telemetry.divergence_rows[0]))
        writer.writeheader()
        writer.writerows(Telemetry.divergence_rows)
    (raw / "telemetry.json").write_text(json.dumps({
        "alpha": known.alpha, "paired_noise": True, "seed": 20268121,
        "walk_teacher_student_pairs": 100,
        "walk_teacher_run_basin_entries": int(torch.isfinite(Telemetry.run_basin_time[:100]).sum()),
        "walk_student_run_basin_entries": int(torch.isfinite(Telemetry.run_basin_time[400:500]).sum()),
    }, indent=2) + "\n", encoding="utf-8")
    return original_close(wrapper)


def main():
    torch.manual_seed(20268121)
    raw = OUT / "raw" / f"alpha_{known.alpha:.4f}"
    raw.mkdir(parents=True, exist_ok=True)
    core.OUT = OUT
    core.RAW = raw
    core.STUDENT = L / "student/stage2l_gait_conditioned_std_student.pt"
    core.Student = MultiplierPolicy
    core.conditions = conditions
    core.RslRlVecEnvWrapper.step = instrumented_step
    core.RslRlVecEnvWrapper.close = instrumented_close
    core.main()


if __name__ == "__main__":
    main()
