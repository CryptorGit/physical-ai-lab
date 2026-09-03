"""Evaluate deterministic closure seeds and optionally record the selected Stage 2Q sequence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import gymnasium as gym
import torch
from torch import nn
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/closure"
CHECKPOINT = (
    REPO
    / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/"
    "stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
)
EXPECTED_SHA = "66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698"
SEED_FIRST = 20270021
SEED_COUNT = 20
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from g1_single_policy.phase_gated_heading import wrapped_heading_error, yaw_from_quat_wxyz  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--record", action="store_true")
parser.add_argument("--selected-seed", type=int, default=SEED_FIRST)
parser.add_argument("--raw-video", default=str(OUT / "raw/exp_012_closure_sequence_raw.mp4"))
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]
if args.record:
    args.enable_cameras = True


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def minimum_jerk(value):
    return 10 * value**3 - 15 * value**4 + 6 * value**5


def ramp(time_s, start, duration, left, right, device):
    value = torch.tensor(max(0.0, min(1.0, (time_s - start) / duration)), device=device)
    return left + (right - left) * minimum_jerk(value)


def schedule(time_s: float, device):
    if time_s < 3:
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), 0, "STAND"
    if time_s < 4:
        return ramp(time_s, 3, 1, 0, .6, device), torch.tensor(0.0, device=device), 1, "WALK"
    if time_s < 5:
        return ramp(time_s, 4, 1, .6, 1.2, device), torch.tensor(0.0, device=device), 1, "WALK"
    if time_s < 7:
        return torch.tensor(1.2, device=device), torch.tensor(0.0, device=device), 1, "WALK"
    if time_s < 9:
        return torch.tensor(1.2, device=device), ramp(time_s, 7, 2, 0, 1, device), 1, "WALK -> RUN"
    if time_s < 10.5:
        return ramp(time_s, 9, 1.5, 1.2, 2.4, device), torch.tensor(1.0, device=device), 2, "RUN"
    if time_s < 14.5:
        return torch.tensor(2.4, device=device), torch.tensor(1.0, device=device), 2, "RUN"
    if time_s < 16:
        return ramp(time_s, 14.5, 1.5, 2.4, 1.2, device), torch.tensor(1.0, device=device), 2, "RUN"
    if time_s < 17:
        return torch.tensor(1.2, device=device), torch.tensor(1.0, device=device), 2, "RUN"
    if time_s < 19:
        return torch.tensor(1.2, device=device), 1 - ramp(time_s, 17, 2, 0, 1, device), 2, "RUN -> WALK"
    if time_s < 21:
        return torch.tensor(1.2, device=device), torch.tensor(0.0, device=device), 3, "WALK"
    if time_s < 22:
        return ramp(time_s, 21, 1, 1.2, .6, device), torch.tensor(0.0, device=device), 3, "WALK"
    if time_s < 23:
        return ramp(time_s, 22, 1, .6, 0, device), torch.tensor(0.0, device=device), 3, "STOP"
    return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), 4, "STOP"


class Policy(nn.Module):
    def __init__(self, path: Path):
        super().__init__()
        state = torch.load(path, map_location="cpu", weights_only=False)["actor_state_dict"]
        self.first_base_weight = nn.Parameter(state["first_base_weight"], requires_grad=False)
        self.first_gait_column = nn.Parameter(state["first_gait_column"], requires_grad=False)
        self.first_bias = nn.Parameter(state["first_bias"], requires_grad=False)
        self.hidden = nn.Sequential(
            nn.ELU(), nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 37),
        )
        self.hidden.load_state_dict(OrderedDict(
            (key.removeprefix("hidden."), value)
            for key, value in state.items() if key.startswith("hidden.")
        ))
        self.register_buffer("log_std_walk", state["distribution.log_std_walk"])
        self.register_buffer("log_std_run", state["distribution.log_std_run"])

    def forward(self, observation, gait):
        first = nn.functional.linear(observation, self.first_base_weight, self.first_bias)
        return self.hidden(first + gait[:, None] * self.first_gait_column.T)


def overlay_frame(frame, lines, label):
    import cv2

    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 18), (835, 88), (9, 14, 24), -1)
    cv2.rectangle(overlay, (1210, 18), (1895, 120), (9, 14, 24), -1)
    cv2.rectangle(overlay, (20, 785), (720, 1055), (9, 14, 24), -1)
    cv2.addWeighted(overlay, .76, frame, .24, 0, frame)
    cv2.putText(frame, "EXP_012 - ONE POLICY, TWO GAITS", (42, 62),
                cv2.FONT_HERSHEY_SIMPLEX, .88, (245, 248, 252), 2, cv2.LINE_AA)
    for index, line in enumerate(("ONE CHECKPOINT", "NO ROUTER", "NO ACTION BLENDING")):
        cv2.putText(frame, line, (1235, 52 + 28 * index),
                    cv2.FONT_HERSHEY_SIMPLEX, .68, (245, 248, 252), 2, cv2.LINE_AA)
    cv2.putText(frame, label, (42, 750), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                (70, 205, 255), 3, cv2.LINE_AA)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (42, 820 + 29 * index),
                    cv2.FONT_HERSHEY_SIMPLEX, .68, (245, 248, 252), 2, cv2.LINE_AA)
    return frame


def main():
    checkpoint = CHECKPOINT.resolve(strict=True)
    if sha256(checkpoint) != EXPECTED_SHA:
        raise RuntimeError("Stage 2Q selected checkpoint SHA mismatch")
    if not SEED_FIRST <= args.selected_seed < SEED_FIRST + SEED_COUNT:
        raise ValueError("selected seed is outside the preregistered set")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(exist_ok=True)
    env_count = 1 if args.record else SEED_COUNT
    selected_index = 0 if args.record else args.selected_seed - SEED_FIRST
    cfg, agent_cfg = resolve_task_config("Isaac-Exp012-G1-Reverse-PhaseR1-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = env_count
    cfg.scene.env_spacing = 40.0
    cfg.episode_length_s = 28.0
    cfg.seed = args.selected_seed if args.record else SEED_FIRST
    cfg.viewer.origin_type = "world"
    cfg.video_recorder.window_width = args.width
    cfg.video_recorder.window_height = args.height
    cfg.commands.base_velocity.debug_vis = False
    agent_cfg.seed = cfg.seed
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    with launch_simulation(cfg, args):
        raw = gym.make(
            "Isaac-Exp012-G1-Reverse-PhaseR1-v0", cfg=cfg,
            render_mode="rgb_array" if args.record else None,
        )
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        import importlib.metadata
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        device = runner.device
        policy = Policy(checkpoint).to(device).eval()
        env = wrapped.unwrapped
        robot = env.scene["robot"]
        command = env.command_manager.get_term("base_velocity")
        command.external_override_enabled = True
        sensor = env.scene.sensors["contact_forces"]
        feet = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        robot_feet = [next(i for i, name in enumerate(robot.body_names) if name == sensor.body_names[j]) for j in feet]
        obs, _ = wrapped.reset()
        obs = obs.to(device)
        reference_yaw = yaw_from_quat_wxyz(robot.data.root_quat_w).clone()
        steps = round(28.0 / float(env.step_dt))
        fallen = torch.zeros(env_count, dtype=torch.bool, device=device)
        dangerous_slip = torch.zeros_like(fallen)
        impact = torch.zeros_like(fallen)
        saturation = torch.zeros_like(fallen)
        slip_streak = torch.zeros(env_count, dtype=torch.long, device=device)
        saturation_streak = torch.zeros_like(slip_streak)
        flight_streak = torch.zeros_like(slip_streak)
        last_landing = torch.full((env_count,), -1, dtype=torch.long, device=device)
        segment_steps = torch.zeros((5, env_count), device=device)
        segment_flight = torch.zeros_like(segment_steps)
        segment_events = torch.zeros((5, SEED_COUNT), dtype=torch.long, device=device)
        segment_safe = torch.zeros_like(segment_events)
        segment_alt = torch.zeros_like(segment_events)
        heading_trace = []
        writer = None
        if args.record:
            import cv2
            raw_path = Path(args.raw_video).resolve()
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), 50.0, (args.width, args.height)
            )
            if not writer.isOpened():
                raise RuntimeError(f"unable to create {raw_path}")
        for step in range(steps):
            time_s = step * float(env.step_dt)
            speed, gait, segment, label = schedule(time_s, device)
            command.external_override[:, 0] = speed
            command.external_override[:, 1:] = 0
            if step == 0:
                obs = wrapped.get_observations().to(device)
            gait_batch = gait.repeat(env_count)
            with torch.inference_mode():
                action = policy(obs["policy"], gait_batch)
                obs, _, dones, extras = wrapped.step(action)
            obs = obs.to(device)
            timeout = extras.get("time_outs", torch.zeros_like(dones)).bool()
            fallen |= dones.bool() & ~timeout
            forces = sensor.data.net_forces_w_history[:, -1, feet, :].norm(dim=-1)
            contacts = forces > 5
            in_flight = contacts.sum(-1) == 0
            previous = flight_streak.clone()
            takeoff = in_flight & (flight_streak == 0)
            flight_streak = torch.where(in_flight, flight_streak + 1, torch.zeros_like(flight_streak))
            landing = ~in_flight & (previous > 0)
            single = landing & (contacts.sum(-1) == 1)
            foot = contacts.long().argmax(-1)
            safe = single & (previous >= 2) & (previous <= 8)
            alternating = safe & (last_landing >= 0) & (foot != last_landing)
            last_landing[single] = foot[single]
            segment_steps[segment] += 1
            segment_flight[segment] += in_flight.float()
            segment_events[segment] += takeoff.long()
            segment_safe[segment] += safe.long()
            segment_alt[segment] += alternating.long()
            heading_error = wrapped_heading_error(reference_yaw, yaw_from_quat_wxyz(robot.data.root_quat_w)).abs()
            heading_trace.append(heading_error.cpu())
            foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1)
            slipping = ((foot_speed > .55) & contacts).any(-1)
            slip_streak = torch.where(slipping, slip_streak + 1, torch.zeros_like(slip_streak))
            dangerous_slip |= slip_streak >= 5
            impact |= forces.amax(-1) > 3500
            limits = robot.data.joint_vel_limits
            if limits.ndim == 3:
                limits = limits[..., 1].abs()
            saturated_now = (robot.data.joint_vel.abs() / limits.clamp_min(1e-6) > .95).any(-1)
            saturation_streak = torch.where(saturated_now, saturation_streak + 1, torch.zeros_like(saturation_streak))
            saturation |= saturation_streak >= 5
            if writer is not None:
                import cv2
                root = robot.data.root_pos_w[selected_index].detach().cpu()
                eye = (float(root[0] - 3.8), float(root[1] - 4.0), 2.25)
                target = (float(root[0] + .55), float(root[1]), .9)
                env.sim.set_camera_view(eye=eye, target=target)
                recorder = getattr(env, "video_recorder", None)
                capture = getattr(recorder, "_capture", None)
                capture_cfg = getattr(capture, "cfg", None)
                if capture_cfg is not None:
                    from isaacsim.core.rendering_manager import ViewportManager
                    ViewportManager.set_camera_view(
                        capture_cfg.camera_prim_path,
                        eye=list(eye),
                        target=list(target),
                    )
                frame = cv2.cvtColor(raw.render(), cv2.COLOR_RGB2BGR)
                contact = contacts[selected_index]
                detected = "RUN" if int(segment) == 2 else "WALK" if int(segment) in (1, 3) else "NEAR-STAND"
                lines = [
                    f"TARGET SPEED   {float(speed):.2f} m/s",
                    f"ACTUAL SPEED   {float(robot.data.root_lin_vel_b[selected_index, 0]):+.2f} m/s",
                    f"GAIT COMMAND   {float(gait):.2f}",
                    f"DETECTED GAIT  {detected}",
                    f"FLIGHT         {bool(in_flight[selected_index])}",
                    f"LEFT / RIGHT   {bool(contact[0])} / {bool(contact[1])}",
                    f"HEADING        {float(heading_error[selected_index]):.3f} rad",
                    f"SHA            {EXPECTED_SHA[:16]}...",
                ]
                writer.write(overlay_frame(frame, lines, label))
        if writer is not None:
            writer.release()
        heading = torch.stack(heading_trace)
        rows = []
        for index in range(env_count):
            metrics = []
            for segment in range(5):
                count = max(float(segment_steps[segment, index]), 1)
                metrics.append({
                    "flight_fraction": float(segment_flight[segment, index] / count),
                    "periodic": (
                        int(segment_events[segment, index]) >= 4
                        and int(segment_safe[segment, index]) >= 3
                        and int(segment_alt[segment, index]) >= 3
                    ),
                })
            moving_success = metrics[1]["flight_fraction"] < .10 and metrics[2]["periodic"] and metrics[3]["flight_fraction"] < .10
            rows.append({
                "seed": args.selected_seed if args.record else SEED_FIRST + index,
                "vector_substream_index": None if args.record else index,
                "fall": bool(fallen[index]),
                "walk_run_segments_success": bool(moving_success),
                "walk_segment_success": metrics[1]["flight_fraction"] < .10,
                "run_segment_success": metrics[2]["periodic"],
                "return_walk_segment_success": metrics[3]["flight_fraction"] < .10,
                "heading_p95_rad": float(torch.quantile(heading[:, index], .95)),
                "dangerous_slip": bool(dangerous_slip[index]),
                "impact_failure": bool(impact[index]),
                "long_dwell_saturation": bool(saturation[index]),
                "final_speed_mps": float(robot.data.root_lin_vel_b[index, 0].abs()),
                "strict_initial_stand": metrics[0]["flight_fraction"] == 0,
                "strict_final_stand": metrics[4]["flight_fraction"] == 0,
            })
        ranking = sorted(
            rows,
            key=lambda row: (
                row["fall"],
                not row["walk_run_segments_success"],
                row["heading_p95_rad"],
                row["dangerous_slip"],
                row["final_speed_mps"],
                row["seed"],
            ),
        )
        for rank, row in enumerate(ranking, 1):
            row["rank"] = rank
        by_seed = {row["seed"]: row for row in ranking}
        selection = {
            "seed_contract": {
                "first": SEED_FIRST,
                "last": SEED_FIRST + SEED_COUNT - 1,
                "execution": "one deterministic vectorized Isaac Lab process",
                "mapping": "seed label = base seed + vector substream index",
            },
            "preregistered_ranking": [
                "fall=0", "all WALK/RUN segments success", "minimum heading error",
                "no dangerous slip", "minimum final speed", "lowest seed",
            ],
            "selected_seed": ranking[0]["seed"],
            "selected": ranking[0],
            "all_fall_free": any(not row["fall"] for row in rows),
            "checkpoint_sha256": EXPECTED_SHA,
            "deterministic_mean": True,
        }
        if args.record:
            (OUT / "video_recording_run.json").write_text(json.dumps({
                "recording_seed": args.selected_seed,
                "num_envs": 1,
                "single_humanoid_scene": True,
                "metrics": rows[0],
                "checkpoint_sha256": EXPECTED_SHA,
                "deterministic_mean": True,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            csv_path = OUT / "video_seed_selection.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer_csv = csv.DictWriter(handle, fieldnames=list(ranking[0]))
                writer_csv.writeheader()
                writer_csv.writerows(sorted(ranking, key=lambda row: row["seed"]))
            (OUT / "video_seed_selection.json").write_text(
                json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        wrapped.close()
        print(json.dumps(selection, sort_keys=True))


if __name__ == "__main__":
    main()
