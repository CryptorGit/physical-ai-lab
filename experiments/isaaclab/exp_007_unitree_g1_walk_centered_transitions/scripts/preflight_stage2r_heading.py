"""Bounded pre-training comparison of smoothed heading-hold candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import torch

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
sys.path[:0] = [str(EXP / "src"), str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src")]
import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
from g1_walk_centered.command_contract import MotionCommand  # noqa: E402
from g1_walk_centered.experts import load_walk_expert  # noqa: E402
from g1_walk_centered.experts.adapters import canonical_state_from_legacy_observation  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

EXPECTED_SHA = "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"
CANDIDATES = (
    (0.4, 0.05, 0.3, 0.15, 0.01),
    (0.4, 0.10, 0.3, 0.15, 0.01),
    (0.6, 0.10, 0.3, 0.15, 0.01),
    (0.6, 0.10, 0.5, 0.25, 0.02),
    (0.8, 0.10, 0.3, 0.15, 0.01),
    (0.8, 0.10, 0.5, 0.25, 0.02),
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--seed", type=int, default=20260725)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def _sha(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(round((len(values) - 1) * q / 100), len(values) - 1)]


def main() -> None:
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    if _sha(checkpoint) != EXPECTED_SHA:
        raise RuntimeError("Parent checkpoint hash mismatch")
    output = Path(args.output)
    if not output.is_absolute():
        output = REPO / output
    output.mkdir(parents=True, exist_ok=True)
    speeds = tuple(speed for _ in CANDIDATES for speed in (0.6, 0.8))
    configs = tuple(candidate for candidate in CANDIDATES for _ in (0.6, 0.8))
    n = len(speeds)
    cfg, agent = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = n
    cfg.seed = args.seed
    cfg.episode_length_s = 10.0
    if args.device:
        cfg.sim.device = args.device
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=cfg), clip_actions=agent.clip_actions)
        env = wrapped.unwrapped
        expert = load_walk_expert(checkpoint, device=env.device)
        robot = env.scene["robot"]
        command_term = env.command_manager.get_term("base_velocity")
        feet, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor = env.scene.sensors["contact_forces"]
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        ankles, _ = robot.find_joints(".*_ankle_pitch_joint")
        wrapped.reset()
        target_heading = robot.data.heading_w.torch.clone()
        filtered = torch.zeros(n, device=env.device)
        previous = torch.zeros(n, 37, device=env.device)
        traces = [{"heading": [], "yaw_cmd": [], "action_rate": [], "ankle": [], "slip": [], "vx": []} for _ in range(n)]
        dt = float(env.step_dt)
        for step in range(round(8.0 / dt)):
            elapsed = step * dt
            u = min(max((elapsed - 1.0) / 2.0, 0.0), 1.0)
            blend = 10 * u**3 - 15 * u**4 + 6 * u**5
            vx_command = torch.tensor(speeds, device=env.device) * blend
            heading_error = torch.atan2(
                torch.sin(target_heading - robot.data.heading_w.torch),
                torch.cos(target_heading - robot.data.heading_w.torch),
            )
            raw = torch.empty(n, device=env.device)
            for i, (kp, kd, limit, alpha, slew) in enumerate(configs):
                desired = max(-limit, min(limit, kp * float(heading_error[i]) - kd * float(robot.data.root_ang_vel_b.torch[i, 2])))
                low_pass = float(filtered[i]) + alpha * (desired - float(filtered[i]))
                filtered[i] += max(-slew, min(slew, low_pass - float(filtered[i])))
                raw[i] = filtered[i]
            command_term.vel_command_b.zero_()
            command_term.vel_command_b[:, 0] = vx_command
            command_term.vel_command_b[:, 2] = raw
            obs = wrapped.get_observations()["policy"]
            state = canonical_state_from_legacy_observation(obs, heading_w_rad=robot.data.heading_w.torch)
            with torch.inference_mode():
                action = expert(state, MotionCommand(vx_command, target_heading, target_yaw_rate_radps=raw))
                wrapped.step(action)
            contact = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :].norm(dim=-1).amax(dim=1) > 5.0
            slip = robot.data.body_lin_vel_w.torch[:, feet, :2].norm(dim=-1)
            effort = robot.data.applied_torque.torch[:, ankles].abs()
            limit = robot.data.joint_effort_limits.torch[:, ankles].abs().clamp_min(1.0e-6)
            for i, trace in enumerate(traces):
                if elapsed >= 3.0:
                    trace["heading"].append(abs(float(heading_error[i])))
                    trace["yaw_cmd"].append(float(raw[i]))
                    trace["action_rate"].append(float(torch.linalg.vector_norm(action[i] - previous[i]) / dt))
                    trace["ankle"].append(float((effort[i] / limit[i]).max()))
                    trace["slip"].append(max([float(slip[i, j]) for j in range(2) if contact[i, j]] or [0.0]))
                    trace["vx"].append(float(robot.data.root_lin_vel_b.torch[i, 0]))
            previous = action.clone()
        rows = []
        for i, trace in enumerate(traces):
            reversals = sum(
                a * b < 0 and abs(a - b) > 0.01
                for a, b in zip(trace["yaw_cmd"], trace["yaw_cmd"][1:])
            )
            duration = max(len(trace["yaw_cmd"]) * dt, dt)
            kp, kd, limit, alpha, slew = configs[i]
            row = {
                "candidate": CANDIDATES.index(configs[i]),
                "speed_mps": speeds[i],
                "k_heading": kp,
                "k_yaw_rate": kd,
                "yaw_rate_limit_radps": limit,
                "low_pass_alpha": alpha,
                "slew_limit_radps_per_step": slew,
                "heading_p95_rad": _pct(trace["heading"], 95),
                "yaw_reversal_hz": reversals / duration,
                "yaw_command_p95_radps": _pct([abs(x) for x in trace["yaw_cmd"]], 95),
                "action_rate_p95": _pct(trace["action_rate"], 95),
                "ankle_effort_p95": _pct(trace["ankle"], 95),
                "ankle_saturation_fraction": sum(x >= 0.95 for x in trace["ankle"]) / len(trace["ankle"]),
                "foot_slip_mean_mps": sum(trace["slip"]) / len(trace["slip"]),
                "actual_speed_mean_mps": sum(trace["vx"]) / len(trace["vx"]),
            }
            rows.append(row)
        aggregates = []
        for candidate in range(len(CANDIDATES)):
            group = [r for r in rows if r["candidate"] == candidate]
            aggregates.append(
                {
                    "candidate": candidate,
                    "config": dict(zip(("k_heading", "k_yaw_rate", "yaw_rate_limit_radps", "low_pass_alpha", "slew_limit_radps_per_step"), CANDIDATES[candidate])),
                    "heading_p95_rad": max(r["heading_p95_rad"] for r in group),
                    "yaw_reversal_hz": max(r["yaw_reversal_hz"] for r in group),
                    "action_rate_p95": max(r["action_rate_p95"] for r in group),
                    "ankle_saturation_fraction": max(r["ankle_saturation_fraction"] for r in group),
                    "foot_slip_mean_mps": max(r["foot_slip_mean_mps"] for r in group),
                }
            )
        selected = min(
            aggregates,
            key=lambda r: (
                r["yaw_reversal_hz"] > 2.0,
                r["heading_p95_rad"],
                r["ankle_saturation_fraction"],
                r["action_rate_p95"],
                r["foot_slip_mean_mps"],
            ),
        )
        result = {
            "purpose": "pre-training bounded controller selection; not a skill evaluation",
            "seed": args.seed,
            "speeds_mps": [0.6, 0.8],
            "episodes_per_candidate_speed": 1,
            "candidates": aggregates,
            "per_case": rows,
            "selection_rule": "reject >2Hz yaw reversal, then heading, ankle saturation, action rate, slip",
            "selected": selected,
        }
        (output / "heading_preflight.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(selected, indent=2))
        wrapped.close()


if __name__ == "__main__":
    main()
