"""EXP 012 parent-policy GUI, including the frozen yaw diagnostic mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
PARENT = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
SELECTED_MANIFEST = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2_pilot1_retry1/selected_checkpoint.json"
PHASE_A_MANIFEST = REPO / (
    "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/"
    "stage2e_phase_a_run_acquisition_preflight/selected_phase_a_checkpoint.json"
)
SELECTED = (
    REPO / json.loads(SELECTED_MANIFEST.read_text(encoding="utf-8"))["checkpoint"]
    if SELECTED_MANIFEST.exists() else PARENT
)
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402
from g1_single_policy.yaw_bias_canceller import G1SpeedConditionedYawBiasCancellerV1  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=("Stand", "Walk", "Run", "Transition", "IntegratedSequence", "RunAcquisition", "YawDiagnosis", "YawCancellation"),
    default="IntegratedSequence" if SELECTED_MANIFEST.exists() else "YawDiagnosis",
)
parser.add_argument("--checkpoint", type=Path, default=SELECTED)
parser.add_argument("--speed", type=float, default=0.6)
parser.add_argument("--yaw-rate", type=float, default=0.0)
parser.add_argument("--controller", choices=("Off", "On"), default="Off")
parser.add_argument("--seed", type=int, default=20261101)
parser.add_argument("--show-floor-guides", action=argparse.BooleanOptionalAction, default=True)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def wrap(value):
    return torch.atan2(torch.sin(value), torch.cos(value))


def main() -> None:
    if args.mode == "RunAcquisition" and not PHASE_A_MANIFEST.exists():
        raise SystemExit("RunAcquisition requires Stage 2E selected_phase_a_checkpoint.json.")
    if args.mode not in ("YawDiagnosis", "YawCancellation") and not SELECTED_MANIFEST.exists():
        raise SystemExit("Integrated locomotion modes require retry1/selected_checkpoint.json.")
    cfg, agent_cfg = resolve_task_config(
        "Isaac-Exp012-G1-SinglePolicy-Bidirectional-v0", "rsl_rl_cfg_entry_point"
    )
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 20.0
    cfg.seed = args.seed
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    cfg.viewer.origin_type = "world"
    cfg.viewer.eye = (-3.8, -4.0, 2.3)
    cfg.viewer.lookat = (0.5, 0.0, 0.9)
    agent_cfg.seed = args.seed
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device

    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp012-G1-SinglePolicy-Bidirectional-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        import importlib.metadata
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(args.checkpoint.resolve()), strict=True, map_location=runner.device)
        policy = runner.get_inference_policy(device=runner.device)
        env, robot = raw.unwrapped, raw.unwrapped.scene["robot"]
        command = env.command_manager.get_term("base_velocity")
        command.external_override_enabled = True
        sensor = env.scene.sensors["contact_forces"]
        run_reward = env.reward_manager.get_term_cfg("safe_periodic_flight").func
        feet = [index for index, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        obs, _ = wrapped.reset()
        obs = obs.to(runner.device)
        reference = robot.data.heading_w[0].clone()
        canceller = G1SpeedConditionedYawBiasCancellerV1(dt=0.02)
        checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        episode_step = 0
        completion_count = 0

        guides = None
        if args.show_floor_guides:
            try:
                import omni.isaac.debug_draw._debug_draw as debug_draw
                guides = debug_draw.acquire_debug_draw_interface()
                starts, ends, colors, widths = [], [], [], []
                for x in range(-5, 71):
                    starts.append((float(x), -1.5, 0.012))
                    ends.append((float(x), 1.5, 0.012))
                    colors.append((0.25, 0.72, 0.90, 1.0) if x % 5 == 0 else (0.55, 0.60, 0.64, 1.0))
                    widths.append(5.0 if x % 5 == 0 else 2.0)
                guides.draw_lines(starts, ends, colors, widths)
            except Exception as error:
                print(f"[YawDiagnosis] floor-guide console fallback: {error}")

        previous_action = torch.zeros((1, 37), device=runner.device)
        while env.sim.is_playing():
            t = episode_step * float(env.step_dt)
            requested_speed = args.speed
            if args.mode == "Stand":
                requested_speed = 0.0
            elif args.mode == "Walk":
                requested_speed = args.speed if args.speed <= 1.2 else 1.2
            elif args.mode == "Run":
                requested_speed = args.speed if args.speed >= 2.4 else 2.6
            elif args.mode == "Transition":
                tau = max(0.0, min(1.0, (t - 2.0) / 1.5))
                requested_speed = 1.2 + (2.6 - 1.2) * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
            elif args.mode == "IntegratedSequence":
                cycle = t % 32.0
                points = (
                    (0., 0.), (2., 0.), (3., .6), (5., .6), (6., 1.2), (8., 1.2),
                    (9.5, 2.4), (12.5, 2.4), (14., 2.6), (17., 2.6),
                    (18.5, 2.4), (21.5, 2.4), (23., 1.2), (25., 1.2),
                    (26., .6), (28., .6), (29., 0.), (32., 0.),
                )
                requested_speed = 0.0
                for (ta, va), (tb, vb) in zip(points, points[1:]):
                    if ta <= cycle < tb:
                        tau = (cycle - ta) / (tb - ta)
                        blend = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
                        requested_speed = va if va == vb else va + (vb - va) * blend
                        break
            elif args.mode == "RunAcquisition":
                cycle = t % 20.0
                if cycle < 1.0:
                    requested_speed = 0.0
                elif cycle < 2.0:
                    tau = cycle - 1.0
                    requested_speed = 1.2 * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
                elif cycle < 3.5:
                    requested_speed = 1.2
                elif cycle < 5.0:
                    tau = (cycle - 3.5) / 1.5
                    blend = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
                    requested_speed = 1.2 + (2.5 - 1.2) * blend
                else:
                    requested_speed = 2.5
            if args.mode == "YawCancellation":
                control = canceller.step(requested_speed, desired_yaw_rate=0.0)
            else:
                control = {
                    "desired_yaw_rate": args.yaw_rate,
                    "offset": 0.0,
                    "policy_yaw_rate": args.yaw_rate,
                }
            if args.controller == "Off":
                control["offset"] = 0.0
                control["policy_yaw_rate"] = control["desired_yaw_rate"]
            command.external_override[0] = torch.tensor(
                (requested_speed, 0.0, control["policy_yaw_rate"]), device=runner.device
            )
            with torch.inference_mode():
                action = policy(obs)
            obs, _, done, _ = wrapped.step(action)
            obs = obs.to(runner.device)
            heading_error = wrap(robot.data.heading_w[0] - reference)
            forces = sensor.data.net_forces_w_history[0, -1, feet].norm(dim=-1)
            contacts = forces > 5.0
            completion_fire = float(run_reward.last_raw_reward[0]) >= 1.0
            precursor_fire = 0.0 < float(run_reward.last_raw_reward[0]) < 1.0
            completion_count += int(completion_fire)
            phase = (
                "DOUBLE" if bool(contacts.all()) else "FLIGHT" if not bool(contacts.any())
                else "LEFT" if bool(contacts[0]) else "RIGHT"
            )
            action_asymmetry = float(
                torch.linalg.vector_norm(action[0, ::2] - previous_action[0, ::2])
            )
            saturation = float((action[0].abs() >= 0.99).float().mean())
            if env.common_step_counter % 25 == 0:
                common = [
                        "EXP_012 G1 SINGLE-POLICY PILOT RETRY",
                        f"TARGET SPEED {requested_speed:.2f}",
                        f"ACTUAL SPEED {float(robot.data.root_lin_vel_b[0, 0]):+.3f}",
                ]
                if args.mode not in ("YawDiagnosis", "YawCancellation"):
                    in_flight = not bool(contacts.any())
                    gait = "STAND" if requested_speed == 0 and abs(float(robot.data.root_lin_vel_b[0, 0])) < .08 else (
                        "RUN_CANDIDATE" if requested_speed >= 2.3 and in_flight else "WALK_LIKE"
                    )
                    common.extend((
                        f"GAIT STATE {gait}",
                        f"YAW BIAS {float(robot.data.root_ang_vel_b[0, 2]):+.3f}",
                        f"HEADING ERROR {float(heading_error):+.3f}",
                        f"FLIGHT {in_flight}",
                        f"CONTACTS {phase}",
                        "SLIP diagnostic",
                        f"IMPACT {float(forces.max()):.1f}N",
                        f"SATURATION {saturation:.3f}",
                        f"FALL {bool(done[0])}",
                        f"CHECKPOINT SHA {checkpoint_sha[:12]}",
                        "UNIQUE CHECKPOINT COUNT 1",
                        "CURRENT PPO LR frozen inference",
                    ))
                    if args.mode == "RunAcquisition":
                        common.extend((
                            f"FLIGHT DURATION {float(run_reward._flight_duration[0]):.3f}s",
                            f"PRECURSOR FIRE {precursor_fire}",
                            f"COMPLETION FIRE {completion_fire}",
                            f"LAST LANDING SIDE {int(run_reward._last_landing_foot[0])}",
                            f"ALTERNATION COUNT {completion_count}",
                        ))
                elif args.mode == "YawCancellation":
                    common.extend((
                        f"DESIRED YAW RATE {control['desired_yaw_rate']:+.3f}",
                        f"POLICY YAW COMMAND {control['policy_yaw_rate']:+.3f}",
                        f"CANCELLATION OFFSET {control['offset']:+.3f}",
                        f"ACTUAL YAW RATE {float(robot.data.root_ang_vel_w[0, 2]):+.3f}",
                        f"HEADING ERROR {float(heading_error):+.3f}",
                        f"SATURATION {saturation:.3f}",
                        f"CONTACTS {phase}",
                        f"FALL {bool(done[0])}",
                    ))
                else:
                    common.extend((
                        f"YAW-RATE COMMAND {args.yaw_rate:+.3f}",
                        f"ACTUAL YAW RATE {float(robot.data.root_ang_vel_w[0, 2]):+.3f}",
                        f"SIGNED HEADING ERROR {float(heading_error):+.3f}",
                        f"RESPONSE GAIN {'n/a' if abs(args.yaw_rate) < 1e-9 else f'{float(robot.data.root_ang_vel_w[0, 2] / args.yaw_rate):+.2f}'}",
                        f"CONTACT PHASE {phase}",
                        f"LEFT/RIGHT ACTION ASYMMETRY {action_asymmetry:.3f}",
                        f"FALL {bool(done[0])}",
                    ))
                print(" | ".join(common))
            root = robot.data.root_pos_w[0].detach().cpu()
            env.sim.set_camera_view(
                eye=(float(root[0] - 3.8), float(root[1] - 4.0), 2.3),
                target=(float(root[0] + 0.5), float(root[1]), 0.9),
            )
            previous_action.copy_(action)
            episode_step += 1
            if bool(done[0]):
                canceller.reset()
                reference = robot.data.heading_w[0].clone()
                episode_step = 0
                completion_count = 0
        if guides is not None:
            guides.clear_lines()
        wrapped.close()


if __name__ == "__main__":
    main()
