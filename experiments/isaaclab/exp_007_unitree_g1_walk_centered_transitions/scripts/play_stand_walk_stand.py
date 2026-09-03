"""GUI playback of the Stage 5 manifest-driven STAND-WALK-STAND round trip."""

from __future__ import annotations

import argparse
import json
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
import g1_walk_centered.tasks  # noqa: E402,F401
from g1_walk_centered.command_contract import MotionCommand  # noqa: E402
from g1_walk_centered.experts import load_walk_expert  # noqa: E402
from g1_walk_centered.experts.adapters import canonical_state_from_legacy_observation  # noqa: E402
from g1_walk_centered.planner import CommandPlanner, ExternalCommand, ExternalCommandKind  # noqa: E402
from g1_walk_centered.router import ExpertRouter  # noqa: E402
from g1_walk_centered.transition_graph import StateGraph  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--speed", type=float, choices=(0.6, 0.8, 1.0, 1.2), default=1.0)
parser.add_argument("--cycles", type=int, choices=(1, 3), default=1)
parser.add_argument("--seed", type=int, default=20260901)
parser.add_argument("--manifest", default=str(EXP / "integration_manifest.json"))
parser.add_argument("--validate-only", action="store_true")
parser.add_argument("--require-valid-stand-contract", action="store_true")
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def minimum_jerk(value):
    value = max(0.0, min(1.0, value))
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def main() -> None:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    graph = StateGraph.from_manifest(EXP / manifest["state_graph"])
    paths = {
        name: (REPO / spec["checkpoint"]).resolve(strict=True)
        for name, spec in manifest["controllers"].items()
        if name in {
            "stage2_model_4246", "stand_to_walk_transition_v1",
            "walk_steady_state_expert_v1", "walk_to_stand_transition_v1",
        }
    }
    print("STATE_GRAPH=modular_state_graph_v1 ACTION_BLEND=false ACTIVE_CONTROLLERS=1")
    print(f"SUPPORTED_WALK_COMMANDS={list((0.6, 0.8, 1.0, 1.2))} REQUEST={args.speed}")
    print("STARTUP_RECOVERY=NOT_A_FORMAL_CAPABILITY RESET_TO_STAND=NOT_IMPLEMENTED")
    if args.validate_only:
        for path in paths.values():
            load_walk_expert(path)
        print("preflight=PASS simulation_started=false")
        return
    cfg, agent = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.episode_length_s = 24.0 if args.cycles == 1 else 65.0
    cfg.viewer.origin_type = "world"
    cfg.viewer.eye = (6.0, -7.5, 3.8)
    cfg.viewer.lookat = (3.0, 0.0, 0.8)
    if args.device:
        cfg.sim.device = args.device
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(
            gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=cfg),
            clip_actions=agent.clip_actions,
        )
        env = wrapped.unwrapped
        models = {name: load_walk_expert(path, device=env.device) for name, path in paths.items()}
        router = ExpertRouter(graph, models, "STAND")
        planner = CommandPlanner(graph)
        robot = env.scene["robot"]
        velocity_term = env.command_manager.get_term("base_velocity")
        sensor = env.scene.sensors["contact_forces"]
        _, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        ankles, _ = robot.find_joints(".*_ankle_pitch_joint")
        wrapped.reset()
        heading = robot.data.heading_w.torch.clone()
        filtered_yaw = torch.zeros(1, device=env.device)
        previous_action = torch.zeros(1, 37, device=env.device)
        phase = "UNINITIALIZED" if args.require_valid_stand_contract else "STAND"
        elapsed, streak, switches, previous_support = 0.0, 0.0, 0, 0
        cycle, result, stop_origin = 1, "IN_PROGRESS", robot.data.root_pos_w.torch[:, :2].clone()
        no_switch = 0.0
        startup_ankle_dwell = 0.0
        stand_contract_valid = not args.require_valid_stand_contract
        graph_route_started = False
        dt = float(env.step_dt)
        max_steps = round((23.0 if args.cycles == 1 else 64.0) / dt)
        for step in range(max_steps):
            if phase == "STAND" and elapsed >= (1.5 if cycle == 1 else 1.0):
                plan = planner.plan_command("STAND", ExternalCommand(ExternalCommandKind.WALK, args.speed))
                router.accept_plan(plan)
                graph_route_started = True
                phase, elapsed, streak, switches = "STAND_TO_WALK", 0.0, 0.0, 0
                heading[:] = robot.data.heading_w.torch
                print(f"EXTERNAL COMMAND=WALK({args.speed}) PLANNED ROUTE=STAND_TO_WALK -> WALK")
            command_vx = (
                args.speed * minimum_jerk(elapsed / 1.5) if phase == "STAND_TO_WALK"
                else args.speed if phase == "WALK"
                else args.speed * (1.0 - minimum_jerk(elapsed / 1.6)) if phase == "WALK_TO_STAND"
                else 0.0
            )
            heading_error = torch.atan2(
                torch.sin(heading - robot.data.heading_w.torch),
                torch.cos(heading - robot.data.heading_w.torch),
            )
            raw = (0.8 * heading_error - 0.10 * robot.data.root_ang_vel_b.torch[:, 2]).clamp(-0.3, 0.3)
            low = filtered_yaw + 0.15 * (raw - filtered_yaw)
            filtered_yaw += (low - filtered_yaw).clamp(-0.01, 0.01)
            if phase in {"UNINITIALIZED", "STAND"}:
                filtered_yaw.zero_()
            velocity_term.vel_command_b.zero_()
            velocity_term.vel_command_b[:, 0] = command_vx
            velocity_term.vel_command_b[:, 2] = filtered_yaw
            legacy = wrapped.get_observations()["policy"]
            previous_consistent = bool((legacy[:, 86:123] == previous_action).all())
            state = canonical_state_from_legacy_observation(
                legacy, heading_w_rad=robot.data.heading_w.torch
            )
            command = MotionCommand(
                torch.tensor([command_vx], device=env.device), heading,
                target_yaw_rate_radps=filtered_yaw,
            )
            active = models["stage2_model_4246"] if phase == "UNINITIALIZED" else router.controller()
            with torch.inference_mode():
                action = active(state, command)
                _, _, done, _ = wrapped.step(action)
            jump = float(torch.linalg.vector_norm(action - previous_action))
            previous_action[:] = action
            contacts = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :].norm(dim=-1).amax(dim=1) > 5.0
            support = int(contacts[0, 0]) + 2 * int(contacts[0, 1])
            speed = float(robot.data.root_lin_vel_b.torch[0, 0])
            horizontal = float(robot.data.root_lin_vel_b.torch[0, :2].norm())
            vertical = abs(float(robot.data.root_lin_vel_w.torch[0, 2]))
            heading_abs = abs(float(heading_error[0]))
            g = robot.data.projected_gravity_b.torch[0]
            roll = float(torch.atan2(g[1], -g[2]))
            pitch = float(torch.atan2(-g[0], torch.sqrt(g[1] ** 2 + g[2] ** 2)))
            ankle = float((
                robot.data.applied_torque.torch[:, ankles].abs()
                / robot.data.joint_effort_limits.torch[:, ankles].abs().clamp_min(1e-6)
            ).amax())
            if phase == "UNINITIALIZED":
                startup_ankle_dwell = startup_ankle_dwell + dt if ankle >= 0.95 else 0.0
                safe = (
                    horizontal <= 0.08 and vertical <= 0.05
                    and abs(roll) <= 0.10 and abs(pitch) <= 0.10
                    and support == 3 and not bool(done[0])
                    and startup_ankle_dwell < 0.20
                    and torch.isfinite(action).all()
                )
                streak = streak + dt if safe else 0.0
                if streak >= 0.4:
                    phase, elapsed, streak = "STAND", 0.0, 0.0
                    stand_contract_valid = True
                    heading[:] = robot.data.heading_w.torch
                    print("SYSTEM STATE=STAND STAND CONTRACT=VALID GRAPH ROUTE STARTED=FALSE")
                elif elapsed >= 2.0:
                    result = "FAIL"
                    print("SYSTEM STATE=UNINITIALIZED STAND CONTRACT=INVALID GRAPH ROUTE STARTED=FALSE")
                    break
            elif phase == "STAND_TO_WALK":
                if support and support != previous_support:
                    switches += 1
                good = (
                    speed >= 0.75 * args.speed and abs(speed - args.speed) <= 0.20
                    and heading_abs <= 0.12 and switches >= 2
                )
                streak = streak + dt if good else 0.0
                if streak >= 0.4:
                    router.complete_transition(completion_condition_pass=True)
                    phase, elapsed, streak = "WALK", 0.0, 0.0
            elif phase == "WALK" and elapsed >= 3.0:
                plan = planner.plan_command("WALK", ExternalCommand(ExternalCommandKind.STOP))
                router.accept_plan(plan)
                phase, elapsed, streak, no_switch = "WALK_TO_STAND", 0.0, 0.0, 0.0
                stop_origin[:] = robot.data.root_pos_w.torch[:, :2]
                print("EXTERNAL COMMAND=STOP PLANNED ROUTE=WALK_TO_STAND -> STAND")
            elif phase == "WALK_TO_STAND":
                no_switch = 0.0 if support and support != previous_support else no_switch + dt
                good = (
                    horizontal <= 0.08 and vertical <= 0.05 and heading_abs <= 0.12
                    and abs(roll) <= 0.10 and abs(pitch) <= 0.10 and support == 3
                    and no_switch >= 0.4
                )
                streak = streak + dt if good else 0.0
                if streak >= 0.4:
                    router.complete_transition(completion_condition_pass=True)
                    phase, elapsed, streak = "STAND", 0.0, 0.0
                    if cycle == args.cycles:
                        result = "PASS"
                    else:
                        cycle += 1
            previous_support = support
            if step % max(1, round(0.5 / dt)) == 0:
                stopping = float(torch.linalg.vector_norm(robot.data.root_pos_w.torch[:, :2] - stop_origin))
                displayed_controller = (
                    "STARTUP_RECOVERY_DIAGNOSTIC_ONLY"
                    if phase == "UNINITIALIZED" else router.active().controller
                )
                print(
                    f"CYCLE={cycle}/{args.cycles} STATE={phase} CONTROLLER={displayed_controller} "
                    f"ROUTE_CURSOR={router.route_cursor} TARGET={command_vx:.3f} ACTUAL={speed:.3f} "
                    f"HEADING_ERROR={heading_abs:.4f} SUPPORT={support} COMPLETION_STREAK={streak:.2f} "
                    f"ACTION_JUMP={jump:.3f} ANKLE_EFFORT={ankle:.3f} STOPPING_DISTANCE={stopping:.3f} "
                    f"PREVIOUS_ACTION_OK={previous_consistent} RESULT={result} "
                    f"STAND_CONTRACT={'VALID' if stand_contract_valid else 'INVALID'} "
                    f"GRAPH_ROUTE_STARTED={graph_route_started} "
                    "STARTUP_RECOVERY=NOT_A_FORMAL_CAPABILITY"
                )
            elapsed += dt
            if result == "PASS" and elapsed >= 5.0:
                break
            if bool(done[0]) or elapsed > 6.0:
                result = "FAIL"
                break
        print(f"FINAL RESULT={result}")
        wrapped.close()


if __name__ == "__main__":
    main()
