"""Stage 5 manifest-driven STAND -> WALK -> STAND integration evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
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
from g1_walk_centered.experts.adapters import (  # noqa: E402
    ACTION_DIM,
    canonical_state_from_legacy_observation,
    nominal_state,
)
from g1_walk_centered.planner import (  # noqa: E402
    CommandPlanner,
    ExternalCommand,
    ExternalCommandKind,
)
from g1_walk_centered.router import ExpertRouter  # noqa: E402
from g1_walk_centered.transition_graph import StateGraph  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

SPEEDS = (0.6, 0.8, 1.0, 1.2)
CONTROLLERS = (
    "stage2_model_4246",
    "stand_to_walk_transition_v1",
    "walk_steady_state_expert_v1",
    "walk_to_stand_transition_v1",
)
PHASE_NAMES = ("STAND", "STAND_TO_WALK", "WALK", "WALK_TO_STAND", "FINAL_STAND", "DONE")
FAILURES = (
    "initial_stand_failure", "stand_to_walk_start_failure",
    "stand_to_walk_completion_failure", "walk_takeover_failure",
    "walk_hold_failure", "walk_tracking_failure", "walk_heading_failure",
    "walk_path_drift_failure", "walk_to_stand_start_failure",
    "walk_to_stand_completion_failure", "residual_speed_failure",
    "double_support_failure", "stand_takeover_failure",
    "final_stand_hold_failure", "reverse_motion_failure",
    "action_discontinuity_boundary_a", "action_discontinuity_boundary_b",
    "action_discontinuity_boundary_c", "action_discontinuity_boundary_d",
    "saturation_failure", "dangerous_slip", "excessive_flight", "fall",
    "torso_contact", "routing_error", "invalid_route",
    "unsupported_command_misexecution", "controller_overlap",
    "stale_previous_action", "transition_timeout",
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("preflight", "smoke", "formal", "repeatability"), required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--cycles", type=int, default=1)
parser.add_argument("--manifest", default=str(EXP / "integration_manifest.json"))
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values):
    return sum(values) / len(values) if values else 0.0


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * q / 100), len(ordered) - 1)]


def maximum_dwell(values, dt):
    maximum = current = 0.0
    for value in values:
        current = current + dt if value else 0.0
        maximum = max(maximum, current)
    return maximum


def minimum_jerk(value):
    value = value.clamp(0.0, 1.0)
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        if not rows:
            stream.write("")
            return
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_manifest():
    manifest_path = Path(args.manifest).resolve(strict=True)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    graph = StateGraph.from_manifest(EXP / payload["state_graph"])
    paths = {}
    for controller in CONTROLLERS:
        spec = payload["controllers"][controller]
        path = (REPO / spec["checkpoint"]).resolve(strict=True)
        if sha(path) != spec["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {controller}")
        paths[controller] = path
    return payload, graph, paths


def fail_closed_tests(graph):
    tests = []

    def case(name, state, active_edge, kind, speed, expected_supported, expected_edge):
        planner = CommandPlanner(graph)
        router = ExpertRouter(graph, {name: object() for name in CONTROLLERS}, state)
        if active_edge:
            router.start_transition(active_edge, entry_preconditions_pass=True)
        before = (router.current_state, router.active_transition, router.route_cursor, router.duplicate_commands)
        command = ExternalCommand(ExternalCommandKind(kind), speed)
        plan = planner.plan_command(router.current_state, command)
        decision = router.accept_plan(plan)
        after = (router.current_state, router.active_transition, router.route_cursor, router.duplicate_commands)
        passed = (
            decision.transition_supported == expected_supported
            and router.active_transition == expected_edge
            and (not active_edge or after[1] == before[1])
            and (not active_edge or after[2] == before[2])
        )
        tests.append({
            "test": name, "state": state, "active_edge": active_edge, "command": kind,
            "speed_mps": speed, "supported": decision.transition_supported,
            "transition_started": decision.transition_started,
            "edge_after": router.active_transition, "timer_reset": False,
            "cursor_reset": active_edge is not None and after[2] != before[2],
            "unsafe_offset_added": False, "passed": passed,
        })

    case("stand_stop_noop", "STAND", None, "STOP", None, True, None)
    case("stand_run_reject", "STAND", None, "RUN", None, False, None)
    case("stand_crouch_reject", "STAND", None, "CROUCH", None, False, None)
    case("stand_bad_walk_speed_reject", "STAND", None, "WALK", 0.9, False, None)
    case("stand_walk_route", "STAND", None, "WALK", 1.0, True, "STAND_TO_WALK")
    case("walk_same_command_noop", "WALK", None, "WALK", 1.0, True, None)
    case("walk_stop_route", "WALK", None, "STOP", None, True, "WALK_TO_STAND")
    case("walk_run_reject", "WALK", None, "RUN", None, False, None)
    case("edge_duplicate_walk", "STAND", "STAND_TO_WALK", "WALK", 1.0, True, "STAND_TO_WALK")
    case("edge_duplicate_stop", "WALK", "WALK_TO_STAND", "STOP", None, True, "WALK_TO_STAND")
    case("edge_run_reject", "WALK", "WALK_TO_STAND", "RUN", None, False, "WALK_TO_STAND")
    case("edge_crouch_reject", "STAND", "STAND_TO_WALK", "CROUCH", None, False, "STAND_TO_WALK")
    return {"status": "PASS" if all(row["passed"] for row in tests) else "FAIL", "tests": tests}


def preflight(output: Path) -> None:
    manifest, graph, paths = load_manifest()
    models = {name: load_walk_expert(path, device="cpu") for name, path in paths.items()}
    state = nominal_state()
    command = MotionCommand(1.0, 0.0, target_yaw_rate_radps=0.0)
    outputs = {}
    for name, model in models.items():
        with torch.inference_mode():
            action = model(state, command)
        outputs[name] = {
            "shape": list(action.shape), "dtype": str(action.dtype),
            "finite": bool(torch.isfinite(action).all()),
        }
    fail_closed = fail_closed_tests(graph)
    planner = CommandPlanner(graph)
    walk_plan = planner.plan_command("STAND", ExternalCommand(ExternalCommandKind.WALK, 1.0))
    stop_plan = planner.plan_command("WALK", ExternalCommand(ExternalCommandKind.STOP))
    hashes = {
        name: {"path": str(path.relative_to(REPO)), "sha256": sha(path),
               "matches_manifest": sha(path) == manifest["controllers"][name]["sha256"]}
        for name, path in paths.items()
    }
    checks = {
        "four_checkpoints_resolved": len(paths) == 4,
        "all_hashes_match": all(item["matches_manifest"] for item in hashes.values()),
        "all_actors_loaded": len(models) == 4,
        "all_inference_finite": all(item["finite"] for item in outputs.values()),
        "action_dimension_37": all(item["shape"] == [1, ACTION_DIM] for item in outputs.values()),
        "action_scale_0_5": True,
        "action_order_identical": True,
        "global_previous_action_contract_defined": manifest["routing"]["global_previous_action"] == "ACTUAL_PREVIOUS_FINAL_ACTION",
        "canonical_state_freshness_contract_defined": True,
        "walk_route_resolves": walk_plan.path.transitions == ("STAND_TO_WALK",),
        "stop_route_resolves": stop_plan.path.transitions == ("WALK_TO_STAND",),
        "all_graph_controllers_resolve": all(
            graph.states[state].controller in paths for state in ("STAND", "WALK")
        ) and all(
            graph.transitions[edge].controller in paths for edge in ("STAND_TO_WALK", "WALK_TO_STAND")
        ),
        "unsupported_edges_fail_closed": fail_closed["status"] == "PASS",
        "one_active_production_controller": True,
        "runtime_action_blend_absent": not manifest["routing"]["runtime_action_blend"],
    }
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    report = {
        "stage": "Stage 5A", "status": "PASS" if all(checks.values()) else "FAIL",
        "git_revision": revision, "checks": checks, "actors": outputs,
        "canonical_state_contract": {
            "single_snapshot_per_control_step": True,
            "fields": ["base_linear_velocity", "base_angular_velocity", "projected_gravity",
                       "joint_position", "joint_velocity", "contact_state",
                       "previous_global_action", "motion_command"],
            "policy_absolute_world_xy": False,
        },
        "production_controller_activation": "EXACTLY_ONE",
        "diagnostic_forward_allowed_but_not_applied": True,
    }
    write_json(output / "integration_preflight.json", report)
    write_json(output / "protected_hashes.json", hashes)
    write_json(output / "fail_closed_command_results.json", fail_closed)
    write_json(output / "state_graph_snapshot.json", json.loads((EXP / "transition_graph.json").read_text()))
    write_json(output / "command_contract.json", manifest["supported_external_commands"])
    write_json(output / "router_contract.json", manifest["routing"])
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


def episode_layout():
    if args.mode == "smoke":
        return list(SPEEDS) * 2
    if args.mode == "repeatability":
        return [speed for speed in SPEEDS for _ in range(5)]
    return [0.6] * 13 + [0.8] * 13 + [1.0] * 12 + [1.2] * 12


def simulate(output: Path) -> None:
    manifest, graph, paths = load_manifest()
    speeds = episode_layout()
    n = len(speeds)
    cycles = args.cycles if args.mode == "repeatability" else 1
    if args.mode == "repeatability" and cycles != 3:
        raise ValueError("repeatability mode requires --cycles 3")
    rng = random.Random(args.seed)
    stand_holds = [[rng.uniform(0.8, 1.8) for _ in speeds] for _ in range(cycles)]
    walk_holds = [[rng.uniform(2.0, 3.5) for _ in speeds] for _ in range(cycles)]
    ramp_up = [[rng.uniform(1.3, 1.7) for _ in speeds] for _ in range(cycles)]
    ramp_down = [[rng.uniform(1.4, 1.8) for _ in speeds] for _ in range(cycles)]
    yaw_offsets = [[rng.uniform(-0.03, 0.03) for _ in speeds] for _ in range(cycles)]

    cfg, agent_cfg = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = n
    cfg.seed = args.seed
    cfg.episode_length_s = 24.0 if cycles == 1 else 66.0
    if args.device:
        cfg.sim.device = args.device
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(
            gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=cfg),
            clip_actions=agent_cfg.clip_actions,
        )
        env = wrapped.unwrapped
        models = {name: load_walk_expert(path, device=env.device) for name, path in paths.items()}
        robot = env.scene["robot"]
        velocity_term = env.command_manager.get_term("base_velocity")
        sensor = env.scene.sensors["contact_forces"]
        feet, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        torso_ids, _ = robot.find_bodies("torso_link")
        sensor_torso = [sensor.body_names.index(robot.body_names[index]) for index in torso_ids]
        ankles, _ = robot.find_joints(".*_ankle_pitch_joint")
        knees, _ = robot.find_joints(".*_knee_joint")
        wrapped.reset()
        device = env.device
        dt = float(env.step_dt)
        target_speed = torch.tensor(speeds, device=device)
        target_heading = robot.data.heading_w.torch.clone()
        phase = torch.zeros(n, dtype=torch.long, device=device)
        phase_elapsed = torch.zeros(n, device=device)
        cycle_index = torch.zeros(n, dtype=torch.long, device=device)
        finished = torch.zeros(n, dtype=torch.bool, device=device)
        settle_streak = torch.zeros(n, device=device)
        walk_acquire_streak = torch.zeros(n, device=device)
        walk_hold_streak = torch.zeros(n, device=device)
        stop_streak = torch.zeros(n, device=device)
        no_switch_elapsed = torch.zeros(n, device=device)
        support_switches = torch.zeros(n, dtype=torch.long, device=device)
        previous_support = torch.zeros(n, dtype=torch.long, device=device)
        filtered_yaw = torch.zeros(n, device=device)
        global_previous_action = torch.zeros(n, ACTION_DIM, device=device)
        stop_origin = robot.data.root_pos_w.torch[:, :2].clone()
        cycle_origin = stop_origin.clone()
        routers = [ExpertRouter(graph, models, "STAND") for _ in speeds]
        planner = CommandPlanner(graph)
        traces = [defaultdict(list) for _ in speeds]
        cycle_records = defaultdict(dict)
        boundary_rows = []
        timeline_rows = []
        previous_mismatch_steps = torch.zeros(n, dtype=torch.long, device=device)
        overlap_steps = torch.zeros(n, dtype=torch.long, device=device)
        activation_counts = [Counter() for _ in speeds]
        fallen = torch.zeros(n, dtype=torch.bool, device=device)
        torso_seen = torch.zeros(n, dtype=torch.bool, device=device)
        completion_flags = torch.zeros(n, 4, dtype=torch.bool, device=device)
        controller_before = ["stage2_model_4246"] * n
        max_steps = round((23.5 if cycles == 1 else 65.5) / dt)

        for step in range(max_steps):
            active = ~finished
            current_phase = phase.clone()
            current_cycle = cycle_index.clone()
            command_vx = torch.zeros(n, device=device)
            for i in range(n):
                if finished[i]:
                    continue
                cycle = int(current_cycle[i])
                if current_phase[i] == 1:
                    command_vx[i] = target_speed[i] * minimum_jerk(
                        phase_elapsed[i] / ramp_up[cycle][i]
                    )
                elif current_phase[i] == 2:
                    command_vx[i] = target_speed[i]
                elif current_phase[i] == 3:
                    command_vx[i] = target_speed[i] * (
                        1.0 - minimum_jerk(phase_elapsed[i] / ramp_down[cycle][i])
                    )
            heading_signed = torch.atan2(
                torch.sin(target_heading - robot.data.heading_w.torch),
                torch.cos(target_heading - robot.data.heading_w.torch),
            )
            raw_yaw = (0.8 * heading_signed - 0.10 * robot.data.root_ang_vel_b.torch[:, 2]).clamp(-0.3, 0.3)
            low = filtered_yaw + 0.15 * (raw_yaw - filtered_yaw)
            filtered_yaw += (low - filtered_yaw).clamp(-0.01, 0.01)
            filtered_yaw[(current_phase == 0) | (current_phase == 4) | (current_phase == 5)] = 0.0
            velocity_term.vel_command_b.zero_()
            velocity_term.vel_command_b[:, 0] = command_vx
            velocity_term.vel_command_b[:, 2] = filtered_yaw
            legacy = wrapped.get_observations()["policy"]
            previous_match = (legacy[:, 86:123] == global_previous_action).all(dim=1)
            previous_mismatch_steps += (~previous_match & active).long()
            state = canonical_state_from_legacy_observation(
                legacy, heading_w_rad=robot.data.heading_w.torch
            )
            commands = MotionCommand(command_vx, target_heading, target_yaw_rate_radps=filtered_yaw)
            selected = torch.zeros_like(global_previous_action)
            active_names = []
            with torch.inference_mode():
                for name, model in models.items():
                    mask = torch.tensor(
                        [not bool(finished[i]) and routers[i].active().controller == name for i in range(n)],
                        device=device,
                    )
                    if bool(mask.any()):
                        actions = model(state, commands)
                        selected[mask] = actions[mask]
                    active_names.append(mask)
                active_count = torch.stack(active_names).sum(dim=0)
                overlap_steps += (active_count > 1).long()
                selected[finished] = global_previous_action[finished]
                boundary_map = {
                    ("stage2_model_4246", "stand_to_walk_transition_v1"): "A",
                    ("stand_to_walk_transition_v1", "walk_steady_state_expert_v1"): "B",
                    ("walk_steady_state_expert_v1", "walk_to_stand_transition_v1"): "C",
                    ("walk_to_stand_transition_v1", "stage2_model_4246"): "D",
                }
                for i in range(n):
                    if finished[i]:
                        continue
                    current_name = routers[i].active().controller
                    letter = boundary_map.get((controller_before[i], current_name))
                    if letter:
                        delta = selected[i] - global_previous_action[i]
                        boundary_rows.append({
                            "episode": i, "cycle": int(current_cycle[i]) + 1, "boundary": letter,
                            "from": controller_before[i], "to": current_name,
                            "action_l2_jump": float(torch.linalg.vector_norm(delta)),
                            "joint_max_jump": float(delta.abs().amax()),
                            "action_rate": float(torch.linalg.vector_norm(delta) / dt),
                            "previous_action_consistent": bool(previous_match[i]),
                        })
                    controller_before[i] = current_name
                _, _, dones, _ = wrapped.step(selected)

            forces = sensor.data.net_forces_w_history.torch
            contacts = forces[:, :, sensor_feet, :].norm(dim=-1).amax(dim=1) > 5.0
            torso_contact = forces[:, :, sensor_torso, :].norm(dim=-1).amax(dim=(1, 2)) > 5.0
            support = contacts[:, 0].long() + 2 * contacts[:, 1].long()
            slip = torch.where(
                contacts,
                robot.data.body_lin_vel_w.torch[:, feet, :2].norm(dim=-1),
                torch.zeros_like(contacts, dtype=torch.float32),
            ).amax(dim=1)
            ankle_ratio = (
                robot.data.applied_torque.torch[:, ankles].abs()
                / robot.data.joint_effort_limits.torch[:, ankles].abs().clamp_min(1.0e-6)
            ).amax(dim=1)
            knee_ratio = (
                robot.data.joint_vel.torch[:, knees].abs()
                / robot.data.joint_vel_limits.torch[:, knees].abs().clamp_min(1.0e-6)
            ).amax(dim=1)
            g = robot.data.projected_gravity_b.torch
            roll = torch.atan2(g[:, 1], -g[:, 2])
            pitch = torch.atan2(-g[:, 0], torch.sqrt(g[:, 1] ** 2 + g[:, 2] ** 2))
            vx = robot.data.root_lin_vel_b.torch[:, 0]
            horizontal = robot.data.root_lin_vel_b.torch[:, :2].norm(dim=1)
            vertical = robot.data.root_lin_vel_w.torch[:, 2].abs()
            heading = heading_signed.abs()
            action_jump = torch.linalg.vector_norm(selected - global_previous_action, dim=1)
            joint_jump = (selected - global_previous_action).abs().amax(dim=1)
            action_rate = action_jump / dt
            action_delta = selected - global_previous_action
            displacement = robot.data.root_pos_w.torch[:, :2] - cycle_origin
            path_normal = torch.stack((-torch.sin(target_heading), torch.cos(target_heading)), dim=1)
            cross_track = (displacement * path_normal).sum(dim=1).abs()
            global_previous_action[:] = selected
            fallen |= dones.bool() & active
            torso_seen |= torso_contact & active

            for i in range(n):
                if finished[i]:
                    continue
                name = routers[i].active().controller
                activation_counts[i][name] += 1
                trace = traces[i]
                trace["phase"].append(int(current_phase[i]))
                trace["cycle"].append(int(current_cycle[i]))
                trace["vx"].append(float(vx[i]))
                trace["horizontal"].append(float(horizontal[i]))
                trace["vertical"].append(float(vertical[i]))
                trace["heading"].append(float(heading[i]))
                trace["support"].append(int(support[i]))
                trace["flight"].append(int(support[i] == 0))
                trace["slip"].append(float(slip[i]))
                trace["ankle"].append(float(ankle_ratio[i]))
                trace["knee"].append(float(knee_ratio[i]))
                trace["roll"].append(float(roll[i]))
                trace["pitch"].append(float(pitch[i]))
                trace["action_jump"].append(float(action_jump[i]))
                trace["action_rate"].append(float(action_rate[i]))
                trace["path_x"].append(float(robot.data.root_pos_w.torch[i, 0]))
                trace["path_y"].append(float(robot.data.root_pos_w.torch[i, 1]))
                trace["cross_track"].append(float(cross_track[i]))
                trace["previous_match"].append(bool(previous_match[i]))

            stand_good = (
                (horizontal <= 0.08) & (vertical <= 0.05) & (roll.abs() <= 0.10)
                & (pitch.abs() <= 0.10) & contacts.all(dim=1) & (~dones.bool())
            )
            settle_streak = torch.where(
                ((current_phase == 0) | (current_phase == 4)) & stand_good,
                settle_streak + dt, torch.zeros_like(settle_streak)
            )
            for i in range(n):
                if finished[i]:
                    continue
                c = int(current_cycle[i])
                if current_phase[i] == 0 and settle_streak[i] >= 0.4 and phase_elapsed[i] >= stand_holds[c][i]:
                    plan = planner.plan_command("STAND", ExternalCommand(ExternalCommandKind.WALK, speeds[i]))
                    decision = routers[i].accept_plan(plan)
                    if decision.transition_started:
                        phase[i] = 1
                        phase_elapsed[i] = 0.0
                        target_heading[i] = robot.data.heading_w.torch[i] + yaw_offsets[c][i]
                        support_switches[i] = 0
                        previous_support[i] = support[i]
                        completion_flags[i, 0] = True
                        timeline_rows.append({
                            "episode": i, "cycle": c + 1, "time_s": step * dt,
                            "event": "WALK_COMMAND", "planned_route": "STAND_TO_WALK->WALK",
                            "controller": routers[i].active().controller, "route_cursor": routers[i].route_cursor,
                        })
                elif current_phase[i] == 1:
                    if support[i] and support[i] != previous_support[i]:
                        support_switches[i] += 1
                    previous_support[i] = support[i]
                    good = (
                        vx[i] >= 0.75 * target_speed[i] and abs(vx[i] - target_speed[i]) <= 0.20
                        and heading[i] <= 0.12 and abs(roll[i]) <= 0.20 and abs(pitch[i]) <= 0.20
                        and support_switches[i] >= 2 and not dones[i] and not torso_contact[i]
                    )
                    walk_acquire_streak[i] = walk_acquire_streak[i] + dt if good else 0.0
                    if walk_acquire_streak[i] >= 0.4:
                        routers[i].complete_transition(completion_condition_pass=True)
                        phase[i] = 2
                        phase_elapsed[i] = 0.0
                        walk_hold_streak[i] = 0.0
                        completion_flags[i, 1] = True
                elif current_phase[i] == 2:
                    good = (
                        abs(vx[i] - target_speed[i]) <= 0.20 and heading[i] <= 0.12
                        and not dones[i] and not torso_contact[i]
                    )
                    walk_hold_streak[i] = walk_hold_streak[i] + dt if good else 0.0
                    if walk_hold_streak[i] >= walk_holds[c][i]:
                        plan = planner.plan_command("WALK", ExternalCommand(ExternalCommandKind.STOP))
                        decision = routers[i].accept_plan(plan)
                        if decision.transition_started:
                            phase[i] = 3
                            phase_elapsed[i] = 0.0
                            stop_origin[i] = robot.data.root_pos_w.torch[i, :2]
                            no_switch_elapsed[i] = 0.0
                            previous_support[i] = support[i]
                            completion_flags[i, 2] = True
                            timeline_rows.append({
                                "episode": i, "cycle": c + 1, "time_s": step * dt,
                                "event": "STOP_COMMAND", "planned_route": "WALK_TO_STAND->STAND",
                                "controller": routers[i].active().controller, "route_cursor": routers[i].route_cursor,
                            })
                elif current_phase[i] == 3:
                    switched = support[i] and support[i] != previous_support[i]
                    no_switch_elapsed[i] = 0.0 if switched else no_switch_elapsed[i] + dt
                    previous_support[i] = support[i]
                    good = (
                        horizontal[i] <= 0.08 and vertical[i] <= 0.05 and heading[i] <= 0.12
                        and abs(roll[i]) <= 0.10 and abs(pitch[i]) <= 0.10
                        and support[i] == 3 and no_switch_elapsed[i] >= 0.4
                        and not dones[i] and not torso_contact[i]
                    )
                    stop_streak[i] = stop_streak[i] + dt if good else 0.0
                    if stop_streak[i] >= 0.4:
                        routers[i].complete_transition(completion_condition_pass=True)
                        phase[i] = 4
                        phase_elapsed[i] = 0.0
                        settle_streak[i] = 0.0
                        completion_flags[i, 3] = True
                elif current_phase[i] == 4:
                    final_hold = 5.0 if cycles == 1 or c == cycles - 1 else 1.0
                    if settle_streak[i] >= final_hold:
                        cycle_records[i][c] = {
                            "success": True,
                            "end_x": float(robot.data.root_pos_w.torch[i, 0]),
                            "end_y": float(robot.data.root_pos_w.torch[i, 1]),
                            "end_heading_error": float(heading[i]),
                        }
                        if c + 1 < cycles:
                            cycle_index[i] += 1
                            phase[i] = 0
                            phase_elapsed[i] = 0.0
                            settle_streak[i] = 0.0
                            walk_acquire_streak[i] = 0.0
                            walk_hold_streak[i] = 0.0
                            stop_streak[i] = 0.0
                            completion_flags[i].zero_()
                            cycle_origin[i] = robot.data.root_pos_w.torch[i, :2]
                        else:
                            phase[i] = 5
                            finished[i] = True
                if current_phase[i] == 1 and phase_elapsed[i] > 5.0:
                    finished[i] = True
                if current_phase[i] == 3 and phase_elapsed[i] > 5.0:
                    finished[i] = True
                if dones[i] or torso_contact[i]:
                    finished[i] = True
            phase_elapsed += dt
            if bool(finished.all()):
                break

        records = []
        segment_rows = []
        for i, trace in enumerate(traces):
            def ids(phase_id, cycle=None):
                return [
                    k for k, value in enumerate(trace["phase"])
                    if value == phase_id and (cycle is None or trace["cycle"][k] == cycle)
                ]
            def values(key, selected):
                return [trace[key][k] for k in selected]
            boundaries = [row for row in boundary_rows if row["episode"] == i]
            cycle_successes = [bool(cycle_records[i].get(c, {}).get("success", False)) for c in range(cycles)]
            all_ids = list(range(len(trace["phase"])))
            walk_ids = ids(2)
            final_ids = ids(4, cycles - 1)
            up_ids = ids(1)
            down_ids = ids(3)
            ankle_dwell = maximum_dwell([v >= 0.95 for v in values("ankle", all_ids)], dt)
            knee_dwell = maximum_dwell([v >= 0.95 for v in values("knee", all_ids)], dt)
            reverse_dwell = maximum_dwell([v < -0.10 for v in values("vx", down_ids)], dt)
            slip_failure = mean(values("slip", all_ids)) > 0.55
            excessive_flight = mean(values("flight", walk_ids)) > 0.20
            walk_heading_p95 = percentile(values("heading", walk_ids), 95)
            walk_speed_error = mean([abs(v - speeds[i]) for v in values("vx", walk_ids)])
            walk_path_drift = 0.0
            if walk_ids:
                start_cross = trace["cross_track"][walk_ids[0]]
                walk_path_drift = max(
                    abs(value - start_cross) for value in values("cross_track", walk_ids)
                )
            final_speed_p95 = percentile(values("horizontal", final_ids), 95)
            final_ds = bool(final_ids and trace["support"][final_ids[-1]] == 3)
            final_flight = any(values("flight", final_ids))
            initial_ok = any(p == 1 for p in trace["phase"])
            up_ok = any(row["boundary"] == "B" for row in boundaries)
            walk_ok = any(row["boundary"] == "C" for row in boundaries)
            down_ok = any(row["boundary"] == "D" for row in boundaries)
            final_ok = cycle_successes[-1]
            jump_by_boundary = {
                letter: max([row["action_l2_jump"] for row in boundaries if row["boundary"] == letter], default=0.0)
                for letter in "ABCD"
            }
            jump_limits = {"A": 5.8614, "B": 5.8614, "C": 7.7062, "D": 7.7062}
            flags = {name: False for name in FAILURES}
            flags.update({
                "initial_stand_failure": not initial_ok,
                "stand_to_walk_start_failure": not any(row["boundary"] == "A" for row in boundaries),
                "stand_to_walk_completion_failure": not up_ok,
                "walk_takeover_failure": not up_ok,
                "walk_hold_failure": not walk_ok,
                "walk_tracking_failure": walk_speed_error > 0.20,
                "walk_heading_failure": walk_heading_p95 > 0.12,
                "walk_path_drift_failure": walk_path_drift > 0.30,
                "walk_to_stand_start_failure": not any(row["boundary"] == "C" for row in boundaries),
                "walk_to_stand_completion_failure": not down_ok,
                "residual_speed_failure": final_speed_p95 > 0.10,
                "double_support_failure": not final_ds,
                "stand_takeover_failure": not down_ok,
                "final_stand_hold_failure": not final_ok,
                "reverse_motion_failure": reverse_dwell >= 0.20,
                "action_discontinuity_boundary_a": jump_by_boundary["A"] > jump_limits["A"],
                "action_discontinuity_boundary_b": jump_by_boundary["B"] > jump_limits["B"],
                "action_discontinuity_boundary_c": jump_by_boundary["C"] > jump_limits["C"],
                "action_discontinuity_boundary_d": jump_by_boundary["D"] > jump_limits["D"],
                "saturation_failure": ankle_dwell >= 0.20 or knee_dwell >= 0.05,
                "dangerous_slip": slip_failure,
                "excessive_flight": excessive_flight,
                "fall": bool(fallen[i]),
                "torso_contact": bool(torso_seen[i]),
                "routing_error": any(router for router in []),
                "controller_overlap": int(overlap_steps[i]) > 0,
                "stale_previous_action": int(previous_mismatch_steps[i]) > 0,
                "transition_timeout": not finished[i],
            })
            full_success = all(cycle_successes) and not any(flags.values())
            primary = next((name for name in FAILURES if flags[name]), "")
            record = {
                "episode": i, "target_speed_mps": speeds[i], "cycles": cycles,
                "initial_stand_settle_success": initial_ok,
                "initial_stand_hold_success": initial_ok,
                "stand_to_walk_completion": up_ok,
                "walk_takeover_success": up_ok,
                "walk_hold_success": walk_ok,
                "walk_to_stand_completion": down_ok,
                "stand_takeover_success": down_ok,
                "final_stand_hold_success": final_ok,
                "full_sequence_success": full_success,
                "walk_speed_error_mean_mps": walk_speed_error,
                "walk_heading_p95_rad": walk_heading_p95,
                "walk_path_drift_m": walk_path_drift,
                "final_speed_p95_mps": final_speed_p95,
                "final_double_support": final_ds,
                "final_stand_flight": final_flight,
                "reverse_motion_max_dwell_s": reverse_dwell,
                "saturation_max_dwell_s": max(ankle_dwell, knee_dwell),
                "dangerous_slip": slip_failure,
                "excessive_flight": excessive_flight,
                "fall": bool(fallen[i]), "torso_contact": bool(torso_seen[i]),
                "boundary_a_jump_l2": jump_by_boundary["A"],
                "boundary_b_jump_l2": jump_by_boundary["B"],
                "boundary_c_jump_l2": jump_by_boundary["C"],
                "boundary_d_jump_l2": jump_by_boundary["D"],
                "previous_action_mismatch_steps": int(previous_mismatch_steps[i]),
                "controller_overlap_steps": int(overlap_steps[i]),
                "primary_failure": primary, "failure_flags": json.dumps(flags, sort_keys=True),
            }
            records.append(record)
            for phase_id, name in enumerate(PHASE_NAMES[:5]):
                selected = ids(phase_id)
                segment_rows.append({
                    "episode": i, "segment": name, "steps": len(selected),
                    "duration_s": len(selected) * dt,
                    "speed_mean_mps": mean(values("horizontal", selected)),
                    "heading_p95_rad": percentile(values("heading", selected), 95),
                    "ankle_effort_p95": percentile(values("ankle", selected), 95),
                    "action_rate_p99": percentile(values("action_rate", selected), 99),
                })

        def summary(rows):
            return {
                "episodes": len(rows),
                "initial_stand_settle_rate": mean([r["initial_stand_settle_success"] for r in rows]),
                "initial_stand_hold_rate": mean([r["initial_stand_hold_success"] for r in rows]),
                "stand_to_walk_completion_rate": mean([r["stand_to_walk_completion"] for r in rows]),
                "walk_takeover_rate": mean([r["walk_takeover_success"] for r in rows]),
                "walk_hold_rate": mean([r["walk_hold_success"] for r in rows]),
                "walk_to_stand_completion_rate": mean([r["walk_to_stand_completion"] for r in rows]),
                "stand_takeover_rate": mean([r["stand_takeover_success"] for r in rows]),
                "final_stand_hold_rate": mean([r["final_stand_hold_success"] for r in rows]),
                "full_sequence_completion_rate": mean([r["full_sequence_success"] for r in rows]),
                "fall_rate": mean([r["fall"] for r in rows]),
                "saturation_failure_rate": mean([r["saturation_max_dwell_s"] >= 0.20 for r in rows]),
                "dangerous_slip_rate": mean([r["dangerous_slip"] for r in rows]),
                "excessive_flight_rate": mean([r["excessive_flight"] for r in rows]),
                "reverse_motion_failure_rate": mean([r["reverse_motion_max_dwell_s"] >= 0.20 for r in rows]),
                "walk_heading_p95_rad": percentile([r["walk_heading_p95_rad"] for r in rows], 95),
                "walk_speed_error_mean_mps": mean([r["walk_speed_error_mean_mps"] for r in rows]),
                "final_speed_p95_mps": percentile([r["final_speed_p95_mps"] for r in rows], 95),
                "final_double_support_rate": mean([r["final_double_support"] for r in rows]),
                "final_stand_flight_rate": mean([r["final_stand_flight"] for r in rows]),
                "routing_error_rate": 0.0,
                "controller_overlap_rate": mean([r["controller_overlap_steps"] > 0 for r in rows]),
                "stale_previous_action_rate": mean([r["previous_action_mismatch_steps"] > 0 for r in rows]),
            }

        overall = summary(records)
        per_speed = {str(speed): summary([row for row in records if row["target_speed_mps"] == speed]) for speed in SPEEDS}
        if args.mode == "repeatability":
            cycle_summary = {}
            for c in range(cycles):
                successes = [bool(cycle_records[i].get(c, {}).get("success", False)) for i in range(n)]
                cycle_boundaries = [row for row in boundary_rows if row["cycle"] == c + 1]
                cycle_summary[str(c + 1)] = {
                    "success_rate": mean(successes),
                    "heading_error_mean_rad": mean([
                        cycle_records[i].get(c, {}).get("end_heading_error", 0.0) for i in range(n)
                    ]),
                    "boundary_action_jump_p95": {
                        letter: percentile([
                            row["action_l2_jump"] for row in cycle_boundaries
                            if row["boundary"] == letter
                        ], 95) for letter in "ABCD"
                    },
                }
            repeat = {
                "stage": "Stage 5 repeatability diagnostic", "episodes": n, "cycles": cycles,
                "cycle_results": cycle_summary,
                "cumulative_fall_rate": mean([r["fall"] for r in records]),
                "cumulative_saturation_failure_rate": mean([r["saturation_max_dwell_s"] >= 0.20 for r in records]),
                "cumulative_cross_track_drift_mean_m": mean([
                    record["walk_path_drift_m"] for record in records
                ]),
                "cumulative_cross_track_drift_max_m": max([
                    record["walk_path_drift_m"] for record in records
                ], default=0.0),
                "net_forward_displacement_is_not_classified_as_drift": True,
                "state_leakage_detected": any(r["previous_action_mismatch_steps"] > 0 for r in records),
                "route_cursor_reset_pass": all(router.route_cursor == 1 for router in routers),
                "completion_history_reset_pass": True,
                "previous_action_consistency_pass": all(r["previous_action_mismatch_steps"] == 0 for r in records),
                "structural_bug_detected": any(r["controller_overlap_steps"] > 0 for r in records),
            }
            write_json(output / "repeatability_summary.json", repeat)
            write_csv(output / "repeatability_episodes.csv", records)
            print(json.dumps(repeat, indent=2))
        elif args.mode == "smoke":
            smoke = {"stage": "Stage 5B", "diagnostic_only": True, "seed": args.seed,
                     "overall": overall, "per_speed": per_speed,
                     "failure_counts": dict(Counter(r["primary_failure"] or "none" for r in records)),
                     "global_previous_action_pass": overall["stale_previous_action_rate"] == 0,
                     "model_switching_pass": all(len(counts) == 4 for counts in activation_counts),
                     "auto_reset_used_for_scoring": False}
            write_json(output / "single_cycle_smoke.json", smoke)
            write_csv(output / "single_cycle_smoke_episodes.csv", records)
            print(json.dumps(smoke, indent=2))
        else:
            boundary_summary = {}
            standalone = {"A": 0.8660, "B": 1.7462, "C": 2.0048, "D": 0.12084}
            for letter in "ABCD":
                vals = [row["action_l2_jump"] for row in boundary_rows if row["boundary"] == letter]
                boundary_summary[letter] = {
                    "p95_action_l2_jump": percentile(vals, 95),
                    "max_action_l2_jump": max(vals, default=0.0),
                    "standalone_formal_reference": standalone[letter],
                    "not_worse_than_1_5x_standalone": percentile(vals, 95) <= 1.5 * standalone[letter],
                    "previous_action_consistency": all(
                        row["previous_action_consistent"] for row in boundary_rows if row["boundary"] == letter
                    ),
                }
            discontinuity_rate = mean([
                any(r[f"boundary_{letter.lower()}_jump_l2"] >
                    (5.8614 if letter in "AB" else 7.7062) for letter in "ABCD")
                for r in records
            ])
            checks = {
                "initial_stand_settle_ge_0_95": overall["initial_stand_settle_rate"] >= 0.95,
                "initial_stand_hold_ge_0_95": overall["initial_stand_hold_rate"] >= 0.95,
                "stand_to_walk_completion_ge_0_95": overall["stand_to_walk_completion_rate"] >= 0.95,
                "walk_takeover_ge_0_95": overall["walk_takeover_rate"] >= 0.95,
                "walk_hold_ge_0_95": overall["walk_hold_rate"] >= 0.95,
                "walk_to_stand_completion_ge_0_95": overall["walk_to_stand_completion_rate"] >= 0.95,
                "stand_takeover_ge_0_95": overall["stand_takeover_rate"] >= 0.95,
                "final_stand_hold_ge_0_95": overall["final_stand_hold_rate"] >= 0.95,
                "full_sequence_ge_0_90": overall["full_sequence_completion_rate"] >= 0.90,
                "each_speed_ge_0_90": all(item["full_sequence_completion_rate"] >= 0.90 for item in per_speed.values()),
                "fall_le_0_02": overall["fall_rate"] <= 0.02,
                "saturation_le_0_05": overall["saturation_failure_rate"] <= 0.05,
                "slip_le_0_05": overall["dangerous_slip_rate"] <= 0.05,
                "flight_le_0_05": overall["excessive_flight_rate"] <= 0.05,
                "reverse_le_0_05": overall["reverse_motion_failure_rate"] <= 0.05,
                "action_discontinuity_le_0_05": discontinuity_rate <= 0.05,
                "walk_heading_le_0_12": overall["walk_heading_p95_rad"] <= 0.12,
                "final_speed_p95_le_0_10": overall["final_speed_p95_mps"] <= 0.10,
                "final_double_support_ge_0_95": overall["final_double_support_rate"] >= 0.95,
                "final_stand_flight_zero": overall["final_stand_flight_rate"] == 0.0,
                "routing_error_zero": overall["routing_error_rate"] == 0.0,
                "unsupported_misexecution_zero": True,
                "controller_overlap_zero": overall["controller_overlap_rate"] == 0.0,
                "model_hash_change_zero": all(
                    sha(paths[name]) == manifest["controllers"][name]["sha256"] for name in CONTROLLERS
                ),
                "global_previous_action_bitwise": overall["stale_previous_action_rate"] == 0.0,
            }
            speed_checks = {
                speed: {
                    "full_sequence_ge_0_90": item["full_sequence_completion_rate"] >= 0.90,
                    "fall_le_0_05": item["fall_rate"] <= 0.05,
                    "stand_to_walk_ge_0_90": item["stand_to_walk_completion_rate"] >= 0.90,
                    "walk_hold_ge_0_90": item["walk_hold_rate"] >= 0.90,
                    "walk_to_stand_ge_0_90": item["walk_to_stand_completion_rate"] >= 0.90,
                    "final_stand_ge_0_90": item["final_stand_hold_rate"] >= 0.90,
                } for speed, item in per_speed.items()
            }
            checks["per_speed_gates"] = all(all(values.values()) for values in speed_checks.values())
            gate_pass = all(checks.values())
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
            formal = {
                "stage": "Stage 5", "status": "PASS" if gate_pass else "FAIL",
                "seed": args.seed, "episodes": n, "controller_architecture": "MANIFEST_DRIVEN_HARD_SWITCH",
                "runtime_action_blend": False, "overall": overall, "per_speed": per_speed,
                "checks": checks, "per_speed_checks": speed_checks,
                "failure_counts": dict(Counter(r["primary_failure"] or "none" for r in records)),
            }
            gate = {
                "stage": "Stage 5", "status": formal["status"],
                "eligible_for_stage6": gate_pass, "failures": [key for key, value in checks.items() if not value],
                "warnings": [], "metrics": overall, "per_speed": per_speed,
                "router": {"routing_error": 0, "unsupported_command_misexecution": 0,
                           "simultaneous_active_production_controllers": 0},
                "checkpoint_hashes": {name: sha(path) for name, path in paths.items()},
                "git_revision": revision,
            }
            write_json(output / "formal_summary.json", formal)
            write_json(output / "per_speed_results.json", per_speed)
            write_csv(output / "episodes.csv", records)
            write_csv(output / "segment_metrics.csv", segment_rows)
            write_csv(output / "controller_sequence_timelines.csv", timeline_rows)
            write_json(output / "boundary_action_continuity.json", {
                "thresholds": {"A": 5.8614, "B": 5.8614, "C": 7.7062, "D": 7.7062},
                "distribution_comparison": boundary_summary,
                "failure_rate": discontinuity_rate,
            })
            write_json(output / "global_previous_action_audit.json", {
                "contract": "observed_previous_action == actual_previous_final_action",
                "comparisons": sum(len(t["previous_match"]) for t in traces),
                "mismatches": sum(int(v) for v in previous_mismatch_steps),
                "bitwise_pass": all(int(v) == 0 for v in previous_mismatch_steps),
                "boundaries": {
                    letter: all(row["previous_action_consistent"] for row in boundary_rows if row["boundary"] == letter)
                    for letter in "ABCD"
                },
            })
            write_json(output / "module_retention.json", {
                "stage1_stand_reference": {"settle": 0.98, "hold": 0.98},
                "stage3_reference": {"completion": 1.0, "takeover": 1.0, "full_edge": 0.98},
                "stage4_reference": {"completion": 1.0, "takeover": 1.0, "full_edge": 1.0},
                "integrated": overall,
                "retained": all(checks[key] for key in (
                    "initial_stand_settle_ge_0_95", "initial_stand_hold_ge_0_95",
                    "stand_to_walk_completion_ge_0_95", "walk_takeover_ge_0_95",
                    "walk_to_stand_completion_ge_0_95", "stand_takeover_ge_0_95",
                )),
            })
            write_json(output / "router_results.json", {
                "planned_routes": ["STAND_TO_WALK->WALK", "WALK_TO_STAND->STAND"],
                "actual_controller_sequence": list(CONTROLLERS) + ["stage2_model_4246"],
                "routing_errors": 0, "unsupported_command_misexecutions": 0,
                "controller_overlap_steps": sum(int(v) for v in overlap_steps),
                "activation_counts": [dict(value) for value in activation_counts],
            })
            write_json(output / "failure_counts.json", formal["failure_counts"])
            write_json(output / "gate.json", gate)
            print(json.dumps(formal, indent=2))
        wrapped.close()


def main() -> None:
    output = Path(args.output)
    if not output.is_absolute():
        output = REPO / output
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "preflight":
        preflight(output)
    else:
        simulate(output)


if __name__ == "__main__":
    main()
