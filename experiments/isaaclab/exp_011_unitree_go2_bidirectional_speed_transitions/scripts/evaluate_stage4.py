"""Validate Stage 4 checkpoints or run the frozen formal/diagnostic evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from isaaclab.app import AppLauncher

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage4_resumed_optimizer_training"
PARENT = (
    REPO / ".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/Assets/Isaac/6.0/"
    "Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("validation", "formal"), required=True)
parser.add_argument("--output", type=Path, default=OUT)
parser.add_argument("--checkpoint", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
sys.argv = [sys.argv[0]]

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: E402,F401
import go2_bidirectional.evaluation as evaluation  # noqa: E402
from go2_bidirectional.command_profiles import transition_command  # noqa: E402
from go2_bidirectional.evaluation import Collector, asymmetry, build_runner, run_sequence, run_steady, run_transitions  # noqa: E402
from go2_bidirectional.metrics import mean  # noqa: E402
from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402

FORMAL_SPEEDS = (0.0, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0)
VALIDATION_SPEEDS = (0.0, 0.6, 1.2, 2.0)
FORMAL_TRANSITIONS = ((0.0, 1.2), (1.2, 2.0), (2.0, 1.2), (1.2, 0.0))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8"); return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def make_env(num_envs, seed, episode_length):
    cfg, agent = resolve_task_config(
        "Isaac-Velocity-Flat-Unitree-Go2-v0", "rsl_rl_cfg_entry_point"
    )
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.episode_length_s = episode_length
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    if args.device:
        cfg.sim.device = args.device
        agent.device = args.device
    return gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=cfg), agent


def validation():
    manifest = json.loads((args.output / "checkpoint_manifest.json").read_text())
    checkpoints = manifest["checkpoints"]
    raw, agent = make_env(10, 20261901, 45.0)
    wrapped, runner, policy = build_runner(raw, agent, Path(checkpoints[0]["path"]))
    collector = Collector(wrapped, policy)
    evaluation.SPEEDS = VALIDATION_SPEEDS
    evaluation.TRANSITIONS = FORMAL_TRANSITIONS
    rows = []
    for item in checkpoints:
        runner.load(str(Path(item["path"])), strict=True, map_location=raw.unwrapped.device)
        temp = args.output / "validation_work" / f"iter_{item['local_iteration']}"
        temp.mkdir(parents=True, exist_ok=True)
        steady_result = run_steady(collector, temp, 20261901)["summaries"]
        transitions = run_transitions(collector, temp, 20261901, steady_result)
        endpoints = (
            steady_result["0.0"]["gate_pass"]
            and all(steady_result[str(speed)]["status"] == "SUPPORTED" for speed in (0.6, 1.2, 2.0))
        )
        sequence = run_sequence(collector, temp, 20261901, steady_result) if endpoints else {"gate_pass": False}
        stand_pass = bool(steady_result["0.0"]["gate_pass"])
        steady_passes = sum(steady_result[str(speed)]["status"] == "SUPPORTED" for speed in (0.6, 1.2, 2.0))
        transition_passes = sum(value["gate_pass"] for value in transitions.values())
        hard_count = int(stand_pass) + steady_passes + transition_passes + int(sequence["gate_pass"])
        slip = mean(
            steady_result[str(speed)]["dangerous_slip_rate"] for speed in (0.6, 1.2, 2.0)
        )
        heading = mean(
            steady_result[str(speed)]["yaw_drift_p95_rad"] for speed in (0.6, 1.2, 2.0)
        )
        rows.append({
            "local_iteration": item["local_iteration"], "checkpoint": item["path"],
            "hard_gate_pass_count": hard_count, "zero_hold_pass": stand_pass,
            "steady_pass_count": steady_passes, "transition_pass_count": transition_passes,
            "reduced_sequence_pass": bool(sequence["gate_pass"]),
            "dangerous_slip_rate_mean": slip, "yaw_drift_p95_mean": heading,
        })
        print(f"VALIDATION iter={item['local_iteration']} hard={hard_count}", flush=True)
    rows.sort(key=lambda row: (
        -row["hard_gate_pass_count"], -int(row["zero_hold_pass"]),
        -row["steady_pass_count"], -row["transition_pass_count"],
        -int(row["reduced_sequence_pass"]), row["dangerous_slip_rate_mean"],
        row["yaw_drift_p95_mean"], row["local_iteration"],
    ))
    selected = rows[0]
    write_csv(args.output / "validation_checkpoint_results.csv", rows)
    dump(args.output / "selected_checkpoint.json", {
        "status": "SELECTED", "selection_frozen": True,
        "checkpoint": selected["checkpoint"], "local_iteration": selected["local_iteration"],
        "score": selected,
        "precedence": [
            "hard-gate pass count", "zero-command hold", "steady-state safety",
            "bidirectional transitions", "reduced sequence", "lower slip", "lower heading drift",
        ],
    })
    wrapped.close()


def move_formal_outputs(output):
    mapping = {
        "stand_results.json": "formal_stand_results.json",
        "steady_state_results.csv": "formal_steady_state_results.csv",
        "steady_state_results.json": "formal_steady_state_results.json",
        "transition_results.csv": "formal_transition_results.csv",
        "transition_results.json": "formal_transition_results.json",
        "full_sequence_results.json": "formal_reduced_sequence.json",
    }
    for source, target in mapping.items():
        source_path = output / source
        if source_path.exists():
            source_path.replace(output / target)


def run_legacy(collector, runner, checkpoints):
    results = {}
    conditions = {
        "vy_+0.5": (1.0, 0.5, 0.0), "vy_-0.5": (1.0, -0.5, 0.0),
        "yaw_+0.5": (1.0, 0.0, 0.5), "yaw_-0.5": (1.0, 0.0, -0.5),
    }
    for policy_name, checkpoint in checkpoints.items():
        runner.load(str(checkpoint), strict=True, map_location=collector.env.device)
        policy_results = {}
        for label, (vx, vy, yaw) in conditions.items():
            collector.env.seed(20262901)
            collector.wrapped.reset()
            samples = {
                "vx": [], "vy": [], "yaw": [], "slip": [], "fall": 0,
                "slip_sum_by_env": [0.0] * 20, "steps": 0,
            }
            for _ in range(round(8.0 / collector.dt)):
                collector.command.vel_command_b[:, 0] = vx
                collector.command.vel_command_b[:, 1] = vy
                collector.command.vel_command_b[:, 2] = yaw
                obs = collector.wrapped.get_observations()
                with torch.inference_mode():
                    action = collector.policy(obs)
                    _, _, dones, _ = collector.wrapped.step(action)
                samples["vx"].extend(collector.robot.data.root_lin_vel_b.torch[:20, 0].cpu().tolist())
                samples["vy"].extend(collector.robot.data.root_lin_vel_b.torch[:20, 1].cpu().tolist())
                samples["yaw"].extend(collector.robot.data.root_ang_vel_b.torch[:20, 2].cpu().tolist())
                forces = collector.sensor.data.net_forces_w_history.torch[
                    :20, :, collector.sensor_ids, :
                ].norm(dim=-1).amax(dim=1)
                contacts = forces > 5.0
                foot_speed = collector.robot.data.body_lin_vel_w.torch[
                    :20, collector.body_ids, :2
                ].norm(dim=-1)
                slip = torch.where(contacts, foot_speed, 0.0).amax(dim=1).cpu().tolist()
                samples["slip"].extend(slip)
                for index, value in enumerate(slip):
                    samples["slip_sum_by_env"][index] += value
                samples["steps"] += 1
                samples["fall"] += int(dones[:20].sum())
            slip_sorted = sorted(samples["slip"])
            policy_results[label] = {
                "vx_mean": mean(samples["vx"]), "vy_mean": mean(samples["vy"]),
                "yaw_rate_mean": mean(samples["yaw"]),
                "tracking_error": (
                    abs(mean(samples["vy"]) - vy) if vy else abs(mean(samples["yaw"]) - yaw)
                ),
                "termination_count": samples["fall"],
                "foot_slip_mean_mps": mean(samples["slip"]),
                "foot_slip_p95_mps": slip_sorted[round(0.95 * (len(slip_sorted) - 1))],
                "dangerous_slip_rate": mean(
                    total / samples["steps"] > 0.55
                    for total in samples["slip_sum_by_env"]
                ),
            }
        results[policy_name] = policy_results
    degradation = {
        label: results["stage4"][label]["tracking_error"] - results["official_parent"][label]["tracking_error"]
        for label in conditions
    }
    return {
        "diagnostic": "forward-only curriculum legacy command retention diagnostic",
        "episodes_per_condition": 20, "results": results, "tracking_error_degradation": degradation,
        "formal_capability_claim": False,
    }


def formal():
    selected = json.loads((args.output / "selected_checkpoint.json").read_text())
    checkpoint = args.checkpoint or Path(selected["checkpoint"])
    raw, agent = make_env(50, 20262901, 50.0)
    wrapped, runner, policy = build_runner(raw, agent, checkpoint)
    collector = Collector(wrapped, policy)
    evaluation.SPEEDS = FORMAL_SPEEDS
    evaluation.TRANSITIONS = FORMAL_TRANSITIONS
    steady = run_steady(collector, args.output, 20262901)["summaries"]
    transitions = run_transitions(collector, args.output, 20262901, steady)
    asymmetry(args.output, steady, transitions)
    endpoints = steady["0.0"]["gate_pass"] and all(
        steady[str(speed)]["status"] == "SUPPORTED" for speed in (0.6, 1.2, 2.0)
    )
    sequence = run_sequence(collector, args.output, 20262901, steady) if endpoints else {
        "status": "NOT_RUN", "reason": "required endpoint not SUPPORTED", "gate_pass": False
    }
    if not endpoints:
        dump(args.output / "full_sequence_results.json", sequence)
    formal_all = (
        steady["0.0"]["gate_pass"]
        and all(steady[str(speed)]["status"] == "SUPPORTED" for speed in FORMAL_SPEEDS[1:])
        and all(value["gate_pass"] for value in transitions.values())
        and sequence["gate_pass"]
    )
    if formal_all:
        # Keep OOD evaluation isolated from formal files.
        traces = collector.run(8.0, lambda _t: (2.5, "hold"), 20262901)
        steady_2p5 = [evaluation.summarize_trace(trace, 2.5, collector.dt) for trace in traces[:20]]
        diagnostics = {"steady_2p5": steady_2p5, "formal_capability": False}
        for source, target in ((1.2, 2.5), (2.5, 1.2)):
            traces = collector.run(
                9.5, lambda t, a=source, b=target: transition_command(t, a, b, 1.5), 20262901
            )
            diagnostics[f"{source}->{target}"] = [
                evaluation.summarize_trace(trace, target, collector.dt, "target_hold")
                for trace in traces[:20]
            ]
        dump(args.output / "diagnostic_2p5_results.json", diagnostics)
    else:
        dump(args.output / "diagnostic_2p5_results.json", {
            "status": "NOT_RUN", "reason": "all formal gates did not pass",
            "formal_capability": False,
        })
    legacy = run_legacy(
        collector, runner, {"official_parent": PARENT, "stage4": Path(checkpoint)}
    )
    dump(args.output / "legacy_command_retention.json", legacy)
    move_formal_outputs(args.output)
    wrapped.close()


if args.mode == "validation":
    validation()
else:
    formal()
simulation_app.close()
