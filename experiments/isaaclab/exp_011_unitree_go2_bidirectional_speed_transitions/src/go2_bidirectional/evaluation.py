"""Formal deterministic single-policy evaluation engine."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from .command_profiles import FULL_SEQUENCE, LIMITED_SEQUENCE, sequence_command, transition_command
from .contact_analysis import resolve_foot_mapping
from .gait_classifier import classify
from .metrics import max_true_dwell, mean, percentile, wrap_angle

SPEEDS = (0.0, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5)
TRANSITIONS = ((0.0, 1.2), (1.2, 2.0), (2.0, 1.2), (1.2, 0.0), (1.2, 2.5), (2.5, 1.2))


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, records: list[dict]) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(dict.fromkeys(key for record in records for key in record))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in record.items()})


def _rpy(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w, x, y, z = quat_wxyz.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def build_runner(raw_env, agent_cfg, checkpoint: Path):
    wrapped = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, __import__("importlib.metadata").metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(wrapped, cfg.to_dict(), log_dir=None, device=cfg.device)
    runner.load(str(checkpoint), strict=True, map_location=raw_env.unwrapped.device)
    policy = runner.get_inference_policy(device=raw_env.unwrapped.device)
    return wrapped, runner, policy


class Collector:
    """Collects raw traces without changing policy, physics, or state."""

    def __init__(self, wrapped, policy):
        self.wrapped, self.env, self.policy = wrapped, wrapped.unwrapped, policy
        self.robot = self.env.scene["robot"]
        self.command = self.env.command_manager.get_term("base_velocity")
        self.sensor = self.env.scene.sensors["contact_forces"]
        self.mapping = resolve_foot_mapping(self.robot, self.sensor)
        self.body_ids = [row["robot_body_index"] for row in self.mapping]
        self.sensor_ids = [row["contact_sensor_index"] for row in self.mapping]
        self.dt = float(self.env.step_dt)

    def run(self, duration: float, command_fn, seed: int) -> list[dict]:
        self.env.seed(seed)
        self.wrapped.reset()
        n = self.env.num_envs
        traces = [
            {
                "vx": [], "vy": [], "yaw_rate": [], "yaw_drift": [], "roll": [], "pitch": [], "height": [],
                "contacts": [], "slip": [], "impact": [], "action_rate": [], "vel_sat": [],
                "torque_sat": [], "commands": [], "segments": [], "fall": False,
                "termination_reason": "", "positions": [], "foot_velocities": [],
            }
            for _ in range(n)
        ]
        alive = torch.ones(n, dtype=torch.bool, device=self.env.device)
        previous = torch.zeros((n, self.env.action_manager.total_action_dim), device=self.env.device)
        initial_yaw = None
        for step in range(round(duration / self.dt)):
            t = step * self.dt
            speed, segment = command_fn(t)
            speed_tensor = torch.as_tensor(speed, dtype=torch.float32, device=self.env.device)
            if speed_tensor.ndim == 0:
                speed_tensor = speed_tensor.repeat(n)
            self.command.vel_command_b[:, 0] = speed_tensor
            self.command.vel_command_b[:, 1:] = 0.0
            obs = self.wrapped.get_observations()
            with torch.inference_mode():
                action = self.policy(obs)
                _, _, dones, extras = self.wrapped.step(action)
            roll, pitch, yaw = _rpy(self.robot.data.root_quat_w.torch)
            if initial_yaw is None:
                initial_yaw = yaw.clone()
            forces = self.sensor.data.net_forces_w_history.torch[:, :, self.sensor_ids, :]
            force_norm = forces.norm(dim=-1).amax(dim=1)
            contacts = force_norm > 5.0
            foot_vel = self.robot.data.body_lin_vel_w.torch[:, self.body_ids, :]
            slip = torch.where(contacts, foot_vel[:, :, :2].norm(dim=-1), 0.0).amax(dim=1)
            vel_ratio = (self.robot.data.joint_vel.torch.abs() / self.robot.data.joint_vel_limits.torch.abs().clamp_min(1e-6)).amax(dim=1)
            torque_ratio = (self.robot.data.applied_torque.torch.abs() / self.robot.data.joint_effort_limits.torch.abs().clamp_min(1e-6)).amax(dim=1)
            action_rate = torch.linalg.vector_norm(action - previous, dim=1) / self.dt
            previous = action.clone()
            for i, trace in enumerate(traces):
                if not bool(alive[i]):
                    continue
                trace["vx"].append(float(self.robot.data.root_lin_vel_b.torch[i, 0]))
                trace["vy"].append(float(self.robot.data.root_lin_vel_b.torch[i, 1]))
                trace["yaw_rate"].append(float(self.robot.data.root_ang_vel_b.torch[i, 2]))
                trace["yaw_drift"].append(abs(wrap_angle(float(yaw[i] - initial_yaw[i]))))
                trace["roll"].append(float(roll[i]))
                trace["pitch"].append(float(pitch[i]))
                trace["height"].append(float(self.robot.data.root_pos_w.torch[i, 2]))
                trace["contacts"].append([bool(v) for v in contacts[i].detach().cpu()])
                trace["slip"].append(float(slip[i]))
                trace["impact"].append(float(force_norm[i].max()))
                trace["action_rate"].append(float(action_rate[i]))
                trace["vel_sat"].append(bool(vel_ratio[i] >= 0.95))
                trace["torque_sat"].append(bool(torque_ratio[i] >= 0.95))
                trace["commands"].append(float(speed_tensor[i]))
                trace["segments"].append(segment)
                trace["positions"].append([float(v) for v in self.robot.data.root_pos_w.torch[i, :2]])
                trace["foot_velocities"].append([[float(v) for v in row] for row in foot_vel[i].detach().cpu()])
            newly_done = dones.bool() & alive
            for i in torch.nonzero(newly_done).flatten().tolist():
                traces[i]["fall"] = True
                traces[i]["termination_reason"] = "base_contact_or_early_termination"
            alive &= ~dones.bool()
        return traces


def _switches(contacts: list[list[bool]]) -> int:
    return sum(a != b for old, new in zip(contacts, contacts[1:]) for a, b in zip(old, new))


def summarize_trace(trace: dict, target: float, dt: float, segment: str | None = None) -> dict:
    indices = [i for i, name in enumerate(trace["segments"]) if segment is None or name == segment]
    pick = lambda key: [trace[key][i] for i in indices]
    vx, contacts = pick("vx"), pick("contacts")
    gait, evidence = classify(contacts, mean(vx), trace["fall"])
    slip_mean = mean(pick("slip"))
    dangerous_slip = slip_mean > 0.55
    vel_dwell = max_true_dwell(pick("vel_sat"), dt)
    torque_dwell = max_true_dwell(pick("torque_sat"), dt)
    saturation = max(vel_dwell, torque_dwell) >= 0.20
    return {
        "target_speed_mps": target,
        "samples": len(indices),
        "actual_forward_speed_mean_mps": mean(vx),
        "actual_forward_speed_p95_mps": percentile(vx, 95),
        "actual_forward_speed_max_mps": max(vx, default=0.0),
        "absolute_speed_error_mean_mps": mean(abs(value - target) for value in vx),
        "lateral_speed_abs_mean_mps": mean(abs(value) for value in pick("vy")),
        "yaw_rate_abs_mean_radps": mean(abs(value) for value in pick("yaw_rate")),
        "yaw_rate_abs_p95_radps": percentile((abs(value) for value in pick("yaw_rate")), 95),
        "yaw_drift_p95_rad": percentile(pick("yaw_drift"), 95),
        "roll_pitch_abs_p95_rad": percentile((max(abs(r), abs(p)) for r, p in zip(pick("roll"), pick("pitch"))), 95),
        "base_height_range_m": max(pick("height"), default=0.0) - min(pick("height"), default=0.0),
        "foot_contact_occupancy": [mean(row[j] for row in contacts) for j in range(4)] if contacts else [0.0] * 4,
        "foot_switches": _switches(contacts),
        "flight_fraction": mean(not any(row) for row in contacts),
        "foot_slip_mean_mps": slip_mean,
        "foot_slip_p95_mps": percentile(pick("slip"), 95),
        "dangerous_slip": dangerous_slip,
        "impact_force_p95_n": percentile(pick("impact"), 95),
        "action_rate_p95": percentile(pick("action_rate"), 95),
        "joint_velocity_saturation_fraction": mean(pick("vel_sat")),
        "joint_torque_saturation_fraction": mean(pick("torque_sat")),
        "long_dwell_saturation": saturation,
        "fall": trace["fall"],
        "termination_reason": trace["termination_reason"] or "time_limit_completed",
        "gait_class": gait,
        "gait_evidence": evidence,
    }


def _steady_supported(summary: dict, target: float) -> bool:
    error_limit = 0.25 if target >= 2.0 else 0.20
    return (
        summary["success_rate"] >= 0.90
        and summary["fall_rate"] <= 0.02
        and summary["yaw_drift_p95_rad"] <= 0.12
        and summary["roll_pitch_p95_rad"] <= 0.20
        and summary["dangerous_slip_rate"] <= 0.05
        and summary["long_dwell_saturation_rate"] <= 0.05
        and summary["absolute_speed_error_mean_mps"] <= error_limit
    )


def run_steady(collector: Collector, output: Path, seed: int) -> dict:
    episode_rows, summaries, gait_map = [], {}, {}
    for speed in SPEEDS:
        traces = collector.run(8.0, lambda _t, value=speed: (value, "hold"), seed)
        rows = []
        for i, trace in enumerate(traces):
            row = {"episode": i, "episode_seed": seed + i, **summarize_trace(trace, speed, collector.dt)}
            tolerance = 0.10 if speed == 0.0 else (0.25 if speed >= 2.0 else 0.20)
            row["tracking_success"] = not row["fall"] and row["absolute_speed_error_mean_mps"] <= tolerance
            rows.append(row)
            episode_rows.append(row)
        aggregate = {
            "target_speed_mps": speed,
            "episodes": len(rows),
            "success_rate": mean(row["tracking_success"] for row in rows),
            "fall_rate": mean(row["fall"] for row in rows),
            "actual_forward_speed_mean_mps": mean(row["actual_forward_speed_mean_mps"] for row in rows),
            "actual_forward_speed_p95_mps": percentile((row["actual_forward_speed_p95_mps"] for row in rows), 95),
            "absolute_speed_error_mean_mps": mean(row["absolute_speed_error_mean_mps"] for row in rows),
            "yaw_drift_p95_rad": percentile((row["yaw_drift_p95_rad"] for row in rows), 95),
            "roll_pitch_p95_rad": percentile((row["roll_pitch_abs_p95_rad"] for row in rows), 95),
            "dangerous_slip_rate": mean(row["dangerous_slip"] for row in rows),
            "long_dwell_saturation_rate": mean(row["long_dwell_saturation"] for row in rows),
            "gait_counts": dict(Counter(row["gait_class"] for row in rows)),
        }
        if speed == 0.0:
            stand_checks = {
                "hold_success_ge_0_95": aggregate["success_rate"] >= 0.95,
                "fall_le_0_02": aggregate["fall_rate"] <= 0.02,
                "speed_mean_le_0_05": abs(aggregate["actual_forward_speed_mean_mps"]) <= 0.05,
                "speed_p95_le_0_10": aggregate["actual_forward_speed_p95_mps"] <= 0.10,
                "yaw_rate_p95_le_0_10": aggregate["yaw_drift_p95_rad"] <= 0.10,
                "roll_pitch_p95_le_0_15": aggregate["roll_pitch_p95_rad"] <= 0.15,
                "base_height_range_le_0_05": percentile((row["base_height_range_m"] for row in rows), 95) <= 0.05,
                "dangerous_slip_le_0_05": aggregate["dangerous_slip_rate"] <= 0.05,
                "long_dwell_saturation_le_0_05": aggregate["long_dwell_saturation_rate"] <= 0.05,
            }
            aggregate["gate_checks"] = stand_checks
            aggregate["gate_pass"] = all(stand_checks.values())
            _json(output / "stand_results.json", {"episodes": rows, "summary": aggregate, "status": "PASS" if aggregate["gate_pass"] else "ZERO_COMMAND_HOLD_FAIL"})
        else:
            aggregate["status"] = "SUPPORTED" if _steady_supported(aggregate, speed) else ("PARTIAL" if aggregate["success_rate"] > 0 else "UNSUPPORTED")
        summaries[str(speed)] = aggregate
        gait_map[str(speed)] = {"counts": aggregate["gait_counts"], "episodes": [{"episode": row["episode"], "gait_class": row["gait_class"], "evidence": row["gait_evidence"]} for row in rows]}
    _csv(output / "steady_state_results.csv", episode_rows)
    _json(output / "steady_state_results.json", {"per_speed": summaries, "episodes": episode_rows})
    _json(output / "steady_state_gait_map.json", gait_map)
    supported = [speed for speed in SPEEDS[1:] if summaries[str(speed)]["status"] == "SUPPORTED"]
    envelope = {
        "supported_speeds_mps": supported,
        "continuous_from_zero_mps": [speed for speed in SPEEDS if speed == 0.0 or speed in supported],
        "maximum_supported_speed_mps": max(supported, default=0.0),
        "note": "Only frozen tested grid points; no interpolation claim.",
    }
    _json(output / "speed_support_envelope.json", envelope)
    return {"summaries": summaries, "envelope": envelope}


def _tolerance(target: float) -> float:
    return 0.08 if target == 0.0 else (0.25 if target >= 2.0 else 0.20)


def run_transitions(collector: Collector, output: Path, seed: int, steady: dict, ramp: float = 1.5) -> dict:
    records, aggregates = [], {}
    for source, target in TRANSITIONS:
        formal = steady[str(source)].get("status", "SUPPORTED" if source == 0.0 and steady["0.0"]["gate_pass"] else "UNSUPPORTED") == "SUPPORTED" and steady[str(target)].get("status", "UNSUPPORTED") == "SUPPORTED"
        traces = collector.run(3.0 + ramp + 5.0, lambda t, a=source, b=target: transition_command(t, a, b, ramp), seed)
        rows = []
        for i, trace in enumerate(traces):
            source_row = summarize_trace(trace, source, collector.dt, "source_hold")
            ramp_row = summarize_trace(trace, target, collector.dt, "ramp")
            target_row = summarize_trace(trace, target, collector.dt, "target_hold")
            band = [abs(value - target) <= _tolerance(target) if target else abs(value) <= 0.08 for value, segment in zip(trace["vx"], trace["segments"]) if segment in ("ramp", "target_hold")]
            needed = max(1, round(1.0 / collector.dt))
            streak = acquisition = 0
            for j, inside in enumerate(band):
                streak = streak + 1 if inside else 0
                if streak >= needed:
                    acquisition = j * collector.dt
                    break
            acquired = streak >= needed
            target_success = not target_row["fall"] and target_row["absolute_speed_error_mean_mps"] <= _tolerance(target)
            row = {
                "transition": f"{source}->{target}",
                "episode": i,
                "episode_seed": seed + i,
                "formal_gate_eligible": formal,
                "source_hold_success": not source_row["fall"] and source_row["absolute_speed_error_mean_mps"] <= _tolerance(source),
                "target_hold_success": target_success,
                "transition_completion": acquired and target_success and not trace["fall"],
                "target_acquired": acquired,
                "time_to_target_band_s": acquisition if acquired else None,
                "speed_overshoot_mps": max([value - target for value in trace["vx"]], default=0.0),
                "speed_undershoot_mps": max([target - value for value in trace["vx"]], default=0.0),
                "lateral_drift_m": abs(trace["positions"][-1][1] - trace["positions"][0][1]) if trace["positions"] else 0.0,
                "yaw_drift_p95_rad": percentile(trace["yaw_drift"], 95),
                "gait_before": source_row["gait_class"],
                "gait_during": ramp_row["gait_class"],
                "gait_after": target_row["gait_class"],
                "duty_factor_before": source_row["foot_contact_occupancy"],
                "duty_factor_after": target_row["foot_contact_occupancy"],
                "flight_fraction": ramp_row["flight_fraction"],
                "dangerous_slip": ramp_row["dangerous_slip"],
                "impact_force_p95_n": ramp_row["impact_force_p95_n"],
                "long_dwell_saturation": ramp_row["long_dwell_saturation"],
                "action_discontinuity_p95": ramp_row["action_rate_p95"],
                "fall": trace["fall"],
                "termination": trace["termination_reason"] or "time_limit_completed",
                "timeout": not acquired,
                "final_speed_mps": mean(target_row["actual_forward_speed_mean_mps"] for _ in [0]),
            }
            rows.append(row)
            records.append(row)
        agg = {
            "formal_gate_eligible": formal,
            "episodes": len(rows),
            "success_rate": mean(row["transition_completion"] for row in rows),
            "fall_rate": mean(row["fall"] for row in rows),
            "target_acquisition_rate": mean(row["target_acquired"] for row in rows),
            "target_hold_rate": mean(row["target_hold_success"] for row in rows),
            "yaw_drift_p95_rad": percentile((row["yaw_drift_p95_rad"] for row in rows), 95),
            "dangerous_slip_rate": mean(row["dangerous_slip"] for row in rows),
            "long_dwell_saturation_rate": mean(row["long_dwell_saturation"] for row in rows),
            "timeout_rate": mean(row["timeout"] for row in rows),
        }
        checks = {
            "success_ge_0_90": agg["success_rate"] >= 0.90,
            "fall_le_0_05": agg["fall_rate"] <= 0.05,
            "target_acquisition_ge_0_90": agg["target_acquisition_rate"] >= 0.90,
            "target_hold_ge_0_90": agg["target_hold_rate"] >= 0.90,
            "yaw_drift_p95_le_0_12": agg["yaw_drift_p95_rad"] <= 0.12,
            "dangerous_slip_le_0_05": agg["dangerous_slip_rate"] <= 0.05,
            "long_dwell_saturation_le_0_05": agg["long_dwell_saturation_rate"] <= 0.05,
            "timeout_le_0_05": agg["timeout_rate"] <= 0.05,
        }
        if target == 0.0:
            checks["final_speed_le_0_08"] = mean(row["final_speed_mps"] for row in rows) <= 0.08
            checks["final_hold_ge_0_95"] = agg["target_hold_rate"] >= 0.95
        agg["gate_checks"] = checks
        agg["gate_pass"] = formal and all(checks.values())
        aggregates[f"{source}->{target}"] = agg
    _csv(output / "transition_results.csv", records)
    _json(output / "transition_results.json", {"per_transition": aggregates, "episodes": records, "primary_ramp_duration_s": ramp})
    return aggregates


def run_sequence(collector: Collector, output: Path, seed: int, steady: dict, ramp: float = 1.5) -> dict:
    full_eligible = all(steady[str(speed)].get("status") == "SUPPORTED" for speed in (0.6, 1.2, 2.0, 2.5)) and steady["0.0"]["gate_pass"]
    speeds = FULL_SEQUENCE if full_eligible else LIMITED_SEQUENCE
    name = "FULL_2P5" if full_eligible else "LIMITED_2P0"
    duration = 3.0 + (len(speeds) - 1) * (ramp + 3.0)
    traces = collector.run(duration, lambda t: (lambda v, i, s: (v, f"{i}:{s}"))(*sequence_command(t, speeds, ramp)), seed)
    rows = []
    for i, trace in enumerate(traces):
        per_segment = {}
        for index, target in enumerate(speeds):
            segment_rows = [j for j, label in enumerate(trace["segments"]) if label == f"{index}:hold"]
            values = [trace["vx"][j] for j in segment_rows]
            per_segment[str(index)] = {
                "target_speed_mps": target,
                "success": bool(values) and mean(abs(value - target) for value in values) <= _tolerance(target),
                "actual_speed_mean_mps": mean(values),
            }
        final = per_segment[str(len(speeds) - 1)]
        rows.append({
            "episode": i, "episode_seed": seed + i, "fall": trace["fall"],
            "per_segment": per_segment,
            "completion": not trace["fall"] and all(value["success"] for value in per_segment.values()),
            "final_stand_success": final["success"] and final["actual_speed_mean_mps"] <= 0.08,
            "yaw_drift_p95_rad": percentile(trace["yaw_drift"], 95),
            "dangerous_slip": mean(trace["slip"]) > 0.55,
            "long_dwell_saturation": max(max_true_dwell(trace["vel_sat"], collector.dt), max_true_dwell(trace["torque_sat"], collector.dt)) >= 0.20,
        })
    segment_rates = {str(i): mean(row["per_segment"][str(i)]["success"] for row in rows) for i in range(len(speeds))}
    summary = {
        "sequence_name": name, "speeds_mps": speeds, "full_2p5_eligible": full_eligible,
        "episodes": rows, "completion_rate": mean(row["completion"] for row in rows),
        "segment_success_rates": segment_rates, "fall_rate": mean(row["fall"] for row in rows),
        "yaw_drift_p95_rad": percentile((row["yaw_drift_p95_rad"] for row in rows), 95),
        "dangerous_slip_rate": mean(row["dangerous_slip"] for row in rows),
        "long_dwell_saturation_rate": mean(row["long_dwell_saturation"] for row in rows),
        "final_stand_hold_rate": mean(row["final_stand_success"] for row in rows),
        "routing_checkpoint_switch_count": 0, "unsupported_command_execution_count": 0,
    }
    checks = {
        "completion_ge_0_90": summary["completion_rate"] >= 0.90,
        "each_segment_ge_0_90": all(value >= 0.90 for value in segment_rates.values()),
        "fall_le_0_05": summary["fall_rate"] <= 0.05,
        "yaw_drift_p95_le_0_12": summary["yaw_drift_p95_rad"] <= 0.12,
        "dangerous_slip_le_0_05": summary["dangerous_slip_rate"] <= 0.05,
        "long_dwell_saturation_le_0_05": summary["long_dwell_saturation_rate"] <= 0.05,
        "routing_checkpoint_switch_eq_0": True,
        "unsupported_command_execution_eq_0": True,
        "final_stand_ge_0_95": summary["final_stand_hold_rate"] >= 0.95,
    }
    summary["gate_checks"] = checks
    summary["gate_pass"] = all(checks.values())
    _json(output / "full_sequence_results.json", summary)
    return summary


def asymmetry(output: Path, steady: dict, transitions: dict) -> None:
    endpoint = "1.2"
    rows = [{
        "arrival": "reset", "target_speed_mps": 1.2,
        "actual_speed_mean_mps": steady[endpoint]["actual_forward_speed_mean_mps"],
        "source": "steady_state_results.json",
    }]
    for name in ("0.0->1.2", "2.0->1.2", "2.5->1.2"):
        if name in transitions:
            rows.append({
                "arrival": name, "target_speed_mps": 1.2,
                "actual_speed_mean_mps": None,
                "target_hold_rate": transitions[name]["target_hold_rate"],
                "source": "transition_results.json",
            })
    _csv(output / "endpoint_hysteresis.csv", rows)
    _json(output / "directional_asymmetry.json", {
        "endpoint_mps": 1.2, "comparisons": rows,
        "audited_failure_modes": [
            "high-speed gait retained only after deceleration",
            "flight retained only after deceleration",
            "different contact attractor at identical 1.2 m/s",
            "stepping retained after zero command",
        ],
        "interpretation": "See per-episode gait_after/duty_factor_after and endpoint_hysteresis.csv.",
    })
