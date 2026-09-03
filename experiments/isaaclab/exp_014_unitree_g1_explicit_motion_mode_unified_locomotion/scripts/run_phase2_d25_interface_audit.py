"""D25 read-only robot/kinematics interface audit.

This is deliberately separate from the planner: no START request, policy update,
snapshot restore, or trajectory search is performed here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

# Importing the D16 runner registers the exp013 task and its local source tree.
import importlib.util


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_d16 = _load("d16_for_d25_interface", Path(__file__).resolve().parent / "run_phase2_d16_train.py")

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d25_model_based_first_step_teacher/raw"


def shape(value):
    return list(value.shape) if hasattr(value, "shape") else None


def serial(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return np.asarray(value).tolist()
    except Exception:
        return repr(value)


def main():
    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    args, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra_args]
    cfg, agent = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 1
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    if args.device:
        cfg.sim.device = agent.device = args.device
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg), clip_actions=agent.clip_actions)
        env = wrapped.unwrapped
        robot = env.scene["robot"]
        view = robot.root_physx_view
        errors = {}
        values = {}
        calls = {
            "jacobians": "get_jacobians",
            "mass_matrices": "get_mass_matrices",
            "masses": "get_masses",
            "inertias": "get_inertias",
        }
        for key, method in calls.items():
            try:
                values[key] = getattr(view, method)()
            except Exception as exc:  # recorded, never hidden
                errors[key] = f"{type(exc).__name__}: {exc}"
        data = robot.data
        body_fields = {}
        for field in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
                      "body_com_pos_w", "body_com_quat_w", "body_com_lin_vel_w", "body_com_ang_vel_w"):
            try:
                body_fields[field] = shape(getattr(data, field))
            except Exception as exc:
                errors[field] = f"{type(exc).__name__}: {exc}"
        term = env.action_manager.get_term("joint_pos")
        action_fields = {}
        for field in ("_scale", "_offset", "_clip", "raw_actions", "processed_actions"):
            try:
                value = getattr(term, field)
                action_fields[field] = {"shape": shape(value), "value": serial(value) if torch.is_tensor(value) and value.numel() <= 100 else None}
            except Exception as exc:
                errors[f"action.{field}"] = f"{type(exc).__name__}: {exc}"
        sensor = env.scene.sensors.get("contact_forces")
        report = {
            "task": "Isaac-Exp013-G1-DirectionalBaseline-v0",
            "device": str(robot.device),
            "body_names": list(robot.body_names),
            "joint_names": list(robot.joint_names),
            "body_fields": body_fields,
            "api_shapes": {key: shape(value) for key, value in values.items()},
            "total_mass_kg": float(np.asarray(values["masses"])[0].sum()) if "masses" in values else None,
            "masses_kg": serial(values["masses"][0]) if "masses" in values else None,
            "joint_position_limits": serial(data.soft_joint_pos_limits[0]),
            "joint_velocity_limits": serial(data.soft_joint_vel_limits[0]),
            "default_joint_positions": serial(data.default_joint_pos[0]),
            "action_term": action_fields,
            "control_dt_s": float(env.step_dt),
            "physics_dt_s": float(env.physics_dt),
            "decimation": int(round(env.step_dt / env.physics_dt)),
            "contact_sensor_present": sensor is not None,
            "errors": errors,
            "kinematics_interface_available": "jacobians" in values and "mass_matrices" in values and "masses" in values,
        }
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "interface_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"kinematics_interface_available": report["kinematics_interface_available"], "api_shapes": report["api_shapes"], "errors": errors}, indent=2))
        wrapped.close()


if __name__ == "__main__":
    main()
