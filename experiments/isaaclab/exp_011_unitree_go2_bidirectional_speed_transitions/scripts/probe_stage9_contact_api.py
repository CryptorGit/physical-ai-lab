"""Read-only runtime probe of the PhysX detailed-contact tensor contract."""

from __future__ import annotations

import argparse
import json
import traceback
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage9_contact_kinematics_heading_diagnosis"
CHECKPOINT = REPO / ".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/Assets/Isaac/6.0/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt"

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app
sys.argv = [sys.argv[0]]

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from isaaclab_physx.sensors import ContactSensorCfg  # noqa: E402

sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: E402,F401
from go2_bidirectional.evaluation import build_runner  # noqa: E402
from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402


def tensor_info(item):
    tensor = wp.to_torch(item)
    finite = tensor[torch.isfinite(tensor)] if tensor.dtype.is_floating_point else tensor.to(torch.int64)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "minimum": float(finite.min()) if finite.numel() else None,
        "maximum": float(finite.max()) if finite.numel() else None,
        "nonzero": int((tensor != 0).sum()),
        "sample": tensor.reshape(-1)[:12].tolist(),
    }


def main():
    cfg, agent = resolve_task_config("Isaac-Velocity-Flat-Unitree-Go2-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 2
    cfg.seed = 20268901
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    cfg.scene.stage9_probe = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FL_foot",
        update_period=0.0,
        track_pose=True,
        track_contact_points=True,
        track_friction_forces=True,
        max_contact_data_count_per_prim=16,
        filter_prim_paths_expr=["/World/ground/terrain/GroundPlane/CollisionPlane"],
    )
    raw = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=cfg)
    wrapped, _, policy = build_runner(raw, agent, CHECKPOINT)
    env = wrapped.unwrapped
    term = env.command_manager.get_term("base_velocity")
    sensor = env.scene.sensors["stage9_probe"]
    wrapped.reset()
    term.vel_command_b[:] = 0.0
    for _ in range(80):
        with torch.inference_mode():
            wrapped.step(policy(wrapped.get_observations()))
    contact = sensor.contact_view.get_contact_data(dt=sensor._sim_physics_dt)
    friction = sensor.contact_view.get_friction_data(dt=sensor._sim_physics_dt)
    result = {
        "contact_tuple_length": len(contact),
        "contact_tuple": [tensor_info(item) for item in contact],
        "friction_tuple_length": len(friction),
        "friction_tuple": [tensor_info(item) for item in friction],
        "contact_view_type": f"{type(sensor.contact_view).__module__}.{type(sensor.contact_view).__name__}",
        "filter_count": int(sensor.contact_view.filter_count),
        "physics_dt": float(sensor._sim_physics_dt),
        "sensor_data_contact_pos_shape": list(sensor.data.contact_pos_w.torch.shape),
        "sensor_data_friction_shape": list(sensor.data.friction_forces_w.torch.shape),
        "available_methods": [
            name for name in dir(sensor.contact_view)
            if any(token in name.lower() for token in ("contact", "friction", "velocity", "patch"))
        ],
        "config": {
            "terrain_static_friction": float(cfg.sim.physics_material.static_friction),
            "terrain_dynamic_friction": float(cfg.sim.physics_material.dynamic_friction),
            "material_randomization": str(cfg.events.physics_material.params),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "contact_api_runtime_probe.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    wrapped.close()


try:
    main()
except Exception as exc:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "contact_api_probe_error.txt").write_text(
        f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", encoding="utf-8"
    )
    raise
finally:
    simulation_app.close()
