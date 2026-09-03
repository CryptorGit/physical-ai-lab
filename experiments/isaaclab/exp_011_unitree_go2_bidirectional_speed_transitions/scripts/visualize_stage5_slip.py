"""Capture predetermined Stage 5 slip-validation frame sequences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage5_endpoint_failure_diagnosis"
SELECTED = (
    REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
    "stage4_resumed_optimizer_training/checkpoints/model_50.pt"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=OUT)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
sys.argv = [sys.argv[0]]

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: E402,F401
from go2_bidirectional.contact_analysis import resolve_foot_mapping  # noqa: E402
from go2_bidirectional.evaluation import build_runner  # noqa: E402
from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402

SEED = 20263901


def choose(rows):
    selected = []
    groups = {
        0.0: [row for row in rows if row["speed_mps"] == 0.0][:2],
        0.6: [row for row in rows if row["speed_mps"] == 0.6][:2],
        1.2: [row for row in rows if row["speed_mps"] == 1.2][:2],
        2.0: [row for row in rows if row["speed_mps"] == 2.0][:2],
    }
    point4 = [row for row in rows if row["speed_mps"] == 0.4]
    groups[0.4] = (
        [row for row in point4 if not row["fall"]][:2]
        + [row for row in point4 if row["fall"]][:2]
    )
    for speed, episodes in groups.items():
        for row in episodes:
            selected.append({
                "speed_mps": speed, "episode": row["episode"], "seed_label": row["seed"],
                "outcome": "fall" if row["fall"] else "success",
                "selection_rule": (
                    "first two successful then first two failed in ascending episode order"
                    if speed == 0.4 else "first two episodes in ascending episode order"
                ),
            })
    return selected


def main():
    raw = json.loads((args.output / "raw_episode_summaries.json").read_text(encoding="utf-8"))
    stage4 = [row for row in raw if row["checkpoint"] == "stage4_selected"]
    selections = choose(stage4)
    cfg, agent = resolve_task_config(
        "Isaac-Velocity-Flat-Unitree-Go2-v0", "rsl_rl_cfg_entry_point"
    )
    cfg.scene.num_envs = 50; cfg.seed = SEED; cfg.episode_length_s = 10.0
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None; cfg.events.push_robot = None
    if args.device:
        cfg.sim.device = args.device
    raw_env = gym.make("Isaac-Velocity-Flat-Unitree-Go2-v0", cfg=cfg, render_mode="rgb_array")
    wrapped, _, policy = build_runner(raw_env, agent, SELECTED)
    env = wrapped.unwrapped; robot = env.scene["robot"]
    sensor = env.scene.sensors["contact_forces"]
    command = env.command_manager.get_term("base_velocity")
    mapping = resolve_foot_mapping(robot, sensor)
    sensor_ids = [row["contact_sensor_index"] for row in mapping]
    body_ids = [row["robot_body_index"] for row in mapping]
    visual_root = args.output / "visual_slip_validation"
    visual_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    capture_steps = {25, 75, 125, 200, 275, 350}
    for selection in selections:
        env.seed(SEED); wrapped.reset()
        index = int(selection["episode"]); speed = float(selection["speed_mps"])
        episode_dir = visual_root / f"{speed:.1f}mps_ep{index:02d}_{selection['outcome']}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        files = []
        for step in range(400):
            command.vel_command_b[:, 0] = speed; command.vel_command_b[:, 1:] = 0.0
            with torch.inference_mode():
                action = policy(wrapped.get_observations())
                _, _, dones, _ = wrapped.step(action)
            if step in capture_steps or bool(dones[index]):
                root = robot.data.root_pos_w.torch[index].cpu()
                env.sim.set_camera_view(
                    eye=(float(root[0] - 2.8), float(root[1] - 2.2), float(root[2] + 1.2)),
                    target=(float(root[0] + 0.4), float(root[1]), float(root[2] - 0.1)),
                )
                image = raw_env.render()
                if image is not None:
                    frame = Image.fromarray(image)
                    draw = ImageDraw.Draw(frame)
                    forces = sensor.data.net_forces_w_history.torch[
                        index, :, sensor_ids, :
                    ].norm(dim=-1).amax(dim=0)
                    contacts = forces > 5.0
                    foot_speed = robot.data.body_lin_vel_w.torch[
                        index, body_ids, :2
                    ].norm(dim=-1)
                    slip_flags = contacts & (foot_speed > 0.55)
                    actual = float(robot.data.root_lin_vel_b.torch[index, 0])
                    text = (
                        f"Stage4 selected | target {speed:.1f} m/s | actual {actual:.3f} m/s\n"
                        f"episode {index} step {step} outcome {selection['outcome']}\n"
                        f"FL/FR/RL/RR contact {[int(v) for v in contacts.cpu().tolist()]}\n"
                        f"foot world speed {[round(v,3) for v in foot_speed.cpu().tolist()]}\n"
                        f"slip flag {[int(v) for v in slip_flags.cpu().tolist()]}\n"
                        f"force N {[round(v,1) for v in forces.cpu().tolist()]}"
                    )
                    draw.rectangle((8, 8, 720, 118), fill=(0, 0, 0))
                    draw.multiline_text((16, 14), text, fill=(255, 255, 255), spacing=3)
                    path = episode_dir / f"frame_{step:03d}.jpg"
                    frame.save(path, quality=92)
                    files.append(str(path.relative_to(args.output)))
            if bool(dones[index]):
                break
        manifest.append({**selection, "frames": files, "frame_count": len(files)})
        print(
            f"stage5_visual speed={speed} episode={index} outcome={selection['outcome']} frames={len(files)}",
            flush=True,
        )
    (args.output / "visual_slip_validation_manifest.json").write_text(
        json.dumps({
            "checkpoint": str(SELECTED), "selection_frozen_before_render": True,
            "selections": manifest, "video_staged": False,
            "overlay_fields": ["contact", "world foot velocity", "slip flag", "contact force"],
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    wrapped.close()


try:
    main()
finally:
    simulation_app.close()
