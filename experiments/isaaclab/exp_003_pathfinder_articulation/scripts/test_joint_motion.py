"""膝、次に肘へ小さなsin波位置目標を与える診断スクリプト。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_USD = (
    REPO_ROOT / "shared" / "models" / "pathfinder"
    / "usd" / "pathfinder_articulation.usd"
)

# この辞書または--sequenceを変えるだけで、テスト対象を切り替えられる。
MOTION_GROUPS = {
    "knee": ["knee_L", "knee_R"],
    "elbow": ["elbow_L", "elbow_R"],
    "left_knee": ["knee_L"],
    "right_knee": ["knee_R"],
    "left_elbow": ["elbow_L"],
    "right_elbow": ["elbow_R"],
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd-path", type=Path, default=DEFAULT_USD)
parser.add_argument(
    "--sequence",
    nargs="+",
    choices=sorted(MOTION_GROUPS),
    default=["knee", "elbow"],
)
parser.add_argument("--segment-seconds", type=float, default=6.0)
parser.add_argument("--amplitude-deg", type=float, default=8.0)
parser.add_argument("--frequency-hz", type=float, default=0.25)
parser.add_argument("--print-period", type=float, default=0.5)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402


def create_robot(usd_path: Path) -> Articulation:
    if not usd_path.is_file():
        raise FileNotFoundError(f"Run the articulation builder first: {usd_path}")
    return Articulation(
        ArticulationCfg(
            prim_path="/World/Pathfinder",
            spawn=sim_utils.UsdFileCfg(usd_path=str(usd_path.resolve())),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.15)),
            actuators={
                "debug_joints": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    effort_limit_sim=150.0,
                    velocity_limit_sim=8.0,
                    stiffness=40.0,
                    damping=4.0,
                )
            },
        )
    )


def main() -> None:
    sim = SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    )
    sim.set_camera_view(eye=(3.5, 3.5, 2.4), target=(0.0, 0.0, 1.0))
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0)
    light_cfg.func("/World/Light", light_cfg)
    robot = create_robot(args_cli.usd_path.resolve())

    sim.reset()
    robot.reset()
    dt = sim.get_physics_dt()
    default_targets = robot.data.default_joint_pos.clone()
    amplitude = math.radians(args_cli.amplitude_deg)
    print_steps = max(1, round(args_cli.print_period / dt))
    segment_steps = max(1, round(args_cli.segment_seconds / dt))

    print(f"Available joints: {robot.joint_names}")
    print(f"Bodies: {robot.body_names}")
    print(f"Motion sequence: {args_cli.sequence}")

    for group_name in args_cli.sequence:
        joint_ids, joint_names = robot.find_joints(
            MOTION_GROUPS[group_name], preserve_order=True
        )
        if not joint_ids:
            raise RuntimeError(
                f"No joints matched {MOTION_GROUPS[group_name]}; "
                f"available={robot.joint_names}"
            )
        print(f"\n[TEST] {group_name}: {joint_names} (indices={joint_ids})")

        for step in range(segment_steps):
            if not simulation_app.is_running():
                return
            elapsed = step * dt
            offset = amplitude * math.sin(
                2.0 * math.pi * args_cli.frequency_hz * elapsed
            )
            targets = default_targets.clone()
            targets[:, joint_ids] += offset
            robot.set_joint_position_target_index(target=targets)
            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)

            if step % print_steps == 0:
                positions = (
                    robot.data.joint_pos[0, joint_ids].detach().cpu().tolist()
                )
                velocities = (
                    robot.data.joint_vel[0, joint_ids].detach().cpu().tolist()
                )
                position_deg = [math.degrees(value) for value in positions]
                velocity_deg = [math.degrees(value) for value in velocities]
                print(
                    f"  t={elapsed:5.2f}s "
                    f"target={math.degrees(offset):+6.2f}deg "
                    f"q_deg={position_deg} qd_deg_s={velocity_deg}"
                )

        robot.set_joint_position_target_index(target=default_targets)
        for _ in range(round(0.5 / dt)):
            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
