"""Pathfinder Articulationを1体spawnし、GUIで確認する（RLなし）。"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_USD = (
    REPO_ROOT
    / "shared"
    / "models"
    / "pathfinder"
    / "usd"
    / "pathfinder_articulation.usd"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd-path", type=Path, default=DEFAULT_USD)
parser.add_argument(
    "--show-collisions",
    action="store_true",
    help="Collision APIが付いた形状を表示する。既定では見た目だけ非表示。",
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=0,
    help="0ならGUIを閉じるまで実行。正数は自動検証用の最大step数。",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

print(f"[DEBUG] AppLauncher args before override: {args_cli}")

# GUI起動を明示的に強制する
args_cli.headless = False

# livestreamが有効だとローカルウィンドウが出ない場合がある
if hasattr(args_cli, "livestream"):
    args_cli.livestream = 0

print(f"[DEBUG] headless={args_cli.headless}")
print(f"[DEBUG] livestream={getattr(args_cli, 'livestream', None)}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402


def create_robot(usd_path: Path) -> Articulation:
    if not usd_path.is_file():
        raise FileNotFoundError(
            f"Articulation USD not found: {usd_path}\n"
            "Run build_pathfinder_articulation.py first."
        )
    cfg = ArticulationCfg(
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
    return Articulation(cfg)


def set_physics_collision_visibility(
    root_path: str = "/World/Pathfinder",
    *,
    visible: bool,
) -> list[str]:
    """CollisionAPI付きPrimの表示だけを一括変更する。物理衝突は残す。"""
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Robot prim not found: {root_path}")

    changed: list[str] = []

    for prim in Usd.PrimRange(root_prim):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue

        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            continue

        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

        changed.append(prim.GetPath().pathString)

    return changed


def main() -> None:
    sim = SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    )
    sim.set_camera_view(eye=(3.5, 3.5, 2.4), target=(0.0, 0.0, 1.0))

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(
        intensity=2500.0, color=(0.8, 0.8, 0.8)
    )
    light_cfg.func("/World/Light", light_cfg)
    robot = create_robot(args_cli.usd_path.resolve())

    changed = set_physics_collision_visibility(
        "/World/Pathfinder",
        visible=args_cli.show_collisions,
    )
    mode = "visible" if args_cli.show_collisions else "hidden"
    print(f"[INFO] CollisionAPI visuals: {mode} ({len(changed)} prims)")
    for path in changed:
        print(f"  - {path}")

    sim.reset()
    robot.reset()
    print(f"Joint count (DoF): {robot.num_joints}")
    print(f"Joint names: {robot.joint_names}")
    print(f"Body count: {robot.num_bodies}")
    print(f"Body names: {robot.body_names}")

    step_count = 0
    while simulation_app.is_running():
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        step_count += 1
        if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
            break


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
