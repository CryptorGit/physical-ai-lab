"""physical-ai-labからIsaac Labを起動する最小実験。

Isaac SimをPythonから起動し、床と立方体を生成して、
物理シミュレーションを一定時間進める。
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


# Isaac Sim起動前に、AppLauncher用の引数を定義します。
parser = argparse.ArgumentParser(
    description="Launch a simple Isaac Lab simulation."
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# この処理でIsaac Simが起動します。
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# Isaac Sim起動後にIsaac Labのシミュレーション関連APIをimportします。
#
# 一部のIsaac Sim APIは、SimulationApp起動前にimportすると
# 正常に初期化されないため、この順番が重要です。
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext


def design_scene() -> None:
    """床、照明、立方体をUSDステージへ生成する。"""
    # 無限平面の床を生成します。
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func(
        "/World/defaultGroundPlane",
        ground_cfg,
    )

    # シーン全体を照らす平行光源を生成します。
    light_cfg = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    light_cfg.func(
        "/World/Light",
        light_cfg,
        translation=(1.0, 0.0, 10.0),
    )

    # 動的な立方体を生成します。
    cube_cfg = sim_utils.CuboidCfg(
        size=(0.5, 0.5, 0.5),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(
            mass=1.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.2, 0.5, 0.8),
        ),
    )

    cube_cfg.func(
        "/World/Cube",
        cube_cfg,
        translation=(0.0, 0.0, 2.0),
    )


def main() -> None:
    """Isaac Labの物理シミュレーションを実行する。"""
    # 物理シミュレーション全体の設定です。
    simulation_cfg = sim_utils.SimulationCfg(
        dt=0.01,
        device=args_cli.device,
        gravity=(0.0, 0.0, -9.80665),
    )

    simulation_context = SimulationContext(
        simulation_cfg,
    )

    # カメラの初期位置と注視点を指定します。
    simulation_context.set_camera_view(
        eye=(4.0, 4.0, 3.0),
        target=(0.0, 0.0, 1.0),
    )

    design_scene()

    # ステージ生成後にシミュレーション状態を初期化します。
    simulation_context.reset()

    print("[INFO] physical-ai-labからIsaac Labを起動しました。")
    print("[INFO] 地球重力で立方体を落下させます。")

    step_count = 0
    maximum_steps = 1000

    while simulation_app.is_running():
        simulation_context.step()

        step_count += 1

        if step_count >= maximum_steps:
            print("[INFO] 最大ステップ数へ到達しました。")
            break


if __name__ == "__main__":
    try:
        main()
    finally:
        # 正常終了でも例外終了でもIsaac Simを閉じます。
        simulation_app.close()