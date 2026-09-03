"""Unitree Go2のMuJoCoモデル構造を確認する。

学習環境を作る前に、モデルの自由度、状態ベクトル、
body、joint、actuatorの名前と個数を表示する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import yaml


EXPERIMENT_DIR = Path(__file__).resolve().parent
LAB_ROOT = EXPERIMENT_DIR.parents[2]
CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"


def load_config() -> dict[str, Any]:
    """YAMLから実験設定を読み込む。

    Returns:
        実験設定を格納した辞書。

    Raises:
        ValueError:
            YAMLファイルが空の場合。
    """
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"設定ファイルが空です: {CONFIG_PATH}")

    return config


def get_name(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    object_id: int,
) -> str:
    """MuJoCo内部IDからオブジェクト名を取得する。

    Args:
        model:
            読み込み済みMuJoCoモデル。
        object_type:
            body、joint、actuatorなどのオブジェクト種別。
        object_id:
            MuJoCo内部の数値ID。

    Returns:
        オブジェクト名。名前がない場合は識別用文字列。
    """
    name = mujoco.mj_id2name(model, object_type, object_id)
    return name if name is not None else f"<unnamed:{object_id}>"


def main() -> None:
    """Go2モデルを読み込み、構成情報を表示する。"""
    config = load_config()

    relative_path = Path(config["model"]["relative_path"])
    model_path = LAB_ROOT / relative_path

    if not model_path.exists():
        raise FileNotFoundError(
            f"Go2モデルが見つかりません: {model_path}"
        )

    model = mujoco.MjModel.from_xml_path(str(model_path))

    gravity = config["physics"]["gravity"]
    model.opt.gravity[:] = gravity
    model.opt.timestep = float(config["physics"]["timestep"])

    print(f"Model path: {model_path}")
    print()
    print("=== Dimensions ===")
    print(f"nq   : {model.nq}  # 位置状態の次元")
    print(f"nv   : {model.nv}  # 速度状態の次元")
    print(f"nu   : {model.nu}  # 制御入力の次元")
    print(f"nbody: {model.nbody}")
    print(f"njnt : {model.njnt}")
    print()

    print("=== Bodies ===")
    for body_id in range(model.nbody):
        print(
            f"{body_id:2d}: "
            f"{get_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)}"
        )

    print()
    print("=== Joints ===")
    for joint_id in range(model.njnt):
        print(
            f"{joint_id:2d}: "
            f"{get_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)}"
        )

    print()
    print("=== Actuators ===")
    for actuator_id in range(model.nu):
        name = get_name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )

        ctrl_min, ctrl_max = model.actuator_ctrlrange[actuator_id]

        print(
            f"{actuator_id:2d}: {name:10s} "
            f"range=[{ctrl_min:.3f}, {ctrl_max:.3f}]"
        )


if __name__ == "__main__":
    main()