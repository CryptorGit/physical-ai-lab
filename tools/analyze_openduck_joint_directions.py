from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_MODULE_DIR = (
    ROOT
    / ".openduck_runtime_source_review"
    / "mini_bdx_runtime"
    / "mini_bdx_runtime"
)
import sys

sys.path.insert(0, str(RUNTIME_MODULE_DIR))

from calibrated_poses import SAFE_INIT_POS, SAFE_JOINT_LIMITS  # noqa: E402
MODEL_PATH = (
    ROOT
    / ".openduck_hardware_source_review"
    / "mini_bdx"
    / "robots"
    / "open_duck_mini_v2"
    / "scene.xml"
)
JOINTS = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    foot_body = {
        "left": mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "left_foot"
        ),
        "right": mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "right_foot"
        ),
    }
    q0_foot = {
        side: data.xpos[body_id].copy() for side, body_id in foot_body.items()
    }
    print("q=0 foot centers:", q0_foot)
    print("World displacement of foot center caused by +0.20 rad:")
    for joint_name in JOINTS:
        side = "left" if joint_name.startswith("left") else "right"
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        qpos_address = model.jnt_qposadr[joint_id]
        data.qpos[qpos_address] = 0.20
        mujoco.mj_forward(model, data)
        displacement = data.xpos[foot_body[side]] - q0_foot[side]
        data.qpos[qpos_address] = 0.0
        print(f"{joint_name:20s} {np.round(displacement, 5)}")

    mujoco.mj_resetData(model, data)
    for joint_name, position in SAFE_INIT_POS.items():
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        data.qpos[model.jnt_qposadr[joint_id]] = position
    mujoco.mj_forward(model, data)
    safe_feet = {
        side: data.xpos[body_id].copy() for side, body_id in foot_body.items()
    }
    print("safe init foot centers:", safe_feet)
    print(
        "safe init foot symmetry |dx|, |z|:",
        abs(safe_feet["left"][0] - safe_feet["right"][0]),
        abs(safe_feet["left"][2] - safe_feet["right"][2]),
    )
    print("safe init contacts:", data.ncon)
    for joint_name, position in SAFE_INIT_POS.items():
        if joint_name in SAFE_JOINT_LIMITS:
            lower, upper = SAFE_JOINT_LIMITS[joint_name]
            print(
                f"{joint_name:20s} q={position:+.3f} "
                f"limit=[{lower:+.3f}, {upper:+.3f}]"
            )


if __name__ == "__main__":
    main()
