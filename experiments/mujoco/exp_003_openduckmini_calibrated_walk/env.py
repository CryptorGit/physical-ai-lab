from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


ROOT = Path(__file__).resolve().parents[3]
RUNTIME_MODULE_DIR = (
    ROOT
    / ".openduck_runtime_source_review"
    / "mini_bdx_runtime"
    / "mini_bdx_runtime"
)
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

LEG_JOINTS = (
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
)

UPPER_JOINTS = (
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "left_antenna",
    "right_antenna",
)

REFERENCE_PERIOD = 68.7896474578


class OpenDuckCalibratedWalkEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        *,
        seed: int | None = None,
        episode_steps: int = 600,
        command_velocity: float = 0.10,
        reference_residual: bool = False,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.model.opt.timestep = 0.002
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 10
        self.episode_steps = episode_steps
        self.command_velocity = command_velocity
        self.reference_residual = reference_residual

        self.joint_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
                for name in LEG_JOINTS
            ],
            dtype=int,
        )
        self.qpos_addresses = self.model.jnt_qposadr[self.joint_ids]
        self.dof_addresses = self.model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
                )
                for name in LEG_JOINTS
            ],
            dtype=int,
        )
        self.upper_joint_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
                for name in UPPER_JOINTS
            ],
            dtype=int,
        )
        self.upper_qpos_addresses = self.model.jnt_qposadr[
            self.upper_joint_ids
        ]
        self.upper_dof_addresses = self.model.jnt_dofadr[
            self.upper_joint_ids
        ]
        self.upper_actuator_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
                )
                for name in UPPER_JOINTS
            ],
            dtype=int,
        )
        self.base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )

        self.init_pos = np.array(
            [SAFE_INIT_POS[name] for name in LEG_JOINTS], dtype=np.float64
        )
        self.lower_limits = np.array(
            [SAFE_JOINT_LIMITS[name][0] for name in LEG_JOINTS],
            dtype=np.float64,
        )
        self.upper_limits = np.array(
            [SAFE_JOINT_LIMITS[name][1] for name in LEG_JOINTS],
            dtype=np.float64,
        )
        self.action_scale = np.array(
            [0.10, 0.08, 0.12, 0.14, 0.10] * 2, dtype=np.float64
        )
        self.kp = np.full(len(LEG_JOINTS), 6.55)
        self.kd = np.full(len(LEG_JOINTS), 0.65)
        self.torque_limit = 3.57

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(LEG_JOINTS),), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(44,), dtype=np.float32
        )
        self.previous_action = np.zeros(len(LEG_JOINTS), dtype=np.float64)
        self.command = np.zeros(3, dtype=np.float64)
        self.step_count = 0
        self.np_random = np.random.default_rng(seed)

    def _joint_position(self) -> np.ndarray:
        return self.data.qpos[self.qpos_addresses].copy()

    def _joint_velocity(self) -> np.ndarray:
        return self.data.qvel[self.dof_addresses].copy()

    def _base_rotation(self) -> np.ndarray:
        return self.data.xmat[self.base_body_id].reshape(3, 3)

    def _observation(self) -> np.ndarray:
        joint_position = self._joint_position()
        joint_velocity = self._joint_velocity()
        base_velocity = self.data.cvel[self.base_body_id]
        projected_gravity = self._base_rotation().T @ np.array([0.0, 0.0, -1.0])
        gait_step = max(0, self.step_count - 50)
        phase = 2.0 * np.pi * gait_step / REFERENCE_PERIOD
        observation = np.concatenate(
            (
                joint_position - self.init_pos,
                joint_velocity * 0.05,
                base_velocity[:3] * 0.25,
                base_velocity[3:] * 0.5,
                projected_gravity,
                self.command,
                self.previous_action,
                np.array([np.sin(phase), np.cos(phase)]),
            )
        )
        return observation.astype(np.float32)

    def _reference_action(self) -> np.ndarray:
        if self.step_count < 50:
            return np.zeros(10, dtype=np.float64)
        gait_step = max(0, self.step_count - 50)
        phase = 2.0 * np.pi * gait_step / REFERENCE_PERIOD
        sine = np.sin(phase)
        hip = -0.9527966407 * sine + 0.4019710422 * np.sin(2.0 * phase)
        roll = -0.1236738554 * np.sin(phase - 1.6407716249)
        left_swing = 0.5 + 0.5 * sine
        right_swing = 0.5 - 0.5 * sine
        action = np.zeros(10, dtype=np.float64)
        action[1] = roll
        action[2] = hip
        action[3] = 0.0958191426 + 0.5607778798 * left_swing
        action[4] = -0.7898836137 * sine
        action[6] = roll
        action[7] = -hip
        action[8] = 0.0958191426 + 0.5607778798 * right_swing
        action[9] = 0.7898836137 * sine
        return np.clip(action, -1.0, 1.0)

    def _apply_pd(self, target: np.ndarray) -> None:
        torque = (
            self.kp * (target - self._joint_position())
            - self.kd * self._joint_velocity()
        )
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.actuator_ids] = np.clip(
            torque, -self.torque_limit, self.torque_limit
        )
        upper_torque = (
            -4.0 * self.data.qpos[self.upper_qpos_addresses]
            - 0.4 * self.data.qvel[self.upper_dof_addresses]
        )
        self.data.ctrl[self.upper_actuator_ids] = np.clip(
            upper_torque, -1.5, 1.5
        )

    def _reward(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        rotation = self._base_rotation()
        upright = float(np.clip(rotation[2, 2], -1.0, 1.0))
        body_velocity = rotation.T @ self.data.cvel[self.base_body_id, 3:]
        yaw_rate = self.data.cvel[self.base_body_id, 2]
        velocity_error = body_velocity[0] - self.command[0]

        forward_tracking = float(np.exp(-200.0 * velocity_error**2))
        forward_speed = float(np.clip(body_velocity[0], -0.2, 0.3))
        lateral_tracking = float(np.exp(-30.0 * body_velocity[1] ** 2))
        yaw_tracking = float(np.exp(-10.0 * (yaw_rate - self.command[2]) ** 2))
        upright_reward = max(0.0, upright) ** 2
        height_reward = float(
            np.exp(-120.0 * (self.data.qpos[2] - 0.17) ** 2)
        )
        action_rate = float(np.mean((action - self.previous_action) ** 2))
        torque_cost = float(
            np.mean((self.data.ctrl[self.actuator_ids] / self.torque_limit) ** 2)
        )
        action_size = float(np.mean(action**2))

        terms = {
            "forward": forward_tracking,
            "forward_speed": forward_speed,
            "lateral": lateral_tracking,
            "yaw": yaw_tracking,
            "upright": upright_reward,
            "height": height_reward,
            "action_rate": action_rate,
            "action_size": action_size,
            "torque": torque_cost,
        }
        reward = (
            6.0 * forward_tracking
            + 40.0 * forward_speed
            + 0.3 * lateral_tracking
            + 0.2 * yaw_tracking
            + 0.7 * upright_reward
            + 0.2 * height_reward
            - 0.04 * action_rate
            - 0.03 * action_size
            - 0.03 * torque_cost
        )
        return float(reward), terms

    def _terminated(self) -> bool:
        upright = self._base_rotation()[2, 2]
        base_height = self.data.qpos[2]
        return bool(base_height < 0.08 or base_height > 0.28 or upright < 0.45)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = self.model.qpos0[:7]
        noise = self.np_random.uniform(-0.01, 0.01, len(LEG_JOINTS))
        self.data.qpos[self.qpos_addresses] = np.clip(
            self.init_pos + noise, self.lower_limits, self.upper_limits
        )
        self.data.qvel[:] = 0.0
        self.command[:] = (self.command_velocity, 0.0, 0.0)
        self.previous_action[:] = 0.0
        self.step_count = 0
        self._apply_pd(self.init_pos)
        mujoco.mj_forward(self.model, self.data)
        return self._observation(), {}

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float64),
            self.action_space.low,
            self.action_space.high,
        )
        reference_action = (
            self._reference_action()
            if self.reference_residual
            else np.zeros_like(action)
        )
        effective_action = (
            np.clip(reference_action + 0.35 * action, -1.0, 1.0)
            if self.reference_residual
            else action
        )
        target = np.clip(
            self.init_pos + self.action_scale * effective_action,
            self.lower_limits,
            self.upper_limits,
        )
        for _ in range(self.frame_skip):
            self._apply_pd(target)
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        reward, reward_terms = self._reward(effective_action)
        if self.reference_residual:
            reference_deviation = float(
                np.mean((effective_action - reference_action) ** 2)
            )
            reward_terms["reference_deviation"] = reference_deviation
            reward -= 2.0 * reference_deviation
        terminated = self._terminated()
        if terminated:
            reward -= 50.0
        truncated = self.step_count >= self.episode_steps
        self.previous_action = effective_action.copy()
        info = {
            "reward_terms": reward_terms,
            "base_height": float(self.data.qpos[2]),
            "command_vx": float(self.command[0]),
        }
        return self._observation(), reward, terminated, truncated, info

    def close(self):
        pass
