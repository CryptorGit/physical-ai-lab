"""Unitree Go2用の地球平面歩行Gymnasium環境。

方策は12関節について-1から1の行動を出力する。
行動は初期関節角からの変位へ変換され、PD制御によって
各モーターのトルクとしてMuJoCoへ入力される。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import yaml
from gymnasium import spaces


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


class Go2FlatEnv(gym.Env[np.ndarray, np.ndarray]):
    """Unitree Go2の平面歩行学習環境。"""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        render_mode: str | None = None,
    ) -> None:
        """環境を初期化する。

        Args:
            config:
                実験設定。省略時はconfig.yamlから読み込む。
            render_mode:
                描画方式。学習時は通常Noneを使う。
        """
        super().__init__()

        self.config = config if config is not None else load_config()
        self.render_mode = render_mode

        model_path = (
            LAB_ROOT
            / Path(self.config["model"]["relative_path"])
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Go2モデルが見つかりません: {model_path}"
            )

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        self.model.opt.gravity[:] = np.asarray(
            self.config["physics"]["gravity"],
            dtype=np.float64,
        )
        self.model.opt.timestep = float(
            self.config["physics"]["timestep"]
        )

        self.frame_skip = int(
            self.config["physics"]["frame_skip"]
        )

        self.base_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.config["model"]["base_body_name"],
        )

        if self.base_body_id < 0:
            raise ValueError("base bodyが見つかりません。")

        # free jointの後に12個の脚関節が並ぶ。
        # qpos:
        #   0:3   本体位置
        #   3:7   本体姿勢クォータニオン
        #   7:19  脚関節角
        #
        # qvel:
        #   0:3   本体並進速度
        #   3:6   本体角速度
        #   6:18  脚関節速度
        self.joint_qpos_slice = slice(7, 19)
        self.joint_qvel_slice = slice(6, 18)

        self.home_qpos = self._load_home_qpos()
        self.home_joint_positions = self.home_qpos[
            self.joint_qpos_slice
        ].copy()

        self.previous_action = np.zeros(
            self.model.nu,
            dtype=np.float32,
        )

        # 方策の出力は12個の連続値。
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32,
        )

        # 観測:
        # 本体鉛直方向 3
        # 本体速度 6
        # 関節角差 12
        # 関節速度 12
        # 前回行動 12
        # 合計45
        observation_size = 3 + 6 + 12 + 12 + 12

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )

        self.max_episode_steps = int(
            float(self.config["episode"]["max_seconds"])
            / (
                self.model.opt.timestep
                * self.frame_skip
            )
        )

        self.episode_step = 0

    def _load_home_qpos(self) -> np.ndarray:
        """モデルから初期姿勢を取得する。

        Returns:
            初期姿勢として使用するqpos。

        Notes:
            指定したkeyframeが存在すれば、そのqposを使用する。
            存在しなければモデル既定値を使用する。
        """
        keyframe_name = self.config["model"].get(
            "keyframe_name",
            "",
        )

        if keyframe_name:
            key_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_KEY,
                keyframe_name,
            )

            if key_id >= 0:
                print(
                    f"Using keyframe '{keyframe_name}' "
                    f"for initial pose."
                )
                return self.model.key_qpos[key_id].copy()

        data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, data)

        print(
            "Keyframe was not found. "
            "Using the model default qpos."
        )
        return data.qpos.copy()

    def _get_observation(self) -> np.ndarray:
        """現在状態から方策へ渡す観測を作る。

        Returns:
            45次元の観測ベクトル。
        """
        # base bodyの回転行列。
        rotation = self.data.xmat[self.base_body_id].reshape(3, 3)

        # 胴体ローカル座標の上方向が、
        # ワールド座標でどちらを向いているか。
        projected_up = rotation[:, 2]

        base_velocity = self.data.qvel[:6]

        joint_position_error = (
            self.data.qpos[self.joint_qpos_slice]
            - self.home_joint_positions
        )

        joint_velocity = self.data.qvel[
            self.joint_qvel_slice
        ]

        observation = np.concatenate(
            [
                projected_up,
                base_velocity,
                joint_position_error,
                joint_velocity,
                self.previous_action,
            ]
        )

        return observation.astype(np.float32)

    def _compute_torque(
        self,
        action: np.ndarray,
    ) -> np.ndarray:
        """方策の行動からPD制御トルクを計算する。

        Args:
            action:
                -1から1の12次元行動。

        Returns:
            各モーターへ入力する12次元トルク。
        """
        position_scale = float(
            self.config["action"]["position_scale"]
        )
        kp = float(self.config["action"]["kp"])
        kd = float(self.config["action"]["kd"])

        target_joint_positions = (
            self.home_joint_positions
            + position_scale * action
        )

        current_joint_positions = self.data.qpos[
            self.joint_qpos_slice
        ]
        current_joint_velocities = self.data.qvel[
            self.joint_qvel_slice
        ]

        torque = (
            kp
            * (
                target_joint_positions
                - current_joint_positions
            )
            - kd * current_joint_velocities
        )

        control_range = self.model.actuator_ctrlrange

        return np.clip(
            torque,
            control_range[:, 0],
            control_range[:, 1],
        )

    def _compute_reward(
        self,
        action: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        """現在状態から報酬を計算する。

        Args:
            action:
                現在の方策行動。

        Returns:
            合計報酬と、各報酬成分。
        """
        reward_config = self.config["reward"]
        target_velocity = float(
            self.config["command"]["target_forward_velocity"]
        )

        rotation = self.data.xmat[self.base_body_id].reshape(3, 3)
        upright_value = float(rotation[2, 2])

        forward_velocity = float(self.data.qvel[0])
        lateral_velocity = float(self.data.qvel[1])

        angular_velocity = self.data.qvel[3:6]

        velocity_error = forward_velocity - target_velocity
        forward_reward = np.exp(-4.0 * velocity_error**2)

        alive_reward = float(reward_config["alive"])

        upright_reward = (
            float(reward_config["upright"])
            * max(0.0, upright_value)
        )

        forward_reward *= float(
            reward_config["forward_velocity"]
        )

        lateral_penalty = (
            float(reward_config["lateral_velocity_cost"])
            * lateral_velocity**2
        )

        angular_penalty = (
            float(reward_config["angular_velocity_cost"])
            * float(np.sum(np.square(angular_velocity)))
        )

        action_penalty = (
            float(reward_config["action_cost"])
            * float(np.sum(np.square(action)))
        )

        action_rate_penalty = (
            float(reward_config["action_rate_cost"])
            * float(
                np.sum(
                    np.square(
                        action - self.previous_action
                    )
                )
            )
        )

        total_reward = (
            alive_reward
            + upright_reward
            + forward_reward
            - lateral_penalty
            - angular_penalty
            - action_penalty
            - action_rate_penalty
        )

        components = {
            "alive": alive_reward,
            "upright": upright_reward,
            "forward": float(forward_reward),
            "lateral_penalty": float(lateral_penalty),
            "angular_penalty": float(angular_penalty),
            "action_penalty": float(action_penalty),
            "action_rate_penalty": float(
                action_rate_penalty
            ),
        }

        return float(total_reward), components

    def _is_fallen(self) -> bool:
        """Go2が転倒状態か判定する。

        Returns:
            転倒と判断した場合はTrue。
        """
        base_height = float(
            self.data.xpos[self.base_body_id][2]
        )

        minimum_height = float(
            self.config["episode"]["minimum_base_height"]
        )
        maximum_height = float(
            self.config["episode"]["maximum_base_height"]
        )

        rotation = self.data.xmat[self.base_body_id].reshape(3, 3)
        upright_value = float(rotation[2, 2])

        return (
            base_height < minimum_height
            or base_height > maximum_height
            or upright_value < 0.25
            or not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """環境を初期状態へ戻す。

        Args:
            seed:
                乱数シード。
            options:
                Gymnasium互換用の追加設定。

        Returns:
            初期観測と補助情報。
        """
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:] = self.home_qpos

        position_noise = float(
            self.config["episode"]["joint_position_noise"]
        )
        velocity_noise = float(
            self.config["episode"]["joint_velocity_noise"]
        )

        self.data.qpos[self.joint_qpos_slice] += (
            self.np_random.uniform(
                low=-position_noise,
                high=position_noise,
                size=12,
            )
        )

        self.data.qvel[self.joint_qvel_slice] = (
            self.np_random.uniform(
                low=-velocity_noise,
                high=velocity_noise,
                size=12,
            )
        )

        self.previous_action.fill(0.0)
        self.episode_step = 0

        # qposを手動変更した後に、派生状態を再計算する。
        mujoco.mj_forward(self.model, self.data)

        return self._get_observation(), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        """行動を適用し、環境を1制御ステップ進める。

        Args:
            action:
                -1から1の12次元行動。

        Returns:
            観測、報酬、終了、時間切れ、補助情報。
        """
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        torque = self._compute_torque(action)

        # 同じ行動をframe_skip回維持する。
        for _ in range(self.frame_skip):
            self.data.ctrl[:] = torque
            mujoco.mj_step(self.model, self.data)

        reward, reward_components = self._compute_reward(action)

        self.episode_step += 1

        terminated = self._is_fallen()
        truncated = (
            self.episode_step >= self.max_episode_steps
        )

        observation = self._get_observation()

        info: dict[str, Any] = {
            "base_height": float(
                self.data.xpos[self.base_body_id][2]
            ),
            "forward_velocity": float(self.data.qvel[0]),
            "reward_components": reward_components,
        }

        self.previous_action = action.copy()

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )