"""Headless, deterministic evaluation of the official Open Duck Mini V2 policy.

This intentionally mirrors the observation and action path in
Open_Duck_Mini_Runtime/scripts/v2_rl_walk_mujoco.py.  It does not import the
JAX training stack, so it can be used as a lightweight regression benchmark.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime

EXPERIMENT = Path(__file__).resolve().parent
WORKSPACE = EXPERIMENT.parents[2]
PLAYGROUND = WORKSPACE / ".openduck_playground_source_review"
sys.path.insert(0, str(PLAYGROUND))

from playground.common.poly_reference_motion_numpy import PolyReferenceMotion


CONTROL_DT = 0.02
SIM_DT = 0.002
DECIMATION = int(CONTROL_DT / SIM_DT)
ACTION_SCALE = 0.25
MAX_MOTOR_VELOCITY = 5.24
HEAD_COUPLED_REAR_SLOPE = 0.984
HEAD_COUPLED_REAR_INTERCEPT = 0.458


@dataclass
class EpisodeResult:
    seed: int
    command: list[float]
    requested_seconds: float
    completed_seconds: float
    completed_steps: int
    fell: bool
    displacement_xyz: list[float]
    mean_velocity_xyz: list[float]
    mean_yaw_rate: float
    minimum_height: float
    minimum_upright: float
    mean_left_contact: float
    mean_right_contact: float
    single_support_rate: float
    flight_rate: float
    action_rms: float
    action_peak: float
    joint_target_limit_violations: int
    head_target_peak: float


class OfficialPolicyEvaluator:
    def __init__(self, scene: Path, policy: Path, reference_data: Path):
        self.scene = scene.resolve()
        self.policy_path = policy.resolve()
        self.reference_data = reference_data.resolve()
        self.calibrated_hardware = "calibrated" in self.scene.stem
        self.use_backward_feedforward = True
        self.backward_phase_rate = 1.0
        self.backward_yaw_joint_gains = np.zeros(2, dtype=np.float64)
        self.backward_yaw_amplitude_gains = np.zeros(2, dtype=np.float64)
        self.backward_turn_profiles: dict[
            int, tuple[np.ndarray, np.ndarray, float]
        ] = {}
        self.backward_turn_minimum_yaw = 0.0
        self.backward_turn_minimum_blend = 0.0
        self.backward_turn_maximum_blend = 1.0
        self.backward_turn_leg_residual_floor = 0.10
        self.backward_profile_yaw_offset = 0.0
        self.policy_yaw_offset = 0.0
        self.backward_yaw_rate_feedback = np.zeros(2, dtype=np.float64)
        if self.calibrated_hardware:
            gait_path = (
                self.scene.parent.parent
                / "data"
                / "optimized_backward_gait.json"
            )
            gait = json.loads(gait_path.read_text(encoding="utf-8"))
            self.backward_phase_rate = float(
                gait["parameters"]["phase_rate"]
            )
            self.backward_residual_scale = 0.12
            self.backward_gait_scales = np.asarray(
                gait["parameters"]["joint_amplitude_scales"],
                dtype=np.float64,
            )
            self.backward_gait_biases = np.asarray(
                gait["parameters"].get("joint_bias_offsets", [0.0] * 10),
                dtype=np.float64,
            )
            calibrated_reference_path = (
                self.scene.parent.parent
                / "data"
                / "polynomial_coefficients_calibrated.pkl"
            )
            backward_reference = PolyReferenceMotion(
                str(calibrated_reference_path)
            )
            self.backward_reference_frames = np.asarray(
                [
                    backward_reference.get_reference_motion(
                        -0.1, 0.0, 0.0, index
                    )
                    for index in range(
                        backward_reference.nb_steps_in_period
                    )
                ],
                dtype=np.float64,
            )
            backward_leg_frames = np.concatenate(
                [
                    self.backward_reference_frames[:, :5],
                    self.backward_reference_frames[:, 11:16],
                ],
                axis=1,
            )
            self.backward_leg_means = backward_leg_frames.mean(axis=0)
            self.backward_leg_deviations = (
                backward_leg_frames - self.backward_leg_means
            )
            self.backward_actuator_indices = np.asarray(
                [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
            )

        self.model = mujoco.MjModel.from_xml_path(str(self.scene))
        self.model.opt.timestep = SIM_DT
        self.session = onnxruntime.InferenceSession(
            str(self.policy_path), providers=["CPUExecutionProvider"]
        )

        policy_input = self.session.get_inputs()[0]
        policy_output = self.session.get_outputs()[0]
        if policy_input.name != "obs" or policy_input.shape != [1, 101]:
            raise ValueError(f"Unexpected policy input: {policy_input.name} {policy_input.shape}")
        if policy_output.shape != [1, 14]:
            raise ValueError(f"Unexpected policy output: {policy_output.shape}")

        with self.reference_data.open("rb") as stream:
            reference = pickle.load(stream)
        first_motion = next(iter(reference.values()))
        self.phase_steps = int(first_motion["period"] * first_motion["fps"])

        self.actuator_joint_ids = self.model.actuator_trnid[:, 0].astype(int)
        self.actuator_qpos_addr = self.model.jnt_qposadr[self.actuator_joint_ids]
        self.actuator_qvel_addr = self.model.jnt_dofadr[self.actuator_joint_ids]
        floor_geom_id = self.model.geom("floor").id
        self.floor_body_id = int(self.model.geom_bodyid[floor_geom_id])
        self.left_foot_body_id = self.model.body("foot_assembly").id
        self.right_foot_body_id = self.model.body("foot_assembly_2").id
        self.trunk_body_id = self.model.body("trunk_assembly").id

    def _sensor(self, data: mujoco.MjData, name: str) -> np.ndarray:
        sensor_id = self.model.sensor(name).id
        start = int(self.model.sensor_adr[sensor_id])
        stop = start + int(self.model.sensor_dim[sensor_id])
        return np.asarray(data.sensordata[start:stop], dtype=np.float64).copy()

    def _backward_feedforward(
        self,
        phase_index: float,
        default_actuator: np.ndarray,
        joint_ranges: np.ndarray,
        bounded_action: np.ndarray,
        gait_scales: np.ndarray | None = None,
        gait_biases: np.ndarray | None = None,
        leg_residual_factor: float = 1.0,
        head_residual_factor: float = 1.0,
    ) -> np.ndarray:
        period = len(self.backward_reference_frames)
        wrapped = phase_index % period
        frame_index = int(np.floor(wrapped))
        next_index = (frame_index + 1) % period
        fraction = wrapped - np.floor(wrapped)
        scales = (
            self.backward_gait_scales
            if gait_scales is None
            else np.asarray(gait_scales, dtype=np.float64)
        )
        biases = (
            self.backward_gait_biases
            if gait_biases is None
            else np.asarray(gait_biases, dtype=np.float64)
        )
        leg_target = self.backward_leg_means + biases + scales * (
            (1.0 - fraction) * self.backward_leg_deviations[frame_index]
            + fraction * self.backward_leg_deviations[next_index]
        )
        targets = default_actuator.copy()
        targets[self.backward_actuator_indices] = leg_target
        residual_scales = np.full(
            len(bounded_action),
            self.backward_residual_scale * leg_residual_factor,
            dtype=np.float64,
        )
        residual_scales[5:9] = (
            self.backward_residual_scale * head_residual_factor
        )
        targets += residual_scales * bounded_action
        safe_lower = default_actuator + 0.9 * (
            joint_ranges[:, 0] - default_actuator
        )
        safe_upper = default_actuator + 0.9 * (
            joint_ranges[:, 1] - default_actuator
        )
        return np.clip(targets, safe_lower, safe_upper)

    def load_backward_turn_profile(self, direction: int, path: Path) -> None:
        payload = json.loads(path.resolve().read_text(encoding="utf-8"))
        parameters = payload["parameters"]
        self.backward_turn_profiles[int(np.sign(direction))] = (
            np.asarray(
                parameters["joint_amplitude_scales"], dtype=np.float64
            ),
            np.asarray(
                parameters.get("joint_bias_offsets", [0.0] * 10),
                dtype=np.float64,
            ),
            float(parameters["phase_rate"]),
        )

    def load_backward_profile(self, path: Path) -> None:
        payload = json.loads(path.resolve().read_text(encoding="utf-8"))
        parameters = payload["parameters"]
        self.backward_gait_scales = np.asarray(
            parameters["joint_amplitude_scales"], dtype=np.float64
        )
        self.backward_gait_biases = np.asarray(
            parameters.get("joint_bias_offsets", [0.0] * 10),
            dtype=np.float64,
        )
        self.backward_phase_rate = float(parameters["phase_rate"])

    def backward_parameters(
        self, yaw_command: float
    ) -> tuple[np.ndarray, np.ndarray, float]:
        direction = int(np.sign(yaw_command))
        if (
            direction == 0
            or direction not in self.backward_turn_profiles
            or abs(float(yaw_command)) < self.backward_turn_minimum_yaw
        ):
            return (
                self.backward_gait_scales,
                self.backward_gait_biases,
                self.backward_phase_rate,
            )
        turn_scales, turn_biases, turn_phase_rate = (
            self.backward_turn_profiles[direction]
        )
        blend = np.clip(
            abs(float(yaw_command)) / 0.2,
            self.backward_turn_minimum_blend,
            self.backward_turn_maximum_blend,
        )
        if float(yaw_command) < 0.0:
            blend = min(float(blend), 0.90)
        return (
            (1.0 - blend) * self.backward_gait_scales
            + blend * turn_scales,
            (1.0 - blend) * self.backward_gait_biases
            + blend * turn_biases,
            (1.0 - blend) * self.backward_phase_rate
            + blend * turn_phase_rate,
        )

    def _feet_contacts(self, data: mujoco.MjData) -> np.ndarray:
        left = False
        right = False
        for index in range(data.ncon):
            contact = data.contact[index]
            body_1 = int(self.model.geom_bodyid[contact.geom1])
            body_2 = int(self.model.geom_bodyid[contact.geom2])
            pair = {body_1, body_2}
            left |= pair == {self.floor_body_id, self.left_foot_body_id}
            right |= pair == {self.floor_body_id, self.right_foot_body_id}
        return np.array([left, right], dtype=np.float32)

    def _observation(
        self,
        data: mujoco.MjData,
        command: np.ndarray,
        default_actuator: np.ndarray,
        motor_targets: np.ndarray,
        action_history: list[np.ndarray],
        phase: float,
    ) -> np.ndarray:
        accelerometer = self._sensor(data, "accelerometer")
        accelerometer[0] += 1.3
        observation = np.concatenate(
            [
                self._sensor(data, "gyro"),
                accelerometer,
                command,
                data.qpos[self.actuator_qpos_addr] - default_actuator,
                data.qvel[self.actuator_qvel_addr] * 0.05,
                action_history[0],
                action_history[1],
                action_history[2],
                motor_targets,
                self._feet_contacts(data),
                [np.cos(phase), np.sin(phase)],
            ]
        ).astype(np.float32)
        if observation.shape != (101,):
            raise RuntimeError(f"Observation has shape {observation.shape}, expected (101,)")
        return observation

    def run_episode(
        self,
        command_xyz: tuple[float, float, float],
        seconds: float,
        seed: int = 0,
        initial_joint_noise: float = 0.0,
        initial_base_speed: float = 0.0,
        reverse_phase_for_negative_x: bool = False,
        absolute_x_observation: bool = False,
        zero_x_observation_for_reverse: bool = False,
        positive_yaw_lateral_compensation: float = 0.0,
        lock_head_targets: bool = False,
        mask_head_action_history: bool = False,
    ) -> EpisodeResult:
        rng = np.random.default_rng(seed)
        data = mujoco.MjData(self.model)
        home = self.model.keyframe("home")
        data.qpos[:] = home.qpos
        joint_ranges = self.model.jnt_range[self.actuator_joint_ids]
        if initial_joint_noise > 0.0:
            joint_noise = rng.uniform(
                -initial_joint_noise,
                initial_joint_noise,
                size=self.model.nu,
            )
            noisy_joints = data.qpos[self.actuator_qpos_addr] + joint_noise
            data.qpos[self.actuator_qpos_addr] = np.clip(
                noisy_joints,
                joint_ranges[:, 0] + 0.005,
                joint_ranges[:, 1] - 0.005,
            )
        data.ctrl[:] = data.qpos[self.actuator_qpos_addr]
        if initial_base_speed > 0.0:
            push_angle = rng.uniform(-np.pi, np.pi)
            push_magnitude = rng.uniform(0.0, initial_base_speed)
            data.qvel[:2] = push_magnitude * np.array(
                [np.cos(push_angle), np.sin(push_angle)]
            )
        mujoco.mj_forward(self.model, data)

        default_actuator = np.asarray(home.ctrl, dtype=np.float64).copy()
        motor_targets = default_actuator.copy()
        previous_targets = default_actuator.copy()
        action_history = [np.zeros(self.model.nu, dtype=np.float32) for _ in range(3)]
        policy_command_xyz = list(command_xyz)
        policy_command_xyz[2] += self.policy_yaw_offset
        if zero_x_observation_for_reverse and command_xyz[0] < -0.02:
            policy_command_xyz[0] = 0.0
        elif absolute_x_observation:
            policy_command_xyz[0] = abs(policy_command_xyz[0])
        if policy_command_xyz[2] > 0.0:
            policy_command_xyz[1] -= positive_yaw_lateral_compensation
        command = np.array(
            [*policy_command_xyz, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
        )

        start_position = data.xpos[self.trunk_body_id].copy()
        previous_position = start_position.copy()
        phase_index = 0.0
        positions: list[np.ndarray] = []
        velocities: list[np.ndarray] = []
        yaw_rates: list[float] = []
        heights: list[float] = []
        uprights: list[float] = []
        contacts: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        joint_target_limit_violations = 0
        head_target_peak = 0.0
        fell = False
        completed_control_steps = 0
        target_control_steps = int(round(seconds / CONTROL_DT))

        for sim_step in range(target_control_steps * DECIMATION):
            mujoco.mj_step(self.model, data)

            if (sim_step + 1) % DECIMATION:
                continue

            if command_xyz[0] < 0.0 and self.calibrated_hardware:
                _, _, phase_delta = self.backward_parameters(
                    command_xyz[2] + self.backward_profile_yaw_offset
                )
            elif reverse_phase_for_negative_x and command_xyz[0] < 0.0:
                phase_delta = -1.0
            else:
                phase_delta = 1.0
            phase_index = (phase_index + phase_delta) % self.phase_steps
            phase = phase_index / self.phase_steps * 2.0 * np.pi
            observation = self._observation(
                data,
                command,
                default_actuator,
                motor_targets,
                action_history,
                phase,
            )
            action = self.session.run(None, {"obs": observation[None, :]})[0][0]
            action = np.asarray(action, dtype=np.float32)
            if mask_head_action_history:
                action[5:9] = 0.0

            action_history = [
                action.copy(),
                action_history[0].copy(),
                action_history[1].copy(),
            ]
            action_for_control = action.copy()
            bounded_action = np.clip(action_for_control, -1.0, 1.0)
            positive_span = 0.9 * (joint_ranges[:, 1] - default_actuator)
            negative_span = 0.9 * (default_actuator - joint_ranges[:, 0])
            directional_span = np.where(
                bounded_action >= 0.0, positive_span, negative_span
            )
            base_span = np.minimum(ACTION_SCALE, directional_span)
            action_magnitude = np.abs(bounded_action)
            target_magnitude = (
                base_span * action_magnitude
                + (directional_span - base_span) * action_magnitude**5
            )
            motor_targets = (
                default_actuator + np.sign(bounded_action) * target_magnitude
            )
            if (
                self.calibrated_hardware
                and self.use_backward_feedforward
                and command_xyz[0] < -0.02
            ):
                backward_scales, backward_biases, _ = self.backward_parameters(
                    command_xyz[2] + self.backward_profile_yaw_offset
                )
                motor_targets = self._backward_feedforward(
                    phase_index,
                    default_actuator,
                    joint_ranges,
                    bounded_action,
                    gait_scales=backward_scales,
                    gait_biases=backward_biases,
                    leg_residual_factor=max(
                        self.backward_turn_leg_residual_floor,
                        1.0
                        - float(np.clip(abs(command_xyz[2]) / 0.2, 0.0, 1.0)),
                    ),
                    head_residual_factor=(
                        1.0
                        - 0.5
                        * float(np.clip(abs(command_xyz[2]) / 0.2, 0.0, 1.0))
                    ),
                )
                left_indices = self.backward_actuator_indices[:5]
                right_indices = self.backward_actuator_indices[5:]
                left_factor = (
                    1.0
                    + self.backward_yaw_amplitude_gains[0] * command_xyz[2]
                )
                right_factor = (
                    1.0
                    + self.backward_yaw_amplitude_gains[1] * command_xyz[2]
                )
                motor_targets[left_indices] = (
                    self.backward_leg_means[:5]
                    + left_factor
                    * (
                        motor_targets[left_indices]
                        - self.backward_leg_means[:5]
                    )
                )
                motor_targets[right_indices] = (
                    self.backward_leg_means[5:]
                    + right_factor
                    * (
                        motor_targets[right_indices]
                        - self.backward_leg_means[5:]
                    )
                )
                motor_targets[[0, 9]] += (
                    self.backward_yaw_joint_gains * command_xyz[2]
                )
                local_yaw_rate = float(
                    (data.xmat[self.trunk_body_id].reshape(3, 3).T @ data.qvel[3:6])[
                        2
                    ]
                )
                feedback_signs = np.asarray([1.0, -1.0])
                motor_targets[[0, 9]] += (
                    self.backward_yaw_rate_feedback[0]
                    * local_yaw_rate
                    * feedback_signs
                )
                motor_targets[[1, 10]] += (
                    self.backward_yaw_rate_feedback[1]
                    * local_yaw_rate
                    * feedback_signs
                )
            maximum_delta = MAX_MOTOR_VELOCITY * CONTROL_DT
            motor_targets = np.clip(
                motor_targets,
                previous_targets - maximum_delta,
                previous_targets + maximum_delta,
            )
            if self.calibrated_hardware:
                coupled_upper = (
                    HEAD_COUPLED_REAR_INTERCEPT
                    - HEAD_COUPLED_REAR_SLOPE * motor_targets[5]
                )
                if motor_targets[6] > coupled_upper:
                    joint_target_limit_violations += 1
                    motor_targets[6] = coupled_upper
                below = motor_targets < self.model.actuator_ctrlrange[:, 0]
                above = motor_targets > self.model.actuator_ctrlrange[:, 1]
                joint_target_limit_violations += int(np.count_nonzero(below | above))
                motor_targets = np.clip(
                    motor_targets,
                    self.model.actuator_ctrlrange[:, 0],
                    self.model.actuator_ctrlrange[:, 1],
                )
            # The hardware loop keeps the controller's pre-lock target for
            # the next slew-limit calculation, then sends a zero head target.
            # Preserve that slightly unusual contract when explicitly asked
            # for runtime-equivalent evaluation.
            previous_targets = motor_targets.copy()
            if lock_head_targets:
                motor_targets[5:9] = 0.0
            head_target_peak = max(
                head_target_peak, float(np.max(np.abs(motor_targets[5:9])))
            )
            data.ctrl[:] = motor_targets

            position = data.xpos[self.trunk_body_id].copy()
            rotation = data.xmat[self.trunk_body_id].reshape(3, 3)
            upright = float(rotation[2, 2])
            contacts_now = self._feet_contacts(data)

            positions.append(position)
            world_velocity = (position - previous_position) / CONTROL_DT
            velocities.append(rotation.T @ world_velocity)
            previous_position = position
            yaw_rates.append(float(self._sensor(data, "global_angvel")[2]))
            heights.append(float(position[2]))
            uprights.append(upright)
            contacts.append(contacts_now)
            actions.append(action)
            completed_control_steps += 1

            if position[2] < 0.08 or upright < 0.25:
                fell = True
                break

        final_position = (
            positions[-1] if positions else data.xpos[self.trunk_body_id].copy()
        )
        warmup = min(int(2.0 / CONTROL_DT), max(0, len(velocities) // 4))
        velocity_window = np.asarray(velocities[warmup:] or velocities)
        yaw_window = np.asarray(yaw_rates[warmup:] or yaw_rates)
        contact_array = np.asarray(contacts)
        action_array = np.asarray(actions)

        return EpisodeResult(
            seed=seed,
            command=list(command_xyz),
            requested_seconds=seconds,
            completed_seconds=completed_control_steps * CONTROL_DT,
            completed_steps=completed_control_steps,
            fell=fell,
            displacement_xyz=(final_position - start_position).tolist(),
            mean_velocity_xyz=np.mean(velocity_window, axis=0).tolist(),
            mean_yaw_rate=float(np.mean(yaw_window)),
            minimum_height=float(np.min(heights)),
            minimum_upright=float(np.min(uprights)),
            mean_left_contact=float(np.mean(contact_array[:, 0])),
            mean_right_contact=float(np.mean(contact_array[:, 1])),
            single_support_rate=float(
                np.logical_xor(contact_array[:, 0], contact_array[:, 1]).mean()
            ),
            flight_rate=float((contact_array.sum(axis=1) == 0).mean()),
            action_rms=float(np.sqrt(np.mean(np.square(action_array)))),
            action_peak=float(np.max(np.abs(action_array))),
            joint_target_limit_violations=joint_target_limit_violations,
            head_target_peak=head_target_peak,
        )


def parse_args() -> argparse.Namespace:
    experiment = Path(__file__).resolve().parent
    workspace = experiment.parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=Path,
        default=workspace
        / ".openduck_playground_source_review"
        / "playground"
        / "open_duck_mini_v2"
        / "xmls"
        / "scene_flat_terrain.xml",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=workspace
        / ".openduck_hardware_source_review"
        / "BEST_WALK_ONNX_2.onnx",
    )
    parser.add_argument(
        "--reference-data",
        type=Path,
        default=workspace
        / ".openduck_playground_source_review"
        / "playground"
        / "open_duck_mini_v2"
        / "data"
        / "polynomial_coefficients.pkl",
    )
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--initial-joint-noise", type=float, default=0.0)
    parser.add_argument("--initial-base-speed", type=float, default=0.0)
    parser.add_argument(
        "--command",
        dest="commands",
        action="append",
        nargs=3,
        type=float,
        metavar=("VX", "VY", "YAW"),
        help="Evaluate only this command; may be repeated.",
    )
    parser.add_argument("--reverse-phase-for-negative-x", action="store_true")
    parser.add_argument("--absolute-x-observation", action="store_true")
    parser.add_argument(
        "--zero-x-observation-for-reverse",
        action="store_true",
    )
    parser.add_argument(
        "--positive-yaw-lateral-compensation",
        type=float,
        default=0.06,
    )
    parser.add_argument(
        "--backward-residual-scale",
        type=float,
        default=0.0,
        help="experimental policy residual added to calibrated reverse targets",
    )
    parser.add_argument(
        "--backward-yaw-joint-gains",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("LEFT", "RIGHT"),
        help="experimental reverse hip-yaw target gains",
    )
    parser.add_argument(
        "--backward-yaw-amplitude-gains",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("LEFT", "RIGHT"),
        help="experimental reverse left/right gait-amplitude gains",
    )
    parser.add_argument(
        "--backward-yaw-rate-feedback",
        nargs=2,
        type=float,
        default=(0.0, 0.0),
        metavar=("HIP_YAW", "HIP_ROLL"),
    )
    parser.add_argument("--backward-gait", type=Path)
    parser.add_argument(
        "--backward-amplitude-scale",
        type=float,
        default=1.0,
        help="Experimental scalar applied to all straight-reverse gait amplitudes.",
    )
    parser.add_argument(
        "--backward-yaw-roll-bias-scale",
        type=float,
        default=1.0,
        help="Scale the four hip yaw/roll center offsets in a reverse profile.",
    )
    parser.add_argument(
        "--backward-yaw-bias-deltas",
        nargs=2,
        type=float,
        default=(0.0, 0.0),
        metavar=("LEFT", "RIGHT"),
        help="Add static left/right hip-yaw center corrections.",
    )
    parser.add_argument(
        "--backward-roll-bias-deltas",
        nargs=2,
        type=float,
        default=(0.0, 0.0),
        metavar=("LEFT", "RIGHT"),
        help="Add static left/right hip-roll center corrections.",
    )
    parser.add_argument("--backward-left-turn-gait", type=Path)
    parser.add_argument("--backward-right-turn-gait", type=Path)
    parser.add_argument("--backward-turn-minimum-yaw", type=float, default=0.0)
    parser.add_argument("--backward-turn-minimum-blend", type=float, default=0.0)
    parser.add_argument("--backward-turn-maximum-blend", type=float, default=1.0)
    parser.add_argument(
        "--backward-turn-leg-residual-floor",
        type=float,
        default=0.10,
        help="Minimum turn leg residual as a fraction of backward residual scale.",
    )
    parser.add_argument("--backward-profile-yaw-offset", type=float, default=0.0)
    parser.add_argument("--policy-yaw-offset", type=float, default=0.0)
    parser.add_argument(
        "--learned-reverse-policy",
        action="store_true",
        help="Use policy targets for reverse instead of periodic feedforward.",
    )
    parser.add_argument(
        "--lock-head-targets",
        action="store_true",
        help="Mirror the hardware loop by commanding all four head targets to zero.",
    )
    parser.add_argument(
        "--mask-head-action-history",
        action="store_true",
        help="Mask action indices 5:9 before storing policy action history.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=experiment / "artifacts" / "official_policy_benchmark.json",
    )
    return parser.parse_args()


def summarize_command(
    command: tuple[float, float, float], episodes: list[EpisodeResult]
) -> dict:
    command_array = np.asarray(command)
    velocities = np.asarray([episode.mean_velocity_xyz for episode in episodes])
    yaw_rates = np.asarray([episode.mean_yaw_rate for episode in episodes])
    displacements = np.asarray([episode.displacement_xyz for episode in episodes])

    linear_command = command_array[:2]
    linear_speed = float(np.linalg.norm(linear_command))
    if linear_speed > 0.0:
        linear_direction = linear_command / linear_speed
        projected_velocity = velocities[:, :2] @ linear_direction
        primary_velocity_error = np.abs(projected_velocity - linear_speed)
        orthogonal_direction = np.asarray(
            [-linear_direction[1], linear_direction[0]]
        )
        orthogonal_velocity = np.abs(
            velocities[:, :2] @ orthogonal_direction
        )
    else:
        primary_velocity_error = np.zeros(len(episodes))
        orthogonal_velocity = (
            np.linalg.norm(velocities[:, :2], axis=1)
            if abs(command[2]) > 0
            else np.zeros(len(episodes))
        )
    yaw_error = (
        np.abs(yaw_rates - command[2])
        if abs(command[2]) > 0
        else np.zeros(len(episodes))
    )
    stop_drift = (
        np.linalg.norm(displacements[:, :2], axis=1)
        if np.allclose(command_array, 0.0)
        else np.zeros(len(episodes))
    )
    return {
        "command": list(command),
        "episode_count": len(episodes),
        "fall_rate": float(np.mean([episode.fell for episode in episodes])),
        "minimum_upright": float(
            np.min([episode.minimum_upright for episode in episodes])
        ),
        "minimum_height": float(
            np.min([episode.minimum_height for episode in episodes])
        ),
        "mean_velocity_xyz": np.mean(velocities, axis=0).tolist(),
        "mean_yaw_rate": float(np.mean(yaw_rates)),
        "maximum_primary_velocity_error": float(np.max(primary_velocity_error)),
        "maximum_orthogonal_velocity": float(np.max(orthogonal_velocity)),
        "maximum_yaw_rate_error": float(np.max(yaw_error)),
        "maximum_stop_drift": float(np.max(stop_drift)),
        "joint_target_limit_violations": int(
            sum(episode.joint_target_limit_violations for episode in episodes)
        ),
        "head_target_peak": float(
            max(episode.head_target_peak for episode in episodes)
        ),
        "minimum_single_support_rate": float(
            min(episode.single_support_rate for episode in episodes)
        ),
        "maximum_flight_rate": float(
            max(episode.flight_rate for episode in episodes)
        ),
    }


def evaluate_acceptance(summaries: list[dict], criteria_path: Path) -> dict:
    criteria_payload = json.loads(criteria_path.read_text(encoding="utf-8"))
    required = criteria_payload["simulation"]["required"]
    required_episode_count = int(criteria_payload["simulation"]["seeds_per_command"])
    checks = []
    for summary in summaries:
        command = summary["command"]
        moving = not np.allclose(command, 0.0)
        checks.append(
            {
                "command": command,
                "enough_episodes": summary["episode_count"] >= required_episode_count,
                "fall_rate": summary["fall_rate"] <= required["fall_rate"],
                "minimum_upright": (
                    summary["minimum_upright"] >= required["minimum_upright"]
                ),
                "minimum_height": (
                    summary["minimum_height"]
                    >= required["minimum_base_height_m"]
                ),
                "primary_velocity_error": (
                    summary["maximum_primary_velocity_error"]
                    <= required["maximum_primary_velocity_error_mps"]
                ),
                "orthogonal_velocity": (
                    summary["maximum_orthogonal_velocity"]
                    <= required["maximum_orthogonal_velocity_mps"]
                ),
                "yaw_rate_error": (
                    summary["maximum_yaw_rate_error"]
                    <= required["maximum_yaw_rate_error_radps"]
                ),
                "stop_drift": (
                    summary["maximum_stop_drift"]
                    <= required["maximum_stop_drift_m"]
                ),
                "joint_limits": (
                    summary["joint_target_limit_violations"]
                    <= required["joint_target_limit_violations"]
                ),
                "head_locked": (
                    summary["head_target_peak"]
                    <= required["head_target_peak_rad"]
                ),
                "genuine_single_support": (
                    not moving
                    or summary["minimum_single_support_rate"]
                    >= required["minimum_single_support_rate"]
                ),
                "bounded_flight": (
                    not moving
                    or summary["maximum_flight_rate"]
                    <= required["maximum_flight_rate"]
                ),
            }
        )
    return {
        "criteria": str(criteria_path.resolve()),
        "checks": checks,
        "passed": all(
            all(value for key, value in check.items() if key != "command")
            for check in checks
        ),
    }


def main() -> None:
    args = parse_args()
    evaluator = OfficialPolicyEvaluator(args.scene, args.policy, args.reference_data)
    evaluator.use_backward_feedforward = not args.learned_reverse_policy
    if evaluator.calibrated_hardware:
        evaluator.backward_residual_scale = args.backward_residual_scale
        evaluator.backward_yaw_joint_gains = np.asarray(
            args.backward_yaw_joint_gains, dtype=np.float64
        )
        evaluator.backward_yaw_amplitude_gains = np.asarray(
            args.backward_yaw_amplitude_gains, dtype=np.float64
        )
        evaluator.backward_yaw_rate_feedback = np.asarray(
            args.backward_yaw_rate_feedback, dtype=np.float64
        )
        evaluator.backward_turn_minimum_yaw = args.backward_turn_minimum_yaw
        evaluator.backward_turn_minimum_blend = args.backward_turn_minimum_blend
        evaluator.backward_turn_maximum_blend = args.backward_turn_maximum_blend
        evaluator.backward_turn_leg_residual_floor = (
            args.backward_turn_leg_residual_floor
        )
        evaluator.backward_profile_yaw_offset = args.backward_profile_yaw_offset
        evaluator.policy_yaw_offset = args.policy_yaw_offset
        if args.backward_gait:
            evaluator.load_backward_profile(args.backward_gait)
        if args.backward_amplitude_scale <= 0.0:
            raise ValueError("--backward-amplitude-scale must be positive")
        evaluator.backward_gait_scales = (
            evaluator.backward_gait_scales * args.backward_amplitude_scale
        )
        evaluator.backward_gait_biases[
            np.asarray([0, 1, 5, 6])
        ] *= args.backward_yaw_roll_bias_scale
        evaluator.backward_gait_biases[
            np.asarray([0, 5])
        ] += np.asarray(args.backward_yaw_bias_deltas, dtype=np.float64)
        evaluator.backward_gait_biases[
            np.asarray([1, 6])
        ] += np.asarray(args.backward_roll_bias_deltas, dtype=np.float64)
        if args.backward_left_turn_gait:
            evaluator.load_backward_turn_profile(
                1, args.backward_left_turn_gait
            )
        if args.backward_right_turn_gait:
            evaluator.load_backward_turn_profile(
                -1, args.backward_right_turn_gait
            )
    commands = (
        [tuple(command) for command in args.commands]
        if args.commands
        else [
            (0.0, 0.0, 0.0),
            (0.10, 0.0, 0.0),
            (-0.10, 0.0, 0.0),
            (0.0, 0.10, 0.0),
            (0.0, -0.10, 0.0),
            (0.0, 0.0, 0.6),
            (0.0, 0.0, -0.6),
        ]
    )
    grouped_results = {
        command: [
            evaluator.run_episode(
                command,
                args.seconds,
                seed=seed,
                initial_joint_noise=args.initial_joint_noise,
                initial_base_speed=args.initial_base_speed,
                reverse_phase_for_negative_x=args.reverse_phase_for_negative_x,
                absolute_x_observation=args.absolute_x_observation,
                zero_x_observation_for_reverse=(
                    args.zero_x_observation_for_reverse
                ),
                positive_yaw_lateral_compensation=(
                    args.positive_yaw_lateral_compensation
                ),
                lock_head_targets=args.lock_head_targets,
                mask_head_action_history=args.mask_head_action_history,
            )
            for seed in range(args.episodes)
        ]
        for command in commands
    }
    results = [
        episode
        for command_episodes in grouped_results.values()
        for episode in command_episodes
    ]
    summaries = [
        summarize_command(command, grouped_results[command]) for command in commands
    ]
    criteria_path = Path(__file__).resolve().parent / "acceptance_criteria.json"
    payload = {
        "policy": str(args.policy.resolve()),
        "scene": str(args.scene.resolve()),
        "control_dt": CONTROL_DT,
        "sim_dt": SIM_DT,
        "action_scale": ACTION_SCALE,
        "episodes_per_command": args.episodes,
        "initial_joint_noise": args.initial_joint_noise,
        "initial_base_speed": args.initial_base_speed,
        "lock_head_targets": args.lock_head_targets,
        "mask_head_action_history": args.mask_head_action_history,
        "backward_amplitude_scale": args.backward_amplitude_scale,
        "results": [asdict(result) for result in results],
        "summaries": summaries,
        "acceptance": evaluate_acceptance(summaries, criteria_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
