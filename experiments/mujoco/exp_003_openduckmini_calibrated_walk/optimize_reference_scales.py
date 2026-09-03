"""Search measured-range periodic targets that produce stable backward gait."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution


EXPERIMENT = Path(__file__).resolve().parent
WORKSPACE = EXPERIMENT.parents[2]
PLAYGROUND = WORKSPACE / ".openduck_playground_source_review"
sys.path.insert(0, str(PLAYGROUND))

from playground.common.poly_reference_motion_numpy import PolyReferenceMotion  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--maxiter", type=int, default=25)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--velocity-weight", type=float, default=1500.0)
    parser.add_argument("--lateral-weight", type=float, default=300.0)
    parser.add_argument("--yaw-weight", type=float, default=300.0)
    parser.add_argument("--roll-pitch-weight", type=float, default=20.0)
    parser.add_argument("--upright-weight", type=float, default=15.0)
    parser.add_argument("--fall-penalty", type=float, default=50.0)
    parser.add_argument("--target-vx", type=float, default=-0.1)
    parser.add_argument("--target-yaw", type=float, default=0.0)
    parser.add_argument("--max-scale", type=float, default=4.0)
    parser.add_argument(
        "--max-bias",
        type=float,
        default=0.0,
        help="Optimize per-joint center offsets in addition to amplitudes.",
    )
    parser.add_argument("--max-phase-rate", type=float, default=3.0)
    parser.add_argument("--max-motor-velocity", type=float, default=5.24)
    parser.add_argument("--initial-joint-noise", type=float, default=0.0)
    parser.add_argument("--initial-base-speed", type=float, default=0.0)
    parser.add_argument(
        "--scene",
        type=Path,
        help="Optional calibrated scene override (defaults to exp_003 source).",
    )
    parser.add_argument(
        "--reference-data",
        type=Path,
        help="Optional calibrated reference override (defaults to exp_003 source).",
    )
    parser.add_argument(
        "--initial-gait",
        type=Path,
        help="Seed the search with parameters from an earlier gait JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT / "artifacts/optimized_backward_gait.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene = (
        args.scene.resolve()
        if args.scene
        else PLAYGROUND
        / "playground/open_duck_mini_v2/xmls/"
        "scene_flat_terrain_backlash_calibrated.xml"
    )
    reference_path = (
        args.reference_data.resolve()
        if args.reference_data
        else PLAYGROUND
        / "playground/open_duck_mini_v2/data/"
        "polynomial_coefficients_calibrated.pkl"
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = 0.002
    home = model.keyframe("home")
    home_ctrl = np.asarray(home.ctrl, dtype=np.float64).copy()
    lower = home_ctrl + 0.9 * (model.actuator_ctrlrange[:, 0] - home_ctrl)
    upper = home_ctrl + 0.9 * (model.actuator_ctrlrange[:, 1] - home_ctrl)

    reference = PolyReferenceMotion(str(reference_path))
    frames = np.asarray(
        [
            reference.get_reference_motion(-0.1, 0.0, 0.0, index)
            for index in range(reference.nb_steps_in_period)
        ],
        dtype=np.float64,
    )
    leg_frames = np.concatenate([frames[:, :5], frames[:, 11:16]], axis=1)
    means = leg_frames.mean(axis=0)
    deviations = leg_frames - means

    floor_body = int(model.geom_bodyid[model.geom("floor").id])
    left_body = model.body("foot_assembly").id
    right_body = model.body("foot_assembly_2").id
    trunk_body = model.body("trunk_assembly").id
    leg_actuators = np.array([0, 1, 2, 3, 4, 9, 10, 11, 12, 13])
    actuator_joint_ids = model.actuator_trnid[:, 0].astype(int)
    actuator_qpos_addr = model.jnt_qposadr[actuator_joint_ids]
    joint_ranges = model.jnt_range[actuator_joint_ids]
    perturbation_rng = np.random.default_rng(args.seed + 991)
    initial_noise = perturbation_rng.uniform(
        -args.initial_joint_noise,
        args.initial_joint_noise,
        size=model.nu,
    )
    push_angle = perturbation_rng.uniform(-np.pi, np.pi)
    initial_velocity = args.initial_base_speed * np.asarray(
        [np.cos(push_angle), np.sin(push_angle)]
    )
    cache: dict[tuple[float, ...], tuple[float, dict]] = {}

    def evaluate(candidate: np.ndarray) -> tuple[float, dict]:
        key = tuple(np.round(candidate, 8))
        if key in cache:
            return cache[key]
        scales = candidate[:10]
        if args.max_bias > 0.0:
            biases = candidate[10:20]
            phase_rate = candidate[20]
        else:
            biases = np.zeros(10, dtype=np.float64)
            phase_rate = candidate[10]
        data = mujoco.MjData(model)
        data.qpos[:] = home.qpos
        data.qpos[actuator_qpos_addr] = np.clip(
            data.qpos[actuator_qpos_addr] + initial_noise,
            joint_ranges[:, 0] + 0.005,
            joint_ranges[:, 1] - 0.005,
        )
        data.qvel[:2] = initial_velocity
        data.ctrl[:] = home_ctrl
        mujoco.mj_forward(model, data)
        start = data.xpos[trunk_body].copy()
        previous_position = start.copy()
        previous_targets = home_ctrl.copy()
        max_delta = args.max_motor_velocity * 0.02
        contacts = []
        uprights = []
        heights = []
        local_velocities = []
        local_angular_velocities = []
        phase = 0.0
        fell = False
        control_steps = int(round(args.seconds / 0.02))

        for _ in range(control_steps):
            frame_index = int(np.floor(phase)) % len(frames)
            next_index = (frame_index + 1) % len(frames)
            fraction = phase - np.floor(phase)
            leg_target = means + biases + scales * (
                (1.0 - fraction) * deviations[frame_index]
                + fraction * deviations[next_index]
            )
            targets = home_ctrl.copy()
            targets[leg_actuators] = leg_target
            targets[5:9] = 0.0
            targets = np.clip(
                targets,
                previous_targets - max_delta,
                previous_targets + max_delta,
            )
            targets = np.clip(targets, lower, upper)
            data.ctrl[:] = targets
            previous_targets = targets
            for _ in range(10):
                mujoco.mj_step(model, data)
            phase = (phase + phase_rate) % len(frames)

            left = False
            right = False
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                pair = {
                    int(model.geom_bodyid[contact.geom1]),
                    int(model.geom_bodyid[contact.geom2]),
                }
                left |= pair == {floor_body, left_body}
                right |= pair == {floor_body, right_body}
            contacts.append([left, right])
            height = float(data.xpos[trunk_body, 2])
            upright = float(data.xmat[trunk_body].reshape(3, 3)[2, 2])
            heights.append(height)
            uprights.append(upright)
            rotation = data.xmat[trunk_body].reshape(3, 3)
            position = data.xpos[trunk_body].copy()
            local_velocities.append(
                rotation.T @ ((position - previous_position) / 0.02)
            )
            previous_position = position
            local_angular_velocities.append(rotation.T @ data.qvel[3:6])
            if upright < 0.65 or height < 0.12:
                fell = True
                break

        elapsed = len(contacts) * 0.02
        displacement = data.xpos[trunk_body].copy() - start
        velocity = displacement / max(elapsed, 1e-9)
        contact_array = np.asarray(contacts, dtype=np.float64)
        local_velocity_array = np.asarray(local_velocities)
        local_angular_velocity_array = np.asarray(local_angular_velocities)
        steady_start = len(local_velocity_array) // 5
        steady_local_velocity = local_velocity_array[steady_start:].mean(axis=0)
        steady_local_angular_velocity = local_angular_velocity_array[
            steady_start:
        ].mean(axis=0)
        single_support = float(
            np.logical_xor(contact_array[:, 0], contact_array[:, 1]).mean()
        )
        cost = (
            args.velocity_weight
            * (steady_local_velocity[0] - args.target_vx) ** 2
            + args.lateral_weight * steady_local_velocity[1] ** 2
            + args.yaw_weight
            * (steady_local_angular_velocity[2] - args.target_yaw) ** 2
            + args.roll_pitch_weight
            * float(
                np.sum(np.square(steady_local_angular_velocity[:2]))
            )
            + args.upright_weight * (1.0 - min(uprights))
            + args.fall_penalty * float(fell)
            + 2.0 * max(0.0, 0.2 - single_support)
        )
        metrics = {
            "cost": float(cost),
            "elapsed_seconds": elapsed,
            "fell": fell,
            "velocity_xyz": velocity.tolist(),
            "steady_mean_local_velocity_xyz": steady_local_velocity.tolist(),
            "steady_mean_local_angular_velocity_xyz": (
                steady_local_angular_velocity.tolist()
            ),
            "minimum_upright": min(uprights),
            "minimum_height": min(heights),
            "left_contact_rate": float(contact_array[:, 0].mean()),
            "right_contact_rate": float(contact_array[:, 1].mean()),
            "single_support_rate": single_support,
        }
        cache[key] = (float(cost), metrics)
        return cache[key]

    generation = 0

    def objective(candidate: np.ndarray) -> float:
        return evaluate(candidate)[0]

    def callback(intermediate_result) -> None:
        nonlocal generation
        generation += 1
        _, metrics = evaluate(intermediate_result.x)
        print(
            f"generation={generation} cost={metrics['cost']:.4f} "
            f"vx={metrics['steady_mean_local_velocity_xyz'][0]:.4f} "
            f"wz={metrics['steady_mean_local_angular_velocity_xyz'][2]:.4f} "
            f"fell={metrics['fell']} "
            f"single={metrics['single_support_rate']:.3f}",
            flush=True,
        )

    initial = None
    if args.initial_gait:
        payload = json.loads(args.initial_gait.resolve().read_text(encoding="utf-8"))
        parameters = payload["parameters"]
        initial_values = list(parameters["joint_amplitude_scales"])
        if args.max_bias > 0.0:
            initial_values += list(parameters.get("joint_bias_offsets", [0.0] * 10))
        initial_values += [parameters["phase_rate"]]
        initial = np.asarray(initial_values, dtype=np.float64)

    bounds = [(0.0, args.max_scale)] * 10
    if args.max_bias > 0.0:
        bounds += [(-args.max_bias, args.max_bias)] * 10
    bounds += [(0.5, args.max_phase_rate)]
    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.seed,
        workers=1,
        updating="immediate",
        polish=False,
        callback=callback,
        tol=1e-4,
        x0=initial,
    )
    _, best_metrics = evaluate(result.x)
    parameter_payload = {
        "joint_amplitude_scales": result.x[:10].tolist(),
        "phase_rate": float(result.x[-1]),
    }
    if args.max_bias > 0.0:
        parameter_payload["joint_bias_offsets"] = result.x[10:20].tolist()
    payload = {
        "scene": str(scene),
        "reference_data": str(reference_path),
        "success": bool(result.success),
        "message": str(result.message),
        "evaluations": int(result.nfev),
        "parameters": parameter_payload,
        "metrics": best_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
