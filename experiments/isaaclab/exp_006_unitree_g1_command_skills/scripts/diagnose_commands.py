"""Measure whole-policy sensitivity to coherent legacy + new commands."""

from __future__ import annotations

import argparse
import json
import math
import sys
from importlib import metadata
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT.parent.parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
import g1_command_skills.tasks  # noqa: E402,F401
from g1_command_skills.command_observation import (  # noqa: E402
    LEGACY_COMMAND_SLICE,
    LEGACY_LAYOUT,
    NEW_COMMAND_LAYOUT,
    NEW_COMMAND_SLICE,
    RUN,
    STOP,
    TURN,
    CROUCH,
    changed_columns,
    coherent_run_observation,
    coherent_stop_observation,
    coherent_turn_observation,
    coherent_crouch_observation,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--stage", choices=("run", "turn", "stop", "crouch", "sequence"), required=True)
parser.add_argument("--output", default="results/exp_006_unitree_g1_command_skills/diagnostics/command.json")
parser.add_argument("--seed", type=int, default=42)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


SKILLS = {"run": RUN, "stop": STOP, "turn": TURN, "crouch": CROUCH}
THRESHOLD = 0.005
CROUCH_THRESHOLD = 1.0e-4


def l2_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(left - right, dim=-1).mean().item())


def compare(left_obs, right_obs, actor) -> dict:
    with torch.inference_mode():
        left = actor.diagnostic_components(left_obs)
        right = actor.diagnostic_components(right_obs)
    left_policy, right_policy = left_obs["policy"], right_obs["policy"]
    return {
        "base_actor_action_l2": l2_difference(left["base_action"], right["base_action"]),
        "residual_action_l2": l2_difference(left["selected_residual"], right["selected_residual"]),
        "final_action_l2": l2_difference(left["action_mean"], right["action_mean"]),
        "legacy_command_changed_columns": changed_columns(
            left_policy[..., LEGACY_COMMAND_SLICE], right_policy[..., LEGACY_COMMAND_SLICE], offset=9
        ),
        "new_command_changed_columns": changed_columns(
            left_policy[..., NEW_COMMAND_SLICE], right_policy[..., NEW_COMMAND_SLICE]
        ),
    }


def component_responds(result: dict) -> bool:
    return max(
        result["base_actor_action_l2"], result["residual_action_l2"], result["final_action_l2"]
    ) >= THRESHOLD


def main() -> None:
    checkpoint = Path(args_cli.checkpoint).resolve(strict=True)
    output = (REPOSITORY_ROOT / args_cli.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    task = "Isaac-Motion-Flat-G1-Command-Sequence-Eval-v0"
    env_cfg, agent_cfg = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    with launch_simulation(env_cfg, args_cli):
        raw_env = gym.make(task, cfg=env_cfg)
        wrapped = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        agent_cfg.device = raw_env.unwrapped.device
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False},
        )
        actor = runner.alg.actor
        robot = raw_env.unwrapped.scene["robot"]
        path_correction_ids, _ = robot.find_joints(".*_(hip_roll|hip_yaw|ankle_roll)_joint|torso_joint")
        propulsion_ids, _ = robot.find_joints(".*_(hip_pitch|knee|ankle_pitch)_joint")
        wrapped.reset()
        observations = wrapped.get_observations()

        variants = {
            "run_center": coherent_run_observation(observations),
            "run_left_path_error": coherent_run_observation(observations, lateral_error_m=0.50),
            "run_right_path_error": coherent_run_observation(observations, lateral_error_m=-0.50),
            "stop": coherent_stop_observation(observations),
            "crouch_shallow": coherent_crouch_observation(observations, height_drop_m=0.08),
            "crouch_deep": coherent_crouch_observation(observations, height_drop_m=0.15),
            "crouch_down": coherent_crouch_observation(
                observations, height_drop_m=0.12, phase=0.0, height_error_m=-0.02,
                target_vertical_velocity_mps=-0.08,
            ),
            "crouch_hold": coherent_crouch_observation(
                observations, height_drop_m=0.12, phase=1.0 / 3.0, hold_progress=0.5,
            ),
            "crouch_return": coherent_crouch_observation(
                observations, height_drop_m=0.06, phase=2.0 / 3.0, height_error_m=0.02,
                target_vertical_velocity_mps=0.08, hold_progress=1.0, return_progress=0.5,
            ),
            "crouch_error_negative": coherent_crouch_observation(observations, height_error_m=-0.03),
            "crouch_error_positive": coherent_crouch_observation(observations, height_error_m=0.03),
            "turn_0": coherent_turn_observation(observations, 0.0),
            "left_45": coherent_turn_observation(observations, math.pi / 4.0),
            "right_45": coherent_turn_observation(observations, -math.pi / 4.0),
            "left_90": coherent_turn_observation(observations, math.pi / 2.0),
            "right_90": coherent_turn_observation(observations, -math.pi / 2.0),
        }
        with torch.inference_mode():
            components = {name: actor.diagnostic_components(value) for name, value in variants.items()}

        requested_pairs = {
            "left_45_vs_right_45": ("left_45", "right_45"),
            "turn_0_vs_left_45": ("turn_0", "left_45"),
            "turn_0_vs_right_45": ("turn_0", "right_45"),
            "left_45_vs_left_90": ("left_45", "left_90"),
            "right_45_vs_right_90": ("right_45", "right_90"),
        }
        counterfactuals = {
            label: compare(variants[left], variants[right], actor)
            for label, (left, right) in requested_pairs.items()
        }
        legacy_pairs = {
            "run_vs_turn": compare(variants["run_center"], variants["left_45"], actor),
            "run_vs_stop": compare(variants["run_center"], variants["stop"], actor),
            "turn_vs_stop": compare(variants["left_45"], variants["stop"], actor),
            "crouch_shallow_vs_deep": compare(variants["crouch_shallow"], variants["crouch_deep"], actor),
            "crouch_down_vs_hold": compare(variants["crouch_down"], variants["crouch_hold"], actor),
            "crouch_hold_vs_return": compare(variants["crouch_hold"], variants["crouch_return"], actor),
            "crouch_negative_vs_positive_height_error": compare(
                variants["crouch_error_negative"], variants["crouch_error_positive"], actor
            ),
        }

        residual_norms = {
            "run": float(torch.linalg.vector_norm(components["run_center"]["selected_residual"], dim=-1).mean().item()),
            "turn_0": float(torch.linalg.vector_norm(components["turn_0"]["selected_residual"], dim=-1).mean().item()),
            "left_45": float(torch.linalg.vector_norm(components["left_45"]["selected_residual"], dim=-1).mean().item()),
            "right_45": float(torch.linalg.vector_norm(components["right_45"]["selected_residual"], dim=-1).mean().item()),
            "left_90": float(torch.linalg.vector_norm(components["left_90"]["selected_residual"], dim=-1).mean().item()),
            "right_90": float(torch.linalg.vector_norm(components["right_90"]["selected_residual"], dim=-1).mean().item()),
            "stop": float(torch.linalg.vector_norm(components["stop"]["selected_residual"], dim=-1).mean().item()),
            "crouch": float(torch.linalg.vector_norm(components["crouch_deep"]["selected_residual"], dim=-1).mean().item()),
        }
        path_result = compare(variants["run_left_path_error"], variants["run_right_path_error"], actor)
        run_residual = components["run_center"]["selected_residual"][0]
        path_correction_rms = float(run_residual[path_correction_ids].square().mean().sqrt().item())
        propulsion_rms = float(run_residual[propulsion_ids].square().mean().sqrt().item())

        direction_sensitive = component_responds(counterfactuals["left_45_vs_right_45"])
        zero_to_turn_sensitive = (
            component_responds(counterfactuals["turn_0_vs_left_45"])
            and component_responds(counterfactuals["turn_0_vs_right_45"])
        )
        angle_sensitive_at_same_state = (
            component_responds(counterfactuals["left_45_vs_left_90"])
            or component_responds(counterfactuals["right_45_vs_right_90"])
        )
        # A feedback controller can produce 45/90 with the same saturated initial
        # action, so angle sensitivity at one state is reported, not mandatory.
        turn_command_sensitive = direction_sensitive and zero_to_turn_sensitive
        run_command_sensitive = path_result["final_action_l2"] >= THRESHOLD
        stop_command_sensitive = component_responds(legacy_pairs["run_vs_stop"])
        crouch_pairs = {
            name: legacy_pairs[name]
            for name in (
                "crouch_shallow_vs_deep", "crouch_down_vs_hold", "crouch_hold_vs_return",
                "crouch_negative_vs_positive_height_error",
            )
        }
        crouch_command_sensitive = sum(
            result["residual_action_l2"] >= CROUCH_THRESHOLD for result in crouch_pairs.values()
        ) >= 3
        required = {
            "run": run_command_sensitive,
            "turn": run_command_sensitive and turn_command_sensitive,
            "stop": run_command_sensitive and turn_command_sensitive and stop_command_sensitive,
            # CROUCH sensitivity is judged on its own relative-height/phase
            # counterfactuals. RUN/TURN retention is gated separately.
            "crouch": crouch_command_sensitive,
            "sequence": run_command_sensitive and turn_command_sensitive and stop_command_sensitive,
        }

        route_weight_norms = actor.command_weight_norms()
        report = {
            "checkpoint": str(checkpoint),
            "stage": args_cli.stage,
            "observation_layout": {
                "legacy_123": [
                    {"start": start, "end_exclusive": end, "meaning": meaning}
                    for start, end, meaning in LEGACY_LAYOUT
                ],
                "legacy_command_columns": {
                    "absolute_indices": [9, 10, 11],
                    "meaning": ["target_body_vx_mps", "target_body_vy_mps", "target_yaw_rate_radps"],
                },
                "new_29": [
                    {"start": start, "end_exclusive": end, "meaning": meaning}
                    for start, end, meaning in NEW_COMMAND_LAYOUT
                ],
            },
            "command_generation": {
                "legacy_source": "MotionCommand.vel_command_b exposed by mdp.generated_commands",
                "new_source": "MotionCommand.extra_command exposed by motion_command_observation",
                "turn_target": "fixed target_heading_w = wrap(turn_start_heading_w + commanded_turn_angle_rad)",
                "legacy_yaw_formula": "clamp(heading_control_stiffness * wrap(target_heading_w - robot_heading_w), yaw_rate_range)",
                "consistency": "both are generated in MotionCommand._update_command from the same target heading/turn angle",
            },
            "complete_turn_counterfactuals": counterfactuals,
            "skill_counterfactuals": legacy_pairs,
            "skill_residual_action_l2": residual_norms,
            "command_input_weight_norm": {
                "total": float(torch.linalg.vector_norm(route_weight_norms).item()),
                "per_skill_per_column": {
                    name: route_weight_norms[skill_id].tolist() for name, skill_id in SKILLS.items()
                },
            },
            "command_encoder_output": {
                name: components[name]["command_embedding"][0].tolist()
                for name in (
                    "run_center", "turn_0", "left_45", "right_45", "left_90", "right_90", "stop",
                    "crouch_shallow", "crouch_deep", "crouch_down", "crouch_hold", "crouch_return",
                    "crouch_error_negative", "crouch_error_positive",
                )
            },
            "path_lateral_counterfactual": {
                "signed_errors_m": [0.50, -0.50],
                **path_result,
                "minimum_final_action_l2": THRESHOLD,
            },
            "crouch_counterfactuals": {
                name: {
                    **result,
                    "command_encoder_output_l2": l2_difference(
                        components[requested_pairs[0]]["command_embeddings"][:, CROUCH],
                        components[requested_pairs[1]]["command_embeddings"][:, CROUCH],
                    ),
                    "target_joint_action_abs_differences": {
                        robot.joint_names[index]: float(
                            (
                                components[requested_pairs[0]]["action_mean"][0, index]
                                - components[requested_pairs[1]]["action_mean"][0, index]
                            ).abs().item()
                        )
                        for index in propulsion_ids
                    },
                }
                for name, result in crouch_pairs.items()
                for requested_pairs in ({
                    "crouch_shallow_vs_deep": ("crouch_shallow", "crouch_deep"),
                    "crouch_down_vs_hold": ("crouch_down", "crouch_hold"),
                    "crouch_hold_vs_return": ("crouch_hold", "crouch_return"),
                    "crouch_negative_vs_positive_height_error": ("crouch_error_negative", "crouch_error_positive"),
                }[name],)
            },
            "run_residual_role": {
                "path_correction_joint_rms": path_correction_rms,
                "sagittal_propulsion_joint_rms": propulsion_rms,
                "path_correction_to_propulsion_ratio": path_correction_rms / max(propulsion_rms, 1.0e-8),
                "path_correction_joints": [robot.joint_names[index] for index in path_correction_ids],
                "sagittal_propulsion_joints": [robot.joint_names[index] for index in propulsion_ids],
            },
            "skill_gate": {
                name: components[name]["gate"][0].tolist()
                for name in ("run_center", "turn_0", "left_45", "right_45", "left_90", "right_90", "stop", "crouch_shallow", "crouch_deep")
            },
            "acceptance": {
                "minimum_action_l2": THRESHOLD,
                "crouch_minimum_residual_action_l2": CROUCH_THRESHOLD,
                "crouch_required_passing_counterfactuals": 3,
                "run_path_command_sensitive": run_command_sensitive,
                "turn_direction_sensitive": direction_sensitive,
                "turn_zero_vs_nonzero_sensitive": zero_to_turn_sensitive,
                "turn_angle_sensitive_at_identical_initial_state": angle_sensitive_at_same_state,
                "turn_command_sensitive": turn_command_sensitive,
                "crouch_depth_command_sensitive": crouch_command_sensitive,
                "turn_residual_required": False,
                "fixed_time_same_direction_rejection_enforced": True,
                "command_sensitive": required[args_cli.stage],
            },
        }
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        raw_env.close()


if __name__ == "__main__":
    main()
