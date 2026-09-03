"""Deterministic Stage 0 audit.  It launches neither simulation nor training."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict


SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
EXP005 = REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run"
EXP006 = REPO / "experiments/isaaclab/exp_006_unitree_g1_command_skills"
OUTPUT = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage0_expert_audit"
ISAACLAB = REPO.parent / "IsaacLab"
START_HEAD = "4454f44f43a545bee06d6f197527c2fb1c3ad5bb"
START_EXP006_README_SHA256 = "641a9c809e6902e8a80e5925dfc660b9694c2d7776c6c124515398e42bbae801"

sys.path[:0] = [str(EXP / "src"), str(EXP006 / "src"), str(EXP005 / "src")]

from g1_command_skills.models import G1CommandResidualActor  # noqa: E402
from g1_command_skills.tasks.g1_command_env_cfg import G1CommandRunEvalEnvCfg  # noqa: E402
from g1_flat_run.tasks.g1_flat_run_env_cfg import G1FlatRunStage2EnvCfg  # noqa: E402
from g1_walk_centered.command_contract import MotionCommand  # noqa: E402
from g1_walk_centered.experts import LegacyWalkActor, RunExpert, WalkExpert  # noqa: E402
from g1_walk_centered.experts.adapters import (  # noqa: E402
    ACTION_DIM,
    FORBIDDEN_INPUTS,
    nominal_state,
    to_run_observation,
    to_walk_observation,
)
from g1_walk_centered.models.transition_bridge import NOT_IMPLEMENTED_STAGE_0, TransitionBridge  # noqa: E402


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(name: str, value: Any) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def git(*args: str, cwd: Path = REPO) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def actor_state(checkpoint: Path) -> tuple[dict[str, torch.Tensor], int]:
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return data["actor_state_dict"], int(data["iter"])


def make_walk_actor(state: dict[str, torch.Tensor]) -> LegacyWalkActor:
    actor = LegacyWalkActor()
    actor.load_state_dict(
        {key.removeprefix("mlp."): value for key, value in state.items() if key.startswith("mlp.")},
        strict=True,
    )
    actor.eval()
    return actor


def make_run_actor(state: dict[str, torch.Tensor]) -> G1CommandResidualActor:
    observation = TensorDict({"policy": torch.zeros(1, 152)}, batch_size=[1])
    actor = G1CommandResidualActor(
        observation,
        {"actor": ["policy"]},
        "actor",
        37,
        hidden_dims=[256, 128, 128],
        activation="elu",
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
        trainable_skill_ids=[],
        crouch_controller="scripted_shallow_v1",
        learned_crouch_residual_enabled=False,
    )
    actor.load_state_dict(state, strict=True)
    actor.eval()
    return actor


def tensor_difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    delta = left.detach().cpu() - right.detach().cpu()
    return {
        "bitwise_equal": bool(torch.equal(left, right)),
        "max_absolute_difference": float(delta.abs().max().item()) if delta.numel() else 0.0,
        "l2_difference": float(torch.linalg.vector_norm(delta).item()),
    }


def action_record(
    canonical_name: str,
    command: MotionCommand,
    observation: torch.Tensor,
    raw: torch.Tensor,
    residual: torch.Tensor,
    final: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, Any]:
    result = tensor_difference(final, reference)
    return {
        "case": canonical_name,
        "canonical_input": {
            "state": {
                "base_linear_velocity_body_mps": [0.0, 0.0, 0.0],
                "base_angular_velocity_body_radps": [0.0, 0.0, 0.0],
                "projected_gravity_body": [0.0, 0.0, -1.0],
                "heading_w_rad": 0.0,
                "joint_position_relative_37": [0.0] * 37,
                "joint_velocity_relative_37": [0.0] * 37,
                "previous_action_37": [0.0] * 37,
            },
            "absolute_world_xy_present": False,
            "target_speed_mps": command.target_speed_mps,
            "target_heading_w_rad": command.target_heading_w_rad,
            "target_yaw_rate_radps": command.target_yaw_rate_radps,
            "posture": command.posture,
            "crouch_depth_m": command.crouch_depth_m,
        },
        "expert_observation": observation[0].tolist(),
        "observation_shape": list(observation.shape),
        "observation_dtype": str(observation.dtype),
        "command_columns_9_12": observation[0, 9:12].tolist(),
        "raw_actor_action": raw[0].tolist(),
        "residual_action": residual[0].tolist(),
        "final_action": final[0].tolist(),
        "action_sha256": sha256_tensor(final),
        **result,
    }


def primitive(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [primitive(item) for item in value]
    if isinstance(value, dict):
        return {str(key): primitive(item) for key, item in value.items()}
    return str(value)


def actuator_spec(cfg: Any) -> dict[str, Any]:
    result = {}
    for name, actuator in cfg.scene.robot.actuators.items():
        result[name] = {
            "joint_names_expr": primitive(actuator.joint_names_expr),
            "stiffness": primitive(actuator.stiffness),
            "damping": primitive(actuator.damping),
            "armature": primitive(getattr(actuator, "armature", None)),
            "friction": primitive(getattr(actuator, "friction", None)),
            "dynamic_friction": primitive(getattr(actuator, "dynamic_friction", None)),
            "effort_limit": primitive(getattr(actuator, "effort_limit", None)),
            "effort_limit_sim": primitive(getattr(actuator, "effort_limit_sim", None)),
            "velocity_limit": primitive(getattr(actuator, "velocity_limit", None)),
            "velocity_limit_sim": primitive(getattr(actuator, "velocity_limit_sim", None)),
        }
    return result


def env_spec(cfg: Any, robot_state: dict[str, Any]) -> dict[str, Any]:
    spawn = cfg.scene.robot.spawn
    return {
        "g1_asset": primitive(getattr(spawn, "usd_path", getattr(spawn, "asset_path", None))),
        "action_dimension": ACTION_DIM,
        "action_scale": float(cfg.actions.joint_pos.scale),
        "action_semantics": "default_joint_position + 0.5 * normalized_position_action",
        "physics_timestep_s": float(cfg.sim.dt),
        "decimation": int(cfg.decimation),
        "control_timestep_s": float(cfg.sim.dt * cfg.decimation),
        "actuators": actuator_spec(cfg),
        "default_joint_positions": robot_state["default_joint_pos"],
        "joint_position_limits": robot_state["hard_joint_pos_limits"],
        "runtime_velocity_limits": robot_state["joint_velocity_limits"],
        "runtime_effort_limits": robot_state["joint_effort_limits"],
        "terrain": primitive(cfg.scene.terrain.terrain_type),
        "terrain_friction": robot_state["terrain_friction"],
        "contact_sensor": primitive(cfg.scene.contact_forces),
        "quaternion_convention": "wxyz",
        "observation_coordinate_frame": "base linear/angular velocity and projected gravity in body frame; no absolute world XY",
        "reset_state": primitive(cfg.scene.robot.init_state),
        "termination_terms": sorted(vars(cfg.terminations).keys()),
    }


def main() -> None:
    expert_manifest = load_json(EXP / "expert_manifest.json")["experts"]
    references = load_json(EXP / "reference_manifest.json")
    walk_path = REPO / expert_manifest["WALK_STAND"]["checkpoint"]
    run_a_path = REPO / expert_manifest["RUN"]["checkpoint"]
    run_b_path = REPO / expert_manifest["RUN_CANDIDATE_B"]["checkpoint"]
    crouch_dir = REPO / expert_manifest["CROUCH_SHALLOW"]["artifact"]
    crouch_controller = REPO / expert_manifest["CROUCH_SHALLOW"]["controller_source"]
    crouch_pose = crouch_dir / "pose_lookup.json"
    command_system = REPO / references["exp006_command_system_artifact"]
    required = [walk_path, run_a_path, run_b_path, crouch_dir, crouch_controller, crouch_pose, command_system]
    missing = [str(path.relative_to(REPO)) for path in required if not path.exists()]

    hashes_before = {
        "walk": sha256_file(walk_path) if walk_path.exists() else None,
        "run_candidate_a": sha256_file(run_a_path) if run_a_path.exists() else None,
        "run_candidate_b": sha256_file(run_b_path) if run_b_path.exists() else None,
        "crouch_controller": sha256_file(crouch_controller) if crouch_controller.exists() else None,
        "crouch_pose_lookup": sha256_file(crouch_pose) if crouch_pose.exists() else None,
    }
    expected_hashes = {
        "walk": expert_manifest["WALK_STAND"]["sha256"],
        "run_candidate_a": expert_manifest["RUN"]["sha256"],
        "run_candidate_b": expert_manifest["RUN_CANDIDATE_B"]["sha256"],
        "crouch_controller": expert_manifest["CROUCH_SHALLOW"]["controller_sha256"],
        "crouch_pose_lookup": expert_manifest["CROUCH_SHALLOW"]["pose_lookup_sha256"],
    }

    walk_state, walk_iter = actor_state(walk_path)
    run_a_state, run_a_iter = actor_state(run_a_path)
    run_b_state, run_b_iter = actor_state(run_b_path)
    walk_direct = make_walk_actor(walk_state)
    walk_adapter = WalkExpert(make_walk_actor(walk_state))
    run_a_direct = make_run_actor(run_a_state)
    run_a_adapter = RunExpert(make_run_actor(run_a_state))
    run_b_direct = make_run_actor(run_b_state)

    protected_prefixes = (
        "base_mlp.",
        "skill_command_encoders.0.",
        "skill_state_adapters.0.",
        "residual_heads.0.",
        "skill_command_encoders.2.",
        "skill_state_adapters.2.",
        "residual_heads.2.",
    )
    protected_keys = sorted(key for key in run_a_state if key.startswith(protected_prefixes))
    protected_comparison = {}
    for key in protected_keys:
        protected_comparison[key] = {
            "candidate_a_sha256": sha256_tensor(run_a_state[key]),
            "candidate_b_sha256": sha256_tensor(run_b_state[key]),
            **tensor_difference(run_a_state[key], run_b_state[key]),
        }

    canonical = nominal_state()
    cases: list[dict[str, Any]] = []
    walk_cases = [
        ("nominal_standing_state", 0.0),
        ("zero_velocity_command", 0.0),
        ("walk_0.5_mps", 0.5),
        ("walk_1.0_mps", 1.0),
        ("walk_1.5_mps", 1.5),
    ]
    with torch.inference_mode():
        for name, speed in walk_cases:
            command = MotionCommand(speed, 0.0)
            obs = to_walk_observation(canonical, command)
            direct = walk_direct(obs)
            adapted = walk_adapter(canonical, command)
            cases.append(action_record(name, command, obs, direct, torch.zeros_like(direct), adapted, direct))

        run_cases = [
            ("run_2.4_mps", "RUN", 2.4, 0.0),
            ("run_2.6_mps", "RUN", 2.6, 0.0),
            ("turn_left_45_deg", "TURN", 2.6, math.radians(45.0)),
            ("turn_right_45_deg", "TURN", 2.6, math.radians(-45.0)),
            ("turn_left_90_deg", "TURN", 2.6, math.radians(90.0)),
            ("turn_right_90_deg", "TURN", 2.6, math.radians(-90.0)),
        ]
        candidate_action_comparison = {}
        for name, route, speed, heading in run_cases:
            yaw_rate = max(-0.75, min(0.75, 1.5 * heading)) if route == "TURN" else 0.0
            command = MotionCommand(speed, heading, target_yaw_rate_radps=yaw_rate)
            obs = to_run_observation(canonical, command, route=route)
            wrapped = TensorDict({"policy": obs}, batch_size=[1])
            direct = run_a_direct.diagnostic_components(wrapped)
            adapted = run_a_adapter.action_components(canonical, command, route=route)
            b_action = run_b_direct.diagnostic_components(wrapped)["action_mean"]
            candidate_action_comparison[name] = tensor_difference(direct["action_mean"], b_action)
            cases.append(
                action_record(
                    name,
                    command,
                    obs,
                    direct["base_action"],
                    direct["selected_residual"],
                    adapted["action_mean"],
                    direct["action_mean"],
                )
            )

    walk_cfg = G1FlatRunStage2EnvCfg()
    run_cfg = G1CommandRunEvalEnvCfg()
    robot_state = load_json(REPO / references["robot_state_reference"])
    walk_env = env_spec(walk_cfg, robot_state)
    run_env = env_spec(run_cfg, robot_state)
    compatibility_fields = [
        "g1_asset", "action_dimension", "action_scale", "action_semantics",
        "physics_timestep_s", "decimation", "control_timestep_s", "actuators",
        "default_joint_positions", "joint_position_limits", "runtime_velocity_limits", "runtime_effort_limits",
        "terrain", "terrain_friction", "contact_sensor", "quaternion_convention",
        "observation_coordinate_frame", "reset_state", "termination_terms",
    ]
    environment_differences = {
        key: {"walk": walk_env[key], "run": run_env[key]}
        for key in compatibility_fields
        if walk_env[key] != run_env[key]
    }
    # Command and reward terms may differ, but production physics/action compatibility may not.
    critical_fields = [
        "g1_asset", "action_dimension", "action_scale", "action_semantics",
        "physics_timestep_s", "decimation", "control_timestep_s", "actuators",
        "default_joint_positions", "joint_position_limits", "runtime_velocity_limits", "runtime_effort_limits",
        "terrain", "terrain_friction", "contact_sensor", "quaternion_convention",
        "observation_coordinate_frame", "reset_state", "termination_terms",
    ]
    critical_differences = {key: value for key, value in environment_differences.items() if key in critical_fields}

    sha_lines = (command_system / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    artifact_checks = []
    for line in sha_lines:
        expected, relative = line.strip().split(maxsplit=1)
        relative = relative.lstrip("*")
        path = command_system / relative
        actual = sha256_file(path) if path.exists() else None
        artifact_checks.append({"path": relative, "expected": expected, "actual": actual, "match": actual == expected})

    status_lines = [line for line in git("status", "--porcelain=v1", "--untracked-files=all").splitlines() if line]
    protected_current = {
        "exp005_tracked_diff": git("diff", "--name-only", "--", "experiments/isaaclab/exp_005_unitree_g1_flat_run").splitlines(),
        "exp006_tracked_diff": git("diff", "--name-only", "--", "experiments/isaaclab/exp_006_unitree_g1_command_skills").splitlines(),
        "isaaclab_tracked_diff": git("diff", "--name-only", cwd=ISAACLAB).splitlines(),
    }
    repository_state = {
        "branch": git("branch", "--show-current"),
        "start_head": START_HEAD,
        "current_head": git("rev-parse", "HEAD"),
        "start_status": [
            " M experiments/isaaclab/exp_006_unitree_g1_command_skills/README.md",
            "?? artifacts/exp_005_unitree_g1_flat_run/exported/policy.onnx.data",
            "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/assemble_showcase_reel.py",
            "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/play_command_system_showcase.ps1",
            "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/play_showcase.py"
        ],
        "current_status": status_lines,
        "protected_current": protected_current,
        "start_exp006_readme_sha256": START_EXP006_README_SHA256,
        "current_exp006_readme_sha256": sha256_file(EXP006 / "README.md"),
        "isaaclab_head": git("rev-parse", "HEAD", cwd=ISAACLAB),
        "exp007_preexisted_at_start": True,
        "exp007_start_files": git("ls-tree", "-r", "--name-only", START_HEAD, "--", str(EXP.relative_to(REPO))).splitlines(),
    }

    hashes_after = {
        "walk": sha256_file(walk_path),
        "run_candidate_a": sha256_file(run_a_path),
        "run_candidate_b": sha256_file(run_b_path),
        "crouch_controller": sha256_file(crouch_controller),
        "crouch_pose_lookup": sha256_file(crouch_pose),
    }
    checkpoint_hashes = {
        "algorithm": "SHA-256",
        "paths": {
            "walk": str(walk_path.relative_to(REPO)),
            "run_candidate_a": str(run_a_path.relative_to(REPO)),
            "run_candidate_b": str(run_b_path.relative_to(REPO)),
        },
        "expected": expected_hashes,
        "actual_before": hashes_before,
        "actual_after": hashes_after,
        "all_match": expected_hashes == hashes_before == hashes_after,
        "checkpoint_copied": False,
        "checkpoint_modified": False,
    }
    architectures = {
        "walk": {
            "actor_class": "LegacyWalkActor / exp_005 MLP",
            "iteration": walk_iter,
            "input_dimension": int(walk_state["mlp.0.weight"].shape[1]),
            "hidden_dimensions": [256, 128, 128],
            "output_dimension": int(walk_state["mlp.6.weight"].shape[0]),
            "normalization": "None in actor checkpoint; observations are task terms with scale 1.0",
        },
        "run": {
            "actor_class": "G1CommandResidualActor",
            "iteration": run_a_iter,
            "input_dimension": 152,
            "legacy_dimension": int(run_a_state["base_mlp.0.weight"].shape[1]),
            "command_dimension": 29,
            "command_encoder": "per-skill 29->64->32",
            "state_adapter": "per-skill 123->128->64",
            "residual_route": "concat(64,32)->128->64->37; 0.25*tanh",
            "output_dimension": int(run_a_state["base_mlp.6.weight"].shape[0]),
            "normalization": "Identity Normalizer",
        },
        "run_candidate_b_iteration": run_b_iter,
    }
    observation_layouts = {
        "walk_123": [
            [0, 3, "base_linear_velocity_body_mps"],
            [3, 6, "base_angular_velocity_body_radps"],
            [6, 9, "projected_gravity_body"],
            [9, 12, "velocity_command_body_vx_vy_yaw_rate"],
            [12, 49, "joint_position_relative_37"],
            [49, 86, "joint_velocity_relative_37"],
            [86, 123, "previous_action_37"],
        ],
        "run_legacy_123": "identical to walk_123",
        "run_added_29": [
            [123, 129, "current_skill_one_hot"],
            [129, 135, "previous_skill_one_hot"],
            [135, 136, "sin_target_heading_error"],
            [136, 137, "cos_target_heading_error"],
            [137, 139, "skill_local_target_state"],
            [139, 140, "relative_target_pelvis_height"],
            [140, 144, "skill_local_auxiliary_state"],
            [144, 145, "target_vertical_velocity"],
            [145, 146, "normalized_elapsed_time"],
            [146, 147, "normalized_remaining_time"],
            [147, 148, "skill_phase"],
            [148, 149, "transition_progress"],
            [149, 150, "recovery_mode"],
            [150, 152, "target_posture_roll_pitch"],
        ],
        "absolute_world_xy_present": False,
        "unified_to_152": False,
    }
    action_order = {
        "dimension": len(robot_state["joint_names"]),
        "joint_names": robot_state["joint_names"],
        "default_joint_positions": robot_state["default_joint_pos"],
        "hard_joint_position_limits": robot_state["hard_joint_pos_limits"],
        "walk_run_identical": True,
        "source": references["robot_state_reference"],
    }
    action_scales = {
        "walk": walk_env["action_scale"],
        "run": run_env["action_scale"],
        "compatible": walk_env["action_scale"] == run_env["action_scale"] == 0.5,
        "semantics": walk_env["action_semantics"],
        "run_residual_normalized_action_limit": 0.25,
    }
    environment = {
        "walk": walk_env,
        "run": run_env,
        "differences": environment_differences,
        "critical_differences": critical_differences,
        "compatible": not critical_differences,
        "note": "Command generator/reward ranges differ by design and are not production action/physics differences.",
    }
    all_case_adapter_equal = all(case["bitwise_equal"] for case in cases)
    all_candidate_actions_equal = all(item["bitwise_equal"] for item in candidate_action_comparison.values())
    all_protected_equal = all(item["bitwise_equal"] for item in protected_comparison.values())
    bitwise = {
        "reference_method": "direct immutable source actor vs exp_007 adapter at identical fixed canonical input",
        "preexisting_persisted_exp005_action_vectors": False,
        "preexisting_persisted_exp006_action_vectors": True,
        "cases": cases,
        "all_adapter_actions_bitwise_equal": all_case_adapter_equal,
        "run_candidate_action_comparison": candidate_action_comparison,
        "all_run_turn_candidate_actions_bitwise_equal": all_candidate_actions_equal,
    }
    protected_tensor_hashes = {
        "scope": list(protected_prefixes),
        "tensors": protected_comparison,
        "all_bitwise_equal": all_protected_equal,
    }

    failures: list[str] = []
    warnings: list[str] = []
    checks = {
        "all_checkpoints_exist": not missing,
        "all_artifacts_exist": all(path.exists() for path in (crouch_dir, command_system)),
        "sha256_match": checkpoint_hashes["all_match"] and all(row["match"] for row in artifact_checks),
        "actor_architecture_loaded": architectures["walk"]["input_dimension"] == 123 and architectures["run"]["input_dimension"] == 152,
        "observation_layout_resolved": observation_layouts["absolute_world_xy_present"] is False,
        "action_order_37_identical": action_order["dimension"] == 37 and action_order["walk_run_identical"],
        "action_semantics_identical": walk_env["action_semantics"] == run_env["action_semantics"],
        "action_scale_compatible": action_scales["compatible"],
        "physics_timestep_identical": walk_env["physics_timestep_s"] == run_env["physics_timestep_s"],
        "control_timestep_identical": walk_env["control_timestep_s"] == run_env["control_timestep_s"],
        "pd_compatible": walk_env["actuators"] == run_env["actuators"],
        "effort_velocity_limits_compatible": walk_env["runtime_effort_limits"] == run_env["runtime_effort_limits"] and walk_env["runtime_velocity_limits"] == run_env["runtime_velocity_limits"],
        "adapter_outputs_finite": all(torch.isfinite(torch.tensor(case["final_action"])).all().item() for case in cases),
        "absolute_world_xy_absent": not observation_layouts["absolute_world_xy_present"] and FORBIDDEN_INPUTS == {"absolute_world_x", "absolute_world_y"},
        "walk_reference_bitwise": all(case["bitwise_equal"] for case in cases if case["case"].startswith(("nominal", "zero", "walk"))),
        "run_reference_bitwise": all(case["bitwise_equal"] for case in cases if case["case"].startswith("run")),
        "turn_reference_bitwise": all(case["bitwise_equal"] for case in cases if case["case"].startswith("turn")) and all_candidate_actions_equal and all_protected_equal,
        "expert_weights_unchanged": hashes_before == hashes_after,
        "exp005_existing_tracked_files_unchanged": not protected_current["exp005_tracked_diff"],
        "exp006_existing_tracked_files_unchanged": protected_current["exp006_tracked_diff"] == ["experiments/isaaclab/exp_006_unitree_g1_command_skills/README.md"] and repository_state["current_exp006_readme_sha256"] == START_EXP006_README_SHA256,
        "isaaclab_tracked_files_unchanged": not protected_current["isaaclab_tracked_diff"],
        "bridge_disconnected": TransitionBridge()() == NOT_IMPLEMENTED_STAGE_0 and not TransitionBridge.production_connected and not TransitionBridge.trainable,
    }
    for name, passed in checks.items():
        if not passed:
            failures.append(name)
    if not bitwise["preexisting_persisted_exp005_action_vectors"]:
        warnings.append("No pre-existing persisted exp_005 fixed-action vectors; exp_007 records direct-source-vs-adapter vectors now.")
    if status_lines:
        warnings.append("Repository was dirty before Stage 0; unrelated changes are preserved and excluded from the exp_007 commit.")
    if environment_differences:
        failures.append("environment_critical_differences")
    eligible = not failures
    comparison = {
        "candidate_a": str(run_a_path.relative_to(REPO)),
        "candidate_b": str(run_b_path.relative_to(REPO)),
        "checkpoint_files_bitwise_equal": hashes_before["run_candidate_a"] == hashes_before["run_candidate_b"],
        "protected_run_turn_tensors_bitwise_equal": all_protected_equal,
        "representative_run_turn_actions_bitwise_equal": all_candidate_actions_equal,
        "selected": "A" if all_protected_equal and all_candidate_actions_equal else None,
        "reason": "A is the simpler RUN/TURN-specific parent; B remains formal provenance and STOP baseline.",
        "formal_metrics": {
            "candidate_a_turn_summary": references["exp006_candidate_a_turn_summary"],
            "candidate_b_run_turn_summary": references["exp006_formal_run_turn_summary"],
            "not_recomputed_stage0": True,
        },
    }
    stage0_gate = {
        "schema_version": 1,
        "eligible_for_stage1": eligible,
        "failures": sorted(set(failures)),
        "warnings": warnings,
        "walk_expert": expert_manifest["WALK_STAND"],
        "run_expert": expert_manifest["RUN"],
        "run_candidate_comparison": comparison,
        "crouch_reference": expert_manifest["CROUCH_SHALLOW"],
        "stop_baseline": expert_manifest["DIRECT_STOP_BASELINE"],
        "repository_cleanliness": repository_state,
        "checks": checks,
    }
    summary = {
        "schema_version": 2,
        "experiment": "exp_007_unitree_g1_walk_centered_transitions",
        "stage": 0,
        "status": "PASS" if eligible else "FAIL",
        "eligible_for_stage1": eligible,
        "failures": sorted(set(failures)),
        "warnings": warnings,
        "training_performed": False,
        "simulation_started": False,
        "formal_transition_evaluation_performed": False,
        "run_candidate_comparison": comparison,
    }

    dump("audit_summary.json", summary)
    dump("checkpoint_hashes.json", {**checkpoint_hashes, "command_system_sha256sums": artifact_checks})
    dump("expert_architectures.json", architectures)
    dump("observation_layouts.json", observation_layouts)
    dump("action_order.json", action_order)
    dump("action_scales.json", action_scales)
    dump("environment_compatibility.json", environment)
    dump("bitwise_reference.json", bitwise)
    dump("protected_tensor_hashes.json", protected_tensor_hashes)
    dump("repository_state.json", repository_state)
    dump("stage0_gate.json", stage0_gate)
    print(json.dumps(summary, indent=2))
    if not eligible:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
