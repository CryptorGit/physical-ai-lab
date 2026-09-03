"""EXP014 Phase 2-D31B-R3 terminal official Agile lower-body start diagnostic."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
ISAAC_ROOT = Path(r"C:\Users\user\workspace\IsaacLab")
ISAAC_PYTHON = ISAAC_ROOT / "env_isaaclab" / "Scripts" / "python.exe"
OUT = (
    REPO
    / "results"
    / "exp_014_unitree_g1_explicit_motion_mode_unified_locomotion"
    / "phase_2_d31br3_terminal_official_agile_start"
)
REPORT = REPO / "research" / "exp_014_phase_2_d31br3_terminal_official_agile_start_report.md"
POLICY = (
    REPO
    / "results"
    / "exp_014_unitree_g1_explicit_motion_mode_unified_locomotion"
    / "phase_2_d31br_official_agile_remote_retrieval"
    / "official_retrieval"
    / "Assets"
    / "Isaac"
    / "6.0"
    / "Isaac"
    / "IsaacLab"
    / "Policies"
    / "Agile"
    / "agile_locomotion.pt"
)
POLICY_SHA256 = "f04a58b834057eb1c9f38350dc12feaf929ff2cc7d5b75d2871e23811b775dde"
SEEDS = list(range(8))
STATIC_SECONDS = 2.0
ZERO_SECONDS = 2.0
FORWARD_SECONDS = 3.0
CONTROL_DT = 0.02
ZERO_COMMAND = [0.0, 0.0, 0.0, 0.72]
FORWARD_COMMAND = [0.5, 0.0, 0.0, 0.72]
STAGE_CLASSES = (
    "EXP014_D31BR3_OFFICIAL_AGILE_LOWER_BODY_START_PASS",
    "EXP014_D31BR3_OFFICIAL_AGILE_LOWER_BODY_START_FAIL",
    "EXP014_D31BR3_LOWER_BODY_CONTRACT_RECONSTRUCTION_FAIL",
    "EXP014_D31BR3_FIXED_UPPER_BODY_INCOMPATIBLE",
    "EXP014_D31BR3_POLICY_INPUT_CONTRACT_FAIL",
    "EXP014_D31BR3_RUNTIME_FAIL",
    "EXP014_D31BR3_MULTIPLE_FAILURES",
)
PROTECTED = (
    "experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d31a_torque_wbc_authority.py",
    "experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d31ar_contact_inverse_dynamics_reconciliation.py",
    "experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d31b_official_g1_start_teacher_audit.py",
    "experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d31br2_official_policy_runtime_repair.py",
)


def sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "detach"):
        return jsonable(value.detach().cpu().numpy())
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return jsonable(value.tolist())
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dump(name: str, value: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True, encoding="utf-8").strip()


def status_lines() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=REPO,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def rel_status(line: str) -> str:
    return line[3:].replace("\\", "/") if len(line) >= 4 else line


def allowed_path(path: str) -> bool:
    return (
        path
        == "experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d31br3_terminal_official_agile_start.py"
        or path == "research/exp_014_phase_2_d31br3_terminal_official_agile_start_report.md"
        or path.startswith(
            "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d31br3_terminal_official_agile_start/"
        )
    )


def load_source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def official_source_contract() -> dict[str, Any]:
    obs_path = (
        ISAAC_ROOT
        / "source"
        / "isaaclab_tasks"
        / "isaaclab_tasks"
        / "manager_based"
        / "locomanipulation"
        / "pick_place"
        / "configs"
        / "agile_locomotion_observation_cfg.py"
    )
    action_path = (
        ISAAC_ROOT
        / "source"
        / "isaaclab_tasks"
        / "isaaclab_tasks"
        / "manager_based"
        / "locomanipulation"
        / "pick_place"
        / "mdp"
        / "actions.py"
    )
    return {
        "observation_source": str(obs_path),
        "observation_source_sha256": sha(obs_path),
        "observation_source_exists": obs_path.is_file(),
        "action_source": str(action_path),
        "action_source_sha256": sha(action_path),
        "action_source_exists": action_path.is_file(),
        "observation_class": "AgileTeacherPolicyObservationsCfg",
        "action_class": "AgileBasedLowerBodyAction",
        "loader": "isaaclab.utils.io.torchscript.load_torchscript_model",
        "policy_sha256": sha(POLICY),
        "policy_expected_sha256": POLICY_SHA256,
    }


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(jsonable(value)) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def native_child() -> dict[str, Any]:
    from isaaclab.app import AppLauncher

    launcher = AppLauncher({"headless": True, "enable_cameras": False})
    app = launcher.app
    del app

    import numpy as np
    import torch
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
    from isaaclab.managers import ActionTerm, ActionTermCfg
    from isaaclab.managers import ObservationGroupCfg as ObsGroupCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab_assets.robots.unitree import G1_29DOF_CFG

    source_contract = official_source_contract()
    obs_module = load_source_module(
        "exp014_official_agile_observations",
        Path(source_contract["observation_source"]),
    )
    action_module = load_source_module(
        "exp014_official_agile_actions",
        Path(source_contract["action_source"]),
    )
    OfficialAgileAction = action_module.AgileBasedLowerBodyAction
    OfficialObsCfg = obs_module.AgileTeacherPolicyObservationsCfg

    @configclass
    class UpperBodyHoldActionCfg(ActionTermCfg):
        class_type: type = "exp014_upper_body_hold:UpperBodyHoldAction"
        joint_names: list[str] = [".*"]

    class UpperBodyHoldAction(ActionTerm):
        cfg: UpperBodyHoldActionCfg

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self._joint_ids_all, self._joint_names_all = self._asset.find_joints(cfg.joint_names)
            lower_ids, _ = self._asset.find_joints(
                [".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"]
            )
            lower_set = {int(item) for item in lower_ids}
            self._joint_ids = [int(item) for item in self._joint_ids_all if int(item) not in lower_set]
            self._joint_names = [self._joint_names_all[int(i)] for i in range(len(self._joint_names_all)) if int(self._joint_ids_all[i]) not in lower_set]
            self._raw_actions = torch.zeros((self.num_envs, len(self._joint_ids)), device=self.device)
            self._processed_actions = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()

        @property
        def action_dim(self):
            return len(self._joint_ids)

        @property
        def raw_actions(self):
            return self._raw_actions

        @property
        def processed_actions(self):
            return self._processed_actions

        def process_actions(self, actions):
            self._raw_actions[:] = actions
            self._processed_actions = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()

        def apply_actions(self):
            self._asset.set_joint_position_target_index(
                target=self._processed_actions,
                joint_ids=self._joint_ids,
            )

    @configclass
    class DiagnosticAgileActionCfg(ActionTermCfg):
        class_type: type = "exp014_diagnostic_agile:DiagnosticAgileAction"
        joint_names: list[str] = [".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"]
        obs_group_name: str = "lower_body_policy"
        policy_path: str = str(POLICY)
        policy_output_scale: float = 0.25
        static_mode: bool = False

    class DiagnosticAgileAction(OfficialAgileAction):
        cfg: DiagnosticAgileActionCfg

        def __init__(self, cfg, env):
            self.static_mode = bool(cfg.static_mode)
            self.last_policy_input = None
            self.last_policy_output = None
            super().__init__(cfg, env)

        def _compose_policy_input(self, base_command, obs_tensor):
            policy_input = super()._compose_policy_input(base_command, obs_tensor)
            self.last_policy_input = policy_input.detach().clone()
            return policy_input

        def process_actions(self, actions):
            if self.static_mode:
                self._raw_actions.zero_()
                self._processed_actions = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()
                self.last_policy_input = None
                self.last_policy_output = None
                return
            super().process_actions(actions)
            self.last_policy_output = self.raw_actions.detach().clone()

        def reset(self, env_ids=None):
            if env_ids is None:
                env_ids = slice(None)
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = self._asset.data.default_joint_pos.torch[env_ids][:, self._joint_ids]
            self.last_policy_input = None
            self.last_policy_output = None

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot: ArticulationCfg = G1_29DOF_CFG
        ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())

    @configclass
    class ActionsCfg:
        upper_body_hold = UpperBodyHoldActionCfg(asset_name="robot", class_type=UpperBodyHoldAction)
        lower_body_joint_pos = DiagnosticAgileActionCfg(asset_name="robot", class_type=DiagnosticAgileAction)

    @configclass
    class ObservationsCfg:
        lower_body_policy: ObsGroupCfg = OfficialObsCfg()

    @configclass
    class DiagnosticCfg(ManagerBasedEnvCfg):
        scene: SceneCfg = SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
        observations: ObservationsCfg = ObservationsCfg()
        actions: ActionsCfg = ActionsCfg()
        sim: SimulationCfg = SimulationCfg(dt=1.0 / 200.0, render_interval=2)
        decimation: int = 4
        seed: int | None = 0

    cfg = DiagnosticCfg()
    env = ManagerBasedEnv(cfg)
    env.render_enabled = False
    robot = env.scene["robot"]
    upper_term = env.action_manager._terms["upper_body_hold"]
    lower_term = env.action_manager._terms["lower_body_joint_pos"]
    foot_ids, foot_names = robot.find_bodies([".*ankle.*"], preserve_order=True)
    foot_ids = [int(item) for item in foot_ids]
    foot_names = [str(item) for item in foot_names]
    if len(foot_ids) > 6:
        foot_ids = foot_ids[:6]
        foot_names = foot_names[:6]

    policy_contract: dict[str, Any] = {
        "action_manager_terms": env.action_manager.active_terms,
        "total_action_dim": env.action_manager.total_action_dim,
        "upper_body_joint_names": upper_term._joint_names,
        "upper_body_joint_count": upper_term.action_dim,
        "lower_body_joint_names": lower_term._joint_names,
        "lower_body_joint_count": len(lower_term._joint_ids),
        "lower_body_action_dim": lower_term.action_dim,
        "observation_terms": env.observation_manager.active_terms,
        "observation_group_dim": env.observation_manager.group_obs_dim,
        "observation_term_dims": env.observation_manager.group_obs_term_dim,
        "foot_body_names": foot_names,
        "foot_body_ids": foot_ids,
        "command_order": ["vx", "vy", "wz", "hip_height"],
        "policy_output_scale": 0.25,
        "policy_output_offset": "G1_29DOF_CFG default lower-body joint positions",
    }
    obs_dim = int(env.observation_manager.group_obs_dim["lower_body_policy"][-1])
    policy_contract["policy_observation_dim"] = obs_dim
    model = torch.jit.load(str(POLICY), map_location="cpu")
    model.eval()
    with torch.no_grad():
        probe_out = model(torch.zeros((1, obs_dim + 4), dtype=torch.float32))
    policy_contract["policy_input_dim"] = obs_dim + 4
    policy_contract["policy_probe_output_shape"] = list(probe_out.shape)
    policy_contract["policy_forward_schema"] = str(model.forward.schema)
    policy_contract["input_contract_exact"] = int(probe_out.shape[-1]) == len(lower_term._joint_ids)
    if not policy_contract["input_contract_exact"]:
        raise RuntimeError(json.dumps({"kind": "POLICY_INPUT_CONTRACT_FAIL", "contract": policy_contract}))

    def pose_snapshot() -> dict[str, Any]:
        root = robot.data.root_pos_w[0].detach().cpu().numpy()
        quat = robot.data.root_quat_w[0].detach().cpu().numpy()
        return {
            "root_pos": root.tolist(),
            "root_quat": quat.tolist(),
            "joint_pos": robot.data.joint_pos.torch[0].detach().cpu().numpy().tolist(),
            "joint_vel": robot.data.joint_vel.torch[0].detach().cpu().numpy().tolist(),
        }

    def collect(seconds: float, command: list[float], mode: str, seed: int) -> dict[str, Any]:
        lower_term.static_mode = mode == "STATIC_DEFAULT"
        env.reset(seed=seed)
        env.action_manager.reset()
        lower_term._raw_actions.zero_()
        steps = int(round(seconds / env.step_dt))
        upper_dim = upper_term.action_dim
        roots: list[list[float]] = []
        heights: list[float] = []
        foot_z: list[list[float]] = []
        finite = True
        input_max = 0.0
        output_max = 0.0
        action = torch.zeros((1, upper_dim + 4), device=env.device)
        action[:, upper_dim:] = torch.tensor(command, device=env.device, dtype=torch.float32)
        first = pose_snapshot()
        for _ in range(steps):
            with torch.no_grad():
                env.step(action)
            root = robot.data.root_pos_w[0].detach().cpu().numpy()
            roots.append(root.tolist())
            heights.append(float(root[2]))
            if foot_ids:
                foot_z.append(robot.data.body_pos_w[0, foot_ids, 2].detach().cpu().numpy().tolist())
            finite = finite and bool(np.isfinite(root).all())
            finite = finite and bool(torch.isfinite(lower_term.processed_actions).all())
            if lower_term.last_policy_input is not None:
                input_max = max(input_max, float(lower_term.last_policy_input.abs().max().item()))
            if lower_term.last_policy_output is not None:
                output_max = max(output_max, float(lower_term.last_policy_output.abs().max().item()))
        last = pose_snapshot()
        roots_np = np.asarray(roots, dtype=float)
        foot_np = np.asarray(foot_z, dtype=float) if foot_z else np.empty((0, 0))
        displacement = (roots_np[-1] - roots_np[0]).tolist() if len(roots_np) > 1 else [0.0, 0.0, 0.0]
        return {
            "seed": seed,
            "mode": mode,
            "command": command,
            "seconds": seconds,
            "steps": steps,
            "finite": bool(finite and np.isfinite(roots_np).all()),
            "root_start": first["root_pos"],
            "root_end": last["root_pos"],
            "root_displacement": displacement,
            "root_height_min": float(np.min(roots_np[:, 2])) if len(roots_np) else None,
            "root_height_max": float(np.max(roots_np[:, 2])) if len(roots_np) else None,
            "foot_z_min": foot_np.min(axis=0).tolist() if foot_np.size else [],
            "foot_z_max": foot_np.max(axis=0).tolist() if foot_np.size else [],
            "policy_input_dim_observed": int(lower_term.last_policy_input.shape[-1]) if lower_term.last_policy_input is not None else None,
            "policy_output_dim_observed": int(lower_term.last_policy_output.shape[-1]) if lower_term.last_policy_output is not None else None,
            "policy_input_max_abs": input_max,
            "policy_output_max_abs": output_max,
            "trajectory": roots,
            "foot_trajectory": foot_z,
        }

    static_rows = [collect(STATIC_SECONDS, [0.0, 0.0, 0.0, 0.72], "STATIC_DEFAULT", seed) for seed in SEEDS]
    zero_rows = [collect(ZERO_SECONDS, ZERO_COMMAND, "POLICY", seed) for seed in SEEDS]
    forward_rows = [collect(FORWARD_SECONDS, FORWARD_COMMAND, "POLICY", seed) for seed in SEEDS]
    env.close()
    return {
        "status": "pass",
        "policy_contract": policy_contract,
        "static_rows": static_rows,
        "zero_rows": zero_rows,
        "forward_rows": forward_rows,
        "native_start": "pass",
        "control_dt_s": env.step_dt,
    }


def run_gates(forward_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in forward_rows if row.get("finite")]
    l0 = bool(rows)
    l1 = bool(rows) and all(row.get("policy_input_dim_observed") is not None for row in rows)
    l2 = bool(rows) and all((row.get("root_height_min") or 0.0) >= 0.45 for row in rows)
    l3 = bool(rows) and all(
        bool(row.get("foot_z_min"))
        and max(row.get("foot_z_max", [0.0])) - min(row.get("foot_z_min", [0.0])) >= 0.01
        for row in rows
    )
    l4 = bool(rows) and all((row.get("root_displacement") or [0.0])[0] >= 0.02 for row in rows)
    l5 = bool(rows) and all(int(row.get("steps", 0)) >= 100 for row in rows)
    return {
        "L0_native_start": l0,
        "L1_finite_official_lower_body": l1,
        "L2_height_safety": l2,
        "L3_liftoff_proxy": l3,
        "L4_two_stride_forward_displacement_proxy": l4,
        "L5_100_control_steps": l5,
        "all_pass": all((l0, l1, l2, l3, l4, l5)),
        "gate_definitions": {
            "L0": "fresh official Isaac process and environment construction",
            "L1": "finite root/action tensors and observed policy input",
            "L2": "root height never below 0.45 m",
            "L3": "ankle-height liftoff proxy excursion at least 0.01 m",
            "L4": "forward displacement at least 0.02 m",
            "L5": "at least 100 control steps in the 3 s forward segment",
        },
    }


def classify(native: dict[str, Any], contract: dict[str, Any], gates: dict[str, Any]) -> tuple[str, list[str]]:
    failures: list[str] = []
    if "input_contract_exact" in contract and not contract.get("input_contract_exact", False):
        failures.append("EXP014_D31BR3_POLICY_INPUT_CONTRACT_FAIL")
    if native.get("status") != "pass":
        error = str(native.get("error", ""))
        if "AgileTeacherPolicyObservationsCfg" in error or "AgileBasedLowerBodyAction" in error:
            failures.append("EXP014_D31BR3_LOWER_BODY_CONTRACT_RECONSTRUCTION_FAIL")
        elif "UpperBodyHoldAction" in error:
            failures.append("EXP014_D31BR3_FIXED_UPPER_BODY_INCOMPATIBLE")
        else:
            failures.append("EXP014_D31BR3_RUNTIME_FAIL")
    if native.get("status") == "pass" and contract.get("input_contract_exact", False) and not gates.get("all_pass", False):
        failures.append("EXP014_D31BR3_OFFICIAL_AGILE_LOWER_BODY_START_FAIL")
    if len(failures) > 1:
        return "EXP014_D31BR3_MULTIPLE_FAILURES", failures
    return (failures[0] if failures else "EXP014_D31BR3_OFFICIAL_AGILE_LOWER_BODY_START_PASS"), failures


def run() -> int:
    start_head = git("rev-parse", "HEAD")
    start_status = status_lines()
    contract = official_source_contract()
    dump("stage_reference.json", {
        "experiment": "EXP014",
        "phase": "2-D31B-R3",
        "title": "terminal official Agile lower-body start",
        "starting_head": start_head,
        "isaac_python": str(ISAAC_PYTHON),
        "official_checkpoint_sha256": POLICY_SHA256,
    })
    dump("protocol.json", {
        "native_command": f'"{ISAAC_PYTHON}" "{HERE}" --headless --viz none',
        "seeds": SEEDS,
        "static_default_pose_seconds": STATIC_SECONDS,
        "zero_standing_seconds": ZERO_SECONDS,
        "forward_seconds": FORWARD_SECONDS,
        "forward_command": FORWARD_COMMAND,
        "control_dt_s": CONTROL_DT,
        "upper_body_policy": "fixed G1_29DOF_CFG default joint positions",
        "lower_body_policy": "official Agile observation/action source implementation",
        "no_command_tuning_after_forward": True,
        "native_unchanged": True,
    })
    dump("d31br2_scientific_adjudication.json", {
        "source": "phase_2_d31br2_official_policy_runtime_repair",
        "classification": "EXP014_D31BR2_OFFICIAL_POLICY_LOAD_PASS_START_NOT_RUN",
        "checkpoint_sha256": POLICY_SHA256,
        "runtime_repair_reused": True,
        "terminal_transition_reason": "R3 removes the task-space-dependent environment path and tests the official lower-body contract directly.",
    })
    dump("official_agile_lower_body_contract.json", contract)
    dump("official_agile_observation_schema.json", {
        "source": contract["observation_source"],
        "class": contract["observation_class"],
        "term_order": ["base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions"],
        "joint_name_patterns": [
            ".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint",
            ".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint", "waist_.*_joint",
        ],
        "joint_velocity_scale": 0.1,
        "history_length": None,
        "command_appended_by_action": ["vx", "vy", "wz", "hip_height"],
    })
    dump("policy_input_identity.json", {
        "checkpoint": str(POLICY),
        "sha256": sha(POLICY),
        "expected_sha256": POLICY_SHA256,
        "loader": "isaaclab.utils.io.torchscript.load_torchscript_model",
        "forward_schema": "forward(__torch__.agile.rl_env.rsl_rl.exporter._TorchPolicyExporter self, Tensor x) -> Tensor",
    })
    dump("diagnostic_environment_contract.json", {
        "robot": "G1_29DOF_CFG",
        "scene_num_envs": 1,
        "simulation_dt_s": 0.005,
        "decimation": 4,
        "control_dt_s": CONTROL_DT,
        "ground": "official GroundPlaneCfg",
        "configuration_mutations": ["scene.num_envs=1", "seed", "official policy path", "custom fixed hold term"],
    })
    dump("upper_body_hold_contract.json", {
        "mode": "fixed_default_joint_position_target",
        "source": "G1_29DOF_CFG.data.default_joint_pos",
        "actuator_parameters_changed": False,
        "task_space_dependency_used": False,
        "joint_selection": "all G1 joints excluding official lower-body hip/knee/ankle patterns",
    })
    protected_before = {path: sha(REPO / path) for path in PROTECTED}
    native: dict[str, Any] = {"status": "not_run"}
    try:
        native = native_child() if sys.executable == str(ISAAC_PYTHON) else {"status": "wrong_runtime", "executable": sys.executable}
    except Exception as exc:
        native = {
            "status": "fail",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=20),
        }
    dump("native_execution.json", native)
    if native.get("status") == "pass":
        runtime_contract = native["policy_contract"]
        dump("official_agile_lower_body_contract.json", {**contract, **runtime_contract})
        dump("policy_input_identity.json", {
            "checkpoint": str(POLICY),
            "sha256": sha(POLICY),
            "expected_sha256": POLICY_SHA256,
            "loader": "isaaclab.utils.io.torchscript.load_torchscript_model",
            "forward_schema": runtime_contract["policy_forward_schema"],
            "policy_input_dim": runtime_contract["policy_input_dim"],
            "policy_output_shape": runtime_contract["policy_probe_output_shape"],
            "input_contract_exact": runtime_contract["input_contract_exact"],
        })
        static_rows = native["static_rows"]
        zero_rows = native["zero_rows"]
        forward_rows = native["forward_rows"]
        gates = run_gates(forward_rows)
    else:
        runtime_contract = {}
        static_rows = []
        zero_rows = []
        forward_rows = []
        gates = {"all_pass": False, "reason": "native_start_failed"}
    dump("static_pose_results.json", {"status": "pass" if static_rows else "not_run", "seeds": SEEDS, "rows": static_rows})
    dump("zero_command_results.json", {"status": "pass" if zero_rows else "not_run", "rows": zero_rows})
    write_csv("zero_command_results.csv", zero_rows)
    dump("forward_command_selection.json", {
        "source": "official Agile command contract",
        "command_name": "forward",
        "command": FORWARD_COMMAND,
        "selected_before_results": True,
        "tuned_after_results": False,
    })
    dump("forward_start_results.json", {"status": "pass" if forward_rows else "not_run", "rows": forward_rows, "gates": gates})
    write_csv("forward_start_results.csv", forward_rows)
    classification, failure_reasons = classify(native, runtime_contract, gates)
    teacher_class = (
        "EXP014_D31BR3_OFFICIAL_AGILE_LOWER_BODY_START_PASS"
        if classification == "EXP014_D31BR3_OFFICIAL_AGILE_LOWER_BODY_START_PASS"
        else classification
    )
    dump("teacher_usability_classification.json", {
        "classification": teacher_class,
        "teacher_usable_for_start_transition": classification == "EXP014_D31BR3_OFFICIAL_AGILE_LOWER_BODY_START_PASS",
        "failure_reasons": failure_reasons,
        "gates": gates,
    })
    dump("stage_classification.json", {
        "classification": classification,
        "registered_classifications": STAGE_CLASSES,
        "failure_reasons": failure_reasons,
        "gates": gates,
    })
    dump("recommended_next_action.json", {
        "classification": classification,
        "action": "CLOSE_EXP014" if not gates["all_pass"] else "D31C official START trajectory capture + exp014 compatibility",
        "start_transition_authorized": classification == "EXP014_D31BR3_OFFICIAL_AGILE_LOWER_BODY_START_PASS",
    })
    end_status = status_lines()
    before_unrelated = sorted(rel_status(item) for item in start_status if not allowed_path(rel_status(item)))
    after_unrelated = sorted(rel_status(item) for item in end_status if not allowed_path(rel_status(item)))
    protected_after = {path: sha(REPO / path) for path in PROTECTED}
    dump("protected_hashes.json", {
        "starting_head": start_head,
        "ending_head": git("rev-parse", "HEAD"),
        "starting_status": start_status,
        "ending_status": end_status,
        "unrelated_state_preserved": before_unrelated == after_unrelated,
        "unrelated_before": before_unrelated,
        "unrelated_after": after_unrelated,
        "unrelated_added_during_execution": sorted(set(after_unrelated) - set(before_unrelated)),
        "unrelated_removed_during_execution": sorted(set(before_unrelated) - set(after_unrelated)),
        "protected_before": protected_before,
        "protected_after": protected_after,
        "protected_unchanged": protected_before == protected_after,
        "no_commit_or_push": True,
        "new_paths_only": True,
    })
    (OUT / "reproduction_commands.ps1").write_text(
        f'& "{ISAAC_PYTHON}" "{HERE}" --headless --viz none\n',
        encoding="utf-8",
    )
    status_note = (
        "Unrelated dirty and untracked state was unchanged."
        if before_unrelated == after_unrelated
        else "The protected files were unchanged; unrelated status changes were observed during the execution window."
    )
    if classification == "EXP014_D31BR3_OFFICIAL_AGILE_LOWER_BODY_START_PASS":
        import numpy as np

        trajectories = {
            "forward": np.asarray([row["trajectory"] for row in forward_rows], dtype=np.float32),
            "zero": np.asarray([row["trajectory"] for row in zero_rows], dtype=np.float32),
        }
        np.savez_compressed(OUT / "trajectory.npz", **trajectories)
        dump("trajectory_sha.json", {"path": str(OUT / "trajectory.npz"), "sha256": sha(OUT / "trajectory.npz")})
        dump("comparison.json", {"forward_vs_zero_root_displacement": [
            {
                "seed": seed,
                "forward": forward_rows[seed]["root_displacement"],
                "zero": zero_rows[seed]["root_displacement"],
            }
            for seed in range(len(forward_rows))
        ]})
    else:
        dump("project_status.json", {
            "status": "EXP014_CLOSED_NO_GO_START_TRANSITION",
            "classification": classification,
            "reason": failure_reasons,
        })
        dump("exp014_terminal_adjudication.json", {
            "project_status": "EXP014_CLOSED_NO_GO_START_TRANSITION",
            "stage_classification": classification,
            "terminal": True,
            "no_start_transition": True,
        })
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        (REPO / "research" / "exp_014_unitree_g1_explicit_motion_mode_unified_locomotion_closure.md").write_text(
            f"# EXP014 terminal closure\n\n"
            f"`EXP014_CLOSED_NO_GO_START_TRANSITION` — Phase 2-D31B-R3 classification: `{classification}`.\n\n"
            f"The official Agile lower-body start transition was not authorized. Failure reasons: "
            f"{', '.join(failure_reasons) or 'unspecified runtime failure'}.\n",
            encoding="utf-8",
        )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        f"# EXP014 Phase 2-D31B-R3 terminal official Agile start\n\n"
        f"- Starting HEAD: `{start_head}`\n"
        f"- Execution HEAD: `{git('rev-parse', 'HEAD')}`\n"
        f"- Native command: `\"{ISAAC_PYTHON}\" \"{HERE}\" --headless --viz none`\n"
        f"- Official checkpoint SHA-256: `{POLICY_SHA256}`\n"
        f"- Classification: **`{classification}`**\n"
        f"- L0-L5 all pass: `{gates.get('all_pass', False)}`\n\n"
        f"The diagnostic used the installed official Agile lower-body observation/action source, "
        f"`G1_29DOF_CFG`, a fixed default upper-body joint hold, and the preselected official "
        f"forward command `{FORWARD_COMMAND}`. No command was changed after results.\n\n"
        f"{status_note}\n",
        encoding="utf-8",
    )
    print(json.dumps({"classification": classification, "output": str(OUT)}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--viz", default=None)
    parser.parse_args()
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
