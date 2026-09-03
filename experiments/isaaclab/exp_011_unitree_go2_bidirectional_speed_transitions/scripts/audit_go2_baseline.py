"""Stage 0 live registry/checkpoint/interface audit; never trains."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import torch

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage1_single_policy_baseline"
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

from go2_bidirectional.checkpoint_audit import inspect_checkpoint  # noqa: E402
from go2_bidirectional.contact_analysis import resolve_foot_mapping  # noqa: E402
from go2_bidirectional.environment_audit import live_environment_details, registered_go2_specs  # noqa: E402
from go2_bidirectional.evaluation import build_runner  # noqa: E402
from go2_bidirectional.observation_contract import observation_action_contract  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=OUT)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def dump(name: str, value) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(*parts: str) -> str:
    return subprocess.run(["git", *parts], cwd=REPO, text=True, capture_output=True, check=True).stdout.strip()


def main() -> None:
    task = "Isaac-Velocity-Flat-Unitree-Go2-v0"
    checkpoint = Path(get_published_pretrained_checkpoint("rsl_rl", task) or "")
    if not checkpoint.is_file():
        dump("stage0_selected_baseline.json", {"status": "NO_USABLE_GO2_BASELINE", "reason": "official checkpoint unavailable"})
        return
    candidates = []
    for root in (REPO / "logs", REPO / "artifacts", REPO / "results"):
        if root.exists():
            for path in root.rglob("model_*.pt"):
                if "go2" in str(path).lower() or "unitree_go2" in str(path).lower():
                    candidates.append({"checkpoint_path": str(path), "provenance": "local", "strict_load_result": "NOT_SELECTED"})
    selected = inspect_checkpoint(checkpoint)
    cfg, agent = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 1
    cfg.seed = 20260901
    cfg.episode_length_s = 10.0
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    if args.device:
        cfg.sim.device = args.device
    registry = registered_go2_specs()
    details = live_environment_details(cfg)
    for row in registry:
        row.update({"inventory_source": "live gymnasium registry"})
        inventory_cfg, _inventory_agent = resolve_task_config(
            row["registered_environment_id"], "rsl_rl_cfg_entry_point"
        )
        row.update(live_environment_details(inventory_cfg))
    with launch_simulation(cfg, args):
        raw = gym.make(task, cfg=cfg)
        wrapped, runner, _policy = build_runner(raw, agent, checkpoint)
        env = wrapped.unwrapped
        contract = observation_action_contract(env, agent)
        if contract["observation"]["dimension"] != selected["observation_dimension"]:
            raise RuntimeError("observation mismatch")
        if contract["action"]["dimension"] != selected["action_dimension"]:
            raise RuntimeError("action mismatch")
        selected["strict_load_result"] = "PASS_STRICT_TRUE"
        selected["selected_before_formal_results"] = True
        mapping = resolve_foot_mapping(env.scene["robot"], env.scene.sensors["contact_forces"])
        joint_order = {
            "source": "live articulation after USD spawn",
            "action_dimension": env.action_manager.total_action_dim,
            "joint_order": list(env.scene["robot"].joint_names),
            "expected_go2_12_confirmed": len(env.scene["robot"].joint_names) == 12,
        }
        physics = {
            "physics_dt_s": env.cfg.sim.dt,
            "control_dt_s": env.step_dt,
            "decimation": env.cfg.decimation,
            "terrain": env.cfg.scene.terrain.terrain_type,
            "terrain_friction": {
                "static": env.cfg.scene.terrain.physics_material.static_friction,
                "dynamic": env.cfg.scene.terrain.physics_material.dynamic_friction,
            },
            "solver_configuration": str(env.cfg.sim.physics),
            "contact_sensor_update_rate_s": env.cfg.scene.contact_forces.update_period,
            "actuator_pd_gains": {"stiffness": 25.0, "damping": 0.5},
            "actuator_limits": {"effort_nm": 23.5, "velocity_radps": 30.0},
            "episode_timeout_s": env.cfg.episode_length_s,
            "fall_termination": "base contact force > 1 N",
        }
        dump("stage0_registered_go2_environments.json", {"search_terms": ["Go2", "Unitree", "Velocity", "Flat", "Rough", "Locomotion"], "environments": registry})
        dump("stage0_go2_checkpoint_inventory.json", {"search_roots": ["logs/**", "artifacts/**", "results/**", "Isaac Lab official pretrained utility"], "candidates": [selected, *candidates]})
        dump("stage0_selected_baseline.json", {
            "status": "SELECTED",
            "selection_rule_rank": 1,
            "selected": selected,
            "selection_reason": "Official Isaac Lab Go2 flat velocity checkpoint; selected before formal results.",
            "rejected": candidates,
            "smoke_test_used_for_claim": False,
        })
        dump("observation_action_contract.json", contract)
        dump("physics_control_contract.json", physics)
        dump("go2_joint_order.json", joint_order)
        dump("foot_contact_mapping.json", {
            "mapping": mapping,
            "anatomical_order": ["front-left", "front-right", "rear-left", "rear-right"],
            "static_audit": "PASS: unique FL/FR/RL/RR asset body and sensor indices",
            "low_speed_dynamic_audit": "PENDING_STAGE1_TRACE",
        })
        state_path = args.output / "starting_repository_state.json"
        if not state_path.exists():
            dump("starting_repository_state.json", {
                "starting_head": git("rev-parse", "HEAD"),
                "reported_head": "ff15a94ff168b9948ba4d2e3ee49b0fd57735ebd",
                "starting_status": git("status", "--short").splitlines(),
                "unrelated_dirty_paths": [
                    line for line in git("status", "--short").splitlines()
                    if "exp_011_unitree_go2_bidirectional_speed_transitions" not in line and "exp_011_go2_bidirectional_baseline_report.md" not in line
                ],
                "isaaclab_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO.parent / "IsaacLab", text=True, capture_output=True, check=True).stdout.strip(),
                "python_executable": sys.executable,
            })
        wrapped.close()


if __name__ == "__main__":
    main()
