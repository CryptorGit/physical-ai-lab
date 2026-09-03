"""Fresh D26V endpoint-window safety capture.

This is a D26V-only diagnostic replay.  It imports the D26U evaluator without
changing it, repeats the same eight reset recipes with the same seed, and
records the per-control-step safety metrics that D26U did not persist.  It
does not replace the D26U source bundle and does not run model-based START
physics, PPO, CEM, validation, held-out, or RUN integration.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
EXP = HERE.parent.parent
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26v_endpoint_gate_and_wbik_v2"
RAW_D26U = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26u_fresh_source_and_offline_execution/raw_d26u_capture"
SEED = 20279941
RECIPES = list(range(8))
SOURCE_STEPS = [125, 132, 129, 125, 128, 125, 130, 129]
CONFIRMATION_END = [75, 82, 79, 75, 78, 75, 80, 79]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


d26u = load_module("exp014_d26u_runtime_for_d26v", EXP / "scripts/run_phase2_d26u_capture.py")


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def tensor_list(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return np.asarray(value).tolist()


def run(mode: str, args, hydra) -> dict:
    print(json.dumps({"d26v_capture": "start", "mode": mode, "repo": str(REPO), "out": str(OUT)}, sort_keys=True), flush=True)
    # The D26U runtime resolves the task through the same Hydra arguments.
    sys.argv = [sys.argv[0], *hydra]
    cfg, agent = d26u.resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 8
    cfg.seed = SEED
    cfg.episode_length_s = 20.0
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    if args.device:
        cfg.sim.device = agent.device = args.device
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    timeline = {"main_step": 0, "pending_extra": False, "call_index": 0, "last": None}
    records: list[dict] = []

    with d26u.launch_simulation(cfg, args):
        wrapped = d26u.RslRlVecEnvWrapper(
            d26u.gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg),
            clip_actions=agent.clip_actions,
        )
        world = d26u.d3.StandWorld(wrapped, d26u.d3.load_resets(), torch.zeros(680))
        actor = d26u.d3.initialize("P0_STAND_PARENT", world.device)[0].eval()
        masses = world.robot.root_physx_view.get_masses()
        if not torch.is_tensor(masses):
            masses = torch.as_tensor(masses, device=world.device, dtype=torch.float32)
        masses = d26u.as_batch(masses.to(world.device), n=8)

        original_step = world.step
        original_runtime = d26u.runtime_arrays

        def wrapped_step(*step_args, **step_kwargs):
            if timeline["pending_extra"]:
                kind = "extra_after_endpoint"
                timeline["pending_extra"] = False
            else:
                timeline["main_step"] += 1
                kind = "main_control_step"
            result = original_step(*step_args, **step_kwargs)
            timeline["call_index"] += 1
            timeline["last"] = {"kind": kind, "main_step": timeline["main_step"], "call_index": timeline["call_index"]}
            if kind == "main_control_step" and timeline["main_step"] in SOURCE_STEPS:
                # D26U calls one extra step immediately after each endpoint
                # capture.  This keeps the diagnostic main-step numbering
                # aligned with the frozen source-control-step metadata.
                timeline["pending_extra"] = True
            return result

        world.step = wrapped_step

        def wrapped_runtime(*runtime_args, **runtime_kwargs):
            state, metrics = original_runtime(*runtime_args, **runtime_kwargs)
            if runtime_kwargs.get("update_safety", True) and timeline["last"] is not None:
                rec = {
                    "kind": timeline["last"]["kind"],
                    "main_step": timeline["last"]["main_step"],
                    "call_index": timeline["last"]["call_index"],
                    "fall": tensor_list(metrics["fall"]),
                    "dangerous_slip": tensor_list(metrics["dangerous_slip"]),
                    "impact": tensor_list(metrics["impact"]),
                    "velocity_saturation": tensor_list(metrics["velocity_saturation"]),
                    "torque_saturation": tensor_list(metrics["torque_saturation"]),
                    "support_loss": tensor_list(metrics["support_loss"]),
                    "nonfinite": tensor_list(metrics["nonfinite"]),
                    "support_count": tensor_list(metrics["support_count"]),
                    "velocity_ratio_max": np.asarray(tensor_list(metrics["velocity_ratio"]), dtype=float).max(axis=1).tolist(),
                    "torque_ratio_max": np.asarray(tensor_list(metrics["torque_ratio"]), dtype=float).max(axis=1).tolist(),
                    "computed_torque": tensor_list(state["computed_torque"]),
                    "applied_torque": tensor_list(state["applied_torque"]),
                    "effort_limits": tensor_list(state["effort_limits"]),
                    "joint_pos": tensor_list(state["joint_pos"]),
                    "joint_vel": tensor_list(state["joint_vel"]),
                    "contact_force": tensor_list(state["contact_force"]),
                    "root_pose": tensor_list(state["root_pose"]),
                    "root_velocity": tensor_list(state["root_velocity"]),
                    "support_state": tensor_list(state["support_state"]),
                }
                records.append(rec)
            return state, metrics

        d26u.runtime_arrays = wrapped_runtime
        result, _ = d26u.evaluate_pass(world, actor, masses, RECIPES, capture_enabled=(mode == "on"))
        print(json.dumps({"d26v_capture": "evaluate_complete", "mode": mode, "source_captured": result.get("source_captured")}, sort_keys=True), flush=True)
        # The launch context owns simulator teardown.  Calling the vec wrapper
        # close here can terminate the Kit process before the D26V-only JSON
        # is flushed on this Windows IsaacLab build.
        # Persist before leaving the Kit context: on this Windows build the
        # context teardown may terminate the interpreter immediately.
        windows = []
        for recipe, source_step, confirm_step in zip(RECIPES, SOURCE_STEPS, CONFIRMATION_END):
            lo = source_step - 49
            selected = [r for r in records if r["kind"] == "main_control_step" and lo <= r["main_step"] <= source_step]
            windows.append({
                "recipe_id": recipe,
                "source_control_step": source_step,
                "confirmation_end_step": confirm_step,
                "window_start_step": lo,
                "window_end_step": source_step,
                "window_record_count": len(selected),
                "records": selected,
            })
        output = {
            "name": "Exp014D26VFreshEndpointWindowCaptureV1",
            "capture_mode": mode,
            "seed": SEED,
            "recipes": RECIPES,
            "source_steps_from_protected_d26u_metadata": SOURCE_STEPS,
            "confirmation_end_steps_from_protected_d26u_metadata": CONFIRMATION_END,
            "lifecycle": "same Exp014FreshS_HOLDSourceLifecycleV2 fresh reset-recipe evaluator; no raw state restore",
            "d26u_bundle_read_only": True,
            "source_bundle": "phase_2_d26u_fresh_source_and_offline_execution/fresh_shold_identity_complete_sources.npz",
            "window_contract": "last 50 main control steps inclusive of endpoint control step; extra next-action step excluded",
            "records": records,
            "windows": windows,
            "source_captured": result["source_captured"],
            "source_control_step": result["source_control_step"],
            "persistent_update": 0,
            "physics_start": 0,
            "model_based_start_physics": 0,
            "raw_snapshot_restore": 0,
        }
        dump(OUT / f"raw_endpoint_window_capture_{mode}.json", output)
        print(json.dumps({"d26v_capture": "written", "path": str(OUT / f"raw_endpoint_window_capture_{mode}.json"), "records": len(records)}, sort_keys=True), flush=True)
        return output

    windows = []
    for recipe, source_step, confirm_step in zip(RECIPES, SOURCE_STEPS, CONFIRMATION_END):
        lo = source_step - 49
        selected = [r for r in records if r["kind"] == "main_control_step" and lo <= r["main_step"] <= source_step]
        windows.append({
            "recipe_id": recipe,
            "source_control_step": source_step,
            "confirmation_end_step": confirm_step,
            "window_start_step": lo,
            "window_end_step": source_step,
            "window_record_count": len(selected),
            "records": selected,
        })
    output = {
        "name": "Exp014D26VFreshEndpointWindowCaptureV1",
        "capture_mode": mode,
        "seed": SEED,
        "recipes": RECIPES,
        "source_steps_from_protected_d26u_metadata": SOURCE_STEPS,
        "confirmation_end_steps_from_protected_d26u_metadata": CONFIRMATION_END,
        "lifecycle": "same Exp014FreshS_HOLDSourceLifecycleV2 fresh reset-recipe evaluator; no raw state restore",
        "d26u_bundle_read_only": True,
        "source_bundle": "phase_2_d26u_fresh_source_and_offline_execution/fresh_shold_identity_complete_sources.npz",
        "window_contract": "last 50 main control steps inclusive of endpoint control step; extra next-action step excluded",
        "records": records,
        "windows": windows,
        "source_captured": result["source_captured"],
        "source_control_step": result["source_control_step"],
        "persistent_update": 0,
        "physics_start": 0,
        "model_based_start_physics": 0,
        "raw_snapshot_restore": 0,
    }
    dump(OUT / f"raw_endpoint_window_capture_{mode}.json", output)
    print(json.dumps({"d26v_capture": "written", "path": str(OUT / f"raw_endpoint_window_capture_{mode}.json"), "records": len(records)}, sort_keys=True), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-mode", choices=("off", "on"), required=True)
    d26u.add_launcher_args(parser)
    args, hydra = d26u.setup_preset_cli(parser)
    run(args.capture_mode, args, hydra)


if __name__ == "__main__":
    main()
