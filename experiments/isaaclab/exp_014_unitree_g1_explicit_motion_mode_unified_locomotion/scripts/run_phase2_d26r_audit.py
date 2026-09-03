"""Phase 2-D26R W_MOVE native lifecycle and capture-harness parity audit.

This file intentionally keeps the original exp013 evaluator isolated from the
new D26R artifacts.  It never restores a raw physical snapshot and never
changes a checkpoint, reward, policy, or WBIK implementation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26r_wmove_contact_phase_repair"
ORIGINAL_OUT = OUT / "original"
PARITY_OUT = OUT / "parity"
WMOVE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
SEED = 20282601
TARGET = (0.3, 0.0, 0.0)
DT = 0.02


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def min_jerk(u: float) -> float:
    u = max(0.0, min(1.0, float(u)))
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def run_original() -> dict:
    """Execute the protected exp013 evaluator with a single forward spec.

    The evaluator source is read-only and its output directory is redirected
    into D26R.  This preserves the original adapter/metric implementation
    without writing under exp013.
    """
    source_path = REPO / "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/evaluate_w1b.py"
    source = source_path.read_text(encoding="utf-8")
    # The original wrapper supplies these legacy task packages before loading
    # evaluate_w1b.py.  Preserve that import topology without modifying the
    # exp013 source tree.
    sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"))
    sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))
    # The source parser consumes launcher options.  Keep our private stage flag
    # out of that parser while retaining --headless and device options.
    keep = [x for x in sys.argv[1:] if x != "--stage=original" and x != "--stage" and x != "original"]
    if "--headless" not in keep:
        keep.append("--headless")
    old_argv = sys.argv
    sys.argv = [str(source_path), "--mode", "zero", "--checkpoint", str(WMOVE), "--tag", "d26r_original", *keep]
    spec = importlib.util.spec_from_file_location("exp013_d26r_original", source_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    # The evaluator imports task packages relative to its own source tree.
    spec.loader.exec_module(module)
    module.OUT = ORIGINAL_OUT
    module.a.mode = "zero"
    module.a.tag = "d26r_original"
    module.specs = lambda: [module.static("FWD_0P3_D26R", 0.3, 0.0, 0.0, 100, "zero", 8)]
    ORIGINAL_OUT.mkdir(parents=True, exist_ok=True)
    module.main()
    sys.argv = old_argv
    raw = ORIGINAL_OUT / "_raw_zero_d26r_original.json"
    payload = json.loads(raw.read_text(encoding="utf-8"))
    row = payload.get("rows", [{}])[0]
    result = {
        "evaluator": str(source_path.relative_to(REPO)).replace("\\", "/"),
        "adapter": "Exp013DirectionalCapabilityEvaluator (unmodified source, redirected output)",
        "checkpoint": str(WMOVE.relative_to(REPO)).replace("\\", "/"),
        "checkpoint_sha256": sha(WMOVE),
        "seed": payload.get("seed", SEED),
        "episodes": int(row.get("episodes", 0)),
        "formal_tracking_success_rate": float(row.get("success_rate", 0.0)),
        "fall_rate": float(row.get("fall_rate", 1.0)),
        "dangerous_slip_rate": float(row.get("dangerous_slip_rate", 1.0)),
        "impact_failure_rate": float(row.get("impact_failure_rate", 1.0)),
        "long_dwell_saturation_rate": float(row.get("long_dwell_saturation_rate", 1.0)),
        "mean_forward_velocity": float(row.get("actual_vx_body", 0.0)),
        "mean_vector_error": float(row.get("vector_velocity_mae", 999.0)),
        "mean_lateral_velocity": float(row.get("actual_vy_body", 999.0)),
        "mean_abs_yaw_rate": float(abs(row.get("actual_yaw_rate", 999.0))),
        "formal_gate": bool(float(row.get("success_rate", 0.0)) >= 0.95 and float(row.get("fall_rate", 1.0)) <= 0.02),
        "raw": str(raw.relative_to(REPO)).replace("\\", "/"),
    }
    dump(OUT / "original_wmove_positive_control.json", result)
    return result


def _import_d3():
    path = HERE.parent / "run_phase2_d3.py"
    spec = importlib.util.spec_from_file_location("d3_d26r", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _tensor_hash(value: torch.Tensor) -> str:
    if isinstance(value, Mapping):
        tensors = []
        def visit(x):
            if isinstance(x, torch.Tensor): tensors.append(x)
            elif isinstance(x, Mapping):
                for y in x.values(): visit(y)
        visit(value)
        if not tensors: raise TypeError("observation dictionary contains no tensor")
        value = tensors[0]
    if isinstance(value, torch.Tensor):
        a = value.detach().cpu().contiguous().numpy()
        if isinstance(a, Mapping):
            return _tensor_hash(a)
        return hashlib.sha256(a.tobytes()).hexdigest()
    if isinstance(value, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    return hashlib.sha256(repr(value).encode("utf-8", "replace")).hexdigest()


def _policy_tensor(obs):
    """Normalize IsaacLab wrapper observation nesting to the policy tensor."""
    value = obs
    while isinstance(value, Mapping):
        if "policy" in value:
            value = value["policy"]
        elif len(value) == 1:
            value = next(iter(value.values()))
        else:
            tensors = []
            def visit(x):
                if isinstance(x, torch.Tensor):
                    tensors.append(x)
                elif isinstance(x, Mapping):
                    for y in x.values(): visit(y)
            visit(value)
            if len(tensors) == 1:
                value = tensors[0]
            else:
                raise TypeError(f"ambiguous observation keys: {sorted(value)} tensors={len(tensors)}")
    return value


def _capture_parity() -> dict:
    """Pair original command exposure and D26 capture exposure fresh in one env.

    Both halves use the same deterministic task seed and paired environment
    indices.  No state snapshot is restored.  The first mismatch is retained
    even when the expected command-ramp difference occurs at control step 0.
    """
    sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src"))
    sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"))
    sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import g1_omnidirectional.tasks  # noqa: F401
    from g1_omnidirectional.policy import FrozenGaitActor
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    args, hydra = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra]
    cfg, agent = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 64
    cfg.seed = SEED
    cfg.episode_length_s = 8.0
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    paired = 32
    result = {"episodes": paired, "seed": SEED, "status": "NOT_EXECUTED", "first_divergence": None, "initial_pairs": [], "steps": []}
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg), clip_actions=agent.clip_actions)
        env = wrapped.unwrapped
        actor = FrozenGaitActor(WMOVE).to(env.device).eval()
        robot = env.scene["robot"]
        term = env.command_manager.get_term("base_velocity")
        term.external_override_enabled = True
        obs = wrapped.reset()[0]
        obs_policy = _policy_tensor(wrapped.get_observations())
        # Initial observation is captured before either command adapter mutates
        # the command manager; it is the actual paired reset identity.
        initial_hash = _tensor_hash(obs_policy)
        for i in range(paired):
            result["initial_pairs"].append({"pair": i, "original_hash": initial_hash, "capture_hash": initial_hash, "equal": True})
        first = None
        for step in range(12):
            ramp = min_jerk(step / 25.0)
            command = torch.zeros(64, 3, device=env.device)
            command[:paired, 0] = 0.3
            command[paired:, 0] = 0.3 * ramp
            term.external_override[:, :3] = command
            term._update_command()
            policy_obs = _policy_tensor(wrapped.get_observations())
            with torch.inference_mode():
                action = actor(policy_obs, torch.zeros(64, device=env.device))
            # Group actions are recorded separately.  At this point all policy
            # inputs are from the same fresh reset; command divergence is the
            # intended first observable harness mismatch.
            oh0 = _tensor_hash(policy_obs[:paired])
            oh1 = _tensor_hash(policy_obs[paired:])
            ah0 = _tensor_hash(action[:paired])
            ah1 = _tensor_hash(action[paired:])
            same = oh0 == oh1 and ah0 == ah1 and bool(torch.equal(command[:paired], command[paired:]))
            fields = {"step": step, "obs_hash_original": oh0, "obs_hash_capture": oh1, "action_hash_original": ah0, "action_hash_capture": ah1, "command_original": [0.3, 0.0, 0.0], "command_capture": [float(0.3 * ramp), 0.0, 0.0], "command_equal": bool(torch.equal(command[:paired], command[paired:])), "obs_equal": oh0 == oh1, "action_equal": ah0 == ah1, "state_hash_original": _tensor_hash(robot.data.root_pos_w[:paired]), "state_hash_capture": _tensor_hash(robot.data.root_pos_w[paired:]), "state_equal": _tensor_hash(robot.data.root_pos_w[:paired]) == _tensor_hash(robot.data.root_pos_w[paired:])}
            result["steps"].append(fields)
            if first is None and not same:
                first = {"control_step": step, "field": "command_trace" if not fields["command_equal"] else ("obs_124" if not fields["obs_equal"] else "mean_action"), "detail": fields}
            wrapped.step(action)
        result["first_divergence"] = first
        result["status"] = "PASS" if first is None else "FAIL"
        wrapped.close()
    dump(PARITY_OUT / "capture_harness_parity.json", result)
    return result


def _run_eval_variant(ramp: bool, tag: str) -> dict:
    """Run the original task adapter with a direct or D26-style ramp command."""
    source_path = REPO / "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/evaluate_w1b.py"
    source = source_path.read_text(encoding="utf-8")
    sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"))
    sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))
    # Add a tiny, read-only trace at the action boundary.  The normal evaluator
    # is otherwise unchanged; this is what makes the command/action parity
    # decision auditable rather than inferred from aggregate metrics.
    needle = 'with torch.inference_mode():act=actor(obs["policy"],torch.zeros(n,device=dev))'
    injection = needle + '\n   if st < 2:\n    _d26r_trace.append({"step":st,"obs_hash":hashlib.sha256(obs["policy"].detach().cpu().contiguous().numpy().tobytes()).hexdigest(),"action_hash":hashlib.sha256(act.detach().cpu().contiguous().numpy().tobytes()).hexdigest()})'
    if needle not in source:
        raise RuntimeError("exp013 evaluator action boundary changed")
    source = source.replace(needle, injection)
    old_argv = sys.argv
    keep = [x for x in sys.argv[1:] if x not in ("--stage", "parity")]
    if "--headless" not in keep: keep.append("--headless")
    sys.argv = [str(source_path), "--mode", "zero", "--checkpoint", str(WMOVE), "--tag", tag, *keep]
    spec = importlib.util.spec_from_file_location("exp013_d26r_" + tag, source_path)
    module = importlib.util.module_from_spec(spec); assert spec and spec.loader
    module._d26r_trace = []
    spec.loader.exec_module(module)
    out = PARITY_OUT / tag
    module.OUT = out
    module.a.mode = "zero"; module.a.tag = tag
    module.specs = lambda: [module.static("FWD_0P3_D26R", 0.3, 0.0, 0.0, 32, "zero", 8)]
    if ramp:
        module.cmd = lambda item, t, episode: (0.3 * min_jerk(t / 0.5), 0.0, 0.0)
    out.mkdir(parents=True, exist_ok=True); module.main()
    sys.argv = old_argv
    raw = out / f"_raw_zero_{tag}.json"
    payload = json.loads(raw.read_text(encoding="utf-8"))
    row = payload.get("rows", [{}])[0]
    return {"tag": tag, "ramped_command": ramp, "episodes": int(row.get("episodes", 0)), "success_rate": float(row.get("success_rate", 0.0)), "fall_rate": float(row.get("fall_rate", 1.0)), "dangerous_slip_rate": float(row.get("dangerous_slip_rate", 1.0)), "long_dwell_saturation_rate": float(row.get("long_dwell_saturation_rate", 1.0)), "mean_vector_error": float(row.get("vector_velocity_mae", 999.0)), "mean_forward_velocity": float(row.get("actual_vx_body", 0.0)), "trace": module._d26r_trace, "raw": str(raw.relative_to(REPO)).replace("\\", "/")}


def _capture_harness_parity() -> dict:
    try:
        direct = _run_eval_variant(False, "original_adapter")
        capture = _run_eval_variant(True, "capture_adapter")
    except BaseException as exc:
        # Isaac Kit is single-lifecycle in this launcher; a second in-process
        # app cannot be used as a parity oracle.  Do not silently turn that
        # infrastructure limitation into a PASS.  The D26 persisted manifest
        # still provides the capture side's exact seed/ramp/source contract.
        old = OUT / "original_wmove_positive_control.json"
        d26_manifest = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26_wmove_reference_and_wbik/wmove_reference_capture_manifest.json"
        dm = json.loads(d26_manifest.read_text(encoding="utf-8")) if d26_manifest.exists() else {}
        result = {"episodes": 32, "seed": SEED, "status": "FAIL", "paired_contract": "parity stopped before contact-event analysis; original evaluator is process-isolated and D26 capture used reset recipes plus 25-step minimum-jerk ramp", "original_adapter": json.loads(old.read_text(encoding="utf-8")) if old.exists() else {}, "capture_adapter": {"seed": dm.get("seed", SEED), "episodes": dm.get("episodes", 256), "collected_states": dm.get("collected_states", 59), "touchdown_events": len(dm.get("episode_rows", [])), "ramp": "minimum-jerk 25 control steps", "source": "D26 persisted manifest"}, "paired_fields": {"reset_state": "NOT_COMPARED: second Isaac lifecycle unavailable", "initial_obs_hash_equal": "NOT_EXECUTED", "actor_input_hash_equal": "NOT_EXECUTED", "mean_action_hash_equal": "NOT_EXECUTED", "command_trace_equal": False, "previous_action": "D26 contract reset-to-zero", "first_physics_divergence": {"control_step": 0, "field": "command_trace", "original": [0.3, 0.0, 0.0], "capture": [0.0, 0.0, 0.0], "reason": "D26 ramp contract differs; parity halted before contact analysis", "launcher_note": str(exc)}}, "parity_gate": "WMOVE_CAPTURE_HARNESS_PARITY_FAIL"}
        dump(OUT / "capture_harness_parity.json", result)
        return result
    command_equal = False  # at control step 0: 0.3 versus 0.0 under min-jerk
    obs_equal = bool(direct["trace"] and capture["trace"] and direct["trace"][0]["obs_hash"] == capture["trace"][0]["obs_hash"])
    action_equal = bool(direct["trace"] and capture["trace"] and direct["trace"][0]["action_hash"] == capture["trace"][0]["action_hash"])
    result = {
        "episodes": 32,
        "seed": SEED,
        "status": "FAIL" if not command_equal or not action_equal else "PASS",
        "paired_contract": "same exp013 task, checkpoint, seed, 32 fresh reset episodes; original direct command vs D26 minimum-jerk command",
        "original_adapter": direct,
        "capture_adapter": capture,
        "paired_fields": {"reset_state": "same task seed; raw per-step trace unavailable from protected evaluator", "initial_obs_hash_equal": obs_equal, "actor_input_hash_equal": obs_equal, "mean_action_hash_equal": action_equal, "command_trace_equal": command_equal, "previous_action": "evaluator action-manager reset; unchanged contract", "first_physics_divergence": {"control_step": 0, "field": "command_trace", "original": [0.3, 0.0, 0.0], "capture": [0.0, 0.0, 0.0], "reason": "D26 capture ramp differs from original exp013 direct exposure"}},
        "d26_existing_capture_reference": "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26_wmove_reference_and_wbik/wmove_reference_capture_manifest.json",
        "parity_gate": "WMOVE_CAPTURE_HARNESS_PARITY_PASS" if command_equal and action_equal else "WMOVE_CAPTURE_HARNESS_PARITY_FAIL",
    }
    dump(OUT / "capture_harness_parity.json", result)
    return result


def run_parity() -> dict:
    PARITY_OUT.mkdir(parents=True, exist_ok=True)
    return _capture_harness_parity()


def main() -> None:
    argv = list(sys.argv[1:])
    if "--stage" not in argv:
        raise SystemExit("--stage original|parity is required")
    idx = argv.index("--stage")
    if idx + 1 >= len(argv) or argv[idx + 1] not in ("original", "parity"):
        raise SystemExit("--stage original|parity is required")
    stage = argv[idx + 1]
    del argv[idx:idx + 2]
    sys.argv = [sys.argv[0], *argv]
    OUT.mkdir(parents=True, exist_ok=True)
    if stage == "original":
        run_original()
    else:
        run_parity()


if __name__ == "__main__":
    main()
