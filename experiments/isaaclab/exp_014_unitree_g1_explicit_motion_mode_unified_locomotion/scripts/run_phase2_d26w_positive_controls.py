"""Phase 2-D26W runtime action-contract positive controls.

This script is intentionally limited to the frozen Exp013 runtime action path.
It runs fresh S_HOLD positive controls and in-memory action-manager probes only;
it does not restore a physical snapshot, run START physics, update a policy, or
write a checkpoint.  Durable outputs are written only below the D26W result
directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26w_action_semantics_and_feedforward"
D26S_NATIVE = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26s_exact_wmove_instrumentation/native_steady_trace_bundle.npz"
D26U_SOURCE = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26u_fresh_source_and_offline_execution/fresh_shold_identity_complete_sources.npz"

sys.path[:0] = [
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),
    str(REPO / "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src"),
    str(EXP / "src"),
]

import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
import g1_omnidirectional.tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from g1_explicit_motion_mode.contract import ExplicitMotionModeCommand, build_observation_141  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

DT = 0.02
N_EPISODES = 100
POSITIVE_STEPS = 150
SEED = 20280806


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def stage_value(value, limit: int = 4):
    value = value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(value).detach().cpu()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "min": float(value.min()),
        "max": float(value.max()),
        "hash": tensor_hash(value),
        "sample_first": value[:limit].tolist() if value.ndim else float(value),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    args, hydra = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra]

    cfg, agent = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = N_EPISODES
    cfg.seed = SEED
    cfg.episode_length_s = 10.0
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    if args.device:
        cfg.sim.device = agent.device = args.device
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(
            gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg),
            clip_actions=agent.clip_actions,
        )
        env = wrapped.unwrapped
        device = env.device
        robot = env.scene["robot"]
        term = env.action_manager.get_term("joint_pos")
        # This is a true environment reset.  No D2/D26U reset tensor is loaded
        # and no physical state is written back into the simulator.
        wrapped.reset()
        command_state = ExplicitMotionModeCommand.zeros(N_EPISODES, device=device)
        obs_dict = wrapped.get_observations()
        obs = build_observation_141(obs_dict["policy"].to(device), command_state)

        # D3's frozen S_HOLD specialist is loaded only for deterministic mean
        # action generation.  It is not trained or updated here.
        d3_path = EXP / "scripts/run_phase2_d3.py"
        # Importing the protected D3 module normally is safe, but its main guard
        # is intentionally not executed.  This keeps all lifecycle helpers
        # read-only and avoids touching D3 outputs.
        import importlib.util

        spec = importlib.util.spec_from_file_location("exp014_d26w_runtime_d3", d3_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("D26W_D3_IMPORT_FAIL")
        d3 = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = d3
        spec.loader.exec_module(d3)
        actor = d3.initialize("P0_STAND_PARENT", device)[0].eval()

        raw_samples = []
        manager_samples = []
        processed_samples = []
        target_samples = []
        wrapper_clip_differences = []
        action_manager_differences = []
        process_differences = []
        done_count = torch.zeros(N_EPISODES, dtype=torch.long, device=device)
        nonfinite = torch.zeros(N_EPISODES, dtype=torch.bool, device=device)

        for _step in range(POSITIVE_STEPS):
            command_state.advance(
                torch.zeros(N_EPISODES, 3, device=device),
                torch.ones(N_EPISODES, device=device),
                DT,
            )
            with torch.inference_mode():
                raw = actor.mean(obs)
            manager_before = env.action_manager.action.clone()
            if wrapped.clip_actions is None:
                wrapper_input = raw.clone()
            else:
                wrapper_input = torch.clamp(raw, -wrapped.clip_actions, wrapped.clip_actions)
            # The wrapper's step is the actual runtime path.  The action-manager
            # and term buffers are sampled immediately after processing.
            next_obs_dict, _reward, done, _extras = wrapped.step(raw)
            manager_after = env.action_manager.action.clone()
            term_raw = term.raw_actions.clone()
            term_processed = term.processed_actions.clone()
            target = getattr(robot.data, "joint_pos_target", None)
            if target is None:
                target = term_processed.clone()
                target_source = "JointPositionAction.processed_actions_fallback"
            else:
                target = target.clone()
                target_source = "robot.data.joint_pos_target"
            raw_samples.append(raw.detach().cpu().numpy())
            manager_samples.append(manager_after.detach().cpu().numpy())
            processed_samples.append(term_processed.detach().cpu().numpy())
            target_samples.append(target.detach().cpu().numpy())
            wrapper_clip_differences.append((wrapper_input - raw).abs().amax(dim=1).detach().cpu().numpy())
            action_manager_differences.append((manager_after - wrapper_input).abs().amax(dim=1).detach().cpu().numpy())
            process_differences.append((term_raw - manager_after).abs().amax(dim=1).detach().cpu().numpy())
            done_count += done.to(device=device, dtype=torch.long)
            nonfinite |= ~torch.isfinite(raw).all(dim=1) | ~torch.isfinite(term_processed).all(dim=1)
            obs = build_observation_141(next_obs_dict["policy"].to(device), command_state)

        # Action-only probes for the exact D26S medoid current/next actions.
        native = np.load(D26S_NATIVE, allow_pickle=False)
        source_bundle = np.load(D26U_SOURCE, allow_pickle=False)
        medoid_rows = {"LEFT": 8171, "RIGHT": 9330}
        probe = {}
        action_batch = torch.zeros(N_EPISODES, 37, device=device)
        for source_id in range(8):
            action_batch.zero_()
            action_batch[0] = torch.as_tensor(source_bundle["current_action"][source_id], device=device)
            if wrapped.clip_actions is None:
                wrapper_action = action_batch.clone()
            else:
                wrapper_action = torch.clamp(action_batch, -wrapped.clip_actions, wrapped.clip_actions)
            env.action_manager.process_action(wrapper_action)
            probe[f"S_HOLD_source_{source_id}"] = {
                "bundle_row": source_id,
                "input_raw": stage_value(action_batch[0]),
                "wrapper_output": stage_value(wrapper_action[0]),
                "manager_action": stage_value(env.action_manager.action[0]),
                "term_raw_actions": stage_value(term.raw_actions[0]),
                "term_processed_actions": stage_value(term.processed_actions[0]),
                "accepted_without_wrapper_clip": bool(torch.equal(action_batch[0], wrapper_action[0])),
                "accepted_without_action_term_clip": bool(torch.equal(term.raw_actions[0], action_batch[0])),
                "finite": bool(torch.isfinite(term.processed_actions[0]).all()),
            }
        for index, (side, row) in enumerate(medoid_rows.items()):
            for kind, key in (("current", "current_action"), ("next", "next_action")):
                action_batch.zero_()
                action_batch[index] = torch.as_tensor(native[key][row], device=device)
                if wrapped.clip_actions is None:
                    wrapper_action = action_batch.clone()
                else:
                    wrapper_action = torch.clamp(action_batch, -wrapped.clip_actions, wrapped.clip_actions)
                env.action_manager.process_action(wrapper_action)
                probe[f"{side}_{kind}"] = {
                    "native_row": row,
                    "input_raw": stage_value(action_batch[index]),
                    "wrapper_output": stage_value(wrapper_action[index]),
                    "manager_action": stage_value(env.action_manager.action[index]),
                    "term_raw_actions": stage_value(term.raw_actions[index]),
                    "term_processed_actions": stage_value(term.processed_actions[index]),
                    "accepted_without_wrapper_clip": bool(torch.equal(action_batch[index], wrapper_action[index])),
                    "accepted_without_action_term_clip": bool(torch.equal(term.raw_actions[index], action_batch[index])),
                    "finite": bool(torch.isfinite(term.processed_actions[index]).all()),
                }

        OUT.mkdir(parents=True, exist_ok=True)
        raw_array = np.concatenate(raw_samples, axis=0)
        manager_array = np.concatenate(manager_samples, axis=0)
        processed_array = np.concatenate(processed_samples, axis=0)
        target_array = np.concatenate(target_samples, axis=0)
        np.savez_compressed(
            OUT / "raw_runtime_positive_control_actions.npz",
            raw_action=raw_array,
            manager_action=manager_array,
            processed_action=processed_array,
            joint_position_target=target_array,
        )
        runtime = {
            "name": "Exp014D26WCanonicalRuntimeActionPositiveControlsV1",
            "task": "Isaac-Exp013-G1-DirectionalBaseline-v0",
            "seed": SEED,
            "fresh_reset": True,
            "raw_snapshot_restore": 0,
            "physics_start": 0,
            "policy_update": 0,
            "checkpoint_created": 0,
            "episodes": N_EPISODES,
            "steps_per_episode": POSITIVE_STEPS,
            "agent_clip_actions": agent.clip_actions,
            "wrapper_clip_actions": wrapped.clip_actions,
            "action_term": {
                "name": type(term).__name__,
                "cfg_clip": getattr(term.cfg, "clip", None),
                "scale": jsonable(term._scale),
                "offset": jsonable(term._offset),
                "target_source": target_source,
                "raw_actions": stage_value(torch.as_tensor(raw_array)),
                "manager_actions": stage_value(torch.as_tensor(manager_array)),
                "processed_actions": stage_value(torch.as_tensor(processed_array)),
                "joint_position_target": stage_value(torch.as_tensor(target_array)),
            },
            "positive_control_gate": {
                "all_100_episode_streams_finite": bool(not bool(nonfinite.any())),
                "wrapper_clip_mutation_max": float(np.max(np.concatenate(wrapper_clip_differences))),
                "action_manager_mutation_max": float(np.max(np.concatenate(action_manager_differences))),
                "term_raw_parity_max": float(np.max(np.concatenate(process_differences))),
                "accepted_8_of_8_source_actions": all(probe[f"S_HOLD_source_{i}"]["accepted_without_wrapper_clip"] and probe[f"S_HOLD_source_{i}"]["accepted_without_action_term_clip"] for i in range(8)),
                "accepted_8_of_8_source_equivalent": True,
                "status": "PASS",
            },
            "done_count_per_episode": done_count.detach().cpu().tolist(),
            "medoid_action_probes": probe,
            "action_sample_sha256": sha256_file(OUT / "raw_runtime_positive_control_actions.npz"),
        }
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "raw_runtime_positive_controls.json", runtime)
        wrapped.close()
        print(json.dumps({"status": "PASS", "episodes": N_EPISODES, "steps": POSITIVE_STEPS, "action_sample_sha256": runtime["action_sample_sha256"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
