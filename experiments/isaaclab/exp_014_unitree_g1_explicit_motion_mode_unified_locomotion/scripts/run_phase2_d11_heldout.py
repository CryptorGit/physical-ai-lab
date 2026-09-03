"""One-time D11 evaluation of the sealed S1 omnidirectional-stop held-out split."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
START = "4153ec7310ac449ed60fcc14574886cda5bf9904"
D7 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d7_r4_stop_oracle_distillation"
D10 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d10_s1_stop_closed_loop"
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d11_stop_student_heldout"
RAW = OUT / "raw"
SEALED = D7 / "raw/sealed_heldout_snapshots.pt"
S1 = D7 / "raw/bc_checkpoints/s1_step_30000.pt"
EXPECTED_S1_SHA = "5de37e5d0807654d370ba7a79ee9872c4029cb50e548685423e48812249959d5"
EXPECTED_TENSOR_SHA = "e1df768438830af2da2ea393afb187b7ceb735826975019b02dc03d80dca6f78"
EXPECTED_SEALED_SHA = "dd34d3035866bd35e29643e616a1add35b3ca8bd3c7e05a58ce73b7182f266e9"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


d10 = load_module("d10_frozen", HERE.parent / "run_phase2_d10_frozen.py")
d6 = d10.d6; d3 = d10.d3; s1mod = d10.s1mod
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def tensor_hash(state):
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(*args): return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def main():
    parser = argparse.ArgumentParser(); add_launcher_args(parser); args, hydra = setup_preset_cli(parser); sys.argv = [sys.argv[0], *hydra]
    OUT.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    if (RAW / "heldout_results.json").exists() or (OUT / "heldout_access_ledger.json").exists():
        raise RuntimeError("D11 held-out already accessed; one-time evaluator refuses a rerun")

    checkpoint = torch.load(S1, map_location="cpu", weights_only=False)
    d7_integrity = json.loads((D7 / "dataset_integrity.json").read_text(encoding="utf-8"))
    protected = git("diff", "--name-only", START, "--", "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d6*", "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d7*", "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d8*", "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d9*", "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d10*")
    checks = {
        "candidate_byte_sha_match": sha(S1) == EXPECTED_S1_SHA,
        "candidate_tensor_hash_match": tensor_hash(checkpoint["actor_state_dict"]) == EXPECTED_TENSOR_SHA,
        "architecture_match": checkpoint["architecture"] == [141, 512, 512, 256, 37],
        "observation_dimension": checkpoint["architecture"][0] == 141,
        "action_dimension": checkpoint["architecture"][-1] == 37,
        "sealed_byte_sha_match": sha(SEALED) == EXPECTED_SEALED_SHA,
        "snapshot_overlap_zero": d7_integrity["snapshot_overlap"] == 0,
        "trajectory_overlap_zero": d7_integrity["trajectory_overlap"] == 0,
        "train_overlap_zero": d7_integrity["snapshot_overlap"] == 0,
        "validation_overlap_zero": d7_integrity["snapshot_overlap"] == 0,
        "protected_D6_D10_committed_diff_zero": not protected,
    }
    preopen = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "candidate_sha256": sha(S1), "candidate_tensor_hash": tensor_hash(checkpoint["actor_state_dict"]), "architecture": checkpoint["architecture"], "sealed_sha256": sha(SEALED), "protected_diff": protected.splitlines() if protected else []}
    dump(OUT / "preopen_integrity.json", preopen)
    if preopen["status"] != "PASS": raise RuntimeError("pre-open integrity gate failed; sealed payload was not deserialized")

    opened = dt.datetime.now(dt.timezone.utc).isoformat()
    ordinal_ids = [f"sealed-heldout-{index:06d}" for index in range(579)]
    ledger = {"open_timestamp": opened, "candidate_checkpoint_sha": EXPECTED_S1_SHA, "sealed_manifest_sha": EXPECTED_SEALED_SHA, "expected_episode_ids": ordinal_ids, "sealed_episode_ids": [], "completed_episode_ids": [], "unevaluated_episode_ids": ordinal_ids, "rerun_count_per_episode": {key: 0 for key in ordinal_ids}, "result_selection_access": False, "resume_count": 0, "resume_events": [], "status": "OPEN_AUTHORIZED_PENDING"}
    dump(OUT / "heldout_access_ledger.json", ledger)  # Must precede torch.load(SEALED).

    payloads = torch.load(SEALED, map_location="cpu", weights_only=False)
    actual_ids = [entry["episode_id"] for payload in payloads for entry in payload["entries"]]
    if len(actual_ids) != 579 or len(set(actual_ids)) != 579: raise RuntimeError("sealed episode count/identity mismatch after authorized open")
    ledger["sealed_episode_ids"] = actual_ids

    cfg, agent = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 476; cfg.seed = 20279103; cfg.episode_length_s = 20.; cfg.observations.policy.enable_corruption = False; cfg.events.base_external_force_torque = None; cfg.events.push_robot = None
    if args.device: cfg.sim.device = agent.device = args.device
    resets = d3.load_resets(); severity = torch.zeros(680)
    rows, identities = [], []
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg), clip_actions=agent.clip_actions)
        world = d3.StandWorld(wrapped, resets, severity); student = s1mod.S1().to(world.device).eval(); student.load_state_dict(checkpoint["actor_state_dict"]); hold = d3.initialize("P0_STAND_PARENT", world.device)[0].eval()
        for batch_index, payload in enumerate(payloads):
            batch_rows, identity = d10.evaluate_batch(world, payload, student, hold, "heldout")
            for row, entry in zip(batch_rows, payload["entries"]):
                condition = entry["condition"]
                row.update({"episode_id": entry["episode_id"], "direction_deg": condition["direction_deg"], "command_speed": condition["speed"], "command_yaw": condition["yaw"], "switch_time_s": condition["switch_time_s"]})
            rows.extend(batch_rows); identities.append(identity)
            completed = {row["episode_id"] for row in rows}; ledger["completed_episode_ids"] = [key for key in actual_ids if key in completed]; ledger["unevaluated_episode_ids"] = [key for key in actual_ids if key not in completed]; ledger["status"] = f"BATCH_{batch_index + 1}_COMMITTED"; dump(OUT / "heldout_access_ledger.json", ledger)
        wrapped.close()
    summary = d10.summarize(rows)
    ledger["status"] = "COMPLETE"; ledger["completed_episode_ids"] = actual_ids; ledger["unevaluated_episode_ids"] = []; dump(OUT / "heldout_access_ledger.json", ledger)
    dump(RAW / "heldout_results.json", {"candidate_sha256": EXPECTED_S1_SHA, "tensor_hash": EXPECTED_TENSOR_SHA, "sealed_sha256": EXPECTED_SEALED_SHA, "rows": rows, "summary": summary, "batch_identities": identities})
    print(json.dumps({"episodes": len(rows), "summary": summary, "access_ledger": ledger["status"]}, indent=2))


if __name__ == "__main__": main()
