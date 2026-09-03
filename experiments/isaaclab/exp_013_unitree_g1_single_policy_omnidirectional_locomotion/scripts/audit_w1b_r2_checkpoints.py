"""Fresh-process sampler-state load/next-assignment audit for every checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import gymnasium as gym
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun"
)
SRC = HERE.parent.parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(
    0,
    str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),
)
sys.path.insert(
    0,
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
)
import isaaclab_tasks  # noqa: E402,F401
import g1_omnidirectional.tasks_w1b_r2  # noqa: E402,F401
from isaaclab_tasks.utils import (  # noqa: E402
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]
schedule = (0, 1, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200)


def digest(*values: torch.Tensor) -> str:
    result = hashlib.sha256()
    for value in values:
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


cfg, _ = resolve_task_config(
    "Isaac-Exp013-G1-W1B-R2-YawWalk-v0", "rsl_rl_cfg_entry_point"
)
cfg.scene.num_envs = 1024
cfg.seed = 20274021
if args.device:
    cfg.sim.device = args.device
rows = []
with launch_simulation(cfg, args):
    env = gym.make("Isaac-Exp013-G1-W1B-R2-YawWalk-v0", cfg=cfg)
    command = env.unwrapped.command_manager.get_term("base_velocity")
    ids = torch.tensor([9, 2, 31, 7, 18], device=env.unwrapped.device)
    for iteration in schedule:
        label = "initial" if iteration == 0 else str(iteration)
        path = OUT / f"checkpoints/model_{label}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        required = {
            "actor_state_dict", "critic_state_dict", "optimizer_state_dict",
            "normalizer_state", "sampler_state_dict", "sampler_state_hash",
        }
        missing = sorted(required - set(payload))
        if missing:
            rows.append({
                "iteration": iteration,
                "status": "EXP013_W1B_R2_SAMPLER_SERIALIZATION_FAIL",
                "missing": missing,
            })
            continue
        state = payload["sampler_state_dict"]
        command.load_sampler_state_dict(state)
        command._resample_command(ids)
        first = digest(
            command.vel_command_b[ids, :3],
            command.sampled_pair_id[ids],
            torch.cuda.get_rng_state(env.unwrapped.device),
        )
        summary_first = command.runtime_summary()
        command.load_sampler_state_dict(state)
        command._resample_command(ids)
        second = digest(
            command.vel_command_b[ids, :3],
            command.sampled_pair_id[ids],
            torch.cuda.get_rng_state(env.unwrapped.device),
        )
        summary_second = command.runtime_summary()
        ok = first == second and summary_first["mirror_residual"] == summary_second["mirror_residual"]
        rows.append({
            "iteration": iteration,
            "status": "PASS" if ok else "EXP013_W1B_R2_SAMPLER_SERIALIZATION_FAIL",
            "next_assignment_hash": first,
            "repeat_hash": second,
            "pending_queue_length": summary_first["pending_queue_length"],
            "rng_hash": summary_first["rng_hash"],
            "sampler_state_hash": payload["sampler_state_hash"],
            "all_required_state_present": True,
        })
    status = all(row["status"] == "PASS" for row in rows)
    (OUT / "pending_queue_serialization_audit.json").write_text(
        json.dumps({
            "status": "PASS" if status else "EXP013_W1B_R2_SAMPLER_SERIALIZATION_FAIL",
            "fresh_process": True,
            "checkpoints": rows,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PASS" if status else "FAIL")
    if not status:
        raise SystemExit("EXP013_W1B_R2_SAMPLER_SERIALIZATION_FAIL")
    env.close()
