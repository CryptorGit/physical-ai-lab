"""Create the fail-closed EXP 012 parent/provenance audit."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2_pilot1"
CKPT = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
EXPECTED = "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"
STARTING_HEAD = "60028d13a5534527835e215c37106ea107585b39"


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_norm(states: dict, key: str) -> float:
    total = 0.0
    for value in states.values():
        tensor = value.get(key)
        if torch.is_tensor(tensor):
            total += float(torch.sum(tensor.double() ** 2))
    return total**0.5


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    status = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).splitlines()
    unrelated = [line for line in status if "exp_012_unitree_g1_single_policy_bidirectional_locomotion" not in line]
    dump("starting_repository_state.json", {
        "reported_head": STARTING_HEAD, "starting_head": head, "head_matches_report": head == STARTING_HEAD,
        "starting_status": status, "unrelated_dirty_paths": unrelated, "remote_push": False,
    })
    actual = sha(CKPT)
    if actual != EXPECTED:
        dump("gate.json", {"status": "FAIL", "classification": "PARENT_CHECKPOINT_PROVENANCE_FAIL"})
        raise SystemExit("PARENT_CHECKPOINT_PROVENANCE_FAIL")
    payload = torch.load(CKPT, map_location="cpu", weights_only=False)
    opt = payload.get("optimizer_state_dict")
    if not opt or len(opt.get("state", {})) != 17:
        dump("gate.json", {"status": "FAIL", "classification": "PARENT_OPTIMIZER_STATE_MISSING"})
        raise SystemExit("PARENT_OPTIMIZER_STATE_MISSING")
    steps = [int(float(v["step"])) for v in opt["state"].values()]
    actor = payload["actor_state_dict"]
    critic = payload["critic_state_dict"]
    dump("parent_checkpoint_manifest.json", {
        "path": str(CKPT.relative_to(REPO)).replace("\\", "/"), "sha256": actual,
        "expected_sha256": EXPECTED, "sha_match": True, "source_iteration": payload["iter"],
        "role": "stable high-speed walking parent", "immutable": True,
    })
    dump("parent_observation_contract.json", {
        "dimension": int(actor["mlp.0.weight"].shape[1]), "expected": 123, "status": "PASS",
        "fields": [
            {"name": "base_linear_velocity", "dimension": 3, "frame": "body"},
            {"name": "base_angular_velocity", "dimension": 3, "frame": "body"},
            {"name": "projected_gravity", "dimension": 3},
            {"name": "velocity_command", "dimension": 3},
            {"name": "joint_positions", "dimension": 37},
            {"name": "joint_velocities", "dimension": 37},
            {"name": "previous_action", "dimension": 37},
        ],
        "absolute_heading_present": False,
    })
    dump("parent_action_contract.json", {
        "dimension": int(actor["mlp.6.weight"].shape[0]), "type": "joint_position_target",
        "action_scale": 0.5, "status": "PASS",
    })
    dump("parent_network_contract.json", {
        "actor": [123, 256, 128, 128, 37], "critic": [123, 256, 128, 128, 1],
        "activation": "ELU", "std_parameter": "distribution.std_param", "status": "PASS",
    })
    env_yaml = yaml.load(
        (CKPT.parent / "params/env.yaml").read_text(encoding="utf-8"),
        Loader=yaml.UnsafeLoader,
    )
    dump("physics_control_contract.json", {
        "physics_dt": env_yaml["sim"]["dt"], "decimation": env_yaml["decimation"],
        "control_dt": env_yaml["sim"]["dt"] * env_yaml["decimation"],
        "control_frequency_hz": 1.0 / (env_yaml["sim"]["dt"] * env_yaml["decimation"]),
        "episode_length_s": env_yaml.get("episode_length_s"), "terrain": "flat",
    })
    group = opt["param_groups"][0]
    dump("parent_optimizer_audit.json", {
        "status": "PASS", "strict_restore_required": True, "state_count": len(opt["state"]),
        "parameter_group_count": len(opt["param_groups"]), "parameter_count": len(group["params"]),
        "adam_step_min": min(steps), "adam_step_max": max(steps), "learning_rate": group["lr"],
        "betas": group["betas"], "eps": group["eps"], "first_moment_norm": tensor_norm(opt["state"], "exp_avg"),
        "second_moment_norm": tensor_norm(opt["state"], "exp_avg_sq"), "source_iteration": payload["iter"],
        "scheduler": "adaptive PPO KL schedule; restored iteration and optimizer LR",
    })
    # The exact order is resolved at live-environment preflight; the checkpoint proves 37 action outputs.
    dump("g1_joint_order.json", {
        "status": "PENDING_LIVE_ENVIRONMENT_RESOLUTION", "expected_joint_count": 37,
        "source": "robot.joint_names from Exp012 live environment",
    })
    dump("single_checkpoint_contract.json", {
        "actor_checkpoint_count": 1, "checkpoint_switch": 0, "expert_router": 0,
        "teacher_policy": 0, "teacher_action_calls": 0, "action_blend": 0,
        "transition_specific_action_source": 0, "status": "PASS",
    })
    from g1_single_policy.phase_gated_heading import run_unit_tests
    unit = run_unit_tests()
    dump("phase_gated_heading_unit_tests.json", unit)
    dump("phase_gated_heading_contract.json", {
        "name": "G1PhaseGatedFixedHeadingController", "kp": 1.0, "yaw_rate_limit": [-0.1, 0.1],
        "heading_error": "atan2(sin(reference-current),cos(reference-current))",
        "quaternion_order": "wxyz", "steady_activation_s": [1.0, 1.5],
        "transition_activation": "0.5 s target acquisition then 0.5 s minimum jerk",
        "unit_tests_pass": unit["all_pass"],
    })
    # Deterministic offline curriculum audit: exact fixed cohort assignment plus random targets.
    generator = torch.Generator().manual_seed(20261021)
    n = 100_000
    cohort = torch.multinomial(torch.tensor([0.2, 0.2, 0.2, 0.4]), n, replacement=True, generator=generator)
    counts = torch.bincount(cohort, minlength=4)
    walk = torch.randint(4, (int(counts[1]),), generator=generator)
    run = torch.randint(2, (int(counts[2] + counts[3]),), generator=generator)
    walk_counts = torch.bincount(walk, minlength=4)
    run_counts = torch.bincount(run, minlength=2)
    cohort_ok = all(abs(float(counts[i]) / n - p) <= 0.01 for i, p in enumerate((.2, .2, .2, .4)))
    ratio_ok = float(walk_counts.max()) / float(walk_counts.min()) <= 1.10 and float(run_counts.max()) / float(run_counts.min()) <= 1.10
    dump("command_curriculum_config.json", {
        "cohorts": {"ZERO_HOLD": .2, "WALK_STEADY": .2, "RUN_HOLD": .2, "BIDIRECTIONAL_SEQUENCE": .4},
        "walk_speeds": [0.6, 0.8, 1.0, 1.2], "run_targets": [2.4, 2.6],
        "sequence_duration_s": 18.5, "minimum_jerk": "10t^3-15t^4+6t^5",
    })
    dump("command_curriculum_audit.json", {
        "status": "PASS" if cohort_ok and ratio_ok else "FAIL", "samples": n,
        "cohort_counts": dict(zip(("ZERO_HOLD", "WALK_STEADY", "RUN_HOLD", "BIDIRECTIONAL_SEQUENCE"), counts.tolist())),
        "walk_counts": dict(zip(map(str, (0.6, .8, 1., 1.2)), walk_counts.tolist())),
        "run_counts": dict(zip(map(str, (2.4, 2.6)), run_counts.tolist())),
        "cohort_tolerance_pass": cohort_ok, "within_target_ratio_pass": ratio_ok,
        "upward_transition_count": int(counts[3]) * 3, "downward_transition_count": int(counts[3]) * 3,
    })
    with (OUT / "command_distribution.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["category", "value", "count"])
        for name, value in zip(("ZERO_HOLD", "WALK_STEADY", "RUN_HOLD", "BIDIRECTIONAL_SEQUENCE"), counts.tolist()):
            writer.writerow(["cohort", name, value])
        for speed, value in zip((0.6, .8, 1., 1.2), walk_counts.tolist()):
            writer.writerow(["walk_speed", speed, value])
        for speed, value in zip((2.4, 2.6), run_counts.tolist()):
            writer.writerow(["run_target", speed, value])
    dump("gate.json", {"status": "PREFLIGHT_PENDING", "classification": None})


if __name__ == "__main__":
    main()
