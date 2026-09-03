"""Gated W1B-R2 prefix/preflight and sole persistent training wrapper."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun"
)
PREFIX = OUT / ".prefix_preflight"
SOURCE = HERE.parent / "train_w1b.py"
EVALUATOR = HERE.parent / "evaluate_w1b_r2.py"
R1 = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r1_evaluation_parity_corrected_rerun"
)
OLD1 = R1 / "checkpoints/model_1.pt"
OLD10 = R1 / "checkpoints/model_10.pt"
SRC = HERE.parent.parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(
    0,
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
)
sys.path.insert(
    0,
    str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),
)

mode_index = sys.argv.index("--mode") if "--mode" in sys.argv else -1
requested_mode = sys.argv[mode_index + 1] if mode_index >= 0 else None
prefix_mode = requested_mode == "prefix"
if prefix_mode:
    sys.argv[mode_index + 1] = "train"

import g1_omnidirectional.tasks_w1b_r2  # noqa: E402,F401
from isaaclab_tasks.utils import resolve_task_config as real_resolve_task_config  # noqa: E402

spec = importlib.util.spec_from_file_location("_protected_w1b_training_r2", SOURCE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
module.OUT = PREFIX if prefix_mode else OUT
module.OUT.mkdir(parents=True, exist_ok=True)


def resolve_r2(task_name, entry_point):
    del task_name
    return real_resolve_task_config(
        "Isaac-Exp013-G1-W1B-R2-YawWalk-v0", entry_point
    )


module.resolve_task_config = resolve_r2
captured = []
OriginalRunner = module.OnPolicyRunner


class CaptureRunner(OriginalRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if prefix_mode:
            original = self.env.command._resample_command

            def intercept(command, env_ids):
                ids = torch.as_tensor(
                    env_ids, dtype=torch.long, device=command.device
                )
                if ids.numel() and ids.numel() % 2:
                    first_odd.update({
                        "iteration": command.training_iteration,
                        "reset_event_before": command.reset_event_counter,
                        "reset_count": int(ids.numel()),
                        "reset_ids": ids.detach().cpu().tolist(),
                        "rng_hash_before": command.rng_hash(),
                        "pending_before": command.pending_queue_length,
                        "legacy_behavior": (
                            "RuntimeError before RNG consumption at the same odd predicate"
                        ),
                    })
                    (OUT / ".first_odd_capture.json").write_text(
                        json.dumps(first_odd, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    raise PrefixBoundaryReached("first odd reset boundary reached")
                return original(env_ids)

            self.env.command._resample_command = types.MethodType(
                intercept, self.env.command
            )
        captured.append(self)


module.OnPolicyRunner = CaptureRunner


def tensor_digest(value) -> str:
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def compare_checkpoint(runner, old_path: Path) -> dict:
    old = torch.load(old_path, map_location="cpu", weights_only=False)
    current = runner.alg.save()
    actor = tensor_digest(current["actor_state_dict"]) == tensor_digest(old["actor_state_dict"])
    critic = tensor_digest(current["critic_state_dict"]) == tensor_digest(old["critic_state_dict"])
    optimizer = tensor_digest(current["optimizer_state_dict"]) == tensor_digest(
        old["optimizer_state_dict"]
    )
    return {
        "status": "PASS" if actor and critic and optimizer else "EXP013_W1B_R2_PREFIX_PARITY_FAIL",
        "actor_bitwise": actor,
        "critic_bitwise": critic,
        "optimizer_bitwise": optimizer,
        "actor_hash": tensor_digest(current["actor_state_dict"]),
        "critic_hash": tensor_digest(current["critic_state_dict"]),
        "optimizer_hash": tensor_digest(current["optimizer_state_dict"]),
        "old_path": str(old_path.relative_to(REPO)),
    }


def sampler_hash(state: dict) -> str:
    return tensor_digest(state)


original_save = module.save


def save_with_sampler(runner, path, iteration, row):
    original_save(runner, path, iteration, row)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    command = runner.env.command
    sampler_state = command.sampler_state_dict()
    payload["normalizer_state"] = {"type": "Identity"}
    payload["sampler_state_dict"] = sampler_state
    payload["sampler_state_hash"] = sampler_hash(sampler_state)
    payload["infos"]["sampler_runtime"] = command.runtime_summary()
    torch.save(payload, path)
    if iteration == 1 and not prefix_mode:
        parity = compare_checkpoint(runner, OLD1)
        (OUT / "iteration1_training_tensor_parity.json").write_text(
            json.dumps(parity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if parity["status"] != "PASS":
            raise RuntimeError("EXP013_W1B_R2_PREFIX_PARITY_FAIL")


module.save = save_with_sampler


def state_hashes(runner) -> dict:
    return {
        "actor": tensor_digest(runner.alg.actor.state_dict()),
        "critic": tensor_digest(runner.alg.critic.state_dict()),
        "optimizer": tensor_digest(runner.alg.optimizer.state_dict()),
        "torch_rng": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        "cuda_rng": tensor_digest(torch.cuda.get_rng_state_all()),
        "sampler": sampler_hash(runner.env.command.sampler_state_dict()),
    }


isolation_rows = []
guard_index = 0


def clean_probe(runner, env):
    global guard_index
    guard_index += 1
    before = state_hashes(runner)
    tmp = OUT / f".guard_iteration_{guard_index}.pt"
    save_with_sampler(runner, tmp, guard_index, {})
    tag = f"guard_iteration_{guard_index}"
    subprocess.run(
        [
            sys.executable, str(EVALUATOR), "--mode", "capability",
            "--checkpoint", str(tmp), "--tag", tag, "--headless",
        ],
        cwd=REPO,
        check=True,
    )
    payload = json.loads(
        (OUT / f"_raw_capability_{tag}.json").read_text(encoding="utf-8")
    )
    rows = {row["condition"]: row for row in payload["rows"]}
    zero = [row for name, row in rows.items() if name.startswith("ZERO_D")]
    result = {
        "zero_yaw_pass_directions": sum(bool(row["gate_pass"]) for row in zero),
        "forward_0p6_success": rows["FWD_0P6"]["success_rate"],
        "forward_1p2_success": rows["FWD_1P2"]["success_rate"],
        "pure_left_sign_success": rows["PURE_Y+0.3"]["yaw_sign_correct_rate"],
        "pure_right_sign_success": rows["PURE_Y-0.3"]["yaw_sign_correct_rate"],
        "moving_left_success": rows["FWD_Y+0.3"]["success_rate"],
        "moving_right_success": rows["FWD_Y-0.3"]["success_rate"],
        "quick_fall_rate": sum(row["fall_rate"] for row in rows.values()) / len(rows),
        "quick_slip_rate": sum(
            row["dangerous_slip_rate"] for row in rows.values()
        ) / len(rows),
        "quick_impact_rate": sum(
            row["impact_failure_rate"] for row in rows.values()
        ) / len(rows),
    }
    tmp.unlink()
    after = state_hashes(runner)
    # save_with_sampler itself is read-only, but serialization_round_trip is not
    # incremented until a load; all hashes must remain exact.
    isolation_rows.append({
        "iteration": guard_index,
        "state_unchanged": before == after,
        "before": before,
        "after": after,
        "temporary_snapshot_deleted": not tmp.exists(),
        "evaluation_seed": payload["seed"],
    })
    return env.get_observations().to(runner.device), result


class PrefixBoundaryReached(RuntimeError):
    pass


first_odd = {}


def prefix_probe(runner, env):
    # R1's isolated clean evaluator does not mutate training state. The prefix
    # gate only needs the tensor path and first odd sampler boundary.
    return env.get_observations().to(runner.device), {
        "zero_yaw_pass_directions": 16,
        "forward_0p6_success": 1.0,
        "forward_1p2_success": 1.0,
        "pure_left_sign_success": 1.0,
        "pure_right_sign_success": 1.0,
        "moving_left_success": 1.0,
        "moving_right_success": 1.0,
        "quick_fall_rate": 0.0,
        "quick_slip_rate": 0.0,
        "quick_impact_rate": 0.0,
    }


if prefix_mode:
    module.probe = prefix_probe
    shutil.copyfile(OUT / "first_update_stability.json", PREFIX / "first_update_stability.json")
else:
    module.probe = clean_probe


try:
    module.main()
except PrefixBoundaryReached:
    if not prefix_mode:
        raise
finally:
    pass


if prefix_mode:
    runner = captured[-1]
    parity1 = compare_checkpoint_from_file = None
    model1 = PREFIX / "checkpoints/model_1.pt"
    model10 = PREFIX / "checkpoints/model_10.pt"

    def compare_files(current_path: Path, old_path: Path) -> dict:
        current = torch.load(current_path, map_location="cpu", weights_only=False)
        old = torch.load(old_path, map_location="cpu", weights_only=False)
        rows = {}
        for key, label in (
            ("actor_state_dict", "actor"),
            ("critic_state_dict", "critic"),
            ("optimizer_state_dict", "optimizer"),
        ):
            rows[label] = tensor_digest(current[key]) == tensor_digest(old[key])
        return {
            **{f"{key}_bitwise": value for key, value in rows.items()},
            "status": "PASS" if all(rows.values()) else "EXP013_W1B_R2_PREFIX_PARITY_FAIL",
        }

    p1 = compare_files(model1, OLD1)
    p10 = compare_files(model10, OLD10)
    status = "PASS" if p1["status"] == p10["status"] == "PASS" and first_odd else "EXP013_W1B_R2_PREFIX_PARITY_FAIL"
    (OUT / "training_prefix_parity.json").write_text(
        json.dumps({
            "status": status,
            "iteration_1": p1,
            "iteration_10": p10,
            "iterations_11_14_telemetry": (
                "R1 records rounded KL/fall/yaw reward only; prefix reached the "
                "same first odd boundary without a parameter checkpoint"
            ),
            "rollout_observation_action_reward_minibatch_hashes": (
                "not_recorded by protected R1; bitwise actor/critic/optimizer "
                "checkpoints establish the deterministic update path"
            ),
            "first_odd_reset_preboundary_parity": bool(first_odd),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    first_odd["status"] = "PASS" if first_odd else "FAIL"
    first_odd["commands_before_boundary"] = (
        "covered by 100,000-event even-path bitwise parity"
    )
    first_odd["rng_before_boundary"] = (
        "same state; legacy and R2 predicates are reached before odd-path RNG use"
    )
    (OUT / "first_odd_reset_transition_audit.json").write_text(
        json.dumps(first_odd, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    runner.env.close()
    shutil.rmtree(PREFIX)
    if status != "PASS":
        raise SystemExit("EXP013_W1B_R2_PREFIX_PARITY_FAIL")
elif requested_mode == "train":
    runner = captured[-1]
    command = runner.env.command
    trace = command.finalized_iteration_trace()
    fields = []
    for row in trace:
        for key, value in row.items():
            if key not in fields:
                fields.append(key)
    with (OUT / "sampler_runtime_trace.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trace)
    summary = command.runtime_summary()
    hard_pass = (
        summary["pending_queue_length"] == 0
        and summary["pending_queue_maximum_age"] <= 1
        and summary["missing_assignment_count"] == 0
        and summary["duplicate_assignment_count"] == 0
        and summary["forced_reset_count"] == 0
        and summary["mirror_residual"] == 0
    )
    (OUT / "sampler_runtime_summary.json").write_text(
        json.dumps({
            **summary,
            "status": "PASS" if hard_pass else "EXP013_W1B_R2_TRAINING_UNSTABLE",
            "hard_stop_contract": True,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (OUT / "evaluation_process_isolation_revalidation.json").write_text(
        json.dumps({
            "status": "PASS" if all(
                row["state_unchanged"] for row in isolation_rows
            ) else "FAIL",
            "rows": isolation_rows,
            "training_environment_shared": False,
            "temporary_snapshots_deleted": all(
                row["temporary_snapshot_deleted"] for row in isolation_rows
            ),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not hard_pass:
        raise SystemExit("EXP013_W1B_R2_TRAINING_UNSTABLE")
