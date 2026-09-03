"""Evaluation-isolated wrapper around the protected original W1B training path."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r1_evaluation_parity_corrected_rerun"
SOURCE = HERE.parent / "train_w1b.py"
EVALUATOR = HERE.parent / "evaluate_w1b_r1.py"
OLD1 = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_yaw_conditioned_omnidirectional_walk/checkpoints/model_1.pt"
for path in (
    REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src",
    REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src",
):
    sys.path.insert(0, str(path))

spec = importlib.util.spec_from_file_location("_protected_original_w1b_training", SOURCE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
module.OUT = OUT
OUT.mkdir(parents=True, exist_ok=True)


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


def compare_with_old(runner) -> dict:
    old = torch.load(OLD1, map_location="cpu", weights_only=False)
    current = runner.alg.save()
    actor_equal = all(
        torch.equal(current["actor_state_dict"][key].cpu(), value.cpu())
        for key, value in old["actor_state_dict"].items()
    )
    critic_equal = all(
        torch.equal(current["critic_state_dict"][key].cpu(), value.cpu())
        for key, value in old["critic_state_dict"].items()
    )
    optimizer_equal = tensor_digest(current["optimizer_state_dict"]) == tensor_digest(old["optimizer_state_dict"])
    old_steps = sorted({
        int(value["step"]) for value in old["optimizer_state_dict"]["state"].values()
        if "step" in value
    })
    current_steps = sorted({
        int(value["step"]) for value in current["optimizer_state_dict"]["state"].values()
        if "step" in value
    })
    status = actor_equal and critic_equal and optimizer_equal and old_steps == current_steps
    return {
        "status": "PASS" if status else "EXP013_W1B_R1_TRAINING_PARITY_FAIL",
        "actor_tensors_bitwise_equal": actor_equal,
        "critic_tensors_bitwise_equal": critic_equal,
        "optimizer_tensors_bitwise_equal": optimizer_equal,
        "adam_steps_old": old_steps,
        "adam_steps_rerun": current_steps,
        "actor_tensor_hash_old": tensor_digest(old["actor_state_dict"]),
        "actor_tensor_hash_rerun": tensor_digest(current["actor_state_dict"]),
        "critic_tensor_hash_old": tensor_digest(old["critic_state_dict"]),
        "critic_tensor_hash_rerun": tensor_digest(current["critic_state_dict"]),
        "optimizer_tensor_hash_old": tensor_digest(old["optimizer_state_dict"]),
        "optimizer_tensor_hash_rerun": tensor_digest(current["optimizer_state_dict"]),
        "serialization_metadata_excluded": True,
    }


captured = []
OriginalRunner = module.OnPolicyRunner


class CaptureRunner(OriginalRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        captured.append(self)


isolation_rows = []
guard_index = 0


def state_hashes(runner) -> dict:
    return {
        "actor": tensor_digest(runner.alg.actor.state_dict()),
        "critic": tensor_digest(runner.alg.critic.state_dict()),
        "optimizer": tensor_digest(runner.alg.optimizer.state_dict()),
        "torch_rng": hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
        "cuda_rng": tensor_digest(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else "not_applicable",
    }


def clean_probe(runner, env):
    global guard_index
    guard_index += 1
    before = state_hashes(runner)
    tmp = OUT / f".guard_iteration_{guard_index}.pt"
    module.save(runner, tmp, guard_index, {})
    tag = f"guard_iteration_{guard_index}"
    subprocess.run([
        sys.executable, str(EVALUATOR), "--mode", "capability",
        "--checkpoint", str(tmp), "--tag", tag, "--headless",
    ], cwd=REPO, check=True)
    payload = json.loads((OUT / f"_raw_capability_{tag}.json").read_text(encoding="utf-8"))
    rows = {row["condition"]: row for row in payload["rows"]}
    zero = [row for name, row in rows.items() if name.startswith("ZERO_D")]
    q = {
        "zero_yaw_pass_directions": sum(bool(row["gate_pass"]) for row in zero),
        "forward_0p6_success": rows["FWD_0P6"]["success_rate"],
        "forward_1p2_success": rows["FWD_1P2"]["success_rate"],
        "pure_left_sign_success": rows["PURE_Y+0.3"]["yaw_sign_correct_rate"],
        "pure_right_sign_success": rows["PURE_Y-0.3"]["yaw_sign_correct_rate"],
        "moving_left_success": rows["FWD_Y+0.3"]["success_rate"],
        "moving_right_success": rows["FWD_Y-0.3"]["success_rate"],
        "quick_fall_rate": sum(row["fall_rate"] for row in rows.values()) / len(rows),
        "quick_slip_rate": sum(row["dangerous_slip_rate"] for row in rows.values()) / len(rows),
        "quick_impact_rate": sum(row["impact_failure_rate"] for row in rows.values()) / len(rows),
    }
    tmp.unlink()
    after = state_hashes(runner)
    isolation_rows.append({
        "iteration": guard_index,
        "state_unchanged": before == after,
        "before": before,
        "after": after,
        "temporary_snapshot_deleted": not tmp.exists(),
        "evaluation_seed": payload["seed"],
        "checkpoint_actor_hash": before["actor"],
    })
    (OUT / "evaluation_process_isolation_audit.json").write_text(
        json.dumps({
            "status": "PASS" if all(row["state_unchanged"] for row in isolation_rows) else "FAIL",
            "guard_rows": isolation_rows,
            "training_environment_shared": False,
            "training_rng_unchanged_by_guard": all(row["state_unchanged"] for row in isolation_rows),
            "temporary_snapshots_deleted": all(row["temporary_snapshot_deleted"] for row in isolation_rows),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # The training environment and its observation are deliberately untouched.
    return env.get_observations().to(runner.device), q


original_save = module.save


def checked_save(runner, path, iteration, row):
    original_save(runner, path, iteration, row)
    if iteration == 1:
        result = compare_with_old(runner)
        (OUT / "iteration1_training_tensor_parity.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        first = json.loads((OUT / "first_update_stability.json").read_text(encoding="utf-8"))
        old_curve = list(csv.DictReader(
            (REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_yaw_conditioned_omnidirectional_walk/training_curves.csv").open(encoding="utf-8")
        ))[0]
        telemetry_keys = ("exact_rollout_kl", "all_step_maximum_kl", "clip_fraction", "mean_action_shift", "value_loss")
        telemetry_diff = {key: abs(float(first[key]) - float(old_curve[key])) for key in telemetry_keys}
        audit = {
            "status": "PASS" if result["status"] == "PASS" and max(telemetry_diff.values()) <= 1e-12 else "EXP013_W1B_R1_TRAINING_PARITY_FAIL",
            "tensor_parity": result,
            "training_telemetry_absolute_difference": telemetry_diff,
            "deterministic_tolerance": 1e-12,
            "rollout_observation_hash": "not_recorded_by_protected_W1B",
            "action_hash": "not_recorded_by_protected_W1B",
            "reward_hash": "not_recorded_by_protected_W1B",
            "advantage_hash": "not_recorded_by_protected_W1B",
            "minibatch_order_hash": "not_recorded_by_protected_W1B",
            "inference": "bitwise final actor/critic/optimizer parity covers the complete one-update training path",
        }
        (OUT / "training_path_noninterference_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if result["status"] != "PASS":
            raise RuntimeError("EXP013_W1B_R1_TRAINING_PARITY_FAIL")


module.OnPolicyRunner = CaptureRunner
if module.a.mode == "train":
    module.probe = clean_probe
    module.save = checked_save
module.main()

if module.a.mode == "preflight":
    parity = compare_with_old(captured[-1])
    (OUT / "iteration1_training_tensor_parity.json").write_text(
        json.dumps(parity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    old_curve = list(csv.DictReader(
        (REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_yaw_conditioned_omnidirectional_walk/training_curves.csv").open(encoding="utf-8")
    ))[0]
    fresh = json.loads((OUT / "first_update_stability.json").read_text(encoding="utf-8"))
    telemetry_keys = (
        "exact_rollout_kl", "all_step_maximum_kl", "clip_fraction",
        "mean_action_shift", "value_loss",
    )
    telemetry_diff = {
        key: abs(float(fresh[key]) - float(old_curve[key])) for key in telemetry_keys
    }
    audit = {
        "status": "PASS" if parity["status"] == "PASS" and max(telemetry_diff.values()) <= 1e-12 else "EXP013_W1B_R1_TRAINING_PARITY_FAIL",
        "tensor_parity": parity,
        "training_telemetry_absolute_difference": telemetry_diff,
        "deterministic_tolerance": 1e-12,
        "rollout_observation_hash": "not_recorded_by_protected_W1B",
        "action_hash": "not_recorded_by_protected_W1B",
        "reward_hash": "not_recorded_by_protected_W1B",
        "advantage_hash": "not_recorded_by_protected_W1B",
        "minibatch_order_hash": "not_recorded_by_protected_W1B",
        "inference": "bitwise final actor/critic/optimizer parity covers the complete one-update training path",
    }
    (OUT / "training_path_noninterference_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if audit["status"] != "PASS":
        raise SystemExit("EXP013_W1B_R1_TRAINING_PARITY_FAIL")
else:
    (OUT / "evaluation_process_isolation_audit.json").write_text(
        json.dumps({
            "status": "PASS" if all(row["state_unchanged"] for row in isolation_rows) else "FAIL",
            "guard_rows": isolation_rows,
            "training_environment_shared": False,
            "training_rng_unchanged_by_guard": all(row["state_unchanged"] for row in isolation_rows),
            "temporary_snapshots_deleted": all(row["temporary_snapshot_deleted"] for row in isolation_rows),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    early = json.loads((OUT / "early_guard.json").read_text(encoding="utf-8"))
    rows = []
    for row in early["rows"]:
        rows.append({
            "iteration": row["iteration"],
            "clean_zero_yaw_pass_directions": row["zero_yaw_pass_directions"],
            "clean_fall_rate": row["quick_fall_rate"],
            "clean_dangerous_slip_rate": row["quick_slip_rate"],
            "clean_impact_rate": row["quick_impact_rate"],
            "noisy_training_rollout_fall_rate": row["fall_rate"],
            "noisy_training_rollout_dangerous_slip_rate": row["dangerous_slip_rate"],
            "noisy_training_rollout_impact_rate": row["impact_failure_rate"],
            "noisy_monitor_diagnostic_only": True,
            "guard_pass": row["guard_pass"],
        })
    with (OUT / "clean_vs_noisy_guard_timeline.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
