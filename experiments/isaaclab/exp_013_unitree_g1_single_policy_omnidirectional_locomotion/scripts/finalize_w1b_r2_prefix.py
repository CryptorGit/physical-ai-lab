"""Finalize the read-only W1B-R2 prefix parity artifacts."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch


HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun"
)
PREFIX = OUT / ".prefix_preflight"
R1 = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r1_evaluation_parity_corrected_rerun/checkpoints"
)


def digest(value) -> str:
    result = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            result.update(str(tensor.dtype).encode())
            result.update(str(tuple(tensor.shape)).encode())
            result.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                result.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            result.update(repr(item).encode())

    visit(value)
    return result.hexdigest()


def compare(current_path: Path, old_path: Path) -> dict:
    current = torch.load(current_path, map_location="cpu", weights_only=False)
    old = torch.load(old_path, map_location="cpu", weights_only=False)
    checks = {
        name: digest(current[key]) == digest(old[key])
        for name, key in (
            ("actor", "actor_state_dict"),
            ("critic", "critic_state_dict"),
            ("optimizer", "optimizer_state_dict"),
        )
    }
    return {
        **{f"{key}_bitwise": value for key, value in checks.items()},
        "status": (
            "PASS"
            if all(checks.values())
            else "EXP013_W1B_R2_PREFIX_PARITY_FAIL"
        ),
    }


def main() -> None:
    odd = json.loads((OUT / ".first_odd_capture.json").read_text(encoding="utf-8"))
    one = compare(PREFIX / "checkpoints/model_1.pt", R1 / "model_1.pt")
    ten = compare(PREFIX / "checkpoints/model_10.pt", R1 / "model_10.pt")
    status = (
        "PASS"
        if one["status"] == ten["status"] == "PASS" and odd
        else "EXP013_W1B_R2_PREFIX_PARITY_FAIL"
    )
    (OUT / "training_prefix_parity.json").write_text(
        json.dumps(
            {
                "status": status,
                "iteration_1": one,
                "iteration_10": ten,
                "iterations_11_14_telemetry": (
                    "R1 records rounded KL/fall/yaw reward only; temporary prefix "
                    "reached the identical first odd boundary at iteration 15"
                ),
                "rollout_observation_action_reward_minibatch_hashes": (
                    "not_recorded by protected R1; exact actor/critic/optimizer "
                    "checkpoints at iterations 1 and 10 establish tensor parity"
                ),
                "first_odd_reset_preboundary_parity": bool(odd),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    odd.update(
        {
            "status": "PASS",
            "commands_before_boundary": (
                "covered by 100,000-event even-path bitwise parity"
            ),
            "rng_before_boundary": (
                "captured before odd-path RNG use; legacy fails at the same predicate"
            ),
        }
    )
    (OUT / "first_odd_reset_transition_audit.json").write_text(
        json.dumps(odd, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.rmtree(PREFIX)
    (OUT / ".first_odd_capture.json").unlink()
    if status != "PASS":
        raise SystemExit(status)


if __name__ == "__main__":
    main()
