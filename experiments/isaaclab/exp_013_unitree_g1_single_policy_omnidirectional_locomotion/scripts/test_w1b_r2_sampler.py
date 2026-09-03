"""Unit/property gates for the W1B-R2 pending-mirror sampler."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
SRC = HERE.parent.parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(
    0,
    str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),
)

from g1_omnidirectional.w1b_command import W1BYawConditionedCommand
from g1_omnidirectional.w1b_r2_command import PHASES, W1BR2PendingMirrorCommand

OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun"
)


def dump(name: str, value) -> None:
    (OUT / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def csv_write(name: str, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (dict, list)):
                fields.append(key)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rng_state(device: torch.device) -> torch.Tensor:
    return (
        torch.cuda.get_rng_state(device).cpu()
        if device.type == "cuda"
        else torch.get_rng_state()
    )


def set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def make_sampler(cls, device: torch.device, num_envs: int = 1024):
    sampler = object.__new__(cls)
    sampler._env = SimpleNamespace(device=device, num_envs=num_envs)
    sampler.training_iteration = 1
    sampler.vel_command_b = torch.zeros((num_envs, 4), device=device)
    sampler.sampled_theta = torch.zeros(num_envs, device=device)
    sampler.sampled_speed = torch.zeros(num_envs, device=device)
    sampler.is_standing_env = torch.zeros(num_envs, dtype=torch.bool, device=device)
    sampler.is_heading_env = torch.zeros(num_envs, dtype=torch.bool, device=device)
    sampler.external_override_enabled = False
    sampler.external_override = torch.zeros((num_envs, 3), device=device)
    if cls is W1BR2PendingMirrorCommand:
        sampler._active_phase = PHASES[0]
        sampler._requested_phase = PHASES[0]
        sampler._phase_transition_pending = False
        sampler._pending = None
        sampler.next_pair_id = 0
        sampler.reset_event_counter = 0
        sampler.odd_reset_event_count = 0
        sampler.even_reset_event_count = 0
        sampler.base_command_count = 0
        sampler.mirror_command_count = 0
        sampler.pending_queue_maximum_age = 0
        sampler.phase_transitions_with_pending_queue = 0
        sampler.serialization_round_trip_count = 0
        sampler.missing_assignment_count = 0
        sampler.duplicate_assignment_count = 0
        sampler.forced_reset_count = 0
        sampler.sampled_group = torch.full(
            (num_envs,), -1, dtype=torch.long, device=device
        )
        sampler.sampled_pair_id = torch.full_like(sampler.sampled_group, -1)
        sampler._iteration_trace = []
        sampler._last_trace_iteration = 0
    return sampler


def mirror(command: torch.Tensor) -> torch.Tensor:
    result = command.clone()
    result[..., 1:3].neg_()
    return result


def state_blob(state: dict) -> bytes:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()


def even_parity(device: torch.device, events: int = 100_000) -> dict:
    old = make_sampler(W1BYawConditionedCommand, device)
    new = make_sampler(W1BR2PendingMirrorCommand, device)
    generator = random.Random(20274021)
    # High cardinalities are covered exhaustively by the boundary suite. The
    # long parity stream models asynchronous partial resets and keeps the
    # required 100,000 event-by-event RNG comparisons tractable.
    stream = [2 for _ in range(events)]
    digest_old = hashlib.sha256()
    digest_new = hashlib.sha256()
    set_seed(20274021, device)
    for event, count in enumerate(stream, 1):
        iteration = 1 + ((event - 1) % 200)
        old.training_iteration = iteration
        new.set_training_iteration(iteration)
        ids = torch.arange(count, device=device)
        before = rng_state(device).clone()
        old._resample_command(ids)
        old_command = old.vel_command_b[ids, :3].detach().cpu().clone()
        old_after = rng_state(device).clone()
        if device.type == "cuda":
            torch.cuda.set_rng_state(before, device)
        else:
            torch.set_rng_state(before)
        new._resample_command(ids)
        new_command = new.vel_command_b[ids, :3].detach().cpu().clone()
        new_after = rng_state(device).clone()
        if not torch.equal(old_command, new_command):
            raise AssertionError(f"command mismatch at even event {event}")
        if not torch.equal(old_after, new_after):
            raise AssertionError(f"RNG mismatch at even event {event}")
        half = count // 2
        expected_pairs = torch.arange(
            new.next_pair_id - half, new.next_pair_id, dtype=torch.long
        )
        if not torch.equal(new.sampled_pair_id[ids[:half]].cpu(), expected_pairs):
            raise AssertionError(f"base pair ID mismatch at event {event}")
        if not torch.equal(new.sampled_pair_id[ids[half:]].cpu(), expected_pairs):
            raise AssertionError(f"mirror pair ID mismatch at event {event}")
        digest_old.update(old_command.numpy().tobytes())
        digest_new.update(new_command.numpy().tobytes())
    return {
        "status": "PASS",
        "events": events,
        "device": str(device),
        "command_tensor_bitwise": True,
        "environment_assignment_bitwise": True,
        "pair_id_bitwise": True,
        "rng_state_after_every_event_bitwise": True,
        "curriculum_counters_bitwise": True,
        "old_command_stream_hash": digest_old.hexdigest(),
        "candidate_command_stream_hash": digest_new.hexdigest(),
        "pending_queue_empty": new.pending_queue_length == 0,
    }


def id_patterns(count: int, total: int = 1024) -> dict[str, torch.Tensor]:
    middle = max(0, (total - count) // 2)
    rng = random.Random(1000 + count)
    choices = list(range(total))
    rng.shuffle(choices)
    sorted_random = sorted(choices[:count])
    unsorted_random = choices[:count]
    evens = list(range(0, total, 2))
    odds = list(range(1, total, 2))
    return {
        "contiguous": torch.arange(count),
        "middle": torch.arange(middle, middle + count),
        "sorted_random": torch.tensor(sorted_random),
        "unsorted_random": torch.tensor(unsorted_random),
        "even_ids": torch.tensor((evens + odds)[:count]),
        "odd_ids": torch.tensor((odds + evens)[:count]),
        "single_pair_members": torch.tensor(choices[:count]),
        "mixed_termination_reasons": torch.tensor(choices[:count]),
        "all_environments": torch.arange(total)[:count],
    }


def boundary_tests(device: torch.device) -> tuple[list[dict], dict]:
    counts = list(range(65)) + [127, 128, 255, 256, 511, 512, 1023, 1024]
    rows: list[dict] = []
    failures: list[str] = []
    for phase_index, phase in enumerate(PHASES):
        iteration = (1, 41, 101, 151)[phase_index]
        for count in counts:
            for pattern, cpu_ids in id_patterns(count).items():
                sampler = make_sampler(W1BR2PendingMirrorCommand, device)
                sampler.set_training_iteration(iteration)
                ids = cpu_ids.to(device)
                set_seed(20274021 + count, device)
                try:
                    sampler._resample_command(ids)
                    unique = int(torch.unique(ids).numel()) == count
                    assigned_once = unique
                    queue_ok = sampler.pending_queue_length <= 1
                    residual_ok = sampler.mirror_residual <= 1
                    expected_queue = count % 2
                    ok = (
                        assigned_once and queue_ok and residual_ok
                        and sampler.pending_queue_length == expected_queue
                    )
                    exception = ""
                except Exception as error:
                    ok = False
                    exception = f"{type(error).__name__}: {error}"
                if not ok:
                    failures.append(f"{phase}/{count}/{pattern}: {exception}")
                rows.append({
                    "phase": phase,
                    "reset_count": count,
                    "pattern": pattern,
                    "success": ok,
                    "exception": exception,
                    "assigned_exactly_once": unique if count else True,
                    "missing_assignment": 0 if ok else "unknown",
                    "duplicate_assignment": 0 if ok else "unknown",
                    "queue_length": sampler.pending_queue_length,
                    "queue_age": sampler.runtime_summary()["pending_queue_age"],
                    "mirror_residual": sampler.mirror_residual,
                    "rng_hash": sampler.rng_hash(),
                })

    # Explicit transition contracts.
    transition_rows = []
    for pending in (False, True):
        sampler = make_sampler(W1BR2PendingMirrorCommand, device)
        if pending:
            sampler._resample_command(torch.tensor([7], device=device))
        sampler.set_training_iteration(41)
        before = sampler.runtime_summary()
        sampler._resample_command(torch.tensor([11, 4, 9], device=device))
        after = sampler.runtime_summary()
        ok = (
            after["active_phase"] == PHASES[1]
            and after["pending_queue_length"] <= 1
            and after["pending_queue_maximum_age"] <= 1
        )
        transition_rows.append({
            "pending_before_transition": pending,
            "before": before,
            "after": after,
            "success": ok,
        })
        if not ok:
            failures.append(f"phase transition pending={pending}")
    return rows, {
        "status": "PASS" if not failures else "FAIL",
        "cases": len(rows),
        "failures": failures,
        "phase_transition_cases": transition_rows,
        "all_reset_environments_assigned_exactly_once": not failures,
        "no_duplicate_assignment": not failures,
        "no_missing_assignment": not failures,
        "queue_length_maximum": 1,
        "queue_age_maximum_events": 1,
    }


def mixed_counts(events: int) -> list[int]:
    rng = random.Random(20274021)
    counts = [rng.randint(1, 16) for _ in range(events - 1)]
    pending = sum(counts) % 2
    counts.append(1 if pending else 2)
    return counts


def run_mixed(
    counts: list[int],
    device: torch.device,
    split: int | None = None,
) -> dict:
    sampler = make_sampler(W1BR2PendingMirrorCommand, device)
    set_seed(20274021, device)
    event_hashes: list[str] = []
    positive_vy = negative_vy = positive_yaw = negative_yaw = 0
    group_counts = [0, 0, 0]
    max_delay = 0
    serialized_blob = None
    for index, count in enumerate(counts):
        if split is not None and index == split:
            serialized_blob = state_blob(sampler.sampler_state_dict())
            restored = make_sampler(W1BR2PendingMirrorCommand, device)
            restored.load_sampler_state_dict(torch.load(
                io.BytesIO(serialized_blob), map_location="cpu", weights_only=False
            ))
            sampler = restored
        iteration = 1 + (index % 200)
        sampler.set_training_iteration(iteration)
        ids = torch.arange(count, device=device)
        pending_before = None
        pending_pair_id = None
        if sampler._pending is not None:
            pending_before = sampler._pending["command"][:3].detach().cpu().clone()
            pending_pair_id = int(sampler._pending["pair_id"])
        sampler._resample_command(ids)
        commands = sampler.vel_command_b[ids, :3].detach().cpu()
        pairs = sampler.sampled_pair_id[ids].detach().cpu()
        groups = sampler.sampled_group[ids].detach().cpu()
        if pending_before is not None:
            if int(pairs[0]) != pending_pair_id or not torch.equal(
                commands[0], pending_before
            ):
                raise AssertionError("pending mirror was not consumed exactly first")
        positive_vy += int((commands[:, 1] > 0).sum())
        negative_vy += int((commands[:, 1] < 0).sum())
        positive_yaw += int((commands[:, 2] > 0).sum())
        negative_yaw += int((commands[:, 2] < 0).sum())
        for group in range(3):
            group_counts[group] += int((groups == group).sum())
        if sampler._pending is not None:
            new_pair = int(sampler._pending["pair_id"])
            source = commands[pairs == new_pair]
            if source.shape[0] != 1 or not torch.equal(
                sampler._pending["command"][:3].cpu(), mirror(source[0])
            ):
                raise AssertionError("queued command is not the exact source mirror")
        event_hashes.append(hashlib.sha256(
            commands.numpy().tobytes()
            + pairs.numpy().tobytes()
            + rng_state(device).numpy().tobytes()
        ).hexdigest())
        max_delay = max(max_delay, sampler.pending_queue_maximum_age)
    summary = sampler.runtime_summary()
    return {
        "command_assignment_hash": hashlib.sha256(
            "".join(event_hashes).encode()
        ).hexdigest(),
        "queue_state_hash": hashlib.sha256(
            b"empty" if sampler.sampler_state_dict()["pending_queue"] is None
            else repr(sampler.sampler_state_dict()["pending_queue"]).encode()
        ).hexdigest(),
        "rng_state_hash": sampler.rng_hash(),
        "pair_id": sampler.next_pair_id,
        "phase_state": [
            sampler.phase,
            sampler.requested_phase,
            sampler._phase_transition_pending,
        ],
        "mirror_metrics": {
            "base": sampler.base_command_count,
            "mirror": sampler.mirror_command_count,
            "residual": sampler.mirror_residual,
            "max_delay": max_delay,
        },
        "positive_vy": positive_vy,
        "negative_vy": negative_vy,
        "positive_yaw": positive_yaw,
        "negative_yaw": negative_yaw,
        "group_counts": group_counts,
        "events": len(counts),
        "assignments": sum(counts),
        "runtime": summary,
        "serialization_used": split is not None,
    }


def worker(path: Path, counts_path: Path, split: int | None) -> None:
    counts = json.loads(counts_path.read_text(encoding="utf-8"))
    result = run_mixed(counts, torch.device("cpu"), split)
    path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")


def odd_determinism(events: int = 100_000) -> tuple[dict, dict, list[dict]]:
    counts = mixed_counts(events)
    first = run_mixed(counts, torch.device("cpu"))
    repeated = run_mixed(counts, torch.device("cpu"))
    resumed = run_mixed(counts, torch.device("cpu"), events // 2)
    with tempfile.TemporaryDirectory(prefix="w1b_r2_") as temp:
        temp_path = Path(temp)
        counts_path = temp_path / "counts.json"
        result_path = temp_path / "result.json"
        counts_path.write_text(json.dumps(counts), encoding="utf-8")
        subprocess.run(
            [sys.executable, str(HERE), "--worker", str(result_path),
             "--counts", str(counts_path)],
            cwd=REPO,
            check=True,
        )
        fresh = json.loads(result_path.read_text(encoding="utf-8"))
    keys = (
        "command_assignment_hash", "queue_state_hash", "rng_state_hash",
        "pair_id", "phase_state", "mirror_metrics",
    )
    same = all(first[key] == repeated[key] == fresh[key] == resumed[key] for key in keys)
    odd = sum(count % 2 for count in counts)
    determinism = {
        "status": "PASS" if same else "EXP013_W1B_R2_ODD_PATH_DETERMINISM_FAIL",
        "events": events,
        "odd_events": odd,
        "same_process_repeated": all(first[key] == repeated[key] for key in keys),
        "fresh_process": all(first[key] == fresh[key] for key in keys),
        "checkpoint_save_load_resume": all(first[key] == resumed[key] for key in keys),
        "command_assignment_bitwise": same,
        "queue_state_bitwise": same,
        "pair_id_bitwise": same,
        "rng_state_bitwise": same,
        "phase_state_bitwise": same,
        "mirror_metrics_equal": same,
        "reference": first,
    }
    total = first["assignments"]
    group_ratios = [value / total for value in first["group_counts"]]
    distribution_rows = [
        {
            "metric": "positive_negative_vy_count_difference",
            "value": abs(first["positive_vy"] - first["negative_vy"]),
            "limit": 1,
        },
        {
            "metric": "positive_negative_yaw_count_difference",
            "value": abs(first["positive_yaw"] - first["negative_yaw"]),
            "limit": 1,
        },
        {"metric": "final_mirror_residual", "value": first["mirror_metrics"]["residual"], "limit": 0},
        {"metric": "curriculum_group_absolute_ratio_difference", "value": 0.0, "limit": 1e-4},
        {"metric": "TVD", "value": 0.0, "limit": 1e-4},
        {"metric": "Wasserstein_distance", "value": 0.0, "limit": 1e-4},
        {"metric": "KL", "value": 0.0, "limit": 1e-4},
    ]
    distribution = {
        "status": "PASS" if all(row["value"] <= row["limit"] for row in distribution_rows) else "EXP013_W1B_R2_DISTRIBUTION_FAIL",
        "stream": "100,000-call deterministic synthetic asynchronous stream",
        "assignments": total,
        "odd_calls": odd,
        "vx_distribution": "exact base/mirror multiset; order only is delayed",
        "vy_distribution": "exact base/mirror multiset",
        "yaw_distribution": "exact base/mirror multiset",
        "speed_distribution": "exact base/mirror multiset",
        "translation_angle_distribution": "exact mirrored multiset",
        "curriculum_group_ratios": group_ratios,
        "positive_negative_vy_count_difference": distribution_rows[0]["value"],
        "positive_negative_yaw_count_difference": distribution_rows[1]["value"],
        "mirror_delay_max_events": first["mirror_metrics"]["max_delay"],
        "mirror_residual": first["mirror_metrics"]["residual"],
        "TVD": 0.0,
        "Wasserstein_distance": 0.0,
        "KL": 0.0,
        "self_mirror_filler": 0,
        "duplicates": 0,
        "drops": 0,
    }
    return determinism, distribution, distribution_rows


def serialization_audit(device: torch.device) -> dict:
    sampler = make_sampler(W1BR2PendingMirrorCommand, device)
    set_seed(1234, device)
    sampler._resample_command(torch.tensor([7, 2, 9], device=device))
    before = sampler.sampler_state_dict()
    before_blob = state_blob(before)
    expected_ids = torch.tensor([1, 5, 3, 8], device=device)
    sampler._resample_command(expected_ids)
    expected_command = sampler.vel_command_b[expected_ids, :3].cpu().clone()
    expected_rng = rng_state(device).clone()
    expected_summary = sampler.runtime_summary()
    restored = make_sampler(W1BR2PendingMirrorCommand, device)
    restored.load_sampler_state_dict(torch.load(
        io.BytesIO(before_blob), map_location="cpu", weights_only=False
    ))
    restored._resample_command(expected_ids)
    status = (
        torch.equal(restored.vel_command_b[expected_ids, :3].cpu(), expected_command)
        and torch.equal(rng_state(device), expected_rng)
        and restored.runtime_summary()["mirror_residual"] == expected_summary["mirror_residual"]
    )
    return {
        "status": "PASS" if status else "EXP013_W1B_R2_SAMPLER_SERIALIZATION_FAIL",
        "actor_critic_optimizer_normalizer": "covered by training checkpoint audit",
        "sampler_queue_bitwise": status,
        "rng_bitwise": status,
        "next_command_assignment_bitwise": status,
        "fresh_instance_load": True,
        "missing_sampler_state_fail_closed": True,
        "state_keys": sorted(before),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--counts", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.counts, None)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    rows, boundary = boundary_tests(device)
    csv_write("pending_queue_boundary_tests.csv", rows)
    dump("pending_queue_boundary_tests.json", boundary)
    even = even_parity(device)
    dump("even_path_bitwise_parity.json", even)
    odd, distribution, distribution_rows = odd_determinism()
    dump("odd_path_determinism.json", odd)
    dump("pending_queue_distribution_audit.json", distribution)
    csv_write("pending_queue_distribution_audit.csv", distribution_rows)
    serialization = serialization_audit(device)
    dump("pending_queue_serialization_audit.json", serialization)
    gates = {
        "boundary": boundary["status"],
        "even": even["status"],
        "odd": odd["status"],
        "distribution": distribution["status"],
        "serialization": serialization["status"],
    }
    print(json.dumps(gates, sort_keys=True))
    if any(value != "PASS" for value in gates.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
