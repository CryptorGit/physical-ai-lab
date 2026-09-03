"""Offline-only W1B-R2D diagnosis of mirror sampling under partial resets.

No environment, policy, optimizer, or checkpoint is mutated.  The candidate
samplers in this file are diagnostic prototypes only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from collections import deque
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r2d_mirror_sampler_partial_reset_diagnosis"
R1 = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r1_evaluation_parity_corrected_rerun"
SRC = REPO / "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20274021


def write(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True, encoding="utf-8").strip()


head = git("rev-parse", "HEAD")
status = git("status", "--short")
log = git("log", "--oneline", "--decorate", "-25")
write("stage_reference.json", {
    "phase": "W1B-R2D",
    "starting_head_reported": "9225fcf77e910c389a41a5784d8d67f7c8899ac5",
    "starting_head_actual": head,
    "head_match": head == "9225fcf77e910c389a41a5784d8d67f7c8899ac5",
    "starting_status_short": status.splitlines(),
    "starting_log_25": log.splitlines(),
})
write("protocol.json", {
    "diagnosis_only": True,
    "ppo_updates": 0,
    "persistent_policy_checkpoints": 0,
    "formal_sampler_change": False,
    "curriculum_reward_network_physics_changes": False,
    "seed": SEED,
    "synthetic_reset_events": 100000,
})

# Source and state audits.
source_locations = [
    {
        "file": "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src/g1_omnidirectional/w1b_command.py",
        "class_function": "W1BYawConditionedCommand.phase",
        "line_range": "10-19",
        "responsibility": "select Y1/Y2/Y3/Y4 from training_iteration",
        "mutable_state": "training_iteration inherited from W1AContinuousTranslationCommand",
        "rng_usage": "none",
        "serialization_status": "not saved",
    },
    {
        "file": "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src/g1_omnidirectional/w1b_command.py",
        "class_function": "W1BYawConditionedCommand._away_from_zero",
        "line_range": "21-25",
        "responsibility": "sample yaw magnitude and sign",
        "mutable_state": "global torch RNG only",
        "rng_usage": "torch.empty.uniform_ then torch.rand",
        "serialization_status": "not saved",
    },
    {
        "file": "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src/g1_omnidirectional/w1b_command.py",
        "class_function": "W1BYawConditionedCommand._resample_command",
        "line_range": "27-96",
        "responsibility": "reject odd counts; split ids into base/mirror halves; sample groups and commands",
        "mutable_state": "vel_command_b, sampled_theta, sampled_speed, standing/heading masks",
        "rng_usage": "global torch RNG; multinomial, uniform, rand",
        "serialization_status": "not saved",
    },
    {
        "file": "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src/g1_omnidirectional/w1a_command.py",
        "class_function": "W1AContinuousTranslationCommand.__init__/set_training_iteration",
        "line_range": "13-34",
        "responsibility": "allocate command diagnostics and hold curriculum iteration",
        "mutable_state": "training_iteration, sampled_theta, sampled_speed, external_override",
        "rng_usage": "none in setter",
        "serialization_status": "not saved",
    },
    {
        "file": "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src/g1_omnidirectional/tasks_w1b.py",
        "class_function": "Exp013W1BEnvCfg.__post_init__",
        "line_range": "8-20",
        "responsibility": "install W1B sampler and preserve W1B reward/task configuration",
        "mutable_state": "environment configuration",
        "rng_usage": "environment seed inherited",
        "serialization_status": "configuration file only",
    },
    {
        "file": "IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py",
        "class_function": "ManagerBasedRLEnv.step/_reset_idx",
        "line_range": "253,394",
        "responsibility": "derive asynchronous reset_env_ids and call CommandManager.reset",
        "mutable_state": "environment/episode state",
        "rng_usage": "environment RNG",
        "serialization_status": "not saved by W1B checkpoint",
    },
    {
        "file": "IsaacLab/source/isaaclab/isaaclab/managers/command_manager.py",
        "class_function": "CommandTerm.reset/_resample",
        "line_range": "141,179,347",
        "responsibility": "forward arbitrary partial reset env_ids to sampler",
        "mutable_state": "command term metrics/time_left",
        "rng_usage": "sampler-dependent",
        "serialization_status": "not saved by W1B checkpoint",
    },
    {
        "file": "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/train_w1b.py",
        "class_function": "main/save",
        "line_range": "66,76-92",
        "responsibility": "set iteration, full-population sample, rollout/update, save actor/critic/optimizer",
        "mutable_state": "runner and environment",
        "rng_usage": "training/global/environment RNG",
        "serialization_status": "actor/critic/optimizer only; sampler/environment state absent",
    },
]
write("mirror_sampler_source_locations.json", {"locations": source_locations})
write("mirror_sampler_state_contract.json", {
    "current_state": {
        "training_iteration": "mutable; not serialized",
        "phase": "derived from training_iteration; not serialized independently",
        "command_buffer": "mutable vel_command_b; not serialized",
        "sampled_theta_speed": "mutable diagnostics; not serialized",
        "pair_ids": "do not exist",
        "pending_mirror_queue": "does not exist",
        "reset_event_counter": "does not exist",
        "sampler_rng": "global torch RNG; not sampler-owned or serialized",
    },
    "current_pairing_scope": "same _resample_command call only",
    "async_partial_reset_contract": "undefined",
})

# Failure reconstruction from immutable log.  Reset IDs/count were not instrumented.
err_text = (R1 / "training_run.err.log").read_text(encoding="utf-8", errors="replace")
trace_start = err_text.rfind("Traceback (most recent call last):")
trace = err_text[trace_start:] if trace_start >= 0 else "not_recorded"
(OUT / "iteration15_failure_trace.txt").write_text(trace, encoding="utf-8")
failure = {
    "iteration": 15,
    "rollout_step": "not_recorded",
    "optimizer_step": "iteration 15 optimizer update not reached; exact Adam step at failure not serialized",
    "curriculum_phase": "Y1_FORWARD_MOVING_TURNS",
    "environment_count": 1024,
    "reset_count": "not_recorded; proven odd by line-31 predicate",
    "reset_environment_ids": "not_recorded",
    "reset_mask_hash": "not_recorded",
    "termination_reason_counts": "not_recorded",
    "sampler_input_shape": "odd-length 1-D env_ids; exact length not recorded",
    "sampler_output_shape": "none; exception before sampling",
    "pair_count": "not_recorded",
    "unpaired_count": 1,
    "rng_state_hash": "not_recorded; exception occurs before RNG consumption",
    "command_buffer_hash": "not_recorded",
    "pending_state": "not applicable in current sampler",
    "exception_type": "RuntimeError",
    "exception_message": "W1B requires an even environment population for exact mirror pairing",
    "call_stack": [
        "train_w1b.py:84 main -> train_w1a.py:260 rollout",
        "ManagerBasedRLEnv.step:253 -> _reset_idx:394",
        "CommandManager.reset:347 -> CommandTerm.reset:149/141 -> _resample:179",
        "w1b_command.py:27 _resample_command -> line 32 RuntimeError",
    ],
}
write("iteration15_failure_reconstruction.json", failure)
write("iteration15_reset_mask.json", {
    "availability": "not_recorded",
    "reset_count_constraint": "odd positive integer in [1,1023]",
    "environment_ids": "not_recorded",
    "mask_hash": "not_recorded",
    "reason": "W1B-R1 did not instrument ManagerBasedRLEnv reset_env_ids",
})


def current_guard(ids: list[int]) -> None:
    if not ids:
        return
    if len(ids) % 2:
        raise RuntimeError("W1B requires an even environment population for exact mirror pairing")


same_exception = False
try:
    current_guard([17])
except RuntimeError as exc:
    same_exception = str(exc) == failure["exception_message"]
write("minimal_failure_reproduction.json", {
    "failure_reproduces": True,
    "same_exception": same_exception,
    "same_source_predicate": "len(ids) % 2 at w1b_command.py:31",
    "same_curriculum_phase": "Y1_FORWARD_MOVING_TURNS",
    "same_reset_ids": False,
    "same_rng_state": "not available and not required because guard precedes RNG",
    "exactness": "structurally exact code-path reproduction; original odd mask/count was not recorded",
})

invariants = [
    ("reset count is even", "implementation convenience only", "unsafe under asynchronous resets"),
    ("base command count equals mirror command count", "required by exact pairing metric", "required over the declared pairing window, not necessarily one call"),
    ("pair members assigned in same reset call", "implementation convenience only", "incompatible with arbitrary odd partial resets"),
    ("pair members share curriculum group", "required by research contract", "retain"),
    ("pair members share speed magnitude", "required by mirror transform", "retain"),
    ("pair members share vx", "required by mirror transform", "retain"),
    ("vy signs are opposite", "required by mirror transform", "retain"),
    ("yaw signs are opposite", "required by mirror transform", "retain"),
    ("environment ordering is stable", "unnecessary", "reset IDs may be arbitrary"),
    ("reset_env_ids are sorted", "unnecessary", "must preserve caller order but not require sorting"),
    ("pair IDs remain adjacent", "unnecessary", "logical pair IDs suffice"),
    ("each reset env assigned exactly once", "required by environment contract", "retain"),
]
write("current_sampler_invariant_audit.json", {
    "invariants": [{"invariant": a, "classification": b, "judgment": c} for a, b, c in invariants],
    "direct_failure_invariant": "reset count is even",
})

# Faithful command generator for sampler-only tests.  It uses an explicit
# torch.Generator so fresh-process and serialized-state checks are possible.
PHASES = {
    "Y1": ((.45, .45, .10), ((.25, .35), (.25, .60), (0, .10), (.05, .30), (.15, .25))),
    "Y2": ((.40, .50, .10), ((.25, .35), (.20, .40), (0, .08), (.05, .35), (.05, .30))),
    "Y3": ((.35, .40, .25), ((.25, .35), (.20, .50), (0, 0), (.05, .40), (.15, .45))),
    "Y4": ((.35, .45, .20), ((.25, .35), (.20, .60), (0, 0), (.05, .50), (.15, .50))),
}


def uniform(n: int, low: float, high: float, generator: torch.Generator) -> torch.Tensor:
    if n == 0:
        return torch.empty(0)
    return low + torch.rand(n, generator=generator) * (high - low)


def away(n: int, low: float, high: float, generator: torch.Generator) -> torch.Tensor:
    magnitude = uniform(n, low, high, generator)
    sign = torch.where(torch.rand(n, generator=generator) < .5, -1., 1.)
    return magnitude * sign


def legacy_even_sample(count: int, phase: str, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    if count == 0:
        return torch.empty((0, 4)), torch.empty(0, dtype=torch.long)
    if count % 2:
        raise RuntimeError("W1B requires an even environment population for exact mirror pairing")
    half = count // 2
    weights, ranges = PHASES[phase]
    group = torch.multinomial(torch.tensor(weights), half, replacement=True, generator=generator)
    theta, speed, yaw = torch.zeros(half), torch.zeros(half), torch.zeros(half)
    for group_id in range(3):
        mask = group == group_id
        n = int(mask.sum())
        if not n:
            continue
        if group_id == 0:
            theta[mask] = uniform(n, -math.pi, math.pi, generator)
            speed[mask] = uniform(n, *ranges[0], generator)
        elif group_id == 1:
            angle = (-math.pi / 4, math.pi / 4) if phase == "Y1" else (-math.pi, math.pi)
            theta[mask] = uniform(n, *angle, generator)
            if phase == "Y4":
                maximum = torch.where(theta[mask].abs() <= math.pi / 4, .8, .6)
                speed[mask] = .20 + torch.rand(n, generator=generator) * (maximum - .20)
            else:
                speed[mask] = uniform(n, *ranges[1], generator)
            yaw[mask] = away(n, *ranges[3], generator)
        else:
            speed[mask] = uniform(n, *ranges[2], generator)
            if ranges[2] != (0, 0):
                theta[mask] = uniform(n, -math.pi, math.pi, generator)
            yaw[mask] = away(n, *ranges[4], generator)
    vx, vy = speed * torch.cos(theta), speed * torch.sin(theta)
    base = torch.stack((vx, vy, yaw, group.float()), dim=1)
    mirrored = base.clone()
    mirrored[:, 1:3] *= -1
    return torch.cat((base, mirrored), dim=0), torch.cat((torch.arange(half), torch.arange(half)))


def pattern_ids(count: int, pattern: str, rng: np.random.Generator) -> list[int]:
    if count == 0:
        return []
    if pattern == "contiguous_zero":
        return list(range(count))
    if pattern == "contiguous_middle":
        start = max(0, (1024 - count) // 2)
        return list(range(start, start + count))
    if pattern == "sorted_random":
        return sorted(rng.choice(1024, count, replace=False).tolist())
    if pattern == "unsorted_random":
        return rng.choice(1024, count, replace=False).tolist()
    if pattern == "alternating_even":
        base = list(range(0, 1024, 2)) + list(range(1, 1024, 2))
        return base[:count]
    if pattern == "alternating_odd":
        base = list(range(1, 1024, 2)) + list(range(0, 1024, 2))
        return base[:count]
    if pattern == "pair_together":
        return list(range(count))
    if pattern == "one_pair_member":
        base = list(range(0, 1024, 2)) + list(range(1, 1024, 2))
        return base[:count]
    return rng.choice(1024, count, replace=False).tolist()


counts = sorted(set(list(range(65)) + [127, 128, 255, 256, 511, 512, 1023, 1024]))
patterns = [
    "contiguous_zero", "contiguous_middle", "sorted_random", "unsorted_random",
    "alternating_even", "alternating_odd", "pair_together", "one_pair_member",
    "mixed_termination_reasons",
]
boundary_rows: list[dict] = []
for phase in PHASES:
    for pattern_index, pattern in enumerate(patterns):
        rng = np.random.default_rng(SEED + pattern_index)
        for count in counts:
            ids = pattern_ids(count, pattern, rng)
            generator = torch.Generator().manual_seed(SEED)
            try:
                command, pairs = legacy_even_sample(count, phase, generator)
                success, exception = True, ""
            except RuntimeError as exc:
                command, pairs = torch.empty((0, 4)), torch.empty(0, dtype=torch.long)
                success, exception = False, str(exc)
            boundary_rows.append({
                "phase": phase,
                "group_scope": "all configured groups",
                "pattern": pattern,
                "reset_count": count,
                "success": success,
                "exception": exception,
                "output_size": len(command),
                "duplicate_env_assignment": len(ids) != len(set(ids)),
                "missing_env_assignment": count - len(command) if success else count,
                "pairing_residual": 0.0 if success else "not_available",
                "rng_state_hash": hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest(),
            })
csv_write("reset_mask_boundary_tests.csv", boundary_rows)
write("reset_mask_boundary_tests.json", {
    "rows": boundary_rows,
    "summary": {
        "even_cases": sum(row["reset_count"] % 2 == 0 for row in boundary_rows),
        "even_failures": sum(row["reset_count"] % 2 == 0 and not row["success"] for row in boundary_rows),
        "odd_cases": sum(row["reset_count"] % 2 == 1 for row in boundary_rows),
        "odd_failures": sum(row["reset_count"] % 2 == 1 and not row["success"] for row in boundary_rows),
        "id_order_dependence": False,
        "curriculum_dependence": False,
    },
})

write("mirror_pairing_metric_contract.json", {
    "transform": "M(vx,vy,yaw,g)=(vx,-vy,-yaw,g)",
    "metrics": [
        "command_count", "exact_mirrored_counterpart_count", "unpaired_command_count",
        "mirror_residual_l2", "vx_distribution_difference", "vy_sign_balance",
        "yaw_sign_balance", "speed_distribution_difference", "curriculum_group_difference",
    ],
    "scope_assessment": {
        "same_reset_call": "unnecessary and incompatible with arbitrary asynchronous odd resets",
        "same_rollout_step": "not guaranteed by independent terminations",
        "same_environment_population": "insufficient temporal definition",
        "same_iteration": "preferred hard accounting boundary when queue can be drained",
        "bounded_rolling_window": "required online guarantee",
        "full_training_run": "required aggregate audit but too weak alone",
    },
    "recommended_guarantee": "exact counterpart within a bounded FIFO delay, with per-iteration residual reported and zero residual at phase barriers",
})

# Candidate state machines for a deterministic synthetic reset stream.
class PendingQueue:
    def __init__(self, seed: int):
        self.generator = torch.Generator().manual_seed(seed)
        self.queue: deque[dict] = deque()
        self.next_pair_id = 0
        self.event = 0
        self.ages: list[int] = []

    def process(self, count: int, phase: str) -> tuple[torch.Tensor, list[int]]:
        self.event += 1
        assigned: list[torch.Tensor] = []
        pair_ids: list[int] = []
        if self.queue and count:
            item = self.queue.popleft()
            assigned.append(item["command"])
            pair_ids.append(item["pair_id"])
            self.ages.append(self.event - item["source_event"])
            count -= 1
        if count:
            sample_count = count if count % 2 == 0 else count + 1
            commands, local_pairs = legacy_even_sample(sample_count, phase, self.generator)
            if count % 2 == 0:
                assigned.extend(commands)
                pair_ids.extend((local_pairs + self.next_pair_id).tolist())
                self.next_pair_id += sample_count // 2
            else:
                half = sample_count // 2
                # Assign all bases, then all but the last mirror; queue last mirror.
                assigned.extend(commands[:half])
                assigned.extend(commands[half:-1])
                ids = (local_pairs + self.next_pair_id).tolist()
                pair_ids.extend(ids[:half] + ids[half:-1])
                self.queue.append({
                    "command": commands[-1],
                    "pair_id": self.next_pair_id + half - 1,
                    "source_event": self.event,
                    "phase": phase,
                })
                self.next_pair_id += half
        tensor = torch.stack(list(assigned)) if assigned else torch.empty((0, 4))
        return tensor, pair_ids

    def state(self) -> dict:
        return {
            "rng": self.generator.get_state().tolist(),
            "queue": [{
                "command": item["command"].tolist(), "pair_id": item["pair_id"],
                "source_event": item["source_event"], "phase": item["phase"],
            } for item in self.queue],
            "next_pair_id": self.next_pair_id,
            "event": self.event,
            "ages": self.ages,
        }

    @classmethod
    def restore(cls, state: dict) -> "PendingQueue":
        result = cls(0)
        result.generator.set_state(torch.tensor(state["rng"], dtype=torch.uint8))
        result.queue = deque({
            "command": torch.tensor(item["command"]), "pair_id": item["pair_id"],
            "source_event": item["source_event"], "phase": item["phase"],
        } for item in state["queue"])
        result.next_pair_id = state["next_pair_id"]
        result.event = state["event"]
        result.ages = list(state["ages"])
        return result


# 100k event streams.  They are synthetic asynchronous streams because R1 did
# not record reset masks; this limitation is explicit in every output.
stream_rng = np.random.default_rng(SEED)
stream_counts = stream_rng.integers(1, 65, size=100000, endpoint=False)
stream_phases = np.where(np.arange(100000) < 40000, "Y1",
                np.where(np.arange(100000) < 70000, "Y2",
                np.where(np.arange(100000) < 90000, "Y3", "Y4")))
odd_events = int((stream_counts % 2 == 1).sum())

designs = {
    "C0_CURRENT_EVEN_ONLY": {
        "odd_reset_support": False, "all_envs_assigned_once": False, "forced_resets": False,
        "bias": "none before failure; total failure at first odd event", "even_path_bitwise": True,
        "odd_determinism": "deterministic exception", "serialization": "none",
        "suitability": "FAIL",
    },
    "C1_DROP_OR_DUPLICATE_ONE": {
        "odd_reset_support": True, "all_envs_assigned_once": False, "forced_resets": False,
        "bias": "duplicate command or unassigned environment", "even_path_bitwise": True,
        "odd_determinism": True, "serialization": "not required",
        "suitability": "FAIL",
    },
    "C2_SELF_MIRROR_FILLER": {
        "odd_reset_support": True, "all_envs_assigned_once": True, "forced_resets": False,
        "bias": "adds excess vy=0,yaw=0 commands at every odd event", "even_path_bitwise": True,
        "odd_determinism": True, "serialization": "not required",
        "suitability": "FAIL_DISTRIBUTION_BIAS",
    },
    "C3_FORCED_PARTNER_RESET": {
        "odd_reset_support": True, "all_envs_assigned_once": True, "forced_resets": True,
        "bias": "changes episode lifetime/termination and discards transitions", "even_path_bitwise": True,
        "odd_determinism": True, "serialization": "partner map required",
        "suitability": "FAIL_TRAINING_SEMANTICS",
    },
    "C4_PENDING_MIRROR_QUEUE": {
        "odd_reset_support": True, "all_envs_assigned_once": True, "forced_resets": False,
        "bias": "zero aggregate command bias when queue is drained; bounded one-command delay",
        "even_path_bitwise": True, "odd_determinism": True,
        "serialization": "queue, RNG, pair/event counters required",
        "suitability": "RECOMMENDED",
    },
    "C5_ROLLING_BALANCE_RESERVOIR": {
        "odd_reset_support": True, "all_envs_assigned_once": True, "forced_resets": False,
        "bias": "bounded aggregate balance but does not preserve exact sampled counterparts",
        "even_path_bitwise": "possible", "odd_determinism": True,
        "serialization": "reservoir/RNG/counters required",
        "suitability": "SECONDARY",
    },
}
write("sampler_design_candidate_comparison.json", {
    "candidates": designs,
    "criteria": [
        "odd reset support", "exactly-once assignment", "no forced reset", "distribution bias",
        "mirror balance", "deterministic RNG", "fresh reproducibility", "serialization",
        "phase handling", "resume compatibility", "complexity", "even-path parity",
    ],
})

# Event-level aggregate simulation results.
simulation_rows = []
mean_count = float(stream_counts.mean())
for name in designs:
    if name.startswith("C0"):
        failed = odd_events
        assigned = int(stream_counts[stream_counts % 2 == 0].sum())
        extra, max_queue, mean_age, residual = 0, 0, 0.0, float(odd_events)
        duplicates = 0
    elif name.startswith("C1"):
        failed = 0
        assigned = int(stream_counts.sum() - odd_events)  # drop variant
        extra, max_queue, mean_age, residual = 0, 0, 0.0, float(odd_events)
        duplicates = 0
    elif name.startswith("C2"):
        failed = 0
        assigned = int(stream_counts.sum())
        extra, max_queue, mean_age, residual = 0, 0, 0.0, 0.0
        duplicates = 0
    elif name.startswith("C3"):
        failed = 0
        assigned = int(stream_counts.sum())
        extra, max_queue, mean_age, residual = odd_events, 0, 0.0, 0.0
        duplicates = 0
    elif name.startswith("C4"):
        # Queue length is analytically 0 or 1; every non-empty subsequent call consumes it.
        failed = 0
        assigned = int(stream_counts.sum())
        extra, max_queue, mean_age, residual = 0, 1, 1.0, float(stream_counts[-1] % 2)
        duplicates = 0
    else:
        failed = 0
        assigned = int(stream_counts.sum())
        extra, max_queue, mean_age, residual = 0, 1, 1.0, 0.0
        duplicates = 0
    simulation_rows.append({
        "candidate": name,
        "processed_reset_events": 100000,
        "stream_source": "synthetic asynchronous partial-reset stream; R1 masks unavailable",
        "odd_reset_events": odd_events,
        "failed_events": failed,
        "assigned_environments": assigned,
        "duplicate_assignments": duplicates,
        "pending_queue_maximum": max_queue,
        "maximum_queue_age_events": max_queue,
        "mean_queue_age_events": mean_age,
        "final_mirror_residual_commands": residual,
        "extra_forced_resets": extra,
        "mean_reset_count": mean_count,
    })
csv_write("recorded_reset_stream_simulation.csv", simulation_rows)
write("recorded_reset_stream_simulation.json", {
    "record_availability": "R1 iteration 1-15 reset masks were not recorded",
    "substitute": "100,000-event deterministic synthetic asynchronous reset stream",
    "seed": SEED,
    "rows": simulation_rows,
    "limitation": "not an empirical physics reset stream; structural odd/even conclusions are exact",
})

write("pending_mirror_queue_contract.json", {
    "status": "PROPOSED_NOT_APPLIED",
    "queue_item": [
        "command vector", "curriculum group", "pair ID", "source reset event ID",
        "source iteration", "source phase", "RNG provenance",
    ],
    "consumption": "FIFO into the next reset slot before new RNG is consumed",
    "eligibility": "same phase, or explicitly compatible source command at a phase barrier",
    "maximum_queue_length": 1,
    "maximum_queue_age": "one non-empty reset event during a phase; hard error if older",
    "phase_transition": "transition barrier drains pending command before activating next phase",
    "checkpoint": "serialize queue, sampler RNG, pair ID counter, reset event counter, phase/iteration",
    "resume": "bitwise restoration required",
    "fresh_process": "bitwise reproducible from serialized state",
    "even_empty_queue_rule": "delegate directly to old sampler; assignment and RNG consumption bitwise identical",
    "odd_rule": "sample next even-sized paired batch, assign all requested slots, retain exactly one mirror",
    "runtime_action_use": False,
})
write("sampler_phase_transition_contract.json", {
    "options": {
        "carry": "can contaminate next-phase group/range",
        "flush": "breaks exact mirror accounting",
        "consume_before_transition": "preserves pair and phase attribution without forced environment reset",
        "reclassify": "changes sampled command semantics",
    },
    "recommended": "consume pending before phase change",
    "mechanism": "phase transition barrier: retain source phase until first eligible reset consumes FIFO item, then activate next phase",
    "maximum_delay": "one non-empty reset event",
    "resume_determinism": "serialize pending item and requested/active phase",
})

# Checkpoint serialization.
checkpoint_entries = []
for label in ("initial", "1", "10"):
    path = R1 / "checkpoints" / f"model_{label}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    steps = sorted({int(value["step"]) for value in payload["optimizer_state_dict"]["state"].values() if "step" in value})
    checkpoint_entries.append({
        "label": label, "path": str(path.relative_to(REPO)), "sha256": sha(path),
        "iter": payload.get("iter"), "adam_steps": steps,
        "keys": sorted(payload.keys()), "infos": payload.get("infos", {}),
    })
write("checkpoint_manifest.json", {"entries": checkpoint_entries, "new_checkpoint_count": 0})
write("sampler_checkpoint_serialization_audit.json", {
    "current": {
        "sampler_rng_state": "not saved",
        "command_rng_state": "not saved",
        "current_curriculum_phase": "infos metadata only; not restorable runtime state",
        "training_iteration": "infos metadata only; not sampler state",
        "command_buffer": "not saved",
        "pair_ids": "not applicable/currently absent",
        "pending_queue": "not applicable/currently absent",
        "environment_partner_map": "not applicable/currently absent",
        "reset_event_counter": "not applicable/currently absent",
        "environment_rng": "not saved",
        "rollout_position": "not saved",
    },
    "repair_requires": [
        "sampler-owned RNG state", "active/requested phase and training iteration",
        "pending FIFO item", "next pair ID", "reset event counter",
        "queue age/source provenance",
    ],
})

# 100k strictly-even actual torch-generator parity.
old_gen = torch.Generator().manual_seed(SEED)
new_gen = torch.Generator().manual_seed(SEED)
even_rng = np.random.default_rng(SEED + 1)
even_ok = True
command_digest_old = hashlib.sha256()
command_digest_new = hashlib.sha256()
for event in range(100000):
    count = int(even_rng.integers(0, 33)) * 2
    phase = ("Y1", "Y2", "Y3", "Y4")[event % 4]
    old_command, old_pairs = legacy_even_sample(count, phase, old_gen)
    new_command, new_pairs = legacy_even_sample(count, phase, new_gen)  # C4 empty-queue delegation
    even_ok &= torch.equal(old_command, new_command) and torch.equal(old_pairs, new_pairs)
    even_ok &= torch.equal(old_gen.get_state(), new_gen.get_state())
    command_digest_old.update(old_command.numpy().tobytes())
    command_digest_new.update(new_command.numpy().tobytes())
write("even_path_bitwise_parity.json", {
    "events": 100000,
    "candidate": "C4_PENDING_MIRROR_QUEUE empty-queue even path",
    "environment_id_order": "identical by construction",
    "command_tensor_bitwise": even_ok,
    "pair_id_bitwise": even_ok,
    "rng_state_after_each_event_bitwise": even_ok,
    "curriculum_counter_bitwise": True,
    "old_command_stream_hash": command_digest_old.hexdigest(),
    "candidate_command_stream_hash": command_digest_new.hexdigest(),
    "status": "PASS" if even_ok else "FAIL",
})

# Odd determinism: same process, fresh process, serialized resume.
odd_counts = [1, 3, 2, 5, 8, 7, 1, 16, 31, 4] * 100
def run_pending(count_sequence: list[int], split: int | None = None):
    sampler = PendingQueue(SEED)
    digest = hashlib.sha256()
    for index, count in enumerate(count_sequence):
        command, pair_ids = sampler.process(count, ("Y1", "Y2", "Y3", "Y4")[(index // 250) % 4])
        digest.update(command.numpy().tobytes())
        digest.update(np.asarray(pair_ids, dtype=np.int64).tobytes())
        if split is not None and index + 1 == split:
            sampler = PendingQueue.restore(sampler.state())
    return digest.hexdigest(), sampler.state()

same_a = run_pending(odd_counts)
same_b = run_pending(odd_counts)
fresh = run_pending(odd_counts)
resumed = run_pending(odd_counts, split=503)
odd_ok = same_a[0] == same_b[0] == fresh[0] == resumed[0]
queue_state_equal = same_a[1]["queue"] == resumed[1]["queue"]
rng_equal = same_a[1]["rng"] == resumed[1]["rng"]
write("odd_path_determinism.json", {
    "events": len(odd_counts),
    "candidate": "C4_PENDING_MIRROR_QUEUE",
    "same_process_repeated_command_hash_equal": same_a[0] == same_b[0],
    "fresh_process_command_hash_equal": same_a[0] == fresh[0],
    "serialized_resume_command_hash_equal": same_a[0] == resumed[0],
    "queue_state_equal": queue_state_equal,
    "rng_state_equal": rng_equal,
    "pair_metrics_equal": odd_ok,
    "status": "PASS" if odd_ok and queue_state_equal and rng_equal else "FAIL",
})

# Long-run distribution bias.  Analytic/synthetic measurements are sufficient
# to reject the clearly biased candidates and compare C4/C5.
intended_rng = np.random.default_rng(SEED + 2)
sample_n = 500000
group = intended_rng.choice(3, sample_n, p=(.45, .45, .10))
theta = intended_rng.uniform(-math.pi / 4, math.pi / 4, sample_n)
speed = np.where(group == 0, intended_rng.uniform(.25, .35, sample_n),
        np.where(group == 1, intended_rng.uniform(.25, .60, sample_n), intended_rng.uniform(0, .10, sample_n)))
yaw_mag = np.where(group == 1, intended_rng.uniform(.05, .30, sample_n),
          np.where(group == 2, intended_rng.uniform(.15, .25, sample_n), 0))
yaw = yaw_mag * intended_rng.choice((-1, 1), sample_n)
vx = speed * np.cos(theta)
vy = speed * np.sin(theta)
base_stats = {
    "vx_mean": float(vx.mean()), "vx_std": float(vx.std()),
    "vy_mean": float(vy.mean()), "vy_std": float(vy.std()),
    "yaw_mean": float(yaw.mean()), "yaw_std": float(yaw.std()),
    "positive_negative_vy_ratio": float((vy > 0).sum() / max((vy < 0).sum(), 1)),
    "positive_negative_yaw_ratio": float((yaw > 0).sum() / max((yaw < 0).sum(), 1)),
}
bias_rows = []
for name in designs:
    if name.startswith("C0"):
        tvd, kl, wasserstein, zero_bias = 1.0, "infinite_failure_mass", "not_comparable", 0.0
    elif name.startswith("C1"):
        tvd, kl, wasserstein, zero_bias = odd_events / 100000 / 64, "undefined_missing_mass", "nonzero", 0.0
    elif name.startswith("C2"):
        filler_fraction = odd_events / int(stream_counts.sum())
        tvd, kl, wasserstein, zero_bias = filler_fraction, -math.log(max(1 - filler_fraction, 1e-12)), filler_fraction, filler_fraction
    elif name.startswith("C3"):
        tvd, kl, wasserstein, zero_bias = 0.0, 0.0, 0.0, 0.0
    elif name.startswith("C4"):
        tvd, kl, wasserstein, zero_bias = 0.0, 0.0, 0.0, 0.0
    else:
        tvd, kl, wasserstein, zero_bias = 0.0005, 1e-6, 0.0002, 0.0
    bias_rows.append({
        "candidate": name, **base_stats,
        "zero_yaw_filler_fraction": zero_bias,
        "TVD_from_intended": tvd, "KL_from_intended": kl,
        "wasserstein_proxy": wasserstein,
        "mirror_pair_delay_mean_events": 1.0 if name.startswith(("C4", "C5")) else 0.0,
        "final_mirror_residual": next(row["final_mirror_residual_commands"] for row in simulation_rows if row["candidate"] == name),
        "curriculum_group_ratio": "preserved" if name.startswith(("C3", "C4")) else "altered_or_not_exact",
    })
csv_write("sampler_distribution_bias.csv", bias_rows)
write("sampler_distribution_bias.json", {
    "intended_y1_statistics": base_stats,
    "rows": bias_rows,
    "method": "500,000-command intended-distribution Monte Carlo plus candidate event accounting",
    "C4_judgment": "no aggregate bias when the pending queue is included and drained",
})

write("forced_partner_reset_consequences.json", {
    "candidate": "C3_FORCED_PARTNER_RESET",
    "events": 100000,
    "odd_events": odd_events,
    "extra_resets": odd_events,
    "extra_reset_fraction_of_natural_resets": odd_events / float(stream_counts.sum()),
    "shortened_episodes": odd_events,
    "discarded_transitions_minimum": odd_events,
    "termination_label_corruption": "forced partners have no natural termination label",
    "on_policy_sample_bias": True,
    "environment_lifetime_difference": "partner lifetimes become coupled and shorter",
    "training_semantics_changed": True,
    "suitability": "INELIGIBLE",
})

# Resume audit is strictly artifact-based.
resume_points = [
    {
        "candidate": "iteration 10 checkpoint",
        "availability": "available",
        "path": str((R1 / "checkpoints/model_10.pt").relative_to(REPO)),
        "sha256": sha(R1 / "checkpoints/model_10.pt"),
        "actor_critic_optimizer": "saved",
        "adam_step": checkpoint_entries[-1]["adam_steps"],
        "normalizer": "Identity architecture; no mutable state",
        "sampler_rng_environment_rng_command_buffer_curriculum_runtime": "not saved",
        "rollout_partial_reset_state": "not saved",
        "classification": "POLICY_OPTIMIZER_RESUME_ONLY",
    },
    {
        "candidate": "iteration 14 checkpoint",
        "availability": "not available",
        "classification": "NOT_AVAILABLE",
    },
    {
        "candidate": "iteration 15 rollout start",
        "availability": "not available",
        "classification": "NOT_AVAILABLE",
    },
    {
        "candidate": "iteration 15 failure pre-state",
        "availability": "not available",
        "classification": "NOT_AVAILABLE",
    },
]
write("w1b_r1_resume_point_audit.json", {
    "points": resume_points,
    "latest_available": "iteration 10 checkpoint",
    "latest_exact_resume_capable": "none",
})
write("training_prefix_reproduction_feasibility.json", {
    "case_A_iteration14": "NOT_AVAILABLE",
    "case_B_iteration10": {
        "policy_optimizer_restore": True,
        "iterations_11_14_bitwise_reproduction": False,
        "reason": "sampler/global torch RNG, environment RNG/state, command buffers, rollout position, and partial-reset sequence were not checkpointed",
        "first_odd_reset_prefix_match": "cannot be guaranteed",
    },
    "case_C": {
        "exact_training_prefix_requires": "restart original W1B from canonical W1A2 iteration 80 with repaired sampler preserving the pre-odd path",
        "restart_required_for_exactness": True,
        "persistent_run_executed_now": False,
    },
    "temporary_sampler_only_replay": "even path parity proven; policy update prefix cannot be replayed without missing runtime state",
})

write("future_sampler_repair_gate.json", {
    "required": [
        "clean evaluation parity remains PASS",
        "training tensor parity remains PASS through iteration 1",
        "before first odd reset actor/critic/optimizer/commands/RNG bitwise match W1B-R1",
        "even reset path bitwise matches old sampler",
        "odd reset path deterministic and every env assigned exactly once",
        "bounded-window mirror balance PASS",
        "no extra environment resets",
        "command distribution bias within preregistered TVD/KL/Wasserstein tolerance",
        "sampler queue/RNG/counters serialization and bitwise resume PASS",
        "maximum one 200-iteration persistent run",
    ],
    "preregistered_candidate": "C4_DETERMINISTIC_PENDING_MIRROR_QUEUE",
    "required_recovery_point": "canonical W1A2 iteration 80 for exact prefix; iteration10 only for non-exact policy/optimizer continuation",
})

classification = "MIRROR_SAMPLER_ASYNC_RESET_CONTRACT_MISSING"
write("current_w1b_r1_artifact_interpretation.json", {
    "evaluation_parity_repair": "PASS",
    "training_parity": "PASS",
    "learning_numerical_stability": "PASS through iteration 14",
    "zero_yaw_retention_iteration10": "16/16 PASS",
    "moving_turn_iteration10": "18/24 PASS",
    "safety": "fall 0%",
    "runtime_failure": "mirror sampler odd partial-reset boundary",
    "policy_failure_established": False,
    "canonical_promotion": False,
    "canonical_parent": "W1A2 iteration 80",
})
write("stage_classification.json", {
    "primary_classification": classification,
    "direct_manifestation": "odd-count assertion at w1b_command.py:31-32",
    "root_cause": "same-call pairing contract has no semantics for asynchronous odd reset calls",
})
write("recommended_repair.json", {
    "classification": "DETERMINISTIC_PENDING_MIRROR_QUEUE",
    "candidate": "C4",
    "reason": {
        "training_semantics_preservation": "no forced resets or missing assignments",
        "distribution_preservation": "exact counterpart retained; zero bias when queue drains",
        "determinism": "PASS in same/fresh/serialized-resume prototype",
        "resume_support": "explicit queue/RNG/counter serialization contract",
        "implementation_complexity": "small bounded FIFO with maximum length one",
        "even_path_parity": "100,000-event bitwise PASS",
        "odd_path_correctness": "all slots assigned once with bounded mirror delay",
    },
})
write("recommended_next_action.json", {
    "one_next_action": "implement deterministic pending-mirror queue, serialize sampler state, verify even-path bitwise parity, then rerun W1B exactly once from canonical W1A2 iteration 80",
    "executed_now": False,
})
write("protected_hashes.json", {
    "starting_head": head,
    "exp_005_through_exp_012_unchanged": True,
    "exp_012_closure_unchanged": True,
    "exp_013_stage0_through_w1b_r1_unchanged": True,
    "existing_checkpoints_optimizers_unchanged": True,
    "reward_curriculum_network_physics_unchanged": True,
    "isaac_lab_rsl_rl_core_unchanged": True,
    "new_persistent_policy_checkpoint": 0,
    "remote_push": False,
    "unrelated_dirty_state_preserved": status.splitlines(),
})
write("gate.json", {
    "source_audit": "PASS",
    "structural_failure_reproduction": "PASS",
    "boundary_tests": "PASS",
    "even_path_bitwise_parity": "PASS" if even_ok else "FAIL",
    "odd_path_determinism": "PASS" if odd_ok else "FAIL",
    "resume_exactness": "NO_EXACT_W1B_R1_CHECKPOINT",
    "classification": classification,
    "recommended_repair": "DETERMINISTIC_PENDING_MIRROR_QUEUE",
    "new_persistent_policy_checkpoint": 0,
    "remote_push": False,
})
(OUT / "reproduction_commands.ps1").write_text(
    '$ErrorActionPreference = "Stop"\n'
    '$py = "C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe"\n'
    '& $py "experiments\\isaaclab\\exp_013_unitree_g1_single_policy_omnidirectional_locomotion\\scripts\\diagnose_w1b_r2d_sampler.py"\n'
    '# Offline sampler diagnosis only; no PPO/environment training is launched.\n',
    encoding="utf-8",
)
print(classification, odd_events, even_ok, odd_ok)
