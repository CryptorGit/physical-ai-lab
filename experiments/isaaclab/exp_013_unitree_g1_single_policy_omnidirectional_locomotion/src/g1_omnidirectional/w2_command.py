"""Deterministic mirror-paired physical-command sequence sampler for W2."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

import torch

from g1_omnidirectional.w1b_r2_command import W1BR2PendingMirrorCommand
from g1_omnidirectional.yaw_calibration import calibrate_yaw


PHASES = (
    "T1_START_STOP_SPEED",
    "T2_DIRECTION_CHANGES",
    "T3_LARGE_REVERSALS",
    "T4_COMBINED_TRANSITIONS",
    "T5_RANDOM_SEQUENCE_CONSOLIDATION",
)
MAX_SEGMENTS = 10


def minimum_jerk(progress: torch.Tensor) -> torch.Tensor:
    x = progress.clamp(0.0, 1.0)
    return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


def _tensor_sha(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


class W2DynamicSequenceCommand(W1BR2PendingMirrorCommand):
    """One actor, physical-target ramps, and exact mirrored sequences."""

    state_version = 2

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n = self.num_envs
        self._active_phase = PHASES[0]
        self._requested_phase = PHASES[0]
        self._phase_transition_pending = False
        self._pending_sequence: dict[str, Any] | None = None
        self.physical_command_b = torch.zeros((n, 3), device=self.device)
        self.actor_command_b = self.vel_command_b
        self.sequence_targets = torch.zeros((n, MAX_SEGMENTS, 3), device=self.device)
        self.sequence_hold_s = torch.zeros((n, MAX_SEGMENTS), device=self.device)
        self.sequence_ramp_s = torch.zeros((n, MAX_SEGMENTS), device=self.device)
        self.sequence_segment_count = torch.ones(n, dtype=torch.long, device=self.device)
        self.sequence_segment_index = torch.zeros(n, dtype=torch.long, device=self.device)
        self.sequence_elapsed_s = torch.zeros(n, device=self.device)
        self.sequence_id = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.transition_id = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.transition_type = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.next_sequence_id = 0
        self.next_transition_id = 0
        self.sequence_base_count = 0
        self.sequence_mirror_count = 0
        self.pending_sequence_maximum_age = 0
        self.sequence_serialization_round_trip_count = 0
        self._legacy_parent_restored = False
        self.external_physical_override = torch.zeros((n, 3), device=self.device)

    @staticmethod
    def phase_for_iteration(iteration: int) -> str:
        if iteration <= 40:
            return PHASES[0]
        if iteration <= 90:
            return PHASES[1]
        if iteration <= 150:
            return PHASES[2]
        if iteration <= 210:
            return PHASES[3]
        return PHASES[4]

    @property
    def pending_queue_length(self) -> int:
        return 0 if self._pending_sequence is None else 1

    @property
    def mirror_residual(self) -> int:
        return self.sequence_base_count - self.sequence_mirror_count

    def set_training_iteration(self, iteration: int) -> None:
        self.training_iteration = int(iteration)
        requested = self.phase_for_iteration(iteration)
        self._requested_phase = requested
        if requested == self._active_phase:
            return
        if self._pending_sequence is None:
            self._active_phase = requested
            self._phase_transition_pending = False
        else:
            self._phase_transition_pending = True
            self.phase_transitions_with_pending_queue += 1

    @property
    def phase(self) -> str:
        return self._active_phase

    def _rand(self, count: int, low: float, high: float) -> torch.Tensor:
        return torch.empty(count, device=self.device).uniform_(low, high)

    def _command(self, speed: torch.Tensor, theta: torch.Tensor,
                 yaw: torch.Tensor) -> torch.Tensor:
        return torch.stack((speed * torch.cos(theta), speed * torch.sin(theta), yaw), -1)

    def _random_command(self, count: int, speed_low: float = 0.0,
                        speed_high: float = 0.4, yaw_high: float = 0.35) -> torch.Tensor:
        theta = self._rand(count, -math.pi, math.pi)
        speed = self._rand(count, speed_low, speed_high)
        yaw = self._rand(count, -yaw_high, yaw_high)
        return self._command(speed, theta, yaw)

    def _sample_descriptor(self, count: int, phase: str) -> dict[str, torch.Tensor]:
        targets = torch.zeros((count, MAX_SEGMENTS, 3), device=self.device)
        holds = torch.zeros((count, MAX_SEGMENTS), device=self.device)
        ramps = torch.zeros((count, MAX_SEGMENTS), device=self.device)
        segment_count = torch.full((count,), 2, dtype=torch.long, device=self.device)
        kind = torch.zeros(count, dtype=torch.long, device=self.device)

        # Every phase retains exact steady endpoints as one-segment sequences.
        weights = {
            PHASES[0]: (0.40, 0.30, 0.30),
            PHASES[1]: (0.35, 0.20, 0.35, 0.10),
            PHASES[2]: (0.30, 0.15, 0.20, 0.25, 0.10),
            PHASES[3]: (0.30, 0.20, 0.15, 0.15, 0.20),
            PHASES[4]: (0.30, 0.30, 0.40),
        }[phase]
        group = torch.multinomial(torch.tensor(weights, device=self.device),
                                  count, replacement=True)
        targets[:, 0] = self._random_command(count, 0.0, 0.4, 0.35)
        targets[:, 1] = targets[:, 0]
        holds[:, 0] = self._rand(count, 1.0, 2.0)
        holds[:, 1] = self._rand(count, 2.5, 4.0)
        ramps[:, 1] = self._rand(count, 0.75, 2.0)

        steady = group == 0
        if steady.any():
            segment_count[steady] = 1
            n = int(steady.sum())
            targets[steady, 0] = self._random_command(n, 0.20, 0.40, 0.35)
            holds[steady, 0] = self._rand(n, 3.0, 5.0)
            kind[steady] = 0

        if phase == PHASES[0]:
            start_stop = group == 1
            speed_change = group == 2
            if start_stop.any():
                n = int(start_stop.sum())
                moving = self._random_command(n, 0.20, 0.35, 0.30)
                start = torch.rand(n, device=self.device) < 0.5
                targets[start_stop, 0] = torch.where(start[:, None], torch.zeros_like(moving), moving)
                targets[start_stop, 1] = torch.where(start[:, None], moving, torch.zeros_like(moving))
                kind[start_stop] = torch.where(start, 1, 2)
            if speed_change.any():
                idx = torch.nonzero(speed_change, as_tuple=False).flatten()
                n = idx.numel()
                theta = self._rand(n, -math.pi, math.pi)
                low = torch.full((n,), 0.10, device=self.device)
                high = torch.full((n,), 0.30, device=self.device)
                forward = theta.abs() <= math.pi / 4
                high[forward] = 0.60
                up = torch.rand(n, device=self.device) < 0.5
                yaw = self._rand(n, -0.30, 0.30)
                targets[idx, 0] = self._command(torch.where(up, low, high), theta, yaw)
                targets[idx, 1] = self._command(torch.where(up, high, low), theta, yaw)
                kind[idx] = 3
        elif phase == PHASES[1]:
            start_speed = group == 1
            direction = group == 2
            yaw_mag = group == 3
            if start_speed.any():
                idx = torch.nonzero(start_speed, as_tuple=False).flatten()
                n = idx.numel()
                theta = self._rand(n, -math.pi, math.pi)
                yaw = self._rand(n, -0.30, 0.30)
                targets[idx, 0] = self._command(torch.full((n,), .10, device=self.device), theta, yaw)
                targets[idx, 1] = self._command(torch.full((n,), .30, device=self.device), theta, yaw)
                kind[idx] = 3
            if direction.any():
                idx = torch.nonzero(direction, as_tuple=False).flatten()
                n = idx.numel()
                theta = self._rand(n, -math.pi, math.pi)
                deltas = torch.tensor([22.5, 45.0, 67.5, 90.0], device=self.device)
                delta = torch.deg2rad(deltas[torch.randint(0, 4, (n,), device=self.device)])
                delta *= torch.where(torch.rand(n, device=self.device) < .5, -1., 1.)
                speed = self._rand(n, .20, .40)
                yaw = self._rand(n, -.30, .30)
                targets[idx, 0] = self._command(speed, theta, yaw)
                targets[idx, 1] = self._command(speed, theta + delta, yaw)
                kind[idx] = 4
            if yaw_mag.any():
                idx = torch.nonzero(yaw_mag, as_tuple=False).flatten()
                n = idx.numel()
                base = self._random_command(n, .20, .40, .30)
                sign = torch.where(base[:, 2] < 0, -1., 1.)
                base[:, 2] = sign * .10
                targets[idx, 0] = base
                targets[idx, 1] = base.clone()
                targets[idx, 1, 2] = sign * .30
                kind[idx] = 5
        elif phase == PHASES[2]:
            single = group == 1
            small = group == 2
            large = group == 3
            yaw_sign = group == 4
            if single.any():
                idx = torch.nonzero(single, as_tuple=False).flatten()
                targets[idx, 0] = torch.zeros((idx.numel(), 3), device=self.device)
                targets[idx, 1] = self._random_command(idx.numel(), .15, .35, .30)
                kind[idx] = 1
            for mask, degrees, label in ((small, (22.5, 45., 67.5, 90.), 4),
                                         (large, (112.5, 135., 157.5, 180.), 6)):
                if mask.any():
                    idx = torch.nonzero(mask, as_tuple=False).flatten()
                    n = idx.numel()
                    theta = self._rand(n, -math.pi, math.pi)
                    choices = torch.tensor(degrees, device=self.device)
                    delta = torch.deg2rad(choices[torch.randint(0, len(degrees), (n,), device=self.device)])
                    delta *= torch.where(torch.rand(n, device=self.device) < .5, -1., 1.)
                    speed = self._rand(n, .15, .35)
                    yaw = self._rand(n, -.30, .30)
                    targets[idx, 0] = self._command(speed, theta, yaw)
                    targets[idx, 1] = self._command(speed, theta + delta, yaw)
                    kind[idx] = label
            if yaw_sign.any():
                idx = torch.nonzero(yaw_sign, as_tuple=False).flatten()
                n = idx.numel()
                base = self._random_command(n, .15, .35, .30)
                mag = self._rand(n, .15, .30)
                base[:, 2] = -mag
                targets[idx, 0] = base
                targets[idx, 1] = base.clone()
                targets[idx, 1, 2] = mag
                kind[idx] = 7
        elif phase == PHASES[3]:
            single, reversal, yaw_sign, combined = (
                group == 1, group == 2, group == 3, group == 4
            )
            if single.any():
                idx = torch.nonzero(single, as_tuple=False).flatten()
                targets[idx, 1] = self._random_command(idx.numel(), 0.0, .40, .35)
                kind[idx] = 4
            if reversal.any():
                idx = torch.nonzero(reversal, as_tuple=False).flatten()
                n = idx.numel()
                theta = self._rand(n, -math.pi, math.pi)
                speed = self._rand(n, .15, .35)
                yaw = self._rand(n, -.35, .35)
                targets[idx, 0] = self._command(speed, theta, yaw)
                targets[idx, 1] = self._command(speed, theta + math.pi, yaw)
                kind[idx] = 6
            if yaw_sign.any():
                idx = torch.nonzero(yaw_sign, as_tuple=False).flatten()
                targets[idx, 1] = targets[idx, 0]
                targets[idx, 1, 2] *= -1
                kind[idx] = 7
            if combined.any():
                idx = torch.nonzero(combined, as_tuple=False).flatten()
                targets[idx, 0] = self._random_command(idx.numel(), .0, .40, .35)
                targets[idx, 1] = self._random_command(idx.numel(), .0, .40, .35)
                kind[idx] = 8
        else:
            structured = group == 1
            random_group = group == 2
            if structured.any():
                idx = torch.nonzero(structured, as_tuple=False).flatten()
                targets[idx, 1] = self._random_command(idx.numel(), .0, .40, .35)
                kind[idx] = 8
            if random_group.any():
                idx = torch.nonzero(random_group, as_tuple=False).flatten()
                n = idx.numel()
                counts = torch.randint(5, MAX_SEGMENTS + 1, (n,), device=self.device)
                segment_count[idx] = counts
                kind[idx] = 9
                for local, env_index in enumerate(idx.tolist()):
                    c = int(counts[local])
                    values = self._random_command(c, .0, .40, .35)
                    # Pre-register stop, reversal and yaw reversal in every random sequence.
                    values[1] = 0.0
                    if c > 2:
                        values[2, :2] = -values[0, :2]
                    if c > 3:
                        values[3, 2] = -values[2, 2] if abs(float(values[2, 2])) > .05 else .3
                    targets[env_index, :c] = values
                    holds[env_index, :c] = self._rand(c, 3.0, 5.0)
                    ramps[env_index, 1:c] = self._rand(c - 1, .75, 2.0)
        return {
            "targets": targets,
            "holds": holds,
            "ramps": ramps,
            "segment_count": segment_count,
            "transition_type": kind,
        }

    @staticmethod
    def _mirror_descriptor(descriptor: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result = {key: value.clone() for key, value in descriptor.items()}
        result["targets"][..., 1:3].neg_()
        return result

    def _slice_descriptor(self, descriptor: dict[str, torch.Tensor], index: int) -> dict[str, Any]:
        return {key: value[index].detach().clone() for key, value in descriptor.items()}

    def _assign_descriptor(
        self, ids: torch.Tensor, descriptor: dict[str, torch.Tensor],
        sequence_ids: torch.Tensor, pair_ids: torch.Tensor
    ) -> None:
        if ids.numel() != torch.unique(ids).numel():
            self.duplicate_assignment_count += 1
            raise RuntimeError("W2 duplicate environment assignment")
        self.sequence_targets[ids] = descriptor["targets"]
        self.sequence_hold_s[ids] = descriptor["holds"]
        self.sequence_ramp_s[ids] = descriptor["ramps"]
        self.sequence_segment_count[ids] = descriptor["segment_count"]
        self.transition_type[ids] = descriptor["transition_type"]
        self.sequence_segment_index[ids] = 0
        self.sequence_elapsed_s[ids] = 0.0
        self.sequence_id[ids] = sequence_ids
        self.sampled_pair_id[ids] = pair_ids
        self.transition_id[ids] = torch.arange(
            self.next_transition_id, self.next_transition_id + ids.numel(),
            device=self.device,
        )
        self.next_transition_id += int(ids.numel())
        self.is_standing_env[ids] = False
        self.is_heading_env[ids] = False
        self.physical_command_b[ids] = descriptor["targets"][:, 0]
        self._apply_calibration(ids)

    def _descriptor_batch(self, item: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device).unsqueeze(0) for key, value in item.items()
                if key in {"targets", "holds", "ramps", "segment_count", "transition_type"}}

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return
        if ids.numel() != torch.unique(ids).numel():
            raise RuntimeError("W2 reset IDs contain duplicates")
        self.reset_event_counter += 1
        if ids.numel() % 2:
            self.odd_reset_event_count += 1
        else:
            self.even_reset_event_count += 1
        cursor = 0
        if self._pending_sequence is not None:
            pending = self._pending_sequence
            age = self.reset_event_counter - int(pending["source_reset_event_id"])
            self.pending_sequence_maximum_age = max(self.pending_sequence_maximum_age, age)
            if age > 1:
                raise RuntimeError("W2 pending mirrored sequence exceeded one reset event")
            desc = self._descriptor_batch(pending)
            self._assign_descriptor(
                ids[:1], desc,
                torch.tensor([pending["sequence_id"]], device=self.device),
                torch.tensor([pending["pair_id"]], device=self.device),
            )
            self.sequence_mirror_count += 1
            self._pending_sequence = None
            cursor = 1
            if self._phase_transition_pending:
                self._active_phase = self._requested_phase
                self._phase_transition_pending = False
        remaining = ids[cursor:]
        if remaining.numel() == 0:
            return
        base_count = (int(remaining.numel()) + 1) // 2
        pair_count = int(remaining.numel()) // 2
        base = self._sample_descriptor(base_count, self._active_phase)
        mirrored = self._mirror_descriptor(base)
        pair_ids = torch.arange(self.next_pair_id, self.next_pair_id + base_count,
                                device=self.device)
        sequence_ids = torch.arange(self.next_sequence_id,
                                    self.next_sequence_id + base_count,
                                    device=self.device)
        self._assign_descriptor(remaining[:base_count], base, sequence_ids, pair_ids)
        self.sequence_base_count += base_count
        if pair_count:
            mirror_desc = {key: value[:pair_count] for key, value in mirrored.items()}
            self._assign_descriptor(
                remaining[base_count:], mirror_desc,
                sequence_ids[:pair_count], pair_ids[:pair_count],
            )
            self.sequence_mirror_count += pair_count
        if base_count > pair_count:
            item = self._slice_descriptor(mirrored, base_count - 1)
            self._pending_sequence = {
                **item,
                "pair_id": int(pair_ids[-1]),
                "sequence_id": int(sequence_ids[-1]),
                "source_reset_event_id": self.reset_event_counter,
                "source_iteration": self.training_iteration,
                "source_phase": self._active_phase,
                "rng_provenance": self.rng_hash(),
            }
        self.next_pair_id += base_count
        self.next_sequence_id += base_count

    def _apply_calibration(self, ids: torch.Tensor | None = None) -> None:
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        self.actor_command_b[ids, :2] = self.physical_command_b[ids, :2]
        self.actor_command_b[ids, 2] = calibrate_yaw(self.physical_command_b[ids, 2])

    def _update_command(self) -> None:
        if self.external_override_enabled:
            self.physical_command_b.copy_(self.external_physical_override)
            self._apply_calibration()
            return
        dt = float(self._env.step_dt)
        self.sequence_elapsed_s += dt
        finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for segment in range(MAX_SEGMENTS):
            active = (self.sequence_segment_index == segment)
            if not active.any():
                continue
            if segment == 0:
                self.physical_command_b[active] = self.sequence_targets[active, 0]
                ready = active & (self.sequence_elapsed_s >= self.sequence_hold_s[:, 0])
            else:
                ramp = self.sequence_ramp_s[:, segment].clamp_min(1e-6)
                progress = self.sequence_elapsed_s / ramp
                blend = minimum_jerk(progress)[:, None]
                previous = self.sequence_targets[:, segment - 1]
                target = self.sequence_targets[:, segment]
                self.physical_command_b[active] = (
                    previous[active] + blend[active] * (target[active] - previous[active])
                )
                ramp_done = active & (progress >= 1.0)
                self.physical_command_b[ramp_done] = target[ramp_done]
                ready = ramp_done & (
                    self.sequence_elapsed_s >=
                    self.sequence_ramp_s[:, segment] + self.sequence_hold_s[:, segment]
                )
            can_advance = ready & (
                self.sequence_segment_index + 1 < self.sequence_segment_count
            )
            finished |= ready & ~can_advance
            self.sequence_segment_index[can_advance] += 1
            self.sequence_elapsed_s[can_advance] = 0.0
        self._apply_calibration()
        completed_ids = torch.nonzero(finished, as_tuple=False).flatten()
        if completed_ids.numel():
            self._resample_command(completed_ids)

    def load_legacy_parent_state_dict(self, state: dict[str, Any]) -> None:
        required = {
            "pending_queue", "sampler_rng_state", "command_rng_state",
            "next_pair_id", "reset_event_counter", "current_command_buffer",
            "active_curriculum_phase", "requested_curriculum_phase",
        }
        missing = sorted(required - set(state))
        if missing:
            raise RuntimeError("EXP013_W2_STRICT_RESUME_FAIL missing " + ",".join(missing))
        if state["pending_queue"] is not None:
            raise RuntimeError("EXP013_W2_STRICT_RESUME_FAIL parent pending queue nonempty")
        sampler_rng = state["sampler_rng_state"].cpu()
        if not torch.equal(sampler_rng, state["command_rng_state"].cpu()):
            raise RuntimeError("EXP013_W2_STRICT_RESUME_FAIL parent RNG mismatch")
        if torch.device(self.device).type == "cuda":
            torch.cuda.set_rng_state(sampler_rng, self.device)
        else:
            torch.set_rng_state(sampler_rng)
        self.next_pair_id = int(state["next_pair_id"])
        self.reset_event_counter = int(state["reset_event_counter"])
        self.next_sequence_id = self.next_pair_id
        self.actor_command_b.copy_(state["current_command_buffer"].to(self.device)[:, :3])
        self.physical_command_b.copy_(self.actor_command_b)
        self._legacy_parent_restored = True
        self._active_phase = PHASES[0]
        self._requested_phase = PHASES[0]

    def sampler_state_dict(self) -> dict[str, Any]:
        device_type = torch.device(self.device).type
        rng = (torch.cuda.get_rng_state(self.device).cpu().clone()
               if device_type == "cuda" else torch.get_rng_state().clone())
        pending = None
        if self._pending_sequence is not None:
            pending = {
                key: value.detach().cpu().clone() if torch.is_tensor(value) else value
                for key, value in self._pending_sequence.items()
            }
        return {
            "state_version": self.state_version,
            "pending_mirrored_sequence": pending,
            "sampler_rng_state": rng,
            "command_rng_state": rng.clone(),
            "rng_backend": device_type,
            "next_pair_id": self.next_pair_id,
            "next_sequence_id": self.next_sequence_id,
            "next_transition_id": self.next_transition_id,
            "reset_event_counter": self.reset_event_counter,
            "active_curriculum_phase": self._active_phase,
            "requested_curriculum_phase": self._requested_phase,
            "phase_transition_pending": self._phase_transition_pending,
            "training_iteration": self.training_iteration,
            "physical_command_buffer": self.physical_command_b.detach().cpu().clone(),
            "actor_command_buffer": self.actor_command_b.detach().cpu().clone(),
            "sequence_targets": self.sequence_targets.detach().cpu().clone(),
            "sequence_hold_s": self.sequence_hold_s.detach().cpu().clone(),
            "sequence_ramp_s": self.sequence_ramp_s.detach().cpu().clone(),
            "sequence_segment_count": self.sequence_segment_count.detach().cpu().clone(),
            "sequence_segment_index": self.sequence_segment_index.detach().cpu().clone(),
            "sequence_elapsed_s": self.sequence_elapsed_s.detach().cpu().clone(),
            "sequence_id": self.sequence_id.detach().cpu().clone(),
            "transition_id": self.transition_id.detach().cpu().clone(),
            "transition_type": self.transition_type.detach().cpu().clone(),
            "sampled_pair_id": self.sampled_pair_id.detach().cpu().clone(),
            "sequence_base_count": self.sequence_base_count,
            "sequence_mirror_count": self.sequence_mirror_count,
            "pending_sequence_maximum_age": self.pending_sequence_maximum_age,
            "sequence_serialization_round_trip_count":
                self.sequence_serialization_round_trip_count,
            "legacy_parent_restored": self._legacy_parent_restored,
            "curriculum_counters": {
                "base": self.sequence_base_count,
                "mirror": self.sequence_mirror_count,
            },
        }

    def load_sampler_state_dict(self, state: dict[str, Any]) -> None:
        required = {
            "state_version", "pending_mirrored_sequence", "sampler_rng_state",
            "command_rng_state", "next_pair_id", "next_sequence_id",
            "next_transition_id", "reset_event_counter",
            "active_curriculum_phase", "requested_curriculum_phase",
            "phase_transition_pending", "physical_command_buffer",
            "actor_command_buffer", "sequence_targets", "sequence_hold_s",
            "sequence_ramp_s", "sequence_segment_count", "sequence_segment_index",
            "sequence_elapsed_s", "sequence_id", "transition_id",
            "transition_type", "curriculum_counters",
        }
        missing = sorted(required - set(state))
        if missing or int(state.get("state_version", -1)) != self.state_version:
            raise RuntimeError("EXP013_W2_SEQUENCE_SAMPLER_FAIL serialization")
        pending = state["pending_mirrored_sequence"]
        self._pending_sequence = None if pending is None else {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in pending.items()
        }
        if self.pending_queue_length > 1:
            raise RuntimeError("EXP013_W2_SEQUENCE_SAMPLER_FAIL queue")
        for name in ("next_pair_id", "next_sequence_id", "next_transition_id",
                     "reset_event_counter", "training_iteration",
                     "sequence_base_count", "sequence_mirror_count",
                     "pending_sequence_maximum_age"):
            setattr(self, name, int(state.get(name, 0)))
        self._active_phase = state["active_curriculum_phase"]
        self._requested_phase = state["requested_curriculum_phase"]
        self._phase_transition_pending = bool(state["phase_transition_pending"])
        mapping = {
            "physical_command_b": "physical_command_buffer",
            "actor_command_b": "actor_command_buffer",
            "sequence_targets": "sequence_targets",
            "sequence_hold_s": "sequence_hold_s",
            "sequence_ramp_s": "sequence_ramp_s",
            "sequence_segment_count": "sequence_segment_count",
            "sequence_segment_index": "sequence_segment_index",
            "sequence_elapsed_s": "sequence_elapsed_s",
            "sequence_id": "sequence_id",
            "transition_id": "transition_id",
            "transition_type": "transition_type",
        }
        for destination, source in mapping.items():
            getattr(self, destination).copy_(state[source].to(self.device))
        if "sampled_pair_id" in state:
            self.sampled_pair_id.copy_(state["sampled_pair_id"].to(self.device))
        rng = state["sampler_rng_state"].cpu()
        if not torch.equal(rng, state["command_rng_state"].cpu()):
            raise RuntimeError("EXP013_W2_SEQUENCE_SAMPLER_FAIL RNG")
        if torch.device(self.device).type == "cuda":
            torch.cuda.set_rng_state(rng, self.device)
        else:
            torch.set_rng_state(rng)
        self.sequence_serialization_round_trip_count += 1

    def rng_hash(self) -> str:
        if torch.device(self.device).type == "cuda":
            return _tensor_sha(torch.cuda.get_rng_state(self.device))
        return _tensor_sha(torch.get_rng_state())

    def runtime_summary(self) -> dict[str, Any]:
        pending_age = 0 if self._pending_sequence is None else (
            self.reset_event_counter -
            int(self._pending_sequence["source_reset_event_id"])
        )
        return {
            "reset_event_count": self.reset_event_counter,
            "odd_reset_event_count": self.odd_reset_event_count,
            "even_reset_event_count": self.even_reset_event_count,
            "pending_queue_length": self.pending_queue_length,
            "pending_queue_age": pending_age,
            "pending_queue_maximum_age": self.pending_sequence_maximum_age,
            "base_sequence_count": self.sequence_base_count,
            "mirror_sequence_count": self.sequence_mirror_count,
            "mirror_residual": self.mirror_residual,
            "missing_assignment_count": self.missing_assignment_count,
            "duplicate_assignment_count": self.duplicate_assignment_count,
            "forced_reset_count": self.forced_reset_count,
            "active_phase": self._active_phase,
            "requested_phase": self._requested_phase,
            "phase_transition_pending": self._phase_transition_pending,
            "serialization_round_trip_count":
                self.sequence_serialization_round_trip_count,
            "rng_hash": self.rng_hash(),
        }
