"""Deterministic pending-mirror command sampler for Phase W1B-R2."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

import torch

from g1_omnidirectional.w1b_command import W1BYawConditionedCommand


PHASES = (
    "Y1_FORWARD_MOVING_TURNS",
    "Y2_ALL_DIRECTION_MOVING_TURNS",
    "Y3_TURN_IN_PLACE_ACQUISITION",
    "Y4_BALANCED_CONSOLIDATION",
)


def _tensor_sha(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


class W1BR2PendingMirrorCommand(W1BYawConditionedCommand):
    """Exact mirror pairs over a window of at most two positive reset events."""

    state_version = 1

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._active_phase = PHASES[0]
        self._requested_phase = PHASES[0]
        self._phase_transition_pending = False
        self._pending: dict[str, Any] | None = None
        self.next_pair_id = 0
        self.reset_event_counter = 0
        self.odd_reset_event_count = 0
        self.even_reset_event_count = 0
        self.base_command_count = 0
        self.mirror_command_count = 0
        self.pending_queue_maximum_age = 0
        self.phase_transitions_with_pending_queue = 0
        self.serialization_round_trip_count = 0
        self.missing_assignment_count = 0
        self.duplicate_assignment_count = 0
        self.forced_reset_count = 0
        self.sampled_group = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.sampled_pair_id = torch.full_like(self.sampled_group, -1)
        self._iteration_trace: list[dict[str, Any]] = []
        self._last_trace_iteration = 0

    @staticmethod
    def phase_for_iteration(iteration: int) -> str:
        if iteration <= 40:
            return PHASES[0]
        if iteration <= 100:
            return PHASES[1]
        if iteration <= 150:
            return PHASES[2]
        return PHASES[3]

    @property
    def phase(self) -> str:
        return getattr(self, "_active_phase", self.phase_for_iteration(
            getattr(self, "training_iteration", 0)
        ))

    @property
    def requested_phase(self) -> str:
        return self._requested_phase

    @property
    def pending_queue_length(self) -> int:
        return 0 if self._pending is None else 1

    @property
    def mirror_residual(self) -> int:
        return self.base_command_count - self.mirror_command_count

    def set_training_iteration(self, iteration: int) -> None:
        if (
            getattr(self, "_last_trace_iteration", 0) > 0
            and int(iteration) != self._last_trace_iteration
        ):
            self._iteration_trace.append({
                "iteration": self._last_trace_iteration,
                **self.runtime_summary(),
            })
        self.training_iteration = int(iteration)
        self._last_trace_iteration = int(iteration)
        requested = self.phase_for_iteration(iteration)
        self._requested_phase = requested
        if requested == self._active_phase:
            return
        if self._pending is None:
            self._active_phase = requested
            self._phase_transition_pending = False
        else:
            self._phase_transition_pending = True
            self.phase_transitions_with_pending_queue += 1

    def _weights(self, phase: str) -> tuple[float, float, float]:
        return {
            PHASES[0]: (.45, .45, .10),
            PHASES[1]: (.40, .50, .10),
            PHASES[2]: (.35, .40, .25),
            PHASES[3]: (.35, .45, .20),
        }[phase]

    def _sample_base(
        self, count: int, phase: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Copy the protected sampler's RNG call order exactly."""
        group = torch.multinomial(
            torch.tensor(self._weights(phase), device=self.device),
            count,
            replacement=True,
        )
        theta = torch.zeros(count, device=self.device)
        speed = torch.zeros(count, device=self.device)
        yaw = torch.zeros(count, device=self.device)
        zero, moving, pure = group == 0, group == 1, group == 2
        if zero.any():
            n = int(zero.sum())
            theta[zero] = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
            speed[zero] = torch.empty(n, device=self.device).uniform_(.25, .35)
        if moving.any():
            n = int(moving.sum())
            if phase == PHASES[0]:
                theta[moving] = torch.empty(n, device=self.device).uniform_(
                    -math.pi / 4, math.pi / 4
                )
                speed[moving] = torch.empty(n, device=self.device).uniform_(.25, .60)
                yaw[moving] = self._away_from_zero(n, .05, .30, self.device)
            elif phase == PHASES[1]:
                theta[moving] = torch.empty(n, device=self.device).uniform_(
                    -math.pi, math.pi
                )
                speed[moving] = torch.empty(n, device=self.device).uniform_(.20, .40)
                yaw[moving] = self._away_from_zero(n, .05, .35, self.device)
            elif phase == PHASES[2]:
                theta[moving] = torch.empty(n, device=self.device).uniform_(
                    -math.pi, math.pi
                )
                speed[moving] = torch.empty(n, device=self.device).uniform_(.20, .50)
                yaw[moving] = self._away_from_zero(n, .05, .40, self.device)
            else:
                theta[moving] = torch.empty(n, device=self.device).uniform_(
                    -math.pi, math.pi
                )
                max_speed = torch.where(
                    theta[moving].abs() <= math.pi / 4, .8, .6
                )
                speed[moving] = .20 + torch.rand(
                    n, device=self.device
                ) * (max_speed - .20)
                yaw[moving] = self._away_from_zero(n, .05, .50, self.device)
        if pure.any():
            n = int(pure.sum())
            if phase == PHASES[0]:
                speed[pure] = torch.empty(n, device=self.device).uniform_(0, .10)
                theta[pure] = torch.empty(n, device=self.device).uniform_(
                    -math.pi, math.pi
                )
                yaw[pure] = self._away_from_zero(n, .15, .25, self.device)
            elif phase == PHASES[1]:
                speed[pure] = torch.empty(n, device=self.device).uniform_(0, .08)
                theta[pure] = torch.empty(n, device=self.device).uniform_(
                    -math.pi, math.pi
                )
                yaw[pure] = self._away_from_zero(n, .05, .30, self.device)
            elif phase == PHASES[2]:
                yaw[pure] = self._away_from_zero(n, .15, .45, self.device)
            else:
                yaw[pure] = self._away_from_zero(n, .15, .50, self.device)
        return group, theta, speed, yaw

    def _assign(
        self,
        ids: torch.Tensor,
        commands: torch.Tensor,
        theta: torch.Tensor,
        speed: torch.Tensor,
        group: torch.Tensor,
        pair_ids: torch.Tensor,
    ) -> None:
        if ids.numel() != torch.unique(ids).numel():
            self.duplicate_assignment_count += int(ids.numel() - torch.unique(ids).numel())
            raise RuntimeError("W1B-R2 duplicate reset environment assignment")
        self.vel_command_b[ids, :3] = commands[:, :3]
        self.sampled_theta[ids] = theta
        self.sampled_speed[ids] = speed
        self.sampled_group[ids] = group
        self.sampled_pair_id[ids] = pair_ids
        self.is_standing_env[ids] = False
        self.is_heading_env[ids] = False

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return
        if self.pending_queue_length > 1:
            raise RuntimeError("W1B-R2 pending queue length exceeded one")
        if ids.numel() != torch.unique(ids).numel():
            self.duplicate_assignment_count += int(ids.numel() - torch.unique(ids).numel())
            raise RuntimeError("W1B-R2 reset_env_ids contain duplicates")

        self.reset_event_counter += 1
        original_count = int(ids.numel())
        if original_count % 2:
            self.odd_reset_event_count += 1
        else:
            self.even_reset_event_count += 1

        cursor = 0
        if self._pending is not None:
            pending = self._pending
            age = self.reset_event_counter - int(pending["source_reset_event_id"])
            self.pending_queue_maximum_age = max(self.pending_queue_maximum_age, age)
            if age > 1:
                raise RuntimeError("W1B-R2 pending mirror exceeded one positive reset event")
            command = pending["command"].to(self.device).reshape(1, 4)
            self._assign(
                ids[:1],
                command,
                pending["theta"].to(self.device).reshape(1),
                pending["speed"].to(self.device).reshape(1),
                torch.tensor([pending["curriculum_group"]], device=self.device),
                torch.tensor([pending["pair_id"]], device=self.device),
            )
            self.mirror_command_count += 1
            self._pending = None
            cursor = 1
            if self._phase_transition_pending:
                self._active_phase = self._requested_phase
                self._phase_transition_pending = False

        remaining = ids[cursor:]
        if remaining.numel() == 0:
            return
        phase = self._active_phase

        if remaining.numel() % 2 == 0:
            half = int(remaining.numel() // 2)
            # _sample_base is a literal extraction of the protected legacy RNG
            # sequence. The parity suite compares it event-by-event.
            group, theta, speed, yaw = self._sample_base(half, phase)
            pair_ids = torch.arange(
                self.next_pair_id,
                self.next_pair_id + half,
                dtype=torch.long,
                device=self.device,
            )
            base = torch.stack(
                (
                    speed * torch.cos(theta),
                    speed * torch.sin(theta),
                    yaw,
                    torch.zeros_like(yaw),
                ),
                dim=-1,
            )
            mirror = base.clone()
            mirror[:, 1:3].neg_()
            self._assign(
                remaining[:half], base, theta, speed, group, pair_ids
            )
            self._assign(
                remaining[half:], mirror, -theta, speed, group, pair_ids
            )
            self.next_pair_id += half
            self.base_command_count += half
            self.mirror_command_count += half
            return

        pair_count = int(remaining.numel() // 2)
        base_count = pair_count + 1
        rng_provenance = self.rng_hash()
        group, theta, speed, yaw = self._sample_base(base_count, phase)
        base = torch.stack(
            (speed * torch.cos(theta), speed * torch.sin(theta), yaw, torch.zeros_like(yaw)),
            dim=-1,
        )
        pair_ids = torch.arange(
            self.next_pair_id,
            self.next_pair_id + base_count,
            dtype=torch.long,
            device=self.device,
        )
        base_ids = remaining[:base_count]
        self._assign(base_ids, base, theta, speed, group, pair_ids)
        self.base_command_count += base_count
        if pair_count:
            mirror_ids = remaining[base_count:]
            mirror = base[:pair_count].clone()
            mirror[:, 1:3].neg_()
            self._assign(
                mirror_ids,
                mirror,
                -theta[:pair_count],
                speed[:pair_count],
                group[:pair_count],
                pair_ids[:pair_count],
            )
            self.mirror_command_count += pair_count
        last = base_count - 1
        pending_command = base[last].clone()
        pending_command[1:3].neg_()
        self._pending = {
            "command": pending_command,
            "theta": (-theta[last]).clone(),
            "speed": speed[last].clone(),
            "curriculum_group": int(group[last]),
            "pair_id": int(pair_ids[last]),
            "source_reset_event_id": self.reset_event_counter,
            "source_iteration": self.training_iteration,
            "source_phase": phase,
            "rng_provenance": rng_provenance,
        }
        self.next_pair_id += base_count

    def rng_hash(self) -> str:
        device_type = torch.device(self.device).type
        if device_type == "cuda":
            return _tensor_sha(torch.cuda.get_rng_state(self.device))
        return _tensor_sha(torch.get_rng_state())

    def sampler_state_dict(self) -> dict[str, Any]:
        device_type = torch.device(self.device).type
        pending = None
        if self._pending is not None:
            pending = {
                key: value.detach().cpu().clone() if torch.is_tensor(value) else value
                for key, value in self._pending.items()
            }
        cuda_rng = (
            torch.cuda.get_rng_state(self.device).cpu().clone()
            if device_type == "cuda"
            else None
        )
        cpu_rng = torch.get_rng_state().clone()
        return {
            "state_version": self.state_version,
            "pending_queue": pending,
            "sampler_rng_state": cuda_rng if cuda_rng is not None else cpu_rng,
            "command_rng_state": cuda_rng.clone() if cuda_rng is not None else cpu_rng.clone(),
            "rng_backend": device_type,
            "next_pair_id": self.next_pair_id,
            "reset_event_counter": self.reset_event_counter,
            "odd_reset_event_count": self.odd_reset_event_count,
            "even_reset_event_count": self.even_reset_event_count,
            "base_command_count": self.base_command_count,
            "mirror_command_count": self.mirror_command_count,
            "pending_queue_maximum_age": self.pending_queue_maximum_age,
            "active_curriculum_phase": self._active_phase,
            "requested_curriculum_phase": self._requested_phase,
            "phase_transition_pending": self._phase_transition_pending,
            "phase_transitions_with_pending_queue": self.phase_transitions_with_pending_queue,
            "training_iteration": self.training_iteration,
            "current_command_buffer": self.vel_command_b.detach().cpu().clone(),
            "sampled_theta": self.sampled_theta.detach().cpu().clone(),
            "sampled_speed": self.sampled_speed.detach().cpu().clone(),
            "sampled_group": self.sampled_group.detach().cpu().clone(),
            "sampled_pair_id": self.sampled_pair_id.detach().cpu().clone(),
            "curriculum_counters": {
                "base": self.base_command_count,
                "mirror": self.mirror_command_count,
            },
            "iteration_trace": self._iteration_trace,
            "last_trace_iteration": self._last_trace_iteration,
        }

    def load_sampler_state_dict(self, state: dict[str, Any]) -> None:
        required = {
            "state_version", "pending_queue", "sampler_rng_state",
            "command_rng_state", "next_pair_id", "reset_event_counter",
            "active_curriculum_phase", "requested_curriculum_phase",
            "phase_transition_pending", "current_command_buffer",
            "curriculum_counters",
        }
        missing = sorted(required - set(state))
        if missing:
            raise RuntimeError(
                "EXP013_W1B_R2_SAMPLER_SERIALIZATION_FAIL missing " + ",".join(missing)
            )
        if int(state["state_version"]) != self.state_version:
            raise RuntimeError("EXP013_W1B_R2_SAMPLER_SERIALIZATION_FAIL version")
        pending = state["pending_queue"]
        if pending is not None:
            pending = {
                key: value.to(self.device) if torch.is_tensor(value) else value
                for key, value in pending.items()
            }
        self._pending = pending
        if self.pending_queue_length > 1:
            raise RuntimeError("EXP013_W1B_R2_SAMPLER_SERIALIZATION_FAIL queue length")
        for name in (
            "next_pair_id", "reset_event_counter", "odd_reset_event_count",
            "even_reset_event_count", "base_command_count", "mirror_command_count",
            "pending_queue_maximum_age", "phase_transitions_with_pending_queue",
            "training_iteration",
        ):
            setattr(self, name, int(state.get(name, 0)))
        self._active_phase = state["active_curriculum_phase"]
        self._requested_phase = state["requested_curriculum_phase"]
        self._phase_transition_pending = bool(state["phase_transition_pending"])
        self._iteration_trace = list(state.get("iteration_trace", []))
        self._last_trace_iteration = int(state.get(
            "last_trace_iteration", self.training_iteration
        ))
        self.vel_command_b.copy_(state["current_command_buffer"].to(self.device))
        for name in ("sampled_theta", "sampled_speed", "sampled_group", "sampled_pair_id"):
            if name in state:
                getattr(self, name).copy_(state[name].to(self.device))
        sampler_rng = state["sampler_rng_state"].cpu()
        command_rng = state["command_rng_state"].cpu()
        if not torch.equal(sampler_rng, command_rng):
            raise RuntimeError("EXP013_W1B_R2_SAMPLER_SERIALIZATION_FAIL RNG aliases differ")
        if torch.device(self.device).type == "cuda":
            torch.cuda.set_rng_state(sampler_rng, self.device)
        else:
            torch.set_rng_state(sampler_rng)
        self.serialization_round_trip_count += 1

    def runtime_summary(self) -> dict[str, Any]:
        pending_age = (
            0 if self._pending is None
            else self.reset_event_counter - int(self._pending["source_reset_event_id"])
        )
        return {
            "reset_event_count": self.reset_event_counter,
            "odd_reset_event_count": self.odd_reset_event_count,
            "even_reset_event_count": self.even_reset_event_count,
            "pending_queue_length": self.pending_queue_length,
            "pending_queue_age": pending_age,
            "pending_queue_maximum_age": self.pending_queue_maximum_age,
            "base_command_count": self.base_command_count,
            "mirror_command_count": self.mirror_command_count,
            "unpaired_count": self.mirror_residual,
            "mirror_residual": self.mirror_residual,
            "phase_transitions_with_pending_queue": self.phase_transitions_with_pending_queue,
            "serialization_round_trip_count": self.serialization_round_trip_count,
            "missing_assignment_count": self.missing_assignment_count,
            "duplicate_assignment_count": self.duplicate_assignment_count,
            "forced_reset_count": self.forced_reset_count,
            "active_phase": self._active_phase,
            "requested_phase": self._requested_phase,
            "phase_transition_pending": self._phase_transition_pending,
            "rng_hash": self.rng_hash(),
        }

    def finalized_iteration_trace(self) -> list[dict[str, Any]]:
        rows = list(self._iteration_trace)
        if self._last_trace_iteration > 0 and (
            not rows or rows[-1]["iteration"] != self._last_trace_iteration
        ):
            rows.append({
                "iteration": self._last_trace_iteration,
                **self.runtime_summary(),
            })
        return rows
