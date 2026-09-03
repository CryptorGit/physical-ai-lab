"""Frozen Stage-4 locomotion actor with gated, skill-specific residual heads."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, HiddenState
from rsl_rl.utils import unpad_trajectories

from g1_command_skills.scripted_crouch import pose_for_depth


LEGACY_OBSERVATION_DIM = 123
COMMAND_OBSERVATION_DIM = 29
SKILL_COUNT = 6
STOP_SKILL_ID = 1
CROUCH_SKILL_ID = 3
STEP_OVER_SKILL_ID = 4
# Action order is the task's preserved 37-joint G1 order.  With the environment
# action scale of 0.5 these normalized residual limits correspond to at most
# 0.125 rad at hip/knee and 0.075 rad at ankle pitch per policy output.
DEFAULT_CROUCH_ACTION_INDICES = (0, 1, 11, 12, 15, 16)
DEFAULT_CROUCH_ACTION_SCALES = (0.25, 0.25, 0.25, 0.25, 0.15, 0.15)
# Action order is the G1 minimal articulation order used by JointPositionActionCfg(".*").
# torso, hip yaw/roll and ankle roll receive +/-0.03; hip pitch receives +/-0.015.
DEFAULT_STOP_CORRECTION_INDICES = (2, 7, 8, 3, 4, 19, 20, 0, 1)
DEFAULT_STOP_CORRECTION_SCALES = (0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.015, 0.015)

_LATEST_STOP_CORRECTION: dict[str, torch.Tensor] = {}


def latest_stop_correction() -> dict[str, torch.Tensor]:
    """Return detached diagnostics from the most recent rollout actor call."""
    return _LATEST_STOP_CORRECTION


def _zero_last_linear(module: nn.Module) -> None:
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            return
    raise RuntimeError("Residual head has no linear output layer")


class G1CommandResidualActor(MLPModel):
    """RSL-RL actor whose Stage-4 base is frozen and whose residuals are skill-local.

    The command layout is owned by ``command_mdp.py``. Columns 0:6 and 6:12 of
    the appended command block are the current and previous skill one-hots;
    column 25 is the 0.4 s transition progress. The deterministic mean is:

    ``base(state) + sum(gate_i * scale * tanh(residual_i(state, command)))``.
    """

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 128, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        legacy_observation_dim: int = LEGACY_OBSERVATION_DIM,
        command_observation_dim: int = COMMAND_OBSERVATION_DIM,
        command_embedding_dim: int = 32,
        state_embedding_dim: int = 64,
        residual_hidden_dims: tuple[int, ...] | list[int] = (64, 64),
        residual_scale: float = 0.25,
        trainable_skill_ids: tuple[int, ...] | list[int] = (0,),
        corrective_command_embedding_dim: int = 16,
        corrective_state_embedding_dim: int = 32,
        corrective_hidden_dims: tuple[int, ...] | list[int] = (32, 32),
        stop_correction_action_indices: tuple[int, ...] | list[int] = DEFAULT_STOP_CORRECTION_INDICES,
        stop_correction_action_scales: tuple[float, ...] | list[float] = DEFAULT_STOP_CORRECTION_SCALES,
        crouch_action_indices: tuple[int, ...] | list[int] = DEFAULT_CROUCH_ACTION_INDICES,
        crouch_action_scales: tuple[float, ...] | list[float] = DEFAULT_CROUCH_ACTION_SCALES,
        train_stop_correction: bool = False,
        crouch_controller: str = "scripted_shallow_v1",
        learned_crouch_residual_enabled: bool = False,
        step_over_controller: str = "scripted_step_over_v0_guarded",
        learned_step_over_residual_enabled: bool = False,
        freeze_base: bool = True,
    ) -> None:
        # Let MLPModel create the distribution and observation plumbing, then
        # replace its monolithic MLP by the experiment-local structured actor.
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )
        del self.mlp
        self.legacy_observation_dim = int(legacy_observation_dim)
        self.command_observation_dim = int(command_observation_dim)
        self.skill_count = SKILL_COUNT
        if self.obs_dim != self.legacy_observation_dim + self.command_observation_dim:
            raise ValueError(
                f"Expected {self.legacy_observation_dim + self.command_observation_dim} actor observations, "
                f"got {self.obs_dim}"
            )
        mean_dim = self.distribution.input_dim if self.distribution is not None else output_dim
        self.base_mlp = MLP(self.legacy_observation_dim, mean_dim, hidden_dims, activation)
        # A second frozen 123-D actor is selected only for CROUCH.  Old
        # checkpoints populate it from base_mlp during load; the CROUCH stage
        # rebase then replaces it with an explicitly evaluated standing actor.
        self.stand_base_mlp = MLP(self.legacy_observation_dim, mean_dim, hidden_dims, activation)
        # Every skill owns its complete learnable residual path.  Keeping only
        # the final head skill-local is insufficient: updating a shared encoder
        # changes frozen heads and caused the observed RUN forgetting in TURN.
        self.skill_command_encoders = nn.ModuleList(
            MLP(self.command_observation_dim, command_embedding_dim, (64,), activation, last_activation=activation)
            for _ in range(self.skill_count)
        )
        self.skill_state_adapters = nn.ModuleList(
            MLP(self.legacy_observation_dim, state_embedding_dim, (128,), activation, last_activation=activation)
            for _ in range(self.skill_count)
        )
        residual_input_dim = command_embedding_dim + state_embedding_dim
        self.residual_heads = nn.ModuleList(
            MLP(residual_input_dim, mean_dim, residual_hidden_dims, activation) for _ in range(self.skill_count)
        )
        for head in self.residual_heads:
            _zero_last_linear(head)
        self.stop_corrective_command_encoder = MLP(
            self.command_observation_dim,
            corrective_command_embedding_dim,
            (32,),
            activation,
            last_activation=activation,
        )
        self.stop_corrective_state_adapter = MLP(
            self.legacy_observation_dim,
            corrective_state_embedding_dim,
            (64,),
            activation,
            last_activation=activation,
        )
        self.stop_corrective_head = MLP(
            corrective_command_embedding_dim + corrective_state_embedding_dim,
            mean_dim,
            corrective_hidden_dims,
            activation,
        )
        _zero_last_linear(self.stop_corrective_head)
        indices = tuple(int(index) for index in stop_correction_action_indices)
        scales = tuple(float(scale) for scale in stop_correction_action_scales)
        if len(indices) != len(scales) or len(set(indices)) != len(indices):
            raise ValueError("STOP correction indices/scales must be unique paired values")
        if any(index < 0 or index >= mean_dim for index in indices):
            raise ValueError(f"Invalid STOP correction action indices: {indices}")
        if any(scale <= 0.0 or scale > 0.05 for scale in scales):
            raise ValueError(f"STOP correction scales must be in (0, 0.05]: {scales}")
        correction_scale = torch.zeros(mean_dim)
        correction_scale[list(indices)] = torch.tensor(scales)
        self.register_buffer("stop_correction_scale", correction_scale)
        self.stop_correction_action_indices = indices
        crouch_indices = tuple(int(index) for index in crouch_action_indices)
        crouch_scales = tuple(float(scale) for scale in crouch_action_scales)
        if len(crouch_indices) != len(crouch_scales) or len(set(crouch_indices)) != len(crouch_indices):
            raise ValueError("CROUCH indices/scales must be unique paired values")
        if any(index < 0 or index >= mean_dim for index in crouch_indices):
            raise ValueError(f"Invalid CROUCH action indices: {crouch_indices}")
        if any(scale <= 0.0 or scale > 0.25 for scale in crouch_scales):
            raise ValueError(f"CROUCH scales must be in (0, 0.25]: {crouch_scales}")
        crouch_scale = torch.zeros(mean_dim)
        crouch_scale[list(crouch_indices)] = torch.tensor(crouch_scales)
        # Non-persistent keeps old model_31 checkpoints strictly loadable; this
        # is a fixed architectural constraint, not learned checkpoint state.
        self.register_buffer("crouch_action_scale", crouch_scale, persistent=False)
        self.crouch_action_indices = crouch_indices
        self.residual_scale = float(residual_scale)
        if crouch_controller not in {"scripted_shallow_v1", "learned_legacy", "disabled"}:
            raise ValueError(f"Unknown CROUCH controller: {crouch_controller}")
        self.crouch_controller = str(crouch_controller)
        self.learned_crouch_residual_enabled = bool(learned_crouch_residual_enabled)
        self.step_over_controller = str(step_over_controller)
        self.learned_step_over_residual_enabled = bool(learned_step_over_residual_enabled)
        self.trainable_skill_ids = tuple(int(skill_id) for skill_id in trainable_skill_ids)
        if any(skill_id < 0 or skill_id >= self.skill_count for skill_id in self.trainable_skill_ids):
            raise ValueError(f"Invalid trainable skill ids: {self.trainable_skill_ids}")
        if freeze_base:
            for parameter in self.base_mlp.parameters():
                parameter.requires_grad_(False)
        for parameter in self.stand_base_mlp.parameters():
            parameter.requires_grad_(False)
        for skill_id, (command_encoder, state_adapter, head) in enumerate(
            zip(self.skill_command_encoders, self.skill_state_adapters, self.residual_heads)
        ):
            if skill_id not in self.trainable_skill_ids:
                for parameter in (*command_encoder.parameters(), *state_adapter.parameters(), *head.parameters()):
                    parameter.requires_grad_(False)
        if not train_stop_correction:
            for parameter in (
                *self.stop_corrective_command_encoder.parameters(),
                *self.stop_corrective_state_adapter.parameters(),
                *self.stop_corrective_head.parameters(),
            ):
                parameter.requires_grad_(False)
        self._last_diagnostics: dict[str, torch.Tensor] = {}
        self.stop_residual_ablation = "current"
        self.stop_residual_joint_indices: dict[str, int] = {}
        self.stop_fixed_feedback_gains = (0.0, 0.0, 0.0, 0.0)

    def configure_stop_fixed_feedback(
        self, k_heading: float, k_yaw_rate: float, k_roll: float = 0.0, k_roll_rate: float = 0.0
    ) -> None:
        """Configure a low-dimensional, bounded STOP feedback controller."""
        gains = (float(k_heading), float(k_yaw_rate), float(k_roll), float(k_roll_rate))
        if any(abs(gain) > 1.0 for gain in gains):
            raise ValueError(f"Unreasonable STOP fixed-feedback gains: {gains}")
        self.stop_fixed_feedback_gains = gains

    def _stop_fixed_feedback(self, legacy: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        k_heading, k_yaw_rate, k_roll, k_roll_rate = self.stop_fixed_feedback_gains
        delta = torch.zeros(
            *legacy.shape[:-1], self.stop_correction_scale.shape[0], device=legacy.device, dtype=legacy.dtype
        )
        if not any(self.stop_fixed_feedback_gains):
            return delta
        heading_error = torch.atan2(command[..., 12], command[..., 13])
        yaw_signal = k_heading * heading_error + k_yaw_rate * (legacy[..., 11] - legacy[..., 5])
        roll = torch.atan2(-legacy[..., 7], -legacy[..., 8])
        roll_signal = k_roll * roll + k_roll_rate * legacy[..., 3]
        delta[..., 2] = (-0.60 * yaw_signal).clamp(-0.03, 0.03)
        delta[..., 7] = (-1.00 * yaw_signal).clamp(-0.03, 0.03)
        delta[..., 8] = (-0.50 * yaw_signal).clamp(-0.03, 0.03)
        delta[..., 3] = roll_signal.clamp(-0.03, 0.03)
        delta[..., 4] = roll_signal.clamp(-0.03, 0.03)
        delta[..., 19] = (0.35 * roll_signal).clamp(-0.03, 0.03)
        delta[..., 20] = (0.35 * roll_signal).clamp(-0.03, 0.03)
        return delta

    def configure_stop_residual_ablation(self, mode: str, joint_indices: dict[str, int]) -> None:
        """Configure an evaluation-only projection of the STOP residual action."""
        allowed = {"current", "yaw_mask", "yaw_ankle_roll_mask", "lateral_mask", "symmetric"}
        if mode not in allowed:
            raise ValueError(f"Unknown STOP residual ablation: {mode}")
        self.stop_residual_ablation = mode
        self.stop_residual_joint_indices = {name: int(index) for name, index in joint_indices.items()}

    def _project_stop_residual(self, residual_actions: torch.Tensor) -> torch.Tensor:
        mode = self.stop_residual_ablation
        if mode == "current":
            return residual_actions
        projected = residual_actions.clone()
        stop = projected[..., 1, :]
        indices = self.stop_residual_joint_indices
        zero_names = ["torso", "left_hip_yaw", "right_hip_yaw"]
        if mode in {"yaw_ankle_roll_mask", "lateral_mask", "symmetric"}:
            zero_names += ["left_ankle_roll", "right_ankle_roll"]
        if mode in {"lateral_mask", "symmetric"}:
            zero_names += ["left_hip_roll", "right_hip_roll"]
        for name in zero_names:
            stop[..., indices[name]] = 0.0
        if mode == "symmetric":
            for left_name, right_name in (
                ("left_hip_pitch", "right_hip_pitch"),
                ("left_knee", "right_knee"),
                ("left_ankle_pitch", "right_ankle_pitch"),
            ):
                average = 0.5 * (stop[..., indices[left_name]] + stop[..., indices[right_name]])
                stop[..., indices[left_name]] = average
                stop[..., indices[right_name]] = average
        return projected

    @staticmethod
    def _expand_legacy_shared_routes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Upgrade old shared encoder/adapter checkpoints without changing RUN actions."""
        if not any(key.startswith("command_encoder.") for key in state_dict):
            return state_dict
        upgraded = state_dict.copy()
        for legacy_prefix, route_prefix in (
            ("command_encoder.", "skill_command_encoders."),
            ("state_adapter.", "skill_state_adapters."),
        ):
            legacy = {
                key.removeprefix(legacy_prefix): value
                for key, value in state_dict.items()
                if key.startswith(legacy_prefix)
            }
            for key in tuple(upgraded):
                if key.startswith(legacy_prefix):
                    del upgraded[key]
            for skill_id in range(SKILL_COUNT):
                for suffix, value in legacy.items():
                    upgraded[f"{route_prefix}{skill_id}.{suffix}"] = value.clone()
        return upgraded

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # model_39 predates skill-local encoders.  Replicating the learned RUN
        # route to each new route makes pure RUN output bitwise equivalent.
        state_dict = self._expand_legacy_shared_routes(state_dict)
        if not any(key.startswith("stop_corrective_") for key in state_dict):
            # model_31 predates the corrective route.  Copy the newly initialized
            # zero-output route into the incoming state so strict loading remains
            # available and the deterministic policy is exactly preserved.
            state_dict = state_dict.copy()
            initialized = super().state_dict()
            for key, value in initialized.items():
                if key.startswith("stop_corrective_") or key == "stop_correction_scale":
                    state_dict[key] = value.clone()
        if not any(key.startswith("stand_base_mlp.") for key in state_dict):
            state_dict = state_dict.copy()
            for key, value in tuple(state_dict.items()):
                if key.startswith("base_mlp."):
                    state_dict[f"stand_base_mlp.{key.removeprefix('base_mlp.')}"] = value.clone()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _mean_and_diagnostics(self, latent: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        legacy = latent[..., : self.legacy_observation_dim]
        command = latent[..., self.legacy_observation_dim :]
        running_base_action = self.base_mlp(legacy)
        standing_base_action = self.stand_base_mlp(legacy)
        command_embeddings = torch.stack([encoder(command) for encoder in self.skill_command_encoders], dim=-2)
        state_embeddings = torch.stack([adapter(legacy) for adapter in self.skill_state_adapters], dim=-2)
        raw_residual_actions = torch.stack([
            (self.crouch_action_scale if skill_id == CROUCH_SKILL_ID else self.residual_scale) * torch.tanh(
                head(torch.cat((state_embeddings[..., skill_id, :], command_embeddings[..., skill_id, :]), dim=-1))
            )
            for skill_id, head in enumerate(self.residual_heads)
        ], dim=-2)
        residual_actions = self._project_stop_residual(raw_residual_actions)
        current_gate = command[..., 0:SKILL_COUNT]
        previous_gate = command[..., SKILL_COUNT : 2 * SKILL_COUNT]
        transition = command[..., 25:26].clamp(0.0, 1.0)
        gate = transition * current_gate + (1.0 - transition) * previous_gate
        gate = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1.0)
        # CROUCH column 20 carries the safety-gated base-option cross-fade.
        # Multiplication by the skill gate prevents selection outside CROUCH.
        standing_skill_gate = gate[..., CROUCH_SKILL_ID : CROUCH_SKILL_ID + 1] + gate[..., STEP_OVER_SKILL_ID : STEP_OVER_SKILL_ID + 1]
        stand_base_gate = standing_skill_gate.clamp(0.0, 1.0) * command[..., 20:21].clamp(0.0, 1.0)
        blended_base_action = (
            (1.0 - stand_base_gate) * running_base_action + stand_base_gate * standing_base_action
        )
        # Exact endpoint selection preserves the old RUN/TURN/STOP arithmetic
        # bitwise and prevents the running action from entering pure CROUCH.
        selected_base_action = torch.where(
            stand_base_gate == 0.0,
            running_base_action,
            torch.where(stand_base_gate == 1.0, standing_base_action, blended_base_action),
        )
        legacy_residual_action = torch.sum(gate.unsqueeze(-1) * residual_actions, dim=-2)
        crouch_gate = gate[..., CROUCH_SKILL_ID : CROUCH_SKILL_ID + 1]
        learned_crouch = crouch_gate * residual_actions[..., CROUCH_SKILL_ID, :]
        residual_action = legacy_residual_action
        if not self.learned_crouch_residual_enabled:
            residual_action = torch.where(
                crouch_gate == 0.0, legacy_residual_action, legacy_residual_action - learned_crouch
            )
        step_gate = gate[..., STEP_OVER_SKILL_ID : STEP_OVER_SKILL_ID + 1]
        learned_step_over = step_gate * residual_actions[..., STEP_OVER_SKILL_ID, :]
        if not self.learned_step_over_residual_enabled:
            residual_action = torch.where(step_gate == 0.0, residual_action, residual_action - learned_step_over)
        raw_residual_action = torch.sum(gate.unsqueeze(-1) * raw_residual_actions, dim=-2)
        command_embedding = torch.sum(gate.unsqueeze(-1) * command_embeddings, dim=-2)
        state_embedding = torch.sum(gate.unsqueeze(-1) * state_embeddings, dim=-2)
        parent_action_mean = selected_base_action + residual_action
        scripted_crouch_offset = torch.zeros_like(parent_action_mean)
        if self.crouch_controller == "scripted_shallow_v1":
            instantaneous_depth = (-command[..., 16]).clamp_min(0.0)
            candidate_offset = pose_for_depth(instantaneous_depth, action_dim=parent_action_mean.shape[-1])
            scripted_crouch_offset = crouch_gate * candidate_offset
            parent_action_mean = torch.where(
                crouch_gate == 0.0, parent_action_mean, parent_action_mean + scripted_crouch_offset
            )
        # v0 is intentionally fail-closed: reachability did not yield a complete
        # safe pose chain, so no STEP_OVER offset is authorized yet.
        scripted_step_over_offset = torch.zeros_like(parent_action_mean)
        corrective_command_embedding = self.stop_corrective_command_encoder(command)
        corrective_state_embedding = self.stop_corrective_state_adapter(legacy)
        corrective_logits = self.stop_corrective_head(
            torch.cat((corrective_state_embedding, corrective_command_embedding), dim=-1)
        )
        bounded_stop_correction = self.stop_correction_scale * torch.tanh(corrective_logits)
        stop_gate = gate[..., STOP_SKILL_ID : STOP_SKILL_ID + 1]
        selected_stop_correction = stop_gate * bounded_stop_correction
        fixed_stop_feedback = stop_gate * self._stop_fixed_feedback(legacy, command)
        total_stop_correction = selected_stop_correction + fixed_stop_feedback
        action_mean = parent_action_mean + total_stop_correction
        diagnostics = {
            "base_action": selected_base_action,
            "running_base_action": running_base_action,
            "standing_base_action": standing_base_action,
            "selected_base_action": selected_base_action,
            "base_action_difference": standing_base_action - running_base_action,
            "stand_base_gate": stand_base_gate,
            "base_crossfade_progress": stand_base_gate,
            "command_embedding": command_embedding,
            "command_embeddings": command_embeddings,
            "state_embedding": state_embedding,
            "state_embeddings": state_embeddings,
            "gate": gate,
            "residual_actions": residual_actions,
            "raw_residual_actions": raw_residual_actions,
            "selected_residual": residual_action,
            "legacy_selected_residual": legacy_residual_action,
            "learned_crouch_residual": learned_crouch,
            "scripted_crouch_offset": scripted_crouch_offset,
            "learned_step_over_residual": learned_step_over,
            "scripted_step_over_offset": scripted_step_over_offset,
            "selected_raw_residual": raw_residual_action,
            "parent_action_mean": parent_action_mean,
            "bounded_stop_correction": bounded_stop_correction,
            "selected_stop_correction": selected_stop_correction,
            "fixed_stop_feedback": fixed_stop_feedback,
            "total_stop_correction": total_stop_correction,
            "parent_action_deviation": action_mean - parent_action_mean,
            "action_mean": action_mean,
        }
        return action_mean, diagnostics

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        latent = self.get_latent(obs, masks, hidden_state)
        mean, diagnostics = self._mean_and_diagnostics(latent)
        self._last_diagnostics = {key: value.detach() for key, value in diagnostics.items()}
        _LATEST_STOP_CORRECTION.clear()
        _LATEST_STOP_CORRECTION.update({
            "corrective_residual": diagnostics["total_stop_correction"].detach(),
            "parent_action_deviation": diagnostics["parent_action_deviation"].detach(),
        })
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mean)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mean)
        return mean

    @torch.no_grad()
    def diagnostic_components(self, obs: TensorDict) -> dict[str, torch.Tensor]:
        """Return deterministic components without changing the simulator."""
        latent = self.get_latent(obs)
        _, diagnostics = self._mean_and_diagnostics(latent)
        return diagnostics

    @torch.no_grad()
    def command_weight_norms(self, skill_id: int | None = None) -> torch.Tensor:
        """Return per-skill L2 norms for every appended command input column."""
        norms = torch.stack([
            torch.linalg.vector_norm(next(layer for layer in encoder if isinstance(layer, nn.Linear)).weight, dim=0)
            for encoder in self.skill_command_encoders
        ])
        return norms if skill_id is None else norms[int(skill_id)]

    def as_jit(self) -> nn.Module:
        return _ExportResidualActor(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxResidualActor(self, verbose)


class _ExportResidualActor(nn.Module):
    def __init__(self, actor: G1CommandResidualActor) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(actor.obs_normalizer)
        self.base_mlp = copy.deepcopy(actor.base_mlp)
        self.skill_command_encoders = copy.deepcopy(actor.skill_command_encoders)
        self.skill_state_adapters = copy.deepcopy(actor.skill_state_adapters)
        self.residual_heads = copy.deepcopy(actor.residual_heads)
        self.stop_corrective_command_encoder = copy.deepcopy(actor.stop_corrective_command_encoder)
        self.stop_corrective_state_adapter = copy.deepcopy(actor.stop_corrective_state_adapter)
        self.stop_corrective_head = copy.deepcopy(actor.stop_corrective_head)
        self.register_buffer("stop_correction_scale", actor.stop_correction_scale.detach().clone())
        self.register_buffer("crouch_action_scale", actor.crouch_action_scale.detach().clone())
        self.legacy_observation_dim = actor.legacy_observation_dim
        self.residual_scale = actor.residual_scale
        self.stop_fixed_feedback_gains = actor.stop_fixed_feedback_gains
        self.deterministic_output = (
            actor.distribution.as_deterministic_output_module()
            if actor.distribution is not None
            else nn.Identity()
        )

    def _stop_fixed_feedback(self, legacy: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        k_heading, k_yaw_rate, k_roll, k_roll_rate = self.stop_fixed_feedback_gains
        delta = torch.zeros(*legacy.shape[:-1], self.stop_correction_scale.shape[0], device=legacy.device, dtype=legacy.dtype)
        if not any(self.stop_fixed_feedback_gains):
            return delta
        heading_error = torch.atan2(command[..., 12], command[..., 13])
        yaw_signal = k_heading * heading_error + k_yaw_rate * (legacy[..., 11] - legacy[..., 5])
        roll = torch.atan2(-legacy[..., 7], -legacy[..., 8])
        roll_signal = k_roll * roll + k_roll_rate * legacy[..., 3]
        delta[..., 2] = (-0.60 * yaw_signal).clamp(-0.03, 0.03)
        delta[..., 7] = (-1.00 * yaw_signal).clamp(-0.03, 0.03)
        delta[..., 8] = (-0.50 * yaw_signal).clamp(-0.03, 0.03)
        delta[..., 3] = roll_signal.clamp(-0.03, 0.03)
        delta[..., 4] = roll_signal.clamp(-0.03, 0.03)
        delta[..., 19] = (0.35 * roll_signal).clamp(-0.03, 0.03)
        delta[..., 20] = (0.35 * roll_signal).clamp(-0.03, 0.03)
        return delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        legacy = x[..., : self.legacy_observation_dim]
        command = x[..., self.legacy_observation_dim :]
        base = self.base_mlp(legacy)
        command_embeddings = [encoder(command) for encoder in self.skill_command_encoders]
        state_embeddings = [adapter(legacy) for adapter in self.skill_state_adapters]
        residuals = torch.stack([
            (self.crouch_action_scale if skill_id == CROUCH_SKILL_ID else self.residual_scale) * torch.tanh(
                head(torch.cat((state_embeddings[skill_id], command_embeddings[skill_id]), dim=-1))
            )
            for skill_id, head in enumerate(self.residual_heads)
        ], dim=-2)
        transition = command[..., 25:26].clamp(0.0, 1.0)
        gate = transition * command[..., 0:6] + (1.0 - transition) * command[..., 6:12]
        gate = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1.0)
        parent = base + torch.sum(gate.unsqueeze(-1) * residuals, dim=-2)
        correction = self.stop_correction_scale * torch.tanh(
            self.stop_corrective_head(torch.cat((
                self.stop_corrective_state_adapter(legacy),
                self.stop_corrective_command_encoder(command),
            ), dim=-1))
        )
        fixed = gate[..., STOP_SKILL_ID : STOP_SKILL_ID + 1] * self._stop_fixed_feedback(legacy, command)
        return self.deterministic_output(parent + gate[..., STOP_SKILL_ID : STOP_SKILL_ID + 1] * correction + fixed)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxResidualActor(_ExportResidualActor):
    is_recurrent = False

    def __init__(self, actor: G1CommandResidualActor, verbose: bool) -> None:
        super().__init__(actor)
        self.verbose = verbose
        self.input_size = actor.obs_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
