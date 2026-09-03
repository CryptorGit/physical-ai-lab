"""Local Stage 2N gait-conditioned actor and endpoint-anchored PPO."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn
from torch.distributions import Normal
from rsl_rl.algorithms import PPO
from rsl_rl.modules import GaussianDistribution
from rsl_rl.models import MLPModel


class GaitConditionedDiagonalGaussian(GaussianDistribution):
    """One diagonal Gaussian head with log-space gait interpolation."""

    def __init__(self, output_dim: int, **_: object) -> None:
        nn.Module.__init__(self)
        self.output_dim = output_dim
        self.log_std_walk = nn.Parameter(torch.zeros(output_dim))
        self.log_std_run = nn.Parameter(torch.zeros(output_dim))
        self._gait: torch.Tensor | None = None
        self._distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def set_gait(self, gait: torch.Tensor) -> None:
        self._gait = gait.reshape(-1, 1)

    def update(self, mean: torch.Tensor) -> None:
        if self._gait is None:
            raise RuntimeError("GAIT_CONDITIONED_STD_GAIT_MISSING")
        gait = self._gait
        if gait.shape[0] != mean.shape[0]:
            gait = gait.reshape(mean.shape[0], 1)
        log_std = (1.0 - gait) * self.log_std_walk + gait * self.log_std_run
        self._distribution = Normal(mean, torch.exp(log_std))

    def sample(self) -> torch.Tensor:
        return self._distribution.sample()

    def deterministic_output(self, mean: torch.Tensor) -> torch.Tensor:
        return mean

    def as_deterministic_output_module(self) -> nn.Module:
        return nn.Identity()

    @property
    def input_dim(self) -> int:
        return self.output_dim

    @property
    def mean(self) -> torch.Tensor:
        return self._distribution.mean

    @property
    def std(self) -> torch.Tensor:
        return self._distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self._distribution.entropy().sum(-1)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        return self.mean, self.std

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self._distribution.log_prob(outputs).sum(-1)

    def kl_divergence(self, old_params, new_params) -> torch.Tensor:
        return torch.distributions.kl_divergence(Normal(*old_params), Normal(*new_params)).sum(-1)

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        return None


class GaitConditionedMLPModel(MLPModel):
    """MLPModel that supplies the appended gait scalar to its Gaussian head."""

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(256, 128, 128),
                 activation="elu", obs_normalization=False, distribution_cfg=None):
        nn.Module.__init__(self)
        self.obs_groups = obs_groups[obs_set]
        self.obs_dim = 124
        self.obs_normalization = False
        self.obs_normalizer = nn.Identity()
        distribution_cfg = dict(distribution_cfg or {})
        distribution_cfg.pop("class_name", None)
        self.distribution = GaitConditionedDiagonalGaussian(output_dim, **distribution_cfg)
        self.first_base_weight = nn.Parameter(torch.empty(256, 123))
        self.first_gait_column = nn.Parameter(torch.empty(256, 1))
        self.first_bias = nn.Parameter(torch.empty(256))
        self.hidden = nn.Sequential(
            nn.ELU(), nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(), nn.Linear(128, output_dim),
        )

    def forward(self, obs: TensorDict, masks=None, hidden_state=None, stochastic_output=False):
        latent = self.get_latent(obs, masks, hidden_state)
        first = nn.functional.linear(latent[..., :123], self.first_base_weight, self.first_bias)
        first = first + latent[..., 123:124] * self.first_gait_column.T
        output = self.hidden(first)
        if self.distribution is not None:
            self.distribution.set_gait(latent[..., -1])
            if stochastic_output:
                self.distribution.update(output)
                return self.distribution.sample()
            return output
        return output


class GaitAnchoredPPO(PPO):
    """PPO with a frozen-reference, endpoint-balanced Gaussian KL anchor."""

    anchor_beta: float = 0.0
    anchor_observations: TensorDict | None = None
    reference_actor: nn.Module | None = None
    last_anchor_stats: dict[str, float] = {}

    def configure_anchor(self, observations: TensorDict, reference_actor: nn.Module, beta: float) -> None:
        self.anchor_observations = observations
        self.reference_actor = reference_actor
        self.anchor_beta = float(beta)
        for parameter in self.reference_actor.parameters():
            parameter.requires_grad_(False)
        self.reference_actor.eval()

    def _anchor_loss(self) -> tuple[torch.Tensor, dict[str, float]]:
        if self.anchor_observations is None or self.reference_actor is None:
            zero = next(self.actor.parameters()).sum() * 0
            return zero, {}
        endpoint = self.anchor_observations["endpoint_id"]
        losses = []
        stats = {}
        for index, name in enumerate(("walk_1p2", "run_1p2", "run_2p4", "run_2p6")):
            ids = torch.nonzero(endpoint == index, as_tuple=False).flatten()
            chosen = ids[torch.randint(len(ids), (min(512, len(ids)),), device=ids.device)]
            batch = TensorDict(
                {"policy": self.anchor_observations["policy"][chosen]},
                batch_size=[chosen.numel()],
                device=chosen.device,
            )
            with torch.no_grad():
                self.reference_actor(batch, stochastic_output=True)
                ref = tuple(value.detach().clone() for value in self.reference_actor.output_distribution_params)
            self.actor(batch, stochastic_output=True)
            cur = self.actor.output_distribution_params
            kl = self.actor.get_kl_divergence(ref, cur).mean()
            losses.append(kl)
            stats[name] = float(kl.detach())
        return torch.stack(losses).mean(), stats

    def update(self) -> dict[str, float]:
        mean_value = mean_surrogate = mean_entropy = mean_anchor = 0.0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        updates = self.num_mini_batches * self.num_learning_epochs
        for batch in generator:
            self.actor(batch.observations, stochastic_output=True)
            log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations)
            params = self.actor.output_distribution_params
            entropy = self.actor.output_entropy
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.no_grad():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, params).mean()
                if kl > self.desired_kl * 2:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif 0 < kl < self.desired_kl / 2:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                for group in self.optimizer.param_groups:
                    group["lr"] = self.learning_rate
            ratio = torch.exp(log_prob - batch.old_actions_log_prob.squeeze(-1))
            surrogate = -batch.advantages.squeeze(-1) * ratio
            clipped = -batch.advantages.squeeze(-1) * torch.clamp(
                ratio, 1 - self.clip_param, 1 + self.clip_param
            )
            surrogate_loss = torch.maximum(surrogate, clipped).mean()
            value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
            value_loss = torch.maximum(
                (values - batch.returns).square(), (value_clipped - batch.returns).square()
            ).mean()
            anchor_loss, anchor_stats = self._anchor_loss()
            loss = (
                surrogate_loss + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean() + self.anchor_beta * anchor_loss
            )
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            mean_value += float(value_loss)
            mean_surrogate += float(surrogate_loss)
            mean_entropy += float(entropy.mean())
            mean_anchor += float(anchor_loss)
            self.last_anchor_stats = anchor_stats
        self.storage.clear()
        return {
            "value": mean_value / updates,
            "surrogate": mean_surrogate / updates,
            "entropy": mean_entropy / updates,
            "anchor": mean_anchor / updates,
        }
