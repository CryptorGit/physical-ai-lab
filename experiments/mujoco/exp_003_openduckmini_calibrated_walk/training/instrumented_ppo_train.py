"""Minimal Brax 0.14.2 PPO loop with complete update-boundary state.

The PPO equations, random split order inside rollout/SGD, normalizer timing, and
minibatch order are copied structurally from the installed Brax 0.14.2
``brax.training.agents.ppo.train`` implementation.  The differences are:

* the completed update state is returned instead of hidden in ``train``;
* one-GPU execution omits the size-one pmap axis (pmean is the identity);
* rollout telemetry is reduced on device and returned only at update boundaries;
* the environment, learner, and RNG state are serializable.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
from pathlib import Path
from typing import Any, Callable, Mapping

from brax import envs
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint as brax_checkpoint
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as brax_train
import jax
import jax.numpy as jnp
import optax

from .device_metrics import aggregate_rollout, tree_l2_norm, tree_nonfinite_count
from .training_state import CounterState, ExactTrainingState, make_counters


@dataclass(frozen=True)
class HarnessConfig:
    seed: int = 20260730
    num_envs: int = 1250
    episode_length: int = 1000
    action_repeat: int = 1
    unroll_length: int = 20
    batch_size: int = 125
    num_minibatches: int = 20
    num_updates_per_batch: int = 4
    learning_rate: float = 3e-4
    entropy_cost: float = 0.005
    discounting: float = 0.97
    reward_scaling: float = 1.0
    gae_lambda: float = 0.95
    clipping_epsilon: float = 0.2
    normalize_advantage: bool = True
    vf_loss_coefficient: float = 0.5
    max_grad_norm: float | None = 1.0
    normalize_observations: bool = True
    normalize_observations_std_eps: float = 1e-3
    normalize_observations_mode: str = "std"
    policy_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
    value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
    policy_obs_key: str = "state"
    value_obs_key: str = "privileged_state"

    @property
    def environment_interactions_per_update(self) -> int:
        return self.batch_size * self.unroll_length * self.num_minibatches

    @property
    def adam_steps_per_update(self) -> int:
        return self.num_updates_per_batch * self.num_minibatches

    def validate(self) -> None:
        samples = self.batch_size * self.num_minibatches
        if samples % self.num_envs:
            raise ValueError(
                "batch_size * num_minibatches must be divisible by num_envs"
            )


def identity_test_config(seed: int = 20260730) -> HarnessConfig:
    """Small non-performance profile; production PPO defaults remain frozen."""
    return HarnessConfig(
        seed=seed,
        num_envs=4,
        unroll_length=2,
        batch_size=1,
        num_minibatches=4,
        num_updates_per_batch=1,
    )


def _remove_pixels(obs: Any) -> Any:
    if not isinstance(obs, Mapping):
        return obs
    return {k: v for k, v in obs.items() if not k.startswith("pixels/")}


def _find_domain_wrapper(env: Any) -> Any:
    current = env
    while current is not None:
        if (
            (hasattr(current, "_sys_v") or hasattr(current, "_mjx_model_v"))
            and hasattr(current, "_in_axes")
        ):
            return current
        current = getattr(current, "env", None)
    raise ValueError("DomainRandomizationVmapWrapper not found")


def _tree_all_equal(left: Any, right: Any) -> bool:
    left_leaves, left_def = jax.tree_util.tree_flatten(jax.device_get(left))
    right_leaves, right_def = jax.tree_util.tree_flatten(jax.device_get(right))
    if left_def != right_def or len(left_leaves) != len(right_leaves):
        return False
    import numpy as np

    return all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(left_leaves, right_leaves)
    )


class InstrumentedPPOHarness:
    """Owns an exact-resumable one-device PPO update function."""

    def __init__(
        self,
        *,
        environment: envs.Env,
        randomization_fn: Callable[..., Any],
        wrap_env_fn: Callable[..., Any],
        parent_checkpoint: Path,
        official_commands: jax.Array,
        config: HarnessConfig,
        instrumented: bool,
    ):
        config.validate()
        if jax.device_count() != 1:
            raise ValueError("harness_v1 requires exactly one visible GPU device")
        self.base_environment = environment
        self.config = config
        self.instrumented = instrumented
        self.official_commands = jnp.asarray(official_commands, dtype=jnp.float32)
        self._transfer_count = 0
        self._transfer_bytes = 0

        key = jax.random.PRNGKey(config.seed)
        global_key, local_key = jax.random.split(key)
        local_key, key_env, eval_key = jax.random.split(local_key, 3)
        key_policy, key_value = jax.random.split(global_key)
        environment_keys = jax.random.split(key_env, config.num_envs)

        self.environment = brax_train._maybe_wrap_env(
            environment,
            True,
            config.num_envs,
            config.episode_length,
            config.action_repeat,
            1,
            key_env,
            wrap_env_fn,
            randomization_fn,
        )
        domain_wrapper = _find_domain_wrapper(self.environment)
        self.randomized_model = getattr(
            domain_wrapper, "_mjx_model_v", getattr(domain_wrapper, "_sys_v", None)
        )
        self.randomized_model_in_axes = domain_wrapper._in_axes

        reset_fn = jax.jit(self.environment.reset)
        environment_state = reset_fn(environment_keys)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), environment_state)

        obs_shape = jax.tree_util.tree_map(
            lambda x: x.shape[1:], environment_state.obs
        )
        normalize = (
            running_statistics.normalize
            if config.normalize_observations
            else lambda x, y: x
        )
        self.network = ppo_networks.make_ppo_networks(
            obs_shape,
            self.environment.action_size,
            preprocess_observations_fn=normalize,
            policy_hidden_layer_sizes=config.policy_hidden_layer_sizes,
            value_hidden_layer_sizes=config.value_hidden_layer_sizes,
            policy_obs_key=config.policy_obs_key,
            value_obs_key=config.value_obs_key,
        )
        self.make_policy = ppo_networks.make_inference_fn(self.network)

        init_params = ppo_losses.PPONetworkParams(
            policy=self.network.policy_network.init(key_policy),
            value=self.network.value_network.init(key_value),
        )
        parent = brax_checkpoint.load(str(parent_checkpoint))
        restored_params = ppo_losses.PPONetworkParams(
            policy=parent[1],
            value=parent[2],
        )
        base_optimizer = optax.adam(config.learning_rate)
        self.optimizer = (
            optax.chain(optax.clip_by_global_norm(config.max_grad_norm), base_optimizer)
            if config.max_grad_norm is not None
            else base_optimizer
        )
        learner_state = brax_train.TrainingState(
            optimizer_state=self.optimizer.init(init_params),
            params=restored_params,
            normalizer_params=parent[0],
            env_steps=types.UInt64(hi=0, lo=0),
        )
        counters = make_counters(config.num_envs)
        self.initial_state = ExactTrainingState(
            learner_state=learner_state,
            environment_state=environment_state,
            learner_rng=local_key,
            rollout_rng=jnp.zeros_like(local_key),
            evaluation_rng=eval_key,
            environment_keys=environment_keys,
            counters=counters,
        )
        self._build_update()

    def _build_update(self) -> None:
        config = self.config
        loss_fn = functools.partial(
            ppo_losses.compute_ppo_loss,
            ppo_network=self.network,
            entropy_cost=config.entropy_cost,
            discounting=config.discounting,
            reward_scaling=config.reward_scaling,
            gae_lambda=config.gae_lambda,
            clipping_epsilon=config.clipping_epsilon,
            normalize_advantage=config.normalize_advantage,
            vf_coefficient=config.vf_loss_coefficient,
        )
        loss_and_grad = jax.value_and_grad(loss_fn, has_aux=True)
        base_env = self.base_environment

        def sidecar_values(state, nstate):
            command = state.info["command"]
            local_velocity = jax.vmap(base_env.get_local_linvel)(nstate.data)
            yaw_velocity = jax.vmap(base_env.get_gyro)(nstate.data)[:, 2:3]
            gravity = jax.vmap(base_env.get_gravity)(nstate.data)
            base_height = nstate.data.qpos[:, base_env._floating_base_qpos_addr + 2]
            fall = (gravity[:, 2] < 0.65) | (base_height < 0.12)
            reward_terms = {
                name.replace("/", "_"): value for name, value in nstate.metrics.items()
            }
            return {
                "command": command,
                "done": nstate.done,
                "fall": fall,
                "truncation": nstate.info["truncation"],
                "episode_start": state.info["steps"] == 0,
                "reward": nstate.reward,
                "actual_velocity": jnp.concatenate(
                    [local_velocity[:, :2], yaw_velocity], axis=-1
                ),
                "action": None,
                "reward_terms": reward_terms,
            }

        def generate_unroll(state, policy, key):
            def actor_step(carry, _):
                current_state, current_key = carry
                action_key, next_key = jax.random.split(current_key)
                actions, policy_extras = policy(current_state.obs, action_key)
                nstate = self.environment.step(current_state, actions)
                transition = types.Transition(
                    observation=current_state.obs,
                    action=actions,
                    reward=nstate.reward,
                    discount=1 - nstate.done,
                    next_observation=nstate.obs,
                    extras={
                        "policy_extras": policy_extras,
                        "state_extras": {
                            "truncation": nstate.info["truncation"],
                            "episode_done": nstate.info["episode_done"],
                        },
                    },
                )
                sidecar = sidecar_values(current_state, nstate)
                sidecar["action"] = actions
                return (nstate, next_key), (transition, sidecar)

            (state, _), (data, sidecar) = jax.lax.scan(
                actor_step, (state, key), (), length=config.unroll_length
            )
            return state, data, sidecar

        def minibatch_step(carry, data, normalizer_params):
            optimizer_state, params, key = carry
            key, key_loss = jax.random.split(key)
            (loss, loss_metrics), grads = loss_and_grad(
                params, normalizer_params, data, key_loss
            )
            params_update, optimizer_state = self.optimizer.update(
                grads, optimizer_state, params
            )
            new_params = optax.apply_updates(params, params_update)
            adam_state = optimizer_state[1][0]
            diagnostics = {
                **loss_metrics,
                "total_loss": loss,
                "actor_gradient_norm": tree_l2_norm(grads.policy),
                "critic_gradient_norm": tree_l2_norm(grads.value),
                "global_gradient_norm": tree_l2_norm(grads),
                "parameter_update_norm": tree_l2_norm(params_update),
                "actor_parameter_norm": tree_l2_norm(new_params.policy),
                "critic_parameter_norm": tree_l2_norm(new_params.value),
                "optimizer_state_norm": tree_l2_norm(optimizer_state),
                "optimizer_first_moment_norm": tree_l2_norm(adam_state.mu),
                "optimizer_second_moment_norm": tree_l2_norm(adam_state.nu),
                "effective_adam_step_scale": tree_l2_norm(params_update)
                / jnp.maximum(tree_l2_norm(grads), 1e-12),
                "gradient_nonfinite_count": tree_nonfinite_count(grads),
                "parameter_nonfinite_count": tree_nonfinite_count(new_params),
                "learning_rate": jnp.asarray(config.learning_rate),
            }
            return (optimizer_state, new_params, key), diagnostics

        def sgd_step(carry, _, data, normalizer_params):
            optimizer_state, params, key = carry
            key, key_perm, key_grad = jax.random.split(key, 3)

            def convert_data(x):
                x = jax.random.permutation(key_perm, x)
                return jnp.reshape(
                    x, (config.num_minibatches, -1) + x.shape[1:]
                )

            shuffled = jax.tree_util.tree_map(convert_data, data)
            (optimizer_state, params, _), diagnostics = jax.lax.scan(
                functools.partial(
                    minibatch_step, normalizer_params=normalizer_params
                ),
                (optimizer_state, params, key_grad),
                shuffled,
                length=config.num_minibatches,
            )
            return (optimizer_state, params, key), diagnostics

        def update(exact_state: ExactTrainingState):
            learner = exact_state.learner_state
            epoch_key, next_learner_rng = jax.random.split(exact_state.learner_rng)
            training_key = jax.random.split(epoch_key, 1)[0]
            key_sgd, key_rollout, _ = jax.random.split(training_key, 3)
            policy = self.make_policy(
                (
                    learner.normalizer_params,
                    learner.params.policy,
                    learner.params.value,
                )
            )

            def collect(carry, _):
                state, key = carry
                current_key, next_key = jax.random.split(key)
                nstate, data, sidecar = generate_unroll(
                    state, policy, current_key
                )
                return (nstate, next_key), (data, sidecar)

            rollout_batches = (
                config.batch_size * config.num_minibatches // config.num_envs
            )
            (environment_state, _), (data, sidecar) = jax.lax.scan(
                collect,
                (exact_state.environment_state, key_rollout),
                (),
                length=rollout_batches,
            )
            done_count = jnp.sum(sidecar["done"], axis=(0, 1)).astype(jnp.int32)
            data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
            data = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
            )
            sidecar = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(x, 1, 2), sidecar
            )
            sidecar = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), sidecar
            )

            normalizer_params = running_statistics.update(
                learner.normalizer_params,
                _remove_pixels(data.observation),
                until_count=None,
            )
            # Reconstruct the exact GAE tensor already consumed by PPO.  This
            # side-channel is pure: it draws no RNG and does not enter the loss.
            time_major = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(x, 0, 1), data
            )
            baseline = self.network.value_network.apply(
                normalizer_params,
                learner.params.value,
                time_major.observation,
            )
            terminal_obs = jax.tree_util.tree_map(
                lambda x: x[-1], time_major.next_observation
            )
            bootstrap_value = self.network.value_network.apply(
                normalizer_params,
                learner.params.value,
                terminal_obs,
            )
            truncation = time_major.extras["state_extras"]["truncation"]
            termination = (1 - time_major.discount) * (1 - truncation)
            gae_returns, advantages = ppo_losses.compute_gae(
                truncation=truncation,
                termination=termination,
                rewards=time_major.reward * config.reward_scaling,
                values=baseline,
                bootstrap_value=bootstrap_value,
                lambda_=config.gae_lambda,
                discount=config.discounting,
            )
            if config.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )
            sidecar["advantage"] = jnp.swapaxes(advantages, 0, 1)
            explained_variance = 1.0 - jnp.var(gae_returns - baseline) / jnp.maximum(
                jnp.var(gae_returns), 1e-12
            )
            (optimizer_state, params, _), optimizer_metrics = jax.lax.scan(
                functools.partial(
                    sgd_step, data=data, normalizer_params=normalizer_params
                ),
                (learner.optimizer_state, learner.params, key_sgd),
                (),
                length=config.num_updates_per_batch,
            )
            environment_interactions = config.environment_interactions_per_update
            learner = brax_train.TrainingState(
                optimizer_state=optimizer_state,
                params=params,
                normalizer_params=normalizer_params,
                env_steps=learner.env_steps + environment_interactions,
            )
            counters = CounterState(
                optimizer_update_count=(
                    exact_state.counters.optimizer_update_count
                    + config.adam_steps_per_update
                ),
                harness_update_count=exact_state.counters.harness_update_count + 1,
                global_environment_interactions=(
                    exact_state.counters.global_environment_interactions
                    + environment_interactions
                ),
                reset_generation=exact_state.counters.reset_generation,
                episode_index=exact_state.counters.episode_index + done_count,
                episode_step=environment_state.info["steps"].astype(jnp.int32),
            )
            new_state = ExactTrainingState(
                learner_state=learner,
                environment_state=environment_state,
                learner_rng=next_learner_rng,
                rollout_rng=key_rollout,
                evaluation_rng=exact_state.evaluation_rng,
                environment_keys=exact_state.environment_keys,
                counters=counters,
            )
            if self.instrumented:
                reduced = aggregate_rollout(
                    sidecar,
                    self.official_commands,
                    num_updates_per_batch=config.num_updates_per_batch,
                    vx_edges=jnp.asarray([-0.12, -0.02, 0.02, 0.12]),
                    vy_edges=jnp.asarray([-0.14, -0.02, 0.02, 0.14]),
                    yaw_edges=jnp.asarray([-0.7, -0.1, 0.1, 0.7]),
                    head_edges=jnp.asarray([-0.5, -0.1, 0.1, 0.5]),
                )
                telemetry = {
                    "rollout": reduced,
                    "optimizer": jax.tree_util.tree_map(
                        jnp.mean, optimizer_metrics
                    ),
                    "explained_variance": explained_variance,
                }
            else:
                telemetry = {"instrumented": jnp.asarray(False)}
            probes = {
                "action": sidecar["action"],
                "reward": sidecar["reward"],
                "done": sidecar["done"],
                "command": sidecar["command"],
                "observation": data.observation,
            }
            return new_state, telemetry, probes

        self._update_fn = jax.jit(update)

    def update(
        self, state: ExactTrainingState
    ) -> tuple[ExactTrainingState, Mapping[str, Any], Mapping[str, Any]]:
        result = self._update_fn(state)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), result)
        return result

    def install_randomized_model(self, model: Any) -> None:
        wrapper = _find_domain_wrapper(self.environment)
        attribute = "_mjx_model_v" if hasattr(wrapper, "_mjx_model_v") else "_sys_v"
        if not _tree_all_equal(getattr(wrapper, attribute), model):
            setattr(wrapper, attribute, jax.device_put(model))
            self.randomized_model = getattr(wrapper, attribute)

    def record_host_transfer(self, telemetry: Mapping[str, Any]) -> None:
        from .device_metrics import telemetry_transfer_bytes

        self._transfer_count += 1
        self._transfer_bytes += telemetry_transfer_bytes(telemetry)

    @property
    def host_transfer_stats(self) -> dict[str, int]:
        return {
            "host_transfer_count": self._transfer_count,
            "host_transfer_bytes": self._transfer_bytes,
        }


def load_official_commands(path: Path) -> jax.Array:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get(
        "commands", payload.get("all_formal_19_commands", payload)
    )
    if entries and isinstance(entries[0], list):
        return jnp.asarray(entries, dtype=jnp.float32)
    return jnp.asarray(
        [[x["vx"], x["vy"], x["yaw_rate"]] for x in entries], dtype=jnp.float32
    )
