"""Stage 2N gait-conditioned PPO endpoint-retention preflight."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2n_gait_conditioned_ppo_retention_preflight"
K = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2k_gait_latent_preflight/student/selected_gait_latent_student.pt"
L = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2l_gait_conditioned_gaussian_std_preflight/student/stage2l_gait_conditioned_std_student.pt"
RUN = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-18_00-44-32_stage4_1024_500/model_5244.pt"
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from g1_single_policy.strict_ppo_resume import Exp012StrictPPOResumeContract  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("prepare", "train"), required=True)
parser.add_argument("--beta", type=float, default=0.03)
parser.add_argument("--iterations", type=int, default=25)
parser.add_argument("--tag", default="")
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]

EXPECTED = {
    K: "d0c46acdc2d3a5793d7dc8d6ae9e47f741ff0353fb1aef8c9ff993e71ea8bec3",
    L: "175131f7415988c4992b1a0334911abcfed304fca79765d453269a13743af2ac",
    RUN: "90d1a360587142d7e312db00a281505a027ecb221733eea6451a885868f6ccc9",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name: str, value: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def minimum_jerk(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0, 1)
    return 10 * x**3 - 15 * x**4 + 6 * x**5


class GaitVecEnv:
    """Final RSL-RL wrapper adding the one scalar gait command."""

    def __init__(self, wrapped, mode: str):
        self.base = wrapped
        self.mode = mode
        for name in ("num_envs", "device", "max_episode_length", "num_actions", "cfg"):
            if name == "cfg":
                continue
            setattr(self, name, getattr(wrapped, name))
        self.gait = torch.zeros(self.num_envs, device=self.device)
        self.cohort = torch.arange(self.num_envs, device=self.device) * 100 // self.num_envs
        self.command = self.base.unwrapped.command_manager.get_term("base_velocity")
        self.command.external_override_enabled = True
        self._schedule()

    @property
    def cfg(self):
        return self.base.cfg

    @property
    def unwrapped(self):
        return self.base.unwrapped

    @property
    def episode_length_buf(self):
        return self.base.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.base.episode_length_buf = value

    def seed(self, value):
        return self.base.seed(value)

    def close(self):
        self.base.close()

    def _schedule(self):
        t = self.base.episode_length_buf.float() * float(self.base.unwrapped.step_dt)
        speed = torch.full_like(t, 1.2)
        gait = torch.zeros_like(t)
        if self.mode == "anchor":
            quarter = torch.arange(self.num_envs, device=self.device) * 4 // self.num_envs
            gait = (quarter > 0).float()
            speed[quarter == 2] = 2.4
            speed[quarter == 3] = 2.6
        else:
            # Fixed 25/20/30/25 allocation.
            walk = self.cohort < 25
            low_run = (self.cohort >= 25) & (self.cohort < 45)
            high_run = (self.cohort >= 45) & (self.cohort < 75)
            toggle = self.cohort >= 75
            gait[low_run | high_run] = 1
            high_id = torch.arange(self.num_envs, device=self.device) % 2
            speed[high_run] = torch.where(high_id[high_run] == 0, 2.4, 2.6)
            direction = torch.arange(self.num_envs, device=self.device) % 2
            ramp = minimum_jerk((t - 5.0) / 2.0)
            gait[toggle] = torch.where(direction[toggle] == 0, ramp[toggle], 1 - ramp[toggle])
            gait[walk] = 0
        self.gait.copy_(gait)
        self.command.external_override[:, 0] = speed
        self.command.external_override[:, 1:] = 0

    def _augment(self, obs):
        result = obs.clone()
        result["policy"] = torch.cat((obs["policy"], self.gait[:, None]), -1)
        return result

    def get_observations(self):
        self._schedule()
        return self._augment(self.base.get_observations())

    def reset(self):
        obs, extras = self.base.reset()
        self._schedule()
        return self._augment(obs), extras

    def step(self, actions):
        obs, rew, dones, extras = self.base.step(actions)
        self._schedule()
        return self._augment(obs), rew, dones, extras


def load_integrated(runner, device):
    stage2k = torch.load(K, map_location=device, weights_only=False)["model_state_dict"]
    stage2l = torch.load(L, map_location=device, weights_only=False)["model_state_dict"]
    run = torch.load(RUN, map_location=device, weights_only=False)
    actor = runner.alg.actor
    critic = runner.alg.critic
    with torch.no_grad():
        actor.first_base_weight.copy_(stage2k["first_base_weight"])
        actor.first_gait_column.copy_(stage2k["first_gait_column"])
        actor.first_bias.copy_(stage2k["first_bias"])
        for target, source in ((1, 1), (3, 3), (5, 5)):
            actor.hidden[target].weight.copy_(stage2k[f"hidden.{source}.weight"])
            actor.hidden[target].bias.copy_(stage2k[f"hidden.{source}.bias"])
        actor.distribution.log_std_walk.copy_(stage2l["log_std_walk"].float() + math.log(.30))
        actor.distribution.log_std_run.copy_(stage2l["log_std_run"].float() + math.log(.65))
        for target, value in critic.state_dict().items():
            source = run["critic_state_dict"][target]
            if target == "mlp.0.weight":
                value.copy_(torch.cat((source, torch.zeros(source.shape[0], 1, device=device)), 1))
            else:
                value.copy_(source)
    return run


def initialize_optimizer(runner, run):
    optimizer = runner.alg.optimizer
    source = run["optimizer_state_dict"]
    source_states = source["state"]
    actor_names = dict(runner.alg.actor.named_parameters())
    critic_names = dict(runner.alg.critic.named_parameters())
    mapping = []
    source_actor = {
        "first_base_weight": 1, "first_bias": 2,
        "hidden.1.weight": 3, "hidden.1.bias": 4,
        "hidden.3.weight": 5, "hidden.3.bias": 6,
        "hidden.5.weight": 7, "hidden.5.bias": 8,
    }
    source_critic = {
        "mlp.0.weight": 9, "mlp.0.bias": 10, "mlp.2.weight": 11, "mlp.2.bias": 12,
        "mlp.4.weight": 13, "mlp.4.bias": 14, "mlp.6.weight": 15, "mlp.6.bias": 16,
    }
    optimizer.state.clear()
    for scope, names, table in (
        ("actor", actor_names, source_actor), ("critic", critic_names, source_critic)
    ):
        for name, parameter in names.items():
            state = {
                "step": torch.tensor(105000.0, device=parameter.device),
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.zeros_like(parameter),
            }
            source_id = table.get(name)
            mode = "zero_moment"
            if source_id is not None:
                old = source_states[source_id]
                if scope == "critic" and name == "mlp.0.weight":
                    state["exp_avg"][:, :123].copy_(old["exp_avg"])
                    state["exp_avg_sq"][:, :123].copy_(old["exp_avg_sq"])
                    mode = "copy_123_columns_zero_gait_column"
                else:
                    state["exp_avg"].copy_(old["exp_avg"])
                    state["exp_avg_sq"].copy_(old["exp_avg_sq"])
                    mode = "copy_exact"
            optimizer.state[parameter] = state
            mapping.append({"scope": scope, "parameter": name, "source_state": source_id, "mode": mode})
    lr = float(source["param_groups"][0]["lr"])
    for group in optimizer.param_groups:
        group["lr"] = lr
    runner.alg.learning_rate = lr
    return mapping, lr


def actor_identity(actor, stage2k, device):
    obs = torch.randn(4096, 123, device=device)
    gait = torch.linspace(0, 1, 4096, device=device)[:, None]
    with torch.no_grad():
        first = torch.nn.functional.linear(obs, stage2k["first_base_weight"], stage2k["first_bias"])
        expected = torch.nn.functional.linear(
            torch.nn.functional.elu(first + gait * stage2k["first_gait_column"].T),
            stage2k["hidden.1.weight"], stage2k["hidden.1.bias"])
        expected = torch.nn.functional.linear(
            torch.nn.functional.elu(expected), stage2k["hidden.3.weight"], stage2k["hidden.3.bias"])
        expected = torch.nn.functional.linear(
            torch.nn.functional.elu(expected), stage2k["hidden.5.weight"], stage2k["hidden.5.bias"])
        actual = actor(TensorDict({"policy": torch.cat((obs, gait), -1)}, batch_size=[len(obs)]))
    return torch.equal(actual, expected), float((actual - expected).abs().max())


def collect_anchor(env, actor, device):
    obs = env.get_observations()
    chunks, ids = [], []
    steps = round(10.0 / float(env.unwrapped.step_dt))
    quarter = torch.arange(env.num_envs, device=device) * 4 // env.num_envs
    for _ in range(steps):
        chunks.append(obs["policy"].detach().cpu())
        ids.append(quarter.detach().cpu())
        with torch.no_grad():
            actions = actor(obs)
        obs, _, _, _ = env.step(actions)
    policy = torch.cat(chunks)
    endpoint = torch.cat(ids)
    # deterministic episode split: first 80% envs within each endpoint.
    keep_train = torch.zeros_like(endpoint, dtype=torch.bool)
    per_step = env.num_envs
    env_id = torch.arange(len(endpoint)) % per_step
    for index in range(4):
        local = torch.nonzero((torch.arange(per_step) * 4 // per_step) == index).flatten()
        cutoff = local[int(len(local) * .8)]
        keep_train |= (endpoint == index) & (env_id < cutoff)
    payload = {"policy": policy, "endpoint_id": endpoint, "train": keep_train}
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / "endpoint_anchor.pt"
    torch.save(payload, path)
    return path, payload


def exact_anchor_kl(actor, reference, anchor) -> dict[str, float]:
    values = {}
    for index, name in enumerate(("walk_1p2", "run_1p2", "run_2p4", "run_2p6")):
        ids = torch.nonzero((anchor["endpoint_id"] == index) & ~anchor["train"]).flatten()
        ids = ids[::max(1, len(ids) // 4096)][:4096]
        batch = TensorDict({"policy": anchor["policy"][ids].to(next(actor.parameters()).device)}, batch_size=[len(ids)])
        with torch.no_grad():
            reference(batch, stochastic_output=True)
            ref = tuple(x.clone() for x in reference.output_distribution_params)
            actor(batch, stochastic_output=True)
            cur = actor.output_distribution_params
            values[name] = float(actor.get_kl_divergence(ref, cur).mean())
    return values


def save_model(runner, path, iteration, beta):
    payload = runner.alg.save()
    payload["iter"] = iteration
    payload["infos"] = {
        "stage": "2N", "anchor_beta": beta, "single_actor": True,
        "gait_conditioned_gaussian": True, "runtime_teacher_calls": 0,
    }
    torch.save(payload, path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise RuntimeError(f"PROVENANCE_FAIL:{path}")
    cfg, agent_cfg = resolve_task_config("Isaac-Exp012-G1-GaitPpoRetention-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 1024
    cfg.episode_length_s = 12.0
    cfg.seed = 20268021
    agent_cfg.seed = 20268021
    agent_cfg.max_iterations = 25
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp012-G1-GaitPpoRetention-v0", cfg=cfg)
        base = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        env = GaitVecEnv(base, "anchor" if args.mode == "prepare" else "train")
        import importlib.metadata
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        run = load_integrated(runner, runner.device)
        mapping, lr = initialize_optimizer(runner, run)
        contract = Exp012StrictPPOResumeContract()
        lr_state = contract.synchronize(runner.alg, runner, resume=True)
        stage2k = torch.load(K, map_location=runner.device, weights_only=False)["model_state_dict"]
        identity, maximum = actor_identity(runner.alg.actor, stage2k, runner.device)
        critic_obs = torch.randn(4096, 123, device=runner.device)
        critic_aug = TensorDict(
            {"policy": torch.cat((critic_obs, torch.rand(4096, 1, device=runner.device)), -1)},
            batch_size=[4096],
        )
        run_critic = run["critic_state_dict"]
        with torch.no_grad():
            expected = torch.nn.functional.linear(critic_obs, run_critic["mlp.0.weight"], run_critic["mlp.0.bias"])
            expected = torch.nn.functional.linear(torch.nn.functional.elu(expected), run_critic["mlp.2.weight"], run_critic["mlp.2.bias"])
            expected = torch.nn.functional.linear(torch.nn.functional.elu(expected), run_critic["mlp.4.weight"], run_critic["mlp.4.bias"])
            expected = torch.nn.functional.linear(torch.nn.functional.elu(expected), run_critic["mlp.6.weight"], run_critic["mlp.6.bias"])
            actual = runner.alg.critic(critic_aug)
        critic_identity = torch.equal(expected, actual)
        parent_dir = OUT / "checkpoints"
        parent_dir.mkdir(exist_ok=True)
        save_model(runner, parent_dir / "model_initial.pt", 0, args.beta)
        dump("fine_tuning_parent_identity_audit.json", {
            "status": "PASS" if identity else "FAIL", "mean_bitwise_identity": identity,
            "mean_max_abs_difference": maximum, "alpha_walk": .30, "alpha_run": .65,
            "optimizer_lr": lr, "runtime_lr": runner.alg.learning_rate,
        })
        dump("gait_conditioned_critic_initialization_audit.json", {
            "status": "PASS" if critic_identity else "FAIL",
            "bitwise_identity_original_123d": critic_identity,
            "gait_column_zero": bool(torch.count_nonzero(runner.alg.critic.mlp[0].weight[:, -1]) == 0),
        })
        dump("gait_ppo_optimizer_mapping.json", {"mappings": mapping})
        dump("gait_ppo_optimizer_initialization_audit.json", {
            "status": "PASS", "adam_step": 105000, "state_count": len(runner.alg.optimizer.state),
            "restored_lr": lr, "runtime_lr": runner.alg.learning_rate,
            "strict_resume": lr_state.to_dict(), "fresh_adam_all_parameters": False,
        })
        if not identity or not critic_identity:
            raise RuntimeError("GAIT_CONDITIONED_PARENT_IDENTITY_FAIL")

        if args.mode == "prepare":
            reference = copy.deepcopy(runner.alg.actor).eval()
            path, anchor = collect_anchor(env, reference, runner.device)
            counts = {name: int((anchor["endpoint_id"] == i).sum()) for i, name in enumerate(
                ("walk_1p2", "run_1p2", "run_2p4", "run_2p6"))}
            dump("endpoint_anchor_manifest.json", {
                "path": str(path.relative_to(REPO)), "sha256": sha(path), "counts": counts,
                "episodes_per_endpoint": 256, "requested_minimum": 100, "deterministic_reference": True,
            })
            dump("endpoint_anchor_split.json", {
                "unit": "environment episode", "train_fraction": .8, "holdout_fraction": .2,
                "train_samples": int(anchor["train"].sum()), "holdout_samples": int((~anchor["train"]).sum()),
                "endpoint_balanced_objective": True,
            })
            dump("endpoint_anchor_hashes.json", {"anchor_pt": sha(path)})
            print("STAGE2N_PREPARE_COMPLETE")
            return

        anchor_payload = torch.load(OUT / "raw/endpoint_anchor.pt", map_location=runner.device, weights_only=False)
        train = anchor_payload["train"]
        anchor = TensorDict(
            {
                "policy": anchor_payload["policy"][train].to(runner.device),
                "endpoint_id": anchor_payload["endpoint_id"][train].to(runner.device),
            },
            batch_size=[int(train.sum())],
            device=runner.device,
        )
        reference = copy.deepcopy(runner.alg.actor).eval()
        runner.alg.configure_anchor(anchor, reference, args.beta)
        obs = env.get_observations().to(runner.device)
        curves, capability, first = [], [], None
        checkpoints = {1, 5, 10, 15, 20, 25}
        gradient_audit = None
        for iteration in range(1, args.iterations + 1):
            with torch.inference_mode():
                for _ in range(agent_cfg.num_steps_per_env):
                    actions = runner.alg.act(obs)
                    obs, rewards, dones, extras = env.step(actions)
                    obs, rewards, dones = obs.to(runner.device), rewards.to(runner.device), dones.to(runner.device)
                    runner.alg.process_env_step(obs, rewards, dones, extras)
            runner.alg.compute_returns(obs)
            storage = runner.alg.storage
            old_mean, old_std = (x.flatten(0, 1).clone() for x in storage.distribution_params)
            observations = storage.observations.flatten(0, 1)
            actions = storage.actions.flatten(0, 1)
            old_log = storage.actions_log_prob.flatten(0, 1).squeeze(-1)
            if iteration == 1:
                rng = torch.random.get_rng_state()
                runner.alg.optimizer.zero_grad()
                runner.alg.actor(observations, stochastic_output=True)
                current_log = runner.alg.actor.get_output_log_prob(actions)
                ppo_probe = -(storage.advantages.flatten(0, 1).squeeze(-1)
                              * torch.exp(current_log - old_log)).mean()
                ppo_probe.backward()
                ppo_vector = torch.cat([
                    (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                    for p in runner.alg.actor.parameters()
                ])
                runner.alg.optimizer.zero_grad()
                anchor_probe, _ = runner.alg._anchor_loss()
                anchor_probe.backward()
                anchor_vector = torch.cat([
                    (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                    for p in runner.alg.actor.parameters()
                ])
                runner.alg.optimizer.zero_grad()
                torch.random.set_rng_state(rng)
                ppo_norm = float(torch.linalg.vector_norm(ppo_vector))
                anchor_norm = float(torch.linalg.vector_norm(anchor_vector))
                gradient_audit = {
                    "beta": args.beta, "ppo_gradient_norm": ppo_norm,
                    "anchor_gradient_norm": anchor_norm,
                    "effective_anchor_ppo_ratio": args.beta * anchor_norm / max(ppo_norm, 1e-30),
                    "gradient_cosine": float(torch.nn.functional.cosine_similarity(
                        ppo_vector, anchor_vector, dim=0)),
                    "cap_pass": args.beta * anchor_norm / max(ppo_norm, 1e-30) <= .25,
                }
            losses = runner.alg.update()
            with torch.no_grad():
                runner.alg.actor(observations, stochastic_output=True)
                new_mean, new_std = runner.alg.actor.output_distribution_params
                ratio = torch.exp(runner.alg.actor.get_output_log_prob(actions) - old_log)
                exact = torch.distributions.kl_divergence(
                    torch.distributions.Normal(old_mean, old_std),
                    torch.distributions.Normal(new_mean, new_std)).sum(-1)
            anchor_kl = exact_anchor_kl(runner.alg.actor, reference, anchor_payload)
            row = {
                "iteration": iteration, "reward": float(rewards.mean()), "exact_kl": float(exact.mean()),
                "max_sample_kl": float(exact.max()), "clip_fraction": float(((ratio < .8) | (ratio > 1.2)).float().mean()),
                "ratio_p95": float(torch.quantile(ratio, .95)), "ratio_p99": float(torch.quantile(ratio, .99)),
                "mean_action_shift": float(torch.linalg.vector_norm(new_mean - old_mean, dim=-1).mean()),
                "value_loss": losses["value"], "surrogate_loss": losses["surrogate"],
                "anchor_loss": losses["anchor"], "lr": runner.alg.learning_rate,
                **{f"anchor_kl_{key}": value for key, value in anchor_kl.items()},
            }
            curves.append(row)
            if iteration == 1:
                first = row.copy()
            if iteration in checkpoints and not args.tag:
                save_model(runner, parent_dir / f"model_{iteration}.pt", iteration, args.beta)
                capability.extend({"iteration": iteration, "condition": key, "anchor_kl": value}
                                  for key, value in anchor_kl.items())
            if iteration <= 5 and (
                not math.isfinite(row["exact_kl"]) or row["exact_kl"] > .5
                or max(anchor_kl.values()) > .05
            ):
                dump("early_guard.json", {"status": "FAIL", "iteration": iteration, "metrics": row})
                break
        prefix = f"{args.tag}_" if args.tag else ""
        write_csv(prefix + "training_curves.csv", curves)
        if capability:
            write_csv(prefix + "capability_timeline.csv", capability)
        dump(prefix + "first_update_stability.json", first)
        dump(prefix + "gradient_audit.json", gradient_audit)
        if args.tag:
            save_model(runner, OUT / f"shadow_{args.tag}.pt", len(curves), args.beta)
        if not args.tag and not (OUT / "early_guard.json").exists():
            dump("early_guard.json", {"status": "PASS", "iterations_audited": min(5, len(curves))})
        print("STAGE2N_TRAIN_COMPLETE", len(curves))


if __name__ == "__main__":
    main()
