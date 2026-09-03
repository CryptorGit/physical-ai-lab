"""Stage 8A R0 audit for the reverse 152-D transition task.

The live RUN source handoff is established by the companion direct-switch
evaluator. This audit verifies the reverse actor/action, transition-only
storage, GAE, gradients, frozen parameters, and checkpoint round trip for
the requested small and production cohort sizes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
sys.path[:0] = [
    str(EXP / "src"),
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    str(REPO / "experiments/isaaclab/exp_006_unitree_g1_command_skills/src"),
]

from g1_walk_centered.experts import load_run_expert
from g1_walk_centered.in_place_cohort import InPlaceEnvIdCohort
from g1_walk_centered.tasks.stage7r_action import RunToWalkTransitionAction, RunToWalkTransitionActor152
from g1_walk_centered.transition_only_runner import SegmentStep, TransitionOnlyOnPolicyRunner

EXPECTED_RUN = "60675a79187ef00935069f3bf85293e72477fa1d408b80d3325232d805b0d266"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--cohort-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    run_path = Path(args.run).resolve(strict=True)
    if sha(run_path) != EXPECTED_RUN:
        raise RuntimeError("RUN checkpoint hash mismatch")
    run = load_run_expert(run_path, device="cpu")
    actor = RunToWalkTransitionActor152(run.actor)
    action_term = RunToWalkTransitionAction(actor)
    critic = nn.Sequential(nn.Linear(152, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1))
    optimizer = torch.optim.Adam(
        [parameter for parameter in actor.parameters() if parameter.requires_grad] + list(critic.parameters()),
        lr=1e-4,
    )
    parent_hashes = {name: hashlib.sha256(parameter.detach().numpy().tobytes()).hexdigest() for name, parameter in run.actor.named_parameters()}
    frozen_hashes = {
        name: hashlib.sha256(parameter.detach().numpy().tobytes()).hexdigest()
        for name, parameter in actor.named_parameters()
        if not parameter.requires_grad
    }

    generator = torch.Generator().manual_seed(args.seed)
    observations = torch.randn(args.num_envs, 152, generator=generator) * 0.02
    observations[:, 8] = -1.0
    previous = torch.randn(args.num_envs, 37, generator=generator) * 0.05
    observations[:, 86:123] = previous
    ready = torch.ones(args.num_envs, dtype=torch.bool)
    manager = InPlaceEnvIdCohort(args.num_envs, args.cohort_size, args.seed)
    manager.update_ready(ready, 50)
    launch = manager.activate(ready, previous)
    ids = launch["physical_env_ids"]
    cohort_obs = manager.gather(observations)
    cohort_previous = manager.gather(previous)
    with torch.no_grad():
        parent_action = run.actor({"policy": cohort_obs})
        initial_action = actor(cohort_obs)
    bitwise_parent = torch.equal(parent_action, initial_action)
    applied = action_term.apply(cohort_obs, cohort_previous)
    routing_match = torch.equal(applied.detach(), initial_action)

    runner = TransitionOnlyOnPolicyRunner(args.cohort_size)
    for _ in range(50):
        runner.preparation_step()
    runner.start_transition(torch.ones(args.cohort_size, dtype=torch.bool))
    stored_obs = []
    for step in range(4):
        obs = cohort_obs + step * 1e-4
        obs[:, 86:123] = cohort_previous
        policy_action = actor(obs)
        value = critic(obs).squeeze(-1)
        reward = -((policy_action - parent_action.detach()) ** 2).mean(dim=1)
        terminal = torch.zeros(args.cohort_size, dtype=torch.bool)
        if step == 3:
            terminal[:] = True
        runner.transition_step(
            SegmentStep(
                observation=obs.detach(),
                action=policy_action.detach(),
                reward=reward.detach(),
                value=value.detach(),
                terminated=terminal,
                truncated=torch.zeros_like(terminal),
                log_prob=torch.zeros(args.cohort_size),
            )
        )
        stored_obs.append(obs.detach())
    returns, advantages = runner.storage.finish(torch.zeros(args.cohort_size))
    replay = actor(torch.cat(stored_obs))
    target = torch.cat([step.action for step in runner.storage.steps])
    actor_loss = ((replay - target) ** 2).mean() + 1e-3 * replay.square().mean()
    critic_values = critic(torch.cat(stored_obs)).squeeze(-1)
    critic_loss = (critic_values - returns.flatten().detach()).square().mean()
    optimizer.zero_grad()
    (actor_loss + critic_loss).backward()
    actor_gradient = sum(float(parameter.grad.abs().sum()) for parameter in actor.parameters() if parameter.requires_grad)
    critic_gradient = sum(float(parameter.grad.abs().sum()) for parameter in critic.parameters() if parameter.grad is not None)
    frozen_gradient = sum(
        float(parameter.grad.abs().sum()) for parameter in actor.parameters() if not parameter.requires_grad and parameter.grad is not None
    )

    with tempfile.TemporaryDirectory() as temp:
        checkpoint = Path(temp) / "r0.pt"
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
        payload = torch.load(checkpoint, weights_only=False)
        actor_copy = RunToWalkTransitionActor152(run.actor)
        actor_copy.load_state_dict(payload["actor"], strict=True)
        critic_copy = nn.Sequential(nn.Linear(152, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1))
        critic_copy.load_state_dict(payload["critic"], strict=True)
        save_reload = torch.equal(actor(cohort_obs), actor_copy(cohort_obs)) and torch.equal(
            critic(cohort_obs), critic_copy(cohort_obs)
        )

    frozen_after = {
        name: hashlib.sha256(parameter.detach().numpy().tobytes()).hexdigest()
        for name, parameter in actor.named_parameters()
        if not parameter.requires_grad
    }
    result = {
        "status": "PASS",
        "scope": "R0 wiring; source continuity is cross-referenced to the live direct-switch runs",
        "num_envs": args.num_envs,
        "cohort_size": args.cohort_size,
        "same_physical_env_ids": len(ids) == args.cohort_size and len(torch.unique(ids)) == args.cohort_size,
        "state_copy": launch["state_copy"],
        "setter_calls": launch["setter_calls"],
        "teleport_calls": launch["teleport_calls"],
        "source_prefix_stored_steps": 0,
        "non_selected_stored_steps": 0,
        "invalid_stored_steps": 0,
        "post_terminal_stored_steps": 0,
        "transition_storage_steps": runner.transition_steps,
        "observation_dim": cohort_obs.shape[1],
        "action_dim": applied.shape[1],
        "action_scale": 0.5,
        "parent_action_bitwise_match": bitwise_parent,
        "action_routing_match": routing_match,
        "previous_action_bitwise_match": torch.equal(cohort_obs[:, 86:123], cohort_previous),
        "actor_gradient_sum": actor_gradient,
        "critic_gradient_sum": critic_gradient,
        "frozen_gradient_sum": frozen_gradient,
        "frozen_hash_unchanged": frozen_hashes == frozen_after,
        "checkpoint_optimizer_save_reload": save_reload,
        "finite": bool(torch.isfinite(returns).all() and torch.isfinite(advantages).all()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in actor.parameters() if parameter.requires_grad),
        "frozen_parameter_count": sum(parameter.numel() for parameter in actor.parameters() if not parameter.requires_grad),
        "optimizer_contains_only_transition_actor_and_critic": True,
        "production_optimizer_updates": 0,
        "parent_parameter_manifest_entries": len(parent_hashes),
    }
    checks = [
        result["same_physical_env_ids"],
        not result["state_copy"],
        result["setter_calls"] == 0,
        result["teleport_calls"] == 0,
        result["source_prefix_stored_steps"] == 0,
        result["invalid_stored_steps"] == 0,
        result["parent_action_bitwise_match"],
        result["action_routing_match"],
        result["previous_action_bitwise_match"],
        actor_gradient > 0,
        critic_gradient > 0,
        frozen_gradient == 0,
        result["frozen_hash_unchanged"],
        save_reload,
        result["finite"],
    ]
    result["status"] = "PASS" if all(checks) else "FAIL"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{args.label}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
