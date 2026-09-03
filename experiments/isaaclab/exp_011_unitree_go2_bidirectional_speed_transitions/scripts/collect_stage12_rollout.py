"""Collect the fixed Stage 12 directionality rollout without optimizer updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage12_tangential_slip_reward_directionality"
RAW = OUT / "raw"
CHECKPOINT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage11_tangential_slip_reduction/checkpoints/model_initial.pt"
WEIGHT = 0.005591959944980788
SEED = 20272901
DT = 0.02

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--family", choices=("steady", "transition", "all"), default="all")
parser.add_argument("--condition-index", type=int)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app
sys.argv = [sys.argv[0]]

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: E402,F401
import go2_bidirectional.stage11_tasks  # noqa: E402,F401
from go2_bidirectional.stage11_tasks.command import wrap_angle, yaw_xyzw  # noqa: E402
from go2_bidirectional.stage6_endpoint_protocol import quat_xyzw_to_gravity_tilt_torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402

STEADY = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0)
TRANSITIONS = (
    (0.0, 0.2), (0.0, 0.4), (0.0, 0.6),
    (0.6, 0.4), (0.6, 0.2), (0.6, 0.0),
    (0.0, 1.2), (1.2, 2.0), (2.0, 1.2), (1.2, 0.0),
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_label(family: str, source: float, target: float) -> str:
    def speed(value):
        return str(value).replace(".", "p")
    return f"{family}_{speed(source)}_{speed(target)}.pt"


def stack(values):
    return torch.stack(values).contiguous()


num_envs = 100 if args.family in ("steady", "all") else 50
cfg, agent_cfg = resolve_task_config(
    "Isaac-Exp011-Go2-Tangential-Slip-v0", "rsl_rl_cfg_entry_point"
)
cfg.scene.num_envs = num_envs
cfg.seed = SEED
cfg.episode_length_s = 60.0
cfg.observations.policy.enable_corruption = False
cfg.events.base_external_force_torque = None
cfg.events.push_robot = None
cfg.rewards.go2_contact_tangential_slip.weight = WEIGHT
agent_cfg.seed = SEED
if args.device:
    cfg.sim.device = args.device
    agent_cfg.device = args.device
raw = gym.make("Isaac-Exp011-Go2-Tangential-Slip-v0", cfg=cfg)
wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
agent_cfg = handle_deprecated_rsl_rl_cfg(
    agent_cfg, __import__("importlib.metadata").metadata.version("rsl-rl-lib")
)
runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(
    str(CHECKPOINT),
    load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": True, "rnd": False},
    strict=True,
    map_location=runner.device,
)
runner.alg.actor.eval()
runner.alg.critic.eval()
env = wrapped.unwrapped
robot = env.scene["robot"]
manager = env.reward_manager
reward_names = list(manager.active_terms)
slip_term = manager.get_term_cfg("go2_contact_tangential_slip").func
command = env.command_manager.get_term("base_velocity")
all_ids = torch.arange(num_envs, device=env.device)
manifest = []


def configure(source: float, target: float, time_s: float) -> None:
    command.source_speed.fill_(source)
    command.target_speed.fill_(target)
    command.source_hold_s.fill_(3.0 if source != target else 0.0)
    command.elapsed_s.fill_(time_s - DT)
    command._update_command()


def collect_condition(family: str, source: float, target: float, episodes: int) -> dict:
    if episodes != num_envs:
        raise ValueError("one environment per registered episode is required")
    env.seed(SEED)
    wrapped.reset()
    command._resample_command(all_ids)
    slip_term.reset(all_ids)
    duration = 8.0 if family == "steady" else 9.5
    steps = round(duration / DT)
    alive = torch.ones(num_envs, dtype=torch.bool, device=env.device)
    previous_action = torch.zeros(num_envs, 12, device=env.device)
    fields = {
        key: [] for key in (
            "observation", "sampled_action", "mean_action", "old_log_prob", "value",
            "reward_terms", "total_reward", "base_reward", "raw_slip_score",
            "weighted_slip_reward", "actual_speed", "heading_error", "phase_gate",
            "foot_contact", "contact_age", "normal_force", "tangential_speed",
            "friction_utilization", "gravity_tilt", "action_rate", "saturation",
            "flight", "fall", "termination", "timeout", "valid",
        )
    }
    for step in range(steps):
        time_s = step * DT
        configure(source, target, time_s)
        observations = wrapped.get_observations()
        with torch.inference_mode():
            sampled_action = runner.alg.actor(observations, stochastic_output=True)
            mean_action, _ = runner.alg.actor.output_distribution_params
            old_log_prob = runner.alg.actor.get_output_log_prob(sampled_action)
            value = runner.alg.critic(observations).squeeze(-1)
            _, total_reward, dones, extras = wrapped.step(sampled_action)
        friction = slip_term.diagnostic_friction_utilization()
        actual_speed = robot.data.root_lin_vel_b.torch[:, 0]
        yaw = yaw_xyzw(robot.data.root_quat_w.torch)
        heading_error = wrap_angle(command.heading_reference - yaw)
        gravity_tilt = quat_xyzw_to_gravity_tilt_torch(robot.data.root_quat_w.torch)
        velocity_ratio = (
            robot.data.joint_vel.torch.abs()
            / robot.data.joint_vel_limits.torch.abs().clamp_min(1.0e-6)
        ).amax(1)
        torque_ratio = (
            robot.data.applied_torque.torch.abs()
            / robot.data.joint_effort_limits.torch.abs().clamp_min(1.0e-6)
        ).amax(1)
        saturation = (velocity_ratio >= 0.95) | (torque_ratio >= 0.95)
        timeout = extras.get("time_outs", torch.zeros_like(dones)).bool()
        fall = dones.bool() & ~timeout
        step_terms = manager._step_reward * DT
        weighted_slip = -WEIGHT * slip_term.last_raw_score * DT
        base_reward = total_reward - weighted_slip
        contact = slip_term.last_normal_force > 5.0
        fields["observation"].append(observations.detach().cpu())
        fields["sampled_action"].append(sampled_action.detach().cpu())
        fields["mean_action"].append(mean_action.detach().cpu())
        fields["old_log_prob"].append(old_log_prob.detach().cpu())
        fields["value"].append(value.detach().cpu())
        fields["reward_terms"].append(step_terms.detach().cpu())
        fields["total_reward"].append(total_reward.detach().cpu())
        fields["base_reward"].append(base_reward.detach().cpu())
        fields["raw_slip_score"].append(slip_term.last_raw_score.detach().cpu())
        fields["weighted_slip_reward"].append(weighted_slip.detach().cpu())
        fields["actual_speed"].append(actual_speed.detach().cpu())
        fields["heading_error"].append(heading_error.detach().cpu())
        fields["phase_gate"].append(command.heading_gate.detach().cpu())
        fields["foot_contact"].append(contact.detach().cpu())
        fields["contact_age"].append(slip_term.contact_age.detach().cpu())
        fields["normal_force"].append(slip_term.last_normal_force.detach().cpu())
        fields["tangential_speed"].append(slip_term.last_foot_speed.detach().cpu())
        fields["friction_utilization"].append(friction.detach().cpu())
        fields["gravity_tilt"].append(gravity_tilt.detach().cpu())
        fields["action_rate"].append((sampled_action - previous_action).norm(dim=1).detach().cpu())
        fields["saturation"].append(saturation.detach().cpu())
        fields["flight"].append((contact.sum(1) == 0).detach().cpu())
        fields["fall"].append(fall.detach().cpu())
        fields["termination"].append(dones.bool().detach().cpu())
        fields["timeout"].append(timeout.detach().cpu())
        fields["valid"].append(alive.detach().cpu())
        alive &= ~dones.bool()
        previous_action = sampled_action.detach()
    configure(source, target, duration)
    final_obs = wrapped.get_observations()
    with torch.inference_mode():
        final_value = runner.alg.critic(final_obs).squeeze(-1).cpu()
    payload = {key: stack(value) for key, value in fields.items()}
    payload.update({
        "family": family,
        "source_speed": source,
        "target_speed": target,
        "episode_seeds": torch.arange(SEED, SEED + episodes),
        "reward_names": reward_names,
        "dt": DT,
        "gamma": float(runner.alg.gamma),
        "lam": float(runner.alg.lam),
        "final_value": final_value,
        "checkpoint": str(CHECKPOINT),
    })
    path = RAW / file_label(family, source, target)
    torch.save(payload, path)
    result = {
        "family": family, "source_speed": source, "target_speed": target,
        "episodes": episodes, "steps": steps, "samples": episodes * steps,
        "path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size,
        "valid_samples": int(payload["valid"].sum()),
        "falls": int(payload["fall"].sum()),
        "nan_inf": sum(
            int((~torch.isfinite(value)).sum())
            for value in payload.values() if torch.is_tensor(value) and value.is_floating_point()
        ),
    }
    print(
        f"STAGE12 {family} {source:g}->{target:g} samples={result['samples']} "
        f"valid={result['valid_samples']} falls={result['falls']}",
        flush=True,
    )
    return result


try:
    if args.family in ("steady", "all"):
        selected_steady = (
            [STEADY[args.condition_index]] if args.condition_index is not None else STEADY
        )
        for speed in selected_steady:
            manifest.append(collect_condition("steady", speed, speed, 100))
    if args.family in ("transition", "all"):
        if num_envs != 50:
            raise RuntimeError("run --family transition separately so num_envs=50")
        selected_transitions = (
            [TRANSITIONS[args.condition_index]] if args.condition_index is not None else TRANSITIONS
        )
        for source, target in selected_transitions:
            manifest.append(collect_condition("transition", source, target, 50))
finally:
    wrapped.close()
    simulation_app.close()

suffix = f"_{args.condition_index}" if args.condition_index is not None else ""
manifest_path = OUT / f"directionality_rollout_manifest_{args.family}{suffix}.json"
manifest_path.write_text(json.dumps({
    "seed_root": SEED,
    "checkpoint_sha256": sha(CHECKPOINT),
    "reward_names": reward_names,
    "files": manifest,
    "production_ppo_update": 0,
    "reward_optimization": 0,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
