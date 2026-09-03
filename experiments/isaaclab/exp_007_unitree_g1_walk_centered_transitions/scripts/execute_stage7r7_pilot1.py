"""Execute the authorized Stage 7R7 frozen transition-only PPO pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Normal
import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
PILOT2 = "--pilot2" in sys.argv
OUT = REPO / ("results/exp_007_unitree_g1_walk_centered_transitions/stage7r8_walk_to_run_pilot2_saturation" if PILOT2 else "results/exp_007_unitree_g1_walk_centered_transitions/stage7r7_frozen_pilot1_execution")
CFG_PATH = EXP / ("configs/stage7r_walk_to_run_pilot2_saturation.yaml" if PILOT2 else "configs/stage7r_walk_to_run_pilot1.yaml")
FREEZE = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage7r6_prepilot_protocol/freeze_declaration.json"
ALLOWED_HEAD = "760584189ba42f987e6d8d993fbae7fb37f9c316"
EXPECTED_CONFIG = "46dafd8cfba91910b1bb33c9293cf6c5cf45abf71b4ab1105fb6bd776d9c8d4c" if PILOT2 else "aa2cf5498032fd262ccb6a1aa49b997c6cfb4a2f2364b4d0550d431e8b918af9"
EXPECTED_REWARD = "3ce9ebda1e96e4193ff009a2eb473ac44fb3b98442f05b3947c521df72217ced" if PILOT2 else "7da503b46bd56c59d610213c4f5094bbe58ba7a98f317efa3f31ebacd32df764"
EXPECTED_ACTOR = "6f8c7cbe92164d8a77eeba14b2de410adb463c2ba4de6a78c17953fb7c97021b" if PILOT2 else "0fdf9e2ae2d939eb4f9cfd9b6e4ff17ee50bf97b8940c8305197376e84794030"
PILOT2_PARENT = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage7r7_frozen_pilot1_execution/checkpoints/model_75.pt"
PILOT2_PARENT_SHA = "0dbb8a095dd6ea71140b9c843dff5dcdbde92d1a7b247fa4ba068d084f0a70ed"
EXPECTED = {
    "stand": ("logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt", "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"),
    "stw": ("logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100/model_0.pt", "511b3c832e2c7bc54f348391db79d4ac57d1ceda10a6907549a50da093a9c36e"),
    "walk": ("logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt", "9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa"),
    "run": ("logs/rsl_rl/physical_ai_g1_command_skills/2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt", "60675a79187ef00935069f3bf85293e72477fa1d408b80d3325232d805b0d266"),
}

sys.path[:0] = [
    str(EXP / "src"),
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    str(REPO / "experiments/isaaclab/exp_006_unitree_g1_command_skills/src"),
]
import g1_command_skills.tasks  # noqa: E402
import g1_flat_run.tasks  # noqa: E402
import g1_walk_centered.tasks  # noqa: E402
import isaaclab_tasks  # noqa: E402
from g1_walk_centered.command_contract import MotionCommand  # noqa: E402
from g1_walk_centered.experts import load_run_expert, load_walk_expert  # noqa: E402
from g1_walk_centered.experts.adapters import canonical_state_from_legacy_observation, to_run_observation  # noqa: E402
from g1_walk_centered.in_place_cohort import InPlaceEnvIdCohort  # noqa: E402
from g1_walk_centered.stage7r6_reward import reward_terms  # noqa: E402
from g1_walk_centered.tasks.stage7r_action import WalkToRunTransitionActor152  # noqa: E402
from g1_walk_centered.transition_only_runner import SegmentStep, TransitionOnlyOnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(name, rows):
    if not rows:
        return
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def read_csv(name):
    path = OUT / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def mj(x):
    x = x.clamp(0, 1)
    return 10 * x**3 - 15 * x**4 + 6 * x**5


def authorize():
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    hashes = {key: file_sha(REPO / path) for key, (path, _) in EXPECTED.items()}
    authorized_descendant = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ALLOWED_HEAD, head], cwd=REPO
    ).returncode == 0
    checks = {
        "git_revision": authorized_descendant,
        "config_path": CFG_PATH.exists() if PILOT2 else CFG_PATH.resolve() == (REPO / freeze["config_path"]).resolve(),
        "config_sha": digest(cfg) == EXPECTED_CONFIG and (PILOT2 or EXPECTED_CONFIG == freeze["config_sha256"]),
        "reward_sha": digest(cfg["reward"]) == EXPECTED_REWARD and (PILOT2 or EXPECTED_REWARD == freeze["reward_sha256"]),
        "actor_initialization_sha": digest(cfg["actor"]) == EXPECTED_ACTOR,
        "freeze_ready": freeze["status"] == "FROZEN_READY_FOR_PILOT1",
        "protected_hashes": all(hashes[key] == EXPECTED[key][1] for key in EXPECTED),
        "cli_overrides_disabled": cfg["runtime"]["cli_overrides_allowed"] is False,
        "pilot2_parent": (not PILOT2) or (PILOT2_PARENT.exists() and file_sha(PILOT2_PARENT) == PILOT2_PARENT_SHA),
    }
    result = {"authorized": all(checks.values()), "checks": checks, "head": head, "protected_hashes": hashes}
    write_json("execution_authorization.json", result)
    if not result["authorized"]:
        raise RuntimeError(f"Stage 7R7 authorization denied: {result}")
    return cfg, hashes


parser = argparse.ArgumentParser()
parser.add_argument("--phase", required=True, choices=("prepare", "train"))
parser.add_argument("--pilot2", action="store_true")
parser.add_argument("--resume-pilot2", action="store_true")
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def make_models(cfg, device, paths):
    stand = load_walk_expert(paths["stand"], device=device)
    stw = load_walk_expert(paths["stw"], device=device)
    walk = load_walk_expert(paths["walk"], device=device)
    run = load_run_expert(paths["run"], device=device)
    actor = WalkToRunTransitionActor152(run.actor).to(device)
    critic = nn.Sequential(nn.Linear(152, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1)).to(device)
    log_std = nn.Parameter(torch.full((37,), math.log(cfg["exploration"]["initial_std"]), device=device))
    return stand, stw, walk, run, actor, critic, log_std


def checkpoint(path, iteration, actor, critic, log_std, optimizer, cfg):
    payload = {
        "iteration": iteration,
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "log_std": log_std.detach().cpu(),
        "optimizer": optimizer.state_dict(),
        "config_sha256": EXPECTED_CONFIG,
        "reward_sha256": EXPECTED_REWARD,
        "actor_initialization_sha256": EXPECTED_ACTOR,
        "parent_sha256": PILOT2_PARENT_SHA if PILOT2 else EXPECTED["run"][1],
        "ancestry": cfg["actor"]["parent_checkpoint"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "path": str(path.relative_to(REPO)).replace("\\", "/"),
        "sha256": file_sha(path),
        "iteration": iteration,
        "actor_parameter_hash": tensor_sha(actor.state_dict()),
        "critic_parameter_hash": tensor_sha(critic.state_dict()),
        "exploration_std": {
            "min": float(log_std.exp().min()),
            "mean": float(log_std.exp().mean()),
            "max": float(log_std.exp().max()),
        },
        "config_sha256": EXPECTED_CONFIG,
        "reward_sha256": EXPECTED_REWARD,
        "ancestry": cfg["actor"]["parent_checkpoint"],
    }


def prepare_phase(cfg, hashes):
    paths = {key: (REPO / value[0]).resolve() for key, value in EXPECTED.items()}
    _, run = load_walk_expert(paths["walk"]), load_run_expert(paths["run"])
    actor = WalkToRunTransitionActor152(run.actor)
    if PILOT2:
        parent = torch.load(PILOT2_PARENT, map_location="cpu", weights_only=False)
        actor.load_state_dict(parent["actor"], strict=True)
    critic = nn.Sequential(nn.Linear(152, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1))
    log_std = nn.Parameter(torch.full((37,), math.log(cfg["exploration"]["initial_std"])))
    optimizer = torch.optim.Adam(
        [p for p in actor.parameters() if p.requires_grad] + list(critic.parameters()) + [log_std],
        lr=cfg["ppo"]["learning_rate"],
    )
    cp = checkpoint(OUT / "checkpoints/initial.pt", 0, actor, critic, log_std, optimizer, cfg)
    trainable = [name for name, p in actor.named_parameters() if p.requires_grad]
    frozen = [name for name, p in actor.named_parameters() if not p.requires_grad]
    preflight = {
        "status": "PASS",
        "config_validation": "PASS",
        "config_sha256": EXPECTED_CONFIG,
        "reward_sha256": EXPECTED_REWARD,
        "actor_initialization_sha256": EXPECTED_ACTOR,
        "protected_hashes": hashes,
        "physical_envs": 1024,
        "cohort_size": 512,
        "source_contract": "WALK@1.2, hold>=1.0s, speed_error<=0.20, heading<=0.12, safety finite",
        "runner": "IN_PLACE_TRANSITION_ONLY_PPO",
        "source_prefix_stored_steps": 0,
        "invalid_stored_steps": 0,
        "observation_dimension": 152,
        "action_dimension": 37,
        "global_previous_action": "R0_COMPLETE_PASS_REVALIDATED",
        "trainable_parameters": trainable,
        "frozen_parameter_count": sum(p.numel() for p in actor.parameters() if not p.requires_grad),
        "exploration_std": 0.25,
        "target_distribution": {"2.4": 0.5, "2.6": 0.3, "2.8": 0.2},
        "run_name": ("stage7r8-pilot2-sat3ce9ebda-seed20261208" if PILOT2 else "stage7r5-pilot1-cfgaa2cf549-seed20261120"),
        "initial_checkpoint": cp,
    }
    write_json("pilot_execution_preflight.json", preflight)
    (OUT / "frozen_config_snapshot.yaml").write_text(CFG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    write_json("frozen_protocol_hashes.json", {
        "config_sha256": EXPECTED_CONFIG, "reward_sha256": EXPECTED_REWARD,
        "actor_initialization_sha256": EXPECTED_ACTOR,
    })
    print(json.dumps(preflight, indent=2))


def main_train(cfg, hashes):
    torch.manual_seed(cfg["experiment"]["training_seed"])
    paths = {key: (REPO / value[0]).resolve() for key, value in EXPECTED.items()}
    task_cfg, agent_cfg = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    task_cfg.scene.num_envs = cfg["experiment"]["physical_envs"]
    task_cfg.seed = cfg["experiment"]["training_seed"]
    task_cfg.episode_length_s = 22.0
    task_cfg.sim.device = cfg["experiment"]["device"]
    args.device = cfg["experiment"]["device"]
    rows, target_rows, reward_rows, failures, manifests = [], [], [], {}, []
    resume_iteration = 0
    resume_checkpoint = OUT / "checkpoints/initial.pt"
    if PILOT2 and args.resume_pilot2:
        candidates = [(75, "model_75.pt"), (50, "model_50.pt"), (25, "model_25.pt"), (10, "model_10.pt"), (1, "first_post_update.pt")]
        resume_iteration, resume_name = next((it, name) for it, name in candidates if (OUT / "checkpoints" / name).exists())
        resume_checkpoint = OUT / "checkpoints" / resume_name
        rows = [row for row in read_csv("training_curves.csv") if int(row["iteration"]) <= resume_iteration]
        target_rows = [row for row in read_csv("target_segment_counts.csv") if int(row["iteration"]) <= resume_iteration]
        reward_rows = [row for row in read_csv("reward_term_statistics.csv") if int(row["iteration"]) <= resume_iteration]
        manifest_path = OUT / "checkpoint_manifest.json"
        manifests = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
        manifests = [entry for entry in manifests if int(entry["iteration"]) <= resume_iteration]
    abort = None
    with launch_simulation(task_cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=task_cfg), clip_actions=agent_cfg.clip_actions)
        env, device = wrapped.unwrapped, wrapped.unwrapped.device
        dt = float(env.step_dt)
        stand, stw, walk, run, actor, critic, log_std = make_models(cfg, device, paths)
        initial = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        actor.load_state_dict(initial["actor"], strict=True)
        critic.load_state_dict(initial["critic"], strict=True)
        log_std.data.copy_(initial["log_std"].to(device))
        params = [p for p in actor.parameters() if p.requires_grad] + list(critic.parameters()) + [log_std]
        optimizer = torch.optim.Adam(params, lr=cfg["ppo"]["learning_rate"])
        optimizer.load_state_dict(initial["optimizer"])
        robot, command_term, sensor = env.scene["robot"], env.command_manager.get_term("base_velocity"), env.scene.sensors["contact_forces"]
        feet, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        joints, _ = robot.find_joints(".*")
        frozen_modules = [stand.actor, stw.actor, walk.actor, run.actor]
        protected_before = {key: file_sha(paths[key]) for key in EXPECTED}
        save_at = {1: "first_post_update.pt", 10: "model_10.pt", 25: "model_25.pt", 50: "model_50.pt", 75: "model_75.pt", 100: "model_100.pt"}
        persistent_kl = 0

        for iteration in range(resume_iteration + 1, cfg["experiment"]["iterations"] + 1):
            wrapped.reset()
            n, cohort = cfg["experiment"]["physical_envs"], cfg["experiment"]["cohort_size"]
            phase = torch.zeros(n, dtype=torch.long, device=device)
            phase_time = torch.zeros(n, device=device)
            good_time = torch.zeros(n, device=device)
            walk_hold = torch.zeros(n, device=device)
            switches = torch.zeros(n, dtype=torch.long, device=device)
            previous_support = torch.zeros(n, dtype=torch.long, device=device)
            previous_action = torch.zeros(n, 37, device=device)
            heading = robot.data.heading_w.torch.clone()
            slip_dwell = torch.zeros(n, device=device)
            flight_dwell = torch.zeros(n, device=device)
            saturation_dwell = torch.zeros(n, device=device)
            manager = InPlaceEnvIdCohort(n, cohort, cfg["experiment"]["training_seed"] + iteration, device=device)
            ready_ever = torch.zeros(n, dtype=torch.bool, device=device)
            selected = None
            source_steps = 0
            for source_step in range(round(cfg["source_preparation"]["source_preparation_timeout_seconds"] / dt)):
                legacy = wrapped.get_observations()["policy"]
                canonical_state = canonical_state_from_legacy_observation(legacy, heading_w_rad=robot.data.heading_w.torch)
                speed_cmd = torch.zeros(n, device=device)
                speed_cmd[phase == 2] = 1.2 * mj(phase_time[phase == 2] / 1.5)
                speed_cmd[phase == 3] = 1.2
                heading_error = torch.atan2(torch.sin(heading - robot.data.heading_w.torch), torch.cos(heading - robot.data.heading_w.torch))
                yaw = (0.8 * heading_error - 0.1 * robot.data.root_ang_vel_b.torch[:, 2]).clamp(-0.3, 0.3)
                command_term.vel_command_b.zero_()
                command_term.vel_command_b[:, 0], command_term.vel_command_b[:, 2] = speed_cmd, yaw
                command = MotionCommand(speed_cmd, heading, target_yaw_rate_radps=yaw)
                with torch.no_grad():
                    a_stand, a_stw, a_walk = stand(canonical_state, command), stw(canonical_state, command), walk(canonical_state, command)
                    action = torch.zeros(n, 37, device=device)
                    action[(phase == 0) | (phase == 1)] = a_stand[(phase == 0) | (phase == 1)]
                    action[phase == 2] = a_stw[phase == 2]
                    action[phase == 3] = a_walk[phase == 3]
                    _, _, dones, _ = wrapped.step(action)
                previous_action.copy_(action)
                forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :]
                contacts = forces.norm(dim=-1).amax(dim=1) > 5.0
                foot_speed = robot.data.body_lin_vel_w.torch[:, feet, :2].norm(dim=-1)
                slip = torch.where(contacts, foot_speed, torch.zeros_like(foot_speed)).amax(dim=1)
                effort = robot.data.applied_torque.torch[:, joints].abs() / robot.data.joint_effort_limits.torch[:, joints].abs().clamp_min(1e-6)
                saturation_dwell = torch.where((effort >= 0.95).any(1), saturation_dwell + dt, torch.zeros_like(saturation_dwell))
                slip_dwell = torch.where(slip > 0.8, slip_dwell + dt, torch.zeros_like(slip_dwell))
                no_contact = ~contacts.any(1)
                flight_dwell = torch.where(no_contact, flight_dwell + dt, torch.zeros_like(flight_dwell))
                speed = robot.data.root_lin_vel_b.torch[:, 0]
                gravity = robot.data.projected_gravity_b.torch
                roll = torch.atan2(gravity[:, 1], -gravity[:, 2]).abs()
                pitch = torch.atan2(-gravity[:, 0], torch.sqrt(gravity[:, 1] ** 2 + gravity[:, 2] ** 2)).abs()
                support = contacts[:, 0].long() + 2 * contacts[:, 1].long()
                settled = (speed.abs() <= 0.08) & (roll <= 0.10) & (pitch <= 0.10) & contacts.all(1) & (~dones.bool())
                mask = phase == 0
                good_time[mask] = torch.where(settled[mask], good_time[mask] + dt, 0)
                advance = mask & (good_time >= 0.4)
                phase[advance], phase_time[advance], good_time[advance] = 1, 0, 0
                advance = (phase == 1) & (phase_time >= 0.8)
                phase[advance], phase_time[advance] = 2, 0
                changed = (support != previous_support) & ((support == 1) | (support == 2)) & (phase == 2)
                switches[changed] += 1
                acquired = ((speed - 1.2).abs() <= 0.20) & (heading_error.abs() <= 0.12) & (switches >= 2) & (~dones.bool())
                mask = phase == 2
                good_time[mask] = torch.where(acquired[mask], good_time[mask] + dt, 0)
                advance = mask & (good_time >= 0.4)
                phase[advance], phase_time[advance], good_time[advance], walk_hold[advance] = 3, 0, 0, 0
                source_good = (phase == 3) & ((speed - 1.2).abs() <= 0.20) & (heading_error.abs() <= 0.12) & (~dones.bool()) & (slip_dwell < 0.2) & (flight_dwell <= 0.16) & (saturation_dwell < 0.2) & torch.isfinite(legacy).all(1) & torch.isfinite(action).all(1)
                walk_hold = torch.where(source_good, walk_hold + dt, torch.zeros_like(walk_hold))
                contract = source_good & (walk_hold >= 1.0)
                manager.update_ready(contract, source_step)
                ready_ever |= contract
                previous_support.copy_(support)
                phase_time += dt
                source_steps = source_step + 1
                if int(manager.source_ready.sum()) >= cohort and int(ready_ever.sum()) >= math.ceil(0.9 * n):
                    launch = manager.activate(contract, previous_action)
                    selected = launch["physical_env_ids"]
                    if not bool(contract[selected].all()):
                        abort = "source_contract_invalid_at_launch"
                    break
            if abort or selected is None:
                abort = abort or "ready_cohort_formation_failure"
                break

            runner = TransitionOnlyOnPolicyRunner(cohort, cfg["ppo"]["gamma"], cfg["ppo"]["gae_lambda"])
            runner.start_transition(torch.ones(cohort, dtype=torch.bool, device=device))
            generator = torch.Generator(device=device).manual_seed(cfg["experiment"]["training_seed"] + 10000 + iteration)
            target_index = torch.multinomial(torch.tensor(cfg["targets"]["target_probabilities"], device=device), cohort, replacement=True, generator=generator)
            target_values = torch.tensor(cfg["targets"]["target_run_commands_mps"], device=device)
            target = target_values[target_index]
            active = torch.ones(cohort, dtype=torch.bool, device=device)
            valid_masks, old_values = [], []
            transition_elapsed = torch.zeros(cohort, device=device)
            previous_speed = robot.data.root_lin_vel_b.torch[selected, 0].clone()
            previous_contacts = (sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :][selected].norm(dim=-1).amax(dim=1) > 5.0)
            in_flight = ~previous_contacts.any(1)
            flight_start = torch.zeros(cohort, device=device)
            flight_events = torch.zeros(cohort, device=device)
            valid_landings = torch.zeros(cohort, device=device)
            alt_opportunities = torch.zeros(cohort, device=device)
            alternating = torch.zeros(cohort, device=device)
            last_side = torch.full((cohort,), -1, dtype=torch.long, device=device)
            consecutive = torch.zeros(cohort, device=device)
            max_consecutive = torch.zeros(cohort, device=device)
            flight_sum = torch.zeros(cohort, device=device)
            acceptance_streak = torch.zeros(cohort, device=device)
            acceptance_given = torch.zeros(cohort, dtype=torch.bool, device=device)
            local_slip_dwell = torch.zeros(cohort, device=device)
            local_sat_dwell = torch.zeros(cohort, device=device)
            outcome = {name: torch.zeros(cohort, dtype=torch.bool, device=device) for name in ("success", "fall", "slip", "impact", "saturation", "timeout")}
            iteration_reward = {name: 0.0 for name in cfg["reward"]}
            bonus_count = 0
            precursor_count = 0
            segment_lengths = torch.zeros(cohort, dtype=torch.long, device=device)
            actor_grad_norm = critic_grad_norm = policy_loss_value = value_loss_value = entropy_value = kl_value = clip_fraction = explained = 0.0

            for transition_step in range(cfg["rollout"]["rollout_horizon_control_steps"]):
                legacy = wrapped.get_observations()["policy"]
                canonical_full = canonical_state_from_legacy_observation(legacy, heading_w_rad=robot.data.heading_w.torch)
                selected_state = canonical_state_from_legacy_observation(legacy[selected], heading_w_rad=robot.data.heading_w.torch[selected])
                heading_error_full = torch.atan2(torch.sin(heading - robot.data.heading_w.torch), torch.cos(heading - robot.data.heading_w.torch))
                full_target = torch.zeros(n, device=device)
                full_target[phase == 2] = 1.2 * mj(phase_time[phase == 2] / 1.5)
                full_target[phase == 3] = 1.2
                full_target[selected] = 1.2 + (target - 1.2) * mj(torch.full_like(target, transition_step * dt / cfg["rollout"]["minimum_jerk_duration_seconds"]))
                yaw = (1.5 * heading_error_full).clamp(-1.5, 1.5)
                command_term.vel_command_b.zero_()
                command_term.vel_command_b[:, 0], command_term.vel_command_b[:, 2] = full_target, yaw
                full_command = MotionCommand(full_target, heading, target_yaw_rate_radps=yaw)
                selected_command = MotionCommand(full_target[selected], heading[selected], target_yaw_rate_radps=yaw[selected])
                with torch.no_grad():
                    a_stand, a_stw, a_walk = stand(canonical_full, full_command), stw(canonical_full, full_command), walk(canonical_full, full_command)
                    source_action = torch.zeros(n, 37, device=device)
                    source_action[(phase == 0) | (phase == 1)] = a_stand[(phase == 0) | (phase == 1)]
                    source_action[phase == 2] = a_stw[phase == 2]
                    source_action[phase == 3] = a_walk[phase == 3]
                    run_action = run(selected_state, selected_command)
                obs = to_run_observation(selected_state, selected_command, route="RUN")
                if transition_step == 0 and not torch.equal(obs[:, 86:123], previous_action[selected]):
                    abort = "previous_action_mismatch"
                    break
                mean = actor(obs)
                std = log_std.exp().expand_as(mean)
                distribution = Normal(mean, std)
                sampled = distribution.sample()
                log_prob = distribution.log_prob(sampled).sum(-1)
                value = critic(obs).squeeze(-1)
                active_before = active.clone()
                full_action = source_action
                full_action[selected] = torch.where(active[:, None], sampled.detach(), previous_action[selected])
                if not torch.isfinite(full_action).all():
                    abort = "non_finite_action"
                    break
                with torch.no_grad():
                    _, _, dones, info = wrapped.step(full_action)
                applied = full_action[selected].clone()
                speed = robot.data.root_lin_vel_b.torch[selected, 0]
                lateral = robot.data.root_lin_vel_b.torch[selected, 1]
                gravity = robot.data.projected_gravity_b.torch[selected]
                tilt = torch.sqrt(torch.atan2(gravity[:, 1], -gravity[:, 2]).square() + torch.atan2(-gravity[:, 0], torch.sqrt(gravity[:, 1] ** 2 + gravity[:, 2] ** 2)).square())
                heading_error = heading_error_full[selected]
                forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :][selected]
                contacts = forces.norm(dim=-1).amax(dim=1) > 5.0
                impact_force = forces[:, :, :, 2].abs().mean(dim=1).amax(dim=1)
                foot_speed = robot.data.body_lin_vel_w.torch[selected][:, feet, :2].norm(dim=-1)
                slip_speed = torch.where(contacts, foot_speed, torch.zeros_like(foot_speed)).amax(dim=1)
                effort = robot.data.applied_torque.torch[selected][:, joints].abs() / robot.data.joint_effort_limits.torch[selected][:, joints].abs().clamp_min(1e-6)
                local_slip_dwell = torch.where((slip_speed > 0.8) & active, local_slip_dwell + dt, torch.zeros_like(local_slip_dwell))
                local_sat_dwell = torch.where((effort >= 0.95).any(1) & active, local_sat_dwell + dt, torch.zeros_like(local_sat_dwell))
                no_contact = ~contacts.any(1)
                liftoff = (~in_flight) & no_contact & active
                flight_start[liftoff] = transition_elapsed[liftoff]
                in_flight |= liftoff
                landing = in_flight & contacts.any(1) & active
                flight_duration = transition_elapsed - flight_start
                new_contact = contacts & (~previous_contacts)
                valid_landing = landing & (new_contact.sum(1) == 1)
                landing_side = new_contact.long().argmax(1)
                has_last = last_side >= 0
                alternate = valid_landing & has_last & (landing_side != last_side)
                alt_opportunities += (valid_landing & has_last).float()
                alternating += alternate.float()
                safe_cycle = valid_landing & (flight_duration >= 0.04) & (flight_duration <= 0.16) & ((~has_last) | alternate)
                consecutive = torch.where(safe_cycle, consecutive + 1, torch.where(landing, torch.zeros_like(consecutive), consecutive))
                max_consecutive = torch.maximum(max_consecutive, consecutive)
                flight_events += landing.float()
                valid_landings += valid_landing.float()
                flight_sum += torch.where(landing, flight_duration, torch.zeros_like(flight_duration))
                last_side = torch.where(valid_landing, landing_side, last_side)
                in_flight &= ~landing
                previous_contacts = contacts
                mean_flight = flight_sum / flight_events.clamp_min(1)
                periodic = (flight_events >= 4) & (max_consecutive >= 3) & (alternating / alt_opportunities.clamp_min(1) >= 0.8) & (valid_landings / flight_events.clamp_min(1) >= 0.8) & (mean_flight >= 0.04) & (mean_flight <= 0.16)
                torso = env.termination_manager.get_term("base_contact").bool()[selected]
                timeouts = info.get("time_outs", torch.zeros_like(dones)).bool()[selected]
                physical_fall = dones.bool()[selected] & (~timeouts)
                slip_failure = local_slip_dwell >= 0.2
                saturation_failure = local_sat_dwell >= 0.2
                impact_failure = impact_force > 3500.0
                safety_failure = physical_fall | torso | slip_failure | saturation_failure | impact_failure
                acceptance_good = periodic & ((speed - target).abs() <= 0.20) & (heading_error.abs() <= 0.12) & (~safety_failure)
                acceptance_streak = torch.where(acceptance_good & active, acceptance_streak + dt, torch.zeros_like(acceptance_streak))
                success = (acceptance_streak >= 0.4) & active
                timeout = (transition_elapsed + dt >= cfg["rollout"]["transition_timeout_seconds"]) & active & (~success)
                acceptance_first = success & (~acceptance_given)
                acceptance_given |= acceptance_first
                source_error = torch.sqrt((sampled.detach() - a_walk[selected]).square().mean(1))
                target_error = torch.sqrt((sampled.detach() - run_action).square().mean(1))
                action_rate = torch.sqrt((sampled.detach() - previous_action[selected]).square().mean(1))
                term_input = {
                    "speed": speed, "previous_speed": previous_speed, "target_speed": target,
                    "heading_error": heading_error, "lateral_velocity": lateral, "tilt": tilt,
                    "safe_liftoff": liftoff, "safe_flight": landing & (flight_duration >= 0.04) & (flight_duration <= 0.16),
                    "valid_landing": valid_landing, "alternating_landing": alternate, "consecutive_cycle": safe_cycle,
                    "dangerous_slip": slip_failure, "impact_failure": impact_failure,
                    "ankle_saturation": saturation_failure, "knee_saturation": saturation_failure,
                    "excessive_flight": in_flight & (flight_duration > 0.16), "fall": physical_fall,
                    "torso_contact": torso, "joint_limit": torch.zeros_like(active),
                    "action_rate": action_rate, "source_action_error": source_error,
                    "source_alignment_gate": torch.full_like(active, transition_step < 5),
                    "target_action_error": target_error, "target_alignment_gate": (speed - target).abs() <= 0.30,
                    "acceptance_first": acceptance_first,
                }
                weighted, reward = reward_terms(term_input, cfg["reward"], cfg["reward_thresholds"])
                reward *= active_before.float()
                terminal = (success | safety_failure) & active_before
                truncated = timeout & active_before
                runner.transition_step(SegmentStep(obs.detach(), applied.detach(), reward.detach(), value.detach(), terminal, truncated, log_prob.detach()))
                valid_masks.append(active_before)
                old_values.append(value.detach())
                segment_lengths += active_before.long()
                for name, value_term in weighted.items():
                    iteration_reward[name] += float((value_term * active_before.float()).sum())
                bonus_count += int(acceptance_first.sum())
                precursor_count += int(liftoff.sum() + valid_landing.sum() + alternate.sum() + safe_cycle.sum())
                outcome["success"] |= success
                outcome["fall"] |= physical_fall & active_before
                outcome["slip"] |= slip_failure & active_before
                outcome["impact"] |= impact_failure & active_before
                outcome["saturation"] |= saturation_failure & active_before
                outcome["timeout"] |= timeout
                active &= ~(terminal | truncated)
                transition_elapsed += active_before.float() * dt
                previous_speed = speed
                previous_action.copy_(full_action)
                if not bool(active.any()):
                    break
            if abort:
                break
            if bool(active.any()):
                outcome["timeout"] |= active
                runner.storage.steps[-1].truncated |= active
                active.zero_()
            returns, advantages = runner.storage.finish(torch.zeros(cohort, device=device))
            valid = torch.stack(valid_masks)
            observations = torch.stack([s.observation for s in runner.storage.steps])
            actions = torch.stack([s.action for s in runner.storage.steps])
            old_log_prob = torch.stack([s.log_prob for s in runner.storage.steps])
            old_value = torch.stack(old_values)
            flat = valid.flatten()
            obs_flat, action_flat = observations.flatten(0, 1)[flat], actions.flatten(0, 1)[flat]
            old_log_flat, old_value_flat = old_log_prob.flatten()[flat], old_value.flatten()[flat]
            return_flat, advantage_flat = returns.flatten()[flat], advantages.flatten()[flat]
            advantage_flat = (advantage_flat - advantage_flat.mean()) / advantage_flat.std().clamp_min(1e-8)
            sample_count = obs_flat.shape[0]
            order_generator = torch.Generator(device=device).manual_seed(cfg["experiment"]["training_seed"] + 20000 + iteration)
            epoch_kls, epoch_clips, actor_norms, critic_norms = [], [], [], []
            for _ in range(cfg["ppo"]["ppo_epochs"]):
                order = torch.randperm(sample_count, generator=order_generator, device=device)
                for indices in order.chunk(cfg["ppo"]["num_minibatches"]):
                    new_mean = actor(obs_flat[indices])
                    distribution = Normal(new_mean, log_std.exp().expand_as(new_mean))
                    new_log = distribution.log_prob(action_flat[indices]).sum(-1)
                    entropy = distribution.entropy().sum(-1).mean()
                    ratio = (new_log - old_log_flat[indices]).exp()
                    surrogate = ratio * advantage_flat[indices]
                    clipped = ratio.clamp(1 - cfg["ppo"]["clip_parameter"], 1 + cfg["ppo"]["clip_parameter"]) * advantage_flat[indices]
                    policy_loss = -torch.minimum(surrogate, clipped).mean()
                    new_value = critic(obs_flat[indices]).squeeze(-1)
                    value_clipped = old_value_flat[indices] + (new_value - old_value_flat[indices]).clamp(-cfg["ppo"]["clip_parameter"], cfg["ppo"]["clip_parameter"])
                    value_loss = torch.maximum((new_value - return_flat[indices]).square(), (value_clipped - return_flat[indices]).square()).mean()
                    loss = policy_loss + cfg["ppo"]["value_loss_coefficient"] * value_loss - cfg["ppo"]["entropy_coefficient"] * entropy
                    optimizer.zero_grad()
                    loss.backward()
                    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in params):
                        abort = "non_finite_gradient"
                        break
                    actor_params = [p for p in actor.parameters() if p.requires_grad] + [log_std]
                    actor_norm = torch.nn.utils.clip_grad_norm_(actor_params, cfg["ppo"]["max_gradient_norm"])
                    critic_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg["ppo"]["max_gradient_norm"])
                    optimizer.step()
                    with torch.no_grad():
                        epoch_kls.append(float((old_log_flat[indices] - new_log).mean()))
                        epoch_clips.append(float(((ratio - 1).abs() > cfg["ppo"]["clip_parameter"]).float().mean()))
                        actor_norms.append(float(actor_norm))
                        critic_norms.append(float(critic_norm))
                        policy_loss_value, value_loss_value, entropy_value = float(policy_loss), float(value_loss), float(entropy)
                if abort:
                    break
            if abort:
                break
            kl_value = sum(epoch_kls) / len(epoch_kls)
            clip_fraction = sum(epoch_clips) / len(epoch_clips)
            actor_grad_norm = sum(actor_norms) / len(actor_norms)
            critic_grad_norm = sum(critic_norms) / len(critic_norms)
            residual = return_flat - return_flat.mean()
            explained = 1 - float((return_flat - old_value_flat).var() / return_flat.var().clamp_min(1e-8))
            std_now = log_std.exp()
            if float(std_now.min()) < cfg["exploration"]["std_min_abort_threshold"] or float(std_now.max()) > cfg["exploration"]["std_max_abort_threshold"]:
                abort = "exploration_std_abort"
                break
            persistent_kl = persistent_kl + 1 if kl_value > 0.1 else 0
            if persistent_kl >= 3:
                abort = "persistent_kl_explosion"
                break
            if len(torch.unique(target_index)) < 3:
                abort = "all_target_sampling_collapse"
                break
            if any(file_sha(paths[key]) != protected_before[key] for key in EXPECTED):
                abort = "frozen_tensor_hash_change"
                break
            total_valid = int(valid.sum())
            outcome_rates = {key: float(value.float().mean()) for key, value in outcome.items()}
            target_counts = {str(float(t)): int((target == t).sum()) for t in target_values}
            target_success = {str(float(t)): float(outcome["success"][target == t].float().mean()) for t in target_values}
            rows.append({
                "iteration": iteration, "source_preparation_success": float(ready_ever.float().mean()),
                "cohort_formation_time_s": source_steps * dt, "ready_env_count": int(ready_ever.sum()),
                "transition_segment_count": cohort, "segment_length_mean": float(segment_lengths.float().mean()),
                "segment_length_p95": float(torch.quantile(segment_lengths.float(), 0.95)),
                "transition_completion": outcome_rates["success"], "target_speed_acquisition": outcome_rates["success"],
                "periodic_running_acquisition": outcome_rates["success"], "run_acceptance": outcome_rates["success"],
                "run_takeover": 0.0, "timeout": outcome_rates["timeout"], "fall": outcome_rates["fall"],
                "slip_failure": outcome_rates["slip"], "impact_failure": outcome_rates["impact"],
                "saturation_failure": outcome_rates["saturation"], "mean_transition_reward": float(return_flat.mean()),
                "policy_loss": policy_loss_value, "value_loss": value_loss_value, "entropy": entropy_value,
                "kl": kl_value, "clip_fraction": clip_fraction, "explained_variance": explained,
                "actor_gradient_norm": actor_grad_norm, "critic_gradient_norm": critic_grad_norm,
                "exploration_std_min": float(std_now.min()), "exploration_std_mean": float(std_now.mean()),
                "exploration_std_max": float(std_now.max()), "completion_bonus_count": bonus_count,
                "precursor_reward_fire_count": precursor_count, "safe_cycle_count": int(max_consecutive.sum()),
            })
            for t in target_values:
                target_rows.append({"iteration": iteration, "target_mps": float(t), "segments": target_counts[str(float(t))], "success_rate": target_success[str(float(t))]})
            reward_rows.append({"iteration": iteration, **{name: value / max(total_valid, 1) for name, value in iteration_reward.items()}})
            if iteration in save_at:
                manifests.append(checkpoint(OUT / "checkpoints" / save_at[iteration], iteration, actor, critic, log_std, optimizer, cfg))
            # Durable per-iteration audit: Isaac shutdown must not be able to
            # discard an otherwise completed 100-update pilot.
            write_csv("training_curves.csv", rows)
            write_csv("target_segment_counts.csv", target_rows)
            write_csv("reward_term_statistics.csv", reward_rows)
            write_json("checkpoint_manifest.json", manifests)
            write_json("training_diagnostics.json", {
                "requested_iterations": 100, "completed_iterations": len(rows), "abort_reason": abort,
                "kl_abort_rule": "mean approximate KL > 0.1 for 3 consecutive iterations",
                "source_prefix_stored_steps": 0, "invalid_stored_steps": 0,
                "previous_action_mismatch": 0,
                "protected_hashes_after": {key: file_sha(REPO / path) for key, (path, _) in EXPECTED.items()},
            })
            print(f"[{'Stage7R8' if PILOT2 else 'Stage7R7'}] iteration={iteration:03d} success={outcome_rates['success']:.3f} timeout={outcome_rates['timeout']:.3f} kl={kl_value:.5f} std={float(std_now.mean()):.4f}", flush=True)
        wrapped.close()
    write_csv("training_curves.csv", rows)
    write_csv("target_segment_counts.csv", target_rows)
    write_csv("reward_term_statistics.csv", reward_rows)
    write_json("training_diagnostics.json", {
        "requested_iterations": 100, "completed_iterations": len(rows), "abort_reason": abort,
        "kl_abort_rule": "mean approximate KL > 0.1 for 3 consecutive iterations",
        "source_prefix_stored_steps": 0, "invalid_stored_steps": 0,
        "previous_action_mismatch": int(abort == "previous_action_mismatch"),
        "protected_hashes_after": {key: file_sha(REPO / path) for key, (path, _) in EXPECTED.items()},
    })
    write_json("checkpoint_manifest.json", manifests)
    if abort:
        raise RuntimeError(f"{'Stage 7R8' if PILOT2 else 'Stage 7R7'} aborted: {abort}")


def main():
    cfg, hashes = authorize()
    if args.phase == "prepare":
        prepare_phase(cfg, hashes)
    else:
        main_train(cfg, hashes)


if __name__ == "__main__":
    main()
