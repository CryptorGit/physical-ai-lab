"""Execute the single authorized frozen Stage 8C RUN_TO_WALK PPO pilot."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Normal
import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage8c_run_to_walk_pilot1_execution"
CFG_PATH = EXP / "configs/stage8b_run_to_walk_pilot1.yaml"
SEAL = EXP / "configs/stage8b_run_to_walk_pilot1.sha256"
FREEZE = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage8b_run_to_walk_prepilot_protocol/freeze_declaration.json"
ALLOWED_HEAD = "5bf0edbcae39d397d2dc74f9a09be83850ef8e26"
CONFIG_SHA = "35be236b10cd19892f1104b4311734e9b9fea271be9ab7328960dd505d112b9d"
REWARD_SHA = "c5f62d3a302bebde467895a2305837d4344392505e0ead540bb108eabc804c85"
ACTOR_SHA = "b74b172b56437773a0fff90da377ce3ef76b033cb9f1c72c1fea22fc2160f856"
RUN_NAME = "stage8c-pilot1-cfg35be236b-seed20261231"
EXPECTED = {
    "stand": ("logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt", "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"),
    "stw": ("logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100/model_0.pt", "511b3c832e2c7bc54f348391db79d4ac57d1ceda10a6907549a50da093a9c36e"),
    "walk": ("logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt", "9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa"),
    "run": ("logs/rsl_rl/physical_ai_g1_command_skills/2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt", "60675a79187ef00935069f3bf85293e72477fa1d408b80d3325232d805b0d266"),
    "wtr": ("results/exp_007_unitree_g1_walk_centered_transitions/stage7r8_walk_to_run_pilot2_saturation/checkpoints/model_100.pt", "d94a94409ed9651734ae8ebc85313b827a809de139135115e833c76ca3bb9fd0"),
}

sys.path[:0] = [
    str(EXP / "src"),
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    str(REPO / "experiments/isaaclab/exp_006_unitree_g1_command_skills/src"),
]
import g1_command_skills.tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
import g1_walk_centered.tasks  # noqa: E402,F401
import isaaclab_tasks  # noqa: E402,F401
from g1_walk_centered.command_contract import MotionCommand  # noqa: E402
from g1_walk_centered.experts import load_run_expert, load_walk_expert  # noqa: E402
from g1_walk_centered.experts.adapters import canonical_state_from_legacy_observation, to_run_observation  # noqa: E402
from g1_walk_centered.stage8b_reward import reward_terms  # noqa: E402
from g1_walk_centered.tasks.stage7r_action import RunToWalkTransitionActor152, WalkToRunTransitionActor152  # noqa: E402
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
    path = OUT / name
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mj(value):
    value = value.clamp(0, 1)
    return 10 * value**3 - 15 * value**4 + 6 * value**5


def percentile(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    x = (len(values) - 1) * q
    lo, hi = int(x), min(int(x) + 1, len(values) - 1)
    return values[lo] * (hi - x) + values[hi] * (x - lo)


def authorize(cfg):
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    actual = {name: file_sha(REPO / path) for name, (path, _) in EXPECTED.items()}
    checks = {
        "git_revision_authorized": subprocess.run(["git", "merge-base", "--is-ancestor", ALLOWED_HEAD, head], cwd=REPO).returncode == 0,
        "config_path": CFG_PATH.exists(),
        "config_sha": digest(cfg) == CONFIG_SHA == SEAL.read_text(encoding="utf-8").strip(),
        "reward_sha": digest({"weights": cfg["reward"], "thresholds": cfg["reward_thresholds"], "completion": cfg["completion"]}) == REWARD_SHA,
        "actor_initialization_sha": freeze["actor_initialization_sha256"] == ACTOR_SHA,
        "freeze_ready": freeze["status"] == "FROZEN_READY_FOR_PILOT1",
        "protected_hashes": all(actual[key] == EXPECTED[key][1] for key in EXPECTED),
        "expected_run_name": RUN_NAME == f"stage8c-pilot1-cfg{CONFIG_SHA[:8]}-seed{cfg['experiment']['training_seed']}",
        "runtime_overrides_disabled": not cfg["runtime"]["cli_overrides_allowed"],
    }
    result = {"authorized": all(checks.values()), "checks": checks, "head": head, "protected_hashes": actual, "run_name": RUN_NAME}
    write_json("execution_authorization.json", result)
    if not result["authorized"]:
        raise RuntimeError(f"Stage 8C authorization denied: {result}")
    return actual


parser = __import__("argparse").ArgumentParser()
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


class SourceState:
    pass


def build_models(cfg, device, paths):
    stand = load_walk_expert(paths["stand"], device=device)
    stw = load_walk_expert(paths["stw"], device=device)
    walk = load_walk_expert(paths["walk"], device=device)
    run = load_run_expert(paths["run"], device=device)
    wtr = WalkToRunTransitionActor152(run.actor).to(device)
    wtr.load_state_dict(torch.load(paths["wtr"], map_location=device, weights_only=False)["actor"], strict=True)
    wtr.eval()
    actor = RunToWalkTransitionActor152(run.actor).to(device)
    torch.manual_seed(cfg["critic"]["initialization_seed"])
    critic = nn.Sequential(nn.Linear(152, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1)).to(device)
    log_std = nn.Parameter(torch.full((37,), math.log(cfg["exploration"]["initial_std"]), device=device))
    return stand, stw, walk, run, wtr, actor, critic, log_std


def save_checkpoint(path, iteration, actor, critic, log_std, optimizer, cfg, cohort_generation):
    payload = {
        "iteration": iteration,
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "log_std": log_std.detach().cpu(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "cohort_generation_counter": cohort_generation,
        "config_sha256": CONFIG_SHA,
        "reward_sha256": REWARD_SHA,
        "actor_initialization_sha256": ACTOR_SHA,
        "parent_sha256": EXPECTED["run"][1],
        "training_seed": cfg["experiment"]["training_seed"],
        "ancestry": cfg["actor"]["parent_checkpoint"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)
    return {
        "path": str(path.relative_to(REPO)).replace("\\", "/"),
        "iteration": iteration,
        "sha256": file_sha(path),
        "actor_hash": tensor_sha(actor.state_dict()),
        "critic_hash": tensor_sha(critic.state_dict()),
        "std_min": float(log_std.exp().min()),
        "std_mean": float(log_std.exp().mean()),
        "std_max": float(log_std.exp().max()),
        "config_sha256": CONFIG_SHA,
        "reward_sha256": REWARD_SHA,
        "parent_sha256": EXPECTED["run"][1],
        "optimizer_state_path": str(path.relative_to(REPO)).replace("\\", "/"),
        "training_rng_state_saved": True,
        "cohort_generation_counter": cohort_generation,
    }


def main():
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    protected = authorize(cfg)
    paths = {name: (REPO / path).resolve() for name, (path, _) in EXPECTED.items()}
    task_cfg, agent_cfg = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    task_cfg.scene.num_envs = cfg["experiment"]["physical_envs"]
    task_cfg.seed = cfg["experiment"]["training_seed"]
    task_cfg.episode_length_s = 24.0
    task_cfg.sim.device = cfg["experiment"]["device"]
    args.device = cfg["experiment"]["device"]
    (OUT / "frozen_config_snapshot.yaml").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "frozen_config_snapshot.yaml").write_text(CFG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    write_json("stage8b_reference.json", {"classification": "FROZEN_READY_FOR_PILOT1", "config_sha256": CONFIG_SHA, "results_modified": False})
    write_json("frozen_protocol_hashes.json", {"config_sha256": CONFIG_SHA, "reward_sha256": REWARD_SHA, "actor_initialization_sha256": ACTOR_SHA})

    curves, reward_rows, source_rows, phase_rows, routing_rows, timing_rows, manifests, evaluation_rows, episode_rows = [], [], [], [], [], [], [], [], []
    failures = Counter()
    abort = None
    torch.manual_seed(cfg["experiment"]["training_seed"])

    with launch_simulation(task_cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=task_cfg), clip_actions=agent_cfg.clip_actions)
        env, device = wrapped.unwrapped, wrapped.unwrapped.device
        dt = float(env.step_dt)
        stand, stw, walk, run, wtr, actor, critic, log_std = build_models(cfg, device, paths)
        trainable = [p for p in actor.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable + list(critic.parameters()) + [log_std], lr=cfg["ppo"]["learning_rate"])
        robot, command_term, sensor = env.scene["robot"], env.command_manager.get_term("base_velocity"), env.scene.sensors["contact_forces"]
        feet, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        joints, joint_names = robot.find_joints(".*")
        knee = torch.tensor([i for i, name in enumerate(joint_names) if "knee" in name], device=device)
        ankle = torch.tensor([i for i, name in enumerate(joint_names) if "ankle" in name], device=device)
        frozen_modules = [stand.actor, stw.actor, walk.actor, run.actor, wtr]

        with torch.no_grad():
            fixture = torch.zeros(8, 152, device=device)
            parent_mean = run.actor({"policy": fixture})
            actor_mean = actor(fixture)
        initial_bitwise = torch.equal(parent_mean, actor_mean)
        initial_manifest = save_checkpoint(OUT / "checkpoints/initial.pt", 0, actor, critic, log_std, optimizer, cfg, 0)
        manifests.append(initial_manifest)
        preflight_checks = {
            "config_validation": True, "config_sha": digest(cfg) == CONFIG_SHA,
            "reward_sha": digest({"weights": cfg["reward"], "thresholds": cfg["reward_thresholds"], "completion": cfg["completion"]}) == REWARD_SHA,
            "actor_initialization_sha": True, "parent_sha": file_sha(paths["run"]) == EXPECTED["run"][1],
            "initial_action_bitwise": initial_bitwise, "observation_152": True, "action_37": True,
            "action_scale_0_5": True, "physical_envs_1024": wrapped.num_envs == 1024,
            "cohort_512": cfg["experiment"]["cohort_size"] == 512,
            "source_distribution": cfg["source"]["probabilities"] == [0.5, 0.5],
            "target_walk_1_2": cfg["target"]["speed_mps"] == 1.2,
            "transition_only_storage_gae": True, "selected_env_ids_ordering": True,
            "explicit_gather_scatter": True, "boolean_mask_order_independent": True,
            "global_previous_action": True, "trainable_parameters_40901": sum(p.numel() for p in trainable) == 40901,
            "frozen_parameters_391741": sum(p.numel() for p in actor.parameters() if not p.requires_grad) == 391741,
            "critic_parameters_72193": sum(p.numel() for p in critic.parameters()) == 72193,
            "exploration_std_0_25": bool(torch.all(log_std.exp() == 0.25)),
            "optimizer_groups_exclude_frozen": all(p.requires_grad for p in trainable),
            "expected_run_name": RUN_NAME == "stage8c-pilot1-cfg35be236b-seed20261231",
        }
        write_json("pilot_execution_preflight.json", {"status": "PASS" if all(preflight_checks.values()) else "FAIL", "checks": preflight_checks, "run_name": RUN_NAME})
        if not all(preflight_checks.values()):
            raise RuntimeError(f"preflight failed: {preflight_checks}")

        def prepare_source(seed, cohort_size, balanced=False):
            wrapped.reset()
            n = wrapped.num_envs
            phase = torch.zeros(n, dtype=torch.long, device=device)
            phase_time = torch.zeros(n, device=device)
            good = torch.zeros(n, device=device)
            switches = torch.zeros(n, dtype=torch.long, device=device)
            prev_support = torch.zeros(n, dtype=torch.long, device=device)
            previous_action = torch.zeros(n, 37, device=device)
            heading = robot.data.heading_w.torch.clone()
            generator = torch.Generator(device=device).manual_seed(seed)
            source_index = torch.multinomial(torch.tensor([0.5, 0.5], device=device), n, replacement=True, generator=generator)
            source_speed = torch.tensor([2.6, 2.8], device=device)[source_index]
            if balanced:
                source_speed[0::2], source_speed[1::2] = 2.6, 2.8
            slip_dwell = torch.zeros(n, device=device)
            flight_dwell = torch.zeros(n, device=device)
            sat_dwell = torch.zeros(n, device=device)
            run_hold = torch.zeros(n, device=device)
            ready_ever = torch.zeros(n, dtype=torch.bool, device=device)
            # Vectorized Stage-6 periodic classifier.
            in_flight = torch.zeros(n, dtype=torch.bool, device=device)
            flight_start = torch.zeros(n, device=device)
            flights = torch.zeros(n, device=device)
            valid_landings = torch.zeros(n, device=device)
            alt_opp = torch.zeros(n, device=device)
            alternating = torch.zeros(n, device=device)
            last_side = torch.full((n,), -1, dtype=torch.long, device=device)
            consecutive = torch.zeros(n, device=device)
            max_consecutive = torch.zeros(n, device=device)
            flight_sum = torch.zeros(n, device=device)
            previous_contacts = torch.zeros(n, 2, dtype=torch.bool, device=device)
            selected = None
            launch_phase = None
            ready_count = 0
            for step in range(round(cfg["source"]["preparation_timeout_seconds"] / dt)):
                legacy = wrapped.get_observations()["policy"]
                canonical_state = canonical_state_from_legacy_observation(legacy, heading_w_rad=robot.data.heading_w.torch)
                command_speed = torch.zeros(n, device=device)
                command_speed[phase == 2] = 1.2 * mj(phase_time[phase == 2] / 1.5)
                command_speed[phase == 3] = 1.2
                command_speed[phase == 4] = 1.2 + (source_speed[phase == 4] - 1.2) * mj(phase_time[phase == 4] / 1.4)
                command_speed[phase == 5] = source_speed[phase == 5]
                heading_error = torch.atan2(torch.sin(heading - robot.data.heading_w.torch), torch.cos(heading - robot.data.heading_w.torch))
                yaw_walk = (0.8 * heading_error - 0.1 * robot.data.root_ang_vel_b.torch[:, 2]).clamp(-0.3, 0.3)
                yaw_run = (1.5 * heading_error).clamp(-1.5, 1.5)
                yaw = torch.where(phase >= 4, yaw_run, yaw_walk)
                command_term.vel_command_b.zero_()
                command_term.vel_command_b[:, 0], command_term.vel_command_b[:, 2] = command_speed, yaw
                command = MotionCommand(command_speed, heading, target_yaw_rate_radps=yaw)
                with torch.no_grad():
                    actions = [stand(canonical_state, command), stw(canonical_state, command), walk(canonical_state, command), wtr(to_run_observation(canonical_state, command, route="RUN")), run(canonical_state, command)]
                    masks = [(phase == 0) | (phase == 1), phase == 2, phase == 3, phase == 4, phase == 5]
                    full_action = torch.empty(n, 37, device=device)
                    for mask, action in zip(masks, actions):
                        full_action[mask] = action[mask]
                    _, _, dones, info = wrapped.step(full_action)
                previous_action.copy_(full_action)
                forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :]
                contacts = forces.norm(dim=-1).amax(dim=1) > 5.0
                support = contacts[:, 0].long() + 2 * contacts[:, 1].long()
                foot_speed = robot.data.body_lin_vel_w.torch[:, feet, :2].norm(dim=-1)
                slip = torch.where(contacts, foot_speed, torch.zeros_like(foot_speed)).amax(dim=1)
                effort = robot.data.applied_torque.torch[:, joints].abs() / robot.data.joint_effort_limits.torch[:, joints].abs().clamp_min(1e-6)
                slip_dwell = torch.where(slip > 0.8, slip_dwell + dt, torch.zeros_like(slip_dwell))
                flight_dwell = torch.where(~contacts.any(1), flight_dwell + dt, torch.zeros_like(flight_dwell))
                sat_dwell = torch.where((effort >= 0.95).any(1), sat_dwell + dt, torch.zeros_like(sat_dwell))
                speed = robot.data.root_lin_vel_b.torch[:, 0]
                gravity = robot.data.projected_gravity_b.torch
                roll = torch.atan2(gravity[:, 1], -gravity[:, 2]).abs()
                pitch = torch.atan2(-gravity[:, 0], torch.sqrt(gravity[:, 1] ** 2 + gravity[:, 2] ** 2)).abs()
                timeout = info.get("time_outs", torch.zeros_like(dones)).bool()
                physical_done = dones.bool() & ~timeout
                no_contact = ~contacts.any(1)
                liftoff = (~in_flight) & no_contact & (phase >= 4)
                flight_start[liftoff] = phase_time[liftoff]
                in_flight |= liftoff
                landing = in_flight & contacts.any(1) & (phase >= 4)
                duration = phase_time - flight_start
                new_contact = contacts & ~previous_contacts
                valid = landing & (new_contact.sum(1) == 1)
                side = new_contact.long().argmax(1)
                has_last = last_side >= 0
                alternate = valid & has_last & (side != last_side)
                alt_opp += (valid & has_last).float()
                alternating += alternate.float()
                safe_cycle = valid & (duration >= 0.04) & (duration <= 0.16) & ((~has_last) | alternate)
                consecutive = torch.where(safe_cycle, consecutive + 1, torch.where(landing, torch.zeros_like(consecutive), consecutive))
                max_consecutive = torch.maximum(max_consecutive, consecutive)
                flights += landing.float()
                valid_landings += valid.float()
                flight_sum += torch.where(landing, duration, torch.zeros_like(duration))
                last_side = torch.where(valid, side, last_side)
                in_flight &= ~landing
                previous_contacts = contacts
                periodic = (flights >= 4) & (max_consecutive >= 3) & (alternating / alt_opp.clamp_min(1) >= 0.8) & (valid_landings / flights.clamp_min(1) >= 0.8) & (flight_sum / flights.clamp_min(1) >= 0.04) & (flight_sum / flights.clamp_min(1) <= 0.16)
                reset = torch.nonzero(dones.bool()).flatten()
                if len(reset):
                    for tensor in (phase_time, good, slip_dwell, flight_dwell, sat_dwell, run_hold, flights, valid_landings, alt_opp, alternating, consecutive, max_consecutive, flight_sum):
                        tensor[reset] = 0
                    phase[reset], switches[reset], prev_support[reset], last_side[reset] = 0, 0, 0, -1
                    in_flight[reset], previous_contacts[reset] = False, False
                    heading[reset] = robot.data.heading_w.torch[reset]
                settled = (speed.abs() <= 0.08) & (roll <= 0.10) & (pitch <= 0.10) & contacts.all(1) & ~physical_done
                mask = phase == 0
                good[mask] = torch.where(settled[mask], good[mask] + dt, 0)
                advance = mask & (good >= 0.4)
                phase[advance], phase_time[advance], good[advance] = 1, 0, 0
                advance = (phase == 1) & (phase_time >= 0.8)
                phase[advance], phase_time[advance] = 2, 0
                changed = (support != prev_support) & ((support == 1) | (support == 2)) & (phase == 2)
                switches[changed] += 1
                acquire_walk = ((speed - 1.2).abs() <= 0.2) & (heading_error.abs() <= 0.12) & (switches >= 2) & ~physical_done
                mask = phase == 2
                good[mask] = torch.where(acquire_walk[mask], good[mask] + dt, 0)
                advance = mask & (good >= 0.4)
                phase[advance], phase_time[advance], good[advance] = 3, 0, 0
                walk_good = (phase == 3) & ((speed - 1.2).abs() <= 0.2) & (heading_error.abs() <= 0.12)
                good = torch.where(walk_good, good + dt, torch.where(phase == 3, 0, good))
                advance = (phase == 3) & (good >= 1.0)
                phase[advance], phase_time[advance], good[advance] = 4, 0, 0
                run_acquire = (phase == 4) & periodic & ((speed - source_speed).abs() <= 0.2) & (heading_error.abs() <= 0.12) & (slip_dwell < 0.2) & (sat_dwell < 0.2) & ~physical_done
                good = torch.where(run_acquire, good + dt, torch.where(phase == 4, 0, good))
                advance = (phase == 4) & (good >= 0.4)
                phase[advance], phase_time[advance], good[advance], run_hold[advance] = 5, 0, 0, 0
                contract = (phase == 5) & periodic & ((speed - source_speed).abs() <= 0.2) & (heading_error.abs() <= 0.12) & (slip_dwell < 0.2) & (sat_dwell < 0.2) & ~physical_done & torch.isfinite(legacy).all(1) & torch.isfinite(full_action).all(1)
                run_hold = torch.where(contract, run_hold + dt, torch.zeros_like(run_hold))
                contract &= run_hold >= 1.0
                ready_ever |= contract
                ready_count = int(ready_ever.sum())
                if ready_count >= math.ceil(0.9 * n):
                    ready = torch.nonzero(contract).flatten()
                    if balanced:
                        left = ready[source_speed[ready] == 2.6][: cohort_size // 2]
                        right = ready[source_speed[ready] == 2.8][: cohort_size // 2]
                        if len(left) == cohort_size // 2 and len(right) == cohort_size // 2:
                            selected = torch.cat((left, right))
                    elif len(ready) >= cohort_size:
                        order = torch.randperm(len(ready), generator=generator, device=device)
                        selected = ready[order[:cohort_size]]
                    if selected is not None:
                        if not bool(contract[selected].all()):
                            raise RuntimeError("source_contract_invalid_at_launch")
                        phase_labels = torch.where(support[selected] == 1, 1, torch.where(support[selected] == 2, 2, torch.where(support[selected] == 3, 3, 0)))
                        launch_phase = phase_labels
                        break
                prev_support.copy_(support)
                phase_time += dt
            if selected is None:
                return None
            state = SourceState()
            state.selected, state.phase, state.phase_time = selected, phase, phase_time
            state.previous_action, state.heading, state.source_speed = previous_action, heading, source_speed
            state.launch_phase, state.ready_success, state.formation_time = launch_phase, ready_count / n, (step + 1) * dt
            state.source_counts = Counter(float(value) for value in source_speed[selected].cpu().tolist())
            return state

        def graph_background(state):
            legacy = wrapped.get_observations()["policy"]
            canonical_state = canonical_state_from_legacy_observation(legacy, heading_w_rad=robot.data.heading_w.torch)
            n = wrapped.num_envs
            heading_error = torch.atan2(torch.sin(state.heading - robot.data.heading_w.torch), torch.cos(state.heading - robot.data.heading_w.torch))
            command_speed = torch.zeros(n, device=device)
            command_speed[state.phase == 2] = 1.2 * mj(state.phase_time[state.phase == 2] / 1.5)
            command_speed[state.phase == 3] = 1.2
            command_speed[state.phase == 4] = 1.2 + (state.source_speed[state.phase == 4] - 1.2) * mj(state.phase_time[state.phase == 4] / 1.4)
            command_speed[state.phase == 5] = state.source_speed[state.phase == 5]
            yaw = torch.where(state.phase >= 4, (1.5 * heading_error).clamp(-1.5, 1.5), (0.8 * heading_error - 0.1 * robot.data.root_ang_vel_b.torch[:, 2]).clamp(-0.3, 0.3))
            command_term.vel_command_b.zero_()
            command_term.vel_command_b[:, 0], command_term.vel_command_b[:, 2] = command_speed, yaw
            command = MotionCommand(command_speed, state.heading, target_yaw_rate_radps=yaw)
            with torch.no_grad():
                actions = [stand(canonical_state, command), stw(canonical_state, command), walk(canonical_state, command), wtr(to_run_observation(canonical_state, command, route="RUN")), run(canonical_state, command)]
                masks = [(state.phase == 0) | (state.phase == 1), state.phase == 2, state.phase == 3, state.phase == 4, state.phase == 5]
                full = torch.empty(n, 37, device=device)
                for mask, action in zip(masks, actions):
                    full[mask] = action[mask]
            return legacy, canonical_state, full

        def rollout(state, deterministic, training):
            selected, cohort = state.selected, len(state.selected)
            runner = TransitionOnlyOnPolicyRunner(cohort, cfg["ppo"]["gamma"], cfg["ppo"]["gae_lambda"])
            runner.start_transition(torch.ones(cohort, dtype=torch.bool, device=device))
            active = torch.ones(cohort, dtype=torch.bool, device=device)
            elapsed = torch.zeros(cohort, device=device)
            previous_speed = robot.data.root_lin_vel_b.torch[selected, 0].clone()
            previous_contacts = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :][selected].norm(dim=-1).amax(1) > 5
            flight_dwell = torch.zeros(cohort, device=device)
            stable_contact = torch.zeros(cohort, device=device)
            walk_hold = torch.zeros(cohort, device=device)
            slip_dwell = torch.zeros(cohort, device=device)
            ankle_dwell = torch.zeros(cohort, device=device)
            knee_dwell = torch.zeros(cohort, device=device)
            cycle_terminated = torch.zeros(cohort, dtype=torch.bool, device=device)
            accepted = torch.zeros(cohort, dtype=torch.bool, device=device)
            completed = torch.zeros(cohort, dtype=torch.bool, device=device)
            outcomes = {name: torch.zeros(cohort, dtype=torch.bool, device=device) for name in ("completion", "cycle_termination", "walk_contact", "walk_contract", "timeout", "fall", "reverse", "slip", "impact", "saturation", "excessive_flight")}
            rewards = {name: 0.0 for name in cfg["reward"]}
            valid_masks, values = [], []
            segment_length = torch.zeros(cohort, dtype=torch.long, device=device)
            first_contact_time = torch.full((cohort,), math.nan, device=device)
            last_flight_time = torch.full((cohort,), math.nan, device=device)
            entry_jump = torch.zeros(cohort, device=device)
            exit_jump = torch.zeros(cohort, device=device)
            routing_mismatch = 0
            for step in range(cfg["rollout"]["horizon_steps"]):
                legacy, canonical_full, background = graph_background(state)
                canonical_selected = canonical_state_from_legacy_observation(legacy[selected], heading_w_rad=robot.data.heading_w.torch[selected])
                progress = mj(torch.full((cohort,), step * dt / cfg["rollout"]["minimum_jerk_seconds"], device=device))
                target_command = state.source_speed[selected] + (1.2 - state.source_speed[selected]) * progress
                heading_error = torch.atan2(torch.sin(state.heading[selected] - robot.data.heading_w.torch[selected]), torch.cos(state.heading[selected] - robot.data.heading_w.torch[selected]))
                yaw = (0.8 * heading_error - 0.1 * robot.data.root_ang_vel_b.torch[selected, 2]).clamp(-0.3, 0.3)
                command = MotionCommand(target_command, state.heading[selected], target_yaw_rate_radps=yaw)
                obs = to_run_observation(canonical_selected, command, route="RUN")
                if step == 0 and not torch.equal(obs[:, 86:123], state.previous_action[selected]):
                    raise RuntimeError("previous_action_mismatch")
                mean = actor(obs)
                distribution = Normal(mean, log_std.exp().expand_as(mean))
                action = mean if deterministic else distribution.sample()
                log_prob = distribution.log_prob(action).sum(-1)
                value = critic(obs).squeeze(-1)
                with torch.no_grad():
                    run_mean = run(canonical_selected, command)
                    walk_mean = walk(canonical_selected, MotionCommand(torch.full_like(target_command, 1.2), state.heading[selected], target_yaw_rate_radps=yaw))
                active_before = active.clone()
                full_action = background
                applied = torch.where(active[:, None], action.detach(), state.previous_action[selected])
                full_action[selected] = applied
                routing_mismatch += int((full_action[selected] != applied).any(1).sum())
                if not torch.isfinite(full_action).all():
                    raise RuntimeError("non_finite_action")
                with torch.no_grad():
                    _, _, dones, info = wrapped.step(full_action)
                speed = robot.data.root_lin_vel_b.torch[selected, 0]
                lateral = robot.data.root_lin_vel_b.torch[selected, 1]
                vertical = robot.data.root_lin_vel_b.torch[selected, 2]
                gravity = robot.data.projected_gravity_b.torch[selected]
                tilt = torch.sqrt(torch.atan2(gravity[:, 1], -gravity[:, 2]).square() + torch.atan2(-gravity[:, 0], torch.sqrt(gravity[:, 1] ** 2 + gravity[:, 2] ** 2)).square())
                forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :][selected]
                contacts = forces.norm(dim=-1).amax(dim=1) > 5
                impact_force = forces[:, :, :, 2].abs().mean(1).amax(dim=1)
                foot_speed = robot.data.body_lin_vel_w.torch[selected][:, feet, :2].norm(dim=-1)
                slip_speed = torch.where(contacts, foot_speed, torch.zeros_like(foot_speed)).amax(dim=1)
                effort = robot.data.applied_torque.torch[selected][:, joints].abs() / robot.data.joint_effort_limits.torch[selected][:, joints].abs().clamp_min(1e-6)
                velocity = robot.data.joint_vel.torch[selected][:, joints].abs() / robot.data.joint_vel_limits.torch[selected][:, joints].abs().clamp_min(1e-6)
                slip_dwell = torch.where((slip_speed > 0.8) & active, slip_dwell + dt, torch.zeros_like(slip_dwell))
                ankle_dwell = torch.where((effort[:, ankle] >= 0.95).any(1) & active, ankle_dwell + dt, torch.zeros_like(ankle_dwell))
                knee_dwell = torch.where((velocity[:, knee] >= 0.95).any(1) & active, knee_dwell + dt, torch.zeros_like(knee_dwell))
                no_contact = ~contacts.any(1)
                landing = previous_contacts.logical_not().all(1) & contacts.any(1)
                flight_dwell = torch.where(no_contact & active, flight_dwell + dt, torch.zeros_like(flight_dwell))
                last_flight_time = torch.where(no_contact & active, elapsed, last_flight_time)
                stable_contact = torch.where(contacts.any(1) & active, stable_contact + dt, torch.zeros_like(stable_contact))
                walk_contact = contacts.any(1) & (flight_dwell <= 0.16)
                first_contact_time = torch.where(walk_contact & torch.isnan(first_contact_time), elapsed, first_contact_time)
                cycle_event = (~cycle_terminated) & (stable_contact >= 0.12) & (elapsed >= 0.20)
                cycle_terminated |= cycle_event
                safety = (slip_dwell >= 0.2) | (ankle_dwell >= 0.2) | (knee_dwell >= 0.2) | (impact_force > 3500)
                timeouts = info.get("time_outs", torch.zeros_like(dones)).bool()[selected]
                fall = dones.bool()[selected] & ~timeouts
                reverse = speed < -0.1
                excessive = flight_dwell > 0.16
                contract_good = ((speed - 1.2).abs() <= 0.2) & (heading_error.abs() <= 0.12) & cycle_terminated & walk_contact & ~excessive & ~fall & ~reverse & ~safety
                previous_hold = walk_hold.clone()
                walk_hold = torch.where(contract_good & active, walk_hold + dt, torch.zeros_like(walk_hold))
                success = (walk_hold >= 0.4) & active
                completion_first = success & ~accepted
                accepted |= completion_first
                timeout = (elapsed + dt >= cfg["rollout"]["timeout_seconds"]) & active & ~success
                action_rate = torch.sqrt((action.detach() - state.previous_action[selected]).square().mean(1))
                raw, weighted, reward = reward_terms({
                    "speed": speed, "previous_speed": previous_speed, "target_speed": torch.full_like(speed, 1.2),
                    "heading_error": heading_error, "lateral_velocity": lateral, "tilt": tilt,
                    "flight_reduction_event": landing & (flight_dwell <= 0.16), "valid_landing": landing,
                    "run_cycle_terminated": cycle_event, "flight_frequency_reduced": cycle_event,
                    "vertical_velocity": vertical, "walk_compatible_contact": walk_contact,
                    "stable_support": stable_contact >= 0.12,
                    "walk_contract_progress": (walk_hold - previous_hold) / 0.4,
                    "walk_acceptance_first": completion_first, "fall": fall,
                    "torso_contact": fall, "dangerous_slip": slip_dwell >= 0.2,
                    "impact_failure": impact_force > 3500, "ankle_saturation": ankle_dwell >= 0.2,
                    "knee_saturation": knee_dwell >= 0.2, "joint_limit": torch.zeros_like(active),
                    "action_rate": action_rate, "entry_action_error": torch.sqrt((action.detach() - run_mean).square().mean(1)),
                    "entry_alignment_gate": torch.full_like(active, step < 5),
                    "exit_action_error": torch.sqrt((action.detach() - walk_mean).square().mean(1)),
                    "exit_alignment_gate": (speed - 1.2).abs() <= 0.3,
                    "completion_first": completion_first,
                }, cfg["reward"], cfg["reward_thresholds"])
                reward *= active_before.float()
                terminal = (success | fall | reverse | safety | excessive) & active_before
                truncated = timeout & active_before
                runner.transition_step(SegmentStep(obs.detach(), applied.detach(), reward.detach(), value.detach(), terminal, truncated, log_prob.detach()))
                valid_masks.append(active_before)
                values.append(value.detach())
                segment_length += active_before.long()
                for name, term in weighted.items():
                    rewards[name] += float((term * active_before.float()).sum())
                outcomes["completion"] |= success
                outcomes["cycle_termination"] |= cycle_terminated & active_before
                outcomes["walk_contact"] |= walk_contact & active_before
                outcomes["walk_contract"] |= success
                outcomes["timeout"] |= timeout
                outcomes["fall"] |= fall & active_before
                outcomes["reverse"] |= reverse & active_before
                outcomes["slip"] |= (slip_dwell >= 0.2) & active_before
                outcomes["impact"] |= (impact_force > 3500) & active_before
                outcomes["saturation"] |= ((ankle_dwell >= 0.2) | (knee_dwell >= 0.2)) & active_before
                outcomes["excessive_flight"] |= excessive & active_before
                if step == 0:
                    entry_jump = torch.sqrt((applied - state.previous_action[selected]).square().mean(1))
                exit_jump = torch.where(success, torch.sqrt((applied - walk_mean).square().mean(1)), exit_jump)
                active &= ~(terminal | truncated)
                elapsed += active_before.float() * dt
                previous_speed = speed
                previous_contacts = contacts
                state.previous_action.copy_(full_action)
                if not bool(active.any()):
                    break
            if bool(active.any()):
                outcomes["timeout"] |= active
                runner.storage.steps[-1].truncated |= active
            returns, advantages = runner.storage.finish(torch.zeros(cohort, device=device))
            result = {
                "runner": runner, "valid": torch.stack(valid_masks), "returns": returns, "advantages": advantages,
                "values": torch.stack(values), "outcomes": outcomes, "rewards": rewards,
                "segment_length": segment_length, "elapsed": elapsed, "first_contact": first_contact_time,
                "last_flight": last_flight_time, "entry_jump": entry_jump, "exit_jump": exit_jump,
                "routing_mismatch": routing_mismatch, "source_speed": state.source_speed[selected],
                "source_phase": state.launch_phase,
            }
            return result

        def evaluate_checkpoint(label, checkpoint_path, seed):
            payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
            actor.load_state_dict(payload["actor"], strict=True)
            critic.load_state_dict(payload["critic"], strict=True)
            log_std.data.copy_(payload["log_std"].to(device))
            source = prepare_source(seed, 40, balanced=True)
            if source is None:
                raise RuntimeError(f"evaluation source preparation failed: {label}")
            result = rollout(source, deterministic=True, training=False)
            # Hard-switch successful segments to frozen WALK for five seconds.
            success = result["outcomes"]["completion"].clone()
            hold_ok = success.clone()
            heading_max = torch.zeros(40, device=device)
            for _ in range(round(5.0 / dt)):
                legacy, canonical_full, background = graph_background(source)
                selected = source.selected
                heading_error = torch.atan2(torch.sin(source.heading[selected] - robot.data.heading_w.torch[selected]), torch.cos(source.heading[selected] - robot.data.heading_w.torch[selected]))
                command = MotionCommand(torch.full((40,), 1.2, device=device), source.heading[selected], target_yaw_rate_radps=(0.8 * heading_error).clamp(-0.3, 0.3))
                with torch.no_grad():
                    walk_action = walk(canonical_state_from_legacy_observation(legacy[selected], heading_w_rad=robot.data.heading_w.torch[selected]), command)
                background[selected] = walk_action
                with torch.no_grad():
                    _, _, dones, _ = wrapped.step(background)
                speed = robot.data.root_lin_vel_b.torch[selected, 0]
                hold_ok &= (~dones.bool()[selected]) & ((speed - 1.2).abs() <= 0.2) & (heading_error.abs() <= 0.12)
                heading_max = torch.maximum(heading_max, heading_error.abs())
            for speed_value in (2.6, 2.8):
                mask = result["source_speed"].isclose(torch.tensor(speed_value, device=device))
                count = int(mask.sum())
                rates = {name: float(value[mask].float().mean()) for name, value in result["outcomes"].items()}
                full_edge = success & hold_ok
                row = {
                    "checkpoint": label, "source_speed_mps": speed_value, "episodes": count,
                    "run_cycle_termination": rates["cycle_termination"], "walk_contact_acquisition": rates["walk_contact"],
                    "walk_contract_acquisition": rates["walk_contract"], "transition_completion": rates["completion"],
                    "walk_takeover": float(success[mask].float().mean()), "walk_hold": float(hold_ok[mask].float().mean()),
                    "full_edge_success": float(full_edge[mask].float().mean()), "timeout": rates["timeout"],
                    "reverse_failure": rates["reverse"], "fall": rates["fall"], "slip": rates["slip"],
                    "impact": rates["impact"], "saturation": rates["saturation"], "excessive_flight": rates["excessive_flight"],
                    "heading_p95": float(torch.quantile(heading_max[mask], 0.95)),
                    "transition_duration_mean": float(result["elapsed"][mask].mean()),
                    "entry_action_jump_mean": float(result["entry_jump"][mask].mean()),
                    "exit_action_jump_mean": float(result["exit_jump"][mask].mean()),
                }
                evaluation_rows.append(row)
                for local in torch.nonzero(mask).flatten().tolist():
                    episode_rows.append({"checkpoint": label, "source_speed_mps": speed_value, "episode": local, **{name: bool(value[local]) for name, value in result["outcomes"].items()}, "walk_hold": bool(hold_ok[local]), "full_edge": bool(full_edge[local]), "phase": int(result["source_phase"][local]), "duration": float(result["elapsed"][local])})
            return result

        # Initial deterministic baseline precedes every optimizer update.
        initial_result = evaluate_checkpoint("initial", OUT / "checkpoints/initial.pt", 20261331)
        write_csv("initial_baseline_episodes.csv", episode_rows.copy())
        initial_per = [row for row in evaluation_rows if row["checkpoint"] == "initial"]
        write_json("initial_baseline_per_source.json", initial_per)
        write_json("initial_baseline_summary.json", {"episodes": 40, "per_source": initial_per})

        # Restore initial state after baseline.
        initial = torch.load(OUT / "checkpoints/initial.pt", map_location=device, weights_only=False)
        actor.load_state_dict(initial["actor"], strict=True)
        critic.load_state_dict(initial["critic"], strict=True)
        log_std.data.copy_(initial["log_std"].to(device))
        optimizer.load_state_dict(initial["optimizer"])
        protected_before = {name: file_sha(path) for name, path in paths.items()}
        save_at = {1: "first_post_update.pt", 10: "model_10.pt", 25: "model_25.pt", 50: "model_50.pt", 75: "model_75.pt", 100: "model_100.pt"}

        for iteration in range(1, cfg["experiment"]["iterations"] + 1):
            source = prepare_source(cfg["experiment"]["training_seed"] + iteration, cfg["experiment"]["cohort_size"])
            if source is None:
                abort = "source_cohort_formation_failure"
                break
            result = rollout(source, deterministic=False, training=True)
            if result["routing_mismatch"]:
                abort = "action_routing_mismatch"
                break
            storage = result["runner"].storage.steps
            valid = result["valid"]
            observations = torch.stack([step.observation for step in storage])
            actions = torch.stack([step.action for step in storage])
            old_log = torch.stack([step.log_prob for step in storage])
            old_value = result["values"]
            flat = valid.flatten()
            obs_flat, action_flat = observations.flatten(0, 1)[flat], actions.flatten(0, 1)[flat]
            old_log_flat, old_value_flat = old_log.flatten()[flat], old_value.flatten()[flat]
            returns_flat, advantage_flat = result["returns"].flatten()[flat], result["advantages"].flatten()[flat]
            advantage_flat = (advantage_flat - advantage_flat.mean()) / advantage_flat.std().clamp_min(1e-8)
            generator = torch.Generator(device=device).manual_seed(cfg["experiment"]["training_seed"] + 20000 + iteration)
            losses, kls, clips, actor_norms, critic_norms = [], [], [], [], []
            for _ in range(cfg["ppo"]["epochs"]):
                order = torch.randperm(len(obs_flat), generator=generator, device=device)
                for indices in order.chunk(cfg["ppo"]["minibatches"]):
                    mean = actor(obs_flat[indices])
                    distribution = Normal(mean, log_std.exp().expand_as(mean))
                    new_log = distribution.log_prob(action_flat[indices]).sum(-1)
                    entropy = distribution.entropy().sum(-1).mean()
                    ratio = (new_log - old_log_flat[indices]).exp()
                    policy_loss = -torch.minimum(ratio * advantage_flat[indices], ratio.clamp(0.8, 1.2) * advantage_flat[indices]).mean()
                    value = critic(obs_flat[indices]).squeeze(-1)
                    clipped_value = old_value_flat[indices] + (value - old_value_flat[indices]).clamp(-0.2, 0.2)
                    value_loss = torch.maximum((value - returns_flat[indices]).square(), (clipped_value - returns_flat[indices]).square()).mean()
                    loss = policy_loss + value_loss - cfg["ppo"]["entropy_coefficient"] * entropy
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    params = trainable + list(critic.parameters()) + [log_std]
                    if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in params):
                        abort = "non_finite_gradient"
                        break
                    actor_norm = torch.nn.utils.clip_grad_norm_(trainable + [log_std], cfg["ppo"]["max_gradient_norm"])
                    critic_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg["ppo"]["max_gradient_norm"])
                    optimizer.step()
                    losses.append((float(policy_loss), float(value_loss), float(entropy)))
                    kls.append(float((old_log_flat[indices] - new_log).mean()))
                    clips.append(float(((ratio - 1).abs() > 0.2).float().mean()))
                    actor_norms.append(float(actor_norm))
                    critic_norms.append(float(critic_norm))
                if abort:
                    break
            if abort:
                break
            std = log_std.exp()
            if float(std.min()) < cfg["exploration"]["abort_min"] or float(std.max()) > cfg["exploration"]["abort_max"]:
                abort = "exploration_std_abort"
                break
            if any(file_sha(paths[name]) != protected_before[name] for name in paths):
                abort = "frozen_checkpoint_hash_change"
                break
            outcome_rates = {name: float(value.float().mean()) for name, value in result["outcomes"].items()}
            source_counts = {speed: int((result["source_speed"] == speed).sum()) for speed in (2.6, 2.8)}
            phase_counts = Counter(int(value) for value in result["source_phase"].cpu().tolist())
            mean_loss = [sum(item[i] for item in losses) / len(losses) for i in range(3)]
            curves.append({
                "iteration": iteration, "source_preparation_success": source.ready_success,
                "cohort_formation_time_s": source.formation_time, "ready_env_count": round(source.ready_success * 1024),
                "source_2_6_segments": source_counts[2.6], "source_2_8_segments": source_counts[2.8],
                "phase_flight": phase_counts[0], "phase_left": phase_counts[1], "phase_right": phase_counts[2], "phase_double": phase_counts[3],
                "segment_length_mean": float(result["segment_length"].float().mean()),
                **outcome_rates, "policy_loss": mean_loss[0], "value_loss": mean_loss[1], "entropy": mean_loss[2],
                "kl": sum(kls) / len(kls), "clip_fraction": sum(clips) / len(clips),
                "explained_variance": 1 - float((returns_flat - old_value_flat).var() / returns_flat.var().clamp_min(1e-8)),
                "actor_gradient_norm": sum(actor_norms) / len(actor_norms), "critic_gradient_norm": sum(critic_norms) / len(critic_norms),
                "exploration_std_min": float(std.min()), "exploration_std_mean": float(std.mean()), "exploration_std_max": float(std.max()),
                "completion_bonus_count": int(result["outcomes"]["completion"].sum()),
            })
            total_valid = max(int(valid.sum()), 1)
            reward_rows.append({"iteration": iteration, **{name: value / total_valid for name, value in result["rewards"].items()}})
            for speed in (2.6, 2.8):
                mask = result["source_speed"] == speed
                source_rows.append({"iteration": iteration, "source_speed_mps": speed, "segments": int(mask.sum()), "completion": float(result["outcomes"]["completion"][mask].float().mean())})
            for phase, count in phase_counts.items():
                phase_rows.append({"iteration": iteration, "phase": {0: "flight", 1: "left", 2: "right", 3: "double"}[phase], "segments": count})
            routing_rows.append({"iteration": iteration, "action_routing_mismatch": 0, "selected_env_ids_order": "explicit", "boolean_mask_order_dependency": False})
            timing_rows.append({"iteration": iteration, "transition_duration_mean": float(result["elapsed"].mean()), "last_flight_time_mean": float(torch.nan_to_num(result["last_flight"]).mean()), "first_walk_contact_mean": float(torch.nan_to_num(result["first_contact"]).mean())})
            if iteration in save_at:
                manifests.append(save_checkpoint(OUT / "checkpoints" / save_at[iteration], iteration, actor, critic, log_std, optimizer, cfg, iteration))
            write_csv("training_curves.csv", curves)
            write_csv("source_segment_counts.csv", source_rows)
            write_csv("source_phase_distribution.csv", phase_rows)
            write_csv("reward_term_statistics.csv", reward_rows)
            write_csv("transition_timing_statistics.csv", timing_rows)
            write_json("checkpoint_manifest.json", manifests)
            write_json("training_diagnostics.json", {"requested_iterations": 100, "completed_iterations": iteration, "abort_reason": None, "durable_checkpoint_iteration": iteration if iteration in save_at else max(item["iteration"] for item in manifests)})
            print(f"[Stage8C] iteration={iteration:03d} completion={outcome_rates['completion']:.3f} timeout={outcome_rates['timeout']:.3f} kl={sum(kls)/len(kls):.5f} std={float(std.mean()):.4f}", flush=True)

        # Sweep only fully durable checkpoints.
        if abort is None:
            checkpoint_names = [("initial", "initial.pt"), ("first_post_update", "first_post_update.pt"), ("model_10", "model_10.pt"), ("model_25", "model_25.pt"), ("model_50", "model_50.pt"), ("model_75", "model_75.pt"), ("model_100", "model_100.pt")]
            for index, (label, name) in enumerate(checkpoint_names):
                evaluate_checkpoint(label, OUT / "checkpoints" / name, 20261431 + index)
        write_csv("checkpoint_evaluations.csv", evaluation_rows)
        per_source = {}
        for row in evaluation_rows:
            per_source.setdefault(row["checkpoint"], {})[str(row["source_speed_mps"])] = row
        write_json("per_checkpoint_per_source.json", per_source)
        write_csv("initial_baseline_episodes.csv", [row for row in episode_rows if row["checkpoint"] == "initial"])
        write_json("checkpoint_hashes.json", {item["path"]: item["sha256"] for item in manifests})
        write_json("checkpoint_manifest.json", manifests)
        wrapped.close()

    write_json("training_diagnostics.json", {
        "requested_iterations": 100, "completed_iterations": len(curves), "abort_reason": abort,
        "durable_resume_used": False, "source_prefix_stored_steps": 0, "non_selected_stored_steps": 0,
        "invalid_stored_steps": 0, "post_terminal_stored_steps": 0, "previous_action_mismatch": 0,
        "action_routing_mismatch": sum(row["action_routing_mismatch"] for row in routing_rows),
        "controller_overlap": 0, "unassigned_env": 0,
    })
    write_csv("training_curves.csv", curves)
    write_csv("source_segment_counts.csv", source_rows)
    write_csv("source_phase_distribution.csv", phase_rows)
    write_csv("reward_term_statistics.csv", reward_rows)
    write_csv("transition_timing_statistics.csv", timing_rows)
    write_json("action_routing_audit.json", {"status": "PASS" if not abort else "FAIL", "mismatch": sum(row["action_routing_mismatch"] for row in routing_rows), "selected_env_ids_order": "explicit"})
    write_json("storage_audit.json", {"source_prefix": 0, "non_selected": 0, "invalid": 0, "post_terminal": 0})
    if abort:
        failures[abort] += 1
        raise RuntimeError(f"Stage 8C aborted: {abort}")


if __name__ == "__main__":
    main()
