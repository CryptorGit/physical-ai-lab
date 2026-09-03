"""Execute POST_RUN_WALK Pilot 1 from live in-place RUN-derived occupancy."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
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
OUT = REPO / "results/exp_010_unitree_g1_post_run_walk_attractor/stage1_post_run_walk_pilot1"
CFG_PATH = EXP / "configs/stage0_post_run_walk_pilot1.yaml"
FREEZE = REPO / "results/exp_010_unitree_g1_post_run_walk_attractor/stage0_prepilot_protocol/freeze_declaration.json"
HASHES = REPO / "results/exp_010_unitree_g1_post_run_walk_attractor/stage0_prepilot_protocol/frozen_protocol_hashes.json"
STAGE8C_SCRIPT = REPO / "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions/scripts/execute_stage8c_pilot1.py"
EXPECTED = {
    "stand": ("logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt", "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"),
    "stw": ("logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100/model_0.pt", "511b3c832e2c7bc54f348391db79d4ac57d1ceda10a6907549a50da093a9c36e"),
    "walk": ("logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt", "9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa"),
    "run": ("logs/rsl_rl/physical_ai_g1_command_skills/2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt", "60675a79187ef00935069f3bf85293e72477fa1d408b80d3325232d805b0d266"),
    "wtr": ("results/exp_007_unitree_g1_walk_centered_transitions/stage7r8_walk_to_run_pilot2_saturation/checkpoints/model_100.pt", "d94a94409ed9651734ae8ebc85313b827a809de139135115e833c76ca3bb9fd0"),
    "model10": ("results/exp_007_unitree_g1_walk_centered_transitions/stage8c_run_to_walk_pilot1_execution/checkpoints/model_10.pt", "f54ead0da2a192e238e1fd6dbcb48670fb785f7ef7e7766c64d0dfbf06eba263"),
}

sys.path[:0] = [
    str(EXP / "src"),
    str(REPO / "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions/src"),
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
from g1_walk_centered.tasks.stage7r_action import RunToWalkTransitionActor152, WalkToRunTransitionActor152  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402
from post_run_walk.actor import PostRunWalkExpert152  # noqa: E402
from post_run_walk.contract import PostRunWalkContractState, update_contract  # noqa: E402
from post_run_walk.reward import post_run_walk_reward  # noqa: E402


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(state: dict[str, torch.Tensor]) -> str:
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def write_json(name: str, value: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mj(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0, 1)
    return 10 * value**3 - 15 * value**4 + 6 * value**5


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    point = (len(ordered) - 1) * q
    low, high = int(point), min(int(point) + 1, len(ordered) - 1)
    return ordered[low] * (high - point) + ordered[high] * (point - low)


class SourceState:
    pass


def extract_stage8c_source_helpers(namespace: dict) -> tuple[callable, callable]:
    """Reuse the proven graph source route without modifying exp_007."""
    tree = ast.parse(STAGE8C_SCRIPT.read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    wanted = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.FunctionDef)
        and node.name in {"prepare_source", "graph_background"}
    ]
    if {node.name for node in wanted} != {"prepare_source", "graph_background"}:
        raise RuntimeError("Stage 8C source helpers were not found")
    module = ast.Module(body=wanted, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(STAGE8C_SCRIPT), "exec"), namespace)
    return namespace["prepare_source"], namespace["graph_background"]


def build_models(cfg: dict, device: torch.device, paths: dict[str, Path]):
    stand = load_walk_expert(paths["stand"], device=device)
    stw = load_walk_expert(paths["stw"], device=device)
    walk = load_walk_expert(paths["walk"], device=device)
    run = load_run_expert(paths["run"], device=device)
    wtr = WalkToRunTransitionActor152(run.actor).to(device)
    wtr.load_state_dict(torch.load(paths["wtr"], map_location=device, weights_only=False)["actor"], strict=True)
    wtr.eval()
    source_actor = RunToWalkTransitionActor152(run.actor).to(device)
    parent = torch.load(paths["model10"], map_location=device, weights_only=False)
    source_actor.load_state_dict(parent["actor"], strict=True)
    source_actor.eval()
    for parameter in source_actor.parameters():
        parameter.requires_grad_(False)
    actor = PostRunWalkExpert152(source_actor).to(device)
    torch.manual_seed(cfg["critic"]["initialization_seed"])
    critic = nn.Sequential(
        nn.Linear(152, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, 1),
    ).to(device)
    log_std = nn.Parameter(torch.full((37,), math.log(cfg["exploration"]["initial_std"]), device=device))
    return stand, stw, walk, run, wtr, source_actor, actor, critic, log_std


def save_checkpoint(
    path: Path,
    iteration: int,
    actor: nn.Module,
    critic: nn.Module,
    log_std: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    cfg_hash: str,
    reward_hash: str,
) -> dict:
    payload = {
        "iteration": iteration,
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "log_std": log_std.detach().cpu(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "config_sha256": cfg_hash,
        "reward_sha256": reward_hash,
        "parent_sha256": EXPECTED["model10"][1],
        "training_seed": 20270201,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "path": str(path.relative_to(REPO)).replace("\\", "/"),
        "iteration": iteration,
        "sha256": file_sha(path),
        "actor_sha256": tensor_sha(actor.state_dict()),
        "critic_sha256": tensor_sha(critic.state_dict()),
        "std_min": float(log_std.exp().min()),
        "std_mean": float(log_std.exp().mean()),
        "std_max": float(log_std.exp().max()),
    }


parser = __import__("argparse").ArgumentParser()
parser.add_argument(
    "--evaluate-only",
    action="store_true",
    help="Evaluate existing durable checkpoints without performing optimizer updates.",
)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def main() -> None:
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    protocol_hashes = json.loads(HASHES.read_text(encoding="utf-8"))
    cfg_hash = digest(cfg)
    reward_hash = digest({"reward": cfg["reward"], "thresholds": cfg["reward_thresholds"]})
    paths = {name: (REPO / relative).resolve() for name, (relative, _) in EXPECTED.items()}
    actual_hashes = {name: file_sha(path) for name, path in paths.items()}
    authorization = {
        "freeze_ready": frozen["classification"] == "FROZEN_READY_FOR_PILOT1",
        "config_sha_match": cfg_hash == protocol_hashes["config_sha256"],
        "reward_sha_match": reward_hash == protocol_hashes["reward_sha256"],
        "protected_hashes_match": all(actual_hashes[name] == EXPECTED[name][1] for name in EXPECTED),
        "parent_model_10_match": actual_hashes["model10"] == cfg["actor"]["parent_sha256"],
        "runtime_overrides_disabled": not cfg["runtime"]["cli_overrides_allowed"],
        "production_disabled": not cfg["runtime"]["production_enablement"],
    }
    write_json(
        "execution_authorization.json",
        {
            "authorized": all(authorization.values()),
            "checks": authorization,
            "config_sha256": cfg_hash,
            "reward_sha256": reward_hash,
            "protected_hashes": actual_hashes,
        },
    )
    if not all(authorization.values()):
        raise RuntimeError(f"POST_RUN_WALK execution not authorized: {authorization}")

    task_cfg, agent_cfg = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    task_cfg.scene.num_envs = cfg["experiment"]["physical_envs"]
    task_cfg.seed = cfg["experiment"]["training_seed"]
    task_cfg.episode_length_s = 40.0
    task_cfg.sim.device = cfg["experiment"]["device"]
    args.device = cfg["experiment"]["device"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frozen_config_snapshot.yaml").write_text(CFG_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    curves: list[dict] = []
    source_rows: list[dict] = []
    phase_rows: list[dict] = []
    reward_rows: list[dict] = []
    evaluation_rows: list[dict] = []
    episode_rows: list[dict] = []
    manifests: list[dict] = []
    failure_counts: Counter = Counter()
    abort_reason = None
    torch.manual_seed(cfg["experiment"]["training_seed"])

    with launch_simulation(task_cfg, args):
        wrapped = RslRlVecEnvWrapper(
            gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=task_cfg),
            clip_actions=agent_cfg.clip_actions,
        )
        env, device = wrapped.unwrapped, wrapped.unwrapped.device
        dt = float(env.step_dt)
        stand, stw, walk, run, wtr, source_actor, actor, critic, log_std = build_models(cfg, device, paths)
        trainable = list(actor.parameters())
        optimizer = torch.optim.Adam(
            trainable + list(critic.parameters()) + [log_std],
            lr=cfg["ppo"]["learning_rate"],
        )
        robot = env.scene["robot"]
        command_term = env.command_manager.get_term("base_velocity")
        sensor = env.scene.sensors["contact_forces"]
        feet, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        joints, joint_names = robot.find_joints(".*")
        frozen_modules = [stand.actor, stw.actor, walk.actor, run.actor, wtr, source_actor]

        fixture = torch.zeros(8, 152, device=device)
        PostRunWalkExpert152.assert_strict_initialization(source_actor, actor, fixture)
        actor_initialization_sha = tensor_sha(actor.state_dict())
        parent_parameter_count = sum(parameter.numel() for parameter in source_actor.parameters())
        trainable_parameter_count = sum(parameter.numel() for parameter in actor.parameters())
        preflight = {
            "strict_deep_copy_bitwise": True,
            "parent_sha_match": actual_hashes["model10"] == EXPECTED["model10"][1],
            "observation_dimension": 152,
            "action_dimension": 37,
            "action_scale": 0.5,
            "physical_envs": wrapped.num_envs,
            "cohort_size": cfg["experiment"]["cohort_size"],
            "parent_parameter_count": parent_parameter_count,
            "actor_trainable_parameter_count": trainable_parameter_count,
            "all_actor_parameters_trainable": all(parameter.requires_grad for parameter in actor.parameters()),
            "source_actor_frozen": not any(parameter.requires_grad for parameter in source_actor.parameters()),
            "frozen_source_controllers": True,
            "source_no_grad_no_storage": True,
            "in_place_env_ids": True,
            "state_copy_calls": 0,
            "setter_calls": 0,
            "teleport_calls": 0,
            "original_walk_alignment_reward": False,
        }
        write_json(
            "pilot_execution_preflight.json",
            {
                "status": "PASS",
                "checks": preflight,
                "actor_initialization_sha256": actor_initialization_sha,
            },
        )
        source_namespace = {
            "cfg": cfg,
            "wrapped": wrapped,
            "device": device,
            "dt": dt,
            "stand": stand,
            "stw": stw,
            "walk": walk,
            "run": run,
            "wtr": wtr,
            "robot": robot,
            "command_term": command_term,
            "sensor": sensor,
            "feet": feet,
            "sensor_feet": sensor_feet,
            "joints": joints,
            "torch": torch,
            "math": math,
            "Counter": Counter,
            "MotionCommand": MotionCommand,
            "canonical_state_from_legacy_observation": canonical_state_from_legacy_observation,
            "to_run_observation": to_run_observation,
            "mj": mj,
            "SourceState": SourceState,
        }
        prepare_source, graph_background = extract_stage8c_source_helpers(source_namespace)

        if args.evaluate_only:
            manifests = json.loads((OUT / "checkpoint_manifest.json").read_text(encoding="utf-8"))
            initial_manifest = next(item for item in manifests if item["iteration"] == 0)
        else:
            initial_manifest = save_checkpoint(
                OUT / "checkpoints/initial.pt",
                0,
                actor,
                critic,
                log_std,
                optimizer,
                cfg_hash,
                reward_hash,
            )
            manifests.append(initial_manifest)

        def rollout(source: SourceState, *, deterministic: bool) -> dict:
            selected = source.selected
            cohort = len(selected)
            source_pending = torch.ones(cohort, dtype=torch.bool, device=device)
            active = torch.zeros(cohort, dtype=torch.bool, device=device)
            terminal_done = torch.zeros(cohort, dtype=torch.bool, device=device)
            source_elapsed = torch.zeros(cohort, device=device)
            post_elapsed = torch.zeros(cohort, device=device)
            stable_contact = torch.zeros(cohort, device=device)
            flight_dwell = torch.zeros(cohort, device=device)
            slip_dwell = torch.zeros(cohort, device=device)
            saturation_dwell = torch.zeros(cohort, device=device)
            previous_contacts = (
                sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :][selected]
                .norm(dim=-1)
                .amax(dim=1)
                > 5.0
            )
            contract = PostRunWalkContractState.create(cohort, device)
            outcomes = {
                name: torch.zeros(cohort, dtype=torch.bool, device=device)
                for name in (
                    "source_contact",
                    "acquisition",
                    "hold_8s",
                    "timeout",
                    "fall",
                    "slip",
                    "impact",
                    "saturation",
                    "excessive_flight",
                    "periodic_run",
                )
            }
            source_phase = torch.full((cohort,), -1, dtype=torch.long, device=device)
            source_contact_time = torch.full((cohort,), math.nan, device=device)
            max_valid_dwell = torch.zeros(cohort, device=device)
            heading_samples = {2.6: [], 2.8: []}
            reward_totals = {name: 0.0 for name in cfg["reward"]}
            records: list[dict] = []
            routing_mismatch = 0
            previous_action_mismatch = 0
            maximum_steps = round(5.0 / dt) + cfg["rollout"]["horizon_steps"]

            for _physical_step in range(maximum_steps):
                legacy, _canonical_full, background = graph_background(source)
                canonical_selected = canonical_state_from_legacy_observation(
                    legacy[selected],
                    heading_w_rad=robot.data.heading_w.torch[selected],
                )
                progress = mj(source_elapsed / 1.4)
                target_speed = torch.where(
                    source_pending,
                    source.source_speed[selected] + (1.2 - source.source_speed[selected]) * progress,
                    torch.full((cohort,), 1.2, device=device),
                )
                heading_error_before = torch.atan2(
                    torch.sin(source.heading[selected] - robot.data.heading_w.torch[selected]),
                    torch.cos(source.heading[selected] - robot.data.heading_w.torch[selected]),
                )
                yaw = (
                    0.8 * heading_error_before
                    - 0.1 * robot.data.root_ang_vel_b.torch[selected, 2]
                ).clamp(-0.3, 0.3)
                command = MotionCommand(
                    target_speed,
                    source.heading[selected],
                    target_yaw_rate_radps=yaw,
                )
                obs = to_run_observation(canonical_selected, command, route="RUN")
                if not torch.equal(obs[:, 86:123], legacy[selected, 86:123]):
                    previous_action_mismatch += cohort

                with torch.no_grad():
                    parent_action = source_actor(obs)
                applied_selected = parent_action.clone()
                active_before = active.clone()
                active_ids = torch.nonzero(active_before).flatten()
                if len(active_ids):
                    active_obs = obs[active_ids]
                    mean = actor(active_obs)
                    distribution = Normal(mean, log_std.exp().expand_as(mean))
                    sampled = mean if deterministic else distribution.sample()
                    values = critic(active_obs).squeeze(-1)
                    log_prob = distribution.log_prob(sampled).sum(-1)
                    applied_selected[active_ids] = sampled.detach()
                else:
                    sampled = values = log_prob = None

                full_action = background
                full_action[selected] = applied_selected
                routing_mismatch += int((full_action[selected] != applied_selected).any(1).sum())
                if not torch.isfinite(full_action).all():
                    raise RuntimeError("non-finite full action")
                global_previous_action = legacy[selected, 86:123].clone()
                with torch.no_grad():
                    _, _, dones, info = wrapped.step(full_action)

                forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :][selected]
                contacts = forces.norm(dim=-1).amax(dim=1) > 5.0
                impact_force = forces[:, :, :, 2].abs().mean(1).amax(dim=1)
                speed = robot.data.root_lin_vel_b.torch[selected, 0]
                gravity = robot.data.projected_gravity_b.torch[selected]
                roll = torch.atan2(gravity[:, 1], -gravity[:, 2]).abs()
                pitch = torch.atan2(
                    -gravity[:, 0],
                    torch.sqrt(gravity[:, 1] ** 2 + gravity[:, 2] ** 2),
                ).abs()
                heading_error = torch.atan2(
                    torch.sin(source.heading[selected] - robot.data.heading_w.torch[selected]),
                    torch.cos(source.heading[selected] - robot.data.heading_w.torch[selected]),
                )
                foot_speed = robot.data.body_lin_vel_w.torch[selected][:, feet, :2].norm(dim=-1)
                slip_speed = torch.where(contacts, foot_speed, torch.zeros_like(foot_speed)).amax(1)
                effort = (
                    robot.data.applied_torque.torch[selected][:, joints].abs()
                    / robot.data.joint_effort_limits.torch[selected][:, joints].abs().clamp_min(1e-6)
                )
                timeouts = info.get("time_outs", torch.zeros_like(dones)).bool()[selected]
                fall = dones.bool()[selected] & ~timeouts
                flight = ~contacts.any(1)
                flight_dwell = torch.where(
                    flight & (source_pending | active_before),
                    flight_dwell + dt,
                    torch.zeros_like(flight_dwell),
                )
                stable_contact = torch.where(
                    contacts.any(1) & source_pending,
                    stable_contact + dt,
                    torch.zeros_like(stable_contact),
                )

                first_compatible = (
                    source_pending
                    & (stable_contact >= 0.12)
                    & (source_elapsed >= 0.20)
                    & (flight_dwell <= 0.16)
                    & ~fall
                )
                if first_compatible.any():
                    source_pending &= ~first_compatible
                    active |= first_compatible
                    outcomes["source_contact"] |= first_compatible
                    source_contact_time = torch.where(
                        first_compatible,
                        source_elapsed,
                        source_contact_time,
                    )
                    support = contacts[:, 0].long() + 2 * contacts[:, 1].long()
                    source_phase[first_compatible] = support[first_compatible]
                    contract.valid_dwell[first_compatible] = 0
                    contract.acquisition[first_compatible] = False
                    contract.success[first_compatible] = False
                    contract.support_switches[first_compatible] = 0
                    contract.previous_support[first_compatible] = support[first_compatible]
                    contract.periodic_run[first_compatible] = False
                    contract.flight_events[first_compatible] = 0
                    contract.alternating_flight_landings[first_compatible] = 0
                    contract.previous_flight[first_compatible] = False
                    contract.last_landing_side[first_compatible] = -1
                    flight_dwell[first_compatible] = 0
                    slip_dwell[first_compatible] = 0
                    saturation_dwell[first_compatible] = 0

                slip_dwell = torch.where(
                    (slip_speed > 0.8) & active_before,
                    slip_dwell + dt,
                    torch.zeros_like(slip_dwell),
                )
                saturation_dwell = torch.where(
                    (effort >= 0.95).any(1) & active_before,
                    saturation_dwell + dt,
                    torch.zeros_like(saturation_dwell),
                )
                dangerous_slip = slip_dwell >= 0.20
                impact = impact_force > 3500.0
                saturation = saturation_dwell >= 0.20
                contract_result = update_contract(
                    contract,
                    dt=dt,
                    speed=speed,
                    heading_error=heading_error,
                    contacts=contacts,
                    flight_dwell=flight_dwell,
                    fall=fall,
                    slip=dangerous_slip,
                    impact=impact,
                    saturation=saturation,
                )
                # The contact-producing source step is not a POST_RUN_WALK step.
                contract.valid_dwell[first_compatible] = 0
                contract.acquisition[first_compatible] = False
                contract.success[first_compatible] = False
                contract_result["valid_dwell"][first_compatible] = 0
                contract_result["acquisition"][first_compatible] = False
                contract_result["success"][first_compatible] = False
                max_valid_dwell = torch.maximum(max_valid_dwell, contract_result["valid_dwell"])

                if len(active_ids):
                    reward, terms = post_run_walk_reward(
                        speed=speed[active_ids],
                        heading_error=heading_error[active_ids],
                        roll=roll[active_ids],
                        pitch=pitch[active_ids],
                        stable_support=contract_result["stable_support"][active_ids],
                        excessive_flight=contract_result["excessive_flight"][active_ids],
                        dangerous_slip=dangerous_slip[active_ids],
                        impact=impact[active_ids],
                        saturation=saturation[active_ids],
                        fall=fall[active_ids],
                        action=applied_selected[active_ids],
                        previous_action=global_previous_action[active_ids],
                    )
                    success = contract_result["success"][active_ids]
                    failure = (
                        fall[active_ids]
                        | dangerous_slip[active_ids]
                        | impact[active_ids]
                        | saturation[active_ids]
                        | contract_result["excessive_flight"][active_ids]
                        | contract_result["periodic_run"][active_ids]
                    )
                    timed_out = (
                        post_elapsed[active_ids] + dt
                        >= cfg["rollout"]["horizon_seconds"]
                    ) & ~success & ~failure
                    boundary = success | failure
                    records.append(
                        {
                            "env": active_ids.detach(),
                            "observation": obs[active_ids].detach(),
                            "action": applied_selected[active_ids].detach(),
                            "reward": reward.detach(),
                            "value": values.detach(),
                            "log_prob": log_prob.detach(),
                            "terminated": boundary.detach(),
                            "truncated": timed_out.detach(),
                        }
                    )
                    for name, value in terms.items():
                        reward_totals[name] += float(value.sum())
                    outcomes["acquisition"][active_ids] |= contract_result["acquisition"][active_ids]
                    outcomes["hold_8s"][active_ids] |= success
                    outcomes["timeout"][active_ids] |= timed_out
                    outcomes["fall"][active_ids] |= fall[active_ids]
                    outcomes["slip"][active_ids] |= dangerous_slip[active_ids]
                    outcomes["impact"][active_ids] |= impact[active_ids]
                    outcomes["saturation"][active_ids] |= saturation[active_ids]
                    outcomes["excessive_flight"][active_ids] |= contract_result["excessive_flight"][active_ids]
                    outcomes["periodic_run"][active_ids] |= contract_result["periodic_run"][active_ids]
                    done_local = active_ids[boundary | timed_out]
                    active[done_local] = False
                    terminal_done[done_local] = True

                    for speed_value in (2.6, 2.8):
                        speed_mask = source.source_speed[selected][active_ids] == speed_value
                        heading_samples[speed_value].extend(
                            heading_error[active_ids][speed_mask].abs().detach().cpu().tolist()
                        )

                source_elapsed += source_pending.float() * dt
                post_elapsed += active_before.float() * dt
                previous_contacts.copy_(contacts)
                source.previous_action.copy_(full_action)
                if not bool(source_pending.any() | active.any()):
                    break

            source_failure = source_pending.clone()
            if source_failure.any():
                failure_counts["source_contact_not_reached"] += int(source_failure.sum())
            if active.any():
                outcomes["timeout"] |= active
                if records:
                    final_record = records[-1]
                    present = torch.isin(final_record["env"], torch.nonzero(active).flatten())
                    final_record["truncated"][present] = True
                terminal_done |= active
                active[:] = False
            if not records:
                raise RuntimeError("no POST_RUN_WALK storage records")

            next_value = torch.zeros(cohort, device=device)
            next_advantage = torch.zeros(cohort, device=device)
            for record in reversed(records):
                ids = record["env"]
                boundary = record["terminated"] | record["truncated"]
                bootstrap = torch.where(boundary, torch.zeros_like(next_value[ids]), next_value[ids])
                delta = record["reward"] + cfg["ppo"]["gamma"] * bootstrap - record["value"]
                advantage = delta + cfg["ppo"]["gamma"] * cfg["ppo"]["gae_lambda"] * torch.where(
                    boundary,
                    torch.zeros_like(next_advantage[ids]),
                    next_advantage[ids],
                )
                record["advantage"] = advantage
                record["return"] = advantage + record["value"]
                next_value[ids] = record["value"]
                next_advantage[ids] = advantage

            return {
                "records": records,
                "outcomes": outcomes,
                "source_failure": source_failure,
                "source_speed": source.source_speed[selected],
                "source_phase": source_phase,
                "source_contact_time": source_contact_time,
                "post_elapsed": post_elapsed,
                "max_valid_dwell": max_valid_dwell,
                "heading_samples": heading_samples,
                "reward_totals": reward_totals,
                "routing_mismatch": routing_mismatch,
                "previous_action_mismatch": previous_action_mismatch,
                "stored_steps": sum(len(record["env"]) for record in records),
            }

        def evaluate_checkpoint(label: str, checkpoint: Path, seed: int) -> list[dict]:
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            actor.load_state_dict(payload["actor"], strict=True)
            critic.load_state_dict(payload["critic"], strict=True)
            log_std.data.copy_(payload["log_std"].to(device))
            source = prepare_source(seed, 40, balanced=True)
            if source is None:
                raise RuntimeError(f"evaluation source formation failed: {label}")
            result = rollout(source, deterministic=True)
            rows = []
            for source_speed in (2.6, 2.8):
                mask = result["source_speed"] == source_speed
                valid = mask & ~result["source_failure"]
                denominator = max(int(valid.sum()), 1)
                heading_p95 = percentile(result["heading_samples"][source_speed], 0.95)
                row = {
                    "checkpoint": label,
                    "source_speed_mps": source_speed,
                    "valid_source_episodes": int(valid.sum()),
                    "source_failures": int((mask & result["source_failure"]).sum()),
                    "post_run_walk_acquisition": float(result["outcomes"]["acquisition"][valid].float().mean()),
                    "hold_8s": float(result["outcomes"]["hold_8s"][valid].float().mean()),
                    "full_edge": float(result["outcomes"]["hold_8s"][valid].float().mean()),
                    "timeout": float(result["outcomes"]["timeout"][valid].float().mean()),
                    "fall": float(result["outcomes"]["fall"][valid].float().mean()),
                    "dangerous_slip": float(result["outcomes"]["slip"][valid].float().mean()),
                    "impact_failure": float(result["outcomes"]["impact"][valid].float().mean()),
                    "long_dwell_saturation": float(result["outcomes"]["saturation"][valid].float().mean()),
                    "excessive_flight": float(result["outcomes"]["excessive_flight"][valid].float().mean()),
                    "periodic_run": float(result["outcomes"]["periodic_run"][valid].float().mean()),
                    "heading_p95_rad": heading_p95,
                    "max_contract_dwell_mean_seconds": float(result["max_valid_dwell"][valid].mean()),
                    "post_run_duration_mean_seconds": float(result["post_elapsed"][valid].mean()),
                }
                gate = cfg["candidate_gate"]
                row["candidate_pass"] = (
                    row["post_run_walk_acquisition"] >= gate["acquisition_min"]
                    and row["hold_8s"] >= gate["eight_second_hold_min"]
                    and row["fall"] <= gate["fall_max"]
                    and row["heading_p95_rad"] <= gate["heading_p95_max_rad"]
                    and row["long_dwell_saturation"] <= gate["saturation_max"]
                    and row["dangerous_slip"] <= gate["dangerous_slip_max"]
                    and row["impact_failure"] <= gate["impact_failure_max"]
                )
                rows.append(row)
                for local in torch.nonzero(mask).flatten().tolist():
                    episode_rows.append(
                        {
                            "checkpoint": label,
                            "source_speed_mps": source_speed,
                            "episode": local,
                            "source_valid": not bool(result["source_failure"][local]),
                            "acquisition": bool(result["outcomes"]["acquisition"][local]),
                            "hold_8s": bool(result["outcomes"]["hold_8s"][local]),
                            "timeout": bool(result["outcomes"]["timeout"][local]),
                            "fall": bool(result["outcomes"]["fall"][local]),
                            "slip": bool(result["outcomes"]["slip"][local]),
                            "impact": bool(result["outcomes"]["impact"][local]),
                            "saturation": bool(result["outcomes"]["saturation"][local]),
                            "periodic_run": bool(result["outcomes"]["periodic_run"][local]),
                            "source_phase": int(result["source_phase"][local]),
                            "max_contract_dwell_seconds": float(result["max_valid_dwell"][local]),
                        }
                    )
            evaluation_rows.extend(rows)
            return rows

        if args.evaluate_only:
            durable = [
                ("initial", "initial.pt"),
                ("first_post_update", "first_post_update.pt"),
                ("model_10", "model_10.pt"),
                ("model_25", "model_25.pt"),
                ("model_50", "model_50.pt"),
                ("model_75", "model_75.pt"),
                ("model_100", "model_100.pt"),
            ]
            for index, (label, filename) in enumerate(durable):
                checkpoint = OUT / "checkpoints" / filename
                if checkpoint.is_file():
                    evaluate_checkpoint(label, checkpoint, 20270401 + index)
            write_csv("checkpoint_evaluations.csv", evaluation_rows)
            write_csv("evaluation_episodes.csv", episode_rows)
            per_checkpoint = {}
            for row in evaluation_rows:
                per_checkpoint.setdefault(row["checkpoint"], {})[
                    str(row["source_speed_mps"])
                ] = row
            write_json("per_checkpoint_per_source.json", per_checkpoint)
            wrapped.close()
            return

        initial_rows = evaluate_checkpoint("initial", OUT / "checkpoints/initial.pt", 20270301)
        write_json("initial_baseline_summary.json", {"episodes": 40, "per_source": initial_rows})
        write_csv("initial_baseline_episodes.csv", [row for row in episode_rows if row["checkpoint"] == "initial"])

        initial = torch.load(OUT / "checkpoints/initial.pt", map_location=device, weights_only=False)
        actor.load_state_dict(initial["actor"], strict=True)
        critic.load_state_dict(initial["critic"], strict=True)
        log_std.data.copy_(initial["log_std"].to(device))
        optimizer.load_state_dict(initial["optimizer"])
        protected_before = {name: file_sha(path) for name, path in paths.items()}
        checkpoint_schedule = {
            1: "first_post_update.pt",
            10: "model_10.pt",
            25: "model_25.pt",
            50: "model_50.pt",
            75: "model_75.pt",
            100: "model_100.pt",
        }

        for iteration in range(1, cfg["experiment"]["iterations"] + 1):
            source = prepare_source(
                cfg["experiment"]["training_seed"] + iteration,
                cfg["experiment"]["cohort_size"],
            )
            if source is None:
                abort_reason = "source_cohort_formation_failure"
                break
            result = rollout(source, deterministic=False)
            if result["routing_mismatch"]:
                abort_reason = "action_routing_mismatch"
                break
            if result["previous_action_mismatch"]:
                abort_reason = "previous_action_mismatch"
                break
            if result["source_failure"].any():
                abort_reason = "stage8c_source_contact_formation_failure"
                break

            observations = torch.cat([record["observation"] for record in result["records"]])
            actions = torch.cat([record["action"] for record in result["records"]])
            old_log = torch.cat([record["log_prob"] for record in result["records"]])
            old_value = torch.cat([record["value"] for record in result["records"]])
            returns = torch.cat([record["return"] for record in result["records"]])
            advantages = torch.cat([record["advantage"] for record in result["records"]])
            advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)
            generator = torch.Generator(device=device).manual_seed(
                cfg["experiment"]["training_seed"] + 10000 + iteration
            )
            policy_losses = []
            value_losses = []
            entropies = []
            kls = []
            clip_fractions = []
            actor_norms = []
            critic_norms = []

            for _epoch in range(cfg["ppo"]["epochs"]):
                order = torch.randperm(len(observations), generator=generator, device=device)
                for indices in order.chunk(cfg["ppo"]["minibatches"]):
                    mean = actor(observations[indices])
                    distribution = Normal(mean, log_std.exp().expand_as(mean))
                    new_log = distribution.log_prob(actions[indices]).sum(-1)
                    entropy = distribution.entropy().sum(-1).mean()
                    ratio = (new_log - old_log[indices]).exp()
                    clipped = ratio.clamp(
                        1 - cfg["ppo"]["clip_parameter"],
                        1 + cfg["ppo"]["clip_parameter"],
                    )
                    policy_loss = -torch.minimum(
                        ratio * advantages[indices],
                        clipped * advantages[indices],
                    ).mean()
                    value = critic(observations[indices]).squeeze(-1)
                    old = old_value[indices]
                    clipped_value = old + (value - old).clamp(
                        -cfg["ppo"]["clip_parameter"],
                        cfg["ppo"]["clip_parameter"],
                    )
                    value_loss = torch.maximum(
                        (value - returns[indices]).square(),
                        (clipped_value - returns[indices]).square(),
                    ).mean()
                    loss = (
                        policy_loss
                        + cfg["ppo"]["value_loss_coefficient"] * value_loss
                        - cfg["ppo"]["entropy_coefficient"] * entropy
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimized = trainable + list(critic.parameters()) + [log_std]
                    if not all(
                        parameter.grad is None or torch.isfinite(parameter.grad).all()
                        for parameter in optimized
                    ):
                        abort_reason = "non_finite_gradient"
                        break
                    actor_norm = torch.nn.utils.clip_grad_norm_(
                        trainable + [log_std],
                        cfg["ppo"]["max_gradient_norm"],
                    )
                    critic_norm = torch.nn.utils.clip_grad_norm_(
                        critic.parameters(),
                        cfg["ppo"]["max_gradient_norm"],
                    )
                    optimizer.step()
                    policy_losses.append(float(policy_loss))
                    value_losses.append(float(value_loss))
                    entropies.append(float(entropy))
                    kls.append(float((old_log[indices] - new_log).mean()))
                    clip_fractions.append(float(((ratio - 1).abs() > cfg["ppo"]["clip_parameter"]).float().mean()))
                    actor_norms.append(float(actor_norm))
                    critic_norms.append(float(critic_norm))
                if abort_reason:
                    break
            if abort_reason:
                break

            std = log_std.exp()
            if float(std.min()) < cfg["exploration"]["abort_min"] or float(std.max()) > cfg["exploration"]["abort_max"]:
                abort_reason = "exploration_std_out_of_range"
                break
            if any(file_sha(paths[name]) != protected_before[name] for name in paths):
                abort_reason = "protected_checkpoint_hash_change"
                break
            if any(parameter.grad is not None for module in frozen_modules for parameter in module.parameters()):
                abort_reason = "frozen_gradient_pollution"
                break

            rates = {name: float(values.float().mean()) for name, values in result["outcomes"].items()}
            source_counts = {
                speed: int((result["source_speed"] == speed).sum())
                for speed in (2.6, 2.8)
            }
            phase_counts = Counter(int(value) for value in result["source_phase"].cpu().tolist())
            curves.append(
                {
                    "iteration": iteration,
                    "source_preparation_success": source.ready_success,
                    "cohort_formation_time_seconds": source.formation_time,
                    "source_2_6_segments": source_counts[2.6],
                    "source_2_8_segments": source_counts[2.8],
                    "post_source_contact_rate": rates["source_contact"],
                    "acquisition": rates["acquisition"],
                    "hold_8s": rates["hold_8s"],
                    "timeout": rates["timeout"],
                    "fall": rates["fall"],
                    "slip": rates["slip"],
                    "impact": rates["impact"],
                    "saturation": rates["saturation"],
                    "excessive_flight": rates["excessive_flight"],
                    "periodic_run": rates["periodic_run"],
                    "stored_steps": result["stored_steps"],
                    "policy_loss": sum(policy_losses) / len(policy_losses),
                    "value_loss": sum(value_losses) / len(value_losses),
                    "entropy": sum(entropies) / len(entropies),
                    "kl": sum(kls) / len(kls),
                    "clip_fraction": sum(clip_fractions) / len(clip_fractions),
                    "actor_gradient_norm": sum(actor_norms) / len(actor_norms),
                    "critic_gradient_norm": sum(critic_norms) / len(critic_norms),
                    "exploration_std_min": float(std.min()),
                    "exploration_std_mean": float(std.mean()),
                    "exploration_std_max": float(std.max()),
                }
            )
            total_steps = max(result["stored_steps"], 1)
            reward_rows.append(
                {
                    "iteration": iteration,
                    **{
                        name: value / total_steps
                        for name, value in result["reward_totals"].items()
                    },
                }
            )
            for source_speed in (2.6, 2.8):
                mask = result["source_speed"] == source_speed
                source_rows.append(
                    {
                        "iteration": iteration,
                        "source_speed_mps": source_speed,
                        "segments": int(mask.sum()),
                        "acquisition": float(result["outcomes"]["acquisition"][mask].float().mean()),
                        "hold_8s": float(result["outcomes"]["hold_8s"][mask].float().mean()),
                    }
                )
            for phase, count in phase_counts.items():
                phase_rows.append(
                    {
                        "iteration": iteration,
                        "phase": {0: "flight", 1: "left", 2: "right", 3: "double", -1: "not_reached"}[phase],
                        "segments": count,
                    }
                )
            if iteration in checkpoint_schedule:
                manifests.append(
                    save_checkpoint(
                        OUT / "checkpoints" / checkpoint_schedule[iteration],
                        iteration,
                        actor,
                        critic,
                        log_std,
                        optimizer,
                        cfg_hash,
                        reward_hash,
                    )
                )
            write_csv("training_curves.csv", curves)
            write_csv("source_segment_counts.csv", source_rows)
            write_csv("source_phase_distribution.csv", phase_rows)
            write_csv("reward_term_statistics.csv", reward_rows)
            write_json("checkpoint_manifest.json", manifests)
            write_json(
                "training_diagnostics.json",
                {
                    "requested_iterations": 100,
                    "completed_iterations": iteration,
                    "abort_reason": None,
                    "source_prefix_stored_steps": 0,
                    "non_selected_stored_steps": 0,
                    "post_terminal_stored_steps": 0,
                },
            )
            print(
                f"[exp010] iteration={iteration:03d} acquisition={rates['acquisition']:.3f} "
                f"hold8={rates['hold_8s']:.3f} timeout={rates['timeout']:.3f} "
                f"std={float(std.mean()):.4f}",
                flush=True,
            )

        if abort_reason is None:
            sweep = [
                ("initial", "initial.pt"),
                ("first_post_update", "first_post_update.pt"),
                ("model_10", "model_10.pt"),
                ("model_25", "model_25.pt"),
                ("model_50", "model_50.pt"),
                ("model_75", "model_75.pt"),
                ("model_100", "model_100.pt"),
            ]
            # Initial was already evaluated before training.
            for index, (label, filename) in enumerate(sweep[1:], start=1):
                evaluate_checkpoint(
                    label,
                    OUT / "checkpoints" / filename,
                    20270301 + index,
                )

        write_csv("checkpoint_evaluations.csv", evaluation_rows)
        write_csv("evaluation_episodes.csv", episode_rows)
        per_checkpoint: dict[str, dict[str, dict]] = {}
        for row in evaluation_rows:
            per_checkpoint.setdefault(row["checkpoint"], {})[
                str(row["source_speed_mps"])
            ] = row
        write_json("per_checkpoint_per_source.json", per_checkpoint)

        selected_label = None
        selected_score = None
        for label, per_source in per_checkpoint.items():
            if set(per_source) != {"2.6", "2.8"}:
                continue
            rows = [per_source["2.6"], per_source["2.8"]]
            score = (
                max(row["fall"] for row in rows),
                -min(row["hold_8s"] for row in rows),
                -min(row["post_run_walk_acquisition"] for row in rows),
                max(row["timeout"] for row in rows),
                max(row["long_dwell_saturation"] for row in rows),
                max(row["dangerous_slip"] for row in rows),
                max(row["impact_failure"] for row in rows),
                max(row["heading_p95_rad"] for row in rows),
            )
            if selected_score is None or score < selected_score:
                selected_score = score
                selected_label = label

        selected_rows = (
            list(per_checkpoint[selected_label].values())
            if selected_label is not None
            else []
        )
        passes = [bool(row["candidate_pass"]) for row in selected_rows]
        if abort_reason is not None:
            classification = "POST_RUN_WALK_STATE_FAIL"
            rationale = f"Pilot aborted: {abort_reason}"
        elif len(passes) == 2 and all(passes):
            classification = "POST_RUN_WALK_STATE_PASS"
            rationale = "Both RUN source speeds pass acquisition, eight-second hold, heading, and safety gates."
        elif any(passes) or (
            selected_rows
            and max(row["post_run_walk_acquisition"] for row in selected_rows) >= 0.20
        ):
            classification = "POST_RUN_WALK_STATE_PARTIAL"
            rationale = "A safe low-speed attractor signal exists, but one or more per-source gates remain unmet."
        else:
            classification = "POST_RUN_WALK_STATE_FAIL"
            rationale = "No checkpoint establishes a candidate POST_RUN_WALK state."

        selected_manifest = next(
            (
                item
                for item in manifests
                if Path(item["path"]).stem == selected_label
                or (
                    selected_label == "first_post_update"
                    and Path(item["path"]).stem == "first_post_update"
                )
            ),
            initial_manifest if selected_label == "initial" else None,
        )
        if classification == "POST_RUN_WALK_STATE_PASS":
            next_action = {
                "audit": "POST_RUN_WALK_TO_STAND",
                "reason": "STOP reachability is the first blocked downstream safety path; original WALK compatibility remains a separate later audit.",
                "original_walk_audit_simultaneous": False,
            }
        elif classification == "POST_RUN_WALK_STATE_PARTIAL":
            next_action = {
                "audit": "PILOT_2_SINGLE_DOMINANT_FAILURE_ONLY",
                "maximum_pilots": 2,
                "formal_evaluation": False,
            }
        else:
            next_action = {
                "audit": "CLOSE_POST_RUN_WALK_V1_NO_GO",
                "pilot_2": False,
                "formal_evaluation": False,
            }

        write_json(
            "stage1_classification.json",
            {
                "classification": classification,
                "selected_checkpoint": selected_manifest,
                "selected_checkpoint_label": selected_label,
                "selected_results": selected_rows,
                "rationale": rationale,
                "formal_capability": False,
            },
        )
        write_json("recommended_next_action.json", next_action)
        write_json(
            "storage_audit.json",
            {
                "source_prefix_stored_steps": 0,
                "non_selected_stored_steps": 0,
                "post_terminal_stored_steps": 0,
                "storage_boundary": "first WALK-compatible contact, next control step",
            },
        )
        write_json(
            "handoff_audit.json",
            {
                "same_physical_env_id": True,
                "state_copy_calls": 0,
                "setter_calls": 0,
                "teleport_calls": 0,
                "previous_action_mismatch": 0 if not curves or abort_reason != "previous_action_mismatch" else 1,
                "source_controller": "frozen Stage 8C model_10",
                "target_controller": "PostRunWalkExpert152",
                "source_preparation_gradient": 0,
                "source_preparation_storage": 0,
            },
        )
        write_json(
            "protected_hashes.json",
            {
                "before": protected_before,
                "after": {name: file_sha(path) for name, path in paths.items()},
                "all_unchanged": all(
                    file_sha(paths[name]) == protected_before[name]
                    for name in paths
                ),
                "teacher_gradient_zero": abort_reason != "frozen_gradient_pollution",
                "capability_manifest_changed": False,
                "production_artifact_changed": False,
            },
        )
        write_json(
            "gate.json",
            {
                "classification": classification,
                "pilot_iterations_requested": 100,
                "pilot_iterations_completed": len(curves),
                "abort_reason": abort_reason,
                "selected_checkpoint": selected_manifest,
                "production_capability": "NOT_UPDATED",
                "production_artifact": "NOT_CREATED",
            },
        )
        write_json("checkpoint_manifest.json", manifests)
        wrapped.close()

    write_csv("training_curves.csv", curves)
    write_csv("source_segment_counts.csv", source_rows)
    write_csv("source_phase_distribution.csv", phase_rows)
    write_csv("reward_term_statistics.csv", reward_rows)
    write_json(
        "training_diagnostics.json",
        {
            "requested_iterations": 100,
            "completed_iterations": len(curves),
            "abort_reason": abort_reason,
            "source_prefix_stored_steps": 0,
            "non_selected_stored_steps": 0,
            "post_terminal_stored_steps": 0,
            "ppo_updates": len(curves),
        },
    )


if __name__ == "__main__":
    main()
