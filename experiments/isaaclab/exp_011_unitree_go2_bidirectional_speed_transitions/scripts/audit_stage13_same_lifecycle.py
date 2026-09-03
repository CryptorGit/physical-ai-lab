"""Audit exposed state that survives an ordinary reset in one Isaac lifecycle."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage13_fresh_process_counterfactual_replay"
CHECKPOINT = REPO / (
    "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
    "stage11_tangential_slip_reduction/checkpoints/model_initial.pt"
)
SEED = 20273901
DT = 0.02

parser = __import__("argparse").ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app
sys.argv = [sys.argv[0]]

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: E402,F401
import go2_bidirectional.stage11_tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402


def digest_bytes(value):
    if isinstance(value, torch.Tensor):
        payload = value.detach().cpu().contiguous().numpy().tobytes()
    else:
        payload = repr(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def flat(value):
    if value is None:
        return None
    if hasattr(value, "torch"):
        value = value.torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


cfg, agent_cfg = resolve_task_config(
    "Isaac-Exp011-Go2-Tangential-Slip-v0", "rsl_rl_cfg_entry_point"
)
cfg.scene.num_envs = 1
cfg.seed = SEED
cfg.episode_length_s = 60.0
cfg.observations.policy.enable_corruption = False
cfg.events.base_external_force_torque = None
cfg.events.push_robot = None
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
    strict=True, map_location=runner.device,
)
runner.alg.actor.eval()
env = wrapped.unwrapped
robot = env.scene["robot"]
command = env.command_manager.get_term("base_velocity")
slip = env.reward_manager.get_term_cfg("go2_contact_tangential_slip").func
sensor = env.scene.sensors["stage11_contact"]
all_ids = torch.arange(1, device=env.device)


def configure(step):
    command.source_speed.fill_(0.2)
    command.target_speed.fill_(0.2)
    command.source_hold_s.zero_()
    command.elapsed_s.fill_(step * DT - DT)
    command._update_command()


def capture(label):
    _, _, _, _, count_raw, start_raw = sensor.contact_view.get_contact_data(
        dt=sensor._sim_physics_dt
    )
    counts = wp.to_torch(count_raw).reshape(1, 4, -1)[:, :, 0].long()
    starts = wp.to_torch(start_raw).reshape(1, 4, -1)[:, :, 0].long()
    action = getattr(env.action_manager, "action", None)
    prev_action = getattr(env.action_manager, "prev_action", None)
    air_time = getattr(sensor.data, "current_air_time", None)
    contact_time = getattr(sensor.data, "current_contact_time", None)
    fields = {
        "environment.episode_step_counter": flat(env.episode_length_buf),
        "environment.common_step_counter": flat(getattr(env, "common_step_counter", None)),
        "environment.reset_buffer": flat(getattr(env, "reset_buf", None)),
        "environment.termination_buffer": flat(getattr(env, "terminated_buf", None)),
        "environment.timeout_buffer": flat(getattr(env, "time_out_buf", None)),
        "command.elapsed_s": flat(command.elapsed_s),
        "command.source_speed": flat(command.source_speed),
        "command.target_speed": flat(command.target_speed),
        "command.source_hold_s": flat(command.source_hold_s),
        "command.cohort": flat(getattr(command, "cohort", None)),
        "command.pair_index": flat(getattr(command, "pair_index", None)),
        "heading.reference": flat(command.heading_reference),
        "heading.gate": flat(command.heading_gate),
        "heading.raw": flat(command.heading_raw),
        "heading.command": flat(command.heading_command),
        "heading.active": flat(command.heading_active),
        "heading.acquisition_age": flat(command.acquisition_age),
        "heading.activation_elapsed": flat(command.activation_elapsed),
        "heading.reference_frozen": flat(command.reference_frozen),
        "heading.history_index": flat(command._history_index),
        "heading.history_count": flat(command._history_count),
        "contact.boolean": flat(slip.last_normal_force > 5.0),
        "contact.age": flat(slip.contact_age),
        "contact.air_time": flat(air_time),
        "contact.contact_time": flat(contact_time),
        "contact.normal_force": flat(slip.last_normal_force),
        "contact.history_speed": flat(slip.last_foot_speed),
        "contact.count": flat(counts),
        "contact.start_index": flat(starts),
        "contact.sensor_timestamp": flat(getattr(sensor, "_timestamp", None)),
        "reward.raw_slip_score": flat(slip.last_raw_score),
        "action.current": flat(action),
        "action.previous": flat(prev_action),
        "robot.last_requested_action": flat(action),
        "robot.last_applied_joint_target": flat(
            getattr(robot.data, "joint_pos_target", None)
        ),
        "robot.applied_effort": flat(getattr(robot.data, "applied_torque", None)),
        "robot.joint_position": flat(robot.data.joint_pos),
        "robot.joint_velocity": flat(robot.data.joint_vel),
        "robot.root_state": flat(robot.data.root_state_w),
        "rng.python_sha256": digest_bytes(random.getstate()),
        "rng.numpy_sha256": digest_bytes(np.random.get_state()),
        "rng.torch_cpu_sha256": digest_bytes(torch.random.get_rng_state()),
        "rng.torch_cuda_sha256": digest_bytes(torch.cuda.get_rng_state_all()),
    }
    return {"label": label, "fields": fields}


def run_episode():
    for step in range(100):
        configure(step)
        observation = wrapped.get_observations()
        with torch.no_grad():
            action = runner.alg.actor(observation, stochastic_output=False)
        wrapped.step(action)


raw.reset(seed=SEED)
command._resample_command(all_ids)
slip.reset(all_ids)
post_reset_1 = capture("post_reset_1")
run_episode()
pre_reset_2 = capture("pre_reset_2")

# Ordinary reset in the same lifecycle: intentionally the Stage 12 failure mode.
env.seed(SEED)
wrapped.reset()
command._resample_command(all_ids)
slip.reset(all_ids)
post_reset_2 = capture("post_reset_2")
run_episode()
pre_reset_3 = capture("pre_reset_3")

rows = []
for comparison, left_snapshot, right_snapshot in (
    ("post_reset_1_vs_post_reset_2", post_reset_1, post_reset_2),
    ("equal_length_episode_1_vs_episode_2", pre_reset_2, pre_reset_3),
):
    for field in sorted(left_snapshot["fields"]):
        left = left_snapshot["fields"][field]
        right = right_snapshot["fields"][field]
        if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
            try:
                maximum = max(
                    (abs(float(a) - float(b)) for a, b in zip(left, right)),
                    default=0.0,
                )
            except (TypeError, ValueError):
                maximum = None
        else:
            maximum = None
        rows.append({
            "comparison": comparison,
            "field": field,
            "left": json.dumps(left, sort_keys=True),
            "right": json.dumps(right, sort_keys=True),
            "equal": left == right,
            "max_abs_difference": maximum,
        })
with (OUT / "same_lifecycle_reset_differences.csv").open(
    "w", newline="", encoding="utf-8"
) as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

audit = {
    "seed": SEED,
    "speed_m_s": 0.2,
    "episodes_in_same_lifecycle": 2,
    "snapshots": [post_reset_1, pre_reset_2, post_reset_2, pre_reset_3],
    "different_fields_after_equal_length_prefix": [
        row["field"] for row in rows
        if row["comparison"] == "equal_length_episode_1_vs_episode_2"
        and not row["equal"]
    ],
    "unavailable_or_unexposed": [
        "reset counter: not exposed by ManagerBasedEnv",
        "command RNG: global RNG only; no separate public generator",
        "event RNG: global RNG only; no separate public generator",
        "observation history internals: no stable public serialization contract",
        "reward manager term-internal states other than registered slip term",
        "event manager internal scheduling state",
        "curriculum manager internal state (no active terms)",
        "actuator internal controller cache beyond public targets/efforts",
        "RigidContactView friction-patch/manifold warm-start cache",
        "PhysX solver/contact warm-start cache",
    ],
    "physx_internal_state": "UNEXPOSED_PHYSX_INTERNAL_STATE",
    "formal_reset_implementation_changed": False,
    "formal_counterfactual_uses_same_lifecycle_reset": False,
}
(OUT / "same_lifecycle_hidden_state_audit.json").write_text(
    json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
wrapped.close()
simulation_app.close()
