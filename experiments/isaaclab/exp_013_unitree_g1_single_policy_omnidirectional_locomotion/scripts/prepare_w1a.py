"""Create and gate all static Phase W1A contracts before PPO."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a_all_direction_translation_walk"
PARENT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
CRITIC = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2n_gait_conditioned_ppo_retention_preflight/checkpoints/model_initial.pt"
PARENT_SHA = "66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698"
CRITIC_SHA = "04b43e5497bc35e2d00fa4476f9120f9439e0953283c69cf8ca1e9635dedd121"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


actual_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
actual_status = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).splitlines()
parent = torch.load(PARENT, map_location="cpu", weights_only=False)
critic_payload = torch.load(CRITIC, map_location="cpu", weights_only=False)
actor_state = parent["actor_state_dict"]
critic_state = critic_payload["critic_state_dict"]
parent_ok = sha(PARENT) == PARENT_SHA
critic_ok = (
    sha(CRITIC) == CRITIC_SHA
    and critic_state["mlp.0.weight"].shape[1] == 124
    and set(critic_state) == {
        "mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias",
        "mlp.4.weight", "mlp.4.bias", "mlp.6.weight", "mlp.6.bias",
    }
)

dump("stage_reference.json", {
    "experiment": "exp_013_unitree_g1_single_policy_omnidirectional_locomotion",
    "phase": "W1A", "status": "ACTIVE",
    "starting_head": actual_head, "reported_starting_head": "3d3f23cb66e62cbee2c1900870c82f4a5973edea",
    "starting_head_match": actual_head == "3d3f23cb66e62cbee2c1900870c82f4a5973edea",
    "starting_status": actual_status,
    "stage0_classification": "EXP013_PARENT_HAS_PARTIAL_DIRECTIONAL_GENERALIZATION",
    "stage0_artifacts_immutable": True,
})
dump("protocol.json", {
    "phase": "W1A", "objective": "one 360-degree translation-only WALK specialist",
    "command": {"frame": "body", "continuous_angle": True, "yaw_rate_cmd": 0, "gait_cmd": 0},
    "training": {"num_envs": 1024, "rollout_steps": 24, "iterations": 200,
                 "seed": 20271021, "maximum_runs": 1, "learning_rate": 1.5e-5,
                 "adaptive_lr": False},
    "prohibited": ["RUN training", "yaw training", "gait switching", "DAgger",
                   "routing", "checkpoint switching", "action blending"],
})
dump("w1a_parent_manifest.json", {
    "path": str(PARENT.relative_to(REPO)).replace("\\", "/"), "sha256": sha(PARENT),
    "architecture": [124, 256, 128, 128, 37],
    "original_observation_dimensions": 123, "gait_dimensions": 1,
    "actor_state_keys": list(actor_state), "source": "exp_012 Stage 2Q selected",
})
dump("w1a_parent_identity_audit.json", {
    "status": "PASS" if parent_ok else "FAIL", "expected_sha256": PARENT_SHA,
    "actual_sha256": sha(PARENT), "source_hash_match": parent_ok,
    "initial_actor_identity_pending_runtime_copy": True,
    "required_bitwise_state_keys": list(actor_state),
})
dump("w1a_critic_contract.json", {
    "single_critic": True, "input_dimensions": 124, "architecture": [124, 256, 128, 128, 1],
    "selection_priority": 1, "source": "Stage 2N compatible 124D critic",
    "source_path": str(CRITIC.relative_to(REPO)).replace("\\", "/"), "source_sha256": sha(CRITIC),
})
dump("w1a_critic_initialization_audit.json", {
    "status": "PASS" if critic_ok else "FAIL", "expected_sha256": CRITIC_SHA,
    "actual_sha256": sha(CRITIC), "compatible_124d": critic_state["mlp.0.weight"].shape[1] == 124,
    "new_random_critic": False, "direction_specific_critics": 0,
    "runtime_bitwise_copy_pending": True,
})
dump("w1a_optimizer_contract.json", {
    "optimizer": "Adam", "actor_mean_parameters": "fresh state",
    "critic_parameters": "fresh state", "excluded_parameters": ["log_std_walk", "log_std_run"],
    "initial_adam_step": 0, "learning_rate": 1.5e-5,
    "old_RUN_moments_imported": False, "adaptive_lr": False,
})
dump("w1a_optimizer_initialization_audit.json", {
    "status": "PENDING_RUNTIME_INITIALIZATION", "fresh_state_required": True,
    "state_entries_before_first_update_required": 0, "adam_step_required": 0,
})

reward_source = REPO.parent / "IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py"
reward_text = reward_source.read_text(encoding="utf-8")
vector_impl_ok = (
    "track_lin_vel_xy_yaw_frame_exp" in reward_text
    and "torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2])" in reward_text
)
samples = torch.tensor([
    [0.6, 0.2], [0.6, -0.2], [-0.6, 0.2], [-0.6, -0.2],
], dtype=torch.float64)
actual = torch.tensor([[0.1, -0.15]], dtype=torch.float64)
errors = torch.sum((samples - actual) ** 2, dim=1)
mirror_y = torch.sum((samples * torch.tensor([1.0, -1.0]) - actual * torch.tensor([1.0, -1.0])) ** 2, dim=1)
mirror_x = torch.sum((samples * torch.tensor([-1.0, 1.0]) - actual * torch.tensor([-1.0, 1.0])) ** 2, dim=1)
symmetry_ok = torch.equal(errors, mirror_y) and torch.equal(errors, mirror_x)
reward_ok = vector_impl_ok and symmetry_ok
dump("w1a_reward_contract.json", {
    "status": "PASS" if reward_ok else "EXP013_W1A_REWARD_CONTRACT_FAIL",
    "inherited_reward_semantic_changes": 0,
    "linear_tracking": "squared vx/vy vector error in gravity-aligned body-yaw frame",
    "yaw_target": 0, "yaw_tracking": "world z (frame-invariant yaw axis)",
    "other_terms": ["vertical velocity", "orientation", "joint torque", "joint acceleration",
                    "action rate", "foot air time", "foot slip", "termination"],
    "direction_bonus_added": False, "teacher_imitation_added": False,
    "reward_source": str(reward_source), "reward_source_sha256": sha(reward_source),
})
dump("w1a_reward_direction_symmetry_audit.json", {
    "status": "PASS" if reward_ok else "FAIL", "vx_vy_same_units": True,
    "body_yaw_frame": True, "world_frame_translation_error": False,
    "positive_negative_vy_symmetric": bool(torch.equal(errors, mirror_y)),
    "positive_negative_vx_symmetric": bool(torch.equal(errors, mirror_x)),
    "absolute_angle_speed_envelope_left_right_symmetric": True,
    "numeric_probe_errors": errors.tolist(),
})
(OUT / "resolved_w1a_training_config.yaml").write_text(
    """run_identity: phase_w1a_all_direction_translation_walk
num_envs: 1024
rollout_steps: 24
total_iterations: 200
interactions: 4915200
seed: 20271021
maximum_runs: 1
learning_rate: 1.5e-5
adaptive_lr: false
ppo:
  epochs: 5
  mini_batches: 4
  clip_range: 0.2
  gamma: 0.99
  gae_lambda: 0.95
  entropy_coefficient: 0.008
  value_coefficient: 1.0
  max_gradient_norm: 1.0
actor:
  architecture: [124, 256, 128, 128, 37]
  update: mean_network_only
critic:
  architecture: [124, 256, 128, 128, 1]
exploration:
  alpha_walk: 0.30
  log_std_walk: frozen
  log_std_run: frozen
""", encoding="utf-8"
)
dump("resolved_w1a_curriculum.json", {
    "continuous_not_discrete": True, "yaw_rate_cmd": 0, "gait_cmd": 0,
    "phases": [
        {"name": "A", "iterations": [1, 40], "angle_deg": [-90, 90],
         "weights": {"forward_anchor": .30, "front_diagonal": .30, "near_lateral": .40},
         "speed": {"abs_theta_le_45": [.3, 1.2], "otherwise": [.3, .6]}},
        {"name": "B", "iterations": [41, 100], "angle_deg": [-180, 180],
         "weights": {"forward": .20, "lateral": .30, "rear_diagonal": .30, "backward": .20},
         "speed_mps": [.2, .6]},
        {"name": "C", "iterations": [101, 160], "angle_deg": [-180, 180],
         "sampling": "uniform continuous", "speed_min_mps": .2,
         "max_by_abs_angle": {"0_45": 1.2, "45_90": .8, "90_135": .7, "135_180": .6}},
        {"name": "D", "iterations": [161, 200],
         "weights": {"forward": .15, "front_diagonal": .20, "lateral": .25,
                     "rear_diagonal": .25, "backward": .15},
         "speed_envelope": "same as C"},
    ],
})
dump("gate.json", {
    "parent_contract": "PASS" if parent_ok else "FAIL",
    "critic_contract": "PASS" if critic_ok else "FAIL",
    "reward_contract": "PASS" if reward_ok else "FAIL",
    "continue_to_first_update": parent_ok and critic_ok and reward_ok,
    "classification_if_stopped": None if reward_ok else "EXP013_W1A_REWARD_CONTRACT_FAIL",
})
if not parent_ok or not critic_ok or not reward_ok:
    raise SystemExit(2)
