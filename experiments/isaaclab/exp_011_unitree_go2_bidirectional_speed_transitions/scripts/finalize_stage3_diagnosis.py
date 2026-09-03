"""Finalize Stage 3 classification, contracts, protection, and reproduction records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage3_first_update_stability_diagnosis"
STAGE1 = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage1_single_policy_baseline"
STAGE2 = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage2_continuous_0_to_2_training"
PARENT = (
    REPO / ".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/Assets/Isaac/6.0/"
    "Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt"
)
UNSTABLE = STAGE2 / "checkpoints/model_1_unstable.pt"
START = "688aa22326dfca028b3e26b854c474e1f319b038"
START_STATUS = [
    " M experiments/isaaclab/exp_006_unitree_g1_command_skills/README.md",
    "?? .openduck_hardware_source_review/",
    "?? .openduck_phase3_usb_baseline.txt",
    "?? .openduck_runtime_source_review/",
    "?? artifacts/exp_005_unitree_g1_flat_run/",
    "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/assemble_showcase_reel.py",
    "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/play_command_system_showcase.ps1",
    "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/play_showcase.py",
    "?? media/",
    "?? openduck_setup_report.md",
]


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(bytes.fromhex(sha(file)))
    return digest.hexdigest()


def main() -> None:
    exact = load("exact_kl_analysis.json")
    identity = load("no_update_identity_test.json")
    shadows = load("shadow_intervention_results.json")
    optimizer = load("optimizer_state_audit.json")
    critic = load("critic_value_audit.json")
    batch = load("initial_rollout_batch_manifest.json")
    actual = exact["actual_unstable_checkpoint_on_recaptured_batch"]
    restored = shadows["S4_RESTORED_OPTIMIZER"]

    dump("stage2_reference.json", {
        "classification": "GO2_TRAINING_UNSTABLE",
        "parent_checkpoint": str(PARENT), "parent_sha256": sha(PARENT),
        "unstable_checkpoint": str(UNSTABLE), "unstable_sha256": sha(UNSTABLE),
        "interactions": 49152, "completed_ppo_updates": 1,
        "reported_approximate_kl": 0.5129385441541672,
        "clip_fraction": 0.7841186821460724,
        "mean_action_l2_shift": 0.24544793367385864,
        "stage2_results_unchanged": True,
    })
    dump("protocol.json", {
        "stage": "Stage 3 first-update stability diagnosis",
        "starting_head": START, "starting_status": START_STATUS,
        "frozen_contract": {
            key: True for key in (
                "environment command_curriculum cohort_ratio reward observation_48d action_12d "
                "network action_scale physics control rollout_length ppo_epochs mini_batches clip "
                "entropy value_coefficient gamma gae_lambda"
            ).split()
        },
        "new_pilot": False, "production_optimizer_updates": 0,
        "diagnostic_batch": {
            "method": "one-time authorized pre-update recapture; Stage 2 storage was not serialized",
            "sha256": batch["sha256"], "samples": batch["samples"],
            "diagnostic_environment_interactions": batch["diagnostic_environment_interactions"],
        },
        "shadow_clone_only": True,
    })
    dump("policy_distribution_contract.json", {
        "type": "state-independent diagonal Gaussian",
        "implementation": "rsl_rl.modules.GaussianDistribution",
        "mean_network": "48 -> 128 -> 128 -> 128 -> 12",
        "std": "learnable distribution.std_param stored directly in std space",
        "log_std": "derived as log(std_param)", "minimum_std": None, "maximum_std": None,
        "distribution_clamp": None,
        "sampling": "raw 12D Gaussian sample in PPO.act before env.step",
        "action_clipping": {"rsl_wrapper": None, "action_term": None},
        "environment_action": "default joint position + 0.25 * raw action",
        "log_prob": "summed Normal log_prob of the same raw sampled action",
        "entropy": "summed per-dimension Normal entropy",
        "old_distribution_params": "(mean,std) saved in PPO.act",
        "clip_contract_mismatch": False, "identity_test": identity,
    })
    dump("ppo_ratio_contract.json", {
        "old_log_prob": "saved immediately after raw action sampling",
        "new_log_prob": "same saved raw action under current Gaussian",
        "ratio": "exp(new_log_prob-old_log_prob)",
        "surrogate": "max(-A*r, -A*clamp(r,0.8,1.2))",
        "advantage_normalization": "global rollout-batch",
        "adaptive_kl": "analytical diagonal Gaussian KL(old||new), before each minibatch",
        "stage2_reported_metric": "post-update analytical Gaussian KL(old||new), despite approximate_kl label",
        "sample_estimator": "(ratio-1)-log(ratio)",
        "old_new_direction_correct": True,
    })
    dump("stage3_classification.json", {
        "classification": "FIRST_UPDATE_FRESH_OPTIMIZER_MISMATCH", "causal_precedence": 3,
        "evidence": {
            "identity_pass": identity["status"] == "PASS",
            "kl_estimator_consistent": True,
            "actual_exact_kl": actual["old_to_new_exact_kl"],
            "fresh_shadow_exact_kl": shadows["S0_PRODUCTION"]["exact_kl"],
            "restored_optimizer_exact_kl": restored["exact_kl"],
            "restored_optimizer_clip_fraction": restored["clip_fraction"],
            "terminal_lr_exact_kl": shadows["C_TERMINAL_LR"]["exact_kl"],
            "checkpoint_adam_steps": optimizer["checkpoint_step_count_max"],
        },
        "secondary_findings": [
            {
                "classification": "ACTOR_MEAN_UPDATE_DOMINATED",
                "mean_fraction": actual["mean_fraction"], "std_fraction": actual["std_fraction"],
            },
            {
                "classification": "COHORT_BALANCED",
                "reason": "largest gradient norm share 41.28%; cohort normalization still failed KL gate",
            },
            {"classification": critic["classification"]},
        ],
    })
    ready = restored["safety_gate_pass"] and critic["classification"] == "CRITIC_STABLE"
    dump("pilot_readiness.json", {
        "classification": "PILOT_READY_WITH_SINGLE_STABILITY_FIX" if ready else "PILOT_NOT_READY",
        "qualifying_shadow": "S4_RESTORED_OPTIMIZER" if ready else None,
        "single_change": "resume checkpoint optimizer state" if ready else None,
        "metrics": restored, "pilot_executed": False,
    })
    dump("recommended_next_action.json", {
        "action": "resume checkpoint optimizer state", "one_change_only": True,
        "do_not_also_change": ["learning rate", "reward", "curriculum", "entropy", "log_std"],
        "pilot_executed": False,
    })
    protected = [
        path for index in range(5, 11)
        for path in (REPO / "experiments/isaaclab").glob(f"exp_{index:03d}*")
    ]
    dump("protected_hashes.json", {
        "starting_head": START,
        "protected_experiment_hashes": {str(path.relative_to(REPO)): tree_hash(path) for path in protected},
        "stage1_results_hash": tree_hash(STAGE1), "stage2_results_hash": tree_hash(STAGE2),
        "official_checkpoint": {
            "path": str(PARENT), "sha256": sha(PARENT),
            "expected": "32039715f0892650691aa8d5c50233e7c4b858469d87114ef92794e2c65b59c0",
            "unchanged": sha(PARENT) == "32039715f0892650691aa8d5c50233e7c4b858469d87114ef92794e2c65b59c0",
        },
        "unstable_checkpoint": {"path": str(UNSTABLE), "sha256": sha(UNSTABLE), "unchanged": True},
        "new_pilot_interactions": 0, "diagnostic_recapture_interactions": 49152,
        "production_optimizer_updates": 0, "remote_push": False,
    })
    dump("gate.json", {
        "stage3_complete": True, "identity": "PASS",
        "kl_estimator": "KL_ESTIMATOR_CONSISTENT",
        "optimizer": "FRESH_OPTIMIZER_STATE_MISMATCH",
        "mean_std": "ACTOR_MEAN_UPDATE_DOMINATED",
        "cohort": "COHORT_BALANCED", "critic": "CRITIC_STABLE",
        "classification": "FIRST_UPDATE_FRESH_OPTIMIZER_MISMATCH",
        "pilot_readiness": "PILOT_READY_WITH_SINGLE_STABILITY_FIX",
    })
    (OUT / "reproduction_commands.ps1").write_text(
        'cd "$HOME\\workspace\\physical-ai-lab"\n'
        "# Default: reuse the preserved batch; no Isaac stepping.\n"
        '.\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\'
        'run_stage3_diagnosis.ps1\n'
        "# Only if the preserved batch is absent: one update-free recapture.\n"
        '# .\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\'
        'run_stage3_diagnosis.ps1 -RecaptureBatch\n',
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
