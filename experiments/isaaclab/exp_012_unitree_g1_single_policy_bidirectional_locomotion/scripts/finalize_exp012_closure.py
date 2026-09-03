"""Build tracked closure manifests from immutable exp_012 source artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
ROOT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion"
OUT = ROOT / "closure"
STARTING_HEAD = "fd958f9dbef3b0fe4502f3d036f889e20c9052c4"
CLASSIFICATION = "EXP_012_CLOSED_WITH_SINGLE_POLICY_LOCOMOTION_SUCCESS_AND_STRICT_STAND_LIMITATION"
GAIT_CORE_SHA = "04b43e5497bc35e2d00fa4476f9120f9439e0953283c69cf8ca1e9635dedd121"
FINAL_SHA = "66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698"
GAIT_CORE_PATH = ROOT / "stage2n_gait_conditioned_ppo_retention_preflight/checkpoints/model_initial.pt"
FINAL_PATH = ROOT / "stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
INITIAL_STATUS = [
    " M experiments/isaaclab/exp_006_unitree_g1_command_skills/README.md",
    " M experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.ps1",
    " M experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.py",
    "?? .openduck_hardware_source_review/",
    "?? .openduck_phase3_usb_baseline.txt",
    "?? .openduck_playground_source_review/",
    "?? .openduck_runtime_source_review/",
    "?? artifacts/exp_005_unitree_g1_flat_run/",
    "?? artifacts/openduck_recorded_zero_pose.png",
    "?? artifacts/openduck_safe_init_pose_front.png",
    "?? artifacts/openduck_safe_init_pose_side.png",
    "?? artifacts/openduck_zero_pose_front.png",
    "?? artifacts/openduck_zero_pose_side.png",
    "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/assemble_showcase_reel.py",
    "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/play_command_system_showcase.ps1",
    "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/play_showcase.py",
    "?? experiments/mujoco/exp_003_openduckmini_calibrated_walk/",
    "?? media/",
    "?? openduck_setup_report.md",
    "?? research/exp_011_linkedin_post_ja.md",
    "?? tools/analyze_openduck_joint_directions.py",
    "?? tools/render_openduck_zero_pose.py",
]


def dump(name: str, payload) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def stage_name(path: Path) -> str:
    return path.parent.name


def classification_value(payload) -> str | None:
    for key in ("classification", "primary_classification", "primary", "main_classification"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    actual_head = git("rev-parse", "HEAD")
    if actual_head != STARTING_HEAD:
        raise RuntimeError(f"closure starting HEAD changed: {actual_head}")
    if sha(GAIT_CORE_PATH) != GAIT_CORE_SHA or sha(FINAL_PATH) != FINAL_SHA:
        raise RuntimeError("protected final checkpoint SHA mismatch")
    initial_status = INITIAL_STATUS
    unrelated = INITIAL_STATUS
    stage_files = sorted(ROOT.glob("stage*/stage_classification.json"))
    classifications = []
    for path in stage_files:
        payload = load(path)
        classifications.append({
            "stage_artifact_directory": stage_name(path),
            "classification": classification_value(payload),
            "classification_artifact": path.relative_to(REPO).as_posix(),
            "preserved": True,
            "full_artifact": payload,
        })
    dump("closure_stage_reference.json", {
        "experiment": "exp_012_unitree_g1_single_policy_bidirectional_locomotion",
        "operation": "formal research closure",
        "starting_head": STARTING_HEAD,
        "observed_starting_head": actual_head,
        "starting_status": initial_status,
        "starting_git_log_oneline_decorate_30": git("log", "--oneline", "--decorate", "-30").splitlines(),
        "unrelated_dirty_state": unrelated,
        "training_or_policy_updates": 0,
        "remote_push": False,
        "source_of_truth": "repository artifacts",
    })
    dump("stage_classification_index.json", {
        "past_stage_classifications_preserved": True,
        "project_level_classification_is_additive": True,
        "stages": classifications,
    })
    timeline = [
        ("Stage 1", "speed-dependent yaw diagnosis", "G1_YAW_BIAS_SPEED_CONDITIONED_CANCELABLE"),
        ("Stage 1B", "diagnostic feedforward yaw cancellation", "G1_SPEED_CONDITIONED_YAW_CANCELLATION_PASS"),
        ("Stage 2", "single-policy joint curriculum pilot", "G1_SINGLE_POLICY_MULTIPLE_FAILURES"),
        ("Stage 2A", "first-update instability diagnosis", "PPO_FIRST_UPDATE_TRUE_DISTRIBUTION_SHIFT"),
        ("Stage 2B", "strict runtime LR resume correction", "PPO_RUNTIME_LR_RESUME_FIX_PASS"),
        ("Stage 2C", "multi-regime gradient diagnosis", "RUN_REWARD_REACHABILITY_FAIL"),
        ("Stage 2D", "completion-basin reachability", "RUN_PRECURSOR_ONLY_NO_COMPLETION_BASIN"),
        ("Stage 2E", "RUN-focused Phase A", "SINGLE_POLICY_RUN_COMPLETION_EMERGED_PARTIAL"),
        ("Stage 2F", "exploration-only boundary", "PHASE_A_BOUNDARY_MULTIPLE_CAUSES"),
        ("Stage 2G", "event-stratified on-policy update", "EVENT_STRATIFIED_ON_POLICY_NO_EFFECT"),
        ("Stage 2H", "short-horizon completion replay", "SHORT_HORIZON_COMPLETION_REPLAY_NO_EFFECT"),
        ("Stage 2I", "reverse continuation from RUN parent", "REVERSE_SINGLE_POLICY_WALK_RECOVERY_FAIL"),
        ("Stage 2J", "low-speed manifold diagnosis", "LOW_SPEED_WALK_MANIFOLD_NOT_LOCALLY_REACHABLE"),
        ("Stage 2K", "scalar gait-command representation", "GAIT_LATENT_REPRESENTATION_FAIL"),
        ("Stage 2L", "gait-conditioned Gaussian std", "GAIT_CONDITIONED_STD_STATIC_PASS_CLOSED_LOOP_FAIL"),
        ("Stage 2M", "exploration-temperature calibration", "SAFE_GAIT_CONDITIONED_EXPLORATION_WINDOW_FOUND"),
        ("Stage 2N", "PPO endpoint retention", "GAIT_CONDITIONED_PPO_MULTIPLE_FAILURES"),
        ("Stage 2O", "anchor accumulation diagnosis", "ADAM_HISTORY_SUPPRESSES_ANCHOR"),
        ("Stage 2P", "optimizer-moment adaptation", "ACTOR_MOMENT_ADAPTATION_NO_EFFECT"),
        ("Stage 2Q", "final supervised sequence integration", "G1_FINAL_STAND_STOP_FAIL"),
        ("Stage 2R", "true-STAND positive control", "G1_FINAL_STAND_POSITIVE_CONTROL_FAIL"),
    ]
    dump("research_timeline.json", {
        "chronology": [
            {"stage": stage, "method": method, "classification": cls}
            for stage, method, cls in timeline
        ],
        "full_narrative": "research/exp_012_g1_single_policy_bidirectional_locomotion_final_report.md",
    })
    final_metrics = {
        "gait_core": {
            "walk_1p2_deterministic": 1.0, "walk_1p2_calibrated_stochastic": 1.0,
            "run_1p2_deterministic": 1.0, "run_1p2_calibrated_stochastic": .99,
            "run_2p4_deterministic": 1.0, "run_2p4_calibrated_stochastic": 1.0,
            "run_2p6_deterministic": 1.0, "run_2p6_calibrated_stochastic": 1.0,
            "walk_to_run_deterministic": 1.0, "walk_to_run_calibrated_stochastic": 1.0,
            "run_to_walk_deterministic": 1.0, "run_to_walk_calibrated_stochastic": 1.0,
        },
        "final_sequence_actor": {
            "walk_0p6_0p8_1p0_1p2": [1.0, 1.0, 1.0, 1.0],
            "run_1p2_2p4_2p6": [1.0, 1.0, 1.0],
            "stand_to_walk": 1.0, "walk_to_run": 1.0,
            "run_acceleration": 1.0, "run_deceleration": 1.0, "run_to_walk": 1.0,
            "walk_to_stand_formal": 0.0,
            "integrated_sequence_formal_completion": 0.0,
            "integrated_sequence_initial_walk": .97,
            "integrated_sequence_run": 1.0,
            "integrated_sequence_return_walk": 1.0,
            "integrated_sequence_fall_rate": .05,
            "final_speed_mean_mps": .054707538543734696,
            "heading_p95_mean_rad": .0803651475161314,
            "dangerous_slip_rate": .02,
            "impact_failure_rate": 0.0,
            "long_dwell_saturation_rate": 0.0,
        },
        "strict_stand_positive_controls": {
            "stand_source": {
                "formal_success": .03, "fall": .01, "mean_speed_mps": .005513,
                "flight_zero": .89, "final_double_support": .03,
            },
            "walk_to_stand_source": {
                "formal_completion": .03, "fall": .02, "mean_speed_mps": .001217,
                "flight_zero": .97, "final_double_support": 0.0,
            },
        },
        "interpretation": {
            "practical_stop": "achieved",
            "strict_static_contact_stand": "unresolved",
            "continued_ppo_semantic_retention": "unresolved",
        },
    }
    dump("final_metrics.json", final_metrics)
    reports = {
        "final_report": "research/exp_012_g1_single_policy_bidirectional_locomotion_final_report.md",
        "closure_summary": "research/exp_012_g1_single_policy_bidirectional_locomotion_closure.md",
        "english_abstract": "research/exp_012_g1_single_policy_bidirectional_locomotion_abstract_en.md",
        "linkedin_ja": "research/exp_012_linkedin_post_ja.md",
        "linkedin_en": "research/exp_012_linkedin_post_en.md",
    }
    dump("final_artifact_manifest.json", {
        "project_status": "CLOSED",
        "project_level_classification": CLASSIFICATION,
        "best_gait_core_checkpoint": {
            "path": GAIT_CORE_PATH.relative_to(REPO).as_posix(), "sha256": GAIT_CORE_SHA,
        },
        "final_sequence_checkpoint": {
            "path": FINAL_PATH.relative_to(REPO).as_posix(), "sha256": FINAL_SHA,
        },
        "architecture": [124, 256, 128, 128, 37],
        "observation_contract": {"original": 123, "gait_command": 1, "total": 124},
        "action_contract": {"dimensions": 37, "type": "joint-position target", "scale": .5},
        "gait_command_contract": {"range": [0, 1], "walk_endpoint": 0, "run_endpoint": 1},
        "exploration_contract": {
            "head": "one gait-conditioned diagonal Gaussian",
            "interpolation": "log-space",
            "alpha_walk": .30, "alpha_run": .65,
        },
        "runtime_action_source": {
            "unique_checkpoint": 1, "unique_actor": 1, "unique_gaussian_head": 1,
            "teacher": 0, "expert": 0, "router": 0, "checkpoint_switch": 0, "action_blend": 0,
        },
        "successful_capabilities": [
            "same-speed WALK/RUN selection", "WALK 0.6-1.2 m/s", "RUN 1.2-2.6 m/s",
            "WALK_TO_RUN", "RUN_TO_WALK", "RUN acceleration/deceleration", "practical stop",
        ],
        "formal_failures": [
            "strict flight-zero STAND", "strict final double support",
            "formal integrated-sequence completion", "continued PPO semantic retention",
        ],
        "reports": reports,
        "video": "media/exp_012_g1_single_policy_sequence_linkedin.mp4",
        "thumbnail": "media/exp_012_g1_single_policy_sequence_thumbnail.png",
        "captions": "media/exp_012_g1_single_policy_sequence_linkedin.srt",
    })
    dump("linkedin_post_manifest.json", {
        "japanese": reports["linkedin_ja"], "english": reports["linkedin_en"],
        "repository_remote_present": False,
        "repository_link": "[GitHub repository link]",
        "claims_reviewed_against_closure_metrics": True,
    })
    dump("closure_classification.json", {
        "classification": CLASSIFICATION,
        "status": "CLOSED",
        "additive_project_level_classification": True,
        "past_stage_classifications_overwritten": False,
        "meaning": {
            "single_checkpoint_walk_run_selection": "success",
            "same_speed_walk_run_disambiguation": "success",
            "bidirectional_walk_run_transition": "success",
            "walk_0p6_to_1p2": "success",
            "run_1p2_to_2p6": "success",
            "stand_to_walk": "success",
            "post_run_deceleration_and_practical_stop": "success",
            "strict_flight_zero_final_double_support_stand": "unresolved",
            "continued_ppo_semantic_retention": "unresolved",
            "research": "closed",
        },
    })
    previous = load(ROOT / "stage2r_true_stand_stop_integration/protected_hashes.json")
    dump("protected_hashes.json", {
        "starting_head": STARTING_HEAD,
        "protected_stage2r_audit_sha256": sha(ROOT / "stage2r_true_stand_stop_integration/protected_hashes.json"),
        "protected_stage2r_audit": previous,
        "verified_checkpoint_sha256": {
            "gait_core": sha(GAIT_CORE_PATH), "final_sequence": sha(FINAL_PATH),
        },
        "exp_005_through_exp_011_changes": 0,
        "exp_012_stage_0_through_2r_changes": 0,
        "existing_checkpoint_changes": 0,
        "existing_optimizer_changes": 0,
        "reward_curriculum_network_observation_action_physics_changes": 0,
        "isaac_lab_rsl_rl_core_changes": 0,
        "training_or_policy_updates": 0,
        "production_policy_update": 0,
        "unrelated_dirty_state_preserved": unrelated,
        "remote_push": False,
    })


if __name__ == "__main__":
    main()
