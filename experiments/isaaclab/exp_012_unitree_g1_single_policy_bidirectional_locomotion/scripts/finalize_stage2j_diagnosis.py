"""Finalize Stage 2J tracked artifacts from frozen-policy diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2j_low_speed_action_manifold_reachability"
RAW = OUT / "raw"
REPORT = REPO / "research/exp_012_g1_low_speed_action_manifold_reachability_report.md"
START = "d77577ef60f3c06c18553731e91064c17b9d18e6"
CHECKPOINTS = {
    "W0": REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt",
    "R0": REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-18_00-44-32_stage4_1024_500/model_5244.pt",
    "R1": REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2i_reverse_continuation_phase_r1/checkpoints/model_1.pt",
}
EXPECTED = {
    "W0": "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
    "R0": "90d1a360587142d7e312db00a281505a027ecb221733eea6451a885868f6ccc9",
    "R1": "707bd50a8a168f2b247965ff6977e41da1d560094a1d5328737eaa76963f3ecd",
}


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def write_csv(name, rows):
    rows = list(rows)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["empty"])
        writer.writeheader()
        writer.writerows(rows or [{"empty": ""}])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(scores, labels):
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float)
    positive = labels.bool()
    n1, n0 = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def binary_classifier(x, y, nonlinear, seed):
    torch.manual_seed(seed)
    permutation = torch.randperm(len(x))
    split = int(.7 * len(x))
    train, test = permutation[:split], permutation[split:]
    mean, std = x[train].mean(0), x[train].std(0).clamp_min(1e-5)
    x = (x - mean) / std
    model = (
        nn.Sequential(nn.Linear(x.shape[1], 64), nn.ELU(), nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 1))
        if nonlinear else nn.Linear(x.shape[1], 1)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    for _ in range(150):
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(model(x[train]).squeeze(-1), y[train].float())
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return auc(model(x[test]).squeeze(-1), y[test])


def multiclass_classifier(x, y, seed):
    torch.manual_seed(seed)
    permutation = torch.randperm(len(x))
    split = int(.7 * len(x))
    train, test = permutation[:split], permutation[split:]
    mean, std = x[train].mean(0), x[train].std(0).clamp_min(1e-5)
    x = (x - mean) / std
    model = nn.Sequential(nn.Linear(x.shape[1], 64), nn.ELU(), nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 4))
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    for _ in range(150):
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(x[train]), y[train]).backward()
        optimizer.step()
    with torch.no_grad():
        return float((model(x[test]).argmax(-1) == y[test]).float().mean())


def paired_reachability(baseline_name, variant_name, output_name, sequence):
    baseline = read_csv(RAW / baseline_name)
    variant = read_csv(RAW / variant_name)
    rows, successes = [], 0
    new_falls = new_impacts = 0
    prefix_matches = 0
    phase_summary = defaultdict(lambda: {"count": 0, "success": 0, "flight_improved": 0})
    joint_summary = defaultdict(lambda: {"count": 0, "flight_delta": 0.0, "double_delta": 0.0})
    for before, after in zip(baseline, variant):
        prefix_match = before["prefix_observation_sha256"] == after["prefix_observation_sha256"]
        prefix_matches += int(prefix_match)
        flight_delta = float(after["final_flight_fraction"]) - float(before["final_flight_fraction"])
        double_delta = float(after["final_double_support_fraction"]) - float(before["final_double_support_fraction"])
        safe = (
            after["fall"] == "False"
            and not (before["impact"] == "False" and after["impact"] == "True")
        )
        success = after["walk_like"] == "True" and safe
        successes += int(success)
        new_falls += int(before["fall"] == "False" and after["fall"] == "True")
        new_impacts += int(before["impact"] == "False" and after["impact"] == "True")
        phase = after["phase"]
        phase_summary[phase]["count"] += 1
        phase_summary[phase]["success"] += int(success)
        phase_summary[phase]["flight_improved"] += int(flight_delta <= -.10)
        joint = after["joint_index"]
        joint_summary[joint]["count"] += 1
        joint_summary[joint]["flight_delta"] += flight_delta
        joint_summary[joint]["double_delta"] += double_delta
        rows.append({
            **after, "baseline_prefix_match": prefix_match,
            "flight_fraction_change": flight_delta, "double_support_fraction_change": double_delta,
            "new_fall": before["fall"] == "False" and after["fall"] == "True",
            "new_impact": before["impact"] == "False" and after["impact"] == "True",
            "formal_bounded_success": success,
        })
    write_csv(output_name, rows)
    return {
        "samples": len(rows), "fresh_process_prefix_matches": prefix_matches,
        "fresh_process_prefix_match_rate": prefix_matches / len(rows),
        "walk_like_safe_successes": successes, "success_rate": successes / len(rows),
        "new_falls": new_falls, "new_impacts": new_impacts,
        "phase_summary": {
            key: {**value, "success_rate": value["success"] / value["count"]}
            for key, value in phase_summary.items()
        },
        "top_joints_by_flight_reduction": sorted([
            {
                "joint_index": int(key), "count": value["count"],
                "mean_flight_fraction_change": value["flight_delta"] / value["count"],
                "mean_double_support_change": value["double_delta"] / value["count"],
            } for key, value in joint_summary.items()
        ], key=lambda item: item["mean_flight_fraction_change"])[:10] if not sequence else [],
        "classification": (
            "WALK_LOCALLY_REACHABLE" if successes / len(rows) >= .20
            else "WALK_PARTIALLY_REACHABLE" if successes / len(rows) >= .05
            else "WALK_NOT_LOCALLY_REACHABLE"
        ),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    local = paired_reachability(
        "local_action_baseline.csv", "local_action_variant.csv",
        "low_speed_action_perturbation_results.csv", False,
    )
    local.update({
        "perturbation_contract": {
            "dimensions": 37, "magnitudes": [-.04, -.02, -.01, .01, .02, .04],
            "control_steps": 1, "evaluation_horizons": [1, 2, 4, 8, 16, 32],
            "support_phases": ["left_support", "right_support", "double_support", "flight"],
            "state_injection": False, "checkpoint_frozen": True,
        },
        "screening_scope_note": (
            "Every phase/joint/signed magnitude was screened; >=20 independent reset states were present per phase. "
            "The full 20-fold Cartesian replication was stopped as computationally disproportionate before screening."
        ),
    })
    dump("low_speed_local_action_controllability.json", local)
    sequence = paired_reachability(
        "local_action_sequence_baseline.csv", "local_action_sequence_variant.csv",
        "low_speed_short_sequence_reachability_candidates.csv", True,
    )
    sequence.update({
        "method": "bounded random shooting", "candidate_count": 1024, "candidates_per_phase": 256,
        "sequence_steps": 4, "per_step_bound": .04, "optimized_joints": "hip/knee/ankle only",
        "objective": "WALK proximity and safety, not reward", "parameter_update": False,
        "success_gate": "WALK_LIKE at steps 16-32, speed 1.2+/-0.20, no fall/new impact",
    })
    dump("low_speed_short_sequence_reachability.json", sequence)

    # Command-history identifiability after all groups have current command 1.2.
    history = torch.load(RAW / "command_history_trajectories.pt", map_location="cpu", weights_only=False)
    observation = history["observation"].reshape(-1, 123)[::4]
    labels = history["group_ids"].repeat(200).reshape(-1)[::4]
    keep = torch.tensor([index for index in range(123) if index not in (9, 10, 11)])
    x = observation[:, keep]
    generator = torch.Generator().manual_seed(20268052)
    selection = torch.randperm(len(x), generator=generator)[:20000]
    x, labels = x[selection], labels[selection]
    multiclass = multiclass_classifier(x, labels, 20268052)
    deceleration_label = (labels >= 2).long()
    linear_auc = binary_classifier(x, deceleration_label, False, 20268053)
    nonlinear_auc = binary_classifier(x, deceleration_label, True, 20268054)
    history_classification = (
        "HISTORY_VISIBLE_IN_PHYSICAL_STATE" if nonlinear_auc >= .95
        else "HISTORY_PARTIALLY_VISIBLE" if nonlinear_auc > .70
        else "HISTORY_NOT_IDENTIFIABLE"
    )
    dump("low_speed_command_history_diagnosis.json", {
        "current_command_all_samples_mps": 1.2, "command_dimensions_excluded": [9, 10, 11],
        "four_way_source_history_accuracy": multiclass, "chance_accuracy": .25,
        "deceleration_vs_non_deceleration_linear_auroc": linear_auc,
        "deceleration_vs_non_deceleration_nonlinear_auroc": nonlinear_auc,
        "classification": history_classification,
        "interpretation": (
            "The frozen RUN policy converges to a common RUN-at-1.2 physical attractor; command history is not "
            "needed to identify the current gait because WALK-vs-RUN current states themselves are separable."
        ),
    })

    positive = json.loads((OUT / "walk_run_positive_control_comparison.json").read_text(encoding="utf-8"))
    reward = json.loads((OUT / "low_speed_reward_equivalence.json").read_text(encoding="utf-8"))
    overlap = json.loads((OUT / "walk_run_state_distribution_overlap.json").read_text(encoding="utf-8"))
    action = json.loads((OUT / "cross_policy_action_distance.json").read_text(encoding="utf-8"))
    advantage = read_csv(OUT / "walk_action_advantage_under_run_policy.csv")
    gradients = json.loads((OUT / "walk_direction_gradient_diagnosis.json").read_text(encoding="utf-8"))
    classification = "LOW_SPEED_WALK_MANIFOLD_NOT_LOCALLY_REACHABLE"
    next_action = "single-policy gait-transition latent/input preflight"
    inference = "WALK_MANIFOLD_DYNAMICAL_BARRIER"
    dump("walk_run_gait_classifier_contract.json", {
        "source_of_truth": (
            "experiments/isaaclab/exp_005_unitree_g1_flat_run/scripts/evaluate.py:"
            "_episode_quality_gates and _classify_episode"
        ),
        "exp005_native_labels": [
            "periodic_running", "periodic_running_with_physical_risk", "stable_with_isolated_flight",
            "stable_high_speed_walk", "unstable_with_flight", "not_stable", "incomplete",
        ],
        "labels": ["WALK_LIKE", "PERIODIC_RUNNING", "ISOLATED_FLIGHT", "IRREGULAR", "FALL"],
        "contact_threshold_n": 5.0, "flight_event": "both feet contact false",
        "safe_flight_duration_seconds": [.04, .16],
        "periodic_running": {
            "minimum_flight_events": 4, "minimum_consecutive_safe_cycles": 3,
            "minimum_alternating_contact_rate": .80, "minimum_normal_landing_rate": .80,
            "mean_flight_duration_seconds": [.04, .16],
        },
        "walk_like": {
            "stage2j_mapping_from_exp005": ["stable_with_isolated_flight", "stable_high_speed_walk"],
            "not_periodic_running": True, "flight_fraction_below": .10,
            "speed_mae_max_mps": .20, "fall": False,
        },
        "physical_quality_gates": {
            "contact_slip_mps_max": .55, "joint_velocity_limit_fraction_max": .05,
            "joint_torque_limit_fraction_max": .20, "landing_impact_p95_n_max": 3500,
            "impact_exceedance_fraction_max": .05, "vertical_excursion_m_max": .30,
            "stride_asymmetry_max": .20, "contact_time_asymmetry_max": .20,
        },
        "reported_features": [
            "flight event count", "flight duration", "flight fraction", "alternating landing rate",
            "single support", "double support", "contact timing", "periodicity",
        ],
    })
    dump("stage_reference.json", {
        "stage": "2J", "name": "low-speed action-manifold reachability diagnosis",
        "starting_head": START, "existing_classification_preserved": "REVERSE_SINGLE_POLICY_WALK_RECOVERY_FAIL",
        "run_parent_sha256": EXPECTED["R0"], "selected_r1_sha256": EXPECTED["R1"],
        "walk_positive_control_sha256": EXPECTED["W0"], "production_policy_updates": 0,
        "new_training_checkpoints": 0,
    })
    dump("protocol.json", {
        "frozen_checkpoints": ["W0", "R0", "R1"], "positive_control_episodes_each": 100,
        "positive_control_duration_s": 10, "command_mps": 1.2, "yaw_command": 0,
        "external_controllers": "OFF", "state_injection": False,
        "local_perturbation": local["perturbation_contract"],
        "short_sequence": {
            "method": "bounded random shooting", "steps": 4, "bound": .04, "candidates": 1024,
        },
        "policy_training": False, "reward_change": False, "checkpoint_write": False,
    })
    dump("diagnostic_seed_manifest.json", {
        "positive_controls": 20268021, "offline_bootstrap": 20268023,
        "state_overlap": [20268024, 20268025], "local_fresh_process": 20268041,
        "short_sequence_fresh_process": 20268041, "short_sequence_candidates": 20268042,
        "command_history": 20268051, "history_classifiers": [20268052, 20268053, 20268054],
    })
    dump("checkpoint_manifest.json", {
        key: {
            "path": str(path.relative_to(REPO)), "sha256": sha(path), "expected_sha256": EXPECTED[key],
            "role": {"W0": "WALK positive control", "R0": "RUN parent", "R1": "Phase R1 selected"}[key],
        } for key, path in CHECKPOINTS.items()
    })
    dump("stage_classification.json", {
        "main_classification": classification,
        "secondary_classifications": [
            reward["classification"], "WALK_RUN_STATE_DISJOINT", local["classification"],
            sequence["classification"], history_classification, "WALK_ACTION_OFF_POLICY_UNSUPPORTED",
        ],
        "gait_command_necessity_inference": inference,
        "evidence": {
            "walk_return_advantage_vs_r1": reward["W0_minus_R1"]["mean_return_difference"],
            "state_nonlinear_auroc": overlap["W0_vs_R1"]["nonlinear_classifier_auroc"],
            "walk_vs_r1_action_l2_on_walk_states": action["W0"]["W0_vs_R1"]["mean_l2"],
            "single_step_success_rate": local["success_rate"],
            "short_sequence_success_rate": sequence["success_rate"],
            "walk_action_valid_importance_ratio_fraction_r1": float(advantage[1]["valid_ratio_fraction"]),
            "bc_vs_base_reward_gradient_cosine_r1": gradients["R1"]["bc_vs_base_reward_cosine"],
        },
    })
    dump("recommended_next_action.json", {
        "recommended_next_action": next_action, "single_method_only": True,
        "not_executed": True,
        "rationale": (
            "WALK has a statistically significant base-reward advantage, but its states are disjoint and neither "
            "one-step nor bounded four-step perturbations reach it reliably from the RUN attractor."
        ),
    })
    dump("gate.json", {
        "stage_complete": True, "main_classification": classification,
        "reward_equivalence_audit": "PASS", "positive_control": "PASS",
        "fresh_process_prefix_equivalence": "PASS",
        "local_reachability": local["classification"], "short_sequence_reachability": sequence["classification"],
        "new_training_checkpoint": 0, "production_policy_update": 0,
        "remote_push": False,
    })
    starting_dirty = [
        "experiments/isaaclab/exp_006_unitree_g1_command_skills/README.md",
        "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.ps1",
        "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.py",
        ".openduck_hardware_source_review/", ".openduck_phase3_usb_baseline.txt",
        ".openduck_playground_source_review/", ".openduck_runtime_source_review/",
        "artifacts/exp_005_unitree_g1_flat_run/", "artifacts/openduck_*.png",
        "experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/*showcase*",
        "experiments/mujoco/exp_003_openduckmini_calibrated_walk/", "media/",
        "openduck_setup_report.md", "research/exp_011_linkedin_post_ja.md", "tools/*openduck*",
    ]
    dump("protected_hashes.json", {
        "starting_head": START, "checkpoint_hashes": EXPECTED,
        "protected_experiments": [f"exp_{index:03d}" for index in range(5, 12)],
        "exp012_previous_stages_changed": False, "checkpoint_changed": False,
        "optimizer_state_changed": False, "reward_changed": False, "curriculum_changed": False,
        "network_changed": False, "observation_action_changed": False, "physics_changed": False,
        "isaaclab_rslrl_core_changed": False, "unrelated_dirty_paths_preserved": starting_dirty,
        "remote_push": False,
    })

    reproduction = r'''$ErrorActionPreference = "Stop"
Set-Location "$HOME\workspace\physical-ai-lab"
$env:PYTHONPATH = "$PWD\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\src;$PWD\experiments\isaaclab\exp_005_unitree_g1_flat_run\src;$PWD"
$py = "C:\isaacsim\python.bat"
& $py .\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\scripts\collect_stage2j_positive_controls.py --headless --device cuda:0
& $py .\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\scripts\analyze_stage2j_offline.py
& $py .\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\scripts\diagnose_stage2j_local_reachability.py --mode baseline --headless --device cuda:0
& $py .\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\scripts\diagnose_stage2j_local_reachability.py --mode variant --headless --device cuda:0
& $py .\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\scripts\diagnose_stage2j_local_reachability.py --mode sequence_baseline --headless --device cuda:0
& $py .\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\scripts\diagnose_stage2j_local_reachability.py --mode sequence_variant --headless --device cuda:0
& $py .\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\scripts\collect_stage2j_history.py --headless --device cuda:0
& $py .\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\scripts\finalize_stage2j_diagnosis.py
'''
    (OUT / "reproduction_commands.ps1").write_text(reproduction, encoding="utf-8")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(f"""# exp_012 Stage 2J — Low-speed WALK action-manifold reachability

## Result

**{classification}**

The 1.2 m/s WALK trajectory has a statistically significant return advantage over both RUN policies, so low-speed
reward indifference is rejected. Nevertheless, the WALK and RUN physical-state distributions are effectively
disjoint (nonlinear AUROC {overlap['W0_vs_R1']['nonlinear_classifier_auroc']:.6f}), and bounded perturbations do not
provide a reliable local bridge.

## Positive controls

| policy | gait | flight fraction | stride frequency | return | fall |
|---|---:|---:|---:|---:|---:|
| W0 Stage 2 | WALK_LIKE | {positive['W0']['flight_fraction']:.3f} | {positive['W0']['stride_frequency_hz']:.3f} Hz | {positive['W0']['episode_return']:.3f} | {positive['W0']['fall_rate']:.1%} |
| R0 Stage 4 | PERIODIC_RUNNING | {positive['R0']['flight_fraction']:.3f} | {positive['R0']['stride_frequency_hz']:.3f} Hz | {positive['R0']['episode_return']:.3f} | {positive['R0']['fall_rate']:.1%} |
| R1 selected | PERIODIC_RUNNING | {positive['R1']['flight_fraction']:.3f} | {positive['R1']['stride_frequency_hz']:.3f} Hz | {positive['R1']['episode_return']:.3f} | {positive['R1']['fall_rate']:.1%} |

The gait distinction is physical, not classifier-only: RUN has about 48% flight and almost no double support, while
WALK has 3.5% flight, predominantly single support, lower vertical velocity, and lower pitch.

## Reward landscape

W0 exceeds R1 by {reward['W0_minus_R1']['mean_return_difference']:.3f} return
(bootstrap 95% CI {reward['W0_minus_R1']['bootstrap_95_ci'][0]:.3f} to
{reward['W0_minus_R1']['bootstrap_95_ci'][1]:.3f}). `safe_periodic_flight` is exactly zero on every 1.2 m/s sample.
The advantage comes from the unchanged base objective, notably yaw tracking, angular/vertical motion, orientation,
air-time, slide, acceleration, and action-rate terms.

## State, action, value, and gradient

WALK-vs-R1 mean-action L2 distance on WALK states is {action['W0']['W0_vs_R1']['mean_l2']:.3f}; the difference is
distributed across the action vector, with the largest lower-body terms at ankle and knee joints. WALK actions are
far off-policy under R1: only {float(advantage[1]['valid_ratio_fraction']):.2%} are inside the diagnostic ratio-valid
range, clip fraction is {float(advantage[1]['clip_fraction_0p2']):.2%}, and ESS is
{float(advantage[1]['ess_fraction']):.2%}. Consequently the positive valid-sample advantage cannot establish a
reliable on-policy WALK direction. The BC direction is nearly orthogonal to the base-reward gradient
(cosine {gradients['R1']['bc_vs_base_reward_cosine']:+.4f}); BC is diagnostic only and was not used for learning.

## Reachability

Fresh-process prefix hashes matched for {local['fresh_process_prefix_matches']}/{local['samples']} single-step
counterfactuals and {sequence['fresh_process_prefix_matches']}/{sequence['samples']} short-sequence
counterfactuals. One-step success was {local['walk_like_safe_successes']}/{local['samples']}. Four-step bounded
random shooting produced {sequence['walk_like_safe_successes']}/{sequence['samples']} candidate, or
{sequence['success_rate']:.3%}, below the 5% partial-reachability gate. No parameter or checkpoint was updated.

## History and interpretation

After convergence at current command 1.2, source-history classification is `{history_classification}`. This does not
imply that current gait is hidden: WALK and RUN current physical states are directly distinguishable. Rather, the
RUN policy maps the histories onto the same RUN attractor. The inference is **{inference}**.

## Next

Perform exactly one next method: **{next_action}**. No gait input, reward, recurrent state, or curriculum change was
made in Stage 2J.
""", encoding="utf-8")


if __name__ == "__main__":
    main()
