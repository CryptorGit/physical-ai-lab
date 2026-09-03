"""Finalize visual review, reproduction metadata, and the Stage 5 report."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage5_endpoint_failure_diagnosis"
REPORT = REPO / "research/exp_011_go2_endpoint_failure_diagnosis_report.md"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    visual = load("visual_slip_validation_manifest.json")
    visual["requested_selection_shortfall"] = {
        "condition": "0.4 m/s failed episodes",
        "requested": 2, "available": sum(
            item["speed_mps"] == 0.4 and item["outcome"] == "fall"
            for item in visual["selections"]
        ),
        "reason": "only one failed episode existed in the fixed 50-episode seed set",
        "no_success_based_seed_substitution": True,
    }
    visual["manual_review"] = {
        "reviewed_frame_sequences": len(visual["selections"]),
        "observations": [
            "Zero-command sequences show stable upright stance.",
            "Moving sequences show the robot remaining upright while the overlay repeatedly flags one or more feet.",
            "At 1.2 m/s, flags coincide with multi-m/s foot-link origin velocity and nonzero contact-history force; they are not limited to the first rendered touchdown frame.",
            "The sole fixed-seed 0.4 m/s failure visibly collapses before termination and is a real physical failure.",
            "Frames cannot identify the instantaneous collision-point velocity; foot-link origin motion must not be described as contact-point slip without a contact-point metric.",
        ],
        "swing_foot_index_error_observed": False,
        "touchdown_only_explanation_sufficient": False,
        "clear_low_speed_fall_observed": True,
    }
    dump("visual_slip_validation_manifest.json", visual)

    gait = load("gait_classifier_audit.json")
    gait["manual_visual_episode_count"] = len(visual["selections"])
    gait["manual_review_conclusion"] = (
        "The manual subset confirms stable locomotion can coexist with IRREGULAR labels. "
        "It does not support relabelling the formal historical results; the classifier "
        "remains diagnostic-only and needs a quadruped alternating-phase contract."
    )
    dump("gait_classifier_audit.json", gait)

    gate = load("gate.json")
    gate["visual_validation"] = "PASS_WITH_ONE_FIXED_SEED_FAILURE_AVAILABLE"
    gate["raw_tensor_staged"] = False
    dump("gate.json", gate)

    (OUT / "reproduction_commands.ps1").write_text(
        'cd "$HOME\\workspace\\physical-ai-lab"\n'
        '.\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\run_stage5_endpoint_diagnosis.ps1\n'
        '# Re-analyze saved telemetry without new interaction\n'
        'python .\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\analyze_stage5_endpoints.py\n'
        'python .\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\finalize_stage5.py\n',
        encoding="utf-8",
    )

    classification = load("stage5_classification.json")
    stand = load("stand_posture_distribution.json")["by_checkpoint"]
    low = load("low_speed_sweep.json")
    slip = load("slip_severity_by_speed.json")
    boundary = load("contact_boundary_slip_audit.json")
    heading = load("low_speed_heading_audit.json")
    paired = slip["paired_stage4_minus_parent"]
    report = f"""# EXP_011 Go2 Endpoint Failure Diagnosis Report

## Status

```text
PRIMARY:
{classification['primary']}

SECONDARY:
{chr(10).join(classification['secondary'])}
```

No PPO update, reward change, curriculum change, checkpoint change, or physics
change was performed.

## Slip

The Stage 1--4 evaluator marks contact from the maximum 3D force in contact
history above 5 N, takes the maximum world-frame planar velocity of the four
foot rigid-body origins, averages that maximum over the episode, and calls the
episode dangerous when the mean exceeds 0.55 m/s. It does not fail on a single
event and requires no minimum contact or contiguous slip duration.

Foot names, robot indices, sensor indices, SI units, and world frame are
consistent. However, `body_lin_vel_w` describes the foot rigid-body origin,
not the instantaneous ground collision point. The official generic
`feet_slide` function uses the same body-origin velocity with a 1 N threshold,
but the official Go2 Flat reward configuration does not include that term;
its weighted contribution is exactly zero.

Contact-boundary-only events account for
`{boundary['boundary_only_fraction']:.1%}` of slip-positive events, below the
80% boundary-dominated criterion. At 0.4/0.6/1.2/2.0 m/s the Stage 4 contact
fractions above 0.5 m/s are
`{', '.join(f"{slip['conditions'][f'stage4_selected:{speed}']['threshold_levels']['0.5']['mean_contact_time_fraction']:.1%}" for speed in (0.4,0.6,1.2,2.0))}`.
Thus the historical binary 100% is not a one-step artifact. It is accompanied
by sustained foot-link-origin motion. Because contact-point velocity was not
available, the report preserves that geometric limitation rather than
overclaiming literal surface sliding.

Paired Stage 4-minus-parent existing-slip mean differences at
0.4/0.6/1.2/2.0 m/s are
`{', '.join(f"{paired[str(speed)]['stage4_minus_parent_existing_slip_mean_mps']:+.3f}" for speed in (0.4,0.6,1.2,2.0))}` m/s.
The effect is mixed rather than a uniform Stage 4 regression.

## Stand

Current Isaac Lab explicitly exposes `root_quat_w.torch` as **xyzw**.
The Stage 1--4 evaluator unbound it as **wxyz**. This is the direct cause of
the reported near-pi roll values during successful stance.

With the correct contract, Stage 4 zero-command results are: fall
`{stand['stage4_selected']['fall_rate']:.1%}`, roll p95
`{stand['stage4_selected']['roll_abs_p95_rad']:.3f}` rad, pitch p95
`{stand['stage4_selected']['pitch_abs_p95_rad']:.3f}` rad, gravity tilt p95
`{stand['stage4_selected']['gravity_tilt_p95_rad']:.3f}` rad, and settle-after-2s
height range p95 `{stand['stage4_selected']['settle_height_range_p95_m']:.4f}` m.
The original height range included the reset/settling transient and was
`{stand['stage4_selected']['height_range_p95_m']:.4f}` m. The settled
nominal-relative tilt deviation is
`{stand['stage4_selected']['settle_nominal_deviation_p95_rad']:.4f}` rad.
This supports `STAND_METRIC_NOT_GO2_APPROPRIATE`, not a real Stage 4 standing
posture failure.

## Low speed

The Stage 4 fall rates at 0.0--0.7 m/s are:
`{', '.join(f"{speed}:{low['stage4_selected'][str(speed)]['fall_rate']:.0%}" for speed in (0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7))}`.
The unstable band is 0.2--0.5 m/s, with the highest failure at 0.2--0.3,
then recovery by 0.7 m/s. At 0.4 m/s, non-fallen yaw p95 remains
`{heading['by_checkpoint_and_outcome']['stage4_selected']['non_fallen']['yaw_p95_rad']:.3f}` rad,
so the heading result is not produced only by the fallen tail. The fixed-seed
visual failure shows a genuine collapse. Contact distributions shift from
stand-like stepping toward irregular locomotion in this band, supporting
`REAL_LOW_SPEED_GAIT_BIFURCATION`.

## Gait classifier

Across 1,000 paired diagnostic episodes, the historical classifier labels
70.1% IRREGULAR; 279 of those are independently classified as stand-like
stepping. Its pair synchrony measures equality rather than alternating-phase
opposition, so it is not a reliable quadruped gait identity metric for these
near-full-duty traces. It remains diagnostic-only.

## Classification and next action

The primary result is `{classification['primary']}` because the proven
quaternion contract bug has precedence. Real sustained foot-link motion and a
real low-speed gait bifurcation are retained as secondary physical findings.

The one next action is:

```text
{load('recommended_next_action.json')['action']}
```

No checkpoint is promoted and no training Pilot is authorized by Stage 5.
"""
    REPORT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
