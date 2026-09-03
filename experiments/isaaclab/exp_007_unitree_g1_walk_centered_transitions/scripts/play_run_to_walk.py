"""GUI diagnostic launcher for the Stage 8A RUN_TO_WALK direct baseline."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[4]

parser = argparse.ArgumentParser()
parser.add_argument("--run-speed", type=float, choices=(2.6, 2.8), required=True)
parser.add_argument("--seed", type=int, default=20261301)
parser.add_argument("--transition-checkpoint", default="")
args = parser.parse_args()
checkpoint = Path(args.transition_checkpoint).resolve(strict=True) if args.transition_checkpoint else None

command = [
    sys.executable,
    str(SCRIPT.parent / "evaluate_run_to_walk.py"),
    "--seed", str(args.seed),
    "--attempts-per-source", "1",
    "--source-speeds", str(args.run_speed),
    "--label", f"gui_{args.run_speed:.1f}",
    "--output", "results/exp_007_unitree_g1_walk_centered_transitions/stage8a_run_to_walk_audit/gui",
    "--stand", "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt",
    "--stand-to-walk", "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100/model_0.pt",
    "--walk", "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt",
    "--run", "logs/rsl_rl/physical_ai_g1_command_skills/2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt",
    "--walk-to-run", "results/exp_007_unitree_g1_walk_centered_transitions/stage7r8_walk_to_run_pilot2_saturation/checkpoints/model_100.pt",
]
print(
    "STATE: RUN_LOW / RUN_TO_WALK / WALK\n"
    f"CURRENT CHECKPOINT: {checkpoint if checkpoint else 'PARAMETER_FREE_STAGE8A'}\n"
    f"ITERATION: {'diagnostic checkpoint' if checkpoint else '0'}\n"
    "CONFIG SHA: 35be236b10cd19892f1104b4311734e9b9fea271be9ab7328960dd505d112b9d\n"
    "SOURCE GRAPH ROUTE: STAND -> STAND_TO_WALK -> WALK@1.2 -> WALK_TO_RUN -> RUN_LOW\n"
    f"SOURCE RUN SPEED: {args.run_speed:.1f} m/s\n"
    "TARGET WALK SPEED: 1.2 m/s\n"
    "RUN SOURCE CONTRACT: LIVE\n"
    "READY ENV COUNT / SELECTED COHORT IDS / SOURCE PHASE: LIVE\n"
    "ACTIVE CONTROLLER: RUN_LOW / RUN_TO_WALK\n"
    "IN-PLACE HANDOFF: TRUE; STATE COPY: NONE\n"
    "PREVIOUS ACTION MATCH / SENSOR-CONTACT CONTINUITY: AUDITED\n"
    "PPO STORAGE ACTIVE / SEGMENT ID / TERMINAL TYPE: DIAGNOSTIC ONLY\n"
    "WALK VALID STREAK / REQUIRED STREAK: LIVE / 20 STEPS (0.4 s)\n"
    "WALK ACQUISITION RAW REWARD: LIVE DIAGNOSTIC\n"
    "WALK ACQUISITION WEIGHTED REWARD: LIVE DIAGNOSTIC\n"
    "CONTRACT BREAK CONDITION: SPEED / HEADING / CONTACT / FLIGHT / SAFETY\n"
    "Runtime overlay metrics: last flight, RUN cycle termination, WALK contact/contract progress, "
    "completion, takeover, heading, reverse velocity, slip, impact, saturation, timeout; optimizer update is disabled.\n"
    "Checkpoint selection is displayed for Stage 8C diagnosis; this legacy visual route remains non-training."
)
raise SystemExit(subprocess.call(command, cwd=ROOT))
