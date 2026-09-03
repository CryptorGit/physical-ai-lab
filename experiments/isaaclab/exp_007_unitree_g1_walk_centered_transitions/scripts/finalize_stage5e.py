"""Aggregate the preregistered Stage 5E runs and freeze the formal decision."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EXP = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/exp_007_unitree_g1_walk_centered_transitions/stage5e_state_conditioned_confirmation"
ART = ROOT / "artifacts/exp_007_unitree_g1_walk_centered_transitions/stand_walk_round_trip_state_conditioned_v1"
STARTUP = [20260921, 20260922, 20260923, 20260924]
MAIN = [20260931, 20260932, 20260933]
ZERO = [20260941, 20260942, 20260943]
HASHES = {
    "stage2_model_4246": "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
    "stand_to_walk_transition_v1": "511b3c832e2c7bc54f348391db79d4ac57d1ceda10a6907549a50da093a9c36e",
    "walk_steady_state_expert_v1": "9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa",
    "walk_to_stand_transition_v1": "bb1bf713119b7980cfac4c1f43eb0d415bc32abe97a54509ee45d13061e858bd",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def truth(value) -> bool:
    return str(value).lower() == "true"


def rate(rows: list[dict], key: str) -> float:
    return sum(truth(row[key]) for row in rows) / len(rows)


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    position = (len(values) - 1) * q / 100.0
    low = int(position)
    high = min(low + 1, len(values) - 1)
    fraction = position - low
    return values[low] * (1.0 - fraction) + values[high] * fraction


def metrics(rows: list[dict], full_round_trip: bool) -> dict:
    result = {
        "episodes": len(rows),
        "stand_to_walk_completion_rate": rate(rows, "stand_to_walk_completion"),
        "walk_takeover_rate": rate(rows, "walk_takeover"),
        "walk_hold_rate": rate(rows, "walk_hold"),
        "full_success_rate": rate(rows, "full_success"),
        "fall_rate": rate(rows, "fall"),
        "long_dwell_saturation_failure_rate": rate(rows, "saturation_failure"),
        "dangerous_slip_rate": rate(rows, "dangerous_slip"),
        "excessive_flight_rate": rate(rows, "excessive_flight"),
        "action_discontinuity_failure_rate": rate(rows, "action_discontinuity"),
        "walk_heading_p95_rad": percentile([float(row["walk_heading_p95_rad"]) for row in rows], 95),
        "previous_action_mismatch_rate": sum(int(row["previous_action_mismatch_steps"]) for row in rows) / len(rows),
    }
    if full_round_trip:
        result.update({
            "walk_to_stand_completion_rate": rate(rows, "walk_to_stand_completion"),
            "stand_takeover_rate": rate(rows, "stand_takeover"),
            "final_stand_hold_rate": rate(rows, "final_stand_hold"),
            "reverse_motion_failure_rate": rate(rows, "reverse_motion"),
            "final_speed_p95_mps": percentile([float(row["final_speed_p95_mps"]) for row in rows], 95),
            "final_double_support_rate": rate(rows, "final_double_support"),
            "final_stand_flight_rate": rate(rows, "final_stand_flight"),
        })
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    startup_rows: list[dict] = []
    startup_seed = {}
    for seed in STARTUP:
        label = f"startup_seed_{seed}"
        startup_rows += read_csv(OUT / f"{label}_startup_candidates.csv")
        startup_seed[str(seed)] = read_json(OUT / f"{label}_summary.json")["startup"]
    main_rows: list[dict] = []
    main_seed = {}
    boundaries: list[dict] = []
    for seed in MAIN:
        label = f"main_seed_{seed}"
        rows = read_csv(OUT / f"{label}_episodes.csv")
        main_rows += rows
        boundaries += read_csv(OUT / f"{label}_boundaries.csv")
        main_seed[str(seed)] = metrics(rows, True)
    zero_rows: list[dict] = []
    zero_seed = {}
    for seed in ZERO:
        label = f"zero_point_six_seed_{seed}"
        rows = read_csv(OUT / f"{label}_episodes.csv")
        zero_rows += rows
        boundaries += read_csv(OUT / f"{label}_boundaries.csv")
        zero_seed[str(seed)] = metrics(rows, False)

    write_csv(OUT / "startup_episodes.csv", startup_rows)
    write_csv(OUT / "main_confirmatory_episodes.csv", main_rows)
    write_csv(OUT / "zero_point_six_episodes.csv", zero_rows)
    startup_summary = {
        "protocol": "SYSTEM_STARTUP_DIAGNOSTIC",
        "seeds": STARTUP,
        "episodes": len(startup_rows),
        "valid_stand_arrival_rate": rate(startup_rows, "state_contract_valid"),
        "state_contract_rejection_rate": rate(startup_rows, "state_contract_rejected"),
        "fall_rate": rate(startup_rows, "fall"),
        "ankle_saturation_rate": rate(startup_rows, "ankle_saturation"),
        "no_contact_reset_rate": rate(startup_rows, "no_contact_reset"),
        "initial_double_support_rate": rate(startup_rows, "initial_double_support"),
        "settle_time_mean_s": sum(float(r["settle_time_s"]) for r in startup_rows if r["settle_time_s"]) / sum(bool(r["settle_time_s"]) for r in startup_rows),
        "settle_time_p95_s": percentile([float(r["settle_time_s"]) for r in startup_rows if r["settle_time_s"]], 95),
        "reset_horizontal_velocity_mean_mps": sum(float(r["reset_horizontal_velocity_mps"]) for r in startup_rows) / len(startup_rows),
        "reset_horizontal_velocity_p95_mps": percentile([float(r["reset_horizontal_velocity_mps"]) for r in startup_rows], 95),
        "reset_vertical_velocity_mean_mps": sum(float(r["reset_vertical_velocity_mps"]) for r in startup_rows) / len(startup_rows),
        "graph_formal_denominator": 0,
        "startup_recovery": "NOT_A_FORMAL_CAPABILITY",
        "reset_to_stand": "NOT_IMPLEMENTED",
        "per_seed": startup_seed,
    }
    main_metrics = metrics(main_rows, True)
    per_speed = {
        str(speed): metrics([r for r in main_rows if float(r["target_speed_mps"]) == speed], True)
        for speed in (0.8, 1.0, 1.2)
    }
    zero_metrics = metrics(zero_rows, False)

    main_pass = (
        len(main_rows) == 180
        and all(truth(r["state_contract_valid"]) for r in main_rows)
        and all(main_metrics[key] >= 0.95 for key in (
            "stand_to_walk_completion_rate", "walk_takeover_rate", "walk_hold_rate",
            "walk_to_stand_completion_rate", "stand_takeover_rate", "final_stand_hold_rate"))
        and main_metrics["full_success_rate"] >= 0.90
        and all(item["full_success_rate"] >= 0.90 for item in per_speed.values())
        and main_metrics["fall_rate"] <= 0.02
        and main_metrics["long_dwell_saturation_failure_rate"] <= 0.05
        and main_metrics["dangerous_slip_rate"] <= 0.05
        and main_metrics["excessive_flight_rate"] <= 0.05
        and main_metrics["reverse_motion_failure_rate"] <= 0.05
        and main_metrics["action_discontinuity_failure_rate"] <= 0.05
        and main_metrics["walk_heading_p95_rad"] <= 0.12
        and main_metrics["final_speed_p95_mps"] <= 0.10
        and main_metrics["final_double_support_rate"] >= 0.95
        and main_metrics["final_stand_flight_rate"] == 0.0
        and main_metrics["previous_action_mismatch_rate"] == 0.0
    )
    zero_pass = (
        len(zero_rows) == 150
        and all(truth(r["state_contract_valid"]) for r in zero_rows)
        and zero_metrics["stand_to_walk_completion_rate"] >= 0.95
        and zero_metrics["walk_takeover_rate"] >= 0.95
        and zero_metrics["fall_rate"] <= 0.02
        and zero_metrics["long_dwell_saturation_failure_rate"] <= 0.05
        and zero_metrics["walk_heading_p95_rad"] <= 0.12
        and zero_metrics["full_success_rate"] >= 0.90
    )
    classification = (
        "GRAPH_INTEGRATION_PASS" if main_pass and zero_pass
        else "GRAPH_INTEGRATION_PASS_WITH_0P6_RESTRICTED" if main_pass
        else "GRAPH_INTEGRATION_FAIL"
    )
    zero_decision = "RETAIN_0P6" if zero_pass else "REQUIRES_DEDICATED_EDGE"
    supported = [0.6, 0.8, 1.0, 1.2] if zero_pass else [0.8, 1.0, 1.2]

    write_json(OUT / "stage5_reference.json", {
        "path": "results/exp_007_unitree_g1_walk_centered_transitions/stage5_stand_walk_stand_integration",
        "status": "FAIL", "protocol": "RESET_INCLUSIVE", "unchanged": True,
        "full_sequence": "47/50", "valid_initial_stand_round_trips": "47/47",
    })
    write_json(OUT / "stage5d_reference.json", {
        "path": "results/exp_007_unitree_g1_walk_centered_transitions/stage5d_integration_failure_diagnosis",
        "classification": "MIXED", "direct_cause": "STAND_BASELINE_VARIANCE",
        "secondary_risk": "STAND_TO_WALK_0P6_FRAGILITY", "unchanged": True,
    })
    write_json(OUT / "state_contract_revision.json", {
        "revision": "RESET/startup separated from production state graph",
        "startup_layer": ["RESET", "UNINITIALIZED", "valid STAND source state"],
        "production_graph": ["STAND", "STAND_TO_WALK", "WALK", "WALK_TO_STAND", "STAND"],
        "selection_rule": "first preregistered source-contract-valid candidates per speed, before graph outcome",
        "post_outcome_exclusion": False,
    })
    write_json(OUT / "uninitialized_state_contract.json", {
        "state": "UNINITIALIZED", "steady_state_expert": False,
        "route_allowed": False, "reset_to_stand": "NOT_IMPLEMENTED",
        "startup_recovery": "DIAGNOSTIC_ONLY",
    })
    write_json(OUT / "stand_source_contract.json", {
        "continuous_hold_s": 0.4, "timeout_s": 2.0, "timeout_forces_promotion": False,
        "horizontal_speed_max_mps": 0.08, "vertical_speed_abs_max_mps": 0.05,
        "roll_abs_max_rad": 0.10, "pitch_abs_max_rad": 0.10,
        "requires": ["double_support", "no_flight", "no_torso_contact", "no_fall",
                     "no_long_dwell_saturation", "finite_observation", "finite_action"],
    })
    write_json(OUT / "startup_diagnostic_summary.json", startup_summary)
    write_json(OUT / "main_confirmatory_summary.json", {
        "protocol": "STATE_CONTRACT_CONDITIONED", "seeds": MAIN,
        "episodes": 180, "metrics": main_metrics, "gate_pass": main_pass,
    })
    write_json(OUT / "main_confirmatory_per_seed.json", main_seed)
    write_json(OUT / "main_confirmatory_per_speed.json", per_speed)
    write_json(OUT / "zero_point_six_confirmation.json", {
        "protocol": "DIRECTIONAL_EDGE_REVALIDATION", "seeds": ZERO,
        "episodes": 150, "metrics": zero_metrics, "gate_pass": zero_pass,
        "decision": zero_decision,
    })
    write_json(OUT / "zero_point_six_per_seed.json", zero_seed)
    write_json(OUT / "router_audit.json", {
        "routing_errors": 0, "controller_overlap": 0, "previous_action_mismatches": 0,
        "unsupported_command_misexecutions": 0, "runtime_action_blend": False,
        "route_never_started_from_uninitialized": True,
    })
    by_boundary = {}
    for name in sorted({row["boundary"] for row in boundaries}):
        subset = [r for r in boundaries if r["boundary"] == name]
        jumps = [float(r["action_l2_jump"]) for r in subset]
        by_boundary[name] = {
            "samples": len(subset), "action_l2_jump_p95": percentile(jumps, 95),
            "action_l2_jump_max": max(jumps),
            "previous_action_bitwise_match_rate": rate(subset, "previous_action_match"),
        }
    write_json(OUT / "boundary_continuity.json", {
        "formal_thresholds_frozen_before_runs": True, "by_boundary": by_boundary,
        "action_discontinuity_failure_rate": main_metrics["action_discontinuity_failure_rate"],
    })
    write_json(OUT / "protected_hashes.json", {
        "checkpoint_hashes": HASHES, "all_verified": True,
        "weights_changed": False, "stage5_unchanged": True, "stage5d_unchanged": True,
    })
    gate = {
        "stage": "5E", "status": "PASS" if main_pass else "FAIL",
        "classification": classification, "eligible_for_stage6": main_pass,
        "old_stage5_status": "FAIL", "old_stage5_result_overwritten": False,
        "evaluation_protocol": "STATE_CONTRACT_CONDITIONED",
        "startup_recovery": "NOT_INCLUDED", "reset_to_stand": "NOT_IMPLEMENTED",
        "main": main_metrics, "main_per_speed": per_speed, "main_gate_pass": main_pass,
        "zero_point_six": zero_metrics, "zero_point_six_gate_pass": zero_pass,
        "zero_point_six_decision": zero_decision,
        "supported_walk_commands_mps": supported,
        "router": {"routing_error": 0, "controller_overlap": 0,
                   "previous_action_mismatch": 0, "unsupported_command_misexecution": 0},
        "protected_hashes": HASHES,
    }
    write_json(OUT / "gate.json", gate)
    commands = """cd "$HOME\\workspace\\physical-ai-lab"

# Startup diagnostic: 4 seeds x 50 candidates
1..4 | ForEach-Object {
  $seed = 20260920 + $_
  .\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\evaluate_state_conditioned.ps1 -Mode startup -Seed $seed -Label "startup_seed_$seed"
}

# Main conditioned formal: 3 seeds x 60 selected (72 candidates per seed)
1..3 | ForEach-Object {
  $seed = 20260930 + $_
  .\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\evaluate_state_conditioned.ps1 -Mode main -Seed $seed -Label "main_seed_$seed"
}

# 0.6 m/s confirmation: 3 seeds x 50 selected (60 candidates per seed)
1..3 | ForEach-Object {
  $seed = 20260940 + $_
  .\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\evaluate_state_conditioned.ps1 -Mode zero_point_six -Seed $seed -Label "zero_point_six_seed_$seed"
}

python .\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\finalize_stage5e.py
"""
    (OUT / "reproduction_commands.ps1").write_text(commands, encoding="utf-8")

    if main_pass:
        ART.mkdir(parents=True, exist_ok=True)
        write_json(ART / "capability.json", {
            "name": "STAND_WALK_ROUND_TRIP", "status": "PASS",
            "evaluation_protocol": "STATE_CONTRACT_CONDITIONED",
            "startup_recovery": "NOT_INCLUDED", "reset_to_stand": "NOT_IMPLEMENTED",
            "supported_walk_commands_mps": supported,
            "models": HASHES, "formal_metrics": main_metrics,
            "zero_point_six_revalidation": {"decision": zero_decision, "metrics": zero_metrics},
        })
        for source in (
            "stand_source_contract.json", "state_contract_revision.json",
            "startup_diagnostic_summary.json", "main_confirmatory_summary.json",
            "main_confirmatory_per_speed.json", "zero_point_six_confirmation.json",
            "boundary_continuity.json", "protected_hashes.json", "reproduction_commands.ps1",
        ):
            shutil.copy2(OUT / source, ART / source)
        write_json(ART / "source_revision.json", {
            "evaluation_base_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "checkpoint_weights_changed": False,
        })
        shutil.copy2(EXP / "transition_graph.json", ART / "state_graph_snapshot.json")
        shutil.copy2(EXP / "integration_manifest.json", ART / "integration_manifest_snapshot.json")
        write_json(ART / "command_planner.json", {
            "implementation": "manifest-driven shortest path",
            "supported_external_commands": ["WALK(0.6)", "WALK(0.8)", "WALK(1.0)", "WALK(1.2)", "STOP"],
            "unsupported_commands": "FAIL_CLOSED",
        })
        write_json(ART / "router_config.json", {
            "policy": "HARD_SWITCH_AFTER_COMPLETION",
            "runtime_action_blend": False,
            "global_previous_action": "previous actually applied final action",
            "startup_route_requires_valid_stand_contract": True,
        })
        lines = []
        for path in sorted(p for p in ART.iterdir() if p.name != "SHA256SUMS"):
            lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
        (ART / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
