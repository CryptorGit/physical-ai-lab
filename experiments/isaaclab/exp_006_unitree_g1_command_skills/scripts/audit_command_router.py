"""Evaluate fail-closed command-system requests without changing a simulator policy."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve(); EXP = SCRIPT.parent.parent; REPO = EXP.parents[2]
sys.path.insert(0, str(EXP / "src"))

from g1_command_skills.command_system import (  # noqa: E402
    CommandSystemRouter, ControllerState, select_controller_action,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
args = parser.parse_args()


def main() -> None:
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(20260723)
    action_bank = {
        "running": torch.randn(37, generator=generator),
        "standing": torch.randn(37, generator=generator),
        "stop": torch.randn(37, generator=generator),
        "crouch": torch.randn(37, generator=generator) * 0.1,
    }
    cases = [
        (ControllerState.RUN, "CROUCH_SHALLOW", 10),
        (ControllerState.TURN, "STAND", 10),
        (ControllerState.STAND, "RUN", 10),
        (ControllerState.CROUCH_SHALLOW, "RUN", 10),
        (ControllerState.STAND, "CROUCH_DEEP", 5),
        (ControllerState.STAND, "STEP_OVER", 5),
        (ControllerState.STAND, "LAND", 5),
    ]
    rows = []
    for initial, requested, count in cases:
        for episode in range(count):
            router = CommandSystemRouter(initial)
            before = select_controller_action(
                initial, running_family_action=action_bank["running"],
                standing_base_action=action_bank["standing"],
                stop_prototype_action=action_bank["stop"],
                crouch_shallow_offset=action_bank["crouch"],
            ).clone()
            decision = router.request(requested)
            after = select_controller_action(
                ControllerState(decision.active_controller_state),
                running_family_action=action_bank["running"],
                standing_base_action=action_bank["standing"],
                stop_prototype_action=action_bank["stop"],
                crouch_shallow_offset=action_bank["crouch"],
            )
            difference = after - before
            expected_reason = (
                "CROSS_BASE_FAMILY_TRANSITION_UNRESOLVED" if requested in {"RUN", "STAND", "CROUCH_SHALLOW"}
                else {"CROUCH_DEEP": "DEEP_CROUCH_RETURN_UNRESOLVED", "STEP_OVER": "OPTIMIZATION_FAILURE", "LAND": "POSITION_OFFSET_LANDING_CONTROLLER_FAILED"}[requested]
            )
            passed = bool(
                not decision.transition_supported and not decision.transition_started
                and decision.active_controller_state == initial.value
                and decision.rejection_reason == expected_reason
                and torch.equal(before, after)
                and not decision.primitive_started
            )
            rows.append({
                "case": f"{initial.value}_TO_{requested}", "episode": episode,
                **decision.to_dict(), "action_bitwise_unchanged": torch.equal(before, after),
                "action_discontinuity_l2": float(torch.linalg.vector_norm(difference)),
                "action_discontinuity_max": float(difference.abs().max()),
                "unsafe_offset_zero": not decision.primitive_started,
                "passed": passed,
            })
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {
        "schema_version": 1, "episodes": len(rows),
        "pass_rate": sum(row["passed"] for row in rows) / len(rows),
        "all_actions_bitwise_unchanged": all(row["action_bitwise_unchanged"] for row in rows),
        "maximum_action_discontinuity": max(row["action_discontinuity_max"] for row in rows),
        "failure_counts": dict(Counter(row["case"] for row in rows if not row["passed"])),
        "case_results": {
            name: {"count": len(selected), "pass_rate": sum(row["passed"] for row in selected) / len(selected)}
            for name in sorted({row["case"] for row in rows})
            for selected in [[row for row in rows if row["case"] == name]]
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
