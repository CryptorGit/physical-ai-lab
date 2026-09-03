from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from safe_gait_experts.routed_evaluation import (
    FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES,
)
from scripts import diagnose_h1_reverse_robustness as h1
from scripts import evaluate_routed_transitions as central


ARTIFACTS = h1.EXP_ROOT / "artifacts"
PROFILE = ARTIFACTS / "optimized_reverse_margin050_slew200_h1_phase7_rate105_candidate_v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_formal_seed_sets_are_exact() -> None:
    assert h1.FORMAL_REVERSE_SIMULATION_SEEDS == tuple(
        20_260_808 + index * 1000 + 2 for index in range(20)
    )
    assert h1.FORMAL_TRANSITION_SIMULATION_SEEDS == tuple(
        22_260_808 + index for index in range(20)
    )
    assert h1.FORMAL_REVERSE_SIMULATION_SEEDS[13] == 20_273_810


def test_candidate_phase_helper_is_exact_for_central_744_mapping() -> None:
    kwargs = {
        "phase_steps": 20.0,
        "phase_delta": 1.872330366914159,
        "current_expert": "reverse",
        "previous_expert": "stand",
        "effective_command": (-0.021, 0.0, 0.0),
        "previous_backward_feedforward_active": False,
        "diagnostic_entry_phase_indices": dict(
            FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES
        ),
        "phase_entry_status": "TEST",
        "diagnostic_only": False,
        "control_step": 9,
        "global_control_tick": 9,
    }
    expected = central.advance_routed_phase(3.5, **kwargs)
    actual = h1.advance_routed_phase_candidate(3.5, **kwargs)
    assert actual == expected


def test_selected_profile_is_minimal_phase_change_and_hardware_prohibited() -> None:
    base = _load(h1.BASE_PROFILE)
    selected = _load(PROFILE)
    assert selected["hardware_deployment"] == "PROHIBITED"
    assert selected["adoption"]["hardware_deployment"] == "PROHIBITED"
    assert selected["adoption"]["status"].startswith("NOT_ADOPTED")
    assert selected["parameters"]["joint_amplitude_scales"] == base["parameters"][
        "joint_amplitude_scales"
    ]
    assert selected["parameters"]["joint_bias_offsets"] == base["parameters"][
        "joint_bias_offsets"
    ]
    assert np.isclose(
        selected["parameters"]["phase_rate"],
        base["parameters"]["phase_rate"] * 1.05,
        rtol=0.0,
        atol=1e-15,
    )
    route = selected["composition"]["required_route_phase_entry"]
    assert route["preincrement_phase_index"] == 7.0
    assert route["mapping_is_preincrement_phase_index"] is True


def test_profile_pins_green_full_substep_evidence() -> None:
    selected = _load(PROFILE)
    evidence = selected["selection_evidence"]
    for binding in evidence.values():
        path = ARTIFACTS / binding["path"]
        assert path.is_file()
        assert _sha256(path) == binding["sha256"]

    straight = _load(ARTIFACTS / evidence["selected_straight_20x30"]["path"])
    selected_summary = next(
        summary
        for summary in straight["ranking"]
        if summary["candidate_id"]
        == evidence["selected_straight_20x30"]["selected_candidate_id"]
    )
    assert selected_summary["central_segment_acceptance_count"] == 20
    assert selected_summary["central_segment_count"] == 20
    assert selected_summary["fall_count"] == 0
    assert selected_summary["qpos_violation_samples"] == 0
    assert selected_summary["audited_physics_substeps"] == 300_000
    assert selected_summary["expected_physics_substeps"] == 300_000
    assert selected_summary["worst_signed_linear_progress_fraction"] >= 0.30

    transition = _load(
        ARTIFACTS / evidence["formal_transition_reverse_prefix_20seed"]["path"]
    )["ranking"][0]
    assert transition["central_segment_acceptance_count"] == 100
    assert transition["central_segment_count"] == 100
    assert transition["fall_count"] == 0
    assert transition["qpos_violation_samples"] == 0
    assert transition["audited_physics_substeps"] == 750_000
    assert transition["expected_physics_substeps"] == 750_000


def test_candidate_grid_has_stable_identity() -> None:
    candidate = h1.Candidate(0.0125, 7.0, 1.05, 1.0, 0.0)
    assert candidate.candidate_id == "b7b7f61e3eecf47c"
    assert h1.candidate_grid(
        h1.parse_args(
            [
                "--caps",
                "0.0125",
                "--phase-entries",
                "7",
                "--phase-rate-factors",
                "1.05",
            ]
        )
    ) == (candidate,)
