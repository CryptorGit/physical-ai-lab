from __future__ import annotations

import ast
from pathlib import Path

from safe_gait_experts.routed_evaluation import COMPOUND_CASES, PRIMITIVE_CASES


EXP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = EXP_ROOT.parents[2]
BUILDER = (
    WORKSPACE
    / "media"
    / "openduck_exp004_h3_release"
    / "build_h3_release_showcase.py"
)


def _literal_showcase_segments() -> tuple[tuple[object, ...], ...]:
    tree = ast.parse(BUILDER.read_text(encoding="utf-8"), filename=str(BUILDER))
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "SHOWCASE_SEGMENTS"
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, tuple)
            return value
    raise AssertionError("SHOWCASE_SEGMENTS literal not found")


def test_showcase_uses_formal_mapping_for_all_twelve_moving_cases() -> None:
    cases = {case.name: case for case in (*PRIMITIVE_CASES, *COMPOUND_CASES)}
    expected_names = (
        "forward",
        "reverse",
        "lateral_left",
        "lateral_right",
        "yaw_left",
        "yaw_right",
        "reverse_turn_left",
        "reverse_turn_right",
        "forward_turn_left",
        "forward_turn_right",
        "forward_lateral_left_turn",
        "forward_lateral_right_turn",
    )
    clips = _literal_showcase_segments()
    assert tuple(clip[0] for clip in clips) == expected_names
    for clip in clips:
        name, _label, physical, policy, duration, expert, role = clip
        case = cases[name]
        assert tuple(physical) == case.command
        assert (None if policy is None else tuple(policy)) == (
            None
            if case.policy_observation_command is None
            else case.policy_observation_command
        )
        assert duration == 6.0
        assert expert == case.expected_expert
        assert role == case.expected_policy_role
