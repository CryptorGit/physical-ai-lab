from __future__ import annotations

from dataclasses import fields

from qmini_population_bwm.data_schema import CanonicalBodilyObservation, TransitionRecord, observation_field_names


def test_teacher_id_is_not_a_model_input_or_transition_field() -> None:
    names = " ".join(observation_field_names()).lower()
    assert "teacher" not in names
    assert "reward" not in names
    assert "hidden" not in names
    assert "teacher_id" not in {field.name for field in fields(CanonicalBodilyObservation)}
    assert "teacher_id" not in {field.name for field in fields(TransitionRecord)}
