from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from scripts import validate_h4_reverse_training_composition as validator


def test_canonical_contract_and_components_pass() -> None:
    audit = validator.validate_contract()
    assert audit["valid"] is True
    assert audit["simulation_250k_pilot_training"] is True
    assert audit["simulation_1m_training"] is False
    assert audit["hardware_deployment"] == "PROHIBITED"
    assert set(audit["component_audit"]) == {
        "selected_teacher",
        "teacher_bank",
        "failure3_screen",
        "minimum_retraining_spec",
    }


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("authorization", "hardware"), True, "authorization drifted: hardware"),
        (("authorization", "simulation_1m_training"), True, "authorization drifted: simulation_1m_training"),
        (("authorization", "candidate_adoption"), True, "authorization drifted: candidate_adoption"),
        (("composition_contract", "maximum_residual_scale"), 0.2, "composition field drifted"),
        (("scope", "actor_observation_width"), 114, "actor width drifted"),
    ],
)
def test_contract_mutations_fail_closed(
    tmp_path: Path, path: tuple[str, str], value: object, message: str
) -> None:
    payload = validator.load_json_strict(validator.DEFAULT_CONTRACT)
    payload[path[0]][path[1]] = value
    candidate = tmp_path / "contract.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(validator.ReverseTrainingCompositionError, match=message):
        validator.validate_contract(candidate)


def test_duplicate_and_nonfinite_json_are_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(validator.ReverseTrainingCompositionError, match="duplicate"):
        validator.load_json_strict(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(validator.ReverseTrainingCompositionError, match="non-finite"):
        validator.load_json_strict(nonfinite)
