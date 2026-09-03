import json
from pathlib import Path
import sys


EXP = Path(__file__).resolve().parents[1]
TOOLS = EXP / "tools"
sys.path.insert(0, str(TOOLS))
from v59_mjx_diagnostic_common import (
    array_sha256,
    canonical_tree_sha256,
    load_pickle,
)


def test_serialized_input_hashes_match_manifest():
    root = EXP / "artifacts" / "v59_mjx_first_step_divergence"
    manifest = json.loads(
        (root / "serialized_input_hashes.json").read_text(encoding="utf-8")
    )
    for case_id in ("D0", "D1a", "D2"):
        payload = load_pickle(root / "inputs" / f"{case_id}.pkl")
        assert canonical_tree_sha256(payload["state"]) == manifest[case_id][
            "state_tree_sha256"
        ]
        assert canonical_tree_sha256(payload["data"]) == manifest[case_id][
            "data_tree_sha256"
        ]
        assert canonical_tree_sha256(payload["model"]) == manifest[case_id][
            "model_tree_sha256"
        ]
        assert array_sha256(payload["motor_target"]) == manifest[case_id][
            "motor_target_sha256"
        ]


def test_termination_prestate_is_explicitly_unavailable():
    root = EXP / "artifacts" / "v59_mjx_first_step_divergence"
    manifest = json.loads(
        (root / "serialized_input_hashes.json").read_text(encoding="utf-8")
    )
    assert manifest["D1b"]["status"] == "unavailable"

