"""Tests for H5's one-weight diagnostic wrapper provenance."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).parents[1] / "scripts" / "create_h5_sameweight_alias_manifest.py"
_SPEC = importlib.util.spec_from_file_location("h5_sameweight_alias", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_alias_keeps_one_hash_bound_unified_source_candidate(tmp_path):
    params = tmp_path / "final_params.pkl"
    params.write_bytes(b"one immutable actor")
    config = tmp_path / "resolved_config.json"
    config.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "expert": "unified",
                "hardware_deployment": "PROHIBITED",
                "qualification_use": "DIAGNOSTIC_REWARD_EXPLORATION_NOT_QUALIFICATION",
                "outputs": {
                    "final_params": {
                        "path": str(params),
                        "sha256": _MODULE.sha256_file(params),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    alias = _MODULE.build_alias_manifest(
        source_manifest=manifest, params=params, domain="planar"
    )

    assert alias["expert"] == "planar"
    assert alias["source_candidate"] == {
        "path": str(manifest),
        "sha256": _MODULE.sha256_file(manifest),
    }
    assert "source_manifest" not in alias
    assert alias["outputs"]["final_params"]["sha256"] == _MODULE.sha256_file(params)
    assert alias["single_weight_alias"]["same_parameter_path"] is True


def test_alias_rejects_params_outside_the_unified_source_binding(tmp_path):
    source_params = tmp_path / "source.pkl"
    source_params.write_bytes(b"source")
    supplied_params = tmp_path / "other.pkl"
    supplied_params.write_bytes(b"other")
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "expert": "unified",
                "hardware_deployment": "PROHIBITED",
                "qualification_use": "DIAGNOSTIC_REWARD_EXPLORATION_NOT_QUALIFICATION",
                "outputs": {
                    "final_params": {
                        "path": str(source_params),
                        "sha256": _MODULE.sha256_file(source_params),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="do not match"):
        _MODULE.build_alias_manifest(
            source_manifest=manifest, params=supplied_params, domain="reverse"
        )
