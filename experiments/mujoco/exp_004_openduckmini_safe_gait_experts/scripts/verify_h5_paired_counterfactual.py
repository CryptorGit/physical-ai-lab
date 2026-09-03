"""Verify the pre-registered H5 V2/V3 paired counterfactual evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


UNAFFECTED_INDEPENDENT_CASES = frozenset(
    {
        "stand",
        "reverse",
        "lateral_left",
        "lateral_right",
        "yaw_left",
        "yaw_right",
        "reverse_turn_left",
        "reverse_turn_right",
    }
)
V2_ID = "OPEN_DUCK_MINI_H5_COMMAND_ROUTING_V2"
V3_ID = "OPEN_DUCK_MINI_H5_UNIFIED_COMMAND_ROUTING_V3_DIRECT_NORMALIZED"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _runs(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    suites = payload.get("suites", {})
    for key in ("primitive_cases", "compound_cases", "transition_cases"):
        values = suites.get(key, [])
        if not isinstance(values, list):
            raise ValueError(f"suite {key} is not a list")
        yield from values


def _segments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [segment for run in _runs(payload) for segment in run.get("segments", [])]


def _strict_segment_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    acceptance = payload.get("suites", {}).get("acceptance", {})
    for group in ("primitives", "compounds", "transitions"):
        for episode in acceptance.get(group, {}).get("episode_checks", []):
            results.extend(episode.get("segments", []))
    return results


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _trace_protocol_errors(payload: dict[str, Any], label: str) -> list[str]:
    errors: list[str] = []
    suites = payload.get("suites", {})
    for group in ("primitive_cases", "compound_cases", "transition_cases"):
        for run_index, run in enumerate(suites.get(group, [])):
            segments = run.get("segments", [])
            for segment in segments:
                trace = segment.get("h5_control_trace", {})
                trace_hash = trace.get("trace_sha256")
                _require(
                    isinstance(trace_hash, str)
                    and len(trace_hash) == 64
                    and all(character in "0123456789abcdef" for character in trace_hash),
                    f"{label}: {segment.get('name')} lacks a SHA-256 trace",
                    errors,
                )
            protocol = run.get("h5_trace_protocol")
            if isinstance(protocol, dict):
                ticks = int(protocol.get("total_control_ticks", -1))
                guards = int(protocol.get("final_guard_call_count", -2))
                _require(ticks > 0, f"{label}: run {run_index} has no trace ticks", errors)
                _require(
                    guards == ticks
                    and protocol.get("exactly_one_guard_call_per_control_tick") is True,
                    f"{label}: run {run_index} guard-count invariant failed",
                    errors,
                )
                traced_ticks = sum(
                    int(segment.get("h5_control_trace", {}).get("control_tick_count", -1))
                    for segment in segments
                )
                _require(
                    traced_ticks == ticks,
                    f"{label}: run {run_index} trace tick accounting failed",
                    errors,
                )
                continue
            # ``_independent_suite`` flattens one-tick schedules but preserves
            # the per-case final-guard audit by segment name.
            audits = run.get("backward_exit_recovery_audits", {})
            if not isinstance(audits, dict):
                errors.append(f"{label}: {group} run {run_index} lacks guard audits")
                continue
            for segment in segments:
                name = str(segment.get("name"))
                audit = audits.get(name, {})
                ticks = int(segment.get("h5_control_trace", {}).get("control_tick_count", -1))
                _require(
                    ticks > 0
                    and audit.get("final_guard_call_count") == ticks
                    and audit.get("final_guard_calls_per_control_tick") == 1,
                    f"{label}: independent guard-count invariant failed: {name}",
                    errors,
                )
    return errors


def _independent_trace_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    suites = payload.get("suites", {})
    for group in ("primitive_cases", "compound_cases"):
        for run in suites.get(group, []):
            for segment in run.get("segments", []):
                name = str(segment.get("name"))
                if name in result:
                    raise ValueError(f"duplicate independently reset case: {name}")
                result[name] = segment
    return result


def verify(v2: dict[str, Any], v3: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    for label, payload, contract, mapper in (
        ("v2", v2, V2_ID, "legacy_h4_compensated"),
        ("v3", v3, V3_ID, "direct_normalized_v3"),
    ):
        provenance = payload.get("provenance", {})
        single = provenance.get("single_policy_mode", {})
        _require(payload.get("hardware_deployment") == "PROHIBITED", f"{label}: hardware gate", errors)
        _require(payload.get("acceptance", {}).get("adoption_allowed") is False, f"{label}: adoption gate", errors)
        _require(payload.get("acceptance", {}).get("release_allowed") is False, f"{label}: release gate", errors)
        _require(provenance.get("h5_command_contract") == contract, f"{label}: contract ID", errors)
        _require(single.get("command_mapper") == mapper, f"{label}: mapper", errors)
        _require(payload.get("policy_bank", {}).get("legacy_fallback", {}).get("count") == 0, f"{label}: fallback", errors)
        _require(payload.get("configuration", {}).get("transition_stand_seconds") == 2.0, f"{label}: transition stand duration", errors)
        _require(len(_segments(payload)) == 38, f"{label}: expected 38 segments", errors)
        _require(
            all(segment.get("gait_quality_metrics", {}).get("measurement_complete") is True for segment in _segments(payload)),
            f"{label}: incomplete gait-quality measurement",
            errors,
        )
        errors.extend(_trace_protocol_errors(payload, label))

    v2_single = v2.get("provenance", {}).get("single_policy_mode", {})
    v3_single = v3.get("provenance", {}).get("single_policy_mode", {})
    _require(
        v2_single.get("params_sha256") == v3_single.get("params_sha256"),
        "paired arms use different actor weights",
        errors,
    )
    same_configuration = dict(v2.get("configuration", {}))
    _require(same_configuration == v3.get("configuration", {}), "paired configurations differ", errors)

    v2_independent = _independent_trace_map(v2)
    v3_independent = _independent_trace_map(v3)
    _require(
        UNAFFECTED_INDEPENDENT_CASES <= set(v2_independent)
        and UNAFFECTED_INDEPENDENT_CASES <= set(v3_independent),
        "paired artifacts omit an unaffected independent case",
        errors,
    )
    trace_equal: dict[str, bool] = {}
    for name in sorted(UNAFFECTED_INDEPENDENT_CASES):
        left = v2_independent.get(name, {}).get("h5_control_trace", {})
        right = v3_independent.get(name, {}).get("h5_control_trace", {})
        equal = left.get("trace_sha256") == right.get("trace_sha256") and (
            left.get("control_tick_count") == right.get("control_tick_count")
        )
        trace_equal[name] = bool(equal)
        _require(equal, f"unaffected independent trace differs: {name}", errors)

    v3_trace_errors = [
        segment.get("h5_control_trace", {}).get("policy_command_max_abs_error")
        for segment in _segments(v3)
    ]
    _require(
        bool(v3_trace_errors)
        and all(isinstance(error, (float, int)) and error <= 1e-12 for error in v3_trace_errors),
        "V3 policy-command fidelity exceeds 1e-12",
        errors,
    )
    strict_v2 = _strict_segment_results(v2)
    strict_v3 = _strict_segment_results(v3)
    return {
        "schema_version": 1,
        "status": "V2_TRAINED_V3_EVALUATED_COUNTERFACTUAL",
        "adoption_allowed": False,
        "release_allowed": False,
        "hardware_deployment": "PROHIBITED",
        "paired_protocol_valid": not errors,
        "errors": errors,
        "unaffected_independent_trace_equal": trace_equal,
        "strict_segment_counts": {
            "v2": {"passed": sum(bool(value.get("passed")) for value in strict_v2), "total": len(strict_v2)},
            "v3": {"passed": sum(bool(value.get("passed")) for value in strict_v3), "total": len(strict_v3)},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {args.output}")
    result = verify(_load(args.v2), _load(args.v3))
    result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["inputs"] = {
        "v2": {"path": str(args.v2.resolve()), "sha256": _sha256(args.v2)},
        "v3": {"path": str(args.v3.resolve()), "sha256": _sha256(args.v3)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "paired_protocol_valid": result["paired_protocol_valid"]}))
    return 0 if result["paired_protocol_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
