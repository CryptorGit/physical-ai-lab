"""Summarize force-contact gait quality for an isolated H4 reverse artifact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


WINDOWS = ("10ms", "20ms", "30ms", "40ms")
EXPECTED_CONTACT_SOURCE = "normal_force_schmitt"
EXPECTED_SLIP_SOURCE = "force_weighted_contact_point_jacobian"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_range(values: Sequence[float | int | None]) -> dict[str, float | None]:
    finite = [float(value) for value in values if value is not None]
    return {
        "minimum": min(finite) if finite else None,
        "maximum": max(finite) if finite else None,
    }


def summarize_candidate(record: Mapping[str, Any]) -> dict[str, Any]:
    summary = record["summary"]
    runs = record["runs"]
    quality = [run["central_run"]["segments"][0]["gait_quality_metrics"] for run in runs]
    if not quality or any(item is None for item in quality):
        raise ValueError("every run must contain gait_quality_metrics")
    if any(set(item["contact_debounce_sensitivity"]) != set(WINDOWS) for item in quality):
        raise ValueError("every run must contain exact 10/20/30/40ms sensitivity")

    sensitivity: dict[str, Any] = {}
    for window in WINDOWS:
        items = [item["contact_debounce_sensitivity"][window] for item in quality]
        sensitivity[window] = {
            "left_touchdowns": value_range([item["left_touchdowns"] for item in items]),
            "right_touchdowns": value_range([item["right_touchdowns"] for item in items]),
            "left_contact_rate": value_range([item["left_contact_rate"] for item in items]),
            "right_contact_rate": value_range([item["right_contact_rate"] for item in items]),
            "single_support_rate": value_range([item["single_support_rate"] for item in items]),
            "flight_rate": value_range([item["flight_rate"] for item in items]),
            "alternating_touchdown_fraction": value_range(
                [item["alternating_touchdown_fraction"] for item in items]
            ),
        }

    per_run_window_spans = []
    for item in quality:
        windows = item["contact_debounce_sensitivity"]
        per_run_window_spans.append(
            {
                "left_touchdown_span": max(windows[name]["left_touchdowns"] for name in WINDOWS)
                - min(windows[name]["left_touchdowns"] for name in WINDOWS),
                "right_touchdown_span": max(windows[name]["right_touchdowns"] for name in WINDOWS)
                - min(windows[name]["right_touchdowns"] for name in WINDOWS),
                "single_support_rate_span": max(windows[name]["single_support_rate"] for name in WINDOWS)
                - min(windows[name]["single_support_rate"] for name in WINDOWS),
                "flight_rate_span": max(windows[name]["flight_rate"] for name in WINDOWS)
                - min(windows[name]["flight_rate"] for name in WINDOWS),
            }
        )

    contact_sources = sorted({item["contact_state_source"] for item in quality})
    slip_sources = sorted({item["stance_slip_measurement_source"] for item in quality})
    force_contact_provenance_passed = (
        contact_sources == [EXPECTED_CONTACT_SOURCE]
        and slip_sources == [EXPECTED_SLIP_SOURCE]
        and all(int(item["contact_force_sample_count"]) == int(item["sample_count"]) for item in quality)
        and all(int(item["contact_velocity_sample_count"]) > 0 for item in quality)
    )
    return {
        "name": summary["name"],
        "candidate_id": summary["candidate_id"],
        "parameters": summary["parameters"],
        "run_count": len(runs),
        "strict_pass_count": int(summary["strict_pass_count"]),
        "strict_passed": bool(summary["strict_passed"]),
        "fall_safety_hard_gate": summary["separate_hard_gates"]["fall_safety_passed"],
        "site_slip_hard_gate": summary["separate_hard_gates"]["site_slip_passed"],
        "motion_safety_envelope": {
            "minimum_speed_ratio": summary["minimum_speed_ratio"],
            "maximum_speed_ratio": summary["maximum_speed_ratio"],
            "maximum_cross_velocity_mps": summary["maximum_cross_velocity_mps"],
            "maximum_uncommanded_yaw_rate_radps": summary[
                "maximum_uncommanded_yaw_rate_radps"
            ],
            "maximum_heading_change_per_6s_rad": summary[
                "maximum_heading_change_per_6s_rad"
            ],
            "minimum_single_support_rate": summary["minimum_single_support_rate"],
            "maximum_single_support_rate": summary["maximum_single_support_rate"],
            "maximum_flight_rate": summary["maximum_flight_rate"],
            "fall_count": summary["fall_count"],
            "failed_check_counts": summary["failed_check_counts"],
        },
        "force_contact_slip": {
            "provenance_passed": force_contact_provenance_passed,
            "contact_state_sources": contact_sources,
            "slip_measurement_sources": slip_sources,
            "contact_force_sample_count": value_range(
                [item["contact_force_sample_count"] for item in quality]
            ),
            "contact_velocity_sample_count": value_range(
                [item["contact_velocity_sample_count"] for item in quality]
            ),
            "combined_rms_mps": value_range(
                [item["stance_slip_rms_mps"] for item in quality]
            ),
            "combined_p95_mps": value_range(
                [item["stance_slip_p95_mps"] for item in quality]
            ),
            "left_rms_mps": value_range(
                [item["left_stance_slip_rms_mps"] for item in quality]
            ),
            "right_rms_mps": value_range(
                [item["right_stance_slip_rms_mps"] for item in quality]
            ),
            "left_p95_mps": value_range(
                [item["left_stance_slip_p95_mps"] for item in quality]
            ),
            "right_p95_mps": value_range(
                [item["right_stance_slip_p95_mps"] for item in quality]
            ),
            "maximum_per_stance_cumulative_slip_m": value_range(
                [item["maximum_per_stance_cumulative_slip_m"] for item in quality]
            ),
        },
        "contact_debounce_sensitivity": sensitivity,
        "maximum_window_sensitivity": {
            key: max(float(item[key]) for item in per_run_window_spans)
            for key in per_run_window_spans[0]
        },
        "startup": summary["central_gait_quality"]["startup"],
        "support": summary["central_gait_quality"]["support"],
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.input.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite quality summary: {output}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    candidates = [summarize_candidate(record) for record in payload["candidates"]]
    result = {
        "schema_version": 1,
        "artifact_kind": "openduckmini_h4_reverse_force_contact_quality_summary",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DIAGNOSTIC_ONLY_NOT_ADOPTED",
        "hardware_deployment": "PROHIBITED",
        "source_artifact": str(source),
        "source_artifact_sha256": sha256(source),
        "source_dependencies": payload["dependencies"],
        "summarizer_sha256_before_output": sha256(Path(__file__).resolve()),
        "promotion_rule": "expand_only_candidates_with_all_failure_seed_strict_passes",
        "strict_pass_candidate_count": sum(item["strict_passed"] for item in candidates),
        "fall_safety_pass_candidate_count": sum(
            item["fall_safety_hard_gate"] for item in candidates
        ),
        "site_slip_pass_candidate_count": sum(
            item["site_slip_hard_gate"] for item in candidates
        ),
        "candidates": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "strict_pass_candidate_count": result["strict_pass_candidate_count"]}))


if __name__ == "__main__":
    main()
