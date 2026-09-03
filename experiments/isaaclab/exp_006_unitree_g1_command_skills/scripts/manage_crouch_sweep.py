"""Validate/migrate CROUCH evaluations and report complete sweep inventories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SCHEMA_VERSION = 2
STANDING_OPTION = "stage2_fastwalk_model4246"
REQUIRED_METRICS = (
    "settle_success_rate", "success_rate", "depth_error_m", "hold_success_rate",
    "return_success_rate", "return_height_error_m", "stand_hold_success_rate",
    "fall_rate", "foot_contact_loss_rate", "saturation_failure_rate", "residual_action_norm",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate_summary(summary_path: Path, checkpoint: Path, episodes: int, upgrade: bool) -> dict:
    summary_path, checkpoint = summary_path.resolve(), checkpoint.resolve(strict=True)
    issues = []
    if not summary_path.exists():
        return {"valid": False, "issues": ["summary_missing"], "summary": str(summary_path)}
    summary = load(summary_path)
    saved_checkpoint = Path(summary.get("checkpoint", ""))
    if not saved_checkpoint.is_absolute():
        saved_checkpoint = (summary_path.parent / saved_checkpoint).resolve()
    else:
        saved_checkpoint = saved_checkpoint.resolve()
    if saved_checkpoint != checkpoint:
        issues.append(f"checkpoint_mismatch:{saved_checkpoint}")
    if int(summary.get("episodes", 0)) < episodes:
        issues.append(f"insufficient_summary_episodes:{summary.get('episodes', 0)}<{episodes}")
    metrics = summary.get("skills", {}).get("CROUCH", {})
    issues.extend(f"missing_metric:{name}" for name in REQUIRED_METRICS if name not in metrics)
    episodes_path = summary_path.parent / "episodes.csv"
    curve_path = summary_path.parent / "crouch_curve.csv"
    episode_rows = read_rows(episodes_path) if episodes_path.exists() else []
    if len(episode_rows) < episodes:
        issues.append(f"insufficient_episode_rows:{len(episode_rows)}<{episodes}")
    candidates = {row.get("standing_base_candidate", "") for row in episode_rows}
    candidates.discard("")
    option_id = summary.get("standing_base_option_id")
    if option_id is None and candidates == {STANDING_OPTION} and upgrade:
        option_id = STANDING_OPTION
        summary["standing_base_option_id"] = option_id
    if option_id != STANDING_OPTION:
        issues.append(f"standing_option_mismatch:{option_id or sorted(candidates)}")
    schema = summary.get("evaluation_schema_version")
    legacy_compatible = schema is None and not issues and curve_path.exists()
    if legacy_compatible and upgrade:
        curve_rows = read_rows(curve_path)
        ankle_columns = ("left_ankle_pitch_residual_saturated", "right_ankle_pitch_residual_saturated")
        values = [
            float(str(row[column]).lower() == "true")
            for row in curve_rows for column in ankle_columns if column in row
        ]
        metrics["ankle_pitch_residual_saturation_fraction"] = sum(values) / len(values) if values else 0.0
        metrics["dangerous_contact_failure_rate"] = metrics["foot_contact_loss_rate"]
        summary["evaluation_schema_version"] = SCHEMA_VERSION
        summary["schema_upgrade"] = {
            "from": "legacy_unversioned", "to": SCHEMA_VERSION,
            "metrics_recomputed": False, "source": "existing episodes.csv and crouch_curve.csv",
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        schema = SCHEMA_VERSION
    if schema != SCHEMA_VERSION:
        issues.append(f"schema_mismatch:{schema}!={SCHEMA_VERSION}")
    return {
        "valid": not issues, "issues": issues, "summary": str(summary_path),
        "checkpoint": str(checkpoint), "episodes": int(summary.get("episodes", 0)),
        "schema_version": schema, "standing_base_option_id": option_id,
        "reused_existing_evaluation": bool(summary.get("schema_upgrade")),
    }


def expected_manifest(model_root: Path, checkpoint: Path, episodes: int, write: bool) -> dict:
    model_root = model_root.resolve()
    validation = validate_summary(model_root / "crouch/normal/summary.json", checkpoint, episodes, False)
    expected = {
        "crouch_normal_summary": validation["valid"],
        "command_diagnostic": (model_root / "command_diagnostic.json").exists(),
        "retention_provenance": (model_root / "retention_provenance.json").exists(),
        "gate": (model_root / "gate.json").exists(),
    }
    missing = [name for name, present in expected.items() if not present and name != "gate"]
    report = {
        "evaluation_schema_version": SCHEMA_VERSION,
        "model": model_root.name,
        "checkpoint": str(checkpoint.resolve(strict=True)),
        "checkpoint_ancestry": {"training_run": str(checkpoint.resolve().parent), "iteration": int(model_root.name.removeprefix("model_"))},
        "expected_outputs": expected,
        "missing_before_gate": missing,
        "ready_for_gate": not missing,
        "summary_validation": validation,
    }
    if write:
        (model_root / "evaluation_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--upgrade-compatible", action="store_true")
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--sweep-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    if args.summary:
        if not args.checkpoint:
            parser.error("--summary requires --checkpoint")
        report = validate_summary(args.summary, args.checkpoint, args.episodes, args.upgrade_compatible)
    elif args.model_root:
        if not args.checkpoint:
            parser.error("--model-root requires --checkpoint")
        report = expected_manifest(args.model_root, args.checkpoint, args.episodes, args.write_manifest)
    elif args.sweep_root:
        if not args.run_dir:
            parser.error("--sweep-root requires --run-dir")
        reports = []
        for checkpoint in sorted(args.run_dir.resolve().glob("model_*.pt"), key=lambda p: int(p.stem.removeprefix("model_"))):
            reports.append(expected_manifest(args.sweep_root / checkpoint.stem, checkpoint, args.episodes, args.write_manifest))
        report = {
            "sweep_root": str(args.sweep_root.resolve()), "run_name": args.run_dir.resolve().name,
            "models": reports,
            "all_missing_before_gate": {
                entry["model"]: entry["missing_before_gate"] for entry in reports if entry["missing_before_gate"]
            },
            "ready_for_gate": all(entry["ready_for_gate"] for entry in reports),
        }
    else:
        parser.error("choose --summary, --model-root, or --sweep-root")
    print(json.dumps(report, indent=2))
    if not report.get("valid", report.get("ready_for_gate", False)):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
