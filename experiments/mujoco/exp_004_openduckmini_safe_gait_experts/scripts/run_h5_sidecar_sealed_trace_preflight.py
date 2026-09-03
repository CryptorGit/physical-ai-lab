"""Verify pure H5 sidecar semantics from a sealed CPU collector trace.

This program deliberately does not construct an environment or invoke a
simulator.  It loads only a hash-bound ``.npz`` trace written by the separate
V4 CPU collector gate, then runs the pure NumPy H5 scoring functions.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from safe_gait_experts.h5_sidecar_quality import (  # noqa: E402
    H5_V3_SIDECAR_QUALITY_CONTRACT_ID,
    H5SidecarDebounceCarry,
    h5_sidecar_score_control_tick,
    h5_sidecar_weighted_reward_delta,
    initialize_h5_sidecar_debounce_carry,
)
from safe_gait_experts.h5_substep_contact_alignment import (  # noqa: E402
    h5_all_substep_quality_update,
)


SEALED_TRACE_FIELD_ORDER = (
    "time_s",
    "normalized_normal_force",
    "tangential_speed_m_s",
    "terminal_after_tick",
    "reset_normalized_force",
    "base_reward",
)
PARENT_REQUIRED_CHECKS = (
    "capture_same_arm_full_raw_equal",
    "capture_same_arm_core_raw_equal",
    "baseline_same_arm_core_raw_equal",
    "capture_vs_baseline_initial_core_raw_equal",
    "capture_vs_baseline_final_core_raw_equal",
    "capture_vs_baseline_history_core_raw_equal",
    "trace_repeat_raw_equal",
    "collector_stablehlo_has_no_h5_substep_token",
    "collector_stablehlo_exactly_one_fail_closed_cpu_callback",
)
LOSS_FIELDS = (
    "strict20ms_slip_rms_loss",
    "slip_tail_loss",
    "force_tail_loss",
    "force_qualified_sample_count",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_array_digest(value: Any) -> str:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("sealed trace arrays may not have object dtype")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def raw_array_equal(left: Any, right: Any) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    return bool(
        left_array.dtype == right_array.dtype
        and left_array.shape == right_array.shape
        and np.ascontiguousarray(left_array).tobytes(order="C")
        == np.ascontiguousarray(right_array).tobytes(order="C")
    )


def ordered_array_bundle_digest(fields: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, value in fields.items():
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(raw_array_digest(value).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def raw_value_digest(value: Any) -> str:
    """Hash array/namedtuple structure without numeric tolerance."""

    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, np.ndarray) or isinstance(item, np.generic):
            digest.update(b"array\0")
            digest.update(raw_array_digest(item).encode("ascii"))
            return
        if isinstance(item, tuple) and hasattr(item, "_fields"):
            digest.update(
                f"namedtuple:{type(item).__module__}.{type(item).__qualname__}\0".encode(
                    "utf-8"
                )
            )
            for name in item._fields:
                digest.update(name.encode("utf-8"))
                digest.update(b"\0")
                visit(getattr(item, name))
            return
        if isinstance(item, tuple):
            digest.update(b"tuple\0")
            for nested in item:
                visit(nested)
            return
        if isinstance(item, list):
            digest.update(b"list\0")
            for nested in item:
                visit(nested)
            return
        if isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item):
                digest.update(str(key).encode("utf-8"))
                digest.update(b"\0")
                visit(item[key])
            return
        digest.update(f"scalar:{type(item).__qualname__}:{item!r}\0".encode("utf-8"))

    visit(value)
    return digest.hexdigest()


def raw_value_equal(left: Any, right: Any) -> bool:
    return raw_value_digest(left) == raw_value_digest(right)


def standalone_source_has_no_environment_runtime_call(source: str) -> bool:
    """Reject direct simulator imports and method-style environment stepping."""

    tree = ast.parse(source)
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".")[0])
    has_step_call = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "step"
        for node in ast.walk(tree)
    )
    return not bool(imported_roots & {"brax", "mujoco"}) and not has_step_call


def resolve_artifact_path(raw_path: str) -> Path:
    """Resolve a parent path when the producer and verifier use different OSes."""

    candidate = Path(raw_path)
    if candidate.is_file():
        return candidate.resolve()
    parts = raw_path.replace("\\", "/").split("/")
    if len(parts) >= 4 and parts[0] == "" and parts[1] == "mnt" and len(parts[2]) == 1:
        windows_candidate = Path(f"{parts[2].upper()}:/{'/'.join(parts[3:])}")
        if windows_candidate.is_file():
            return windows_candidate.resolve()
    raise FileNotFoundError(f"sealed trace parent path does not exist: {raw_path}")


def load_json_strict(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("sealed sidecar parent must be a JSON object")
    return value


def load_sealed_trace(parent: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sealed = parent.get("sealed_trace")
    if not isinstance(sealed, dict):
        raise RuntimeError("parent has no sealed trace export")
    if tuple(sealed.get("field_order", ())) != SEALED_TRACE_FIELD_ORDER:
        raise RuntimeError("sealed trace field order drifted")
    trace_path = resolve_artifact_path(str(sealed.get("path", "")))
    if sha256_file(trace_path) != sealed.get("sha256"):
        raise RuntimeError("sealed trace file SHA-256 does not match parent")
    with np.load(trace_path, allow_pickle=False) as loaded:
        if set(loaded.files) != set(SEALED_TRACE_FIELD_ORDER):
            raise RuntimeError("sealed trace keys do not match contract")
        fields = {
            name: np.array(loaded[name], copy=True)
            for name in SEALED_TRACE_FIELD_ORDER
        }
    expected_digests = sealed.get("field_raw_bytes_sha256", {})
    actual_digests = {name: raw_array_digest(value) for name, value in fields.items()}
    if actual_digests != expected_digests:
        raise RuntimeError("sealed trace field bytes do not match parent")
    if ordered_array_bundle_digest(fields) != sealed.get("ordered_field_bundle_sha256"):
        raise RuntimeError("sealed trace bundle digest does not match parent")
    return fields, {"path": str(trace_path), **sealed}


def validate_parent(parent: Mapping[str, Any], *, parent_path: Path) -> dict[str, bool]:
    sources = parent.get("bound_inputs_pre_and_post", {})
    no_ppo = parent.get("no_ppo_tripwire", {})
    checks = parent.get("checks", {})
    source_paths = {
        "h4_training_alignment": EXP_ROOT / "safe_gait_experts/h4_training_alignment.py",
        "h5_substep_contact_alignment": EXP_ROOT
        / "safe_gait_experts/h5_substep_contact_alignment.py",
        "h5_sidecar_quality": EXP_ROOT / "safe_gait_experts/h5_sidecar_quality.py",
    }
    result = {
        "parent_sha256_readable": parent_path.is_file(),
        "parent_status_cpu_collector_pass": parent.get("status")
        == "V4_COLLECTOR_TRACE_RAW_PARITY_PASS_NOT_A_TRAINING_CANDIDATE",
        "parent_hardware_prohibited": parent.get("hardware_deployment") == "PROHIBITED",
        "parent_all_required_collector_checks_pass": all(
            checks.get(name) is True for name in PARENT_REQUIRED_CHECKS
        ),
        "parent_no_ppo": (
            no_ppo.get("ppo_train_called") is False
            and no_ppo.get("checkpoint_written") is False
            and no_ppo.get("training_run_directory_created") is False
            and no_ppo.get("preflight_returns_before_ppo_path") is True
        ),
        "parent_trace_raw_repeat_passed": parent.get("trace_repeat", {}).get("raw_equal")
        is True,
        "h4_source_matches_parent": sources.get("h4_training_alignment", {}).get(
            "sha256"
        )
        == sha256_file(source_paths["h4_training_alignment"]),
        "h5_substep_source_matches_parent": sources.get(
            "h5_substep_contact_alignment", {}
        ).get("sha256")
        == sha256_file(source_paths["h5_substep_contact_alignment"]),
        "h5_sidecar_source_matches_parent": sources.get("h5_sidecar_quality", {}).get(
            "sha256"
        )
        == sha256_file(source_paths["h5_sidecar_quality"]),
    }
    if not all(result.values()):
        failed = sorted(name for name, passed in result.items() if not passed)
        raise RuntimeError(f"sealed sidecar parent validation failed: {failed}")
    return result


def initialize_batched_carry(reset_force: np.ndarray) -> tuple[H5SidecarDebounceCarry, ...]:
    return tuple(
        initialize_h5_sidecar_debounce_carry(reset_force[lane], xp=np)
        for lane in range(reset_force.shape[0])
    )


def score_sequence(
    force: np.ndarray,
    speed: np.ndarray,
    times: np.ndarray,
    terminal_after_tick: np.ndarray,
    reset_force: np.ndarray,
    *,
    initial_carry: tuple[H5SidecarDebounceCarry, ...] | None = None,
) -> tuple[
    tuple[H5SidecarDebounceCarry, ...],
    dict[str, np.ndarray],
    np.ndarray,
    tuple[tuple[H5SidecarDebounceCarry, ...], ...],
]:
    carry = initialize_batched_carry(reset_force) if initial_carry is None else initial_carry
    loss_rows: dict[str, list[list[Any]]] = {name: [] for name in LOSS_FIELDS}
    delta_rows: list[list[Any]] = []
    carry_history: list[tuple[H5SidecarDebounceCarry, ...]] = []
    for tick in range(force.shape[0]):
        scores = tuple(
            h5_sidecar_score_control_tick(
                force[tick, lane],
                speed[tick, lane],
                times_s=times[tick, lane],
                reset_normalized_force=reset_force[lane],
                carry=carry[lane],
                terminal_after_tick=terminal_after_tick[tick, lane],
                xp=np,
            )
            for lane in range(force.shape[1])
        )
        for name in LOSS_FIELDS:
            loss_rows[name].append([getattr(score.losses, name) for score in scores])
        delta_rows.append(
            [
                h5_sidecar_weighted_reward_delta(
                    score.losses,
                    strict20ms_slip_rms_scale=-1.0,
                    slip_tail_scale=-1.0,
                    force_tail_scale=-1.0,
                    xp=np,
                )
                for score in scores
            ]
        )
        carry = tuple(score.carry for score in scores)
        carry_history.append(carry)
    return (
        carry,
        {name: np.asarray(rows) for name, rows in loss_rows.items()},
        np.asarray(delta_rows),
        tuple(carry_history),
    )


def score_known_bad_cases(reset_force: np.ndarray, dtype: np.dtype[Any]) -> tuple[Any, Any]:
    times = np.arange(10, dtype=dtype) * np.asarray(0.002, dtype=dtype)
    bad_slip = h5_sidecar_score_control_tick(
        np.full((10, 2), 0.5, dtype=dtype),
        np.full((10, 2), 0.04, dtype=dtype),
        times_s=times,
        reset_normalized_force=reset_force[0],
        carry=initialize_h5_sidecar_debounce_carry(reset_force[0], xp=np),
        terminal_after_tick=False,
        xp=np,
    )
    bad_force = h5_sidecar_score_control_tick(
        np.full((10, 2), 2.0, dtype=dtype),
        np.zeros((10, 2), dtype=dtype),
        times_s=times,
        reset_normalized_force=reset_force[0],
        carry=initialize_h5_sidecar_debounce_carry(reset_force[0], xp=np),
        terminal_after_tick=False,
        xp=np,
    )
    return bad_slip, bad_force


def run(parent_path: Path, output_path: Path) -> dict[str, Any]:
    parent_path = parent_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite sidecar evidence: {output_path}")
    parent = load_json_strict(parent_path)
    parent_validation = validate_parent(parent, parent_path=parent_path)
    fields, sealed_manifest = load_sealed_trace(parent)
    force = fields["normalized_normal_force"]
    speed = fields["tangential_speed_m_s"]
    times = fields["time_s"]
    terminal = fields["terminal_after_tick"].astype(bool)
    reset_force = fields["reset_normalized_force"]
    if (
        force.shape != (20, 2, 10, 2)
        or speed.shape != force.shape
        or times.shape != (20, 2, 10)
        or terminal.shape != (20, 2)
        or reset_force.shape != (2, 2)
        or fields["base_reward"].shape != (20, 2)
    ):
        raise RuntimeError("sealed sidecar input shape drifted from B=2/T=20")
    if not (
        np.all(np.isfinite(force))
        and np.all(np.isfinite(speed))
        and np.all(np.isfinite(times))
        and np.all(np.diff(times, axis=2) > 0.0)
    ):
        raise RuntimeError("sealed sidecar input contains invalid trace values")

    before = {name: raw_array_digest(value) for name, value in fields.items()}
    no_terminal = np.zeros_like(terminal, dtype=bool)
    full_carry, full_losses, full_delta, _full_carry_history = score_sequence(
        force, speed, times, no_terminal, reset_force
    )
    repeat_carry, repeat_losses, repeat_delta, _repeat_carry_history = score_sequence(
        force, speed, times, no_terminal, reset_force
    )
    split = 9
    prefix_carry, prefix_losses, prefix_delta, _prefix_carry_history = score_sequence(
        force[:split], speed[:split], times[:split], no_terminal[:split], reset_force
    )
    suffix_carry, suffix_losses, suffix_delta, _suffix_carry_history = score_sequence(
        force[split:],
        speed[split:],
        times[split:],
        no_terminal[split:],
        reset_force,
        initial_carry=prefix_carry,
    )
    split_losses = {
        name: np.concatenate((prefix_losses[name], suffix_losses[name]), axis=0)
        for name in LOSS_FIELDS
    }
    split_delta = np.concatenate((prefix_delta, suffix_delta), axis=0)

    direct_debounce_equal = True
    for lane in range(2):
        direct = h5_all_substep_quality_update(
            force[:, lane].reshape(200, 2),
            speed[:, lane].reshape(200, 2),
            initial_debounce=initialize_h5_sidecar_debounce_carry(
                reset_force[lane], xp=np
            ).debounce,
            times_s=times[:, lane].reshape(200),
            xp=np,
        ).debounce
        direct_debounce_equal = direct_debounce_equal and raw_value_equal(
            full_carry[lane].debounce, direct
        )

    forced_terminal = no_terminal.copy()
    forced_terminal[4, 0] = True
    forced_terminal[11, 1] = True
    forced_carry, forced_losses, forced_delta, forced_carry_history = score_sequence(
        force, speed, times, forced_terminal, reset_force
    )
    del forced_carry
    terminal_tick_loss_equal = all(
        raw_array_equal(full_losses[name][tick, lane], forced_losses[name][tick, lane])
        for tick, lane in ((4, 0), (11, 1))
        for name in LOSS_FIELDS
    )
    terminal_tick_delta_equal = all(
        raw_array_equal(full_delta[tick, lane], forced_delta[tick, lane])
        for tick, lane in ((4, 0), (11, 1))
    )

    def fresh_next_tick_equal(tick: int, lane: int, *, reset_time: bool) -> bool:
        lane_times = times[tick + 1, lane]
        if reset_time:
            lane_times = np.arange(10, dtype=lane_times.dtype) * np.asarray(
                0.002, dtype=lane_times.dtype
            )
        continued = h5_sidecar_score_control_tick(
            force[tick + 1, lane],
            speed[tick + 1, lane],
            times_s=lane_times,
            reset_normalized_force=reset_force[lane],
            carry=forced_carry_history[tick][lane],
            terminal_after_tick=False,
            xp=np,
        )
        fresh = h5_sidecar_score_control_tick(
            force[tick + 1, lane],
            speed[tick + 1, lane],
            times_s=lane_times,
            reset_normalized_force=reset_force[lane],
            carry=initialize_h5_sidecar_debounce_carry(reset_force[lane], xp=np),
            terminal_after_tick=False,
            xp=np,
        )
        return raw_value_equal(continued, fresh)

    manual_delta = (
        np.asarray(-1.0) * full_losses["strict20ms_slip_rms_loss"]
        + np.asarray(-1.0) * full_losses["slip_tail_loss"]
        + np.asarray(-1.0) * full_losses["force_tail_loss"]
    )
    bad_slip, bad_force = score_known_bad_cases(reset_force, force.dtype)
    after = {name: raw_array_digest(value) for name, value in fields.items()}
    sidecar_source = (EXP_ROOT / "safe_gait_experts/h5_sidecar_quality.py").read_text(
        encoding="utf-8"
    )
    script_source = Path(__file__).read_text(encoding="utf-8")
    checks = {
        "parent_hash_bound_and_reproduced": all(parent_validation.values()),
        "sealed_trace_file_and_field_bytes_match_parent": True,
        "sidecar_contract_id_bound": H5_V3_SIDECAR_QUALITY_CONTRACT_ID
        == "H5_V3_SIDECAR_QUALITY_20260812",
        "same_trace_sidecar_repeat_raw_equal": raw_value_equal(
            (full_carry, full_losses, full_delta),
            (repeat_carry, repeat_losses, repeat_delta),
        ),
        "unroll_boundary_result_raw_equal": raw_value_equal(
            (full_losses, full_delta), (split_losses, split_delta)
        ),
        "unroll_boundary_carry_raw_equal": raw_value_equal(full_carry, suffix_carry),
        "continuous_200_sample_debounce_raw_equal": direct_debounce_equal,
        "terminal_tick_loss_raw_equal": terminal_tick_loss_equal,
        "terminal_tick_reward_delta_raw_equal": terminal_tick_delta_equal,
        "asynchronous_terminal_next_tick_resets_raw_equal": (
            fresh_next_tick_equal(4, 0, reset_time=False)
            and fresh_next_tick_equal(11, 1, reset_time=False)
        ),
        "terminal_time_reset_next_tick_raw_equal": (
            fresh_next_tick_equal(4, 0, reset_time=True)
            and fresh_next_tick_equal(11, 1, reset_time=True)
        ),
        "weighted_reward_delta_added_once_raw_equal": raw_array_equal(
            full_delta, manual_delta
        ),
        "known_bad_slip_cost_nonzero": bool(
            np.asarray(bad_slip.losses.strict20ms_slip_rms_loss) > 0.0
            and np.asarray(bad_slip.losses.slip_tail_loss) > 0.0
        ),
        "known_bad_force_cost_nonzero": bool(
            np.asarray(bad_force.losses.force_tail_loss) > 0.0
        ),
        "sealed_input_and_base_reward_unchanged": before == after,
        "sidecar_source_has_no_simulator_or_ppo_call": not any(
            token in sidecar_source
            for token in ("mjx.", "env.step", "ppo.", "checkpoint")
        ),
        "standalone_source_has_no_environment_runtime_call": (
            standalone_source_has_no_environment_runtime_call(script_source)
        ),
    }
    payload = {
        "schema_version": 1,
        "artifact_kind": "openduckmini_h5_pure_sidecar_sealed_trace_no_ppo_preflight",
        "status": (
            "CPU_PURE_H5_SIDECAR_SEALED_TRACE_NO_PPO_PASS"
            if all(checks.values())
            else "CPU_PURE_H5_SIDECAR_SEALED_TRACE_NO_PPO_FAIL"
        ),
        "hardware_deployment": "PROHIBITED",
        "next_authority": "CPU_SIDECAR_ONLY_NO_PPO",
        "parent": {
            "path": str(parent_path),
            "sha256": sha256_file(parent_path),
            "validation": parent_validation,
            "collector_trace_raw_tree_sha256": parent.get("trace_repeat", {}).get(
                "first_raw_tree_sha256"
            ),
        },
        "sealed_trace": sealed_manifest,
        "bound_sources": {
            "standalone_sidecar_runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "h4_training_alignment": {
                "path": str(EXP_ROOT / "safe_gait_experts/h4_training_alignment.py"),
                "sha256": sha256_file(
                    EXP_ROOT / "safe_gait_experts/h4_training_alignment.py"
                ),
            },
            "h5_substep_contact_alignment": {
                "path": str(EXP_ROOT / "safe_gait_experts/h5_substep_contact_alignment.py"),
                "sha256": sha256_file(
                    EXP_ROOT / "safe_gait_experts/h5_substep_contact_alignment.py"
                ),
            },
            "h5_sidecar_quality": {
                "path": str(EXP_ROOT / "safe_gait_experts/h5_sidecar_quality.py"),
                "sha256": sha256_file(
                    EXP_ROOT / "safe_gait_experts/h5_sidecar_quality.py"
                ),
            },
        },
        "execution": {
            "runtime": "NumPy-only sealed trace scoring",
            "environment_instances": 0,
            "simulator_calls": 0,
            "ppo_calls": 0,
            "checkpoint_writes": 0,
            "reward_application": "delta_scored_only_not_added_to_env_reward",
            "batch_size": 2,
            "control_steps": 20,
            "substeps_per_control": 10,
        },
        "checks": checks,
        "raw_hashes": {
            "first_sidecar_output": raw_value_digest(
                (full_carry, full_losses, full_delta)
            ),
            "second_sidecar_output": raw_value_digest(
                (repeat_carry, repeat_losses, repeat_delta)
            ),
            "base_reward_before": before["base_reward"],
            "base_reward_after": after["base_reward"],
            "sealed_trace_bundle": ordered_array_bundle_digest(fields),
        },
        "known_bad_losses": {
            "slip": {
                name: float(np.asarray(getattr(bad_slip.losses, name)))
                for name in LOSS_FIELDS
            },
            "force": {
                name: float(np.asarray(getattr(bad_force.losses, name)))
                for name in LOSS_FIELDS
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.parent, args.output)
    print(json.dumps({"status": result["status"], "checks": result["checks"]}, indent=2))
    return 0 if result["status"].endswith("_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
