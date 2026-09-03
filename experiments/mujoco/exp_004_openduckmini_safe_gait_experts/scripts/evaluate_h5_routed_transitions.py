"""Diagnostic H5 routed strict suite.

This evaluator reuses the frozen MuJoCo safety/quality machinery while wiring
two command-conditioned 116-wide actors through the H5 target-space bridge.
It is intentionally not an adoption or release producer.  A failing suite is
still written as evidence so the next PDCA cycle has complete route metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Mapping, Sequence

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))

from safe_gait_experts.h4_post_training import (  # noqa: E402
    H4_ACTOR_OBSERVATION_WIDTH,
    infer_h4_action_numpy,
    mask_h4_head_action,
    sha256_file,
    validate_h4_params,
)
from safe_gait_experts.h5_routed_policy import (  # noqa: E402
    H5DomainCandidate,
    H5RoutedPolicyBank,
)
from safe_gait_experts.h5_target_contract import (  # noqa: E402
    H5_CONTRACT_ID,
    h5_domain_for_route,
)
from safe_gait_experts.h5_command_contract import (  # noqa: E402
    H5_COMMAND_CONTRACT_ID,
    H5_REVERSE_PHASE_DELTA_BINS_PER_CONTROL,
    H5_UNIFIED_COMMAND_MAPPER_LEGACY_V2,
    H5_UNIFIED_COMMAND_MAPPER_DIRECT_V3,
    H5_UNIFIED_COMMAND_MAPPER_SUPPORTED_MODES,
    canonical_h5_unified_command_mapper,
    h5_unified_direct_policy_command,
    h5_unified_command_contract_id,
    h5_unified_command_contract_manifest,
    h5_planar_policy_command,
    h5_reverse_policy_command,
    h5_unified_policy_command,
)
from router import RouterConfig  # noqa: E402
from safe_gait_experts.routed_evaluation import (  # noqa: E402
    REQUIRED_POLICY_ROLES,
    canonical_policy_role,
    parse_policy_assignments,
    suite_acceptance,
    AcceptanceThresholds,
)
from scripts import evaluate_h4_routed_transitions as h4  # noqa: E402
from scripts import evaluate_routed_transitions as routed  # noqa: E402


_H5_UNIFIED_SINGLE_WEIGHT_MODE = False
_H5_UNIFIED_COMMAND_MAPPER = "legacy_h4_compensated"


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _sha(value: str) -> str:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise argparse.ArgumentTypeError("SHA256 must be lowercase hex")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", action="append", required=True, metavar="ROLE=PATH")
    parser.add_argument("--generated-root", type=_path, required=True)
    parser.add_argument("--output", type=_path, required=True)
    parser.add_argument(
        "--h5-substep-capture-npz",
        type=_path,
        default=None,
        help=(
            "Optional simulation-only capture of the exact 2 ms force/slip "
            "payload consumed by the frozen gait-quality evaluator.  This "
            "does not change the policy, target guard, or physics trajectory."
        ),
    )
    parser.add_argument(
        "--h5-policy-observation-capture-npz",
        type=_path,
        default=None,
        help=(
            "Optional simulation-only capture of exact 116-wide actor inputs "
            "for an offline phase-use probe.  It does not alter inference."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--transition-seconds", type=float, default=6.0)
    parser.add_argument("--transition-stand-seconds", type=float, default=2.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.5)
    parser.add_argument("--initial-joint-noise-scale", type=float, default=0.0)
    parser.add_argument("--initial-base-speed", type=float, default=0.0)
    parser.add_argument(
        "--unified-single-weight",
        action="store_true",
        help=(
            "Use the same 116-wide H5 weight for both candidate domains and "
            "map the physical [vx, vy, wz] command continuously through the "
            "unified H5 command contract."
        ),
    )
    parser.add_argument(
        "--unified-command-mapper",
        choices=H5_UNIFIED_COMMAND_MAPPER_SUPPORTED_MODES,
        default="legacy_h4_compensated",
        help=(
            "Policy-observation map for a unified-weight diagnostic. "
            "direct_normalized_v3 is a no-training ablation and does not make "
            "the current weight deployable under that contract."
        ),
    )
    for domain in ("planar", "reverse"):
        parser.add_argument(f"--h5-{domain}-params", type=_path, required=True)
        parser.add_argument(f"--h5-{domain}-params-sha256", type=_sha, required=True)
        parser.add_argument(f"--h5-{domain}-manifest", type=_path, required=True)
        parser.add_argument(f"--h5-{domain}-manifest-sha256", type=_sha, required=True)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Return nonzero when every strict route/transition gate does not pass.",
    )
    return parser


def _load_candidate(args: argparse.Namespace, domain: str) -> H5DomainCandidate:
    params_path = getattr(args, f"h5_{domain}_params").resolve()
    manifest_path = getattr(args, f"h5_{domain}_manifest").resolve()
    params_sha = getattr(args, f"h5_{domain}_params_sha256")
    manifest_sha = getattr(args, f"h5_{domain}_manifest_sha256")
    if sha256_file(params_path) != params_sha:
        raise ValueError(f"H5 {domain} params SHA256 mismatch")
    if sha256_file(manifest_path) != manifest_sha:
        raise ValueError(f"H5 {domain} manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "COMPLETED"
        or manifest.get("expert") != domain
        or manifest.get("qualification_use")
        != "DIAGNOSTIC_REWARD_EXPLORATION_NOT_QUALIFICATION"
        or manifest.get("hardware_deployment") != "PROHIBITED"
    ):
        raise ValueError(f"H5 {domain} candidate is not a completed diagnostic run")
    output_record = manifest.get("outputs", {}).get("final_params", {})
    if Path(str(output_record.get("path", ""))).resolve() != params_path:
        raise ValueError(f"H5 {domain} final params path is not manifest-bound")
    with params_path.open("rb") as stream:
        params = pickle.load(stream)
    audit = validate_h4_params(params)
    if audit.get("actor_observation_width") != H4_ACTOR_OBSERVATION_WIDTH:
        raise ValueError(f"H5 {domain} actor observation width is not 116")
    training = _load_training_command_provenance(
        manifest,
        params_path=params_path,
        params_sha256=params_sha,
        domain=domain,
    )
    return H5DomainCandidate(
        domain=domain,
        params=params,
        params_sha256=params_sha,
        manifest_sha256=manifest_sha,
        training_manifest_sha256=training["training_manifest_sha256"],
        training_resolved_config_sha256=training["training_resolved_config_sha256"],
        training_command_contract_id=training["training_command_contract_id"],
        training_command_mapper=training["training_command_mapper"],
        training_command_mapper_inferred_from_v2_contract=training[
            "training_command_mapper_inferred_from_v2_contract"
        ],
    )


def _read_bound_json_record(
    record: Mapping[str, Any], *, label: str
) -> tuple[Path, dict[str, Any], str]:
    """Read one JSON artifact only after validating its recorded SHA256."""

    path = Path(str(record.get("path", ""))).resolve()
    expected_sha = record.get("sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError(f"{label} has no valid SHA256 binding")
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ValueError(f"{label} hash/path binding mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return path, payload, expected_sha


def _load_training_command_provenance(
    candidate_manifest: Mapping[str, Any],
    *,
    params_path: Path,
    params_sha256: str,
    domain: str,
) -> dict[str, Any]:
    """Resolve a candidate's immutable unified-training command contract.

    A wrapper can use either the old ``source_manifest`` spelling or the
    builder's ``source_candidate`` spelling.  V2 artifacts predate an explicit
    mapper field, so and only so a missing mapper is migrated to legacy V2 when
    its immutable command-contract ID proves that interpretation.
    """

    source_records = [
        (label, candidate_manifest.get(label))
        for label in ("source_manifest", "source_candidate")
        if candidate_manifest.get(label) is not None
    ]
    if len(source_records) != 1 or not isinstance(source_records[0][1], Mapping):
        raise ValueError(f"H5 {domain} candidate requires one bound source run manifest")
    source_label, raw_source = source_records[0]
    training_path, training_manifest, training_sha = _read_bound_json_record(
        raw_source, label=f"H5 {domain} {source_label}"
    )
    if (
        training_manifest.get("status") != "COMPLETED"
        or training_manifest.get("expert") != "unified"
        or training_manifest.get("hardware_deployment") != "PROHIBITED"
        or training_manifest.get("qualification_use")
        != "DIAGNOSTIC_REWARD_EXPLORATION_NOT_QUALIFICATION"
    ):
        raise ValueError(f"H5 {domain} source run is not a completed unified diagnostic")
    source_params = training_manifest.get("outputs", {}).get("final_params", {})
    if (
        Path(str(source_params.get("path", ""))).resolve() != params_path
        or source_params.get("sha256") != params_sha256
    ):
        raise ValueError(f"H5 {domain} source run final params are not wrapper-bound")
    raw_config = training_manifest.get("resolved_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError(f"H5 {domain} source run has no bound resolved config")
    _, config, config_sha = _read_bound_json_record(
        raw_config, label=f"H5 {domain} source resolved config"
    )
    contract_id = config.get("h5_command_contract_id")
    contract_manifest = config.get("h5_command_contract")
    if (
        not isinstance(contract_id, str)
        or not isinstance(contract_manifest, Mapping)
        or contract_manifest.get("contract_id") != contract_id
    ):
        raise ValueError(f"H5 {domain} source command contract is unbound")
    raw_mapper = config.get("h5_unified_command_mapper")
    inferred = False
    if raw_mapper is None:
        if contract_id != H5_COMMAND_CONTRACT_ID:
            raise ValueError(f"H5 {domain} source mapper is absent outside legacy V2")
        mapper = H5_UNIFIED_COMMAND_MAPPER_LEGACY_V2
        inferred = True
    else:
        mapper = canonical_h5_unified_command_mapper(str(raw_mapper))
        if h5_unified_command_contract_id(mapper) != contract_id:
            raise ValueError(f"H5 {domain} source mapper/contract mismatch")
    return {
        "training_manifest_path": str(training_path),
        "training_manifest_sha256": training_sha,
        "training_resolved_config_sha256": config_sha,
        "training_command_contract_id": contract_id,
        "training_command_mapper": mapper,
        "training_command_mapper_inferred_from_v2_contract": inferred,
    }


class H5RoutedSimulator(h4.H4RoutedSimulator):
    """Use H5's decoded desired target while frozen schedule owns the guard."""

    def __init__(self, *args: Any, **kwargs: Any):
        strict_actor_only = bool(kwargs.pop("strict_actor_only", False))
        super().__init__(*args, **kwargs)
        self.strict_actor_only = strict_actor_only
        self._h5_trace_records: dict[int, dict[str, Any]] = {}
        self._h5_substep_measurement_records: list[dict[str, Any]] | None = None
        self._h5_policy_observation_records: list[dict[str, Any]] | None = None
        if strict_actor_only:
            # The copied frozen schedule asks the evaluator for a backward
            # phase delta before advancing the route.  H5 owns that phase
            # contract, so do not let a legacy backward profile participate in
            # strict actor execution even as an incidental timing source.
            self.evaluator.backward_parameters = self._h5_backward_parameters

    def enable_h5_substep_measurement_capture(self) -> None:
        """Enable append-only capture of evaluator-owned 2 ms measurements."""

        if self._h5_substep_measurement_records is not None:
            raise RuntimeError("H5 substep measurement capture is already enabled")
        self._h5_substep_measurement_records = []

    def _capture_substep_measurement(self, **record: Any) -> None:
        """Receive a copied payload from the frozen post-physics evaluator hook."""

        records = self._h5_substep_measurement_records
        if records is None:
            return
        expected = {
            "run_seed",
            "segment_index",
            "segment_name",
            "time_s",
            "normal_force_fraction",
            "tangential_speed_mps",
        }
        if set(record) != expected:
            raise RuntimeError("unexpected H5 substep measurement capture payload")
        normal_force = np.asarray(record["normal_force_fraction"], dtype=np.float64)
        speed = np.asarray(record["tangential_speed_mps"], dtype=np.float64)
        if (
            normal_force.shape != (2,)
            or speed.shape != (2,)
            or not np.all(np.isfinite(normal_force))
            or not np.all(np.isfinite(speed))
            or np.any(normal_force < 0.0)
            or np.any(speed < 0.0)
        ):
            raise RuntimeError("invalid H5 substep force/slip measurement payload")
        records.append(
            {
                "run_seed": int(record["run_seed"]),
                "segment_index": int(record["segment_index"]),
                "segment_name": str(record["segment_name"]),
                "time_s": float(record["time_s"]),
                "normal_force_fraction": normal_force.copy(),
                "tangential_speed_mps": speed.copy(),
            }
        )

    def export_h5_substep_measurement_capture(self) -> dict[str, np.ndarray]:
        """Return a typed, lossless payload after a completed simulation suite."""

        records = self._h5_substep_measurement_records
        if not records:
            raise RuntimeError("H5 substep measurement capture is empty or disabled")
        return {
            "run_seed": np.asarray([row["run_seed"] for row in records], dtype=np.int64),
            "segment_index": np.asarray(
                [row["segment_index"] for row in records], dtype=np.int32
            ),
            "segment_name": np.asarray([row["segment_name"] for row in records]),
            "time_s": np.asarray([row["time_s"] for row in records], dtype=np.float64),
            "normal_force_fraction": np.asarray(
                [row["normal_force_fraction"] for row in records], dtype=np.float64
            ),
            "tangential_speed_mps": np.asarray(
                [row["tangential_speed_mps"] for row in records], dtype=np.float64
            ),
        }

    def enable_h5_policy_observation_capture(self) -> None:
        """Enable append-only 116-wide actor-input capture for offline probes."""

        if self._h5_policy_observation_records is not None:
            raise RuntimeError("H5 policy observation capture is already enabled")
        self._h5_policy_observation_records = []

    def _capture_policy_observation(self, **record: Any) -> None:
        records = self._h5_policy_observation_records
        if records is None:
            return
        expected = {
            "run_seed",
            "segment_index",
            "segment_name",
            "control_step",
            "observation",
        }
        if set(record) != expected:
            raise RuntimeError("unexpected H5 policy observation capture payload")
        observation = np.asarray(record["observation"], dtype=np.float32)
        if (
            observation.shape != (H4_ACTOR_OBSERVATION_WIDTH,)
            or not np.all(np.isfinite(observation))
        ):
            raise RuntimeError("invalid H5 policy observation capture payload")
        records.append(
            {
                "run_seed": int(record["run_seed"]),
                "segment_index": int(record["segment_index"]),
                "segment_name": str(record["segment_name"]),
                "control_step": int(record["control_step"]),
                "observation": observation.copy(),
            }
        )

    def export_h5_policy_observation_capture(self) -> dict[str, np.ndarray]:
        records = self._h5_policy_observation_records
        if not records:
            raise RuntimeError("H5 policy observation capture is empty or disabled")
        return {
            "run_seed": np.asarray([row["run_seed"] for row in records], dtype=np.int64),
            "segment_index": np.asarray(
                [row["segment_index"] for row in records], dtype=np.int32
            ),
            "segment_name": np.asarray([row["segment_name"] for row in records]),
            "control_step": np.asarray(
                [row["control_step"] for row in records], dtype=np.int32
            ),
            "observation": np.asarray(
                [row["observation"] for row in records], dtype=np.float32
            ),
        }

    @staticmethod
    def _trace_update_array(hasher: Any, values: np.ndarray) -> None:
        array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
        hasher.update(array.tobytes(order="C"))

    def _capture_control_trace(
        self,
        *,
        segment_index: int,
        segment_name: str,
        control_step: int,
        global_control_tick: int,
        requested_command: np.ndarray,
        effective_command: np.ndarray,
        policy_command: np.ndarray,
        raw_action: np.ndarray,
        candidate_targets: np.ndarray,
        desired_targets: np.ndarray,
        applied_targets: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> None:
        """Hash one complete H5 control tick for paired-contract evidence."""

        record = self._h5_trace_records.setdefault(
            int(segment_index),
            {
                "segment_name": str(segment_name),
                "hasher": hashlib.sha256(),
                "control_tick_count": 0,
                "policy_command_max_abs_error": 0.0,
            },
        )
        hasher = record["hasher"]
        hasher.update(np.asarray((segment_index, control_step, global_control_tick), dtype="<i8").tobytes())
        for values in (
            requested_command,
            effective_command,
            policy_command,
            raw_action,
            candidate_targets,
            desired_targets,
            applied_targets,
            qpos,
            qvel,
        ):
            self._trace_update_array(hasher, values)
        if _H5_UNIFIED_SINGLE_WEIGHT_MODE:
            expected_policy = (
                h5_unified_direct_policy_command(effective_command)
                if _H5_UNIFIED_COMMAND_MAPPER
                == H5_UNIFIED_COMMAND_MAPPER_DIRECT_V3
                else h5_unified_policy_command(effective_command)
            )
            error = float(
                np.max(np.abs(np.asarray(policy_command) - expected_policy))
            )
            record["policy_command_max_abs_error"] = max(
                float(record["policy_command_max_abs_error"]), error
            )
        record["control_tick_count"] = int(record["control_tick_count"]) + 1

    def run_schedule(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Attach exact H5 control-trace digests without changing H3 output."""

        self._h5_trace_records = {}
        result = super().run_schedule(*args, **kwargs)
        total_control_ticks = 0
        for index, segment in enumerate(result["segments"]):
            record = self._h5_trace_records.get(index)
            if record is None:
                trace = {
                    "schema_version": 1,
                    "segment_name": segment["name"],
                    "control_tick_count": 0,
                    "trace_sha256": hashlib.sha256(b"").hexdigest(),
                    "policy_command_max_abs_error": None,
                }
            else:
                trace = {
                    "schema_version": 1,
                    "segment_name": record["segment_name"],
                    "control_tick_count": int(record["control_tick_count"]),
                    "trace_sha256": record["hasher"].hexdigest(),
                    "policy_command_max_abs_error": (
                        float(record["policy_command_max_abs_error"])
                        if _H5_UNIFIED_SINGLE_WEIGHT_MODE
                        else None
                    ),
                }
            total_control_ticks += int(trace["control_tick_count"])
            segment["h5_control_trace"] = trace
        guard_calls = int(
            result["backward_exit_recovery_audit"]["final_guard_call_count"]
        )
        if guard_calls != total_control_ticks:
            raise RuntimeError("H5 trace guard-call count does not match control ticks")
        direct_fidelity_errors = [
            trace["policy_command_max_abs_error"]
            for segment in result["segments"]
            if (trace := segment["h5_control_trace"])["policy_command_max_abs_error"]
            is not None
        ]
        maximum_fidelity_error = (
            float(max(direct_fidelity_errors)) if direct_fidelity_errors else None
        )
        if (
            _H5_UNIFIED_SINGLE_WEIGHT_MODE
            and _H5_UNIFIED_COMMAND_MAPPER == H5_UNIFIED_COMMAND_MAPPER_DIRECT_V3
            and maximum_fidelity_error is not None
            and maximum_fidelity_error > 1e-12
        ):
            raise RuntimeError("V3 direct policy command differs from its contract")
        result["h5_trace_protocol"] = {
            "schema_version": 1,
            "contents": [
                "requested_command",
                "effective_command",
                "policy_command",
                "raw_action",
                "candidate_targets",
                "desired_targets",
                "applied_targets",
                "post_control_qpos",
                "post_control_qvel",
            ],
            "total_control_ticks": total_control_ticks,
            "final_guard_call_count": guard_calls,
            "exactly_one_guard_call_per_control_tick": True,
            "mapper": (
                _H5_UNIFIED_COMMAND_MAPPER
                if _H5_UNIFIED_SINGLE_WEIGHT_MODE
                else None
            ),
            "maximum_policy_command_contract_error": maximum_fidelity_error,
        }
        return result

    @staticmethod
    def _h5_backward_parameters(
        yaw_command: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        del yaw_command
        return (
            np.zeros(10, dtype=np.float64),
            np.zeros(10, dtype=np.float64),
            H5_REVERSE_PHASE_DELTA_BINS_PER_CONTROL,
        )

    def _policy_target(
        self,
        applied_action: np.ndarray,
        effective_command: np.ndarray,
        phase_index: float,
        default: np.ndarray,
    ) -> np.ndarray:
        bank = self.bank
        step = getattr(bank, "last_step", None)
        if step is None:
            raise RuntimeError("H5 target bridge did not publish a route step")
        desired = np.asarray(step.blended_targets, dtype=np.float64)
        if desired.shape != (14,) or not np.all(np.isfinite(desired)):
            raise RuntimeError("H5 desired target is not finite and 14-wide")
        # H5 is actor-authoritative in every domain, including reverse.  The
        # frozen H4 reverse feedforward/profile is intentionally not reachable
        # from the strict H5 evaluator: otherwise a failing actor can be
        # masked by an unrelated legacy target generator.
        return desired


class H5SafeGaitRouter(routed.SafeGaitRouter):
    """H5 router with a reachable -0.03 lateral compound endpoint.

    The frozen default linear hysteresis enters at 0.033 m/s, while the
    formal forward-lateral-right endpoint requests -0.03 m/s. H5 owns this
    diagnostic router contract and lowers only the linear hysteresis to 0.005
    (entry 0.030, exit 0.020); yaw and slew guards remain unchanged.
    """

    LINEAR_HYSTERESIS = 0.005

    def __init__(self) -> None:
        super().__init__(RouterConfig(linear_hysteresis=self.LINEAR_HYSTERESIS))


def _h5_resolve_policy_observation_command(
    routed_expert: str,
    effective_command: Sequence[float],
    *,
    backward_residual_scale: float,
    override: Sequence[float] | None = None,
) -> tuple[np.ndarray, float, bool]:
    """Mirror the H5 training mapper selected by the evaluation contract."""

    effective = np.asarray(effective_command, dtype=np.float64)
    if effective.shape != (3,) or not np.all(np.isfinite(effective)):
        raise ValueError("H5 effective command must be finite and 3-wide")
    h4._H4_RUN_CONTEXT["physical_command"] = effective.copy()
    if _H5_UNIFIED_SINGLE_WEIGHT_MODE:
        # The frozen routed cases carry legacy per-domain policy overrides.
        # They are not allowed to shadow the one unified [vx, vy, wz]
        # command contract: otherwise strict evaluation would silently test a
        # different observation map from unified training.
        policy = (
            h5_unified_direct_policy_command(effective)
            if _H5_UNIFIED_COMMAND_MAPPER == H5_UNIFIED_COMMAND_MAPPER_DIRECT_V3
            else h5_unified_policy_command(effective)
        )
        return policy, 0.0, True
    if override is not None:
        policy = np.asarray(override, dtype=np.float64)
        if policy.shape != (3,) or not np.all(np.isfinite(policy)):
            raise ValueError("H5 policy override must be finite and 3-wide")
        return policy.copy(), 0.0, True
    domain = h5_domain_for_route(str(routed_expert))
    policy = (
        h5_reverse_policy_command(effective)
        if domain == "reverse"
        else h5_planar_policy_command(effective)
    )
    return policy, 0.0, True


def _h5_advance_routed_phase(
    phase_index: float,
    *,
    phase_steps: float,
    phase_delta: float,
    current_expert: str,
    previous_expert: str | None,
    effective_command: Sequence[float],
    previous_backward_feedforward_active: bool,
    diagnostic_entry_phase_indices: Mapping[str, float] | None = None,
    phase_entry_status: str = "H5_SIMULATION_DIAGNOSTIC",
    diagnostic_only: bool = True,
    control_step: int | None = None,
    global_control_tick: int | None = None,
) -> tuple[float, bool, dict[str, Any] | None]:
    """Use H5 reverse entry 7.0 without invoking the frozen 6.0 validator."""

    del diagnostic_entry_phase_indices, diagnostic_only
    command = np.asarray(effective_command, dtype=np.float64)
    active = bool(command[0] < -0.02)
    if h5_domain_for_route(str(current_expert)) == "reverse" and active:
        return h4.H4RoutedPolicyBank.advance_phase(
            phase_index=phase_index,
            phase_steps=phase_steps,
            phase_delta=H5_REVERSE_PHASE_DELTA_BINS_PER_CONTROL,
            current_expert=current_expert,
            previous_expert=previous_expert,
            effective_command=effective_command,
            previous_backward_feedforward_active=previous_backward_feedforward_active,
            control_step=0 if control_step is None else control_step,
            global_control_tick=0 if global_control_tick is None else global_control_tick,
        )
    # The frozen helper is used for non-reverse phase continuity, but no H4
    # diagnostic reverse-entry map is supplied because H5 owns its 7.0 entry.
    return routed.advance_routed_phase(
        phase_index,
        phase_steps=phase_steps,
        phase_delta=phase_delta,
        current_expert=current_expert,
        previous_expert=previous_expert,
        effective_command=effective_command,
        previous_backward_feedforward_active=previous_backward_feedforward_active,
        diagnostic_entry_phase_indices=None,
        phase_entry_status=phase_entry_status,
        diagnostic_only=False,
        control_step=control_step,
        global_control_tick=global_control_tick,
    )


def _build_simulator(args: argparse.Namespace) -> tuple[Any, H5RoutedPolicyBank, dict[str, Any]]:
    global _H5_UNIFIED_COMMAND_MAPPER, _H5_UNIFIED_SINGLE_WEIGHT_MODE
    _H5_UNIFIED_SINGLE_WEIGHT_MODE = bool(args.unified_single_weight)
    _H5_UNIFIED_COMMAND_MAPPER = canonical_h5_unified_command_mapper(
        str(getattr(args, "unified_command_mapper", "legacy_h4_compensated"))
    )
    strict_actor_only = bool(getattr(args, "strict_actor_only", False))
    policies = parse_policy_assignments(args.policy)
    if set(policies) != set(REQUIRED_POLICY_ROLES):
        raise ValueError("exactly the eight legacy policy roles are required")
    candidates = {
        domain: _load_candidate(args, domain) for domain in ("planar", "reverse")
    }
    unified_training_binding: dict[str, Any] | None = None
    if _H5_UNIFIED_SINGLE_WEIGHT_MODE:
        if candidates["planar"].params_sha256 != candidates["reverse"].params_sha256:
            raise ValueError(
                "--unified-single-weight requires identical planar/reverse params SHA256"
            )
        provenance_fields = (
            "training_manifest_sha256",
            "training_resolved_config_sha256",
            "training_command_contract_id",
            "training_command_mapper",
            "training_command_mapper_inferred_from_v2_contract",
        )
        for field in provenance_fields:
            if getattr(candidates["planar"], field) != getattr(
                candidates["reverse"], field
            ):
                raise ValueError(
                    "--unified-single-weight requires one shared source training "
                    f"provenance; {field} differs between aliases"
                )
        unified_training_binding = {
            field: getattr(candidates["planar"], field)
            for field in provenance_fields
        }
    generated_root = args.generated_root.resolve()
    routed.validate_exact_generated_assets(generated_root)
    asset_paths = routed.generated_asset_paths(generated_root)
    mujoco, onnxruntime, runtime, runtime_provenance = routed._load_runtime(
        include_provenance=True
    )
    base_bank = routed.RoutedPolicyBank(policies, onnxruntime)
    bank = H5RoutedPolicyBank(
        base_bank,
        candidates,
        allow_legacy_fallback=not strict_actor_only,
    )
    bank.integration_return_raw_action = True
    evaluator = runtime.OfficialPolicyEvaluator(
        asset_paths["scene"], policies["stand"], asset_paths["reference"]
    )
    evaluator.backward_residual_scale = 0.0
    if not strict_actor_only:
        # Exploratory teacher programs intentionally reuse the audited profile
        # generator.  This branch is never used by the strict H5 evaluator.
        evaluator.load_backward_profile(routed.FORMAL_FIXED_BACKWARD_PROFILE)
        evaluator.load_backward_turn_profile(1, routed.FORMAL_FIXED_BACKWARD_LEFT_PROFILE)
        evaluator.load_backward_turn_profile(-1, asset_paths["backward_right"])
        evaluator.backward_turn_minimum_yaw = 0.0
        evaluator.backward_turn_minimum_blend = 0.0
        evaluator.backward_turn_maximum_blend = 1.0
    model_evidence = routed.validate_model_contract(evaluator)
    simulator = H5RoutedSimulator(
        evaluator,
        bank,
        mujoco,
        runtime,
        leg_target_margin_rad=routed.RUNTIME_TARGET_SAFETY_MARGIN_RAD,
        target_slew_rate_rad_s=routed.RUNTIME_TARGET_SLEW_RATE_RAD_S,
        diagnostic_noncontract_safety=False,
        left_knee_extra_upper_margin_rad=0.0,
        diagnostic_reverse_entry_phase_indices=None,
        diagnostic_unadopted_backward_exit_recovery=False,
        formal_candidate_default=False,
        strict_actor_only=strict_actor_only,
    )
    simulator.reverse_entry_phase_indices = {
        "reverse": 7.0,
        "reverse_turn_left": 7.0,
        "reverse_turn_right": 7.0,
    }
    simulator.phase_entry_diagnostic_only = True
    simulator.phase_entry_status = "H5_SIMULATION_DIAGNOSTIC"
    # The H4 cloned schedule resolves this symbol through its own globals.
    h4._RUN_SCHEDULE_GLOBALS["resolve_policy_observation_command"] = (
        _h5_resolve_policy_observation_command
    )
    h4._RUN_SCHEDULE_GLOBALS["advance_routed_phase"] = _h5_advance_routed_phase
    h4._RUN_SCHEDULE_GLOBALS["SafeGaitRouter"] = H5SafeGaitRouter
    active_command_contract = (
        h5_unified_command_contract_id(_H5_UNIFIED_COMMAND_MAPPER)
        if _H5_UNIFIED_SINGLE_WEIGHT_MODE
        else None
    )
    training_command_mapper_compatible = bool(
        not _H5_UNIFIED_SINGLE_WEIGHT_MODE
        or (
            unified_training_binding is not None
            and unified_training_binding["training_command_contract_id"]
            == active_command_contract
            and unified_training_binding["training_command_mapper"]
            == _H5_UNIFIED_COMMAND_MAPPER
        )
    )
    metadata = {
        "h5_contract": H5_CONTRACT_ID,
        "h5_command_contract": (
            h5_unified_command_contract_id(_H5_UNIFIED_COMMAND_MAPPER)
            if _H5_UNIFIED_SINGLE_WEIGHT_MODE
            else H5_COMMAND_CONTRACT_ID
        ),
        "h5_unified_command_contract": (
            h5_unified_command_contract_manifest(_H5_UNIFIED_COMMAND_MAPPER)
            if _H5_UNIFIED_SINGLE_WEIGHT_MODE
            else None
        ),
        "router_contract": {
            "class": "H5SafeGaitRouter",
            "linear_deadband": 0.025,
            "linear_hysteresis": H5SafeGaitRouter.LINEAR_HYSTERESIS,
            "entry_threshold_mps": 0.030,
            "exit_threshold_mps": 0.020,
        },
        "hardware_deployment": "PROHIBITED",
        "single_policy_mode": {
            "enabled": _H5_UNIFIED_SINGLE_WEIGHT_MODE,
            "params_sha256": (
                candidates["planar"].params_sha256
                if _H5_UNIFIED_SINGLE_WEIGHT_MODE
                else None
            ),
            "command_mapper": (
                _H5_UNIFIED_COMMAND_MAPPER
                if _H5_UNIFIED_SINGLE_WEIGHT_MODE
                else "domain_routed_h5_policy_command"
            ),
            "training_command_mapper_compatible": bool(
                training_command_mapper_compatible
            ),
            "training_command_provenance": unified_training_binding,
        },
        "candidate_domains": {
            domain: {
                "params_path": str(getattr(args, f"h5_{domain}_params").resolve()),
                "params_sha256": getattr(args, f"h5_{domain}_params_sha256"),
                "manifest_path": str(getattr(args, f"h5_{domain}_manifest").resolve()),
                "manifest_sha256": getattr(args, f"h5_{domain}_manifest_sha256"),
                "training_manifest_sha256": candidates[domain].training_manifest_sha256,
                "training_resolved_config_sha256": (
                    candidates[domain].training_resolved_config_sha256
                ),
                "training_command_contract_id": (
                    candidates[domain].training_command_contract_id
                ),
                "training_command_mapper": candidates[domain].training_command_mapper,
                "training_command_mapper_inferred_from_v2_contract": (
                    candidates[domain].training_command_mapper_inferred_from_v2_contract
                ),
            }
            for domain in ("planar", "reverse")
        },
        "target_space_composition": {
            "raw_action_blend": False,
            "decode_each_candidate_before_blend": True,
            "single_final_guard_owned_by_schedule": True,
            "reverse_first_observation_phase_index": 7.0,
            "reverse_phase_delta_bins_per_control": H5_REVERSE_PHASE_DELTA_BINS_PER_CONTROL,
            "continuous_target_guard_and_contact_history": True,
        },
        "legacy_execution": {
            "h5_actor_authoritative": True,
            "strict_actor_only": strict_actor_only,
            "legacy_backward_profiles_loaded": not strict_actor_only,
            "legacy_backward_profiles_authoritative": False,
            "legacy_fallback_allowed": not strict_actor_only,
        },
        "model_contract": model_evidence,
        "runtime_provenance": runtime_provenance,
        "policy_bank": bank.manifest(),
    }
    return simulator, bank, metadata


def _run_suites(args: argparse.Namespace, simulator: Any) -> dict[str, Any]:
    primitive = routed._independent_suite(
        simulator,
        routed.PRIMITIVE_CASES,
        seed_base=args.seed,
        episodes=args.episodes,
        seconds=args.seconds,
        joint_noise_scale=args.initial_joint_noise_scale,
        initial_base_speed=args.initial_base_speed,
        warmup_seconds=args.warmup_seconds,
    )
    compound = routed._independent_suite(
        simulator,
        routed.COMPOUND_CASES,
        seed_base=args.seed + 1_000_000,
        episodes=args.episodes,
        seconds=args.seconds,
        joint_noise_scale=args.initial_joint_noise_scale,
        initial_base_speed=args.initial_base_speed,
        warmup_seconds=args.warmup_seconds,
    )
    definition = routed.transition_schedule(
        args.transition_seconds, args.transition_stand_seconds
    )
    transitions = [
        simulator.run_schedule(
            definition,
            seed=args.seed + 2_000_000 + index,
            joint_noise_scale=args.initial_joint_noise_scale,
            initial_base_speed=args.initial_base_speed,
            warmup_seconds=args.warmup_seconds,
        )
        for index in range(args.episodes)
    ]
    thresholds = AcceptanceThresholds()
    primitive_acceptance = suite_acceptance(
        primitive,
        [case.name for case in routed.PRIMITIVE_CASES],
        thresholds,
        require_gait_quality=True,
    )
    compound_acceptance = suite_acceptance(
        compound,
        [case.name for case in routed.COMPOUND_CASES],
        thresholds,
        require_gait_quality=True,
    )
    transition_acceptance = suite_acceptance(
        transitions,
        [row[0] for row in definition],
        thresholds,
        require_gait_quality=True,
    )
    return {
        "locomotion_action_count_excluding_stand": 12,
        "primitive_cases": primitive,
        "compound_cases": compound,
        "transition_cases": transitions,
        "acceptance": {
            "primitives": primitive_acceptance,
            "compounds": compound_acceptance,
            "transitions": transition_acceptance,
            "passed": bool(
                primitive_acceptance["passed"]
                and compound_acceptance["passed"]
                and transition_acceptance["passed"]
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {args.output}")
    if (
        args.h5_substep_capture_npz is not None
        and args.h5_substep_capture_npz.exists()
    ):
        raise FileExistsError(
            "refusing to overwrite H5 substep capture: "
            f"{args.h5_substep_capture_npz}"
        )
    if (
        args.h5_policy_observation_capture_npz is not None
        and args.h5_policy_observation_capture_npz.exists()
    ):
        raise FileExistsError(
            "refusing to overwrite H5 policy observation capture: "
            f"{args.h5_policy_observation_capture_npz}"
        )
    if args.episodes <= 0 or args.seconds <= 0.0 or args.transition_seconds <= 0.0:
        raise ValueError("episodes and durations must be positive")
    # The command-line evaluator is the strict evidence producer. Diagnostic
    # scripts call _build_simulator directly with strict_actor_only=False so
    # they can still construct auditable teacher data.
    args.strict_actor_only = True
    simulator, bank, metadata = _build_simulator(args)
    if args.h5_substep_capture_npz is not None:
        simulator.enable_h5_substep_measurement_capture()
    if args.h5_policy_observation_capture_npz is not None:
        simulator.enable_h5_policy_observation_capture()
    suites = _run_suites(args, simulator)
    bank_manifest = bank.manifest()
    if bank_manifest["legacy_fallback"]["count"] != 0:
        raise RuntimeError(
            "strict H5 evaluation observed a legacy fallback; refusing evidence"
        )
    payload = {
        "schema_version": 1,
        "evaluator_id": "openduckmini-exp004-h5-routed-transition-v1",
        "evaluation_mode": "H5_INTEGRATED_STRICT_DIAGNOSTIC",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_deployment": "PROHIBITED",
        "configuration": {
            "seed": args.seed,
            "episodes": args.episodes,
            "seconds": args.seconds,
            "transition_seconds": args.transition_seconds,
            "transition_stand_seconds": args.transition_stand_seconds,
            "warmup_seconds": args.warmup_seconds,
            "initial_joint_noise_scale": args.initial_joint_noise_scale,
            "initial_base_speed": args.initial_base_speed,
            "unified_single_weight": args.unified_single_weight,
            "target_margin_rad": routed.RUNTIME_TARGET_SAFETY_MARGIN_RAD,
            "target_slew_rate_rad_s": routed.RUNTIME_TARGET_SLEW_RATE_RAD_S,
        },
        "strict_thresholds": asdict(AcceptanceThresholds()),
        "provenance": metadata,
        "policy_bank": bank_manifest,
        "suites": suites,
        "acceptance": {
            "all_strict_quality_safety_transition_gates_passed": suites["acceptance"]["passed"],
            "adoption_allowed": False,
            "release_allowed": False,
            "hardware_deployment": "PROHIBITED",
        },
    }
    if args.h5_substep_capture_npz is not None:
        capture = simulator.export_h5_substep_measurement_capture()
        args.h5_substep_capture_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.h5_substep_capture_npz, **capture)
        capture_sha256 = sha256_file(args.h5_substep_capture_npz)
        payload["h5_substep_measurement_capture"] = {
            "path": str(args.h5_substep_capture_npz),
            "sha256": capture_sha256,
            "sample_count": int(capture["time_s"].shape[0]),
            "fields": sorted(capture),
            "measurement_source": (
                "frozen_gait_quality_post_physics_force_weighted_contact_kinematics"
            ),
            "hardware_deployment": "PROHIBITED",
        }
    if args.h5_policy_observation_capture_npz is not None:
        observation_capture = simulator.export_h5_policy_observation_capture()
        args.h5_policy_observation_capture_npz.parent.mkdir(
            parents=True, exist_ok=True
        )
        np.savez_compressed(
            args.h5_policy_observation_capture_npz, **observation_capture
        )
        payload["h5_policy_observation_capture"] = {
            "path": str(args.h5_policy_observation_capture_npz),
            "sha256": sha256_file(args.h5_policy_observation_capture_npz),
            "sample_count": int(observation_capture["observation"].shape[0]),
            "shape": list(observation_capture["observation"].shape),
            "measurement_source": "pre_inference_frozen_116wide_actor_observation",
            "hardware_deployment": "PROHIBITED",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "strict_passed": suites["acceptance"]["passed"], "hardware_deployment": "PROHIBITED"}, indent=2))
    return 0 if suites["acceptance"]["passed"] or not args.require_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
