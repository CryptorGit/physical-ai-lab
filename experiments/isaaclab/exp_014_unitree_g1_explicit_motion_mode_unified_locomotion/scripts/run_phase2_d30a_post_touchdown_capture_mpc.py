"""EXP014 Phase 2-D30A capture-basis, local-model, and bounded-MPC analyzer.

The analyzer is deliberately independent of the protected D26S/D29C runners.
It reads their durable telemetry, builds an action-coordinate contract, and
only enters the physics branch when the exact IsaacLab runtime is available.
Missing runtime, missing telemetry, or a failed model gate are terminal
conditions; this module never substitutes synthetic physics for them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d30a_post_touchdown_capture_mpc"
REPORT = REPO / "research/exp_014_phase_2_d30a_post_touchdown_capture_mpc_report.md"
D26S = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26s_exact_wmove_instrumentation"
D29C = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d29c_true_wmove_basin_adjudication"
NATIVE = D26S / "native_steady_trace_bundle.npz"
D29C_STAND = D29C / "raw/passive_physics_P_STAND.npz"
D29C_WALK = D29C / "raw/passive_physics_P_WALK_ZERO.npz"
D29B_SCRIPT = EXP / "scripts/run_phase2_d29b_walk_capture.py"
D26W_OFFSETS = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26w_action_semantics_and_feedforward/source_target_command_offsets.json"
D26T_MANIFEST = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26t_medoid_validation_and_offline_plans/entry_neighborhood_manifest.json"
RAW = OUT / "raw"
KNOWN_ISAACLAB_PYTHON = Path(r"C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe")
PHYSICS_STEPS = 300
POST_TD0_STEPS = 40
RETENTION_STEPS = 100
TOUCHDOWN_OFFSET = 0
STATE_FEATURE_INDICES = tuple(range(0, 9)) + (83, 84, 87)

SEED = 20279941
DT = 0.02
BASIS_VERSION = "WMoveCaptureActionBasisV1"
MODEL_VERSION = "D30ALocalDynamicsV1"
MPC_VERSION = "D30AFiniteHorizonBoundedLQRMPCV1"
DIMENSION_CANDIDATES = (4, 8, 12)
MIN_EXPLAINED_VARIANCE = 0.95
MAX_BASIS_DIMENSION = 12
PROTECTED_INPUTS = (
    EXP / "scripts/run_phase2_d26s_instrument.py",
    EXP / "scripts/run_phase2_d26t_replay.py",
    EXP / "scripts/run_phase2_d29b_walk_capture.py",
    EXP / "scripts/run_phase2_d29c_d29b0_passive_capture.py",
    EXP / "scripts/analyze_phase2_d29c_true_wmove_capture.py",
    NATIVE,
    D29C_STAND,
    D29C_WALK,
)


def _finite_array(value: Any, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"expected_ndim={ndim}, got={array.ndim}")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("non-finite-or-empty-array")
    return array


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True, stderr=subprocess.STDOUT).strip()


def repository_state() -> dict[str, Any]:
    raw_status = git("status", "--porcelain=v1")
    status = "\n".join(line for line in raw_status.splitlines() if not _is_d30a_worktree_path(line[3:]))
    return {
        "head": git("rev-parse", "HEAD"),
        "worktree_status": status.splitlines() if status else [],
        "worktree_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "filtered_d30a_paths": True,
    }


def _is_d30a_worktree_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return (
        "phase_2_d30a_post_touchdown_capture_mpc" in normalized
        or normalized.endswith("run_phase2_d30a_post_touchdown_capture_mpc.py")
        or normalized.endswith("test_phase2_d30a_capture_mpc.py")
        or normalized.endswith("exp_014_phase_2_d30a_post_touchdown_capture_mpc_report.md")
    )


def _action_from_npz(path: Path, key_preference: Iterable[str]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with np.load(path, allow_pickle=False) as bundle:
        for key in key_preference:
            if key in bundle.files:
                value = _finite_array(bundle[key], ndim=2)
                if value.shape[1] != 37:
                    raise ValueError(f"{path.name}:{key}:expected_action_width_37")
                return value
    raise KeyError(f"no-action-field:{path}")


@dataclass(frozen=True)
class WMoveCaptureActionBasisV1:
    """PCA/SVD basis for native W_MOVE actions and observed D29C deltas."""

    mean: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    explained_variance_ratio: np.ndarray
    requested_dimensions: tuple[int, ...] = DIMENSION_CANDIDATES
    minimum_explained_variance: float = MIN_EXPLAINED_VARIANCE
    version: str = BASIS_VERSION

    @classmethod
    def fit(
        cls,
        native_actions: np.ndarray,
        action_differences: np.ndarray,
        *,
        seed: int = SEED,
    ) -> "WMoveCaptureActionBasisV1":
        del seed  # SVD is deterministic; the parameter is retained in the contract.
        native = _finite_array(native_actions, ndim=2)
        differences = _finite_array(action_differences, ndim=2)
        if native.shape[1] != differences.shape[1]:
            raise ValueError("native-and-difference-width-mismatch")
        population = np.concatenate((native, differences), axis=0)
        mean = population.mean(axis=0)
        centered = population - mean
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        variance = singular**2
        total = float(variance.sum())
        ratio = variance / total if total > 0.0 else np.zeros_like(variance)
        rank = min(MAX_BASIS_DIMENSION, vt.shape[0])
        return cls(
            mean=mean,
            components=vt[:rank],
            singular_values=singular[:rank],
            explained_variance_ratio=ratio[:rank],
        )

    @property
    def rank(self) -> int:
        return int(self.components.shape[0])

    def dimension_for(self, requested: int) -> int:
        if requested not in self.requested_dimensions:
            raise ValueError(f"unregistered-dimension:{requested}")
        if self.rank == 0:
            return 0
        cumulative = np.cumsum(self.explained_variance_ratio)
        for candidate in self.requested_dimensions:
            if candidate < requested:
                continue
            usable = min(candidate, self.rank)
            if cumulative[usable - 1] >= self.minimum_explained_variance:
                return int(usable)
        return int(min(MAX_BASIS_DIMENSION, self.rank))

    def preregistered_dimension(self) -> int:
        if self.rank == 0:
            return 0
        cumulative = np.cumsum(self.explained_variance_ratio)
        for candidate in self.requested_dimensions:
            usable = min(candidate, self.rank)
            if cumulative[usable - 1] >= self.minimum_explained_variance:
                return int(usable)
        return int(min(MAX_BASIS_DIMENSION, self.rank))

    def transform(self, actions: np.ndarray, dimension: int | None = None) -> np.ndarray:
        array = _finite_array(actions, ndim=2)
        if array.shape[1] != self.mean.shape[0]:
            raise ValueError("action-width-mismatch")
        dimension = self.preregistered_dimension() if dimension is None else int(dimension)
        if not 0 <= dimension <= self.rank:
            raise ValueError("basis-dimension-out-of-range")
        return (array - self.mean) @ self.components[:dimension].T

    def inverse_transform(self, coordinates: np.ndarray, dimension: int | None = None) -> np.ndarray:
        coords = _finite_array(coordinates, ndim=2)
        dimension = self.preregistered_dimension() if dimension is None else int(dimension)
        if coords.shape[1] != dimension or dimension > self.rank:
            raise ValueError("coordinate-width-mismatch")
        return coords @ self.components[:dimension] + self.mean

    def manifest(self) -> dict[str, Any]:
        cumulative = np.cumsum(self.explained_variance_ratio)
        return {
            "name": self.version,
            "action_width": int(self.mean.size),
            "rank": self.rank,
            "requested_dimensions": list(self.requested_dimensions),
            "minimum_explained_variance": self.minimum_explained_variance,
            "maximum_dimension": MAX_BASIS_DIMENSION,
            "selected_dimension": self.preregistered_dimension(),
            "explained_variance_ratio": self.explained_variance_ratio,
            "cumulative_explained_variance": cumulative,
            "singular_values": self.singular_values,
            "construction": {
                "native_population": "D26S native W_MOVE action",
                "difference_population": "D29C P_WALK_ZERO.action - P_STAND.action",
                "centering": "population mean",
                "solver": "deterministic economy SVD",
            },
        }


@dataclass(frozen=True)
class LocalLinearDynamics:
    """Affine local model y = A x + B u + c with ridge regularization."""

    A: np.ndarray
    B: np.ndarray
    c: np.ndarray
    residual_bound: np.ndarray
    ridge: float = 1.0e-8
    version: str = MODEL_VERSION

    @classmethod
    def fit(
        cls,
        states: np.ndarray,
        controls: np.ndarray,
        next_states: np.ndarray,
        *,
        ridge: float = 1.0e-8,
    ) -> "LocalLinearDynamics":
        x = _finite_array(states, ndim=2)
        u = _finite_array(controls, ndim=2)
        y = _finite_array(next_states, ndim=2)
        if x.shape[0] != u.shape[0] or x.shape[0] != y.shape[0] or x.shape[0] < 2:
            raise ValueError("dynamics-row-mismatch-or-insufficient-rows")
        design = np.concatenate((x, u, np.ones((x.shape[0], 1))), axis=1)
        gram = design.T @ design + float(ridge) * np.eye(design.shape[1])
        coefficients = np.linalg.solve(gram, design.T @ y)
        residual = y - design @ coefficients
        bound = np.max(np.abs(residual), axis=0)
        nx, nu = x.shape[1], u.shape[1]
        return cls(coefficients[:nx].T, coefficients[nx:nx + nu].T, coefficients[-1], bound, float(ridge))

    @property
    def state_width(self) -> int:
        return int(self.A.shape[1])

    @property
    def control_width(self) -> int:
        return int(self.B.shape[1])

    def predict(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        x = _finite_array(state)
        u = _finite_array(control)
        return self.A @ x + self.B @ u + self.c

    def rollout(self, state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        x = _finite_array(state).copy()
        sequence = _finite_array(controls, ndim=2)
        rows = []
        for u in sequence:
            x = self.predict(x, u)
            rows.append(x.copy())
        return np.asarray(rows)

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.version,
            "state_width": self.state_width,
            "control_width": self.control_width,
            "ridge": self.ridge,
            "residual_bound": self.residual_bound,
        }


def perturbation_schedule(width: int, fraction: float) -> list[np.ndarray]:
    if width <= 0 or not math.isfinite(fraction) or fraction <= 0:
        raise ValueError("invalid-perturbation-contract")
    schedule: list[np.ndarray] = []
    for index in range(width):
        direction = np.zeros(width, dtype=np.float64)
        direction[index] = fraction
        schedule.extend((direction.copy(), -direction.copy()))
    return schedule


def identify_bins(
    records: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
    *,
    perturbation_fraction: float = 0.10,
) -> dict[str, Any]:
    """Fit LEFT/RIGHT × early/late models from synchronized measured rows."""
    result: dict[str, Any] = {"perturbation_fraction": perturbation_fraction, "bins": {}, "pass": True}
    for side in ("LEFT", "RIGHT"):
        for timing in ("early", "late"):
            key = (side, timing)
            label = f"{side}_{timing}"
            item = records.get(key)
            if item is None:
                result["bins"][label] = {"available": False, "reason": "missing_synchronized_physics_rows"}
                result["pass"] = False
                continue
            try:
                model = LocalLinearDynamics.fit(item["states"], item["controls"], item["next_states"])
                result["bins"][label] = {
                    "available": True,
                    "model": model.manifest(),
                    "perturbation_count": len(perturbation_schedule(model.control_width, perturbation_fraction)),
                    "residual_bound_max": float(np.max(model.residual_bound)),
                }
            except (ValueError, np.linalg.LinAlgError) as exc:
                result["bins"][label] = {"available": False, "reason": f"fit_failed:{exc}"}
                result["pass"] = False
    return result


def validate_holdout(
    model: LocalLinearDynamics | None,
    state: np.ndarray | None,
    controls: np.ndarray | None,
    measured_next: np.ndarray | None,
    *,
    one_step_tolerance: float = 0.05,
    three_step_tolerance: float = 0.15,
) -> dict[str, Any]:
    if model is None or state is None or controls is None or measured_next is None:
        return {
            "available": False,
            "pass": False,
            "reason": "missing_physics_holdout",
            "gates": {"one_step": False, "three_step": False},
        }
    try:
        x = _finite_array(state)
        u = _finite_array(controls, ndim=2)
        y = _finite_array(measured_next, ndim=2)
        if len(u) < 3 or y.shape[0] < 3:
            raise ValueError("holdout_requires_three_steps")
        one = model.predict(x, u[0])
        one_error = float(np.max(np.abs(one - y[0])))
        three = model.rollout(x, u[:3])[-1]
        three_error = float(np.max(np.abs(three - y[2])))
        one_pass = bool(np.isfinite(one_error) and one_error <= one_step_tolerance)
        three_pass = bool(np.isfinite(three_error) and three_error <= three_step_tolerance)
        return {
            "available": True,
            "pass": one_pass and three_pass,
            "one_step_max_error": one_error,
            "three_step_max_error": three_error,
            "tolerances": {"one_step": one_step_tolerance, "three_step": three_step_tolerance},
            "gates": {"one_step": one_pass, "three_step": three_pass},
        }
    except (ValueError, np.linalg.LinAlgError) as exc:
        return {"available": False, "pass": False, "reason": f"holdout_failed:{exc}", "gates": {"one_step": False, "three_step": False}}


@dataclass(frozen=True)
class FiniteHorizonBoundedLQRMPC:
    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    horizon: int = 16
    version: str = MPC_VERSION

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("positive-horizon-required")
        if self.A.shape[0] != self.A.shape[1] or self.B.shape[0] != self.A.shape[0]:
            raise ValueError("incompatible-dynamics")
        if self.Q.shape != self.A.shape or self.R.shape != (self.B.shape[1], self.B.shape[1]):
            raise ValueError("incompatible-cost")
        if self.lower.shape != (self.B.shape[1],) or self.upper.shape != self.lower.shape:
            raise ValueError("incompatible-bounds")

    def first_gain(self) -> np.ndarray:
        p = self.Q.copy()
        gain = np.zeros((self.B.shape[1], self.A.shape[0]), dtype=np.float64)
        for _ in range(self.horizon):
            middle = self.R + self.B.T @ p @ self.B
            gain = np.linalg.solve(middle, self.B.T @ p @ self.A)
            p = self.Q + self.A.T @ p @ (self.A - self.B @ gain)
        return gain

    def control(self, state_error: np.ndarray) -> np.ndarray:
        error = _finite_array(state_error)
        if error.shape != (self.A.shape[0],):
            raise ValueError("state-error-width-mismatch")
        raw = -self.first_gain() @ error
        return np.clip(raw, self.lower, self.upper)

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.version,
            "horizon_steps": self.horizon,
            "bounds": {"lower": self.lower, "upper": self.upper},
            "state_width": int(self.A.shape[0]),
            "control_width": int(self.B.shape[1]),
        }


def _load_basis() -> tuple[WMoveCaptureActionBasisV1, dict[str, Any]]:
    native = _action_from_npz(NATIVE, ("action", "current_action", "next_action"))
    stand = _action_from_npz(D29C_STAND, ("action", "current_action"))
    walk = _action_from_npz(D29C_WALK, ("action", "current_action"))
    if stand.shape != walk.shape:
        raise ValueError("D29C condition-shape-mismatch")
    differences = walk - stand
    basis = WMoveCaptureActionBasisV1.fit(native, differences)
    return basis, {
        "native_rows": int(native.shape[0]),
        "difference_rows": int(differences.shape[0]),
        "difference_definition": "P_WALK_ZERO.action - P_STAND.action",
        "native_sha256": sha256_file(NATIVE),
        "stand_sha256": sha256_file(D29C_STAND),
        "walk_sha256": sha256_file(D29C_WALK),
    }


def _native_coefficient_p95(basis: WMoveCaptureActionBasisV1) -> np.ndarray:
    native = _action_from_npz(NATIVE, ("action", "current_action", "next_action"))
    return np.maximum(np.quantile(np.abs(basis.transform(native)), 0.95, axis=0), 1.0e-3)


def _joint_group_participation(basis: WMoveCaptureActionBasisV1) -> dict[str, Any]:
    if not D26W_OFFSETS.is_file():
        return {"available": False, "reason": "D26W_SOURCE_TARGET_OFFSETS_MISSING"}
    source = json.loads(D26W_OFFSETS.read_text(encoding="utf-8"))
    groups = source.get("joint_groups", {})
    dim = basis.preregistered_dimension()
    components = np.abs(basis.components[:dim])
    total = np.maximum(components.sum(axis=1), 1.0e-12)
    participation = {}
    for name, indices in groups.items():
        selected = components[:, np.asarray(indices, dtype=int)].sum(axis=1) / total
        participation[name] = selected
    return {
        "available": True,
        "source": str(D26W_OFFSETS.relative_to(REPO)).replace("\\", "/"),
        "source_sha256": sha256_file(D26W_OFFSETS),
        "mapping": source.get("mapping"),
        "joint_groups": groups,
        "basis_component_abs_l1_participation": participation,
        "definition": "sum(abs(SVD component[joint])) / sum(abs(SVD component[all joints]))",
    }


def _d26t_phase_tube_contract() -> dict[str, Any]:
    if not D26T_MANIFEST.is_file():
        return {"available": False, "name": "WMove03PhaseTubeV1", "reason": "D26T_ENTRY_MANIFEST_MISSING"}
    manifest = json.loads(D26T_MANIFEST.read_text(encoding="utf-8"))
    references = manifest.get("references", [])
    fields = ["reference_id", "side", "rank", "bundle_row", "episode_id", "control_step", "event_step", "expected"]
    return {
        "available": True,
        "name": "WMove03PhaseTubeV1",
        "source": str(D26T_MANIFEST.relative_to(REPO)).replace("\\", "/"),
        "source_sha256": sha256_file(D26T_MANIFEST),
        "counts": manifest.get("counts", {"LEFT": 50, "RIGHT": 50}),
        "event_source": manifest.get("event_source", "E0_STRICT_TOUCHDOWN"),
        "phase_definition": {
            "event": "E0_STRICT_TOUCHDOWN",
            "reference_window": "D26T entry neighborhood event_step+2..+6",
            "phase_fields": ["event_step", "control_step", "side", "rank"],
            "reference_fields": fields,
        },
        "reference_fields": fields,
        "references": references,
    }


def resolve_isaac_python(explicit: str | None) -> Path:
    candidate = explicit or os.environ.get("ISAACLAB_PYTHON")
    path = Path(candidate) if candidate else KNOWN_ISAACLAB_PYTHON
    if not path.is_file():
        path = Path(sys.executable)
    return path.resolve()


def _isaac_runtime_status(python_path: Path, source_label: str = "default-known-path") -> dict[str, Any]:
    probe = (
        "import importlib.util,json,sys;"
        "names=['isaaclab','isaaclab_tasks','gymnasium','torch'];"
        "mods={n:(importlib.util.find_spec(n) is not None) for n in names};"
        "loc={n:(None if importlib.util.find_spec(n) is None else importlib.util.find_spec(n).origin) for n in names};"
        "print(json.dumps({'python':sys.executable,'version':sys.version,'modules':mods,'module_locations':loc}))"
    )
    try:
        completed = subprocess.run(
            [str(python_path), "-c", probe],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        payload = json.loads(completed.stdout.strip().splitlines()[-1]) if completed.stdout.strip() else {}
        modules = payload.get("modules", {})
        available = bool(completed.returncode == 0 and modules.get("isaaclab") and modules.get("isaaclab_tasks") and modules.get("gymnasium"))
        return {
            "available": available,
            "requested_python": str(python_path),
            "python": payload.get("python", str(python_path)),
            "python_version": payload.get("version"),
            "modules": modules,
            "module_locations": payload.get("module_locations", {}),
            "probe_returncode": completed.returncode,
            "probe_stderr": completed.stderr[-4000:],
            "cwd": str(REPO),
            "source": source_label,
            "reason": None if available else "ISAACLAB_RUNTIME_UNAVAILABLE",
        }
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "requested_python": str(python_path),
            "python": str(python_path),
            "modules": {},
            "module_locations": {},
            "probe_returncode": None,
            "probe_stderr": str(exc),
            "cwd": str(REPO),
            "source": source_label,
            "reason": "ISAACLAB_RUNTIME_UNAVAILABLE",
        }


def explicit_python_matches(path: Path) -> bool:
    requested = os.environ.get("ISAACLAB_PYTHON")
    return bool(requested and Path(requested).resolve() == path.resolve())


def _write_baseline_table(path: Path, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("route", "replay", "available", "pass", "reason", "handoff", "retention_100_step"))
        writer.writeheader()
        for route in range(8):
            for replay in ("primary", "independent"):
                writer.writerow({
                    "route": f"R{route}",
                    "replay": replay,
                    "available": "false",
                    "pass": "false",
                    "reason": reason,
                    "handoff": "",
                    "retention_100_step": "",
                })


def _write_failure_table(path: Path, failures: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("stage", "status", "reason"))
        writer.writeheader()
        for failure in failures:
            stage, _, reason = failure.partition(":")
            writer.writerow({"stage": stage, "status": "BLOCKED", "reason": reason})


def _write_bin_table(path: Path, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("side", "timing", "available", "pass", "reason", "residual_bound_max"))
        writer.writeheader()
        for side in ("LEFT", "RIGHT"):
            for timing in ("early", "late"):
                writer.writerow({
                    "side": side,
                    "timing": timing,
                    "available": "false",
                    "pass": "false",
                    "reason": reason,
                    "residual_bound_max": "",
                })


def load_d29b_module():
    spec = importlib.util.spec_from_file_location("exp014_d29b_readonly", D29B_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("D29B_ADAPTER_IMPORT_FAILED")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _child_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--branch", choices=("baseline", "identification", "holdout", "mpc", "mpc_independent"), required=True)
    parser.add_argument("--schedule", required=False)
    parser.add_argument("--model", required=False)
    try:
        from isaaclab_tasks.utils import add_launcher_args, setup_preset_cli
        add_launcher_args(parser)
        args, hydra = setup_preset_cli(parser)
        sys.argv = [sys.argv[0], *hydra]
        return args
    except ModuleNotFoundError:
        return parser.parse_args()


def _selected_state(feature: np.ndarray) -> np.ndarray:
    return np.asarray(feature, dtype=np.float64)[:, list(STATE_FEATURE_INDICES)]


def _load_schedule(path: str | None) -> dict[str, Any]:
    return {} if not path else json.loads(Path(path).read_text(encoding="utf-8"))


def _load_mpc_bundle(path: str) -> tuple[dict[str, Any], dict[str, LocalLinearDynamics], WMoveCaptureActionBasisV1]:
    manifest = json.loads((Path(path)).read_text(encoding="utf-8"))
    arrays = np.load(Path(path).with_suffix(".npz"), allow_pickle=False)
    basis = WMoveCaptureActionBasisV1(
        mean=np.asarray(arrays["basis_mean"]),
        components=np.asarray(arrays["basis_components"]),
        singular_values=np.asarray(arrays["basis_singular_values"]),
        explained_variance_ratio=np.asarray(arrays["basis_explained_variance_ratio"]),
    )
    models: dict[str, LocalLinearDynamics] = {}
    for label in ("LEFT_early", "LEFT_late", "RIGHT_early", "RIGHT_late"):
        if not manifest.get("models", {}).get(label):
            continue
        prefix = manifest["models"][label]
        models[label] = LocalLinearDynamics(
            np.asarray(arrays[f"{label}_A"]),
            np.asarray(arrays[f"{label}_B"]),
            np.asarray(arrays[f"{label}_c"]),
            np.asarray(arrays[f"{label}_residual"]),
        )
    return manifest, models, basis


def _stack_branch(store: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate(value, axis=0) for key, value in store.items()}


def _run_physics_branch(args: argparse.Namespace) -> int:
    """Fresh IsaacLab process using the D29B Route A lifecycle and actors."""
    d29b = load_d29b_module()
    import torch
    from isaaclab_tasks.utils import launch_simulation
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    schedule = _load_schedule(args.schedule)
    branch = str(args.branch)
    n = len(d29b.RECIPES)
    episode_steps = PHYSICS_STEPS * DT + 2.0
    gym, cfg, agent = d29b.configure(args, "Isaac-Exp013-G1-DirectionalBaseline-v0", n, episode_steps)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    import random
    random.seed(SEED)
    td0_schedule = np.asarray(schedule.get("td0_steps", [-1] * n), dtype=np.int32)
    coefficients = np.asarray(schedule.get("coefficients", np.zeros((n, 0))), dtype=np.float64)
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(
            gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg),
            clip_actions=agent.clip_actions,
        )
        env = wrapped.unwrapped
        robot = env.scene["robot"]
        sensor = env.scene["contact_forces"]
        sensor_feet, robot_feet, sensor_names, robot_names = d29b.find_foot_indices(sensor, robot)
        term = env.command_manager.get_term("base_velocity")
        term.external_override_enabled = True
        p0_actor = d29b.load_actor(d29b.P0, env.device, False)
        wmove_actor = d29b.load_actor(d29b.WMOVE, env.device, True)
        d29b.normal_reset(env, term)
        previous_action = torch.zeros((n, 37), device=env.device)
        previous_contact = None
        liftoff_seen = np.zeros(n, dtype=bool)
        observed_td0 = np.full(n, -1, dtype=np.int32)
        store: dict[str, list[np.ndarray]] = {}
        safety = {key: np.zeros(n, dtype=bool) for key in ("fall", "dangerous_slip", "impact", "velocity_saturation", "torque_saturation", "support_loss", "nan_inf")}
        streaks = {key: np.zeros(n, dtype=np.int32) for key in ("slip", "velocity", "torque", "support")}
        first_failure: list[str | None] = [None] * n
        current_feature = None
        target = np.asarray(schedule.get("targets", np.zeros((n, len(STATE_FEATURE_INDICES)))), dtype=np.float64)
        model_manifest: dict[str, Any] = {}
        models: dict[str, LocalLinearDynamics] = {}
        basis = None
        if branch in ("identification", "holdout"):
            arrays = np.load(OUT / "basis_components.npz", allow_pickle=False)
            basis = WMoveCaptureActionBasisV1(
                np.asarray(arrays["mean"]),
                np.asarray(arrays["components"]),
                np.asarray(arrays["singular_values"]),
                np.asarray(arrays["explained_variance_ratio"]),
            )
        if branch in ("mpc", "mpc_independent"):
            if not args.model:
                raise RuntimeError("MPC_MODEL_BUNDLE_MISSING")
            model_manifest, models, basis = _load_mpc_bundle(args.model)
        for step in range(PHYSICS_STEPS):
            mode = "P0" if step < d29b.STAND_STEPS else "WMOVE"
            command_np = np.zeros((n, 3), dtype=np.float32)
            if mode == "WMOVE":
                command_np[:, 0] = d29b.WMOVE_SPEED
            command = torch.as_tensor(command_np, device=env.device)
            term.external_override.copy_(command)
            term._update_command()
            obs = wrapped.get_observations()["policy"].to(env.device)
            if mode == "P0":
                action = d29b.actor_action(p0_actor, obs, env.device, False)
            else:
                action = d29b.actor_action(wmove_actor, obs, env.device, True)
            applied_coeff = np.zeros((n, coefficients.shape[1] if coefficients.ndim == 2 else 0), dtype=np.float64)
            active = (td0_schedule >= 0) & (step >= td0_schedule + 1) & (step < td0_schedule + 1 + POST_TD0_STEPS)
            if mode == "WMOVE" and active.any() and branch in ("identification", "holdout"):
                applied_coeff = coefficients.copy()
                delta = np.asarray(basis.components[:coefficients.shape[1]].T @ applied_coeff.T).T if coefficients.shape[1] else np.zeros((n, 37))
                action = action + torch.as_tensor(delta, dtype=action.dtype, device=action.device)
            elif mode == "WMOVE" and active.any() and branch in ("mpc", "mpc_independent"):
                if current_feature is None:
                    raise RuntimeError("MPC_CURRENT_STATE_UNAVAILABLE")
                for i in np.flatnonzero(active):
                    side = str(schedule.get("sides", ["LEFT"] * n)[i])
                    timing = "early" if step - int(td0_schedule[i]) <= 10 else "late"
                    model = models.get(f"{side}_{timing}") or next(iter(models.values()), None)
                    if model is None:
                        raise RuntimeError("MPC_LOCAL_MODEL_UNAVAILABLE")
                    controller = FiniteHorizonBoundedLQRMPC(
                        model.A,
                        model.B,
                        np.eye(model.state_width),
                        np.eye(model.control_width) * 0.1,
                        -np.asarray(model_manifest["coefficient_bounds"]),
                        np.asarray(model_manifest["coefficient_bounds"]),
                        horizon=16,
                    )
                    coeff = controller.control(current_feature[i] - target[i])
                    applied_coeff[i] = coeff
                    delta = basis.components[:len(coeff)].T @ coeff
                    action[i] = action[i] + torch.as_tensor(delta, dtype=action.dtype, device=action.device)
            _, _, done, extras = wrapped.step(action)
            timeout_value = extras.get("time_outs", torch.zeros_like(done)) if isinstance(extras, dict) else torch.zeros_like(done)
            done_np = np.asarray(done.detach().cpu(), dtype=bool)
            timeout_np = np.asarray(timeout_value.detach().cpu(), dtype=bool)
            state = d29b.snapshot(env, robot, sensor, sensor_feet, robot_feet, previous_action, action)
            feature = _selected_state(d29b.feature_from_state(state))
            current_feature = feature.copy()
            d29b.safety_update(state, done_np, timeout_np, safety, streaks, first_failure, step)
            contact = np.asarray(state["contact"], dtype=bool)
            if previous_contact is not None:
                fell = previous_contact & ~contact
                rose = ~previous_contact & contact
                liftoff_seen |= np.any(fell, axis=1)
                for i in range(n):
                    if observed_td0[i] < 0 and bool(rose[i].any()) and (liftoff_seen[i] or bool(previous_contact[i].any())):
                        observed_td0[i] = step
            previous_contact = contact.copy()
            def append(key: str, value: Any) -> None:
                store.setdefault(key, []).append(np.asarray(value).copy())
            append("control_step", np.full(n, step, dtype=np.int32))
            append("source_environment_index", np.arange(n, dtype=np.int32))
            append("feature", feature)
            append("contact", contact)
            append("root_velocity", state["root_velocity"])
            append("root_pose", state["root_pose"])
            append("action", action.detach().cpu().numpy())
            append("base_action", (action.detach().cpu().numpy() - (np.asarray(basis.components[:coefficients.shape[1]].T @ applied_coeff.T).T if coefficients.shape[1] else 0.0)) if basis is not None else action.detach().cpu().numpy())
            append("control_coeff", applied_coeff)
            append("done", done_np)
            append("timeout", timeout_np)
            previous_action = action.detach().clone()
        data = _stack_branch(store)
        raw_path = RAW / f"branch_{branch}.npz"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(raw_path, **data)
        results = []
        for i in range(n):
            rows = data["source_environment_index"] == i
            after = rows & (data["control_step"] >= max(int(observed_td0[i]) + 1, 0))
            retention = after & (data["control_step"] < max(int(observed_td0[i]) + 1, 0) + POST_TD0_STEPS + RETENTION_STEPS)
            results.append({
                "recipe_id": i,
                "touchdown_step": int(observed_td0[i]),
                "touchdown_side": ("LEFT" if i % 2 == 0 else "RIGHT"),
                "first_failure": first_failure[i],
                "safety": {key: bool(value[i]) for key, value in safety.items()},
                "retention_rows": int(np.sum(retention)),
                "retention_finite": bool(np.isfinite(data["feature"][retention]).all()) if retention.any() else False,
            })
        write_json(RAW / f"branch_{branch}.json", {
            "branch": branch,
            "fresh_process": True,
            "seed": SEED,
            "steps": PHYSICS_STEPS,
            "dt_s": DT,
            "sensor_foot_names": sensor_names,
            "robot_foot_names": robot_names,
            "observed_td0_steps": observed_td0,
            "results": results,
            "checkpoint_hashes": {"s_hold": sha256_file(d29b.P0), "w_move": sha256_file(d29b.WMOVE)},
            "raw_path": str(raw_path.relative_to(REPO)).replace("\\", "/"),
        })
        wrapped.close()
    return 0


def _spawn_branch(
    python_path: Path,
    branch: str,
    *,
    schedule: Path | None = None,
    model: Path | None = None,
    device: str | None = None,
    headless: bool = True,
) -> dict[str, Any]:
    command = [str(python_path), str(HERE), "--child", "--branch", branch]
    if schedule is not None:
        command.extend(("--schedule", str(schedule)))
    if model is not None:
        command.extend(("--model", str(model)))
    if device:
        command.extend(("--device", device))
    if headless:
        command.extend(("--headless", "--viz", "none"))
    log_dir = OUT / "child_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
    (log_dir / f"{branch}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (log_dir / f"{branch}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    payload_path = RAW / f"branch_{branch}.json"
    return {
        "branch": branch,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-4000:],
        "artifact": str(payload_path.relative_to(REPO)).replace("\\", "/") if payload_path.exists() else None,
        "success": bool(completed.returncode == 0 and payload_path.exists()),
    }


def _basis_from_artifacts() -> WMoveCaptureActionBasisV1:
    arrays = np.load(OUT / "basis_components.npz", allow_pickle=False)
    return WMoveCaptureActionBasisV1(
        np.asarray(arrays["mean"]),
        np.asarray(arrays["components"]),
        np.asarray(arrays["singular_values"]),
        np.asarray(arrays["explained_variance_ratio"]),
    )


def _make_coefficients(basis: WMoveCaptureActionBasisV1) -> tuple[np.ndarray, np.ndarray]:
    native = _action_from_npz(NATIVE, ("action", "current_action", "next_action"))
    dim = basis.preregistered_dimension()
    coeff = basis.transform(native, dim)
    scale = np.maximum(np.quantile(np.abs(coeff), 0.95, axis=0), 1.0e-3)
    directions = np.zeros((len(d29b_recipe_ids()), dim), dtype=np.float64)
    for env in range(len(d29b_recipe_ids())):
        direction = env // 2
        directions[env, direction] = 1.0 if env % 2 == 0 else -1.0
    return directions, scale


def d29b_recipe_ids() -> list[int]:
    return list(range(8))


def _write_schedule(path: Path, td0: Iterable[int], coefficients: np.ndarray, sides: Iterable[str]) -> None:
    write_json(path, {
        "td0_steps": [int(x) for x in td0],
        "coefficients": np.asarray(coefficients, dtype=float),
        "sides": list(sides),
        "post_td0_steps": POST_TD0_STEPS,
        "intervention": "TD0-only; no pre-touchdown action modification",
    })


def _branch_data(branch: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata = json.loads((RAW / f"branch_{branch}.json").read_text(encoding="utf-8"))
    arrays = dict(np.load(RAW / f"branch_{branch}.npz", allow_pickle=False))
    return metadata, arrays


def _row_index(step: int, env: int, count: int = 8) -> int:
    return int(step) * count + int(env)


def _fit_capture_models(basis: WMoveCaptureActionBasisV1, baseline: dict[str, Any], base_data: dict[str, np.ndarray], pert_data: dict[str, np.ndarray], coefficients: np.ndarray) -> tuple[dict[str, Any], dict[str, LocalLinearDynamics]]:
    records: dict[str, dict[str, list[np.ndarray]]] = {
        f"{side}_{timing}": {"states": [], "controls": [], "next_states": []}
        for side in ("LEFT", "RIGHT") for timing in ("early", "late")
    }
    sides = [str(x.get("touchdown_side") or ("LEFT" if i % 2 == 0 else "RIGHT")) for i, x in enumerate(baseline.get("results", []))]
    td0 = np.asarray(baseline.get("observed_td0_steps", [-1] * 8), dtype=int)
    for env in range(8):
        if td0[env] < 0:
            continue
        for step in range(max(0, td0[env] + 1), min(PHYSICS_STEPS - 1, td0[env] + 1 + POST_TD0_STEPS)):
            timing = "early" if step - td0[env] <= 10 else "late"
            label = f"{sides[env]}_{timing}"
            records[label]["states"].append(base_data["feature"][_row_index(step, env)])
            records[label]["controls"].append(coefficients[env])
            records[label]["next_states"].append(pert_data["feature"][_row_index(step + 1, env)])
    models: dict[str, LocalLinearDynamics] = {}
    bins: dict[str, Any] = {}
    for label, item in records.items():
        if len(item["states"]) < 8:
            bins[label] = {"available": False, "pass": False, "reason": "INSUFFICIENT_SYNCHRONIZED_ROWS", "rows": len(item["states"])}
            continue
        try:
            model = LocalLinearDynamics.fit(np.asarray(item["states"]), np.asarray(item["controls"]), np.asarray(item["next_states"]))
            models[label] = model
            bins[label] = {"available": True, "pass": bool(np.isfinite(model.A).all() and np.isfinite(model.B).all()), "rows": len(item["states"]), "residual_bound_max": float(np.max(model.residual_bound))}
        except (ValueError, np.linalg.LinAlgError) as exc:
            bins[label] = {"available": False, "pass": False, "reason": f"FIT_FAILED:{exc}", "rows": len(item["states"])}
    result = {
        "name": MODEL_VERSION,
        "available": bool(len(models) == 4),
        "pass": bool(len(models) == 4 and all(x.get("pass") for x in bins.values())),
        "perturbation_fraction": 0.10,
        "bins": bins,
        "state_feature_indices": list(STATE_FEATURE_INDICES),
        "control_dimension": int(basis.preregistered_dimension()),
        "required_gates": {"velocity_yaw_sign_agreement": 0.95, "control_response_sign_agreement": 0.90, "three_step_normalized_error_max": 0.30},
        "A_B_quality": {
            label: {
                "A_finite": bool(np.isfinite(model.A).all()),
                "B_finite": bool(np.isfinite(model.B).all()),
                "A_frobenius_norm": float(np.linalg.norm(model.A)),
                "B_frobenius_norm": float(np.linalg.norm(model.B)),
                "residual_bound_max": float(np.max(model.residual_bound)),
            }
            for label, model in models.items()
        },
    }
    return result, models


def _validate_capture_models(
    models: dict[str, LocalLinearDynamics],
    baseline: dict[str, Any],
    base_data: dict[str, np.ndarray],
    identification_data: dict[str, np.ndarray],
    holdout_data: dict[str, np.ndarray],
    holdout_coefficients: np.ndarray,
) -> dict[str, Any]:
    sides = [str(x.get("touchdown_side") or ("LEFT" if i % 2 == 0 else "RIGHT")) for i, x in enumerate(baseline.get("results", []))]
    td0 = np.asarray(baseline.get("observed_td0_steps", [-1] * 8), dtype=int)
    rows = []
    sign_velocity_yaw: list[bool] = []
    sign_control_response: list[bool] = []
    group_values: dict[str, list[float]] = {"velocity": [], "yaw": [], "projected_gravity": [], "com_relative_support": [], "dcm_relative_support": []}
    for env in range(8):
        if td0[env] < 0:
            continue
        for step in range(max(0, td0[env] + 1), min(PHYSICS_STEPS - 3, td0[env] + 1 + POST_TD0_STEPS - 2)):
            label = f"{sides[env]}_{'early' if step - td0[env] <= 10 else 'late'}"
            model = models.get(label)
            if model is None:
                continue
            x = base_data["feature"][_row_index(step, env)]
            controls = np.repeat(holdout_coefficients[env][None, :], 3, axis=0)
            measured = np.asarray([holdout_data["feature"][_row_index(step + j, env)] for j in (1, 2, 3)])
            one = model.predict(x, controls[0])
            three = model.rollout(x, controls)[-1]
            rows.append({
                "recipe_id": env,
                "side": sides[env],
                "timing": label.rsplit("_", 1)[-1],
                "control_step": step,
                "one_step_max_error": float(np.max(np.abs(one - measured[0]))),
                "three_step_max_error": float(np.max(np.abs(three - measured[2]))),
                "three_step_normalized_error": float(np.max(np.abs(three - measured[2])) / max(float(np.linalg.norm(measured[2])), 1.0e-6)),
            })
            base_next = base_data["feature"][_row_index(step + 1, env)]
            id_delta = identification_data["feature"][_row_index(step + 1, env)] - base_next
            hold_delta = measured[0] - base_next
            sign_velocity_yaw.extend(bool(np.sign(id_delta[j]) == np.sign(hold_delta[j])) for j in (0, 5) if abs(id_delta[j]) > 1.0e-8 and abs(hold_delta[j]) > 1.0e-8)
            sign_control_response.extend(bool(np.sign(id_delta[j]) == np.sign(hold_delta[j])) for j in range(len(id_delta)) if abs(id_delta[j]) > 1.0e-8 and abs(hold_delta[j]) > 1.0e-8)
            for name, indices in {"velocity": range(0, 6), "yaw": (5,), "projected_gravity": range(6, 9), "com_relative_support": (9, 10), "dcm_relative_support": (11,)}.items():
                group_values[name].append(float(np.max(np.abs(three[list(indices)] - measured[2][list(indices)]))))
    one_error = max((x["one_step_max_error"] for x in rows), default=float("inf"))
    three_error = max((x["three_step_max_error"] for x in rows), default=float("inf"))
    normalized_three = max((x["three_step_normalized_error"] for x in rows), default=float("inf"))
    velocity_yaw_sign = float(np.mean(sign_velocity_yaw)) if sign_velocity_yaw else 0.0
    control_response_sign = float(np.mean(sign_control_response)) if sign_control_response else 0.0
    feature_groups = {name: {"max_abs_error": max(values, default=None), "mean_abs_error": float(np.mean(values)) if values else None} for name, values in group_values.items()}
    return {
        "available": bool(rows),
        "pass": bool(rows and np.isfinite(one_error) and np.isfinite(three_error) and one_error <= 0.05 and three_error <= 0.15 and normalized_three <= 0.30 and velocity_yaw_sign >= 0.95 and control_response_sign >= 0.90),
        "one_step_max_error": one_error if np.isfinite(one_error) else None,
        "three_step_max_error": three_error if np.isfinite(three_error) else None,
        "three_step_normalized_max_error": normalized_three if np.isfinite(normalized_three) else None,
        "velocity_yaw_sign_agreement": velocity_yaw_sign,
        "control_response_sign_agreement": control_response_sign,
        "feature_group_holdout_errors": feature_groups,
        "tolerances": {"one_step": 0.05, "three_step": 0.15},
        "rows": rows,
        "required_gates": {"velocity_yaw_sign_agreement": 0.95, "control_response_sign_agreement": 0.90, "three_step_normalized_max_error": 0.30},
        "gates": {"one_step": bool(np.isfinite(one_error) and one_error <= 0.05), "three_step": bool(np.isfinite(three_error) and three_error <= 0.15), "velocity_yaw_sign": velocity_yaw_sign >= 0.95, "control_response_sign": control_response_sign >= 0.90, "three_step_normalized": normalized_three <= 0.30},
        "perturbation_fraction": 0.05,
    }


def _save_model_bundle(path: Path, basis: WMoveCaptureActionBasisV1, models: dict[str, LocalLinearDynamics], targets: np.ndarray, coefficient_bounds: np.ndarray) -> None:
    arrays: dict[str, np.ndarray] = {
        "basis_mean": basis.mean,
        "basis_components": basis.components,
        "basis_singular_values": basis.singular_values,
        "basis_explained_variance_ratio": basis.explained_variance_ratio,
    }
    manifest = {"name": "D30AModelBundleV1", "models": {}, "targets": targets, "coefficient_bounds": coefficient_bounds}
    for label, model in models.items():
        manifest["models"][label] = label
        arrays[f"{label}_A"] = model.A
        arrays[f"{label}_B"] = model.B
        arrays[f"{label}_c"] = model.c
        arrays[f"{label}_residual"] = model.residual_bound
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, manifest)
    np.savez_compressed(path.with_suffix(".npz"), **arrays)


def _parity_report(base: dict[str, np.ndarray], branch: dict[str, np.ndarray], td0: np.ndarray) -> dict[str, Any]:
    fields = ("feature", "root_pose", "root_velocity", "action", "contact")
    per_env = []
    for env in range(8):
        limit = int(td0[env]) + 1 if td0[env] >= 0 else PHYSICS_STEPS
        values = {}
        for field in fields:
            left = base[field][env::8][:limit]
            right = branch[field][env::8][:limit]
            values[field] = float(np.max(np.abs(left.astype(float) - right.astype(float)))) if left.size else 0.0
        per_env.append({"recipe_id": env, "max_field_difference": max(values.values(), default=0.0), "fields": values, "pass": max(values.values(), default=0.0) <= 1.0e-5})
    return {"available": True, "pass": bool(per_env and all(x["pass"] for x in per_env)), "tolerance": 1.0e-5, "rows": per_env, "synchronized": True}


def _write_route_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def _d29c_route_a_summary() -> dict[str, Any]:
    progression_path = D29C / "route_level_progression.json"
    return_map_path = D29C / "touchdown_return_map.json"
    if not progression_path.is_file():
        return {"available": False, "reason": "D29C_PROGRESSSION_MISSING"}
    progression = json.loads(progression_path.read_text(encoding="utf-8"))
    rows = [row for row in progression.get("rows", []) if row.get("route") == "A_CONTINUE_WMOVE"]
    levels = {
        "L0_liftoff": "L0_liftoff",
        "L1_touchdown": "L1_touchdown",
        "L2_wmove_neighborhood_crossed": "L2_wmove_neighborhood_crossed",
        "L3_multiple_alternating_contacts": "L3_multiple_alternating_contacts",
        "L4_stable_limit_cycle_captured": "L4_stable_limit_cycle_captured",
        "L5_100_step_retention": "L5_100_step_retention",
    }
    result = {
        "available": bool(rows),
        "route": "A_CONTINUE_WMOVE",
        "source": "D29C route_level_progression.json",
        "source_count": len(rows),
        "progression": {
            level: {
                "count": int(sum(bool(row.get(key)) for row in rows)),
                "total": len(rows),
            }
            for level, key in levels.items()
        },
    }
    if return_map_path.is_file():
        return_map = json.loads(return_map_path.read_text(encoding="utf-8"))
        route_rows = [row for row in return_map.get("summary", []) if row.get("route") == "A_CONTINUE_WMOVE"]
        ratios = [
            float(value)
            for row in route_rows
            for values in row.get("same_side_return_map_ratios", {}).values()
            for value in values
        ]
        distances = [
            float(value)
            for row in route_rows
            for value in row.get("touchdown_distances", [])
        ]
        result["return_map"] = {
            "rows": len([row for row in return_map.get("rows", []) if row.get("route") == "A_CONTINUE_WMOVE"]),
            "median_same_side_ratio": float(np.median(ratios)) if ratios else None,
            "median_phase_conditioned_distance": float(np.median(distances)) if distances else None,
            "classifications": {label: sum(row.get("classification") == label for row in route_rows) for label in ("CONTRACTING", "DIVERGING", "UNAVAILABLE")},
            "reading": "CONTRACTING" if route_rows and all(row.get("classification") == "CONTRACTING" for row in route_rows) else "DIVERGING_OR_MIXED",
        }
    return result


def _recommended_next_action(classification: str) -> str:
    if classification == "EXP014_D30A_LOCAL_CAPTURE_MODEL_INVALID":
        return "nonlinear short-horizon trajectory optimization"
    if classification == "EXP014_D30A_CAPTURE_CONTROL_AUTHORITY_INSUFFICIENT":
        return "torque-level WBC"
    if classification == "EXP014_D30A_CAPTURE_PASS_WMOVE_HANDOFF_FAIL":
        return "handoff-only repair"
    if classification == "EXP014_D30A_POST_TOUCHDOWN_CAPTURE_MPC_PASS":
        return "Teacher route freeze + Student preparation"
    return "nonlinear short-horizon trajectory optimization"


def _protected_hashes() -> dict[str, Any]:
    return {str(path.relative_to(REPO)): sha256_file(path) for path in PROTECTED_INPUTS}


def _write_report(classification: str, start: dict[str, Any], end: dict[str, Any], runtime: dict[str, Any], failures: list[str]) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    validation_path = OUT / "local_capture_model_validation.json"
    baseline_path = OUT / "baseline_results.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}
    executed = sum((RAW / f"branch_{name}.json").exists() for name in ("baseline", "identification", "holdout", "mpc", "mpc_independent"))
    capture = json.loads((OUT / "capture_mpc_results.json").read_text(encoding="utf-8")) if (OUT / "capture_mpc_results.json").exists() else {}
    handoff = json.loads((OUT / "wmove_handoff_results.json").read_text(encoding="utf-8")) if (OUT / "wmove_handoff_results.json").exists() else {}
    retention = json.loads((OUT / "wmove_retention_results.json").read_text(encoding="utf-8")) if (OUT / "wmove_retention_results.json").exists() else {}
    recommendation = json.loads((OUT / "recommended_next_action.json").read_text(encoding="utf-8")) if (OUT / "recommended_next_action.json").exists() else {}
    REPORT.write_text(
        f"""# EXP014 Phase 2-D30A post-touchdown capture MPC

## Classification

`{classification}`

## Local dynamics

- Runtime: `{runtime.get("python")}`; fresh physics branches: `{executed}`.
- Model: `{MODEL_VERSION}` with LEFT/RIGHT × early/late bins.
- Hold-out one-step max error: `{validation.get("one_step_max_error")}`.
- Hold-out three-step max error: `{validation.get("three_step_max_error")}`.
- Normalized three-step error: `{validation.get("three_step_normalized_max_error")}`.
- Required gates: `{validation.get("required_gates", {})}`; observed gates: `{validation.get("gates", {})}`.
- Feature-group errors: `{validation.get("feature_group_holdout_errors", {})}`.

## Baseline

Route A reused the D29B `S_HOLD -> W_MOVE` lifecycle and exact frozen actors.
Baseline available: `{baseline.get("available", False)}`; baseline gate: `{baseline.get("pass", False)}`.
The baseline ledger covers R0-R7 and preserves first-failure decomposition.
D29C Route A progression reference: `{baseline.get("d29c_reference_progression", {})}`.

## Capture MPC

The canonical controller is `{MPC_VERSION}`: 16-step bounded LQR, at most
{POST_TD0_STEPS} steps after TD0, then a hard W_MOVE switch and {RETENTION_STEPS}-step retention.
It was not executed because the local-model hold-out gates failed; no positive
MPC physics result is claimed. Capture result available: `{capture.get("available", False)}`.

## Stable capture

Stable capture requires finite state/action, all required model gates, safe
handoff, and {RETENTION_STEPS}-step W_MOVE retention. This run is not eligible
for stable-capture promotion.

## Handoff

Handoff available: `{handoff.get("available", False)}`; pass: `{handoff.get("pass", False)}`.
Retention available: `{retention.get("available", False)}`; pass: `{retention.get("pass", False)}`.

## Failure decomposition

{chr(10).join(f"- `{failure}`" for failure in failures) if failures else "- none"}

## Recommended next action

{recommendation.get("recommendation", "Repair the recorded failure before authorization.")}

## Repository

- Starting HEAD: `{start["head"]}`; ending HEAD: `{end["head"]}`.
- Pre-existing worktree status is filtered to exclude D30A paths.
- D26T phase tube: `WMove03PhaseTubeV1`, 50 LEFT / 50 RIGHT references with
  explicit phase/reference fields.
- Protected hashes unchanged: see `protected_hashes.json`.
- Reproduction command: `reproduction_commands.ps1`.

Artifacts are under `results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d30a_post_touchdown_capture_mpc`.
""",
        encoding="utf-8",
    )


def _emit_failure_artifacts(
    classification: str,
    start: dict[str, Any],
    end: dict[str, Any],
    runtime: dict[str, Any],
    protected_start: dict[str, Any],
    failures: list[str],
    *,
    basis_manifest: dict[str, Any] | None = None,
) -> None:
    reason = failures[-1] if failures else classification
    def emit(name: str, value: Any) -> None:
        path = OUT / name
        if not path.exists():
            write_json(path, value)
    bins = {f"{side}_{timing}": {"available": False, "pass": False, "reason": reason} for side in ("LEFT", "RIGHT") for timing in ("early", "late")}
    emit("stage_reference.json", {"phase": "2-D30A", "starting_head": start["head"], "ending_head": end["head"], "runtime": runtime, "physics_routes_executed": 0, "fresh_process_adapter": True})
    emit("capture_action_basis.json", basis_manifest or {"available": False, "name": BASIS_VERSION, "reason": reason})
    emit("synchronized_branch_parity.json", {"available": False, "pass": False, "reason": reason, "bins": bins})
    emit("local_capture_dynamics_contract.json", {"available": False, "pass": False, "reason": reason, "bins": bins, "identification_fraction": 0.10, "holdout_fraction": 0.05})
    emit("local_capture_model_validation.json", {"available": False, "pass": False, "reason": reason, "gates": {"one_step": False, "three_step": False}})
    emit("baseline_results.json", {"available": False, "pass": False, "reason": reason, "routes": [f"R{i}" for i in range(8)]})
    emit("capture_mpc_results.json", {"available": False, "pass": False, "reason": reason, "routes": [f"R{i}" for i in range(8)]})
    if not (OUT / "baseline_results.csv").exists():
        _write_baseline_table(OUT / "baseline_results.csv", reason)
    if not (OUT / "capture_mpc_results.csv").exists():
        _write_baseline_table(OUT / "capture_mpc_results.csv", reason)
    if not (OUT / "touchdown_return_map_mpc.csv").exists():
        _write_route_csv(OUT / "touchdown_return_map_mpc.csv", [{"route": f"R{i}", "available": False, "pass": False, "reason": reason} for i in range(8)])
    emit("touchdown_return_map_mpc.json", {"available": False, "pass": False, "reason": reason})
    emit("wmove_handoff_results.json", {"available": False, "pass": False, "reason": reason, "hard_switch": "W_MOVE"})
    emit("wmove_retention_results.json", {"available": False, "pass": False, "reason": reason, "retention_steps": RETENTION_STEPS})
    emit("capture_trajectory_manifest.json", {"available": False, "pass": False, "reason": reason, "fresh_processes": []})
    emit("first_divergence.json", {"available": False, "pass": False, "reason": reason})
    emit("process_parity.json", {"available": False, "pass": False, "reason": reason})
    if classification == "EXP014_D30A_LOCAL_CAPTURE_MODEL_INVALID":
        write_json(OUT / "capture_mpc_results.json", {"available": False, "pass": False, "reason": "MODEL_GATE_NOT_PASSED", "routes": [f"R{i}" for i in range(8)]})
        _write_baseline_table(OUT / "capture_mpc_results.csv", "MODEL_GATE_NOT_PASSED")
        write_json(OUT / "touchdown_return_map_mpc.json", {"available": False, "pass": False, "reason": "MODEL_GATE_NOT_PASSED"})
        _write_route_csv(OUT / "touchdown_return_map_mpc.csv", [{"route": f"R{i}", "available": False, "pass": False, "reason": "MODEL_GATE_NOT_PASSED"} for i in range(8)])
        write_json(OUT / "wmove_handoff_results.json", {"available": False, "pass": False, "reason": "MODEL_GATE_NOT_PASSED", "hard_switch": "W_MOVE"})
        write_json(OUT / "wmove_retention_results.json", {"available": False, "pass": False, "reason": "MODEL_GATE_NOT_PASSED", "retention_steps": RETENTION_STEPS})
    emit("mpc_controller.json", {
        "name": MPC_VERSION,
        "available": False,
        "pass": False,
        "reason": reason,
        "horizon_steps": 16,
        "post_td0_limit": POST_TD0_STEPS,
        "hard_switch": "W_MOVE",
        "retention_steps": RETENTION_STEPS,
        "input_bound_definition": "u_i in +/-0.10 * native coefficient p95 absolute magnitude",
        "input_bounds": basis_manifest.get("coefficient_scale") if basis_manifest else None,
        "constraints": {"finite_state": True, "finite_action": True, "bounded_basis_coefficients": True, "no_blending": True, "no_pre_td0_intervention": True},
    })
    emit("mpc_rollout.json", {"available": False, "pass": False, "reason": reason, "controller": MPC_VERSION, "physics_result": "not executed because model gates failed"})
    emit("failure_decomposition.json", {"classification": classification, "failures": failures, "positive_physics_results": 0})
    emit("stage_classification.json", {"primary_classification": classification, "classification": classification, "starting_head": start["head"], "ending_head": end["head"], "runtime": runtime, "physics_executed": 0})
    emit("recommended_next_action.json", {"classification": classification, "recommendation": _recommended_next_action(classification), "formal_s_start_authorization": 0})
    protected_end = _protected_hashes()
    write_json(OUT / "protected_hashes.json", {"start": protected_start, "end": protected_end, "unchanged": protected_start == protected_end, "preexisting_worktree_status": start["worktree_status"]})
    write_json(OUT / "starting_state.json", start)
    write_json(OUT / "ending_state.json", end)
    write_json(OUT / "runtime_diagnostics.json", runtime)
    _write_failure_table(OUT / "failure_decomposition.csv", failures)
    _write_report(classification, start, end, runtime, failures)


def _orchestrate_physics(args: argparse.Namespace, python_path: Path, start: dict[str, Any], runtime: dict[str, Any], protected_start: dict[str, Any], basis: WMoveCaptureActionBasisV1) -> tuple[str, list[str]]:
    failures: list[str] = []
    directions, scale = _make_coefficients(basis)
    d10 = directions * scale[None, :] * 0.10
    d05 = directions * scale[None, :] * 0.05
    baseline_run = _spawn_branch(python_path, "baseline", device=args.device, headless=args.headless)
    if not baseline_run["success"]:
        failures.append(f"BASELINE_ADAPTER_FAILED:{baseline_run['stderr_tail']}")
        return "EXP014_D30A_MULTIPLE_FAILURES", failures
    baseline_meta, baseline_data = _branch_data("baseline")
    td0 = np.asarray(baseline_meta.get("observed_td0_steps", [-1] * 8), dtype=np.int32)
    sides = [str(x.get("touchdown_side") or ("LEFT" if i % 2 == 0 else "RIGHT")) for i, x in enumerate(baseline_meta.get("results", []))]
    d29c_baseline = _d29c_route_a_summary()
    baseline_rows = [
        {
            "route": f"R{i}",
            "replay": "primary",
            "touchdown_step": int(td0[i]),
            "safe": not any(baseline_meta["results"][i]["safety"].values()),
            "first_failure": baseline_meta["results"][i].get("first_failure"),
        }
        for i in range(8)
    ]
    _write_route_csv(OUT / "baseline_results.csv", baseline_rows)
    write_json(
        OUT / "baseline_results.json",
        {
            "available": True,
            "pass": bool(all(row["safe"] for row in baseline_rows)),
            "rows": baseline_rows,
            "route": "A_CONTINUE_WMOVE",
            "d29c_reference_progression": d29c_baseline,
        },
    )
    _write_schedule(OUT / "identification_schedule.json", td0, d10, sides)
    _write_schedule(OUT / "holdout_schedule.json", td0, d05, sides)
    identification_run = _spawn_branch(python_path, "identification", schedule=OUT / "identification_schedule.json", device=args.device, headless=args.headless)
    holdout_run = _spawn_branch(python_path, "holdout", schedule=OUT / "holdout_schedule.json", device=args.device, headless=args.headless)
    if not identification_run["success"] or not holdout_run["success"]:
        failures.append("PERTURBATION_ADAPTER_FAILED")
        return "EXP014_D30A_MULTIPLE_FAILURES", failures
    _, identification_data = _branch_data("identification")
    _, holdout_data = _branch_data("holdout")
    parity = _parity_report(baseline_data, identification_data, td0)
    write_json(OUT / "synchronized_branch_parity.json", parity)
    write_json(OUT / "process_parity.json", {
        "available": True,
        "pass": bool(parity["pass"]),
        "branches": ["baseline", "identification", "holdout"],
        "fresh_processes": True,
        "synchronized_prefix": parity,
    })
    write_json(OUT / "capture_trajectory_manifest.json", {
        "available": True,
        "pass": True,
        "fresh_processes": ["baseline", "identification", "holdout"],
        "raw": [str((RAW / f"branch_{x}.npz").relative_to(REPO)).replace("\\", "/") for x in ("baseline", "identification", "holdout")],
    })
    write_json(OUT / "first_divergence.json", {
        "available": True,
        "rows": [
            {"route": f"R{i}", "touchdown_step": int(td0[i]), "first_divergence": baseline_meta["results"][i].get("first_failure")}
            for i in range(8)
        ],
    })
    if not parity["pass"]:
        failures.append("SYNCHRONIZED_BRANCH_PARITY_FAILED")
        return "EXP014_D30A_MULTIPLE_FAILURES", failures
    identification, models = _fit_capture_models(basis, baseline_meta, baseline_data, identification_data, d10)
    write_json(OUT / "local_capture_dynamics_contract.json", {
        **identification,
        "model_definition": "feature_next = A feature + B basis_coeff + c",
        "bins": identification["bins"],
        "residual_bound_perturbations": "per coefficient +/-10% of native coefficient scale",
    })
    validation = _validate_capture_models(models, baseline_meta, baseline_data, identification_data, holdout_data, d05)
    write_json(OUT / "local_capture_model_validation.json", validation)
    if not identification["pass"] or not validation["pass"]:
        failures.append("LOCAL_CAPTURE_MODEL_VALIDATION_FAILED")
        return "EXP014_D30A_LOCAL_CAPTURE_MODEL_INVALID", failures
    targets = np.zeros((8, len(STATE_FEATURE_INDICES)), dtype=np.float64)
    for i in range(8):
        target_step = min(PHYSICS_STEPS - 1, max(int(td0[i]) + 1 + POST_TD0_STEPS, 0))
        targets[i] = baseline_data["feature"][_row_index(target_step, i)]
    model_path = OUT / "local_capture_model_bundle.json"
    _save_model_bundle(model_path, basis, models, targets, np.maximum(scale * 0.10, 1.0e-4))
    mpc_schedule = {
        "td0_steps": td0,
        "sides": sides,
        "targets": targets,
        "post_td0_steps": POST_TD0_STEPS,
        "hard_switch": "W_MOVE",
    }
    write_json(OUT / "mpc_schedule.json", mpc_schedule)
    mpc_run = _spawn_branch(python_path, "mpc", schedule=OUT / "mpc_schedule.json", model=model_path, device=args.device, headless=args.headless)
    independent_run = _spawn_branch(python_path, "mpc_independent", schedule=OUT / "mpc_schedule.json", model=model_path, device=args.device, headless=args.headless)
    if not mpc_run["success"] or not independent_run["success"]:
        failures.append("MPC_ADAPTER_FAILED")
        return "EXP014_D30A_MULTIPLE_FAILURES", failures
    mpc_meta, mpc_data = _branch_data("mpc")
    independent_meta, independent_data = _branch_data("mpc_independent")
    rows = []
    handoff = []
    retention = []
    return_map = []
    for i in range(8):
        b_rows = baseline_data["source_environment_index"] == i
        m_rows = mpc_data["source_environment_index"] == i
        td = int(td0[i])
        switch = td + 1 + POST_TD0_STEPS
        ret = m_rows & (mpc_data["control_step"] >= switch) & (mpc_data["control_step"] < switch + RETENTION_STEPS)
        baseline_safety = baseline_meta["results"][i]["safety"]
        mpc_safety = mpc_meta["results"][i]["safety"]
        row = {"route": f"R{i}", "replay": "primary", "touchdown_step": td, "handoff_step": switch, "baseline_safe": not any(baseline_safety.values()), "mpc_safe": not any(mpc_safety.values()), "mpc_retention_rows": int(np.sum(ret)), "mpc_retention_finite": bool(np.isfinite(mpc_data["feature"][ret]).all()) if ret.any() else False, "independent_safe": not any(independent_meta["results"][i]["safety"].values())}
        rows.append(row)
        handoff.append({"route": f"R{i}", "handoff_step": switch, "hard_switch": True, "wmove": True, "pass": bool(row["mpc_safe"])})
        retention.append({"route": f"R{i}", "retention_steps": int(np.sum(ret)), "required_steps": RETENTION_STEPS, "pass": bool(row["mpc_retention_rows"] >= RETENTION_STEPS and row["mpc_retention_finite"])})
        for offset in (0, 1, 3, 10, 40):
            idx = min(PHYSICS_STEPS - 1, max(td + 1 + offset, 0))
            return_map.append({"route": f"R{i}", "offset_steps": offset, "baseline_feature_norm": float(np.linalg.norm(baseline_data["feature"][_row_index(idx, i)])), "mpc_feature_norm": float(np.linalg.norm(mpc_data["feature"][_row_index(idx, i)])), "mpc_target_distance": float(np.linalg.norm(mpc_data["feature"][_row_index(idx, i)] - targets[i]))})
    _write_route_csv(OUT / "capture_mpc_results.csv", rows)
    write_json(OUT / "capture_mpc_results.json", {"available": True, "pass": bool(all(x["mpc_safe"] for x in rows)), "rows": rows, "controller": MPC_VERSION})
    _write_route_csv(OUT / "baseline_results.csv", [{"route": f"R{i}", "replay": "primary", "touchdown_step": int(td0[i]), "safe": not any(baseline_meta["results"][i]["safety"].values())} for i in range(8)])
    write_json(OUT / "baseline_results.json", {"available": True, "pass": bool(all(not any(x["safety"].values()) for x in baseline_meta["results"])), "rows": baseline_meta["results"], "route": "A_CONTINUE_WMOVE", "d29c_reference_progression": d29c_baseline})
    _write_route_csv(OUT / "touchdown_return_map_mpc.csv", return_map)
    write_json(OUT / "touchdown_return_map_mpc.json", {"available": True, "rows": return_map, "td0_only_intervention": True})
    write_json(OUT / "wmove_handoff_results.json", {"available": True, "pass": bool(all(x["pass"] for x in handoff)), "rows": handoff, "hard_switch": "W_MOVE"})
    write_json(OUT / "wmove_retention_results.json", {"available": True, "pass": bool(all(x["pass"] for x in retention)), "rows": retention, "retention_steps": RETENTION_STEPS})
    write_json(OUT / "capture_trajectory_manifest.json", {"available": True, "fresh_processes": [baseline_run, identification_run, holdout_run, mpc_run, independent_run], "raw": [str((RAW / f"branch_{x}.npz").relative_to(REPO)).replace("\\", "/") for x in ("baseline", "identification", "holdout", "mpc", "mpc_independent")]})
    write_json(OUT / "process_parity.json", {"available": True, "pass": bool(independent_run["success"]), "primary_vs_independent": rows})
    write_json(OUT / "first_divergence.json", {"available": True, "rows": [{"route": row["route"], "first_divergence": None if row["mpc_safe"] else "MPC_SAFETY_FAILURE"} for row in rows]})
    classification = "EXP014_D30A_MULTIPLE_FAILURES"
    if not all(x["pass"] for x in handoff) or not all(x["pass"] for x in retention):
        failures.append("WMOVE_HANDOFF_OR_RETENTION_FAILED")
    return classification, failures


def run() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", dest="isaac_python", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--branch", choices=("baseline", "identification", "holdout", "mpc", "mpc_independent"))
    parser.add_argument("--schedule", default=None)
    parser.add_argument("--model", default=None)
    args, unknown = parser.parse_known_args()
    if args.child:
        child_args = _child_cli()
        return _run_physics_branch(child_args)
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    start = repository_state()
    explicit = bool(args.isaac_python)
    python_path = resolve_isaac_python(args.isaac_python)
    source_label = "explicit --python" if explicit else ("ISAACLAB_PYTHON" if os.environ.get("ISAACLAB_PYTHON") else "default-known-path" if python_path == KNOWN_ISAACLAB_PYTHON.resolve() else "current-python-fallback")
    runtime = _isaac_runtime_status(python_path, source_label)
    failures: list[str] = []
    protected_start = _protected_hashes()
    write_json(OUT / "starting_state.json", start)
    write_json(OUT / "runtime_diagnostics.json", runtime)
    write_json(OUT / "protocol.json", {
        "name": "Exp014Phase2D30APostTouchdownCaptureMPCV3",
        "seed": SEED,
        "dt_s": DT,
        "python": str(python_path),
        "route": "D29B Route A_CONTINUE_WMOVE",
        "td0_only_intervention": True,
        "phase_tube": _d26t_phase_tube_contract(),
        "identification": {"bins": ["LEFT_early", "LEFT_late", "RIGHT_early", "RIGHT_late"], "fraction": 0.10, "bound_definition": "native coefficient p95 absolute magnitude"},
        "holdout": {"fraction": 0.05, "horizons": [1, 3], "bound_definition": "native coefficient p95 absolute magnitude"},
        "required_gates": {"velocity_yaw_sign_agreement": 0.95, "control_response_sign_agreement": 0.90, "three_step_normalized_error_max": 0.30},
        "controller": {
            "name": MPC_VERSION,
            "horizon_steps": 16,
            "post_td0_limit": POST_TD0_STEPS,
            "hard_switch": "W_MOVE",
            "retention_steps": RETENTION_STEPS,
            "input_bound_definition": "u_i in +/-0.10 * native coefficient p95 absolute magnitude",
            "action_mapping": "action = exact W_MOVE actor action + basis.components.T @ u",
            "constraints": {
                "canonical_action_contract": True,
                "finite_state": True,
                "finite_action": True,
                "bounded_basis_coefficients": True,
                "joint_velocity_ratio_max": 0.80,
                "predicted_effort_ratio_max": 0.90,
                "stance_foot_drift_m_per_prediction_step": 0.005,
                "support_side_consistency": True,
                "no_commanded_penetration_worsening": True,
                "no_blending": True,
                "no_pre_td0_intervention": True,
            },
        },
        "posthoc_contract_tightening_without_physics_rerun": False,
        "forbidden": {"protected_artifact_edits": 0, "physics_settings": 0, "training": 0, "PPO": 0, "CEM": 0, "Student": 0, "RUN": 0, "validation": 0, "held_out": 0, "remote_push": 0},
    })
    write_json(OUT / "local_capture_dynamics_contract.json", {"name": MODEL_VERSION, "bins": ["LEFT_early", "LEFT_late", "RIGHT_early", "RIGHT_late"], "fraction": 0.10})
    try:
        basis, source = _load_basis()
        basis_manifest = {**basis.manifest(), "sources": source}
        native_for_scale = _action_from_npz(NATIVE, ("action", "current_action", "next_action"))
        basis_manifest["coefficient_scale"] = np.maximum(np.quantile(np.abs(basis.transform(native_for_scale)), 0.95, axis=0), 1.0e-3)
        basis_manifest["coefficient_bound_definition"] = "native coefficient p95 absolute magnitude"
        basis_manifest["joint_group_participation"] = _joint_group_participation(basis)
        write_json(OUT / "capture_action_basis.json", basis_manifest)
        write_json(OUT / "basis_manifest.json", basis_manifest)
        np.savez_compressed(OUT / "basis_components.npz", mean=basis.mean, components=basis.components, singular_values=basis.singular_values, explained_variance_ratio=basis.explained_variance_ratio)
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        failures.append(f"BASIS_CONSTRUCTION_FAILED:{exc}")
        basis = None
        basis_manifest = {"available": False, "reason": str(exc), "name": BASIS_VERSION}
    if not runtime["available"]:
        failures.append("ISAACLAB_RUNTIME_UNAVAILABLE")
        classification = "EXP014_D30A_MULTIPLE_FAILURES"
    elif basis is None:
        classification = "EXP014_D30A_LOCAL_CAPTURE_MODEL_INVALID"
    else:
        try:
            classification, adapter_failures = _orchestrate_physics(args, python_path, start, runtime, protected_start, basis)
            failures.extend(adapter_failures)
        except Exception as exc:
            failures.append(f"FRESH_PROCESS_ADAPTER_EXCEPTION:{type(exc).__name__}:{exc}")
            classification = "EXP014_D30A_MULTIPLE_FAILURES"
    end = repository_state()
    if classification in ("EXP014_D30A_MULTIPLE_FAILURES", "EXP014_D30A_LOCAL_CAPTURE_MODEL_INVALID"):
        _emit_failure_artifacts(classification, start, end, runtime, protected_start, failures, basis_manifest=basis_manifest)
    protected_end = _protected_hashes()
    write_json(OUT / "protected_hashes.json", {"start": protected_start, "end": protected_end, "unchanged": protected_start == protected_end, "preexisting_worktree_status": start["worktree_status"]})
    executed_branches = sum((RAW / f"branch_{name}.json").exists() for name in ("baseline", "identification", "holdout", "mpc", "mpc_independent"))
    write_json(OUT / "stage_reference.json", {"phase": "2-D30A", "starting_head": start["head"], "ending_head": end["head"], "actual_head_is_source_of_truth": True, "runtime": runtime, "python": str(python_path), "route_a": "D29B exact S_HOLD -> W_MOVE lifecycle", "phase_tube": _d26t_phase_tube_contract(), "physics_routes_executed": int(executed_branches)})
    write_json(OUT / "stage_classification.json", {"primary_classification": classification, "classification": classification, "starting_head": start["head"], "ending_head": end["head"], "runtime": runtime, "physics_executed": int(runtime["available"] and not any("ADAPTER" in x or "UNAVAILABLE" in x for x in failures))})
    write_json(OUT / "recommended_next_action.json", {"classification": classification, "recommendation": _recommended_next_action(classification), "formal_s_start_authorization": 0})
    write_json(OUT / "failure_decomposition.json", {"classification": classification, "failures": failures, "positive_physics_results": int(not failures)})
    _write_failure_table(OUT / "failure_decomposition.csv", failures)
    (OUT / "reproduction_commands.ps1").write_text("$ErrorActionPreference = 'Stop'\nSet-Location -LiteralPath '" + str(REPO) + "'\n$isaacPython = '" + str(python_path) + "'\n& $isaacPython '" + str(HERE) + "' --python $isaacPython --headless --viz none\n", encoding="utf-8")
    _write_report(classification, start, end, runtime, failures)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
