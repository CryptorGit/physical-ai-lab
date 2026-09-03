"""Freeze source, parent, training, and evaluation provenance before training."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import jax
import jaxlib
import mujoco
import optax
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import losses as ppo_losses


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP_ROOT.parents[2]
OUT = EXP_ROOT / "artifacts" / "v60_bounded_yaw_pilot"
SOURCE_ROOT = Path("/home/user/openduck_training_backward_v23_20260729")
PARENT = Path(
    "/home/user/openduck_training_runs/"
    "coupled_head_original_stand_backward_v45_50m/"
    "2026_07_29_235335_47349760"
)
PARENT_ONNX = Path(str(PARENT) + ".onnx")


HISTORICAL_FILES = (
    "playground/common/poly_reference_motion.py",
    "playground/common/randomize.py",
    "playground/common/rewards.py",
    "playground/open_duck_mini_v2/base.py",
    "playground/open_duck_mini_v2/constants.py",
    "playground/open_duck_mini_v2/custom_rewards.py",
    "playground/open_duck_mini_v2/joystick.py",
    "playground/open_duck_mini_v2/data/optimized_backward_gait.json",
    "playground/open_duck_mini_v2/data/optimized_backward_left_turn_gait.json",
    "playground/open_duck_mini_v2/data/optimized_backward_right_turn_gait.json",
    "playground/open_duck_mini_v2/data/polynomial_coefficients_calibrated.pkl",
    "playground/open_duck_mini_v2/xmls/open_duck_mini_v2_backlash_calibrated.xml",
    "playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash_calibrated.xml",
)
LOCAL_FILES = (
    "scripts/train_v60_bounded_yaw_pilot.py",
    "scripts/evaluate_v60_bounded_yaw_pilot.py",
    "scripts/evaluate_v59_corrected_15s_diagnostic.py",
    "tools/v60_yaw_objective.py",
    "tools/analyze_v60_yaw_objective.py",
    "tools/freeze_v60_provenance.py",
    "tools/convert_v60_pickle_checkpoint_to_orbax.py",
    "tools/export_v59_stochastic_trace.py",
    "tools/v59_mjx_diagnostic_common.py",
    "tests/test_v60_old_yaw_objective_contract.py",
    "tests/test_v60_new_yaw_objective_contract.py",
    "tests/test_v60_yaw_symmetry.py",
    "tests/test_v60_paired_arm_identity.py",
    "tests/test_v60_reward_only_difference.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def array_tree_sha256(tree: Any) -> str:
    digest = hashlib.sha256()
    paths, definition = jax.tree_util.tree_flatten_with_path(tree)
    digest.update(str(definition).encode())
    for path, leaf in paths:
        array = __import__("numpy").asarray(leaf)
        name = jax.tree_util.keystr(path).encode()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def run(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(EXP_ROOT / "scripts"))
    import train_v60_bounded_yaw_pilot as training

    source_files: dict[str, dict[str, Any]] = {}
    archive_members: list[tuple[Path, str]] = []
    for relative in HISTORICAL_FILES:
        path = SOURCE_ROOT / relative
        source_files[f"historical/{relative}"] = {
            "path": str(path),
            "sha256": sha256(path),
            "size": path.stat().st_size,
        }
        archive_members.append((path, f"historical/{relative}"))
    for relative in LOCAL_FILES:
        path = EXP_ROOT / relative
        source_files[f"experiment/{relative}"] = {
            "path": str(path),
            "sha256": sha256(path),
            "size": path.stat().st_size,
        }
        archive_members.append((path, f"experiment/{relative}"))

    archive = OUT / "immutable_source_snapshot_pre_training.tar.gz"
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as tar:
        for path, arcname in archive_members:
            info = tar.gettarinfo(str(path), arcname=arcname)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as stream:
                tar.addfile(info, stream)

    branch = run(["git", "branch", "--show-current"], cwd=REPO_ROOT)
    head = run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    status = run(
        ["git", "status", "--short", "--untracked-files=all", "--", str(EXP_ROOT)],
        cwd=REPO_ROOT,
    )
    source_git = {
        "branch": run(["git", "branch", "--show-current"], cwd=SOURCE_ROOT),
        "head": run(["git", "rev-parse", "HEAD"], cwd=SOURCE_ROOT),
        "status": run(
            ["git", "status", "--short", "--untracked-files=all"],
            cwd=SOURCE_ROOT,
        ),
    }
    nvidia = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    baseline = {
        "frozen_before_training": True,
        "git": {"branch": branch, "head": head, "status": status},
        "historical_source_git": source_git,
        "source_files": source_files,
        "immutable_archive": {
            "path": str(archive),
            "sha256": sha256(archive),
            "size": archive.stat().st_size,
            "member_count": len(archive_members),
        },
        "runtime": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "mujoco": mujoco.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "x64_enabled": bool(jax.config.jax_enable_x64),
            "jit_enabled": not bool(jax.config.jax_disable_jit),
            "nvidia_smi": nvidia,
        },
    }
    (OUT / "baseline_source_manifest.json").write_text(
        json.dumps(baseline, indent=2) + "\n", encoding="utf-8"
    )

    params = checkpoint.load(str(PARENT))
    normalizer, actor, critic = params
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(learning_rate=3e-4)
    )
    network_params = ppo_losses.PPONetworkParams(policy=actor, value=critic)
    optimizer_state = optimizer.init(network_params)
    parent_manifest = {
        "package": "v52 hybrid controller",
        "checkpoint_path": str(PARENT),
        "checkpoint_tree_sha256": tree_sha256(PARENT),
        "checkpoint_format": "Orbax PyTree: normalizer, actor, critic",
        "normalizer_sha256": array_tree_sha256(normalizer),
        "actor_sha256": array_tree_sha256(actor),
        "critic_sha256": array_tree_sha256(critic),
        "optimizer_state_in_parent": False,
        "fresh_optimizer_initial_state_sha256": array_tree_sha256(
            optimizer_state
        ),
        "optimizer_initialization": "optax clip_by_global_norm(1.0) + Adam(3e-4), zero moments",
        "onnx_path": str(PARENT_ONNX),
        "onnx_sha256": sha256(PARENT_ONNX),
        "reverse_profiles": {
            key: source_files[f"historical/playground/open_duck_mini_v2/data/{key}"]
            for key in (
                "optimized_backward_gait.json",
                "optimized_backward_left_turn_gait.json",
                "optimized_backward_right_turn_gait.json",
            )
        },
        "teacher_routing_source_sha256": source_files[
            "historical/playground/open_duck_mini_v2/joystick.py"
        ]["sha256"],
        "scene_sha256": source_files[
            "historical/playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash_calibrated.xml"
        ]["sha256"],
    }
    (OUT / "parent_checkpoint_manifest.json").write_text(
        json.dumps(parent_manifest, indent=2) + "\n", encoding="utf-8"
    )

    control = training.resolved_config("control")
    treatment = training.resolved_config("treatment")
    differences = {
        key: {"control": control.get(key), "treatment": treatment.get(key)}
        for key in sorted(control)
        if control.get(key) != treatment.get(key)
    }
    allowed = {"objective_name", "run_name", "output_path"}
    unexpected = sorted(set(differences) - allowed)
    training_contract = {
        "valid_for_training": not unexpected,
        "allowed_intentional_differences": sorted(allowed),
        "resolved_differences": differences,
        "unexpected_differences": unexpected,
        "common_random_numbers": {
            "seed": training.SEED,
            "same_backend": True,
            "same_environment_and_policy_rng_derivation": True,
            "qualification": (
                "reward-dependent termination can change later random draw "
                "consumption; CRN is exact until environment histories diverge"
            ),
        },
        "control": control,
        "treatment": treatment,
    }
    (OUT / "training_contract.json").write_text(
        json.dumps(training_contract, indent=2) + "\n", encoding="utf-8"
    )
    for arm, config in (("control", control), ("treatment", treatment)):
        path = OUT / "configs" / f"{arm}_resolved_config.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    evaluation_contract = {
        "backend": "GPU MJX only",
        "underlying_evaluator": str(
            EXP_ROOT / "scripts" / "evaluate_v59_corrected_15s_diagnostic.py"
        ),
        "seconds": 15,
        "formal_acceptance_eligible": False,
        "enough_episodes": False,
        "diagnostic_only": True,
        "primary": {
            "commands": ["yaw_left_0p6", "yaw_right_0p6"],
            "condition_d_seeds": 5,
            "condition_s_seeds": 5,
        },
        "retention": {
            "minimum_commands": [
                "stand",
                "forward",
                "backward",
                "left_strafe",
                "right_strafe",
                "representative_forward_yaw",
                "representative_backward_yaw",
            ],
            "condition_d_seeds": 3,
            "condition_s_seeds": 3,
        },
        "implementation_runs_full_19x5_and_subsets_retention_rows": True,
    }
    (OUT / "evaluation_contract.json").write_text(
        json.dumps(evaluation_contract, indent=2) + "\n", encoding="utf-8"
    )

    # Keep both requested naming conventions byte-identical.
    for source, alias in (
        ("old_objective_contract.json", "objective_contract_old.json"),
        ("new_objective_contract.json", "objective_contract_new.json"),
    ):
        (OUT / alias).write_bytes((OUT / source).read_bytes())

    provenance_md = f"""# v60 Source Provenance

- Main repository: `{branch}` at `{head}`.
- Historical MJX source: `{source_git['branch']}` at `{source_git['head']}`.
- Both trees contain uncommitted/untracked material; no commit was created.
- Immutable source archive: `{archive.name}`, SHA-256 `{sha256(archive)}`.
- Parent Orbax tree: `{parent_manifest['checkpoint_tree_sha256']}`.
- Parent ONNX: `{parent_manifest['onnx_sha256']}`.
- Parent contains normalizer, actor and critic but no Adam state.
- Fresh optimizer initial-state hash: `{parent_manifest['fresh_optimizer_initial_state_sha256']}`.
- Arm diff gate: `{'PASS' if not unexpected else 'FAIL'}`.
"""
    (OUT / "source_provenance.md").write_text(
        provenance_md, encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "archive_sha256": sha256(archive),
                "parent_tree_sha256": parent_manifest[
                    "checkpoint_tree_sha256"
                ],
                "arm_diff_gate": not unexpected,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
