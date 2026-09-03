"""Pure contracts and metrics for routed OpenDuckMini simulation acceptance.

The MuJoCo/ONNX runtime lives in ``scripts/evaluate_routed_transitions.py``.
Everything in this module is CPU-only and independently unit-testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from .contract import (
    ACTUATOR_JOINT_ORDER,
    CONTRACT,
    HEAD_JOINTS,
    LEG_TARGET_MARGIN_RAD,
    RESET_NOISE_MARGIN_RAD,
    SAFE_INIT_POS,
    SAFE_JOINT_LIMITS,
    TARGET_SLEW_LIMIT_RAD_PER_S,
)
from .gait_quality import rederive_gait_quality_acceptance
from target_safety import (
    BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD,
    BACKWARD_EXIT_RECOVERY_HOLD_SECONDS,
    BACKWARD_EXIT_RECOVERY_HOLD_TICKS,
    BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD,
    backward_exit_recovery_contract,
)


SCHEMA_VERSION = 1
EVALUATOR_ID = "openduckmini-exp004-routed-transition-v1"
HEAD_ACTION_INDICES = (5, 6, 7, 8)
EXP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = EXP_ROOT.parents[2]

ADOPTED_REVERSE_PROFILE_SHA256 = (
    "fd2f3a6c129ed0c37a9014dbad1813764cca35cdb54dc0b63d39f82b925e2306"
)
ADOPTED_REVERSE_LEFT_PROFILE_SHA256 = (
    "474d5fe1b25859167d53aa70eac496986414ea759a4b3293c7f064dc7e1c870a"
)
ADOPTED_REVERSE_RIGHT_PROFILE_SHA256 = (
    "1db80d8763ca991813954a3297f4a67c8e076265a0bbb2b9f8a10a518ff9a0ba"
)
REVERSE_V1_ADOPTION_STATUS = "REJECTED_AWAITING_REOPTIMIZATION"
REVERSE_V1_MEASURED_FORWARD_VELOCITY_MPS = -0.00156

BASE_V22_POLICY_SHA256 = (
    "f7a2731330cd3be52858989b021423a5f363cc4a8f9850512281da745a7617c0"
)
FORMAL_POLICY_SHA256_ALLOWLIST = frozenset({BASE_V22_POLICY_SHA256})
MAX_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD = 0.05
DIAGNOSTIC_REVERSE_V3_PROFILE_SHA256 = (
    "af7f14c2c4877a088b9320d59625bd37e41677ddc3a3802761df1e982179373e"
)
DIAGNOSTIC_REVERSE_V3_PROFILE_PATH = (
    EXP_ROOT / "artifacts" / "optimized_reverse_margin050_slew200_candidate_v3.json"
).resolve()
DIAGNOSTIC_REVERSE_TURN_PROFILE_SHA256 = MappingProxyType(
    {
        "left": "b36f14dc1bbacfbf998adc00f6e6fe62d1f14a4a8de034b1b0b18ae5bccb8703",
        "right": "e2229527d435d03636c091ca7b435ed3be483b0e74293d28a2ff927995bea16b",
    }
)
DIAGNOSTIC_REVERSE_V3_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD = 0.0125
FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD = 0.0125
FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_UPPER_TARGET_RAD = 0.413034
H3_CANDIDATE_SELECTION_STATUS = (
    "FORMAL_CANDIDATE_H3_5X15_PASSED_PENDING_20X30"
)
FORMAL_CANDIDATE_STATUS = "ADOPTED_SIMULATION_ONLY"
H3_FAST_EXIT_SAFETY_STATUS = (
    "FORMAL_CANDIDATE_H3_FAST_EXIT_SAFETY_PASSED_PENDING_COMBINED_5X15"
)
H2_5X15_SELECTION_STATUS = (
    "FORMAL_CANDIDATE_H2_5X15_PASSED_PENDING_20X30"
)
H2_COMPONENT_STATUS = (
    "FORMAL_CANDIDATE_H2_COMPONENTS_PASSED_PENDING_COMBINED_5X15"
)
FORMAL_CANDIDATE_MASTER_SEED = 20_260_808
FORMAL_CANDIDATE_STRAIGHT_PROFILE_SHA256 = (
    "0a3c0849124b397ca1cb60ae0b5f5783a2e545f1a03108846fa8c60cd5d8bb5b"
)
FORMAL_CANDIDATE_PROFILE_SHA256S = MappingProxyType(
    {
        "straight": FORMAL_CANDIDATE_STRAIGHT_PROFILE_SHA256,
        **dict(DIAGNOSTIC_REVERSE_TURN_PROFILE_SHA256),
    }
)
FORMAL_CANDIDATE_PROFILE_PATHS = MappingProxyType(
    {
        "straight": (
            EXP_ROOT
            / "artifacts"
            / "optimized_reverse_margin050_slew200_h1_phase7_rate105_candidate_v1.json"
        ).resolve(),
        "left": (
            EXP_ROOT
            / "artifacts"
            / "reverse_turn_candidates_v1"
            / "optimized_reverse_turn_left_margin050_slew200_candidate_v1.json"
        ).resolve(),
        "right": (
            EXP_ROOT
            / "artifacts"
            / "reverse_turn_candidates_v1"
            / "optimized_reverse_turn_right_margin050_slew200_candidate_v1.json"
        ).resolve(),
    }
)
H2_5X15_SELECTION_EVIDENCE_SHA256 = (
    "6f65bef5053da5962442eca3bf46b855a36691aa9bbad84496c9892b36ee0de4"
)
H2_5X15_SELECTION_EVIDENCE_PATH = (
    EXP_ROOT / "artifacts" / "h2_combined_candidate_5x15_seed20260808_v1.json"
).resolve()
H2_5X15_SELECTION_EVIDENCE_SHA256_ALLOWLIST = frozenset(
    {H2_5X15_SELECTION_EVIDENCE_SHA256}
)
FORMAL_CANDIDATE_SELECTION_EVIDENCE_SHA256 = (
    "f040a9c6f9783b7d50dd5590389d3c81411e8f3a7fa9dd155e8ac78175d5ff56"
)
FORMAL_CANDIDATE_SELECTION_EVIDENCE_PATH = (
    EXP_ROOT / "artifacts" / "h3_combined_candidate_5x15_seed20260808_v1.json"
).resolve()
FORMAL_CANDIDATE_SELECTION_EVIDENCE_SHA256_ALLOWLIST = frozenset(
    {FORMAL_CANDIDATE_SELECTION_EVIDENCE_SHA256}
)
H2_COMPONENT_SELECTION_EVIDENCE_SHA256 = (
    "bfaf052235e15262c34a794896e2c63a62bd1bd934998a77b7f6ea6c54009133"
)
H2_COMPONENT_SELECTION_EVIDENCE_PATH = (
    EXP_ROOT
    / "artifacts"
    / "h2_integrated_phase744_rate105_recovery0175_hold13_transition20x9_v1.json"
).resolve()
H2_SUPERSEDED_ADOPTION_EVIDENCE_SHA256 = (
    "bd7e8a79b32880fa63e54570854682b5b8912f1cdafeed8e80273501dc6ef611"
)
H2_SUPERSEDED_ADOPTION_EVIDENCE_PATH = (
    EXP_ROOT
    / "artifacts"
    / "h2_formal_candidate_pending_20x30_seed20260808_v1.json"
).resolve()
FORMAL_ADOPTION_EVIDENCE_SHA256 = (
    "1aea58904598cfba8ea4ef572f9473bba647eacc695f7fce3fcaa1b8646391aa"
)
FORMAL_ADOPTION_EVIDENCE_PATH = (
    EXP_ROOT
    / "artifacts"
    / "h3_formal_candidate_pending_20x30_seed20260808_v1.json"
).resolve()
FORMAL_ADOPTION_EVIDENCE_SHA256_ALLOWLIST = frozenset(
    {FORMAL_ADOPTION_EVIDENCE_SHA256}
)
H2_SUPERSEDED_ADOPTION_STATUS = "SUPERSEDED_H2_ADOPTION_LINEAGE"
H2_SUPERSEDED_RECOVERY_EXTRA_UPPER_MARGIN_RAD = 0.0175
H2_SUPERSEDED_RECOVERY_UPPER_TARGET_RAD = 0.408034
H2_SUPERSEDED_RECOVERY_HOLD_TICKS = 13
H2_SUPERSEDED_RECOVERY_HOLD_SECONDS = 0.26
H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256 = (
    "090e09cc2d82c1f42112a5f30a85cd93d940213956d6ec902fb4089875fb855a"
)
H3_FAST_EXIT_SAFETY_EVIDENCE_PATH = (
    EXP_ROOT
    / "artifacts"
    / "h2_aggressive_short_transition_recovery0225_hold13_20seed_v1.json"
).resolve()
H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256_ALLOWLIST = frozenset(
    {H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256}
)
H3_FAST_EXIT_EXPECTED_MOTION_FAILURES = (
    (22_260_809, "transition_reverse_turn_left", "signed_yaw_progress"),
    (22_260_816, "transition_reverse_turn_right", "signed_linear_progress"),
    (22_260_817, "transition_reverse_turn_right", "signed_linear_progress"),
    (22_260_818, "transition_reverse_turn_right", "signed_linear_progress"),
    (22_260_819, "transition_reverse_turn_right", "signed_linear_progress"),
    (22_260_822, "transition_reverse_turn_right", "signed_linear_progress"),
    (22_260_823, "transition_reverse_turn_right", "signed_linear_progress"),
    (22_260_824, "transition_reverse", "signed_linear_progress"),
    (22_260_824, "transition_reverse_turn_right", "signed_linear_progress"),
    (22_260_825, "transition_reverse_turn_right", "signed_linear_progress"),
    (22_260_826, "transition_reverse", "signed_linear_progress"),
)
HISTORICAL_FAILED_FORMAL_CANDIDATE_EVIDENCE_SHA256 = (
    "e975a078f452bdfe215d136b015b16d8b6b89f69f8777874fef80b6836efaead"
)
HISTORICAL_FAILED_FORMAL_CANDIDATE_EVIDENCE_PATH = (
    EXP_ROOT / "artifacts" / "formal_candidate_pending_20x30_seed20260808_v1.json"
).resolve()
H1_STRAIGHT_20X30_EVIDENCE_SHA256 = (
    "ff9412da4a6813151b82894553e789231cc20717ab377dd3fb0c24a1d2da2a5e"
)
H1_STRAIGHT_20X30_EVIDENCE_PATH = (
    EXP_ROOT / "artifacts" / "h1_phase57_rate105_formal20x30s_v1.json"
).resolve()
H1_TRANSITION_PREFIX_20SEED_EVIDENCE_SHA256 = (
    "5cfff9e96d363797433ec50f8e4f18af25469597b0bdb2623a28ecdbfbc42f19"
)
H1_TRANSITION_PREFIX_20SEED_EVIDENCE_PATH = (
    EXP_ROOT
    / "artifacts"
    / "h1_phase7_rate105_formal_transition_reverse_prefix_20seed_v1.json"
).resolve()
H1_REJECTED_COUPLED_CAP_EVIDENCE_SHA256 = (
    "1edd0d0e560f6ca832af1500d659d51493e1269aa65dc429c0bb42f067a74389"
)
H1_REJECTED_COUPLED_CAP_EVIDENCE_PATH = (
    EXP_ROOT / "artifacts" / "h1_phase7_rate105_cap01625_formal20x30s_v1.json"
).resolve()
LEGACY_STAGE_A_5X15_EVIDENCE_SHA256 = (
    "8fe375ce044d86987364909df3b7122a9108ef58316d294a5e6e3f82ed30b51c"
)
ATOMIC_REVERSE_TURN_COMMANDS = MappingProxyType(
    {
        "left": (-0.03, 0.0, 0.20),
        "right": (-0.04, 0.0, -0.20),
    }
)

# This mapping is not encoded in any candidate profile, so the immutable
# v3/turn profile hashes remain unchanged. Values are phase indices immediately
# before the normal per-profile increment. Stage A promotes the same values to
# the default formal-candidate execution bundle; the diagnostic alias remains
# available only to reproduce the historical selection path.
BACKWARD_FAMILY_EXPERTS = frozenset(
    {"reverse", "reverse_turn_left", "reverse_turn_right"}
)
FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES = MappingProxyType(
    {
        "reverse": 7.0,
        "reverse_turn_left": 4.0,
        "reverse_turn_right": 4.0,
    }
)
FROZEN_DIAGNOSTIC_REVERSE_ENTRY_PHASE_INDICES = MappingProxyType(
    {
        "reverse": 6.0,
        "reverse_turn_left": 4.0,
        "reverse_turn_right": 4.0,
    }
)
DIAGNOSTIC_REVERSE_PHASE_ENTRY_EVIDENCE_SHA256 = (
    "c78643b1c4deee8c293c6f27535190e4b8ca8d80f809de7fa72fa2ffc6751742"
)
DIAGNOSTIC_REVERSE_PHASE_ENTRY_EVIDENCE_PATH = (
    EXP_ROOT
    / "artifacts"
    / "reverse_transition_candidate_v3_phase_entry006_5x15_v1.json"
).resolve()
DIAGNOSTIC_REVERSE_PHASE_ENTRY_SOURCE_REVERSE_ENDPOINT_MPS = -0.075
DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_EVIDENCE_SHA256 = (
    "588f652320efd7758a540fbeb85a273047de518a6e28aeb9dc3ff52dc0368504"
)
DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_EVIDENCE_PATH = (
    EXP_ROOT
    / "artifacts"
    / "backward_exit_recovery_phase644_5seed_prefix_v1.json"
).resolve()
DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_SOURCE_REVERSE_ENDPOINT_MPS = -0.075
CURRENT_FORMAL_REVERSE_ENDPOINT_MPS = -0.050
DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_FIXED_SEEDS = (
    22_260_808,
    22_260_809,
    22_260_810,
    22_260_811,
    22_260_812,
)

FROZEN_RUNTIME_DEPENDENCY_PATHS = MappingProxyType(
    {
        "official_policy_evaluator": (
            WORKSPACE_ROOT
            / "experiments"
            / "mujoco"
            / "exp_003_openduckmini_calibrated_walk"
            / "evaluate_official_policy.py"
        ).resolve(),
        "polynomial_reference_motion": (
            WORKSPACE_ROOT
            / ".openduck_playground_source_review"
            / "playground"
            / "common"
            / "poly_reference_motion_numpy.py"
        ).resolve(),
        "playground_package_init": (
            WORKSPACE_ROOT
            / ".openduck_playground_source_review"
            / "playground"
            / "__init__.py"
        ).resolve(),
        "playground_common_package_init": (
            WORKSPACE_ROOT
            / ".openduck_playground_source_review"
            / "playground"
            / "common"
            / "__init__.py"
        ).resolve(),
    }
)
FROZEN_RUNTIME_DEPENDENCY_SHA256 = MappingProxyType(
    {
        "official_policy_evaluator": (
            "5825d879e41001bcea0f3bb8741e11198671117723916c9c9a011c4fe5ca5cbc"
        ),
        "polynomial_reference_motion": (
            "a11ac12a5318ba1ba365693434d8eda23c7240c6480613f9574587a7a1d4e2c1"
        ),
        "playground_package_init": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
        "playground_common_package_init": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
    }
)
FROZEN_RUNTIME_DEPENDENCY_ROOT_SHA256 = (
    "a40d4920049b349334b6d5567859fc2f7533d8fe6648f5127f0c7e4ce54dc098"
)
FROZEN_RUNTIME_VERSIONS = MappingProxyType(
    {
        "python": "3.12.3",
        "numpy": "2.5.1",
        "mujoco": "3.11.0",
        "onnxruntime": "1.28.0",
    }
)
FROZEN_RUNTIME_BINARY_SHA256 = MappingProxyType(
    {
        "python_executable": (
            "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118"
        ),
        "libmujoco": (
            "7b47f06aa840d4f8fa3c66c2027d52de570bac6d11b9ddd19d55ae1469d40e22"
        ),
        "libonnxruntime": (
            "aa4079d18f4ea7a5f3a94d80cd4bbe0f2740436626622d64d793803a20381083"
        ),
        "onnxruntime_pybind": (
            "6a31ea840051445c5176aface4c7474e09f40dd22d3e2d1106a52ef86d55fd83"
        ),
        "numpy_core": (
            "3aac21341706a466577d7c7caab9c5013e98ffb62cf98ae3bc46a3e1004f3ce0"
        ),
    }
)

FROZEN_GENERATED_ROOT = (
    Path(__file__).resolve().parents[1] / "artifacts" / "generated_playground"
).resolve()
FROZEN_GENERATED_PRIMARY_SHA256 = MappingProxyType(
    {
        "manifest": "5d99971a546120eb9d3b3eae04b5dc356d3bec13750fd1ab5edb8e9db463ef4d",
        "scene": "92ede737aebfdb1ab266764e9b1de114b246a5d85fa2cf0335878fa0b6915a39",
        "model": "5dd10ce3da0238fdfbeaaada9d5dc6ca64dccdbf10fe478b3e8ac73a85ec2699",
        "reference": "26b4a80d3800c79f731035ffc310aea8d7b8da177674871c7fe412d2741ef6ad",
    }
)
_FROZEN_GENERATED_MESH_SHA256 = MappingProxyType(
    {
        "antenna.stl": "8d18ff3ed5dc5beb89c15b95fc886e4d57f5fd8f76969532fe83f679eb0f08e6",
        "body_back.stl": "7166e8ccd5aa659566bf84c659a36c33248740a18d1eb04b0e34904dde71e733",
        "body_front.stl": "5c98c85f91a979df57ecda622b64f35689efd2d076df0ca584186daf773cb47c",
        "body_middle_bottom.stl": "3f91cfeeffc197d39a5e11e26811825cbffdaedcc3eea0f4195d701f994eb203",
        "body_middle_top.stl": "0929c3d0c45b9b41273b1333655c467bec994f6489df5ea2c7c1298e4c4d7d39",
        "foot_bottom_pla.stl": "4290199f08b3e8eb4ecd4f07c32c4d77ecf82839396ccd7d6062eea96cef8e6d",
        "foot_bottom_tpu.stl": "dfa3c58757e1d838b462d225c2dbfbb517e8fc57e3c6a9979c4894c801386e8a",
        "foot_side.stl": "3d564b42ef9c2bae340fd0b06cfadfe905c713ff84ad40b21d4181128d225138",
        "foot_top.stl": "0712dd6aec2e6d6c821b8bfcd0ba71bf7c8017f63889de5f6847e62814624a40",
        "head.stl": "e81b30c44d10d93194091d5fbb65bd4ccb6bd4e9504c027344c247d2ffc6eaf4",
        "head_bot_sheet.stl": "39cdafc528646e1ca627b357e8f9886a9f51df3d53dcfd566d3b7979fd592953",
        "head_pitch_to_yaw.stl": "a45c8dc5fb75ec001fd1e94185db635c60aa68f163482395d6c55db4ac678238",
        "head_yaw_to_roll.stl": "3a94ac2824e94326565a3e07f8794c9246c90f78c119a4577835b9c70745557b",
        "left_antenna_holder.stl": "01536add58f330d36fd9c63c4a6caeb414a1d931a27e501f63567a0785f93d23",
        "left_cache.stl": "8b50aa42f2f34b1e99c0702e6d6d64ee1b0a7fa7e856d97558b2388310930f85",
        "left_knee_to_ankle_left_sheet.stl": "e69aff1a7baeedd18676348ea72d185229722f8b1348bb5b2ffcafc7de9715bf",
        "left_knee_to_ankle_right_sheet.stl": "2518ec33ff2b018304b80026603d3f88e8e41f27474b5f43f1f7e4b5e10c1461",
        "left_roll_to_pitch.stl": "1af14b0e3bef3298d20f3d5b985848aaf9dc69097c8b449c43fc63181ee01d66",
        "leg_spacer.stl": "4f23745dac3598daf9bf6089e3864a318d091a5613af8544aecf3e68d6a6772e",
        "neck_left_sheet.stl": "2dc8e910a537509a018191c4e2a0b535b78c4ff822b6d2cf39f34427c83e38a6",
        "neck_right_sheet.stl": "afd4c04b1081a0a68d92a017fabc16d093f1ddeed52a212b5e50585cc2bb928e",
        "right_antenna_holder.stl": "39604b35cbe5f0070371fc460e51f89b09c20a06c003bff9cf87f7685234648a",
        "right_cache.stl": "391a1477b223cea4bbea37a64d154188294093c677cc52f62880ec7f001f061e",
        "right_roll_to_pitch.stl": "22d3e6f5e0afd386bc0bf288077ae99f3822f82ea4c1e6e47bd36fdcc21d1599",
        "roll_motor_bottom.stl": "3dc60095901ac2332cfb798dcd53149c2e98abeb636637f6bd7e8b9757fa9ecf",
        "roll_motor_top.stl": "89872314b73e926dc1a37dc4d0d4756bf74e82316a8f207cb160924ec0372e62",
        "trunk_bottom.stl": "d2382a154ccb241911225a560cf49985d363d476699a9198eb4ad9d5bfbb5e36",
        "trunk_top.stl": "943a7c75a81ede71fc985ecc37ccd8efec05b9be604302e04af8c9fa3a516c16",
    }
)
_GENERATED_ASSET_PREFIX = "playground/open_duck_mini_v2/xmls/assets/"
FROZEN_GENERATED_DEPENDENCY_SHA256 = MappingProxyType(
    {
        "playground/open_duck_mini_v2/xmls/open_duck_mini_v2_backlash_hardware_safe_calibrated.xml": MappingProxyType(
            {
                "kind": "xml",
                "sha256": FROZEN_GENERATED_PRIMARY_SHA256["model"],
            }
        ),
        **{
            f"{_GENERATED_ASSET_PREFIX}{name}": MappingProxyType(
                {"kind": "mesh", "sha256": digest}
            )
            for name, digest in _FROZEN_GENERATED_MESH_SHA256.items()
        },
    }
)
FROZEN_GENERATED_DEPENDENCY_ROOT_SHA256 = (
    "b8edb35a8d83b358a4234dd37664c3deccb0c01f36edb88b3dedf550831b9982"
)

LEGACY_ADOPTED_REVERSE_PROFILE_SHA256_ALLOWLISTS = MappingProxyType(
    {
        "straight": frozenset({ADOPTED_REVERSE_PROFILE_SHA256}),
        "left": frozenset({ADOPTED_REVERSE_LEFT_PROFILE_SHA256}),
        "right": frozenset({ADOPTED_REVERSE_RIGHT_PROFILE_SHA256}),
    }
)
# H3's immutable profile bank is adopted for simulation-only execution.  Its
# hashes remain independently gated from both evaluation and release evidence.
FORMAL_CANDIDATE_REVERSE_PROFILE_SHA256_ALLOWLISTS = MappingProxyType(
    {
        label: frozenset({digest})
        for label, digest in FORMAL_CANDIDATE_PROFILE_SHA256S.items()
    }
)
FORMAL_H3_CANDIDATE_REVERSE_PROFILE_SHA256_ALLOWLISTS = (
    FORMAL_CANDIDATE_REVERSE_PROFILE_SHA256_ALLOWLISTS
)
# The public formal allowlist now denotes the adopted H3 runtime profiles.
FORMAL_REVERSE_PROFILE_SHA256_ALLOWLISTS = (
    FORMAL_CANDIDATE_REVERSE_PROFILE_SHA256_ALLOWLISTS
)
# Historical API alias: profile hashes alone never grant current adoption.
FORMAL_H2_ADOPTED_REVERSE_PROFILE_SHA256_ALLOWLISTS = (
    FORMAL_CANDIDATE_REVERSE_PROFILE_SHA256_ALLOWLISTS
)
FORMAL_REVERSE_ADOPTION_EVIDENCE_SHA256_ALLOWLISTS = MappingProxyType(
    {
        label: frozenset({FORMAL_ADOPTION_EVIDENCE_SHA256})
        for label in ("straight", "left", "right")
    }
)
# Backward-compatible name; this is the adoption evidence gate, never the
# Stage-A candidate-selection evidence allowlist above.
FORMAL_REVERSE_EVIDENCE_SHA256_ALLOWLISTS = (
    FORMAL_REVERSE_ADOPTION_EVIDENCE_SHA256_ALLOWLISTS
)
FORMAL_REVERSE_ADOPTION_STATUSES = MappingProxyType(
    {
        label: FORMAL_CANDIDATE_STATUS
        for label in ("straight", "left", "right")
    }
)
FORMAL_REVERSE_COMMAND_CASE_NAMES = frozenset(
    {
        "reverse",
        "reverse_turn_left",
        "reverse_turn_right",
        "transition_reverse",
        "transition_reverse_turn_left",
        "transition_reverse_turn_right",
    }
)
FORMAL_REVERSE_COMMAND_CASE_ADOPTION_EVIDENCE_SHA256_ALLOWLISTS = (
    MappingProxyType(
        {
            name: frozenset({FORMAL_ADOPTION_EVIDENCE_SHA256})
            for name in FORMAL_REVERSE_COMMAND_CASE_NAMES
        }
    )
)
FORMAL_REVERSE_COMMAND_CASE_SAFETY_EVIDENCE_SHA256_ALLOWLISTS = (
    MappingProxyType(
        {
            name: frozenset({H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256})
            for name in FORMAL_REVERSE_COMMAND_CASE_NAMES
        }
    )
)
REVERSE_ADOPTION_BLOCKING_STATUS_MARKERS = (
    "BLOCKED",
    "REJECTED",
    "PENDING",
    "DIAGNOSTIC",
    "NOT_ADOPTED",
)

REQUIRED_POLICY_ROLES = (
    "stand",
    "forward",
    "reverse",
    "lateral_left",
    "lateral_right",
    "yaw_left",
    "yaw_right",
    "compound",
)

POLICY_ROLE_ALIASES = {
    "reverse_turn_left": "compound",
    "reverse_turn_right": "compound",
}

PROHIBITED_POLICY_LABELS = frozenset(
    {
        "v59",
        "v60",
        "all_direction_v59",
        "all_direction_v60",
        "omnidirectional_v59",
        "omnidirectional_v60",
    }
)


@dataclass(frozen=True)
class CommandCase:
    name: str
    command: tuple[float, float, float]
    policy_observation_command: tuple[float, float, float] | None = None
    expected_expert: str = ""
    expected_policy_role: str = ""
    validation_status: str = "FORMAL_CANDIDATE"
    validation_evidence_sha256: str | None = None
    safety_evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        command = np.asarray(self.command, dtype=np.float64)
        if not self.name or command.shape != (3,) or not np.all(np.isfinite(command)):
            raise ValueError("command cases require a name and finite vx/vy/yaw")
        if not self.validation_status:
            raise ValueError("command cases require an explicit validation status")
        if self.validation_evidence_sha256 is not None and (
            len(self.validation_evidence_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.validation_evidence_sha256
            )
        ):
            raise ValueError("command case evidence must be a lowercase SHA-256")
        if self.safety_evidence_sha256 is not None and (
            len(self.safety_evidence_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.safety_evidence_sha256
            )
        ):
            raise ValueError("command case safety evidence must be a lowercase SHA-256")
        expected_role = POLICY_ROLE_ALIASES.get(
            self.expected_expert, self.expected_expert
        )
        if (
            not self.expected_expert
            or self.expected_policy_role not in REQUIRED_POLICY_ROLES
            or expected_role != self.expected_policy_role
        ):
            raise ValueError(
                "command cases require an explicit consistent expert/policy role"
            )
        if self.expected_expert in PROHIBITED_POLICY_LABELS:
            raise ValueError("command cases cannot expect a prohibited expert")
        if self.policy_observation_command is not None:
            policy_command = np.asarray(
                self.policy_observation_command, dtype=np.float64
            )
            if policy_command.shape != (3,) or not np.all(np.isfinite(policy_command)):
                raise ValueError(
                    "policy_observation_command must be a finite vx/vy/yaw triplet"
                )


# Seven primitives means stand plus the six signed single-axis motions.
PRIMITIVE_CASES = (
    CommandCase(
        "stand",
        (0.0, 0.0, 0.0),
        expected_expert="stand",
        expected_policy_role="stand",
        validation_status="UNCHANGED_BASELINE",
    ),
    CommandCase(
        "forward",
        (0.05, 0.0, 0.0),
        (0.10, 0.0, 0.0),
        expected_expert="forward",
        expected_policy_role="forward",
    ),
    CommandCase(
        "reverse",
        (-0.050, 0.0, 0.0),
        expected_expert="reverse",
        expected_policy_role="reverse",
        validation_status=FORMAL_CANDIDATE_STATUS,
        validation_evidence_sha256=FORMAL_ADOPTION_EVIDENCE_SHA256,
        safety_evidence_sha256=H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256,
    ),
    CommandCase(
        "lateral_left",
        (0.0, 0.06, 0.0),
        (0.0, 0.10, 0.0),
        expected_expert="lateral_left",
        expected_policy_role="lateral_left",
    ),
    CommandCase(
        "lateral_right",
        (0.0, -0.06, 0.0),
        (0.0, -0.10, 0.0),
        expected_expert="lateral_right",
        expected_policy_role="lateral_right",
    ),
    CommandCase(
        "yaw_left",
        (0.0, 0.0, 0.30),
        (0.0, -0.06, 0.60),
        expected_expert="yaw_left",
        expected_policy_role="yaw_left",
    ),
    CommandCase(
        "yaw_right",
        (0.0, 0.0, -0.30),
        (0.0, 0.0, -0.80),
        expected_expert="yaw_right",
        expected_policy_role="yaw_right",
    ),
)

# Only endpoints that passed the formal exp_004 acceptance are adopted here.
# The wider training curriculum is intentionally not an acceptance surface.
COMPOUND_CASES = (
    CommandCase(
        "forward_turn_left",
        (0.04, 0.00, 0.30),
        (0.08, 0.00, 0.30),
        expected_expert="compound",
        expected_policy_role="compound",
    ),
    CommandCase(
        "forward_turn_right",
        (0.04, 0.00, -0.22),
        (0.08, 0.00, -0.45),
        expected_expert="compound",
        expected_policy_role="compound",
    ),
    CommandCase(
        "forward_lateral_left_turn",
        (0.04, 0.05, 0.17),
        (0.06, 0.05, 0.20),
        expected_expert="compound",
        expected_policy_role="compound",
    ),
    CommandCase(
        "forward_lateral_right_turn",
        (0.04, -0.03, -0.15),
        (0.06, -0.05, -0.35),
        expected_expert="compound",
        expected_policy_role="compound",
    ),
    CommandCase(
        "reverse_turn_left",
        (-0.03, 0.00, 0.20),
        expected_expert="reverse_turn_left",
        expected_policy_role="compound",
        validation_status=FORMAL_CANDIDATE_STATUS,
        validation_evidence_sha256=FORMAL_ADOPTION_EVIDENCE_SHA256,
        safety_evidence_sha256=H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256,
    ),
    CommandCase(
        "reverse_turn_right",
        (-0.04, 0.00, -0.20),
        expected_expert="reverse_turn_right",
        expected_policy_role="compound",
        validation_status=FORMAL_CANDIDATE_STATUS,
        validation_evidence_sha256=FORMAL_ADOPTION_EVIDENCE_SHA256,
        safety_evidence_sha256=H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256,
    ),
)

# Non-adoptable low-physical/high-policy-observation excitation comparison.
# Positive-yaw policy input retains exp_003's -0.06 lateral compensation.
POLICY_COMMAND_DIAGNOSTIC_CASES = (
    CommandCase(
        "diagnostic_forward_low_physical_high_policy",
        (0.05, 0.0, 0.0),
        (0.10, 0.0, 0.0),
        expected_expert="forward",
        expected_policy_role="forward",
        validation_status="DIAGNOSTIC_ONLY",
    ),
    CommandCase(
        "diagnostic_lateral_left_low_physical_high_policy",
        (0.0, 0.06, 0.0),
        (0.0, 0.10, 0.0),
        expected_expert="lateral_left",
        expected_policy_role="lateral_left",
        validation_status="DIAGNOSTIC_ONLY",
    ),
    CommandCase(
        "diagnostic_lateral_right_low_physical_high_policy",
        (0.0, -0.06, 0.0),
        (0.0, -0.10, 0.0),
        expected_expert="lateral_right",
        expected_policy_role="lateral_right",
        validation_status="DIAGNOSTIC_ONLY",
    ),
    CommandCase(
        "diagnostic_yaw_left_low_physical_high_policy",
        (0.0, 0.0, 0.30),
        (0.0, -0.06, 0.60),
        expected_expert="yaw_left",
        expected_policy_role="yaw_left",
        validation_status="DIAGNOSTIC_ONLY",
    ),
    CommandCase(
        "diagnostic_yaw_right_rejected_policy_minus_090",
        (0.0, 0.0, -0.30),
        (0.0, 0.0, -0.90),
        expected_expert="yaw_right",
        expected_policy_role="yaw_right",
        validation_status="REJECTED_DIAGNOSTIC_FELL",
    ),
    CommandCase(
        "diagnostic_yaw_right_candidate_a",
        (0.0, 0.0, -0.25),
        (0.0, 0.0, -0.60),
        expected_expert="yaw_right",
        expected_policy_role="yaw_right",
        validation_status="DIAGNOSTIC_ONLY",
    ),
    CommandCase(
        "diagnostic_yaw_right_candidate_b",
        (0.0, 0.0, -0.25),
        (0.0, 0.0, -0.70),
        expected_expert="yaw_right",
        expected_policy_role="yaw_right",
        validation_status="DIAGNOSTIC_ONLY",
    ),
    CommandCase(
        "diagnostic_yaw_right_candidate_c",
        (0.0, 0.0, -0.30),
        (0.0, 0.0, -0.80),
        expected_expert="yaw_right",
        expected_policy_role="yaw_right",
        validation_status="SELECTED_FORMAL_CANDIDATE_SOURCE",
    ),
)

REJECTED_POLICY_COMMAND_DIAGNOSTIC_CASES = frozenset(
    {"diagnostic_yaw_right_rejected_policy_minus_090"}
)

TRANSITION_CASES = (
    CommandCase(
        "transition_stand_0",
        (0.0, 0.0, 0.0),
        expected_expert="stand",
        expected_policy_role="stand",
        validation_status="UNCHANGED_BASELINE",
    ),
    CommandCase(
        "transition_forward",
        (0.05, 0.0, 0.0),
        (0.10, 0.0, 0.0),
        expected_expert="forward",
        expected_policy_role="forward",
    ),
    CommandCase(
        "transition_stand_after_forward",
        (0.0, 0.0, 0.0),
        expected_expert="stand",
        expected_policy_role="stand",
        validation_status="UNCHANGED_BASELINE",
    ),
    CommandCase(
        "transition_reverse",
        (-0.050, 0.0, 0.0),
        expected_expert="reverse",
        expected_policy_role="reverse",
        validation_status=FORMAL_CANDIDATE_STATUS,
        validation_evidence_sha256=FORMAL_ADOPTION_EVIDENCE_SHA256,
        safety_evidence_sha256=H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256,
    ),
    CommandCase(
        "transition_stand_after_reverse",
        (0.0, 0.0, 0.0),
        expected_expert="stand",
        expected_policy_role="stand",
        validation_status="UNCHANGED_BASELINE",
    ),
    CommandCase(
        "transition_reverse_turn_left",
        (-0.03, 0.0, 0.20),
        expected_expert="reverse_turn_left",
        expected_policy_role="compound",
        validation_status=FORMAL_CANDIDATE_STATUS,
        validation_evidence_sha256=FORMAL_ADOPTION_EVIDENCE_SHA256,
        safety_evidence_sha256=H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256,
    ),
    CommandCase(
        "transition_stand_after_reverse_turn_left",
        (0.0, 0.0, 0.0),
        expected_expert="stand",
        expected_policy_role="stand",
        validation_status="UNCHANGED_BASELINE",
    ),
    CommandCase(
        "transition_reverse_turn_right",
        (-0.04, 0.0, -0.20),
        expected_expert="reverse_turn_right",
        expected_policy_role="compound",
        validation_status=FORMAL_CANDIDATE_STATUS,
        validation_evidence_sha256=FORMAL_ADOPTION_EVIDENCE_SHA256,
        safety_evidence_sha256=H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256,
    ),
    CommandCase(
        "transition_stand_after_reverse_turn_right",
        (0.0, 0.0, 0.0),
        expected_expert="stand",
        expected_policy_role="stand",
        validation_status="UNCHANGED_BASELINE",
    ),
    CommandCase(
        "transition_lateral_left",
        (0.0, 0.06, 0.0),
        (0.0, 0.10, 0.0),
        expected_expert="lateral_left",
        expected_policy_role="lateral_left",
    ),
    CommandCase(
        "transition_stand_after_lateral_left",
        (0.0, 0.0, 0.0),
        expected_expert="stand",
        expected_policy_role="stand",
        validation_status="UNCHANGED_BASELINE",
    ),
    CommandCase(
        "transition_lateral_right",
        (0.0, -0.06, 0.0),
        (0.0, -0.10, 0.0),
        expected_expert="lateral_right",
        expected_policy_role="lateral_right",
    ),
    CommandCase(
        "transition_stand_after_lateral_right",
        (0.0, 0.0, 0.0),
        expected_expert="stand",
        expected_policy_role="stand",
        validation_status="UNCHANGED_BASELINE",
    ),
    CommandCase(
        "transition_yaw_left",
        (0.0, 0.0, 0.30),
        (0.0, -0.06, 0.60),
        expected_expert="yaw_left",
        expected_policy_role="yaw_left",
    ),
    CommandCase(
        "transition_stand_after_yaw_left",
        (0.0, 0.0, 0.0),
        expected_expert="stand",
        expected_policy_role="stand",
        validation_status="UNCHANGED_BASELINE",
    ),
    CommandCase(
        "transition_yaw_right",
        (0.0, 0.0, -0.30),
        (0.0, 0.0, -0.80),
        expected_expert="yaw_right",
        expected_policy_role="yaw_right",
    ),
    CommandCase(
        "transition_stand_after_yaw_right",
        (0.0, 0.0, 0.0),
        expected_expert="stand",
        expected_policy_role="stand",
        validation_status="UNCHANGED_BASELINE",
    ),
    CommandCase(
        "transition_forward_turn_left",
        (0.04, 0.0, 0.30),
        (0.08, 0.0, 0.30),
        expected_expert="compound",
        expected_policy_role="compound",
    ),
    CommandCase(
        "transition_stand_after_forward_turn_left",
        (0.0, 0.0, 0.0),
        expected_expert="stand",
        expected_policy_role="stand",
        validation_status="UNCHANGED_BASELINE",
    ),
    CommandCase(
        "transition_forward_turn_right",
        (0.04, 0.0, -0.22),
        (0.08, 0.0, -0.45),
        expected_expert="compound",
        expected_policy_role="compound",
    ),
    CommandCase(
        "transition_stand_after_forward_turn_right",
        (0.0, 0.0, 0.0),
        expected_expert="stand",
        expected_policy_role="stand",
        validation_status="UNCHANGED_BASELINE",
    ),
    CommandCase(
        "transition_forward_lateral_left_turn",
        (0.04, 0.05, 0.17),
        (0.06, 0.05, 0.20),
        expected_expert="compound",
        expected_policy_role="compound",
    ),
    CommandCase(
        "transition_stand_after_forward_lateral_left_turn",
        (0.0, 0.0, 0.0),
        expected_expert="stand",
        expected_policy_role="stand",
        validation_status="UNCHANGED_BASELINE",
    ),
    CommandCase(
        "transition_forward_lateral_right_turn",
        (0.04, -0.03, -0.15),
        (0.06, -0.05, -0.35),
        expected_expert="compound",
        expected_policy_role="compound",
    ),
    CommandCase(
        "transition_stand_final",
        (0.0, 0.0, 0.0),
        expected_expert="stand",
        expected_policy_role="stand",
        validation_status="UNCHANGED_BASELINE",
    ),
)


@dataclass(frozen=True)
class AcceptanceThresholds:
    minimum_upright: float = 0.85
    minimum_height_m: float = 0.12
    maximum_primary_velocity_error_mps: float = 0.06
    maximum_orthogonal_velocity_mps: float = 0.05
    maximum_stationary_linear_speed_mps: float = 0.04
    maximum_yaw_rate_error_radps: float = 0.25
    maximum_uncommanded_yaw_rate_radps: float = 0.20
    minimum_signed_linear_progress_fraction: float = 0.30
    minimum_signed_yaw_progress_fraction: float = 0.30
    minimum_moving_single_support_rate: float = 0.05
    maximum_flight_rate: float = 0.05
    maximum_stop_drift_m: float = 0.10
    maximum_head_applied_action: float = 0.0
    maximum_head_target_rad: float = 0.0
    maximum_safe_limit_violations: int = 0
    maximum_command_clip_events: int = 0


def transition_schedule(
    moving_seconds: float, stand_seconds: float
) -> tuple[
    tuple[
        str,
        tuple[float, float, float],
        float,
        tuple[float, float, float] | None,
        str,
        str,
    ],
    ...,
]:
    """Materialize the continuous no-reset command-transition schedule."""

    if moving_seconds <= 0.0 or stand_seconds <= 0.0:
        raise ValueError("transition durations must be positive")
    return tuple(
        (
            case.name,
            case.command,
            stand_seconds if np.allclose(case.command, 0.0) else moving_seconds,
            case.policy_observation_command,
            case.expected_expert,
            case.expected_policy_role,
        )
        for case in TRANSITION_CASES
    )


def command_case_validation_gate(
    cases: Sequence[CommandCase],
) -> dict[str, Any]:
    """Require independent H3 safety and formal-adoption case bindings."""

    nonadoptable_markers = ("BLOCKED", "DIAGNOSTIC", "REJECTED", "PENDING")
    blocked = [
        {
            "name": case.name,
            "validation_status": case.validation_status,
        }
        for case in cases
        if any(marker in case.validation_status.upper() for marker in nonadoptable_markers)
    ]
    reverse_cases = {
        case.name: case
        for case in cases
        if case.name in FORMAL_REVERSE_COMMAND_CASE_NAMES
    }
    safety_evidence_failures = [
        {
            "name": name,
            "validation_status": case.validation_status,
            "safety_evidence_sha256": case.safety_evidence_sha256,
        }
        for name, case in reverse_cases.items()
        if case.validation_status != FORMAL_CANDIDATE_STATUS
        or case.safety_evidence_sha256
        not in FORMAL_REVERSE_COMMAND_CASE_SAFETY_EVIDENCE_SHA256_ALLOWLISTS[
            name
        ]
    ]
    safety_evidence_bindings = {
        name: {
            "status": case.validation_status,
            "evidence_sha256": case.safety_evidence_sha256,
            "evidence_hash_allowlisted": not any(
                failure["name"] == name
                for failure in safety_evidence_failures
            ),
        }
        for name, case in sorted(reverse_cases.items())
    }
    adoption_evidence_bindings = {
        name: {
            "status": case.validation_status,
            "evidence_sha256": case.validation_evidence_sha256,
            "evidence_hash_allowlisted": bool(
                case.validation_evidence_sha256
                in FORMAL_REVERSE_COMMAND_CASE_ADOPTION_EVIDENCE_SHA256_ALLOWLISTS[
                    name
                ]
            ),
        }
        for name, case in sorted(reverse_cases.items())
    }
    complete = set(reverse_cases) == set(FORMAL_REVERSE_COMMAND_CASE_NAMES)
    safety_component_passed = bool(
        complete and not safety_evidence_failures
    )
    adoption_evidence_passed = bool(
        complete
        and all(
            binding["evidence_hash_allowlisted"]
            for binding in adoption_evidence_bindings.values()
        )
    )
    return {
        "passed": bool(
            complete
            and not blocked
            and safety_component_passed
            and adoption_evidence_passed
        ),
        "safety_component_passed": safety_component_passed,
        "adoption_evidence_passed": adoption_evidence_passed,
        "case_count": len(cases),
        "nonadoptable_case_count": len(blocked),
        "nonadoptable_cases": blocked,
        "nonadoptable_status_markers": list(nonadoptable_markers),
        "reverse_safety_component_evidence_case_count": len(reverse_cases),
        "reverse_safety_component_evidence_bindings": safety_evidence_bindings,
        "reverse_safety_component_evidence_failure_count": len(
            safety_evidence_failures
        ),
        "reverse_safety_component_evidence_failures": safety_evidence_failures,
        "reverse_adoption_evidence_case_count": len(reverse_cases),
        "reverse_adoption_evidence_bindings": adoption_evidence_bindings,
        "reverse_adoption_evidence_failure_count": sum(
            not binding["evidence_hash_allowlisted"]
            for binding in adoption_evidence_bindings.values()
        ),
        "reverse_adoption_evidence_failures": [
            {
                "name": name,
                "validation_status": binding["status"],
                "validation_evidence_sha256": binding["evidence_sha256"],
            }
            for name, binding in adoption_evidence_bindings.items()
            if not binding["evidence_hash_allowlisted"]
        ],
    }


def canonical_policy_role(routed_expert: str) -> str:
    """Resolve router-only reverse-turn labels to the compound policy role."""

    if routed_expert in PROHIBITED_POLICY_LABELS:
        raise ValueError(f"prohibited policy label: {routed_expert}")
    role = POLICY_ROLE_ALIASES.get(routed_expert, routed_expert)
    if role not in REQUIRED_POLICY_ROLES:
        raise ValueError(f"router selected an unregistered role: {routed_expert}")
    return role


def validate_diagnostic_reverse_entry_phase_indices(
    phase_indices: Mapping[str, float] | None,
) -> dict[str, float] | None:
    """Validate the explicit, frozen diagnostic-only phase-entry mapping."""

    if phase_indices is None:
        return None
    if not isinstance(phase_indices, Mapping):
        raise ValueError("diagnostic reverse entry phase indices must be a mapping")
    expected = dict(FROZEN_DIAGNOSTIC_REVERSE_ENTRY_PHASE_INDICES)
    if set(phase_indices) != set(expected):
        raise ValueError(
            "diagnostic reverse entry phase mapping must contain exactly "
            f"{sorted(expected)}"
        )
    validated: dict[str, float] = {}
    for expert, frozen_value in expected.items():
        value = float(phase_indices[expert])
        if not np.isfinite(value):
            raise ValueError(f"diagnostic {expert} entry phase index must be finite")
        if value != frozen_value:
            raise ValueError(
                f"diagnostic {expert} entry phase index must remain exactly "
                f"{frozen_value}"
            )
        validated[expert] = value
    return validated


def _validate_formal_candidate_reverse_entry_phase_indices(
    phase_indices: Mapping[str, float] | None,
) -> dict[str, float] | None:
    if phase_indices is None:
        return None
    if not isinstance(phase_indices, Mapping):
        raise ValueError("formal-candidate reverse entry phases must be a mapping")
    expected = dict(FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES)
    if set(phase_indices) != set(expected):
        raise ValueError("formal-candidate reverse entry phase keys drifted")
    validated = {key: float(phase_indices[key]) for key in expected}
    if any(
        not np.isfinite(value) or value != expected[key]
        for key, value in validated.items()
    ):
        raise ValueError("formal-candidate reverse entry phase values drifted")
    return validated


def advance_routed_phase(
    phase_index: float,
    *,
    phase_steps: float,
    phase_delta: float,
    current_expert: str,
    previous_expert: str | None,
    effective_command: Sequence[float],
    previous_backward_feedforward_active: bool,
    diagnostic_entry_phase_indices: Mapping[str, float] | None = None,
    phase_entry_status: str = "DIAGNOSTIC_UNADOPTED",
    diagnostic_only: bool = True,
    control_step: int | None = None,
    global_control_tick: int | None = None,
) -> tuple[float, bool, dict[str, Any] | None]:
    """Advance gait phase and apply one pinned reset at feedforward entry.

    Entry is keyed to the exact tick on which backward feedforward first becomes
    active (``effective vx < -0.02``), never to the router's earlier switch
    request.  Remaining active while switching between backward-family experts
    therefore cannot cause a second reset.
    """

    current = float(phase_index)
    count = float(phase_steps)
    delta = float(phase_delta)
    command = np.asarray(effective_command, dtype=np.float64)
    if (
        not np.isfinite(current)
        or not np.isfinite(count)
        or count <= 0.0
        or not np.isfinite(delta)
        or delta <= 0.0
    ):
        raise ValueError("phase index, phase steps, and phase delta must be finite")
    if command.shape != (3,) or not np.all(np.isfinite(command)):
        raise ValueError("effective command must be a finite vx/vy/yaw triplet")
    if not isinstance(phase_entry_status, str) or not phase_entry_status:
        raise ValueError("phase-entry status must be a non-empty string")
    if not isinstance(diagnostic_only, (bool, np.bool_)):
        raise ValueError("phase-entry diagnostic_only must be boolean")
    mapping = (
        validate_diagnostic_reverse_entry_phase_indices(
            diagnostic_entry_phase_indices
        )
        if bool(diagnostic_only)
        else _validate_formal_candidate_reverse_entry_phase_indices(
            diagnostic_entry_phase_indices
        )
    )
    backward_feedforward_active = bool(command[0] < -0.02)
    event: dict[str, Any] | None = None
    if (
        mapping is not None
        and backward_feedforward_active
        and not bool(previous_backward_feedforward_active)
        and current_expert in mapping
    ):
        before_reset = current
        current = mapping[current_expert]
        first_feedforward_phase = (current + delta) % count
        event = {
            "control_step": None if control_step is None else int(control_step),
            "global_control_tick": (
                None if global_control_tick is None else int(global_control_tick)
            ),
            "previous_expert": previous_expert,
            "current_expert": current_expert,
            "effective_command": command.tolist(),
            "activation_predicate": "effective_vx_lt_negative_0p02_false_to_true",
            "previous_backward_feedforward_active": False,
            "current_backward_feedforward_active": True,
            "global_phase_index_before_reset": before_reset,
            "reset_preincrement_phase_index": current,
            "profile_phase_rate": delta,
            "first_feedforward_phase_index": first_feedforward_phase,
            "phase_steps": count,
            "status": phase_entry_status,
            "formal_candidate": False,
            "adopted_simulation_only": not bool(diagnostic_only),
            "diagnostic_only": bool(diagnostic_only),
        }
    advanced = (current + delta) % count
    if event is not None and advanced != event["first_feedforward_phase_index"]:
        raise RuntimeError("phase-entry reset produced inconsistent phase")
    return advanced, backward_feedforward_active, event


def policy_yaw_observation_offset(
    routed_expert: str,
    effective_command: Sequence[float],
    *,
    backward_residual_scale: float,
) -> float:
    """Return the accepted route-scoped policy yaw observation correction.

    The physical/requested command is never changed.  Feedforward-only reverse
    routes have no causal policy residual and therefore receive no offset.
    """

    command = np.asarray(effective_command, dtype=np.float64)
    if command.shape != (3,) or not np.all(np.isfinite(command)):
        raise ValueError("effective command must be a finite vx/vy/yaw triplet")
    residual = float(backward_residual_scale)
    if not np.isfinite(residual) or residual < 0.0:
        raise ValueError("backward residual scale must be finite and non-negative")
    role = canonical_policy_role(routed_expert)
    if command[0] < -0.02 and residual == 0.0:
        return 0.0
    if role == "yaw_right":
        return -0.30
    if role in {"forward", "compound"} and command[2] < 0.0:
        return -0.15
    return 0.0


def resolve_policy_observation_command(
    routed_expert: str,
    effective_command: Sequence[float],
    *,
    backward_residual_scale: float,
    override: Sequence[float] | None = None,
) -> tuple[np.ndarray, float, bool]:
    """Resolve the final policy-visible command without changing physics.

    A diagnostic override is already the final three-axis policy command, so
    route-specific yaw corrections are deliberately bypassed rather than
    double-applied.
    """

    effective = np.asarray(effective_command, dtype=np.float64)
    if effective.shape != (3,) or not np.all(np.isfinite(effective)):
        raise ValueError("effective command must be a finite triplet")
    if override is not None:
        policy_command = np.asarray(override, dtype=np.float64)
        if policy_command.shape != (3,) or not np.all(np.isfinite(policy_command)):
            raise ValueError("policy observation override must be a finite triplet")
        return policy_command.copy(), 0.0, True
    yaw_offset = policy_yaw_observation_offset(
        routed_expert,
        effective,
        backward_residual_scale=backward_residual_scale,
    )
    policy_command = effective.copy()
    policy_command[2] += yaw_offset
    return policy_command, yaw_offset, False


def parse_policy_assignments(assignments: Iterable[str]) -> dict[str, Path]:
    """Parse repeated ``ROLE=PATH`` CLI values and require all eight roles."""

    result: dict[str, Path] = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"policy assignment must be ROLE=PATH: {assignment!r}")
        role, raw_path = assignment.split("=", 1)
        role = role.strip()
        raw_path = raw_path.strip()
        if role in PROHIBITED_POLICY_LABELS:
            raise ValueError(f"prohibited policy role: {role}")
        if role not in REQUIRED_POLICY_ROLES:
            raise ValueError(f"unknown policy role: {role}")
        if role in result:
            raise ValueError(f"duplicate policy role: {role}")
        if not raw_path:
            raise ValueError(f"empty policy path for role: {role}")
        result[role] = Path(raw_path).expanduser()
    missing = sorted(set(REQUIRED_POLICY_ROLES) - set(result))
    if missing:
        raise ValueError(f"missing required policy roles: {missing}")
    return result


def blend_and_mask_actions(
    from_action: Sequence[float],
    to_action: Sequence[float],
    blend_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend two routed actions, then enforce the four-channel head lock."""

    old = np.asarray(from_action, dtype=np.float64)
    new = np.asarray(to_action, dtype=np.float64)
    if old.shape != (14,) or new.shape != (14,):
        raise ValueError("expert actions must each have shape (14,)")
    if not np.all(np.isfinite(old)) or not np.all(np.isfinite(new)):
        raise ValueError("expert actions must be finite")
    alpha = float(blend_alpha)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("blend_alpha must be finite and in [0, 1]")
    raw_blended = (1.0 - alpha) * old + alpha * new
    applied = raw_blended.copy()
    applied[list(HEAD_ACTION_INDICES)] = 0.0
    return raw_blended, applied


def _safe_excess(
    values: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    return np.maximum(np.maximum(lower - values, values - upper), 0.0)


def build_target_envelope(
    joint_names: Sequence[str] = ACTUATOR_JOINT_ORDER,
    *,
    leg_margin_rad: float = LEG_TARGET_MARGIN_RAD,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the runtime target envelope inside the physical SAFE limits.

    The ten leg bounds are moved inward by ``leg_margin_rad``.  The four head
    channels remain the singleton target ``[0, 0]`` and therefore cannot be
    relaxed by changing the leg margin.
    """

    names = tuple(joint_names)
    if len(names) != 14 or set(names) != set(ACTUATOR_JOINT_ORDER):
        raise ValueError("target envelope requires the exact 14 actuator names")
    margin = float(leg_margin_rad)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("leg target margin must be finite and non-negative")
    lower = np.zeros(14, dtype=np.float64)
    upper = np.zeros(14, dtype=np.float64)
    for index, name in enumerate(names):
        if name in HEAD_JOINTS:
            continue
        safe_lower, safe_upper = SAFE_JOINT_LIMITS[name]
        lower[index] = safe_lower + margin
        upper[index] = safe_upper - margin
        if lower[index] > upper[index]:
            raise ValueError(f"{name} is too narrow for target margin {margin}")
    return lower, upper


def audit_reset_qpos(
    reset_qpos: Sequence[float],
    joint_names: Sequence[str] = ACTUATOR_JOINT_ORDER,
    *,
    noise_applied: bool,
    reset_noise_margin_rad: float = RESET_NOISE_MARGIN_RAD,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Audit exact-home or noisy-reset qpos before the first physics step."""

    names = tuple(joint_names)
    values = np.asarray(reset_qpos, dtype=np.float64)
    if len(names) != 14 or set(names) != set(ACTUATOR_JOINT_ORDER):
        raise ValueError("reset audit requires the exact 14 actuator names")
    if values.shape != (14,) or not np.all(np.isfinite(values)):
        raise ValueError("reset qpos must be one finite 14-axis vector")
    margin = float(reset_noise_margin_rad)
    if not np.isfinite(margin) or margin != RESET_NOISE_MARGIN_RAD:
        raise ValueError("reset noise margin does not match the frozen contract")
    head_indices = np.asarray(
        [index for index, name in enumerate(names) if name in HEAD_JOINTS]
    )
    leg_indices = np.asarray(
        [index for index, name in enumerate(names) if name not in HEAD_JOINTS]
    )
    lower = np.asarray([SAFE_JOINT_LIMITS[names[index]][0] for index in leg_indices])
    upper = np.asarray([SAFE_JOINT_LIMITS[names[index]][1] for index in leg_indices])
    leg_values = values[leg_indices]
    physical_excess = _safe_excess(leg_values, lower, upper)
    noise_margin_excess = (
        _safe_excess(leg_values, lower + margin, upper - margin)
        if noise_applied
        else np.zeros_like(leg_values)
    )
    expected_home = np.asarray([SAFE_INIT_POS[name] for name in names])
    exact_home_error = np.abs(values - expected_home)
    physical_violations = int(np.count_nonzero(physical_excess > tolerance))
    noise_margin_violations = int(
        np.count_nonzero(noise_margin_excess > tolerance)
    )
    head_peak = float(np.max(np.abs(values[head_indices])))
    exact_home_passed = bool(
        noise_applied or np.max(exact_home_error) <= tolerance
    )
    passed = bool(
        physical_violations == 0
        and noise_margin_violations == 0
        and head_peak <= tolerance
        and exact_home_passed
    )
    return {
        "passed": passed,
        "noise_applied": bool(noise_applied),
        "reset_noise_margin_rad": margin,
        "applied_inward_margin_rad": margin if noise_applied else 0.0,
        "exact_safe_init_required": not bool(noise_applied),
        "exact_safe_init_passed": exact_home_passed,
        "maximum_exact_safe_init_error_rad": float(np.max(exact_home_error)),
        "head_qpos_peak_rad": head_peak,
        "physical_safe_limit_violations": physical_violations,
        "noise_margin_violations": noise_margin_violations,
        "maximum_physical_safe_excess_rad": float(np.max(physical_excess)),
        "maximum_noise_margin_excess_rad": float(np.max(noise_margin_excess)),
        "reset_qpos_rad": {
            name: float(values[index]) for index, name in enumerate(names)
        },
    }


def audit_control_first_startup(
    reset_targets: Sequence[float],
    desired_targets: Sequence[float],
    applied_targets: Sequence[float],
    joint_names: Sequence[str] = ACTUATOR_JOINT_ORDER,
    *,
    control_dt: float,
    leg_target_margin_rad: float = LEG_TARGET_MARGIN_RAD,
    target_slew_limit_rad_per_s: float = TARGET_SLEW_LIMIT_RAD_PER_S,
    physics_steps_before_control: int = 0,
    guard_calls_before_control: int = 0,
    guard_calls_for_first_tick: int = 1,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Audit the first routed policy control applied before the first ``mj_step``.

    The reset target remains the exact/physically-SAFE reset state.  This audit
    independently reconstructs the first policy/profile target after the
    inward desired envelope and exactly one control-tick slew.  A home-only
    precharge followed by a second policy guard call cannot satisfy the gate.
    """

    names = tuple(joint_names)
    if len(names) != 14 or set(names) != set(ACTUATOR_JOINT_ORDER):
        raise ValueError("startup audit requires the exact 14 actuator names")
    vectors = {
        "reset_targets": np.asarray(reset_targets, dtype=np.float64),
        "desired_targets": np.asarray(desired_targets, dtype=np.float64),
        "applied_targets": np.asarray(applied_targets, dtype=np.float64),
    }
    for label, values in vectors.items():
        if values.shape != (14,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{label} must be one finite 14-axis vector")
    try:
        dt = float(control_dt)
        margin = float(leg_target_margin_rad)
        slew_limit = float(target_slew_limit_rad_per_s)
        physics_steps = int(physics_steps_before_control)
        guard_calls_before = int(guard_calls_before_control)
        guard_calls_first_tick = int(guard_calls_for_first_tick)
    except (TypeError, ValueError) as exc:
        raise ValueError("startup audit scalars must be finite numeric values") from exc
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("startup control dt must be finite and positive")
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("startup target margin must be finite and non-negative")
    if not np.isfinite(slew_limit) or slew_limit <= 0.0:
        raise ValueError("startup target slew limit must be finite and positive")
    if physics_steps != physics_steps_before_control or physics_steps < 0:
        raise ValueError("physics_steps_before_control must be a non-negative integer")
    if guard_calls_before != guard_calls_before_control or guard_calls_before < 0:
        raise ValueError("guard_calls_before_control must be a non-negative integer")
    if (
        guard_calls_first_tick != guard_calls_for_first_tick
        or guard_calls_first_tick < 0
    ):
        raise ValueError("guard_calls_for_first_tick must be a non-negative integer")

    reset = vectors["reset_targets"]
    requested_desired = vectors["desired_targets"]
    applied = vectors["applied_targets"]
    physical_lower, physical_upper = build_target_envelope(
        names, leg_margin_rad=0.0
    )
    target_lower, target_upper = build_target_envelope(
        names, leg_margin_rad=margin
    )
    head_indices = np.asarray(
        [index for index, name in enumerate(names) if name in HEAD_JOINTS]
    )
    leg_indices = np.asarray(
        [index for index, name in enumerate(names) if name not in HEAD_JOINTS]
    )

    desired = np.clip(requested_desired, target_lower, target_upper)
    desired[head_indices] = 0.0
    maximum_delta = slew_limit * dt
    expected = reset.copy()
    expected[leg_indices] += np.clip(
        desired[leg_indices] - reset[leg_indices],
        -maximum_delta,
        maximum_delta,
    )
    expected = np.clip(expected, physical_lower, physical_upper)
    expected[head_indices] = 0.0

    applied_error = np.abs(applied - expected)
    reset_physical_excess = _safe_excess(reset, physical_lower, physical_upper)
    applied_physical_excess = _safe_excess(applied, physical_lower, physical_upper)
    desired_margin_excess = _safe_excess(desired, target_lower, target_upper)
    applied_delta = np.abs(applied - reset)
    slew_excess = np.maximum(applied_delta[leg_indices] - maximum_delta, 0.0)
    head_peak = float(np.max(np.abs(applied[head_indices])))
    passed = bool(
        physics_steps == 0
        and guard_calls_before == 0
        and guard_calls_first_tick == 1
        and np.max(applied_error) <= tolerance
        and np.max(reset_physical_excess) <= tolerance
        and np.max(applied_physical_excess) <= tolerance
        and np.max(desired_margin_excess) <= tolerance
        and np.max(slew_excess) <= tolerance
        and head_peak <= tolerance
    )
    return {
        "passed": passed,
        "mode": "control_first",
        "control_applied_before_first_physics_step": physics_steps == 0,
        "physics_steps_before_control": physics_steps,
        "guard_calls_before_control": guard_calls_before,
        "guard_calls_for_first_tick": guard_calls_first_tick,
        "exactly_one_guard_call_for_first_tick": guard_calls_first_tick == 1,
        "home_only_precharge_used": False,
        "required_tick_order": [
            "observe",
            "route",
            "policy_or_profile",
            "guard_once",
            "apply_ctrl",
            "physics",
            "post_step_metrics_and_audit",
        ],
        "control_dt_seconds": dt,
        "leg_target_margin_rad": margin,
        "target_slew_limit_rad_per_s": slew_limit,
        "maximum_target_delta_per_tick_rad": maximum_delta,
        "guarded_output_matches_reconstructed_step": bool(
            np.max(applied_error) <= tolerance
        ),
        "maximum_guarded_output_error_rad": float(np.max(applied_error)),
        "maximum_applied_target_delta_rad": float(
            np.max(applied_delta[leg_indices])
        ),
        "maximum_target_slew_excess_rad": float(np.max(slew_excess)),
        "reset_target_physical_safe_violations": int(
            np.count_nonzero(reset_physical_excess > tolerance)
        ),
        "desired_target_margin_violations": int(
            np.count_nonzero(desired_margin_excess > tolerance)
        ),
        "applied_target_physical_safe_violations": int(
            np.count_nonzero(applied_physical_excess > tolerance)
        ),
        "head_target_peak_rad": head_peak,
        "reset_targets_rad": {
            name: float(reset[index]) for index, name in enumerate(names)
        },
        "policy_preclip_desired_targets_rad": {
            name: float(requested_desired[index]) for index, name in enumerate(names)
        },
        "margin_clipped_desired_targets_rad": {
            name: float(desired[index]) for index, name in enumerate(names)
        },
        "applied_targets_rad": {
            name: float(applied[index]) for index, name in enumerate(names)
        },
    }


class PhysicsSubstepAudit:
    """Accumulate state, fall, and foot-contact facts after every ``mj_step``."""

    def __init__(
        self,
        joint_names: Sequence[str] = ACTUATOR_JOINT_ORDER,
        *,
        minimum_height_m: float = 0.12,
        minimum_upright: float = 0.65,
        tolerance: float = 1e-9,
    ) -> None:
        self.joint_names = tuple(joint_names)
        if len(self.joint_names) != 14 or set(self.joint_names) != set(
            ACTUATOR_JOINT_ORDER
        ):
            raise ValueError("PhysicsSubstepAudit requires the exact 14 actuator names")
        self.head_indices = np.asarray(
            [
                index
                for index, name in enumerate(self.joint_names)
                if name in HEAD_JOINTS
            ],
            dtype=np.int64,
        )
        self.leg_indices = np.asarray(
            [
                index
                for index, name in enumerate(self.joint_names)
                if name not in HEAD_JOINTS
            ],
            dtype=np.int64,
        )
        self.leg_lower = np.asarray(
            [SAFE_JOINT_LIMITS[self.joint_names[index]][0] for index in self.leg_indices]
        )
        self.leg_upper = np.asarray(
            [SAFE_JOINT_LIMITS[self.joint_names[index]][1] for index in self.leg_indices]
        )
        self.minimum_height_limit_m = float(minimum_height_m)
        self.minimum_upright_limit = float(minimum_upright)
        self.tolerance = float(tolerance)
        if (
            not np.isfinite(self.minimum_height_limit_m)
            or not np.isfinite(self.minimum_upright_limit)
            or not np.isfinite(self.tolerance)
            or self.tolerance < 0.0
        ):
            raise ValueError("substep audit thresholds must be finite")
        self.sample_count = 0
        self.leg_joint_sample_count = 0
        self.contact_sample_count = 0
        self.single_support_count = 0
        self.flight_count = 0
        self.qpos_limit_violations = 0
        self.nonfinite_leg_qpos_samples = 0
        self.nonfinite_state_samples = 0
        self.nonfinite_full_qpos_samples = 0
        self.nonfinite_full_qvel_samples = 0
        self.nonfinite_pose_samples = 0
        self.height_fall_samples = 0
        self.upright_fall_samples = 0
        self.maximum_qpos_excess_rad = 0.0
        self.head_qpos_peak_rad = 0.0
        self.minimum_height_m = float("inf")
        self.minimum_upright = float("inf")
        self.first_termination_sample: int | None = None
        self.joint_qpos_min = np.full(14, np.inf, dtype=np.float64)
        self.joint_qpos_max = np.full(14, -np.inf, dtype=np.float64)

    @property
    def termination_required(self) -> bool:
        return bool(
            self.nonfinite_state_samples
            or self.height_fall_samples
            or self.upright_fall_samples
        )

    def update(
        self,
        *,
        joint_qpos: Sequence[float],
        full_qpos: Sequence[float],
        full_qvel: Sequence[float],
        height_m: float,
        upright: float,
        feet_contacts: Sequence[bool | int],
    ) -> None:
        values = np.asarray(joint_qpos, dtype=np.float64)
        all_qpos = np.asarray(full_qpos, dtype=np.float64)
        all_qvel = np.asarray(full_qvel, dtype=np.float64)
        contact_values = np.asarray(feet_contacts)
        if values.shape != (14,):
            raise ValueError("substep joint_qpos must be one 14-axis vector")
        if (
            contact_values.shape != (2,)
            or not np.all(np.isfinite(contact_values.astype(np.float64)))
            or not np.all(np.isin(contact_values, (0, 1, False, True)))
        ):
            raise ValueError("substep feet_contacts must be two finite boolean values")
        contacts = contact_values.astype(bool)
        height_value = float(height_m)
        upright_value = float(upright)
        self.sample_count += 1
        self.leg_joint_sample_count += len(self.leg_indices)
        self.contact_sample_count += 1
        self.single_support_count += int(bool(contacts[0]) ^ bool(contacts[1]))
        self.flight_count += int(not bool(contacts[0]) and not bool(contacts[1]))

        finite_joint = np.isfinite(values)
        self.joint_qpos_min[finite_joint] = np.minimum(
            self.joint_qpos_min[finite_joint], values[finite_joint]
        )
        self.joint_qpos_max[finite_joint] = np.maximum(
            self.joint_qpos_max[finite_joint], values[finite_joint]
        )
        finite_leg = finite_joint[self.leg_indices]
        leg_values = values[self.leg_indices]
        self.nonfinite_leg_qpos_samples += int(np.count_nonzero(~finite_leg))
        finite_excess = np.zeros_like(leg_values)
        finite_excess[finite_leg] = _safe_excess(
            leg_values[finite_leg],
            self.leg_lower[finite_leg],
            self.leg_upper[finite_leg],
        )
        self.qpos_limit_violations += int(
            np.count_nonzero(finite_excess > self.tolerance)
        )
        self.maximum_qpos_excess_rad = max(
            self.maximum_qpos_excess_rad,
            float(np.max(finite_excess, initial=0.0)),
        )
        finite_head = finite_joint[self.head_indices]
        if np.any(finite_head):
            self.head_qpos_peak_rad = max(
                self.head_qpos_peak_rad,
                float(
                    np.max(
                        np.abs(values[self.head_indices][finite_head]),
                        initial=0.0,
                    )
                ),
            )

        qpos_finite = bool(np.all(np.isfinite(all_qpos)))
        qvel_finite = bool(np.all(np.isfinite(all_qvel)))
        pose_finite = bool(np.isfinite(height_value) and np.isfinite(upright_value))
        self.nonfinite_full_qpos_samples += int(not qpos_finite)
        self.nonfinite_full_qvel_samples += int(not qvel_finite)
        self.nonfinite_pose_samples += int(not pose_finite)
        state_nonfinite = not (qpos_finite and qvel_finite and pose_finite)
        self.nonfinite_state_samples += int(state_nonfinite)
        if np.isfinite(height_value):
            self.minimum_height_m = min(self.minimum_height_m, height_value)
            self.height_fall_samples += int(
                height_value < self.minimum_height_limit_m
            )
        if np.isfinite(upright_value):
            self.minimum_upright = min(self.minimum_upright, upright_value)
            self.upright_fall_samples += int(
                upright_value < self.minimum_upright_limit
            )
        if self.termination_required and self.first_termination_sample is None:
            self.first_termination_sample = self.sample_count

    def to_dict(self) -> dict[str, Any]:
        def finite_or_none(value: float) -> float | None:
            return float(value) if np.isfinite(value) else None

        return {
            "sample_count": self.sample_count,
            "leg_joint_sample_count": self.leg_joint_sample_count,
            "contact_sample_count": self.contact_sample_count,
            "contact_sample_count_matches_sample_count": (
                self.contact_sample_count == self.sample_count
            ),
            "single_support_count": self.single_support_count,
            "flight_count": self.flight_count,
            "single_support_rate": (
                self.single_support_count / self.contact_sample_count
                if self.contact_sample_count
                else 0.0
            ),
            "flight_rate": (
                self.flight_count / self.contact_sample_count
                if self.contact_sample_count
                else 0.0
            ),
            "contact_sampling_stage": "immediately_after_each_mj_step",
            "qpos_limit_violations": self.qpos_limit_violations,
            "maximum_qpos_excess_rad": self.maximum_qpos_excess_rad,
            "nonfinite_leg_qpos_samples": self.nonfinite_leg_qpos_samples,
            "nonfinite_state_samples": self.nonfinite_state_samples,
            "nonfinite_full_qpos_samples": self.nonfinite_full_qpos_samples,
            "nonfinite_full_qvel_samples": self.nonfinite_full_qvel_samples,
            "nonfinite_pose_samples": self.nonfinite_pose_samples,
            "height_fall_samples": self.height_fall_samples,
            "upright_fall_samples": self.upright_fall_samples,
            "fall_or_nonfinite_detected": self.termination_required,
            "first_termination_sample": self.first_termination_sample,
            "minimum_height_limit_m": self.minimum_height_limit_m,
            "minimum_upright_limit": self.minimum_upright_limit,
            "minimum_height_m": finite_or_none(self.minimum_height_m),
            "minimum_upright": finite_or_none(self.minimum_upright),
            "head_qpos_peak_rad": self.head_qpos_peak_rad,
            "joint_qpos_min_rad": {
                name: finite_or_none(self.joint_qpos_min[index])
                for index, name in enumerate(self.joint_names)
            },
            "joint_qpos_max_rad": {
                name: finite_or_none(self.joint_qpos_max[index])
                for index, name in enumerate(self.joint_names)
            },
        }


class SafetyAudit:
    """Accumulate separate policy, target, and observed-position safety facts."""

    def __init__(
        self,
        joint_names: Sequence[str] = ACTUATOR_JOINT_ORDER,
        *,
        leg_target_margin_rad: float = LEG_TARGET_MARGIN_RAD,
        target_slew_limit_rad_per_s: float = TARGET_SLEW_LIMIT_RAD_PER_S,
    ):
        self.joint_names = tuple(joint_names)
        if len(self.joint_names) != 14 or set(self.joint_names) != set(
            ACTUATOR_JOINT_ORDER
        ):
            raise ValueError("SafetyAudit requires the exact 14 actuator names")
        self.head_indices = np.asarray(
            [index for index, name in enumerate(self.joint_names) if name in HEAD_JOINTS],
            dtype=np.int64,
        )
        self.leg_indices = np.asarray(
            [index for index, name in enumerate(self.joint_names) if name not in HEAD_JOINTS],
            dtype=np.int64,
        )
        self.leg_lower = np.asarray(
            [SAFE_JOINT_LIMITS[self.joint_names[index]][0] for index in self.leg_indices]
        )
        self.leg_upper = np.asarray(
            [SAFE_JOINT_LIMITS[self.joint_names[index]][1] for index in self.leg_indices]
        )
        target_lower, target_upper = build_target_envelope(
            self.joint_names, leg_margin_rad=leg_target_margin_rad
        )
        self.leg_target_margin_rad = float(leg_target_margin_rad)
        self.target_slew_limit_rad_per_s = float(target_slew_limit_rad_per_s)
        if (
            not np.isfinite(self.target_slew_limit_rad_per_s)
            or self.target_slew_limit_rad_per_s <= 0.0
        ):
            raise ValueError("target slew limit must be finite and positive")
        self.leg_target_lower = target_lower[self.leg_indices]
        self.leg_target_upper = target_upper[self.leg_indices]
        self.sample_count = 0
        self.nonfinite_sample_count = 0
        self.raw_policy_head_action_peak = 0.0
        self.applied_head_action_peak = 0.0
        self.head_target_peak_rad = 0.0
        self.head_qpos_peak_rad = 0.0
        self.preclip_target_limit_violations = 0
        self.applied_target_limit_violations = 0
        self.preclip_target_margin_violations = 0
        self.desired_target_margin_violations = 0
        self.applied_target_margin_violations = 0
        self.unauthorized_applied_target_margin_violations = 0
        self.startup_margin_transition_joint_samples = 0
        self.target_slew_violations = 0
        self.qpos_limit_violations = 0
        self.maximum_preclip_target_excess_rad = 0.0
        self.maximum_applied_target_excess_rad = 0.0
        self.maximum_preclip_target_margin_excess_rad = 0.0
        self.maximum_desired_target_margin_excess_rad = 0.0
        self.maximum_applied_target_margin_excess_rad = 0.0
        self.maximum_target_slew_rate_rad_per_s = 0.0
        self.maximum_qpos_excess_rad = 0.0
        self._target_min = np.full(14, np.inf)
        self._target_max = np.full(14, -np.inf)
        self._qpos_min = np.full(14, np.inf)
        self._qpos_max = np.full(14, -np.inf)

    def update(
        self,
        *,
        raw_policy_action: Sequence[float],
        applied_action: Sequence[float],
        preclip_targets: Sequence[float],
        margin_clipped_targets: Sequence[float],
        applied_targets: Sequence[float],
        previous_applied_targets: Sequence[float],
        joint_qpos: Sequence[float],
        control_dt: float,
        tolerance: float = 1e-9,
    ) -> None:
        arrays = [
            np.asarray(value, dtype=np.float64)
            for value in (
                raw_policy_action,
                applied_action,
                preclip_targets,
                margin_clipped_targets,
                applied_targets,
                previous_applied_targets,
                joint_qpos,
            )
        ]
        if any(value.shape != (14,) for value in arrays):
            raise ValueError("all audited vectors must have shape (14,)")
        if not all(np.all(np.isfinite(value)) for value in arrays):
            self.nonfinite_sample_count += 1
            return
        raw_action, action, raw_target, desired_target, target, previous, qpos = arrays
        dt = float(control_dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("control_dt must be finite and positive")
        self.sample_count += 1

        self.raw_policy_head_action_peak = max(
            self.raw_policy_head_action_peak,
            float(np.max(np.abs(raw_action[self.head_indices]))),
        )
        self.applied_head_action_peak = max(
            self.applied_head_action_peak,
            float(np.max(np.abs(action[self.head_indices]))),
        )
        self.head_target_peak_rad = max(
            self.head_target_peak_rad,
            float(np.max(np.abs(target[self.head_indices]))),
        )
        self.head_qpos_peak_rad = max(
            self.head_qpos_peak_rad,
            float(np.max(np.abs(qpos[self.head_indices]))),
        )

        raw_excess = _safe_excess(
            raw_target[self.leg_indices], self.leg_lower, self.leg_upper
        )
        target_excess = _safe_excess(
            target[self.leg_indices], self.leg_lower, self.leg_upper
        )
        qpos_excess = _safe_excess(
            qpos[self.leg_indices], self.leg_lower, self.leg_upper
        )
        raw_margin_excess = _safe_excess(
            raw_target[self.leg_indices], self.leg_target_lower, self.leg_target_upper
        )
        desired_margin_excess = _safe_excess(
            desired_target[self.leg_indices],
            self.leg_target_lower,
            self.leg_target_upper,
        )
        target_margin_excess = _safe_excess(
            target[self.leg_indices], self.leg_target_lower, self.leg_target_upper
        )
        previous_margin_excess = _safe_excess(
            previous[self.leg_indices], self.leg_target_lower, self.leg_target_upper
        )
        slew_rates = np.abs(
            target[self.leg_indices] - previous[self.leg_indices]
        ) / dt
        slew_violation_mask = (
            slew_rates > self.target_slew_limit_rad_per_s + tolerance
        )
        self.preclip_target_limit_violations += int(np.count_nonzero(raw_excess > tolerance))
        self.applied_target_limit_violations += int(
            np.count_nonzero(target_excess > tolerance)
        )
        self.preclip_target_margin_violations += int(
            np.count_nonzero(raw_margin_excess > tolerance)
        )
        self.desired_target_margin_violations += int(
            np.count_nonzero(desired_margin_excess > tolerance)
        )
        self.applied_target_margin_violations += int(
            np.count_nonzero(target_margin_excess > tolerance)
        )
        startup_mask = target_margin_excess > tolerance
        authorized_startup_mask = (
            startup_mask
            & (desired_margin_excess <= tolerance)
            & (target_excess <= tolerance)
            & (~slew_violation_mask)
            & (target_margin_excess < previous_margin_excess - tolerance)
        )
        self.startup_margin_transition_joint_samples += int(
            np.count_nonzero(authorized_startup_mask)
        )
        self.unauthorized_applied_target_margin_violations += int(
            np.count_nonzero(startup_mask & ~authorized_startup_mask)
        )
        self.target_slew_violations += int(np.count_nonzero(slew_violation_mask))
        self.qpos_limit_violations += int(np.count_nonzero(qpos_excess > tolerance))
        self.maximum_preclip_target_excess_rad = max(
            self.maximum_preclip_target_excess_rad, float(np.max(raw_excess))
        )
        self.maximum_applied_target_excess_rad = max(
            self.maximum_applied_target_excess_rad, float(np.max(target_excess))
        )
        self.maximum_preclip_target_margin_excess_rad = max(
            self.maximum_preclip_target_margin_excess_rad,
            float(np.max(raw_margin_excess)),
        )
        self.maximum_desired_target_margin_excess_rad = max(
            self.maximum_desired_target_margin_excess_rad,
            float(np.max(desired_margin_excess)),
        )
        self.maximum_applied_target_margin_excess_rad = max(
            self.maximum_applied_target_margin_excess_rad,
            float(np.max(target_margin_excess)),
        )
        self.maximum_target_slew_rate_rad_per_s = max(
            self.maximum_target_slew_rate_rad_per_s,
            float(np.max(slew_rates)),
        )
        self.maximum_qpos_excess_rad = max(
            self.maximum_qpos_excess_rad, float(np.max(qpos_excess))
        )
        self._target_min = np.minimum(self._target_min, target)
        self._target_max = np.maximum(self._target_max, target)
        self._qpos_min = np.minimum(self._qpos_min, qpos)
        self._qpos_max = np.maximum(self._qpos_max, qpos)

    def to_dict(self) -> dict[str, Any]:
        def extrema(values: np.ndarray) -> dict[str, float | None]:
            return {
                name: (None if not np.isfinite(values[index]) else float(values[index]))
                for index, name in enumerate(self.joint_names)
            }

        return {
            "sample_count": self.sample_count,
            "nonfinite_sample_count": self.nonfinite_sample_count,
            "raw_policy_head_action_peak": self.raw_policy_head_action_peak,
            "applied_head_action_peak": self.applied_head_action_peak,
            "head_target_peak_rad": self.head_target_peak_rad,
            "head_qpos_peak_rad": self.head_qpos_peak_rad,
            "leg_target_margin_rad": self.leg_target_margin_rad,
            "target_slew_limit_rad_per_s": self.target_slew_limit_rad_per_s,
            "preclip_target_limit_violations": self.preclip_target_limit_violations,
            "applied_target_limit_violations": self.applied_target_limit_violations,
            "preclip_target_margin_violations": self.preclip_target_margin_violations,
            "desired_target_margin_violations": self.desired_target_margin_violations,
            "applied_target_margin_violations": self.applied_target_margin_violations,
            "unauthorized_applied_target_margin_violations": (
                self.unauthorized_applied_target_margin_violations
            ),
            "startup_margin_transition_joint_samples": (
                self.startup_margin_transition_joint_samples
            ),
            "target_slew_violations": self.target_slew_violations,
            "qpos_limit_violations": self.qpos_limit_violations,
            "maximum_preclip_target_excess_rad": self.maximum_preclip_target_excess_rad,
            "maximum_applied_target_excess_rad": self.maximum_applied_target_excess_rad,
            "maximum_preclip_target_margin_excess_rad": (
                self.maximum_preclip_target_margin_excess_rad
            ),
            "maximum_desired_target_margin_excess_rad": (
                self.maximum_desired_target_margin_excess_rad
            ),
            "maximum_applied_target_margin_excess_rad": (
                self.maximum_applied_target_margin_excess_rad
            ),
            "maximum_qpos_excess_rad": self.maximum_qpos_excess_rad,
            "maximum_target_slew_rate_rad_per_s": (
                self.maximum_target_slew_rate_rad_per_s
            ),
            "applied_target_min_rad": extrema(self._target_min),
            "applied_target_max_rad": extrema(self._target_max),
            "joint_qpos_min_rad": extrema(self._qpos_min),
            "joint_qpos_max_rad": extrema(self._qpos_max),
        }


def compute_motion_metrics(
    command: Sequence[float],
    local_velocity_xyz: Sequence[Sequence[float]],
    local_yaw_rate: Sequence[float],
    *,
    displacement_xyz: Sequence[float],
    minimum_height_m: float,
    minimum_upright: float,
    mean_effective_command: Sequence[float],
    mean_policy_observation_command: Sequence[float] | None = None,
    single_support_rate: float,
    flight_rate: float,
    contact_sample_count: int | None = None,
    contact_rate_sample_source: str = "caller_supplied",
    diagnostic_control_endpoint_single_support_rate: float | None = None,
    diagnostic_control_endpoint_flight_rate: float | None = None,
    diagnostic_control_endpoint_contact_sample_count: int | None = None,
) -> dict[str, Any]:
    """Compute signed tracking and orthogonal metrics for one segment."""

    command_array = np.asarray(command, dtype=np.float64)
    velocity = np.asarray(local_velocity_xyz, dtype=np.float64)
    yaw_rate = np.asarray(local_yaw_rate, dtype=np.float64)
    displacement = np.asarray(displacement_xyz, dtype=np.float64)
    effective = np.asarray(mean_effective_command, dtype=np.float64)
    policy_observation = np.asarray(
        effective
        if mean_policy_observation_command is None
        else mean_policy_observation_command,
        dtype=np.float64,
    )
    if (
        command_array.shape != (3,)
        or displacement.shape != (3,)
        or effective.shape != (3,)
        or policy_observation.shape != (3,)
    ):
        raise ValueError(
            "command, displacement, effective command, and policy observation "
            "command must be triplets"
        )
    if velocity.ndim != 2 or velocity.shape[1] != 3 or len(velocity) == 0:
        raise ValueError("local_velocity_xyz must contain at least one triplet")
    if yaw_rate.shape != (len(velocity),):
        raise ValueError("yaw-rate and velocity sample counts must match")
    if not all(
        np.all(np.isfinite(value))
        for value in (
            command_array,
            velocity,
            yaw_rate,
            displacement,
            effective,
            policy_observation,
        )
    ):
        raise ValueError("motion samples must be finite")
    support_rate = float(single_support_rate)
    airborne_rate = float(flight_rate)
    if (
        not np.isfinite(support_rate)
        or not 0.0 <= support_rate <= 1.0
        or not np.isfinite(airborne_rate)
        or not 0.0 <= airborne_rate <= 1.0
    ):
        raise ValueError("contact rates must be finite and in [0, 1]")
    if contact_sample_count is not None and int(contact_sample_count) < 0:
        raise ValueError("contact sample count must be non-negative")
    diagnostic_rates = (
        diagnostic_control_endpoint_single_support_rate,
        diagnostic_control_endpoint_flight_rate,
    )
    if any(
        value is not None
        and (not np.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0)
        for value in diagnostic_rates
    ):
        raise ValueError("diagnostic control-endpoint contact rates are invalid")

    mean_velocity = np.mean(velocity, axis=0)
    mean_yaw = float(np.mean(yaw_rate))
    linear_command = command_array[:2]
    linear_speed = float(np.linalg.norm(linear_command))
    if linear_speed > 0.0:
        direction = linear_command / linear_speed
        orthogonal_direction = np.asarray((-direction[1], direction[0]))
        primary_velocity = float(mean_velocity[:2] @ direction)
        orthogonal_velocity = float(mean_velocity[:2] @ orthogonal_direction)
        primary_error = abs(primary_velocity - linear_speed)
        orthogonal_abs = abs(orthogonal_velocity)
    else:
        primary_velocity = 0.0
        primary_error = 0.0
        orthogonal_velocity = float(np.linalg.norm(mean_velocity[:2]))
        orthogonal_abs = orthogonal_velocity

    return {
        "sample_count": int(len(velocity)),
        "command": command_array.tolist(),
        "physical_command": command_array.tolist(),
        "mean_effective_command": effective.tolist(),
        "mean_policy_observation_command": policy_observation.tolist(),
        "mean_local_velocity_xyz": mean_velocity.tolist(),
        "mean_local_yaw_rate": mean_yaw,
        "commanded_linear_speed": linear_speed,
        "projected_primary_velocity": primary_velocity,
        "primary_velocity_error": float(primary_error),
        "signed_orthogonal_velocity": orthogonal_velocity,
        "absolute_orthogonal_velocity": orthogonal_abs,
        "yaw_rate_error": abs(mean_yaw - float(command_array[2])),
        "uncommanded_yaw_rate": abs(mean_yaw) if command_array[2] == 0.0 else 0.0,
        "displacement_xyz": displacement.tolist(),
        "planar_displacement": float(np.linalg.norm(displacement[:2])),
        "minimum_height_m": float(minimum_height_m),
        "minimum_upright": float(minimum_upright),
        "single_support_rate": support_rate,
        "flight_rate": airborne_rate,
        "contact_sample_count": (
            None if contact_sample_count is None else int(contact_sample_count)
        ),
        "contact_rate_sample_source": str(contact_rate_sample_source),
        "diagnostic_control_endpoint_single_support_rate": (
            None
            if diagnostic_control_endpoint_single_support_rate is None
            else float(diagnostic_control_endpoint_single_support_rate)
        ),
        "diagnostic_control_endpoint_flight_rate": (
            None
            if diagnostic_control_endpoint_flight_rate is None
            else float(diagnostic_control_endpoint_flight_rate)
        ),
        "diagnostic_control_endpoint_contact_sample_count": (
            None
            if diagnostic_control_endpoint_contact_sample_count is None
            else int(diagnostic_control_endpoint_contact_sample_count)
        ),
    }


def summarize_backward_exit_recovery_steps(
    step_audits: Sequence[Mapping[str, Any]],
    *,
    enabled: bool,
    expected_sample_count: int,
    diagnostic_only: bool = False,
) -> dict[str, Any]:
    """Summarize one segment's Stage-A recovery composition ticks."""

    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("backward-exit recovery enabled must be boolean")
    if not isinstance(diagnostic_only, (bool, np.bool_)):
        raise ValueError("backward-exit recovery diagnostic_only must be boolean")
    if isinstance(expected_sample_count, bool) or expected_sample_count < 0:
        raise ValueError("expected recovery sample count must be non-negative")
    steps = list(step_audits)
    if not all(isinstance(step, Mapping) for step in steps):
        raise ValueError("recovery step audits must be mappings")

    control_ticks = [step.get("control_tick") for step in steps]
    control_ticks_valid = all(
        isinstance(tick, int) and not isinstance(tick, bool) and tick >= 0
        for tick in control_ticks
    )
    control_ticks_consecutive = bool(
        not control_ticks
        or (
            control_ticks_valid
            and control_ticks
            == list(range(int(control_ticks[0]), int(control_ticks[0]) + len(steps)))
        )
    )
    enabled_values_match = all(step.get("enabled") is bool(enabled) for step in steps)
    exit_steps = [step for step in steps if step.get("exit_event") is True]
    active_steps = [step for step in steps if step.get("recovery_active") is True]
    cancel_steps = [step for step in steps if step.get("reentry_cancelled") is True]
    cap_violations = [step for step in steps if step.get("cap_violation") is True]

    exit_activation_valid = all(
        step.get("recovery_active") is True
        and step.get("backward_feedforward_active") is False
        and step.get("recovery_tick_index") == 1
        for step in exit_steps
    )
    active_tick_indices_valid = all(
        isinstance(step.get("recovery_tick_index"), int)
        and not isinstance(step.get("recovery_tick_index"), bool)
        and 1 <= int(step["recovery_tick_index"]) <= BACKWARD_EXIT_RECOVERY_HOLD_TICKS
        for step in active_steps
    )
    cap_composition_valid = all(
        step.get("cap_upper_target_rad")
        == BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD
        and _finite_number(
            step.get("composed_left_knee_target_rad"),
            "backward_exit_recovery.composed_left_knee_target_rad",
        )
        <= BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD
        and _finite_number(
            step.get("cap_excess_rad"),
            "backward_exit_recovery.cap_excess_rad",
        )
        == 0.0
        and step.get("cap_violation") is False
        for step in active_steps
    )
    inactive_release_valid = all(
        step.get("recovery_tick_index") is None
        and step.get("cap_upper_target_rad") is None
        and _finite_number(
            step.get("cap_excess_rad"),
            "backward_exit_recovery.inactive_cap_excess_rad",
        )
        == 0.0
        and step.get("cap_violation") is False
        for step in steps
        if step.get("recovery_active") is not True
    )
    cancellations_valid = all(
        step.get("backward_feedforward_active") is True
        and step.get("recovery_active") is False
        and int(step.get("remaining_ticks_after_step", -1)) == 0
        for step in cancel_steps
    )
    disabled_path_inert = bool(
        enabled
        or (
            not exit_steps
            and not active_steps
            and not cancel_steps
            and not cap_violations
        )
    )
    maximum_composed = (
        None
        if not active_steps
        else max(
            float(step["composed_left_knee_target_rad"])
            for step in active_steps
        )
    )
    checks = {
        "sample_count_matches_completed_control_steps": (
            len(steps) == expected_sample_count
        ),
        "enabled_value_matches_runtime": enabled_values_match,
        "control_ticks_consecutive": control_ticks_consecutive,
        "exit_tick_is_first_active_tick": exit_activation_valid,
        "active_tick_indices_within_frozen_hold": active_tick_indices_valid,
        "active_cap_composition_valid": cap_composition_valid,
        "inactive_ticks_release_cap_immediately": inactive_release_valid,
        "backward_reentry_cancels_remaining_ticks": cancellations_valid,
        "cap_violation_count_zero": len(cap_violations) == 0,
        "disabled_path_inert": disabled_path_inert,
        "final_guard_called_exactly_once_per_sample": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    return {
        "enabled": bool(enabled),
        "status": (
            "DIAGNOSTIC_UNADOPTED"
            if diagnostic_only
            else FORMAL_CANDIDATE_STATUS
        ),
        "formal_candidate_only": False,
        "adopted_simulation_only": not bool(diagnostic_only),
        "diagnostic_unadopted_only": bool(diagnostic_only),
        "sample_count": len(steps),
        "expected_sample_count": int(expected_sample_count),
        "final_guard_call_count": len(steps),
        "exit_event_count": len(exit_steps),
        "exit_event_control_ticks": [int(step["control_tick"]) for step in exit_steps],
        "active_tick_count": len(active_steps),
        "active_control_ticks": [int(step["control_tick"]) for step in active_steps],
        "reentry_cancel_count": len(cancel_steps),
        "reentry_cancel_control_ticks": [
            int(step["control_tick"]) for step in cancel_steps
        ],
        "cap_upper_target_rad": BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD,
        "maximum_composed_left_knee_target_rad": maximum_composed,
        "cap_violation_count": len(cap_violations),
        "remaining_ticks_after_segment": (
            0 if not steps else int(steps[-1]["remaining_ticks_after_step"])
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }


def backward_exit_recovery_state_acceptance(
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an episode-level :class:`BackwardExitRecovery` state audit."""

    if not isinstance(audit, Mapping):
        return {"passed": False, "checks": {"mapping": False}}
    enabled = audit.get("enabled")
    events = audit.get("events")
    events_list = list(events) if isinstance(events, list) else []
    event_counts_valid = bool(
        isinstance(events, list)
        and int(audit.get("exit_event_count", -1)) == len(events_list)
        and int(audit.get("active_tick_count", -1))
        == sum(int(event.get("active_tick_count", -1)) for event in events_list)
        and int(audit.get("completed_event_count", -1))
        == sum(event.get("status") == "COMPLETED" for event in events_list)
        and int(audit.get("reentry_cancel_count", -1))
        == sum(
            event.get("status") == "CANCELLED_BY_BACKWARD_REENTRY"
            for event in events_list
        )
    )
    event_lifecycles_valid = bool(
        all(
            isinstance(event.get("start_control_tick"), int)
            and not isinstance(event.get("start_control_tick"), bool)
            and event.get("cap_upper_target_rad")
            == BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD
            and (
                (
                    event.get("status") == "COMPLETED"
                    and int(event.get("active_tick_count", -1))
                    == BACKWARD_EXIT_RECOVERY_HOLD_TICKS
                    and int(event.get("end_control_tick_exclusive", -1))
                    == int(event["start_control_tick"])
                    + BACKWARD_EXIT_RECOVERY_HOLD_TICKS
                )
                or (
                    event.get("status") == "CANCELLED_BY_BACKWARD_REENTRY"
                    and 1
                    <= int(event.get("active_tick_count", -1))
                    < BACKWARD_EXIT_RECOVERY_HOLD_TICKS
                    and int(event.get("cancel_control_tick", -1))
                    == int(event["start_control_tick"])
                    + int(event["active_tick_count"])
                )
            )
            for event in events_list
        )
    )
    disabled_path_inert = bool(
        (
            enabled is False
            and audit.get("disabled_path_inert") is True
            and len(events_list) == 0
            and int(audit.get("exit_event_count", -1)) == 0
            and int(audit.get("active_tick_count", -1)) == 0
            and int(audit.get("reentry_cancel_count", -1)) == 0
            and int(audit.get("remaining_ticks", -1)) == 0
        )
        or enabled is True
    )
    active_count = int(audit.get("active_tick_count", -1))
    maximum_composed = audit.get("maximum_composed_left_knee_target_rad")
    maximum_valid = bool(
        (active_count == 0 and maximum_composed is None)
        or (
            active_count > 0
            and isinstance(maximum_composed, (int, float))
            and not isinstance(maximum_composed, bool)
            and np.isfinite(float(maximum_composed))
            and float(maximum_composed)
            <= BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD
        )
    )
    execution_mode_markers_valid = bool(
        (
            audit.get("formal_candidate_only") is False
            and audit.get("diagnostic_unadopted_only") is False
            and audit.get("adopted_simulation_only") is True
        )
        or (
            audit.get("formal_candidate_only") is False
            and audit.get("diagnostic_unadopted_only") is True
            and audit.get("adopted_simulation_only") in (None, False)
        )
    )
    checks = {
        "mapping": True,
        "enabled_is_boolean": isinstance(enabled, bool),
        "execution_mode_markers_exclusive": execution_mode_markers_valid,
        "runtime_contract_exact": bool(
            isinstance(audit.get("contract"), Mapping)
            and dict(audit["contract"]) == backward_exit_recovery_contract()
        ),
        "reset_count_at_least_one": int(audit.get("reset_count", -1)) >= 1,
        "reset_clear_on_schedule_start": (
            audit.get("reset_clear_on_schedule_start") is True
        ),
        "composition_before_final_guard": (
            audit.get("composition_before_final_guard") is True
        ),
        "state_machine_reported_pass": audit.get("passed") is True,
        "event_counts_consistent": event_counts_valid,
        "event_lifecycles_complete_or_cancelled": event_lifecycles_valid,
        "remaining_recovery_ticks_zero": int(audit.get("remaining_ticks", -1)) == 0,
        "cap_violation_count_zero": int(audit.get("cap_violation_count", -1)) == 0,
        "maximum_composed_target_within_cap": maximum_valid,
        "disabled_path_inert": disabled_path_inert,
        "final_guard_exactly_once_per_control_tick": (
            int(audit.get("final_guard_calls_per_control_tick", -1)) == 1
            and int(audit.get("final_guard_call_count", -1))
            == int(audit.get("control_tick_count", -2))
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    return {"passed": all(checks.values()), "checks": checks}


def segment_acceptance(
    segment: Mapping[str, Any],
    thresholds: AcceptanceThresholds = AcceptanceThresholds(),
    *,
    require_gait_quality: bool = False,
) -> dict[str, Any]:
    """Evaluate every mandatory safety and motion check for one segment."""

    command = np.asarray(segment["command"], dtype=np.float64)
    metrics = segment["metrics"]
    audit = segment["safety_audit"]
    substep_audit = segment.get("physics_substep_audit", {})
    substep_sample_count = int(substep_audit.get("sample_count", 0))
    completed_physics_substeps = int(segment.get("completed_physics_substeps", -1))
    expected_physics_substeps = int(segment.get("expected_physics_substeps", -1))
    substep_minimum_height = substep_audit.get("minimum_height_m")
    substep_minimum_upright = substep_audit.get("minimum_upright")
    substep_contact_sample_count = int(
        substep_audit.get("contact_sample_count", 0)
    )
    substep_single_support_rate = float(
        substep_audit.get("single_support_rate", -1.0)
    )
    substep_flight_rate = float(substep_audit.get("flight_rate", -1.0))
    routing = segment["routing"]
    backward_exit_recovery_audit = segment.get(
        "backward_exit_recovery_audit", {}
    )
    expected_expert = segment.get("expected_expert")
    expected_policy_role = segment.get("expected_policy_role")
    steady_steps = int(routing.get("steady_state_steps", 0))
    steady_experts = dict(routing.get("steady_state_routed_expert_steps", {}))
    steady_roles = dict(routing.get("steady_state_policy_role_steps", {}))
    moving_linear = float(np.linalg.norm(command[:2])) > 0.0
    moving_yaw = abs(float(command[2])) > 0.0
    moving = moving_linear or moving_yaw
    gait_quality_required = bool(require_gait_quality)
    gait_quality_metrics = segment.get("gait_quality_metrics")
    gait_quality_result = segment.get("gait_quality_acceptance")
    gait_quality_present = bool(
        isinstance(gait_quality_metrics, Mapping)
        and isinstance(gait_quality_result, Mapping)
    )
    rederived_gait_quality: dict[str, Any] | None = None
    gait_quality_rederivation_error: str | None = None
    if gait_quality_present:
        try:
            rederived_gait_quality = rederive_gait_quality_acceptance(
                gait_quality_metrics
            ).as_dict()
        except (KeyError, TypeError, ValueError, IndexError) as error:
            gait_quality_rederivation_error = f"{type(error).__name__}: {error}"
    gait_quality_acceptance_exact = bool(
        rederived_gait_quality is not None
        and dict(gait_quality_result) == rederived_gait_quality
    )
    def quality_float(value: object, default: float = np.nan) -> float:
        try:
            return float(value)
        except (TypeError, ValueError, OverflowError):
            return default

    def quality_int(value: object, default: int = -1) -> int:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return default

    physics_timestep_s = quality_float(segment.get("physics_timestep_s"))
    completed_seconds = quality_float(segment.get("completed_seconds"))
    requested_seconds = quality_float(segment.get("requested_seconds"))
    expected_quality_duration_s = completed_physics_substeps * physics_timestep_s
    gait_quality_duration_exact = bool(
        gait_quality_present
        and np.isfinite(physics_timestep_s)
        and physics_timestep_s > 0.0
        and np.isfinite(completed_seconds)
        and np.isfinite(requested_seconds)
        and abs(completed_seconds - expected_quality_duration_s) <= 1.0e-12
        and (
            not bool(segment.get("completed"))
            or abs(requested_seconds - expected_quality_duration_s)
            <= physics_timestep_s / 2.0 + 1.0e-12
        )
        and abs(
            quality_float(gait_quality_metrics.get("duration_s"), np.inf)
            - expected_quality_duration_s
        )
        <= physics_timestep_s / 2.0 + 1.0e-12
        and abs(
            quality_float(
                gait_quality_metrics.get("physics_timestep_s"), np.inf
            )
            - physics_timestep_s
        )
        <= 1.0e-12
        and quality_float(
            gait_quality_metrics.get("maximum_timestep_error_s"), np.inf
        )
        <= 1.0e-12
    )
    signed_yaw_progress = float(metrics["mean_local_yaw_rate"]) * float(command[2])
    checks = {
        "completed": bool(segment["completed"]),
        "no_fall": not bool(segment["fell"]),
        "minimum_upright": metrics["minimum_upright"] >= thresholds.minimum_upright,
        "minimum_height": metrics["minimum_height_m"] >= thresholds.minimum_height_m,
        "primary_velocity": (
            not moving_linear
            or metrics["primary_velocity_error"]
            <= thresholds.maximum_primary_velocity_error_mps
        ),
        "signed_linear_progress": (
            not moving_linear
            or metrics["projected_primary_velocity"]
            >= (
                thresholds.minimum_signed_linear_progress_fraction
                * metrics["commanded_linear_speed"]
            )
        ),
        "orthogonal_velocity": (
            metrics["absolute_orthogonal_velocity"]
            <= (
                thresholds.maximum_orthogonal_velocity_mps
                if moving_linear
                else thresholds.maximum_stationary_linear_speed_mps
            )
        ),
        "yaw_rate": (
            metrics["yaw_rate_error"] <= thresholds.maximum_yaw_rate_error_radps
            if moving_yaw
            else metrics["uncommanded_yaw_rate"]
            <= thresholds.maximum_uncommanded_yaw_rate_radps
        ),
        "signed_yaw_progress": (
            not moving_yaw
            or (
                signed_yaw_progress > 0.0
                and abs(float(metrics["mean_local_yaw_rate"]))
                >= (
                    thresholds.minimum_signed_yaw_progress_fraction
                    * abs(float(command[2]))
                )
            )
        ),
        "moving_single_support": (
            not (moving_linear or moving_yaw)
            or substep_single_support_rate
            >= thresholds.minimum_moving_single_support_rate
        ),
        "flight_rate": substep_flight_rate <= thresholds.maximum_flight_rate,
        "contact_rates_from_all_physics_substeps": (
            substep_contact_sample_count > 0
            and substep_contact_sample_count == substep_sample_count
            and bool(
                substep_audit.get(
                    "contact_sample_count_matches_sample_count", False
                )
            )
            and metrics.get("contact_rate_sample_source")
            == "physics_substeps_after_mj_step"
            and int(metrics.get("contact_sample_count", -1))
            == substep_contact_sample_count
            and float(metrics.get("single_support_rate", -1.0))
            == substep_single_support_rate
            and float(metrics.get("flight_rate", -1.0)) == substep_flight_rate
        ),
        "reverse_entry_phase_audit": bool(
            routing.get("reverse_entry_phase", {}).get("passed", False)
        ),
        "backward_exit_recovery_audit": bool(
            isinstance(backward_exit_recovery_audit, Mapping)
            and backward_exit_recovery_audit.get("passed") is True
            and int(
                backward_exit_recovery_audit.get("cap_violation_count", -1)
            )
            == 0
            and int(
                backward_exit_recovery_audit.get("final_guard_call_count", -1)
            )
            == int(backward_exit_recovery_audit.get("sample_count", -2))
            and bool(
                backward_exit_recovery_audit.get("checks", {}).get(
                    "exit_tick_is_first_active_tick", False
                )
            )
            and bool(
                backward_exit_recovery_audit.get("checks", {}).get(
                    "inactive_ticks_release_cap_immediately", False
                )
            )
        ),
        "stop_drift": (
            moving_linear
            or moving_yaw
            or metrics["planar_displacement"] <= thresholds.maximum_stop_drift_m
        ),
        "head_applied_action_locked": (
            audit["applied_head_action_peak"]
            <= thresholds.maximum_head_applied_action
        ),
        "head_target_locked": (
            audit["head_target_peak_rad"] <= thresholds.maximum_head_target_rad
        ),
        "preclip_targets_safe": (
            audit["preclip_target_limit_violations"]
            <= thresholds.maximum_safe_limit_violations
        ),
        "applied_targets_safe": (
            audit["applied_target_limit_violations"]
            <= thresholds.maximum_safe_limit_violations
        ),
        "desired_targets_inside_margin": (
            audit["desired_target_margin_violations"]
            <= thresholds.maximum_safe_limit_violations
        ),
        "startup_margin_transition_authorized": (
            audit["unauthorized_applied_target_margin_violations"]
            <= thresholds.maximum_safe_limit_violations
        ),
        "target_slew_safe": (
            audit["target_slew_violations"]
            <= thresholds.maximum_safe_limit_violations
        ),
        "joint_qpos_safe": (
            audit["qpos_limit_violations"] <= thresholds.maximum_safe_limit_violations
        ),
        "finite": audit["nonfinite_sample_count"] == 0,
        "all_physics_substeps_audited": (
            substep_sample_count > 0
            and substep_sample_count == completed_physics_substeps
        ),
        "completed_physics_substep_count": (
            not bool(segment["completed"])
            or (
                expected_physics_substeps > 0
                and substep_sample_count == expected_physics_substeps
            )
        ),
        "substep_joint_qpos_safe": int(
            substep_audit.get("qpos_limit_violations", 1)
        )
        <= thresholds.maximum_safe_limit_violations,
        "substep_finite": int(substep_audit.get("nonfinite_state_samples", 1))
        == 0,
        "substep_no_fall": (
            int(substep_audit.get("height_fall_samples", 1)) == 0
            and int(substep_audit.get("upright_fall_samples", 1)) == 0
        ),
        "substep_minimum_height": (
            substep_minimum_height is not None
            and float(substep_minimum_height) >= thresholds.minimum_height_m
        ),
        "substep_minimum_upright": (
            substep_minimum_upright is not None
            and float(substep_minimum_upright) >= thresholds.minimum_upright
        ),
        "steady_route_expected_expert": (
            bool(expected_expert)
            and steady_steps > 0
            and steady_experts == {expected_expert: steady_steps}
        ),
        "steady_route_expected_policy_role": (
            bool(expected_policy_role)
            and steady_steps > 0
            and steady_roles == {expected_policy_role: steady_steps}
        ),
        "steady_route_sample_count": steady_steps == int(metrics["sample_count"]),
        "prohibited_experts_absent": (
            int(routing.get("prohibited_expert_steps", 1)) == 0
            and int(routing.get("steady_state_prohibited_expert_steps", 1)) == 0
        ),
        "atomic_endpoint_exact": (
            not bool(routing.get("atomic_endpoint_required", False))
            or int(routing.get("atomic_endpoint_mismatch_steps", 1)) == 0
        ),
        "command_not_clipped": (
            int(routing["command_clip_events"])
            <= thresholds.maximum_command_clip_events
        ),
        "gait_quality_present": (
            not gait_quality_required or gait_quality_present
        ),
        "gait_quality_all_states_sampled": (
            not gait_quality_required
            or (
                gait_quality_present
                and gait_quality_metrics.get("measurement_complete") is True
                and quality_int(gait_quality_metrics.get("sample_count"))
                == substep_sample_count + 1
            )
        ),
        "gait_quality_duration_exact": (
            not gait_quality_required or gait_quality_duration_exact
        ),
        "gait_quality_force_contact_source": (
            not gait_quality_required
            or (
                gait_quality_present
                and gait_quality_metrics.get("contact_state_source")
                == "normal_force_schmitt"
                and quality_int(
                    gait_quality_metrics.get("contact_force_sample_count")
                )
                == substep_sample_count + 1
            )
        ),
        "gait_quality_contact_velocity_source": (
            not gait_quality_required
            or (
                gait_quality_present
                and gait_quality_metrics.get("stance_slip_measurement_source")
                == "force_weighted_contact_point_jacobian"
                and quality_int(
                    gait_quality_metrics.get(
                        "contact_velocity_payload_sample_count"
                    )
                )
                == substep_sample_count + 1
                and quality_int(
                    gait_quality_metrics.get("contact_velocity_sample_count")
                )
                == quality_int(
                    gait_quality_metrics.get(
                        "left_contact_velocity_sample_count"
                    )
                )
                + quality_int(
                    gait_quality_metrics.get(
                        "right_contact_velocity_sample_count"
                    )
                )
            )
        ),
        "gait_quality_trunk_pose_source": (
            not gait_quality_required
            or (
                gait_quality_present
                and gait_quality_metrics.get("trunk_pose_measurement_source")
                == "mujoco_shadow_xpos_xmat_after_mj_forward"
                and quality_int(
                    gait_quality_metrics.get("trunk_yaw_sample_count")
                )
                == substep_sample_count + 1
            )
        ),
        "gait_quality_acceptance_rederived": (
            not gait_quality_required
            or (
                gait_quality_present
                and gait_quality_rederivation_error is None
                and rederived_gait_quality is not None
            )
        ),
        "gait_quality_acceptance_untampered": (
            not gait_quality_required or gait_quality_acceptance_exact
        ),
        "strict_gait_quality": (
            not gait_quality_required
            or (
                gait_quality_acceptance_exact
                and rederived_gait_quality is not None
                and rederived_gait_quality.get("passed") is True
            )
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    failures = [name for name, value in checks.items() if not value]
    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "rederived_gait_quality_acceptance": rederived_gait_quality,
        "gait_quality_rederivation_error": gait_quality_rederivation_error,
    }


def suite_acceptance(
    episodes: Sequence[Mapping[str, Any]],
    expected_segment_names: Sequence[str],
    thresholds: AcceptanceThresholds = AcceptanceThresholds(),
    *,
    require_gait_quality: bool = False,
) -> dict[str, Any]:
    """Require every expected segment in every seed to pass."""

    expected = tuple(expected_segment_names)
    episode_checks = []
    for episode in episodes:
        segments = episode["segments"]
        actual = tuple(segment["name"] for segment in segments)
        segment_checks = [
            {
                "name": segment["name"],
                **segment_acceptance(
                    segment,
                    thresholds,
                    require_gait_quality=require_gait_quality,
                ),
            }
            for segment in segments
        ]
        if "reset_qpos_audits" in episode:
            reset_audits = list(episode["reset_qpos_audits"].values())
        elif "reset_qpos_audit" in episode:
            reset_audits = [episode["reset_qpos_audit"]]
        else:
            reset_audits = []
        reset_qpos_passed = bool(reset_audits) and all(
            bool(audit.get("passed")) for audit in reset_audits
        )
        if "control_first_startup_audits" in episode:
            startup_audits = list(
                episode["control_first_startup_audits"].values()
            )
        elif "control_first_startup_audit" in episode:
            startup_audits = [episode["control_first_startup_audit"]]
        else:
            startup_audits = []
        control_first_startup_passed = bool(startup_audits) and all(
            bool(audit.get("passed"))
            and bool(audit.get("control_applied_before_first_physics_step"))
            and bool(audit.get("exactly_one_guard_call_for_first_tick"))
            and audit.get("home_only_precharge_used") is False
            for audit in startup_audits
        )
        if "backward_exit_recovery_audits" in episode:
            recovery_audits = list(
                episode["backward_exit_recovery_audits"].values()
            )
        elif "backward_exit_recovery_audit" in episode:
            recovery_audits = [episode["backward_exit_recovery_audit"]]
        else:
            recovery_audits = []
        recovery_acceptances = [
            backward_exit_recovery_state_acceptance(audit)
            for audit in recovery_audits
        ]
        backward_exit_recovery_passed = bool(recovery_acceptances) and all(
            result["passed"] for result in recovery_acceptances
        )
        episode_checks.append(
            {
                "seed": int(episode["seed"]),
                "expected_segment_order": actual == expected,
                "reset_qpos_audits": reset_audits,
                "reset_qpos_passed": reset_qpos_passed,
                "control_first_startup_audits": startup_audits,
                "control_first_startup_passed": control_first_startup_passed,
                "backward_exit_recovery_audits": recovery_audits,
                "backward_exit_recovery_acceptances": recovery_acceptances,
                "backward_exit_recovery_passed": (
                    backward_exit_recovery_passed
                ),
                "segments": segment_checks,
                "passed": (
                    actual == expected
                    and reset_qpos_passed
                    and control_first_startup_passed
                    and backward_exit_recovery_passed
                    and not bool(episode["fell"])
                    and all(check["passed"] for check in segment_checks)
                ),
            }
        )
    return {
        "thresholds": asdict(thresholds),
        "episode_checks": episode_checks,
        "passed": bool(episodes) and all(check["passed"] for check in episode_checks),
    }


def hardware_gate(simulation_passed: bool) -> dict[str, Any]:
    """Return the non-promotable hardware gate, regardless of sim outcome."""

    if CONTRACT["deployment"]["hardware_status"] != "PROHIBITED":
        raise ValueError("exp_004 contract unexpectedly permits hardware")
    return {
        "status": "PROHIBITED",
        "hardware_deployment_allowed": False,
        "simulation_acceptance_passed": bool(simulation_passed),
        "simulation_pass_does_not_promote_hardware": True,
        "reason": CONTRACT["deployment"]["reason"],
        "remaining_required_gates": list(CONTRACT["deployment"]["required_gates"]),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_source_dependency_root_sha256(
    entries: Mapping[str, Mapping[str, Any]],
) -> str:
    """Hash a source closure by immutable workspace-relative path and digest."""

    rows: list[tuple[str, str]] = []
    for label, record in entries.items():
        relative_path = str(record.get("relative_path", ""))
        digest = str(record.get("sha256", "")).lower()
        if not relative_path or len(digest) != 64:
            raise ValueError(f"invalid runtime source dependency record: {label}")
        rows.append((relative_path.replace("\\", "/"), digest))
    serialized = "".join(
        f"{relative_path}\0{digest}\n"
        for relative_path, digest in sorted(rows)
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def capture_runtime_source_dependency_closure(
    paths: Mapping[str, Path],
    *,
    expected_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Capture and optionally hard-verify one runtime Python source closure."""

    if not paths:
        raise ValueError("runtime source dependency closure cannot be empty")
    if expected_sha256 is not None and set(paths) != set(expected_sha256):
        raise ValueError("runtime source dependency labels do not match allowlist")
    entries: dict[str, dict[str, Any]] = {}
    for label, source_path in paths.items():
        resolved = Path(source_path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"missing runtime source dependency: {resolved}")
        digest = sha256_file(resolved)
        expected = None if expected_sha256 is None else expected_sha256[label]
        if expected is not None and digest != str(expected).lower():
            raise ValueError(
                f"runtime source dependency hash mismatch for {label}: "
                f"expected {expected}, got {digest}"
            )
        try:
            relative_path = resolved.relative_to(WORKSPACE_ROOT).as_posix()
        except ValueError:
            relative_path = resolved.as_posix()
        entries[label] = {
            "path": str(resolved),
            "relative_path": relative_path,
            "sha256": digest,
            "expected_sha256": expected,
            "hash_allowlisted": expected is not None,
            "verified": expected is None or digest == expected,
        }
    return {
        "dependency_count": len(entries),
        "entries": entries,
        "root_sha256": runtime_source_dependency_root_sha256(entries),
        "all_hashes_verified": all(record["verified"] for record in entries.values()),
    }


def validate_frozen_runtime_source_dependencies() -> dict[str, Any]:
    """Hard-gate the external exp003/playground source import closure."""

    expected_paths = dict(FROZEN_RUNTIME_DEPENDENCY_PATHS)
    for label, source_path in expected_paths.items():
        if Path(source_path).resolve() != FROZEN_RUNTIME_DEPENDENCY_PATHS[label]:
            raise ValueError(f"runtime source dependency path mismatch: {label}")
    closure = capture_runtime_source_dependency_closure(
        expected_paths,
        expected_sha256=FROZEN_RUNTIME_DEPENDENCY_SHA256,
    )
    if closure["root_sha256"] != FROZEN_RUNTIME_DEPENDENCY_ROOT_SHA256:
        raise ValueError("runtime source dependency closure root hash mismatch")
    return closure


def validate_runtime_versions(actual_versions: Mapping[str, str]) -> dict[str, Any]:
    """Require the exact WSL package versions used by the locked evaluator."""

    expected = dict(FROZEN_RUNTIME_VERSIONS)
    if set(actual_versions) != set(expected):
        raise ValueError("runtime version keys do not match the frozen contract")
    actual = {key: str(actual_versions[key]) for key in expected}
    mismatches = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if actual[key] != expected[key]
    }
    if mismatches:
        raise ValueError(f"runtime version mismatch: {mismatches}")
    return {
        "expected": expected,
        "actual": actual,
        "exact_versions_verified": True,
    }


def _validated_current_formal_reverse_endpoint_mps() -> float:
    reverse_cases = [
        case for case in TRANSITION_CASES if case.name == "transition_reverse"
    ]
    if len(reverse_cases) != 1 or reverse_cases[0].command != (
        CURRENT_FORMAL_REVERSE_ENDPOINT_MPS,
        0.0,
        0.0,
    ):
        raise ValueError("current formal reverse endpoint constant/schedule mismatch")
    return CURRENT_FORMAL_REVERSE_ENDPOINT_MPS


def validate_diagnostic_reverse_phase_entry_evidence(
    path: Path = DIAGNOSTIC_REVERSE_PHASE_ENTRY_EVIDENCE_PATH,
) -> dict[str, Any]:
    """Pin the straight-reverse 5x15 diagnostic that selected entry phase 6."""

    resolved = path.resolve()
    if resolved != DIAGNOSTIC_REVERSE_PHASE_ENTRY_EVIDENCE_PATH:
        raise ValueError("diagnostic phase-entry evidence path must remain pinned")
    if not resolved.is_file():
        raise FileNotFoundError(f"missing diagnostic phase-entry evidence: {resolved}")
    digest = sha256_file(resolved)
    if digest != DIAGNOSTIC_REVERSE_PHASE_ENTRY_EVIDENCE_SHA256:
        raise ValueError("diagnostic phase-entry evidence hash mismatch")
    payload = _load_strict_json_object(resolved, "diagnostic phase-entry evidence")
    if payload.get("schema_version") != 1 or payload.get("artifact_kind") != (
        "openduckmini_reverse_transition_phase_reset_diagnostic"
    ):
        raise ValueError("unexpected diagnostic phase-entry evidence schema")
    if (
        payload.get("status") != "TRANSITION_SCREEN_PASS"
        or payload.get("hardware_deployment") != "PROHIBITED"
        or payload.get("simulation_only") is not True
        or payload.get("diagnostic_unadopted") is not True
    ):
        raise ValueError("diagnostic phase-entry evidence status is not usable")
    candidate = payload.get("candidate_profile")
    configuration = payload.get("configuration")
    acceptance = payload.get("acceptance")
    adoption = payload.get("adoption")
    if not all(
        isinstance(value, Mapping)
        for value in (candidate, configuration, acceptance, adoption)
    ):
        raise ValueError("diagnostic phase-entry evidence metadata is incomplete")
    phase_reset = configuration.get("phase_reset")
    if not isinstance(phase_reset, Mapping):
        raise ValueError("diagnostic phase-entry evidence lacks phase_reset")
    schedule = configuration.get("schedule")
    if (
        not isinstance(schedule, list)
        or len(schedule) != 4
        or not all(isinstance(item, Mapping) for item in schedule)
    ):
        raise ValueError("diagnostic phase-entry evidence schedule mismatch")
    schedule_names = [str(item.get("name", "")) for item in schedule]
    reverse_command = np.asarray(schedule[3].get("command"), dtype=np.float64)
    if (
        schedule_names
        != [
            "transition_stand_0",
            "transition_forward",
            "transition_stand_after_forward",
            "transition_reverse",
        ]
        or reverse_command.shape != (3,)
        or not np.array_equal(
            reverse_command,
            np.asarray(
                [
                    DIAGNOSTIC_REVERSE_PHASE_ENTRY_SOURCE_REVERSE_ENDPOINT_MPS,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            ),
        )
    ):
        raise ValueError("diagnostic phase-entry evidence command prefix mismatch")
    preincrement = _finite_number(
        phase_reset.get("router_preactivation_phase_steps"),
        "phase_reset.router_preactivation_phase_steps",
    )
    if (
        candidate.get("sha256") != DIAGNOSTIC_REVERSE_V3_PROFILE_SHA256
        or tuple(configuration.get("fixed_perturb_seeds", ()))
        != DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_FIXED_SEEDS
        or _finite_number(
            configuration.get("leg_target_margin_rad"),
            "phase_entry.configuration.leg_target_margin_rad",
        )
        != LEG_TARGET_MARGIN_RAD
        or _finite_number(
            configuration.get("target_slew_rate_rad_per_s"),
            "phase_entry.configuration.target_slew_rate_rad_per_s",
        )
        != TARGET_SLEW_LIMIT_RAD_PER_S
        or _finite_number(
            configuration.get("left_knee_extra_upper_margin_rad"),
            "phase_entry.configuration.left_knee_extra_upper_margin_rad",
        )
        != DIAGNOSTIC_REVERSE_V3_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
        or _finite_number(
            configuration.get("backward_residual_scale"),
            "phase_entry.configuration.backward_residual_scale",
        )
        != 0.0
        or phase_reset.get("enabled") is not True
        or phase_reset.get("activation")
        != "first effective vx < -0.02 after non-backward"
        or preincrement != FROZEN_DIAGNOSTIC_REVERSE_ENTRY_PHASE_INDICES["reverse"]
        or acceptance.get("passed") is not True
        or int(acceptance.get("actual_episode_passes", -1)) != 5
        or "NOT_ADOPTED" not in str(adoption.get("status", "")).upper()
    ):
        raise ValueError("diagnostic phase-entry evidence contract mismatch")
    current_endpoint = _validated_current_formal_reverse_endpoint_mps()
    endpoint_matched = bool(
        DIAGNOSTIC_REVERSE_PHASE_ENTRY_SOURCE_REVERSE_ENDPOINT_MPS
        == current_endpoint
    )
    return {
        "path": str(resolved),
        "sha256": digest,
        "schema_validated": True,
        "all_json_numbers_finite": True,
        "status": payload["status"],
        "candidate_profile_sha256": candidate["sha256"],
        "straight_preincrement_phase_index": preincrement,
        "episode_passes": int(acceptance["actual_episode_passes"]),
        "selection_scope": "straight_reverse_phase6_component_selection",
        "source_reverse_endpoint_mps": (
            DIAGNOSTIC_REVERSE_PHASE_ENTRY_SOURCE_REVERSE_ENDPOINT_MPS
        ),
        "current_formal_reverse_endpoint_mps": current_endpoint,
        "source_endpoint_matches_current_formal_endpoint": endpoint_matched,
        "current_endpoint_status": "CURRENT_ENDPOINT_REQUALIFICATION_REQUIRED",
        "usable_as_current_straight_endpoint_evidence": False,
        "adopted": False,
        "adoption_eligible": False,
        "hardware_deployment": "PROHIBITED",
    }


def validate_diagnostic_backward_exit_recovery_evidence(
    path: Path = DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_EVIDENCE_PATH,
) -> dict[str, Any]:
    """Strictly pin the five-seed recovery selection diagnostic.

    The source prefix used the then-current straight reverse endpoint of
    ``-0.075 m/s``.  It is valid evidence for selecting the reverse-left exit
    recovery composition, but never evidence for the current ``-0.050 m/s``
    endpoint.  That mismatch is explicit and keeps adoption fail-closed.
    """

    resolved = path.resolve()
    if resolved != DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_EVIDENCE_PATH:
        raise ValueError("diagnostic backward-exit recovery evidence path must remain pinned")
    if not resolved.is_file():
        raise FileNotFoundError(
            f"missing diagnostic backward-exit recovery evidence: {resolved}"
        )
    digest = sha256_file(resolved)
    if digest != DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_EVIDENCE_SHA256:
        raise ValueError("diagnostic backward-exit recovery evidence hash mismatch")
    payload = _load_strict_json_object(
        resolved, "diagnostic backward-exit recovery evidence"
    )
    if payload.get("schema_version") != 1 or payload.get("artifact_kind") != (
        "openduckmini_backward_exit_recovery_diagnostic"
    ):
        raise ValueError("unexpected diagnostic backward-exit recovery evidence schema")
    if (
        payload.get("status") != "RECOVERY_SCREEN_PASS"
        or payload.get("hardware_deployment") != "PROHIBITED"
        or payload.get("simulation_only") is not True
        or payload.get("diagnostic_unadopted") is not True
        or payload.get("central_evaluator_modified_by_this_diagnostic") is not False
    ):
        raise ValueError("diagnostic backward-exit recovery evidence status is not usable")

    configuration = payload.get("configuration")
    hard_gate = payload.get("hard_gate")
    baseline_reproduction = payload.get("baseline_reproduction")
    selection = payload.get("selection")
    adoption = payload.get("adoption")
    provenance = payload.get("provenance")
    if not all(
        isinstance(value, Mapping)
        for value in (
            configuration,
            hard_gate,
            baseline_reproduction,
            selection,
            adoption,
            provenance,
        )
    ):
        raise ValueError("diagnostic backward-exit recovery metadata is incomplete")

    schedule = configuration.get("schedule")
    if (
        not isinstance(schedule, list)
        or len(schedule) != 7
        or not all(isinstance(item, Mapping) for item in schedule)
    ):
        raise ValueError("diagnostic backward-exit recovery schedule mismatch")
    schedule_names = [str(item.get("name", "")) for item in schedule]
    expected_names = [
        "transition_stand_0",
        "transition_forward",
        "transition_stand_after_forward",
        "transition_reverse",
        "transition_stand_after_reverse",
        "transition_reverse_turn_left",
        "transition_stand_after_reverse_turn_left",
    ]
    reverse_command = np.asarray(schedule[3].get("command"), dtype=np.float64)
    reverse_left_command = np.asarray(schedule[5].get("command"), dtype=np.float64)
    if (
        schedule_names != expected_names
        or reverse_command.shape != (3,)
        or not np.array_equal(
            reverse_command,
            np.asarray(
                [
                    DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_SOURCE_REVERSE_ENDPOINT_MPS,
                    0.0,
                    0.0,
                ]
            ),
        )
        or reverse_left_command.shape != (3,)
        or not np.array_equal(
            reverse_left_command,
            np.asarray(ATOMIC_REVERSE_TURN_COMMANDS["left"]),
        )
    ):
        raise ValueError("diagnostic backward-exit recovery command prefix mismatch")

    phase_indices = configuration.get("diagnostic_reverse_entry_phase_indices")
    if not isinstance(phase_indices, Mapping) or dict(phase_indices) != dict(
        FROZEN_DIAGNOSTIC_REVERSE_ENTRY_PHASE_INDICES
    ):
        raise ValueError("diagnostic backward-exit recovery phase mapping mismatch")
    if (
        tuple(configuration.get("fixed_seeds", ()))
        != DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_FIXED_SEEDS
        or _finite_number(
            configuration.get("leg_target_margin_rad"),
            "configuration.leg_target_margin_rad",
        )
        != LEG_TARGET_MARGIN_RAD
        or _finite_number(
            configuration.get("target_slew_rate_rad_per_s"),
            "configuration.target_slew_rate_rad_per_s",
        )
        != TARGET_SLEW_LIMIT_RAD_PER_S
        or _finite_number(
            configuration.get("positive_noise_reset_qpos_inward_margin_rad"),
            "configuration.positive_noise_reset_qpos_inward_margin_rad",
        )
        != RESET_NOISE_MARGIN_RAD
        or _finite_number(
            configuration.get("reverse_profile_left_knee_extra_upper_margin_rad"),
            "configuration.reverse_profile_left_knee_extra_upper_margin_rad",
        )
        != DIAGNOSTIC_REVERSE_V3_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
        or configuration.get("recovery_scope")
        != "every_backward_feedforward_exit"
        or configuration.get("recovery_activation")
        != "effective_vx_lt_-0.02_true_to_false"
    ):
        raise ValueError("diagnostic backward-exit recovery safety contract mismatch")

    selected = selection.get("selected_strategy")
    summary = selection.get("selected_summary")
    if not isinstance(selected, Mapping) or not isinstance(summary, Mapping):
        raise ValueError("diagnostic backward-exit recovery selection is missing")
    selected_contract = {
        "name": "cap0125_hold0250ms_instant",
        "cap_rad": DIAGNOSTIC_REVERSE_V3_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD,
        "hold_seconds": 0.25,
        "release_seconds": 0.0,
        "control_dt_s": 0.02,
        "duration_quantization": "ceil(seconds / control_dt)",
        "hold_ticks": BACKWARD_EXIT_RECOVERY_HOLD_TICKS,
        "release_ticks": 0,
        "applied_hold_seconds": BACKWARD_EXIT_RECOVERY_HOLD_SECONDS,
        "applied_release_seconds": 0.0,
        "release_shape": "instant",
        "restriction_area_rad_s": 0.0032500000000000003,
    }
    if dict(selected) != selected_contract:
        raise ValueError("diagnostic backward-exit recovery selected strategy mismatch")
    if (
        selection.get("passed") is not True
        or int(summary.get("passed_episode_count", -1)) != 5
        or int(summary.get("total_prefix_qpos_limit_violations", -1)) != 0
        or int(summary.get("recovery_stand_qpos_limit_violations", -1)) != 0
        or _finite_number(
            summary.get("maximum_recovery_stand_qpos_excess_rad"),
            "selection.maximum_recovery_stand_qpos_excess_rad",
        )
        != 0.0
        or baseline_reproduction.get("passed") is not True
        or hard_gate.get("gates_relaxed") is not False
        or int(hard_gate.get("maximum_qpos_limit_violations", -1)) != 0
        or int(hard_gate.get("maximum_falls", -1)) != 0
        or int(hard_gate.get("maximum_target_limit_margin_or_slew_violations", -1))
        != 0
        or int(hard_gate.get("maximum_route_violations", -1)) != 0
        or hard_gate.get("head_action_and_target_exact_zero") is not True
        or "NOT_ADOPTED" not in str(adoption.get("status", "")).upper()
        or adoption.get("hardware_deployment") != "PROHIBITED"
    ):
        raise ValueError("diagnostic backward-exit recovery hard gate mismatch")

    script = provenance.get("script")
    profiles = provenance.get("profiles")
    policy = provenance.get("policy")
    generated_assets = provenance.get("generated_assets")
    model_contract = provenance.get("model_contract")
    if not all(
        isinstance(value, Mapping)
        for value in (
            script,
            profiles,
            policy,
            generated_assets,
            model_contract,
        )
    ):
        raise ValueError("diagnostic backward-exit recovery provenance is incomplete")
    expected_profile_hashes = {
        "straight": DIAGNOSTIC_REVERSE_V3_PROFILE_SHA256,
        **dict(DIAGNOSTIC_REVERSE_TURN_PROFILE_SHA256),
    }
    generated_closure = generated_assets.get("dependency_closure")
    if (
        script.get("sha256")
        != "469f5678008dd09b11eeb4763353d0fa1d597dc33a4b0df2927e850c0e4a7d82"
        or policy.get("sha256") != BASE_V22_POLICY_SHA256
        or generated_assets.get("contract") != "hardware_safe_simulation_only"
        or generated_assets.get("real_hardware_deployment_allowed") is not False
        or not isinstance(generated_closure, Mapping)
        or generated_closure.get("root_sha256")
        != FROZEN_GENERATED_DEPENDENCY_ROOT_SHA256
        or model_contract.get("home_matches_safe_init") is not True
        or model_contract.get("leg_ranges_match_safe_limits") is not True
        or model_contract.get("head_home_targets_zero") is not True
        or any(
            not isinstance(profiles.get(label), Mapping)
            or profiles[label].get("sha256") != expected_hash
            for label, expected_hash in expected_profile_hashes.items()
        )
    ):
        raise ValueError("diagnostic backward-exit recovery profile/policy hash mismatch")

    selected_results = [
        result
        for result in payload.get("results", ())
        if isinstance(result, Mapping)
        and isinstance(result.get("strategy"), Mapping)
        and result["strategy"].get("name") == selected_contract["name"]
    ]
    if len(selected_results) != 1:
        raise ValueError("diagnostic backward-exit recovery result selection mismatch")
    selected_result = selected_results[0]
    episodes = selected_result.get("episodes")
    if (
        selected_result.get("passed") is not True
        or int(selected_result.get("episode_count", -1)) != 5
        or int(selected_result.get("passed_episode_count", -1)) != 5
        or not isinstance(episodes, list)
        or not all(isinstance(episode, Mapping) for episode in episodes)
        or tuple(int(episode.get("seed", -1)) for episode in episodes)
        != DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_FIXED_SEEDS
        or not all(episode.get("passed") is True for episode in episodes)
        or not all(len(episode.get("recovery_events", ())) == 2 for episode in episodes)
        or not all(
            all(
                int(event.get("positive_cap_tick_count", -1))
                == BACKWARD_EXIT_RECOVERY_HOLD_TICKS
                and _finite_number(
                    event.get("cap_rad"), "recovery_event.cap_rad"
                )
                == DIAGNOSTIC_REVERSE_V3_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
                for event in episode.get("recovery_events", ())
            )
            for episode in episodes
        )
    ):
        raise ValueError("diagnostic backward-exit recovery episode evidence mismatch")

    current_endpoint = _validated_current_formal_reverse_endpoint_mps()
    endpoint_matched = bool(
        DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_SOURCE_REVERSE_ENDPOINT_MPS
        == current_endpoint
    )
    return {
        "path": str(resolved),
        "sha256": digest,
        "schema_validated": True,
        "all_json_numbers_finite": True,
        "status": payload["status"],
        "selection_scope": "reverse_turn_left_to_stand_recovery",
        "profile_sha256s": expected_profile_hashes,
        "policy_sha256": BASE_V22_POLICY_SHA256,
        "selected_strategy": dict(selected),
        "selected_upper_target_rad": (
            FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_UPPER_TARGET_RAD
        ),
        "episode_passes": 5,
        "source_reverse_endpoint_mps": (
            DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_SOURCE_REVERSE_ENDPOINT_MPS
        ),
        "current_formal_reverse_endpoint_mps": current_endpoint,
        "source_endpoint_matches_current_formal_endpoint": endpoint_matched,
        "current_endpoint_status": "CURRENT_ENDPOINT_REQUALIFICATION_REQUIRED",
        "usable_as_current_straight_endpoint_evidence": False,
        "adopted": False,
        "adoption_eligible": False,
        "hardware_deployment": "PROHIBITED",
    }


def validate_diagnostic_backward_exit_recovery_execution_bundle(
    recovery_evidence: Mapping[str, Any],
    executed_reverse_profiles: Mapping[str, Mapping[str, Any]],
    policy_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind recovery execution to the exact profile and base-policy bank."""

    expected_profiles = recovery_evidence.get("profile_sha256s")
    policy_roles = policy_provenance.get("roles")
    if (
        not isinstance(expected_profiles, Mapping)
        or not isinstance(policy_roles, Mapping)
        or set(expected_profiles) != {"straight", "left", "right"}
        or set(executed_reverse_profiles) != {"straight", "left", "right"}
        or set(policy_roles) != set(REQUIRED_POLICY_ROLES)
        or not all(
            isinstance(record, Mapping)
            for record in executed_reverse_profiles.values()
        )
        or not all(isinstance(record, Mapping) for record in policy_roles.values())
    ):
        raise ValueError(
            "diagnostic backward-exit recovery execution bundle is incomplete"
        )
    expected_policy_sha256 = str(recovery_evidence.get("policy_sha256", ""))
    executed_profile_hashes = {
        label: str(executed_reverse_profiles[label].get("sha256", ""))
        for label in ("straight", "left", "right")
    }
    if executed_profile_hashes != dict(expected_profiles):
        raise ValueError(
            "diagnostic backward-exit recovery requires the exact hash-pinned "
            "straight/left/right profile bank"
        )
    executed_policy_hashes = {
        role: str(policy_roles[role].get("sha256", ""))
        for role in REQUIRED_POLICY_ROLES
    }
    if (
        len(expected_policy_sha256) != 64
        or set(executed_policy_hashes.values()) != {expected_policy_sha256}
    ):
        raise ValueError(
            "diagnostic backward-exit recovery requires the exact hash-pinned "
            "base-v22 policy bank"
        )
    return {
        "passed": True,
        "profile_sha256s": executed_profile_hashes,
        "policy_sha256": expected_policy_sha256,
        "policy_roles": list(REQUIRED_POLICY_ROLES),
    }


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is prohibited: {value}")


def _assert_finite_json(value: Any, *, location: str = "$") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not np.isfinite(value):
            raise ValueError(f"non-finite number at {location}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_json(item, location=f"{location}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_json(item, location=f"{location}.{key}")
        return
    raise ValueError(f"unsupported JSON value at {location}: {type(value).__name__}")


def _load_strict_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    _assert_finite_json(payload)
    return payload


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _validate_superseded_stage_a_5x15_selection_evidence(
    path: Path = FORMAL_CANDIDATE_SELECTION_EVIDENCE_PATH,
) -> dict[str, Any]:
    """Strictly pin the 5x15 Stage-A execution-bundle selection artifact.

    This gate is intentionally separate from formal adoption evidence.  A
    successful result authorizes only the default candidate execution bundle;
    it cannot make a command case, simulation release, package, or hardware
    deployment adoptable.
    """

    resolved = path.resolve()
    if resolved != FORMAL_CANDIDATE_SELECTION_EVIDENCE_PATH:
        raise ValueError("formal-candidate selection evidence path must remain pinned")
    if not resolved.is_file():
        raise FileNotFoundError(
            f"missing formal-candidate selection evidence: {resolved}"
        )
    digest = sha256_file(resolved)
    if (
        digest != FORMAL_CANDIDATE_SELECTION_EVIDENCE_SHA256
        or digest not in FORMAL_CANDIDATE_SELECTION_EVIDENCE_SHA256_ALLOWLIST
    ):
        raise ValueError("formal-candidate selection evidence hash mismatch")
    payload = _load_strict_json_object(resolved, "formal-candidate selection evidence")

    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("evaluator_id") != EVALUATOR_ID
        or payload.get("simulation_suite_acceptance_passed") is not True
        or payload.get("simulation_acceptance_passed") is not False
        or "DIAGNOSTIC_UNADOPTED_REVERSE_PROFILE_BANK"
        not in str(payload.get("evaluation_mode", ""))
    ):
        raise ValueError("formal-candidate selection evidence top-level status mismatch")

    configuration = payload.get("configuration")
    release = payload.get("release_qualification")
    phase_contract = payload.get("diagnostic_reverse_phase_entry_contract")
    recovery_contract = payload.get("diagnostic_backward_exit_recovery_contract")
    command_contract = payload.get("command_mapping_contract")
    if not all(
        isinstance(value, Mapping)
        for value in (
            configuration,
            release,
            phase_contract,
            recovery_contract,
            command_contract,
        )
    ):
        raise ValueError("formal-candidate selection metadata is incomplete")
    expected_configuration = {
        "seed": FORMAL_CANDIDATE_MASTER_SEED,
        "episodes": 5,
        "seconds": 15.0,
        "transition_seconds": 15.0,
        "transition_stand_seconds": 5.0,
        "warmup_seconds": 1.5,
        "initial_base_speed": 0.1,
        "initial_joint_noise_scale": 1.0,
        "backward_residual_scale": 0.0,
        "leg_target_margin_rad": LEG_TARGET_MARGIN_RAD,
        "target_slew_rate_rad_per_s": TARGET_SLEW_LIMIT_RAD_PER_S,
        "reset_noise_margin_rad": RESET_NOISE_MARGIN_RAD,
        "left_knee_extra_upper_margin_rad": (
            DIAGNOSTIC_REVERSE_V3_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
        ),
    }
    for key, expected in expected_configuration.items():
        if configuration.get(key) != expected:
            raise ValueError(f"formal-candidate configuration mismatch: {key}")
    if (
        not isinstance(configuration.get("diagnostic_unadopted_reverse_profile"), str)
        or not isinstance(
            configuration.get("diagnostic_unadopted_reverse_left_profile"), str
        )
        or not isinstance(
            configuration.get("diagnostic_unadopted_reverse_right_profile"), str
        )
        or configuration.get("diagnostic_unadopted_backward_exit_recovery") is not True
        or dict(
            configuration.get(
                "diagnostic_unadopted_reverse_entry_phase_indices", {}
            )
        )
        != dict(FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES)
    ):
        raise ValueError("formal-candidate historical execution bundle mismatch")
    release_actual = release.get("actual")
    if (
        not isinstance(release_actual, Mapping)
        or release_actual.get("master_seed") != FORMAL_CANDIDATE_MASTER_SEED
        or release_actual.get("episodes") != 5
        or release_actual.get("seconds") != 15.0
        or release.get("scale_matches_frozen_contract") is not False
        or release.get("release_qualification_eligible") is not False
    ):
        raise ValueError("formal-candidate selection scale/status mismatch")

    phase_mapping = phase_contract.get("preincrement_phase_indices")
    recovery_runtime = recovery_contract.get("runtime_contract")
    if (
        phase_contract.get("enabled") is not True
        or phase_contract.get("current_formal_reverse_endpoint_mps")
        != CURRENT_FORMAL_REVERSE_ENDPOINT_MPS
        or not isinstance(phase_mapping, Mapping)
        or dict(phase_mapping) != dict(FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES)
        or recovery_contract.get("enabled") is not True
        or recovery_contract.get("current_formal_reverse_endpoint_mps")
        != CURRENT_FORMAL_REVERSE_ENDPOINT_MPS
        or not isinstance(recovery_runtime, Mapping)
        or recovery_runtime.get("enabled_by_default") is not False
        or recovery_runtime.get("diagnostic_unadopted_only") is not True
        or recovery_runtime.get("extra_upper_margin_rad")
        != BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD
        or recovery_runtime.get("upper_target_rad")
        != BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD
        or recovery_runtime.get("hold_control_ticks")
        != BACKWARD_EXIT_RECOVERY_HOLD_TICKS
        or recovery_runtime.get("hold_seconds") != BACKWARD_EXIT_RECOVERY_HOLD_SECONDS
    ):
        raise ValueError("formal-candidate phase/recovery selection mismatch")

    validation_gate = command_contract.get("validation_status_gate")
    expected_blocked_names = {
        "reverse",
        "reverse_turn_left",
        "reverse_turn_right",
        "transition_reverse",
        "transition_reverse_turn_left",
        "transition_reverse_turn_right",
    }
    if (
        not isinstance(validation_gate, Mapping)
        or validation_gate.get("passed") is not False
        or validation_gate.get("case_count") != 38
        or validation_gate.get("nonadoptable_case_count") != 6
        or {
            str(record.get("name", ""))
            for record in validation_gate.get("nonadoptable_cases", ())
            if isinstance(record, Mapping)
        }
        != expected_blocked_names
    ):
        raise ValueError("formal-candidate reverse command cases are not fail-closed")

    suites = payload.get("suites")
    expected_segment_counts = {"primitives": 35, "compounds": 30, "transitions": 125}
    if not isinstance(suites, Mapping) or set(suites) != set(expected_segment_counts):
        raise ValueError("formal-candidate suite set mismatch")
    episodes: list[Mapping[str, Any]] = []
    segments: list[Mapping[str, Any]] = []
    acceptance_checks: list[Mapping[str, Any]] = []
    accepted_segments: list[Mapping[str, Any]] = []
    reset_audits: list[Mapping[str, Any]] = []
    startup_audits: list[Mapping[str, Any]] = []
    recovery_state_audits: list[Mapping[str, Any]] = []
    for suite_name, expected_segment_count in expected_segment_counts.items():
        suite = suites[suite_name]
        if not isinstance(suite, Mapping):
            raise ValueError(f"formal-candidate {suite_name} suite is not a mapping")
        suite_episodes = suite.get("episodes")
        acceptance = suite.get("acceptance")
        if (
            not isinstance(suite_episodes, list)
            or len(suite_episodes) != 5
            or not all(isinstance(item, Mapping) for item in suite_episodes)
            or not isinstance(acceptance, Mapping)
            or acceptance.get("passed") is not True
        ):
            raise ValueError(f"formal-candidate {suite_name} acceptance mismatch")
        suite_segments = [
            segment
            for episode in suite_episodes
            for segment in episode.get("segments", ())
            if isinstance(segment, Mapping)
        ]
        if len(suite_segments) != expected_segment_count:
            raise ValueError(f"formal-candidate {suite_name} segment count mismatch")
        suite_checks = acceptance.get("episode_checks")
        if (
            not isinstance(suite_checks, list)
            or len(suite_checks) != 5
            or not all(isinstance(item, Mapping) for item in suite_checks)
            or not all(item.get("passed") is True for item in suite_checks)
        ):
            raise ValueError(f"formal-candidate {suite_name} episode checks failed")
        episodes.extend(suite_episodes)
        segments.extend(suite_segments)
        acceptance_checks.extend(suite_checks)
        for check in suite_checks:
            accepted_segments.extend(
                item
                for item in check.get("segments", ())
                if isinstance(item, Mapping)
            )
            reset_audits.extend(
                item
                for item in check.get("reset_qpos_audits", ())
                if isinstance(item, Mapping)
            )
            startup_audits.extend(
                item
                for item in check.get("control_first_startup_audits", ())
                if isinstance(item, Mapping)
            )
            recovery_state_audits.extend(
                item
                for item in check.get("backward_exit_recovery_audits", ())
                if isinstance(item, Mapping)
            )

    if (
        len(episodes) != 15
        or len(segments) != 190
        or len(accepted_segments) != 190
        or not all(item.get("passed") is True for item in accepted_segments)
        or not all(episode.get("fell") is False for episode in episodes)
        or not all(
            episode.get("completed_segment_count")
            == episode.get("requested_segment_count")
            for episode in episodes
        )
        or not all(
            segment.get("completed") is True
            and segment.get("fell") is False
            and segment.get("completed_physics_substeps")
            == segment.get("expected_physics_substeps")
            for segment in segments
        )
    ):
        raise ValueError("formal-candidate episode/segment completion mismatch")

    safety_zero_fields = (
        "applied_target_limit_violations",
        "desired_target_margin_violations",
        "nonfinite_sample_count",
        "preclip_target_limit_violations",
        "qpos_limit_violations",
        "target_slew_violations",
        "unauthorized_applied_target_margin_violations",
        "maximum_applied_target_excess_rad",
        "maximum_desired_target_margin_excess_rad",
        "maximum_preclip_target_excess_rad",
        "maximum_qpos_excess_rad",
        "applied_head_action_peak",
        "head_target_peak_rad",
    )
    physics_zero_fields = (
        "height_fall_samples",
        "upright_fall_samples",
        "nonfinite_full_qpos_samples",
        "nonfinite_full_qvel_samples",
        "nonfinite_leg_qpos_samples",
        "nonfinite_pose_samples",
        "nonfinite_state_samples",
        "qpos_limit_violations",
        "maximum_qpos_excess_rad",
    )
    physics_sample_count = 0
    contact_sample_count = 0
    leg_qpos_sample_count = 0
    control_sample_count = 0
    phase_audits: list[Mapping[str, Any]] = []
    recovery_segment_audits: list[Mapping[str, Any]] = []
    for segment in segments:
        safety = segment.get("safety_audit")
        physics = segment.get("physics_substep_audit")
        routing = segment.get("routing")
        recovery = segment.get("backward_exit_recovery_audit")
        if not all(
            isinstance(value, Mapping)
            for value in (safety, physics, routing, recovery)
        ):
            raise ValueError("formal-candidate segment audit is incomplete")
        if any(safety.get(field) != 0 for field in safety_zero_fields):
            raise ValueError("formal-candidate target/qpos safety audit failed")
        if (
            safety.get("leg_target_margin_rad") != LEG_TARGET_MARGIN_RAD
            or safety.get("target_slew_limit_rad_per_s")
            != TARGET_SLEW_LIMIT_RAD_PER_S
            or safety.get("maximum_target_slew_rate_rad_per_s")
            > TARGET_SLEW_LIMIT_RAD_PER_S + 2e-15
        ):
            raise ValueError("formal-candidate target guard contract mismatch")
        if (
            any(physics.get(field) != 0 for field in physics_zero_fields)
            or physics.get("fall_or_nonfinite_detected") is not False
            or physics.get("contact_sample_count_matches_sample_count") is not True
            or physics.get("contact_sample_count") != physics.get("sample_count")
            or physics.get("minimum_height_m") < physics.get("minimum_height_limit_m")
            or physics.get("minimum_upright") < physics.get("minimum_upright_limit")
            or routing.get("command_clip_events") != 0
            or routing.get("prohibited_expert_steps") != 0
        ):
            raise ValueError("formal-candidate physics/routing safety audit failed")
        phase = routing.get("reverse_entry_phase")
        if not isinstance(phase, Mapping) or phase.get("passed") is not True:
            raise ValueError("formal-candidate phase-entry audit failed")
        if recovery.get("passed") is not True:
            raise ValueError("formal-candidate recovery segment audit failed")
        physics_sample_count += int(physics["sample_count"])
        contact_sample_count += int(physics["contact_sample_count"])
        leg_qpos_sample_count += int(physics["leg_joint_sample_count"])
        control_sample_count += int(safety["sample_count"])
        phase_audits.append(phase)
        recovery_segment_audits.append(recovery)
    if (
        physics_sample_count != 1_100_000
        or contact_sample_count != 1_100_000
        or leg_qpos_sample_count != 11_000_000
        or control_sample_count != 110_000
    ):
        raise ValueError("formal-candidate audited sample totals mismatch")

    phase_events = [
        event
        for audit in phase_audits
        for event in audit.get("events", ())
        if isinstance(event, Mapping)
    ]
    phase_counts = {
        expert: sum(event.get("current_expert") == expert for event in phase_events)
        for expert in FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES
    }
    if (
        len(phase_audits) != 190
        or sum(int(audit.get("event_count", -1)) for audit in phase_audits) != 30
        or sum(
            int(audit.get("backward_feedforward_entry_count", -1))
            for audit in phase_audits
        )
        != 30
        or sum(
            int(audit.get("within_backward_family_active_switch_count", -1))
            for audit in phase_audits
        )
        != 0
        or phase_counts != {expert: 10 for expert in phase_counts}
        or any(
            event.get("reset_preincrement_phase_index")
            != FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES[
                str(event.get("current_expert", ""))
            ]
            for event in phase_events
        )
    ):
        raise ValueError("formal-candidate phase-entry event totals mismatch")

    if (
        len(recovery_segment_audits) != 190
        or sum(int(audit.get("exit_event_count", -1)) for audit in recovery_segment_audits)
        != 15
        or sum(int(audit.get("active_tick_count", -1)) for audit in recovery_segment_audits)
        != 195
        or sum(int(audit.get("cap_violation_count", -1)) for audit in recovery_segment_audits)
        != 0
        or sum(
            int(audit.get("reentry_cancel_count", -1))
            for audit in recovery_segment_audits
        )
        != 0
        or any(audit.get("remaining_ticks_after_segment") != 0 for audit in recovery_segment_audits)
        or sum(int(audit.get("sample_count", -1)) for audit in recovery_segment_audits)
        != 110_000
        or sum(
            int(audit.get("final_guard_call_count", -1))
            for audit in recovery_segment_audits
        )
        != 110_000
    ):
        raise ValueError("formal-candidate recovery segment totals mismatch")

    if (
        len(reset_audits) != 70
        or len(startup_audits) != 70
        or len(recovery_state_audits) != 70
        or not all(audit.get("passed") is True for audit in reset_audits)
        or not all(audit.get("passed") is True for audit in startup_audits)
        or not all(audit.get("passed") is True for audit in recovery_state_audits)
        or sum(int(audit.get("physical_safe_limit_violations", -1)) for audit in reset_audits)
        != 0
        or sum(int(audit.get("noise_margin_violations", -1)) for audit in reset_audits)
        != 0
        or any(audit.get("head_qpos_peak_rad") != 0.0 for audit in reset_audits)
        or any(
            audit.get("control_applied_before_first_physics_step") is not True
            or audit.get("exactly_one_guard_call_for_first_tick") is not True
            or audit.get("physics_steps_before_control") != 0
            or audit.get("guard_calls_for_first_tick") != 1
            or audit.get("applied_target_physical_safe_violations") != 0
            for audit in startup_audits
        )
        or sum(int(audit.get("exit_event_count", -1)) for audit in recovery_state_audits)
        != 15
        or sum(int(audit.get("active_tick_count", -1)) for audit in recovery_state_audits)
        != 195
        or sum(int(audit.get("completed_event_count", -1)) for audit in recovery_state_audits)
        != 15
        or sum(int(audit.get("cap_violation_count", -1)) for audit in recovery_state_audits)
        != 0
        or sum(int(audit.get("remaining_ticks", -1)) for audit in recovery_state_audits)
        != 0
        or sum(int(audit.get("control_tick_count", -1)) for audit in recovery_state_audits)
        != 110_000
        or sum(int(audit.get("final_guard_call_count", -1)) for audit in recovery_state_audits)
        != 110_000
    ):
        raise ValueError("formal-candidate reset/startup/recovery acceptance mismatch")

    reverse_profiles = payload.get("reverse_profile_evidence")
    policy_provenance = payload.get("policy_provenance")
    if not isinstance(reverse_profiles, Mapping) or not isinstance(
        policy_provenance, Mapping
    ):
        raise ValueError("formal-candidate profile/policy evidence is incomplete")
    executed_profiles = reverse_profiles.get("executed_profiles")
    policy_roles = policy_provenance.get("roles")
    if (
        not isinstance(executed_profiles, Mapping)
        or set(executed_profiles) != set(FORMAL_CANDIDATE_PROFILE_SHA256S)
        or {
            label: str(record.get("sha256", ""))
            for label, record in executed_profiles.items()
            if isinstance(record, Mapping)
        }
        != dict(FORMAL_CANDIDATE_PROFILE_SHA256S)
        or not all(
            isinstance(record, Mapping)
            and record.get("schema_validated") is True
            and record.get("adopted") is False
            and record.get("adoption_eligible") is False
            for record in executed_profiles.values()
        )
        or not isinstance(policy_roles, Mapping)
        or set(policy_roles) != set(REQUIRED_POLICY_ROLES)
        or policy_provenance.get("all_roles_allowlisted") is not True
        or any(
            not isinstance(record, Mapping)
            or record.get("sha256") != BASE_V22_POLICY_SHA256
            or record.get("formal_base_v22_allowlisted") is not True
            or record.get("adopted") is not True
            for record in policy_roles.values()
        )
    ):
        raise ValueError("formal-candidate profile/policy hash bank mismatch")

    provenance = payload.get("runtime_dependency_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("formal-candidate runtime provenance is missing")
    pre = provenance.get("pre_import")
    post = provenance.get("post_evaluation")
    runtime_data_pre = provenance.get("runtime_model_and_data_pre_evaluation")
    if not all(isinstance(value, Mapping) for value in (pre, post, runtime_data_pre)):
        raise ValueError("formal-candidate pre/post provenance is incomplete")
    closure_contract = {
        "exp004_source_and_contract_snapshot": (
            "e7101bc907054e1a098e3a66f81c4d0437e044a2af66376de3cf41e9f8023b4c",
            9,
        ),
        "external_hard_allowlisted_source_closure": (
            FROZEN_RUNTIME_DEPENDENCY_ROOT_SHA256,
            4,
        ),
        "hard_allowlisted_runtime_binary_closure": (
            "4e382762ffe85e33ba4839969088fed6f27cae1b35f6e0247a7d1b18937abe5f",
            5,
        ),
    }
    for label, (root_sha256, dependency_count) in closure_contract.items():
        pre_closure = pre.get(label)
        post_closure = post.get(label)
        if (
            not isinstance(pre_closure, Mapping)
            or not isinstance(post_closure, Mapping)
            or dict(pre_closure) != dict(post_closure)
            or pre_closure.get("root_sha256") != root_sha256
            or pre_closure.get("dependency_count") != dependency_count
            or pre_closure.get("all_hashes_verified") is not True
            or not all(
                isinstance(record, Mapping) and record.get("verified") is True
                for record in pre_closure.get("entries", {}).values()
            )
        ):
            raise ValueError(f"formal-candidate provenance closure mismatch: {label}")
    runtime_data_post = post.get("runtime_model_and_data_closure")
    if (
        provenance.get("verified") is not True
        or provenance.get("pre_post_source_and_data_hashes_unchanged") is not True
        or provenance.get("all_onnx_sessions_cpu_only_verified") is not True
        or not isinstance(runtime_data_post, Mapping)
        or dict(runtime_data_pre) != dict(runtime_data_post)
        or runtime_data_pre.get("root_sha256")
        != "e07dcca21a8620f1dd68fb85cae8e923d57b4b7f431eb06a841e3b411f3eba2a"
        or runtime_data_pre.get("dependency_count") != 48
        or runtime_data_pre.get("all_hashes_verified") is not True
        or any(
            providers != ["CPUExecutionProvider"]
            for providers in provenance.get("onnx_session_execution_providers", {}).values()
        )
    ):
        raise ValueError("formal-candidate runtime data/provider provenance mismatch")
    runtime_environment = provenance.get("runtime_environment")
    if (
        not isinstance(runtime_environment, Mapping)
        or runtime_environment.get("exact_versions_verified") is not True
        or runtime_environment.get("actual") != dict(FROZEN_RUNTIME_VERSIONS)
        or runtime_environment.get("expected") != dict(FROZEN_RUNTIME_VERSIONS)
        or runtime_environment.get("onnxruntime_build_commit_verified")
        != "45de2a8b06"
    ):
        raise ValueError("formal-candidate runtime version provenance mismatch")

    hardware = payload.get("hardware_gate")
    adoption = payload.get("adoption_contract")
    if (
        not isinstance(hardware, Mapping)
        or hardware.get("status") != "PROHIBITED"
        or hardware.get("hardware_deployment_allowed") is not False
        or not isinstance(adoption, Mapping)
        or adoption.get("passed") is not False
        or adoption.get("reverse_profile_adopted") is not False
    ):
        raise ValueError("formal-candidate adoption/hardware gate mismatch")

    return {
        "path": str(resolved),
        "sha256": digest,
        "hash_allowlisted_for_candidate_selection": True,
        "hash_allowlisted_for_adoption": False,
        "schema_validated": True,
        "all_json_numbers_finite": True,
        "status": H3_CANDIDATE_SELECTION_STATUS,
        "selection_scope": "combined_phase644_recovery13_endpointm050_5x15",
        "master_seed": FORMAL_CANDIDATE_MASTER_SEED,
        "profile_sha256s": dict(FORMAL_CANDIDATE_PROFILE_SHA256S),
        "policy_sha256": BASE_V22_POLICY_SHA256,
        "reverse_endpoint_mps": CURRENT_FORMAL_REVERSE_ENDPOINT_MPS,
        "phase_preincrement_indices": dict(
            FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES
        ),
        "backward_exit_recovery": {
            "enabled": True,
            "hold_control_ticks": BACKWARD_EXIT_RECOVERY_HOLD_TICKS,
            "hold_seconds": BACKWARD_EXIT_RECOVERY_HOLD_SECONDS,
            "upper_target_rad": BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD,
        },
        "suite_episode_count": len(episodes),
        "segment_pass_count": len(accepted_segments),
        "physics_substep_count": physics_sample_count,
        "contact_sample_count": contact_sample_count,
        "phase_entry_event_count": len(phase_events),
        "recovery_exit_event_count": 15,
        "recovery_active_tick_count": 195,
        "pre_post_provenance_unchanged": True,
        "candidate_execution_eligible": True,
        "formal_20x30_required": True,
        "adopted": False,
        "adoption_eligible": False,
        "simulation_acceptance_eligible": False,
        "hardware_deployment": "PROHIBITED",
    }


def _validate_h2_component_selection_evidence(
    path: Path = H2_COMPONENT_SELECTION_EVIDENCE_PATH,
) -> dict[str, Any]:
    """Validate the H2 component bundle without promoting combined acceptance.

    The allowlisted artifact binds the new straight phase/rate component and
    the decoupled exit-only recovery cap.  It is sufficient to execute the
    candidate by default, but deliberately cannot satisfy combined 5x15,
    formal 20x30, adoption, package, or hardware gates.
    """

    resolved = path.resolve()
    if resolved != H2_COMPONENT_SELECTION_EVIDENCE_PATH:
        raise ValueError("H2 component evidence path must remain pinned")
    if not resolved.is_file():
        raise FileNotFoundError(
            f"missing formal-candidate selection evidence: {resolved}"
        )
    digest = sha256_file(resolved)
    if (
        digest != H2_COMPONENT_SELECTION_EVIDENCE_SHA256
    ):
        raise ValueError("H2 component evidence hash mismatch")
    payload = _load_strict_json_object(resolved, "H2 component evidence")

    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("artifact_kind")
        != "openduckmini_h2_integrated_straight_and_exit_recovery_transition_qualification"
        or payload.get("status") != "DIAGNOSTIC_20X9_PASS_NOT_ADOPTED"
        or payload.get("simulation_only") is not True
        or payload.get("adoption_status") != "NOT_ADOPTED_PENDING_CENTRAL_INTEGRATION"
        or payload.get("central_contract_package_or_runtime_modified_by_this_run")
        is not False
        or payload.get("hardware_deployment") != "PROHIBITED"
    ):
        raise ValueError("formal-candidate H2 component top-level status mismatch")

    configuration = payload.get("configuration")
    selection = payload.get("selection")
    qualification = payload.get("qualification")
    provenance = payload.get("provenance")
    handoff = payload.get("handoff")
    if not all(
        isinstance(value, Mapping)
        for value in (configuration, selection, qualification, provenance, handoff)
    ):
        raise ValueError("formal-candidate H2 component metadata is incomplete")

    expected_configuration = {
        "formal_master_seed": FORMAL_CANDIDATE_MASTER_SEED,
        "moving_seconds": 30.0,
        "stand_seconds": 5.0,
        "warmup_seconds": 1.5,
        "initial_joint_noise_scale": 1.0,
        "initial_base_speed_mps": 0.1,
        "target_margin_rad": LEG_TARGET_MARGIN_RAD,
        "target_slew_rate_rad_per_s": TARGET_SLEW_LIMIT_RAD_PER_S,
        "reverse_endpoint_mps": CURRENT_FORMAL_REVERSE_ENDPOINT_MPS,
        "profile_extra_upper_margin_rad_all_backward_families": (
            FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
        ),
        "profile_left_knee_upper_target_rad": (
            FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_UPPER_TARGET_RAD
        ),
        "exit_recovery_extra_upper_margin_rad": (
            H2_SUPERSEDED_RECOVERY_EXTRA_UPPER_MARGIN_RAD
        ),
        "exit_recovery_left_knee_upper_target_rad": (
            H2_SUPERSEDED_RECOVERY_UPPER_TARGET_RAD
        ),
        "exit_recovery_hold_control_ticks": H2_SUPERSEDED_RECOVERY_HOLD_TICKS,
        "exit_recovery_hold_seconds": BACKWARD_EXIT_RECOVERY_HOLD_SECONDS,
        "exit_recovery_release": "instant_after_hold",
        "straight_phase_rate_rad_per_control_tick": 1.965946885259867,
    }
    for key, expected in expected_configuration.items():
        if configuration.get(key) != expected:
            raise ValueError(f"formal-candidate H2 configuration mismatch: {key}")
    if (
        dict(configuration.get("phase_preincrement_indices", {}))
        != dict(FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES)
        or configuration.get("transition_seeds")
        != [22_260_808 + index for index in range(20)]
    ):
        raise ValueError("formal-candidate H2 phase/seed configuration mismatch")

    selected = selection.get("selected_candidate")
    coupled_rejected = selection.get("coupled_profile_cap_rejected")
    focus = selection.get("focus5_comparisons")
    if (
        not isinstance(selected, Mapping)
        or selected.get("profile_cap_held_at_h1_value") is not True
        or selected.get("profile_extra_upper_margin_rad")
        != FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
        or selected.get("profile_upper_target_rad")
        != FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_UPPER_TARGET_RAD
        or selected.get("recovery_extra_upper_margin_rad")
        != H2_SUPERSEDED_RECOVERY_EXTRA_UPPER_MARGIN_RAD
        or selected.get("recovery_upper_target_rad")
        != H2_SUPERSEDED_RECOVERY_UPPER_TARGET_RAD
        or selected.get("recovery_hold_ticks")
        != H2_SUPERSEDED_RECOVERY_HOLD_TICKS
        or selected.get("recovery_hold_seconds")
        != H2_SUPERSEDED_RECOVERY_HOLD_SECONDS
        or selected.get("recovery_release") != "instant_after_hold"
        or not isinstance(coupled_rejected, Mapping)
        or coupled_rejected.get("candidate_extra_upper_margin_rad") != 0.01625
        or coupled_rejected.get("fall_count") != 1
        or coupled_rejected.get("qpos_limit_violation_samples") != 0
        or not isinstance(focus, list)
        or len(focus) != 3
        or [item.get("passed") for item in focus] != [False, False, True]
        or [item.get("qpos_limit_violation_samples") for item in focus[:2]]
        != [4, 4]
    ):
        raise ValueError("formal-candidate H2 cap separation/selection mismatch")

    result = qualification.get("result")
    expected_summary = {
        "episode_count": 20,
        "passed_episode_count": 20,
        "segment_count": 180,
        "passed_segment_count": 180,
        "fall_count": 0,
        "qpos_limit_violation_samples": 0,
        "expected_physics_substeps": 1_450_000,
        "completed_physics_substeps": 1_450_000,
        "audited_physics_substeps": 1_450_000,
        "contact_samples": 1_450_000,
        "target_limit_margin_slew_violation_count": 0,
        "maximum_head_action_or_target_peak": 0.0,
        "route_violation_segment_count": 0,
        "motion_contact_violation_segment_count": 0,
        "phase_entry_event_count": 60,
        "recovery_exit_event_count": 60,
        "recovery_active_tick_count": 780,
    }
    if (
        qualification.get("passed") is not True
        or qualification.get("exact_scale_passed") is not True
        or not isinstance(result, Mapping)
        or result.get("passed") is not True
        or result.get("all_physics_and_contact_counts_exact") is not True
    ):
        raise ValueError("formal-candidate H2 qualification did not pass")
    for key, expected in expected_summary.items():
        if result.get(key) != expected:
            raise ValueError(f"formal-candidate H2 result mismatch: {key}")
    if (
        result.get("maximum_left_knee_qpos_rad") != 0.4740208521265097
        or result.get("minimum_left_knee_safe_upper_margin_rad")
        != 0.001513147873490328
    ):
        raise ValueError("formal-candidate H2 left-knee audit mismatch")

    episodes = result.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 20:
        raise ValueError("formal-candidate H2 episode count mismatch")
    segments = [
        segment
        for episode in episodes
        if isinstance(episode, Mapping)
        for segment in episode.get("segments", ())
        if isinstance(segment, Mapping)
    ]
    if (
        len(segments) != 180
        or not all(episode.get("passed") is True for episode in episodes)
        or not all(
            segment.get("passed") is True
            and segment.get("completed") is True
            and segment.get("fell") is False
            and isinstance(segment.get("hard_checks"), Mapping)
            and all(value is True for value in segment["hard_checks"].values())
            for segment in segments
        )
    ):
        raise ValueError("formal-candidate H2 per-segment hard gates failed")

    def require_embedded_file(
        record: Any, expected_path: Path, expected_sha256: str, label: str
    ) -> None:
        if (
            not isinstance(record, Mapping)
            or not isinstance(record.get("path"), str)
            or not _portable_workspace_path_matches(record["path"], expected_path)
            or record.get("sha256") != expected_sha256
            or not expected_path.is_file()
            or sha256_file(expected_path) != expected_sha256
        ):
            raise ValueError(f"formal-candidate H2 dependency mismatch: {label}")

    profile_records = provenance.get("profiles")
    if not isinstance(profile_records, Mapping):
        raise ValueError("formal-candidate H2 profile provenance is missing")
    for label in ("straight", "left", "right"):
        require_embedded_file(
            profile_records.get(label),
            FORMAL_CANDIDATE_PROFILE_PATHS[label],
            FORMAL_CANDIDATE_PROFILE_SHA256S[label],
            f"profile_{label}",
        )
    require_embedded_file(
        provenance.get("historical_h1_failure_evidence"),
        HISTORICAL_FAILED_FORMAL_CANDIDATE_EVIDENCE_PATH,
        HISTORICAL_FAILED_FORMAL_CANDIDATE_EVIDENCE_SHA256,
        "historical_failed_formal_20x30",
    )
    straight_records = provenance.get("straight_selection_evidence")
    expected_straight_dependencies = (
        (H1_STRAIGHT_20X30_EVIDENCE_PATH, H1_STRAIGHT_20X30_EVIDENCE_SHA256),
        (
            H1_TRANSITION_PREFIX_20SEED_EVIDENCE_PATH,
            H1_TRANSITION_PREFIX_20SEED_EVIDENCE_SHA256,
        ),
        (
            H1_REJECTED_COUPLED_CAP_EVIDENCE_PATH,
            H1_REJECTED_COUPLED_CAP_EVIDENCE_SHA256,
        ),
    )
    if not isinstance(straight_records, list) or len(straight_records) != 3:
        raise ValueError("formal-candidate H2 straight evidence set mismatch")
    for index, (expected_path, expected_sha256) in enumerate(
        expected_straight_dependencies
    ):
        require_embedded_file(
            straight_records[index], expected_path, expected_sha256, f"straight_{index}"
        )
    require_embedded_file(
        provenance.get("policy"),
        WORKSPACE_ROOT
        / ".openduck_runtime_source_review"
        / "calibrated_hybrid_policy_v22.onnx",
        BASE_V22_POLICY_SHA256,
        "base_v22_policy",
    )

    historical = _load_strict_json_object(
        HISTORICAL_FAILED_FORMAL_CANDIDATE_EVIDENCE_PATH,
        "historical failed formal candidate",
    )
    historical_failures = [
        segment
        for suite in historical.get("suites", {}).values()
        if isinstance(suite, Mapping)
        for episode in suite.get("acceptance", {}).get("episode_checks", ())
        if isinstance(episode, Mapping)
        for segment in episode.get("segments", ())
        if isinstance(segment, Mapping) and segment.get("passed") is False
    ]
    if (
        historical.get("simulation_suite_acceptance_passed") is not False
        or historical.get("simulation_acceptance_passed") is not False
        or len(historical_failures) != 6
        or {str(item.get("name", "")) for item in historical_failures}
        != {
            "reverse",
            "transition_reverse",
            "transition_stand_after_reverse",
            "transition_stand_after_reverse_turn_left",
        }
        or historical.get("hardware_gate", {}).get("status") != "PROHIBITED"
    ):
        raise ValueError("historical failed formal candidate is not pinned as rejected")

    straight_payload = _load_strict_json_object(
        H1_STRAIGHT_20X30_EVIDENCE_PATH, "H1 straight 20x30 evidence"
    )
    straight_summary = next(
        (
            item
            for item in straight_payload.get("ranking", ())
            if isinstance(item, Mapping)
            and item.get("candidate_id") == "b7b7f61e3eecf47c"
        ),
        None,
    )
    prefix_payload = _load_strict_json_object(
        H1_TRANSITION_PREFIX_20SEED_EVIDENCE_PATH,
        "H1 transition-prefix evidence",
    )
    prefix_summary = next(iter(prefix_payload.get("ranking", ())), None)
    rejected_payload = _load_strict_json_object(
        H1_REJECTED_COUPLED_CAP_EVIDENCE_PATH,
        "H1 rejected coupled-cap evidence",
    )
    rejected_summary = next(iter(rejected_payload.get("ranking", ())), None)
    if (
        not isinstance(straight_summary, Mapping)
        or straight_summary.get("central_segment_count") != 20
        or straight_summary.get("central_segment_acceptance_count") != 20
        or straight_summary.get("fall_count") != 0
        or straight_summary.get("qpos_violation_samples") != 0
        or straight_summary.get("audited_physics_substeps") != 300_000
        or not isinstance(prefix_summary, Mapping)
        or prefix_summary.get("central_segment_count") != 100
        or prefix_summary.get("central_segment_acceptance_count") != 100
        or prefix_summary.get("fall_count") != 0
        or prefix_summary.get("qpos_violation_samples") != 0
        or prefix_summary.get("audited_physics_substeps") != 750_000
        or not isinstance(rejected_summary, Mapping)
        or rejected_summary.get("central_segment_acceptance_count") != 19
        or rejected_summary.get("fall_count") != 1
    ):
        raise ValueError("formal-candidate H2 straight component evidence mismatch")

    providers = provenance.get("onnx_providers")
    if (
        provenance.get("source_closure_unchanged") is not True
        or not isinstance(providers, Mapping)
        or set(providers) != set(REQUIRED_POLICY_ROLES)
        or any(value != ["CPUExecutionProvider"] for value in providers.values())
        or handoff.get("recommended_for_central_integration") is not True
        or handoff.get("package_remains_blocked") is not True
        or handoff.get("requires_full_frozen_20x30_all_suite_requalification")
        is not True
        or handoff.get("hardware_deployment") != "PROHIBITED"
    ):
        raise ValueError("formal-candidate H2 provenance/handoff mismatch")

    return {
        "path": str(resolved),
        "sha256": digest,
        "hash_allowlisted_for_candidate_selection": True,
        "hash_allowlisted_for_adoption": False,
        "schema_validated": True,
        "all_json_numbers_finite": True,
        "status": H2_COMPONENT_STATUS,
        "selection_scope": "h2_components_phase744_rate105_recovery0175",
        "master_seed": FORMAL_CANDIDATE_MASTER_SEED,
        "profile_sha256s": dict(FORMAL_CANDIDATE_PROFILE_SHA256S),
        "policy_sha256": BASE_V22_POLICY_SHA256,
        "reverse_endpoint_mps": CURRENT_FORMAL_REVERSE_ENDPOINT_MPS,
        "phase_preincrement_indices": dict(
            FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES
        ),
        "profile_left_knee_cap": {
            "extra_upper_margin_rad": (
                FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
            ),
            "upper_target_rad": FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_UPPER_TARGET_RAD,
        },
        "backward_exit_recovery": {
            "enabled": True,
            "extra_upper_margin_rad": H2_SUPERSEDED_RECOVERY_EXTRA_UPPER_MARGIN_RAD,
            "hold_control_ticks": H2_SUPERSEDED_RECOVERY_HOLD_TICKS,
            "hold_seconds": H2_SUPERSEDED_RECOVERY_HOLD_SECONDS,
            "upper_target_rad": H2_SUPERSEDED_RECOVERY_UPPER_TARGET_RAD,
        },
        "component_transition_episode_count": 20,
        "component_transition_segment_pass_count": 180,
        "component_transition_physics_substep_count": 1_450_000,
        "component_transition_contact_sample_count": 1_450_000,
        "phase_entry_event_count": 60,
        "recovery_exit_event_count": 60,
        "recovery_active_tick_count": 780,
        "historical_failed_formal_20x30_sha256": (
            HISTORICAL_FAILED_FORMAL_CANDIDATE_EVIDENCE_SHA256
        ),
        "historical_failed_acceptance_count": 6,
        "straight_20x30_component_evidence_sha256": (
            H1_STRAIGHT_20X30_EVIDENCE_SHA256
        ),
        "transition_prefix_component_evidence_sha256": (
            H1_TRANSITION_PREFIX_20SEED_EVIDENCE_SHA256
        ),
        "candidate_execution_eligible": True,
        "combined_5x15_required": True,
        "combined_5x15_passed": False,
        "formal_20x30_required": True,
        "adopted": False,
        "adoption_eligible": False,
        "simulation_acceptance_eligible": False,
        "hardware_deployment": "PROHIBITED",
    }


def _validate_h2_5x15_selection_evidence(
    path: Path = H2_5X15_SELECTION_EVIDENCE_PATH,
) -> dict[str, Any]:
    """Strictly pin the passing H2 combined 5x15 candidate selection run.

    The selected artifact is a screening-scale run and therefore authorizes
    only the no-flag candidate defaults.  It cannot satisfy the independent
    20x30, adoption, simulation-release, package, or hardware gates.
    """

    resolved = path.resolve()
    if resolved != H2_5X15_SELECTION_EVIDENCE_PATH:
        raise ValueError("formal-candidate selection evidence path must remain pinned")
    if not resolved.is_file():
        raise FileNotFoundError(
            f"missing formal-candidate selection evidence: {resolved}"
        )
    digest = sha256_file(resolved)
    if (
        digest != H2_5X15_SELECTION_EVIDENCE_SHA256
        or digest not in H2_5X15_SELECTION_EVIDENCE_SHA256_ALLOWLIST
    ):
        raise ValueError("formal-candidate selection evidence hash mismatch")
    payload = _load_strict_json_object(resolved, "formal-candidate selection evidence")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("evaluator_id") != EVALUATOR_ID
        or payload.get("evaluation_mode")
        != "FORMAL_CANDIDATE_DEFAULT_PENDING_ADOPTION"
        or payload.get("simulation_suite_acceptance_passed") is not True
        or payload.get("simulation_acceptance_passed") is not False
    ):
        raise ValueError("formal-candidate H2 5x15 top-level status mismatch")

    component = _validate_h2_component_selection_evidence()
    embedded_component = payload.get("formal_candidate_selection_evidence")
    if not isinstance(embedded_component, Mapping):
        raise ValueError("formal-candidate H2 component lineage is missing")
    for key, expected in component.items():
        actual = embedded_component.get(key)
        if key == "path":
            if not isinstance(actual, str) or not _portable_workspace_path_matches(
                actual, H2_COMPONENT_SELECTION_EVIDENCE_PATH
            ):
                raise ValueError("formal-candidate H2 component path mismatch")
        elif actual != expected:
            raise ValueError(f"formal-candidate H2 component binding mismatch: {key}")

    configuration = payload.get("configuration")
    release = payload.get("release_qualification")
    phase_contract = payload.get("formal_reverse_phase_entry_contract")
    recovery_contract = payload.get("formal_backward_exit_recovery_contract")
    command_contract = payload.get("command_mapping_contract")
    execution_bundle = payload.get("formal_candidate_execution_bundle")
    if not all(
        isinstance(value, Mapping)
        for value in (
            configuration,
            release,
            phase_contract,
            recovery_contract,
            command_contract,
            execution_bundle,
        )
    ):
        raise ValueError("formal-candidate H2 5x15 metadata is incomplete")
    expected_configuration = {
        "seed": FORMAL_CANDIDATE_MASTER_SEED,
        "episodes": 5,
        "seconds": 15.0,
        "transition_seconds": 15.0,
        "transition_stand_seconds": 5.0,
        "warmup_seconds": 1.5,
        "initial_base_speed": 0.1,
        "initial_joint_noise_scale": 1.0,
        "backward_residual_scale": 0.0,
        "leg_target_margin_rad": LEG_TARGET_MARGIN_RAD,
        "target_slew_rate_rad_per_s": TARGET_SLEW_LIMIT_RAD_PER_S,
        "reset_noise_margin_rad": RESET_NOISE_MARGIN_RAD,
        "left_knee_extra_upper_margin_rad": (
            FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
        ),
        "left_knee_profile_upper_target_rad": (
            FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_UPPER_TARGET_RAD
        ),
        "formal_candidate_default": True,
        "formal_candidate_status": H2_COMPONENT_STATUS,
        "backward_exit_recovery_enabled": True,
        "diagnostic_unadopted_policy": False,
        "diagnostic_unadopted_reverse_profile": None,
        "diagnostic_unadopted_reverse_left_profile": None,
        "diagnostic_unadopted_reverse_right_profile": None,
        "diagnostic_unadopted_backward_exit_recovery": False,
        "diagnostic_unadopted_reverse_entry_phase_indices": None,
        "diagnostic_noncontract_safety": False,
        "policy_command_diagnostic_suite": False,
        "control_first_startup_required": True,
        "home_only_startup_precharge_used": False,
        "physics_steps_allowed_before_startup_control": 0,
        "guard_calls_per_control_tick": 1,
    }
    for key, expected in expected_configuration.items():
        if configuration.get(key) != expected:
            raise ValueError(f"formal-candidate H2 5x15 configuration mismatch: {key}")
    if dict(configuration.get("executed_reverse_entry_phase_indices", {})) != dict(
        FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES
    ):
        raise ValueError("formal-candidate H2 5x15 phase mapping mismatch")
    executed_paths = configuration.get("executed_reverse_profile_paths")
    if not isinstance(executed_paths, Mapping) or any(
        not isinstance(executed_paths.get(label), str)
        or not _portable_workspace_path_matches(
            str(executed_paths[label]), FORMAL_CANDIDATE_PROFILE_PATHS[label]
        )
        for label in ("straight", "left", "right")
    ):
        raise ValueError("formal-candidate H2 5x15 profile path mismatch")

    release_actual = release.get("actual")
    if (
        not isinstance(release_actual, Mapping)
        or release_actual.get("master_seed") != FORMAL_CANDIDATE_MASTER_SEED
        or release_actual.get("episodes") != 5
        or release_actual.get("seconds") != 15.0
        or release_actual.get("transition_seconds") != 15.0
        or release_actual.get("transition_stand_seconds") != 5.0
        or release.get("diagnostic_mode_disabled") is not True
        or release.get("master_seed_matches_recommendation") is not True
        or release.get("scale_matches_frozen_contract") is not False
        or release.get("release_qualification_eligible") is not False
        or release.get("screening_cannot_promote_release_or_adoption") is not True
        or release.get("status") != "SCREENING_CANDIDATE"
    ):
        raise ValueError("formal-candidate H2 5x15 screening gate mismatch")

    expected_bundle = {
        "passed": True,
        "status": H2_COMPONENT_STATUS,
        "candidate_selection_evidence_sha256": H2_COMPONENT_SELECTION_EVIDENCE_SHA256,
        "profile_sha256s": dict(FORMAL_CANDIDATE_PROFILE_SHA256S),
        "policy_sha256": BASE_V22_POLICY_SHA256,
        "policy_roles": list(REQUIRED_POLICY_ROLES),
        "reverse_endpoint_mps": CURRENT_FORMAL_REVERSE_ENDPOINT_MPS,
        "phase_preincrement_indices": dict(
            FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES
        ),
        "profile_left_knee_cap": {
            "extra_upper_margin_rad": (
                FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
            ),
            "upper_target_rad": FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_UPPER_TARGET_RAD,
        },
        "backward_exit_recovery": {
            "enabled": True,
            "extra_upper_margin_rad": H2_SUPERSEDED_RECOVERY_EXTRA_UPPER_MARGIN_RAD,
            "hold_control_ticks": H2_SUPERSEDED_RECOVERY_HOLD_TICKS,
            "hold_seconds": H2_SUPERSEDED_RECOVERY_HOLD_SECONDS,
            "upper_target_rad": H2_SUPERSEDED_RECOVERY_UPPER_TARGET_RAD,
        },
        "adopted": False,
        "adoption_eligible": False,
        "simulation_acceptance_eligible": False,
        "hardware_deployment": "PROHIBITED",
    }
    if dict(execution_bundle) != expected_bundle:
        raise ValueError("formal-candidate H2 component execution binding mismatch")

    phase_mapping = phase_contract.get("preincrement_phase_indices")
    recovery_runtime = recovery_contract.get("runtime_contract")
    if (
        phase_contract.get("enabled") is not True
        or phase_contract.get("enabled_by_default") is not True
        or phase_contract.get("diagnostic_only") is not False
        or phase_contract.get("status") != H2_COMPONENT_STATUS
        or phase_contract.get("current_formal_reverse_endpoint_mps")
        != CURRENT_FORMAL_REVERSE_ENDPOINT_MPS
        or not isinstance(phase_mapping, Mapping)
        or dict(phase_mapping) != dict(FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES)
        or recovery_contract.get("enabled") is not True
        or recovery_contract.get("enabled_by_default") is not True
        or recovery_contract.get("diagnostic_unadopted_only") is not False
        or recovery_contract.get("status") != H2_COMPONENT_STATUS
        or recovery_contract.get("selection_evidence_sha256")
        != H2_COMPONENT_SELECTION_EVIDENCE_SHA256
        or not isinstance(recovery_runtime, Mapping)
        or recovery_runtime.get("candidate_selection_evidence_sha256")
        != H2_COMPONENT_SELECTION_EVIDENCE_SHA256
        or recovery_runtime.get("extra_upper_margin_rad")
        != H2_SUPERSEDED_RECOVERY_EXTRA_UPPER_MARGIN_RAD
        or recovery_runtime.get("upper_target_rad")
        != H2_SUPERSEDED_RECOVERY_UPPER_TARGET_RAD
        or recovery_runtime.get("hold_control_ticks")
        != H2_SUPERSEDED_RECOVERY_HOLD_TICKS
        or recovery_runtime.get("hold_seconds")
        != H2_SUPERSEDED_RECOVERY_HOLD_SECONDS
        or recovery_runtime.get("release") != "instant_after_hold"
    ):
        raise ValueError("formal-candidate H2 phase/recovery contract mismatch")

    validation_gate = command_contract.get("validation_status_gate")
    expected_pending_names = {
        "reverse",
        "reverse_turn_left",
        "reverse_turn_right",
        "transition_reverse",
        "transition_reverse_turn_left",
        "transition_reverse_turn_right",
    }
    pending_records = (
        validation_gate.get("nonadoptable_cases", ())
        if isinstance(validation_gate, Mapping)
        else ()
    )
    if (
        not isinstance(validation_gate, Mapping)
        or validation_gate.get("passed") is not False
        or validation_gate.get("case_count") != 38
        or validation_gate.get("nonadoptable_case_count") != 6
        or {
            str(record.get("name", ""))
            for record in pending_records
            if isinstance(record, Mapping)
        }
        != expected_pending_names
        or any(
            not isinstance(record, Mapping)
            or record.get("validation_status") != H2_COMPONENT_STATUS
            for record in pending_records
        )
    ):
        raise ValueError("formal-candidate H2 command cases are not fail-closed")

    suites = payload.get("suites")
    expected_segment_counts = {"primitives": 35, "compounds": 30, "transitions": 125}
    if not isinstance(suites, Mapping) or set(suites) != set(expected_segment_counts):
        raise ValueError("formal-candidate H2 suite set mismatch")
    episodes: list[Mapping[str, Any]] = []
    segments: list[Mapping[str, Any]] = []
    accepted_segments: list[Mapping[str, Any]] = []
    reset_audits: list[Mapping[str, Any]] = []
    startup_audits: list[Mapping[str, Any]] = []
    recovery_state_audits: list[Mapping[str, Any]] = []
    for suite_name, expected_segment_count in expected_segment_counts.items():
        suite = suites[suite_name]
        suite_episodes = suite.get("episodes") if isinstance(suite, Mapping) else None
        acceptance = suite.get("acceptance") if isinstance(suite, Mapping) else None
        if (
            not isinstance(suite_episodes, list)
            or len(suite_episodes) != 5
            or not all(isinstance(item, Mapping) for item in suite_episodes)
            or not isinstance(acceptance, Mapping)
            or acceptance.get("passed") is not True
        ):
            raise ValueError(f"formal-candidate H2 {suite_name} acceptance mismatch")
        suite_segments = [
            segment
            for episode in suite_episodes
            for segment in episode.get("segments", ())
            if isinstance(segment, Mapping)
        ]
        suite_checks = acceptance.get("episode_checks")
        if (
            len(suite_segments) != expected_segment_count
            or not isinstance(suite_checks, list)
            or len(suite_checks) != 5
            or not all(
                isinstance(item, Mapping) and item.get("passed") is True
                for item in suite_checks
            )
        ):
            raise ValueError(f"formal-candidate H2 {suite_name} segment checks failed")
        episodes.extend(suite_episodes)
        segments.extend(suite_segments)
        for check in suite_checks:
            accepted_segments.extend(
                item for item in check.get("segments", ()) if isinstance(item, Mapping)
            )
            reset_audits.extend(
                item
                for item in check.get("reset_qpos_audits", ())
                if isinstance(item, Mapping)
            )
            startup_audits.extend(
                item
                for item in check.get("control_first_startup_audits", ())
                if isinstance(item, Mapping)
            )
            recovery_state_audits.extend(
                item
                for item in check.get("backward_exit_recovery_audits", ())
                if isinstance(item, Mapping)
            )
    if (
        len(episodes) != 15
        or len(segments) != 190
        or len(accepted_segments) != 190
        or not all(item.get("passed") is True for item in accepted_segments)
        or not all(
            episode.get("fell") is False
            and episode.get("completed_segment_count")
            == episode.get("requested_segment_count")
            for episode in episodes
        )
        or not all(
            segment.get("completed") is True
            and segment.get("fell") is False
            and segment.get("completed_physics_substeps")
            == segment.get("expected_physics_substeps")
            for segment in segments
        )
    ):
        raise ValueError("formal-candidate H2 episode/segment completion mismatch")

    safety_zero_fields = (
        "applied_target_limit_violations",
        "desired_target_margin_violations",
        "nonfinite_sample_count",
        "preclip_target_limit_violations",
        "qpos_limit_violations",
        "target_slew_violations",
        "unauthorized_applied_target_margin_violations",
        "maximum_applied_target_excess_rad",
        "maximum_desired_target_margin_excess_rad",
        "maximum_preclip_target_excess_rad",
        "maximum_qpos_excess_rad",
        "applied_head_action_peak",
        "head_target_peak_rad",
    )
    physics_zero_fields = (
        "height_fall_samples",
        "upright_fall_samples",
        "nonfinite_full_qpos_samples",
        "nonfinite_full_qvel_samples",
        "nonfinite_leg_qpos_samples",
        "nonfinite_pose_samples",
        "nonfinite_state_samples",
        "qpos_limit_violations",
        "maximum_qpos_excess_rad",
    )
    physics_sample_count = 0
    contact_sample_count = 0
    leg_qpos_sample_count = 0
    control_sample_count = 0
    phase_audits: list[Mapping[str, Any]] = []
    recovery_segment_audits: list[Mapping[str, Any]] = []
    for segment in segments:
        safety = segment.get("safety_audit")
        physics = segment.get("physics_substep_audit")
        routing = segment.get("routing")
        recovery = segment.get("backward_exit_recovery_audit")
        if not all(isinstance(value, Mapping) for value in (safety, physics, routing, recovery)):
            raise ValueError("formal-candidate H2 segment audit is incomplete")
        if any(safety.get(field) != 0 for field in safety_zero_fields):
            raise ValueError("formal-candidate H2 target/qpos safety audit failed")
        if (
            safety.get("leg_target_margin_rad") != LEG_TARGET_MARGIN_RAD
            or safety.get("target_slew_limit_rad_per_s")
            != TARGET_SLEW_LIMIT_RAD_PER_S
            or safety.get("maximum_target_slew_rate_rad_per_s")
            > TARGET_SLEW_LIMIT_RAD_PER_S + 2e-15
            or any(physics.get(field) != 0 for field in physics_zero_fields)
            or physics.get("fall_or_nonfinite_detected") is not False
            or physics.get("contact_sample_count_matches_sample_count") is not True
            or physics.get("contact_sample_count") != physics.get("sample_count")
            or routing.get("command_clip_events") != 0
            or routing.get("prohibited_expert_steps") != 0
            or recovery.get("passed") is not True
        ):
            raise ValueError("formal-candidate H2 physics/routing/recovery audit failed")
        phase = routing.get("reverse_entry_phase")
        if not isinstance(phase, Mapping) or phase.get("passed") is not True:
            raise ValueError("formal-candidate H2 phase-entry audit failed")
        physics_sample_count += int(physics["sample_count"])
        contact_sample_count += int(physics["contact_sample_count"])
        leg_qpos_sample_count += int(physics["leg_joint_sample_count"])
        control_sample_count += int(safety["sample_count"])
        phase_audits.append(phase)
        recovery_segment_audits.append(recovery)
    if (
        physics_sample_count != 1_100_000
        or contact_sample_count != 1_100_000
        or leg_qpos_sample_count != 11_000_000
        or control_sample_count != 110_000
    ):
        raise ValueError("formal-candidate H2 audited sample totals mismatch")

    phase_events = [
        event
        for audit in phase_audits
        for event in audit.get("events", ())
        if isinstance(event, Mapping)
    ]
    phase_counts = {
        expert: sum(event.get("current_expert") == expert for event in phase_events)
        for expert in FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES
    }
    if (
        len(phase_audits) != 190
        or sum(int(audit.get("event_count", -1)) for audit in phase_audits) != 30
        or sum(
            int(audit.get("backward_feedforward_entry_count", -1))
            for audit in phase_audits
        )
        != 30
        or sum(
            int(audit.get("within_backward_family_active_switch_count", -1))
            for audit in phase_audits
        )
        != 0
        or phase_counts != {expert: 10 for expert in phase_counts}
        or any(
            event.get("reset_preincrement_phase_index")
            != FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES[
                str(event.get("current_expert", ""))
            ]
            for event in phase_events
        )
    ):
        raise ValueError("formal-candidate H2 phase-entry event totals mismatch")
    if (
        len(recovery_segment_audits) != 190
        or sum(int(audit.get("exit_event_count", -1)) for audit in recovery_segment_audits)
        != 15
        or sum(int(audit.get("active_tick_count", -1)) for audit in recovery_segment_audits)
        != 195
        or sum(int(audit.get("cap_violation_count", -1)) for audit in recovery_segment_audits)
        != 0
        or any(audit.get("remaining_ticks_after_segment") != 0 for audit in recovery_segment_audits)
        or sum(int(audit.get("sample_count", -1)) for audit in recovery_segment_audits)
        != 110_000
        or sum(
            int(audit.get("final_guard_call_count", -1))
            for audit in recovery_segment_audits
        )
        != 110_000
    ):
        raise ValueError("formal-candidate H2 recovery segment totals mismatch")
    if (
        len(reset_audits) != 70
        or len(startup_audits) != 70
        or len(recovery_state_audits) != 70
        or not all(audit.get("passed") is True for audit in reset_audits)
        or not all(audit.get("passed") is True for audit in startup_audits)
        or not all(audit.get("passed") is True for audit in recovery_state_audits)
        or sum(int(audit.get("physical_safe_limit_violations", -1)) for audit in reset_audits)
        != 0
        or sum(int(audit.get("noise_margin_violations", -1)) for audit in reset_audits)
        != 0
        or any(audit.get("head_qpos_peak_rad") != 0.0 for audit in reset_audits)
        or any(
            audit.get("control_applied_before_first_physics_step") is not True
            or audit.get("exactly_one_guard_call_for_first_tick") is not True
            or audit.get("physics_steps_before_control") != 0
            or audit.get("guard_calls_for_first_tick") != 1
            or audit.get("applied_target_physical_safe_violations") != 0
            for audit in startup_audits
        )
        or sum(int(audit.get("exit_event_count", -1)) for audit in recovery_state_audits)
        != 15
        or sum(int(audit.get("active_tick_count", -1)) for audit in recovery_state_audits)
        != 195
        or sum(int(audit.get("completed_event_count", -1)) for audit in recovery_state_audits)
        != 15
        or sum(int(audit.get("cap_violation_count", -1)) for audit in recovery_state_audits)
        != 0
        or sum(int(audit.get("remaining_ticks", -1)) for audit in recovery_state_audits)
        != 0
        or sum(int(audit.get("control_tick_count", -1)) for audit in recovery_state_audits)
        != 110_000
        or sum(int(audit.get("final_guard_call_count", -1)) for audit in recovery_state_audits)
        != 110_000
    ):
        raise ValueError("formal-candidate H2 reset/startup/recovery acceptance mismatch")

    reverse_profiles = payload.get("reverse_profile_evidence")
    policy_provenance = payload.get("policy_provenance")
    executed_profiles = (
        reverse_profiles.get("executed_profiles")
        if isinstance(reverse_profiles, Mapping)
        else None
    )
    policy_roles = policy_provenance.get("roles") if isinstance(policy_provenance, Mapping) else None
    if (
        not isinstance(executed_profiles, Mapping)
        or set(executed_profiles) != set(FORMAL_CANDIDATE_PROFILE_SHA256S)
        or {
            label: str(record.get("sha256", ""))
            for label, record in executed_profiles.items()
            if isinstance(record, Mapping)
        }
        != dict(FORMAL_CANDIDATE_PROFILE_SHA256S)
        or not all(
            isinstance(record, Mapping)
            and record.get("schema_validated") is True
            and record.get("formal_candidate") is True
            and record.get("candidate_hash_allowlisted") is True
            and record.get("adopted") is False
            and record.get("adoption_eligible") is False
            for record in executed_profiles.values()
        )
        or not isinstance(policy_roles, Mapping)
        or set(policy_roles) != set(REQUIRED_POLICY_ROLES)
        or policy_provenance.get("all_roles_allowlisted") is not True
        or any(
            not isinstance(record, Mapping)
            or record.get("sha256") != BASE_V22_POLICY_SHA256
            or record.get("formal_base_v22_allowlisted") is not True
            for record in policy_roles.values()
        )
    ):
        raise ValueError("formal-candidate H2 profile/policy hash bank mismatch")

    provenance = payload.get("runtime_dependency_provenance")
    pre = provenance.get("pre_import") if isinstance(provenance, Mapping) else None
    post = provenance.get("post_evaluation") if isinstance(provenance, Mapping) else None
    runtime_data_pre = (
        provenance.get("runtime_model_and_data_pre_evaluation")
        if isinstance(provenance, Mapping)
        else None
    )
    if not all(isinstance(value, Mapping) for value in (provenance, pre, post, runtime_data_pre)):
        raise ValueError("formal-candidate H2 pre/post provenance is incomplete")
    closure_contract = {
        "exp004_source_and_contract_snapshot": (
            "7d49fe3d537d66e8ab80c75d46856ba4a93202cf2221ed681ca820134fc538bb",
            9,
        ),
        "external_hard_allowlisted_source_closure": (
            "a40d4920049b349334b6d5567859fc2f7533d8fe6648f5127f0c7e4ce54dc098",
            4,
        ),
        "hard_allowlisted_runtime_binary_closure": (
            "4e382762ffe85e33ba4839969088fed6f27cae1b35f6e0247a7d1b18937abe5f",
            5,
        ),
    }
    for label, (root_sha256, dependency_count) in closure_contract.items():
        pre_closure = pre.get(label)
        post_closure = post.get(label)
        if (
            not isinstance(pre_closure, Mapping)
            or not isinstance(post_closure, Mapping)
            or dict(pre_closure) != dict(post_closure)
            or pre_closure.get("root_sha256") != root_sha256
            or pre_closure.get("dependency_count") != dependency_count
            or pre_closure.get("all_hashes_verified") is not True
        ):
            raise ValueError(f"formal-candidate H2 provenance mismatch: {label}")
    runtime_data_post = post.get("runtime_model_and_data_closure")
    runtime_entries = runtime_data_pre.get("entries")
    if (
        provenance.get("verified") is not True
        or provenance.get("pre_post_source_and_data_hashes_unchanged") is not True
        or provenance.get("all_onnx_sessions_cpu_only_verified") is not True
        or not isinstance(runtime_data_post, Mapping)
        or dict(runtime_data_pre) != dict(runtime_data_post)
        or runtime_data_pre.get("root_sha256")
        != "b1cc8c5f85b4911160cdd23a12a432451f5dd63133fc66695e194328c3bab7b3"
        or runtime_data_pre.get("dependency_count") != 52
        or runtime_data_pre.get("all_hashes_verified") is not True
        or not isinstance(runtime_entries, Mapping)
        or runtime_entries.get("formal_candidate_selection_evidence", {}).get("sha256")
        != H2_COMPONENT_SELECTION_EVIDENCE_SHA256
        or any(
            providers != ["CPUExecutionProvider"]
            for providers in provenance.get("onnx_session_execution_providers", {}).values()
        )
    ):
        raise ValueError("formal-candidate H2 runtime data/provider provenance mismatch")

    hardware = payload.get("hardware_gate")
    adoption = payload.get("adoption_contract")
    reverse_adoption = payload.get("reverse_profile_adoption")
    if (
        not isinstance(hardware, Mapping)
        or hardware.get("status") != "PROHIBITED"
        or hardware.get("hardware_deployment_allowed") is not False
        or not isinstance(adoption, Mapping)
        or adoption.get("passed") is not False
        or adoption.get("reverse_profile_adopted") is not False
        or adoption.get("formal_candidate_pending") is not True
        or not isinstance(reverse_adoption, Mapping)
        or reverse_adoption.get("passed") is not False
        or reverse_adoption.get("status") != "FAIL_CLOSED"
        or any(reverse_adoption.get("evidence_hash_allowlists", {}).values())
    ):
        raise ValueError("formal-candidate H2 adoption/hardware gate mismatch")

    return {
        "path": str(resolved),
        "sha256": digest,
        "hash_allowlisted_for_candidate_selection": True,
        "hash_allowlisted_for_adoption": False,
        "schema_validated": True,
        "all_json_numbers_finite": True,
        "status": H2_5X15_SELECTION_STATUS,
        "selection_scope": "h2_combined_phase744_rate105_recovery0175_5x15",
        "master_seed": FORMAL_CANDIDATE_MASTER_SEED,
        "profile_sha256s": dict(FORMAL_CANDIDATE_PROFILE_SHA256S),
        "policy_sha256": BASE_V22_POLICY_SHA256,
        "reverse_endpoint_mps": CURRENT_FORMAL_REVERSE_ENDPOINT_MPS,
        "phase_preincrement_indices": dict(FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES),
        "profile_left_knee_cap": {
            "extra_upper_margin_rad": FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD,
            "upper_target_rad": FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_UPPER_TARGET_RAD,
        },
        "backward_exit_recovery": {
            "enabled": True,
            "extra_upper_margin_rad": H2_SUPERSEDED_RECOVERY_EXTRA_UPPER_MARGIN_RAD,
            "hold_control_ticks": H2_SUPERSEDED_RECOVERY_HOLD_TICKS,
            "hold_seconds": H2_SUPERSEDED_RECOVERY_HOLD_SECONDS,
            "upper_target_rad": H2_SUPERSEDED_RECOVERY_UPPER_TARGET_RAD,
        },
        "component_selection_evidence_sha256": H2_COMPONENT_SELECTION_EVIDENCE_SHA256,
        "suite_episode_count": len(episodes),
        "segment_pass_count": len(accepted_segments),
        "physics_substep_count": physics_sample_count,
        "contact_sample_count": contact_sample_count,
        "phase_entry_event_count": len(phase_events),
        "recovery_exit_event_count": 15,
        "recovery_active_tick_count": 195,
        "pre_post_provenance_unchanged": True,
        "candidate_execution_eligible": True,
        "combined_5x15_required": False,
        "combined_5x15_passed": True,
        "formal_20x30_required": True,
        "adopted": False,
        "adoption_eligible": False,
        "simulation_acceptance_eligible": False,
        "hardware_deployment": "PROHIBITED",
    }


def validate_h3_fast_exit_safety_evidence(
    path: Path = H3_FAST_EXIT_SAFETY_EVIDENCE_PATH,
) -> dict[str, Any]:
    """Validate H3's fast stress record as safety-only component evidence.

    The source artifact intentionally failed 11 short-horizon motion checks.
    Those failures are part of the pinned contract; only its independently
    recomputed safety subset may select the pending H3 runtime candidate.
    """

    resolved = path.resolve()
    if resolved != H3_FAST_EXIT_SAFETY_EVIDENCE_PATH:
        raise ValueError("H3 fast-exit safety evidence path must remain pinned")
    if not resolved.is_file():
        raise FileNotFoundError(f"missing H3 fast-exit safety evidence: {resolved}")
    if resolved.stat().st_size != 5_906_295:
        raise ValueError("H3 fast-exit safety evidence size mismatch")
    digest = sha256_file(resolved)
    if (
        digest != H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256
        or digest not in H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256_ALLOWLIST
    ):
        raise ValueError("H3 fast-exit safety evidence hash mismatch")
    payload = _load_strict_json_object(resolved, "H3 fast-exit safety evidence")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_kind")
        != "openduckmini_h2_aggressive_short_transition_recovery_confirmation"
        or payload.get("status") != "DIAGNOSTIC_FAIL"
        or payload.get("passed") is not False
        or payload.get("central_suite_acceptance") is None
        or payload.get("simulation_only") is not True
        or payload.get("adoption_status") != "NOT_ADOPTED_DIAGNOSTIC_ONLY"
        or payload.get("release_evidence_eligible") is not False
        or payload.get("hardware_deployment") != "PROHIBITED"
        or payload.get("central_evaluator_contract_package_docs_modified")
        is not False
    ):
        raise ValueError("H3 fast-exit source status must remain diagnostic-fail")

    configuration = payload.get("configuration")
    summary = payload.get("summary")
    episodes = payload.get("episodes")
    central = payload.get("central_suite_acceptance")
    provenance = payload.get("provenance")
    if not all(
        isinstance(value, Mapping)
        for value in (configuration, summary, central, provenance)
    ) or not isinstance(episodes, list):
        raise ValueError("H3 fast-exit safety evidence structure is incomplete")
    recovery = configuration.get("recovery_candidate")
    schedule = configuration.get("full_transition_schedule")
    expected_seeds = list(range(22_260_808, 22_260_828))
    if (
        configuration.get("episodes") != 20
        or configuration.get("master_seed") != FORMAL_CANDIDATE_MASTER_SEED
        or configuration.get("transition_seeds") != expected_seeds
        or configuration.get("moving_seconds") != 2.0
        or configuration.get("stand_seconds") != 1.0
        or configuration.get("warmup_seconds") != 0.5
        or configuration.get("initial_joint_noise_scale") != 1.0
        or configuration.get("initial_base_speed_mps") != 0.1
        or configuration.get("backward_residual_scale") != 0.0
        or configuration.get("target_margin_rad") != LEG_TARGET_MARGIN_RAD
        or configuration.get("target_slew_rate_rad_per_s")
        != TARGET_SLEW_LIMIT_RAD_PER_S
        or configuration.get("phase_preincrement_indices")
        != dict(FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES)
        or configuration.get("profile_extra_upper_margin_rad_unchanged")
        != FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
        or not isinstance(recovery, Mapping)
        or recovery.get("profile_cap_held_at_h1_value") is not True
        or recovery.get("profile_extra_upper_margin_rad")
        != FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
        or recovery.get("profile_upper_target_rad")
        != FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_UPPER_TARGET_RAD
        or recovery.get("recovery_extra_upper_margin_rad")
        != BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD
        or recovery.get("recovery_upper_target_rad")
        != BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD
        or recovery.get("recovery_hold_ticks")
        != BACKWARD_EXIT_RECOVERY_HOLD_TICKS
        or recovery.get("recovery_hold_seconds")
        != BACKWARD_EXIT_RECOVERY_HOLD_SECONDS
        or recovery.get("recovery_release") != "instant_after_hold"
        or not isinstance(schedule, list)
        or len(schedule) != len(TRANSITION_CASES)
        or [item.get("name") for item in schedule]
        != [case.name for case in TRANSITION_CASES]
    ):
        raise ValueError("H3 fast-exit configuration mismatch")

    if len(episodes) != 20:
        raise ValueError("H3 fast-exit evidence requires exactly 20 episodes")
    raw_segments: list[Mapping[str, Any]] = []
    phase_events = 0
    recovery_exits = 0
    recovery_ticks = 0
    left_knee_maximum = -np.inf
    for episode_index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            raise ValueError("H3 fast-exit episode must be an object")
        segments = episode.get("segments")
        state_acceptance = episode.get("backward_exit_recovery_state_acceptance")
        if (
            episode.get("seed") != expected_seeds[episode_index]
            or not isinstance(segments, list)
            or len(segments) != len(TRANSITION_CASES)
            or [segment.get("name") for segment in segments]
            != [case.name for case in TRANSITION_CASES]
            or not isinstance(state_acceptance, Mapping)
            or state_acceptance.get("passed") is not True
            or episode.get("reset_qpos_audit", {}).get("passed") is not True
            or episode.get("control_first_startup_audit", {}).get("passed")
            is not True
        ):
            raise ValueError("H3 fast-exit episode audit/order mismatch")
        raw_segments.extend(segments)
        state_audit = episode.get("backward_exit_recovery_state_audit")
        if (
            not isinstance(state_audit, Mapping)
            or state_audit.get("exit_event_count") != 3
            or state_audit.get("active_tick_count") != 39
            or state_audit.get("cap_violation_count") != 0
            or state_audit.get("passed") is not True
        ):
            raise ValueError("H3 fast-exit recovery state mismatch")
        recovery_exits += int(state_audit["exit_event_count"])
        recovery_ticks += int(state_audit["active_tick_count"])

    expected_substeps = completed_substeps = audited_substeps = contact_samples = 0
    for segment in raw_segments:
        if not isinstance(segment, Mapping):
            raise ValueError("H3 fast-exit segment must be an object")
        target = segment.get("target_head_slew_audit")
        physics = segment.get("physics_substep_audit")
        routing = segment.get("routing_audit")
        recovery_audit = segment.get("backward_exit_recovery_audit")
        if not all(
            isinstance(value, Mapping)
            for value in (target, physics, routing, recovery_audit)
        ):
            raise ValueError("H3 fast-exit segment audit is incomplete")
        if (
            segment.get("completed") is not True
            or segment.get("fell") is not False
            or target.get("qpos_limit_violations") != 0
            or target.get("applied_target_limit_violations") != 0
            or target.get("desired_target_margin_violations") != 0
            or target.get("unauthorized_applied_target_margin_violations") != 0
            or target.get("target_slew_violations") != 0
            or target.get("nonfinite_sample_count") != 0
            or target.get("applied_head_action_peak") != 0.0
            or target.get("head_target_peak_rad") != 0.0
            or physics.get("qpos_limit_violations") != 0
            or physics.get("nonfinite_state_samples") != 0
            or physics.get("height_fall_samples") != 0
            or physics.get("upright_fall_samples") != 0
            or routing.get("command_clip_events") != 0
            or routing.get("prohibited_expert_steps") != 0
            or routing.get("atomic_endpoint_mismatch_steps") != 0
            or recovery_audit.get("cap_violation_count") != 0
            or recovery_audit.get("passed") is not True
        ):
            raise ValueError("H3 fast-exit safety/route subset did not pass")
        expected = int(segment.get("expected_physics_substeps", -1))
        completed = int(segment.get("completed_physics_substeps", -2))
        audited = int(physics.get("sample_count", -3))
        contact = int(physics.get("contact_sample_count", -4))
        if expected <= 0 or not (expected == completed == audited == contact):
            raise ValueError("H3 fast-exit substep/contact counts are not exact")
        expected_substeps += expected
        completed_substeps += completed
        audited_substeps += audited
        contact_samples += contact
        phase = routing.get("reverse_entry_phase")
        if not isinstance(phase, Mapping) or phase.get("passed") is not True:
            raise ValueError("H3 fast-exit phase audit mismatch")
        phase_events += int(phase.get("event_count", -1))
        joint_max = physics.get("joint_qpos_max_rad")
        if not isinstance(joint_max, Mapping):
            raise ValueError("H3 fast-exit qpos maxima are missing")
        left_knee_maximum = max(
            left_knee_maximum, float(joint_max.get("left_knee", np.inf))
        )

    if (
        len(raw_segments) != 500
        or expected_substeps != 370_000
        or completed_substeps != 370_000
        or audited_substeps != 370_000
        or contact_samples != 370_000
        or phase_events != 60
        or recovery_exits != 60
        or recovery_ticks != 780
        or left_knee_maximum != 0.47363927723644006
    ):
        raise ValueError("H3 fast-exit aggregate safety counts mismatch")

    episode_checks = central.get("episode_checks")
    failures: list[tuple[int, str, str]] = []
    if not isinstance(episode_checks, list) or len(episode_checks) != 20:
        raise ValueError("H3 central acceptance audit count mismatch")
    central_segment_count = central_segment_pass_count = 0
    for episode in episode_checks:
        seed = int(episode.get("seed", -1))
        checks = episode.get("segments")
        if not isinstance(checks, list) or len(checks) != len(TRANSITION_CASES):
            raise ValueError("H3 central acceptance segment count mismatch")
        for segment in checks:
            central_segment_count += 1
            passed = segment.get("passed") is True
            central_segment_pass_count += int(passed)
            check_map = segment.get("checks")
            if not isinstance(check_map, Mapping):
                raise ValueError("H3 central segment checks are missing")
            false_checks = sorted(
                str(name) for name, value in check_map.items() if value is not True
            )
            if passed and false_checks:
                raise ValueError("H3 passing segment contains a false check")
            if not passed:
                if len(false_checks) != 1:
                    raise ValueError("H3 motion failure must contain one false check")
                failures.append(
                    (seed, str(segment.get("name", "")), false_checks[0])
                )
    if (
        central.get("passed") is not False
        or central_segment_count != 500
        or central_segment_pass_count != 489
        or tuple(failures) != H3_FAST_EXIT_EXPECTED_MOTION_FAILURES
    ):
        raise ValueError("H3 expected 11 motion failures changed")

    expected_summary = {
        "episode_count": 20,
        "passed_episode_count": 10,
        "segment_count": 500,
        "passed_segment_count": 489,
        "motion_contact_failure_segment_count": 11,
        "fall_count": 0,
        "expected_physics_substeps": 370_000,
        "completed_physics_substeps": 370_000,
        "audited_physics_substeps": 370_000,
        "contact_samples": 370_000,
        "leg_joint_qpos_samples": 3_700_000,
        "qpos_limit_violation_samples": 0,
        "nonfinite_sample_count": 0,
        "route_failure_segment_count": 0,
        "recovery_exit_event_count": 60,
        "recovery_active_tick_count": 780,
        "recovery_state_acceptance_failure_count": 0,
        "reset_audit_passed_count": 20,
        "startup_audit_passed_count": 20,
        "all_physics_and_contact_counts_exact": True,
        "all_onnx_sessions_cpu_only": True,
        "pre_post_source_binary_data_unchanged": True,
        "central_suite_acceptance_passed": False,
    }
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        raise ValueError("H3 fast-exit summary mismatch")
    target_counts = summary.get("target_safety_violation_counts")
    if not isinstance(target_counts, Mapping) or any(
        target_counts.get(key) != 0
        for key in (
            "applied_target_limit_violations",
            "desired_target_margin_violations",
            "target_slew_violations",
            "unauthorized_applied_target_margin_violations",
        )
    ):
        raise ValueError("H3 summary target-safety counts are not zero")
    minimum_margin = float(summary.get("minimum_left_knee_safe_upper_margin_rad"))
    if (
        summary.get("maximum_left_knee_qpos_rad") != left_knee_maximum
        or minimum_margin != 0.0018947227635599528
        or not np.isclose(
            minimum_margin,
            float(SAFE_JOINT_LIMITS["left_knee"][1]) - left_knee_maximum,
            atol=1e-15,
            rtol=0.0,
        )
    ):
        raise ValueError("H3 left-knee safety margin mismatch")

    runtime = provenance.get("runtime")
    profiles = provenance.get("profiles")
    policy = provenance.get("policy")
    if not all(isinstance(value, Mapping) for value in (runtime, profiles, policy)):
        raise ValueError("H3 provenance is incomplete")
    pre = runtime.get("pre_import")
    post = runtime.get("post_evaluation")
    data_pre = runtime.get("runtime_model_and_data_pre_evaluation")
    if not all(isinstance(value, Mapping) for value in (pre, post, data_pre)):
        raise ValueError("H3 runtime provenance snapshots are missing")
    expected_roots = {
        "exp004_source_and_contract_snapshot": (9, "9bfef60fbca3c3e6ab34ff2c0f237ac6c85e15ab6ce9d10c2fcbaa5653bc1a9b"),
        "external_hard_allowlisted_source_closure": (4, "a40d4920049b349334b6d5567859fc2f7533d8fe6648f5127f0c7e4ce54dc098"),
        "hard_allowlisted_runtime_binary_closure": (5, "4e382762ffe85e33ba4839969088fed6f27cae1b35f6e0247a7d1b18937abe5f"),
    }
    for label, (count, root) in expected_roots.items():
        before = pre.get(label)
        after = post.get(label)
        if (
            not isinstance(before, Mapping)
            or not isinstance(after, Mapping)
            or dict(before) != dict(after)
            or before.get("dependency_count") != count
            or before.get("root_sha256") != root
            or before.get("all_hashes_verified") is not True
        ):
            raise ValueError(f"H3 provenance mismatch: {label}")
    data_post = post.get("runtime_model_and_data_closure")
    if (
        runtime.get("pre_post_source_and_data_hashes_unchanged") is not True
        or runtime.get("all_onnx_sessions_cpu_only_verified") is not True
        or not isinstance(data_post, Mapping)
        or dict(data_pre) != dict(data_post)
        or data_pre.get("dependency_count") != 54
        or data_pre.get("root_sha256")
        != "86619303ec142c6febb24a9914828971c91b0ec3f37242dcabc1cd607a80bae9"
        or data_pre.get("all_hashes_verified") is not True
        or any(
            providers != ["CPUExecutionProvider"]
            for providers in runtime.get(
                "onnx_session_execution_providers", {}
            ).values()
        )
    ):
        raise ValueError("H3 runtime data/provider provenance mismatch")
    if (
        set(profiles) != {"straight", "left", "right"}
        or {
            label: record.get("sha256")
            for label, record in profiles.items()
        }
        != dict(FORMAL_CANDIDATE_PROFILE_SHA256S)
        or any(
            record.get("composition", {}).get(
                "left_knee_extra_upper_margin_rad"
            )
            != FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
            for record in profiles.values()
        )
        or policy.get("all_roles_allowlisted") is not True
        or policy.get("adoption_eligible") is not True
        or any(
            record.get("sha256") != BASE_V22_POLICY_SHA256
            for record in policy.get("roles", {}).values()
        )
    ):
        raise ValueError("H3 profile/policy provenance mismatch")

    return {
        "path": str(resolved),
        "sha256": digest,
        "hash_allowlisted_for_safety_component": True,
        "status": H3_FAST_EXIT_SAFETY_STATUS,
        "source_artifact_status": "DIAGNOSTIC_FAIL",
        "safety_only_component": True,
        "safety_subset_passed": True,
        "central_suite_acceptance_passed": False,
        "source_artifact_passed": False,
        "episode_count": 20,
        "segment_count": 500,
        "safety_passed_segment_count": 500,
        "central_passed_segment_count": 489,
        "motion_failure_count": 11,
        "motion_failures": [
            {"seed": seed, "name": name, "check": check}
            for seed, name, check in H3_FAST_EXIT_EXPECTED_MOTION_FAILURES
        ],
        "physics_substep_count": 370_000,
        "contact_sample_count": 370_000,
        "leg_qpos_sample_count": 3_700_000,
        "phase_entry_event_count": 60,
        "recovery_exit_event_count": 60,
        "recovery_active_tick_count": 780,
        "minimum_left_knee_safe_upper_margin_rad": minimum_margin,
        "profile_sha256s": dict(FORMAL_CANDIDATE_PROFILE_SHA256S),
        "policy_sha256": BASE_V22_POLICY_SHA256,
        "phase_preincrement_indices": dict(
            FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES
        ),
        "profile_left_knee_cap": {
            "extra_upper_margin_rad": (
                FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
            ),
            "upper_target_rad": FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_UPPER_TARGET_RAD,
        },
        "backward_exit_recovery": {
            "enabled": True,
            "extra_upper_margin_rad": BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD,
            "upper_target_rad": BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD,
            "hold_control_ticks": BACKWARD_EXIT_RECOVERY_HOLD_TICKS,
            "hold_seconds": BACKWARD_EXIT_RECOVERY_HOLD_SECONDS,
        },
        "combined_5x15_required": True,
        "adoption_evidence": False,
        "adoption_eligible": False,
        "simulation_acceptance_eligible": False,
        "release_evidence": False,
        "hardware_deployment": "PROHIBITED",
    }


def validate_formal_candidate_selection_evidence(
    path: Path = FORMAL_CANDIDATE_SELECTION_EVIDENCE_PATH,
) -> dict[str, Any]:
    """Validate H3's passing combined 5x15 candidate-selection record.

    The artifact was produced by the preceding fail-closed H3 runtime, so its
    embedded status and execution bundle must remain in that source state.
    This validator independently derives only the next candidate status; it
    does not authorize adoption, simulation release, packaging, or hardware.
    """

    resolved = path.resolve()
    if resolved != FORMAL_CANDIDATE_SELECTION_EVIDENCE_PATH:
        raise ValueError("formal-candidate selection evidence path must remain pinned")
    if not resolved.is_file():
        raise FileNotFoundError(
            f"missing formal-candidate selection evidence: {resolved}"
        )
    if resolved.stat().st_size != 4_776_899:
        raise ValueError("formal-candidate H3 5x15 evidence size mismatch")
    digest = sha256_file(resolved)
    if (
        digest != FORMAL_CANDIDATE_SELECTION_EVIDENCE_SHA256
        or digest not in FORMAL_CANDIDATE_SELECTION_EVIDENCE_SHA256_ALLOWLIST
    ):
        raise ValueError("formal-candidate selection evidence hash mismatch")
    payload = _load_strict_json_object(resolved, "formal-candidate selection evidence")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("evaluator_id") != EVALUATOR_ID
        or payload.get("evaluation_mode") != H3_FAST_EXIT_SAFETY_STATUS
        or payload.get("simulation_suite_acceptance_passed") is not True
        or payload.get("simulation_acceptance_passed") is not False
    ):
        raise ValueError("formal-candidate H3 5x15 top-level status mismatch")

    historical_selection = _validate_h2_5x15_selection_evidence()
    safety_component = validate_h3_fast_exit_safety_evidence()

    def require_embedded(
        label: str,
        actual: Any,
        expected: Mapping[str, Any],
        expected_path: Path,
    ) -> None:
        if not isinstance(actual, Mapping):
            raise ValueError(f"formal-candidate H3 {label} is missing")
        for key, expected_value in expected.items():
            actual_value = actual.get(key)
            if key == "path":
                if not isinstance(actual_value, str) or not _portable_workspace_path_matches(
                    actual_value, expected_path
                ):
                    raise ValueError(f"formal-candidate H3 {label} path mismatch")
            elif actual_value != expected_value:
                raise ValueError(
                    f"formal-candidate H3 {label} binding mismatch: {key}"
                )

    require_embedded(
        "superseded H2 selection lineage",
        payload.get("formal_candidate_selection_evidence"),
        historical_selection,
        H2_5X15_SELECTION_EVIDENCE_PATH,
    )
    require_embedded(
        "fast-exit safety component",
        payload.get("h3_fast_exit_safety_component_evidence"),
        safety_component,
        H3_FAST_EXIT_SAFETY_EVIDENCE_PATH,
    )

    configuration = payload.get("configuration")
    release = payload.get("release_qualification")
    execution_bundle = payload.get("formal_candidate_execution_bundle")
    phase_contract = payload.get("formal_reverse_phase_entry_contract")
    recovery_contract = payload.get("formal_backward_exit_recovery_contract")
    command_contract = payload.get("command_mapping_contract")
    if not all(
        isinstance(value, Mapping)
        for value in (
            configuration,
            release,
            execution_bundle,
            phase_contract,
            recovery_contract,
            command_contract,
        )
    ):
        raise ValueError("formal-candidate H3 5x15 metadata is incomplete")

    expected_configuration = {
        "seed": FORMAL_CANDIDATE_MASTER_SEED,
        "episodes": 5,
        "seconds": 15.0,
        "transition_seconds": 15.0,
        "transition_stand_seconds": 5.0,
        "warmup_seconds": 1.5,
        "initial_base_speed": 0.1,
        "initial_joint_noise_scale": 1.0,
        "backward_residual_scale": 0.0,
        "leg_target_margin_rad": LEG_TARGET_MARGIN_RAD,
        "target_slew_rate_rad_per_s": TARGET_SLEW_LIMIT_RAD_PER_S,
        "reset_noise_margin_rad": RESET_NOISE_MARGIN_RAD,
        "left_knee_extra_upper_margin_rad": (
            FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
        ),
        "left_knee_profile_upper_target_rad": (
            FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_UPPER_TARGET_RAD
        ),
        "formal_candidate_default": True,
        "formal_candidate_status": H3_FAST_EXIT_SAFETY_STATUS,
        "formal_adopted_default": False,
        "formal_adopted_status": None,
        "backward_exit_recovery_enabled": True,
        "diagnostic_unadopted_policy": False,
        "diagnostic_unadopted_reverse_profile": None,
        "diagnostic_unadopted_reverse_left_profile": None,
        "diagnostic_unadopted_reverse_right_profile": None,
        "diagnostic_unadopted_backward_exit_recovery": False,
        "diagnostic_unadopted_reverse_entry_phase_indices": None,
        "diagnostic_noncontract_safety": False,
        "policy_command_diagnostic_suite": False,
        "control_first_startup_required": True,
        "home_only_startup_precharge_used": False,
        "physics_steps_allowed_before_startup_control": 0,
        "guard_calls_per_control_tick": 1,
    }
    for key, expected in expected_configuration.items():
        if configuration.get(key) != expected:
            raise ValueError(f"formal-candidate H3 5x15 configuration mismatch: {key}")
    if dict(configuration.get("executed_reverse_entry_phase_indices", {})) != dict(
        FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES
    ):
        raise ValueError("formal-candidate H3 5x15 phase mapping mismatch")
    executed_paths = configuration.get("executed_reverse_profile_paths")
    if not isinstance(executed_paths, Mapping) or any(
        not isinstance(executed_paths.get(label), str)
        or not _portable_workspace_path_matches(
            str(executed_paths[label]), FORMAL_CANDIDATE_PROFILE_PATHS[label]
        )
        for label in ("straight", "left", "right")
    ):
        raise ValueError("formal-candidate H3 5x15 profile path mismatch")

    release_actual = release.get("actual")
    if (
        not isinstance(release_actual, Mapping)
        or release_actual.get("master_seed") != FORMAL_CANDIDATE_MASTER_SEED
        or release_actual.get("episodes") != 5
        or release_actual.get("seconds") != 15.0
        or release_actual.get("transition_seconds") != 15.0
        or release_actual.get("transition_stand_seconds") != 5.0
        or release.get("diagnostic_mode_disabled") is not True
        or release.get("master_seed_matches_recommendation") is not True
        or release.get("scale_matches_frozen_contract") is not False
        or release.get("release_qualification_eligible") is not False
        or release.get("screening_cannot_promote_release_or_adoption") is not True
        or release.get("status") != "SCREENING_CANDIDATE"
    ):
        raise ValueError("formal-candidate H3 5x15 screening gate mismatch")

    source_bundle_expected = {
        "passed": True,
        "formal_candidate": True,
        "status": H3_FAST_EXIT_SAFETY_STATUS,
        "candidate_selection_evidence_sha256": H2_5X15_SELECTION_EVIDENCE_SHA256,
        "candidate_selection_is_superseded_h2_profile_lineage": True,
        "safety_component_evidence_sha256": H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256,
        "safety_component_only": True,
        "safety_subset_passed": True,
        "central_suite_acceptance_passed": False,
        "motion_failure_count": 11,
        "profile_sha256s": dict(FORMAL_CANDIDATE_PROFILE_SHA256S),
        "policy_sha256": BASE_V22_POLICY_SHA256,
        "policy_roles": list(REQUIRED_POLICY_ROLES),
        "reverse_endpoint_mps": CURRENT_FORMAL_REVERSE_ENDPOINT_MPS,
        "phase_preincrement_indices": dict(
            FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES
        ),
        "profile_left_knee_cap": {
            "extra_upper_margin_rad": (
                FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
            ),
            "upper_target_rad": FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_UPPER_TARGET_RAD,
        },
        "backward_exit_recovery": {
            "enabled": True,
            "extra_upper_margin_rad": BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD,
            "hold_control_ticks": BACKWARD_EXIT_RECOVERY_HOLD_TICKS,
            "hold_seconds": BACKWARD_EXIT_RECOVERY_HOLD_SECONDS,
            "upper_target_rad": BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD,
        },
        "combined_5x15_required": True,
        "superseded_h2_adoption_evidence_sha256": (
            H2_SUPERSEDED_ADOPTION_EVIDENCE_SHA256
        ),
        "adopted": False,
        "adoption_eligible": False,
        "simulation_acceptance_eligible": False,
        "hardware_deployment": "PROHIBITED",
    }
    if dict(execution_bundle) != source_bundle_expected:
        raise ValueError("formal-candidate H3 source execution bundle mismatch")

    phase_mapping = phase_contract.get("preincrement_phase_indices")
    recovery_runtime = recovery_contract.get("runtime_contract")
    if (
        phase_contract.get("enabled") is not True
        or phase_contract.get("enabled_by_default") is not True
        or phase_contract.get("diagnostic_only") is not False
        or phase_contract.get("status") != H3_FAST_EXIT_SAFETY_STATUS
        or phase_contract.get("combined_5x15_required") is not True
        or phase_contract.get("fast_exit_safety_passed") is not True
        or phase_contract.get("safety_component_only") is not True
        or phase_contract.get("requires_formal_20x30_requalification") is not True
        or phase_contract.get("current_endpoint_requalified") is not False
        or not isinstance(phase_mapping, Mapping)
        or dict(phase_mapping) != dict(FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES)
        or recovery_contract.get("enabled") is not True
        or recovery_contract.get("enabled_by_default") is not True
        or recovery_contract.get("diagnostic_unadopted_only") is not False
        or recovery_contract.get("status") != H3_FAST_EXIT_SAFETY_STATUS
        or recovery_contract.get("selection_evidence_sha256")
        != H2_5X15_SELECTION_EVIDENCE_SHA256
        or recovery_contract.get("safety_component_evidence_sha256")
        != H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256
        or not isinstance(recovery_runtime, Mapping)
        or recovery_runtime.get("extra_upper_margin_rad")
        != BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD
        or recovery_runtime.get("upper_target_rad")
        != BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD
        or recovery_runtime.get("hold_control_ticks")
        != BACKWARD_EXIT_RECOVERY_HOLD_TICKS
        or recovery_runtime.get("hold_seconds")
        != BACKWARD_EXIT_RECOVERY_HOLD_SECONDS
        or recovery_runtime.get("release") != "instant_after_hold"
    ):
        raise ValueError("formal-candidate H3 phase/recovery contract mismatch")

    validation_gate = command_contract.get("validation_status_gate")
    expected_pending_names = {
        "reverse",
        "reverse_turn_left",
        "reverse_turn_right",
        "transition_reverse",
        "transition_reverse_turn_left",
        "transition_reverse_turn_right",
    }
    pending_records = (
        validation_gate.get("nonadoptable_cases", ())
        if isinstance(validation_gate, Mapping)
        else ()
    )
    if (
        not isinstance(validation_gate, Mapping)
        or validation_gate.get("passed") is not False
        or validation_gate.get("case_count") != 38
        or validation_gate.get("nonadoptable_case_count") != 6
        or validation_gate.get("reverse_safety_component_evidence_failure_count")
        != 0
        or validation_gate.get("reverse_adoption_evidence_failure_count") != 6
        or {
            str(record.get("name", ""))
            for record in pending_records
            if isinstance(record, Mapping)
        }
        != expected_pending_names
        or any(
            not isinstance(record, Mapping)
            or record.get("validation_status") != H3_FAST_EXIT_SAFETY_STATUS
            for record in pending_records
        )
    ):
        raise ValueError("formal-candidate H3 command cases are not fail-closed")

    suites = payload.get("suites")
    expected_suites = {
        "primitives": (list(range(20_260_808, 20_260_813)), 35),
        "compounds": (list(range(21_260_808, 21_260_813)), 30),
        "transitions": (list(range(22_260_808, 22_260_813)), 125),
    }
    if not isinstance(suites, Mapping) or set(suites) != set(expected_suites):
        raise ValueError("formal-candidate H3 suite set mismatch")
    episodes: list[Mapping[str, Any]] = []
    segments: list[Mapping[str, Any]] = []
    accepted_segments: list[Mapping[str, Any]] = []
    reset_audits: list[Mapping[str, Any]] = []
    startup_audits: list[Mapping[str, Any]] = []
    recovery_state_audits: list[Mapping[str, Any]] = []
    for suite_name, (expected_seeds, expected_segment_count) in expected_suites.items():
        suite = suites[suite_name]
        suite_episodes = suite.get("episodes") if isinstance(suite, Mapping) else None
        acceptance = suite.get("acceptance") if isinstance(suite, Mapping) else None
        checks = acceptance.get("episode_checks") if isinstance(acceptance, Mapping) else None
        if (
            not isinstance(suite_episodes, list)
            or [episode.get("seed") for episode in suite_episodes] != expected_seeds
            or not isinstance(checks, list)
            or [check.get("seed") for check in checks] != expected_seeds
            or acceptance.get("passed") is not True
            or not all(check.get("passed") is True for check in checks)
        ):
            raise ValueError(f"formal-candidate H3 {suite_name} acceptance mismatch")
        suite_segments = [
            segment
            for episode in suite_episodes
            for segment in episode.get("segments", ())
            if isinstance(segment, Mapping)
        ]
        if len(suite_segments) != expected_segment_count:
            raise ValueError(f"formal-candidate H3 {suite_name} segment count mismatch")
        suite_accepted = [
            segment
            for check in checks
            for segment in check.get("segments", ())
            if isinstance(segment, Mapping)
        ]
        if (
            len(suite_accepted) != expected_segment_count
            or not all(
                item.get("passed") is True
                and isinstance(item.get("checks"), Mapping)
                and all(value is True for value in item["checks"].values())
                for item in suite_accepted
            )
        ):
            raise ValueError(f"formal-candidate H3 {suite_name} stored checks failed")
        for check in checks:
            reset_audits.extend(
                record
                for record in check.get("reset_qpos_audits", ())
                if isinstance(record, Mapping)
            )
            startup_audits.extend(
                record
                for record in check.get("control_first_startup_audits", ())
                if isinstance(record, Mapping)
            )
            recovery_state_audits.extend(
                record
                for record in check.get("backward_exit_recovery_audits", ())
                if isinstance(record, Mapping)
            )
        episodes.extend(suite_episodes)
        segments.extend(suite_segments)
        accepted_segments.extend(suite_accepted)

    if (
        len(episodes) != 15
        or len(segments) != 190
        or len(accepted_segments) != 190
        or not all(
            episode.get("fell") is False
            and episode.get("completed_segment_count")
            == episode.get("requested_segment_count")
            for episode in episodes
        )
        or not all(
            segment.get("completed") is True
            and segment.get("fell") is False
            and segment.get("completed_physics_substeps")
            == segment.get("expected_physics_substeps")
            and segment_acceptance(segment).get("passed") is True
            for segment in segments
        )
    ):
        raise ValueError("formal-candidate H3 episode/segment acceptance mismatch")

    safety_zero_fields = (
        "applied_target_limit_violations",
        "desired_target_margin_violations",
        "nonfinite_sample_count",
        "preclip_target_limit_violations",
        "qpos_limit_violations",
        "target_slew_violations",
        "unauthorized_applied_target_margin_violations",
        "maximum_applied_target_excess_rad",
        "maximum_desired_target_margin_excess_rad",
        "maximum_preclip_target_excess_rad",
        "maximum_qpos_excess_rad",
        "applied_head_action_peak",
        "head_target_peak_rad",
    )
    physics_zero_fields = (
        "height_fall_samples",
        "upright_fall_samples",
        "nonfinite_full_qpos_samples",
        "nonfinite_full_qvel_samples",
        "nonfinite_leg_qpos_samples",
        "nonfinite_pose_samples",
        "nonfinite_state_samples",
        "qpos_limit_violations",
        "maximum_qpos_excess_rad",
    )
    physics_count = contact_count = leg_count = control_count = 0
    authorized_margin_count = startup_margin_count = preclip_margin_count = 0
    minimum_height = np.inf
    minimum_upright = np.inf
    phase_audits: list[Mapping[str, Any]] = []
    recovery_audits: list[Mapping[str, Any]] = []
    for segment in segments:
        safety = segment.get("safety_audit")
        physics = segment.get("physics_substep_audit")
        routing = segment.get("routing")
        recovery = segment.get("backward_exit_recovery_audit")
        if not all(isinstance(value, Mapping) for value in (safety, physics, routing, recovery)):
            raise ValueError("formal-candidate H3 segment audit is incomplete")
        phase = routing.get("reverse_entry_phase")
        if (
            any(safety.get(field) != 0 for field in safety_zero_fields)
            or safety.get("leg_target_margin_rad") != LEG_TARGET_MARGIN_RAD
            or safety.get("target_slew_limit_rad_per_s")
            != TARGET_SLEW_LIMIT_RAD_PER_S
            or safety.get("maximum_target_slew_rate_rad_per_s")
            > TARGET_SLEW_LIMIT_RAD_PER_S + 2e-15
            or any(physics.get(field) != 0 for field in physics_zero_fields)
            or physics.get("fall_or_nonfinite_detected") is not False
            or physics.get("contact_sample_count_matches_sample_count") is not True
            or physics.get("contact_sample_count") != physics.get("sample_count")
            or routing.get("command_clip_events") != 0
            or routing.get("prohibited_expert_steps") != 0
            or routing.get("atomic_endpoint_mismatch_steps") != 0
            or not isinstance(phase, Mapping)
            or phase.get("passed") is not True
            or recovery.get("passed") is not True
            or recovery.get("cap_violation_count") != 0
        ):
            raise ValueError("formal-candidate H3 safety/physics/route audit failed")
        physics_count += int(physics["sample_count"])
        contact_count += int(physics["contact_sample_count"])
        leg_count += int(physics["leg_joint_sample_count"])
        control_count += int(safety["sample_count"])
        authorized_margin_count += int(safety["applied_target_margin_violations"])
        startup_margin_count += int(safety["startup_margin_transition_joint_samples"])
        preclip_margin_count += int(safety["preclip_target_margin_violations"])
        minimum_height = min(minimum_height, float(physics["minimum_height_m"]))
        minimum_upright = min(minimum_upright, float(physics["minimum_upright"]))
        phase_audits.append(phase)
        recovery_audits.append(recovery)
    if (
        physics_count != 1_100_000
        or contact_count != 1_100_000
        or leg_count != 11_000_000
        or control_count != 110_000
        or authorized_margin_count != 40
        or startup_margin_count != 40
        or authorized_margin_count != startup_margin_count
        or preclip_margin_count != 110_403
        or minimum_height != 0.17911993
        or minimum_upright != 0.9785479266972336
    ):
        raise ValueError("formal-candidate H3 audited aggregate mismatch")

    phase_events = [
        event
        for audit in phase_audits
        for event in audit.get("events", ())
        if isinstance(event, Mapping)
    ]
    phase_counts = {
        expert: sum(event.get("current_expert") == expert for event in phase_events)
        for expert in FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES
    }
    if (
        len(phase_audits) != 190
        or len(phase_events) != 30
        or phase_counts != {expert: 10 for expert in phase_counts}
        or any(
            event.get("reset_preincrement_phase_index")
            != FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES[
                str(event.get("current_expert", ""))
            ]
            for event in phase_events
        )
        or sum(int(audit.get("exit_event_count", -1)) for audit in recovery_audits)
        != 15
        or sum(int(audit.get("active_tick_count", -1)) for audit in recovery_audits)
        != 195
        or sum(int(audit.get("sample_count", -1)) for audit in recovery_audits)
        != 110_000
        or sum(int(audit.get("final_guard_call_count", -1)) for audit in recovery_audits)
        != 110_000
    ):
        raise ValueError("formal-candidate H3 phase/recovery totals mismatch")
    if (
        len(reset_audits) != 70
        or len(startup_audits) != 70
        or len(recovery_state_audits) != 70
        or not all(audit.get("passed") is True for audit in reset_audits)
        or not all(audit.get("passed") is True for audit in startup_audits)
        or not all(audit.get("passed") is True for audit in recovery_state_audits)
        or sum(
            int(audit.get("exit_event_count", -1))
            for audit in recovery_state_audits
        )
        != 15
        or sum(
            int(audit.get("active_tick_count", -1))
            for audit in recovery_state_audits
        )
        != 195
        or sum(
            int(audit.get("completed_event_count", -1))
            for audit in recovery_state_audits
        )
        != 15
        or sum(
            int(audit.get("cap_violation_count", -1))
            for audit in recovery_state_audits
        )
        != 0
        or sum(
            int(audit.get("remaining_ticks", -1))
            for audit in recovery_state_audits
        )
        != 0
        or sum(
            int(audit.get("control_tick_count", -1))
            for audit in recovery_state_audits
        )
        != 110_000
        or sum(
            int(audit.get("final_guard_call_count", -1))
            for audit in recovery_state_audits
        )
        != 110_000
    ):
        raise ValueError("formal-candidate H3 reset/startup/recovery state audit failed")

    reverse_profiles = payload.get("reverse_profile_evidence")
    executed_profiles = (
        reverse_profiles.get("executed_profiles")
        if isinstance(reverse_profiles, Mapping)
        else None
    )
    policy_provenance = payload.get("policy_provenance")
    policy_roles = (
        policy_provenance.get("roles")
        if isinstance(policy_provenance, Mapping)
        else None
    )
    if (
        not isinstance(executed_profiles, Mapping)
        or {
            label: record.get("sha256")
            for label, record in executed_profiles.items()
            if isinstance(record, Mapping)
        }
        != dict(FORMAL_CANDIDATE_PROFILE_SHA256S)
        or any(
            not isinstance(record, Mapping)
            or record.get("composition", {}).get(
                "left_knee_extra_upper_margin_rad"
            )
            != FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
            for record in executed_profiles.values()
        )
        or not isinstance(policy_roles, Mapping)
        or set(policy_roles) != set(REQUIRED_POLICY_ROLES)
        or policy_provenance.get("all_roles_allowlisted") is not True
        or any(
            not isinstance(record, Mapping)
            or record.get("sha256") != BASE_V22_POLICY_SHA256
            or record.get("formal_base_v22_allowlisted") is not True
            for record in policy_roles.values()
        )
    ):
        raise ValueError("formal-candidate H3 profile/policy provenance mismatch")

    provenance = payload.get("runtime_dependency_provenance")
    pre = provenance.get("pre_import") if isinstance(provenance, Mapping) else None
    post = provenance.get("post_evaluation") if isinstance(provenance, Mapping) else None
    runtime_data_pre = (
        provenance.get("runtime_model_and_data_pre_evaluation")
        if isinstance(provenance, Mapping)
        else None
    )
    if not all(isinstance(value, Mapping) for value in (provenance, pre, post, runtime_data_pre)):
        raise ValueError("formal-candidate H3 provenance is incomplete")
    closure_contract = {
        "exp004_source_and_contract_snapshot": (
            "316bd4eec7afcc609d76508c1ff767c64c9ea0d35bcea8be03831264a90e75e6",
            9,
        ),
        "external_hard_allowlisted_source_closure": (
            "a40d4920049b349334b6d5567859fc2f7533d8fe6648f5127f0c7e4ce54dc098",
            4,
        ),
        "hard_allowlisted_runtime_binary_closure": (
            "4e382762ffe85e33ba4839969088fed6f27cae1b35f6e0247a7d1b18937abe5f",
            5,
        ),
    }
    for label, (root_sha256, dependency_count) in closure_contract.items():
        before = pre.get(label)
        after = post.get(label)
        if (
            not isinstance(before, Mapping)
            or not isinstance(after, Mapping)
            or dict(before) != dict(after)
            or before.get("root_sha256") != root_sha256
            or before.get("dependency_count") != dependency_count
            or before.get("all_hashes_verified") is not True
        ):
            raise ValueError(f"formal-candidate H3 provenance mismatch: {label}")
    runtime_data_post = post.get("runtime_model_and_data_closure")
    runtime_entries = runtime_data_pre.get("entries")
    if (
        provenance.get("verified") is not True
        or provenance.get("pre_post_source_and_data_hashes_unchanged") is not True
        or provenance.get("all_onnx_sessions_cpu_only_verified") is not True
        or not isinstance(runtime_data_post, Mapping)
        or dict(runtime_data_pre) != dict(runtime_data_post)
        or runtime_data_pre.get("root_sha256")
        != "6888ce93ab26bf1d741a46d9f17e1bd2c8aa0b6cba0a86e8c003fd1d755c6b5a"
        or runtime_data_pre.get("dependency_count") != 55
        or runtime_data_pre.get("all_hashes_verified") is not True
        or not isinstance(runtime_entries, Mapping)
        or runtime_entries.get("formal_candidate_selection_evidence", {}).get(
            "sha256"
        )
        != H2_5X15_SELECTION_EVIDENCE_SHA256
        or runtime_entries.get("h3_fast_exit_safety_component_evidence", {}).get(
            "sha256"
        )
        != H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256
        or any(
            providers != ["CPUExecutionProvider"]
            for providers in provenance.get(
                "onnx_session_execution_providers", {}
            ).values()
        )
    ):
        raise ValueError("formal-candidate H3 runtime data/provider provenance mismatch")

    hardware = payload.get("hardware_gate")
    adoption = payload.get("adoption_contract")
    reverse_adoption = payload.get("reverse_profile_adoption")
    if (
        not isinstance(hardware, Mapping)
        or hardware.get("status") != "PROHIBITED"
        or hardware.get("hardware_deployment_allowed") is not False
        or not isinstance(adoption, Mapping)
        or adoption.get("passed") is not False
        or adoption.get("reverse_profile_adopted") is not False
        or not isinstance(reverse_adoption, Mapping)
        or reverse_adoption.get("passed") is not False
        or reverse_adoption.get("status") != "FAIL_CLOSED"
        or any(reverse_adoption.get("evidence_hash_allowlists", {}).values())
    ):
        raise ValueError("formal-candidate H3 adoption/hardware gate mismatch")

    return {
        "path": str(resolved),
        "sha256": digest,
        "hash_allowlisted_for_candidate_selection": True,
        "hash_allowlisted_for_adoption": False,
        "schema_validated": True,
        "all_json_numbers_finite": True,
        "status": H3_CANDIDATE_SELECTION_STATUS,
        "source_artifact_status": H3_FAST_EXIT_SAFETY_STATUS,
        "selection_scope": "h3_combined_phase744_recovery0225_5x15",
        "master_seed": FORMAL_CANDIDATE_MASTER_SEED,
        "profile_sha256s": dict(FORMAL_CANDIDATE_PROFILE_SHA256S),
        "policy_sha256": BASE_V22_POLICY_SHA256,
        "reverse_endpoint_mps": CURRENT_FORMAL_REVERSE_ENDPOINT_MPS,
        "phase_preincrement_indices": dict(FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES),
        "profile_left_knee_cap": {
            "extra_upper_margin_rad": FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD,
            "upper_target_rad": FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_UPPER_TARGET_RAD,
        },
        "backward_exit_recovery": {
            "enabled": True,
            "extra_upper_margin_rad": BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD,
            "hold_control_ticks": BACKWARD_EXIT_RECOVERY_HOLD_TICKS,
            "hold_seconds": BACKWARD_EXIT_RECOVERY_HOLD_SECONDS,
            "upper_target_rad": BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD,
        },
        "safety_component_evidence_sha256": H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256,
        "superseded_h2_selection_evidence_sha256": H2_5X15_SELECTION_EVIDENCE_SHA256,
        "superseded_h2_adoption_evidence_sha256": (
            H2_SUPERSEDED_ADOPTION_EVIDENCE_SHA256
        ),
        "suite_episode_count": 15,
        "segment_pass_count": 190,
        "physics_substep_count": physics_count,
        "contact_sample_count": contact_count,
        "control_sample_count": control_count,
        "leg_qpos_sample_count": leg_count,
        "minimum_height_m": minimum_height,
        "minimum_upright": minimum_upright,
        "authorized_startup_margin_transition_joint_samples": startup_margin_count,
        "preclip_target_margin_violations": preclip_margin_count,
        "phase_entry_event_count": len(phase_events),
        "recovery_exit_event_count": 15,
        "recovery_active_tick_count": 195,
        "pre_post_provenance_unchanged": True,
        "candidate_execution_eligible": True,
        "combined_5x15_required": False,
        "combined_5x15_passed": True,
        "formal_20x30_required": True,
        "adopted": False,
        "adoption_eligible": False,
        "simulation_acceptance_eligible": False,
        "release_evidence": False,
        "hardware_deployment": "PROHIBITED",
    }


def validate_superseded_h2_adoption_evidence(
    path: Path = H2_SUPERSEDED_ADOPTION_EVIDENCE_PATH,
) -> dict[str, Any]:
    """Validate the independently audited H2 20x30 as superseded lineage.

    This artifact was intentionally produced while all adoption gates were
    pending. H3 changes the recovery cap, so this record cannot authorize the
    current runtime, package release, or hardware use.
    """

    resolved = path.resolve()
    if resolved != H2_SUPERSEDED_ADOPTION_EVIDENCE_PATH:
        raise ValueError("superseded H2 adoption evidence path must remain pinned")
    if not resolved.is_file():
        raise FileNotFoundError(f"missing formal adoption evidence: {resolved}")
    digest = sha256_file(resolved)
    if digest != H2_SUPERSEDED_ADOPTION_EVIDENCE_SHA256:
        raise ValueError("superseded H2 adoption evidence hash mismatch")
    if resolved.stat().st_size != 18_091_288:
        raise ValueError("formal adoption evidence size mismatch")
    payload = _load_strict_json_object(resolved, "formal adoption evidence")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("evaluator_id") != EVALUATOR_ID
        or payload.get("evaluation_mode")
        != "FORMAL_CANDIDATE_DEFAULT_PENDING_ADOPTION"
        or payload.get("simulation_suite_acceptance_passed") is not True
        or payload.get("simulation_acceptance_passed") is not False
    ):
        raise ValueError("formal adoption evidence top-level gate mismatch")

    qualification = payload.get("release_qualification")
    configuration = payload.get("configuration")
    if not isinstance(qualification, Mapping) or not isinstance(
        configuration, Mapping
    ):
        raise ValueError("formal adoption evidence metadata is incomplete")
    actual = qualification.get("actual")
    expected_scale = {
        "episodes": 20,
        "seconds": 30.0,
        "transition_seconds": 30.0,
        "transition_stand_seconds": 5.0,
        "warmup_seconds": 1.5,
        "initial_joint_noise_scale": 1.0,
        "initial_base_speed": 0.1,
        "master_seed": FORMAL_CANDIDATE_MASTER_SEED,
    }
    if (
        qualification.get("status") != "RELEASE_QUALIFICATION"
        or qualification.get("release_qualification_eligible") is not True
        or qualification.get("scale_matches_frozen_contract") is not True
        or qualification.get("diagnostic_mode_disabled") is not True
        or qualification.get("master_seed_matches_recommendation") is not True
        or not isinstance(actual, Mapping)
        or dict(actual) != expected_scale
        or configuration.get("seed") != FORMAL_CANDIDATE_MASTER_SEED
        or configuration.get("formal_candidate_default") is not True
        or configuration.get("formal_candidate_status")
        != H2_5X15_SELECTION_STATUS
        or dict(configuration.get("executed_reverse_entry_phase_indices", {}))
        != dict(FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES)
        or configuration.get("left_knee_extra_upper_margin_rad")
        != FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
        or configuration.get("left_knee_profile_upper_target_rad")
        != FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_UPPER_TARGET_RAD
    ):
        raise ValueError("formal adoption evidence scale/configuration mismatch")
    executed_paths = configuration.get("executed_reverse_profile_paths")
    if not isinstance(executed_paths, Mapping) or any(
        not isinstance(executed_paths.get(label), str)
        or not _portable_workspace_path_matches(
            str(executed_paths[label]), FORMAL_CANDIDATE_PROFILE_PATHS[label]
        )
        for label in ("straight", "left", "right")
    ):
        raise ValueError("formal adoption evidence profile path mismatch")

    selection = _validate_h2_5x15_selection_evidence()
    embedded_selection = payload.get("formal_candidate_selection_evidence")
    if not isinstance(embedded_selection, Mapping):
        raise ValueError("formal adoption evidence selection binding is missing")
    for key, expected in selection.items():
        actual_value = embedded_selection.get(key)
        if key == "path":
            if not isinstance(actual_value, str) or not _portable_workspace_path_matches(
                actual_value, H2_5X15_SELECTION_EVIDENCE_PATH
            ):
                raise ValueError("formal adoption selection path mismatch")
        elif actual_value != expected:
            raise ValueError(f"formal adoption selection mismatch: {key}")
    execution_bundle = payload.get("formal_candidate_execution_bundle")
    if (
        not isinstance(execution_bundle, Mapping)
        or execution_bundle.get("passed") is not True
        or execution_bundle.get("status") != H2_5X15_SELECTION_STATUS
        or execution_bundle.get("candidate_selection_evidence_sha256")
        != H2_5X15_SELECTION_EVIDENCE_SHA256
        or dict(execution_bundle.get("profile_sha256s", {}))
        != dict(FORMAL_CANDIDATE_PROFILE_SHA256S)
        or execution_bundle.get("policy_sha256") != BASE_V22_POLICY_SHA256
        or set(execution_bundle.get("policy_roles", ()))
        != set(REQUIRED_POLICY_ROLES)
        or dict(execution_bundle.get("phase_preincrement_indices", {}))
        != dict(FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES)
        or execution_bundle.get("profile_left_knee_cap")
        != {
            "extra_upper_margin_rad": (
                FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
            ),
            "upper_target_rad": FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_UPPER_TARGET_RAD,
        }
        or execution_bundle.get("backward_exit_recovery")
        != {
            "enabled": True,
            "extra_upper_margin_rad": H2_SUPERSEDED_RECOVERY_EXTRA_UPPER_MARGIN_RAD,
            "hold_control_ticks": H2_SUPERSEDED_RECOVERY_HOLD_TICKS,
            "hold_seconds": H2_SUPERSEDED_RECOVERY_HOLD_SECONDS,
            "upper_target_rad": H2_SUPERSEDED_RECOVERY_UPPER_TARGET_RAD,
        }
        or execution_bundle.get("adopted") is not False
        or execution_bundle.get("adoption_eligible") is not False
        or execution_bundle.get("simulation_acceptance_eligible") is not False
        or execution_bundle.get("hardware_deployment") != "PROHIBITED"
    ):
        raise ValueError("formal adoption execution-bundle binding mismatch")

    suites = payload.get("suites")
    expected_segment_counts = {
        "primitives": 140,
        "compounds": 120,
        "transitions": 500,
    }
    if not isinstance(suites, Mapping) or set(suites) != set(expected_segment_counts):
        raise ValueError("formal adoption suite set mismatch")
    episodes: list[Mapping[str, Any]] = []
    segments: list[Mapping[str, Any]] = []
    accepted_segments: list[Mapping[str, Any]] = []
    reset_audits: list[Mapping[str, Any]] = []
    startup_audits: list[Mapping[str, Any]] = []
    recovery_state_audits: list[Mapping[str, Any]] = []
    explicit_check_true_count = 0

    def validate_checks(value: Any) -> None:
        nonlocal explicit_check_true_count
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in {"checks", "hard_checks"}:
                    if not isinstance(item, Mapping) or not item or any(
                        check is not True for check in item.values()
                    ):
                        raise ValueError("formal adoption explicit checks failed")
                    explicit_check_true_count += len(item)
                validate_checks(item)
        elif isinstance(value, list):
            for item in value:
                validate_checks(item)

    for suite_name, expected_segment_count in expected_segment_counts.items():
        suite = suites[suite_name]
        suite_episodes = suite.get("episodes") if isinstance(suite, Mapping) else None
        acceptance = suite.get("acceptance") if isinstance(suite, Mapping) else None
        suite_checks = acceptance.get("episode_checks") if isinstance(acceptance, Mapping) else None
        if (
            not isinstance(suite_episodes, list)
            or len(suite_episodes) != 20
            or not all(isinstance(item, Mapping) for item in suite_episodes)
            or not isinstance(acceptance, Mapping)
            or acceptance.get("passed") is not True
            or not isinstance(suite_checks, list)
            or len(suite_checks) != 20
            or not all(
                isinstance(item, Mapping) and item.get("passed") is True
                for item in suite_checks
            )
        ):
            raise ValueError(f"formal adoption {suite_name} acceptance mismatch")
        suite_segments = [
            segment
            for episode in suite_episodes
            for segment in episode.get("segments", ())
            if isinstance(segment, Mapping)
        ]
        if len(suite_segments) != expected_segment_count:
            raise ValueError(f"formal adoption {suite_name} segment count mismatch")
        episodes.extend(suite_episodes)
        segments.extend(suite_segments)
        for check in suite_checks:
            accepted_segments.extend(
                item for item in check.get("segments", ()) if isinstance(item, Mapping)
            )
            reset_audits.extend(
                item
                for item in check.get("reset_qpos_audits", ())
                if isinstance(item, Mapping)
            )
            startup_audits.extend(
                item
                for item in check.get("control_first_startup_audits", ())
                if isinstance(item, Mapping)
            )
            recovery_state_audits.extend(
                item
                for item in check.get("backward_exit_recovery_audits", ())
                if isinstance(item, Mapping)
            )
        validate_checks(suite)
    if (
        len(episodes) != 60
        or len(segments) != 760
        or len(accepted_segments) != 760
        or explicit_check_true_count <= 28_120
        or not all(item.get("passed") is True for item in accepted_segments)
        or not all(
            episode.get("fell") is False
            and episode.get("completed_segment_count")
            == episode.get("requested_segment_count")
            for episode in episodes
        )
        or not all(
            segment.get("completed") is True
            and segment.get("fell") is False
            and segment.get("completed_physics_substeps")
            == segment.get("expected_physics_substeps")
            for segment in segments
        )
    ):
        raise ValueError("formal adoption episode/segment completion mismatch")

    safety_zero_fields = (
        "applied_target_limit_violations",
        "desired_target_margin_violations",
        "nonfinite_sample_count",
        "preclip_target_limit_violations",
        "qpos_limit_violations",
        "target_slew_violations",
        "unauthorized_applied_target_margin_violations",
        "maximum_applied_target_excess_rad",
        "maximum_desired_target_margin_excess_rad",
        "maximum_preclip_target_excess_rad",
        "maximum_qpos_excess_rad",
        "applied_head_action_peak",
        "head_target_peak_rad",
    )
    physics_zero_fields = (
        "height_fall_samples",
        "upright_fall_samples",
        "nonfinite_full_qpos_samples",
        "nonfinite_full_qvel_samples",
        "nonfinite_leg_qpos_samples",
        "nonfinite_pose_samples",
        "nonfinite_state_samples",
        "qpos_limit_violations",
        "maximum_qpos_excess_rad",
    )
    physics_samples = 0
    contact_samples = 0
    leg_qpos_samples = 0
    control_samples = 0
    applied_margin_transitions = 0
    preclip_margin_samples = 0
    phase_audits: list[Mapping[str, Any]] = []
    recovery_audits: list[Mapping[str, Any]] = []
    for segment in segments:
        safety = segment.get("safety_audit")
        physics = segment.get("physics_substep_audit")
        routing = segment.get("routing")
        recovery = segment.get("backward_exit_recovery_audit")
        if not all(isinstance(value, Mapping) for value in (safety, physics, routing, recovery)):
            raise ValueError("formal adoption segment audit is incomplete")
        phase = routing.get("reverse_entry_phase")
        if (
            any(safety.get(field) != 0 for field in safety_zero_fields)
            or safety.get("applied_target_margin_violations")
            != safety.get("startup_margin_transition_joint_samples")
            or safety.get("leg_target_margin_rad") != LEG_TARGET_MARGIN_RAD
            or safety.get("target_slew_limit_rad_per_s")
            != TARGET_SLEW_LIMIT_RAD_PER_S
            or safety.get("maximum_target_slew_rate_rad_per_s")
            > TARGET_SLEW_LIMIT_RAD_PER_S + 2e-15
            or any(physics.get(field) != 0 for field in physics_zero_fields)
            or physics.get("fall_or_nonfinite_detected") is not False
            or physics.get("contact_sample_count_matches_sample_count") is not True
            or physics.get("contact_sample_count") != physics.get("sample_count")
            or physics.get("minimum_height_m") < physics.get("minimum_height_limit_m")
            or physics.get("minimum_upright") < physics.get("minimum_upright_limit")
            or routing.get("command_clip_events") != 0
            or routing.get("prohibited_expert_steps") != 0
            or not isinstance(phase, Mapping)
            or phase.get("passed") is not True
            or recovery.get("passed") is not True
        ):
            raise ValueError("formal adoption safety/physics/routing audit failed")
        physics_samples += int(physics["sample_count"])
        contact_samples += int(physics["contact_sample_count"])
        leg_qpos_samples += int(physics["leg_joint_sample_count"])
        control_samples += int(safety["sample_count"])
        applied_margin_transitions += int(safety["applied_target_margin_violations"])
        preclip_margin_samples += int(safety["preclip_target_margin_violations"])
        phase_audits.append(phase)
        recovery_audits.append(recovery)
    if (
        physics_samples != 8_150_000
        or contact_samples != 8_150_000
        or leg_qpos_samples != 81_500_000
        or control_samples != 815_000
        or applied_margin_transitions != 147
        or preclip_margin_samples != 819_362
    ):
        raise ValueError("formal adoption audited sample totals mismatch")

    phase_events = [
        event
        for audit in phase_audits
        for event in audit.get("events", ())
        if isinstance(event, Mapping)
    ]
    phase_counts = {
        expert: sum(event.get("current_expert") == expert for event in phase_events)
        for expert in FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES
    }
    if (
        len(phase_events) != 120
        or phase_counts != {expert: 40 for expert in phase_counts}
        or any(
            event.get("reset_preincrement_phase_index")
            != FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES[
                str(event.get("current_expert", ""))
            ]
            for event in phase_events
        )
        or sum(int(audit.get("exit_event_count", -1)) for audit in recovery_audits)
        != 60
        or sum(int(audit.get("active_tick_count", -1)) for audit in recovery_audits)
        != 780
        or sum(int(audit.get("cap_violation_count", -1)) for audit in recovery_audits)
        != 0
        or sum(int(audit.get("sample_count", -1)) for audit in recovery_audits)
        != 815_000
        or sum(int(audit.get("final_guard_call_count", -1)) for audit in recovery_audits)
        != 815_000
    ):
        raise ValueError("formal adoption phase/recovery totals mismatch")
    if (
        len(reset_audits) != 280
        or len(startup_audits) != 280
        or len(recovery_state_audits) != 280
        or not all(audit.get("passed") is True for audit in reset_audits)
        or not all(audit.get("passed") is True for audit in startup_audits)
        or not all(audit.get("passed") is True for audit in recovery_state_audits)
        or sum(int(audit.get("physical_safe_limit_violations", -1)) for audit in reset_audits)
        != 0
        or sum(int(audit.get("noise_margin_violations", -1)) for audit in reset_audits)
        != 0
        or any(audit.get("head_qpos_peak_rad") != 0.0 for audit in reset_audits)
        or any(
            audit.get("control_applied_before_first_physics_step") is not True
            or audit.get("exactly_one_guard_call_for_first_tick") is not True
            or audit.get("physics_steps_before_control") != 0
            or audit.get("guard_calls_for_first_tick") != 1
            for audit in startup_audits
        )
        or sum(int(audit.get("exit_event_count", -1)) for audit in recovery_state_audits)
        != 60
        or sum(int(audit.get("active_tick_count", -1)) for audit in recovery_state_audits)
        != 780
        or sum(int(audit.get("completed_event_count", -1)) for audit in recovery_state_audits)
        != 60
        or sum(int(audit.get("cap_violation_count", -1)) for audit in recovery_state_audits)
        != 0
        or sum(int(audit.get("control_tick_count", -1)) for audit in recovery_state_audits)
        != 815_000
        or sum(int(audit.get("final_guard_call_count", -1)) for audit in recovery_state_audits)
        != 815_000
    ):
        raise ValueError("formal adoption reset/startup/recovery audits mismatch")

    reverse_profiles = payload.get("reverse_profile_evidence")
    executed_profiles = (
        reverse_profiles.get("executed_profiles")
        if isinstance(reverse_profiles, Mapping)
        else None
    )
    policy = payload.get("policy_provenance")
    policy_roles = policy.get("roles") if isinstance(policy, Mapping) else None
    if (
        not isinstance(executed_profiles, Mapping)
        or {
            label: record.get("sha256")
            for label, record in executed_profiles.items()
            if isinstance(record, Mapping)
        }
        != dict(FORMAL_CANDIDATE_PROFILE_SHA256S)
        or not all(
            isinstance(record, Mapping)
            and record.get("schema_validated") is True
            and record.get("adopted") is False
            and record.get("adoption_eligible") is False
            for record in executed_profiles.values()
        )
        or not isinstance(policy_roles, Mapping)
        or set(policy_roles) != set(REQUIRED_POLICY_ROLES)
        or policy.get("adoption_eligible") is not True
        or any(
            not isinstance(record, Mapping)
            or record.get("sha256") != BASE_V22_POLICY_SHA256
            or record.get("formal_base_v22_allowlisted") is not True
            for record in policy_roles.values()
        )
    ):
        raise ValueError("formal adoption profile/policy binding mismatch")

    provenance = payload.get("runtime_dependency_provenance")
    pre = provenance.get("pre_import") if isinstance(provenance, Mapping) else None
    post = provenance.get("post_evaluation") if isinstance(provenance, Mapping) else None
    data_pre = (
        provenance.get("runtime_model_and_data_pre_evaluation")
        if isinstance(provenance, Mapping)
        else None
    )
    if not all(isinstance(value, Mapping) for value in (provenance, pre, post, data_pre)):
        raise ValueError("formal adoption provenance is incomplete")
    closure_contract = {
        "exp004_source_and_contract_snapshot": (
            "35f81cc82e1d073dc1ee0223751ed22a05db8498548f0a82ad3c77c45627dc2e",
            9,
        ),
        "external_hard_allowlisted_source_closure": (
            "a40d4920049b349334b6d5567859fc2f7533d8fe6648f5127f0c7e4ce54dc098",
            4,
        ),
        "hard_allowlisted_runtime_binary_closure": (
            "4e382762ffe85e33ba4839969088fed6f27cae1b35f6e0247a7d1b18937abe5f",
            5,
        ),
    }
    for label, (root_sha256, count) in closure_contract.items():
        before = pre.get(label)
        after = post.get(label)
        if (
            not isinstance(before, Mapping)
            or not isinstance(after, Mapping)
            or dict(before) != dict(after)
            or before.get("root_sha256") != root_sha256
            or before.get("dependency_count") != count
            or before.get("all_hashes_verified") is not True
        ):
            raise ValueError(f"formal adoption provenance mismatch: {label}")
    data_post = post.get("runtime_model_and_data_closure")
    if (
        provenance.get("verified") is not True
        or provenance.get("pre_post_source_and_data_hashes_unchanged") is not True
        or provenance.get("all_onnx_sessions_cpu_only_verified") is not True
        or not isinstance(data_post, Mapping)
        or dict(data_pre) != dict(data_post)
        or data_pre.get("root_sha256")
        != "6987cd08c733640e98b22e84943b888de2bca3d31da8cd4f5d86ca87044b8e2e"
        or data_pre.get("dependency_count") != 53
        or data_pre.get("all_hashes_verified") is not True
        or any(
            providers != ["CPUExecutionProvider"]
            for providers in provenance.get("onnx_session_execution_providers", {}).values()
        )
    ):
        raise ValueError("formal adoption runtime data/provider mismatch")

    adoption = payload.get("adoption_contract")
    reverse_adoption = payload.get("reverse_profile_adoption")
    command_gate = payload.get("command_mapping_contract", {}).get(
        "validation_status_gate"
    )
    phase = payload.get("formal_reverse_phase_entry_contract")
    recovery = payload.get("formal_backward_exit_recovery_contract")
    hardware = payload.get("hardware_gate")
    if (
        not isinstance(adoption, Mapping)
        or adoption.get("formal_candidate_pending") is not True
        or adoption.get("passed") is not False
        or not isinstance(reverse_adoption, Mapping)
        or reverse_adoption.get("passed") is not False
        or not isinstance(command_gate, Mapping)
        or command_gate.get("passed") is not False
        or command_gate.get("nonadoptable_case_count") != 6
        or not isinstance(phase, Mapping)
        or phase.get("status") != H2_5X15_SELECTION_STATUS
        or phase.get("adopted") is not False
        or phase.get("adoption_eligible") is not False
        or not isinstance(recovery, Mapping)
        or recovery.get("status") != H2_5X15_SELECTION_STATUS
        or recovery.get("adopted") is not False
        or recovery.get("adoption_eligible") is not False
        or not isinstance(hardware, Mapping)
        or hardware.get("status") != "PROHIBITED"
        or hardware.get("hardware_deployment_allowed") is not False
    ):
        raise ValueError("formal adoption source artifact was not fail-closed")

    return {
        "path": str(resolved),
        "sha256": digest,
        "hash_allowlisted_for_adoption": False,
        "status": H2_SUPERSEDED_ADOPTION_STATUS,
        "superseded_lineage_only": True,
        "selection_evidence_sha256": H2_5X15_SELECTION_EVIDENCE_SHA256,
        "profile_sha256s": dict(FORMAL_CANDIDATE_PROFILE_SHA256S),
        "policy_sha256": BASE_V22_POLICY_SHA256,
        "phase_preincrement_indices": dict(FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES),
        "profile_left_knee_cap": {
            "extra_upper_margin_rad": FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD,
            "upper_target_rad": FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_UPPER_TARGET_RAD,
        },
        "backward_exit_recovery": {
            "extra_upper_margin_rad": H2_SUPERSEDED_RECOVERY_EXTRA_UPPER_MARGIN_RAD,
            "upper_target_rad": H2_SUPERSEDED_RECOVERY_UPPER_TARGET_RAD,
            "hold_control_ticks": H2_SUPERSEDED_RECOVERY_HOLD_TICKS,
            "hold_seconds": H2_SUPERSEDED_RECOVERY_HOLD_SECONDS,
        },
        "reverse_command_case_names": sorted(FORMAL_REVERSE_COMMAND_CASE_NAMES),
        "suite_episode_count": len(episodes),
        "segment_pass_count": len(accepted_segments),
        "physics_substep_count": physics_samples,
        "contact_sample_count": contact_samples,
        "leg_qpos_sample_count": leg_qpos_samples,
        "control_sample_count": control_samples,
        "reset_startup_recovery_audit_count": len(reset_audits),
        "phase_entry_event_count": len(phase_events),
        "recovery_exit_event_count": 60,
        "recovery_active_tick_count": 780,
        "adoption_eligible": False,
        "simulation_acceptance_eligible": False,
        "package_release_evidence": False,
        "hardware_deployment": "PROHIBITED",
    }


def validate_formal_adoption_evidence(
    path: Path = FORMAL_ADOPTION_EVIDENCE_PATH,
) -> dict[str, Any]:
    """Strictly validate H3's immutable 20x30 adoption record.

    The source artifact is intentionally fail-closed because it was produced
    before adoption.  This validator re-derives its complete safety, motion,
    phase, recovery, policy, and provenance result before allowing its hash to
    authorize the simulation-only runtime.  It is never package-release or
    hardware evidence.
    """

    resolved = path.resolve()
    if resolved != FORMAL_ADOPTION_EVIDENCE_PATH:
        raise ValueError("formal H3 adoption evidence path must remain pinned")
    if not resolved.is_file():
        raise FileNotFoundError(f"missing formal H3 adoption evidence: {resolved}")
    digest = sha256_file(resolved)
    if (
        digest != FORMAL_ADOPTION_EVIDENCE_SHA256
        or digest not in FORMAL_ADOPTION_EVIDENCE_SHA256_ALLOWLIST
    ):
        raise ValueError("formal H3 adoption evidence hash mismatch")
    if resolved.stat().st_size != 18_597_453:
        raise ValueError("formal H3 adoption evidence size mismatch")
    payload = _load_strict_json_object(resolved, "formal H3 adoption evidence")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("evaluator_id") != EVALUATOR_ID
        or payload.get("evaluation_mode") != H3_CANDIDATE_SELECTION_STATUS
        or payload.get("simulation_suite_acceptance_passed") is not True
        or payload.get("simulation_acceptance_passed") is not False
    ):
        raise ValueError("formal H3 adoption source gate mismatch")

    selection = validate_formal_candidate_selection_evidence()
    safety_component = validate_h3_fast_exit_safety_evidence()
    superseded_h2 = validate_superseded_h2_adoption_evidence()

    def require_embedded(
        label: str,
        actual: Any,
        expected: Mapping[str, Any],
        expected_path: Path,
    ) -> None:
        if not isinstance(actual, Mapping):
            raise ValueError(f"formal H3 adoption {label} is missing")
        for key, expected_value in expected.items():
            actual_value = actual.get(key)
            if key == "path":
                if not isinstance(actual_value, str) or not _portable_workspace_path_matches(
                    actual_value, expected_path
                ):
                    raise ValueError(f"formal H3 adoption {label} path mismatch")
            elif actual_value != expected_value:
                raise ValueError(f"formal H3 adoption {label} mismatch: {key}")

    require_embedded(
        "candidate selection",
        payload.get("formal_candidate_selection_evidence"),
        selection,
        FORMAL_CANDIDATE_SELECTION_EVIDENCE_PATH,
    )
    require_embedded(
        "safety component",
        payload.get("h3_fast_exit_safety_component_evidence"),
        safety_component,
        H3_FAST_EXIT_SAFETY_EVIDENCE_PATH,
    )
    require_embedded(
        "superseded H2 adoption lineage",
        payload.get("superseded_h2_adoption_evidence"),
        superseded_h2,
        H2_SUPERSEDED_ADOPTION_EVIDENCE_PATH,
    )

    qualification = payload.get("release_qualification")
    configuration = payload.get("configuration")
    if not isinstance(qualification, Mapping) or not isinstance(configuration, Mapping):
        raise ValueError("formal H3 adoption metadata is incomplete")
    expected_scale = {
        "episodes": 20,
        "seconds": 30.0,
        "transition_seconds": 30.0,
        "transition_stand_seconds": 5.0,
        "warmup_seconds": 1.5,
        "initial_joint_noise_scale": 1.0,
        "initial_base_speed": 0.1,
        "master_seed": FORMAL_CANDIDATE_MASTER_SEED,
    }
    if (
        qualification.get("status") != "RELEASE_QUALIFICATION"
        or qualification.get("release_qualification_eligible") is not True
        or qualification.get("scale_matches_frozen_contract") is not True
        or qualification.get("diagnostic_mode_disabled") is not True
        or qualification.get("master_seed_matches_recommendation") is not True
        or dict(qualification.get("actual", {})) != expected_scale
        or configuration.get("seed") != FORMAL_CANDIDATE_MASTER_SEED
        or configuration.get("formal_candidate_default") is not True
        or configuration.get("formal_candidate_status")
        != H3_CANDIDATE_SELECTION_STATUS
        or configuration.get("formal_adopted_default") is not False
        or configuration.get("formal_adopted_status") is not None
        or configuration.get("backward_residual_scale") != 0.0
        or configuration.get("leg_target_margin_rad") != LEG_TARGET_MARGIN_RAD
        or configuration.get("target_slew_rate_rad_per_s")
        != TARGET_SLEW_LIMIT_RAD_PER_S
        or configuration.get("reset_noise_margin_rad") != RESET_NOISE_MARGIN_RAD
        or configuration.get("left_knee_extra_upper_margin_rad")
        != FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
        or configuration.get("left_knee_profile_upper_target_rad")
        != FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_UPPER_TARGET_RAD
        or configuration.get("backward_exit_recovery_enabled") is not True
        or any(
            configuration.get(key)
            for key in (
                "diagnostic_unadopted_policy",
                "diagnostic_unadopted_backward_exit_recovery",
                "diagnostic_noncontract_safety",
                "policy_command_diagnostic_suite",
            )
        )
        or any(
            configuration.get(key) is not None
            for key in (
                "diagnostic_unadopted_reverse_profile",
                "diagnostic_unadopted_reverse_left_profile",
                "diagnostic_unadopted_reverse_right_profile",
                "diagnostic_unadopted_reverse_entry_phase_indices",
            )
        )
        or dict(configuration.get("executed_reverse_entry_phase_indices", {}))
        != dict(FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES)
    ):
        raise ValueError("formal H3 adoption scale/configuration mismatch")
    executed_paths = configuration.get("executed_reverse_profile_paths")
    if not isinstance(executed_paths, Mapping) or any(
        not isinstance(executed_paths.get(label), str)
        or not _portable_workspace_path_matches(
            str(executed_paths[label]), FORMAL_CANDIDATE_PROFILE_PATHS[label]
        )
        for label in ("straight", "left", "right")
    ):
        raise ValueError("formal H3 adoption profile path mismatch")

    bundle = payload.get("formal_candidate_execution_bundle")
    if (
        not isinstance(bundle, Mapping)
        or bundle.get("passed") is not True
        or bundle.get("status") != H3_CANDIDATE_SELECTION_STATUS
        or bundle.get("candidate_selection_evidence_sha256")
        != FORMAL_CANDIDATE_SELECTION_EVIDENCE_SHA256
        or bundle.get("safety_component_evidence_sha256")
        != H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256
        or bundle.get("superseded_h2_adoption_evidence_sha256")
        != H2_SUPERSEDED_ADOPTION_EVIDENCE_SHA256
        or dict(bundle.get("profile_sha256s", {}))
        != dict(FORMAL_CANDIDATE_PROFILE_SHA256S)
        or bundle.get("policy_sha256") != BASE_V22_POLICY_SHA256
        or set(bundle.get("policy_roles", ())) != set(REQUIRED_POLICY_ROLES)
        or dict(bundle.get("phase_preincrement_indices", {}))
        != dict(FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES)
        or bundle.get("profile_left_knee_cap")
        != {
            "extra_upper_margin_rad": FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD,
            "upper_target_rad": FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_UPPER_TARGET_RAD,
        }
        or bundle.get("backward_exit_recovery")
        != {
            "enabled": True,
            "extra_upper_margin_rad": BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD,
            "hold_control_ticks": BACKWARD_EXIT_RECOVERY_HOLD_TICKS,
            "hold_seconds": BACKWARD_EXIT_RECOVERY_HOLD_SECONDS,
            "upper_target_rad": BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD,
        }
        or bundle.get("adopted") is not False
        or bundle.get("adoption_eligible") is not False
        or bundle.get("simulation_acceptance_eligible") is not False
        or bundle.get("hardware_deployment") != "PROHIBITED"
    ):
        raise ValueError("formal H3 adoption execution-bundle mismatch")

    suites = payload.get("suites")
    expected_suites = {
        "primitives": (list(range(20_260_808, 20_260_828)), 140),
        "compounds": (list(range(21_260_808, 21_260_828)), 120),
        "transitions": (list(range(22_260_808, 22_260_828)), 500),
    }
    if not isinstance(suites, Mapping) or set(suites) != set(expected_suites):
        raise ValueError("formal H3 adoption suite set mismatch")
    episodes: list[Mapping[str, Any]] = []
    segments: list[Mapping[str, Any]] = []
    accepted_segments: list[Mapping[str, Any]] = []
    reset_audits: list[Mapping[str, Any]] = []
    startup_audits: list[Mapping[str, Any]] = []
    recovery_state_audits: list[Mapping[str, Any]] = []
    for suite_name, (expected_seeds, expected_segment_count) in expected_suites.items():
        suite = suites[suite_name]
        suite_episodes = suite.get("episodes") if isinstance(suite, Mapping) else None
        acceptance = suite.get("acceptance") if isinstance(suite, Mapping) else None
        checks = acceptance.get("episode_checks") if isinstance(acceptance, Mapping) else None
        if (
            not isinstance(suite_episodes, list)
            or [episode.get("seed") for episode in suite_episodes] != expected_seeds
            or not isinstance(checks, list)
            or [check.get("seed") for check in checks] != expected_seeds
            or acceptance.get("passed") is not True
            or not all(isinstance(check, Mapping) and check.get("passed") is True for check in checks)
        ):
            raise ValueError(f"formal H3 adoption {suite_name} acceptance mismatch")
        suite_segments = [
            segment
            for episode in suite_episodes
            for segment in episode.get("segments", ())
            if isinstance(segment, Mapping)
        ]
        suite_accepted = [
            segment
            for check in checks
            for segment in check.get("segments", ())
            if isinstance(segment, Mapping)
        ]
        if (
            len(suite_segments) != expected_segment_count
            or len(suite_accepted) != expected_segment_count
            or not all(
                record.get("passed") is True
                and isinstance(record.get("checks"), Mapping)
                and len(record["checks"]) == 37
                and all(value is True for value in record["checks"].values())
                for record in suite_accepted
            )
        ):
            raise ValueError(f"formal H3 adoption {suite_name} segment checks failed")
        for check in checks:
            reset_audits.extend(
                item for item in check.get("reset_qpos_audits", ()) if isinstance(item, Mapping)
            )
            startup_audits.extend(
                item for item in check.get("control_first_startup_audits", ()) if isinstance(item, Mapping)
            )
            recovery_state_audits.extend(
                item for item in check.get("backward_exit_recovery_audits", ()) if isinstance(item, Mapping)
            )
        episodes.extend(suite_episodes)
        segments.extend(suite_segments)
        accepted_segments.extend(suite_accepted)
    if (
        len(episodes) != 60
        or len(segments) != 760
        or len(accepted_segments) != 760
        or sum(len(item["checks"]) for item in accepted_segments) != 28_120
        or not all(
            episode.get("fell") is False
            and episode.get("completed_segment_count") == episode.get("requested_segment_count")
            for episode in episodes
        )
        or not all(
            segment.get("completed") is True
            and segment.get("fell") is False
            and segment.get("completed_physics_substeps") == segment.get("expected_physics_substeps")
            and segment_acceptance(segment).get("passed") is True
            for segment in segments
        )
    ):
        raise ValueError("formal H3 adoption episode/segment completion mismatch")

    safety_zero_fields = (
        "applied_target_limit_violations",
        "desired_target_margin_violations",
        "nonfinite_sample_count",
        "preclip_target_limit_violations",
        "qpos_limit_violations",
        "target_slew_violations",
        "unauthorized_applied_target_margin_violations",
        "maximum_applied_target_excess_rad",
        "maximum_desired_target_margin_excess_rad",
        "maximum_preclip_target_excess_rad",
        "maximum_qpos_excess_rad",
        "applied_head_action_peak",
        "head_target_peak_rad",
    )
    physics_zero_fields = (
        "height_fall_samples",
        "upright_fall_samples",
        "nonfinite_full_qpos_samples",
        "nonfinite_full_qvel_samples",
        "nonfinite_leg_qpos_samples",
        "nonfinite_pose_samples",
        "nonfinite_state_samples",
        "qpos_limit_violations",
        "maximum_qpos_excess_rad",
    )
    physics_samples = contact_samples = leg_samples = control_samples = 0
    applied_margin = startup_margin = preclip_margin = 0
    phase_audits: list[Mapping[str, Any]] = []
    recovery_audits: list[Mapping[str, Any]] = []
    minimum_height = minimum_upright = np.inf
    maximum_left_knee = -np.inf
    for segment in segments:
        safety = segment.get("safety_audit")
        physics = segment.get("physics_substep_audit")
        routing = segment.get("routing")
        recovery = segment.get("backward_exit_recovery_audit")
        if not all(isinstance(value, Mapping) for value in (safety, physics, routing, recovery)):
            raise ValueError("formal H3 adoption segment audit is incomplete")
        phase = routing.get("reverse_entry_phase")
        if (
            any(safety.get(field) != 0 for field in safety_zero_fields)
            or safety.get("applied_target_margin_violations")
            != safety.get("startup_margin_transition_joint_samples")
            or safety.get("leg_target_margin_rad") != LEG_TARGET_MARGIN_RAD
            or safety.get("target_slew_limit_rad_per_s") != TARGET_SLEW_LIMIT_RAD_PER_S
            or safety.get("maximum_target_slew_rate_rad_per_s")
            > TARGET_SLEW_LIMIT_RAD_PER_S + 2e-15
            or any(physics.get(field) != 0 for field in physics_zero_fields)
            or physics.get("fall_or_nonfinite_detected") is not False
            or physics.get("contact_sample_count_matches_sample_count") is not True
            or physics.get("contact_sample_count") != physics.get("sample_count")
            or routing.get("command_clip_events") != 0
            or routing.get("prohibited_expert_steps") != 0
            or routing.get("atomic_endpoint_mismatch_steps") != 0
            or not isinstance(phase, Mapping)
            or phase.get("passed") is not True
            or recovery.get("passed") is not True
            or recovery.get("cap_violation_count") != 0
        ):
            raise ValueError("formal H3 adoption safety/physics/routing audit failed")
        physics_samples += int(physics["sample_count"])
        contact_samples += int(physics["contact_sample_count"])
        leg_samples += int(physics["leg_joint_sample_count"])
        control_samples += int(safety["sample_count"])
        applied_margin += int(safety["applied_target_margin_violations"])
        startup_margin += int(safety["startup_margin_transition_joint_samples"])
        preclip_margin += int(safety["preclip_target_margin_violations"])
        minimum_height = min(minimum_height, float(physics["minimum_height_m"]))
        minimum_upright = min(minimum_upright, float(physics["minimum_upright"]))
        maximum_left_knee = max(
            maximum_left_knee,
            float(physics["joint_qpos_max_rad"]["left_knee"]),
        )
        phase_audits.append(phase)
        recovery_audits.append(recovery)
    if (
        physics_samples != 8_150_000
        or contact_samples != 8_150_000
        or leg_samples != 81_500_000
        or control_samples != 815_000
        or applied_margin != 147
        or startup_margin != 147
        or preclip_margin != 819_203
        or minimum_height != 0.17911993
        or minimum_upright != 0.9777608163890137
        or maximum_left_knee != 0.4736497298325716
        or SAFE_JOINT_LIMITS["left_knee"][1] - maximum_left_knee
        != 0.0018842701674284257
    ):
        raise ValueError("formal H3 adoption audited aggregate mismatch")

    phase_events = [
        event
        for audit in phase_audits
        for event in audit.get("events", ())
        if isinstance(event, Mapping)
    ]
    phase_counts = {
        expert: sum(event.get("current_expert") == expert for event in phase_events)
        for expert in FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES
    }
    if (
        len(phase_events) != 120
        or phase_counts != {expert: 40 for expert in phase_counts}
        or any(
            event.get("reset_preincrement_phase_index")
            != FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES[str(event.get("current_expert", ""))]
            for event in phase_events
        )
        or sum(int(audit.get("exit_event_count", -1)) for audit in recovery_audits) != 60
        or sum(int(audit.get("active_tick_count", -1)) for audit in recovery_audits) != 780
        or sum(int(audit.get("cap_violation_count", -1)) for audit in recovery_audits) != 0
        or sum(int(audit.get("sample_count", -1)) for audit in recovery_audits) != 815_000
        or sum(int(audit.get("final_guard_call_count", -1)) for audit in recovery_audits) != 815_000
    ):
        raise ValueError("formal H3 adoption phase/recovery totals mismatch")
    if (
        len(reset_audits) != 280
        or len(startup_audits) != 280
        or len(recovery_state_audits) != 280
        or not all(audit.get("passed") is True for audit in reset_audits)
        or not all(audit.get("passed") is True for audit in startup_audits)
        or not all(audit.get("passed") is True for audit in recovery_state_audits)
        or sum(int(audit.get("exit_event_count", -1)) for audit in recovery_state_audits) != 60
        or sum(int(audit.get("active_tick_count", -1)) for audit in recovery_state_audits) != 780
        or sum(int(audit.get("completed_event_count", -1)) for audit in recovery_state_audits) != 60
        or sum(int(audit.get("cap_violation_count", -1)) for audit in recovery_state_audits) != 0
        or sum(int(audit.get("remaining_ticks", -1)) for audit in recovery_state_audits) != 0
        or sum(int(audit.get("control_tick_count", -1)) for audit in recovery_state_audits) != 815_000
        or sum(int(audit.get("final_guard_call_count", -1)) for audit in recovery_state_audits) != 815_000
    ):
        raise ValueError("formal H3 adoption reset/startup/recovery audit mismatch")

    linear_segments = [
        segment for segment in segments if segment["metrics"]["commanded_linear_speed"] > 0.0
    ]
    yaw_segments = [
        segment for segment in segments if abs(segment["metrics"]["command"][2]) > 0.0
    ]
    moving_segments = [
        segment for segment in segments if any(abs(value) > 0.0 for value in segment["metrics"]["command"])
    ]
    stand_segments = [segment for segment in segments if segment not in moving_segments]
    yaw_only_segments = [
        segment
        for segment in yaw_segments
        if segment["metrics"]["commanded_linear_speed"] == 0.0
    ]
    uncommanded_yaw_segments = [
        segment for segment in segments if abs(segment["metrics"]["command"][2]) == 0.0
    ]
    performance = {
        "minimum_signed_linear_progress_fraction": min(
            segment["metrics"]["projected_primary_velocity"]
            / segment["metrics"]["commanded_linear_speed"]
            for segment in linear_segments
        ),
        "minimum_signed_yaw_progress_fraction": min(
            np.sign(segment["metrics"]["command"][2])
            * segment["metrics"]["mean_local_yaw_rate"]
            / abs(segment["metrics"]["command"][2])
            for segment in yaw_segments
        ),
        "maximum_primary_velocity_error_mps": max(
            segment["metrics"]["primary_velocity_error"] for segment in linear_segments
        ),
        "maximum_orthogonal_velocity_mps": max(
            segment["metrics"]["absolute_orthogonal_velocity"] for segment in linear_segments
        ),
        "maximum_yaw_only_planar_velocity_mps": max(
            segment["metrics"]["absolute_orthogonal_velocity"]
            for segment in yaw_only_segments
        ),
        "maximum_yaw_rate_error_radps": max(
            segment["metrics"]["yaw_rate_error"] for segment in yaw_segments
        ),
        "maximum_uncommanded_yaw_rate_radps": max(
            segment["metrics"]["uncommanded_yaw_rate"]
            for segment in uncommanded_yaw_segments
        ),
        "maximum_stop_drift_m": max(
            segment["metrics"]["planar_displacement"] for segment in stand_segments
        ),
        "minimum_moving_single_support_rate": min(
            segment["metrics"]["single_support_rate"] for segment in moving_segments
        ),
        "maximum_flight_rate": max(
            segment["metrics"]["flight_rate"] for segment in moving_segments
        ),
    }
    expected_performance = {
        "minimum_signed_linear_progress_fraction": 0.3595826926676137,
        "minimum_signed_yaw_progress_fraction": 0.4936553773470118,
        "maximum_primary_velocity_error_mps": 0.028009207275213346,
        "maximum_orthogonal_velocity_mps": 0.024010891338336268,
        "maximum_yaw_only_planar_velocity_mps": 0.02668671634496424,
        "maximum_yaw_rate_error_radps": 0.10269293300510374,
        "maximum_uncommanded_yaw_rate_radps": 0.1395255079553353,
        "maximum_stop_drift_m": 0.04004149774890449,
        "minimum_moving_single_support_rate": 0.11593333333333333,
        "maximum_flight_rate": 0.005666666666666667,
    }
    if performance != expected_performance:
        raise ValueError(
            "formal H3 adoption performance extrema mismatch: "
            f"actual={performance!r}, expected={expected_performance!r}"
        )

    profiles = payload.get("reverse_profile_evidence")
    executed_profiles = profiles.get("executed_profiles") if isinstance(profiles, Mapping) else None
    policy = payload.get("policy_provenance")
    policy_roles = policy.get("roles") if isinstance(policy, Mapping) else None
    if (
        not isinstance(executed_profiles, Mapping)
        or {label: record.get("sha256") for label, record in executed_profiles.items()}
        != dict(FORMAL_CANDIDATE_PROFILE_SHA256S)
        or any(
            not isinstance(record, Mapping)
            or record.get("schema_validated") is not True
            or record.get("composition", {}).get("left_knee_extra_upper_margin_rad")
            != FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
            for record in executed_profiles.values()
        )
        or not isinstance(policy_roles, Mapping)
        or set(policy_roles) != set(REQUIRED_POLICY_ROLES)
        or policy.get("mode") != "FORMAL_BASE_V22_ONLY"
        or policy.get("adoption_eligible") is not True
        or policy.get("all_roles_allowlisted") is not True
        or any(
            not isinstance(record, Mapping)
            or record.get("sha256") != BASE_V22_POLICY_SHA256
            or record.get("formal_base_v22_allowlisted") is not True
            for record in policy_roles.values()
        )
    ):
        raise ValueError("formal H3 adoption profile/policy provenance mismatch")

    provenance = payload.get("runtime_dependency_provenance")
    pre = provenance.get("pre_import") if isinstance(provenance, Mapping) else None
    post = provenance.get("post_evaluation") if isinstance(provenance, Mapping) else None
    data_pre = provenance.get("runtime_model_and_data_pre_evaluation") if isinstance(provenance, Mapping) else None
    environment = provenance.get("runtime_environment") if isinstance(provenance, Mapping) else None
    if not all(isinstance(value, Mapping) for value in (provenance, pre, post, data_pre, environment)):
        raise ValueError("formal H3 adoption provenance is incomplete")
    closure_contract = {
        "exp004_source_and_contract_snapshot": (
            "91e2d1db37e5fd704b0fb35f2d08df4aadb5d0de14273eed96f0047847c064c4",
            9,
        ),
        "external_hard_allowlisted_source_closure": (
            "a40d4920049b349334b6d5567859fc2f7533d8fe6648f5127f0c7e4ce54dc098",
            4,
        ),
        "hard_allowlisted_runtime_binary_closure": (
            "4e382762ffe85e33ba4839969088fed6f27cae1b35f6e0247a7d1b18937abe5f",
            5,
        ),
    }
    for label, (root_sha256, count) in closure_contract.items():
        before = pre.get(label)
        after = post.get(label)
        if (
            not isinstance(before, Mapping)
            or not isinstance(after, Mapping)
            or dict(before) != dict(after)
            or before.get("root_sha256") != root_sha256
            or before.get("dependency_count") != count
            or before.get("all_hashes_verified") is not True
        ):
            raise ValueError(f"formal H3 adoption provenance mismatch: {label}")
    data_post = post.get("runtime_model_and_data_closure")
    entries = data_pre.get("entries")
    if (
        provenance.get("verified") is not True
        or provenance.get("pre_post_source_and_data_hashes_unchanged") is not True
        or provenance.get("all_onnx_sessions_cpu_only_verified") is not True
        or not isinstance(data_post, Mapping)
        or dict(data_pre) != dict(data_post)
        or data_pre.get("root_sha256")
        != "29fb3f01dc07552c82ce73889aa73bbd144499b2558b311982490ed9659e1a1a"
        or data_pre.get("dependency_count") != 56
        or data_pre.get("all_hashes_verified") is not True
        or not isinstance(entries, Mapping)
        or entries.get("formal_candidate_selection_evidence", {}).get("sha256")
        != FORMAL_CANDIDATE_SELECTION_EVIDENCE_SHA256
        or entries.get("h3_fast_exit_safety_component_evidence", {}).get("sha256")
        != H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256
        or entries.get("formal_adoption_evidence", {}).get("sha256")
        != H2_SUPERSEDED_ADOPTION_EVIDENCE_SHA256
        or entries.get("superseded_h2_candidate_selection_evidence", {}).get("sha256")
        != H2_5X15_SELECTION_EVIDENCE_SHA256
        or environment.get("exact_versions_verified") is not True
        or dict(environment.get("actual", {})) != dict(FROZEN_RUNTIME_VERSIONS)
        or environment.get("onnxruntime_build_commit_verified") != "45de2a8b06"
        or any(
            providers != ["CPUExecutionProvider"]
            for providers in provenance.get("onnx_session_execution_providers", {}).values()
        )
    ):
        raise ValueError("formal H3 adoption runtime data/provider mismatch")

    adoption = payload.get("adoption_contract")
    reverse_adoption = payload.get("reverse_profile_adoption")
    command_gate = payload.get("command_mapping_contract", {}).get("validation_status_gate")
    phase_contract = payload.get("formal_reverse_phase_entry_contract")
    recovery_contract = payload.get("formal_backward_exit_recovery_contract")
    hardware = payload.get("hardware_gate")
    if (
        not isinstance(adoption, Mapping)
        or adoption.get("formal_candidate_pending") is not True
        or adoption.get("passed") is not False
        or not isinstance(reverse_adoption, Mapping)
        or reverse_adoption.get("passed") is not False
        or not isinstance(command_gate, Mapping)
        or command_gate.get("passed") is not False
        or command_gate.get("safety_component_passed") is not True
        or command_gate.get("reverse_safety_component_evidence_failure_count") != 0
        or command_gate.get("reverse_adoption_evidence_failure_count") != 6
        or not isinstance(phase_contract, Mapping)
        or phase_contract.get("status") != H3_CANDIDATE_SELECTION_STATUS
        or phase_contract.get("adopted") is not False
        or phase_contract.get("adoption_eligible") is not False
        or not isinstance(recovery_contract, Mapping)
        or recovery_contract.get("status") != H3_CANDIDATE_SELECTION_STATUS
        or recovery_contract.get("adopted") is not False
        or recovery_contract.get("adoption_eligible") is not False
        or recovery_contract.get("runtime_contract", {}).get("extra_upper_margin_rad")
        != BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD
        or recovery_contract.get("runtime_contract", {}).get("upper_target_rad")
        != BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD
        or not isinstance(hardware, Mapping)
        or hardware.get("status") != "PROHIBITED"
        or hardware.get("hardware_deployment_allowed") is not False
    ):
        raise ValueError("formal H3 adoption source artifact was not fail-closed")

    return {
        "path": str(resolved),
        "sha256": digest,
        "hash_allowlisted_for_adoption": True,
        "status": FORMAL_CANDIDATE_STATUS,
        "selection_evidence_sha256": FORMAL_CANDIDATE_SELECTION_EVIDENCE_SHA256,
        "safety_component_evidence_sha256": H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256,
        "superseded_h2_adoption_evidence_sha256": H2_SUPERSEDED_ADOPTION_EVIDENCE_SHA256,
        "profile_sha256s": dict(FORMAL_CANDIDATE_PROFILE_SHA256S),
        "policy_sha256": BASE_V22_POLICY_SHA256,
        "phase_preincrement_indices": dict(FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES),
        "profile_left_knee_cap": {
            "extra_upper_margin_rad": FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD,
            "upper_target_rad": FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_UPPER_TARGET_RAD,
        },
        "backward_exit_recovery": {
            "extra_upper_margin_rad": BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD,
            "upper_target_rad": BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD,
            "hold_control_ticks": BACKWARD_EXIT_RECOVERY_HOLD_TICKS,
            "hold_seconds": BACKWARD_EXIT_RECOVERY_HOLD_SECONDS,
        },
        "reverse_command_case_names": sorted(FORMAL_REVERSE_COMMAND_CASE_NAMES),
        "suite_episode_count": 60,
        "segment_pass_count": 760,
        "acceptance_check_true_count": 28_120,
        "physics_substep_count": physics_samples,
        "contact_sample_count": contact_samples,
        "leg_qpos_sample_count": leg_samples,
        "control_sample_count": control_samples,
        "reset_startup_recovery_audit_count": 280,
        "phase_entry_event_count": len(phase_events),
        "recovery_exit_event_count": 60,
        "recovery_active_tick_count": 780,
        "authorized_startup_margin_transition_joint_samples": startup_margin,
        "preclip_target_margin_violations": preclip_margin,
        "maximum_left_knee_qpos_rad": maximum_left_knee,
        "minimum_left_knee_safe_upper_margin_rad": (
            SAFE_JOINT_LIMITS["left_knee"][1] - maximum_left_knee
        ),
        "minimum_height_m": minimum_height,
        "minimum_upright": minimum_upright,
        "performance_extrema": performance,
        "pre_post_provenance_unchanged": True,
        "adoption_eligible": True,
        "simulation_acceptance_eligible": True,
        "package_release_evidence": False,
        "hardware_deployment": "PROHIBITED",
    }


def validate_policy_provenance(
    policies: Mapping[str, Path], *, diagnostic_unadopted: bool
) -> dict[str, Any]:
    """Pin all formal roles to base-v22 or explicitly quarantine diagnostics."""

    if set(policies) != set(REQUIRED_POLICY_ROLES):
        raise ValueError("policy provenance requires exactly the eight formal roles")
    roles: dict[str, Any] = {}
    for role in REQUIRED_POLICY_ROLES:
        path = policies[role].resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing {role} ONNX policy: {path}")
        if path.suffix.lower() != ".onnx":
            raise ValueError(f"{role} policy must be an ONNX file")
        digest = sha256_file(path)
        allowlisted = digest in FORMAL_POLICY_SHA256_ALLOWLIST
        if not diagnostic_unadopted and not allowlisted:
            raise ValueError(
                f"formal {role} policy is not frozen base-v22: {digest}"
            )
        roles[role] = {
            "path": str(path),
            "sha256": digest,
            "formal_base_v22_allowlisted": allowlisted,
            "adopted": bool(allowlisted and not diagnostic_unadopted),
        }
    all_allowlisted = all(
        record["formal_base_v22_allowlisted"] for record in roles.values()
    )
    return {
        "mode": (
            "DIAGNOSTIC_UNADOPTED_POLICY"
            if diagnostic_unadopted
            else "FORMAL_BASE_V22_ONLY"
        ),
        "required_roles": list(REQUIRED_POLICY_ROLES),
        "formal_sha256_allowlist": sorted(FORMAL_POLICY_SHA256_ALLOWLIST),
        "all_roles_allowlisted": all_allowlisted,
        "diagnostic_unadopted": bool(diagnostic_unadopted),
        "adoption_eligible": bool(all_allowlisted and not diagnostic_unadopted),
        "roles": roles,
    }


def generated_asset_paths(generated_root: Path) -> dict[str, Path]:
    package = generated_root / "playground" / "open_duck_mini_v2"
    return {
        "manifest": generated_root / "hardware_safe_manifest.json",
        "scene": package
        / "xmls"
        / "scene_flat_terrain_backlash_hardware_safe_calibrated.xml",
        "model": package
        / "xmls"
        / "open_duck_mini_v2_backlash_hardware_safe_calibrated.xml",
        "reference": package / "data" / "polynomial_coefficients_calibrated.pkl",
        "backward_default": package / "data" / "optimized_backward_gait.json",
        "backward_right": package
        / "data"
        / "optimized_backward_right_turn_gait.json",
    }


def _require_under_root(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the frozen generated root: {resolved}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    return resolved


def _portable_workspace_path_matches(recorded_path: str, actual_path: Path) -> bool:
    """Compare a recorded Windows/WSL path by its immutable workspace suffix."""

    normalized = recorded_path.replace("\\", "/")
    marker = "/experiments/"
    marker_index = normalized.lower().find(marker)
    if marker_index >= 0:
        recorded_relative = normalized[marker_index + 1 :]
    elif normalized.lower().startswith("experiments/"):
        recorded_relative = normalized
    else:
        workspace_marker = f"/{WORKSPACE_ROOT.name.lower()}/"
        workspace_index = normalized.lower().find(workspace_marker)
        if workspace_index < 0:
            return False
        recorded_relative = normalized[
            workspace_index + len(workspace_marker) :
        ]
    try:
        actual_relative = actual_path.resolve().relative_to(WORKSPACE_ROOT).as_posix()
    except ValueError:
        return False
    return recorded_relative == actual_relative


def discover_mjcf_dependency_closure(
    scene: Path, generated_root: Path
) -> dict[str, dict[str, str]]:
    """Resolve every transitive local XML, mesh, hfield, and texture file."""

    root = generated_root.resolve()
    scene = _require_under_root(scene, root, "generated scene")
    visited_xml: set[Path] = set()
    dependencies: dict[Path, str] = {}

    def register(path: Path, kind: str, label: str) -> Path:
        resolved = _require_under_root(path, root, label)
        previous = dependencies.get(resolved)
        if previous is not None and previous != kind:
            raise ValueError(
                f"MJCF dependency has conflicting kinds {previous}/{kind}: {resolved}"
            )
        dependencies[resolved] = kind
        return resolved

    def local_asset_path(xml_path: Path, directory: str, filename: str, label: str) -> Path:
        candidate = Path(filename)
        directory_path = Path(directory)
        if candidate.is_absolute() or directory_path.is_absolute():
            raise ValueError(f"{label} must be generated-root-relative")
        return xml_path.parent / directory_path / candidate

    def visit(xml_path: Path) -> None:
        xml_path = _require_under_root(xml_path, root, "MJCF XML")
        if xml_path in visited_xml:
            return
        visited_xml.add(xml_path)
        try:
            xml_root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            raise ValueError(f"invalid MJCF XML: {xml_path}") from exc
        compiler = xml_root.find("compiler")
        compiler_values = {} if compiler is None else compiler.attrib
        asset_dir = compiler_values.get("assetdir", "")
        mesh_dir = compiler_values.get("meshdir", asset_dir)
        texture_dir = compiler_values.get("texturedir", asset_dir)

        for include in xml_root.iter("include"):
            filename = include.get("file", "")
            if not filename:
                raise ValueError(f"MJCF include is missing file: {xml_path}")
            included = register(
                local_asset_path(xml_path, "", filename, "MJCF include"),
                "xml",
                "MJCF include",
            )
            visit(included)
        asset_specs = (
            ("mesh", mesh_dir, "mesh"),
            ("hfield", asset_dir, "hfield"),
            ("texture", texture_dir, "texture"),
        )
        for tag, directory, kind in asset_specs:
            for element in xml_root.iter(tag):
                filename = element.get("file")
                if filename:
                    register(
                        local_asset_path(xml_path, directory, filename, f"MJCF {kind}"),
                        kind,
                        f"MJCF {kind}",
                    )

    visit(scene)
    return {
        path.relative_to(root).as_posix(): {
            "kind": kind,
            "sha256": sha256_file(path),
        }
        for path, kind in sorted(
            dependencies.items(), key=lambda item: item[0].as_posix()
        )
    }


def dependency_closure_root_sha256(
    closure: Mapping[str, Mapping[str, str]]
) -> str:
    canonical = "".join(
        f"{relative}\0{record['kind']}\0{record['sha256']}\n"
        for relative, record in sorted(closure.items())
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_exact_generated_assets(generated_root: Path) -> dict[str, Any]:
    """Verify the immutable generated root and its complete MJCF closure."""

    resolved_root = generated_root.resolve()
    if resolved_root != FROZEN_GENERATED_ROOT:
        raise ValueError(
            "generated root must remain pinned to "
            f"{FROZEN_GENERATED_ROOT}, got {resolved_root}"
        )
    paths = generated_asset_paths(resolved_root)
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing generated {label}: {path}")
    primary_hashes = {
        label: sha256_file(paths[label])
        for label in ("manifest", "scene", "model", "reference")
    }
    for label, expected in FROZEN_GENERATED_PRIMARY_SHA256.items():
        if primary_hashes[label] != expected:
            raise ValueError(
                f"generated {label} immutable hash mismatch: "
                f"expected {expected}, got {primary_hashes[label]}"
            )
    manifest = _load_strict_json_object(paths["manifest"], "generated manifest")
    if manifest.get("contract") != "hardware_safe_simulation_only":
        raise ValueError("generated manifest is not the hardware-safe simulation contract")
    if manifest.get("real_hardware_deployment_allowed") is not False:
        raise ValueError("generated manifest must prohibit hardware deployment")

    manifest_keys = {
        "scene": "generated_scene",
        "model": "generated_model",
        "reference": "generated_reference",
        "backward_default": "legacy_v22_optimized_backward_gait",
        "backward_right": "legacy_v22_optimized_backward_right_turn_gait",
    }
    verified: dict[str, Any] = {}
    for label, manifest_key in manifest_keys.items():
        actual = sha256_file(paths[label])
        files = manifest.get("files")
        if not isinstance(files, Mapping) or not isinstance(
            files.get(manifest_key), Mapping
        ):
            raise ValueError(f"generated manifest is missing files.{manifest_key}")
        expected = str(files[manifest_key].get("sha256", ""))
        if actual != expected:
            raise ValueError(
                f"generated {label} hash mismatch: expected {expected}, got {actual}"
            )
        if label in FROZEN_GENERATED_PRIMARY_SHA256 and (
            actual != FROZEN_GENERATED_PRIMARY_SHA256[label]
        ):
            raise ValueError(f"generated {label} is not the frozen release")
        verified[label] = {
            "path": str(paths[label]),
            "sha256": actual,
            "manifest_key": manifest_key,
        }
    closure = discover_mjcf_dependency_closure(paths["scene"], resolved_root)
    expected_closure = {
        relative: dict(record)
        for relative, record in FROZEN_GENERATED_DEPENDENCY_SHA256.items()
    }
    if set(closure) != set(expected_closure):
        raise ValueError(
            "generated dependency closure mismatch: "
            f"missing={sorted(set(expected_closure) - set(closure))}, "
            f"unexpected={sorted(set(closure) - set(expected_closure))}"
        )
    for relative, expected_record in expected_closure.items():
        if closure[relative] != expected_record:
            raise ValueError(
                f"generated dependency immutable hash/kind mismatch: {relative}"
            )
    closure_root = dependency_closure_root_sha256(closure)
    if closure_root != FROZEN_GENERATED_DEPENDENCY_ROOT_SHA256:
        raise ValueError("generated dependency closure root hash mismatch")
    return {
        "contract": manifest["contract"],
        "real_hardware_deployment_allowed": False,
        "generated_root": {
            "path": str(resolved_root),
            "pinned": True,
        },
        "manifest": {
            "path": str(paths["manifest"]),
            "sha256": primary_hashes["manifest"],
        },
        "verified_files": verified,
        "dependency_closure": {
            "entry_count": len(closure),
            "xml_count": sum(
                record["kind"] == "xml" for record in closure.values()
            ),
            "mesh_count": sum(
                record["kind"] == "mesh" for record in closure.values()
            ),
            "hfield_count": sum(
                record["kind"] == "hfield" for record in closure.values()
            ),
            "texture_count": sum(
                record["kind"] == "texture" for record in closure.values()
            ),
            "root_sha256": closure_root,
            "entries": closure,
        },
    }


def validate_reverse_profile_schema(
    path: Path, *, diagnostic_unadopted: bool
) -> dict[str, Any]:
    """Validate profile structure, all finite values, and the dynamic knee cap."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing reverse profile: {resolved}")
    payload = _load_strict_json_object(resolved, "reverse profile")
    parameters = payload.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("reverse profile is missing parameters")
    scales = parameters.get("joint_amplitude_scales")
    if not isinstance(scales, list) or len(scales) != 10:
        raise ValueError("reverse profile must contain 10 amplitude scales")
    finite_scales = [
        _finite_number(value, f"joint_amplitude_scales[{index}]")
        for index, value in enumerate(scales)
    ]
    biases = parameters.get("joint_bias_offsets", [0.0] * 10)
    if not isinstance(biases, list) or len(biases) != 10:
        raise ValueError("reverse profile must contain 10 bias offsets when supplied")
    finite_biases = [
        _finite_number(value, f"joint_bias_offsets[{index}]")
        for index, value in enumerate(biases)
    ]
    phase_rate = _finite_number(parameters.get("phase_rate"), "phase_rate")
    if phase_rate <= 0.0:
        raise ValueError("reverse profile phase_rate must be positive")

    candidate_schema = "schema_version" in payload
    composition = payload.get("composition", {})
    if not isinstance(composition, Mapping):
        raise ValueError("reverse profile composition must be an object")
    extra_margin = _finite_number(
        composition.get("left_knee_extra_upper_margin_rad", 0.0),
        "composition.left_knee_extra_upper_margin_rad",
    )
    if not 0.0 <= extra_margin <= MAX_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD:
        raise ValueError(
            "left-knee extra upper margin must be in "
            f"[0, {MAX_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD}]"
        )
    if candidate_schema:
        if payload.get("schema_version") != 1:
            raise ValueError("reverse candidate schema_version must be 1")
        if payload.get("artifact_kind") != (
            "openduckmini_reverse_feedforward_profile_candidate"
        ):
            raise ValueError("unexpected reverse candidate artifact_kind")
        if not isinstance(payload.get("release_id"), str) or not payload["release_id"]:
            raise ValueError("reverse candidate release_id is required")
        if not isinstance(payload.get("status"), str) or not payload["status"]:
            raise ValueError("reverse candidate status is required")
        if payload.get("hardware_deployment") != "PROHIBITED":
            raise ValueError("reverse candidate must prohibit hardware deployment")
        frozen_composition = {
            "backward_residual_scale": 0.0,
            "leg_target_margin_rad": LEG_TARGET_MARGIN_RAD,
            "target_slew_rate_rad_per_s": TARGET_SLEW_LIMIT_RAD_PER_S,
            "positive_noise_reset_qpos_inward_margin_rad": RESET_NOISE_MARGIN_RAD,
            "head_target_value": 0.0,
        }
        for key, expected in frozen_composition.items():
            actual = _finite_number(composition.get(key), f"composition.{key}")
            if actual != expected:
                raise ValueError(
                    f"reverse candidate composition.{key} must remain {expected}"
                )
        head_indices = composition.get("head_target_indices")
        if head_indices != list(HEAD_ACTION_INDICES):
            raise ValueError("reverse candidate head target indices must remain frozen")
        adoption = payload.get("adoption")
        if not isinstance(adoption, Mapping):
            raise ValueError("reverse candidate adoption metadata is required")
        if diagnostic_unadopted:
            status = str(adoption.get("status", "")).upper()
            if "NOT_ADOPTED" not in status or adoption.get("simulation_only") is not True:
                raise ValueError(
                    "diagnostic reverse candidate must be explicitly not adopted "
                    "and simulation-only"
                )
    elif diagnostic_unadopted:
        raise ValueError(
            "diagnostic unadopted reverse profile requires candidate schema_version 1"
        )

    return {
        "release_id": str(payload.get("release_id", resolved.stem)),
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "schema": (
            "openduckmini_reverse_feedforward_profile_candidate_v1"
            if candidate_schema
            else "legacy_reverse_parameters_v1"
        ),
        "schema_validated": True,
        "all_json_numbers_finite": True,
        "parameters": {
            "joint_amplitude_scale_count": len(finite_scales),
            "joint_bias_offset_count": len(finite_biases),
            "phase_rate": phase_rate,
        },
        "composition": {
            "left_knee_extra_upper_margin_rad": extra_margin,
            "left_knee_upper_target_formula": (
                "SAFE_upper - leg_target_margin_rad - "
                "left_knee_extra_upper_margin_rad"
            ),
        },
        "diagnostic_unadopted": bool(diagnostic_unadopted),
        "adopted": False if diagnostic_unadopted else None,
    }


def validate_adopted_reverse_profiles(
    straight: Path, left: Path, right: Path
) -> dict[str, Any]:
    """Verify the three exact profiles permitted on the formal evaluation path."""

    specifications = {
        "straight": (
            straight.resolve(),
            "optimized_reverse_exact_safe_v1",
            ADOPTED_REVERSE_PROFILE_SHA256,
        ),
        "left": (
            left.resolve(),
            "optimized_reverse_left_exact_safe_v1",
            ADOPTED_REVERSE_LEFT_PROFILE_SHA256,
        ),
        "right": (
            right.resolve(),
            "optimized_backward_right_turn_gait_legacy",
            ADOPTED_REVERSE_RIGHT_PROFILE_SHA256,
        ),
    }
    evidence: dict[str, Any] = {}
    for label, (path, release_id, expected_hash) in specifications.items():
        profile = validate_reverse_profile_schema(
            path, diagnostic_unadopted=False
        )
        if profile["sha256"] != expected_hash:
            raise ValueError(
                f"formal reverse {label} profile hash mismatch: "
                f"expected {expected_hash}, got {profile['sha256']}"
            )
        profile["release_id"] = release_id
        profile["formal_hash_allowlisted"] = True
        evidence[label] = profile
    return evidence


def validate_diagnostic_unadopted_reverse_profile(path: Path) -> dict[str, Any]:
    evidence = validate_reverse_profile_schema(path, diagnostic_unadopted=True)
    evidence.update(
        {
            "formal_hash_allowlisted": False,
            "adopted": False,
            "adoption_eligible": False,
            "execution_mode": "DIAGNOSTIC_UNADOPTED_REVERSE_PROFILE",
        }
    )
    return evidence


def validate_diagnostic_unadopted_reverse_turn_profile(
    path: Path,
    *,
    direction: str,
    straight_base_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one atomic turn candidate against the exact v3 base profile."""

    if direction not in ATOMIC_REVERSE_TURN_COMMANDS:
        raise ValueError("atomic reverse-turn direction must be left or right")
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing reverse-turn profile: {resolved}")
    payload = _load_strict_json_object(resolved, "reverse-turn profile")
    if payload.get("schema_version") != 1:
        raise ValueError("reverse-turn candidate schema_version must be 1")
    if payload.get("artifact_kind") != (
        "openduckmini_margin_aware_atomic_reverse_turn_profile_candidate"
    ):
        raise ValueError("unexpected reverse-turn candidate artifact_kind")
    if not isinstance(payload.get("release_id"), str) or not payload["release_id"]:
        raise ValueError("reverse-turn candidate release_id is required")
    if not isinstance(payload.get("status"), str) or not payload["status"]:
        raise ValueError("reverse-turn candidate status is required")
    if payload.get("hardware_deployment") != "PROHIBITED":
        raise ValueError("reverse-turn candidate must prohibit hardware deployment")
    if payload.get("simulation_only") is not True:
        raise ValueError("reverse-turn candidate must be simulation-only")
    if payload.get("direction") != direction:
        raise ValueError("reverse-turn candidate direction mismatch")
    atomic_command = payload.get("atomic_command")
    if not isinstance(atomic_command, list) or len(atomic_command) != 3:
        raise ValueError("reverse-turn candidate atomic_command must have 3 values")
    finite_command = tuple(
        _finite_number(value, f"atomic_command[{index}]")
        for index, value in enumerate(atomic_command)
    )
    if finite_command != ATOMIC_REVERSE_TURN_COMMANDS[direction]:
        raise ValueError("reverse-turn candidate atomic_command mismatch")

    parameters = payload.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("reverse-turn candidate is missing parameters")
    for key in ("joint_amplitude_scales", "joint_bias_offsets"):
        values = parameters.get(key)
        if not isinstance(values, list) or len(values) != 10:
            raise ValueError(f"reverse-turn candidate {key} must have 10 values")
        for index, value in enumerate(values):
            _finite_number(value, f"parameters.{key}[{index}]")
    phase_rate = _finite_number(parameters.get("phase_rate"), "parameters.phase_rate")
    if phase_rate <= 0.0:
        raise ValueError("reverse-turn candidate phase_rate must be positive")

    if straight_base_evidence.get("sha256") != DIAGNOSTIC_REVERSE_V3_PROFILE_SHA256:
        raise ValueError("reverse-turn candidate requires the exact diagnostic v3 base")
    composition = payload.get("composition")
    if not isinstance(composition, Mapping):
        raise ValueError("reverse-turn candidate composition is required")
    if composition.get("straight_reverse_base_sha256") != (
        DIAGNOSTIC_REVERSE_V3_PROFILE_SHA256
    ):
        raise ValueError("reverse-turn candidate base v3 hash mismatch")
    recorded_base_path = composition.get("straight_reverse_base_profile")
    if not isinstance(recorded_base_path, str) or (
        not _portable_workspace_path_matches(
            recorded_base_path,
            Path(str(straight_base_evidence.get("path", ""))),
        )
    ):
        raise ValueError("reverse-turn candidate base v3 path mismatch")
    frozen_composition = {
        "backward_residual_scale": 0.0,
        "leg_target_margin_rad": LEG_TARGET_MARGIN_RAD,
        "target_slew_rate_rad_per_s": TARGET_SLEW_LIMIT_RAD_PER_S,
        "left_knee_extra_upper_margin_rad": (
            DIAGNOSTIC_REVERSE_V3_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
        ),
        "positive_noise_reset_qpos_inward_margin_rad": RESET_NOISE_MARGIN_RAD,
        "head_target_value": 0.0,
    }
    for key, expected in frozen_composition.items():
        actual = _finite_number(composition.get(key), f"composition.{key}")
        if actual != expected:
            raise ValueError(
                f"reverse-turn candidate composition.{key} must remain {expected}"
            )
    if composition.get("head_target_indices") != list(HEAD_ACTION_INDICES):
        raise ValueError("reverse-turn candidate head target indices mismatch")
    if (
        straight_base_evidence.get("composition", {}).get(
            "left_knee_extra_upper_margin_rad"
        )
        != DIAGNOSTIC_REVERSE_V3_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
    ):
        raise ValueError("reverse-turn candidate cap contradicts the v3 base")

    adoption = payload.get("adoption")
    if not isinstance(adoption, Mapping):
        raise ValueError("reverse-turn candidate adoption metadata is required")
    if "NOT_ADOPTED" not in str(adoption.get("status", "")).upper():
        raise ValueError("reverse-turn candidate must remain explicitly not adopted")
    if adoption.get("hardware_deployment") != "PROHIBITED":
        raise ValueError("reverse-turn candidate adoption must prohibit hardware")
    pilot = payload.get("pilot_evidence")
    if not isinstance(pilot, Mapping):
        raise ValueError("reverse-turn candidate pilot evidence is required")
    if pilot.get("all_passed") is not True or (
        pilot.get("all_physics_substeps_audited") is not True
    ):
        raise ValueError("reverse-turn candidate pilot evidence is not passing")

    return {
        "release_id": payload["release_id"],
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "schema": (
            "openduckmini_margin_aware_atomic_reverse_turn_profile_candidate_v1"
        ),
        "schema_validated": True,
        "all_json_numbers_finite": True,
        "direction": direction,
        "atomic_command": list(finite_command),
        "parameters": {
            "joint_amplitude_scale_count": 10,
            "joint_bias_offset_count": 10,
            "phase_rate": phase_rate,
        },
        "composition": {
            "straight_reverse_base_sha256": DIAGNOSTIC_REVERSE_V3_PROFILE_SHA256,
            "left_knee_extra_upper_margin_rad": (
                DIAGNOSTIC_REVERSE_V3_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
            ),
        },
        "formal_hash_allowlisted": False,
        "diagnostic_unadopted": True,
        "adopted": False,
        "adoption_eligible": False,
        "execution_mode": "DIAGNOSTIC_UNADOPTED_REVERSE_TURN_PROFILE",
    }


def validate_formal_candidate_reverse_profiles(
    straight: Path,
    left: Path,
    right: Path,
) -> dict[str, Any]:
    """Validate the unchanged profile bank for adopted simulation-only use."""

    requested = {
        "straight": straight.resolve(),
        "left": left.resolve(),
        "right": right.resolve(),
    }
    if requested != dict(FORMAL_CANDIDATE_PROFILE_PATHS):
        raise ValueError("formal-candidate profile paths must remain exactly pinned")
    straight_evidence = validate_diagnostic_unadopted_reverse_profile(
        requested["straight"]
    )
    # The unchanged atomic turn artifacts retain an immutable provenance link
    # to v3.  Validate that historical base independently; it is not the H2
    # straight profile that executes for the straight reverse route.
    turn_base_evidence = validate_diagnostic_unadopted_reverse_profile(
        DIAGNOSTIC_REVERSE_V3_PROFILE_PATH
    )
    evidence = {
        "straight": straight_evidence,
        "left": validate_diagnostic_unadopted_reverse_turn_profile(
            requested["left"],
            direction="left",
            straight_base_evidence=turn_base_evidence,
        ),
        "right": validate_diagnostic_unadopted_reverse_turn_profile(
            requested["right"],
            direction="right",
            straight_base_evidence=turn_base_evidence,
        ),
    }
    for label, record in evidence.items():
        digest = str(record.get("sha256", ""))
        if digest not in FORMAL_CANDIDATE_REVERSE_PROFILE_SHA256_ALLOWLISTS[label]:
            raise ValueError(f"formal-candidate {label} profile hash mismatch")
        record.update(
            {
                "status": FORMAL_CANDIDATE_STATUS,
                "formal_candidate": False,
                "candidate_hash_allowlisted": True,
                "formal_hash_allowlisted": True,
                "adoption_hash_allowlisted": True,
                "diagnostic_unadopted": False,
                "adopted": True,
                "adoption_eligible": True,
                "simulation_acceptance_eligible": True,
                "adoption_evidence_sha256": FORMAL_ADOPTION_EVIDENCE_SHA256,
                "execution_mode": "FORMAL_H3_ADOPTED_SIMULATION_ONLY",
                "hardware_deployment": "PROHIBITED",
            }
        )
    return evidence


def validate_formal_candidate_execution_bundle(
    candidate_evidence: Mapping[str, Any],
    adoption_evidence: Mapping[str, Any],
    executed_reverse_profiles: Mapping[str, Mapping[str, Any]],
    policy_provenance: Mapping[str, Any],
    phase_entry_indices: Mapping[str, float],
    *,
    backward_exit_recovery_enabled: bool,
    safety_component_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Cross-bind H3 selection, safety, and formal adoption evidence."""

    if safety_component_evidence is None:
        safety_component_evidence = validate_h3_fast_exit_safety_evidence()
    policy_roles = policy_provenance.get("roles")
    expected_profiles = candidate_evidence.get("profile_sha256s")
    profile_cap = candidate_evidence.get("profile_left_knee_cap")
    recovery_cap = candidate_evidence.get("backward_exit_recovery")
    if (
        candidate_evidence.get("sha256")
        != FORMAL_CANDIDATE_SELECTION_EVIDENCE_SHA256
        or adoption_evidence.get("sha256") != FORMAL_ADOPTION_EVIDENCE_SHA256
        or adoption_evidence.get("hash_allowlisted_for_adoption") is not True
        or adoption_evidence.get("status") != FORMAL_CANDIDATE_STATUS
        or adoption_evidence.get("selection_evidence_sha256")
        != FORMAL_CANDIDATE_SELECTION_EVIDENCE_SHA256
        or adoption_evidence.get("safety_component_evidence_sha256")
        != H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256
        or adoption_evidence.get("adoption_eligible") is not True
        or adoption_evidence.get("simulation_acceptance_eligible") is not True
        or candidate_evidence.get("candidate_execution_eligible") is not True
        or candidate_evidence.get("status") != H3_CANDIDATE_SELECTION_STATUS
        or candidate_evidence.get("combined_5x15_passed") is not True
        or candidate_evidence.get("combined_5x15_required") is not False
        or candidate_evidence.get("formal_20x30_required") is not True
        or candidate_evidence.get("safety_component_evidence_sha256")
        != H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256
        or candidate_evidence.get("adoption_eligible") is not False
        or candidate_evidence.get("simulation_acceptance_eligible") is not False
        or safety_component_evidence.get("sha256")
        != H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256
        or safety_component_evidence.get("safety_subset_passed") is not True
        or safety_component_evidence.get("safety_only_component") is not True
        or safety_component_evidence.get("source_artifact_passed") is not False
        or safety_component_evidence.get("central_suite_acceptance_passed")
        is not False
        or safety_component_evidence.get("motion_failure_count") != 11
        or safety_component_evidence.get("adoption_evidence") is not False
        or safety_component_evidence.get("adoption_eligible") is not False
        or not isinstance(expected_profiles, Mapping)
        or dict(expected_profiles) != dict(FORMAL_CANDIDATE_PROFILE_SHA256S)
        or set(executed_reverse_profiles) != {"straight", "left", "right"}
        or {
            label: str(record.get("sha256", ""))
            for label, record in executed_reverse_profiles.items()
        }
        != dict(FORMAL_CANDIDATE_PROFILE_SHA256S)
        or not all(
            record.get("formal_candidate") is False
            and record.get("candidate_hash_allowlisted") is True
            and record.get("adoption_hash_allowlisted") is True
            and record.get("adopted") is True
            and record.get("adoption_eligible") is True
            and record.get("adoption_evidence_sha256")
            == FORMAL_ADOPTION_EVIDENCE_SHA256
            for record in executed_reverse_profiles.values()
        )
        or not isinstance(policy_roles, Mapping)
        or set(policy_roles) != set(REQUIRED_POLICY_ROLES)
        or any(
            not isinstance(record, Mapping)
            or record.get("sha256") != BASE_V22_POLICY_SHA256
            or record.get("formal_base_v22_allowlisted") is not True
            for record in policy_roles.values()
        )
        or dict(phase_entry_indices)
        != dict(FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES)
        or backward_exit_recovery_enabled is not True
        or not isinstance(profile_cap, Mapping)
        or profile_cap.get("extra_upper_margin_rad")
        != FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
        or profile_cap.get("upper_target_rad")
        != FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_UPPER_TARGET_RAD
        or not isinstance(recovery_cap, Mapping)
        or recovery_cap.get("extra_upper_margin_rad")
        != BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD
        or recovery_cap.get("upper_target_rad")
        != BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD
        or recovery_cap.get("hold_control_ticks")
        != BACKWARD_EXIT_RECOVERY_HOLD_TICKS
    ):
        raise ValueError("formal-candidate execution bundle does not match evidence")
    return {
        "passed": True,
        "status": FORMAL_CANDIDATE_STATUS,
        "candidate_selection_evidence_sha256": (
            FORMAL_CANDIDATE_SELECTION_EVIDENCE_SHA256
        ),
        "adoption_evidence_sha256": FORMAL_ADOPTION_EVIDENCE_SHA256,
        "candidate_selection_is_superseded_h2_profile_lineage": False,
        "superseded_h2_selection_evidence_sha256": (
            H2_5X15_SELECTION_EVIDENCE_SHA256
        ),
        "safety_component_evidence_sha256": (
            H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256
        ),
        "safety_component_only": False,
        "safety_component_evidence_is_safety_only": True,
        "safety_subset_passed": True,
        "central_suite_acceptance_passed": True,
        "motion_failure_count": 0,
        "safety_component_motion_failure_count": 11,
        "superseded_h2_adoption_evidence_sha256": (
            H2_SUPERSEDED_ADOPTION_EVIDENCE_SHA256
        ),
        "adoption_evidence_sha256": FORMAL_ADOPTION_EVIDENCE_SHA256,
        "profile_sha256s": dict(FORMAL_CANDIDATE_PROFILE_SHA256S),
        "policy_sha256": BASE_V22_POLICY_SHA256,
        "policy_roles": list(REQUIRED_POLICY_ROLES),
        "reverse_endpoint_mps": CURRENT_FORMAL_REVERSE_ENDPOINT_MPS,
        "phase_preincrement_indices": dict(
            FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES
        ),
        "profile_left_knee_cap": {
            "extra_upper_margin_rad": (
                FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD
            ),
            "upper_target_rad": FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_UPPER_TARGET_RAD,
        },
        "backward_exit_recovery": {
            "enabled": True,
            "extra_upper_margin_rad": BACKWARD_EXIT_RECOVERY_EXTRA_UPPER_MARGIN_RAD,
            "hold_control_ticks": BACKWARD_EXIT_RECOVERY_HOLD_TICKS,
            "hold_seconds": BACKWARD_EXIT_RECOVERY_HOLD_SECONDS,
            "upper_target_rad": BACKWARD_EXIT_RECOVERY_LEFT_KNEE_UPPER_TARGET_RAD,
        },
        "formal_candidate": False,
        "combined_5x15_required": False,
        "combined_5x15_passed": True,
        "formal_20x30_required": False,
        "adopted": True,
        "adoption_eligible": True,
        "simulation_acceptance_eligible": True,
        "hardware_deployment": "PROHIBITED",
    }


def derive_reverse_profile_adoption(
    profile_evidence: Mapping[str, Mapping[str, Any]],
    evaluation_evidence: Mapping[str, Mapping[str, Any]],
    statuses: Mapping[str, str] = FORMAL_REVERSE_ADOPTION_STATUSES,
    *,
    profile_hash_allowlists: Mapping[str, frozenset[str]] = (
        FORMAL_REVERSE_PROFILE_SHA256_ALLOWLISTS
    ),
    evidence_hash_allowlists: Mapping[str, frozenset[str]] = (
        FORMAL_REVERSE_EVIDENCE_SHA256_ALLOWLISTS
    ),
) -> dict[str, Any]:
    """Derive adoption only from two independent hashes and nonblocked status."""

    labels = ("straight", "left", "right")
    roles: dict[str, Any] = {}
    for label in labels:
        profile_hash = str(profile_evidence.get(label, {}).get("sha256", ""))
        evidence_hash = str(evaluation_evidence.get(label, {}).get("sha256", ""))
        status = str(statuses.get(label, "MISSING_STATUS"))
        status_upper = status.upper()
        status_nonblocked = bool(status) and not any(
            marker in status_upper
            for marker in REVERSE_ADOPTION_BLOCKING_STATUS_MARKERS
        )
        profile_allowlisted = profile_hash in profile_hash_allowlists.get(
            label, frozenset()
        )
        evidence_allowlisted = bool(evidence_hash) and evidence_hash in (
            evidence_hash_allowlists.get(label, frozenset())
        )
        passed = bool(
            profile_allowlisted and evidence_allowlisted and status_nonblocked
        )
        roles[label] = {
            "profile_sha256": profile_hash,
            "profile_hash_allowlisted": profile_allowlisted,
            "evidence_sha256": evidence_hash or None,
            "evidence_hash_allowlisted": evidence_allowlisted,
            "status": status,
            "status_nonblocked": status_nonblocked,
            "passed": passed,
        }
    passed = all(record["passed"] for record in roles.values())
    return {
        "passed": passed,
        "status": "ADOPTED_SIMULATION_ONLY" if passed else "FAIL_CLOSED",
        "derivation": (
            "allowlisted_profile_hash AND allowlisted_evidence_hash "
            "AND nonblocked_status for straight/left/right"
        ),
        "profile_hash_allowlists": {
            label: sorted(profile_hash_allowlists.get(label, frozenset()))
            for label in labels
        },
        "evidence_hash_allowlists": {
            label: sorted(evidence_hash_allowlists.get(label, frozenset()))
            for label in labels
        },
        "blocking_status_markers": list(
            REVERSE_ADOPTION_BLOCKING_STATUS_MARKERS
        ),
        "roles": roles,
    }


__all__ = [
    "AcceptanceThresholds",
    "ADOPTED_REVERSE_LEFT_PROFILE_SHA256",
    "ADOPTED_REVERSE_PROFILE_SHA256",
    "ADOPTED_REVERSE_RIGHT_PROFILE_SHA256",
    "BASE_V22_POLICY_SHA256",
    "BACKWARD_FAMILY_EXPERTS",
    "ATOMIC_REVERSE_TURN_COMMANDS",
    "COMPOUND_CASES",
    "CommandCase",
    "EVALUATOR_ID",
    "DIAGNOSTIC_REVERSE_V3_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD",
    "DIAGNOSTIC_REVERSE_V3_PROFILE_SHA256",
    "DIAGNOSTIC_REVERSE_TURN_PROFILE_SHA256",
    "CURRENT_FORMAL_REVERSE_ENDPOINT_MPS",
    "DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_EVIDENCE_PATH",
    "DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_EVIDENCE_SHA256",
    "DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_FIXED_SEEDS",
    "DIAGNOSTIC_BACKWARD_EXIT_RECOVERY_SOURCE_REVERSE_ENDPOINT_MPS",
    "DIAGNOSTIC_REVERSE_PHASE_ENTRY_EVIDENCE_PATH",
    "DIAGNOSTIC_REVERSE_PHASE_ENTRY_EVIDENCE_SHA256",
    "DIAGNOSTIC_REVERSE_PHASE_ENTRY_SOURCE_REVERSE_ENDPOINT_MPS",
    "FORMAL_POLICY_SHA256_ALLOWLIST",
    "FORMAL_CANDIDATE_MASTER_SEED",
    "FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD",
    "FORMAL_CANDIDATE_PROFILE_LEFT_KNEE_UPPER_TARGET_RAD",
    "FORMAL_CANDIDATE_PROFILE_PATHS",
    "FORMAL_CANDIDATE_PROFILE_SHA256S",
    "FORMAL_CANDIDATE_STRAIGHT_PROFILE_SHA256",
    "FORMAL_CANDIDATE_REVERSE_ENTRY_PHASE_INDICES",
    "FORMAL_CANDIDATE_REVERSE_PROFILE_SHA256_ALLOWLISTS",
    "FORMAL_CANDIDATE_SELECTION_EVIDENCE_PATH",
    "FORMAL_CANDIDATE_SELECTION_EVIDENCE_SHA256",
    "FORMAL_CANDIDATE_SELECTION_EVIDENCE_SHA256_ALLOWLIST",
    "FORMAL_CANDIDATE_STATUS",
    "FORMAL_ADOPTION_EVIDENCE_PATH",
    "FORMAL_ADOPTION_EVIDENCE_SHA256",
    "FORMAL_ADOPTION_EVIDENCE_SHA256_ALLOWLIST",
    "FORMAL_H2_ADOPTED_REVERSE_PROFILE_SHA256_ALLOWLISTS",
    "FORMAL_H3_CANDIDATE_REVERSE_PROFILE_SHA256_ALLOWLISTS",
    "HISTORICAL_FAILED_FORMAL_CANDIDATE_EVIDENCE_PATH",
    "HISTORICAL_FAILED_FORMAL_CANDIDATE_EVIDENCE_SHA256",
    "H2_COMPONENT_SELECTION_EVIDENCE_PATH",
    "H2_COMPONENT_SELECTION_EVIDENCE_SHA256",
    "H2_COMPONENT_STATUS",
    "H2_5X15_SELECTION_EVIDENCE_PATH",
    "H2_5X15_SELECTION_EVIDENCE_SHA256",
    "H2_5X15_SELECTION_EVIDENCE_SHA256_ALLOWLIST",
    "H2_5X15_SELECTION_STATUS",
    "H2_SUPERSEDED_ADOPTION_STATUS",
    "H2_SUPERSEDED_ADOPTION_EVIDENCE_PATH",
    "H2_SUPERSEDED_ADOPTION_EVIDENCE_SHA256",
    "H3_FAST_EXIT_EXPECTED_MOTION_FAILURES",
    "H3_FAST_EXIT_SAFETY_EVIDENCE_PATH",
    "H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256",
    "H3_FAST_EXIT_SAFETY_EVIDENCE_SHA256_ALLOWLIST",
    "H3_FAST_EXIT_SAFETY_STATUS",
    "H3_CANDIDATE_SELECTION_STATUS",
    "FORMAL_REVERSE_ADOPTION_STATUSES",
    "FORMAL_REVERSE_ADOPTION_EVIDENCE_SHA256_ALLOWLISTS",
    "FORMAL_REVERSE_COMMAND_CASE_SAFETY_EVIDENCE_SHA256_ALLOWLISTS",
    "FORMAL_REVERSE_EVIDENCE_SHA256_ALLOWLISTS",
    "FORMAL_REVERSE_PROFILE_SHA256_ALLOWLISTS",
    "FROZEN_GENERATED_DEPENDENCY_ROOT_SHA256",
    "FROZEN_GENERATED_DEPENDENCY_SHA256",
    "FROZEN_GENERATED_PRIMARY_SHA256",
    "FROZEN_GENERATED_ROOT",
    "FROZEN_DIAGNOSTIC_REVERSE_ENTRY_PHASE_INDICES",
    "FROZEN_RUNTIME_BINARY_SHA256",
    "FROZEN_RUNTIME_DEPENDENCY_PATHS",
    "FROZEN_RUNTIME_DEPENDENCY_ROOT_SHA256",
    "FROZEN_RUNTIME_DEPENDENCY_SHA256",
    "FROZEN_RUNTIME_VERSIONS",
    "HEAD_ACTION_INDICES",
    "MAX_LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD",
    "POLICY_ROLE_ALIASES",
    "POLICY_COMMAND_DIAGNOSTIC_CASES",
    "PRIMITIVE_CASES",
    "PROHIBITED_POLICY_LABELS",
    "REQUIRED_POLICY_ROLES",
    "REJECTED_POLICY_COMMAND_DIAGNOSTIC_CASES",
    "REVERSE_V1_ADOPTION_STATUS",
    "REVERSE_V1_MEASURED_FORWARD_VELOCITY_MPS",
    "SCHEMA_VERSION",
    "PhysicsSubstepAudit",
    "SafetyAudit",
    "TRANSITION_CASES",
    "advance_routed_phase",
    "audit_control_first_startup",
    "audit_reset_qpos",
    "backward_exit_recovery_state_acceptance",
    "blend_and_mask_actions",
    "build_target_envelope",
    "canonical_policy_role",
    "command_case_validation_gate",
    "compute_motion_metrics",
    "capture_runtime_source_dependency_closure",
    "dependency_closure_root_sha256",
    "derive_reverse_profile_adoption",
    "discover_mjcf_dependency_closure",
    "generated_asset_paths",
    "hardware_gate",
    "parse_policy_assignments",
    "policy_yaw_observation_offset",
    "resolve_policy_observation_command",
    "runtime_source_dependency_root_sha256",
    "segment_acceptance",
    "sha256_file",
    "suite_acceptance",
    "summarize_backward_exit_recovery_steps",
    "transition_schedule",
    "validate_adopted_reverse_profiles",
    "validate_diagnostic_unadopted_reverse_profile",
    "validate_diagnostic_backward_exit_recovery_evidence",
    "validate_diagnostic_backward_exit_recovery_execution_bundle",
    "validate_diagnostic_unadopted_reverse_turn_profile",
    "validate_diagnostic_reverse_entry_phase_indices",
    "validate_diagnostic_reverse_phase_entry_evidence",
    "validate_exact_generated_assets",
    "validate_formal_candidate_selection_evidence",
    "validate_formal_adoption_evidence",
    "validate_superseded_h2_adoption_evidence",
    "validate_h3_fast_exit_safety_evidence",
    "validate_formal_candidate_execution_bundle",
    "validate_formal_candidate_reverse_profiles",
    "validate_frozen_runtime_source_dependencies",
    "validate_policy_provenance",
    "validate_reverse_profile_schema",
    "validate_runtime_versions",
]
