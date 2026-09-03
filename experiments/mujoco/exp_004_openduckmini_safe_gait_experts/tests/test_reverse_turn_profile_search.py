from __future__ import annotations

import numpy as np

from scripts.search_margin_aware_reverse_turn_profiles import (
    BASE_PROFILE,
    LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD,
    MINIMUM_PROGRESS_FRACTION,
    PILOT_SECONDS,
    TURNS,
    ProfileParameters,
    effective_to_loaded_profile,
    profile_blend_for_direction,
    validate_base_profile,
)


def test_atomic_reverse_turn_gate_is_frozen() -> None:
    assert TURNS["left"].command == (-0.03, 0.0, 0.20)
    assert TURNS["right"].command == (-0.04, 0.0, -0.20)
    assert TURNS["left"].minimum_reverse_vx == -0.009
    assert TURNS["right"].minimum_reverse_vx == -0.012
    assert TURNS["left"].minimum_signed_yaw_rate == 0.06
    assert TURNS["right"].minimum_signed_yaw_rate == 0.06
    assert MINIMUM_PROGRESS_FRACTION == 0.30
    assert PILOT_SECONDS == 15.0
    assert LEFT_KNEE_EXTRA_UPPER_MARGIN_RAD == 0.0125


def test_right_loaded_profile_inverts_runtime_blend() -> None:
    base = ProfileParameters(
        np.linspace(0.5, 1.4, 10),
        np.linspace(-0.1, 0.1, 10),
        1.5,
    )
    effective = ProfileParameters(
        np.linspace(1.0, 1.9, 10),
        np.linspace(0.1, -0.1, 10),
        2.5,
    )
    loaded = effective_to_loaded_profile(base, effective, -1)
    reconstructed = profile_blend_for_direction(base, loaded, -1)
    np.testing.assert_allclose(
        reconstructed.amplitude_scales, effective.amplitude_scales
    )
    np.testing.assert_allclose(reconstructed.bias_offsets, effective.bias_offsets)
    assert reconstructed.phase_rate == effective.phase_rate


def test_search_base_is_exact_candidate_v3() -> None:
    parameters, payload = validate_base_profile(BASE_PROFILE)
    assert payload["release_id"] == "optimized_reverse_margin050_slew200_candidate_v3"
    assert parameters.amplitude_scales.shape == (10,)
    assert parameters.bias_offsets.shape == (10,)
    assert parameters.phase_rate > 0.0

