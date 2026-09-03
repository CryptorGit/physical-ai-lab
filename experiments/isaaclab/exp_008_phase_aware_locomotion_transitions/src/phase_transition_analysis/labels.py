"""Stage 0 contact-break labels."""

from __future__ import annotations

import numpy as np

BREAK_REASONS = ("contact", "speed", "heading", "flight", "safety", "none")


def add_labels(frame):
    """Add fixed primary labels without mutating policy inputs."""
    result = frame.copy()
    steps = result["steps_until_break"].to_numpy(dtype=np.int64)
    for horizon in (1, 3, 5):
        result[f"contact_break_within_{horizon}"] = (
            (result["break_reason"].to_numpy() == "contact") & (steps >= 0) & (steps <= horizon)
        ).astype(np.int8)
    result["will_reach_20_step_walk_contract"] = result["episode_success"].astype(np.int8)
    result["break_reason_class"] = result["break_reason"].map({name: index for index, name in enumerate(BREAK_REASONS)}).astype(np.int8)
    return result
