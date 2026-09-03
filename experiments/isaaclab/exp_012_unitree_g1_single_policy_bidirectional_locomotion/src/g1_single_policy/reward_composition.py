"""Reward provenance constants; implementation is inherited unchanged from exp_005."""

RUN_GATE_MPS = 2.3
ALLOWED_NEW_TERMS = ("safe_periodic_flight",)


def run_reward_isolated(requested_vx, reward_value) -> bool:
    return requested_vx >= RUN_GATE_MPS or reward_value == 0.0
