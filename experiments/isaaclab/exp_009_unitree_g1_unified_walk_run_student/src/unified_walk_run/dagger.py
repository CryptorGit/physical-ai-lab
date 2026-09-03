"""One-round DAgger routing contract; no reverse-failure labels."""


def teacher_route(regime: str, target_speed: float) -> str:
    if regime == "walk_to_run":
        return "walk_to_run"
    if regime == "steady" and target_speed <= 1.2:
        return "walk"
    if regime == "steady" and target_speed >= 2.4:
        return "run"
    raise ValueError("unsupported DAgger route; RUN_TO_WALK is intentionally excluded")
