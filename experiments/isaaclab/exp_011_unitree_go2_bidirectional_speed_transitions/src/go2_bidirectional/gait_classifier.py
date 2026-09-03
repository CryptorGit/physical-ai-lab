"""Frozen diagnostic four-foot gait classifier."""

from __future__ import annotations

from .metrics import mean

LABELS = ("STAND", "WALK_LIKE", "TROT_LIKE", "PACE_LIKE", "BOUND_LIKE", "FLIGHT_RICH", "IRREGULAR", "FALL")
THRESHOLDS = {
    "contact_force_n": 5.0,
    "stand_speed_mps": 0.08,
    "stand_contact_occupancy_min": 0.90,
    "flight_rich_fraction_min": 0.20,
    "pair_synchrony_min": 0.70,
    "walk_single_or_triple_support_min": 0.25,
}


def synchrony(a: list[bool], b: list[bool]) -> float:
    return mean(x == y for x, y in zip(a, b))


def classify(contacts: list[list[bool]], actual_speed: float, fall: bool = False) -> tuple[str, dict]:
    if fall:
        return "FALL", {"reason": "fall"}
    if not contacts:
        return "IRREGULAR", {"reason": "empty_trace"}
    columns = list(map(list, zip(*contacts)))
    duty = [mean(column) for column in columns]
    flight = mean(not any(row) for row in contacts)
    diag = mean((synchrony(columns[0], columns[3]), synchrony(columns[1], columns[2])))
    ipsi = mean((synchrony(columns[0], columns[2]), synchrony(columns[1], columns[3])))
    fore_hind = mean((synchrony(columns[0], columns[1]), synchrony(columns[2], columns[3])))
    non_pair = mean(sum(row) in (1, 3) for row in contacts)
    evidence = {
        "duty_factor": duty,
        "flight_fraction": flight,
        "diagonal_pair_synchrony": diag,
        "ipsilateral_pair_synchrony": ipsi,
        "fore_hind_pair_synchrony": fore_hind,
        "single_or_triple_support_fraction": non_pair,
        "actual_speed_mps": actual_speed,
    }
    if actual_speed <= THRESHOLDS["stand_speed_mps"] and min(duty) >= THRESHOLDS["stand_contact_occupancy_min"]:
        label = "STAND"
    elif flight >= THRESHOLDS["flight_rich_fraction_min"]:
        label = "FLIGHT_RICH"
    elif diag >= THRESHOLDS["pair_synchrony_min"] and diag > max(ipsi, fore_hind) + 0.05:
        label = "TROT_LIKE"
    elif ipsi >= THRESHOLDS["pair_synchrony_min"] and ipsi > max(diag, fore_hind) + 0.05:
        label = "PACE_LIKE"
    elif fore_hind >= THRESHOLDS["pair_synchrony_min"] and fore_hind > max(diag, ipsi) + 0.05:
        label = "BOUND_LIKE"
    elif non_pair >= THRESHOLDS["walk_single_or_triple_support_min"]:
        label = "WALK_LIKE"
    else:
        label = "IRREGULAR"
    return label, evidence

