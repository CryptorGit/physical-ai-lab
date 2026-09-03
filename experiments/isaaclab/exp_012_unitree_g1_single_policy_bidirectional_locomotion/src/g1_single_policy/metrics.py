"""Small metric helpers."""

from __future__ import annotations

def percentage(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0
