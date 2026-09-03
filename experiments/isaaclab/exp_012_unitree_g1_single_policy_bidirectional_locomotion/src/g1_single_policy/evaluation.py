"""Shared evaluation condition definitions."""

STEADY = (0.0, 0.6, 0.8, 1.0, 1.2, 2.4, 2.6)
TRANSITIONS = ((0.0, 0.6), (0.6, 1.2), (1.2, 2.4), (1.2, 2.6),
               (2.4, 1.2), (2.6, 1.2), (1.2, 0.6), (0.6, 0.0))
