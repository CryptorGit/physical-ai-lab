#!/usr/bin/env python3
"""Summarize independent NumPy versus MJX motor-target replay parity."""
import csv
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
rows = list(csv.DictReader(path.open(encoding="utf-8")))
values = [float(r["max_abs_error"]) for r in rows]
print(json.dumps({
    "rows": len(rows),
    "max_abs_error": max(values),
    "threshold": 1e-6,
    "pass": max(values) <= 1e-6,
}, indent=2))
