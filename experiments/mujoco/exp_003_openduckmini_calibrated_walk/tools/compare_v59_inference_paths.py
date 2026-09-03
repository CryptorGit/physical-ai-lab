#!/usr/bin/env python3
"""Summarize actor parity rows produced by export_v59_golden_trace.py."""
import csv
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
rows = list(csv.DictReader(path.open(encoding="utf-8")))
summary = {
    "rows": len(rows),
    "python_max_abs_error": max(float(r["python_max_abs_error"]) for r in rows),
    "onnx_max_abs_error": max(float(r["onnx_max_abs_error"]) for r in rows),
}
print(json.dumps(summary, indent=2))
