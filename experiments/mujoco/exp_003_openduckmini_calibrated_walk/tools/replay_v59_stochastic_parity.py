#!/usr/bin/env python3
"""Validate saved v59 sample-injection parity rows without rerunning physics."""
import csv
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
with path.open(encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
result = {
    "cases": len(rows),
    "pass": all(row["pass"].lower() == "true" for row in rows),
    "first_divergences": [
        {
            "command_id": row["command_id"],
            "environment_index": row["environment_index"],
            "stage": row["first_divergence_stage"] or None,
            "step": row["first_divergence_step"] or None,
        }
        for row in rows
    ],
}
print(json.dumps(result, indent=2))
