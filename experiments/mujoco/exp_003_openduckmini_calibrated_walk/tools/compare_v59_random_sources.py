#!/usr/bin/env python3
"""Check that every stochastic trace metadata/event file is paired."""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
npz = {path.stem for path in root.glob("*.npz")}
metadata = {
    path.name.removesuffix(".metadata.json")
    for path in root.glob("*.metadata.json")
}
events = {
    path.name.removesuffix(".random_samples.jsonl")
    for path in root.glob("*.random_samples.jsonl")
}
print(json.dumps({
    "trace_count": len(npz),
    "metadata_count": len(metadata),
    "event_stream_count": len(events),
    "all_paired": npz == metadata == events,
}, indent=2))
