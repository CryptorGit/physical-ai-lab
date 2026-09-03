#!/usr/bin/env python3
"""Training-equivalent diagnostic smoke entrypoint (never a formal evaluator).

The implementation is intentionally delegated to the immutable historical MJX
environment through export_v59_golden_trace.py, which also gates execution on
actor and motor-target parity.
"""
from pathlib import Path
import runpy

tool = Path(__file__).resolve().parents[1] / "tools" / "export_v59_golden_trace.py"
runpy.run_path(str(tool), run_name="__main__")
