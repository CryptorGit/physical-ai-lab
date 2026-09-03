"""Read-only filesystem helpers for the D11 durability forensic audit."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def utc_iso(timestamp): return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def within_window(stat, start_timestamp, end_timestamp):
    return start_timestamp <= stat.st_mtime <= end_timestamp or start_timestamp <= stat.st_ctime <= end_timestamp


def relevance(path):
    text = str(path).lower().replace("\\", "/")
    if "phase_2_d11_stop_student_heldout" in text: return "DIRECT_D11"
    if "isaac" in text and any(token in text for token in ("log", "stdout", "stderr", "crash")): return "POSSIBLE_RUNTIME_LOG"
    if any(token in text for token in ("python", "physx", "cuda", "kit")): return "POSSIBLE_PROCESS_ARTIFACT"
    return "TIME_WINDOW_ONLY"


def inventory_record(path, root):
    path = Path(path); before = path.stat(); digest = sha256_file(path); after = path.stat()
    return {"absolute_path": str(path.resolve()), "relative_path": str(path.relative_to(root)) if path.is_relative_to(root) else "", "size": before.st_size, "creation_time_utc": utc_iso(before.st_ctime), "modification_time_utc": utc_iso(before.st_mtime), "sha256": digest, "file_type": path.suffix.lower() or "NO_EXTENSION", "owner_process_candidate": "D11 evaluator" if "phase_2_d11" in str(path).lower() else "unknown", "D11_relevance": relevance(path), "read_only_audit_preserved_mtime": before.st_mtime_ns == after.st_mtime_ns}
