"""Verify a calibrated runtime ZIP directly, without extracting it."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive.resolve()) as archive:
        manifest_names = [
            name
            for name in archive.namelist()
            if name.endswith("/calibrated_release_manifest.json")
            or name == "calibrated_release_manifest.json"
        ]
        if len(manifest_names) != 1:
            raise ValueError(
                f"Expected one release manifest, found {manifest_names}"
            )
        manifest_name = manifest_names[0]
        root = PurePosixPath(manifest_name).parent
        manifest = json.loads(archive.read(manifest_name))
        results = []
        for relative, expected in manifest["files"].items():
            member = str(root / PurePosixPath(relative))
            try:
                payload = archive.read(member)
            except KeyError:
                actual = None
            else:
                actual = hashlib.sha256(payload).hexdigest().upper()
            results.append(
                {
                    "path": member,
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "passed": actual == expected,
                }
            )
    report = {
        "archive": str(args.archive.resolve()),
        "release": manifest["release"],
        "passed": all(result["passed"] for result in results),
        "files": results,
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
