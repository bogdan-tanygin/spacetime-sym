# Copyright (C) 2024-2026 Dr. Bogdan Tanygin <info@deeptech.business>

"""Synchronize citation metadata with spacetime/version.py.

Usage:
    python scripts/sync_version.py          # update metadata
    python scripts/sync_version.py --check  # check without changing files
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def package_version() -> str:
    """Read the literal __version__ without importing the package."""
    source = (ROOT / "spacetime" / "version.py").read_text(encoding="utf-8")
    assignments = [
        node.value
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__version__"
                for target in node.targets)
    ]
    if len(assignments) != 1:
        raise ValueError("Expected exactly one __version__ assignment in spacetime/version.py")
    version = ast.literal_eval(assignments[0])
    if not isinstance(version, str) or not version:
        raise ValueError("__version__ must be a non-empty string literal")
    return version


def replace_once(text: str, pattern: str, replacement: str) -> str:
    updated, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected one version field, found {count}: {pattern}")
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify without writing files")
    args = parser.parse_args()

    version = package_version()
    citation = ROOT / "CITATION.cff"
    codemeta = ROOT / "codemeta.json"
    original_cff = citation.read_text(encoding="utf-8")
    original_json = codemeta.read_text(encoding="utf-8")

    updated_cff = replace_once(
        original_cff,
        r"^version:[ \t]*[^\r\n]+$",
        f"version: {version}",
    )
    updated_json = replace_once(
        original_json,
        r'(^[ \t]*"version"[ \t]*:[ \t]*")[^"\r\n]*(")',
        rf'\g<1>{version}\g<2>',
    )
    if json.loads(updated_json)["version"] != version:
        raise ValueError("Invalid CodeMeta version")

    needs_update = False
    for path, old, new in (
        (citation, original_cff, updated_cff),
        (codemeta, original_json, updated_json),
    ):
        changed = old != new
        print(f"{path.name}: {'out of sync' if changed else 'up to date'} ({version})")
        if changed:
            needs_update = True
            if not args.check:
                path.write_text(new, encoding="utf-8")

    return 1 if args.check and needs_update else 0


if __name__ == "__main__":
    raise SystemExit(main())
