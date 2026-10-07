"""One-time migration: add `expected_flags: []` to every case missing it.

Never prints case content -- production.json isn't to be read/analyzed before
Phase 8 (see CLAUDE.md), so this only reports a count.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config

FILES = [config.ROOT / "evals" / "golden" / "golden.json", config.ROOT / "evals" / "golden" / "production.json"]


def add_expected_flags(path: Path) -> int:
    """Add `expected_flags: []` to each case in `path` that lacks it. Returns the count changed."""
    cases = json.loads(path.read_text())
    changed = 0
    for case in cases:
        if "expected_flags" not in case:
            case["expected_flags"] = []
            changed += 1
    path.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
    return changed


if __name__ == "__main__":
    for file_path in FILES:
        n = add_expected_flags(file_path)
        print(f"{file_path.name}: {n} cases updated")
