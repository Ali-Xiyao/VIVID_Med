"""Write the immutable source-tree manifest required by a source-only server deployment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bives_cxr.provenance import _git_tracked_source_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    snapshot = _git_tracked_source_snapshot(args.root, require_clean=True)
    if snapshot is None:
        raise SystemExit("source manifest must be generated from a Git checkout")
    snapshot["kind"] = "source_archive"
    output = args.output or (args.root / ".bives_source_manifest.json")
    output.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
