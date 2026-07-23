from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcsd_cxr.config import DatasetRegistry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate RCSD-CXR data references")
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--roles", nargs="+", required=True)
    parser.add_argument(
        "--allow-paper2",
        action="store_true",
        help="allow roles reserved for the separate paper-two protocol",
    )
    args = parser.parse_args()
    registry = DatasetRegistry.load(args.registry)
    selected = registry.select_roles(args.roles)
    for record in selected:
        print(f"{record.name}: {record.path} [{record.test_exposure}]")
    errors = registry.validate_paths(args.roles, paper1=not args.allow_paper2)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"PASS: {len(selected)} dataset references validated from {registry.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
