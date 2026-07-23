"""Compute a deterministic relative-path/size tree digest without reading content."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


def walk(root: Path) -> tuple[int, int, str]:
    entries: list[tuple[str, int]] = []
    stack = [root]
    while stack:
        directory = stack.pop()
        with os.scandir(directory) as iterator:
            for item in iterator:
                if item.is_symlink():
                    raise RuntimeError(f"refusing symlink in audited tree: {item.path}")
                if item.is_dir(follow_symlinks=False):
                    stack.append(Path(item.path))
                elif item.is_file(follow_symlinks=False):
                    path = Path(item.path)
                    entries.append((path.relative_to(root).as_posix(), item.stat().st_size))
    digest = hashlib.sha256()
    total = 0
    for relative, size in sorted(entries):
        digest.update(f"{relative}\t{size}\n".encode("utf-8"))
        total += size
    return len(entries), total, digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    for root in args.roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        count, total, digest = walk(root)
        print(f"{root}|{count}|{total}|{digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
