"""Small provenance and manifest helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def teacher_weight_authority(path: Path) -> dict[str, object]:
    index = path / "model.safetensors.index.json"
    single = path / "model.safetensors"
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        names = sorted(set(payload["weight_map"].values()))
        files = [index, *(path / name for name in names)]
    elif single.is_file():
        files = [single]
    else:
        raise FileNotFoundError(f"no safetensors authority under {path}")
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(file)
    return {
        "files": {
            file.name: {
                "bytes": file.stat().st_size,
                "sha256": sha256_file(file),
            }
            for file in files
        }
    }
