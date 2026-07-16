"""Validate and publish isolated next-stage postprocess outputs.

This helper keeps concurrent workers from writing the same canonical JSON file.
Workers should write to an isolated path first, then this script publishes the
validated result to the canonical output path or records why no publish happened.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp.replace(path)


def file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime": stat.st_mtime,
    }


def sorted_existing(paths: list[Path], prefer: str) -> list[Path]:
    existing = [path for path in paths if path.exists()]
    if prefer == "newest":
        return sorted(existing, key=lambda path: path.stat().st_mtime, reverse=True)
    return existing


def merge_visual_modes(payloads: list[tuple[Path, dict[str, Any]]]) -> tuple[dict[str, Any], dict[str, Any]]:
    merged = dict(payloads[0][1])
    mode_by_name: dict[str, dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    for path, payload in payloads:
        modes = payload.get("modes")
        if not isinstance(modes, list):
            raise ValueError(f"{path} does not contain a modes list")
        for mode in modes:
            if not isinstance(mode, dict) or not mode.get("mode"):
                raise ValueError(f"{path} contains an invalid mode row")
            name = str(mode["mode"])
            current = mode_by_name.get(name)
            if current is None:
                mode_by_name[name] = dict(mode)
                continue
            old_examples = int(current.get("examples") or -1)
            new_examples = int(mode.get("examples") or -1)
            duplicates.append({"mode": name, "kept_examples": max(old_examples, new_examples), "source": str(path)})
            if new_examples > old_examples:
                mode_by_name[name] = dict(mode)
    order = []
    for _, payload in payloads:
        for mode in payload.get("modes", []):
            name = str(mode["mode"])
            if name not in order:
                order.append(name)
    merged["modes"] = [mode_by_name[name] for name in order]
    details = {"merged_modes": order, "duplicates": duplicates}
    return merged, details


def build_payload(args: argparse.Namespace, inputs: list[Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in inputs:
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} is not a JSON object")
        loaded.append((path, payload))

    if args.mode == "visual-modes":
        merged, details = merge_visual_modes(loaded)
    elif args.mode == "promote":
        selected = sorted_existing(inputs, args.prefer)[0]
        merged = read_json(selected)
        if not isinstance(merged, dict):
            raise ValueError(f"{selected} is not a JSON object")
        details = {"selected": str(selected)}
    else:
        if len(loaded) == 1:
            selected = loaded[0][0]
            merged = dict(loaded[0][1])
            details = {"selected": str(selected)}
        elif all(isinstance(payload.get("modes"), list) for _, payload in loaded):
            merged, details = merge_visual_modes(loaded)
        else:
            selected = sorted_existing(inputs, args.prefer)[0]
            merged = read_json(selected)
            if not isinstance(merged, dict):
                raise ValueError(f"{selected} is not a JSON object")
            details = {
                "selected": str(selected),
                "note": "multiple non-visual JSON objects; selected one complete payload",
            }

    merged["_isolation_merge"] = {
        "created_at": now_iso(),
        "mode": args.mode,
        "inputs": [file_info(path) for path in inputs],
        **details,
    }
    return merged, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inputs", required=True, nargs="+", type=Path)
    parser.add_argument("--mode", choices=["auto", "promote", "visual-modes"], default="auto")
    parser.add_argument("--prefer", choices=["newest", "first"], default="newest")
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = sorted_existing(args.inputs, args.prefer)
    manifest_path = args.manifest or args.output.with_name(f"{args.output.stem}.merge_manifest.json")
    manifest: dict[str, Any] = {
        "created_at": now_iso(),
        "output": str(args.output),
        "inputs": [str(path) for path in args.inputs],
        "existing_inputs": [file_info(path) for path in inputs],
        "mode": args.mode,
    }
    if not inputs:
        manifest["status"] = "no_inputs"
        write_json_atomic(manifest_path, manifest)
        raise SystemExit("No existing isolated inputs to merge")
    if args.output.exists() and not args.overwrite:
        manifest["status"] = "output_exists"
        manifest["output_info"] = file_info(args.output)
        write_json_atomic(manifest_path, manifest)
        if args.allow_existing:
            print(json.dumps(manifest, indent=2, ensure_ascii=False))
            return
        raise SystemExit(f"Output already exists: {args.output}")

    payload, details = build_payload(args, inputs)
    write_json_atomic(args.output, payload)
    manifest["status"] = "published"
    manifest["details"] = details
    manifest["output_info"] = file_info(args.output)
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
