"""Download the frozen CheXlocalize v1.0 validation-only Redivis assets."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import sys
from pathlib import Path

import redivis

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bives_cxr.chexlocalize_acquisition import select_validation_files


DATASET_REFERENCE = "aimi.chexlocalize:efx9:v1_0"
DEFAULT_DESTINATION = Path(
    r"H:\Xiyao_Wang\000_Public Dataset\CheXlocalize\redivis_v1_0\validation"
)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-parallelization", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset = redivis.dataset("AIMI.CheXlocalize", version="1.0").get()
    if dataset.qualified_reference.lower() != DATASET_REFERENCE:
        raise RuntimeError(
            f"unexpected Redivis dataset identity: {dataset.qualified_reference}"
        )
    tables = dataset.list_tables()
    if len(tables) != 1 or tables[0].name != "CheXlocalize":
        raise RuntimeError("unexpected CheXlocalize Redivis table layout")
    table = tables[0]
    files = table.list_files()
    rows = [
        {
            "path": str(file.path).replace("\\", "/"),
            "size": int(file.size),
            "md5": bytes(file.hash).hex(),
            "uri": file.uri,
        }
        for file in files
    ]
    selected = select_validation_files(rows)
    selected_paths = {str(row["path"]) for row in selected}
    selected_files_all = [
        file
        for file in files
        if str(file.path).replace("\\", "/").lstrip("/") in selected_paths
    ]
    selected_files_all.sort(
        key=lambda file: str(file.path).replace("\\", "/").lstrip("/")
    )
    if len(selected_files_all) != len(selected):
        raise RuntimeError("Redivis file objects do not match the frozen inventory")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid CheXlocalize download shard")
    selected_files = [
        file
        for index, file in enumerate(selected_files_all)
        if index % args.shard_count == args.shard_index
    ]
    shard_paths = {
        str(file.path).replace("\\", "/").lstrip("/") for file in selected_files
    }
    destination = args.destination.resolve()
    total_bytes = sum(int(row["size"]) for row in selected)
    free_bytes = shutil.disk_usage(destination.anchor).free
    if free_bytes < total_bytes + 2 * 1024**3:
        raise RuntimeError(
            f"insufficient free space: {free_bytes} bytes for {total_bytes}-byte release"
        )
    destination.mkdir(parents=True, exist_ok=True)

    inventory = {
        "schema_version": "chexlocalize-validation-inventory-v1",
        "dataset_reference": dataset.qualified_reference,
        "table_uri": table.uri,
        "file_count": len(selected),
        "total_bytes": total_bytes,
        "test_opened": False,
        "files": selected,
    }
    inventory["canonical_sha256"] = _canonical_sha256(inventory)
    (destination / "validation_inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    def download_one(file: object) -> int:
        relative = Path(str(file.path).replace("\\", "/"))
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        file.download(str(target), overwrite=args.overwrite, progress=False)
        return int(file.size)

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_parallelization
    ) as executor:
        futures = [executor.submit(download_one, file) for file in selected_files]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            if completed % 100 == 0 or completed == len(selected_files):
                print(f"DOWNLOADED {completed}/{len(selected_files)}")

    missing = [
        str(row["path"])
        for row in selected
        if str(row["path"]) in shard_paths
        and not (destination / Path(str(row["path"]))).is_file()
    ]
    if missing:
        raise RuntimeError(f"download incomplete: {len(missing)} files missing")
    print(
        json.dumps(
            {
                "status": "validation_download_shard_complete",
                "dataset_reference": dataset.qualified_reference,
                "destination": destination.as_posix(),
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "file_count": len(selected_files),
                "total_bytes": sum(int(file.size) for file in selected_files),
                "canonical_sha256": inventory["canonical_sha256"],
                "test_opened": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
