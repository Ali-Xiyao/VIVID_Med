"""Portable, fail-closed dataset registry handling."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Mapping

import yaml


FORBIDDEN_PAPER1_ROLES = {"paper2_only", "paper2_reserved_test"}


@dataclass(frozen=True)
class DatasetRecord:
    name: str
    path: Path | None
    kind: str
    parent: str | None
    roles: tuple[str, ...]
    status: str
    test_exposure: str

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, object]) -> "DatasetRecord":
        raw_path = value.get("path")
        roles = value.get("roles", [])
        if not isinstance(roles, list) or not all(isinstance(x, str) for x in roles):
            raise ValueError(f"dataset {name!r} roles must be a list of strings")
        required = ("kind", "status", "test_exposure")
        missing = [key for key in required if not isinstance(value.get(key), str)]
        if missing:
            raise ValueError(f"dataset {name!r} is missing string fields: {missing}")
        return cls(
            name=name,
            path=Path(raw_path) if isinstance(raw_path, str) and raw_path else None,
            kind=str(value["kind"]),
            parent=str(value["parent"]) if value.get("parent") is not None else None,
            roles=tuple(roles),
            status=str(value["status"]),
            test_exposure=str(value["test_exposure"]),
        )


class DatasetRegistry:
    def __init__(self, records: Mapping[str, DatasetRecord], source: Path):
        self.records = dict(records)
        self.source = source

    @classmethod
    def load(cls, path: str | Path | None = None) -> "DatasetRegistry":
        selected = path or os.environ.get("RCSD_DATA_REGISTRY")
        if not selected:
            raise RuntimeError(
                "set RCSD_DATA_REGISTRY or pass a registry path explicitly"
            )
        source = Path(selected)
        if not source.is_file():
            raise FileNotFoundError(f"dataset registry not found: {source}")
        with source.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("dataset registry requires schema_version: 1")
        datasets = payload.get("datasets")
        if not isinstance(datasets, dict):
            raise ValueError("dataset registry requires a datasets mapping")
        records = {
            name: DatasetRecord.from_mapping(name, value)
            for name, value in datasets.items()
            if isinstance(name, str) and isinstance(value, dict)
        }
        if len(records) != len(datasets):
            raise ValueError("every dataset entry must be a mapping with a string name")
        registry = cls(records, source.resolve())
        registry.validate_lineage()
        return registry

    def validate_lineage(self) -> None:
        for record in self.records.values():
            if record.parent is None:
                continue
            parents = [item.strip() for item in record.parent.split(",")]
            missing = [item for item in parents if item not in self.records]
            if missing:
                raise ValueError(f"dataset {record.name!r} has unknown parents: {missing}")

    def select_roles(self, roles: Iterable[str]) -> list[DatasetRecord]:
        wanted = set(roles)
        return [r for r in self.records.values() if wanted.intersection(r.roles)]

    def validate_paths(
        self,
        roles: Iterable[str],
        *,
        paper1: bool = True,
    ) -> list[str]:
        selected = self.select_roles(roles)
        if not selected:
            return ["no datasets matched the requested roles"]
        errors: list[str] = []
        for record in selected:
            if paper1 and FORBIDDEN_PAPER1_ROLES.intersection(record.roles):
                errors.append(f"{record.name}: forbidden paper-one role")
            if record.path is None:
                errors.append(f"{record.name}: no local path")
            elif not record.path.exists():
                errors.append(f"{record.name}: missing path {record.path}")
        return errors
