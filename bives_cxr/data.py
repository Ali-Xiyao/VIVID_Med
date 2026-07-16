"""Manifest schema and fail-fast dataset utilities for BiVES-CXR."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler

from .decoder import STATE_NAMES


REQUIRED_FIELDS = {
    "sample_id",
    "patient_id",
    "image_path",
    "group_id",
    "canonical_statement_id",
    "statement_text",
    "state",
}


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED_FIELDS - set(row)
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            state = str(row["state"]).lower()
            if state not in STATE_NAMES:
                raise ValueError(f"{path}:{line_number} invalid state: {state}")
            row["state"] = state
            rows.append(row)
    if not rows:
        raise ValueError(f"empty BiVES manifest: {path}")
    return rows


class BiVESManifestDataset(Dataset):
    """Load images without converting IO failures into false insufficient samples."""

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path = ".",
        statement_to_index: dict[str, int] | None = None,
    ) -> None:
        self.rows = read_manifest(manifest_path)
        self.data_root = Path(data_root)
        if statement_to_index is None:
            statement_ids = sorted({str(row["canonical_statement_id"]) for row in self.rows})
            statement_to_index = {statement_id: index for index, statement_id in enumerate(statement_ids)}
        unknown = {
            str(row["canonical_statement_id"])
            for row in self.rows
            if str(row["canonical_statement_id"]) not in statement_to_index
        }
        if unknown:
            raise ValueError(f"manifest contains statement IDs absent from the training vocabulary: {sorted(unknown)[:5]}")
        self.statement_to_index = dict(statement_to_index)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = self.data_root / image_path
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        return {
            **row,
            "image": image,
            "image_path": str(image_path),
            "statement_index": self.statement_to_index[str(row["canonical_statement_id"])],
            "state_index": STATE_NAMES.index(str(row["state"])),
        }


class SameStatementStateBatchSampler(Sampler[list[int]]):
    """Yield batches composed of exact matched ``group_id`` S/C/U/I quartets."""

    def __init__(
        self,
        dataset: BiVESManifestDataset,
        groups_per_batch: int = 1,
        states: tuple[str, ...] = tuple(STATE_NAMES),
        shuffle: bool = True,
        seed: int = 17,
        drop_last: bool = False,
        require_complete_groups: bool = True,
    ) -> None:
        if groups_per_batch <= 0:
            raise ValueError("groups_per_batch must be positive")
        unknown_states = set(states) - set(STATE_NAMES)
        if unknown_states:
            raise ValueError(f"unknown BiVES states: {sorted(unknown_states)}")
        self.dataset = dataset
        self.groups_per_batch = int(groups_per_batch)
        self.states = tuple(states)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        group_statements: dict[str, set[str]] = defaultdict(set)
        group_texts: dict[str, set[str]] = defaultdict(set)
        for index, row in enumerate(dataset.rows):
            group_id = str(row["group_id"])
            grouped[group_id][str(row["state"])].append(index)
            group_statements[group_id].add(str(row["canonical_statement_id"]))
            group_texts[group_id].add(normalize_statement_text(row["statement_text"]))
        invalid: dict[str, dict[str, object]] = {}
        for group_id, state_rows in grouped.items():
            state_counts = {state: len(state_rows.get(state, [])) for state in self.states}
            if (
                any(count != 1 for count in state_counts.values())
                or set(state_rows) != set(self.states)
                or len(group_statements[group_id]) != 1
                or len(group_texts[group_id]) != 1
            ):
                invalid[group_id] = {
                    "state_counts": state_counts,
                    "statement_ids": sorted(group_statements[group_id]),
                    "statement_texts": sorted(group_texts[group_id]),
                }
        if invalid and require_complete_groups:
            examples = list(invalid.items())[:5]
            raise ValueError(
                f"{len(invalid)} group_id quartets violate the exact S/C/U/I contract; "
                f"examples={examples}"
            )
        self.groups = {
            group_id: {state: state_rows[state][0] for state in self.states}
            for group_id, state_rows in grouped.items()
            if group_id not in invalid
        }
        if not self.groups:
            raise ValueError("no valid group_id S/C/U/I quartets are available")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        groups = len(self.groups)
        if self.drop_last:
            return groups // self.groups_per_batch
        return (groups + self.groups_per_batch - 1) // self.groups_per_batch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        group_ids = sorted(self.groups)
        if self.shuffle:
            rng.shuffle(group_ids)
        current: list[int] = []
        group_count = 0
        for group_id in group_ids:
            rows_by_state = self.groups[group_id]
            for state in self.states:
                current.append(rows_by_state[state])
            group_count += 1
            if group_count == self.groups_per_batch:
                yield current
                current = []
                group_count = 0
        if current and not self.drop_last:
            yield current


def build_group_loss_indices(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Build aligned S/C pairs and uncertain indices from a collated group batch."""

    grouped: dict[str, dict[str, int]] = defaultdict(dict)
    for index, item in enumerate(batch):
        group_id = str(item["group_id"])
        state = str(item["state"])
        if state in grouped[group_id]:
            raise ValueError(f"batch contains duplicate {state} rows for group_id {group_id}")
        grouped[group_id][state] = index

    support: list[int] = []
    contradict: list[int] = []
    uncertain: list[int] = []
    for group_id, state_indices in grouped.items():
        missing = set(STATE_NAMES) - set(state_indices)
        if missing:
            raise ValueError(
                f"group_id quartet {group_id!r} is incomplete; missing={sorted(missing)}"
            )
        support.append(state_indices["support"])
        contradict.append(state_indices["contradict"])
        uncertain.append(state_indices["uncertain"])
    if not support:
        raise ValueError("batch has no complete group_id quartets")
    return {
        "support_pair_indices": torch.tensor(support, dtype=torch.long),
        "contradict_pair_indices": torch.tensor(contradict, dtype=torch.long),
        "uncertain_indices": torch.tensor(uncertain, dtype=torch.long),
    }


def normalize_statement_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def statement_text_by_id(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Return a unique normalized ontology mapping from manifest rows."""

    texts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        texts[str(row["canonical_statement_id"])].add(
            normalize_statement_text(row["statement_text"])
        )
    inconsistent = {
        statement_id: sorted(values)
        for statement_id, values in texts.items()
        if len(values) != 1
    }
    if inconsistent:
        raise ValueError(
            "canonical statement IDs map to inconsistent normalized text: "
            f"{list(inconsistent.items())[:5]}"
        )
    return {statement_id: next(iter(values)) for statement_id, values in texts.items()}
