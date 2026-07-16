import json
import os
from collections import Counter

LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]


def analyze(path: str):
    if not os.path.exists(path):
        print("MISSING", path)
        return None

    state_counts = Counter()
    label_state_counts = {name: Counter() for name in LABELS}
    missing_counts = Counter()
    num_samples = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            num_samples += 1
            sample = json.loads(line)
            findings = sample.get("findings", {}) or {}
            for name in LABELS:
                if name not in findings:
                    missing_counts[name] += 1
            for name, info in findings.items():
                if name not in label_state_counts:
                    continue
                state = info.get("state")
                key = "null" if state is None else state
                state_counts[key] += 1
                label_state_counts[name][key] += 1

    total_states = sum(state_counts.values())
    overall_ratio = {k: round(v / total_states, 4) for k, v in state_counts.items()} if total_states else {}

    rows = []
    for name in LABELS:
        c = label_state_counts[name]
        present = c.get("present", 0)
        absent = c.get("absent", 0)
        uncertain = c.get("uncertain", 0)
        null = c.get("null", 0)
        missing = missing_counts[name]
        total = present + absent + uncertain + null + missing
        null_missing = null + missing
        ratio = null_missing / total if total else 0
        rows.append((ratio, name, present, absent, uncertain, null, missing, total))
    rows.sort(reverse=True)

    print("\nFILE", path)
    print("samples", num_samples)
    print("overall state counts", dict(state_counts))
    print("overall state ratios", overall_ratio)
    print("per-label null+missing ratios:")
    for ratio, name, present, absent, uncertain, null, missing, total in rows:
        print(f"{name:28s} null+missing={null+missing:6d}/{total:6d} ({ratio:6.2%})  present={present:6d} absent={absent:6d} uncertain={uncertain:6d} null={null:6d} missing={missing:6d}")


analyze(r"./data/dataset/processed/chexpert_ums_train.jsonl")
analyze(r"./data/dataset/processed/chexpert_ums_val.jsonl")
