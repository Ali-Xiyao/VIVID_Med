"""Generate VIVID-Med clinical instruction JSONL records.

The proposal's report-grounded V2/V3 data requires raw report text. Current
CheXpert-small UMS splits in this repo do not include reports, so this script
supports an immediately runnable V1 label-to-QA path and an optional GLM mode
for OpenAI-compatible endpoints when report/text fields become available.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_LABELS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Fracture",
    "Support Devices",
]

STATE_ANSWERS = {
    "present": "{finding} is present according to the structured label.",
    "absent": "{finding} is absent according to the structured label.",
    "uncertain": "{finding} is uncertain according to the structured label.",
}

DEFAULT_API_BASE = "https://open.bigmodel.cn/api/coding/paas/v4"


def read_jsonl(path: Path, max_samples: int | None = None, skip_samples: int = 0) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < skip_samples:
                continue
            if max_samples is not None and len(rows) >= max_samples:
                break
            rows.append(json.loads(line))
    return rows


def read_existing_sample_ids(path: Path) -> set[str]:
    sample_ids: set[str] = set()
    if not path.exists():
        return sample_ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("sample_id") is not None:
                sample_ids.add(str(row.get("sample_id")))
    return sample_ids


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sample_id(sample: dict[str, Any], fallback_index: int) -> str:
    extensions = sample.get("extensions") or {}
    value = extensions.get("sample_id")
    if value is None:
        value = extensions.get("original_path") or fallback_index
    return str(value)


def image_path(sample: dict[str, Any]) -> str:
    return str((sample.get("extensions") or {}).get("original_path", ""))


def normalize_state(state: Any) -> str:
    if state in {"present", "absent", "uncertain"}:
        return str(state)
    return "null"


def uncertainty_value(sample: dict[str, Any], finding: str, state: str) -> str | None:
    if state == "uncertain":
        return "uncertain"
    if state in {"present", "absent"}:
        return "definite"
    value = (sample.get("uncertainty") or {}).get(finding)
    if value is True:
        return "uncertain"
    if value is False:
        return "definite"
    return None


def base_record(
    sample: dict[str, Any],
    idx: int,
    finding: str,
    state: str,
    answer_type: str,
    source_version: str,
    source_mode: str,
) -> dict[str, Any]:
    answerable = state in {"present", "absent", "uncertain"}
    has_report = bool(sample.get("report"))
    return {
        "instruction_id": "",
        "sample_id": sample_id(sample, idx),
        "image_path": image_path(sample),
        "report": sample.get("report"),
        "question": "",
        "answer": "",
        "finding": finding,
        "state": state,
        "answerability": "answerable" if answerable else "not_answerable",
        "uncertainty": uncertainty_value(sample, finding, state),
        "laterality": None,
        "location": None,
        "severity": None,
        "evidence_phrase": None,
        "evidence_source": "structured_label" if answerable and not has_report else "none",
        "answer_type": answer_type,
        "visual_dependency": "medium" if answerable else "low",
        "counterfactual_type": None,
        "quality_flags": [source_version, "report_text_available" if has_report else "no_report_text"],
        "source_version": source_version,
        "source_mode": source_mode,
        "metadata": {
            "study_view": sample.get("study_view"),
            "patient_age": (sample.get("extensions") or {}).get("patient_age"),
            "patient_sex": (sample.get("extensions") or {}).get("patient_sex"),
        },
    }


def make_finding_record(
    sample: dict[str, Any],
    idx: int,
    finding: str,
    state: str,
    source_version: str,
    source_mode: str,
) -> dict[str, Any]:
    record = base_record(sample, idx, finding, state, "finding_verification", source_version, source_mode)
    record["question"] = f"Is {finding} present, absent, uncertain, or not mentioned in this chest X-ray label?"
    record["answer"] = STATE_ANSWERS[state].format(finding=finding)
    return record


def make_answerability_record(
    sample: dict[str, Any],
    idx: int,
    finding: str,
    source_version: str,
    source_mode: str,
) -> dict[str, Any]:
    record = base_record(sample, idx, finding, "null", "answerability", source_version, source_mode)
    record["question"] = f"Is {finding} answerable from the available structured label for this image?"
    record["answer"] = f"Not answerable. The structured label does not mention {finding}."
    return record


def make_counterfactual_record(
    sample: dict[str, Any],
    idx: int,
    finding: str,
    state: str,
    source_version: str,
    source_mode: str,
) -> dict[str, Any]:
    record = base_record(sample, idx, finding, state, "counterfactual_choice", source_version, source_mode)
    options = {
        "A": f"{finding} is present.",
        "B": f"{finding} is absent.",
        "C": f"{finding} is uncertain.",
        "D": f"{finding} is not mentioned.",
    }
    correct = {"present": "A", "absent": "B", "uncertain": "C"}.get(state, "D")
    option_text = " ".join(f"{key}. {value}" for key, value in options.items())
    record["question"] = f"Which statement best matches the structured label? {option_text}"
    record["answer"] = f"{correct}. {options[correct]}"
    record["counterfactual_type"] = "state_choice"
    record["visual_dependency"] = "medium" if state != "null" else "low"
    return record


def generate_local_records(
    sample: dict[str, Any],
    idx: int,
    labels: list[str],
    source_version: str,
    max_instructions: int,
    include_counterfactual: bool,
) -> list[dict[str, Any]]:
    findings = sample.get("findings") or {}
    answerable: list[tuple[str, str]] = []
    missing: list[str] = []

    for finding in labels:
        state = normalize_state((findings.get(finding) or {}).get("state"))
        if state == "null":
            missing.append(finding)
        else:
            answerable.append((finding, state))

    records: list[dict[str, Any]] = []
    for finding, state in answerable:
        records.append(make_finding_record(sample, idx, finding, state, source_version, "local"))
        if include_counterfactual and len(records) < max_instructions:
            records.append(make_counterfactual_record(sample, idx, finding, state, source_version, "local"))
        if len(records) >= max_instructions:
            break

    if len(records) < max_instructions:
        for finding in missing:
            records.append(make_answerability_record(sample, idx, finding, source_version, "local"))
            if len(records) >= max_instructions:
                break

    sid = sample_id(sample, idx)
    for rec_idx, record in enumerate(records):
        record["instruction_id"] = f"{sid}_{source_version}_{rec_idx:03d}"

    return records


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError("No JSON object found in GLM response")
        text = match.group(0)
    return json.loads(text)


def call_glm(
    api_key: str,
    api_base: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    timeout: int,
) -> str:
    endpoint = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept-Language": "en-US,en",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def glm_prompt(sample: dict[str, Any], labels: list[str], source_version: str, max_instructions: int) -> str:
    report = sample.get("report")
    payload = {
        "sample_id": sample_id(sample, 0),
        "image_path": image_path(sample),
        "source_version": source_version,
        "max_instructions": max_instructions,
        "target_findings": labels,
        "report": report,
        "ums_schema": {
            finding: {
                "state": normalize_state(((sample.get("findings") or {}).get(finding) or {}).get("state")),
                "answerable": bool((sample.get("answerability") or {}).get(finding, False)),
                "uncertain": (sample.get("uncertainty") or {}).get(finding),
            }
            for finding in labels
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def generate_glm_records(
    sample: dict[str, Any],
    idx: int,
    labels: list[str],
    source_version: str,
    max_instructions: int,
    prompt_template: str,
    api_key: str,
    api_base: str,
    model: str,
    temperature: float,
    timeout: int,
) -> list[dict[str, Any]]:
    content = call_glm(
        api_key=api_key,
        api_base=api_base,
        model=model,
        messages=[
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": glm_prompt(sample, labels, source_version, max_instructions)},
        ],
        temperature=temperature,
        timeout=timeout,
    )
    parsed = extract_json_object(content)
    instructions = parsed.get("instructions")
    if not isinstance(instructions, list):
        raise ValueError("GLM response JSON does not contain instructions list")

    records: list[dict[str, Any]] = []
    sid = sample_id(sample, idx)
    for rec_idx, item in enumerate(instructions[:max_instructions]):
        if not isinstance(item, dict):
            continue
        finding = str(item.get("finding") or "global")
        state = normalize_state(item.get("state"))
        record = base_record(
            sample=sample,
            idx=idx,
            finding=finding,
            state=state,
            answer_type=str(item.get("answer_type") or "finding_verification"),
            source_version=source_version,
            source_mode="glm",
        )
        for key in record.keys():
            value = item.get(key)
            if key in item and value is not None and value != "":
                record[key] = value
        record["instruction_id"] = f"{sid}_{source_version}_{rec_idx:03d}"
        record["sample_id"] = sid
        record["image_path"] = image_path(sample)
        flags = list(record.get("quality_flags") or [])
        desired_flags = [source_version]
        desired_flags.append("report_text_available" if sample.get("report") else "no_report_text")
        if "no_report_text" in flags and sample.get("report"):
            flags = [flag for flag in flags if flag != "no_report_text"]
        for flag in desired_flags:
            if flag not in flags:
                flags.append(flag)
        record["quality_flags"] = flags
        records.append(record)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--api-log", type=Path)
    parser.add_argument("--parse-errors", type=Path)
    parser.add_argument("--prompt", type=Path, default=Path("prompts/glm_instruction_generation_v1.txt"))
    parser.add_argument("--version", default="v1_label_to_qa")
    parser.add_argument("--mode", choices=["local", "glm"], default="local")
    parser.add_argument("--model", default=os.environ.get("GLM_MODEL", "glm-5.2"))
    parser.add_argument("--api-base", default=os.environ.get("GLM_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--api-key-env", default="GLM_API_KEY")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--max-instructions-per-sample", type=int, default=8)
    parser.add_argument("--include-counterfactual", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--stream-output", action="store_true", help="Append records per sample as they complete.")
    parser.add_argument("--resume", action="store_true", help="With --stream-output, skip sample_ids already present in output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    samples = read_jsonl(args.input, args.max_samples, args.skip_samples)
    prompt_template = args.prompt.read_text(encoding="utf-8") if args.prompt.exists() else ""
    api_key = os.environ.get(args.api_key_env, "")
    if args.mode == "glm" and not api_key:
        raise SystemExit(f"{args.api_key_env} is not set")

    all_records: list[dict[str, Any]] = []
    errors = 0
    skipped = 0
    existing_sample_ids = read_existing_sample_ids(args.output) if args.resume else set()
    if args.stream_output and args.output.exists() and not args.resume:
        args.output.unlink()

    for idx, sample in enumerate(samples):
        sid = sample_id(sample, idx)
        if args.stream_output and args.resume and sid in existing_sample_ids:
            skipped += 1
            continue
        try:
            if args.mode == "local":
                records = generate_local_records(
                    sample=sample,
                    idx=idx,
                    labels=DEFAULT_LABELS,
                    source_version=args.version,
                    max_instructions=args.max_instructions_per_sample,
                    include_counterfactual=args.include_counterfactual,
                )
            else:
                records = generate_glm_records(
                    sample=sample,
                    idx=idx,
                    labels=DEFAULT_LABELS,
                    source_version=args.version,
                    max_instructions=args.max_instructions_per_sample,
                    prompt_template=prompt_template,
                    api_key=api_key,
                    api_base=args.api_base,
                    model=args.model,
                    temperature=args.temperature,
                    timeout=args.timeout,
                )
                if args.api_log:
                    append_jsonl(
                        args.api_log,
                        {
                            "sample_index": idx,
                            "sample_id": sample_id(sample, idx),
                            "status": "ok",
                            "num_records": len(records),
                            "model": args.model,
                            "api_base": args.api_base,
                        },
                    )
                if args.sleep:
                    time.sleep(args.sleep)
            if args.stream_output:
                for record in records:
                    append_jsonl(args.output, record)
            else:
                all_records.extend(records)
        except (ValueError, KeyError, urllib.error.URLError, TimeoutError) as exc:
            errors += 1
            if args.parse_errors:
                append_jsonl(
                    args.parse_errors,
                    {
                        "sample_index": idx,
                        "sample_id": sample_id(sample, idx),
                        "error": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            if args.api_log:
                append_jsonl(
                    args.api_log,
                    {
                        "sample_index": idx,
                        "sample_id": sample_id(sample, idx),
                        "status": "error",
                        "error": type(exc).__name__,
                    },
                )
            if args.sleep:
                time.sleep(args.sleep)

    if not args.stream_output:
        write_jsonl(args.output, all_records)
    elif not args.output.exists():
        write_jsonl(args.output, [])

    output_records = len(all_records)
    if args.stream_output:
        output_records = sum(1 for line in args.output.open("r", encoding="utf-8") if line.strip())
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "mode": args.mode,
                "samples": len(samples),
                "skip_samples": args.skip_samples,
                "records": output_records,
                "errors": errors,
                "skipped": skipped,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
