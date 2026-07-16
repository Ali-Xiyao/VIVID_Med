"""Extract P4-v2 clinical facts from report manifests with a GLM-compatible API."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import time
import urllib.error
from pathlib import Path
from typing import Any

from generate_clinical_instructions import DEFAULT_API_BASE, call_glm


ALLOWED_FINDINGS = [
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
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

FINDING_ALIASES = {
    re.sub(r"[^a-z0-9]+", "", item.lower()): item for item in ALLOWED_FINDINGS
}
FINDING_ALIASES.update(
    {
        "pleuraleffusions": "Pleural Effusion",
        "effusion": "Pleural Effusion",
        "pneumothoraces": "Pneumothorax",
        "opacity": "Lung Opacity",
        "opacities": "Lung Opacity",
        "supportdevice": "Support Devices",
        "supportdevices": "Support Devices",
    }
)

ALLOWED_STATES = {"present", "absent", "uncertain"}
ALLOWED_LOCATIONS = {"left", "right", "bilateral", "basilar", "apical", "diffuse", None}
ALLOWED_SEVERITIES = {"tiny", "small", "mild", "moderate", "large", "severe", None}
ALLOWED_CERTAINTY = {"definite", "uncertain"}
ALLOWED_VISUAL_DEP = {"high", "medium", "low"}


def read_jsonl(path: Path, max_samples: int | None = None, skip_samples: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx < skip_samples:
                continue
            if max_samples is not None and len(rows) >= max_samples:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sample_id(sample: dict[str, Any], fallback_index: int) -> str:
    extensions = sample.get("extensions") or {}
    return str(extensions.get("sample_id") or extensions.get("original_path") or fallback_index)


def image_path(sample: dict[str, Any]) -> str:
    return str((sample.get("extensions") or {}).get("original_path") or "")


def read_existing_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = row.get("sample_id")
            if sid is not None:
                seen.add(str(sid))
    return seen


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


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


def normalize_finding(value: Any) -> str | None:
    key = re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    return FINDING_ALIASES.get(key)


def normalize_choice(value: Any, allowed: set[str | None], default: str | None) -> str | None:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"", "none", "null", "n/a", "na", "unknown"}:
        return default
    return text if text in allowed else default


def normalize_fact(item: dict[str, Any], report: str) -> dict[str, Any] | None:
    finding = normalize_finding(item.get("finding"))
    state = normalize_choice(item.get("state"), ALLOWED_STATES, None)
    if finding is None or state is None:
        return None
    evidence = str(item.get("evidence_span") or "").strip()
    if evidence and evidence.casefold() not in report.casefold():
        evidence = ""
    certainty = normalize_choice(item.get("certainty"), ALLOWED_CERTAINTY, "uncertain" if state == "uncertain" else "definite")
    return {
        "finding": finding,
        "state": state,
        "evidence_span": evidence,
        "location": normalize_choice(item.get("location"), ALLOWED_LOCATIONS, None),
        "severity": normalize_choice(item.get("severity"), ALLOWED_SEVERITIES, None),
        "certainty": certainty,
        "visual_dependency": normalize_choice(item.get("visual_dependency"), ALLOWED_VISUAL_DEP, "medium"),
    }


def build_user_payload(sample: dict[str, Any], sid: str) -> str:
    payload = {
        "sample_id": sid,
        "allowed_findings": ALLOWED_FINDINGS,
        "report": sample.get("report") or "",
    }
    return json.dumps(payload, ensure_ascii=False)


def extract_facts(
    sample: dict[str, Any],
    sid: str,
    prompt_template: str,
    api_key: str,
    api_base: str,
    model: str,
    temperature: float,
    timeout: int,
) -> dict[str, Any]:
    report = str(sample.get("report") or "")
    content = call_glm(
        api_key=api_key,
        api_base=api_base,
        model=model,
        messages=[
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": build_user_payload(sample, sid)},
        ],
        temperature=temperature,
        timeout=timeout,
    )
    parsed = extract_json_object(content)
    raw_facts = parsed.get("facts") or []
    if not isinstance(raw_facts, list):
        raise ValueError("GLM response JSON does not contain facts list")
    facts = []
    seen: set[tuple[str, str, str | None, str | None]] = set()
    for item in raw_facts:
        if not isinstance(item, dict):
            continue
        fact = normalize_fact(item, report)
        if fact is None:
            continue
        key = (fact["finding"], fact["state"], fact["location"], fact["severity"])
        if key in seen:
            continue
        seen.add(key)
        facts.append(fact)
    unmentioned = [
        normalize_finding(value)
        for value in (parsed.get("unmentioned_findings") or [])
        if normalize_finding(value) is not None
    ]
    return {
        "sample_id": sid,
        "image_path": image_path(sample),
        "report": report,
        "facts": facts,
        "unmentioned_findings": sorted(set(unmentioned)),
        "source": "glm_fact_extraction",
        "model": model,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--api-log", type=Path)
    parser.add_argument("--parse-errors", type=Path)
    parser.add_argument("--prompt", type=Path, default=Path("prompts/glm_p4v2_fact_extraction.txt"))
    parser.add_argument("--model", default=os.environ.get("GLM_MODEL", "glm-5.2"))
    parser.add_argument("--api-base", default=os.environ.get("GLM_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--api-key-env", default="ZHIPU_API_KEY")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--target-output-count", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is not set")
    prompt_template = args.prompt.read_text(encoding="utf-8")
    samples = read_jsonl(args.input, max_samples=args.max_samples, skip_samples=args.skip_samples)
    existing_sample_ids = read_existing_sample_ids(args.output) if args.resume else set()
    starting_output_count = count_jsonl_rows(args.output) if args.resume else 0
    if args.output.exists() and not args.resume:
        args.output.unlink()
    errors = 0
    skipped = 0
    written = 0
    for local_idx, sample in enumerate(samples):
        if args.target_output_count is not None and starting_output_count + written >= args.target_output_count:
            break
        sid = sample_id(sample, args.skip_samples + local_idx)
        if sid in existing_sample_ids:
            skipped += 1
            continue
        try:
            row = extract_facts(
                sample=sample,
                sid=sid,
                prompt_template=prompt_template,
                api_key=api_key,
                api_base=args.api_base,
                model=args.model,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            append_jsonl(args.output, row)
            written += 1
            if args.api_log:
                append_jsonl(
                    args.api_log,
                    {
                        "sample_id": sid,
                        "status": "ok",
                        "num_facts": len(row["facts"]),
                        "model": args.model,
                        "api_base": args.api_base,
                    },
                )
        except (
            ValueError,
            KeyError,
            json.JSONDecodeError,
            urllib.error.URLError,
            TimeoutError,
            http.client.RemoteDisconnected,
            ConnectionResetError,
        ) as exc:
            errors += 1
            if args.parse_errors:
                append_jsonl(
                    args.parse_errors,
                    {
                        "sample_id": sid,
                        "status": "error",
                        "error": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            if args.api_log:
                append_jsonl(args.api_log, {"sample_id": sid, "status": "error", "error": type(exc).__name__})
        if args.sleep:
            time.sleep(args.sleep)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "samples": len(samples),
                "written": written,
                "skipped": skipped,
                "errors": errors,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
