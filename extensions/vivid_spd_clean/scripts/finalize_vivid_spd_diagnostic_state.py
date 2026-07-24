"""Reconcile a preserved verdict with a queue stopped in reporting failure."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-state", required=True, type=Path)
    parser.add_argument("--verdict", required=True, type=Path)
    args = parser.parse_args()
    state = json.loads(args.queue_state.read_text(encoding="utf-8"))
    verdict = json.loads(args.verdict.read_text(encoding="utf-8"))
    if state.get("terminal_reason") != (
        "diagnostic_verdict_implementation_failed"
    ):
        raise ValueError("queue is not the preserved reporting failure")
    if verdict.get("verdict") not in {"TERMINAL_NO_GO", "REPAIR_NOMINATED"}:
        raise ValueError("unsupported diagnostic verdict")
    state.update(
        status=(
            "terminal_no_go"
            if verdict["verdict"] == "TERMINAL_NO_GO"
            else "repair_nominated"
        ),
        stage="terminal",
        verdict=verdict["verdict"],
        returncode=0,
        terminal_reason=(
            "bounded_diagnostics_do_not_support_repair"
            if verdict["verdict"] == "TERMINAL_NO_GO"
            else "bounded_diagnostics_support_single_repair"
        ),
        finalized_after_reporting_repair=True,
        finished_at_unix=time.time(),
    )
    args.queue_state.write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
