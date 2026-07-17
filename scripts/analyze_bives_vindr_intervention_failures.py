from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bives_cxr.intervention_diagnostics import (  # noqa: E402
    file_sha256,
    render_markdown,
    summarize_intervention_failures,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only failure taxonomy for frozen VinDr intervention rows."
    )
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--primary-dilation", type=float, default=0.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.rows.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = summarize_intervention_failures(
        rows,
        primary_dilation=args.primary_dilation,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    rows_sha256 = file_sha256(args.rows)
    summary["intervention_rows_sha256"] = rows_sha256
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(
        render_markdown(summary, rows_sha256=rows_sha256), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "row_count": len(rows),
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
                "intervention_rows_sha256": rows_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
