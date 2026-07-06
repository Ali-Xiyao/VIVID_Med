"""Smoke-test the TMI module components on synthetic tensors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

from case_study_modules_common import FINAL_DIR, write_csv_rows, write_md_table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import (
    AnswerabilityUncertaintyHead,
    CaseDrivenCurriculumScheduler,
    ClinicalConsistencyHead,
    ClinicalEvidenceQuery,
    DomainRobustAdapter,
    HardNegativeMemoryBank,
)


COLUMNS = ["module", "status", "input_shape", "output_shape", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=FINAL_DIR / "module_smoke_results.csv")
    parser.add_argument("--output-md", type=Path, default=FINAL_DIR / "module_smoke_results.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    patch_tokens = torch.randn(2, 16, 32)
    ceq = ClinicalEvidenceQuery(num_findings=8, embed_dim=32, num_heads=4)
    ceq_out = ceq(patch_tokens)
    rows.append({"module": "CEQ", "status": "passed", "input_shape": list(patch_tokens.shape), "output_shape": list(ceq_out["evidence"].shape), "notes": "finding evidence embeddings"})

    auch = AnswerabilityUncertaintyHead(embed_dim=32)
    auch_out = auch(ceq_out["evidence"])
    auch_loss = auch.loss(auch_out, answerable=torch.ones(2, 8), uncertain=torch.zeros(2, 8), state=torch.zeros(2, 8, dtype=torch.long))
    rows.append({"module": "AUCH", "status": "passed", "input_shape": list(ceq_out["evidence"].shape), "output_shape": list(auch_out["state_logits"].shape), "notes": f"loss={float(auch_loss):.6f}"})

    bank = HardNegativeMemoryBank()
    bank.add_batch(["a", "b"], torch.randn(2, 32), [{"finding": "Effusion", "state": "present"}, {"finding": "Effusion", "state": "absent"}])
    mined = bank.mine(torch.randn(32), sample_id="a", finding="Effusion", state="present")
    rows.append({"module": "HNMB", "status": "passed", "input_shape": "[2,32]", "output_shape": len(mined), "notes": "nearest opposite metadata mining"})

    dra = DomainRobustAdapter(embed_dim=32, domain_count=2)
    adapted = dra(torch.randn(4, 32), domain_id=1)
    rows.append({"module": "DRA", "status": "passed", "input_shape": "[4,32]", "output_shape": list(adapted.shape), "notes": "residual domain adapter"})

    ccsh = ClinicalConsistencyHead(image_dim=32, statement_dim=32)
    logits = ccsh(torch.randn(4, 32), torch.randn(4, 32))
    rows.append({"module": "CCSH", "status": "passed", "input_shape": "[4,32]+[4,32]", "output_shape": list(logits.shape), "notes": "support/contradict/uncertain logits"})

    scheduler = CaseDrivenCurriculumScheduler()
    annotated = scheduler.annotate([{"finding": "Effusion", "failure_type": "laterality"}], [{"finding": "Effusion", "failure_type": "laterality"}])
    rows.append({"module": "CDCS", "status": "passed", "input_shape": "1 sample + 1 case", "output_shape": len(annotated), "notes": f"weight={annotated[0]['cdcs_sampling_weight']:.3f}"})

    write_csv_rows(args.output_csv, rows, COLUMNS)
    write_md_table(args.output_md, "TMI Module Smoke Results", rows, COLUMNS)


if __name__ == "__main__":
    main()
