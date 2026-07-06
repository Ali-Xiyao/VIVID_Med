"""GLM Coding Plan entry point for clinical instruction generation.

This is a thin compatibility wrapper around ``generate_clinical_instructions.py``
that defaults to GLM mode and the report-grounded prompt. The API key is read
only from an environment variable and is never written to disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_clinical_instructions import main  # noqa: E402


def ensure_default_arg(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    ensure_default_arg("--mode", "glm")
    ensure_default_arg("--prompt", "prompts/glm_instruction_generation_report_grounded_v2.txt")
    main()
