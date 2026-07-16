"""VSL-CXR v5 wrapper: summarize current calibration evidence."""

import sys

from build_vsl_auxiliary_audits import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "calibration"]
    main()
