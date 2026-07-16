"""VSL-CXR v5 wrapper: audit current hard-negative pair sources."""

import sys

from build_vsl_auxiliary_audits import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "data_sources"]
    main()
