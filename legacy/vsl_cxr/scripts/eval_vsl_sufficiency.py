"""VSL-CXR v5 wrapper: summarize sufficiency-label formal runs."""

import sys

from build_vsl_auxiliary_audits import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "sufficiency"]
    main()
