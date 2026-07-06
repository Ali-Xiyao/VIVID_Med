import sys

from cvcp_ccsh_driver import main


if __name__ == "__main__":
    if "--mode" not in sys.argv:
        sys.argv[1:1] = ["--mode", "ab_swap"]
    main("summarize")
