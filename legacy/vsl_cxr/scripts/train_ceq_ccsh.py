import sys

from cvcp_ccsh_driver import main


if __name__ == "__main__":
    if "--default-modules" not in sys.argv and "--module" not in sys.argv:
        sys.argv[1:1] = ["--default-modules", "CEQ", "CCSH"]
    main("train_module_stack")
