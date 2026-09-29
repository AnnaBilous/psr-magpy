#!/usr/bin/env python
"""Build the docs by running ``make docs``."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["make", "docs"], cwd=root)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
