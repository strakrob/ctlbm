#!/usr/bin/env python3

import csv
import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_rect_duct.py PATH/TO/diagnostics.csv", file=sys.stderr)
        return 1

    path = pathlib.Path(sys.argv[1])
    if not path.is_file():
        print(f"diagnostics file not found: {path}", file=sys.stderr)
        return 1

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        print("diagnostics file is empty", file=sys.stderr)
        return 1

    last = rows[-1]
    for key, value in last.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
