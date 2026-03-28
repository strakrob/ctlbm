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
    print(f"step: {last['step']}")
    print(f"total_mass: {last['total_mass']}")
    print(f"mean_density: {last['mean_density']}")
    print(f"bulk_velocity: {last['bulk_velocity']}")
    print(f"flow_rate: {last['flow_rate']}")
    print(f"max_streamwise_velocity: {last['max_streamwise_velocity']}")
    print(f"residual: {last['residual']}")
    print(f"l2_error: {last['l2_error']}")
    print(f"balance_metric: {last['balance_metric']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
