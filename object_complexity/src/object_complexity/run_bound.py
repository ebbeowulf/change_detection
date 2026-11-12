#!/usr/bin/env python3
"""Run one recall lower-bound method: python run_bound.py --method binomial --model siglip2-so400m ..."""
import argparse

from bound_methods import METHODS
from bound_methods.base import RecallBound


def main():
    p = argparse.ArgumentParser(description="Evaluate a recall lower-bound method, leaving one category out at a time.")
    p.add_argument("--method", type=str, required=True, choices=list(METHODS), help="Bound method to run")
    RecallBound.add_arguments(p)
    for cls in METHODS.values():
        cls.add_method_arguments(p)
    args = p.parse_args()
    METHODS[args.method](args).run()


if __name__ == "__main__":
    main()