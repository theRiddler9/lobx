"""A workload for profiling: the adaptive strategy on a few test seeds, the slowest part of eval.

Usage:
    py-spy record --format flamegraph --rate 200 -o docs/figures/flamegraph.svg -- \
        python benchmarks/profile_eval.py --seeds 8
(py-spy is a profiling tool, not a project dependency: `pip install py-spy`.)
"""

import argparse
import time
from pathlib import Path

from lobx.eval.runner import DEFAULT_MODEL_PATH, evaluate, obtain_model


def main() -> None:
    """Run the adaptive strategy on ``--seeds`` test seeds and print how long it took."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    model = obtain_model(args.model)
    start = time.perf_counter()
    evaluate(["adaptive"], args.seeds, model=model)
    elapsed = time.perf_counter() - start
    print(
        f"adaptive on {args.seeds} seeds: {elapsed:.1f} s ({elapsed / args.seeds:.2f} s per seed)"
    )


if __name__ == "__main__":
    main()
