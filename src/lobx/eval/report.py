"""Turn per-seed results into a table and a plot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only: no display is needed (or wanted) on a server or in CI
import matplotlib.pyplot as plt  # noqa: E402

from lobx.eval.harness import SeedResult  # noqa: E402
from lobx.eval.stats import Summary, summarize  # noqa: E402


def summarize_results(results: Sequence[SeedResult]) -> tuple[Summary, int]:
    """Return the summary over completed runs, and how many runs did not complete.

    Incomplete runs have no shortfall; they are counted, never silently dropped, so the table can
    show them.
    """
    values = [r.shortfall_bps for r in results if r.shortfall_bps is not None]
    return summarize(values), len(results) - len(values)


def results_table(rows: Mapping[str, Sequence[SeedResult]]) -> str:
    """Return a markdown table with one row per strategy (shortfall in bps, lower is better)."""
    lines = [
        "| strategy | seeds | incomplete | mean (bps) | std (bps) | 95% CI for the mean (bps) |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, results in rows.items():
        s, incomplete = summarize_results(results)
        lines.append(
            f"| {name} | {s.n} | {incomplete} | {s.mean:.2f} | {s.std:.2f} "
            f"| [{s.ci_low:.2f}, {s.ci_high:.2f}] |"
        )
    return "\n".join(lines)


def plot_results(rows: Mapping[str, Sequence[SeedResult]], path: Path) -> None:
    """Draw each strategy's mean shortfall with its 95% confidence interval, and write a PNG."""
    names = list(rows)
    summaries = [summarize_results(rows[name])[0] for name in names]
    means = [s.mean for s in summaries]
    errors = [[s.mean - s.ci_low for s in summaries], [s.ci_high - s.mean for s in summaries]]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(names, means, yerr=errors, capsize=6, color="#4C72B0")
    ax.set_ylabel("implementation shortfall (bps)")
    ax.set_title("Mean shortfall with 95% CI (simulated market, lower is better)")
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
