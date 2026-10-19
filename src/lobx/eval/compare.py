"""Paired comparison of strategies: same seeds, same starting market, per-seed differences.

Comparing two means with their own confidence intervals throws information away. Every strategy
runs on the same seeds, so seed 1042 gives a TWAP cost and an adaptive cost *in the same market*.
The per-seed difference cancels most of the luck of the draw (a market that happens to fall
hurts every strategy), leaving a much sharper estimate of which strategy is really cheaper.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

from lobx.eval.harness import SeedResult
from lobx.eval.stats import Summary, summarize


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """``a`` versus ``b`` over the seeds where both completed."""

    a: str
    b: str
    used: int
    excluded: int  # seeds where at least one of the two did not complete
    difference: Summary  # a - b per seed, in bps: negative means a is cheaper
    a_cheaper: int  # seeds where a's shortfall was strictly lower


def paired_comparison(
    a_name: str, a: Sequence[SeedResult], b_name: str, b: Sequence[SeedResult]
) -> PairedComparison:
    """Compare two strategies' results seed by seed."""
    b_by_seed = {r.seed: r for r in b}
    if set(b_by_seed) != {r.seed for r in a}:
        raise ValueError("a paired comparison needs both strategies run on the same seeds")
    differences: list[float] = []
    a_cheaper = 0
    for ra in a:
        rb = b_by_seed[ra.seed]
        if ra.shortfall_bps is None or rb.shortfall_bps is None:
            continue
        differences.append(ra.shortfall_bps - rb.shortfall_bps)
        a_cheaper += ra.shortfall_bps < rb.shortfall_bps
    return PairedComparison(
        a_name,
        b_name,
        len(differences),
        len(a) - len(differences),
        summarize(differences),
        a_cheaper,
    )


def verdict(c: PairedComparison) -> str:
    """Say which strategy is cheaper, or that the data cannot tell, from the interval's position.

    ``difference`` is a - b. If its whole 95% interval is below zero, a is cheaper; if it is all
    above zero, b is. An interval that contains zero means the data cannot separate them.
    """
    if c.difference.ci_high < 0:
        return f"{c.a} cheaper"
    if c.difference.ci_low > 0:
        return f"{c.b} cheaper"
    return "no clear difference"


def comparison_table(results: Mapping[str, Sequence[SeedResult]]) -> str:
    """Return a markdown table comparing every strategy with each one listed before it."""
    lines = [
        "| comparison | seeds used | a - b, mean (bps) | 95% CI (bps) | a cheaper on | verdict |",
        "|---|---:|---:|---|---:|---|",
    ]
    for b_name, a_name in combinations(results, 2):
        c = paired_comparison(a_name, results[a_name], b_name, results[b_name])
        d = c.difference
        lines.append(
            f"| {a_name} - {b_name} | {c.used} | {d.mean:.2f} | [{d.ci_low:.2f}, {d.ci_high:.2f}] "
            f"| {c.a_cheaper / c.used:.0%} of seeds | {verdict(c)} |"
        )
    return "\n".join(lines)
