"""Summary statistics with honest uncertainty: mean, standard deviation, 95% confidence interval."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# Two-sided 95% critical values of Student's t for 1 to 4 degrees of freedom (exact, from tables).
_T_EXACT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
_Z_975 = 1.959964  # the standard normal's 97.5th percentile


def t_critical_95(df: int) -> float:
    """Return the two-sided 95% critical value of Student's t with ``df`` degrees of freedom.

    For ``df`` of 5 or more this uses the Cornish-Fisher expansion of the t quantile around the
    normal quantile, accurate to about 0.003 at df=10 and better beyond; for fewer it uses the
    exact table. (scipy would do this in one call but is not an allowed dependency.)
    """
    if df < 1:
        raise ValueError(f"degrees of freedom must be at least 1, got {df}")
    if df in _T_EXACT:
        return _T_EXACT[df]
    z = _Z_975
    return (
        z
        + (z**3 + z) / (4 * df)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3)
    )


@dataclass(frozen=True, slots=True)
class Summary:
    """Mean, sample standard deviation, and 95% confidence interval for the mean."""

    n: int
    mean: float
    std: float
    ci_low: float
    ci_high: float


def summarize(values: Sequence[float]) -> Summary:
    """Summarize ``values`` (needs at least two).

    ``std`` is the *sample* standard deviation (divides by n-1). The interval is for the *mean*:
    ``mean +/- t * std / sqrt(n)``. It says how well the mean is pinned down by n runs, not how
    much a single run varies (that is ``std``).
    """
    n = len(values)
    if n < 2:
        raise ValueError(f"need at least 2 values to estimate a spread, got {n}")
    mean = math.fsum(values) / n
    std = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (n - 1))
    half_width = t_critical_95(n - 1) * std / math.sqrt(n)
    return Summary(n, mean, std, mean - half_width, mean + half_width)
