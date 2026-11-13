"""Market impact modeling and Almgren-Chriss optimal execution trajectory.

Implements the classical Almgren & Chriss (2000) optimal execution framework:
- Closed-form discrete trajectory generator balancing market impact against price volatility risk.
- Expected shortfall E[x] and variance of shortfall V[x] computation.
- Square-root law and linear market impact calibration.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lobx.types import BookSnapshot, Fill


@dataclass(frozen=True)
class AlmgrenChrissParams:
    """Parameters governing the Almgren-Chriss optimal execution model."""

    total_shares: int
    total_time_s: float
    num_slices: int
    volatility: float  # asset volatility (e.g. sigma in ticks/sqrt(s) or annual vol)
    temporary_impact_eta: float  # eta parameter (penalty per unit trade rate)
    permanent_impact_gamma: float = 0.0  # gamma parameter (permanent price shift)
    risk_aversion_lambda: float = 1e-4  # trader risk aversion parameter


@dataclass(frozen=True)
class AlmgrenChrissTrajectory:
    """Optimal discrete inventory trajectory and slice schedule."""

    times: tuple[float, ...]  # t_0, t_1, ..., t_N
    inventory: tuple[float, ...]  # x_0, x_1, ..., x_N
    slice_quantities: tuple[int, ...]  # n_1, ..., n_N
    expected_shortfall: float
    variance: float
    kappa: float


def compute_almgren_chriss_trajectory(
    params: AlmgrenChrissParams,
) -> AlmgrenChrissTrajectory:
    """Compute discrete optimal inventory trajectory and integer slice quantities."""
    total_x = float(params.total_shares)
    total_t = float(params.total_time_s)
    n_steps = max(1, params.num_slices)
    tau = total_t / n_steps

    sigma = max(1e-8, params.volatility)
    eta = max(1e-8, params.temporary_impact_eta)
    gamma = params.permanent_impact_gamma
    lam = max(0.0, params.risk_aversion_lambda)

    # When risk aversion is zero (risk-neutral), optimal trajectory is pure linear TWAP
    if lam < 1e-12:
        kappa = 0.0
        times = tuple(k * tau for k in range(n_steps + 1))
        inventory = tuple(total_x * (1.0 - k / n_steps) for k in range(n_steps + 1))
        # Equal slices
        base_slice = int(total_x // n_steps)
        rem = int(total_x - base_slice * n_steps)
        slices = [base_slice + (1 if k < rem else 0) for k in range(n_steps)]

        expected_shortfall = 0.5 * gamma * (total_x**2) + eta * sum(
            (n / tau) ** 2 * tau for n in slices
        )
        variance = (sigma**2) * tau * sum(x**2 for x in inventory[1:])

        return AlmgrenChrissTrajectory(
            times=times,
            inventory=inventory,
            slice_quantities=tuple(slices),
            expected_shortfall=expected_shortfall,
            variance=variance,
            kappa=0.0,
        )

    # General risk-averse case: kappa = (1/tau) * arccosh(1 + 0.5 * lambda * sigma^2 * tau^2 / eta)
    arg = 1.0 + 0.5 * (lam * (sigma**2) * (tau**2)) / eta
    kappa_tau = math.acosh(arg)
    kappa = kappa_tau / tau

    times_list: list[float] = []
    inv_list: list[float] = []

    # x_j = X * sinh(kappa * (T - t_j)) / sinh(kappa * T)
    sinh_kt = math.sinh(min(700.0, kappa * total_t))
    if sinh_kt == 0.0:
        sinh_kt = 1e-8

    for j in range(n_steps + 1):
        t_j = j * tau
        times_list.append(t_j)
        if j == 0:
            inv_list.append(total_x)
        elif j == n_steps:
            inv_list.append(0.0)
        else:
            val = total_x * math.sinh(min(700.0, kappa * (total_t - t_j))) / sinh_kt
            inv_list.append(max(0.0, min(total_x, val)))

    # Compute integer slice quantities n_j = x_{j-1} - x_j
    raw_slices = [inv_list[j - 1] - inv_list[j] for j in range(1, n_steps + 1)]
    # Round preserving sum = X
    slices = [max(0, int(round(s))) for s in raw_slices]
    diff = int(params.total_shares - sum(slices))
    if diff != 0 and n_steps > 0:
        # Distribute discrepancy across slices
        idx = 0
        while diff > 0:
            slices[idx % n_steps] += 1
            diff -= 1
            idx += 1
        while diff < 0:
            if slices[idx % n_steps] > 0:
                slices[idx % n_steps] -= 1
                diff += 1
            idx += 1

    expected_shortfall = 0.5 * gamma * (total_x**2) + eta * sum(
        (n / tau) ** 2 * tau for n in slices
    )
    variance = (sigma**2) * tau * sum(x**2 for x in inv_list[1:])

    return AlmgrenChrissTrajectory(
        times=tuple(times_list),
        inventory=tuple(inv_list),
        slice_quantities=tuple(slices),
        expected_shortfall=expected_shortfall,
        variance=variance,
        kappa=kappa,
    )


def square_root_market_impact(
    trade_qty: int,
    market_volume: int,
    volatility: float,
    y_factor: float = 0.5,
) -> float:
    """Compute expected instantaneous price impact via the empirical Square-Root Law.

    I(Q) = Y * sigma * sqrt(Q / V)
    """
    if trade_qty <= 0 or market_volume <= 0:
        return 0.0
    participation_rate = float(trade_qty) / float(market_volume)
    return y_factor * volatility * math.sqrt(participation_rate)


def estimate_linear_impact(
    trades: Sequence[Fill],
    snapshots: Sequence[BookSnapshot],
) -> tuple[float, float]:
    """Estimate permanent impact gamma and temporary impact eta from historical fills.

    Uses regression of price shifts on trade signs and trade sizes.
    Returns (gamma, eta).
    """
    if len(trades) < 5:
        return (1e-6, 1e-4)

    # Simplified empirical proxy: regression of trade-by-trade price displacement
    sizes = np.array([float(t.qty) for t in trades], dtype=np.float64)
    prices = np.array([float(t.price) for t in trades], dtype=np.float64)

    # Price differences
    dp = np.diff(prices)
    dsizes = sizes[1:]

    var_s = float(np.var(dsizes))
    if var_s < 1e-6:
        return (1e-6, 1e-4)

    cov = float(np.cov(dsizes, dp)[0, 1])
    gamma = max(1e-8, cov / var_s)
    eta = max(1e-6, gamma * 2.0)
    return (gamma, eta)
