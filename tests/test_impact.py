"""Tests for Almgren-Chriss optimal trajectory generator and impact models."""

import pytest

from lobx.analytics.impact import (
    AlmgrenChrissParams,
    compute_almgren_chriss_trajectory,
    square_root_market_impact,
)


def test_almgren_chriss_zero_risk_aversion_is_twap() -> None:
    params = AlmgrenChrissParams(
        total_shares=1000,
        total_time_s=60.0,
        num_slices=5,
        volatility=1.0,
        temporary_impact_eta=0.01,
        risk_aversion_lambda=0.0,
    )
    traj = compute_almgren_chriss_trajectory(params)
    assert traj.kappa == 0.0
    assert sum(traj.slice_quantities) == 1000
    # Equal slices: 200 each
    assert traj.slice_quantities == (200, 200, 200, 200, 200)
    assert traj.inventory[0] == 1000.0
    assert traj.inventory[-1] == 0.0


def test_almgren_chriss_risk_aversion_front_loads_execution() -> None:
    params = AlmgrenChrissParams(
        total_shares=1000,
        total_time_s=60.0,
        num_slices=5,
        volatility=1.0,
        temporary_impact_eta=0.01,
        risk_aversion_lambda=1e-3,  # moderate risk aversion
    )
    traj = compute_almgren_chriss_trajectory(params)
    assert traj.kappa > 0.0
    assert sum(traj.slice_quantities) == 1000

    # Front-loading: first slice should be strictly larger than last slice
    assert traj.slice_quantities[0] > traj.slice_quantities[-1]

    # Inventory must decrease monotonically to 0
    for i in range(len(traj.inventory) - 1):
        assert traj.inventory[i] >= traj.inventory[i + 1]
    assert traj.inventory[-1] == 0.0


def test_square_root_law_scaling() -> None:
    imp1 = square_root_market_impact(trade_qty=100, market_volume=10_000, volatility=0.02)
    imp4 = square_root_market_impact(trade_qty=400, market_volume=10_000, volatility=0.02)
    # 4x volume should double the impact (sqrt(4) = 2)
    assert imp4 == pytest.approx(imp1 * 2.0, rel=1e-5)
