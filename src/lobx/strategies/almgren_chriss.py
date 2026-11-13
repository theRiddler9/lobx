"""Almgren-Chriss optimal execution strategy.

Implements the institutional Almgren & Chriss (2000) execution schedule,
balancing market impact costs against volatility risk through an explicit
risk-aversion parameter lambda.
"""

from __future__ import annotations

from lobx.analytics.impact import AlmgrenChrissParams, compute_almgren_chriss_trajectory
from lobx.sim.clock import NS_PER_S
from lobx.strategies.base import ParentOrder
from lobx.strategies.schedule import ScheduledStrategy


class AlmgrenChrissStrategy(ScheduledStrategy):
    """Execute parent order following the Almgren-Chriss optimal trajectory."""

    def __init__(
        self,
        num_slices: int = 10,
        risk_aversion: float = 1e-4,
        volatility: float = 1.0,
        temporary_impact_eta: float = 0.01,
        permanent_impact_gamma: float = 0.0,
    ) -> None:
        """Initialize Almgren-Chriss strategy with market and risk parameters."""
        if num_slices <= 0:
            raise ValueError(f"num_slices must be greater than 0, got {num_slices}")
        self.risk_aversion = max(0.0, risk_aversion)
        self.volatility = max(1e-8, volatility)
        self.temporary_impact_eta = max(1e-8, temporary_impact_eta)
        self.permanent_impact_gamma = permanent_impact_gamma

        # Temporary initial weights; replaced upon reset(parent)
        super().__init__([1.0] * num_slices)

    def reset(self, parent: ParentOrder) -> None:
        """Compute the optimal Almgren-Chriss slice schedule for this parent order."""
        total_time_s = (parent.end_ns - parent.start_ns) / NS_PER_S
        params = AlmgrenChrissParams(
            total_shares=parent.qty,
            total_time_s=max(1.0, total_time_s),
            num_slices=self.num_slices,
            volatility=self.volatility,
            temporary_impact_eta=self.temporary_impact_eta,
            permanent_impact_gamma=self.permanent_impact_gamma,
            risk_aversion_lambda=self.risk_aversion,
        )
        traj = compute_almgren_chriss_trajectory(params)
        weights = [float(q) for q in traj.slice_quantities]
        if sum(weights) <= 0:
            weights = [1.0] * self.num_slices

        self._weights = tuple(weights)
        super().reset(parent)
