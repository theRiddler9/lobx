"""Pairs execution strategy for statistical arbitrage and delta-neutral portfolios.

Coordinates simultaneous execution across two correlated assets (e.g. Long A, Short B)
while dynamically controlling leg risk, execution slippage, and net delta exposure.
"""

from __future__ import annotations

from dataclasses import dataclass

from lobx.sim.actions import Action, Submit
from lobx.sim.cross_market import CrossMarketView
from lobx.types import Side


@dataclass(frozen=True)
class PairsParentOrder:
    """Institutional parent order specifying simultaneous execution across two assets."""

    agent_id: int
    symbol_a: str
    side_a: Side
    qty_a: int
    symbol_b: str
    side_b: Side
    qty_b: int
    start_ns: int
    end_ns: int
    max_leg_imbalance: int = 50


@dataclass
class PairsExecutionMetrics:
    """Summary metrics of pairs order execution."""

    executed_a: int
    executed_b: int
    arrival_mid_a: float
    arrival_mid_b: float
    vwap_a: float
    vwap_b: float
    shortfall_a_bps: float
    shortfall_b_bps: float
    combined_shortfall_bps: float
    max_unhedged_qty: int


class PairsExecutionStrategy:
    """Executes a 2-legged pairs order with dynamic leg-risk throttling."""

    def __init__(self, parent: PairsParentOrder, num_slices: int = 5) -> None:
        """Initialize PairsExecutionStrategy with parent order specification."""
        self.parent = parent
        self.num_slices = max(1, num_slices)
        self.slice_qty_a = max(1, parent.qty_a // self.num_slices)
        self.slice_qty_b = max(1, parent.qty_b // self.num_slices)

        self._arrival_mid_a: float | None = None
        self._arrival_mid_b: float | None = None

        self._executed_a = 0
        self._executed_b = 0
        self._cost_a = 0.0
        self._cost_b = 0.0
        self._max_unhedged = 0

    @property
    def agent_id(self) -> int:
        """Return the agent ID from parent order."""
        return self.parent.agent_id

    @property
    def arrival_mid_a(self) -> float | None:
        """Return recorded arrival mid-price for symbol A."""
        return self._arrival_mid_a

    @property
    def arrival_mid_b(self) -> float | None:
        """Return recorded arrival mid-price for symbol B."""
        return self._arrival_mid_b

    def wake(self, ts: int, view: CrossMarketView) -> list[Action]:
        """Wake up on timer and execute step."""
        return self.step(ts, view)

    def step(self, ts: int, view: CrossMarketView) -> list[Action]:
        """Compute child orders for the current slice, throttling unhedged legs."""
        actions: list[Action] = []
        if ts < self.parent.start_ns:
            return actions

        # Record arrival mid prices on first step
        if self._arrival_mid_a is None:
            self._arrival_mid_a = view.mid_price(self.parent.symbol_a) or 10000.0
        if self._arrival_mid_b is None:
            self._arrival_mid_b = view.mid_price(self.parent.symbol_b) or 10000.0

        # Read actual filled quantities from positions
        pos_a = abs(view.position(self.parent.agent_id, self.parent.symbol_a))
        pos_b = abs(view.position(self.parent.agent_id, self.parent.symbol_b))
        self._executed_a = min(pos_a, self.parent.qty_a)
        self._executed_b = min(pos_b, self.parent.qty_b)

        # Track leg imbalance
        imbalance = abs(self._executed_a - self._executed_b)
        self._max_unhedged = max(self._max_unhedged, imbalance)

        rem_a = self.parent.qty_a - self._executed_a
        rem_b = self.parent.qty_b - self._executed_b

        if rem_a <= 0 and rem_b <= 0:
            return actions

        # Is this the final deadline sweep?
        deadline = ts >= self.parent.end_ns

        # Dynamic leg control: if Leg A is far ahead, pause A and aggressively fill B
        allow_a = rem_a > 0 and (
            deadline or (self._executed_a - self._executed_b < self.parent.max_leg_imbalance)
        )
        allow_b = rem_b > 0 and (
            deadline or (self._executed_b - self._executed_a < self.parent.max_leg_imbalance)
        )

        if allow_a:
            order_qty_a = rem_a if deadline else min(rem_a, self.slice_qty_a)
            actions.append(
                Submit.market(
                    side=self.parent.side_a,
                    qty=order_qty_a,
                    symbol=self.parent.symbol_a,
                )
            )

        if allow_b:
            order_qty_b = rem_b if deadline else min(rem_b, self.slice_qty_b)
            actions.append(
                Submit.market(
                    side=self.parent.side_b,
                    qty=order_qty_b,
                    symbol=self.parent.symbol_b,
                )
            )

        return actions

    def compute_metrics(self, view: CrossMarketView) -> PairsExecutionMetrics:
        """Compute execution metrics and implementation shortfall in basis points."""
        pos_a = abs(view.position(self.parent.agent_id, self.parent.symbol_a))
        pos_b = abs(view.position(self.parent.agent_id, self.parent.symbol_b))
        self._executed_a = min(pos_a, self.parent.qty_a)
        self._executed_b = min(pos_b, self.parent.qty_b)

        arr_a = self._arrival_mid_a or 10000.0
        arr_b = self._arrival_mid_b or 10000.0

        # Estimate final benchmark prices
        cur_a = view.mid_price(self.parent.symbol_a) or arr_a
        cur_b = view.mid_price(self.parent.symbol_b) or arr_b

        # Leg A shortfall bps
        sign_a = 1.0 if self.parent.side_a is Side.BUY else -1.0
        shortfall_a = ((cur_a - arr_a) / arr_a) * 10_000.0 * sign_a

        # Leg B shortfall bps
        sign_b = 1.0 if self.parent.side_b is Side.BUY else -1.0
        shortfall_b = ((cur_b - arr_b) / arr_b) * 10_000.0 * sign_b

        # Combined portfolio shortfall bps
        combined_shortfall = (shortfall_a + shortfall_b) / 2.0

        return PairsExecutionMetrics(
            executed_a=self._executed_a,
            executed_b=self._executed_b,
            arrival_mid_a=arr_a,
            arrival_mid_b=arr_b,
            vwap_a=cur_a,
            vwap_b=cur_b,
            shortfall_a_bps=shortfall_a,
            shortfall_b_bps=shortfall_b,
            combined_shortfall_bps=combined_shortfall,
            max_unhedged_qty=self._max_unhedged,
        )
