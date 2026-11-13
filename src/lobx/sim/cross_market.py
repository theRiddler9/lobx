"""Multi-asset cross-market simulation environment.

Simulates multiple concurrent symbol order books, cross-asset information propagation,
lead-lag price discovery dynamics, and multi-symbol agent interactions.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from lobx.engine.book_py import PythonOrderBook
from lobx.engine.multi_book import MultiOrderBook
from lobx.engine.protocol import OrderBook
from lobx.sim.actions import Action, Cancel, Submit
from lobx.sim.clock import Clock, WakeQueue
from lobx.sim.rng import AgentStreams
from lobx.types import BookSnapshot, Fill, Order, OrderType, Side

logger = logging.getLogger(__name__)


@runtime_checkable
class CrossMarketView(Protocol):
    """Protocol for viewing multi-asset market states."""

    @property
    def symbols(self) -> tuple[str, ...]:
        """Return all registered symbols."""
        ...

    def snapshot(self, symbol: str, depth: int = 5) -> BookSnapshot:
        """Fetch depth-K snapshot for a symbol."""
        ...

    def best_bid(self, symbol: str) -> int | None:
        """Fetch best bid price for a symbol."""
        ...

    def best_ask(self, symbol: str) -> int | None:
        """Fetch best ask price for a symbol."""
        ...

    def mid_price(self, symbol: str) -> float | None:
        """Fetch current mid price for a symbol."""
        ...

    def spread(self, symbol: str) -> int | None:
        """Fetch bid-ask spread in ticks for a symbol."""
        ...

    def imbalance(self, symbol: str, depth: int = 1) -> float:
        """Compute depth-K imbalance: (Q_bid - Q_ask)/(Q_bid + Q_ask)."""
        ...

    def position(self, agent_id: int, symbol: str) -> int:
        """Return agent's net position in a symbol."""
        ...

    def cash(self, agent_id: int) -> float:
        """Return agent's current cash balance."""
        ...

    def resting_orders(self, agent_id: int, symbol: str) -> tuple[int, ...]:
        """Return IDs of resting orders for an agent in a symbol."""
        ...


@runtime_checkable
class CrossMarketAgent(Protocol):
    """An agent interacting with multiple symbols in the CrossMarket."""

    agent_id: int

    def wake(self, ts: int, view: CrossMarketView) -> list[Action]:
        """Produce actions across one or multiple symbols upon waking."""
        ...


@dataclass(frozen=True)
class CrossMarketConfig:
    """Configuration for multi-asset cross-market simulation."""

    duration_ns: int = 60 * 1_000_000_000  # 1 minute
    symbols: tuple[str, ...] = ("LEAD", "LAG")
    initial_prices: dict[str, int] = field(default_factory=lambda: {"LEAD": 10000, "LAG": 10000})
    seed: int = 42
    lead_lag_beta: float = 1.0  # LAG price drift sensitivity to LEAD
    lag_delay_ns: int = 50_000_000  # 50 ms delay for information transmission


@dataclass
class CrossMarketResult:
    """Results and statistics collected from a multi-asset simulation run."""

    duration_ns: int
    trades_per_symbol: dict[str, list[Fill]]
    final_mid_prices: dict[str, float | None]
    agent_positions: dict[int, dict[str, int]]
    agent_cash: dict[int, float]


class CrossMarket:
    """Discrete-event simulation engine managing multiple correlated order books."""

    def __init__(
        self,
        config: CrossMarketConfig,
        book_factory: Callable[[], OrderBook] | None = None,
    ) -> None:
        """Initialize CrossMarket with configuration and books."""
        self._cfg = config
        self._clock = Clock()
        self._queue = WakeQueue()
        self._streams = AgentStreams(config.seed)
        self._multi_book = MultiOrderBook(
            symbols=config.symbols,
            book_factory=book_factory or PythonOrderBook,
        )

        self._agents: dict[int, CrossMarketAgent] = {}
        self._wake_intervals: dict[int, int] = {}

        # Bookkeeping per agent: agent_id -> symbol -> position
        self._positions: dict[int, dict[str, int]] = {}
        # Cash per agent: agent_id -> cash
        self._cash: dict[int, float] = {}
        # Resting orders: agent_id -> symbol -> set of order_id
        self._resting: dict[int, dict[str, set[int]]] = {}
        # Mapping order_id -> (agent_id, symbol)
        self._order_meta: dict[int, tuple[int, str]] = {}
        self._next_order_id = 1

        self._trades: dict[str, list[Fill]] = {s: [] for s in config.symbols}
        self._price_drift: dict[str, int] = {s: 0 for s in config.symbols}

    @property
    def symbols(self) -> tuple[str, ...]:
        """Return registered symbols."""
        return self._cfg.symbols

    def add_agent(self, agent: CrossMarketAgent, wake_interval_ns: int = 100_000_000) -> None:
        """Register a multi-asset agent and schedule its initial wake."""
        aid = agent.agent_id
        self._agents[aid] = agent
        self._wake_intervals[aid] = wake_interval_ns
        self._positions[aid] = {s: 0 for s in self._cfg.symbols}
        self._cash[aid] = 0.0
        self._resting[aid] = {s: set() for s in self._cfg.symbols}
        self._queue.push(0, aid)

    def snapshot(self, symbol: str, depth: int = 5) -> BookSnapshot:
        """Fetch depth-K snapshot for a symbol."""
        return self._multi_book.snapshot(symbol, depth)

    def best_bid(self, symbol: str) -> int | None:
        """Fetch best bid for a symbol."""
        return self._multi_book.best_bid(symbol)

    def best_ask(self, symbol: str) -> int | None:
        """Fetch best ask for a symbol."""
        return self._multi_book.best_ask(symbol)

    def mid_price(self, symbol: str) -> float | None:
        """Fetch current mid price for a symbol."""
        return self._multi_book.mid_price(symbol)

    def spread(self, symbol: str) -> int | None:
        """Fetch bid-ask spread in ticks for a symbol."""
        return self._multi_book.spread(symbol)

    def imbalance(self, symbol: str, depth: int = 1) -> float:
        """Compute depth-K imbalance for a symbol."""
        return self._multi_book.imbalance(symbol, depth)

    def position(self, agent_id: int, symbol: str) -> int:
        """Return agent's net position in a symbol."""
        return self._positions.get(agent_id, {}).get(symbol, 0)

    def cash(self, agent_id: int) -> float:
        """Return agent's cash balance."""
        return self._cash.get(agent_id, 0.0)

    def resting_orders(self, agent_id: int, symbol: str) -> tuple[int, ...]:
        """Return IDs of resting orders for an agent in a symbol."""
        return tuple(sorted(self._resting.get(agent_id, {}).get(symbol, set())))

    def seed_initial_liquidity(self, depth_per_side: int = 5, qty_per_level: int = 100) -> None:
        """Seed initial resting bids and asks around initial prices."""
        for sym in self._cfg.symbols:
            base_p = self._cfg.initial_prices.get(sym, 10000)
            for i in range(1, depth_per_side + 1):
                # Bid
                oid = self._next_order_id
                self._next_order_id += 1
                self._multi_book.submit(
                    sym,
                    Order(
                        order_id=oid,
                        agent_id=0,
                        side=Side.BUY,
                        type=OrderType.LIMIT,
                        price=base_p - i * 2,
                        qty=qty_per_level,
                        ts=0,
                    ),
                )
                # Ask
                oid2 = self._next_order_id
                self._next_order_id += 1
                self._multi_book.submit(
                    sym,
                    Order(
                        order_id=oid2,
                        agent_id=0,
                        side=Side.SELL,
                        type=OrderType.LIMIT,
                        price=base_p + i * 2,
                        qty=qty_per_level,
                        ts=0,
                    ),
                )

    def step(self) -> bool:
        """Advance simulation by waking the next scheduled agent."""
        if not self._queue:
            return False

        ts, aid = self._queue.pop()
        if ts > self._cfg.duration_ns:
            return False

        self._clock.advance_to(ts)
        agent = self._agents[aid]
        actions = agent.wake(ts, self)

        for act in actions:
            self._apply_action(aid, act, ts)

        # Reschedule agent
        interval = self._wake_intervals.get(aid, 100_000_000)
        self._queue.push(ts + interval, aid)
        return True

    def _apply_action(self, agent_id: int, act: Action, ts: int) -> None:
        """Execute agent action on the targeted symbol book."""
        if isinstance(act, Submit):
            sym = act.symbol or self._cfg.symbols[0]
            oid = self._next_order_id
            self._next_order_id += 1
            order = Order(
                order_id=oid,
                agent_id=agent_id,
                side=act.side,
                type=act.order_type,
                price=act.price,
                qty=act.qty,
                ts=ts,
            )
            self._order_meta[oid] = (agent_id, sym)
            fills = self._multi_book.submit(sym, order)
            self._trades[sym].extend(fills)

            filled_qty = 0
            for f in fills:
                filled_qty += f.qty
                # Trade updates cash and positions
                taker_sym = sym
                taker_cost = f.price * f.qty
                if act.side is Side.BUY:
                    self._positions[agent_id][taker_sym] += f.qty
                    self._cash[agent_id] -= taker_cost
                else:
                    self._positions[agent_id][taker_sym] -= f.qty
                    self._cash[agent_id] += taker_cost

                # Maker updates if maker belongs to our registered agents
                if f.maker_order_id in self._order_meta:
                    maker_aid, maker_sym = self._order_meta[f.maker_order_id]
                    if maker_aid in self._positions:
                        if act.side is Side.BUY:
                            # Maker was selling
                            self._positions[maker_aid][maker_sym] -= f.qty
                            self._cash[maker_aid] += taker_cost
                        else:
                            # Maker was buying
                            self._positions[maker_aid][maker_sym] += f.qty
                            self._cash[maker_aid] -= taker_cost

            # If limit order was not completely filled, track resting
            if act.order_type is OrderType.LIMIT and filled_qty < act.qty:
                self._resting[agent_id][sym].add(oid)

        elif isinstance(act, Cancel):
            sym = act.symbol or self._cfg.symbols[0]
            if self._multi_book.cancel(sym, act.order_id, ts) and (
                act.order_id in self._resting[agent_id][sym]
            ):
                self._resting[agent_id][sym].remove(act.order_id)

    def run(self) -> CrossMarketResult:
        """Run simulation to configured duration and return summary."""
        while self.step():
            pass

        return CrossMarketResult(
            duration_ns=self._clock.now,
            trades_per_symbol={s: list(trades) for s, trades in self._trades.items()},
            final_mid_prices={s: self.mid_price(s) for s in self._cfg.symbols},
            agent_positions={aid: dict(pos) for aid, pos in self._positions.items()},
            agent_cash=dict(self._cash),
        )
