"""The market: the one place where agents, the engine, and time meet.

``Market.run`` is the simulation loop. It repeatedly takes the agent whose turn is next, lets it
look at the market and answer with actions, and applies those actions to the matching engine. The
market is the only component that stamps order ids, agent ids and timestamps, which is why a run is
a pure function of ``(config, seed)``: nothing in it depends on the wall clock, on thread timing or
on global random state.

The market also *is* the ``MarketView`` agents look through (see ``lobx.sim.agents.base``). It keeps
the bookkeeping that the engine does not: each agent's position, the last trade price, and which
resting orders belong to whom.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from lobx.engine.book_py import PythonOrderBook
from lobx.engine.event_log import EventLog
from lobx.engine.logged_book import LoggedBook
from lobx.engine.protocol import OrderBook
from lobx.errors import InvalidOrderError, SimulationError
from lobx.sim.actions import Action, Cancel
from lobx.sim.agents.base import Agent
from lobx.sim.clock import Clock, WakeQueue
from lobx.sim.population import MarketConfig, build_background_agents
from lobx.sim.rng import AgentStreams
from lobx.types import BookSnapshot, Fill, Order, Side

logger = logging.getLogger(__name__)

_FINAL_SNAPSHOT_DEPTH = 10


@dataclass(frozen=True, slots=True)
class Sample:
    """The top of the book at one sampling instant (``None`` when that side was empty)."""

    ts: int
    best_bid: int | None
    best_ask: int | None

    @property
    def spread(self) -> int | None:
        """Return best ask minus best bid in ticks, or ``None`` if either side was empty."""
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    @property
    def mid2(self) -> int | None:
        """Return best bid plus best ask (twice the mid-price), or ``None`` if a side was empty."""
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_bid + self.best_ask


@dataclass(frozen=True, slots=True)
class RunResult:
    """Everything a finished run hands back."""

    config: MarketConfig
    log: EventLog
    final_snapshot: BookSnapshot
    samples: tuple[Sample, ...]
    trades: int
    self_trades: int
    # The largest absolute position each agent held at any moment, keyed by agent id.
    peak_abs_position: dict[int, int]
    # What kind of agent each id was, e.g. {0: "MarketMaker", 2: "NoiseTrader"}.
    roles: dict[int, str]


class Market:
    """A simulated exchange with agents trading on it. Build it, call ``run`` once."""

    def __init__(
        self,
        config: MarketConfig,
        agents: Sequence[Agent] | None = None,
        book: OrderBook | None = None,
    ) -> None:
        """Set up a run.

        ``agents`` defaults to the background cast described by ``config``. Pass your own list to
        add a strategy under test (include the background agents too, if you want them). ``book``
        defaults to the Python reference engine; pass another ``OrderBook`` to run on it instead.
        """
        self.config = config
        self._agents: dict[int, Agent] = {}
        for agent in agents if agents is not None else build_background_agents(config):
            if agent.agent_id in self._agents:
                raise SimulationError(f"two agents share the id {agent.agent_id}")
            self._agents[agent.agent_id] = agent
        self.book = LoggedBook(
            book if book is not None else PythonOrderBook(),
            EventLog(seed=config.seed, config=config.as_dict()),
        )
        self.book.add_fill_listener(self._on_fill)
        self._streams = AgentStreams(config.seed)
        self._clock = Clock()
        self._wakes = WakeQueue()
        # Every agent starts flat, so an agent that never trades still shows up in the results.
        self._positions: dict[int, int] = dict.fromkeys(self._agents, 0)
        self._peak_abs_position: dict[int, int] = dict.fromkeys(self._agents, 0)
        self._last_trade: int | None = None
        self._trades = 0
        self._self_trades = 0
        self._next_order_id = 1
        self._samples: list[Sample] = []
        self._ran = False

    # ------------------------------------------------------------------ MarketView (read-only)

    @property
    def ts(self) -> int:
        """Return the current simulation time in nanoseconds."""
        return self._clock.now

    def best_bid(self) -> int | None:
        """Return the highest resting buy price, or ``None`` if there are no bids."""
        return self.book.best_bid()

    def best_ask(self) -> int | None:
        """Return the lowest resting sell price, or ``None`` if there are no asks."""
        return self.book.best_ask()

    def mid2(self) -> int | None:
        """Return best bid plus best ask (twice the mid-price), or ``None`` if a side is empty."""
        bid, ask = self.book.best_bid(), self.book.best_ask()
        return None if bid is None or ask is None else bid + ask

    def last_trade_price(self) -> int | None:
        """Return the price of the most recent trade, or ``None`` before the first trade."""
        return self._last_trade

    def reference_price(self) -> int:
        """Return the mid-price if both sides exist, else the last trade, else the opening price."""
        mid2 = self.mid2()
        if mid2 is not None:
            return mid2 // 2
        return self._last_trade if self._last_trade is not None else self.config.opening_price

    def snapshot(self, depth: int) -> BookSnapshot:
        """Return up to ``depth`` aggregated price levels per side, best first."""
        return self.book.snapshot(depth)

    def position(self, agent_id: int) -> int:
        """Return how many shares ``agent_id`` is long (positive) or short (negative)."""
        return self._positions.get(agent_id, 0)

    def open_orders(self, agent_id: int) -> tuple[int, ...]:
        """Return the ids of ``agent_id``'s resting orders, oldest first."""
        return self.book.tracker.open_order_ids(agent_id)

    # ------------------------------------------------------------------------------ the loop

    def run(self) -> RunResult:
        """Simulate until ``config.duration_ns`` and return the result. Can only be called once."""
        if self._ran:
            raise SimulationError("a Market can only be run once; build a new one to run again")
        self._ran = True
        duration = self.config.duration_ns
        interval = self.config.sample_interval_ns

        for agent in self._agents.values():
            first = agent.first_wake(self._streams.for_agent(agent.agent_id))
            if first < 0:
                raise SimulationError(f"agent {agent.agent_id} asked for a first wake before 0")
            self._wakes.push(first, agent.agent_id)

        next_sample = interval
        while self._wakes:
            ts, agent_id = self._wakes.pop()
            if ts > duration:
                break
            # Record the book at each sampling instant that passed. The book has not changed since
            # the previous event, so what we read is exactly its state at that instant.
            while next_sample <= ts:
                self._take_sample(next_sample)
                next_sample += interval
            self._clock.advance_to(ts)
            agent = self._agents[agent_id]
            rng = self._streams.for_agent(agent_id)
            for action in agent.wake(ts, self, rng):
                self._apply(agent_id, action, ts)
            following = agent.next_wake(ts, rng)
            if following is not None:
                if following <= ts:
                    raise SimulationError(
                        f"agent {agent_id} asked to wake at {following}, "
                        f"which is not after the current time {ts}"
                    )
                self._wakes.push(following, agent_id)
        while next_sample <= duration:
            self._take_sample(next_sample)
            next_sample += interval

        return RunResult(
            config=self.config,
            log=self.book.log,
            final_snapshot=self.book.snapshot(_FINAL_SNAPSHOT_DEPTH),
            samples=tuple(self._samples),
            trades=self._trades,
            self_trades=self._self_trades,
            peak_abs_position=dict(self._peak_abs_position),
            roles={agent_id: type(a).__name__ for agent_id, a in self._agents.items()},
        )

    # ------------------------------------------------------------------------------ internals

    def _apply(self, agent_id: int, action: Action, ts: int) -> None:
        """Carry out one action on behalf of ``agent_id`` at time ``ts``."""
        if isinstance(action, Cancel):
            owner = self.book.tracker.agent_of(action.order_id)
            if owner is not None and owner != agent_id:
                raise SimulationError(
                    f"agent {agent_id} tried to cancel order {action.order_id}, "
                    f"which belongs to agent {owner}"
                )
            self.book.cancel(action.order_id, ts)
            return
        order = Order(
            order_id=self._next_order_id,
            agent_id=agent_id,
            side=action.side,
            type=action.order_type,
            price=action.price,
            qty=action.qty,
            ts=ts,
        )
        self._next_order_id += 1
        try:
            self.book.submit(order)
        except InvalidOrderError as error:
            raise SimulationError(
                f"agent {agent_id} sent an order the engine rejected: {error}"
            ) from error

    def _on_fill(self, fill: Fill, maker_agent: int, taker_agent: int) -> None:
        """Update positions and trade statistics for one trade."""
        buyer, seller = (
            (taker_agent, maker_agent)
            if fill.taker_side is Side.BUY
            else (maker_agent, taker_agent)
        )
        self._move_position(buyer, fill.qty)
        self._move_position(seller, -fill.qty)
        self._last_trade = fill.price
        self._trades += 1
        if buyer == seller:
            self._self_trades += 1
            logger.debug("self-trade by agent %d: %d at %d", buyer, fill.qty, fill.price)

    def _move_position(self, agent_id: int, change: int) -> None:
        position = self._positions.get(agent_id, 0) + change
        self._positions[agent_id] = position
        if abs(position) > self._peak_abs_position.get(agent_id, 0):
            self._peak_abs_position[agent_id] = abs(position)

    def _take_sample(self, ts: int) -> None:
        self._samples.append(Sample(ts, self.book.best_bid(), self.book.best_ask()))
