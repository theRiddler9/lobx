"""Run an execution strategy inside a simulated market and measure what it cost.

One run = one seed = one market. The strategy is added to the usual background agents as one more
participant; it trades under its own agent id, and every fill that involves that id is collected.
Because the market is a pure function of ``(config, seed)``, running two strategies on the same
seed gives them the same starting market, which is what makes a paired comparison fair.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from lobx.engine.book_py import PythonOrderBook
from lobx.engine.native import NativeOrderBook, native_available
from lobx.engine.protocol import OrderBook
from lobx.eval.metrics import implementation_shortfall_bps
from lobx.sim.agents.base import Agent
from lobx.sim.clock import NS_PER_S
from lobx.sim.market import Market
from lobx.sim.population import MarketConfig, build_background_agents
from lobx.strategies.agent import StrategyAgent
from lobx.strategies.base import ExecutionStrategy, ParentOrder
from lobx.types import Fill, FillEvent, Side

StrategyFactory = Callable[[], ExecutionStrategy]
EngineFactory = Callable[[], OrderBook]
# Builds a passive extra participant (for example a book recorder) given the next free agent id.
ObserverFactory = Callable[[int], Agent]


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """What to execute and when.

    The defaults sell 3,000 shares (about 20% of the market's volume over the window) in 30 slices
    of 100 shares. AGENTS.md mentions 50,000 shares as an example; the simulated market trades only
    about 3,000 shares a minute, so the order is scaled to the market, not the other way round.
    """

    parent_qty: int = 3_000
    side: Side = Side.SELL
    start_ns: int = 60 * NS_PER_S  # warm-up: let the book fill before the order arrives
    window_ns: int = 300 * NS_PER_S
    num_slices: int = 30
    # A strategy that "watches" wakes this often (the ML snapshot interval) and starts this long
    # before the window to build history. The lookback must be a whole number of intervals and
    # longer than the model's 26-snapshot memory (5.2 s).
    watch_interval_ns: int = 200_000_000
    watch_lookback_ns: int = 6 * NS_PER_S

    def market_config(self, seed: int) -> MarketConfig:
        """Return the market configuration for ``seed``: it ends when the parent window does."""
        return MarketConfig(seed=seed, duration_ns=self.start_ns + self.window_ns)

    def parent(self, agent_id: int) -> ParentOrder:
        """Return the parent order, traded under ``agent_id``."""
        return ParentOrder(
            agent_id, self.side, self.parent_qty, self.start_ns, self.start_ns + self.window_ns
        )


@dataclass(frozen=True, slots=True)
class SeedResult:
    """The outcome of one strategy on one seed."""

    seed: int
    executed: int
    arrival_mid2: int
    # ``None`` when the parent order did not complete, so no shortfall could be measured.
    shortfall_bps: float | None


@dataclass(frozen=True, slots=True)
class RunDetail:
    """A ``SeedResult`` together with the strategy's own fills, in execution order."""

    result: SeedResult
    fills: tuple[Fill, ...]


def default_engine() -> EngineFactory:
    """Return the fastest available engine: the C++ one if built, else the Python reference."""
    return NativeOrderBook if native_available() else PythonOrderBook


def run_detailed(
    factory: StrategyFactory,
    seed: int,
    config: EvalConfig | None = None,
    engine: EngineFactory | None = None,
    watching: bool = False,
    observer: ObserverFactory | None = None,
) -> RunDetail:
    """Run a fresh strategy on the market for ``seed``; return its shortfall and its fills.

    ``watching`` strategies wake every ``config.watch_interval_ns`` (see ``EvalConfig``) instead of
    once per slice. ``observer`` adds one extra agent that must trade nothing (the service uses it
    to record the book); it takes the id after the strategy's, and every agent draws from its own
    random stream, so it cannot change the market (tests/test_harness.py checks this).
    """
    config = config or EvalConfig()
    market_config = config.market_config(seed)
    agents = build_background_agents(market_config)
    parent = config.parent(agent_id=len(agents))  # the next free id
    agent = (
        StrategyAgent(
            factory(), parent, config.num_slices, config.watch_interval_ns, config.watch_lookback_ns
        )
        if watching
        else StrategyAgent(factory(), parent, config.num_slices)
    )
    participants: list[Agent] = [*agents, agent]
    if observer is not None:
        participants.append(observer(parent.agent_id + 1))
    market = Market(market_config, participants, book=(engine or default_engine())())

    own_fills: list[Fill] = []

    def collect(fill: Fill, maker_agent: int, taker_agent: int) -> None:
        if parent.agent_id in (maker_agent, taker_agent):
            own_fills.append(fill)

    market.book.add_fill_listener(collect)
    market.run()

    executed = sum(fill.qty for fill in own_fills)
    if agent.arrival_mid2 is None:  # the strategy window never started: a configuration error
        raise RuntimeError(f"seed {seed}: the strategy never woke up")
    shortfall = (
        implementation_shortfall_bps(parent, own_fills, agent.arrival_mid2)
        if executed == parent.qty
        else None
    )
    return RunDetail(SeedResult(seed, executed, agent.arrival_mid2, shortfall), tuple(own_fills))


def run_one_seed(
    factory: StrategyFactory,
    seed: int,
    config: EvalConfig | None = None,
    engine: EngineFactory | None = None,
    watching: bool = False,
) -> SeedResult:
    """Run a fresh strategy on the market for ``seed`` and return its implementation shortfall."""
    return run_detailed(factory, seed, config, engine, watching).result


def run_many(
    factory: StrategyFactory,
    seeds: Iterable[int],
    config: EvalConfig | None = None,
    engine: EngineFactory | None = None,
    watching: bool = False,
) -> list[SeedResult]:
    """Run ``factory``'s strategy on every seed, in order."""
    return [run_one_seed(factory, seed, config, engine, watching) for seed in seeds]


def background_trades(
    seed: int, config: EvalConfig | None = None, engine: EngineFactory | None = None
) -> list[tuple[int, int]]:
    """Return ``(ts, qty)`` of every trade in the market for ``seed`` with no strategy running."""
    config = config or EvalConfig()
    result = Market(config.market_config(seed), book=(engine or default_engine())()).run()
    return [(e.fill.ts, e.fill.qty) for e in result.log if isinstance(e, FillEvent)]
