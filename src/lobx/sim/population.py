"""The settings of a simulated market and the cast of background agents it starts with.

``MarketConfig`` is everything a run depends on besides its random seed. It is written into the
header of the event log, so a log says exactly how it was produced.

The background agents are the "ambient" market: market makers that keep quotes in the book, noise
traders that supply random order flow, and momentum traders that chase trends. Strategies under
test (the execution agents of phase 4) are added on top of this cast, never mixed into it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from lobx.sim.agents.base import Agent
from lobx.sim.agents.market_maker import MarketMaker
from lobx.sim.agents.momentum import MomentumTrader
from lobx.sim.agents.noise import NoiseTrader
from lobx.sim.clock import NS_PER_S


@dataclass(frozen=True, slots=True)
class MarketConfig:
    """Every setting of a run except the seed. Times are in nanoseconds, prices in ticks."""

    seed: int = 0
    duration_ns: int = 600 * NS_PER_S
    opening_price: int = 10_000
    num_market_makers: int = 2
    num_noise_traders: int = 100
    num_momentum_traders: int = 10
    market_maker_interval_ns: int = 1 * NS_PER_S
    noise_interval_ns: int = 5 * NS_PER_S
    momentum_interval_ns: int = 2 * NS_PER_S
    sample_interval_ns: int = 1 * NS_PER_S
    # How deep and how thick the market makers' quotes are. Thin quotes are what let the price
    # move: with 10 shares at each of 3 levels per maker, the price simply never budged.
    market_maker_quote_qty: int = 3
    market_maker_levels: int = 2
    # How aggressive the noise traders are: the chance that an order is a market order, and the
    # scale of the heavy-tailed size those market orders are drawn from.
    noise_p_market: float = 0.3
    noise_market_qty_scale: float = 5.0

    def __post_init__(self) -> None:
        """Reject settings that cannot describe a run, with a message naming the field."""
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {self.seed!r}")
        for name in (
            "duration_ns",
            "opening_price",
            "market_maker_interval_ns",
            "noise_interval_ns",
            "momentum_interval_ns",
            "sample_interval_ns",
            "market_maker_quote_qty",
            "market_maker_levels",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer of at least 1, got {value!r}")
        if isinstance(self.noise_p_market, bool) or not 0.0 <= self.noise_p_market <= 1.0:
            raise ValueError(f"noise_p_market must be between 0 and 1, got {self.noise_p_market!r}")
        if isinstance(self.noise_market_qty_scale, bool) or self.noise_market_qty_scale <= 0:
            raise ValueError(
                f"noise_market_qty_scale must be above 0, got {self.noise_market_qty_scale!r}"
            )
        for name in ("num_market_makers", "num_noise_traders", "num_momentum_traders"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")

    def as_dict(self) -> dict[str, int | float]:
        """Return the settings as plain numbers, ready for the event log header."""
        return asdict(self)


def build_background_agents(config: MarketConfig) -> list[Agent]:
    """Create the background cast for ``config``.

    Agent ids are handed out in a fixed order: market makers first, then noise traders, then
    momentum traders. A run is reproducible only if the same config always gives the same ids,
    because each agent's random stream is derived from its id.

    Each market maker quotes a little wider than the one before it (1 tick from the price, then 2,
    then 3...), so the book has depth at several distances from the price instead of one stack of
    identical orders.
    """
    agents: list[Agent] = []
    next_id = 0
    for index in range(config.num_market_makers):
        agents.append(
            MarketMaker(
                next_id,
                config.market_maker_interval_ns,
                half_spread_ticks=1 + index,
                levels=config.market_maker_levels,
                quote_qty=config.market_maker_quote_qty,
            )
        )
        next_id += 1
    for _ in range(config.num_noise_traders):
        agents.append(
            NoiseTrader(
                next_id,
                config.noise_interval_ns,
                p_market=config.noise_p_market,
                market_qty_scale=config.noise_market_qty_scale,
            )
        )
        next_id += 1
    for _ in range(config.num_momentum_traders):
        agents.append(MomentumTrader(next_id, config.momentum_interval_ns))
        next_id += 1
    return agents
