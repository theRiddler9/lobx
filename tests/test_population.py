import json

import pytest

from lobx.sim.agents.base import Agent
from lobx.sim.agents.market_maker import MarketMaker
from lobx.sim.agents.momentum import MomentumTrader
from lobx.sim.agents.noise import NoiseTrader
from lobx.sim.population import MarketConfig, build_background_agents


def test_default_config_is_valid() -> None:
    config = MarketConfig()
    assert config.seed == 0
    assert config.duration_ns == 600_000_000_000


def test_config_is_frozen() -> None:
    with pytest.raises(AttributeError):
        MarketConfig().seed = 5  # type: ignore[misc]


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_bad_seed_is_rejected(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        MarketConfig(seed=seed)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field",
    [
        "duration_ns",
        "opening_price",
        "market_maker_interval_ns",
        "noise_interval_ns",
        "momentum_interval_ns",
        "sample_interval_ns",
        "market_maker_quote_qty",
        "market_maker_levels",
    ],
)
@pytest.mark.parametrize("value", [0, -5, True, 2.5])
def test_non_positive_or_non_integer_settings_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        MarketConfig(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field", ["num_market_makers", "num_noise_traders", "num_momentum_traders"]
)
def test_negative_agent_counts_are_rejected(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        MarketConfig(**{field: -1})


@pytest.mark.parametrize("value", [-0.1, 1.5, True])
def test_noise_p_market_must_be_a_probability(value: object) -> None:
    with pytest.raises(ValueError, match="noise_p_market"):
        MarketConfig(noise_p_market=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0.0, -1.0, True])
def test_noise_market_qty_scale_must_be_positive(value: object) -> None:
    with pytest.raises(ValueError, match="noise_market_qty_scale"):
        MarketConfig(noise_market_qty_scale=value)  # type: ignore[arg-type]


def test_aggressiveness_and_depth_settings_reach_the_agents() -> None:
    config = MarketConfig(
        num_market_makers=1,
        num_noise_traders=1,
        num_momentum_traders=0,
        market_maker_quote_qty=7,
        market_maker_levels=4,
        noise_p_market=0.5,
        noise_market_qty_scale=9.0,
    )
    maker, noise = build_background_agents(config)
    assert isinstance(maker, MarketMaker)
    assert (maker.quote_qty, maker.levels) == (7, 4)
    assert isinstance(noise, NoiseTrader)
    assert (noise.p_market, noise.market_qty_scale) == (0.5, 9.0)


def test_zero_agents_of_a_kind_is_allowed() -> None:
    assert build_background_agents(MarketConfig(num_noise_traders=0)) is not None


def test_as_dict_is_plain_json() -> None:
    data = MarketConfig(seed=3).as_dict()
    assert data["seed"] == 3
    assert json.loads(json.dumps(data)) == data


def test_population_counts_types_and_id_order() -> None:
    config = MarketConfig(num_market_makers=2, num_noise_traders=5, num_momentum_traders=3)
    agents = build_background_agents(config)
    assert [a.agent_id for a in agents] == list(range(10))
    kinds = [type(a) for a in agents]
    assert kinds == [MarketMaker] * 2 + [NoiseTrader] * 5 + [MomentumTrader] * 3
    assert all(isinstance(a, Agent) for a in agents)


def test_market_makers_quote_progressively_wider() -> None:
    agents = build_background_agents(MarketConfig(num_market_makers=3, num_noise_traders=0))
    spreads = [a.half_spread_ticks for a in agents if isinstance(a, MarketMaker)]
    assert spreads == [1, 2, 3]


def test_config_intervals_reach_the_agents() -> None:
    config = MarketConfig(
        num_market_makers=1,
        num_noise_traders=1,
        num_momentum_traders=1,
        market_maker_interval_ns=7,
        noise_interval_ns=11,
        momentum_interval_ns=13,
    )
    maker, noise, momentum = build_background_agents(config)
    assert isinstance(maker, MarketMaker) and maker.interval_ns == 7
    assert isinstance(noise, NoiseTrader) and noise.mean_interval_ns == 11
    assert isinstance(momentum, MomentumTrader) and momentum.interval_ns == 13
