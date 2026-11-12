from pathlib import Path

import pytest

from lobx.data.lobster import generate_lobster_sample
from lobx.engine.book_py import PythonOrderBook
from lobx.sim.actions import Action, Submit
from lobx.sim.agents.base import MarketView
from lobx.sim.replay_market import ReplayMarket
from lobx.strategies.base import ParentOrder
from lobx.strategies.twap import TwapStrategy
from lobx.types import OrderType, Side


@pytest.fixture
def lobster_files(tmp_path: Path) -> tuple[Path, Path]:
    msg_path = tmp_path / "test_messages.csv"
    book_path = tmp_path / "test_orderbook.csv"
    generate_lobster_sample(
        msg_path,
        book_path,
        num_events=60,
        start_time_s=34200.0,
        seed_price=10_000,
    )
    return msg_path, book_path


def test_replay_market_without_strategy(lobster_files: tuple[Path, Path]) -> None:
    msg_path, book_path = lobster_files
    market = ReplayMarket(message_path=msg_path, book=PythonOrderBook())
    result = market.run()

    assert result.events_replayed == 60
    assert result.total_market_volume > 0
    assert result.executed == 0
    assert result.shortfall_bps is None


def test_replay_market_with_twap_strategy(lobster_files: tuple[Path, Path]) -> None:
    msg_path, book_path = lobster_files
    start_ns = 34_200 * 1_000_000_000
    end_ns = start_ns + 5 * 1_000_000_000  # 5 seconds execution window

    parent = ParentOrder(
        agent_id=42,
        side=Side.SELL,
        qty=200,
        start_ns=start_ns,
        end_ns=end_ns,
    )
    strategy = TwapStrategy(num_slices=2)

    market = ReplayMarket(
        message_path=msg_path,
        strategy=strategy,
        parent=parent,
        wake_interval_ns=1_000_000_000,
        book=PythonOrderBook(),
    )
    result = market.run()

    assert result.events_replayed == 60
    assert result.arrival_mid2 > 0
    # TWAP submitted orders against historical stream
    assert len(result.fills) > 0
    assert result.executed > 0


def test_replay_market_queue_priority_enforced(tmp_path: Path) -> None:
    msg_path = tmp_path / "queue_test_msg.csv"
    # Construct exact message stream:
    # 1. Historical resting sell limit order: 500 shares at 10020
    # 2. Strategy wakes and places sell limit order: 100 shares at 10020 (qty_ahead is 500)
    # 3. Aggressive buy trade arrives: 300 shares at 10020 -> Should NOT fill strategy
    # 4. Another trade arrives: 300 shares at 10020 -> Consumes 200 ahead, fills 100 of strategy!
    lines = [
        "34200.000000000,1,101,500,10020,-1\n",  # Resting sell 500 @ 10020
        "34200.100000000,1,102,500,10000,1\n",  # Resting buy 500 @ 10000
        "34200.200000000,4,101,300,10020,-1\n",  # Trade 300 @ 10020 (aggressor buy)
        "34200.300000000,4,101,300,10020,-1\n",  # Trade 300 @ 10020 (consumes 200 ahead, 100 of us)
    ]
    msg_path.write_text("".join(lines), encoding="utf-8")

    start_ns = 34_200 * 1_000_000_000
    end_ns = start_ns + 1_000_000_000

    parent = ParentOrder(
        agent_id=1,
        side=Side.SELL,
        qty=100,
        start_ns=start_ns,
        end_ns=end_ns,
    )

    # Strategy that places a passive limit order at best ask (10020)
    class PassiveQuoter:
        def reset(self, parent: ParentOrder) -> None:
            self._parent = parent
            self._done = False

        def step(self, ts: int, view: MarketView) -> list[Action]:
            if not self._done:
                self._done = True
                return [
                    Submit(
                        side=Side.SELL,
                        order_type=OrderType.LIMIT,
                        price=10020,
                        qty=100,
                    )
                ]
            return []

    market = ReplayMarket(
        message_path=msg_path,
        strategy=PassiveQuoter(),
        parent=parent,
        wake_interval_ns=50_000_000,
        book=PythonOrderBook(),
    )
    result = market.run()

    # The 100 shares should be completely filled after the 2nd trade!
    assert result.executed == 100
    assert len(result.fills) == 1
    assert result.fills[0].price == 10020
    assert result.fills[0].qty == 100
    assert result.shortfall_bps is not None
