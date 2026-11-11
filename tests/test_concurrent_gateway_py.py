"""Python tests for the C++ concurrent order book and asynchronous gateway."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from lobx.engine.native import native_available

pytestmark = pytest.mark.skipif(not native_available(), reason="C++ engine not built")


def test_concurrent_order_book_python() -> None:
    from lobx import _native

    book = _native.NativeConcurrentBook()
    assert book.resting_orders() == 0
    assert book.best_bid() is None
    assert book.best_ask() is None

    # Submit resting buy
    fills1 = book.submit(1, 10, 0, 0, 100, 50, 1000)
    assert len(fills1) == 0
    assert book.best_bid() == 100
    assert book.resting_orders() == 1
    assert book.total_submits() == 1

    # Crossing sell
    fills2 = book.submit(2, 20, 1, 0, 100, 30, 2000)
    assert len(fills2) == 1
    assert fills2[0][4] == 30  # qty
    assert book.total_fills() == 1
    assert book.total_traded_volume() == 30

    # Cancel
    assert book.cancel(1, 3000)
    assert book.resting_orders() == 0
    assert book.total_cancels() == 1


def test_exchange_gateway_python() -> None:
    from lobx import _native

    gateway = _native.NativeGateway(1024, 1024)
    assert not gateway.is_running()
    gateway.start()
    assert gateway.is_running()

    # Submit resting limit buy
    seq1 = gateway.submit_order(10, 1, 0, 0, 200, 50, 1000)
    assert seq1 == 1
    assert gateway.wait_until_processed(seq1, 2000)

    # Submit crossing limit sell
    seq2 = gateway.submit_order(11, 2, 1, 0, 200, 20, 2000)
    assert seq2 == 2
    assert gateway.wait_until_processed(seq2, 2000)

    # Read fill from egress
    event = gateway.poll_event()
    assert event is not None
    seq_id, ev_type, order_id, agent_id, price, qty, ts, success = event
    assert seq_id == 2
    assert ev_type == 0  # Fill
    assert price == 200
    assert qty == 20
    assert success is True

    # Multi-threaded submissions from Python ThreadPoolExecutor
    def submit_batch(client_id: int) -> int:
        last_seq = 0
        for i in range(25):
            oid = client_id * 1000 + i + 100
            last_seq = gateway.submit_order(oid, client_id, 0, 0, 190, 5, oid * 10)
        return last_seq

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(submit_batch, c) for c in range(1, 5)]
        max_seq = max(f.result() for f in futures)

    assert gateway.wait_until_processed(max_seq, 5000)
    assert gateway.total_requests_processed() >= max_seq

    gateway.stop()
    assert not gateway.is_running()
