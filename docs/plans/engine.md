# Engine plan

Owner area: `src/lobx/engine/`, `cpp/`.

## Goal

A correct limit order book with price-time priority, implemented twice (readable Python reference,
fast C++) behind the `OrderBook` protocol, and proven equal by a differential test.

## Data structures (Python reference)

- `PriceLevel`: a FIFO queue of resting orders at one price plus a running total quantity.
- `BookSide`: price levels keyed by price. Bids iterate from high to low, asks from low to high.
- `PythonOrderBook`: two sides, a dict from `order_id` to the resting order, the last timestamp.

Why these: dict plus sorted price keys is easy to read and explain. The C++ engine uses an ordered
map per side and an order-id index for O(1) cancel; the Python version stays simple on purpose.

## Phases

- Phase 1: Python engine, unit tests, hypothesis property tests.
- Phase 2: append-only event log and replay.
- Phase 3: C++ engine, pybind11 bindings, differential test, benchmark.

## Test strategy

- Unit tests: one scenario per rule in `docs/contracts/shared-contracts.md`.
- Property tests: random order streams never break the invariants (book not crossed, positive
  remaining quantity, fills never exceed order quantity).
- Differential test (Phase 3): same stream into both engines, identical fills and snapshots.

## Risks

- Subtle rule drift between the two engines. Mitigation: rules are written once in the contract and
  the differential test runs over a million random orders.
