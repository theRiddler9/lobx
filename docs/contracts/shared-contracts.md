# Shared contracts

These are the types and rules that cross component boundaries. A change here goes through its own
`contract/<topic>` branch and merges to `main` before dependent work continues.

## Global rules

- **Prices are integers in ticks.** No floats for prices or quantities, ever.
- **Time is an integer in nanoseconds** since simulation start.
- **Randomness is explicit.** Every random draw comes from a seeded generator passed in as an
  argument. No global `random` or `np.random`.
- **A run is a pure function of `(config, seed)`.** Same inputs give byte-identical event logs.

## Types (`src/lobx/types.py`)

| Type | Fields | Meaning |
|---|---|---|
| `Side` | `BUY`, `SELL` | Which side of the book an order is on. |
| `OrderType` | `LIMIT`, `MARKET` | Limit orders have a price and may rest. Market orders have no price and never rest. |
| `Order` | `order_id`, `agent_id`, `side`, `type`, `price`, `qty`, `ts` | An instruction to the engine. `price` is `None` for market orders. |
| `Fill` | `maker_order_id`, `taker_order_id`, `taker_side`, `price`, `qty`, `ts` | One trade between a resting (maker) order and an incoming (taker) order. |
| `BookSnapshot` | `ts`, `bids`, `asks` | Aggregated depth. Each level is `(price, total_qty)`, best level first. |

All of them are frozen dataclasses: plain data, no behavior, no validation. Validation lives in the
engine, so the Python and C++ engines can share one set of rules.

`Fill.taker_side` is an addition to the first draft of the contract in `AGENTS.md`. The trade-flow
imbalance feature in Phase 5 needs to know who the aggressor was, and recovering it from order ids
would force the ML code to reach into the event log.

## The `OrderBook` protocol (`src/lobx/engine/protocol.py`)

```python
class OrderBook(Protocol):
    def submit(self, order: Order) -> list[Fill]: ...
    def cancel(self, order_id: int, ts: int) -> bool: ...
    def snapshot(self, depth: int) -> BookSnapshot: ...
    def best_bid(self) -> int | None: ...
    def best_ask(self) -> int | None: ...
```

Both the Python reference engine and the C++ engine implement it. They must produce identical fills
and snapshots for any order stream; Phase 3 enforces this with a differential test.

## Matching rules

These are the standard exchange rules. Each is a decision the C++ engine must reproduce exactly.

1. **Price-time priority.** Better prices match first. At the same price, the earlier order matches
   first.
2. **Trades execute at the resting order's price**, not the incoming order's price.
3. **Limit orders** match against the opposite side while the prices cross (a buy matches asks at or
   below its limit; a sell matches bids at or above its limit). Any unfilled remainder rests in the
   book at its limit price.
4. **Market orders** match against the opposite side until filled or the side is empty. Any
   unfilled remainder is **discarded** and never rests.
5. **Partial fills** are allowed. One incoming order can produce several fills, returned in
   execution order.
6. **Self-trades are allowed.** The engine does not look at `agent_id` when matching. Phase 2 logs
   them.
7. **No hidden or iceberg orders.** The whole quantity of a resting order is visible.
8. **Cancel** removes a resting order and returns `True`. It returns `False` if the order is
   unknown, already fully filled, or already cancelled. A cancel never raises for an unknown id.

## Validation rules

The engine raises a specific error (see `src/lobx/errors.py`) instead of silently accepting bad input.

| Rule | Error |
|---|---|
| `qty` must be an integer greater than 0 | `InvalidOrderError` |
| a `LIMIT` order must have an integer `price` greater than 0 | `InvalidOrderError` |
| a `MARKET` order must have `price` equal to `None` | `InvalidOrderError` |
| `order_id` must not belong to an order that is currently resting | `DuplicateOrderIdError` |
| `ts` must not be earlier than the `ts` of the previous call to `submit` or `cancel` | `InvalidOrderError` |
| `snapshot(depth)` needs `depth` greater than 0 | `ValueError` |

Order ids should be unique for the whole run. The market allocates them, so the engine only checks
the cheap case (ids currently resting).

## Snapshot rules

- `snapshot(depth)` returns at most `depth` price levels per side.
- Bids are sorted from highest to lowest price, asks from lowest to highest.
- Quantity at a level is the sum of the remaining quantity of all resting orders at that price.
- Empty levels never appear.
- The snapshot `ts` is the `ts` of the last call to `submit` or `cancel` (0 before the first call).

## Invariants every engine must keep

- The book is never crossed: best bid is strictly lower than best ask whenever both exist.
- Remaining quantity of every resting order is greater than 0.
- Total quantity filled for an incoming order never exceeds its `qty`.
- Sum of fill quantities on the maker and taker sides is equal for every trade.

## Contracts added in later phases

These are listed in `AGENTS.md` and get their code stubs when the phase that needs them starts:

- `Event` types and the event log (Phase 2)
- `Agent`, `MarketView`, `Action` (Phase 2)
- `ParentOrder`, `ExecutionStrategy` (Phase 4)
- `extract_features` (Phase 5) with the rule that features at time `t` use only data with
  timestamp at or before `t`
- `implementation_shortfall_bps` (Phase 4)
