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

## Events and the event log (Phase 2)

Every run writes an append-only **event log**. It is the single record of what happened: replaying
it alone must rebuild the final book exactly, and later phases read it to build ML features.

### Event types (`src/lobx/types.py`)

| Type | Fields | Meaning |
|---|---|---|
| `SubmitEvent` | `order` | An order the engine accepted. Orders the engine rejects are never logged. |
| `CancelEvent` | `order_id`, `ts`, `cancelled` | A cancel request and its outcome (`False` if the order was already gone). |
| `FillEvent` | `fill`, `self_trade` | One trade. `self_trade` is `True` when maker and taker belong to the same agent. |

`Event` is the union of the three.

### Ordering rules

- Events are in the order the engine handled them; `ts` never goes backwards.
- The `FillEvent`s caused by an order come **directly after** its `SubmitEvent`, in execution order.
- A cancel is logged whether or not it found an order, so a replay can check the answer.

### File format (`.eventlog`)

UTF-8 text, one JSON object per line, written with compact separators and a fixed key order so the
same run always produces the same bytes.

1. Line 1 is the header: `{"type":"meta","format":1,"seed":<int>,"config":{...}}`.
2. Every other line is one event: `{"type":"submit",...}`, `{"type":"cancel",...}` or
   `{"type":"fill",...}`, with one JSON field per dataclass field. Enums are written by name
   (`"BUY"`, `"LIMIT"`); a market order has `"price":null`.

A line that cannot be parsed raises `EventLogFormatError`.

### Replay rules

- Replay feeds every `SubmitEvent` and `CancelEvent` to a fresh engine, in order. `FillEvent`s are
  not fed in: they are what the replay must **reproduce**.
- If the engine produces different fills, or a different cancel result, replay raises
  `ReplayMismatchError` with the position of the first difference.
- Replay works on any engine that implements the `OrderBook` protocol, so the Python and C++ engines
  can both be checked against the same log.

## Agents, actions, and the market view (Phase 2)

Code: `src/lobx/sim/actions.py` and `src/lobx/sim/agents/base.py`.

### Actions

An action is a **request**, not an order. The agent states what it wants and the market stamps on
the order id, agent id and timestamp.

| Action | Fields | Meaning |
|---|---|---|
| `Submit` | `side`, `order_type`, `price`, `qty` | Place a new order. `Submit.limit(...)` and `Submit.market(...)` build the two kinds. |
| `Cancel` | `order_id` | Cancel one of the agent's **own** resting orders. Cancelling another agent's order raises `SimulationError`. |

Actions returned from one wake are carried out in list order, all at the wake's timestamp.

### The `Agent` protocol

```python
class Agent(Protocol):
    agent_id: int

    def first_wake(self, rng: Generator) -> int: ...
    def wake(self, ts: int, view: MarketView, rng: Generator) -> list[Action]: ...
    def next_wake(self, ts: int, rng: Generator) -> int | None: ...
```

- Times are absolute nanoseconds. `next_wake` must return a time strictly after `ts`, or `None` to
  retire the agent. Waking at the same instant forever would stall the clock, so the market rejects it.
- `rng` is a `numpy.random.Generator` owned by that agent. All randomness comes from it.
- Agents are woken in time order. Agents scheduled for the same instant wake in the order they were
  scheduled, so a run is deterministic.

### The `MarketView` protocol

Read-only. `ts`, `best_bid()`, `best_ask()`, `mid2()` (twice the mid-price, an integer),
`last_trade_price()`, `reference_price()` (mid, else last trade, else the opening price; always
defined), `snapshot(depth)`, `position(agent_id)` (signed shares), and `open_orders(agent_id)`
(resting order ids, oldest first).

### Randomness

Each agent gets its own stream, `SeedSequence(seed, spawn_key=(agent_id,))`. A stream depends only
on the run seed and the agent id, so adding or removing another agent never changes the numbers
an existing agent draws.

## Contracts added in later phases

These are listed in `AGENTS.md` and get their code stubs when the phase that needs them starts:

- `ParentOrder`, `ExecutionStrategy` (Phase 4)
- `extract_features` (Phase 5) with the rule that features at time `t` use only data with
  timestamp at or before `t`
- `implementation_shortfall_bps` (Phase 4)
