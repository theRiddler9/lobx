# Phase 1 notes: the Python reference engine

## What this phase built

A working limit order book, written to be read. `PythonOrderBook` accepts limit orders, market
orders and cancels, matches them by the standard exchange rules, and returns the trades (fills) it
produced. It has no agents, no clock and no randomness: you give it orders, it gives you fills.

In Phase 3 a fast C++ engine must behave exactly like this one. That is why this file favours
clarity over speed, and why the rules it follows are written down in
`docs/contracts/shared-contracts.md`.

## Key concepts

**Limit order book.** Two lists of waiting orders. *Bids* are buyers, *asks* are sellers. The best
bid (highest price a buyer will pay) and the best ask (lowest price a seller will accept) sit at the
top. The gap between them is the *spread*. While the book is healthy the best bid is strictly lower
than the best ask; if they touched, a trade would already have happened.

**Limit order vs market order.** A limit order says "buy up to this price". Whatever cannot trade
right away *rests* in the book and waits. A market order says "buy now at any price". It takes
whatever is there and its unfilled remainder is thrown away, because it has no price to rest at.

**Price-time priority.** Two rules decide who trades first. Better price wins (a higher bid, a lower
ask). At the same price, the order that arrived first wins. Every exchange works this way, and every
matching rule in `PythonOrderBook._match` is one of these two.

**Trades happen at the resting order's price.** If a sell sits at 100 and a buyer arrives willing to
pay 105, the trade is at 100. The buyer gets a better price than they asked for. The incoming order
is the *taker*, the waiting order is the *maker*.

**Partial fills.** An order can trade in pieces. A waiting order that is only partly filled keeps its
place in the queue with a smaller quantity. One incoming order can also produce several fills when it
sweeps through several resting orders ("walking the book").

**Price levels.** All orders at one price are stored together, in arrival order. The snapshot adds up
their quantities, which is the "depth" you see on a trading screen.

**`OrderedDict` as a queue.** `PriceLevel` keeps its orders in an `OrderedDict` keyed by order id. It
remembers insertion order, so it behaves as a FIFO queue, and it can also remove an order from the
middle in O(1), which is exactly what a cancel needs.

**Validation before mutation.** `_validate` checks every rule before the book is touched. A rejected
order therefore leaves no trace: nothing rests, nothing trades, and the engine clock does not move.

**Property-based testing.** A normal test checks one example you thought of. A property test states a
rule ("the book is never crossed") and lets `hypothesis` try hundreds of random order streams,
shrinking any failure to a small case. Here it also compares the engine with `_NaiveBook`, a
deliberately slow and obviously-correct model. When two independent implementations agree on
thousands of random streams, both are probably right. Phase 3 reuses this idea: Python versus C++.

**Hand-computed fixtures.** `tests/fixtures/book_scenarios.json` holds scenarios whose expected
fills were worked out on paper. They test the engine against the *rules*, not against its own
output, and the C++ engine will run the same file.

## Decisions made in this phase

- Timestamps may repeat but never go backwards, across both `submit` and `cancel`. This catches a
  whole class of simulator bugs early and keeps replay honest.
- An order id may be reused once the order is gone (filled or cancelled). The engine only refuses an
  id that is currently resting. The market will allocate unique ids for a whole run.
- Cancelling an unknown or finished order returns `False` instead of raising, because a cancel
  racing with a fill is normal.
- One mistake along the way is worth knowing about: a test file was merged while failing, because the
  check command hid pytest's exit code behind a pipe. It was fixed in a follow-up commit
  (`fix(test): parse lowercase side names in scenario fixtures`) rather than by rewriting `main`.
  Lesson: a check that cannot fail is not a check.

## Where to look in the code

- `src/lobx/engine/price_level.py`: `PriceLevel`, the FIFO queue at one price.
- `src/lobx/engine/book_side.py`: `BookSide`, all the levels of bids or asks.
- `src/lobx/engine/book_py.py`: `PythonOrderBook`. Read `submit`, then `_match`, then `cancel`.
- `tests/test_book_scenarios.py` and `tests/fixtures/book_scenarios.json`: the paper-checked cases.
- `tests/test_book_properties.py`: the random streams and the naive model.

## What to read next

- An introduction to limit order books (bid, ask, spread, depth, price-time priority).
- The `hypothesis` documentation, "What you can generate and how".
- Python docs for `collections.OrderedDict` and `bisect`.

## Practice exercises

1. **Re-implement cancel without looking.** Close `cancel` in `book_py.py`, write your own version
   from the rules in `shared-contracts.md`, then run `tests/test_book_cancel.py` and the property
   tests. Which edge cases did you miss on the first try?
2. **Add fill-or-kill.** Add an order type that must be filled completely and immediately, or not
   at all. Start in the contract (a new `OrderType` goes through a `contract/` branch), then add the
   scenario to `book_scenarios.json` before writing any engine code, then make it pass.
3. **Break it on purpose.** Change `_crosses` so a buyer accepts asks *strictly below* its limit
   instead of at or below. Run the tests. Which ones fail, and does the failure message tell you
   what you broke? Revert it afterwards.
