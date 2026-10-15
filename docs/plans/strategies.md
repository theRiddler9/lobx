# Strategies plan

Owner area: `src/lobx/strategies/`.

## Goal

Execute a large parent order (for example, sell 50,000 shares over 30 simulated minutes) while
minimizing implementation shortfall.

## Strategies

- TWAP: equal slices at equal time intervals. The simple baseline.
- VWAP: slices follow an estimated intraday volume profile.
- Adaptive (ML): uses a short-horizon price-move prediction to choose between passive limit orders
  and aggressive market orders. Must finish the parent order by the deadline.

## Rules

- Strategies read snapshots and return actions. They never mutate the book.
- Strategies never use data from the future.
- The parent order always completes by the deadline; any remainder is crossed with market orders.
- Child orders never exceed the parent quantity in total.

## Metric

Implementation shortfall in basis points against the arrival mid price. See `evaluation.md`.
