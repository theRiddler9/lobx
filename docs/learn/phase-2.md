# Phase 2 notes: event log, replay, and the simulator

## What this phase built

A deterministic market. Agents (market makers, noise traders, momentum traders) take turns trading
on the Phase 1 engine, and everything that happens is written to an append-only event log that can
be replayed exactly. `python -m lobx.cli run --seed 0` simulates ten minutes in under a second;
`python -m lobx.cli replay run-seed-0.eventlog` rebuilds the final book from the file and checks it.

## Key concepts

**Event log.** A record of every accepted order, every cancel (with its outcome) and every fill. The
inputs are enough to rebuild a book; the fills are the answer key that replay checks against.

**Replay.** Feed the logged inputs to a fresh engine and compare the engine's fills with the logged
ones. If they match, the log fully describes the run. In Phase 3 the same log is replayed on the C++
engine, which is how the two engines are shown to agree.

**Determinism.** A run is a pure function of `(config, seed)`. Nothing reads the wall clock or global
random state, agents are woken in a fixed order, and each agent has its own random stream derived
from the seed and its id. Tests prove it: same seed gives identical logs, even in separate
processes with different `PYTHONHASHSEED`.

**Discrete-event simulation.** Time is an integer number of nanoseconds that jumps from one agent
turn to the next (`WakeQueue`). Nothing "waits"; that is why a ten minute run takes under a second.

**Poisson arrivals.** Noise traders wake after exponentially distributed gaps, the standard model for
orders arriving from many independent traders.

**Inventory skew.** A market maker shifts its quotes against its position so that it drifts back
towards zero, and stops quoting the side that would push it past a hard limit.

**Fat tails.** Real price changes have far more extreme moves than a normal distribution. Excess
kurtosis measures this (0 for a normal distribution); the simulated market scores about 60.

**A lesson about tuning.** The first default market passed on paper but was frozen: the price moved
0.04 ticks per second. Passing a statistic is not the same as being useful. Thinner market maker
quotes and more aggressive noise traders gave about 1 tick per second, which is enough for execution
and prediction experiments later. `tests/test_sim_realism.py` now guards against a frozen price.

## Where to look in the code

- `src/lobx/engine/event_log.py`, `event_codec.py`, `logged_book.py`, `replay.py`: the log.
- `src/lobx/sim/clock.py`, `rng.py`: time and random streams.
- `src/lobx/sim/agents/`: noise, market maker, momentum.
- `src/lobx/sim/population.py`: `MarketConfig` and the default cast.
- `src/lobx/sim/market.py`: the loop. Read `Market.run`, then `_apply`.
- `src/lobx/cli.py`: the `run` and `replay` commands.

## What to read next

- Avellaneda and Stoikov, "High-frequency trading in a limit order book" (skim the idea).
- A short introduction to discrete-event simulation and Poisson processes.
- The ABIDES paper, for how a much larger simulator is organised (design reference only).

## Practice exercises

1. **Add an agent.** Write a `ValueTrader` that buys when the price is below a hidden fundamental
   value and sells when above. Add unit tests with `FakeView` (see `tests/fakes.py`), then add it to
   `build_background_agents`. What happens to the return statistics?
2. **Break determinism on purpose.** In a scratch copy, make the market iterate over a `set` of
   agents, or call `random.random()` in an agent. Which test in `tests/test_determinism.py` fails?
3. **Replay a doctored log.** Run the CLI, change one price in a fill line of the file by hand, and
   replay it. Read the error message, then find the line in `replay.py` that produced it.
