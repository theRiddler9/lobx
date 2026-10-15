# Simulator plan

Owner area: `src/lobx/sim/`.

## Goal

A deterministic multi-agent market that drives the engine and produces realistic-enough order flow
for execution experiments.

## Pieces

- Clock and event queue: agents wake in timestamp order, ties broken by agent id.
- Seeded random streams: one independent stream per agent, derived from the run seed.
- Agents: noise traders (Poisson arrivals), a market maker with inventory skew, a momentum trader.
- Market: applies agent actions to the engine and publishes snapshots to strategies.

## Determinism rules

- No global random state. Every draw uses the agent's own generator.
- Single-threaded. A run is a pure function of `(config, seed)`.
- Same seed must give a byte-identical event log (tested).

## Realism checks (Phase 2 exit)

- The spread is positive on average.
- Mid-price returns have fat tails (kurtosis above that of a normal distribution).
- The market maker's inventory stays bounded.

## Design reference

The ABIDES agent mix (noise, value, momentum, market maker) is a design reference only. Nothing is
copied from it.
