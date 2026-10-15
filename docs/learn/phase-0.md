# Phase 0 notes: scaffold, docs, contracts

## What this phase built

No trading logic yet. Phase 0 sets up the project so every later phase has clear rules to follow:
a Python package, strict tooling, a plan, written contracts, and automated checks.

## Key concepts

**Src layout.** The package lives in `src/lobx/` instead of the repository root. This stops tests
from accidentally importing the source folder instead of the installed package, so a broken install
shows up early.

**`pyproject.toml`.** One file describes the package (name, Python version) and configures the tools
(ruff, mypy, pytest). The dev tools are pinned to exact versions, and `uv.lock` records the full
dependency tree so every machine and CI run uses the same versions.

**Strict typing.** `mypy --strict` makes the type checker reject missing or loose annotations. In an
engine where an integer price accidentally becoming a float would silently break matching, this
catches mistakes before tests run.

**Contracts.** `docs/contracts/shared-contracts.md` is the written agreement between components:
the data types, the matching rules, the validation rules. The Python engine, the C++ engine, the
simulator and the strategies all depend on it. Writing it first means the C++ engine in Phase 3 can
be built to the same rules instead of copying whatever the Python code happens to do.

**Integer ticks and integer nanoseconds.** Prices are counts of the smallest price step, and time is
a count of nanoseconds. Floats cannot represent most decimal prices exactly, and two engines doing
floating-point math can disagree in the last digit. Integers make "identical output" provable.

**Frozen dataclasses.** `Order`, `Fill` and `BookSnapshot` cannot change after creation. If one part
of the code could modify an order, another part would not be able to trust it. Frozen data also makes
event logs and replays safe to reason about.

**Protocols.** `OrderBook` in `engine/protocol.py` says what an engine must do without saying how.
Any class with those five methods counts, so the Python and C++ engines can be swapped freely.

**Git workflow.** Work happens on short-lived branches, in small commits, merged with merge commits
(no squash) so the history reads as a progress log.

## Where to look in the code

- `pyproject.toml`: package metadata and tool settings.
- `src/lobx/types.py`: the contract types.
- `src/lobx/errors.py`: the error hierarchy.
- `src/lobx/engine/protocol.py`: the `OrderBook` protocol.
- `docs/contracts/shared-contracts.md`: the rules every engine must follow.
- `.github/workflows/ci.yml`: the checks that run on every push.

## What to read next

- The Python docs on `dataclasses` (especially `frozen` and `slots`).
- The Python docs on `typing.Protocol`.
- A short article on limit order books: what a bid, an ask, and the spread are.

## Practice exercises

1. **Add a field safely.** Add an optional `client_tag: str | None = None` to `Order`. Which tests
   and which contract text need updating? Make the change on a `contract/` branch.
2. **Write a second stub.** Write a small class that satisfies `OrderBook` but always rejects every
   order by raising `InvalidOrderError`. Add a test proving `isinstance(x, OrderBook)` is true.
3. **Break the type checker on purpose.** Change `price: int | None` to `price: float | None` in a
   scratch copy of `types.py` and run `mypy`. What breaks, and why would that be a problem later?
