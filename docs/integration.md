# Integration log

Merge checkpoints, blockers, and decisions that affect more than one component. If a contract is
blocking, record it here and ask before working around it.

## Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-10-16 | `Fill` gets a `taker_side` field (extends the first draft in `AGENTS.md`). | Phase 5 trade-flow features need the aggressor side without reading the event log. |
| 2026-10-16 | Exceptions live in `src/lobx/errors.py`. | Shared by the Python and C++ engines (the bindings translate to the same classes). |
| 2026-10-16 | Packaging starts on `setuptools`; it moves to `scikit-build-core` in Phase 3. | There is no C++ code to build until Phase 3. |
| 2026-10-16 | `ruff format` skips `*.md`. | `AGENTS.md` holds illustrative pseudo-code that must stay as written. |
| 2026-10-16 | Timestamps may repeat but never go backwards across `submit` and `cancel`; an id may be reused once its order is gone. | Catches simulator clock bugs early; keeps the engine check cheap (only resting ids are tracked). The C++ engine must match. |

## Blockers

None.

## Merge checkpoints

- Contracts merge to `main` before dependent work starts.
- Rebase on `origin/main` before opening a PR; merge with merge commits, no squash.
