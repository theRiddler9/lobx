# Phase 7: the service layer

## What this phase built

A way to use LOBX without writing Python: an HTTP API that starts simulated runs, a WebSocket that
streams the order book, a Streamlit dashboard that draws it and compares strategies, and a Docker
setup that starts everything with one command. It adds **no trading logic**; every number still
comes from the Phase 4-6 code.

## Key concepts

- **Accepted, not finished (HTTP 202).** A simulation takes seconds, so `POST /runs` answers at
  once with an id and the work happens in a worker thread; the client polls `GET /runs/{id}`.
- **Validation at the door.** pydantic models (`schemas.py`) reject a bad request (unknown strategy,
  zero shares) with a 422 before any simulation starts.
- **A factory, not globals.** `create_app(...)` builds an app whose state (job store, worker pool)
  lives in local variables. Tests build their own app with a short scenario; nothing is shared.
- **Shared state between threads.** A worker writes a job; request handlers read it. This is safe
  only because fields are assigned whole, a job moves forward only, and the snapshot list only
  grows. The WebSocket reads the *status first*, then the snapshot count, so it cannot miss the
  last snapshots (read `stream.py` for why).
- **WebSocket.** One long-lived connection; the server pushes messages (`snapshot`, then `done`).
  Here it replays the recorded book at a chosen pace.
- **Optional dependencies.** The web stack is an extra (`pip install -e ".[service]"`), so the
  engine and simulator install without it.
- **Testing a server for real.** `tests/conftest.py` starts uvicorn on a free port in a thread;
  the client and dashboard tests talk to it over actual HTTP and WebSocket.
- **Containers.** One image, two commands (API, dashboard). Models are trained at build time.
  Docker could not be built in the sandbox (no route to Docker Hub), so the first real build is
  still to do on your machine; if it fails, the likely culprit is a system package in the
  Dockerfile.

## Where to look

- `src/lobx/service/app.py` (routes), `jobs.py`, `execution.py`, `stream.py`, `client.py`
- `dashboard/app.py` (the whole UI, in two small functions)
- `tests/test_service_app.py`, `tests/test_service_client.py`, `tests/test_dashboard.py`

## What to read next

FastAPI's tutorial pages on request bodies and WebSockets, then Streamlit's "app execution model"
(why the script re-runs on every click).

## Practice exercises

1. Add `GET /runs` that lists all runs with their status. Write the test first.
2. Make the stream send the strategy's fills as they happen (a `fill` message between snapshots)
   and draw them as markers in the dashboard.
3. Add a `DELETE /runs/{id}` that forgets a finished run, and decide what should happen if the
   run is still going. Write down the choice and test it.
