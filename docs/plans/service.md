# Service plan

Owner area: `src/lobx/service/`, `dashboard/`.

## Goal

Make the project demoable: start a run through an API, watch the book live, compare strategies.

## Pieces

- FastAPI app: health endpoint, submit a parent order and start a run, fetch results.
- WebSocket: stream book snapshots from a running simulation.
- Streamlit dashboard: live book view and strategy comparison view.
- Docker and compose file so `docker compose up` serves everything.

## Rules

- The service wires existing components and adds no new trading logic.
- Cut line: the dashboard and WebSocket are the first things to drop if time runs short.

## What was built (Phase 7)

| Piece | Where |
|---|---|
| `GET /health` | `src/lobx/service/app.py` |
| `POST /runs` (202), `GET /runs/{id}` | same; jobs in `jobs.py`, execution in `execution.py` |
| `POST /comparisons` (202), `GET /comparisons/{id}` | same |
| `WS /runs/{id}/stream?delay_ms=` | `stream.py` |
| `python -m lobx.cli serve` | `src/lobx/cli.py` |
| API client (stdlib HTTP + websockets) | `src/lobx/service/client.py` |
| Streamlit dashboard | `dashboard/app.py` |
| `Dockerfile`, `compose.yaml` | repo root |

Interactive API docs are served at `/docs` by FastAPI.
