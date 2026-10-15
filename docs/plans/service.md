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
