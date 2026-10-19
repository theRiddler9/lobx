"""The WebSocket book stream: send each recorded book snapshot to a client, then the outcome.

Messages are JSON objects with a ``type``:

* ``snapshot``: ``{"type", "index", "ts_ns", "bids": [[price, qty], ...], "asks": [...]}``, best
  level first, one per simulated second, in order.
* ``done``: ``{"type": "done", "run": <RunInfo>}``, always the last message.

The run is simulated in a worker thread much faster than real time, so the stream paces itself:
it sends one snapshot every ``delay_s`` seconds, like a replay of the run as it unfolds.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import WebSocket, WebSocketDisconnect

from lobx.service.jobs import RunJob
from lobx.types import BookSnapshot

logger = logging.getLogger(__name__)

POLL_S = 0.02  # how often to look for a new snapshot when the client has caught up


def snapshot_message(index: int, snapshot: BookSnapshot) -> dict[str, object]:
    """Return the JSON-ready message for one snapshot."""
    return {
        "type": "snapshot",
        "index": index,
        "ts_ns": snapshot.ts,
        "bids": [list(level) for level in snapshot.bids],
        "asks": [list(level) for level in snapshot.asks],
    }


async def stream_run(websocket: WebSocket, job: RunJob, delay_s: float) -> None:
    """Accept the connection and stream ``job``'s book until the run is over."""
    await websocket.accept()
    sent = 0
    try:
        while True:
            # Read the status *before* the snapshot count. The worker appends the last snapshot
            # and only then marks the job finished, so "finished" seen first means every snapshot
            # is already in the list; reading in the other order could miss the last ones.
            finished = job.status in ("done", "failed")
            snapshots = job.recorder.snapshots if job.recorder is not None else []
            if sent < len(snapshots):
                await websocket.send_json(snapshot_message(sent, snapshots[sent]))
                sent += 1
                await asyncio.sleep(delay_s)
            elif finished:
                break
            else:
                await asyncio.sleep(POLL_S)
        await websocket.send_json({"type": "done", "run": job.info().model_dump()})
        await websocket.close()
    except WebSocketDisconnect:
        logger.info("client left the stream of %s after %d snapshots", job.id, sent)
