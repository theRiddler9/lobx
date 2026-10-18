import numpy as np
import pytest

from fakes import FakeView
from lobx.sim.agents.recorder import SnapshotRecorder

RNG = np.random.default_rng(0)


def test_records_evenly_spaced_snapshots_stamped_with_the_wake_time() -> None:
    recorder = SnapshotRecorder(9, interval_ns=100, depth=5)
    view = FakeView(bid=99, ask=101, ts=0)
    times = [recorder.first_wake(RNG)]
    for _ in range(2):
        assert recorder.wake(times[-1], view, RNG) == []
        following = recorder.next_wake(times[-1], RNG)
        assert following is not None
        times.append(following)
    assert times == [100, 200, 300]
    assert [s.ts for s in recorder.snapshots] == [100, 200]
    assert recorder.snapshots[0].bids == ((99, 1),)


def test_rejects_bad_settings() -> None:
    with pytest.raises(ValueError):
        SnapshotRecorder(1, interval_ns=0, depth=5)
