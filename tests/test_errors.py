"""Tests for the error hierarchy."""

import pytest

from lobx.errors import DuplicateOrderIdError, InvalidOrderError, LobxError


def test_hierarchy() -> None:
    assert issubclass(InvalidOrderError, LobxError)
    assert issubclass(InvalidOrderError, ValueError)
    assert issubclass(DuplicateOrderIdError, InvalidOrderError)


def test_duplicate_id_can_be_caught_as_invalid_order() -> None:
    with pytest.raises(InvalidOrderError, match="already resting"):
        raise DuplicateOrderIdError("order id 5 is already resting")


def test_new_errors_share_the_lobx_base() -> None:
    from lobx.errors import EventLogFormatError, LobxError, ReplayMismatchError, SimulationError

    for error in (EventLogFormatError, ReplayMismatchError, SimulationError):
        assert issubclass(error, LobxError)
    assert issubclass(EventLogFormatError, ValueError)
