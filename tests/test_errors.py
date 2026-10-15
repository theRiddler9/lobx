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
