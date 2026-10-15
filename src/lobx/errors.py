"""Exceptions raised by LOBX components.

The engine never accepts bad input silently: it raises one of these with a message that says what
was wrong. The rules that trigger each error are listed in ``docs/contracts/shared-contracts.md``.
"""


class LobxError(Exception):
    """Base class for every LOBX-specific error."""


class InvalidOrderError(LobxError, ValueError):
    """An order or call violated a validation rule (bad quantity, price, or timestamp)."""


class DuplicateOrderIdError(InvalidOrderError):
    """An order id is already resting in the book."""
