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


class EventLogFormatError(LobxError, ValueError):
    """An event log line could not be parsed (bad JSON, unknown event kind, missing field)."""


class ReplayMismatchError(LobxError):
    """Replaying a log produced different fills or cancel results than the log recorded."""


class SimulationError(LobxError):
    """An agent or the simulator broke a rule, such as cancelling another agent's order."""


class NativeEngineUnavailableError(LobxError, ImportError):
    """The C++ engine extension is not built, so ``NativeOrderBook`` cannot be used."""
