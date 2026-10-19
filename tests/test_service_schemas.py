import pytest

pytest.importorskip("pydantic")

from pydantic import ValidationError  # noqa: E402

from lobx.eval.seeds import TEST_SEEDS  # noqa: E402
from lobx.service.schemas import ComparisonRequest, RunRequest  # noqa: E402


def test_a_run_request_needs_only_a_strategy() -> None:
    request = RunRequest(strategy="twap")
    assert request.seed == TEST_SEEDS.start
    assert request.parent_qty == 3_000
    assert request.side == "sell"


@pytest.mark.parametrize(
    "bad",
    [
        {"strategy": "magic"},
        {"strategy": "twap", "parent_qty": 0},
        {"strategy": "twap", "parent_qty": 20_001},
        {"strategy": "twap", "parent_qty": 1.5},
        {"strategy": "twap", "seed": -1},
        {"strategy": "twap", "side": "hold"},
    ],
)
def test_bad_run_requests_are_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RunRequest(**bad)


def test_comparison_limits() -> None:
    assert ComparisonRequest(strategies=["twap", "vwap"]).seeds == 10
    bad_requests: list[dict[str, object]] = [
        {"strategies": []},
        {"strategies": ["twap"], "seeds": 1},
        {"strategies": ["twap"], "seeds": 201},
        {"strategies": ["twap", "vwap", "passive", "adaptive", "twap"]},
    ]
    for bad in bad_requests:
        with pytest.raises(ValidationError):
            ComparisonRequest(**bad)
