"""Tests for lobx multi-sim CLI command."""

import pytest

from lobx.cli import main


def test_cli_multi_sim_pairs(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "multi-sim",
            "--symbols",
            "LEAD,LAG",
            "--duration",
            "1",
            "--strategy",
            "pairs",
            "--shares",
            "50",
            "--slices",
            "2",
            "--engine",
            "python",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "simulated multi-asset market across ('LEAD', 'LAG')" in out
    assert "[LEAD] trades:" in out
    assert "[LAG] trades:" in out
    assert "executed LEAD: 50/50 shares" in out
    assert "executed LAG: 50/50 shares" in out
    assert "combined pairs shortfall:" in out


def test_cli_multi_sim_lead_lag(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "multi-sim",
            "--symbols",
            "SPY,QQQ",
            "--duration",
            "1",
            "--strategy",
            "lead-lag",
            "--shares",
            "10",
            "--engine",
            "native",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "simulated multi-asset market across ('SPY', 'QQQ')" in out
    assert "arbitrageur trades executed:" in out
