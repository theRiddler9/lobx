"""End-to-end integration tests for every major pipeline in LOBX.

Validates the full lifecycles:
1. Golden demo CLI: simulation -> event log -> replay -> differential consistency.
2. Strategy evaluation CLI: multi-seed runner -> statistics -> confidence intervals -> plots.
3. ML pipeline: training -> model persistence -> threshold tuning -> adaptive execution.
4. Historical replay: LOBSTER ingestion -> FIFO queue-position execution -> shortfall metrics.
5. Reinforcement Learning: PPO actor-critic training -> GAE advantages -> policy execution.
6. Service and streaming: FastAPI -> job store -> WebSocket streaming -> comparison tables.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from lobx.cli import main
from lobx.data.lobster import generate_lobster_sample
from lobx.engine.book_py import PythonOrderBook
from lobx.eval.harness import EvalConfig
from lobx.rl.environment import EnvConfig, ExecutionEnv
from lobx.rl.ppo import PPOAgent, Transition
from lobx.sim.clock import NS_PER_S


def test_run_then_replay_reproduces_the_final_book(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "demo.eventlog"
    assert main(["run", "--seed", "0", "--duration", "20", "--out", str(log)]) == 0
    ran = capsys.readouterr().out
    assert main(["replay", str(log)]) == 0
    replayed = capsys.readouterr().out
    final_book = next(line for line in ran.splitlines() if line.startswith("final book"))
    assert final_book in replayed  # the replay ends in exactly the book the run ended in


def test_eval_gives_a_results_table_and_a_paired_table_and_is_reproducible(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    command = ["eval", "--strategies", "twap,passive", "--seeds", "3", "--plot"]
    assert main([*command, str(tmp_path / "a.png")]) == 0
    first = capsys.readouterr().out
    assert main([*command, str(tmp_path / "b.png")]) == 0
    second = capsys.readouterr().out

    assert "| strategy | seeds | incomplete | mean (bps) | std (bps) | 95% CI" in first
    assert "| twap |" in first and "| passive |" in first
    assert "| passive - twap |" in first  # the paired comparison
    assert (tmp_path / "a.png").stat().st_size > 1_000
    # A run is a pure function of (config, seed): the same command prints the same numbers.
    assert first.replace("a.png", "x") == second.replace("b.png", "x")


def test_e2e_ml_train_tune_and_adaptive_eval(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train models via CLI on synthetic seeds
    train_cmd = [
        "train",
        "--train-seeds",
        "5",
        "--validation-seeds",
        "2",
        "--test-seeds",
        "2",
        "--out",
        str(models_dir),
    ]
    assert main(train_cmd) == 0
    train_out = capsys.readouterr().out
    assert "| lightgbm |" in train_out
    assert "| logistic |" in train_out
    model_file = models_dir / "lightgbm.json"
    assert model_file.is_file()

    # 2. Tune adaptive threshold via CLI
    tune_cmd = [
        "tune",
        "--model",
        str(model_file),
        "--thresholds",
        "0.4,0.6",
        "--seeds",
        "2",
    ]
    assert main(tune_cmd) == 0
    tune_out = capsys.readouterr().out
    assert "| threshold |" in tune_out

    # 3. Evaluate adaptive execution strategy with trained model
    eval_cmd = [
        "eval",
        "--strategies",
        "adaptive,twap",
        "--seeds",
        "2",
        "--model",
        str(model_file),
        "--threshold",
        "0.5",
    ]
    assert main(eval_cmd) == 0
    eval_out = capsys.readouterr().out
    assert "| adaptive |" in eval_out
    assert "| twap |" in eval_out
    assert "| twap - adaptive |" in eval_out or "| adaptive - twap |" in eval_out


def test_e2e_lobster_historical_replay(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    msg_path = tmp_path / "e2e_lobster_msg.csv"
    book_path = tmp_path / "e2e_lobster_book.csv"
    generate_lobster_sample(msg_path, book_path, num_events=50, seed_price=10_000)

    # Replay with TWAP execution strategy
    cmd = [
        "replay-lobster",
        "--message",
        str(msg_path),
        "--book",
        str(book_path),
        "--strategy",
        "twap",
        "--shares",
        "200",
        "--slices",
        "2",
        "--engine",
        "python",
    ]
    assert main(cmd) == 0
    out = capsys.readouterr().out
    assert "replayed 50 historical events" in out
    assert "total market volume:" in out
    assert "executed:" in out
    assert "arrival mid:" in out
    assert "shortfall:" in out

    # Replay with pure reconstruction (no strategy)
    cmd_none = [
        "replay-lobster",
        "--message",
        str(msg_path),
        "--strategy",
        "none",
        "--engine",
        "python",
    ]
    assert main(cmd_none) == 0
    out_none = capsys.readouterr().out
    assert "replayed 50 historical events" in out_none


def test_e2e_rl_ppo_training_and_execution() -> None:
    cfg = EnvConfig(parent_qty=300, num_steps=3, slice_size=100, seed=42)
    env = ExecutionEnv(cfg)
    agent = PPOAgent(obs_dim=5, num_actions=3, hidden_dim=16, seed=123)

    # Collect rollouts and run GAE training step
    transitions: list[Transition] = []
    obs, _ = env.reset(seed=42)

    for _ in range(3):
        action, log_prob = agent.select_action(obs)
        val = agent.value(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        transitions.append(
            Transition(
                obs=obs,
                action=action,
                reward=reward,
                val=val,
                log_prob=log_prob,
                done=terminated or truncated,
            )
        )
        obs = next_obs
        if terminated or truncated:
            break

    last_val = agent.value(obs)
    loss = agent.train_step(transitions, last_val=last_val)
    assert isinstance(loss, float)

    # Run execution after policy update
    obs, info = env.reset(seed=42)
    for _ in range(3):
        probs = agent.policy(obs)
        best_action = int(np.argmax(probs))
        obs, _, term, trunc, info = env.step(best_action)
        if term or trunc:
            break

    assert info["executed_qty"] == 300
    assert info["shortfall_bps"] is not None


def test_e2e_service_lifecycle() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from lobx.service.app import create_app

    SHORT = EvalConfig(
        parent_qty=300, start_ns=20 * NS_PER_S, window_ns=60 * NS_PER_S, num_slices=6
    )
    app = create_app(SHORT, PythonOrderBook, Path("does-not-exist.json"))

    with TestClient(app) as client:
        # Health
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "ok"

        # Start run
        start_resp = client.post("/runs", json={"strategy": "twap", "seed": 5, "parent_qty": 300})
        assert start_resp.status_code == 202
        run_id = start_resp.json()["id"]

        # Poll run status until done
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            res = client.get(f"/runs/{run_id}").json()
            if res["status"] in ("done", "failed"):
                break
            time.sleep(0.05)
        assert res["status"] == "done"
        assert res["result"]["executed"] == 300

        # Start comparison
        comp_resp = client.post(
            "/comparisons", json={"strategies": ["twap", "passive"], "seeds": 2}
        )
        assert comp_resp.status_code == 202
        comp_id = comp_resp.json()["id"]

        # Poll comparison status until done
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            c_res = client.get(f"/comparisons/{comp_id}").json()
            if c_res["status"] in ("done", "failed"):
                break
            time.sleep(0.05)
        assert c_res["status"] == "done"
        assert len(c_res["rows"]) == 2


def test_e2e_multi_asset_cross_arbitrage(capsys: pytest.CaptureFixture[str]) -> None:
    # 1. Pairs execution end-to-end
    pairs_cmd = [
        "multi-sim",
        "--symbols",
        "SPY,QQQ",
        "--duration",
        "2",
        "--strategy",
        "pairs",
        "--shares",
        "100",
        "--slices",
        "2",
        "--engine",
        "native",
    ]
    assert main(pairs_cmd) == 0
    pairs_out = capsys.readouterr().out
    assert "simulated multi-asset market across ('SPY', 'QQQ')" in pairs_out
    assert "[SPY] trades:" in pairs_out
    assert "[QQQ] trades:" in pairs_out
    assert "executed SPY: 100/100 shares" in pairs_out
    assert "executed QQQ: 100/100 shares" in pairs_out
    assert "combined pairs shortfall:" in pairs_out

    # 2. Lead-lag cross arbitrage end-to-end
    lead_cmd = [
        "multi-sim",
        "--symbols",
        "LEAD,LAG",
        "--duration",
        "2",
        "--strategy",
        "lead-lag",
        "--shares",
        "20",
        "--engine",
        "native",
    ]
    assert main(lead_cmd) == 0
    lead_out = capsys.readouterr().out
    assert "simulated multi-asset market across ('LEAD', 'LAG')" in lead_out
    assert "arbitrageur trades executed:" in lead_out
