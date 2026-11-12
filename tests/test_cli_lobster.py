from pathlib import Path

from lobx.cli import main
from lobx.data.lobster import generate_lobster_sample


def test_cli_replay_lobster_twap(tmp_path: Path, capsys: object) -> None:
    msg_path = tmp_path / "cli_msg.csv"
    book_path = tmp_path / "cli_book.csv"
    generate_lobster_sample(msg_path, book_path, num_events=40)

    exit_code = main(
        [
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
    )
    assert exit_code == 0


def test_cli_replay_lobster_none(tmp_path: Path) -> None:
    msg_path = tmp_path / "cli_msg_none.csv"
    book_path = tmp_path / "cli_book_none.csv"
    generate_lobster_sample(msg_path, book_path, num_events=30)

    exit_code = main(
        [
            "replay-lobster",
            "--message",
            str(msg_path),
            "--strategy",
            "none",
            "--engine",
            "python",
        ]
    )
    assert exit_code == 0


def test_cli_replay_lobster_missing_args() -> None:
    try:
        main(["replay-lobster"])
    except SystemExit as exc:
        assert exc.code != 0
