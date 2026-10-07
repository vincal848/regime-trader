"""Step 5: playbooks are data parsed into typed rules, never code (spec §7)."""

from pathlib import Path

import pytest

from regime_trader.playbook import Playbook, PlaybookError, evaluate_signal, parse_playbook

PLAYBOOK_DIR = Path(__file__).resolve().parents[1] / "playbooks"

EXAMPLE = """# CALM_UP: trend following

Rationale in prose, ignored by the parser.

```toml
state = "CALM_UP"
entry = "trend > 0.5 and ret > 0"
exit = "trend < 0"
stop_loss_vol = 2.0
take_profit_vol = 4.0
max_size = 0.8
max_hold_bars = 35
invalidation = "Two consecutive closes below the 70-bar EMA."
```
"""


def _with(**changes: str) -> str:
    text = EXAMPLE
    for key, value in changes.items():
        lines = [line for line in text.splitlines() if not line.startswith(f"{key} =")]
        text = "\n".join(lines).replace("```toml\n", f"```toml\n{key} = {value}\n", 1)
    return text


def test_a_playbook_parses_into_typed_fields() -> None:
    playbook = parse_playbook(EXAMPLE)
    assert isinstance(playbook, Playbook)
    assert playbook.state == "CALM_UP"
    assert (playbook.stop_loss_vol, playbook.take_profit_vol, playbook.max_size) == (2.0, 4.0, 0.8)
    assert playbook.max_hold_bars == 35
    assert "70-bar EMA" in playbook.invalidation


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"leverage": "3"}, "unknown"),
        ({"entry": '"momentum > 1"'}, "feature"),
        ({"entry": "\"__import__('os').system('x') > 0\""}, "condition"),
        ({"entry": '"trend > 0; rm -rf /"'}, "condition"),
        ({"entry": '"trend == 0"'}, "condition"),
        ({"max_size": "1.5"}, "max_size"),
        ({"max_size": "-0.1"}, "max_size"),
        ({"stop_loss_vol": "0"}, "stop_loss_vol"),
    ],
)
def test_invalid_playbooks_are_rejected(changes: dict[str, str], message: str) -> None:
    with pytest.raises(PlaybookError, match=message):
        parse_playbook(_with(**changes))


def test_a_playbook_needs_exactly_one_toml_block() -> None:
    with pytest.raises(PlaybookError, match="toml"):
        parse_playbook("# no parameters here")


ROW = {"ret": 0.001, "rv": 0.002, "range": 0.003, "volume_ratio": 1.2, "trend": 0.8}


def test_entry_and_exit_conditions() -> None:
    playbook = parse_playbook(EXAMPLE)
    assert evaluate_signal(playbook, ROW, in_position=False, bars_held=0).enter
    assert not evaluate_signal(playbook, {**ROW, "ret": -0.001}, in_position=False, bars_held=0).enter
    assert evaluate_signal(playbook, {**ROW, "trend": -0.2}, in_position=True, bars_held=3).exit
    assert not evaluate_signal(playbook, ROW, in_position=True, bars_held=3).exit


def test_and_binds_tighter_than_or() -> None:
    playbook = parse_playbook(_with(entry='"trend > 5 or ret > 0 and volume_ratio > 1"'))
    assert evaluate_signal(playbook, ROW, in_position=False, bars_held=0).enter  # ret>0 and vol>1
    assert not evaluate_signal(playbook, {**ROW, "volume_ratio": 0.5}, False, 0).enter


def test_z_features_and_negative_numbers_are_allowed() -> None:
    playbook = parse_playbook(_with(entry='"z_ret < -1.5"'))
    assert evaluate_signal(playbook, {**ROW, "z_ret": -2.0}, False, 0).enter


def test_never_and_always() -> None:
    flat = parse_playbook(_with(entry='"never"', exit='"always"'))
    assert not evaluate_signal(flat, ROW, False, 0).enter
    assert evaluate_signal(flat, ROW, True, 1).exit


def test_missing_features_never_trigger_an_entry() -> None:
    playbook = parse_playbook(EXAMPLE)
    assert not evaluate_signal(playbook, {**ROW, "trend": float("nan")}, False, 0).enter


def test_holding_too_long_forces_an_exit() -> None:
    playbook = parse_playbook(EXAMPLE)
    signal = evaluate_signal(playbook, ROW, in_position=True, bars_held=35)
    assert signal.exit
    assert "max_hold_bars" in signal.reason


@pytest.mark.parametrize("state", ["CALM_UP", "CHOP", "STRESS", "CRASH"])
def test_the_starting_playbooks_parse(state: str) -> None:
    playbook = parse_playbook((PLAYBOOK_DIR / f"{state}.md").read_text(encoding="utf-8"))
    assert playbook.state == state


def test_the_crash_playbook_never_enters() -> None:
    crash = parse_playbook((PLAYBOOK_DIR / "CRASH.md").read_text(encoding="utf-8"))
    assert crash.max_size == 0.0
    assert not evaluate_signal(crash, ROW, False, 0).enter
