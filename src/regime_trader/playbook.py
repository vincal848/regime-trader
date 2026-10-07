"""Per-state playbooks: Markdown written by the research layer, parsed into typed rules (spec §7).

A playbook is prose plus exactly one fenced TOML block:

    state, entry, exit, stop_loss_vol, take_profit_vol, max_size, max_hold_bars, invalidation

`entry` and `exit` are conditions in a deliberately tiny grammar, which is
parsed and never evaluated as code:

    condition := "always" | "never" | clause (("and" | "or") clause)*
    clause    := FEATURE ("<" | "<=" | ">" | ">=") NUMBER

`and` binds tighter than `or`. FEATURE must be one of the pipeline's raw or
z-scored features. Stops and take-profits are multiples of the 21-bar
realized volatility (`rv`, in log-return units), measured from the entry
price. A comparison against a missing (NaN) feature is false, so missing
data can never trigger an entry.
"""

from __future__ import annotations

import math
import operator
import re
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from regime_trader.features import FEATURES, Z_FEATURES

ALLOWED_FEATURES = frozenset(FEATURES) | frozenset(Z_FEATURES)
KEYS = {
    "state",
    "entry",
    "exit",
    "stop_loss_vol",
    "take_profit_vol",
    "max_size",
    "max_hold_bars",
    "invalidation",
}
_OPERATORS: dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}
_CLAUSE = re.compile(r"^\s*([a-z_]+)\s*(<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)\s*$")
_TOML_BLOCK = re.compile(r"```toml\n(.*?)```", re.DOTALL)


class PlaybookError(ValueError):
    """A playbook failed to parse or validate."""


@dataclass(frozen=True)
class Clause:
    feature: str
    op: str
    threshold: float

    def holds(self, row: Mapping[str, float]) -> bool:
        value = row.get(self.feature, math.nan)
        return not math.isnan(value) and _OPERATORS[self.op](value, self.threshold)


@dataclass(frozen=True)
class Condition:
    """Disjunction of conjunctions: any group whose clauses all hold.
    () means never; ((),) means always."""

    groups: tuple[tuple[Clause, ...], ...]

    def holds(self, row: Mapping[str, float]) -> bool:
        return any(all(clause.holds(row) for clause in group) for group in self.groups)


@dataclass(frozen=True)
class Playbook:
    state: str
    entry: Condition
    exit: Condition
    stop_loss_vol: float
    take_profit_vol: float
    max_size: float
    max_hold_bars: int  # 0: no time limit
    invalidation: str


@dataclass(frozen=True)
class Signal:
    enter: bool
    exit: bool
    reason: str


def parse_condition(text: str) -> Condition:
    text = text.strip()
    if text == "always":
        return Condition(((),))
    if text == "never":
        return Condition(())
    groups = []
    for disjunct in re.split(r"\s+or\s+", text):
        clauses = []
        for conjunct in re.split(r"\s+and\s+", disjunct):
            match = _CLAUSE.match(conjunct)
            if match is None:
                raise PlaybookError(f"invalid condition clause {conjunct!r}: expected 'feature <op> number'")
            feature, op, number = match.groups()
            if feature not in ALLOWED_FEATURES:
                raise PlaybookError(f"unknown feature {feature!r}; allowed: {sorted(ALLOWED_FEATURES)}")
            clauses.append(Clause(feature, op, float(number)))
        groups.append(tuple(clauses))
    return Condition(tuple(groups))


def parse_playbook(markdown: str) -> Playbook:
    blocks = _TOML_BLOCK.findall(markdown)
    if len(blocks) != 1:
        raise PlaybookError(f"a playbook needs exactly one ```toml block, found {len(blocks)}")
    try:
        values = tomllib.loads(blocks[0])
    except tomllib.TOMLDecodeError as error:
        raise PlaybookError(f"invalid toml: {error}") from error
    unknown, missing = set(values) - KEYS, KEYS - set(values)
    if unknown or missing:
        raise PlaybookError(f"unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    if not 0.0 <= values["max_size"] <= 1.0:
        raise PlaybookError(f"max_size must be within [0, 1], got {values['max_size']}")
    for key in ("stop_loss_vol", "take_profit_vol"):
        if not values[key] > 0:
            raise PlaybookError(f"{key} must be positive, got {values[key]}")
    if not isinstance(values["max_hold_bars"], int) or values["max_hold_bars"] < 0:
        raise PlaybookError(f"max_hold_bars must be a non-negative integer, got {values['max_hold_bars']}")
    return Playbook(
        state=str(values["state"]),
        entry=parse_condition(str(values["entry"])),
        exit=parse_condition(str(values["exit"])),
        stop_loss_vol=float(values["stop_loss_vol"]),
        take_profit_vol=float(values["take_profit_vol"]),
        max_size=float(values["max_size"]),
        max_hold_bars=int(values["max_hold_bars"]),
        invalidation=str(values["invalidation"]),
    )


def evaluate_signal(
    playbook: Playbook, row: Mapping[str, float], in_position: bool, bars_held: int
) -> Signal:
    """Entry and exit signals from one bar's features. Stops and
    take-profits are price levels, enforced by the execution layer."""
    if in_position:
        if playbook.max_hold_bars and bars_held >= playbook.max_hold_bars:
            return Signal(False, True, f"max_hold_bars {playbook.max_hold_bars} reached")
        if playbook.exit.holds(row):
            return Signal(False, True, f"{playbook.state} exit condition")
        return Signal(False, False, "hold")
    if playbook.max_size > 0 and playbook.entry.holds(row):
        return Signal(True, False, f"{playbook.state} entry condition")
    return Signal(False, False, "no entry")
