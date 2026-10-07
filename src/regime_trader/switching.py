"""Which playbook is active, decided by code with hysteresis (spec §8).

`step_switch` runs once per bar, with that bar's filtered and next-state
probabilities:

1. A challenger takes over only above `takeover_prob` ...
2. ... and only after leading for `hold_bars` consecutive bars ...
3. ... and never inside the `cooldown_bars` after the previous switch.
4. If P(next bar in a STRESS or CRASH state) > `early_cut_prob`, size is
   multiplied by `early_cut_factor` before any switch happens.
5. If the top two probabilities are within `uncertainty_gap`, size is zero.
6. Unhealthy input (a model error, NaN, stale or failed data) means size
   zero, and the challenger's count restarts.

Before any state has ever taken over there is no active regime, and size is
zero. Every rule that fires is recorded in `Regime.reasons`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
DANGER_PREFIXES = ("STRESS", "CRASH")


@dataclass(frozen=True)
class SwitchConfig:
    takeover_prob: float = 0.70
    hold_bars: int = 3
    cooldown_bars: int = 7
    early_cut_prob: float = 0.25
    early_cut_factor: float = 0.5
    uncertainty_gap: float = 0.15


@dataclass(frozen=True)
class SwitchState:
    active: str | None  # None until the first takeover
    challenger: str | None
    streak: int
    cooldown: int


INITIAL = SwitchState(active=None, challenger=None, streak=0, cooldown=0)


@dataclass(frozen=True)
class Regime:
    active: str | None
    switched: bool
    size_multiplier: float
    reasons: tuple[str, ...]


def _is_danger(label: str) -> bool:
    return label.startswith(DANGER_PREFIXES)


def step_switch(
    state: SwitchState,
    labels: tuple[str, ...],
    filtered: FloatArray,
    next_state: FloatArray,
    healthy: bool,
    config: SwitchConfig,
) -> tuple[SwitchState, Regime]:
    state = replace(state, cooldown=max(state.cooldown - 1, 0))
    if not healthy:
        reasons = ("unhealthy input (model error, stale or invalid data): flat",)
        return replace(state, challenger=None, streak=0), Regime(state.active, False, 0.0, reasons)

    order = np.argsort(filtered)[::-1]
    leader, leader_p, runner_up_p = labels[order[0]], float(filtered[order[0]]), float(filtered[order[1]])
    reasons: list[str] = []
    switched = False

    if leader != state.active and leader_p > config.takeover_prob:
        streak = state.streak + 1 if state.challenger == leader else 1
        state = replace(state, challenger=leader, streak=streak)
        if streak >= config.hold_bars and state.cooldown == 0:
            reasons.append(f"switch {state.active} -> {leader} (p={leader_p:.3f}, led {streak} bars)")
            state = SwitchState(active=leader, challenger=None, streak=0, cooldown=config.cooldown_bars)
            switched = True
    else:
        state = replace(state, challenger=None, streak=0)

    multiplier = 0.0 if state.active is None else 1.0
    if state.active is None:
        reasons.append("no regime has taken over yet: flat")
    danger_next = float(sum(p for label, p in zip(labels, next_state, strict=True) if _is_danger(label)))
    if danger_next > config.early_cut_prob:
        multiplier *= config.early_cut_factor
        reasons.append(
            f"next-bar stress/crash probability {danger_next:.3f} > {config.early_cut_prob}: size cut"
        )
    if leader_p - runner_up_p < config.uncertainty_gap:
        multiplier = 0.0
        reasons.append(f"uncertain: top two states within {leader_p - runner_up_p:.3f}: flat")
    return state, Regime(state.active, switched, multiplier, tuple(reasons))
