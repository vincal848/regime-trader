"""Position size from edge and confidence (spec §9).

    size = min(kelly / 4, state cap, playbook max) * P(active) * (1 - H / ln K) * multiplier

where H is the entropy of the filtered probabilities (0 when certain, ln K
when uniform) and the multiplier comes from the switching rules (0, 0.5 or
1). The result is a fraction of equity, never negative. The hard limits in
`risk` are checked separately on every order: sizing proposes, risk
disposes.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
KELLY_FRACTION = 0.25


def entropy(probabilities: FloatArray) -> float:
    positive = probabilities[probabilities > 0]
    return float(-(positive * np.log(positive)).sum())


def kelly_fraction(returns: FloatArray) -> float:
    """Full-Kelly fraction mean / variance of per-bar strategy returns (0 if no edge)."""
    if len(returns) < 2:
        return 0.0
    variance = float(returns.var(ddof=1))
    return max(float(returns.mean()) / variance, 0.0) if variance > 0 else 0.0


def target_fraction(
    probabilities: FloatArray, active: int, kelly: float, cap: float, playbook_max: float, multiplier: float
) -> float:
    certainty = 1.0 - entropy(probabilities) / float(np.log(len(probabilities)))
    size = (
        min(KELLY_FRACTION * kelly, cap, playbook_max) * float(probabilities[active]) * certainty * multiplier
    )
    return max(size, 0.0)
