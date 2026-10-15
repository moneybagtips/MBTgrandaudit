"""Market-implied scoreline distribution from the odds that bookmakers do quote.

Bookmakers price 1X2 and a main Over/Under line for almost every game, but rarely
GG2+ (and not always BTTS). Here, the margin is removed from the quoted odds and
the home/away expected-goal rates of a Dixon-Coles score matrix are solved so the
matrix reproduces those probabilities. Any other market (BTTS, GG2+, other
O/U lines) can then be read off that matrix: the market's view of an unquoted market.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .markets import implied_from_odds, markets
from .models.base import dc_tau_matrix, poisson_matrix

RHO = -0.05   # typical low-score dependence; markets don't reveal it on their own


def implied_matrix(odds: dict, max_goals: int = 10, rho: float = RHO):
    """Return (matrix, info) or None if the odds are insufficient.

    odds: decimal odds with keys H, D, A and optionally line, over, under.
    info: fitted expected goals, the probabilities matched and the fit error.
    """
    target = implied_from_odds(odds)
    if not all(k in target for k in ("home", "draw", "away")):
        return None
    line = odds.get("line")
    over_key = f"over_{line}" if line is not None else None
    if over_key not in target:
        over_key = None
    g = np.arange(max_goals + 1)
    i, j = np.indices((max_goals + 1, max_goals + 1))

    def matrix(x):
        lh, la = np.exp(x)
        return dc_tau_matrix(poisson_matrix(lh, la, max_goals), lh, la, rho)

    def resid(x):
        m = matrix(x)
        r = [m[i > j].sum() - target["home"], m[i < j].sum() - target["away"],
             m[i == j].sum() - target["draw"]]
        if over_key:
            r.append(m[(i + j) > line].sum() - target[over_key])
        return np.array(r)

    fit = least_squares(resid, np.log([1.4, 1.1]), bounds=(np.log(0.05), np.log(6.0)))
    m = matrix(fit.x)
    lh, la = np.exp(fit.x)
    return m, {"xg_home": float(lh), "xg_away": float(la),
               "matched": ["1X2"] + ([f"O/U {line}"] if over_key else []),
               "fit_error": float(np.sqrt(np.mean(fit.fun ** 2)))}


def implied_markets(odds: dict, ou_lines=(0.5, 1.5, 2.5, 3.5, 4.5)) -> dict | None:
    res = implied_matrix(odds)
    if res is None:
        return None
    m, info = res
    out = {k: v for k, v in markets(m, ou_lines).items() if k != "top_scores"}
    return out | {"info": info}
