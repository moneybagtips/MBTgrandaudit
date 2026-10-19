"""Market-implied scoreline distribution from the odds that bookmakers do quote.

Bookmakers price 1X2 and a main Over/Under line for almost every game, but rarely
GG2+ (and not always BTTS). Here, the margin is removed from the quoted odds and
the home/away expected-goal rates of a Dixon-Coles score matrix are solved so the
matrix reproduces those probabilities. Any other market (BTTS, GG2+, other
O/U lines) can then be read off that matrix: the market's view of an unquoted market.

When you have entered prices for BTTS, GG2+ or other O/U lines (odds form / `odds add`),
they are matched too. A two-sided pair (e.g. GG2+ Yes and No) is de-margined on its
own; a single side uses the power-method exponent of the game's 1X2 book.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .markets import demargin, implied_from_odds, markets
from .models.base import dc_tau_matrix, poisson_matrix

RHO = -0.05   # typical low-score dependence; markets don't reveal it on their own


def _power_k(odds: list[float]) -> float:
    """Exponent k of the power method for one book: sum (1/o_i)^k = 1."""
    inv = np.array([1 / o for o in odds], dtype=float)
    if inv.sum() <= 1.0:
        return 1.0
    lo, hi = 1.0, 3.0
    for _ in range(60):
        k = (lo + hi) / 2
        lo, hi = (k, hi) if (inv ** k).sum() > 1 else (lo, k)
    return (lo + hi) / 2


def extra_targets(extra: dict | None, k: float, main_line=None) -> dict:
    """Probabilities to match from entered prices: btts_yes, gg2_yes and over_<line>."""
    out: dict = {}
    if not extra:
        return out
    single = lambda o: min((1 / o) ** k, 0.99)
    pairs = [("btts_yes", "btts_no"), ("gg2_yes", "gg2_no")]
    lines = {key.split("_", 1)[1] for key in extra if key.startswith(("over_", "under_"))}
    pairs += [(f"over_{ln}", f"under_{ln}") for ln in sorted(lines) if main_line is None
              or float(ln) != float(main_line)]
    for yes, no in pairs:
        y, n = extra.get(yes), extra.get(no)
        if y and n:
            out[yes] = demargin([y, n])[0]
        elif y:
            out[yes] = single(y)
        elif n:
            out[yes] = 1 - single(n)
    return out


def implied_matrix(odds: dict, max_goals: int = 10, rho: float = RHO, extra: dict | None = None):
    """Return (matrix, info) or None if the odds are insufficient.

    odds: decimal odds with keys H, D, A and optionally line, over, under.
    extra: entered decimal odds by market key (btts_yes, gg2_no, over_3.5, ...), matched too.
    info: fitted expected goals, the probabilities matched and the fit error.
    """
    target = implied_from_odds(odds)
    if not all(k in target for k in ("home", "draw", "away")):
        return None
    line = odds.get("line")
    over_key = f"over_{line}" if line is not None else None
    if over_key not in target:
        over_key = None
    more = extra_targets(extra, _power_k([odds["H"], odds["D"], odds["A"]]), line if over_key else None)
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
        for key, p in more.items():
            if key.startswith("over_"):
                r.append(m[(i + j) > float(key[5:])].sum() - p)
            else:   # btts_yes / gg2_yes: both sides score at least n
                n = 1 if key.startswith("btts") else 2
                r.append(m[n:, n:].sum() - p)
        return np.array(r)

    fit = least_squares(resid, np.log([1.4, 1.1]), bounds=(np.log(0.05), np.log(6.0)))
    m = matrix(fit.x)
    lh, la = np.exp(fit.x)
    return m, {"xg_home": float(lh), "xg_away": float(la),
               "matched": ["1X2"] + ([f"O/U {line}"] if over_key else [])
               + [{"btts_yes": "BTTS", "gg2_yes": "GG2+"}.get(k, "O/U " + k[5:]) for k in more],
               "fit_error": float(np.sqrt(np.mean(fit.fun ** 2)))}


def implied_markets(odds: dict, ou_lines=(0.5, 1.5, 2.5, 3.5, 4.5), extra: dict | None = None) -> dict | None:
    res = implied_matrix(odds, extra=extra)
    if res is None:
        return None
    m, info = res
    out = {k: v for k, v in markets(m, ou_lines).items() if k != "top_scores"}
    return out | {"info": info}
