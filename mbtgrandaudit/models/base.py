"""Model interface and shared scoreline-distribution helpers."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import poisson

GOALS = np.arange(11)


@dataclass
class MatchContext:
    """Everything a model may use to price one fixture."""
    home: str
    away: str
    as_of: pd.Timestamp
    home_form: pd.DataFrame = field(default_factory=pd.DataFrame)  # all-competition matches
    away_form: pd.DataFrame = field(default_factory=pd.DataFrame)
    h2h: pd.DataFrame = field(default_factory=pd.DataFrame)        # meetings, from home team's view
    league_home_goals: float = 1.5
    league_away_goals: float = 1.2
    prior: tuple[float, float] | None = None  # (lambda_home, lambda_away) from the main strength model


class Model:
    name = "model"
    description = ""

    def fit(self, hist: pd.DataFrame, as_of: pd.Timestamp, target_div: str) -> "Model":
        return self

    def predict(self, ctx: MatchContext, max_goals: int = 10) -> np.ndarray | None:
        raise NotImplementedError

    def expected_goals(self, ctx: MatchContext) -> tuple[float, float] | None:
        return None


def poisson_matrix(lh: float, la: float, max_goals: int = 10) -> np.ndarray:
    g = np.arange(max_goals + 1)
    return np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))


def dc_tau_matrix(m: np.ndarray, lh: float, la: float, rho: float) -> np.ndarray:
    m = m.copy()
    m[0, 0] *= max(1 - lh * la * rho, 0)
    m[0, 1] *= max(1 + lh * rho, 0)
    m[1, 0] *= max(1 + la * rho, 0)
    m[1, 1] *= max(1 - rho, 0)
    return m / m.sum()


def nb_pmf(k: np.ndarray, mu: float | np.ndarray, r: float) -> np.ndarray:
    """Negative binomial (NB2) pmf with mean mu and size r (variance mu + mu^2/r)."""
    return np.exp(gammaln(k + r) - gammaln(r) - gammaln(k + 1)
                  + r * np.log(r / (r + mu)) + k * np.log(mu / (r + mu)))


def bivariate_poisson_matrix(l1: float, l2: float, l3: float, max_goals: int = 10) -> np.ndarray:
    """Karlis & Ntzoufras bivariate Poisson: X = A + C, Y = B + C, C ~ Poisson(l3)."""
    g = np.arange(max_goals + 1)
    pa, pb, pc = poisson.pmf(g, l1), poisson.pmf(g, l2), poisson.pmf(g, l3)
    m = np.zeros((max_goals + 1, max_goals + 1))
    for k in range(max_goals + 1):
        m[k:, k:] += pc[k] * np.outer(pa[: max_goals + 1 - k], pb[: max_goals + 1 - k])
    return m


def time_weights(dates: pd.Series, as_of: pd.Timestamp, xi: float) -> np.ndarray:
    days = (as_of - dates).dt.days.to_numpy(dtype=float)
    return np.exp(-xi * np.clip(days, 0, None))
