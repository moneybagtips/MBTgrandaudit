"""Sequential rating systems (Elo, Pi-ratings) mapped to scoreline distributions.

Ratings are updated match by match through the full history. The pre-match
rating gap of every recent match in the target division is then regressed on
the goals each side scored (Poisson regression), turning a rating gap into a
pair of expected-goal rates and therefore a full score matrix.
"""
from __future__ import annotations


import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base import MatchContext, Model, dc_tau_matrix, poisson_matrix, time_weights


class GoalMapper:
    """log E[home goals] = a0 + a1*x ; log E[away goals] = b0 + b1*x (time-weighted)."""

    def fit(self, x, yh, ya, w):
        def make(y):
            def f(p):
                eta = p[0] + p[1] * x
                mu = np.exp(eta)
                g = w * (y - mu)
                return -(w * (y * eta - mu)).sum(), -np.array([g.sum(), (g * x).sum()])
            return minimize(f, [np.log(max(np.average(y, weights=w), .1)), 0.0], jac=True).x
        self.a, self.b = make(yh), make(ya)
        return self

    def __call__(self, x):
        return float(np.exp(self.a[0] + self.a[1] * x)), float(np.exp(self.b[0] + self.b[1] * x))


class _RatingModel(Model):
    rho = -0.06

    def __init__(self, xi: float = 0.0019):
        self.xi = xi
        self.fitted = False

    def _initial(self, div: str, tiers: dict) -> float:
        raise NotImplementedError

    def _feature(self, home: str, away: str) -> float:
        raise NotImplementedError

    def _update(self, home: str, away: str, hg: int, ag: int) -> None:
        raise NotImplementedError

    def fit(self, hist, as_of, target_div):
        h = hist[hist["date"] < as_of].dropna(subset=["hg", "ag"]).sort_values("date")
        if len(h) < 50:
            return self
        # tier order: target division first, lower tiers get lower starting ratings
        divs = list(dict.fromkeys([target_div] + sorted(h["div"].unique())))
        tiers = {d: i for i, d in enumerate(divs)}
        self._reset()
        feats, rows = [], []
        for r in h.itertuples(index=False):
            for t in (r.home, r.away):
                if t not in self.seen:
                    self.seen.add(t)
                    self._init_team(t, self._initial(r.div, tiers))
            if r.div == target_div:
                feats.append(self._feature(r.home, r.away))
                rows.append(r)
            self._update(r.home, r.away, int(r.hg), int(r.ag))
        sub = pd.DataFrame(rows)
        # skip the first season (burn-in) when fitting the goal mapping
        keep = (sub["date"] >= sub["date"].min() + pd.Timedelta(days=300)).to_numpy()
        if keep.sum() < 100:
            keep = np.ones(len(sub), dtype=bool)
        x = np.array(feats)[keep]
        sub = sub[keep]
        w = time_weights(sub["date"], as_of, self.xi)
        self.mapper = GoalMapper().fit(x, sub["hg"].to_numpy(float), sub["ag"].to_numpy(float), w)
        self.fitted = True
        return self

    def expected_goals(self, ctx: MatchContext):
        if not self.fitted:
            return None
        for t in (ctx.home, ctx.away):
            if t not in self.seen:
                self._init_team(t, self._newcomer())
        return self.mapper(self._feature(ctx.home, ctx.away))

    def predict(self, ctx, max_goals=10):
        xg = self.expected_goals(ctx)
        if xg is None:
            return None
        return dc_tau_matrix(poisson_matrix(*xg, max_goals), *xg, self.rho)


class Elo(_RatingModel):
    """Goal-difference weighted Elo (World Football Elo style), with home advantage."""
    name = "elo"
    description = "Elo ratings (margin-of-victory weighted) mapped to goals by Poisson regression"

    K, HFA = 20.0, 65.0

    def _reset(self):
        self.r, self.seen = {}, set()

    def _init_team(self, t, v):
        self.r[t] = v

    def _initial(self, div, tiers):
        return 1500.0 - 120.0 * tiers[div]

    def _newcomer(self):
        return float(np.percentile(list(self.r.values()), 20)) if self.r else 1450.0

    def _feature(self, home, away):
        return (self.r[home] + self.HFA - self.r[away]) / 400.0

    def _update(self, home, away, hg, ag):
        d = self.r[home] + self.HFA - self.r[away]
        exp_h = 1 / (1 + 10 ** (-d / 400))
        res = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        gd = abs(hg - ag)
        g = 1 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
        delta = self.K * g * (res - exp_h)
        self.r[home] += delta
        self.r[away] -= delta

    def ratings(self):
        return pd.Series(self.r).sort_values(ascending=False)


class PiRating(_RatingModel):
    """Pi-ratings (Constantinou & Fenton 2013): separate home/away ratings in goal units."""
    name = "pi_rating"
    description = "Pi-ratings (separate home/away goal-based ratings) mapped to goals"

    LAM, GAM, B, C = 0.054, 0.79, 10.0, 3.0

    def _reset(self):
        self.rh, self.ra, self.seen = {}, {}, set()

    def _init_team(self, t, v):
        self.rh[t] = v
        self.ra[t] = v

    def _initial(self, div, tiers):
        return -0.35 * tiers[div]

    def _newcomer(self):
        return float(np.percentile(list(self.rh.values()), 20)) if self.rh else -0.3

    def _gd(self, r):
        return np.sign(r) * (self.B ** (abs(r) / self.C) - 1)

    def _feature(self, home, away):
        return self._gd(self.rh[home]) - self._gd(self.ra[away])

    def _update(self, home, away, hg, ag):
        err = (hg - ag) - self._feature(home, away)
        psi = np.sign(err) * self.C * np.log10(1 + abs(err))
        old_h, old_a = self.rh[home], self.ra[away]
        self.rh[home] += psi * self.LAM
        self.ra[home] += (self.rh[home] - old_h) * self.GAM
        self.ra[away] -= psi * self.LAM
        self.rh[away] += (self.ra[away] - old_a) * self.GAM
