"""Recent-form (all competitions) and head-to-head models."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import MatchContext, Model, dc_tau_matrix, poisson_matrix

FORM_COLUMNS = ["date", "competition", "venue", "opponent", "gf", "ga", "friendly"]


def team_matches_from_history(hist: pd.DataFrame, team: str) -> pd.DataFrame:
    """A team's league matches from football-data rows, in the same shape as ESPN schedules."""
    h = hist.dropna(subset=["hg", "ag"])
    home = h[h["home"] == team]
    away = h[h["away"] == team]
    out = pd.concat([
        pd.DataFrame({"date": home["date"], "competition": home["div"], "venue": "home",
                      "opponent": home["away"], "gf": home["hg"], "ga": home["ag"]}),
        pd.DataFrame({"date": away["date"], "competition": away["div"], "venue": "away",
                      "opponent": away["home"], "gf": away["ag"], "ga": away["hg"]}),
    ])
    out["friendly"] = False
    return out.sort_values("date").reset_index(drop=True)


def recent(form: pd.DataFrame, as_of: pd.Timestamp, n: int) -> pd.DataFrame:
    if form is None or form.empty:
        return pd.DataFrame(columns=FORM_COLUMNS)
    f = form[(form["date"] < as_of) & (~form["friendly"].astype(bool))]
    f = f[f["date"] >= as_of - pd.Timedelta(days=270)]   # ignore stale form
    return f.tail(n)


def _venue_factors(ctx: MatchContext) -> tuple[float, float, float]:
    mu = (ctx.league_home_goals + ctx.league_away_goals) / 2
    return mu, ctx.league_home_goals / mu, ctx.league_away_goals / mu


def form_rates(f: pd.DataFrame, ctx: MatchContext, half_life: float = 4.0,
               pseudo: float = 3.0) -> tuple[float, float, int]:
    """Venue-neutral, recency-weighted goals scored / conceded per match, shrunk to the league mean."""
    mu, hf, af = _venue_factors(ctx)
    if f.empty:
        return mu, mu, 0
    w = 0.5 ** (np.arange(len(f))[::-1] / half_life)
    home = (f["venue"] == "home").to_numpy()
    gf = f["gf"].to_numpy(float) / np.where(home, hf, af)
    ga = f["ga"].to_numpy(float) / np.where(home, af, hf)
    att = (w @ gf + pseudo * mu) / (w.sum() + pseudo)
    dfn = (w @ ga + pseudo * mu) / (w.sum() + pseudo)
    return float(att), float(dfn), len(f)


class FormModel(Model):
    name = "form"
    description = "Recency-weighted goals for/against over the last N matches in all competitions"

    def __init__(self, n_matches: int = 10):
        self.n = n_matches

    def expected_goals(self, ctx):
        fh, fa = recent(ctx.home_form, ctx.as_of, self.n), recent(ctx.away_form, ctx.as_of, self.n)
        if len(fh) < 3 or len(fa) < 3:
            return None
        mu, hf, af = _venue_factors(ctx)
        ah, dh, _ = form_rates(fh, ctx)
        aa, da, _ = form_rates(fa, ctx)
        return ah * da / mu * hf, aa * dh / mu * af

    def predict(self, ctx, max_goals=10):
        xg = self.expected_goals(ctx)
        if xg is None:
            return None
        return dc_tau_matrix(poisson_matrix(*xg, max_goals), *xg, -0.06)


class H2HModel(Model):
    """Head-to-head meetings (all competitions), shrunk toward the strength-model prior.

    lambda = (k * prior + sum(w_i * venue-adjusted goals_i)) / (k + sum(w_i)),
    with w_i halving every 3 years, so a long, lopsided rivalry moves the
    estimate but one old result barely does.
    """
    name = "h2h"
    description = "Head-to-head history, venue-adjusted, Bayesian-shrunk toward the strength model"

    def __init__(self, k: float = 4.0, half_life_years: float = 3.0):
        self.k, self.hl = k, half_life_years

    def expected_goals(self, ctx):
        h = ctx.h2h
        if h is None or h.empty or ctx.prior is None:
            return None
        h = h[h["date"] < ctx.as_of]
        if len(h) < 2:
            return None
        mu, hf, af = _venue_factors(ctx)
        years = (ctx.as_of - h["date"]).dt.days.to_numpy() / 365.25
        w = 0.5 ** (years / self.hl)
        home = (h["venue"] == "home").to_numpy()
        # convert each meeting into "if it had been played at this fixture's venues"
        gf = h["gf"].to_numpy(float) * np.where(home, 1, hf / af)
        ga = h["ga"].to_numpy(float) * np.where(home, 1, af / hf)
        ph, pa = ctx.prior
        return ((self.k * ph + w @ gf) / (self.k + w.sum()),
                (self.k * pa + w @ ga) / (self.k + w.sum()))

    def predict(self, ctx, max_goals=10):
        xg = self.expected_goals(ctx)
        if xg is None:
            return None
        return dc_tau_matrix(poisson_matrix(*xg, max_goals), *xg, -0.06)
