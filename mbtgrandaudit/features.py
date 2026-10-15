"""Numeric match features, frozen at prediction time for later model training."""
from __future__ import annotations

import numpy as np

from .models.base import MatchContext
from .models.form import form_rates, recent


def _rest_days(form, as_of) -> float | None:
    if form is None or form.empty:
        return None
    past = form[form["date"] < as_of]
    return float((as_of - past["date"].max()).days) if len(past) else None


def _form_stats(prefix: str, form, ctx: MatchContext) -> dict:
    f = recent(form, ctx.as_of, 10)
    att, dfn, n = form_rates(f, ctx)
    out = {f"{prefix}_form_n": n, f"{prefix}_form_att": att, f"{prefix}_form_def": dfn,
           f"{prefix}_rest_days": _rest_days(form, ctx.as_of)}
    if n:
        gf, ga = f["gf"].to_numpy(float), f["ga"].to_numpy(float)
        out |= {f"{prefix}_form_ppg": float(np.mean(np.where(gf > ga, 3, np.where(gf == ga, 1, 0)))),
                f"{prefix}_form_btts": float(np.mean((gf > 0) & (ga > 0))),
                f"{prefix}_form_o25": float(np.mean(gf + ga > 2.5))}
    return out


def extract(models, ctx: MatchContext) -> dict:
    """Ratings from every fitted model plus form, rest and head-to-head numbers."""
    by = {m.name: m for m in models}
    feats: dict = {"league_home_goals": ctx.league_home_goals, "league_away_goals": ctx.league_away_goals}
    dc = by.get("dixon_coles")
    if dc is not None and getattr(dc, "fitted", False):
        for side, team in (("home", ctx.home), ("away", ctx.away)):
            att, dfn = dc._team(team)
            feats[f"{side}_dc_att"], feats[f"{side}_dc_def"] = float(att), float(dfn)
        feats["dc_home_adv"] = float(dc.home_adv)
    for name in ("dixon_coles", "xg_poisson", "elo", "pi_rating"):
        m = by.get(name)
        xg = m.expected_goals(ctx) if m is not None else None
        if xg:
            feats[f"{name}_xg_home"], feats[f"{name}_xg_away"] = xg
    elo = by.get("elo")
    if elo is not None and getattr(elo, "fitted", False):
        feats["home_elo"], feats["away_elo"] = elo.r.get(ctx.home), elo.r.get(ctx.away)
    pi = by.get("pi_rating")
    if pi is not None and getattr(pi, "fitted", False):
        feats["home_pi_home"], feats["away_pi_away"] = pi.rh.get(ctx.home), pi.ra.get(ctx.away)
    feats |= _form_stats("home", ctx.home_form, ctx)
    feats |= _form_stats("away", ctx.away_form, ctx)
    h = ctx.h2h
    h = h[h["date"] < ctx.as_of] if h is not None and not h.empty else h
    n = 0 if h is None or h.empty else len(h)
    feats["h2h_n"] = n
    if n:
        gf, ga = h["gf"].to_numpy(float), h["ga"].to_numpy(float)
        feats |= {"h2h_home_ppg": float(np.mean(np.where(gf > ga, 3, np.where(gf == ga, 1, 0)))),
                  "h2h_goals": float(np.mean(gf + ga)), "h2h_draw_rate": float(np.mean(gf == ga))}
    return {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else
                float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
            for k, v in feats.items()}
