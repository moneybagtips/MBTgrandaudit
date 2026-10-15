"""Walk-forward backtest: refit weekly, predict the following week, score every market."""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import LEAGUES, WEIGHTS_FILE, Settings
from .data import footballdata as fd
from .markets import markets
from .models import MatchContext, build_models, ensemble, load_weights
from .features import extract as extract_features
from .implied import implied_matrix
from .models.form import team_matches_from_history
from .storage import model_version

MARKET_KEYS = ["1x2", "btts", "gg2", "o1.5", "o2.5", "o3.5"]


def _outcome_probs(mk: dict, hg: int, ag: int) -> dict:
    """Probability each model gave to what actually happened, per market."""
    res = "home" if hg > ag else "draw" if hg == ag else "away"
    t = hg + ag
    return {
        "1x2": mk[res],
        "btts": mk["btts_yes"] if hg > 0 and ag > 0 else mk["btts_no"],
        "gg2": mk["gg2_yes"] if hg > 1 and ag > 1 else mk["gg2_no"],
        "o1.5": mk["over_1.5"] if t > 1.5 else mk["under_1.5"],
        "o2.5": mk["over_2.5"] if t > 2.5 else mk["under_2.5"],
        "o3.5": mk["over_3.5"] if t > 3.5 else mk["under_3.5"],
    }


def _rps(mk: dict, hg: int, ag: int) -> float:
    p = np.array([mk["home"], mk["draw"], mk["away"]])
    o = np.zeros(3)
    o[0 if hg > ag else 1 if hg == ag else 2] = 1
    return float(((np.cumsum(p) - np.cumsum(o))[:2] ** 2).sum() / 2)


def run(league_key: str, test_seasons: int = 1, settings: Settings | None = None,
        today: date | None = None, log=print, offline: bool = False, store=None) -> dict:
    """If ``store`` (a storage.Store) is given, every prediction is also written to the
    prediction log as source='backtest', with odds and result, for later training."""
    s = settings or Settings()
    lg = LEAGUES[league_key]
    today = today or date.today()
    cur = fd.season_start_year(today)
    divs = [lg.fd_code] + ([lg.lower_fd_code] if lg.lower_fd_code else [])
    hist = fd.load_history(divs, s.seasons + test_seasons, today, offline)
    hist = hist[hist["season"] < cur]                      # completed seasons only
    test = hist[(hist["div"] == lg.fd_code) & (hist["season"] >= cur - test_seasons)].dropna(subset=["hg", "ag"])
    if test.empty:
        raise RuntimeError("no completed seasons to test on")
    log(f"[{lg.key}] backtesting {len(test)} matches, seasons "
        f"{fd.season_code(cur - test_seasons)}..{fd.season_code(cur - 1)}, weekly refits")

    weights = load_weights(s, league_key)
    version = model_version(s) if store is not None else None
    week = test["date"].dt.to_period("W-MON")
    rows = []          # per match, per model: outcome probabilities
    names = [m.name for m in build_models(s)] + ["market_view", "ensemble"]
    for i, (_, wk) in enumerate(test.groupby(week)):
        as_of = wk["date"].min().normalize()
        first = fd.season_start_year(as_of.date()) - s.seasons + 1
        train = hist[(hist["date"] < as_of) & (hist["season"] >= first)]
        models = [m.fit(train, as_of, lg.fd_code) for m in build_models(s)]
        top = train[(train["div"] == lg.fd_code) & (train["date"] >= as_of - pd.Timedelta(days=400))]
        lhg, lag = float(top["hg"].mean()), float(top["ag"].mean())
        dc = models[0]
        for r in wk.itertuples():
            fh = team_matches_from_history(train, r.home)
            fa = team_matches_from_history(train, r.away)
            h2h = fh[fh["opponent"] == r.away]
            ctx = MatchContext(r.home, r.away, as_of, fh, fa, h2h, lhg, lag)
            ctx.prior = dc.expected_goals(ctx)
            mats = {m.name: m.predict(ctx, s.max_goals) for m in models}
            # market view from the odds available before kick-off (never closing odds)
            pre = {"H": r.odds_h, "D": r.odds_d, "A": r.odds_a, "line": 2.5, "over": r.odds_o25,
                   "under": r.odds_u25}
            pre = {k: (None if pd.isna(v) else float(v)) for k, v in pre.items()}
            if pre["over"] is None or pre["under"] is None:
                pre["line"] = None
            implied = implied_matrix(pre, s.max_goals) if pre["H"] and pre["D"] and pre["A"] else None
            mats["market_view"] = implied[0] if implied else None
            if mats.get("dixon_coles") is None:
                continue
            mats["ensemble"] = ensemble(mats, weights)
            hg, ag = int(r.hg), int(r.ag)
            if store is not None:
                _store_backtest(store, version, league_key, r, as_of, ctx, models, mats, weights, s)
            rec = {"date": r.date, "home": r.home, "away": r.away, "hg": hg, "ag": ag}
            for n in names:
                mat = mats.get(n)
                mk = markets(mat if mat is not None else mats["dixon_coles"], s.ou_lines)
                rec[n] = {**_outcome_probs(mk, hg, ag), "rps": _rps(mk, hg, ag),
                          "hit": max(("home", "draw", "away"), key=mk.get) ==
                          ("home" if hg > ag else "draw" if hg == ag else "away"),
                          "available": mat is not None}
            # bookmaker benchmark (closing-ish average odds from football-data)
            if pd.notna(r.odds_h) and pd.notna(r.odds_d) and pd.notna(r.odds_a):
                inv = np.array([1 / r.odds_h, 1 / r.odds_d, 1 / r.odds_a])
                inv /= inv.sum()
                res = 0 if hg > ag else 1 if hg == ag else 2
                book = {"1x2": inv[res], "rps": _rps(dict(zip(("home", "draw", "away"), inv)), hg, ag),
                        "hit": int(np.argmax(inv)) == res}
                if pd.notna(r.odds_o25) and pd.notna(r.odds_u25):
                    o, u = 1 / r.odds_o25, 1 / r.odds_u25
                    book["o2.5"] = (o if hg + ag > 2.5 else u) / (o + u)
                rec["bookmaker"] = book
            rows.append(rec)
        if i % 8 == 0:
            log(f"  week of {as_of.date()}: {len(rows)} matches scored")
            if store is not None:
                store.commit()
    if store is not None:
        store.commit()
    return {"league": league_key, "rows": rows, "models": names}


def _store_backtest(store, version, league_key, r, as_of, ctx, models, mats, weights, s):
    event_id = f"fd:{r.div}:{r.date:%Y%m%d}:{r.home}:{r.away}"
    model_mats = {k: v for k, v in mats.items() if k != "ensemble" and v is not None}
    used = {k: weights.get(k, 0.0) for k in model_mats if weights.get(k, 0.0) > 0}
    tot = sum(used.values()) or 1.0
    store.add_prediction(
        event_id=event_id, league=league_key, home=r.home, away=r.away, model_home=r.home,
        model_away=r.away, kickoff=r.date, predicted_at=as_of, version=version, source="backtest",
        matrix=mats["ensemble"], markets=markets(mats["ensemble"], s.ou_lines),
        weights={k: v / tot for k, v in used.items()}, features=extract_features(models, ctx),
        model_matrices=model_mats)
    for kind, cols, when in (("prediction", ("odds_h", "odds_d", "odds_a", "odds_o25", "odds_u25"), as_of),
                             ("closing", ("odds_ch", "odds_cd", "odds_ca", "odds_co25", "odds_cu25"), r.date)):
        vals = [getattr(r, c) for c in cols]
        if all(pd.notna(v) for v in vals[:3]):
            odds = dict(zip(("H", "D", "A"), vals[:3])) | {"source": "football-data avg"}
            if pd.notna(vals[3]) and pd.notna(vals[4]):
                odds |= {"line": 2.5, "over": vals[3], "under": vals[4]}
            store.add_odds(event_id, when, kind, odds)
    store.add_result(event_id, league_key, int(r.hg), int(r.ag), "football-data")


def summarise(result: dict) -> pd.DataFrame:
    rows = result["rows"]
    out = {}
    for n in result["models"] + ["bookmaker"]:
        recs = [r[n] for r in rows if n in r]
        if not recs:
            continue
        d = {"matches": len(recs)}
        for k in MARKET_KEYS:
            vals = [r[k] for r in recs if k in r]
            if vals:
                d[f"logloss_{k}"] = float(-np.mean(np.log(np.clip(vals, 1e-9, 1))))
        d["rps_1x2"] = float(np.mean([r["rps"] for r in recs]))
        d["acc_1x2"] = float(np.mean([r["hit"] for r in recs]))
        out[n] = d
    return pd.DataFrame(out).T


def optimise_weights(result: dict, settings: Settings | None = None) -> dict:
    """Ensemble weights minimising combined log loss across all target markets."""
    s = settings or Settings()
    names = [n for n in result["models"] if n != "ensemble"]
    P = np.array([[[r[n][k] for k in MARKET_KEYS] for n in names] for r in result["rows"]])  # (N, M, K)

    def loss(z):
        w = np.exp(z - z.max())
        w /= w.sum()
        mix = np.einsum("nmk,m->nk", P, w)
        return -np.log(np.clip(mix, 1e-9, 1)).mean()

    z0 = np.log([max(s.default_weights.get(n, 0.05), 1e-3) for n in names])
    z = minimize(loss, z0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-7}).x
    w = np.exp(z - z.max())
    w /= w.sum()
    # in-sample optimum overfits; blend 70/30 with the defaults to keep model diversity
    prior = np.array([s.default_weights.get(n, 0.05) for n in names])
    w = 0.7 * w + 0.3 * prior / prior.sum()
    w /= w.sum()
    return {n: round(float(v), 4) for n, v in zip(names, w)}


def save_weights(league_key: str, weights: dict) -> None:
    saved = {}
    if WEIGHTS_FILE.exists():
        saved = json.loads(WEIGHTS_FILE.read_text())
    saved[league_key] = weights
    WEIGHTS_FILE.write_text(json.dumps(saved, indent=2))
