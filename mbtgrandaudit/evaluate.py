"""Score the prediction log: accuracy per market, calibration, and betting value (ROI / CLV)."""
from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd

from .config import Settings, market_group
from .implied import implied_markets
from .markets import demargin, markets
from .storage import Store, unpack

# market key -> (probability key in markets(), outcome function of (hg, ag))
BINARY = {
    "home": lambda h, a: h > a, "draw": lambda h, a: h == a, "away": lambda h, a: h < a,
    "btts_yes": lambda h, a: (h > 0) & (a > 0), "gg2_yes": lambda h, a: (h > 1) & (a > 1),
    "over_1.5": lambda h, a: h + a > 1.5, "over_2.5": lambda h, a: h + a > 2.5,
    "over_3.5": lambda h, a: h + a > 3.5,
}
CALIBRATION_KEYS = ["home", "draw", "away", "btts_yes", "gg2_yes", "over_2.5"]
H2H_MARKETS = [("1X2", None), ("BTTS", "btts_yes"), ("GG2+", "gg2_yes"), ("O/U 1.5", "over_1.5"),
               ("O/U 2.5", "over_2.5"), ("O/U 3.5", "over_3.5")]


def outcome(key: str, hg, ag):
    """Did selection ``key`` (a markets() key such as 'gg2_no' or 'over_3.5') win?"""
    hg, ag = np.asarray(hg), np.asarray(ag)
    if key in ("home", "draw", "away"):
        return {"home": hg > ag, "draw": hg == ag, "away": hg < ag}[key]
    side, _, rest = key.partition("_")
    if side in ("btts", "gg2"):
        n = 1 if side == "btts" else 2
        yes = (hg >= n) & (ag >= n)
        return yes if rest == "yes" else ~yes
    line = float(rest)
    return (hg + ag > line) if side == "over" else (hg + ag < line)


def _odds_dict(row) -> dict:
    return {"H": row.get("home"), "D": row.get("draw"), "A": row.get("away"), "line": row.get("line"),
            "over": row.get("over_odds"), "under": row.get("under_odds")}


def market_implied(store: Store, preds: pd.DataFrame, columns) -> pd.DataFrame:
    """The market's view of every market, solved from each game's closing (else
    prediction-time) 1X2 + O/U odds. NaN where a game has no odds."""
    odds = store.odds_for(preds["event_id"]).sort_values("captured_at")
    out = pd.DataFrame(np.nan, index=preds.index, columns=columns)
    if odds.empty:
        return out
    best = odds[odds["kind"] == "closing"].groupby("event_id").last()
    best = best.combine_first(odds[odds["kind"] == "prediction"].groupby("event_id").last())
    for i, ev in zip(preds.index, preds["event_id"]):
        if ev in best.index:
            o = {k: (None if pd.isna(v) else float(v)) for k, v in _odds_dict(best.loc[ev]).items()}
            view = implied_markets(o)
            if view:
                view.pop("info")
                out.loc[i] = [view.get(c, np.nan) for c in columns]
    return out


def head_to_head(preds: pd.DataFrame, model: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Paired log-loss comparison, model vs market-implied, on the same games.
    diff < 0 means the model was better; |z| > 2 is unlikely to be luck."""
    hg, ag = preds["hg"].to_numpy(), preds["ag"].to_numpy()
    ok = market["home"].notna().to_numpy() & model["home"].notna().to_numpy()
    rows = {}
    for label, key in H2H_MARKETS:
        if key is None:
            res = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))[ok]
            pm = model.loc[ok, ["home", "draw", "away"]].to_numpy(float)[np.arange(ok.sum()), res]
            pk = market.loc[ok, ["home", "draw", "away"]].to_numpy(float)[np.arange(ok.sum()), res]
        else:
            y = outcome(key, hg, ag)[ok]
            pm = np.where(y, model.loc[ok, key], 1 - model.loc[ok, key]).astype(float)
            pk = np.where(y, market.loc[ok, key], 1 - market.loc[ok, key]).astype(float)
        d = -np.log(np.clip(pm, 1e-9, 1)) + np.log(np.clip(pk, 1e-9, 1))
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
        rows[label] = {"matches": int(len(d)), "model_logloss": float(-np.log(np.clip(pm, 1e-9, 1)).mean()),
                       "market_logloss": float(-np.log(np.clip(pk, 1e-9, 1)).mean()),
                       "diff": float(d.mean()), "z": float(d.mean() / se) if se else np.nan}
    return pd.DataFrame(rows).T


def load(store: Store, **filters) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Settled predictions with market probabilities for the ensemble and each model."""
    preds = store.settled(**filters)
    if preds.empty:
        return preds, {}
    preds["hg"], preds["ag"] = preds["home_goals"].to_numpy(), preds["away_goals"].to_numpy()
    ens = _market_frame([unpack(b) for b in preds["matrix"]])
    ens.index = preds.index
    probs = {"ensemble": ens}
    mo = store.model_matrices(preds["id"])
    for name, g in mo.groupby("model"):
        g = g.drop_duplicates("prediction_id").set_index("prediction_id").reindex(preds["id"])
        ok = g["matrix"].notna().to_numpy()
        f = pd.DataFrame(np.nan, index=preds.index, columns=ens.columns)
        if ok.any():
            f.loc[ok] = _market_frame([unpack(b) for b in g["matrix"][ok]]).to_numpy()
        probs[name] = f
    return preds, probs


def _market_frame(matrices) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in markets(m).items() if k != "top_scores"}
                         for m in matrices])


def _ll(p, y):
    p = np.clip(np.where(y, p, 1 - p), 1e-9, 1)
    return float(-np.mean(np.log(p)))


def scores(preds: pd.DataFrame, probs: dict) -> pd.DataFrame:
    hg, ag = preds["hg"].to_numpy(), preds["ag"].to_numpy()
    res = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    out = {}
    for name, f in probs.items():
        ok = f["home"].notna().to_numpy()
        if ok.sum() == 0:
            continue
        P = f.loc[ok, ["home", "draw", "away"]].to_numpy(float)
        r = res[ok]
        d = {"matches": int(ok.sum()),
             "logloss_1x2": float(-np.mean(np.log(np.clip(P[np.arange(len(r)), r], 1e-9, 1)))),
             "rps_1x2": float(np.mean(((np.cumsum(P, 1) - np.cumsum(np.eye(3)[r], 1))[:, :2] ** 2).sum(1) / 2)),
             "acc_1x2": float(np.mean(P.argmax(1) == r))}
        for key, label in (("btts_yes", "btts"), ("gg2_yes", "gg2"), ("over_1.5", "o1.5"),
                           ("over_2.5", "o2.5"), ("over_3.5", "o3.5")):
            y = BINARY[key](hg[ok], ag[ok])
            p = f.loc[ok, key].to_numpy(float)
            d[f"logloss_{label}"] = _ll(p, y)
            d[f"brier_{label}"] = float(np.mean((p - y) ** 2))
        out[name] = d
    return pd.DataFrame(out).T


def market_baseline(store: Store, preds: pd.DataFrame) -> dict:
    """Log loss of de-margined closing odds (or prediction-time odds if no closing snapshot)."""
    odds = store.odds_for(preds["event_id"])
    if odds.empty:
        return {}
    odds = odds.sort_values("captured_at")
    best = odds[odds["kind"] == "closing"].groupby("event_id").last()
    best = best.combine_first(odds[odds["kind"] == "prediction"].groupby("event_id").last())
    m = preds.join(best[["home", "draw", "away", "over_odds", "under_odds"]].add_prefix("o_"), on="event_id")
    ok = m[["o_home", "o_draw", "o_away"]].notna().all(1).to_numpy()
    if not ok.any():
        return {}
    inv = 1 / m.loc[ok, ["o_home", "o_draw", "o_away"]].to_numpy(float)
    inv = np.array([demargin(row) for row in 1 / inv])   # same power method as the market view
    hg, ag = m.loc[ok, "hg"].to_numpy(), m.loc[ok, "ag"].to_numpy()
    r = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    out = {"matches": int(ok.sum()),
           "logloss_1x2": float(-np.mean(np.log(inv[np.arange(len(r)), r]))),
           "acc_1x2": float(np.mean(inv.argmax(1) == r))}
    ou = m[["o_over_odds", "o_under_odds"]].notna().all(1).to_numpy()
    if ou.any():
        io, iu = 1 / m.loc[ou, "o_over_odds"].to_numpy(float), 1 / m.loc[ou, "o_under_odds"].to_numpy(float)
        p = np.array([demargin([a, b])[0] for a, b in zip(1 / io, 1 / iu)])
        out["logloss_o2.5"] = _ll(p, (m.loc[ou, "hg"] + m.loc[ou, "ag"]).to_numpy() > 2.5)
    return out


def calibration(preds: pd.DataFrame, f: pd.DataFrame, bins: int = 10) -> dict:
    """Per market: list of (mean predicted, observed frequency, count) per probability bin."""
    hg, ag = preds["hg"].to_numpy(), preds["ag"].to_numpy()
    out = {}
    for key in CALIBRATION_KEYS:
        p = f[key].to_numpy(float)
        y = BINARY[key](hg, ag).astype(float)
        edges = np.linspace(0, 1, bins + 1)
        idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        rows = [(float(p[idx == b].mean()), float(y[idx == b].mean()), int((idx == b).sum()))
                for b in range(bins) if (idx == b).sum() >= 5]
        out[key] = rows
    return out


def betting(store: Store, preds: pd.DataFrame, f: pd.DataFrame, settings: Settings | None = None
            ) -> pd.DataFrame:
    """Flat-stake value bets at the first price recorded for each selection, with a
    per-market minimum edge; CLV against the last closing price where one exists.

    Prices come from the bookmaker odds snapshots (1X2 + main O/U line) and from the
    market_odds table (any market, including odds entered by hand)."""
    s = settings or Settings()
    rows = []
    for _, r in store.odds_for(preds["event_id"]).iterrows():
        o = _odds_dict(r)
        for key, k in (("home", "H"), ("draw", "D"), ("away", "A")):
            rows.append((r["event_id"], key, r["captured_at"], r["kind"], o[k]))
        if pd.notna(o["line"]):
            rows.append((r["event_id"], f"over_{o['line']:g}", r["captured_at"], r["kind"], o["over"]))
            rows.append((r["event_id"], f"under_{o['line']:g}", r["captured_at"], r["kind"], o["under"]))
    mo = store.market_odds_for(preds["event_id"])
    rows += list(mo[["event_id", "market", "captured_at", "kind", "odds"]].itertuples(index=False, name=None))
    prices = pd.DataFrame(rows, columns=["event_id", "market", "captured_at", "kind", "odds"]).dropna()
    if prices.empty:
        return pd.DataFrame()
    kick = dict(zip(preds["event_id"], preds["kickoff"]))
    prices = prices[prices["captured_at"] <= prices["event_id"].map(kick)].sort_values("captured_at")
    take = prices[prices["kind"] != "closing"].groupby(["event_id", "market"])["odds"].first()
    close = prices[prices["kind"] == "closing"].groupby(["event_id", "market"])["odds"].last()
    pos = {ev: i for i, ev in enumerate(preds["event_id"])}
    bets = []
    for (ev, key), o in take.items():
        i = pos.get(ev)
        if i is None or key not in f.columns:
            continue
        p = f[key].iloc[i]
        group = market_group(key)
        if pd.isna(p) or p * o - 1 < s.min_edge.get(group, 0.03):
            continue
        won = bool(outcome(key, preds["hg"].iloc[i], preds["ag"].iloc[i]))
        c = close.get((ev, key), np.nan)
        bets.append({"event_id": ev, "kickoff": preds["kickoff"].iloc[i], "league": preds["league"].iloc[i],
                     "match": f"{preds['home'].iloc[i]} v {preds['away'].iloc[i]}", "market": group,
                     "selection": key, "prob": p, "odds": o, "edge": p * o - 1, "closing_odds": c,
                     "clv": o / c - 1 if pd.notna(c) else np.nan,
                     "won": won, "profit": (o - 1) if won else -1.0})
    return pd.DataFrame(bets)


def _summary(b: pd.DataFrame) -> dict:
    clv = b["clv"].dropna()
    return {"bets": len(b), "win_rate": float(b["won"].mean()),
            "avg_odds": float(b["odds"].mean()), "roi": float(b["profit"].mean()),
            "profit_units": float(b["profit"].sum()),
            # standard error of ROI: how far luck alone could plausibly move it
            "roi_se": float(b["profit"].std(ddof=1) / np.sqrt(len(b))) if len(b) > 1 else None,
            "with_closing": int(len(clv)), "avg_clv": float(clv.mean()) if len(clv) else None,
            "beat_close": float((clv > 0).mean()) if len(clv) else None}


def betting_summary(bets: pd.DataFrame) -> dict:
    if bets.empty:
        return {"bets": 0}
    out = _summary(bets)
    out["by_market"] = {g: _summary(b) for g, b in bets.groupby("market")}
    return out


# --------------------------------------------------------------------- report
def _svg_calibration(rows, title):
    W, H, P = 220, 220, 28
    x = lambda v: P + v * (W - 2 * P)
    y = lambda v: H - P - v * (H - 2 * P)
    pts = "".join(f'<circle cx="{x(a):.1f}" cy="{y(b):.1f}" r="{min(3 + n ** .5 / 3, 9):.1f}" '
                  f'fill="var(--bar)" fill-opacity=".75"><title>predicted {a:.1%} · observed {b:.1%} · n={n}'
                  f'</title></circle>' for a, b, n in rows)
    line = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b, _ in rows)
    ticks = "".join(f'<text x="{x(t):.0f}" y="{H - 8}" text-anchor="middle">{int(t * 100)}</text>'
                    f'<text x="{P - 6}" y="{y(t) + 4:.0f}" text-anchor="end">{int(t * 100)}</text>'
                    for t in (0, .5, 1))
    return (f'<div class="box"><h3>{html.escape(title)}</h3><svg viewBox="0 0 {W} {H}" width="100%" '
            f'style="max-width:260px;font-size:10px;fill:var(--mute)" role="img" '
            f'aria-label="Calibration plot for {html.escape(title)}">'
            f'<rect x="{P}" y="{P}" width="{W - 2 * P}" height="{H - 2 * P}" fill="none" stroke="var(--line)"/>'
            f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" stroke="var(--bar2)" stroke-dasharray="4 3"/>'
            f'<polyline points="{line}" fill="none" stroke="var(--bar)" stroke-width="1.5"/>{pts}{ticks}'
            f'</svg><div class="mute" style="font-size:11.5px">x: predicted % · y: actual %</div></div>')


def _table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in [""] + list(df.columns))
    body = ""
    for idx, row in df.iterrows():
        cells = "".join(f"<td>{fmt.get(c, '{:.4f}').format(v) if pd.notna(v) else '–'}</td>"
                        for c, v in row.items())
        body += f"<tr><td>{html.escape(str(idx))}</td>{cells}</tr>"
    return f'<div class="scroll"><table><tr>{head}</tr>{body}</table></div>'


def card_html(title: str, table: pd.DataFrame, base: dict, calib: dict, bet_sum: dict,
              note: str = "", vs_market: pd.DataFrame | None = None) -> str:
    cols = ["matches", "logloss_1x2", "rps_1x2", "acc_1x2", "logloss_btts", "logloss_gg2",
            "logloss_o2.5", "brier_o2.5"]
    t = table[[c for c in cols if c in table.columns]].copy()
    if base:
        t.loc["bookmaker"] = pd.Series(base)
    t = t.sort_values("logloss_1x2")
    fmt = {"matches": "{:.0f}", "acc_1x2": "{:.1%}"}
    plots = "".join(_svg_calibration(rows, k.replace("_", " ")) for k, rows in calib.items() if rows)
    if bet_sum.get("bets"):
        bm = pd.DataFrame({g: {"bets": v["bets"], "win rate": v["win_rate"], "avg odds": v["avg_odds"],
                               "ROI": v["roi"], "± (1 s.e.)": v["roi_se"], "avg CLV": v["avg_clv"],
                               "beat close": v["beat_close"]}
                           for g, v in {**bet_sum["by_market"], "all": bet_sum}.items()}).T
        bet = ('<p class="mute">Flat 1-unit stakes on every price where model edge ≥ the market\'s '
               'threshold. ROI within ±2 s.e. of zero is indistinguishable from luck.</p>'
               + _table(bm, {"bets": "{:.0f}", "win rate": "{:.0%}", "avg odds": "{:.2f}",
                             "ROI": "{:+.1%}", "± (1 s.e.)": "{:.1%}", "avg CLV": "{:+.1%}",
                             "beat close": "{:.0%}"}))
    else:
        bet = '<p class="mute">No value bets with recorded odds yet.</p>'
    vm = ""
    if vs_market is not None and len(vs_market):
        vm = ('<h3 style="margin:16px 0 4px;font-size:14px">Model vs market-implied</h3>'
              '<div class="mute" style="margin-bottom:6px">Market view solved from each game\'s 1X2 + O/U '
              'odds, so markets nobody quotes (GG2+) still get a benchmark. diff &lt; 0: model better; '
              '|z| &gt; 2: unlikely to be luck.</div>'
              + _table(vs_market, {"matches": "{:.0f}", "diff": "{:+.4f}", "z": "{:+.1f}"}))
    return (f'<div class="card"><h2>{html.escape(title)}</h2>'
            f'<div class="mute" style="margin:4px 0 8px">{note}Lower log loss / RPS is better.</div>'
            f'{_table(t, fmt)}{vm}<h3 style="margin:16px 0 4px;font-size:14px">Calibration (ensemble)</h3>'
            f'<div class="grid">{plots}</div><h3 style="margin:16px 0 4px;font-size:14px">Betting value</h3>{bet}</div>')


def evaluate(store: Store, source: str | None, league: str | None = None, version: str | None = None,
             since: str | None = None) -> dict | None:
    preds, probs = load(store, source=source, league=league, version=version, since=since)
    if preds.empty:
        return None
    ens = probs["ensemble"]
    implied = market_implied(store, preds, ens.columns)
    has_market = bool(implied["home"].notna().any())
    bets = betting(store, preds, ens)
    table = scores(preds, probs | ({"market_implied": implied} if has_market else {}))
    return {"preds": preds, "ensemble": ens, "scores": table, "baseline": market_baseline(store, preds),
            "calibration": calibration(preds, ens), "bets": bets, "bet_summary": betting_summary(bets),
            "vs_market": head_to_head(preds, ens, implied) if has_market else None}


def summary_json(ev: dict) -> str:
    return json.dumps({"scores": ev["scores"].to_dict(orient="index"), "baseline": ev["baseline"],
                       "betting": ev["bet_summary"]}, indent=2, default=str)


# ------------------------------------------------------------ settled games, one by one
RESULTS_FILE = "results.html"
GAME_MARKETS = [("1X2", None), ("BTTS", "btts"), ("GG2+", "gg2"), ("O/U 2.5", "over_2.5")]


def _game_picks(e: pd.Series, hg: int, ag: int) -> list[tuple[str, str, float, bool]]:
    """(market, pick, probability, correct) for the ensemble's call in each market."""
    out = []
    res = "home" if hg > ag else "draw" if hg == ag else "away"
    sel = max(("home", "draw", "away"), key=lambda k: e[k])
    out.append(("1X2", {"home": "1", "draw": "X", "away": "2"}[sel], float(e[sel]), sel == res))
    for label, key in (("BTTS", "btts"), ("GG2+", "gg2")):
        yes = e[f"{key}_yes"] >= 0.5
        p = e[f"{key}_yes"] if yes else e[f"{key}_no"]
        out.append((label, "Yes" if yes else "No", float(p), bool(outcome(f"{key}_{'yes' if yes else 'no'}", hg, ag))))
    over = e["over_2.5"] >= 0.5
    out.append(("O/U 2.5", "Over" if over else "Under", float(e["over_2.5"] if over else e["under_2.5"]),
                bool(outcome("over_2.5" if over else "under_2.5", hg, ag))))
    return out


def settled_games(ev: dict) -> list[dict]:
    """Every settled prediction with its picks and the result, newest first."""
    from .config import LEAGUES
    preds, ens, bets = ev["preds"], ev["ensemble"], ev["bets"]
    bets_by_event = {} if bets.empty else {k: g for k, g in bets.groupby("event_id")}
    games = []
    for i in preds.index:
        r = preds.loc[i]
        hg, ag = int(r["hg"]), int(r["ag"])
        m = unpack(r["matrix"])
        sh, sa = np.unravel_index(np.argmax(m), m.shape)
        vb = bets_by_event.get(r["event_id"])
        games.append({
            "kickoff": pd.Timestamp(r["kickoff"]), "league": LEAGUES[r["league"]].name if r["league"] in LEAGUES
            else r["league"], "home": r["home"], "away": r["away"], "hg": hg, "ag": ag,
            "picks": _game_picks(ens.loc[i], hg, ag),
            "score_pick": (int(sh), int(sa), float(m[sh, sa])), "score_hit": (int(sh), int(sa)) == (hg, ag),
            "value_bets": [] if vb is None else [
                (b.selection, float(b.odds), bool(b.won)) for b in vb.itertuples()],
        })
    games.sort(key=lambda g: g["kickoff"], reverse=True)
    return games


def _sel_label(key: str) -> str:
    if key in ("home", "draw", "away"):
        return {"home": "1", "draw": "X", "away": "2"}[key]
    side, _, rest = key.partition("_")
    return {"btts": "BTTS ", "gg2": "GG2+ "}.get(side, side.capitalize() + " ") + (
        rest.capitalize() if side in ("btts", "gg2") else rest)


def games_summary_html(games: list[dict]) -> str:
    """Hit rate per market vs. what the model expected (the mean probability of its picks)."""
    if not games:
        return ""
    rows = ""
    for idx, (label, _) in enumerate(GAME_MARKETS):
        hits = [g["picks"][idx][3] for g in games]
        probs = [g["picks"][idx][2] for g in games]
        rows += (f"<tr><td>{label}</td><td>{sum(hits)} / {len(hits)}</td><td><b>{np.mean(hits):.0%}</b></td>"
                 f"<td>{np.mean(probs):.0%}</td></tr>")
    score_hits = sum(g["score_hit"] for g in games)
    rows += (f"<tr><td>Exact score</td><td>{score_hits} / {len(games)}</td>"
             f"<td><b>{score_hits / len(games):.0%}</b></td>"
             f"<td>{np.mean([g['score_pick'][2] for g in games]):.0%}</td></tr>")
    return ('<div class="scroll"><table><tr><th>Market</th><th>Picks correct</th><th>Hit rate</th>'
            '<th>Model expected</th></tr>' + rows + '</table></div>'
            '<p class="mute" style="margin-top:4px">"Model expected" is the average probability the model gave '
            'its own picks. A hit rate close to it means the probabilities are honest; consistently below it '
            'means the model is over-confident.</p>')


def games_table_html(games: list[dict], limit: int | None = None) -> str:
    """Prediction vs result for each settled game, with a league filter."""
    from .report import filter_chips
    rows = ""
    for i, g in enumerate(games, 1):
        cells = "".join(
            f'<td class="{"hit" if ok else "miss"}">{"✓" if ok else "✗"} {html.escape(pick)} '
            f'<span class="mute">{p:.0%}</span></td>' for _, pick, p, ok in g["picks"])
        sh, sa, sp = g["score_pick"]
        vb = ", ".join(f'<span class="{"hit" if w else "miss"}">{"✓" if w else "✗"} '
                       f'{html.escape(_sel_label(sel))} @{o:.2f}</span>' for sel, o, w in g["value_bets"]) or "–"
        hidden = " hidden" if limit and i > limit else ""
        rows += (f'<tr data-league="{_slug(g["league"])}"{hidden}>'
                 f'<td class="nw">{g["kickoff"]:%d %b %Y}</td>'
                 f'<td class="l">{html.escape(g["league"])}</td>'
                 f'<td class="l">{html.escape(g["home"])} v {html.escape(g["away"])}</td>'
                 f'<td><b>{g["hg"]}-{g["ag"]}</b></td>{cells}'
                 f'<td class="{"hit" if g["score_hit"] else "mute"}">{sh}-{sa} <span class="mute">{sp:.0%}</span></td>'
                 f'<td>{vb}</td></tr>')
    return ('<div class="filter-wrap" data-key="results">'
            + filter_chips(g["league"] for g in games)
            + f'<div class="scroll"><table class="filterable" data-limit="{limit or 0}"><thead><tr>'
            '<th>Date</th><th class="l">League</th><th class="l">Match</th><th>Result</th><th>1X2</th><th>BTTS</th><th>GG2+</th>'
            '<th>O/U 2.5</th><th>Likeliest score</th><th>Value bets</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>'
            '<p class="mute filter-empty" hidden>No settled games in the selected leagues.</p></div>')


def _slug(text: str) -> str:
    from .report import slug
    return slug(text)


def games_section_html(ev: dict, limit: int = 25) -> str:
    """Card for the track-record page: summary + the latest ``limit`` games + link to all."""
    games = settled_games(ev)
    if not games:
        return ""
    return (f'<div class="card"><div class="head"><h2>Settled predictions</h2>'
            f'<a href="{RESULTS_FILE}">View all {len(games)} settled games →</a></div>'
            f'<div class="mute" style="margin:4px 0 8px">What the tool predicted before kick-off next to the '
            f'final score. The latest {min(limit, len(games))} are shown below, newest first; ✓ = pick was right.</div>'
            f'{games_summary_html(games)}<h3 style="margin:16px 0 4px;font-size:14px">Latest games</h3>'
            f'{games_table_html(games, limit)}</div>')


def games_page_body(ev: dict) -> str:
    games = settled_games(ev)
    return (f'<div class="card"><h2>{len(games)} settled games</h2>{games_summary_html(games)}</div>'
            f'<div class="card">{games_table_html(games)}</div>')
