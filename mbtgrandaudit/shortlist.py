"""The 80% shortlist: main predictions whose model probability is at least 80% on a market
with bookmaker prices, upcoming and settled.

Prices are the stored pre-kick-off quotes: the fetched bookmaker feed and hand entries
(market_odds) plus the ESPN snapshot (odds table, 1X2 and its O/U line). Per bookmaker the
latest quote before kick-off counts; "best" is the highest of those. Only main ('live')
predictions count, and feed-fed ones are left out as in the track record.
"""
from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd

from .evaluate import drop_feed_tainted, outcome
from .report import SHORTLIST_FILE as FILE, _local, page, with_filter_js
from .storage import Store, _iso

THRESHOLD = 0.80
LINES = ("0.5", "1.5", "2.5", "3.5", "4.5")
KEYS = (["home", "draw", "away", "btts_yes", "btts_no", "gg2_yes", "gg2_no"]
        + [f"{side}_{ln}" for ln in LINES for side in ("over", "under")])
BOOKS = ("Bet365", "Pinnacle")
PRICES = [("best", "Best price")] + [(b.lower(), b) for b in BOOKS]


def market_of(key: str) -> str:
    if key in ("home", "draw", "away"):
        return "1X2"
    if key.startswith("btts"):
        return "BTTS"
    if key.startswith("gg2"):
        return "GG2+"
    return "O/U " + key.split("_")[1]


def label(key: str) -> str:
    return {"home": "Home win", "draw": "Draw", "away": "Away win", "btts_yes": "BTTS yes", "btts_no": "BTTS no",
            "gg2_yes": "GG2+ yes", "gg2_no": "GG2+ no"}.get(key) or key.replace("_", " ").capitalize()


def candidates(preds: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    """One row per (prediction, selection) with model probability >= threshold."""
    rows = []
    for r in preds.itertuples():
        mk = json.loads(r.markets)
        for key in KEYS:
            p = mk.get(key)
            if p is not None and np.isfinite(p) and p >= threshold:
                rows.append({"event_id": str(r.event_id), "league": r.league, "home": r.home, "away": r.away,
                             "kickoff": r.kickoff, "key": key, "market": market_of(key), "p": float(p)})
    return pd.DataFrame(rows, columns=["event_id", "league", "home", "away", "kickoff", "key", "market", "p"])


def quotes(store: Store, event_ids) -> pd.DataFrame:
    """Every stored non-closing quote: event_id, market, bookmaker, captured_at, odds."""
    ids = list(map(str, event_ids))
    cols = ["event_id", "market", "bookmaker", "captured_at", "odds"]
    if not ids:
        return pd.DataFrame(columns=cols)
    mo = store.market_odds_for(ids)
    mo = mo[mo["kind"] != "closing"][cols] if not mo.empty else pd.DataFrame(columns=cols)
    rows = []
    od = store.odds_for(ids)
    for r in od[od["kind"] == "prediction"].itertuples():
        book = r.source or "ESPN"
        rows += [(r.event_id, k, book, r.captured_at, getattr(r, k)) for k in ("home", "draw", "away")]
        if pd.notna(r.line) and f"{r.line:.1f}" in LINES:
            rows += [(r.event_id, f"over_{r.line:.1f}", book, r.captured_at, r.over_odds),
                     (r.event_id, f"under_{r.line:.1f}", book, r.captured_at, r.under_odds)]
    q = pd.concat([mo, pd.DataFrame(rows, columns=cols)], ignore_index=True).dropna(subset=["odds"])
    q["odds"] = q["odds"].astype(float)
    q["bookmaker"] = q["bookmaker"].fillna("").replace("", "entered")
    return q[q["odds"] > 1.0]


def price_table(q: pd.DataFrame, kickoff: dict) -> pd.DataFrame:
    """Per (event_id, market): best price and its bookmaker, plus each of BOOKS, from each
    bookmaker's latest quote before kick-off."""
    out_cols = ["best", "best_book"] + [b.lower() for b in BOOKS]
    if q.empty:
        return pd.DataFrame(columns=out_cols)
    q = q[q["captured_at"] <= q["event_id"].map(kickoff)]
    if q.empty:
        return pd.DataFrame(columns=out_cols)
    last = q.sort_values("captured_at").groupby(["event_id", "market", "bookmaker"])["odds"].last().reset_index()
    top = last.loc[last.groupby(["event_id", "market"])["odds"].idxmax()].set_index(["event_id", "market"])
    out = pd.DataFrame({"best": top["odds"], "best_book": top["bookmaker"]})
    for b in BOOKS:
        out[b.lower()] = last[last["bookmaker"] == b].set_index(["event_id", "market"])["odds"]
    return out


def _with_prices(store: Store, c: pd.DataFrame) -> pd.DataFrame:
    if c.empty:
        return c.assign(**{k: pd.Series(dtype=float) for k in ("best", "bet365", "pinnacle")}, best_book="")
    kick = dict(zip(c["event_id"], c["kickoff"]))
    pt = price_table(quotes(store, kick), kick)
    return c.join(pt, on=["event_id", "key"]).assign(fair=lambda d: 1 / d["p"])


def upcoming(store: Store, now, threshold: float = THRESHOLD) -> pd.DataFrame:
    """Shortlist candidates from the latest main prediction of each fixture not yet kicked off."""
    preds = store.query(
        "SELECT p.* FROM predictions p WHERE p.source = 'live' AND p.kickoff > ? AND p.id = ("
        " SELECT p2.id FROM predictions p2 WHERE p2.event_id = p.event_id AND p2.source = 'live'"
        " ORDER BY p2.predicted_at DESC, p2.id DESC LIMIT 1) ORDER BY p.kickoff", (_iso(now),))
    return _with_prices(store, candidates(preds, threshold)).sort_values(["kickoff", "p"], ascending=[True, False])


def settled(store: Store, threshold: float = THRESHOLD) -> pd.DataFrame:
    """Settled shortlist picks (latest pre-kick-off main prediction) with their result."""
    preds = drop_feed_tainted(store, store.settled(source="live"))
    c = candidates(preds, threshold)
    if c.empty:
        return c.assign(won=pd.Series(dtype=bool))
    goals = preds.assign(event_id=preds["event_id"].astype(str)).set_index("event_id")[["home_goals", "away_goals"]]
    g = goals.reindex(c["event_id"])
    c["won"] = [bool(outcome(k, h, a)) for k, h, a in zip(c["key"], g["home_goals"], g["away_goals"])]
    return _with_prices(store, c)


def summary(s: pd.DataFrame, col: str) -> dict:
    """Hit rate and flat 1-unit profit at price column ``col``, over picks priced there."""
    b = s[s[col].notna()]
    if b.empty:
        return {"n": 0}
    profit = np.where(b["won"], b[col] - 1, -1.0)
    return {"n": len(b), "hit": float(b["won"].mean()), "avg_p": float(b["p"].mean()),
            "avg_odds": float(b[col].mean()), "profit": float(profit.sum()), "roi": float(profit.mean()),
            "roi_se": float(profit.std(ddof=1) / np.sqrt(len(b))) if len(b) > 1 else None}


def record(s: pd.DataFrame) -> dict:
    """{price column: {"all": summary, "market": {...}, "league": {...}}}."""
    out = {}
    for col, _ in PRICES:
        out[col] = {"all": summary(s, col),
                    "market": {m: summary(g, col) for m, g in s.groupby("market")} if len(s) else {},
                    "league": {lg: summary(g, col) for lg, g in s.groupby("league")} if len(s) else {}}
    return out


# --------------------------------------------------------------------- html
def _odds(v) -> str:
    return "–" if v is None or pd.isna(v) else f"{v:.2f}"


def _edge(p, o) -> str:
    if o is None or pd.isna(o):
        return '<td class="mute">–</td>'
    e = p * o - 1
    return f'<td class="{"hit" if e > 0 else "miss"}">{e:+.1%}</td>'


def _upcoming_table(u: pd.DataFrame) -> str:
    if u.empty:
        return '<p class="mute">No upcoming main prediction reaches 80% on a priced market.</p>'
    rows = "".join(
        f'<tr data-league="{html.escape(r.league)}"><td class="l nw">{_local(pd.Timestamp(r.kickoff).to_pydatetime())}</td>'
        f'<td class="l">{html.escape(r.home)} v {html.escape(r.away)}</td><td class="l">{html.escape(r.league)}</td>'
        f'<td class="l">{label(r.key)}</td><td>{r.p:.1%}</td><td>{r.fair:.2f}</td>'
        f'<td>{_odds(r.best)}<div class="mute small">{html.escape(str(r.best_book)) if pd.notna(r.best) else ""}</div></td>'
        f'{_edge(r.p, r.best)}<td>{_odds(r.bet365)}</td>{_edge(r.p, r.bet365)}'
        f'<td>{_odds(r.pinnacle)}</td>{_edge(r.p, r.pinnacle)}</tr>' for r in u.itertuples())
    return ('<div class="scroll"><table><thead><tr><th class="l">Kick-off</th><th class="l">Match</th>'
            '<th class="l">League</th><th class="l">Pick</th><th>Model</th><th>Fair</th><th>Best</th><th>Edge</th>'
            '<th>Bet365</th><th>Edge</th><th>Pinnacle</th><th>Edge</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>')


def _cells(d: dict) -> str:
    if not d.get("n"):
        return '<td class="mute">0</td><td></td><td></td><td></td>'
    roi = f'{d["roi"]:+.1%}' + (f' <span class="mute small">± {d["roi_se"]:.1%}</span>' if d.get("roi_se") else "")
    return (f'<td>{d["n"]}</td><td>{d["hit"]:.1%}</td><td>{d["avg_odds"]:.2f}</td>'
            f'<td class="{"hit" if d["profit"] > 0 else "miss"}">{d["profit"]:+.2f} u · {roi}</td>')


def _record_table(rec: dict, part: str, title: str) -> str:
    names = ["all"] if part == "all" else sorted(set().union(*(rec[c][part] for c, _ in PRICES)))
    if not names:
        return ""
    head = "".join(f'<th colspan="4">{h}</th>' for _, h in PRICES)
    sub = "<th>Picks</th><th>Hit</th><th>Avg odds</th><th>Profit · ROI</th>" * len(PRICES)
    rows = ""
    for n in names:
        rows += f'<tr><td class="l">{html.escape(n if part != "all" else "All picks")}</td>'
        rows += "".join(_cells(rec[c]["all"] if part == "all" else rec[c][part].get(n, {"n": 0})) for c, _ in PRICES)
        rows += "</tr>"
    return (f'<h3 style="margin:14px 0 4px">{title}</h3><div class="scroll"><table>'
            f'<tr><th class="l"></th>{head}</tr><tr><th class="l"></th>{sub}</tr>{rows}</table></div>')


def page_body(up: pd.DataFrame, done: pd.DataFrame, threshold: float = THRESHOLD) -> str:
    rec = record(done)
    priced = int(done["best"].notna().sum()) if len(done) else 0
    hit_all = f"{done['won'].mean():.1%}" if len(done) else "–"
    intro = (f'<div class="mute">Main predictions (hand-entered prices only in the market view) whose model '
             f'probability is at least {threshold:.0%} on 1X2, O/U 0.5–4.5, BTTS or GG2+. Fair = 1 / probability; '
             f'edge = probability × odds − 1. Prices are each bookmaker\'s latest stored quote before kick-off '
             f'(API-Football feed, ESPN/DraftKings and odds you entered); "best" is the highest. Profit is flat '
             f'1-unit stakes, counted only where a price was stored. For research; bet responsibly.</div>')
    rec_html = (f'<div class="mute">{len(done)} settled picks, hit rate {hit_all}; {priced} with a stored '
                f'pre-kick-off price. ± is one standard error of ROI: small samples swing a lot.</div>'
                + _record_table(rec, "all", "Overall")
                + _record_table(rec, "market", "By market")
                + _record_table(rec, "league", "By league"))
    return (f'<div class="card">{intro}</div>'
            f'<div class="card"><h2>Upcoming ({len(up)} picks)</h2>{_upcoming_table(up)}</div>'
            f'<div class="card"><h2>Running record</h2>{rec_html}</div>')


def write(store: Store, path, now, nav: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write the shortlist page to ``path``; returns (upcoming, settled) picks."""
    up, done = upcoming(store, now), settled(store)
    path.write_text(with_filter_js(page("80% shortlist", page_body(up, done), now, nav)), encoding="utf-8")
    return up, done
