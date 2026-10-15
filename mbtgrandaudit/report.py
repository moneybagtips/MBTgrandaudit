"""Console, JSON/CSV and HTML output for predictions."""
from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path

from .models import build_models
from .config import Settings

MODEL_DESC = {m.name: m.description for m in build_models(Settings())} | {
    "market_view": "Bookmaker view: score matrix solved from the game's 1X2 + O/U odds (margin removed)"}


def pct(p: float | None) -> str:
    return "-" if p is None else f"{100 * p:.1f}%"


def _local(dt: datetime) -> str:
    return dt.astimezone().strftime("%a %d %b %H:%M")


# ------------------------------------------------------------------- console
def console(preds, detail: bool = False) -> str:
    lines = []
    for p in preds:
        e, a, fx = p.ensemble, p.analysis, p.fixture
        lines.append("=" * 92)
        lines.append(f"{_local(fx.kickoff)}  {fx.home} vs {fx.away}   "
                     f"expected goals {e['xg_home']:.2f} - {e['xg_away']:.2f}")
        lines.append("-" * 92)
        mk = p.market.get("implied", {})
        mline = (f"   market {pct(mk.get('home'))}/{pct(mk.get('draw'))}/{pct(mk.get('away'))}"
                 if "home" in mk else "")
        lines.append(f"  1X2     1 {pct(e['home']):>6}   X {pct(e['draw']):>6}   2 {pct(e['away']):>6}{mline}")
        lines.append(f"  BTTS    Yes {pct(e['btts_yes']):>6}  No {pct(e['btts_no']):>6}     "
                     f"GG2+  Yes {pct(e['gg2_yes']):>6}  No {pct(e['gg2_no']):>6}")
        lines.append("  O/U     " + "  ".join(f"O{l} {pct(e[f'over_{l}'])}" for l in (0.5, 1.5, 2.5, 3.5, 4.5)
                                            if f"over_{l}" in e))
        view = p.market.get("implied_model")
        if view:
            info = p.market["implied_info"]
            lines.append(f"  Market  BTTS {pct(view['btts_yes'])}  GG2+ {pct(view['gg2_yes'])}  "
                         f"O2.5 {pct(view['over_2.5'])}  xG {info['xg_home']:.2f}-{info['xg_away']:.2f}"
                         f"   (implied from {' + '.join(info['matched'])} odds)")
        lines.append("  Scores  " + ", ".join(f"{h}-{a_} {pct(q)}" for h, a_, q in e["top_scores"]))
        lines.append("  Picks   " + " | ".join(
            f"{k['market']}: {k['pick']} {pct(k['prob'])} ({k['confidence']}"
            + (f", mkt {pct(k['market_prob'])}" if k.get("market_prob") is not None else "")
            + (f", @{k['price']:.2f} edge {k['edge']:+.0%}" if k.get("edge") is not None else "")
            + (" VALUE" if k.get("value") else " CHECK: far from market" if k.get("check") else "") + ")"
            for k in p.picks if k["market"] in ("1X2", "BTTS", "GG2+", "O/U 2.5") or k.get("value") or k.get("check")))
        for side, key in ((fx.home, "home_form"), (fx.away, "away_form")):
            f = a[key]
            if f["n"]:
                lines.append(f"  Form    {side[:22]:22s} {f['seq']:>6}  W{f['w']} D{f['d']} L{f['l']}  "
                             f"GF {f['gf']:.2f} GA {f['ga']:.2f}  BTTS {pct(f['btts'])}  O2.5 {pct(f['over25'])}")
        h = a["h2h"]
        if h["n"]:
            lines.append(f"  H2H     last {h['n']}: {fx.home} W{h['w']} D{h['d']} L{h['l']}  "
                         f"avg {h['gf'] + h['ga']:.2f} goals  BTTS {pct(h['btts'])}  O2.5 {pct(h['over25'])}  "
                         f"recent: " + ", ".join(g["score"] for g in h["games"][:5]))
        else:
            lines.append("  H2H     no previous meetings found")
        if detail:
            lines.append(f"  {'model':20s}{'weight':>7}{'xG':>12}{'1':>7}{'X':>7}{'2':>7}"
                         f"{'BTTS':>7}{'GG2+':>7}{'O2.5':>7}")
            for name, m in p.per_model.items():
                lines.append(f"  {name:20s}{pct(p.weights.get(name, 0)):>7}"
                             f"{m['xg_home']:>6.2f}-{m['xg_away']:<5.2f}"
                             f"{pct(m['home']):>7}{pct(m['draw']):>7}{pct(m['away']):>7}"
                             f"{pct(m['btts_yes']):>7}{pct(m['gg2_yes']):>7}{pct(m['over_2.5']):>7}")
        for w in p.warnings:
            lines.append(f"  ! {w}")
    return "\n".join(lines)


# ---------------------------------------------------------------- json / csv
def to_records(preds, league: str) -> list[dict]:
    out = []
    for p in preds:
        rec = {
            "league": league, "kickoff": p.fixture.kickoff.isoformat(),
            **{k: v for k, v in p.ensemble.items() if k != "top_scores"},
            "home_team": p.fixture.home, "away_team": p.fixture.away,
            "model_home": p.home, "model_away": p.away,
            "top_scores": [f"{h}-{a}:{q:.4f}" for h, a, q in p.ensemble["top_scores"]],
            "picks": p.picks, "market": p.market, "weights": p.weights,
            "per_model": {k: {kk: vv for kk, vv in v.items() if kk != "top_scores"}
                          for k, v in p.per_model.items()},
            "analysis": p.analysis, "warnings": p.warnings,
        }
        out.append(rec)
    return out


def write_json(records, path: Path):
    path.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")


def write_csv(records, path: Path):
    import pandas as pd
    flat = [{k: v for k, v in r.items()
             if not isinstance(v, (dict, list))} for r in records]
    pd.DataFrame(flat).to_csv(path, index=False)


# ---------------------------------------------------------------------- html
CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#16191d;--mute:#5c6470;--line:#e3e6ea;--acc:#1f7a4d;
--acc-soft:#e3f3ea;--warn:#a15c00;--bar:#2f6fed;--bar2:#9aa3ad;--bar3:#d0553f}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f1216;--card:#171b21;--ink:#e8ebef;
--mute:#98a1ad;--line:#2a3039;--acc:#4cc38a;--acc-soft:#173326;--warn:#f0a646;--bar:#5b8def;--bar2:#5d6672;--bar3:#e2725d}}
:root[data-theme="dark"]{--bg:#0f1216;--card:#171b21;--ink:#e8ebef;--mute:#98a1ad;--line:#2a3039;--acc:#4cc38a;
--acc-soft:#173326;--warn:#f0a646;--bar:#5b8def;--bar2:#5d6672;--bar3:#e2725d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1080px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:0}.mute{color:var(--mute)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:14px 0}
.head{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:baseline}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin-top:12px}
.box{border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.box h3{margin:0 0 6px;font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--mute)}
.bar{display:flex;height:22px;border-radius:6px;overflow:hidden;font-size:12px;color:#fff;font-weight:600}
.bar span{display:flex;align-items:center;justify-content:center;min-width:0;white-space:nowrap;overflow:hidden}
.row{display:flex;justify-content:space-between;gap:8px}
.pick{display:inline-block;background:var(--acc-soft);color:var(--acc);border-radius:999px;padding:2px 10px;
font-weight:600;font-size:12px;margin:2px 4px 2px 0}
a{color:var(--bar);text-decoration:none}a:hover{text-decoration:underline}
.nav{margin-bottom:12px;font-size:13px}.jump{line-height:1.9;font-size:13px}
.tile{display:block;color:var(--ink)}.tile:hover{border-color:var(--bar);text-decoration:none}
.pick.value{background:var(--acc);color:var(--card)}
.pick.low{background:transparent;color:var(--mute);border:1px solid var(--line)}
table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:5px 6px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}
.scroll{overflow-x:auto}details{margin-top:10px}summary{cursor:pointer;color:var(--mute)}
.warn{color:var(--warn);font-size:12.5px}.seq b{display:inline-block;width:18px;text-align:center;border-radius:4px;
margin-right:2px;font-size:11px;color:#fff}.W{background:#2e9d62}.D{background:#8a929c}.L{background:#cf4a3a}
"""


def _bar(e):
    return (f'<div class="bar"><span style="width:{e["home"]*100:.1f}%;background:var(--bar)">{pct(e["home"])}</span>'
            f'<span style="width:{e["draw"]*100:.1f}%;background:var(--bar2)">{pct(e["draw"])}</span>'
            f'<span style="width:{e["away"]*100:.1f}%;background:var(--bar3)">{pct(e["away"])}</span></div>')


def _seq(s):
    return '<span class="seq">' + "".join(f'<b class="{c}">{c}</b>' for c in s) + "</span>"


def _form_box(title, f):
    if not f["n"]:
        return f'<div class="box"><h3>{html.escape(title)}</h3><span class="mute">No data</span></div>'
    games = "".join(f"<tr><td>{g['date']}</td><td>{'H' if g['venue']=='home' else 'A'} "
                    f"{html.escape(str(g['opponent']))}</td><td>{g['score']}</td>"
                    f"<td class='mute'>{html.escape(str(g['comp']))[:18]}</td></tr>" for g in f["games"])
    return (f'<div class="box"><h3>{html.escape(title)}</h3>{_seq(f["seq"])}'
            f'<div class="row"><span>W{f["w"]} D{f["d"]} L{f["l"]}</span>'
            f'<span>GF {f["gf"]:.2f} · GA {f["ga"]:.2f}</span></div>'
            f'<div class="row mute"><span>BTTS {pct(f["btts"])}</span><span>O2.5 {pct(f["over25"])}</span>'
            f'<span>CS {pct(f["clean_sheets"])}</span></div>'
            f'<details><summary>matches</summary><div class="scroll"><table>{games}</table></div></details></div>')


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def league_file(league: str) -> str:
    return f"league-{slug(league)}.html"


def match_anchor(fx) -> str:
    return "m-" + slug(str(fx.event_id))


def page(title: str, body: str, generated: datetime | None = None, nav: str = "") -> str:
    """A complete standalone HTML page in the report style."""
    stamp = (f'<div class="mute">Generated {generated.astimezone().strftime("%Y-%m-%d %H:%M")}</div>'
             if generated else "")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>{CSS}</style></head><body><main>
{f'<nav class="nav">{nav}</nav>' if nav else ''}<h1>{html.escape(title)}</h1>{stamp}
{body}
<p class="mute" style="margin-top:32px">Models: {', '.join(MODEL_DESC)}. Fair odds = 1 / probability.
Edge = ensemble probability × bookmaker odds − 1. For research and entertainment; bet responsibly.</p>
</main></body></html>"""


def _match_card(p) -> str:
    e, a, fx = p.ensemble, p.analysis, p.fixture
    picks = "".join(
        f'<span class="pick {"value" if k.get("value") else "low" if k["confidence"]=="low" else ""}">'
        f'{k["market"]}: {k["pick"]} {pct(k["prob"])}'
        + (f' · mkt {pct(k["market_prob"])}' if k.get("market_prob") is not None else "")
        + (f' · @{k["price"]:.2f} edge {k["edge"]:+.0%}' if k.get("edge") is not None else "")
        + (" · VALUE" if k.get("value") else " · check: far from market" if k.get("check") else "") + "</span>"
        for k in p.picks)
    view = p.market.get("implied_model", {})
    mv = lambda key: f' <span class="mute">mkt {pct(view[key])}</span>' if key in view else ""
    mk = p.market.get("implied", {})
    market = (f'<div class="row mute" style="margin-top:4px"><span>Market ({html.escape(str(p.market.get("source")))})</span>'
              f'<span>{pct(mk.get("home"))} / {pct(mk.get("draw"))} / {pct(mk.get("away"))}</span></div>'
              if "home" in mk else "")
    ou = "".join(f"<tr><td>{l}</td><td>{pct(e[f'over_{l}'])}</td><td>{pct(e[f'under_{l}'])}</td>"
                 + (f"<td class='mute'>{pct(view[f'over_{l}'])}</td>" if view else "") + "</tr>"
                 for l in (0.5, 1.5, 2.5, 3.5, 4.5) if f"over_{l}" in e)
    scores = "".join(f"<tr><td>{h}-{a_}</td><td>{pct(q)}</td></tr>" for h, a_, q in e["top_scores"])
    h = a["h2h"]
    h2h = (f'{_seq(h["seq"])}<div class="row"><span>{html.escape(fx.home)} W{h["w"]} D{h["d"]} L{h["l"]}</span>'
           f'<span>{h["gf"] + h["ga"]:.2f} goals/game</span></div>'
           f'<div class="row mute"><span>BTTS {pct(h["btts"])}</span><span>GG2+ {pct(h["gg2"])}</span>'
           f'<span>O2.5 {pct(h["over25"])}</span></div>'
           + "<div class='scroll'><table>" + "".join(
               f"<tr><td>{g['date']}</td><td>{'H' if g['venue']=='home' else 'A'}</td><td>{g['score']}</td>"
               f"<td class='mute'>{html.escape(str(g['comp']))[:18]}</td></tr>" for g in h["games"][:8])
           + "</table></div>") if h["n"] else '<span class="mute">No previous meetings found</span>'
    models = "".join(
        f"<tr><td title='{html.escape(MODEL_DESC.get(n, ''))}'>{n}</td><td>{pct(p.weights.get(n, 0))}</td>"
        f"<td>{m['xg_home']:.2f}-{m['xg_away']:.2f}</td><td>{pct(m['home'])}</td><td>{pct(m['draw'])}</td>"
        f"<td>{pct(m['away'])}</td><td>{pct(m['btts_yes'])}</td><td>{pct(m['gg2_yes'])}</td>"
        f"<td>{pct(m['over_2.5'])}</td></tr>" for n, m in p.per_model.items())
    warns = "".join(f'<div class="warn">{html.escape(w)}</div>' for w in p.warnings)
    return f"""
<div class="card" id="{match_anchor(fx)}"><div class="head"><h2>{html.escape(fx.home)} <span class="mute">vs</span> {html.escape(fx.away)}</h2>
<span class="mute">{_local(fx.kickoff)}{' · ' + html.escape(fx.venue) if fx.venue else ''}</span></div>
<div class="row mute" style="margin:6px 0"><span>Expected goals {e['xg_home']:.2f} – {e['xg_away']:.2f}</span>
<span>1 · X · 2</span></div>{_bar(e)}{market}
<div style="margin-top:10px">{picks}</div>
<div class="grid">
<div class="box"><h3>Goals markets</h3><div class="row"><span>BTTS yes</span><span><b>{pct(e['btts_yes'])}</b>{mv('btts_yes')}</span></div>
<div class="row"><span>GG2+ yes</span><span><b>{pct(e['gg2_yes'])}</b>{mv('gg2_yes')}</span></div>
<table><tr><th>Line</th><th>Over</th><th>Under</th>{'<th>Mkt over</th>' if view else ''}</tr>{ou}</table>
{'<div class="mute" style="font-size:11.5px;margin-top:4px">mkt = implied from the bookmaker 1X2 + O/U odds</div>' if view else ''}</div>
<div class="box"><h3>Likeliest scores</h3><table>{scores}</table></div>
<div class="box"><h3>Head to head</h3>{h2h}</div>
{_form_box(fx.home + ' – form (all comps)', a['home_form'])}
{_form_box(fx.away + ' – form (all comps)', a['away_form'])}
</div>
<details><summary>Model breakdown ({len(p.per_model)} components)</summary><div class="scroll"><table>
<tr><th>Model</th><th>Weight</th><th>xG</th><th>1</th><th>X</th><th>2</th><th>BTTS</th><th>GG2+</th><th>O2.5</th></tr>
{models}</table></div></details>{warns}</div>"""


def html_report(preds_by_league: dict, path: Path, generated: datetime,
                track_record: str | None = "track_record.html") -> list[Path]:
    """Main page at ``path`` plus one page per league beside it. ``track_record`` is the
    track-record page to link to (None: not available yet). Returns the files written."""
    path = Path(path)
    main_name = path.name
    tr_link = (f'<a href="{html.escape(track_record)}">Track record</a>' if track_record
               else '<span class="mute">Track record (none yet)</span>')
    nav = f'<a href="{html.escape(main_name)}">← All leagues</a> · {tr_link}'
    written = []

    def link(lg, fx):
        return (f'<a href="{league_file(lg)}#{match_anchor(fx)}">'
                f'{html.escape(fx.home)} v {html.escape(fx.away)}</a>')

    # ---- one page per league
    for league, preds in preds_by_league.items():
        out = path.with_name(league_file(league))
        jump = " · ".join(f'<a href="#{match_anchor(p.fixture)}">{html.escape(p.fixture.home)} v '
                          f'{html.escape(p.fixture.away)}</a>' for p in preds)
        body = (f'<div class="card"><h2>{len(preds)} fixtures</h2><div class="jump">{jump}</div></div>'
                + "".join(_match_card(p) for p in preds))
        out.write_text(page(f"{league} predictions", body, generated, nav), encoding="utf-8")
        written.append(out)

    # ---- main page: cross-league summaries + links
    allp = [(lg, p) for lg, preds in preds_by_league.items() for p in preds]

    def board(title, items):
        items = sorted(items, key=lambda t: -t[0])[:8]
        rows = "".join(f"<tr><td>{link(lg, fx)}</td><td>{pick}</td><td>{pct(q)}</td><td>{1 / q:.2f}</td></tr>"
                       for q, lg, fx, pick in items)
        return (f'<div class="box"><h3>{title}</h3><table><tr><th>Match</th><th>Pick</th><th>Prob.</th>'
                f'<th>Fair</th></tr>{rows}</table></div>')

    pick_of = lambda p, mkt: next(k for k in p.picks if k["market"] == mkt)
    boards = [
        board("1X2", [(pick_of(p, "1X2")["prob"], lg, p.fixture, pick_of(p, "1X2")["pick"]) for lg, p in allp]),
        board("Both teams to score", [(pick_of(p, "BTTS")["prob"], lg, p.fixture, pick_of(p, "BTTS")["pick"])
                                      for lg, p in allp]),
        board("Over / Under 2.5", [(pick_of(p, "O/U 2.5")["prob"], lg, p.fixture, pick_of(p, "O/U 2.5")["pick"])
                                   for lg, p in allp]),
        board("GG2+ — likeliest Yes", [(p.ensemble["gg2_yes"], lg, p.fixture, "Yes") for lg, p in allp]),
    ]

    draw_rows = ""
    for i, (lg, p) in enumerate(sorted(allp, key=lambda t: -t[1].ensemble["draw"])[:20], 1):
        e, fx = p.ensemble, p.fixture
        h, a_, q = e["top_scores"][0]
        mk = p.market.get("implied", {}).get("draw")
        draw_rows += (
            f"<tr><td>{i}</td><td>{_local(fx.kickoff)}</td><td>{html.escape(lg)}</td>"
            f"<td>{link(lg, fx)}</td><td><b>{pct(e['draw'])}</b></td>"
            f"<td>{e['home'] * 100:.0f} / {e['away'] * 100:.0f}</td>"
            f"<td>{e['xg_home']:.2f}–{e['xg_away']:.2f}</td><td>{h}-{a_} ({pct(q)})</td>"
            f"<td>{pct(e['under_2.5'])}</td><td>{pct(mk) if mk else '–'}</td><td>{1 / e['draw']:.2f}</td></tr>")
    draws_card = (
        '<div class="card"><h2>Likeliest draws</h2>'
        '<div class="mute" style="margin:4px 0 8px">Fixtures ranked by ensemble draw probability. '
        'Draws rarely top 31%; a high value means an evenly matched, low-scoring game, '
        'not that a draw is expected.</div><div class="scroll"><table>'
        '<tr><th>#</th><th>Kick-off</th><th>League</th><th>Match</th><th>Draw</th><th>1 / 2</th>'
        '<th>Exp. goals</th><th>Likeliest score</th><th>U2.5</th><th>Market draw</th><th>Fair</th></tr>'
        f'{draw_rows}</table></div></div>')

    tiles = ""
    for league, preds in preds_by_league.items():
        kicks = sorted(p.fixture.kickoff for p in preds)
        span = (kicks[0].astimezone().strftime("%a %d %b") +
                ("" if len(kicks) == 1 else " – " + kicks[-1].astimezone().strftime("%a %d %b")))
        value = sum(1 for p in preds for k in p.picks if k.get("value"))
        tiles += (f'<a class="box tile" href="{league_file(league)}"><h3>{html.escape(league)}</h3>'
                  f'<div class="row"><b>{len(preds)} fixture{"s" if len(preds) != 1 else ""}</b>'
                  f'<span class="mute">{span}</span></div>'
                  f'<div class="row mute"><span>{value} VALUE pick{"s" if value != 1 else ""}</span>'
                  f'<span>View →</span></div></a>')
    leagues_card = f'<div class="card"><h2>Leagues</h2><div class="grid">{tiles}</div></div>'
    if track_record:
        tr_card = (f'<div class="card"><div class="head"><h2>Track record</h2>'
                   f'<a href="{html.escape(track_record)}">Open track record →</a></div>'
                   '<div class="mute" style="margin-top:4px">How past predictions scored: accuracy per '
                   'market against the bookmakers, calibration, and the result of every value bet.</div></div>')
    else:
        tr_card = ('<div class="card"><h2>Track record</h2><div class="mute">Not available yet: it appears once '
                   'logged predictions (or a backfill) have settled results.</div></div>')

    body = (f'<div class="mute">{len(allp)} fixtures · {len(preds_by_league)} leagues · ensemble of '
            f'{len(MODEL_DESC)} components · probabilities, not certainties</div>'
            f'{leagues_card}<div class="card"><h2>Strongest calls by market</h2><div class="grid">'
            f'{"".join(boards)}</div></div>{draws_card}{tr_card}')
    path.write_text(page("MBTgrandaudit predictions", body, generated, tr_link), encoding="utf-8")
    written.insert(0, path)
    return written
