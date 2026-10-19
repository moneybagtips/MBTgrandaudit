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
.tile{display:block;color:var(--ink)}a.tile:hover{border-color:var(--bar);text-decoration:none}
.tile.idle{opacity:.6;border-style:dashed}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 10px}
.chip{border:1px solid var(--line);border-radius:999px;padding:3px 10px;background:var(--card);color:var(--ink);
font:inherit;font-size:12.5px;cursor:pointer}
.chip[aria-pressed="true"]{border-color:var(--bar);background:var(--acc-soft)}
.chip[aria-pressed="false"]{opacity:.5;text-decoration:line-through}
.chip-all{font-weight:600}tr[hidden]{display:none}
.hit{color:var(--acc)}.miss{color:var(--bar3)}td.hit,td.miss{white-space:nowrap}
th.l,td.l{text-align:left}.nw{white-space:nowrap}
.note{border-left:4px solid var(--warn)}
table.odds input.price{width:68px;padding:3px 4px;border:1px solid var(--line);border-radius:6px;
background:var(--card);color:var(--ink);font:inherit;font-size:12.5px}
table.odds td{vertical-align:top}.edge{display:inline-block;min-width:3.5em;margin-left:4px;font-size:12px}
.small{font-size:11px}#bookmaker{padding:3px 6px;border:1px solid var(--line);border-radius:6px;
background:var(--card);color:var(--ink);font:inherit}
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


DRAWS_FILE = "draws.html"

# League filter for tables. Each .filter-wrap holds chips (data-league) and a
# table.filterable whose rows carry data-league; excluded leagues are kept in localStorage
# per data-key, so e.g. the main page and draws.html share one choice. A table shows the
# first data-limit rows that survive the filter (0 = all), renumbered in td.rk if present.
FILTER_JS = """(function(){
document.querySelectorAll('.filter-wrap').forEach(function(w){
  var KEY='mbt-filter-'+w.dataset.key, off;
  try{off=new Set(JSON.parse(localStorage.getItem(KEY)||'[]'));}catch(e){off=new Set();}
  function apply(){
    var t=w.querySelector('table.filterable'), lim=parseInt(t.dataset.limit||'0',10)||Infinity, n=0;
    t.querySelectorAll('tbody tr').forEach(function(r){
      var show=!off.has(r.dataset.league)&&n<lim;
      if(show){n++;var rk=r.querySelector('.rk');if(rk){rk.textContent=n;}}
      r.hidden=!show;
    });
    var empty=w.querySelector('.filter-empty'); if(empty){empty.hidden=n>0;}
    w.querySelectorAll('.chip[data-league]').forEach(function(c){
      c.setAttribute('aria-pressed',String(!off.has(c.dataset.league)));
    });
    try{localStorage.setItem(KEY,JSON.stringify(Array.from(off)));}catch(e){}
  }
  w.querySelectorAll('.chip[data-league]').forEach(function(c){
    c.addEventListener('click',function(){
      var l=c.dataset.league; if(off.has(l)){off.delete(l);}else{off.add(l);} apply();
    });
  });
  w.querySelectorAll('.chip-all').forEach(function(c){
    c.addEventListener('click',function(){off.clear();apply();});
  });
  apply();
});
})();"""


def with_filter_js(html_page: str) -> str:
    return html_page.replace("</body>", f"<script>{FILTER_JS}</script></body>")


def filter_chips(leagues) -> str:
    """'Show all' plus one toggle per league, with its row count."""
    counts: dict[str, int] = {}
    for lg in leagues:
        counts[lg] = counts.get(lg, 0) + 1
    chips = "".join(f'<button type="button" class="chip" data-league="{slug(lg)}" aria-pressed="true">'
                    f'{html.escape(lg)} <span class="mute">{n}</span></button>' for lg, n in sorted(counts.items()))
    return (f'<div class="chips" role="group" aria-label="Filter by league">'
            f'<button type="button" class="chip chip-all">Show all</button>{chips}</div>')


VALUES_FILE = "values.html"
ODDS_FILE = "odds.html"
FORM_MARKETS = [("gg2_yes", "GG2+ Yes"), ("gg2_no", "GG2+ No"), ("btts_yes", "BTTS Yes"), ("btts_no", "BTTS No")]

# Odds entry form: live edge per box; "Open GitHub issue" builds a pre-filled issue that the
# odds-entry workflow imports (one line per price: event id | market | odds | match).
ODDS_JS = r"""(function(){
var form=document.getElementById('odds-form'); if(!form){return;}
var repo=form.dataset.repo, need=JSON.parse(form.dataset.need);
function group(m){return m.split('_')[0];}
function entries(){
  var out=[];
  form.querySelectorAll('input.price').forEach(function(i){
    var o=parseFloat((i.value||'').replace(',','.'));
    if(o>1){var r=i.closest('tr');out.push({ev:r.dataset.event,label:r.dataset.label,home:r.dataset.home,
      away:r.dataset.away,market:i.dataset.market,odds:o.toFixed(2)});}
  });
  return out;
}
function refresh(){
  form.querySelectorAll('input.price').forEach(function(i){
    var o=parseFloat((i.value||'').replace(',','.')), p=parseFloat(i.dataset.p), out=i.nextElementSibling;
    if(!(o>1)){out.textContent='';out.className='edge';return;}
    var e=p*o-1, ok=e>=need[group(i.dataset.market)];
    out.textContent=(e>=0?'+':'')+Math.round(e*100)+'%'+(ok?' VALUE':'');
    out.className='edge '+(ok?'hit':'miss');
  });
  var n=entries().length;
  document.getElementById('odds-count').textContent=n+' price'+(n===1?'':'s')+' entered';
}
function body(es){
  var bm=document.getElementById('bookmaker').value.trim();
  return 'Odds entered on the MBTgrandaudit odds form.\n'+(bm?'Bookmaker: '+bm+'\n':'')+
    '\n> Format: event id | market | decimal odds | match\n\n'+
    es.map(function(e){return e.ev+' | '+e.market+' | '+e.odds+' | '+e.label;}).join('\n')+'\n';
}
function csv(es){
  var bm=document.getElementById('bookmaker').value.trim(), q=function(s){return '"'+String(s).replace(/"/g,'""')+'"';};
  return 'home,away,market,odds,bookmaker\n'+es.map(function(e){
    return [q(e.home),q(e.away),e.market,e.odds,q(bm)].join(',');}).join('\n')+'\n';
}
form.addEventListener('input',refresh);
document.getElementById('odds-submit').addEventListener('click',function(){
  var es=entries(), msg=document.getElementById('odds-msg');
  if(!es.length){msg.textContent='Enter at least one price first.';return;}
  document.getElementById('odds-csv').value=csv(es);
  if(!repo){msg.textContent='GitHub repository unknown: copy the CSV below and run odds import.';return;}
  var title='Odds: '+es.length+' price'+(es.length===1?'':'s')+' ('+new Date().toISOString().slice(0,10)+')';
  window.open('https://github.com/'+repo+'/issues/new?title='+encodeURIComponent(title)+
              '&body='+encodeURIComponent(body(es)),'_blank','noopener');
  msg.textContent='A pre-filled GitHub issue has opened in a new tab: press "Create" to submit.';
});
document.getElementById('odds-csv-btn').addEventListener('click',function(){
  document.getElementById('odds-csv').value=csv(entries());
});
refresh();
})();"""


def _odds_form_body(preds_by_league: dict, link, repo: str | None, min_edge: dict) -> str:
    """Odds entry form for markets bookmakers rarely quote on ESPN (GG2+, BTTS)."""
    sections = ""
    for league, preds in preds_by_league.items():
        rows = ""
        for p in sorted(preds, key=lambda p: p.fixture.kickoff):
            fx, e = p.fixture, p.ensemble
            label = f"{fx.home} v {fx.away}"
            cells = "".join(
                f'<td><input class="price" type="number" inputmode="decimal" step="0.01" min="1.01" '
                f'data-market="{key}" data-p="{e[key]:.6f}" placeholder="fair {1 / e[key]:.2f}" '
                f'aria-label="{html.escape(label)} {name} odds"><span class="edge"></span>'
                f'<div class="mute small">model {pct(e[key])}</div></td>' for key, name in FORM_MARKETS)
            rows += (f'<tr data-event="{html.escape(str(fx.event_id))}" data-label="{html.escape(label)}" '
                     f'data-home="{html.escape(fx.home)}" data-away="{html.escape(fx.away)}">'
                     f'<td class="nw">{_local(fx.kickoff)}</td><td class="l">{link(league, fx)}</td>{cells}</tr>')
        heads = "".join(f"<th>{name}</th>" for _, name in FORM_MARKETS)
        sections += (f'<h3 style="margin:18px 0 6px">{html.escape(league)}</h3><div class="scroll"><table class="odds">'
                     f'<thead><tr><th>Kick-off</th><th class="l">Match</th>{heads}</tr></thead>'
                     f'<tbody>{rows}</tbody></table></div>')
    need = json.dumps({k: v for k, v in min_edge.items()})
    how = ("Press <b>Open GitHub issue</b>: a pre-filled issue opens in a new tab. Press <b>Create</b> and a "
           "GitHub Action saves the prices, replies with each edge and refreshes these pages a few minutes later. "
           "Only issues opened by the repository owner are processed."
           if repo else "This copy of the page doesn't know its GitHub repository, so use the CSV below.")
    return (f'<div class="card note">Enter decimal odds for any market below, then submit them all at once. '
            f'{how} Prices only count if entered before kick-off.</div>'
            f'<div class="card" id="odds-form" data-repo="{html.escape(repo or "")}" data-need=\'{need}\'>'
            '<div class="row" style="align-items:center;flex-wrap:wrap;gap:8px">'
            '<label>Bookmaker <input id="bookmaker" type="text" placeholder="optional, e.g. Bet365"></label>'
            '<span class="mute" id="odds-count"></span></div>'
            f'{sections}'
            '<div class="row" style="margin-top:14px;flex-wrap:wrap;gap:8px;justify-content:flex-start">'
            '<button type="button" class="chip chip-all" id="odds-submit">Open GitHub issue</button>'
            '<button type="button" class="chip" id="odds-csv-btn">Show as CSV</button>'
            '<span class="mute" id="odds-msg" role="status"></span></div>'
            '<details style="margin-top:10px"><summary>CSV for <code>python -m mbtgrandaudit odds import '
            'my_prices.csv</code></summary><textarea id="odds-csv" rows="6" readonly '
            'style="width:100%;margin-top:6px;font:12px monospace"></textarea></details></div>')


PREDICTIONS_FILE = "predictions.html"
MORE_SHARE = 0.20   # "View more" pages: the top 20% of fixtures...


def more_count(total: int, card: int) -> int:
    """...but always at least 10 more rows than the main-page card, and never more than exist."""
    return int(min(total, max(card + 10, -(-total * MORE_SHARE // 1))))


def _model_pick(p, market: str) -> tuple[str, float]:
    """The ensemble's own call in a market: ('1'|'X'|'2', 'Yes'|'No' or 'Over 2.5'|'Under 2.5', prob)."""
    e = p.ensemble
    if market == "1x2":
        key = max(("home", "draw", "away"), key=lambda k: e[k])
        return {"home": "1", "draw": "X", "away": "2"}[key], e[key]
    if market in ("btts", "gg2"):
        yes = e[f"{market}_yes"] >= 0.5
        return ("Yes", e[f"{market}_yes"]) if yes else ("No", e[f"{market}_no"])
    over = e["over_2.5"] >= 0.5
    return ("Over 2.5", e["over_2.5"]) if over else ("Under 2.5", e["under_2.5"])


def _picks_table(items, link, model_pick: str | None = None) -> str:
    """Fixtures ranked by the probability of one selection, with a league filter.
    ``model_pick`` adds a column with the model's own call in that market (e.g. 'gg2'), so a
    ranked 'Yes' at 26% visibly reads as the model expecting 'No'."""
    rows = ""
    for i, (q, pick, key, lg, p) in enumerate(items, 1):
        mp = p.market.get("implied_model", {}).get(key)
        price = p.market.get("prices", {}).get(key)
        edge = p.market.get("edge", {}).get(key)
        own = ""
        if model_pick:
            call, cp = _model_pick(p, model_pick)
            own = f'<td class="nw"><b>{call}</b> <span class="mute">{pct(cp)}</span></td>'
        rows += (f'<tr data-league="{slug(lg)}"><td class="rk">{i}</td><td class="nw">{_local(p.fixture.kickoff)}</td>'
                 f'<td class="l">{html.escape(lg)}</td><td class="l">{link(lg, p.fixture)}</td><td>{pick}</td>'
                 f'<td><b>{pct(q)}</b></td>{own}<td>{1 / q:.2f}</td><td>{pct(mp) if mp is not None else "–"}</td>'
                 f'<td>{f"{price[0]:.2f}" if price else "–"}</td>'
                 f'<td>{f"{edge:+.0%}" if edge is not None else "–"}</td></tr>')
    own_head = "<th>Model's pick</th>" if model_pick else ""
    return ('<div class="filter-wrap" data-key="picks">' + filter_chips(lg for *_, lg, _ in items)
            + '<div class="scroll"><table class="filterable"><thead><tr><th>#</th><th>Kick-off</th>'
            '<th class="l">League</th><th class="l">Match</th><th>Selection</th><th>Prob.</th>'
            f'{own_head}<th>Fair</th><th>Market</th><th>Price</th><th>Edge</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>'
            '<p class="mute filter-empty" hidden>No fixtures in the selected leagues.</p></div>')


def _all_predictions_body(allp, link) -> str:
    """Every upcoming fixture with the model's call in each market, laid out like the
    settled-games table on the track record (without the result)."""
    rows = ""
    for lg, p in sorted(allp, key=lambda t: t[1].fixture.kickoff):
        cells = ""
        for market in ("1x2", "btts", "gg2", "ou"):
            call, cp = _model_pick(p, market)
            cells += f'<td class="nw"><b>{call.replace(" 2.5", "")}</b> <span class="mute">{pct(cp)}</span></td>'
        h, a_, q = p.ensemble["top_scores"][0]
        vals = ", ".join(f'{html.escape(k["market"].replace(" (value)", ""))} {html.escape(k["pick"])} '
                         f'@{k["price"]:.2f} <span class="hit">{k["edge"]:+.0%}</span>'
                         for k in p.picks if k.get("value")) or "–"
        rows += (f'<tr data-league="{slug(lg)}"><td class="nw">{_local(p.fixture.kickoff)}</td>'
                 f'<td class="l">{html.escape(lg)}</td><td class="l">{link(lg, p.fixture)}</td>{cells}'
                 f'<td class="nw">{h}-{a_} <span class="mute">{pct(q)}</span></td><td>{vals}</td></tr>')
    return ('<div class="card"><div class="mute" style="margin-bottom:8px">The model\'s call in each market '
            'for every upcoming fixture, in kick-off order: the outcome it rates most likely, with its '
            'probability. Same layout as the settled games on the track record, which add the result. '
            'Click a match for the full analysis.</div>'
            '<div class="filter-wrap" data-key="predictions">' + filter_chips(lg for lg, _ in allp)
            + '<div class="scroll"><table class="filterable"><thead><tr><th>Kick-off</th><th class="l">League</th>'
            '<th class="l">Match</th><th>1X2</th><th>BTTS</th><th>GG2+</th><th>O/U 2.5</th>'
            '<th>Likeliest score</th><th>Value picks</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>'
            '<p class="mute filter-empty" hidden>No fixtures in the selected leagues.</p></div></div>')


def _values_body(values, link) -> str:
    note = ('<div class="card note"><b>Treat this as a list of where the model disagrees with the price, '
            'not as bets to place.</b> A pick is listed when the model\'s probability × the bookmaker\'s odds '
            'clears that market\'s edge threshold and the models stay within 10 points of the market\'s own '
            'view. In backtests, betting every value pick lost money; the track record shows the current '
            'figures.</div>')
    if not values:
        return note + '<div class="card mute">No value picks in this run. They need bookmaker odds (from ESPN or ' \
                      '<code>odds add</code>) and a gap between the model and the price.</div>'
    rows = ""
    for lg, p, k in values:
        rows += (f'<tr data-league="{slug(lg)}"><td class="nw">{_local(p.fixture.kickoff)}</td>'
                 f'<td class="l">{html.escape(lg)}</td><td class="l">{link(lg, p.fixture)}</td>'
                 f'<td>{html.escape(k["market"].replace(" (value)", ""))}</td><td>{html.escape(k["pick"])}</td>'
                 f'<td><b>{pct(k["prob"])}</b></td>'
                 f'<td>{pct(k["market_prob"]) if k.get("market_prob") is not None else "–"}</td>'
                 f'<td>{k["price"]:.2f} <span class="mute">{html.escape(str(k.get("price_source") or ""))}</span></td>'
                 f'<td><b>{k["edge"]:+.0%}</b></td><td>{k["fair"]:.2f}</td></tr>')
    return (note + '<div class="card"><div class="filter-wrap" data-key="value">'
            + filter_chips(lg for lg, _, _ in values)
            + '<div class="scroll"><table class="filterable"><thead><tr><th>Kick-off</th><th class="l">League</th>'
            '<th class="l">Match</th><th>Market</th><th>Pick</th><th>Model</th><th>Market view</th>'
            '<th>Price</th><th>Edge</th><th>Fair</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>'
            '<p class="mute filter-empty" hidden>No value picks in the selected leagues.</p></div></div>')


def _draws_table(ranked, link, limit: int | None = None, total: int | None = None) -> str:
    """Draw-probability table with a league filter. ``limit`` rows are shown after filtering;
    ``total`` (on the "view more" page) is how many fixtures the shown top share came from."""
    rows = ""
    for i, (lg, p) in enumerate(ranked, 1):
        e, fx = p.ensemble, p.fixture
        h, a_, q = e["top_scores"][0]
        mk = p.market.get("implied", {}).get("draw")
        call, cp = _model_pick(p, "1x2")
        hidden = " hidden" if limit and i > limit else ""   # without JavaScript: plain top N
        rows += (
            f'<tr data-league="{slug(lg)}"{hidden}><td class="rk">{i}</td><td>{_local(fx.kickoff)}</td>'
            f"<td>{html.escape(lg)}</td><td>{link(lg, fx)}</td><td><b>{pct(e['draw'])}</b></td>"
            f'<td class="nw"><b>{call}</b> <span class="mute">{pct(cp)}</span></td>'
            f"<td>{e['home'] * 100:.0f} / {e['away'] * 100:.0f}</td>"
            f"<td>{e['xg_home']:.2f}–{e['xg_away']:.2f}</td><td>{h}-{a_} ({pct(q)})</td>"
            f"<td>{pct(e['under_2.5'])}</td><td>{pct(mk) if mk else '–'}</td><td>{1 / e['draw']:.2f}</td></tr>")
    scope = (f"top {limit} of the leagues selected below" if limit else
             f"the top 20%: {len(ranked)} of {total} fixtures" if total else "every fixture")
    return (
        '<div class="filter-wrap" data-key="draws">'
        f'<div class="mute" style="margin:4px 0 8px">Fixtures ranked by ensemble draw probability ({scope}). '
        'Draws rarely top 31%; a high value means an evenly matched, low-scoring game, '
        'not that a draw is expected. "Model\'s pick" is the outcome the model actually rates most likely.</div>'
        f'{filter_chips(lg for lg, _ in ranked)}'
        f'<div class="scroll"><table class="filterable" data-limit="{limit or 0}"><thead>'
        '<tr><th>#</th><th>Kick-off</th><th>League</th><th>Match</th><th>Draw</th><th>Model\'s pick</th>'
        '<th>1 / 2</th>'
        '<th>Exp. goals</th><th>Likeliest score</th><th>U2.5</th><th>Market draw</th><th>Fair</th></tr>'
        f'</thead><tbody>{rows}</tbody></table></div>'
        '<p class="mute filter-empty" hidden>No fixtures in the selected leagues.</p></div>')


def html_report(preds_by_league: dict, path: Path, generated: datetime,
                track_record: str | None = "track_record.html", idle: dict | None = None,
                days: int | None = None, repo: str | None = None,
                min_edge: dict | None = None) -> list[Path]:
    """Main page at ``path`` plus one page per league beside it. ``track_record`` is the
    track-record page to link to (None: not available yet). ``idle`` maps leagues with no
    fixtures in the window to their next kick-off (or None), so they still get a tile.
    Returns the files written."""
    path = Path(path)
    main_name = path.name
    tr_link = (f'<a href="{html.escape(track_record)}">Track record</a>' if track_record
               else '<span class="mute">Track record (none yet)</span>')
    nav = (f'<a href="{html.escape(main_name)}">← All leagues</a> · <a href="{PREDICTIONS_FILE}">All predictions</a>'
           f' · <a href="{DRAWS_FILE}">Draws</a> · <a href="{ODDS_FILE}">Enter odds</a> · {tr_link}')
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

    # ---- strongest calls: top 10 per market on the main page, top 20% on its own page
    pick_of = lambda p, mkt: next(k for k in p.picks if k["market"] == mkt)

    def from_pick(label):
        return lambda p: (pick_of(p, label)["prob"], pick_of(p, label)["pick"], pick_of(p, label)["key"])

    market_pages = [
        ("1X2", "picks-1x2.html", from_pick("1X2")),
        ("Both teams to score", "picks-btts.html", from_pick("BTTS")),
        ("Over / Under 2.5", "picks-ou25.html", from_pick("O/U 2.5")),
        ("GG2+ — likeliest Yes", "picks-gg2.html", lambda p: (p.ensemble["gg2_yes"], "Yes", "gg2_yes")),
    ]
    boards = []
    for title, fname, get in market_pages:
        items = sorted(((*get(p), lg, p) for lg, p in allp), key=lambda t: -t[0])
        rows = "".join(f"<tr><td>{link(lg, p.fixture)}</td><td>{pick}</td><td>{pct(q)}</td><td>{1 / q:.2f}</td></tr>"
                       for q, pick, _, lg, p in items[:10])
        k = more_count(len(items), 10)
        boards.append(f'<div class="box"><h3>{title}</h3><table><tr><th>Match</th><th>Pick</th><th>Prob.</th>'
                      f'<th>Fair</th></tr>{rows}</table>'
                      f'<div style="margin-top:6px;text-align:right"><a href="{fname}">View more →</a>'
                      f'</div></div>')
        out = path.with_name(fname)
        gg2 = fname == "picks-gg2.html"
        intro = (f"The {k} fixtures (top 20% of {len(items)}) where GG2+ Yes is most likely. Even here the "
                 "probability usually stays below 50%, so the model's own pick, in the \"Model's pick\" column, "
                 "is mostly <b>No</b>: read this as where both teams scoring twice is most plausible, not as "
                 "predictions that it will happen." if gg2 else
                 f"The {k} fixtures (top 20% of {len(items)}) where the model is most confident in this market.")
        body = (f'<div class="card"><div class="mute" style="margin-bottom:8px">{intro} "Market" is the '
                f'bookmakers\' view of the same selection (implied from their 1X2 + O/U odds); price and edge '
                f'appear where odds exist. For every fixture\'s call in every market, see '
                f'<a href="{PREDICTIONS_FILE}">all predictions</a>.</div>'
                + _picks_table(items[:k], link, "gg2" if gg2 else None) + "</div>")
        out.write_text(with_filter_js(page(f"Strongest calls: {title} (top 20%)", body, generated, nav)),
                       encoding="utf-8")
        written.append(out)

    # ---- value picks across all leagues
    values = sorted(((lg, p, k) for lg, p in allp for k in p.picks if k.get("value")),
                    key=lambda t: (t[1].fixture.kickoff, -(t[2]["edge"] or 0)))
    out = path.with_name(VALUES_FILE)
    out.write_text(with_filter_js(page("Value picks", _values_body(values, link), generated, nav)),
                   encoding="utf-8")
    written.append(out)

    # ---- every fixture, every market
    out = path.with_name(PREDICTIONS_FILE)
    out.write_text(with_filter_js(page("All predictions", _all_predictions_body(allp, link), generated, nav)),
                   encoding="utf-8")
    written.append(out)

    # ---- odds entry form
    out = path.with_name(ODDS_FILE)
    form = _odds_form_body(preds_by_league, link, repo, min_edge or Settings().min_edge)
    out.write_text(page("Enter odds", form, generated, nav).replace("</body>", f"<script>{ODDS_JS}</script></body>"),
                   encoding="utf-8")
    written.append(out)

    ranked = sorted(allp, key=lambda t: -t[1].ensemble["draw"])
    draws_card = (
        f'<div class="card"><div class="head"><h2>Likeliest draws</h2>'
        f'<a href="{DRAWS_FILE}">View more →</a></div>'
        + _draws_table(ranked, link, limit=20) + "</div>")
    k = more_count(len(ranked), 20)
    draws_page = page("Likeliest draws (top 20%)",
                      f'<div class="card">{_draws_table(ranked[:k], link, total=len(ranked))}'
                      f'<p class="mute" style="margin-top:8px">For every fixture\'s call in every market, see '
                      f'<a href="{PREDICTIONS_FILE}">all predictions</a>.</p></div>', generated, nav)
    out = path.with_name(DRAWS_FILE)
    out.write_text(with_filter_js(draws_page), encoding="utf-8")
    written.append(out)

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
    window = f"the next {days} days" if days else "this window"
    for league, nxt in (idle or {}).items():
        when = (f'Next: {nxt.astimezone().strftime("%a %d %b")}' if nxt is not None
                else "Next fixture not published yet")
        tiles += (f'<div class="box tile idle"><h3>{html.escape(league)}</h3>'
                  f'<div class="row"><b>No fixtures</b><span class="mute">in {window}</span></div>'
                  f'<div class="row mute"><span>{when}</span></div></div>')
    leagues_card = (f'<div class="card"><h2>Leagues</h2><div class="grid">{tiles}</div>'
                    f'<div class="row" style="margin-top:12px"><span class="mute">{len(values)} VALUE '
                    f'pick{"s" if len(values) != 1 else ""} across all leagues</span>'
                    f'<span><a href="{ODDS_FILE}">Enter odds</a> · '
                    f'<a href="{VALUES_FILE}">View all value picks →</a></span></div>'
                    f'<div class="row" style="margin-top:6px"><span class="mute">{len(allp)} fixtures: every '
                    f'market, one table</span><a href="{PREDICTIONS_FILE}"><b>View all predictions →</b></a>'
                    f'</div></div>')
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
    main = page("MBTgrandaudit predictions", body, generated,
                f'<a href="{PREDICTIONS_FILE}">All predictions</a> · <a href="{DRAWS_FILE}">Draws</a> · '
                f'<a href="{ODDS_FILE}">Enter odds</a> · {tr_link}')
    path.write_text(with_filter_js(main), encoding="utf-8")
    written.insert(0, path)
    return written
