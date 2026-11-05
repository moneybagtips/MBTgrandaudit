"""Read bookmaker odds from market-list screenshots (e.g. SportyBet's GG/NG 2+ tab).

A vision model turns each image into JSON (the selected market tab, the column headers,
and one row per fixture with its date, kick-off and prices). Everything after that is
plain code here: the market is decided from the selected tab by a fixed mapping, dates
come from the date separators, and each row is matched to a logged upcoming fixture.

The result is a set of proposed lines in the odds-form issue format
(`event id | market | odds | match`), which `odds from-issue` saves as they are.
"""
from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .data.names import BUILTIN, _norm, _user_aliases, match_team

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_MARGIN = 0.15          # a two-sided pair implying more overround than this is likely misread
MIN_MARGIN = -0.01

PROMPT = """You are reading a screenshot of a bookmaker's market list (for example SportyBet).
Return ONLY a JSON object, no prose, with this shape:

{
  "bookmaker": "<name if visible, else null>",
  "market_tab": "<text of the SELECTED market tab, exactly as shown>",
  "market_tab_confidence": "high" | "low",
  "columns": ["<column header>", ...],
  "line": <goal line as a number if the screen shows one (e.g. Over/Under 2.5), else null>,
  "rows": [
    {"date": "<MM/DD from the nearest date separator ABOVE this row, or null if none is visible above it>",
     "weekday": "<weekday word from that separator, or null>",
     "time": "<kick-off HH:MM, 24-hour>",
     "bookie_id": "<fixture id shown, or null>",
     "home": "<first team>", "away": "<second team>",
     "prices": [<decimal odds per column, in column order; null for a locked or blank box>]}
  ],
  "warnings": ["<anything cut off, unreadable or unusual>"]
}

Rules:
- The market is the SELECTED tab only: the one that is bold, highlighted or underlined. Never infer
  the market from the column headers (a "GG/NG 2+" tab and a "GG/NG" tab both have GG and NG columns).
  If the selected tab is cut off, unclear, or you cannot tell which tab is selected, set
  market_tab_confidence to "low".
- Copy prices exactly as shown. Do not compute or round. Fractional or American odds: convert to decimal.
- Ignore market counts (like "+1324"), badges (like "BEST ODDS"), stream, stats and favourite icons.
- A date separator (like "Saturday, 11/07") applies to every row below it until the next separator.
- Leave out a row that is cut off so its teams or prices can't be read, and say so in warnings.
- Do not guess anything you cannot read."""


# ---------------------------------------------------------------- the model call

def _media_type(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "webp": "image/webp", "gif": "image/gif"}.get(ext, "image/png")


def _parse_json(text: str) -> dict:
    """The first JSON object in a model reply (tolerates code fences and stray prose)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("the model reply holds no JSON object")
    return json.loads(m.group(0))


def read_image(path: Path, api_key: str | None = None, model: str | None = None,
               timeout: float = 120) -> dict:
    """Send one screenshot to the Claude API and return its parsed JSON reading."""
    import requests
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY is not set")
    data = base64.standard_b64encode(Path(path).read_bytes()).decode()
    body = {
        "model": model or os.environ.get("MBT_VISION_MODEL", DEFAULT_MODEL),
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": _media_type(Path(path)), "data": data}},
            {"type": "text", "text": PROMPT},
        ]}],
    }
    r = requests.post(API_URL, json=body, timeout=timeout,
                      headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                               "content-type": "application/json"})
    if r.status_code != 200:
        raise ValueError(f"Claude API error {r.status_code}: {r.text[:300]}")
    text = "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")
    return _parse_json(text)


# ---------------------------------------------------------------- market mapping

def _tab_key(text: str) -> str:
    t = (text or "").lower().replace(" ", "")
    t = t.replace("bts", "btts") if "btts" not in t else t
    return re.sub(r"[^a-z0-9/+.]", "", t)


def market_codes(tab: str, columns: list[str], line=None) -> list[str]:
    """Form market codes for each column of the selected tab, or ValueError saying why not.

    The tab decides the market; the columns only have to agree with it."""
    t = _tab_key(tab)
    cols = [c.strip().lower() for c in columns or []]
    if not t:
        raise ValueError("the selected market tab couldn't be read")
    if any(w in t for w in ("half", "1sthalf", "2ndhalf", "1h", "2h", "doublechance", "dc", "handicap",
                             "corner", "card", "exact", "correct", "odd/even", "oddeven", "draw")) \
            and t not in {"1x2"}:
        raise ValueError(f"the '{tab}' market isn't supported")
    if re.fullmatch(r"(gg/ng|btts)(2\+|2plus|\+2)", t) or t in {"gg/ng2+", "gg2+/ng2+", "btts2+", "gg/ng2"}:
        codes, want = ["gg2", "ng2"], [("gg", "yes"), ("ng", "no")]
    elif t in {"gg/ng", "btts", "gg/nogg", "bothteamstoscore", "btts/nobtts"}:
        codes, want = ["gg", "ng"], [("gg", "yes"), ("ng", "no")]
    elif t == "1x2":
        codes, want = ["1", "X", "2"], [("1", "home"), ("x", "draw"), ("2", "away")]
    elif t in {"o/u", "over/under", "overunder", "goalsover/under", "totalgoals"} or t.startswith("over/under"):
        m = re.search(r"(\d+\.\d)", t)
        ln = line if line is not None else (float(m.group(1)) if m else None)
        if ln is None:
            raise ValueError("an Over/Under screen with no visible goal line")
        ln = float(ln)
        if abs(ln * 2 - round(ln * 2)) > 1e-9 or round(ln * 2) % 2 == 0:
            raise ValueError(f"Over/Under {ln:g}: only half-goal lines (0.5, 1.5, 2.5 ...) are supported")
        codes, want = [f"o{ln:g}", f"u{ln:g}"], [("o", "over"), ("u", "under")]
    else:
        raise ValueError(f"the '{tab}' market isn't supported")
    if len(cols) != len(codes):
        raise ValueError(f"the '{tab}' tab should have {len(codes)} columns but the screenshot shows "
                         f"{len(cols)} ({', '.join(columns or []) or 'none'})")
    for c, alts in zip(cols, want):
        if not any(c == a or c.startswith(a) for a in alts):
            raise ValueError(f"the '{tab}' tab's columns ({', '.join(columns)}) don't fit that market")
    return codes


# ---------------------------------------------------------------- dates and teams

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def resolve_date(md: str | None, today: date, weekday: str | None = None) -> tuple[date | None, str | None]:
    """'11/07' -> the next 7 November on or after yesterday; (date, problem)."""
    if not md:
        return None, None
    m = re.fullmatch(r"\s*(\d{1,2})[/.-](\d{1,2})\s*", md)
    if not m:
        return None, f"unreadable date '{md}'"
    month, day = int(m.group(1)), int(m.group(2))
    for year in (today.year, today.year + 1):
        try:
            d = date(year, month, day)
        except ValueError:
            return None, f"impossible date '{md}'"
        if d >= today - timedelta(days=1):
            break
    if weekday:
        w = weekday.strip().lower()
        if w in _WEEKDAYS and _WEEKDAYS[d.weekday()] != w:
            return d, f"{md} is a {_WEEKDAYS[d.weekday()].title()}, not a {weekday.title()}"
    return d, None


def _reverse_aliases() -> dict[str, set[str]]:
    rev: dict[str, set[str]] = {}
    for table in (BUILTIN, _user_aliases()):
        for a, b in table.items():
            for x, y in ((a, b), (b, a)):
                rev.setdefault(_norm(x), set()).add(_norm(y))
    return rev


def same_team(screen: str, logged: str, rev: dict | None = None) -> bool:
    """Does a bookmaker's team name ('Man City') name this logged team ('Manchester City')?"""
    s, lg = _norm(screen), _norm(logged)
    if not s or not lg:
        return False
    if s == lg:
        return True
    rev = _reverse_aliases() if rev is None else rev
    if lg in rev.get(s, ()) or s in rev.get(lg, ()):
        return True
    return match_team(screen, [logged]) == logged or match_team(logged, [screen]) == screen


# ---------------------------------------------------------------- the import

@dataclass
class Proposal:
    lines: list[tuple[str, str, float, str]] = field(default_factory=list)   # event, code, odds, label
    review: list[dict] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)     # whole screenshots that gave nothing
    bookmaker: str = ""


def _match_row(row: dict, d: date | None, events: pd.DataFrame, rev: dict):
    hits = [e for e in events.itertuples()
            if same_team(row.get("home", ""), e.home, rev) and same_team(row.get("away", ""), e.away, rev)]
    if d is not None and hits:
        near = [e for e in hits if abs((pd.Timestamp(e.kickoff).date() - d).days) <= 1]
        hits = near
    if len(hits) > 1 and row.get("time") and d is not None:
        try:
            want = pd.Timestamp(f"{d} {row['time']}")
            best = min(hits, key=lambda e: abs(pd.Timestamp(e.kickoff).tz_localize(None) - want))
            hits = [best]
        except (ValueError, TypeError):
            pass
    return hits


def propose(readings: list[dict], events: pd.DataFrame, now: datetime,
            predictions: dict | None = None, default_bookmaker: str = "SportyBet") -> Proposal:
    """Turn the model's readings (one per screenshot, in order) into proposed price lines.

    `events` = logged upcoming fixtures (event_id, home, away, kickoff); `predictions` maps an
    event id to its markets dict, for the edge column of the review table."""
    out, rev, carry = Proposal(), _reverse_aliases(), None
    today = now.date()
    seen: dict[tuple[str, str], int] = {}
    for i, rd in enumerate(readings, 1):
        tag = f"Screenshot {i}" if len(readings) > 1 else "Screenshot"
        if rd.get("bookmaker") and not out.bookmaker:
            out.bookmaker = str(rd["bookmaker"])
        if rd.get("error"):
            out.errors.append(f"{tag}: couldn't be read ({rd['error']})")
            continue
        try:
            if str(rd.get("market_tab_confidence", "")).lower() != "high":
                raise ValueError(f"the selected market tab isn't clear (read as '{rd.get('market_tab')}')")
            codes = market_codes(rd.get("market_tab", ""), rd.get("columns") or [], rd.get("line"))
        except ValueError as e:
            out.errors.append(f"{tag}: no prices proposed, {e}. Reply with the lines and the market filled in.")
            continue
        for w in rd.get("warnings") or []:
            out.skipped.append(f"- {tag}: {w}")
        for row in rd.get("rows") or []:
            label_in = f"{row.get('home', '?')} v {row.get('away', '?')}"
            d, problem = resolve_date(row.get("date"), today, row.get("weekday"))
            if d is None and not row.get("date"):
                d = carry                      # rows above the first separator: the previous date
            elif d is not None:
                carry = d
            if problem:
                out.skipped.append(f"- {label_in}: skipped, {problem}")
                continue
            hits = _match_row(row, d, events, rev)
            if not hits:
                out.skipped.append(f"- {label_in}: no logged upcoming fixture matches")
                continue
            if len(hits) > 1:
                out.skipped.append(f"- {label_in}: matches more than one fixture ("
                                   + "; ".join(f"{e.home} v {e.away} {str(e.kickoff)[:10]}" for e in hits) + ")")
                continue
            ev = hits[0]
            if pd.Timestamp(ev.kickoff).tz_localize(None) <= pd.Timestamp(now).tz_localize(None):
                out.skipped.append(f"- {ev.home} v {ev.away}: skipped, already kicked off")
                continue
            prices = list(row.get("prices") or [])
            if len(prices) != len(codes):
                out.skipped.append(f"- {label_in}: {len(prices)} prices for {len(codes)} columns")
                continue
            vals = []
            for p in prices:
                try:
                    vals.append(float(str(p).replace(",", ".")) if p is not None else None)
                except ValueError:
                    vals.append(None)
            if any(v is not None and v <= 1 for v in vals):
                out.skipped.append(f"- {label_in}: skipped, a price at or below 1.00 ({prices})")
                continue
            if all(v is not None for v in vals):
                margin = sum(1 / v for v in vals) - 1
                if not MIN_MARGIN <= margin <= MAX_MARGIN:
                    out.skipped.append(f"- {label_in}: skipped, the prices {prices} imply a {margin:+.0%} "
                                       "margin, likely a misread")
                    continue
            label = f"{ev.home} v {ev.away}"
            mk = (predictions or {}).get(str(ev.event_id)) or {}
            for code, v in zip(codes, vals):
                if v is None:
                    continue
                k = (str(ev.event_id), code)
                if k in seen:                  # the same price on a later screenshot wins
                    old = out.lines[seen[k]]
                    if abs(old[2] - v) > 1e-9:
                        out.skipped.append(f"- {label} {code}: {old[2]:.2f} replaced by {v:.2f} from a later screenshot")
                    out.lines[seen[k]] = (k[0], code, v, label)
                    continue
                seen[k] = len(out.lines)
                out.lines.append((k[0], code, v, label))
            out.review.append({"date": str(d) if d else "", "time": row.get("time") or "", "read": label_in,
                               "match": label, "event": str(ev.event_id), "codes": codes, "prices": vals,
                               "model": mk})
    out.bookmaker = out.bookmaker or default_bookmaker
    return out


_KEY = {"1": "home", "X": "draw", "2": "away", "gg": "btts_yes", "ng": "btts_no",
        "gg2": "gg2_yes", "ng2": "gg2_no"}


def _model_key(code: str) -> str:
    if code in _KEY:
        return _KEY[code]
    return ("over_" if code[0] == "o" else "under_") + code[1:]


def report(p: Proposal) -> str:
    """Markdown reply: the review table, the skipped rows and the proposed lines."""
    out = []
    if p.lines:
        out.append(f"Read **{len(p.lines)}** price{'s' if len(p.lines) != 1 else ''} from "
                   f"{len(p.review)} fixture{'s' if len(p.review) != 1 else ''}. Check them against the screenshot.")
        out += ["", "| Date | Time | As read | Matched fixture | Market | Odds | Model | Edge |",
                "|---|---|---|---|---|---|---|---|"]
        for r in p.review:
            for code, v in zip(r["codes"], r["prices"]):
                if v is None:
                    continue
                prob = r["model"].get(_model_key(code))
                model = f"{prob:.1%}" if prob else "–"
                edge = f"{prob * v - 1:+.1%}" if prob else "–"
                out.append(f"| {r['date']} | {r['time']} | {r['read']} | {r['match']} | {code} | {v:.2f} | "
                           f"{model} | {edge} |")
    else:
        out.append("No prices could be proposed from the screenshot.")
    if p.errors:
        out += [""] + [f"**{e}**" for e in p.errors]
    if p.skipped:
        out += ["", "Not proposed:"] + p.skipped
    if p.lines:
        out += ["", "Proposed lines (the format `odds from-issue` reads):", "", "```",
                f"Bookmaker: {p.bookmaker}"]
        out += [f"{ev} | {code} | {v:.2f} | {label}" for ev, code, v, label in p.lines]
        out += ["```"]
    return "\n".join(out)


def upcoming_events(store, now: datetime) -> pd.DataFrame:
    return store.find_events("", "", after=now - timedelta(hours=3))


def predictions_for(store, event_ids) -> dict:
    out = {}
    for ev in event_ids:
        pred = store.latest_prediction(str(ev))
        if pred is not None:
            try:
                out[str(ev)] = json.loads(pred["markets"])
            except (TypeError, ValueError):
                pass
    return out
