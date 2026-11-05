"""ESPN public site API: fixtures, bookmaker odds and all-competition team schedules."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd

from .cache import fetch

API = "https://site.api.espn.com/apis/site/v2/sports/soccer"


@dataclass
class Fixture:
    event_id: str
    kickoff: datetime
    league: str
    home: str
    away: str
    home_espn_id: str | None = None
    away_espn_id: str | None = None
    status: str = "scheduled"
    venue: str | None = None
    odds: dict = field(default_factory=dict)   # decimal odds, keys: H D A over under line
    home_score: float | None = None
    away_score: float | None = None
    neutral: bool = False


def american_to_decimal(v) -> float | None:
    try:
        x = float(str(v).replace("+", ""))
    except (TypeError, ValueError):
        return None
    if x == 0:
        return None
    return 1 + x / 100 if x > 0 else 1 + 100 / abs(x)


def _score(c: dict) -> float | None:
    s = c.get("score")
    if isinstance(s, dict):
        s = s.get("value", s.get("displayValue"))
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _parse_odds(comp: dict) -> dict:
    try:
        return _parse_odds_unsafe(comp)
    except (AttributeError, TypeError, ValueError, KeyError):
        return {}   # odds are optional; never let a malformed block drop the fixture


def _parse_odds_unsafe(comp: dict) -> dict:
    out: dict = {}
    for o in comp.get("odds") or []:
        if not isinstance(o, dict):
            continue
        for key, side in (("H", "homeTeamOdds"), ("A", "awayTeamOdds"), ("D", "drawOdds")):
            v = american_to_decimal((o.get(side) or {}).get("moneyLine"))
            if v and key not in out:
                out[key] = v
        ml = o.get("moneyline") or {}
        for key, side in (("H", "home"), ("A", "away"), ("D", "draw")):
            v = american_to_decimal(((ml.get(side) or {}).get("close") or {}).get("odds"))
            if v and key not in out:
                out[key] = v
        if o.get("overUnder") is not None and "line" not in out:
            out["line"] = float(o["overUnder"])
            tot = o.get("total") or {}
            for key in ("over", "under"):
                v = american_to_decimal(((tot.get(key) or {}).get("close") or {}).get("odds"))
                if v:
                    out[key] = v
        if out:
            out["source"] = (o.get("provider") or {}).get("name", "ESPN")
            break
    return out


def _parse_events(text: str | None, espn_code: str) -> list[Fixture]:
    if not text:
        return []
    res = []
    for ev in json.loads(text).get("events", []):
        comp = ev["competitions"][0]
        teams = {c["homeAway"]: c for c in comp["competitors"]}
        if "home" not in teams or "away" not in teams:
            continue
        status = ev.get("status", {}).get("type", {})
        res.append(Fixture(
            event_id=ev["id"],
            kickoff=datetime.fromisoformat(ev["date"].replace("Z", "+00:00")),
            league=espn_code,
            home=teams["home"]["team"]["displayName"],
            away=teams["away"]["team"]["displayName"],
            home_espn_id=teams["home"]["team"]["id"],
            away_espn_id=teams["away"]["team"]["id"],
            status="completed" if status.get("completed") else status.get("state", "pre"),
            venue=(comp.get("venue") or {}).get("fullName"),
            odds=_parse_odds(comp),
            home_score=_score(teams["home"]),
            away_score=_score(teams["away"]),
            neutral=bool(comp.get("neutralSite")),
        ))
    return res


def _unique(fx: list[Fixture]) -> list[Fixture]:
    seen, uniq = set(), []
    for f in sorted(fx, key=lambda f: f.kickoff):
        if f.event_id not in seen:
            seen.add(f.event_id)
            uniq.append(f)
    return uniq


def fixtures(espn_code: str, start: date, days: int, offline: bool = False,
             ttl: float = 1800) -> list[Fixture]:
    def day(d: date) -> list[Fixture]:
        url = f"{API}/{espn_code}/scoreboard?dates={d:%Y%m%d}"
        return _parse_events(fetch(url, ttl=ttl, offline=offline), espn_code)

    with ThreadPoolExecutor(8) as ex:
        out = [f for lst in ex.map(day, [start + timedelta(i) for i in range(days)]) for f in lst]
    return _unique(out)


def month_results(espn_code: str, start: date, end: date, today: date | None = None,
                  offline: bool = False) -> list[Fixture]:
    """Every match of a competition from ``start``'s month to ``end``'s month (one scoreboard
    request per month: ESPN accepts dates=YYYYMM). Past months are cached for good."""
    today = today or date.today()
    months, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)

    def one(ym):
        yy, mm = ym
        done = (yy, mm) < (today.year, today.month) and not (
            (today.year, today.month) == ((yy + 1, 1) if mm == 12 else (yy, mm + 1)) and today.day <= 3)
        url = f"{API}/{espn_code}/scoreboard?dates={yy}{mm:02d}&limit=1000"
        try:
            return _parse_events(fetch(url, ttl=None if done else 3 * 3600, offline=offline), espn_code)
        except Exception:
            return []

    with ThreadPoolExecutor(6) as ex:
        out = [f for lst in ex.map(one, months) for f in lst]
    return _unique(out)


def team_info(team_id: str, offline: bool = False) -> tuple[str | None, str | None]:
    """(league slug, country code) of a team: the slug of its defaultLeague (e.g. 'aut.1';
    for clubs ESPN files under a UEFA competition it is that competition instead) and the
    country prefix of the team's own slug (e.g. 'aut' from 'aut.rb-salzburg')."""
    try:
        text = fetch(f"{API}/all/teams/{team_id}", ttl=30 * 86400, offline=offline)
        team = (json.loads(text).get("team") or {}) if text else {}
    except Exception:
        return None, None
    league = None
    for link in (team.get("defaultLeague") or {}).get("links") or []:
        href = link.get("href", "")
        if "/name/" in href:
            league = href.split("/name/")[1].split("/")[0] or None
            break
    league = league or team.get("leagueAbbrev") or None
    slug = str(team.get("slug") or "")
    return league, (slug.split(".")[0] if "." in slug else None)


def team_league(team_id: str, offline: bool = False) -> str | None:
    return team_info(team_id, offline)[0]


def _schedule_events(url: str, ttl: float | None, offline: bool) -> list[dict]:
    text = fetch(url, ttl=ttl, offline=offline)
    return json.loads(text).get("events", []) if text else []


def team_matches(team_id: str, espn_code: str, seasons: list[int], current: int,
                 offline: bool = False) -> pd.DataFrame:
    """Completed matches in all competitions for one team (ESPN team id)."""
    urls = []
    for y in seasons:
        ttl = 6 * 3600 if y >= current else None
        urls.append((f"{API}/all/teams/{team_id}/schedule?season={y}", ttl))
        urls.append((f"{API}/{espn_code}/teams/{team_id}/schedule?season={y}", ttl))
    rows, seen = [], set()
    for url, ttl in urls:
        try:
            events = _schedule_events(url, ttl, offline)
        except Exception:
            continue
        for ev in events:
            if ev["id"] in seen:
                continue
            comp = ev["competitions"][0]
            if not comp.get("status", ev.get("status", {})).get("type", {}).get("completed"):
                continue
            teams = {c["homeAway"]: c for c in comp["competitors"]}
            if "home" not in teams or "away" not in teams:
                continue
            side = "home" if teams["home"]["team"]["id"] == str(team_id) else "away"
            other = "away" if side == "home" else "home"
            gf, ga = _score(teams[side]), _score(teams[other])
            if gf is None or ga is None:
                continue
            seen.add(ev["id"])
            league = ev.get("league") or {}
            ts = pd.Timestamp(ev["date"])
            rows.append({
                "date": ts.tz_convert(None) if ts.tzinfo else ts,
                "competition": league.get("name") or league.get("abbreviation") or "",
                "venue": side,
                "opp_id": teams[other]["team"]["id"],
                "opponent": teams[other]["team"]["displayName"],
                "gf": gf,
                "ga": ga,
                "friendly": "friendly" in (str(league.get("slug", "")) + str(league.get("name", ""))).lower(),
            })
    if not rows:
        return pd.DataFrame(columns=["date", "competition", "venue", "opp_id", "opponent", "gf", "ga", "friendly"])
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
