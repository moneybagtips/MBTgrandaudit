"""API-Football (api-sports.io) pre-match odds: every bookmaker's 1X2, O/U 0.5-4.5 and BTTS.

The free plan allows 100 requests a day and 10 a minute, only for dates from yesterday to
tomorrow, and blocks league/season filters for the current season. So odds are fetched
one fixture at a time (odds?fixture=ID), after one fixtures?date call that maps
API-Football's fixture ids onto the logged ESPN fixtures by team names and kick-off.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from datetime import datetime

import pandas as pd

from ..config import ROOT
from .names import match_team

BASE = "https://v3.football.api-sports.io/"
ENV_FILE = ROOT / ".env"
LINES = ("0.5", "1.5", "2.5", "3.5", "4.5")
MAX_KICKOFF_GAP_H = 3          # API-Football and ESPN kick-off times rarely differ by more


def api_key() -> str | None:
    """API_FOOTBALL_KEY from the environment, else from a local .env file."""
    key = os.environ.get("API_FOOTBALL_KEY")
    if not key and ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*API_FOOTBALL_KEY\s*=\s*(.+?)\s*$", line)
            if m:
                key = m.group(1).strip("'\"")
    return key or None


class QuotaExhausted(RuntimeError):
    pass


class Client:
    """Spaces requests to the per-minute limit and stops at a request budget."""

    def __init__(self, key: str, budget: int = 90, min_interval: float = 6.5, opener=None):
        self.key, self.budget, self.min_interval = key, budget, min_interval
        self.used, self._last = 0, 0.0
        self.remaining: int | None = None          # daily requests left, from the response header
        self._open = opener or urllib.request.urlopen

    def get(self, path: str) -> dict:
        if self.used >= self.budget or self.remaining == 0:
            raise QuotaExhausted(f"request budget used ({self.used} this run, {self.remaining} left today)")
        wait = self._last + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(BASE + path, headers={"x-apisports-key": self.key})
        with self._open(req, timeout=30) as r:
            data = json.load(r)
            left = r.headers.get("x-ratelimit-requests-remaining")
        self._last, self.used = time.monotonic(), self.used + 1
        if left is not None and str(left).isdigit():
            self.remaining = int(left)
        errors = data.get("errors")
        if errors:
            text = json.dumps(errors)
            if re.search(r"(?i)request|limit", text):
                raise QuotaExhausted(text)
            raise RuntimeError(f"API-Football error on {path.split('?')[0]}: {text}")
        return data

    def fixtures(self, day) -> list[dict]:
        return self.get(f"fixtures?date={pd.Timestamp(day):%Y-%m-%d}")["response"]

    def odds(self, fixture_id: int) -> list[dict]:
        return self.get(f"odds?fixture={int(fixture_id)}")["response"]


def parse_odds(response: list[dict]) -> list[tuple[str, str, float]]:
    """(bookmaker, markets() key, decimal odds) for 1X2, O/U 0.5-4.5 and BTTS."""
    out = []
    for item in response:
        for bm in item.get("bookmakers", []):
            book = bm.get("name") or f"book {bm.get('id')}"
            for bet in bm.get("bets", []):
                name = bet.get("name")
                for v in bet.get("values", []):
                    key = _key(name, str(v.get("value")))
                    try:
                        odd = float(v.get("odd"))
                    except (TypeError, ValueError):
                        continue
                    if key and odd > 1.0:
                        out.append((book, key, odd))
    return out


def _key(bet: str | None, value: str) -> str | None:
    if bet == "Match Winner":
        return {"Home": "home", "Draw": "draw", "Away": "away"}.get(value)
    if bet == "Both Teams Score":
        return {"Yes": "btts_yes", "No": "btts_no"}.get(value)
    if bet == "Goals Over/Under":
        m = re.fullmatch(r"(Over|Under) (\d+(?:\.\d+)?)", value)
        if m and m.group(2) in LINES:
            return f"{m.group(1).lower()}_{m.group(2)}"
    return None


def match_fixtures(ours: pd.DataFrame, api: list[dict]) -> dict[str, int]:
    """{our event_id: API-Football fixture id}; both teams must match and kick-offs agree."""
    rows = [{"id": f["fixture"]["id"], "kickoff": pd.Timestamp(f["fixture"]["date"]).tz_convert("UTC"),
             "home": f["teams"]["home"]["name"], "away": f["teams"]["away"]["name"]} for f in api]
    out = {}
    for ev in ours.itertuples():
        ko = pd.Timestamp(ev.kickoff)
        ko = ko.tz_localize("UTC") if ko.tzinfo is None else ko.tz_convert("UTC")
        near = [r for r in rows if abs((r["kickoff"] - ko).total_seconds()) <= MAX_KICKOFF_GAP_H * 3600]
        home = match_team(ev.home, sorted({r["home"] for r in near}))
        if home is None:
            continue
        same = [r for r in near if r["home"] == home]
        away = match_team(ev.away, sorted({r["away"] for r in same}))
        hit = [r for r in same if r["away"] == away]
        if len(hit) == 1:
            out[str(ev.event_id)] = hit[0]["id"]
    return out


def fetch(store, day: datetime, now: datetime, client: Client, log=print, skip_hours: float = 12) -> dict:
    """Store every bookmaker's prices for the logged fixtures kicking off on ``day`` (UTC).

    Fixtures already fetched within ``skip_hours`` are skipped, so a re-run spends no
    requests on them. Earliest kick-offs go first in case the budget runs out."""
    start = pd.Timestamp(day).tz_localize(None).normalize().tz_localize("UTC")
    start = max(start, pd.Timestamp(now).tz_convert("UTC"))
    end = pd.Timestamp(day).tz_localize(None).normalize().tz_localize("UTC") + pd.Timedelta(days=1)
    stats = {"logged": 0, "matched": 0, "fetched": 0, "with_odds": 0, "rows": 0, "skipped": 0}
    if start >= end:
        return stats
    ours = store.query("SELECT DISTINCT event_id, league, home, away, kickoff FROM predictions"
                       " WHERE source = 'live' AND kickoff >= ? AND kickoff < ? ORDER BY kickoff",
                       (start.floor("s").isoformat(), end.isoformat()))
    stats["logged"] = len(ours)
    if ours.empty:
        return stats
    recent = store.query("SELECT DISTINCT event_id FROM market_odds WHERE kind = 'prediction' AND captured_at >= ?",
                         ((pd.Timestamp(now).tz_convert("UTC") - pd.Timedelta(hours=skip_hours)).floor("s").isoformat(),))
    done = set(recent["event_id"].astype(str))
    ids = match_fixtures(ours, client.fixtures(start))
    stats["matched"] = len(ids)
    for ev in ours.itertuples():
        ev_id = str(ev.event_id)
        if ev_id not in ids:
            log(f"  ? {ev.league} {ev.home} v {ev.away}: no API-Football fixture matched")
            continue
        if ev_id in done:
            stats["skipped"] += 1
            continue
        try:
            prices = parse_odds(client.odds(ids[ev_id]))
        except QuotaExhausted as e:
            log(f"  stopping: {e}")
            break
        stats["fetched"] += 1
        if prices:
            store.add_market_odds_many([(ev_id, key, odd, now, "prediction", book) for book, key, odd in prices])
            store.commit()
            stats["with_odds"] += 1
            stats["rows"] += len(prices)
        log(f"  {ev.league} {ev.home} v {ev.away}: {len(prices)} prices from "
            f"{len({b for b, _, _ in prices})} bookmakers")
    return stats
