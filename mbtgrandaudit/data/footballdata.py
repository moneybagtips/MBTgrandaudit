"""Historical league results from football-data.co.uk (results, xG, odds)."""
from __future__ import annotations

import io
from datetime import date

import numpy as np
import pandas as pd

from .cache import fetch

BASE = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
FIXTURES = "https://www.football-data.co.uk/fixtures.csv"
EXTRA = "https://www.football-data.co.uk/new/{code}.csv"
NEW_FIXTURES = "https://www.football-data.co.uk/new_league_fixtures.csv"
EXTRA_COUNTRY = {"SWZ": "Switzerland", "USA": "USA", "DNK": "Denmark", "NOR": "Norway", "JPN": "Japan"}
# "extra" leagues: one file per country holding every season; code -> league name in the file
EXTRA_LEAGUES = {"SWZ": "Super League", "USA": "MLS", "DNK": "Superliga", "NOR": "Eliteserien",
                 "JPN": "J1 League"}

COLUMNS = ["date", "div", "home", "away", "hg", "ag", "hxg", "axg", "hs", "as", "hst", "ast",
           "odds_h", "odds_d", "odds_a", "odds_o25", "odds_u25",
           "odds_ch", "odds_cd", "odds_ca", "odds_co25", "odds_cu25"]


def season_start_year(d: date) -> int:
    return d.year if d.month >= 7 else d.year - 1


def league_season(d: date, calendar: bool = False) -> int:
    """Season label (its starting year) that date ``d`` falls in."""
    return d.year if calendar else season_start_year(d)


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def _first(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam"])
    out = pd.DataFrame({
        "date": pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce", format="mixed"),
        "div": raw["Div"].astype(str),
        "home": raw["HomeTeam"].astype(str).str.strip(),
        "away": raw["AwayTeam"].astype(str).str.strip(),
        "hg": _first(raw, ["FTHG", "HG"]),
        "ag": _first(raw, ["FTAG", "AG"]),
        "hxg": _first(raw, ["HxG"]),
        "axg": _first(raw, ["AxG"]),
        "hs": _first(raw, ["HS"]),
        "as": _first(raw, ["AS"]),
        "hst": _first(raw, ["HST"]),
        "ast": _first(raw, ["AST"]),
        "odds_h": _first(raw, ["AvgH", "BbAvH", "B365H"]),
        "odds_d": _first(raw, ["AvgD", "BbAvD", "B365D"]),
        "odds_a": _first(raw, ["AvgA", "BbAvA", "B365A"]),
        "odds_o25": _first(raw, ["Avg>2.5", "BbAv>2.5", "B365>2.5"]),
        "odds_u25": _first(raw, ["Avg<2.5", "BbAv<2.5", "B365<2.5"]),
        "odds_ch": _first(raw, ["AvgCH", "B365CH"]),          # closing odds (2019-20 onwards)
        "odds_cd": _first(raw, ["AvgCD", "B365CD"]),
        "odds_ca": _first(raw, ["AvgCA", "B365CA"]),
        "odds_co25": _first(raw, ["AvgC>2.5", "B365C>2.5"]),
        "odds_cu25": _first(raw, ["AvgC<2.5", "B365C<2.5"]),
    })
    return out.dropna(subset=["date"])


def load_season(div: str, start_year: int, today: date, offline: bool = False) -> pd.DataFrame:
    current = start_year >= season_start_year(today)
    text = fetch(BASE.format(season=season_code(start_year), div=div),
                 ttl=6 * 3600 if current else None, offline=offline)
    if not text or "HomeTeam" not in text[:2000]:
        return pd.DataFrame(columns=COLUMNS)
    raw = pd.read_csv(io.StringIO(text), on_bad_lines="skip", low_memory=False)
    df = _normalise(raw)
    df["season"] = start_year
    return df


def load_extra(code: str, offline: bool = False) -> pd.DataFrame:
    """All seasons of an "extra" league (football-data.co.uk/new/<code>.csv).

    These files carry results and closing odds only (no shots, xG or pre-match odds);
    seasons are labelled "2026" or "2025/2026", stored here as the starting year."""
    text = fetch(EXTRA.format(code=code), ttl=6 * 3600, offline=offline)
    if not text or "Home" not in text[:500]:
        return pd.DataFrame(columns=COLUMNS + ["season"])
    raw = pd.read_csv(io.StringIO(text), on_bad_lines="skip", low_memory=False)
    raw = raw[raw["League"].astype(str).str.strip() == EXTRA_LEAGUES[code]]
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce", format="mixed"),
        "div": code,
        "home": raw["Home"].astype(str).str.strip(),
        "away": raw["Away"].astype(str).str.strip(),
        "hg": pd.to_numeric(raw["HG"], errors="coerce"),
        "ag": pd.to_numeric(raw["AG"], errors="coerce"),
        "odds_ch": _first(raw, ["AvgCH", "PSCH"]),
        "odds_cd": _first(raw, ["AvgCD", "PSCD"]),
        "odds_ca": _first(raw, ["AvgCA", "PSCA"]),
        "season": pd.to_numeric(raw["Season"].astype(str).str[:4], errors="coerce"),
    })
    for col in COLUMNS:
        if col not in df:
            df[col] = np.nan
    return df.dropna(subset=["date", "season"]).astype({"season": int})


def load_history(divs: list[str], seasons: int, today: date, offline: bool = False,
                 calendar: bool = False) -> pd.DataFrame:
    """Results for ``divs`` over the current and previous ``seasons-1`` seasons."""
    start = league_season(today, calendar)
    first = start - seasons + 1
    frames = []
    for div in divs:
        if div in EXTRA_LEAGUES:
            ex = load_extra(div, offline)
            frames.append(ex[(ex["season"] >= first) & (ex["season"] <= start)])
        else:
            frames += [load_season(div, y, today, offline) for y in range(first, start + 1)]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=COLUMNS + ["season"])
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("date").reset_index(drop=True)


def _kickoff(dates: pd.Series, times: pd.Series | None) -> pd.Series:
    """football-data dates + UK local times -> UTC timestamps (midday if no time given)."""
    t = times.fillna("12:00").astype(str) if times is not None else "12:00"
    local = pd.to_datetime(dates.astype(str) + " " + t, dayfirst=True, errors="coerce", format="mixed")
    return local.dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")


def load_fixtures(div: str, offline: bool = False, ttl: float = 3 * 3600) -> pd.DataFrame:
    """Upcoming fixtures published by football-data.co.uk, with average 1X2 (and, for the
    main leagues, O/U 2.5) odds. Covers the next few days only; ESPN is the primary source."""
    extra = div in EXTRA_LEAGUES
    text = fetch(NEW_FIXTURES if extra else FIXTURES, ttl=ttl, offline=offline)
    marker = "Home" if extra else "HomeTeam"
    if not text or marker not in text[:2000]:
        return pd.DataFrame(columns=COLUMNS + ["kickoff"])
    raw = pd.read_csv(io.StringIO(text), on_bad_lines="skip", low_memory=False)
    if raw.empty:
        return pd.DataFrame(columns=COLUMNS + ["kickoff"])
    if extra:
        raw = raw[(raw["Country"].astype(str).str.strip() == EXTRA_COUNTRY[div])
                  & (raw["League"].astype(str).str.strip() == EXTRA_LEAGUES[div])]
        df = pd.DataFrame({"date": pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce", format="mixed"),
                           "div": div, "home": raw["Home"].astype(str).str.strip(),
                           "away": raw["Away"].astype(str).str.strip(),
                           "odds_h": _first(raw, ["AvgH", "PSH"]), "odds_d": _first(raw, ["AvgD", "PSD"]),
                           "odds_a": _first(raw, ["AvgA", "PSA"])})
        for col in COLUMNS:
            if col not in df:
                df[col] = np.nan
    else:
        df = _normalise(raw)
        raw = raw.loc[df.index]
    df["kickoff"] = _kickoff(raw["Date"], raw["Time"] if "Time" in raw else None)
    return df[df["div"] == div].dropna(subset=["date"]).reset_index(drop=True)


def fd_event_id(div: str, day, home: str, away: str) -> str:
    """Event id for fixtures that only football-data knows (same format as the backfill)."""
    return f"fd:{div}:{pd.Timestamp(day):%Y%m%d}:{home}:{away}"


def parse_fd_event_id(event_id: str) -> tuple[str, pd.Timestamp, str, str] | None:
    parts = str(event_id).split(":", 4)
    if len(parts) != 5 or parts[0] != "fd":
        return None
    return parts[1], pd.Timestamp(parts[2]), parts[3], parts[4]
