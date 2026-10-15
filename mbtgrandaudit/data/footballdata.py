"""Historical league results from football-data.co.uk (results, xG, odds)."""
from __future__ import annotations

import io
from datetime import date

import numpy as np
import pandas as pd

from .cache import fetch

BASE = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
FIXTURES = "https://www.football-data.co.uk/fixtures.csv"

COLUMNS = ["date", "div", "home", "away", "hg", "ag", "hxg", "axg", "hs", "as", "hst", "ast",
           "odds_h", "odds_d", "odds_a", "odds_o25", "odds_u25",
           "odds_ch", "odds_cd", "odds_ca", "odds_co25", "odds_cu25"]


def season_start_year(d: date) -> int:
    return d.year if d.month >= 7 else d.year - 1


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


def load_history(divs: list[str], seasons: int, today: date, offline: bool = False) -> pd.DataFrame:
    """Results for ``divs`` over the current and previous ``seasons-1`` seasons."""
    start = season_start_year(today)
    frames = [load_season(div, y, today, offline)
              for div in divs for y in range(start - seasons + 1, start + 1)]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=COLUMNS + ["season"])
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("date").reset_index(drop=True)


def load_fixtures(div: str, offline: bool = False) -> pd.DataFrame:
    """Upcoming fixtures published by football-data.co.uk (fallback source)."""
    text = fetch(FIXTURES, ttl=3 * 3600, offline=offline)
    if not text or "HomeTeam" not in text[:2000]:
        return pd.DataFrame(columns=COLUMNS)
    raw = pd.read_csv(io.StringIO(text), on_bad_lines="skip", low_memory=False)
    if raw.empty:
        return pd.DataFrame(columns=COLUMNS)
    df = _normalise(raw)
    return df[df["div"] == div].reset_index(drop=True)
