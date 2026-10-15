"""Map ESPN team names onto football-data.co.uk team names."""
from __future__ import annotations

import difflib
import json
import re
import unicodedata

from ..config import ALIASES_FILE

# ESPN display name -> football-data.co.uk name, for names fuzzy matching gets wrong.
BUILTIN = {
    "Manchester United": "Man United", "Manchester City": "Man City",
    "Nottingham Forest": "Nott'm Forest", "Wolverhampton Wanderers": "Wolves",
    "Brighton & Hove Albion": "Brighton", "Tottenham Hotspur": "Tottenham",
    "Newcastle United": "Newcastle", "West Ham United": "West Ham",
    "Sheffield Wednesday": "Sheffield Weds", "Sheffield United": "Sheffield United",
    "Queens Park Rangers": "QPR", "West Bromwich Albion": "West Brom",
    "Atlético Madrid": "Ath Madrid", "Athletic Club": "Ath Bilbao",
    "Real Sociedad": "Sociedad", "Celta Vigo": "Celta", "Rayo Vallecano": "Vallecano",
    "Espanyol": "Espanol", "Deportivo Alavés": "Alaves", "Alavés": "Alaves",
    "Real Betis": "Betis", "Real Oviedo": "Oviedo", "Sporting Gijón": "Sp Gijon",
    "Deportivo La Coruña": "La Coruna", "Deportivo": "La Coruna", "Racing Santander": "Santander",
    "Bayern Munich": "Bayern Munich", "Borussia Dortmund": "Dortmund",
    "Borussia Mönchengladbach": "M'gladbach", "Bayer Leverkusen": "Leverkusen",
    "Eintracht Frankfurt": "Ein Frankfurt", "FC Cologne": "FC Koln", "1. FC Köln": "FC Koln",
    "VfB Stuttgart": "Stuttgart", "VfL Wolfsburg": "Wolfsburg", "VfL Bochum": "Bochum",
    "TSG Hoffenheim": "Hoffenheim", "Mainz": "Mainz", "FC Augsburg": "Augsburg",
    "1. FC Heidenheim 1846": "Heidenheim", "Hamburg SV": "Hamburg", "St. Pauli": "St Pauli",
    "Fortuna Düsseldorf": "Fortuna Dusseldorf", "SC Freiburg": "Freiburg",
    "Internazionale": "Inter", "AC Milan": "Milan", "Hellas Verona": "Verona",
    "AS Roma": "Roma", "Paris Saint-Germain": "Paris SG", "Saint-Étienne": "St Etienne",
    "Stade Rennais": "Rennes", "Olympique Lyonnais": "Lyon", "Olympique de Marseille": "Marseille",
    "AS Monaco": "Monaco", "Paris FC": "Paris FC", "Stade Brestois": "Brest",
    "PSV Eindhoven": "PSV Eindhoven", "AZ Alkmaar": "AZ Alkmaar", "Sparta Rotterdam": "Sparta Rotterdam",
    "Go Ahead Eagles": "Go Ahead Eagles", "FC Twente": "Twente", "FC Utrecht": "Utrecht",
    "FC Groningen": "Groningen", "Sporting CP": "Sp Lisbon", "SC Braga": "Sp Braga",
    "FC Porto": "Porto", "Vitória de Guimarães": "Guimaraes", "Union St.-Gilloise": "St. Gilloise",
    "Club Brugge": "Club Brugge", "Royal Antwerp": "Antwerp", "Celtic": "Celtic", "Rangers": "Rangers",
    "Heart of Midlothian": "Hearts", "Hibernian": "Hibernian", "Galatasaray": "Galatasaray",
    "Istanbul Basaksehir": "Buyuksehyr", "Goztepe": "Goztep",
    "Olympiacos": "Olympiakos", "PAOK Salonika": "PAOK",
    # MLS
    "LAFC": "Los Angeles FC", "Red Bull New York": "New York Red Bulls", "LA Galaxy": "Los Angeles Galaxy",
    # Denmark
    "AGF": "Aarhus", "F.C. København": "FC Copenhagen", "FC København": "FC Copenhagen",
    "FC Copenhagen": "FC Copenhagen", "Randers": "Randers FC",
    # Norway
    "Hamarkameratene": "HamKam", "HamKam": "HamKam", "Bodø/Glimt": "Bodo/Glimt", "FK Bodø/Glimt": "Bodo/Glimt",
    "Aalesunds FK": "Aalesund", "Lillestrøm": "Lillestrom", "Tromsø": "Tromso", "Tromsø IL": "Tromso",
    "Rosenborg BK": "Rosenborg", "Molde FK": "Molde", "Sandefjord Fotball": "Sandefjord",
    # Japan ("Tokyo Verdy" must not fuzzy-match "FC Tokyo")
    "Tokyo Verdy 1969": "Verdy", "Tokyo Verdy": "Verdy", "Urawa Red Diamonds": "Urawa Reds",
    "Yokohama F. Marinos": "Yokohama F. Marinos", "Yokohama F Marinos": "Yokohama F. Marinos",
    # Switzerland
    "BSC Young Boys": "Young Boys", "FC Basel": "Basel", "FC Zürich": "Zurich", "FC Lugano": "Lugano",
    "FC Luzern": "Luzern", "FC St. Gallen": "St. Gallen", "FC Sion": "Sion", "FC Thun": "Thun",
    "Grasshopper Club Zürich": "Grasshoppers", "Grasshopper": "Grasshoppers", "Servette FC": "Servette",
    "FC Lausanne-Sport": "Lausanne", "FC Vaduz": "Vaduz",
}

_STOP = {"fc", "cf", "afc", "sc", "ac", "as", "cd", "ud", "sd", "rc", "rcd", "ss", "us", "vfb", "vfl",
         "tsg", "sv", "fk", "sk", "club", "de", "the", "1.", "calcio", "1"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(t for t in s.split() if t not in _STOP)


def _user_aliases() -> dict:
    if ALIASES_FILE.exists():
        try:
            return json.loads(ALIASES_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def match_team(name: str, candidates: list[str]) -> str | None:
    """Best football-data candidate for an ESPN team name, or None."""
    if not candidates:
        return None
    for table in (_user_aliases(), BUILTIN):
        if name in table and table[name] in candidates:
            return table[name]
    if name in candidates:
        return name
    n = _norm(name)
    normed = {_norm(c): c for c in candidates}
    if n in normed:
        return normed[n]
    # token containment ("Leeds United" -> "Leeds")
    for nc, c in normed.items():
        if nc and (nc in n.split() or n.startswith(nc + " ") or nc.startswith(n + " ")):
            return c
    best = difflib.get_close_matches(n, list(normed), n=1, cutoff=0.6)
    return normed[best[0]] if best else None
