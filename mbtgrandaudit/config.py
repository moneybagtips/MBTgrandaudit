"""Static configuration: supported leagues and model defaults."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".cache"
ALIASES_FILE = ROOT / "team_aliases.json"
WEIGHTS_FILE = ROOT / "ensemble_weights.json"


@dataclass(frozen=True)
class League:
    key: str                 # short CLI key, e.g. "EPL"
    name: str
    fd_code: str             # football-data.co.uk division code
    espn_code: str           # ESPN league slug
    lower_fd_code: str | None = None  # second tier, used to rate promoted teams
    calendar: bool = False   # season runs within a calendar year (MLS, Nordic leagues)


LEAGUES: dict[str, League] = {
    lg.key: lg
    for lg in [
        League("EPL", "English Premier League", "E0", "eng.1", "E1"),
        League("CHAMP", "English Championship", "E1", "eng.2", "E2"),
        League("LALIGA", "Spanish La Liga", "SP1", "esp.1", "SP2"),
        League("BUNDES", "German Bundesliga", "D1", "ger.1", "D2"),
        League("SERIEA", "Italian Serie A", "I1", "ita.1", "I2"),
        League("LIGUE1", "French Ligue 1", "F1", "fra.1", "F2"),
        League("ERED", "Dutch Eredivisie", "N1", "ned.1"),
        League("PRIMEIRA", "Portuguese Primeira Liga", "P1", "por.1"),
        League("JUPILER", "Belgian Pro League", "B1", "bel.1"),
        League("SUPERLIG", "Turkish Super Lig", "T1", "tur.1"),
        League("SPL", "Scottish Premiership", "SC0", "sco.1", "SC1"),
        League("GREECE", "Greek Super League", "G1", "gre.1"),
        # football-data.co.uk "extra" leagues: one file per country, closing odds only
        League("SWISS", "Swiss Super League", "SWZ", "sui.1"),
        League("MLS", "Major League Soccer", "USA", "usa.1", calendar=True),
        League("DENMARK", "Danish Superliga", "DNK", "den.1"),
        League("NORWAY", "Norwegian Eliteserien", "NOR", "nor.1", calendar=True),
        # J1 moved to an August-May season from 2026-27 (earlier seasons were calendar years)
        League("JLEAGUE", "Japanese J1 League", "JPN", "jpn.1"),
    ]
}

MAJOR = ["EPL", "LALIGA", "BUNDES", "SERIEA", "LIGUE1"]


@dataclass
class Settings:
    seasons: int = 3                 # seasons of league data used to fit models
    h2h_seasons: int = 10            # seasons scanned for league head-to-head history
    xi: float = 0.0019               # time-decay per day (half-life ~ 1 year)
    max_goals: int = 10              # score matrix is (max_goals+1)^2
    form_matches: int = 10           # all-competition matches used for form
    ou_lines: tuple = (0.5, 1.5, 2.5, 3.5, 4.5)
    # minimum model edge (prob * odds - 1) before a price counts as value, per market
    # group: niche markets carry bigger margins and far more variance (see README)
    min_edge: dict = field(default_factory=lambda: {
        "1x2": 0.03, "ou": 0.03, "btts": 0.05, "gg2": 0.10,
    })
    # if the model is further than this from the market-implied probability, the
    # "edge" is more likely missing information (injuries, line-ups) than value:
    # the pick is flagged "check" instead of VALUE
    max_disagreement: float = 0.10
    default_weights: dict = field(default_factory=lambda: {
        "dixon_coles": 0.22,
        "poisson": 0.10,
        "bivariate_poisson": 0.12,
        "negative_binomial": 0.10,
        "xg_poisson": 0.14,
        "elo": 0.12,
        "pi_rating": 0.08,
        "form": 0.07,
        "h2h": 0.05,
        # market view (solved from the game's 1X2 + O/U odds). Used only when the game
        # has odds; 1.2 against the models' 1.0 = 55% of the blend. Tuned by backtest.
        "market_view": 1.2,
    })


def market_group(market: str) -> str:
    """'home'/'draw'/'away' -> 1x2, 'over_2.5' -> ou, 'btts_yes' -> btts, 'gg2_no' -> gg2."""
    if market in ("home", "draw", "away"):
        return "1x2"
    if market.startswith(("over_", "under_")):
        return "ou"
    return market.split("_")[0]
