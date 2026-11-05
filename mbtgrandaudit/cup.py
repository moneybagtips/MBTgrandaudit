"""UEFA club competitions: one cross-league model for cup fixtures.

The league predictor fits each division on its own, with its own intercept, so a
team's rating only means something against its own league. Cup fixtures need one
scale. This module builds it from:

* football-data.co.uk results of every European domestic league in ``config.LEAGUES``
  (the domestic games rank teams within each league), and
* every Champions League, Europa League and Conference League result on ESPN
  (the European games place the leagues against each other).

Teams are keyed by their football-data name when they play in a covered league (the
ESPN team's ``defaultLeague`` says which, then the usual name matching), else by their
ESPN name with a group ``x:<ESPN league slug>`` such as ``x:aut.1``: those teams are
rated from their European results alone, shrunk toward the other clubs of their country.

Two rating models run on that data, both hierarchical (team = group level + team):

* ``cross_dc``: Dixon-Coles attack/defence, ridge-shrunk toward the team's league mean
  (fitted twice: the second fit's priors are the group means of the first);
* ``cross_elo``: margin-weighted Elo with a per-league offset that moves only on
  European results, mapped to goal rates by Poisson regression on European games.

Form and head-to-head come from ESPN all-competition schedules as in the leagues; the
market view comes from the fixture's odds when ESPN has them.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .config import EURO_SLUGS, LEAGUES, Settings
from .data import espn, footballdata as fd
from .data.names import match_team
from .markets import markets
from .models import MatchContext, Model, ensemble, load_weights
from .models.base import dc_tau_matrix, poisson_matrix, time_weights
from .models.form import FormModel, H2HModel, team_matches_from_history
from .models.ratings import GoalMapper
from .models.strength import NEWCOMER_PRIOR, DixonColes
from .predictor import LeaguePredictor, Prediction

EURO_DIV = "EURO"
UNKNOWN_GROUP = "x:?"
# domestic leagues whose clubs never enter UEFA competitions: no use for the cup scale
NON_UEFA = {"usa.1", "jpn.1"}
# ridge centre for a club from an uncovered league before its country has a mean: the
# leagues we cover are Europe's stronger ones, so start a little below their average
UNCOVERED_PRIOR = (-0.10, 0.10)
COMP_NAMES = {"uefa.champions": "Champions League", "uefa.europa": "Europa League",
              "uefa.europa.conf": "Conference League"}


def covered_slugs() -> dict[str, str]:
    """ESPN league slug -> football-data division, for the domestic leagues on the cup scale."""
    return {lg.espn_code: lg.fd_code for lg in LEAGUES.values()
            if not lg.cup and lg.fd_code and lg.espn_code not in NON_UEFA}


def country(group: str) -> str:
    """'x:aut.1' -> 'aut', 'E0' -> the league name."""
    if group.startswith("x:"):
        return group[2:].split(".")[0]
    return next((lg.name for lg in LEAGUES.values() if lg.fd_code == group), group)


# ---------------------------------------------------------------------- data
@dataclass
class CrossData:
    """Domestic + European results on one set of team names."""
    hist: pd.DataFrame                     # date div home away hg ag season comp neutral
    groups: dict[str, str]                 # team -> fd division or 'x:<slug>'
    id_name: dict[str, str]                # ESPN team id -> team name in ``hist``
    candidates: dict[str, list[str]]       # fd division -> its teams in the history
    unmatched: list[str] = field(default_factory=list)   # covered-league clubs whose name didn't match
    offline: bool = False

    @property
    def euro(self) -> pd.DataFrame:
        return self.hist[self.hist["div"] == EURO_DIV]

    def resolve(self, espn_id: str | None, name: str) -> str:
        """Model name for an ESPN team, adding it (and its group) if it is new."""
        if espn_id and str(espn_id) in self.id_name:
            return self.id_name[str(espn_id)]
        league = espn.team_info(espn_id, self.offline)[0] if espn_id else None
        team, group = _map_team(name, league, self.candidates, self.groups)
        if espn_id:
            self.id_name[str(espn_id)] = team
        self.groups.setdefault(team, group)
        return team


def _country_divs(code: str | None) -> list[str]:
    """Covered divisions of a country ('eng' -> ['E0', 'E1'])."""
    return [d for slug, d in covered_slugs().items() if code and slug.split(".")[0] == code]


def _strict(name: str, cands: list[str]) -> str | None:
    """Alias or exact (normalised) name match only: safe to run across every league."""
    from .data.names import BUILTIN, _norm, _user_aliases
    for table in (_user_aliases(), BUILTIN):
        if table.get(name) in cands:
            return table[name]
    n = _norm(name)
    return next((c for c in cands if c == name or _norm(c) == n), None)


def _locate(name: str, league: str | None, candidates: dict) -> tuple[str | None, str]:
    """(football-data name or None, group) for an ESPN team.

    The team's domestic league slug decides when it is one we cover (fuzzy name matching
    within that league, as the league predictor does). Anywhere else only an alias or
    exact name counts: ESPN files many clubs under the UEFA competition instead of a
    league, and a relegated or promoted club (e.g. 'esp.2') sits outside its country's
    covered division. An unmatched club from a covered country joins that country's top
    division as a newcomer; other clubs get the group 'x:<country>' when ESPN names their
    league, else 'x:?'."""
    div = covered_slugs().get(league or "")
    domestic = bool(league) and not league.startswith("uefa")
    country_code = league.split(".")[0] if domestic else None
    if div:
        hit = match_team(name, candidates.get(div, []))
        if hit:
            return hit, div
    for d, cands in candidates.items():
        hit = _strict(name, cands)
        if hit:
            return hit, d
    home_divs = _country_divs(country_code)
    if div or home_divs:
        return None, div or home_divs[0]       # covered country, no domestic record matched
    return None, f"x:{country_code}" if country_code else UNKNOWN_GROUP


def _map_team(name: str, league: str | None, candidates: dict, groups: dict) -> tuple[str, str]:
    hit, group = _locate(name, league, candidates)
    if hit:
        return hit, group
    # uncovered (or unmatched) club: its ESPN name, unless that clashes with another team
    if name in groups and groups[name] != group:
        return f"{name} ({group})", group
    return name, group


def load_cross_data(today: date, seasons: int, offline: bool = False, log=print) -> CrossData:
    """``seasons`` seasons of domestic and European results up to ``today``."""
    frames = []
    for lg in LEAGUES.values():
        if lg.cup or not lg.fd_code or lg.espn_code in NON_UEFA:
            continue
        h = fd.load_history([lg.fd_code], seasons, today, offline, lg.calendar)
        if len(h):
            frames.append(h[["date", "div", "home", "away", "hg", "ag", "season"]])
    dom = pd.concat(frames, ignore_index=True).dropna(subset=["hg", "ag"])
    candidates = {d: sorted(set(g["home"]) | set(g["away"])) for d, g in dom.groupby("div")}

    start = date(fd.season_start_year(today) - seasons + 1, 7, 1)
    log(f"[cups] loading European results since {start:%b %Y} ({', '.join(EURO_SLUGS)})")
    euro_fx = [f for slug in EURO_SLUGS for f in espn.month_results(slug, start, today, today, offline)
               if f.status == "completed" and f.home_score is not None and f.away_score is not None]
    ids = {}
    for f in sorted(euro_fx, key=lambda f: f.kickoff):          # latest display name wins
        ids[str(f.home_espn_id)], ids[str(f.away_espn_id)] = f.home, f.away
    with ThreadPoolExecutor(8) as ex:
        info = dict(zip(ids, ex.map(lambda i: espn.team_info(i, offline), ids)))

    groups = {}
    for d, teams in candidates.items():          # a club in two tiers keeps its top-tier group
        for t in teams:
            groups.setdefault(t, d)
    id_name, unmatched = {}, []
    for tid, name in ids.items():
        team, group = _map_team(name, info.get(tid, (None, None))[0], candidates, groups)
        if not group.startswith("x:") and team not in candidates.get(group, []):
            unmatched.append(f"{name} ({group})")
        id_name[tid] = team
        groups.setdefault(team, group)
    rows = [{"date": pd.Timestamp(f.kickoff).tz_convert(None).normalize(), "div": EURO_DIV,
             "home": id_name[str(f.home_espn_id)], "away": id_name[str(f.away_espn_id)],
             "hg": float(f.home_score), "ag": float(f.away_score),
             "season": fd.season_start_year(f.kickoff.date()), "comp": f.league, "neutral": f.neutral,
             "event_id": f.event_id}
            for f in euro_fx]
    euro = pd.DataFrame(rows)
    dom = dom.assign(comp=dom["div"], neutral=False, event_id=None)
    hist = pd.concat([dom, euro], ignore_index=True).sort_values("date").reset_index(drop=True)
    if unmatched:
        log(f"[cups] {len(unmatched)} clubs from covered countries without a football-data record "
            f"(promoted clubs rate as newcomers to the league; add real name mismatches to "
            f"team_aliases.json): {', '.join(unmatched[:12])}")
    log(f"[cups] {len(dom)} domestic + {len(euro)} European matches, "
        f"{sum(g.startswith('x:') for g in groups.values())} clubs from uncovered leagues")
    return CrossData(hist, groups, id_name, candidates, unmatched, offline)


_DATA_CACHE: dict = {}


def cross_data(today: date, seasons: int, offline: bool = False, log=print) -> CrossData:
    """load_cross_data, shared by the UCL and UEL predictors of one run."""
    key = (today, seasons, offline)
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = load_cross_data(today, seasons, offline, log)
    return _DATA_CACHE[key]


# -------------------------------------------------------------------- models
def _group_means(values: dict[str, float], groups: dict[str, str], pseudo: float) -> dict[str, float]:
    """Mean value per group, shrunk toward 0 with ``pseudo`` phantom teams."""
    acc = defaultdict(list)
    for t, v in values.items():
        acc[groups.get(t, UNKNOWN_GROUP)].append(v)
    return {g: float(np.sum(v) / (len(v) + pseudo)) for g, v in acc.items()}


class CrossDixonColes(DixonColes):
    """Dixon-Coles over domestic + European games, team ratings shrunk toward their
    league's mean rather than toward zero (so a league's level is set by its European
    results, not pulled to the average of all leagues)."""
    name = "cross_dc"
    description = ("Cross-league Dixon-Coles: every European league plus UEFA results, "
                   "teams shrunk toward their league's level")

    def __init__(self, groups: dict[str, str], xi: float = 0.0019, l2: float = 2.0, passes: int = 2):
        super().__init__(xi, l2)
        self.groups, self.passes = groups, passes
        self.group_att: dict[str, float] = {}
        self.group_def: dict[str, float] = {}

    def _priors(self, h, prior_att, prior_def):
        pa, pd_ = prior_att.copy(), prior_def.copy()
        for t, i in self.tidx.items():
            g = self.groups.get(t, UNKNOWN_GROUP)
            if g in self.group_att:
                ga, gd = self.group_att[g], self.group_def[g]
            elif g.startswith("x:"):
                ga, gd = self.group_att.get("x:*", UNCOVERED_PRIOR[0]), self.group_def.get("x:*", UNCOVERED_PRIOR[1])
            else:
                ga, gd = 0.0, 0.0
            # newcomers keep their offset below their group's level
            pa[i] += ga
            pd_[i] += gd
        return pa, pd_

    def fit(self, hist, as_of, target_div=EURO_DIV):
        self.group_att, self.group_def = {}, {}
        for i in range(self.passes):
            super().fit(hist, as_of, target_div)
            if not self.fitted or i == self.passes - 1:
                break
            self.group_att = _group_means(self.att, self.groups, 1.0)
            self.group_def = _group_means(self.dfn, self.groups, 1.0)
            unc = [t for t in self.att if self.groups.get(t, UNKNOWN_GROUP).startswith("x:")]
            if unc:
                self.group_att["x:*"] = float(np.mean([self.att[t] for t in unc]))
                self.group_def["x:*"] = float(np.mean([self.dfn[t] for t in unc]))
        return self

    def _team(self, t):
        if t in self.att:
            return self.att[t], self.dfn[t]
        g = self.groups.get(t, UNKNOWN_GROUP)
        ga = self.group_att.get(g, self.group_att.get("x:*", UNCOVERED_PRIOR[0]) if g.startswith("x:") else 0.0)
        gd = self.group_def.get(g, self.group_def.get("x:*", UNCOVERED_PRIOR[1]) if g.startswith("x:") else 0.0)
        return ga + NEWCOMER_PRIOR[0], gd + NEWCOMER_PRIOR[1]

    def expected_goals(self, ctx: MatchContext):
        xg = super().expected_goals(ctx)
        if xg is None or not ctx.neutral:
            return xg
        s = float(np.exp(self.home_adv / 2))          # neutral venue: split the home edge
        return xg[0] / s, xg[1] * s

    def strength(self, t) -> float:
        a, d = self._team(t)
        return float(a - d)


class CrossElo(Model):
    """Elo on one scale for every club: rating = league offset + team rating.

    Domestic games move only the team ratings (the offset cancels); European games also
    move the two leagues' offsets, so a league's level follows its clubs' European results.
    The pre-match gap of every European game is mapped to goal rates by Poisson regression.
    """
    name = "cross_elo"
    description = "Cross-league Elo: team rating plus a league offset learned from UEFA results"
    K, K_OFF, HFA = 20.0, 4.0, 65.0
    UNCOVERED_START = -80.0         # first offset of an uncovered league (Elo points)
    rho = -0.06

    def __init__(self, groups: dict[str, str], xi: float = 0.0019):
        self.groups, self.xi = groups, xi
        self.fitted = False

    def _group(self, t):
        return self.groups.get(t, UNKNOWN_GROUP)

    def _new(self, t):
        g = self._group(t)
        if g not in self.off:
            self.off[g] = self.UNCOVERED_START if g.startswith("x:") else 0.0
        mates = [v for u, v in self.rel.items() if self._group(u) == g]
        # newcomers to a league start in its bottom fifth; a club from an uncovered country
        # with no European record starts at its country's level
        self.rel[t] = float(np.percentile(mates, 20)) if len(mates) >= 5 else 0.0

    def rating(self, t) -> float:
        if t not in self.rel:
            self._new(t)
        return self.off[self._group(t)] + self.rel[t]

    def fit(self, hist, as_of, target_div=EURO_DIV):
        h = hist[hist["date"] < as_of].dropna(subset=["hg", "ag"]).sort_values("date", kind="stable")
        self.off, self.rel = {}, {}
        feats, rows = [], []
        neutral = h["neutral"].to_numpy(bool) if "neutral" in h else np.zeros(len(h), bool)
        for r, neu in zip(h.itertuples(index=False), neutral):
            rh, ra = self.rating(r.home), self.rating(r.away)
            d = rh + (0.0 if neu else self.HFA) - ra
            euro = r.div == target_div
            if euro:
                feats.append(d / 400.0)
                rows.append((r.date, r.hg, r.ag))
            exp_h = 1 / (1 + 10 ** (-d / 400))
            res = 1.0 if r.hg > r.ag else 0.5 if r.hg == r.ag else 0.0
            gd = abs(r.hg - r.ag)
            g = 1 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
            delta = self.K * g * (res - exp_h)
            self.rel[r.home] += delta
            self.rel[r.away] -= delta
            gh, ga = self._group(r.home), self._group(r.away)
            if euro and gh != ga:
                self.off[gh] += self.K_OFF * g * (res - exp_h)
                self.off[ga] -= self.K_OFF * g * (res - exp_h)
        if len(rows) < 100:
            return self
        sub = pd.DataFrame(rows, columns=["date", "hg", "ag"])
        keep = (sub["date"] >= sub["date"].min() + pd.Timedelta(days=330)).to_numpy()   # burn-in
        if keep.sum() < 100:
            keep = np.ones(len(sub), dtype=bool)
        x = np.array(feats)[keep]
        sub = sub[keep]
        w = time_weights(sub["date"], as_of, self.xi)
        self.mapper = GoalMapper().fit(x, sub["hg"].to_numpy(float), sub["ag"].to_numpy(float), w)
        self.fitted = True
        return self

    def expected_goals(self, ctx: MatchContext):
        if not self.fitted:
            return None
        d = self.rating(ctx.home) + (0.0 if ctx.neutral else self.HFA) - self.rating(ctx.away)
        return self.mapper(d / 400.0)

    def predict(self, ctx, max_goals=10):
        xg = self.expected_goals(ctx)
        if xg is None:
            return None
        return dc_tau_matrix(poisson_matrix(*xg, max_goals), *xg, self.rho)

    def league_offsets(self) -> pd.Series:
        return pd.Series(self.off).sort_values(ascending=False)


def build_cup_models(settings: Settings, groups: dict) -> list[Model]:
    return [CrossDixonColes(groups, settings.xi), CrossElo(groups, settings.xi),
            FormModel(settings.form_matches), H2HModel()]


def cup_seasons(settings: Settings) -> int:
    """Seasons of data for the cup scale: the leagues' window plus one for the Elo burn-in."""
    return settings.seasons + 1


# ----------------------------------------------------------------- predictor
class CupPredictor(LeaguePredictor):
    """LeaguePredictor for a UEFA competition, on the cross-league models."""
    prior_model = "cross_dc"

    def prepare(self) -> "CupPredictor":
        lg, today = self.league, self.as_of.date()
        self.data = cross_data(today, cup_seasons(self.s), self.offline, self.log)
        hist = self.data.hist[self.data.hist["date"] < self.as_of]
        self.hist = self.h2h_hist = hist
        euro = self.data.euro
        euro = euro[(euro["date"] < self.as_of) & (euro["date"] >= self.as_of - pd.Timedelta(days=730))]
        if len(euro) < 100:
            raise RuntimeError("Not enough European results to fit the cup models")
        self.league_hg, self.league_ag = float(euro["hg"].mean()), float(euro["ag"].mean())
        comp = euro[euro["comp"] == lg.espn_code]
        cur = comp[comp["season"] == comp["season"].max()] if len(comp) else comp
        self.candidates = sorted(set(cur["home"]) | set(cur["away"]))
        self.all_teams = sorted(set(hist["home"]) | set(hist["away"]))
        self.log(f"[{lg.key}] fitting cross-league models on {len(hist)} matches")
        self.models = [m.fit(hist, self.as_of, EURO_DIV) for m in build_cup_models(self.s, self.data.groups)]
        self.weights = load_weights(self.s, lg.key, default=self.s.cup_weights)
        self.prepared = True
        return self

    def fixtures(self, days: int) -> list[espn.Fixture]:
        try:
            fx = espn.fixtures(self.league.espn_code, self.as_of.date(), days, self.offline)
        except Exception as e:
            self.log(f"[{self.league.key}] ESPN fixtures unavailable ({e})")
            fx = []
        return [f for f in fx if f.status not in ("completed", "post", "in")]

    def _euro_games(self, team: str) -> int:
        e = self.data.euro
        e = e[e["date"] < self.as_of]
        return int(((e["home"] == team) | (e["away"] == team)).sum())

    def _teams(self, fx):
        warnings = []
        names = []
        for side_name, espn_id in ((fx.home, fx.home_espn_id), (fx.away, fx.away_espn_id)):
            team = self.data.resolve(espn_id, side_name) if espn_id else (self.resolve(side_name) or side_name)
            self.data.groups.setdefault(team, UNKNOWN_GROUP)
            names.append(team)
            g = self.data.groups[team]
            n_euro = self._euro_games(team)
            dom = (self.hist["div"] != EURO_DIV) & ((self.hist["home"] == team) | (self.hist["away"] == team))
            if g.startswith("x:"):
                known = g != UNKNOWN_GROUP
                where = (f"plays in a league this tool doesn't model (country code '{g[2:]}')" if known
                         else "plays in a league this tool doesn't model (ESPN gives no country)")
                pool = "its country's other clubs" if known else "the other clubs from unmodelled leagues"
                basis = (f"rated from its {n_euro} European games only, shrunk toward {pool}" if n_euro
                         else f"no European games in the data either, so it gets the average of {pool}")
                warnings.append(f"{side_name} {where}: {basis}. Treat this prediction with extra caution.")
            elif not dom.any():
                warnings.append(f"{side_name}: no {country(g)} results matched (a promoted club, or a name "
                                f"to add to team_aliases.json); rated as a newcomer to that league"
                                + (f" plus its {n_euro} European games." if n_euro else "."))
        return names[0], names[1], warnings

    def _features(self, ctx):
        dc = next(m for m in self.models if m.name == "cross_dc")
        elo = next(m for m in self.models if m.name == "cross_elo")
        feats = super()._features(ctx)
        for side, t in (("home", ctx.home), ("away", ctx.away)):
            if dc.fitted:
                feats[f"{side}_xdc_att"], feats[f"{side}_xdc_def"] = map(float, dc._team(t))
            if elo.fitted:
                feats[f"{side}_xelo"] = float(elo.rating(t))
                feats[f"{side}_xelo_league"] = float(elo.off.get(self.data.groups.get(t, UNKNOWN_GROUP), 0.0))
        for name in ("cross_dc", "cross_elo"):
            m = dc if name == "cross_dc" else elo
            xg = m.expected_goals(ctx) if m.fitted else None
            if xg:
                feats[f"{name}_xg_home"], feats[f"{name}_xg_away"] = map(float, xg)
        return feats

    def predict(self, fx, extra_odds=None) -> Prediction:
        pred = super().predict(fx, extra_odds)
        pred.analysis["groups"] = {"home": self.data.groups.get(pred.home), "away": self.data.groups.get(pred.away)}
        return pred

    def ratings_table(self) -> pd.DataFrame:
        if not self.prepared:
            self.prepare()
        dc = next(m for m in self.models if m.name == "cross_dc")
        elo = next(m for m in self.models if m.name == "cross_elo")
        teams = self.candidates or self.all_teams
        df = pd.DataFrame({
            "league": [country(self.data.groups.get(t, UNKNOWN_GROUP)) for t in teams],
            "elo": [elo.rating(t) for t in teams],
            "league_offset": [elo.off.get(self.data.groups.get(t, UNKNOWN_GROUP), 0.0) for t in teams],
            "attack": [dc._team(t)[0] for t in teams],
            "defence": [dc._team(t)[1] for t in teams],
        }, index=teams)
        df["dc_strength"] = df["attack"] - df["defence"]
        return df.sort_values("elo", ascending=False)


# ------------------------------------------------------------------ backtest
CUP_MARKET_KEYS = ["1x2", "btts", "gg2", "o1.5", "o2.5", "o3.5"]


def backtest(keys: list[str], test_seasons: int = 2, settings: Settings | None = None,
             today: date | None = None, offline: bool = False, log=print) -> dict:
    """Walk-forward test on completed European seasons of the given competitions: refit
    weekly on everything before the week, predict its games, score every component.

    ESPN keeps no odds for past European games, so there is no market view here: this
    measures the statistical components and tunes their share of the non-market weight."""
    from .backtest import _outcome_probs, _rps

    s = settings or Settings()
    today = today or date.today()
    cur = fd.season_start_year(today)
    data = load_cross_data(today, cup_seasons(s) + test_seasons, offline, log)
    slugs = {LEAGUES[k].espn_code: k for k in keys}
    euro = data.euro
    test = euro[euro["comp"].isin(slugs) & (euro["season"] >= cur - test_seasons) & (euro["season"] < cur)]
    if test.empty:
        raise RuntimeError("no completed European seasons to test on")
    log(f"[cups] backtesting {len(test)} {'/'.join(keys)} matches, seasons "
        f"{fd.season_code(cur - test_seasons)}..{fd.season_code(cur - 1)}, weekly refits")
    weights = load_weights(s, keys[0], default=s.cup_weights)
    names = ["cross_dc", "cross_elo", "form", "h2h", "ensemble"]
    rows = []
    for i, (_, wk) in enumerate(test.groupby(test["date"].dt.to_period("W-MON"))):
        as_of = wk["date"].min().normalize()
        first = fd.season_start_year(as_of.date()) - cup_seasons(s) + 1
        train = data.hist[(data.hist["date"] < as_of) & (data.hist["season"] >= first)]
        models = [m.fit(train, as_of, EURO_DIV) for m in build_cup_models(s, data.groups)]
        e = train[(train["div"] == EURO_DIV) & (train["date"] >= as_of - pd.Timedelta(days=730))]
        lhg, lag = float(e["hg"].mean()), float(e["ag"].mean())
        for r in wk.itertuples():
            fh, fa = team_matches_from_history(train, r.home), team_matches_from_history(train, r.away)
            ctx = MatchContext(r.home, r.away, as_of, fh, fa, fh[fh["opponent"] == r.away], lhg, lag,
                               neutral=bool(r.neutral))
            ctx.prior = models[0].expected_goals(ctx)
            mats = {m.name: m.predict(ctx, s.max_goals) for m in models}
            if mats.get("cross_dc") is None:
                continue
            mats["ensemble"] = ensemble(mats, weights)
            hg, ag = int(r.hg), int(r.ag)
            rec = {"date": r.date, "comp": slugs[r.comp], "home": r.home, "away": r.away, "hg": hg, "ag": ag,
                   "groups": (data.groups.get(r.home), data.groups.get(r.away))}
            for n in names:
                mat = mats.get(n)
                mk = markets(mat if mat is not None else mats["cross_dc"], s.ou_lines)
                rec[n] = {**_outcome_probs(mk, hg, ag), "rps": _rps(mk, hg, ag),
                          "hit": max(("home", "draw", "away"), key=mk.get) ==
                          ("home" if hg > ag else "draw" if hg == ag else "away"),
                          "available": mat is not None}
            rows.append(rec)
        if i % 5 == 0:
            log(f"  week of {as_of.date()}: {len(rows)} matches scored")
    # naive benchmark: the training-window base rates of each outcome, for scale
    base = _base_rates(data.euro[data.euro["season"] < cur - test_seasons])
    for rec in rows:
        rec["base_rates"] = {**_outcome_probs(base, rec["hg"], rec["ag"]), "rps": _rps(base, rec["hg"], rec["ag"]),
                             "hit": rec["hg"] > rec["ag"], "available": True}
    return {"league": "/".join(keys), "rows": rows, "models": names + ["base_rates"]}


def _base_rates(euro: pd.DataFrame) -> dict:
    hg, ag = euro["hg"].to_numpy(), euro["ag"].to_numpy()
    tot = hg + ag
    out = {"home": float(np.mean(hg > ag)), "draw": float(np.mean(hg == ag)), "away": float(np.mean(hg < ag)),
           "btts_yes": float(np.mean((hg > 0) & (ag > 0))), "gg2_yes": float(np.mean((hg > 1) & (ag > 1)))}
    out["btts_no"], out["gg2_no"] = 1 - out["btts_yes"], 1 - out["gg2_yes"]
    for line in (0.5, 1.5, 2.5, 3.5, 4.5):
        out[f"over_{line}"] = float(np.mean(tot > line))
        out[f"under_{line}"] = 1 - out[f"over_{line}"]
    return out


def optimise_weights(result: dict, settings: Settings | None = None) -> dict:
    """Cup weights from a backtest: tune the statistical components (as the league tuner
    does), then give the market view its default share, since the backtest has no odds."""
    from .backtest import optimise_weights as _opt

    s = settings or Settings()
    res = {**result, "models": [n for n in result["models"] if n not in ("ensemble", "base_rates")]}
    prior = Settings(default_weights={k: v for k, v in s.cup_weights.items() if k != "market_view"})
    w = _opt(res, prior)
    mv = s.cup_weights.get("market_view", 0.0)
    share = mv / sum(s.cup_weights.values())
    out = {k: round(v * (1 - share), 4) for k, v in w.items()}
    if mv:
        out["market_view"] = round(share, 4)
    return out


def by_group(result: dict) -> pd.DataFrame:
    """1X2 log loss of the ensemble split by whether both clubs are from covered leagues."""
    recs = []
    for r in result["rows"]:
        unc = sum(str(g or UNKNOWN_GROUP).startswith("x:") for g in r["groups"])
        recs.append({"uncovered_clubs": unc, "logloss_1x2": -np.log(max(r["ensemble"]["1x2"], 1e-9)),
                     "base_logloss_1x2": -np.log(max(r["base_rates"]["1x2"], 1e-9)),
                     "acc_1x2": r["ensemble"]["hit"]})
    df = pd.DataFrame(recs)
    return df.groupby("uncovered_clubs").agg(matches=("acc_1x2", "size"), logloss_1x2=("logloss_1x2", "mean"),
                                             base_logloss_1x2=("base_logloss_1x2", "mean"),
                                             acc_1x2=("acc_1x2", "mean"))
