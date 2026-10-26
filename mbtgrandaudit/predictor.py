"""Fixture discovery, match analysis and ensemble prediction for one league."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from .config import LEAGUES, Settings, market_group
from .data import espn, footballdata as fd
from .data.names import match_team
from .markets import fair_odds, implied_from_odds, markets
from .models import MatchContext, build_models, ensemble, load_weights
from .features import extract as extract_features
from .implied import implied_markets, implied_matrix
from .models.form import recent, team_matches_from_history


@dataclass
class Prediction:
    fixture: espn.Fixture
    home: str                     # football-data name used by the models
    away: str
    ensemble: dict
    per_model: dict[str, dict]
    weights: dict[str, float]
    analysis: dict
    market: dict = field(default_factory=dict)
    picks: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    matrix: np.ndarray | None = None                  # ensemble scoreline probabilities
    model_only: dict = field(default_factory=dict)    # ensemble markets without the market view
    model_matrices: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)      # inputs frozen at prediction time
    variant: "Prediction | None" = None               # same fixture, market view fed Pinnacle's prices


_DIV_NAMES = {lg.fd_code: lg.name for lg in LEAGUES.values()} | {
    "E2": "English League One", "SP2": "Spanish Segunda", "D2": "German 2. Bundesliga",
    "I2": "Italian Serie B", "F2": "French Ligue 2", "SC1": "Scottish Championship"}


def _summary(f: pd.DataFrame) -> dict:
    if f.empty:
        return {"n": 0}
    gf, ga = f["gf"].to_numpy(float), f["ga"].to_numpy(float)
    res = np.where(gf > ga, "W", np.where(gf == ga, "D", "L"))
    tot = gf + ga
    return {
        "n": len(f),
        "w": int((res == "W").sum()), "d": int((res == "D").sum()), "l": int((res == "L").sum()),
        "seq": "".join(res[-6:]),
        "gf": float(gf.mean()), "ga": float(ga.mean()),
        "btts": float(((gf > 0) & (ga > 0)).mean()),
        "gg2": float(((gf > 1) & (ga > 1)).mean()),
        "over25": float((tot > 2.5).mean()),
        "clean_sheets": float((ga == 0).mean()),
        "games": [
            {"date": r.date.strftime("%Y-%m-%d"), "comp": _DIV_NAMES.get(str(r.competition), str(r.competition)), "venue": r.venue,
             "opponent": r.opponent, "score": f"{int(r.gf)}-{int(r.ga)}"}
            for r in f.tail(10).iloc[::-1].itertuples()
        ],
    }


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the same match coming from two sources (dates may differ by a day in UTC)."""
    if df.empty:
        return df
    df = df.sort_values("date").reset_index(drop=True)
    keep, last = [], None
    for r in df.itertuples():
        key = (r.venue, int(r.gf), int(r.ga))
        if last is not None and key == last[0] and (r.date - last[1]).days <= 2:
            continue
        keep.append(r.Index)
        last = (key, r.date)
    return df.loc[keep].reset_index(drop=True)


def _confidence(p: float, kind: str, spread: float) -> str:
    hi, mid = (0.60, 0.48) if kind == "1x2" else (0.70, 0.60)
    tier = 2 if p >= hi else 1 if p >= mid else 0
    if spread > 0.08:   # models disagree: trust the consensus less
        tier -= 1
    return "high" if tier >= 2 else "medium" if tier == 1 else "low"


_LABELS = {"1x2": "1X2", "btts": "BTTS", "gg2": "GG2+", "ou": "O/U"}


def _sel_name(key: str) -> str:
    """'gg2_yes' -> 'Yes', 'over_3.5' -> 'Over 3.5', 'home' -> '1'."""
    if key in ("home", "draw", "away"):
        return {"home": "1", "draw": "X", "away": "2"}[key]
    side, _, rest = key.partition("_")
    return rest.capitalize() if side in ("btts", "gg2") else f"{side.capitalize()} {rest}"


class LeaguePredictor:
    def __init__(self, league_key: str, settings: Settings | None = None,
                 as_of: datetime | None = None, offline: bool = False, log=print):
        self.league = LEAGUES[league_key]
        self.s = settings or Settings()
        self.now = as_of or datetime.now(timezone.utc)
        self.as_of = pd.Timestamp(self.now).tz_convert(None) if pd.Timestamp(self.now).tzinfo \
            else pd.Timestamp(self.now)
        self.offline = offline
        self.log = log
        self.prepared = False

    # ------------------------------------------------------------ preparation
    def prepare(self) -> "LeaguePredictor":
        lg, today = self.league, self.as_of.date()
        divs = [lg.fd_code] + ([lg.lower_fd_code] if lg.lower_fd_code else [])
        self.log(f"[{lg.key}] loading {self.s.seasons} seasons of results ({', '.join(divs)})")
        self.hist = fd.load_history(divs, self.s.seasons, today, self.offline, lg.calendar)
        self.h2h_hist = fd.load_history(divs, self.s.h2h_seasons, today, self.offline, lg.calendar)
        if self.hist.empty:
            raise RuntimeError(f"No historical data for {lg.name}")
        top = self.hist[(self.hist["div"] == lg.fd_code) & (self.hist["date"] < self.as_of)]
        recent_top = top[top["date"] >= self.as_of - pd.Timedelta(days=400)]
        recent_top = recent_top if len(recent_top) > 100 else top
        self.league_hg = float(recent_top["hg"].mean())
        self.league_ag = float(recent_top["ag"].mean())
        cur = fd.league_season(today, lg.calendar)
        cur_teams = top[top["season"] == cur]
        if len(cur_teams) < 10:
            cur_teams = top[top["season"] == top["season"].max()]
        self.candidates = sorted(set(cur_teams["home"]) | set(cur_teams["away"]))
        self.all_teams = sorted(set(self.hist["home"]) | set(self.hist["away"]))

        self.log(f"[{lg.key}] fitting {len(build_models(self.s))} models on {len(self.hist)} matches")
        self.models = [m.fit(self.hist, self.as_of, lg.fd_code) for m in build_models(self.s)]
        self.weights = load_weights(self.s, lg.key)
        self.prepared = True
        return self

    # --------------------------------------------------------------- fixtures
    def fixtures(self, days: int) -> list[espn.Fixture]:
        start = self.as_of.date()
        try:
            fx = espn.fixtures(self.league.espn_code, start, days, self.offline)
        except Exception as e:  # network trouble: fall back to football-data list
            self.log(f"[{self.league.key}] ESPN fixtures unavailable ({e}); using football-data")
            fx = []
        if not fx:
            df = fd.load_fixtures(self.league.fd_code, self.offline)
            end = pd.Timestamp(start) + pd.Timedelta(days=days)
            now = pd.Timestamp(self.now)
            for r in df[(df["date"] >= pd.Timestamp(start)) & (df["date"] < end)].itertuples():
                if pd.isna(r.kickoff) or r.kickoff <= now:
                    continue
                odds = {k: float(v) for k, v in (("H", r.odds_h), ("D", r.odds_d), ("A", r.odds_a),
                                                 ("over", r.odds_o25), ("under", r.odds_u25)) if pd.notna(v)}
                if "over" in odds:
                    odds["line"] = 2.5
                fx.append(espn.Fixture(fd.fd_event_id(self.league.fd_code, r.date, r.home, r.away),
                                       r.kickoff.to_pydatetime(), self.league.fd_code, r.home, r.away,
                                       odds=odds | {"source": "football-data avg"} if odds else {}))
        return [f for f in fx if f.status not in ("completed", "post", "in")]

    def resolve(self, name: str) -> str | None:
        return match_team(name, self.candidates) or match_team(name, self.all_teams)

    # ------------------------------------------------------------ team context
    def _espn_matches(self, team_id: str | None) -> pd.DataFrame:
        if not team_id or str(team_id).startswith("fd-"):
            return pd.DataFrame()
        cur = fd.league_season(self.as_of.date(), self.league.calendar)
        try:
            return espn.team_matches(team_id, self.league.espn_code, [cur - 1, cur], cur, self.offline)
        except Exception:
            return pd.DataFrame()

    def context(self, fx: espn.Fixture, home: str, away: str) -> tuple[MatchContext, dict]:
        with ThreadPoolExecutor(2) as ex:
            eh, ea = ex.map(self._espn_matches, [fx.home_espn_id, fx.away_espn_id])
        fdh = team_matches_from_history(self.hist, home)
        fda = team_matches_from_history(self.hist, away)
        # ESPN covers all competitions; football-data fills any gap (older league games)
        form_h = _dedupe(pd.concat([eh, fdh], ignore_index=True)) if not eh.empty else fdh
        form_a = _dedupe(pd.concat([ea, fda], ignore_index=True)) if not ea.empty else fda
        form_h = form_h[form_h["date"] < self.as_of]
        form_a = form_a[form_a["date"] < self.as_of]

        # head to head: football-data league meetings (10 seasons) + ESPN all competitions
        h2h_fd = team_matches_from_history(self.h2h_hist, home)
        h2h_fd = h2h_fd[h2h_fd["opponent"] == away]
        h2h_espn = eh[eh["opp_id"] == str(fx.away_espn_id)] if not eh.empty else pd.DataFrame()
        h2h = _dedupe(pd.concat([h2h_fd, h2h_espn], ignore_index=True)) if not h2h_espn.empty else h2h_fd
        h2h = h2h[h2h["date"] < self.as_of]

        ctx = MatchContext(home, away, self.as_of, form_h, form_a, h2h,
                           self.league_hg, self.league_ag)
        n = self.s.form_matches
        analysis = {
            "home_form": _summary(recent(form_h, self.as_of, n)),
            "away_form": _summary(recent(form_a, self.as_of, n)),
            "home_home_form": _summary(recent(form_h[form_h["venue"] == "home"], self.as_of, 6)),
            "away_away_form": _summary(recent(form_a[form_a["venue"] == "away"], self.as_of, 6)),
            "h2h": _summary(h2h.tail(12)),
            "all_comp_source": "ESPN (all competitions)" if not (eh.empty and ea.empty) else "league only",
        }
        return ctx, analysis

    # ------------------------------------------------------------- prediction
    def predict(self, fx: espn.Fixture, extra_odds: dict | None = None) -> Prediction:
        """extra_odds: {market key: decimal odds} for markets ESPN doesn't quote (e.g. gg2_yes)."""
        if not self.prepared:
            self.prepare()
        warnings = []
        home, away = self.resolve(fx.home), self.resolve(fx.away)
        if home is None or away is None:
            missing = fx.home if home is None else fx.away
            warnings.append(f"No history found for '{missing}' - using newcomer prior. "
                            f"Add it to team_aliases.json if it is a naming mismatch.")
            home, away = home or fx.home, away or fx.away
        ctx, analysis = self.context(fx, home, away)

        mats, per_model = {}, {}
        dc = next(m for m in self.models if m.name == "dixon_coles")
        ctx.prior = dc.expected_goals(ctx)
        for m in self.models:
            try:
                mat = m.predict(ctx, self.s.max_goals)
            except Exception as e:  # one bad model must not sink the prediction
                warnings.append(f"{m.name} failed: {e}")
                mat = None
            if mat is not None:
                mats[m.name] = mat
                per_model[m.name] = markets(mat, self.s.ou_lines)
        model_only = markets(ensemble(mats, self.weights), self.s.ou_lines)
        implied = implied_matrix(fx.odds, self.s.max_goals, extra=extra_odds) if fx.odds else None
        if implied is not None:
            mats["market_view"] = implied[0]
            per_model["market_view"] = markets(implied[0], self.s.ou_lines)
        ens_matrix = ensemble(mats, self.weights)
        ens = markets(ens_matrix, self.s.ou_lines)
        used = {k: self.weights.get(k, 0) for k in mats if self.weights.get(k, 0) > 0}
        tot = sum(used.values())
        used = {k: v / tot for k, v in used.items()}

        pred = Prediction(fx, home, away, ens, per_model, used, analysis, warnings=warnings,
                          matrix=ens_matrix, model_matrices=mats,
                          features=extract_features(self.models, ctx))
        pred.model_only = model_only
        pred.market = self._market(fx.odds, ens, extra_odds or {})
        pred.picks = self._picks(ens, per_model, pred.market, model_only)
        return pred

    def _market(self, odds: dict, ens: dict, extra: dict) -> dict:
        """Bookmaker prices, de-margined probabilities, the market-implied view of every
        market (solved from 1X2 + O/U, plus any entered BTTS/GG2+/O-U prices), and the
        model's edge wherever there is a price."""
        if not odds and not extra:
            return {}
        prices = {}   # market key -> (decimal odds, source)
        src = odds.get("source") or "bookmaker"
        for key, k in (("home", "H"), ("draw", "D"), ("away", "A")):
            if odds.get(k):
                prices[key] = (float(odds[k]), src)
        if odds.get("line") is not None:
            for side in ("over", "under"):
                if odds.get(side):
                    prices[f"{side}_{odds['line']}"] = (float(odds[side]), src)
        for key, o in extra.items():
            prices[key] = (float(o), "manual")
        edges = {k: ens[k] * o - 1 for k, (o, _) in prices.items() if k in ens}
        out = {"odds": odds, "implied": implied_from_odds(odds) if odds else {},
               "prices": prices, "edge": edges, "source": odds.get("source") if odds else "manual"}
        view = implied_markets(odds, self.s.ou_lines, extra) if odds else None
        if view:
            out["implied_info"] = view.pop("info")
            out["implied_model"] = view
        return out

    def _picks(self, ens: dict, per_model: dict, market: dict, model_only: dict) -> list[dict]:
        def spread(key):   # disagreement among the statistical models (not the market view)
            vals = [pm[key] for name, pm in per_model.items() if key in pm and name != "market_view"]
            return float(np.std(vals)) if len(vals) > 1 else 0.0

        view = market.get("implied_model", {})

        def pick(label, sel, key, kind, spread_key):
            p = ens[key]
            sp = spread(spread_key)
            edge = market.get("edge", {}).get(key)
            price = market.get("prices", {}).get(key)
            mp = view.get(key)
            enough = edge is not None and edge >= self.s.min_edge.get(market_group(key), 0.03)
            # models far from the market view: more likely missing information than value
            disputed = enough and mp is not None and abs(model_only.get(key, p) - mp) > self.s.max_disagreement
            return {"market": label, "pick": sel, "key": key, "prob": p, "fair": fair_odds(p),
                    "spread": sp, "confidence": _confidence(p, kind, sp), "market_prob": mp,
                    "price": price[0] if price else None, "price_source": price[1] if price else None,
                    "edge": edge, "value": enough and not disputed, "check": disputed}

        opts = {"1": "home", "X": "draw", "2": "away"}
        sel = max(opts, key=lambda k: ens[opts[k]])
        picks = [pick("1X2", sel, opts[sel], "1x2", opts[sel])]
        for label, key in (("BTTS", "btts"), ("GG2+", "gg2")):
            yes = ens[f"{key}_yes"] >= 0.5
            picks.append(pick(label, "Yes" if yes else "No", f"{key}_{'yes' if yes else 'no'}", "bin",
                              f"{key}_yes"))
        for line in self.s.ou_lines:
            over = ens[f"over_{line}"] >= 0.5
            picks.append(pick(f"O/U {line}", f"{'Over' if over else 'Under'} {line}",
                              f"{'over' if over else 'under'}_{line}", "bin", f"over_{line}"))
        # a priced selection the model likes but doesn't favour (e.g. GG2+ Yes at long odds)
        for key, (o, srcname) in market.get("prices", {}).items():
            if key in ens and not any(pk["key"] == key for pk in picks):
                pk = pick(_LABELS.get(market_group(key), key), _sel_name(key), key, "bin", key)
                if pk["value"] or pk["check"]:
                    pk["market"] += " (value)" if pk["value"] else " (check)"
                    picks.append(pk)
        return picks
