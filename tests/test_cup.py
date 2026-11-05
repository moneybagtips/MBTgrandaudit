"""UEFA cups: config, team mapping, the cross-league models and the cup predictor."""
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from mbtgrandaudit import cli, cup
from mbtgrandaudit.config import CUPS, DOMESTIC, LEAGUES, Settings
from mbtgrandaudit.data import espn
from mbtgrandaudit.models import MatchContext, load_weights


def test_cups_configured_and_in_all():
    assert CUPS == ["UCL", "UEL"]
    assert LEAGUES["UCL"].espn_code == "uefa.champions" and LEAGUES["UEL"].espn_code == "uefa.europa"
    assert all(LEAGUES[k].cup and not LEAGUES[k].fd_code for k in CUPS)
    assert not set(CUPS) & set(DOMESTIC)
    assert set(CUPS) <= set(cli._leagues(["all"]))          # the daily Predict run includes them
    assert isinstance(cli._predictor("UCL"), cup.CupPredictor)
    assert not isinstance(cli._predictor("EPL"), cup.CupPredictor)
    # the cup scale leaves out leagues whose clubs never play UEFA competitions
    slugs = cup.covered_slugs()
    assert "eng.1" in slugs and "aut.1" not in slugs and "usa.1" not in slugs and "jpn.1" not in slugs


def test_cup_weights_never_fall_back_to_league_defaults():
    s = Settings()
    w = load_weights(s, "NO_SUCH_CUP", default=s.cup_weights)
    assert w == s.cup_weights and "dixon_coles" not in w
    assert max(w, key=w.get) == "market_view"                # the market leads, as in the leagues
    assert load_weights(s, "NO_SUCH_LEAGUE") != s.cup_weights


def test_locate_teams():
    cands = {"E0": ["Man City", "Arsenal"], "P1": ["Moreirense", "Porto"], "D1": ["Heidenheim", "Bayern Munich"]}
    assert cup._locate("Manchester City", "eng.1", cands) == ("Man City", "E0")
    # a promoted club must not fuzzy-match a different club of its country
    assert cup._locate("Torreense", "por.1.promotion.relegation", cands) == (None, "P1")
    # ESPN filed under a UEFA competition: exact/alias match across the covered leagues
    assert cup._locate("1. FC Heidenheim 1846", "uefa.europa.conf", cands) == ("Heidenheim", "D1")
    assert cup._locate("SK Sturm Graz", "aut.1", cands) == (None, "x:aut")
    assert cup._locate("Qarabag", "uefa.champions", cands) == (None, "x:?")
    assert cup.country("x:aut") == "aut" and cup.country("E0") == "English Premier League"


def _two_leagues(seed=0, gap=0.45):
    """League A (E0) is stronger than league B (x:bbb) by ``gap`` log-goals; only the
    'European' games between them can tell."""
    rng = np.random.default_rng(seed)
    a = [f"A{i}" for i in range(8)]
    b = [f"B{i}" for i in range(8)]
    strength = {t: s for t, s in zip(a, np.linspace(-0.2, 0.2, 8))} | \
               {t: s - gap for t, s in zip(b, np.linspace(-0.2, 0.2, 8))}
    rows, d = [], pd.Timestamp("2024-08-01")
    for rnd in range(56):
        for teams, div in ((a, "E0"), (b, "BBB")):
            perm = rng.permutation(teams)
            for h, aw in zip(perm[::2], perm[1::2]):
                lh = np.exp(0.15 + 0.2 + strength[h] - strength[aw])
                la = np.exp(0.15 + strength[aw] - strength[h])
                rows.append((d, div, h, aw, rng.poisson(lh), rng.poisson(la), False))
        if rnd % 2:
            for h, aw in zip(rng.permutation(a)[:4], rng.permutation(b)[:4]):
                if rng.random() < 0.5:
                    h, aw = aw, h
                lh = np.exp(0.15 + 0.2 + strength[h] - strength[aw])
                la = np.exp(0.15 + strength[aw] - strength[h])
                rows.append((d + pd.Timedelta(days=3), cup.EURO_DIV, h, aw, rng.poisson(lh), rng.poisson(la), False))
        d += pd.Timedelta(days=7)
    hist = pd.DataFrame(rows, columns=["date", "div", "home", "away", "hg", "ag", "neutral"]).astype({"hg": float, "ag": float})
    hist["season"] = np.where(hist["date"] < pd.Timestamp("2025-07-01"), 2024, 2025)
    hist["comp"] = np.where(hist["div"] == cup.EURO_DIV, "uefa.champions", hist["div"])
    groups = {t: "E0" for t in a} | {t: "x:bbb" for t in b}
    return hist, groups, d


def test_cross_models_place_leagues_on_one_scale():
    hist, groups, end = _two_leagues()
    dc = cup.CrossDixonColes(groups).fit(hist, end)
    elo = cup.CrossElo(groups).fit(hist, end)
    assert dc.fitted and elo.fitted
    a_mean = np.mean([dc.strength(f"A{i}") for i in range(8)])
    b_mean = np.mean([dc.strength(f"B{i}") for i in range(8)])
    assert a_mean - b_mean > 0.3                              # true gap 0.9 in attack - defence
    assert elo.off["E0"] > elo.off["x:bbb"] + 40
    # the middle of league B against the middle of league A, at B's ground
    ctx = MatchContext("B4", "A4", end, league_home_goals=1.4, league_away_goals=1.2)
    for m in (dc, elo):
        lh, la = m.expected_goals(ctx)
        assert la > lh
    # a newcomer from league B starts at B's level, below an established A club
    assert dc.strength("B_new") < dc.strength("A0")
    groups["B_new"] = "x:bbb"
    assert elo.rating("B_new") < elo.rating("A0")
    # a neutral venue removes the home edge
    lh, la = dc.expected_goals(MatchContext("A4", "A4", end, neutral=True))
    assert lh == pytest.approx(la)


def test_month_results_one_request_per_month(monkeypatch):
    calls = []
    payload = ('{"events": [{"id": "%s", "date": "2025-10-01T19:00Z", "status": {"type": {"completed": true}},'
               '"competitions": [{"neutralSite": false, "competitors": ['
               '{"homeAway": "home", "score": "2", "team": {"id": "1", "displayName": "Home FC"}},'
               '{"homeAway": "away", "score": "1", "team": {"id": "2", "displayName": "Away FC"}}]}]}]}')

    def fake_fetch(url, ttl, offline=False):
        calls.append((url, ttl))
        return payload % url[-15:]
    monkeypatch.setattr(espn, "fetch", fake_fetch)
    out = espn.month_results("uefa.champions", date(2025, 9, 1), date(2025, 11, 20), today=date(2025, 11, 20))
    assert [u.split("dates=")[1][:6] for u, _ in calls] and len(calls) == 3
    assert {u.split("dates=")[1][:6] for u, _ in calls} == {"202509", "202510", "202511"}
    ttl = {u.split("dates=")[1][:6]: t for u, t in calls}
    assert ttl["202509"] is None and ttl["202511"] is not None     # past months cached for good
    assert len(out) == 3 and out[0].home_score == 2 and out[0].status == "completed"


def test_settle_and_snapshot_use_cup_scoreboard(monkeypatch):
    seen = []
    monkeypatch.setattr(espn, "fixtures", lambda code, start, days, offline=False, ttl=1800: seen.append(code) or [])
    cli._fixtures_by_id("UEL", date(2026, 11, 5), ttl=600)
    cli._fixtures_by_id("UCL", date(2026, 11, 5), ttl=600)
    assert seen == ["uefa.europa", "uefa.champions"]


def test_cup_predictor_end_to_end(monkeypatch):
    hist, groups, end = _two_leagues(seed=3)
    cand = {"E0": [f"A{i}" for i in range(8)]}
    data = cup.CrossData(hist, groups, {"101": "A1", "202": "B2"}, cand)
    monkeypatch.setattr(cup, "cross_data", lambda *a, **k: data)
    monkeypatch.setattr(espn, "team_matches", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(espn, "team_info", lambda *a, **k: (None, None))
    p = cup.CupPredictor("UCL", as_of=end.to_pydatetime().replace(tzinfo=timezone.utc), log=lambda *_: None).prepare()
    assert set(p.weights) == set(Settings().cup_weights)
    fx = espn.Fixture("e1", datetime.now(timezone.utc) + timedelta(days=1), "uefa.champions", "Club A1", "Club B2",
                      "101", "202", odds={"H": 1.6, "D": 4.0, "A": 5.5, "line": 2.5, "over": 1.8, "under": 2.0,
                                          "source": "DraftKings"})
    pred = p.predict(fx)
    assert (pred.home, pred.away) == ("A1", "B2")
    assert {"cross_dc", "cross_elo", "market_view"} <= set(pred.per_model)
    assert pred.weights["market_view"] > 0.5
    assert sum(pred.ensemble[k] for k in ("home", "draw", "away")) == pytest.approx(1)
    assert pred.ensemble["home"] > pred.ensemble["away"]
    assert any("doesn't model" in w for w in pred.warnings)          # B2's league isn't covered
    assert "home_xelo" in pred.features and pred.analysis["groups"] == {"home": "E0", "away": "x:bbb"}
    # a club never seen before, with no country: newcomer prior and a warning
    fx2 = espn.Fixture("e2", fx.kickoff, "uefa.champions", "Club A2", "Brand New FC", None, "999")
    pred2 = p.predict(fx2)
    assert pred2.away == "Brand New FC" and data.groups["Brand New FC"] == cup.UNKNOWN_GROUP
    assert any("ESPN gives no country" in w for w in pred2.warnings)
    assert "market_view" not in pred2.per_model                       # no odds: models only


def test_cup_optimised_weights_keep_market_share():
    rows = []
    rng = np.random.default_rng(1)
    for _ in range(60):
        rec = {}
        for n in ("cross_dc", "cross_elo", "form", "h2h", "ensemble", "base_rates"):
            rec[n] = {k: float(rng.uniform(0.2, 0.8)) for k in ("1x2", "btts", "gg2", "o1.5", "o2.5", "o3.5")}
            rec[n] |= {"rps": 0.2, "hit": True, "available": True}
        rows.append(rec)
    res = {"rows": rows, "models": ["cross_dc", "cross_elo", "form", "h2h", "ensemble", "base_rates"]}
    w = cup.optimise_weights(res)
    s = Settings()
    share = s.cup_weights["market_view"] / sum(s.cup_weights.values())
    assert w["market_view"] == pytest.approx(share, abs=1e-3)
    assert sum(w.values()) == pytest.approx(1, abs=1e-3) and "base_rates" not in w
