import numpy as np
import pandas as pd
import pytest

from mbtgrandaudit.data.espn import american_to_decimal
from mbtgrandaudit.data.names import match_team
from mbtgrandaudit.markets import implied_from_odds, markets
from mbtgrandaudit.models.base import (MatchContext, bivariate_poisson_matrix, dc_tau_matrix,
                                      poisson_matrix)
from mbtgrandaudit.models.ratings import Elo
from mbtgrandaudit.models.strength import DixonColes


def test_markets_consistency():
    m = markets(poisson_matrix(1.6, 1.1))
    assert m["home"] + m["draw"] + m["away"] == pytest.approx(1)
    assert m["btts_yes"] + m["btts_no"] == pytest.approx(1)
    assert m["gg2_yes"] < m["btts_yes"]
    assert m["over_1.5"] > m["over_2.5"] > m["over_3.5"]
    assert m["xg_home"] == pytest.approx(1.6, abs=1e-3)
    # independent Poisson: BTTS = (1 - e^-lh)(1 - e^-la)
    assert m["btts_yes"] == pytest.approx((1 - np.exp(-1.6)) * (1 - np.exp(-1.1)), abs=1e-6)


def test_dixon_coles_raises_draws_when_rho_negative():
    base = poisson_matrix(1.3, 1.1)
    adj = dc_tau_matrix(base, 1.3, 1.1, -0.1)
    assert np.trace(adj) > np.trace(base)
    assert adj.sum() == pytest.approx(1)


def test_bivariate_reduces_to_poisson():
    assert np.allclose(bivariate_poisson_matrix(1.4, 1.0, 0.0), poisson_matrix(1.4, 1.0))
    bp = markets(bivariate_poisson_matrix(1.2, 0.8, 0.2))
    assert bp["xg_home"] == pytest.approx(1.4, abs=1e-3)


def test_odds_helpers():
    assert american_to_decimal("+150") == pytest.approx(2.5)
    assert american_to_decimal(-200) == pytest.approx(1.5)
    imp = implied_from_odds({"H": 2.0, "D": 3.5, "A": 4.0})
    assert sum(imp.values()) == pytest.approx(1)


def test_name_matching():
    cands = ["Man United", "Man City", "Leeds", "Nott'm Forest", "Sheffield Weds", "Sheffield United"]
    assert match_team("Manchester United", cands) == "Man United"
    assert match_team("Leeds United", cands) == "Leeds"
    assert match_team("Nottingham Forest", cands) == "Nott'm Forest"
    assert match_team("Sheffield Wednesday", cands) == "Sheffield Weds"


def _synthetic(n_rounds=60, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(10)]
    strength = np.linspace(-0.5, 0.5, 10)
    rows, d = [], pd.Timestamp("2024-08-01")
    for r in range(n_rounds):
        perm = rng.permutation(10)
        for a, b in zip(perm[::2], perm[1::2]):
            lh = np.exp(0.15 + 0.25 + strength[a] - strength[b] * 0.5)
            la = np.exp(0.15 + strength[b] - strength[a] * 0.5)
            rows.append({"date": d, "div": "X", "home": teams[a], "away": teams[b],
                         "hg": rng.poisson(lh), "ag": rng.poisson(la), "season": 2024})
        d += pd.Timedelta(days=4)
    return pd.DataFrame(rows), d


def test_models_recover_strength_order():
    hist, as_of = _synthetic()
    dc = DixonColes().fit(hist, as_of, "X")
    elo = Elo().fit(hist, as_of, "X")
    for model in (dc, elo):
        strong = markets(model.predict(MatchContext("T9", "T0", as_of)))
        weak = markets(model.predict(MatchContext("T0", "T9", as_of)))
        assert strong["home"] > 0.6 > weak["home"]
    assert 0.1 < dc.home_adv < 0.45


def test_store_roundtrip(tmp_path):
    from mbtgrandaudit.storage import Store, pack, unpack
    m = poisson_matrix(1.5, 1.0)
    assert np.allclose(unpack(pack(m)), m / m.sum(), atol=1e-6)
    st = Store(tmp_path / "t.db")
    kw = dict(league="EPL", home="A", away="B", model_home="A", model_away="B",
              kickoff=pd.Timestamp("2026-01-10 15:00"), version="v1", source="live", matrix=m,
              markets=markets(m), weights={"poisson": 1.0}, features={"x": 1.0},
              model_matrices={"poisson": m})
    st.add_prediction(event_id="1", predicted_at=pd.Timestamp("2026-01-09"), **kw)
    st.add_prediction(event_id="1", predicted_at=pd.Timestamp("2026-01-10 14:00"), **kw)
    st.add_prediction(event_id="1", predicted_at=pd.Timestamp("2026-01-10 16:00"), **kw)  # after kick-off
    st.add_result("1", "EPL", 2, 1, "test")
    st.commit()
    s = st.settled()
    assert len(s) == 1 and s["predicted_at"].iloc[0].startswith("2026-01-10T14:00")
    assert len(st.model_matrices(s["id"])) == 1
    st.close()


@pytest.mark.skipif(not __import__("os").environ.get("TEST_DATABASE_URL"),
                    reason="set TEST_DATABASE_URL to test the Postgres backend")
def test_postgres_store_matches_sqlite(tmp_path):
    import os
    from mbtgrandaudit.storage import Store
    m = poisson_matrix(1.4, 1.1)
    pg = Store(url=os.environ["TEST_DATABASE_URL"], schema="mbt_pytest")
    pg.execute("DROP SCHEMA IF EXISTS mbt_pytest CASCADE")
    pg.commit()
    pg.close()
    pg = Store(url=os.environ["TEST_DATABASE_URL"], schema="mbt_pytest")
    lite = Store(tmp_path / "t.db")
    try:
        for st in (lite, pg):
            for when in ("2026-01-09", "2026-01-10 14:00"):
                st.add_prediction(event_id="9", league="EPL", home="A", away="B", model_home="A",
                                  model_away="B", kickoff=pd.Timestamp("2026-01-10 15:00"),
                                  predicted_at=pd.Timestamp(when), version="v1", source="live", matrix=m,
                                  markets=markets(m), weights={"poisson": 1.0}, features={},
                                  model_matrices={"poisson": m, "elo": m})
            st.add_odds("9", pd.Timestamp("2026-01-10 14:00"), "prediction",
                        {"H": 2.1, "D": 3.4, "A": 3.6, "line": 2.5, "over": 1.9, "under": 1.95})
            st.add_odds("9", pd.Timestamp("2026-01-10 14:50"), "closing", {"H": 2.0, "D": 3.5, "A": 3.8})
            st.add_result("9", "EPL", 1, 1, "test")
            st.commit()
        a, b = lite.settled(), pg.settled()
        assert len(a) == len(b) == 1 and a["predicted_at"][0] == b["predicted_at"][0]
        assert np.allclose(np.frombuffer(bytes(a["matrix"][0]), np.uint8),
                           np.frombuffer(bytes(b["matrix"][0]), np.uint8))
        assert len(pg.model_matrices(b["id"])) == 2
        assert len(pg.odds_for(["9"])) == len(lite.odds_for(["9"])) == 2
        assert lite.counts() == pg.counts()
    finally:
        pg.execute("DROP SCHEMA IF EXISTS mbt_pytest CASCADE")
        pg.close()
        lite.close()


def test_odds_issue_import(tmp_path):
    from datetime import datetime, timezone
    from mbtgrandaudit.cli import _apply_issue, _parse_issue
    from mbtgrandaudit.storage import Store
    m = poisson_matrix(1.5, 1.2)
    st = Store(tmp_path / "t.db")
    for ev, ko in (("101", "2099-01-01 15:00"), ("102", "2000-01-01 15:00")):   # future / past kick-off
        st.add_prediction(event_id=ev, league="EPL", home="Home", away="Away", model_home="Home",
                          model_away="Away", kickoff=pd.Timestamp(ko), predicted_at=pd.Timestamp("2000-01-01"),
                          version="v1", source="live", matrix=m, markets=markets(m), weights={},
                          features={}, model_matrices={})
    body = ("Odds entered on the MBTgrandaudit odds form.\nBookmaker: Book\n\n"
            "> Format: event id | market | decimal odds | match\n\n"
            "101 | gg2_yes | 6.50 | Home v Away\n101 | ng2 | 1,10 | Home v Away\n"
            "102 | gg2_yes | 7.00 | Home v Away\n999 | gg2_yes | 7.00 | X v Y\n101 | o3 | 2.0 | Home v Away\n")
    bookmaker, lines = _parse_issue(body)
    assert bookmaker == "Book" and len(lines) == 5
    md, saved = _apply_issue(st, body, datetime.now(timezone.utc))
    assert saved == 2                                   # 101 gg2_yes + gg2_no (comma decimal)
    assert "already kicked off" in md and "unknown fixture" in md and "half-goal" in md
    mo = st.market_odds_for(["101"])
    assert sorted(mo["market"]) == ["gg2_no", "gg2_yes"] and set(mo["bookmaker"]) == {"Book"}
    st.close()


def test_market_view_uses_entered_prices():
    from mbtgrandaudit.implied import implied_markets
    base = {"H": 2.65, "D": 3.55, "A": 2.6, "line": 2.5, "over": 1.69, "under": 2.15}
    plain = implied_markets(base)
    high = implied_markets(base, extra={"gg2_yes": 3.2})      # market thinks GG2+ is likelier
    low = implied_markets(base, extra={"gg2_no": 1.08})       # ... or less likely
    assert high["gg2_yes"] > plain["gg2_yes"] > low["gg2_yes"]
    assert "GG2+" in high["info"]["matched"]
    assert abs(high["home"] - plain["home"]) < 0.02           # 1X2 still matched
    both = implied_markets(base, extra={"btts_yes": 1.55, "btts_no": 2.35, "over_3.5": 2.6, "under_3.5": 1.45})
    assert {"BTTS", "O/U 3.5"} <= set(both["info"]["matched"])


def _apif_fixture(fid, date, home, away):
    return {"fixture": {"id": fid, "date": date}, "teams": {"home": {"name": home}, "away": {"name": away}}}


def test_apifootball_parse_and_match():
    from mbtgrandaudit.data.apifootball import match_fixtures, parse_odds
    resp = [{"bookmakers": [
        {"name": "Bet365", "bets": [
            {"name": "Match Winner", "values": [{"value": "Home", "odd": "1.50"}, {"value": "Draw", "odd": "4.00"},
                                                {"value": "Away", "odd": "6.00"}]},
            {"name": "Goals Over/Under", "values": [{"value": "Over 1.5", "odd": "1.20"},
                                                    {"value": "Under 1.5", "odd": "4.50"},
                                                    {"value": "Over 2.25", "odd": "1.60"},    # Asian line: ignored
                                                    {"value": "Over 5.5", "odd": "9.00"}]},   # beyond 4.5: ignored
            {"name": "Both Teams Score", "values": [{"value": "Yes", "odd": "1.80"}, {"value": "No", "odd": "1.95"}]},
            {"name": "Exact Score", "values": [{"value": "1:0", "odd": "7.00"}]}]},
        {"name": "Pinnacle", "bets": [
            {"name": "Match Winner", "values": [{"value": "Home", "odd": "1.55"}, {"value": "Draw", "odd": "bad"}]}]}]}]
    got = parse_odds(resp)
    assert ("Bet365", "over_1.5", 1.2) in got and ("Bet365", "btts_no", 1.95) in got
    assert ("Pinnacle", "home", 1.55) in got
    assert {k for _, k, _ in got} == {"home", "draw", "away", "over_1.5", "under_1.5", "btts_yes", "btts_no"}
    ours = pd.DataFrame([
        {"event_id": "1", "home": "Wolverhampton Wanderers", "away": "Portsmouth", "kickoff": "2026-10-20T19:00:00+00:00"},
        {"event_id": "2", "home": "Manchester United", "away": "Leeds United", "kickoff": "2026-10-21T19:00:00+00:00"},
        {"event_id": "3", "home": "Celtic", "away": "Rangers", "kickoff": "2026-10-21T12:00:00+00:00"}])
    api = [_apif_fixture(11, "2026-10-20T19:00:00+00:00", "Wolves", "Portsmouth"),
           _apif_fixture(12, "2026-10-21T19:00:00+00:00", "Manchester United", "Leeds"),
           _apif_fixture(13, "2026-10-21T19:00:00+00:00", "Manchester City", "Leeds"),
           _apif_fixture(14, "2026-10-22T12:00:00+00:00", "Celtic", "Rangers")]   # a day off: no match
    assert match_fixtures(ours, api) == {"1": 11, "2": 12}


def test_apifootball_fetch_stores_and_skips(tmp_path):
    from datetime import datetime, timezone
    from mbtgrandaudit.data.apifootball import Client, fetch
    from mbtgrandaudit.storage import Store
    m = poisson_matrix(1.5, 1.2)
    st = Store(tmp_path / "t.db")
    for ev, home, away in (("101", "Arsenal", "Chelsea"), ("102", "Everton", "Fulham")):
        st.add_prediction(event_id=ev, league="EPL", home=home, away=away, model_home=home, model_away=away,
                          kickoff=pd.Timestamp("2026-10-21 15:00", tz="UTC"), predicted_at=pd.Timestamp("2026-10-20"),
                          version="v1", source="live", matrix=m, markets=markets(m), weights={}, features={},
                          model_matrices={})
    odds = {"bookmakers": [{"name": "Bet365", "bets": [
        {"name": "Both Teams Score", "values": [{"value": "Yes", "odd": "1.70"}, {"value": "No", "odd": "2.10"}]}]}]}
    calls = []

    class Fake:
        def __init__(self, path):
            calls.append(path)
            body = ([_apif_fixture(1, "2026-10-21T15:00:00+00:00", "Arsenal", "Chelsea"),
                     _apif_fixture(2, "2026-10-21T15:00:00+00:00", "Everton", "Fulham")]
                    if path.startswith("fixtures") else [odds])
            self.data, self.headers = {"errors": [], "response": body}, {"x-ratelimit-requests-remaining": "50"}

        def __enter__(self):
            import io, json
            r = io.StringIO(json.dumps(self.data))
            r.headers = self.headers
            return r

        def __exit__(self, *a):
            return False

    client = Client("k", budget=10, min_interval=0, opener=lambda req, timeout: Fake(req.full_url.split(".io/")[1]))
    now = datetime(2026, 10, 20, 7, tzinfo=timezone.utc)
    s = fetch(st, datetime(2026, 10, 21, tzinfo=timezone.utc), now, client, log=lambda *_: None)
    assert s["matched"] == 2 and s["rows"] == 4 and client.used == 3 and client.remaining == 50
    mo = st.market_odds_for(["101", "102"])
    assert len(mo) == 4 and set(mo["kind"]) == {"prediction"} and set(mo["bookmaker"]) == {"Bet365"}
    s = fetch(st, datetime(2026, 10, 21, tzinfo=timezone.utc), now, client, log=lambda *_: None)
    assert s["skipped"] == 2 and s["fetched"] == 0 and client.used == 4      # only the fixtures call
    tight = Client("k", budget=2, min_interval=0, opener=lambda req, timeout: Fake(req.full_url.split(".io/")[1]))
    later = datetime(2026, 10, 20, 21, tzinfo=timezone.utc)                    # past the 12h skip window
    s = fetch(st, datetime(2026, 10, 21, tzinfo=timezone.utc), later, tight, log=lambda *_: None)
    assert s["fetched"] == 1 and tight.used == 2                               # budget stops the run
    st.close()
