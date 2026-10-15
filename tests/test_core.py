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
