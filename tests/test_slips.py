"""Flexibet slip maths, the builders and settlement (no database needed)."""
import math
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from mbtgrandaudit import slips as sl


def test_esp_matches_brute_force():
    v = [0.5, 1.2, 3.0, 0.1, 2.2]
    e = sl.esp(v, 3)
    for k in range(4):
        assert e[k] == pytest.approx(sum(math.prod(c) for c in combinations(v, k)))


def test_loss_dist_is_binomial_for_equal_legs():
    d = sl.loss_dist([0.8] * 10)
    assert d.sum() == pytest.approx(1)
    for m in range(11):
        assert d[m] == pytest.approx(math.comb(10, m) * 0.2 ** m * 0.8 ** (10 - m))


def test_cut_odds_steps_match_real_slip_ratios():
    # a real 50-leg SportyBet slip: full 62,491,510.82; 1 cut 7,567,522.56; 2 cuts 495,218.91.
    # The rule's step ratio from 1 to 2 cuts is sum(e0..e2)/sum(e0..e1), independent of c.
    o = [3.97, 1.09, 4.21, 1.08, 1.12, 1.11, 1.14, 1.13, 1.12, 1.04] + [1.12] * 30 + [3.5] * 10
    r = sl.cut_odds(o, 1) / sl.cut_odds(o, 2)
    assert 5 < r < 50                       # same order as the real 15.3
    assert sl.cut_odds(o, 0) > sl.cut_odds(o, 1) > sl.cut_odds(o, 2) > sl.cut_odds(o, 3)


def test_multiplier_interpolates():
    assert sl.multiplier(30) == pytest.approx(2.0)
    assert sl.multiplier(50) == pytest.approx(4.3)
    assert 2.0 < sl.multiplier(40) < 4.3


def _pool(n=150, seed=1):
    rng = np.random.default_rng(seed)
    py = rng.uniform(0.10, 0.31, n)
    margin = 1.06
    return pd.DataFrame(dict(
        event_id=[f"e{i}" for i in range(n)], league=["L"] * n,
        kickoff=[f"2026-11-21 {12 + i % 9}:00" for i in range(n)], match=[f"H{i} v A{i}" for i in range(n)],
        py=py, pn=1 - py, oy=np.round(1 / (py * margin), 2), on=np.round(1 / ((1 - py) * margin), 2)))


def test_aggressive_slips_reach_the_max_payout():
    df = _pool()
    slips = sl.aggressive(df, n_slips=3, tries=800)
    assert slips
    for s in slips:
        assert len(s.legs) == 50 and sum(l.pick == "Yes" for l in s.legs) == 11 and s.cuts == 2
        assert s.est_cut_odds * 10 >= sl.MAX_PAYOUT
        assert len({l.event_id for l in s.legs}) == 50


def test_middle_slips_hit_their_floor_and_use_fewer_yes_legs():
    df = _pool()
    slips = sl.middle(df, n_slips=3, tries=800)
    assert slips
    for s in slips:
        y = sum(l.pick == "Yes" for l in s.legs)
        assert 7 <= y <= 9 and s.cuts in (1, 2)
        assert s.est_cut_odds >= 15000
    # the No legs are the most confident ones
    no = {l.event_id for l in slips[0].legs if l.pick == "No"}
    yes = {l.event_id for l in slips[0].legs if l.pick == "Yes"}
    best_no = [i for i in df.sort_values("pn", ascending=False)["event_id"] if i not in yes][: len(no)]
    assert no == set(best_no)


def test_too_small_a_pool_builds_nothing():
    assert sl.aggressive(_pool(20), tries=50) == []
    assert sl.middle(_pool(20), tries=50) == []


def test_leg_result():
    assert sl.leg_result("Yes", 2, 2) == "won"
    assert sl.leg_result("Yes", 2, 1) == "lost"
    assert sl.leg_result("No", 3, 1) == "won"
    assert sl.leg_result("No", 2, 3) == "lost"
    assert sl.leg_result("No", None, None) == "open"


def test_settle_counts_cuts(tmp_path):
    legs = [sl.Leg(f"e{i}", "L", "2026-11-21 15:00", f"H{i} v A{i}", "No", 0.85, 1.12) for i in range(10)]
    legs[0].pick, legs[0].odds = "Yes", 4.0
    s = sl.Slip("middle", "Middle slip 1", 1, legs)
    from datetime import datetime, timezone
    sl.save([s], tmp_path / "s.json", 10, datetime(2026, 11, 20, tzinfo=timezone.utc))
    [s2], stake = sl.load(tmp_path / "s.json")
    res = {f"e{i}": (1, 0) for i in range(10)}            # Yes leg loses, every No wins
    md, done = sl.settle([s2], res, stake)
    assert done and "WON with 1 lost" in md
    res["e1"] = (2, 2)                                     # a second loss: over the 1 cut
    md, _ = sl.settle([s2], res, stake)
    assert "| Lost |" in md
    md, done = sl.settle([s2], {"e0": (2, 2)}, stake)      # others still to play
    assert not done and "Still alive" in md


def test_entered_odds_stay_out_of_the_model_by_default(monkeypatch):
    from mbtgrandaudit.config import Settings
    monkeypatch.delenv("MBT_ENTERED_ODDS_IN_MODEL", raising=False)
    assert Settings().entered_odds_in_model is False
    monkeypatch.setenv("MBT_ENTERED_ODDS_IN_MODEL", "1")
    assert Settings().entered_odds_in_model is True
