"""Screenshot odds import: market rules, dates, matching, and a golden reading of a real
SportyBet GG/NG 2+ screenshot (recorded model output, so no API call is made)."""
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from mbtgrandaudit import screenshot as shot
from mbtgrandaudit.cli import _apply_issue, _market_key

READING = json.loads((Path(__file__).parent / "fixtures" / "sportybet_gg2_reading.json").read_text())
NOW = datetime(2026, 11, 5, 17, 0, tzinfo=timezone.utc)

# logged fixtures as ESPN names them
EVENTS = pd.DataFrame([
    ("401", "Everton", "Coventry City", "2026-11-06 20:00:00+00:00"),
    ("402", "Leeds United", "Tottenham Hotspur", "2026-11-07 12:30:00+00:00"),
    ("403", "Arsenal", "Hull City", "2026-11-07 15:00:00+00:00"),
    ("404", "Fulham", "Newcastle United", "2026-11-07 15:00:00+00:00"),
    ("405", "Nottingham Forest", "Manchester City", "2026-11-07 17:30:00+00:00"),
    ("406", "Manchester United", "Chelsea", "2026-11-08 16:30:00+00:00"),
], columns=["event_id", "home", "away", "kickoff"])


@pytest.mark.parametrize("tab, cols, line, codes", [
    ("GG/NG 2+", ["GG", "NG"], None, ["gg2", "ng2"]),
    ("GG/NG", ["GG", "NG"], None, ["gg", "ng"]),
    ("BTTS", ["Yes", "No"], None, ["gg", "ng"]),
    ("1X2", ["1", "X", "2"], None, ["1", "X", "2"]),
    ("Over/Under", ["Over", "Under"], 2.5, ["o2.5", "u2.5"]),
])
def test_market_codes(tab, cols, line, codes):
    assert shot.market_codes(tab, cols, line) == codes
    for c in codes:
        _market_key(c)                     # every code is one the issue import accepts


@pytest.mark.parametrize("tab, cols, line, why", [
    ("1st Half 1X2", ["1", "X", "2"], None, "isn't supported"),
    ("Double Chance", ["1X", "12", "X2"], None, "isn't supported"),
    ("GG/NG 2+", ["1", "X", "2"], None, "columns"),       # tab and columns disagree
    ("Over/Under", ["Over", "Under"], None, "goal line"),
    ("Over/Under", ["Over", "Under"], 3, "half-goal"),
    ("", ["GG", "NG"], None, "couldn't be read"),
])
def test_market_codes_refuse(tab, cols, line, why):
    with pytest.raises(ValueError, match=why):
        shot.market_codes(tab, cols, line)


def test_resolve_date():
    assert shot.resolve_date("11/07", date(2026, 11, 5)) == (date(2026, 11, 7), None)
    assert shot.resolve_date("01/03", date(2026, 12, 30))[0] == date(2027, 1, 3)    # over new year
    d, problem = shot.resolve_date("11/07", date(2026, 11, 5), "Friday")
    assert d == date(2026, 11, 7) and "Saturday" in problem
    assert shot.resolve_date(None, date(2026, 11, 5)) == (None, None)


def test_same_team():
    assert shot.same_team("Man City", "Manchester City")
    assert shot.same_team("Tottenham", "Tottenham Hotspur")
    assert shot.same_team("Newcastle", "Newcastle United")
    assert not shot.same_team("Man City", "Manchester United")


def test_golden_sportybet_gg2():
    p = shot.propose([READING], EVENTS, NOW, {"405": {"gg2_yes": 0.25, "gg2_no": 0.75}})
    assert not p.errors and p.bookmaker == "SportyBet"
    assert p.lines == [
        ("401", "gg2", 4.98, "Everton v Coventry City"), ("401", "ng2", 1.14, "Everton v Coventry City"),
        ("402", "gg2", 5.21, "Leeds United v Tottenham Hotspur"), ("402", "ng2", 1.13, "Leeds United v Tottenham Hotspur"),
        ("403", "gg2", 7.78, "Arsenal v Hull City"), ("403", "ng2", 1.05, "Arsenal v Hull City"),
        ("404", "gg2", 4.17, "Fulham v Newcastle United"), ("404", "ng2", 1.19, "Fulham v Newcastle United"),
        ("405", "gg2", 4.50, "Nottingham Forest v Manchester City"),
        ("405", "ng2", 1.17, "Nottingham Forest v Manchester City"),
    ]
    md = shot.report(p)
    assert "Bookmaker: SportyBet" in md and "405 | gg2 | 4.50 | Nottingham Forest v Manchester City" in md
    assert "+12.5%" in md                   # 0.25 * 4.50 - 1
    assert "11/08" in md                     # the model's warning is passed on


def test_unclear_tab_proposes_nothing():
    p = shot.propose([dict(READING, market_tab_confidence="low")], EVENTS, NOW)
    assert not p.lines and "isn't clear" in p.errors[0]


def test_rows_above_first_separator_take_previous_date():
    second = {"market_tab": "GG/NG 2+", "market_tab_confidence": "high", "columns": ["GG", "NG"],
              "rows": [{"date": None, "time": "16:30", "home": "Man United", "away": "Chelsea",
                        "prices": [5.0, 1.15]}]}
    p = shot.propose([READING, second], EVENTS, NOW)
    assert ("406", "gg2", 5.0, "Manchester United v Chelsea") in p.lines   # no date: matched on teams


def test_misread_and_kicked_off_rows_skipped():
    rows = [dict(READING["rows"][0], prices=[4.98, 4.14]),              # 44% margin: misread
            dict(READING["rows"][1], prices=[0.9, 1.13])]
    p = shot.propose([dict(READING, rows=rows)], EVENTS, NOW)
    assert not p.lines and "margin" in "\n".join(p.skipped) and "1.00" in "\n".join(p.skipped)
    late = shot.propose([READING], EVENTS, datetime(2026, 11, 7, 13, 0, tzinfo=timezone.utc))
    assert "already kicked off" in "\n".join(late.skipped)


def test_later_screenshot_wins():
    again = dict(READING, rows=[dict(READING["rows"][0], prices=[5.10, 1.13])])
    p = shot.propose([READING, again], EVENTS, NOW)
    assert ("401", "gg2", 5.10, "Everton v Coventry City") in p.lines
    assert sum(1 for ln in p.lines if ln[:2] == ("401", "gg2")) == 1


def test_proposed_lines_import_through_issue_flow(tmp_path):
    from mbtgrandaudit.markets import markets
    from mbtgrandaudit.models.base import poisson_matrix
    from mbtgrandaudit.storage import Store
    st = Store(tmp_path / "t.db")
    m = poisson_matrix(1.5, 1.2)
    for r in EVENTS.itertuples():
        st.add_prediction(event_id=r.event_id, league="EPL", home=r.home, away=r.away, model_home=r.home,
                          model_away=r.away, kickoff=pd.Timestamp("2099-01-01 15:00"),
                          predicted_at=pd.Timestamp("2000-01-01"), version="v1", source="live", matrix=m,
                          markets=markets(m), weights={}, features={}, model_matrices={})
    st.commit()
    p = shot.propose([READING], shot.upcoming_events(st, NOW).assign(
        kickoff=lambda d: d["event_id"].map(dict(zip(EVENTS.event_id, EVENTS.kickoff)))), NOW)
    md = shot.report(p)
    block = md.split("```")[1]
    _, saved = _apply_issue(st, block, NOW)
    assert saved == 10
    st.close()
