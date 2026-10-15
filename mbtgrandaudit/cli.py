"""Command-line interface.

    python -m mbtgrandaudit leagues
    python -m mbtgrandaudit fixtures --league EPL --days 7
    python -m mbtgrandaudit predict --league EPL LALIGA --days 7 --html report.html
    python -m mbtgrandaudit match EPL "Arsenal" "Chelsea"
    python -m mbtgrandaudit ratings --league EPL
    python -m mbtgrandaudit backtest --league EPL --seasons 2 --save-weights

Prediction log (SQLite, data/predictions.db):
    python -m mbtgrandaudit predict --log            # store predictions
    python -m mbtgrandaudit snapshot-odds            # record closing odds shortly before kick-off
    python -m mbtgrandaudit settle                   # fill in final scores
    python -m mbtgrandaudit backfill --seasons 2     # add walk-forward history
    python -m mbtgrandaudit evaluate --source live --html reports/track_record.html
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import backtest, evaluate, report
from .config import LEAGUES, MAJOR, Settings, market_group
from .data import espn, footballdata as fd
from .data.espn import Fixture
from .predictor import LeaguePredictor
from . import storage
from .storage import Store, model_version


def _leagues(arg: list[str] | None) -> list[str]:
    if not arg:
        return MAJOR
    if arg == ["all"]:
        return list(LEAGUES)
    keys = [a.upper() for a in arg]
    bad = [k for k in keys if k not in LEAGUES]
    if bad:
        sys.exit(f"Unknown league(s): {', '.join(bad)}. Run `python -m mbtgrandaudit leagues`.")
    return keys


def _settings(a) -> Settings:
    s = Settings()
    if getattr(a, "train_seasons", None):
        s.seasons = a.train_seasons
    return s


def cmd_leagues(a):
    print(f"{'key':10s}{'league':30s}{'football-data':>15s}{'ESPN':>10s}  second tier")
    for lg in LEAGUES.values():
        print(f"{lg.key:10s}{lg.name:30s}{lg.fd_code:>15s}{lg.espn_code:>10s}  {lg.lower_fd_code or '-'}")
    print(f"\nDefault (no --league): {' '.join(MAJOR)}.  Use --league all for every league.")


def cmd_fixtures(a):
    for key in _leagues(a.league):
        p = LeaguePredictor(key, offline=a.offline, log=lambda *_: None)
        fx = p.fixtures(a.days)
        print(f"\n{LEAGUES[key].name} - next {a.days} days: {len(fx)} fixtures")
        for f in fx:
            print(f"  {report._local(f.kickoff)}  {f.home} vs {f.away}")


def cmd_predict(a):
    s = _settings(a)
    by_league, records = {}, []
    idle = {}   # league name -> next kick-off beyond the window (None if none found)
    for key in _leagues(a.league):
        p = LeaguePredictor(key, s, offline=a.offline)
        fx = p.fixtures(a.days)
        if a.team:
            needle = a.team.lower()
            fx = [f for f in fx if needle in f.home.lower() or needle in f.away.lower()]
        if not fx:
            print(f"[{key}] no upcoming fixtures in the next {a.days} days")
            if not a.team:   # keep the league on the main page, with its next fixture date
                later = p.fixtures(LOOKAHEAD_DAYS)
                idle[LEAGUES[key].name] = later[0].kickoff if later else None
            continue
        p.prepare()
        print(f"[{key}] analysing {len(fx)} fixtures (form + head-to-head across all competitions)")
        extra = _entered_odds([f.event_id for f in fx])
        preds = [p.predict(f, extra.get(f.event_id)) for f in fx]
        by_league[LEAGUES[key].name] = preds
        print(report.console(preds, a.detail))
        records += report.to_records(preds, key)
        if a.log:
            store = Store()
            version, now = model_version(s), datetime.now(timezone.utc)
            for pr in preds:
                store.add_live(pr, key, version, now)
            store.close()
            print(f"[{key}] logged {len(preds)} predictions (model version {version})")
    _write_outputs(a, by_league, records, idle, a.days)


def _entered_odds(event_ids) -> dict:
    """{event_id: {market: latest odds}} from the market_odds table (e.g. `odds add`)."""
    if not storage.configured():
        return {}
    store = Store()
    try:
        mo = store.market_odds_for(event_ids)
    finally:
        store.close()
    if mo.empty:
        return {}
    mo = mo[mo["kind"] != "closing"].sort_values("captured_at")
    latest = mo.groupby(["event_id", "market"])["odds"].last()
    out: dict = {}
    for (ev, market), o in latest.items():
        out.setdefault(ev, {})[market] = float(o)
    return out


def _track_record_page(source: str | None = None) -> tuple[str, dict] | None:
    """Full track-record page: live results if any have settled, else the backtest."""
    if not storage.configured():
        return None
    store = Store()
    try:
        options = [(source, f"Track record: {source or 'all'}", "")] if source is not None else [
            ("live", "Track record: live predictions", "Settled predictions logged with --log. "),
            ("backtest", "Track record: backtest (no settled live predictions yet)",
             "Walk-forward backfill on completed seasons. ")]
        for src, title, note in options:
            ev = evaluate.evaluate(store, src)
            if ev is not None:
                card = evaluate.card_html(title, ev["scores"], ev["baseline"], ev["calibration"],
                                          ev["bet_summary"], note, ev["vs_market"])
                return card, ev
    finally:
        store.close()
    return None


def _fixtures_by_id(league: str, day, ttl: float) -> dict:
    # ESPN groups games by US date, so look at the day either side as well
    fx = espn.fixtures(LEAGUES[league].espn_code, day - timedelta(days=1), 3, ttl=ttl)
    return {f.event_id: f for f in fx}


def cmd_settle(a):
    store = Store()
    pending = store.unsettled(datetime.now(timezone.utc) - timedelta(hours=2, minutes=30))
    if pending.empty:
        print("nothing to settle")
        return store.close()
    pending["day"] = pd.to_datetime(pending["kickoff"], utc=True).dt.date
    is_fd = pending["event_id"].astype(str).str.startswith("fd:")
    settled = _settle_from_football_data(store, pending[is_fd])
    for (league, day), g in pending[~is_fd].groupby(["league", "day"]):
        by_id = _fixtures_by_id(league, day, ttl=600)
        for ev in g["event_id"]:
            f = by_id.get(ev)
            if f and f.status == "completed" and f.home_score is not None and f.away_score is not None:
                store.add_result(ev, league, int(f.home_score), int(f.away_score), "espn")
                settled += 1
    store.close()
    print(f"settled {settled} of {len(pending)} finished fixtures")


def _settle_from_football_data(store, pending) -> int:
    """Results for fixtures only football-data knows (e.g. leagues ESPN doesn't cover)."""
    n = 0
    for league, g in pending.groupby("league"):
        lg = LEAGUES[league]
        hist = fd.load_history([lg.fd_code], 1, datetime.now(timezone.utc).date(), calendar=lg.calendar)
        hist = hist.dropna(subset=["hg", "ag"])
        for ev in g["event_id"]:
            parsed = fd.parse_fd_event_id(ev)
            if parsed is None:
                continue
            _, day, home, away = parsed
            m = hist[(hist["home"] == home) & (hist["away"] == away) & ((hist["date"] - day).abs().dt.days <= 1)]
            if len(m):
                store.add_result(ev, league, int(m["hg"].iloc[0]), int(m["ag"].iloc[0]), "football-data")
                n += 1
    return n


def cmd_snapshot_odds(a):
    store = Store()
    now = datetime.now(timezone.utc)
    up = store.upcoming(now, now + timedelta(hours=a.hours))
    if up.empty:
        print(f"no logged fixtures kicking off in the next {a.hours:g}h")
        return store.close()
    up["day"] = pd.to_datetime(up["kickoff"], utc=True).dt.date
    n = 0
    is_fd = up["event_id"].astype(str).str.startswith("fd:")
    for league, g in up[is_fd].groupby("league"):
        fx = fd.load_fixtures(LEAGUES[league].fd_code, ttl=300)
        for ev in g["event_id"]:
            _, day, home, away = fd.parse_fd_event_id(ev)
            m = fx[(fx["home"] == home) & (fx["away"] == away)]
            if len(m) and pd.notna(m["odds_h"].iloc[0]):
                r = m.iloc[0]
                store.add_odds(ev, now, "closing", {"H": r.odds_h, "D": r.odds_d, "A": r.odds_a,
                                                     "source": "football-data avg"})
                n += 1
    up = up[~is_fd]
    for (league, day), g in up.groupby(["league", "day"]):
        by_id = _fixtures_by_id(league, day, ttl=300)
        for ev in g["event_id"]:
            f = by_id.get(ev)
            if f and f.odds:
                store.add_odds(ev, now, "closing", f.odds)
                n += 1
    store.close()
    print(f"captured closing odds for {n} of {len(up)} fixtures")


def cmd_backfill(a):
    s = _settings(a)
    store = Store()
    for key in _leagues(a.league):
        res = backtest.run(key, a.seasons, s, offline=a.offline, store=store)
        print(f"[{key}] stored {len(res['rows'])} backtest predictions")
    print("log now holds:", store.counts())
    store.close()


def cmd_evaluate(a):
    if not storage.configured():
        sys.exit("No prediction log yet. Run `predict --log` or `backfill` first.")
    store = Store()
    source = None if a.source == "all" else a.source
    league = a.league[0].upper() if a.league else None
    ev = evaluate.evaluate(store, source, league, a.version, a.since)
    if ev is None:
        store.close()
        sys.exit("No settled predictions match those filters (run `settle` after games finish).")
    pd.set_option("display.width", 220)
    print(f"\n{len(ev['preds'])} settled predictions (source={a.source}"
          f"{', league=' + league if league else ''})\n")
    print(ev["scores"].round(4).to_string())
    if ev["baseline"]:
        print("\nbookmaker:", {k: round(v, 4) for k, v in ev["baseline"].items()})
    print("\ncalibration (ensemble), predicted -> actual by probability bin:")
    for key, rows in ev["calibration"].items():
        print(f"  {key:9s} " + "  ".join(f"{p:.0%}->{o:.0%}(n{n})" for p, o, n in rows))
    if ev["vs_market"] is not None:
        print("\nmodel vs market-implied (diff < 0 = model better, |z| > 2 unlikely to be luck):")
        print(ev["vs_market"].round(4).to_string())
    bs = ev["bet_summary"]
    if bs.get("bets"):
        print("\nvalue bets (flat stakes, per-market edge thresholds):")
        for g, v in {**bs["by_market"], "all": bs}.items():
            clv = f"  CLV {v['avg_clv']:+.1%}" if v.get("avg_clv") is not None else ""
            se = f" ± {v['roi_se']:.1%}" if v.get("roi_se") else ""
            print(f"  {g:5s} {v['bets']:5d} bets  win {v['win_rate']:.0%}  odds {v['avg_odds']:.2f}  "
                  f"ROI {v['roi']:+.1%}{se}{clv}")
    else:
        print("\nno value bets with recorded odds")
    if a.bets_csv and not ev["bets"].empty:
        Path(a.bets_csv).parent.mkdir(parents=True, exist_ok=True)
        ev["bets"].to_csv(a.bets_csv, index=False)
        print(f"wrote {a.bets_csv}")
    if a.html:
        path = Path(a.html)
        path.parent.mkdir(parents=True, exist_ok=True)
        card = evaluate.card_html(f"Track record: {a.source}", ev["scores"], ev["baseline"],
                                  ev["calibration"], ev["bet_summary"], vs_market=ev["vs_market"])
        path.write_text(report.page("MBTgrandaudit track record", card, datetime.now(timezone.utc),
                                    '<a href="index.html">← Predictions</a>'), encoding="utf-8")
        print(f"wrote {path}")
    store.close()


MARKET_ALIASES = {
    "1": "home", "x": "draw", "2": "away", "home": "home", "draw": "draw", "away": "away",
    "gg": "btts_yes", "btts": "btts_yes", "ng": "btts_no",
    "gg2": "gg2_yes", "gg2+": "gg2_yes", "btts2": "gg2_yes", "ng2": "gg2_no",
}


def _market_key(text: str) -> str:
    """'gg2+' -> gg2_yes, 'o3.5' -> over_3.5, 'u2.5' -> under_2.5, 'btts_no' -> btts_no."""
    t = text.strip().lower().replace(" ", "")
    if t in MARKET_ALIASES:
        return MARKET_ALIASES[t]
    for prefix, side in (("over", "over"), ("under", "under"), ("o", "over"), ("u", "under")):
        rest = t[len(prefix):].lstrip("_")
        if t.startswith(prefix) and rest.replace(".", "", 1).isdigit():
            if not rest.endswith(".5"):
                raise ValueError(f"'{text}': only half-goal lines (0.5, 1.5, 2.5 ...) are supported; "
                                 "whole lines can be refunded (push) and aren't priced by the model")
            return f"{side}_{rest}"
    if t in {"btts_yes", "btts_no", "gg2_yes", "gg2_no"}:
        return t
    raise ValueError(f"unknown market '{text}' (try 1, X, 2, gg, ng, gg2, ng2, o2.5, u3.5)")


def _add_one(store, home, away, market, odds, bookmaker, now) -> str:
    key = _market_key(market)
    events = store.find_events(home, away, after=now - timedelta(hours=3))
    if events.empty:
        return f"  ! no upcoming logged fixture matches '{home}' v '{away}' (run `predict --log` first)"
    if len(events) > 1:
        names = "; ".join(f"{r.home} v {r.away} ({r.kickoff[:10]})" for r in events.itertuples())
        return f"  ! '{home}' v '{away}' is ambiguous: {names}"
    ev = events.iloc[0]
    store.add_market_odds(ev["event_id"], key, odds, now, "manual", bookmaker)
    line = f"  {ev['home']} v {ev['away']}: {key} @ {odds:.2f}"
    pred = store.latest_prediction(ev["event_id"])
    if pred is not None:
        mk = json.loads(pred["markets"])
        if key in mk:
            prob, edge = mk[key], mk[key] * odds - 1
            need = Settings().min_edge.get(market_group(key), 0.03)
            line += (f"  model {prob:.1%} (fair {1 / prob:.2f})  edge {edge:+.1%}  "
                     f"{'VALUE' if edge >= need else 'no value'} (needs {need:.0%})")
    return line


def cmd_odds(a):
    store = Store()
    now = datetime.now(timezone.utc)
    try:
        if a.action == "add":
            print(_add_one(store, a.home, a.away, a.market, a.price, a.bookmaker or "", now))
        elif a.action == "import":
            df = pd.read_csv(a.file)
            missing = {"home", "away", "market", "odds"} - set(df.columns)
            if missing:
                sys.exit(f"CSV needs columns home, away, market, odds (missing: {', '.join(missing)})")
            for r in df.itertuples():
                print(_add_one(store, r.home, r.away, r.market, float(r.odds),
                               getattr(r, "bookmaker", "") if "bookmaker" in df.columns else "", now))
        else:  # list
            mo = store.query("SELECT m.*, p.home, p.away, p.kickoff FROM market_odds m JOIN"
                             " (SELECT DISTINCT event_id, home, away, kickoff FROM predictions"
                             "  WHERE source = 'live') p ON p.event_id = m.event_id"
                             " WHERE p.kickoff >= ? ORDER BY p.kickoff",
                             (storage._iso(now - timedelta(days=a.days)),))
            if mo.empty:
                print("no entered odds")
            for r in mo.itertuples():
                print(f"  {r.kickoff[:16]}  {r.home} v {r.away}  {r.market} @ {r.odds:.2f}"
                      f"  {r.bookmaker or ''}  ({r.kind}, {r.captured_at[:16]})")
        store.commit()
    except ValueError as e:
        sys.exit(str(e))
    finally:
        store.close()


def cmd_db(a):
    if a.migrate_from:
        src_path = Path(a.migrate_from)
        if not src_path.exists():
            sys.exit(f"{src_path} not found")
        if storage.database_url() is None:
            sys.exit("Set DATABASE_URL (environment or .env) to the Postgres/Supabase database first.")
        src, dst = Store(path=src_path), Store()
        print(f"copying {src.where} -> {dst.where}")
        print("source holds:", src.counts())
        dst.copy_from(src)
        print("destination now holds:", dst.counts())
        src.close()
        dst.close()
        return
    if not storage.configured():
        sys.exit("No prediction log yet.")
    store = Store()
    print(store.where)
    print(store.counts())
    store.close()


LOOKAHEAD_DAYS = 45   # how far to look for an idle league's next fixture


def cmd_match(a):
    s = _settings(a)
    key = _leagues([a.league])[0]
    p = LeaguePredictor(key, s, offline=a.offline).prepare()
    # use the real fixture (ESPN ids give all-competition form) if it is scheduled soon
    fx = next((f for f in p.fixtures(21) if p.resolve(f.home) == p.resolve(a.home)
               and p.resolve(f.away) == p.resolve(a.away)), None)
    if fx is None:
        fx = Fixture("adhoc", datetime.now(timezone.utc) + timedelta(hours=1), key, a.home, a.away)
        print("(not found in upcoming fixtures - using league-only form)")
    pred = p.predict(fx)
    print(report.console([pred], detail=True))
    _write_outputs(a, {LEAGUES[key].name: [pred]}, report.to_records([pred], key))


def cmd_ratings(a):
    key = _leagues(a.league)[0]
    p = LeaguePredictor(key, _settings(a), offline=a.offline).prepare()
    dc = next(m for m in p.models if m.name == "dixon_coles")
    elo = next(m for m in p.models if m.name == "elo")
    df = dc.ratings().loc[lambda d: d.index.isin(p.candidates)]
    df["elo"] = [elo.r.get(t) for t in df.index]
    df["exp_scored_vs_avg"] = (df["attack"]).map(lambda v: f"{100 * (2.718281828 ** v - 1):+.0f}%")
    df["exp_conceded_vs_avg"] = (df["defence"]).map(lambda v: f"{100 * (2.718281828 ** v - 1):+.0f}%")
    df["strength"] = df["attack"] - df["defence"]
    print(f"\n{LEAGUES[key].name} - Dixon-Coles ratings (home adv {dc.home_adv:.3f}, rho {dc.rho:.3f})\n")
    print(df.sort_values("strength", ascending=False).round(3).to_string())


def cmd_backtest(a):
    s = _settings(a)
    for key in _leagues(a.league):
        res = backtest.run(key, a.seasons, s, offline=a.offline)
        table = backtest.summarise(res)
        pd.set_option("display.width", 200)
        print(f"\n{LEAGUES[key].name} - lower log loss / RPS is better\n")
        print(table.round(4).to_string())
        w = backtest.optimise_weights(res, s)
        print("\nOptimised ensemble weights:", ", ".join(f"{k} {v:.2f}" for k, v in w.items()))
        if a.save_weights:
            backtest.save_weights(key, w)
            print("saved to ensemble_weights.json - used automatically for", key)


def _write_outputs(a, by_league, records, idle: dict | None = None, days: int | None = None):
    for opt in ("json", "csv", "html"):
        if getattr(a, opt, None):
            Path(getattr(a, opt)).parent.mkdir(parents=True, exist_ok=True)
    if getattr(a, "json", None):
        report.write_json(records, Path(a.json))
        print(f"wrote {a.json}")
    if getattr(a, "csv", None):
        report.write_csv(records, Path(a.csv))
        print(f"wrote {a.csv}")
    if getattr(a, "html", None) and (by_league or idle):
        path = Path(a.html)
        now = datetime.now(timezone.utc)
        tr = _track_record_page()
        tr_name = None
        if tr is not None:
            tr_name = "track_record.html"
            path.with_name(tr_name).write_text(
                report.page("MBTgrandaudit track record", tr[0], now,
                            f'<a href="{path.name}">← Predictions</a>'), encoding="utf-8")
            print(f"wrote {path.with_name(tr_name)}")
        for f in report.html_report(by_league, path, now, tr_name, idle, days):
            print(f"wrote {f}")
        if a.open:
            webbrowser.open(path.resolve().as_uri())


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="mbtgrandaudit", description="Football match prediction engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, league_nargs="*"):
        p.add_argument("--league", "-l", nargs=league_nargs, help="league key(s), e.g. EPL LALIGA, or 'all'")
        p.add_argument("--offline", action="store_true", help="use cached data only")
        p.add_argument("--train-seasons", type=int, help="seasons of history to fit on (default 3)")

    def outputs(p):
        p.add_argument("--html", help="write an HTML report to this path")
        p.add_argument("--open", action="store_true", help="open the HTML report in a browser")
        p.add_argument("--json", help="write full predictions as JSON")
        p.add_argument("--csv", help="write a flat CSV of market probabilities")

    sub.add_parser("leagues", help="list supported leagues").set_defaults(fn=cmd_leagues)

    p = sub.add_parser("fixtures", help="list upcoming fixtures")
    common(p)
    p.add_argument("--days", "-d", type=int, default=7)
    p.set_defaults(fn=cmd_fixtures)

    p = sub.add_parser("predict", help="predict upcoming fixtures")
    common(p)
    outputs(p)
    p.add_argument("--days", "-d", type=int, default=7)
    p.add_argument("--team", "-t", help="only fixtures involving this team")
    p.add_argument("--detail", action="store_true", help="show every model's numbers")
    p.add_argument("--log", action="store_true", help="save predictions to the SQLite log for training")
    p.set_defaults(fn=cmd_predict)

    sub.add_parser("settle", help="fill in results for logged predictions").set_defaults(fn=cmd_settle)

    p = sub.add_parser("snapshot-odds", help="record closing odds for logged fixtures kicking off soon")
    p.add_argument("--hours", type=float, default=3.0)
    p.set_defaults(fn=cmd_snapshot_odds)

    p = sub.add_parser("backfill", help="write walk-forward backtest predictions into the log")
    common(p)
    p.add_argument("--seasons", type=int, default=2, help="completed seasons to backfill")
    p.set_defaults(fn=cmd_backfill)

    p = sub.add_parser("evaluate", help="score the prediction log: accuracy, calibration, ROI, CLV")
    p.add_argument("--source", choices=["live", "backtest", "all"], default="live")
    p.add_argument("--league", "-l", nargs=1, help="one league key")
    p.add_argument("--version", help="only this model version")
    p.add_argument("--since", help="only kickoffs on/after this date (YYYY-MM-DD)")
    p.add_argument("--html", help="write a track-record HTML page")
    p.add_argument("--bets-csv", help="write the value-bet ledger as CSV")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("odds", help="enter bookmaker odds by hand (e.g. GG2+) for logged fixtures")
    osub = p.add_subparsers(dest="action", required=True)
    q = osub.add_parser("add", help='e.g. odds add Arsenal Leeds gg2 7.50 --bookmaker Bet365')
    q.add_argument("home", help="part of the home team name")
    q.add_argument("away", help="part of the away team name")
    q.add_argument("market", help="1, X, 2, gg, ng, gg2, ng2, o2.5, u3.5 ...")
    q.add_argument("price", type=float, help="decimal odds")
    q.add_argument("--bookmaker")
    q = osub.add_parser("import", help="CSV with columns home, away, market, odds[, bookmaker]")
    q.add_argument("file")
    q = osub.add_parser("list", help="show entered odds")
    q.add_argument("--days", type=int, default=1, help="include fixtures from this many days ago")
    p.set_defaults(fn=cmd_odds)

    p = sub.add_parser("db", help="show what the prediction log holds, or migrate SQLite to Postgres")
    p.add_argument("--migrate-from", metavar="SQLITE_FILE",
                   help="copy a local SQLite log (e.g. data/predictions.db) into DATABASE_URL")
    p.set_defaults(fn=cmd_db)

    p = sub.add_parser("match", help="predict a single match")
    p.add_argument("league")
    p.add_argument("home")
    p.add_argument("away")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--train-seasons", type=int)
    outputs(p)
    p.set_defaults(fn=cmd_match)

    p = sub.add_parser("ratings", help="show fitted team ratings")
    common(p)
    p.set_defaults(fn=cmd_ratings)

    p = sub.add_parser("backtest", help="walk-forward backtest and ensemble weight tuning")
    common(p)
    p.add_argument("--seasons", type=int, default=1, help="completed seasons to test on")
    p.add_argument("--save-weights", action="store_true", help="store tuned weights for live use")
    p.set_defaults(fn=cmd_backtest)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
