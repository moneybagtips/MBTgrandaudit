"""GG/NG 2+ Flexibet accumulator slips: build them from logged predictions and entered odds,
and settle them against final scores.

Flexibet (SportyBet) pays a reduced price when k or fewer legs lose. Calibrated on real
slips, the price at "k cuts" is close to

    odds_k = c(n) * F / sum_{j<=k} e_j(o_i - 1)

where F is the full accumulator price, e_j the j-th elementary symmetric polynomial of the
legs' (odds - 1) and c(n) a multiplier that grows with the number of legs n (about 1.4 at
20 legs, 2 at 30, 4.3 at 50). The step ratios between cut levels match real slips to ~5%;
c itself is only an estimate, so the slip pages say what the cut price should roughly show.

Two tiers are built:

* aggressive: 50 legs, 11 GG2+ Yes and 39 GG2+ No, 2 cuts, aimed at the max payout;
* middle: 7-9 Yes legs, 1-2 cuts, the cut price aimed at 15,000+, chosen to maximise
  chance x capped payout. The No legs are the model's most confident GG2+ No picks.

A slip's chance of paying is the Poisson-binomial probability of k or fewer losing legs
under the model's probabilities.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# (legs, multiplier) points measured on real SportyBet slips
C_POINTS = [(20, 1.4), (30, 2.0), (50, 4.3)]
MAX_PAYOUT = 1_500_000.0     # GHS, SportyBet
MAX_CUTS = 3                 # what Flexibet allowed on a 50-leg slip


def multiplier(n: int) -> float:
    xs, ys = zip(*C_POINTS)
    return float(np.interp(n, xs, ys))


def esp(values, k: int) -> np.ndarray:
    """Elementary symmetric polynomials e_0..e_k of the values."""
    e = np.zeros(k + 1)
    e[0] = 1.0
    for v in values:
        for j in range(k, 0, -1):
            e[j] += e[j - 1] * v
    return e


def cut_odds(odds, k: int, n: int | None = None) -> float:
    """Estimated Flexibet price when up to k legs may lose (k=0 is the full price)."""
    o = np.asarray(odds, float)
    full = float(np.prod(o))
    if k == 0:
        return full
    return multiplier(n or len(o)) * full / float(esp(o - 1, k).sum())


def loss_dist(probs) -> np.ndarray:
    """P(exactly m legs lose) for m = 0..n, legs independent with these win probabilities."""
    d = np.zeros(len(probs) + 1)
    d[0] = 1.0
    for p in probs:
        nd = d * p
        nd[1:] += d[:-1] * (1 - p)
        d = nd
    return d


@dataclass
class Leg:
    event_id: str
    league: str
    kickoff: str
    match: str
    pick: str          # 'Yes' or 'No' (GG2+)
    prob: float        # model probability the pick wins
    odds: float


@dataclass
class Slip:
    tier: str
    name: str
    cuts: int
    legs: list[Leg] = field(default_factory=list)

    @property
    def full_odds(self) -> float:
        return float(np.prod([l.odds for l in self.legs]))

    @property
    def est_cut_odds(self) -> float:
        return cut_odds([l.odds for l in self.legs], self.cuts)

    @property
    def chance(self) -> float:
        return float(loss_dist([l.prob for l in self.legs])[: self.cuts + 1].sum())


# ---------------------------------------------------------------- the pool

def pool(store, start: datetime, end: datetime, now: datetime) -> pd.DataFrame:
    """One row per logged fixture kicking off in [start, end] after now that has a live
    prediction and entered GG2+ odds on both sides: event_id, league, kickoff, match,
    py, pn (model GG2+ Yes / No), oy, on (latest entered prices)."""
    ev = store.upcoming(max(start, now), end)
    if ev.empty:
        return pd.DataFrame(columns=["event_id", "league", "kickoff", "match", "py", "pn", "oy", "on"])
    mo = store.market_odds_for(ev["event_id"])
    mo = mo[mo["market"].isin(["gg2_yes", "gg2_no"])].sort_values("captured_at")
    last = mo.groupby(["event_id", "market"])["odds"].last().unstack() if not mo.empty else pd.DataFrame()
    rows = []
    for r in ev.drop_duplicates("event_id").itertuples():
        eid = str(r.event_id)
        if eid not in last.index or last.loc[eid].isna().any():
            continue
        pred = store.latest_prediction(eid)
        if pred is None:
            continue
        mk = json.loads(pred["markets"])
        if "gg2_yes" not in mk:
            continue
        rows.append(dict(event_id=eid, league=r.league, kickoff=str(r.kickoff)[:16],
                         match=f"{pred['home']} v {pred['away']}", py=float(mk["gg2_yes"]),
                         pn=float(mk.get("gg2_no", 1 - mk["gg2_yes"])),
                         oy=float(last.loc[eid, "gg2_yes"]), on=float(last.loc[eid, "gg2_no"])))
    return pd.DataFrame(rows)


def _legs(df: pd.DataFrame, yes_ids, no_ids) -> list[Leg]:
    by = df.set_index("event_id")
    out = [Leg(i, by.at[i, "league"], by.at[i, "kickoff"], by.at[i, "match"], "Yes",
               by.at[i, "py"], by.at[i, "oy"]) for i in yes_ids]
    out += [Leg(i, by.at[i, "league"], by.at[i, "kickoff"], by.at[i, "match"], "No",
                by.at[i, "pn"], by.at[i, "on"]) for i in no_ids]
    return sorted(out, key=lambda l: (l.league, l.kickoff, l.match))


# ---------------------------------------------------------------- builders

class _Scorer:
    """Scores a choice of Yes legs: the rest of the slip is the most confident No picks."""

    def __init__(self, df, n_legs, cuts, stake, min_cut, objective):
        self.df, self.n, self.k = df, n_legs, cuts
        self.stake, self.min_cut, self.objective = stake, min_cut, objective
        d = df.set_index("event_id")
        self.py, self.pn, self.oy, self.on = (d[c].to_dict() for c in ("py", "pn", "oy", "on"))
        self.no_order = df.sort_values("pn", ascending=False)["event_id"].tolist()
        self.cache = {}

    def __call__(self, yes_ids):
        key = tuple(sorted(yes_ids))
        if key in self.cache:
            return self.cache[key]
        ys = set(key)
        no_ids = [i for i in self.no_order if i not in ys][: self.n - len(key)]
        out = None
        if len(key) + len(no_ids) >= self.n:
            odds = [self.oy[i] for i in key] + [self.on[i] for i in no_ids]
            od = cut_odds(odds, self.k)
            if od >= self.min_cut:
                p = float(loss_dist([self.py[i] for i in key] + [self.pn[i] for i in no_ids])[: self.k + 1].sum())
                val = p if self.objective == "chance" else p * min(od * self.stake, MAX_PAYOUT)
                out = (val, key, tuple(no_ids))
        self.cache[key] = out
        return out


def _score(df, yes_ids, n_legs, cuts, stake, min_cut, objective):
    s = _Scorer(df, n_legs, cuts, stake, min_cut, objective)(yes_ids)
    return s and (s[0], _legs(df, s[1], s[2]))


def _diverse(df, n_slips, n_legs, n_yes, cuts, stake, min_cut, objective, yes_pool, seed, tries):
    rng = random.Random(seed)
    cands = df.sort_values("py", ascending=False)["event_id"].tolist()[:yes_pool]
    if len(cands) < n_yes:
        return []
    score = _Scorer(df, n_legs, cuts, stake, min_cut, objective)
    for _ in range(tries):
        score(rng.sample(cands, n_yes))
    found = lambda: sorted((v for v in score.cache.values() if v), key=lambda v: -v[0])
    for v, ys, _ in found()[:30]:          # hill-climb the best few
        cur, cur_v, improved = set(ys), v, True
        while improved:
            improved = False
            for a in list(cur):
                for b in cands:
                    if b in cur:
                        continue
                    s = score(cur - {a} | {b})
                    if s and s[0] > cur_v:
                        cur, cur_v, improved = set(s[1]), s[0], True
                        break
                if improved:
                    break
    chosen = []
    for v, ys, nos in found():
        if all(len(set(ys) ^ set(c[0])) >= 4 for c in chosen):
            chosen.append((ys, nos))
        if len(chosen) == n_slips:
            break
    return [_legs(df, ys, nos) for ys, nos in chosen]


def aggressive(df, n_slips=10, n_legs=50, n_yes=11, cuts=2, stake=10.0, seed=7, tries=5000):
    target = MAX_PAYOUT / stake
    got = _diverse(df, n_slips, n_legs, n_yes, cuts, stake, target, "chance", 30, seed, tries)
    return [Slip("aggressive", f"Aggressive slip {i}", cuts, legs) for i, legs in enumerate(got, 1)]


def middle_shape(df, stake=10.0, min_cut=15000.0, yes_range=(7, 8, 9), cut_range=(1, 2),
                 leg_range=range(35, 51)):
    """The (yes legs, total legs, cuts) with the best chance x capped payout, using the top
    Yes and top No picks; None if no shape reaches min_cut."""
    top_y = df.sort_values("py", ascending=False)["event_id"].tolist()
    best = None
    for y in yes_range:
        for n in leg_range:
            if n > len(df):
                break
            for k in cut_range:
                s = _score(df, top_y[:y], n, k, stake, min_cut, "payout")
                if s and (best is None or s[0] > best[0]):
                    best = (s[0], y, n, k)
    return best and best[1:]


def middle(df, n_slips=10, stake=10.0, min_cut=15000.0, seed=11, tries=5000):
    shape = middle_shape(df, stake, min_cut)
    if not shape:
        return []
    y, n, k = shape
    got = _diverse(df, n_slips, n, y, k, stake, min_cut, "payout", 20, seed, tries)
    return [Slip("middle", f"Middle slip {i}", k, legs) for i, legs in enumerate(got, 1)]


# ---------------------------------------------------------------- output

def to_markdown(slips: list[Slip], stake: float, now: datetime, title: str) -> str:
    out = [f"# {title}", "",
           f"Built {now:%d %b %Y %H:%M} UTC from the live prediction log and the latest entered "
           f"SportyBet GG/NG 2+ prices. Stake GHS {stake:g} per slip. Kick-off times are UTC.", "",
           "The cut price is an estimate (it can be a few % off). If SportyBet shows a little less, "
           "swap the weakest No leg for the next one down.", ""]
    for tier, label in (("aggressive", "Aggressive"), ("middle", "Middle tier")):
        ts = [s for s in slips if s.tier == tier]
        if not ts:
            continue
        tot = sum(s.chance for s in ts)
        out += [f"## {label}", "",
                f"{len(ts)} slips. Chance at least one pays: about 1 in {1 / tot:,.0f}." if tot else "", ""]
        for s in ts:
            y = sum(l.pick == "Yes" for l in s.legs)
            pay = min(s.est_cut_odds * stake, MAX_PAYOUT)
            out += [f"### {s.name}: {len(s.legs)} legs ({y} Yes), Flexibet {s.cuts} cut{'s' * (s.cuts != 1)}", "",
                    f"Full odds {s.full_odds:,.0f}; est. {s.cuts}-cut odds {s.est_cut_odds:,.0f} "
                    f"(pays about GHS {pay:,.0f}); chance 1 in {1 / s.chance:,.0f}.", "",
                    "| # | League | Kick-off | Match | Pick | Model | Odds |", "|---|---|---|---|---|---|---|"]
            for i, l in enumerate(s.legs, 1):
                pick = f"**GG2+ {l.pick}**" if l.pick == "Yes" else f"GG2+ {l.pick}"
                out.append(f"| {i} | {l.league} | {l.kickoff[5:16]} | {l.match} | {pick} | {l.prob:.1%} | {l.odds:.2f} |")
            out.append("")
    return "\n".join(out)


def save(slips: list[Slip], path: Path, stake: float, now: datetime):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"built_at": now.isoformat(), "stake": stake,
                                "slips": [dict(asdict(s)) for s in slips]}, indent=1), encoding="utf-8")


def load(path: Path) -> tuple[list[Slip], float]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    slips = [Slip(s["tier"], s["name"], s["cuts"], [Leg(**l) for l in s["legs"]]) for s in d["slips"]]
    return slips, float(d.get("stake", 10))


# ---------------------------------------------------------------- settlement

def leg_result(pick: str, hg, ag) -> str:
    """'won', 'lost' or 'open' (no score yet)."""
    if hg is None or ag is None or (isinstance(hg, float) and math.isnan(hg)):
        return "open"
    yes = hg >= 2 and ag >= 2
    return "won" if (pick == "Yes") == yes else "lost"


def settle(slips: list[Slip], results: dict, stake: float) -> tuple[str, bool]:
    """Markdown report and whether every leg has a result. `results` maps event id to
    (home goals, away goals); a fixture missing from it is still open."""
    lines = ["| Slip | Legs | Cuts allowed | Lost | Open | Result | Lost legs |", "|---|---|---|---|---|---|---|"]
    done = True
    for s in slips:
        lost, open_ = [], 0
        for l in s.legs:
            hg, ag = results.get(l.event_id, (None, None))
            r = leg_result(l.pick, hg, ag)
            if r == "open":
                open_ += 1
            elif r == "lost":
                lost.append(f"{l.match} ({l.pick}, {hg}-{ag})")
        done &= open_ == 0
        if len(lost) > s.cuts:
            res = "Lost"
        elif open_:
            res = "Still alive"
        else:
            # Flexibet pays the price of the cut level set on the slip, however many legs lost
            od = s.est_cut_odds if s.cuts else s.full_odds
            res = f"WON with {len(lost)} lost, about GHS {min(od * stake, MAX_PAYOUT):,.0f}"
        lines.append(f"| {s.name} | {len(s.legs)} | {s.cuts} | {len(lost)} | {open_} | {res} | {'; '.join(lost)} |")
    return "\n".join(lines), done
