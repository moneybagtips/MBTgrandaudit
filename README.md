# MBTgrandaudit

A football (soccer) prediction engine. It pulls upcoming fixtures for major
professional leagues, analyses each match (recent form in all competitions and
head-to-head history), and prices four betting markets with an ensemble of nine
statistical models:

| Market | Meaning |
|---|---|
| **1X2** | Home win / draw / away win |
| **GG / BTTS** | Both teams score at least once |
| **GG2+** | Both teams score two or more |
| **O/U** | Total goals over/under 0.5, 1.5, 2.5, 3.5, 4.5 |

Every model produces a full **scoreline probability matrix** P(home = i, away = j).
All markets come from that one matrix, so 1X2, BTTS, GG2+ and O/U never
contradict each other.

## Quick start

```bash
pip install -r requirements.txt
python -m mbtgrandaudit predict                      # top-5 leagues, next 7 days
python -m mbtgrandaudit predict -l EPL -d 3 --detail # one league, per-model breakdown
python -m mbtgrandaudit predict -l all --html reports/week.html --open
python -m mbtgrandaudit match EPL "Arsenal" "Chelsea"
python -m mbtgrandaudit fixtures -l SERIEA -d 10
python -m mbtgrandaudit ratings -l BUNDES
python -m mbtgrandaudit backtest -l EPL --seasons 2 --save-weights
python -m mbtgrandaudit leagues
```

Outputs: console summary, `--html` (a self-contained report), `--json` (everything,
including per-model numbers and match analysis) and `--csv` (flat market probabilities).
Use `--offline` to run from cached data only.

## Leagues

EPL, Championship, La Liga, Bundesliga, Serie A, Ligue 1, Eredivisie, Primeira Liga,
Belgian Pro League (Jupiler), Süper Lig, Scottish Premiership, Greek Super League,
Swiss Super League, MLS, Danish Superliga, Norwegian Eliteserien, Japanese J1 League
(`python -m mbtgrandaudit leagues`). With no `--league`, it runs the big five.

The last five come from football-data.co.uk's "extra leagues" files, which differ:
* **Results and closing odds only.** No shots, xG or pre-match odds, so `xg_poisson`
  falls back to goals. In backtests the market view is left out (closing odds would
  leak future information) and the bookmaker benchmark uses closing odds.
  Live predictions still get the market view from ESPN's pre-match odds.
* **No second tier**, so promoted teams start from the below-average newcomer prior.
* **Seasons:** MLS and Eliteserien run on calendar years. Switzerland, Denmark and
  (from 2026-27) Japan run August to May.
* **Switzerland isn't covered by ESPN** for the current season. Fixtures (with 1X2 odds),
  closing odds and results all come from football-data. It publishes fixtures only a few
  days ahead, so Swiss games appear in the predictions close to each matchday, and
  without all-competition form.

## Data sources (free, no API key)

* **football-data.co.uk**: league results for the last 3 seasons (10 for
  head-to-head), plus shots, xG (where published) and average bookmaker odds.
  Second-tier results are loaded too, so promoted teams arrive with a rating.
* **ESPN public API**: fixtures, bookmaker odds, and each team's
  **all-competition** schedule (cups, European games), used for form and
  head-to-head.

Responses are cached in `.cache/`. Past seasons are cached forever; current data refreshes every 0.5–6 hours.

## The models

| Model | Idea |
|---|---|
| `dixon_coles` | Attack/defence Poisson with the Dixon-Coles low-score correction (ρ), time-decayed MLE with analytic gradients |
| `poisson` | Maher's independent Poisson attack/defence model |
| `bivariate_poisson` | Karlis-Ntzoufras bivariate Poisson; fitted covariance term for correlated scoring |
| `negative_binomial` | Same structure with over-dispersed (NB2) goal counts; dispersion fitted by MLE |
| `xg_poisson` | Strength model fitted on 0.65·xG + 0.35·goals. Uses published xG where available, otherwise a shots-based xG proxy |
| `elo` | Margin-weighted Elo, with rating gaps mapped to goal rates by Poisson regression |
| `pi_rating` | Constantinou-Fenton pi-ratings (separate home/away ratings in goal units), mapped to goals |
| `form` | Recency-weighted, venue-adjusted goals for/against over the last 10 games in **all competitions** |
| `h2h` | Head-to-head meetings (league + cups), venue-adjusted and Bayesian-shrunk toward Dixon-Coles |
| `market_view` | Not a statistical model. A score matrix solved from the game's own 1X2 + O/U odds (see below). Used only when the game has odds |

All strength models use exponential time decay (half-life about a year), a ridge prior
on team ratings, and per-division intercepts. Teams new to a league with no
lower-tier history start below average.

The **ensemble** is a weighted mixture of the score matrices: the nine models plus the
market view when the game has odds. Without odds, the market view's weight drops out
and the models carry the prediction alone. One weight per component, not per market,
keeps every market consistent with the others, because they all come from one matrix.

Weights come from `ensemble_weights.json`, written by `backtest --save-weights`. They
minimise combined log loss over 1X2, BTTS, GG2+ and O/U 1.5/2.5/3.5, blended 70/30 with
the defaults to limit overfitting. The backtest builds the market view from odds
available **before** kick-off, never closing odds. Tuned market-view share: EPL 56%,
Bundesliga 74%, Serie A 82%, La Liga and Ligue 1 86%. Leagues without tuned weights use
the defaults in `config.py` (55%).

## Markets without bookmaker odds (GG2+, BTTS, other O/U lines)

Bookmakers quote 1X2 and one Over/Under line for almost every game, but GG2+ rarely.
The tool fills the gap three ways:

1. **Market view.** It removes the margin from the quoted 1X2 + O/U odds, then solves
   for the home/away expected goals of a Dixon-Coles score matrix that reproduces them.
   That matrix gives the market's view of every other market. The reports show it as
   `mkt` next to each model probability, `evaluate` uses it as a benchmark, and the
   ensemble includes it as a component. The margin is removed with the **power method**,
   which takes more off long shots than proportional scaling. On the backfill,
   proportional scaling put long shots (odds of 5+) at 13.8% and the power method at
   12.7%; they actually won 12.1%.

   **Entered prices are matched too.** If you enter BTTS, GG2+ or extra O/U prices for a
   fixture (see [Entering odds](#entering-odds-gg2-btts-and-other-markets)), the market view
   is fitted to them as well as to 1X2 + O/U. That's the most direct market information on
   GG2+ available.
   * A two-sided pair (e.g. GG2+ Yes **and** No) has its margin removed on its own.
   * A single side has the margin removed with the exponent of the game's 1X2 book.
   * The reports list what was matched, e.g. "implied from 1X2 + O/U 2.5 + GG2+ odds".
2. **Entering odds by hand** for logged fixtures. Prices are stored in the log; the next
   `predict` shows the edge, and `evaluate` counts the bets. See
   [Entering odds](#entering-odds-gg2-btts-and-other-markets) below for the three ways to do it.
3. **Per-market thresholds** in `config.py`. A price counts as VALUE only if the model's edge
   clears the threshold for that market (`min_edge`: 1X2 3%, O/U 3%, BTTS 5%, GG2+ 10%)
   **and** the statistical models (without the market view) are within 10 points of the
   market view (`max_disagreement`). Bigger gaps are flagged *check*: they are usually
   information the models lack (injuries, line-ups), not value. Because the ensemble
   now leans on the market, edges are smaller and VALUE flags rarer.

Backfill findings (3,504 games):
* GG2+, models only vs the market view from prediction-time odds: level (no
  significant difference).
* Against the closing-odds market view, the market is better on every market.

## Entering odds (GG2+, BTTS and other markets)

ESPN supplies 1X2 and one Over/Under line. For anything else, e.g. GG2+ from your
bookmaker, enter the prices yourself in one of three ways. Prices only count for
fixtures the daily run has logged, and only if entered before kick-off.

### 1. The odds form on the Pages site (no terminal; works on a phone)

1. Open **Enter odds** (`odds.html`), linked in the header of every page and under the
   Leagues section. It lists upcoming fixtures by league, with boxes for GG2+ Yes/No and
   BTTS Yes/No.
2. Type decimal odds into any boxes. Each box shows the model's probability and fair odds,
   and as you type it shows the edge, green with **VALUE** if the price clears that market's
   threshold. You can also fill in the bookmaker name (optional).
3. Press **Open GitHub issue**. A pre-filled issue titled `Odds: …` opens in a new tab, with
   one line per price (`event id | market | odds | match`). Press **Create**. You must be
   signed in to GitHub as the repository owner.
4. The **Odds entry** workflow (`.github/workflows/odds-entry.yml`) then:
   * saves the prices to the database,
   * replies on the issue with each price's model probability, fair odds, edge and verdict,
     and lists any lines it skipped (unknown fixture, already kicked off, bad odds),
   * closes the issue,
   * re-runs **Predict**, so the pages show the new edges a few minutes later.

Things to know:
* **Only issues opened by the repository owner are processed.** The repo is public, so
  anyone can open an issue; the workflow ignores everyone else's.
* **The prices you enter are public**, in the issue and its reply.
* **Each submission uses about a minute of Actions time.** That's free on a public repository.
* **If the page doesn't know its GitHub repository** (e.g. a copy opened from disk without a
  git remote), use **Show as CSV** and option 3 instead.

### 2. One price at a time (terminal)

```bash
python -m mbtgrandaudit odds add Arsenal Leeds gg2 7.50 --bookmaker Bet365
```

Team names only need to contain part of the real name. If a name matches more than one
upcoming fixture, the command lists them so you can be more specific.

### 3. Many prices from a CSV file (terminal)

```bash
python -m mbtgrandaudit odds import my_prices.csv
```

`my_prices.csv` needs the columns `home, away, market, odds`, and optionally `bookmaker`.
There is one line per price; the form's **Show as CSV** button produces exactly this
format:

```
home,away,market,odds,bookmaker
Everton,Chelsea,gg2,6.50,Bet365
Brentford,Liverpool,ng2,1.12,Bet365
```

Check what's stored with `python -m mbtgrandaudit odds list`.

**Market codes:**
* `1`, `X`, `2`: match result
* `gg` / `ng`: BTTS yes / no
* `gg2` / `ng2`: GG2+ yes / no
* `o2.5` / `u3.5` etc.: over / under, half-goal lines only

The full key names (`gg2_yes`, `btts_no`, `over_2.5` …) work too.

Commands 2 and 3 write to whichever database is configured. With `DATABASE_URL` in your
`.env`, that's Supabase.

**Entered prices also move the prediction.** A BTTS, GG2+ or O/U price you enter is
treated as market information: it feeds the market view, and so the ensemble. As a
result, the edge shown against that same price is smaller and more cautious than the
statistical models alone would give.

For example, Tottenham v Coventry: GG2+ was 15.9% before a GG2+ Yes price of 3.20 was
entered, and 18.6% after. A price far from the models still shows up through the
**check** flag, which compares the models alone with the market view.

## Reading the output

* **Fair odds** = 1 / probability. A bet has value only if the bookmaker pays more than this.
* **Edge** = ensemble probability × bookmaker odds − 1, wherever there is a price
  (ESPN, or odds you entered).
* **mkt** = the market's view of that market, solved from the game's 1X2 + O/U odds.
* **Confidence** depends on the probability and on how much the nine models agree.
  If they disagree a lot (std > 8 pts), the confidence drops a tier.

## Backtest (walk-forward, weekly refit)

Out-of-sample test: weights tuned on 2024-25 only, then scored on 2025-26. The figures
are log loss averaged over the big five leagues; lower is better.

| Blend | 1X2 | BTTS | GG2+ | O/U 1.5 | O/U 2.5 | O/U 3.5 |
|---|---|---|---|---|---|---|
| Statistical models only | 0.995 | 0.689 | 0.426 | 0.539 | 0.682 | 0.599 |
| Models + market view (tuned) | 0.980 | 0.684 | 0.425 | 0.533 | 0.676 | 0.595 |
| Market view only | 0.978 | 0.683 | 0.426 | 0.531 | 0.675 | 0.595 |

Adding the market view is a clear improvement. But the statistical models add almost
nothing on top of it: the market view alone is about as good. The closing price is
better still. No public-data model reliably beats the closing market, and this one
doesn't either.

What the models do give you:
* probabilities for games without odds
* the market's view priced into markets nobody quotes (GG2+)
* a second opinion. Where the models and the market disagree, the backtest says the
  market is usually right, unless you know something it doesn't.

In backtests, the `form` model only sees league games; live, it uses all competitions.

## Prediction log and track record

Predictions can be stored in a log that builds a training and evaluation dataset
over time. If `DATABASE_URL` is set (in the environment or a git-ignored `.env`
file), the log is a Postgres database such as **Supabase**. Otherwise it is a local
SQLite file (`data/predictions.db`). To move an existing SQLite log over, run
`python -m mbtgrandaudit db --migrate-from data/predictions.db`.

To run everything on a schedule without this PC (GitHub Actions + Supabase), see
**[DEPLOY.md](DEPLOY.md)**.

```bash
python -m mbtgrandaudit predict -l all --log    # store predictions (repeat daily; latest pre-kick-off one counts)
python -m mbtgrandaudit snapshot-odds           # run ~hourly on match days: closing odds for games in the next 3h
python -m mbtgrandaudit settle                  # fill in final scores for finished games
python -m mbtgrandaudit backfill --seasons 2    # add walk-forward predictions for completed seasons
python -m mbtgrandaudit evaluate --source live --html reports/track_record.html --bets-csv reports/bets.csv
python -m mbtgrandaudit db                      # what the log holds
```

| Table | Contents |
|---|---|
| `predictions` | one row per fixture × prediction time. Ensemble scoreline probabilities (compressed), market probabilities, weights, **features frozen at prediction time** (ratings, form, rest days, head-to-head), model version, `source` = `live` or `backtest` |
| `model_outputs` | each model's scoreline probabilities for that prediction |
| `odds` | bookmaker odds at prediction time and closing snapshots |
| `results` | final scores |

Storing whole score matrices means any market, including ones added later, can be
recomputed exactly. `model_version` is a hash of the model code and settings, so
predictions from different versions never get mixed up. Filter on it with
`evaluate --version`.

`evaluate` reports for each market and model:
* log loss, Brier score, RPS and accuracy, against the bookmaker baseline
* calibration (predicted vs. actual by probability bin)
* betting value: flat-stake ROI on picks with an edge of 3% or more, and **closing line
  value** (CLV, whether you beat the closing price), the earliest reliable sign of real edge
* **GG2+ by bookmaker O/U line**: GG2+ accuracy split by the main Over/Under line quoted
  before kick-off. Bookmakers move it to 3.5 for games they expect to be open. For each
  line it compares a market view fitted to that line, one built from 1X2 odds alone, and
  the ensemble.

  On the first 87 live O/U 3.5 games, the market view beat the ensemble: log loss 0.536 vs
  0.543. If that holds over a few hundred games, the ensemble should lean more on the
  market view when the line is 3.5.

  In a backtest across 17 leagues, league corrections, score spread, team GG2+ rates and
  shots data didn't improve GG2+ (11,063 games); this odds signal is the one lead.

When you pass `--html`, `predict` adds a track-record card to the report: live results
once any are settled, the backtest until then.

Notes:
* Backfilled rows use league-only form, and the ensemble weights were tuned on those
  same seasons. Treat `backtest` numbers as optimistic and `live` as the truth.
* `snapshot-odds` and `settle` need to run on a schedule, e.g. Windows Task Scheduler.

## Fixing team names

ESPN and football-data name teams differently ("Nottingham Forest" vs "Nott'm Forest").
Matching uses an alias table plus fuzzy matching. If a prediction warns
`No history found for 'X'`, add the mapping to `team_aliases.json`:

```json
{ "ESPN Name": "football-data name" }
```

## Tests

```bash
python -m pytest tests
```

*For research and entertainment. Bet responsibly.*
