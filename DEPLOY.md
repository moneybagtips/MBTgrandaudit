# Running MBTgrandaudit in the cloud (GitHub Actions + Supabase)

GitHub Actions runs the jobs on a schedule. Supabase (hosted Postgres) holds the
prediction log. Once set up, nothing runs on your PC.

| Workflow | Schedule (UTC) | Does |
|---|---|---|
| `Predict` | daily 06:00 | predicts the next 7 days for every league, logs to Supabase, builds the reports |
| `Closing odds` | every 30 min, 09:00–21:30 | records odds for logged games kicking off within the hour |
| `Settle results` | every 4 h | fills in final scores |
| `Re-tune ensemble weights` | monthly | backtests the big five and commits new `ensemble_weights.json` |
| `Tests` | on every push | runs the test suite against SQLite and a throwaway Postgres |
| `Keepalive` | monthly | re-enables all scheduled workflows (see "Public repository" below) |
| `Odds entry` | when you open an `Odds: …` issue | saves prices from the odds form, replies with the edges, closes the issue and re-runs Predict (owner's issues only; see README "Entering odds") |

All of them can also be started by hand: **Actions** tab, pick the workflow, **Run workflow**.

## 1. Create the Supabase database

1. Create a free project at [supabase.com](https://supabase.com). Pick a region near you
   and save the database password.
2. Open **Connect** (top of the project dashboard) and copy the **Session pooler**
   connection string. It looks like
   `postgresql://postgres.<project-ref>:[YOUR-PASSWORD]@aws-0-<region>.pooler.supabase.com:5432/postgres`.
   Put your password in place of `[YOUR-PASSWORD]`.
   * Use the *session pooler*, not the "direct connection". The direct host is IPv6-only,
     and GitHub's runners only have IPv4.

The tool creates its tables itself on first connect, in a schema called `mbtgrandaudit`.
Supabase's auto-generated REST API only exposes `public` by default, so the
predictions are not reachable with the project's public API keys. You can browse
them in **Table Editor** by switching the schema selector to `mbtgrandaudit`.

## 2. Copy your existing log into Supabase (from this PC, once)

Create a file called `.env` in the project folder. It is git-ignored, so it never
gets committed.

```
DATABASE_URL=postgresql://postgres.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres
```

Then:

```bash
pip install -r requirements.txt
```

```bash
python -m mbtgrandaudit db --migrate-from data/predictions.db
```

```bash
python -m mbtgrandaudit db
```

The last command should print `postgres: ...` with the same counts as your local
log. From now on, local commands use Supabase too while `.env` exists. Delete or
rename `.env` to go back to the local SQLite file.

## 3. Put the code on GitHub

Create a repository named **MBTgrandaudit** and push this folder, using either GitHub
Desktop or the CLI. Public is recommended: Actions minutes and Pages are free and
unlimited. Read "Public repository" below for what becomes visible. For a private
repo, use `--private` instead.

```bash
gh auth login
```

`gh repo create --push` needs at least one commit, so commit the project first. The
`.gitignore` already keeps `.env`, `data/`, `.cache/` and `reports/` out of the repo.

```bash
git init -b main
```

```bash
git add -A
```

```bash
git commit -m "Initial commit"
```

Check that `git ls-files` lists no `.env` or `.db` file, then create the repo and push:

```bash
gh repo create MBTgrandaudit --public --source . --push
```

## 4. Give the workflows the database URL

**Settings → Secrets and variables → Actions → New repository secret**:
name `DATABASE_URL`, value = the same connection string. Or:

```bash
gh secret set DATABASE_URL
```

## 5. (Optional) Publish the reports as a web page

Free on a public repository. A private one needs a paid plan (GitHub Pro or Team).

1. **Settings → Pages → Build and deployment → Source: GitHub Actions**
2. Add a repository *variable* (not a secret) `PUBLISH_PAGES` = `true`:

```bash
gh variable set PUBLISH_PAGES --body true
```

The predictions page (`index.html`) and `track_record.html` then appear at
`https://<user>.github.io/MBTgrandaudit/`.

Without Pages, every `Predict` run still attaches the reports as a downloadable
**reports** artifact (kept 14 days) on the run's summary page.

## 6. First run

**Actions → Predict → Run workflow.** It takes about 5–8 minutes. Then check
**Closing odds** and **Settle results** by running them once by hand.

## Public repository: what is and isn't visible

| Private | Public |
|---|---|
| The Supabase database and its contents | Code, `ensemble_weights.json`, `team_aliases.json` |
| The `DATABASE_URL` secret. It is never shown, pull requests from forks don't receive it, and logs print only `postgres (DATABASE_URL)` | Workflow logs, which include every prediction printed by `Predict` |
| Your local `.env`, `data/`, `.cache/`, `reports/` (git-ignored) | Run artifacts (any signed-in GitHub user) and the Pages site, if enabled |

In short: the tips are public, the data and credentials are not.

Also:
* Keep **Settings → Actions → General → "Require approval for first-time contributors"**
  on (the default). That stops a stranger's pull request from running workflows unreviewed.
* GitHub turns off scheduled workflows in public repos after 60 days without repository
  activity. The `Keepalive` workflow re-enables them all monthly through the API. If a
  schedule ever stops, **Actions → (workflow) → Enable workflow** turns it back on.

## Costs and limits

* **GitHub Actions:** public repositories are free and unlimited, so you can make the
  odds snapshots more frequent (e.g. `*/15 9-21 * * *`). Private ones get
  2,000 free minutes a month, and this setup uses roughly 1,100–1,300 (odds snapshots
  ≈ 780, settling ≈ 180, daily predictions ≈ 200, re-tune ≈ 10). To cut usage, change
  the `Closing odds` cron in `.github/workflows/odds.yml` to every hour (`0 9-21 * * *`).
* **Scheduled runs can start late:** GitHub often delays them by 5–20 minutes and
  occasionally skips one. The tool always uses the latest snapshot taken before
  kick-off, so a late or missed run only makes that closing line a little less fresh.
* **Supabase free tier:** 500 MB database. The current log is about 36 MB, and live
  logging adds roughly 1–2 MB a week. Free projects pause after a week with no
  activity, but the 30-minute odds job keeps it active.
* **Security:** the connection string gives full database access. Keep it only in
  the GitHub secret and your local `.env`. Never commit it.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `DATABASE_URL secret is not set` | Step 4 |
| `connection ... timeout` / `Network is unreachable` | You used the direct-connection URL. Switch to the **session pooler** URL |
| `password authentication failed` | Wrong password in the URL. Reset it under Project Settings → Database |
| `No history found for 'X'` in the logs | New team-name mismatch. Add it to `team_aliases.json` and push |
| ESPN requests fail only on GitHub | ESPN is blocking cloud IPs. The tool falls back to football-data fixtures, but odds and all-competition form will be missing |
