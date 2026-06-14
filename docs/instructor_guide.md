# Instructor Guide — FIN 4850: Quantitative Finance Lab

*ConvexPi platform, Fall 2026*

---

## Course Overview

FIN 4850 is a hands-on quantitative finance course built around two platform components:

- **The Lab** — a synthetic equity market where students discover alpha signals, build multi-factor strategies, and submit them to an automated grader
- **The Arena** — a live limit order book where students deploy algorithmic agents and compete in real time

The six missions scaffold students from first principles (overfitting) through professional-level topics (optimal market making, real-data factor research). Each mission is a self-contained Jupyter notebook; students submit strategies through the web platform.

---

## Suggested 14-Week Schedule

| Week | Mission | Topic | Arena session |
|------|---------|-------|---------------|
| 1 | — | Setup + Python review | — |
| 2 | **Mission 1** | Overfitting game | — |
| 3 | Mission 1 continued | Grade reports, IC analysis | — |
| 4 | **Mission 2** | Naive market maker | Open session (ungraded) |
| 5 | Mission 2 continued | Inventory management | Graded season 1 |
| 6 | **Mission 3** | Alpha discovery | — |
| 7 | Mission 3 continued | FDR, walk-forward | — |
| 8 | **Mission 4** | Strategy library | — |
| 9 | Mission 4 continued | IC-weighted composites | Season 1 ends, Season 2 opens |
| 10 | **Mission 5** | Real-data Lab | — |
| 11 | Mission 5 continued | Factor decay, macro regimes | — |
| 12 | **Mission 6** | Advanced Arena agents | Graded Season 2 |
| 13 | Mission 6 continued | Grand tournament | — |
| 14 | — | Final presentations | Season 2 ends |

---

## Pre-Semester Checklist

### 1. Supabase (database + auth)

- [ ] Create a Supabase project at [supabase.com](https://supabase.com)
- [ ] Run `web/supabase/schema.sql` in the SQL editor to create all tables
- [ ] Copy the project URL and `anon` key into `web/.env.local`
- [ ] Copy the `service_role` key for the grader worker

### 2. Vercel (web app)

- [ ] Import the repo into Vercel (or `vercel deploy` from `web/`)
- [ ] Set environment variables: `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_KEY`, `NEXT_PUBLIC_SENTRY_DSN` (optional), `DISCORD_WEBHOOK_URL` (optional)
- [ ] Confirm the app loads at your Vercel URL

### 3. Railway (Arena server + grader worker)

- [ ] Deploy `Dockerfile` (Arena server) — set `ADMIN_TOKEN`, `PORT=8765`
- [ ] Deploy `Dockerfile.grader` (grader worker) — set `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `MARKET_SEED`, `DISCORD_WEBHOOK_URL`
- [ ] Confirm health check: `curl http://railway-url:8766/health`
- [ ] Save the Arena WebSocket URL (`ws://railway-url:8765`) — students need this for Mission 2

### 4. Classroom setup

- [ ] Create a **cohort** on the platform (type: `classroom`, visibility: `private`)
- [ ] Share the cohort invite link with students
- [ ] Test a sample submission from a student account
- [ ] Optionally: create a second cohort (type: `competition`) for graded Arena seasons

### 5. Colab access (student machines)

All mission notebooks have an "Open in Colab" badge. Verify:
- [ ] The badge URLs point to the correct GitHub branch
- [ ] `pip install convexpi` succeeds in a fresh Colab session
- [ ] yfinance is available in Colab (Mission 5 needs it: `pip install yfinance`)

---

## Platform Features for Instructors

### Instructor Dashboard

Navigate to `/dashboard/instructor/[cohort-slug]`. Shows:
- **Queue stats** — pending/running/failed submissions, members with no submission
- **Student progress table** — submission count, best OOS Sharpe, alphas found, last status
- **Submission log** — full history with IS/OOS Sharpe and overfitting ratio per submission

### Arena Season Management

From the instructor dashboard, open the **Seasons** tab to:
- Create a new season (give it a name and optional description)
- End a season — this triggers an automatic Discord embed with the top-3 podium
- View the public season archive at `/seasons`

Suggested schedule:
- **Season 1 (Weeks 4–9):** open, ungraded — students build intuition
- **Season 2 (Weeks 12–14):** graded — counts toward final grade

### Per-Cohort Planted Alpha Config

In the Supabase dashboard, set `cohorts.market_config` (JSONB) to override the default planted alphas:

```json
{
  "planted_alphas": [
    {"feature": "mom_1m",  "strength_bps": 8,  "start_day": 0},
    {"feature": "val_bm",  "strength_bps": 3,  "start_day": 100}
  ]
}
```

The grader worker reads this at runtime. Use stronger alphas early in the semester (easier to discover), then reduce strength in later cohorts to increase difficulty.

---

## Grading Rubric

### Mission assignments (70% of grade)

Each mission has a Jupyter notebook with clearly marked challenge sections. Assess on:

| Criterion | Weight | Description |
|---|---|---|
| Correct implementation | 40% | Code runs without errors; strategy interface matches spec |
| OOS Sharpe > 0 | 20% | Strategy earns positive risk-adjusted return on holdout data |
| Overfitting ratio > 0.7 | 20% | OOS/IS Sharpe ratio — penalizes in-sample overfit |
| Written analysis | 20% | Student explains *why* their approach works (or fails) |

For missions without a grader submission (Missions 2, 6): assess notebook quality, correct use of the Arena API, and completeness of challenge solutions.

### Arena Season (30% of grade)

Final mark-to-market PnL on Season 2 leaderboard, normalized:
- Top 10%: A
- Top 30%: B
- Participated (positive PnL): C
- Negative PnL or no participation: D/F

---

## Common Student Issues

### "My strategy has high IS Sharpe but fails OOS"

This is the intended lesson of Mission 1. Point them to the overfitting ratio in the grade report. Common causes:
- Too many features, too short a training window (p > n problem)
- Grid search on IS data → optimistic selection bias
- Using `val_bm` or `qual_roe` features that are very low-IC on this market's planted alphas

### "The grader returned an error"

Check the **submission log** in the instructor dashboard — errors are printed with full stack traces. Common causes:
- Code uses `import` statements blocked by the sandbox (network, `os.system`, etc.)
- Strategy returns a non-array signal or has the wrong shape
- Runtime exception in `on_day()` method

### "My Arena agent disconnects"

The Arena server requires a ping every 20 seconds. Have students check:
- They're running `websockets >= 11.0` (`pip install -q websockets`)
- Their `on_tick` method doesn't block for longer than ~1 second
- The Railway service hasn't hit its memory limit (check Railway logs)

### "I can't connect to the Arena"

Verify the health check is passing: `curl http://your-railway-url:8766/health`. If it fails, the Arena server may have crashed — check Railway logs and redeploy if necessary.

### "The Anomaly Graveyard / Seasons page isn't loading"

Both pages are server-rendered and read from static JSON files (`public/anomaly-stats.json`). If the file is missing or stale, run:

```bash
python deploy/compute_anomaly_stats.py
```

and commit the updated JSON. The GitHub Actions cron (`0 4 1 * *`) does this monthly automatically.

---

## Extending the Platform

### Adding a new mission

1. Create `missions/mission_07_*/notebook.ipynb` following the existing structure
2. Add an "Open in Colab" badge in the first cell
3. Add the mission to the table in `README.md` and `docs/ROADMAP.md`

### Adding a new strategy to the library

Strategies live in `src/convexpi/lab/strategies.py`. Each strategy is a class with an `on_day()` method:

```python
class MyStrategy(Strategy):
    def on_day(self, day, features, prices, portfolio) -> np.ndarray:
        sig = features.get('mom_12m', np.zeros(len(prices)))
        return _ls_weights(sig)
```

Register it in the `STRATEGIES` dict at the bottom of the file. Add tests to `tests/lab/test_strategies.py`.

### Adding a new Arena agent

Agents live in `src/convexpi/arena/agents.py`. Extend `Agent` and override `on_tick()`:

```python
class MyAgent(Agent):
    def on_tick(self, state: MarketState) -> list[Order]:
        ...
```

Export from `src/convexpi/arena/__init__.py` and add tests to `tests/arena/test_agents.py`.

### Adjusting the grader's planted alphas globally

Edit `src/convexpi/lab/synth.py` — the `PlantedAlpha` dataclass and the default alpha list in `SyntheticMarket.__init__`. Set `MARKET_SEED` on the grader worker Railway service to change the random seed across all submissions.

---

## Semester Debrief Recommendations

After the final Arena season ends:

1. **Show the leaderboard trajectory** — pull `arena_rankings` from Supabase and plot each student's PnL over the season
2. **Reveal the planted alphas** — show which features had the highest correlation with returns, and compare to what students actually discovered
3. **Run the Anomaly Graveyard analysis live** — `python deploy/compute_anomaly_stats.py` in class to show how real-world factor Sharpes decay post-publication
4. **Compare Arena winners to Lab winners** — often different students excel at different tasks. Discuss why market-making skill and alpha-discovery skill may be independent

---

## Contact and Support

- **GitHub Issues:** [github.com/convexpi/lab/issues](https://github.com/convexpi/lab/issues)
- **Platform docs:** `docs/` directory — quickstart, deployment guide, architecture
- **Test suite:** `pytest tests/ -v` (316 tests) and `cd web && npx vitest run` (93 tests)
