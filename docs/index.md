# ConvexPi

**An open-source teaching platform for AI and ML in quantitative finance.**

ConvexPi gives students a place to run real trading strategies against a realistic exchange, discover alpha signals with known ground truth, and compare results on a live leaderboard — all without touching real money or real data licenses.

---

## Two surfaces, one platform

### Arena

A discrete-time limit-order-book exchange simulator. Students write Python agents, connect over WebSocket, and compete against a background population of noise traders, market makers, momentum traders, and an informed trader with private fundamental information.

Instructors can trigger volatility shocks mid-session to test student risk management. Risk limits and survival scoring are enforced automatically.

### Lab

A daily-data backtesting harness built around a synthetic equity panel. The panel generator plants alpha signals of known strength — so the grader can verify whether a student found a real signal or just fit noise. A hidden holdout set is used for the final OOS grade, preventing in-sample overfitting from hiding.

---

## Why synthetic data?

Real data has a fundamental problem for education: you can't verify whether a student found a real signal or mined historical noise. With planted alphas of known strength, the grader has ground truth. No real-data platform can do this.

The `convexpi` package is deterministic given a seed — every student in a cohort runs against the exact same hidden market.

---

## Key design choices

| Decision | Reason |
|---|---|
| Discrete ticks, batch matching, fair shuffle | No latency racing — signal quality wins, not speed |
| Integer cent prices | No float-comparison bugs; itself a teachable lesson |
| OOS grading is the only trusted score | IS Sharpe is self-reported and untrusted |
| Planted alphas with known ground truth | Grader can verify discovery vs noise fitting |
| Per-cohort alpha config | Instructors customize without code changes |
| Buggy agents can't crash the market | Exceptions caught per-agent, per-tick |

---

## Get started

- [Quickstart](quickstart.md) — install, run the Arena demo, grade a strategy
- [Launch Checklist](LAUNCH.md) — PyPI publish, GitHub org, seed script, Vercel deploy
- [Instructor Guide](instructor_guide.md) — week-by-week schedule, grading rubric, deployment checklist
- [Mission 1](missions/mission-01.md) — the overfitting game
- [Architecture](platform-architecture.md) — how everything fits together
- [Roadmap](ROADMAP.md) — phased backlog and feature status
