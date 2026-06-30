# Mission 1: The Overfitting Game

**Notebooks** — pick your language; the grader scores all three identically:

| Language | Notebook |
|---|---|
| Python | `missions/mission_01_overfitting/notebook.ipynb` |
| R | `missions/mission_01_overfitting/notebook_r.ipynb` |
| Julia | `missions/mission_01_overfitting/notebook_julia.ipynb` |

## Learning objectives

- Understand the difference between in-sample and out-of-sample performance
- Experience the grid-search trap firsthand
- Interpret a ConvexPi grade report
- Build intuition for why the overfitting ratio matters more than IS Sharpe

## The trap

Students start with a simple strategy, then are invited to grid-search hyperparameters to maximize IS Sharpe. The IS Sharpe climbs. The OOS Sharpe does not. The grade report shows an overfitting ratio well below 0.70.

This is the lesson: **the grader is immune to in-sample optimization**. The hidden holdout data was never touched.

## Parts

| Part | Task |
|---|---|
| 1. Setup | Install `convexpi`, load synthetic market |
| 2. Explore | Plot price paths, return distributions, feature correlation matrix |
| 3. Baseline | Single-feature z-score strategy, submit to grader |
| 4. IC analysis | Compute Spearman rank IC per feature, identify which are informative |
| 5. Grid-search trap | Exhaustively tune lookback, threshold, and feature selection |
| 6. Submit | Compare IS vs OOS Sharpe; interpret overfitting_ratio |
| 7. Challenges | Fix the overfit strategy; reach OOS Sharpe > 0.5 with ratio > 0.70 |

## Key metric: overfitting ratio

```
overfitting_ratio = OOS Sharpe / IS Sharpe
```

A ratio above 0.70 means your strategy generalizes well. Below 0.30 means you fit the in-sample noise. The grade uses OOS Sharpe as the primary score.

## Grade report fields

| Field | Meaning |
|---|---|
| `is_sharpe` | Annualized Sharpe on training data (untrusted) |
| `oos_sharpe` | Annualized Sharpe on hidden holdout (the real grade) |
| `overfitting_ratio` | `oos_sharpe / is_sharpe` — target > 0.70 |
| `alphas_discovered` | How many planted signals your strategy loaded onto |
| `total_alphas` | Total planted signals in this cohort's market |
