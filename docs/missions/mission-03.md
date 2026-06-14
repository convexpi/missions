# Mission 3: Alpha Discovery

**Notebook:** `missions/mission_03_alpha_discovery/notebook.ipynb`

## Learning objectives

- Frame signal discovery as a multiple-testing problem
- Compute and interpret the Information Coefficient (IC)
- Apply Benjamini-Hochberg FDR correction
- Use walk-forward validation to avoid IS IC overfitting
- Build a composite signal from validated features

## The problem

The synthetic market contains planted alpha signals of known strength, hidden among noise features. Students must identify which features are real alphas using statistical tests — without peeking at the grader's ground truth. The challenge: with many features, naive significance testing produces false positives.

## Parts

| Part | Task |
|---|---|
| 1. Setup | Load market data, understand feature structure |
| 2. IC analysis | Compute daily Spearman rank IC per feature |
| 3. Multiple testing | t-test on mean IC; apply BH FDR at α=0.05 |
| 4. Walk-forward | 120-day train / 20-day OOS windows; measure OOS IC-IR |
| 5. Signal decay | IC at each lag 1–10 to estimate halflife |
| 6. Composite | Weight features by walk-forward IC-IR, submit to grader |

## Information Coefficient

IC is the Spearman rank correlation between your signal and next-day returns, computed cross-sectionally:

```python
from scipy.stats import spearmanr

def compute_daily_ic(signal_day, returns_day):
    corr, _ = spearmanr(signal_day, returns_day)
    return corr
```

A feature with mean IC > 0.03 and IC-IR (mean/std) > 0.5 is worth including.

## Multiple testing: BH FDR

With 10+ features tested, you expect false positives at α=0.05. BH FDR controls the expected fraction of false discoveries:

```python
from statsmodels.stats.multitest import multipletests

reject, pvals_adj, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
```

## Walk-forward IC

Don't use IS IC to select features — it's as vulnerable to overfitting as IS Sharpe. Use OOS IC from a rolling walk-forward:

```python
# 120-day train, step 20 days, measure IC on next 20 days
for t in range(120, n_days - 20, 20):
    train_ics = compute_daily_ics(features[:t], returns[:t])
    oos_ics   = compute_daily_ics(features[t:t+20], returns[t:t+20])
```

Weight your composite signal by each feature's OOS IC-IR.
