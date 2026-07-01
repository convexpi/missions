#!/usr/bin/env python3
"""Builds missions/mission_01_overfitting/notebook_r.ipynb — the faithful R port."""
import json, os

def md(s):   return {"cell_type": "markdown", "metadata": {}, "source": _lines(s)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(s)}
def _lines(s):
    s = s.strip("\n")
    parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]

cells = []

cells.append(md(r"""
# Mission 1: The Overfitting Game — R edition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/convexpi/missions/blob/main/missions/mission_01_overfitting/notebook_r.ipynb)

**Learning objectives**
- Build intuition for in-sample vs. out-of-sample performance
- Experience the overfitting trap firsthand
- Submit an **R** strategy and interpret your OOS grade report
- Understand the Information Coefficient (IC) and its role in strategy evaluation

> This is the R port of Mission 1. You write your strategy in R, and the grader runs your R
> `on_day()` over a hidden holdout and scores it with the **same engine** as Python and Julia —
> so the same idea earns the same OOS Sharpe in any language.

---

## Background

A **strategy** is a rule that ranks stocks and bets on the ranking being predictive of future
returns. The market gives you historical data to *develop* on (in-sample, IS), but you only profit
if the strategy works on *new* data it has never seen (out-of-sample, OOS).

**Overfitting** is tuning your strategy to historical noise: it looks brilliant in-sample and falls
apart out-of-sample. This mission makes that gap visible, then shows the cure.
"""))

cells.append(md("## Part 0: Setup"))

cells.append(code(r"""
# Install the convexpi R package (market data + one-call submit) from GitHub.
if (!requireNamespace("convexpi", quietly = TRUE)) {
  if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
  remotes::install_github("convexpi/convexpi-r", upgrade = "never")
}
library(convexpi)

# The market comes from the Python engine (convexpi-lab) via reticulate — make sure it's there.
if (!reticulate::py_module_available("convexpi.lab")) {
  reticulate::py_install("convexpi-lab", pip = TRUE)
}
set.seed(42)
cat("Ready.\n")
"""))

cells.append(md(r"""
---
## Part 1: Explore the Synthetic Market

The ConvexPi Lab uses a **synthetic market** — a simulated panel of stock prices and features.
`synthetic_market("train")` returns the exact in-sample panel the grader fits on (deterministic from
the seed). The hidden holdout ("test") is what you are scored on — you never see its returns; the
grader does.
"""))

cells.append(code(r"""
m <- synthetic_market("train")        # the exact market the grader uses
prices   <- m$prices                  # days x stocks matrix
features <- m$features                # named list of days x stocks matrices
feat_names <- names(features)

cat("prices :", nrow(prices), "days x", ncol(prices), "stocks\n")
cat("features:", paste(feat_names, collapse = ", "), "\n")
"""))

cells.append(code(r"""
# Daily cross-sectional returns from prices: ret[t,] = prices[t+1,]/prices[t,] - 1
ret <- prices[-1, ] / prices[-nrow(prices), ] - 1   # (days-1) x stocks
cat("returns:", nrow(ret), "days x", ncol(ret), "stocks\n")
cat("mean daily return:", sprintf("%.5f", mean(ret)), "\n")
"""))

cells.append(code(r"""
# Visualise the market's cumulative return (equal-weighted, in-sample)
mkt <- rowMeans(ret)
cum <- cumprod(1 + mkt) - 1
plot(cum * 100, type = "l", col = "steelblue", lwd = 1.5,
     main = "Market cumulative return (equal-weighted, IS)",
     xlab = "Trading day", ylab = "Return (%)")
abline(h = 0, col = "grey", lwd = 0.5)
"""))

cells.append(md(r"""
---
## Part 2: A Baseline — what does noise look like?

A strategy here returns a vector of **target weights** (one per stock) each day. We will build a
small backtester so you can *feel* the overfitting trap before submitting. Start with a random
strategy — pure noise, no alpha.
"""))

cells.append(code(r"""
# A simple long/short backtester: go long the top_k by score, short the bottom_k, equal weight.
# signal_fn(t) returns a score vector (one per stock) for day t.
simple_backtest <- function(signal_fn, top_k = 20, cost_bps = 5) {
  n <- ncol(prices); prev_w <- rep(0, n); daily <- numeric(nrow(ret))
  for (t in seq_len(nrow(ret))) {
    s <- signal_fn(t)
    ord <- order(s)                       # ascending
    w <- rep(0, n)
    w[tail(ord, top_k)] <-  1 / top_k     # long the highest scores
    w[head(ord, top_k)] <- -1 / top_k     # short the lowest scores
    cost <- cost_bps / 1e4 * sum(abs(w - prev_w))
    daily[t] <- sum(w * ret[t, ]) - cost
    prev_w <- w
  }
  list(sharpe = mean(daily) / (sd(daily) + 1e-9) * sqrt(252),
       ann_ret = mean(daily) * 252, daily = daily)
}

rng <- function(seed) { set.seed(seed); function(t) rnorm(ncol(prices)) }
r_random <- simple_backtest(rng(0))
cat(sprintf("Random strategy  IS Sharpe = %.3f   IS Ann Ret = %.2f%%\n",
            r_random$sharpe, 100 * r_random$ann_ret))
"""))

cells.append(code(r"""
plot(cumprod(1 + r_random$daily) - 1, type = "l", lwd = 1.5,
     main = "Random strategy equity (noise baseline)",
     xlab = "Trading day", ylab = "P&L (cum)")
abline(h = 0, col = "grey", lwd = 0.5)
"""))

cells.append(md(r"""
---
## Part 3: Find a Real Signal

Exactly one feature is a planted **alpha** — a small but real predictive relationship with next-day
returns. The rest are noise. A good way to measure predictiveness is the **Information Coefficient
(IC)**: the rank correlation between today's signal and tomorrow's cross-sectional returns. Mean
IC > 0 means the signal is directionally useful; **IC-IR** = mean(IC) / sd(IC) × √252 measures how
reliable it is.
"""))

cells.append(code(r"""
# Spearman (rank) IC of a feature vs next-day returns, day by day.
ic_series <- function(fmat) {
  vapply(seq_len(nrow(ret)),
         function(t) cor(fmat[t, ], ret[t, ], method = "spearman"),
         numeric(1))
}

summary_ic <- data.frame(feature = feat_names, mean_ic = NA_real_, ic_ir = NA_real_)
op <- par(mfrow = c(length(feat_names), 1), mar = c(2.5, 4, 2, 1))
for (i in seq_along(feat_names)) {
  ics <- ic_series(features[[feat_names[i]]])
  mic <- mean(ics, na.rm = TRUE)
  icir <- mic / (sd(ics, na.rm = TRUE) + 1e-9) * sqrt(252)
  summary_ic[i, c("mean_ic", "ic_ir")] <- c(mic, icir)
  roll <- filter(ics, rep(1/20, 20), sides = 1)   # 20-day moving average
  plot(roll, type = "l", lwd = 1.2, col = "steelblue",
       main = sprintf("%s | mean IC=%.4f | IC-IR=%.2f", feat_names[i], mic, icir),
       xlab = "", ylab = "rolling IC")
  abline(h = 0, col = "grey", lwd = 0.5)
}
par(op)
print(summary_ic, digits = 4)
"""))

cells.append(md("**Exercise 3.1** — Which feature has the highest mean IC? Is it reliable (|IC-IR| > 1.5)?"))

cells.append(code(r"""
# The feature with the highest mean IC (auto-picked here; confirm it matches the chart).
best_feature <- summary_ic$feature[which.max(summary_ic$mean_ic)]
cat("Highest mean IC:", best_feature,
    sprintf("(IC-IR = %.2f)\n", summary_ic$ic_ir[which.max(summary_ic$mean_ic)]))
"""))

cells.append(md(r"""
---
## Part 4: The Overfitting Trap

The temptation: instead of picking the feature with the best *true* IC, pick whatever combination
scores the best **in-sample Sharpe** after trying hundreds of variations. This is **data snooping**
(p-hacking). We try 200 random feature-weight vectors at three different `top_k` cutoffs — 600
strategies — and keep the best IS Sharpe.
"""))

cells.append(code(r"""
nf <- length(feat_names)
# Stack features into a list of day-matrices we can combine linearly.
weighted_signal <- function(w) {
  w <- w / (sqrt(sum(w^2)) + 1e-8)
  function(t) {
    s <- numeric(ncol(prices))
    for (i in seq_len(nf)) s <- s + w[i] * features[[feat_names[i]]][t, ]
    s
  }
}

grid <- data.frame()
best <- list(sharpe = -Inf)
trial <- 0
for (top_k in c(10, 20, 40)) {
  for (j in seq_len(200)) {
    set.seed(trial); trial <- trial + 1
    w <- rnorm(nf)
    r <- simple_backtest(weighted_signal(w), top_k = top_k)
    grid <- rbind(grid, data.frame(top_k = top_k, sharpe = r$sharpe))
    if (r$sharpe > best$sharpe) best <- list(sharpe = r$sharpe, top_k = top_k, w = w)
  }
}
cat(sprintf("Tested %d strategies. Best IS Sharpe = %.3f (top_k=%d)\n",
            nrow(grid), best$sharpe, best$top_k))
"""))

cells.append(code(r"""
hist(grid$sharpe, breaks = 40, col = "steelblue", border = "white",
     main = "Distribution of IS Sharpe across the search grid", xlab = "IS Sharpe")
abline(v = best$sharpe, col = "red", lwd = 2)
legend("topright", sprintf("Best selected: %.2f", best$sharpe), col = "red", lwd = 2, bty = "n")
"""))

cells.append(md(r"""
**The trap:** that best IS Sharpe was *selected* from 600 trials — by chance alone some weights look
great in-sample. The grader, scoring on data you have never seen, will not be so impressed. Let's
find out.
"""))

cells.append(md(r"""
---
## Part 5: Submit to the Grader

Your strategy is an R function `on_day(day, features, prices, portfolio)` that returns a vector of
target weights. The grader runs it over the hidden holdout and reports your **OOS Sharpe**, plus
which planted alphas you recovered.

Create an API key at **convexpi.ai/settings/api-keys** (account menu → API keys) and set it below.

### 5A: Submit the grid-search "winner" (the overfit strategy)
"""))

cells.append(code(r"""
# Embed the grid-winning weights into an on_day() that linearly combines the features.
wvec <- best$w / (sqrt(sum(best$w^2)) + 1e-8)
overfit_code <- sprintf('
on_day <- function(day, features, prices, portfolio) {
  w <- c(%s)
  fn <- c(%s)
  s <- rep(0, length(prices))
  for (i in seq_along(fn)) { v <- features[[fn[i]]]; v[!is.finite(v)] <- 0; s <- s + w[i] * v }
  g <- sum(abs(s)); if (g > 0) s / g else s
}',
  paste(sprintf("%.6f", wvec), collapse = ", "),
  paste(sprintf('"%s"', feat_names), collapse = ", "))
cat(overfit_code)
"""))

cells.append(code(r"""
# Set your key once, then submit. Get one at https://www.convexpi.ai/settings/api-keys
Sys.setenv(CONVEXPI_API_KEY = "cpk_...")    # <- your key
if (nchar(Sys.getenv("CONVEXPI_API_KEY")) > 8 &&
    Sys.getenv("CONVEXPI_API_KEY") != "cpk_...") {
  submit("overfit-gridsearch-r", overfit_code)   # slug defaults to "demo-fall-2026"
} else {
  cat("No CONVEXPI_API_KEY set. Either set it above and re-run, or paste the code\n",
      "printed in the previous cell into the web editor at\n",
      "https://www.convexpi.ai/compete/demo-fall-2026/submit\n")
}
"""))

cells.append(md(r"""
**Exercise 5.1** — Is the OOS Sharpe higher or lower than the IS Sharpe you selected (the best from the grid)? Why?

### 5B: Submit the principled strategy (single factor, highest IC)

No tuning beyond the one decision you justified with the IC analysis — a cross-sectional z-score of
the best feature.
"""))

cells.append(code(r"""
principled_code <- sprintf('
on_day <- function(day, features, prices, portfolio) {
  raw <- features[["%s"]]; raw[!is.finite(raw)] <- 0
  z <- (raw - mean(raw)) / (sd(raw) + 1e-8)   # cross-sectional z-score
  g <- sum(abs(z)); if (g > 0) z / g else z
}', best_feature)
cat(principled_code)

if (nchar(Sys.getenv("CONVEXPI_API_KEY")) > 8 &&
    Sys.getenv("CONVEXPI_API_KEY") != "cpk_...") {
  submit("principled-single-factor-r", principled_code)
} else {
  cat("\n(Set CONVEXPI_API_KEY above to submit, or paste into the web editor.)\n")
}
"""))

cells.append(md(r"""
**Exercise 5.2** — Which strategy had the higher OOS Sharpe, the grid-search winner or the
principled single factor? What does that tell you about the link between IS Sharpe and OOS
performance?

---
## Part 6: Interpret Your Grade Report

Each completed submission has a report. The key fields:

| Field | Meaning |
|---|---|
| `oos_sharpe` | Annualised Sharpe on the hidden holdout — **this is your rank** |
| `is_sharpe` | Sharpe on the in-sample window (for comparison) |
| `oos_max_dd` | Worst peak-to-trough decline OOS |
| `alpha_details[].discovered` | Did you recover each planted alpha? (`corr`, `signal_ir`, `planted_bps`) |

**What to look for:**

| OOS Sharpe | Interpretation |
|---|---|
| < 0 | Loses money OOS — likely overfit |
| 0 – 0.5 | Weak or noisy; could be real or lucky |
| 0.5 – 1.5 | Decent for a real strategy with no look-ahead |
| > 1.5 | Suspiciously high — check for leaks |

A strategy that **discovered** the planted alpha (`mom_1m`) with a high `signal_ir` found the real
signal — even if costs trimmed the headline Sharpe.
"""))

cells.append(md(r"""
---
## Part 7: Challenges

Compare grade reports with classmates.

- **A (Easy):** Replace the z-score in 5B with a **rank** transform (`rank(raw) - mean(rank(raw))`).
  Does OOS Sharpe improve?
- **B (Medium):** Use an **EMA** of the signal (half-lives 5, 10, 20 days, carried in `portfolio` or
  a closure). What happens to turnover and IC as the half-life grows?
- **C (Medium):** Combine two features with weights fit on in-sample data (OLS of lagged features →
  next-day return). Does it beat the best single feature OOS?
- **D (Hard):** Walk-forward: refit weights on a rolling 100-day window each day. Does it beat the
  static weights OOS? What does that say about regime stationarity?
"""))

cells.append(code(r"""
# Challenge A starter — rank transform, tested in the local backtester.
ranked_signal <- function(t) {
  raw <- features[[best_feature]][t, ]
  rank(raw) - mean(rank(raw))
}
r_ranked <- simple_backtest(ranked_signal, top_k = 20)
cat(sprintf("Ranked strategy IS Sharpe = %.3f\n", r_ranked$sharpe))
"""))

cells.append(md(r"""
---
## Wrap-up

1. **IS Sharpe is not OOS Sharpe.** Searching over many strategies inflates in-sample performance;
   the more knobs, the wider the gap.
2. **IC is a better guide than Sharpe** for picking signals — it measures prediction directly.
3. **Simple and robust beats complex and tuned.** A single-factor z-score often beats a grid-search
   winner OOS.
4. **Same engine, any language.** Your R `on_day` is scored exactly like the Python and Julia
   versions — pick the language you think in.

---
*Next: Mission 3 (Alpha Discovery) — also available in R.*
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "R", "language": "R", "name": "ir"},
        "language_info": {"name": "R"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = os.path.join(os.path.dirname(__file__), "..", "missions",
                   "mission_01_overfitting", "notebook_r.ipynb")
out = os.path.abspath(out)
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")
print("wrote", out, "with", len(cells), "cells")
