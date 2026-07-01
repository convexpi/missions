#!/usr/bin/env python3
"""Builds missions/mission_09_pairs_trading/notebook_r.ipynb — the faithful R port."""
import json, os

def md(s):   return {"cell_type": "markdown", "metadata": {}, "source": _lines(s)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(s)}
def _lines(s):
    s = s.strip("\n"); parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]

cells = []

cells.append(md(r"""
# Mission 9: Pairs Trading & Statistical Arbitrage — R edition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/convexpi/missions/blob/main/missions/mission_09_pairs_trading/notebook_r.ipynb)

**Advanced elective.** Every strategy so far was *cross-sectional* — rank many stocks at one moment.
Pairs trading is the canonical *time-series* alternative: find two assets tied together by a long-run
equilibrium and bet that temporary divergences snap back. It's the textbook example of **statistical
arbitrage** — and of how a relationship that looks ironclad in-sample can quietly fall apart.

**Learning objectives**
- Distinguish **correlation** from **cointegration** — and why only the latter gives a tradeable spread
- Test for cointegration (OLS hedge ratio + an ADF unit-root test) and form a stationary spread
- Trade the spread with a **z-score** entry/exit rule and evaluate it
- See **spurious cointegration**: scanning many pairs manufactures false equilibria that break OOS

Everything runs locally in R; there's no submission.
"""))

cells.append(md("## Part 0: Setup"))

cells.append(code(r"""
if (!requireNamespace("convexpi", quietly = TRUE)) {
  install.packages("convexpi",
    repos = c("https://convexpi.r-universe.dev", "https://cloud.r-project.org"))
}
library(convexpi)
if (!reticulate::py_module_available("convexpi.lab")) {
  reticulate::py_install("convexpi-lab", pip = TRUE)   # needed for the Part 4 scan
}
set.seed(7)

# OLS hedge ratio: slope of b on a (how many units of A hedge one unit of B).
slope <- function(a, b) sum((a - mean(a)) * (b - mean(b))) / sum((a - mean(a))^2)

# Augmented Dickey-Fuller t-statistic (constant + k lags). Very negative => reject a unit root
# => the series is stationary (mean-reverting). We compare against critical values below.
adf_stat <- function(y, k = floor((length(y) - 1)^(1/3))) {
  n <- length(y); dy <- diff(y); ylag <- y[1:(n - 1)]; m <- n - 1
  idx <- (k + 1):m
  X <- cbind(1, ylag[idx])
  if (k > 0) for (i in 1:k) X <- cbind(X, dy[idx - i])
  yv <- dy[idx]
  XtX <- t(X) %*% X; b <- solve(XtX, t(X) %*% yv)
  resid <- yv - X %*% b; dof <- length(yv) - ncol(X)
  se2 <- (sum(resid^2) / dof) * solve(XtX)[2, 2]
  as.numeric(b[2] / sqrt(se2))
}
# 5% critical values: single series ≈ -2.86; Engle-Granger cointegration (2 series, β estimated) ≈ -3.34.
CRIT_COINT <- -3.34
cat("Ready.\n")
"""))

cells.append(md(r"""
---
## Part 1: Correlation is not cointegration

Two prices can be highly *correlated* day to day yet drift apart forever. Pairs trading needs
something stronger: **cointegration** — the prices share a common stochastic trend, so a particular
linear combination (the **spread**) is *stationary*, wandering around a fixed mean and returning to
it. Let's manufacture one of each.
"""))

cells.append(code(r"""
T <- 800
common <- cumsum(rnorm(T))                      # a shared stochastic trend
A <- 50 + common + rnorm(T)                     # both prices ride `common`...
B <- 20 + 0.8 * common + rnorm(T)               # ...B with sensitivity (beta) 0.8

X <- cumsum(rnorm(T))                           # control: two INDEPENDENT random walks
Y <- cumsum(rnorm(T))

op <- par(mfrow = c(1, 2), mar = c(4, 4, 2, 1))
matplot(cbind(A, B), type = "l", lty = 1, col = c("steelblue", "tomato"),
        main = "Cointegrated pair", xlab = "day", ylab = "price"); legend("topleft", c("A", "B"), col = c("steelblue", "tomato"), lty = 1, bty = "n")
matplot(cbind(X, Y), type = "l", lty = 1, col = c("steelblue", "tomato"),
        main = "Independent random walks", xlab = "day", ylab = "price"); legend("topleft", c("X", "Y"), col = c("steelblue", "tomato"), lty = 1, bty = "n")
par(op)
"""))

cells.append(md(r"""
Both panels *look* like prices that move together — which is exactly why we need a test, not a chart.

---
## Part 2: Testing for cointegration

Engle-Granger recipe: regress one price on the other for a **hedge ratio** β, form the spread
`B − βA`, and test whether that spread is **stationary** with an ADF test. A test statistic below the
critical value means "the spread reverts" — the pair is cointegrated.
"""))

cells.append(code(r"""
hedge_and_spread <- function(a, b) { beta <- slope(a, b); list(beta = beta, spread = b - beta * a) }

for (nm in c("cointegrated (A,B)", "independent (X,Y)")) {
  d <- if (nm == "cointegrated (A,B)") hedge_and_spread(A, B) else hedge_and_spread(X, Y)
  stat <- adf_stat(d$spread)
  verdict <- if (stat < CRIT_COINT) "COINTEGRATED" else "not cointegrated"
  cat(sprintf("%-20s beta=%5.2f  ADF(spread)=%7.3f  (5%% crit %.2f)  -> %s\n",
              nm, d$beta, stat, CRIT_COINT, verdict))
}
"""))

cells.append(code(r"""
sp_AB <- hedge_and_spread(A, B)$spread
sp_XY <- hedge_and_spread(X, Y)$spread
plot(sp_AB - mean(sp_AB), type = "l", col = "steelblue", lwd = 1.3,
     ylim = range(c(sp_AB - mean(sp_AB), sp_XY - mean(sp_XY))),
     main = "A tradeable spread reverts to its mean", xlab = "day", ylab = "spread")
lines(sp_XY - mean(sp_XY), col = "tomato", lwd = 1.3)
abline(h = 0, lwd = 0.8)
legend("topleft", c("A,B spread (stationary)", "X,Y spread (wanders)"),
       col = c("steelblue", "tomato"), lty = 1, bty = "n")
"""))

cells.append(md(r"""
**Exercise 2.1** — Lower B's sensitivity to the common trend toward zero (`0.8 → 0.1`) and add more
idiosyncratic noise. At what point does the ADF test stop calling the pair cointegrated?
Cointegration is a matter of degree — weak cointegration is barely tradeable after costs.

---
## Part 3: Trading the spread

Once you trust the spread, standardise it into a rolling **z-score**. When it stretches far from its
mean (`|z| > entry`) bet on reversion — short the spread when high, long it when low — and close as it
returns (`|z| < exit`).
"""))

cells.append(code(r"""
backtest_pair <- function(a, b, beta, entry = 2, exit = 0.5, lookback = 60) {
  s <- b - beta * a; n <- length(s); z <- rep(NA_real_, n)
  for (t in (lookback + 1):n) { win <- s[(t - lookback):(t - 1)]; z[t] <- (s[t] - mean(win)) / (sd(win) + 1e-9) }
  ds <- diff(s); pos <- 0; pnl <- c(); states <- c()
  for (t in (lookback + 1):(n - 1)) {
    if (pos == 0) { if (z[t] > entry) pos <- -1 else if (z[t] < -entry) pos <- 1 }
    else if (abs(z[t]) < exit) pos <- 0
    pnl <- c(pnl, pos * ds[t]); states <- c(states, pos)
  }
  list(z = z, states = states, pnl = pnl,
       sharpe = mean(pnl) / (sd(pnl) + 1e-9) * sqrt(252))
}

beta <- slope(A, B)
r <- backtest_pair(A, B, beta)
cat(sprintf("hedge ratio beta : %.2f\npair Sharpe      : %.2f\ntotal spread P&L : %.1f  (round-trips: %d)\n",
            beta, r$sharpe, sum(r$pnl), sum(diff(r$states) != 0)))

op <- par(mfrow = c(2, 1), mar = c(3, 4, 2, 1))
plot(r$z, type = "l", ylab = "spread z-score", main = "Entry at |z|>2, exit near 0")
abline(h = c(-2, 0, 2), col = c("tomato", "black", "tomato"), lty = c(2, 1, 2))
plot(cumsum(r$pnl), type = "l", ylab = "cumulative P&L", xlab = "day")
par(op)
"""))

cells.append(md(r"""
**Exercise 3.1** — Sweep `entry` over `c(1.0, 1.5, 2.0, 2.5, 3.0)`. Wider thresholds trade less often
but each trade is a stronger signal. Where is Sharpe best, and what happens to the round-trip count?
(Recall Mission 8: more round-trips → more transaction costs.)

---
## Part 4: The danger — spurious cointegration

Scan thousands of pairs for cointegration and you'll find plenty **by chance**. At 5% significance,
~5% of unrelated pairs "pass" in-sample. Those aren't equilibria — they're noise that briefly rhymed,
and they fall apart out of sample. We scan a synthetic universe, flag pairs cointegrated **in-sample**,
then check how many still pass on the **holdout**.
"""))

cells.append(code(r"""
Pis  <- synthetic_market("train")$prices[, 1:60]     # 60 assets, in-sample window
Poos <- synthetic_market("test")$prices[, 1:60]      # holdout window (the honest test)
N <- ncol(Pis)

coint_ok <- function(a, b) adf_stat(b - slope(a, b) * a) < CRIT_COINT
is_hits <- 0; survivors <- 0; broken <- NULL
for (i in 1:(N - 1)) for (j in (i + 1):N) {
  if (coint_ok(Pis[, i], Pis[, j])) {
    is_hits <- is_hits + 1
    if (coint_ok(Poos[, i], Poos[, j])) survivors <- survivors + 1
    else if (is.null(broken)) broken <- c(i, j)
  }
}
checked <- N * (N - 1) / 2
cat(sprintf("pairs scanned            : %d\n", checked))
cat(sprintf("'cointegrated' in-sample : %d  (%.1f%% — near the 5%% you'd expect by chance)\n",
            is_hits, 100 * is_hits / checked))
cat(sprintf("still cointegrated OOS   : %d  (%.0f%% of the in-sample hits)\n",
            survivors, 100 * survivors / max(is_hits, 1)))
cat("\nMost 'discovered' pairs are spurious. Finding cointegration is easy; finding it OOS is the job.\n")
"""))

cells.append(code(r"""
# A spurious pair: mean-reverting in-sample, then drifts on the holdout (using the IS hedge ratio).
if (!is.null(broken)) {
  i <- broken[1]; j <- broken[2]
  beta_is <- slope(Pis[, i], Pis[, j])
  sp_is  <- Pis[, j]  - beta_is * Pis[, i]
  sp_oos <- Poos[, j] - beta_is * Poos[, i]
  plot(seq_along(sp_is), sp_is - mean(sp_is), type = "l", col = "steelblue",
       xlim = c(1, length(sp_is) + length(sp_oos)),
       ylim = range(c(sp_is - mean(sp_is), sp_oos - mean(sp_is))),
       main = sprintf("Spurious pair (assets %d,%d): the equilibrium was never real", i, j),
       xlab = "day", ylab = "spread")
  lines(seq(length(sp_is) + 1, length(sp_is) + length(sp_oos)), sp_oos - mean(sp_is), col = "tomato")
  abline(v = length(sp_is), lty = 3); abline(h = 0, lwd = 0.6)
  legend("topleft", c("in-sample (looks tradeable)", "holdout (drifts away)"),
         col = c("steelblue", "tomato"), lty = 1, bty = "n")
} else cat("No spurious example found in this scan.\n")
"""))

cells.append(md(r"""
This is Mission 1's lesson in a time-series costume. The defences are the same: **out-of-sample
validation** (does the pair still cointegrate on the holdout?), **multiple-testing awareness** (you
tested thousands of pairs — tighten the threshold, Bonferroni/FDR), and an **economic prior** (pairs
with a real reason to be linked — same sector, same supply chain — persist far more than pairs your
scan stumbled on).

---
## Challenge

Build an honest pairs pipeline on the synthetic universe:

1. Keep only pairs that cointegrate **in-sample *and*** on a validation slice (a true OOS filter).
2. Backtest the z-score rule on the survivors, net of costs (reuse Mission 8 — pairs can be
   high-turnover).
3. Compare the survivors' net Sharpe against the full in-sample set. How much "edge" was spurious?

Publish it to **[/projects](https://convexpi.ai/projects)** — a clean demonstration of
spurious-cointegration discipline is exactly what the leaderboard rewards.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "R", "language": "R", "name": "ir"},
                   "language_info": {"name": "R"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "missions",
                                   "mission_09_pairs_trading", "notebook_r.ipynb"))
with open(out, "w") as f:
    json.dump(nb, f, indent=1); f.write("\n")
print("wrote", out, "with", len(cells), "cells")
