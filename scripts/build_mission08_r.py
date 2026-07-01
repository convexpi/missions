#!/usr/bin/env python3
"""Builds missions/mission_08_cost_of_trading/notebook_r.ipynb — the faithful R port."""
import json, os

def md(s):   return {"cell_type": "markdown", "metadata": {}, "source": _lines(s)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(s)}
def _lines(s):
    s = s.strip("\n"); parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]

cells = []

cells.append(md(r"""
# Mission 8: The Cost of Trading — R edition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/convexpi/missions/blob/main/missions/mission_08_cost_of_trading/notebook_r.ipynb)

**Advanced elective.** You found an alpha — now try to *keep* it. This mission is about the gap
between a frictionless backtest and a tradeable strategy. The villain is **turnover**: every
rebalance pays the spread, and a signal that looks great traded daily can be a guaranteed loser once
costs are real. Everything here runs locally in R; there's no submission.

**Learning objectives**
- Quantify how **transaction costs** scale with turnover and erase paper alpha
- Use **rebalance frequency** and **no-trade bands** to trade turnover against signal freshness
- Find a strategy's **break-even cost** — the TC at which its edge disappears
- Reason about **capacity**: why size itself moves the price against you
"""))

cells.append(md("## Part 0: Setup"))

cells.append(code(r"""
if (!requireNamespace("convexpi", quietly = TRUE)) {
  if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
  remotes::install_github("convexpi/convexpi-r", upgrade = "never")
}
library(convexpi)
if (!reticulate::py_module_available("convexpi.lab")) {
  reticulate::py_install("convexpi-lab", pip = TRUE)
}
m <- synthetic_market("train")
prices   <- m$prices
features <- m$features
ret <- prices[-1, ] / prices[-nrow(prices), ] - 1     # (days-1) x stocks, next-day returns

# A backtester that mirrors the Python engine's accounting: a strategy(t, portfolio) -> target
# weights, applied on a rebalance cadence, charged tc_bps on the weight it changes.
run_bt <- function(strategy, tc_bps = 0, rebalance_every = 1, warmup = 60) {
  T <- nrow(ret); daily <- c(); turn <- c(); portfolio <- rep(0, ncol(prices))
  for (t in (warmup + 1):T) {
    w <- if ((t - (warmup + 1)) %% rebalance_every == 0) strategy(t, portfolio) else portfolio
    traded <- sum(abs(w - portfolio))                 # total weight changed today
    daily  <- c(daily, sum(w * ret[t, ]) - tc_bps / 1e4 * traded)
    turn   <- c(turn, traded)
    portfolio <- w
  }
  list(sharpe = mean(daily) / (sd(daily) + 1e-9) * sqrt(252),
       ann_ret = mean(daily) * 252,
       turnover_annual = mean(turn) * 252)
}
cat("Ready.", nrow(prices), "days x", ncol(prices), "stocks\n")
"""))

cells.append(md(r"""
---
## Part 1: The same alpha, two different costs

One honest signal: 1-month momentum (`mom_1m`, a planted alpha). The strategy is a quintile
long/short at gross exposure 1.0. Run it **frictionless**, then with a realistic **20 bps** one-way
cost — rebalancing daily either way.
"""))

cells.append(code(r"""
# Quintile long/short on one feature, gross exposure 1.0.
momentum <- function(feature = "mom_1m") function(t, portfolio) {
  s <- features[[feature]][t, ]; s[!is.finite(s)] <- 0
  z <- (s - mean(s)) / (sd(s) + 1e-9)
  k <- max(1, length(z) %/% 5)
  ord <- order(z); w <- rep(0, length(z))
  w[tail(ord, k)] <- 1; w[head(ord, k)] <- -1
  w / (sum(abs(w)) + 1e-9)
}

for (lab in list(c("frictionless (0 bps)", 0), c("realistic (20 bps)", 20))) {
  r <- run_bt(momentum(), tc_bps = as.numeric(lab[2]), rebalance_every = 1)
  cat(sprintf("%-22s Sharpe=%6.3f  ann_ret=%7.2f%%  turnover=%6.1fx\n",
              lab[1], r$sharpe, 100 * r$ann_ret, r$turnover_annual))
}
"""))

cells.append(md(r"""
The frictionless backtest looks like a real edge. Add 20 bps traded daily and the **same signal
becomes a money-loser** — costs eat it alive. The backtest didn't lie about the alpha; it lied about
what you'd *net*. Turnover is the bridge.

---
## Part 2: Rebalance frequency — turnover vs freshness

The simplest turnover lever is *how often you trade*. Rebalance less often to cut costs, at the price
of letting positions drift from the signal. Sweep it at 20 bps:
"""))

cells.append(code(r"""
rbs <- c(1, 2, 5, 10, 21, 42); sh <- c(); to <- c()
cat(sprintf("%9s%9s%10s%10s\n", "rebal(d)", "Sharpe", "turnover", "ann_ret"))
for (rb in rbs) {
  r <- run_bt(momentum(), tc_bps = 20, rebalance_every = rb)
  sh <- c(sh, r$sharpe); to <- c(to, r$turnover_annual)
  cat(sprintf("%9d%9.3f%9.1fx%10.2f%%\n", rb, r$sharpe, r$turnover_annual, 100 * r$ann_ret))
}
plot(rbs, sh, type = "b", pch = 19, col = "steelblue", xlab = "rebalance every N days",
     ylab = "net Sharpe", main = "Trading less often: higher net Sharpe (at 20 bps)")
abline(h = 0, lwd = 0.8)
"""))

cells.append(md(r"""
**Exercise 2.1** — Does the *gross* (frictionless) Sharpe also rise with slower rebalancing, or is
the improvement entirely a cost effect? Re-run the sweep at `tc_bps = 0` to separate the two.

---
## Part 3: Break-even cost

Every strategy has a cost at which its net edge hits zero — the **break-even cost**, the single most
useful number for judging tradeability. Fix a sensible rebalance frequency and sweep the cost.
"""))

cells.append(code(r"""
costs <- c(0, 5, 10, 20, 30, 50, 75, 100)
sharpes <- sapply(costs, function(c) run_bt(momentum(), tc_bps = c, rebalance_every = 5)$sharpe)
for (i in seq_along(costs)) cat(sprintf("  tc=%3d bps -> net Sharpe %6.3f\n", costs[i], sharpes[i]))

plot(costs, sharpes, type = "b", pch = 19, xlab = "one-way transaction cost (bps)",
     ylab = "net Sharpe", main = "Break-even cost: where the edge disappears")
abline(h = 0, col = "crimson", lty = 2)

cross <- which(diff(sign(sharpes)) != 0)
if (length(cross)) {
  i <- cross[1]
  be <- costs[i] + (costs[i + 1] - costs[i]) * sharpes[i] / (sharpes[i] - sharpes[i + 1])
  cat(sprintf("\nApprox break-even cost (rebal=5d): ~%.0f bps\n", be))
}
"""))

cells.append(md(r"""
If your break-even cost is comfortably above what you'd actually pay (spread + impact + fees), the
edge survives; if it's below, the alpha is real on paper and worthless in practice.

---
## Part 4: Reducing turnover with a no-trade band

Trading less *often* is blunt. A **no-trade band** is smarter: move a position only when its target
has drifted far enough from where you already are, so you stop churning on noise while staying
responsive day to day.
"""))

cells.append(code(r"""
banded_momentum <- function(feature = "mom_1m", band = 0) {
  base <- momentum(feature)
  function(t, portfolio) {
    target <- base(t, portfolio)
    if (band <= 0) return(target)
    ifelse(abs(target - portfolio) > band, target, portfolio)   # hold unless drift exceeds band
  }
}

cat(sprintf("%6s%9s%10s%10s\n", "band", "Sharpe", "turnover", "ann_ret"))
for (band in c(0, 0.002, 0.005, 0.01)) {
  r <- run_bt(banded_momentum(band = band), tc_bps = 20, rebalance_every = 1)
  cat(sprintf("%6.3f%9.3f%9.1fx%10.2f%%\n", band, r$sharpe, r$turnover_annual, 100 * r$ann_ret))
}
"""))

cells.append(md(r"""
A band lets you rebalance *daily* (responsive to the signal) while paying far less than naive daily
trading — often beating a fixed slow rebalance.

**Exercise 4.1** — Compare the *best* band against the *best* rebalance frequency from Part 2 at
20 bps. Which wins on net Sharpe, and why might a band be preferable when the signal moves sharply?

---
## Part 5: Capacity — when your own size moves the price

So far cost-per-trade was constant. In reality **larger orders move the price against you** (market
impact), so cost *rises with size*. A square-root model says trading a fraction `q` of a name's daily
volume costs about `impact ≈ k · √q`. That caps how much capital a strategy can run — its
**capacity**.
"""))

cells.append(code(r"""
base_bps <- 5; k_bps <- 8; adv_usd <- 5e6          # spread+fees, impact coef, avg daily volume/name
aums <- c(1e6, 1e7, 5e7, 1e8, 5e8, 1e9)
cap_sh <- sapply(aums, function(aum) {
  q  <- (aum / 40) / adv_usd                        # fraction of ADV per name (~20 long + 20 short)
  tc <- base_bps + k_bps * sqrt(max(q, 0))
  run_bt(momentum(), tc_bps = tc, rebalance_every = 5)$sharpe
})
for (i in seq_along(aums)) cat(sprintf("  AUM $%7.0fM -> net Sharpe %6.3f\n", aums[i] / 1e6, cap_sh[i]))
plot(aums, cap_sh, type = "b", pch = 19, log = "x", xlab = "assets under management ($)",
     ylab = "net Sharpe", main = "Capacity: impact rises with size until the edge is gone")
abline(h = 0, col = "crimson", lty = 2)
"""))

cells.append(md(r"""
Capacity is why a strategy spectacular at \$1M can be mediocre at \$1B: the alpha per dollar is
finite and impact taxes every extra dollar harder. Small funds trade high-turnover signals big funds
can't touch; large funds are pushed toward slow, high-capacity factors.

---
## Challenge

Take *your* best strategy from an earlier mission and produce its **cost report**:

1. Turnover and net Sharpe at 0 / 10 / 20 / 50 bps.
2. Its break-even cost.
3. A turnover-reduction variant (slower rebalance or a band) that improves net Sharpe.
4. A one-paragraph honest verdict: at what cost and what AUM does it stop being worth trading?

Publish it to **[/projects](https://convexpi.ai/projects)** — a clear-eyed capacity analysis is what
separates a backtest from a strategy.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "R", "language": "R", "name": "ir"},
                   "language_info": {"name": "R"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "missions",
                                   "mission_08_cost_of_trading", "notebook_r.ipynb"))
with open(out, "w") as f:
    json.dump(nb, f, indent=1); f.write("\n")
print("wrote", out, "with", len(cells), "cells")
