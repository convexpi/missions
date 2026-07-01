#!/usr/bin/env python3
"""Builds missions/mission_04_strategy_library/notebook_r.ipynb — the faithful R port."""
import json, os

def md(s):   return {"cell_type": "markdown", "metadata": {}, "source": _lines(s)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(s)}
def _lines(s):
    s = s.strip("\n"); parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]

cells = []

cells.append(md(r"""
# Mission 4: The Strategy Library — R edition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/convexpi/missions/blob/main/missions/mission_04_strategy_library/notebook_r.ipynb)

**Learning objectives**
- Understand the economic intuition behind canonical quant strategies (momentum, value, quality,
  size, risk-based)
- **Build the strategy zoo yourself in R** and run a tournament across all of them
- Diagnose *why* strategies fail: crowding, transaction costs, regimes
- Build your own composite by IC/rank combination
- Confront IS vs OOS — even canonical strategies overfit when you pick the in-sample winner

> The Python mission calls a built-in strategy library. Here you **implement** that library
> natively in R — each strategy is the same signal→weight rule, run through one backtester. The
> lessons (which family wins, the cost tax, the IS/OOS gap) are identical.
"""))

cells.append(md("## Part 0: Setup"))

cells.append(code(r"""
if (!requireNamespace("convexpi", quietly = TRUE)) {
  install.packages("convexpi",
    repos = c("https://convexpi.r-universe.dev", "https://cloud.r-project.org"))
}
library(convexpi)
if (!reticulate::py_module_available("convexpi.lab")) {
  reticulate::py_install("convexpi-lab", pip = TRUE)
}

# Load both splits: train = in-sample, test = the hidden holdout (peek only at the very end).
tr <- synthetic_market("train"); te <- synthetic_market("test")
mk <- function(m) list(P = m$prices, F = m$features,
                       R = m$prices[-1, ] / m$prices[-nrow(m$prices), ] - 1)
TRAIN <- mk(tr); TEST <- mk(te)
cat("train:", nrow(TRAIN$P), "days | test:", nrow(TEST$P), "days | features:",
    paste(names(TRAIN$F), collapse = ", "), "\n")
"""))

cells.append(md(r"""
---
## Part 1: The building blocks

Every strategy turns a cross-sectional signal into a weight vector with two primitives: a
**cross-sectional z-score**, and a **quintile long/short** (long the top 20%, short the bottom 20%,
gross exposure 1).
"""))

cells.append(code(r"""
zscore <- function(x) {
  v <- is.finite(x); if (sum(v) < 2) return(rep(0, length(x)))
  mu <- mean(x[v]); s <- sd(x[v]); ifelse(v, (x - mu) / (s + 1e-9), 0)
}
# Long top quintile (+1), short bottom quintile (-1), normalise to gross 1.
ls_weights <- function(sig, q = 0.2, long_only = FALSE) {
  v <- is.finite(sig); if (sum(v) < 10) return(rep(0, length(sig)))
  lo <- quantile(sig[v], q); hi <- quantile(sig[v], 1 - q)
  w <- rep(0, length(sig)); w[v & sig >= hi] <- 1
  if (!long_only) w[v & sig <= lo] <- -1
  tot <- sum(abs(w)); if (tot > 0) w / tot else w
}
rank_corr <- function(x, y) {
  v <- is.finite(x) & is.finite(y); if (sum(v) < 5) return(0)
  cor(rank(x[v]), rank(y[v]))
}
cat("primitives ready\n")
"""))

cells.append(md(r"""
---
## Part 2: The strategy zoo

Eighteen canonical strategies across five families. Each is a function `strat(t, portfolio)` that
reads the market at day `t` (via the globals `F`, `R`, `P` — set to whichever split we're running)
and returns target weights.
"""))

cells.append(code(r"""
# `F`, `R`, `P` are the current split's features / returns / prices (set before each run).
F <- TRAIN$F; R <- TRAIN$R; P <- TRAIN$P

STRATEGIES <- list(
  equal_weight   = function(t, pf) rep(1 / ncol(P), ncol(P)),                       # baseline
  momentum_12_1  = function(t, pf) ls_weights(F[["mom_12m"]][t, ]),                 # Jegadeesh-Titman
  momentum_3m    = function(t, pf) ls_weights(F[["mom_3m"]][t, ]),
  momentum_1m    = function(t, pf) ls_weights(F[["mom_1m"]][t, ]),
  reversal_1w    = function(t, pf) ls_weights(F[["reversal_1w"]][t, ]),             # short-term contrarian
  ts_momentum    = function(t, pf) {                                                # time-series momentum
    sig <- F[["mom_12m"]][t, ]; v <- is.finite(sig); n <- sum(v)
    if (n == 0) return(rep(0, length(sig)))
    w <- ifelse(v, sign(sig) / n, 0); tot <- sum(abs(w)); if (tot > 0) w / tot else w
  },
  value_bm           = function(t, pf) ls_weights(F[["val_bm"]][t, ]),             # Fama-French value
  value_bm_long_only = function(t, pf) ls_weights(F[["val_bm"]][t, ], long_only = TRUE),
  quality_roe    = function(t, pf) ls_weights(F[["qual_roe"]][t, ]),               # Novy-Marx quality
  betting_against_beta = function(t, pf) ls_weights(-F[["vol_1m"]][t, ]),          # Frazzini-Pedersen
  size_premium   = function(t, pf) ls_weights(-F[["size_cap"]][t, ]),              # Banz small-cap
  fama_french_3  = function(t, pf)                                                 # SMB + HML blend
    ls_weights(0.5 * zscore(F[["val_bm"]][t, ]) + 0.5 * zscore(-F[["size_cap"]][t, ])),
  multi_factor_rank = function(t, pf)                                              # rank-sum blend
    ls_weights((zscore(F[["mom_12m"]][t, ]) + zscore(F[["val_bm"]][t, ]) + zscore(F[["qual_roe"]][t, ])) / 3),
  ic_weighted    = function(t, pf) {                                               # rolling IC-weighted
    sigs <- c("mom_12m", "val_bm", "qual_roe"); lo <- max(1, t - 60)
    icm <- sapply(sigs, function(f) mean(sapply(lo:(t - 1), function(s) rank_corr(F[[f]][s, ], R[s, ]))))
    w <- pmax(0.05, icm); w <- w / sum(w)
    comp <- rep(0, ncol(P)); for (i in seq_along(sigs)) comp <- comp + w[i] * zscore(F[[sigs[i]]][t, ])
    ls_weights(comp)
  },
  inv_vol        = function(t, pf) {                                               # risk parity (long-only)
    vol <- F[["vol_1m"]][t, ]; v <- is.finite(vol) & vol > 0
    if (sum(v) < 2) return(rep(0, length(vol)))
    w <- rep(0, length(vol)); w[v] <- 1 / vol[v]; tot <- sum(w); if (tot > 0) w / tot else w
  },
  min_variance   = function(t, pf) {                                              # lowest-vol quintile
    vol <- F[["vol_1m"]][t, ]; v <- is.finite(vol); if (sum(v) < 5) return(rep(0, length(vol)))
    thr <- quantile(vol[v], 0.2); w <- ifelse(v & vol <= thr, 1, 0); tot <- sum(w); if (tot > 0) w / tot else w
  },
  dual_momentum  = function(t, pf) {                                              # cross-sec gated by absolute
    cs <- F[["mom_12m"]][t, ]; ls_weights(ifelse(cs > 0, cs, NA), long_only = TRUE)
  },
  trend_filter   = function(t, pf) {                                              # momentum, flat when trend down
    sig <- F[["mom_12m"]][t, ]; if (mean(sig[is.finite(sig)]) > 0) ls_weights(sig) else rep(0, ncol(P))
  }
)
cat("registered", length(STRATEGIES), "strategies\n")
"""))

cells.append(md(r"""
---
## Part 3: The tournament

`run_metrics()` runs one strategy through a daily backtester (10 bps one-way cost, 63-day warmup) and
reports the seven metrics you'd judge a strategy on. `compare_all()` runs the whole zoo.
"""))

cells.append(code(r"""
run_metrics <- function(strat, warmup = 63, tc_bps = 10) {
  n <- ncol(P); portfolio <- rep(0, n); daily <- c(); turn <- c()
  for (t in (warmup + 1):nrow(R)) {
    w <- strat(t, portfolio); w[!is.finite(w)] <- 0
    traded <- sum(abs(w - portfolio))
    daily <- c(daily, sum(w * R[t, ]) - tc_bps / 1e4 * traded); turn <- c(turn, traded); portfolio <- w
  }
  eq <- cumprod(1 + daily); dd <- eq / cummax(eq) - 1; maxdd <- min(dd)
  ann_ret <- mean(daily) * 252
  list(cum = eq - 1, daily = daily,
       sharpe = mean(daily) / (sd(daily) + 1e-9) * sqrt(252),
       annual_return = ann_ret, annual_vol = sd(daily) * sqrt(252),
       max_drawdown = maxdd, calmar = ann_ret / (abs(maxdd) + 1e-9),
       annual_turnover = mean(turn) * 252, hit_rate = mean(daily > 0))
}
compare_all <- function(strats, warmup = 63, tc_bps = 10) {
  do.call(rbind, lapply(names(strats), function(k) {
    m <- run_metrics(strats[[k]], warmup, tc_bps)
    data.frame(strategy = k, sharpe = m$sharpe, ann_return = m$annual_return,
               max_dd = m$max_drawdown, calmar = m$calmar,
               turnover = m$annual_turnover, hit_rate = m$hit_rate)
  }))
}

F <- TRAIN$F; R <- TRAIN$R; P <- TRAIN$P          # run the tournament in-sample
results <- compare_all(STRATEGIES)
results <- results[order(-results$sharpe), ]
print(results, digits = 3, row.names = FALSE)
"""))

cells.append(code(r"""
o <- results[order(results$sharpe), ]
barplot(o$sharpe, names.arg = o$strategy, horiz = TRUE, las = 1, cex.names = 0.6,
        col = ifelse(o$sharpe > 0, "#2ecc71", "#e74c3c"),
        main = "Strategy tournament — in-sample Sharpe (10 bps)", xlab = "Sharpe")
abline(v = 0, lwd = 0.8)
"""))

cells.append(md(r"""
**Discussion:** Which family dominates this market? Note the `turnover` column — a strategy turning
over 40×/yr at 10 bps pays ~8%/yr in costs before any alpha. How do the "smart" strategies beat plain
`equal_weight`?

---
## Part 4: Anatomy of a strategy

Open up cross-sectional momentum: how does the `mom_12m` signal map to portfolio weights on a single
day? A clean long/short is roughly market-neutral (net exposure ≈ 0).
"""))

cells.append(code(r"""
t0 <- 300
w <- STRATEGIES$momentum_12_1(t0, rep(0, ncol(P)))
sig <- F[["mom_12m"]][t0, ]
op <- par(mfrow = c(1, 2), mar = c(4, 4, 2, 1))
hist(w[w != 0], breaks = 30, col = "steelblue", main = "Weight distribution (day 300)", xlab = "weight")
plot(sig, w, pch = 19, cex = 0.4, col = rgb(0, 0, 0, 0.4), xlab = "mom_12m signal",
     ylab = "portfolio weight", main = "Signal vs weight"); abline(h = 0, v = 0, lwd = 0.6)
par(op)
cat(sprintf("Long %d | Short %d | net exposure %.4f (≈0)\n",
            sum(w > 0), sum(w < 0), sum(w)))
"""))

cells.append(md(r"""
---
## Part 5: Cumulative returns — when each approach earns

The table flattens time. Equity curves show *when* each strategy makes its money — and how correlated
they are (diversification).
"""))

cells.append(code(r"""
focus <- c("equal_weight", "momentum_12_1", "value_bm", "quality_roe",
           "betting_against_beta", "ic_weighted", "fama_french_3")
curves <- lapply(focus, function(k) run_metrics(STRATEGIES[[k]])$cum)
ymax <- max(sapply(curves, max)); ymin <- min(sapply(curves, min))
plot(NA, xlim = c(1, length(curves[[1]])), ylim = c(ymin, ymax),
     xlab = "trading day", ylab = "cumulative return", main = "Cumulative returns — IS (10 bps)")
abline(h = 0, lty = 2, lwd = 0.6)
for (i in seq_along(focus)) lines(curves[[i]], col = i, lwd = 1.5)
legend("topleft", legend = focus, col = seq_along(focus), lwd = 1.5, cex = 0.7, bty = "n")
"""))

cells.append(md(r"""
---
## Part 6: The transaction-cost tax

High-turnover strategies look great at 0 bps and collapse under realistic costs. Sweep the cost for a
handful of strategies and watch the ranking change.
"""))

cells.append(code(r"""
tc_levels <- c(0, 5, 10, 20, 30)
focus_keys <- c("equal_weight", "momentum_12_1", "momentum_3m", "value_bm",
                "quality_roe", "ic_weighted", "inv_vol")
tc_mat <- sapply(tc_levels, function(tc) sapply(focus_keys, function(k) run_metrics(STRATEGIES[[k]], tc_bps = tc)$sharpe))
colnames(tc_mat) <- paste0(tc_levels, "bps")
print(round(tc_mat, 3))
matplot(tc_levels, t(tc_mat), type = "b", pch = 19, lty = 1, col = seq_along(focus_keys),
        xlab = "transaction cost (bps/side)", ylab = "Sharpe", main = "TC sensitivity")
abline(h = 0, lty = 2, lwd = 0.6)
legend("topright", legend = focus_keys, col = seq_along(focus_keys), lty = 1, pch = 19, cex = 0.6, bty = "n")
"""))

cells.append(md(r"""
---
## Part 7: Build your own composite

Combine any signals by z-scoring and blending, then take the quintile long/short. Try adding or
dropping a signal, or inverting one (e.g. `size_cap` for a small-cap tilt).
"""))

cells.append(code(r"""
multi_factor <- function(signals, invert = character(0), long_only = FALSE) function(t, pf) {
  comp <- rep(0, ncol(P))
  for (f in signals) { z <- zscore(F[[f]][t, ]); if (f %in% invert) z <- -z; comp <- comp + z }
  ls_weights(comp / length(signals), long_only = long_only)
}

mine <- multi_factor(c("mom_12m", "qual_roe", "val_bm"))     # EDIT THIS
m <- run_metrics(mine)
cat(sprintf("your composite  Sharpe=%.3f  ann_ret=%.2f%%  turnover=%.1fx  maxDD=%.1f%%\n",
            m$sharpe, 100 * m$annual_return, m$annual_turnover, 100 * m$max_drawdown))
"""))

cells.append(md(r"""
---
## Part 8: IS vs OOS — the reality check

Everything above was in-sample. Take the **top-5 IS strategies** and run them on the hidden holdout.
OOS Sharpe is almost always lower — the gap is overfitting, even for canonical strategies nobody fit
to this data. **You look at OOS once.**
"""))

cells.append(code(r"""
top5 <- head(results$strategy, 5)

F <- TRAIN$F; R <- TRAIN$R; P <- TRAIN$P
is_sh  <- sapply(top5, function(k) run_metrics(STRATEGIES[[k]])$sharpe)
F <- TEST$F;  R <- TEST$R;  P <- TEST$P            # switch to the holdout — the one peek
oos_sh <- sapply(top5, function(k) run_metrics(STRATEGIES[[k]])$sharpe)
F <- TRAIN$F; R <- TRAIN$R; P <- TRAIN$P           # restore

cmp <- data.frame(strategy = top5, IS_sharpe = is_sh, OOS_sharpe = oos_sh)
print(cmp, digits = 3, row.names = FALSE)
barplot(t(as.matrix(cmp[, c("IS_sharpe", "OOS_sharpe")])), beside = TRUE, names.arg = top5,
        las = 2, cex.names = 0.6, col = c("steelblue", "coral"),
        legend.text = c("IS", "OOS"), main = "Top-5 IS strategies: IS vs OOS Sharpe")
abline(h = 0, lwd = 0.8)
"""))

cells.append(md(r"""
---
## Part 9: Challenges

- **A — Low-turnover portfolio:** build a strategy with IS Sharpe > 0.3 *and* turnover < 5×.
  (`inv_vol` and `min_variance` have naturally low turnover.)
- **B — Regime filter:** compare `trend_filter` (momentum, flat when the trend is down) against raw
  `momentum_12_1` on Calmar (return / max drawdown). Does the filter help?
- **C — Beat the registry:** design a strategy with higher **OOS** Sharpe than every strategy here.
  Develop only on train; check the holdout once. Document your reasoning.
"""))

cells.append(code(r"""
# Challenge B starter — Calmar of raw vs trend-filtered momentum (in-sample).
for (k in c("momentum_12_1", "trend_filter")) {
  m <- run_metrics(STRATEGIES[[k]])
  cat(sprintf("%-14s Sharpe=%.3f  maxDD=%.1f%%  Calmar=%.3f\n",
              k, m$sharpe, 100 * m$max_drawdown, m$calmar))
}
"""))

cells.append(md(r"""
---
## Wrap-up

1. **No single strategy dominates all regimes** — the IS winner is rarely the OOS winner.
2. **Transaction costs are multiplicative with turnover** — a 30× strategy needs 30× more gross alpha
   to break even.
3. **IC-weighting adapts but can overfit** — a 60-day IC window is one input, not the whole model.
4. **Regime filtering helps only if you define the regime ex ante** — post-hoc filtering is snooping.
5. **You built the whole zoo in R** — the same signal→weight rules, one backtester, and to *submit*
   any of these you'd wrap its logic in `on_day(day, features, prices, portfolio)` (see Missions 1/3).

*Next: Mission 5 — the same strategies on real market data.*
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "R", "language": "R", "name": "ir"},
                   "language_info": {"name": "R"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "missions",
                                   "mission_04_strategy_library", "notebook_r.ipynb"))
with open(out, "w") as f:
    json.dump(nb, f, indent=1); f.write("\n")
print("wrote", out, "with", len(cells), "cells")
