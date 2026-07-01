#!/usr/bin/env python3
"""Builds missions/mission_03_alpha_discovery/notebook_r.ipynb — the faithful R port."""
import json, os

def md(s):   return {"cell_type": "markdown", "metadata": {}, "source": _lines(s)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(s)}
def _lines(s):
    s = s.strip("\n"); parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]

cells = []

cells.append(md(r"""
# Mission 3: Alpha Discovery — R edition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/convexpi/missions/blob/main/missions/mission_03_alpha_discovery/notebook_r.ipynb)

**Learning objectives**
- Frame signal search as a multiple-testing problem and apply corrections
- Use walk-forward IC to validate a signal before committing to it
- Understand signal decay and its implications for turnover and transaction costs
- Build a multi-signal composite that is robust out-of-sample — and submit it in **R**

---

## Background

The synthetic market has several features. Most are pure noise; one or two are **planted alphas**
with a small but genuine predictive relationship to next-day returns. Your job is to find them
*without overfitting*:

- Test 10 features and pick the best, and the best looks good **even if all 10 are noise** (multiple
  testing).
- A signal that shines in-sample may be riding a regime that won't hold out-of-sample.
- Transaction costs can erase alpha that looked significant in a frictionless backtest.

We work through a discovery pipeline that addresses all three, then submit an R strategy scored by
the same engine as Python and Julia.
"""))

cells.append(md("## Part 0: Setup"))

cells.append(code(r"""
if (!requireNamespace("convexpi", quietly = TRUE)) {
  if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
  remotes::install_github("convexpi/convexpi-r", upgrade = "never")
}
library(convexpi)
if (!reticulate::py_module_available("convexpi.lab")) {
  reticulate::py_install("convexpi-lab", pip = TRUE)   # the market engine, via reticulate
}
set.seed(42)

m <- synthetic_market("train")
prices   <- m$prices
features <- m$features
feat_names <- names(features)
ret <- prices[-1, ] / prices[-nrow(prices), ] - 1     # (days-1) x stocks, next-day returns

# A compact long/short backtester (long top_k, short bottom_k, equal weight) for local checks.
simple_backtest <- function(signal_fn, top_k = 20, cost_bps = 10) {
  n <- ncol(prices); prev_w <- rep(0, n); daily <- numeric(nrow(ret))
  for (t in seq_len(nrow(ret))) {
    s <- signal_fn(t); ord <- order(s); w <- rep(0, n)
    w[tail(ord, top_k)] <-  1 / top_k; w[head(ord, top_k)] <- -1 / top_k
    daily[t] <- sum(w * ret[t, ]) - cost_bps / 1e4 * sum(abs(w - prev_w)); prev_w <- w
  }
  list(sharpe = mean(daily) / (sd(daily) + 1e-9) * sqrt(252), ann_ret = mean(daily) * 252)
}
cat("Ready.", nrow(prices), "days x", ncol(prices), "stocks |", length(feat_names), "features\n")
"""))

cells.append(md(r"""
---
## Part 1: Naive Search — and Why It Fails

The wrong way: compute the in-sample IC for every feature, rank, and crown the winner. We measure
the daily Spearman IC of each feature vs next-day returns, then summarise with a one-sample t-test of
"is the mean daily IC different from zero?".
"""))

cells.append(code(r"""
# Daily Spearman IC per feature -> matrix (days-1) x n_features
ics <- sapply(feat_names, function(nm) {
  fm <- features[[nm]]
  vapply(seq_len(nrow(ret)), function(t) cor(fm[t, ], ret[t, ], method = "spearman"), numeric(1))
})

n_obs   <- nrow(ics)
mean_ic <- colMeans(ics)
sd_ic   <- apply(ics, 2, sd)
ic_ir   <- mean_ic / sd_ic * sqrt(252)
t_stat  <- mean_ic / (sd_ic / sqrt(n_obs))
p_value <- 2 * pt(-abs(t_stat), df = n_obs - 1)          # two-sided one-sample t-test

summary_ic <- data.frame(feature = feat_names, mean_IC = mean_ic, IC_IR = ic_ir,
                         t_stat = t_stat, p_value = p_value, row.names = NULL)
print(summary_ic[order(-summary_ic$mean_IC), ], digits = 4, row.names = FALSE)
"""))

cells.append(md(r"""
**Exercise 1.1** — Which feature has the highest IS IC? Is its p-value below 0.05?

Now the crucial correction. With many features tested, some look significant by chance. Apply the
**Benjamini-Hochberg** false-discovery-rate correction (built into R as `p.adjust`).
"""))

cells.append(code(r"""
summary_ic$p_adj_BH   <- p.adjust(summary_ic$p_value, method = "BH")
summary_ic$significant <- summary_ic$p_adj_BH < 0.05

print(summary_ic[order(-summary_ic$mean_IC),
                 c("feature", "mean_IC", "IC_IR", "p_value", "p_adj_BH", "significant")],
      digits = 4, row.names = FALSE)
cat("\nSurvived FDR correction:",
    paste(summary_ic$feature[summary_ic$significant], collapse = ", "), "\n")
"""))

cells.append(md("**Exercise 1.2** — How many features survive FDR correction? Does the naive winner still look significant?"))

cells.append(md(r"""
---
## Part 2: Walk-Forward IC Validation

A single IS IC number conflates regime effects with genuine alpha. Walk-forward validation asks: is
the IC *consistently* positive across many sub-periods, or driven by one lucky stretch?
"""))

cells.append(code(r"""
walk_forward_ic <- function(fm, window = 120, step = 20) {
  T <- nrow(ret); starts <- integer(0); vals <- numeric(0); pos <- window
  while (pos + step <= T) {
    oos <- vapply(pos:(pos + step - 1),
                  function(t) cor(fm[t, ], ret[t, ], method = "spearman"), numeric(1))
    starts <- c(starts, pos); vals <- c(vals, mean(oos)); pos <- pos + step
  }
  data.frame(start = starts, mean_ic = vals)
}

op <- par(mfrow = c(length(feat_names), 1), mar = c(2.5, 4, 2, 1))
for (nm in feat_names) {
  wf <- walk_forward_ic(features[[nm]])
  n_pos <- sum(wf$mean_ic > 0)
  barplot(wf$mean_ic, col = ifelse(wf$mean_ic > 0, "steelblue", "tomato"), border = NA,
          main = sprintf("%s — positive in %d/%d windows", nm, n_pos, nrow(wf)),
          ylab = "OOS IC")
  abline(h = 0, lwd = 0.5)
}
par(op)
"""))

cells.append(md(r"""
**Exercise 2.1** — Which feature is positive in the most walk-forward windows? Does it match the IS
IC ranking?

**Exercise 2.2** — Is there a feature that looks good early in training but flat later? What does that
say about regime stationarity?
"""))

cells.append(md(r"""
---
## Part 3: Signal Decay and Turnover

Alpha decays: today's rank may not predict returns several days out. Decay tells you how often to
trade — and therefore your transaction-cost burden.
"""))

cells.append(code(r"""
signal_decay <- function(fm, max_lag = 10) {
  vapply(1:max_lag, function(lag) {
    mean(vapply(seq_len(nrow(ret) - lag + 1),
                function(t) cor(fm[t, ], ret[t + lag - 1, ], method = "spearman"), numeric(1)))
  }, numeric(1))
}

lags <- 1:10
decay_mat <- sapply(feat_names, function(nm) signal_decay(features[[nm]]))
matplot(lags, decay_mat, type = "b", pch = 19, lty = 1, lwd = 1.5,
        col = seq_along(feat_names), xlab = "Lag (days)", ylab = "Mean IC",
        main = "Signal decay: IC vs forward lag")
abline(h = 0, col = "grey", lwd = 0.5)
legend("topright", legend = feat_names, col = seq_along(feat_names), lty = 1, pch = 19, bty = "n")
"""))

cells.append(md(r"""
**Exercise 3.1** — At what lag does each signal's IC fall to ~zero? That's its **half-life** — it
sets your rebalance frequency.

**Exercise 3.2** — At 10 bps/trade and a 1-day half-life, roughly how much annual alpha (bps) do you
need to break even? *Hint: daily IC × √252 ≈ annualised IC-IR; multiply by signal vol.*
"""))

cells.append(md(r"""
---
## Part 4: A Robust Composite Signal

If several signals survive, combining them can raise IC-IR (their noise diversifies away). The key:
weight each by its **walk-forward IC-IR** (the estimate of its predictive power), not its IS IC.
Only positive contributors get weight.
"""))

cells.append(code(r"""
wf_icir <- sapply(feat_names, function(nm) {
  wf <- walk_forward_ic(features[[nm]])
  max(0, mean(wf$mean_ic) / (sd(wf$mean_ic) + 1e-9))
})
weights <- wf_icir / (sum(wf_icir) + 1e-9)
cat("Walk-forward IC-IR weights:\n")
for (nm in names(sort(weights, decreasing = TRUE))) {
  cat(sprintf("  %-8s %.3f  (raw IC-IR %.3f)\n", nm, weights[[nm]], wf_icir[[nm]]))
}

# Local check: backtest the z-scored, IC-IR-weighted composite in-sample.
composite_signal <- function(t) {
  s <- rep(0, ncol(prices))
  for (nm in feat_names) if (weights[[nm]] > 0) {
    raw <- features[[nm]][t, ]; raw[!is.finite(raw)] <- 0
    s <- s + weights[[nm]] * (raw - mean(raw)) / (sd(raw) + 1e-8)
  }
  s
}
r_comp <- simple_backtest(composite_signal, top_k = 20, cost_bps = 10)
cat(sprintf("\nComposite IS Sharpe = %.3f  |  IS Ann Ret = %.2f%%\n",
            r_comp$sharpe, 100 * r_comp$ann_ret))
"""))

cells.append(md(r"""
---
## Part 5: Submit to the Grader

Your `on_day` rebuilds the same IC-IR-weighted, z-scored composite from your fitted weights and
returns target weights. The grader scores it on the hidden holdout.

Create an API key at **convexpi.ai/settings/api-keys** (account menu → API keys) and set it below.
"""))

cells.append(code(r"""
pos_w <- weights[weights > 0]
composite_code <- sprintf('
on_day <- function(day, features, prices, portfolio) {
  w  <- c(%s)
  fn <- c(%s)
  s <- rep(0, length(prices))
  for (i in seq_along(fn)) {
    raw <- features[[fn[i]]]; raw[!is.finite(raw)] <- 0
    z <- (raw - mean(raw)) / (sd(raw) + 1e-8)      # cross-sectional z-score
    s <- s + w[i] * z
  }
  g <- sum(abs(s)); if (g > 0) s / g else s
}',
  paste(sprintf("%.6f", pos_w), collapse = ", "),
  paste(sprintf('"%s"', names(pos_w)), collapse = ", "))
cat(composite_code)

Sys.setenv(CONVEXPI_API_KEY = "cpk_...")    # <- your key
if (nchar(Sys.getenv("CONVEXPI_API_KEY")) > 8 && Sys.getenv("CONVEXPI_API_KEY") != "cpk_...") {
  submit("ic-ir-composite-r", composite_code)   # slug defaults to "demo-fall-2026"
} else {
  cat("\n(Set CONVEXPI_API_KEY above to submit, or paste the code into the web editor at\n",
      "https://www.convexpi.ai/compete/demo-fall-2026/submit)\n")
}
"""))

cells.append(md(r"""
---
## Part 6: Challenges

- **A (Easy):** Weight by *mean IC* instead of IC-IR. Which gives the better OOS Sharpe, and why?
- **B (Medium):** Some features flip sign OOS. If a feature's IC is negative over the last 60
  training days, zero its weight (starter below). Does it help robustness?
- **C (Medium):** Estimate each signal's optimal holding period (the lag maximising
  `IC × (1 - 2·TC/IC)`) and rebalance at that frequency instead of daily.
- **D (Hard):** Shrink each walk-forward IC toward zero by its estimation uncertainty before using it
  as a weight. Does shrinkage help OOS?
"""))

cells.append(code(r"""
# Challenge B starter — recency filter over the last 60 training days.
recency <- 60; T <- nrow(ret)
recent_ic <- sapply(feat_names, function(nm) {
  mean(vapply((T - recency):(T - 1),
              function(t) cor(features[[nm]][t, ], ret[t, ], method = "spearman"), numeric(1)))
})
for (nm in names(sort(recent_ic, decreasing = TRUE)))
  cat(sprintf("  %-8s recent IC %.4f  %s\n", nm, recent_ic[[nm]], if (recent_ic[[nm]] > 0) "keep" else "drop"))

filtered <- ifelse(recent_ic > 0, weights, 0)
if (sum(filtered) > 0) filtered <- filtered / sum(filtered)
cat("\nFiltered + renormalised weights:\n"); print(round(filtered, 3))
"""))

cells.append(md(r"""
---
## Wrap-up

1. **Multiple testing is unavoidable.** Test more than one hypothesis and you need a correction;
   Benjamini-Hochberg controls the false-discovery rate, naive p < 0.05 does not.
2. **Walk-forward consistency beats IS magnitude.** Positive in 80% of windows > a high IS IC from
   one lucky stretch.
3. **Signal decay sets your cost structure.** A short-lived signal needs frequent rebalancing; if
   TC > alpha you can't trade it profitably.
4. **Combine with humility.** Weight by *estimated* predictive power, and let shrinkage/recency
   filters pull toward the prior that most signals are noise.
5. **Same engine, any language.** Your R composite is scored exactly like the Python and Julia ones.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "R", "language": "R", "name": "ir"},
                   "language_info": {"name": "R"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "missions",
                                   "mission_03_alpha_discovery", "notebook_r.ipynb"))
with open(out, "w") as f:
    json.dump(nb, f, indent=1); f.write("\n")
print("wrote", out, "with", len(cells), "cells")
