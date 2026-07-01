#!/usr/bin/env python3
"""Builds missions/mission_05_real_data/notebook_r.ipynb — the faithful R port.

Approach: real-data acquisition (yfinance + RealDataMarket) is bridged from Python via reticulate —
there is no clean, uniform native equivalent — but every strategy and all analysis run natively in R,
reusing the Mission 4 zoo. The lesson is real-data discipline, not data plumbing.
"""
import json, os

def md(s):   return {"cell_type": "markdown", "metadata": {}, "source": _lines(s)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(s)}
def _lines(s):
    s = s.strip("\n"); parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]

cells = []

cells.append(md(r"""
# Mission 5: Real-Data Lab — R edition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/convexpi/missions/blob/main/missions/mission_05_real_data/notebook_r.ipynb)

**Learning objectives**
- Pull a real equity panel and run **Mission 4's strategy tournament** on actual market history
- Understand **survivorship bias** and **point-in-time (look-ahead)** discipline
- Replicate the IS/OOS **factor decay** documented in the Anomaly Graveyard
- See which factors hold up on real data vs the synthetic market

> **How this port works.** Fetching real prices (yfinance) and building the feature panel
> (`RealDataMarket`) is Python infrastructure with no clean, uniform R/Julia equivalent — so we
> **bridge it via `reticulate`** and pull the resulting arrays into R. Every strategy and all the
> analysis then run **natively in R**, reusing the zoo you built in Mission 4. The lesson here is
> real-data discipline, not data plumbing.
"""))

cells.append(md("## Part 0: Setup"))

cells.append(code(r"""
if (!requireNamespace("convexpi", quietly = TRUE)) {
  if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
  remotes::install_github("convexpi/convexpi-r", upgrade = "never")
}
library(convexpi)
# The data pipeline needs the Python engine (with the real-data extras) + yfinance.
for (pkg in c("convexpi-lab[real-data]", "yfinance")) {
  mod <- if (grepl("yfinance", pkg)) "yfinance" else "convexpi.lab.real_data"
  if (!reticulate::py_module_available(mod)) reticulate::py_install(pkg, pip = TRUE)
}
set.seed(42)
cat("Ready.\n")
"""))

cells.append(md(r"""
---
## Part 1: Why real data?

The synthetic market from Missions 1–4 has a known structure: planted alphas, Gaussian returns,
stationary features. Real markets are messier — and that's the point.

| Property | Synthetic | Real |
|---|---|---|
| Return distribution | Gaussian | Fat-tailed, skewed |
| Feature stationarity | Yes | No — factors decay |
| Survivorship bias | None | Significant |
| Macro regimes | None | Recessions, rate cycles |
| Publication bias | None | Strategies get crowded |

**Survivorship-bias warning:** we use 50 *current* S&P 500 names. That overweights stocks that
*survived* and ignores those delisted or removed — which inflates most factor backtests. Acknowledge
it in any conclusion you draw.
"""))

cells.append(md(r"""
---
## Part 2: Load a real price panel (bridged from Python)

We fetch adjusted closes for 50 liquid S&P 500 names (2010–2024) via yfinance, run inside the bridged
Python. If the download fails (offline), we fall back to a deterministic synthetic panel with the same
shape — the rest of the notebook is identical either way.
"""))

cells.append(code(r"""
reticulate::py_run_string("
import yfinance as yf, pandas as pd, numpy as np
TICKERS = ['AAPL','MSFT','GOOGL','META','NVDA','ADBE','CRM','INTC','AMD','QCOM',
           'JPM','BAC','WFC','GS','MS','BLK','AXP','USB','PNC','COF',
           'JNJ','UNH','PFE','ABBV','MRK','TMO','DHR','ABT','BMY','AMGN',
           'AMZN','HD','MCD','NKE','SBUX','TGT','COST','LOW','KO','PEP',
           'XOM','CVX','BA','CAT','GE','MMM','HON','UPS','LMT','RTX']
try:
    raw = yf.download(TICKERS, start='2010-01-01', end='2024-01-01', auto_adjust=True, progress=False)
    prices_df = raw['Close'].dropna(how='all', axis=1).dropna(how='any', axis=0)
    prices_df = prices_df.loc[:, prices_df.isna().mean() < 0.05]
    USING_REAL = True
except Exception as e:
    from convexpi.lab import SyntheticMarket
    sm = SyntheticMarket(n_stocks=50, n_days=3500, seed=2010); p = sm.prices('all')
    idx = pd.date_range('2010-01-04', periods=p.shape[0], freq='B')
    prices_df = pd.DataFrame(p * 100, index=idx, columns=[f'SYN{i:02d}' for i in range(p.shape[1])])
    USING_REAL = False
")
using_real <- reticulate::py$USING_REAL
cat(if (using_real) "Downloaded real prices" else "Using synthetic fallback", ":",
    dim(reticulate::py$prices_df)[1], "days x", dim(reticulate::py$prices_df)[2], "stocks\n")
"""))

cells.append(md(r"""
---
## Part 3: Build the market and pull it into R

`RealDataMarket.from_prices()` turns the price frame into the same interface `Backtest`/`compare`
use — computing point-in-time price features (momentum, reversal, volatility, size). We build it in
Python, then pull the prices and feature matrices into native R objects. `val_bm`/`qual_roe` aren't
derivable from prices, so value/quality strategies don't apply here.
"""))

cells.append(code(r"""
rd <- reticulate::import("convexpi.lab.real_data")
mkt <- rd$RealDataMarket$from_prices(reticulate::py$prices_df,
                                     train_frac = 0.70, tc_bps = 10,
                                     load_fred = FALSE, load_french = FALSE)

# Pull numpy arrays into R (reticulate converts 2-D numpy -> R matrix, dict -> named list).
pull <- function(split) list(P = mkt$prices(split), F = mkt$features(split))
TRAIN <- pull("train"); TEST <- pull("test"); ALL <- pull("all")
feat_names <- names(TRAIN$F)
cat("split sizes — train:", nrow(TRAIN$P), " test:", nrow(TEST$P), " all:", nrow(ALL$P), "\n")
cat("features:", paste(feat_names, collapse = ", "), "\n")
"""))

cells.append(code(r"""
# Native building blocks + backtester (same as Mission 4). `F`,`R`,`P` = current window.
zscore <- function(x) { v <- is.finite(x); if (sum(v) < 2) return(rep(0, length(x)))
  ifelse(v, (x - mean(x[v])) / (sd(x[v]) + 1e-9), 0) }
ls_weights <- function(sig, q = 0.2, long_only = FALSE) {
  v <- is.finite(sig); if (sum(v) < 10) return(rep(0, length(sig)))
  lo <- quantile(sig[v], q); hi <- quantile(sig[v], 1 - q); w <- rep(0, length(sig))
  w[v & sig >= hi] <- 1; if (!long_only) w[v & sig <= lo] <- -1
  tot <- sum(abs(w)); if (tot > 0) w / tot else w
}
run_metrics <- function(strat, warmup = 252, tc_bps = 10) {
  n <- ncol(PX); portfolio <- rep(0, n); daily <- c(); turn <- c()
  for (t in (warmup + 1):nrow(RET)) {
    w <- strat(t, portfolio); w[!is.finite(w)] <- 0
    traded <- sum(abs(w - portfolio))
    daily <- c(daily, sum(w * RET[t, ]) - tc_bps / 1e4 * traded); turn <- c(turn, traded); portfolio <- w
  }
  eq <- cumprod(1 + daily); dd <- eq / cummax(eq) - 1; ann <- mean(daily) * 252
  list(cum = eq - 1, sharpe = mean(daily) / (sd(daily) + 1e-9) * sqrt(252),
       ann_return = ann, max_dd = min(dd), turnover = mean(turn) * 252)
}
set_window <- function(w) { FEAT <<- w$F; RET <<- w$P[-1, ] / w$P[-nrow(w$P), ] - 1; PX <<- w$P }
set_window(TRAIN)
cat("backtester ready\n")
"""))

cells.append(md(r"""
---
## Part 4: The tournament on real data

The **price-only** subset of the Mission 4 zoo — the strategies that work without value/quality
features. Run the same tournament, now on actual market history.
"""))

cells.append(code(r"""
STRATEGIES <- list(
  equal_weight   = function(t, pf) rep(1 / ncol(PX), ncol(PX)),
  momentum_12_1  = function(t, pf) ls_weights(FEAT[["mom_12m"]][t, ]),
  momentum_3m    = function(t, pf) ls_weights(FEAT[["mom_3m"]][t, ]),
  reversal_1w    = function(t, pf) ls_weights(FEAT[["reversal_1w"]][t, ]),
  ts_momentum    = function(t, pf) { sig <- FEAT[["mom_12m"]][t, ]; v <- is.finite(sig); n <- sum(v)
    if (n == 0) return(rep(0, length(sig))); w <- ifelse(v, sign(sig) / n, 0)
    tot <- sum(abs(w)); if (tot > 0) w / tot else w },
  betting_against_beta = function(t, pf) ls_weights(-FEAT[["vol_1m"]][t, ]),
  size_premium   = function(t, pf) ls_weights(-FEAT[["size_cap"]][t, ]),
  inv_vol        = function(t, pf) { vol <- FEAT[["vol_1m"]][t, ]; v <- is.finite(vol) & vol > 0
    if (sum(v) < 2) return(rep(0, length(vol))); w <- rep(0, length(vol)); w[v] <- 1 / vol[v]
    tot <- sum(w); if (tot > 0) w / tot else w },
  min_variance   = function(t, pf) { vol <- FEAT[["vol_1m"]][t, ]; v <- is.finite(vol)
    if (sum(v) < 5) return(rep(0, length(vol))); thr <- quantile(vol[v], 0.2)
    w <- ifelse(v & vol <= thr, 1, 0); tot <- sum(w); if (tot > 0) w / tot else w },
  trend_filter   = function(t, pf) { sig <- FEAT[["mom_12m"]][t, ]
    if (mean(sig[is.finite(sig)]) > 0) ls_weights(sig) else rep(0, ncol(PX)) },
  dual_momentum  = function(t, pf) { cs <- FEAT[["mom_12m"]][t, ]; ls_weights(ifelse(cs > 0, cs, NA), long_only = TRUE) }
)

set_window(TRAIN)
res <- do.call(rbind, lapply(names(STRATEGIES), function(k) {
  m <- run_metrics(STRATEGIES[[k]])
  data.frame(strategy = k, sharpe = m$sharpe, ann_return = m$ann_return,
             max_dd = m$max_dd, turnover = m$turnover)
}))
res <- res[order(-res$sharpe), ]
print(res, digits = 3, row.names = FALSE)

o <- res[order(res$sharpe), ]
barplot(o$sharpe, names.arg = o$strategy, horiz = TRUE, las = 1, cex.names = 0.6,
        col = ifelse(o$sharpe > 0, "#2ecc71", "#e74c3c"),
        main = "Tournament — real data, IS Sharpe (10 bps)", xlab = "Sharpe")
abline(v = 0, lwd = 0.8)
"""))

cells.append(md(r"""
**Discussion:** Compare this to Mission 4's synthetic tournament. Which strategies improved on real
data, and which collapsed? Momentum and low-risk (BAB / inv-vol / min-variance) usually travel better
to real data than the synthetic market's planted alphas.

---
## Part 5: The point-in-time trap

Real backtests are often inflated by **look-ahead bias** in macro data: FRED revises history, and GDP
/ inflation are released weeks late. `RealDataMarket` enforces point-in-time by lagging macro features
(`macro_lag`). The discipline: **never trade on data you couldn't have had that morning** — always
`macro_lag ≥ 1`. (Enable `load_fred=TRUE, load_french=TRUE` when building the market to explore
Fama-French and yield-curve features; they're lagged for you.)

---
## Part 6: Replicating factor decay

The Anomaly Graveyard tracks whether anomalies survive out of sample. Split each strategy's history at
the IS/OOS boundary and compare Sharpe — a strategy that degrades sharply post-split suggests crowding
or risk exposure, not durable alpha.
"""))

cells.append(code(r"""
sharpe_in_window <- function(strat, w, from, to, warmup) {
  set_window(list(P = w$P[from:to, ], F = lapply(w$F, function(m) m[from:to, ])))
  run_metrics(strat, warmup = warmup)$sharpe
}
split <- nrow(TRAIN$P); n_all <- nrow(ALL$P)
decay <- c("momentum_12_1", "ts_momentum", "betting_against_beta", "inv_vol", "equal_weight")
dec <- do.call(rbind, lapply(decay, function(k) {
  is_s  <- sharpe_in_window(STRATEGIES[[k]], ALL, 1, split, warmup = 252)
  oos_s <- sharpe_in_window(STRATEGIES[[k]], ALL, split, n_all, warmup = 63)
  data.frame(strategy = k, IS = is_s, OOS = oos_s, decay = is_s - oos_s)
}))
set_window(TRAIN)
print(dec, digits = 3, row.names = FALSE)
barplot(t(as.matrix(dec[, c("IS", "OOS")])), beside = TRUE, names.arg = dec$strategy,
        las = 2, cex.names = 0.6, col = c("steelblue", "coral"),
        legend.text = c("IS", "OOS"), main = "IS vs OOS factor decay — real data")
abline(h = 0, lwd = 0.8)
"""))

cells.append(md(r"""
---
## Part 7: Regime dependence

Momentum is famously fragile after sharp reversals. `trend_filter` switches momentum off during
down-trend regimes. Compare its equity curve to raw momentum.
"""))

cells.append(code(r"""
set_window(TRAIN)
raw  <- run_metrics(STRATEGIES[["momentum_12_1"]])
filt <- run_metrics(STRATEGIES[["trend_filter"]])
plot(raw$cum, type = "l", col = "steelblue", lwd = 1.5,
     ylim = range(c(raw$cum, filt$cum)), xlab = "trading day", ylab = "cumulative return",
     main = "Raw vs trend-filtered momentum (IS, real data)")
lines(filt$cum, col = "orange", lwd = 1.5); abline(h = 0, lty = 2, lwd = 0.6)
legend("topleft", c("CS momentum (raw)", "+ trend filter"),
       col = c("steelblue", "orange"), lty = 1, bty = "n")
cat(sprintf("raw: Sharpe %.2f  maxDD %.1f%%  |  filtered: Sharpe %.2f  maxDD %.1f%%\n",
            raw$sharpe, 100 * raw$max_dd, filt$sharpe, 100 * filt$max_dd))
"""))

cells.append(md(r"""
---
## Challenge

1. Rebuild the market with `load_french = TRUE` and add `ValueTilt`/`QualityTilt` back to the
   tournament using the Fama-French factor features. Do value/quality survive on real data?
2. Reproduce the Anomaly Graveyard's momentum decay: split at a publication-era year (e.g. 2015) and
   compare pre/post Sharpe.
3. Net every strategy of costs at 0/10/20/50 bps (Mission 8) and re-rank. Which "edges" survive real
   frictions?

Publish your findings to **[/projects](https://convexpi.ai/projects)**.

---
## Wrap-up

1. **Real data decays.** Synthetic planted alphas are stationary; real factors crowd and fade — the
   IS/OOS gap is bigger and more honest.
2. **Survivorship and look-ahead inflate everything.** Current-index universes and un-lagged macro
   data manufacture edges that don't exist.
3. **Interop is a tool.** You pulled real data through Python and did the quant work in R — the same
   engine scores every language, and the same discipline applies to every dataset.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "R", "language": "R", "name": "ir"},
                   "language_info": {"name": "R"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "missions",
                                   "mission_05_real_data", "notebook_r.ipynb"))
with open(out, "w") as f:
    json.dump(nb, f, indent=1); f.write("\n")
print("wrote", out, "with", len(cells), "cells")
