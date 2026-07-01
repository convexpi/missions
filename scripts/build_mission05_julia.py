#!/usr/bin/env python3
"""Builds missions/mission_05_real_data/notebook_julia.ipynb — the faithful Julia port.

Real-data acquisition (yfinance + RealDataMarket) is bridged from Python via PyCall; every strategy
and all analysis run natively in Julia, reusing the Mission 4 zoo.
"""
import json, os

def md(s):   return {"cell_type": "markdown", "metadata": {}, "source": _lines(s)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(s)}
def _lines(s):
    s = s.strip("\n"); parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]

cells = []

cells.append(md(r"""
# Mission 5: Real-Data Lab — Julia edition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/convexpi/missions/blob/main/missions/mission_05_real_data/notebook_julia.ipynb)

**Learning objectives**
- Pull a real equity panel and run **Mission 4's strategy tournament** on actual market history
- Understand **survivorship bias** and **point-in-time (look-ahead)** discipline
- Replicate the IS/OOS **factor decay** documented in the Anomaly Graveyard
- See which factors hold up on real data vs the synthetic market

> **How this port works.** Fetching real prices (yfinance) and building the feature panel
> (`RealDataMarket`) is Python infrastructure with no clean, uniform R/Julia equivalent — so we
> **bridge it via `PyCall`** and pull the resulting arrays into Julia. Every strategy and all the
> analysis then run **natively in Julia**, reusing the zoo you built in Mission 4. The lesson here is
> real-data discipline, not data plumbing.
"""))

cells.append(md("## Part 0: Setup"))

cells.append(code(r"""
using Pkg
Pkg.add(url = "https://github.com/convexpi/ConvexPi.jl")
Pkg.add(["Statistics", "UnicodePlots", "PyCall"])
using ConvexPi
using Statistics, UnicodePlots, PyCall

# The data pipeline needs the Python engine (real-data extras) + yfinance.
try
    pyimport("yfinance"); pyimport("convexpi.lab.real_data")
catch
    run(`$(PyCall.python) -m pip install --quiet "convexpi-lab[real-data]" yfinance`)
end
println("Ready.")
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

We fetch adjusted closes for 50 liquid S&P 500 names (2010–2024) via yfinance. If the download fails
(offline), we fall back to a deterministic synthetic panel with the same shape — the rest of the
notebook is identical either way.
"""))

cells.append(code(r'''
py"""
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
"""
using_real = py"USING_REAL"
println(using_real ? "Downloaded real prices" : "Using synthetic fallback", ": ",
        py"prices_df.shape[0]", " days x ", py"prices_df.shape[1]", " stocks")
'''))

cells.append(md(r"""
---
## Part 3: Build the market and pull it into Julia

`RealDataMarket.from_prices()` turns the price frame into the same interface `Backtest`/`compare`
use — computing point-in-time price features (momentum, reversal, volatility, size). We build it in
Python, then pull the prices and feature matrices into native Julia arrays. `val_bm`/`qual_roe`
aren't derivable from prices, so value/quality strategies don't apply here.
"""))

cells.append(code(r"""
rd = pyimport("convexpi.lab.real_data")
mkt = rd.RealDataMarket.from_prices(py"prices_df",
                                    train_frac = 0.70, tc_bps = 10,
                                    load_fred = false, load_french = false)

# Pull numpy arrays into Julia.
function pull(split)
    F = Dict(string(k) => convert(Matrix{Float64}, v) for (k, v) in mkt.features(split))
    (P = convert(Matrix{Float64}, mkt.prices(split)), F = F)
end
TRAIN = pull("train"); TEST = pull("test"); ALL = pull("all")
feat_names = collect(keys(TRAIN.F))
println("split sizes — train: ", size(TRAIN.P, 1), "  test: ", size(TEST.P, 1),
        "  all: ", size(ALL.P, 1))
println("features: ", join(feat_names, ", "))
"""))

cells.append(code(r"""
# Native building blocks + backtester (same as Mission 4). FEAT/RET/PX = current window.
function zscore(x)
    v = isfinite.(x); sum(v) < 2 && return zeros(length(x))
    mu = mean(x[v]); s = std(x[v]); [isfinite(xi) ? (xi - mu) / (s + 1e-9) : 0.0 for xi in x]
end
function ls_weights(sig; q = 0.2, long_only = false)
    v = isfinite.(sig); sum(v) < 10 && return zeros(length(sig))
    lo = quantile(sig[v], q); hi = quantile(sig[v], 1 - q); w = zeros(length(sig))
    for i in eachindex(sig)
        isfinite(sig[i]) || continue
        if sig[i] >= hi; w[i] = 1.0 elseif !long_only && sig[i] <= lo; w[i] = -1.0 end
    end
    tot = sum(abs.(w)); tot > 0 ? w ./ tot : w
end
function run_metrics(strat; warmup = 252, tc_bps = 10)
    n = size(PX, 2); portfolio = zeros(n); daily = Float64[]; turn = Float64[]
    for t in (warmup+1):size(RET, 1)
        w = strat(t, portfolio); w[.!isfinite.(w)] .= 0.0
        traded = sum(abs.(w .- portfolio))
        push!(daily, sum(w .* RET[t, :]) - tc_bps / 1e4 * traded); push!(turn, traded); portfolio = w
    end
    eq = cumprod(1 .+ daily); peak = accumulate(max, eq); dd = eq ./ peak .- 1
    (cum = eq .- 1, sharpe = mean(daily) / (std(daily) + 1e-9) * sqrt(252),
     ann_return = mean(daily) * 252, max_dd = minimum(dd), turnover = mean(turn) * 252)
end
function set_window(w)
    global FEAT = w.F
    global RET = w.P[2:end, :] ./ w.P[1:end-1, :] .- 1
    global PX = w.P
end
set_window(TRAIN)
println("backtester ready")
"""))

cells.append(md(r"""
---
## Part 4: The tournament on real data

The **price-only** subset of the Mission 4 zoo — the strategies that work without value/quality
features. Run the same tournament, now on actual market history.
"""))

cells.append(code(r"""
STRATEGIES = [
  "equal_weight"   => (t, pf) -> fill(1 / size(PX, 2), size(PX, 2)),
  "momentum_12_1"  => (t, pf) -> ls_weights(FEAT["mom_12m"][t, :]),
  "momentum_3m"    => (t, pf) -> ls_weights(FEAT["mom_3m"][t, :]),
  "reversal_1w"    => (t, pf) -> ls_weights(FEAT["reversal_1w"][t, :]),
  "ts_momentum"    => function (t, pf)
      sig = FEAT["mom_12m"][t, :]; v = isfinite.(sig); n = sum(v)
      n == 0 && return zeros(length(sig))
      w = [v[i] ? sign(sig[i]) / n : 0.0 for i in eachindex(sig)]
      tot = sum(abs.(w)); tot > 0 ? w ./ tot : w
  end,
  "betting_against_beta" => (t, pf) -> ls_weights(-FEAT["vol_1m"][t, :]),
  "size_premium"   => (t, pf) -> ls_weights(-FEAT["size_cap"][t, :]),
  "inv_vol"        => function (t, pf)
      vol = FEAT["vol_1m"][t, :]; v = isfinite.(vol) .& (vol .> 0)
      sum(v) < 2 && return zeros(length(vol))
      w = zeros(length(vol)); w[v] .= 1 ./ vol[v]; tot = sum(w); tot > 0 ? w ./ tot : w
  end,
  "min_variance"   => function (t, pf)
      vol = FEAT["vol_1m"][t, :]; v = isfinite.(vol); sum(v) < 5 && return zeros(length(vol))
      thr = quantile(vol[v], 0.2); w = [v[i] && vol[i] <= thr ? 1.0 : 0.0 for i in eachindex(vol)]
      tot = sum(w); tot > 0 ? w ./ tot : w
  end,
  "trend_filter"   => function (t, pf)
      sig = FEAT["mom_12m"][t, :]; mean(sig[isfinite.(sig)]) > 0 ? ls_weights(sig) : zeros(size(PX, 2))
  end,
  "dual_momentum"  => function (t, pf)
      cs = FEAT["mom_12m"][t, :]; ls_weights([c > 0 ? c : NaN for c in cs]; long_only = true)
  end,
]
sd = Dict(STRATEGIES)

set_window(TRAIN)
res = [(name, run_metrics(fn)) for (name, fn) in STRATEGIES]
sort!(res, by = r -> -r[2].sharpe)
println(rpad("strategy", 22), rpad("Sharpe", 9), rpad("annRet%", 9), rpad("maxDD%", 9), "turnover")
for (name, m) in res
    println(rpad(name, 22), rpad(round(m.sharpe, digits = 3), 9),
            rpad(round(100 * m.ann_return, digits = 1), 9),
            rpad(round(100 * m.max_dd, digits = 1), 9), round(m.turnover, digits = 1))
end
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
`macro_lag ≥ 1`. (Enable `load_fred=true, load_french=true` when building the market to explore
Fama-French and yield-curve features; they're lagged for you.)

---
## Part 6: Replicating factor decay

The Anomaly Graveyard tracks whether anomalies survive out of sample. Split each strategy's history at
the IS/OOS boundary and compare Sharpe — a strategy that degrades sharply post-split suggests crowding
or risk exposure, not durable alpha.
"""))

cells.append(code(r"""
function sharpe_in_window(strat, w, from, to; warmup)
    set_window((P = w.P[from:to, :], F = Dict(k => v[from:to, :] for (k, v) in w.F)))
    run_metrics(strat; warmup = warmup).sharpe
end
split = size(TRAIN.P, 1); n_all = size(ALL.P, 1)
decay = ["momentum_12_1", "ts_momentum", "betting_against_beta", "inv_vol", "equal_weight"]
println(rpad("strategy", 22), rpad("IS", 9), rpad("OOS", 9), "decay")
for k in decay
    is_s  = sharpe_in_window(sd[k], ALL, 1, split; warmup = 252)
    oos_s = sharpe_in_window(sd[k], ALL, split, n_all; warmup = 63)
    println(rpad(k, 22), rpad(round(is_s, digits = 3), 9), rpad(round(oos_s, digits = 3), 9),
            round(is_s - oos_s, digits = 3))
end
set_window(TRAIN)
"""))

cells.append(md(r"""
---
## Part 7: Regime dependence

Momentum is famously fragile after sharp reversals. `trend_filter` switches momentum off during
down-trend regimes. Compare its equity curve to raw momentum.
"""))

cells.append(code(r"""
set_window(TRAIN)
raw = run_metrics(sd["momentum_12_1"]); filt = run_metrics(sd["trend_filter"])
plt = lineplot(raw.cum, name = "CS momentum (raw)", title = "Raw vs trend-filtered momentum (IS, real data)",
               xlabel = "day", ylabel = "cum ret", width = 72, height = 14)
lineplot!(plt, filt.cum, name = "+ trend filter")
display(plt)
println("raw: Sharpe ", round(raw.sharpe, digits = 2), "  maxDD ", round(100 * raw.max_dd, digits = 1),
        "%  |  filtered: Sharpe ", round(filt.sharpe, digits = 2), "  maxDD ", round(100 * filt.max_dd, digits = 1), "%")
"""))

cells.append(md(r"""
---
## Challenge

1. Rebuild the market with `load_french = true` and add `ValueTilt`/`QualityTilt` back to the
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
3. **Interop is a tool.** You pulled real data through Python and did the quant work in Julia — the
   same engine scores every language, and the same discipline applies to every dataset.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Julia 1.10", "language": "julia", "name": "julia-1.10"},
                   "language_info": {"name": "julia", "file_extension": ".jl", "mimetype": "application/julia"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "missions",
                                   "mission_05_real_data", "notebook_julia.ipynb"))
with open(out, "w") as f:
    json.dump(nb, f, indent=1); f.write("\n")
print("wrote", out, "with", len(cells), "cells")
