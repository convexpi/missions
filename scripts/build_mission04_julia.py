#!/usr/bin/env python3
"""Builds missions/mission_04_strategy_library/notebook_julia.ipynb — the faithful Julia port."""
import json, os

def md(s):   return {"cell_type": "markdown", "metadata": {}, "source": _lines(s)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(s)}
def _lines(s):
    s = s.strip("\n"); parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]

cells = []

cells.append(md(r"""
# Mission 4: The Strategy Library — Julia edition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/convexpi/missions/blob/main/missions/mission_04_strategy_library/notebook_julia.ipynb)

**Learning objectives**
- Understand the economic intuition behind canonical quant strategies (momentum, value, quality,
  size, risk-based)
- **Build the strategy zoo yourself in Julia** and run a tournament across all of them
- Diagnose *why* strategies fail: crowding, transaction costs, regimes
- Build your own composite by IC/rank combination
- Confront IS vs OOS — even canonical strategies overfit when you pick the in-sample winner

> The Python mission calls a built-in strategy library. Here you **implement** that library natively
> in Julia — each strategy is the same signal→weight rule, run through one backtester. The lessons
> (which family wins, the cost tax, the IS/OOS gap) are identical.
"""))

cells.append(md("## Part 0: Setup"))

cells.append(code(r"""
using Pkg
Pkg.add(url = "https://github.com/convexpi/ConvexPi.jl")
Pkg.add(["Statistics", "StatsBase", "UnicodePlots", "PyCall"])
using ConvexPi
using Statistics, StatsBase, UnicodePlots, PyCall

try
    pyimport("convexpi.lab")
catch
    run(`$(PyCall.python) -m pip install --quiet convexpi-lab`)
end

# Load both splits: train = in-sample, test = the hidden holdout (peek only at the very end).
tr = synthetic_market("train"); te = synthetic_market("test")
mk(m) = (P = m.prices, F = m.features, R = m.prices[2:end, :] ./ m.prices[1:end-1, :] .- 1)
TRAIN = mk(tr); TEST = mk(te)
println("train: ", size(TRAIN.P, 1), " days | test: ", size(TEST.P, 1),
        " days | features: ", join(keys(TRAIN.F), ", "))
"""))

cells.append(md(r"""
---
## Part 1: The building blocks

Every strategy turns a cross-sectional signal into weights with two primitives: a **cross-sectional
z-score**, and a **quintile long/short** (long the top 20%, short the bottom 20%, gross exposure 1).
"""))

cells.append(code(r"""
function zscore(x)
    v = isfinite.(x); sum(v) < 2 && return zeros(length(x))
    mu = mean(x[v]); s = std(x[v])
    [isfinite(xi) ? (xi - mu) / (s + 1e-9) : 0.0 for xi in x]
end

# Long top quintile (+1), short bottom quintile (-1), normalise to gross 1.
function ls_weights(sig; q = 0.2, long_only = false)
    v = isfinite.(sig); sum(v) < 10 && return zeros(length(sig))
    lo = quantile(sig[v], q); hi = quantile(sig[v], 1 - q)
    w = zeros(length(sig))
    for i in eachindex(sig)
        isfinite(sig[i]) || continue
        if sig[i] >= hi
            w[i] = 1.0
        elseif !long_only && sig[i] <= lo
            w[i] = -1.0
        end
    end
    tot = sum(abs.(w)); tot > 0 ? w ./ tot : w
end

function rank_corr(x, y)
    v = isfinite.(x) .& isfinite.(y); sum(v) < 5 && return 0.0
    corspearman(x[v], y[v])
end
println("primitives ready")
"""))

cells.append(md(r"""
---
## Part 2: The strategy zoo

Eighteen canonical strategies across five families. Each is a function `(t, portfolio)` that reads
the market at day `t` (via the globals `F`, `R`, `P` — set to whichever split we're running) and
returns target weights. We keep them in an ordered list so the tournament is reproducible.
"""))

cells.append(code(r"""
# `F`, `R`, `P` are the current split's features / returns / prices (set before each run).
F = TRAIN.F; R = TRAIN.R; P = TRAIN.P

STRATEGIES = [
  "equal_weight"   => (t, pf) -> fill(1 / size(P, 2), size(P, 2)),                  # baseline
  "momentum_12_1"  => (t, pf) -> ls_weights(F["mom_12m"][t, :]),                    # Jegadeesh-Titman
  "momentum_3m"    => (t, pf) -> ls_weights(F["mom_3m"][t, :]),
  "momentum_1m"    => (t, pf) -> ls_weights(F["mom_1m"][t, :]),
  "reversal_1w"    => (t, pf) -> ls_weights(F["reversal_1w"][t, :]),                # short-term contrarian
  "ts_momentum"    => function (t, pf)                                              # time-series momentum
      sig = F["mom_12m"][t, :]; v = isfinite.(sig); n = sum(v)
      n == 0 && return zeros(length(sig))
      w = [v[i] ? sign(sig[i]) / n : 0.0 for i in eachindex(sig)]
      tot = sum(abs.(w)); tot > 0 ? w ./ tot : w
  end,
  "value_bm"           => (t, pf) -> ls_weights(F["val_bm"][t, :]),                # Fama-French value
  "value_bm_long_only" => (t, pf) -> ls_weights(F["val_bm"][t, :]; long_only = true),
  "quality_roe"    => (t, pf) -> ls_weights(F["qual_roe"][t, :]),                  # Novy-Marx quality
  "betting_against_beta" => (t, pf) -> ls_weights(-F["vol_1m"][t, :]),             # Frazzini-Pedersen
  "size_premium"   => (t, pf) -> ls_weights(-F["size_cap"][t, :]),                 # Banz small-cap
  "fama_french_3"  => (t, pf) ->                                                    # SMB + HML blend
      ls_weights(0.5 .* zscore(F["val_bm"][t, :]) .+ 0.5 .* zscore(-F["size_cap"][t, :])),
  "multi_factor_rank" => (t, pf) ->                                                 # rank-sum blend
      ls_weights((zscore(F["mom_12m"][t, :]) .+ zscore(F["val_bm"][t, :]) .+ zscore(F["qual_roe"][t, :])) ./ 3),
  "ic_weighted"    => function (t, pf)                                              # rolling IC-weighted
      sigs = ["mom_12m", "val_bm", "qual_roe"]; lo = max(1, t - 60)
      icm = [mean([rank_corr(F[f][s, :], R[s, :]) for s in lo:(t - 1)]) for f in sigs]
      w = max.(0.05, icm); w = w ./ sum(w)
      comp = zeros(size(P, 2))
      for i in eachindex(sigs); comp .+= w[i] .* zscore(F[sigs[i]][t, :]); end
      ls_weights(comp)
  end,
  "inv_vol"        => function (t, pf)                                              # risk parity (long-only)
      vol = F["vol_1m"][t, :]; v = isfinite.(vol) .& (vol .> 0)
      sum(v) < 2 && return zeros(length(vol))
      w = zeros(length(vol)); w[v] .= 1 ./ vol[v]; tot = sum(w); tot > 0 ? w ./ tot : w
  end,
  "min_variance"   => function (t, pf)                                             # lowest-vol quintile
      vol = F["vol_1m"][t, :]; v = isfinite.(vol); sum(v) < 5 && return zeros(length(vol))
      thr = quantile(vol[v], 0.2); w = [v[i] && vol[i] <= thr ? 1.0 : 0.0 for i in eachindex(vol)]
      tot = sum(w); tot > 0 ? w ./ tot : w
  end,
  "dual_momentum"  => function (t, pf)                                             # cross-sec gated by absolute
      cs = F["mom_12m"][t, :]; ls_weights([c > 0 ? c : NaN for c in cs]; long_only = true)
  end,
  "trend_filter"   => function (t, pf)                                             # momentum, flat when trend down
      sig = F["mom_12m"][t, :]; mean(sig[isfinite.(sig)]) > 0 ? ls_weights(sig) : zeros(size(P, 2))
  end,
]
println("registered ", length(STRATEGIES), " strategies")
"""))

cells.append(md(r"""
---
## Part 3: The tournament

`run_metrics` runs one strategy through a daily backtester (10 bps one-way cost, 63-day warmup) and
reports the seven metrics you'd judge a strategy on. We run the whole zoo in-sample.
"""))

cells.append(code(r"""
function run_metrics(strat; warmup = 63, tc_bps = 10)
    n = size(P, 2); portfolio = zeros(n); daily = Float64[]; turn = Float64[]
    for t in (warmup + 1):size(R, 1)
        w = strat(t, portfolio); w[.!isfinite.(w)] .= 0.0
        traded = sum(abs.(w .- portfolio))
        push!(daily, sum(w .* R[t, :]) - tc_bps / 1e4 * traded); push!(turn, traded); portfolio = w
    end
    eq = cumprod(1 .+ daily); peak = accumulate(max, eq); dd = eq ./ peak .- 1; maxdd = minimum(dd)
    ann_ret = mean(daily) * 252
    (cum = eq .- 1, sharpe = mean(daily) / (std(daily) + 1e-9) * sqrt(252), annual_return = ann_ret,
     max_drawdown = maxdd, calmar = ann_ret / (abs(maxdd) + 1e-9),
     annual_turnover = mean(turn) * 252, hit_rate = mean(daily .> 0))
end

F = TRAIN.F; R = TRAIN.R; P = TRAIN.P
results = [(name, run_metrics(fn)) for (name, fn) in STRATEGIES]
sort!(results, by = r -> -r[2].sharpe)
println(rpad("strategy", 22), rpad("Sharpe", 9), rpad("annRet%", 9), rpad("maxDD%", 9), rpad("turnover", 10), "hit%")
for (name, m) in results
    println(rpad(name, 22), rpad(round(m.sharpe, digits = 3), 9),
            rpad(round(100 * m.annual_return, digits = 1), 9),
            rpad(round(100 * m.max_drawdown, digits = 1), 9),
            rpad(round(m.annual_turnover, digits = 1), 10), round(100 * m.hit_rate, digits = 1))
end
"""))

cells.append(code(r"""
# A horizontal view of the tournament Sharpes.
names_sorted = [r[1] for r in results]; sh = [r[2].sharpe for r in results]
barplot(reverse(names_sorted), reverse(sh .- minimum(sh) .+ 1e-6),
        title = "Tournament Sharpe (shifted ≥0 for display)", width = 50)
"""))

cells.append(md(r"""
**Discussion:** Which family dominates? Note the `turnover` column — a strategy turning over 40×/yr
at 10 bps pays ~8%/yr in costs before any alpha. How do the "smart" strategies beat `equal_weight`?

---
## Part 4: Anatomy of a strategy

Open up cross-sectional momentum: how does `mom_12m` map to weights on a single day? A clean
long/short is roughly market-neutral (net exposure ≈ 0).
"""))

cells.append(code(r"""
t0 = 300
mom_fn = STRATEGIES[2][2]              # momentum_12_1
w = mom_fn(t0, zeros(size(P, 2)))
println("Long ", count(>(0), w), " | Short ", count(<(0), w),
        " | net exposure ", round(sum(w), digits = 4), " (≈0)")
display(histogram(w[w .!= 0], nbins = 25, title = "Weight distribution (day 300)", xlabel = "weight"))
"""))

cells.append(md(r"""
---
## Part 5: Cumulative returns — when each approach earns

Equity curves show *when* each strategy makes its money — and how correlated they are.
"""))

cells.append(code(r"""
focus = ["equal_weight", "momentum_12_1", "value_bm", "quality_roe",
         "betting_against_beta", "ic_weighted", "fama_french_3"]
sd = Dict(STRATEGIES)
plt = nothing
for (i, k) in enumerate(focus)
    cum = run_metrics(sd[k]).cum
    if i == 1
        global plt = lineplot(cum, name = k, title = "Cumulative returns — IS (10 bps)",
                              xlabel = "day", ylabel = "cum ret", width = 72, height = 15)
    else
        lineplot!(plt, cum, name = k)
    end
end
plt
"""))

cells.append(md(r"""
---
## Part 6: The transaction-cost tax

High-turnover strategies look great at 0 bps and collapse under realistic costs. Sweep the cost and
watch the ranking change.
"""))

cells.append(code(r"""
tc_levels = [0, 5, 10, 20, 30]
focus_keys = ["equal_weight", "momentum_12_1", "momentum_3m", "value_bm",
              "quality_roe", "ic_weighted", "inv_vol"]
plt2 = nothing
for (i, k) in enumerate(focus_keys)
    ys = [run_metrics(sd[k]; tc_bps = tc).sharpe for tc in tc_levels]
    println(rpad(k, 16), join([rpad(round(y, digits = 2), 8) for y in ys]))
    if i == 1
        global plt2 = lineplot(tc_levels, ys, name = k, title = "TC sensitivity",
                               xlabel = "bps/side", ylabel = "Sharpe", width = 72, height = 14)
    else
        lineplot!(plt2, tc_levels, ys, name = k)
    end
end
plt2
"""))

cells.append(md(r"""
---
## Part 7: Build your own composite

Combine any signals by z-scoring and blending, then take the quintile long/short. Try adding or
dropping a signal, or inverting one (e.g. `size_cap` for a small-cap tilt).
"""))

cells.append(code(r"""
function multi_factor(signals; invert = String[], long_only = false)
    (t, pf) -> begin
        comp = zeros(size(P, 2))
        for f in signals
            z = zscore(F[f][t, :]); f in invert && (z = -z); comp .+= z
        end
        ls_weights(comp ./ length(signals); long_only = long_only)
    end
end

mine = multi_factor(["mom_12m", "qual_roe", "val_bm"])     # EDIT THIS
m = run_metrics(mine)
println("your composite  Sharpe=", round(m.sharpe, digits = 3),
        "  annRet=", round(100 * m.annual_return, digits = 2), "%  turnover=",
        round(m.annual_turnover, digits = 1), "x  maxDD=", round(100 * m.max_drawdown, digits = 1), "%")
"""))

cells.append(md(r"""
---
## Part 8: IS vs OOS — the reality check

Everything above was in-sample. Take the **top-5 IS strategies** and run them on the hidden holdout.
OOS Sharpe is almost always lower — the gap is overfitting, even for canonical strategies. **You look
at OOS once.**
"""))

cells.append(code(r"""
top5 = [r[1] for r in results[1:5]]

F = TRAIN.F; R = TRAIN.R; P = TRAIN.P
is_sh = [run_metrics(sd[k]).sharpe for k in top5]
F = TEST.F;  R = TEST.R;  P = TEST.P           # switch to the holdout — the one peek
oos_sh = [run_metrics(sd[k]).sharpe for k in top5]
F = TRAIN.F; R = TRAIN.R; P = TRAIN.P          # restore

println(rpad("strategy", 22), rpad("IS Sharpe", 12), "OOS Sharpe")
for i in eachindex(top5)
    println(rpad(top5[i], 22), rpad(round(is_sh[i], digits = 3), 12), round(oos_sh[i], digits = 3))
end
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
for k in ["momentum_12_1", "trend_filter"]
    m = run_metrics(sd[k])
    println(rpad(k, 14), " Sharpe=", round(m.sharpe, digits = 3),
            "  maxDD=", round(100 * m.max_drawdown, digits = 1), "%  Calmar=", round(m.calmar, digits = 3))
end
"""))

cells.append(md(r"""
---
## Wrap-up

1. **No single strategy dominates all regimes** — the IS winner is rarely the OOS winner.
2. **Transaction costs are multiplicative with turnover** — a 30× strategy needs 30× more gross alpha
   to break even.
3. **IC-weighting adapts but can overfit** — a 60-day IC window is one input, not the whole model.
4. **Regime filtering helps only if you define the regime ex ante** — post-hoc filtering is snooping.
5. **You built the whole zoo in Julia** — the same signal→weight rules, one backtester, and to
   *submit* any of these you'd wrap its logic in `on_day(day, features, prices, portfolio)` (see
   Missions 1/3).

*Next: Mission 5 — the same strategies on real market data.*
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Julia 1.10", "language": "julia", "name": "julia-1.10"},
                   "language_info": {"name": "julia", "file_extension": ".jl", "mimetype": "application/julia"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "missions",
                                   "mission_04_strategy_library", "notebook_julia.ipynb"))
with open(out, "w") as f:
    json.dump(nb, f, indent=1); f.write("\n")
print("wrote", out, "with", len(cells), "cells")
