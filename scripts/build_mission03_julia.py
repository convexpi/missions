#!/usr/bin/env python3
"""Builds missions/mission_03_alpha_discovery/notebook_julia.ipynb — the faithful Julia port."""
import json, os

def md(s):   return {"cell_type": "markdown", "metadata": {}, "source": _lines(s)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(s)}
def _lines(s):
    s = s.strip("\n"); parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]

cells = []

cells.append(md(r"""
# Mission 3: Alpha Discovery — Julia edition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/convexpi/missions/blob/main/missions/mission_03_alpha_discovery/notebook_julia.ipynb)

**Learning objectives**
- Frame signal search as a multiple-testing problem and apply corrections
- Use walk-forward IC to validate a signal before committing to it
- Understand signal decay and its implications for turnover and transaction costs
- Build a multi-signal composite that is robust out-of-sample — and submit it in **Julia**

---

## Background

The synthetic market has several features. Most are pure noise; one or two are **planted alphas**
with a small but genuine predictive relationship to next-day returns. Your job is to find them
*without overfitting*:

- Test 10 features and pick the best, and the best looks good **even if all 10 are noise** (multiple
  testing).
- A signal that shines in-sample may be riding a regime that won't hold out-of-sample.
- Transaction costs can erase alpha that looked significant in a frictionless backtest.

We work through a discovery pipeline that addresses all three, then submit a Julia strategy scored by
the same engine as Python and R.
"""))

cells.append(md("## Part 0: Setup"))

cells.append(code(r"""
using Pkg
Pkg.add(url = "https://github.com/convexpi/ConvexPi.jl")
Pkg.add(["StatsBase", "UnicodePlots", "PyCall", "Distributions"])
using ConvexPi
using Statistics, LinearAlgebra, StatsBase, UnicodePlots, PyCall, Distributions
import Random; Random.seed!(42)

try
    pyimport("convexpi.lab")
catch
    run(`$(PyCall.python) -m pip install --quiet convexpi-lab`)  # the market engine, via PyCall
end

m = synthetic_market("train")
prices = m.prices
features = m.features
feat_names = collect(keys(features))
ret = prices[2:end, :] ./ prices[1:end-1, :] .- 1     # (days-1) x stocks, next-day returns

# A compact long/short backtester (long top_k, short bottom_k) for local checks.
function simple_backtest(signal_fn; top_k = 20, cost_bps = 10)
    n = size(prices, 2); prev_w = zeros(n); daily = zeros(size(ret, 1))
    for t in 1:size(ret, 1)
        s = signal_fn(t); ord = sortperm(s); w = zeros(n)
        w[ord[end-top_k+1:end]] .= 1 / top_k; w[ord[1:top_k]] .= -1 / top_k
        daily[t] = sum(w .* ret[t, :]) - cost_bps / 1e4 * sum(abs.(w .- prev_w)); prev_w = w
    end
    (sharpe = mean(daily) / (std(daily) + 1e-9) * sqrt(252), ann_ret = mean(daily) * 252)
end
println("Ready. ", size(prices, 1), " days x ", size(prices, 2), " stocks | ",
        length(feat_names), " features")
"""))

cells.append(md(r"""
---
## Part 1: Naive Search — and Why It Fails

The wrong way: compute the in-sample IC for every feature, rank, and crown the winner. We measure the
daily Spearman IC of each feature vs next-day returns, then summarise with a one-sample t-test of
"is the mean daily IC different from zero?".
"""))

cells.append(code(r"""
ic_of(nm) = [corspearman(features[nm][t, :], ret[t, :]) for t in 1:size(ret, 1)]
ics = Dict(nm => ic_of(nm) for nm in feat_names)
n_obs = size(ret, 1)

# one-sample t-test p-value per feature
function summarise(nm)
    v = ics[nm]; m_ic = mean(v); s_ic = std(v)
    t = m_ic / (s_ic / sqrt(n_obs))
    p = 2 * ccdf(TDist(n_obs - 1), abs(t))
    (feature = nm, mean_IC = m_ic, IC_IR = m_ic / s_ic * sqrt(252), t_stat = t, p_value = p)
end
rows = [summarise(nm) for nm in feat_names]
sort!(rows, by = r -> -r.mean_IC)
for r in rows
    println(rpad(r.feature, 9), " mean_IC=", rpad(round(r.mean_IC, digits = 4), 9),
            " IC-IR=", rpad(round(r.IC_IR, digits = 2), 7), " p=", round(r.p_value, digits = 4))
end
"""))

cells.append(md(r"""
**Exercise 1.1** — Which feature has the highest IS IC? Is its p-value below 0.05?

Now the crucial correction. With many features tested, some look significant by chance. Apply the
**Benjamini-Hochberg** false-discovery-rate correction (a few lines, since it isn't in Base).
"""))

cells.append(code(r"""
function bh_adjust(p::Vector{Float64})
    n = length(p); o = sortperm(p); adj = zeros(n); prev = 1.0
    for k in n:-1:1
        prev = min(prev, p[o[k]] * n / k)
        adj[o[k]] = prev
    end
    adj
end

pvals = [r.p_value for r in rows]
padj = bh_adjust(pvals)
for (r, pa) in zip(rows, padj)
    println(rpad(r.feature, 9), " p=", rpad(round(r.p_value, digits = 4), 8),
            " p_adj_BH=", rpad(round(pa, digits = 4), 8), pa < 0.05 ? "  significant" : "")
end
println("\nSurvived FDR: ",
        join([rows[i].feature for i in eachindex(rows) if padj[i] < 0.05], ", "))
"""))

cells.append(md("**Exercise 1.2** — How many features survive FDR correction? Does the naive winner still look significant?"))

cells.append(md(r"""
---
## Part 2: Walk-Forward IC Validation

A single IS IC number conflates regime effects with genuine alpha. Walk-forward validation asks: is
the IC *consistently* positive across many sub-periods, or driven by one lucky stretch?
"""))

cells.append(code(r"""
function walk_forward_ic(fm; window = 120, step = 20)
    T = size(ret, 1); starts = Int[]; vals = Float64[]; pos = window
    while pos + step <= T
        oos = [corspearman(fm[t, :], ret[t, :]) for t in pos:(pos + step - 1)]
        push!(starts, pos); push!(vals, mean(oos)); pos += step
    end
    (starts = starts, vals = vals)
end

for nm in feat_names
    wf = walk_forward_ic(features[nm])
    npos = count(>(0), wf.vals)
    display(lineplot(wf.starts, wf.vals, title = "$nm — positive in $npos/$(length(wf.vals)) windows",
                     xlabel = "window start", ylabel = "OOS IC", width = 70, height = 8))
end
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
signal_decay(fm; max_lag = 10) =
    [mean([corspearman(fm[t, :], ret[t + lag - 1, :]) for t in 1:(size(ret, 1) - lag + 1)])
     for lag in 1:max_lag]

lags = 1:10
plt = lineplot(collect(lags), signal_decay(features[feat_names[1]]), name = feat_names[1],
               title = "Signal decay: IC vs forward lag", xlabel = "lag (days)",
               ylabel = "mean IC", width = 70, height = 12)
for nm in feat_names[2:end]
    lineplot!(plt, collect(lags), signal_decay(features[nm]), name = nm)
end
plt
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
function icir_of(nm)
    w = walk_forward_ic(features[nm])
    max(0.0, mean(w.vals) / (std(w.vals) + 1e-9))
end
wf_icir = Dict(nm => icir_of(nm) for nm in feat_names)
tot = sum(values(wf_icir))
weights = Dict(nm => wf_icir[nm] / (tot + 1e-9) for nm in feat_names)

println("Walk-forward IC-IR weights:")
for nm in sort(feat_names, by = n -> -weights[n])
    println("  ", rpad(nm, 9), round(weights[nm], digits = 3),
            "  (raw IC-IR ", round(wf_icir[nm], digits = 3), ")")
end

function composite_signal(t)
    s = zeros(size(prices, 2))
    for nm in feat_names
        weights[nm] > 0 || continue
        raw = copy(features[nm][t, :]); raw[.!isfinite.(raw)] .= 0.0
        s .+= weights[nm] .* (raw .- mean(raw)) ./ (std(raw) + 1e-8)
    end
    s
end
rc = simple_backtest(composite_signal; top_k = 20, cost_bps = 10)
println("\nComposite IS Sharpe = ", round(rc.sharpe, digits = 3),
        "  |  IS Ann Ret = ", round(100 * rc.ann_ret, digits = 2), "%")
"""))

cells.append(md(r"""
---
## Part 5: Submit to the Grader

Your `on_day` rebuilds the same IC-IR-weighted, z-scored composite from your fitted weights and
returns target weights. The grader scores it on the hidden holdout.

Create an API key at **convexpi.ai/settings/api-keys** (account menu → API keys) and set it below.
"""))

cells.append(code(r'''
pos = [(nm, weights[nm]) for nm in feat_names if weights[nm] > 0]
w_lit = "[" * join([string(round(w, digits = 6)) for (_, w) in pos], ", ") * "]"
fn_lit = "[" * join(["\"$nm\"" for (nm, _) in pos], ", ") * "]"
composite_code = """
function on_day(day, features, prices, portfolio)
    w = $w_lit
    fn = $fn_lit
    s = zeros(length(prices))
    for i in eachindex(fn)
        raw = copy(features[fn[i]]); raw[.!isfinite.(raw)] .= 0.0
        z = (raw .- mean(raw)) ./ (std(raw) + 1e-8)   # cross-sectional z-score
        s .+= w[i] .* z
    end
    g = sum(abs.(s)); return g > 0 ? s ./ g : s
end
"""
println(composite_code)

ENV["CONVEXPI_API_KEY"] = "cpk_..."    # <- your key
if get(ENV, "CONVEXPI_API_KEY", "") != "cpk_..." && length(get(ENV, "CONVEXPI_API_KEY", "")) > 8
    submit("ic-ir-composite-julia", composite_code)   # slug defaults to "demo-fall-2026"
else
    println("\n(Set CONVEXPI_API_KEY above to submit, or paste into the web editor at")
    println("https://www.convexpi.ai/compete/demo-fall-2026/submit)")
end
'''))

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
recency = 60; T = size(ret, 1)
recent_ic = Dict(nm => mean([corspearman(features[nm][t, :], ret[t, :]) for t in (T - recency):(T - 1)])
                 for nm in feat_names)
for nm in sort(feat_names, by = n -> -recent_ic[n])
    println("  ", rpad(nm, 9), "recent IC ", rpad(round(recent_ic[nm], digits = 4), 9),
            recent_ic[nm] > 0 ? "keep" : "drop")
end
filtered = Dict(nm => (recent_ic[nm] > 0 ? weights[nm] : 0.0) for nm in feat_names)
ftot = sum(values(filtered))
ftot > 0 && (filtered = Dict(nm => filtered[nm] / ftot for nm in feat_names))
println("\nFiltered + renormalised: ", Dict(nm => round(filtered[nm], digits = 3) for nm in feat_names))
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
5. **Same engine, any language.** Your Julia composite is scored exactly like the Python and R ones.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Julia 1.10", "language": "julia", "name": "julia-1.10"},
                   "language_info": {"name": "julia", "file_extension": ".jl", "mimetype": "application/julia"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "missions",
                                   "mission_03_alpha_discovery", "notebook_julia.ipynb"))
with open(out, "w") as f:
    json.dump(nb, f, indent=1); f.write("\n")
print("wrote", out, "with", len(cells), "cells")
