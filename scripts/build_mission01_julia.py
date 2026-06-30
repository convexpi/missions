#!/usr/bin/env python3
"""Builds missions/mission_01_overfitting/notebook_julia.ipynb — the faithful Julia port."""
import json, os

def md(s):   return {"cell_type": "markdown", "metadata": {}, "source": _lines(s)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(s)}
def _lines(s):
    s = s.strip("\n")
    parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]

cells = []

cells.append(md(r"""
# Mission 1: The Overfitting Game — Julia edition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/convexpi/missions/blob/main/missions/mission_01_overfitting/notebook_julia.ipynb)

**Learning objectives**
- Build intuition for in-sample vs. out-of-sample performance
- Experience the overfitting trap firsthand
- Submit a **Julia** strategy and interpret your OOS grade report
- Understand the Information Coefficient (IC) and its role in strategy evaluation

> This is the Julia port of Mission 1. You write your strategy in Julia, and the grader runs your
> Julia `on_day()` over a hidden holdout and scores it with the **same engine** as Python and R —
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
# Install the ConvexPi Julia package (market data + one-call submit) and a few helpers.
using Pkg
Pkg.add(url = "https://github.com/convexpi/ConvexPi.jl")
Pkg.add(["StatsBase", "UnicodePlots", "PyCall"])   # rank correlation + plots + Python bridge
using ConvexPi
using Statistics, LinearAlgebra, StatsBase, UnicodePlots
using PyCall
import Random; Random.seed!(42)

# The market comes from the Python engine (convexpi-lab) via PyCall — make sure it's there.
try
    pyimport("convexpi.lab")
catch
    run(`$(PyCall.python) -m pip install --quiet convexpi-lab`)
end
println("Ready.")
"""))

cells.append(md(r"""
---
## Part 1: Explore the Synthetic Market

The ConvexPi Lab uses a **synthetic market** — a simulated panel of stock prices and features.
`synthetic_market("train")` returns the exact in-sample panel the grader fits on (deterministic from
the seed). The hidden holdout (`"test"`) is what you are scored on — you never see its returns; the
grader does.
"""))

cells.append(code(r"""
m = synthetic_market("train")          # the exact market the grader uses
prices   = m.prices                    # days x stocks matrix
features = m.features                  # Dict of days x stocks matrices
feat_names = collect(keys(features))

println("prices : ", size(prices, 1), " days x ", size(prices, 2), " stocks")
println("features: ", join(feat_names, ", "))
"""))

cells.append(code(r"""
# Daily cross-sectional returns from prices: ret[t,:] = prices[t+1,:] ./ prices[t,:] .- 1
ret = prices[2:end, :] ./ prices[1:end-1, :] .- 1     # (days-1) x stocks
println("returns: ", size(ret, 1), " days x ", size(ret, 2), " stocks")
println("mean daily return: ", round(mean(ret), digits = 5))
"""))

cells.append(code(r"""
# Visualise the market's cumulative return (equal-weighted, in-sample)
mkt = vec(mean(ret, dims = 2))
cum = cumprod(1 .+ mkt) .- 1
lineplot(cum .* 100, title = "Market cumulative return (EW, IS)",
         xlabel = "day", ylabel = "%", width = 70, height = 12)
"""))

cells.append(md(r"""
---
## Part 2: A Baseline — what does noise look like?

A strategy here returns a vector of **target weights** (one per stock) each day. We will build a
small backtester so you can *feel* the overfitting trap before submitting. Start with a random
strategy — pure noise, no alpha.
"""))

cells.append(code(r"""
# Long/short backtester: long the top_k by score, short the bottom_k, equal weight.
# signal_fn(t) returns a score vector (one per stock) for day t.
function simple_backtest(signal_fn; top_k = 20, cost_bps = 5)
    n = size(prices, 2); prev_w = zeros(n); daily = zeros(size(ret, 1))
    for t in 1:size(ret, 1)
        s = signal_fn(t); ord = sortperm(s); w = zeros(n)
        w[ord[end-top_k+1:end]] .=  1 / top_k    # long the highest scores
        w[ord[1:top_k]]        .= -1 / top_k     # short the lowest scores
        cost = cost_bps / 1e4 * sum(abs.(w .- prev_w))
        daily[t] = sum(w .* ret[t, :]) - cost
        prev_w = w
    end
    (sharpe = mean(daily) / (std(daily) + 1e-9) * sqrt(252),
     ann_ret = mean(daily) * 252, daily = daily)
end

randsig(seed) = (Random.seed!(seed); t -> randn(size(prices, 2)))
r_random = simple_backtest(randsig(0))
println("Random strategy  IS Sharpe = ", round(r_random.sharpe, digits = 3),
        "   IS Ann Ret = ", round(100 * r_random.ann_ret, digits = 2), "%")
"""))

cells.append(code(r"""
eq = cumprod(1 .+ r_random.daily) .- 1
lineplot(eq, title = "Random strategy equity (noise baseline)",
         xlabel = "day", ylabel = "P&L", width = 70, height = 12)
"""))

cells.append(md(r"""
---
## Part 3: Find a Real Signal

Exactly one feature is a planted **alpha** — a small but real predictive relationship with next-day
returns. The rest are noise. A good way to measure predictiveness is the **Information Coefficient
(IC)**: the rank correlation between today's signal and tomorrow's cross-sectional returns. Mean
IC > 0 means the signal is directionally useful; **IC-IR** = mean(IC) / std(IC) × √252 measures how
reliable it is.
"""))

cells.append(code(r"""
# Spearman (rank) IC of a feature vs next-day returns, day by day.
ic_series(fmat) = [corspearman(fmat[t, :], ret[t, :]) for t in 1:size(ret, 1)]

summary_ic = Dict{String,NamedTuple}()
for name in feat_names
    ics = ic_series(features[name])
    mic = mean(ics)
    icir = mic / (std(ics) + 1e-9) * sqrt(252)
    summary_ic[name] = (mean_ic = mic, ic_ir = icir)
    println(rpad(name, 10), "  mean IC=", rpad(round(mic, digits = 4), 9),
            "  IC-IR=", round(icir, digits = 2))
end

# Plot the planted-alpha candidate's rolling IC
best_feature = argmax(name -> summary_ic[name].mean_ic, feat_names)
ics = ic_series(features[best_feature])
roll = [mean(ics[max(1, t-19):t]) for t in 1:length(ics)]   # 20-day moving average
lineplot(roll, title = "Rolling IC: $best_feature", xlabel = "day",
         ylabel = "IC", width = 70, height = 10)
"""))

cells.append(md("**Exercise 3.1** — Which feature has the highest mean IC? Is it reliable (|IC-IR| > 1.5)?"))

cells.append(code(r"""
println("Highest mean IC: ", best_feature,
        "  (IC-IR = ", round(summary_ic[best_feature].ic_ir, digits = 2), ")")
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
nf = length(feat_names)
function weighted_signal(w)
    w = w / (norm(w) + 1e-8)
    function (t)
        s = zeros(size(prices, 2))
        for (i, name) in enumerate(feat_names)
            s .+= w[i] .* features[name][t, :]
        end
        s
    end
end

grid_sharpe = Float64[]
best = (sharpe = -Inf, top_k = 0, w = zeros(nf))
trial = 0
for top_k in (10, 20, 40), _ in 1:200
    Random.seed!(trial); global trial += 1
    w = randn(nf)
    r = simple_backtest(weighted_signal(w); top_k = top_k)
    push!(grid_sharpe, r.sharpe)
    if r.sharpe > best.sharpe
        global best = (sharpe = r.sharpe, top_k = top_k, w = w)
    end
end
println("Tested ", length(grid_sharpe), " strategies. Best IS Sharpe = ",
        round(best.sharpe, digits = 3), " (top_k=", best.top_k, ")")
"""))

cells.append(code(r"""
histogram(grid_sharpe, nbins = 40, title = "Distribution of IS Sharpe across the grid",
          xlabel = "IS Sharpe", width = 60)
println("Best selected (red line in spirit): ", round(best.sharpe, digits = 2))
"""))

cells.append(md(r"""
**The trap:** that best IS Sharpe was *selected* from 600 trials — by chance alone some weights look
great in-sample. The grader, scoring on data you have never seen, will not be so impressed. Let's
find out.
"""))

cells.append(md(r"""
---
## Part 5: Submit to the Grader

Your strategy is a Julia function `on_day(day, features, prices, portfolio)` that returns a vector of
target weights. The grader runs it over the hidden holdout and reports your **OOS Sharpe**, plus
which planted alphas you recovered.

Create an API key at **convexpi.ai/settings/api-keys** (account menu → API keys) and set it below.

### 5A: Submit the grid-search "winner" (the overfit strategy)
"""))

cells.append(code(r'''
# Embed the grid-winning weights into an on_day() that linearly combines the features.
wvec = best.w / (norm(best.w) + 1e-8)
w_lit = "[" * join(round.(wvec, digits = 6), ", ") * "]"
fn_lit = "[" * join(["\"$n\"" for n in feat_names], ", ") * "]"
overfit_code = """
function on_day(day, features, prices, portfolio)
    w = $w_lit
    fn = $fn_lit
    s = zeros(length(prices))
    for i in eachindex(fn)
        v = copy(features[fn[i]]); v[.!isfinite.(v)] .= 0.0
        s .+= w[i] .* v
    end
    g = sum(abs.(s)); return g > 0 ? s ./ g : s
end
"""
println(overfit_code)
'''))

cells.append(code(r"""
# Set your key once, then submit. Get one at https://www.convexpi.ai/settings/api-keys
ENV["CONVEXPI_API_KEY"] = "cpk_..."    # <- your key
if get(ENV, "CONVEXPI_API_KEY", "") != "cpk_..." && length(get(ENV, "CONVEXPI_API_KEY", "")) > 8
    submit("overfit-gridsearch-julia", overfit_code)   # slug defaults to "demo-fall-2026"
else
    println("No CONVEXPI_API_KEY set. Either set it above and re-run, or paste the code")
    println("printed in the previous cell into the web editor at")
    println("https://www.convexpi.ai/compete/demo-fall-2026/submit")
end
"""))

cells.append(md(r"""
**Exercise 5.1** — Is the OOS Sharpe higher or lower than the IS Sharpe you selected (the best from
the grid)? Why?

### 5B: Submit the principled strategy (single factor, highest IC)

No tuning beyond the one decision you justified with the IC analysis — a cross-sectional z-score of
the best feature.
"""))

cells.append(code(r'''
principled_code = """
function on_day(day, features, prices, portfolio)
    raw = copy(features["$best_feature"]); raw[.!isfinite.(raw)] .= 0.0
    z = (raw .- mean(raw)) ./ (std(raw) + 1e-8)   # cross-sectional z-score
    g = sum(abs.(z)); return g > 0 ? z ./ g : z
end
"""
println(principled_code)

if get(ENV, "CONVEXPI_API_KEY", "") != "cpk_..." && length(get(ENV, "CONVEXPI_API_KEY", "")) > 8
    submit("principled-single-factor-julia", principled_code)
else
    println("\n(Set CONVEXPI_API_KEY above to submit, or paste into the web editor.)")
end
'''))

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

- **A (Easy):** Replace the z-score in 5B with a **rank** transform
  (`ordinalrank(raw) .- mean(ordinalrank(raw))`). Does OOS Sharpe improve?
- **B (Medium):** Use an **EMA** of the signal (half-lives 5, 10, 20 days, carried via `portfolio`
  or a closure). What happens to turnover and IC as the half-life grows?
- **C (Medium):** Combine two features with weights fit on in-sample data (OLS of lagged features →
  next-day return). Does it beat the best single feature OOS?
- **D (Hard):** Walk-forward: refit weights on a rolling 100-day window each day. Does it beat the
  static weights OOS? What does that say about regime stationarity?
"""))

cells.append(code(r"""
# Challenge A starter — rank transform, tested in the local backtester.
ranked_signal(t) = (raw = features[best_feature][t, :];
                    ordinalrank(raw) .- mean(ordinalrank(raw)))
println("Ranked strategy IS Sharpe = ", round(simple_backtest(ranked_signal).sharpe, digits = 3))
"""))

cells.append(md(r"""
---
## Wrap-up

1. **IS Sharpe is not OOS Sharpe.** Searching over many strategies inflates in-sample performance;
   the more knobs, the wider the gap.
2. **IC is a better guide than Sharpe** for picking signals — it measures prediction directly.
3. **Simple and robust beats complex and tuned.** A single-factor z-score often beats a grid-search
   winner OOS.
4. **Same engine, any language.** Your Julia `on_day` is scored exactly like the Python and R
   versions — pick the language you think in.

---
*Next: Mission 3 (Alpha Discovery) — also available in Julia.*
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Julia 1.10", "language": "julia", "name": "julia-1.10"},
        "language_info": {"name": "julia", "file_extension": ".jl", "mimetype": "application/julia"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = os.path.join(os.path.dirname(__file__), "..", "missions",
                   "mission_01_overfitting", "notebook_julia.ipynb")
out = os.path.abspath(out)
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")
print("wrote", out, "with", len(cells), "cells")
