#!/usr/bin/env python3
"""Builds missions/mission_09_pairs_trading/notebook_julia.ipynb — the faithful Julia port."""
import json, os

def md(s):   return {"cell_type": "markdown", "metadata": {}, "source": _lines(s)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(s)}
def _lines(s):
    s = s.strip("\n"); parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]

cells = []

cells.append(md(r"""
# Mission 9: Pairs Trading & Statistical Arbitrage — Julia edition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/convexpi/missions/blob/main/missions/mission_09_pairs_trading/notebook_julia.ipynb)

**Advanced elective.** Every strategy so far was *cross-sectional* — rank many stocks at one moment.
Pairs trading is the canonical *time-series* alternative: find two assets tied together by a long-run
equilibrium and bet that temporary divergences snap back. It's the textbook example of **statistical
arbitrage** — and of how a relationship that looks ironclad in-sample can quietly fall apart.

**Learning objectives**
- Distinguish **correlation** from **cointegration** — and why only the latter gives a tradeable spread
- Test for cointegration (OLS hedge ratio + an ADF unit-root test) and form a stationary spread
- Trade the spread with a **z-score** entry/exit rule and evaluate it
- See **spurious cointegration**: scanning many pairs manufactures false equilibria that break OOS

Everything runs locally in Julia; there's no submission.
"""))

cells.append(md("## Part 0: Setup"))

cells.append(code(r"""
using Pkg
Pkg.add(url = "https://github.com/convexpi/ConvexPi.jl")
Pkg.add(["Statistics", "LinearAlgebra", "UnicodePlots", "PyCall"])
using ConvexPi
using Statistics, LinearAlgebra, UnicodePlots, PyCall
import Random; Random.seed!(7)

try
    pyimport("convexpi.lab")
catch
    run(`$(PyCall.python) -m pip install --quiet convexpi-lab`)   # needed for the Part 4 scan
end

# OLS hedge ratio: slope of b on a (how many units of A hedge one unit of B).
slope(a, b) = sum((a .- mean(a)) .* (b .- mean(b))) / sum((a .- mean(a)).^2)

# Augmented Dickey-Fuller t-statistic (constant + k lags). Very negative => reject a unit root
# => the series is stationary (mean-reverting). We compare against critical values below.
function adf_stat(y; k = floor(Int, (length(y) - 1)^(1/3)))
    n = length(y); dy = diff(y); ylag = y[1:n-1]; m = n - 1; idx = (k+1):m
    X = hcat(ones(length(idx)), ylag[idx])
    for i in 1:k
        X = hcat(X, dy[idx .- i])
    end
    yv = dy[idx]
    XtX = X'X; b = XtX \ (X'yv)
    resid = yv - X * b; dof = length(yv) - size(X, 2)
    se2 = (sum(resid.^2) / dof) * inv(XtX)[2, 2]
    b[2] / sqrt(se2)
end
# 5% critical values: single series ≈ -2.86; Engle-Granger cointegration (2 series, β estimated) ≈ -3.34.
const CRIT_COINT = -3.34
println("Ready.")
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
T = 800
common = cumsum(randn(T))                       # a shared stochastic trend
A = 50 .+ common .+ randn(T)                     # both prices ride `common`...
B = 20 .+ 0.8 .* common .+ randn(T)             # ...B with sensitivity (beta) 0.8

X = cumsum(randn(T))                             # control: two INDEPENDENT random walks
Y = cumsum(randn(T))

p1 = lineplot(A, name = "A", title = "Cointegrated pair", xlabel = "day", ylabel = "price",
              width = 70, height = 10); lineplot!(p1, B, name = "B"); display(p1)
p2 = lineplot(X, name = "X", title = "Independent random walks", xlabel = "day", ylabel = "price",
              width = 70, height = 10); lineplot!(p2, Y, name = "Y"); display(p2)
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
hedge_and_spread(a, b) = (beta = slope(a, b); (beta = beta, spread = b .- beta .* a))

for (nm, ab) in [("cointegrated (A,B)", (A, B)), ("independent (X,Y)", (X, Y))]
    d = hedge_and_spread(ab[1], ab[2])
    stat = adf_stat(d.spread)
    verdict = stat < CRIT_COINT ? "COINTEGRATED" : "not cointegrated"
    println(rpad(nm, 20), " beta=", rpad(round(d.beta, digits = 2), 6),
            " ADF(spread)=", rpad(round(stat, digits = 3), 8),
            " (5% crit ", CRIT_COINT, ") -> ", verdict)
end
"""))

cells.append(code(r"""
sp_AB = hedge_and_spread(A, B).spread
sp_XY = hedge_and_spread(X, Y).spread
plt = lineplot(sp_AB .- mean(sp_AB), name = "A,B spread (stationary)",
               title = "A tradeable spread reverts to its mean", xlabel = "day", ylabel = "spread",
               width = 72, height = 12)
lineplot!(plt, sp_XY .- mean(sp_XY), name = "X,Y spread (wanders)")
plt
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
function backtest_pair(a, b, beta; entry = 2.0, exit = 0.5, lookback = 60)
    s = b .- beta .* a; n = length(s); z = fill(NaN, n)
    for t in (lookback+1):n
        win = s[(t-lookback):(t-1)]; z[t] = (s[t] - mean(win)) / (std(win) + 1e-9)
    end
    ds = diff(s); pos = 0; pnl = Float64[]; states = Int[]
    for t in (lookback+1):(n-1)
        if pos == 0
            if z[t] > entry; pos = -1 elseif z[t] < -entry; pos = 1 end
        elseif abs(z[t]) < exit
            pos = 0
        end
        push!(pnl, pos * ds[t]); push!(states, pos)
    end
    (z = z, states = states, pnl = pnl, sharpe = mean(pnl) / (std(pnl) + 1e-9) * sqrt(252))
end

beta = slope(A, B)
r = backtest_pair(A, B, beta)
println("hedge ratio beta : ", round(beta, digits = 2))
println("pair Sharpe      : ", round(r.sharpe, digits = 2))
println("total spread P&L : ", round(sum(r.pnl), digits = 1),
        "  (round-trips: ", count(!=(0), diff(r.states)), ")")

display(lineplot(filter(!isnan, r.z), title = "Spread z-score (entry at |z|>2, exit near 0)",
                 xlabel = "day", ylabel = "z", width = 72, height = 8))
display(lineplot(cumsum(r.pnl), title = "Cumulative spread P&L", xlabel = "day", ylabel = "P&L",
                 width = 72, height = 8))
"""))

cells.append(md(r"""
**Exercise 3.1** — Sweep `entry` over `[1.0, 1.5, 2.0, 2.5, 3.0]`. Wider thresholds trade less often
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
Pis  = synthetic_market("train").prices[:, 1:60]     # 60 assets, in-sample window
Poos = synthetic_market("test").prices[:, 1:60]      # holdout window (the honest test)

coint_ok(a, b) = adf_stat(b .- slope(a, b) .* a) < CRIT_COINT
function scan(Pis, Poos)
    N = size(Pis, 2); h = 0; surv = 0; broken = nothing
    for i in 1:N-1, j in (i+1):N
        if coint_ok(Pis[:, i], Pis[:, j])
            h += 1
            if coint_ok(Poos[:, i], Poos[:, j])
                surv += 1
            elseif broken === nothing
                broken = (i, j)
            end
        end
    end
    (h = h, surv = surv, broken = broken, checked = N * (N - 1) ÷ 2)
end

s = scan(Pis, Poos)
println("pairs scanned            : ", s.checked)
println("'cointegrated' in-sample : ", s.h, "  (", round(100 * s.h / s.checked, digits = 1),
        "% — near the 5% you'd expect by chance)")
println("still cointegrated OOS   : ", s.surv, "  (", round(100 * s.surv / max(s.h, 1), digits = 0),
        "% of the in-sample hits)")
println("\nMost 'discovered' pairs are spurious. Finding cointegration is easy; finding it OOS is the job.")
"""))

cells.append(code(r"""
# A spurious pair: mean-reverting in-sample, then drifts on the holdout (using the IS hedge ratio).
if s.broken !== nothing
    i, j = s.broken
    beta_is = slope(Pis[:, i], Pis[:, j])
    sp_is  = Pis[:, j]  .- beta_is .* Pis[:, i]
    sp_oos = Poos[:, j] .- beta_is .* Poos[:, i]
    n1 = length(sp_is)
    plt = lineplot(1:n1, sp_is .- mean(sp_is), name = "in-sample (looks tradeable)",
                   title = "Spurious pair (assets $i,$j): the equilibrium was never real",
                   xlabel = "day", ylabel = "spread", width = 72, height = 12)
    lineplot!(plt, (n1+1):(n1+length(sp_oos)), sp_oos .- mean(sp_is), name = "holdout (drifts away)")
    display(plt)
else
    println("No spurious example found in this scan.")
end
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
      "metadata": {"kernelspec": {"display_name": "Julia 1.10", "language": "julia", "name": "julia-1.10"},
                   "language_info": {"name": "julia", "file_extension": ".jl", "mimetype": "application/julia"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "missions",
                                   "mission_09_pairs_trading", "notebook_julia.ipynb"))
with open(out, "w") as f:
    json.dump(nb, f, indent=1); f.write("\n")
print("wrote", out, "with", len(cells), "cells")
