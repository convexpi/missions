#!/usr/bin/env python3
"""Builds missions/mission_08_cost_of_trading/notebook_julia.ipynb — the faithful Julia port."""
import json, os

def md(s):   return {"cell_type": "markdown", "metadata": {}, "source": _lines(s)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(s)}
def _lines(s):
    s = s.strip("\n"); parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]

cells = []

cells.append(md(r"""
# Mission 8: The Cost of Trading — Julia edition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/convexpi/missions/blob/main/missions/mission_08_cost_of_trading/notebook_julia.ipynb)

**Advanced elective.** You found an alpha — now try to *keep* it. This mission is about the gap
between a frictionless backtest and a tradeable strategy. The villain is **turnover**: every
rebalance pays the spread, and a signal that looks great traded daily can be a guaranteed loser once
costs are real. Everything here runs locally in Julia; there's no submission.

**Learning objectives**
- Quantify how **transaction costs** scale with turnover and erase paper alpha
- Use **rebalance frequency** and **no-trade bands** to trade turnover against signal freshness
- Find a strategy's **break-even cost** — the TC at which its edge disappears
- Reason about **capacity**: why size itself moves the price against you
"""))

cells.append(md("## Part 0: Setup"))

cells.append(code(r"""
using Pkg
Pkg.add(url = "https://github.com/convexpi/ConvexPi.jl")
Pkg.add(["UnicodePlots", "PyCall"])
using ConvexPi
using Statistics, UnicodePlots, PyCall

try
    pyimport("convexpi.lab")
catch
    run(`$(PyCall.python) -m pip install --quiet convexpi-lab`)
end
m = synthetic_market("train")
prices = m.prices
features = m.features
ret = prices[2:end, :] ./ prices[1:end-1, :] .- 1     # (days-1) x stocks, next-day returns

# A backtester mirroring the Python engine's accounting: a strategy(t, portfolio) -> target weights,
# applied on a rebalance cadence, charged tc_bps on the weight it changes.
function run_bt(strategy; tc_bps = 0, rebalance_every = 1, warmup = 60)
    T = size(ret, 1); daily = Float64[]; turn = Float64[]; portfolio = zeros(size(prices, 2))
    for t in (warmup + 1):T
        w = ((t - (warmup + 1)) % rebalance_every == 0) ? strategy(t, portfolio) : portfolio
        traded = sum(abs.(w .- portfolio))            # total weight changed today
        push!(daily, sum(w .* ret[t, :]) - tc_bps / 1e4 * traded); push!(turn, traded)
        portfolio = w
    end
    (sharpe = mean(daily) / (std(daily) + 1e-9) * sqrt(252),
     ann_ret = mean(daily) * 252, turnover_annual = mean(turn) * 252)
end
println("Ready. ", size(prices, 1), " days x ", size(prices, 2), " stocks")
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
function momentum(feature = "mom_1m")
    function (t, portfolio)
        s = copy(features[feature][t, :]); s[.!isfinite.(s)] .= 0.0
        z = (s .- mean(s)) ./ (std(s) + 1e-9)
        k = max(1, length(z) ÷ 5)
        ord = sortperm(z); w = zeros(length(z))
        w[ord[end-k+1:end]] .= 1.0; w[ord[1:k]] .= -1.0
        w ./ (sum(abs.(w)) + 1e-9)
    end
end

for (lab, tc) in [("frictionless (0 bps)", 0.0), ("realistic (20 bps)", 20.0)]
    r = run_bt(momentum(); tc_bps = tc, rebalance_every = 1)
    println(rpad(lab, 22), " Sharpe=", rpad(round(r.sharpe, digits = 3), 8),
            " ann_ret=", rpad(round(100 * r.ann_ret, digits = 2), 8), "%  turnover=",
            round(r.turnover_annual, digits = 1), "x")
end
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
rbs = [1, 2, 5, 10, 21, 42]
res = [run_bt(momentum(); tc_bps = 20, rebalance_every = rb) for rb in rbs]
println(rpad("rebal(d)", 9), rpad("Sharpe", 9), rpad("turnover", 10), "ann_ret")
for (rb, r) in zip(rbs, res)
    println(rpad(rb, 9), rpad(round(r.sharpe, digits = 3), 9),
            rpad(string(round(r.turnover_annual, digits = 1)) * "x", 10),
            round(100 * r.ann_ret, digits = 2), "%")
end
lineplot(rbs, [r.sharpe for r in res], title = "Trading less often: net Sharpe (at 20 bps)",
         xlabel = "rebalance every N days", ylabel = "net Sharpe", width = 70, height = 10)
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
costs = [0, 5, 10, 20, 30, 50, 75, 100]
sharpes = [run_bt(momentum(); tc_bps = c, rebalance_every = 5).sharpe for c in costs]
for (c, s) in zip(costs, sharpes)
    println("  tc=", rpad(c, 4), "bps -> net Sharpe ", round(s, digits = 3))
end
display(lineplot(costs, sharpes, title = "Break-even cost: where the edge disappears",
                 xlabel = "one-way transaction cost (bps)", ylabel = "net Sharpe",
                 width = 70, height = 10))

cross = findfirst(i -> sign(sharpes[i]) != sign(sharpes[i+1]), 1:length(costs)-1)
if cross !== nothing
    be = costs[cross] + (costs[cross+1] - costs[cross]) * sharpes[cross] /
         (sharpes[cross] - sharpes[cross+1])
    println("\nApprox break-even cost (rebal=5d): ~", round(be), " bps")
end
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
function banded_momentum(feature = "mom_1m", band = 0.0)
    base = momentum(feature)
    function (t, portfolio)
        target = base(t, portfolio)
        band <= 0 && return target
        [abs(target[i] - portfolio[i]) > band ? target[i] : portfolio[i] for i in eachindex(target)]
    end
end

println(rpad("band", 6), rpad("Sharpe", 9), rpad("turnover", 10), "ann_ret")
for band in [0.0, 0.002, 0.005, 0.01]
    r = run_bt(banded_momentum("mom_1m", band); tc_bps = 20, rebalance_every = 1)
    println(rpad(band, 6), rpad(round(r.sharpe, digits = 3), 9),
            rpad(string(round(r.turnover_annual, digits = 1)) * "x", 10),
            round(100 * r.ann_ret, digits = 2), "%")
end
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
function cap_sharpe(aum; base_bps = 5, k_bps = 8, adv = 5e6)
    q = (aum / 40) / adv                              # fraction of ADV per name (~20 long + 20 short)
    run_bt(momentum(); tc_bps = base_bps + k_bps * sqrt(max(q, 0)), rebalance_every = 5).sharpe
end
aums = [1e6, 1e7, 5e7, 1e8, 5e8, 1e9]
cap = [cap_sharpe(aum) for aum in aums]
for (aum, s) in zip(aums, cap)
    println("  AUM \$", rpad(round(Int, aum / 1e6), 7), "M -> net Sharpe ", round(s, digits = 3))
end
lineplot(log10.(aums), cap, title = "Capacity: impact rises with size until the edge is gone",
         xlabel = "log10 AUM (\$)", ylabel = "net Sharpe", width = 70, height = 10)
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
      "metadata": {"kernelspec": {"display_name": "Julia 1.10", "language": "julia", "name": "julia-1.10"},
                   "language_info": {"name": "julia", "file_extension": ".jl", "mimetype": "application/julia"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "missions",
                                   "mission_08_cost_of_trading", "notebook_julia.ipynb"))
with open(out, "w") as f:
    json.dump(nb, f, indent=1); f.write("\n")
print("wrote", out, "with", len(cells), "cells")
