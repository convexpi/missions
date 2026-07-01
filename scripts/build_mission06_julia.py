#!/usr/bin/env python3
"""Builds missions/mission_06_advanced_agents/notebook_julia.ipynb — bridged Julia port."""
import json, os

def md(s):   return {"cell_type": "markdown", "metadata": {}, "source": _lines(s)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(s)}
def _lines(s):
    s = s.strip("\n"); parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]

cells = []

cells.append(md(r"""
# Mission 6: Advanced Arena Agents — Julia edition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/convexpi/missions/blob/main/missions/mission_06_advanced_agents/notebook_julia.ipynb)

**Advanced elective.** An in-process market simulator pits a zoo of agents against each other: noise
traders, an informed trader, a momentum chaser, and market makers. You'll see **adverse selection**
cost a naive maker money, watch the **Avellaneda-Stoikov** optimal maker manage inventory, and tune
it to win the tournament.

**Learning objectives**
- See adverse selection empirically: a naive MM profits against noise, bleeds against informed flow
- Understand the Avellaneda-Stoikov inventory-aware maker and its `gamma`/`kappa` knobs
- Run a multi-agent tournament and read the leaderboard

> **How this port works.** The simulator (`convexpi.arena.Market` + the agent zoo) is Python
> infrastructure with no native equivalent, so we drive it through **`PyCall`** and analyse the
> results natively in Julia. Fully custom agents subclass the Python `Agent` (see the Python edition
> or PyCall's `@pydef`); here we run the built-in zoo and **tune Avellaneda-Stoikov**, the core
> lesson and fully language-agnostic.
"""))

cells.append(md("## Part 0: Setup"))

cells.append(code(r'''
using Pkg; Pkg.add(["PyCall", "UnicodePlots"])
using PyCall, UnicodePlots
try
    pyimport("convexpi.arena.market")
catch
    run(`$(PyCall.python) -m pip install --quiet convexpi-arena`)
end
# Define the simulator + agent zoo helpers once in the bridged Python.
py"""
import pandas as pd
from convexpi.arena.market import Market
from convexpi.arena import (NoiseTrader, NaiveMarketMaker, MomentumTrader, InformedTrader,
                            AvellanedaStoikov)

def _pnl_dict(agents, n_ticks=1000, seed=42):
    telem = {a.agent_id: [] for a in agents}
    def _collect(m):
        mark = m.engine.last_price or (m.fundamental.value if hasattr(m.fundamental, "value") else None)
        if mark is None: return
        for aid in telem:
            acct = m.accounts.get(aid)
            if acct: telem[aid].append(acct.value(mark) / 100)
    market = Market(agents, n_ticks=n_ticks, seed=seed)
    for t in range(1, n_ticks + 1): market.at_tick(t, _collect)
    market.run()
    return {aid: [float(x) for x in rows] for aid, rows in telem.items()}

def noise(n=3): return [NoiseTrader(f"nt{i+1}", seed=10+i) for i in range(n)]
def scenario_A(): return _pnl_dict(noise(4) + [NaiveMarketMaker("naive_mm", seed=40)])
def scenario_B(): return _pnl_dict(noise(3) + [InformedTrader("inf1", seed=20), NaiveMarketMaker("naive_mm", seed=40)])
def scenario_C():
    return _pnl_dict(noise(3) + [InformedTrader("inf1", seed=20), MomentumTrader("mom1", seed=30),
                                 NaiveMarketMaker("naive_mm", seed=40),
                                 AvellanedaStoikov("as_mm", seed=50, gamma=0.1, kappa=1.5, size=15, horizon=1000)])
def as_sweep(gammas):
    out = []
    for g in gammas:
        d = _pnl_dict(noise(3) + [InformedTrader("inf1", seed=20), NaiveMarketMaker("naive_mm", seed=40),
                                  AvellanedaStoikov("as_mm", seed=50, gamma=float(g), kappa=1.5, size=15, horizon=1000)])
        out.append(d["as_mm"][-1])
    return out
"""
println("Ready.")
'''))

cells.append(md(r"""
---
## Part 1: The anatomy of adverse selection

Run a naive market maker in two worlds: **noise traders only** (it earns the spread) and **with an
informed trader** (who picks off stale quotes).
"""))

cells.append(code(r"""
A = py"scenario_A()"; B = py"scenario_B()"
na = convert(Vector{Float64}, A["naive_mm"]); nb = convert(Vector{Float64}, B["naive_mm"])
plt = lineplot(na, name = "noise only", title = "Naive MM: PnL over time",
               xlabel = "tick", ylabel = "PnL (\$)", width = 72, height = 14)
lineplot!(plt, nb, name = "+ informed"); display(plt)
println("final PnL — noise only: \$", round(na[end], digits = 2), "   + informed: \$", round(nb[end], digits = 2))
"""))

cells.append(md(r"""
The same maker that profits against noise **bleeds against informed flow** — that's adverse
selection. The fix is inventory management and wider spreads when flow looks toxic.

---
## Part 2: Avellaneda-Stoikov optimal market making

The A-S maker quotes a reservation price shifted by inventory and a spread governed by risk aversion
`gamma` and order-arrival intensity `kappa`. Head-to-head with the naive MM, in a market that also
has informed and momentum traders.
"""))

cells.append(code(r"""
C = py"scenario_C()"
nm = convert(Vector{Float64}, C["naive_mm"]); as_mm = convert(Vector{Float64}, C["as_mm"])
plt = lineplot(nm, name = "NaiveMarketMaker", title = "Naive MM vs Avellaneda-Stoikov",
               xlabel = "tick", ylabel = "PnL (\$)", width = 72, height = 14)
lineplot!(plt, as_mm, name = "AvellanedaStoikov"); display(plt)
println("final PnL — naive: \$", round(nm[end], digits = 2), "   A-S: \$", round(as_mm[end], digits = 2))
"""))

cells.append(md(r"""
**Discussion:** the naive maker keeps a tight symmetric spread and accumulates inventory it can't
offload; A-S skews and widens to control inventory risk.

---
## Part 3: Tuning risk aversion

`gamma` is A-S's risk-aversion knob: higher `gamma` → more inventory-averse (wider, more skewed
quotes, fewer fills). Sweep it and find where final PnL peaks in this market.
"""))

cells.append(code(r"""
gammas = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
finals = convert(Vector{Float64}, py"as_sweep"(gammas))
for (g, f) in zip(gammas, finals)
    println("  gamma=", rpad(g, 5), " -> final PnL \$", round(f, digits = 2))
end
lineplot(gammas, finals, title = "Avellaneda-Stoikov: tuning risk aversion",
         xlabel = "gamma", ylabel = "A-S final PnL (\$)", width = 72, height = 12)
"""))

cells.append(md(r"""
---
## Part 4: Build your own (challenge)

To author a fully custom agent you subclass the Python `Agent` (same `on_tick(state)` contract as
Mission 2). From Julia you can do this with PyCall's `@pydef`, or write it in the Python edition of
this mission. Ideas that beat the zoo:

- Combine A-S inventory control with a directional signal (momentum or mean-reversion).
- Read `state.depth` to detect informed flow and pause quoting.
- Widen the spread during high-volatility stretches; go flat when the spread collapses.

Add your agent to a `scenario_*`-style run and see if it tops the leaderboard.

---
## Wrap-up

1. **Adverse selection is the maker's central risk** — profits against noise, losses against
   information.
2. **Avellaneda-Stoikov** prices inventory risk into the quotes; `gamma`/`kappa` trade fills against
   inventory control.
3. **The tournament is the test** — a strategy that looks good alone can lose against adversaries.

→ Next: Mission 7, Queue Dynamics (L3).
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Julia 1.10", "language": "julia", "name": "julia-1.10"},
                   "language_info": {"name": "julia", "file_extension": ".jl", "mimetype": "application/julia"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "missions",
                                   "mission_06_advanced_agents", "notebook_julia.ipynb"))
with open(out, "w") as f:
    json.dump(nb, f, indent=1); f.write("\n")
print("wrote", out, "with", len(cells), "cells")
