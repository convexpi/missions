#!/usr/bin/env python3
"""Builds missions/mission_06_advanced_agents/notebook_r.ipynb — bridged R port of the in-process sim."""
import json, os

def md(s):   return {"cell_type": "markdown", "metadata": {}, "source": _lines(s)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(s)}
def _lines(s):
    s = s.strip("\n"); parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]

# The Python sim (Market + agent zoo + run_sim) is infrastructure with no native equivalent, so we
# define it once in the bridged Python and drive it from R.
PY_SETUP = r'''
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
'''

cells = []

cells.append(md(r"""
# Mission 6: Advanced Arena Agents — R edition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/convexpi/missions/blob/main/missions/mission_06_advanced_agents/notebook_r.ipynb)

**Advanced elective.** An in-process market simulator pits a zoo of agents against each other:
noise traders, an informed trader, a momentum chaser, and market makers. You'll see **adverse
selection** cost a naive maker money, watch the **Avellaneda-Stoikov** optimal maker manage inventory,
and tune it to win the tournament.

**Learning objectives**
- See adverse selection empirically: a naive MM profits against noise, bleeds against informed flow
- Understand the Avellaneda-Stoikov inventory-aware maker and its `gamma`/`kappa` knobs
- Run a multi-agent tournament and read the leaderboard

> **How this port works.** The simulator (`convexpi.arena.Market` + the agent zoo) is Python
> infrastructure with no native equivalent, so we drive it through **`reticulate`** and analyse the
> results natively in R. Fully custom agents subclass the Python `Agent` (see the Python notebook or
> `reticulate::PyClass`); here we run the built-in zoo and **tune Avellaneda-Stoikov**, which is the
> core lesson and fully language-agnostic.
"""))

cells.append(md("## Part 0: Setup"))

cells.append(code(r'''
library(reticulate)
if (!py_module_available("convexpi.arena.market")) py_install("convexpi-arena", pip = TRUE)
# Define the simulator + agent zoo helpers once in the bridged Python.
py_run_string(r"(PY_SETUP)")
cat("Ready.\n")
'''.replace("r\"(PY_SETUP)\"", "PY_SETUP_PLACEHOLDER")))

cells.append(md(r"""
---
## Part 1: The anatomy of adverse selection

Run a naive market maker in two worlds: **noise traders only** (it should earn the spread) and
**with an informed trader** (who knows where price is going and picks off stale quotes).
"""))

cells.append(code(r"""
A <- py$scenario_A()      # noise only
B <- py$scenario_B()      # + informed trader
na <- unlist(A$naive_mm); nb <- unlist(B$naive_mm)
plot(na, type = "l", col = "steelblue", lwd = 1.5, ylim = range(c(na, nb)),
     xlab = "tick", ylab = "PnL ($)", main = "Naive MM: PnL over time")
lines(nb, col = "coral", lwd = 1.5); abline(h = 0, lwd = 0.8)
legend("topleft", c("noise only", "+ informed"), col = c("steelblue", "coral"), lty = 1, bty = "n")
cat(sprintf("final PnL — noise only: $%.2f   + informed: $%.2f\n", tail(na, 1), tail(nb, 1)))
"""))

cells.append(md(r"""
The same maker that profits against noise **bleeds against informed flow** — that's adverse
selection. The fix is inventory management and wider spreads when flow looks toxic.

---
## Part 2: Avellaneda-Stoikov optimal market making

The A-S maker quotes a reservation price shifted by inventory and a spread that depends on risk
aversion `gamma` and order-arrival intensity `kappa`. Put it head-to-head with the naive MM in a
market that also has informed and momentum traders.
"""))

cells.append(code(r"""
C <- py$scenario_C()
nm <- unlist(C$naive_mm); as_mm <- unlist(C$as_mm)
plot(nm, type = "l", col = "coral", lwd = 1.5, ylim = range(c(nm, as_mm)),
     xlab = "tick", ylab = "PnL ($)", main = "Naive MM vs Avellaneda-Stoikov")
lines(as_mm, col = "steelblue", lwd = 1.5); abline(h = 0, lwd = 0.8)
legend("topleft", c("NaiveMarketMaker", "AvellanedaStoikov"), col = c("coral", "steelblue"), lty = 1, bty = "n")
cat(sprintf("final PnL — naive: $%.2f   A-S: $%.2f\n", tail(nm, 1), tail(as_mm, 1)))
"""))

cells.append(md(r"""
**Discussion:** the naive maker keeps a tight symmetric spread and accumulates inventory it can't
offload; A-S skews and widens to control inventory risk. Which ends with the smaller inventory swings?

---
## Part 3: Tuning risk aversion

`gamma` is A-S's risk-aversion knob: higher `gamma` → more inventory-averse (wider, more skewed
quotes, fewer fills). Sweep it and see where the maker's final PnL peaks in this market.
"""))

cells.append(code(r"""
gammas <- c(0.01, 0.05, 0.1, 0.3, 0.5, 1.0)
finals <- unlist(py$as_sweep(gammas))
for (i in seq_along(gammas)) cat(sprintf("  gamma=%.2f -> final PnL $%.2f\n", gammas[i], finals[i]))
plot(gammas, finals, type = "b", pch = 19, col = "steelblue", log = "x",
     xlab = "gamma (risk aversion)", ylab = "A-S final PnL ($)",
     main = "Avellaneda-Stoikov: tuning risk aversion")
abline(h = 0, col = "grey", lwd = 0.5)
"""))

cells.append(md(r"""
---
## Part 4: Build your own (challenge)

To author a fully custom agent you subclass the Python `Agent` (same `on_tick(state)` contract as
Mission 2). From R you can do this with `reticulate::PyClass(inherit = reticulate::import("convexpi.arena")$Agent, ...)`,
or write it in the Python edition of this mission. Ideas that beat the zoo:

- Combine A-S inventory control with a directional signal (momentum or mean-reversion).
- Read `state$depth` to detect informed flow and pause quoting.
- Widen the spread during high-volatility stretches; go flat when the spread collapses.

Then add your agent to a `scenario_*`-style run and see if it tops the leaderboard.

---
## Wrap-up

1. **Adverse selection is the maker's central risk** — profits against noise, losses against
   information.
2. **Avellaneda-Stoikov** manages it by pricing inventory risk into the quotes; `gamma`/`kappa`
   trade fills against inventory control.
3. **The tournament is the test** — a strategy that looks good alone can lose against adversaries.

→ Next: Mission 7, Queue Dynamics (L3).
"""))

# splice PY_SETUP into the setup cell
built = json.dumps({"cells": cells}, indent=1)
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "R", "language": "R", "name": "ir"},
                   "language_info": {"name": "R"}},
      "nbformat": 4, "nbformat_minor": 5}
# replace placeholder in the setup cell source with a proper reticulate::py_run_string of PY_SETUP
for c in nb["cells"]:
    src = "".join(c["source"])
    if "PY_SETUP_PLACEHOLDER" in src:
        new = src.replace("py_run_string(PY_SETUP_PLACEHOLDER)",
                          "py_run_string(PY_SETUP)")
        # define PY_SETUP as an R string literal above the call
        r_literal = "PY_SETUP <- r\"---(\n" + PY_SETUP.strip("\n") + "\n)---\"\n"
        new = new.replace("library(reticulate)\n", "library(reticulate)\n" + r_literal, 1)
        c["source"] = [l + "\n" for l in new.split("\n")[:-1]] + [new.split("\n")[-1]]

out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "missions",
                                   "mission_06_advanced_agents", "notebook_r.ipynb"))
with open(out, "w") as f:
    json.dump(nb, f, indent=1); f.write("\n")
print("wrote", out, "with", len(cells), "cells")
