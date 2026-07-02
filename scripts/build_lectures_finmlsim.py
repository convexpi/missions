#!/usr/bin/env python3
"""Builds two finmlsim-powered lectures: lectures/stylized_facts.ipynb and lectures/optimal_execution.ipynb."""
import json, os

def md(s):   return {"cell_type": "markdown", "metadata": {}, "source": _lines(s)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(s)}
def _lines(s):
    s = s.strip("\n"); parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]

def write(name, cells):
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lectures", name))
    with open(out, "w") as f:
        json.dump(nb, f, indent=1); f.write("\n")
    print("wrote", out, "with", len(cells), "cells")

INSTALL = r"""
import sys
if 'google.colab' in sys.modules:
    !pip install -q "finmlsim[analysis]"
import numpy as np
import matplotlib.pyplot as plt
import finmlsim as fms
print('ready — finmlsim', fms.__version__)
"""

# ---------------------------------------------------------------------------
# Lecture 1: Stylized facts
# ---------------------------------------------------------------------------
sf = []
sf.append(md(r"""
# Do Our Markets Look Real? Stylized Facts

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/convexpi/missions/blob/main/lectures/stylized_facts.ipynb)

Real asset returns aren't Gaussian white noise. They share a handful of robust **stylized facts**
(Cont, 2001): heavy tails, near-zero autocorrelation of returns but **persistent** autocorrelation of
*absolute/squared* returns (volatility clustering), and a gain/loss asymmetry. A simulator — or a
synthetic market you grade on — is only as honest as the stylized facts it reproduces. We use
[`finmlsim`](https://github.com/convexpi/finmlsim) to generate returns with and without these facts
and measure the difference.
"""))
sf.append(md("## Setup"))
sf.append(code(INSTALL))
sf.append(md(r"""
## 1. Fat tails

A Gaussian says a 5-sigma day is a once-in-a-millennium event; markets have them every few years. A
GARCH process with Student-t innovations produces the heavy tails we actually see.
"""))
sf.append(code(r"""
gauss = fms.simulate.gaussian(n=5000, seed=0)          # thin-tailed control
garch = fms.simulate.garch(n=5000, dist="t", seed=0)   # clustering + fat tails

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for r, name, c in [(gauss, "Gaussian", "steelblue"), (garch, "GARCH-t", "crimson")]:
    z = (r - r.mean()) / r.std()
    ax[0].hist(z, bins=120, density=True, histtype="step", lw=1.5, label=name, color=c)
ax[0].set_yscale("log"); ax[0].set_title("Return distribution (log scale)"); ax[0].legend()
ax[0].set_xlabel("standardised return")
from scipy import stats
stats.probplot(garch, dist="norm", plot=ax[1]); ax[1].set_title("GARCH-t vs Normal Q-Q")
plt.tight_layout(); plt.show()
print(f"excess kurtosis — Gaussian {fms.stylized.summary(gauss)['excess_kurtosis']:.2f}, "
      f"GARCH-t {fms.stylized.summary(garch)['excess_kurtosis']:.2f}  (0 = normal)")
"""))
sf.append(md(r"""
## 2. Volatility clustering

"Large changes tend to be followed by large changes." Returns themselves are almost uncorrelated
(you can't predict tomorrow's *direction*), but their **magnitude** is highly autocorrelated — so
you *can* predict tomorrow's *volatility*.
"""))
sf.append(code(r"""
def acf(x, lags=40):
    x = x - x.mean(); denom = np.sum(x * x)
    return np.array([np.sum(x[k:] * x[:-k]) / denom for k in range(1, lags + 1)])

fig, ax = plt.subplots(figsize=(9, 3.2))
ax.bar(np.arange(1, 41) - 0.15, acf(garch), width=0.3, label="returns", color="steelblue")
ax.bar(np.arange(1, 41) + 0.15, acf(np.abs(garch)), width=0.3, label="|returns|", color="crimson")
ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("lag (days)"); ax.set_ylabel("autocorrelation")
ax.set_title("GARCH-t: returns ≈ uncorrelated, |returns| persistent (clustering)"); ax.legend()
plt.tight_layout(); plt.show()
"""))
sf.append(md(r"""
## 3. Scoring the facts

`finmlsim.stylized.summary` reduces a series to the numbers that matter. A realistic market has
**excess kurtosis > 0**, **return autocorrelation ≈ 0**, and **squared-return autocorrelation > 0**.
"""))
sf.append(code(r"""
import pandas as pd
tbl = pd.DataFrame({"Gaussian": fms.stylized.summary(gauss), "GARCH-t": fms.stylized.summary(garch)})
print(tbl.to_string(float_format=lambda v: f"{v:.4f}"))
"""))
sf.append(md(r"""
## Why this matters for ConvexPi

The Lab's synthetic market already has fat tails and regime vol; its **realistic mode**
(`SyntheticMarket(idio_process="garch")`, powered by this same `finmlsim` GARCH) adds genuine
volatility *clustering* on top — so a strategy meets the same risk texture as real markets before it
ever touches live data.

## Takeaways

1. **Gaussian is a lie for returns** — real tails are heavy; size your risk for 5-sigma days.
2. **Direction is ~unpredictable, volatility is not** — clustering is the most exploitable stylized fact.
3. **Measure, don't assume** — `stylized.summary` tells you whether a simulated (or real) series is honest.

→ These facts underpin risk in every mission; the clustering point returns in Mission 8 (cost of trading).
"""))
write("stylized_facts.ipynb", sf)

# ---------------------------------------------------------------------------
# Lecture 2: Optimal execution (Almgren-Chriss)
# ---------------------------------------------------------------------------
ex = []
ex.append(md(r"""
# Optimal Execution: the Almgren-Chriss Frontier

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/convexpi/missions/blob/main/lectures/optimal_execution.ipynb)

You have a big position to unwind. Sell it all now and you pay huge **market impact**; sell it slowly
and you're exposed to price **risk** the whole time. Almgren-Chriss (2000) makes this precise: choose
the liquidation trajectory that minimises `E[cost] + λ · Var[cost]`, where `λ` is your risk aversion.
We trace the trade-off with [`finmlsim`](https://github.com/convexpi/finmlsim).
"""))
ex.append(md("## Setup"))
ex.append(code(INSTALL))
ex.append(md(r"""
## 1. Patient vs. aggressive trajectories

Higher `λ` (more risk-averse) front-loads selling to cut exposure; `λ → 0` (risk-neutral) trades
evenly to minimise impact. Each curve is the position remaining over the trading day.
"""))
ex.append(code(r"""
fig, ax = plt.subplots(figsize=(9, 3.6))
for lam, c in [(1e-7, "steelblue"), (1e-6, "seagreen"), (1e-5, "crimson")]:
    res = fms.simulate.almgren_chriss(X=1.0, T=1.0, N=50, sigma=0.02, eta=2.5e-6, lam=lam)
    ax.plot(np.linspace(0, 1, len(res["x"])), res["x"], lw=1.8, label=f"λ={lam:g}")
ax.set_xlabel("time (fraction of horizon)"); ax.set_ylabel("shares remaining")
ax.set_title("Almgren-Chriss liquidation trajectories"); ax.legend(); ax.axhline(0, color="k", lw=0.6)
plt.tight_layout(); plt.show()
"""))
ex.append(md(r"""
## 2. The efficient frontier

Sweep `λ` and plot each strategy's **expected cost** against its **cost risk** (standard deviation).
The frontier is the set of un-improvable trade-offs — you can't cut risk without paying more impact.
"""))
ex.append(code(r"""
lams = np.logspace(-8, -4, 25)
pts = [fms.simulate.almgren_chriss(X=1.0, T=1.0, N=50, sigma=0.02, eta=2.5e-6, lam=l) for l in lams]
E = np.array([p["E_cost"] for p in pts]); S = np.array([p["sd_cost"] for p in pts])

fig, ax = plt.subplots(figsize=(7.5, 4))
sc = ax.scatter(S, E * 1e4, c=np.log10(lams), cmap="viridis", s=30)
ax.set_xlabel("cost risk  (sd of cost)"); ax.set_ylabel("expected cost (bps)")
ax.set_title("Execution efficient frontier (colour = log₁₀ λ)")
plt.colorbar(sc, label="log₁₀ risk aversion λ"); plt.tight_layout(); plt.show()
print("risk-neutral (λ→0): lowest expected cost, highest risk;  risk-averse: the opposite.")
"""))
ex.append(md(r"""
## Why this matters for ConvexPi

Mission 8 shows turnover-times-spread eating your alpha at the *portfolio* level. Almgren-Chriss is
the same trade-off one level down — inside a single order. A strategy's *paper* alpha is only real if
it survives both: the per-order impact of getting in, and the per-rebalance cost of staying in.

## Takeaways

1. **Impact vs. timing risk is the core execution trade-off** — trading faster cuts risk but costs impact.
2. **λ (risk aversion) picks your point on the frontier** — there is no single "optimal" speed, only a trade-off.
3. **Impact is convex in size** — which is why *capacity* (Mission 8) is finite.

→ Pair this with Mission 8 (the cost of trading) for the portfolio-level view.
"""))
write("optimal_execution.ipynb", ex)
