#!/usr/bin/env python3
"""Builds missions/mission_02_marketmaker/notebook_julia.ipynb — the faithful Julia port (live Arena)."""
import json, os

def md(s):   return {"cell_type": "markdown", "metadata": {}, "source": _lines(s)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(s)}
def _lines(s):
    s = s.strip("\n"); parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]

cells = []

cells.append(md(r"""
# Mission 2: Market-Maker in the Arena — Julia edition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/convexpi/missions/blob/main/missions/mission_02_marketmaker/notebook_julia.ipynb)

**Learning objectives**
- Understand the economics of market-making: earning the spread vs. adverse selection
- Build a **live** agent with the Julia `ConvexPi` Arena client (`run_agent`)
- Measure inventory risk and attribute PnL (spread income vs. inventory mark-to-market)
- Iterate on quoting logic and climb the live leaderboard

## Background

A **market-maker** continuously posts bid and ask limit orders, earning the spread when both sides
fill. The risk: an informed trader knows where prices are going and hits your stale quotes —
**adverse selection**. Good makers quote a spread wide enough to cover it, **manage inventory**, and
react quickly. The Arena is a discrete-time limit-order book: each tick you get the book state and
return orders. Watch your PnL and maker % at
[/compete/arena-book](https://convexpi.ai/compete/arena-book).
"""))

cells.append(md("## Part 0: Setup"))

cells.append(code(r"""
using Pkg
Pkg.add(url = "https://github.com/convexpi/ConvexPi.jl")
Pkg.add(["Statistics", "UnicodePlots"])
using ConvexPi
using Statistics, UnicodePlots

AGENT_ID = "mm_jl_student01"   # <- change to your unique handle (your name on the leaderboard)
println("Ready.")
"""))

cells.append(md(r"""
## Part 1: The agent contract

Your agent is a function `on_tick(state)` returning a vector of orders. Each tick the server hands
you a read-only `state` (a NamedTuple):

| field | meaning |
|---|---|
| `state.best_bid` / `state.best_ask` | top of book, in **integer cents** (`nothing` if empty) |
| `state.mid` / `state.spread` | convenience values |
| `state.last_price` | last trade price (cents) |
| `state.position`, `state.cash` | your signed inventory and cash (cents) |
| `state.my_open_orders` | `[Dict("order_id"=>, "side"=>, "price"=>, "qty"=>), ...]` |

Order helpers: `arena_limit(side, price, qty)`, `arena_market_order(side, qty)`, `arena_cancel(id)`.
`run_agent(on_tick; agent_id, max_ticks)` connects, runs, and returns a telemetry vector.
"""))

cells.append(md(r"""
## Part 2: A naive market maker

Quote a fixed half-spread around the mid, re-quoting each tick. No inventory management.
"""))

cells.append(code(r"""
function naive_mm(state)
    (state.best_bid === nothing || state.best_ask === nothing) && return Dict[]
    mid = (state.best_bid + state.best_ask) ÷ 2
    orders = Dict[]
    for o in state.my_open_orders; push!(orders, arena_cancel(o["order_id"])); end   # pull old quotes
    push!(orders, arena_limit("buy", mid - 5, 10)); push!(orders, arena_limit("sell", mid + 5, 10))
    orders
end

# Runs live for 200 ticks (~a few minutes at 1 tick/s) and returns telemetry.
df = run_agent(naive_mm; agent_id = AGENT_ID, max_ticks = 200)
println("collected ", length(df), " ticks")
"""))

cells.append(md("## Part 3: Analyse telemetry"))

cells.append(code(r"""
function plot_telem(df; title = "Naive MM")
    isempty(df) && (println("No data — check the Arena connection."); return)
    tk = [r.tick for r in df]; pnl = [r.pnl for r in df]
    rel = pnl .- pnl[1]                              # PnL relative to the start
    pos = [r.position for r in df]; lp = [r.last_price for r in df]
    display(lineplot(tk, rel, title = "$title — PnL (\$)", xlabel = "tick", ylabel = "PnL", width = 72, height = 8))
    display(lineplot(tk, pos, title = "Inventory (shares)", xlabel = "tick", ylabel = "pos", width = 72, height = 6))
    display(lineplot(tk, lp, title = "Market price (\$)", xlabel = "tick", ylabel = "px", width = 72, height = 6))
    println("Final PnL: \$", round(rel[end], digits = 2), "   Max |inventory|: ", maximum(abs.(pos)), " shares")
end
plot_telem(df; title = "Naive MM")
"""))

cells.append(md(r"""
### Diagnosing performance

| Symptom | Likely cause | Fix |
|---|---|---|
| PnL drifts negative despite trades | Adverse selection | Widen the spread or skew against inventory |
| Inventory grows one-sided | No inventory management | Add a position limit and skew quotes |
| Very few fills | Spread too wide | Tighten the spread or quote larger |

## Part 4: Inventory-aware market maker

Skew quotes by inventory: when you're long, lower both quotes so you sell more eagerly; cap the
absolute position; size down as inventory grows.
"""))

cells.append(code(r"""
function make_inventory_mm(; half_spread = 6, qty = 10, skew = 0.5, max_pos = 50)
    function (state)
        (state.best_bid === nothing || state.best_ask === nothing) && return Dict[]
        pos = state.position
        orders = Dict[]
        for o in state.my_open_orders; push!(orders, arena_cancel(o["order_id"])); end
        abs(pos) >= max_pos && return orders                    # at the cap: quote nothing new
        mid = (state.best_bid + state.best_ask) / 2
        skewed = mid - pos * skew                               # long -> push quotes down
        bid = round(Int, skewed - half_spread); ask = round(Int, skewed + half_spread)
        (bid >= state.best_ask || ask <= state.best_bid) && return orders   # never cross
        q = max(1, round(Int, qty * max(0.2, 1 - abs(pos) / max_pos)))      # shrink size as inventory grows
        push!(orders, arena_limit("buy", bid, q)); push!(orders, arena_limit("sell", ask, q))
        orders
    end
end

df2 = run_agent(make_inventory_mm(); agent_id = AGENT_ID * "_v2", max_ticks = 200)
plot_telem(df2; title = "Inventory-aware MM")
"""))

cells.append(md(r"""
## Part 5: PnL attribution

Split PnL into inventory mark-to-market (price moves on the shares you hold) vs. the residual, which
is your spread income.
"""))

cells.append(code(r"""
function attribute_pnl(df)
    tk = [r.tick for r in df]; pnl = [r.pnl for r in df]
    pos = [r.position for r in df]; lp = [r.last_price for r in df]
    rel = pnl .- pnl[1]
    pc = [0.0; diff(lp)]                                        # price change per tick
    mtm = cumsum([0.0; pos[1:end-1]] .* pc)                     # prior inventory x today's move
    (tick = tk, total = rel, mtm = mtm, spread = rel .- mtm)
end
if !isempty(df2)
    a = attribute_pnl(df2)
    plt = lineplot(a.tick, a.total, name = "Total", title = "PnL attribution (\$)",
                   xlabel = "tick", ylabel = "PnL", width = 72, height = 12)
    lineplot!(plt, a.tick, a.spread, name = "Spread income")
    lineplot!(plt, a.tick, a.mtm, name = "Inventory MTM")
    display(plt)
end
"""))

cells.append(md(r"""
## Part 6: Challenges

- **A (easy):** volatility-adaptive spread — widen when recent prices are choppy (starter below).
- **B (medium):** back off after a streak of adverse fills (use the `on_fill` argument of `run_agent`).
- **C (hard):** estimate informed-flow pressure from the buy/sell imbalance in `state.recent_trades`
  and widen when it's high.
"""))

cells.append(code(r"""
# Challenge A starter — a volatility-adaptive maker (a closure that remembers recent prices).
function make_vol_adaptive_mm(; k = 0.5, qty = 10, skew = 0.5, max_pos = 50)
    prices = Float64[]
    function (state)
        (state.best_bid === nothing || state.best_ask === nothing) && return Dict[]
        if state.last_price !== nothing
            push!(prices, state.last_price); length(prices) > 20 && popfirst!(prices)
        end
        hs = length(prices) >= 5 ? max(3, round(Int, k * std(prices) / 100)) : 6
        pos = state.position
        orders = Dict[]
        for o in state.my_open_orders; push!(orders, arena_cancel(o["order_id"])); end
        abs(pos) >= max_pos && return orders
        mid = (state.best_bid + state.best_ask) / 2; skewed = mid - pos * skew
        bid = round(Int, skewed - hs); ask = round(Int, skewed + hs)
        (bid >= state.best_ask || ask <= state.best_bid) && return orders
        q = max(1, round(Int, qty * max(0.2, 1 - abs(pos) / max_pos)))
        push!(orders, arena_limit("buy", bid, q)); push!(orders, arena_limit("sell", ask, q))
        orders
    end
end
# df3 = run_agent(make_vol_adaptive_mm(); agent_id = AGENT_ID * "_v3", max_ticks = 200)
println("VolAdaptive MM ready — uncomment the run line to trade it.")
"""))

cells.append(md(r"""
## Wrap-up

1. **Spread is not free money** — it compensates for adverse selection.
2. **Inventory is risk** — a maker who ignores it is a directional trader with extra steps.
3. **Skewing quotes** is the textbook fix; **volatility** governs the optimal spread.
4. **Same Arena, any language** — your Julia `on_tick` speaks the same protocol as Python and R.

→ Next: Mission 3, Alpha Discovery.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Julia 1.10", "language": "julia", "name": "julia-1.10"},
                   "language_info": {"name": "julia", "file_extension": ".jl", "mimetype": "application/julia"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "missions",
                                   "mission_02_marketmaker", "notebook_julia.ipynb"))
with open(out, "w") as f:
    json.dump(nb, f, indent=1); f.write("\n")
print("wrote", out, "with", len(cells), "cells")
