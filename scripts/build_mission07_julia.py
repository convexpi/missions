#!/usr/bin/env python3
"""Builds missions/mission_07_queue_dynamics/notebook_julia.ipynb — bridged Julia port of the L3 sim."""
import json, os
PIN = "8af48e512e15a29a4b398b70a7ca9d9e812a2e04"

def md(s):   return {"cell_type": "markdown", "metadata": {}, "source": _lines(s)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(s)}
def _lines(s):
    s = s.strip("\n"); parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]

cells = []

cells.append(md(r"""
# Mission 7: Queue Dynamics (L3) — Julia edition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/convexpi/missions/blob/main/missions/mission_07_queue_dynamics/notebook_julia.ipynb)

**Advanced elective.** Mission 2 traded an *aggregated* (L2) book. Real exchanges are
**order-by-order (L3)**: every resting order has an identity and a FIFO queue place. Whether your
passive order makes money depends on *where you are in the queue, and whether the market moves before
you reach the front.*

**Learning objectives**
- Explain FIFO queue priority and why **queue position** is the maker's core asset
- Simulate a resting limit order order-by-order: drain the queue ahead, then fill
- Model the **cancel race**: latency vs **adverse selection**
- Observe queue dynamics on a **real Bitstamp BTC/USD L3 feed**

> **How this port works.** The L3 engine (`convexpi.arena.mbo`) is Python infrastructure with no
> native Julia equivalent, so we drive it through **`PyCall`** — you build event streams and read
> results as ordinary Julia objects, and interpret them natively. The mechanics are identical.
"""))

cells.append(md("## Part 0: Setup"))

cells.append(code(r"""
using Pkg; Pkg.add(["PyCall", "UnicodePlots", "Statistics"])
using PyCall, UnicodePlots, Statistics
# Pin the exact L3 engine commit so the queue mechanics match this lesson.
try
    pyimport("convexpi.arena.mbo")
catch
    run(`$(PyCall.python) -m pip install --quiet "git+https://github.com/convexpi/arena.git@%PIN%"`)
end
mbo = pyimport("convexpi.arena.mbo")
println("Ready.")
""".replace("%PIN%", PIN)))

cells.append(md(r"""
---
## Part 1: The L3 mental model

An L3 stream is `created`/`deleted` order events and `trade` events. Each order has `id`, price `p`,
amount `a`, side `s` (0 = bid, 1 = ask), microsecond `t`. `L3Book` replays them into a book where each
price level is a **FIFO list of order ids**.
"""))

cells.append(code(r"""
book = mbo.L3Book()
for oid in (101, 102, 103)                       # three 1.0 BTC bids at $60,000, in order
    book.apply(Dict("k"=>"o","e"=>"created","id"=>oid,"p"=>60000.0,"a"=>1.0,"s"=>0,"t"=>oid))
end
println("best bid / ask: ", book.best_bid(), " / ", book.best_ask())
println("size at \$60,000 bid: ", book.size_at(0, 60000.0), " BTC")
println("queue (front -> back): ", book.order_ids_at(0, 60000.0))

book.apply(Dict("k"=>"o","e"=>"deleted","id"=>101,"p"=>60000.0,"a"=>1.0,"s"=>0,"t"=>200))
println("after 101 cancels: ", book.order_ids_at(0, 60000.0), " -> ", book.size_at(0, 60000.0), " BTC")
"""))

cells.append(md(r"""
**A price level is not a number, it's a line of people.** Join with 2 BTC already resting and every
one must trade or cancel before a satoshi of yours fills.

---
## Part 2: Queue position to fill

`simulate_passive_order` rests `size` at `price` (side 0 = buy) at event index `enter_idx`, replays
order-by-order, and tracks the queue **ahead** until you fill or cancel.
"""))

cells.append(code(r"""
P = 60000.0
function build_stream()
    ev = []; t = 0
    for oid in (1, 2, 3)                          # 3 BTC resting ahead of us
        push!(ev, Dict("k"=>"o","e"=>"created","id"=>oid,"p"=>P,"a"=>1.0,"s"=>0,"t"=>t)); t += 1_000_000
    end
    enter = length(ev)                            # we join HERE (0-based) -> 3.0 BTC ahead
    push!(ev, Dict("k"=>"o","e"=>"created","id"=>9,"p"=>P-1,"a"=>5.0,"s"=>0,"t"=>t)); t += 1_000_000
    for _ in 1:5                                  # sell market orders hit the bid, 1 BTC every 2s
        push!(ev, Dict("k"=>"t","p"=>P,"a"=>1.0,"s"=>1,"t"=>t)); t += 2_000_000
    end
    (ev = ev, enter = enter)
end

s = build_stream()
r = mbo.simulate_passive_order(s.ev, side=0, price=P, enter_idx=s.enter, size=0.5)
println("queue ahead at entry : ", round(r.initial_queue_ahead, digits=2), " BTC")
println("filled               : ", r.filled)
println("time to fill         : ", round(r.time_to_fill_s, digits=1), " s")
"""))

cells.append(code(r"""
ts = [(x[1] - r.enter_ts) / 1e6 for x in r.queue_trace]
qa = [Float64(x[2]) for x in r.queue_trace]
lineplot(ts, qa, title = "Queue position draining to a fill",
         xlabel = "seconds since we joined", ylabel = "BTC ahead", width = 72, height = 12)
"""))

cells.append(md(r"""
**Exercise 2.1** — Predict the fill time, then set `size = 2.0` and re-run. Why does a larger order
take no longer to *start* filling — and what about partial fills?

---
## Part 3: The cancel race & adverse selection

Post a quote, decide to leave after `cancel_after_s`; your cancel takes `latency_us` to arrive. Fast
cancel escapes; slow cancel is filled inside the window — **adverse selection**.
"""))

cells.append(code(r"""
s = build_stream()
base = mbo.simulate_passive_order(s.ev, side=0, price=P, enter_idx=s.enter, size=0.5)
fast = mbo.simulate_passive_order(s.ev, side=0, price=P, enter_idx=s.enter, size=0.5,
                                  cancel_after_s=1.0, latency_us=100_000)   # decide @1s, lands +0.1s
slow = mbo.simulate_passive_order(s.ev, side=0, price=P, enter_idx=s.enter, size=0.5,
                                  cancel_after_s=7.0, latency_us=5_000_000) # decide @7s, lands +5s
for (nm, res) in [("no cancel", base), ("fast cancel", fast), ("slow cancel", slow)]
    outcome = res.filled ? "FILLED" : (res.cancelled ? "cancelled" : "open")
    println(rpad(nm, 12), outcome)
end
"""))

cells.append(md(r"""
The **fast cancel escapes**; the **slow cancel is filled before the cancel lands** — only latency
changed. That's why low latency is worth so much: to *get out in time*.

**Exercise 3.1** — Hold `cancel_after_s = 2.0` and sweep `latency_us` over
`[50_000, 500_000, 2_000_000, 6_000_000]`. Find the latency where the outcome flips.

---
## Part 4: Real data — a Bitstamp BTC/USD L3 feed

~75 seconds of a **real order-by-order feed**. The replay + sampling runs in the bridged L3 engine
for speed; we pull the summary back into Julia.
"""))

cells.append(code(r'''
py"""
import urllib.request
from convexpi.arena.mbo import L3Book, simulate_passive_order, load_l3
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/convexpi/arena/%PIN%/data/btcusd_l3_sample.jsonl",
    "btcusd_l3_sample.jsonl")
events = load_l3("btcusd_l3_sample.jsonl")
span_s = (events[-1]['t'] - events[0]['t']) / 1e6
book = L3Book(); samples = []
for i, e in enumerate(events):
    if e['k'] == 'o':
        book.apply(e)
    if i > 800 and i % 150 == 0:
        bb, ba = book.best_bid(), book.best_ask()
        if bb is not None and ba is not None and bb < ba:
            samples.append((i, bb))
ttfs = [r.time_to_fill_s for (i, bb) in samples
        for r in [simulate_passive_order(events, side=0, price=bb, enter_idx=i, size=0.02)] if r.filled]
adverse = {lat: sum(simulate_passive_order(events, side=0, price=bb, enter_idx=i, size=0.02,
                    cancel_after_s=0.2, latency_us=lat*1000).filled for (i, bb) in samples)
           for lat in [10, 100, 1000, 5000]}
n_samples = len(samples); n_events = len(events)
"""
println(py"n_events", " events over ", round(py"span_s", digits=1), " s")
println("clean entry points        : ", py"n_samples")
ttfs = py"ttfs"; nsamp = py"n_samples"
println("filled within the window  : ", length(ttfs), " (", round(100*length(ttfs)/max(nsamp,1)), "%)")
length(ttfs) > 0 && println("median time-to-fill       : ", round(median(ttfs), digits=2), " s")
println("\nadverse fills (cancel decided 0.2s after posting):")
for lat in [10, 100, 1000, 5000]
    println("  latency ", lpad(lat,5), " ms -> adverse fills: ", py"adverse"[lat], "/", nsamp)
end
'''.replace("%PIN%", PIN)))

cells.append(md(r"""
Even in a short sample: **slower cancels get adversely selected more often.** On a live venue that's
the line between a profitable and unprofitable maker.

---
## Part 5: Go live (optional)

The same dynamics run on the **Realistic Exchange (L3)** arena — real FIFO queue, latency on cancels.
Connect with the same Arena client as Mission 2 (`run_agent` from `ConvexPi`):
➡️ **[/compete/arena-l3](https://convexpi.ai/compete/arena-l3)**.

---
## Challenge

Build an **adverse-selection-aware quoting rule** and test it offline: rest a buy at the best bid but
cancel if the best ask ticks down past your price; compare fill and adverse-fill rates against a naive
always-rest quote, and find a threshold that beats naive on net. Publish to
**[/projects](https://convexpi.ai/projects)**.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Julia 1.10", "language": "julia", "name": "julia-1.10"},
                   "language_info": {"name": "julia", "file_extension": ".jl", "mimetype": "application/julia"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "missions",
                                   "mission_07_queue_dynamics", "notebook_julia.ipynb"))
with open(out, "w") as f:
    json.dump(nb, f, indent=1); f.write("\n")
print("wrote", out, "with", len(cells), "cells")
