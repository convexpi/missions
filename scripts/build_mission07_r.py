#!/usr/bin/env python3
"""Builds missions/mission_07_queue_dynamics/notebook_r.ipynb — bridged R port of the L3 queue sim."""
import json, os

PIN = "8af48e512e15a29a4b398b70a7ca9d9e812a2e04"

def md(s):   return {"cell_type": "markdown", "metadata": {}, "source": _lines(s)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(s)}
def _lines(s):
    s = s.strip("\n"); parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]

cells = []

cells.append(md(r"""
# Mission 7: Queue Dynamics (L3) — R edition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/convexpi/missions/blob/main/missions/mission_07_queue_dynamics/notebook_r.ipynb)

**Advanced elective.** Mission 2 traded an *aggregated* (L2) book. Real exchanges are
**order-by-order (L3)**: every resting order has an identity and a place in a first-in-first-out
(FIFO) queue. Whether your passive order makes money depends on a question L2 can't ask — *where are
you in the queue, and will the market move before you reach the front?*

**Learning objectives**
- Explain FIFO queue priority and why **queue position** is the maker's core asset
- Simulate a resting limit order order-by-order: drain the queue ahead, then fill
- Model the **cancel race**: a maker pulling a quote against latency, and **adverse selection**
- Observe queue dynamics on a **real Bitstamp BTC/USD L3 feed**

> **How this port works.** The L3 matching engine (`convexpi.arena.mbo`) is Python infrastructure
> with no native R equivalent, so we drive it through **`reticulate`** — you build event streams and
> read results as ordinary R objects, and interpret them with native R. The mechanics are identical.
"""))

cells.append(md("## Part 0: Setup"))

cells.append(code(r"""
library(reticulate)
# Pin the exact L3 engine commit so the queue mechanics match this lesson.
if (!py_module_available("convexpi.arena.mbo")) {
  py_install("git+https://github.com/convexpi/arena.git@%PIN%", pip = TRUE)
}
mbo <- import("convexpi.arena.mbo")
cat("Ready.\n")
""".replace("%PIN%", PIN)))

cells.append(md(r"""
---
## Part 1: The L3 mental model

An L3 stream is a sequence of **order** events (`created`/`deleted`) and **trade** events. Each order
has an `id`, price `p`, remaining amount `a`, side `s` (0 = buy/bid, 1 = sell/ask) and a microsecond
timestamp `t`. `L3Book` replays these into a book where each price level is a **FIFO list of order
ids**. Build a tiny book by hand and look at the queue at one price.
"""))

cells.append(code(r"""
book <- mbo$L3Book()
# Three makers post 1.0 BTC bids at $60,000, in this order.
for (oid in c(101L, 102L, 103L)) {
  book$apply(list(k = "o", e = "created", id = oid, p = 60000, a = 1.0, s = 0L, t = oid))
}
cat("best bid / ask:", book$best_bid(), "/", book$best_ask(), "\n")
cat("size at $60,000 bid:", book$size_at(0L, 60000), "BTC\n")
cat("queue (front -> back):", unlist(book$order_ids_at(0L, 60000)), "\n")

# Order 101 cancels; the queue closes up and 102 is now at the front.
book$apply(list(k = "o", e = "deleted", id = 101L, p = 60000, a = 1.0, s = 0L, t = 200L))
cat("after 101 cancels:", unlist(book$order_ids_at(0L, 60000)), "->", book$size_at(0L, 60000), "BTC\n")
"""))

cells.append(md(r"""
**A price level is not a number, it's a line of people.** Join the bid at \$60,000 with 2 BTC already
resting and you are *behind* 2 BTC — every one must trade or cancel before a satoshi of yours fills.

---
## Part 2: Queue position to fill

`simulate_passive_order` rests `size` at `price` (side 0 = buy) at event index `enter_idx`, replays
the stream order-by-order, and tracks how much queue is **ahead** until you fill or cancel. Scenario:
3 BTC rest ahead of us, then sell market orders hit the bid and drain the queue.
"""))

cells.append(code(r"""
P <- 60000
build_stream <- function() {
  ev <- list(); t <- 0L
  for (oid in c(1L, 2L, 3L)) {                 # 3 BTC resting ahead of us on the bid
    ev[[length(ev) + 1L]] <- list(k = "o", e = "created", id = oid, p = P, a = 1.0, s = 0L, t = t); t <- t + 1000000L
  }
  enter <- length(ev)                          # we join HERE (0-based index) -> 3.0 BTC ahead
  ev[[length(ev) + 1L]] <- list(k = "o", e = "created", id = 9L, p = P - 1, a = 5.0, s = 0L, t = t); t <- t + 1000000L
  for (i in 1:5) {                             # sell market orders hit the bid, 1 BTC each, every 2s
    ev[[length(ev) + 1L]] <- list(k = "t", p = P, a = 1.0, s = 1L, t = t); t <- t + 2000000L
  }
  list(ev = ev, enter = as.integer(enter))
}

s <- build_stream()
r <- mbo$simulate_passive_order(s$ev, side = 0L, price = P, enter_idx = s$enter, size = 0.5)
cat(sprintf("queue ahead at entry : %.2f BTC\n", r$initial_queue_ahead))
cat("filled               :", r$filled, "\n")
cat(sprintf("time to fill         : %.1f s\n", r$time_to_fill_s))
"""))

cells.append(code(r"""
# Visualise the queue ahead of us draining to zero, then our fill.
trace <- r$queue_trace
ts <- sapply(trace, function(x) (x[[1]] - r$enter_ts) / 1e6)
qa <- sapply(trace, function(x) x[[2]])
plot(ts, qa, type = "s", lwd = 2, col = "steelblue",
     xlab = "seconds since we joined the queue", ylab = "BTC ahead of us",
     main = "Queue position draining to a fill")
abline(h = 0, lwd = 0.8)
if (!is.null(r$fill_ts)) points((r$fill_ts - r$enter_ts) / 1e6, 0, col = "crimson", pch = 19)
"""))

cells.append(md(r"""
**Exercise 2.1** — You rested 0.5 BTC behind 3.0 BTC; each market order was 1 BTC every 2 s. Predict
the fill time, then set `size = 2.0` and re-run. Why does a larger order take no longer to *start*
filling — and what should you think about for partial fills?

---
## Part 3: The cancel race & adverse selection

You post a quote, then after `cancel_after_s` decide to leave; your cancel takes `latency_us` to
reach the exchange. **Fast cancel** escapes; **slow cancel** gets filled inside the latency window —
you were **adversely selected**.
"""))

cells.append(code(r"""
s <- build_stream()
base <- mbo$simulate_passive_order(s$ev, side = 0L, price = P, enter_idx = s$enter, size = 0.5)
fast <- mbo$simulate_passive_order(s$ev, side = 0L, price = P, enter_idx = s$enter, size = 0.5,
                                   cancel_after_s = 1.0, latency_us = 100000L)   # decide @1s, lands +0.1s
slow <- mbo$simulate_passive_order(s$ev, side = 0L, price = P, enter_idx = s$enter, size = 0.5,
                                   cancel_after_s = 7.0, latency_us = 5000000L)  # decide @7s, lands +5s
for (nm in c("no cancel", "fast cancel", "slow cancel")) {
  res <- switch(nm, "no cancel" = base, "fast cancel" = fast, "slow cancel" = slow)
  outcome <- if (isTRUE(res$filled)) "FILLED" else if (isTRUE(res$cancelled)) "cancelled" else "open"
  cat(sprintf("%-12s %s\n", nm, outcome))
}
"""))

cells.append(md(r"""
The **fast cancel escapes**; the **slow cancel is filled before the cancel lands** — same market,
only latency changed. That's why low latency is worth so much: not just to be first in the queue, but
to *get out in time*.

**Exercise 3.1** — Hold `cancel_after_s = 2.0` and sweep `latency_us` over
`c(50000, 500000, 2000000, 6000000)`. Find the latency where the outcome flips from cancelled to
filled — your adverse-selection boundary for this scenario.

---
## Part 4: Real data — a Bitstamp BTC/USD L3 feed

Now ~75 seconds of a **real order-by-order Bitstamp feed**. The replay + entry-point sampling runs in
the (bridged) L3 engine for speed; we pull the summary back into R.
"""))

cells.append(code(r"""
download.file(paste0("https://raw.githubusercontent.com/convexpi/arena/%PIN%/data/btcusd_l3_sample.jsonl"),
              "btcusd_l3_sample.jsonl", quiet = TRUE)
events <- mbo$load_l3("btcusd_l3_sample.jsonl")
span_s <- (events[[length(events)]]$t - events[[1]]$t) / 1e6
cat(sprintf("%d events over %.1f s\n", length(events), span_s))

# Sample entry points and time-to-fill, plus adverse fills vs cancel latency — done in Python for
# speed (thousands of order-by-order steps), returned as plain numbers.
py$events <- events
reticulate::py_run_string("
from convexpi.arena.mbo import L3Book, simulate_passive_order
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
adverse = {}
for lat_ms in [10, 100, 1000, 5000]:
    adverse[lat_ms] = sum(simulate_passive_order(events, side=0, price=bb, enter_idx=i, size=0.02,
                          cancel_after_s=0.2, latency_us=lat_ms*1000).filled for (i, bb) in samples)
n_samples = len(samples)
")
cat(sprintf("clean entry points        : %d\n", py$n_samples))
cat(sprintf("filled within the window  : %d (%.0f%%)\n", length(py$ttfs),
            100 * length(py$ttfs) / max(py$n_samples, 1)))
if (length(py$ttfs) > 0) cat(sprintf("median time-to-fill       : %.2f s\n", median(unlist(py$ttfs))))
cat("\nadverse fills (cancel decided 0.2s after posting):\n")
for (lat in c("10", "100", "1000", "5000"))
  cat(sprintf("  latency %5s ms -> adverse fills: %d/%d\n", lat, py$adverse[[lat]], py$n_samples))
""".replace("%PIN%", PIN)))

cells.append(md(r"""
Even in a short sample the direction is unmistakable: **slower cancels get adversely selected more
often.** On a live venue this is the difference between a profitable and unprofitable maker.

---
## Part 5: Go live (optional)

The same dynamics run on the **Realistic Exchange (L3)** arena — your limit orders take a real FIFO
place and only fill at the front, with latency on your cancels. Connect with the same Arena client as
Mission 2 (`run_agent` from the `convexpi` R package):
➡️ **[/compete/arena-l3](https://convexpi.ai/compete/arena-l3)**.

---
## Challenge

Build an **adverse-selection-aware quoting rule** and test it offline on the real data: rest a buy at
the best bid, but cancel if the best ask ticks down past your price; compare fill rate and adverse-fill
rate against a naive always-rest quote, and find a threshold that beats naive on net. Publish it to
**[/projects](https://convexpi.ai/projects)**.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "R", "language": "R", "name": "ir"},
                   "language_info": {"name": "R"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "missions",
                                   "mission_07_queue_dynamics", "notebook_r.ipynb"))
with open(out, "w") as f:
    json.dump(nb, f, indent=1); f.write("\n")
print("wrote", out, "with", len(cells), "cells")
