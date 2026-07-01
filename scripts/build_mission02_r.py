#!/usr/bin/env python3
"""Builds missions/mission_02_marketmaker/notebook_r.ipynb — the faithful R port (live Arena)."""
import json, os

def md(s):   return {"cell_type": "markdown", "metadata": {}, "source": _lines(s)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(s)}
def _lines(s):
    s = s.strip("\n"); parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]

cells = []

cells.append(md(r"""
# Mission 2: Market-Maker in the Arena — R edition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/convexpi/missions/blob/main/missions/mission_02_marketmaker/notebook_r.ipynb)

**Learning objectives**
- Understand the economics of market-making: earning the spread vs. adverse selection
- Build a **live** agent with the R `convexpi` Arena client (`run_agent`)
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
if (!requireNamespace("convexpi", quietly = TRUE)) {
  if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
  remotes::install_github("convexpi/convexpi-r", upgrade = "never")
}
# The Arena client trades over WebSocket — needs these two packages.
for (p in c("websocket", "later")) if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
library(convexpi)

AGENT_ID <- "mm_r_student01"   # <- change to your unique handle (this is your name on the leaderboard)
cat("Ready.\n")
"""))

cells.append(md(r"""
## Part 1: The agent contract

Your agent is a function `on_tick(state)` that returns a list of orders. Each tick the server hands
you a read-only `state`:

| field | meaning |
|---|---|
| `state$best_bid` / `state$best_ask` | top of book, in **integer cents** (`NULL` if empty) |
| `state$mid` / `state$spread` | convenience values |
| `state$last_price` | last trade price (cents) |
| `state$position`, `state$cash` | your signed inventory and cash (cents) |
| `state$my_open_orders` | `list(list(order_id=, side=, price=, qty=), ...)` |

Order helpers: `arena_limit(side, price, qty)`, `arena_market_order(side, qty)`, `arena_cancel(order_id)`.
`run_agent(on_tick, agent_id, max_ticks)` connects, runs, and returns a telemetry data frame.
"""))

cells.append(md(r"""
## Part 2: A naive market maker

Quote a fixed half-spread around the mid, re-quoting each tick. No inventory management.
"""))

cells.append(code(r"""
naive_mm <- function(state) {
  if (is.null(state$best_bid) || is.null(state$best_ask)) return(list())
  mid <- (state$best_bid + state$best_ask) %/% 2
  orders <- lapply(state$my_open_orders, function(o) arena_cancel(o$order_id))   # pull old quotes
  c(orders, list(arena_limit("buy", mid - 5, 10), arena_limit("sell", mid + 5, 10)))
}

# Runs live for 200 ticks (~a few minutes at 1 tick/s) and returns telemetry.
df <- run_agent(naive_mm, agent_id = AGENT_ID, max_ticks = 200)
cat("collected", nrow(df), "ticks\n"); tail(df, 3)
"""))

cells.append(md("## Part 3: Analyse telemetry"))

cells.append(code(r"""
plot_telem <- function(df, title = "Naive MM") {
  if (nrow(df) == 0) { cat("No data — check the Arena connection.\n"); return(invisible()) }
  rel <- df$pnl - df$pnl[1]                      # PnL relative to the start
  op <- par(mfrow = c(3, 1), mar = c(3, 4, 2, 1))
  plot(df$tick, rel, type = "l", col = "steelblue", lwd = 1.5, ylab = "PnL ($)", main = title)
  abline(h = 0, col = "grey", lwd = 0.5)
  plot(df$tick, df$position, type = "l", col = "darkorange", lwd = 1.2, ylab = "inventory")
  abline(h = 0, col = "grey", lwd = 0.5)
  plot(df$tick, df$last_price, type = "l", col = "grey", ylab = "price ($)", xlab = "tick")
  par(op)
  cat(sprintf("Final PnL: $%.2f   Max |inventory|: %d shares\n", rel[length(rel)], max(abs(df$position))))
}
plot_telem(df, "Naive MM")
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
make_inventory_mm <- function(half_spread = 6, qty = 10, skew = 0.5, max_pos = 50) function(state) {
  if (is.null(state$best_bid) || is.null(state$best_ask)) return(list())
  pos <- state$position
  orders <- lapply(state$my_open_orders, function(o) arena_cancel(o$order_id))
  if (abs(pos) >= max_pos) return(orders)                      # at the cap: quote nothing new
  mid <- (state$best_bid + state$best_ask) / 2
  skewed <- mid - pos * skew                                   # long -> push quotes down
  bid <- round(skewed - half_spread); ask <- round(skewed + half_spread)
  if (bid >= state$best_ask || ask <= state$best_bid) return(orders)   # never cross
  q <- max(1, round(qty * max(0.2, 1 - abs(pos) / max_pos)))   # shrink size as inventory grows
  c(orders, list(arena_limit("buy", bid, q), arena_limit("sell", ask, q)))
}

df2 <- run_agent(make_inventory_mm(), agent_id = paste0(AGENT_ID, "_v2"), max_ticks = 200)
plot_telem(df2, "Inventory-aware MM")
"""))

cells.append(md(r"""
## Part 5: PnL attribution

Split PnL into inventory mark-to-market (price moves on the shares you hold) vs. the residual, which
is your spread income.
"""))

cells.append(code(r"""
attribute_pnl <- function(df) {
  rel <- df$pnl - df$pnl[1]
  price_change <- c(0, diff(df$last_price))
  mtm <- c(0, head(df$position, -1)) * price_change            # yesterday's inventory x today's move
  data.frame(tick = df$tick, total = rel, mtm = cumsum(mtm), spread = rel - cumsum(mtm))
}
if (nrow(df2) > 0) {
  a <- attribute_pnl(df2)
  plot(a$tick, a$total, type = "l", lwd = 2, col = "black", xlab = "tick", ylab = "PnL ($)",
       main = "PnL attribution", ylim = range(c(a$total, a$mtm, a$spread)))
  lines(a$tick, a$spread, lty = 2, lwd = 1.5, col = "steelblue")
  lines(a$tick, a$mtm, lty = 3, lwd = 1.5, col = "tomato")
  abline(h = 0, col = "grey", lwd = 0.5)
  legend("topleft", c("Total", "Spread income", "Inventory MTM"),
         col = c("black", "steelblue", "tomato"), lty = c(1, 2, 3), lwd = c(2, 1.5, 1.5), bty = "n")
}
"""))

cells.append(md(r"""
## Part 6: Challenges

- **A (easy):** volatility-adaptive spread — widen when recent prices are choppy (starter below).
- **B (medium):** back off after a streak of adverse fills (use the `on_fill` callback of `run_agent`).
- **C (hard):** estimate informed-flow pressure from the buy/sell imbalance in `state$recent_trades`
  and widen when it's high.
"""))

cells.append(code(r"""
# Challenge A starter — a volatility-adaptive maker (a closure that remembers recent prices).
make_vol_adaptive_mm <- function(k = 0.5, qty = 10, skew = 0.5, max_pos = 50) {
  prices <- numeric(0)
  function(state) {
    if (is.null(state$best_bid) || is.null(state$best_ask)) return(list())
    if (!is.null(state$last_price)) prices <<- tail(c(prices, state$last_price), 20)
    hs <- if (length(prices) >= 5) max(3, round(k * sd(prices) / 100)) else 6
    pos <- state$position
    orders <- lapply(state$my_open_orders, function(o) arena_cancel(o$order_id))
    if (abs(pos) >= max_pos) return(orders)
    mid <- (state$best_bid + state$best_ask) / 2; skewed <- mid - pos * skew
    bid <- round(skewed - hs); ask <- round(skewed + hs)
    if (bid >= state$best_ask || ask <= state$best_bid) return(orders)
    q <- max(1, round(qty * max(0.2, 1 - abs(pos) / max_pos)))
    c(orders, list(arena_limit("buy", bid, q), arena_limit("sell", ask, q)))
  }
}
# df3 <- run_agent(make_vol_adaptive_mm(), agent_id = paste0(AGENT_ID, "_v3"), max_ticks = 200)
cat("VolAdaptive MM ready — uncomment the run line to trade it.\n")
"""))

cells.append(md(r"""
## Wrap-up

1. **Spread is not free money** — it compensates for adverse selection.
2. **Inventory is risk** — a maker who ignores it is a directional trader with extra steps.
3. **Skewing quotes** is the textbook fix; **volatility** governs the optimal spread.
4. **Same Arena, any language** — your R `on_tick` speaks the same protocol as Python and Julia.

→ Next: Mission 3, Alpha Discovery.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "R", "language": "R", "name": "ir"},
                   "language_info": {"name": "R"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "missions",
                                   "mission_02_marketmaker", "notebook_r.ipynb"))
with open(out, "w") as f:
    json.dump(nb, f, indent=1); f.write("\n")
print("wrote", out, "with", len(cells), "cells")
