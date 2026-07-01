# smoke_test_arena.R — paste into a Colab R runtime (or run: Rscript smoke_test_arena.R)
# Confirms the two transports the R missions rely on:
#   1. the reticulate bridge to the Python market engine (Missions 1/3/4/5)
#   2. the Arena WebSocket client (Mission 2)
# A passing run prints "ALL SMOKE CHECKS PASSED". The Arena agent places NO orders (passive observer),
# so it does not touch the leaderboard.

if (!requireNamespace("convexpi", quietly = TRUE)) {
  if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
  remotes::install_github("convexpi/convexpi-r", upgrade = "never")
}
for (p in c("websocket", "later")) if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
library(convexpi)

# 1) Bridge check — pull the exact synthetic market the grader uses.
if (!reticulate::py_module_available("convexpi.lab")) reticulate::py_install("convexpi-lab", pip = TRUE)
m <- synthetic_market("train")
stopifnot(nrow(m$prices) > 100, ncol(m$prices) > 10, "mom_1m" %in% names(m$features))
cat(sprintf("[1/2] bridge OK  — market %d days x %d stocks, features: %s\n",
            nrow(m$prices), ncol(m$prices), paste(names(m$features)[1:3], collapse = ", ")))

# 2) WebSocket check — connect and observe 10 ticks (no orders placed).
noop <- function(state) list()
df <- run_agent(noop, agent_id = "smoke_r_readonly", max_ticks = 10)
stopifnot(nrow(df) >= 5, all(c("tick", "pnl", "position", "mid", "last_price") %in% names(df)))
cat(sprintf("[2/2] Arena WS OK — observed %d ticks, last mid $%.2f\n", nrow(df), tail(df$mid, 1)))

cat("ALL SMOKE CHECKS PASSED\n")
