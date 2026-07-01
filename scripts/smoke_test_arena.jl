# smoke_test_arena.jl — paste into a Colab Julia runtime (or run: julia smoke_test_arena.jl)
# Confirms the two transports the Julia missions rely on:
#   1. the PyCall bridge to the Python market engine (Missions 1/3/4/5)
#   2. the Arena WebSocket client (Mission 2)
# A passing run prints "ALL SMOKE CHECKS PASSED". The Arena agent places NO orders (passive observer).

using Pkg
Pkg.add(url = "https://github.com/convexpi/ConvexPi.jl")
Pkg.add(["PyCall"])
using ConvexPi, PyCall

# make sure the Python market engine is importable by PyCall
try
    pyimport("convexpi.lab")
catch
    run(`$(PyCall.python) -m pip install --quiet convexpi-lab`)
end

# 1) Bridge check — pull the exact synthetic market the grader uses.
m = synthetic_market("train")
@assert size(m.prices, 1) > 100 && size(m.prices, 2) > 10 && haskey(m.features, "mom_1m")
println("[1/2] bridge OK  — market ", size(m.prices, 1), " days x ", size(m.prices, 2),
        " stocks, has mom_1m: ", haskey(m.features, "mom_1m"))

# 2) WebSocket check — connect and observe 10 ticks (no orders placed).
noop(state) = Dict[]
df = run_agent(noop; agent_id = "smoke_jl_readonly", max_ticks = 10)
@assert length(df) >= 5 && hasproperty(df[end], :mid)
println("[2/2] Arena WS OK — observed ", length(df), " ticks, last mid \$", round(df[end].mid, digits = 2))

println("ALL SMOKE CHECKS PASSED")
