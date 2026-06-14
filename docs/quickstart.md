# Quickstart

## Install

```bash
git clone https://github.com/convexpi/lab.git
cd aiinfinance
pip install -e .
```

## Run the Arena demo

A 2000-tick in-process simulation with 13 background agents and a mid-session volatility shock:

```bash
python examples/arena_demo.py
```

## Run the Lab demo

Generate a synthetic equity panel, plant alpha signals, and grade 6 strategies:

```bash
python examples/lab_demo.py
```

## Run the Arena server (local)

```bash
convexpi-server                            # WebSocket on :8765, health check on :8766
convexpi-server --tick-interval 0.2       # fast classroom demo
convexpi-server --admin-token secret      # enable instructor console
```

Watch the order book in a second terminal:

```bash
convexpi-viz
```

## Connect an agent

```python
from convexpi.arena import RemoteAgent

class MyAgent(RemoteAgent):
    def on_tick(self, state):
        if state.mid and state.position < 50:
            return [self.limit('buy', round(state.mid) - 5, 5)]
        return []

MyAgent('alice').start()
```

## Grade a strategy directly

```python
import numpy as np
from convexpi.lab import SyntheticMarket, Strategy, Grader

market = SyntheticMarket(seed=42)

class MyStrategy(Strategy):
    def on_day(self, day, features, prices, portfolio):
        # features: dict of {name: np.ndarray(n_stocks,)}
        # Return target weights as np.ndarray(n_stocks,)
        sig = features.get('mom_1m', np.zeros(len(prices)))
        sig = np.nan_to_num(sig)
        total = np.abs(sig).sum()
        return sig / total if total > 0 else np.zeros(len(prices))

report = Grader(market).evaluate(MyStrategy())
report.print()
```

Output includes IS/OOS Sharpe, overfitting ratio, max drawdown, annualized return, turnover, and a breakdown of which planted alphas were discovered.

## Run the web app (local)

```bash
cd web
cp .env.local.example .env.local   # fill in Supabase credentials
npm install
npm run dev -- --port 3001
```

Apply the database schema once in the Supabase SQL editor:

```sql
-- paste contents of web/supabase/schema.sql
```

## Run the grader worker (local)

```bash
SUPABASE_URL=https://xxx.supabase.co \
SUPABASE_SERVICE_KEY=eyJ... \
MARKET_SEED=42 \
python deploy/grader_worker.py
```

The worker polls Supabase for `pending` submissions, runs them in a subprocess sandbox, and writes grade reports back.

## Run tests

```bash
# Python tests (316 total)
pytest tests/ -v

# Web tests (100 total)
cd web && npx vitest run
```
