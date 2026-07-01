# Mission 2: Market Maker

**Notebooks** — pick your language; all three trade the same live Arena:

| Language | Notebook |
|---|---|
| Python | `missions/mission_02_marketmaker/notebook.ipynb` |
| R | `missions/mission_02_marketmaker/notebook_r.ipynb` |
| Julia | `missions/mission_02_marketmaker/notebook_julia.ipynb` |

## Learning objectives

- Understand how spread income and inventory risk interact
- Experience adverse selection from the informed trader
- Implement inventory-aware quote skewing
- Decompose PnL into spread income vs mark-to-market

## The problem

A naive market maker posts symmetric quotes around mid and earns the spread from noise traders. But the informed trader — who has private information about the fundamental price — consistently takes the other side of those quotes at the worst moment. The market maker earns small credits from noise flow and suffers large losses from informed flow. Net PnL is negative.

## Parts

| Part | Task |
|---|---|
| 1. Setup | Install `convexpi`, connect to Arena as observer |
| 2. NaiveMarketMaker | Fixed half-spread, symmetric quotes, no inventory management |
| 3. Telemetry | Analyze fill rate, position drift, realized vs unrealized PnL |
| 4. InventoryAwareMarketMaker | Skew quotes by `position × skew_per_share` |
| 5. PnL attribution | Decompose into `spread_pnl` and `mtm_pnl` |
| 6. Challenge | VolAdaptiveMM — widen spread when realized volatility is high |

## Inventory skewing

When long, shift both bid and ask downward to mean-revert the position:

```python
skew = self.position * self.skew_per_share  # e.g. 0.02 cents per share
bid = mid - self.half_spread - skew
ask = mid + self.half_spread - skew
```

Size scaling:

```python
size_scale = max(0.2, 1 - abs(self.position) / self.max_position)
```

## PnL decomposition

```python
# Mark-to-market: inventory × daily price change
mtm_pnl = position.shift(1) * price_change / 100

# Spread: total PnL minus MTM
spread_pnl = total_pnl - mtm_pnl.cumsum()
```

A healthy market maker has positive `spread_pnl` (earning from noise) and small `mtm_pnl` (not being run over by informed flow).
