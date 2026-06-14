# Missions

Missions are the core learning units. Each is a self-contained Jupyter notebook that walks students through a concept by having them build, break, and fix a real strategy.

The notebooks live in `missions/` in the repo and can be opened in Google Colab with one click.

| # | Mission | Core concept | Skills |
|---|---|---|---|
| 1 | [Overfitting Game](mission-01.md) | IS vs OOS Sharpe | Grid-search trap, grade report interpretation |
| 2 | [Market Maker](mission-02.md) | Inventory risk | Adverse selection, PnL attribution |
| 3 | [Alpha Discovery](mission-03.md) | Signal validation | IC analysis, FDR correction, walk-forward |

## Design principles

- Each mission has a **known failure mode** that students are expected to encounter and diagnose.
- Grading uses a **hidden holdout** — students cannot know the OOS period in advance.
- The **AI tutor** is available at `/tutor` for coaching on any mission.
- Missions build on each other: Mission 1 teaches overfitting before Mission 3 tests signal discovery.
