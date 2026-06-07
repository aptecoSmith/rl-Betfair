---
id: 01KTJ0K80R3GBZ7XR96VPF1WZB
type: concept
cloud: shared
status: seed
created: 2026-06-07
updated: 2026-06-07
tags: [work]
sources: [src-04294a]
aliases: [BetManager, bet-manager, bet_manager]
---

# BetManager

`env/bet_manager.py::BetManager` — tracks every matched order in the current race ("bet count" = distinct matched orders, not netted positions). Re-created per race; `env.bet_manager.bets` is last-race-only. For the full-day bet history read `env.all_settled_bets`. Exposes accessors used by the reward path, including the `get_naked_per_pair_pnls(market_id)` accessor added by the [[naked-pnl-asymmetry-per-pair-fix]]. Stub — see CLAUDE.md §"Bet accounting" for the full contract. [[glossary]]
