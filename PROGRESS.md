# Progress — rl-betfair

## Completed Sessions

### Session 0.1 — Repo scaffold, config & ProgressTracker
**Status:** Done
**Commit:** Initial repo setup

- Python venv, requirements.txt, .gitignore, .env.example, config.yaml, conftest.py
- `training/progress_tracker.py` — rolling-window ETA tracker with `to_dict()` for WebSocket
- Config discussion: learning rate capped at 5e-4 (PPO+LSTM stability), LSTM sizes up to 2048 (RTX 3090 headroom), observation window up to 360 ticks (full 30-min pre-race)

### Session 0.2 — Data extractor
**Status:** Done

- `data/extractor.py` — MySQL → Parquet via `ResolvedMarketSnaps.SnapJson`
- Market-level fields (venue, inPlay, marketTime, etc.) parsed from SnapJson in Python — avoids timestamp join with `updates` table that produces zero rows
- Both pre-race and in-play ticks extracted (bet restriction enforced by env, not extractor)
- Two Parquet files per day: ticks + runner metadata

### Session 0.3 — Episode builder & feature engineer
**Status:** Done

- `data/episode_builder.py` — Day → [Race → [Tick]] typed dataclasses
- `parse_snap_json` handles both real nested layout and flat test layout
- `data/feature_engineer.py` — 25 market features, 6 velocity features, 93 per-runner features (tick, metadata, cross-runner, velocity), TickHistory for temporal deltas
- NaN handling: velocity features on early ticks and BSP/SPN/SPF replaced with 0.0 in the env

### Session 0.4 — Order book & bet manager
**Status:** Done

- `env/order_book.py` — match_back consumes AvailableToLay, match_lay consumes AvailableToBack, partial fills walk levels 0→1→2, unmatched cancelled
- `env/bet_manager.py` — budget, open liability, realised P&L tracking. Back deducts stake; lay reserves liability = stake × (price − 1). settle_race handles winner/loser P&L for both sides

### Session 0.5 — Gymnasium environment
**Status:** Done

- `env/betfair_env.py` — one episode = one racing day, budget carries across races
- Observation: 1338-dim (market 25 + velocity 6 + runners 93×14 + agent state 5)
- Action: 28-dim Box (14 action_signal + 14 stake_fraction)
- Reward: race P&L + early pick bonus − efficiency penalty + end-of-day bonus
- Proportional staking: stake = fraction of current budget (compounding)

**Decision:** Agent observes ALL ticks (pre-race + in-play) but can only place bets on pre-race ticks. In-play price movement is valuable signal for learning about future races.

**Decision:** Budget carries across races within a day (not per-race reset). The agent must learn to manage its bankroll across the full day.

### Session 1.1 — Data extraction from real DB
**Status:** Done

- Extracted 2026-03-26: 4,182 ticks, 53 markets (WIN + EACH_WAY), 497 runners
- 27 schema/quality tests + 10 integration tests spot-checking against DB
- MySQL on localhost:3306 (Docker, hotDataRefactored + coldData)

### Session 1.2 — Policy network (architecture v1: PPO + LSTM)
**Status:** Done

- `agents/policy_network.py` — PPOLSTMPolicy registered as "ppo_lstm_v1"
- Per-runner shared MLP encoder → mean+max pooling → market MLP → LSTM → actor (per-runner) + critic (global)
- Hidden state carries across ticks AND races within a day
- Orthogonal init (gain=sqrt(2) general, 0.01 actor, 1.0 critic)
- 63 unit tests + 9 integration tests

### Session 1.3 + 1.4 — PPO trainer + Model registry & scoreboard
**Status:** Done

- `agents/ppo_trainer.py` — rollout collection with LSTM hidden state, GAE (gamma=0.99, lambda=0.95), clipped surrogate + value loss + entropy bonus, gradient clipping, mini-batch PPO updates. ProgressTracker integration, JSON-lines logging, asyncio.Queue progress events. 36 unit tests.
- `registry/model_store.py` — SQLite with 4 tables (models, evaluation_runs, evaluation_days, evaluation_bets). Save/load PyTorch weights. Full CRUD. 24 unit tests.
- `registry/scoreboard.py` — composite score (35% win_rate + 30% sharpe + 15% mean_daily_pnl + 20% efficiency). Ranking, discard candidate detection. 22 unit tests.
- 8 integration tests covering train→save→load→score pipeline on real data.

**Improvement:** Added `market_name`, `market_type`, `n_runners` to `Race` dataclass. Extractor now pulls `market_type` from SnapJson and `market_name` from coldData.marketOnDates. Re-extracted 2026-03-26 with new columns.

---

## Skipped / Deferred Sessions

### Session 1.5 — End-to-end single agent run
**Status:** Blocked — requires 2+ days of extracted data

The evaluation methodology requires a chronological train/test split (earliest ~50% train, later ~50% test). With only one day (2026-03-26), we cannot create a meaningful split. Will complete once a second day of data is available.

**Dependencies:** Run `python -m data.extractor` after another race day has been recorded by StreamRecorder1.

---

## Test Count

| Session | Unit tests added | Integration tests added | Running total |
|---------|-----------------|------------------------|---------------|
| 0.1     | 38              | 2                      | 40            |
| 0.2     | 49              | 10                     | 99            |
| 0.3     | 90              | 10                     | 199           |
| 0.4     | 61              | 10                     | 270           |
| 0.5     | 36              | 7                      | 313           |
| 1.1     | 27              | 10                     | 350           |
| 1.2     | 63              | 9                      | 422           |
| 1.3+1.4 | 82              | 8                      | 512           |
| *Misc*  | 12              | —                      | **524 + 57**  |

**Current total: 524 unit + 57 integration = 581 tests, all passing.**

---

## Key Decisions & Diversions

1. **SnapJson over updates join** (Session 0.2) — The `updates` and `ResolvedMarketSnaps` tables record timestamps independently. An exact join produces zero rows. Market-level fields are now extracted from SnapJson in Python instead.

2. **Both WIN and EACH_WAY markets** (Session 1.1) — The model plays both market types. No filtering by market type.

3. **Proportional staking** (Session 0.5) — Stake = fraction of current budget, not fixed £. This means winning days compound and losing days naturally shrink stakes.

4. **In-play observation, pre-race betting** (Session 0.5) — The agent sees in-play ticks (valuable signal about how races resolve) but can only place bets before the off. This is both realistic (Betfair delays in-play) and informative (the model learns what happens after the off to improve pre-race decisions in later races).

5. **Race dataclass enrichment** (Session 1.3+1.4) — Added `market_name`, `market_type`, `n_runners` to Race. These were missing from the original design but are useful for evaluation reporting and filtering.

6. **Learning rate cap at 5e-4** (Session 0.1) — Higher LRs destabilise PPO+LSTM training. The original PLAN.md range of 1e-3 was tightened during config review.
