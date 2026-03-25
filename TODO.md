# TODO — rl-betfair

## Session Rules (apply to every item below)

Each TODO item is its own Claude Code session. Before closing a session:
1. **Test** — every item must have automated tests (pytest) that pass
2. **Commit** — `git commit` with a descriptive message
3. **Push** — `git push` to `aptecoSmith/rl-Betfair` on GitHub

No item is done until all three steps are complete.

---

## Phase 0 — Foundation (no live data yet)

### Session 0.1 — Repo scaffold & config
- Create Python virtual environment (`python -m venv .venv`)
- Write `requirements.txt` (torch, gymnasium, stable-baselines3, pymysql,
  sqlalchemy, pandas, pyarrow, fastapi, uvicorn, pydantic, pytest, python-dotenv)
- Write `config.yaml` (DB connection, population size, reward coefficients,
  paths, architecture name)
- Write `.gitignore` (venv, processed data, model weights, `*.env`, `__pycache__`)
- Write `.env.example` (DB password placeholder)
- Write `conftest.py` (shared pytest fixtures)
- **Test:** pytest confirms config loads correctly, all required keys present

### Session 0.2 — Data extractor
- `data/extractor.py` — connect to MySQL (localhost:3307), query
  `ResolvedMarketSnaps` joined with ColdData tables (runner metadata, weather,
  results, market catalogue), export one Parquet file per day to `data/processed/`
- **Test:** pytest with a mock MySQL connection — verify SQL queries, correct
  join logic, output schema matches expected Parquet columns

### Session 0.3 — Episode builder
- `data/episode_builder.py` — load a day's Parquet, construct
  `Day → [Race → [Tick]]` hierarchy as typed dataclasses
- `data/feature_engineer.py` — derive price velocity, implied probability,
  overround, cross-runner relative features (rank by price, price gaps)
- **Test:** pytest with synthetic Parquet data — verify Day/Race/Tick structure,
  feature values, edge cases (single runner, missing prices, removed runners)

### Session 0.4 — Order book & bet manager
- `env/order_book.py` — realistic bet matching: back bets consume
  AvailableToLay volume level by level; lay bets consume AvailableToBack;
  partial fills supported; unmatched remainder cancelled
- `env/bet_manager.py` — track open bets, matched size, realised P&L,
  remaining budget, lay liability
- **Test:** pytest with known order book snapshots — verify full fills, partial
  fills, no-fill cases, liability calculation, budget enforcement

### Session 0.5 — Gymnasium environment
- `env/betfair_env.py` — Gymnasium `Env` subclass:
  - Observation space: full state vector (see PLAN.md State Representation)
  - Action space: per-runner (action_type, stake), masked for budget
  - Step: advance one tick, apply actions, compute reward
  - Episode: one full day across all races; budget carries across races
  - Reward: per-race P&L + early pick bonus + efficiency penalty (all
    coefficients from config.yaml)
- **Test:** pytest with synthetic episodes — verify observation shape, action
  masking, reward calculation for known scenarios, episode termination,
  budget carry-across-races

---

## Phase 1 — First Agent (requires 2+ days of data)

### Session 1.1 — Data extraction from real DB
- Run `data/extractor.py` against live MySQL (localhost:3307)
- Verify Parquet output for first real day(s) — check row counts, spot-check
  prices against known race data
- **Test:** pytest validates Parquet schema and non-null key fields on real output

### Session 1.2 — Policy network (architecture v1: PPO + LSTM)
- `agents/policy_network.py` — implement the v1 architecture (documented in
  PLAN.md Architecture section):
  - Per-runner MLP encoder (shared weights, permutation-invariant)
  - Market-level MLP encoder
  - LSTM (carries hidden state across ticks and across races within a day)
  - Actor head: per-runner (action logits + stake magnitude)
  - Critic head: scalar V(s)
- Architecture registered by name (`"ppo_lstm_v1"`) via the architecture
  registry (see PLAN.md) so future architectures can be swapped via config
- **Test:** pytest — forward pass with correct input shapes, output shapes,
  hidden state carries correctly, gradient flows through all components

### Session 1.3 — PPO trainer (single agent)
- `agents/ppo_trainer.py` — PPO training loop for one agent:
  rollout collection, advantage estimation (GAE), policy + value loss,
  gradient clipping, entropy bonus
- Logs per-episode: reward, P&L, bet count, loss terms → `logs/` directory
- **Test:** pytest — trainer runs for N steps on synthetic env without error,
  loss decreases over a trivial known environment

### Session 1.4 — Model registry
- `registry/model_store.py` — SQLite schema and CRUD:
  `models`, `evaluation_runs`, `evaluation_days`, `evaluation_bets` tables
  (full schema in PLAN.md)
- `registry/scoreboard.py` — compute composite score from per-day metrics
  (win_rate, sharpe, mean_daily_pnl, efficiency), rank all active models
- **Test:** pytest — create models, save/load weights, record evaluation runs,
  verify composite score formula, verify ranking order

### Session 1.5 — End-to-end single agent run
- Train one agent on first real training days
- Evaluate on first real test days — write results to registry
- Inspect scoreboard output, bet log, per-day P&L
- Verify nothing obviously broken (reward signal sensible, bets being matched,
  P&L not exploding)
- **Test:** pytest integration test — full train→evaluate→registry pipeline
  completes without error on real data

---

## Phase 2 — Population & Genetics

### Session 2.1 — Population manager
- `agents/population_manager.py` — initialise N agents (N from config) each
  with randomised hyperparameters drawn from defined search ranges
- Hyperparameter schema: learning_rate, ppo_clip, entropy_coeff, lstm_hidden,
  mlp_layers, mlp_hidden, reward_early_pick_bonus, reward_efficiency_penalty,
  observation_window_ticks (see PLAN.md)
- **Test:** pytest — population initialises with correct size, all hyperparams
  within valid ranges, no two agents identical

### Session 2.2 — Genetic selection
- Implement tournament selection: top 50% by composite score survive
- Implement elitism: top N_elite agents (config) always survive unchanged
- Discard policy: mark models as `discarded` in registry only if win_rate,
  mean_pnl, AND sharpe all fall below thresholds (never discard for bad days alone)
- **Test:** pytest with mock scored population — verify survivors, elites
  preserved, discard logic applied correctly

### Session 2.3 — Genetic operators & logging
- Implement hyperparameter crossover: for each hyperparameter, randomly
  inherit from parent A or parent B (uniform crossover)
- Implement mutation: Gaussian noise on continuous params, occasional
  discrete jumps on integer params (layer count, hidden size)
- **Genetic event logging** — every selection/crossover/mutation event written
  to `logs/genetics/YYYY-MM-DD_genN.log` and stored in `genetic_events` SQLite
  table:
  - Generation number
  - Parent model IDs → child model ID
  - For each hyperparameter: parent_a_value, parent_b_value, inherited_from,
    mutation_applied (bool), mutation_delta, final_value
  - Selection reason: "elite", "top_50pct", "bred_from: [id_a, id_b]"
  - A human-readable summary line, e.g.:
    `"Child abc123 inherited LR from parent_a (0.0003), LSTM hidden from parent_b
     (256→512 after mutation +256). Survived as elite."`
- **Test:** pytest — crossover produces valid child params, mutation stays in
  bounds, all genetic events logged with correct parent/child IDs

### Session 2.4 — Training orchestrator
- `training/run_training.py` — full generational loop:
  initialise population → for each generation: train all agents → evaluate all
  on validation days → score → select → breed → repeat
- `training/evaluator.py` — run a model on each test day independently,
  return per-day metrics dict and full bet log for registry
- Progress events emitted to a queue (consumed by API WebSocket in Phase 3)
- **Test:** pytest — orchestrator runs 2 generations on synthetic env, registry
  updated correctly, genetic log written

### Session 2.5 — First multi-generation run
- Run population training on all available real data (N generations, N from config)
- Inspect genetic logs — verify trait inheritance is recorded and legible
- Inspect scoreboard — verify population diversity, no premature convergence
- **Test:** pytest integration — N generations completes, all models in registry,
  genetic_events populated, scoreboard non-trivial

---

## Phase 3 — API & UI

### Session 3.1 — FastAPI backend core
- `api/main.py` — FastAPI app with CORS, lifespan (opens registry DB)
- `api/schemas.py` — Pydantic models for all request/response types
- `api/routers/models.py`:
  - `GET /models` — scoreboard (all active models, ranked, with per-day win_rate)
  - `GET /models/{id}` — model detail (hyperparams, architecture, metrics history)
  - `GET /models/{id}/lineage` — ancestry tree (parent/child IDs + hyperparams)
  - `GET /models/{id}/genetics` — genetic event log for this model's creation
- **Test:** pytest with TestClient — all endpoints return correct schemas,
  lineage traversal correct, error cases handled

### Session 3.2 — Training & replay API
- `api/routers/training.py`:
  - `GET /training/status` — current generation, agents trained, time elapsed
  - `WebSocket /ws/training` — live progress stream (agent name, episode reward,
    loss, P&L) consumed by Angular training monitor
- `api/routers/replay.py`:
  - `GET /replay/{model_id}/{date}` — all races for that model+day
  - `GET /replay/{model_id}/{date}/{race_id}` — full tick-by-tick state +
    agent actions for one race (order book at each tick, bet events overlaid)
- **Test:** pytest with TestClient — replay endpoint returns correct tick
  sequence, bet events at correct timestamps, WebSocket emits messages

### Session 3.3 — Angular scaffold & scoreboard
- `ng new frontend` inside repo (Angular 18+, standalone components)
- Configure proxy to FastAPI (`localhost:8000`)
- Scoreboard page:
  - Model rankings table: rank, model ID (short), generation, architecture name,
    win_rate, sharpe, mean_daily_pnl, efficiency, composite_score
  - Colour-coded by generation number
  - Trend sparkline (last 10 evaluation composite scores)
  - Click row → Model Detail
- **Test:** Angular unit tests on scoreboard component; e2e (Playwright or
  Cypress) against mock API

### Session 3.4 — Training monitor page
- Training Monitor page:
  - Current generation progress (which agent training, episode N of M)
  - Live reward / loss chart (line chart, WebSocket feed)
  - Population grid: N agents, colour = current status (training / evaluated /
    selected / discarded)
- **Test:** Angular unit tests; mock WebSocket feed

### Session 3.5 — Model detail & lineage page
- Model Detail page:
  - Hyperparameter table (all params, with diff vs parent highlighted)
  - Architecture name + description
  - Per-day P&L bar chart (colour: green=profitable, red=loss)
  - Genetic origin panel: "Bred from [parent_a] × [parent_b] — inherited N
    traits from A, M traits from B, K mutations applied"
  - Full genetic event log for this model (human-readable lines from Session 2.3)
  - Lineage tree (visual ancestor graph, at least 3 generations deep)
- **Test:** Angular unit tests on detail + lineage components

### Session 3.6 — Race replay page
- Race Replay page:
  - Select model → select date → select race (dropdown populated from API)
  - Main view: LTP price chart per runner (one line per runner, colour-coded),
    x-axis = time to off (counts down to 0), animated playback with speed control
  - Order book panel: best 3 back/lay prices+sizes for selected runner,
    updates with the cursor
  - Side panel (action log): chronological list of agent bets for this race —
    time, runner, action, price, stake, matched size, outcome (won/lost)
  - Click action → cursor jumps to that tick on the chart
  - Winner highlighted in green at race close
  - Summary bar: total bets, P&L, early picks
- **Test:** Angular unit tests; verify cursor/action sync logic

### Session 3.7 — Bet explorer page
- Bet Explorer page:
  - All bets for a selected model across all evaluation days
  - Filterable by: date, race, runner name, action (back/lay), outcome
  - Sortable columns: time to off, price, stake, P&L
  - Summary stats: total bets, bet precision, P&L per bet, total P&L
- **Test:** Angular unit tests on filter/sort logic

---

## Phase 4 — Scale & Refine

### Session 4.1 — Dataset growth retraining
- Retrain all active models on full accumulated dataset (triggered manually
  when dataset passes 30-day milestone)
- Verify scoreboard re-ranks correctly — old models evaluated against new test days

### Session 4.2 — Reward coefficient tuning
- Run ablation: vary early_pick_bonus and efficiency_penalty coefficients
  across population, compare scoreboard outcomes
- Update config.yaml defaults based on findings

### Session 4.3 — Architecture v2: Transformer policy
- Implement `"ppo_transformer_v1"` architecture (multi-head self-attention
  over runner features + positional encoding for time) in `agents/policy_network.py`
- Register in architecture registry (see PLAN.md)
- Train a sub-population using this architecture; compare on scoreboard
- Document findings in PLAN.md Architecture section

### Session 4.4 — Runner identity embeddings
- Add learned embedding layers for jockey_name, trainer_name, horse_name
  (high-cardinality categoricals) to the per-runner encoder
- Benchmark vs non-embedding baseline on scoreboard

### Session 4.5 — Performance profiling
- Profile full training run: identify CPU vs GPU bottlenecks
- Optimise data loading (prefetching, pinned memory, worker count)
- Optimise episode construction (vectorised numpy vs Python loops)

---

## Phase 5 — Live Trading

### Session 5.1 — Live environment wrapper
- `live/live_env.py` — Gymnasium env that reads from StreamRecorder1's
  MySQL HotData in real time instead of Parquet replay
- Mirrors `betfair_env.py` observation/action interface exactly

### Session 5.2 — Paper trading
- Run best-ranked model(s) in paper trading mode on live days
- Log paper bets to registry under a `paper` evaluation type
- Compare paper P&L vs backtest P&L on same days as sanity check

### Session 5.3 — Live trading integration
- Connect to Betfair betting API (OrderManager from StreamRecorder1 codebase)
- Kill switch: halt all trading if daily drawdown exceeds threshold (config)
- Real money only after sustained paper trading validation

---

## Future Architecture Experiments (backlog — no session assigned yet)

These are documented here so they are not forgotten. Assign to a session when ready.

- Soft Actor-Critic (SAC) — better for continuous action spaces than PPO
- Rainbow DQN — if we discretise the action space heavily
- IMPALA / Ape-X — distributed training across multiple CPU workers feeding one GPU
- World model (DreamerV3 style) — model learns a latent representation of the
  market, plans ahead
- Multi-agent self-play — agents trade against each other in simulation to
  develop more robust strategies
- Hierarchical RL — high-level policy selects which race to focus on;
  low-level policy handles tick-by-tick betting
