# TODO — rl-betfair

## Session Rules (apply to every item below)

Each TODO item is its own Claude Code session. Before closing a session:
1. **Test** — every item must have automated tests (pytest) that pass
2. **Commit** — `git commit` with a descriptive message
3. **Push** — `git push` to `aptecoSmith/rl-Betfair` on GitHub

No item is done until all three steps are complete.

---

## Phase 0 — Foundation (no live data yet)

### Session 0.1 — Repo scaffold, config & ProgressTracker
- Create Python virtual environment (`python -m venv .venv`)
- Write `requirements.txt` (torch, gymnasium, stable-baselines3, pymysql,
  sqlalchemy, pandas, pyarrow, fastapi, uvicorn, pydantic, pytest, python-dotenv)
- Write `config.yaml` (DB connection, population size, reward coefficients,
  paths, architecture name)
- **⚠️ Have a conversation with the user before finalising config.yaml** —
  specifically around hyperparameter search ranges (learning rate, LSTM size,
  entropy coefficient, etc.). The ranges in PLAN.md are a starting point but
  the user should sign off before they become the defaults that seed the first
  population.
- Write `.gitignore` (venv, processed data, model weights, `*.env`, `__pycache__`)
- Write `.env.example` (DB password placeholder)
- Write `conftest.py` (shared pytest fixtures)
- `training/progress_tracker.py` — `ProgressTracker` class (see PLAN.md ETA
  section): rolling-window ETA for item and process, `to_dict()` for WebSocket,
  `_fmt()` helper for human-readable durations ("4m 12s", "1h 18m", etc.)
- **Test:** pytest confirms config loads correctly, all required keys present;
  ProgressTracker tests: correct rolling average, correct ETA on tick(), correct
  formatting, handles zero-completed edge case
- **Integration test:** config loads from real config.yaml, ProgressTracker
  ticks and produces sane ETAs

### Session 0.2 — Data extractor
- `data/extractor.py` — connect to MySQL (localhost:3306), query
  `ResolvedMarketSnaps` with ColdData tables (results, weather, runner
  metadata), export one Parquet file per day to `data/processed/`.
  Market-level fields (venue, marketTime, inPlay, etc.) extracted from
  SnapJson in Python — no timestamp join with `updates` table.
  All ticks included (pre-race and in-play); bet placement restriction
  is enforced by the environment, not the extractor.
- Uses `ProgressTracker` — emits per-day and total-days ETA to stdout/log
  (WebSocket integration comes in Session 3.2)
- **Test:** pytest with a mock MySQL connection — verify SQL queries, correct
  join logic, output schema matches expected Parquet columns
- **Integration test:** extract a real day from MySQL → verify Parquet files
  produced, correct columns, rows include pre-race and in-play ticks, runners
  have selection_id

### Session 0.3 — Episode builder
- `data/episode_builder.py` — load a day's Parquet, construct
  `Day → [Race → [Tick]]` hierarchy as typed dataclasses.
  `parse_snap_json` handles the real nested SnapJson layout
  (`MarketRunners` → `RunnerId`/`Definition`/`Prices`) and the flat test layout.
  Both pre-race and in-play ticks are included.
- `data/feature_engineer.py` — derive price velocity, implied probability,
  overround, cross-runner relative features (rank by price, price gaps) - have a conversation with the user here about features.
- **Test:** pytest with synthetic Parquet data — verify Day/Race/Tick structure,
  feature values, edge cases (single runner, missing prices, removed runners)
- **Integration test:** load real extracted day → verify races have ticks with
  parsed runners, order books have valid PriceSize values, runner metadata
  present, winners identified, ticks ordered by sequence. Feature engineering
  produces results for all races/ticks with expected keys.

### Session 0.4 — Order book & bet manager
- `env/order_book.py` — realistic bet matching: back bets consume
  AvailableToLay volume level by level; lay bets consume AvailableToBack;
  partial fills supported; unmatched remainder cancelled
- `env/bet_manager.py` — track open bets, matched size, realised P&L,
  remaining budget, lay liability
- **Test:** pytest with known order book snapshots — verify full fills, partial
  fills, no-fill cases, liability calculation, budget enforcement
- **Integration test:** match back/lay bets against real order books from
  extracted data. Place bets and settle a real race — verify P&L is sane.
  Simulate betting £1 across every race in a full day — verify all bets
  settled, no open liability, budget non-negative.

### Session 0.5 — Gymnasium environment
- `env/betfair_env.py` — Gymnasium `Env` subclass:
  - Observation space: full state vector (see PLAN.md State Representation)
  - Action space: per-runner (action_type, stake), masked for budget
  - Step: advance one tick, apply actions, compute reward
  - Episode: one full day across all races; budget carries across races - double check this with user.  Some doubt over whether we want a budget per race to ensure the model can attempt to engage in that race, versus the model taking into account that its money needs to last the day.  A sensible solution is needed.  Perhaps its worth keeping two scoreboards - models scored per race and per day?
  - Agent observes all ticks (pre-race + in-play) but can only place bets on
    pre-race ticks (in_play == False)
  - Reward: per-race P&L + early pick bonus + efficiency penalty (all
    coefficients from config.yaml)
- **Test:** pytest with synthetic episodes — verify observation shape, action
  masking, reward calculation for known scenarios, episode termination,
  budget carry-across-races
- **Integration test:** run a full episode on real extracted data — verify
  observations produced for every tick, bets only placed pre-race, races
  settled correctly, budget tracks across races, final P&L matches sum of
  race P&Ls

---

## Phase 1 — First Agent (requires 2+ days of data)

### Session 1.1 — Data extraction from real DB
- Run `data/extractor.py` against live MySQL (localhost:3306)
- Verify Parquet output for first real day(s) — check row counts, spot-check
  prices against known race data
- **Test:** pytest validates Parquet schema and non-null key fields on real output
- **Integration test:** extract all available dates, verify each produces valid
  Parquet, spot-check order book depth and runner counts against DB

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
- **Integration test:** build observation vectors from real extracted data,
  feed through policy network forward pass, verify output shapes match
  real runner counts

### Session 1.3 — PPO trainer (single agent)
- `agents/ppo_trainer.py` — PPO training loop for one agent:
  rollout collection, advantage estimation (GAE), policy + value loss,
  gradient clipping, entropy bonus
- Uses `ProgressTracker` for episode-level and total-episodes ETA
- Logs per-episode: reward, P&L, bet count, loss terms → `logs/` directory
- Progress dict published to a `asyncio.Queue` for the WebSocket to consume
- **Test:** pytest — trainer runs for N steps on synthetic env without error,
  loss decreases over a trivial known environment, progress events emitted
- **Integration test:** train one agent for a small number of episodes on real
  data, verify loss computed, bets placed and settled, P&L recorded

### Session 1.4 — Model registry
- `registry/model_store.py` — SQLite schema and CRUD:
  `models`, `evaluation_runs`, `evaluation_days`, `evaluation_bets` tables
  (full schema in PLAN.md)
- `registry/scoreboard.py` — compute composite score from per-day metrics
  (win_rate, sharpe, mean_daily_pnl, efficiency), rank all active models
- **Test:** pytest — create models, save/load weights, record evaluation runs,
  verify composite score formula, verify ranking order
- **Integration test:** train agent on real data → save to registry → load
  back → verify weights match, metadata correct, scoreboard ranks it

### Session 1.5 — End-to-end single agent run
- Train one agent on first real training days
- Evaluate on first real test days — write results to registry
- Inspect scoreboard output, bet log, per-day P&L
- Verify nothing obviously broken (reward signal sensible, bets being matched,
  P&L not exploding)
- **Integration test:** full train → evaluate → registry pipeline on real data
  — verify per-day metrics recorded, bet log populated, composite score
  computed, scoreboard non-empty

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
- **Integration test:** initialise population from real config → verify all
  agents have valid hyperparams, can each produce a forward pass on real data

### Session 2.2 — Genetic selection
- Implement tournament selection: top 50% by composite score survive
- Implement elitism: top N_elite agents (config) always survive unchanged
- Discard policy: mark models as `discarded` in registry only if win_rate,
  mean_pnl, AND sharpe all fall below thresholds (never discard for bad days alone)
- **Test:** pytest with mock scored population — verify survivors, elites
  preserved, discard logic applied correctly
- **Integration test:** train a small population on real data → score →
  select → verify correct number survive, elites preserved, discards applied

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
- **Integration test:** breed children from real trained parents → verify
  genetic log file written, genetic_events table populated, child hyperparams
  within valid ranges

### Session 2.4 — Training orchestrator
- `training/run_training.py` — full generational loop:
  initialise population → for each generation: train all agents → evaluate all
  on validation days → score → select → breed → repeat
- `training/evaluator.py` — run a model on each test day independently,
  return per-day metrics dict and full bet log for registry
- **Two-level ProgressTracker at every stage:**
  - Outer tracker: agents completed / total agents in generation
  - Inner tracker (training): episodes completed / total episodes
  - Inner tracker (evaluation): test days completed / total test days
  - Separate trackers for genetics phase (children bred / total to breed)
- All progress events written to a shared `asyncio.Queue` consumed by the API
  WebSocket — single canonical progress stream for the whole run
- Phase transitions emit `phase_start` / `phase_complete` events with summary
- **Test:** pytest — orchestrator runs 2 generations on synthetic env, registry
  updated correctly, genetic log written, all progress events emitted in order
- **Integration test:** run 2 generations on real data (small population) →
  verify registry updated, progress events emitted in correct phase order,
  genetic log populated, scoreboard re-ranked

### Session 2.5 — First multi-generation run
- Run population training on all available real data (N generations, N from config)
- Inspect genetic logs — verify trait inheritance is recorded and legible
- Inspect scoreboard — verify population diversity, no premature convergence
- **Integration test:** N generations on real data — all models in registry,
  genetic_events populated, scoreboard non-trivial, bet logs present for
  every evaluation day

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
- **Integration test:** start API against real registry DB → verify scoreboard
  endpoint returns trained models, model detail includes real hyperparams and
  per-day metrics, lineage traversal returns correct parent chain

### Session 3.2 — Training & replay API
- `api/routers/training.py`:
  - `GET /training/status` — current run snapshot: phase, process ETA, item ETA,
    last completed agent score, generation number
  - `WebSocket /ws/training` — consumes the shared `asyncio.Queue` from the
    orchestrator; broadcasts `progress`, `phase_start`, `phase_complete`,
    `agent_complete`, `run_complete` events (full schema in PLAN.md ETA section)
  - Clients that connect mid-run receive the latest state immediately on connect
    (no waiting for the next event)
- `api/routers/replay.py`:
  - `GET /replay/{model_id}/{date}` — all races for that model+day
  - `GET /replay/{model_id}/{date}/{race_id}` — full tick-by-tick state +
    agent actions for one race (order book at each tick, bet events overlaid)
- **Test:** pytest with TestClient — replay endpoint returns correct tick
  sequence, bet events at correct timestamps, WebSocket emits messages
- **Integration test:** replay endpoint against real evaluation data — verify
  tick sequence matches Parquet, bet events at correct timestamps, race P&L
  matches registry records

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
- **Integration test:** e2e against real API — scoreboard loads with trained
  models, click-through to model detail works

### Session 3.4 — Training monitor page
- Training Monitor page:
  - **Two persistent ETA bars always at the top** (see PLAN.md ETA section):
    - Process bar: "Generation 4 · 7 of 20 agents trained · ETA 1h 18m"
    - Item bar: "Training model_x1y2z3 · episode 312/1000 · ETA 4m 12s"
  - Both bars update in real time via WebSocket
  - Phase label updates as the run moves through extraction → training →
    evaluation → genetics → scoring
  - Live reward / loss chart (line chart, WebSocket feed)
  - Population grid: N agents, colour = current status (training / evaluated /
    selected / discarded)
  - If no run in progress: shows last run summary + time since completed
  - Scoreboard page shows status chip: Idle / Running (ETA Xh Ym) / Error
- **Test:** Angular unit tests; mock WebSocket feed; verify ETA bars update
  correctly on each event type
- **Integration test:** trigger a real training run → verify WebSocket events
  arrive in correct phase order, ETA bars update, population grid reflects
  agent states

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
- **Integration test:** view a real model's detail page → verify hyperparams
  match registry, P&L chart renders, lineage tree shows correct parents

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
- **Integration test:** replay a real evaluated race → verify LTP chart data
  matches Parquet ticks, bet events overlay at correct timestamps, winner
  highlighted correctly

### Session 3.7 — Bet explorer page
- Bet Explorer page:
  - All bets for a selected model across all evaluation days
  - Filterable by: date, race, runner name, action (back/lay), outcome
  - Sortable columns: time to off, price, stake, P&L
  - Summary stats: total bets, bet precision, P&L per bet, total P&L
- **Test:** Angular unit tests on filter/sort logic
- **Integration test:** load real evaluation bets → verify filter/sort works,
  summary stats match registry aggregates, P&L per bet is sane

---

## Phase 4 — Scale & Refine

### Session 4.1 — Dataset growth retraining
- Retrain all active models on full accumulated dataset
- Trigger threshold is configurable in `config.yaml` (`retraining.min_days`,
  default 30) — change it without touching code
- UI should expose this setting on a config/settings page so it can be adjusted
  without editing files
- Trigger options: manual (button in UI), automatic when threshold crossed,
  or scheduled (e.g. weekly) — configurable
- Verify scoreboard re-ranks correctly — old models evaluated against new test days
- **Integration test:** add new data days → trigger retraining → verify old
  models re-evaluated on new test days, scoreboard re-ranked, no regressions
  in existing metrics

### Session 4.2 — Reward coefficient tuning
- Run ablation: vary early_pick_bonus and efficiency_penalty coefficients
  across population, compare scoreboard outcomes
- Update config.yaml defaults based on findings
- **Integration test:** train populations with different reward configs → verify
  scoreboard reflects coefficient differences, bet counts change with
  efficiency_penalty

### Session 4.3 — Architecture v2: Transformer policy
- Implement `"ppo_transformer_v1"` architecture (multi-head self-attention
  over runner features + positional encoding for time) in `agents/policy_network.py`
- Register in architecture registry (see PLAN.md)
- Train a sub-population using this architecture; compare on scoreboard
- Document findings in PLAN.md Architecture section
- **Integration test:** train transformer agent on real data → verify forward
  pass, training loop, evaluation all work end-to-end; scoreboard includes
  both LSTM and transformer models side by side

### Session 4.4 — Runner identity embeddings
- Add learned embedding layers for jockey_name, trainer_name, horse_name
  (high-cardinality categoricals) to the per-runner encoder
- Benchmark vs non-embedding baseline on scoreboard
- **Integration test:** train with embeddings on real data → verify embedding
  layers receive gradients, no OOM on real vocabulary sizes, scoreboard
  comparison valid

### Session 4.5 — Raw observation input mode
- Add an observation mode that feeds the raw order book data (3-level
  back/lay prices+sizes per runner, LTP, volume, timestamps) directly into
  the network alongside or instead of the hand-crafted features from
  `feature_engineer.py`
- The goal is to let the agent discover features and correlations that the
  human engineer didn't think of — this is the whole point of deep RL
- Implement as a config toggle (`observation_mode: "engineered" | "raw" | "both"`)
  so the population can evolve which mode works best
- `"raw"` mode: per-runner input is the flat price/size grid + LTP + volume +
  timestamps + raw metadata numerics; no derived features
- `"both"` mode: concatenation of raw and engineered — lets the agent use
  engineered features as a shortcut while still having access to raw data
  for discovering novel patterns
- `"engineered"` mode: current behaviour (derived features only)
- Register as a hyperparameter in the genetic system so the population can
  explore which mode (or combination) produces the best agents
- **Test:** pytest — verify observation shapes for all three modes; verify
  raw mode produces identical results in backtest vs live (same data in,
  same features out)
- **Integration test:** train agents in all three modes on real data → verify
  observation dimensions correct, training completes, population can mix modes

### Session 4.6 — Performance profiling & optimisation

Profiling data from Session 2.6 (1 agent, 1 day = 53 races, 4,182 ticks):

| Operation | Time | % of train+eval | Notes |
|---|---|---|---|
| Data loading (Parquet + JSON) | 3.35s | — | One-time per run |
| Feature engineering (×2) | 2.04s | 12% | Same day re-computed for train AND eval |
| Rollout collection (env.step) | 8.83s | 55% | CPU-bound Python loop |
| PPO update (gradient descent) | 1.78s | 11% | On GPU, 264 mini-batches |
| Evaluation (inference loop) | 5.77s | 36% | CPU-bound Python loop |
| **GPU utilisation** | **55 MB / 24,576 MB** | **0.2%** | RTX 3090 massively underused |

At full scale (20 agents × 30 days × 5 gens): feature engineering alone = 20 min
of redundant re-computation; rollout+eval = ~2.4 h/gen, all sequential.

**Items (ordered by impact):**

1. ✅ **Cache `engineer_day()` results** — done early as a quick win (see below)
2. **Vectorise env.step() loop** — rollout is pure Python iterating tick-by-tick.
   Observation arrays are already pre-computed numpy. Batch forward passes across
   ticks where the agent does nothing, or move step logic to vectorised numpy ops.
3. **Parallel agent training** — agents trained sequentially; 55 MB GPU per agent
   on 24 GB card = room for 4–8 agents in parallel. Use `torch.multiprocessing`
   or overlap CPU rollout with GPU PPO updates via async workers.
4. **Batch rollout across days** — instead of one env per day sequentially, batch
   multiple days into a vectorised env (like SB3 `SubprocVecEnv`) so the GPU gets
   larger batches per forward pass.
5. **Optimise `_build_day()` JSON parsing** — 3.35s to load 1 day (100s at 30 days).
   `parse_snap_json()` per row is likely the bottleneck. Consider `orjson` or
   pre-parsing to a more efficient format.
6. **Pinned memory for GPU transfers** — use `pin_memory=True` on DataLoader or
   manual `tensor.pin_memory()` for obs/action batches to speed CPU→GPU copies.
- **Integration test:** benchmark before/after — verify training throughput
  improved, no correctness regressions (same model on same data produces
  same P&L)

---

## Phase 5 — Live Trading

### Session 5.1 — Live environment wrapper
- `live/live_env.py` — Gymnasium env that reads from StreamRecorder1's
  MySQL HotData in real time instead of Parquet replay
- Mirrors `betfair_env.py` observation/action interface exactly
- **Integration test:** connect to live MySQL → verify observations produced
  in real time, same shape as backtest env, agent can produce actions

### Session 5.2 — Paper trading
- Run best-ranked model(s) in paper trading mode on live days
- Log paper bets to registry under a `paper` evaluation type
- Compare paper P&L vs backtest P&L on same days as sanity check
- **Integration test:** paper trade a full day → verify bets logged in registry
  as `paper` type, P&L recorded, compare against backtest of same day to
  check for divergence

### Session 5.3 — Live trading integration
- Connect to Betfair betting API (OrderManager from StreamRecorder1 codebase)
- Kill switch: halt all trading if daily drawdown exceeds threshold (config)
- Real money only after sustained paper trading validation
- **Integration test:** place a minimal real bet (£0.01) on a test market →
  verify order placed via API, confirmation received, kill switch triggers
  correctly on simulated drawdown

---

## ⚠️ CRITICAL — Infrastructure fixes (Session 2.6)

These block meaningful training runs and must be done before further model work.

### Session 2.6a — Install CUDA-enabled PyTorch
- Current: `torch 2.11.0+cpu` — RTX 3090 (24 GB VRAM) is sitting idle
- Uninstall CPU-only torch: `pip uninstall torch torchvision torchaudio`
- Install CUDA build: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
- Update `requirements.txt` with the CUDA index URL
- Set `training.require_gpu: true` in `config.yaml`
- Verify: `torch.cuda.is_available()` returns `True`, `torch.cuda.get_device_name(0)` returns the 3090
- Run a quick training sanity check (1 agent, 1 epoch) and confirm GPU utilisation via `nvidia-smi`
- **Test:** existing `test_require_gpu_passes_with_gpu` should now pass instead of skip

### Session 2.6b — Migrate evaluation_bets from SQLite to Parquet
- **Problem:** 31K individual bet records per agent per day written to SQLite.
  At full scale (20 agents × 30 test days × 30K bets = 18M rows/generation)
  this is an unacceptable bottleneck and the DB file will balloon.
- **Solution:** Keep SQLite for metadata (models, evaluation_runs,
  evaluation_days, genetic_events — small tables, low write volume). Move
  evaluation_bets to Parquet files:
  - Write path: one Parquet file per evaluation day per model:
    `registry/bet_logs/{run_id}/{date}.parquet`
  - Schema: same columns as the current `evaluation_bets` SQLite table
  - Read path: `ModelStore.get_evaluation_bets(run_id)` reads from Parquet
    instead of SQLite — returns the same `EvaluationBetRecord` list
  - `Evaluator` writes a DataFrame directly to Parquet (one `pd.to_parquet`
    call, no row-by-row inserts)
  - Remove `evaluation_bets` table from SQLite schema (keep for migration
    period if needed, but new writes go to Parquet only)
  - Remove `record_evaluation_bet()` and `record_evaluation_bets_batch()`
    from ModelStore — replaced by Parquet writes in the Evaluator
  - Replay API (Session 3.2) will read from Parquet — fast columnar reads
- **Test:** verify Parquet bet logs written, readable, schema matches,
  `get_evaluation_bets` returns correct data from Parquet files
- **Integration test:** run evaluation on real data → verify Parquet files
  created with correct row counts, old SQLite path still works for any
  pre-existing data

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
- **Hierarchical RL** ⭐ — high-level policy selects which race to focus on
  (or how much budget to allocate); low-level policy handles tick-by-tick
  betting within the chosen race. A natural fit for this problem — a race day
  genuinely is two-level. Kept in Phase 4 rather than v1 because: (a) credit
  assignment across two levels is hard when neither policy is trained yet, and
  (b) our LSTM day-episode already lets the agent implicitly skip races by
  doing nothing. Validate the low-level policy first, then add the explicit
  hierarchy. When the time comes, implement as `hierarchical_ppo_v1` in the
  architecture registry — no other code changes needed.
