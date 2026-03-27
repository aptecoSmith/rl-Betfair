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
- MySQL on localhost:3306 (local install, hotDataRefactored + coldData)

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

### Session 2.1 — Population manager
**Status:** Done

- `agents/population_manager.py` — initialise N agents with randomised hyperparameters drawn from config.yaml search ranges
- `HyperparamSpec` dataclass + `parse_search_ranges()` parses config; `sample_hyperparams()` handles float, float_log, int, int_choice types
- `validate_hyperparams()` checks all params within defined ranges
- `PopulationManager` class: takes config + optional ModelStore, computes obs/action dims from env constants, creates policies via architecture registry
- `initialise_population(generation, seed)` — creates N agents, each with unique randomised hyperparams, registers in model store, saves initial weights
- `load_agent(model_id)` — reconstructs agent from stored hyperparams + weights
- `AgentRecord` dataclass wraps model_id, generation, hyperparams, architecture_name, policy
- 44 unit tests + 7 integration tests (real config, real data forward pass, model store round-trip)
- Session 1.5 skipped (blocked on 2+ days of data); Session 2.1 does not depend on it

### Session 2.2 — Genetic selection
**Status:** Done

- `select()` method on `PopulationManager` — tournament selection with elitism
  - Sorts by composite_score descending
  - Top `n_elite` (config, default 3) always survive as elites
  - Top `selection_top_pct` (config, default 50%) survive overall (includes elites)
  - Returns `SelectionResult` with elites, survivors, eliminated, and ranked_scores
- `apply_discard_policy()` — marks models as "discarded" in registry only if ALL of:
  - win_rate < min_win_rate (default 0.35)
  - mean_daily_pnl < min_mean_pnl (default 0.0)
  - sharpe < min_sharpe (default -0.5)
  - Never discards for bad days alone; all three must fail simultaneously
- 37 unit tests + 9 integration tests (scored population → select → discard → verify store status)

### Session 2.3 — Genetic operators & logging
**Status:** Done

- `crossover()` — uniform crossover: for each hyperparameter, randomly inherit from parent A or B. Returns child hyperparams + inheritance map.
- `mutate()` — Gaussian noise on float/float_log params (sigma = 10% of range), ±1 step on int/int_choice params. All results clamped to valid ranges. Configurable mutation_rate (default 0.3).
- `breed()` — fills population back to full size by breeding children from survivors. Picks two parents at random from survivors, applies crossover + mutation, creates policy, registers in model store with parent IDs.
- `log_generation()` — writes human-readable log to `logs/genetics/gen_N_YYYY-MM-DD.log` and records all events to `genetic_events` SQLite table.
- `genetic_events` table added to model_store.py: event_id, generation, event_type, child/parent IDs, per-hyperparameter inheritance/mutation details, selection_reason, human_summary.
- `GeneticEventRecord` dataclass + `record_genetic_event()` / `get_genetic_events()` CRUD on ModelStore.
- `BreedingRecord` dataclass captures full breeding details per child.
- `mutation_rate` added to config.yaml under `population`.
- 36 unit tests + 10 integration tests (full select → breed → log pipeline, SQLite events, log files, child lineage, weight round-trip)

### Session 2.4 — Training orchestrator
**Status:** Done

- `training/evaluator.py` — `Evaluator` class runs a trained policy on each test day independently:
  - Fresh budget per day, no LSTM carry-over between days (each day is a clean episode)
  - Deterministic actions (uses action mean, no sampling) for reproducible evaluation
  - Records per-day metrics (day_pnl, bet_count, winning_bets, bet_precision, pnl_per_bet, early_picks, profitable)
  - Records full bet log (market_id, runner_id, action, price, stake, matched_size, outcome, pnl)
  - Persists evaluation run + day records + bet records to ModelStore
  - Emits progress events to asyncio.Queue per test day
- `training/run_training.py` — `TrainingOrchestrator` full generational loop:
  - Initialise population (generation 0) with randomised hyperparams
  - Per generation: train all agents → evaluate all on test days → score via scoreboard → apply discard policy → select survivors → breed next generation → log genetic events
  - Two-level ProgressTracker at every stage:
    - Outer: agents completed / total agents in generation
    - Inner (training): delegated to PPOTrainer's own ProgressTracker per episode
    - Inner (evaluation): delegated to Evaluator's own ProgressTracker per test day
  - All progress events flow to shared asyncio.Queue for WebSocket consumption
  - Phase transitions emit `phase_start` / `phase_complete` events with summary dicts
  - Phases: training → evaluating → scoring → selecting → breeding (per gen), run_complete (final)
  - Handles insufficient data gracefully: if no test days available, uses training days for evaluation with a warning logged
  - Weights saved to ModelStore after each agent's training completes
  - `GenerationResult` and `TrainingRunResult` dataclasses capture full results
- 18 unit tests for `Evaluator` (init, empty inputs, synthetic day evaluation, persistence, progress events, metric correctness, deterministic actions)
- 25 unit tests for `TrainingOrchestrator` (init, empty inputs, single generation, two generations with selection/breeding/genetic logging, progress event phases and ordering, result structure, weights persistence)
- 6 integration tests (evaluator on real data, 2-gen orchestrator on real data: registry updated, events in correct order, genetic log populated, scoreboard re-ranked)

### Session 2.5 — First multi-generation run
**Status:** Done

- Ran 3 generations on real data (2026-03-26, population=6, epochs=2, seed=42) — 56 minutes on CPU
- `scripts/run_session_2_5.py` — standalone runner script with full summary output
- Results (1 day, train=test — scores optimistic as expected):
  - 11 total models created (6 gen0 + 3 gen0-children + 2 gen1-children)
  - 152,861 evaluation bets recorded
  - Genetic logs legible: SELECTION + BREEDING sections, per-hyperparameter trait inheritance, mutation deltas
  - Scoreboard: top 5 models clustered at score ≈ 0.689 (all win_rate=1.0, which is expected with train=test)
  - Population diversity: 3 archetypes visible — "bet on everything" (P&L £562M), "cautious" (P&L -£100), "moderate" (P&L £12-93M)
  - No premature convergence in hyperparams: mutation keeps generating diversity across learning_rate, lstm_hidden_size, mlp_hidden_size
- **Performance improvements applied during session:**
  - GPU auto-detection: `TrainingOrchestrator` auto-detects CUDA, logs GPU name + VRAM. Prominent WARNING when falling back to CPU with install instructions
  - `training.require_gpu` config flag: when `true`, raises `RuntimeError` if no CUDA detected — fail-fast for production
  - Batch bet insertion: `ModelStore.record_evaluation_bets_batch()` uses `executemany` in a single transaction — eliminates the 31K individual INSERT + COMMIT bottleneck
  - Duplicate import fix in evaluator.py
- **Observation: PyTorch CPU-only build installed** — `torch 2.11.0+cpu`. Need to reinstall with CUDA support: `pip install torch --index-url https://download.pytorch.org/whl/cu124`
- 7 unit tests (GPU detection: auto-detect, explicit override, require_gpu enforcement; batch bet insertion: correctness, empty noop)
- 12 integration tests (full 2-gen pipeline: all models registered, genetic events in SQLite, genetic log files non-empty, scoreboard non-trivial, bet logs present, progress events in correct phase order)

### Session 2.6 — CUDA PyTorch + Parquet bet logs
**Status:** Done

#### 2.6a — CUDA-enabled PyTorch
- Uninstalled CPU-only `torch 2.11.0+cpu`, installed `torch 2.11.0+cu126` (cu124 had no Python 3.14 wheels)
- RTX 3090 detected: 24 GB VRAM, `torch.cuda.is_available()` returns `True`
- Updated `requirements.txt` with CUDA install instructions (`--index-url https://download.pytorch.org/whl/cu126`)
- Set `training.require_gpu: true` in `config.yaml`
- Fixed `total_mem` → `total_memory` (renamed in PyTorch 2.11)
- Fixed device propagation bug: `TrainingOrchestrator` was passing raw `device` param (None) to `Evaluator` instead of resolved `self.device` — caused CPU/CUDA tensor mismatch during evaluation
- GPU training sanity check passed: 1 agent, 1 epoch, 53 races, 55 MB VRAM used
- `test_require_gpu_passes_with_gpu` now passes instead of skipping

#### 2.6b — Evaluation bets migrated from SQLite to Parquet
- **Problem resolved:** Session 2.5 identified SQLite as a bottleneck for 18M+ bet rows per generation
- **New write path:** `ModelStore.write_bet_logs_parquet(run_id, date, records)` writes one Parquet file per evaluation day per model to `registry/bet_logs/{run_id}/{date}.parquet`
- **New read path:** `ModelStore.get_evaluation_bets(run_id)` reads all Parquet files under `registry/bet_logs/{run_id}/`, returns same `EvaluationBetRecord` list
- **Removed:** `record_evaluation_bet()`, `record_evaluation_bets_batch()`, and the `evaluation_bets` SQLite table
- **Evaluator updated:** `Evaluator.evaluate()` now calls `write_bet_logs_parquet()` directly (one `pd.to_parquet` call per day)
- SQLite retained for metadata only: `models`, `evaluation_runs`, `evaluation_days`, `genetic_events`
- `bet_logs_dir` defaults to `registry/bet_logs/` (sibling to `models.db`)
- 7 new Parquet tests: write/read round-trip, multiple bets, multiple days, empty records, schema validation, no-data returns empty, bet_logs_dir creation
- Fixed pre-existing bug in `test_integration_population_manager.py`: swapped `load_day` arguments + wrong BetfairEnv constructor call

**Bug fixes (pre-existing):**
- `test_integration_population_manager.py::test_forward_pass_on_real_observation` — arguments to `load_day()` were swapped, and `BetfairEnv` was called with `days=` (plural) instead of `day=` (singular)

### Session 3.1 + 3.2 — FastAPI backend (core + training/replay API)
**Status:** Done

#### 3.1 — FastAPI backend core
- `api/main.py` — FastAPI app with CORS (allow `localhost:4200`), lifespan opens registry DB via ModelStore, wires Scoreboard and asyncio.Queue for progress events
- `api/schemas.py` — 16 Pydantic models: ScoreboardEntry, ScoreboardResponse, ModelDetail, DayMetric, LineageNode/Response, GeneticEvent/Response, TrainingStatus, ProgressSnapshot, WSEvent, BetEvent, RaceSummary, ReplayDayResponse, TickRunner, ReplayTick, ReplayRaceResponse
- `api/routers/models.py`:
  - `GET /models` — scoreboard: all active models ranked by composite score, with win_rate, sharpe, efficiency, generation, architecture
  - `GET /models/{id}` — model detail: hyperparams, architecture, per-day metrics history from latest evaluation run
  - `GET /models/{id}/lineage` — BFS ancestry traversal: walks parent_a/parent_b chain to roots, no duplicates
  - `GET /models/{id}/genetics` — genetic event log for this model's creation (crossover, mutation events)
- 15 unit tests (scoreboard empty/ranked/sorted, detail found/not-found/metrics/no-eval, lineage root/traversal/dedup/hyperparams, genetics empty/events)
- 11 integration tests against real extracted data (scoreboard ranking, model detail with real hyperparams, lineage chain, genetics, replay with real Parquet ticks)

#### 3.2 — Training & replay API
- `api/routers/training.py`:
  - `GET /training/status` — current run snapshot: running flag, phase, generation, two-level progress (process + item ETAs), detail string
  - `WebSocket /ws/training` — consumes shared asyncio.Queue, broadcasts progress/phase_start/phase_complete/agent_complete/run_complete events, 30s keepalive ping, mid-run clients receive latest state on connect
- `api/routers/replay.py`:
  - `GET /replay/{model_id}/{date}` — all races for model+day: race_id, venue, market_start_time, n_runners, bet_count, race_pnl
  - `GET /replay/{model_id}/{date}/{race_id}` — full tick-by-tick: snap_json parsed to per-runner order books (back/lay 3 levels), bet events overlaid at correct timestamps, winner_selection_id, race_pnl
  - Reads bet logs from Parquet (`registry/bet_logs/{run_id}/{date}.parquet`)
  - Reads tick data from extracted Parquet (`data/processed/{date}.parquet`)
- 6 WebSocket unit tests (idle ping, latest-on-connect, queue broadcast, run_complete/phase_start state transitions, latest_event tracking)
- 11 replay unit tests (model/run/tick not-found, day returns races with metadata, race tick sequence/runners parsed/bets overlaid/metadata/no-bets)
- Integration tests verify tick count matches raw Parquet, bet timestamps correct, race P&L matches registry

**Dependencies installed:** fastapi 0.135.2, uvicorn 0.42.0, httpx 0.28.1, pydantic 2.12.5, starlette 1.0.0

### Session 3.3 — Angular scaffold & scoreboard
**Status:** Done

- `ng new frontend` — Angular 21 app with standalone components, vitest, SCSS
- Proxy config (`proxy.conf.json`) routes `/api` → `http://localhost:8000` (FastAPI backend)
- **Scoreboard page** (`src/app/scoreboard/`):
  - Model rankings table: rank, model ID (short, full in tooltip), generation, architecture, win_rate, sharpe, mean_daily_pnl, efficiency, composite_score
  - Sorted by composite_score descending; null scores ranked last
  - Generation colour-coding: 10-colour palette, applied as left border on rows and badge background
  - Click row → navigates to `/models/:id` (Model Detail route — stub page, built in Session 3.5)
  - Loading state, error state, empty state handled
  - Formatted values: win_rate as %, mean_daily_pnl as £, sharpe/efficiency as decimals
- **Model Detail stub** (`src/app/model-detail/`) — placeholder component for Session 3.5
- **Routes:** `/` → redirect to `/scoreboard`, `/scoreboard` (lazy-loaded), `/models/:id` (lazy-loaded)
- **ApiService** (`src/app/services/api.service.ts`) — `GET /api/models` → `ScoreboardResponse`
- **TypeScript model** (`src/app/models/scoreboard.model.ts`) — mirrors `api/schemas.py` `ScoreboardEntry`
- 21 unit tests (vitest): component creation, loading/error/empty states, table columns, row rendering, short ID display, sorting, ranking, null score handling, generation colours with cycling, border-left colour, badge rendering, navigation on click, formatted values, data refresh
- 6 integration tests (vitest, skip when API not running): load real models, render table, columns present, sort order, valid data, click navigation
- 2 app-level tests: component creation, router-outlet presence

### Session 3.4 — Training monitor, app header & live system metrics
**Status:** Done

#### Backend — system metrics endpoint
- `api/routers/system.py` — `GET /system/metrics` returns CPU %, RAM (used/total MB, %), disk I/O and usage, GPU (utilisation %, VRAM used/total, temperature, name)
- Uses `psutil` for CPU/RAM/disk, `pynvml` for GPU (graceful fallback: `gpu: null` when unavailable)
- `GpuMetrics` and `SystemMetrics` Pydantic schemas added to `api/schemas.py`
- Router wired in `api/main.py`
- `psutil>=5.9.0` and `pynvml>=11.5.0` added to `requirements.txt`

#### Frontend — app shell header (persistent across all pages)
- `header/header.ts` — standalone component imported by `App`, visible on every page
- **Status chip:** Idle (green) / Running with ETA (amber) — click navigates to `/training`
- **Progress summary:** phase label + generation + completed/total (%) — shown when a run is active
- **System metrics panel:** GPU (util % + VRAM), CPU %, RAM (GB), SSD (GB) — polled every 3 seconds via `SystemMetricsService`
- Phase labels mapped: extracting, building, training, evaluating, selecting, breeding, scoring
- `app.html` updated: `<app-header />` above `<router-outlet />`
- `app.scss` added dark background for main content area

#### Frontend — training monitor page
- `training-monitor/training-monitor.ts` — standalone component at `/training` route
- **Two persistent ETA bars:** process bar (purple gradient) and item bar (cyan gradient), real-time via WebSocket
- **Phase banner:** animated dot + full phase description
- **Detail line:** monospace display of current episode/reward/loss
- **Reward/loss charts:** SVG line charts built from WebSocket progress events; reward chart includes zero-line
- **Population grid:** coloured cells for each agent — pending (grey), training (amber, pulsing), evaluated (purple), selected (green), discarded (red)
- **Idle state:** last run summary (JSON) or "No training run in progress" message
- Grid legend for all 5 agent statuses

#### Frontend — services
- `services/training.service.ts` — WebSocket connection to `/ws/training` with auto-reconnect (3s), HTTP polling fallback for initial state, parses reward/loss from progress detail strings for chart data
- `services/system-metrics.service.ts` — polls `GET /api/system/metrics` every 3 seconds
- `models/training.model.ts` — TypeScript interfaces: `ProgressSnapshot`, `TrainingStatus`, `WSEvent`
- `models/system.model.ts` — TypeScript interfaces: `GpuMetrics`, `SystemMetrics`

#### Routes & app wiring
- `/training` route added (lazy-loaded `TrainingMonitor`)
- `App` component imports `Header` and `RouterOutlet`
- `ApiService` unchanged (header/monitor use dedicated services)

#### Tests
- **Python:** 11 unit tests (6 endpoint tests: response schema, field types, values, GPU null; 5 GPU helper tests: available, import error, nvml error, temp error, bytes name) + 1 integration test (real hardware)
- **Angular:** 22 header tests (creation, title, idle/running states, status chip DOM, progress summary, phase labels ×7, GPU/CPU/RAM/disk labels, null metrics, metrics panel count, navigation) + 32 training monitor tests (creation, title, idle message, ETA bars, process/item data, phase labels ×7, detail line, chart empties, reward/loss paths, empty path <2 points, agent grid, agent classes ×5, no grid when empty, chart cards, chart titles, timeSinceCompleted ×5) + 19 WebSocket message flow tests (idle start, progress→running, run_complete→idle, lastRunCompletedAt from timestamp/Date.now, latestEvent tracking, ping ignored, reward/loss extraction, accumulation, non-progress ignored, clearHistory, phase_start+progress ETA bars, partial event preserves fields, negative reward, malformed JSON, full lifecycle) + 3 app tests (creation, router-outlet, header presence)
- All 95 Angular tests pass, 6 integration skipped (API not running)

**Dependencies installed:** psutil 7.2.2, pynvml 13.0.1 (nvidia-ml-py 13.595.45)

### Session 3.8 — Admin tools page
**Status:** Done

#### Backend — admin router (`api/routers/admin.py`)
- `GET /admin/days` — list extracted days with metadata (date, tick_count, race_count, file_size_bytes). Reads Parquet files from `data/processed/`
- `GET /admin/backup-days` — list dates in backup folder not yet extracted. Returns dates from `data/backup/` absent in `data/processed/`
- `GET /admin/agents` — list all models (active + discarded) with ID, generation, architecture, status, score, created_at
- `DELETE /admin/days/{date}` — remove a day's Parquet files (ticks + runners) and all `evaluation_days` records for that date
- `DELETE /admin/agents/{model_id}` — full cascade delete: weights file, all evaluation runs/days, bet log Parquets, genetic events (child only), model record. Does NOT cascade parent/child references in other models
- `POST /admin/import-day` — import single day from MySQL via `DataExtractor.extract_date()`. Returns success/failure with detail message
- `POST /admin/import-range` — import date range. Returns immediately with job_id; runs extraction in background asyncio task. Emits progress events to WebSocket via existing `progress_queue`. Skips existing dates unless `force: true`
- `POST /admin/reset` — delete all models, evaluation data, genetic events, weight files, bet log dirs. Requires `{"confirm": "DELETE_EVERYTHING"}`. Preserves extracted Parquet data
- 12 Pydantic schemas added to `api/schemas.py`: `ExtractedDay`, `ExtractedDaysResponse`, `BackupDay`, `BackupDaysResponse`, `AdminAgentEntry`, `AdminAgentsResponse`, `ImportDayRequest/Response`, `ImportRangeRequest/Response`, `ResetRequest/Response`, `AdminDeleteResponse`
- `backup_data` path added to `config.yaml` under `paths`
- Router wired into `api/main.py`

#### Frontend — admin page (`src/app/admin/`)
- **Manage Days** section: table of extracted days (date, tick count, race count, file size) with delete button per row. Confirmation dialog before deletion
- **Import Days** section: list of backup dates with single-day import button, "Import All" button with count, date-range picker for bulk import. Progress bar for multi-day imports
- **Manage Agents** section: table of all models (active + discarded) with model ID (short, full in tooltip), generation, architecture, status badge (colour-coded), composite score, created date, delete button. Confirmation dialog. Discarded rows shown at reduced opacity
- **Reset** section: "Start Afresh" button with two-step confirmation — dialog lists exactly what will be deleted, user must type `DELETE_EVERYTHING` to enable the confirm button
- Success/error message banners with auto-dismiss after 5 seconds
- Dark theme consistent with existing UI (scoreboard, training monitor)
- `AdminService` methods added to `ApiService`: `getExtractedDays`, `getBackupDays`, `getAdminAgents`, `deleteDay`, `deleteAgent`, `importDay`, `importRange`, `resetSystem`
- TypeScript models in `models/admin.model.ts`
- Route: `/admin` (lazy-loaded)
- Header updated with nav links: Scoreboard, Training, Admin (using `RouterLink` + `RouterLinkActive`)

#### Tests
- **Python:** 31 unit tests: list days (empty, metadata, sorted), list backup days (empty, new-only, dir-not-exists), list agents (empty, all-statuses), delete day (existing, nonexistent, invalid format, cascade eval days), delete agent (full artefacts, nonexistent, no parent cascade), import day (success, no data, invalid date, extractor error), import range (queue dates, skip existing, force reimport, all existing, invalid dates, start>end), reset (wrong confirmation, clears everything, empty registry)
- **Python integration:** 4 tests: create→delete day with eval day preservation, create→delete agent with full cleanup, create→reset with Parquet preservation, list days after delete
- **Angular:** 42 unit tests (vitest): component creation, page title, loading states, empty states, days table rendering/columns/data, delete day dialog show/cancel/confirm, backup days rendering/import buttons/Import All, single day import, import range, agents table rendering/columns/short ID/status badges/discarded class, delete agent dialog, reset dialog show/cancel/confirm text validation/API call, formatBytes helper, shortId helper, success/error messages, section headers
- All 802 Python tests pass, all 137 Angular tests pass

### Session 3.5 — Model detail & lineage page
**Status:** Done

#### Frontend — Model Detail page (`src/app/model-detail/`)
- **Header:** model short ID (full in tooltip), generation badge (colour-coded), status badge (green active / red discarded), architecture name + description, composite score display
- **Genetic Origin panel:** "Bred from [parent_a] × [parent_b] — inherited N traits from A, M traits from B, K mutations applied". Seed models show "Seed model (no parents — generation 0)". Parent IDs are clickable links to navigate to their detail pages
- **Hyperparameter table:** all params sorted alphabetically, with diff highlighting (orange dot + row highlight) for values that differ from parent_a. Uses `parentHyperparams` computed from lineage data
- **Per-day P&L bar chart:** SVG bar chart with green (profitable) / red (loss) bars, sorted by date. Zero line, legend with max absolute value. Responsive via viewBox
- **Genetic event log:** scrollable list of all genetic events for this model's creation. Displays `human_summary` when available, falls back to structured display (event type badge, hyperparameter, inherited_from, mutation delta)
- **Lineage tree:** SVG ancestor graph, at least 3 generations deep. Nodes grouped by generation (newest at top), centred per row. Edges connect children to parents. Current model highlighted with thicker border. Nodes are clickable to navigate to that model. Node colour matches generation palette. Shows short ID + composite score (or "Gen N" if no score)
- **Metrics summary:** grid cards for test days, profitable days, total P&L (colour-coded), total bets
- **Back navigation:** "← Scoreboard" button in header

#### TypeScript models & API service
- `models/model-detail.model.ts` — `ModelDetailResponse`, `DayMetric`, `LineageNode`, `LineageResponse`, `GeneticEvent`, `GeneticsResponse`
- `ApiService` methods: `getModelDetail(id)`, `getModelLineage(id)`, `getModelGenetics(id)`

#### Tests
- **Angular:** 51 unit tests (vitest): component creation, route param reading, short ID, API calls on init, loading/error states, generation badge/colour/cycling, architecture display, composite score, status badge/discarded, hyperparams table rendering/sorting/values/diff highlighting/diff markers/no-parent-no-markers, genetic origin bred/seed/card/parent-links, P&L chart section/bar data/SVG bars/zero line/no-data/date sorting, genetic events rendering/human summary/empty section, lineage tree section/no-data/tree nodes/SVG nodes/current highlight/edges/3-gen-deep, metrics summary/no-history/totalPnl/totalBets, format helpers (scientific/normal/string), navigation (back/model click), back button, chart legend
- **Angular integration:** 6 tests (vitest, skip when API not running): load real model, hyperparams match registry, P&L chart renders, lineage tree loads, correct parents, architecture + generation display
- All 804 Python tests pass, all 188 Angular tests pass (12 skipped — integration)

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
| 2.1     | 44              | 7                      | **568 + 64**  |
| 2.2     | 37              | 9                      | **605 + 73**  |
| 2.3     | 36              | 10                     | **641 + 83**  |
| 2.4     | 43              | 6                      | **684 + 89**  |
| 2.5     | 7               | 12                     | **691 + 101** |
| 2.6     | 16              | 1                      | **707 + 102** |
| 3.1+3.2 | 36              | 11                     | **743 + 113** |
| 3.3     | 23 (Angular)    | 6 (Angular)            | **743 + 113** (Python) + **23 + 6** (Angular) |
| 3.4     | 11 (Python) + 76 (Angular) | 1 (Python) | **754 + 114** (Python) + **95 + 6** (Angular) |
| 3.8     | 27 (Python) + 42 (Angular) | 4 (Python) | **781 + 118** (Python) + **137 + 6** (Angular) |
| 3.5     | 51 (Angular)    | 6 (Angular)            | **781 + 118** (Python) + **188 + 12** (Angular) |

**Python total: 804 passed, 2 skipped, 102 deselected.**
**Angular total: 188 passed, 12 skipped (integration — API not running).**

---

## Key Decisions & Diversions

1. **SnapJson over updates join** (Session 0.2) — The `updates` and `ResolvedMarketSnaps` tables record timestamps independently. An exact join produces zero rows. Market-level fields are now extracted from SnapJson in Python instead.

2. **Both WIN and EACH_WAY markets** (Session 1.1) — The model plays both market types. No filtering by market type.

3. **Proportional staking** (Session 0.5) — Stake = fraction of current budget, not fixed £. This means winning days compound and losing days naturally shrink stakes.

4. **In-play observation, pre-race betting** (Session 0.5) — The agent sees in-play ticks (valuable signal about how races resolve) but can only place bets before the off. This is both realistic (Betfair delays in-play) and informative (the model learns what happens after the off to improve pre-race decisions in later races).

5. **Race dataclass enrichment** (Session 1.3+1.4) — Added `market_name`, `market_type`, `n_runners` to Race. These were missing from the original design but are useful for evaluation reporting and filtering.

6. **Learning rate cap at 5e-4** (Session 0.1) — Higher LRs destabilise PPO+LSTM training. The original PLAN.md range of 1e-3 was tightened during config review.

7. **Graceful insufficient data handling** (Session 2.4) — When no test days are available (e.g. only 1 day extracted), the orchestrator uses training days for evaluation with a logged warning. Results are optimistic and should not be trusted for ranking. This unblocks the full pipeline even with a single day of data.

8. **Evaluation bets → Parquet** (Session 2.6b) — Session 2.5 identified SQLite as a bottleneck for 18M+ rows/generation. Migrated evaluation_bets from SQLite to Parquet files (`registry/bet_logs/{run_id}/{date}.parquet`). SQLite retained for metadata tables only. Single `pd.to_parquet` call per day replaces thousands of INSERTs.

9. **CUDA cu126 not cu124** (Session 2.6a) — Python 3.14 had no cu124 wheels. Used cu126 instead. RTX 3090 confirmed working with 55 MB VRAM for single-agent training.

10. **Device propagation fix** (Session 2.6a) — `TrainingOrchestrator` was passing the raw `device` constructor parameter (None) to the `Evaluator` instead of the resolved `self.device` ("cuda"). This caused CPU/CUDA tensor mismatch errors that only appeared when actually running on GPU.
