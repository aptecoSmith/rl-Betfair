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

### Session 2.7a — PolledMarketSnapshots + RaceStatusEvents
**Status:** Done

#### Extractor — dual source with auto-detect
- `data/extractor.py` — auto-detects tick source per date: if `PolledMarketSnapshots` has data, use it (higher frequency ~5s); otherwise fall back to `ResolvedMarketSnaps` (legacy ~180s conflation)
- New SQL queries: `SQL_POLLED_HAS_DATE`, `SQL_POLLED_TICKS`, `SQL_POLLED_AVAILABLE_DATES`, `SQL_RACE_STATUS_EVENTS`
- `_polled_runners_to_snap_json()` — converts the polled `RunnersJson` format (`selectionId`, `state.*`, `exchange.*`) into the nested `SnapJson` format (`MarketRunners → RunnerId/Definition/Prices`) so `parse_snap_json` and all downstream code works unchanged
- `_enrich_polled_ticks()` — pulls venue, market_start_time, market_type from the `updates` table (polled snapshots don't carry these)
- `_join_race_status()` — as-of merge of `RaceStatusEvents` by market_id + timestamp: for each tick, finds the most recent race status event at or before that tick's timestamp
- `get_available_dates()` now merges dates from both sources
- `has_polled_data(date)` public method for checking availability
- `race_status` column added to `TICKS_COLUMNS`

#### Episode builder
- `race_status: str | None` field added to `Tick` dataclass (after `winner_selection_id`)
- `_row_to_tick()` reads `race_status` from Parquet, handles missing column (backward-compatible with old files)

#### Feature engineer
- `RACE_STATUSES` constant: 6 known statuses (`parading`, `going down`, `going behind`, `under orders`, `at the post`, `off`)
- `market_tick_features()` now produces 6 one-hot features (`race_status_parading`, `race_status_going_down`, etc.) — all zeros when `race_status` is None (old data)
- `TickHistory` tracks race status changes: `_last_race_status`, `_last_status_change_tick`, `_tick_counter`
- `market_velocity_features()` now includes `time_since_status_change` — normalised (ticks_since_change * 5s / 1800s), clamped to [0, 1]
- `reset()` clears race status tracking state

#### Environment
- `MARKET_KEYS` extended with 6 race status one-hot keys → `MARKET_DIM` = 31 (was 25)
- `MARKET_VELOCITY_KEYS` extended with `time_since_status_change` → `VELOCITY_DIM` = 7 (was 6)
- Total `obs_dim` = 31 + 7 + (93 × 14) + 5 = **1345** (was 1338, delta +7)
- `MARKET_TOTAL_DIM` in `policy_network.py` updated to 43 (was 36)
- Fully backward-compatible: old Parquet files with no `race_status` column produce all-zero race status features

#### Notes
- Existing models (trained on obs_dim=1338) are incompatible with the new obs_dim=1345. Retraining required after this change

#### Validated with real polled data (2026-03-28)
- BetfairPoller captured 390 polled snapshots across 69 markets + 36 race status events
- Full pipeline validated end-to-end: MySQL → extract → Parquet → episode builder → feature engineer → BetfairEnv (obs_dim=1345, 50 steps, no NaNs)
- Three bugs found and fixed during real data validation:
  1. `merge_asof` in `_join_race_status` — pandas 3.0 requires globally sorted `on` key even with `by` grouping; fixed by sorting on `timestamp` alone instead of `["market_id", "timestamp"]`
  2. `_enrich_polled_ticks` — `updates` table only has legacy data; added coldData fallback (Todays_Markets → marketdescription → Event) to populate venue, market_start_time, market_type for polled markets
  3. `_polled_runners_to_snap_json` — price ladder keys were passed through as lowercase (`price`/`size`) from polled source; normalised to uppercase (`Price`/`Size`) to match legacy SnapJson format
- `test_real_extraction.py` — fixed `test_ticks_and_runners_share_markets` to compare same-date pairs only (was cross-joining all dates via independent parametrization)

#### Tests
- **47 unit tests** (test_session_2_7a.py): polled→snap conversion (16 tests incl. null/empty/invalid/missing state, parse_snap_json round-trip), race status join (4 tests: basic, no events, empty ticks, multiple markets), extractor auto-detect (4 tests: no table, no rows, rows exist, legacy adds race_status), episode builder (4 tests: race_status field, None default, all values, backward compat with old Parquet), feature engineer race status (8 tests: one-hot all zeros/parading/under orders/off/case insensitive/all 6 statuses, RACE_STATUSES constant), time_since_status_change (5 tests: initial zero, increases, resets on change, clamps at 1.0, reset clears), env dimensions (6 tests: market_dim, velocity_dim, keys contain statuses, obs_dim, backward compat env episode)
- **7 integration tests** (test_integration_session_2_7a.py): auto-detect on real DB, available_dates includes legacy, extract_date legacy fallback, load_day backward compat, feature engineer handles None, time_since_status_change present, env runs full episode with old data
- All 883 Python tests pass (was 858), 3 skipped, 102 deselected

### Session 2.7b — RaceCardRunners (PastRacesJson)
**Status:** Done

#### Extractor — RaceCardRunners merge
- `data/extractor.py` — new `SQL_RACECARD_RUNNERS` query fetches `PastRacesJson`, `TimeformComment`, `RecentForm`, and overlapping metadata fields from `hotDataRefactored.RaceCardRunners`
- `_merge_racecard_runners()` — left-joins RaceCardRunners onto coldData runners by (market_id, selection_id). Adds 3 new columns to runners Parquet: `timeform_comment`, `recent_form`, `past_races_json`
- Field override: where RaceCardRunners has fresher non-null values for AGE, WEIGHT_VALUE, JOCKEY_NAME, TRAINER_NAME, DAYS_SINCE_LAST_RUN, SEX_TYPE, these override stale coldData values
- `RUNNERS_COLUMNS` extended from 37 → 40 columns
- Graceful fallback: if `RaceCardRunners` table doesn't exist, new columns default to None

#### Episode builder — PastRace dataclass
- New `PastRace` frozen dataclass: date, course, distance_yards, going, going_abbr, bsp, ip_max, ip_min, race_type, jockey, official_rating, position, field_size
- `_parse_position()` — parses `"3/6"` → (3, 6), `"U/9"` → (None, 9), etc.
- `_parse_past_races_json()` — JSON array → tuple of PastRace objects; handles null/empty/malformed gracefully
- `RunnerMeta` extended with 3 new fields (with defaults for backward compat): `past_races: tuple[PastRace, ...]`, `timeform_comment: str`, `recent_form: str`
- `_build_runner_meta()` — reads new columns from Parquet, handles missing columns for old files

#### Feature engineer — 17 new features per runner
- `past_race_features(meta, venue)` → 17 features:
  - Course form: `pr_course_runs`, `pr_course_wins`, `pr_course_win_rate` (case-insensitive venue match)
  - Distance form: `pr_distance_runs`, `pr_distance_wins` (±440 yards / ±2 furlongs tolerance)
  - Going form: `pr_going_runs`, `pr_going_wins`, `pr_going_win_rate` (going abbreviation match)
  - BSP: `pr_avg_bsp` (log-normed), `pr_best_bsp` (log-normed), `pr_bsp_trend` (linear slope, negative = improving)
  - Performance: `pr_avg_position`, `pr_best_position`, `pr_runs_count`, `pr_completion_rate`
  - Form trend: `pr_improving_form` (1.0 if last 3 positions descending)
  - Rest pattern: `pr_days_between_runs_avg`
- All features default to NaN when `past_races` is empty
- `runner_meta_features()` now prefers `recent_form` (from RaceCardRunners) over `form` (coldData) when available
- Called from `engineer_tick()` alongside existing `runner_meta_features()`

#### Environment
- `RUNNER_KEYS` extended with 17 past race feature keys → `RUNNER_DIM` = 110 (was 93)
- Total `obs_dim` = 31 + 7 + (110 × 14) + 5 = **1583** (was 1345, delta +238)
- Fully backward-compatible: old Parquet files without new columns produce all-zero past race features

#### Notes
- Existing models (trained on obs_dim=1345) are incompatible with the new obs_dim=1583. Retraining required
- 688/740 RaceCardRunners have `PastRacesJson` populated; 740/740 have `TimeformComment`

#### Tests
- **47 unit tests** (test_session_2_7b.py): position parsing (9 tests), PastRacesJson parsing (8 tests: valid/dnf/missing fields/empty/null/malformed), past_race_features (17 tests: no history/all keys/course form/case insensitive/no match/distance/going/BSP/trend/position/completion/improving form true/false/insufficient/days between runs), recent_form preference (2 tests), extractor merge (2 tests: new columns present, count=40), episode builder backward compat (4 tests), env dimensions (5 tests: RUNNER_DIM=110/keys count/past race keys/obs_dim=1583/no duplicates)
- **7 integration tests** (test_integration_session_2_7b.py): extraction has new columns, past_races_json populated, timeform_comment populated, past_races loaded into RunnerMeta, timeform_comment loaded, pr_* features populated, env runs full episode at obs_dim=1583
- All 895 Python tests pass, 19 skipped, 102 deselected

### Session 2.8 — Time-aware LSTM and time delta features
**Status:** Done

#### Feature engineer — time delta features
- `data/feature_engineer.py` — `TickHistory` now tracks tick timestamps (epoch seconds) in `_timestamp_history`
- `market_velocity_features()` produces 4 new features:
  - `seconds_since_last_tick` — wall-clock gap since previous tick, normalised by 300s (5 min), clamped [0, 1]. 0 for first tick
  - `seconds_spanned_3` — wall-clock time covered by last 3 ticks, normalised by 180s (3 min), clamped [0, 1]. 0 when < 3 ticks
  - `seconds_spanned_5` — same for 5-tick window, normalised by 300s
  - `seconds_spanned_10` — same for 10-tick window, normalised by 600s
- `reset()` clears timestamp history

#### Environment — observation space update
- `MARKET_VELOCITY_KEYS` extended with 4 time delta keys → `VELOCITY_DIM` = 11 (was 7)
- Total `obs_dim` = 31 + 11 + (110 × 14) + 5 = **1587** (was 1583, delta +4)
- `MARKET_TOTAL_DIM` in `policy_network.py` updated to 47 (was 43)

#### Policy network — TimeLSTMCell + PPOTimeLSTMPolicy
- `TimeLSTMCell` — custom LSTM cell where the forget gate incorporates time delta:
  `f_t = sigmoid(W_f @ [h, x] + W_dt * delta_t + b_f)`. Larger delta → more forgetting.
  `W_dt` is a learned parameter (one scalar per hidden unit), initialised to zero
- `PPOTimeLSTMPolicy` — wraps `TimeLSTMCell`, same interface as `PPOLSTMPolicy`.
  Extracts `seconds_since_last_tick` from the observation and feeds it to the cell
  at each timestep. Processes sequences by stepping through the cell (not `nn.LSTM`)
- Registered as `ppo_time_lstm_v1` in architecture registry

#### Population manager — str_choice support
- `HyperparamSpec.type` now supports `"str_choice"` for string-valued choices
- `sample_hyperparams()`, `validate_hyperparams()`, `mutate()` all handle `str_choice`
- `initialise_population()` uses sampled `architecture_name` when present in hp specs
- `mutate()` for `str_choice` jumps to adjacent choice in the list (no numeric delta)

#### Config
- `config.yaml` — `architecture_name` added to `hyperparameters.search_ranges` as `str_choice` with `[ppo_lstm_v1, ppo_time_lstm_v1]`. The genetic system can now evolve architecture alongside hyperparameters

#### Notes
- Existing models (trained on obs_dim=1583) are incompatible with the new obs_dim=1587. Retraining required
- `W_dt` initialised to zero means the TimeLSTM behaves identically to standard LSTM until trained. This is intentional — the network learns the time-awareness from data

#### Tests
- **44 unit tests** (test_session_2_8.py): time delta features (12 tests: first tick zero, 5s gap, 180s gap, clamp, uniform 5s spans, non-uniform spans, insufficient ticks, spanned_5, spanned_10, reset, all keys present), env dimensions (8 tests: VELOCITY_DIM=11, MARKET_DIM=31, RUNNER_DIM=110, keys present, obs_dim=1587, MARKET_TOTAL_DIM=47, no duplicates, env episode), TimeLSTMCell (6 tests: shapes, forget gate responds, zero delta deterministic, gradients flow, 2D input, monotonic with positive W_dt), PPOTimeLSTMPolicy (9 tests: output types, hidden state, sequence, architecture_name, description, action distribution, init_hidden, gradients, time delta affects hidden), architecture registry (3 tests: registered, create_policy, both architectures), str_choice (5 tests: parse, sample, validate valid/invalid, config entry), backward compat (2 tests: defaults to zero, old features present)
- **7 integration tests** (test_integration_session_2_8.py): time features populated on real data, first tick zero, nonzero after first, env full episode, forward pass on real obs, hidden state decay differs 5s vs 180s, training completes
- All 946 Python tests pass, 19 skipped, 102 deselected

---

### Session 1.5 — End-to-end single agent run
**Status:** Done

#### Data extraction
- `DataExtractor.get_available_dates()` discovered 2 dates in MySQL: 2026-03-26 (legacy ResolvedMarketSnaps) and 2026-03-28 (polled PolledMarketSnapshots)
- 2026-03-26 extracted: 29.6 MB ticks Parquet (4,182 ticks, 53 races), 71 KB runners Parquet
- 2026-03-28 already extracted: 200 KB ticks (390 ticks, 69 races), 272 KB runners

#### Chronological train/test split
- **Train:** 2026-03-26 (1 day, 53 races, 4,182 ticks)
- **Test:** 2026-03-28 (1 day, 69 races, 390 ticks)

#### Training (single agent, 3 epochs, CPU)
- `scripts/run_session_1_5.py` — trains population=1 via `TrainingOrchestrator`, evaluates on test split, prints scoreboard + per-day P&L + bet log + sanity checks
- Architecture: `ppo_lstm_v1` (randomised hyperparams, seed=42)
- Completed in 32.4s on CPU (PyTorch CPU-only build — CUDA wheels not installed)
- Training showed expected untrained-agent behaviour: epochs 1–2 reward ~-100 (losing entire budget), epoch 3 found degenerate "bet on everything" strategy (reward 211M, 21K bets — exploiting training environment)
- 9/10 sanity checks passed; the failing check correctly flagged the exploding training reward

#### Evaluation on test split (2026-03-28)
- P&L: -£99.98 (lost nearly entire £100 budget)
- 62 bets, 0 winning, precision 0.000
- All bets were backs at 1000.00 odds (max price) — agent has not learned a useful strategy
- Composite score: -0.1661 (win_rate=0.00, sharpe=0.00, mean_pnl=-99.98, efficiency=-0.08)

#### Pipeline verification
- Model registered in SQLite (active status, weights file saved)
- Evaluation run recorded with correct train_cutoff_date and test_days
- Per-day metrics persisted: day_pnl, bet_count, winning_bets, bet_precision, pnl_per_bet, early_picks, profitable
- Bet log written to Parquet (62 bets with all fields populated)
- Composite score computed via Scoreboard, scoreboard non-empty
- Progress events emitted: phase_start/phase_complete for training, evaluating, scoring, run_complete

#### Notes
- The untrained single agent with random hyperparams and 3 epochs on 1 training day is not expected to be profitable. The purpose of Session 1.5 is pipeline validation, not profitable trading
- The training reward explosion (epoch 3) is a known issue with the "bet on everything" archetype (documented in Session 2.5). This will be addressed by reward tuning and longer training in later sessions
- PyTorch CUDA wheels need reinstalling (`pip install torch --index-url https://download.pytorch.org/whl/cu126`) — CPU-only build was present at runtime

#### Tests
- **27 integration tests** (test_integration_session_1_5.py): model registered (4 tests: count, active, weights, architecture), evaluation run (3 tests: exists, test_days count, train_cutoff_date), per-day metrics (6 tests: count matches, dates match, pnl finite, pnl bounded, precision in range, profitable flag consistent), bet log (4 tests: bets recorded, fields populated, pnl sums match, Parquet files exist), scoreboard (4 tests: composite score computed, rank_all, score in valid range, metrics populated), progress events (6 tests: training/evaluating/scoring phases, run_complete, progress events, no selection/breeding for single agent)
- All 27 integration tests pass. 982 non-integration tests pass (5 pre-existing failures in 2.7b/2.8 integration tests due to DB state changes — unrelated to Session 1.5)

### Session 4.6 — Performance profiling & optimisation
**Status:** Done

#### Benchmark script
- `scripts/benchmark.py` — reusable profiling tool, measures wall-clock times for: data loading, feature engineering, rollout collection, PPO update, evaluation
- `--output` flag saves results as JSON for comparison; `--compare BEFORE AFTER` prints side-by-side delta table
- Baseline established: data loading 3.95s, rollout 8.48s (54%), PPO 1.75s (11%), eval 5.42s (35%), total train+eval 15.65s

#### Optimisation 1: orjson for JSON parsing
- `data/episode_builder.py` — replaced `json.loads()` with `orjson.loads()` for `parse_snap_json()` and `_parse_past_races_json()`
- Graceful fallback to stdlib `json` if orjson not installed
- `orjson>=3.9.0` added to `requirements.txt`
- **Result: data loading 3.95s -> 2.71s (31.5% faster)**

#### Optimisation 2: Optimised rollout loop
- `agents/ppo_trainer.py` — `_collect_rollout()` rewritten:
  - Pre-allocates a reusable GPU tensor buffer (avoids per-step tensor creation)
  - Wraps entire loop in single `torch.no_grad()` context
  - Manual Normal distribution sampling (avoids `Normal.sample()` + `log_prob()` overhead)
  - In-place `np.clip` for actions
- `training/evaluator.py` — `_evaluate_day()` optimised with same pre-allocated GPU buffer pattern
- **Result: rollout 8.48s -> 5.65s (33.3% faster, 493 -> 740 steps/s)**

#### Optimisation 3: Pinned memory for GPU transfers
- `agents/ppo_trainer.py` — `_ppo_update()` uses `pin_memory()` + `non_blocking=True` for CPU->GPU transfer of obs/action/log_prob batches
- Falls back to direct `torch.from_numpy()` on CPU

#### Optimisation 4: Parallel evaluation infrastructure
- `training/run_training.py` — evaluation phase supports `ThreadPoolExecutor` for multi-agent parallel evaluation
- Controlled by `training.eval_workers` config (default: 1 = sequential, for backward compatibility)
- Each worker creates its own `Evaluator` instance — thread-safe (read-only feature cache, no shared mutable state)
- Capped at `os.cpu_count()` to avoid oversubscription

#### Before/after comparison

| Operation | Before | After | Improvement |
|---|---|---|---|
| Data loading | 3.950s | 2.706s | **31.5% faster** |
| Feature engineering | 1.463s | 1.373s | 6.2% faster |
| Rollout collection | 8.476s | 5.653s | **33.3% faster** |
| PPO update | 1.746s | 1.847s | -5.8% (pinned memory overhead on small batch) |
| Evaluation | 5.423s | 4.782s | **11.8% faster** |
| **Total train+eval** | **15.645s** | **12.282s** | **1.27x speedup** |
| Rollout throughput | 493 steps/s | 740 steps/s | **50% faster** |

At full scale (20 agents x 30 days x 5 generations): estimated savings of ~45 minutes per generation from rollout speedup alone.

#### Tests
- **21 unit tests** (test_session_4_6.py): orjson parsing (10 tests: simple/nested/bytes/empty/unicode/snap nested/snap flat/past races valid/empty/malformed), optimised rollout (3 tests: transitions produced/valid fields/last done), pinned memory PPO (2 tests: CPU/CUDA), parallel eval config (2 tests: default workers/cap at CPU count), optimised evaluation (1 test: day records), benchmark script (2 tests: compare function/output file), orjson fallback (1 test)
- **8 integration tests** (test_integration_session_4_6.py): orjson real data (2 tests: all ticks parsed/prices valid), rollout on real data (3 tests: completes/PPO no NaNs/evaluation), benchmark results (3 tests: before exists/after exists/after faster)
- All 952 Python tests pass (2 skipped, 129 deselected DB-dependent integration), no regressions

### Session 3.6 + 3.7 — Race replay page & bet explorer page
**Status:** Done

#### Backend — bet explorer endpoint
- `api/routers/replay.py` — new `GET /{model_id}/bets` endpoint returns all evaluation bets for a model with summary stats (total_bets, total_pnl, bet_precision, pnl_per_bet)
- Route placed before `/{model_id}/{date}` to avoid path parameter collision
- `api/schemas.py` — new `ExplorerBet` and `BetExplorerResponse` Pydantic models
- Bugfix: `model_store.py` line 410 — `r.keys()` → `rows[0].keys()` (NameError when evaluation_days had rows)

#### Frontend — Race Replay page (`src/app/race-replay/`)
- **Selectors:** model → date → race (cascading dropdowns populated from API)
- **LTP price chart:** SVG line chart per runner (colour-coded), x-axis = time to off (counts down to 0), cursor line tracks current tick position
- **Playback controls:** play/pause button, speed control (1x/2x/5x/10x), tick slider, tick counter, time-to-off display
- **Order book panel:** best 3 back/lay prices+sizes for selected runner, updates with cursor position. Back/lay sides colour-coded (blue/red)
- **Action log panel:** chronological list of agent bets — time, runner, action type (BACK/LAY), price, stake, P&L. Click bet → cursor jumps to that tick
- **Runner legend:** colour-coded buttons, click to select runner for order book. Winner marked with green "W" badge
- **Summary bar:** total bets, race P&L (colour-coded), early picks count, venue, winner name
- Winner highlighted throughout: thicker chart line, green badge in legend, green stat in summary
- Route: `/replay` (lazy-loaded)

#### Frontend — Bet Explorer page (`src/app/bet-explorer/`)
- **Model selector:** dropdown populated from scoreboard API
- **Summary stats bar:** total bets, bet precision (%), P&L per bet, total P&L — all update with filters
- **Filter controls:** date (dropdown), race (dropdown), runner name (text search, case-insensitive), action (back/lay), outcome (won/lost). Clear button resets all
- **Sortable table:** columns for date, runner, action, time to off, price, stake, matched size, outcome, P&L. Click column headers to sort (toggle asc/desc). Sort indicators (▲/▼)
- **Visual styling:** action badges (blue back / red lay), outcome badges, P&L colour-coding, won/lost row left-border indicators
- **Results count:** "Showing N of M bets" updates with filters
- Route: `/bets` (lazy-loaded)

#### Shared changes
- **Header:** added "Replay" and "Bets" nav links between Training and Admin
- **Routes:** `/replay` and `/bets` added to `app.routes.ts` (lazy-loaded)
- **ApiService:** `getReplayDay()`, `getReplayRace()`, `getModelBets()` methods added
- **TypeScript models:** `replay.model.ts` (BetEvent, RaceSummary, TickRunner, ReplayTick, ReplayDayResponse, ReplayRaceResponse), `bet-explorer.model.ts` (ExplorerBet, BetExplorerResponse)
- Dark theme consistent with existing UI (#1e1e2e panels, #e0e0e0 text, #81c784 green, #e57373 red)

#### Tests
- **Python:** 6 new unit tests (test_api_replay.py): bet explorer model not found, no eval run, returns all bets, summary stats, bet fields, empty bets
- **Angular Race Replay:** 48 unit tests: component creation, page title, model loading/error, empty state, loading/error display, selectors rendered/disabled, date population, race population, race data loading, summary bar/stats/early picks, LTP chart rendering/runner data/winner marking/SVG paths, runner legend/winner badge, order book panel/selected runner/empty message/tick changes, action log/items/empty/bet click, playback controls/play button/toggle/speed/tick counter/slider/seek, time to off compute/display, cursor position/line, auto-select runner, runner change, helpers (shortId/formatSecondsToOff/runnerColour), winner ID/display, error handling (race/day load), edge cases (no ticks/no winner/destroy), state reset on model change
- **Angular Bet Explorer:** 45 unit tests: component creation, page title, model loading/error, empty state, loading/error display, model selector/load/error, summary bar/stats/empty stats, filters (date/action/outcome/runner/race/combined/clear/update stats), unique dates/races, sorting (default/toggle/price/pnl asc/indicator), table rendering/rows/columns, empty messages (no bets/filter empty), results count/filtered count, action badges (back/lay), outcome badges (won/lost), helpers (shortId/formatSecondsToOff), state reset
- **Angular integration:** 5 race replay tests + 6 bet explorer tests (skip when API not running)
- All 978 Python unit tests pass, all 285 Angular tests pass (24 skipped — integration)

### Session 4.9 — Start/stop training from the UI
**Status:** Done

#### Backend
- `training/run_training.py` — `TrainingOrchestrator` gains `stop_event: threading.Event` param. `_check_stop()` tested between agents, before evaluation, and between generations. Emits `run_stopped` phase_complete event on stop.
- `api/routers/training.py`:
  - `POST /training/start` — loads available dates from Parquet, splits chronologically ~50/50, spawns `TrainingOrchestrator.run()` via `asyncio.to_thread()`. Returns immediately with `{run_id, train_days, test_days, n_generations, n_epochs}`. Rejects 409 if already running, 400 if no data.
  - `POST /training/stop` — sets `stop_event`, orchestrator halts after current agent. Returns `{detail: "Stop requested..."}`. Rejects 409 if not running.
- `api/main.py` — `app.state.stop_event` (threading.Event) and `app.state.training_task` added to lifespan
- `api/schemas.py` — `StartTrainingRequest`, `StartTrainingResponse`, `StopTrainingResponse`

#### Frontend
- `training-monitor.ts` — Start Training form (generations + epochs inputs) shown when idle. Stop Training button (red) shown when running. Loading states for both.
- `api.service.ts` — `startTraining()` and `stopTraining()` methods
- `training-monitor.scss` — styled form fields, start/stop buttons

#### Tests
- **15 unit tests** (test_session_4_9.py): orchestrator stop_event (6 tests: accepted/default/false/true/emits event/only once), API endpoints (5 tests: start rejects running/no data/returns config, stop rejects not running/sets event), schemas (4 tests)
- All 962 Python tests pass, no regressions

### Session 4.7 — Opportunity window metric
**Status:** Done

#### Motivation
Measure whether a model's profitable bets found genuine market inefficiencies (price available for many seconds) vs fleeting noise (available for one tick). This is an **evaluation metric only** — does not affect reward signal or training. A "sniper" (short windows) and a "value finder" (long windows) are both valid strategies; the metric helps identify model archetypes.

#### Changes
- `env/bet_manager.py` — `Bet.tick_index: int = -1` field records which tick the bet was placed on
- `env/betfair_env.py` — `_process_action()` sets `bet.tick_index` at bet placement; `_MIN_STAKE` raised to £2.00 (Betfair Exchange minimum)
- `training/evaluator.py` — `compute_opportunity_window()` scans backward/forward through race ticks checking price availability, converts span to seconds. Evaluator now populates `tick_timestamp`, `seconds_to_off`, and `opportunity_window_s` on bet records, plus day-level mean/median aggregates
- `registry/model_store.py` — `EvaluationBetRecord.opportunity_window_s`, `EvaluationDayRecord.mean/median_opportunity_window_s`, SQL migration, Parquet I/O (backward-compatible)
- `registry/scoreboard.py` — `ModelScore.mean_opportunity_window_s` (informational, NOT part of composite score)

#### Tests
- **20 unit tests** (test_session_4_7.py): opportunity window computation (9 tests), tick_index (2), bet/day/score records (6), Parquet round-trip + backward compat (2), SQLite round-trip (1)
- **2 integration tests** (test_integration_session_4_7.py): real data windows computed, tick_timestamp populated
- All 980 Python tests pass, no regressions

### Session 4.10 — Budget-per-race and bet limits
**Status:** Done

#### Motivation
With the old carry-over budget, proportional staking caused exponential compounding — a £100 budget could grow to £10,000+ by race 5, allowing absurdly large bets. Agents exploited this by laying every runner (most lay bets win since most runners lose), compounding into millions. This session fixes the economics to be realistic.

#### Changes
- **Budget reset per race** — `env/betfair_env.py` creates a fresh `BetManager(starting_budget)` at the start of each race. Day P&L = sum of per-race P&Ls (true accounting). An agent that bets on nothing shows £0 P&L, not £100 "profit".
- **Max bets per race** — `config.yaml: training.max_bets_per_race: 20` (configurable). `_process_action()` checks `bm.race_bet_count()` and stops placing bets when the limit is reached. Prevents tick-spamming.
- **Accumulated positions** — `BetManager.get_positions(market_id)` returns net back/lay exposure and bet count per `selection_id` within a race. `BetManager.race_bet_count(market_id)` counts bets for a specific race.
- **Agent state observation** — `AGENT_STATE_DIM` increased from 5 to 6 (+`day_pnl_norm`). New `POSITION_DIM = 3` per runner: back_exposure, lay_exposure, runner_bet_count. Total per-runner position vector = 3 × 14 = 42 dims appended to observation.
- **Observation dim change** — obs_dim: 1587 → 1630 (+1 agent state + 42 position features). All policy network `_split_obs()` methods updated to extract and concatenate position features with runner features. `RUNNER_INPUT_DIM = RUNNER_DIM + POSITION_DIM = 113` used for runner encoder input.
- **Info dict** — `day_pnl` key added. `bet_count` and `winning_bets` now sum across completed races + current race (avoids double-counting). `budget_before` in `RaceRecord` now stores `starting_budget` (not post-bet economic value).
- **Breaking change** — all existing models are incompatible (different env dynamics and observation space). Registry must be cleared before retraining.

#### Tests
- **7 new BetManager tests**: `get_positions` (empty, single back, accumulated, lay, mixed, filtered by market), `race_bet_count`
- **12 new BetfairEnv tests**: budget resets between races (3), day P&L accounting (2), do-nothing zero P&L, voided race doesn't affect next race, max bets enforced (3), position tracking in observation (4)
- Updated 15 existing tests for new budget semantics, obs_dim (1630), AGENT_STATE_DIM (6), MARKET_TOTAL_DIM (48), RUNNER_INPUT_DIM (113)
- All 487+ Python unit tests pass

---

## Skipped / Deferred Sessions

(none currently)

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
| 2.7a    | 47              | 7                      | **828 + 125** (Python) + **188 + 12** (Angular) |
| 2.7b    | 47              | 7                      | **875 + 132** (Python) + **188 + 12** (Angular) |
| 2.8     | 44              | 7                      | **919 + 139** (Python) + **188 + 12** (Angular) |
| 1.5     | 0               | 27                     | **919 + 166** (Python) + **188 + 12** (Angular) |
| 4.6     | 21              | 8                      | **940 + 174** (Python) + **188 + 12** (Angular) |
| 3.6+3.7 | 6 (Python) + 93 (Angular) | 11 (Angular) | **946 + 174** (Python) + **285 + 24** (Angular) |
| 4.7     | 20              | 2                      | **966 + 176** (Python) + **285 + 24** (Angular) |
| 4.10    | 19              | 0                      | **985 + 176** (Python) + **285 + 24** (Angular) |

**Python total: 487+ unit tests passed. 19 tests added in Session 4.10 (7 BetManager + 12 BetfairEnv). 15 existing tests updated for new obs_dim/budget semantics.**
**Angular total: 285 passed, 24 skipped (integration — API not running). (Unchanged.)**

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

11. **Polled → SnapJson normalisation** (Session 2.7a) — `PolledMarketSnapshots.RunnersJson` uses a different layout (`selectionId`, `state.*`, `exchange.*`) than `ResolvedMarketSnaps.SnapJson` (`MarketRunners → RunnerId/Definition/Prices`). Rather than adding a second parser, the polled format is normalised into SnapJson format at extraction time, keeping all downstream code unchanged.

12. **obs_dim breaking change** (Session 2.7a) — Adding 7 race status features changes obs_dim from 1338 to 1345. Existing trained models are incompatible and must be retrained. This is acceptable because the model registry is empty (no production models yet).

13. **obs_dim breaking change** (Session 2.7b) — Adding 17 past race features per runner changes obs_dim from 1345 to 1583 (+238 = 17 × 14 max_runners). Same rationale as decision 12 — no production models yet.

14. **RaceCardRunners field override** (Session 2.7b) — When both coldData (RunnerMetaData) and RaceCardRunners have the same field (age, weight, jockey, trainer, days_since_last_run, gender), prefer RaceCardRunners as it's fetched from Timeform race cards on race day and is fresher than the static coldData snapshot. Override only when the RaceCardRunners value is non-null.

15. **recent_form over FORM** (Session 2.7b) — `RunnerMetaData.FORM` is a static snapshot. `RaceCardRunners.RecentForm` is fresher. `runner_meta_features()` now prefers `recent_form` when available, falling back to `form` when empty.

16. **Budget-per-race over carry-over** (Session 4.10) — With proportional staking and carry-over budget, early wins compounded exponentially (£100 → £10,000+ by race 5). Agents exploited this by laying every runner. Per-race budget reset eliminates this exploit. Day P&L = sum of race P&Ls. Capital at risk = `starting_budget × races_played`.

17. **Position features in observation** (Session 4.10) — Per-runner position info (back/lay exposure, bet count) added to observation so the agent can manage accumulated positions within a race. Concatenated with runner features in the policy network's runner encoder (RUNNER_INPUT_DIM = 113).
