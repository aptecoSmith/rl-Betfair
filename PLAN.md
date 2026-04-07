# rl-betfair — Architecture & Design Plan

## Vision

A population-based reinforcement learning system that learns to trade on the
Betfair horse racing exchange. Multiple RL agents compete and evolve over time.
The best-performing agents are kept, the weakest discarded, and new generations
inherit traits from their parents. Eventually, the best models trade live.

---

## Core Principles

- **No strategy hints.** The agent learns everything from data. No engineered
  features that encode human trading intuition.
- **Realistic simulation.** Bet matching checks historical order book volume —
  a back bet only fills if there was enough lay-side liquidity at that price.
- **Day-as-episode.** One episode = one full racing day. Budget, open positions,
  and race history carry across races within the day. The agent can learn
  intra-day patterns (morning vs afternoon, venue sequences, market fatigue).
- **Growing experience.** As the dataset grows, models are retrained on
  all available history. The train/validation split is always by date (no
  lookahead). ~50% train, ~50% validation.
- **Persistent scoreboard.** Every model that has ever been created is tracked.
  On each training run, all non-discarded models are evaluated. A model created
  months ago may still rank highly.
- **Local GPU only.** RTX 3090 (24 GB VRAM). No cloud services.
- **Always show ETAs.** Every long-running operation exposes two-level progress:
  item ETA (current thing) and process ETA (everything). Visible in the UI at
  all times so the user can check in and know when to come back.

---

## Episode Structure

```
Day episode begins (budget = £100)
  │
  ├── Race 1 (e.g., 11:30 Newmarket)
  │     ├── Tick t=0  → agent observes state, may act (back/lay/nothing)
  │     ├── Tick t=1  → ...
  │     ├── ...
  │     └── Race closes → bets settled, P&L realised, budget updated
  │
  ├── Race 2 (e.g., 12:00 Ascot)
  │     └── ... same structure ...
  │
  └── Final race closes → episode ends, total day P&L computed
```

The agent's hidden state (LSTM or Transformer) persists across all ticks in
the day, allowing it to carry context from earlier races into later ones.

---

## Action Space (per tick)

At every 5-second tick, for each active runner in the current race, the agent
can choose from:

- **Back** at current best back price, stake = S (continuous, 0 to budget_remaining)
- **Lay** at current best lay price, stake = S (continuous, 0 to max_liability)
- **Do nothing**

Since there can be up to ~14 runners per race, the action space is:
`N_runners × {back, lay, nothing} × continuous stake`

Implementation: hierarchical or factored action head — the network outputs a
vector of (action_type, stake) per runner, and we mask out invalid actions
(e.g., can't back more than budget allows).

---

## Reward Signal

Primary reward (end of each race):
- Net P&L for all bets placed in that race

Bonus reward modifier:
- If the winning runner was correctly backed (net positive position) and the
  first bet on it was placed ≥ 5 minutes before the off: apply a multiplier
  (e.g. ×1.2 to ×1.5). Encourages early confident picks without mandating them.

Efficiency penalty:
- A small per-bet cost (e.g., 0.01 reward units) to discourage excessive
  low-conviction betting. Calibrate so it doesn't override genuine P&L signals.

End-of-day bonus:
- Bonus reward proportional to total day P&L, to reinforce day-level strategy.

The composite scoring formula for model ranking on the scoreboard:
```
score = day_PnL - (bet_count × efficiency_penalty) + early_pick_bonus
```
Exact coefficients to be tuned empirically.

---

## Evaluation Methodology

Training and testing are separated **chronologically** (no lookahead).
With N days of data: the earliest ~50% are training days, the later ~50% are
test days. This ratio is re-computed on every training run as data grows.

### Per-day evaluation (not aggregate)

A model is evaluated by running it independently on **each test day** and
recording a full metrics snapshot per day. The model never "remembers" across
test days — each day is a fresh episode (budget reset to £100).

Per-day metrics recorded:
- `date` — the test day
- `day_pnl` — net P&L for that day (£)
- `bet_count` — total bets placed across all races
- `winning_bets` — bets that returned net positive
- `bet_precision` — winning_bets / bet_count (fraction of bets that made money)
- `pnl_per_bet` — day_pnl / bet_count (economy — reward per decision made)
- `early_picks` — bets placed ≥5 min before off on the eventual winner (net positive)
- `profitable` — boolean, day_pnl > 0

### Scoreboard composite score

The scoreboard score is computed from the **distribution** of per-day results,
not a single aggregate. This rewards consistency over lucky streaks.

```
win_rate         = profitable_days / total_test_days
mean_daily_pnl   = mean(day_pnl)
sharpe           = mean(day_pnl) / std(day_pnl)   # consistency-adjusted return
max_loss_day     = min(day_pnl)                    # worst single day
bet_precision    = mean(winning_bets / bet_count)  # fraction of bets that made money
pnl_per_bet      = mean(day_pnl / bet_count)       # economy: reward per decision

# Efficiency rewards fewer, better bets. A model making 3 winning bets
# outscores one making 20 bets for the same total P&L.
efficiency = (bet_precision × 0.5) + (normalised_pnl_per_bet × 0.5)

composite_score = (win_rate  × 0.35)
                + (sharpe    × 0.30)   # normalised to [-1, 1] range
                + (mean_daily_pnl × 0.15)   # normalised by starting budget
                + (efficiency     × 0.20)   # weighted up — quality over quantity
```

Coefficients are configurable in `config.yaml` and may themselves be subject
to tuning.

### Discard policy

A model is a candidate for discard only if **all** of the following are true:
- win_rate < 0.35 (fewer than 35% of test days profitable)
- mean_daily_pnl < 0 (net losing on average)
- sharpe < -0.5 (consistently negative relative to its own variance)

A model is **never** discarded solely for having a run of bad days. Bad days
are expected — a model that survives 45/50 test days profitably is valuable
even if the other 5 are losses.

Discarding is also manual — the user can override from the scoreboard UI.

### Historical re-evaluation

On every training run, all non-discarded models (including old ones) are
re-evaluated against the current full test set. A model created months ago
may rank highly against newer ones if its strategy remains robust as the
dataset grows. This is the persistent scoreboard.

---

## State Representation (per tick)

The observation vector for each tick is a concatenation of:

### Market-level features (for the current race)
- Time to scheduled off (seconds, normalised)
- Time since market opened (seconds, normalised)
- Is in-play flag (0/1) — should always be 0 if pre-race only
- Total matched volume (log-normalised)
- Number of active runners
- Market overround (sum of implied probabilities)

### Per-runner features (for each runner, padded to max_runners=14)
- Best 3 back prices (or 0 if not available)
- Best 3 back sizes (log-normalised)
- Best 3 lay prices
- Best 3 lay sizes
- Last traded price (LTP)
- Implied probability from LTP (1/LTP)
- Price velocity (LTP change over last 3 ticks, 10 ticks)
- Volume traded on this runner (log-normalised)
- BSP near / far indicators
- Forecast price (from RunnerMetaData, normalised)
- Official rating (normalised within race)
- Days since last run (normalised)
- Stall draw (normalised within race)

### Weather features (from WeatherObservations, PRE_RACE snapshot)
- Temperature (°C, normalised)
- Precipitation (mm, normalised)
- Wind speed (m/s, normalised)
- WMO weather code (one-hot or embedding)

### Agent state (within-episode context)
- Current budget remaining (normalised by starting budget)
- Current open liability (total lay exposure, normalised)
- Number of bets placed today so far
- Number of races completed today
- Time of day (normalised, 0=10:00, 1=22:00)

**No runner identity features** (horse name, jockey name as strings) in the
vector directly — these are categorical and high-cardinality. Options:
- Embed them (learned embedding layer) — preferred for v2
- Omit for v1 (market signals only)
 - talk to the user about this - he feels that we may well want to track horses or jockeys across time.  For example, we might learn that a horse is always backed late in a market - this would be useful to know.

---

## Bet Matching Simulation

When the agent places a back bet on runner R at stake S:
1. Look at the historical `AvailableToLay` ladder at that tick
2. The best lay price in the order book is the back price the agent pays
3. If `AvailableToLay[0].Size >= S`, the full bet is matched
4. If `AvailableToLay[0].Size < S`, partial fill at level 0; remainder tries level 1, then level 2
5. Any unmatched remainder is cancelled (no resting orders in v1)

Lay bets mirror this using `AvailableToBack`.

This is deterministic from `ResolvedMarketSnaps` — we have the full order book
at every tick so no assumptions are needed.

---

## RL Architecture

### Architecture Registry

All architectures are registered by name and selectable via `config.yaml`.
This allows different agents in the population to use different architectures,
and new architectures to be added without touching training or evaluation code.

```python
# agents/architecture_registry.py
REGISTRY = {
    "ppo_lstm_v1":        PPOLSTMPolicy,        # current default
    "ppo_transformer_v1": PPOTransformerPolicy,  # Phase 4
    # future entries added here
}
```

Each architecture class implements a common interface:
- `__init__(self, obs_dim, action_dim, hyperparams)`
- `forward(self, obs, hidden_state) → (action_dist, value, new_hidden_state)`
- `architecture_name: str` (class attribute, matches registry key)
- `description: str` (plain-English summary stored in model registry)

The architecture name is stored in the model registry alongside model weights,
so the correct class is always used when loading a saved model.

### Architecture v1: PPO + LSTM (`ppo_lstm_v1`)

**Why this first:** PPO is the most widely validated deep RL algorithm for
continuous action spaces. LSTM handles the sequential nature of the tick stream
and carries cross-race context within a day. Together they are well-understood,
stable to train, and a strong baseline before more exotic architectures.

**Structure:**
```
Input: state vector (variable-length, padded to max_runners=14)
  │
  ├── Runner feature encoder: per-runner MLP (shared weights)
  │     → permutation-invariant runner embeddings
  │
  ├── Market feature encoder: MLP
  │     → market-level embedding
  │
  ├── Concatenate runner embeddings + market embedding
  │
  ├── LSTM — hidden state h carries across:
  │     - ticks within a race
  │     - races within a day (key: allows intra-day pattern learning)
  │
  ├── Actor head: per-runner (action_type logits + stake magnitude)
  │
  └── Critic head: scalar V(s)
```

**Training algorithm:** PPO (clipped surrogate objective, GAE advantage
estimation, entropy bonus for exploration).

**Known limitations (reasons to try alternatives later):**
- LSTM can struggle with very long sequences (a full day = thousands of ticks)
- PPO is on-policy — sample efficiency is lower than off-policy methods (SAC)
- Continuous stake output requires careful normalisation

### Future Architectures (documented for reference)

| Name | Algorithm | Sequence model | When to try |
|---|---|---|---|
| `ppo_transformer_v1` | PPO | Multi-head self-attention | Phase 4 — when dataset > 60 days |
| `sac_lstm_v1` | SAC (off-policy) | LSTM | If PPO sample efficiency becomes a bottleneck |
| `rainbow_dqn_v1` | Rainbow DQN | None (discretised actions) | If continuous actions prove too hard to train |
| `dreamer_v1` | DreamerV3 (world model) | RSSM (recurrent latent) | Long-term experiment — learn a market model |
| `hierarchical_v1` | Hierarchical RL | Two-level LSTM | If race-selection becomes a distinct subproblem |

Each of these is a backlog item in TODO.md. They are not planned for
immediate implementation — document first, build when baseline is validated.

### Population & Genetic Evolution

- Population size: **N=20** to start (configurable in `config.yaml`)
- Each generation: train all N agents on training data, evaluate on all
  validation days individually, score, select, breed
- **Selection**: top 50% by composite score survive
- **Elitism**: top N_elite (default 3) always survive unchanged
- **Crossover**: uniform — for each hyperparameter, randomly inherit from
  parent A or parent B
- **Mutation**: Gaussian noise on continuous params; occasional discrete jumps
  on integer params (layer count, hidden size)
- **Architecture crossover**: an agent can inherit a different architecture
  from a parent (if the population contains mixed architectures in later phases)

Hyperparameters subject to evolution:

| Hyperparameter | Type | Search range |
|---|---|---|
| `learning_rate` | float (log scale) | 1e-5 → 1e-3 |
| `ppo_clip_epsilon` | float | 0.1 → 0.3 |
| `entropy_coefficient` | float | 0.001 → 0.05 |
| `lstm_hidden_size` | int | 64, 128, 256, 512 |
| `mlp_hidden_size` | int | 64, 128, 256 |
| `mlp_layers` | int | 1 → 4 |
| `early_pick_bonus_min` / `early_pick_bonus_max` | float | 1.0 → 1.8 |
| `reward_efficiency_penalty` | float | 0.001 → 0.05 |
| `architecture_name` | str | keys in REGISTRY |

> Note: `observation_window_ticks` was retired in arch-exploration Session 1
> (sampled but unused); the scalar `reward_early_pick_bonus` was split into
> `_min`/`_max` in arch-exploration Session 3. `config.yaml` is the
> authoritative list of live genes.

### Genetic Event Logging

Every selection, crossover, and mutation event is logged in two places:

**1. Human-readable log file** (`logs/genetics/gen_{N}_YYYY-MM-DD.log`):
```
=== Generation 4 — 2026-05-12 ===

SELECTION
  Survived (elite):   model_a1b2 [score=0.82], model_c3d4 [score=0.79]
  Survived (top 50%): model_e5f6 [score=0.71], model_g7h8 [score=0.68], ...
  Discarded:          model_i9j0 [score=0.21, win_rate=0.28, 3 strikes]

BREEDING
  Child: model_x1y2z3
    Parent A: model_a1b2 (score=0.82)
    Parent B: model_e5f6 (score=0.71)
    Trait inheritance:
      learning_rate:         0.0003 (from A)  ← no mutation
      ppo_clip_epsilon:      0.2    (from B)  ← no mutation
      entropy_coefficient:   0.012  (from A) → 0.015  (mutated +0.003)
      lstm_hidden_size:      256    (from B) → 512    (mutated +256, discrete jump)
      mlp_hidden_size:       128    (from A)  ← no mutation
      mlp_layers:            2      (from A)  ← no mutation
      observation_window:    10     (from B) → 8      (mutated -2)
      reward_early_pick:     1.3    (from A)  ← no mutation
      reward_efficiency:     0.01   (from B)  ← no mutation
      architecture_name:     ppo_lstm_v1 (from A)
    Summary: 3 traits mutated. Higher LSTM capacity inherited from B.
             Increased entropy may improve exploration.
```

**2. SQLite table** (`registry/models.db` → `genetic_events`):

| Column | Description |
|---|---|
| `event_id` | UUID |
| `generation` | int |
| `event_type` | `selection`, `crossover`, `mutation`, `discard` |
| `child_model_id` | UUID (null for selection/discard events) |
| `parent_a_id` | UUID |
| `parent_b_id` | UUID (null if no second parent) |
| `hyperparameter` | which param this event relates to |
| `parent_a_value` | value from parent A |
| `parent_b_value` | value from parent B |
| `inherited_from` | `A`, `B`, or `mutation` |
| `mutation_delta` | numeric delta if mutated, else null |
| `final_value` | the value the child received |
| `selection_reason` | `elite`, `top_50pct`, `bred`, `discarded` |
| `human_summary` | the one-line summary string (for UI display) |

This table feeds the **Model Detail** page in the UI — every model's lineage
shows exactly which traits it inherited and which were mutations.

---

## Data Pipeline

```
MySQL (localhost:3307)
  │
  ├── ColdData: runner metadata, weather, results, market catalogue
  └── HotData: ResolvedMarketSnaps (full order book at each tick)
          │
          └── extractor.py
                │  Joins: snaps + runner metadata + weather + results
                │  Groups by: date → race → tick
                │  Outputs: one Parquet file per day
                │
                └── data/processed/YYYY-MM-DD.parquet
                          │
                          └── episode_builder.py
                                │  Constructs Day objects (list of Race objects,
                                │  each Race = list of Tick observations)
                                └── Ready for Gymnasium env
```

Source: `ResolvedMarketSnaps` (full state at each tick, not deltas). Each snap
contains the complete order book — no reconstruction needed.

Train/validation split: chronological. E.g., with 60 days of data, days 1-30
are train, days 31-60 are validation. Re-split each training run as data grows.

---

## Project Structure

```
rl-betfair/
├── PLAN.md                      ← this file
├── TODO.md                      ← current task list
├── requirements.txt
├── config.yaml                  ← population size, reward coefficients, etc.
│
├── data/
│   ├── extractor.py             ← MySQL → Parquet export
│   ├── episode_builder.py       ← Parquet → Episode objects
│   ├── feature_engineer.py      ← derived features (velocity, implied prob)
│   └── processed/               ← one .parquet per day (gitignored)
│
├── env/
│   ├── betfair_env.py           ← Gymnasium day-episode environment
│   ├── order_book.py            ← realistic bet matching simulation
│   └── bet_manager.py           ← tracks open bets, liability, P&L
│
├── agents/
│   ├── policy_network.py        ← LSTM policy + value networks (PyTorch)
│   ├── ppo_trainer.py           ← PPO training loop for one agent
│   └── population_manager.py   ← population, selection, crossover, mutation
│
├── registry/
│   ├── model_store.py           ← save/load models + metadata (SQLite)
│   └── scoreboard.py            ← compute and rank all non-discarded models
│
├── training/
│   ├── run_training.py          ← entry point: train population on latest data
│   └── evaluator.py             ← run a model on validation data, return metrics
│
├── api/
│   ├── main.py                  ← FastAPI app
│   ├── routers/
│   │   ├── models.py            ← scoreboard, model detail, lineage
│   │   ├── training.py          ← training status, progress
│   │   └── replay.py            ← event replay data for UI
│   └── schemas.py               ← Pydantic models
│
└── frontend/                    ← Angular app
    ├── src/app/
    │   ├── scoreboard/          ← model rankings
    │   ├── training-monitor/    ← live training progress
    │   ├── race-replay/         ← replay a model's decisions on a given day
    │   └── model-detail/        ← lineage, hyperparams, bet history
    └── ...
```

---

## ETA Tracking (Cross-Cutting)

Every long-running operation publishes two-level progress in real time:
- **Item ETA** — time remaining for the current atomic unit of work
- **Process ETA** — time remaining for the entire operation

### Operations that emit progress

| Operation | Item unit | Process unit |
|---|---|---|
| Data extraction | 1 day extracted | All days to extract |
| Episode building | 1 day built | All days to build |
| Agent training (one agent) | 1 episode | All episodes for this agent |
| Population training (full generation) | 1 agent trained | All agents in the generation |
| Evaluation (one agent) | 1 test day | All test days for this agent |
| Full scoreboard re-evaluation | 1 model evaluated | All models to re-evaluate |
| Genetic selection + breeding | 1 child bred | All children to create |

### ETA calculation — `ProgressTracker` utility

```python
# training/progress_tracker.py
class ProgressTracker:
    def __init__(self, total: int, label: str, rolling_window: int = 10):
        self.total = total
        self.label = label
        self.completed = 0
        self._times: deque = deque(maxlen=rolling_window)
        self._last_tick = time.monotonic()

    def tick(self):
        """Call once per completed item."""
        now = time.monotonic()
        self._times.append(now - self._last_tick)
        self._last_tick = now
        self.completed += 1

    @property
    def item_eta_seconds(self) -> float | None:
        """ETA for the NEXT item based on rolling average."""
        return mean(self._times) if self._times else None

    @property
    def process_eta_seconds(self) -> float | None:
        """ETA for all remaining items."""
        if not self._times:
            return None
        remaining = self.total - self.completed
        return mean(self._times) * remaining

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "completed": self.completed,
            "total": self.total,
            "pct": round(self.completed / self.total * 100, 1),
            "item_eta_s": self.item_eta_seconds,
            "process_eta_s": self.process_eta_seconds,
            "item_eta_human": _fmt(self.item_eta_seconds),
            "process_eta_human": _fmt(self.process_eta_seconds),
        }
```

Uses a **rolling window of the last N completions** (default 10) rather than
elapsed/fraction. This handles warmup correctly — the first few items being
slow don't distort the whole estimate.

### WebSocket message schema

All progress events are published to the WebSocket at `/ws/training`.
The frontend consumes this stream for all ETA display.

```json
{
  "event": "progress",
  "timestamp": "2026-05-12T14:23:01Z",
  "run_id": "abc123",
  "phase": "training",
  "process": {
    "label": "Generation 4 — training 20 agents",
    "completed": 7,
    "total": 20,
    "pct": 35.0,
    "item_eta_human": "6 min",
    "process_eta_human": "1 h 18 min"
  },
  "item": {
    "label": "Training agent model_x1y2z3 (ppo_lstm_v1)",
    "completed": 312,
    "total": 1000,
    "pct": 31.2,
    "item_eta_human": "4 min 12 sec",
    "process_eta_human": "6 min 05 sec"
  },
  "detail": "Episode 312 | reward=+1.24 | P&L=+£3.40 | loss=0.0042"
}
```

Other event types on the same WebSocket:
- `phase_start` — a new phase begins (extraction, training, evaluation, genetics)
- `phase_complete` — phase done, with summary stats
- `agent_complete` — one agent finished training, with final metrics
- `run_complete` — entire training run done, scoreboard updated

### UI display

The Training Monitor page shows two persistent ETA bars at the top:

```
┌─────────────────────────────────────────────────────────┐
│ PROCESS  Generation 4 · 7 of 20 agents trained          │
│ ████████░░░░░░░░░░░░░░░░░ 35%    Process ETA: 1h 18m   │
├─────────────────────────────────────────────────────────┤
│ ITEM     Training model_x1y2z3 (ppo_lstm_v1)            │
│ ████████████░░░░░░░░░░░░░ 31%    Item ETA:    4m 12s   │
└─────────────────────────────────────────────────────────┘
```

These bars are always visible regardless of which sub-page of the Training
Monitor is active. If no run is in progress, they show the last completed
run's summary and elapsed time.

The scoreboard page also shows a small status chip:
- 🟢 Idle — last run completed 2h ago
- 🟡 Running — Generation 4 in progress · ETA 1h 18m
- 🔴 Error — last run failed (click for details)

### Phases and their labels

| Phase constant | Label shown in UI |
|---|---|
| `extracting` | Extracting market data from MySQL |
| `building` | Building training episodes |
| `training` | Training agents (Generation N) |
| `evaluating` | Evaluating models on test days |
| `selecting` | Genetic selection |
| `breeding` | Breeding next generation |
| `scoring` | Updating scoreboard |

---

## Evaluation Storage (for Replay)

During each evaluation run, in addition to per-day aggregate metrics, we persist
a full action log so any race can be replayed in the UI.

### `evaluation_runs` table (SQLite)
- `run_id` (UUID) — unique per evaluation
- `model_id` — FK to model
- `evaluated_at` — timestamp
- `train_cutoff_date` — last training day used
- `test_days` — JSON list of dates evaluated

### `evaluation_days` table
- `run_id`, `date`, `day_pnl`, `bet_count`, `winning_bets`, `bet_precision`,
  `pnl_per_bet`, `early_picks`, `profitable`

### `evaluation_bets` table
- `run_id`, `date`, `race_id`, `market_id`
- `tick_timestamp` — when the bet was placed
- `seconds_to_off` — time until scheduled start at bet placement
- `runner_id`, `runner_name`
- `action` — `back` or `lay`
- `price` — price at which the bet was placed
- `stake` — amount staked
- `matched_size` — how much was actually filled (realistic matching)
- `outcome` — `won`, `lost`, `void`
- `pnl` — net P&L for this individual bet

The market tick data itself (order book state at every tick) is not duplicated
in the registry — it lives in the Parquet files keyed by date. The replay API
joins `evaluation_bets` with the Parquet tick data on (date, market_id,
tick_timestamp) to reconstruct the full picture.

---

## Model Registry (Persistent Scoreboard)

Each model record stores:
- Unique ID (UUID)
- Generation number
- Parent model IDs (for lineage tracking)
- Hyperparameters (JSON)
- Architecture spec (JSON)
- Creation date
- Last evaluation date
- Per-run metrics: `{date, train_pnl, val_pnl, bet_count, early_picks, score}`
- Status: `active | discarded`
- Weights file path

SQLite database (`registry/models.db`) — simple, local, no server needed.

---

## UI Pages (Angular)

| Page | What it shows |
|---|---|
| **Scoreboard** | All active models ranked by composite score, colour-coded by generation, trend sparkline |
| **Training Monitor** | Current training run progress — which agent is training, episode rewards, loss curves (WebSocket feed) |
| **Model Detail** | Hyperparameters, lineage tree, per-day P&L chart, bet breakdown |
| **Race Replay** | Pick a model + a date + a race → watch the market play out (LTP price chart per runner, animated order book depth), with a side panel showing agent actions chronologically. Click an action to jump to that moment. Winner highlighted at end. Summary: bet count, P&L, early picks. |
| **Bet Explorer** | All bets placed by a model across validation data — filter by race, runner, outcome. Shows bet precision and P&L per bet across the dataset. |

---

## Development Phases

### Phase 0 — Foundation (no data yet)
- Set up repo structure, requirements, config
- Build data extractor (even with 0 days, get the SQL queries right)
- Build Gymnasium environment (unit-testable with synthetic data)
- Build bet matching simulation

### Phase 1 — First Agent (2+ days of data)
- Extract first real episodes
- Train a single PPO agent end-to-end
- Verify reward signal, check P&L is sane
- Build model registry and scoreboard (even for one model)

### Phase 2 — Population
- Implement population manager (N=20 agents)
- Implement genetic selection and mutation
- Run first multi-generation experiment
- Basic FastAPI backend + scoreboard endpoint

### Phase 3 — UI
- Angular frontend connected to API
- Training monitor with WebSocket progress
- Race replay viewer

### Phase 4 — Scale & Refine
- More data → retrain all models
- Tune reward coefficients
- Experiment with Transformer instead of LSTM
- Experiment with runner embeddings (jockey, trainer)

### Phase 5 — Live Trading
- Connect best model to StreamRecorder1 live feed
- Paper trade first (simulate bets, no real money)
- Real money only after sustained validation performance

---

## Open Questions (to revisit)

- Exact reward coefficient values — set empirically after Phase 1
- Whether to embed runner identity (jockey/trainer) — defer to Phase 4
- Crossover mechanism for neural network weights — weight interpolation vs
  architecture crossover (NEAT-style) — start with hyperparameter-only crossover
- Transformer vs LSTM — LSTM first, evaluate later
- Whether to include in-play ticks in episodes (currently excluded — pre-race only)
