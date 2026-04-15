# RL Project Brief — Betfair Horse Racing

Use this document as context when starting a new conversation about building
the reinforcement learning project. It describes the data that StreamRecorder1
captures, how to access it, and the design questions that need answering.

---

## 1. What StreamRecorder1 Captures

StreamRecorder1 connects to the Betfair Exchange streaming API and records
every price update for every GB horse racing WIN market, from morning until
close of play. It writes to two MySQL databases:

### ColdData (`coldData`) — populated once daily

| Table | What it holds |
|---|---|
| `Todays_venues` | Venue names + date |
| `Todays_markets` | Full Betfair MarketCatalogue (market name, description, event) |
| `marketOnDates` | Market ID, venue, start time, runner count, market name |
| `runnerdescription` | Horse name, selection ID, handicap, sort priority |
| `RunnerMetaData` | Jockey, trainer, owner, form, age, sex, weight, cloth number, sire/dam breeding, official rating, stall draw, forecast price |
| `marketResults` | Winner selection ID, venue, market type, event info, country code |
| `VenueLocations` | ~60 GB racecourses with GPS lat/lon (static seed data) |
| `WeatherObservations` | Per-race weather at three points: PRE_RACE (30 min before), AT_OFF (race start), POST_RACE (result declared). Fields: temperature, precipitation, wind speed/direction, humidity, WMO weather code, full raw JSON |

### HotData (`hotDataRefactored`) — updated every ~5 seconds per market

| Table | What it holds |
|---|---|
| `updates` | One row per market snapshot: market ID, venue, market name, start time, status (OPEN/SUSPENDED/CLOSED), in-play flag, traded volume, number of runners/active runners, bet delay, BSP market flag |
| `marketRunnerSnapAtTimes` | One row per runner per snapshot: selection ID, runner name, handicap, status (ACTIVE/WINNER/LOSER), sort priority, adjustment factor, BSP |
| `MarketRunnerPrices` | Per-runner price state: last traded price (LTP), SP near/far, traded volume |
| Price ladder tables (`AvailableToBack`, `AvailableToLay`, `BestAvailableToBack`, `BestAvailableToLay`, `Traded`, `StartingPriceBack`, `StartingPriceLay`, etc.) | Multi-level price/size ladders (level 1 = best price, up to 3 levels deep) |
| `RawMarketChanges` | Raw Betfair JSON deltas as received, with sequence numbers. Intended for exact replay of the stream |
| `ResolvedMarketSnaps` | Full resolved market state JSON at each tick, with sequence numbers. Each snap is the complete picture after applying all deltas |

### Data Volumes (approximate per day)

- ~30-60 race meetings per day
- ~200-400 individual markets (races)
- ~8-14 runners per race
- Updates every ~5 seconds per market, from market creation until settlement
- A busy day can produce 500k+ update rows and corresponding runner/price rows

### Access

Both databases are MySQL 8.0 running in Docker, exposed on `localhost:3307`.
Connection: `Server=localhost;Port=3307;UserID=root;Password=<from .env>`.

The raw Betfair stream data (`RawMarketChanges` + `ResolvedMarketSnaps`) is
the richest source — it contains everything Betfair sent, at full fidelity,
with sequence numbers for deterministic replay.

---

## 2. What Makes This Data Useful for RL

Each race is a natural **episode**:
- Clear start (market OPEN) and end (market CLOSED with WINNER)
- Observable state transitions: OPEN → SUSPENDED (at the off) → OPEN (in-play) → CLOSED
- Known outcome (winner selection ID in `marketResults`)
- Rich time-series of price movements leading up to and during the race

The price ladder is essentially a **limit order book** — the agent can observe
how odds move, where money is being placed, and how the market reacts to
information (e.g., going in-play).

Weather data provides environmental context that correlates with race outcomes
(going conditions, visibility, etc.).

Runner metadata (form, jockey, trainer, breeding, ratings) provides
fundamental features that the market may or may not have fully priced in.

---

## 3. Design Questions to Discuss

### 3a. Problem Formulation

- **What is the agent optimising for?** Options include:
  - Bet placement (when/what/how much to back or lay)
  - Market making (continuous back+lay to capture spread)
  - Value detection (identify mispriced runners, signal only)
  - Race outcome prediction (classification, not strictly RL)
- **Action space**: discrete (back runner X at current price, lay runner X, no-op) vs continuous (price, stake)?
- **Reward signal**: P&L per race? Sharpe ratio over a session? ROI?
- **Episode structure**: one episode per race, or one episode per trading day?

### 3b. State Representation

- **Observation at each timestep**: what goes into the state vector?
  - Current price ladder (back/lay prices and sizes at each level)
  - Price velocity / momentum (rate of change of LTP, volume)
  - Time until race start / time since going in-play
  - Market-level features (total matched volume, number of active runners)
  - Runner-level features (form, rating, jockey stats, weight)
  - Weather (temperature, precipitation, wind, going inference)
  - Cross-runner relative features (rank by price, price gaps)
- **Raw vs engineered features**: use the raw price ladder, or pre-compute derived features (implied probabilities, overround, spread)?
- **Sequence length**: how many historical ticks to include? Fixed window or variable?

### 3c. Data Pipeline

- **Source**: pull from `ResolvedMarketSnaps` (full state at each tick) or reconstruct from `RawMarketChanges` (deltas)?
- **Joining**: need to join HotData snapshots with ColdData (runner metadata, weather, results) to build complete training episodes
- **Labelling**: winner from `marketResults.WinnerSelectionId`
- **Train/test split**: by date (no lookahead), or by venue, or random?
- **Storage format for training**: Parquet? HDF5? SQLite? In-memory datasets?

### 3d. Technology Stack

- **Language**: Python (PyTorch / Stable-Baselines3 / RLlib) or stay in .NET?
- **Data extraction**: direct MySQL queries, or build an export pipeline that produces flat files?
- **Environment**: custom Gymnasium env that replays historical episodes?
- **Simulation**: the exchange is a real market — do we need a simulator for order fills, or treat it as a signal-only problem first?

### 3e. Project Structure

- Separate repo or monorepo with StreamRecorder1?
- How to keep the data pipeline in sync with schema changes?
- Do we need a feature store / intermediate representation between raw DB and training?

---

## 4. Available Betfair Price Ladder Fields

For reference, here is every price field captured per runner per tick:

| Field | Description |
|---|---|
| `LastTradedPrice` (LTP) | Most recent matched price |
| `TradedVolume` | Total volume matched on this runner |
| `StartingPriceNear` | BSP near-side indicator |
| `StartingPriceFar` | BSP far-side indicator |
| `AvailableToBack[0..2]` | Best 3 back prices + sizes (what you can back at) |
| `AvailableToLay[0..2]` | Best 3 lay prices + sizes (what you can lay at) |
| `BestAvailableToBack[0..2]` | Best available to back (virtual) |
| `BestAvailableToLay[0..2]` | Best available to lay (virtual) |
| `StartingPriceBack[]` | BSP back offers |
| `StartingPriceLay[]` | BSP lay offers |
| `Traded[]` | Recently traded price/size pairs |

Each price entry is a `(Level, Price, Size)` tuple. Level 1 = best price.

---

## 5. Runner Metadata Fields

From `RunnerMetaData` (populated once daily from Betfair catalogue):

| Field | RL Relevance |
|---|---|
| `FORM` | Recent race results string (e.g., "1234-21") |
| `OFFICIAL_RATING` | BHA handicap rating |
| `ADJUSTED_RATING` | Adjusted for conditions |
| `AGE` | Horse age |
| `SEX_TYPE` | Mare/Gelding/Colt/etc. |
| `WEIGHT_VALUE` / `WEIGHT_UNITS` | Weight carried |
| `JOCKEY_NAME` / `JOCKEY_CLAIM` | Jockey + claim allowance |
| `TRAINER_NAME` | Trainer |
| `STALL_DRAW` | Starting position (flat racing) |
| `DAYS_SINCE_LAST_RUN` | Freshness |
| `SIRE_NAME` / `DAM_NAME` / `DAMSIRE_NAME` | Breeding (distance/ground preferences) |
| `FORECASTPRICE_NUMERATOR` / `DENOMINATOR` | Morning forecast odds |
| `CLOTH_NUMBER` | Saddle cloth number |
| `WEARING` | Equipment (blinkers, visor, etc.) |

---

## 6. Key Timestamps in a Race Lifecycle

```
Market created (in Betfair catalogue)
  │
  ├── PRE_RACE weather fetch (StartTime - 30 min)
  │
  ├── Market OPEN — prices streaming, pre-race trading
  │
  ├── Market SUSPENDED — "at the off", race about to start
  │   └── AT_OFF weather fetch
  │
  ├── Market OPEN (inPlay=true) — in-play trading, prices volatile
  │
  ├── Market SUSPENDED — photo finish or stewards enquiry (sometimes)
  │
  └── Market CLOSED — result declared
      ├── Runner status = WINNER / LOSER / REMOVED
      ├── POST_RACE weather fetch
      └── MarketResult created in ColdData
```

---

## 7. Suggested Starting Prompt

Use this when opening a new conversation:

> I'm building a reinforcement learning system for Betfair horse racing
> exchange trading. I have a live data capture pipeline (StreamRecorder1)
> that records every price tick, runner metadata, race results, and weather
> for all GB WIN markets daily. The full data description is in
> `rl-project-brief.md` — please read it first.
>
> I'd like to discuss [specific topic from section 3], starting with
> [your priority].
