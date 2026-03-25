# StreamRecorder1 Database Schema

Source: EF Core model classes in `StreamRecorder1/StreamRecorder1/Models/`.
Both databases run in Docker, MySQL 8.0, exposed on `localhost:3307`.

Connection string: `Server=localhost;Port=3307;Database=<db>;UserID=root;Password=<from .env>`

---

## ColdData database (`coldData`)

Populated once daily from the Betfair catalogue API. Reference / metadata.

---

### `Todays_venues`

One row per venue running races today.

| Column | Type | Notes |
|---|---|---|
| `id` | int (PK) | Auto-increment |
| `venue` | string | Venue name, e.g. "Newmarket" |
| `date` | DateTime | Date of the meeting |

---

### `Todays_markets` (EF: `MarketCatalogue`)

One row per market (race). Full Betfair catalogue entry.

| Column | Type | Notes |
|---|---|---|
| `MarketId` | string (PK) | Betfair market ID, e.g. "1.234567890" |
| `MarketName` | string | e.g. "2m Hcap" |
| `IsMarketDataDelayed` | bool | |
| `Description` | → MarketDescription | Nested (separate table via EF) |
| `Runners` | → RunnerDescription[] | Nested list |
| `EventType` | → EventType | Nested |
| `Event` | → Event | Nested |
| `Competition` | → Competition | Nested |

---

### `marketOnDates`

Flat summary of each market — easier to query than the full catalogue.

| Column | Type | Notes |
|---|---|---|
| `Id` | int (PK) | Auto-increment |
| `MarketId` | string | FK to Todays_markets |
| `venueName` | string | Venue name |
| `StartTime` | DateTime? | Scheduled start time |
| `MarketName` | string | Race name |
| `Runners` | int | Number of runners |

---

### `runnerdescription`

One row per runner per market.

| Column | Type | Notes |
|---|---|---|
| `Id` | int (PK) | Auto-increment |
| `SelectionId` | long | Betfair selection ID — the runner's permanent ID |
| `RunnerName` | string | Horse name |
| `Handicap` | double | Handicap value |
| `SortPriority` | int | Display order in race card |
| `Metadata` | → RunnerMetaData | Nested (separate table) |

---

### `RunnerMetaData`

Extended runner information. All fields stored as `string?` — parse to numeric
as needed. Joined to `runnerdescription` via EF navigation.

| Column | Notes |
|---|---|
| `Id` | int (PK) |
| `SIRE_NAME` | Sire (father) name |
| `DAM_NAME` | Dam (mother) name |
| `DAMSIRE_NAME` | Damsire (maternal grandfather) name |
| `SIRE_YEAR_BORN` | |
| `DAM_YEAR_BORN` | |
| `DAMSIRE_YEAR_BORN` | |
| `SIRE_BRED` | Country bred |
| `DAM_BRED` | |
| `DAMSIRE_BRED` | |
| `BRED` | Horse's country of breeding |
| `OFFICIAL_RATING` | BHA handicap rating (numeric string) |
| `ADJUSTED_RATING` | Adjusted for conditions (numeric string) |
| `AGE` | Horse age in years (numeric string) |
| `SEX_TYPE` | Mare / Gelding / Colt / Filly / etc. |
| `COLOUR_TYPE` | Bay / Grey / Chestnut / etc. |
| `WEIGHT_VALUE` | Weight carried (numeric string) |
| `WEIGHT_UNITS` | Units, e.g. "LB" |
| `JOCKEY_NAME` | Jockey full name |
| `JOCKEY_CLAIM` | Claim allowance in lbs (numeric string, "0" if none) |
| `TRAINER_NAME` | Trainer full name |
| `OWNER_NAME` | Owner name |
| `STALL_DRAW` | Starting stall number (numeric string; blank for jumps) |
| `CLOTH_NUMBER` | Saddle cloth number (numeric string) |
| `CLOTH_NUMBER_ALPHA` | Alpha variant of cloth number |
| `FORM` | Recent form string, e.g. "1234-21" (most recent rightmost) |
| `DAYS_SINCE_LAST_RUN` | Days since previous race (numeric string) |
| `WEARING` | Equipment worn, e.g. "Blinkers", "Visor", "" |
| `FORECASTPRICE_NUMERATOR` | Morning forecast odds numerator (numeric string) |
| `FORECASTPRICE_DENOMINATOR` | Morning forecast odds denominator (numeric string) |
| `COLOURS_DESCRIPTION` | Jockey colours description |
| `COLOURS_FILENAME` | Jockey colours image filename |
| `runnerId` | Internal runner ID string |

**Derived forecast price:** `FORECASTPRICE_NUMERATOR / FORECASTPRICE_DENOMINATOR + 1`
(convert fractional odds to decimal).

---

### `marketResults`

One row per settled market. Written when StreamRecorder1 detects a WINNER.

| Column | Type | Notes |
|---|---|---|
| `MarketResultId` | int (PK) | Auto-increment |
| `MarketId` | string | **Join key** — links to all other tables |
| `MarketName` | string | Race name |
| `MarketStartTime` | DateTime | Scheduled start time |
| `MarketType` | string | e.g. "WIN" |
| `MarketStatus` | string | Should be "CLOSED" |
| `EventId` | string | Betfair event ID |
| `EventName` | string | |
| `CountryCode` | string | e.g. "GB" |
| `Venue` | string | Venue name |
| `CompetitionId` | string | |
| `CompetitionName` | string | |
| `Timezone` | string | |
| `WinnerSelectionId` | int | **The winner** — FK to `runnerdescription.SelectionId` |

Unique index on `(MarketId, WinnerSelectionId)` — prevents duplicate winner rows.

---

### `VenueLocations`

Static seed data. ~60 GB racecourses with GPS coordinates.

| Column | Type | Notes |
|---|---|---|
| `Id` | int (PK) | |
| `VenueName` | string | e.g. "Newmarket" — unique |
| `Latitude` | double | WGS84 |
| `Longitude` | double | WGS84 |

Used to fetch weather from Open-Meteo API for each race.

---

### `WeatherObservations`

Three observations per market: PRE_RACE (30 min before off), AT_OFF (race
start), POST_RACE (result declared).

| Column | Type | Notes |
|---|---|---|
| `Id` | int (PK) | |
| `MarketId` | string | **Join key** |
| `VenueName` | string | |
| `Latitude` | double | |
| `Longitude` | double | |
| `ObservationTime` | DateTime | UTC time of the observation point |
| `ObservationType` | string | `"PRE_RACE"`, `"AT_OFF"`, or `"POST_RACE"` |
| `Temperature` | double? | °C |
| `Precipitation` | double? | mm |
| `WindSpeed` | double? | m/s |
| `WindDirection` | double? | degrees (0=N, 90=E, 180=S, 270=W) |
| `Humidity` | double? | % |
| `WeatherCode` | int? | WMO code (0=clear, 61=rain, 71=snow, etc.) |
| `RawJson` | string? | Full Open-Meteo API response |
| `FetchedAt` | DateTime | When the fetch was made |

Unique index on `(MarketId, ObservationType)`.

**For RL use:** join on `MarketId` and filter `ObservationType = 'PRE_RACE'`
to get the conditions known before the race starts.

---

---

## HotData database (`hotDataRefactored`)

Updated continuously during the day. High-velocity streaming data.

---

### `updates`

One row per market snapshot (every ~5 seconds per market).
The top-level tick record — everything else hangs off this.

| Column | Type | Notes |
|---|---|---|
| `UpdateId` | int (PK) | Auto-increment |
| `time` | DateTime | Timestamp of this snapshot |
| `MarketId` | string | **Join key** |
| `Venue` | string | |
| `MarketName` | string | |
| `MarketStartTime` | DateTime? | Scheduled start |
| `MarketType` | string | e.g. "WIN" |
| `MarketStatus` | string | `"OPEN"`, `"SUSPENDED"`, `"CLOSED"` |
| `EventId` | string | |
| `EventType` | string | |
| `EventTypeId` | string | "7" = horse racing |
| `EventName` | string | |
| `CompetitionId` | string | |
| `CompetitionName` | string | |
| `CountryCode` | string | |
| `NumberOfWinners` | int? | Usually 1 |
| `NumberOfRunners` | int? | Total runners including removed |
| `NumberOfActiveRunners` | int? | Runners still in the race |
| `TradedVolume` | double | Total volume matched on this market |
| `TotalMatched` | string | (string — parse to double) |
| `CrossMatching` | bool? | |
| `InPlay` | bool? | `true` after the off |
| `BetDelay` | int? | Seconds of delay (in-play only) |
| `BspMarket` | bool? | BSP available |
| `Complete` | bool? | Market fully settled |
| `OpenDate` | DateTime? | When market went live |

Unique index on `(MarketId, time)`.

**For RL use:** filter `InPlay = false` to get pre-race ticks only.
Order by `time` to replay chronologically.

---

### `marketRunnerSnapAtTimes`

One row per runner per snapshot. Foreign key to `updates.UpdateId`.

| Column | Type | Notes |
|---|---|---|
| `MarketRunnerSnapAtTimeId` | int (PK) | Auto-increment |
| `Timestamp` | DateTime | |
| `RunnerName` | string | Horse name |
| `MarketId` | string | |
| `SelectionId` | long | **Join key to ColdData runner tables** |
| `HandicapId` | double? | |
| `UpdateId` | int | FK to `updates` |
| `RunnerId` | → RunnerId | Nested (SelectionId + Handicap) |
| `Definition` | → RunnerDefinition | Nested — see below |
| `MarketRunnerPrices` | → Prices | Nested — see below |

Index on `UpdateId`.

---

### `RunnerDefinition` (nested in marketRunnerSnapAtTimes)

Runner status at this snapshot point.

| Column | Type | Notes |
|---|---|---|
| `RunnerDefinitionId` | int (PK) | Auto-increment |
| `SortPriority` | int? | Race card position |
| `RemovalDate` | DateTime? | Set if runner was removed before the race |
| `Id` | long? | SelectionId (redundant with parent) |
| `Handicap` | double? | |
| `AdjustmentFactor` | double? | Reduction factor if a runner was removed |
| `Bsp` | double? | Final BSP (set after race) |
| `Status` | string | `"ACTIVE"`, `"WINNER"`, `"LOSER"`, `"REMOVED"`, `"PLACED"` |

---

### `MarketRunnerPrices` (EF: `CustomFinancials.Prices`)

Price state for one runner at one snapshot. FK to `marketRunnerSnapAtTimes`.

| Column | Type | Notes |
|---|---|---|
| `MarketRunnerPricesid` | int (PK) | Auto-increment |
| `LastTradedPrice` | double | Most recent matched price (LTP) |
| `StartingPriceNear` | double | BSP near-side indicator |
| `StartingPriceFar` | double | BSP far-side indicator |
| `TradedVolume` | double | Total volume on this runner |
| `MarketRunnerSnapAtTimeId` | int | FK |

Plus the following child ladder tables, each containing up to 3 rows
(Level 1 = best price, Level 3 = third best):

| Child table | Direction | Notes |
|---|---|---|
| `AvailableToBack` | Back side | What you can back at right now |
| `AvailableToLay` | Lay side | What you can lay at right now |
| `BestAvailableToBack` | Back | Virtual/consolidated best back |
| `BestAvailableToLay` | Lay | Virtual/consolidated best lay |
| `BestDisplayAvailableToBack` | Back | Display prices (may differ from best) |
| `BestDisplayAvailableToLay` | Lay | Display prices |
| `Traded` | Both | Recently traded price/size pairs |
| `StartingPriceBack` | Back | BSP back offers |
| `StartingPriceLay` | Lay | BSP lay offers |

Each ladder table row (`PriceSize`):

| Column | Type | Notes |
|---|---|---|
| `PriceSizeID` | int (PK) | |
| `Level` | int | 1 = best, 2 = second best, 3 = third best |
| `Price` | double? | Odds (e.g. 4.5) |
| `Size` | double? | Volume available at this price (£) |

**For bet matching simulation:** use `AvailableToLay` to fill a back bet
(lay side is the counterparty). Use `AvailableToBack` to fill a lay bet.

---

### `RawMarketChanges`

Raw Betfair streaming JSON deltas, exactly as received. For exact replay.

| Column | Type | Notes |
|---|---|---|
| `Id` | long (PK) | |
| `MarketId` | string | |
| `Timestamp` | DateTime | |
| `SequenceNumber` | long | Betfair sequence number — order by this for replay |
| `IsImage` | bool | `true` = full snapshot, `false` = delta |
| `RawJson` | longtext | Raw Betfair delta JSON |

Indexes on `(MarketId, SequenceNumber)` and `(MarketId, Timestamp)`.

---

### `ResolvedMarketSnaps`

**Primary RL training source.** Complete resolved market state after each
delta is applied. One row per tick per market.

| Column | Type | Notes |
|---|---|---|
| `Id` | long (PK) | |
| `MarketId` | string | **Join key** |
| `Timestamp` | DateTime | UTC time of this snapshot |
| `SequenceNumber` | long | Order by this for deterministic replay |
| `SnapJson` | longtext | Full market state JSON (all runners, all ladders) |

Indexes on `(MarketId, SequenceNumber)` and `(MarketId, Timestamp)`.

The `SnapJson` contains the full `MarketSnap` object — all runners, all price
ladders, all fields — at this point in time. The data extractor parses this
JSON and flattens it into the Parquet episode files.

---

## Key Join Pattern for RL Extraction

```sql
-- All pre-race ticks for a given day, with results
SELECT
    u.MarketId,
    u.time             AS Timestamp,
    u.Venue,
    u.MarketStartTime,
    u.NumberOfActiveRunners,
    u.TradedVolume,
    u.InPlay,
    rms.SequenceNumber,
    rms.SnapJson,      -- parse this for full order book
    mr.WinnerSelectionId,
    wo.Temperature,
    wo.Precipitation,
    wo.WindSpeed,
    wo.WindDirection,
    wo.Humidity,
    wo.WeatherCode
FROM updates u
JOIN ResolvedMarketSnaps rms
    ON rms.MarketId = u.MarketId
    AND rms.Timestamp = u.time   -- or nearest tick by sequence
LEFT JOIN marketResults mr
    ON mr.MarketId = u.MarketId
LEFT JOIN WeatherObservations wo
    ON wo.MarketId = u.MarketId
    AND wo.ObservationType = 'PRE_RACE'
WHERE u.InPlay = false
  AND DATE(u.MarketStartTime) = '2026-03-26'
ORDER BY u.MarketId, rms.SequenceNumber;
```

Runner metadata is joined separately via `SelectionId`:
```sql
-- Runner metadata for a market
SELECT rd.SelectionId, rd.RunnerName, rd.SortPriority, rd.Handicap,
       rm.*
FROM runnerdescription rd
JOIN RunnerMetaData rm ON rm.Id = rd.Id   -- EF navigation FK
WHERE rd.MarketCatalogueMarketId = '1.234567890';
```

---

## Notes for the Data Extractor

- `ResolvedMarketSnaps.SnapJson` is the richest source — parse it for the
  full order book rather than joining the ladder tables individually.
- The ladder tables (`AvailableToBack`, etc.) exist for direct queries but
  joining all of them per tick is expensive. Use `SnapJson` instead.
- `WinnerSelectionId` in `marketResults` is typed as `int` but Betfair
  selection IDs are `long` — cast carefully.
- `RunnerMetaData` fields are all `string?` — parse numerics (rating, weight,
  stall draw, etc.) in the feature engineer, not the extractor.
- Weather nulls are possible if the Open-Meteo fetch failed — handle gracefully.
- `DAYS_SINCE_LAST_RUN = ""` is common for first-time runners — treat as NaN.
