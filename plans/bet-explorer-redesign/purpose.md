# Bet Explorer Redesign

## Problem

The current bet explorer is a flat table with 13 columns.  You can't tell at
a glance which race a bet belongs to without reading the venue+time columns,
and there's no way to see whether a bet was each-way or what the place terms
were.  The outcome column shows "won" or "lost" with no distinction between
a straight win and a place-only EW settlement.

## Goal

Redesign the bet explorer to match the grouped, collapsible layout from the
ai-betfair recommendations page:

```
Venue (e.g. "Cork") — bet count, total stake, P&L
  └─ Race (e.g. "12:15 — Cork Maiden Hurdle") — bet count, P&L
       └─ Bet card: runner, action, price, stake, outcome, P&L
            └─ EW legs (if each-way): win leg + place leg breakdown
```

## Key features

### 1. Grouped display (venue → race → bets)

- **Venue header**: collapsible, shows aggregate bet count + stake + P&L
- **Race header**: collapsible, shows time + race name + EW terms
  (e.g. "1/4 odds, 3 places") + bet count + P&L
- **Bet cards**: inline cards (not table rows) with colour-coded action
  badge (BACK blue / LAY pink)

### 2. EW terms in race header

For each-way markets, display: `EW: 1/4 odds, 3 places`
This comes from `each_way_divisor` and `number_of_places` on the race/bet
record (see ew-metadata-pipeline plan).

### 3. EW leg breakdown

When a bet is each-way, show expandable sub-rows:
```
Win leg   £5.00 @ 4.30   +£15.68
Place leg £5.00 @ 1.83   +£3.92
```
This mirrors the ai-betfair recommendations page's `ew-legs` component.

### 4. Settlement type badges

Replace binary "won"/"lost" with richer badges:
- **WON** (green) — straight win or EW winner (both legs paid)
- **PLACED** (amber/yellow) — EW place-only (place leg paid, win leg lost)
- **LOST** (red) — no payout

### 5. Filter: WIN / EW / BOTH

New filter dropdown (default: BOTH) that filters by `is_each_way`:
- **WIN**: only straight win bets (`is_each_way === false`)
- **EW**: only each-way bets (`is_each_way === true`)
- **BOTH**: all bets (default)

### 6. Retain existing filters

Keep: Date, Race, Runner (text search), Action (Back/Lay), Outcome (Won/Lost)
Add: Bet Type (WIN/EW/BOTH)

### 7. Summary stats bar

Keep existing: Total Bets, Bet Precision, P&L per Bet, Total P&L
Add: EW Bets count, Win Bets count

## Inspiration: ai-betfair recommendations page

Key patterns to adopt:
- `venue-group` → `venue-header` (click to toggle) → `race-slot` → `market-group` → `rec-card`
- Chevron toggles (▸/▾) for expand/collapse
- `ew-legs` sub-component with per-leg breakdown
- Aggregate P&L at venue and race level with green/red colouring
- Sidebar with latest races for quick navigation (optional for v1)

## Dependencies

- **ew-metadata-pipeline** plan must be implemented first — the API needs to
  return `is_each_way`, `each_way_divisor`, `number_of_places`,
  `settlement_type`, `effective_place_odds` per bet.

## Files to modify

### Backend (API)
- `api/routers/replay.py` — update bet explorer endpoint to include EW fields
- `api/models/` — update ExplorerBet response model

### Frontend
- `frontend/src/app/bet-explorer/bet-explorer.html` — full template rewrite
- `frontend/src/app/bet-explorer/bet-explorer.ts` — grouping logic, new
  computed signals for venue/race aggregation, expand/collapse state
- `frontend/src/app/bet-explorer/bet-explorer.scss` — new styles for cards,
  groups, badges, EW legs
- `frontend/src/app/models/bet-explorer.model.ts` — update ExplorerBet
  interface with EW fields

## Layout sketch

```
┌─────────────────────────────────────────────────────────────┐
│ Stats bar: Total Bets | Win Bets | EW Bets | Precision | P&L│
├─────────────────────────────────────────────────────────────┤
│ Filters: [Date ▼] [Action ▼] [Outcome ▼] [Type: BOTH ▼]   │
│          [Runner search...]                                  │
├─────────────────────────────────────────────────────────────┤
│ ▾ Cork — 8 bets, £240.00 staked, +£32.50                   │
│   ▾ 12:15 — Cork Maiden Hurdle — EW 1/4, 3 places — +£18  │
│     ┌──────────────────────────────────────────────────┐    │
│     │ BACK  Fancy Horse  £20.00 @ 5.50  PLACED  +£4.28│    │
│     │   Win leg   £10.00 @ 5.50  -£10.00              │    │
│     │   Place leg £10.00 @ 2.13  +£10.69              │    │
│     └──────────────────────────────────────────────────┘    │
│     ┌──────────────────────────────────────────────────┐    │
│     │ BACK  Dark Storm   £15.00 @ 3.20  WON    +£31.35│    │
│     │   Win leg   £7.50 @ 3.20   +£15.68              │    │
│     │   Place leg £7.50 @ 1.55   +£3.92               │    │
│     └──────────────────────────────────────────────────┘    │
│   ▸ 12:45 — Cork Handicap Chase — 3 bets — -£12.00        │
│ ▸ Pontefract — 5 bets, £180.00 staked, -£22.30            │
└─────────────────────────────────────────────────────────────┘
```
