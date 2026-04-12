# Session: Fix RaceCard Data Gap

Read `plans/issues-12-04-2026/07-racecard-data-gap/` before starting.
Follow `master_todo.md`. This is a cross-repo issue — changes are in
`C:\Users\jsmit\source\repos\StreamRecorder1\BetfairPoller\`, not
in rl-betfair.

## Context

The BetfairPoller has a `RaceCardClient` that fetches horse details
(form, past races, Timeform comments) from Betfair's public racing-info
API and persists them to `RaceCardRunners` in MySQL. The code works —
when called, it persists complete data. But it's only being called for
~2.5% of markets.

## Key files in StreamRecorder1

| File | Purpose |
|------|---------|
| `BetfairPoller/Program.cs:184-216` | Race card fetch loop |
| `BetfairPoller/Configuration.cs:21` | `RACE_CARD_FETCH_ENABLED` setting |
| `BetfairPoller/RaceCard/RaceCardClient.cs` | Fetch + persist logic |
| `docker-compose.yml:80-99` | Poller container config |

## The fetch loop (Program.cs:192-216)

```csharp
// Fetch race cards for markets starting within the next hour
_ = Task.Run(async () => {
    while (!ct.IsCancellationRequested) {
        await Task.Delay(TimeSpan.FromMinutes(5), ct);
        // Get upcoming markets...
        foreach (var market in upcomingMarkets) {
            if (fetchedRaceCards.Contains(market.MarketId)) continue;
            fetchedRaceCards.Add(market.MarketId);
            await raceCardClient.FetchAndPersistAsync(market.MarketId, dbFactory, ct);
            await Task.Delay(Random.Shared.Next(2000, 5000), ct);
        }
    }
});
```

## Probable issues

1. "Next hour" window is too narrow — markets outside this window are
   never fetched
2. 5-minute outer loop delay + 2-5s per market = slow throughput
3. In-memory HashSet resets on container restart
4. No logging of how many markets are pending vs fetched

## What to change

1. Widen window to "all markets today" or "next 6 hours"
2. Reduce delays — the racing-info API is lightweight
3. On startup, pre-populate HashSet from existing RaceCardData in DB
4. Add logging: markets found, already fetched, newly fetching
5. Write a backfill script for existing data

## Constraints

- Don't break the main polling loop — race card fetch runs as a
  background task
- Respect Betfair API rate limits — don't hammer it, but 1s between
  fetches should be fine
- The racing-info API may not serve data for past markets — test
  before attempting large backfills
- After fixing, re-extract affected days in rl-betfair and verify
  form features are populated

## Commit

In StreamRecorder1 repo:
`fix: widen race card fetch window + persist fetched set across restarts`
