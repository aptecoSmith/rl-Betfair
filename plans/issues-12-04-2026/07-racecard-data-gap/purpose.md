# 07 — RaceCard Data Gap

## What

The BetfairPoller's race card fetch is capturing only ~2.5% of markets
(18 out of 712 since March 31). This means ~97% of training data has
no horse form features — the model trains with all-NaN for 24 form
dimensions per runner.

Investigate why the fetch rate is so low and fix it. This is a
StreamRecorder/BetfairPoller issue, not rl-betfair code.

## Why

- 24 features per runner (7 recent form + 17 past race history) are
  computed from RaceCardRunners data. Without it, the model can't
  learn from horse form, course form, going form, BSP trends, or
  improving/declining performance.
- The code is correct — `RaceCardClient.FetchAndPersistAsync` works
  and persists data when called. The problem is it's not being called
  for most markets.

## Evidence

```
hotDataRefactored.RaceCardRunners:    224 rows, 18 markets
hotDataRefactored.PolledMarketSnapshots: 712 markets (same period)
Coverage: 2.5%

RaceCardData.FetchedAt dates: 2026-03-31 through 2026-04-10
  ~1-2 markets per day captured
  All have full data (PastRacesJson + TimeformComment populated)
```

## Root cause hypothesis

`BetfairPoller/Program.cs:192-216` — the race card fetch loop:
1. Fetches cards for "markets starting within the next hour"
2. Runs with 2-5 second random delay between fetches
3. Uses an in-memory `HashSet<string>` to skip already-fetched markets

Likely issues:
- The "next hour" window is too narrow — by the time the loop gets
  to a market, it may have already started
- The 2-5s delay limits throughput to ~20 markets/minute — not enough
  if there are 50+ markets per hour on busy days
- The loop may be competing with the main polling loop for resources
- The `fetchedRaceCards` HashSet resets on container restart — if the
  poller restarts frequently, it may refetch the same few markets
  instead of progressing through new ones
