# Master TODO — RaceCard Data Gap

## Session 1: Diagnose + fix in StreamRecorder

### Diagnose

- [ ] Check BetfairPoller container logs for race card fetch activity:
      `docker logs streamrecorder1-betfair-poller-1 | grep -i "race card"`
- [ ] Check if `RACE_CARD_FETCH_ENABLED` is effectively true in the
      running container
- [ ] Check the fetch loop timing — how many markets does it attempt
      per cycle? Is the "next hour" window catching markets?
- [ ] Check for HTTP errors — is the Betfair racing-info API returning
      errors or rate-limiting?
- [ ] Check container restart frequency — if the poller restarts often,
      the in-memory HashSet loses track of what's been fetched

### Fix the fetch window

- [ ] Widen the fetch window from "next hour" to "next 3 hours" or
      "all markets today" — race cards don't change, so fetching early
      is fine
- [ ] Reduce the inter-fetch delay from 2-5s to 1-2s — the Betfair
      racing-info API is public and lightweight
- [ ] Consider fetching race cards on a separate timer from the main
      polling loop — currently it's a background Task.Run but may be
      starved

### Fix persistence across restarts

- [ ] On startup, query existing RaceCardData.MarketId from DB and
      pre-populate `fetchedRaceCards` HashSet — prevents refetching
      but ensures new markets are caught
- [ ] Log the count: "Race cards already fetched: N, markets pending: M"

### Backfill existing data

- [ ] Write a one-off backfill script that fetches race cards for all
      702 markets in PolledMarketSnapshots that don't have RaceCardData
- [ ] Note: Betfair's racing-info API may not serve data for past
      markets — test with a recent market first. If it only works for
      upcoming markets, we can only backfill recent ones and ensure
      going-forward capture is complete

### Re-extract training data

- [ ] After backfill: re-extract affected dates from MySQL to parquet
      via the admin UI or `POST /api/admin/import-day`
- [ ] Verify the re-extracted parquets have populated `past_races_json`
      and `timeform_comment` columns
- [ ] The 3 skipped tests in `test_integration_session_2_7b.py` should
      now pass instead of skipping

### Verify

- [ ] Monitor over 1-2 race days: RaceCardRunners coverage should be
      >90% of PolledMarketSnapshots markets
- [ ] Re-run: `python -m pytest tests/ --timeout=120 -q` — all green,
      session 2.7b tests no longer skip
