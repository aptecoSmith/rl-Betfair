# Hard Constraints

- Changes are in the StreamRecorder1 repo, not rl-betfair.
- Don't break the main polling loop — race card fetching is a
  background task and must not starve the primary market snapshot
  capture.
- Respect Betfair API rate limits — the racing-info API is public
  but don't abuse it. 1s between fetches minimum.
- The racing-info API may not serve data for past markets — verify
  before attempting backfills.
- After fixing: re-extract affected dates and verify form features
  are populated in the parquet files.
