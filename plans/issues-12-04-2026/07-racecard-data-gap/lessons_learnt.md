# Lessons Learnt — RaceCard Data Gap

## From investigation

- The RaceCardClient code is correct — when called, it persists
  complete data with PastRacesJson and TimeformComment for all runners.
  The 18 markets that were captured have 100% data coverage.

- The problem is throughput. The fetch loop runs every 5 minutes,
  checks for markets starting within the next hour, and takes 2-5s
  per market. On a busy day with 80+ markets, this can only capture
  a fraction before markets start and leave the window.

- The in-memory `fetchedRaceCards` HashSet is a restart vulnerability.
  The poller container likely restarts periodically (docker restart
  policy: `on-failure`), losing the set and potentially refetching
  the same early markets while missing later ones.

- 97% of training data is currently missing 24 form features per
  runner. The model has been training without horse form, course
  form, going form, BSP trends, and improving/declining form signals.
  Fixing this and re-extracting could meaningfully improve model
  quality.
