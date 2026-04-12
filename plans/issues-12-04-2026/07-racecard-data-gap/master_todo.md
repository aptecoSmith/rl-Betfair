# Master TODO — RaceCard Data Gap

## Session 1: Diagnose + fix in StreamRecorder

### Diagnose

- [x] Check BetfairPoller container logs for race card fetch activity
- [x] Check if `RACE_CARD_FETCH_ENABLED` is effectively true
- [x] Check the fetch loop timing — narrow 30-min window confirmed
- [x] Check for HTTP errors — **Cloudflare 403 after 5-6 requests**
      (JA3 TLS fingerprinting, root cause of 97% data gap)
- [x] Check container restart frequency — HashSet resets compound the
      problem but aren't the primary cause

### Fix the Cloudflare 403

- [x] Identify root cause: Cloudflare JA3 fingerprinting blocks
      OpenSSL-based TLS stacks (HttpClient + regular curl in Docker)
- [x] Install curl-impersonate (Chrome variant) in Docker image
- [x] Rewrite RaceCardClient to use curl_chrome116 subprocess
- [x] Verify: 5/5 burst requests return HTTP 200 from inside Docker

### Fix the fetch window

- [x] Widen the fetch window to all markets today (24h, configurable)
- [x] Reduce the inter-fetch delay from 2-5s to 1-2s
- [x] Reduce outer loop sleep from 5min to 2min

### Fix persistence across restarts

- [x] Pre-populate fetchedRaceCards from DB on startup
- [x] Log count of already-fetched vs pending markets

### Backfill existing data

- [x] Backfill 123 missing markets via Python script (host-side)
- [x] Coverage: 134/134 markets (was 11/134)

### Re-extract training data

- [ ] After monitoring: re-extract affected dates from MySQL to parquet
- [ ] Verify re-extracted parquets have populated `past_races_json`
      and `timeform_comment` columns
- [ ] The 3 skipped tests in `test_integration_session_2_7b.py` should
      now pass instead of skipping

### Verify

- [ ] Monitor Apr 13 (first full day): RaceCardRunners coverage >90%
- [ ] Re-run: `python -m pytest tests/ --timeout=120 -q` — all green,
      session 2.7b tests no longer skip
- [ ] Commit in StreamRecorder1 repo
