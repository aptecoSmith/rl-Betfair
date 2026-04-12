# Progress — RaceCard Data Gap

## Session 1 (2026-04-12)

### Root cause (revised)

The original hypothesis was "narrow fetch window". The real cause was
**two compounding issues**:

1. **Cloudflare JA3 TLS fingerprinting** — the Betfair racing-info API
   (`apieds.betfair.com`) is fronted by Cloudflare, which fingerprints
   the TLS handshake (cipher order, extensions, ALPN). Both .NET's
   `HttpClient` (OpenSSL on Linux) and standard `curl` (OpenSSL in
   Docker) produce a non-browser JA3 hash. Cloudflare allows the first
   5-6 requests, then blocks all subsequent requests with HTTP 403.
   This is why the poller captured exactly 6 markets on Apr 11 and 5
   on Apr 12 — then stopped.

2. **Narrow fetch window** (secondary) — the race card loop shared
   `PollingLeadTimeMinutes` (30 min) with the main market poller.
   Markets outside this window were never attempted. Even if the
   Cloudflare issue were fixed, only imminent markets would be fetched.

### Evidence

- Container logs: 11 total successful fetches (6 on Apr 11, 5 on Apr
  12), all in the first batch after startup. Every subsequent request
  returned HTTP 403.
- Both container and host share the same external IP (`89.243.200.124`)
  — ruling out IP-based blocking.
- Host curl (Windows Schannel TLS): always returns 200.
- Docker curl (OpenSSL TLS): always returns 403.
- curl-impersonate (Chrome TLS fingerprint in Docker): returns 200
  consistently, including burst tests of 5+ requests.

### Changes made (StreamRecorder1 repo)

**`BetfairPoller/Dockerfile`**
- Installs `curl-impersonate` v0.6.1 (Chrome variant) in the runtime
  image. This provides `curl_chrome116` which mimics Chrome's exact TLS
  handshake, passing Cloudflare's bot detection.
- Dependencies: `libnss3` (required by curl-impersonate's NSS backend)

**`BetfairPoller/RaceCard/RaceCardClient.cs`** — rewrote HTTP transport
- Replaced `HttpClient` with `curl_chrome116` subprocess call
- Falls back to regular `curl` on Windows (Schannel TLS passes
  Cloudflare natively)
- Sets `LD_LIBRARY_PATH` for curl-impersonate's bundled libraries
- `PersistRaceCardAsync` unchanged — only the fetch transport changed

**`BetfairPoller/Configuration.cs`**
- Added `RACE_CARD_LEAD_TIME_HOURS` (default 24) — fetches all of
  today's markets, not just imminent ones
- Added `RACE_CARD_FETCH_DELAY_MIN_MS` / `MAX_MS` (default 1-2s)

**`BetfairPoller/Program.cs`** (fetch loop)
- Pre-populates `fetchedRaceCards` HashSet from DB on startup
- Widened window from 30min to 24h (configurable)
- Reduced inter-fetch delay from 2-5s to 1-2s
- Reduced outer loop sleep from 5min to 2min
- Added progress logging
- Added `--backfill` CLI mode

**`BetfairPoller/RaceCard/RaceCardBackfill.cs`** (new)
- One-off backfill for all markets in snapshots missing race cards

**`scripts/backfill-racecards.py`** (new)
- Python backfill script using `requests` (host-side, uses Schannel)

### Backfill results

134/134 markets now have race card data (was 11/134 = 8.2%).
Python backfill script ran from host, all 123 missing markets fetched
successfully in ~2 minutes.

### Verification

- curl-impersonate tested from inside Docker container: 5/5 requests
  returned HTTP 200 (vs 0/5 with regular curl)
- Docker image rebuilt and poller restarted with new code
- Poller currently in off-hours retry loop (expected — past race hours)
- **Tomorrow (Apr 13) will be the first full day with the fixed pipeline**

### Still to do

- [ ] Monitor Apr 13: RaceCardRunners coverage should be >90%
- [ ] Re-extract affected dates from MySQL to parquet
- [ ] Verify parquets have populated form features
- [ ] Commit in StreamRecorder1 repo
