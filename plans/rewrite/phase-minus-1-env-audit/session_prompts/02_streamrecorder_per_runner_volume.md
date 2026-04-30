# Session prompt — restore per-runner traded-volume capture in StreamRecorder1

Use this prompt to open a new session in a fresh context, working
in **`C:\Users\jsmit\source\repos\StreamRecorder1`** (NOT the
`rl-betfair` repo where this prompt lives). The prompt is
self-contained — it briefs you on the problem, the evidence, the
diagnosis to date, and the constraints. Do not require any context
from the session that scaffolded it.

---

## The question

**Restore per-runner cumulative traded volume capture in the
production polling pipeline so downstream consumers
(`rl-betfair` simulator, supervised scorers, replay UI) can
reconstruct passive-fill mechanics from the data.**

## Why this matters

The `rl-betfair` Phase −1 env audit
(`C:\Users\jsmit\source\repos\rl-betfair\plans\rewrite\phase-minus-1-env-audit\audit_findings.md`,
finding F7) discovered that the simulator's documented
passive-fill mechanic is dormant in production. The cause is
upstream: every row of
`hotdatarefactored.polledmarketsnapshots.RunnersJson` has
`state.totalMatched = 0.0` for every runner, even on healthy
pre-race markets with £100k–£1M of market-level matched volume.

The simulator depends on this field to gate passive fills against
trade flow. Without it, paired-arb passives at unique tick-offset
prices fill on the very next tick regardless of whether actual
trading occurred. Every cohort training metric for the past N
months has been measured against this artificial fill model
rather than the spec'd Betfair behaviour.

This data IS available from Betfair — the operator confirmed
"we definitely used to get this data at some point". This session
verifies that, locates the regression, and chooses a fix.

## What you'll find when you start

There are **two pollers** in `StreamRecorder1\`:

1. **`StreamRecorder1\StreamRecorder1\`** — the original. Hits
   Betfair's public `listMarketBook` REST API with
   `priceProjection.priceData = ["SP_AVAILABLE", "SP_TRADED",
   "EX_TRADED", "EX_ALL_OFFERS"]`
   (`StreamRecorder1\StreamRecorder1\ApiInteractions\BettingApiInteractions.cs:295–315`).
   Writes to the `ResolvedMarketSnaps` table. The `EX_TRADED`
   request flag is the spec-faithful way to get per-runner
   cumulative traded volume — Betfair returns it on every
   `runner.totalMatched` field.
2. **`StreamRecorder1\BetfairPoller\`** — the new one (likely
   the production poller today). Hits the *internal* ero AJAX
   endpoint `https://ero.betfair.com/www/sports/exchange/readonly/v1/bymarket`
   with `types=MARKET_STATE,RUNNER_STATE,RUNNER_EXCHANGE_PRICES_BEST`
   (`StreamRecorder1\BetfairPoller\Polling\BetfairPollingClient.cs:62`).
   Writes to the `polledmarketsnapshots` table. The `types` enum
   it requests does NOT include any value that returns per-runner
   traded volume.

In the live `hotdatarefactored` MySQL DB:
- `polledmarketsnapshots` exists and is being populated.
- `ResolvedMarketSnaps` does **not** exist.

So the migration from the original poller to BetfairPoller
retired the table that was capturing the field, and the new
table's source endpoint doesn't expose the field. The data was
lost in the switchover.

## What to do

### 1. Verify the diagnosis (~15 min)

a. Confirm `BetfairPoller` is the production poller today (not
   `StreamRecorder1`). Check `Dockerfile`s, deployment configs,
   and any process-management scripts. If both are running,
   note that.

b. Confirm `polledmarketsnapshots.RunnersJson` rows have
   `state.totalMatched = 0` on every runner across recent rows.
   A representative spot-check (5 rows from the last 7 days,
   pre-race rows where the row's `TotalMatched` column is
   £10k+) is enough.

c. Check the git history of
   `BetfairPoller\Polling\BetfairPollingClient.cs` to find when
   the `types=...` enum was set, what it was set to before, and
   whether per-runner `totalMatched` ever WAS populated under
   the BetfairPoller path.

### 2. Establish what the ero endpoint can return (~30 min)

The ero AJAX endpoint is undocumented (it's the JSON the
betfair.com website calls internally), so this is a research
step. Two plausible outcomes:

**Outcome A — ero supports a `types` value that returns
per-runner traded volume.** Look for enum values like
`RUNNER_EXCHANGE_TRADED_VOLUME`, `EXCHANGE_TRADED`,
`RUNNER_TRADED`, etc. Check the original Betfair website's
network tab via browser dev-tools while watching a pre-race
horse market — the website itself displays per-runner volumes,
so it must be requesting them somehow. Capture the request,
note the `types=...` it sends.

If this works, the fix is **one line** in
`BetfairPollingClient.cs` line 62 — add the missing types value
to the existing list, and a corresponding field on
`BetfairPollingResponse.cs::RunnerStateNode` to deserialize it
(or update the existing `TotalMatched` mapping if it just
becomes populated).

**Outcome B — ero genuinely doesn't expose per-runner volume.**
The internal endpoint is best-effort and not a complete view of
the exchange. In that case, BetfairPoller can't get this signal
without changing data sources.

### 3. Pick a fix path (operator decision, but recommend one)

- **Path 1: Extend `types` enum on ero endpoint (if Outcome A
  applies).** One-line change, no extra API call cost. Spec-faithful.
  Production-deployable in one PR.
- **Path 2: Restore the original `StreamRecorder1` poller
  alongside `BetfairPoller`.** Two parallel pollers writing to
  two tables; downstream `rl-betfair` reads from
  `ResolvedMarketSnaps`. Pros: known to work; minimal code
  change. Cons: doubles API call cost, doubles MySQL write
  volume, runs two processes for what should be one.
- **Path 3: Add a parallel `listMarketBook` REST call inside
  `BetfairPoller` and merge the per-runner totalMatched into
  the polled snapshot before writing.** One process, but adds
  N API calls per poll cycle (one listMarketBook call per N
  active markets, batchable).
- **Path 4: Subscribe `BetfairPoller` to Betfair's Stream API
  for `tv` per-price arrays and sum at write time.** Largest
  change to BetfairPoller's architecture. Most data-rich (gets
  per-price granularity) but requires a Stream subscription.

Recommend **Path 1** if Outcome A is achievable, **Path 3**
otherwise. **Path 2** as a fallback if neither fits in one
session. **Path 4** is out of scope here — flag it as a
follow-on if the operator wants stream-grade data later.

### 4. Implement and verify (~60 min, depends on path)

Whatever path is chosen, the verification target is the same:

- A row in `polledmarketsnapshots` (or whichever table the new
  data lands in) where `RunnersJson[i].state.totalMatched > 0`
  on at least one active runner of a healthy pre-race market.
- Cross-check the value: sum of per-runner `totalMatched`
  should be roughly comparable to (and ≤) the row's
  market-level `TotalMatched` column (single-sided convention).
- Ideally, the per-runner figures climb monotonically across
  pre-race ticks (cumulative, not per-tick delta).

### 5. Hand-off to rl-betfair (~5 min)

Once the fix lands and a test poll captures non-zero per-runner
volumes, the `rl-betfair` side needs to reprocess affected
parquet files via `data/extractor.py` and re-run the F7
regression test:

```
cd C:\Users\jsmit\source\repos\rl-betfair
.venv\Scripts\python -m pytest tests/test_per_runner_total_matched_data.py -v
```

The test currently fails on the
`TestRealParquetPerRunnerTotalMatched` class. Once the fix
lands and parquets are reprocessed, all four tests must pass.
Operator triages the rl-betfair-side reprocess separately —
it's not your job to drive it from this session, just leave
clear instructions in your write-up.

### 6. Write up the result (~15 min)

A new file at
`StreamRecorder1\plans\f7-per-runner-volume\session_01_findings.md`
(or wherever StreamRecorder1's plan-folder convention places
session notes — adapt to whatever's already there) with:

- The diagnosis confirmed in step 1.
- The ero-endpoint research outcome (A or B).
- Which fix path was chosen and why.
- Code change summary (file path + line range, or PR link if
  one was opened).
- Verification evidence (one row's worth of
  `RunnersJson.state.totalMatched > 0` shown).
- Hand-off note to rl-betfair: which parquet days need
  reprocessing, and the F7 test to confirm.

## Hard constraints

- **Don't touch anything in `rl-betfair`.** This session works
  in `StreamRecorder1` only. The audit findings doc and F7
  regression test in rl-betfair are separate artefacts; they
  should not be modified from here. (Operator may run the F7
  test in a separate session to validate.)
- **Don't change the database schema** unless the chosen fix
  path requires it. Adding a column or table is fine if needed
  for Path 4; for Paths 1–3 the existing `RunnersJson` longtext
  is sufficient (just put the new field in the existing JSON).
- **Don't drop the `polledmarketsnapshots` table** even if you
  resurrect `ResolvedMarketSnaps`. The two coexisting is fine
  during transition; the operator decides retirement timing.
- **Don't break existing `BetfairPoller` behaviour.** The other
  fields it captures (LTP, ladders, market status, in-play
  flag) must keep working. Only ADD per-runner volume; don't
  refactor the rest.
- **Don't widen scope to "rewrite the poller from scratch".**
  Even if the code is awkward, this session is laser-focused on
  restoring per-runner volume.

## Out of scope

- Any change to `rl-betfair` (tests, simulator, parquet
  reprocess).
- Restructuring `StreamRecorder1` beyond the per-runner-volume
  capture itself.
- Per-price `tv` arrays (Path 4) — flag for follow-on if the
  fast win doesn't get there.
- Live trading client changes (`ai-betfair`).
- `coldData` schema or any non-`hotdatarefactored` work.

## Useful pointers

- **Production poller (broken):**
  `StreamRecorder1\BetfairPoller\Polling\BetfairPollingClient.cs:62`
  — the URL with `types=...` enum.
- **Production poller response map:**
  `StreamRecorder1\BetfairPoller\Polling\BetfairPollingResponse.cs:111–125`
  — `RunnerStateNode.TotalMatched` exists (so deserialization
  works); the field just never gets a non-zero value because
  the server doesn't include it under the requested `types`.
- **Production poller mapper:**
  `StreamRecorder1\BetfairPoller\Polling\PollingResponseMapper.cs:41`
  — `TotalMatched = market.State?.TotalMatched ?? 0` mapping;
  same issue as above.
- **Original poller (working pattern, retired):**
  `StreamRecorder1\StreamRecorder1\ApiInteractions\BettingApiInteractions.cs:295–315`
  — `listMarketBook` request with `EX_TRADED` in priceData.
- **Live MySQL DB:** `localhost:3306`, root creds in
  `C:\Users\jsmit\source\repos\rl-betfair\.env`. Tables of
  interest: `hotdatarefactored.polledmarketsnapshots`. Sample
  query for diagnosis:
  ```
  SELECT Id, MarketId, TotalMatched, RunnersJson
  FROM polledmarketsnapshots
  WHERE TotalMatched > 100000 AND InPlay = 0
  ORDER BY Id DESC LIMIT 3;
  ```
- **F7 regression test (read-only, hand-off target):**
  `C:\Users\jsmit\source\repos\rl-betfair\tests\test_per_runner_total_matched_data.py`.
  Two tests in `TestRealParquetPerRunnerTotalMatched` fail
  today and must pass after the fix + reprocess.
- **F7 root analysis:**
  `C:\Users\jsmit\source\repos\rl-betfair\plans\rewrite\phase-minus-1-env-audit\audit_findings.md`
  finding F7 — full diagnosis including the empirical
  proof-of-fill-on-zero-volume mini-simulation.

## Estimate

Single session, 1.5–3 hours.

- 15 min: confirm diagnosis (DB spot-check, git blame).
- 30 min: research ero endpoint capabilities (browser dev-tools
  on betfair.com network tab).
- 0–60 min: implement chosen path. Path 1 is ~10 min if
  Outcome A; Path 3 is ~60 min for HTTP client + merge code +
  rate-limit handling.
- 30 min: verify in DB.
- 15 min: write up.

If you find yourself heading toward 4+ hours, stop and write up
where you are. Don't try to land all four paths in one session.
The operator will sequence follow-ons.
