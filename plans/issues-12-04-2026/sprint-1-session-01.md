# Sprint 1, Session 1: RaceCard Data Gap (Issue 07)

Read `CLAUDE.md` and `plans/issues-12-04-2026/07-racecard-data-gap/`
before starting. Follow `master_todo.md`. Mark items done as you go and
update `progress.md` at the end.

This is the highest-impact issue in the backlog. 97% of training data
is missing 24 form features per runner because the BetfairPoller only
captured RaceCardRunners for 18 out of 712 markets.

## Scope

This is a **cross-repo** session. Changes are in
`C:\Users\jsmit\source\repos\StreamRecorder1\BetfairPoller\`, not
rl-betfair.

1. Diagnose why the race card fetch loop only captures ~2.5% of markets
2. Fix the fetch window, timing, and restart resilience
3. Attempt a backfill of existing markets (API may not serve past data)
4. Re-extract affected dates in rl-betfair
5. Verify form features are populated in parquet files

## Key context

- The fetch loop is at `BetfairPoller/Program.cs:192-216`
- `RACE_CARD_FETCH_ENABLED` defaults to true, not overridden in docker-compose
- 18 markets captured, all with 100% data — the code works when called
- Probable issues: narrow 1-hour window, 2-5s delay, in-memory HashSet
  resets on container restart

See `plans/issues-12-04-2026/07-racecard-data-gap/session_prompt.md`
for full details.

## After this session

Run a training session to validate that form features are now populated.
Models trained after this fix will have access to all 24 form dimensions.
