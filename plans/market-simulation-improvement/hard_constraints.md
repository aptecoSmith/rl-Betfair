# Hard Constraints — Market Simulation Improvement

Rules that apply to every session of this plan. A PR that violates any
of these should be rejected in review, not debated.

1. **No ladder walking, anywhere, ever.** The single-price match rule
   in `ExchangeMatcher._match` is load-bearing; re-introducing
   ladder-walking caused the pre-`f7a09fc` phantom-profit bug. Any PR
   that adds a loop over levels inside `_match` or its callers is
   wrong by default.

2. **No drop of the LTP junk filter in the aggressive path.** The
   ±`max_price_deviation_pct` gate is what keeps £1–£1000 stale
   parked orders from filling. The `force_close=True` relaxation in
   `ExchangeMatcher` is the ONLY sanctioned exception, and it's
   env-initiated only. Do not extend the relaxation to agent-
   initiated paths.

3. **Passive crossability gate stays in force.** Phase 1 of
   `PassiveOrderBook.on_tick` must gate volume accumulation by a
   crossability check. Session 02 replaces the LTP-single-price
   gate with a per-price gate; it does NOT remove the gate entirely.

4. **Byte-identical default for any new knob.** If Session 02
   introduces a config toggle (e.g. `use_per_price_crossability:
   bool = False`), the default must reproduce pre-plan behaviour so
   in-flight training runs aren't disturbed. Flip the default in a
   follow-up commit only after validation.

5. **Regression test per behaviour change.** Every session that
   modifies `bet_manager.py` must land a unit test that pins the new
   behaviour — the 2026-04-22 crossability bug hid for months
   because no test locked the queue-depletion rule to crossable
   trades only. Do not repeat the pattern.

6. **MIN_BET_STAKE change requires a grep pass.** Session 03 must
   check every reference to the constant (including test fixtures
   that hard-code `stake=2.00`) and confirm none of them are
   load-bearing for a test that would silently pass on the new £1
   floor. Any test asserting a specific rejection at £1.50 needs
   to be re-examined.

7. **No touch to reward-shape terms.** Matured-arb bonus, MTM
   shaping, naked clip, early-pick, precision, efficiency — all
   off-limits in this plan. Those are governed by other plans
   (`arb-signal-cleanup`, `reward-densification`,
   `naked-clip-and-stability`). If a session's fix surfaces a
   reward-shape issue, write it up in `progress.md` and queue a
   follow-on plan; do not fix it inline.

8. **One smoke validation per session.** Before marking a session
   complete, run a 1-agent × 1-day training smoke on a known date
   and compare passive-fill counts / locked P&L / matured-arb
   counts against the pre-session baseline. Sessions have a known
   expected direction of shift (documented in their prompt); a
   shift in the unexpected direction is a failure even if training
   doesn't crash.

9. **Scoreboard-row comparability.** Runs started after a session
   lands are NOT byte-identical to pre-session runs if the session
   changed fill behaviour. Record a line in `progress.md` stating
   which scoreboard columns are comparable across the boundary and
   which are not, so the operator doesn't false-compare.

10. **Spec first, code second.** Any new simulator behaviour added
    in this plan must be reflected in
    `docs/betfair_market_model.md` — either under §4 (simulator
    mapping) if it's a faithful implementation of a documented
    Betfair rule, or under §5 (approximations) if it's a deliberate
    simplification. The doc and the code must not diverge.
