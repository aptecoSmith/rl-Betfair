# Hard constraints

Cross-session invariants. The probe is operator-driven (not
autonomous), but the constraints below are inviolable. If
progress would require violating any of them, stop and surface.

## 1. Value-edge formula correctness is load-bearing

(Corrected 2026-05-25 during Phase 2 implementation — the
initial scaffold drafted the back form with commission applied
to the stake, not just the net winnings, and the lay form with
the win-side multiplied by `(P-1)`. Both were off by small
amounts; the form below is the canonical one used by
`env.scalping_math.value_bet_edge`.)

The two formulas:

```python
# Back: stake S at price P, hold to settle.
#   Win pays S × (P − 1) × (1 − c)  (commission on net winnings only)
#   Lose pays −S
# EV per £1 stake:
edge_back = pwin * (price - 1) * (1 - c) - (1 - pwin)

# Lay: stake S at price P, liability S × (P − 1), hold to settle.
#   Win (runner LOSES) pays S × (1 − c)
#   Lose (runner WINS) pays −S × (P − 1)
# EV per £1 stake (NOT per £1 liability):
edge_lay = (1 - pwin) * (1 - c) - pwin * (price - 1)
```

Note `edge_lay` is per £1 STAKE, not per £1 liability. Callers
using fixed-liability sizing should still gate on this edge —
the conversion (multiply by `1 / (P − 1)`) is just a positive
scalar, so the SIGN of the edge is preserved and the
+0.05-threshold gate fires identically.

Phase 2 MUST land a unit test exercising both formulas. Tests
in `tests/test_value_bet_edge.py`:

| Case | pwin | price | side | c | edge | Threshold 0.05 |
|---|---|---|---|---|---|---|
| Reference back, near zero | 0.50 | 2.10 | back | 0.05 | +0.0225 | REFUSE |
| Reference back, above | 0.50 | 2.30 | back | 0.05 | +0.1175 | ACCEPT |
| Lay short favourite | 0.80 | 1.50 | lay | 0.05 | −0.21 | REFUSE |
| Lay outsider | 0.05 | 15.0 | lay | 0.05 | +0.2025 | ACCEPT |
| c=0 collapses (back) | 0.50 | 2.00 | back | 0.0 | 0.0 | REFUSE |
| c=0 collapses (lay) | 0.50 | 2.00 | lay | 0.0 | 0.0 | REFUSE |

If the gate refuses or accepts the wrong cases, the entire
probe is invalid.

## 2. Default-off byte-identical for every new knob

When all new kwargs are at their default values — i.e.

- `--strategy-mode` defaults to `arb` (the existing default)
- `value_edge_threshold = 0.0` (gate refuses nothing)
- `directional_back_stake = None` (no override)
- `directional_lay_liability = None` (no override)

— env stepping, action processing, and reward streams must be
**bit-for-bit identical** to pre-plan behaviour on scalping
cohorts. Existing scalping-cohort regression tests must
continue to pass without modification.

## 3. Audit launch flag wiring before launching the probe

Per `memory/feedback_audit_launch_wiring.md`: the
`--strategy-mode` flag has both an env-side consumer (action
interpretation) AND a policy-side consumer (action mask, if
applicable). After Phase 2 lands, grep `strategy_mode` across
`training_v2/cohort/worker.py`, `agents_v2/discrete_policy.py`,
and `env/betfair_env.py` to confirm the flag reaches every
named consumer. Write a regression test in the
`tests/test_v2_*.py` family that asserts the resolution path
is OR-semantics (not `.get(key, default)`).

A pre-flight smoke (Phase 3) that observes `gate_refusals > 0`
when `value_edge_threshold = 0.05` is the operational
sign-off — if the gate doesn't refuse anything, it isn't
wired in.

## 4. Strategy mode is cohort-wide, NOT per-agent

`--strategy-mode` is set once at cohort launch and pinned for
every agent in that cohort. Not a gene. Not mutated. Mixing
arb and value_win agents in one GA population is incoherent
(reward shape differs, fitness comparison meaningless).

## 5. No GA, no BC, no shaping additions in this probe

The probe tests whether the gate + pwin signal + sizing is
sufficient to extract per-bet EV. Adding GA mutation, BC
warmup, or new shaping terms confounds the question.

- 5 agents fresh-init, no GA generations beyond 1
- No `--bc-pretrain-steps`
- `value_win` mode's reward zeroing of early-pick / precision /
  efficiency / drawdown / spread-cost / inactivity is the
  intended state — DO NOT add shaping back to compensate. The
  whole point is to test the raw settle signal.

The probe is allowed exactly **one** parameter: the
`value_edge_threshold`. Pinned at 0.05 per probe; if the verdict
is "close but no cigar", the follow-on plan sweeps the
threshold — not this one.

## 6. Pre-registered verdict criteria

The four PASS/FAIL thresholds in `README.md::Success bar` are
PRE-REGISTERED. After Phase 4 / 5 results come in, you read the
numbers ONCE and apply the criteria. No "the EV is +£0.30
which is close to £0.50, let's call it a soft pass and tune
the threshold to push it over". A miss is a miss.

The pwin-calibration circuit-breaker (last row of the success
table) is independent: a calibration failure in the admitted
set INVALIDATES the per-bet EV number even if it's positive,
because the gate is being applied to a region where the
predictor is unreliable. Treat that case as "no signal yet —
back to the predictor" not as a PASS.

## 7. force_close has no role in this plan

Directional bets hold to settle by construction. Probe
configuration does NOT pass `force_close_before_off_seconds`.
If `value_win` mode has a leftover force-close path that fires
anyway, treat that as a bug — `value_win` is supposed to be
"place and hold"; force-closing at T-N would convert the bet
into a directional scalp and confound the probe.

Phase 3 smoke MUST verify: no bet in the smoke-day output has
a `force_close=True` flag.

## 8. Per-bet logging is required

Per `memory/feedback_per_bet_logging.md`. The probe rollout
MUST write `registry/<TAG>/bet_logs/<agent_id>.jsonl` per agent
per day. Schema MUST include `runner_champion_p_win` at the
moment of bet placement and the final `outcome` /
`final_pnl` — the per-bet EV and Sharpe calculations read
from this file, not from any aggregate.

Schema (minimum):

```
agent_id, day, bet_id, market_id, selection_id,
side ("back"/"lay"), price_matched, stake_matched, liability,
runner_champion_p_win, race_max_pwin, tick_time_to_off_s,
value_edge_at_placement,
final_outcome ("settle_win"/"settle_lose"),
final_pnl
```

The post-probe analysis script reads these files and computes:

- mean(final_pnl) — per-bet EV
- std(final_pnl) — per-bet σ
- mean / std — per-bet Sharpe
- predicted-vs-realised win-rate by `runner_champion_p_win`
  decile (the calibration circuit-breaker)

## 9. Phase commit hygiene

One commit per phase:

- Phase 0: `plan(non-scalping-directional-probe): scaffold`
- Phase 1: `findings(non-scalping-directional-probe): value_win sanity smoke`
- Phase 2: `feat(non-scalping-directional-probe): --strategy-mode CLI + value-bet gate + sizing override`
- Phase 3: `findings(non-scalping-directional-probe): pre-flight smoke verdict`
- Phase 4: `findings(non-scalping-directional-probe): probe A (back) results`
- Phase 5: `findings(non-scalping-directional-probe): probe B (lay) results`
- Phase 6: `findings(non-scalping-directional-probe): verdict`

Local commits only; do not push.

## 10. Stop conditions

The probe stops and writes a stop-condition entry to
`autonomous_run_log.md` (create if needed) on any of:

1. Phase 1 sanity smoke reveals `value_win` mode is broken
   (env crashes, bets never place, reward NaN). Surface;
   this plan blocks on fixing the codepath first.
2. Phase 2 wiring audit (§3) finds a silent-swallow bug like
   the one in `memory/feedback_audit_launch_wiring.md`. Fix
   and re-audit before launching the probe.
3. Phase 3 pre-flight smoke shows `gate_refusals == 0`
   despite `value_edge_threshold > 0` — gate isn't wired.
4. Phase 3 smoke shows `force_close=True` bets in value_win
   mode — codepath bug (§7).
5. pwin calibration circuit-breaker (§6) trips on either
   probe.

A stop is a real outcome, not a failure mode to work around.
Surface the issue and pause.
