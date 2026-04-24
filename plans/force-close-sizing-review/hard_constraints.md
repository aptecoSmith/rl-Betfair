---
plan: force-close-sizing-review
status: draft
---

# Hard constraints — force-close-sizing-review

## §1 — Keep the "strictly better per pair" invariant

Whichever option is picked, the per-pair outcome must remain: at
T−N, a force-close's P&L is within ±£spread_cost of zero, not
within ±£100s of directional variance. Option 3 (fractional
sizing) is the only one that can relax this per-pair; it does so
explicitly by leaving (1−k) stake naked, and the operator must
consent to that tradeoff in writing before coding.

## §2 — No changes to the relaxed-matcher path without testing

`env/exchange_matcher.py::ExchangeMatcher::match_back/match_lay`
with `force_close=True` currently drops the LTP requirement, skips
the ±50% junk filter, and keeps the hard `max_back_price` cap. Any
option that touches the matcher (Option 1 cap tightening, Option 3
fractional) must keep the single-level no-walking invariant
(CLAUDE.md "Order matching: single-price, no walking") and must
add a test that exercises the changed behaviour on a real ladder.

## §3 — Agent-initiated closes remain unchanged

`close_signal`-triggered closes go through the strict match
(`force_close=False`). This plan does not touch that path. If
Option 2 (time-phased escalation) wants to SHAPE close_signal
usage inside the soft-close window, it does so via reward
accounting, not by changing the matcher.

## §4 — Scoreboard comparability notes

Any option that changes the reward magnitude (Options 1, 2, 3, 4
all do in some way) requires:
- A JSONL-row field `force_close_sizing_option` naming the active
  variant.
- A note in the plan's `lessons_learnt.md` under "Scoreboard
  comparability" stating which metrics are still comparable to
  pre-change rows.
- Updates to CLAUDE.md's "Force-close at T−N" subsection.

Option 5 (do nothing) is the only one that keeps scoreboard
rows comparable.

## §5 — Overdraft semantics preserved unless Option changes them

CLAUDE.md "Overdraft allowed for force-close" documents that the
per-race budget gate is bypassed on force-close. This stays unless
Option 3 (fractional) explicitly requires a different budget
semantic, in which case the change has to pass through plan
review.

## §6 — Count accounting

`arbs_force_closed`, `arbs_closed`, `arbs_naked`, and
`arbs_completed` (settlement counters in `RaceStats`) keep their
current definitions. Options 3 and 4 must define precisely which
bucket a partial or skipped force-close lands in before coding
begins.

- Option 3 fractional close: the matched portion counts as
  `arbs_force_closed`; the residual naked portion counts as
  `arbs_naked` — NOT `arbs_force_closed_partial` or similar new
  bucket.
- Option 4 budget cap: pairs past the cap settle naked and count
  as `arbs_naked`. The count of skips must be exposed on the info
  dict (e.g. `force_close_refused_budget`) parallel to the
  existing `force_close_refused_no_book` / `_place` / `_above_cap`.

## §7 — Do not shape force-close success

Force-closes are excluded from matured-arb bonus (`n_matured =
completed + closed`, NOT `+ force_closed`) and excluded from the
`+£1 per close_signal success` bonus per
`plans/arb-signal-cleanup/hard_constraints.md §7, §14`. Do NOT
add a bonus term for force-close success. Option 2's "time-phased
escalation" can shape `close_signal` usage during the pre-force-
close window, but shaping a successful force-close inverts the
incentive the whole mechanism exists to preserve.

## §8 — The review is a write-up first

Session 01's deliverable is a `design_review.md` that
quantifies each option's per-race aggregate cost on a replay-probe
run, NOT a prototype implementation. The operator picks the path;
implementation is session 02+.

Rationale: cohort W's gen-1 data makes it easy to over-fit a
design to one observation. A replay-probe run + an explicit
options tradeoff matrix resists that.

## §9 — Do not touch PPO trainer

This plan is env/reward-side only. If an option requires the PPO
trainer to know about force-close events (it shouldn't), the
option is rejected. Use the info-dict surface.

## §10 — Worker safety

Same probe-interference rules as `ppo-kl-fix`: don't run pytest or
pytest-x while the `arb-signal-cleanup-probe` worker is live. Any
replay-probe runs this plan spawns must use a separate registry
sandbox.

## §11 — Don't ship before PPO fix lands

The force-close mechanism's performance under a fully-trained
policy is different from its performance under a BC-pretrained-
plus-stuck policy (which is what cohort W's gen-1 measurement
gave us). Session 02+ implementation should not ship until the
`ppo-kl-fix` plan has landed and at least one re-measurement
exists under trained-policy conditions. Otherwise we're sizing
the fix to a confounded observation.
