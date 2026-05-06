---
session: phase-13-directional-scalping / S01
phase: rewrite/phase-13-directional-scalping
parent_purpose: ../purpose.md
---

# S01 — Feature audit: what does the policy see at decision time?

## Context

Read `purpose.md` and `hard_constraints.md` first. This session is
**diagnostic, not implementation**. The deliverable is a markdown
findings document that other sessions will reference. No code
changes, no new modules, no tests. Output goes in
`plans/rewrite/phase-13-directional-scalping/findings.md`.

The motivating question: a human scalper says "this price is too far
out, it'll come in" and acts on that. Does the policy see the inputs
that justify or contradict that read? If yes, the alpha pathway is a
training-signal problem (S02 / S03 / S05 fix it). If no, this plan is
incomplete and a follow-on feature-extension plan is needed before
S03 can possibly work.

## What to do

1. Read [env/betfair_env.py:256-346](../../../../env/betfair_env.py)
   — the `MARKET_KEYS`, `MARKET_VELOCITY_KEYS`, `RUNNER_KEYS` lists.
   These are the static observation features.

2. Read [env/features.py](../../../../env/features.py) end-to-end.
   Understand how each feature is computed from raw ladder data.

3. Read [env/betfair_env.py:1162-1200](../../../../env/betfair_env.py)
   — the `_features_to_array` and `_get_obs` methods. Understand
   how `(market, velocity, runner) × (agent_state, position)` are
   concatenated into the policy's observation.

4. Read [agents_v2/discrete_policy.py](../../../../agents_v2/
   discrete_policy.py) — `DiscreteLSTMPolicy.forward`. Trace the
   flow `obs → encoder → LSTM → fill_prob_head + mature_prob_head
   + actor_head`. Identify what the actor sees AT the moment it
   chooses an action.

5. Tabulate feature presence vs the human scalper's mental model.
   For each of the categories below, list:
   - Features PRESENT in the obs vector that map onto this
     category, with the exact key.
   - Features ABSENT from the obs vector that arguably should be
     there.

   Categories (the human scalper's read):

   a. **Static price level.** The current LTP and recent
      ladder geometry. ("Where is the price right now?")

   b. **Recent direction.** Has the price been moving up or down,
      and how fast? ("Has it been trending?")

   c. **Order book pressure.** Imbalance between back-side and
      lay-side queue, weight of money, microprice. ("Where is
      the next move likely to come from?")

   d. **Trade flow / aggression.** Volume traded recently,
      buy-vs-sell pressure, traded-volume ladder per price.
      ("Is real money moving the price?")

   e. **Market structure.** Time-to-off, race status (parading,
      going down, etc.), each-way terms, total matched.
      ("What kind of market is this and how long do I have?")

   f. **Cross-runner context.** Is this runner drifting while
      another shortens? Where is the field's money flowing?
      ("Is the move idiosyncratic or part of a wider shift?")

   g. **My own position.** What pairs do I have open, what is
      their MTM, when did I open them? ("What's my book look
      like?")

6. For each ABSENT feature you flag in steps a–g, give a one-
   sentence justification of why it would matter for direction
   prediction. Prioritise ones that are **already in the data
   pipeline** (e.g. the memory note flags TradedVolumeLadder as
   captured-but-unused) since those are cheap follow-ons. Mark
   features that would require new data ingestion as out-of-
   scope-for-this-plan.

7. Compute a confidence read on the alpha hypothesis:

   - **Strong signal already there** — if categories a, b, c, d
     all have multiple obs features and the gaps are minor (e.g.
     "longer-window pressure, but 3/5/10-tick is already in").
     The policy has the inputs to predict direction; S02 / S03 /
     S05 should work. No feature-extension follow-on needed.

   - **Partial signal** — if categories a, b, c, e are well-
     represented but d (trade flow) and f (cross-runner) are
     thin. S03 may still work but would benefit from a feature-
     extension follow-on plan. Flag the specific 2–3 features
     to add and recommend a phase-14-feature-extension plan as
     parallel work.

   - **Insufficient signal** — if categories b, c, d are all
     thin or absent. S03 cannot succeed; this plan should pause
     until features land. Recommend reordering: feature-extension
     plan FIRST, then resume Phase 13 from S02.

## Deliverable

A new file:
`plans/rewrite/phase-13-directional-scalping/findings.md`

Structure:

```markdown
---
plan: rewrite/phase-13-directional-scalping
parent_purpose: ./purpose.md
session: S01
landed: <YYYY-MM-DD>
---

# S01 findings — feature audit

## Summary

(One paragraph: which confidence read above, and the key gap if any.)

## Per-category presence/absence table

| Category | Present (obs key) | Absent (justified addition) | Action |
|---|---|---|---|
| a. Static price level | ltp, back_price_1..3, lay_price_1..3, ... | — | none |
| b. Recent direction | ltp_velocity_3/5/10, ... | longer-window (e.g. 30-tick velocity) | optional follow-on |
| c. Order book pressure | obi_topN, weighted_microprice, ... | (fill in) | (fill in) |
| d. Trade flow | traded_delta, vol_delta_3/5/10 | TradedVolumeLadder per-price (memory note) | follow-on cheap |
| e. Market structure | time_to_off_seconds, race_status_*, ... | — | none |
| f. Cross-runner | ltp_rank, gap_to_favourite, vol_rank | trade-flow rank, money-flow asymmetry | follow-on |
| g. Position | has_open_arb, passive_fill_proximity, seconds_since_passive_placed | — | none |

## Direction-prediction hypothesis

(Three short paragraphs, one per confidence read. Pick one and explain.)

## Recommended follow-on (if any)

(Either: "none — proceed to S02" / "feature-extension plan as parallel
work, list specific features" / "feature-extension plan FIRST, pause
this plan".)

## Notes for S02

(Anything S02 should know — e.g. "the existing `mid_drift` feature
is already a near-direction signal at 1-tick horizon; the new label
should NOT use that as an obs but rather as a soft-baseline against
which to evaluate whether the new head adds anything").
```

## Stop conditions

- **Stop and ask** if categories b, c, AND d are all judged thin.
  That changes the plan's ordering and the operator should decide
  whether to defer Phase 13.

- **Stop and ask** if you find that some currently-listed obs
  features are computed but DROPPED before reaching the policy
  (e.g. computed in `engineer_tick` but not in `RUNNER_KEYS`). Such
  features are zero-cost wins — flag the discovery and propose
  adding them to `RUNNER_KEYS` in a one-line follow-on PR rather
  than a new plan.

- **Do not** edit `RUNNER_KEYS` or any feature module in this
  session. The audit's job is to document the state, not change
  it. Edits go in S02 if narrow (and only if `OBS_SCHEMA_VERSION`
  doesn't need to bump) or in a follow-on plan otherwise.

## Done when

- `plans/rewrite/phase-13-directional-scalping/findings.md` exists
  with the structure above filled out from real reads of the code.
- The "confidence read" paragraph names one of the three
  conclusions explicitly.
- `lessons_learnt.md` has a one-paragraph entry for S01 summarising
  the conclusion + any unexpected discoveries.
- Commit: `docs(rewrite): phase-13 S01 - feature audit findings`.
- No code files modified.
