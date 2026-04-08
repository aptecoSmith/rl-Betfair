# Design Decisions — Research-Driven

Load-bearing decisions made in this folder, with rationale. Append
one entry per decision. Each entry answers three questions:

1. **What** was decided
2. **Why** (the tradeoff — what we gave up)
3. **When to revisit** (concrete trigger, not "someday")

This file is not a changelog. It is the place a future session looks
when about to re-litigate a decision, to check whether the original
reasoning still applies. If it still applies, don't re-litigate; if
it no longer applies, record the reversal as a *new* entry.

For frozen context from earlier phases, read
`arch-exploration/lessons_learnt.md` and
`next_steps/design_decisions.md` — several of those entries are
load-bearing here too.

---

## 2026-04-07 — Cancel is bundled with passive orders

**What:** The "cancel unmatched bet" action is treated as a
co-requisite of the passive-order regime, not as a separate
proposal. P3 in `proposals.md` is named "Passive orders + cancel
action (bundled)". Neither half can be promoted into a session on
its own.

**Why:** Cancel without a resting state is a no-op — the matcher's
`unmatched_stake` is conceptually cancelled the moment a bet is
placed today. Resting state without a cancel action is a trap — the
agent commits liquidity it cannot withdraw if the market moves
against it. The research material's three-way decision
(join / cross / cancel) only makes sense as a unit. Splitting them
across sessions would mean either shipping a no-op action that
confuses exploration, or shipping a feature that the policy can't
back out of. Tradeoff: the bundled session is larger and harder to
review than two smaller ones, accepted in exchange for not shipping
either half in a broken state.

**When to revisit:** If, during P3 implementation, it turns out the
cancel half is genuinely separable (e.g. an ergonomic helper that
doesn't change the action space), reconsider the split. Otherwise
this stays bundled.

---

## 2026-04-07 — `modify` action deliberately not added

**What:** No `modify` (cancel-and-replace at new price) action will
be added to the action space. Price moves are expressed as cancel +
new place.

**Why:** Real Betfair has a `replaceOrders` API call, so the
real-world precedent exists. But adding it to the simulator action
space costs an extra dimension and an extra exploration burden for a
behaviour that is already expressible. The agent learning to chain
cancel + place gets the same outcome with one less knob. Tradeoff:
slightly worse latency in live (two API calls vs one); accepted
because the simulator does not model API call latency anyway and
because real Betfair `replaceOrders` doesn't preserve queue
position, so the simulator would have to re-snapshot queue-ahead
either way.

**When to revisit:** If the live latency cost of cancel + place
becomes measurable and material in `ai-betfair` deployment metrics,
add `modify` to both repos. The trigger is a real number, not a
hunch.

---

## 2026-04-07 — Hand-engineered features over end-to-end learning

**What:** P1 adds explicit feature columns (OBI, weighted
microprice, traded delta, mid drift) to the observation vector
rather than letting the policy learn equivalents from raw ladder
rows.

**Why:** Three reasons (full version in `analysis.md` §3): sample
budget is small (~9 days of data), neural nets are bad at the
operations these features need (division, deltas, weighted
reductions over variable-length lists), and hand-computed values
are inspectable in eval logs. End-to-end learning of the same
signals would burn a sample budget we don't have. Tradeoff:
features must be kept in sync between `rl-betfair` and `ai-betfair`
(see `hard_constraints.md` #12), and any future architecture with
genuine end-to-end ambition will overlap with these features
redundantly.

**When to revisit:** If a future architecture (transformer with
proper attention over the ladder, e.g.) demonstrably learns
equivalent signals from raw ladders during arch-exploration follow-up
work, the hand-engineered features become redundant and can be
trimmed. Trigger: the architecture beats the P1 baseline on the
same eval window without using the engineered columns.

---

## 2026-04-07 — Live uses real order stream, not the queue estimator

**What:** The queue-position estimator added in P4 lives only in
the simulator. `ai-betfair` live inference reads matched/remaining
sizes directly from the Betfair order stream and never falls back
to the simulator's estimator.

**Why:** The estimator is a coarse approximation built from
historical parquet data because the parquet doesn't tell us how the
real queue evolved. Live, the exchange tells us the truth — and
the truth is always preferable when available. Tradeoff: divergence
between sim and live behaviour around queue dynamics is expected
and acceptable. The simulator is for *training*, not for *modelling
physical reality*; we accept the cost of a mismatch on this axis.

**When to revisit:** Never on the live side — using the real stream
when it exists is unconditionally correct. On the sim side, only
revisit if the estimator turns out to be so coarse that it gives
the policy systematically wrong signals (e.g. agents trained to
"trust" the estimator's fill timing, then surprised in live). In
that case, calibrate the estimator against logged live order events
rather than replacing live with the estimator.

---

## 2026-04-07 — Phantom-fill fix in `ai-betfair` is a prerequisite, not a co-task

**What:** The phantom-fill bug audited in `downstream_knockon.md` §0
is treated as a hard prerequisite for any research-driven session
that ships a new policy. It is *not* allowed to be folded into a
P1/P2 session as "while we're at it".

**Why:** The bug is in a different repo, has a different reviewer
context, and touches the live order-stream subscription, which is
the most consequential code path in `ai-betfair`. Bundling it into
a sim-side session would dilute the review and risk it being
shipped under-tested. Tradeoff: serialising the work lengthens
calendar time before any new features land in production; accepted
because the alternative is shipping new features into a runtime
that fabricates state.

**When to revisit:** Once the phantom-fill fix has been in
production for at least one full eval window (~9 days) without
state-drift incidents, this constraint is satisfied permanently and
the prerequisite is just history. No further revisit needed.

---

## 2026-04-07 — Phantom-fill gate is on deployment, not on training-side work

**What:** Refinement of the previous entry. The phantom-fill
prerequisite gates **deployment of a new policy into live
`ai-betfair`**, not the *training-side work* in this repo. Sessions
in `master_todo.md` Phase 1 and Phase 2 may be merged to master and
run training while the cross-repo fix is still in flight. Only the
hand-over of a new policy for live trading is blocked.

**Why:** The original framing in `master_todo.md` Phase 0 read as
"no item ships at all until ai-betfair is fixed", which would
serialise two streams of work that are perfectly capable of running
in parallel. The training-side improvements (P1, P2, etc.) and the
live-wrapper fix are part of the same family — "the
simulator/wrapper has been too optimistic about fills" — and they
reinforce each other. Sequencing them strictly was overreach
masquerading as caution. The operator caught this on review and was
right to push back. Tradeoff: a small risk that someone forgets the
deployment gate when the time comes to ship a policy. Mitigation:
the gate is recorded in three places now (`hard_constraints.md` #8,
`master_todo.md` Phase 0, this entry).

**When to revisit:** If a research-driven session ever produces a
training-side change that *cannot* be tested in isolation from the
live wrapper (e.g. a feature that depends on live-only data the
parquet replay can't synthesise), then the gate has to move
upstream and this entry needs revising. Not foreseen, but flagged.

---

## 2026-04-07 — R-2 self-depletion is a sim-side bug, not a research-driven feature

**What:** The matcher's failure to subtract its own previously-
matched volume from visible liquidity (`bugs.md` R-2) is treated as
an **independent sim bug**, not as a sub-item of any P1–P5
proposal. It can ship at any time, in or out of a research-driven
session, and does not gate or get gated by the rest of this
folder's work.

**Why:** R-2 is mechanically independent of P1–P5 — a fix to it
doesn't depend on hand-engineered features, spread cost shaping,
passive orders, queue position, or UI annotations. It's a
correctness bug in the existing aggressive-fill code path. The
only weak coupling is that P3/P4 (passive orders) would make R-2
*worse* by stretching the window over which the agent can
accumulate stale-liquidity fills, so R-2 should land *before* P3
to avoid building new code on a buggy foundation. Tradeoff:
filing R-2 here rather than in `next_steps/bugs.md` couples it
visually to research-driven work even though it isn't research-
driven; accepted because the operator surfaced it in this context
and it's easier for the operator's mental model to keep related
context together.

**When to revisit:** If the next_steps/ phase decides to pull R-2
into its own bugs list (as a B-prefix entry), this folder's R-2
should become a one-line cross-reference and the canonical entry
moves to `next_steps/bugs.md`. The two should not exist in
parallel — bugs have one home.
