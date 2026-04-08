# Not Doing — Research-Driven

Items considered and deliberately parked. Each entry records what
was parked, why, and the concrete trigger that would justify
promoting it. The point is to stop the same idea being re-proposed
in a future session without anyone remembering it was already
weighed.

Promotion is allowed — but only against the recorded trigger, and
only with a fresh `design_decisions.md` entry recording the
reversal. "I changed my mind" is not a trigger; "the eval metric
showed X" is.

---

## ND-1 — `modify` action (cancel-and-replace at one price)

**What it would be:** A new action verb that atomically moves a
resting order to a different price, mapping to Betfair's
`replaceOrders` API.

**Why parked:** Cancel + new place expresses the same outcome with
one fewer action dimension to explore. Real Betfair `replaceOrders`
doesn't preserve queue position anyway, so the simulator would
have to re-snapshot queue-ahead either way. Adding this would be
pure action-space bloat. See `design_decisions.md` 2026-04-07
entry.

**Trigger to promote:** A measured live-latency cost in
`ai-betfair` deployment metrics that is materially worse for
cancel + place vs `replaceOrders`. The trigger is a real number,
not a hunch — file the measurement before re-opening this.

---

## ND-2 — End-to-end learning of pressure features (no hand-engineering)

**What it would be:** Skipping P1's hand-engineered features and
relying on the policy network to learn order-book imbalance,
weighted microprice, and traded direction directly from raw ladder
rows.

**Why parked:** Three reasons in `analysis.md` §3 — sample budget
too small, neural nets bad at the operations these features need,
and learned latents are not inspectable in eval logs. End-to-end
learning would burn a sample budget we don't have.

**Trigger to promote:** A future architecture (likely a transformer
with proper attention over the ladder) that demonstrably learns
equivalent signals from raw inputs *during arch-exploration
follow-up work*, beating the P1 baseline on the same eval window
without using the engineered columns. At that point P1's columns
become redundant and can be trimmed.

---

## ND-3 — Calibrating the queue estimator against the live order stream

**What it would be:** Building a feedback loop that compares the
simulator's queue estimator against logged live order events, and
tuning the estimator to match.

**Why parked:** The simulator estimator is a deliberate
approximation. Live uses the real order stream and doesn't need
calibration. The only reason to calibrate the estimator would be
if it was actively misleading the policy in a way that surfaces
in deployed behaviour. Pre-emptive calibration is speculative work.

**Trigger to promote:** A documented incident where a
P3/P4-trained policy behaves systematically differently in live
than in sim, and the diff is traced to queue-fill timing rather
than to one of the other usual suspects (latency, action mapping,
state reconciliation). At that point — and only then — calibration
becomes a non-speculative response to a measured problem.

---

## ND-4 — Modelling latency in the simulator

**What it would be:** Adding a fixed-tick or fixed-millisecond
delay to bet placement so the policy is trained against something
resembling real network + matching latency.

**Why parked:** Plausible-and-real, but we don't currently have
data on what the live latency distribution looks like, so any
number we picked would be a guess. P3/P4 are also fine without it
— passive orders are *deliberately* not in a hurry, and aggressive
orders' latency is dominated by what the matcher does on the next
tick anyway. Adding latency now would be design-by-fear.

**Trigger to promote:** A measured live-latency distribution from
`ai-betfair` order-stream logs once the phantom-fill bug is fixed.
At that point we have a real number to feed in. (Also recorded as
Q5 in `open_questions.md`.)

---

## ND-5 — Per-runner separate policies

**What it would be:** Training one policy per runner slot instead
of a shared one. Came up briefly when discussing whether the
weighted-microprice feature should be shared or per-slot.

**Why parked:** Slot identity is meaningless across races (slot 3
is a different horse every race), so per-slot policies would have
nothing to specialise on. The shared policy with per-slot inputs
is the correct architecture and is already what the codebase does.
Recording this so it doesn't get re-proposed during P1's feature
work.

**Trigger to promote:** None. This is parked permanently. If the
data model changes such that slot identity gains meaning across
races, revisit — but that would be a much larger change than this
folder is concerned with.

---

When adding a new entry, pick the next free ND-number. Do not
reuse numbers from closed/promoted entries — if an item is
promoted, leave its ND entry in place with a "promoted on YYYY-MM-DD
to session NN" line at the bottom, so the audit trail is preserved.
