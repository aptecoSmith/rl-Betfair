---
id: 01KTFXZGMGGDQ7XMNNT0G69V7E
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-0c3388]
aliases: [canonicalise random ids, pair_id self-parity, golden harness]
---

# Golden harness: canonicalise opaque random ids (and profile the real path)

A bit-identity golden harness must **canonicalise opaque random identifiers** (uuids, object ids,
dict-iteration handles) to structural equivalence, or self-parity false-alarms on them.

## What it is

Self-parity (capture twice, same seed/weights/env) failed on run #1 — but only on `pair_id`
(`uuid.uuid4().hex[:12]`, random by design); every other quantity matched exactly. The pairing
*structure* (which bets share a pair, in order) is deterministic; only the string isn't. Fix: the
comparator canonicalises pair_ids to first-appearance group indices and compares the group sequence —
**not** by dropping pair_id from the compare (that would blind the gate to a real pairing-structure
regression). Two related profiling lessons from the same work: **cProfile would have lied** (it inflates
per-step wall ~50% and misses the batched collector's inline structure — per-phase `perf_counter`
timers on the *real* path showed the forward is kernel-launch-bound and `collector_other` is 39% of
rollout); and a "867 s/agent-train-day" figure was a mislabelled *cluster*-day wall (~100 s true
marginal) — always check what a "per-agent" number divides by before optimising against it.

## Why it matters

Build the golden harness first, and make it compare *structure* not opaque handles. Measure on the real
path, not a profiler that distorts it, and sanity-check the denominator of any per-unit cost.

## Sources
- `src-0c3388` lessons_learnt.md (js_desktop:present)
