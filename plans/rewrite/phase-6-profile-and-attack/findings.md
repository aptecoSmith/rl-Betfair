---
plan: rewrite/phase-6-profile-and-attack
opened: 2026-05-03
---

# Phase 6 — profile-and-attack: cumulative findings

## Per-session ms/tick table

Single-day, 5-episode CPU rollout on 2026-04-23 from
`data/processed_amber_v2_window/`, `--seed 42`. Median ms/tick
across the 5 episodes is the headline number; this is the
measurement-protocol fix vs Phase 4's 1-episode point estimates
(see `purpose.md` §"Candidate optimisations" item H).

The pre-Phase-6 baseline is the median across 5 episodes from
`logs/discrete_ppo_v2/run_cpu_post_sync_fix.jsonl` (the same file
used as Phase 4's pre-baseline; Phase 4's S01–S07 were measured
1-episode and didn't move the median, so this is the right
baseline for Phase 6 too):

| Episode | ms/tick |
|---|---|
| 1 | 10.003 |
| 2 | 9.595 |
| 3 | 8.456 |
| 4 | 9.197 |
| 5 | 10.558 |
| **median** | **9.595** |

Per-session post-change rows are 5-episode runs written to
`logs/discrete_ppo_v2/phase6_sNN_post.jsonl`.

| Session | ms/tick CPU (5-ep median) | Δ vs prev | Parity regime | Tests added |
|---|---|---|---|---|
| Pre-Phase-6 baseline | 9.595 | — | — | — |
| + S01 (profile + assess) | TBD (no code change) | — | n/a (assessment-only) | 0 |
| v1 reference (`ppo_lstm_v1`) | 2.94 | — | — | — |

## Phase 4 lessons inherited

Phase 6's premise rests on Phase 4's PARTIAL verdict. Three causes
identified post-mortem (full thread in `plans/rewrite/phase-4-
restore-speed/findings.md`):

1. **Single-episode measurement is too noisy to detect per-session
   wins.** Episode-to-episode variance was 8.0–10.7 ms/tick on
   identical code (S01's 5-ep table); per-session targets averaged
   ~0.3–1.0 ms predicted, well below the noise floor of a 1-ep
   sample. Phase 6 fixes this with a 5-episode median per session.
2. **Per-session targets were each individually small.** Each
   Phase 4 session correctly removed real overhead — but each
   target turned out to be tens of ms across 12 k ticks on a
   115 s episode wall. Three orders of magnitude below where the
   real cost lives. Phase 6 attacks costs sized in single-digit
   ms/tick (1.4–2.8 ms/tick for batched scorer, 30–50 % of forward
   for `torch.compile`), large enough to clear the noise floor.
3. **The phase was scoped from a code-read of `rollout.py`, not a
   profile.** None of the targeted call sites turned out to be
   dominant. Phase 6's Session 01 fixes this directly: profile
   first, decide targets from the data.

The work Phase 4 landed remains correct and shipped — bit-identity
is preserved end-to-end, including CUDA self-parity post-S07. The
structural improvements (incremental attribution, pre-allocated
buffers, disabled distribution validation, sampled invariant
assert, RolloutBatch namedtuple, masked_fill mask path) survive
into Phase 6 unchanged.

## Session 01 — profile and assess

**Status: NOT YET RUN.** Session prompt at
`session_prompts/01_profile_and_assess.md`.

This row populated when the session lands. Expected fields:

- Per-subsystem share of wall time (env_shim / env step / policy
  forward / rollout collector + PPO update).
- Top 5 hot frames (function, file:line, % wall).
- Estimated upper-bound speedup from each candidate optimisation
  on the menu in `purpose.md` §"Candidate optimisations" given
  the observed costs.
- Ranked target list for Session 02+.
- Verdict: GREEN (single dominant cost identified) /
  DISTRIBUTED (multi-target) / SURPRISING (profile contradicts
  expectations).

## Session 02+ — TBD

Written *after* Session 01 lands. The candidate menu in
`purpose.md` is the expected shape; the actual target order and
content depends on Session 01's assessment. Likely candidates
(from purpose.md §"Candidate optimisations"):

- **A.** Batched LightGBM `booster.predict()` (Regime A,
  bit-identical, ~1.4–2.8 ms/tick estimated)
- **B.** Slot-to-tick-runner-index cache (Regime A, 100–500 µs/tick)
- **C.** C-API direct LightGBM call (Regime A, compounds with A)
- **D.** Treelite AOT-compiled scorer (Regime B, fp32-aggregation)
- **E.** ONNX Runtime scorer (Regime B, alt to D)
- **F.** `torch.compile` on policy forward (Regime B, 30–50 %
  forward speedup)
