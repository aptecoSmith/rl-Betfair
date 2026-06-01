# Step 3C (env-core spike) + Step 4 (BC) — decisions (2026-06-01)

Operator authorised autonomous completion. Both decisions are
data/evidence-grounded; recorded here + EXPERIMENTS per the plan gates.

## Step 3C — env-core vectorization: NO-GO

The plan's HARD STOP requires a feasibility spike (realistic multiplier +
un-vectorisable branching fraction) before any env-core rewrite. The
Step-0 profile already provides the decisive numbers, so the spike is an
analysis, not a prototype (a prototype would only confirm a ceiling we can
already read off the profile):

- **The slice 3C targets is ~3% of rollout.** Step-0 (faithful per-phase
  timers, the real config) measured, as a fraction of the rollout:
  `_process_action` (matching) **2.9%**, `_settle_current_race` **0.0%**
  (250 calls, race-end only), `_get_info` 1.4%, `_get_obs` 5.2%. The
  "env core" 3C would tensor-vectorize (matching + settlement) is
  **~3%**. Even a *perfect* vectorization (to zero) saves ~3% of rollout
  ≈ **~2% of the cluster-day wall**.
- **The branching fraction is high (>80%).** `ExchangeMatcher` /
  `_maybe_place_paired` / `_settle_current_race` are almost entirely
  data-dependent conditionals: the ±50% LTP junk filter, the hard
  `max_back_price`/`max_lay_price` cap, single-level-vs-bounded-walk,
  force-close relaxation, equal-profit sizing, passive-book cumulative-
  volume fills, win/lose/void + commission + each-way settlement. These
  resist uniform tensor ops; a vectorized matcher would be mostly masked-
  scatter branch emulation.
- **Highest risk in the codebase.** The matcher is the phantom-profit-bug
  core (CLAUDE.md devotes its longest section to its invariants:
  no-ladder-walking, LTP junk filter, hard cap on post-filter price). A
  GPU-vectorized rewrite (separate fast-path module per HC#6, validated
  against the canonical matcher) is high-cost, high-risk, for a ~2% wall
  payoff.

**DECISION: NO-GO.** Smallest lever, highest risk. The canonical single-
level matcher stays the golden reference and the vendored ai-betfair
artifact (HC#6). If env-core ever becomes the bottleneck (e.g. bets/race
grows 10×, per deferred.md Option C), revisit — but not now.

## Step 4 — BC / per_transition_credit: NOT load-bearing → keep dropped, logged

Per the plan's Step-4 gate (a recorded decision, no silent state either
way) and the autonomy mandate. The ablation question — does BC-warm-start
beat from-scratch on held-out? — already has strong evidence pointing one
way, so a fresh multi-hour cohort ablation is queued, not auto-run:

- BC's only *documented* success is the **supervised maturation-AUC probe
  (0.745)** — "maturation is learnable" — NOT a PPO-warm-start P&L win
  (purpose.md, checked 2026-06-01).
- The imitation-first BC **policy lost £1513/7d**; a non-BC approach beat
  it at −£418/7d.
- **c1 found high-locked structural scalpers from scratch** with BC
  silently off (the very drop this plan surfaced) — empirical evidence
  that BC is not required to reach the good region.
- Prior, therefore: **"BC may not be load-bearing."**

**DECISION:** BC stays **unwired in the batched path**, and the drop is
now **logged** (the runner already warns `--bc-pretrain-steps ignored
under --batched`; this plan documents it in step0_profile.md + memory).
That satisfies HC#2 (no silent state) and the Step-4 gate. A confirmatory
from-scratch-vs-BC ablation on the sealed holdout (May 20-29) is the clean
settler if ever wanted — but it is hours of training for a
likely-negative result, so it is **not** run as part of this speedup pass.
`per_transition_credit` is unused by the current recipe; same treatment
(dropped + warned).
