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
| + S01 (profile + assess) | 8.389 (1-ep, profile run only) | — | n/a (assessment-only) | 0 |
| v1 reference (`ppo_lstm_v1`) | 2.94 | — | — | — |

The S01 row is the 1-episode wall reported by the py-spy profile run
(99.6 s / 11872 ticks). It is NOT a 5-ep median — Session 01's
deliverable is the profile + assessment, and the profile run is the
measurement. The 5-ep median protocol kicks in from Session 02 (per
the per-session contract in `purpose.md` §"Per-session contract").
8.389 ms/tick sits inside the 8.0–10.7 episode-to-episode noise band
documented in Phase 4, so it is consistent with the pre-Phase-6
median of 9.595 ms/tick within sample noise.

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

**Verdict: GREEN with surprises.** A single dominant subsystem at
~74 % of per-tick rollout wall is identified
(`compute_extended_obs` and its scorer pipeline). Within that
subsystem the cost is split across three roughly co-equal
contributors and one previously-uncatalogued hot path. Session 02
attacks the largest profile-validated win (Candidate A, batched
scorer + batched calibrator); the SURPRISING contributor
(`feature_extractor._spread_in_ticks` calling
`env.tick_ladder.tick_offset` ~2500 times per call) is documented
for purpose.md triage and a likely Session 03.

### Profile run

- **Wall time:** 99.6 s episode (n_steps=11872) → **8.389 ms/tick**
  (1-ep, single point estimate; sits inside the 8.0–10.7 ms/tick
  episode-to-episode noise band from Phase 4 S01).
- **py-spy samples:** 12,962 raw at 100 Hz; 11,804 inside the main
  thread call tree (used as denominator for percentages below).
- **SVG:** [`logs/discrete_ppo_v2/phase6_s01_profile.svg`](../../../logs/discrete_ppo_v2/phase6_s01_profile.svg).
- **JSONL:** [`logs/discrete_ppo_v2/phase6_s01_post.jsonl`](../../../logs/discrete_ppo_v2/phase6_s01_post.jsonl).
- **Episode metrics:** reward=−1455.6, day_pnl=−578.1, n_updates=744,
  approx_kl=0.036 (max=0.139). Healthy update path; rollout was
  representative.

### Per-subsystem share of wall

Percentages are of `_collect` (the per-tick rollout loop body),
which itself is 78.5 % of total py-spy samples. Using `_collect`
as the denominator gives clean ms/tick numbers: every 1 % of
`_collect` = 0.084 ms/tick. PPO update is reported separately
since it lives outside `_collect` and runs once per episode.

| Subsystem | % of _collect | Approx ms/tick |
|---|---|---|
| **env_shim wrapper** (`compute_extended_obs` + scorer + calibrator + feature_extractor) | **73.8 %** | **6.19** |
| &nbsp;&nbsp;↳ `feature_extractor.extract` (28× per tick) | 40.0 % | 3.35 |
| &nbsp;&nbsp;↳ `lightgbm.Booster.predict` (28× per tick) | 26.9 % | 2.26 |
| &nbsp;&nbsp;↳ `isotonic._transform` (calibrator, 28× per tick) | 14.1 % | 1.19 |
| &nbsp;&nbsp;↳ env_shim Python self / overhead | ~3 % | ~0.25 |
| **env step** (`env/betfair_env.py::step` + matcher + bet manager) | 8.2 % | 0.69 |
| **policy forward** (LSTM + heads + dist sampling) | 10.1 % | 0.84 |
| **rollout collector overhead** (rest of `_collect` not in env_shim/policy) | 7.9 % | 0.67 |
| **Other / noise** (stdlib bookkeeping inside `_collect`) | ~0 % | ~0 |

Caveats on the env_shim row: `feature_extractor.extract` (3.35
ms/tick) and `lightgbm.Booster.predict` (2.26 ms/tick) and
`isotonic._transform` (1.19 ms/tick) overlap with the env_shim
wrapper number — they are children, not siblings. Sub-rows sum
to ~6.5 ms/tick of which the wrapper itself adds ~0.25 ms of
self time (the Python for-loop body in `compute_extended_obs:360`).

PPO update (`_update_from_batch`, called once per episode):
**0.77 ms/tick equivalent** when amortised across 11,872 ticks
(7.0 % of total py-spy samples). Smaller than expected from the
Phase 4 lessons hypothesis (see "Implications" below).

### Top 5 hot frames (self time, per-tick rollout)

Self-time = a frame's own samples minus the sum of its children.
This is what would be saved if the frame were eliminated entirely
and its children re-parented. Percentages are of `_collect`.

| # | ms/tick | % of rollout | Frame |
|---|---|---|---|
| 1 | 0.96 | 11.5 % | `tick_offset` ([env/tick_ladder.py:116](../../../env/tick_ladder.py#L116)) — called from `feature_extractor._spread_in_ticks` |
| 2 | 0.66 | 7.9 % | `__inner_predict_np2d` (`lightgbm/basic.py:1320`) — LightGBM C-extension predict wrapper |
| 3 | 0.56 | 6.7 % | `_band_for` ([env/tick_ladder.py:53](../../../env/tick_ladder.py#L53)) — ladder band linear scan |
| 4 | 0.55 | 6.6 % | `_band_for` ([env/tick_ladder.py:54](../../../env/tick_ladder.py#L54)) — same fn, adjacent bytecode line |
| 5 | 0.34 | 4.1 % | `forward` (`torch/nn/modules/rnn.py:1178`) — `nn.LSTM` forward |

Combining all `_band_for` lines (53 + 54 + 55): **1.21 ms/tick =
14.4 %** of rollout. Combining all `tick_offset` lines (116 + 110
+ 106): **1.24 ms/tick = 14.7 %**. The `tick_ladder` helpers
together (`tick_offset` + `_band_for` + `snap_to_tick`) are
**~28 % of rollout = 2.4 ms/tick** — almost entirely reachable via
the `feature_extractor._spread_in_ticks` chain inside
`compute_extended_obs`. **This is the single biggest "unlisted"
hot path in the profile.**

### Per-candidate upper-bound speedup

Estimates against the candidate menu in `purpose.md`
§"Candidate optimisations". Recovery numbers are conservative
(assume the wrapper overhead is the only saveable portion; the
underlying C/C++ kernel time is unchanged).

| Candidate | Target frame % rollout | Upper-bound recovery (ms/tick) | Notes |
|---|---|---|---|
| A. Batched booster.predict | 26.9 % (lightgbm) | **~1.6** | Batching collapses 28 calls → 1; saves Python wrapper, ctypes marshalling, NaN/finite checks. C-extension `__inner_predict_np2d` (0.66 ms/tick) unchanged. Wrapper portion ~1.6 ms/tick of the 2.26 ms/tick. **Naturally extends to batching the calibrator (Candidate A′ below).** |
| A′. Batched calibrator (companion to A) | 14.1 % (isotonic) | **~1.0** | Same 28→1 pattern on `IsotonicRegression.predict`. `check_array` (~0.5 ms/tick), `_assert_all_finite` (~0.1), `clip`/`_wrapfunc` (~0.3) all paid once. Underlying scipy `_evaluate` interpolation kernel (~0.13 ms/tick) unchanged. NOT explicitly on the menu but mechanically identical to A — fold it into A's session. |
| B. Slot-index cache | < 0.5 % | **< 0.05** | The `next(j for j, r in enumerate(...))` is invisible at the profile's resolution. compute_extended_obs:360 self time is 0.143 ms/tick total, of which the iterator is one fraction. **Don't bother as a standalone session.** |
| C. C-API direct LightGBM | depends on A | **~0.05** post-A | Saves the residual Python-wrapper overhead in `Booster.predict` *after batching*. With A shipped the wrapper runs once per tick; C-API saves ~50 µs/tick on top. Without A: ~1.4 ms/tick (28 × 50 µs). C is a substitute for A, not a complement. |
| D. Treelite | 7.9 % (lightgbm C kernel) | **~0.4** | 2–5× on `__inner_predict_np2d` (0.66 ms/tick). Mid-range estimate ~0.4 ms/tick. Compounds with A. Regime B. |
| E. ONNX Runtime | 7.9 % (lightgbm C kernel) | **~0.4** | Same shape and ballpark as D. Pick whichever has cleaner integration with the existing `model.lgb` artefact. Regime B. |
| F. `torch.compile` on policy forward | 10.1 % (policy fwd) | **~0.3** | 30–50 % of `discrete_policy.forward`'s 0.84 ms/tick. ~0.25–0.42 ms/tick. Regime B. |
| H. Multi-episode measurement | n/a | n/a | Folded into Session 01's deliverable (profile run = 1-ep wall measurement). Ship multi-episode protocol from Session 02 onwards. |

**SURPRISING / off-menu candidate (S):** rewrite
`feature_extractor._spread_in_ticks(best_back, best_lay)` to be
O(1) — currently it loops up to 50 times calling
`tick_offset(best_back, n, +1)`, and each `tick_offset(price, n,
+1)` itself loops `n` times calling `_band_for`. Worst case 2500
`_band_for` calls per `_spread_in_ticks`, called 28× per tick.
The function returns a deterministic float (number of ticks
between two prices) — a closed-form using the ladder bands is
both bit-identical AND O(1). Estimated recovery: most of the
~2.4 ms/tick currently spent in `tick_ladder` helpers via the
scorer feature path → **~1.5–2.0 ms/tick**. Same ballpark as A.
Lives in `training_v2/scorer/feature_extractor.py` — in scope
(not an env edit). Would need a purpose.md candidate addition
before it can ship as a session target (per §"What this session
does NOT do" — operator triage).

### Ranked target list for Session 02+

Ordered by expected impact × inverse-cost. Each row's
recovery estimate is the upper bound from the per-candidate
table above, discounted by ~25 % for measurement noise and
implementation slack.

1. **Candidate A + A′ (batched booster.predict + batched
   calibrator.predict).** Estimated recovery **~2.0 ms/tick**
   (1.2 from A, 0.8 from A′ after slack). Regime A
   (bit-identical — LightGBM's per-row and batched paths
   produce identical floats; isotonic regression is a
   piecewise-linear function evaluated independently per row,
   also identical). Effort ~3 h. Single biggest profile-validated
   win on the existing menu, attacks the dominant subsystem
   directly. **Justification:** booster + calibrator are 41 % of
   per-tick rollout; the Python-wrapper portion (~75 % of that)
   is paid 28× per tick today and exactly once per tick after
   batching.
2. **SURPRISING candidate S (O(1)
   `_spread_in_ticks` in feature_extractor.py).** Estimated
   recovery **~1.5 ms/tick** after slack. Regime A (bit-identical
   via algebraic equivalent). Effort ~3-4 h. **Justification:**
   `tick_ladder` helpers via the scorer feature path are 28 % of
   rollout — the single biggest "unlisted" hot path in the profile.
   Requires a purpose.md candidate addition (operator triage)
   before it can be a session target. Strongly recommended for
   triage based on the magnitude of the profile evidence.
3. **Candidate F (`torch.compile` on policy forward).**
   Estimated recovery **~0.3 ms/tick** after slack. Regime B
   (fp32-aggregation parity). Effort ~4 h. **Justification:**
   smaller win than A or S but compounds cleanly with both, and
   the policy forward is the second-largest non-env_shim cost
   (10 % of rollout = 0.84 ms/tick). Defer until A + A′ ship so
   the post-A baseline is stable; without that, the 30–50 %
   speedup applies to a moving target.

Sessions 02–04 attacking these three in order targets a
cumulative ~3.8 ms/tick recovery from the 8.4 ms/tick baseline
→ ~4.6 ms/tick post. That clears the ≤ 4.0 ms/tick GREEN bar
with one further session of headroom (likely D or E to attack
the surviving 0.66 ms/tick LightGBM C-kernel cost after A).

### Verdict rationale

**GREEN with surprises.** The profile identifies a single
dominant subsystem (`compute_extended_obs` and its scorer
pipeline at 73.8 % of per-tick rollout = 6.2 ms/tick), so the
GREEN criterion ("single dominant cost ≥ 30 % of wall
identified; Session 02 attacks it") is met by a wide margin.
Inside that subsystem the cost splits across three roughly
co-equal Python-wrapper contributors (feature_extractor 40 %,
lightgbm wrapper 27 %, calibrator 14 %), so attacking it well
takes more than one fix — but every fix lands on the same
subsystem, so the "one-fix-per-session × multiple sessions"
shape is unambiguous.

The "with surprises" qualifier covers two findings the Phase 4
lessons did not anticipate:

1. **The calibrator wrapper overhead is 14 % of rollout — bigger
   than expected and almost as costly per call as the LightGBM
   call itself.** sklearn's `IsotonicRegression.predict` runs
   `check_array` + `_assert_all_finite` + `np.clip` + scipy's
   `interp1d._evaluate` for every single 1-row prediction. The
   underlying interpolation kernel is tiny (~0.13 ms/tick); the
   wrapper is ~1.05 ms/tick. Same 28-call pathology as the
   booster — and the same fix (batch the call) collapses both.
2. **`feature_extractor._spread_in_ticks` is calling
   `env.tick_ladder.tick_offset` thousands of times per tick.**
   The function counts how many ticks fit between two prices via
   a linear walk; for typical horse-market spreads the inner loop
   runs 1–5 times, but its O(n²) shape (each step does its own
   `_band_for` band scan) and the 28 calls per tick add up. The
   profile shows tick_ladder helpers at ~28 % of rollout =
   ~2.4 ms/tick. None of it is in the env (these calls all
   originate in `training_v2/scorer/feature_extractor.py`,
   which is in scope). Closed-form replacement is bit-identical
   and ~1.5–2 ms/tick recoverable.

### Implications for Phase 4 lessons

The Phase 4 lessons hypothesised that "the env_shim's
`compute_extended_obs` running the LightGBM scorer ~28× per
tick, the env step's matcher/bet-manager path, the PPO update's
~744 mini-batch SGD steps per episode, and Python/torch dispatch
overhead" together dominated the wall (`purpose.md` §"Purpose"
item 3). The profile **partially confirms** and **partially
refutes** this list:

- **Confirmed:** env_shim's compute_extended_obs running the
  scorer 28× per tick is the dominant cost. At 73.8 % of
  per-tick rollout this is even bigger than the lessons
  suggested. The candidate menu's emphasis on batched-scorer as
  the highest-impact target (Candidate A) is right.
- **Refined:** the cost inside compute_extended_obs is NOT
  concentrated in the LightGBM call — it splits across the
  feature extraction (40 %), the LightGBM wrapper (27 %), and
  the calibrator wrapper (14 %). The LightGBM C-extension
  itself is only 8 % of rollout. So Treelite/ONNX (Candidates
  D, E) attack a smaller slice than the menu's framing
  suggested; A + A′ attack a much larger one.
- **Refuted:** PPO update (`_update_from_batch`) is **0.77
  ms/tick equivalent** (7 % of total py-spy samples), not a
  dominant cost. The 744 mini-batch SGD steps per episode are
  expensive in absolute wall (~6.4 s of the 129.6 s py-spy run)
  but small per tick. Candidate G ("PPO update mini-batch
  consolidation") was correctly filed out-of-scope on a
  *correctness* basis; the profile additionally shows it would
  be a low-impact target even if scope allowed.
- **Refuted:** the env step's matcher/bet-manager path is **not
  a dominant cost** — `env/betfair_env.py::step` is 8.2 % of
  rollout = 0.69 ms/tick. The Phase 6 hard constraint #1 ("no
  env edits") forecloses attacking it anyway, but the
  profile shows there's little to attack there.

The biggest meta-lesson is that **the Phase 4 hypothesis listed
the right system (env_shim scorer) but missed the within-system
breakdown.** Two of the three big contributors inside
compute_extended_obs (the calibrator wrapper, the
`_spread_in_ticks` tick_ladder hot path) are genuinely surprising
and would not have been targeted in any code-read-driven phase.
This validates the Phase 6 premise — profile first, then attack.

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
