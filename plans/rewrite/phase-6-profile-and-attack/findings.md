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
| + S02 (batched booster + batched calibrator) | **6.923** | −2.672 vs pre-S02 5-ep baseline | A (bit-identical) | 4 (1 integration + 3 unit) |
| + S03 (O(1) `_spread_in_ticks`) | **3.968** | −2.955 vs S02 | A (bit-identical) | 8 (1 integration + 7 unit) |
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

## Session 02 — batched booster + batched calibrator

**Verdict: PARTIAL.** 5-ep median **6.923 ms/tick** sits in the
PARTIAL band `(6.5, 8.0]` per the per-session bar in the session
prompt. Recovery against the proper pre-Phase-6 5-ep baseline of
9.595 ms/tick is **−2.672 ms/tick** — *larger* than the predicted
~2.0 ms/tick after slack (1.2 from A, 0.8 from A′). Recovery
against the noisier S01 1-ep point estimate of 8.389 ms/tick is
**−1.466 ms/tick**, *smaller* than predicted. The verdict is
PARTIAL by the absolute-threshold rule but the change shipped its
predicted recovery against the right (5-ep median) baseline.

### What shipped

- Two-pass batched rewrite of
  `agents_v2/env_shim.py::compute_extended_obs`. Pass 1 collects
  the K priceable runner-side feature vectors and their
  destination indices in `extra`; Pass 2 calls
  `booster.predict(matrix)` once and `calibrator.predict(raw)`
  once on the stacked `(K, n_features)` matrix; Pass 3 scatters
  back, gating per-element via a vectorised `np.isfinite` mask.
  Pre-batch filtering (status, LTP, extract-failure) and the
  per-row finiteness contract are preserved unchanged.
- `tests/test_env_shim_batched_scorer.py`: 1 integration test
  + 3 smoke unit tests. The integration test runs one full
  episode on `--seed 42 --day 2026-04-23 --device cpu` against
  both the per-row reference (frozen copy of pre-S02 code, bound
  via `types.MethodType` monkeypatch) and the production batched
  path; asserts byte-equality on every step's `obs` and `mask`,
  per-step `info["raw_pnl_reward"]`, final `info["day_pnl"]`, and
  the first 100 `log_prob_action` entries from the rollout's
  collected transitions. Marked `@pytest.mark.slow` because each
  episode takes ~95 s; the smoke unit tests take <1 s and run by
  default.
- `tests/test_agents_v2_env_shim.py::TestScorerWiring`: two
  pre-existing tests updated. Their booster mocks returned a
  fixed `(1,)` array regardless of input shape — fine under the
  per-row K=1 call shape, broken under the batched K=N shape.
  The mocks now return `np.full(x.shape[0], FIXED, …)` so they
  work under both call shapes.

### Parity result

**Bit-identity verified end-to-end.** The pre-write sanity
check on synthetic inputs confirmed booster and calibrator
both produce identical floats between batched and per-row
paths (max abs diff = 0.0 for both). The full-episode
integration test confirmed this carries through to every step's
obs/mask/info and the first 100 log-prob entries — no ULP
divergence in either direction.

### ms/tick recovery vs prediction

| Reference baseline | Pre | Post | Δ | Predicted | Verdict |
|---|---|---|---|---|---|
| Pre-Phase-6 5-ep median | 9.595 | 6.923 | **−2.672** | ~2.0 | exceeds prediction |
| S01 1-ep point | 8.389 | 6.923 | −1.466 | ~2.0 | undershoots |
| Absolute target ≤ 6.5 | n/a | 6.923 | n/a | yes | **misses by 0.4** |

Per-episode walls (sec): 77.86, 75.03, 83.36, 83.98, 82.19.
Per-episode ms/tick: 6.559, 6.320, 7.022, 7.074, 6.923 (median
**6.923**, mean 6.779; n_steps fixed at 11872 per episode).

### Root-cause hypothesis for the threshold miss

The PARTIAL verdict is driven by anchor choice, not by the
optimisation underdelivering. The Session 01 1-ep point estimate
of 8.389 ms/tick sat at the low end of the 8.0–10.7 ms/tick
episode-to-episode noise band documented in Phase 4 S01; the
true pre-S02 5-ep median is closer to the 9.595 ms/tick of the
Phase 4 final baseline (also a 5-ep median, same data, same
seed). The post-S02 absolute target ≤ 6.5 ms/tick was derived as
8.4 − 2.0 = 6.4 from the noisy 1-ep anchor; against the proper
5-ep anchor (9.6 − 2.0 = 7.6) we shipped recovery at 6.923,
exceeding the predicted destination by 0.7 ms/tick. The change
ships the win it was scoped to ship; the threshold was set
against the wrong baseline.

### Implications for Session 03

The recovery vs the 5-ep baseline (2.67 ms/tick) is consistent
with Session 01's per-candidate upper-bound estimates for A + A′
(2.0 ms/tick after slack from a notional 2.6 ms/tick combined
ceiling). Session 03's Candidate S (O(1) `_spread_in_ticks`)
estimated recovery ~1.5 ms/tick after slack should land us
around **5.4–5.5 ms/tick** post-S03, comfortably in GREEN
territory and within striking distance of the GREEN-with-stretch
≤ 3.0 v1-parity bar after one further session (Candidate F,
`torch.compile`).

The 5-ep noise discipline is paying off: per-episode ms/tick
spans 6.32–7.07 (range 0.75 ms), the median is well-defined,
and the 2.67 ms/tick recovery is multiple times the
within-cohort spread. A 1-ep point estimate would not have
distinguished this verdict from "in the noise".

### Surprises

- The integration test ran in 190 s for two full episodes,
  faster than expected — the extra ~3 s overhead per episode for
  the env.step capture wrapper is negligible.
- No pre-existing v2 trainer / rollout / collector / cohort /
  policy tests broke. Two scorer-wiring tests in
  `test_agents_v2_env_shim.py` had to be updated because their
  booster mocks were per-row-shape-coupled — that surfaced
  immediately, not as a flake. The fix made the mocks
  call-shape-agnostic, which is strictly the right contract for
  a wiring test anyway.

## Session 03 — O(1) `_spread_in_ticks`

**Verdict: GREEN.** 5-ep median **3.968 ms/tick** sits well inside
the GREEN band (recovery ≥ 1.0 ms vs the post-S02 baseline). Recovery
against the post-S02 5-ep median of 6.923 ms/tick is **−2.955
ms/tick** — almost **2× the predicted ~1.5 ms/tick after slack**.
The cumulative pre-Phase-6 → post-S03 result (9.595 → 3.968) clears
the Phase 6 GREEN success bar of ≤ 4.0 ms/tick (2.42× speedup),
and brings v2 within 1.0 ms/tick of v1 reference (2.94).

### What shipped

- Closed-form O(1) rewrite of
  `training_v2/scorer/feature_extractor.py::_spread_in_ticks`. Two new
  module-level helpers: `_band_index_and_snap` (inline band-table
  lookup + grid snap, mirrors `env.tick_ladder.snap_to_tick` semantics)
  and the rewritten `_spread_in_ticks` (band-by-band walk bounded by
  `_N_BANDS = 10`). Module-level constants `_BAND_LOWS`,
  `_BAND_HIGHS`, `_BAND_STEPS`, `_N_BANDS` precomputed at import from
  `env.tick_ladder._LADDER_BANDS` (the single source of truth — ladder
  edits in env automatically flow through). The new implementation
  imports BAND CONSTANTS only from `env/tick_ladder.py`; it does NOT
  call any function from that module. The public function signature
  (`_spread_in_ticks(best_back, best_lay) -> float`) is unchanged.
  The single call site inside `extract()` at line 224 is unchanged.

- **Algorithm note (deviation from prompt sketch).** The prompt's
  sketch snapped BOTH prices and computed tick distance between them.
  This breaks bit-identity when `best_lay` is between two ticks but
  closer to the lower tick (e.g. `(2.10, 2.105)` — the original walk
  returns 1, snap-both returns 0, because `snap(2.105) = 2.10` rounds
  DOWN). The shipped algorithm snaps `best_back` only and walks
  band-by-band to find the smallest `k >= 1` with
  `walked_price(k) >= best_lay - 1e-9` — preserving the original's
  ceiling semantic against the unsnapped raw `best_lay`. Pre-write
  10k random-pair check + 752 hand-constructed edge cases (every band
  boundary, ±tick prices, MIN/MAX clamping, near-half-tick FP edges):
  **0 diffs** vs the iterative walk oracle.

- `tests/test_feature_extractor_spread_in_ticks.py`: 7 unit tests.
  - `test_closed_form_matches_walk_on_10k_random_pairs` — load-bearing
    bit-identity guard. NaN-aware byte equality across 10 000 random
    `(best_back, best_lay)` pairs spanning `[1.01, 1000.0]`. Walk
    oracle is kept private to the test file
    (`_spread_in_ticks_walk_oracle`).
  - `test_zero_spread_returns_zero` — `back >= lay` short-circuit.
  - `test_single_band_spread` — `(2.10, 2.20) == 5.0`.
  - `test_cross_band_spread` — `(1.95, 2.10) == 10.0` (5 ticks of 0.01
    + 5 ticks of 0.02).
  - `test_spread_above_cap_returns_nan` — `(1.50, 3.00)` returns NaN
    (50+ ticks of 0.01 saturates the 49-tick cap).
  - `test_min_price_clamping` — `(1.005, 1.05) == 4.0`
    (1.005 snaps up to 1.01, then 4 ticks of 0.01).
  - `test_max_price_clamping` — `(995.0, 1005.0)` returns NaN
    (995 snaps to 1000 via banker's rounding `round(89.5) = 90`;
    best_lay > MAX so the walk never reaches it). Prompt left the
    snap behaviour open; documented inline in the test.

- `tests/test_phase6_s03_episode_parity.py`: 1 integration test
  (slow). Runs one full episode on `--seed 42 --day 2026-04-23
  --device cpu` twice — once with the production closed form and
  once with the pre-S03 iterative walk monkey-patched in via
  `monkeypatch.setattr("training_v2.scorer.feature_extractor.
  _spread_in_ticks", _walk_spread_in_ticks)` — and asserts byte-
  equality on every per-step `obs` and `mask`, per-step
  `info["raw_pnl_reward"]`, final `info["day_pnl"]`, and the first
  100 `log_prob_action` entries from the rollout. Mirrors the S02
  integration test shape. ~143 s wall for both episodes.

### Parity result

**Bit-identity verified end-to-end.** Pre-write 10k random + 752
edge-case check: 0 diffs. Episode-parity integration test (real-data
price distributions on the 2026-04-23 amber day): 0 diffs across all
11 872 steps' `obs` / `mask` / `raw_pnl_reward` / final `day_pnl` /
first-100 `log_prob_action`. The closed form is byte-equivalent to
the iterative walk on every input the rollout actually exercised.

### ms/tick recovery vs prediction

| Reference baseline | Pre | Post | Δ | Predicted | Verdict |
|---|---|---|---|---|---|
| Post-S02 5-ep median | 6.923 | **3.968** | **−2.955** | ~1.5 | **2× prediction** |
| Pre-Phase-6 5-ep median (9.595) | 9.595 | 3.968 | −5.627 | n/a (cumulative) | 2.42× speedup |
| Phase 6 GREEN bar (≤ 4.0 ms/tick) | n/a | 3.968 | n/a | yes | **passes by 0.03** |
| GREEN-with-stretch (≤ 3.0, v1 parity) | n/a | 3.968 | n/a | no | misses by 0.97 |

Per-episode walls (sec): 44.23, 46.19, 47.10, 49.04, 48.05.
Per-episode ms/tick: 3.726, 3.891, 3.968, 4.130, 4.048 (median
**3.968**, mean 3.953; range 0.40 ms; n_steps fixed at 11872).

### Why 2× the predicted recovery

The S01 estimate of ~1.5 ms/tick after slack was conservative on two
axes: (1) it assumed only the wrapper-overhead portion of the
~2.4 ms/tick `tick_ladder` cost was saveable, but the closed form
also eliminates the per-call `_band_for` band-scan cascade (each old
`tick_offset(price, n, +1)` did `n` band scans, and the `_spread_in_
ticks` outer loop called `tick_offset` up to 49 times per call —
worst-case ~2400 band scans per call, called 28× per tick); (2)
removing the 28 calls per tick to a function-with-import (the old
implementation re-resolved `from env.tick_ladder import tick_offset`
inside `_spread_in_ticks` on every call) shed Python attribute /
import-cache overhead the profile didn't isolate as a separate hot
frame. Both effects compound. The 2.96 ms/tick observed recovery is
consistent with the full 2.4 ms/tick of `tick_ladder` cost identified
in S01 (Top-5 hot frames combining `tick_offset` + `_band_for`)
disappearing entirely, plus a small downstream cache-locality bonus
from no longer interleaving 28 LightGBM matrix preparations with 28
band-scan cascades per tick.

### Implications for Session 04+

We are at **3.97 ms/tick**, 0.03 inside the Phase 6 GREEN bar. The
remaining headroom to v1 parity (2.94) is 1.03 ms/tick. From the S01
profile, the candidates that compose with shipped S02 + S03 are:

- **Candidate F** (`torch.compile` on policy forward, ~0.3 ms/tick
  after slack, Regime B). Attacks the policy forward — now the
  largest non-env_shim contributor since `tick_ladder` cost is gone.
- **Candidate D / E** (Treelite or ONNX Runtime, ~0.4 ms/tick after
  slack, Regime B). Attacks the residual LightGBM C-kernel
  (~0.66 ms/tick) inside the now-batched booster path.

Either alone closes ~30 % of the remaining gap. Combined they could
land us inside 3.0 ms/tick (GREEN-with-stretch). Both are Regime B;
the Phase 6 contract requires per-session declaration. Operator
triage decides Session 04's target after this commit lands.

A **re-profile** before Session 04 is recommended: with `tick_ladder`
gone from the hot list and the booster wrapper gone, the post-S03
profile shape will be different from S01's. Whatever was previously
hidden behind those two rises into view.

### Surprises

- The recovery exceeded prediction by 2×. The S01 estimate was based
  on the wrapper-overhead share of the visible `tick_ladder` cost; in
  practice the closed form also eliminates the worst-case algorithmic
  cost (49 outer × 49 inner band scans for cross-band spreads, even
  though typical horse spreads only exercise 1–5 outer iterations).
- The pre-write 10k synthetic check covered all edge cases the
  integration test would have surfaced — no real-data divergence
  appeared in the integration test run despite the algorithm having
  asymmetric snap semantics (snap-back-only with ceiling-walk-to-
  unsnapped-target). The 1e-9 slack in the original comparison
  absorbs all FP error in the closed-form arithmetic; `+`/`*` over
  a single multiply-add is well within the 1e-12 ULP-error budget
  the slack provides.
- No pre-existing tests broke. The 10 tests in
  `test_scorer_v1_inference.py` and `test_env_shim_batched_scorer.py`
  all still pass — the closed form is byte-equivalent, the contract
  the scorer pipeline depends on is preserved.

## Phase 6 verdict — GREEN, closed at S03

Two code-shipping sessions cleared the Phase 6 success bar. Operator
decision (2026-05-03): close Phase 6 here rather than run Session 04.

| Bar | Target | Achieved |
|---|---|---|
| §1 CPU ms/tick (5-ep median) | ≤ 4.0 | **3.968** (passes by 0.03) |
| §2 Per-session parity (Regime A) | bit-identical | **PASS** (S02 + S03) |
| §3 CUDA self-parity (Regime A) | preserved | PASS (no Regime B sessions shipped) |
| §4 Pre-existing v2 tests pass | all | **PASS** |
| §5 Cohort behavioural drift | none | n/a (no Regime B) |
| Cumulative speedup | ≥ 2.4× (per Phase 6 note) | **2.42×** (9.595 → 3.968) |
| GREEN-with-stretch (≤ 3.0, v1 parity) | stretch | **not achieved** (3.97 vs 2.94) |

**Why close here.** Post-S03 re-profile (2026-05-03,
[`logs/discrete_ppo_v2/phase6_s04_profile.svg`](../../../logs/discrete_ppo_v2/phase6_s04_profile.svg))
showed the rollout's hot-frame distribution is now dominated by two
subsystems:

| Rank | Subsystem | ms/tick | % rollout | In Phase 6 scope? |
|---|---|---|---|---|
| 1 | policy_forward (LSTM + heads + sampling) | 0.889 | 37.9% | yes |
| 2 | env_step (features, matcher, bet_manager, _process_action's tick_offset) | 0.583 | 24.8% | **no** (HC#1) |
| 3 | env_shim/feature_extractor (post-S03 residual) | 0.264 | 11.2% | yes |
| 4 | rollout collector self | 0.129 | 5.5% | yes |
| 5 | env_shim/booster_predict (LightGBM C-kernel only) | 0.125 | 5.3% | yes |
| 6 | env_shim/calibrator_predict | 0.115 | 4.9% | yes |
| ... | (others) | < 0.1 each | — | mixed |
| (PPO update equivalent) | 0.72 | n/a | **no** (Candidate G) |

The two largest remaining costs together (env_step 0.58 + PPO update
0.72 = 1.30 ms/tick) are both out of Phase 6 scope. The remaining
in-scope target ceiling is:

- **Candidate F (`torch.compile`)** — predicted 0.25–0.35 ms/tick
  recovery on the 0.89 ms/tick policy forward. Regime B. ~4 h.
- **Candidate D / E (Treelite / ONNX)** — predicted 0.05–0.08 ms/tick
  recovery on the now-small 0.125 ms/tick LightGBM C-kernel.
  Regime B. ~4 h. Diminished by S02; not worth a session at current
  size.

Best-case in-scope cumulative recovery: ~0.35 ms/tick → ~3.6 ms/tick
post-Session-04, still 0.66 ms/tick above v1 parity. **The path to
v1 parity (2.94) requires opening the env step**, which exits Phase 6
by definition.

## Phase 7 hand-off

Bullet-point scope sketch — full purpose.md to be written when Phase 7
opens. Targets identified by post-S03 profile; estimates conservative.

1. **`env/tick_ladder.py::tick_offset` and `ticks_between` closed-form
   rewrite.** Same shape as S03 — bit-identical band-arithmetic
   replacement. `tick_offset` is called from
   `env/betfair_env.py::_process_action` (matcher path) and
   `env/scalping_math.py::min_arb_ticks_for_profit`. `ticks_between`
   is called from `env/betfair_env.py:1286`. Profile shows
   `_band_for` + `tick_offset` in the env_step path at ~0.07 ms/tick
   self-time leaves; cumulative through `_process_action` is larger.
   Estimated recovery: ~0.10–0.15 ms/tick. Bit-identical (Regime A).
2. **`env/features.py` per-tick feature engineering** —
   `compute_traded_delta`, `compute_mid_drift`, `compute_book_churn`
   together ~0.07 ms/tick of leaves; cumulative through `_features_to_array`
   is larger (1.74 % of total = ~0.06 ms/tick at the leaf, but the
   cumulative chain is several × this). Likely candidates: vectorise
   per-tick computations, cache repeated runner traversals. Estimated
   recovery: ~0.10 ms/tick.
3. **`env/bet_manager.py` per-tick lookups** — `race_bet_count`,
   per-tick aggregations. Smaller; ~0.03 ms/tick. Defer unless 1+2
   under-deliver.
4. **(Phase 6 deferred) Candidate F (`torch.compile`)** — could ship
   independently of env work; predicted ~0.3 ms/tick. Cleanest test
   of "is the policy compile-stable on our shape" before any larger
   architectural change.
5. **(Phase 6 deferred) Candidate D / E (Treelite / ONNX)** — predicted
   ~0.05 ms/tick now. Probably never worth it on its own; bundle into
   a "small-wins" cleanup session or skip.

Combined Phase 7 ceiling (1 + 2 + 3 + 4): ~0.55 ms/tick → ~3.4 ms/tick
post-Phase-7. Still 0.46 ms/tick above v1 parity. Closing the final
gap likely requires a structural change (matcher rewrite, env step
loop reorganisation) that exceeds the per-session-bit-identity
discipline both phases inherit.

**Operator triage gate.** Phase 7 scoping should weigh the ~0.55
ms/tick remaining headroom against (a) RL-training throughput needs
on the 66-agent scale-up (gated separately), (b) the v1 deletion
gate, and (c) reward-shape / training-stability work in
`plans/arb-signal-cleanup/` and successors. If the wall is no longer
the binding constraint on iteration speed at the scale we actually
run, Phase 7 may be deferred indefinitely.
