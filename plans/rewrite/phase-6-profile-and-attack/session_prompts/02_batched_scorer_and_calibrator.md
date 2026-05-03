# Session prompt — Phase 6 Session 02: batched scorer + batched calibrator

Use this prompt to open a new session in a fresh context.
Self-contained — does not require context from the session that
scaffolded it.

---

## The task

Replace the 28× per-tick scalar `booster.predict()` and
`calibrator.predict()` calls in `agents_v2/env_shim.py::
compute_extended_obs` with **one batched call to each**, run on a
single stacked `(K, 30)` feature matrix where K = number of
priceable runner-sides for the current tick (1–28).

This implements **Candidate A + Candidate A′** from
`purpose.md` §"Candidate optimisations". Both are mechanically
identical and Session 01's findings recommend they ship together
to keep the per-session measurement above the per-episode noise
floor (see `findings.md` §"Per-candidate upper-bound speedup"
A′ row).

**Parity regime: A (bit-identical).** LightGBM's per-row and
batched predict paths produce identical floats for the same
inputs. Isotonic regression is a piecewise-linear function
evaluated independently per row — also identical. The
per-session correctness guard is per-tick byte-equality of `obs`,
`mask`, `hidden_state`, `per_runner_reward` against a fresh
pre-change baseline on `--seed 42 --day 2026-04-23`.

**Estimated recovery:** ~2.0 ms/tick after slack (1.2 from A,
0.8 from A′; see `findings.md` ranked target list entry 1).
Pre-baseline is the Phase 4 final at 9.595 ms/tick (5-ep median);
Session 01 1-ep point estimate was 8.389 ms/tick. Post-S02
target is **≤ 6.5 ms/tick (5-ep median)**.

End-of-session bar:

1. **Code change.** `compute_extended_obs` collects the K
   priceable runner-side feature vectors into one
   `(K, n_features)` float32 ndarray, calls
   `booster.predict(matrix)` once → `(K,)` raw output,
   calls `calibrator.predict(raw)` once → `(K,)` calibrated
   output, scatters back into the `extra` array using the slot
   indices captured during the collection pass. Skipped slots
   (inactive runner, missing LTP, NaN feature row) never enter
   the batch in the first place — they keep the existing
   "default zero, leave alone" behaviour.
2. **Fault isolation preserved.** A single bad row (NaN feature,
   non-finite calibrator output) must skip exactly that row's
   slot, not the whole batch. The pre-batch implementation does
   this via the per-runner `try/except` and per-row finiteness
   check (`env_shim.py:387–390`). The batched form replicates
   the contract by: (a) NaN-bearing feature rows never enter
   the batch — they are filtered upstream during collection
   (existing `try/except` around `feature_extractor.extract`
   stays in place), and (b) post-calibrator finiteness is
   checked element-wise on the `(K,)` output, with non-finite
   slots left at zero.
3. **Bit-identity test.** A new test in
   `tests/test_env_shim_batched_scorer.py` runs one full
   episode on `--seed 42 --day 2026-04-23 --device cpu` against
   both the pre-change and post-change implementations
   (parametrise via a temporary monkeypatch toggle that selects
   the per-row vs batched path) and asserts byte-equality on
   the final state of:
   - `np.array_equal` on each step's `obs`
   - `np.array_equal` on each step's `mask`
   - `info["raw_pnl_reward"]` on every step
   - `info["day_pnl"]` at episode end
   - The first 100 elements of the rollout's collected
     `transitions` log_probs (any divergence by ULP causes a
     test failure with a clear message).
   This is the load-bearing per-session correctness guard — do
   not refactor it into smaller unit tests; the integration
   surface is what catches a contract drift.
4. **Smoke unit tests** for the batched path in isolation
   (same file):
   - `test_batched_scorer_matches_per_row_on_synthetic_inputs`
     — fabricate K=10 random feature vectors, run both paths,
     assert exact equality.
   - `test_batched_scorer_skips_nan_row_in_isolation` — K=5
     where row 2 has a NaN; assert rows 0,1,3,4 produce the
     same outputs as their per-row equivalents and row 2's
     slot is left at 0.
   - `test_batched_calibrator_finiteness_check_per_row` — K=5
     where row 3's booster output is finite but the calibrator
     somehow returns inf (forge via a tiny crafted calibrator);
     assert row 3's slot is left at 0 and rows 0,1,2,4 are
     written.
5. **Measurement.** 5-episode CPU run on
   `--day 2026-04-23 --data-dir data/processed_amber_v2_window
   --seed 42 --device cpu`, written to
   `logs/discrete_ppo_v2/phase6_s02_post.jsonl`. Report the
   median ms/tick across the 5 episodes (Phase 6's per-session
   contract item 4). Cross-comparable with Phase 6's pre-Phase-6
   baseline median 9.595 ms/tick.
6. **Verdict logged in findings.md** as one of:
   - **GREEN**: median ms/tick ≤ 6.5 (the ~2 ms recovery
     target lands within slack).
   - **PARTIAL**: median ms/tick in (6.5, 8.0] — recovery
     smaller than predicted but real.
   - **FAIL**: median ms/tick > 8.0 — change shipped no
     measurable wall improvement; investigate before Session 03
     (likely a bit-identity break that forced a fallback path,
     or the batched call's wrapper overhead dominates the
     per-row overhead it replaced).

You — the session's claude — own all measurement and verdict
attribution. The operator does not. If the verdict is PARTIAL or
FAIL, write a 2–3 sentence root-cause hypothesis in the findings
row and stop; the operator will triage whether to proceed to
Session 03 or re-attack S02.

## What you need to read first

1. `plans/rewrite/phase-6-profile-and-attack/purpose.md` —
   especially §"Candidate optimisations" entries A and A′, and
   §"Hard constraints".
2. `plans/rewrite/phase-6-profile-and-attack/findings.md` —
   Session 01's per-subsystem table and §"Per-candidate
   upper-bound speedup" A and A′ rows.
3. `agents_v2/env_shim.py::compute_extended_obs` lines 317–396 —
   the full per-tick scorer pipeline. Pay particular attention
   to:
   - The `for slot in range(self._N)` loop and the slot-skip
     conditions (lines 337–348).
   - The runner-index lookup (lines 351–357) — Candidate B
     territory but not in scope this session.
   - The 28-call pattern at lines 358–393 (per side for back/lay).
   - The fault-isolation `try/except` at line 359–374.
   - The post-calibrator finiteness gate at lines 388–390.
4. `training_v2/scorer/feature_extractor.py::FeatureExtractor.
   extract` — the function that builds each per-runner feature
   dict. Returns a dict mapping feature_name → float; the env_shim
   currently materialises it into an `(1, n_features)` float32
   ndarray per call. **Do not modify** this function in S02 —
   it's the next session's target (Candidate S).
5. `models/scorer_v1/calibrator.joblib` is loaded as
   `self._calibrator` in `_load_scorer_artefacts` (env_shim.py
   line 415). It's a `sklearn.isotonic.IsotonicRegression`
   instance whose `.predict(x)` accepts a 1-D array of any
   length. Verify this via a one-liner before writing the
   batched code.
6. The Phase 4 PARTIAL verdict's per-session table in
   `plans/rewrite/phase-4-restore-speed/findings.md` — context
   for why per-episode noise discipline matters.

## Implementation

```python
# Sketch — actual code lives in agents_v2/env_shim.py.

def compute_extended_obs(self, base_obs):
    extra = np.zeros(2 * self._N, dtype=np.float32)
    race = self._current_race()
    tick = self._current_tick()
    if race is None or tick is None:
        return np.concatenate([base_obs, extra]).astype(np.float32, copy=False)

    slot_map = self.env._slot_maps[self.env._race_idx]
    runner_by_sid = {r.selection_id: r for r in tick.runners}
    feature_names = self._feature_spec["feature_names"]

    # Pass 1: collect K priceable runner-side feature vectors + their
    # destination indices in `extra`. Skip rows for inactive / missing /
    # unpriceable / extract-failed runners — same skip semantics as
    # the per-row path.
    rows = []           # list of (n_features,) float32 arrays
    extra_idx = []      # list of int — destination index in `extra`
    for slot in range(self._N):
        sid = slot_map.get(slot)
        if sid is None:
            continue
        runner = runner_by_sid.get(sid)
        if runner is None or runner.status != "ACTIVE":
            continue
        ltp = runner.last_traded_price
        if ltp is None or ltp <= 1.0:
            continue
        try:
            runner_idx_in_tick = next(
                j for j, r in enumerate(tick.runners) if r.selection_id == sid
            )
        except StopIteration:
            continue
        for side_idx, side in enumerate(("back", "lay")):
            try:
                feat_dict = self._feature_extractor.extract(
                    race=race, tick_idx=self.env._tick_idx,
                    runner_idx=runner_idx_in_tick, side=side,
                )
            except Exception:
                logger.debug("FeatureExtractor.extract failed for ...", exc_info=True)
                continue
            row = np.asarray(
                [feat_dict[name] for name in feature_names], dtype=np.float32,
            )
            rows.append(row)
            extra_idx.append(2 * slot + side_idx)

    if not rows:
        return np.concatenate([base_obs, extra]).astype(np.float32, copy=False)

    # Pass 2: one batched booster call, one batched calibrator call.
    matrix = np.stack(rows, axis=0)            # (K, n_features) float32
    raw = self._booster.predict(matrix)        # (K,) float32
    cal = self._calibrator.predict(np.asarray(raw))  # (K,) float64 typically

    # Pass 3: scatter back. Skip non-finite calibrator outputs (per-row
    # finiteness gate, preserved from per-row path).
    finite = np.isfinite(cal)
    clipped = np.clip(cal, 0.0, 1.0).astype(np.float32, copy=False)
    for k, dest in enumerate(extra_idx):
        if not finite[k]:
            continue
        extra[dest] = float(clipped[k])

    return np.concatenate([base_obs, extra]).astype(np.float32, copy=False)
```

The structural change is two passes (collect K rows + their
destination indices, then batch-predict + scatter) instead of
one nested loop with per-row predicts. The pre-`predict`
filtering (status, LTP, extract-failure) lives unchanged in
pass 1; the per-row finiteness gate moves from inside the loop
to a vectorised `np.isfinite` mask in pass 3.

### Bit-identity verification before writing the test

Before writing the integration test, do a one-shot sanity check
in a Python REPL or scratch script:

```python
import numpy as np, joblib, lightgbm as lgb
booster = lgb.Booster(model_file="models/scorer_v1/model.lgb")
cal = joblib.load("models/scorer_v1/calibrator.joblib")
rng = np.random.default_rng(42)
matrix = rng.normal(size=(28, 30)).astype(np.float32)

per_row_raw = np.asarray([booster.predict(matrix[i:i+1])[0] for i in range(28)])
batch_raw = booster.predict(matrix)
assert np.array_equal(per_row_raw, batch_raw), \
    f"booster ULP drift: max abs diff {np.max(np.abs(per_row_raw - batch_raw))}"

per_row_cal = np.asarray([cal.predict(np.asarray([per_row_raw[i]]))[0] for i in range(28)])
batch_cal = cal.predict(per_row_raw)
assert np.array_equal(per_row_cal, batch_cal), \
    f"calibrator ULP drift: max abs diff {np.max(np.abs(per_row_cal - batch_cal))}"
```

If either assertion fails, **stop and re-spec.** Bit-identity is
the load-bearing parity guard for this session; if the call
produces ULP-different outputs the change becomes Regime B and
the test shape needs to be `allclose(rtol=1e-5, atol=1e-7)` plus
end-to-end behavioural-parity validation (per `purpose.md`
§"Parity regimes" Regime B). That's a different session.

The Phase 4b candidate doc claims LightGBM is bit-identical
between batched and per-row paths; the calibrator's bit-identity
hasn't been previously verified. Run both checks before
committing to Regime A.

### Measurement protocol

```bash
# Single-day 5-episode CPU run; write to JSONL then summarise.
python -m training_v2.discrete_ppo.train \
    --day 2026-04-23 \
    --data-dir data/processed_amber_v2_window \
    --n-episodes 5 \
    --seed 42 \
    --out logs/discrete_ppo_v2/phase6_s02_post.jsonl \
    --device cpu

# Median ms/tick = median(ep_wall_seconds * 1000 / n_steps) across the 5 rows.
# Use python or jq to extract; do not eyeball.
```

Five-episode median per Phase 6's per-session contract item 4
(was 1-episode in Phase 4; the protocol change is what gives us
the noise discipline to detect <2 ms wins).

## Hard constraints

1. **No env edits.** `env/betfair_env.py`, `env/bet_manager.py`,
   `env/exchange_matcher.py`, `env/tick_ladder.py` — all
   off-limits. The change lives entirely in
   `agents_v2/env_shim.py`.
2. **No feature_extractor edits.** `training_v2/scorer/
   feature_extractor.py` is Session 03's territory (Candidate
   S). Touching it here breaks the per-session attribution
   discipline.
3. **No GA gene additions.** No reward-shape changes. No v1
   imports. (Inherited from `purpose.md` §"Hard constraints".)
4. **Regime A is non-negotiable for this session.** If the
   pre-write bit-identity check fails on either booster or
   calibrator, stop. Do not silently relax to Regime B.
5. **One fix per session** — do not also fix Candidate B
   (slot-index cache) or any other in-passing observation. File
   it as a follow-on if motivated.
6. **Five-episode median measurement.** A 1-episode point
   estimate sits inside the 8.0–10.7 ms/tick noise band per
   Phase 4 S01; if the verdict depends on a 1-ep number it is
   not load-bearing.
7. **Preserve fault isolation.** A single bad runner (NaN
   feature, non-finite calibrator output) must skip its own
   slot only — never the whole batch. Tests #4 in the
   end-of-session bar are the regression guard.

## Out of scope

- Candidate B (slot-index cache). Profile shows it's < 0.05
  ms/tick recoverable; not worth a session.
- Candidate C (C-API direct LightGBM). After A ships, C is
  ~50 µs/tick recoverable. Re-evaluate post-S02.
- Candidate S (`_spread_in_ticks` rewrite). Session 03.
- Candidate F (`torch.compile`). Session 04 candidate at
  earliest, after re-profile.
- D / E (Treelite / ONNX). Sessions 04+ candidates at earliest.
- Profiling. The 5-ep median is the measurement; no py-spy
  re-run needed unless the verdict is PARTIAL/FAIL and
  diagnosis requires it.

## Deliverables

- Modified `agents_v2/env_shim.py` — `compute_extended_obs`
  rewritten as two-pass batched path.
- New `tests/test_env_shim_batched_scorer.py` with the four
  tests listed in the end-of-session bar (1 integration +
  3 unit).
- `logs/discrete_ppo_v2/phase6_s02_post.jsonl` — 5-episode
  measurement run. `git add -f` (logs/ is gitignored).
- `plans/rewrite/phase-6-profile-and-attack/findings.md` —
  Session 02 row populated in the per-session ms/tick table
  AND a Session 02 narrative section below Session 01's,
  matching the Session 01 section's shape (verdict, what
  shipped, parity result, ms/tick recovery vs prediction,
  any surprises). The narrative section is short — no
  re-profile data unless the verdict required one.
- Commit: `feat(rewrite): phase-6 S02 (GREEN|PARTIAL|FAIL) -
  batched booster.predict + batched calibrator.predict` with
  the cumulative ms/tick in the commit body and a one-line
  recovery summary (predicted vs observed).

## Estimate

~3 h:

- 30 min: pre-write bit-identity sanity check on synthetic
  inputs for both booster and calibrator. This is the
  go/no-go gate for Regime A.
- 1 h: rewrite `compute_extended_obs` as two-pass batched
  path. Preserve all skip semantics and fault isolation.
- 1 h: write the four tests (1 integration, 3 unit). The
  integration test is the load-bearing one — most of the
  hour is on the toggle mechanism that lets it run both
  paths in the same pytest invocation without polluting
  module state.
- 30 min: 5-episode measurement run (~10 min wall) + median
  computation + findings.md update + commit.

If past 4 h, stop and check scope. The most likely scope
break is "I noticed Candidate B / C while in here" — file it
as a follow-on and stop. Per `purpose.md` hard constraint #7
(one fix per session).

## What this session does NOT do

- **Does not touch the feature_extractor.** That's Session 03.
  Even if the profile-driven ranking made it tempting to
  attack the larger 2.4 ms/tick `_spread_in_ticks` cost
  first, the order matters: A + A′ first because (a) it ships
  in Regime A trivially, (b) the post-A profile shape is
  different and informs whether S still has the same
  ranking, and (c) the per-session noise discipline depends
  on landing one fix at a time.
- **Does not re-profile.** A 5-ep median is enough to
  validate the verdict. Re-profile is Session 04's
  pre-decision activity.
- **Does not change parity regime mid-flight.** If
  pre-implementation bit-identity check fails, stop and
  re-spec — do not silently relax to Regime B (purpose.md
  §"Hard constraints" item 6).
