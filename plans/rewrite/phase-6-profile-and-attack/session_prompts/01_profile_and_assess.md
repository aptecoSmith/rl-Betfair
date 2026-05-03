# Session prompt — Phase 6 Session 01: profile and assess

Use this prompt to open a new session in a fresh context.
Self-contained — does not require context from the session that
scaffolded it.

---

## The task

Run `py-spy` over a 1-day single-episode CPU rollout, produce a
flamegraph, and write a structured assessment that identifies
where wall time *actually* goes. Output a ranked target list for
Session 02+.

**No code change is required this session.** The deliverable is
the profile, the SVG, and a written assessment with verdict.

This is the load-bearing session of Phase 6 — Phase 4 attempted
seven optimisations without a profile and shipped PARTIAL on every
one because the targets weren't actually hot. Don't repeat that
mistake. **Read the profile first; recommend targets second.**

End-of-session bar:

1. **Profile run completed.** A py-spy SVG flamegraph is saved to
   `logs/discrete_ppo_v2/phase6_s01_profile.svg`. The underlying
   1-episode CPU run is logged to `logs/discrete_ppo_v2/phase6_
   s01_post.jsonl` so the per-episode wall is recorded alongside
   the profile.
2. **Per-subsystem cost table.** `findings.md` records the
   percentage of wall time spent in each of the four candidate
   subsystems:
   - `env_shim` (anything in `agents_v2/env_shim.py`, including
     scorer prep and the `booster.predict` / `calibrator.predict`
     calls)
   - `env step` (`env/betfair_env.py::step`, matcher, bet
     manager, reward accumulators)
   - `policy forward` (`agents_v2/discrete_policy.py::
     DiscreteLSTMPolicy.forward` and everything it calls into —
     LSTM, head Linears, masked Categorical / Beta sampling)
   - `rollout collector + PPO update` (`training_v2/discrete_ppo/
     rollout.py` and `trainer.py`)
3. **Top 5 hot frames absolute.** Listed in `findings.md` with
   function name, file:line, cumulative % of total wall, and
   approximate µs/tick.
4. **Per-candidate upper-bound speedup estimates.** For each item
   on the candidate menu in `purpose.md` §"Candidate
   optimisations" (A–H), estimate the maximum ms/tick recoverable
   given what the profile actually shows. A candidate whose
   target frame is 0.2 % of wall gets a near-zero upper bound;
   a candidate whose target frame is 30 % of wall gets a
   correspondingly large one.
5. **Ranked target list for Session 02+.** Top 3 candidates,
   ordered by expected impact × inverse-cost. Each entry includes:
   - Estimated ms/tick recovery.
   - Parity regime needed (A or B).
   - Rough effort estimate (h).
   - One-sentence justification grounded in the profile.
6. **Verdict logged** as one of:
   - **GREEN**: single dominant cost ≥ 30 % of wall identified;
     Session 02 attacks it.
   - **DISTRIBUTED**: cost spread across multiple subsystems with
     no single ≥ 30 % frame. Session 02 picks the
     highest-leverage combination.
   - **SURPRISING**: profile contradicts expectations (e.g.
     `np.asarray` is 40 % of wall, or a frame nobody anticipated
     dominates). Session 02 attacks whatever the profile shows.

You — the session's claude — assess the profile. The operator does
not. If the assessment requires interpretation (e.g. "is this
frame a Python wrapper or a C++ call?"), make the call yourself
based on the file path and surrounding frames. Don't punt.

## What you need to read first

1. `plans/rewrite/phase-6-profile-and-attack/purpose.md` — phase
   purpose, parity regimes, candidate menu, hard constraints.
2. `plans/rewrite/phase-4-restore-speed/findings.md` — Phase 4's
   verdict and the three causes that motivate Phase 6. Pay
   special attention to the per-session "Why the win was smaller
   than expected" sections — they document what each Phase 4
   session *thought* the dominant cost was vs what it turned out
   to be.
3. `agents_v2/env_shim.py` — read the full `compute_extended_obs`
   path (lines 263–396). The 28-call-per-tick scorer pattern is
   the leading hypothesis for the dominant cost; verify or refute
   from the profile.
4. `agents_v2/discrete_policy.py::DiscreteLSTMPolicy.forward`
   (lines 333–386) — the full per-tick policy forward.
5. `training_v2/discrete_ppo/rollout.py::_collect` — the per-tick
   rollout loop; you'll see most of it as the call-stack root in
   the flamegraph.
6. `training_v2/discrete_ppo/trainer.py::_ppo_update` — the PPO
   update path; ~744 mini-batch SGD steps per episode is the
   second leading hypothesis.

## Implementation

```bash
# 1. Install py-spy if not already installed (it's a sampling
#    profiler — minimal runtime overhead, no code instrumentation).
pip install py-spy

# 2. Profile a 1-episode CPU run on the seed-42 / 2026-04-23
#    baseline. py-spy attaches to the process and samples the
#    Python call stack at ~100 Hz. The --rate, --duration, and
#    --idle flags are tuned for our episode shape:
#    - --rate 100: 100 samples/sec is plenty at our scale and
#      keeps the SVG small.
#    - --duration 0: profile until the process exits naturally.
#    - --idle: include idle / I/O frames (they tell us if we're
#      blocking on something).
py-spy record \
    --rate 100 \
    --output logs/discrete_ppo_v2/phase6_s01_profile.svg \
    --idle \
    -- python -m training_v2.discrete_ppo.train \
        --day 2026-04-23 \
        --data-dir data/processed_amber_v2_window \
        --n-episodes 1 \
        --seed 42 \
        --out logs/discrete_ppo_v2/phase6_s01_post.jsonl \
        --device cpu
```

The SVG is interactive — open it in a browser. Width = % of
wall time spent in the function (and its descendants); depth =
call-stack depth.

### Reading the flamegraph

For each candidate subsystem, identify the entry point and
measure its width in the flamegraph:

- **env_shim**: look for frames under `compute_extended_obs` in
  `agents_v2/env_shim.py`. The scorer call (`booster.predict`)
  will be a child frame; LightGBM internals will appear as
  C-extension frames.
- **env step**: look for `env/betfair_env.py::step` and its
  descendants. Matcher and bet manager frames are children.
- **policy forward**: look for `DiscreteLSTMPolicy.forward` in
  `agents_v2/discrete_policy.py`. The LSTM call (`nn.LSTM`)
  will be a child; head Linears appear as `nn.Linear.forward`.
- **rollout collector + PPO update**: look for
  `RolloutCollector._collect` and `PPOTrainer._ppo_update` as
  call-stack roots; their descendants are everything in the
  trainer / rollout modules. Most of the profile's leaf frames
  will sit under one of these two roots.

py-spy's SVG also has a search box (top-right) — type a
substring and matching frames are highlighted, with cumulative %
shown. Use this to get exact percentages instead of eyeballing
widths.

### Estimating upper-bound speedups

For each candidate (A–H in `purpose.md`):

- **A. Batched scorer.** Speedup ≈ `(scorer_call_pct − scorer_kernel_pct) × wall_ms`.
  The Python wrapper overhead (input validation, ctypes
  marshalling, output massaging) is what batching removes. The
  C++ tree-walking time is unchanged. If `booster.predict`'s
  Python frames are 25 % of wall and the LightGBM C-extension
  frames underneath are 5 %, the batched form recovers ~20 %
  of wall — minus the small fixed cost of the new batched call.
- **B. Slot-index cache.** Speedup ≈ time spent in
  `next(j for j, r in enumerate(...))` per slot per side per
  tick. Easy to spot in the profile if non-trivial.
- **C. C-API direct.** Speedup ≈ Python wrapper overhead in
  `booster.predict` *that survives after batching*. If A is
  shipped first this is much smaller than naively summing.
- **D / E. Treelite / ONNX.** Speedup ≈ time in LightGBM C++
  tree-walking. Their internal kernels are different but
  serve the same role; estimating their improvement requires
  benchmarking on a sample feature matrix (out of scope for
  Session 01 — defer to the session that proposes them).
- **F. `torch.compile`.** Speedup ≈ 30–50 % of the policy
  forward's wall, ballpark from PyTorch's published
  benchmarks on small CPU networks. Apply to whatever %
  `forward` is in the profile.
- **H. Multi-episode measurement.** Already folded into
  Session 01's deliverable — the run is single-episode (the
  profile run *is* the measurement, and the profile-driven
  re-baselining gives Session 02+ the floor it needs).

### Writing the assessment

Open `findings.md` and replace the Session 01 row's "TBD"
markers. Concrete shape:

```markdown
## Session 01 — profile and assess

**Verdict: <GREEN | DISTRIBUTED | SURPRISING>**

### Profile run

- Wall time: <X>s (n_steps=11872 → <Y> ms/tick).
- SVG: `logs/discrete_ppo_v2/phase6_s01_profile.svg`.
- JSONL: `logs/discrete_ppo_v2/phase6_s01_post.jsonl`.

### Per-subsystem share of wall

| Subsystem | % of wall | Approx ms/tick |
|---|---|---|
| env_shim (scorer prep + predict + calibrator) | <X.X> % | <Y.YY> |
| env step (env.step, matcher, bet manager) | <X.X> % | <Y.YY> |
| policy forward (LSTM + heads + dist sampling) | <X.X> % | <Y.YY> |
| rollout collector + PPO update | <X.X> % | <Y.YY> |
| Other / unattributed | <X.X> % | <Y.YY> |

### Top 5 hot frames

1. `<file>:<line> <function>` — <X.X> % of wall, ~<Y> µs/tick
2. ...

### Per-candidate upper-bound speedup

| Candidate | Target frame % wall | Upper-bound recovery (ms/tick) | Notes |
|---|---|---|---|
| A. Batched booster.predict | <X> % | <Y.YY> | <one line> |
| B. Slot-index cache | <X> % | <Y.YY> | <one line> |
| C. C-API direct | <X> % | <Y.YY> | <one line> |
| D. Treelite | <X> % | <Y.YY> | requires Regime B |
| E. ONNX Runtime | <X> % | <Y.YY> | requires Regime B |
| F. torch.compile | <X> % | <Y.YY> | requires Regime B |

### Ranked target list for Session 02+

1. **<Candidate name>** — est. <Y.YY> ms/tick recovery, Regime
   <A|B>, ~<H> h effort. <One-sentence justification grounded
   in the profile.>
2. ...
3. ...

### Verdict rationale

<2-3 sentences explaining why the chosen verdict (GREEN /
DISTRIBUTED / SURPRISING). For GREEN: name the dominant frame
and what attacks it. For DISTRIBUTED: name the multi-target
combination. For SURPRISING: explain what the profile showed
that contradicts expectations.>

### Implications for Phase 4 lessons

<1-2 sentences: did the profile confirm the Phase-4-lessons
hypothesis (env_shim scorer + PPO update dominate)? Or did it
show something else? This frames the rest of Phase 6.>
```

## Hard constraints

1. **No code change ships in this session.** The deliverable is
   the profile + assessment + ranked target list. If you find
   something so obvious you want to fix it on the spot — don't.
   File it as the Session 02 candidate and stop. The
   one-fix-per-session contract starts working only when each
   session's measurement is unambiguously attributable to one
   change.
2. **The profile must run on the same baseline as Phase 4.**
   Same day (2026-04-23), same data directory
   (`data/processed_amber_v2_window`), same seed (42), same
   device (CPU), same `--n-episodes 1`. Cross-comparable rows
   are load-bearing for the verdict-session writeup.
3. **You assess the profile.** The operator cannot. Reading a
   flamegraph is the work; don't summarise the SVG and ask the
   operator what they think. If a frame is ambiguous (e.g.
   "lightgbm/basic.py:Booster.predict" — is the cost in the
   Python wrapper or the C++ kernel?) — open the file, read the
   wrapper, decide based on what's visible in the profile, and
   document your reasoning.
4. **Be explicit about uncertainty.** py-spy is a sampling
   profiler; sub-1 % frames are noise. Frames whose attribution
   is uncertain (e.g. shared C-extension code called from
   multiple paths) — say so. The next session's prompt will be
   written based on this assessment; downstream confidence
   tracks upstream confidence.
5. **No new GA gene additions / no reward-shape changes / no
   env edits / no v1 imports** — all inherited from
   `purpose.md` §"Hard constraints".

## Out of scope

- Profiling the GPU run. The Phase 3 Session 01b CUDA self-
  parity bar uses CUDA but Phase 6's measurement baseline is
  CPU. A separate GPU profile is its own session if ever
  needed.
- Profiling cohort-batched runs (`BatchedRolloutCollector`).
  Phase 6 is single-process, single-agent throughput.
- Profiling the env_shim's feature extraction in isolation
  (i.e. without the scorer call). That's a Session 02+
  candidate if the profile motivates it.
- cProfile-based deterministic profiling. py-spy's sampling
  approach is what we want for "where does time go" questions
  at the µs scale; cProfile's per-call instrumentation
  distorts microbenchmarks.

## Deliverables

- `logs/discrete_ppo_v2/phase6_s01_profile.svg` — py-spy
  flamegraph.
- `logs/discrete_ppo_v2/phase6_s01_post.jsonl` — 1-episode
  training row for cross-comparability.
- `plans/rewrite/phase-6-profile-and-attack/findings.md` —
  Session 01 row populated per the template above. The
  pre-existing skeleton in the file gives the structure;
  replace the "Status: NOT YET RUN" block with the assessment.
- Commit: `feat(rewrite): phase-6 S01 (GREEN|DISTRIBUTED|
  SURPRISING) - profile + assessment, target list for S02+`.
  Include the verdict and the top-1 ranked target in the commit
  body. **Add the SVG and JSONL to the commit** so the profile
  output is reproducible — these are not normally ignored
  artifacts but the `logs/` dir is gitignored, so use
  `git add -f`.

## Estimate

~2 h:

- 30 min: install py-spy, run the profile, save the SVG.
  (The 1-ep training run is ~2 min wall; py-spy adds <5 %
  overhead and no warmup. The bulk of the 30 min is one-time
  install + verifying the SVG opens correctly.)
- 1 h: read the flamegraph, extract per-subsystem percentages
  and top-5 hot frames, fill in the per-candidate upper-bound
  table.
- 30 min: write the verdict rationale, the Phase-4-lessons
  implications paragraph, the ranked target list. Update
  findings.md, stage the SVG and JSONL, commit.

If past 3 h, stop and check scope — likely the profile is
showing something unexpected and you've spent budget on a tangent.
The verdict, target list, and findings.md update are the
load-bearing outputs; don't over-polish the assessment write-up.

## What this session does NOT do

- **Does not pick the Session 02 implementation.** It picks the
  Session 02 *target* (and its parity regime). The Session 02
  prompt — written separately, after this session lands —
  specifies the implementation, the test shape, and the
  measurement protocol.
- **Does not propose new candidate optimisations beyond the
  menu in purpose.md.** If the profile reveals a totally new hot
  frame (e.g. "60 % of wall is in `np.asarray` calls in the
  rollout loop"), document the surprise in the verdict-rationale
  paragraph but keep the target list anchored on the existing
  menu unless one of those candidates *is* what attacks the
  surprise frame. The operator triages new-candidate proposals
  before they enter the menu — that's a purpose.md edit, not a
  Session 01 deliverable.
- **Does not re-run Phase 4 measurements.** The Phase 4 final
  baseline (post-S07, 11.009 ms/tick on 1-ep) is in
  `phase4_s07_post.jsonl` and is what Session 02 measures
  against. Phase 6's own pre-baseline is the median of the
  pre-Phase-4 5-ep run from `run_cpu_post_sync_fix.jsonl`
  (9.595 ms/tick) — same as in `findings.md`'s opening table.
