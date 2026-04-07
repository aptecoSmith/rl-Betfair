# Next Steps — after arch-exploration phase

Review candidates identified at the end of `plans/arch-exploration/`.
These are things we deferred, consciously skipped, or discovered but
did not act on while building the genetic exploration infrastructure.

This file is a **backlog**, not a plan. Items are reviewed and
promoted into their own session prompts (either in this folder or a
new sub-folder) when we decide to tackle them.

Rough ordering: most likely to matter at the top. Re-order after each
real multi-generation run — results will reprioritise everything.

---

## 1. The actual multi-generation exploration run

Session 9 was explicitly a **shakeout** — 21 agents, one generation, 4
train + 2 test days, verifying infrastructure. The whole point of the
arch-exploration phase was to enable a real search, and we haven't run
one yet.

Deserves its own session prompt with:

- Dataset span (how many train days, how many held-out test days)
- Population size
- Number of generations
- Fitness metric (mean test-day P&L? best-day? risk-adjusted?)
- Stop criteria (convergence? fixed budget?)
- What to log for post-run analysis

Budget this before anything else on the list — the results will
reprioritise everything below.

**Status:** not started.

---

## 2. Hold-cost reward term (deferred from Session 7)

Session 7 scope block: *"The 'hold cost per open-liability-tick' idea
discussed in design review. That has the same asymmetric pitfalls and
deserves its own design pass — not this session."*

Now that Option D (reflection-symmetric range position) gave us a
template for making closed-form shaped terms zero-mean by
construction, a second design pass for hold-cost has a decent starting
point. The naive form — `−ε × liability × ticks_open` — is strictly
non-positive and would bias policies toward under-betting. A
reflection-style formulation (e.g. centred on the agent's own moving
average of liability-time) might work.

**Status:** design pass needed before any code.

---

## 3. Risky PPO knobs we consciously skipped (deferred from Session 2)

Session 2 left these out with the rationale *"interact with rollout
length and GPU memory":*

- `mini_batch_size`
- `ppo_epochs`
- `max_grad_norm`

Now that the shakeout ran cleanly we have a baseline. Worth
revisiting, at minimum the first two with narrow safe ranges.
`max_grad_norm` is the least interesting of the three — don't promote
it unless a real run shows gradient-scale variance mattering.

**Status:** ready to schema-expand, no design work needed.

---

## 4. Arch-specific ranges for genes beyond `learning_rate` (deferred from Session 6)

Session 6 scoped `TrainingPlan.arch_lr_ranges` to `learning_rate` only.
Transformer and LSTM have meaningfully different sensible ranges for:

- `entropy_coefficient`
- `ppo_clip_epsilon`
- Structural knobs (`lstm_*` are meaningless for transformers;
  `transformer_*` are meaningless for LSTMs)

The backend plumbing exists — widening it is mostly schema work on
`TrainingPlan.from_dict` plus UI updates.

**Status:** schema + UI work, no design.

---

## 5. Fourth architecture candidates

From the original design review, never pursued:

- **Feedforward + attention-over-runners baseline** — a non-recurrent
  sanity check. Answers "do we actually need recurrence for this
  problem?" Good negative result is as useful as a good positive one.
- **Hierarchical model** — per-runner transformer block pooled into a
  market transformer. More expressive than the current mean/max pool
  and would test whether the flat per-tick embedding is a bottleneck.

Both are additive (plug into the existing architecture registry) and
either could be a single session. The feedforward baseline is cheaper
and more informative per hour of effort — do that one first.

**Status:** either is a standalone session; not blocked on anything.

---

## 6. Pre-existing test failure never fixed

`plans/arch-exploration/lessons_learnt.md` (Session 1 entry):

> `tests/test_population_manager.py:408` hardcodes `obs_dim == 1630`
> but current value is 1636. Confirmed stale on plain `master` via
> `git stash`. Left alone — fixing drive-by failures is exactly the
> "while I'm here" scope creep Session 1's plan warns against.

Still broken. Ten-minute fix, should be part of the next housekeeping
sweep.

**Status:** trivial, just do it.

---

## 7. Coverage math upgrades (deferred from Session 4)

Session 4 explicitly deferred: *"A more sophisticated scheme (Latin
hypercube, Bayesian bandit, etc.) is explicitly deferred."* The
simple decile-bucket scheme shipped.

After a few real multi-gen runs we'll know whether the simple scheme
is actually biting. Revisit **only if** it is — no speculative
upgrades.

**Status:** parking lot until Session 10 results exist.

---

## 8. LSTM `num_layers > 2` (deferred from Session 5)

Session 5 clamped to `{1, 2}` with the note: *"Revisit only if
Session 9 results show a compelling reason."* Session 9 didn't look
for this.

**Status:** parking lot until real-run data says otherwise.

---

## 9. Market / runner encoder changes

Kept stock across Sessions 5 and 6 on the principle that encoder
changes belong in a dedicated session. Still valid.

There's no architectural reason to believe the current MLP encoders
are optimal, but there's also no empirical signal saying they aren't.
Don't touch until we have a reason.

**Status:** parking lot until a real run gives us a reason.

---

## 10. Optimiser / schedule work (deferred from Session 6)

Session 6 explicitly excluded: *"LR warmup, weight decay, any
optimiser change. The existing Adam/AdamW and single-LR setup
stays."*

Transformer agents specifically often want LR warmup. Worth
revisiting if the real multi-gen run from #1 shows transformer agents
under-training relative to LSTM agents of equivalent capacity.

**Status:** wait for #1 results, then decide.

---

## 11. Housekeeping

Small drive-by items worth a single sweep session:

- Legacy `reward_early_pick_bonus` scalar gene was split into `_min` /
  `_max` in Session 3 — confirm no stale references remain in
  `config.yaml`, frontend, docs, or test fixtures.
- `observation_window_ticks` was retired in Session 1 — same check.
- Session 8 proposed a read-only "schema inspector" view. Confirm it
  actually shipped.
- Any other `TODO` / `FIXME` comments added during sessions 1-9 that
  weren't triaged at the time.

**Status:** DONE (Session 10, 2026-04-07). See `progress.md`.

---

## 12. Pre-existing fast-suite failures (surfaced during Session 10)

Running the default-marker suite (`pytest tests/`) at the start of
Session 10 turned up **18 failed + 5 errors** that are not caused by
anything in the Session 10 diff — verified by stashing the diff and
re-running the same files. These are infrastructure / environment
issues, not logic bugs, and are out of scope for Session 10 per the
"do not fix integration tests that were already broken" rule. Filed
here so they don't get lost.

Grouped by likely root cause:

- **Stale port 18002** — `tests/test_e2e_training.py` fixture
  `worker_proc` asserts the port is free. A previous crashed run
  left a worker bound. Needs a teardown fix or a `killall` in the
  fixture `autouse=True` cleanup.
- **MySQL / real-data dependent** —
  `tests/test_integration_session_2_7b.py`,
  `tests/test_integration_session_2_8.py`,
  `tests/test_real_extraction.py`, and
  `tests/test_session_2_8.py::TestPPOTimeLSTMPolicy::test_gradients_flow`
  all require locally-extracted parquet data or a running MySQL.
  They should arguably be `@pytest.mark.integration`-marked and
  excluded from the default addopts filter (they are currently
  unmarked and only ever pass on machines where the data fixtures
  are hydrated).
- **Worker IPC / API status** — `tests/test_training_worker.py`
  `TestWorkerStartup` errors and `tests/test_api_training.py`
  `TestTrainingStatus` failures look like the same
  running-worker-expected dependency.
- **Session 4 endpoint drift** —
  `tests/test_session_4_9.py::TestStartEndpoint::test_start_returns_run_config`
  and `TestStopEndpoint::test_stop_sets_event`. May be a legitimate
  regression, may be a fixture issue. Needs investigation.

**Disposition:** park for now. Revisit if Session 11 pre-flight
touches the training-worker / API code path and needs confidence
in those tests, or as its own dedicated sweep session.

---

## Suggested sequencing

1. **#11 Housekeeping sweep** + **#6 obs_dim test fix** — an hour,
   clears debt.
2. **#1 Session 10: real multi-generation exploration run** — the
   thing everything else is downstream of.
3. **Read the results**, then pick between:
   - #2 hold-cost design pass
   - #3 risky PPO knobs
   - #5 fourth architecture (start with the feedforward baseline)
   - #10 optimiser / LR warmup
   based on what the run actually showed needs fixing.

#4, #7, #8, #9 are parking-lot items — don't do them speculatively.
