# Not Doing — deferred and rejected items

Items from `next_steps.md` that I have decided **not** to write
session prompts for, with reasons. This file is append-only.
Promoting an item out of here into a real session is allowed — but
only with new evidence that changes the reasoning recorded below.

The items in this file are still listed as parking-lot entries in
`next_steps.md`; this file is the explanation of *why* they live
in the parking lot.

---

## #7 — Coverage math upgrades (Latin hypercube, Bayesian bandit)

**Status:** parked indefinitely.

**Reason:** The simple decile-bucket coverage scheme shipped in
Session 4 of arch-exploration and survived Session 9 without
complaint. It is fit for purpose **until a real multi-generation
run shows it isn't.** Upgrading coverage math without concrete
evidence would be textbook speculative infrastructure: the upgrade
itself is plausible, but the problem it solves is imaginary.

A Bayesian-bandit-style search would also change the interpretation
of every gene correlation in the Session 11+ analysis, because
"undersampled" would no longer mean "the decile is empty" — it
would mean "the posterior is uncertain". That's a harder thing to
teach to the human reviewing results, and the benefit is unclear
until we've seen the simple scheme fail.

**Promotion criteria (any one of these):**
- Session 11 or later run shows that good agents cluster in a
  narrow corner of the search space while the planner keeps
  re-sampling from already-explored deciles.
- A run of 5+ generations produces no monotone fitness improvement
  and coverage analysis implicates the sampler as a likely cause.
- Someone complains that the current coverage page is uninformative
  in a way a posterior-based view would fix.

Until one of those happens: leave it alone.

---

## #8 — LSTM `num_layers > 2`

**Status:** parked.

**Reason:** Session 5 of arch-exploration capped the gene at
`{1, 2}` with the note *"The sequence of 1338-dim observations is
short enough that deep LSTMs are mostly a regularization disaster.
Revisit only if Session 9 results show a compelling reason."*
Session 9 did not look for this signal, and Session 11 has not yet
happened. Promoting this to a session now would be speculative.

Training a 3-layer or 4-layer LSTM on a relatively small dataset
(the restored hot/cold data is on the order of tens of days) is
the kind of thing that usually produces worse results *and* takes
more compute, so the asymmetric risk is bad.

**Promotion criteria:**
- Session 11+ run shows that the best 2-layer LSTMs consistently
  beat the best 1-layer LSTMs by a meaningful margin **and** the
  2-layer ones are not already overfitting (i.e. train/test gap is
  narrow).
- Dataset grows to the point where 3+ layer LSTMs are
  pre-theoretically plausible (roughly: 50+ training days).

---

## #9 — Market / runner encoder changes

**Status:** parked.

**Reason:** The current MLP encoders are shared across every
architecture (`ppo_lstm_v1`, `ppo_time_lstm_v1`,
`ppo_transformer_v1`, and the planned `ppo_feedforward_v1` and
`ppo_hierarchical_v1`). Touching them invalidates the comparison
between architectures, because a single encoder change affects all
of them equally and the signal from architecture ablations gets
muddled.

There is also no concrete evidence the encoders are a bottleneck.
They are MLPs applied to already-feature-engineered observations
from `env/observation_builder.py`. If the bottleneck is in the
feature engineering rather than the encoders, changing the
encoders won't help. And if the bottleneck is in the sequence
model, changing the encoders again won't help.

**Promotion criteria:**
- Session 11+ run shows that every architecture plateaus at
  roughly the same fitness despite meaningfully different
  sequence-model capacity, suggesting the encoders (not the
  sequence models) are the bottleneck.
- A dedicated ablation study (a mini-session in its own right)
  shows the encoder output is a low-rank bottleneck — e.g. the
  effective rank of the pooled runner embedding is much smaller
  than `mlp_hidden_size`.
- Someone proposes a specific alternative encoder with a
  principled reason to expect it to help (e.g. explicitly
  modelling price-book geometry). "Let's try a bigger MLP" is
  not a principled reason.

---

## Other items considered and kept in scope

Listed here for transparency — I considered parking these too
but decided they earned a session prompt:

- **#2 Hold-cost reward term** — parked for a single session in
  arch-exploration for asymmetry concerns, but the Option D
  template from Session 7 gives us a concrete path forward. Kept
  as Session 12 with the same design-pass-first discipline.
- **#3 Risky PPO knobs (`mini_batch_size`, `ppo_epochs`)** — the
  original rationale for deferral was uncertainty about safe
  ranges. Now that Session 9 established a baseline we can
  calibrate against, the ranges are pickable. Kept as Session 13.
  `max_grad_norm` is explicitly marked optional and left
  conditional on Session 11 evidence.
- **#4 Arch-specific ranges beyond `learning_rate`** — genuine
  follow-on from Session 6. Kept as Session 14.
- **#5 Fourth architecture candidates** — kept as *two* sessions
  (15 and 16) because the feedforward baseline and the
  hierarchical runner-attention model answer different questions.
- **#10 Optimiser / LR warmup** — kept but **gated on Session 11
  evidence** in the session prompt itself. If the run doesn't
  show a problem this session solves, its own preamble tells you
  to reconsider whether to run it.
- **#11 Housekeeping** — kept as Session 10, and deliberately
  placed first so Session 11 runs against a clean baseline.
