# Backbone signal-preservation probe — 2026-05-24

## Setup

Loaded Phase-15 cohort agent 1 (UUID `55aea2b6-…`, trained 16 days,
direction_prob_loss_weight gene draw = 0.515, hidden_size = 256). Ran
its policy's LSTM backbone forward on 20,000 (tick, runner) obs
vectors from the properly-populated `2026-04-11` oracle cache (the
one re-scanned with `--use-direction-predictor`). For each sample,
captured `lstm_last` ∈ R^256 — the exact representation the
`direction_prob_head` consumes during training.

Then trained two logistic-regression classifiers (sklearn LogReg,
class_weight='balanced', StandardScaler):

* **(A)** raw 574-dim obs → label
* **(B)** 256-dim `lstm_last` → label

Held-out 20 % split. Compared val BCE descent vs the pos-weighted
uniform-0.5 random floor.

## Results

```
                                            uniform   val      descent
                                            floor     BCE      (rel)

A: full obs (574-dim)        → label_back   1.137    1.010    +11.2%
A: full obs (574-dim)        → label_lay    1.134    0.918    +19.1%

B: backbone hidden (256-dim) → label_back   1.137    1.149     −1.1%
B: backbone hidden (256-dim) → label_lay    1.134    1.118     +1.5%
```

## Verdict

**The backbone destroys the direction signal.** Raw obs carries
11-19 % BCE-relative descent; the LSTM hidden state preserves
essentially nothing. The 256-dim representation that
`direction_prob_head` reads is, from the labels' perspective, noise.

This means:

* **`direction_prob_loss_weight` increases will not help.** Even at
  10× the current gene ceiling, supervising a head that reads near-
  zero-signal-content has no learning surface to climb.

* **The fix has to be architectural.** Specifically, the
  direction_prob_head needs a path to the raw obs that bypasses the
  policy-optimised backbone.

## Recommended fix

Add a residual obs path into `direction_prob_head`:

```python
# Current (agents_v2/discrete_policy.py around line 711):
direction_logits = self.direction_prob_head(lstm_last)

# Proposed:
direction_input = torch.cat([
    lstm_last,                           # (batch, hidden_size)
    obs_predictor_columns,               # (batch, max_runners * 12)
], dim=-1)
direction_logits = self.direction_prob_head(direction_input)
```

Where `obs_predictor_columns` is the slice of the obs vector
containing just the 12 dir_* columns × `max_runners` (168 features
for max_runners=14).

This:

* Keeps the supervised auxiliary loss + actor_input column-feed
  contract intact.
* Adds 168 input dims to `direction_prob_head` (the head goes from
  `hidden_size × max_runners` weights to `(hidden_size + 168) ×
  max_runners`).
* Trivial param overhead (~43k extra weights for hidden_size=256,
  max_runners=14).
* Architecture-hash break — load_state_dict on pre-fix weights
  will fail, which is the correct behaviour (the head shape genuinely
  changed). Operator must launch a fresh cohort.

## Why we expect this to work

The raw-obs logreg (A) achieves 11-19 % descent. The new head's
input dim INCLUDES the same raw obs columns the logreg uses (plus
the lstm_last context). The head's loss-weighted gradient flow
will pull weights on the residual columns directly, without
relying on the backbone to preserve the signal. Empirically, a
single linear layer is enough capacity to recover the signal a
logreg recovers (they're the same model up to optimisation
details).

## Open questions

* **Per-runner vs global residual path.** The 12 predictor columns
  are PER-RUNNER in obs. Each runner's direction_prob_head output
  should ideally read its OWN runner's 12 columns, not all
  14×12=168. A per-runner head architecture is cleaner. For now
  the global concat is a workable first cut; per-runner refinement
  can come in a follow-on if the simple concat doesn't fully
  recover signal.

* **Should we apply the same fix to `fill_prob_head` and
  `mature_prob_head`?** Likely yes — same backbone-destroys-signal
  argument applies. But those heads' BCE descents during the
  Phase-15 cohort suggest they're already learning (fill_bce=0.003,
  mat_bce=0.110 on agent 1 day 10). So the issue may be specific
  to direction, where the signal is more subtle. Stay narrow on
  direction for this round; revisit the other heads if/when
  direction works.

* **Does the value/policy gradient flowing into the backbone
  through `lstm_last` ALSO destroy fill/mature signal?** The
  backbone-probe doesn't test this. Worth a follow-on probe.
