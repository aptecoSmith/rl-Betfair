# Direction predictor / label alignment

**Date opened:** 2026-05-24

**Status:** investigation + fix in progress

## What's broken

The agent's policy carries an internal `direction_prob_head` that's
supervised by an offline BCE label (`label_back` / `label_lay` from
`data/direction_labels/`). The head is wired into `actor_head`'s
per-runner input column so the agent uses its calibrated direction
forecast in its action distribution. The pretrained `betfair-predictors`
direction model (a Conv1D) ALSO sits in the obs vector — its 12 per-
runner outputs (`dir_q*_*m` quantiles + `dir_fire_*` booleans) are
intended to give the policy the offline predictor's view "for free."

The 2026-05-24 cohort 1779613306 (12 agents, Phase-15, gen 1 agent 1
finished) revealed that the direction head **does not learn**. Across
all 16 training days, `dir_bce_back/lay` stays at the pos-weighted
random-uniform-0.5 floor (~1.14). Diagnostics:

* `tools/direction_signal_probe.py` (linear logreg on full 574-dim obs
  vs labels): descends BCE ~10–12% relative. Signal exists in obs
  SOMEWHERE.

* `tools/direction_head_inspection.py` (per-bet correlation of each of
  the 12 direction-predictor obs columns vs the offline labels):
  **NONE of the 12 columns exceeds |rho|=0.05 with the labels**. The
  "best" column for `label_back` is `dir_fire_drift` at rho=−0.050.
  Effectively noise.

* The same script confirms the head output is concentrated near 0.5
  (std=0.029 for back, 0.057 for lay) — head ISN'T learning task-
  specific features from the other obs columns either.

So the head is starved on two fronts: the supervised label doesn't
match the predictor's signal, and the head can't recover signal from
the rest of obs strongly enough to descend BCE.

## Root cause (smoking gun)

`env/betfair_env.py:2156-2164` openly admits the issue:

> The obs schema reserves 9 dir_qXX_Xm slots (3 horizons × 3 quantiles),
> named historically as `_1m`, `_3m`, `_7m`. Different predictor
> manifests may declare different horizon tuples (the pre-2026-05-22
> V2 champion used 1m/3m/7m; **the V4 champion uses 3m/7m/15m**). To
> keep the obs schema fixed and let the agent re-train against whatever
> the current predictor provides, we emit the FIRST THREE of the
> predictor's horizons into the obs slots in declared order, preserving
> the existing key names as positional labels (the agent learns from
> positional features, not from the names).

Combined with the manifest:

```json
"horizons": ["3m", "7m", "15m"]
"val_metrics": { "dir_acc_k5_7m": 0.6185, "dir_fires_k5_7m": 2936, ... }
```

The pretrained predictor:
* Forecasts at 3m / 7m / 15m horizons
* Its "fire" decision = "will the LTP move ≥5 ticks in the next 7
  minutes?" (k=5 ticks, primary 7m horizon)

Our offline labels (`training_v2/direction_label_scan.py`):
* `direction_horizon_ticks = 60` (a tick-count horizon, NOT a time
  horizon — ticks are event-driven, ~5-15 ticks/min depending on
  pre-off proximity)
* `direction_threshold_ticks = 5` (matches the predictor's k=5)
* Label = "did LTP move ≥5 ticks in the next 60 ticks?"

The threshold matches. The horizon does not — the predictor looks
7 MINUTES ahead while our labels look ~60 TICKS ahead (anywhere from
~4 minutes pre-off down to a few seconds in late pre-off). The two
events are statistically uncorrelated at the per-tick level.

## What this costs us today

Two of the 7 evolving GA genes — `direction_prob_loss_weight` and
`bc_direction_target_weight` — train a head against a label that
doesn't match the predictor signal in obs. The head can't descend
BCE meaningfully no matter what weight the GA picks, so:

1. The GA wastes 2/7 of its gene budget on a knob with near-zero
   traction.
2. The head feeds `actor_head` a column with no information — every
   tick, every runner gets the same ~0.5. The actor can't use the
   "direction advisor" pathway.
3. The agent only has the **raw predictor obs columns** (the 12
   `dir_q*_*m` + `dir_fire_*`) to learn from for direction signal —
   and those columns ARE in obs, but their semantic horizon (3m/7m/
   15m) is misaligned with the per-tick action decisions the agent
   is making.

This isn't a tooling bug, it's a **semantic data alignment bug**.
The predictor and the labels were spec'd separately, with different
horizon conventions, and the obs layer silently masks the mismatch.

## Why fixing matters

The whole point of `--use-direction-predictor` is to let the agent
*outsource* the "which way is price about to move" question to a
specialist offline model. That specialist exists, was trained
properly, and its `dir_acc_k5_7m=0.629` validation accuracy on a
sealed test set shows it works. We just aren't letting the policy
extract that signal because the supervised head's labels send the
head learning a different (incompatible) prediction.

The cohort cannot make meaningful use of the direction predictor
until labels and predictor speak the same language.

## High-level fix

Two coordinated changes:

1. **Regenerate the offline direction labels using a TIME horizon
   aligned with the predictor's 7m horizon** (the primary one the
   predictor's `dir_fires_k5_7m` is calibrated on). Specifically:
   change `direction_label_scan.py` from "look forward N ticks" to
   "look forward N seconds," and default to ~420 seconds. Per-cache
   file naming changes (new schema) so old caches are clearly
   incompatible.

2. **Renamed obs columns are out of scope for THIS plan** — they're
   cosmetic. The agent reads them positionally, not by name. The
   2026-05-24 commit comment in betfair_env.py already explains this.
   We add a follow-up plan to fix the names if it becomes a source of
   developer confusion, but not today.

## Success criterion

After the fix:

* `tools/direction_head_inspection.py` on a freshly-trained agent
  (with `direction_prob_loss_weight > 0.1`) shows |rho(head_out,
  label)| ≥ 0.15 — head meaningfully aligned with label.

* `tools/direction_signal_probe.py` (linear probe) on the new labels
  descends BCE ≥ 20% relative (vs the ~10% on misaligned labels) —
  the predictor obs columns now carry direct signal for the new
  labels.

* `dir_bce_back/lay` on the per-day log line descends meaningfully
  below 1.14 over the course of a 16-day training run for agents
  with `direction_prob_loss_weight ≥ 0.5`.

If any of these fails to improve, this plan didn't address the right
root cause and we revisit (alternative roots: head architecture is
too small, backbone is destroying the signal, etc.).
