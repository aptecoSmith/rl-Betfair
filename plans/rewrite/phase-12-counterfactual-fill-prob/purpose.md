---
plan: rewrite/phase-12-counterfactual-fill-prob
status: DRAFT
opened: 2026-05-06
depends_on: rewrite/phase-8-oracle-bc-pretrain (S03 + overnight findings)
---

# Phase 12 — Counterfactual fill-prob labels

## Why this plan exists

After Phase 7 (BCE aux heads) + Phase 8 (BC pretrain) + Phase 9
(per-transition credit), maturation rate is stuck at ~0.20 across
every cohort we've run. The 2026-05-06 cross-cohort breakdown
([phase-8-oracle-bc-pretrain/findings.md](../phase-8-oracle-bc-pretrain/findings.md))
shows the **natural-fill rate** (passive leg fills via market
matching) is 0.17 – 0.21 in EVERY cohort tested — Phase 7 S06,
all three S03 arms, both overnight cohorts. Force-close vs naked
is just *what happens to the 75-80 % that never fill*; nothing
moves the fill rate itself.

This is not a training-signal problem. It is an **information
problem**. The fill_prob_head and mature_prob_head are both
trained on the agent's own rollout experience — they can only
label "did the pair I opened actually fill?". They cannot label
the counterfactual "would a pair opened HERE have filled?" for
ticks the agent didn't act on. Without that signal, the actor
has no way to discriminate fillable from non-fillable opens at
decision time.

Phase 12 builds an offline counterfactual fill predictor. For
every priceable (tick, runner, side) the env matcher would accept
as a valid open, we forward-simulate across the remaining ticks
to label whether the passive leg WOULD have filled before
force-close / in-play. Train fill_prob_head on those offline
labels instead of the agent's rollout outcomes. The actor's
input vector now carries a calibrated per-runner P(fill) at every
decision point.

## What this is NOT

- **Not a replacement for the oracle.** The arb oracle
  identifies *profitable* moments (where the back+lay
  combination locks net profit after commission). Phase 12
  identifies *fillable* moments (where the passive would actually
  match). The intersection is what we want to teach the actor:
  "open here AND likely fill". Both labels feed the same
  decision.
- **Not a replacement for per-transition credit.** Phase 9's
  per-transition mature_prob credit is still the right shape —
  it just couldn't compensate for the actor's lack of fill
  information. Phase 12 fixes the root cause; Phase 9 stays.
- **Not in scope for Phase 12 V1: realistic queue-position
  modelling.** The first version uses a conservative
  "price-reachable" label (did best_back ever cross our P_lay?).
  Realistic fill simulation (queue position + traded volume
  through level) is a V2 enhancement.

## The label, precisely

For each pre-race tick T in race R, for each active + priceable
runner k whose top-of-book passes the env matcher checks (junk
filter, price cap, MIN_STAKE budget), compute:

```
P_back  = best_available_to_back_after_junk_filter
P_lay   = tick_offset(P_back, arb_spread_ticks, direction=-1)
T_close = T_force_close (T - force_close_before_off_seconds)
            OR T_in_play (whichever is earlier)

label_back_first(T, k) =
    1.0  if exists t in [T+1, T_close] such that
           best_back_at_tick_t(k) ≤ P_lay
    0.0  otherwise
```

`best_back_at_tick_t ≤ P_lay` means the lay-side queue at price
P_lay is reachable — a market participant placing an aggressive
back at P_lay would have crossed our passive lay. The conservative
interpretation is "fill possible, but queue position not
guaranteed". An UPPER BOUND on real fill probability under a
given setup.

The symmetric label for lay-first scalps is computed the same
way swapping sides:

```
P_lay   = best_available_to_lay_after_junk_filter
P_back  = tick_offset(P_lay, arb_spread_ticks, direction=+1)
label_lay_first(T, k) =
    1.0  if exists t in [T+1, T_close] such that
           best_lay_at_tick_t(k) ≥ P_back
    0.0  otherwise
```

We label both sides per (tick, runner) so the actor sees a
per-side fill predictor, matching the existing
`fill_prob_per_runner` head shape (one scalar per runner — though
to be precise, the head currently emits a single number per
runner; a per-side version would need (max_runners × 2) outputs.
Resolve in S01.)

## Why this should work

The 0.17 – 0.21 ceiling is set by the joint distribution of
(decision moment, market evolution). At any randomly-chosen tick,
P(20-tick favourable move within 60s) is small. But that
probability is not uniform — some ticks have far higher P(fill)
than others (e.g. high-volume periods just before the off, when
new info arrives and prices move). The actor's job is to identify
high-fill-probability moments AND restrict opens to those.

Currently the actor has only:
- Static features (price ladder, time-to-off, traded volume)
- Phase 0 scorer features (calibrated P(mature | features) for the
  exact features the LightGBM was trained on)
- The two BCE head outputs (fill_prob_per_runner and
  mature_prob_per_runner from the agent's own rollout history)

The Phase 0 scorer already targets a maturation-shaped label, but
it was trained on a different setup and per cohort findings has
not delivered selectivity. The two BCE heads are biased toward
what the agent already does.

A counterfactual fill_prob head trained on offline labels would:
- Cover every priceable tick × runner combination, not just
  what the agent acted on.
- Be uncorrelated with current policy behaviour.
- Provide a per-decision signal the actor can use BEFORE choosing
  whether to open.

If the natural fill rate has any predictability from the obs
features (which it should — pre-trade pressure, traded volume
velocity, time-to-off all correlate with movement magnitude), the
head will learn that mapping. The actor then has the input it
needs to be selective.

## Hard constraints

§1  Offline only. Same as the oracle. The label generator never
    runs inside the training loop.

§2  Deterministic. Same data + same config → same labels byte-for-
    byte. Sort labels by (tick_index, runner_idx, side) before
    writing.

§3  Match env matcher rules. A label of 1.0 must require that the
    env's `ExchangeMatcher` would accept the corresponding open at
    tick T (junk filter, price cap, budget). Otherwise the label
    teaches the actor about opens it cannot make.

§4  Cache format includes obs_schema_version + arb_spread_ticks.
    Different arb_spread → different labels. `load_labels` raises
    on mismatch.

§5  V1 uses the conservative "price-reachable" label. Realistic
    queue-position fill simulation is deferred to V2 once V1
    establishes the mechanism works.

§6  fill_prob_head's BCE loss uses offline labels only. The
    existing label (from agent rollouts) is removed entirely —
    not blended. Two label sources fighting on the same head
    produces ambiguous gradients.

§7  mature_prob_head is unchanged. It tracks "pair completes
    favourably" which is a different question (and is correctly
    labelled by per-transition credit from Phase 9). fill_prob
    answers "passive leg fills"; mature_prob answers "pair locks
    profit". Both feed actor_head independently.

§8  Cache invalidates when `arb_spread_ticks`,
    `force_close_before_off_seconds`, the junk-filter %, or the
    matcher's price cap changes. Header.json carries all four.

§9  Counterfactual scan walks every priceable runner-tick. On a
    typical training day (~9000 pre-race ticks × ~14 runners ×
    2 sides) this is ~250k labels. Performance budget: scan-day
    completes in under 5 minutes per day on CPU. If exceeded, the
    plan stops to optimise before continuing.

## Success bar

S01 (label generator + cache):
- Offline scan produces a `.npz` cache for at least one v2
  training day.
- Determinism test: scan twice, byte-identical labels.
- Per-day fill rate density printed: fraction of labels = 1.0.
  Sanity check — should be in the 0.20 – 0.40 range for the
  conservative price-reachable label, slightly above the
  observed natural-fill rate (since the label is an upper bound
  on real fill probability).

S02 (wire head to offline labels):
- Trainer's `fill_prob_loss_weight` BCE term reads from the
  offline label cache instead of agent-rollout-derived labels.
- The label-loading code raises a clear error if the cache is
  missing for a training day.
- Existing `mature_prob_loss_weight` path is untouched (Phase 9's
  per-transition credit still drives that head).
- Integration test: 1-agent dummy training run produces non-zero
  `fill_prob_bce_mean` in episode stats; the label distribution
  in the loss matches the offline cache (sample 50 transitions,
  assert label values come from the cache).

S03 (validation cohort):
- 12-agent × 3-gen cohort with `fill_prob_loss_weight` enabled,
  comparing offline-label arm vs no-fill-prob arm.
- Gate: offline-label arm achieves natural fill rate ≥ 0.25 by
  gen 3 (vs the stable 0.17 – 0.21 ceiling without).
- If the gate fails, V1 conservative label was insufficient —
  proceed to V2 (queue-position-aware fill simulation).

## Design decisions (resolved in session prompts)

The three open questions previously listed here have been
resolved in S01 / S02 prompts. Summarised:

1. **Per-side per-runner labels.** Each priceable (tick, runner)
   emits separate `label_back` and `label_lay` fields. The head
   widens to `(max_runners × 2)`; `actor_head` first layer
   bumps by 1 column. Architecture-hash break documented in
   hard_constraints §8.

2. **Per-cohort cache regen.** Cache filename embeds
   `spread{N}_fc{M}` for the `arb_spread_ticks` and
   `force_close_before_off_seconds` keys. A cohort with a
   different spread / fc combo regenerates its own cache.
   Multi-spread tensor support is V2.

3. **Class-balanced BCE per side.** `pos_weight_back =
   N_neg_back / N_pos_back` and same for lay, computed once at
   cache-load time. No focal loss, no sub-sampling. Two scalars
   broadcast over their respective per-side losses; the pair is
   averaged into the trainer's `fill_prob_loss_weight` term.

## Session structure (rough)

| Session | Deliverable |
|---|---|
| S01 | Offline label generator, cache CLI, determinism test, density print |
| S02 | Wire head to offline labels; remove agent-rollout label path |
| S03 | Validation cohort, fill rate gate |

## What's NOT in scope

- Queue-position-aware fill simulation (V2, only if V1 passes
  S03 gate).
- Multi-arb-spread label tensor (V2 if multi-spread cohorts are
  needed).
- Replacing mature_prob_head's label source (Phase 9 owns that).
- Replacing the oracle (Phase 8 owns that).
- Per-side fill prediction in BC pretrain. BC currently uses
  oracle's "open here" target; whether it should also use the
  fill predictor is a separate question (probably yes, but
  defer until V1 head is validated standalone).
