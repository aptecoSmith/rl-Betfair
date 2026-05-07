---
plan: rewrite/phase-14-direction-gate
parent_purpose: ./purpose.md
authored: 2026-05-07 (pre-implementation)
purpose: Verify the planned interventions actually deliver the
         experimental results to the trained model.
---

# Sense check — does the plan deliver?

Each row of the experimental story below is checked against the
planned mechanism. The question for each: **will the trained
model receive what the probe demonstrated?**

## Step-by-step trace

### Probe finding 1: per-runner MLP head extracts ~10× more lift than single Linear

| Fact | Plan response |
|---|---|
| Probe trained `Linear(115→64) → ReLU → Linear(64→2)` per-runner. | S01 puts the SAME architecture (per-runner MLP) into `direction_prob_head`. |
| Probe used 64-hidden MLP. | S01 uses `actor_mlp_hidden=64` (the existing default in `DiscreteLSTMPolicy`). Same. |
| Probe input was a 115-dim per-runner feature slice. | The cohort's policy input per slot is `(slot_emb_i, lstm_last)` — `runner_embed_dim + hidden_size = 16 + 128 = 144` dim. NOT a raw RUNNER_KEYS slice. |

⚠ **Gap:** the probe and cohort don't see the same inputs to the
head. The probe sees the runner's raw feature slice (115 dims).
The cohort's head sees the LSTM's compressed hidden state (128
dims) plus a learned slot embedding (16 dims). The LSTM is
supposed to encode the relevant per-runner info into `lstm_last`,
and the slot embedding lets the head tell runners apart.

**Risk:** the LSTM may not push per-runner info into a per-slot-
addressable representation. In phase-13, the LSTM saw the full
obs (1792 dims; market + all 14 runners' features) and produced
a single 128-dim shared `lstm_last`. The single-Linear head's
failure was that 128-shared dims couldn't carry 14 runners'
direction signals. **A per-runner MLP fed by the SAME shared
`lstm_last` may have the same issue.**

The fix that actually mirrors the probe would be to feed the head
**the per-runner FEATURE SLICE** (RUNNER_KEYS for that runner),
not the LSTM's compressed state. But that's a bigger change.

**Mitigation in the plan:**
- The slot embedding `runner_slot_embedding` is LEARNED across
  PPO updates. It can in principle learn to encode "runner 5's
  context" if `lstm_last` carries enough per-runner info.
- The `actor_head` already operates on this same `(slot_emb_i,
  lstm_last)` input and produces meaningful per-runner action
  logits. So we know the architecture CAN extract per-runner
  signal — the actor isn't broken, only the direction head's
  shape was.
- If S04 shows BCE still flat on arm A at gen 1-2, this is the
  smoking gun and we revisit with a feature-slice input path.

**Verdict on Step 1:** probably-OK, with a documented risk that
the per-runner input pathway via shared lstm_last may not match
the probe's per-runner-slice input. **Watching metric in S04:
direction BCE trajectory on arm A. If flat, re-architecture.**

### Probe finding 2: 8 augmented features add 50-70% lift on top of base

| Fact | Plan response |
|---|---|
| Probe added `ltp_velocity_30/60`, `vol_delta_30/60`, 4 TradedVolumeLadder features. | S02 adds the same 8 features to `RUNNER_KEYS`. |
| Probe extracted features per (tick, runner) from the day's raw data. | S02 wires them into `feature_engineer.py` so they flow through the standard obs pipeline. |
| Probe's `ltp_velocity_30` was `(ltp_now - ltp_30_ticks_ago) / ltp_30_ticks_ago`. | S02 uses the same formula. ✓ |
| TradedVolumeLadder needed parquet snap_json parsing. | S02 adds `RunnerSnap.traded_volume_ladder` populated at parse time. ✓ |
| Probe used `pd.std`-based standardisation. | The cohort's policy doesn't standardise inputs at the obs level; it applies `input_proj = Linear → ReLU` first which can in-principle standardise via learned weights. But raw scale of `vol_delta_60` is in the thousands — its first-layer gradient may dominate other features at fresh init. |

⚠ **Gap:** scale mismatch. The probe normalised features per-day;
the cohort's policy doesn't do per-day normalisation, but other
volume features in `RUNNER_KEYS` already have `_log` companions
(`vol_delta_3_log` etc.) for this reason.

**Mitigation:**
- The probe's MLP works WITH these scales because it normalises.
  The cohort's policy's first `Linear` layer should learn to
  scale them out, but might take more training.
- S02's `compute_traded_volume_*` and the velocity features
  should add `_log` companions for the heavy-tail volume ones.

⚠ **Action needed:** S02 should also add `vol_delta_30_log` and
`vol_delta_60_log` (matching the existing `vol_delta_3_log` /
`_5_log` / `_10_log` pattern). Plan currently lists 8 features;
should be 10 (4 velocity-style + 2 vol-deltas + 2 vol-delta-logs +
4 ladder features = 12... actually let me recount).

Existing pattern in `RUNNER_KEYS`:
```
"vol_delta_3", "vol_delta_3_log",
"vol_delta_5", "vol_delta_5_log",
"vol_delta_10", "vol_delta_10_log",
```

So matching pattern, `vol_delta_30` and `vol_delta_60` should
have log companions: 2 raw + 2 log = 4 vol features. Plus 2 LTP
velocities + 4 ladder features = 10 total, not 8.

**Fix:** update S02 prompt to ship 10 features, not 8.
RUNNER_DIM 115 → 125 (not 123). Defer the actor_input shape
chain — S03's gate doesn't depend on this dim, but `actor_head[0]`
in `DiscreteLSTMPolicy` reads `runner_embed_dim + hidden +
4` — it's per-runner not per-feature, so unaffected. Good.

**Verdict on Step 2:** mostly-OK; flagging the missing `_log`
features as a small spec correction to S02.

### Probe finding 3: OOS calibration is 8-13× back-side lift; 3 of 3 OOS days profitable at empirical cost ratio

| Fact | Plan response |
|---|---|
| Probe trained on 4 pooled days, evaluated on 3 separate held-out days. | S04 cohort trains on 6 days, evaluates on 3 held-out. ✓ MORE training data than the probe. |
| Probe's MLP took 600-2000 supervised steps. | The cohort's PPO update sees ~400 mini-batch updates per training day × 6 days = ~2400 updates per generation × 4 generations = ~10000 updates. Far more than the probe. ✓ |
| Probe achieved 45-63% mature rate at T ∈ [0.90, 0.95]. | S04's gate gene range is [0.5, 0.95]. GA can reach the probe's optimum. ✓ |
| Probe's signal is from the augmented features. | S04's arms BOTH have S02 active. ✓ |
| Probe used label = "did LTP cross threshold within horizon". | The cohort's matched-pair outcome is "did the passive ACTUALLY fill". The label is necessary but NOT sufficient — gated opens may produce LOWER mature rate than the label-based prediction. |

⚠ **Gap:** label vs realised fill. The probe's "mature rate" is
a label-based metric, not a fill-based metric. The cohort's
matched-pair outcome depends on:
1. The price moving (label condition) — probe accounts for this.
2. The passive having queue position when price reaches it.
3. Counter-party volume at the rest price being sufficient.

Conditions 2 and 3 are NOT tested in the probe.

**Implication:** if the probe says 45-63% mature at T=0.95, the
realised cohort number may be 30-50%. Still above the 34.8%
break-even, but tighter. **The 35% bar in the success criterion
already accounts for this (set deliberately at break-even, not at
the probe's optimistic 45-63%).**

**Verdict on Step 3:** OK; the label-vs-fill gap is documented
risk and the success bar is already conservative.

### Probe finding 4: per-pair P&L is £3.37 mat / £1.80 force; break-even 34.8%

| Fact | Plan response |
|---|---|
| Numbers from 92 phase-13 cohort eval rows. | Phase 14's S04 will produce ITS OWN per-pair P&L numbers. Phase 14's matched-pair P&L could differ if the gate changes WHICH pairs the agent opens (e.g. only high-confidence-direction pairs may have higher locked spreads if directional confidence correlates with stable priceability). |
| Cost ratio drives break-even at 34.8%. | S04's mature rate gate is set at 35% (above break-even). |

✓ **No gap.** The empirical numbers underpin the success
criteria; if the cohort reproduces similar per-pair magnitudes
(±20%), the 35% bar holds. If phase-14's per-pair numbers
diverge substantially (e.g. lower locked because gated opens
land on different pairs), recompute the break-even on the new
data.

### Probe finding 5: gate at T=0.90-0.95 produces 233-1554 opens/day

| Fact | Plan response |
|---|---|
| Probe data: 233 opens at T=0.95 on day with 55k priceable rows; cohort opens ~410 at all-rows pace. | S03's gate range is [0.5, 0.95]; agents drawing 0.95 will see ~233 opens/day baseline. |
| Phase-13 hard_constraint §10: agents drawing 0.99+ never open. | S03 clamps upper to 0.95. ✓ |
| Probe's data assumes the agent opens on EVERY priceable row above threshold. | The cohort's policy opens once per tick (categorical action chosen from masked logits). NOT every row. The cohort effectively samples the highest-conviction action per tick. |

⚠ **Gap:** the probe counts "rows above threshold" as "opens";
the cohort opens at most ONCE per tick. So at T=0.95 the cohort
might open FAR fewer than 233 pairs — maybe 50-100. This
changes the per-day P&L math: fewer opens × £0.5-1.5 per open =
much smaller absolute day P&L.

**Implication:** the probe's "+£820/day" projection at the
loosest threshold is overstated for the cohort. Realistic
expectation is more like +£25-100/day per agent.

**Mitigation:** the success criterion (`eval_day_pnl > 0`) does
NOT pin a magnitude. Any positive P&L counts. So the gate just
needs to flip sign, not deliver £800/day.

⚠ **Action needed:** add a clarification to `purpose.md` that
the probe's day-P&L numbers are upper bounds, not cohort
expectations. The win condition is "positive", not "+£X".

**Verdict on Step 5:** OK with documentation tweak.

### Probe finding 6: the cohort's policy didn't act on direction signal in phase 13

| Fact | Plan response |
|---|---|
| Phase 13's NULL means PPO didn't credit-assign through the head. | S03 adds the hard mask — even if PPO ignores the head, the env enforces the gate. |
| The hard mask is mechanical, not learned. | Per S03 D1-D2, the mask is applied at logits level. The agent literally cannot OPEN below threshold. |
| Phase 13 had the head feeding `actor_input`. Phase 14 keeps that. | The head's output continues to feed `actor_input`. S03 ADDS a mask on top; doesn't replace the wiring. |

✓ **No gap.** S03 is the explicit redundancy for the phase-13
failure mode.

## Summary of action items from sense check

1. **S02 spec correction:** add `vol_delta_30_log` and
   `vol_delta_60_log` to match the existing `_log` pattern. 10
   features, not 8. RUNNER_DIM 115 → 125.

2. **purpose.md clarification:** the probe's per-day P&L numbers
   (+£200-£500/day projection) are UPPER BOUNDS based on
   row-counting at threshold; the cohort opens once per tick so
   real per-day P&L will be LOWER. Success criterion ("positive
   day_pnl") is unaffected.

3. **S04 watching metric:** flag direction BCE trajectory on
   arm A (gate-OFF). If flat across gens 1-2, the per-runner
   head architecture isn't extracting signal from `lstm_last`
   alone — the deeper fix (feed the head a per-runner feature
   slice, not just slot_emb + lstm_last) becomes the next plan.

4. **Risk register:** the label-vs-fill gap means realised
   mature rate may be lower than the probe's label-based
   predictions. The 35% success bar is conservative enough.

## Plan-level go/no-go

The plan is sound. Three risks are documented with mitigations:

- **Per-runner head over shared lstm_last:** mitigated by S04's
  early-gen BCE watch.
- **Feature scale:** mitigated by adding `_log` companions
  (action item 1).
- **Label vs fill gap:** mitigated by setting the success bar
  conservatively at break-even.

S03's hard mask is a load-bearing redundancy: even if PPO
fails to learn (the phase-13 mode), the env enforces the gate.

**Proceed to implementation.**
