# BC Label Augmentation

## Why

Round 3 (2026-05-25 `plans/bc-exit-recovery/`) confirmed two
behaviour shifts cleanly:

1. **BC pretrain lifts mat% from 1.3% → 5–6%** by relocating the
   policy's open distribution onto the oracle's preferred regions.
   Selection works.
2. **BC pretrain destroys close_signal** (cls% 31.7% → 0.8–2.2%)
   and inflates opens (134 → 230+) producing 89% force-close rate.
   Exit is broken.

The single deploy-candidate-shape cell of the whole sweep —
**E7 (pwin_back + BC=500)** — sat at opens 138, mat% 5.2%, but
**fc% still 65%**. Locked term was rock-solid (+£20 ± 1.7) across
all 4 agents, confirming the selection mechanism is real. The
day_pnl came in at -£66, dragged by -£123 of force-close losses
and dependent on naked-direction variance to bridge the gap. Anti-
lottery guard fails (locked/σ_naked = 0.32, target > 0.5).

Round 3 also confirmed that **shaped reward channels have no
behavioural authority at probe scale** (4 agents × 1 gen × 3 train
days). matured_arb_bonus, open_cost, and close_signal_bonus all
landed within probe noise across N1/N2/N3/E5/E6 — none moved
mat%, opens, or fc%. The only mechanisms that bit are those with
direct gradient authority: BC's supervised loss, pwin_back's
action-mask filter.

So the question for round 4 is structural: **how do we modify BC
itself so its supervised loss teaches the close action, not just
the open action?** Shaped partners are dead; we need the same
training-loss surface that delivered BC's selection win to also
deliver the exit behaviour.

## Diagnosis: why BC kills close

`training_v2/discrete_ppo/bc_pretrain.py:476-483` is the loss
shape:

```python
oracle_target_actions = torch.tensor(
    [action_space.encode(ActionType.OPEN_BACK, int(s.runner_idx))
     for s in batch],
    dtype=torch.long, device=device,
)
out = policy(obs_t)
oracle_ce = F.cross_entropy(out.logits, oracle_target_actions)
```

Every BC sample's target is OPEN_BACK (or OPEN_LAY via the
optional direction-target map at lines 484-524). The
cross-entropy against a single positive action class **pushes that
class's probability up and, via softmax normalisation, pushes
every other class's probability DOWN**. After 200+ BC steps, the
CLOSE action's logit has been depressed across all contexts — not
because BC explicitly trained it to zero, but because BC never
gave CLOSE any positive gradient anywhere.

The behavioural collapse follows mechanically:

- Sample (T, R) → BC target OPEN_BACK_slot(R) → softmax of
  unbounded actor_head logits squeezes everything else (CLOSE,
  NOOP, REQUOTE) toward zero.
- Across 200+ steps on diverse obs vectors, the policy learns
  "OPEN_BACK is the right answer on oracle-positive obs" — but
  also, by softmax side-effect, "every OTHER action is the wrong
  answer always".
- At rollout time the policy then almost never picks CLOSE
  regardless of the open-pair state in the obs — because the
  CLOSE logit has been depressed below the OPEN_LAY / OPEN_BACK
  logits on every observation.

Three deficits in the current BC pool cause this:

1. **No negative-open samples.** BC never sees an obs and a
   target "no, NOOP here" — so the NOOP class's softmax share
   has no positive gradient anywhere.
2. **No close-positive samples.** BC never sees an obs with a
   target "CLOSE this pair" — so the CLOSE class's softmax share
   has no positive gradient anywhere.
3. **Position dims are zero on every sample.** `arb_oracle.py:
   158` constructs `zero_position = np.zeros(position_dim)` and
   uses it on every sample. The policy at BC time never observes
   the "agent has an open pair" state, so even if we added
   close-positive samples we'd be training "close on a
   zero-position obs" — which is meaningless at rollout time.

Each deficit is independently fixable.

## Label-class menu

Before picking phases, lay out the full menu of new label classes
we *could* add. Each row is independently implementable; the plan
below selects a subset and justifies the choice.

| ID  | Sample shape                                  | Obs requirement       | Target class    | What it teaches the policy                          | Eng cost |
|-----|-----------------------------------------------|-----------------------|-----------------|-----------------------------------------------------|----------|
| L0  | (existing) oracle-positive open, back-first   | fresh agent (zero pos)| OPEN_BACK       | "open back here, this is profitable"                | (current) |
| L1  | (existing) oracle-positive open, lay-first    | fresh agent (zero pos)| OPEN_LAY        | "open lay here" — from direction-target overlay     | (current) |
| **L2**  | **oracle-negative tick × runner**         | fresh agent (zero pos)| **NOOP**        | "DON'T open here; most ticks are not opportunities" | **cheap**    |
| L3a | open pair that WILL force-close, mid-life     | populated pair-state  | CLOSE_slot      | "close this pair — forward-walk says it won't mature" | high     |
| L3b | open pair, ANY life-stage, forward-walk says close-now > hold-future | populated pair-state  | CLOSE_slot      | "close when spread has drifted in your favour"      | high     |
| **L4**  | **open pair that WILL mature naturally, mid-life** | **populated pair-state** | **NOOP**       | "HOLD this pair — the spread will fill naturally"   | **high** |
| L5  | open pair, spread drifted unfavourably         | populated pair-state  | REQUOTE_slot    | "re-quote your passive at a tighter target"          | high     |
| L6  | lay-first oracle-positive (separate scan)      | fresh agent           | OPEN_LAY        | enriches L1 with structurally-different opportunities | medium   |

Notes on the menu:

- **L2 is the cheapest** because it reuses the existing obs
  vector with zero position dims. Every tick × runner *not* in
  the L0/L1 positive set is a valid L2 candidate. Subsample to
  prevent overwhelming the positive signal.
- **L3a vs L3b** is the close-decision design choice. L3a piggy-
  backs on the strict `mature_prob` label (per-pair, exists in
  env memory at training time) — for any pair the env recorded
  as force-closed, label CLOSE positive at every tick of its
  life. L3b is the principled forward-walk: at each tick of the
  pair's life, compute hold-vs-close P&L and label CLOSE only
  when close-now strictly dominates. L3b is more accurate but
  computationally heavier (per-tick ladder lookahead).
- **L4 is the symmetric pair to L3.** If we ONLY add CLOSE
  labels, BC pushes CLOSE mass up everywhere — including on
  pairs that would have matured if held. L4 teaches NOOP (hold)
  on natural-mature pairs so the policy distinguishes the two
  pair states.
- **L5 is more speculative** — the env already supports REQUOTE
  but the policy rarely picks it. Adding REQUOTE labels would
  require oracle logic for "when is re-quoting better than
  holding or closing", which is non-trivial. Deferred until
  L3/L4 land and we see what's left.
- **L6 is orthogonal** — it just enriches the open side. The
  current oracle scan is back-first only; a parallel lay-first
  scan would add ~50-100% more positive samples to L1. Worth
  doing but separate from the close-side problem.

## Plan

Two-phase plan, each phase has its own validation probe so we can
isolate which mechanism bites.

### Phase A — Adds L2 (negative-open NOOP)

Cheapest engineering. Targets deficit #1 only. Single new label
class.

**Sample generation.** Walk pre-race ticks the same way `scan_day`
already does. For each `(tick_index, runner_idx)` that is NOT in
the existing oracle-positive set, emit a `NegativeOracleSample`
with the same obs vector and `target = NOOP`. Subsample to
roughly match the positive-sample count (so the loss isn't
overwhelmed by negative gradient).

**Loss shape.** Concat the two pools into one BC pool, target
class per-sample. The existing cross-entropy code already
handles arbitrary target classes — we just feed it NOOP indices
on negative samples and OPEN_BACK / OPEN_LAY on positive samples.

**Mechanism it fixes.** The softmax normalisation now has
positive gradient on NOOP across the obs distribution. The
post-BC policy's NOOP mass is preserved → fewer speculative
opens at rollout time → opens drop back from 230 toward 100-180.

**What it does NOT fix.** Close action still has no positive
gradient anywhere. Close mass still decays via softmax. So this
should reduce over-opening but probably won't fully fix fc% —
because every surviving open still gets force-closed for the
same reason as PC3.

### Phase B — Adds L3a + L4 (paired close / hold labels)

Larger engineering. Targets deficits #2 and #3 together. Adds two
new label classes simultaneously because they're a structural pair
— close labels without matching hold labels would push CLOSE mass
up indiscriminately.

**Why L3a and not L3b for the close target.** L3a (mature_prob
negation) reuses the existing strict per-pair label and just spreads
it across the pair's life-ticks; L3b (forward-walk per-tick close-
vs-hold P&L) is more accurate but requires per-tick ladder lookahead
during the oracle scan. L3a is the cheaper first cut. If Phase B
validation shows the policy closes too aggressively (mat% drops
because pairs that would have matured are being closed early), we
upgrade to L3b in a Phase B' iteration.

**Why L4 (HOLD) is non-negotiable in Phase B.** Without L4, BC pushes
CLOSE mass up on every (tick, pair-state) sample. The policy then
closes everything — including would-be-mature pairs. L4 trains the
discrimination: "this pair-state will mature, leave it alone".

**Sample generation.** For each oracle-positive `(T_open, R)`,
walk forward in the same race. At each subsequent pre-race tick
`T_close ∈ (T_open, T_off - 120s)`:

1. Synthesize the obs vector with the position dims populated as
   if the agent had opened at `T_open` (back leg matched at the
   then-LTP, passive lay placed at `T_open`'s arb_spread).
2. Decide the close label by forward-walking the ladder:
   - If holding to `T_off - 120s` would result in force-close
     (passive never fills) → label CLOSE at `T_close` with
     weight `(T_close - T_open) / window`. Earlier-tick weight 0,
     last-tick weight 1.
   - If holding would result in natural maturation → label
     NOOP at `T_close` (hold).
3. Emit `CloseDecisionSample` rows accordingly.

This produces a large pool — every oracle-positive open spawns
~30-60 close-decision samples. Subsample aggressively.

**Loss shape.** Add a third CE target type to bc_pretrain.py.
The aggregator already supports multiple target types via the
direction-target map; CLOSE / NOOP follow the same pattern with
target indices encoded via
`action_space.encode(ActionType.CLOSE, slot)` etc.

**Mechanism it fixes.** Both CLOSE and NOOP now have positive
gradient on samples where the obs reflects an open pair. The
post-BC policy's close logic is GROUNDED — it has learned which
pair states should be closed early and which should be held.

**Engineering scope:**
- Extend `OracleSample` (or add a sibling type) with `target_action_type` field.
- Extend the obs-synthesis step in `scan_day` to populate position dims.
- Extend `build_oracle_target_map` / equivalent in bc_pretrain.py
  to emit the right action target per sample type.
- Re-scan oracle caches for the 3 training days. ~1h offline.

### Validation experiments

Both phases get a probe cell against the E7 reference:

| cell    | flags                                                      |
|---------|------------------------------------------------------------|
| **F0 (ref = E7)** | pwin_back=0.20 + BC=500 (already in registry)  |
| **F1** (Phase A)  | pwin_back=0.20 + BC=500 + NOOP-augmented BC pool |
| **F2** (Phase B)  | pwin_back=0.20 + BC=500 + close-augmented BC pool |
| **F3**            | pwin_back=0.20 + BC=500 + BOTH augmentations    |

4 cells × 25 min = ~1.7h. Compares directly against E7 on the
same train/eval days.

## Acceptance criteria

Same as round 3 (`plans/bc-exit-recovery/purpose.md`):

| metric          | target                  |
|-----------------|-------------------------|
| opens/day       | 100 – 180               |
| mat%            | ≥ 5%                    |
| fc%             | ≤ 50%                   |
| day_pnl         | > -£100 (beat C2)       |
| locked/σ_naked  | > 0.5                   |

E7 already hits 3/5 (opens 138 ✓, mat% 5.2% ✓, day_pnl -£66 ✓,
fc% 65% ✗, locked/σ_naked 0.32 ✗). The augmentation succeeds
if **F1 or F2 or F3** lifts fc% under 50% AND keeps the other
three criteria. Headline win = F3 hits all 5.

### Per-phase expected signature

**Phase A (F1) — if NOOP augmentation works:**
- Opens drop (toward 100-130 range).
- mat% holds or slightly rises.
- fc% drops modestly (because fewer pairs are opened to begin with).
- cls% might rise as the softmax pressure on close eases — but
  not guaranteed because close still has no positive gradient.

**Phase B (F2) — if close augmentation works:**
- cls% rises substantially (toward 20-30% range).
- fc% drops substantially (toward 30-40% range).
- Opens might stay elevated unless paired with NOOP samples.
- day_pnl improves.

**Phase A+B (F3) — if both compose:**
- Opens in the 100-180 band (NOOP samples suppress over-opening).
- mat% at or above 5% (BC's selection lift preserved).
- cls% 15-25% (close-positive samples ground close behaviour).
- fc% under 50% (close substitution + over-opening reduction).
- day_pnl > -£50.

### Failure modes

- F1 reduces opens but mat% drops with them — NOOP samples
  diluted the open-positive signal. Resample with higher
  positive:negative ratio.
- F2 raises cls% but mat% drops — close labels too aggressive
  (too early in pair life). Tune the weight function.
- F3 doesn't improve over F2 — composition interferes. Run them
  separately as deploy options.

## Hard constraints

- Same train days (`2026-04-06, 2026-04-08, 2026-04-09`) and eval
  days (`2026-04-10, 2026-04-17, 2026-04-21, 2026-05-03, 2026-05-06`)
  as rounds 1-3.
- pwin_back=0.20 always on (E7's confirmed env lever).
- BC=500 always on (round 3's partner-reference dose).
- Direction gate off (D-cells decided).
- All 3 predictors loaded; lean obs.

## Estimated effort

- **Phase A code + tests:** ~3-4h.
- **Phase A oracle re-scan:** ~1h offline.
- **Phase B code + tests:** ~1-2 days. The position-dim
  synthesis is the bulk of the work; needs to mirror the env's
  `_process_action` open path so the synthetic obs matches what
  the policy would see at rollout time.
- **Phase B oracle re-scan:** ~2-3h offline (much bigger pool).
- **Validation cells:** ~1.7h wall.

Phase A is plausibly a one-day deliverable. Phase B is a 2-3 day
deliverable. Recommended sequencing: ship Phase A, validate, then
decide whether B is needed.

## Out of scope (queued for follow-up plans if needed)

Two label classes from the menu that are NOT in Phase A or B,
plus engineering work that's adjacent but separate:

- **L3b (forward-walk per-tick close-vs-hold P&L)** — upgrade
  path from L3a if Phase B's mature_prob-negation close target is
  too coarse (i.e., the policy closes good pairs because L3a
  spreads the close signal across the entire life of any pair
  the env happened to force-close). Adds per-tick ladder
  lookahead during the oracle scan; ~1 day extra eng.
- **L5 (REQUOTE labels)** — the env supports a REQUOTE action
  that the policy rarely picks. Oracle logic for "when is re-quoting
  better than holding or closing" is non-trivial — needs forward-
  walk on hypothetical re-quote-then-fill outcomes. Defer until
  Phase A/B land and we see whether REQUOTE under-use is still
  the right problem to fix.
- **L6 (lay-first oracle scan)** — the current oracle scan in
  `arb_oracle.py` is back-first only. A parallel lay-first scan
  would add ~50-100% more positive open samples to L1. Worth
  doing for general oracle quality but separate from the close-
  side problem this plan is solving.
- **Modifying BC's loss formulation.** This plan only adds new
  samples; the existing cross-entropy code is unchanged.
- **Re-training the frozen C11 direction head.** It's already
  load-bearing and unrelated.
- **Pair-lifecycle-aware shaped reward.** Already confirmed dead
  lever at probe scale in round 3.
- **Gradient-knob pinning (lr, gae_lambda).** Still owed from
  round 3's deferred Group D. Independent of this plan.
- **Longer training runs (multi-generation BC).** If Phase A and
  B both fail at 1 gen, the next move is to test whether
  multi-generation PPO can recover close behaviour on its own
  given BC's selection lift. Separate plan.

## Related work

- `plans/bc-exit-recovery/purpose.md` — round 3 design + results.
- `plans/oracle-alignment-investigation/findings.md` — close-penalty
  + BC pretrain results that motivated this plan.
- `plans/arb-curriculum/session_prompts/01_oracle_scan.md` — the
  original oracle scan design that this plan extends.
- `training_v2/arb_oracle.py` — oracle sample generator.
- `training_v2/discrete_ppo/bc_pretrain.py:455-540` — BC loss shape
  (the function this plan extends).
