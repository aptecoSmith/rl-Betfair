---
plan: per-runner-credit
status: H1-confirmed-by-code-inspection
opened: 2026-04-26
author: investigation session
---

# Findings — volume-vs-selectivity asymmetry, H1 investigation

## TL;DR

**H1 confirmed by code inspection. Stop here per the prompt's
stop conditions.**

The fill_prob_head's BCE label is broader than even the
successor prompt suggested. The label is **not** "passive will
fill before race-off"; it is "this pair will end the day with
≥2 matched legs in `env.all_settled_bets`, regardless of HOW
the second leg arrived." Force-closed pairs and agent-closed
pairs both produce a SETTLED close leg in `bm.bets`, so they
get the same label (`1.0`) as naturally-matured pairs.

This is the proximate cause of cohort-F's
`ρ(fill_prob_loss_weight, fc_rate) = +0.469`: a
better-trained fill_prob_head is more confident at predicting
"this pair will reach 2 legs" — but ~77 % of those 2-leg
resolutions are force-closes. A confident fill_prob signal is
literally telling the actor "open this runner; the env will
end up settling it (one way or another)."

The actor cannot use fill_prob to discriminate "will mature"
from "will force-close" because the auxiliary head has no
training signal that distinguishes them.

## The evidence — three lines of code

### 1. The label assignment

`agents/ppo_trainer.py:1494–1506` (verbatim):

```python
for pair_id, (tr_idx, slot_idx) in pair_to_transition.items():
    legs = pair_bets.get(pair_id, [])
    count = len(legs)
    if count <= 0:
        continue
    if not (0 <= tr_idx < len(rollout.transitions)):
        continue
    tr = rollout.transitions[tr_idx]
    if slot_idx < tr.fill_prob_labels.shape[0]:
        tr.fill_prob_labels[slot_idx] = 1.0 if count >= 2 else 0.0
```

`pair_bets` is built from `env.all_settled_bets`
(line 1491: `for b in env.all_settled_bets`).

The label is `1.0` iff there are ≥2 settled bets with the
same `pair_id`. NO check on the second leg's nature — natural
passive fill, agent close, or env force-close all count
identically.

### 2. Force-close legs land in `bm.bets`

`env/betfair_env.py:2523–2536` (`_attempt_close`):

```python
if close_side is BetSide.BACK:
    close_bet = bm.place_back(
        runner, close_stake, market_id=race.market_id,
        max_price=self._max_back_price,
        pair_id=pair_id,
        force_close=force_close,
    )
else:
    close_bet = bm.place_lay(...)
...
close_bet.close_leg = True
close_bet.force_close = force_close
```

`place_back` and `place_lay` append the matched bet to
`bm.bets` (`env/bet_manager.py:970, 1073`). The force-close
flag is set AFTER the bet is appended; settlement
classification reads it later, but the bet is already in
`bm.bets` — and therefore in `all_settled_bets` — by then.

### 3. The cohort-F secondary correlation is consistent

Cohort-F's ρ(fill_prob_loss_weight, fc_rate) = +0.469 is
exactly the prediction of this hypothesis:

- High `fill_prob_loss_weight` → well-trained head → confident
  predictions.
- The head's training target lumps force-closes in with
  matures (this finding).
- An actor conditioning on fill_prob is therefore being
  steered toward "high-fill-prob" runners, ~77 % of whose
  successful 2-leg resolutions are force-closes.
- Higher gene → more steering on the wrong objective →
  HIGHER force-close rate. ρ = +0.469 (positive).

Cohort-O / cohort-O2 (no fill_prob → actor pathway) had no
mechanism for this confound to appear, which is why the
correlation only surfaced in cohort-F.

## What this means for H2 and H3

**Not investigated.** Per the prompt's stop conditions, H1
being the proximate cause of the broken signal is enough
to flag-and-stop. H2 (per-tick credit assignment via GAE) and
H3 (actor signal IS varying, optimisation finds global lower)
are still open as MECHANISMS by which the policy might fail
even after H1 is fixed — but they are NOT the proximate
cause of the asymmetry observed in cohort-F.

If a follow-on plan adds a "will-mature-without-intervention"
auxiliary head and the policy STILL responds with global
volume shrinkage rather than per-runner selectivity, then H2
or H3 become the next investigations. Until then, the
auxiliary's training signal is the load-bearing problem.

## Why this wasn't caught earlier

The fill_prob_head was built under
`plans/scalping-active-management/session_prompts/02_fill_prob_head.md`,
when force-close didn't exist. The 2026-04-21 force-close
plan (`arb-signal-cleanup`) added env-placed close legs that
land in `bm.bets` via the SAME placement path as agent-placed
close legs. The fill_prob label classifier was never
revisited; it still uses the count-of-legs heuristic that was
correct under the pre-force-close world (matured = 2 legs;
naked = 1 leg) but became ambiguous once the env could place
a second leg of its own initiative.

The cohort-O and cohort-O2 probes ran without fill_prob in
the actor path, so the head's mis-targeting was inert at the
gradient level — the head trained to its (broken) objective,
nothing read it. Cohort-F is the first probe that wired the
head's output INTO the actor, which is why the +0.469
correlation only became observable now.

This is consistent with the lessons-learnt note:
`plans/fill-prob-in-actor/lessons_learnt.md` Section
"Secondary: ρ(fill_prob_loss_weight, fc_rate) = +0.469"
(line 286–308) — that section flagged the wrong-direction
correlation and said "(1) is more likely and motivates a
successor investigation." This investigation confirms (1).

## Recommended follow-on (do NOT implement in this session)

A new plan, working title **`mature-prob-head`**:

1. **Add a second auxiliary head** (`mature_prob_head` or
   `nofc_prob_head`) trained on a label that SPECIFICALLY
   excludes force-closes:

   ```python
   # New label classifier — distinguishes the four outcomes
   #   class A: pair has aggressive only → naked (label 0.0)
   #   class B: pair has aggressive + close_leg with
   #            force_close=True → force-closed (label 0.0)
   #   class C: pair has aggressive + close_leg with
   #            force_close=False → agent-closed (label 1.0)
   #   class D: pair has aggressive + non-close_leg passive →
   #            naturally matured (label 1.0)
   #
   # The fill_prob_head's existing label is class A=0, B/C/D=1.
   # The new head's label is A=0, B=0, C=1, D=1. Force-close
   # is the ONLY behavioural difference.
   ```

   The classification per-pair already lives in
   `_settle_current_race` (search for `is_force_closed` —
   `env/betfair_env.py:2842`). The trainer's episode-end
   backfill loop just needs to read those classification
   bits via the `force_close` and `close_leg` flags on the
   bet objects in `pair_bets[pair_id]`, the same way it
   currently distinguishes naked from non-naked.

2. **Feed THAT head's output into actor_head**, alongside
   (or instead of) fill_prob. The architectural pathway is
   already proven by `plans/fill-prob-in-actor`; this is a
   one-line concat addition + dim bump in
   `agents/policy_network.py` (one per policy class).

3. **Architecture-hash break.** The actor_head input dim
   goes from `runner_emb + backbone + 1` to
   `runner_emb + backbone + 2` (or +1 if mature_prob
   replaces fill_prob entirely — design choice). PyTorch's
   strict-load already refuses the cross-shape via the
   existing `actor_head[0].weight` shape check. No new
   versioning machinery needed. Three policy classes.
   Same regression-guard test pattern as
   `tests/test_policy_network.py::TestFillProbInActor`
   (12 tests = 4 per class × 3 classes).

4. **Probe shape.** Same as cohort-F (12 agents,
   ppo_time_lstm_v1, 18 eps). Primary correlation:
   ρ(open_cost, fc_rate). Secondary:
   ρ(mature_prob_loss_weight, fc_rate) — under the H1
   diagnosis, this should now be NEGATIVE (well-trained
   head → fc_rate drops) where cohort-F was +0.469.

   If the new probe lands ρ(open_cost, fc_rate) ≤ −0.5,
   the volume-vs-selectivity asymmetry is fixed and
   selective-open-shaping can re-open. If it lands within
   ±0.2 again, H2 (per-tick credit assignment) becomes the
   next investigation — that's an architectural surgery
   (per-runner value head, distributional critic, or
   discrete-action reformulation) and not a single-session
   landing.

5. **Should fill_prob_head stay?** Open question. Two reads:
   - Keep both: `mature_prob` for selectivity (negative
     space "don't open this"); `fill_prob` for hedging
     decisions (the original design rationale). Costs
     one input dim, a small NLL term, and a duplicate
     auxiliary head.
   - Replace: `mature_prob` subsumes the useful information
     for the actor; the auxiliary BCE on a strict subset
     label is just a stricter teacher. Cleaner architecture
     but loses the original "will the passive fill" forecast.

   Defer this decision to the new plan's design step; the
   minimal-change path is to ADD the head, run the probe,
   then prune fill_prob if mature_prob alone produces the
   selectivity response.

## What's NOT in scope of the follow-on

Per the successor prompt's "Out of scope" section, all of:

- per-runner value head,
- discrete-action reformulation,
- env reward shape changes (`open_cost` stays at default 0.0),
- re-running cohort-O / cohort-O2 / cohort-F.

The follow-on stays small: one auxiliary head, one BCE label
swap, one architectural concat, one probe.

## Hard constraints carried over

These were inherited from the successor prompt and remain
binding for any follow-on plan:

- The fill_prob → actor_head change in
  `agents/policy_network.py` STAYS. (The new
  `mature_prob_head` either supplements or replaces; it
  does not revert the existing pathway.)
- `open_cost` gene STAYS in env-init at default 0.0.
- BC pretrain target stays unchanged.
- Entropy controller stays unchanged.

## Stop

Per the successor prompt's stop conditions:

> If a one-line investigation reveals H1 (oracle label is
> "fill" not "mature") to be the proximate cause — flag it,
> draft a follow-on plan to add a "will-mature" auxiliary
> head, and stop. Don't implement.

That condition is met. This file is the flag and the
follow-on draft. Operator's call whether to open the
`mature-prob-head` plan from here.
