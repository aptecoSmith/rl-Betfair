---
id: 01KTG90VRZNP3KVFQ4GZZX75J5
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-0fd276]
aliases: [freeze post-BC, v8 breakthrough, BC the direction head, requires_grad False after BC]
---

# Freeze the BC-pretrained head post-pretrain (the v8 breakthrough)

The phase-15 config that first **lands positive eval pnl**: BC-pretrain the calibrated head, then FREEZE
it so PPO can't corrupt the calibration. v8 (`_phase15_smoke_md`): both agents positive (+£20.04, +£39.80)
— phase-14 baseline mean was −£73; v7 (single-day BC, no freeze) was −£3 mean; v8 mean = +£30.

## What it is

The full pipeline, all six pieces load-bearing together:
1. **LayerNorm** on `direction_prob_head` so raw obs doesn't saturate the sigmoid
   ([[layernorm-for-raw-obs-heads]]).
2. **BC trains `direction_prob_head` AND `actor_head`** — the pre-existing BC pretrainer trained only
   `actor_head`; phase-15 expanded `_BC_TARGET_NAMES` to include the direction head (2000 supervised
   BCE-with-logits steps on cached labels).
3. **Multi-day pooled BC** (3–5 days) — unambiguous (1,0)/(0,1) label density jumps ~24K → 55K–110K.
4. **Freeze post-BC** — `direction_prob_head` gets `requires_grad_(False)`; the auxiliary BCE during PPO
   is effectively a no-op (gradients flow but don't update), so [[adam-ratios-away-aux-loss-weight]] no
   longer matters.
5. **Gate threshold 0.85 + warmup** — the calibrated head rarely exceeds 0.85 except on genuine
   high-confidence opens → ~10× drop in bet count (400 → 36–94); selective bets are profitable.
6. **Detach direction prob from `actor_input`** — belt-and-braces so PPO's actor pathway can't corrupt
   the frozen head.

v8 agent 1 hit 72.2% mature rate (2× the phase-14 break-even of 34.8%) and force-close 5/18 = 28% vs the
pre-phase-15 70–80% — the selective gate + calibrated head skip most "won't mature" opens, dropping the
force-close rate by 2–3×.

## Why it matters

This is the concrete payoff of the input-pathway fix ([[input-pathway-over-head-capacity]]): with the
predictor properly trained AND preserved, the gate mechanism does what the strategic thesis predicted —
selective opens, high mature rate, positive eval pnl. Freeze (not detach) is the load-bearing
preservation; detach is defensive. It validates the maturation-as-learnable, imitation-first direction;
success is read off the [[bce-trajectory-load-bearing-diagnostic]].

## Sources
- `src-0fd276` lessons_learnt.md (js_desktop:present)
