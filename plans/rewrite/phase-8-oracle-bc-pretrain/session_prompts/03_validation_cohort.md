---
session: phase-8-oracle-bc-pretrain / S03
phase: rewrite/phase-8-oracle-bc-pretrain
parent_purpose: ../purpose.md
depends_on: S02
depends_on_external: phase-9-per-transition-credit S01+S02+S03 (smoke) GREEN
---

# S03 — compact 3-arm combined probe (~72 minutes)

## Why one probe covers both phases

Phase 9 (per-transition credit) and Phase 8 (oracle BC) both target the
same actor decision. Testing them in separate multi-hour cohorts would
require 3–4 days of wall time before seeing any signal. Instead, a single
compact probe answers both questions simultaneously:

- **Does per-transition credit produce cleaner selectivity signal than
  per-slot?** (Phase 9's question)
- **Does BC warm-start add marginal gain on top of clean label credit?**
  (Phase 8's question)

3 arms × 8 agents × 1 generation × 3 training days + 1 eval day ≈ 72
minutes total. This is the only multi-arm validation run in either phase.

## Arm design

| Arm | `per_transition_credit` | `bc_pretrain_steps` | Label |
|-----|------------------------|---------------------|-------|
| A   | false                  | 0                   | per-slot baseline |
| B   | false → **true**       | 0                   | per-transition only |
| C   | true                   | 500                 | BC + per-transition |

All three arms:
- `mature_prob_loss_weight` gene active, range `[0.5, 4.0]` (8 agents
  gives enough spread to compute ρ across the range)
- `fill_prob_loss_weight = 0.0` cohort-wide (broken label; don't confound)
- `risk_loss_weight = 0.0` cohort-wide (secondary effect)
- `open_cost = 0.0`, `force_close_before_off_seconds = 0` (clean of S06
  env overrides — test the mechanisms, not compounded env knobs)
- Same seed, same training days, same eval day across all three arms
- Same gene distribution for corresponding agent indices (arms are
  comparable agent-by-agent)

Output dirs:
- `registry/_phase8_s03_A_perslot_{timestamp}/`
- `registry/_phase8_s03_B_pertrans_{timestamp}/`
- `registry/_phase8_s03_C_bc_pertrans_{timestamp}/`

## Runner command

```
# Arm A — per-slot baseline
python -m training_v2.cohort.runner \
  --n-agents 8 --generations 1 --days 3 \
  --device cuda --seed 42 \
  --data-dir data/processed \
  --per-transition-credit false \
  --bc-pretrain-steps 0 \
  --reward-overrides fill_prob_loss_weight=0.0 risk_loss_weight=0.0 \
  --output-dir registry/_phase8_s03_A_perslot_{timestamp}

# Arm B — per-transition only
python -m training_v2.cohort.runner \
  --n-agents 8 --generations 1 --days 3 \
  --device cuda --seed 42 \
  --data-dir data/processed \
  --per-transition-credit true \
  --bc-pretrain-steps 0 \
  --reward-overrides fill_prob_loss_weight=0.0 risk_loss_weight=0.0 \
  --output-dir registry/_phase8_s03_B_pertrans_{timestamp}

# Arm C — BC + per-transition
python -m training_v2.cohort.runner \
  --n-agents 8 --generations 1 --days 3 \
  --device cuda --seed 42 \
  --data-dir data/processed \
  --per-transition-credit true \
  --bc-pretrain-steps 500 \
  --reward-overrides fill_prob_loss_weight=0.0 risk_loss_weight=0.0 \
  --output-dir registry/_phase8_s03_C_bc_pertrans_{timestamp}
```

Run arms sequentially unless you have 3 GPUs. With one GPU the total
wall time is ~72 minutes (3 × 24 minutes for 8 agents × 3 days × 1 gen).

## What to measure

For each agent, from the eval log and scoreboard:
- `mature_prob_loss_weight` (gene value, needed for ρ)
- `maturation_rate` = `(arbs_matured + arbs_closed) / (bets / 2)`
- `eval_total_reward`
- `eval_day_pnl`

From the per-update log:
- `n_mature_targets` per mini-batch (Arms B and C only; confirms Phase 9
  is active — expect 2–10 targets/mini-batch vs ~11k in Arm A)

## Success gates

### Gate 1 — Phase 9: per-transition credit improves selectivity signal

Compute Pearson ρ between `mature_prob_loss_weight` and `maturation_rate`
across all 8 agents in each arm.

| Arm | Expected ρ | Gate |
|-----|-----------|------|
| A (per-slot)        | ≈ 0 (consistent with Phase 7 S03 result) | — |
| B (per-transition)  | ≥ +0.3 | **PASS / FAIL** |

A positive ρ in Arm B means higher BCE weight reliably produces higher
maturation rate — the selectivity signal Phase 9 was designed to provide.

### Gate 2 — Phase 8: BC adds marginal gain on top of clean labels

Compare mean `maturation_rate` across Arm C vs Arm B agents.

| Comparison | Gate |
|-----------|------|
| Arm C mean_mr ≥ Arm B mean_mr + **1 pp** | **PASS / FAIL** |

The 1 pp gate tests BC's marginal contribution specifically when
per-transition credit is providing a clean signal. Without clean labels
(Arm A), BC's warm-start effect is diluted — so the comparison must be
B vs C, not A vs C.

### Gate 3 — concentration check

Mean `n_mature_targets` per mini-batch in Arms B and C ≤ 10 (well below
the 64 mini-batch size). If ≥ 50, the per-transition tracking is
inadvertently broadcasting.

### Anti-collapse gate

No agent in Arm C should finish with `bets = 0` or
`maturation_rate = NaN`. Arm C has both mechanisms active —
BC compresses entropy, per-transition BCE concentrates gradient at open
steps. If the entropy warmup handshake is misconfigured, the first PPO
update boosts alpha and collapses the actor. Any collapse here means
the warmup in Phase 8 S02 is broken.

## Analysis template

Write to `plans/rewrite/phase-8-oracle-bc-pretrain/findings.md`:

```
## S03 combined validation

### Probe design
Arm A: per-slot, no BC (baseline)
Arm B: per-transition, no BC
Arm C: per-transition + BC (bc_pretrain_steps=500)
8 agents × 1 gen × 3 days. seed=42 across all arms.

### Gate 1 — ρ(mature_prob_loss_weight, maturation_rate)
| Arm | ρ | Gate |
|-----|---|------|
| A (per-slot)       | X | — (baseline) |
| B (per-transition) | Y | ≥ +0.3: PASS / FAIL |

### Gate 2 — BC marginal contribution
| | Arm B (PTC only) | Arm C (BC + PTC) |
|---|---|---|
| mean_mr | X | Y |
| delta | — | Z pp |
| gate (≥ 1 pp) | — | PASS / FAIL |

### Gate 3 — concentration check
Arm B mean n_mature_targets/mini-batch = X (gate ≤ 10: PASS / FAIL)
Arm C mean n_mature_targets/mini-batch = Y (gate ≤ 10: PASS / FAIL)

### Anti-collapse (Arm C)
[any bets=0 or NaN agents? at what bc_pretrain_steps / mature_prob_loss_weight?]

### 4-cell summary table
| Mechanism | gen-1 mean_mr |
|---|---|
| per-slot, no BC (Arm A)       | X |
| per-transition, no BC (Arm B) | Y |
| per-transition + BC (Arm C)   | Z |

### BC wall-time cost
bc_pretrain_steps=500 per agent added approximately N seconds of wall time.

### Verdict
GREEN / AMBER / RED + one paragraph.

### Recommended follow-on
```

**Verdict criteria:**

- **GREEN**: Gate 1 PASS + Gate 2 PASS + no collapse + concentration OK
  → ship both phases; enable `per_transition_credit=true` as default and
  include `bc_pretrain_steps` in main GA gene sweep.
- **AMBER (Phase 9 only)**: Gate 1 ρ = 0.1–0.3 (directional but below
  gate) + Gate 2 PASS → per-transition credit directional but weak; try
  wider `mature_prob_loss_weight` range or more agents. BC effect is
  real.
- **AMBER (Phase 8 only)**: Gate 1 PASS + Gate 2 near-miss (0.5–1.0 pp)
  → try higher `bc_pretrain_steps` (e.g. 1000) or wider oracle coverage.
- **RED**: Gate 1 FAIL (ρ ≈ 0 in Arm B) → per-transition credit doesn't
  fix selectivity. Investigate whether `n_mature_targets` is as expected
  (if 0 everywhere, Phase 9 S02 is silently broken). Gate 2 is not
  meaningful if Gate 1 fails.

## Done when

- All three arms complete without error.
- ρ, mean_mr deltas, `n_mature_targets`, and BC wall-time recorded.
- `findings.md` written with verdict.
- `lessons_learnt.md` updated.
- Commit: `docs(rewrite): phase-8 S03 + phase-9 validation - combined
  3-arm probe [{GREEN|AMBER|RED}]`.
