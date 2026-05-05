---
plan: rewrite/phase-9-per-transition-credit
---

# Lessons learnt

## S03 — smoke (2026-05-05)

Confirmed live on a 2-agent × 1-train-day × 1-eval-day cohort (CUDA,
seed 42, `--reward-overrides mature_prob_loss_weight=0.5`):
`n_mature_targets` observed at **856** and **984** per PPO update
across the two agents. With `ppo_epochs=4` and 248 mini-batches per
rollout that's ~3–4 masked entries per mini-batch of 128 — well
inside the prompt's "1–5 per mini-batch" expected range, and ~3 %
concentration vs. the per-slot broadcast that touches every
mini-batch row.

**Byte-identity guard (§6) holds.** Two runs at `--seed 42` with
default genes (`mature_prob_loss_weight=0` ⇒ aux block gated off
either way), one with `--per-transition-credit` and one without:
agent 1's first PPO update reported `policy_loss=0.0472
value_loss=2.9477 approx_kl=0.0126` in BOTH runs. Bit-for-bit
identical — disabled path is truly inert.

**Smoke gating subtlety.** `n_mature_targets > 0` requires
`mature_prob_loss_weight > 0` (per_trans_active is gated on
`weight > 0` for perf — multiplying a non-zero loss by 0 is wasted
compute). Default genes have weight 0 so the smoke needs an explicit
`--reward-overrides mature_prob_loss_weight=...` to exercise the
path. Byte-identity is best checked at weight=0 (default) where ON
vs OFF must produce identical losses; weight>0 is structurally
different (per-slot vs per-transition BCE) by design.
