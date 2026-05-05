---
session: phase-9-per-transition-credit / S03
phase: rewrite/phase-9-per-transition-credit
parent_purpose: ../purpose.md
depends_on: S02
---

# S03 — smoke: confirm the flag works, doesn't crash, doesn't regress

## Context

S02 wired per-transition credit behind `per_transition_credit=true`. This
session is a quick smoke only — not a statistical validation. The statistical
validation of whether per-transition credit actually improves selectivity
happens in Phase 8 S03, which runs a compact 3-arm probe testing both
mechanisms together. There's no point running a separate multi-hour Phase 9
cohort first; the combined probe is the right measurement unit.

This session's job: confirm the code doesn't crash, the byte-identity
guard holds, and `n_mature_targets` is plausible.

Wall time target: **under 10 minutes**.

## Smoke run

```
python -m training_v2.cohort.runner \
  --n-agents 2 --generations 1 --days 2 \
  --device cuda --seed 42 \
  --data-dir data/processed \
  --per-transition-credit true \
  --output-dir registry/_phase9_s03_smoke_{timestamp}
```

Check the per-update log for at least one line containing
`n_mature_targets=N` where N > 0. If N = 0 on every update across both
agents and both days, the tracking is silent — stop and investigate before
proceeding to Phase 8 S01.

## Byte-identity check

Run the same command with `--per-transition-credit false`. Extract
`policy_loss`, `value_loss`, `approx_kl` from one agent's per-update log
lines and assert they are identical to the `per_transition_credit=true`
run's first agent at the same seed.

This is the §6 regression guard from `hard_constraints.md`. If it fails,
do not continue to Phase 8 — the disabled path is broken.

## Done when

- Both smoke runs complete without error.
- `n_mature_targets > 0` observed at least once.
- Byte-identity confirmed manually from log lines.
- `lessons_learnt.md` gets one line: observed `n_mature_targets` range
  (e.g. "1–5 per mini-batch, as expected").
- Commit: `docs(rewrite): phase-9 S03 smoke - per-transition credit
  confirmed live`.

**Next: Phase 8 S01 (oracle port). No waiting.**
