---
id: 01KTFTFE2ZB011FAAW9KNVETCQ
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-042412]
aliases: [C11, direction head winner, sweep_c11]
---

# Direction-head C11 (sweep winner)

The winning direction-head architecture from the 20-variant sweep (2026-05-24):
`LayerNorm -> Linear(23, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, 2)`, trained with
**`pos_weight=1` (unweighted BCE)**.

## What it is

It combines round-2's wider+deeper winner (C9) with round-1's calibration finding (unweighted BCE).
Pareto-best on the acceptance criteria: highest mean Pearson (+0.2921, +7.4% over the C0 baseline) and
lowest mean Brier (0.1433, **-37.2%** over baseline) with AUC essentially tied with the deeper
variants. The headline is the **Brier (calibration) win, not the Pearson lift** — C11's output is a
usable probability (a "25% chance" resolves favourably ~25% of the time), so the actor's
`direction_gate_threshold` gene becomes meaningful. It is consistent across all 10 eval days (leads or
ties on 9/10). Promote via `--direction-head-manifest models/direction_head/sweep_c11` (drop the
mutually-exclusive `direction_prob_loss_weight` / `bc_direction_target_weight` genes).

## Why it matters

The recommended direction predictor head; its calibrated output is usable as a probability by the
actor's gate threshold. Built on [[pos-weight-balanced-destroys-calibration]] and
[[mlp-width-depth-scaling]], and bounded by [[direction-head-data-ceiling]].

## Sources
- `src-042412` findings.md (js_desktop:present)
