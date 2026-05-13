# Autonomous run log — scalping-lay-quality-gate

Per-iteration log of the autonomous run. Each iteration writes
one entry using the template at the bottom of
`session_prompts/00_autonomous_full_run.md`.

## 2026-05-13 — Phase 0, iteration 1

**State entering iteration:** plan folder existed with only
`session_prompts/00_autonomous_full_run.md`; predecessor
`scalping-race-confidence-gate` complete (mean +£39.40/day,
3/5 profitable held-out).
**Work done:**
- Verified no active cohort process is running
  (`Get-Process python` showed only the predecessor watcher).
- Created `plans/scalping-lay-quality-gate/README.md`.
- Created `plans/scalping-lay-quality-gate/hard_constraints.md`.
- Created `plans/scalping-lay-quality-gate/master_todo.md`.
- Created `session_prompts/01_probe.md` through
  `06_compare_and_verdict.md`.
- Created this `autonomous_run_log.md`.
**Tests run:** none (scaffolding only).
**Decisions made:**
- Inherit `race_confidence_threshold = 0.50` (predecessor's
  smoke-PASS value).
- Phase 1 will set `predictor_p_win_lay_threshold` and
  `lay_price_max` empirically.
- Two reeval watchers (force_close=0 + 120) per
  `memory/project_force_close_train_vs_deploy.md`.
- Phase 2a and 2b will be committed separately so variables
  are separable for analysis.
**Outstanding for this phase:** commit scaffold with message
`plan(scalping-lay-quality-gate): scaffold next stack-on plan`.
**Next iteration's focus:** Phase 1 — re-run
`tools/probe_lay_outcome_distribution.py` on 2026-04-28/29/30
to set Phase 3 defaults.

## 2026-05-13 22:31 — Phase 1, iteration 1

**State entering iteration:** Phase 0 scaffold committed
(`d8edc53`).
**Work done:**
- Ran `python -m tools.probe_lay_outcome_distribution
  --days 2026-04-28 2026-04-29 2026-04-30
  --race-confidence-threshold 0.50 --lay-threshold 0.40
  --device cuda` (output `/tmp/layq/probe_2026-04-28_30.txt`).

**Probe output (key tables):**

OVERALL: 1173 gate-eligible tuples, EV/£ stake = **−£0.0350**
(matches 2026-05-13 baseline; profile has NOT shifted).

LAY-PRICE BUCKETS:

| Bucket | n | lay_winrate | EV/£stake | avg_loss_when_lost |
|---|---:|---:|---:|---:|
| 2-5 | 77 | 80.5% | +£0.1727 | −£3.25 |
| 5-10 | 287 | 86.4% | −£0.0293 | −£6.57 |
| 10-20 | 310 | 93.2% | +£0.0097 | −£13.62 |
| 20-50 | 308 | 95.5% | **−£0.3896** | **−£29.57** |
| > 50 | 191 | 99.0% | +£0.3717 | −£59.00 |

PREDICTOR-CALIBRATION CHECK:

| pwin band | n | realised | avg_lay_price | EV/£stake |
|---|---:|---:|---:|---:|
| 0.00-0.10 | 866 | 5.9% | 56.31 | −£0.0801 |
| 0.10-0.20 | 205 | 10.7% | 7.91 | **+£0.3741** |
| 0.20-0.30 | 80 | 20.0% | 6.94 | **−£0.7362** |
| 0.30-0.40 | 22 | 9.1% | 8.15 | +£0.4773 |

**Decisions made (Phase 3 defaults locked):**

- `lay_price_max = 20` — drops the 20-50 leverage trap
  (n=308, EV −£0.39). The 5-10 bucket sits at −£0.029
  which is within the §3 −£0.05 bar; the > 50 bucket
  (n=191, EV +£0.37) looks attractive but rests on a 99%
  win rate at avg leverage £59 per loss — a 1% shift in
  win rate would crater it. Capping at 20 keeps the
  reliable buckets (2-5, 5-10, 10-20) and drops the
  documented bleed bucket.
- `predictor_p_win_lay_threshold = 0.20` — drops the
  0.20-0.30 calibration hole (n=80, EV −£0.74) and the
  0.30-0.40 outlier (n=22, too small to trust). At
  threshold 0.20 the blended EV is approximately
  (866 × −£0.08 + 205 × +£0.37) / 1071 ≈ **+£0.006/£**
  across n=1071 (down from 1173).

**Sanity check on the "lowest bucket where EV ≥ 0 and
n ≥ 100" rule:**
- threshold 0.10 → admits 866 at EV −£0.08 → negative
- threshold 0.20 → admits 1071 at EV ~+£0.006 → positive ✓
- threshold 0.30 → admits 1151 at EV −£0.046 → back
  negative
- threshold 0.40 (current) → admits 1173 at EV −£0.035

→ 0.20 is uniquely optimal.

**Cross-intersection note:** Phase 4 smoke will measure the
EV of the actual admitted set (pwin ≤ 0.20 AND
lay_price ≤ 20). The per-bucket marginals above suggest
this set will sit at EV ≥ 0; the smoke verifies.

**Tests run:** none (probe + analysis only).
**Outstanding for this phase:** commit Phase 1 with
`autonomous_run_log.md` updated + probe output reference.
**Next iteration's focus:** Phase 2a — wire per-bet logging
during training-eval rollouts.
