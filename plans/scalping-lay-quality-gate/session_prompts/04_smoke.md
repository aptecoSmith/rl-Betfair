# 04 — Pre-flight smoke (binary PASS / FAIL)

See `session_prompts/00_autonomous_full_run.md` Phase 4 for the
full driver.

Extend `tools/smoke_race_confidence_gate.py` or write a sibling.
Four §3-style thresholds, ALL must PASS:

| Threshold | Bar |
|---|---|
| `race_qualification_rate` | ≥ 30 % |
| `legal_ratio` | ≤ 80 % |
| `expected_per_£_lay_EV` on admitted set | ≥ −£0.05 (NEW) |
| `bets_matched` (uniform-random rollout, full-day est.) | ≥ 50 |

ANY FAIL → STOP, write diagnostic. The new EV threshold is the
load-bearing one: if the gate-tuned admitted set isn't +EV
(or close), the lay-quality-gate hypothesis is wrong.

Smoke day: 2026-05-04.

Commit: `findings(scalping-lay-quality-gate): pre-flight smoke
verdict`.
