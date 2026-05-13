# 06 — Compare + verdict

See `session_prompts/00_autonomous_full_run.md` Phase 6 for the
full driver.

Read BOTH reeval JSONL files. Compute:

```
                          force_close=0     force_close=120
mean per-day pnl          £X.X              £Y.Y
median per-day pnl        £X.X              £Y.Y
profitable / 5            N                 M
locked / naked split      ...               ...
```

Write `findings.md` with:

- Both verdicts side-by-side.
- vs race-confidence-gate baseline (+£39.40/day, 3/5
  profitable).
- vs success bar (Modest > +£39/day & 3/5; Strong > +£70/day &
  4/5).
- Lessons learnt — especially whether per-bet logs + obs
  features changed close-discipline behaviour (compare
  close_signal fire rate per agent per day vs predecessor's
  ~3-4 / day).
- Recommended next plan from the README's branches.

Commit. STOP.
