# Autonomous run log — scalping-tight-naked-variance

Per-iteration log of the autonomous run. Each iteration appends one
entry using the template at the bottom of
`session_prompts/00_autonomous_full_run.md`.

The plan exits at the end of Phase 1d (verdict commits). Phase 2 is
operator-gated per `hard_constraints.md §22`.

<!-- iterations append below this line -->

## 2026-05-14 20:48 — Phase 0+1, iteration 1 (BLOCKED on 1c)

**State entering iteration:** plan scaffolded, no work yet.
**Work done:**
- Phase 0 complete: `tools/build_naked_variance_report.py` +
  `tests/test_naked_variance_report.py`. Five tests pass. Committed
  as `83a21a6`.
- Phase 1a complete: verified `reevaluate_cohort.py` already accepts
  `--filter-agent-ids`. No code change.
- Phase 1b complete: ran the report on raceconf + layq. Wrote
  `naked_variance_report.csv` and `phase1_top5_union.txt` into each
  cohort dir. 14 unique agents per cohort.
- Phase 1c BLOCKED. All 14 raceconf agents failed weight load on
  the first eval day with `input_proj.0.weight` shape mismatch
  ([64, 504] checkpoint vs [64, 574] current). Verified directly:
  a saved checkpoint has `obs_schema_version=8`; current
  `env.betfair_env.OBS_SCHEMA_VERSION` is 9 (commit `045174d`,
  today at 14:54). Stop condition #3 fired.
- Phase 1d: wrote `phase1_verdict.md` documenting the block and
  presenting three remediation options for operator decision.

**Tests run:** `pytest tests/test_naked_variance_report.py -v` → 5/5
pass.
**Decisions made:** STOPPED at stop condition #3 per the autonomous
prompt. Did NOT auto-revert `045174d`; did NOT auto-create a
worktree (operator decision per `hard_constraints.md §22` spirit).
Did NOT launch Phase 2.
**Outstanding for this phase:** held-out reeval (8 jobs) once
operator picks a remediation path.
**Next iteration's focus:** wait for operator input.

## 2026-05-15 12:13 — Phase 1c complete, Phase 1d done

**State entering iteration:** Phase 1c reeval chain (8 jobs) launched
from per-cohort eval worktrees at 22:02 BST 2026-05-14.
**Work done:**
- All 8 reevals completed cleanly between 22:02 and 11:33 BST,
  14 rows per JSONL. Total wall: 13h 31m.
- Wrote `C:/tmp/phase1_analyze.py` to roll the 8 JSONLs into the
  per-selector × per-cohort × per-window × per-fc verdict table.
- Generated `phase1_verdict_table.csv` (40 cells) and amended
  `phase1_verdict.md` with the headline numbers, full per-cell
  table, and band verdict.
**Tests run:** None this iteration; analysis only.
**Decisions made:** All 20 fc-paired cells land in REGRESSION
territory (fc=120 mean < 0). No band cleared. Phase 1 hypothesis
(variance-aware selection over existing populations surfaces
deployable agents) is REJECTED on held-out data. Phase 2 (retrain
with variance-aware reward + fc=120 in training) is the only
remaining path per the plan; recipes documented in verdict.md.
**Outstanding for this phase:** verdict + analysis CSV need
committing.
**Next iteration's focus:** Phase 1 EXITS here. Operator gate per
hard_constraints.md s22 — operator picks Phase 2A / Phase 2B /
BOTH / retire.

## 2026-05-16 07:48 — Phase 2A complete

**State entering iteration:** Cohort _predictor_SCALPING_tnv_raceconf_
1778852093 complete (96/96 agents at 02:34). Phase 3 reeval (4 jobs)
underway since 02:40.
**Work done:**
- All 4 Phase 3 reevals completed cleanly at 07:30. 10-agent union
  top-5 reevald on 2 windows x 2 fc settings = 4 JSONLs.
- Analysed results vs Phase 1 explicit null:
  - new fc=0 top-5 mean PnL: -£9.73 vs null -£40.50 = +£30.77 BETTER
  - new fc=120 top-5 mean PnL: -£20.09 vs null -£16.92 = tied
  - Best agent 32ed9e32 (gen 0, beta=0.00133): -£0.76/day fc=120 new,
    4/7 profitable days, naked_std £101 (Modest band's std ceiling).
- Wrote `findings.md` with full verdict, in-sample gen trend, and
  three follow-on recipes (default: re-run Phase 2A WITH fc=120 in
  training).
**Tests run:** None this iteration; analysis only.
**Decisions made:** Phase 2A complete. No band cleared but variance-
penalty mechanism CONFIRMED to produce selection pressure in-sample
(46% max-span reduction gen 0-3, beta_med 0.016 -> 0.030). The
train-vs-deploy fc asymmetry from the layq predecessor remains —
operator's decision to defer fc=120 in training was the limiting
factor.
**Outstanding for this phase:** verdict + log need committing.
**Next iteration's focus:** plan exits here; operator decides
follow-on (Option A: re-run with fc=120; Option B: narrow sweep;
Option C: deploy 32ed9e32 directly).

## 2026-05-17 11:09 — tnv2 reeval + tnv3 training heartbeat

**State entering iteration:** Two background jobs running. tnv2
top-10-by-in-sample-day_pnl held-out reeval (4 jobs, launched
2026-05-17 09:52, partial results visible). tnv3 cohort training
(96-agent, day_pnl_per_std selector, fc=120 in training, β∈[0,0.10],
early-stop-patience=3, launched 2026-05-17 09:50).
**Work done:** read both status panels + launcher logs only; no code
changes.
**Tests run:** None.
**Observations:**
- tnv2 fc=0 oldwindow finished (10/10). 1/10 profitable: only
  `abd438ea` at +£24.13/day. Other 9 between −£146 and −£425/day.
  4c217d70 specifically: in-sample +£19 → held-out −£146/day fc=0
  (1/3 profitable). Predicted phenotype-collapse confirmed.
- tnv2 fc=120 oldwindow 6/10 done; all 6 negative (−£97 to −£251/day).
- tnv3 cohort: 4/96 agents (gen 0) done; mean_fc_pnl −£83/day
  already showing force-close cost still dominates early gen 0
  (mean_locked +£84, mean_naked −£45, so fc cost is the
  per-day-pnl-killer). Naked range 250–374 (median 264). Too
  early to tell whether day_pnl_per_std selector will shift the
  GA — gen 1+ will be the first signal.
- Cohort ETA per panel: 27.6h remaining (1654 min) — slower than
  original 20:00-Mon estimate; revised ETA ~Tue afternoon.
**Decisions made:** No action this heartbeat. Don't relaunch tnv3.
Don't write tnv2 verdict yet (need fc=120 oldwindow + both newwindow
jobs). Continue monitoring.
**Outstanding for this phase:** tnv2 reeval (~3-4h to finish all 4
jobs); tnv3 cohort (~28h); tnv2 verdict + tnv3 verdict + EXPERIMENTS.md
scaffold queued.
**Next iteration's focus:** check tnv2 reeval progress; if fc=120
oldwindow done and at least one newwindow job started, start drafting
the tnv2 verdict.

## 2026-05-17 11:40 — tnv2 fc=120 oldwindow complete; clear regression

**State entering iteration:** scheduled heartbeat fired. tnv2 fc=120
oldwindow finished at 11:27:58. fc=0 newwindow started 11:27:58
(agent 1/10 in flight). tnv3 at 5/96.
**Work done:** read full fc=120 oldwindow table; aggregated headline
numbers; updated this log only. Also seeded `plans/EXPERIMENTS.md`
with 6 historical entries + tnv2/tnv3 placeholders earlier this
session (uncommitted — bundle with tnv2 verdict).
**Tests run:** None.
**Observations:**
- **tnv2 fc=120 oldwindow: 0/10 profitable.** Per-agent pnl/day:
  −97, −224, −219, −251, −176, −194, −243, −222, −207, −267. Mean
  ≈ **−£210/day**. layq null is −£16.92/day → tnv2 is **−£193/day
  WORSE than null** at deployment fc. This is a REGRESSION band
  per the prompt's verdict table, not just "no improvement".
- Mechanism: locked floor RAISED to +£160 to +£213/d (avg +£180,
  vs layq's +£114). But pair counts climbed even faster — 379–568
  bets/day vs layq's ~200/day — and fc cost is now −£126 to −£346
  per day (vs layq's −£69). Net: the locked productivity gained
  was eaten 2–3× by the new fc cost. Training-at-fc=120 changed
  the locked phenotype but didn't teach selectivity at the open
  decision.
- tnv2 fc=0 oldwindow: 1/10 profitable, mean ~−£170/d (also worse
  than its null of −£40.50/d).
- tnv3 gen 0 (5/12 in this gen by my reading; cohort is 96 total):
  mean_fc_pnl still −£86. New agent `96027f17` has the tightest
  span yet (133.6) — but still −£63/d. Too early for the
  selection-pressure signal; that lands at gen 1.
**Decisions made:** Don't commit anything yet. Wait for both
newwindow jobs to finish before drafting the verdict — the verdict
hinges on the 7-day forward number for deployment comparison,
not the 3-day oldwindow. Don't touch tnv3.
**Outstanding for this phase:** fc=0 newwindow + fc=120 newwindow
(~3-4h to both). tnv3 gen 0 finish (~5h to all 12 agents).
**Next iteration's focus:** check newwindow progress; if at least
one is complete, start drafting tnv2 verdict.

## 2026-05-17 12:12 — fc=0 newwindow surfaces abd438ea positive signal

**State entering iteration:** scheduled heartbeat. tnv2 fc=0
newwindow at 4/10 (in flight on 4c217d70, day 2026-05-09). tnv3 at
7/12 gen 0.
**Work done:** read status panels only.
**Tests run:** None.
**Observations:**
- tnv2 fc=0 newwindow first 3 agents done:
  - **abd438ea: +£115.89/d, 5/7 profitable, range −£353 to +£484.**
    Same agent that was +£24/d on fc=0 oldwindow (the only positive
    there too). Two windows × 8/10 profitable days × consistently
    positive mean → this is starting to look real, but variance is
    enormous (min −£353/d).
  - Remaining 2 agents (f3a53c16, 451ecdd8) likely negative based
    on partial-day rows visible.
- tnv3 gen 0 (7/12): new best agent `28d95a3d` at −£20/d, span
  204.9, locked +£101. Still no agent positive on day_pnl in gen
  0. mean_fc_pnl steady at −£87. Whether day_pnl_per_std selector
  actually shifts the GA toward fc-light agents is the gen-1
  question.
**Decisions made:** Don't draft verdict yet — wait for fc=0
newwindow to finish so the abd438ea signal can be confirmed at
job completion (and fc=120 newwindow numbers land).
**Outstanding:** fc=0 newwindow (~80 min); fc=120 newwindow (~120
min). tnv3 gen 0 finish (~90 min).
**Next iteration's focus:** check fc=0 newwindow completion;
abd438ea single-agent signal investigation.

## 2026-05-17 12:43 — fc=0 newwindow at 6/10; tied with layq null

**State entering iteration:** scheduled heartbeat. tnv2 fc=0
newwindow at 6/10 (d7d02839 in flight, day 2026-05-07). tnv3 at
9/96.
**Work done:** read full fc=0 newwindow per-agent table; computed
mean across the 6 done.
**Tests run:** None.
**Observations:**
- tnv2 fc=0 newwindow 6/10 done. Mean per-day pnl = −£35/d
  (vs layq null −£40.50/d → ~£5/d better, essentially TIED).
  Per-agent: abd438ea +£116 (5/7), f3a53c16 +£134 (3/7),
  451ecdd8 −£250 (1/7), 4c217d70 −£189 (3/7), 42072f65 +£82 (4/7),
  d7d02839 −£101 (3/7). 3/6 agents positive; 19/42 days positive.
- Per-day variance is enormous on the positive agents: f3a53c16
  has individual days from −£236 to +£704; 4c217d70 from −£758 to
  +£525. The mean-positive agents are catching ±£500–700/day
  naked swings on lucky days. Not a stable signal.
- tnv3 9/96 (gen 0 only). New best `d50e5e95`: −£14/d, span 133,
  locked +£105, fc −£80. Even the tightest-variance highest-locked
  gen-0 agent eats £80/d on force-close. That's the operative
  ceiling on day_pnl in gen 0; whether selection in gen 1+ can
  break it is the question.
**Decisions made:** Wait for fc=0 newwindow finish (~50 more min)
+ fc=120 newwindow (~140 min) before verdict. The fc=120 newwindow
is the deployment-relevant cell.
**Outstanding:** 4 more agents on fc=0 newwindow; full fc=120
newwindow run.
**Next iteration's focus:** check fc=0 newwindow completion and
whether fc=120 newwindow has started.

## 2026-05-17 13:14 — fc=0 newwindow at 9/10; verdict shape solidifies

**State entering iteration:** heartbeat. tnv2 fc=0 newwindow 9/10
(6eb5dde3 in flight, day 2026-05-12). tnv3 at 10/96.
**Work done:** read agents 7–8 from fc=0 newwindow; recomputed mean.
**Tests run:** None.
**Observations:**
- fc=0 newwindow 8/10 done. Mean = **−£49/d** (vs null −£40.50/d
  → marginally WORSE than null, not better). 3/8 agents positive
  (abd438ea, f3a53c16, 42072f65). The 6-agent +£5/d advantage was
  noise; agents 7 (d41abbde, −£73/d) and 8 (e7d70a0f, −£111/d)
  dragged the mean back below null.
- Verdict shape across the 3 done cells:
  - fc=0 oldwindow: 1/10 prof, mean ~−£170 → REGRESSION on null −£41
  - fc=120 oldwindow: 0/10 prof, mean ~−£210 → REGRESSION on null −£17
  - fc=0 newwindow: 3/8 prof, mean −£49 → tied / slightly worse than null
- fc=120 newwindow (the most deployment-relevant) starts when fc=0
  finishes (~13:30) and runs ~140 min → ~15:50. Based on the
  oldwindow pattern (fc=120 is ~£170 worse than fc=0 because of
  surprise fc cost the agents weren't trained against… wait,
  they WERE trained at fc=120; but pair counts climbed so fc still
  eats them — see 11:40 entry), I expect fc=120 newwindow to be
  −£200 to −£250/d. Will confirm at next reading.
- tnv3 still 10/96 gen 0. New stable best: `d50e5e95` at −£14/d.
**Decisions made:** Verdict drafting can begin after fc=0 newwindow
done. But fc=120 newwindow numbers are required to write the
"Phase 1d / verdict" cell that matters most — wait for that before
committing.
**Outstanding:** fc=0 newwindow last 1-2 agents; full fc=120
newwindow.
**Next iteration's focus:** by 13:44 fc=0 newwindow should be done,
fc=120 newwindow ~15 min in.

## 2026-05-17 13:45 — fc=0 newwindow done; gen 0 of tnv3 done

**State entering iteration:** heartbeat. fc=0 newwindow DONE 13:26;
fc=120 newwindow STARTED 13:26 (~18 min in, agent 1 abd438ea done).
tnv3 gen 0 complete (12 agents).
**Work done:** pulled full agent table for fc=0 newwindow; aggregated.
**Tests run:** None.
**Observations:**
- **tnv2 fc=0 newwindow 10/10 done.** Mean = **−£50.53/d** vs layq
  null −£40.50/d → tnv2 is **−£10/d WORSE than null** on the
  deployment-realistic 7-day forward window. 3/10 agents profitable.
  Day-level 30/70 (43%).
- **tnv2 fc=120 newwindow first agent**: abd438ea −£131.16/d, 0/7
  profitable. Trajectory matches the oldwindow pattern; expect
  −£200 to −£250/d mean when all 10 done.
- **Verdict shape now clear: tnv2 is REGRESSION across all 4 cells**:
  - fc=0 old: mean ~−£170, 1/10 prof, vs null −£41 → regression
  - fc=120 old: mean ~−£210, 0/10 prof, vs null −£17 → regression
  - fc=0 new: mean −£50.53, 3/10 prof, vs null −£40 → slight regression
  - fc=120 new: pending; abd438ea first agent suggests deep regression
- **tnv3 gen 0 complete** (12/96). All agents negative on day_pnl.
  Top 3: 56c4173f (−£10), d50e5e95 (−£14), 28d95a3d (−£20). Pattern:
  locked +£83-105, tight naked (−£3 to −£22), but fc cost −£80-100
  for all. The fc cost is the operative ceiling. Gen 1 children will
  test whether day_pnl_per_std selection can cut fc cost via lower
  pair counts (bets 169-188 in gen 0 top-3) or higher maturation
  (mr 0.30-0.37). Average wall 18.3 min/agent → gen 1 (12 more
  agents) finishes ~17:30.
**Decisions made:** Wait for fc=120 newwindow to finish (~15:46
ETA) before drafting tnv2 verdict.
**Outstanding:** fc=120 newwindow (9 more agents); tnv3 gen 1+.
**Next iteration's focus:** by 14:16 fc=120 newwindow will be ~50
min in (3-4 agents done); start to see the actual fc=120 trajectory.

## 2026-05-17 14:17 — fc=120 newwindow at 4/10 deep negative; tnv3 gen 1 starts

**State entering iteration:** heartbeat. fc=120 newwindow 4/10 done.
tnv3 at 14/96 — gen 1 starting (2 agents in gen 1).
**Work done:** read fc=120 newwindow agent table; aggregated gen-0
vs gen-1 stats.
**Tests run:** None.
**Observations:**
- **tnv2 fc=120 newwindow 4/10:** mean = **−£162/day** vs layq null
  −£17 → tnv2 is **−£145/d WORSE** than null. 1/4 agents profitable.
  Pattern: locked floor +£173-222/d, but fc cost is the killer
  (per agent: see oldwindow numbers — same shape). The verdict
  across all 4 cells is REGRESSION on layq null.
- **tnv3 gen 1 first 2 agents — day_pnl_per_std selection signal
  surfacing**:
  - mean day_pnl: gen 0 −£46 → gen 1 (n=2) **−£18** (+£28/d gain)
  - mean naked: gen 0 −£29 → gen 1 −£16 (halved — the variance
    lever is doing what tnv1 already proved it could do)
  - **mean fc_pnl: gen 0 −£86 → gen 1 −£82 (only −£4 reduction)**
  - mean locked: gen 0 +£88 → gen 1 +£94 (slight improvement)
  - best agent in gen 1: **−£2/d** (break-even!)
- Interpretation: day_pnl_per_std IS selecting for better day_pnl,
  but via the SAME tnv1 mechanism (tighter naked + slightly higher
  locked), NOT via cutting fc cost. The selection isn't teaching
  the policy to open fewer pairs (mean_bets unchanged 178 → 180).
  If trajectory holds at +£28/gen, gen 7 lands at mean ~−£20/d /
  best ~+£20/d — still below Modest band's ≥+£50/d. But only 2
  gen-1 agents seen; wait for more before committing the read.
**Decisions made:** Wait for fc=120 newwindow finish before
drafting tnv2 verdict (full 10/10 needed for the regression
strength claim). Wait several more gen-1 agents before forming
the tnv3 trajectory thesis.
**Outstanding:** fc=120 newwindow 6 more (~80 min); tnv3 gen 1
finishing 10 more agents (~3-4h).
**Next iteration's focus:** by 14:47 fc=120 newwindow ~80 min in
(~7/10 agents done); tnv3 gen 1 may have 4-5 agents — enough
to read the selection trajectory more confidently.

## 2026-05-17 14:48 — tnv2 fc=120 newwindow 7/10 catastrophic; tnv3 gen 1 first +ve agent

**State entering iteration:** heartbeat. fc=120 newwindow 7/10
done. tnv3 at 16/96 (gen 1 has 4 agents now).
**Work done:** read fc=120 newwindow last 3 agent rows; aggregated.
Computed gen 0 vs gen 1 breakdown.
**Tests run:** None.
**Observations:**
- **tnv2 fc=120 newwindow 7/10:** mean = **−£174/d** (vs null
  −£17 → −£157/d WORSE). 4/49 days profitable (8%). 0/7 agents
  profitable per the prof column. Catastrophic regression.
- **tnv3 gen 1 (4 agents):** mean **−£15/d** (gen 0 was −£46),
  best **+£19/d** (gen 0 best −£10), 1/4 profitable.
  Breakdown of the +£31/d gen-0→gen-1 lift: naked tightening
  £20 (mean naked −£29 → −£9), locked rising £7 (+£88 → +£95),
  fc cost unchanged (−£86 → −£85). **Fc cost still won't move.**
- Linear extrapolation: gen 7 mean ~+£75/d → would clear Modest.
  But improvement curves usually flatten; the real measure is
  whether gen 2-3 keep improving or plateau. The first +ve agent
  (+£19) is the strongest leading indicator yet — if more like it
  appear, the cohort has a deployable candidate even short of
  band-clearing.
**Decisions made:** Wait for fc=120 newwindow finish (~45 min)
before drafting tnv2 verdict — but the verdict is now "REGRESSION
across all 4 cells" pending the last 3 agents.
**Outstanding:** fc=120 newwindow 3 more agents (~45 min); tnv3
gen 1 completion + gen 2 start.
**Next iteration's focus:** by 15:18 fc=120 newwindow ~110 min in
(~9/10 done); start drafting tnv2 verdict structure.

## 2026-05-17 15:19 — fc=120 newwindow 9/10; findings_tnv2.md ready to amend

**State entering iteration:** heartbeat. fc=120 newwindow 9/10
(b9ea7d3d in flight). tnv3 at 17/96.
**Work done:** read all 9 fc=120 newwindow agent rows; checked
plan dir — `findings_tnv2.md` already exists with the in-sample
analysis and predicted the held-out result. Need to append the
held-out reeval table to it (not write a new file).
**Tests run:** None.
**Observations:**
- **tnv2 fc=120 newwindow 9/10:** mean = **−£176.31/d**. 0/9 agents
  profitable. Day-level 5/63 profitable (8%). Mirrors fc=120
  oldwindow shape (mean −£210/d, 0/10 profitable). The 10th agent
  (b9ea7d3d) shouldn't move the headline.
- All 4 verdict cells now landed (modulo 1 agent):
  - fc=0 old: −£162/d, 1/10 prof
  - fc=120 old: −£210/d, 0/10 prof
  - fc=0 new: −£50/d, 3/10 prof
  - fc=120 new: −£176/d (9/10), 0/9 prof
- vs layq null: regression on all four — much worse than null on
  the three high-fc-or-old-window cells, slight worse on fc=0 new.
- tnv3 gen 1 weakening at 5 agents (mean −£15 → −£23 as 5th agent
  landed bad). Median crashed −£2 → −£34. Best stays +£19. Need
  to see the full gen 1 mean (likely 12 agents).
**Decisions made:** Wait one more heartbeat for the 10th agent.
Then amend findings_tnv2.md with the full reeval table and verdict.
**Outstanding:** fc=120 newwindow 1 more agent (~10 min);
findings_tnv2.md amendment; commit (with EXPERIMENTS.md).
**Next iteration's focus:** write the held-out reeval section in
findings_tnv2.md; commit.

## 2026-05-17 15:36 — tnv2 verdict written; uncommitted; tnv3 gen 1 weakening

**State entering iteration:** scheduled heartbeat. fc=120 newwindow
DONE at 15:21:40. All 4 tnv2 reevals complete.
**Work done:**
- Computed final per-cell aggregates from the 4 JSONLs (Python
  one-liner). All 40 agent-rows × 50 day-rows.
- Amended `findings_tnv2.md` with new top-of-file verdict line and
  "## Held-out reeval results (2026-05-17)" section: per-cell
  aggregate table, vs-null comparison, band verdict, why-it-failed
  diagnosis, single-positive-signal call-out (abd438ea), outcome.
- Updated `plans/EXPERIMENTS.md` tnv2 stub from "pending" to the
  full REGRESSION verdict with numbers.
**Tests run:** None.
**Outstanding:** Don't commit yet per operator instruction. Wait
for tnv3 verdict to bundle.
**tnv3 gen 1 (6/12 agents):** mean now −£29 (drift from −£15 at
n=4 → −£23 at n=5 → −£29 at n=6). Median −£34. Best +£19 (one
outlier). mean_locked +£96, mean_naked −£18, mean_fc_pnl
**−£90 (WORSE than gen 0)**. The fc cost is actually climbing as
the GA selects locked-rewarding parents → same volume-of-opens
phenotype reproducing. If this holds across gen 2-3, tnv3 will
also fail.
**Decisions made:** Don't commit. Continue monitoring tnv3 for
gen 1 finish (6 more agents, ~110 min) and gen 2 trajectory.
**Next iteration's focus:** wait for gen 1 to complete, see if
the +£19 best agent and the trajectory hold or if gen 1 settles
around the −£30 mean.

## 2026-05-17 16:09 — tnv3 gen 1 settling around −£30; fc cost climbing

**State entering iteration:** heartbeat. tnv3 at 20/96 (gen 1 at 8/12).
**Work done:** read tnv3 status panel only.
**Tests run:** None.
**Observations:**
- **tnv3 gen 1 trajectory firming up around −£30 mean.** Sequence:
  −£15 (n=4) → −£23 (n=5) → −£29 (n=6) → −£31 (n=8). The +£28/d
  gen-0→gen-1 lift visible at n=4 has shrunk to +£15/d at n=8.
- **mean_fc_pnl is climbing under selection**: gen 0 −£86 → gen 1
  (n=8) −£91. day_pnl_per_std is improving day_pnl via:
  - naked tightening (mean −£29 → −£18, +£11/d)
  - locked rising (+£88 → +£97, +£9/d)
  - But fc cost climbing −£5/d makes it net +£15/d.
- Bimodal gen 1 naked span: median 199 (tighter than gen 0's
  250 ✓) but max 728 (much wider than gen 0's 374). GA is splitting
  into "tight" and "blowup" sub-populations.
- **Best agent in cohort still cd44b121 (gen 1, +£19/d).** Profile:
  locked +£105, naked +£27, bets 186, mr 0.38. Naked +£27 is
  in-sample lucky (similar to tnv2's 4c217d70 which then collapsed
  on held-out).
- Trajectory implication: if gen 1 mean settles at −£30 and the
  per-gen lift halves each gen (typical curve), gen 7 lands at
  mean ≈ +£0 with best ≈ +£40 — still short of Modest band's
  ≥+£50/d cohort mean.
**Decisions made:** Don't act. Don't commit. Continue monitoring.
**Outstanding:** gen 1 last 4 agents (~70 min); gen 2 start.
**Next iteration's focus:** by 16:39 gen 1 should be 11-12/12;
gen 2 starting. The gen-2 mean is the cleaner trajectory signal
(gen 1 still has parent overlap with gen 0 children).

## 2026-05-17 16:25 — tnv3 STOPPED on mechanism; plan EXITS

**State entering iteration:** operator decision after in-flight
mechanism analysis (this session's "GA selection can't fix a
reward-side problem" diagnosis). tnv3 running at 20/96 agents.
**Work done:**
- Killed three python processes: tnv3 trainer (PID 14476), tnv3
  status watcher (47368), tnv2 reeval watcher (51424). All tree-
  killed cleanly (taskkill /T /F).
- Wrote `findings_tnv3.md` with the mechanism diagnosis: GA
  selection on day_pnl_per_std improved day_pnl via tighter naked
  + higher locked (the tnv1 lever) but did NOT cut fc cost
  (climbed −£86 → −£91 across gens). Diagnosed two structural
  reasons: (a) PPO's per-step gradient is unchanged between tnv2
  and tnv3 so children inherit the locked-rewarding policy
  regardless of which parents reproduce; (b) fc cost is
  substitutable for naked variance in the selector, so heavy-fc
  agents score higher when day_pnl is held constant.
- Updated `plans/EXPERIMENTS.md` tnv3 stub to the REJECTED verdict.
- Did NOT run held-out reeval — operator decision; the in-sample
  trajectory direction was sufficient evidence.
**Tests run:** None.
**Decisions made:**
- Plan exits. The variance-aware-selection thesis is rejected on
  mechanism. The next experiment must be reward-side, not
  selection-side. ~22h GPU freed.
- Candidates for the next plan (documented in findings_tnv3.md):
  per-tick fc-cost shaped penalty mirroring selective-open-shaping;
  raise close_signal bonus to compete with fc cost magnitudes;
  feed fc-prob into actor_head.
**Outstanding:** Commit findings_tnv3.md, findings_tnv2.md, EXPERIMENTS.md,
and this log together. No more wakeups.
**Next iteration's focus:** none — loop ends.
