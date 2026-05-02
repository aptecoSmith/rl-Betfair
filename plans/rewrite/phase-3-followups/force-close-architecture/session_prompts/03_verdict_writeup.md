# Session prompt — force-close-architecture Session 03: verdict + writeup

Use this prompt to open a new session in a fresh context. The prompt
is self-contained — it briefs you on the task and the writeup
deliverables. Do not require any context from the session that
scaffolded this prompt.

---

## When to load this session

Load this session when EITHER:

- Session 01 produced a GREEN verdict (target-£-pair sizing alone
  hits mean fc ≤ 0.30 AND ≥ 4/12 positive eval P&L) — Session 02
  was skipped, this session writes up the single-fix win.
- Session 02 produced a verdict (GREEN / PARTIAL / FAIL) — this
  session writes up the two-cohort comparison and any
  operator-authorised stacked run.

Do NOT load this session if Session 01 shipped FAIL and the
operator has not yet decided whether to spend Session 02's GPU
budget. The Session 03 writeup waits for whichever cohort tree
the operator commits to.

## The task

This is a writeup-only session. No new code, no new cohort runs.

End-of-session bar:

1. **`findings.md` complete** with the Session-by-Session results,
   the cross-cohort comparison table, and the verdict (GREEN /
   PARTIAL / RED / RED-with-caveat).
2. **`lessons_learnt.md` written** — what the data taught us
   about the rewrite premise, regardless of verdict. Format
   matches the project's existing lessons-learnt files (see
   `plans/arb-curriculum/lessons_learnt.md` or
   `plans/arb-signal-cleanup/lessons_learnt.md` for the
   conventions: dated entries, lead with the lesson, then
   evidence, then the workflow change it justifies).
3. **`plans/rewrite/README.md` updated** with the verdict and
   the next-step pointer (Phase-4 scale-up green-light if GREEN;
   v1-revert sketch if RED).
4. **CLAUDE.md updated** with any new mechanics that landed in
   Session 01 or 02 — same level of detail as existing entries
   like §"Force-close at T−N (2026-04-21)" or §"Per-step
   mark-to-market shaping (2026-04-19)". Default values, gene
   ranges, byte-identical-when-disabled invariants, the
   regression-guard tests that pin them.
5. **Plan status flipped** in
   `plans/rewrite/phase-3-followups/force-close-architecture/purpose.md`
   frontmatter (`status: green; complete` or
   `status: red; complete` or `status: red-with-caveat; complete`).
6. **Phase-4 scale-up gate decision recorded** in the rewrite
   README. If GREEN, the scale-up plan can open; if RED, scale-up
   stays gated.

## What you need to read first

1. `plans/rewrite/phase-3-followups/force-close-architecture/purpose.md`
   — this plan's purpose, success bar, hard constraints. Verify
   the verdict you're writing up matches the criteria there.
2. `plans/rewrite/phase-3-followups/force-close-architecture/findings.md`
   — Session 01 (and 02 if it ran) results. The numbers your
   writeup synthesises.
3. `plans/rewrite/phase-3-followups/no-betting-collapse/findings.md`
   — the AMBER v2 baseline + operator review that motivated this
   plan. The comparison floor your verdict is graded against.
4. `plans/rewrite/README.md` — the rewrite's overall status and
   the existing Phase-3 verdict block. Your update plugs into
   the existing structure.
5. `plans/arb-signal-cleanup/lessons_learnt.md` (or any other
   lessons-learnt file under `plans/`) — for the file format
   conventions.
6. `CLAUDE.md` — for the convention of how mechanics changes are
   documented (default values, gene ranges, byte-identical
   invariants, load-bearing regression guards).
7. The cohort scoreboards directly:
   - `registry/v2_amber_v2_baseline_1777577990/scoreboard.jsonl`
     (comparison floor)
   - `registry/v2_force_close_arch_session01_target_pnl_<ts>/scoreboard.jsonl`
     (Session 01 cohort)
   - `registry/v2_force_close_arch_session02_stop_close_<ts>/scoreboard.jsonl`
     (Session 02 cohort, if it ran)

   Cross-check the numbers in findings.md against the raw rows.
   If they disagree, the writeup uses the raw rows; flag the
   discrepancy in lessons_learnt.md.

## What to do

### 1. Re-score every cohort under one analysis (~30 min)

Run the Bar 6 tool on every relevant registry dir and capture
output verbatim:

```
python C:/tmp/v2_phase3_bar6.py registry/v2_amber_v2_baseline_1777577990 \
    > /tmp/bar6_amber_v2.txt
python C:/tmp/v2_phase3_bar6.py registry/v2_force_close_arch_session01_target_pnl_<ts> \
    > /tmp/bar6_session01.txt
# (Session 02 only if it ran)
python C:/tmp/v2_phase3_bar6.py registry/v2_force_close_arch_session02_stop_close_<ts> \
    > /tmp/bar6_session02.txt
```

Compute the per-session NEW metrics:

- Session 01: median target-PnL refusal rate, median policy-close
  fraction.
- Session 02: median stop-close fraction, naked-back catastrophe
  count (per-pair settle loss > £200).

These come from the per-agent inline analyses in Sessions 01 and
02's session prompts; copy the snippets and run them again here
so the writeup numbers are guaranteed reproducible.

### 2. Fill the cross-cohort comparison table in findings.md (~30 min)

| Metric | AMBER v2 | Session 01 | Session 02 (if ran) |
|---|---|---|---|
| mean fc_rate | 0.809 | ? | ? |
| ρ(entropy_coeff, fc_rate) | −0.532 | ? | ? |
| positive eval P&L (count / 12) | 2 | ? | ? |
| median policy-close fraction | (baseline) | ? | ? |
| median target-PnL refusal rate | n/a | ? | n/a |
| median stop-close fraction | n/a | n/a | ? |
| naked-back catastrophes (loss > £200) | (count) | ? | ? |
| cohort wall (h) | 3.5 | ? | ? |

Add a one-paragraph interpretation under the table: which
mechanics change moved which metric, by how much, and what that
implies about the rewrite premise.

### 3. Write the verdict block (~30 min)

Apply the success bar from purpose.md §"Success bar":

- Cohort meets `mean fc ≤ 0.30 AND ≥ 4/12 positive AND wall ≤ 4 h`
  → **GREEN**.
- Cohort meets `mean fc ≤ 0.30 AND ≥ 4/12 positive AND wall > 4 h`
  → **GREEN-with-throughput-caveat**. Open
  `phase-3-followups/throughput-fix/`.
- Otherwise → **RED**.

Verdict block format (matches `no-betting-collapse/findings.md`
§"Verdict — GREEN-with-caveat"):

| Item | Status |
|---|---|
| Session 01 (target-£-pair sizing) | ? |
| Session 02 (stop-close) | ? (or "skipped — S01 GREEN") |
| mean fc ≤ 0.30 | ? |
| ≥ 4/12 positive eval P&L | ? |
| Verdict | **GREEN** / **RED** / **RED-with-caveat** |
| Phase-4 scale-up gate | unblocked / blocked |

### 4. Write lessons_learnt.md (~45 min)

Lead with the architectural lesson — what the data taught us.

If GREEN: name which mechanic was load-bearing and what the
*absence* of that mechanic (in v1, in AMBER v1, in AMBER v2)
silently masked. The "force-close as crutch" framing from the
operator review is the lead-in; the lesson is the specific
mechanism that broke it.

If RED: name what the data refuted. The operator review's
mechanics hypothesis ("first-class £-target + stop-loss → fc
falls"). If even the mechanics-level fix didn't move fc rate,
that's a fact the rewrite's whole architecture has to answer
to. Don't soften the framing — the next person to read this
file needs to know the architecture didn't survive contact with
the data.

Cover at minimum:

1. **The metric lesson.** Bar 6a as defined was uninformative
   across NOOP-vs-trading regimes. Document the better metric
   (whatever it turned out to be — policy-close fraction
   probably). This is a lesson for every future cohort, not
   just this plan.
2. **The "tune mechanics, not coefficients" lesson.** If
   Session 01 / 02 worked where the original
   matured_arb / naked_loss_anneal / mark_to_market tree
   hadn't, document that the diagnostic order matters: fix the
   mechanics first, then tune coefficients on the new
   mechanics. Don't tune coefficients on broken mechanics.
3. **The "stop-loss is structural, not learned" lesson.** If
   Session 02 worked, the lesson is that some abstractions are
   too foundational for the policy to learn from delayed
   reward — they have to be in the env. List the candidates for
   future structural-not-learned conversions
   (the operator's "we wouldn't leave a back bet on" rule;
   risk-of-ruin caps; per-race exposure bounds).
4. **The cross-cohort comparison lesson.** If multiple cohorts
   agree on direction but disagree on magnitude, document the
   variance — that informs how seriously to take a single
   cohort's verdict in future plans.

Use the dated-entry format (`### YYYY-MM-DD — <title>`) so
later sessions can append.

### 5. Update CLAUDE.md (~30 min)

If Session 01 landed and shipped: add a section like

```
## Target-£-pair sizing (2026-05-XX)

When `reward.target_pnl_pair_sizing_enabled = True`, the
auto-paired passive's price is solved from a £-target instead
of a tick distance. The agent's `arb_spread` action dim
(unchanged in count and range) reinterprets as
`target_pair_pnl ∈ [£0.20, £5.00]` (linear). [...]
```

Cover: what the env does, the byte-identical invariant when
disabled, the architecture-hash break (action-dim semantics
shifted), the load-bearing tests in
`tests/test_forced_arbitrage.py::TestTargetPnlPairSizing`.

If Session 02 landed and shipped: same pattern for stop-close,
including the naked-lay long-odds carve-out and the
strict-matcher invariant.

If a session ran but didn't ship (RED): do NOT add it to
CLAUDE.md. Document in lessons_learnt.md instead.

### 6. Update plans/rewrite/README.md (~15 min)

Find the existing Phase-3 verdict block. Append:

```
### force-close-architecture follow-on (2026-05-XX)

[GREEN / RED / RED-with-caveat] — [one-line summary].

Cohort: `registry/v2_force_close_arch_session0X_<ts>/`.
Scoreboard: ...
Plan: `plans/rewrite/phase-3-followups/force-close-architecture/`.

Phase-4 scale-up gate: [unblocked / blocked].
```

If GREEN, draft a one-paragraph kickoff for the Phase-4 scale-up
plan ("scale 12 → 66 agents on the GREEN mechanics; same eval
day; same protocol"). Do NOT scaffold the Phase-4 plan dir in
this session — that's scope-creep; let Phase-4 own its own
opening.

If RED, draft a one-paragraph kickoff for the v1-revert /
post-mortem ("rewrite premise is refuted at the mechanics level;
revert path is X, Y, Z"). Same scope discipline — don't open
the post-mortem plan in this session.

### 7. Flip plan status (~5 min)

In
`plans/rewrite/phase-3-followups/force-close-architecture/purpose.md`
frontmatter:

```yaml
status: green; complete
closed: 2026-05-XX
```

(or `red; complete` / `red-with-caveat; complete`).

## Stop conditions

- **findings.md numbers don't match the raw scoreboards** →
  stop and re-run the analysis. The writeup is built on numbers
  that survive re-derivation; if they don't, the analysis is
  wrong somewhere.
- **The verdict you're about to write disagrees with the
  cohort's data** (e.g. you want to ship GREEN but the data
  shows mean fc=0.45) → stop and ask the operator. The success
  bar is fixed in purpose.md; you don't get to soften it in the
  writeup. If the data says RED, the writeup says RED.
- **A new mechanics question surfaces during writeup** → flag
  it for a follow-on plan, don't bundle into this session.
  Writeup-session scope is locked; new investigation goes to a
  new plan.

## Hard constraints

1. **No new code in this session.** Writeup only. Any test
   discrepancy that surfaces during analysis is a separate fix
   in a separate commit.
2. **No new cohort runs.** The verdict is on the data already
   collected.
3. **No softening the success bar.** Apply the criteria from
   purpose.md exactly.
4. **CLAUDE.md only documents what shipped.** RED sessions
   don't add to CLAUDE.md (they go in lessons_learnt.md).
5. **README.md update is one block, not a rewrite.** The
   existing rewrite README has an established structure;
   append, don't restructure.

## Out of scope

- Phase-4 scale-up plan scaffolding.
- v1-revert plan scaffolding.
- Throughput-fix plan scaffolding.
- New mechanics ablations.
- Re-running any cohort.
- Modifying scoreboard.jsonl rows directly.

## Useful pointers

- All cohort scoreboards under `registry/`.
- Bar 6 tool: `C:/tmp/v2_phase3_bar6.py`.
- Existing lessons_learnt format examples:
  `plans/arb-curriculum/lessons_learnt.md`,
  `plans/arb-signal-cleanup/lessons_learnt.md`.
- CLAUDE.md mechanics-doc examples:
  §"Force-close at T−N (2026-04-21)",
  §"Per-step mark-to-market shaping (2026-04-19)",
  §"BC pretrain (2026-04-19)".
- Existing rewrite README:
  `plans/rewrite/README.md`.

## Estimate

2.5–3 h, all writeup:

- 30 min: re-score all cohorts.
- 30 min: fill cross-cohort table.
- 30 min: verdict block.
- 45 min: lessons_learnt.md.
- 30 min: CLAUDE.md.
- 15 min: rewrite README + plan status flip.

If past 4 h, stop and check — the writeup shouldn't grow
beyond what the data supports. If you're padding the writeup
to make a marginal verdict feel decisive, the verdict probably
isn't decisive.
