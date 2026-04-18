# Scalping Equal-Profit Sizing — Session 03 prompt

Documentation pass + cross-plan tidy. Prose only — no code
changes. Closes the loop on Sessions 01–02 so the next operator
opening CLAUDE.md cold understands which sizing formula is in
use today and why.

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — especially the "Reward-scale
  change — call out loudly" section.
- [`../hard_constraints.md`](../hard_constraints.md) — §12
  (CLAUDE.md update preserves history) and §21 (don't bundle
  the activation re-run into a code commit).
- [`02_wire_placement.md`](02_wire_placement.md) — the commit
  hash from Session 02 lives here once it lands; you'll need it
  for the cross-link.
- `CLAUDE.md` — read the existing "Order matching: single-price,
  no walking" section and the surrounding scalping notes
  ("Reward function: raw vs shaped" especially) so the new
  paragraph slots in cleanly.
- `plans/scalping-active-management/lessons_learnt.md` and
  `plans/scalping-active-management/progress.md` — for the
  cross-link and the brief append-only entry to that plan.

## What to do

### 1. Update CLAUDE.md

Find the "Order matching: single-price, no walking" section.
Below it (or as a sub-section, depending on how the file flows
when you read it) add:

```markdown
## Equal-profit pair sizing (scalping)

The auto-paired passive (and the closing leg from
`scalping-close-signal`) is sized to **equalise net profit on
both race outcomes after commission**, not to equalise exposure.

The formula:

    S_lay = S_back × [P_back × (1 − c) + c] / (P_lay − c)

(or symmetrically for lay-first scalps:
`S_back = S_lay × [P_lay × (1 − c) + c] / (P_back − c)`).

Worked example — Back £16 @ 8.20, passive lay at 6.00, c=0.05:

    S_lay = 16 × [8.20×0.95 + 0.05] / (6.00 − 0.05)
          = 16 × 7.84 / 5.95
          = £21.08

Settles to:

    win  = 16 × 7.20 × 0.95 − 21.08 × 5.00 = +£4.04
    lose = −16 + 21.08 × 0.95              = +£4.03

Both outcomes net the same ≈ £4.03. `locked_pnl =
min(win, lose)` therefore reports the actual lock, not the
near-zero floor of an over-laid trade.

**Historical note (audit trail).** Before commit
`<session 02 hash>` (2026-04-18,
`plans/scalping-equal-profit-sizing`) the sizing was
`S_lay = S_back × P_back / P_lay` — a formula derived under the
assumption of zero commission. With Betfair's 5% commission this
form equalises *exposure* (`stake × (price − 1)`) rather than
P&L, producing pairs whose win-side payoff was tiny and lose-side
payoff was much larger. Pre-fix scoreboard rows reflect that
behaviour and are valid pre-fix references; post-fix scoreboard
rows are the new comparison surface.
```

Replace `<session 02 hash>` with the actual short hash from
Session 02's commit.

If the existing CLAUDE.md text mentions `S = S_b × P_b / P_l`
anywhere (e.g. inside the "Reward function" section's prose),
update that mention to point at the new formula or strike it
with a "see equal-profit pair sizing section" pointer.

### 2. Cross-link entries

In `plans/scalping-active-management/lessons_learnt.md`, append:

```markdown
- 2026-04-18: the Session-01 sizing comment ("derived from
  demanding equal P&L in win and lose outcomes") was true in
  intent but wrong in math — the derivation only holds at
  c=0. Fixed by `plans/scalping-equal-profit-sizing/`
  Session 02 (commit `<hash>`). Until that fix landed, the
  re-quote / paired-passive code was over-laying every back-
  first scalp by `[1 − (1 − c)]/(1 − c) ≈ 5%` of the back stake,
  producing a near-zero win-side payoff and a much bigger
  lose-side payoff. Locked_pnl values from runs before the fix
  systematically understated equal-profit-equivalent locks.
```

In `plans/scalping-active-management/progress.md`, no changes
needed — that plan's session entries are append-only history of
what happened at the time. The lessons_learnt cross-link is the
right place to record the post-hoc correction.

### 3. Append "Reading the new locked numbers" to this plan's
   `progress.md`

After the Session-03 entry you'll add at the bottom of this
session, also add a brief operator-facing paragraph explaining
the comparison cliff:

```markdown
### Reading the new locked numbers (operator note)

After Session 02 (commit `<hash>`), `scalping_locked_pnl`
values from new training runs are NOT directly comparable to
scoreboard rows from runs before that commit. The new values
reflect equal-profit-balanced pair P&L; the old values reflected
the worst-case floor of over-laid pairs.

Rule of thumb:

- Old `locked_pnl` ≈ NEW `locked_pnl` × `(1 − c)` for tight
  spreads. Roughly: take the new number and multiply by 0.95 to
  get the comparable old-formula floor, OR take an old number
  and divide by 0.95 (then add roughly the win-side cliff) to
  estimate the equal-profit-equivalent.
- For wide spreads (well into the profitable zone) the two
  formulas converge; the difference is largest right at the
  commission edge.

When in doubt: the post-fix number is the one a real-world
scalping calculator would produce.
```

Adjust commit hash and tighten the wording when you write it.

### 4. Append the Session-03 entry to this plan's `progress.md`

Following the convention in
`scalping-active-management/progress.md`:

```markdown
## Session 03 — CLAUDE.md + cross-plan notes (2026-04-XX)

**Landed.** Commit `<hash>`.

- CLAUDE.md gains a new "Equal-profit pair sizing (scalping)"
  section after "Order matching: single-price, no walking".
  Includes the formula, worked example, and historical
  audit-trail note pointing at Session 02's commit.
- `plans/scalping-active-management/lessons_learnt.md` appended
  with a one-paragraph entry recording the original sizing
  comment was correct in intent but wrong in math.
- `progress.md` (this file) gets an operator-facing
  "Reading the new locked numbers" paragraph explaining the
  pre-vs-post-fix scoreboard comparability cliff.

No code changes this session. Test suite untouched.
```

## Exit criteria

- CLAUDE.md updated, reads cleanly cold.
- The two cross-plan references (lessons_learnt.md +
  progress.md) appended.
- `progress.md` of this plan has Session 03 entry.
- `git diff` shows only `.md` files changed; no code, no test
  file edits.

## Acceptance

A reader who has never seen this plan opens CLAUDE.md, finds
the "Equal-profit pair sizing" section, and can answer "which
sizing formula does the env use today and why" without opening
any other file.

## Commit

One commit, type `docs`. References Sessions 01 + 02 commit
hashes. No reward-scale change (no code changed); explicit
"docs-only" line in the body so the reader doesn't have to
diff to confirm.

```
docs(scalping): equal-profit sizing formula in CLAUDE.md

Adds the "Equal-profit pair sizing (scalping)" section to
CLAUDE.md after "Order matching: single-price, no walking".
Includes the formula, the canonical worked example (Back £16
@ 8.20 / Lay @ 6.00 / c=5% → locked £4.03), and a historical
audit-trail note pointing at the Session-02 commit
(`<hash>`) where the env switched off the old wrong formula.

Cross-references:
- plans/scalping-active-management/lessons_learnt.md gets a
  one-paragraph entry recording that the original Session-01
  sizing comment was correct in intent but wrong in math.
- plans/scalping-equal-profit-sizing/progress.md gets an
  operator-facing "Reading the new locked numbers" paragraph
  explaining the pre-vs-post-fix scoreboard comparability
  cliff.

Docs-only commit. No code changes; test suite untouched.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## After Session 03

The reward-scale fix is COMPLETE on the prose side. The UI
display fix (Session 04) is a separate concern that won't change
behaviour but will tidy the activity-log surface. If the
operator wants to launch the activation re-run before Session
04 lands, that's fine — the locked numbers will already be
correct, the only difference Session 04 makes is the `£` →
`@` cosmetic on price displays.

Proceed to [`04_ui_display_fix.md`](04_ui_display_fix.md).
