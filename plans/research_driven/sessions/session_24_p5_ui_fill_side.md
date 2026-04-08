# Session 24 — P5: UI fill-side annotation

## Before you start — read these

- `../purpose.md`
- `../proposals.md` P5
- `../master_todo.md` Phase 1
- `../lessons_learnt.md` — the "crossing the spread is the only
  mode" entry explains *why* this annotation exists: the
  operator was confused by the fill price showing as the lay
  side of the book on back bets. This session prevents that
  confusion recurring.
- `../manual_testing_plan.md`
- `../ui_additions.md` — the row for this work.
- The replay UI source files (grep the repo for "bet row" or
  similar to locate them).

## Goal

Add a tiny one-character (or equivalent compact) annotation to
each bet row in the replay UI showing which side of the book
the fill came from:

- Back bet → filled at best lay → annotate with something like
  `L→B` meaning "came from lay side, it's a back bet".
- Lay bet → filled at best back → annotate as `B→L`.

The exact visual treatment is at the implementer's discretion
as long as it is (a) compact, (b) consistent across the three
races in the manual test, and (c) doesn't overlap the fill-price
column on a normal window size.

## Inputs — constraints to obey

1. **No new data pipeline changes.** The fill price and bet
   side are already on the `Bet` object. This session is UI-
   only.
2. **Do not rename existing columns.** Add, don't refactor.
3. **Preserve dark-mode compatibility.** If the UI has a dark
   theme, the annotation uses a colour that works in both.
4. **No new config knobs.** The annotation is always on. It is
   cheap enough that an on/off toggle is not worth the code.

## Steps

1. **Locate the replay UI bet-row rendering code.** Grep for
   terms like `average_price`, `matched_stake`, or `BetSide`
   near UI files. Read the existing rendering to understand
   how a row is currently assembled.

2. **Compute the annotation string from the bet.** Pure
   function, no new data needed — `side` and `average_price`
   alone are enough.

3. **Place the annotation in the row.** Either as a new mini-
   column or as a superscript/suffix on the existing price
   column. Pick whichever is visually less disruptive.

4. **Check light and dark themes.** If both exist, eyeball
   both.

## Tests to add

Create `tests/research_driven/test_p5_fill_side_annotation.py`
(or add to an existing UI snapshot test file if one exists):

1. **Annotation string for a back bet.** Input: a `Bet` with
   `side=BACK` and a known fill price. Output: the annotation
   string matches the expected format.
2. **Annotation string for a lay bet.** Same but `side=LAY`.
3. **Snapshot test of one bet row.** If the UI already has
   snapshot tests, add a new snapshot showing the annotation.
   Otherwise skip — don't add a new snapshot framework just
   for this.

All fast.

## Manual tests (this is where most of the value is)

- **Open three races in the replay UI.** Each race has at least
  one back bet and at least one lay bet. Confirm every bet row
  shows the new annotation and it is readable.
- **Resize the UI window to a smaller size.** Confirm the
  annotation does not overlap the fill-price column or any
  other column.
- **Switch theme (if applicable).** Confirm the annotation is
  visible in both light and dark modes.
- **Spot-check one operator confusion source.** Find a back bet
  where the fill price is noticeably different from the mid —
  the kind of row that originally triggered the research — and
  confirm the annotation makes it obvious the fill came from
  the lay side.

## Session exit criteria

- Annotation visible on every bet row in three spot-checked
  races.
- No column overlap at a normal window size.
- `progress.md` Session 24 entry with before/after screenshots
  attached to the commit (or linked from the entry).
- `ui_additions.md` entry for the replay UI annotation ticked.
- `ui_additions.md` entry for the `ai-betfair` live dashboard
  *remains open* — this session is only the replay UI. The
  live dashboard version is owed in the `ai-betfair` repo.
- `master_todo.md` Phase 1 P5 box ticked.
- Commit.

## Do not

- Do not refactor the existing bet-row code. Add, don't
  rewrite.
- Do not add a config knob to toggle the annotation. It's
  always on.
- Do not also update the `ai-betfair` live dashboard in this
  session. That's a separate repo, separate session, separate
  reviewer context. Leave the `ui_additions.md` row for the
  live dashboard open.
- Do not block this session on sessions 19–23. P5 is
  completely independent of P1 and P2 — it can land at any
  time after session 18. If the operator wants it early for
  ergonomic reasons, it is fine to do out of order.
