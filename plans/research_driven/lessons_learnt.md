# Lessons Learnt — Research-Driven

Append-only. One entry per surprising finding. "Surprising" includes
*successful* non-obvious decisions, not only mistakes — the bias is
to overshare here, because the cost of writing an entry is one
paragraph and the cost of repeating an avoidable mistake in three
sessions' time is real.

Cross-references to `arch-exploration/lessons_learnt.md` and
`next_steps/lessons_learnt.md` are welcome — anything load-bearing
in those files still applies here, and entries that build on them
should cite them rather than repeat them.

Format per entry:

```
## YYYY-MM-DD — Session NN — Short title

**What happened:** factual.
**Why it was surprising:** what the prior expectation was.
**What changes because of it:** code, tests, docs, or just future
behaviour. If "nothing", say so explicitly.
```

---

## 2026-04-07 — Pre-session — Crossing the spread is the *only* mode

**What happened:** While reading `research/research.txt` on money
pressure / queue position / crossing the spread, the operator
noticed the model picks lay prices as bet prices for back bets, and
asked whether the matcher needed work to *enable* crossing the
spread. Auditing `env/exchange_matcher.py` and
`env/betfair_env.py::_process_action` showed the answer is the
opposite: the matcher *only* crosses the spread, and the agent has
no other action verb. The screenshots are correct, not buggy.

**Why it was surprising:** The framing of the research material
(join queue / cross spread / cancel as three distinct decisions)
implied the simulator was modelling at least two of the three. It
turns out it's modelling only one, and that one is forced.

**What changes because of it:** Two things. (1) The whole
research-driven planning folder was reframed around adding the
*passive* regime rather than enabling the aggressive one — see
`analysis.md` §1 and `proposals.md` P3. (2) `hard_constraints.md`
gained an explicit "no `modify` action; cancel + place is the
canonical move" rule, because the temptation to add `modify` only
appears once you're already thinking about passive orders, and we
want the rule recorded before that thinking starts.

---

## 2026-04-07 — Pre-session — Phantom fills exist in `ai-betfair`

**What happened:** The operator reported that `ai-betfair` has
declared "bets on today" for trades that demonstrably had no
liquidity to match against on the real exchange. This is the live
equivalent of the optimistic-fill assumption that the simulator
itself made before `exchange_matcher.py` was tightened.

**Why it was surprising:** We had assumed the live wrapper was
subscribing to the Betfair order stream and using it as the source
of truth. It is apparently not — it's treating the policy's action
emission as the fact of the fill. This means every research-driven
improvement we ship would be wasted while the bug is open: a policy
trained against ground truth in sim, deployed into a runtime that
fabricates state.

**What changes because of it:** Three things. (1) `bugs.md` got an
R-1 entry recording the bug. (2) `design_decisions.md` recorded the
prerequisite-not-co-task rule (the fix has to ship in `ai-betfair`
before any new policy from this folder lands in production).
(3) `master_todo.md` Phase 0 makes that prerequisite explicit at
the top of the execution queue.

---

## 2026-04-07 — Pre-session — Matcher does not deplete its own already-matched volume

**What happened:** The operator inspected the 14:00 race in
`ai-betfair` and found two bets on the winner, £12.10 followed by
£17 about 12 seconds later, against a visible book that had only
£21 of liquidity at that price. Audit traced the simulator side of
the same logic to `env/exchange_matcher.py::_match`, which is
stateless across ticks: it reads `top.size` from each tick's
ladder snapshot directly, with no deduction for fills the agent
already made at the same price level. Filed as `bugs.md` R-2.

**Why it was surprising:** The matcher is the file that already
got hardened against the phantom-profit incident — junk filtering,
single-price rule, hard cap after filter. It had been treated as
"the careful one" since that work landed. R-2 shows there's a
whole different category of phantom liquidity it doesn't address:
*self*-depletion. Realising the existing hardening only protects
against *third-party* junk in the book, not against the agent's
own stacked orders, was the surprise.

**What changes because of it:**
- `bugs.md` R-2 entry with sim-fix sketch (per-(market, selection,
  side, price) accumulator on `BetManager`).
- `downstream_knockon.md` §0a documents the live-side equivalent
  in `ai-betfair` and explains why §0 and §0a are distinct
  bugs that need parallel fixes.
- `master_todo.md` Phase 0 records R-2 as an in-repo task that
  should land before P3/P4 (passive orders make the bug worse by
  stretching the depletion window).
- `design_decisions.md` records R-2 as an independent sim bug,
  not a sub-item of any P1–P5 proposal.
- Operator follow-up flagged: pull the `ai-betfair` order-stream
  log for the actual 14:00 race to determine whether the live
  observation is R-1 (phantom fill, second match never happened),
  R-2 (over-placed against stale local view), or a snapshot
  artefact (real Betfair liquidity replenished between data
  ticks). All three are possible and the fix differs.

---

## 2026-04-07 — Pre-session — Phantom-fill prerequisite was over-scoped

**What happened:** The first draft of `master_todo.md` Phase 0
gated *all* research-driven work on the `ai-betfair` phantom-fill
fix. The operator pushed back: "isn't that the opposite of what we
are trying to do? Check if those constraints really make sense."
Re-reading the constraints showed an inconsistency:
`hard_constraints.md` #8 said the gate was on **deployment** of a
new policy, which was correct, but `master_todo.md` Phase 0 said
it was on **shipping any item**, which was overreach. The
inconsistency had been written into the same folder on the same
day.

**Why it was surprising:** The framing felt safe at draft time
("don't ship into a runtime that lies"), but it conflated two very
different gates. Training-side work in this repo can proceed
without ever touching `ai-betfair`; only the *deployment hand-over*
needs to wait for the live wrapper fix. The two streams are part
of the same family of "the simulator/wrapper has been too
optimistic about fills" and they should run *in parallel*, not
sequentially. Sequencing them masquerades as caution but is
actually a bottleneck.

**What changes because of it:**
- `master_todo.md` Phase 0 rewritten to be a deployment-gate, not
  a programme-wide block. Training-side sessions may merge to
  master without waiting for the cross-repo fix.
- `hard_constraints.md` #8 reworded to be explicit about the
  training-vs-deployment split, citing the new
  `design_decisions.md` entry.
- New `design_decisions.md` entry "Phantom-fill gate is on
  deployment, not on training-side work" captures the refinement
  and credits the operator's review for catching it.
- General lesson logged here: when a constraint feels safe but
  serialises work that could parallelise, it is probably
  over-scoped. The next time this folder writes a "prerequisite"
  rule, ask "what is the *narrowest* scope at which this
  prerequisite is actually load-bearing?" before committing to
  the broader version.

---

## (no further entries yet)

New entries get appended below this line in chronological order.
Don't reformat or rewrite earlier entries — if a finding turns out
to be wrong, write a new entry recording the reversal and cite the
earlier one.
