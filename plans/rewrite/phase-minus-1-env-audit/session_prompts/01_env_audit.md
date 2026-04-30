# Session prompt — Phase −1: env audit

Use this prompt to open a new session in a fresh context. The prompt
is self-contained — it briefs you on the question, the evidence, and
the constraints. Do not require any context from the session that
scaffolded this prompt.

---

## The question

**Does the simulator (`env/`) faithfully implement the Betfair market
model documented in `docs/betfair_market_model.md`?**

If yes — green-light Phase 0 of the rewrite. The new trainer can be
built on top of the existing env with full confidence.

If no — file each divergence as a finding with a severity rating, and
gate Phase 0 on whichever divergences are load-bearing for credit
assignment, P&L correctness, or the supervised scorer's labels.

If inconclusive (spec ambiguous, real-Betfair behaviour debatable,
divergence theoretical-but-untested) — write up what was found, flag
which behaviours need real-Betfair data to validate, and stop.

## Why this question, why now

The rewrite (`plans/rewrite/README.md`) is replacing the trainer and
policy but **keeping the env unchanged**. Before we build a new
trainer on a 3,400-line simulator that has had ~6 months of
incremental changes, we need to verify the simulator actually models
what we think it models.

This isn't paranoia: the H1 finding (`plans/per-runner-credit/
findings.md`) two days ago surfaced a correctness issue in
`fill_prob_head`'s LABEL — the env wasn't wrong, but the trainer's
assumption about what the env's `bm.bets` contained was. A fresh-eyes
audit before the rewrite will catch any analogous "env is correct,
but the spec we're building against is different from what the env
actually does" gaps.

This is also cheap insurance. 1–2 hours of focused reading vs the
risk of building a new trainer on a misunderstood simulator and only
discovering it 3 weeks later.

## What you need to read first

1. `docs/betfair_market_model.md` — the operator's spec for how
   Betfair markets behave and what the simulator should approximate.
   This is the **ground truth** for the audit; the env is correct iff
   it matches this.
2. `CLAUDE.md` — sections "Bet accounting", "Order matching",
   "Equal-profit pair sizing", "Force-close at T−N",
   "`info[realised_pnl]` is last-race-only". These are the
   correctness facts we already know about the env.
3. `env/betfair_env.py` — focus on:
   - `_settle_current_race` (settlement logic, P&L attribution)
   - `step` (per-tick dispatch: open-cost, force-close, MTM)
   - `_attempt_close` (force-close placement)
   - `reset` (race state initialisation)
4. `env/exchange_matcher.py` — `ExchangeMatcher.match_back`,
   `match_lay`, `pick_top_price`, `_match`. The single-price /
   LTP-junk-filter / hard-cap rules.
5. `env/bet_manager.py` — `BetManager.place_back`, `place_lay`, the
   `Bet` dataclass, `force_close` and `close_leg` flags.
6. Tests for these modules — particularly
   `tests/test_exchange_matcher.py`, `tests/test_bet_manager.py`,
   `tests/test_betfair_env.py`. Existing tests show what behaviours
   are already pinned.

## What to do

### 1. Read the spec end-to-end (~30 min)

`docs/betfair_market_model.md` is 547 lines. Read it without
referring to the code. Make a private list of "things the spec says
the simulator should do" — keep it tight, ~20–40 bullet points.

### 2. Cross-check each spec point against the code (~45 min)

For each bullet, find the line(s) of env / matcher / bet_manager
that implement it. Three outcomes per bullet:

- **MATCH** — code does what spec says. Note the file:line and move on.
- **DIVERGE** — code does something different. Note: what the code
  does, what the spec says, severity (correctness / cosmetic / open
  question), and whether it's load-bearing for the rewrite.
- **NOT IMPLEMENTED** — spec describes behaviour the code doesn't
  attempt. Note severity.

Start a list as you go. Don't try to fix anything; just observe.

### 3. Cross-check the CLAUDE.md correctness facts (~15 min)

The "Bet accounting", "Order matching", "Equal-profit pair sizing",
"Force-close at T−N" sections in CLAUDE.md document specific
correctness invariants we've already established. Spot-check that
each invariant still holds in the current code:

- Order matching: single-price, no walking, LTP-required, ±50 % junk
  filter, hard cap inside the matcher.
- Force-close: relaxed matcher path, budget overdraft permitted,
  equal-profit sizing.
- Equal-profit formula: `S_lay = S_back × [P_back × (1−c) + c] / (P_lay − c)`.
- Bet count: distinct matched orders, not netted positions.
- `info["realised_pnl"]` is last-race-only; `info["day_pnl"]` is the
  episode-true value.
- `bet_manager.bets` is last-race-only; `env.all_settled_bets` is the
  episode-cumulative source of truth.

If any of these now fail, that's a regression — flag it as the
highest-severity finding.

### 4. Run the existing env / matcher / bet_manager test suites (~10 min)

```
python -m pytest tests/test_exchange_matcher.py \
                tests/test_bet_manager.py \
                tests/test_betfair_env.py \
                tests/test_forced_arbitrage.py \
                -v
```

All MUST pass. Any failure here is a regression in unrelated work
that landed since the tests were written; flag it.

### 5. Probe a few high-risk behaviours by reading code carefully (~15 min)

These are the behaviours where a quiet bug would be most damaging
for the new trainer. Read the implementing code line-by-line and
satisfy yourself it does what the spec says:

a. **Settlement P&L attribution.** A race with M matured pairs, N
   force-closed pairs, K naked pairs, and L agent-closed pairs —
   trace one of each through `_settle_current_race`. Does each
   contribute to the correct accumulator (`scalping_locked_pnl`,
   `scalping_force_closed_pnl`, naked term, `scalping_closed_pnl`)?
   Does `race_pnl` equal the sum?

b. **Single-price matching with stake > top-level size.** Place a
   £100 back when the top-of-book lay level has only £20 size. Does
   the matcher return a match of £20 (NOT walk to the next level)?
   Does the unmatched £80 get cancelled (not spilled)?

c. **Force-close overdraft.** With per-race budget exhausted, does a
   force-close still place at MIN_BET_STAKE or higher? Does it
   bypass the lay-liability scale-down?

d. **Equal-profit sizing on the lay leg.** Trace
   `equal_profit_lay_stake` for a known case. Does the formula match
   `S_lay = S_back × [P_back × (1−c) + c] / (P_lay − c)` with c =
   0.05?

e. **`env.all_settled_bets` accumulates across races.** Does the
   list grow with every race's matched bets? Is it reset per
   episode or per race? (Spec: per episode.)

### 6. Write up findings (`audit_findings.md`)

A new file at `plans/rewrite/phase-minus-1-env-audit/
audit_findings.md` with:

- **Spec conformance summary**: total bullets checked, count
  MATCH / DIVERGE / NOT IMPLEMENTED.
- **Findings list**: one entry per DIVERGE / NOT IMPLEMENTED bullet,
  with severity (`blocker | correctness-non-blocker | cosmetic |
  open-question`) and a one-line "why this matters for the new
  trainer".
- **CLAUDE.md correctness facts**: PASS / FAIL line per invariant.
- **Test suite status**: pass count + any failures.
- **Verdict**: one of:
  - **GREEN** — env matches spec; Phase 0 unblocked.
  - **AMBER** — minor divergences exist; fix in their own session
    BEFORE Phase 0 starts. List the fixes needed.
  - **RED** — load-bearing divergence; pause the rewrite, escalate to
    operator before doing anything else.

## Stop conditions

- GREEN verdict → write `audit_findings.md`, message operator
  "Phase −1 GREEN, ready for Phase 0", **stop**.
- AMBER verdict → write `audit_findings.md`, draft a session prompt
  for each fix (one fix per session, do NOT bundle), message operator
  "Phase −1 AMBER, N fixes needed before Phase 0", **stop**.
- RED verdict → write `audit_findings.md`, message operator
  "Phase −1 RED — pause rewrite, please review", **stop**.

## Hard constraints

- **Do not modify any code.** This is a read-only audit. If you spot
  a bug, file it as a finding; do not fix it in this session. Fixes
  get their own session prompts so the change is isolated and
  reviewable.
- **Do not write new tests.** Same reason.
- **Do not "improve" the spec doc.** If the spec is ambiguous, file it
  as an open-question finding; do not edit `docs/betfair_market_model
  .md` in this session.
- **Do not start Phase 0.** Even if the verdict is GREEN, don't open
  the next session. Hand control back to the operator.

## Out of scope

- Refactoring env / matcher / bet_manager.
- Adding new env features (new market types, new constraints, etc.).
- Reviewing the trainer or policy code.
- Reviewing the data pipeline.
- Real-Betfair validation against live data (the spec is the ground
  truth for this session).

## Useful pointers

- `docs/betfair_market_model.md:1` — the spec.
- `env/betfair_env.py:_settle_current_race` (~line 2842).
- `env/betfair_env.py::_attempt_close` (~line 2523).
- `env/exchange_matcher.py::_match` — the load-bearing match
  function.
- `env/bet_manager.py:103–133` — the `Bet` dataclass with
  `close_leg` and `force_close` flags.
- `env/scalping_math.py::equal_profit_lay_stake` — the equal-profit
  helper.
- `tests/test_exchange_matcher.py` — pinned matcher behaviours.
- `tests/test_forced_arbitrage.py::TestSelectiveOpenShaping` — the
  selective-open-shaping invariants.

## Estimate

Single session, 1–2 hours of focused reading.

- 30 min: spec read.
- 45 min: spec ↔ code cross-check.
- 15 min: CLAUDE.md correctness facts.
- 10 min: run test suites.
- 15 min: high-risk behaviour probes.
- 30 min: write up `audit_findings.md`.

If you find yourself doing more than that — particularly if you start
fixing things — stop and check whether scope has crept beyond the
audit question. If yes, write up what you have, flag the scope creep,
and stop.
