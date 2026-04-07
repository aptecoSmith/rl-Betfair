# Session 7 — Drawdown-aware shaping (DESIGN PASS FIRST)

## Before you start — read these

- `plans/arch-exploration/purpose.md`
- `plans/arch-exploration/master_todo.md` (Session 7)
- `plans/arch-exploration/testing.md`
- `plans/arch-exploration/progress.md` — confirm Sessions 1-6 done.
- `plans/arch-exploration/lessons_learnt.md`
- `plans/arch-exploration/ui_additions.md`
- Repo root `CLAUDE.md` — the zero-mean-for-random-policies rule is
  the single hardest constraint on this session.

## Goal

Add a shaped reward term that encourages the agent to limit
intra-day drawdown, without biasing random policies toward
conservatism.

## Why this session starts with design, not code

The obvious formulation is wrong:

```
drawdown_penalty = − ε × max(peak_day_pnl − current_day_pnl, 0) / starting_budget
```

This is strictly non-positive. A random-betting agent will
accumulate drawdowns and therefore accumulate negative shaped
reward. That teaches the agent "betting less is strictly better",
which is exactly the asymmetric-shaping bug we spent a whole session
fixing in the phantom-profit investigation.

We need a zero-mean formulation. Options:

### Option A — Centred around expected-random drawdown

Precompute (or estimate online) the expected drawdown of a random
betting policy given the day's variance. Subtract it from the
observed drawdown:

```
shaped = − ε × (observed_drawdown − expected_random_drawdown) / starting_budget
```

Agents doing worse than random drawdown lose reward; agents doing
better than random gain reward. Expected value for a random agent
is exactly zero.

Complication: "expected random drawdown" depends on the day's
realised variance, which you only know in hindsight. Could compute
it in post-hoc settlement using the actual bet magnitudes and a
permutation bootstrap. Ugly but tractable.

### Option B — Paired agent reference

Compare the agent's drawdown to the drawdown a fixed "null policy"
(e.g. "bet nothing" or "bet uniformly random of matching volume")
would have had on the same day. Use the difference, sign-preserved,
as the shaped term. Zero-mean by construction.

Complication: requires running a shadow policy per episode.
Non-trivial compute overhead.

### Option C — Drop the feature, use a different risk knob

Instead of drawdown, penalise high open liability (variance-of-P&L
proxy) using a formulation that's symmetric by construction. For
example, reward `-ε × (open_liability_zscore)²` centred on a moving
average of the agent's own liability — so "higher-than-usual-for-me"
is penalised and "lower-than-usual-for-me" is rewarded. Zero-mean
within the agent's own trajectory.

Complication: may just reproduce what `efficiency_penalty` is
already doing indirectly.

### Design-pass exercise

**Do not write any code until one of these (or a fourth option you
come up with) is chosen and written up in this file, below the line
of dashes, with:**

- A worked example showing the expected shaped reward for a random
  policy is exactly zero (or arbitrarily close).
- A worked example showing a drawdown-avoiding policy gets positive
  shaped reward.
- A worked example showing a drawdown-amplifying policy gets
  negative shaped reward of similar magnitude.
- A description of where in `_settle_current_race` the term is
  computed and which accumulator (`_cum_shaped_reward` — this is
  shaping, not real money) it lands in.
- A gene name, type, and sensible range.

Paste the design here, commit it as "design pass for Session 7", and
only then start implementation.

---

## After design approval — scope

**In scope once design is approved:**
- Implement the chosen formulation in `_settle_current_race`.
- Add the gene to `config.yaml` search_ranges.
- Plumb via the reward-overrides path from Session 1.
- Tests (see below).
- Append UI work to `ui_additions.md` Session 7 items.

**Out of scope:**
- The "hold cost per open-liability-tick" idea discussed in design
  review. That has the same asymmetric pitfalls and deserves its own
  design pass — not this session.

## Tests (once code exists)

Create `tests/arch_exploration/test_drawdown_shaping.py`:

1. **Gene sampling.** Gene present, in range.

2. **Zero-mean for random policy.** Build a synthetic day where the
   agent bets uniformly at random on every race. Run the full
   settlement. Average the shaped contribution over N=1000 seeds.
   Assert the mean is within 2 standard errors of zero. This is the
   critical invariant test.

3. **Drawdown-avoiding policy gets positive shaping.** Construct a
   synthetic agent that closes positions early (lower drawdown than
   random) and assert shaped contribution > 0.

4. **Drawdown-amplifying policy gets negative shaping.** Construct
   a synthetic agent that increases bet sizing after losses and
   assert shaped contribution < 0.

5. **Raw+shaped invariant.** `raw + shaped ≈ total_reward` on a
   synthetic day with the new term active.

6. **Bucketing.** The new term lands in `_cum_shaped_reward`, NOT
   `_cum_raw_reward`. Assert via `info["shaped_bonus"]`.

All CPU, all fast except the zero-mean test which may need N=1000
synthetic episodes — budget ~5 seconds, don't let it balloon.

## Session exit criteria

- Design pass approved and pasted into this file below the dashes.
- All tests pass.
- `progress.md` Session 7 entry describing the chosen formulation.
- `lessons_learnt.md` updated with any gotchas discovered during the
  design pass. **Especially** document the false starts — they are
  useful to future-you.
- `ui_additions.md` Session 7 items present.
- Commit.

## Do not

- Do not skip the design pass. The commit message "design pass for
  Session 7" should exist before any implementation code is
  committed.
- Do not use an asymmetric formulation, even if it looks harmless.
  The zero-mean test must pass by construction, not by luck.
- Do not route the new term through `_cum_raw_reward`. It is
  shaping; it must not pretend to be real money.

---

## DESIGN PASS (2026-04-07)

### Chosen formulation — Option D: reflection-symmetric range position

A fourth option, not among A–C. The drawdown signal is emitted per
race settlement as:

```
peak_t   = max(peak_{t-1},  day_pnl_t)      # running high
trough_t = min(trough_{t-1}, day_pnl_t)     # running low
shaped_drawdown_t = ε × (2·day_pnl_t − peak_t − trough_t) / starting_budget
```

with `peak_0 = trough_0 = 0` (i.e. before any race is settled).

Intuition: `(2·day_pnl − peak − trough) / (peak − trough)` ∈ [−1, +1]
is the signed position inside the running range — `+1` at a fresh
high, `−1` at a fresh low, `0` at the midpoint. We keep the un-
normalised numerator divided by `starting_budget` instead of the
range span so the term is well defined when `peak == trough` (early
in the day) and scales with £, like the rest of the reward stack.

The sum over the whole day telescopes to a quantity that is zero in
expectation for any path-reflection-symmetric policy (see proof
below).

### Why Option D and not A–C

- **Option A (centred around expected-random drawdown)** requires a
  per-day bootstrap of random-policy drawdown in hindsight.
  Technically possible, but it puts an O(N²) permutation loop
  inside `_settle_current_race` and makes the invariant depend on
  bootstrap noise, not algebra.
- **Option B (paired shadow policy)** doubles episode cost —
  unacceptable for a CPU-only session, and inconsistent with the
  rest of the shaping stack which is all closed-form.
- **Option C (liability z-score centred on own moving average)**
  overlaps with `efficiency_penalty` and reduces to "bet less than
  you've been betting lately", which is already punished.
- **Option D** is closed-form, O(1) per race, provably zero-mean
  under reflection, and gives signal on the precise thing we care
  about: how much time day_pnl spends near the running low vs the
  running high.

### Zero-mean proof (reflection symmetry)

Let `X_t = day_pnl_t` be the cumulative P&L trajectory after race
`t`, with `X_0 = 0`. Define `M_t = max_{0..t} X` and
`m_t = min_{0..t} X`, starting at `M_0 = m_0 = 0`.

Consider the reflected trajectory `X'_t = −X_t`. Then
`M'_t = max_{0..t} (−X) = −min_{0..t} X = −m_t` and likewise
`m'_t = −M_t`. So the per-race shaped term on the reflected path is

```
2·X'_t − M'_t − m'_t = −2·X_t − (−m_t) − (−M_t)
                     = −(2·X_t − M_t − m_t)
```

The sum over the full trajectory is antisymmetric under reflection.
A random back/lay policy at fair prices produces a trajectory whose
reflection is equally likely (lay at price `p` is the exact
sign-flip of back at the same price, up to commission), so each
path and its reflection contribute equal and opposite shaped reward.
Averaged over the reflection pair — and therefore averaged over the
whole random-policy distribution — the shaped term is **exactly
zero**. Commission introduces a tiny drift away from this
(sub-per-cent in magnitude at the default 5 % rate), comparable to
the already-accepted drift in the existing `early_pick_bonus` and
`precision_bonus` terms which are also "zero-mean modulo
commission".

### Worked example 1 — random policy, four-race day

Path A, cumulative day_pnl after each race: `[+10, −10, +5, 0]`.

| race | day_pnl | peak | trough | 2·X − peak − trough |
|------|---------|------|--------|---------------------|
| 1    | +10     | 10   | 0      | 2·10 − 10 − 0 = +10 |
| 2    | −10     | 10   | −10    | 2·(−10) − 10 − (−10) = −20 |
| 3    | +5      | 10   | −10    | 2·5 − 10 − (−10) = +10 |
| 4    | 0       | 10   | −10    | 2·0 − 10 − (−10) = 0 |

Sum of shaped numerator: `+10 − 20 + 10 + 0 = 0`.
Reflected path A' = `[−10, +10, −5, 0]`:

| race | day_pnl | peak | trough | 2·X − peak − trough |
|------|---------|------|--------|---------------------|
| 1    | −10     | 0    | −10    | −10 |
| 2    | +10     | 10   | −10    | +20 |
| 3    | −5      | 10   | −10    | −10 |
| 4    | 0       | 10   | −10    | 0 |

Sum: `−10 + 20 − 10 + 0 = 0`. Both paths individually happen to be
zero here; in general paths and their reflections cancel pairwise.
E[shaped] over the random-policy distribution = 0.

### Worked example 2 — drawdown-avoiding policy

Steady small profits: day_pnl trajectory `[+2, +4, +6, +8]`.

| race | day_pnl | peak | trough | 2·X − peak − trough |
|------|---------|------|--------|---------------------|
| 1    | +2      | 2    | 0      | +2 |
| 2    | +4      | 4    | 0      | +4 |
| 3    | +6      | 6    | 0      | +6 |
| 4    | +8      | 8    | 0      | +8 |

Sum: `+20`. Normalised by `starting_budget=100`, with `ε=0.05`:
`shaped ≈ +0.010` per episode. Positive, as required.

### Worked example 3 — drawdown-amplifying policy

A policy that keeps losing after an initial loss (monotone
drawdown): `[−10, −5, −15, −10]`.

| race | day_pnl | peak | trough | 2·X − peak − trough |
|------|---------|------|--------|---------------------|
| 1    | −10     | 0    | −10    | 2·(−10) − 0 − (−10) = −10 |
| 2    | −5      | 0    | −10    | 2·(−5) − 0 − (−10) = 0 |
| 3    | −15     | 0    | −15    | 2·(−15) − 0 − (−15) = −15 |
| 4    | −10     | 0    | −15    | 2·(−10) − 0 − (−15) = −5 |

Sum: `−30`. Normalised: `shaped ≈ −0.015` per episode. Negative,
similar magnitude to example 2 (20 vs 30 numerator).

### Implementation location

In `env/betfair_env.py`:

1. `__init__` reads `reward_cfg["drawdown_shaping_weight"]` (default
   `0.0`) into `self._drawdown_shaping_weight`. Zero disables the
   feature so existing runs are byte-identical unless the gene is
   explicitly set.
2. `reset()` initialises `self._day_pnl_peak = 0.0` and
   `self._day_pnl_trough = 0.0` alongside the existing
   `_cum_*_reward` bookkeeping.
3. `_settle_current_race` — after `self._day_pnl += race_pnl` and
   the existing `early_pick_bonus` / `precision_reward` /
   `efficiency_cost` block, compute:

   ```python
   if self._drawdown_shaping_weight > 0.0 and self.starting_budget > 0:
       self._day_pnl_peak = max(self._day_pnl_peak, self._day_pnl)
       self._day_pnl_trough = min(self._day_pnl_trough, self._day_pnl)
       drawdown_term = (
           self._drawdown_shaping_weight
           * (2.0 * self._day_pnl - self._day_pnl_peak - self._day_pnl_trough)
           / self.starting_budget
       )
   else:
       drawdown_term = 0.0

   shaped = early_pick_bonus + precision_reward - efficiency_cost + drawdown_term
   ```

   `drawdown_term` lands in `shaped`, which accumulates into
   `self._cum_shaped_reward` — **not** `_cum_raw_reward`. It is
   training signal, not cash. The running peak/trough are pure
   diagnostic state and never touch `_day_pnl` or `BetManager`.
4. `_REWARD_OVERRIDE_KEYS` gains `"drawdown_shaping_weight"` so the
   per-agent override path from Session 1 can set it.

### Gene, type, range

- **Gene name (hyperparameter / schema):** `reward_drawdown_shaping`
  (matches the existing `reward_efficiency_penalty` /
  `reward_precision_bonus` naming — "reward_" prefix signals the
  subsystem).
- **Env reward-config key (the thing `reward_overrides` carries):**
  `drawdown_shaping_weight` (matches `efficiency_penalty` /
  `precision_bonus` naming — no redundant "reward_" prefix inside
  the reward block).
- **Mapping:** `_REWARD_GENE_MAP["reward_drawdown_shaping"] =
  ("drawdown_shaping_weight",)` in `agents/ppo_trainer.py`.
- **Type:** `float` (`type: float` in `config.yaml` search_ranges).
- **Range:** `[0.0, 0.2]`. Rationale: at the top end, a
  trajectory that spent the whole day at the running trough
  contributes roughly `−0.2 × N_races` to shaped reward, i.e.
  `≈ −5` on a 25-race day — comparable to the existing
  `precision_bonus` (max 3.0) and `terminal_bonus_weight` (max 3.0)
  contributions. The lower bound `0.0` disables the term cleanly
  for agents that don't want it (survivors from earlier generations
  with the old genome).
- **Default in `config.yaml` reward block:** `0.0`, so any non-
  genetic training path stays on the existing reward stack.

### Not in scope (explicit)

- No per-tick drawdown shaping. Only race-settlement-time updates —
  the rest of the shaping stack already lives in
  `_settle_current_race` and the zero-mean proof assumes the
  discrete settlement schedule.
- No "hold cost per open-liability-tick" (explicitly out of scope
  per the session plan — needs its own design pass).
- No exposure of `_day_pnl_peak` / `_day_pnl_trough` on
  `info[...]` yet. Happy to add if a diagnostic test asks for it,
  but no logging consumer needs it today.

---
