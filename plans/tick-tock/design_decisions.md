# Design decisions to settle before building (and the pitfalls)

These are the questions the build session must answer. Several are pitfalls we
have already paid for once.

## CARDINAL — judge Tock vs Tick on HELD-OUT days
We have burned ourselves repeatedly on in-sample vs held-out (memory:
`feedback_always_eval_holdout` — a recipe scored +£260/d in-sample, −£175/d
held-out). A Tock "validating" a hypothesis MUST mean its champions beat the
Tick's on days **neither era trained nor selected on**, scored on **locked_pnl +
naked variance** — NOT a lucky `day_pnl`. Fix a held-out day set up front; never
let it leak into training or selection. Non-negotiable.

## How hard to pin a Tock
Spectrum: (a) pin *all* genes to the hypothesis (tests one point, closes off
everything else), (b) narrow-sample around the hypothesis, (c) pin the 3-5
hypothesised drivers + full-sample the rest. **Lean (c)** — test the hypothesis
without closing off the un-hypothesised genes. Mechanically: today we can pin a
gene to a *value* (`--reward-overrides` / per-gene flags) or sample its *full*
range (`--enable-all-genes`); we cannot yet sample a *narrow band* around a value.
Decide whether "pin drivers + full-sample rest" is enough for v1, or whether to add
a narrow-range-per-gene sampling option.

## Era tagging (or the population becomes soup)
Every cohort needs a metadata stamp: `tick|tock`, the hypothesis, the pinned
genes, the parent analysis report. So the phenotype tool can (i) pool data across
eras for bigger n, (ii) compare tick-vs-tock distributions, (iii) trace which Tock
tested which hypothesis. Needs a small per-cohort metadata file written at launch
+ the tool reading it.

## Marginal correlations ≠ the joint optimum
The analysis gives one-gene-at-a-time correlations; genes interact (X may only help
when Y is high). Use it to pick *candidate* drivers; let the **Tock empirically
test the joint recipe** (it does). The loop self-corrects — a wrong joint
hypothesis fails its held-out test and you have learned something. Don't over-trust
a marginal "driver."

## The compositional-rate trap
maturation / close / naked / stop / force-close rates **sum to ~1** (a simplex of
`pairs_opened`), so raising one mechanically lowers others and the correlations are
partly artefacts of the denominator. The behaviour we actually want is *more
RESOLUTION (matured + closed), less naked/force-close, AND better P&L / lower naked
variance* — not maximising one rate in isolation. **Define the target as a
P&L/variance OUTCOME, with the rates as diagnostics.**

## n / when to commit a Tock
48 agents across many genes × behaviours with no multiple-comparison correction is
thin — early hypotheses will be noisy. They firm up as Ticks accumulate. Consider
gating a Tock on a minimum effect size / confidence rather than chasing the first
weak signal.

## Schedule — strict alternation vs adaptive
Strict Tick-Tock (every other era) is the safe default (guarantees exploration).
Could go adaptive (Tock again if it validated, Tick if it failed) — but only once
the loop has shown it converges. **Start strict.**

## The accumulating population
For the analysis to gain power, pool agents across eras (a growing store). Decide:
one shared cohort dir (champions + register accumulate, as the wrapper already
does within a campaign) vs per-era dirs + a pooling step in the tool. The tool
currently reads ONE cohort dir; cross-dir pooling (split by tag) is an extension.

## Selection metric reminders (from memory, don't relitigate)
- Rank/select on **locked_pnl**, not total `day_pnl` (naked is ~zero-EV variance;
  day-pnl-top surfaces naked-lucky agents that won't generalise).
- **naked variance per leg** is the deployment-critical metric (hard ceiling
  σ_leg ≲ £30). A Tock that "matures more" but still has σ_leg of £150 has not won.
- `force_close` stays **0 in training** (keep the naked signal); apply
  `force_close=120` only at deploy/held-out eval.

---

## RESOLVED — 2026-06-06 design session

Every open question above was settled. The actionable spec is `build_plan.md`;
this is the index of what was decided and the one new finding.

- **Held-out (cardinal):** sealed-7 judge (only 7 of the 10 named sealed dates
  exist on disk); score **locked_pnl + σ_naked_leg**, never day_pnl; report
  **fc=0 AND fc=120**; log every peek; hold a clean final-test set back for the
  eventual deploy candidate. *Erosion handled by "reuse sealed-7 + peek-ledger".*
- **How hard to pin → band-seed + drift.** A new fresh-blood **`seed_bands`**
  mechanism (`--seed-gene NAME=LO:HI`) enables the driver, seeds R1 fresh blood
  within the band, and lets breeding drift from there. This is the keystone code
  change. Subsumes the "narrow-range sampling" idea — point-seed + drift ≈ a
  band. The pin must land in `sample_fresh_blood_genes` (the R1 funnel:
  `init_pbt_population`/`breed_pbt` → `_fresh` → it), **not** `reward_overrides`
  (which never reaches the spec, isn't recorded in `hyperparameters`, and can't
  drift). `make_offspring` already resets *disabled* genes to default, so a seed
  must **enable** the gene to survive breeding.
- **Era tagging → one shared leaderboard** stamped `era_id`/`era_type`/
  `hypothesis_id`; work folder holds the mds. Phenotype discovery runs
  **tick-only** (tock-pinned genes have ~0 variance and would corrupt it).
- **Engine → two file-coupled loops** (python era-loop + scheduled Claude
  worker, marker-file handshake). **Fully autonomous** steady state; the operator
  reviews only the *first* analysis+hypothesis. **Multi-candidate + self-critique**
  brain. Strict alternation. Build order: A–D + one manual cycle, then E–F.
- **Pitfall caught at design time:** the first recipe (`current_state.md §2`)
  pins `bc_learning_rate` high **and** `bc_pretrain_steps→0`, but BC never runs at
  0 steps, so the LR is inert — the +0.52 maturation correlation is a
  co-inheritance confound. Resolve at the first-hypothesis review; it motivates
  the brain's **gene-dependency consistency** self-critique check.
