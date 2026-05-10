# Planning prompt — rl-betfair v3 (or targeted integration of betfair-predictors models)

Use this prompt to open a planning session in a fresh context, working
in **`C:\Users\jsmit\source\repos\rl-betfair`** (the existing v2). The
prompt is self-contained — it briefs the planner on the proposal,
what exists in both repos, the diagnostic decision the plan must
make first, and the artefact this planning session should produce.
**Do not start writing code in this session. The deliverable is a
plan document.**

---

## The proposal under evaluation

The operator wants to consider **rewriting `rl-betfair` as
`rl-betfair-v3`**, designed from the ground up to use the two
production predictor models from `betfair-predictors` (a calibrated
race-outcome champion at `production/race-outcome/` and a ranking
companion at `production/race-outcome-ranker/`, plus the existing
price-mover champion at `production/direction-predictor/`) as
first-class observation inputs — rather than carrying forward the
multi-head observation accretion of v2.

**The user's framing:** "We would leave rl-betfair as it is, and
carry over as much as we can, but this one would, instead of using
heads and all sorts, use the two models made along with whatever else
was useful, and do its trading days using them."

The operator is unsure whether this is a good idea. **The first job
of this planning session is to honestly answer that question.** Do
not assume the rewrite is the answer. Diagnose first.

## PREREQUISITE — read first

Before forming any opinion:

- `rl-betfair/CLAUDE.md`, `rl-betfair/PLAN.md`, `rl-betfair/PROGRESS.md`,
  `rl-betfair/TODO.md` — current v2 state and direction.
- `rl-betfair/README.md`, `rl-betfair/docs/` — documented architecture.
- `rl-betfair/env/` — the Betfair env (the most load-bearing module
  in v2; understand it before recommending what to keep).
- `rl-betfair/agents/` and `rl-betfair/agents_v2/` — agent code that
  has already accreted across v1→v2.
- `rl-betfair/training/` and `rl-betfair/training_v2/` — same for
  training infrastructure.
- `rl-betfair/data/` — extractor, episode_builder, processed parquet
  pipeline. The `data/processed/*.parquet` files are the same
  artefact `betfair-predictors` consumes.
- `rl-betfair/data/restore-backups.sh` — the manual restore workflow
  the operator uses to ingest StreamRecorder1 backups into the live DB
  and re-extract parquets.
- `betfair-predictors/production/direction-predictor/manifest.json`
  + `README.md` — price-mover champion contract.
- `betfair-predictors/production/race-outcome/manifest.json` +
  `README.md` + `segment_performance.json` — calibrated race-outcome
  champion contract.
- `betfair-predictors/production/race-outcome-ranker/manifest.json`
  + `README.md` + `segment_performance.json` — ranker contract.
- `betfair-predictors/plans/race-outcome-predictor/findings.md` —
  the autonomous-run summary (what the two race-outcome models
  represent, how they were trained, what their strengths and
  weaknesses are).
- `betfair-predictors/plans/race-outcome-predictor/lessons_learnt.md`
  — the durable design vocabulary developed during the run
  (output contracts, segment_performance sidecars,
  two-models-for-different-consumers, value-at-inference-time,
  consumer-specific success criteria).
- `betfair-predictors/docs/intended_consumer.md` — the documented
  RL integration shape (currently unimplemented; that absence is
  itself relevant context).

Read those before opining on rewrite vs integration. Many of the
decisions that look obvious from the outside ("just use the new
models!") have load-bearing detail in the existing env code that
you must understand first.

## Operating principles for THIS planning session

1. **No code. Output is a plan document.** The deliverable is
   `rl-betfair/plans/v3-or-not/plan.md` (or
   `rl-betfair/plans/predictor-integration/plan.md` if the
   diagnosis lands on "no rewrite, just integrate"). Code goes in
   subsequent sessions, gated by the operator's review of the plan.

2. **Diagnose before proposing.** The first deliverable is the
   diagnosis section. It should answer:
   - What in v2 is **load-bearing** (must survive any v3 or
     integration plan)?
   - What in v2 is **vestigial** (heads, code paths, observation
     features that the predictor models supersede)?
   - What in v2 is **mid** — works but would benefit from cleanup
     even without a rewrite?
   - Honestly: how much of v2 is the messy-but-correct env code
     that any v3 has to re-derive, and how much is genuinely
     replaceable?

3. **Frame three options, not one.** The plan must consider:
   - **A. Full rewrite (rl-betfair-v3).** What it would look like,
     what it costs, what risks it carries, what it preserves vs
     re-derives.
   - **B. Targeted integration into rl-betfair v2.** Add a thin
     observation-feature layer that loads the predictor manifests,
     wires `p_win` / `p_placed` / `ranker_top1_high_confidence_flag`
     into the existing observation space, and gates everything
     behind `observations.use_race_outcome_predictor: false` (the
     opt-in flag from `docs/intended_consumer.md`). No rewrite.
   - **C. Partial rewrite — refactor specific modules.** Identify
     the 1-3 modules where v2 has accumulated the most cruft AND
     where the predictor integration is most natural. Refactor only
     those; leave the rest of v2 intact.

   For each option, the plan must give: scope, estimated cost
   (in operator-hours and operator-attention-weeks), preserved
   capabilities, lost capabilities, risk profile, and go/no-go
   gates.

4. **Identify the load-bearing why.** A rewrite needs a one-line
   "why this is worth the cost" answer. The plan must surface that
   answer or admit there isn't one. Vague answers like "the code is
   messy" don't justify a rewrite; specific answers like "v2's
   observation builder hardcodes a 26-feature shape that the agent
   conflates with the price-mover signal, and the new predictor
   outputs need a redesigned shape" do.

5. **Carry-forward inventory.** Whatever option wins, the plan
   inventories what carries forward from v2:
   - Env mechanics (Betfair API, market replay, action processing
     for back/lay bets, settlement). Probably keep all of this.
   - Data extractor (the parquet pipeline). Probably keep.
   - Agent + training architecture. Most likely candidate for
     replacement or significant refactor.
   - Observation builder. Most likely candidate for redesign around
     the predictor outputs.
   - Reward shaping, episode definition. Decide explicitly per item.

6. **Define the predictor integration contract.** The plan must
   specify, for each of the three production models, exactly how
   v3 (or integrated-v2) consumes their outputs. Reference each
   model's `manifest.json::output_contract` (champion has p_win
   + p_placed; ranker has ranker_score / rank / softmax_share /
   top1_flag / top1_high_confidence_flag). The integration points
   are:
   - **Where in the env loop** are the models called (per market,
     per tick, per episode start)?
   - **What features** end up in the observation vector?
   - **How is `segment_performance.json` consumed?** Per the
     race-outcome README pattern, consumer code loads the sidecar
     at startup and routes per-bucket. The plan must say where
     this routing lives in the env / agent / observation builder.
   - **How is the opt-in flag wired?**
     `observations.use_race_outcome_predictor: false` (default off,
     opt-in per cohort). Specify where the flag lives, who reads
     it, and the byte-identical-observations guarantee when the
     flag is off.

7. **Define a comparison protocol.** v2 stays alive. The plan must
   say how v2 and (v3-or-integrated) get compared. Cohort runs,
   identical evaluation set, what metrics decide which version
   becomes the production training driver.

8. **Define success criteria.** When is the v3 (or integration)
   "done"? Possible bars:
   - Smoke training run completes end-to-end on a small cohort.
   - The predictor outputs visibly influence agent behaviour
     (the agent's actions vary when the flag is on vs off).
   - A real cohort comparison shows v3 (or integrated-v2) is
     **at least as good as v2** on the existing eval metrics.
   - v3 (or integrated-v2) **beats** v2 on a documented metric
     by a documented margin.

   Pick a bar that's achievable in the plan's time-box.

## What to capture in the diagnosis

Specifically, the diagnosis section should answer:

| question | scope of answer |
|---|---|
| What does the v2 env do that any successor MUST also do? | List of mechanics (replay, API, settlement, etc.) |
| What v2 modules have the most accumulated edits / cruft? | Specific file paths + brief reason |
| Where does v2 currently consume the price-mover champion? | Specific call sites; assess whether the integration is clean or messy |
| What's NOT working well in v2 today? | Concrete operator-experienced pain points, not generalities |
| What's the cost of a full rewrite, in operator-time? | Honest estimate (e.g., "8 weeks operator-attention" not "we'll see") |
| What's the cost of targeted integration alone? | Honest estimate |
| What does the operator gain from a rewrite that integration can't deliver? | If nothing concrete, the plan should recommend integration |

## Recommended output structure

```
rl-betfair/plans/<chosen-name>/
├── plan.md                  ← top-level plan; the deliverable
├── purpose.md               ← why this work; the load-bearing why
├── diagnosis.md             ← the v2 audit; what's load-bearing vs cruft
├── options_compared.md      ← A vs B vs C with cost / scope / risk
├── carry_forward.md         ← inventory of what survives from v2
├── integration_contract.md  ← exact predictor wiring spec
├── comparison_protocol.md   ← how v2 vs new gets evaluated
├── success_criteria.md      ← when "done"
├── master_todo.md           ← session-level breakdown if execution proceeds
└── session_prompts/         ← per-session prompts for the implementation phase
    └── ...
```

If the diagnosis recommends "B: targeted integration only," several
of these files collapse — the integration contract becomes the main
artefact. That's fine; the planning artefact still records WHY the
rewrite was rejected.

## What this planning session should NOT do

- Don't start writing v3 code.
- Don't move files around in `rl-betfair/`.
- Don't decide the rewrite is happening before the diagnosis is in.
- Don't underestimate the env code — it's the load-bearing module
  in v2 and probably the biggest source of "things you don't
  realise the old code did until they break."
- Don't skip reading the existing predictor production READMEs
  and segment_performance JSONs. The integration contract has to
  reference them.

## Acceptance criteria for the plan-document deliverable

- `plan.md` (or equivalent top-level doc) exists.
- Diagnosis section is honest — names specific modules, specific
  pain points, specific load-bearing parts.
- Three options (A: full rewrite / B: targeted integration / C:
  partial rewrite) are compared explicitly with cost, scope, risk.
- Predictor integration contract is specified, referencing each
  production model's `manifest.json::output_contract`.
- Comparison protocol against v2 is defined.
- A clear go/no-go recommendation is made, with the load-bearing
  reason for the recommendation in one paragraph.
- Operator can read the plan top-to-bottom and decide rewrite-or-not
  on the basis of it. No further questions.

## After this session

If the diagnosis lands on **A** (full rewrite): the plan-document
becomes the spec for an `rl-betfair-v3` plan with its own
`master_todo.md` and session prompts, mirroring the structure of
`betfair-predictors/plans/race-outcome-predictor/`. The
autonomous-run pattern from there is reusable — including the
operator-override discipline and the lessons_learnt accumulation.

If the diagnosis lands on **B** (targeted integration): the plan-
document becomes the spec for an integration plan in
`rl-betfair/plans/predictor-integration/` with its own session
prompts. Smaller scope, faster delivery, leaves v2's working
code alone.

If the diagnosis lands on **C** (partial rewrite): the plan-
document scopes which modules and why. Probably 1-3 sessions of
targeted work rather than a full rewrite cycle.

The autonomous-loop pattern (`/loop @<prompt>` in the v3 or
integration plan's `session_prompts/00_autonomous_full_run.md`) can
drive whichever option the diagnosis chooses.

---

## Why this prompt exists in betfair-predictors and not rl-betfair

The proposal is a CROSS-REPO change — rl-betfair will consume
betfair-predictors' production models. The integration contract
references files in this repo (manifests, segment_performance.json,
READMEs). Storing this prompt here keeps the cross-repo dependency
visible in the predictor repo's own plan history, even though the
work it triggers happens in rl-betfair.

A copy of this prompt should also land in `rl-betfair/incoming/` so
the rl-betfair side has a record. (Mirrors the
`StreamRecorder1/incoming/sp_capture_regression_*.md` pattern.)
