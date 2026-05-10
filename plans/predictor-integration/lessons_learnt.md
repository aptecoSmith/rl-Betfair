# Lessons learnt — predictor-integration

This file is the durable record of non-obvious things this plan
discovered while executing. Empty at plan-open. Each session
appends as evidence demands. Format matches the convention from
`plans/rewrite/phase-7-port-aux-heads/lessons_learnt.md`:

```
## YYYY-MM-DD — Session NN: short title

The lesson in a paragraph or two. Includes:
- WHAT happened (specific observation)
- WHY it surprised us (or why we should remember it)
- WHAT to apply (specific guidance for future plans)
- LINKS (file paths, commit hashes, log lines)
```

Cross-cutting lessons that apply beyond this plan should be
promoted to CLAUDE.md once the plan exits.

---

## 2026-05-10 — Session 01: real `segment_performance.json` carries `consumer_hint = "neutral"` (not in contracts)

**WHAT happened.** Loaded the production `segment_performance.json`
sidecars for the race-outcome champion and ranker. The champion's
JSON contains four hint values:
`{"strong", "weak", "neutral", "insufficient_data"}`. The ranker's
contains only `{"strong", "insufficient_data"}`. `predictor_contracts.md`
§1's "Hints — RECEIVED BY THE LOADER" enumerates only three
(`strong`, `weak`, `insufficient_data`).

**WHY it surprised.** The contracts doc was authored from the manifest
text summary, which only mentions strong / weak / insufficient_data.
The actual JSON has buckets in the middle ground (e.g. `sp_band =
odds-on(<2.0)` returns `neutral` — the model neither has edge nor
has it lost; the bucket is real but the alpha is not). The loader's
strict `ConsumerHint(value)` enum constructor raised on first contact
with `neutral`.

**WHAT to apply.** When wiring an external contract, smoke-load the
real artefact end-to-end before locking the schema. The contract
doc is a summary of the producer's intent; the JSON is the truth.
Updated `predictors/segment_router.py::ConsumerHint` to include
`NEUTRAL`; the `lookup` reducer treats it as not-strong (no STRONG
vote) but not aggressively skip-worthy either — it falls through to
the INSUFFICIENT_DATA tier when no STRONG axis votes. Worth back-
porting to `predictor_contracts.md` §1 in a follow-on edit so the
next consumer doesn't trip the same enum.

**LINKS.** `predictors/segment_router.py` (this plan, Session 01).
`betfair-predictors/production/race-outcome/segment_performance.json`
contains the literal `"neutral"` value at e.g.
`by_sp_band[odds-on(<2.0)]`.

## 2026-05-10 — Session 01: GBM `EncoderState` is not persisted with weights

**WHAT happened.** Inspected the production
`weights.joblib` blobs for both the race-outcome champion
(`1c15250ee90d1b65`) and ranker (`b23018bf5c8bcc70`). The pickled
dicts carry `win` / `placed` / `win_ranker` LightGBM heads,
`feature_names`, `FitArtifacts`, and `params` — but the
`EncoderState` produced by
`scripts/outcome_predictor/datasets.py::fit_encoders(train_df, variant)`
is NOT in the payload. It's also not in
`registry/outcome_predictor/<experiment_id>.joblib`. The
training-side script fits encoders, applies them, and discards
the state object once features are baked into the matrix.

**WHY it surprised.** The `intended_consumer.md` §"What rl-betfair
needs from a release" lists "weights file", "manifest", and
"stable inference code" — implying inference is reproducible from
those three. It is, IF the consumer can re-fit the encoders
against the same training corpus the model trained on. That
re-fit step is mechanically simple but couples the rl-betfair
worker to the predictor repo's `data/outcome_dataset/` parquet
tree, which is not part of the production-bundle contract.

**WHAT to apply.** Workaround in this plan: lazy-fit the encoder
at `PredictorBundle.from_manifests` time using `load_split` +
`fit_encoders` from the predictor repo (~16s end-to-end bundle
load including F2 aggregates; encoder alone is ~1-2s). The fit
state is held read-only on the bundle (`champion_encoder`,
`ranker_encoder`); `predict_race` will call `apply_encoders(df,
state)` per market. Cold-start values (course / sex / headgear
unseen during predictor training) map to UNKNOWN per the
predictor repo's §9 contract, so this is safe even when
rl-betfair sees a course the predictor never saw.

Filed a request to persist the encoder alongside weights at
`betfair-predictors/incoming/persist_encoder_state_alongside_weights.md`
per the cross-repo postbox convention. Once that lands, the
bundle's lazy-fit can drop to a single `joblib.load` of the
sidecar.

**LINKS.**
- `predictors/loader.py::_fit_categorical_encoder` (the workaround).
- `betfair-predictors/scripts/outcome_predictor/datasets.py::fit_encoders`
  / `apply_encoders` / `EncoderState`.
- `betfair-predictors/incoming/persist_encoder_state_alongside_weights.md`
  (the postbox drop).
