---
plan: trajectory-retrieval-probe
status: draft
---

# Hard constraints — trajectory-retrieval-probe

Lock rules for the probe. Any violation rolls back the commit.

## §1 — Zero production-path edits

The probe touches **none** of the following:

- `env/` (any file)
- `agents/`, `agents_v2/`
- `training/`, `training_v2/`
- `tests/` (no new or modified tests in the production suite)
- `config.yaml`, `pyproject.toml`, top-level requirements files
- Any `plans/` folder other than `plans/trajectory-retrieval-probe/`

Allowed surfaces:

- New files under `scripts/` (probe entry point + helpers)
- New files under `plans/trajectory-retrieval-probe/`
- New scratch output directory under `scratch/trajectory_retrieval/`
  (gitignored; created by the script if absent)

Rationale: this is a side-thread experiment with a binary outcome
(go / no-go). It must not interleave with production training runs
or the operator's deploy candidates. The "no production-path edits"
rule is what makes the probe safe to run in parallel with anything
else.

## §2 — Held-out split fixed before any feature engineering

The day split is locked at the start of Phase 1 and never changes:

- **Index days:** `2026-04-06` through `2026-05-04` (29 days
  inclusive of the day-file gaps in `data/processed/`).
- **Query days:** `2026-05-05` through `2026-05-14` (10 days).
- **Validation days:** `2026-05-15` through `2026-05-20` (6 days)
  — never looked at during feature iteration; only used for the
  final go/no-go decision.

(Earlier draft of this file claimed "60 days index" / "16 days
query" — that was based on a miscount of `data/processed/` which
holds both day and `_runners` parquets, so 86 files = 43 days. The
date ranges above are the authoritative split. Reality of 29 / 10
/ 6 days gives plenty of feature rows: post-tick-direction-fix,
Phase 2 produces 18,033 index / 6,934 query / 4,703 validation
feature rows out of 32,903 race-runners (≈90 % coverage).)

Reason: if we look at validation-day results during feature
iteration, we've leaked. The decision rule in
[purpose.md](purpose.md) is meaningful only against an honest split.

The cohort-side leak risk noted in
`feedback/project_select_days_data_dir_dependence.md` doesn't
directly apply (the probe doesn't share days with any training
selector), but the same hygiene principle does — we fix the split
before we look at any data.

## §3 — No look-ahead in feature construction

Every feature at query time `D` is computed from ticks `≤ D`.
Specifically:

- Slopes and volatilities use only ticks in `[D − window, D]`.
- "Cumulative traded volume" sums only ticks `≤ D`.
- Favourite-rank uses LTPs at exactly `D`, not any tick after.
- Form / past-races data from the runners parquet is timestamp-free
  at runner level (pre-race static), but if we ever incorporate
  *intra-race* features from elsewhere, the `≤ D` invariant binds.

A `test_no_lookahead_in_features` smoke check in the probe script
asserts that perturbing any tick at index `> D` leaves the feature
vector at `D` unchanged. Cheap to run; load-bearing.

## §4 — Decision rule frozen before validation pass

The four outcome bands in [purpose.md](purpose.md#decision-rule) are
the literal text used to interpret final results. We do not
post-hoc adjust thresholds ("10 %" → "8 %") after seeing numbers.

If results land between bands (e.g. beats B1 by 7 %), the outcome
is "marginal" by the locked rule — not "actually pretty good, let's
keep going". The decision rule's whole point is to defend against
motivated reasoning at the moment the experiment most invites it.

## §5 — No FAISS / no fancy index

v1 uses `sklearn.neighbors.NearestNeighbors` (brute-force). At ~77k
× 10-float vectors this is ~3 MB, sub-second per query. Adding
FAISS or any approximate-NN library at probe stage is
over-engineering and adds an install dependency we don't need.

If the probe succeeds and a follow-on plan needs >1M vectors, FAISS
goes in that plan, not this one.

## §6 — Failure modes are reported, not silenced

Three specific failure modes must surface in the probe's output
parquet rather than being averaged away:

- **Query rows with insufficient history** (race started <30 min
  before its earliest tick, or runner had <10 ticks before D) —
  marked with a quality flag, excluded from the headline MAE, but
  counted in the report.
- **Queries with low neighbour agreement** (high variance across
  top-k continuations) — reported as a separate metric. A method
  that's right on confident queries and refuses to predict on
  unconfident ones is more useful than one that's mediocre on
  everything.
- **Per-venue and per-favourite-rank breakdown** — averages can hide
  the case where the method works on favourites but fails on
  outsiders (or vice versa). Both breakdowns ship in the report.

## §7 — Scratch outputs are throwaway

Everything written to `scratch/trajectory_retrieval/` is ephemeral.
The path is gitignored. If the probe fails, the directory is
deleted; if it succeeds, the headline numbers move into
[findings.md](findings.md) and the raw artifacts are still
considered throwaway (re-runnable from the script).

This stops the probe's intermediate state from drifting into
production data paths.
