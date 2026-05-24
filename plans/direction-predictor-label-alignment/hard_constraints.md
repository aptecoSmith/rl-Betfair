# Hard constraints — direction predictor / label alignment

Invariants the fix must NOT violate. These are load-bearing — surface
in tests so future drift is caught at CI time.

## §1. Label cache backward compatibility

Old `data/direction_labels/<date>/horizon60_thresh5_fc60_*.json|npz`
files MUST continue to load via the existing
`training_v2.direction_label_scan.load_labels` API without error.

If the new horizon mode emits new cache files, they MUST live at a
different path (e.g. include the new horizon-units suffix in the
filename) so the two cache schemas coexist. Old runs replicating
prior cohorts must be able to read the old caches; new runs MUST
use the new caches.

**Rationale:** existing scoreboard rows reference the old cache
files via the per-day artifact paths; deleting them would break
reproducibility.

## §2. Default horizon picks "7m" because the predictor fires at 7m

The label cache scan default for the new "time horizon" mode MUST
be 7 minutes (~420 seconds), matching `val_metrics.dir_acc_k5_7m`
in the predictor manifest.

Operator can override the seconds via a new CLI flag; the gene-side
`direction_horizon_ticks` stays an operator-pinned constant in the
trainer until a follow-on plan promotes it.

**Rationale:** the 2026-05-24 findings show the predictor's 7m fire
output is the primary deployment signal (`dir_fires_k5_7m=2936`
on val, `backtest_pnl_k5_7m=2670` on test). Aligning labels to the
same horizon is the smallest change that closes the semantic gap.

## §3. Threshold of 5 ticks stays the default

The predictor fires at k=5 ticks (`dir_acc_k5_7m`). Our label's
default `direction_threshold_ticks=5` already matches. The fix must
NOT change this.

## §4. Force-close horizon respected

When `force_close_before_off_seconds > 0`, the label scan still
clips the forward window so the label doesn't peek past the force-
close boundary (which is when an agent would have to flatten any
open pair in live). The new time-based scan MUST clip the same way.

**Rationale:** an agent that learns "back will be favourable in 7
minutes" is useless if it can only hold the position for the next
30 seconds before forced flatten.

## §5. Cache header schema bump

The new cache files MUST carry a new `label_version` value (e.g.
"2026-05-24-time-horizon-v1") so the loader can detect mismatched
caches at load time. The existing pre-flight check
(`_preflight_cache_schema_check` in `training_v2/cohort/runner.py`)
MUST be extended to validate the cache's `label_version` against
the env's currently-expected value, mirroring the existing
`obs_schema_version` check.

**Rationale:** today's 2026-05-24 cohort crash on stale schema
versions is the precedent — fail fast at launch, with re-scan
commands in the error message.

## §6. Regression tests

The fix MUST land alongside regression tests in
`tests/training_v2/test_direction_label_scan.py` covering:

* §1: an old-format cache loads without error.
* §2: the new default horizon is 420 seconds (or the
  manifest-derived constant).
* §3: the threshold default is still 5 ticks.
* §4: a synthetic race where the price would cross +5 ticks at
  T+600s (10 min) but the force-close horizon is T+300s (5 min)
  produces label_back=0, not 1.
* §5: cohort runner pre-flight raises ValueError when label_version
  doesn't match the env's expected version.

## §7. No silent override during transition

The fix MUST NOT silently fall back to the old-format cache when
the new-format cache is missing — that would mean a cohort runs on
stale labels without the operator knowing. The pre-flight check
catches this; verify it does.

## §8. Linear probe re-run is the acceptance gate

After the fix, `tools/direction_signal_probe.py` (run on the NEW
caches) MUST report BCE descent ≥ 20% relative versus the
uniform-0.5 pos-weighted floor. If it doesn't, the label
generation is still misaligned and we revisit before launching the
cohort.

## §9. No predictor changes

This plan does NOT modify the pretrained `direction-predictor`
model or its manifest. The predictor is the fixed point we're
aligning labels TO, not the other way around. If a future plan
wants to retrain the predictor, that's a separate effort with its
own validation chain in `betfair-predictors/`.
