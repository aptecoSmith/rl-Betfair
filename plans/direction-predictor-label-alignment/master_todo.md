# Master todo — direction predictor / label alignment

Steps in execution order. Each step has an explicit acceptance
criterion so we know when to move on.

## Step 1 — Verify the predictor's training-label semantics

Read the `betfair-predictors` project's label-generation code (the one
that produced the offline labels the predictor was trained against)
to confirm:

* It uses CLOCK TIME forward windows (not tick counts).
* It uses k=5 ticks as the favourable threshold.
* The primary horizon for the "fire" output is 7 minutes.
* Whether `dir_fire_drift` means "price moves OUT" or "lay-first
  favourable" — these are the same thing on Betfair, but make sure.

Paths to read:
* `C:/Users/jsmit/source/repos/betfair-predictors/scripts/predictor/` —
  training scripts
* `C:/Users/jsmit/source/repos/betfair-predictors/scripts/predictor/datasets.py`
  — feature columns + label generation

**Acceptance:** documented in this file as a §addendum so the next
session doesn't re-derive it.

## Step 2 — Add a time-horizon mode to `direction_label_scan.py`

Add a parameter `direction_horizon_seconds` (float) alongside the
existing `direction_horizon_ticks` (int). When the new param is set,
the scan looks forward in CLOCK TIME using `tick.timestamp - open_tick.
timestamp`, accumulating ticks until the cumulative time exceeds the
horizon. When unset, falls back to the existing tick-count behavior
(back-compat per `hard_constraints.md §1`).

Cache file naming change: when seconds is used, write to
`time_horizon{N}s_thresh{T}_fc{FC}_{header,npz}.json` so old caches
coexist (§1).

Header schema: bump `label_version` (§5) so the loader can reject
mixed cache types.

**Acceptance:** `python -m training_v2.direction_label_cli scan
--dates 2026-04-11 --horizon-seconds 420 --threshold-ticks 5
--force-close-before-off-seconds 60` writes a new file at the new
path with the new label_version.

## Step 3 — Extend the loader + cohort plumbing

`training_v2/direction_label_scan.load_labels`: detect file naming
pattern, load either old or new format, set `loaded.horizon_units`
accordingly. Header check raises ValueError on mismatch.

`training_v2/cohort/runner.py::_preflight_cache_schema_check`: extend
to also verify `label_version` matches when the new horizon mode is
active.

`training_v2/discrete_ppo/trainer.py`: read whichever cache type the
operator/gene asked for, surface the active horizon on the per-day
log line.

**Acceptance:** trainer loads new-format caches without code path
changes elsewhere; pre-flight rejects mismatches.

## Step 4 — Re-scan all 16 training days at 420s horizon

```
python -m training_v2.direction_label_cli scan \
  --dates 2026-04-06,2026-04-08,2026-04-09,2026-04-11,2026-04-12, \
          2026-04-13,2026-04-15,2026-04-16,2026-04-19,2026-04-20, \
          2026-04-22,2026-04-24,2026-04-26,2026-05-02,2026-05-04, \
          2026-05-05 \
  --horizon-seconds 420 --threshold-ticks 5 \
  --force-close-before-off-seconds 60
```

Plus the 10 eval days + 14 monitor days for downstream eval.

**Acceptance:** new cache files exist at the new paths; per-day
positive-class rate is roughly stable (we expect ~10-30% positive
on a 7m horizon, possibly higher than the 60-tick horizon's 15-18%
because 7m is a much larger window — that's fine).

## Step 5 — Re-run `direction_signal_probe.py` on the new labels

```
python tools/direction_signal_probe.py 2026-04-11 2026-04-15 \
  2026-04-19 2026-05-05 --label-root data/direction_labels_7m
```
(or whatever the new dir is called)

**Acceptance:** validation BCE descent ≥ 20% relative below the
uniform-0.5 pos-weighted floor on a held-out 20% split (per §8).
If <20%, the alignment isn't right and we revisit before relaunch.

## Step 6 — Re-run `direction_head_inspection.py` brute-force scan

Run the inspection on agent 1's saved weights using the new labels.
The per-column correlation table SHOULD now show at least one
column (probably `dir_q*_3m` slot = predictor's 7m output OR
`dir_fire_*`) at |rho| ≥ 0.30 with the new labels.

**Acceptance:** at least one obs column correlates |rho| ≥ 0.30 with
new label. (Stronger than logreg in Step 5 because we're looking at
the strongest single column, not a linear combination.)

## Step 7 — Add regression tests per `hard_constraints.md §6`

New file: `tests/training_v2/test_direction_label_scan_time_horizon.py`.
Covers the six tests in §6.

**Acceptance:** `pytest tests/training_v2/test_direction_label_scan_time_horizon.py
-q` passes all six.

## Step 8 — Relaunch cohort with the new labels

Same 12-agent × 3-gen × 16-day config from cohort 1779613306, but
the per-day worker loads the new time-horizon caches.

Operator command additions:
* `--reward-overrides direction_horizon_seconds=420` (or whatever the
  flag ends up being)
* Other genes unchanged from the prior launch

**Acceptance:** cohort starts past pre-flight; gen 1 agent 1's day 5
shows `dir_bce_back/lay` descending below 1.10 (down from 1.14 at the
old floor).

## Notes / open questions

**Q: ticks vs seconds — does scan_day have access to per-tick timestamps?**

A: Yes. The existing `_resolve_close_tick` already uses
`market_start_ts` and per-tick timestamps via the race data. We
just expose a forward-by-time variant of the same walk. Path:
`training_v2/direction_label_scan.scan_day`.

**Q: Will the env's obs schema need updating?**

A: No. The env passes the predictor's quantile output positionally
into stale-named slots (`dir_q*_1m` etc.). The agent learns from
position, not from names. The names ARE misleading and worth
fixing in a follow-on plan, but they don't block this fix.

**Q: Will BC pretrain need re-running?**

A: BC pretrain happens inline at the start of each agent's training,
loading whichever direction_labels cache is configured. So the new
labels flow through automatically once the cache files are in
place. No separate BC step needed.
