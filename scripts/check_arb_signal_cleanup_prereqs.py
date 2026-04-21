"""Pre-launch checklist for arb-signal-cleanup-probe.

Run this BEFORE launching each cohort to confirm the plan files are in
place, config.yaml floors are at their disabled defaults, and the prior
probe is in a clean terminal state.

Usage (from repo root):
    python scripts/check_arb_signal_cleanup_prereqs.py

Exit codes:
    0: all checks pass -- safe to launch
    1: one or more checks failed -- do not launch yet

Notes
-----
* The three-cohort ablation is implemented as three serial plan files
  (``arb-signal-cleanup-probe-A / -B / -C``) because ``TrainingPlan``
  has no per-sub-population gene override. This script verifies all
  three exist with ``status == "draft"``.
* ``force_close_before_off_seconds`` and ``shaped_penalty_warmup_eps``
  are CONFIG-LEVEL knobs; the plan data model cannot override them per
  cohort. This script enforces that the CONFIG FLOOR is 0 / 0 --
  i.e. non-probe runs inherit the disabled default. The operator must
  FLIP those values in ``config.yaml`` between cohort launches as
  documented in ``plans/arb-signal-cleanup/progress.md`` Session 03's
  launch sequence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


CONFIG_PATH = Path("config.yaml")
ORACLE_CACHE_DIR = Path("data/oracle_cache")
TRAINING_PLANS_DIR = Path("registry/training_plans")
PRIOR_PROBE_PLAN_ID = "277bbf49-8a2b-4d84-b8a3-3b9286e115eb"
COHORT_PLANS = {
    "A": "arb-signal-cleanup-probe-A",
    "B": "arb-signal-cleanup-probe-B",
    "C": "arb-signal-cleanup-probe-C",
}
EXPECTED_SEEDS = {"A": 8101, "B": 8102, "C": 8103}
EXPECTED_POP = 16
EXPECTED_GENS = 4


def _check(label: str, passed: bool, detail: str = "") -> bool:
    tag = "PASS" if passed else "FAIL"
    msg = f"  [{tag}] {label}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    return passed


def main() -> int:
    print("=" * 60)
    print("arb-signal-cleanup-probe pre-launch checklist")
    print("=" * 60)
    results: list[bool] = []

    # -- 1. config.yaml floors ------------------------------------------------
    try:
        config = yaml.safe_load(CONFIG_PATH.read_text())
    except Exception as exc:
        _check("config.yaml readable", False, str(exc))
        return 1

    constraints = config.get("training", {}).get("betting_constraints", {})
    fcs = int(constraints.get("force_close_before_off_seconds", 0))
    results.append(_check(
        "config.yaml constraints.force_close_before_off_seconds == 0 (floor)",
        fcs == 0,
        (
            f"Current value: {fcs}. Plan files enable this per-cohort "
            "(A, C) but the config-level floor must be 0 so non-probe "
            "runs don't accidentally inherit it. Operator flips the "
            "value in config.yaml immediately before each cohort "
            "launch per progress.md launch sequence."
        ),
    ))

    training_cfg = config.get("training", {})
    warmup = int(training_cfg.get("shaped_penalty_warmup_eps", 0))
    results.append(_check(
        "config.yaml training.shaped_penalty_warmup_eps == 0 (floor)",
        warmup == 0,
        (
            f"Current value: {warmup}. Same floor-vs-launch reasoning "
            "as force_close: operator flips this between cohort "
            "launches per progress.md."
        ),
    ))

    # -- 2. three plan files exist and are draft -----------------------------
    if not TRAINING_PLANS_DIR.exists():
        results.append(_check(
            "registry/training_plans/ exists", False, "Directory missing.",
        ))
    else:
        plans = {}
        for p in TRAINING_PLANS_DIR.glob("*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                plans[d.get("name", "?")] = d
            except Exception:
                continue

        for cohort, expected_name in COHORT_PLANS.items():
            plan = plans.get(expected_name)
            if plan is None:
                results.append(_check(
                    f"Plan '{expected_name}' exists",
                    False,
                    (
                        f"Missing -- run the Session 03 plan-creation "
                        f"script or recreate the file in "
                        f"{TRAINING_PLANS_DIR}/."
                    ),
                ))
                continue

            status_ok = plan.get("status") == "draft"
            seed_ok = plan.get("seed") == EXPECTED_SEEDS[cohort]
            pop_ok = plan.get("population_size") == EXPECTED_POP
            gens_ok = plan.get("n_generations") == EXPECTED_GENS
            anneal = plan.get("naked_loss_anneal")
            anneal_ok = (
                isinstance(anneal, dict)
                and anneal.get("start_gen") == 0
                and anneal.get("end_gen") == 2
            )
            cohort_ok = plan.get("plan_cohort") == cohort
            hp = plan.get("hp_ranges", {}) or {}
            ctx = hp.get("transformer_ctx_ticks", {}) or {}
            ctx_pinned_ok = (
                ctx.get("type") == "int_choice"
                and list(ctx.get("choices", [])) == [256]
            )
            # Cohort A/B expect alpha_lr gene drawn from [1e-2, 1e-1];
            # cohort C omits the gene so the trainer default 1e-2 pins.
            alpha = hp.get("alpha_lr") if cohort in ("A", "B") else None
            if cohort in ("A", "B"):
                alpha_ok = (
                    isinstance(alpha, dict)
                    and alpha.get("type") == "float_log"
                    and abs(float(alpha.get("min", 0)) - 0.01) < 1e-9
                    and abs(float(alpha.get("max", 0)) - 0.1) < 1e-9
                )
            else:
                alpha_ok = "alpha_lr" not in hp

            detail = (
                f"plan_id={plan.get('plan_id', '?')[:8]}... "
                f"status={plan.get('status')!r} seed={plan.get('seed')} "
                f"pop={plan.get('population_size')} "
                f"gens={plan.get('n_generations')} "
                f"plan_cohort={plan.get('plan_cohort')!r} "
                f"ctx_ticks_pinned={ctx_pinned_ok} "
                f"alpha_lr_gene_ok={alpha_ok} "
                f"naked_loss_anneal={anneal}"
            )
            all_ok = (
                status_ok and seed_ok and pop_ok and gens_ok
                and anneal_ok and cohort_ok and ctx_pinned_ok and alpha_ok
            )
            results.append(_check(
                f"Plan '{expected_name}' fields match spec",
                all_ok,
                detail,
            ))

        # -- 3. prior probe still recorded as failed --------------------------
        prior_path = TRAINING_PLANS_DIR / f"{PRIOR_PROBE_PLAN_ID}.json"
        if prior_path.exists():
            try:
                prior = json.loads(prior_path.read_text(encoding="utf-8"))
                results.append(_check(
                    "Prior arb-curriculum-probe (277bbf49) status == 'failed'",
                    prior.get("status") == "failed",
                    (
                        f"status={prior.get('status')!r}. "
                        "Historical record only -- this is a sanity "
                        "check, not a gate."
                    ),
                ))
            except Exception as exc:
                results.append(_check(
                    "Prior arb-curriculum-probe plan readable", False, str(exc),
                ))
        else:
            # Not fatal -- the prior plan may have been manually removed.
            results.append(_check(
                "Prior arb-curriculum-probe (277bbf49) still in registry",
                False,
                (
                    "Not found. If you removed it deliberately, ignore "
                    "this. Otherwise recover from registry archive."
                ),
            ))

    # -- 4. oracle cache has >=1 date ----------------------------------------
    if ORACLE_CACHE_DIR.exists():
        date_dirs = [d for d in ORACLE_CACHE_DIR.iterdir() if d.is_dir()]
        npz_count = sum(
            1 for d in date_dirs
            if (d / "oracle_samples.npz").exists()
        )
        results.append(_check(
            "Oracle cache has >=1 date with samples (BC prereq)",
            npz_count > 0,
            (
                f"Dates with cache: {npz_count}/{len(date_dirs)}. "
                "Run: python -m training.arb_oracle scan --dates <dates>"
            ),
        ))
    else:
        results.append(_check(
            "data/oracle_cache/ exists",
            False,
            "Run the oracle scan on the training-date window.",
        ))

    # -- Summary --------------------------------------------------------------
    print()
    passed = sum(results)
    total = len(results)
    print(f"Result: {passed}/{total} checks passed")
    if passed == total:
        print(
            "OK All checks passed. Before launching each cohort, flip "
            "config.yaml per progress.md launch sequence:\n"
            "  Cohort A: force_close=30, warmup_eps=10\n"
            "  Cohort B: force_close=0,  warmup_eps=0\n"
            "  Cohort C: force_close=30, warmup_eps=10"
        )
        return 0
    print("FAIL Fix the failed checks before launching.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
