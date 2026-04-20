"""Pre-launch checklist for arb-curriculum-probe.

Run this BEFORE launching Session 07 to confirm the Session 06 manual
operator steps (archive + reset + config changes + oracle scan) are done.

Usage (from repo root):
    python scripts/check_arb_curriculum_prereqs.py

Exit codes:
    0: all checks pass — safe to launch
    1: one or more checks failed — do not launch yet
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import yaml


PROBE_PLAN_NAME = "arb-curriculum-probe"
REQUIRED_CURRICULUM_MODE = "density_desc"
ORACLE_CACHE_DIR = Path("data/oracle_cache")
REGISTRY_DB = Path("registry/models.db")
TRAINING_PLANS_DIR = Path("registry/training_plans")
CONFIG_PATH = Path("config.yaml")


def _check(label: str, passed: bool, detail: str = "") -> bool:
    tag = "PASS" if passed else "FAIL"
    msg = f"  [{tag}] {label}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    return passed


def main() -> int:
    print("=" * 60)
    print("arb-curriculum-probe pre-launch checklist")
    print("=" * 60)
    results: list[bool] = []

    # ── 1. config.yaml has curriculum_day_order: density_desc ─────────────
    try:
        config = yaml.safe_load(CONFIG_PATH.read_text())
        mode = config.get("training", {}).get("curriculum_day_order", "random")
        results.append(_check(
            f"config.yaml curriculum_day_order = {REQUIRED_CURRICULUM_MODE!r}",
            mode == REQUIRED_CURRICULUM_MODE,
            f"Current value: {mode!r}  "
            f"(set training.curriculum_day_order: {REQUIRED_CURRICULUM_MODE} in config.yaml)",
        ))
    except Exception as exc:
        results.append(_check("config.yaml readable", False, str(exc)))

    # ── 2. registry/models.db has 0 models (fresh registry) ───────────────
    if REGISTRY_DB.exists():
        try:
            conn = sqlite3.connect(REGISTRY_DB)
            n = conn.execute("SELECT count(*) FROM models").fetchone()[0]
            conn.close()
            results.append(_check(
                "registry/models.db has 0 models (fresh registry)",
                n == 0,
                f"Found {n} model(s) — run ModelStore() to initialise a fresh registry, "
                "or verify you archived the old one.",
            ))
        except Exception as exc:
            results.append(_check("registry/models.db readable", False, str(exc)))
    else:
        results.append(_check(
            "registry/models.db exists",
            False,
            "File missing — run: from registry.model_store import ModelStore; ModelStore()",
        ))

    # ── 3. arb-curriculum-probe plan exists in draft status ───────────────
    if TRAINING_PLANS_DIR.exists():
        plans = list(TRAINING_PLANS_DIR.glob("*.json"))
        probe_plan = None
        for p in plans:
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                if d.get("name") == PROBE_PLAN_NAME:
                    probe_plan = d
                    break
            except Exception:
                continue
        if probe_plan is not None:
            status = probe_plan.get("status", "?")
            pid = probe_plan.get("plan_id", "?")[:8]
            results.append(_check(
                f"Plan '{PROBE_PLAN_NAME}' exists with status=draft",
                status == "draft",
                f"plan_id={pid}... status={status!r}",
            ))
            # Verify key fields
            seed = probe_plan.get("seed")
            pop = probe_plan.get("population_size")
            gens = probe_plan.get("n_generations")
            anneal = probe_plan.get("naked_loss_anneal")
            results.append(_check(
                "Plan fields: seed=7919, pop=33, gens=4, naked_loss_anneal set",
                seed == 7919 and pop == 33 and gens == 4 and anneal is not None,
                f"seed={seed} pop={pop} gens={gens} anneal={anneal}",
            ))
        else:
            results.append(_check(
                f"Plan '{PROBE_PLAN_NAME}' found in registry/training_plans/",
                False,
                f"Checked {len(plans)} plan(s). "
                "Run scripts/create_arb_curriculum_probe.py or Session 06 manually.",
            ))
    else:
        results.append(_check(
            "registry/training_plans/ exists",
            False,
            "Directory missing.",
        ))

    # ── 4. oracle cache exists for at least some training dates ───────────
    if ORACLE_CACHE_DIR.exists():
        date_dirs = [d for d in ORACLE_CACHE_DIR.iterdir() if d.is_dir()]
        npz_count = sum(
            1 for d in date_dirs
            if (d / "oracle_samples.npz").exists()
        )
        zero_density = []
        for d in date_dirs:
            hdr = d / "header.json"
            if hdr.exists():
                try:
                    data = json.loads(hdr.read_text())
                    if float(data.get("density", 0)) == 0.0:
                        zero_density.append(d.name)
                except Exception:
                    pass
        results.append(_check(
            f"Oracle cache has ≥1 date with samples",
            npz_count > 0,
            f"Dates with cache: {npz_count}/{len(date_dirs)}. "
            "Run: python -m training.arb_oracle scan --dates <dates>",
        ))
        if zero_density:
            print(
                f"         ⚠ {len(zero_density)} date(s) have density=0 "
                f"(sparse days): {zero_density[:5]}"
            )
    else:
        results.append(_check(
            "data/oracle_cache/ exists",
            False,
            "Run: python -m training.arb_oracle scan --dates <your_training_dates>",
        ))

    # ── 5. episodes.jsonl is empty (fresh start) ──────────────────────────
    episodes_log = Path("logs/training/episodes.jsonl")
    if episodes_log.exists():
        size = episodes_log.stat().st_size
        results.append(_check(
            "logs/training/episodes.jsonl is empty (fresh start)",
            size == 0,
            f"Current size: {size} bytes. "
            "If this is pre-run data, archive it first (Session 06 step 3).",
        ))
    else:
        results.append(_check(
            "logs/training/episodes.jsonl exists",
            False,
            "Create it: New-Item -ItemType File -Force logs/training/episodes.jsonl",
        ))

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    passed = sum(results)
    total = len(results)
    print(f"Result: {passed}/{total} checks passed")
    if passed == total:
        print("✓ All checks passed — safe to launch arb-curriculum-probe.")
        return 0
    else:
        print("✗ Fix the failed checks before launching.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
