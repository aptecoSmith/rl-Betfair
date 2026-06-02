"""Step A1 (bc-getting-it-right): build + cache the labelled maturation dataset.

For each requested day, run ONE labelled scan
``scan_day(..., maturation_conditioned=False, maturation_label_out=labels)``
— emits EVERY spread-placeable candidate's full predictor-injected obs
AND a matured/not flag (the env-faithful forward walk to T-fc=120). This
is the SAME data the LightGBM 0.76 probe used, so a policy's mature_prob
AUC measured on it is directly comparable to 0.76.

Positives = matured candidates; HARD NEGATIVES = placeable-but-force-close
(plan §3). Cached per-day to ``<out_dir>/<day>.npz`` so the expensive scan
is paid once.

Day groups (``--days``):
  train8  : the 8 evenly-spaced train days the LightGBM probe used
            (Apr 6 -> May 19). Reproduces the 0.76 reference.
  holdout : the 7 reserved holdout days (May 20,21,22,25,27,28,29). NEVER
            train/select/threshold-tune on these (plan §2).
  all     : train8 + holdout.
  <csv>   : explicit comma-separated YYYY-MM-DD list.

Idempotent: a day whose npz already exists is skipped unless --force.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from training_v2.arb_oracle import _load_config, scan_day  # noqa: E402
from agents_v2.env_shim import DEFAULT_SCORER_DIR  # noqa: E402

DATA_DIR = Path("data/processed")
PRED_MANIFESTS = (
    "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json",
    "C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json",
    "C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json",
)
HOLDOUT_DAYS = [
    "2026-05-20", "2026-05-21", "2026-05-22", "2026-05-25",
    "2026-05-27", "2026-05-28", "2026-05-29",
]
DEFAULT_OUT_DIR = Path("plans/bc-getting-it-right/_cache")


def existing_train_days(n: int = 8) -> list[str]:
    """``n`` evenly-spaced EXISTING processed days in [04-06, 05-19].

    Matches ``maturation_predictability_probe._existing_train_days`` so the
    cached train split is the exact set behind the 0.76 reference.
    """
    import re
    pat = re.compile(r"^2026-\d{2}-\d{2}$")
    days = sorted(
        p.stem for p in DATA_DIR.glob("2026-*.parquet")
        if pat.match(p.stem) and "2026-04-06" <= p.stem <= "2026-05-19"
    )
    if len(days) <= n:
        return days
    idx = np.linspace(0, len(days) - 1, n).round().astype(int)
    return [days[i] for i in sorted(set(idx.tolist()))]


def all_train_days() -> list[str]:
    import re
    pat = re.compile(r"^2026-\d{2}-\d{2}$")
    return sorted(
        p.stem for p in DATA_DIR.glob("2026-*.parquet")
        if pat.match(p.stem) and "2026-04-06" <= p.stem <= "2026-05-19"
    )


def resolve_days(spec: str) -> list[str]:
    if spec == "train8":
        return existing_train_days(8)
    if spec == "holdout":
        return list(HOLDOUT_DAYS)
    if spec == "all":
        return existing_train_days(8) + list(HOLDOUT_DAYS)
    if spec == "alltrain":
        return all_train_days()
    return [d.strip() for d in spec.split(",") if d.strip()]


def make_bundle():
    from predictors import PredictorBundle
    champ, rank, dirm = PRED_MANIFESTS
    return PredictorBundle.from_manifests(
        champion_manifest=Path(champ),
        ranker_manifest=Path(rank),
        direction_manifest=Path(dirm),
    )


def build_day(date: str, cfg, bundle, out_dir: Path, force: bool) -> dict | None:
    out = out_dir / f"{date}.npz"
    if out.exists() and not force:
        d = np.load(out)
        n = int(d["matured"].shape[0])
        rate = float(d["matured"].mean()) if n else 0.0
        print(f"[{date}] cached n={n:>6} matured_rate={rate:.4f} (skip)", flush=True)
        return {"date": date, "n": n, "matured_rate": rate, "cached": True}

    t0 = time.time()
    labels: dict = {}
    samples = scan_day(
        date, DATA_DIR, cfg, scorer_dir=DEFAULT_SCORER_DIR,
        predictor_bundle=bundle,
        use_race_outcome_predictor=True,
        use_direction_predictor=True,
        predictor_lean_obs=False,
        maturation_conditioned=False,        # emit ALL candidates
        force_close_before_off_seconds=120.0,
        maturation_label_out=labels,         # ...but label each one
    )
    if not samples:
        print(f"[{date}] no samples", flush=True)
        return None
    obs = np.stack([s.obs for s in samples], axis=0).astype(np.float32)
    matured = np.array(
        [1 if labels.get((s.tick_index, s.runner_idx), False) else 0
         for s in samples],
        dtype=np.int8,
    )
    runner_idx = np.array([s.runner_idx for s in samples], dtype=np.int16)
    tick_index = np.array([s.tick_index for s in samples], dtype=np.int32)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out, obs=obs, matured=matured,
        runner_idx=runner_idx, tick_index=tick_index,
    )
    dt = time.time() - t0
    rate = float(matured.mean())
    print(f"[{date}] n={len(matured):>6} matured={int(matured.sum()):>6} "
          f"rate={rate:.4f} obs_dim={obs.shape[1]} {dt:.0f}s -> {out.name}",
          flush=True)
    return {"date": date, "n": int(len(matured)), "matured_rate": rate,
            "obs_dim": int(obs.shape[1]), "secs": round(dt, 1), "cached": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default="all",
                    help="train8 | holdout | all | alltrain | <csv of YYYY-MM-DD>")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    days = resolve_days(args.days)
    out_dir = Path(args.out_dir)
    print(f"days={days}", flush=True)
    print(f"out_dir={out_dir}", flush=True)

    cfg = _load_config()
    print("loading predictor bundle...", flush=True)
    bundle = make_bundle()

    rows = []
    for d in days:
        r = build_day(d, cfg, bundle, out_dir, args.force)
        if r is not None:
            rows.append(r)

    n_total = sum(r["n"] for r in rows)
    print(f"\nDONE {len(rows)} days, {n_total} candidates total.", flush=True)


if __name__ == "__main__":
    main()
