"""Phase 0 memory profile — single-worker RSS/USS breakdown.

`plans/gauntlet-pipeline/` Phase 0. Puts real numbers on the ~10 GB
per-worker working set the multiprocess cohort shows, to decide whether the
gauntlet's per-run footprint is dominated by (a) shared memmap pages that RSS
double-counts across workers (cheap — one physical copy) or (b) genuinely
private per-worker state like the predictor bundle (the thing that scales with
worker count).

The crux metric is **USS vs RSS**:
  * RSS counts every resident page, INCLUDING file-backed memmap pages and
    shared library pages that are physically shared across all N workers.
  * USS (Unique Set Size) counts only pages private to this process — i.e. the
    true marginal cost of adding ONE more worker.
If most of the 10 GB WS is the memmapped `static_obs` cache, RSS will be large
but USS small, and day-growth costs ONE shared copy (the design intent). If the
predictor bundle is large and private, USS will be large and it scales ×workers.

This faithfully replays the worker's own load sequence
(`multiproc_worker._train_agent_worker`): thread-pin → torch import →
predictor bundle from manifests → memmapped `DayStaticObs` per day →
`train_one_agent` on one train + one eval day with the predictors-ON
static_obs cache injected.

Usage (box free):
    python -m tools.profile_worker_memory \
        --cache-dir registry/tt_tock_004/mp_static_obs_cache_full \
        --train-day 2026-04-06 --eval-day 2026-04-07
"""
from __future__ import annotations

# Thread-pin BEFORE torch/numpy import — mirror the worker exactly so the
# measured footprint matches a real worker (no BLAS pool oversubscription).
import os

for _v in ("MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import gc
import threading
import time
import tracemalloc
from pathlib import Path

import psutil

_PROC = psutil.Process()


def _mem() -> tuple[float, float]:
    """(rss_gb, uss_gb). USS may be unavailable on some platforms -> nan."""
    info = _PROC.memory_full_info()
    rss = info.rss / 1e9
    uss = getattr(info, "uss", 0) / 1e9 or float("nan")
    return rss, uss


class _PeakSampler:
    """Background thread sampling peak RSS/USS during a long call."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.peak_rss = 0.0
        self.peak_uss = 0.0
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            rss, uss = _mem()
            self.peak_rss = max(self.peak_rss, rss)
            if uss == uss:
                self.peak_uss = max(self.peak_uss, uss)
            time.sleep(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2.0)


def _stage(label: str, rss: float, uss: float, prev_rss: float,
           prev_uss: float, rows: list) -> tuple[float, float]:
    d_rss = rss - prev_rss
    d_uss = uss - prev_uss
    rows.append((label, rss, uss, d_rss, d_uss))
    print(f"  {label:38} rss={rss:6.2f}GB uss={uss:6.2f}GB  "
          f"(+{d_rss:+5.2f} rss / {d_uss:+5.2f} uss)")
    return rss, uss


def main(argv=None) -> int:
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", type=Path,
                   default=Path("registry/tt_tock_004/mp_static_obs_cache_full"))
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--train-day", default="2026-04-06")
    p.add_argument("--eval-day", default="2026-04-07")
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--predictor-root", type=Path,
                   default=Path("../betfair-predictors/production"))
    p.add_argument("--output", type=Path,
                   default=Path("plans/gauntlet-pipeline/phase0_mem_profile.txt"))
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    rows: list = []
    tracemalloc.start(10)
    rss0, uss0 = _mem()
    print("Phase 0 — single-worker memory profile (thread-pinned, predictors ON)")
    print(f"cache={args.cache_dir}  train={args.train_day}  eval={args.eval_day}")
    print("-" * 78)
    prss, puss = _stage("baseline (interpreter + psutil)", rss0, uss0,
                        rss0, uss0, rows)

    # 1) Heavy imports (torch/numpy + worker module).
    import numpy as np  # noqa: F401
    import torch
    torch.set_num_threads(1)
    from training_v2.cohort.worker import train_one_agent
    from training_v2.cohort.genes import CohortGenes
    from training_v2.cohort.static_obs_cache import DayStaticObs
    rss, uss = _mem()
    prss, puss = _stage("after torch/numpy/worker import", rss, uss,
                        prss, puss, rows)

    # 2) Predictor bundle from manifests (the per-worker private cost).
    from predictors import PredictorBundle
    root = args.predictor_root
    bundle = PredictorBundle.from_manifests(
        champion_manifest=str(root / "race-outcome" / "manifest.json"),
        ranker_manifest=str(root / "race-outcome-ranker" / "manifest.json"),
        direction_manifest=str(root / "direction-predictor" / "manifest.json"),
    )
    gc.collect()
    rss, uss = _mem()
    prss, puss = _stage("after predictor bundle load", rss, uss,
                        prss, puss, rows)
    bundle_rss_gb = rows[-1][3]
    bundle_uss_gb = rows[-1][4]

    # 3) Memmapped DayStaticObs for both days (the SHARED day cache).
    static_obs_cache: dict = {}
    npy_bytes = 0
    for day in (args.train_day, args.eval_day):
        npy = args.cache_dir / f"static_obs_{day}.npy"
        side = args.cache_dir / f"meta_{day}.pkl"
        npy_bytes += npy.stat().st_size
        art = DayStaticObs.load(str(npy), str(side), mmap=True)
        # Touch the array so its pages fault in (worst case for RSS) —
        # this is what makes the shared-vs-private distinction meaningful.
        _ = float(np.asarray(art.static_obs_flat[:1]).sum())
        static_obs_cache[day] = art
    rss, uss = _mem()
    prss, puss = _stage(f"after mmap {len(static_obs_cache)} days "
                        f"({npy_bytes/1e6:.0f}MB on disk)", rss, uss,
                        prss, puss, rows)

    # 4) The real train+eval (one day each) under the static_obs cache.
    genes = CohortGenes(
        learning_rate=3e-4, entropy_coeff=1e-3, clip_range=0.2,
        gae_lambda=0.95, value_coeff=0.5, mini_batch_size=64,
        hidden_size=int(args.hidden_size),
    )
    print("  running train_one_agent (1 train + 1 eval day) ...")
    t0 = time.perf_counter()
    with _PeakSampler() as sampler:
        result = train_one_agent(
            agent_id="phase0_profile",
            genes=genes,
            days_to_train=[args.train_day],
            eval_days=[args.eval_day],
            data_dir=args.data_dir,
            device="cpu",
            seed=0,
            model_store=None,
            predictor_bundle=bundle,
            use_race_outcome_predictor=True,
            use_direction_predictor=True,
            predictor_lean_obs=False,
            static_obs_cache=static_obs_cache,
        )
    wall = time.perf_counter() - t0
    rss, uss = _mem()
    prss, puss = _stage(f"after train_one_agent ({wall:.0f}s)", rss, uss,
                        prss, puss, rows)
    print(f"    peak during train: rss={sampler.peak_rss:.2f}GB "
          f"uss={sampler.peak_uss:.2f}GB")
    print(f"    eval day_pnl={result.eval.day_pnl:.1f} "
          f"bets={result.eval.bet_count} locked={result.eval.locked_pnl:.2f}")

    # Python-level allocation snapshot (won't capture LightGBM/torch C++ or
    # the memmap; useful for the pure-Python overhead).
    snap = tracemalloc.take_snapshot()
    top = snap.statistics("lineno")[:12]
    tracemalloc.stop()

    # ── Report ────────────────────────────────────────────────────────────
    L: list[str] = []
    A = L.append
    A("=" * 78)
    A("PHASE 0 — SINGLE-WORKER MEMORY PROFILE")
    A("=" * 78)
    A(f"cache={args.cache_dir}")
    A(f"train_day={args.train_day} eval_day={args.eval_day} "
      f"hidden_size={args.hidden_size}  predictors ON, full obs (2050-d)")
    A("")
    A(f"{'stage':40} {'rss_gb':>8} {'uss_gb':>8} {'drss':>7} {'duss':>7}")
    A("-" * 78)
    for label, rss_v, uss_v, d_rss, d_uss in rows:
        A(f"{label:40} {rss_v:8.2f} {uss_v:8.2f} {d_rss:+7.2f} {d_uss:+7.2f}")
    A("-" * 78)
    A(f"peak-during-train: rss={sampler.peak_rss:.2f}GB "
      f"uss={sampler.peak_uss:.2f}GB")
    A("")
    A("INTERPRETATION")
    A(f"  predictor bundle private cost (USS delta): {bundle_uss_gb:+.2f} GB "
      f"-> this scales x n_workers")
    A(f"  static_obs memmap ({npy_bytes/1e6:.0f}MB/2 days on disk): RSS includes")
    A("    the faulted file-backed pages, but they are ONE physical copy shared")
    A("    across workers (USS excludes them) -> day-growth = one shared copy.")
    A(f"  final RSS-USS gap = {(prss - puss):.2f} GB (shared: libs + memmap)")
    A("")
    A("PYTHON-LEVEL TOP ALLOCATIONS (tracemalloc; excludes C++/memmap)")
    for st in top:
        fr = st.traceback[0]
        A(f"  {st.size/1e6:7.1f} MB  {fr.filename}:{fr.lineno}")
    A("")
    report = "\n".join(L) + "\n"
    print()
    print(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
