"""pwin-as-direction probe (operator pivot 2026-05-31).

NOT value betting (hold-to-settlement = directional gambling, rejected).
The hypothesis: the race-outcome predictor's mispricing (champion_p_win vs
market implied_prob) is a PRICE-DIRECTION signal — if the market corrects
toward the predictor, the price moves predictably, and a predictable move is
what makes a SCALP mature. We stay pure scalping (lock the spread, exit
before the off); we just aim at the mispricings the market is likely to
correct.

Mechanic (back-first scalps — what the oracle scans):
  under-priced (pwin > implied → odds too generous) → market corrects by the
  price SHORTENING → a back-first scalp (back high, passive lay lower) fills.
So back-first maturation should RISE with mispricing = (pwin − implied).

Decisive question: does mispricing isolate a HIGH-maturation subset (≫ the
~12% base, ideally >40-50% so the locked edge clears the force-close toll)
that the pooled mature_prob head (already sees pwin; maxed ~0.745 AUC / ~13%
realised) isn't capturing? Reuses the held-out maturation dataset — no
re-scan. If yes → condition scalp opens on mispricing. If ~flat → the market
isn't correcting in our pre-off window and we're truly at the wall.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bc-getting-it-right" / "_scripts"))

from env.betfair_env import RUNNER_KEYS, RUNNER_DIM, MARKET_DIM, VELOCITY_DIM  # noqa: E402

CACHE = Path("plans/bc-getting-it-right/_cache")
HOLDOUT = ["2026-05-20", "2026-05-21", "2026-05-22", "2026-05-25",
           "2026-05-27", "2026-05-28", "2026-05-29"]
I_PWIN = RUNNER_KEYS.index("champion_p_win")
I_IMPL = RUNNER_KEYS.index("implied_prob")
I_LTP = RUNNER_KEYS.index("ltp")
OFF = MARKET_DIM + VELOCITY_DIM


def _load(days):
    obs_l, mat_l, ri_l = [], [], []
    for d in days:
        z = np.load(CACHE / f"{d}.npz")
        obs_l.append(z["obs"]); mat_l.append(z["matured"].astype(np.int32))
        ri_l.append(z["runner_idx"].astype(np.int64))
    obs = np.concatenate(obs_l); mat = np.concatenate(mat_l); ri = np.concatenate(ri_l)
    base = OFF + ri * RUNNER_DIM
    rows = np.arange(len(ri))
    pwin = obs[rows, base + I_PWIN]
    impl = obs[rows, base + I_IMPL]
    ltp = obs[rows, base + I_LTP]
    return pwin, impl, ltp, mat


def _bucketed(signal, matured, name, n_bins=10):
    print(f"\n=== maturation rate by {name} decile (holdout) ===", flush=True)
    order = np.argsort(signal)
    n = len(signal)
    base = matured.mean()
    print(f"  base maturation rate = {base:.3f}  (n={n})", flush=True)
    print(f"  {'decile':>6} {'signal range':>22} {'n':>7} {'mat%':>7} {'lift':>6}",
          flush=True)
    for b in range(n_bins):
        lo_i = b * n // n_bins
        hi_i = (b + 1) * n // n_bins
        idx = order[lo_i:hi_i]
        s = signal[idx]; m = matured[idx]
        print(f"  {b:>6} [{s.min():>8.4f},{s.max():>8.4f}] {len(idx):>7} "
              f"{100*m.mean():>6.1f} {m.mean()/max(base,1e-9):>6.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="plans/bc-to-ppo/_scripts/pwin_direction_probe.json")
    args = ap.parse_args()
    from sklearn.metrics import roc_auc_score

    pwin, impl, ltp, matured = _load(HOLDOUT)
    print(f"holdout candidates n={len(matured)} matured_rate={matured.mean():.3f}", flush=True)
    print(f"pwin mean={pwin.mean():.3f} implied mean={impl.mean():.3f} "
          f"ltp mean={ltp.mean():.2f}", flush=True)

    # Candidate signals, all "higher = more under-priced = price should shorten".
    misp_abs = pwin - impl                       # absolute prob divergence
    misp_rel = (pwin - impl) / np.clip(impl, 1e-4, None)  # relative divergence
    value_edge = pwin * ltp - 1.0                # EV of backing £1 (>0 = +EV back)

    signals = {
        "mispricing_abs (pwin-implied)": misp_abs,
        "mispricing_rel ((pwin-implied)/implied)": misp_rel,
        "value_edge (pwin*ltp-1)": value_edge,
    }

    result = {"n": int(len(matured)), "base_rate": round(float(matured.mean()), 4),
              "signals": {}}
    print("\n==== STANDALONE AUC of each signal for predicting maturation ====", flush=True)
    print(f"  (reference: full-feature LightGBM 0.745 ; chance 0.5)", flush=True)
    for name, sig in signals.items():
        finite = np.isfinite(sig)
        auc = float(roc_auc_score(matured[finite], sig[finite]))
        # top-decile maturation (most under-priced)
        order = np.argsort(sig[finite])[::-1]
        top = order[: max(1, len(order) // 10)]
        top_mat = float(matured[finite][top].mean())
        result["signals"][name] = {
            "auc": round(auc, 4),
            "top_decile_mat": round(top_mat, 4),
            "top_decile_lift": round(top_mat / max(matured.mean(), 1e-9), 3),
        }
        print(f"  {name:42s} AUC={auc:.4f}  top10%mat={top_mat:.3f} "
              f"(lift {top_mat/max(matured.mean(),1e-9):.2f}x)", flush=True)

    # Decile tables for the two best framings.
    _bucketed(misp_abs, matured, "mispricing_abs (pwin-implied)")
    _bucketed(value_edge, matured, "value_edge (pwin*ltp-1)")

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
