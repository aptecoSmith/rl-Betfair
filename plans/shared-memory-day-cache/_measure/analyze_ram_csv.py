"""Summarise a ram_watch.csv into the plateau + spike figures (Step 3).

Reads the CSV produced by ram_watch.ps1 and reports, for the run:
  - peak commit (the true OOM driver) and min Available (the kill metric)
  - the per-worker private footprint at steady state
  - the training-start spike (max during the first ~minutes)

Usage: python analyze_ram_csv.py ram_watch.csv [label]
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else path.stem
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("timestamp") or not ln.strip():
            continue
        if ln.startswith("LOWRAM_KILL"):
            print(f"  *** LOWRAM_KILL fired: {ln} ***")
            continue
        parts = ln.split(",")
        if len(parts) < 6:
            continue
        ts, avail_mb, commit_gb, py_n, ws_gb, priv_gb = parts[:6]
        try:
            rows.append((
                ts, float(avail_mb), float(commit_gb), int(float(py_n)),
                float(ws_gb), float(priv_gb),
            ))
        except ValueError:
            continue
    if not rows:
        print(f"{label}: no data rows")
        return

    avail = [r[1] for r in rows]
    commit = [r[2] for r in rows]
    pyn = [r[3] for r in rows]
    ws = [r[4] for r in rows]
    priv = [r[5] for r in rows]
    total_phys_gb = None  # not in csv; report relative

    min_avail = min(avail)
    max_commit = max(commit)
    max_pyn = max(pyn)
    max_ws = max(ws)
    max_priv = max(priv)
    # Steady state = last third of samples (workers fully loaded).
    tail = rows[len(rows) * 2 // 3:]
    if tail:
        steady_priv = sum(r[5] for r in tail) / len(tail)
        steady_pyn = max(r[3] for r in tail)
        per_worker = steady_priv / steady_pyn if steady_pyn else 0.0
    else:
        steady_priv = per_worker = 0.0
        steady_pyn = 0

    print(f"=== {label} ({len(rows)} samples) ===")
    print(f"  min Available (kill metric) : {min_avail/1024:6.1f} GB "
          f"({min_avail:.0f} MB)")
    print(f"  peak Commit (OOM driver)    : {max_commit:6.1f} GB")
    print(f"  peak python count           : {max_pyn}")
    print(f"  peak python working-set     : {max_ws:6.1f} GB")
    print(f"  peak python PRIVATE         : {max_priv:6.1f} GB")
    print(f"  steady python PRIVATE       : {steady_priv:6.1f} GB "
          f"over {steady_pyn} procs")
    print(f"  → per-worker PRIVATE (steady): {per_worker:5.1f} GB")
    print(f"  (Private excludes file-backed memmap pages — the shared "
          f"static_obs is NOT counted here, which is the point.)")


if __name__ == "__main__":
    main()
