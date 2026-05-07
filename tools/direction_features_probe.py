"""Feature-augmentation probe — does longer-window pressure +
TradedVolumeLadder unlock direction prediction?

Phase-13 follow-up to ``direction_head_supervised_probe.py``. The
prior probe established that the existing obs vector carries weak
directional alpha (high-confidence-positive bin only ~2× baseline at
horizon=60). This probe asks: do features that are NOT in the obs
vector — longer-window velocities and per-price traded volume ladder
summaries — carry stronger directional signal?

Setup. For each (pre-race tick, priceable runner) on a single day:

* **Base features** (existing): the per-runner slice of the env's
  ``RUNNER_KEYS`` (115 dims). Captures the obs the policy currently
  has.

* **Augmented features** (new, 8 dims):
  - ``ltp_velocity_30``, ``ltp_velocity_60`` — per-tick lookback of
    30 / 60 (≈ 8 / 16 minutes wall on this data's tick spacing).
  - ``vol_delta_30``, ``vol_delta_60`` — runner traded-volume delta
    over the same 30 / 60 tick windows.
  - ``vol_above_ltp_frac`` — fraction of the runner's
    ``TradedVolumeLadder`` size at prices > LTP (drift money).
  - ``vol_below_ltp_frac`` — same at prices < LTP (steam money).
  - ``vol_ladder_imbalance`` — ``(above - below) / (above + below)``.
  - ``vol_weighted_price_dist`` — size-weighted mean of ladder
    prices minus current LTP, in tick units.

Train two MLPs with bit-identical seed + arch:

* "base"      — `Linear(115 → 64) → ReLU → Linear(64 → 2)`
* "augmented" — `Linear(123 → 64) → ReLU → Linear(64 → 2)`

The architecture is a thin MLP because the previous probe already
showed a single Linear can extract some calibration; the question is
whether the EXTRA columns push that calibration further. Both heads
are smaller than the policy's path so both are fair-weight tests of
the input information.

Usage::

    python -m tools.direction_features_probe \\
        --date 2026-05-03 --horizon-ticks 60 --threshold-ticks 5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data.episode_builder import load_day
from data.feature_engineer import engineer_day
from env.betfair_env import RUNNER_KEYS
from env.tick_ladder import ticks_between
from training_v2.arb_oracle import _load_config
from training_v2.direction_label_scan import load_labels


def _runner_slot_map(race) -> dict:
    """Sorted-sid → slot map (mirrors BetfairEnv's per-race map)."""
    sids = set()
    for tick in race.ticks:
        for r in tick.runners:
            sids.add(r.selection_id)
    return {sid: i for i, sid in enumerate(sorted(sids))}


def _load_traded_volume_ladders(date: str, data_dir: Path) -> dict:
    """Parse ``snap_json`` rows in the parquet to extract per-runner
    TradedVolumeLadder entries indexed by ``(market_id,
    sequence_number, selection_id) -> [(price, size), ...]``.

    The episode_builder's ``RunnerSnap`` does NOT expose this field;
    we go to the parquet directly.
    """
    import json
    import pandas as pd
    parquet = Path(data_dir) / f"{date}.parquet"
    print(f"  parsing TradedVolumeLadder from {parquet} ...")
    t0 = time.monotonic()
    df = pd.read_parquet(parquet, columns=[
        "market_id", "sequence_number", "snap_json",
    ])
    out: dict = {}
    n_with_ladder = 0
    n_total = 0
    for row in df.itertuples(index=False):
        try:
            snap = json.loads(row.snap_json)
        except Exception:
            continue
        runners = snap.get("MarketRunners", [])
        for r in runners:
            sid = r.get("RunnerId", {}).get("SelectionId")
            prices = r.get("Prices", {})
            tvl = prices.get("TradedVolumeLadder", []) or []
            n_total += 1
            if not tvl:
                continue
            n_with_ladder += 1
            # Tuple of (price, size) for compactness.
            ladder = [
                (float(x.get("Price") or 0.0), float(x.get("Size") or 0.0))
                for x in tvl
            ]
            out[(row.market_id, int(row.sequence_number), int(sid))] = ladder
    print(f"    parsed {n_total} runner-snaps; "
          f"{n_with_ladder} carry a ladder; "
          f"wall = {time.monotonic() - t0:.1f}s")
    return out


def build_features(date: str, data_dir: Path):
    """Walk a day; return per-(global_pre_race_tick, runner_slot) feature
    rows for every priceable runner.

    Returns
    -------
    base_feat : (N, 115) float32
    aug_feat  : (N, 8)   float32
    tick_idx  : (N,) int32 — global pre-race tick (matches scan_day)
    runner_idx: (N,) int32
    sid_arr   : (N,) int64
    runner_to_max_slots : int — max slots seen across day's races
    """
    day = load_day(date, data_dir)
    if not day.races:
        raise RuntimeError(f"{date}: no races")

    tvl_map = _load_traded_volume_ladders(date, data_dir)

    print(f"  feature engineering full day...")
    t0 = time.monotonic()
    eng = engineer_day(day)
    print(f"    engineer_day wall = {time.monotonic() - t0:.1f}s")

    base_rows = []
    aug_rows = []
    tick_rows = []
    runner_rows = []
    sid_rows = []
    max_slot = 0

    global_pre = 0
    for race_idx, race in enumerate(day.races):
        slot_map = _runner_slot_map(race)
        max_slot = max(max_slot, max(slot_map.values()) + 1 if slot_map else 0)

        # Per-runner ltp + traded_volume time series for the longer-window
        # velocity / volume features. Indexed by raw race-tick index;
        # NaN where the runner is non-active or unpriceable.
        n_ticks = len(race.ticks)
        ltp_series = {sid: np.full(n_ticks, np.nan, dtype=np.float64)
                      for sid in slot_map}
        vol_series = {sid: np.full(n_ticks, np.nan, dtype=np.float64)
                      for sid in slot_map}
        ladder_by_tick = {}
        for t, tick in enumerate(race.ticks):
            for r in tick.runners:
                if r.selection_id not in ltp_series:
                    continue
                if r.status != "ACTIVE":
                    continue
                ltp = r.last_traded_price
                if ltp and ltp > 1.0:
                    ltp_series[r.selection_id][t] = float(ltp)
                vol_series[r.selection_id][t] = float(r.total_matched)

        # Iterate pre-race ticks. ``eng[race_idx]`` carries the
        # engineered feature dicts — same shape the env reads.
        race_features = eng[race_idx]
        for tick_idx_in_race, tick in enumerate(race.ticks):
            if tick.in_play:
                continue
            tick_feat = race_features[tick_idx_in_race]
            runner_feats = tick_feat["runners"]

            for r in tick.runners:
                sid = r.selection_id
                slot = slot_map.get(sid)
                if slot is None:
                    continue
                if r.status != "ACTIVE":
                    continue
                ltp = r.last_traded_price
                if not ltp or ltp <= 1.0:
                    continue
                if sid not in runner_feats:
                    continue

                # Base: 115 RUNNER_KEYS values. Missing keys land at 0.0
                # — same fallback the env applies in _features_to_array.
                base = np.zeros(len(RUNNER_KEYS), dtype=np.float32)
                rf = runner_feats[sid]
                for i, k in enumerate(RUNNER_KEYS):
                    v = rf.get(k, 0.0)
                    if v is None:
                        v = 0.0
                    base[i] = float(v)

                # Augmented: 8 new features.
                # Velocity over 30/60 ticks. ``ltp_series`` carries the
                # full per-runner sequence; we look back tick_idx_in_race
                # - lag and skip if NaN.
                aug = np.zeros(8, dtype=np.float32)
                for j, lag in enumerate((30, 60)):
                    src_t = tick_idx_in_race - lag
                    if src_t >= 0:
                        prev = ltp_series[sid][src_t]
                        if not np.isnan(prev) and prev > 0:
                            aug[j] = (ltp - prev) / prev
                # Vol delta over 30/60.
                for j, lag in enumerate((30, 60)):
                    src_t = tick_idx_in_race - lag
                    cur = float(r.total_matched)
                    if src_t >= 0:
                        prev = vol_series[sid][src_t]
                        if not np.isnan(prev):
                            aug[2 + j] = max(0.0, cur - prev)

                # TradedVolumeLadder — parsed from raw snap_json above.
                tvl = tvl_map.get(
                    (race.market_id, int(tick.sequence_number), int(sid))
                )
                if tvl:
                    above = sum(
                        size for price, size in tvl if price > ltp
                    )
                    below = sum(
                        size for price, size in tvl if price < ltp
                    )
                    total = above + below
                    if total > 0:
                        aug[4] = above / total
                        aug[5] = below / total
                        aug[6] = (above - below) / total
                        wap = sum(
                            size * price for price, size in tvl
                        ) / total
                        # Signed tick distance from current LTP.
                        if wap > ltp:
                            aug[7] = float(ticks_between(ltp, wap))
                        elif wap < ltp:
                            aug[7] = -float(ticks_between(ltp, wap))

                base_rows.append(base)
                aug_rows.append(aug)
                tick_rows.append(global_pre)
                runner_rows.append(slot)
                sid_rows.append(sid)

            global_pre += 1

    base_arr = np.stack(base_rows) if base_rows else np.zeros((0, len(RUNNER_KEYS)), dtype=np.float32)
    aug_arr = np.stack(aug_rows) if aug_rows else np.zeros((0, 8), dtype=np.float32)
    tick_arr = np.array(tick_rows, dtype=np.int32)
    runner_arr = np.array(runner_rows, dtype=np.int32)
    sid_arr = np.array(sid_rows, dtype=np.int64)
    return base_arr, aug_arr, tick_arr, runner_arr, sid_arr


def _per_row_label_lookup(
    labels, tick_arr, runner_arr,
):
    """Build a `(label_back, label_lay, mask)` triple aligned with the
    per-row arrays from build_features. Rows without a matching cache
    entry get mask=False.
    """
    by_key = {(r.tick_index, r.runner_idx): (r.label_back, r.label_lay)
              for r in labels}
    n = tick_arr.shape[0]
    label_back = np.zeros(n, dtype=np.float32)
    label_lay = np.zeros(n, dtype=np.float32)
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        v = by_key.get((int(tick_arr[i]), int(runner_arr[i])))
        if v is not None:
            label_back[i], label_lay[i] = v
            mask[i] = True
    return label_back, label_lay, mask


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),  # back_logit, lay_logit
        )
    def forward(self, x):
        return self.net(x)


def train_one(
    *,
    feats: np.ndarray,
    label_back: np.ndarray,
    label_lay: np.ndarray,
    mask: np.ndarray,
    name: str,
    device: str,
    n_steps: int,
    batch_size: int,
    log_every: int,
    seed: int,
):
    print(f"\n--- {name} (in_dim={feats.shape[1]}) ---")
    # Filter to masked rows.
    sel = np.where(mask)[0]
    feats_s = feats[sel]
    lback_s = label_back[sel]
    llay_s = label_lay[sel]
    n = sel.shape[0]
    pos_back = float(lback_s.mean())
    pos_lay = float(llay_s.mean())
    print(f"  N={n}  pos_back={pos_back:.4f}  pos_lay={pos_lay:.4f}")

    # NaN guard — feature engineer emits NaN for missing past-race
    # values etc; replace with 0 before standardisation.
    feats_s = np.nan_to_num(feats_s, nan=0.0, posinf=0.0, neginf=0.0)
    # Per-feature standardisation. Tiny MLP needs scaled inputs to
    # avoid the first layer's gradient being dominated by raw price /
    # volume magnitudes. Compute on the training rows only — same
    # rows we'll back-test on so this is in-sample. For a probe that's
    # fine (we only care about whether augmented columns help).
    mu = feats_s.mean(axis=0, keepdims=True)
    sigma = feats_s.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)  # constant cols → /1
    feats_z = (feats_s - mu) / sigma
    feats_z = np.nan_to_num(feats_z, nan=0.0, posinf=0.0, neginf=0.0)

    torch.manual_seed(int(seed))
    model = _MLP(in_dim=feats_z.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    pw_back = torch.tensor(
        (1 - pos_back) / max(pos_back, 1e-6),
        dtype=torch.float32, device=device,
    )
    pw_lay = torch.tensor(
        (1 - pos_lay) / max(pos_lay, 1e-6),
        dtype=torch.float32, device=device,
    )

    feats_t = torch.from_numpy(feats_z.astype(np.float32)).to(device)
    lback_t = torch.from_numpy(lback_s).to(device)
    llay_t = torch.from_numpy(llay_s).to(device)
    rng = np.random.default_rng(int(seed))

    for step in range(n_steps):
        idx = rng.integers(0, n, size=batch_size)
        idx_t = torch.from_numpy(idx).to(device)
        x = feats_t.index_select(0, idx_t)
        yb = lback_t.index_select(0, idx_t)
        yl = llay_t.index_select(0, idx_t)
        out = model(x)
        bback = nn.functional.binary_cross_entropy_with_logits(
            out[:, 0], yb, pos_weight=pw_back,
        )
        blay = nn.functional.binary_cross_entropy_with_logits(
            out[:, 1], yl, pos_weight=pw_lay,
        )
        loss = bback + blay
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0 or step == n_steps - 1:
            print(f"  step={step:4d}  loss={loss.item():.4f}  "
                  f"bce_back={bback.item():.4f}  bce_lay={blay.item():.4f}")

    # Full-day in-sample eval.
    with torch.no_grad():
        out = model(feats_t)
        p_back = torch.sigmoid(out[:, 0])
        p_lay = torch.sigmoid(out[:, 1])
        # Calibration bins.
        def cal(p, y, n_bins=5):
            sidx = torch.argsort(p)
            chunks = torch.chunk(sidx, n_bins)
            return [(float(p[c].mean().item()), float(y[c].mean().item())) for c in chunks]
        cal_back = cal(p_back, lback_t)
        cal_lay = cal(p_lay, llay_t)
        # AUC-like signal: ratio of top-quintile realised vs bottom-quintile realised.
        top_back = cal_back[-1][1]
        bot_back = cal_back[0][1]
        top_lay = cal_lay[-1][1]
        bot_lay = cal_lay[0][1]
        # Mean BCE across full set.
        full_back = float(nn.functional.binary_cross_entropy_with_logits(
            out[:, 0], lback_t, pos_weight=pw_back,
        ).item())
        full_lay = float(nn.functional.binary_cross_entropy_with_logits(
            out[:, 1], llay_t, pos_weight=pw_lay,
        ).item())

    print(f"\n  Calibration (predicted P → realised P, 5 bins):")
    for i in range(len(cal_back)):
        pb, rb = cal_back[i]
        pl, rl = cal_lay[i]
        print(f"    bin {i}: back P_pred={pb:.3f} P_real={rb:.3f}   "
              f"lay P_pred={pl:.3f} P_real={rl:.3f}")
    lift_back = top_back / max(bot_back, 1e-6)
    lift_lay = top_lay / max(bot_lay, 1e-6)
    print(f"  Top-vs-bottom-quintile realised-rate ratio:")
    print(f"    back: {top_back:.4f} / {bot_back:.4f} = {lift_back:.2f}×  "
          f"baseline = {pos_back:.4f}")
    print(f"    lay : {top_lay:.4f} / {bot_lay:.4f} = {lift_lay:.2f}×  "
          f"baseline = {pos_lay:.4f}")
    return {
        "name": name,
        "in_dim": feats.shape[1],
        "n_rows": n,
        "full_bce_back": full_back,
        "full_bce_lay": full_lay,
        "cal_back": cal_back,
        "cal_lay": cal_lay,
        "top_back": top_back, "bot_back": bot_back, "lift_back": lift_back,
        "top_lay": top_lay, "bot_lay": bot_lay, "lift_lay": lift_lay,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-05-03")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--horizon-ticks", type=int, default=60)
    ap.add_argument("--threshold-ticks", type=int, default=5)
    ap.add_argument("--fc-secs", type=float, default=60.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    config = _load_config()
    data_dir = Path(args.data_dir)

    t0 = time.monotonic()
    print(f"=== feature build for {args.date} ===")
    base_arr, aug_arr, tick_arr, runner_arr, sid_arr = build_features(
        args.date, data_dir,
    )
    print(f"  rows = {base_arr.shape[0]}  base_dim = {base_arr.shape[1]}  "
          f"aug_dim = {aug_arr.shape[1]}  wall = {time.monotonic() - t0:.1f}s")
    # Smoke-check augmented features have non-zero variance.
    aug_std = aug_arr.std(axis=0)
    print(f"  augmented feature stds: {aug_std}")
    aug_nonzero = (aug_std > 1e-6).sum()
    print(f"  augmented features with non-zero variance: {aug_nonzero}/{aug_arr.shape[1]}")

    print(f"\n=== loading direction labels ===")
    labels = load_labels(
        args.date, data_dir,
        direction_horizon_ticks=args.horizon_ticks,
        direction_threshold_ticks=args.threshold_ticks,
        force_close_before_off_seconds=args.fc_secs,
        strict=True,
    )
    print(f"  {len(labels)} label rows")
    label_back, label_lay, mask = _per_row_label_lookup(
        labels, tick_arr, runner_arr,
    )
    print(f"  matched rows: {int(mask.sum())} / {mask.shape[0]}")

    # Two head-to-head training configs with identical seed.
    out_base = train_one(
        feats=base_arr,
        label_back=label_back, label_lay=label_lay, mask=mask,
        name="base (115)", device=args.device,
        n_steps=args.steps, batch_size=args.batch_size,
        log_every=args.log_every, seed=args.seed,
    )
    out_aug = train_one(
        feats=np.concatenate([base_arr, aug_arr], axis=1),
        label_back=label_back, label_lay=label_lay, mask=mask,
        name="base+augmented (115+8)", device=args.device,
        n_steps=args.steps, batch_size=args.batch_size,
        log_every=args.log_every, seed=args.seed,
    )

    print("\n=== summary ===")
    print(f"{'config':<25} {'BCE_back':>10} {'BCE_lay':>10} "
          f"{'lift_back':>10} {'lift_lay':>10}")
    for r in (out_base, out_aug):
        print(f"{r['name']:<25} {r['full_bce_back']:>10.4f} "
              f"{r['full_bce_lay']:>10.4f} {r['lift_back']:>10.2f}× "
              f"{r['lift_lay']:>9.2f}×")

    print(f"\nLift = top-quintile realised P / bottom-quintile realised P.")
    print(f"Higher = head's confidence ranking is sharper.")
    print(f"If base+augmented lift > base lift by ≥ 30 %, augmented features carry")
    print(f"meaningful directional alpha and a feature-extension plan is justified.")


if __name__ == "__main__":
    main()
