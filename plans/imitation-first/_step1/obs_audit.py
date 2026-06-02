"""Step 1b: value-domain audit of the FULL obs (143-d/runner + predictors).

The campaign trained on LEAN obs (23-d/runner); this plan switches to
full obs. Full obs dims are less battle-tested, so before trusting BC we
check for unnormalized / leaky / degenerate / non-finite dims that
shape-domain checks would miss (memory feedback_feature_engineering_
diagnostics: a single ~90σ dim is more likely a bug than a fat tail).

Builds one holdout env at full obs + predictors, walks ~600 pre-race
ticks collecting the raw obs vector each tick, and reports per-dim
min/max/mean/std + the worst-magnitude dims + any non-finite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from training_v2.arb_oracle import _load_config  # noqa: E402

# Reuse the probe's env/bundle builders.
import importlib.util as _u  # noqa: E402
_spec = _u.spec_from_file_location(
    "_probe", str(Path(__file__).with_name("bc_learnability_probe.py")),
)
_probe = _u.module_from_spec(_spec)
_spec.loader.exec_module(_probe)


def main():
    cfg = _load_config()
    print("loading predictor bundle...", flush=True)
    bundle = _probe._make_bundle()
    date = "2026-05-20"
    env, shim = _probe._build_env(date, cfg, bundle)
    obs_dim = int(shim.obs_dim)
    print(f"obs_dim={obs_dim}", flush=True)

    obs0, _ = env.reset()
    obs_list = [np.asarray(obs0, dtype=np.float64)]
    # Walk ~600 steps collecting the env obs (NOT shim-extended — the
    # policy sees shim obs, but the env obs is the bulk; we collect the
    # shim-extended obs to audit exactly what the policy ingests).
    # Use the shim to get the extended obs each step via a NOOP drive.
    from agents_v2.action_space import compute_mask  # noqa: F401
    space = shim.action_space
    obs = obs0
    steps = 0
    done = False
    # Drive shim with NOOP to get shim-extended obs (what the policy sees).
    s_obs, _ = shim.reset()
    obs_list = [np.asarray(s_obs, dtype=np.float64)]
    while not done and steps < 600:
        s_obs, _r, term, trunc, _i = shim.step(0)
        obs_list.append(np.asarray(s_obs, dtype=np.float64))
        done = bool(term or trunc)
        steps += 1

    X = np.stack(obs_list, axis=0)  # (N, obs_dim)
    print(f"collected {X.shape[0]} obs vectors of dim {X.shape[1]}", flush=True)

    finite = np.isfinite(X)
    n_nonfinite = int((~finite).sum())
    print(f"non-finite entries: {n_nonfinite}", flush=True)
    if n_nonfinite:
        bad_dims = np.where((~finite).any(axis=0))[0]
        print(f"  non-finite dims: {bad_dims[:30].tolist()}", flush=True)

    Xf = np.where(finite, X, 0.0)
    dmin = Xf.min(axis=0)
    dmax = Xf.max(axis=0)
    dmean = Xf.mean(axis=0)
    dstd = Xf.std(axis=0)
    absmax = np.abs(Xf).max(axis=0)

    print(f"\nglobal: min={dmin.min():.3f} max={dmax.max():.3f} "
          f"mean|x|={np.abs(Xf).mean():.3f}", flush=True)

    # Worst-magnitude dims (candidate unnormalized).
    order = np.argsort(absmax)[::-1]
    print("\nTop 20 dims by abs-max (candidate unnormalized/outlier):",
          flush=True)
    for i in order[:20]:
        print(f"  dim {int(i):>4}  absmax={absmax[i]:>10.2f} "
              f"min={dmin[i]:>9.2f} max={dmax[i]:>9.2f} "
              f"mean={dmean[i]:>8.3f} std={dstd[i]:>8.3f}", flush=True)

    n_over_50 = int((absmax > 50).sum())
    n_over_20 = int((absmax > 20).sum())
    print(f"\ndims with abs-max > 50: {n_over_50} / {obs_dim}", flush=True)
    print(f"dims with abs-max > 20: {n_over_20} / {obs_dim}", flush=True)
    print(f"dims that are all-zero across the walk: "
          f"{int((absmax == 0).sum())} / {obs_dim}", flush=True)


if __name__ == "__main__":
    main()
