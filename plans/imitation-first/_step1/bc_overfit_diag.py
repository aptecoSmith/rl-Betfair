"""Diagnostic: can the v2 policy (input-norm ON) OVERFIT a fixed batch of
oracle positives to OPEN_BACK-on-the-right-runner?

If train_acc climbs high → the architecture CAN learn obs→which-runner;
the canary's NOOP collapse was the negatives/balance, fix that. If it
stays ~0 → the actor's per-runner routing (pooled lstm_last bottleneck)
can't express the mapping → need aux-head supervision or arch change.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import importlib.util as _u
_spec = _u.spec_from_file_location("_c", str(Path(__file__).with_name("bc_fullnet_canary.py")))
_c = _u.module_from_spec(_spec); _spec.loader.exec_module(_c)

from agents_v2.action_space import ActionType
from agents_v2.discrete_policy import DiscreteLSTMPolicy
from training_v2.arb_oracle import _load_config, load_samples

DATA_DIR = Path("data/processed")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _load_config()
    bundle = _c._bundle()
    env, shim = _c._env("2026-04-06", cfg, bundle)
    obs_dim = int(shim.obs_dim); space = shim.action_space
    runner_dim = int(env.active_runner_dim)
    print(f"obs_dim={obs_dim} action_n={space.n} max_runners={space.max_runners}", flush=True)

    pos = load_samples("2026-04-06", DATA_DIR, strict=False)
    X = np.stack([s.obs for s in pos], axis=0).astype(np.float32)
    mean = X.mean(0); std = X.std(0)

    rng = np.random.default_rng(0)
    idx = rng.choice(len(pos), size=min(2048, len(pos)), replace=False)
    ob = torch.tensor(X[idx], dtype=torch.float32, device=device)
    tg = torch.tensor([space.encode(ActionType.OPEN_BACK, int(pos[i].runner_idx)) for i in idx],
                      dtype=torch.long, device=device)
    print(f"overfit batch={len(idx)} distinct_runner_targets={len(set(tg.tolist()))}", flush=True)

    for tag, train_norm in (("input_norm_ON", True),):
        torch.manual_seed(0)
        p = DiscreteLSTMPolicy(obs_dim, space, hidden_size=256, runner_dim=runner_dim,
                               input_norm=train_norm).to(device)
        if train_norm:
            p.set_input_norm_stats(mean, std)
        opt = torch.optim.Adam(p.parameters(), lr=1e-3)
        p.train()
        print(f"\n[{tag}] overfitting a FIXED batch (lr=1e-3)...", flush=True)
        for step in range(3001):
            out = p(ob)
            loss = F.cross_entropy(out.logits, tg)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 300 == 0:
                acc = float((out.logits.argmax(-1) == tg).float().mean().item())
                # also: fraction predicting NOOP (action 0)
                noop_frac = float((out.logits.argmax(-1) == 0).float().mean().item())
                print(f"  step {step:>4} loss {loss.item():.4f} train_acc {acc:.4f} noop_frac {noop_frac:.3f}", flush=True)


if __name__ == "__main__":
    main()
