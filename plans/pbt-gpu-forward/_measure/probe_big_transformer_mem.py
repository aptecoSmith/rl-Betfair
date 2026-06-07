"""Peak GPU memory for the BIGGEST transformers the widened size genes can
now draw, to validate the --gpu-lane-max-concurrent cap. Measures a batched
PPO-update-style forward+backward (mini_batch=64 over the full ctx buffer),
which is where a transformer's activation memory peaks (O(batch*heads*ctx^2)).

Decision rule (operator, task #6): set the cap so cap * peak(biggest) < 24 GB
with headroom. If 2 * peak > ~22 GB, recommend cap=1 (or gradient checkpoint).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from agents_v2.action_space import DiscreteActionSpace  # noqa: E402
from agents_v2.policy_factory import build_policy  # noqa: E402
from training_v2.cohort.genes import CohortGenes  # noqa: E402

BASE = dict(learning_rate=3e-4, entropy_coeff=0.01, clip_range=0.2,
            gae_lambda=0.95, value_coeff=0.5, mini_batch_size=64)


def g(d, depth, h, ctx, ffn, pos="learned"):
    return CohortGenes(
        architecture="transformer", hidden_size=d, transformer_depth=depth,
        transformer_heads=h, transformer_ctx_ticks=ctx,
        transformer_ffn_mult=ffn, transformer_pos_encoding=pos, **BASE)


def peak_mem(genes, mini_batch=64, obs_dim=512):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sp = DiscreteActionSpace(max_runners=14)
    p = build_policy(genes, obs_dim=obs_dim, action_space=sp).to("cuda")
    hid = tuple(t.to("cuda") for t in p.init_hidden(mini_batch))
    obs = torch.randn(mini_batch, obs_dim, device="cuda")
    out = p(obs, hidden_state=hid)
    # A scalar surrogate of the real update loss — exercises the full backward.
    loss = (out.logits.float().pow(2).mean()
            + out.value_per_runner.float().pow(2).mean())
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 1e9
    del p, out, loss, hid, obs
    return peak


CONFIGS = [
    ("d256/L3/h8/ctx256/ffn2  (validated ~4GB agent)", g(256, 3, 8, 256, 2)),
    ("d512/L6/h16/ctx256/ffn4 (BIGGEST learned)",      g(512, 6, 16, 256, 4)),
    ("d512/L6/h16/ctx256/ffn4 (BIGGEST rope)",  g(512, 6, 16, 256, 4, "rope")),
    ("d512/L4/h8/ctx256/ffn2  (mid-large)",            g(512, 4, 8, 256, 2)),
]

print(f"cuda={torch.cuda.is_available()} "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
print("Peak GPU mem, batched update step (mini_batch=64, full ctx buffer):")
for name, genes in CONFIGS:
    try:
        pk = peak_mem(genes)
        twofit = "2 FIT" if 2 * pk < 22.0 else ("ONLY 1 FITS" if pk < 22 else "OOM-RISK")
        print(f"  {name:46s} peak={pk:5.2f} GB  -> cap2={2*pk:5.2f}GB [{twofit}]",
              flush=True)
    except RuntimeError as e:
        print(f"  {name:46s} OOM/ERROR: {str(e)[:60]}", flush=True)
