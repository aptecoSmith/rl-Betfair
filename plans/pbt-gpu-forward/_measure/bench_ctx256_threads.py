"""ctx256 transformer forward @ 1 vs 4 threads (the gap bench_forward.py left).

The campaign capped transformers at ctx<=64; the operator's question is whether
a *big* ctx256 transformer's forward is large enough that threading it would
help. Times the steady-state batch=1 single-tick forward (the rollout shape),
full obs (2254-d), at cpu/1 and cpu/4, and scales to a per-day forward wall
(12,368 steps). Forward SHARE is estimated against the predictor/env/update
floor measured at ctx64 (agent-day 482s - forward 19s = 463s), which is an
UPPER bound on the share (the ctx256 PPO update is bigger -> real floor higher
-> real share lower). Decision rule: if even this upper bound is small, or the
1->4 thread speedup is poor, threading transformers isn't worth it.
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import torch  # noqa: E402

from agents_v2.action_space import DiscreteActionSpace  # noqa: E402
from agents_v2.policy_factory import build_policy  # noqa: E402
from training_v2.cohort.genes import CohortGenes  # noqa: E402

torch.manual_seed(0)
STEPS_PER_DAY = 12368
CTX64_FLOOR_S = 463.0  # 482 agent-day - 19 forward (ctx64 d256), arch-indep floor
SPACE = DiscreteActionSpace(max_runners=14)


def genes_tf(d, ctx, depth, heads):
    return CohortGenes(
        learning_rate=3e-4, entropy_coeff=0.01, clip_range=0.2,
        gae_lambda=0.95, value_coeff=0.5, mini_batch_size=64,
        hidden_size=d, architecture="transformer",
        transformer_depth=depth, transformer_heads=heads,
        transformer_ctx_ticks=ctx)


def bench(d, ctx, depth, heads, threads, device="cpu", obs_dim=2254, W=300):
    torch.set_num_threads(threads)
    g = genes_tf(d, ctx, depth, heads)
    p = build_policy(g, obs_dim=obs_dim, action_space=SPACE,
                     input_norm=True).to(device).eval()
    warm = ctx + 60
    obs = torch.randn(warm + W, 1, obs_dim, device=device)
    h = p.init_hidden(1)
    if isinstance(h, tuple):
        h = tuple(x.to(device) if torch.is_tensor(x) else x for x in h)
    with torch.no_grad():
        for t in range(warm):
            h = p(obs[t], hidden_state=h).new_hidden_state
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for t in range(warm, warm + W):
            h = p(obs[t], hidden_state=h).new_hidden_state
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    ms = dt / W * 1000.0
    fwd_day = ms / 1000.0 * STEPS_PER_DAY
    return ms, fwd_day


print(f"threads_avail={torch.get_num_threads()} (will override per run)")
print(f"\n{'config':28s} {'thr':>4s} {'ms/tick':>9s} {'fwd/day':>9s} "
      f"{'~share(UB)':>11s}")
# ctx64 d256 as the reference point (matches bench_forward.py's 1.51ms/3.9%),
# then the ctx256 d256 case at 1 and 4 threads.
RUNS = [
    ("TF d256 ctx256 depth3 h8 cpu", 256, 256, 3, 8, 1, "cpu"),
    ("TF d256 ctx256 depth3 h8 cpu", 256, 256, 3, 8, 4, "cpu"),
    ("TF d256 ctx256 depth3 h8 CUDA", 256, 256, 3, 8, 1, "cuda"),
]
for lbl, d, ctx, dep, he, th, dev in RUNS:
    if dev == "cuda" and not torch.cuda.is_available():
        continue
    ms, fwd = bench(d, ctx, dep, he, th, device=dev)
    share_ub = 100.0 * fwd / (CTX64_FLOOR_S + fwd)
    print(f"{lbl:32s} {th:>4d} {ms:9.2f} {fwd:8.0f}s {share_ub:10.1f}%")
    sys.stdout.flush()
