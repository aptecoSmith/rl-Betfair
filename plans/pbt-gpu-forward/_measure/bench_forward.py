"""Step-0 Measurement 2/3 — forward wall + forward-share, by arch/device.

Times steady-state batch=1 single-tick forwards (the rollout shape) for the
real architectures the PBT campaign trained, on CPU (1 + N threads) and CUDA,
and scales to a per-day forward wall (12,368 steps/day). Comparing that to the
measured predictors-ON agent-day (from the register) gives the FORWARD SHARE —
the fraction of the agent-day the GPU lane could ever touch (HC#2: the lane only
moves the forward; predictors/matcher stay on CPU).

CAVEAT: runner_dim=None (flat obs projection) — the proven build path at
arbitrary obs_dim. Real full-obs policies set runner_dim (per-runner input
embedding), a modest extra forward cost, so forward-share here is a slight
UNDER-estimate. Good enough for a gate that hinges on order-of-magnitude.
Frozen direction head omitted (small per-runner head). No-grad, input_norm=True.
"""
import time, json, os, sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import torch
from agents_v2.action_space import DiscreteActionSpace
from agents_v2.policy_factory import build_policy
from training_v2.cohort.genes import CohortGenes

torch.manual_seed(0)
STEPS_PER_DAY = 12368


def genes(arch, hidden, ctx=None, depth=None, heads=None):
    kw = {}
    if arch == "transformer":
        kw = dict(transformer_depth=depth, transformer_heads=heads,
                  transformer_ctx_ticks=ctx)
    return CohortGenes(
        learning_rate=3e-4, entropy_coeff=0.01, clip_range=0.2,
        gae_lambda=0.95, value_coeff=0.5, mini_batch_size=64,
        hidden_size=hidden, architecture=arch, **kw)


def max_runners_from_ckpt():
    for ln in open("registry/pbt_long/scoreboard.jsonl"):
        if not ln.strip():
            continue
        p = json.loads(ln).get("weights_path")
        if p and os.path.exists(p):
            sd = torch.load(p, map_location="cpu")
            sd = sd.get("weights", sd) if isinstance(sd, dict) else sd
            for k in sd:
                if k.endswith("value_head.weight"):
                    return int(sd[k].shape[0])
    return 14


MR = max_runners_from_ckpt()
SPACE = DiscreteActionSpace(max_runners=MR)
print(f"max_runners={MR} action_n={SPACE.n} cuda={torch.cuda.is_available()}")


def bench(arch, hidden, obs_dim, device, threads, ctx=None, depth=None, heads=None):
    torch.set_num_threads(threads)
    g = genes(arch, hidden, ctx, depth, heads)
    p = build_policy(g, obs_dim=obs_dim, action_space=SPACE, input_norm=True)
    p = p.to(device).eval()
    warm = (ctx or 0) + 60
    W = 600
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
    return ms, ms / 1000.0 * STEPS_PER_DAY


# (label, arch, hidden, obs_dim, ctx, depth, heads, measured_agentday_s)
CONFIGS = [
    ("LSTM h128 full",         "lstm", 128, 2254, None, None, None, 210),
    ("LSTM h1024 full",        "lstm", 1024, 2254, None, None, None, 452),
    ("TF d256 L1h8 ctx64 full", "transformer", 256, 2254, 64, 1, 8, 482),
    ("TF d128 L2h4 ctx64 full", "transformer", 128, 2254, 64, 2, 4, 412),
    ("TF d64 L2h8 ctx32 full",  "transformer", 64, 2254, 32, 2, 8, 247),
]
DEVICES = [("cpu", 1), ("cpu", 6), ("cuda", 1)]

print(f"\n{'config':26s} {'dev':8s} {'ms/tick':>8s} {'fwd/day':>8s} "
      f"{'agentday':>8s} {'fwd_share':>9s}")
for (lbl, arch, hid, od, ctx, dep, he, ad) in CONFIGS:
    for dev, th in DEVICES:
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        try:
            ms, pd = bench(arch, hid, od, dev, th, ctx, dep, he)
            print(f"{lbl:26s} {dev + '/' + str(th):8s} {ms:8.2f} {pd:8.0f} "
                  f"{ad:8d} {100 * pd / ad:8.1f}%")
        except Exception as e:
            print(f"{lbl:26s} {dev}/{th:<6} ERR {type(e).__name__}: {e}")
    sys.stdout.flush()
