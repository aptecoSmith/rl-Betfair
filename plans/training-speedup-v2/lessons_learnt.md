# training-speedup-v2 — lessons learnt

Log anything that bites — especially gate failures and their root cause.

---

## Step 0 (2026-06-01) — the `--batched` path silently drops FOUR more features

The plan exists because `--batched` silently dropped BC. Profiling the
real config surfaced that the drop is **much wider than BC**. Verified
by code-read + a build probe + the c1 log env-build timing (triple
confirmation, never on inference alone):

1. **predictors (race-outcome + direction)** — `train_cluster_batched`
   never passes `predictor_bundle` to `_build_env_for_day`, so the env
   resolves `use_*_predictor=None → cfg default False`. The CLI flags
   `--use-race-outcome-predictor --use-direction-predictor` were
   accepted and **silently ignored** for batched training. obs stays
   2254-d but the 2226 predictor slots are zero-filled; pwin / direction
   / race-confidence gates are no-ops. **This means c1 AND c2 are
   predictor-less runs** — a correctness finding that affects how their
   "predictor-gated scalping" science is interpreted.
2. **feature_cache (phase-3 F.1)** — not threaded into the batched
   path; each cluster agent re-runs `engineer_day`. ~165 s/cluster-day
   wasted (11× redundant). Pure speedup left on the floor.
3. **input_norm** — hardcoded ON in `train_one_agent` (non-batched),
   absent in `train_cluster_batched`'s policy construction.
4. **bc_pretrain / per_transition_credit** — the originally-known
   drops; the runner at least *warns* on these two. It does NOT warn on
   predictors / feature_cache / input_norm.

**Lesson:** when a path forks (sequential `train_one_agent` vs batched
`train_cluster_batched`), every kwarg the canonical path threads is a
silent-drop candidate in the fork. The fix-shaped takeaway for Step 3A:
the batched path must be brought to **feature parity** with
`train_one_agent` (predictor_bundle, feature_cache, input_norm, BC) or
each omission logged — exactly HC#2. A diff of the two functions'
`_build_env_for_day` / policy-construction call sites is the cheap
detector.

## Step 0 — cProfile would have lied; per-phase wall timers on the real path did not

The Phase-3 profile used cProfile (inflates per-step wall ~50 %, and
misses the batched collector's structure entirely). Step 0 instead
monkeypatched `time.perf_counter` accumulators onto the **real**
`BatchedRolloutCollector` / `BetfairEnv` / shim methods and ran the
actual code. Two things only this approach showed:
- the forward is **kernel-launch-bound, not FLOP-bound** (hidden=128
  reconciled within 6 % of c1's hidden=256);
- `collector_other` (sampling + RNG + copies) is **39 % of rollout** —
  invisible to a function-level cProfile sort because it is spread
  across inline collector code, not one hot function.

## Step 3A — a PyTorch update will NOT unlock vmap-over-LSTM (verified)

Operator asked whether updating torch could unlock `vmap` over `nn.LSTM`
(which would let a batch=N forward stack different-weight agents the easy
way). Verified it will not:
- Empirically: `vmap`+`functional_call` over a raw `nn.LSTM` FAILS on the
  installed torch 2.11.0+cu126 — `Batching rule not implemented for
  aten::lstm.input`.
- Structurally: `nn.LSTM` dispatches to the fused `aten::lstm` op
  (cuDNN / mkldnn / native), a monolithic kernel with no batching rule.
  This is a design property, not a version bug — the matching GRU issue
  (pytorch#134606) has sat open/triaged since 2024 with no fix, and the
  same gap spans LSTM/GRU/RNN. A version bump cannot add a rule that
  doesn't exist.
- Even if a future torch added it, bumping torch in THIS repo is not
  "simple": it's a bit-identical-gated training stack; a torch update
  shifts RNG streams + cuDNN numerics and would invalidate the Step-1
  golden fixtures and risk HC#8 (dynamics change).
- The only real workaround is to NOT use the fused op — reimplement the
  LSTM with primitive matmuls (manual weight-stacking + `bmm`), which I
  proved matches a per-agent `LSTMCell` loop to **1.49e-08**. So manual
  stacking is the path regardless of torch version.

**Lesson:** "just update the library" is rarely simple in a
bit-identical-gated stack — the update itself breaks the gate. Verify the
blocker is structural (it was) before treating a version bump as an option.

## Step 1 — the harness caught a real nondeterminism on its FIRST run

Self-parity (capture twice, same seed + same weights + same env) failed
on the very first run — but ONLY on `pair_id`. Every other quantity
(obs, actions, values, rewards, prices, P&L, counts) matched exactly.
Root cause: `pair_id = uuid.uuid4().hex[:12]` (env/betfair_env.py:3739)
— a random handle, nondeterministic by design. The pairing STRUCTURE
(which bets share a pair, in order) is deterministic; only the string is
not. Fix: the comparator canonicalises pair_ids to first-appearance
group indices and compares the group sequence, never the literal string.

**Lesson:** a golden harness must canonicalise opaque random identifiers
(uuids, object ids, dict-iteration-dependent handles) to structural
equivalence, or self-parity false-alarms. That the harness surfaced this
on run #1 — before any speedup — is exactly the point of building it
first (HC#1). Do NOT "fix" it by dropping pair_id from the compare
entirely; that would blind the gate to a real pairing-structure
regression. Compare the structure.

## Step 1 — capture on CPU; fix the policy weights as a controlled input

Two design choices made GATE (a) a clean signal: (1) capture on CPU (the
env is CPU; the LSTM forward is deterministic on CPU; no cuDNN
nondeterminism), and (2) treat policy weights as an INPUT — fresh-init at
a fixed seed, `input_norm=True` structurally with identity stats, no BC.
The gate tests "fast path reproduces slow path given fixed inputs," so
BC/real-input-norm-stats are irrelevant to it and were correctly left
out (they'd add nondeterminism + cost for zero gate value). BC is a
Step-4 question, not a harness dependency.

## Step 0 — "867 s/agent-train-day" was a mislabelled cluster-day wall

867 s is the per-day wall of an ~11-agent batched **cluster**, written
into every agent's `wall_time_sec` by `train_cluster_batched`
(`# cluster-wide wall; per-agent share would need finer profiling`).
Reading it as a solo-agent cost (≈100 s true marginal) would have sent
the whole effort chasing the wrong 8×. Always check what a "per-agent"
number actually divides by before optimising against it.
