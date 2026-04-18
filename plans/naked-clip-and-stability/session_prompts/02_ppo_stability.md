# Session 02 prompt — PPO stability: KL early-stop + ratio clamp + per-arch LR

## PREREQUISITE — read first

- [`../purpose.md`](../purpose.md) — in particular the §2 PPO
  instability description and the `policy_loss=1.04e17`
  evidence from transformer `0a8cacd3` ep 1.
- [`../hard_constraints.md`](../hard_constraints.md). §9 (KL
  early-stop at epoch granularity, NOT mini-batch), §10
  (ratio clamp `[−20, +20]` before `.exp()`), §11 (per-arch
  initial LR — transformer halved), §12 (existing 5-update
  warmup stays), §20 (tests green on commit), §22 (synthetic
  high-KL test required).
- [`../master_todo.md`](../master_todo.md) — Session 02
  deliverables and exit criteria.
- `plans/policy-startup-stability/progress.md` — the
  advantage-normalisation fix this session builds on.
- `agents/ppo_trainer.py` — the file being edited.
  Specifically the existing warmup
  (`agents/ppo_trainer.py:1114`), advantage normalisation
  (`agents/ppo_trainer.py:1271`), and advantage clamp
  (`agents/ppo_trainer.py:1232`).

## Locate the code

```
grep -n "surrogate\|ratio\|log_ratio\|new_logp\|old_logp\|_ppo_update\|_lr_warmup\|approx_kl\|for epoch in\|for .* in .*range.*epochs" agents/ppo_trainer.py
```

The PPO epoch loop should be obvious from the grep. Confirm
before editing:
1. Where `log_ratio` (or equivalent) is computed — this is
   where the clamp goes, BEFORE the `.exp()` call.
2. Where the epoch loop lives — this is where the KL
   early-stop check fires.
3. Whether the trainer already tracks an approximate KL
   anywhere (some implementations do for logging). If so,
   reuse the computation; don't duplicate.

## What to do

### 1. Ratio clamp

Wherever `ratio = (new_logp - old_logp).exp()` is computed,
replace with:

```python
log_ratio = torch.clamp(new_logp - old_logp, min=-20.0, max=20.0)
ratio = log_ratio.exp()
```

This is a numerical backstop per `hard_constraints.md §10`.
In the common case |log_ratio| ≪ 20, so gradients are
unchanged. It only bites when KL has already run away — at
which point the clamp prevents numerical overflow while the
KL early-stop (below) aborts the rest of the epoch loop.

### 2. KL early-stop

At the END of each PPO epoch (after the inner mini-batch
loop but before incrementing the epoch counter), compute
approximate KL and break if it exceeds the threshold:

```python
with torch.no_grad():
    # Approximate KL across the whole rollout as evaluated
    # by the current (post-this-epoch) policy.
    approx_kl = (old_logp - new_logp_full_rollout).mean().item()

if approx_kl > self.kl_early_stop_threshold:
    self._record_kl_early_stop(approx_kl, epoch)
    break
```

Two plumbing notes:
- `new_logp_full_rollout` needs one forward pass over the
  full rollout under the current policy. If the trainer
  already tracks this for logging, reuse. Otherwise the
  cheap option is to cache `new_logp` from the last
  mini-batch of the epoch and approximate from that — but
  the proper option is a single no-grad forward over the
  whole rollout, which is cheap relative to the epoch of
  backward passes you just ran.
- Default threshold `0.03` — literature standard
  (Andrychowicz et al. 2021, Engstrom et al. 2020). Expose
  as `self.kl_early_stop_threshold` initialised from
  `hp.get("kl_early_stop_threshold", 0.03)` so the GA gene
  system can mutate it later if useful.

### 3. Per-architecture initial LR

The transformer's default learning rate halves. Locate the
transformer's hyperparameter defaults:

```
grep -rn "ppo_transformer_v1\|transformer_v1" agents/
grep -rn "architecture.*default\|default.*hyperparam" agents/
```

Whatever file registers the transformer's defaults (likely
`agents/ppo_transformer_v1.py` or
`agents/architecture_registry.py`), halve the default
`learning_rate`. If the current default is `3e-4`, new value
is `1.5e-4`. If it's something else, halve that.

Add a code comment cross-linking this plan:

```python
# Transformer action heads saturate on the first PPO update
# at the shared default LR — transformer `0a8cacd3` ep-1
# policy_loss=1.04e17 regression confirmed the shared LR is
# too hot for this arch. Halving here gives the warmup +
# KL-early-stop + ratio-clamp defences headroom to catch
# any residual instability.
# See plans/naked-clip-and-stability/purpose.md §2.
```

Do NOT add a per-architecture override mechanism if one
doesn't exist — just change the default. The architecture
registration pattern (one file per arch) already gives
per-arch config.

### 4. Warmup coverage audit

The 5-update linear LR warmup (`agents/ppo_trainer.py:1114`)
applies to `self.optimizer.param_groups`. Confirm it fires
for all three architectures. The path:

```
grep -n "self.optimizer\|self\._base_learning_rate\|param_group" agents/ppo_trainer.py
```

If every architecture creates its optimizer through the same
`PPOTrainer` constructor, the warmup applies uniformly — no
fix needed. If any architecture bypasses the constructor
(constructs its own optimizer elsewhere), fix the bypass so
it goes through the same warmup.

Do NOT extend warmup to 10 updates unless the smoke test
(Session 04) fails with 5.

### 5. Tests

New tests in `tests/test_ppo_trainer.py` (or a new file if
the trainer test is already large):

```python
class TestPPOStability:
    def test_ratio_clamp_prevents_overflow(self):
        """Fabricate a log_ratio of 50 (would overflow
        float32.exp()). Assert the clamped ratio is exp(20)
        ≈ 4.85e8, not inf, and that surrogate loss is
        finite."""

    def test_kl_early_stop_fires_on_high_kl(self):
        """Construct a rollout where the optimal PPO update
        moves policy far enough that approx_kl > 0.03 after
        epoch 1. Assert subsequent epochs are skipped
        (epoch counter records 1 actual pass, not the
        configured N)."""

    def test_kl_early_stop_does_not_fire_on_normal_rollout(self):
        """Typical rollout with small policy update —
        approx_kl ≈ 0.005 — all configured epochs run."""

    def test_transformer_default_lr_halved(self):
        """Transformer architecture registry reports
        learning_rate half of LSTM's."""

    def test_warmup_applied_on_first_update_all_archs(self):
        """Parameterised over [transformer, lstm,
        time_lstm]. First _ppo_update call applies
        warmup_factor of 1/5. Fifth call applies 1.0."""
```

The high-KL fabrication test is the load-bearing one per
`hard_constraints.md §22`. Make it deterministic — seed any
RNG, use a fixed synthetic rollout.

### 6. Regression guard

The existing advantage-normalisation test from
`plans/policy-startup-stability/` must still pass. The
changes in this session don't touch normalisation; the test
exists as a regression guard. Run it explicitly:

```
pytest tests/test_ppo_advantage_normalisation.py -v
```

(If the test file lives somewhere else, find via grep and
run that path.)

### 7. Synthetic large-reward smoke

Add one integration-ish unit test:

```python
def test_large_reward_does_not_explode_policy_loss(self):
    """Synthesise a rollout with rewards in ±£500 range —
    typical scalping magnitude. Run one full PPO update.
    Assert policy_loss < 100."""
```

This is the smallest possible regression net for the
transformer `0a8cacd3` failure mode. Not a replacement for
the smoke-test gate (Session 04), but a cheap unit-level
guard.

### 8. Full suite

```
pytest tests/ -q
```

Must be green.

### 9. Commit

```
fix(agents): PPO KL early-stop, ratio clamp, per-arch LR for transformer

Three layered defences against first-update policy
explosion on fresh agents:

1. log_ratio clamped to [-20, +20] before exp() —
   numerical backstop.
2. KL early-stop at epoch granularity — breaks remaining
   PPO epochs for the current rollout when approx_kl
   exceeds 0.03 (literature standard).
3. Transformer default learning_rate halved — action heads
   saturate on the first update at the shared default LR.

Motivation: naked-clip-and-stability/purpose.md §2.
Transformer `0a8cacd3-3c44-47d1-a1c3-15791862a4e6` ep 1
(2026-04-18) logged policy_loss=1.04e17 despite the
advantage-normalisation fix from policy-startup-stability
(commit 8b8ca67). Normalisation bounds the advantage
magnitude, not the policy-ratio. These three layers close
the remaining gap.

See plans/naked-clip-and-stability/.

Tests: N new in tests/test_ppo_trainer.py
(TestPPOStability). pytest tests/ -q: <delta>.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Cross-session rules

- No new hyperparameters beyond `kl_early_stop_threshold`
  (exposed for GA).
- Existing advantage normalisation and warmup untouched.
- No reward-path changes (that's Session 01).
- No entropy changes (that's Session 03).

## After Session 02

1. Append a `progress.md` entry: commit hash, the three
   layered fixes, test counts, warmup audit conclusion.
2. Hand back for Session 03 (entropy + centering).
