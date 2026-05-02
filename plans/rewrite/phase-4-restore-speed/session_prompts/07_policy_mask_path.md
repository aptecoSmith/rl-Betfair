# Session prompt — Phase 4 Session 07: policy forward mask-path cleanup

Use this prompt to open a new session in a fresh context. Self-
contained.

---

## The task

`DiscreteLSTMPolicy._apply_mask` in
`agents_v2/discrete_policy.py` (lines 333–358) does two things
per forward call (i.e. per tick at rollout time) that don't
need to:

```python
# (lines 355-358)
neg_inf = torch.tensor(
    float("-inf"), dtype=logits.dtype, device=logits.device,
)
return torch.where(mask, logits, neg_inf)
```

1. **`torch.tensor(float("-inf"), ...)` per call.** Constructs
   a new 0-d tensor every tick, allocator churn for a value
   that's literally constant across the whole episode (and the
   whole training run).
2. **`torch.where(mask, logits, neg_inf)` instead of
   `logits.masked_fill(~mask, float("-inf"))`.** Both produce
   the same numerical result; `masked_fill` is the more
   idiomatic and marginally faster op (one kernel launch
   instead of two-tensor selection).

Same numbers, faster execution. The expected per-tick win is
small individually (~50–200 µs) but compounds across 12 k
ticks/episode and is pure overhead.

End-of-session bar:

1. **CPU bit-identity preserved on `masked_logits`.** Pre/post
   `_apply_mask` produces byte-identical output for the same
   input on a fixed seed. Strict equality.
2. **Categorical sample / log_prob bit-identical** end-to-end
   on the rollout (the consumer of `masked_logits`). This is
   the strict downstream guard.
3. **`neg_inf` tensor constructed at most once per policy
   instance**, not per call. Verified by a patch + count test.
4. **All pre-existing v2 tests pass on CPU.** Especially
   `tests/test_discrete_policy*.py` if present, and the cohort
   eval-rollout tests.
5. **CUDA self-parity test still passes.**
6. **ms/tick measurement** vs Session 06's baseline.
7. **Verdict** GREEN / PARTIAL / FAIL.

## What you need to read first

1. `plans/rewrite/phase-4-restore-speed/purpose.md` and
   `findings.md` (Sessions 01–06 rows).
2. `agents_v2/discrete_policy.py` lines 276–358 — the full
   `forward` and `_apply_mask`.
3. The PyTorch docs / source for
   `torch.Tensor.masked_fill` vs `torch.where`. Confirm
   bit-equivalence: `where(mask, x, y)` returns `x` where
   `mask` is True and `y` where False; `masked_fill(mask, v)`
   sets `x[mask] = v` in place (or out-of-place via
   `.masked_fill(...)`). To match the existing semantics
   (legal=keep logits, illegal=−inf), use
   `logits.masked_fill(~mask, float("-inf"))`.
4. `tests/test_discrete_policy*.py` (or similar — search for
   `_apply_mask` callers and tests of `DiscreteLSTMPolicy.
   forward`). The new bit-identity test extends or sits next
   to these.

## Implementation sketch

In `agents_v2/discrete_policy.py::DiscreteLSTMPolicy`:

```python
def __init__(self, ...):
    super().__init__(...)
    ...
    # Constant -inf tensor for the mask path, allocated once.
    # Buffer (not parameter) so it moves with .to(device) but
    # doesn't see gradient. dtype=float32 matches logits' dtype
    # under the standard PyTorch defaults; explicit cast on
    # first use guards against fp16/bf16 surprises.
    self.register_buffer(
        "_mask_neg_inf",
        torch.tensor(float("-inf"), dtype=torch.float32),
        persistent=False,
    )

def _apply_mask(self, logits, mask):
    if mask is None:
        return logits
    if mask.shape != logits.shape:
        if mask.dim() == 1 and mask.shape[0] == logits.shape[1]:
            mask = mask.unsqueeze(0).expand_as(logits)
        else:
            raise ValueError(...)
    if mask.dtype != torch.bool:
        mask = mask.bool()
    # masked_fill(~mask, -inf): legal positions keep logits,
    # illegal positions become -inf. Equivalent to the old
    # torch.where(mask, logits, neg_inf).
    return logits.masked_fill(~mask, float("-inf"))
```

Note: `logits.masked_fill(~mask, float("-inf"))` doesn't
actually need the cached buffer — `masked_fill` accepts a
Python float for the value scalar and PyTorch handles the cast
internally. The buffer-pattern in the sketch above is the
defensive shape if `torch.where` is preferred for any reason
(e.g. autograd subtleties). **Recommendation: ship just the
`masked_fill` change; drop the buffer entirely.** Strictly
fewer allocations per tick, simpler code.

If keeping the buffer pattern, register as `persistent=False`
so it doesn't bloat the state_dict and doesn't require a
state-dict migration for existing v2 policies.

## Tests to add

In `tests/test_v2_policy_mask_path.py` (new file):

1. `test_apply_mask_bit_identical_to_pre_session_07` —
   construct a fresh `DiscreteLSTMPolicy` at fixed seed; for a
   sweep of (logits, mask) inputs, assert pre/post `_apply_mask`
   output is byte-equal. Use `torch.equal` (strict).

2. `test_categorical_sample_bit_identical_to_pre_session_07_on_fixed_rng_state`
   — at fixed `torch.manual_seed`, sample `Categorical(logits=
   masked_logits).sample()` for a sweep of inputs; assert pre/
   post equality. This is the consumer-side strictness check.

3. `test_neg_inf_tensor_not_allocated_per_call` — patch
   `torch.tensor` (or whichever constructor was being called)
   on the discrete_policy module and run 1000
   `policy(obs, hidden_state, mask)` calls. Assert call count
   for the patched constructor is ≤ 1 (the one-time init or
   nothing).

4. `test_masked_fill_does_not_mutate_logits_input` — pass a
   logits tensor in, capture a clone before, call _apply_mask,
   assert the input clone equals the input post-call.
   `masked_fill` is the OUT-of-place variant; `masked_fill_`
   is in-place — verify the right one is being used.

5. `test_apply_mask_preserves_dtype_and_device` — fp32 input
   stays fp32 output; CPU input stays CPU output. (Defensive;
   the float-literal `-inf` shouldn't promote, but verify.)

## Hard constraints

1. **Bit-identity end-to-end.** The downstream consumer
   (`Categorical.sample`) is the load-bearing guard, not just
   the local `_apply_mask` output. Any 1-bit difference in
   `masked_logits` propagates through softmax → multinomial →
   sampled action and the whole episode diverges.
2. **Don't touch the four head-projection lines** (logits_head,
   stake_alpha_head, stake_beta_head, value_head). Fusing
   those into a single Linear changes weight-init RNG
   consumption order and breaks the bit-identity test on a
   fresh-init policy. Out of scope for this session.
3. **Don't touch the LSTM forward or input_proj.** Same
   reason: weight-init RNG ordering.
4. **Don't touch the Beta or Categorical wrapper construction
   in this session.** Session 03 owns the global validation
   toggle (which covers Categorical too); this session owns
   only the mask path.
5. **`masked_fill` (out-of-place) not `masked_fill_`
   (in-place).** The input `logits` tensor is shared across
   the masked/unmasked outputs of the policy
   (`DiscretePolicyOutput` exposes both `logits` and
   `masked_logits` separately); in-place would corrupt
   `logits`.

## Deliverables

- `agents_v2/discrete_policy.py` — `_apply_mask` uses
  `masked_fill`; constant `-inf` not allocated per call.
- `tests/test_v2_policy_mask_path.py` (new) with the five
  tests above.
- `findings.md` updated with this session's row.
- Commit: `feat(rewrite): phase-4 S07 (GREEN|PARTIAL) - cache
  -inf in _apply_mask, switch to masked_fill`.

## Estimate

~1 h. If past 1.5 h, stop — likely a downstream consumer
(beyond Categorical) is reading `masked_logits` in a way that
the new shape doesn't satisfy. Document and decide.

## What this session does NOT do

- **Does not fuse the four output heads.** Init-time RNG
  ordering breaks bit-identity; out of scope.
- **Does not touch the env_shim's `compute_extended_obs`** —
  the per-tick LightGBM scorer call is potentially a much
  larger win, but env_shim edits are explicitly out of scope
  per `purpose.md` §"Hard constraints" §9. Filed as a Phase-4b
  candidate; see `purpose.md` §"Phase 4b candidates".
- **Does not touch the LSTM cell.** `nn.LSTM(batch_first=True)`
  vs `batch_first=False` is a cuDNN-layout question that
  becomes interesting at higher batch sizes; at our batch=1,
  ctx=1 it's irrelevant.
